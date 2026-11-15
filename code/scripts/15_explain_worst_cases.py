#!/usr/bin/env python3
"""Explain frozen gate/circuit errors using evaluation-only single-FET checks.

No fitting, model selection, or reference-result replacement occurs. Fresh
single-FET decks reproduce the actual recorded gate biases, including internal
gate voltages away from a supply rail. Only the explanatory JSON/LaTeX and
regenerable diagnostic logs are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lusr.sampling import CASES, INPUT_COLUMNS
from lusr.spice import DEFAULT_MODEL, fresh_deck, find_ngspice, run_deck, sha256


def read(root, relative):
    return pd.read_csv(root / relative, float_precision="round_trip", dtype={"input_bits": str})


def power(row, currents):
    return ((row["vd"] - row["vs"]) * currents[0]
            + (row["vg"] - row["vs"]) * currents[1]
            + (row["vb"] - row["vs"]) * currents[3])


def percent_error(predicted, reference, floor=1e-18):
    return 100 * (np.asarray(predicted) - reference) / np.maximum(np.abs(reference), floor)


def validate_artifacts(root, analysis=None):
    """Check explanatory evidence without running SPICE or loading test models."""
    root = Path(root)
    if analysis is None:
        analysis = json.loads((root / "results/error_analysis.json").read_text())
    if analysis["status"] != "PASSED" or not analysis["evaluation_only"]:
        raise ValueError("Worst-case explanation has not passed its evaluation checks")
    if analysis["analysis_source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Worst-case explanation source changed; regenerate its artifacts")
    if analysis["assigned_model_sha256"] != sha256(DEFAULT_MODEL):
        raise ValueError("Worst-case explanation used another model card")
    for name, digest in analysis["input_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(root / name) != digest:
            raise ValueError(f"Worst-case explanation input changed: {name}")
    gates = read(root, "results/gate_predictions.csv")
    devices = read(root, "results/gate_transistor_predictions.csv")
    expected = gates.relative_error.abs().nlargest(5).index.tolist()
    if [g["gate_row"] for g in analysis["worst_gates"]] != expected:
        raise ValueError("Worst-case attribution does not cover the five largest gate errors")
    for record in analysis["worst_gates"]:
        gate = gates.loc[record["gate_row"]]
        rows = devices[devices.gate_row == record["gate_row"]].set_index("device")
        if set(rows.index) != {d["device"] for d in record["devices"]}:
            raise ValueError("Incomplete per-device gate attribution")
        reference = gate.spice_leakage_A * gate.vdd
        total = 0.
        for device in record["devices"]:
            row = rows.loc[device["device"]]
            currents = [device["spice_currents_A"][t] for t in ("id", "ig", "is", "ib")]
            measured = power(row, currents)
            contribution = 100 * (row.power_closed_W - measured) / reference
            if not np.isclose(measured, device["spice_power_W"], rtol=1e-12, atol=1e-25):
                raise ValueError("Incorrect single-FET explanatory power")
            if not np.isclose(contribution, device["gate_error_contribution_pp"], rtol=1e-10, atol=1e-10):
                raise ValueError("Incorrect device error attribution")
            total += measured
        if not np.isclose(total, reference, rtol=2e-6, atol=1e-20):
            raise ValueError("Explanatory single-FET powers fail whole-gate reconstruction")
    circuit = read(root, "results/circuit_predictions.csv")
    row = circuit.loc[circuit.relative_error.abs().idxmax()]
    worst = analysis["worst_circuit"]
    if (int(row.corner_id), int(row.vector)) != (worst["corner_id"], worst["vector"]):
        raise ValueError("Wrong complete-circuit outlier")
    components = [100 * (row.ml_leakage_A - row.standalone_spice_sum_A) / row.spice_leakage_A,
                  100 * (row.standalone_spice_sum_A - row.spice_leakage_A) / row.spice_leakage_A]
    if not np.allclose(components, [worst["surrogate_contribution_pp"], worst["composition_contribution_pp"]], rtol=1e-10, atol=1e-10):
        raise ValueError("Incorrect circuit error decomposition")
    if not np.isclose(sum(components), worst["total_error_percent"], rtol=1e-10, atol=1e-10):
        raise ValueError("Circuit error contributions do not sum to the total")
    if analysis["or2_00_gate_bias_check"]["corners"] != 200 or len(analysis["training_reference_checks"]) != 12:
        raise ValueError("Incomplete controlled bias or simulator reference checks")
    return {"status": "PASSED", "gate_attributions": 5, "controlled_bias_corners": 200,
            "fresh_single_fet_simulations": analysis["fresh_single_fet_simulations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CODE_ROOT.parent)
    parser.add_argument("--ngspice", help="Executable used for the evaluation-only checks")
    args = parser.parse_args()
    root = args.output_root.resolve()
    executable = find_ngspice(args.ngspice)
    directory = root / "logs/worst_case_explanation"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "results/ml_manifest.json").read_text())
    if manifest["status"] != "COMPLETED":
        raise ValueError("Explain the frozen, completed model evaluation only")
    for case in CASES:
        if sha256(root / f"models/{case}.joblib") != manifest["cases"][case]["selected_model_sha256"]:
            raise ValueError(f"Frozen model changed: {case}")
    gates = read(root, "results/gate_predictions.csv")
    devices = read(root, "results/gate_transistor_predictions.csv")
    circuit = read(root, "results/circuit_predictions.csv")
    corners = read(root, "data/corners.csv").set_index("corner_id")
    cache = {}

    def simulate(case, values):
        row = {key: float(values[key]) for key in INPUT_COLUMNS}
        key = (case, *(row[k] for k in INPUT_COLUMNS))
        if key not in cache:
            # fresh_deck intentionally validates the assigned fixed gate bias.
            # Instantiate that unchanged implementation, then replace all four
            # voltage-source values with the measured in-gate query. No clipping.
            scaffold = dict(row, vd=0., vs=0.,
                            vg=row["vdd"] if case in ("nmos_on", "pmos_off") else 0.,
                            vb=row["vdd"] if case.startswith("pmos") else 0.)
            deck = fresh_deck(case, scaffold, DEFAULT_MODEL)
            for terminal in "dgsb":
                deck, count = re.subn(rf"(?m)^V{terminal} {terminal} 0 \S+$",
                                     f"V{terminal} {terminal} 0 {row['v' + terminal]:.17g}", deck)
                if count != 1:
                    raise ValueError("Missing independent terminal source")
            values, _ = run_deck(deck, directory / f"fet_{len(cache):04d}.cir", executable, 1)
            cache[key] = np.array([values[0][t] for t in ("id", "ig", "is", "ib")])
        return cache[key]

    # Check the same simulator/model against spread-out original training data.
    reference_checks = []
    for case in CASES:
        frame = read(root, f"data/transistors/{case}.csv")
        for index in (0, len(frame) // 2, len(frame) - 1):
            row = frame.iloc[index]
            currents = simulate(case, row)
            reference = row[["id", "ig", "is", "ib"]].to_numpy(dtype=float)
            if not np.allclose(currents, reference, rtol=2e-6, atol=2e-15):
                raise ValueError(f"Fresh single-FET reference mismatch: {case}/{index}")
            reference_checks.append({"case": case, "row_index": index,
                                     "max_absolute_difference_A": float(np.max(np.abs(currents - reference)))})

    worst = gates.loc[gates.relative_error.abs().nlargest(5).index]
    attributions = []
    report_table = []
    for gate_row, gate in worst.iterrows():
        reference_power = float(gate.spice_leakage_A * gate.vdd)
        records = []
        for _, row in devices[devices.gate_row == gate_row].iterrows():
            currents = simulate(row["case"], row)
            measured_power = float(power(row, currents))
            predicted_power = float(row.power_closed_W)
            records.append({"device": row.device, "case": row["case"],
                            "spice_currents_A": dict(zip(("id", "ig", "is", "ib"), currents.tolist())),
                            "spice_power_W": measured_power, "ml_power_W": predicted_power,
                            "leakage_share_percent": 100 * measured_power / reference_power,
                            "device_error_percent": float(percent_error(predicted_power, measured_power)),
                            "gate_error_contribution_pp": 100 * (predicted_power - measured_power) / reference_power})
        measured_sum = sum(r["spice_power_W"] for r in records)
        if not np.isclose(measured_sum, reference_power, rtol=2e-6, atol=1e-20):
            raise ValueError("Single-FET powers do not reconstruct the whole-gate SPICE power")
        dominant = max(records, key=lambda r: abs(r["gate_error_contribution_pp"]))
        group = [r for r in records if r["case"] == dominant["case"]]
        contribution = sum(r["gate_error_contribution_pp"] for r in group)
        label = ("Output-inverter PMOS" if gate.circuit == "OR2" else
                 "Two NMOS" if gate.circuit == "NOR2" else "NMOS")
        entry = {"gate_row": int(gate_row), "gate": gate.circuit, "input_bits": gate.input_bits,
                 "corner_id": int(gate.corner_id), "signed_error_percent": float(100 * gate.relative_error),
                 "fresh_power_sum_relative_error": (measured_sum - reference_power) / reference_power,
                 "dominant_case": dominant["case"], "dominant_contribution_pp": contribution,
                 "devices": records}
        attributions.append(entry)
        report_table.append([f"{gate.circuit} {gate.input_bits} / {int(gate.corner_id)}",
                             f"{100 * gate.relative_error:+.3f}", label, f"{contribution:+.3f}"])

    # Controlled counterfactual: change only the output-inverter OFF PMOS gate
    # back to VDD for every OR2/00 corner; compare against the same frozen model.
    select = (devices.gate == "OR2") & (devices.vector == 0) & (devices.device == "g_pi")
    inverter = devices[select].sort_values("corner_id").copy()
    actual_power, rail_power = [], []
    for _, row in inverter.iterrows():
        actual_power.append(power(row, simulate("pmos_off", row)))
        rail = row.copy()
        rail["vg"] = rail["vdd"]
        rail_power.append(power(rail, simulate("pmos_off", rail)))
    actual_power, rail_power = np.array(actual_power), np.array(rail_power)
    rail_inputs = inverter[list(INPUT_COLUMNS)].to_numpy().copy()
    rail_inputs[:, INPUT_COLUMNS.index("vg")] = rail_inputs[:, INPUT_COLUMNS.index("vdd")]
    model = joblib.load(root / "models/pmos_off.joblib")
    with threadpool_limits(limits=1):
        rail_currents = model.predict(rail_inputs)
    rail_ml = np.array([power(dict(zip(INPUT_COLUMNS, x)), y) for x, y in zip(rail_inputs, rail_currents)])
    sag = 1000 * (inverter.vdd - inverter.vg)
    sag_increase = percent_error(actual_power, rail_power)
    real_error = percent_error(inverter.power_closed_W.to_numpy(), actual_power)
    rail_error = percent_error(rail_ml, rail_power)
    sag_summary = {"corners": len(inverter), "median_gate_sag_mV": float(sag.median()),
                   "maximum_gate_sag_mV": float(sag.max()),
                   "median_true_power_increase_percent": float(np.median(sag_increase)),
                   "maximum_true_power_increase_percent": float(np.max(sag_increase)),
                   "median_actual_bias_model_error_percent": float(np.median(real_error)),
                   "median_rail_bias_model_error_percent": float(np.median(rail_error)),
                   "sag_temperature_spearman": float(sag.corr(inverter.temp, method="spearman"))}

    off_test = {}
    for case in ("nmos_off", "pmos_off"):
        frame = read(root, f"data/transistors/{case}.csv")
        prediction = read(root, f"results/test_predictions/{case}.csv")
        x = frame.iloc[prediction.row_index.to_numpy()]
        y = prediction[[t + "_spice" for t in ("id", "ig", "is", "ib")]].to_numpy()
        p = prediction[[t + "_ml" for t in ("id", "ig", "is", "ib")]].to_numpy()
        ref = ((x.vd - x.vs).to_numpy() * y[:, 0] + (x.vg - x.vs).to_numpy() * y[:, 1]
               + (x.vb - x.vs).to_numpy() * y[:, 3]) / x.vdd.to_numpy()
        pred = ((x.vd - x.vs).to_numpy() * p[:, 0] + (x.vg - x.vs).to_numpy() * p[:, 1]
                + (x.vb - x.vs).to_numpy() * p[:, 3]) / x.vdd.to_numpy()
        error = np.abs(percent_error(pred, ref))
        off_test[case] = {"rows": len(error), "p95_power_error_percent": float(np.quantile(error, .95)),
                          "p99_power_error_percent": float(np.quantile(error, .99)),
                          "power_error_above_4_4_percent_count": int(np.sum(error > 4.4))}

    row = circuit.loc[circuit.relative_error.abs().idxmax()]
    circuit_worst = {"corner_id": int(row.corner_id), "vector": int(row.vector), "input_bits": row.input_bits,
                     "temp_C": float(row.temp), "vdd_V": float(row.vdd),
                     "total_error_percent": float(100 * row.relative_error),
                     "surrogate_contribution_pp": float(100 * (row.ml_leakage_A - row.standalone_spice_sum_A) / row.spice_leakage_A),
                     "composition_contribution_pp": float(100 * (row.standalone_spice_sum_A - row.spice_leakage_A) / row.spice_leakage_A)}
    local = next(g for g in attributions if g["gate"] == "INV")
    nmos = next(d for d in local["devices"] if d["case"] == "nmos_off")
    corner = corners.loc[local["corner_id"]]
    nmos_outlier = {"corner_id": local["corner_id"],
                    "toxe_corner_percentile": float(100 * (corners.n_toxe <= corner.n_toxe).mean()),
                    "vdd_corner_percentile": float(100 * (corners.vdd <= corner.vdd).mean()),
                    "device_error_percent": nmos["device_error_percent"],
                    "leakage_share_percent": nmos["leakage_share_percent"]}
    paragraphs = [
        (f"OR2 00: the NOR output drives the output-inverter OFF PMOS gate below VDD. "
         f"Across 200 corners this shortfall has median {sag_summary['median_gate_sag_mV']:.2f} mV "
         f"and maximum {sag_summary['maximum_gate_sag_mV']:.2f} mV. "
         f"Changing only that gate voltage back to VDD in fresh single-FET simulations shows that "
         f"the shortfall raises its true power by a median {np.median(sag_increase):.2f}% "
         f"and up to {np.max(sag_increase):.2f}%. The frozen model's median signed error is "
         f"{np.median(rail_error):+.2f}% at the prescribed rail bias, but {np.median(real_error):+.2f}% "
         f"at the actual internal-node bias. OFF training fixes the gate at its prescribed rail; "
         f"the learned OFF channel features omit gate-voltage variation. This explains the "
         f"systematic underprediction, especially at hot corners."),
        (f"INV 0 and NOR2 00 at corner {local['corner_id']}: the OFF NMOS group supplies approximately "
         f"{nmos['leakage_share_percent']:.1f}% of the gate leakage and is overpredicted by about "
         f"{nmos['device_error_percent']:.2f}% per device, despite an exactly 0 V gate bias. "
         f"This is a local regression outlier rather than the OR2 gate-bias mechanism. Only "
         f"{off_test['nmos_off']['power_error_above_4_4_percent_count']} of the 15,000 held-out NMOS OFF "
         f"power predictions exceed 4.4% error. This corner combines relatively thin NMOS oxide "
         f"(lower {nmos_outlier['toxe_corner_percentile']:.1f}% of corners) with high VDD "
         f"(upper {100 - nmos_outlier['vdd_corner_percentile']:.0f}% of corners). The separate causal effect "
         f"of those process parameters has not been isolated."),
        (f"Complete parity checker: the worst row is input {row.input_bits}, corner {int(row.corner_id)} "
         f"at {row.temp:.1f} degrees C, with signed error {100 * row.relative_error:+.3f}%. "
         f"Relative to the same full-circuit SPICE reference, the frozen surrogate contributes "
         f"{circuit_worst['surrogate_contribution_pp']:+.3f} percentage points and replacing the "
         f"connected circuit by standalone ideal-input gates contributes "
         f"{circuit_worst['composition_contribution_pp']:+.3f} percentage points. These contributions "
         f"sum to the total error. Thus the prescribed ideal-input/no-loading hierarchy dominates "
         f"this circuit outlier. This diagnostic uses existing gate and complete-circuit references "
         f"only for evaluation; it does not change any ML prediction.")]
    inputs = ["results/ml_manifest.json", "results/gate_predictions.csv", "results/gate_transistor_predictions.csv",
              "results/circuit_predictions.csv", "data/corners.csv",
              *(f"data/transistors/{case}.csv" for case in CASES),
              *(f"results/test_predictions/{case}.csv" for case in ("nmos_off", "pmos_off"))]
    result = {"schema_version": 1, "status": "PASSED", "evaluation_only": True,
              "simulator": subprocess.run([str(executable), "--version"], capture_output=True, text=True, check=True).stdout.strip(),
              "simulator_sha256": sha256(executable), "assigned_model_sha256": sha256(DEFAULT_MODEL),
              "analysis_source_sha256": sha256(Path(__file__)),
              "input_sha256": {name: sha256(root / name) for name in inputs},
              "fresh_single_fet_simulations": len(cache), "training_reference_checks": reference_checks,
              "worst_gates": attributions, "or2_00_gate_bias_check": sag_summary,
              "off_test_power_errors": off_test, "nmos_local_outlier": nmos_outlier,
              "worst_circuit": circuit_worst,
              "report_intro": "Fresh single-FET simulations at the recorded in-gate biases reconstruct each of the five worst gates' SPICE powers. The table attributes signed gate error to the dominant OFF-device group. Contributions are percentage points of whole-gate error; they include each group's leakage share.",
              "report_table": report_table, "report_paragraphs": paragraphs}
    destination = root / "results/error_analysis.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    validate_artifacts(root, result)
    print(f"PASSED: {len(cache)} fresh single-FET checks; five gate attributions and worst circuit explained.")
    print(destination)


if __name__ == "__main__":
    main()
