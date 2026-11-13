#!/usr/bin/env python3
"""Independent conservation, topology, completeness, and DC-order checks.

Creates separate manually wired CMOS decks, retaining all logs. This is a
verification script; no reference CSV or production netlist is overwritten.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lusr.circuits import (GATES, circuit_deck, parity_cells, process_model,
                           propagate, simulate_corner)
from lusr.model_card import read_model_card
from lusr.spice import DEFAULT_MODEL, OPTIONS, find_ngspice, sha256

AUDIT = ROOT / "logs/circuit_reference_validation"
FET = tuple[str, str, str, str, str, str]


def gate_devices(kind: str, prefix="q", inputs=("a", "b"), output="out") -> list[FET]:
    """Independent static CMOS wiring; tuple is name, model, drain, gate, source, body."""
    a, b = inputs[0], inputs[-1]
    x, z = prefix + "_x", prefix + "_z"
    def inv(name, incoming, outgoing):
        return [(name+"n", "nmos", outgoing, incoming, "0", "0"),
                (name+"p", "pmos", outgoing, incoming, "vdd", "vdd")]
    if kind == "INV":
        return inv(prefix, a, output)
    stage = z if kind in ("AND2", "OR2") else output
    if kind in ("NAND2", "AND2"):
        devices = [(prefix+"n0", "nmos", stage, a, x, "0"),
                   (prefix+"n1", "nmos", x, b, "0", "0"),
                   (prefix+"p0", "pmos", stage, a, "vdd", "vdd"),
                   (prefix+"p1", "pmos", stage, b, "vdd", "vdd")]
    elif kind in ("NOR2", "OR2"):
        devices = [(prefix+"n0", "nmos", stage, a, "0", "0"),
                   (prefix+"n1", "nmos", stage, b, "0", "0"),
                   (prefix+"p0", "pmos", x, a, "vdd", "vdd"),
                   (prefix+"p1", "pmos", stage, b, x, "vdd")]
    else:
        raise ValueError(kind)
    return devices + (inv(prefix+"i", z, output) if kind in ("AND2", "OR2") else [])


def independent_parity() -> list[FET]:
    devices = []
    def exclusive_or(name, left, right, output):
        t, u, v = name+"t", name+"u", name+"v"
        for suffix, a, b, out in (("0", left, right, t), ("1", left, t, u),
                                  ("2", right, t, v), ("3", u, v, output)):
            devices.extend(gate_devices("NAND2", name+suffix, (a, b), out))
    for pair in range(4):
        exclusive_or(f"r{pair}", f"a{2*pair}", f"a{2*pair+1}", f"p{pair}")
    exclusive_or("r4", "p0", "p1", "p4")
    exclusive_or("r5", "p2", "p3", "p5")
    exclusive_or("r6", "p4", "p5", "out")
    return devices


def independent_model(corner: dict) -> str:
    lines = DEFAULT_MODEL.read_text().splitlines()
    prefix = None
    count = 0
    for index, line in enumerate(lines):
        match = re.match(r"\.model\s+(nmos|pmos)\b", line, re.I)
        if match:
            prefix = match.group(1)[0].lower()
        if prefix:
            def replace(match):
                nonlocal count
                count += 1
                key = match.group(1).lower()
                return key + " = " + format(corner[prefix+"_"+key], ".17g")
            lines[index] = re.sub(r"\b(vth0|toxe|ndep|u0|xj)\s*=\s*\S+", replace, line, flags=re.I)
    assert count == 10
    return "\n".join(lines)


def independent_deck(corner, kind, vector, probes=False):
    count = 8 if kind == "PARITY8" else (1 if kind == "INV" else 2)
    names = [f"a{i}" for i in range(8)] if kind == "PARITY8" else list("ab"[:count])
    shifts = list(range(8)) if kind == "PARITY8" else list(reversed(range(count)))
    levels = [(vector >> i) & 1 for i in shifts]
    fets = independent_parity() if kind == "PARITY8" else gate_devices(kind, inputs=tuple(names))
    vdd = corner["vdd"]
    lines = [f"Independent audit {kind} vector {vector}", independent_model(corner), OPTIONS,
             f'.temp {corner["temp"]:.17g}', f"Vdd vdd 0 {vdd:.17g}"]
    for index, (name, bit) in enumerate(zip(names, levels)):
        lines.append(f"Vin{index} {name} 0 {bit*vdd:.17g}")
    for index, (_, model, *terminals) in enumerate(fets):
        device_nodes = []
        for terminal, net in zip("dgsb", terminals):
            local = f"f{index}{terminal}"
            if probes:
                lines.append(f"Vprobe{index}{terminal} {net} {local} 0")
            device_nodes.append(local if probes else net)
        width = 64e-9 if model == "nmos" else 128e-9
        lines.append(f'M{index} {" ".join(device_nodes)} {model} L=32n W={width:.17g}')
    sources = ["i(Vdd)"] + [f"i(Vin{i})" for i, bit in enumerate(levels) if bit]
    lines += [".control", "set numdgt=17", "op",
              "let ext_current = -(" + "+".join(sources) + ")",
              "let supply_current = -i(Vdd)", "let out_voltage = v(out)",
              "print ext_current supply_current out_voltage"]
    if probes:
        for index in range(len(fets)):
            for terminal in "dgsb":
                lines += [f"let c{index}{terminal} = i(Vprobe{index}{terminal})",
                          f"let v{index}{terminal} = v(f{index}{terminal})",
                          f"print c{index}{terminal} v{index}{terminal}"]
    return "\n".join(lines + ["quit", ".endc", ".end", ""]), fets


def run_independent(corner, kind, vector, exe, label, probes=False):
    deck, fets = independent_deck(corner, kind, vector, probes)
    directory = AUDIT / label
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind.lower()}_{vector:03d}.cir"
    path.write_text(deck)
    run = subprocess.run([str(exe), "-n", "-b", str(path)], cwd=directory,
                         capture_output=True, text=True, timeout=120)
    log = run.stdout + "\nSTDERR:\n" + run.stderr
    path.with_suffix(".log").write_text(log)
    if run.returncode or re.search(r"(?im)^\s*(error|fatal)\b|simulation\(s\) aborted", log):
        raise RuntimeError(f"Independent audit simulation failed: {path}")
    data = {m.group(1): float(m.group(2)) for line in run.stdout.splitlines()
            if (m := re.fullmatch(r"\s*([a-z][a-z0-9_]*)\s*=\s*(\S+)\s*", line))}
    required = {"ext_current", "supply_current", "out_voltage"}
    if probes:
        required |= {f"{prefix}{index}{terminal}" for prefix in "cv"
                     for index in range(len(fets)) for terminal in "dgsb"}
    assert required <= set(data) and all(math.isfinite(data[key]) for key in required)
    return data, fets


def close(actual, expected, absolute=2e-14, relative=2e-6):
    tolerance = absolute + relative * abs(expected)
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Mismatch {actual} vs {expected}, tolerance {tolerance}")
    return abs(actual - expected)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    global ROOT, AUDIT
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args()
    ROOT=args.output_root.resolve()
    AUDIT=ROOT/'logs/circuit_reference_validation'
    exe = find_ngspice()
    AUDIT.mkdir(parents=True, exist_ok=True)
    corners = [{key: float(value) for key, value in row.items()} for row in read_csv(ROOT/"data/corners.csv")]
    gate_rows = read_csv(ROOT/"data/gates/reference.csv")
    circuit_rows = read_csv(ROOT/"data/circuit/reference.csv")
    assert len(corners) == 200 and {int(c["corner_id"]) for c in corners} == set(range(200))
    nominal_values = read_model_card(DEFAULT_MODEL).nominal
    for corner in corners:
        assert .81 <= corner["vdd"] <= .99 and -40 <= corner["temp"] <= 125
        for kind, parameters in nominal_values.items():
            for key, value in parameters.items():
                assert .9 <= corner[kind[0]+"_"+key]/value <= 1.1
    expected_gates = {(c, g, v) for c in range(200) for g in GATES for v in range(2 if g == "INV" else 4)}
    actual_gates = {(int(r["corner_id"]), r["circuit"], int(r["vector"])) for r in gate_rows}
    actual_circuit = {(int(r["corner_id"]), int(r["vector"])) for r in circuit_rows}
    assert len(gate_rows) == 3600 and actual_gates == expected_gates
    assert len(circuit_rows) == 51200 and actual_circuit == {(c, v) for c in range(200) for v in range(256)}
    by_corner = {int(c["corner_id"]): c for c in corners}
    for row in gate_rows + circuit_rows:
        corner = by_corner[int(row["corner_id"])]
        assert float(row["vdd"]) == corner["vdd"] and float(row["temp"]) == corner["temp"]
    gate0 = {(r["circuit"], int(r["vector"])): r for r in gate_rows if int(r["corner_id"]) == 0}
    circuit0 = {int(r["vector"]): r for r in circuit_rows if int(r["corner_id"]) == 0}
    for vector in range(256):
        nodes, _ = propagate(vector)
        assert nodes["out"] == vector.bit_count() % 2
    assert len(parity_cells()) == 28 and len(independent_parity()) == 112
    report = {"status": "PASSED", "model_sha256": sha256(DEFAULT_MODEL),
              "corner_sha256": sha256(ROOT/"data/corners.csv"),
              "gate_reference_sha256": sha256(ROOT/"data/gates/reference.csv"),
              "circuit_reference_sha256": sha256(ROOT/"data/circuit/reference.csv"),
              "completeness": {"corners": 200, "gate_rows": 3600, "circuit_rows": 51200,
                               "no_missing_or_duplicate_keys": True, "corner_vdd_temperature_match": True,
                               "all_corner_parameters_within_assigned_bounds": True},
              "topology": {"boolean_vectors_checked": 256, "nand2_cells": 28, "fets": 112,
                           "parity_integer_bit_order": "a0=LSB; displayed input_bits=a7...a0",
                           "gate_bit_order": "a=MSB,b=LSB"},
              "instrumented_gates": [], "reverse_order": {}, "fresh_vectors": []}
    # The independently generated corner cards must agree in all swept values.
    audit_model = AUDIT/"independent_corner0.pm"
    root_model = AUDIT/"production_corner0.pm"
    audit_model.write_text(independent_model(corners[0]))
    root_model.write_text(process_model(corners[0]))
    assert read_model_card(audit_model).nominal == read_model_card(root_model).nominal
    for kind in GATES:
        for vector in range(2 if kind == "INV" else 4):
            data, fets = run_independent(corners[0], kind, vector, exe, "instrumented_gates", probes=True)
            power_terms = [data[f"v{i}{t}"] * data[f"c{i}{t}"] for i in range(len(fets)) for t in "dgsb"]
            fet_power = math.fsum(power_terms)
            external_power = data["ext_current"] * corners[0]["vdd"]
            power_error = close(fet_power, external_power, absolute=2e-14)
            maximum_kcl = max(abs(math.fsum(data[f"c{i}{t}"] for t in "dgsb")) for i in range(len(fets)))
            reference = gate0[(kind, vector)]
            reference_error = close(data["ext_current"], float(reference["spice_leakage_A"]))
            # Source-relative form should agree after KCL closure; this checks the
            # exact aggregation formula used for transistor-model predictions.
            source_relative = math.fsum(
                (data[f"v{i}{t}"] - data[f"v{i}s"]) * data[f"c{i}{t}"]
                for i in range(len(fets)) for t in "dgb")
            close(source_relative, external_power, absolute=2e-14)
            report["instrumented_gates"].append({"gate": kind, "vector": vector, "corner_id": 0,
                "terminal_probe_count": 4*len(fets), "external_power_W": external_power,
                "terminal_power_sum_W": fet_power, "source_relative_power_W": source_relative,
                "power_conservation_abs_error_W": power_error, "max_terminal_KCL_abs_A": maximum_kcl,
                "reference_leakage_abs_error_A": reference_error})
    print("Verified all18 gate cases with independent four-terminal ammeters and source-power conservation", flush=True)
    reverse = simulate_corner(corners[0], "PARITY8", exe, AUDIT/"reverse_parity", list(reversed(range(256))), cache=False)
    errors = [close(row["spice_leakage_A"], float(circuit0[row["vector"]]["spice_leakage_A"])) for row in reverse]
    output_errors = [close(row["output_V"], float(circuit0[row["vector"]]["output_V"]), absolute=2e-9) for row in reverse]
    report["reverse_order"] = {"corner_id": 0, "vectors": 256,
                               "max_leakage_difference_A": max(errors), "max_output_difference_V": max(output_errors)}
    selected = [0, 1, 2, 3, 85, 127, 128, 170, 254, 255]
    nominal = {"corner_id": -1, "vdd": .9, "temp": 27.}
    for kind, parameters in nominal_values.items():
        nominal.update({kind[0]+"_"+key: value for key, value in parameters.items()})
    root_nominal = simulate_corner(nominal, "PARITY8", exe, AUDIT/"nominal_reference", selected, cache=False)
    nominal_lookup = {row["vector"]: row for row in root_nominal}
    for label, corner, lookup in (("fresh_actual", corners[0], circuit0), ("fresh_nominal", nominal, nominal_lookup)):
        for vector in selected:
            data, _ = run_independent(corner, "PARITY8", vector, exe, label)
            reference = lookup[vector]
            leakage_error = close(data["ext_current"], float(reference["spice_leakage_A"]))
            output_error = close(data["out_voltage"], float(reference["output_V"]), absolute=2e-9)
            expected = vector.bit_count() % 2
            fraction = data["out_voltage"] / corner["vdd"]
            assert fraction > .8 if expected else fraction < .2
            report["fresh_vectors"].append({"condition": label, "vector": vector,
                "expected_output": expected, "output_V": data["out_voltage"],
                "leakage_A": data["ext_current"], "reference_difference_A": leakage_error,
                "output_reference_difference_V": output_error})
    report["review_findings"] = [
        "Standard CMOS pull-up/pull-down topologies, complemented composite gates and all fixed W/L values verified.",
        "Every source/drain/body terminal keeps the model/simulator default junction geometry; no fabricated layout areas.",
        "Source-current convention includes the main VDD supply and all input drivers at VDD; grounded drivers contribute zero power.",
        "All four measured terminal currents are positive into each device, so sum(Vterminal*Iterminal) equals external supplied DC power.",
        "VDD and temperature columns agree exactly with the shared corner table; independent derived model agrees in all ten varied process values.",
        "Within one corner, only input sources change between operating points. Reversed vector order and independently fresh decks agree within stated tolerance.",
        "Printed eight-bit input strings run a7 through a0, whereas actual parity netlist inputs a0 through a7 use LSB first; this is explicit in metadata.",
    ]
    report["tolerances"] = {"current_abs_A": 2e-14, "power_abs_W": 2e-14,
                            "relative": 2e-6, "output_abs_V": 2e-9}
    destination = ROOT/"reports/circuit_reference_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASSED: {destination}; max gate power discrepancy={max(r['power_conservation_abs_error_W'] for r in report['instrumented_gates']):.3g} W")


if __name__ == "__main__":
    main()
