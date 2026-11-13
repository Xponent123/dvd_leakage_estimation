#!/usr/bin/env python3
"""Strict final deliverable/provenance audit. No fitting or simulations occur.

Run after 07_build_report.py. Missing artifacts or stale checksums are errors;
--inventory-only lists pending files without opening held-out result values.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOURCE_ROOT
sys.path.insert(0, str(SOURCE_ROOT))
from lusr.spice import CASES, CURRENT_COLUMNS, DEFAULT_MODEL, INPUT_COLUMNS, OPTIONS, sha256

ORIGINAL_MODEL_SHA256 = "56710120d5a2ae9e32a1908e1ee29a98f692a1dec0501eed8976ea6e6bea021e"
BASELINE_FAMILIES = {"random_forest", "extra_trees", "hist_gradient_boosting", "mlp"}
SHARED_FILES = ("requirements.lock.txt", "scripts/run_revision.sh")
GATE_COUNTS = {"INV": 2, "NAND2": 4, "NOR2": 4, "AND2": 4, "OR2": 4}
REPORT = "reports/LUSR_8bit_Parity_Leakage_Report"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(relative):
    return json.loads((ROOT/relative).read_text())


def rows(relative, header=None):
    with (ROOT/relative).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if header is not None:
            require(tuple(reader.fieldnames or ()) == tuple(header), f"Wrong CSV schema: {relative}")
        return list(reader)


def verify_hash(relative, expected, *, source=False):
    path = Path(relative)
    path = path if path.is_absolute() else (SOURCE_ROOT if source else ROOT)/path
    require(sha256(path) == expected, f"Checksum mismatch: {path}")


def finite(records, columns, name):
    require(all(math.isfinite(float(r[c])) for r in records for c in columns), f"Nonfinite values: {name}")


def inventory():
    files = ["README.md", "data/input_manifest.json",
             "data/corners.csv", "data/transistors/simulation_metadata.json", "results/ml_manifest.json",
             "reports/spice_both_manifest.json", "reports/circuit_reference_validation.json",
             "reports/data_checks/data_checks.json", "reports/data_checks/trend_manifest.json",
             "reports/data_checks/trends.csv", "results/transistor_validation_models.csv",
             "results/transistor_test_metrics.csv", "results/transistor_test_terminals.csv",
             "data/gates/reference.csv", "data/circuit/reference.csv", "results/gate_predictions.csv",
             "results/circuit_predictions.csv", "results/gate_metrics.csv", "results/circuit_metrics_by_vector.csv",
             "results/hierarchy_summary.json", REPORT+".pdf", REPORT+".md", "netlists/circuit/parity8.cir"]
    for case in CASES:
        files += [f"data/inputs/{case}_inputs.csv", f"data/splits/{case}_split.csv", f"data/transistors/{case}.csv",
                  f"models/{case}.joblib", f"results/test_predictions/{case}.csv", f"netlists/transistors/{case}.cir",
                  f"reports/figures/{case}_parity.png"]
    files += [f"netlists/gates/{gate.lower()}.cir" for gate in GATE_COUNTS]
    files += [f"reports/figures/{name}.png" for name in ("gate_parity", "circuit_parity", "architecture", "corner_error")]
    files += [f"reports/data_checks/plots/{name}.png" for name in
              ("off_current_vs_temperature", "off_current_vs_vth0", "gate_current_vs_toxe")]
    return files


def audit():
    source_hash = sha256(DEFAULT_MODEL)
    require(source_hash == ORIGINAL_MODEL_SHA256, "Original assigned 32nm_MGK model was changed")
    inputs = load("data/input_manifest.json")
    simulations = load("data/transistors/simulation_metadata.json")
    ml = load("results/ml_manifest.json")
    require(ml["status"] == "COMPLETED", "ML test evaluation has not been finalized")
    require(inputs["model"]["sha256"] == source_hash == simulations["model_sha256"], "Model provenance mismatch")
    verify_hash("data/input_manifest.json", ml["input_manifest_sha256"])
    checks = {"original_model_unchanged": source_hash, "transistor_datasets": {}, "models": {}}
    if "frozen_utc" in ml:
        require("test_completed_utc" in ml, "Frozen revision lacks test-completion timestamp")
        require(datetime.fromisoformat(ml["frozen_utc"]) <= datetime.fromisoformat(ml["test_completed_utc"]),
                "Test completion precedes model freeze")
        if "test_started_utc" in ml:
            require(datetime.fromisoformat(ml["frozen_utc"]) <= datetime.fromisoformat(ml["test_started_utc"]) <= datetime.fromisoformat(ml["test_completed_utc"]),
                    "Test start timestamp is outside freeze/completion interval")
        require(bool(ml.get("selection")), "Frozen revision lacks declared selection criterion")
        for name in ("protocol.json", *(f"{case}.csv" for case in CASES)):
            require("experiments/physics/boundary_validation/"+name in ml.get("validation_evidence", {}),
                    f"Frozen revision lacks independent boundary validation provenance: {name}")
        checks["model_freeze"] = {"frozen_utc": ml["frozen_utc"], "test_completed_utc": ml["test_completed_utc"]}
    for case in CASES:
        entry = inputs["datasets"][case]
        simulated = simulations["cases"][case]
        selected = ml["cases"][case]
        input_path = "data/"+entry["input_csv"]
        split_path = "data/"+entry["split_csv"]
        data_path = f"data/transistors/{case}.csv"
        verify_hash(input_path, entry["input_sha256"])
        verify_hash(split_path, entry["split_sha256"])
        verify_hash(data_path, simulated["output_sha256"])
        require(simulated["input_sha256"] == entry["input_sha256"], f"Simulation input provenance: {case}")
        measured = rows(data_path, (*INPUT_COLUMNS, *CURRENT_COLUMNS))
        sampled = rows(input_path, INPUT_COLUMNS)
        count = int(entry["rows"])
        require(count >= 10000 and len(measured) == len(sampled) == count, f"Incorrect dataset count: {case}")
        finite(measured, (*INPUT_COLUMNS, *CURRENT_COLUMNS), case)
        require(all(all(float(a[c]) == float(b[c]) for c in INPUT_COLUMNS) for a, b in zip(measured, sampled)), f"Simulation inputs changed: {case}")
        partitions = rows(split_path, ("row_index", "split"))
        require(len(partitions) == count and {int(r["row_index"]) for r in partitions} == set(range(count)), f"Invalid partition coverage: {case}")
        sizes = {name: sum(r["split"] == name for r in partitions) for name in ("train", "validation", "test")}
        ntrain, nvalidation = count*70//100, count*15//100
        expected_sizes = {"train": ntrain, "validation": nvalidation, "test": count-ntrain-nvalidation}
        require(sizes == expected_sizes == entry["split_counts"], f"Wrong partition sizes: {case}")
        provenance = selected["provenance"]
        require(provenance["current_sha256"] == simulated["output_sha256"] and
                provenance["input_sha256"] == entry["input_sha256"] and
                provenance["split_sha256"] == entry["split_sha256"] and provenance["partition_counts"] == sizes,
                f"Training provenance mismatch: {case}")
        model_path = f"models/{case}.joblib"
        verify_hash(model_path, selected["selected_model_sha256"])
        require(selected["reload_validation"].startswith("PASS"), f"Serialization check absent: {case}")
        prediction_path = f"results/test_predictions/{case}.csv"
        verify_hash(prediction_path, selected["test_predictions_sha256"])
        prediction = rows(prediction_path)
        expected_test = {int(r["row_index"]) for r in partitions if r["split"] == "test"}
        require(len(prediction) == sizes["test"] and {int(r["row_index"]) for r in prediction} == expected_test, f"Held-out prediction row mismatch: {case}")
        finite(prediction, [f"{terminal}_{origin}" for terminal in CURRENT_COLUMNS for origin in ("spice", "ml")], prediction_path)
        require(all(float(r[f"{t}_spice"]) == float(measured[int(r["row_index"])][t]) for r in prediction for t in CURRENT_COLUMNS), f"Held-out SPICE labels changed: {case}")
        checks["transistor_datasets"][case] = {"rows": count, "splits": sizes, "sha256": sha256(ROOT/data_path)}
        checks["models"][case] = {"sha256": selected["selected_model_sha256"], "family": selected["selected_family"], "test_rows": sizes["test"]}
    for relative, digest in ml.get("evaluation_source_sha256", {}).items():
        verify_hash(relative, digest, source=True)
    for relative, digest in ml.get("source_sha256", {}).items():
        verify_hash(relative, digest, source=True)
    for relative, digest in ml.get("validation_evidence", {}).items():
        verify_hash(relative, digest, source=True)
    if ml.get("selection_file"):
        # The original absolute execution path is provenance, not a dependency
        # on that machine. Verify the packaged selection beside these results.
        verify_hash("results/revision_selection.json", ml["selection_file_sha256"])
    if ml.get("validation_comparison_sha256"):
        verify_hash("results/transistor_validation_models.csv", ml["validation_comparison_sha256"])
    if ml.get("comparison_sha256"):
        verify_hash("results/transistor_validation_models.csv", ml["comparison_sha256"])
    checks["validation_evidence_files"] = len(ml.get("validation_evidence", {}))
    comparison = rows("results/transistor_validation_models.csv")
    families = ml.get("comparison_families", sorted(BASELINE_FAMILIES))
    require(len(families) == len(set(families)) == 4, "Exactly four distinct comparison families must be declared")
    require(len(comparison) == 16 and {(r["case"], r["model"]) for r in comparison} == {(c, f) for c in CASES for f in families}, "Model comparison is not four declared families × four cases")
    require(all(ml["cases"][case]["selected_family"] in families for case in CASES), "Selected model family missing from declared comparison")
    for case in CASES:
        chosen = [row for row in comparison if row["case"] == case and row.get("selected", "").lower() == "true"]
        require(len(chosen) == 1 and chosen[0]["model"] == ml["cases"][case]["selected_family"],
                f"Selected comparison row disagrees with model manifest: {case}")
    checks["comparison_families"] = families
    require(len(rows("results/transistor_test_metrics.csv")) == 4, "Expected four held-out model metric rows")
    require(len(rows("results/transistor_test_terminals.csv")) == 16, "Expected16 held-out terminal metric rows")
    corners = rows("data/corners.csv")
    require(len(corners) == 200 and {int(r["corner_id"]) for r in corners} == set(range(200)), "Expected200 unique corners")
    verify_hash("data/corners.csv", inputs["evaluation_corners"]["sha256"])
    spice = load("reports/spice_both_manifest.json")
    require(spice["corners"] == 200 and spice["status"] == "SIMULATED" and spice["model_sha256"] == source_hash, "Production simulation manifest invalid")
    verify_hash("data/corners.csv", spice["corner_sha256"])
    circuit_audit = load("reports/circuit_reference_validation.json")
    require(circuit_audit["status"] == "PASSED", "Independent circuit validation has not passed")
    require(circuit_audit["model_sha256"] == source_hash, "Independent reference audit used a different model")
    verify_hash("data/corners.csv", circuit_audit["corner_sha256"])
    require(len(circuit_audit["instrumented_gates"]) == 18 and circuit_audit["reverse_order"]["vectors"] == 256, "Circuit audit coverage incomplete")
    for label, count in (("gates", 3600), ("circuit", 51200)):
        reference_path = f"data/{label}/reference.csv"
        verify_hash(reference_path, spice["datasets"][label]["sha256"])
        verify_hash(reference_path, circuit_audit["gate_reference_sha256" if label == "gates" else "circuit_reference_sha256"])
        reference = rows(reference_path)
        predicted = rows(f"results/{'gate' if label == 'gates' else 'circuit'}_predictions.csv")
        expected_keys = ({(c, g, v) for c in range(200) for g, n in GATE_COUNTS.items() for v in range(n)}
                         if label == "gates" else {(c, "PARITY8", v) for c in range(200) for v in range(256)})
        key = lambda r: (int(r["corner_id"]), r["circuit"], int(r["vector"]))
        require(len(reference) == len(predicted) == count and {key(r) for r in reference} == {key(r) for r in predicted} == expected_keys,
                f"Missing/duplicate gate or circuit coverage: {label}")
        lookup = {key(r): r for r in reference}
        finite(predicted, ("spice_leakage_A", "ml_leakage_A", "output_V", "relative_error"), label)
        require(all(math.isclose(float(r["spice_leakage_A"]), float(lookup[key(r)]["spice_leakage_A"]), rel_tol=1e-14, abs_tol=1e-28) for r in predicted), f"Reference values changed in predictions: {label}")
        checks[label] = {"reference_rows": count, "predicted_rows": count, "sha256": sha256(ROOT/reference_path)}
    gate_metrics = rows("results/gate_metrics.csv")
    require(len(gate_metrics) == 18 and all(int(r["count"]) == 200 for r in gate_metrics), "Incorrect18-case gate metrics")
    vector_metrics = rows("results/circuit_metrics_by_vector.csv")
    require(len(vector_metrics) == 256 and {int(r["vector"]) for r in vector_metrics} == set(range(256)) and all(int(r["count"]) == 200 for r in vector_metrics), "Incorrect256-vector circuit metrics")
    hierarchy = load("results/hierarchy_summary.json")
    if hierarchy.get("hierarchy_source_sha256"):
        verify_hash("scripts/06_evaluate_hierarchy.py", hierarchy["hierarchy_source_sha256"], source=True)
    checks["independent_accuracy_recalculation"] = {}
    for label in ("gate", "circuit"):
        predicted = rows(f"results/{label}_predictions.csv")
        errors = [abs(float(r["ml_leakage_A"])-float(r["spice_leakage_A"]))/max(abs(float(r["spice_leakage_A"])),1e-18) for r in predicted]
        accuracy = 100*(1-sum(errors)/len(errors))
        require(math.isclose(accuracy,hierarchy[label]["mean_accuracy_percent"],rel_tol=0,abs_tol=1e-10),
                f"Reported mean accuracy does not match all prediction rows: {label}")
        checks["independent_accuracy_recalculation"][label] = {"rows":len(errors),"mean_accuracy_percent":accuracy,"meets_99_percent":accuracy>=99}
    if "ml_manifest_sha256" in hierarchy:
        verify_hash("results/ml_manifest.json", hierarchy["ml_manifest_sha256"])
    if "reference_audit_sha256" in hierarchy:
        verify_hash("reports/circuit_reference_validation.json", hierarchy["reference_audit_sha256"])
    for relative, digest in hierarchy.get("evaluation_source_sha256", {}).items():
        verify_hash(relative, digest, source=True)
    for case in CASES:
        verify_hash(f"models/{case}.joblib", hierarchy["model_hashes"][case])
    data_checks = load("reports/data_checks/data_checks.json")
    require(data_checks["source_model"]["sha256"] == source_hash, "Data-check model mismatch")
    for case in CASES:
        verify_hash(f"data/transistors/{case}.csv", data_checks["transistor_datasets"][case]["sha256"])
    trends = load("reports/data_checks/trend_manifest.json")
    verify_hash("reports/data_checks/trends.csv", trends["csv_sha256"])
    require(trends["rows"] == trends["fresh_process_count"] == 246, "Controlled trend dataset is incomplete")
    netlist_counts = {**{f"netlists/transistors/{case}.cir": 1 for case in CASES},
                      **{f"netlists/gates/{name}.cir": n for name, n in (("inv", 2), ("nand2", 4), ("nor2", 4), ("and2", 6), ("or2", 6))},
                      "netlists/circuit/parity8.cir": 112}
    for relative, expected in netlist_counts.items():
        text = (ROOT/relative).read_text()
        require(OPTIONS in text, f"Required simulator options missing: {relative}")
        require(len(re.findall(r"(?im)^M\S+\s", text)) == expected, f"Incorrect transistor count: {relative}")
        if "/transistors/" in relative:
            require(all(re.search(rf"(?im)^V{t}\s+{t}\s+0\s", text) for t in "dgsb"), f"Missing independent terminal source: {relative}")
    checks["netlist_device_counts"] = netlist_counts
    extractor = shutil.which("pdftotext")
    require(extractor is not None, "pdftotext is required for final report validation")
    pdf_text = subprocess.run([extractor, str(ROOT/(REPORT+".pdf")), "-"], capture_output=True, text=True, check=True).stdout
    require(len(pdf_text) > 5000, "Report PDF contains too little extracted text")
    for label, content in (("PDF", pdf_text), ("Markdown", (ROOT/(REPORT+".md")).read_text())):
        normalized = re.sub(r"\s+", " ", content).lower()
        total_transistors = sum(item["rows"] for item in checks["transistor_datasets"].values())
        for token in ("lusr", "32nm_mgk", f"{total_transistors:,}", "51,200", "200", "256", "112", "0.9"):
            require(token in normalized, f"Report {label} lacks required context/count: {token}")
        require(not re.search(r"\b(todo|tbd|fixme)\b|insert (?:result|figure|table) here|\[placeholder\]", normalized), f"Unfinished placeholder in report {label}")
    checks["typed_report"] = {"pdf_sha256": sha256(ROOT/(REPORT+".pdf")), "text_characters": len(pdf_text)}
    return checks


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=SOURCE_ROOT,
                        help="Artifact directory; shared code and setup files remain under the project root")
    parser.add_argument("--inventory-only", action="store_true", help="List missing artifacts; does not inspect held-out results or certify completion")
    args = parser.parse_args()
    ROOT = args.output_root.expanduser().resolve()
    required = inventory()
    missing = [p for p in required if not (ROOT/p).is_file() or (ROOT/p).stat().st_size == 0]
    missing += [f"shared:{p}" for p in SHARED_FILES if not (SOURCE_ROOT/p).is_file() or (SOURCE_ROOT/p).stat().st_size == 0]
    required_count = len(required)+len(SHARED_FILES)
    if args.inventory_only:
        print(json.dumps({"status": "INVENTORY_ONLY", "output_root": str(ROOT), "required_files": required_count, "missing": missing}, indent=2))
        return
    destination = ROOT/"reports/deliverables_validation.json"
    try:
        require(not missing, "Missing required deliverables: " + ", ".join(missing))
        result = {"status": "PASSED", "output_root": str(ROOT), "required_files": required_count, "checks": audit()}
    except Exception as exc:
        result = {"status": "FAILED", "error": str(exc), "missing": missing}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    destination.write_text(json.dumps(result, indent=2)+"\n")
    total = sum(item["rows"] for item in result["checks"]["transistor_datasets"].values())
    print(f"PASSED: {required_count} required files, {total:,} transistor rows, 3,600 gate rows, 51,200 circuit rows; {destination}")


if __name__ == "__main__":
    main()
