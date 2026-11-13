#!/usr/bin/env python3
"""Compare four families on validation data and test only each selected model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from lusr.ml import (
    CURRENT_COLUMNS, DERIVED_FEATURES, ERROR_FLOOR_A, MODEL_FAMILIES,
    kcl_metrics, make_candidate, regression_metrics,
)
from lusr.sampling import CASES, DEFAULT_SEED, INPUT_COLUMNS, SPLIT_NAMES


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_numeric_csv(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader))
        if header != columns:
            raise ValueError(f"{path}: expected columns {columns}, found {header}")
        values = np.array([[float(value) for value in row] for row in reader], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: invalid or non-finite values")
    return values


def load_case(data_root: Path, case: str, input_manifest: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    metadata = input_manifest["datasets"][case]
    input_path = data_root / metadata["input_csv"]
    split_path = data_root / metadata["split_csv"]
    current_path = data_root / "transistors" / f"{case}.csv"
    for path, key in ((input_path, "input_sha256"), (split_path, "split_sha256")):
        if digest(path) != metadata[key]:
            raise ValueError(f"Input provenance mismatch for {path}")
    inputs = read_numeric_csv(input_path, INPUT_COLUMNS)
    measured = read_numeric_csv(current_path, (*INPUT_COLUMNS, *CURRENT_COLUMNS))
    if len(inputs) != metadata["rows"] or measured.shape[0] != inputs.shape[0]:
        raise ValueError(f"Incorrect row count for {case}")
    if not np.array_equal(measured[:, :len(INPUT_COLUMNS)], inputs):
        raise ValueError(f"Simulated input rows do not match sampled input rows for {case}")
    rows: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    with split_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_index", "split"]:
            raise ValueError(f"Invalid split header in {split_path}")
        for row in reader:
            if row["split"] not in rows:
                raise ValueError(f"Unknown partition {row['split']}")
            rows[row["split"]].append(int(row["row_index"]))
    partitions = {name: np.asarray(indices, dtype=np.int64) for name, indices in rows.items()}
    if not np.array_equal(np.sort(np.concatenate(tuple(partitions.values()))), np.arange(len(inputs))):
        raise ValueError(f"Overlapping or incomplete split indices for {case}")
    if any(len(indices) < 2 for indices in partitions.values()):
        raise ValueError("Each split needs at least two rows to report R²")
    provenance = {
        "current_csv": str(current_path), "current_sha256": digest(current_path),
        "input_sha256": metadata["input_sha256"], "split_sha256": metadata["split_sha256"],
        "partition_counts": {name: len(indices) for name, indices in partitions.items()},
    }
    return inputs, measured[:, len(INPUT_COLUMNS):], partitions, provenance


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def family_comparison(rows: list[dict], selection: str = "mape") -> list[dict]:
    """One validation-selected candidate per family and case (four by four)."""
    metric = "mape_pct" if selection == "mape" else "transformed_rmse"
    best = {}
    for row in rows:
        key = row["case"], row["model"]
        if key not in best or float(row[metric]) < float(best[key][metric]):
            best[key] = row
    return list(best.values())


def serializable_parameters(parameters: dict) -> dict:
    result = {}
    for key, value in parameters.items():
        if value is None or isinstance(value, (str, int, float, bool, tuple, list)):
            result[key] = value
        else:
            result[key] = repr(value)
    return result


def evaluate_selected(data_root: Path, output: Path, model_root: Path, cases: list[str], jobs: int) -> None:
    """Evaluate already finalized models; this stage never fits or selects a model."""
    manifest_path = output / "ml_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing"]["target"] = "Baseline: median-scale signed-current asinh. Tuned variants: candidate-specific current/conductance representation, median or numerical 1e-18 scale, and training-only standardization. See each selected_target definition and candidate settings."
    with (output / "transistor_validation_candidates.csv").open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    selection = "mape" if "MAPE" in manifest["selection"] else "transformed_rmse"
    write_rows(output / "transistor_validation_baselines.csv", [row for row in candidates if row["variant"] == "baseline"])
    write_rows(output / "transistor_validation_models.csv", family_comparison(candidates, selection))
    input_manifest = json.loads((data_root / "input_manifest.json").read_text(encoding="utf-8"))
    test_rows, terminal_rows = [], []
    for case in cases:
        metadata = manifest["cases"][case]
        if "test_predictions" in metadata:
            raise ValueError(f"Final test already evaluated for {case}; refusing an accidental repeat")
        inputs, currents, partitions, provenance = load_case(data_root, case, input_manifest)
        if provenance != metadata["provenance"]:
            raise ValueError(f"Dataset changed since model selection for {case}")
        model_path = model_root / f"{case}.joblib"
        if digest(model_path) != metadata["selected_model_sha256"]:
            raise ValueError(f"Selected model changed for {case}")
        model = joblib.load(model_path)
        metadata["selected_target"] = {
            "representation": model.target_mode, "asinh_scale_mode": model.target_scale_mode,
            "asinh_scales": model.training_target_scale.tolist(),
            "standardize_transformed_targets": model.standardize_targets,
            "scales_units_in_output_order": ["S", "A", "S", "A"] if model.target_mode == "conductance" else ["A"] * 4,
        }
        testing = partitions["test"]
        with threadpool_limits(limits=jobs):
            predicted = model.predict(inputs[testing])
        test_rows.append({"case": case, "model": model.family, "variant": model.variant, "rows": len(testing), **regression_metrics(currents[testing], predicted), **kcl_metrics(predicted)})
        for index, terminal in enumerate(CURRENT_COLUMNS):
            terminal_rows.append({"case": case, "model": model.family, "variant": model.variant, "terminal": terminal, **regression_metrics(currents[testing, index], predicted[:, index])})
        prediction_rows = []
        for position, row_index in enumerate(testing):
            row = {"row_index": int(row_index)}
            for index, terminal in enumerate(CURRENT_COLUMNS):
                row[f"{terminal}_spice"] = currents[row_index, index]
                row[f"{terminal}_ml"] = predicted[position, index]
            prediction_rows.append(row)
        path = output / "test_predictions" / f"{case}.csv"
        write_rows(path, prediction_rows)
        metadata["test_predictions"] = str(path)
        metadata["test_predictions_sha256"] = digest(path)
        print(f"[{case}] finalized {model.family}/{model.variant}: held-out test evaluated.", flush=True)
    write_rows(output / "transistor_test_metrics.csv", test_rows)
    write_rows(output / "transistor_test_terminals.csv", terminal_rows)
    manifest["status"] = "COMPLETED"
    manifest["evaluation_source_sha256"] = {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in (Path(__file__).resolve(), PROJECT_ROOT / "lusr" / "ml.py")}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--smoke", action="store_true", help="small estimators for pipeline checks, not final models")
    parser.add_argument("--defer-test", action="store_true", help="save validation-selected models without opening test targets for evaluation")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate saved finalized models once; no fitting or selection")
    parser.add_argument("--selection", choices=("mape", "transformed_rmse"), default="mape")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4 to avoid oversubscribing simulation work")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.smoke and (args.output.resolve() == PROJECT_ROOT / "results" or args.models.resolve() == PROJECT_ROOT / "models"):
        parser.error("--smoke needs separate --output and --models paths, to protect final outputs")
    data_root, output, model_root = args.data.resolve(), args.output.resolve(), args.models.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        evaluate_selected(data_root, output, model_root, args.cases, args.jobs)
        return
    input_manifest_path = data_root / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    validation_rows, validation_terminal_rows = [], []
    manifest = {
        "status": "VALIDATION_COMPLETED_SMOKE" if args.smoke else "VALIDATION_COMPLETED",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "input_manifest_sha256": digest(input_manifest_path),
        "seed": args.seed, "worker_threads": args.jobs, "input_columns": INPUT_COLUMNS,
        "derived_feature_columns": DERIVED_FEATURES, "current_columns": CURRENT_COLUMNS,
        "selection": "Minimum pooled validation " + ("MAPE with 1e-18 A denominator floor" if args.selection == "mape" else "RMSE in asinh(I / training-median scale) space") + "; candidate order breaks exact ties.",
        "preprocessing": {
            "input": "StandardScaler fitted only to the training partition, after deterministic physical features.",
            "target": "Baseline: median-scale signed-current asinh. Tuned variants: candidate-specific current/conductance representation, median or numerical 1e-18 scale, and training-only standardization. See selected_target and candidate settings.",
            "mlp_internal_split": "15% of training partition only, used for early stopping; main validation and test rows are excluded.",
            "tuning": "Additional smooth tanh and ReLU MLPs compare training-only transformed-target standardization, smaller asinh scales, and conductance targets. All use identical held-out validation indices.",
            "conductance": "Gd=(Id+(Ig+Ib)/2)/(Vd-Vs), Gs=-(Is+(Ig+Ib)/2)/(Vd-Vs), Ig, Ib; exact inverse on nonzero sampled Vds. At Vds=0 use the continuous symmetric-terminal limit. Four outputs remain independently learned; no KCL projection.",
            "floor_scale_units": "For conductance targets the numerical asinh scale is 1e-18 S for Gd/Gs and 1e-18 A for Ig/Ib; it is a transform hyperparameter, not a physical current floor.",
        },
        "metrics": {
            "relative_error": "abs(ML-SPICE)/max(abs(SPICE),1e-18 A)", "error_floor_A": ERROR_FLOOR_A,
            "mean_accuracy_pct": "100*(1-mean(relative_error)); unbounded below and never clipped",
            "pooled_relative_error": "All four signed terminal-current arrays flattened together; counts/floor usage reported.",
            "raw_r2": "Per-terminal raw-current R², uniformly averaged for the pooled row; sklearn finite convention for constant references.",
            "kcl": "Raw four-current predictions are reported; no KCL closure or clipping in saved model outputs.",
        },
        "held_out_policy": "No gate or circuit data enter training. Validate all candidates; evaluate the selected model once on the untouched test indices. Do not refit on validation rows.",
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in (Path(__file__).resolve(), PROJECT_ROOT / "lusr" / "ml.py")},
        "cases": {},
    }
    for case in args.cases:
        inputs, currents, partitions, provenance = load_case(data_root, case, input_manifest)
        training, validation = partitions["train"], partitions["validation"]
        selected, best_score, candidates = None, float("inf"), {}
        case_seed = args.seed + CASES.index(case) * 100
        case_start = time.monotonic()
        candidate_specs = [(family, "baseline") for family in MODEL_FAMILIES] + [("mlp", variant) for variant in ("tanh_median", "tanh_conductance_floor", "relu_conductance_floor")]
        for family_index, (family, variant) in enumerate(candidate_specs):
            candidate_seed = case_seed + family_index
            candidate_name = f"{family}/{variant}"
            model = make_candidate(family, case, candidate_seed, jobs=args.jobs, smoke=args.smoke, variant=variant)
            print(f"[{case}] fitting {candidate_name} on {len(training)} training rows...", flush=True)
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=args.jobs):
                warnings.simplefilter("always")
                model.fit(inputs[training], currents[training])
                transformed_prediction = model.predict_transformed(inputs[validation])
                prediction = model.predict(inputs[validation])
            transformed_rmse = float(np.sqrt(np.mean(np.square(transformed_prediction - model.transform_currents(currents[validation])))))
            elapsed = time.monotonic() - started
            metrics = regression_metrics(currents[validation], prediction)
            validation_rows.append({"case": case, "model": family, "variant": variant, "selected": False, "rows": len(validation), "transformed_rmse": transformed_rmse, **metrics, **kcl_metrics(prediction)})
            for terminal_index, terminal in enumerate(CURRENT_COLUMNS):
                validation_terminal_rows.append({"case": case, "model": family, "variant": variant, "terminal": terminal, **regression_metrics(currents[validation, terminal_index], prediction[:, terminal_index])})
            candidate_info = {
                "seed": candidate_seed, "hyperparameters": serializable_parameters(model.estimator.get_params(deep=True)),
                "training_seconds": elapsed, "validation_transformed_rmse": transformed_rmse,
                "target_scales_A": model.current_scale.tolist(),
                "representation_scales": model.training_target_scale.tolist(),
                "target_mode": model.target_mode, "target_scale_mode": model.target_scale_mode,
                "standardize_targets": model.standardize_targets,
                "warnings": [str(item.message) for item in captured],
            }
            if hasattr(model.estimator, "n_iter_"):
                candidate_info["iterations_run"] = int(model.estimator.n_iter_)
            candidates[candidate_name] = candidate_info
            print(f"[{case}] {candidate_name}: val transformed RMSE={transformed_rmse:.6g}; median error={metrics['median_relative_error_pct']:.3g}%; p95={metrics['p95_relative_error_pct']:.3g}%; MAPE={metrics['mape_pct']:.3g}%; {elapsed:.1f}s", flush=True)
            selection_score = metrics["mape_pct"] if args.selection == "mape" else transformed_rmse
            if selection_score < best_score:
                selected, best_score = model, selection_score
            # These checkpoints make completed validation work inspectable during long fits.
            write_rows(output / "transistor_validation_baselines.csv", [row for row in validation_rows if row["variant"] == "baseline"])
            write_rows(output / "transistor_validation_models.csv", family_comparison(validation_rows, args.selection))
            write_rows(output / "transistor_validation_candidates.csv", validation_rows)
            write_rows(output / "transistor_validation_terminals.csv", validation_terminal_rows)
        if selected is None:
            raise RuntimeError(f"No fitted model for {case}")
        for row in validation_rows:
            if row["case"] == case:
                row["selected"] = row["model"] == selected.family and row["variant"] == selected.variant
        model_path = model_root / f"{case}.joblib"
        joblib.dump(selected, model_path, compress=3)
        reloaded = joblib.load(model_path)
        with threadpool_limits(limits=args.jobs):
            before = selected.predict(inputs[validation[:16]])
            after = reloaded.predict(inputs[validation[:16]])
            if not np.allclose(before, after, rtol=1e-12, atol=ERROR_FLOOR_A * 1e-6):
                raise ValueError(f"Serialized model failed reload prediction check for {case}")
        manifest["cases"][case] = {
            "provenance": provenance, "candidates": candidates, "selected_family": selected.family,
            "selected_variant": selected.variant, "selection_score": best_score,
            "selected_model": str(model_path), "selected_model_sha256": digest(model_path),
            "reload_validation": "PASS: validation-subset predictions match after reload to rtol=1e-12, atol=1e-24 A",
            "total_seconds": time.monotonic() - case_start,
        }
        (output / "ml_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[{case}] validation selected {selected.family}/{selected.variant}; saved {model_path}; test not evaluated.", flush=True)
    write_rows(output / "transistor_validation_baselines.csv", [row for row in validation_rows if row["variant"] == "baseline"])
    write_rows(output / "transistor_validation_models.csv", family_comparison(validation_rows, args.selection))
    write_rows(output / "transistor_validation_candidates.csv", validation_rows)
    write_rows(output / "transistor_validation_terminals.csv", validation_terminal_rows)
    (output / "ml_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.defer_test:
        evaluate_selected(data_root, output, model_root, args.cases, args.jobs)
    print(f"Completed {len(args.cases)} case(s); results: {output}", flush=True)


if __name__ == "__main__":
    main()
