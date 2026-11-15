#!/usr/bin/env python3
"""Prepare sampled inputs and partitions. This script does not run SPICE.

Run from any working directory, for example:
    python code/scripts/01_prepare_inputs.py
    python code/scripts/01_prepare_inputs.py --samples 20 --corners 3 --output /tmp/lusr-smoke

All CSVs here contain input parameters only. The four required terminal-current
datasets must be produced later from actual single-transistor SPICE simulations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lusr.model_card import PROCESS_PARAMETERS, read_model_card
from lusr.sampling import (
    CASES, CHANNEL_LENGTH, CORNER_COLUMNS, DEFAULT_CORNERS, DEFAULT_SAMPLES,
    DEFAULT_SEED, INPUT_COLUMNS, NOMINAL_VDD, SPLIT_NAMES, TEMPERATURE_RANGE,
    relative_bounds, sample_corners, sample_transistors, split_indices,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, data: dict[str, np.ndarray], columns: tuple[str, ...]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for values in zip(*(data[name] for name in columns), strict=True):
            writer.writerow(format(float(value), ".17g") for value in values)
    temporary.replace(path)
    return sha256(path)


def write_splits(path: Path, partitions: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("row_index", "split"))
        for name in SPLIT_NAMES:
            writer.writerows((int(index), name) for index in partitions[name])
    temporary.replace(path)
    return sha256(path)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT.parent / "models" / "32nm_MGK.pm")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--samples", type=positive_int, default=DEFAULT_SAMPLES, help="rows per transistor case (default: 10000)")
    parser.add_argument("--corners", type=positive_int, default=DEFAULT_CORNERS, help="common evaluation corners (default: 200)")
    parser.add_argument("--seed", type=nonnegative_int, default=20261024, help="explicit master seed (default: 20261024, the seed used for the submitted data)")
    args = parser.parse_args()
    model = read_model_card(args.model)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Separate stable streams prevent sample-count changes from changing corners
    # and prevent input-draw counts from changing partition draws.
    stream_names = tuple(f"samples/{case}" for case in CASES) + tuple(f"split/{case}" for case in CASES) + ("corners",)
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(stream_names))
    streams = {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(stream_names, child_seeds, strict=True)
    }
    datasets = {}
    for case in CASES:
        data = sample_transistors(case, args.samples, model.nominal, streams[f"samples/{case}"])
        partitions = split_indices(args.samples, streams[f"split/{case}"])
        input_path = output / "inputs" / f"{case}_inputs.csv"
        split_path = output / "splits" / f"{case}_split.csv"
        datasets[case] = {
            "status": "NOT_SIMULATED",
            "rows": args.samples,
            "input_csv": str(input_path.relative_to(output)),
            "input_sha256": write_csv(input_path, data, INPUT_COLUMNS),
            "split_csv": str(split_path.relative_to(output)),
            "split_sha256": write_splits(split_path, partitions),
            "split_counts": {name: len(values) for name, values in partitions.items()},
        }
    corner_path = output / "corners.csv"
    corners = sample_corners(args.corners, model.nominal, streams["corners"])
    corners_hash = write_csv(corner_path, corners, CORNER_COLUMNS)

    manifest = {
        "schema_version": 1,
        "status": "INPUTS_PREPARED_NOT_SIMULATED",
        "currents": "No id/ig/is/ib values exist in these input files; actual SPICE simulation is required.",
        "assignment": {
            "team": "LUSR", "assigned_model": "32nm_MGK", "nominal_vdd_V": NOMINAL_VDD,
            "channel_length_m": CHANNEL_LENGTH, "test_circuit": "8-bit Parity Checker",
        },
        "model": model.as_manifest(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "random": {
            "master_seed": args.seed, "seed_selection": "Explicit computational reproducibility configuration, not a physical assumption.",
            "generator": "numpy.random.Generator(PCG64(SeedSequence child))",
            "streams": {name: list(child.spawn_key) for name, child in zip(stream_names, child_seeds, strict=True)},
            "independence": "Separate streams per case, split, and evaluation corners; independent uniforms, with vd/vs conditional on vdd and vg/vb fixed by case.",
        },
        "ranges": {
            "vdd_V": relative_bounds(NOMINAL_VDD), "temp_C": TEMPERATURE_RANGE,
            "w_m": [CHANNEL_LENGTH, 8 * CHANNEL_LENGTH], "vd_V": [0, "vdd"], "vs_V": [0, "vdd"],
            "process": {
                kind: {parameter: relative_bounds(model.nominal[kind][parameter]) for parameter in PROCESS_PARAMETERS}
                for kind in ("nmos", "pmos")
            },
        },
        "fixed_bias": {
            "nmos_on": {"vg": "vdd", "vb": 0}, "nmos_off": {"vg": 0, "vb": 0},
            "pmos_on": {"vg": 0, "vb": "vdd"}, "pmos_off": {"vg": "vdd", "vb": "vdd"},
        },
        "input_columns": INPUT_COLUMNS,
        "input_units": {"vdd": "V", "temp": "degC", "w": "m", "vd": "V", "vs": "V", "vg": "V", "vb": "V", "vth0": "model-card units", "toxe": "model-card units", "ndep": "model-card units", "u0": "model-card units", "xj": "model-card units"},
        "split_policy": {"fractions": [0.70, 0.15, 0.15], "names": SPLIT_NAMES, "rounding": "floor(n*70/100), floor(n*15/100), remainder to test", "indices": "zero-based data-row positions, excluding the CSV header"},
        "datasets": datasets,
        "evaluation_corners": {
            "status": "NOT_SIMULATED", "rows": args.corners, "csv": "corners.csv", "sha256": corners_hash,
            "columns": CORNER_COLUMNS, "use": "Common corners for gate and complete-circuit evaluation only; never for training.",
            "process_sharing": "All NMOS share one process draw; all PMOS share a separate draw within a corner.",
            "gate_widths": "Not specified; circuit widths require a user decision before gate/circuit simulation.",
        },
        "validation": "All rows checked for finite values, required ranges, fixed gate/body bias, and disjoint exhaustive partition indices.",
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in (Path(__file__).resolve(), PROJECT_ROOT / "lusr" / "sampling.py", PROJECT_ROOT / "lusr" / "model_card.py")
        },
    }
    manifest_path = output / "input_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Prepared {len(CASES)} x {args.samples} input rows and {args.corners} evaluation corners.")
    print(f"Seed: {args.seed}; channel length: {CHANNEL_LENGTH:g} m; nominal VDD: {NOMINAL_VDD:g} V.")
    print("Status: NOT SIMULATED. No terminal-current or accuracy results were generated.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
