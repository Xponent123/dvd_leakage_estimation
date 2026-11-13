#!/usr/bin/env python3
"""Generate four terminal-current datasets after validating batched ngspice."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lusr.spice import (CASES, CURRENT_COLUMNS, DEFAULT_MODEL, INPUT_COLUMNS,
                        find_ngspice, fresh_deck, sha256, simulate_rows, validate_fresh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", help="ngspice executable; defaults to bundled copy or PATH")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/transistors")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/transistors")
    parser.add_argument("--example-dir", type=Path, default=ROOT / "netlists/transistors")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--limit", type=int, help="Smoke only: first N rows; output under output-dir/smoke_N")
    args = parser.parse_args()
    executable = find_ngspice(args.ngspice)
    output_dir = args.output_dir
    log_dir = args.log_dir
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        output_dir = output_dir / f"smoke_{args.limit}"
        log_dir = log_dir / f"smoke_{args.limit}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, check=True).stdout
    (log_dir / "ngspice_version.txt").write_text(version)
    metadata = {"model": str(args.model.resolve()), "model_sha256": sha256(args.model),
                "ngspice": str(executable), "current_sign": "positive into each FET terminal = negative independent voltage-source branch current",
                "junction_geometry": "assigned model/simulator defaults; no invented AD/AS/PD/PS", "cases": {}}
    started = time.monotonic()

    def progress(case, done, total, cached):
        if done == total or done % 1000 == 0:
            print(f"{case}: {done}/{total} simulated rows ({time.monotonic() - started:.1f} s elapsed)", flush=True)

    for case in args.cases:
        source = args.input_dir / f"{case}_inputs.csv"
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INPUT_COLUMNS:
                raise ValueError(f"Input columns do not match assignment schema in {source}: {reader.fieldnames}")
            rows = [{key: float(value) for key, value in row.items()} for row in reader]
        if args.limit is not None:
            rows = rows[:args.limit]
        print(f"{case}: validating five batch rows against fresh instantiated model cards", flush=True)
        validation = validate_fresh(case, rows, executable, args.model, log_dir / "fresh_validation")
        example_dir = args.example_dir
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / f"{case}.cir").write_text(fresh_deck(case, rows[0], args.model))
        currents, detail = simulate_rows(case, rows, executable, args.model, log_dir / case,
                                        args.chunk_size, args.workers, progress)
        destination = output_dir / f"{case}.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(*INPUT_COLUMNS, *CURRENT_COLUMNS))
            writer.writeheader()
            writer.writerows({**row, **current} for row, current in zip(rows, currents))
        temporary.replace(destination)
        kcl = [abs(sum(current.values())) for current in currents]
        relative_kcl = [residual / max(max(abs(v) for v in current.values()), 1e-30)
                        for residual, current in zip(kcl, currents)]
        detail.update({"input_sha256": sha256(source), "output_sha256": sha256(destination),
                       "fresh_validation": validation, "max_abs_kcl_A": max(kcl),
                       "max_relative_kcl": max(relative_kcl)})
        metadata["cases"][case] = detail
        metadata["elapsed_seconds"] = time.monotonic() - started
        (output_dir / "simulation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{case}: saved {destination}; max |Id+Ig+Is+Ib|={max(kcl):.4g} A", flush=True)
    print(f"Completed {sum(v['row_count'] for v in metadata['cases'].values())} real SPICE samples in {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
