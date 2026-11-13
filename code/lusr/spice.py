"""Auditable single-FET ngspice operating-point simulations for Part A.

All reported currents are positive INTO the MOS terminal. Source/drain junction
areas and perimeters are not fabricated: they use the assigned model/simulator
defaults because the assignment supplies no layout geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_COLUMNS = ("vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj")
CURRENT_COLUMNS = ("id", "ig", "is", "ib")
PROCESS_COLUMNS = ("vth0", "toxe", "ndep", "u0", "xj")
CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT.parent / "models" / "32nm_MGK.pm"
PROTOCOL_VERSION = "single-fet-reset-alter-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_ngspice(explicit: str | None = None) -> Path:
    """Prefer an explicit executable, then the project's portable installation."""
    candidate = explicit or os.environ.get("NGSPICE")
    if candidate:
        found = shutil.which(candidate)
        path = Path(found or candidate).expanduser().resolve()
    else:
        local = ROOT / "tools/local/usr/bin/ngspice"
        found = shutil.which("ngspice")
        path = local if local.exists() else Path(found or "ngspice")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError("ngspice executable missing; set NGSPICE or --ngspice")
    return path.resolve()


def _number(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"Nonfinite SPICE input: {value}")
    return format(float(value), ".17g")


def _validate(case: str, rows: list[dict[str, float]]) -> None:
    if case not in CASES:
        raise ValueError(f"Unknown transistor case {case}")
    if not rows:
        raise ValueError("No transistor input rows")
    pmos = case.startswith("pmos")
    for index, row in enumerate(rows):
        for key in INPUT_COLUMNS:
            _number(row[key])
        expected_gate = row["vdd"] if case in ("nmos_on", "pmos_off") else 0.0
        expected_body = row["vdd"] if pmos else 0.0
        if row["vg"] != expected_gate or row["vb"] != expected_body:
            raise ValueError(f"Incorrect fixed gate/body voltages at {case} row {index}")
        if not (0 <= row["vd"] <= row["vdd"] and 0 <= row["vs"] <= row["vdd"]):
            raise ValueError(f"Terminal voltage outside [0,VDD] at {case} row {index}")
        if row["w"] <= 0:
            raise ValueError(f"Nonpositive width at {case} row {index}")


def _circuit(case: str, row: dict[str, float], model: str) -> str:
    kind = case.split("_")[0]
    return "\n".join([
        f"LUSR 32nm_MGK {case}; currents positive into FET; L=32nm",
        model, OPTIONS, f".temp {_number(row['temp'])}",
        *[f"V{terminal} {terminal} 0 {_number(row['v' + terminal])}" for terminal in "dgsb"],
        f"M1 d g s b {kind} L=32n W={_number(row['w'])}",
    ])


def _measure(index: int) -> list[str]:
    return ["op", f"echo LUSR_ROW {index}",
            "let id = -i(Vd)", "let ig = -i(Vg)",
            "let is = -i(Vs)", "let ib = -i(Vb)",
            "print id ig is ib", "destroy all"]


def batch_deck(case: str, rows: list[dict[str, float]], model_path: Path) -> str:
    """Reset before each row to clear BSIM size/temperature parameter caches."""
    _validate(case, rows)
    kind = case.split("_")[0]
    lines = [_circuit(case, rows[0], f'.include "{model_path.resolve()}"'),
             ".control", "set numdgt=17", "set noaskquit"]
    for index, row in enumerate(rows):
        lines.append("reset")
        lines.extend(f"altermod {kind} {key} = {_number(row[key])}" for key in PROCESS_COLUMNS)
        lines.append(f"alter M1 w = {_number(row['w'])}")
        lines.extend(f"alter V{terminal} dc = {_number(row['v' + terminal])}" for terminal in "dgsb")
        lines.append(f"set temp = {_number(row['temp'])}")
        lines.extend(_measure(index))
    return "\n".join([*lines, "quit", ".endc", ".end", ""])


def fresh_deck(case: str, row: dict[str, float], model_path: Path = DEFAULT_MODEL) -> str:
    """Freshly instantiate a model card: independent reference for altermod."""
    _validate(case, [row])
    kind = case.split("_")[0]
    text = model_path.read_text()
    blocks = re.split(r"(?im)(?=^\s*\.model\s)", text)
    selected = 0
    for index, block in enumerate(blocks):
        if not re.match(rf"(?i)\s*\.model\s+{kind}\s+{kind}\b", block):
            continue
        selected += 1
        for key in PROCESS_COLUMNS:
            pattern = rf"(?i)(\b{key}\s*=\s*)[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
            block, count = re.subn(pattern, lambda m: m.group(1) + _number(row[key]), block)
            if count != 1:
                raise ValueError(f"Expected one {kind} {key} in model, found {count}")
        blocks[index] = block
    if selected != 1:
        raise ValueError(f"Expected one {kind} model card, found {selected}")
    circuit = _circuit(case, row, "* Embedded assigned model; only five specified process parameters replaced\n" + "".join(blocks))
    return "\n".join([circuit, ".control", "set numdgt=17", *_measure(0), "quit", ".endc", ".end", ""])


def _parse_output(output: str, expected_rows: int) -> list[dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    current = None
    for line in output.splitlines():
        match = re.fullmatch(r"\s*LUSR_ROW\s+(\d+)\s*", line)
        if match:
            current = int(match.group(1))
            if current in records:
                raise RuntimeError(f"Duplicate simulation row {current}")
            records[current] = {}
            continue
        match = re.fullmatch(r"\s*(id|ig|is|ib)\s*=\s*(\S+)\s*", line)
        if match and current is not None:
            key, number = match.groups()
            value = float(number)
            if key in records[current] or not math.isfinite(value):
                raise RuntimeError(f"Duplicate/nonfinite current at simulation row {current}: {line}")
            records[current][key] = value
    if set(records) != set(range(expected_rows)):
        raise RuntimeError(f"Missing/excess simulation rows: expected {expected_rows}, found {len(records)}")
    for index, record in records.items():
        if set(record) != set(CURRENT_COLUMNS):
            raise RuntimeError(f"Incomplete terminal currents at simulation row {index}: {record}")
    return [records[index] for index in range(expected_rows)]


def run_deck(deck: str, path: Path, executable: Path, expected_rows: int) -> tuple[list[dict[str, float]], int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deck)
    proc = subprocess.run([str(executable), "-n", "-b", str(path.resolve())], cwd=path.parent,
                          capture_output=True, text=True, timeout=max(120, expected_rows * 3))
    output = proc.stdout + "\nSTDERR:\n" + proc.stderr
    path.with_suffix(".log").write_text(output)
    # ngspice can return success even when an interactive control command failed.
    failure = re.search(r"(?im)^\s*(?:error\b|fatal\b|fatal error\b)|simulation\(s\) aborted|timestep too small|operating point failed", output)
    if proc.returncode != 0 or failure:
        raise RuntimeError(f"ngspice failed (status {proc.returncode}); inspect {path.with_suffix('.log')}")
    results = _parse_output(proc.stdout, expected_rows)
    warning_count = len(re.findall(r"(?im)^\s*warning\b", output))
    return results, warning_count


def validate_fresh(case: str, rows: list[dict[str, float]], executable: Path,
                   model_path: Path, directory: Path) -> dict:
    """Check five spread-out input rows against separately instantiated models."""
    indices = sorted({round(i * (len(rows) - 1) / 4) for i in range(5)})
    subset = [rows[i] for i in indices]
    values, warnings = run_deck(batch_deck(case, subset, model_path), directory / f"{case}_batch.cir", executable, len(subset))
    errors = []
    for local, (source_index, row) in enumerate(zip(indices, subset)):
        reference, _ = run_deck(fresh_deck(case, row, model_path), directory / f"{case}_fresh_{source_index:05d}.cir", executable, 1)
        for key in CURRENT_COLUMNS:
            expected = reference[0][key]
            actual = values[local][key]
            error = abs(actual - expected)
            # Sparse matrix cancellation can shift tiny branch currents by
            # a few 1e-16 A even when the model and biases are identical.
            tolerance = 2e-15 + 2e-6 * abs(expected)
            if error > tolerance:
                raise RuntimeError(f"Batch/fresh mismatch for {case} row {source_index} {key}: {actual} versus {expected}; inspect {directory}")
            errors.append({"input_row": source_index, "terminal": key, "batch": actual,
                           "fresh": expected, "absolute_error_A": error, "tolerance_A": tolerance})
    report = {"case": case, "passed": True, "checked_input_rows": indices,
              "batch_warning_count": warnings, "comparisons": errors}
    (directory / f"{case}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def simulate_rows(case: str, rows: list[dict[str, float]], executable: Path,
                  model_path: Path = DEFAULT_MODEL, directory: Path | None = None,
                  chunk_size: int = 250, workers: int = 4, progress=None) -> tuple[list[dict[str, float]], dict]:
    """Parallel external ngspice batches; successful exact-input chunks resume."""
    _validate(case, rows)
    directory = directory or ROOT / "logs/transistors" / case
    directory.mkdir(parents=True, exist_ok=True)
    if chunk_size <= 0 or workers <= 0:
        raise ValueError("chunk_size and workers must be positive")
    identity = {"protocol": PROTOCOL_VERSION, "case": case, "model_sha256": sha256(model_path),
                "executable_sha256": sha256(executable), "options": OPTIONS}
    results: list[dict[str, float] | None] = [None] * len(rows)
    tasks = [(start, rows[start:start + chunk_size]) for start in range(0, len(rows), chunk_size)]

    def run_chunk(task):
        start, subset = task
        key = hashlib.sha256(json.dumps({**identity, "rows": subset}, sort_keys=True).encode()).hexdigest()
        path = directory / f"chunk_{start:05d}.cir"
        cache = path.with_suffix(".json")
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("key") == key and len(saved.get("results", [])) == len(subset):
                if all(set(r) == set(CURRENT_COLUMNS) and all(math.isfinite(x) for x in r.values()) for r in saved["results"]):
                    return start, saved["results"], saved.get("warning_count", 0), True
        values, warnings = run_deck(batch_deck(case, subset, model_path), path, executable, len(subset))
        temporary = cache.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"key": key, "warning_count": warnings, "results": values}, indent=2) + "\n")
        temporary.replace(cache)
        return start, values, warnings, False

    warning_count = cached_chunks = completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_chunk, task) for task in tasks]
        for future in as_completed(futures):
            start, values, warnings, cached = future.result()
            results[start:start + len(values)] = values
            warning_count += warnings
            cached_chunks += int(cached)
            completed += len(values)
            if progress:
                progress(case, completed, len(rows), cached)
    if any(value is None for value in results):
        raise RuntimeError(f"Incomplete assembled simulation output for {case}")
    return results, {**identity, "row_count": len(rows), "chunk_count": len(tasks),
                     "warning_count": warning_count, "cached_chunks": cached_chunks}
