"""Assignment-defined random inputs. No currents are estimated in this module."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .model_card import PROCESS_PARAMETERS


DEFAULT_SEED = 20261022  # Computational reproducibility choice, not a physical parameter.
DEFAULT_SAMPLES = 10_000
DEFAULT_CORNERS = 200
NOMINAL_VDD = 0.9
CHANNEL_LENGTH = 32e-9
PROCESS_VARIATION = 0.1
TEMPERATURE_RANGE = (-40.0, 125.0)
CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
INPUT_COLUMNS = ("vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROCESS_PARAMETERS)
CORNER_COLUMNS = (
    "corner_id", "vdd", "temp",
    *(f"n_{parameter}" for parameter in PROCESS_PARAMETERS),
    *(f"p_{parameter}" for parameter in PROCESS_PARAMETERS),
)
SPLIT_NAMES = ("train", "validation", "test")


def relative_bounds(nominal: float) -> tuple[float, float]:
    """Nominal ±10%, with ordered endpoints for negative PMOS thresholds."""
    return tuple(sorted((nominal * (1 - PROCESS_VARIATION), nominal * (1 + PROCESS_VARIATION))))


def _positive_count(count: int) -> None:
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count <= 0:
        raise ValueError("Sample count must be a positive integer")


def sample_transistors(
    case: str,
    count: int,
    nominal: Mapping[str, Mapping[str, float]],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw independent inputs, with Vd/Vs conditional on the sampled VDD.

    ON/OFF labels follow the brief's gate/body bias table; they do not classify
    conduction from the resulting source voltage or sampled threshold voltage.
    Length is fixed at CHANNEL_LENGTH and recorded in the run manifest, because
    the assignment's required dataset does not have a length column.
    """
    _positive_count(count)
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; expected one of {CASES}")
    kind, state = case.split("_")
    vdd = rng.uniform(*relative_bounds(NOMINAL_VDD), size=count)
    sampled = {
        "vdd": vdd,
        "temp": rng.uniform(*TEMPERATURE_RANGE, size=count),
        "w": rng.uniform(CHANNEL_LENGTH, 8 * CHANNEL_LENGTH, size=count),
        "vd": rng.uniform(0.0, vdd, size=count),
        "vs": rng.uniform(0.0, vdd, size=count),
        "vg": vdd.copy() if (kind == "nmos") == (state == "on") else np.zeros(count),
        "vb": np.zeros(count) if kind == "nmos" else vdd.copy(),
    }
    for parameter in PROCESS_PARAMETERS:
        sampled[parameter] = rng.uniform(*relative_bounds(nominal[kind][parameter]), size=count)
    validate_transistors(case, sampled, nominal)
    return sampled


def sample_corners(
    count: int,
    nominal: Mapping[str, Mapping[str, float]],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw common evaluation corners with separate NMOS/PMOS process values.

    Each corner is shared by every gate input case and the complete parity
    circuit. Widths are a circuit-design choice and are deliberately absent.
    Evaluation corners must never be added to transistor training datasets.
    """
    _positive_count(count)
    corners = {
        "corner_id": np.arange(count, dtype=np.int64),
        "vdd": rng.uniform(*relative_bounds(NOMINAL_VDD), size=count),
        "temp": rng.uniform(*TEMPERATURE_RANGE, size=count),
    }
    for kind, prefix in (("nmos", "n"), ("pmos", "p")):
        for parameter in PROCESS_PARAMETERS:
            corners[f"{prefix}_{parameter}"] = rng.uniform(
                *relative_bounds(nominal[kind][parameter]), size=count
            )
    validate_corners(corners, nominal)
    return corners


def split_indices(count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Persistable 70/15/15 partition; rounding remainder goes to the test set."""
    _positive_count(count)
    order = rng.permutation(count)
    ntrain, nvalidation = count * 70 // 100, count * 15 // 100
    partitions = {
        "train": order[:ntrain],
        "validation": order[ntrain:ntrain + nvalidation],
        "test": order[ntrain + nvalidation:],
    }
    joined = np.concatenate(tuple(partitions.values()))
    if not np.array_equal(np.sort(joined), np.arange(count)):
        raise ValueError("Split indices must be disjoint and cover every input row exactly once")
    return partitions


def _validate_columns(data: Mapping[str, np.ndarray], columns: tuple[str, ...]) -> int:
    if tuple(data) != columns:
        raise ValueError(f"Expected columns {columns}, got {tuple(data)}")
    count = len(data[columns[0]])
    _positive_count(count)
    for name, values in data.items():
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid shape or non-finite values in {name}")
    return count


def _check_range(name: str, values: np.ndarray, low: float | np.ndarray, high: float | np.ndarray) -> None:
    if np.any(values < low) or np.any(values > high):
        raise ValueError(f"Values outside assigned range in {name}")


def validate_transistors(
    case: str,
    data: Mapping[str, np.ndarray],
    nominal: Mapping[str, Mapping[str, float]],
) -> None:
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    count = _validate_columns(data, INPUT_COLUMNS)
    kind, state = case.split("_")
    _check_range("vdd", data["vdd"], *relative_bounds(NOMINAL_VDD))
    _check_range("temp", data["temp"], *TEMPERATURE_RANGE)
    _check_range("w", data["w"], CHANNEL_LENGTH, 8 * CHANNEL_LENGTH)
    for terminal in ("vd", "vs"):
        _check_range(terminal, data[terminal], 0.0, data["vdd"])
    gate_high = (kind == "nmos") == (state == "on")
    expected_gate = data["vdd"] if gate_high else np.zeros(count)
    expected_body = data["vdd"] if kind == "pmos" else np.zeros(count)
    if not np.array_equal(data["vg"], expected_gate) or not np.array_equal(data["vb"], expected_body):
        raise ValueError(f"Incorrect gate or body bias for {case}")
    for parameter in PROCESS_PARAMETERS:
        _check_range(parameter, data[parameter], *relative_bounds(nominal[kind][parameter]))


def validate_corners(
    data: Mapping[str, np.ndarray], nominal: Mapping[str, Mapping[str, float]]
) -> None:
    count = _validate_columns(data, CORNER_COLUMNS)
    if not np.array_equal(data["corner_id"], np.arange(count)):
        raise ValueError("Corner IDs must be consecutive and zero based")
    _check_range("vdd", data["vdd"], *relative_bounds(NOMINAL_VDD))
    _check_range("temp", data["temp"], *TEMPERATURE_RANGE)
    for kind, prefix in (("nmos", "n"), ("pmos", "p")):
        for parameter in PROCESS_PARAMETERS:
            name = f"{prefix}_{parameter}"
            _check_range(name, data[name], *relative_bounds(nominal[kind][parameter]))
