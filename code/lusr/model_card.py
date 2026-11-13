"""Read the assigned BSIM model without altering the supplied model card."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re


PROCESS_PARAMETERS = ("vth0", "toxe", "ndep", "u0", "xj")
_MODEL_HEADER = re.compile(
    r"^[ \t]*\.model\s+(\S+)\s+(nmos|pmos)\b", re.IGNORECASE | re.MULTILINE
)
_NUMBER = re.compile(
    r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)"
    r"(?P<suffix>meg|[tgkmunpf])?",
    re.IGNORECASE,
)
_SCALE = {
    "": 1.0, "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}


@dataclass(frozen=True)
class ModelCard:
    """Model provenance and the five explicitly swept nominal parameters."""

    path: Path
    sha256: str
    model_names: dict[str, str]
    nominal: dict[str, dict[str, float]]

    def as_manifest(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "model_names": self.model_names,
            "nominal_parameters": self.nominal,
            "parameter_units": "Unmodified BSIM model-card units; no unit conversion.",
        }


def parse_spice_number(token: str) -> float:
    """Parse a literal SPICE number; reject expressions or ambiguous trailing text."""
    match = _NUMBER.fullmatch(token)
    if match is None:
        raise ValueError(f"Unsupported SPICE numeric literal: {token!r}")
    number = float(match.group("number").replace("D", "e").replace("d", "e"))
    value = number * _SCALE[(match.group("suffix") or "").lower()]
    if not math.isfinite(value):
        raise ValueError(f"Non-finite SPICE value: {token!r}")
    return value


def read_model_card(path: str | Path) -> ModelCard:
    """Extract exactly one NMOS and one PMOS model and retain the source SHA-256.

    Matching is case insensitive and parameter names have explicit boundaries,
    so (for example) `u0` cannot accidentally match a different parameter. The
    original file is read only; its exact bytes determine the provenance hash.
    """
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    text = "\n".join(
        re.split(r"[;$]", line, maxsplit=1)[0]
        for line in lines
        if not line.lstrip().startswith("*")
    )
    headers = list(_MODEL_HEADER.finditer(text))
    # All .model declarations delimit a block, including any non-MOS model.
    boundaries = list(re.finditer(r"^[ \t]*\.model\b", text, re.IGNORECASE | re.MULTILINE))
    names: dict[str, str] = {}
    nominal: dict[str, dict[str, float]] = {}
    for header in headers:
        name, kind = header.group(1), header.group(2).lower()
        if kind in nominal:
            raise ValueError(f"Expected one {kind} model; found another named {name!r}")
        end = next((item.start() for item in boundaries if item.start() > header.start()), len(text))
        block = text[header.end():end]
        values: dict[str, float] = {}
        for parameter in PROCESS_PARAMETERS:
            pattern = rf"(?<![\w]){re.escape(parameter)}\s*=\s*([^\s()]+)"
            matches = re.findall(pattern, block, flags=re.IGNORECASE)
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {parameter} in {kind} model {name!r}; "
                    f"found {len(matches)}"
                )
            value = parse_spice_number(matches[0])
            if parameter != "vth0" and value <= 0:
                raise ValueError(f"Expected positive {kind} {parameter}; got {value}")
            values[parameter] = value
        names[kind] = name
        nominal[kind] = values
    if set(nominal) != {"nmos", "pmos"}:
        raise ValueError(f"Model file must contain NMOS and PMOS models; found {sorted(nominal)}")
    return ModelCard(source, hashlib.sha256(raw).hexdigest(), names, nominal)
