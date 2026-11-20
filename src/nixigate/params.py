"""Load params.yaml. Every entry needs a value, a unit and a `source` (URL/DOI) or a `design` reason."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

ROOT = Path(__file__).resolve().parents[2]


class MissingSourceError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    value: float
    unit: str
    source: str
    design: str
    note: str


def load_params(path: str | Path = ROOT / "params.yaml") -> MappingProxyType:
    raw = yaml.safe_load(Path(path).read_text())
    out = {}
    for name, d in raw.items():
        src = str(d.get("source", "") or "").strip()
        des = str(d.get("design", "") or "").strip()
        if not (src or des) or "TODO" in (src + des):
            raise MissingSourceError(f"{name}: needs a source URL/DOI or a design reason")
        if src and not src.startswith(("http://", "https://")):
            raise MissingSourceError(f"{name}: source must be a URL or https://doi.org/ link")
        if "value" not in d or "unit" not in d:
            raise MissingSourceError(f"{name}: value and unit are required")
        out[name] = Param(d["value"], d["unit"], src, des, str(d.get("note", "")))
    return MappingProxyType(out)


def v(P, name):
    """Shorthand: the numeric value of a parameter."""
    return P[name].value
