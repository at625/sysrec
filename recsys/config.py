"""Config loading. YAML files in configs/ map onto these dataclasses.

Sections a milestone doesn't define a dataclass for stay available as plain dicts in
`Config.sections`, so adding a new config block never breaks older scripts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str
    processed_dir: str
    positive_threshold: float
    val_start: int
    test_start: int
    test_end: int
    user_fraction: float = 1.0


@dataclass
class Config:
    seed: int
    data: DataConfig
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    sections: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    @property
    def processed(self) -> Path:
        return Path(self.data.processed_dir)

    @property
    def artifacts(self) -> Path:
        p = Path(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results(self) -> Path:
        p = Path(self.results_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.sections.get(name, {}))


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config; `overrides` uses dotted keys, e.g. {"data.user_fraction": 0.1}."""
    raw = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    known = {f.name for f in dataclasses.fields(Config)}
    sections = {k: v for k, v in raw.items() if k not in known}
    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        artifacts_dir=raw.get("artifacts_dir", "artifacts"),
        results_dir=raw.get("results_dir", "results"),
        sections=sections,
        path=str(path),
    )


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse CLI `key=value` overrides; values are YAML-parsed (so 0.1 -> float, true -> bool)."""
    out: dict[str, Any] = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out
