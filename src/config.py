"""Config loader — single source for paths, feature groups, targets."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def raw_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["raw"]


def processed_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["processed"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["reports"]
    p.mkdir(parents=True, exist_ok=True)
    return p
