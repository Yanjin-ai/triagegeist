"""Phase 1 — load + join raw tables into stay-level frames.

`python -m src.data.build` writes processed train/val/test parquet to data/processed/.
Join key: patient_id (per-visit unique; train ∩ test = ∅ → stratified split is leakage-safe).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import load_config, processed_dir, raw_dir

SCORED_TARGET = "triage_acuity"
AUX_TARGETS = ["disposition", "ed_los_hours"]
LEAKAGE = ["disposition", "ed_los_hours"]  # never used as features


def _read(cfg, name):
    return pd.read_csv(raw_dir(cfg) / cfg["data"]["files"][name])


def load_joined(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) joined with chief_complaints + patient_history."""
    cfg = cfg or load_config()
    idc = cfg["data"]["id_col"]
    train, test = _read(cfg, "train"), _read(cfg, "test")
    # chief_complaints has chief_complaint_system too (already in train/test) -> keep raw only
    cc = _read(cfg, "chief_complaints")[[idc, "chief_complaint_raw"]]
    hist = _read(cfg, "patient_history")

    def join(df):
        return df.merge(cc, on=idc, how="left").merge(hist, on=idc, how="left")

    return join(train), join(test)


def split(cfg: dict | None = None):
    """Stratified train/val split on the scored target; write parquet."""
    cfg = cfg or load_config()
    train, test = load_joined(cfg)
    tr, va = train_test_split(
        train,
        test_size=cfg["split"]["val_frac"],
        random_state=cfg["split"]["seed"],
        stratify=train[SCORED_TARGET],
    )
    out = processed_dir(cfg)
    for name, df in [("train", tr), ("val", va), ("test", test)]:
        df.reset_index(drop=True).to_parquet(out / f"{name}.parquet")
    print(f"wrote train={len(tr)} val={len(va)} test={len(test)} -> {out}")
    return tr, va, test


if __name__ == "__main__":
    split()
