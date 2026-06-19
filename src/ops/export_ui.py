"""P8 — export a self-contained data feed for the static triage-copilot UI.

Reads data/processed/ops_table.parquet, samples a realistic waiting-room snapshot,
computes ED-level aggregates over the full cohort, and writes app/data.js
(`window.TRIAGE_DATA = {...}`) so app/index.html opens directly via file:// (no server).

    python -m src.ops.export_ui
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import ROOT, load_config, processed_dir
from ..ops.prioritize import POLICIES

SNAPSHOT_N = 120
N_BEDS = 40
BED_BUCKETS = {"resuscitation/immediate bed", "high-frequency monitoring", "standard bed"}


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def main():
    cfg = load_config()
    df = pd.read_parquet(processed_dir(cfg) / "ops_table.parquet")

    # realistic waiting-room snapshot (stratified-ish by acuity via random sample)
    snap = df.sample(SNAPSHOT_N, random_state=7).reset_index(drop=True)
    ent_hi = float(np.nanpercentile(snap["entropy"], 75))

    patients = [{k: _clean(v) for k, v in row.items()} for _, row in snap.iterrows()]

    # full-cohort aggregates (flow), and snapshot census (point-in-time)
    bucket_counts = df["bucket"].value_counts().to_dict()
    snap_bed_demand = int(snap["bucket"].isin(BED_BUCKETS).sum())
    shift_flow = []
    if "shift" in df:
        for sh, sub in df.groupby("shift", observed=True):
            shift_flow.append({
                "shift": sh, "n": int(len(sub)),
                "expected_admissions": round(float(sub["p_admit"].sum()), 1),
                "admission_rate": round(float(sub["p_admit"].mean()), 3),
                "senior_review": int(sub["senior_review"].sum()),
                "mean_los": round(float(sub["pred_los"].mean()), 2),
            })

    data = {
        "meta": {
            "model": "model_structured_v1 (text-blind)",
            "cohort_n": int(len(df)),
            "snapshot_n": SNAPSHOT_N,
            "n_beds": N_BEDS,
            "ent_hi": round(ent_hi, 4),
            "ln5": round(float(np.log(5)), 6),
        },
        "policies": POLICIES,
        "patients": patients,
        "cohort": {
            "expected_admissions": round(float(df["p_admit"].sum()), 0),
            "admission_rate": round(float(df["p_admit"].mean()), 3),
            "mean_los": round(float(df["pred_los"].mean()), 2),
            "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
            "snapshot_bed_demand": snap_bed_demand,
            "snapshot_bed_pressure": round(snap_bed_demand / N_BEDS, 2),
            "shift_flow": shift_flow,
        },
    }

    app_dir = ROOT / "app"
    app_dir.mkdir(exist_ok=True)
    payload = "window.TRIAGE_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
    (app_dir / "data.js").write_text(payload)
    # also a plain json for notebooks / other consumers
    (app_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"wrote app/data.js + app/data.json ({len(patients)} patients, "
          f"cohort {data['meta']['cohort_n']:,}; bed pressure "
          f"{data['cohort']['snapshot_bed_pressure']})")


if __name__ == "__main__":
    main()
