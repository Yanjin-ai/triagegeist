"""P6 — ED-level aggregates from per-patient ops predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ed_load_summary(df: pd.DataFrame, n_beds: int = 30) -> dict:
    """df columns: acuity, p_critical, p_admit, pred_los, bucket, entropy, senior_review."""
    n = len(df)
    pred_admits = float(df["p_admit"].sum())
    occupied = float((df["bucket"].isin(
        ["resuscitation/immediate bed", "high-frequency monitoring", "standard bed"]).sum()))
    return {
        "n_patients": n,
        "acuity_counts": df["acuity"].value_counts().sort_index().to_dict(),
        "bucket_counts": df["bucket"].value_counts().to_dict(),
        "expected_admissions": round(pred_admits, 1),
        "admission_rate": round(pred_admits / n, 3) if n else 0.0,
        "bed_demand": int(occupied),
        "bed_pressure": round(occupied / n_beds, 2),
        "mean_pred_los_h": round(float(df["pred_los"].mean()), 2),
        "high_uncertainty_critical": int(df["senior_review"].sum()),
    }


def scenario(df: pd.DataFrame, mask: np.ndarray, name: str, n_beds: int = 30) -> dict:
    s = ed_load_summary(df[mask], n_beds=n_beds)
    s["scenario"] = name
    return s
