"""P5 — error taxonomy for the text-blind structured model.

Categorizes errors and isolates 'physiology-silent severe' cases: truly critical
(acuity 1/2, i.e. the text-encoded severe label) that physiology alone misses.
Also flags the genuinely ambiguous complaint cores (1-vs-2 in the data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEXT_COL = "chief_complaint_raw"


def ambiguous_cores(train: pd.DataFrame, cc: pd.DataFrame, id_col="patient_id",
                    target="triage_acuity") -> set[str]:
    m = train[[id_col, target]].merge(cc[[id_col, TEXT_COL]], on=id_col)
    m["core"] = m[TEXT_COL].str.split(",").str[0].str.strip().str.lower()
    g = m.groupby("core")[target].nunique()
    return set(g[g > 1].index)


def taxonomy(df: pd.DataFrame, y_col, pred_col) -> dict:
    yt, yp = df[y_col].to_numpy(), df[pred_col].to_numpy()
    err = yt != yp
    delta = (yp - yt)[err]
    crit = np.isin(yt, (1, 2))
    return {
        "n": len(df), "n_errors": int(err.sum()),
        "error_rate": round(err.mean(), 4),
        "adjacent_errors": int((np.abs(yp - yt) == 1).sum()),
        "large_errors(>=2)": int((np.abs(yp - yt) >= 2).sum()),
        "undertriage_errors": int((delta > 0).sum()),
        "overtriage_errors": int((delta < 0).sum()),
        "physiology_silent_severe": int((crit & (yp >= 3)).sum()),  # true 1/2 -> pred 3+
    }


def physiology_silent_examples(df, y_col, pred_col, n=8) -> pd.DataFrame:
    crit = np.isin(df[y_col].to_numpy(), (1, 2))
    miss = df[crit & (df[pred_col].to_numpy() >= 3)]
    cols = [c for c in [TEXT_COL, y_col, pred_col, "news2_score", "spo2",
                        "gcs_total", "shock_index", "mental_status_triage"] if c in df]
    return miss[cols].head(n)
