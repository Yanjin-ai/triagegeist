"""P4 — predictive uncertainty: entropy + top1–top2 margin, and breakdown by bin."""
from __future__ import annotations

import numpy as np
import pandas as pd


def entropy(proba) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1)
    return -(p * np.log(p)).sum(1)


def margin(proba) -> np.ndarray:
    s = np.sort(proba, axis=1)
    return s[:, -1] - s[:, -2]


def uncertainty_breakdown(y_true, y_pred, unc, ambiguous=None, n_q=4) -> pd.DataFrame:
    """By uncertainty quartile: error rate, critical-undertriage rate, ambiguous share."""
    y_true, y_pred, unc = (np.asarray(a) for a in (y_true, y_pred, unc))
    q = pd.qcut(unc, n_q, labels=[f"Q{i+1}" for i in range(n_q)], duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": y_pred, "q": q})
    df["err"] = (df.y != df.p).astype(int)
    crit = np.isin(y_true, (1, 2))
    df["undertri"] = ((y_pred > y_true) & crit).astype(int)
    if ambiguous is not None:
        df["amb"] = np.asarray(ambiguous).astype(int)
    rows = []
    for g, sub in df.groupby("q", observed=True):
        row = {"uncertainty_q": g, "n": len(sub),
               "error_rate": round(sub.err.mean(), 4),
               "undertriage_crit": round(sub.undertri.mean(), 4)}
        if ambiguous is not None:
            row["ambiguous_share"] = round(sub.amb.mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)
