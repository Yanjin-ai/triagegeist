"""P5 — subgroup fairness audit with bootstrap CIs + protective override rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import quadratic_weighted_kappa, undertriage_rate


def bootstrap_ci(y_true, y_pred, fn, n=400, seed=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = [fn(y_true[s], y_pred[s])
            for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def subgroup_audit(df, y_col, pred_col, group_col, n_boot=400, min_n=30) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        yt, yp = sub[y_col].values, sub[pred_col].values
        n_crit = int(np.isin(yt, (1, 2)).sum())
        ut = undertriage_rate(yt, yp)
        ut_lo, ut_hi = bootstrap_ci(yt, yp, undertriage_rate, n_boot) if len(sub) >= min_n else (np.nan, np.nan)
        qwk = quadratic_weighted_kappa(yt, yp) if len(sub) >= min_n else np.nan
        if n_crit == 0:
            ci = "no acuity-1/2 cases"
        elif ut_lo == ut_lo:  # not NaN
            ci = f"[{ut_lo:.3f}, {ut_hi:.3f}]"
        else:
            ci = "n<min"
        rows.append({
            group_col: g, "n": len(sub), "n_critical": n_crit,
            "qwk": round(qwk, 4) if qwk == qwk else np.nan,
            "undertriage_crit": round(ut, 4) if ut == ut else np.nan,
            "undertri_CI95": ci,
        })
    return pd.DataFrame(rows).sort_values("undertriage_crit", ascending=False)


# ---- protective override: high-risk physiology must not be triaged 4/5 ----
def high_risk_mask(df) -> np.ndarray:
    def col(name, default):
        return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(default, index=df.index)
    risk = (
        (col("news2_score", 0) >= 7)
        | (col("spo2", 100) < 90)
        | (col("gcs_total", 15) <= 13)
        | (col("shock_index", 0) >= 1.0)
    )
    if "mental_status_triage" in df:
        risk = risk | df["mental_status_triage"].isin(
            ["confused", "agitated", "drowsy", "unresponsive"]
        )
    return risk.fillna(False).to_numpy()


def apply_override(df, pred, cap=3) -> np.ndarray:
    """Cap predicted acuity at `cap` (more urgent) for high-risk physiology rows."""
    pred = np.asarray(pred).copy()
    m = high_risk_mask(df)
    pred[m] = np.minimum(pred[m], cap)
    return pred
