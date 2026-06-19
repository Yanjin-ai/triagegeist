"""Phase 2 — metrics harness reused by every model & the fairness audit.

Primary: Quadratic Weighted Kappa (ordinal). Also accuracy, macro-F1, per-class
recall (acuity-1 is rare & safety-critical), and undertriage rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def undertriage_rate(y_true, y_pred, critical=(1, 2)) -> float:
    """Fraction of truly-critical cases assigned a LESS urgent (higher) acuity.

    Higher acuity number = less urgent, so pred > true on a critical case = undertriage.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isin(y_true, critical)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] > y_true[mask]).mean())


def overtriage_rate(y_true, y_pred, low=(4, 5)) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isin(y_true, low)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] < y_true[mask]).mean())


def evaluate(y_true, y_pred, labels=(1, 2, 3, 4, 5)) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "undertriage_critical": undertriage_rate(y_true, y_pred),
        "overtriage_low": overtriage_rate(y_true, y_pred),
        "recall_per_class": {int(l): float(r) for l, r in zip(labels, rec)},
    }


def confusion(y_true, y_pred, labels=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(cm, index=[f"true{l}" for l in labels],
                        columns=[f"pred{l}" for l in labels])


def subgroup_report(df: pd.DataFrame, y_true_col, y_pred_col, group_col) -> pd.DataFrame:
    """Per-subgroup QWK + undertriage rate — the equity audit table (Phase 5)."""
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        m = evaluate(sub[y_true_col], sub[y_pred_col])
        rows.append({group_col: g, "n": len(sub), "qwk": m["qwk"],
                     "accuracy": m["accuracy"],
                     "undertriage_critical": m["undertriage_critical"]})
    return pd.DataFrame(rows).sort_values("undertriage_critical", ascending=False)


def format_report(name: str, m: dict) -> str:
    rpc = " ".join(f"{k}:{v:.3f}" for k, v in m["recall_per_class"].items())
    return (
        f"[{name}] QWK={m['qwk']:.4f} acc={m['accuracy']:.4f} "
        f"macroF1={m['macro_f1']:.4f} undertriage(1,2)={m['undertriage_critical']:.4f} "
        f"overtriage(4,5)={m['overtriage_low']:.4f} | recall/class {rpc}"
    )
