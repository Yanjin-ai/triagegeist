"""P4 — decision-boundary tuning for the ESI-4 weak spot.

argmax over (proba * bias) where bias multiplies a target class. Search the bias
that improves acuity-4 recall + QWK WITHOUT hurting critical (1,2) recall.
Interpretable, monotone, and safety-aware.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import evaluate


def apply_bias(proba, classes, bias_vec) -> np.ndarray:
    return classes[(proba * bias_vec).argmax(1)]


def search_class4_bias(proba, y_true, classes, grid=None) -> pd.DataFrame:
    """Grid over a multiplicative boost on acuity-4; report the safety trade-off."""
    grid = np.round(np.arange(1.0, 2.01, 0.1), 2) if grid is None else grid
    j4 = int(np.where(classes == 4)[0][0])
    y_true = np.asarray(y_true)
    rows = []
    for b in grid:
        bias = np.ones(len(classes)); bias[j4] = b
        pred = apply_bias(proba, classes, bias)
        m = evaluate(y_true, pred)
        rows.append({
            "bias_acuity4": b, "QWK": round(m["qwk"], 4), "acc": round(m["accuracy"], 4),
            "recall_acuity4": round(m["recall_per_class"][4], 4),
            "recall_acuity1": round(m["recall_per_class"][1], 4),
            "recall_acuity2": round(m["recall_per_class"][2], 4),
            "undertriage(1,2)": round(m["undertriage_critical"], 4),
            "overtriage(4,5)": round(m["overtriage_low"], 4),
        })
    return pd.DataFrame(rows)
