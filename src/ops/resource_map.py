"""P6 — map predictions to interpretable operational buckets.

Inputs per patient: predicted acuity, P(critical)=p1+p2, P(admit), predicted LOS.
Rules are deliberately transparent (a clinician can read them). Honesty note: this
mapping is clinically meaningful but, like the protective override (P5), reflects
physiology/outcome logic that may diverge from the text-determined acuity labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = [
    "resuscitation/immediate bed",  # 0
    "high-frequency monitoring",    # 1
    "standard bed",                 # 2
    "fast-track/minor care",        # 3
    "likely discharge",             # 4
]


def assign_bucket(acuity, p_admit, pred_los) -> pd.Series:
    acuity = np.asarray(acuity)
    p_admit = np.asarray(p_admit)
    pred_los = np.asarray(pred_los)
    out = np.empty(len(acuity), dtype=object)

    out[acuity == 1] = BUCKETS[0]
    out[acuity == 2] = BUCKETS[1]
    out[acuity == 3] = BUCKETS[2]
    out[acuity == 4] = BUCKETS[3]
    out[acuity == 5] = BUCKETS[4]

    # escalate acuity-3 with high admission risk to monitoring
    out[(acuity == 3) & (p_admit >= 0.5)] = BUCKETS[1]
    # de-escalate acuity-4 with very low admit risk + short stay to discharge lane
    out[(acuity == 4) & (p_admit < 0.15) & (pred_los < 2.0)] = BUCKETS[4]
    return pd.Series(out)
