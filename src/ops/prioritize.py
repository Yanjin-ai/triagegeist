"""P6 — transparent queue prioritization (no RL; interpretable by design).

priority = w_urgency*urgency + w_crit*P(critical) + w_unc*uncertainty_flag + w_wait*wait
urgency = (5 - acuity)/4 ∈ [0,1].  Higher priority = seen sooner.

Two named policies expose the safety/throughput trade-off. A "needs senior review"
flag floats high-uncertainty potentially-critical cases regardless of score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = {
    # safety-first: weight criticality + uncertainty heavily
    "safety_first": dict(w_urgency=0.45, w_crit=0.35, w_unc=0.12, w_wait=0.08),
    # throughput: lean on acuity, let waits age cases up faster
    "throughput": dict(w_urgency=0.55, w_crit=0.15, w_unc=0.05, w_wait=0.25),
}


def _norm_entropy(entropy) -> np.ndarray:
    e = np.asarray(entropy, dtype=float)
    return e / np.log(5)  # max entropy for 5 classes


def priority_score(acuity, p_critical, entropy, wait_hours, policy="safety_first") -> np.ndarray:
    w = POLICIES[policy]
    urgency = (5 - np.asarray(acuity)) / 4
    unc = _norm_entropy(entropy)
    wait = np.clip(np.asarray(wait_hours, dtype=float) / 6.0, 0, 1)  # saturates at 6h
    return (w["w_urgency"] * urgency + w["w_crit"] * np.asarray(p_critical)
            + w["w_unc"] * unc + w["w_wait"] * wait)


def needs_senior_review(acuity, p_critical, entropy, ent_hi=None) -> np.ndarray:
    """High uncertainty AND plausibly critical → human/senior review queue."""
    entropy = np.asarray(entropy)
    ent_hi = np.nanpercentile(entropy, 75) if ent_hi is None else ent_hi
    return ((entropy >= ent_hi) & ((np.asarray(p_critical) >= 0.2) | (np.asarray(acuity) <= 2)))


def rank_queue(df: pd.DataFrame, policy="safety_first") -> pd.DataFrame:
    out = df.copy()
    out["priority"] = priority_score(
        out["acuity"], out["p_critical"], out["entropy"],
        out.get("wait_hours", 0.0), policy,
    )
    out["senior_review"] = needs_senior_review(out["acuity"], out["p_critical"], out["entropy"])
    return out.sort_values("priority", ascending=False).reset_index(drop=True)
