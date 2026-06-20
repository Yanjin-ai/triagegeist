"""SOTA uncertainty — split-conformal prediction (Adaptive Prediction Sets, APS).

Replaces ad-hoc entropy thresholding with a *distribution-free coverage guarantee*:
for a chosen miscoverage alpha, the returned acuity set contains the true acuity with
probability >= 1 - alpha (marginal). Singletons = confident auto-triage; multi-class
sets = ambiguous -> principled human deferral (cf. conformal cost-aware triage,
Nature Sci Rep 2026). Calibrated on a held-out split; reproducible (deterministic APS).
"""
from __future__ import annotations

import numpy as np

CLASSES = np.array([1, 2, 3, 4, 5])


class APSConformal:
    def __init__(self, alpha: float = 0.10, classes=CLASSES):
        self.alpha = alpha
        self.classes = np.asarray(classes)
        self.qhat_: float | None = None

    def _aps_scores_true(self, P, y):
        """APS nonconformity per row: cumulative prob from the top down to the true class."""
        order = np.argsort(-P, axis=1)
        sorted_p = np.take_along_axis(P, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        y_idx = np.array([np.where(self.classes == t)[0][0] for t in y])
        rank_true = (order == y_idx[:, None]).argmax(1)
        return cum[np.arange(len(y)), rank_true]

    def fit(self, P_cal, y_cal) -> "APSConformal":
        E = self._aps_scores_true(P_cal, np.asarray(y_cal))
        n = len(E)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.qhat_ = float(np.quantile(E, level, method="higher"))
        return self

    def predict_sets(self, P):
        """Return a list of sorted class-label lists, one per row."""
        order = np.argsort(-P, axis=1)
        sorted_p = np.take_along_axis(P, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        k = (cum >= self.qhat_).argmax(1)  # first index that reaches qhat
        return [sorted(self.classes[order[i, : k[i] + 1]].tolist()) for i in range(len(P))]

    def evaluate(self, P, y) -> dict:
        y = np.asarray(y)
        sets = self.predict_sets(P)
        cover = np.mean([t in s for t, s in zip(y, sets)])
        sizes = np.array([len(s) for s in sets])
        singleton = sizes == 1
        # undertriage exposure: among truly critical (1/2), did the set's max (least urgent) miss?
        crit = np.isin(y, (1, 2))
        return {
            "alpha": self.alpha, "target_coverage": 1 - self.alpha,
            "empirical_coverage": round(float(cover), 4),
            "mean_set_size": round(float(sizes.mean()), 3),
            "auto_rate_singletons": round(float(singleton.mean()), 4),
            "defer_rate": round(float((~singleton).mean()), 4),
            "qhat": round(self.qhat_, 4),
            "n_critical": int(crit.sum()),
        }
