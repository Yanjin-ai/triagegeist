"""P4 — calibration: ECE (confidence + class-wise), isotonic OVR, reliability tables."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _binary_ece(y, p, n_bins=10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(y)
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1]) if i else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum():
            e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return float(e)


def confidence_ece(y_true, proba, classes, n_bins=10) -> float:
    """Top-label (confidence) ECE."""
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    correct = (classes[proba.argmax(1)] == y_true).astype(float)
    return _binary_ece(correct, conf, n_bins)


def classwise_ece(y_true, proba, classes, n_bins=10):
    y_true = np.asarray(y_true)
    per = {int(c): _binary_ece((y_true == c).astype(float), proba[:, j], n_bins)
           for j, c in enumerate(classes)}
    return float(np.mean(list(per.values()))), per


def critical_coarse_ece(y_true, proba, classes, critical=(1, 2), n_bins=10) -> float:
    """Calibration of P(critical) = sum of acuity-1,2 probabilities."""
    y_true = np.asarray(y_true)
    idx = [np.where(classes == c)[0][0] for c in critical]
    p_crit = proba[:, idx].sum(1)
    y_crit = np.isin(y_true, critical).astype(float)
    return _binary_ece(y_crit, p_crit, n_bins)


class IsotonicOVR:
    """One-vs-rest isotonic calibration; rows renormalized to sum 1."""

    def __init__(self, classes):
        self.classes = np.asarray(classes)
        self.cals: dict[int, IsotonicRegression] = {}

    def fit(self, proba_cal, y_cal) -> "IsotonicOVR":
        y_cal = np.asarray(y_cal)
        for j, c in enumerate(self.classes):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(proba_cal[:, j], (y_cal == c).astype(float))
            self.cals[int(c)] = ir
        return self

    def transform(self, proba) -> np.ndarray:
        out = np.column_stack([self.cals[int(c)].transform(proba[:, j])
                               for j, c in enumerate(self.classes)])
        out = np.clip(out, 1e-9, None)
        return out / out.sum(1, keepdims=True)


def reliability_table(y_true, proba, classes, n_bins=10):
    """Confidence-reliability bins for plotting (centers, mean_conf, mean_acc, n)."""
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    correct = (classes[proba.argmax(1)] == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, mc, ma, n = [], [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        centers.append((bins[i] + bins[i + 1]) / 2)
        mc.append(conf[m].mean() if m.sum() else np.nan)
        ma.append(correct[m].mean() if m.sum() else np.nan)
        n.append(int(m.sum()))
    return np.array(centers), np.array(mc), np.array(ma), np.array(n)
