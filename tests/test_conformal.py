"""Conformal guarantee — empirical coverage on held-out data should meet 1-alpha."""
import numpy as np

from src.analysis.conformal import APSConformal


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(1, 6, size=n)
    # probabilities that concentrate (noisily) on the true class
    P = rng.dirichlet(np.ones(5), size=n)
    for i in range(n):
        P[i, y[i] - 1] += 2.0
    P /= P.sum(1, keepdims=True)
    return P, y


def test_marginal_coverage_holds():
    Pc, yc = _synthetic(4000, 0)
    Pt, yt = _synthetic(4000, 1)
    for alpha in (0.1, 0.2):
        conf = APSConformal(alpha).fit(Pc, yc)
        cov = conf.evaluate(Pt, yt)["empirical_coverage"]
        assert cov >= (1 - alpha) - 0.03, f"coverage {cov} < target {1-alpha}"


def test_sets_are_nonempty_and_valid_labels():
    Pc, yc = _synthetic(1000, 2)
    conf = APSConformal(0.1).fit(Pc, yc)
    sets = conf.predict_sets(Pc[:50])
    assert all(len(s) >= 1 for s in sets)
    assert all(set(s).issubset({1, 2, 3, 4, 5}) for s in sets)
