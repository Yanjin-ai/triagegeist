"""Phase 3 v1 guards — text join determinism, frozen vocab, fusion reproducibility."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.build import load_joined
from src.features.text import TextBranch
from src.models.fusion import TEXT_COL, search_alpha


def test_text_join_is_deterministic():
    cfg = load_config()
    a, _ = load_joined(cfg)
    b, _ = load_joined(cfg)
    pd.testing.assert_series_equal(a[TEXT_COL], b[TEXT_COL])
    assert a[TEXT_COL].isna().sum() == 0  # full join coverage


def test_vectorizer_vocab_is_frozen_after_fit():
    cfg = load_config()
    train, _ = load_joined(cfg)
    sample = train.head(3000)
    tb = TextBranch().fit(sample[TEXT_COL], sample["triage_acuity"].astype(int))
    vocab_before = tb.vocab_size
    # transforming unseen text must NOT change the fitted vocabulary
    _ = tb.predict_proba(["brand new unseen complaint xyzzy"])
    assert tb.vocab_size == vocab_before


def test_fusion_alpha_search_is_deterministic():
    rng = np.random.default_rng(0)
    p_tab = rng.dirichlet(np.ones(5), size=400)
    p_text = rng.dirichlet(np.ones(5), size=400)
    y = rng.integers(1, 6, size=400)
    assert search_alpha(p_tab, p_text, y) == search_alpha(p_tab, p_text, y)
