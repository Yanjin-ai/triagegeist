"""Leakage guard — outcome columns must never reach the model matrix."""
import pandas as pd

from src.config import load_config
from src.data.build import LEAKAGE, load_joined
from src.features.transform import FeatureTransformer


def test_no_leakage_columns_in_features():
    cfg = load_config()
    train, _ = load_joined(cfg)
    ft = FeatureTransformer(cfg).fit(train.head(1000))
    assert not (set(LEAKAGE) & set(ft.model_columns)), "leakage column in features!"


def test_transform_is_deterministic_and_skew_free():
    cfg = load_config()
    train, test = load_joined(cfg)
    ft = FeatureTransformer(cfg).fit(train)
    a = ft.transform(test.head(500))[ft.model_columns]
    b = ft.transform(test.head(500))[ft.model_columns]
    pd.testing.assert_frame_equal(a, b)
    # same columns offline (train) and online (test)
    assert list(ft.transform(train.head(10))[ft.model_columns].columns) == ft.model_columns


def test_pain_sentinel_recoded():
    cfg = load_config()
    train, _ = load_joined(cfg)
    ft = FeatureTransformer(cfg).fit(train)
    out = ft.transform(train)
    assert (out["pain_score"] == -1).sum() == 0
    assert out["pain_missing"].sum() > 0
