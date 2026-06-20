"""Reusable tabular GBDT trainer — returns calibrated class probabilities.

Engine pluggable: LightGBM if importable (Kaggle), else sklearn HGB (local).
Used by both the plain baseline (gbdt.py) and the fusion model (fusion.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from ..data.build import SCORED_TARGET
from ..features.transform import FeatureTransformer

CLASSES = np.array([1, 2, 3, 4, 5])


@dataclass
class TabularResult:
    engine: str
    classes_: np.ndarray
    p_val: np.ndarray      # (n_val, 5)
    p_test: np.ndarray     # (n_test, 5)
    ft: FeatureTransformer
    model: object
    cols: list | None = None

    def predict_proba(self, df) -> np.ndarray:
        """Score an arbitrary frame (e.g. a single intake) → (n, 5) aligned to 1..5."""
        X = self.ft.transform(df)[self.cols]
        return _align_proba(self.model, X)


def _align_proba(model, X) -> np.ndarray:
    """Return proba columns aligned to CLASSES (1..5), filling absent classes with 0."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(model.classes_):
        out[:, np.where(CLASSES == c)[0][0]] = proba[:, j]
    return out


def fit_tabular(train, val, test, cfg) -> TabularResult:
    ft = FeatureTransformer(cfg).fit(train)
    cols = ft.model_columns
    Xtr, Xva, Xte = (ft.transform(d)[cols] for d in (train, val, test))
    ytr = train[SCORED_TARGET].astype(int).values
    sw = compute_sample_weight(class_weight="balanced", y=ytr)
    cat_cols = [c for c in ft.categorical if c in cols]

    try:
        import lightgbm as lgb
        engine = "lightgbm"
    except Exception:
        engine = "hgb"

    if engine == "lightgbm":
        import lightgbm as lgb
        for c in cat_cols:
            for X in (Xtr, Xva, Xte):
                X[c] = X[c].astype("category")
        model = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(Xtr, ytr, sample_weight=sw, categorical_feature=cat_cols)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=1.0, categorical_features="from_dtype",
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        model.fit(Xtr, ytr, sample_weight=sw)

    return TabularResult(engine, CLASSES, _align_proba(model, Xva),
                         _align_proba(model, Xte), ft, model, cols)


def proba_to_pred(p: np.ndarray) -> np.ndarray:
    return CLASSES[p.argmax(axis=1)]
