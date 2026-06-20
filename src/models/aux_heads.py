"""P6 — auxiliary heads (text-blind): admission risk + ED length-of-stay.

`disposition` and `ed_los_hours` are train-only labels → trained here to power the
ops layer (admission risk, LOS/resource signal). Not scored on the leaderboard.

admit_positive = disposition needs inpatient resources
              = {admitted, observation, transferred, deceased}
(discharged / lwbs / lama = not admitted).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.transform import FeatureTransformer

ADMIT_POSITIVE = {"admitted", "observation", "transferred", "deceased"}


@dataclass
class AuxResult:
    p_admit_val: np.ndarray
    p_admit_test: np.ndarray
    los_val: np.ndarray
    los_test: np.ndarray
    clf: object = None
    reg: object = None
    ft: object = None
    cols: list = None

    def predict(self, df):
        X = self.ft.transform(df)[self.cols]
        return self.clf.predict_proba(X)[:, 1], np.clip(self.reg.predict(X), 0, None)


def fit_aux(train, val, test, cfg) -> AuxResult:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    ft = FeatureTransformer(cfg).fit(train)
    cols = ft.model_columns
    Xtr, Xva, Xte = (ft.transform(d)[cols] for d in (train, val, test))

    y_admit = train["disposition"].isin(ADMIT_POSITIVE).astype(int).values
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, categorical_features="from_dtype",
        early_stopping=True, random_state=42,
    ).fit(Xtr, y_admit)

    y_los = train["ed_los_hours"].astype(float).values
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, categorical_features="from_dtype",
        early_stopping=True, random_state=42,
    ).fit(Xtr, y_los)

    return AuxResult(
        clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1],
        np.clip(reg.predict(Xva), 0, None), np.clip(reg.predict(Xte), 0, None),
        clf=clf, reg=reg, ft=ft, cols=cols,
    )
