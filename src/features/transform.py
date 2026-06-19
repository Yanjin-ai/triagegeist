"""Phase 1 — feature transform shared by offline training and online serving.

One `FeatureTransformer.transform()` → no train/serve skew. Tree-model friendly:
numeric stays float (NaN kept for native handling), categoricals → pandas category.
Text (`chief_complaint_raw`) is passed through untouched for the fusion model (P3);
tree baselines ignore it. Leakage columns are never emitted.
"""
from __future__ import annotations

import pandas as pd

from ..config import load_config

LEAKAGE = ["disposition", "ed_los_hours"]

# Clinically grouped comorbidity counts (only columns that exist are summed).
HX_GROUPS = {
    "hx_cardiopulmonary": ["hx_copd", "hx_asthma", "hx_heart_failure",
                            "hx_atrial_fibrillation", "hx_coronary_artery_disease",
                            "hx_peripheral_vascular_disease"],
    "hx_metabolic": ["hx_diabetes_type1", "hx_diabetes_type2", "hx_obesity",
                     "hx_hypothyroidism", "hx_hyperthyroidism", "hx_ckd"],
    "hx_neurocognitive": ["hx_dementia", "hx_stroke_prior", "hx_epilepsy",
                          "hx_depression", "hx_anxiety"],
    "hx_highrisk": ["hx_malignancy", "hx_liver_disease", "hx_immunosuppressed",
                    "hx_coagulopathy", "hx_hiv"],
}


class FeatureTransformer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        f = self.cfg["features"]
        self.numeric = (
            f["numeric_vitals"] + f["anthropometric"] + f["counts"] + f["numeric_context"]
        )
        self.categorical = list(f["categorical"])
        self.text_col = f["text"]
        self.history_prefix = f["history_prefix"]
        self.pain_sentinel = self.cfg["sentinels"]["pain_score"]
        self._cat_dtypes: dict[str, pd.CategoricalDtype] = {}

    @property
    def feature_cols(self) -> list[str]:
        return self.numeric + self.categorical + self._hx_cols

    def _hx(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.startswith(self.history_prefix)]

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        self._hx_cols = self._hx(df)
        for c in self.categorical:
            cats = pd.Index(df[c].astype("string").fillna("__NA__").unique())
            self._cat_dtypes[c] = pd.CategoricalDtype(categories=cats)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # pain_score sentinel -> NaN + explicit missing flag
        df["pain_missing"] = (df["pain_score"] == self.pain_sentinel).astype("int8")
        df.loc[df["pain_score"] == self.pain_sentinel, "pain_score"] = pd.NA
        # co-missing vitals indicator (BP not taken)
        df["bp_missing"] = df["systolic_bp"].isna().astype("int8")

        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        for flag in ["pain_missing", "bp_missing"]:
            out[flag] = df[flag]
        for c in self._hx_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
        # derived clinically-grouped comorbidity counts
        for g, members in HX_GROUPS.items():
            present = [m for m in members if m in out.columns]
            out[g] = out[present].sum(axis=1).astype("int16") if present else 0
        for c in self.categorical:
            dt = self._cat_dtypes.get(c)
            s = df[c].astype("string").fillna("__NA__")
            out[c] = s.astype(dt) if dt is not None else s.astype("category")
        if self.text_col in df.columns:
            out[self.text_col] = df[self.text_col].astype("string").fillna("")
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def model_columns(self) -> list[str]:
        """Columns fed to tree models (excludes raw text)."""
        return (
            self.numeric
            + ["pain_missing", "bp_missing"]
            + self._hx_cols
            + list(HX_GROUPS.keys())
            + self.categorical
        )
