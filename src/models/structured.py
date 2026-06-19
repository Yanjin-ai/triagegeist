"""model_structured_v1 — the clinically-honest, TEXT-BLIND core model.

Why this is the canonical model (see reports/INSIGHT_label_leakage.md):
the acuity label is a near-deterministic function of `chief_complaint_raw`, so any
model that sees the text trivially scores QWK ≈ 1.0 (leaderboard = lookup). The
text-blind structured-physiology model (vitals, demographics, context, history +
derived comorbidity groups) is the version where calibration, undertriage and
fairness are clinically meaningful. ALL P4/P5/P6 governance runs on THIS model.

It is exactly `fit_tabular` (which never ingests raw text — `FeatureTransformer`
emits `chief_complaint_raw` only as a passthrough that tree models don't use).
Pinned: seed 42, feature space = `FeatureTransformer.model_columns`, balanced weights.
"""
from __future__ import annotations

from .tabular import CLASSES, fit_tabular, proba_to_pred  # re-export

MODEL_ID = "model_structured_v1"

__all__ = ["fit_tabular", "proba_to_pred", "CLASSES", "MODEL_ID"]
