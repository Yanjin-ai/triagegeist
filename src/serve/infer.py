"""Online inference — the interaction layer.

A `TriageEngine` you query with a single patient intake (dict) and get back a full
triage decision: calibrated acuity distribution, a *conformal prediction set* with a
coverage guarantee, P(critical), admission risk, predicted LOS, resource bucket,
queue priority, a human-deferral flag, and an offline complaint-normalization trace.

    from src.serve.infer import TriageEngine
    eng = TriageEngine().build()           # trains once (~seconds), caches in process
    eng.predict({ "age": 72, "sex": "F", "chief_complaint_raw": "central chest pain ...",
                  "news2_score": 9, "spo2": 90, ... })

    python -m src.serve.infer --example    # runs three example intakes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.calibration import IsotonicOVR
from ..analysis.conformal import APSConformal
from ..config import load_config, processed_dir
from ..data.build import SCORED_TARGET
from ..models.aux_heads import fit_aux
from ..models.structured import fit_tabular
from ..ops.prioritize import POLICIES, needs_senior_review, priority_score
from ..ops.resource_map import assign_bucket
from .governance import data_provenance, decision_trace, oversight

CLASSES = np.array([1, 2, 3, 4, 5])
RED_FLAGS = ["haemorrhage", "hemorrhage", "sepsis", "septic", "arrest", "unresponsive",
             "stroke", "meningitis", "thunderclap", "anaphylaxis", "stab", "gunshot",
             "necrotis", "failure", "hypoxia", "haemoptysis", "embolism", "shock",
             "status asthmaticus", "stemi", "seizure"]


class TriageEngine:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.built = False

    def build(self) -> "TriageEngine":
        cfg = load_config()
        out = processed_dir(cfg)
        train, val, test = (pd.read_parquet(out / f"{n}.parquet") for n in ("train", "val", "test"))
        self.cfg, self.train = cfg, train
        self.tab = fit_tabular(train, val, test, cfg)
        yv = val[SCORED_TARGET].astype(int).to_numpy()
        self.iso = IsotonicOVR(CLASSES).fit(self.tab.p_val, yv)
        # conformal on raw model scores (softer than peaked isotonic probs)
        from sklearn.model_selection import train_test_split
        ic, _ = train_test_split(np.arange(len(val)), test_size=0.5, random_state=42, stratify=yv)
        self.conf = APSConformal(self.alpha, CLASSES).fit(self.tab.p_val[ic], yv[ic])
        self.aux = fit_aux(train, val, test, cfg)
        self.ent_hi = float(np.nanpercentile(
            -(self.iso.transform(self.tab.p_val) * np.log(np.clip(self.iso.transform(self.tab.p_val), 1e-12, 1))).sum(1), 75))
        ft = self.tab.ft
        self.required_raw = sorted((set(ft.numeric) | set(ft.categorical) | set(ft._hx_cols)
                                    | {"pain_score", "systolic_bp"}) - {ft.text_col})
        self.built = True
        return self

    @staticmethod
    def _normalize(text):
        t = str(text or "").lower()
        parts = str(text or "").split(",")
        flags = sorted({f for f in RED_FLAGS if f in t})
        return {"core": parts[0].strip(), "qualifiers": [p.strip() for p in parts[1:] if p.strip()],
                "red_flags": flags}

    def predict(self, intake: dict, policy: str = "safety_first") -> dict:
        if not self.built:
            self.build()
        df = pd.DataFrame([intake])
        raw = self.tab.predict_proba(df)            # raw model scores (for conformal)
        proba = self.iso.transform(raw)[0]          # calibrated (for display / acuity)
        model_acuity = int(CLASSES[proba.argmax()])
        p_crit = float(proba[0] + proba[1])
        ent = float(-(np.clip(proba, 1e-12, 1) * np.log(np.clip(proba, 1e-12, 1))).sum())
        pred_set = self.conf.predict_sets(raw)[0]
        override_triggered = bool(
            (intake.get("news2_score") or 0) >= 7 or (intake.get("spo2") or 100) < 90
            or (intake.get("gcs_total") or 15) <= 13)
        # safety net APPLIED: high-risk physiology caps the suggestion at ESI ≤3
        acuity = min(model_acuity, 3) if override_triggered else model_acuity
        p_admit, los = self.aux.predict(df)
        p_admit, los = float(p_admit[0]), float(los[0])
        bucket = assign_bucket([acuity], [p_admit], [los]).iloc[0]
        prio = round(float(priority_score([acuity], [p_crit], [ent], [intake.get("wait_hours", 0.0)], policy)[0]), 4)
        defer = len(pred_set) > 1 or bool(needs_senior_review([acuity], [p_crit], [ent], self.ent_hi)[0])

        prob_map = {int(c): round(float(p), 4) for c, p in zip(CLASSES, proba)}
        prov = data_provenance(intake, self.required_raw)
        result = {
            "acuity": acuity,
            "model_acuity": model_acuity,
            "acuity_proba": prob_map,
            "conformal_set": pred_set,
            "conformal_coverage": round(1 - self.alpha, 2),
            "p_critical": round(p_crit, 4),
            "admission_risk": round(p_admit, 4),
            "predicted_los_h": round(los, 2),
            "resource_bucket": bucket,
            "priority": round(prio, 4),
            "uncertainty_entropy": round(ent, 4),
            "safety_override_triggered": override_triggered,
            "defer_to_human": defer,
            "complaint_trace": self._normalize(intake.get("chief_complaint_raw")),
            "data_provenance": prov,
            "decision_trace": decision_trace(prov, prob_map, pred_set, acuity, self.alpha,
                                             override_triggered, bucket, prio, policy, defer),
            "oversight": oversight(intake, acuity, p_crit, defer, override_triggered, bucket),
        }
        return result


EXAMPLES = [
    {"label": "critical", "intake": {
        "age": 68, "sex": "M", "language": "Finnish", "insurance_type": "public",
        "arrival_mode": "ambulance", "chief_complaint_raw": "central chest pain radiating to arm, sudden",
        "chief_complaint_system": "cardiovascular", "news2_score": 9, "spo2": 89, "heart_rate": 122,
        "respiratory_rate": 26, "systolic_bp": 95, "gcs_total": 14, "pain_score": 9,
        "mental_status_triage": "alert"}},
    {"label": "ambiguous", "intake": {
        "age": 55, "sex": "F", "language": "Somali", "insurance_type": "none",
        "arrival_mode": "walk-in", "chief_complaint_raw": "acute angle closure glaucoma with nausea",
        "chief_complaint_system": "ophthalmic", "news2_score": 2, "spo2": 98, "heart_rate": 84,
        "respiratory_rate": 16, "systolic_bp": 138, "gcs_total": 15, "pain_score": 7,
        "mental_status_triage": "alert"}},
    {"label": "minor", "intake": {
        "age": 24, "sex": "M", "language": "English", "insurance_type": "private",
        "arrival_mode": "walk-in", "chief_complaint_raw": "small cut wound check",
        "chief_complaint_system": "dermatological", "news2_score": 0, "spo2": 99, "heart_rate": 70,
        "respiratory_rate": 14, "systolic_bp": 124, "gcs_total": 15, "pain_score": 2,
        "mental_status_triage": "alert"}},
]


def main():
    import json
    eng = TriageEngine().build()
    for ex in EXAMPLES:
        r = eng.predict(ex["intake"])
        print(f"\n=== {ex['label']}: {ex['intake']['chief_complaint_raw']!r} ===")
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
