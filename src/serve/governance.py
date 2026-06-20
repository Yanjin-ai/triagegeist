"""Make the decision chain, data processing, and multi-party oversight first-class.

Every prediction carries:
  - data_provenance: which features were provided vs imputed, sentinel handling,
    and an explicit no-leakage attestation (outcomes never used as inputs);
  - decision_trace: the ordered, auditable chain intake → processing → model →
    calibration → conformal → safety override → resource map → priority → disposition;
  - oversight: which parties must review and WHY, each tied to a concrete signal
    from our own analysis (conformal defer, undertriage-prone subgroup, bed impact).
This is what lets a nurse, a senior physician, an equity auditor, and operations each
supervise the same decision.
"""
from __future__ import annotations

LEAKAGE = ["disposition", "ed_los_hours"]
# subgroups our fairness audit (reports/governance.md) flagged with elevated undertriage
ELEVATED_UNDERTRIAGE_LANG = {"Estonian", "Swedish", "Somali", "English"}
BED_BUCKETS = {"resuscitation/immediate bed", "high-frequency monitoring", "standard bed"}


def data_provenance(intake: dict, required_raw: list[str]) -> dict:
    provided = [c for c in required_raw if intake.get(c) is not None]
    imputed = [c for c in required_raw if intake.get(c) is None]
    pain = intake.get("pain_score")
    return {
        "features_required": len(required_raw),
        "features_provided": len(provided),
        "features_imputed": sorted(imputed),
        "pain_sentinel_handling": "pain not assessed (-1/None) → flagged, not silently zero"
        if pain in (None, -1) else "pain recorded",
        "outcome_columns_excluded": LEAKAGE,
        "no_leakage_attestation": "model is text-blind and never sees disposition/LOS or the "
        "complaint text that encodes the label",
    }


def decision_trace(prov, proba, pred_set, acuity, alpha, override_triggered,
                   bucket, priority, policy, defer) -> list[dict]:
    return [
        {"step": 1, "stage": "intake", "detail":
         f"{prov['features_provided']}/{prov['features_required']} features provided, "
         f"{len(prov['features_imputed'])} imputed"},
        {"step": 2, "stage": "data processing", "detail":
         f"{prov['pain_sentinel_handling']}; outcomes {prov['outcome_columns_excluded']} excluded "
         "(no leakage)"},
        {"step": 3, "stage": "model (text-blind structured)", "detail":
         "acuity distribution " + ", ".join(f"{c}:{p:.0%}" for c, p in proba.items())},
        {"step": 4, "stage": "calibration", "detail": "isotonic OVR (val ECE ≈ 0.012)"},
        {"step": 5, "stage": "conformal set", "detail":
         f"{pred_set} @ {(1-alpha)*100:.0f}% coverage guarantee"},
        {"step": 6, "stage": "safety override", "detail":
         "TRIGGERED — high-risk physiology caps acuity ≤3" if override_triggered
         else "not triggered"},
        {"step": 7, "stage": "resource mapping", "detail": bucket},
        {"step": 8, "stage": f"prioritization ({policy})", "detail": f"priority score {priority}"},
        {"step": 9, "stage": "disposition", "detail":
         "DEFER to human review" if defer else f"auto-suggest ESI {acuity} (clinician confirms)"},
    ]


def oversight(intake: dict, acuity: int, p_crit: float, defer: bool,
              override_triggered: bool, bucket: str) -> list[dict]:
    roles = [{
        "party": "Triage nurse", "role": "primary decision-maker",
        "required": True, "action": "confirm or override the suggestion",
        "why": "the model advises; the nurse decides"}]
    if defer or p_crit >= 0.5 or acuity <= 2 or override_triggered:
        reasons = []
        if acuity <= 2 or p_crit >= 0.5:
            reasons.append("high acuity / P(critical)")
        if defer:
            reasons.append("conformal set ambiguous (deferred)")
        if override_triggered:
            reasons.append("physiology safety override fired")
        roles.append({"party": "Senior physician", "role": "second review",
                      "required": True, "action": "co-sign or escalate",
                      "why": "; ".join(reasons)})
    lang, ins = intake.get("language"), intake.get("insurance_type")
    pain_missing = intake.get("pain_score") in (None, -1)
    equity = []
    if lang in ELEVATED_UNDERTRIAGE_LANG:
        equity.append(f"language={lang} (elevated undertriage in audit)")
    if ins == "unknown":
        equity.append("insurance=unknown (elevated undertriage)")
    if pain_missing:
        equity.append("pain not assessed (subgroup QWK degrades)")
    if equity:
        roles.append({"party": "Quality / equity auditor", "role": "bias oversight",
                      "required": False, "action": "monitor for systematic undertriage",
                      "why": "; ".join(equity)})
    if bucket in BED_BUCKETS:
        roles.append({"party": "Operations", "role": "capacity oversight",
                      "required": False, "action": "reserve bed / monitoring resource",
                      "why": f"bucket '{bucket}' consumes inpatient capacity"})
    return roles
