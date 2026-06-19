# Model card — `model_structured_v1`

## Overview
- **Task:** predict emergency-department triage acuity (ESI 1–5, ordinal) at triage time.
- **Type:** gradient-boosted decision trees (LightGBM on Kaggle / sklearn HistGradientBoosting locally),
  multiclass with balanced class weights; isotonic one-vs-rest probability calibration.
- **Inputs (text-blind):** vitals + derived (shock index, NEWS2), anthropometrics, demographics
  (age, sex, language, insurance), arrival context, history counts, 25 `hx_*` comorbidity flags +
  grouped comorbidity counts, `pain_missing`/`bp_missing` indicators. **Excludes raw complaint text by design.**
- **Outputs:** calibrated acuity distribution; argmax acuity; derived `P(critical)=P(1)+P(2)`.
- **Auxiliary heads:** admission-risk classifier, ED-LOS regressor (train-only labels; power the ops layer).

## Intended use
Research / hackathon demonstration of ED triage decision *support*. The model **advises**; it is not an
autonomous triage decision-maker. **Not validated or intended for real clinical deployment.**

## Training & evaluation
- Stratified split on acuity (seed 42); leakage-safe (per-visit-unique `patient_id`, train ∩ test = ∅).
- Outcome columns (`disposition`, `ed_los_hours`) are blocklisted as features.
- **Internal val:** QWK ≈ 0.93, accuracy ≈ 0.85; acuity-1/2 recall ≈ 0.95/0.97; undertriage(1,2) ≈ 2–3%.
- **Calibration:** confidence ECE 0.022 → 0.012 (isotonic).

## Known limitations & risks
- **Label leakage in the dataset:** acuity is near-deterministic in the complaint text, so the public
  leaderboard (text models) is uninformative about clinical generalization. This model deliberately ignores
  text to stay clinically meaningful. See `reports/INSIGHT_label_leakage.md`.
- **Subgroup disparities:** higher undertriage for some language groups (e.g. Estonian) and
  `insurance=unknown`; degraded performance when pain was not assessed. Reported with bootstrap CIs.
- **Physiology vs label divergence:** a physiology-based safety override reduces agreement with these
  synthetic labels (an honest negative result, not a bug).
- **Synthetic data:** distributions are literature-calibrated but not real patients; do not infer real-world
  performance.

## Ethical considerations
Triage errors are safety-critical and unequally distributed; any deployment would require prospective
validation, subgroup monitoring, human oversight, and an undertriage safety net. Use is non-commercial research.
