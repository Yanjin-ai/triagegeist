# DATA DICTIONARY — Triagegeist (confirmed from real files, 2026-06-19)

Source: `kagglehub.competition_download('triagegeist')` → `./triagegeist/`.
Synthetic Finnish-ED dataset (sites Helsinki/Oulu/Tampere/Turku; Laitinen-Fredriksson Foundation).

## Files & join
| File | Rows | Key | Notes |
|---|---|---|---|
| `train.csv` | 80,000 × 40 | `patient_id` | features + 3 train-only labels |
| `test.csv` | 20,000 × 37 | `patient_id` | features only (no labels) |
| `chief_complaints.csv` | 100,000 × 3 | `patient_id` | `chief_complaint_raw` (free text) + `chief_complaint_system` |
| `patient_history.csv` | 100,000 × 26 | `patient_id` | 25 `hx_*` binary comorbidity flags |
| `sample_submission.csv` | 20,000 × 2 | `patient_id` | `triage_acuity` (all `3`) |

- `patient_id` is **per-visit unique**: no duplicates in train, **train ∩ test = ∅** → a stratified split is leakage-safe (no patient-level leakage).
- `chief_complaints` & `patient_history` cover **train + test** (join coverage = 1.0).

## Targets
| Column | Where | Type | Use |
|---|---|---|---|
| **`triage_acuity`** | train only | ordinal 1–5 | **SCORED submission target** (1=most urgent … 5=least) |
| `disposition` | train only | categorical(7) | **auxiliary multitask head** + ops/analysis (not scored) |
| `ed_los_hours` | train only | float [0, 17.5] | auxiliary head (resource/LOS) + ops (not scored) |

- `triage_acuity` distribution: {1: 3222 (4.0%), 2: 13439 (16.8%), 3: 28921 (36.2%), 4: 23020 (28.8%), 5: 11398 (14.2%)} — **imbalanced; class 1 is rare & safety-critical**.
- `disposition`: discharged 39028, admitted 24601, transferred 5203, observation 4337, lwbs 3656, lama 2764, **deceased 411**. → admission/deterioration proxies for the ops layer.
- ⚠️ **Scoring metric not yet confirmed** — check the competition *Evaluation* tab. Ordinal target ⇒ most likely **Quadratic Weighted Kappa**; could be accuracy/macro-F1. Build to optimize QWK while reporting accuracy + macro-F1. (Open item, `DECISIONS.md`.)

## Features (37, present in train & test)
**Vitals / scores (numeric):** `systolic_bp` `diastolic_bp` `mean_arterial_pressure` `pulse_pressure` `heart_rate` `respiratory_rate` `temperature_c` `spo2` `gcs_total` `pain_score` `shock_index` `news2_score`
**Anthropometric:** `weight_kg` `height_cm` `bmi`
**History counts:** `num_prior_ed_visits_12m` `num_prior_admissions_12m` `num_active_medications` `num_comorbidities`
**Categorical clinical:** `pain_location` `mental_status_triage` (alert/drowsy/confused/agitated…) `chief_complaint_system` (14 systems, ~balanced)
**Context:** `site_id` (5) `triage_nurse_id` (50) `arrival_mode` (e.g. walk-in/ambulance) `transport_origin` `shift` `arrival_hour` `arrival_day` `arrival_month` `arrival_season`
**Demographics (equity axes):** `age` `age_group` (pediatric/young_adult/middle_aged/elderly) `sex` (F/M/Other) `language` (Finnish 55% / Swedish / Russian / Estonian / Arabic / Somali / English / Other) `insurance_type` (public/private/none/military/unknown)
**Text (separate file):** `chief_complaint_raw` — short clinical phrase, highly predictive of acuity.
**Comorbidities (separate file):** 25 `hx_*` binary flags (hypertension, diabetes, heart_failure, ckd, malignancy, dementia, pregnant, substance_use_disorder, stroke_prior, …).

## Missingness & sentinels (train)
- `pain_score == -1` → **"not assessed" sentinel** (13.9%). Recode to NaN + add `pain_missing` indicator (also a fairness axis).
- `systolic_bp`/`diastolic_bp`/`mean_arterial_pressure`/`pulse_pressure`/`shock_index` ≈ **5.18%** missing (co-missing — BP not taken).
- `respiratory_rate` 3.83%, `temperature_c` 0.72%. Others ~0.
- `news2_score` & `shock_index` are derived from vitals → triage-time legal features (no leakage).

## Leakage rules
- **Never** use `disposition`, `ed_los_hours` as input features (outcomes).
- All listed features are knowable at triage time → allowed.

## Analysis hooks (rubric goldmine)
- **Undertriage:** predicted acuity > true acuity on safety-critical cases (true 1/2). Stratify by `language`, `insurance_type`, `sex`, `age_group`, `pain_missing`.
- **Inter-rater / systematic bias:** `triage_nurse_id` (50) and `site_id` (5) — site means are ~equal (3.31–3.33), so look within matched cases for nurse-level drift.
