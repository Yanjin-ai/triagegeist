# Data statement

## Source & nature
- Provided by the **Triagegeist** Kaggle competition (`kagglehub.competition_download('triagegeist')`),
  attributed to the Laitinen-Fredriksson Foundation.
- **Synthetic** emergency-department data styled on a Finnish ED system (sites Helsinki / Oulu / Tampere /
  Turku); distributions described as calibrated to MIMIC-IV-ED and ESI triage literature.
- Not real patient records. No PHI. **Not for clinical use or real-world inference.**

## Contents
- `train.csv` (80k) — intake features + train-only labels `triage_acuity`, `disposition`, `ed_los_hours`.
- `test.csv` (20k) — intake features only.
- `chief_complaints.csv` (100k) — `chief_complaint_raw` free text + `chief_complaint_system`.
- `patient_history.csv` (100k) — 25 `hx_*` binary comorbidity flags.
- All join on `patient_id` (per-visit unique). Full data dictionary: `docs/DATA_DICTIONARY.md`.

## Known data properties / caveats
- **Acuity is near-deterministic in the complaint text** (99.7% phrase→single acuity; 99.8% test phrases seen
  in train) — the label appears generated from the complaint. The leaderboard is effectively a lookup.
- `pain_score=-1` is a "not assessed" sentinel (13.9%); BP/shock-index ~5% co-missing — missingness is
  informative and modeled explicitly.
- Class imbalance (acuity-1 ≈ 4%); the safety-critical errors are rare.
- Subgroup composition includes immigrant-language groups (Arabic, Somali, Estonian, …) and varied insurance
  status, enabling — and motivating — an equity audit.

## Licensing & use
- Competition data is for **non-commercial research**; **redistribution is prohibited**. This repository does
  **not** include the data (it is git-ignored); reproduce by downloading via `kagglehub` under the competition
  rules. Derived artifacts here (code, notebook, UI, reports) contain only aggregate statistics and a small
  number of illustrative synthetic rows.
