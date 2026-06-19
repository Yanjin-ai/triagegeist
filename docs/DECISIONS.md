# Decision log (append-only)

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-19 | Treat as a **judged project** competition (model/analysis/prototype), optimize for the rubric (clinical relevance, technical quality, insight/honesty, novelty/impact) — not a leaderboard metric. | Competition framing; maximizes scoring surface. |
| 2026-06-19 | **Data via `kagglehub.competition_download('triagegeist')`** — no PhysioNet credentialing path needed. | Confirmed by user. |
| 2026-06-19 | **Primary build/training environment = Kaggle free-GPU notebooks**; mirror `src/` as a Kaggle utility/dataset for `import`. | Confirmed by user; reproducibility-first; enables transformer + ClinicalBERT. |
| 2026-06-19 | **Build the full three-layer stack**, with the **operations console (Version C)** as the final deliverable form. | Confirmed by user; differentiates on impact/novelty. |
| 2026-06-19 | **Multitask targets**: acuity (P0), admission (P0), deterioration (P1), resource bucket (P1). | Rubric rewards going beyond plain ESI 5-class. |
| 2026-06-19 | **LLM is assistive, not adjudicator** (normalization / reasoning trace / guideline RAG / summary). | Clinical safety + auditability; avoids black-box "replace the nurse". |

| 2026-06-19 | **Phase 0 recon: hybrid competition.** Scored target = `triage_acuity` (1–5). `disposition` + `ed_los_hours` are train-only → reframed as **auxiliary multitask heads + ops/analysis signals**, not scored. | Real data inspected; `sample_submission.csv` predicts `triage_acuity`. |
| 2026-06-19 | **Split = stratified on acuity (seed 42).** Dropped the patient-level/time-split plan. | `patient_id` is per-visit unique; train ∩ test = ∅ → no patient leakage. |
| 2026-06-19 | **Handle `pain_score=-1` as a missing sentinel + `pain_missing` indicator**; treat BP/RR co-missingness explicitly. | Confirmed in recon (13.9% sentinel; ~5% BP missing). |
| 2026-06-19 | **MIETIC/MIMIC not shipped** — dataset is a self-contained synthetic Finnish-ED set. RAG guideline corpus (Phase 7) sourced separately (public ESI handbook snippets). | No PhysioNet tables in the download. |
| 2026-06-19 | **CRITICAL: acuity label is a near-deterministic function of `chief_complaint_raw`** (99.7% phrase→single-acuity; 99.8% test phrases seen verbatim in train). Text/fusion ⇒ QWK ≈ 1.0 — leaderboard is a lookup. See `reports/INSIGHT_label_leakage.md`. | Phase 3 v1 probes. |
| 2026-06-19 | **Adopt dual-track strategy.** Track 1 (leaderboard formality): `submission_fusion.csv` (≈1.0). **Track 2 (judged, the real work): the text-blind structured-physiology model (QWK ≈ 0.93)** is the honest core for calibration/fairness/ops; the writeup leads with the leakage finding. | Reporting ≈1.0 as a result would be dishonest; clinical value lives in the structured model. |

## Open decisions (to resolve at the noted phase)
- **Exact scoring metric** (QWK vs accuracy vs macro-F1) — **confirm on Evaluation tab** before tuning (assumed QWK).
- GBDT acuity as multiclass vs ordinal-regression-to-rank — **Phase 2** (test both).
- UI framework: Streamlit vs React+FastAPI — **Phase 8** (driven by demo-hosting).
- Demo hosting target (Kaggle can't host always-on) — **Phase 8**.
