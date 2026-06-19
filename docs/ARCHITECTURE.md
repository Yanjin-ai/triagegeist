# Architecture — Triage Intelligence Stack

## Two runtime modes
**Offline (training) pipeline** and **online (inference) service** share `src/` code so behavior is identical.

```
OFFLINE                                   ONLINE
raw (kagglehub)                           triage intake (vitals, demo,
   │ ETL/join                             arrival, history, complaint)
   ▼                                          │
cohort + labels  ──┐                          ▼
   │               │                     feature transform (same code)
feature transform  │                          │
   ▼               │                          ▼
train/val/test  ───┘                     multimodal model + calibration
   │                                          │
   ├─ GBDT / fusion training                  ├─ acuity distribution
   ├─ calibration fit                         ├─ admission risk
   ├─ fairness + OOD audit                    ├─ deterioration risk
   ▼                                          ├─ resource bucket
model artifacts + calibrators ───────────────┤   confidence (uncertainty)
   (versioned in reports/ / Kaggle dataset)   └─ evidence trace
                                                  │
                                              ┌───┴────────────────┐
                                         OPS LAYER            LLM ASSIST
                                         queue ranking        complaint norm
                                         resource buckets     reasoning trace
                                         ED load / alerts     guideline RAG
                                         bias monitor         case summary
                                                  │
                                              CLINICAL UI (3 pages)
                                              queue · patient · ops console
```

## Module responsibilities (`src/`)
- **data/** — `download.py` (kagglehub), `cohort.py` (filters/dedup), `labels.py` (target engineering), `build.py` (one-command rebuild → `data/processed/`).
- **features/** — vitals/derived, demographics, arrival, history, text passthrough; a single `transform()` used offline *and* online (no train/serve skew).
- **models/** — `gbdt.py` (baseline multitask), `fusion.py` (FT-Transformer + ClinicalBERT late fusion), `calibrate.py`, `uncertainty.py`.
- **analysis/** — `metrics.py` (AUROC/AUPRC/QWK/Brier/ECE, per-subgroup), `fairness.py`, `reliability.py`, `ood.py`, `error_taxonomy.py`.
- **ops/** — `resource_map.py` (risk → operational bucket), `prioritize.py` (queue policy), `ed_load.py` (aggregate dashboards).
- **llm/** — `normalize.py`, `reasoning.py`, `rag.py` (retrieval over ESI/MIETIC snippets), `summary.py`. LLM advises only; never overrides the predictor.
- **serve/** — `infer.py` (intake → full prediction object + evidence), thin API for the UI.

## Key design rules
1. **No train/serve skew:** one `transform()`, one schema contract.
2. **Triage-time only:** features must be knowable at triage; leakage audit enforced in tests.
3. **Calibrated + uncertain:** every risk ships with a calibrated probability and an uncertainty estimate.
4. **Undertriage-protective thresholds:** operating points favor recall on critical outcomes; trade-offs documented.
5. **Auditability:** every recommendation carries feature attribution + retrieved guideline + similar-case rationale.
6. **LLM is assistive:** normalization/explanation/RAG only — outputs are grounded and cited.
