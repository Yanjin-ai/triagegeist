# Triagegeist — Master Plan (Triage Intelligence Stack)

> **One-line thesis:** This is *not* a medical Q&A bot. It is a **multimodal risk-prediction engine at the core, with resource scheduling and explainable analysis as the landing point** — an ED triage decision-support system.

This file is the single source of truth for the project. Any session (human or agent) should be able to read this top-to-bottom and know exactly what to do next. Pair it with `PROGRESS.md` (live state) and `ARCHITECTURE.md` (system design).

---

## 0. Competition framing (locked understanding)

- **Type:** **Hybrid** — a scored Kaggle leaderboard (predict `triage_acuity` 1–5 on the test set) **plus** a judged *project* (model / analysis / prototype). Powered by the Laitinen-Fredriksson Foundation. The two prongs reinforce each other: the scored classifier is L1's primary head; the stack (L2/L3 + LLM) is the judged story. *(Updated 2026-06-19 after Phase 0 recon — see `DATA_DICTIONARY.md`.)*
- **Judged on:** clinical relevance, technical quality, insight / honesty about limitations, novelty / impact potential.
- **Data:** delivered through the competition itself —
  ```python
  import kagglehub
  path = kagglehub.competition_download('triagegeist')
  print("Path to competition files:", path)
  ```
  Derived from **MIMIC-IV-ED** (edstays, triage, vitalsign, diagnosis, medrecon, pyxis) and likely the **MIETIC** instruction corpus. **Exact schema is confirmed in Phase 0, not assumed.**
- **Known data caveats (judge-relevant):** hidden bias, missingness, local practice drift, systematic **undertriage** concern. These are *features to analyze*, not bugs to hide.

### Decisions locked with the user (see `DECISIONS.md`)
1. Data: official Kaggle competition data via `kagglehub`. No PhysioNet credentialing needed.
2. Compute: **Kaggle free-GPU notebooks** are the primary training/serving environment → reproducibility-first; model ambition can include transformers/ClinicalBERT.
3. Scope: **full three-layer stack**, with the **operations console (Version C)** as the final form.

---

## 1. What we are building

A **triage copilot**: predict → explain → support operations. Three layers + an LLM assist.

```
┌─────────────────────────────────────────────────────────────┐
│ L3  DECISION SUPPORT / OPS   queue ranking · resource buckets │
│                              ops console · bias monitor        │
├─────────────────────────────────────────────────────────────┤
│ L2  ANALYSIS / GOVERNANCE    calibration · subgroup fairness   │
│                              uncertainty · OOD · error taxonomy│
├─────────────────────────────────────────────────────────────┤
│ L1  MULTIMODAL PREDICTION    tabular(vitals+demo+arrival) +    │
│                              text(chief complaint) late fusion │
│                              multitask: acuity/admit/deteriorate│
└─────────────────────────────────────────────────────────────┘
        ⤷ LLM ASSIST (not adjudicator): complaint normalization,
          reasoning trace, guideline-grounded RAG, case summary
```

### Prediction targets (multitask)
| Target | Type | Source / definition | Priority |
|---|---|---|---|
| **Acuity (ESI 1–5)** | ordinal | `triage_acuity` — **the scored leaderboard target** | P0 |
| **Disposition** | categorical(7) | `disposition` (admitted/deceased/observation/…), **train-only** → aux head + admission/deterioration proxy for ops | P1 |
| **ED LOS** | regression | `ed_los_hours`, **train-only** → aux head + resource/LOS signal | P1 |
| **Resource-need bucket** | ordinal | derived from acuity + LOS + disposition → operational bucket | P1 |

> Metric: optimize **Quadratic Weighted Kappa** (ordinal) while reporting accuracy + macro-F1 + per-class recall (esp. acuity-1). Confirm exact metric on the Evaluation tab.

> **Design rule:** never make the model the final arbiter. The LLM and model *advise*; outputs carry calibrated confidence + an evidence trace.

---

## 2. Roadmap — phases, deliverables, acceptance gates

Each phase has an **acceptance gate**. Do not advance until the gate is green. `PROGRESS.md` tracks which gate we're at.

### Phase 0 — Setup & data recon  ·  *gate: data understood*
- [ ] Download via `kagglehub`; inventory every file/table, row counts, columns, dtypes, missingness.
- [ ] Write a **data dictionary** (`docs/DATA_DICTIONARY.md`) from the *actual* files (not assumptions).
- [ ] Confirm which targets are constructable (esp. deterioration); pick the deterioration definition.
- [ ] Decide split strategy (patient-level, time-based holdout to respect drift; no leakage across ED stays of same patient).
- [ ] EDA notebook: label prevalence, subgroup composition (age/sex/race/arrival), vitals distributions, text length.
- **Gate:** `DATA_DICTIONARY.md` exists; targets + splits + cohort defined and justified.

### Phase 1 — Data pipeline / feature store  ·  *gate: reproducible features*
- [ ] ETL: join edstays↔triage↔vitalsign↔diagnosis↔medrecon into one stay-level frame.
- [ ] Cohort filters (age, valid triage, dedup) → `src/data/cohort.py`.
- [ ] Label engineering → `src/data/labels.py` (acuity, admit, deteriorate, resource bucket).
- [ ] Feature engineering: vitals (+derived: shock index, news-like flags), demographics, arrival mode, history counts, chief-complaint raw text → `src/features/`.
- [ ] Deterministic train/val/test split saved to `data/processed/`.
- [ ] Leakage audit (no post-triage info leaking into triage-time features).
- **Gate:** `make features` (or `python -m src.data.build`) reproduces processed splits from raw; leakage audit passes.

### Phase 2 — Baselines  ·  *gate: honest baseline numbers*
- [ ] Tabular-only multitask **GBDT** (LightGBM/XGBoost) per target.
- [ ] Text-only baseline (TF-IDF + linear, and a frozen ClinicalBERT embedding + head).
- [ ] Metrics harness → `src/analysis/metrics.py`: AUROC, AUPRC, macro-F1, ordinal acuity (QWK), **Brier + ECE**, per-subgroup.
- [ ] Baseline report table in `reports/`.
- **Gate:** reproducible baseline metrics committed; metrics harness reused by all later models.

> ⚠️ **Reframed after Phase 3 v1** (see `reports/INSIGHT_label_leakage.md`): the acuity label is a near-deterministic function of the complaint text, so text/fusion ⇒ QWK ≈ 1.0 (leaderboard = lookup). The **honest, clinically meaningful model is text-blind structured-physiology (QWK ≈ 0.93)** — P4/P5/P6 governance runs on *that*; deep fusion (ClinicalBERT) is deprioritized.

### Phase 3 — Multimodal fusion  ·  *gate: fusion ≥ baselines*  ·  ✅ v1 done (TF-IDF), deep fusion deprioritized
- [ ] Tabular encoder (FT-Transformer / TabTransformer) + text encoder (ClinicalBERT) **late fusion**, multitask heads.
- [ ] Training loop with class weighting / focal loss for undertriage-sensitive targets.
- [ ] Ablations: tabular-only vs text-only vs fusion; document lift.
- **Gate:** fusion model beats best baseline on primary targets *and* is calibration-comparable; ablation table written.

### Phase 4 — Calibration & uncertainty  ·  *gate: trustworthy probabilities*
- [ ] Post-hoc calibration (temperature / isotonic) per target; reliability curves.
- [ ] Uncertainty: MC-dropout or deep ensemble → predictive entropy / variance.
- [ ] Decision thresholds tuned for **undertriage protection** (high-recall on critical), with explicit operating points.
- **Gate:** ECE materially reduced; reliability curves + chosen operating points documented.

### Phase 5 — Fairness, error taxonomy, OOD  ·  *gate: limitations characterized*
- [ ] Subgroup metrics by age band, sex, race, arrival mode, pain-score missingness: AUROC gap, **undertriage rate**, equalized-odds gap.
- [ ] Error taxonomy: where/why undertriage happens; representative cases.
- [ ] OOD / drift detection on time-based holdout; flag low-confidence regions.
- [ ] Mitigation experiment (reweighting or group-aware thresholds) + honest trade-off discussion.
- **Gate:** fairness audit + error taxonomy + OOD section written for the report (judge-facing).

### Phase 6 — Resource mapping & operational logic  ·  *gate: predictions → actions*
- [ ] Map outputs → operational buckets: *immediate bed · high-frequency monitoring · fast-track · priority re-eval*.
- [ ] Queue prioritization policy (risk × wait × deterioration) → `src/ops/prioritize.py`.
- [ ] ED-level aggregates: risk load, per-acuity counts, predicted deterioration count, bed-pressure signal.
- **Gate:** given a batch of intakes, system emits ranked queue + resource recommendations + ED load summary.

### Phase 7 — LLM assist (auxiliary, not adjudicator)  ·  *gate: grounded, traceable*
- [ ] Chief-complaint normalization (free text → structured concepts).
- [ ] Reasoning trace / case summary generation (templated, model-grounded).
- [ ] Guideline-grounded RAG: "why ESI-2 not ESI-3" using ESI/MIETIC snippets with citations.
- [ ] Guardrails: never overrides the predictor; always cites evidence.
- **Gate:** for a sample case, LLM returns normalized complaint + cited rationale; no ungrounded claims.

### Phase 8 — Product: triage copilot UI  ·  *gate: demo works end-to-end*
- [ ] **Queue page:** ranked waiting room, risk badges, confidence.
- [ ] **Patient page:** acuity distribution, high-risk flags, resource suggestion, confidence, "why" (feature attribution + retrieved guideline + similar case).
- [ ] **Ops console (Version C, final form):** ED risk load, acuity buckets, predicted deteriorations, resource/bed-pressure alerts, **bias monitor**.
- [ ] Wire to inference service (`src/serve/`); seed with held-out cases.
- **Gate:** clickable demo runs the three pages on real held-out data.

### Phase 9 — Packaging & submission  ·  *gate: submittable*
- [ ] Clean **Kaggle notebook(s)**: training, evaluation, fairness, demo — runnable top-to-bottom on Kaggle.
- [ ] **Writeup** mapped to the rubric (clinical relevance · technical quality · insight/limitations · novelty/impact).
- [ ] Demo recording / hosted prototype link.
- [ ] Reproducibility: pinned env, seeds, one-command rebuild, model card + data statement.
- **Gate:** submission package complete and self-contained; dry-run on a fresh Kaggle session passes.

---

## 3. Tech stack
- **Lang/ML:** Python, pandas/Polars, LightGBM/XGBoost, PyTorch, HuggingFace (ClinicalBERT/Bio_ClinicalBERT), scikit-learn (calibration/metrics), `rtdl`/FT-Transformer.
- **Analysis:** custom metrics harness, `fairlearn`-style subgroup audit, reliability/ECE.
- **LLM/RAG:** Claude (latest: Opus 4.8 / Sonnet 4.6 / Haiku 4.5) for normalization/reasoning/RAG; small local embedder for retrieval over ESI/MIETIC snippets.
- **Product:** lightweight web UI (Streamlit or React + FastAPI). Decide at Phase 8 start based on demo-hosting needs.
- **Env:** primary = Kaggle GPU notebook; mirror code as a Kaggle Utility Script / dataset so notebooks `import src`.

## 4. Repo layout
```
docs/      PLAN.md · ARCHITECTURE.md · PROGRESS.md · DECISIONS.md · DATA_DICTIONARY.md
src/       data/ features/ models/ analysis/ ops/ llm/ serve/
notebooks/ 00_recon · 01_eda · 02_baselines · 03_fusion · 04_calibration · 05_fairness · 09_submission
configs/   config.yaml (paths, targets, splits, model params)
app/       triage copilot UI (Phase 8)
data/      raw/ interim/ processed/   (gitignored)
reports/   metrics tables, figures, writeup draft
tests/     pipeline + leakage + metrics unit tests
```

## 5. Working agreement (how an agent auto-advances this)
1. Read `PROGRESS.md` → find the current phase and the next unchecked task.
2. Do the smallest next task; keep code in `src/`, experiments in `notebooks/`.
3. Update `PROGRESS.md` (check the box, append a dated log line) and the harness task list.
4. Never advance past an **acceptance gate** until it is green.
5. Anything that needs the user (Kaggle account actions, hosting, irreversible/external steps) → stop and ask.
6. Be honest in the report: surface bias, missingness, and failure modes — the rubric rewards it.

## 6. Risks & open questions (resolve in Phase 0 unless noted)
- Exact deterioration label availability (ICU transfer / mortality fields present?).
- Whether MIETIC corpus ships with the competition data or needs separate access.
- Free-GPU time limits → keep transformer training checkpointed and resumable.
- Demo hosting choice (Kaggle can't host an always-on app) → decide at Phase 8.
