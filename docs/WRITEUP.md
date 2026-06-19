# Triagegeist — Write-up

**An honest ED triage decision-support stack: predict → calibrate → audit → operate.**

> One-paragraph summary. In this synthetic dataset the triage acuity label is a *near-deterministic
> function of the chief-complaint text* — so a text model scores QWK ≈ 1.0 and the public leaderboard
> is essentially a lookup table. Rather than pass that off as clinical skill, we treat it as the central
> dataset finding and build the **clinically honest, text-blind structured-physiology model** (QWK ≈ 0.93),
> then wrap it in calibration, fairness auditing, and an **ED operations console**. The deliverable is a
> reproducible pipeline (`src/`), a self-contained Kaggle notebook, and an interactive triage-copilot UI (`app/`).

---

## 1. Clinical relevance
Real triage acuity reflects physiology and context, not a fixed phrase→score table. We therefore make the
**text-blind structured model the decision engine** — it uses vitals, NEWS2/shock-index, demographics,
arrival context, and comorbidity history, the information a nurse actually weighs. On this model the things
that matter clinically have teeth:
- **Undertriage protection.** Acuity-1 is only 4% of cases and the dangerous error is *under*-triage
  (assigning a less-urgent level to a critical patient). We measure undertriage on truly-critical (acuity 1/2)
  cases directly and keep acuity-1/2 recall high (≈0.95/0.97).
- **Calibrated risk.** Operational decisions need trustworthy probabilities, not just a class — we calibrate
  and report ECE, and expose `P(critical)` and admission risk.
- **Missingness is clinical signal.** `pain_score=-1` ("not assessed", 13.9%) and co-missing BP are modeled
  explicitly with indicator flags rather than silently imputed away.

## 2. Technical quality
- **Leakage-safe pipeline** (`src/data`, `src/features`): 4-table join on `patient_id`, a single shared
  `transform()` (no train/serve skew, enforced by tests), stratified split (valid because `patient_id`
  is per-visit unique and train ∩ test = ∅), and an explicit outcome blocklist (`disposition`, `ed_los_hours`
  never used as features).
- **Engine-pluggable GBDT** (`model_structured_v1`): LightGBM on Kaggle, sklearn HistGradientBoosting locally,
  identical logic/seed. Balanced class weights for the rare safety-critical class.
- **Governance** (`src/analysis`): isotonic calibration (confidence ECE **0.022 → 0.012**; class-wise ECE
  0.012 → 0.009), predictive-entropy uncertainty (highest-entropy quartile = 38% error vs 0.2% lowest →
  a natural human-review queue), an ESI-4 decision-boundary trade-off grid (recall 0.79→0.89 with no loss
  to acuity-1/2 recall), **bootstrap-CI subgroup fairness**, and an error taxonomy.
- **Tests:** leakage, transform determinism, frozen text vocab, fusion reproducibility, ops logic — all green.

## 3. Insight & honesty (the spine)
The headline finding (`reports/INSIGHT_label_leakage.md`):
- **99.7%** of unique complaint phrases map to exactly one acuity; **99.8%** of test complaints appear
  verbatim in train; only **8** core symptoms are genuinely ambiguous.
- Consequence: TF-IDF/​fusion ⇒ QWK ≈ 1.0 — a **property of the data-generating process**, not a model win.

We make three honest moves a score-chaser would not:
1. We **do not** put the ≈1.0 text model in the decision view; it is shown only as a dataset insight.
2. We quantify a **negative result**: a physiology-based protective override (high-risk vitals never triaged
   4/5) *lowers* agreement (QWK 0.93→0.895) precisely because the labels are text-determined — evidence of the
   risk of trusting complaint-encoded triage, which a real ED would still want guarded.
3. We surface **fairness flags with uncertainty**: undertriage is higher for Estonian-speaking (7.4%
   [2.6–13.2]) and `insurance=unknown` (5.9%) vs Finnish (2.5%); the `pain_missing=1` group's QWK collapses
   to 0.78. We report sample sizes and CIs so differences are not over-read.

## 4. Novelty & impact
We go beyond acuity classification to a deployable **triage intelligence stack**:
- **Auxiliary heads** (admission risk, ED-LOS) trained text-blind to drive operations.
- **Resource buckets** (resuscitation / high-freq monitoring / standard / fast-track / likely-discharge) and a
  **transparent queue-priority policy** = f(urgency, P(critical), uncertainty, wait), with `safety_first` vs
  `throughput` variants and a senior-review flag.
- **ED operations console**: expected admissions, bed-pressure, per-shift flow.
- An interactive **triage-copilot UI** (`app/`) with three views and an offline **AI-assist** panel
  (complaint normalization, red-flag detection, grounded case summary), ready to wire to Claude for guideline
  RAG when an API key is present.

---

## Reproduce
```bash
pip install -r requirements.txt
python -m src.data.build          # join + stratified split
python -m src.models.gbdt         # text-blind baseline -> submission.csv
python -m src.models.fusion       # text/fusion ablation (shows the leakage)
python -m src.analysis.governance # calibration + fairness + error taxonomy -> reports/governance.md
python -m src.ops.console         # ops layer -> reports/ops_console.md + ops_table.parquet
python -m src.ops.export_ui       # -> app/data.js ; open app/index.html
python scripts/build_notebook.py  # -> notebooks/triagegeist_submission.ipynb
pytest -q                         # all tests
```
See `docs/MODEL_CARD.md` and `docs/DATA_STATEMENT.md`. **Synthetic data; not for clinical use.**
