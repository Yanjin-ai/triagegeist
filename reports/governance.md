# Governance report — model_structured_v1 (text-blind structured)

Runs on the clinically-honest text-blind model; the leaderboard (text) is a lookup (see `INSIGHT_label_leakage.md`). Val split → calibration 50% / eval 50% (stratified, seed 42).

## P4 — Calibration (eval)

| metric                |   uncalibrated |   isotonic |
|:----------------------|---------------:|-----------:|
| confidence ECE        |         0.0223 |     0.0119 |
| class-wise ECE (mean) |         0.0124 |     0.0085 |
| P(critical) ECE       |         0.0017 |     0.0023 |

Class-wise ECE (isotonic): acuity1:0.0034, acuity2:0.0034, acuity3:0.0121, acuity4:0.0142, acuity5:0.0095

Reliability curve: `reports/figures/reliability.png`

## P4 — ESI-4 decision-boundary trade-off (eval, calibrated proba)

|   bias_acuity4 |    QWK |    acc |   recall_acuity4 |   recall_acuity1 |   recall_acuity2 |   undertriage(1,2) |   overtriage(4,5) |
|---------------:|-------:|-------:|-----------------:|-----------------:|-----------------:|-------------------:|------------------:|
|            1   | 0.9304 | 0.856  |           0.7903 |           0.8926 |           0.9782 |             0.0328 |            0.1538 |
|            1.1 | 0.9285 | 0.8518 |           0.8134 |           0.8926 |           0.9782 |             0.0328 |            0.143  |
|            1.2 | 0.9278 | 0.8492 |           0.8273 |           0.8926 |           0.9782 |             0.0328 |            0.1337 |
|            1.3 | 0.9265 | 0.8463 |           0.8349 |           0.8926 |           0.9782 |             0.0328 |            0.1302 |
|            1.4 | 0.9266 | 0.8465 |           0.8355 |           0.8926 |           0.9782 |             0.0328 |            0.1302 |
|            1.5 | 0.9259 | 0.8452 |           0.8407 |           0.8926 |           0.9782 |             0.0328 |            0.136  |
|            1.6 | 0.9253 | 0.8452 |           0.8575 |           0.8926 |           0.9782 |             0.0328 |            0.1426 |
|            1.7 | 0.9254 | 0.8453 |           0.8615 |           0.8926 |           0.9782 |             0.0328 |            0.1403 |
|            1.8 | 0.9252 | 0.8448 |           0.8644 |           0.8926 |           0.9782 |             0.0328 |            0.1387 |
|            1.9 | 0.9249 | 0.8445 |           0.8667 |           0.8926 |           0.9782 |             0.0328 |            0.1403 |
|            2   | 0.9228 | 0.84   |           0.8864 |           0.8926 |           0.9782 |             0.0328 |            0.1364 |

Baseline QWK=0.9304. Safety-constrained pick: bias_acuity4=1.0 → QWK=0.9304, recall_acuity4=0.7903, undertriage(1,2)=0.0328 (1/2 recall not reduced).

## P4 — Uncertainty (predictive entropy quartiles, eval)

| uncertainty_q   |    n |   error_rate |   undertriage_crit |   ambiguous_share |
|:----------------|-----:|-------------:|-------------------:|------------------:|
| Q1              | 1611 |       0.0019 |             0.0012 |            0.0062 |
| Q2              | 1442 |       0.0305 |             0.0125 |            0.0007 |
| Q3              | 1452 |       0.1674 |             0.0069 |            0      |
| Q4              | 1495 |       0.3839 |             0.0074 |            0.0007 |

## P4 — Conformal prediction (APS, distribution-free coverage)

Split-conformal on calibrated probabilities: returns an acuity *set* guaranteed to contain the true acuity with prob ≥ 1−α (marginal). Singletons → confident auto-triage; multi-class sets → principled human deferral (cf. conformal cost-aware clinical triage).

|   target_coverage |   empirical_coverage |   mean_set_size |   auto_rate_singletons |   defer_rate |
|------------------:|---------------------:|----------------:|-----------------------:|-------------:|
|              0.95 |               1      |           4.252 |                 0.0502 |       0.9498 |
|              0.9  |               0.9998 |           3.67  |                 0.1047 |       0.8953 |
|              0.8  |               0.9997 |           2.507 |                 0.1767 |       0.8233 |

At α=0.10: auto-triaged (singleton) **10.5%** with undertriage(1,2)=0.0016; deferred **89.5%** with undertriage(1,2)=0.0636 — deferral concentrates the residual risk.

> The sets are conservative (empirical coverage > target) *because physiology underdetermines the text-driven label* — APS honestly responds by deferring most cases, while the small auto-triaged set is highly reliable. This is the correct behaviour for a partly-unlearnable target, and it gives the ops layer a guaranteed-coverage human-in-the-loop rule.

## P5 — Subgroup fairness audit (eval, bootstrap 95% CI)


### by `language`

| language   |    n |   n_critical |    qwk |   undertriage_crit | undertri_CI95   |
|:-----------|-----:|-------------:|-------:|-------------------:|:----------------|
| Estonian   |  392 |           81 | 0.9216 |             0.0741 | [0.026, 0.132]  |
| Somali     |  258 |           58 | 0.9191 |             0.0517 | [0.000, 0.115]  |
| Swedish    |  474 |           93 | 0.9377 |             0.043  | [0.009, 0.083]  |
| English    |  603 |          128 | 0.9324 |             0.0391 | [0.008, 0.078]  |
| Russian    |  446 |           85 | 0.9308 |             0.0353 | [0.000, 0.076]  |
| Arabic     |  294 |           67 | 0.9311 |             0.0299 | [0.000, 0.079]  |
| Finnish    | 3260 |          686 | 0.9298 |             0.0248 | [0.013, 0.039]  |
| Other      |  273 |           52 | 0.9415 |             0.0192 | [0.000, 0.063]  |

### by `insurance_type`

| insurance_type   |    n |   n_critical |    qwk |   undertriage_crit | undertri_CI95   |
|:-----------------|-----:|-------------:|-------:|-------------------:|:----------------|
| unknown          |  163 |           34 | 0.9437 |             0.0588 | [0.000, 0.135]  |
| public           | 3639 |          748 | 0.9283 |             0.0348 | [0.023, 0.047]  |
| private          | 1475 |          320 | 0.9308 |             0.0312 | [0.015, 0.051]  |
| none             |  481 |           95 | 0.9367 |             0.0211 | [0.000, 0.048]  |
| military         |  242 |           53 | 0.9367 |             0.0189 | [0.000, 0.060]  |

### by `sex`

| sex   |    n |   n_critical |    qwk |   undertriage_crit | undertri_CI95   |
|:------|-----:|-------------:|-------:|-------------------:|:----------------|
| M     | 2881 |          558 | 0.9271 |             0.0394 | [0.023, 0.056]  |
| Other |  144 |           30 | 0.9232 |             0.0333 | [0.000, 0.120]  |
| F     | 2975 |          662 | 0.9336 |             0.0272 | [0.015, 0.039]  |

### by `age_group`

| age_group   |    n |   n_critical |    qwk |   undertriage_crit | undertri_CI95   |
|:------------|-----:|-------------:|-------:|-------------------:|:----------------|
| elderly     | 1655 |          341 | 0.9375 |             0.0411 | [0.021, 0.060]  |
| middle_aged | 2129 |          432 | 0.9203 |             0.037  | [0.019, 0.055]  |
| young_adult | 1758 |          373 | 0.9326 |             0.0241 | [0.008, 0.040]  |
| pediatric   |  458 |          104 | 0.9409 |             0.0192 | [0.000, 0.043]  |

### by `pain_missing`

|   pain_missing |    n |   n_critical |    qwk |   undertriage_crit | undertri_CI95       |
|---------------:|-----:|-------------:|-------:|-------------------:|:--------------------|
|              0 | 5133 |         1250 | 0.9411 |             0.0328 | [0.024, 0.044]      |
|              1 |  867 |            0 | 0.7786 |           nan      | no acuity-1/2 cases |

### by `arrival_mode`

| arrival_mode      |    n |   n_critical |    qwk |   undertriage_crit | undertri_CI95   |
|:------------------|-----:|-------------:|-------:|-------------------:|:----------------|
| transfer          |  609 |          117 | 0.922  |             0.0427 | [0.009, 0.082]  |
| helicopter        |  202 |           51 | 0.9303 |             0.0392 | [0.000, 0.091]  |
| walk-in           | 2923 |          598 | 0.9277 |             0.0385 | [0.024, 0.053]  |
| brought_by_family |  388 |           78 | 0.9205 |             0.0256 | [0.000, 0.070]  |
| ambulance         | 1643 |          361 | 0.939  |             0.0249 | [0.011, 0.043]  |
| police            |  235 |           45 | 0.9382 |             0      | [0.000, 0.000]  |

## P5 — Protective override (high-risk physiology never triaged 4/5)

- rows overridden (high-risk but predicted 4/5): **410**
- undertriage(1,2): 0.0328 → 0.0328
- overtriage(4,5): 0.1538 → 0.2751
- QWK: 0.9304 → 0.8950

> **Interpretation:** the physiology-based safety override *degrades* the metrics here. This is expected and informative: because the labels are a near-deterministic function of the complaint text (not physiology), high-risk vital-sign flags diverge from the labels. The override thus *quantifies the clinical risk* of trusting complaint-encoded triage — a real ED would want the physiology safety net even though it lowers agreement with these synthetic labels.

## P5 — Error taxonomy (eval)

|                          |    value |
|:-------------------------|---------:|
| n                        | 6000     |
| n_errors                 |  864     |
| error_rate               |    0.144 |
| adjacent_errors          |  854     |
| large_errors(>=2)        |   10     |
| undertriage_errors       |  450     |
| overtriage_errors        |  414     |
| physiology_silent_severe |   15     |

### 'Physiology-silent severe' examples (true acuity 1/2, predicted ≥3)

| chief_complaint_raw                                 |   y |   pred_acuity |   news2_score |   spo2 |   gcs_total |   shock_index | mental_status_triage   |
|:----------------------------------------------------|----:|--------------:|--------------:|-------:|------------:|--------------:|:-----------------------|
| significant self-neglect with medical complications |   2 |             3 |             6 |   95.3 |          15 |         1.164 | agitated               |
| stab wound abdomen stable with rigors               |   2 |             3 |             7 |   93.8 |          15 |         2.533 | alert                  |
| severe hypertensive emergency with rigors           |   2 |             3 |             3 |   87.5 |          15 |         0.475 | confused               |
| haemoptysis significant with rigors                 |   2 |             3 |             5 |   92.9 |          15 |         0.733 | drowsy                 |
| respiratory failure with hypoxia, constant          |   2 |             3 |             7 |   93.1 |          15 |         1.024 | alert                  |
| bacterial meningitis suspected                      |   2 |             3 |             2 |   99.1 |          15 |         0.962 | confused               |
| eczema herpeticum, onset today                      |   2 |             3 |             1 |   94   |          15 |         0.604 | confused               |
| epiglottitis suspected with associated nausea       |   2 |             3 |             6 |   94   |          15 |         1.042 | agitated               |

## P5 — OOD / novelty (test set)

- test complaints **novel** (unseen verbatim in train): **63 / 20000** (0.32%) — the only rows requiring true generalization rather than lookup.
- Operational OOD proxy: predictive entropy (above) — high-entropy rows concentrate errors and are the natural queue for human review / LLM normalization (Phase 7).
