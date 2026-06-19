# Phase 3 v1 ablation (val)

| model               |    QWK |    acc |   macroF1 |   recall_acuity4 |   undertriage(1,2) |
|:--------------------|-------:|-------:|----------:|-----------------:|-------------------:|
| tabular (hgb)       | 0.9295 | 0.8514 |    0.8702 |           0.7362 |             0.022  |
| text-only (tfidf)   | 0.9994 | 0.9988 |    0.9962 |           1      |             0.0028 |
| fusion (alpha=0.30) | 1      | 0.9999 |    0.9997 |           1      |             0.0004 |

Best fusion alpha = 0.30 (alpha*tabular + (1-alpha)*text)

## Fusion confusion (val)

|       |   pred1 |   pred2 |   pred3 |   pred4 |   pred5 |
|:------|--------:|--------:|--------:|--------:|--------:|
| true1 |     482 |       1 |       0 |       0 |       0 |
| true2 |       0 |    2016 |       0 |       0 |       0 |
| true3 |       0 |       0 |    4338 |       0 |       0 |
| true4 |       0 |       0 |       0 |    3453 |       0 |
| true5 |       0 |       0 |       0 |       0 |    1710 |

## Fusion undertriage by language (val)

| language   |    n |     qwk |   accuracy |   undertriage_critical |
|:-----------|-----:|--------:|-----------:|-----------------------:|
| Finnish    | 6578 | 0.99993 |   0.999848 |            0.000720981 |
| Arabic     |  581 | 1       |   1        |            0           |
| English    | 1203 | 1       |   1        |            0           |
| Estonian   |  740 | 1       |   1        |            0           |
| Other      |  570 | 1       |   1        |            0           |
| Russian    |  887 | 1       |   1        |            0           |
| Somali     |  489 | 1       |   1        |            0           |
| Swedish    |  952 | 1       |   1        |            0           |
