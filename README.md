# Triagegeist — an honest ED triage intelligence stack

A text-blind **risk-prediction core** with **calibration, fairness auditing, and an ED operations console**
as the landing point — an emergency-department triage decision-support copilot. Built for the
[Triagegeist Kaggle competition](https://www.kaggle.com/competitions/triagegeist).

**Not** a medical Q&A bot, and **not** a leaderboard chase. Predict → calibrate → audit → operate.

## The headline finding
In this synthetic dataset the acuity label is a **near-deterministic function of the chief-complaint text**
(99.7% of phrases map to one acuity; 99.8% of test phrases are seen verbatim in train). So a text model
scores **QWK ≈ 1.0 and the public leaderboard is a lookup table**. We treat that as the central *insight*,
not a result, and build the **clinically honest text-blind structured model (QWK ≈ 0.93)** as the decision
engine. → [`reports/INSIGHT_label_leakage.md`](reports/INSIGHT_label_leakage.md)

## What's here
| Layer | Code | Output |
|---|---|---|
| Pipeline (leakage-safe) | `src/data`, `src/features` | `data/processed/` |
| Text-blind core + aux heads | `src/models` (`structured`, `gbdt`, `fusion`, `aux_heads`) | `submission*.csv` |
| Governance | `src/analysis` (calibration, uncertainty, fairness, error taxonomy) | `reports/governance.md` |
| Operations | `src/ops` (resource_map, prioritize, ed_load, console) | `reports/ops_console.md`, `ops_table.parquet` |
| Triage-copilot UI | `app/` (`index.html` + generated `data.js`) | 3 views: Queue · Patient · Ops |
| Submission | `notebooks/triagegeist_submission.ipynb` | self-contained, runs on Kaggle |

## Judge-facing docs
- **[`docs/WRITEUP.md`](docs/WRITEUP.md)** — rubric-mapped write-up (clinical relevance · technical quality ·
  insight/honesty · novelty/impact).
- **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)** · **[`docs/DATA_STATEMENT.md`](docs/DATA_STATEMENT.md)**
- `docs/PLAN.md` · `docs/PROGRESS.md` · `docs/ARCHITECTURE.md` · `docs/DECISIONS.md` · `docs/DATA_DICTIONARY.md`

## Reproduce
```bash
pip install -r requirements.txt
python -m src.data.download        # download competition data (Kaggle auth) -> ./triagegeist/
python -m src.data.build           # join + stratified leakage-safe split
python -m src.models.gbdt          # text-blind baseline      -> submission.csv  (QWK ~0.93)
python -m src.models.fusion        # text/fusion ablation      -> shows the leakage (QWK ~1.0)
python -m src.analysis.governance  # calibration + fairness    -> reports/governance.md + reliability.png
python -m src.ops.console          # ops layer                 -> reports/ops_console.md + ops_table.parquet
python -m src.ops.export_ui        # UI feed                   -> app/data.js
python scripts/build_notebook.py   # submission notebook       -> notebooks/triagegeist_submission.ipynb
pytest -q                          # all tests green
```

## Run the UI
```bash
python -m http.server 8077 --directory app   # open http://localhost:8077
```
Three views (Queue · Patient · Ops console), policy toggle (`safety_first` / `throughput`), an "honest core"
banner, and an offline AI-assist panel. Decisions use the text-blind model only.

> **Data:** synthetic, non-commercial research use, redistribution prohibited — **not for clinical use**. The
> data is git-ignored; download via `kagglehub` under the competition rules.
