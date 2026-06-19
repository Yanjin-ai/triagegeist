# Triage Copilot UI (P8)

Self-contained static web app — three views: **Queue · Patient · Ops console**.
Fed by `data.js` (generated from the model pipeline), so it opens with no backend.

## Regenerate the data feed
```bash
python -m src.ops.console      # builds data/processed/ops_table.parquet
python -m src.ops.export_ui    # writes app/data.js + app/data.json
```

## Run it
- **Any static server** (avoids file:// quirks):
  ```bash
  python -m http.server 8077 --directory app    # then open http://localhost:8077
  ```
- Or just open `app/index.html` (it loads `data.js` via <script>, which works over file://).

## Notes
- Decision views use **`model_structured_v1`** (text-blind). Text/fusion (LB ≈ 1.0) is deliberately
  excluded from decisions — see `reports/INSIGHT_label_leakage.md`.
- The **AI assist** panel runs offline (templated). P7 will wire Claude (Opus 4.8) + guideline RAG
  when an API key is present, with this template as the fallback.
- Queue priority + senior-review recompute client-side from the policy weights, mirroring
  `src/ops/prioritize.py`.
