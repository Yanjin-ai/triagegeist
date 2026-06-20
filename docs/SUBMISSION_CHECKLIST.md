# Kaggle submission checklist

Triagegeist is a **judged hackathon**: "a well-documented Kaggle Notebook with a working proof-of-concept and
a thorough writeup is a complete and valid submission." Everything below is ready in this repo; the remaining
steps are the ones only you (account holder) can perform.

## Before you submit — confirm on the competition page
- [ ] **Deadline** — check the countdown on the Overview tab.
- [ ] **Submission mechanism** — Notebook submission vs writeup form vs CSV leaderboard (a `sample_submission.csv`
      exists, so a CSV track may be scored). Read **Overview / Rules / Submit**.
- [ ] **Scoring metric** — confirm it's Quadratic Weighted Kappa (Evaluation tab). We optimize QWK regardless.

## 1. Notebook (primary deliverable)
- [ ] On Kaggle: **New Notebook** → upload `notebooks/triagegeist_submission.ipynb`.
- [ ] Add the **Triagegeist** competition as a data source (it mounts at `/kaggle/input/triagegeist`; the
      notebook auto-detects this path).
- [ ] **Run All** → confirm it completes top-to-bottom (it is self-contained; no `src/` import needed).
      LightGBM is auto-selected on Kaggle; it writes `submission.csv` in the last cell.
- [ ] **Save Version** (Save & Run All).

## 2. Writeup
- [ ] Paste [`docs/WRITEUP.md`](WRITEUP.md) into the notebook's intro markdown **or** the competition's writeup
      field. It is already mapped to the four judging axes (clinical relevance · technical quality ·
      insight/honesty · novelty/impact) and leads with the leakage finding.
- [ ] Link the GitHub repo and the demo instructions for the proof-of-concept (UI + live inference).

## 3. Leaderboard CSV (if a scored track exists)
- [ ] **Honest model:** submit `submission.csv` (text-blind structured, QWK ≈ 0.93).
- [ ] **Or LB formality:** `submission_fusion.csv` (text/fusion, ≈ 1.0 — but only because of the documented
      label leakage; disclose this in the writeup).
- [ ] Regenerate any submission with: `python -m src.models.gbdt` / `python -m src.models.fusion`.

## 4. Proof-of-concept (strengthens the submission)
- [ ] Mention the interactive prototype: dashboard (`python -m http.server 8077 --directory app`) and live
      inference (`python -m src.serve.app`). See [`DEMO.md`](DEMO.md) for the click path and talking points.
- [ ] Optional: record a short screen capture of the three views + a live prediction + an override → audit log.

## 5. Reproducibility & honesty (judge trust)
- [ ] Repo includes `MODEL_CARD.md`, `DATA_STATEMENT.md`, pinned `requirements.txt`, fixed seeds, `pytest` suite.
- [ ] Data is git-ignored (redistribution prohibited) — the notebook downloads via `kagglehub`.
- [ ] Limitations are stated plainly (leakage, synthetic data, TRL 3–4, no external validation).

## What's already done in this repo
✅ self-contained notebook · ✅ rubric-mapped writeup · ✅ model card + data statement · ✅ comparison vs
mature products · ✅ governance/decision-chain · ✅ 3 submissions · ✅ static UI + live inference · ✅ 12/12 tests.
