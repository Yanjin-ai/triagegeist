# Demo walkthrough (for judges / screenshots)

A 3-minute tour of the triage copilot. The UI runs **without any data download** — `app/data.js`
ships pre-generated.

## Start it
```bash
python -m http.server 8077 --directory app   # then open http://localhost:8077
```
Or just open `app/index.html` directly.

---

## The 60-second story
> The acuity label in this synthetic data is near-deterministically encoded in the chief-complaint text,
> so a text model scores QWK ≈ 1.0 and the leaderboard is a lookup. We refuse to call that clinical skill.
> The app makes **decisions with the text-blind structured model** and surrounds it with calibration,
> fairness, and operations. The banner at the top states exactly this.

---

## Click path

### 1) Queue (triage waiting room)
- Patients are **priority-ranked**; ESI is color-coded (red=1 … grey=5).
- Toggle **`safety_first` ↔ `throughput`** (top-left) — watch the ordering shift: safety_first pushes
  high `P(critical)` / high-uncertainty cases up; throughput lets long waits age cases up faster.
- Tick **high-risk / high-uncertainty / waited ≥3h** to filter.
- A red **REVIEW** tag marks high-uncertainty potentially-critical cases for a senior.
- **Click any row** → Patient.

*Talking point:* priority is a transparent weighted score (no black-box RL) — see `src/ops/prioritize.py`.

### 2) Patient card
- **Left** — intake & vitals (note `pain: not assessed` is shown, not hidden).
- **Middle** — **calibrated** acuity probability bars, admission risk, predicted LOS, resource bucket,
  uncertainty.
- **Right — AI assist (offline)** — normalized complaint + **red-flag detection**, a grounded case summary,
  "why this priority", and whether the **physiology safety-net override would trigger**.

*Talking point:* pick a high-acuity case (e.g. a "status asthmaticus / unresponsive" patient) to show the
red flags and the override note. This is where Claude + guideline RAG plugs in when an API key is set.

### 3) Ops console
- KPI cards: **expected admissions**, admission rate, mean LOS, **bed pressure** (turns red >1.0).
- **Resource bucket mix** across the full cohort.
- **Flow by shift** (labeled *flow, not census* — we're careful about operational semantics).

*Talking point:* this is the "optimize triage decisions" half of the competition — single-patient risk
becomes department-level operations.

---

## If asked "but what's your leaderboard score?"
Two submissions exist: `submission_fusion.csv` (text, ≈1.0 — the lookup) and `submission.csv`
(text-blind honest core, QWK ≈ 0.93). The interesting work is the second; the first only proves the leakage.
See `reports/INSIGHT_label_leakage.md`.

## Supporting evidence to show
- `reports/governance.md` — calibration (ECE 0.022→0.012), fairness CIs, error taxonomy.
- `reports/figures/reliability.png` — reliability curve before/after calibration.
- `notebooks/triagegeist_submission.ipynb` — the whole story, runnable top-to-bottom.
