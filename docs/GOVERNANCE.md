# Governance by design — how the product embodies its decision chain

A model that *claims* to be trustworthy is not the same as a product that *shows* its reasoning and lets
multiple parties supervise it. This system makes three things first-class and visible in the live
inference surface (`src/serve/`), not just buried in reports.

## 1. The decision chain is explicit and traceable
Every prediction returns an ordered `decision_trace` (rendered as a numbered list in the UI):
1. **intake** — N/of-M features provided, K imputed
2. **data processing** — pain sentinel flagged (not silently zeroed); outcome columns excluded
3. **model (text-blind structured)** — acuity distribution
4. **calibration** — isotonic OVR (val ECE ≈ 0.012)
5. **conformal set** — set @ 1−α coverage guarantee
6. **safety override** — high-risk physiology caps the suggestion at ESI ≤3 (applied, with the raw
   `model_acuity` kept visible)
7. **resource mapping** — operational bucket
8. **prioritization** — transparent weighted score (policy)
9. **disposition** — auto-suggest (clinician confirms) vs DEFER to human

Nothing is a black box: the chain shows *what transformed the input into the recommendation*.

## 2. Data processing is surfaced as provenance
`data_provenance` reports, per decision:
- how many model features were **provided vs imputed**;
- the **pain-sentinel** handling (`-1`/None → flagged, not silently zero);
- an explicit **no-leakage attestation**: the model is text-blind and never sees `disposition`/`ed_los_hours`
  or the complaint text that encodes the label.

This lets a reviewer judge *how much to trust* a given decision (a 13/61-feature intake is weaker evidence
than a complete one).

## 3. The decision is multi-party supervisable
`oversight` attaches the parties who must/should review, each tied to a concrete signal from our own
analysis — so oversight is *earned by evidence*, not decorative:

| Party | Trigger | Role |
|---|---|---|
| **Triage nurse** | always | primary decision-maker — confirms or overrides |
| **Senior physician** | conformal defer · P(critical)≥0.5 · ESI ≤2 · override fired | second review / co-sign |
| **Quality / equity auditor** | subgroup our audit flagged (e.g. language=Estonian, insurance=unknown, pain not assessed) | bias oversight |
| **Operations** | bucket consumes inpatient capacity | capacity oversight |

### Accountability: an append-only audit log
`src/serve/audit.py` writes one JSON line per **decision** and per **human action** (confirm / override +
reason), linked by id. `GET /audit` returns the recent log. This is the minimal substrate real CDS needs:
a record of what the model suggested and what a human did with it.

```
POST /predict  → decision logged  (#1: model_acuity 4 → acuity 3, defer, reviewers=[nurse, senior])
POST /override → human action #2  (nurse, "override", reason="clinical gestalt: sepsis, escalate")
```

## Honest scope
This embodies the *governance design* in the product; it is **not** production identity/auth, a secure
tamper-proof store, or HIPAA-grade access control. On synthetic data it demonstrates the pattern a real
deployment would harden. See `docs/CLINICAL_READINESS` notes in `docs/COMPARISON.md` for the deployment gap.
