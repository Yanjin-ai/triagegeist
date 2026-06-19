"""P6 — ops console orchestrator.

Builds the per-patient operational object for the incoming (test) cohort and the
ED-level views, then writes reports/ops_console.md + data/processed/ops_table.parquet
(consumed by the Phase 8 UI).

    python -m src.ops.console
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.calibration import IsotonicOVR
from ..analysis.uncertainty import entropy
from ..config import ROOT, load_config, processed_dir, reports_dir
from ..data.build import SCORED_TARGET
from ..models.aux_heads import fit_aux
from ..models.structured import fit_tabular, proba_to_pred
from .ed_load import ed_load_summary, scenario
from .prioritize import POLICIES, needs_senior_review, rank_queue
from .resource_map import assign_bucket

CLASSES = np.array([1, 2, 3, 4, 5])
SNAPSHOT_N = 50
N_BEDS = 30


def _load():
    out = processed_dir(load_config())
    return tuple(pd.read_parquet(out / f"{n}.parquet") for n in ("train", "val", "test"))


def build_ops_table(cfg) -> pd.DataFrame:
    train, val, test = _load()
    tab = fit_tabular(train, val, test, cfg)
    iso = IsotonicOVR(CLASSES).fit(tab.p_val, val[SCORED_TARGET].astype(int).to_numpy())
    p_test = iso.transform(tab.p_test)
    aux = fit_aux(train, val, test, cfg)

    acuity = proba_to_pred(p_test)
    p_crit = p_test[:, 0] + p_test[:, 1]
    ent = entropy(p_test)
    bucket = assign_bucket(acuity, aux.p_admit_test, aux.los_test)

    rng = np.random.default_rng(42)
    # simulated current wait (demo only — real-time input in deployment); longer for low acuity
    wait = np.round(rng.exponential(scale=0.4 * acuity), 2)

    idc = cfg["data"]["id_col"]
    df = pd.DataFrame({
        idc: test[idc].values,
        "acuity": acuity, "p_critical": p_crit.round(4),
        "p_admit": aux.p_admit_test.round(4), "pred_los": aux.los_test.round(2),
        "entropy": ent.round(4), "bucket": bucket, "wait_hours": wait,
        "senior_review": needs_senior_review(acuity, p_crit, ent),
    })
    for j, c in enumerate(CLASSES):
        df[f"p_acuity{c}"] = p_test[:, j].round(4)
    # carry a few context cols for the UI/patient card
    for col in ["age", "sex", "language", "insurance_type", "arrival_mode", "shift",
                "chief_complaint_raw", "chief_complaint_system", "news2_score", "spo2",
                "heart_rate", "respiratory_rate", "systolic_bp", "gcs_total",
                "pain_score", "mental_status_triage", "num_comorbidities"]:
        if col in test:
            df[col] = test[col].values
    return df


def main():
    cfg = load_config()
    df = build_ops_table(cfg)
    df.to_parquet(processed_dir(cfg) / "ops_table.parquet")

    snap = df.sample(SNAPSHOT_N, random_state=42).reset_index(drop=True)
    out = []
    def w(s=""): out.append(s)

    w("# ED operations console (prototype) — text-blind structured model\n")
    w(f"Incoming cohort = test set ({len(df):,}). Snapshot = {SNAPSHOT_N} sampled patients "
      "as a waiting-room view. `wait_hours` is simulated for the demo (a real-time input in deployment).\n")
    w("> Honesty carryover (P5): buckets/priority encode physiology+outcome logic that can diverge "
      "from the text-determined acuity labels; in a real ED that divergence is the *point*.\n")

    # ranked queue (both policies, top 12 of snapshot)
    for pol in POLICIES:
        rq = rank_queue(snap, policy=pol).head(12)
        show = rq[["patient_id", "acuity", "p_critical", "p_admit", "entropy",
                   "bucket", "wait_hours", "priority", "senior_review"]]
        w(f"## Ranked queue — policy = `{pol}` (top 12 of snapshot)\n")
        w(show.to_markdown(index=False)); w("")

    # ED load on snapshot
    w(f"## ED load summary (snapshot, n_beds={N_BEDS})\n")
    s = ed_load_summary(snap, n_beds=N_BEDS)
    w(pd.Series(s, dtype=object).to_frame("value").to_markdown()); w("")

    # scenarios by shift
    w("## Scenario flow by shift (full cohort — flow counts, not point-in-time census)\n")
    rows = []
    for sh in [v for v in df.get("shift", pd.Series(dtype=object)).dropna().unique()]:
        rows.append(scenario(df, (df["shift"] == sh).to_numpy(), sh, n_beds=N_BEDS))
    if rows:
        cols = ["scenario", "n_patients", "expected_admissions", "admission_rate",
                "high_uncertainty_critical", "mean_pred_los_h"]
        w(pd.DataFrame(rows)[cols].to_markdown(index=False)); w("")

    (reports_dir(cfg) / "ops_console.md").write_text("\n".join(out))
    print(f"buckets: {df['bucket'].value_counts().to_dict()}")
    print(f"cohort expected admissions: {df['p_admit'].sum():.0f} / {len(df)} "
          f"({df['p_admit'].mean()*100:.1f}%); mean pred LOS {df['pred_los'].mean():.2f}h")
    print(f"high-uncertainty-critical (senior review) in snapshot: "
          f"{int(rank_queue(snap)['senior_review'].sum())}/{SNAPSHOT_N}")
    print("wrote reports/ops_console.md + data/processed/ops_table.parquet")


if __name__ == "__main__":
    main()
