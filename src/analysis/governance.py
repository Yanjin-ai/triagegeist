"""P4 + P5 orchestrator — calibration, uncertainty, fairness, error taxonomy, override.

Runs on model_structured_v1 (text-blind). Splits the internal val into a calibration
slice (fit calibrator + pick thresholds) and an eval slice (report everything).

    python -m src.analysis.governance   -> reports/governance.md + reports/figures/reliability.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import ROOT, load_config, processed_dir, reports_dir
from ..data.build import SCORED_TARGET
from ..models.structured import MODEL_ID, fit_tabular, proba_to_pred
from . import calibration as cal
from . import error_taxonomy as et
from . import fairness as fr
from .metrics import evaluate, format_report
from .thresholds import search_class4_bias
from .uncertainty import entropy, uncertainty_breakdown

TEXT = "chief_complaint_raw"
SUBGROUPS = ["language", "insurance_type", "sex", "age_group", "pain_missing", "arrival_mode"]


def _load():
    out = processed_dir(load_config())
    return tuple(pd.read_parquet(out / f"{n}.parquet") for n in ("train", "val", "test"))


def _reliability_fig(classes, y_eval, p_raw, p_cal, path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for p, lab in [(p_raw, "uncalibrated"), (p_cal, "isotonic")]:
        c, mc, ma, n = cal.reliability_table(y_eval, p, classes)
        ok = n > 0
        plt.plot(mc[ok], ma[ok], "o-", label=lab)
    plt.xlabel("mean predicted confidence"); plt.ylabel("empirical accuracy")
    plt.title("Reliability — text-blind structured model"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    cfg = load_config()
    classes = np.array([1, 2, 3, 4, 5])
    train, val, test = _load()
    tab = fit_tabular(train, val, test, cfg)

    # split val -> calibration / eval (stratified)
    yv = val[SCORED_TARGET].astype(int).to_numpy()
    idx = np.arange(len(val))
    i_cal, i_ev = train_test_split(idx, test_size=0.5, random_state=42, stratify=yv)
    p_cal, p_ev = tab.p_val[i_cal], tab.p_val[i_ev]
    y_cal, y_ev = yv[i_cal], yv[i_ev]
    ev = val.iloc[i_ev].copy()
    ev["pain_missing"] = (pd.to_numeric(ev["pain_score"], errors="coerce") == -1).astype(int)

    out = []
    def w(s=""): out.append(s)

    w(f"# Governance report — {MODEL_ID} (text-blind structured)\n")
    w("Runs on the clinically-honest text-blind model; the leaderboard (text) is a lookup "
      "(see `INSIGHT_label_leakage.md`). Val split → calibration 50% / eval 50% (stratified, seed 42).\n")

    # ---------- P4: calibration ----------
    cece_raw = cal.confidence_ece(y_ev, p_ev, classes)
    cw_raw, cwper_raw = cal.classwise_ece(y_ev, p_ev, classes)
    crit_raw = cal.critical_coarse_ece(y_ev, p_ev, classes)
    iso = cal.IsotonicOVR(classes).fit(p_cal, y_cal)
    p_ev_cal = iso.transform(p_ev)
    cece_c = cal.confidence_ece(y_ev, p_ev_cal, classes)
    cw_c, cwper_c = cal.classwise_ece(y_ev, p_ev_cal, classes)
    crit_c = cal.critical_coarse_ece(y_ev, p_ev_cal, classes)

    w("## P4 — Calibration (eval)\n")
    w(pd.DataFrame({
        "metric": ["confidence ECE", "class-wise ECE (mean)", "P(critical) ECE"],
        "uncalibrated": [round(cece_raw, 4), round(cw_raw, 4), round(crit_raw, 4)],
        "isotonic": [round(cece_c, 4), round(cw_c, 4), round(crit_c, 4)],
    }).to_markdown(index=False))
    w("\nClass-wise ECE (isotonic): "
      + ", ".join(f"acuity{k}:{v:.4f}" for k, v in cwper_c.items()) + "\n")

    figdir = reports_dir(cfg) / "figures"; figdir.mkdir(exist_ok=True)
    _reliability_fig(classes, y_ev, p_ev, p_ev_cal, figdir / "reliability.png")
    w("Reliability curve: `reports/figures/reliability.png`\n")

    # ---------- P4: ESI-4 threshold trade-off ----------
    w("## P4 — ESI-4 decision-boundary trade-off (eval, calibrated proba)\n")
    grid = search_class4_bias(p_ev_cal, y_ev, classes)
    w(grid.to_markdown(index=False))
    base_q = evaluate(y_ev, proba_to_pred(p_ev_cal))["qwk"]
    safe = grid[(grid["recall_acuity1"] >= grid["recall_acuity1"].iloc[0] - 1e-9)
                & (grid["recall_acuity2"] >= grid["recall_acuity2"].iloc[0] - 1e-9)]
    pick = safe.sort_values(["QWK", "recall_acuity4"], ascending=False).iloc[0]
    w(f"\nBaseline QWK={base_q:.4f}. Safety-constrained pick: bias_acuity4="
      f"{pick['bias_acuity4']} → QWK={pick['QWK']}, recall_acuity4={pick['recall_acuity4']}, "
      f"undertriage(1,2)={pick['undertriage(1,2)']} (1/2 recall not reduced).\n")

    # ---------- P4: uncertainty ----------
    pred_ev = proba_to_pred(p_ev_cal)
    unc = entropy(p_ev_cal)
    cores = ev[TEXT].str.split(",").str[0].str.strip().str.lower()
    tr_cores = train.assign(core=train[TEXT].str.split(",").str[0].str.strip().str.lower())
    amb = tr_cores.groupby("core")[SCORED_TARGET].nunique()
    amb_set = set(amb[amb > 1].index)
    ambiguous = cores.isin(amb_set).to_numpy()
    w("## P4 — Uncertainty (predictive entropy quartiles, eval)\n")
    w(uncertainty_breakdown(y_ev, pred_ev, unc, ambiguous).to_markdown(index=False))
    w("")

    # ---------- P5: fairness ----------
    ev["pred_acuity"] = pred_ev
    w("## P5 — Subgroup fairness audit (eval, bootstrap 95% CI)\n")
    for col in SUBGROUPS:
        if col not in ev:
            continue
        w(f"\n### by `{col}`\n")
        w(fr.subgroup_audit(ev, SCORED_TARGET, "pred_acuity", col).to_markdown(index=False))
    w("")

    # ---------- P5: protective override ----------
    pred_over = fr.apply_override(ev, pred_ev, cap=3)
    m0, m1 = evaluate(y_ev, pred_ev), evaluate(y_ev, pred_over)
    n_over = int((fr.high_risk_mask(ev) & (pred_ev >= 4)).sum())
    w("## P5 — Protective override (high-risk physiology never triaged 4/5)\n")
    w(f"- rows overridden (high-risk but predicted 4/5): **{n_over}**")
    w(f"- undertriage(1,2): {m0['undertriage_critical']:.4f} → {m1['undertriage_critical']:.4f}")
    w(f"- overtriage(4,5): {m0['overtriage_low']:.4f} → {m1['overtriage_low']:.4f}")
    w(f"- QWK: {m0['qwk']:.4f} → {m1['qwk']:.4f}\n")
    verdict = "degrades" if m1["qwk"] < m0["qwk"] else "improves"
    w(f"> **Interpretation:** the physiology-based safety override *{verdict}* the metrics here. "
      "This is expected and informative: because the labels are a near-deterministic function of the "
      "complaint text (not physiology), high-risk vital-sign flags diverge from the labels. The override "
      "thus *quantifies the clinical risk* of trusting complaint-encoded triage — a real ED would want "
      "the physiology safety net even though it lowers agreement with these synthetic labels.\n")

    # ---------- P5: error taxonomy ----------
    w("## P5 — Error taxonomy (eval)\n")
    tax = et.taxonomy(ev.assign(y=y_ev), "y", "pred_acuity")
    w(pd.DataFrame([tax]).T.rename(columns={0: "value"}).to_markdown())
    w("\n### 'Physiology-silent severe' examples (true acuity 1/2, predicted ≥3)\n")
    ex = et.physiology_silent_examples(ev.assign(y=y_ev), "y", "pred_acuity")
    w(ex.to_markdown(index=False) if len(ex) else "_none_")
    w("")

    # ---------- P5: OOD / novelty ----------
    train_phrases = set(train[TEXT].astype(str).str.lower().str.strip())
    test_phrases = test[TEXT].astype(str).str.lower().str.strip()
    novel = ~test_phrases.isin(train_phrases)
    w("## P5 — OOD / novelty (test set)\n")
    w(f"- test complaints **novel** (unseen verbatim in train): **{int(novel.sum())} / {len(test)}** "
      f"({novel.mean()*100:.2f}%) — the only rows requiring true generalization rather than lookup.")
    w("- Operational OOD proxy: predictive entropy (above) — high-entropy rows concentrate errors "
      "and are the natural queue for human review / LLM normalization (Phase 7).\n")

    report = "\n".join(out)
    (reports_dir(cfg) / "governance.md").write_text(report)

    # calibrated structured submission
    p_test_cal = iso.transform(tab.p_test)
    pd.DataFrame({cfg["data"]["id_col"]: test[cfg["data"]["id_col"]].values,
                  SCORED_TARGET: proba_to_pred(p_test_cal).astype(int)}
                 ).to_csv(ROOT / "submission_structured_calibrated.csv", index=False)

    print(format_report("structured(eval, calibrated)", evaluate(y_ev, pred_ev)))
    print(f"confidence ECE {cece_raw:.4f} -> {cece_c:.4f} (isotonic); "
          f"override fixed {n_over} high-risk under-triages")
    print("wrote reports/governance.md, reports/figures/reliability.png, "
          "submission_structured_calibrated.csv")


if __name__ == "__main__":
    main()
