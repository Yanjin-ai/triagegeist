"""Phase 3 v1 — cheap multimodal: tabular GBDT ⊕ TF-IDF text, late probability fusion.

    python -m src.models.fusion

Trains tabular + text branches, searches the fusion weight alpha on val
(p = alpha*p_tab + (1-alpha)*p_text) to maximize QWK, writes an ablation table
(baseline vs text-only vs fusion) and submission_fusion.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.metrics import confusion, evaluate, format_report, subgroup_report
from ..config import ROOT, load_config, processed_dir, reports_dir
from ..data.build import SCORED_TARGET, split
from ..features.text import TextBranch
from .tabular import fit_tabular, proba_to_pred

TEXT_COL = "chief_complaint_raw"


def _load_splits(cfg):
    out = processed_dir(cfg)
    if not (out / "train.parquet").exists():
        return split(cfg)
    return tuple(pd.read_parquet(out / f"{n}.parquet") for n in ("train", "val", "test"))


def search_alpha(p_tab, p_text, y_val, grid=None):
    grid = np.linspace(0, 1, 21) if grid is None else grid
    best = (0.0, -1.0)
    for a in grid:
        qwk = evaluate(y_val, proba_to_pred(a * p_tab + (1 - a) * p_text))["qwk"]
        if qwk > best[1]:
            best = (float(a), qwk)
    return best  # (alpha, qwk)


def main():
    cfg = load_config()
    train, val, test = _load_splits(cfg)
    yva = val[SCORED_TARGET].astype(int).values

    tab = fit_tabular(train, val, test, cfg)
    txt = TextBranch().fit(train[TEXT_COL], train[SCORED_TARGET].astype(int))
    p_text_val, p_text_test = txt.predict_proba(val[TEXT_COL]), txt.predict_proba(test[TEXT_COL])

    alpha, _ = search_alpha(tab.p_val, p_text_val, yva)
    p_fus_val = alpha * tab.p_val + (1 - alpha) * p_text_val
    p_fus_test = alpha * tab.p_test + (1 - alpha) * p_text_test

    models = {
        f"tabular ({tab.engine})": proba_to_pred(tab.p_val),
        "text-only (tfidf)": proba_to_pred(p_text_val),
        f"fusion (alpha={alpha:.2f})": proba_to_pred(p_fus_val),
    }
    metrics = {name: evaluate(yva, pred) for name, pred in models.items()}

    # ---- ablation table ----
    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name, "QWK": round(m["qwk"], 4), "acc": round(m["accuracy"], 4),
            "macroF1": round(m["macro_f1"], 4),
            "recall_acuity4": round(m["recall_per_class"][4], 4),
            "undertriage(1,2)": round(m["undertriage_critical"], 4),
        })
    table = pd.DataFrame(rows)
    print("\n=== Ablation (val) ===")
    print(table.to_string(index=False))
    for name in models:
        print(format_report(name, metrics[name]))
    print("\nFusion confusion (val):")
    print(confusion(yva, models[f"fusion (alpha={alpha:.2f})"]))

    # subgroup undertriage for the fusion model
    valdf = val.copy()
    valdf["pred_acuity"] = models[f"fusion (alpha={alpha:.2f})"]
    fair = subgroup_report(valdf, SCORED_TARGET, "pred_acuity", "language")
    print("\nFusion undertriage by language (val):")
    print(fair.to_string(index=False))

    rep = reports_dir(cfg)
    (rep / "ablation.md").write_text(
        "# Phase 3 v1 ablation (val)\n\n" + table.to_markdown(index=False)
        + f"\n\nBest fusion alpha = {alpha:.2f} (alpha*tabular + (1-alpha)*text)\n\n"
        + "## Fusion confusion (val)\n\n"
        + confusion(yva, models[f"fusion (alpha={alpha:.2f})"]).to_markdown()
        + "\n\n## Fusion undertriage by language (val)\n\n"
        + fair.to_markdown(index=False) + "\n"
    )

    sub = pd.DataFrame({
        cfg["data"]["id_col"]: test[cfg["data"]["id_col"]].values,
        SCORED_TARGET: proba_to_pred(p_fus_test).astype(int),
    })
    sub.to_csv(ROOT / "submission_fusion.csv", index=False)
    print(f"\ntext vocab size = {txt.vocab_size:,}; wrote submission_fusion.csv + reports/ablation.md")


if __name__ == "__main__":
    main()
