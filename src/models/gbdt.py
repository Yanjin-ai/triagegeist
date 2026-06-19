"""Phase 2 — tabular GBDT baseline for the scored target (triage_acuity).

Thin wrapper over src.models.tabular.fit_tabular so the baseline and the fusion
model share identical tabular training (no skew).

    python -m src.models.gbdt   # train, eval on val, write reports/ + submission.csv
"""
from __future__ import annotations

import pandas as pd

from ..analysis.metrics import confusion, evaluate, format_report, subgroup_report
from ..config import ROOT, load_config, processed_dir, reports_dir
from ..data.build import SCORED_TARGET, split
from .tabular import fit_tabular, proba_to_pred


def _load_splits(cfg):
    out = processed_dir(cfg)
    if not (out / "train.parquet").exists():
        print("processed splits missing -> building...")
        return split(cfg)
    return tuple(pd.read_parquet(out / f"{n}.parquet") for n in ("train", "val", "test"))


def main():
    cfg = load_config()
    train, val, test = _load_splits(cfg)
    yva = val[SCORED_TARGET].astype(int).values

    tab = fit_tabular(train, val, test, cfg)
    pred_va, pred_te = proba_to_pred(tab.p_val), proba_to_pred(tab.p_test)

    m = evaluate(yva, pred_va)
    line = format_report(f"GBDT-{tab.engine}", m)
    print(line)
    print(confusion(yva, pred_va))

    valdf = val.copy()
    valdf["pred_acuity"] = pred_va
    fair = subgroup_report(valdf, SCORED_TARGET, "pred_acuity", "language")
    print("\nUndertriage by language (val):")
    print(fair.to_string(index=False))

    rep = reports_dir(cfg)
    (rep / "baseline_metrics.txt").write_text(
        line + "\n\nConfusion (val):\n" + confusion(yva, pred_va).to_string()
        + "\n\nUndertriage by language (val):\n" + fair.to_string(index=False) + "\n"
    )
    valdf[[cfg["data"]["id_col"], SCORED_TARGET, "pred_acuity"]].to_parquet(
        rep / "val_predictions.parquet"
    )

    sub = pd.DataFrame({
        cfg["data"]["id_col"]: test[cfg["data"]["id_col"]].values,
        SCORED_TARGET: pred_te.astype(int),
    })
    sub.to_csv(ROOT / "submission.csv", index=False)
    print(f"\nwrote submission.csv ({len(sub)} rows) and reports/ artifacts")


if __name__ == "__main__":
    main()
