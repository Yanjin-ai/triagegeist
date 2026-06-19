"""Phase 0 — download competition data and inventory it.

Run:  python -m src.data.download
Needs Kaggle auth (~/.kaggle/kaggle.json) or run inside a Kaggle notebook.

This is the first runnable step. It downloads the data, then prints a full
inventory (files, row counts, columns, dtypes, missingness) which we paste into
docs/DATA_DICTIONARY.md. Nothing here assumes the schema — it reports it.
"""
from __future__ import annotations

import os
from pathlib import Path


def download() -> str:
    import kagglehub

    path = kagglehub.competition_download("triagegeist")
    print("Path to competition files:", path)
    return path


def inventory(root: str) -> None:
    import pandas as pd

    root_path = Path(root)
    tabular = sorted(
        p for p in root_path.rglob("*")
        if p.suffix.lower() in {".csv", ".gz", ".parquet", ".tsv", ".json", ".jsonl"}
    )
    print(f"\n=== {len(tabular)} data file(s) under {root} ===")
    for p in tabular:
        size_mb = p.stat().st_size / 1e6
        print(f"\n--- {p.relative_to(root_path)}  ({size_mb:.1f} MB) ---")
        try:
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            elif p.suffix.lower() in {".json", ".jsonl"}:
                df = pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
            else:
                df = pd.read_csv(p, nrows=200_000)
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not parse: {exc})")
            continue
        print(f"  rows (sampled≤200k): {len(df):,}  cols: {df.shape[1]}")
        miss = (df.isna().mean() * 100).round(1)
        for col in df.columns:
            print(f"    {col:<28} {str(df[col].dtype):<12} miss={miss[col]:>5}%")


if __name__ == "__main__":
    root = os.environ.get("TRIAGEGEIST_DATA") or download()
    inventory(root)
    print(
        "\nNext: paste this inventory into docs/DATA_DICTIONARY.md, "
        "confirm targets/splits in configs/config.yaml, then start notebooks/01_eda.ipynb."
    )
