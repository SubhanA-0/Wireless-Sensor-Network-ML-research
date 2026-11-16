"""Step 1: look at the data and check it matches the paper's Table 1 and correlation figure.

Writes: results/tables/describe.csv, sanity_check_vs_paper_table1.csv, correlations_vs_paper.csv
        results/figures/eda_distributions.png, eda_correlation.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.config import load_config, out_dir
from src.data import FEATURES, TARGET, load_data
from src.paper_reference import CORR_WITH_ALE, TABLE1
from src.plots import plot_correlation, plot_distributions


def main():
    cfg = load_config()
    df = load_data(cfg)
    tdir, fdir = out_dir(cfg, "tables"), out_dir(cfg, "figures")

    print(f"Shape: {df.shape}   (paper: 107 rows, 4 features + target)")
    print("Missing values per column:\n", df.isna().sum().to_string())
    print("Duplicate rows:", int(df.duplicated().sum()))

    desc = df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.to_csv(tdir / "describe.csv")
    print("\nSummary statistics:\n", desc.round(3).to_string())

    # --- is this the same dataset the paper used? compare with paper Table 1 ---
    rows = []
    for col, stats in TABLE1.items():
        for stat, paper_val in stats.items():
            ours = desc.loc[col, stat]
            rows.append({"variable": col, "statistic": stat, "paper": paper_val, "ours": round(float(ours), 4),
                         "match": bool(np.isclose(ours, paper_val, rtol=0.01, atol=0.011))})
    chk = pd.DataFrame(rows)
    chk.to_csv(tdir / "sanity_check_vs_paper_table1.csv", index=False)
    n_ok = int(chk["match"].sum())
    print(f"\nPaper Table 1 check: {n_ok}/{len(chk)} statistics match (rtol 1%).")
    if n_ok < len(chk):
        print("Mismatches:\n", chk[~chk["match"]].to_string(index=False))
        print("-> If many mismatch, you may have a different version of the dataset. Note it in your report.")

    # --- correlations with ALE ---
    corr = df.corr()[TARGET].drop(TARGET)
    cmp = pd.DataFrame({"paper": pd.Series(CORR_WITH_ALE), "ours": corr}).loc[FEATURES]
    cmp["difference"] = cmp["ours"] - cmp["paper"]
    cmp.to_csv(tdir / "correlations_vs_paper.csv")
    print("\nCorrelation with ALE:\n", cmp.round(4).to_string())

    plot_distributions(df, fdir / "eda_distributions.png")
    plot_correlation(df, fdir / "eda_correlation.png")
    print("\nSaved tables to results/tables and figures to results/figures. Next: python scripts/02_tune_and_evaluate.py")


if __name__ == "__main__":
    main()
