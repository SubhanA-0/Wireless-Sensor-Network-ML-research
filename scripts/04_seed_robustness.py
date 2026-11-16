"""Step 4 (recommended): how much do the results change if the random 80/20 split changes?

With only 107 rows the test set has ~22 rows, so a single split is noisy. This repeats the whole
split -> tune -> test pipeline for several seeds and reports mean +/- std. It answers: "is the paper's
claim that Random Search beats Grid Search and Bayesian Optimization stable, or just one lucky split?"

Runtime: n_seeds x (time of step 2). Set robustness.n_seeds in configs/config.yaml.
Writes: results/tables/seed_robustness_long.csv, seed_robustness_summary.csv, results/figures/seed_robustness.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.config import load_config, out_dir
from src.data import load_data
from src.metrics import METRIC_NAMES
from src.paper_reference import OPTIMIZER_LABELS
from src.pipeline import run_phase1
from src.plots import plot_seed_robustness


def main():
    cfg = load_config()
    df = load_data(cfg)
    tdir, fdir = out_dir(cfg, "tables"), out_dir(cfg, "figures")
    n = cfg["robustness"]["n_seeds"]
    rows = []
    for i in range(n):
        seed = cfg["seed"] + i
        print(f"[{i + 1}/{n}] seed {seed} ...", flush=True)
        res = run_phase1(df, cfg, seed=seed)
        for opt in ("GS", "RS", "BO"):
            rows.append({"seed": seed, "optimizer": opt, **{m: res["metrics"][opt][m] for m in METRIC_NAMES},
                         "time_s": res["tuned"][opt]["time_s"]})
        rows.append({"seed": seed, "optimizer": "Baseline", **{m: res["metrics"]["Baseline"][m] for m in METRIC_NAMES},
                     "time_s": 0.0})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(tdir / "seed_robustness_long.csv", index=False)

    summ = long_df.groupby("optimizer")[METRIC_NAMES + ["time_s"]].agg(["mean", "std"])
    summ.index = [OPTIMIZER_LABELS.get(i, "Baseline RF") for i in summ.index]
    summ.to_csv(tdir / "seed_robustness_summary.csv")
    print("\nMean and std over seeds:\n", summ.round(4).to_string())

    # how often does each optimizer win (lowest test RMSE) across seeds?
    wins = (long_df[long_df["optimizer"] != "Baseline"].loc[lambda d: d.groupby("seed")["RMSE"].idxmin()]
            ["optimizer"].value_counts())
    print("\nNumber of splits where each optimizer had the lowest test RMSE:\n", wins.to_string())

    plot_seed_robustness(long_df[long_df["optimizer"] != "Baseline"], fdir / "seed_robustness.png")
    print("\nSaved. Use the mean +/- std table in your report next to the single-split results.")


if __name__ == "__main__":
    main()
