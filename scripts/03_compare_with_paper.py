"""Step 3: put our numbers next to the paper's and print the points you need for the 'differences' discussion.

Needs the outputs of step 2. Writes results/figures/{paper_vs_ours_metrics,feature_importance}.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from scipy.stats import spearmanr

from src.config import load_config, out_dir, resolve
from src.paper_reference import FEATURE_IMPORTANCE, METRICS as PAPER, OPTIMIZER_LABELS
from src.plots import plot_importance, plot_metric_comparison


def main():
    cfg = load_config()
    tdir, fdir = out_dir(cfg, "tables"), out_dir(cfg, "figures")
    res_path = resolve("results/phase1_results.json")
    if not res_path.exists():
        sys.exit("Run scripts/02_tune_and_evaluate.py first.")
    res = json.loads(res_path.read_text())
    comp = pd.read_csv(tdir / "paper_vs_ours_metrics.csv")
    imp = pd.read_csv(tdir / "feature_importance.csv", index_col=0)

    print("=== Paper-reported vs ours (test set) ===")
    print(comp.round(4).to_string(index=False))
    print("\nNote: the paper's MSE and RMSE are not consistent with each other; rely on RMSE/MAE/R2/MAPE for comparison.")

    # Do we reach the same conclusion about which optimizer is best?
    paper_rank = sorted(PAPER, key=lambda o: PAPER[o]["RMSE"])
    ours_rank = sorted(["GS", "RS", "BO"], key=lambda o: res["metrics"][o]["RMSE"])
    print("\nOptimizer ranking by test RMSE (best first):")
    print("  paper:", " < ".join(OPTIMIZER_LABELS[o] for o in paper_rank))
    print("  ours :", " < ".join(OPTIMIZER_LABELS[o] for o in ours_rank))
    print("  best by CV score (our selection rule):", OPTIMIZER_LABELS[res["selected_by_cv"]])

    sel = res["selected_by_cv"]
    feats = list(FEATURE_IMPORTANCE)
    rho = spearmanr([FEATURE_IMPORTANCE[f] for f in feats], [imp.loc[f, sel] for f in feats]).statistic
    print(f"\nFeature-importance rank agreement with paper (Spearman rho, {OPTIMIZER_LABELS[sel]}): {rho:.2f}")
    print("Our importances:", {f: round(float(imp.loc[f, sel]), 3) for f in feats})
    print("Paper importances:", FEATURE_IMPORTANCE)

    plot_metric_comparison(comp, fdir / "paper_vs_ours_metrics.png")
    plot_importance(imp, sel, fdir / "feature_importance.png")
    print("\nSaved figures. Now write the 'Comparison with the original paper' section of your report.")


if __name__ == "__main__":
    main()
