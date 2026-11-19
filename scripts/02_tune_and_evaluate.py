"""Step 2: baseline, tuning and test-set evaluation.

  80/20 split -> baseline RF -> tune RF with Grid / Random / Bayesian search (5-fold CV, train set only)
  -> evaluate each tuned RF on the held-out test set -> save tables and figures.

Writes: results/tables/{tuning_summary,test_metrics,feature_importance,test_predictions,
                        paper_vs_ours_metrics,paper_vs_ours_hyperparameters}.csv, results/phase1_results.json
        results/figures/pred_vs_actual.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

from src.config import load_config, out_dir
from src.data import load_data
from src.pipeline import metrics_table, run_phase1, save_phase1, tuning_table
from src.plots import plot_pred_vs_actual


def main():
    cfg = load_config()
    df = load_data(cfg)
    print(f"Loaded {len(df)} rows. Seed = {cfg['seed']}.")
    res = run_phase1(df, cfg)

    print(f"\nTrain rows: {res['n_train']}, test rows: {res['n_test']}")
    print("\n=== Tuning (train set, 5-fold CV) ===")
    print(tuning_table(res["tuned"]).round(3).to_string(index=False))
    print("\n=== Test-set metrics ===")
    print(metrics_table(res["metrics"]).round(4).to_string(index=False))
    print(f"\nBest optimizer by CV score: {res['selected_by_cv']}")

    save_phase1(res, cfg)
    plot_pred_vs_actual(res["y_test"], res["preds"], res["metrics"], out_dir(cfg, "figures") / "pred_vs_actual.png")
    print("\nSaved results to results/.")


if __name__ == "__main__":
    main()
