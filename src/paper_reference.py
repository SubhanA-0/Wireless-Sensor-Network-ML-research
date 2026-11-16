"""Numbers REPORTED BY THE BASE PAPER (Nti, Rout, Yeboah, Decision Analytics Journal 12 (2024) 100510).

These are copied from the paper, not produced by our code. Keep them clearly separate from our results.
Known issues in the paper's own numbers (see README, "Notes on the paper"):
  * Reported MSE and RMSE are not mutually consistent (e.g. GS: RMSE 0.238 -> MSE ~0.057, but MSE reported as 0.046).
  * RS's MSE is described as the lowest but is never printed, so it is None here.
  * GS and RS are both reported with exactly 52.74 s.
"""

OPTIMIZER_LABELS = {"GS": "Grid Search", "RS": "Random Search", "BO": "Bayesian Opt."}

# Table 1: summary statistics of the study data (paper, Section 4.1)
TABLE1 = {
    "anchor_ratio": {"mean": 20.52, "std": 6.74, "min": 10, "25%": 15, "50%": 18, "75%": 30, "max": 30},
    "trans_range": {"mean": 17.88, "std": 3.11, "min": 12, "25%": 15, "50%": 17, "75%": 20, "max": 25},
    "node_density": {"mean": 159.81, "std": 71.19, "min": 100, "25%": 100, "50%": 100, "75%": 200, "max": 300},
    "iterations": {"mean": 47.89, "std": 24.67, "min": 14, "25%": 30, "50%": 40, "75%": 70, "max": 100},
    "ale": {"mean": 0.98, "std": 0.41, "min": 0.39, "25%": 0.66, "50%": 0.90, "75%": 1.20, "max": 2.57},
}

# Correlation of each feature with ALE (paper, Section 4.1 / Fig. 4)
CORR_WITH_ALE = {
    "anchor_ratio": -0.074997,
    "trans_range": 0.109309,
    "node_density": -0.645927,
    "iterations": -0.400394,
}

# Table 2: best hyperparameters found by each optimizer, and tuning time in seconds
BEST_PARAMS = {
    "GS": {"n_estimators": 400, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    "RS": {"n_estimators": 10, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
    "BO": {"n_estimators": 400, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 2},
}
TUNING_TIME_S = {"GS": 52.74, "RS": 52.74, "BO": 150.93}

# Section 4.3 / Fig. 6: test-set performance of RF tuned by each optimizer
METRICS = {
    "GS": {"MAE": 0.171, "MSE": 0.046, "RMSE": 0.238, "R2": 0.712, "MAPE": 0.145},
    "RS": {"MAE": 0.156, "MSE": None, "RMSE": 0.220, "R2": 0.753, "MAPE": 0.140},
    "BO": {"MAE": 0.190, "MSE": 0.047, "RMSE": 0.251, "R2": 0.681, "MAPE": 0.159},
}

# Fig. 5: feature importance of the RF
FEATURE_IMPORTANCE = {
    "iterations": 0.4621,
    "node_density": 0.3669,
    "anchor_ratio": 0.1084,
    "trans_range": 0.0626,
}
