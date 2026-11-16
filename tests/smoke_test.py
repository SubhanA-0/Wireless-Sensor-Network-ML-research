"""Smoke test: checks that your installation and the code work, WITHOUT needing the real dataset.

It builds a small SYNTHETIC table (random numbers that only mimic the shape of the real data), runs the
pipeline with tiny search budgets and checks that nothing crashes and the outputs look sane.

!! The synthetic numbers mean nothing scientifically. Never use them in your report. !!

Run:  python tests/smoke_test.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import FEATURES, TARGET, standardise_columns
from src.metrics import METRIC_NAMES
from src.pipeline import comparison_table, run_phase1


def make_synthetic(n=107, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "anchor_ratio": rng.choice([10, 15, 18, 20, 25, 30], n),
        "trans_range": rng.integers(12, 26, n),
        "node_density": rng.choice([100, 150, 200, 250, 300], n),
        "iterations": rng.integers(14, 101, n),
    })
    df[TARGET] = (3.0 - 0.008 * df["node_density"] - 0.008 * df["iterations"] + 0.02 * df["trans_range"]
                  + rng.normal(0, 0.15, n)).clip(0.3, None)
    return df


def main():
    # 1) column handling: a UCI-style file with a 6th "std" column must be reduced to 4 features + target
    raw = make_synthetic()
    raw["sd_ale"] = 0.1
    clean = standardise_columns(raw.rename(columns={"anchor_ratio": "Anchor Ratio"}))
    assert list(clean.columns) == FEATURES + [TARGET], clean.columns
    print("ok: column standardisation drops the std column")

    # 2) tiny end-to-end run
    cfg = copy.deepcopy(load_config())
    cfg["search_space"] = {"n_estimators": [10, 30], "max_depth": [None, 5], "min_samples_split": [2, 5],
                           "min_samples_leaf": [1, 2]}
    cfg["budget"] = {"random_search_trials": 4, "bayesian_trials": 4}
    res = run_phase1(clean, cfg)
    assert set(res["metrics"]) == {"Baseline", "GS", "RS", "BO"}
    for k, m in res["metrics"].items():
        assert set(m) == set(METRIC_NAMES) and np.isfinite(list(m.values())).all(), (k, m)
    assert res["n_train"] + res["n_test"] == len(clean)
    assert res["selected_by_cv"] in {"GS", "RS", "BO"}
    comp = comparison_table(res["metrics"])
    assert len(comp) == 3 * len(METRIC_NAMES)
    print("ok: pipeline runs end to end; R2 per model:",
          {k: round(v["R2"], 2) for k, v in res["metrics"].items()})
    print("\nSmoke test passed. (Numbers above come from synthetic data and are meaningless.)")


if __name__ == "__main__":
    main()
