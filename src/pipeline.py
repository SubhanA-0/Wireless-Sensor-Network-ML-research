"""Phase 1 pipeline glue: baseline -> tune with GS/RS/BO -> evaluate on the test set -> compare with the paper.

Both the numbered scripts and the notebook call these functions, so they always agree.
"""
import json

import numpy as np
import pandas as pd

from .config import out_dir
from .data import FEATURES, split_data
from .metrics import METRIC_NAMES, compute_metrics
from .paper_reference import BEST_PARAMS, METRICS as PAPER_METRICS, OPTIMIZER_LABELS, TUNING_TIME_S
from .tuning import OPTIMIZERS, make_pipeline


def tune_all(X_train, y_train, cfg, seed, which=("GS", "RS", "BO")):
    """Run each optimizer on the TRAIN set only. Returns {optimizer: result dict}."""
    return {name: OPTIMIZERS[name](X_train, y_train, cfg, seed) for name in which}


def evaluate_params(params, X_train, y_train, X_test, y_test, seed):
    """Fit an RF with `params` on train, score on test. Returns (metrics, fitted model, predictions)."""
    model = make_pipeline(params, seed).fit(X_train, y_train)
    pred = model.predict(X_test)
    return compute_metrics(y_test, pred), model, pred


def evaluate_all(tuned, X_train, y_train, X_test, y_test, seed):
    """Baseline RF (default settings) + RF tuned by each optimizer, all scored on the same test set."""
    metrics, models, preds = {}, {}, {}
    metrics["Baseline"], models["Baseline"], preds["Baseline"] = evaluate_params({}, X_train, y_train, X_test, y_test, seed)
    for name, res in tuned.items():
        metrics[name], models[name], preds[name] = evaluate_params(res["best_params"], X_train, y_train, X_test, y_test, seed)
    return metrics, models, preds


def select_by_cv(tuned):
    """Optimizer with the best cross-validation score (the test set is not used for selection)."""
    return max(tuned, key=lambda k: tuned[k]["cv_score"])


def feature_importances(models):
    """Impurity-based RF feature importances for each fitted model, as a DataFrame (rows = features)."""
    return pd.DataFrame({name: m.named_steps["rf"].feature_importances_ for name, m in models.items()}, index=FEATURES)


def run_phase1(df, cfg, seed=None):
    """The whole Phase 1 experiment on one train/test split. Returns a dict of results."""
    seed = cfg["seed"] if seed is None else seed
    X_train, X_test, y_train, y_test = split_data(df, cfg, seed)
    tuned = tune_all(X_train, y_train, cfg, seed)
    metrics, models, preds = evaluate_all(tuned, X_train, y_train, X_test, y_test, seed)
    return {
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "tuned": tuned,
        "metrics": metrics,
        "importances": feature_importances(models),
        "selected_by_cv": select_by_cv(tuned),
        "y_test": y_test.to_numpy(),
        "preds": {k: np.asarray(v) for k, v in preds.items()},
    }


# ---------- tables ----------

def tuning_table(tuned):
    rows = []
    for name, r in tuned.items():
        rows.append({
            "optimizer": OPTIMIZER_LABELS[name], **r["best_params"],
            "cv_rmse": r["cv_rmse"], "time_s": r["time_s"], "n_trials": r["n_trials"],
        })
    return pd.DataFrame(rows)


def metrics_table(metrics):
    labels = {"Baseline": "Baseline RF (defaults)", **OPTIMIZER_LABELS}
    return pd.DataFrame(
        [{"model": labels[k], **{m: v[m] for m in METRIC_NAMES}} for k, v in metrics.items()]
    )


def comparison_table(metrics):
    """Long table: paper-reported vs our value for each optimizer and metric."""
    rows = []
    for opt in ("GS", "RS", "BO"):
        if opt not in metrics:
            continue
        for m in METRIC_NAMES:
            p, o = PAPER_METRICS[opt][m], metrics[opt][m]
            rows.append({
                "optimizer": OPTIMIZER_LABELS[opt], "metric": m,
                "paper_reported": p, "ours": o,
                "difference": None if p is None else o - p,
            })
    return pd.DataFrame(rows)


def params_comparison_table(tuned):
    rows = []
    for opt, r in tuned.items():
        for k, v in r["best_params"].items():
            rows.append({"optimizer": OPTIMIZER_LABELS[opt], "hyperparameter": k,
                         "paper_best": BEST_PARAMS[opt][k], "ours_best": v})
        rows.append({"optimizer": OPTIMIZER_LABELS[opt], "hyperparameter": "tuning_time_s",
                     "paper_best": TUNING_TIME_S[opt], "ours_best": round(r["time_s"], 2)})
    return pd.DataFrame(rows)


# ---------- saving ----------

def save_phase1(res, cfg):
    """Write all Phase 1 tables and a JSON dump to results/."""
    tdir = out_dir(cfg, "tables")
    tuning_table(res["tuned"]).to_csv(tdir / "tuning_summary.csv", index=False)
    metrics_table(res["metrics"]).to_csv(tdir / "test_metrics.csv", index=False)
    res["importances"].to_csv(tdir / "feature_importance.csv")
    comparison_table(res["metrics"]).to_csv(tdir / "paper_vs_ours_metrics.csv", index=False)
    params_comparison_table(res["tuned"]).to_csv(tdir / "paper_vs_ours_hyperparameters.csv", index=False)
    pd.DataFrame({"y_true": res["y_test"], **{k: v for k, v in res["preds"].items()}}).to_csv(
        tdir / "test_predictions.csv", index=False)
    dump = {
        "seed": res["seed"], "n_train": res["n_train"], "n_test": res["n_test"],
        "selected_by_cv": res["selected_by_cv"],
        "tuned": res["tuned"], "metrics": res["metrics"],
    }
    with open(tdir.parent / "phase1_results.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    return tdir
