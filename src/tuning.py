"""Hyperparameter optimizers for the Random Forest: Grid Search, Random Search, Bayesian Optimization.

The base paper tunes four RF hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf)
with GridSearchCV, RandomizedSearchCV and Bayesian optimization (scikit-optimize / Hyperopt), 5-fold CV,
scoring = neg_mean_squared_error.

Our choices (the paper does not give exact grids or budgets):
  * Grid Search      : the discrete grid from configs/config.yaml (every combination).
  * Random Search    : `random_search_trials` random draws from the integer ranges spanned by that grid.
  * Bayesian Opt.    : Optuna's TPE sampler, `bayesian_trials` trials, same integer ranges.
    (Optuna replaces scikit-optimize/Hyperopt because it is actively maintained. TPE is a Bayesian-optimisation-style
    sequential model-based method. This is a documented deviation from the paper.)
Every candidate is scored with the same KFold splits, so the comparison between optimizers is fair.
"""
import time

import numpy as np
import optuna
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAM_NAMES = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]


def make_pipeline(params=None, seed=42):
    """MinMaxScaler -> RandomForestRegressor. The scaler is fitted inside each CV fold (no leakage).

    Note: Random Forests are scale-invariant, so Min-Max scaling does not change the predictions.
    It is kept only to follow the paper's preprocessing.
    """
    params = dict(params or {})
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("rf", RandomForestRegressor(random_state=seed, n_jobs=1, **params)),
    ])


def _cv(cfg, seed):
    return KFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=seed)


def _clean_params(params):
    """Strip the 'rf__' prefix and convert numpy ints to plain ints."""
    out = {}
    for k, v in params.items():
        k = k.replace("rf__", "")
        out[k] = None if v is None else int(v)
    return {k: out[k] for k in PARAM_NAMES}


def _int_ranges(space):
    """(low, high) integer range for each numeric hyperparameter, taken from the config grid."""
    return {k: (min(space[k]), max(space[k])) for k in ("n_estimators", "min_samples_split", "min_samples_leaf")}


def _result(best_params, best_score, elapsed, n_trials):
    return {
        "best_params": _clean_params(best_params),
        "cv_score": float(best_score),             # neg MSE (higher is better)
        "cv_rmse": float(np.sqrt(-best_score)),    # easier to read
        "time_s": float(elapsed),
        "n_trials": int(n_trials),
    }


def run_grid_search(X_train, y_train, cfg, seed):
    space = cfg["search_space"]
    grid = {f"rf__{k}": list(v) for k, v in space.items()}
    gs = GridSearchCV(
        make_pipeline(seed=seed), grid, cv=_cv(cfg, seed),
        scoring=cfg["cv"]["scoring"], n_jobs=-1, refit=False,
    )
    t0 = time.perf_counter()
    gs.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    n = int(np.prod([len(v) for v in grid.values()]))
    return _result(gs.best_params_, gs.best_score_, elapsed, n)


def run_random_search(X_train, y_train, cfg, seed):
    space = cfg["search_space"]
    r = _int_ranges(space)
    dist = {
        "rf__n_estimators": randint(r["n_estimators"][0], r["n_estimators"][1] + 1),
        "rf__max_depth": list(space["max_depth"]),
        "rf__min_samples_split": randint(r["min_samples_split"][0], r["min_samples_split"][1] + 1),
        "rf__min_samples_leaf": randint(r["min_samples_leaf"][0], r["min_samples_leaf"][1] + 1),
    }
    n_iter = cfg["budget"]["random_search_trials"]
    rs = RandomizedSearchCV(
        make_pipeline(seed=seed), dist, n_iter=n_iter, cv=_cv(cfg, seed),
        scoring=cfg["cv"]["scoring"], n_jobs=-1, random_state=seed, refit=False,
    )
    t0 = time.perf_counter()
    rs.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return _result(rs.best_params_, rs.best_score_, elapsed, n_iter)


def run_bayesian(X_train, y_train, cfg, seed):
    space = cfg["search_space"]
    r = _int_ranges(space)
    cv = _cv(cfg, seed)
    scoring = cfg["cv"]["scoring"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *r["n_estimators"]),
            "max_depth": trial.suggest_categorical("max_depth", list(space["max_depth"])),
            "min_samples_split": trial.suggest_int("min_samples_split", *r["min_samples_split"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *r["min_samples_leaf"]),
        }
        scores = cross_val_score(make_pipeline(params, seed), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))

    n_trials = cfg["budget"]["bayesian_trials"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.perf_counter() - t0
    return _result(study.best_params, study.best_value, elapsed, n_trials)


OPTIMIZERS = {"GS": run_grid_search, "RS": run_random_search, "BO": run_bayesian}
