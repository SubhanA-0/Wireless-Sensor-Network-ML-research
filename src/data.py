"""Loading and splitting the ALE dataset.

Dataset: "Average Localization Error (ALE) in sensor node localization process in WSNs"
Singh, Kotiyal, Sharma, Nagar, Lee (2020), UCI ML Repository, id 844, DOI 10.24432/C5ZP6R.
107 rows. Four features + the target ALE. UCI also ships a 6th column (the standard deviation of ALE)
which the base paper does NOT use as a feature. We always drop it, because it describes the target and
would leak information.
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve

FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]
TARGET = "ale"
_STD_TOKENS = {"sd", "std", "stdev", "stddev", "standard", "deviation"}


def _canonical_name(col):
    """Map whatever the CSV calls a column to our canonical name."""
    c = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
    tokens = set(c.split("_"))
    if tokens & _STD_TOKENS:
        return "ale_std"
    if "anchor" in c:
        return "anchor_ratio"
    if "trans" in c or c.endswith("range"):
        return "trans_range"
    if "density" in c:
        return "node_density"
    if "iter" in c:
        return "iterations"
    if c in {"ale", "average_localization_error", "average_localisation_error"}:
        return "ale"
    return c


def standardise_columns(df):
    """Rename columns to canonical names, check all required ones exist, drop everything else.

    Returns a DataFrame with exactly FEATURES + [TARGET] and prints what was dropped.
    """
    new_names = [_canonical_name(c) for c in df.columns]
    if len(set(new_names)) != len(new_names):
        raise ValueError(f"Two columns map to the same name. Original columns: {list(df.columns)}")
    df = df.copy()
    df.columns = new_names

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing}. Columns found after renaming: {list(df.columns)}. "
            "Check the CSV header (see data/README.md)."
        )
    dropped = [c for c in df.columns if c not in FEATURES + [TARGET]]
    if dropped:
        print(f"[data] dropped non-feature columns: {dropped} (the ALE std column must not be used as a feature)")
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    return df


def download_uci(out_path=None, uci_id=844):
    """Download the dataset from the UCI repository and save a clean CSV (needs internet)."""
    from ucimlrepo import fetch_ucirepo  # imported here so the rest of the repo works offline

    d = fetch_ucirepo(id=uci_id)
    raw = d.data.original  # features + targets (+ any extra columns) in one DataFrame
    df = standardise_columns(raw)
    out_path = Path(out_path) if out_path else resolve("data/raw/ale_wsn.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[data] saved {df.shape[0]} rows x {df.shape[1]} cols to {out_path}")
    return df


def load_data(cfg=None, path=None):
    """Read the cleaned CSV (data/raw/ale_wsn.csv by default) and return a DataFrame."""
    if path is None:
        rel = cfg["data"]["raw_path"] if cfg else "data/raw/ale_wsn.csv"
        path = resolve(rel)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run `python scripts/00_download_data.py`, or download the CSV manually from the UCI page "
            "(see data/README.md) and save it at that path."
        )
    return standardise_columns(pd.read_csv(path))


def split_data(df, cfg, seed=None):
    """Train/test split as in the paper (80/20). Returns X_train, X_test, y_train, y_test."""
    seed = cfg["seed"] if seed is None else seed
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        shuffle=cfg["split"]["shuffle"],
        random_state=seed,
    )
