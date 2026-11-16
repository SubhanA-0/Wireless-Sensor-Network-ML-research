"""Load configs/config.yaml and resolve paths relative to the repo root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path=None):
    """Return the config as a plain dict."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path):
    """Turn a repo-relative path from the config into an absolute Path (creating folders for outputs)."""
    return ROOT / rel_path


def out_dir(cfg, key):
    """Absolute path of an output folder ('figures' or 'tables'); created if missing."""
    p = resolve(cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p
