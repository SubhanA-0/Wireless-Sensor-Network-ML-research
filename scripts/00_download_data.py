"""Step 0: download the dataset from the UCI repository and save a clean CSV to data/raw/ale_wsn.csv.

Usage:  python scripts/00_download_data.py
If you have no internet, download the CSV by hand instead (see data/README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve
from src.data import download_uci


def main():
    cfg = load_config()
    out = resolve(cfg["data"]["raw_path"])
    try:
        df = download_uci(out, uci_id=cfg["data"]["uci_id"])
    except ConnectionError:
        sys.exit("Could not reach the UCI server. Download the CSV manually (see data/README.md) "
                 f"and save it as {out}")
    print(df.head())
    print("Done. Next: python scripts/01_eda.py")


if __name__ == "__main__":
    main()
