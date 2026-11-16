"""Run the whole Phase 1 pipeline: download data (if missing) -> EDA -> tune/evaluate -> compare with paper -> seed robustness.

Usage:
    python run_all.py                    # everything
    python run_all.py --skip-robustness  # skip the slower multi-seed experiment (step 4)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, resolve  # noqa: E402


def run(script):
    print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-robustness", action="store_true", help="skip scripts/04_seed_robustness.py")
    args = ap.parse_args()

    if not resolve(load_config()["data"]["raw_path"]).exists():
        run("00_download_data.py")
    for s in ["01_eda.py", "02_tune_and_evaluate.py", "03_compare_with_paper.py"]:
        run(s)
    if not args.skip_robustness:
        run("04_seed_robustness.py")
    print("\nAll done. Tables are in results/tables, figures in results/figures.")


if __name__ == "__main__":
    main()
