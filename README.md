# Predicting Average Localization Error of WSNs: Phase 1 Reproduction

Course: Machine Learning, semester project (Research Paper Implementation and Research Contribution).
Group: _<group number>_ · Members: _<names and roll numbers>_

This repository reproduces the main experiment of

> I. K. Nti, S. S. Rout, J. Yeboah, **"An optimized ensemble model for predicting average localization error of wireless sensor networks"**, *Decision Analytics Journal* 12 (2024) 100510. https://doi.org/10.1016/j.dajour.2024.100510

**In one paragraph.** In a wireless sensor network, most sensors estimate their own position from a few "anchor" sensors. The average gap between estimated and true position is the Average Localization Error (ALE). The paper trains a **Random Forest** to predict ALE from four network settings (anchor ratio, transmission range, node density, iterations) and compares three ways of tuning the forest: **Grid Search, Random Search and Bayesian Optimization**. We re-implement this, report our own numbers, and compare them honestly with the paper's.

---

## Quick start

```bash
# 1. environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. check the install works (uses synthetic numbers, takes seconds)
python tests/smoke_test.py

# 3. get the real dataset (needs internet; manual alternative in data/README.md)
python scripts/00_download_data.py

# 4. run everything
python run_all.py                    # add --skip-robustness for a faster run
```

Or open `notebooks/01_phase1_reproduction.ipynb` and *Run All*. It uses the same code and the same seed.

**Runtime:** Grid Search (400 combinations x 5 folds) is the slow part, roughly 1 to 3 minutes on a laptop. `run_all.py` with the default 5-seed robustness experiment can take 10 to 20 minutes. Lower `robustness.n_seeds` in `configs/config.yaml` to shorten it.

---

## Repository layout

```
├── README.md                 <- this file
├── PLAN.md                   <- work plan and checklists
├── requirements.txt          <- dependencies
├── run_all.py                <- runs the whole pipeline in order
├── configs/
│   └── config.yaml           <- seed, split, CV, search spaces, budgets (all settings live here)
├── data/
│   ├── README.md             <- how to get the dataset
│   └── raw/ale_wsn.csv       <- created by scripts/00_download_data.py
├── src/                      <- reusable code (the notebook and scripts both use it)
│   ├── config.py   data.py   metrics.py   tuning.py   pipeline.py   plots.py
│   └── paper_reference.py    <- numbers REPORTED BY THE PAPER (kept separate from our results)
├── scripts/                  <- numbered steps, run in order
│   ├── 00_download_data.py        01_eda.py        02_tune_and_evaluate.py
│   └── 03_compare_with_paper.py   04_seed_robustness.py
├── notebooks/
│   └── 01_phase1_reproduction.ipynb   <- explained walkthrough of the whole experiment
├── tests/
│   └── smoke_test.py         <- quick install/logic check on synthetic data
├── results/
│   ├── tables/               <- CSV outputs (generated)
│   ├── figures/              <- PNG outputs (generated)
│   └── phase1_results.json   <- machine-readable summary (generated)
├── report/
│   └── phase1_report_outline.md   <- section skeleton for the short report
└── paper/                    <- reserved for the Phase 2 conference paper
```

## What each script does

| Step | Script | Produces |
|---|---|---|
| 0 | `scripts/00_download_data.py` | `data/raw/ale_wsn.csv` (4 features + ALE; the extra std column is dropped) |
| 1 | `scripts/01_eda.py` | summary stats, check against the paper's Table 1, correlations, distribution and correlation figures |
| 2 | `scripts/02_tune_and_evaluate.py` | baseline RF, Grid/Random/Bayesian tuning, test-set metrics, feature importance, predicted-vs-actual plot |
| 3 | `scripts/03_compare_with_paper.py` | paper-vs-ours tables and figures, optimizer ranking, feature-importance rank agreement |
| 4 | `scripts/04_seed_robustness.py` | the whole experiment repeated over several split seeds (mean ± std) |

## Experimental setup (what is fixed in `configs/config.yaml`)

* Data: 107 rows, 4 features, target ALE (m). Split 80/20, `random_state = 42`.
* Preprocessing: Min-Max scaling fitted on the training data only (inside a scikit-learn `Pipeline`).
* Model: `RandomForestRegressor`. Tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
* Tuning: 5-fold CV on the training set, scoring `neg_mean_squared_error`; Random Search and Bayesian Optimization get 50 trials each.
* Metrics: MAE, MSE, RMSE, R2, MAPE (MAPE is a fraction, as in the paper).

## Deviations from the paper (stated openly)

1. **Bayesian optimization uses Optuna (TPE)**, not scikit-optimize/Hyperopt (maintenance and compatibility). It is still a sequential model-based optimizer.
2. **Search grids and trial budgets are our choice.** The paper only gives the ranges (its Table 2). Grid Search uses a discrete grid inside those ranges; Random Search and Bayesian Optimization sample integers over the same ranges.
3. **Optimizer selection uses the cross-validation score**, not the test score (the paper selects on the test set, which leaks).
4. **A fixed seed (42)** is used; the paper reports none. We also report variation across seeds (step 4).
5. **Scaling is fitted inside the CV pipeline** (no leakage).

## Notes on the paper (things we noticed while reproducing)

* The reported MSE and RMSE are not consistent with each other (e.g. Grid Search: RMSE 0.238 implies an MSE of about 0.057, but 0.046 is reported); RS's MSE is never printed. We therefore compare mainly on RMSE, MAE, R2 and MAPE.
* The text mentions six optimization algorithms, but only three (GS, RS, BO) have results.
* Grid and Random Search are both listed with 52.74 s, which looks like a copy error.
* Random Search's best forest has only 10 trees, and the test set has about 22 rows, so the ranking of optimizers is probably fragile. Step 4 tests this.

## Honesty notes

* Everything in `results/` is produced by this code. Paper-reported numbers live only in `src/paper_reference.py` and are always labelled "paper".
* The dataset is **simulated** (modified Cuckoo Search simulations by Singh et al.), not measured on real hardware.
* `tests/smoke_test.py` uses synthetic numbers purely to check that the code runs. Never quote them.

## Data and citations

* Dataset: A. Singh, V. Kotiyal, S. Sharma, J. Nagar, C.-C. Lee, *Average Localization Error (ALE) in sensor node localization process in WSNs*, UCI Machine Learning Repository, 2020. https://doi.org/10.24432/C5ZP6R
* Dataset paper: A. Singh et al., "A machine learning approach to predict the average localization error with applications to wireless sensor networks," *IEEE Access* 8 (2020) 208253-208263.
* Libraries: scikit-learn, pandas, NumPy, SciPy, Optuna, Matplotlib, Seaborn.
* Code adapted from public repositories: **none** (all code written by the group).
