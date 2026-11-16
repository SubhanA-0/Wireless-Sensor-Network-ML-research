# PLAN.md: Phase 1 work plan

**Project:** reproduce *"An Optimized Ensemble Model for Predicting Average Localization Error of WSNs"* (Nti, Rout, Yeboah, 2024).
**Phase 1 deadline:** _<fill in>_ · **Group:** _<fill in>_

## Phase 1 deliverables (from the brief)

| # | Deliverable | Where it comes from in this repo | Done |
|---|---|---|---|
| D1 | Short report: paper, methodology, dataset, implementation, setup, results | `report/phase1_report_outline.md` + files in `results/` | ☐ |
| D2 | Comparison of our results with the paper's, with a discussion of differences | `scripts/03_compare_with_paper.py`, `paper_vs_ours_*.csv/png`, notebook section 8-9 | ☐ |
| D3 | Complete source code | `src/`, `scripts/`, `notebooks/`, `configs/` | ☐ |
| D4 | README with execution instructions and dependencies | `README.md`, `requirements.txt` | ☐ |

## The one-paragraph picture

Predict ALE (meters) from 4 network settings with a Random Forest. Tune it three ways (Grid, Random, Bayesian), 5-fold CV on the training set, then score once on a held-out 20% test set. Compare with the paper's numbers. Because the dataset has only 107 rows, also repeat with several seeds to see how stable the conclusions are.

---

## Work blocks (each is roughly half a day to a day)

### Block 0: setup and data ☐
- [ ] Create the repo from the zip, `python -m venv .venv`, `pip install -r requirements.txt`
- [ ] `python tests/smoke_test.py` passes
- [ ] `python scripts/00_download_data.py` (or manual download, see `data/README.md`)
- [ ] Fill in group number and member names in `README.md`
- [ ] Read the paper once; confirm the facts in `src/paper_reference.py` against the PDF
- [ ] First git commit

### Block 1: understand and check the data ☐
- [ ] `python scripts/01_eda.py`
- [ ] Table 1 check: how many statistics match the paper? Note any mismatch for the report
- [ ] Look at `eda_distributions.png` and `eda_correlation.png`. Can you explain in one sentence why node density lowers ALE?

### Block 2: baseline and tuning ☐
- [ ] Open the notebook, run sections 0-6 (or `python scripts/02_tune_and_evaluate.py`)
- [ ] Check `results/tables/tuning_summary.csv` and `test_metrics.csv`
- [ ] Sanity check: baseline RF vs tuned RF. Is tuning actually helping?

### Block 3: compare with the paper ☐
- [ ] `python scripts/03_compare_with_paper.py`
- [ ] Does our optimizer ranking match the paper's (Random < Grid < Bayesian by RMSE)?
- [ ] Do our feature importances match (iterations > node density > anchor ratio > transmission range)?
- [ ] Write 5 to 8 sentences on the differences and their likely causes (use the prompts in notebook section 9)

### Block 4: stability check ☐
- [ ] `python scripts/04_seed_robustness.py` (set `robustness.n_seeds` in `configs/config.yaml`; default 5)
- [ ] Does "Random Search is best" survive different splits? Put the mean ± std table in the report

### Block 5: report ☐
- [ ] Write the short report following `report/phase1_report_outline.md`
- [ ] Add a workflow diagram, the key tables and 3 to 5 figures
- [ ] Every number in the report appears in a file under `results/`
- [ ] Paper-reported numbers and our numbers are always labelled separately

### Block 6: packaging and final check ☐
- [ ] Fresh-environment test: new venv, `pip install -r requirements.txt`, `python run_all.py`
- [ ] Remove scratch files; commit `results/` (small) or leave instructions to regenerate
- [ ] README reads correctly for someone who has never seen the project
- [ ] Every group member can explain: what ALE is, why we scale inside the pipeline, why we select by CV and not by test score, what the paper's weaknesses are
- [ ] Submit through the platform the instructor announced

---

## Suggested split of work (if you are a group)

| Person | Focus |
|---|---|
| A | Blocks 0-1, data and EDA, repo hygiene |
| B | Blocks 2-3, tuning and paper comparison |
| C | Blocks 4-5, robustness experiment and report writing |
| All | Block 6 review, plus each person explains one section in a mock defence |

## Rules we follow (academic integrity, from the brief)

- Never write "reproduced" for something we did not actually run.
- Never tune settings until we match the paper. A documented gap is fine.
- Disclose all external code. (We wrote everything ourselves.)
- Cite the paper, the dataset and the libraries.

## Known limitations to mention in the report

- Only 107 rows (a test set of about 22), so single-split results are noisy.
- The data is simulated, not measured on real sensors.
- The paper gives no seed, exact grids or budgets, so exact numbers cannot be matched.
- The paper's reported MSE and RMSE are inconsistent with each other.

---

## Phase 2 preview (for after Phase 1)

Candidate research question: *Which hyperparameter optimizer gives the most reliable ALE predictions under an equal compute budget, and do those models stay trustworthy under noisy inputs?*

Possible pieces (pick 2 to 3 and get instructor approval):
1. **Fair optimizer comparison** using nested cross-validation, with equal budgets, convergence curves and a significance test.
2. **Robustness study:** add noise and missing values to the features and plot error against noise level.
3. **Explainability:** SHAP values compared with plain RF importance.
4. **Stronger baselines:** SVR, XGBoost/LightGBM, stacking (other ALE papers report SVR doing very well, so check literature before claiming novelty).

The Phase 1 code is built so these slot in: new modules in `src/`, new numbered scripts in `scripts/`, paper source in `paper/`.
