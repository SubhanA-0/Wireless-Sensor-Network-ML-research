# Phase 1 short report: outline

Fill each section from the generated files in `results/`. Keep it short. The brief asks for a *short* report.
Write every sentence in your own words. Only claim what the code actually ran.

## Cover page
Project title · base paper title · group number · members and roll numbers · course · submission date.

## 1. Introduction and problem understanding
What ALE is, why predicting it matters (setup time and cost), and the task: predict ALE from 4 network settings (regression).

## 2. Base paper summary
Their approach (RF tuned by GS/RS/BO, 5-fold CV, 80/20 split), their key claims (Random Search best; R2 0.753, RMSE 0.220; iterations and node density most important). Explain in your own words.

## 3. Dataset and preprocessing
UCI dataset 844, 107 rows, 4 features + target, simulated data. Table 1 check (`sanity_check_vs_paper_table1.csv`), figures `eda_distributions.png`, `eda_correlation.png`. Min-Max scaling, 80/20 split, std column dropped and why.

## 4. Methodology and implementation
Workflow diagram (data -> split -> scale -> tune with GS/RS/BO on train via 5-fold CV -> refit -> test). Hyperparameters and ranges (`configs/config.yaml`).

## 5. Experimental setup
Python version, libraries and versions, seed, split, CV, scoring, trial budgets, hardware (your laptop), runtime.

## 6. Results
`tuning_summary.csv`, `test_metrics.csv`, `pred_vs_actual.png`, `feature_importance.png`, and the seed-robustness table (`seed_robustness_summary.csv`, `seed_robustness.png`).

## 7. Comparison with the original paper
`paper_vs_ours_metrics.csv`, `paper_vs_ours_hyperparameters.csv`, `paper_vs_ours_metrics.png`. Explain differences: seed and split, our grids and budgets, Optuna instead of scikit-optimize, small test set, the paper's inconsistencies (MSE vs RMSE, identical GS/RS times).

## 8. Discussion and conclusion
What worked, what did not, challenges, limitations (107 simulated rows, one split, no real deployment data), and what Phase 2 will improve.

## 9. References
Base paper · dataset (UCI, DOI 10.24432/C5ZP6R) · Singh et al. 2020 (IEEE Access) · libraries used.
