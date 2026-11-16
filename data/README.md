# Dataset

**Average Localization Error (ALE) in sensor node localization process in WSNs**
Singh, Kotiyal, Sharma, Nagar, Lee (2020). UCI Machine Learning Repository, dataset id **844**.
DOI: https://doi.org/10.24432/C5ZP6R · Page: https://archive.ics.uci.edu/dataset/844

107 rows. Columns: `anchor_ratio`, `trans_range` (m), `node_density`, `iterations`, `ale` (m, the target), plus a
last column with the standard deviation of ALE. The standard-deviation column is **not a feature** in the base paper;
our loader drops it automatically (keeping it would leak information about the target).

The data comes from modified Cuckoo Search **simulations**, not from real hardware.

## Option A: automatic (needs internet)

```bash
python scripts/00_download_data.py
```

This saves a clean CSV at `data/raw/ale_wsn.csv`.

## Option B: manual

1. Open the UCI page above and download the data file (CSV).
2. Save it as `data/raw/ale_wsn.csv`.
3. The loader tolerates different header spellings (for example `Anchor Ratio`, `trans_range`, `ALE`) and drops the std column.
   If it complains about missing columns, rename the header to: `anchor_ratio, trans_range, node_density, iterations, ale`.

## Check you have the right data

Run `python scripts/01_eda.py`. It compares your summary statistics with the paper's Table 1 and prints how many match
(ALE should have mean 0.98, min 0.39, max 2.57, 107 rows). If many statistics differ, say so in your report.

The file is tiny, so you can commit it to git. Dataset licence and terms are on the UCI page.
