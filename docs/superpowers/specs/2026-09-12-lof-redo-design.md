# LOF Redo — Design Spec

**Author:** Amir (individual contribution to the group EWS project)
**Date:** 2026-09-12
**Scope:** `Amir/` only — rebuilding the Local Outlier Factor (LOF) preprocessing and modeling pipeline from a new raw dataset. Does not touch the other five members' pipelines (DBSCAN, Autoencoder, Isolation Forest, OCSVM, PCA).

## 1. Why this redo

The previous LOF pipeline (`Amir/lof.ipynb`, `laporan_lof.md`) claimed to be "unsupervised murni" but was not:

1. `n_neighbors` was chosen via grid search evaluated against `crisis_label` (AUC-ROC) — label-guided tuning, not label-free.
2. `contamination` was set literally equal to the crisis ratio (0.1919) — a more direct label injection than the `n_neighbors` issue.
3. No train/tune/test split existed — the same full dataset was used for both tuning and reporting the final AUC-ROC, so the reported score measured fit-to-label rather than generalization.
4. This makes the six-way algorithm comparison not apples-to-apples: AE/OCSVM/PCA fit only on label-filtered normal rows, DBSCAN uses no label at all, and LOF sat at an undocumented, inconsistent point in between.

Separately, while inspecting the ground truth: `Laporan_UTS.md` §3.3 documents `crisis_label` as built from exactly six historical episodes (specific country/year combinations), which sums to ~179 rows. The group-level `data_with_labels.csv` instead reports 329 crisis rows spread across every year 1990–2024, and Amir's own old report is internally inconsistent about this (a 179-row per-crisis breakdown table vs. a "329 observations" summary elsewhere in the same document). This is now moot: Amir has an externally-sourced ground truth file, `Amir/ground_truth_imf.csv`, with per-country-year crisis flags (`banking_crisis`, `currency_crisis`, `sovereign_debt_crisis`, plus a combined `is_crisis`) covering all 1715 raw rows. `crisis_label` will be built by joining this file on `(economy, year)` and taking `is_crisis`, rather than either of the two old, mutually-inconsistent counts, and rather than a hand-rolled six-episode rule. This is a strictly better ground truth: it captures 229 crisis-country-years with per-country granularity (e.g. Turkey currency crises in 1991/1994/1996, Nigerian banking crises, Peru 1996) instead of assuming a handful of named global episodes hit every country uniformly — GFC 2008–2009 for instance flags only 32 of the 98 possible country-years, not all 49 countries × 2 years as a hand-rolled rule would assume.

## 2. Data

- Source: `Amir/raw_data_master.csv` — 1715 rows (49 countries × 35 years, 1990–2024), no header issues, one row per country-year.
- 14 raw indicators: `GDP_Growth`, `Inflation_CPI`, `Unemployment`, `Current_Account_GDP`, `Reserves_Months_Imports`, `Exchange_Rate`, `FDI_Inflows_GDP`, `Exports_GDP`, `Imports_GDP`, `Gross_Savings_GDP`, `Investment_GDP`, `Manufacturing_Value`, `Domestic_Credit_GDP`, `Broad_Money_Growth`.
- Missingness ranges 0.2% (`GDP_Growth`) to 25.0% (`Broad_Money_Growth`); no feature exceeds the 50% drop threshold.
- Ground truth: `Amir/ground_truth_imf.csv` — 1715 rows, columns `economy, year, banking_crisis, currency_crisis, sovereign_debt_crisis, crisis_name, is_crisis`. Verified directly against the raw file: exact same 49 economies, same 1990–2024 range, zero duplicate `(economy, year)` pairs, zero rows in the raw panel without a matching ground-truth row. `is_crisis` is not simply the OR of the three subtype flags — it also captures COVID-19 2020 as a crisis year for all 49 economies even though that event isn't a banking/currency/debt crisis by the Laeven-Valencia-style taxonomy the subtype columns follow; this is intentional, not a data bug. Total `is_crisis = 1` rows: 229 (13.4%).

## 3. Preprocessing pipeline

Applied in this order:

1. **Sort** by `economy`, then `year`.
2. **Exchange rate transform:** `Exchange_Depreciation = groupby('economy')['Exchange_Rate'].pct_change() * 100`; drop the nominal `Exchange_Rate` column. This structurally produces a NaN for each country's first observed year — expected, not a data defect. Feature count stays at 14.
3. **Missingness quality filter (on raw missingness, before any filling):** drop any feature with >50% missing (none qualify on this dataset — verify and record the check, don't skip it just because we expect it to pass); then, among the retained features, drop any row with >30% missing. This must run before interpolation/imputation — checking it afterward is meaningless, since the imputation fallback in step 5 fills every remaining NaN by construction, so a post-imputation missingness check would never trigger.
4. **Interpolation:** for every retained feature column, linear interpolation per country (`groupby('economy')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))`). Verified empirically: this fills interior gaps with true linear interpolation, and fills leading/trailing gaps by carrying the nearest valid value in that direction (e.g. `[NaN, NaN, 10, 20, NaN, 40, NaN, NaN]` → `[10, 10, 10, 20, 30, 40, 40, 40]`). This also resolves the year-1 `Exchange_Depreciation` NaN by extending from the next valid year.
5. **Remaining-NaN fallback:** interpolation can only leave a NaN behind when a country has zero valid observations at all for a column (nothing to interpolate from or extend). For any such remaining NaN, attempt per-country median first (a no-op in this exact scenario, kept as defense-in-depth in case future data breaks the "every country has ≥1 valid value" assumption); fall back to `KNNImputer` (k=5, default Euclidean) fit across the full feature matrix for whatever the median pass couldn't fill. After this step zero NaNs remain in the retained columns.
6. **Scaling:** `RobustScaler()` fit on all 14 features across the full panel (median/IQR-based — chosen over `StandardScaler` for resilience to macro shocks like hyperinflation or currency collapse, consistent with the group's guidance that distance-based methods need outlier-resistant scaling).
7. **Crisis label (built last, never upstream of steps 1–6):** left-join the pipeline's DataFrame with `Amir/ground_truth_imf.csv` on `(economy, year)`, and set `crisis_label = is_crisis`. Raise an error rather than silently producing a NaN if any row fails to match (it shouldn't — coverage was verified in §2 — but a future change to either file's country codes should fail loudly, not produce a silently-wrong label).

Output artifact: `Amir/data_cleaned_v2.csv`. Verified against the real raw file: 0 features exceed the 50% threshold; 90 rows exceed the 30% row threshold (concentrated in each country's earliest 1990s years, where World Bank coverage is sparsest) and are dropped. Of the 229 `is_crisis = 1` rows, 226 survive this filter — the 3 lost are Poland 1992–1994 ("Transition Banking Crisis"), which falls inside Poland's sparse post-transition data years. State this explicitly in the eventual report as a known limitation, not a silent gap. Final artifact: 1625 rows, 14 scaled features + `economy` + `year` + `crisis_label`, with 226 crisis rows (13.9% of 1625).

## 4. Modeling (LOF) — fully label-free hyperparameter selection

Not implemented in this plan iteration (preprocessing only); recorded here so the eventual modeling plan implements the same design that was reviewed and approved:

- **`n_neighbors`:** computed across a k-range (k_lb–k_ub) reflecting plausible regional peer-group sizes; take the max LOF score per point across that range, per Breunig et al.'s (2000) own robust-range recommendation. No `crisis_label` reference anywhere in this step.
- **`contamination='auto'`:** sklearn's literature-grounded ~1.5 LOF-score threshold, used only to produce hard 0/1 predictions for confusion-matrix metrics (Precision/Recall/F1). The headline metrics (AUC-ROC, Average Precision) use the continuous LOF score directly and require no threshold.
- **Evaluation:** one primary transductive LOF fit over the full standardized panel (matches how Breunig's method and the rest of the group's algorithms are normally run), plus a stratified-resampling robustness check (repeated stratified subsamples, refit transductively each time, report AUC-ROC/F1 as mean ± spread) instead of a single-shot number.
- **Reporting:** `laporan_lof_v2.md` states explicitly that LOF now uses zero label information anywhere before evaluation — the same level as DBSCAN — so the six-algorithm comparison has an honest, explicit anchor point per algorithm.

## 5. Deliverables and versioning

All new artifacts are created alongside the old ones in `Amir/` — nothing is overwritten, preserving the audit trail of the methodology correction:

- `Amir/data_cleaned_v2.csv` (preprocessing output, this plan)
- `Amir/preprocessing_lof_v2.py` (preprocessing pipeline code, this plan)
- `Amir/lof_v2.ipynb`, `Amir/laporan_lof_v2.md`, `Amir/hasil_lof_v2.csv`, `Amir/grid_search_lof_v2.csv`, `Amir/ringkasan_model_lof_v2.csv` (modeling phase, future plan)

## 6. Out of scope

- Other team members' pipelines (DBSCAN, Autoencoder, Isolation Forest, OCSVM, PCA) — not modified. The group-wide feature-engineering note (Exchange_Rate → depreciation, sorted interpolation, scaler choice per method) is Amir's to relay to teammates; this spec only implements it for Amir's own LOF work.
- Reconciling the old 329-row `data_with_labels.csv` crisis count — noted as a discrepancy in §1, not fixed at the group-data level.
