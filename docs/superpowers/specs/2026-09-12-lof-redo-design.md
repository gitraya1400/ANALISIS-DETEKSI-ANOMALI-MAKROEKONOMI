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

## 2. Mandatory group-wide output rules

These come from the group/lecturer, apply across all six algorithms (not just LOF), and override any conflicting design choice made anywhere else in this spec:

| Rule | Applies to |
|---|---|
| Exactly 1,715 rows in every output file — **no row may ever be dropped**, at any pipeline stage | Preprocessing (§4) |
| Row order: `economy` ascending, then `year` ascending (1990–2024) | Preprocessing (§4) |
| Currency features must be `Exchange_Depreciation` (% change), never a raw nominal exchange rate | Preprocessing (§4) |
| Binary prediction convention: `1` = Anomali/Krisis, `0` = Normal | Modeling (§5) |
| Random seed fixed at `42` everywhere randomness is used | Modeling (§5) |
| `hasil_[metode].csv` must contain exactly the columns `['economy', 'year', 'anomaly_score', 'predicted_anomaly']` — nothing more, nothing less (no `crisis_label` column in this file; that's joined in separately at the majority-voting aggregation stage) | Modeling (§5) |

The "no row may ever be dropped" rule is why §4 below no longer has a row-level missingness filter — an earlier version of this spec had one (dropping 90 sparse rows) before this rule was confirmed. It's superseded; see §4 step 3 for how full coverage is achieved without dropping anything.

## 3. Data

- Source: `Amir/raw_data_master.csv` — 1715 rows (49 countries × 35 years, 1990–2024), no header issues, one row per country-year.
- 14 raw indicators: `GDP_Growth`, `Inflation_CPI`, `Unemployment`, `Current_Account_GDP`, `Reserves_Months_Imports`, `Exchange_Rate`, `FDI_Inflows_GDP`, `Exports_GDP`, `Imports_GDP`, `Gross_Savings_GDP`, `Investment_GDP`, `Manufacturing_Value`, `Domestic_Credit_GDP`, `Broad_Money_Growth`.
- Missingness ranges 0.2% (`GDP_Growth`) to 25.0% (`Broad_Money_Growth`) at the column level; no feature exceeds the 50% drop threshold.
- 14 specific (economy, feature) pairs have **zero** valid observations across that country's entire 35-year span — these are the cells interpolation cannot reach and `KNNImputer` must fill (§4 step 5): Nigeria has no data at all for `Exports_GDP`, `Imports_GDP`, `Gross_Savings_GDP`, `Investment_GDP` (a national-accounts reporting gap); the 10 Eurozone countries in the panel (`AUT, BEL, DEU, ESP, FRA, GRC, IRL, ITA, NLD, PRT`) have no data for `Broad_Money_Growth` because that indicator is a Eurozone-level monetary aggregate, not reported per member state. Verified directly against the raw file — this is real data structure, not a defect.
- Ground truth: `Amir/ground_truth_imf.csv` — 1715 rows, columns `economy, year, banking_crisis, currency_crisis, sovereign_debt_crisis, crisis_name, is_crisis`. Verified directly against the raw file: exact same 49 economies, same 1990–2024 range, zero duplicate `(economy, year)` pairs, zero rows in the raw panel without a matching ground-truth row. `is_crisis` is not simply the OR of the three subtype flags — it also captures COVID-19 2020 as a crisis year for all 49 economies even though that event isn't a banking/currency/debt crisis by the Laeven-Valencia-style taxonomy the subtype columns follow; this is intentional, not a data bug. Total `is_crisis = 1` rows: 229 (13.4% of 1715).

## 4. Preprocessing pipeline

Applied in this order. No step drops a row, per §2.

1. **Sort** by `economy`, then `year`.
2. **Exchange rate transform:** `Exchange_Depreciation = groupby('economy')['Exchange_Rate'].pct_change() * 100`; drop the nominal `Exchange_Rate` column. This structurally produces a NaN for each country's first observed year — expected, not a data defect. Feature count stays at 14.
3. **Missingness quality filter — features only, no rows:** drop any feature with >50% missing (measured on raw missingness, before any filling; none qualify on this dataset — verify and record the check, don't skip it just because we expect it to pass). There is no row-level drop of any kind, per §2.
4. **Interpolation:** for every retained feature column, linear interpolation per country (`groupby('economy')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))`). Verified empirically: this fills interior gaps with true linear interpolation, and fills leading/trailing gaps by carrying the nearest valid value in that direction (e.g. `[NaN, NaN, 10, 20, NaN, 40, NaN, NaN]` → `[10, 10, 10, 20, 30, 40, 40, 40]`). This also resolves the year-1 `Exchange_Depreciation` NaN by extending from the next valid year.
5. **Remaining-NaN fallback:** interpolation can only leave a NaN behind when a country has zero valid observations at all for a column — exactly the 14 (economy, feature) pairs listed in §3 (490 cells: 4 features × 35 years for Nigeria, plus 1 feature × 35 years × 10 Eurozone countries). For any such remaining NaN, attempt per-country median first (a no-op in this exact scenario, kept as defense-in-depth in case future data breaks the "every country has ≥1 valid value" assumption); fall back to `KNNImputer` (k=5, default Euclidean) fit across the full feature matrix for whatever the median pass couldn't fill — this works because each affected row still has its other 10–13 features present, giving `KNNImputer` enough signal to find real nearest neighbors. Verified on the real data: zero NaNs remain after this step, and the row count stays at 1715.
6. **Scaling:** `RobustScaler()` fit on all 14 features across the full panel (median/IQR-based — chosen over `StandardScaler` for resilience to macro shocks like hyperinflation or currency collapse, consistent with the group's guidance that distance-based methods need outlier-resistant scaling).
7. **Crisis label (built last, never upstream of steps 1–6):** left-join the pipeline's DataFrame with `Amir/ground_truth_imf.csv` on `(economy, year)`, and set `crisis_label = is_crisis`. Raise an error rather than silently producing a NaN if any row fails to match (it shouldn't — coverage was verified in §3 — but a future change to either file's country codes should fail loudly, not produce a silently-wrong label).

Output artifact: `Amir/data_cleaned_v2.csv` — **exactly 1715 rows** (matching §2's mandatory row count, verified: 0 features dropped, 0 rows dropped), sorted `economy` then `year`, 14 scaled features + `economy` + `year` + `crisis_label`, with all 229 crisis rows preserved (13.4% of 1715).

## 5. Modeling (LOF) — fully label-free hyperparameter selection

Not implemented in this plan iteration (preprocessing only); recorded here so the eventual modeling plan implements the same design that was reviewed and approved, plus the mandatory conventions from §2:

- **`n_neighbors`:** computed across a k-range (k_lb–k_ub) reflecting plausible regional peer-group sizes; take the max LOF score per point across that range, per Breunig et al.'s (2000) own robust-range recommendation. No `crisis_label` reference anywhere in this step.
- **`contamination='auto'`:** sklearn's literature-grounded ~1.5 LOF-score threshold, used only to produce hard 0/1 predictions for confusion-matrix metrics (Precision/Recall/F1). The headline metrics (AUC-ROC, Average Precision) use the continuous LOF score directly and require no threshold.
- **Binary prediction convention (§2):** sklearn's `fit_predict` returns `-1` (outlier) / `1` (inlier) — this must be converted to `predicted_anomaly = 1` for anomaly/crisis, `predicted_anomaly = 0` for normal (i.e. `predicted_anomaly = (fit_predict_result == -1).astype(int)`), not left in sklearn's native encoding.
- **Random seed (§2):** any step involving randomness — the stratified-resampling robustness check below, and `KNNImputer`/`RobustScaler` calls carried over from preprocessing if re-run inside the modeling notebook — must fix `random_state=42`. (Note: `LocalOutlierFactor` and `RobustScaler` themselves have no randomness to seed; this applies to `train_test_split`, resampling, or any other stochastic step the modeling plan introduces.)
- **Evaluation:** one primary transductive LOF fit over the full standardized panel (matches how Breunig's method and the rest of the group's algorithms are normally run), plus a stratified-resampling robustness check (repeated stratified subsamples, refit transductively each time, report AUC-ROC/F1 as mean ± spread) instead of a single-shot number. Use `random_state=42` for the resampling.
- **Output file format (§2):** `Amir/hasil_lof_v2.csv` must contain exactly `['economy', 'year', 'anomaly_score', 'predicted_anomaly']` — no `crisis_label` column (that gets joined in later, at the group's majority-voting aggregation stage, not stored per-algorithm).
- **Reporting:** `laporan_lof_v2.md` states explicitly that LOF now uses zero label information anywhere before evaluation — the same level as DBSCAN — so the six-algorithm comparison has an honest, explicit anchor point per algorithm.

## 6. Deliverables and versioning

All new artifacts are created alongside the old ones in `Amir/` — nothing is overwritten, preserving the audit trail of the methodology correction:

- `Amir/data_cleaned_v2.csv` (preprocessing output, this plan)
- `Amir/preprocessing_lof_v2.py` (preprocessing pipeline code, this plan)
- `Amir/lof_v2.ipynb`, `Amir/laporan_lof_v2.md`, `Amir/hasil_lof_v2.csv` (columns per §5), `Amir/grid_search_lof_v2.csv`, `Amir/ringkasan_model_lof_v2.csv` (modeling phase, future plan)

## 7. Out of scope

- Other team members' pipelines (DBSCAN, Autoencoder, Isolation Forest, OCSVM, PCA) — not modified. The group-wide feature-engineering note (Exchange_Rate → depreciation, sorted interpolation, scaler choice per method) and the six mandatory output rules in §2 are Amir's to relay to teammates; this spec only implements them for Amir's own LOF work.
- Reconciling the old 329-row `data_with_labels.csv` crisis count — noted as a discrepancy in §1, not fixed at the group-data level.
