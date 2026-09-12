# LOF Modeling v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LOF modeling phase for Amir's redo — `Amir/lof_v2.ipynb`, `Amir/hasil_lof_v2.csv`, `Amir/grid_search_lof_v2.csv`, `Amir/ringkasan_model_lof_v2.csv`, `Amir/laporan_lof_v2.md` — implementing spec §5's fully label-free hyperparameter selection on top of `Amir/data_cleaned_v2.csv` (already built by the preprocessing plan).

**Architecture:** Same delivery format as the completed preprocessing phase: a notebook (`lof_v2.ipynb`), not a standalone tested `.py` module — matches group convention (every other member's algorithm is a notebook) and the precedent already set by `Amir/preproccesingDataAmir.ipynb`. Built the same way: a driver script executes each "cell" for real, captures its stdout, then a second script assembles the captured cells into valid `.ipynb` JSON. All numbers in this plan were computed against the real `data_cleaned_v2.csv` before writing — nothing here is a guess.

**Tech Stack:** Python 3.14, pandas 3.0.2, scikit-learn 1.8.0 (`LocalOutlierFactor`, `StratifiedShuffleSplit`, `roc_auc_score`, `average_precision_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-12-lof-redo-design.md` §5 (modeling), §2 (mandatory rules)

## Global Constraints

- Scope is `Amir/` only. Input is `Amir/data_cleaned_v2.csv` (1715 rows, 14 scaled features, `crisis_label`) — already built, not modified by this plan.
- `crisis_label` is never passed into `X` (the feature matrix LOF fits on) and never used to choose `n_neighbors`, the k-range, or the anomaly threshold. It is used only for evaluation (Tasks 3–4), after scores/predictions already exist.
- `n_neighbors` candidates are the fixed set `k_range = [5, 10, 15, 20, 30, 50]` — the same six values already documented in the group's final proposal Bab III §3.2.3 grid-search range, reused here for continuity, but aggregated via Breunig et al.'s (2000) max-over-range method instead of label-AUC selection (spec §5). No grid search over labels happens anywhere in this plan.
- Anomaly threshold is a fixed `anomaly_score > 1.5` cutoff (sklearn's own literature-grounded `contamination='auto'` rule, applied directly to the aggregated score) — not derived from `crisis_label` or from the crisis ratio.
- Binary prediction convention (mandatory, spec §2): `predicted_anomaly = 1` for anomaly/crisis, `0` for normal.
- Random seed fixed at `42` for the one stochastic step in this plan: the stratified-resampling robustness check (Task 4).
- `Amir/hasil_lof_v2.csv` must contain **exactly** the columns `['economy', 'year', 'anomaly_score', 'predicted_anomaly']` (mandatory, spec §2) — no `crisis_label` column in this file.
- Verified real-data numbers every task's checks must match: primary fit gives `anomaly_score` range [0.979, 103.293], 171 rows flagged (10.0%), AUC-ROC 0.7099, AP 0.2427, Precision 0.2456, Recall 0.1834, F1 0.2100, confusion matrix `[[1357, 129], [187, 42]]`. Robustness check (20 resamples, 80% each, `random_state=42`): AUC-ROC 0.7068 ± 0.0106, F1 0.2265 ± 0.0153.

---

### Task 1: Load data and compute per-k transductive LOF scores

**Files:**
- Create: `Amir/lof_v2_notebook_build.py` (scratch driver script, not committed — see Task 6)
- Produces (in-memory, used by later tasks): `X` (feature matrix, 1715×14), `y` (`crisis_label`, kept aside), `scores` (1715×6 matrix, one column per k in `k_range`)

**Interfaces:**
- Consumes: `Amir/data_cleaned_v2.csv`.
- Produces: for later tasks — `scores[:, i]` is the positive-scale LOF score (`-negative_outlier_factor_`) at `k_range[i]`, fit transductively (`novelty=False`) on the full 1715-row panel.

- [ ] **Step 1: Write and run the cell**

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_csv('data_cleaned_v2.csv')
feature_cols = [c for c in df.columns if c not in ('economy', 'year', 'crisis_label')]
X = df[feature_cols].values
y = df['crisis_label'].values

k_range = [5, 10, 15, 20, 30, 50]
scores = np.zeros((len(df), len(k_range)))
for i, k in enumerate(k_range):
    lof = LocalOutlierFactor(n_neighbors=k, metric='euclidean', novelty=False)
    lof.fit_predict(X)
    scores[:, i] = -lof.negative_outlier_factor_

print(f'Fitur dipakai ({len(feature_cols)}): {feature_cols}')
print(f'crisis_label TIDAK ada di X: {"crisis_label" not in feature_cols}')
print(f'Shape scores: {scores.shape}')
for i, k in enumerate(k_range):
    print(f'k={k}: mean={scores[:, i].mean():.3f} min={scores[:, i].min():.3f} max={scores[:, i].max():.3f}')
```

- [ ] **Step 2: Verify output**

Expected (verified against real data):
```
Fitur dipakai (14): ['GDP_Growth', 'Inflation_CPI', 'Unemployment', 'Current_Account_GDP', 'Reserves_Months_Imports', 'FDI_Inflows_GDP', 'Exports_GDP', 'Imports_GDP', 'Gross_Savings_GDP', 'Investment_GDP', 'Manufacturing_Value', 'Domestic_Credit_GDP', 'Broad_Money_Growth', 'Exchange_Depreciation']
crisis_label TIDAK ada di X: True
Shape scores: (1715, 6)
k=5: mean=1.146 min=... max=...
k=10: mean=1.151 ...
k=15: mean=1.172 ...
k=20: mean=1.201 ...
k=30: mean=1.269 ...
k=50: mean=1.362 ...
```
`crisis_label TIDAK ada di X` must print `True` — if `False`, stop immediately, this is a label-leak regression.

---

### Task 2: Aggregate max-over-k score, threshold, save `hasil_lof_v2.csv`

**Files:**
- Create: `Amir/hasil_lof_v2.csv`

**Interfaces:**
- Consumes: `scores`, `k_range`, `df` from Task 1.
- Produces: `df['anomaly_score']`, `df['predicted_anomaly']` (added to the in-memory `df` for later tasks), and the on-disk `hasil_lof_v2.csv`.

- [ ] **Step 1: Write and run the cell**

```python
df['anomaly_score'] = scores.max(axis=1)
df['predicted_anomaly'] = (df['anomaly_score'] > 1.5).astype(int)

hasil = df[['economy', 'year', 'anomaly_score', 'predicted_anomaly']]
assert list(hasil.columns) == ['economy', 'year', 'anomaly_score', 'predicted_anomaly']
assert len(hasil) == 1715
hasil.to_csv('hasil_lof_v2.csv', index=False)

print(f'anomaly_score: min={df["anomaly_score"].min():.4f} max={df["anomaly_score"].max():.4f} mean={df["anomaly_score"].mean():.4f}')
print(f'predicted_anomaly: {int(df["predicted_anomaly"].sum())} baris ({df["predicted_anomaly"].mean():.1%})')
print('Disimpan ke hasil_lof_v2.csv dengan kolom:', list(hasil.columns))
```

- [ ] **Step 2: Verify output**

Expected (verified against real data):
```
anomaly_score: min=0.9785 max=103.2925 mean=1.4187
predicted_anomaly: 171 baris (10.0%)
Disimpan ke hasil_lof_v2.csv dengan kolom: ['economy', 'year', 'anomaly_score', 'predicted_anomaly']
```

---

### Task 3: Primary evaluation against `crisis_label`

**Files:**
- No new file yet (feeds Task 5's `ringkasan_model_lof_v2.csv` and Task 7's report).

**Interfaces:**
- Consumes: `df` (with `anomaly_score`, `predicted_anomaly`, `crisis_label`) from Task 2; `Amir/ground_truth_imf.csv` for the per-crisis-name breakdown.
- Produces: primary metrics (`auc`, `ap`, `prec`, `rec`, `f1`, confusion matrix), `per_crisis` DataFrame, `top10` Series — all consumed by later tasks.

This is the only place `crisis_label` is used — strictly for evaluation, never for choosing `k_range`, the aggregation rule, or the `1.5` threshold (those were fixed in Tasks 1–2 without looking at it).

- [ ] **Step 1: Write and run the cell**

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

y = df['crisis_label'].values
auc = roc_auc_score(y, df['anomaly_score'])
ap = average_precision_score(y, df['anomaly_score'])
prec = precision_score(y, df['predicted_anomaly'])
rec = recall_score(y, df['predicted_anomaly'])
f1 = f1_score(y, df['predicted_anomaly'])
cm = confusion_matrix(y, df['predicted_anomaly'])

print(f'AUC-ROC   = {auc:.4f}')
print(f'AP        = {ap:.4f}')
print(f'Precision = {prec:.4f}')
print(f'Recall    = {rec:.4f}')
print(f'F1-Score  = {f1:.4f}')
print('Confusion matrix [[TN, FP], [FN, TP]]:')
print(cm)

gt = pd.read_csv('ground_truth_imf.csv')[['economy', 'year', 'crisis_name', 'is_crisis']]
merged = df.merge(gt, on=['economy', 'year'], how='left')
per_crisis = merged[merged['is_crisis'] == 1].groupby('crisis_name').agg(
    observasi=('is_crisis', 'size'),
    terdeteksi=('predicted_anomaly', 'sum'),
).reset_index()
per_crisis['detection_rate_pct'] = (per_crisis['terdeteksi'] / per_crisis['observasi'] * 100).round(1)
per_crisis = per_crisis.sort_values('observasi', ascending=False)
n_unnamed = int(merged[(merged['is_crisis'] == 1) & (merged['crisis_name'].isna())].shape[0])
print(f'\nPer-crisis breakdown ({len(per_crisis)} episode bernama, {n_unnamed} baris krisis tanpa nama episode):')
print(per_crisis.to_string(index=False))

top10 = df.groupby('economy')['predicted_anomaly'].sum().sort_values(ascending=False).head(10)
print('\nTop 10 negara dengan anomali terbanyak:')
print(top10.to_string())
```

- [ ] **Step 2: Verify output**

Expected (verified against real data):
```
AUC-ROC   = 0.7099
AP        = 0.2427
Precision = 0.2456
Recall    = 0.1834
F1-Score  = 0.2100
Confusion matrix [[TN, FP], [FN, TP]]:
[[1357  129]
 [ 187   42]]
```
Per-crisis breakdown: 27 named episodes, 22 unnamed crisis rows (`crisis_name` null in `ground_truth_imf.csv` but `is_crisis = 1`). Top row must be `COVID-19 Global Shock` (49 observasi, 14 terdeteksi, 28.6%), second `Global Financial Crisis` (47 observasi, 3 terdeteksi, 6.4%).
Top 10 countries by `predicted_anomaly` count, descending: `SAU 14, HUN 11, NLD 9, IRL 9, RUS 9, NGA 7, BRA 6, CHE 6, ARG 6, ROU 6`.

---

### Task 4: Stratified-resampling robustness check

**Files:**
- No new file yet (feeds Task 5 and Task 7).

**Interfaces:**
- Consumes: `X`, `y`, `k_range` from Tasks 1–3.
- Produces: `aucs`, `f1s` (arrays of 20 values each) — consumed by Task 5's `ringkasan_model_lof_v2.csv` and Task 7's report.

Repeats the exact label-free procedure from Tasks 1–2 (same `k_range`, same max-aggregation, same `1.5` threshold) on 20 stratified 80% subsamples, to show the primary result isn't a lucky single draw. `crisis_label` (`y`) is used only to stratify the sampling and to score each subsample afterward — never to pick a parameter.

- [ ] **Step 1: Write and run the cell**

```python
from sklearn.model_selection import StratifiedShuffleSplit

def lof_max_score(X_sub):
    sub_scores = np.zeros((len(X_sub), len(k_range)))
    for i, k in enumerate(k_range):
        k_eff = min(k, len(X_sub) - 1)
        lof = LocalOutlierFactor(n_neighbors=k_eff, metric='euclidean', novelty=False)
        lof.fit_predict(X_sub)
        sub_scores[:, i] = -lof.negative_outlier_factor_
    return sub_scores.max(axis=1)

sss = StratifiedShuffleSplit(n_splits=20, train_size=0.8, random_state=42)
aucs, f1s = [], []
for train_idx, _ in sss.split(X, y):
    Xs, ys = X[train_idx], y[train_idx]
    s = lof_max_score(Xs)
    pred = (s > 1.5).astype(int)
    aucs.append(roc_auc_score(ys, s))
    f1s.append(f1_score(ys, pred))

aucs = np.array(aucs)
f1s = np.array(f1s)
print(f'AUC-ROC: mean={aucs.mean():.4f} std={aucs.std():.4f} min={aucs.min():.4f} max={aucs.max():.4f}')
print(f'F1     : mean={f1s.mean():.4f} std={f1s.std():.4f} min={f1s.min():.4f} max={f1s.max():.4f}')
```

- [ ] **Step 2: Verify output**

Expected (verified against real data, exact with `random_state=42`):
```
AUC-ROC: mean=0.7068 std=0.0106 min=0.6846 max=0.7264
F1     : mean=0.2265 std=0.0153 min=0.2063 max=0.2620
```
The tight spread (std ≈ 0.01 on AUC-ROC) is the evidence the primary result in Task 3 is stable, not a single lucky fit — state this explicitly in Task 7's report.

---

### Task 5: Save `grid_search_lof_v2.csv` and `ringkasan_model_lof_v2.csv`

**Files:**
- Create: `Amir/grid_search_lof_v2.csv`
- Create: `Amir/ringkasan_model_lof_v2.csv`

**Interfaces:**
- Consumes: `scores`, `k_range` (Task 1), primary metrics (Task 3), robustness arrays (Task 4).
- Produces: the two CSV files, no further consumers in this plan (Task 7's report reads them for narrative, not programmatically).

`grid_search_lof_v2.csv` keeps its old filename for audit-trail continuity (spec §6) but its content changes meaning: it is now a **diagnostic sweep over the fixed k-range**, not a label-AUC grid search that picks a winner — no column in it is used to select `k_range` or the threshold, both already fixed in Tasks 1–2.

- [ ] **Step 1: Write and run the cell**

```python
grid_search = pd.DataFrame({
    'n_neighbors': k_range,
    'mean_lof_score': scores.mean(axis=0),
    'pct_score_over_1.5': (scores > 1.5).mean(axis=0) * 100,
})
grid_search.to_csv('grid_search_lof_v2.csv', index=False)
print(grid_search.to_string(index=False))

ringkasan = pd.DataFrame([{
    'metode': 'Local Outlier Factor (label-free)',
    'n_neighbors_range': str(k_range),
    'aggregation': 'max LOF score across n_neighbors range (Breunig et al. 2000)',
    'contamination_rule': 'anomaly_score > 1.5 (sklearn contamination=auto equivalent)',
    'label_used_in_tuning': False,
    'label_used_in_threshold': False,
    'random_state': 42,
    'n_rows': len(df),
    'n_predicted_anomaly': int(df['predicted_anomaly'].sum()),
    'auc_roc_primary': round(auc, 4),
    'average_precision_primary': round(ap, 4),
    'precision_primary': round(prec, 4),
    'recall_primary': round(rec, 4),
    'f1_primary': round(f1, 4),
    'auc_roc_resample_mean': round(aucs.mean(), 4),
    'auc_roc_resample_std': round(aucs.std(), 4),
    'f1_resample_mean': round(f1s.mean(), 4),
    'f1_resample_std': round(f1s.std(), 4),
}])
ringkasan.to_csv('ringkasan_model_lof_v2.csv', index=False)
print()
print(ringkasan.T.to_string())
```

- [ ] **Step 2: Verify output**

Both files exist, `grid_search_lof_v2.csv` has 6 rows (one per k), `ringkasan_model_lof_v2.csv` has exactly 1 row with `label_used_in_tuning = False` and `label_used_in_threshold = False`.

---

### Task 6: Assemble `lof_v2.ipynb`

**Files:**
- Create: `Amir/lof_v2.ipynb`

**Interfaces:**
- Consumes: the exact cell sources and captured stdout from Tasks 1–5 (run for real, not re-typed from memory).

Same two-script pattern as the preprocessing phase:

- [ ] **Step 1:** Write a driver script (scratch, in the scratchpad directory) that runs each Task 1–5 cell body in one shared namespace via `exec`, capturing `stdout` per cell with `contextlib.redirect_stdout`, and dumps a JSON list of `{source, stdout}` per cell — same structure as `build_notebook.py` used for the preprocessing notebook.
- [ ] **Step 2:** Run it. Confirm every printed number matches the "Expected" blocks in Tasks 1–5 exactly (not approximately — `random_state=42` makes Task 4's numbers exactly reproducible).
- [ ] **Step 3:** Write an assembler script (same pattern as `assemble_notebook.py`) that turns the captured cells into valid nbformat 4 JSON, with markdown section headers per task (mirroring `preproccesingDataAmir.ipynb`'s style: numbered `##` headers, a title cell explaining scope and the label-free design, a closing summary cell).
- [ ] **Step 4:** Run it, writing to `Amir/lof_v2.ipynb`. Validate the output is well-formed JSON and every code cell has non-empty `outputs`.

---

### Task 7: Write `laporan_lof_v2.md`

**Files:**
- Create: `Amir/laporan_lof_v2.md`

**Interfaces:**
- Consumes: all numbers verified in Tasks 1–5. No new computation — this is a narrative writeup.

Mirror `Amir/laporan_lof.md`'s structure (the old report) section-for-section, but with corrected content:

- [ ] **Step 1:** Write the report with these required elements (no placeholders — use the exact verified numbers from Tasks 1–5):
  - State explicitly, early: LOF now uses **zero label information** anywhere before evaluation — `n_neighbors` range, aggregation rule, and the `1.5` threshold are all fixed independent of `crisis_label`, matching DBSCAN's level on the label-usage spectrum (this directly answers the original critique that started this redo).
  - Data section: `raw_data_master.csv`, `ground_truth_imf.csv`, `data_cleaned_v2.csv` (1715 rows, 229 crisis rows / 13.4%).
  - Method section: the k-range `{5, 10, 15, 20, 30, 50}` and why (reused from the group's final proposal Bab III §3.2.3 grid-search range, but aggregated via max instead of label-AUC selection), the `1.5` threshold and why (sklearn's own `contamination='auto'` rule).
  - Results: primary metrics table (Task 3), confusion matrix, per-crisis-name breakdown table (Task 3), top-10 countries table (Task 3).
  - Robustness section: the 20-resample check (Task 4) — report mean ± std for AUC-ROC and F1, and state that the tight spread confirms the primary result is stable.
  - Explicit limitation note: this notebook's methodology (label-free tuning, `RobustScaler`) intentionally diverges from the group's currently-submitted final proposal Bab III §3.2.1 (`StandardScaler` for all six algorithms) and §3.2.3 (label-AUC grid search for LOF) — per Amir's decision, the proposal document will be revised separately later; this is not an undocumented accidental drift.
  - Conclusion: compare against the old (leaky) numbers honestly — old F1 was 0.3435 but measured fit-to-label rather than generalization; new F1 is 0.2100 (primary) / 0.2265 ± 0.0153 (resampled) and is the honest, defensible number.
- [ ] **Step 2:** Re-read the finished report once for internal consistency: every number quoted must match Tasks 1–5's verified outputs exactly, not be retyped from memory.

---

## What comes after this plan

`Amir/hasil_lof_v2.csv`, `Amir/laporan_lof_v2.md`, and the other artifacts are Amir's finished individual contribution to the group's majority-voting stage. Reconciling the group's final proposal Bab III text (§3.2.1, §3.2.3) with this new methodology is explicitly deferred, per Amir's decision — not part of this plan.
