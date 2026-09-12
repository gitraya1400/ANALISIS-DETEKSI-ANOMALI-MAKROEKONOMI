# LOF Preprocessing v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Amir/preprocessing_lof_v2.py`, a tested pipeline that turns `Amir/raw_data_master.csv` into `Amir/data_cleaned_v2.csv` — the label-free-ready, RobustScaler-standardized panel with a correctly reconstructed `crisis_label` — ready for the (separately planned) LOF modeling phase.

**Architecture:** One pure-function-per-step module (`preprocessing_lof_v2.py`) with a `run_pipeline()` orchestrator at the bottom. Each function takes and returns a `pandas.DataFrame`, has no hidden state, and is unit-tested against small synthetic fixtures. A final integration test runs the whole pipeline against the real raw file and checks the exact verified numbers below.

**Tech Stack:** Python 3.14 (`python` / `py -3` on this machine — both resolve to the same interpreter with pandas 3.0.2 and scikit-learn 1.8.0 installed; `pytest` is **not** installed, so tests are plain scripts using `assert`, run directly with `python <file>.py`). No new dependencies are needed or should be added.

**Spec:** `docs/superpowers/specs/2026-09-12-lof-redo-design.md` (§2–§3)

## Global Constraints

- Scope is `Amir/` only. Do not modify any other team member's folder (`raya/`, `bram/`, `deka/`) or the group-level root files (`data_cleaned.csv`, `data_with_labels.csv`, etc.).
- Nothing existing gets overwritten. All new files use the `_v2` suffix.
- Missingness filter (feature >50%, then row >30%) runs on **raw** missingness, before interpolation/imputation — never after (a post-imputation check is a dead check, since imputation fills every remaining NaN by construction).
- `crisis_label` is built last, strictly from the six-episode rule in the spec, and never referenced by any earlier step.
- Scaler is `RobustScaler`, not `StandardScaler` (per group guidance for distance-based methods).
- Verified real-data numbers the integration test must match exactly: 0 features dropped; 90 rows dropped (all non-crisis); final shape 1625 rows; `crisis_label` sums to 175 (10.8% of 1625).

---

### Task 1: `load_raw_data`

**Files:**
- Create: `Amir/preprocessing_lof_v2.py`
- Test: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Produces: `load_raw_data(path: str) -> pd.DataFrame` — reads the CSV, sorts by `economy` then `year` ascending, resets the index, and ensures `year` is `int64`. Later tasks call this first.

- [ ] **Step 1: Write the failing test**

Create `Amir/test_preprocessing_lof_v2.py` with:

```python
import os
import tempfile

import pandas as pd

from preprocessing_lof_v2 import load_raw_data


def test_load_raw_data_sorts_by_economy_and_year():
    csv_text = (
        "economy,year,GDP_Growth,Exchange_Rate\n"
        "BBB,1991,2.0,50\n"
        "AAA,1990,1.0,100\n"
        "AAA,1991,1.5,110\n"
        "BBB,1990,2.5,45\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw.csv")
        with open(path, "w") as f:
            f.write(csv_text)
        df = load_raw_data(path)

    assert list(zip(df["economy"], df["year"])) == [
        ("AAA", 1990),
        ("AAA", 1991),
        ("BBB", 1990),
        ("BBB", 1991),
    ]
    assert df["year"].dtype.name == "int64"
    assert list(df.index) == [0, 1, 2, 3]
    print("test_load_raw_data_sorts_by_economy_and_year: OK")


if __name__ == "__main__":
    test_load_raw_data_sorts_by_economy_and_year()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ModuleNotFoundError: No module named 'preprocessing_lof_v2'` (the module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `Amir/preprocessing_lof_v2.py`:

```python
import pandas as pd


def load_raw_data(path):
    df = pd.read_csv(path)
    df = df.sort_values(["economy", "year"]).reset_index(drop=True)
    df["year"] = df["year"].astype("int64")
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: prints `test_load_raw_data_sorts_by_economy_and_year: OK` and exits 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add load_raw_data with economy/year sort"
```

---

### Task 2: `add_exchange_depreciation`

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Consumes: a DataFrame shaped like `load_raw_data()`'s output (has `economy`, `year`, `Exchange_Rate`).
- Produces: `add_exchange_depreciation(df: pd.DataFrame) -> pd.DataFrame` — adds `Exchange_Depreciation` (per-country year-over-year % change), drops `Exchange_Rate`. Later tasks depend on `Exchange_Depreciation` existing and `Exchange_Rate` being gone.

- [ ] **Step 1: Write the failing test**

Append to `Amir/test_preprocessing_lof_v2.py` (add the import and the test, and add the new call to the `__main__` block):

```python
from preprocessing_lof_v2 import load_raw_data, add_exchange_depreciation
```

```python
def test_add_exchange_depreciation_per_country_pct_change():
    df = pd.DataFrame({
        "economy": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "year": [1990, 1991, 1992, 1990, 1991],
        "Exchange_Rate": [100.0, 110.0, 121.0, 50.0, 45.0],
    })
    result = add_exchange_depreciation(df)

    assert "Exchange_Rate" not in result.columns
    assert "Exchange_Depreciation" in result.columns

    values = result["Exchange_Depreciation"].tolist()
    assert pd.isna(values[0])  # AAA 1990: no prior year
    assert abs(values[1] - 10.0) < 1e-9  # AAA 1991: (110-100)/100*100
    assert abs(values[2] - 10.0) < 1e-9  # AAA 1992: (121-110)/110*100
    assert pd.isna(values[3])  # BBB 1990: no prior year
    assert abs(values[4] - (-10.0)) < 1e-9  # BBB 1991: (45-50)/50*100
    print("test_add_exchange_depreciation_per_country_pct_change: OK")
```

Add the call `test_add_exchange_depreciation_per_country_pct_change()` inside the `if __name__ == "__main__":` block.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'add_exchange_depreciation'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
def add_exchange_depreciation(df):
    df = df.copy()
    df["Exchange_Depreciation"] = (
        df.groupby("economy")["Exchange_Rate"].pct_change() * 100
    )
    df = df.drop(columns=["Exchange_Rate"])
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: both tests print `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add Exchange_Rate to Exchange_Depreciation transform"
```

---

### Task 3: `drop_high_missing`

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Consumes: a DataFrame and a list of feature column names to check.
- Produces: `drop_high_missing(df: pd.DataFrame, feature_cols: list[str], feature_thresh: float = 0.5, row_thresh: float = 0.3) -> tuple[pd.DataFrame, list[str]]` — returns `(filtered_df, dropped_feature_names)`. Must run on **raw** (pre-interpolation) missingness. Later tasks (interpolation, imputation) only ever see the columns that survive this filter.

- [ ] **Step 1: Write the failing test**

Append import and test:

```python
from preprocessing_lof_v2 import (
    load_raw_data,
    add_exchange_depreciation,
    drop_high_missing,
)
```

```python
def test_drop_high_missing_drops_sparse_feature_and_sparse_row():
    df = pd.DataFrame({
        "feat_a": [None, None, None, 1.0],   # 3/4 = 75% missing -> dropped (feature)
        "feat_b": [1.0, 2.0, 3.0, None],
        "feat_c": [1.0, 2.0, 3.0, None],
        "feat_d": [1.0, 2.0, 3.0, 5.0],
    })
    # after dropping feat_a: row 3 has feat_b, feat_c missing = 2/3 = 66.7% -> dropped (row)
    # rows 0,1,2 have 0% missing among feat_b/c/d -> kept

    result, dropped = drop_high_missing(df, ["feat_a", "feat_b", "feat_c", "feat_d"])

    assert dropped == ["feat_a"]
    assert list(result.columns) == ["feat_b", "feat_c", "feat_d"]
    assert len(result) == 3
    assert result["feat_d"].tolist() == [1.0, 2.0, 3.0]
    print("test_drop_high_missing_drops_sparse_feature_and_sparse_row: OK")
```

Add the call to `__main__`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'drop_high_missing'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
def drop_high_missing(df, feature_cols, feature_thresh=0.5, row_thresh=0.3):
    df = df.copy()
    feature_missing_rate = df[feature_cols].isna().mean()
    dropped_features = feature_missing_rate[
        feature_missing_rate > feature_thresh
    ].index.tolist()

    remaining_cols = [c for c in feature_cols if c not in dropped_features]
    df = df.drop(columns=dropped_features)

    row_missing_rate = df[remaining_cols].isna().mean(axis=1)
    df = df[row_missing_rate <= row_thresh].reset_index(drop=True)

    return df, dropped_features
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: all three tests print `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add raw-missingness feature/row filter"
```

---

### Task 4: `interpolate_per_country`

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Consumes: a DataFrame (post `drop_high_missing`) and the list of surviving feature columns.
- Produces: `interpolate_per_country(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame` — linear-interpolates each feature within each country's time series, extending to fill leading/trailing gaps with the nearest valid value. Only fails to fill a cell when a country has zero valid observations for that column across its whole series (handled by Task 5).

- [ ] **Step 1: Write the failing test**

Append import and test:

```python
from preprocessing_lof_v2 import (
    load_raw_data,
    add_exchange_depreciation,
    drop_high_missing,
    interpolate_per_country,
)
import numpy as np
```

```python
def test_interpolate_per_country_fills_interior_and_edges():
    df = pd.DataFrame({
        "economy": ["AAA"] * 8 + ["BBB"] * 3,
        "year": list(range(1990, 1998)) + list(range(1990, 1993)),
        "val": [np.nan, np.nan, 10.0, 20.0, np.nan, 40.0, np.nan, np.nan]
        + [100.0, 100.0, 100.0],
    })

    result = interpolate_per_country(df, ["val"])

    aaa_values = result.loc[result["economy"] == "AAA", "val"].tolist()
    assert aaa_values == [10.0, 10.0, 10.0, 20.0, 30.0, 40.0, 40.0, 40.0]

    bbb_values = result.loc[result["economy"] == "BBB", "val"].tolist()
    assert bbb_values == [100.0, 100.0, 100.0]
    print("test_interpolate_per_country_fills_interior_and_edges: OK")
```

Add the call to `__main__`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'interpolate_per_country'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
def interpolate_per_country(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        df[col] = df.groupby("economy")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: all four tests print `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add per-country linear interpolation"
```

---

### Task 5: `impute_remaining`

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Consumes: a DataFrame (post `interpolate_per_country`) and the feature column list.
- Produces: `impute_remaining(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame` — fills any NaN interpolation couldn't reach (a country with zero valid values for that column) via per-country median first, then `KNNImputer(n_neighbors=5)` for whatever is still missing. Guarantees zero NaNs remain in `feature_cols` on return. Task 8's integration test relies on this guarantee.

- [ ] **Step 1: Write the failing test**

Append import and tests:

```python
from preprocessing_lof_v2 import (
    load_raw_data,
    add_exchange_depreciation,
    drop_high_missing,
    interpolate_per_country,
    impute_remaining,
)
```

```python
def test_impute_remaining_is_noop_when_nothing_missing():
    df = pd.DataFrame({
        "economy": ["AAA", "AAA", "BBB", "BBB"],
        "year": [1990, 1991, 1990, 1991],
        "g": [1.0, 2.0, 3.0, 4.0],
        "h": [10.0, 20.0, 30.0, 40.0],
    })
    result = impute_remaining(df, ["g", "h"])
    assert result["g"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert result["h"].tolist() == [10.0, 20.0, 30.0, 40.0]
    print("test_impute_remaining_is_noop_when_nothing_missing: OK")


def test_impute_remaining_fills_country_missing_entire_series():
    df = pd.DataFrame({
        "economy": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        "year": [1990, 1991, 1992, 1990, 1991, 1992],
        # AAA has zero valid values for "g" (whole series missing after interpolation)
        "g": [None, None, None, 3.0, 4.0, 5.0],
        "h": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })
    result = impute_remaining(df, ["g", "h"])
    assert result[["g", "h"]].isna().sum().sum() == 0
    print("test_impute_remaining_fills_country_missing_entire_series: OK")
```

Add both calls to `__main__`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'impute_remaining'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
from sklearn.impute import KNNImputer


def impute_remaining(df, feature_cols):
    df = df.copy()
    median_by_country = df.groupby("economy")[feature_cols].transform("median")
    df[feature_cols] = df[feature_cols].fillna(median_by_country)

    if df[feature_cols].isna().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: all six tests print `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add per-country-median then KNN fallback imputation"
```

---

### Task 6: `scale_features`

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Consumes: a fully-imputed DataFrame and the feature column list.
- Produces: `scale_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame` — replaces `feature_cols` in place with their `RobustScaler` transform (median/IQR-based). Non-feature columns (`economy`, `year`) pass through unchanged.

- [ ] **Step 1: Write the failing test**

Append import and test:

```python
from preprocessing_lof_v2 import (
    load_raw_data,
    add_exchange_depreciation,
    drop_high_missing,
    interpolate_per_country,
    impute_remaining,
    scale_features,
)
```

```python
def test_scale_features_matches_robust_scaler_formula():
    df = pd.DataFrame({
        "economy": ["AAA"] * 5,
        "year": [1990, 1991, 1992, 1993, 1994],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    # median=3, Q1=2, Q3=4, IQR=2 -> (x - 3) / 2
    result = scale_features(df, ["val"])
    expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for got, want in zip(result["val"].tolist(), expected):
        assert abs(got - want) < 1e-9
    assert result["economy"].tolist() == ["AAA"] * 5
    print("test_scale_features_matches_robust_scaler_formula: OK")
```

Add the call to `__main__`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'scale_features'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
from sklearn.preprocessing import RobustScaler


def scale_features(df, feature_cols):
    df = df.copy()
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: all seven tests print `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add RobustScaler standardization"
```

---

### Task 7: `build_crisis_label`

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`

**Interfaces:**
- Consumes: a DataFrame with `economy` and `year` columns (any point after Task 1 works; convention is to call this last).
- Produces: `build_crisis_label(df: pd.DataFrame) -> pd.DataFrame` — adds an integer `crisis_label` column (1/0) per the spec's six-episode rule. Also exports `CRISIS_EPISODES` at module level so Task 8 (and any future report code) can reference the same source of truth instead of duplicating the list.

- [ ] **Step 1: Write the failing test**

Append import and test:

```python
from preprocessing_lof_v2 import (
    load_raw_data,
    add_exchange_depreciation,
    drop_high_missing,
    interpolate_per_country,
    impute_remaining,
    scale_features,
    build_crisis_label,
)
```

```python
def test_build_crisis_label_matches_six_episode_rule():
    df = pd.DataFrame({
        "economy": ["IDN", "IDN", "RUS", "ARG", "USA", "FRA", "GRC", "JPN", "JPN"],
        "year":    [1997,  1999,  1998,  2001,  2008,  1990,  2011,  2020,  1990],
    })
    result = build_crisis_label(df)
    assert result["crisis_label"].tolist() == [1, 0, 1, 1, 1, 0, 1, 1, 0]
    print("test_build_crisis_label_matches_six_episode_rule: OK")
```

Add the call to `__main__`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'build_crisis_label'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
CRISIS_EPISODES = [
    (["IDN", "THA", "MYS", "KOR", "PHL"], [1997, 1998]),
    (["RUS"], [1998]),
    (["ARG"], [2001, 2002]),
    ("ALL", [2008, 2009]),
    (["GRC", "PRT", "IRL", "ESP", "ITA"], [2010, 2011, 2012]),
    ("ALL", [2020]),
]


def build_crisis_label(df):
    df = df.copy()
    label = pd.Series(0, index=df.index)
    for countries, years in CRISIS_EPISODES:
        if countries == "ALL":
            mask = df["year"].isin(years)
        else:
            mask = df["economy"].isin(countries) & df["year"].isin(years)
        label = label.where(~mask, 1)
    df["crisis_label"] = label
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: all eight tests print `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py
git commit -m "feat(lof-v2): add six-episode crisis_label construction"
```

---

### Task 8: `run_pipeline` orchestration + real-data integration test

**Files:**
- Modify: `Amir/preprocessing_lof_v2.py`
- Modify: `Amir/test_preprocessing_lof_v2.py`
- Creates at runtime (not committed): `Amir/data_cleaned_v2.csv`

**Interfaces:**
- Consumes: all seven functions from Tasks 1–7.
- Produces: `run_pipeline(raw_path: str, output_path: str) -> pd.DataFrame` — runs the full pipeline in order, writes the result to `output_path`, prints a short diagnostic summary, and returns the final DataFrame. This is the only function the future modeling plan needs to call.

- [ ] **Step 1: Write the failing test**

Append import and test:

```python
from preprocessing_lof_v2 import run_pipeline
```

```python
def test_run_pipeline_on_real_data_matches_verified_counts():
    raw_path = os.path.join(os.path.dirname(__file__), "raw_data_master.csv")
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "data_cleaned_v2.csv")
        result = run_pipeline(raw_path, output_path)

        assert os.path.exists(output_path)
        reloaded = pd.read_csv(output_path)
        assert len(reloaded) == len(result)

    feature_cols = [c for c in result.columns if c not in ("economy", "year", "crisis_label")]
    assert len(feature_cols) == 14
    assert result[feature_cols].isna().sum().sum() == 0
    assert len(result) == 1625
    assert int(result["crisis_label"].sum()) == 175
    print("test_run_pipeline_on_real_data_matches_verified_counts: OK")
```

Add the call to `__main__`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: `ImportError: cannot import name 'run_pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Amir/preprocessing_lof_v2.py`:

```python
RAW_FEATURE_COLS = [
    "GDP_Growth", "Inflation_CPI", "Unemployment", "Current_Account_GDP",
    "Reserves_Months_Imports", "FDI_Inflows_GDP", "Exports_GDP", "Imports_GDP",
    "Gross_Savings_GDP", "Investment_GDP", "Manufacturing_Value",
    "Domestic_Credit_GDP", "Broad_Money_Growth", "Exchange_Depreciation",
]


def run_pipeline(raw_path, output_path):
    df = load_raw_data(raw_path)
    df = add_exchange_depreciation(df)
    df, dropped_features = drop_high_missing(df, RAW_FEATURE_COLS)
    remaining_cols = [c for c in RAW_FEATURE_COLS if c not in dropped_features]
    df = interpolate_per_country(df, remaining_cols)
    df = impute_remaining(df, remaining_cols)
    df = scale_features(df, remaining_cols)
    df = build_crisis_label(df)
    df.to_csv(output_path, index=False)

    n_crisis = int(df["crisis_label"].sum())
    print(f"Dropped features (>50% missing): {dropped_features}")
    print(
        f"Rows: {len(df)}, Crisis rows: {n_crisis} "
        f"({n_crisis / len(df):.1%} of retained rows)"
    )
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        run_pipeline(sys.argv[1], sys.argv[2])
```

Note: `run_pipeline` references `RAW_FEATURE_COLS`, a name distinct from the test file's own `feature_cols` local variable — don't confuse the two when reading this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Amir && python test_preprocessing_lof_v2.py`
Expected: all nine tests print `OK`, exit 0. Also run the CLI path directly and confirm the printed summary matches the verified numbers:

Run: `cd Amir && python preprocessing_lof_v2.py raw_data_master.csv data_cleaned_v2.csv`
Expected output:
```
Dropped features (>50% missing): []
Rows: 1625, Crisis rows: 175 (10.8% of retained rows)
```

- [ ] **Step 5: Commit**

```bash
git add Amir/preprocessing_lof_v2.py Amir/test_preprocessing_lof_v2.py Amir/data_cleaned_v2.csv
git commit -m "feat(lof-v2): add run_pipeline orchestrator and generate data_cleaned_v2.csv"
```

---

## What comes after this plan

`Amir/data_cleaned_v2.csv` is the input the future LOF modeling plan (spec §4: label-free `n_neighbors` range selection, `contamination='auto'`, transductive-primary + stratified-resampling-robustness evaluation) will consume. That plan is intentionally not part of this one — this plan's sole deliverable is a correct, tested preprocessing pipeline.
