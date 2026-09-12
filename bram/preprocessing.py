# =============================================================================
# preprocessing.py
# Preprocessing pipeline untuk DBSCAN Anomaly Detection
# Data: Makroekonomi 49 Negara (1990-2024)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# --- Konfigurasi -----------------------------------------------------------
DATA_PATH               = "raw_data_master.csv"
OUTPUT_DIR              = "output"
PCA_VARIANCE_THRESHOLD  = 0.90   # simpan komponen yang menjelaskan >= 90% varians

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
def load_data(path: str) -> pd.DataFrame:
    """Membaca raw data dari CSV."""
    df = pd.read_csv(path)
    print(f"[LOAD] Data berhasil dibaca: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


# ===========================================================================
# 2. FEATURE ENGINEERING -- Exchange Rate -> Depreciation (% YoY)
# ===========================================================================
def add_exchange_depreciation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengganti Exchange_Rate (level nominal) dengan Exchange_Depreciation
    yaitu persen perubahan nilai tukar year-over-year per negara.

    Nilai positif  = depresiasi (mata uang melemah)
    Nilai negatif  = apresiasi  (mata uang menguat)
    """
    df = df.sort_values(["economy", "year"]).copy()

    df["Exchange_Depreciation"] = (
        df.groupby("economy")["Exchange_Rate"]
          .pct_change() * 100
    )

    df.drop(columns=["Exchange_Rate"], inplace=True)

    missing_new = df["Exchange_Depreciation"].isna().sum()
    print(f"[FE] Exchange_Rate -> Exchange_Depreciation (% YoY). "
          f"Missing setelah transformasi: {missing_new}")
    return df


# ===========================================================================
# 3. IMPUTASI MISSING VALUES
# ===========================================================================
def impute_missing(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Strategi imputasi dua-tahap:
      1. Median per economy  (data time-series per negara lebih relevan)
      2. Median global       (fallback jika masih NaN)
    """
    df = df.copy()

    print("\n[IMPUTASI] Missing values SEBELUM imputasi:")
    before = df[feature_cols].isnull().sum()
    print(before[before > 0].to_string())

    # Tahap 1: median per economy
    df[feature_cols] = df.groupby("economy")[feature_cols].transform(
        lambda x: x.fillna(x.median())
    )

    # Tahap 2: median global (fallback)
    remaining = df[feature_cols].isnull().sum()
    if remaining.sum() > 0:
        print("\n[IMPUTASI] Sisa missing -> fallback median global:")
        print(remaining[remaining > 0].to_string())
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    total = df[feature_cols].isnull().sum().sum()
    print(f"\n[IMPUTASI] Missing values SETELAH imputasi: {total}")
    return df


# ===========================================================================
# 4. STANDARISASI (Z-SCORE)
# ===========================================================================
def standardize(df: pd.DataFrame, feature_cols: list):
    """
    Standarisasi fitur menggunakan StandardScaler (mean=0, std=1).
    Wajib untuk DBSCAN karena berbasis jarak Euclidean.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"\n[STANDARISASI] StandardScaler diterapkan pada {len(feature_cols)} fitur.")
    return df_scaled, scaler


# ===========================================================================
# 5. PCA -- REDUKSI DIMENSI
# ===========================================================================
def apply_pca(df_scaled: pd.DataFrame, feature_cols: list,
              variance_threshold: float = 0.90):
    """
    Menerapkan PCA. Jumlah komponen dipilih otomatis berdasarkan
    threshold varians kumulatif.
    """
    pca_full = PCA()
    pca_full.fit(df_scaled[feature_cols])

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= variance_threshold) + 1)

    print(f"\n[PCA] Threshold varians  : {variance_threshold*100:.0f}%")
    print(f"[PCA] Komponen dipilih   : {n_components} "
          f"(dari {len(feature_cols)} fitur asli)")
    print(f"[PCA] Varians tercakup   : {cumvar[n_components-1]*100:.2f}%")

    pca = PCA(n_components=n_components, random_state=42)
    pca_matrix = pca.fit_transform(df_scaled[feature_cols])

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = df_scaled[["economy", "year"]].copy().reset_index(drop=True)
    df_pca[pca_cols] = pca_matrix

    print("\n[PCA] Explained variance per komponen:")
    for i, (var, cumv) in enumerate(
        zip(pca.explained_variance_ratio_, cumvar[:n_components])
    ):
        print(f"  PC{i+1}: {var*100:.2f}%  (kumulatif: {cumv*100:.2f}%)")

    return df_pca, pca, n_components


# ===========================================================================
# 6. VISUALISASI DIAGNOSTIK
# ===========================================================================
def plot_missing_heatmap(df: pd.DataFrame, feature_cols: list):
    pivot = df.pivot_table(
        index="economy", values=feature_cols,
        aggfunc=lambda x: x.isna().sum()
    )
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, cmap="Reds", linewidths=0.3, ax=ax,
                cbar_kws={"label": "Jumlah Missing"})
    ax.set_title("Missing Values per Kolom per Negara", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "missing_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"[VIZ] Saved: {path}")
    plt.show()
    plt.close()


def plot_distribution_comparison(df_imputed: pd.DataFrame,
                                 df_scaled: pd.DataFrame,
                                 feature_cols: list):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    df_imputed[feature_cols].boxplot(ax=axes[0], rot=45)
    axes[0].set_title("Distribusi Fitur -- SEBELUM Standarisasi", fontweight="bold")
    axes[0].set_ylabel("Nilai Asli")

    df_scaled[feature_cols].boxplot(ax=axes[1], rot=45)
    axes[1].set_title("Distribusi Fitur -- SETELAH Standarisasi (z-score)", fontweight="bold")
    axes[1].set_ylabel("z-score")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "distribution_before_after.png")
    plt.savefig(path, dpi=150)
    print(f"[VIZ] Saved: {path}")
    plt.show()
    plt.close()


def plot_pca_variance(pca: PCA):
    n = len(pca.explained_variance_ratio_)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(1, n+1), pca.explained_variance_ratio_ * 100,
            color="steelblue", alpha=0.7, label="Variance per PC")
    ax2.plot(range(1, n+1), cumvar * 100,
             color="tomato", marker="o", linewidth=2, label="Kumulatif")
    ax2.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90% threshold")

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color="steelblue")
    ax2.set_ylabel("Kumulatif Variance (%)", color="tomato")
    ax1.set_title("PCA -- Scree Plot", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "pca_scree_plot.png")
    plt.savefig(path, dpi=150)
    print(f"[VIZ] Saved: {path}")
    plt.show()
    plt.close()


def plot_pca_scatter(df_pca: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8))
    economies = df_pca["economy"].unique()
    cmap = plt.cm.get_cmap("tab20", len(economies))

    for i, eco in enumerate(economies):
        mask = df_pca["economy"] == eco
        ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                   color=cmap(i), label=eco, s=30, alpha=0.7)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Scatter Plot PC1 vs PC2 (setelah PCA)", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2, fontsize=7)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "pca_scatter.png")
    plt.savefig(path, dpi=150)
    print(f"[VIZ] Saved: {path}")
    plt.show()
    plt.close()


# ===========================================================================
# 7. SIMPAN HASIL
# ===========================================================================
def save_outputs(df_scaled: pd.DataFrame, df_pca: pd.DataFrame):
    path_scaled = os.path.join(OUTPUT_DIR, "data_scaled.csv")
    df_scaled.to_csv(path_scaled, index=False)
    print(f"\n[SAVE] Data standarisasi -> {path_scaled}")

    path_pca = os.path.join(OUTPUT_DIR, "data_pca.csv")
    df_pca.to_csv(path_pca, index=False)
    print(f"[SAVE] Data PCA          -> {path_pca}")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    print("=" * 65)
    print("  PREPROCESSING PIPELINE -- DBSCAN ANOMALY DETECTION")
    print("=" * 65)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Feature Engineering: Exchange Rate -> % YoY Depreciation
    df = add_exchange_depreciation(df)

    # 3. Tentukan kolom fitur
    feature_cols = [c for c in df.columns if c not in ["economy", "year"]]
    print(f"\n[INFO] Fitur yang digunakan ({len(feature_cols)}):")
    print("  " + ", ".join(feature_cols))

    # 4. Visualisasi missing sebelum imputasi
    plot_missing_heatmap(df, feature_cols)

    # 5. Imputasi
    df_imputed = impute_missing(df, feature_cols)

    # 6. Standarisasi
    df_scaled, scaler = standardize(df_imputed, feature_cols)

    # 7. Visualisasi distribusi sebelum vs sesudah standarisasi
    plot_distribution_comparison(df_imputed, df_scaled, feature_cols)

    # 8. PCA
    df_pca, pca, n_components = apply_pca(
        df_scaled, feature_cols,
        variance_threshold=PCA_VARIANCE_THRESHOLD
    )
    plot_pca_variance(pca)
    plot_pca_scatter(df_pca)

    # 9. Simpan output
    save_outputs(df_scaled, df_pca)

    print("\n" + "=" * 65)
    print("  PREPROCESSING SELESAI")
    print("=" * 65)
    print(f"\n  Output tersimpan di: ./{OUTPUT_DIR}/")
    print("  data_scaled.csv  -> gunakan untuk DBSCAN per-negara")
    print("  data_pca.csv     -> gunakan untuk DBSCAN global")
    print("\n  Lanjutkan ke: dbscan_global.py  &  dbscan_per_country.py")

    return df_scaled, df_pca, feature_cols, scaler, pca


if __name__ == "__main__":
    df_scaled, df_pca, feature_cols, scaler, pca = main()
