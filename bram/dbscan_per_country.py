# =============================================================================
# dbscan_per_country.py
# DBSCAN Anomaly Detection -- Pendekatan PER NEGARA
# Input  : output/data_scaled.csv  (14 fitur standar z-score)
# Output : output/hasil_per_country.csv
#          output/kdist_per_country_[eco].png  (sample negara)
#          output/anomaly_heatmap_percountry.png
#          output/anomaly_timeline_percountry.png
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: simpan file tanpa plt.show() blocking
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Konfigurasi ──────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join("output", "data_scaled.csv")
OUTPUT_DIR  = "output"
RANDOM_SEED = 42
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "hasil_per_country.csv")

# Parameter DBSCAN per-negara
# Dengan 35 titik per negara, pakai min_samples kecil
MIN_SAMPLES = 3
K_NEIGHBORS = 3       # untuk k-distance plot per negara
MIN_EPS     = 1.0     # batas bawah eps: dengan 14 fitur terstandarisasi,
                      # eps < 1.0 terlalu ketat dan menyebabkan anomaly rate > 30%

# Negara untuk plot k-distance (sample representatif)
SAMPLE_ECONOMIES = ["IDN", "ARG", "GRC", "USA", "CHN", "TUR"]


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data(path: str) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ["economy", "year"]]
    print(f"[LOAD] {df.shape[0]} baris, {len(feat_cols)} fitur")
    print(f"       Negara: {df['economy'].nunique()} | "
          f"Periode: {df['year'].min()}-{df['year'].max()}")
    return df, feat_cols


# =============================================================================
# 2. PILIH EPS OTOMATIS VIA K-DISTANCE (per negara)
# =============================================================================
def find_eps_country(X: np.ndarray, k: int) -> float:
    """
    Kneedle Algorithm per negara:
    Sort k-distance ascending, normalisasi ke [0,1], temukan titik
    dengan jarak tegak lurus maksimum ke garis diagonal (knee sejati).
    Minimum eps = MIN_EPS untuk mencegah anomaly rate yang tidak wajar.
    """
    if len(X) <= k:
        return MIN_EPS
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dist_asc = np.sort(dists[:, k-1])        # ASCENDING

    n = len(k_dist_asc)
    if n < 3:
        return max(float(np.median(k_dist_asc)), MIN_EPS)

    x_norm = np.linspace(0, 1, n)
    y_min, y_max = k_dist_asc.min(), k_dist_asc.max()
    if y_max == y_min:
        return max(float(y_min), MIN_EPS)

    y_norm   = (k_dist_asc - y_min) / (y_max - y_min)
    perp_dist = np.abs(y_norm - x_norm)
    knee_idx  = int(np.argmax(perp_dist))
    eps       = float(k_dist_asc[knee_idx])

    return max(eps, MIN_EPS)   # batas bawah: tidak boleh < MIN_EPS


# =============================================================================
# 3. HITUNG ANOMALY SCORE
# =============================================================================
def compute_anomaly_score(X: np.ndarray, labels: np.ndarray,
                           core_indices: np.ndarray) -> np.ndarray:
    """
    Anomaly score = jarak min ke core point terdekat, dinormalisasi [0, 1].
    Noise points (-1) cenderung jauh dari core points -> score tinggi.
    """
    if len(core_indices) == 0:
        return np.ones(len(X))

    core_pts = X[core_indices]
    nbrs = NearestNeighbors(n_neighbors=1).fit(core_pts)
    dists, _ = nbrs.kneighbors(X)
    raw = dists.ravel()

    # Normalisasi [0,1]
    if raw.max() == raw.min():
        return np.zeros(len(X))
    scores = (raw - raw.min()) / (raw.max() - raw.min())
    return scores


# =============================================================================
# 4. JALANKAN DBSCAN PER NEGARA
# =============================================================================
def run_dbscan_per_country(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Iterasi setiap economy, fit DBSCAN, kembalikan DataFrame hasil.
    """
    results = []
    economies = sorted(df["economy"].unique())
    print(f"\n[DBSCAN PER-NEGARA] Memproses {len(economies)} negara...")
    print("-" * 55)

    for eco in economies:
        mask   = df["economy"] == eco
        df_eco = df[mask].copy().reset_index(drop=True)
        X      = df_eco[feat_cols].values

        # Tentukan eps otomatis
        eps = find_eps_country(X, k=K_NEIGHBORS)

        # Fit DBSCAN
        db     = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="euclidean")
        labels = db.fit_predict(X)
        core_idx = db.core_sample_indices_

        n_noise = (labels == -1).sum()

        # Anomaly score
        scores = compute_anomaly_score(X, labels, core_idx)

        # Biner: -1 -> 1 (anomali), lainnya -> 0
        predicted = np.where(labels == -1, 1, 0)

        print(f"  {eco}: eps={eps:.3f} | noise={n_noise}/35 "
              f"({n_noise/len(labels)*100:.0f}%)")

        # Gabungkan ke hasil
        tmp = df_eco[["economy", "year"]].copy()
        tmp["anomaly_score"]     = np.round(scores, 6)
        tmp["predicted_anomaly"] = predicted
        results.append(tmp)

    df_result = pd.concat(results, ignore_index=True)
    return df_result


# =============================================================================
# 5. VISUALISASI
# =============================================================================
def plot_kdist_sample(df: pd.DataFrame, feat_cols: list,
                      economies: list, save_path: str):
    """K-distance plot untuk sample negara."""
    n = len(economies)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i, eco in enumerate(economies):
        X = df[df["economy"] == eco][feat_cols].values
        if len(X) <= K_NEIGHBORS:
            continue
        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(X)
        dists, _ = nbrs.kneighbors(X)
        k_dist = np.sort(dists[:, K_NEIGHBORS-1])[::-1]
        eps = find_eps_country(X, K_NEIGHBORS)

        axes[i].plot(k_dist, color="steelblue", linewidth=1.5)
        axes[i].axhline(eps, color="tomato", linestyle="--",
                        label=f"eps={eps:.3f}")
        axes[i].set_title(f"{eco}", fontweight="bold")
        axes[i].set_xlabel("Titik"); axes[i].set_ylabel(f"{K_NEIGHBORS}-Distance")
        axes[i].legend(fontsize=8)

    fig.suptitle("K-Distance Plot Sample Negara (Per-Country DBSCAN)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


def plot_anomaly_heatmap(df_result: pd.DataFrame, save_path: str):
    """Heatmap economy x year -- anomali per negara."""
    pivot = df_result.pivot_table(
        index="economy", columns="year",
        values="predicted_anomaly", aggfunc="max"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(22, 14))
    sns.heatmap(pivot, cmap="Reds", linewidths=0.2, linecolor="gray",
                ax=ax, cbar_kws={"label": "Anomali (1=Ya, 0=Tidak)"},
                vmin=0, vmax=1)

    # Garis vertikal krisis global
    years = list(pivot.columns)
    for crisis_yr, label in [(1997,"AFC"),(2008,"GFC"),(2020,"COVID")]:
        if crisis_yr in years:
            x_pos = years.index(crisis_yr) + 0.5
            ax.axvline(x_pos, color="navy", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.text(x_pos+0.1, -0.5, label, color="navy",
                    fontsize=8, fontweight="bold")

    ax.set_title("Peta Anomali Per Negara per Tahun (DBSCAN Per-Country)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Tahun"); ax.set_ylabel("Negara")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


def plot_anomaly_timeline(df_result: pd.DataFrame, save_path: str):
    """Jumlah negara anomali per tahun (per-country)."""
    timeline = (df_result.groupby("year")["predicted_anomaly"]
                         .sum().reset_index())
    timeline.columns = ["year", "n_anomali"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(timeline["year"], timeline["n_anomali"],
           color=["tomato" if v > 3 else "steelblue"
                  for v in timeline["n_anomali"]],
           edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Negara Anomali")
    ax.set_title("Jumlah Negara Anomali per Tahun (DBSCAN Per-Country)",
                 fontweight="bold")
    ax.set_xticks(timeline["year"])
    ax.set_xticklabels(timeline["year"], rotation=45, ha="right")

    for crisis_yr, label in [(1997,"AFC"),(2008,"GFC"),(2020,"COVID")]:
        if crisis_yr in timeline["year"].values:
            ax.axvline(crisis_yr, color="navy", linestyle="--", alpha=0.6)
            ax.text(crisis_yr+0.1, timeline["n_anomali"].max()*0.95,
                    label, color="navy", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


def plot_score_country(df_result: pd.DataFrame,
                        economies: list, save_path: str):
    """Line plot anomaly score per tahun untuk sample negara."""
    n = len(economies)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i, eco in enumerate(economies):
        sub = df_result[df_result["economy"] == eco].sort_values("year")
        color_pts = ["tomato" if a == 1 else "steelblue"
                     for a in sub["predicted_anomaly"]]
        axes[i].fill_between(sub["year"], sub["anomaly_score"],
                             alpha=0.2, color="steelblue")
        axes[i].plot(sub["year"], sub["anomaly_score"],
                     color="steelblue", linewidth=1.2)
        axes[i].scatter(sub["year"], sub["anomaly_score"],
                        c=color_pts, s=40, zorder=5)
        axes[i].set_title(f"{eco}", fontweight="bold")
        axes[i].set_xlabel("Tahun"); axes[i].set_ylabel("Anomaly Score")
        axes[i].set_ylim(0, 1.05)

        # Tandai titik anomali
        anomali = sub[sub["predicted_anomaly"] == 1]
        for _, row in anomali.iterrows():
            axes[i].annotate(str(int(row["year"])),
                             (row["year"], row["anomaly_score"]),
                             textcoords="offset points", xytext=(0, 6),
                             fontsize=6, color="tomato", ha="center")

    fig.suptitle("Anomaly Score per Tahun -- Sample Negara (DBSCAN Per-Country)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    np.random.seed(RANDOM_SEED)
    print("=" * 65)
    print("  DBSCAN PER NEGARA -- ANOMALY DETECTION")
    print("=" * 65)

    # 1. Load
    df, feat_cols = load_data(INPUT_PATH)

    # 2. K-distance plot sample negara
    plot_kdist_sample(
        df, feat_cols, SAMPLE_ECONOMIES,
        os.path.join(OUTPUT_DIR, "kdist_per_country_sample.png")
    )

    # 3. Jalankan DBSCAN per negara
    df_result = run_dbscan_per_country(df, feat_cols)

    # 4. Simpan CSV
    df_result.to_csv(OUTPUT_CSV, index=False)
    total_anomali = df_result["predicted_anomaly"].sum()
    total_rows    = len(df_result)
    print(f"\n[SAVE] Hasil -> {OUTPUT_CSV}")
    print(f"  Total anomali  : {total_anomali} "
          f"({total_anomali/total_rows*100:.2f}%)")
    print(f"  Total normal   : {total_rows - total_anomali}")

    # 5. Visualisasi
    plot_anomaly_heatmap(
        df_result,
        os.path.join(OUTPUT_DIR, "anomaly_heatmap_percountry.png")
    )
    plot_anomaly_timeline(
        df_result,
        os.path.join(OUTPUT_DIR, "anomaly_timeline_percountry.png")
    )
    plot_score_country(
        df_result, SAMPLE_ECONOMIES,
        os.path.join(OUTPUT_DIR, "anomaly_score_percountry_sample.png")
    )

    # 6. Summary per negara
    print("\n[INFO] Ringkasan anomali per negara:")
    summary = (df_result.groupby("economy")["predicted_anomaly"]
                        .agg(["sum", "count"])
                        .rename(columns={"sum": "n_anomali", "count": "n_total"})
               )
    summary["pct"] = (summary["n_anomali"] / summary["n_total"] * 100).round(1)
    summary_sorted = summary.sort_values("n_anomali", ascending=False)
    print(summary_sorted.to_string())

    # 7. Top tahun anomali lintas negara
    print("\n[INFO] Top 20 anomali per-negara (skor tertinggi):")
    top = (df_result[df_result["predicted_anomaly"] == 1]
           .sort_values("anomaly_score", ascending=False)
           .head(20))
    print(top.to_string(index=False))

    print("\n" + "=" * 65)
    print("  DBSCAN PER NEGARA SELESAI")
    print("=" * 65)

    return df_result


if __name__ == "__main__":
    df_result = main()
