# =============================================================================
# dbscan_global.py
# DBSCAN Anomaly Detection -- Pendekatan GLOBAL (semua negara digabung)
# Input  : output/data_pca.csv  (9 Principal Components)
# Output : output/hasil_global.csv
#          output/kdist_global.png
#          output/anomaly_global_pc1pc2.png
#          output/anomaly_timeline_global.png
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: simpan file tanpa plt.show() blocking
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import os

warnings.filterwarnings("ignore")

# ── Konfigurasi ──────────────────────────────────────────────────────────────
INPUT_PATH   = os.path.join("output", "data_pca.csv")
OUTPUT_DIR   = "output"
RANDOM_SEED  = 42
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "hasil_global.csv")

# Parameter DBSCAN -- akan di-tune via k-distance plot
# Aturan umum: min_samples ~ ln(n) atau 2*n_dims
K_NEIGHBORS  = 9          # = n_components PCA (dipakai juga sebagai min_samples)
MIN_SAMPLES  = 9
EPS_AUTO     = True        # True = cari eps otomatis via Kneedle algorithm
SCALE_PCA    = True        # Standarisasi ulang komponen PCA agar jarak seimbang


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data(path: str) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(path)
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    print(f"[LOAD] {df.shape[0]} baris, {len(pc_cols)} PC features: {pc_cols}")
    return df, pc_cols


# =============================================================================
# 2. PILIH EPS OTOMATIS VIA K-DISTANCE PLOT (Kneedle Algorithm)
# =============================================================================
def find_eps(X: np.ndarray, k: int, save_path: str) -> float:
    """
    Kneedle Algorithm:
    1. Hitung k-distance setiap titik
    2. Urutkan ASCENDING
    3. Normalisasi sumbu x dan y ke [0, 1]
    4. Temukan titik dengan jarak tegak lurus maksimum ke garis diagonal
       (0,0)-(1,1) -> ini adalah "knee" sejati dari kurva
    5. eps = nilai k-distance di titik knee tersebut
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist_asc = np.sort(distances[:, k-1])        # ASCENDING

    n = len(k_dist_asc)
    x_norm = np.linspace(0, 1, n)
    y_min, y_max = k_dist_asc.min(), k_dist_asc.max()
    y_norm = (k_dist_asc - y_min) / (y_max - y_min + 1e-12)

    # Jarak tegak lurus ke garis diagonal (0,0)-(1,1): |y-x| / sqrt(2)
    perp_dist = np.abs(y_norm - x_norm)
    knee_idx  = int(np.argmax(perp_dist))
    eps_auto  = float(k_dist_asc[knee_idx])

    # Plot (kurva ascending = konvensi standar k-distance plot DBSCAN)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n), k_dist_asc, color="steelblue", linewidth=1.5,
            label=f"{k}-Distance (ascending)")
    ax.axhline(eps_auto, color="tomato", linestyle="--",
               label=f"eps (Kneedle) = {eps_auto:.4f}")
    ax.axvline(knee_idx, color="gray", linestyle=":", alpha=0.7,
               label=f"knee idx = {knee_idx}")
    ax.set_xlabel("Titik (diurutkan ascending berdasarkan jarak)")
    ax.set_ylabel(f"{k}-Distance")
    ax.set_title(f"K-Distance Plot -- Global DBSCAN (Kneedle, k={k})",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[KDIST] eps Kneedle = {eps_auto:.4f} (knee idx={knee_idx}) | "
          f"Saved: {save_path}")
    plt.close()

    return eps_auto


# =============================================================================
# 3. HITUNG ANOMALY SCORE
# =============================================================================
def compute_anomaly_score(X: np.ndarray, labels: np.ndarray,
                           core_indices: np.ndarray) -> np.ndarray:
    """
    Anomaly score = jarak minimum ke core point terdekat, dinormalisasi [0, 1].
    Semakin besar score -> semakin anomali.
    """
    if len(core_indices) == 0:
        # Edge case: tidak ada core point -- semua dianggap anomali
        return np.ones(len(X))

    core_points = X[core_indices]
    # Jarak setiap titik ke core point terdekat
    nbrs = NearestNeighbors(n_neighbors=1).fit(core_points)
    dists, _ = nbrs.kneighbors(X)
    raw_scores = dists.ravel()

    # Normalisasi ke [0, 1] dengan MinMaxScaler
    scores = MinMaxScaler().fit_transform(raw_scores.reshape(-1, 1)).ravel()
    return scores


# =============================================================================
# 4. JALANKAN DBSCAN
# =============================================================================
def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> tuple:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X)
    core_idx = db.core_sample_indices_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"\n[DBSCAN GLOBAL]")
    print(f"  eps         : {eps:.4f}")
    print(f"  min_samples : {min_samples}")
    print(f"  Jumlah klaster : {n_clusters}")
    print(f"  Jumlah noise   : {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    print(f"  Core points    : {len(core_idx)}")

    return labels, core_idx


# =============================================================================
# 5. VISUALISASI
# =============================================================================
def plot_anomaly_scatter(df: pd.DataFrame, labels: np.ndarray,
                          anomaly_scores: np.ndarray, save_path: str):
    """Scatter PC1 vs PC2 dengan warna anomali score."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Kiri: color by cluster
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap("tab20", max(len(unique_labels), 1))
    for i, lbl in enumerate(unique_labels):
        mask  = labels == lbl
        color = "red" if lbl == -1 else cmap(i)
        label = f"Noise (Anomali)" if lbl == -1 else f"Klaster {lbl}"
        axes[0].scatter(df.loc[mask, "PC1"], df.loc[mask, "PC2"],
                        c=[color], label=label, s=25, alpha=0.7)
    axes[0].set_title("Klaster DBSCAN Global (PC1 vs PC2)", fontweight="bold")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    # Kanan: color by anomaly score
    sc = axes[1].scatter(df["PC1"], df["PC2"], c=anomaly_scores,
                         cmap="RdYlGn_r", s=25, alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], label="Anomaly Score [0=normal, 1=anomali]")
    axes[1].set_title("Anomaly Score Global (PC1 vs PC2)", fontweight="bold")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


def plot_anomaly_timeline(df_result: pd.DataFrame, save_path: str):
    """Jumlah anomali per tahun (global)."""
    timeline = (df_result.groupby("year")["predicted_anomaly"]
                          .sum().reset_index())
    timeline.columns = ["year", "n_anomali"]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(timeline["year"], timeline["n_anomali"],
                  color=["tomato" if v > 0 else "steelblue"
                         for v in timeline["n_anomali"]],
                  edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Negara Anomali")
    ax.set_title("Jumlah Anomali Global per Tahun (DBSCAN)", fontweight="bold")
    ax.set_xticks(timeline["year"])
    ax.set_xticklabels(timeline["year"], rotation=45, ha="right")

    # Anotasi krisis global
    for crisis_yr, label in [(1997,"AFC"),(2008,"GFC"),(2020,"COVID")]:
        if crisis_yr in timeline["year"].values:
            ax.axvline(crisis_yr, color="navy", linestyle="--", alpha=0.6)
            ax.text(crisis_yr+0.1, timeline["n_anomali"].max()*0.95,
                    label, color="navy", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


def plot_anomaly_heatmap(df_result: pd.DataFrame, save_path: str):
    """Heatmap economy x year dengan nilai anomali."""
    pivot = df_result.pivot_table(
        index="economy", columns="year",
        values="predicted_anomaly", aggfunc="max"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(20, 12))
    import seaborn as sns
    sns.heatmap(pivot, cmap="Reds", linewidths=0.2, linecolor="gray",
                ax=ax, cbar_kws={"label": "Anomali (1=Ya, 0=Tidak)"},
                vmin=0, vmax=1)
    ax.set_title("Peta Anomali Global per Negara per Tahun (DBSCAN)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Tahun"); ax.set_ylabel("Negara")
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
    print("  DBSCAN GLOBAL -- ANOMALY DETECTION")
    print("=" * 65)

    # 1. Load
    df, pc_cols = load_data(INPUT_PATH)
    X_raw = df[pc_cols].values

    # 2. Re-standarisasi komponen PCA (opsional via SCALE_PCA)
    #    PCA menghasilkan komponen dengan varians berbeda (PC1 >> PC9).
    #    Re-standarisasi membuat setiap PC berkontribusi setara ke jarak Euclidean.
    if SCALE_PCA:
        scaler_pca = StandardScaler()
        X = scaler_pca.fit_transform(X_raw)
        print("[INFO] PCA components di-standarisasi ulang (StandardScaler) "
              "agar setiap PC berkontribusi setara ke jarak Euclidean.")
    else:
        X = X_raw

    # 3. Tentukan eps via Kneedle
    eps_path = os.path.join(OUTPUT_DIR, "kdist_global.png")
    if EPS_AUTO:
        eps = find_eps(X, k=K_NEIGHBORS, save_path=eps_path)
    else:
        eps = 2.5   # set manual jika EPS_AUTO = False

    # 4. Jalankan DBSCAN
    labels, core_idx = run_dbscan(X, eps=eps, min_samples=MIN_SAMPLES)

    # 5. Anomaly score
    scores = compute_anomaly_score(X, labels, core_idx)

    # 6. Buat prediksi biner: -1 (noise) -> 1 (anomali), lainnya -> 0
    predicted = np.where(labels == -1, 1, 0)

    # 6. Buat DataFrame hasil
    df_result = df[["economy", "year"]].copy().reset_index(drop=True)
    df_result["anomaly_score"]      = np.round(scores, 6)
    df_result["predicted_anomaly"]  = predicted

    # 7. Simpan CSV
    df_result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVE] Hasil -> {OUTPUT_CSV}")
    print(f"  Total anomali  : {predicted.sum()} "
          f"({predicted.sum()/len(predicted)*100:.2f}%)")
    print(f"  Total normal   : {(predicted == 0).sum()}")

    # 8. Visualisasi
    # Buat df_plot dengan nilai PC1/PC2 dari X yang sudah di-scale
    df_plot = df[["economy", "year"]].copy().reset_index(drop=True)
    df_plot[pc_cols] = X
    plot_anomaly_scatter(df_plot, labels, scores,
                         os.path.join(OUTPUT_DIR, "anomaly_global_pc1pc2.png"))
    plot_anomaly_timeline(df_result,
                          os.path.join(OUTPUT_DIR, "anomaly_timeline_global.png"))
    plot_anomaly_heatmap(df_result,
                         os.path.join(OUTPUT_DIR, "anomaly_heatmap_global.png"))

    # 9. Top anomali
    top = (df_result[df_result["predicted_anomaly"] == 1]
           .sort_values("anomaly_score", ascending=False)
           .head(20))
    print("\n[INFO] Top 20 anomali (global):")
    print(top.to_string(index=False))

    print("\n" + "=" * 65)
    print("  DBSCAN GLOBAL SELESAI")
    print("=" * 65)

    return df_result


if __name__ == "__main__":
    df_result = main()
