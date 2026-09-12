"""
01_pca_pipeline.py
===================
Implementasi PCA-based Anomaly Detection untuk EWS Krisis Ekonomi
(bagian tugas: Kelompok 4).

PERBAIKAN METODOLOGI (mengikuti arahan agar adil dibandingkan dengan LOF):
---------------------------------------------------------------------------
Sebelumnya PCA dilatih hanya dari 70% data normal (mengikuti pola train/
val/test seperti Autoencoder). Ini membuat hasil PCA tidak bisa dibandingkan
apple-to-apple dengan algoritma lain (LOF, Isolation Forest, dll) yang
di-scoring pada seluruh dataset.

Perbaikan yang diterapkan di script ini:
1. StandardScaler DAN PCA di-fit HANYA dari seluruh baris normal
   (crisis_label == 0), yaitu 1.385 baris -- BUKAN cuma 70%-nya.
   -> Prinsip "PCA hanya belajar dari kondisi normal" tetap terjaga,
      tidak ada informasi krisis yang bocor ke tahap pelatihan model.
2. SPE (Squared Prediction Error / Q-statistic) dan T^2 (Hotelling's
   T-squared) dihitung untuk SELURUH 1.714 baris (normal + krisis),
   persis seperti cara LOF di-scoring pada seluruh dataset.
3. UCL (Upper Control Limit) tetap dihitung HANYA dari sebaran SPE/T^2
   pada data normal (bukan dari seluruh data), sehingga tidak ada
   label krisis yang "bocor" ke penentuan ambang batas anomali.

Output:
- results/pca_scores.csv          -> skor SPE, T^2, dan label anomali per baris
- results/pca_model_summary.csv   -> explained variance, jumlah komponen, UCL
- results/contribution_matrix.csv -> kontribusi tiap variabel terhadap SPE per baris
"""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import (
    RAW_CSV, RESULTS_DIR, FEATURE_COLS, LABEL_COL, ID_COLS,
    EXPLAINED_VARIANCE_THRESHOLD, ALPHA, RANDOM_STATE,
)
import os


def load_data():
    df = pd.read_csv(RAW_CSV)
    return df


def fit_scaler_and_pca(df):
    """Fit StandardScaler & PCA hanya dari SELURUH baris normal."""
    normal_mask = df[LABEL_COL] == 0
    X_normal_raw = df.loc[normal_mask, FEATURE_COLS].values

    n_normal = X_normal_raw.shape[0]
    print(f"[INFO] Jumlah baris normal untuk fit scaler & PCA: {n_normal}")

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal_raw)

    # Fit PCA penuh dulu (semua komponen) agar eigenvalue komponen yang
    # dibuang (discarded) tersedia untuk formula UCL SPE (Jackson-Mudholkar).
    pca_full = PCA(n_components=None, random_state=RANDOM_STATE)
    pca_full.fit(X_normal_scaled)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, EXPLAINED_VARIANCE_THRESHOLD) + 1)
    k = min(k, len(cum_var))

    print(f"[INFO] Jumlah komponen utama (k) untuk >= {EXPLAINED_VARIANCE_THRESHOLD*100:.0f}% "
          f"cumulative explained variance: {k} dari {len(cum_var)} fitur")
    print(f"[INFO] Cumulative explained variance pada k={k}: {cum_var[k-1]*100:.2f}%")

    return scaler, pca_full, k, n_normal


def compute_spe_t2(X_scaled, pca_full, k):
    """Hitung SPE dan T^2 untuk setiap baris menggunakan k komponen utama."""
    components_k = pca_full.components_[:k]          # (k, p)
    eigenvalues_k = pca_full.explained_variance_[:k]  # (k,)

    scores = X_scaled @ components_k.T                # (n, k) -- skor PC
    X_reconstructed = scores @ components_k            # (n, p)
    residual = X_scaled - X_reconstructed               # (n, p)

    spe = np.sum(residual ** 2, axis=1)                 # Q-statistic
    t2 = np.sum((scores ** 2) / eigenvalues_k, axis=1)  # Hotelling's T^2

    return spe, t2, residual, scores


def spe_ucl_jackson_mudholkar(eigenvalues_all, k, alpha=ALPHA):
    """UCL untuk SPE berdasarkan distribusi chi-square approx.
    (Jackson & Mudholkar, 1979) menggunakan eigenvalue komponen yang DIBUANG.
    """
    lambdas_discarded = eigenvalues_all[k:]
    # Jika semua komponen dipertahankan (k == p), tidak ada residual -> SPE = 0
    if len(lambdas_discarded) == 0:
        return 0.0

    theta1 = np.sum(lambdas_discarded)
    theta2 = np.sum(lambdas_discarded ** 2)
    theta3 = np.sum(lambdas_discarded ** 3)

    if theta1 == 0:
        return 0.0

    h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2) if theta2 != 0 else 1.0
    c_alpha = norm.ppf(1 - alpha)

    term = (c_alpha * np.sqrt(2 * theta2 * h0 ** 2) / theta1) + 1 + \
           (theta2 * h0 * (h0 - 1) / theta1 ** 2)
    ucl_spe = theta1 * (term ** (1.0 / h0))
    return float(ucl_spe)


def t2_ucl_f_distribution(k, n, alpha=ALPHA):
    """UCL untuk T^2 berdasarkan distribusi F, dengan n = jumlah data NORMAL
    yang digunakan untuk melatih PCA (bukan seluruh dataset)."""
    f_crit = f_dist.ppf(1 - alpha, k, n - k)
    ucl_t2 = (k * (n - 1) * (n + 1)) / (n * (n - k)) * f_crit
    return float(ucl_t2)


def main():
    df = load_data()
    scaler, pca_full, k, n_normal = fit_scaler_and_pca(df)

    # Transform SELURUH dataset (normal + krisis) dengan scaler yang sama
    X_all_scaled = scaler.transform(df[FEATURE_COLS].values)

    spe_all, t2_all, residual_all, scores_all = compute_spe_t2(X_all_scaled, pca_full, k)

    # --- UCL dihitung HANYA dari sebaran normal (tidak ada leakage label) ---
    ucl_spe = spe_ucl_jackson_mudholkar(pca_full.explained_variance_, k, ALPHA)
    ucl_t2 = t2_ucl_f_distribution(k, n_normal, ALPHA)

    print(f"[INFO] UCL SPE (alpha={ALPHA}) : {ucl_spe:.4f}")
    print(f"[INFO] UCL T^2 (alpha={ALPHA}) : {ucl_t2:.4f}")

    is_anom_spe = (spe_all > ucl_spe).astype(int)
    is_anom_t2 = (t2_all > ucl_t2).astype(int)
    is_anom_combined = ((is_anom_spe == 1) | (is_anom_t2 == 1)).astype(int)

    # --- Simpan skor per observasi ---
    out = df[ID_COLS + [LABEL_COL]].copy()
    out["SPE"] = spe_all
    out["T2"] = t2_all
    out["UCL_SPE"] = ucl_spe
    out["UCL_T2"] = ucl_t2
    out["is_anomaly_SPE"] = is_anom_spe
    out["is_anomaly_T2"] = is_anom_t2
    out["is_anomaly_combined"] = is_anom_combined
    # skor gabungan ternormalisasi (rasio terhadap UCL masing2, diambil max)
    out["combined_score"] = np.maximum(spe_all / ucl_spe, t2_all / ucl_t2)

    scores_path = os.path.join(RESULTS_DIR, "pca_scores.csv")
    out.to_csv(scores_path, index=False)
    print(f"[OK] Skor SPE/T2 disimpan ke: {scores_path}")

    # --- Simpan ringkasan model PCA ---
    p = len(FEATURE_COLS)
    summary = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(p)],
        "eigenvalue": pca_full.explained_variance_,
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
        "retained": [i < k for i in range(p)],
    })
    summary_path = os.path.join(RESULTS_DIR, "pca_model_summary.csv")
    summary.to_csv(summary_path, index=False)

    meta = pd.DataFrame({
        "metric": ["n_normal_train", "n_total", "n_features", "k_components",
                   "cumulative_var_at_k", "UCL_SPE", "UCL_T2", "alpha"],
        "value": [n_normal, len(df), p, k,
                  np.cumsum(pca_full.explained_variance_ratio_)[k-1],
                  ucl_spe, ucl_t2, ALPHA],
    })
    meta_path = os.path.join(RESULTS_DIR, "pca_run_metadata.csv")
    meta.to_csv(meta_path, index=False)
    print(f"[OK] Ringkasan model PCA disimpan ke: {summary_path}")
    print(f"[OK] Metadata run disimpan ke: {meta_path}")

    # --- Simpan matriks kontribusi (residual^2 per fitur, untuk contribution plot) ---
    contrib = pd.DataFrame(residual_all ** 2, columns=FEATURE_COLS)
    contrib.insert(0, "economy", df["economy"].values)
    contrib.insert(1, "year", df["year"].values)
    contrib.insert(2, LABEL_COL, df[LABEL_COL].values)
    contrib.insert(3, "is_anomaly_combined", is_anom_combined)
    contrib_path = os.path.join(RESULTS_DIR, "contribution_matrix.csv")
    contrib.to_csv(contrib_path, index=False)
    print(f"[OK] Matriks kontribusi (residual^2) disimpan ke: {contrib_path}")

    return out, summary, meta, contrib


if __name__ == "__main__":
    main()
