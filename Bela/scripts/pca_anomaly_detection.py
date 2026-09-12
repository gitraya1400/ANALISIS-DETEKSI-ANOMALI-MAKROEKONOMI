# -*- coding: utf-8 -*-
"""
PCA-based Anomaly Detection — Early Warning System Krisis Ekonomi
Versi script (non-interaktif) dari notebooks/PCA_Anomaly_Detection_EWS.ipynb

Menjalankan pipeline lengkap dan MENYIMPAN semua hasil ke folder:
- ../results/  (CSV hasil deteksi & ringkasan metrik)
- ../figures/  (PNG semua grafik)

Jalankan dari dalam folder scripts/:
    python pca_anomaly_detection.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interaktif, cuma simpan file, tidak buka window
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
)
from scipy.stats import chi2, f as f_dist

plt.rcParams["figure.figsize"] = (8, 4)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data_with_labels.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_spe(X, pca_model):
    """Hitung SPE (reconstruction error) untuk setiap baris di X."""
    X_proj = pca_model.transform(X)
    X_reconstructed = pca_model.inverse_transform(X_proj)
    return np.sum((X - X_reconstructed) ** 2, axis=1)


def spe_ucl(spe_train, alpha=0.05):
    """UCL SPE via pendekatan chi-square (Jackson & Mudholkar, 1979)."""
    mean_spe = np.mean(spe_train)
    var_spe = np.var(spe_train)
    g = var_spe / (2 * mean_spe)
    h = (2 * mean_spe ** 2) / var_spe
    return g * chi2.ppf(1 - alpha, h)


def compute_t2(X, pca_model):
    """Hotelling's T^2 untuk setiap baris di X."""
    X_proj = pca_model.transform(X)
    eigenvalues = pca_model.explained_variance_
    return np.sum((X_proj ** 2) / eigenvalues, axis=1)


def t2_ucl(n_train, k, alpha=0.05):
    """UCL T^2 via distribusi F."""
    f_crit = f_dist.ppf(1 - alpha, k, n_train - k)
    return (k * (n_train - 1) / (n_train - k)) * f_crit


def detect_anomaly(X, pca_model, ucl_spe, ucl_t2):
    spe = compute_spe(X, pca_model)
    t2 = compute_t2(X, pca_model)
    is_anomaly = (spe > ucl_spe) | (t2 > ucl_t2)
    return pd.DataFrame({"SPE": spe, "T2": t2, "is_anomaly": is_anomaly.astype(int)})


def spe_contribution(x_row, pca_model, feature_names):
    x_row = x_row.reshape(1, -1)
    x_proj = pca_model.transform(x_row)
    x_reconstructed = pca_model.inverse_transform(x_proj)
    residual_sq = (x_row - x_reconstructed) ** 2
    return pd.Series(residual_sq.flatten(), index=feature_names).sort_values(ascending=False)


def effect_size(df_scaled, feature, group_col="is_anomaly"):
    g1 = df_scaled.loc[df_scaled[group_col] == 1, feature]
    g0 = df_scaled.loc[df_scaled[group_col] == 0, feature]
    pooled_std = np.sqrt((g1.var() + g0.var()) / 2)
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0
    return (g1.mean() - g0.mean()) / pooled_std


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    # ---------- 1. Load data ----------
    df = pd.read_csv(DATA_PATH)
    log(f"Shape data: {df.shape}")
    log(f"Distribusi crisis_label:\n{df['crisis_label'].value_counts(normalize=True)}\n")

    # ---------- 2. Preprocessing ----------
    exclude_cols = ["economy", "year", "crisis_label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    log(f"Jumlah fitur dipakai: {len(feature_cols)} -> {feature_cols}\n")

    normal_df = df[df["crisis_label"] == 0].copy()
    anomaly_df = df[df["crisis_label"] == 1].copy()

    train_df, temp_df = train_test_split(normal_df, test_size=0.30, random_state=42)
    val_normal_df, test_normal_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    val_anomaly_df, test_anomaly_df = train_test_split(anomaly_df, test_size=0.50, random_state=42)

    val_df = pd.concat([val_normal_df, val_anomaly_df]).sample(frac=1, random_state=42)
    test_df = pd.concat([test_normal_df, test_anomaly_df]).sample(frac=1, random_state=42)

    log(f"Train (normal only): {train_df.shape}")
    log(f"Validation (normal+anomali): {val_df.shape}")
    log(f"Test (normal+anomali): {test_df.shape}\n")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_val = val_df["crisis_label"].values
    y_test = test_df["crisis_label"].values

    # ---------- 3. Fit PCA & pilih k ----------
    pca_full = PCA(n_components=len(feature_cols), random_state=42)
    pca_full.fit(X_train)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.argmax(cum_var >= 0.95) + 1)
    log(f"Jumlah komponen terpilih (k) untuk >=95% explained variance: {k} dari {len(feature_cols)}\n")

    plt.figure()
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
    plt.axhline(0.95, color="red", linestyle="--", label="95% threshold")
    plt.axvline(k, color="green", linestyle="--", label=f"k = {k}")
    plt.xlabel("Jumlah Komponen Utama")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot — Pemilihan Jumlah Komponen PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_scree_plot.png"), dpi=150)
    plt.close()

    pca = PCA(n_components=k, random_state=42)
    pca.fit(X_train)
    log(f"PCA final dilatih dengan k={k} komponen")
    log(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}\n")

    # ---------- 4. SPE & UCL ----------
    spe_train = compute_spe(X_train, pca)
    ucl_spe = spe_ucl(spe_train, alpha=0.05)
    log(f"UCL SPE (95%): {ucl_spe:.4f}")

    # ---------- 5. T2 & UCL ----------
    n_train = X_train.shape[0]
    ucl_t2 = t2_ucl(n_train, k, alpha=0.05)
    log(f"UCL T^2 (95%): {ucl_t2:.4f}\n")

    # ---------- 6. Klasifikasi anomali ----------
    val_result = detect_anomaly(X_val, pca, ucl_spe, ucl_t2)
    test_result = detect_anomaly(X_test, pca, ucl_spe, ucl_t2)
    val_result["crisis_label"] = y_val
    test_result["crisis_label"] = y_test

    plt.figure(figsize=(10, 4))
    colors = np.where(test_result["crisis_label"] == 1, "red", "steelblue")
    plt.scatter(range(len(test_result)), test_result["SPE"], c=colors, s=15, alpha=0.7)
    plt.axhline(ucl_spe, color="black", linestyle="--", label="UCL")
    plt.xlabel("Index Observasi (Test Set)")
    plt.ylabel("SPE")
    plt.title("Control Chart — SPE pada Data Test\n(merah = crisis_label aktual, garis = UCL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_control_chart_spe.png"), dpi=150)
    plt.close()

    # ---------- 7. Evaluasi ----------
    def anomaly_score(result_df):
        return (result_df["SPE"] / ucl_spe) + (result_df["T2"] / ucl_t2)

    test_result["anomaly_score"] = anomaly_score(test_result)
    y_true = test_result["crisis_label"]
    y_score = test_result["anomaly_score"]
    y_pred = test_result["is_anomaly"]

    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_score),
        "F1-Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Average Precision": average_precision_score(y_true, y_score),
    }
    log("=== METRIK EVALUASI (Test Set) ===")
    for name, val in metrics.items():
        log(f"{name:20s}: {val:.4f}")
    log()

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log("=== CONFUSION MATRIX (Test Set) ===")
    log(f"True Positive  (anomali & memang krisis)      : {tp}")
    log(f"False Positive (anomali tapi TIDAK krisis)     : {fp}")
    log(f"False Negative (TIDAK anomali tapi krisis)     : {fn}")
    log(f"True Negative  (tidak anomali & memang normal) : {tn}\n")

    # ---------- 8. Contribution plot (top anomaly) ----------
    idx_top_anomaly = test_result["anomaly_score"].idxmax()
    row_position = test_result.index.get_loc(idx_top_anomaly)
    x_example = X_test[row_position]
    contrib = spe_contribution(x_example, pca, feature_cols)

    test_df_reset = test_df.reset_index(drop=True)
    top_row_info = test_df_reset.iloc[row_position]
    log(f"Observasi anomaly_score tertinggi: {top_row_info['economy']} tahun {int(top_row_info['year'])}")
    log(f"Top 5 variabel penyumbang:\n{contrib.head(5)}\n")

    plt.figure()
    contrib.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Kontribusi terhadap SPE (residual^2)")
    plt.title(f"Contribution Plot — {top_row_info['economy']} {int(top_row_info['year'])}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_contribution_plot_top_anomaly.png"), dpi=150)
    plt.close()

    # ---------- 9. Interpretasi: effect size ----------
    combined = pd.concat(
        [test_df_reset[["economy", "year"] + feature_cols].reset_index(drop=True),
         test_result.reset_index(drop=True)],
        axis=1
    )
    X_test_scaled_df = pd.DataFrame(X_test, columns=feature_cols)
    combined_scaled = pd.concat([X_test_scaled_df, test_result.reset_index(drop=True)], axis=1)

    effect_sizes = pd.Series(
        {f: effect_size(combined_scaled, f) for f in feature_cols}
    ).sort_values(key=abs, ascending=False)
    log("=== EFFECT SIZE (Anomali vs Normal, Test Set) ===")
    log(f"{effect_sizes}\n")

    plt.figure(figsize=(8, 6))
    effect_sizes.plot(kind="barh")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Effect size (standardized mean difference)")
    plt.title("Variabel Paling Membedakan Anomali vs Normal (Test Set)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_effect_size.png"), dpi=150)
    plt.close()

    # ---------- Simpan hasil ke CSV ----------
    test_result_export = pd.concat(
        [test_df_reset[["economy", "year"]].reset_index(drop=True), test_result.reset_index(drop=True)],
        axis=1
    )
    test_result_export.to_csv(os.path.join(RESULTS_DIR, "test_result.csv"), index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df["k_components"] = k
    metrics_df["ucl_spe"] = ucl_spe
    metrics_df["ucl_t2"] = ucl_t2
    metrics_df["true_positive"] = tp
    metrics_df["false_positive"] = fp
    metrics_df["false_negative"] = fn
    metrics_df["true_negative"] = tn
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)

    effect_sizes.rename("effect_size").to_csv(os.path.join(RESULTS_DIR, "effect_sizes.csv"))

    with open(os.path.join(RESULTS_DIR, "run_log.txt"), "w") as f:
        f.write("\n".join(log_lines))

    log(f"Selesai. Hasil tersimpan di: {RESULTS_DIR}")
    log(f"Grafik tersimpan di: {FIGURES_DIR}")

    return {
        "metrics": metrics,
        "k": k,
        "ucl_spe": ucl_spe,
        "ucl_t2": ucl_t2,
        "confusion": (tp, fp, fn, tn),
        "effect_sizes": effect_sizes,
        "top_anomaly": (top_row_info["economy"], int(top_row_info["year"])),
        "contrib_top5": contrib.head(5),
    }


if __name__ == "__main__":
    main()
