# =============================================================================
# evaluate.py
# Evaluasi hasil DBSCAN vs Ground Truth (ground_truth_imf.csv)
# Metrik Supervised  : Accuracy, Precision, Recall, F1, ROC-AUC
# Metrik Unsupervised: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz
# Metrik Tambahan    : PR Curve, Average Precision, per-jenis-krisis
# Output : output/evaluasi_global.csv
#          output/evaluasi_per_country.csv
#          output/eval_confusion_*.png
#          output/eval_roc_*.png
#          output/eval_pr_curve.png
#          output/eval_cluster_quality.png
#          output/eval_per_crisis_type.png
#          output/eval_ringkasan.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
import os

warnings.filterwarnings("ignore")

# -- Konfigurasi --------------------------------------------------------------
GT_PATH           = "ground_truth_imf.csv"
HASIL_GLOBAL      = os.path.join("output", "hasil_global.csv")
HASIL_PER_COUNTRY = os.path.join("output", "hasil_per_country.csv")
OUTPUT_DIR        = "output"


# =============================================================================
# 1. LOAD & MERGE
# =============================================================================
def load_and_merge(hasil_path: str, gt_path: str,
                   label: str) -> pd.DataFrame:
    """
    Gabungkan hasil DBSCAN dengan ground truth on (economy, year).
    Kolom kunci: anomaly_score, predicted_anomaly, is_crisis
    """
    hasil = pd.read_csv(hasil_path)
    gt    = pd.read_csv(gt_path)[["economy", "year",
                                   "banking_crisis", "currency_crisis",
                                   "sovereign_debt_crisis", "crisis_name",
                                   "is_crisis"]]
    merged = hasil.merge(gt, on=["economy", "year"], how="inner")
    print(f"[{label}] Merged: {len(merged)} baris "
          f"(DBSCAN: {len(hasil)}, GT: {len(gt)})")
    print(f"  GT is_crisis=1 : {merged['is_crisis'].sum()}")
    print(f"  Pred anomali=1 : {merged['predicted_anomaly'].sum()}")
    return merged


# =============================================================================
# 2. HITUNG METRIK EVALUASI
# =============================================================================
def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    """
    Hitung metrik klasifikasi biner:
    - y_true : is_crisis (ground truth IMF)
    - y_pred : predicted_anomaly (DBSCAN)
    - y_score: anomaly_score (untuk ROC-AUC)
    """
    y_true  = df["is_crisis"].values
    y_pred  = df["predicted_anomaly"].values
    y_score = df["anomaly_score"].values

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_score)
    except Exception:
        auc_score = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"\n[EVAL {label}]")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}  (dari sekian prediksi anomali, berapa % benar)")
    print(f"  Recall     : {rec:.4f}  (dari sekian krisis nyata, berapa % terdeteksi)")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc_score:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return dict(
        metode=label, accuracy=acc, precision=prec,
        recall=rec, f1=f1, roc_auc=auc_score,
        TP=tp, FP=fp, TN=tn, FN=fn,
        n_total=len(df), n_crisis_gt=int(y_true.sum()),
        n_anomali_pred=int(y_pred.sum())
    )


# =============================================================================
# 3. METRIK PER NEGARA (untuk per_country)
# =============================================================================
def compute_metrics_per_economy(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung metrik evaluasi per negara."""
    rows = []
    for eco, sub in df.groupby("economy"):
        y_true  = sub["is_crisis"].values
        y_pred  = sub["predicted_anomaly"].values
        y_score = sub["anomaly_score"].values

        if y_true.sum() == 0 and y_pred.sum() == 0:
            f1, prec, rec, auc_val = 1.0, 1.0, 1.0, 1.0   # keduanya prediksi benar (no crisis)
        elif y_true.sum() == 0:
            f1, prec, rec, auc_val = 0.0, 0.0, 0.0, float("nan")
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc_val = roc_auc_score(y_true, y_score)
            except Exception:
                auc_val = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append(dict(
            economy=eco,
            n_crisis_gt=int(y_true.sum()),
            n_anomali_pred=int(y_pred.sum()),
            TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            roc_auc=round(auc_val, 4) if not np.isnan(auc_val) else float("nan")
        ))

    return pd.DataFrame(rows).sort_values("f1", ascending=False)


# =============================================================================
# 4. VISUALISASI CONFUSION MATRIX
# =============================================================================
def plot_confusion(cm: np.ndarray, label: str, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Normal (0)", "Pred Anomali (1)"],
                yticklabels=["GT Normal (0)", "GT Krisis (1)"],
                linewidths=0.5, linecolor="gray",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {label}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Ground Truth (IMF)")
    ax.set_xlabel("Prediksi DBSCAN")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


# =============================================================================
# 5. VISUALISASI ROC CURVE
# =============================================================================
def plot_roc(df: pd.DataFrame, label: str, save_path: str):
    y_true  = df["is_crisis"].values
    y_score = df["anomaly_score"].values

    if y_true.sum() == 0:
        print(f"[ROC] Tidak ada krisis di {label}, skip.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="tomato", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="tomato")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(f"ROC Curve — {label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


# =============================================================================
# 6. VISUALISASI SUMMARY F1 PER NEGARA (per-country)
# =============================================================================
def plot_f1_per_economy(df_eco: pd.DataFrame, save_path: str):
    df_sorted = df_eco.sort_values("f1", ascending=True)
    colors = ["tomato" if f1 < 0.3 else
              "orange" if f1 < 0.6 else
              "steelblue" for f1 in df_sorted["f1"]]

    fig, ax = plt.subplots(figsize=(10, 14))
    bars = ax.barh(df_sorted["economy"], df_sorted["f1"],
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="F1 = 0.5")
    ax.set_xlabel("F1-Score")
    ax.set_title("F1-Score per Negara — DBSCAN Per-Country vs IMF Ground Truth",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend()

    # Tambahkan label nilai
    for bar, val in zip(bars, df_sorted["f1"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


# =============================================================================
# 7. VISUALISASI RINGKASAN METRIK (perbandingan global vs per-country)
# =============================================================================
def plot_summary_comparison(metrics_list: list, save_path: str):
    labels   = [m["metode"] for m in metrics_list]
    metrics  = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    m_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m_dict, lbl) in enumerate(zip(metrics_list, labels)):
        vals = [m_dict[m] for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=lbl,
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Metrik Evaluasi: DBSCAN Global vs Per-Negara",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[VIZ] Saved: {save_path}")
    plt.close()



# =============================================================================
# D. METRIK KUALITAS KLASTER (UNSUPERVISED)
# =============================================================================
def eval_cluster_quality(data_path: str, pred_labels: np.ndarray,
                         label: str) -> dict:
    """
    Hitung metrik kualitas klaster tanpa ground truth:
    - Silhouette Score  : [-1, 1], makin tinggi makin baik
    - Davies-Bouldin    : [0, inf], makin rendah makin baik
    - Calinski-Harabasz : [0, inf], makin tinggi makin baik

    Catatan: label DBSCAN -1 (noise/anomali) diplot sebagai klaster tersendiri.
    Metrik dihitung pada semua titik yang punya label >= 0 (hanya klaster, bukan noise)
    karena silhouette tidak terdefinisi untuk titik noise.
    """
    df = pd.read_csv(data_path)
    feat_cols = [c for c in df.columns if c not in ["economy", "year"]]
    X = df[feat_cols].values

    # Gunakan pred_labels sebagai proxy klaster: 0=normal, 1=anomali
    # (bisa dianggap 2 klaster untuk keperluan metrik ini)
    if len(np.unique(pred_labels)) < 2:
        print(f"  [{label}] Hanya 1 klaster -- metrik tidak bisa dihitung.")
        return dict(metode=label, silhouette=np.nan,
                    davies_bouldin=np.nan, calinski_harabasz=np.nan)

    try:
        sil  = silhouette_score(X, pred_labels, sample_size=min(1000, len(X)),
                                random_state=42)
        db   = davies_bouldin_score(X, pred_labels)
        ch   = calinski_harabasz_score(X, pred_labels)
    except Exception as e:
        print(f"  [{label}] Error: {e}")
        sil, db, ch = np.nan, np.nan, np.nan

    print(f"\n[CLUSTER QUALITY - {label}]")
    print(f"  Silhouette Score   : {sil:.4f}  (ideal: mendekati +1)")
    print(f"  Davies-Bouldin     : {db:.4f}  (ideal: mendekati 0)")
    print(f"  Calinski-Harabasz  : {ch:.2f}   (ideal: semakin besar)")
    print()
    print(f"  Interpretasi Silhouette ({sil:.4f}):")
    if sil > 0.5:
        print(f"    BAIK -- klaster terpisah dengan jelas")
    elif sil > 0.25:
        print(f"    SEDANG -- ada tumpang tindih antar klaster")
    else:
        print(f"    LEMAH -- klaster tidak terpisah dengan jelas (wajar untuk outlier detection)")

    return dict(metode=label, silhouette=round(sil, 4),
                davies_bouldin=round(db, 4),
                calinski_harabasz=round(ch, 2))


def plot_cluster_quality(results: list, save_path: str):
    """Visualisasi metrik kualitas klaster (bar chart perbandingan)."""
    labels = [r["metode"] for r in results]
    metrics = {
        "Silhouette Score\n(lebih tinggi lebih baik, max=1)": "silhouette",
        "Davies-Bouldin\n(lebih rendah lebih baik)":          "davies_bouldin",
        "Calinski-Harabasz\n(lebih tinggi lebih baik)":       "calinski_harabasz",
    }
    colors = ["#f4a261", "#e63946"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Metrik Kualitas Klaster DBSCAN (Unsupervised — Tanpa Ground Truth)",
                 fontsize=13, fontweight="bold")

    for ax, (title, key) in zip(axes, metrics.items()):
        vals = [r[key] for r in results]
        bars = ax.bar(labels, vals, color=colors[:len(labels)],
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height())*0.02,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        if key == "silhouette":
            ax.set_ylim(min(0, min(vals)-0.05), 1.0)
            ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
            ax.axhline(0.5, color="green", lw=1, ls="--", alpha=0.4,
                       label="0.5 (threshold baik)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ] Saved: {save_path}")
    plt.close()

    # Simpan ke CSV
    csv_path = save_path.replace(".png", ".csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")


# =============================================================================
# E. PRECISION-RECALL CURVE + AVERAGE PRECISION
# =============================================================================
def plot_pr_curve(df_global: pd.DataFrame, df_pc: pd.DataFrame,
                  save_path: str):
    """
    Precision-Recall Curve lebih informatif dari ROC untuk data tidak seimbang.
    Baseline PR = proporsi krisis di data (random classifier).
    Average Precision (AP) = area di bawah PR curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Precision-Recall Curve vs ROC Curve\n"
                 "(PR Curve lebih informatif untuk data tidak seimbang)",
                 fontsize=12, fontweight="bold")

    colors = {"GLOBAL": "#f4a261", "PER-COUNTRY": "#e63946"}
    datasets = {"GLOBAL": df_global, "PER-COUNTRY": df_pc}

    # Panel kiri: PR Curve
    ax_pr = axes[0]
    for name, df in datasets.items():
        y_true  = df["is_crisis"].values
        y_score = df["anomaly_score"].values
        baseline = y_true.mean()

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax_pr.plot(rec, prec, color=colors[name], lw=2,
                   label=f"{name} (AP={ap:.3f})")
        ax_pr.fill_between(rec, prec, alpha=0.08, color=colors[name])
        print(f"  [{name}] Average Precision (AP): {ap:.4f}")
        print(f"  [{name}] Baseline (random):       {baseline:.4f}")

    ax_pr.axhline(baseline, color="gray", ls=":", lw=1.5,
                  label=f"Baseline acak (AP={baseline:.3f})")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.05])
    ax_pr.legend(fontsize=9)
    ax_pr.grid(alpha=0.3, ls=":")
    ax_pr.text(0.5, 0.15,
               f"Baseline = {baseline:.1%}\n(jika acak memilih {baseline:.1%} data sebagai krisis)",
               ha="center", fontsize=8, color="gray", transform=ax_pr.transAxes,
               style="italic")

    # Panel kanan: ROC comparison (sebagai perbandingan)
    ax_roc = axes[1]
    for name, df in datasets.items():
        y_true  = df["is_crisis"].values
        y_score = df["anomaly_score"].values
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[name], lw=2,
                    label=f"{name} (AUC={roc_auc:.3f})")
        ax_roc.fill_between(fpr, tpr, alpha=0.08, color=colors[name])
    ax_roc.plot([0,1],[0,1], color="gray", ls=":", lw=1.5, label="Random (AUC=0.5)")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.set_title("ROC Curve (untuk perbandingan)", fontsize=11, fontweight="bold")
    ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.05])
    ax_roc.legend(fontsize=9)
    ax_roc.grid(alpha=0.3, ls=":")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ] Saved: {save_path}")
    plt.close()


# =============================================================================
# F. EVALUASI PER JENIS KRISIS
# =============================================================================
def eval_per_crisis_type(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Evaluasi DBSCAN Per-Negara berdasarkan jenis krisis:
    - Banking Crisis
    - Currency Crisis
    - Sovereign Debt Crisis
    - COVID-19 (2020)
    Tujuan: Mana jenis krisis yang paling mudah/sulit dideteksi?
    """
    merged = df.copy()
    gt = pd.read_csv(GT_PATH)
    req_cols = ["banking_crisis", "currency_crisis", "sovereign_debt_crisis", "crisis_name", "is_crisis"]
    for c in req_cols:
        if c not in merged.columns:
            merged = merged.merge(gt[["economy", "year", c]], on=["economy", "year"], how="left")

    crisis_cols = {
        "Banking Crisis":       "banking_crisis",
        "Currency Crisis":      "currency_crisis",
        "Sovereign Debt":       "sovereign_debt_crisis",
    }
    # Tambah COVID-19 (2020)
    merged["covid_crisis"] = (
        (merged["year"] == 2020) & (merged["is_crisis"] == 1)
    ).astype(int)
    crisis_cols["COVID-19 (2020)"] = "covid_crisis"

    rows = []
    print(f"\n  {'Jenis Krisis':<22} {'N Krisis':>9} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>7} {'AP':>7}")
    print("  " + "-" * 75)

    for name, col in crisis_cols.items():
        y_true  = (merged[col] == 1).astype(int).values
        y_pred  = merged["predicted_anomaly"].astype(int).values
        y_score = merged["anomaly_score"].values

        n_k = int(y_true.sum())
        if n_k == 0:
            continue

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        try:
            ap = average_precision_score(y_true, y_score)
        except Exception:
            ap = float("nan")

        print(f"  {name:<22} {n_k:>9} {tp:>5} {fp:>5} {fn:>5} "
              f"{prec:>10.3f} {rec:>8.3f} {f1:>7.3f} {ap:>7.3f}")

        rows.append(dict(
            jenis_krisis=name,
            n_krisis=n_k,
            TP=tp, FP=fp, FN=fn,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            avg_precision=round(ap, 4) if not np.isnan(ap) else np.nan
        ))

    df_result = pd.DataFrame(rows)

    # Visualisasi
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Evaluasi DBSCAN Per-Negara: Berdasarkan Jenis Krisis",
                 fontsize=13, fontweight="bold")

    palette = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d"]
    jenis   = df_result["jenis_krisis"].tolist()

    for ax, (metric, ylabel, title) in zip(axes, [
        ("recall",        "Recall",           "Recall per Jenis Krisis\n(Berapa % krisis terdeteksi)"),
        ("precision",     "Precision",        "Precision per Jenis Krisis\n(Berapa % prediksi yang benar)"),
        ("avg_precision", "Avg. Precision",   "Average Precision per Jenis Krisis\n(Area bawah PR Curve)"),
    ]):
        vals  = df_result[metric].tolist()
        bars  = ax.bar(jenis, vals, color=palette[:len(jenis)],
                       edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", labelsize=8, rotation=10)
        ax.grid(axis="y", alpha=0.3, ls=":")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ] Saved: {save_path}")
    plt.close()

    return df_result


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  EVALUASI DBSCAN vs GROUND TRUTH IMF")
    print("=" * 65)

    # -- A. GLOBAL -------------------------------------------------
    print("\n" + "-" * 40)
    print("  [A] DBSCAN GLOBAL")
    print("-" * 40)
    df_global = load_and_merge(HASIL_GLOBAL, GT_PATH, "GLOBAL")
    metrics_global = compute_metrics(df_global, "GLOBAL")

    cm_global = confusion_matrix(
        df_global["is_crisis"], df_global["predicted_anomaly"]
    )
    plot_confusion(cm_global, "DBSCAN Global",
                   os.path.join(OUTPUT_DIR, "eval_confusion_global.png"))
    plot_roc(df_global, "DBSCAN Global",
             os.path.join(OUTPUT_DIR, "eval_roc_global.png"))

    # Simpan detail per-baris (global)
    df_global.to_csv(
        os.path.join(OUTPUT_DIR, "evaluasi_global.csv"), index=False
    )
    print(f"[SAVE] evaluasi_global.csv (detail per baris)")

    # -- B. PER NEGARA ---------------------------------------------
    print("\n" + "-" * 40)
    print("  [B] DBSCAN PER NEGARA")
    print("-" * 40)
    df_pc = load_and_merge(HASIL_PER_COUNTRY, GT_PATH, "PER-COUNTRY")
    metrics_pc = compute_metrics(df_pc, "PER-COUNTRY")

    cm_pc = confusion_matrix(
        df_pc["is_crisis"], df_pc["predicted_anomaly"]
    )
    plot_confusion(cm_pc, "DBSCAN Per-Negara",
                   os.path.join(OUTPUT_DIR, "eval_confusion_percountry.png"))
    plot_roc(df_pc, "DBSCAN Per-Negara",
             os.path.join(OUTPUT_DIR, "eval_roc_percountry.png"))

    # Metrik per negara
    df_eco_metrics = compute_metrics_per_economy(df_pc)
    df_eco_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "evaluasi_per_country.csv"), index=False
    )
    print(f"\n[SAVE] evaluasi_per_country.csv (metrik per negara)")
    print("\n[INFO] Metrik per negara (sorted by F1):")
    print(df_eco_metrics.to_string(index=False))

    plot_f1_per_economy(
        df_eco_metrics,
        os.path.join(OUTPUT_DIR, "eval_f1_per_economy.png")
    )

    # -- C. PERBANDINGAN -------------------------------------------
    plot_summary_comparison(
        [metrics_global, metrics_pc],
        os.path.join(OUTPUT_DIR, "eval_summary_comparison.png")
    )

    # -- D. METRIK UNSUPERVISED (kualitas klaster) -----------------
    print("\n" + "-" * 40)
    print("  [D] METRIK KUALITAS KLASTER (UNSUPERVISED)")
    print("-" * 40)
    clust_global = eval_cluster_quality(
        os.path.join(OUTPUT_DIR, "data_pca.csv"),
        df_global["predicted_anomaly"].values,
        "GLOBAL (PCA)"
    )
    clust_pc = eval_cluster_quality(
        os.path.join(OUTPUT_DIR, "data_scaled.csv"),
        df_pc["predicted_anomaly"].values,
        "PER-COUNTRY (Scaled)"
    )
    plot_cluster_quality(
        [clust_global, clust_pc],
        os.path.join(OUTPUT_DIR, "eval_cluster_quality.png")
    )

    # -- E. PRECISION-RECALL CURVE ---------------------------------
    print("\n" + "-" * 40)
    print("  [E] PRECISION-RECALL CURVE")
    print("-" * 40)
    plot_pr_curve(
        df_global, df_pc,
        os.path.join(OUTPUT_DIR, "eval_pr_curve.png")
    )

    # -- F. EVALUASI PER JENIS KRISIS ------------------------------
    print("\n" + "-" * 40)
    print("  [F] EVALUASI PER JENIS KRISIS")
    print("-" * 40)
    crisis_types_df = eval_per_crisis_type(
        df_pc,
        os.path.join(OUTPUT_DIR, "eval_per_crisis_type.png")
    )
    crisis_types_df.to_csv(
        os.path.join(OUTPUT_DIR, "eval_per_crisis_type.csv"), index=False
    )
    print(f"[SAVE] eval_per_crisis_type.csv")

    # -- G. RINGKASAN AKHIR ----------------------------------------
    print("\n" + "=" * 65)
    print("  RINGKASAN METRIK EVALUASI")
    print("=" * 65)
    summary_df = pd.DataFrame([metrics_global, metrics_pc])
    cols_show = ["metode", "accuracy", "precision", "recall",
                 "f1", "roc_auc", "TP", "FP", "TN", "FN"]
    print(summary_df[cols_show].to_string(index=False))
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "eval_ringkasan.csv"), index=False
    )
    print(f"\n[SAVE] eval_ringkasan.csv")
    print("\n  Output tersimpan di: ./output/")
    print("  eval_ringkasan.csv          -> ringkasan metrik global vs per-negara")
    print("  evaluasi_global.csv         -> detail baris global")
    print("  evaluasi_per_country.csv    -> metrik F1 per negara")
    print("  eval_confusion_*.png        -> confusion matrix")
    print("  eval_roc_*.png              -> ROC curve")
    print("  eval_pr_curve.png           -> Precision-Recall curve")
    print("  eval_cluster_quality.png    -> metrik kualitas klaster")
    print("  eval_per_crisis_type.png    -> evaluasi per jenis krisis")
    print("  eval_f1_per_economy.png     -> F1 per negara")
    print("  eval_summary_comparison.png -> perbandingan metrik")

    print("\n" + "=" * 65)
    print("  EVALUASI SELESAI")
    print("=" * 65)

    return summary_df, df_eco_metrics


if __name__ == "__main__":
    summary_df, df_eco_metrics = main()
