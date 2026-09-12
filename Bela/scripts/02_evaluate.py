"""
02_evaluate.py
==============
Evaluasi hasil deteksi anomali PCA (SPE, T^2, combined_score) terhadap
crisis_label sebagai ground truth eksternal, DIHITUNG UNTUK SELURUH 1.714
BARIS (normal + krisis) -- konsisten dengan cara LOF dievaluasi, sehingga
perbandingan antar-algoritma di laporan kelompok menjadi adil (apple-to-apple).

Output:
- results/evaluation_metrics.csv -> AUC-ROC, AP, Precision, Recall, F1 (skor kontinu & label biner)
- results/effect_size.csv        -> Cohen's d & Cliff's delta (SPE dan T2: krisis vs normal)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from config import RESULTS_DIR, LABEL_COL


def cohens_d(x_crisis, x_normal):
    """Cohen's d dengan pooled standard deviation."""
    n1, n2 = len(x_crisis), len(x_normal)
    v1, v2 = np.var(x_crisis, ddof=1), np.var(x_normal, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x_crisis) - np.mean(x_normal)) / pooled_sd


def cliffs_delta(x_crisis, x_normal):
    """Cliff's delta -- ukuran effect size non-parametrik, lebih tepat untuk
    SPE/T^2 yang distribusinya skewed (bukan normal). Rentang [-1, 1];
    |delta| ~ 0.147/0.33/0.474 adalah ambang kecil/sedang/besar (Romano et al., 2006).
    Dihitung dengan cara efisien memakai peringkat (setara Mann-Whitney U).
    """
    n1, n2 = len(x_crisis), len(x_normal)
    all_vals = np.concatenate([x_crisis, x_normal])
    ranks = pd.Series(all_vals).rank().values
    rank_sum_crisis = ranks[:n1].sum()
    # U statistic (Mann-Whitney) untuk grup crisis
    u_crisis = rank_sum_crisis - n1 * (n1 + 1) / 2.0
    delta = (2 * u_crisis) / (n1 * n2) - 1
    return delta


def interpret_delta(delta):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def interpret_cohend(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def main():
    scores_path = os.path.join(RESULTS_DIR, "pca_scores.csv")
    df = pd.read_csv(scores_path)

    y_true = df[LABEL_COL].values

    # ---------------------------------------------------------------
    # 1. Metrik berbasis skor kontinu (tidak bergantung 1 threshold)
    # ---------------------------------------------------------------
    rows = []
    for score_name in ["SPE", "T2", "combined_score"]:
        y_score = df[score_name].values
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        rows.append({"score": score_name, "metric": "AUC-ROC", "value": auc})
        rows.append({"score": score_name, "metric": "Average Precision (AP)", "value": ap})

    # ---------------------------------------------------------------
    # 2. Metrik berbasis label biner (hasil thresholding dengan UCL)
    # ---------------------------------------------------------------
    for label_name, y_pred_col in [
        ("SPE > UCL_SPE", "is_anomaly_SPE"),
        ("T2 > UCL_T2", "is_anomaly_T2"),
        ("Combined (SPE atau T2)", "is_anomaly_combined"),
    ]:
        y_pred = df[y_pred_col].values
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rows.append({"score": label_name, "metric": "Precision", "value": prec})
        rows.append({"score": label_name, "metric": "Recall", "value": rec})
        rows.append({"score": label_name, "metric": "F1-Score", "value": f1})
        rows.append({"score": label_name, "metric": "True Positive", "value": tp})
        rows.append({"score": label_name, "metric": "False Positive", "value": fp})
        rows.append({"score": label_name, "metric": "True Negative", "value": tn})
        rows.append({"score": label_name, "metric": "False Negative", "value": fn})

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[OK] Metrik evaluasi disimpan ke: {metrics_path}")
    print(metrics_df[metrics_df["metric"].isin(["AUC-ROC", "Average Precision (AP)",
                                                 "Precision", "Recall", "F1-Score"])]
          .to_string(index=False))

    # ---------------------------------------------------------------
    # 3. Effect size: krisis vs normal, untuk SPE dan T2
    # ---------------------------------------------------------------
    effect_rows = []
    for score_name in ["SPE", "T2", "combined_score"]:
        x_crisis = df.loc[df[LABEL_COL] == 1, score_name].values
        x_normal = df.loc[df[LABEL_COL] == 0, score_name].values

        d = cohens_d(x_crisis, x_normal)
        delta = cliffs_delta(x_crisis, x_normal)

        effect_rows.append({
            "score": score_name,
            "mean_crisis": np.mean(x_crisis),
            "mean_normal": np.mean(x_normal),
            "median_crisis": np.median(x_crisis),
            "median_normal": np.median(x_normal),
            "cohens_d": d,
            "cohens_d_interpretation": interpret_cohend(d),
            "cliffs_delta": delta,
            "cliffs_delta_interpretation": interpret_delta(delta),
        })

    effect_df = pd.DataFrame(effect_rows)
    effect_path = os.path.join(RESULTS_DIR, "effect_size.csv")
    effect_df.to_csv(effect_path, index=False)
    print(f"\n[OK] Effect size disimpan ke: {effect_path}")
    print(effect_df.to_string(index=False))

    return metrics_df, effect_df


if __name__ == "__main__":
    main()
