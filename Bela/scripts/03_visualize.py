"""
03_visualize.py
================
Membuat 4 visualisasi utama untuk bagian PCA-based Anomaly Detection:

1. figures/01_scree_plot.png        -> explained variance per komponen + kumulatif
2. figures/02_control_chart.png     -> SPE & T^2 tiap observasi vs UCL (normal vs krisis)
3. figures/03_contribution_plot.png -> rata-rata kontribusi tiap variabel pada anomali
4. figures/04_effect_size_plot.png  -> Cohen's d & Cliff's delta (krisis vs normal)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR, FEATURE_COLS, LABEL_COL

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_NORMAL = "#4C72B0"
COLOR_CRISIS = "#C44E52"
COLOR_UCL = "#333333"


def plot_scree(summary_path, meta_path, out_path):
    summary = pd.read_csv(summary_path)
    meta = pd.read_csv(meta_path).set_index("metric")["value"]
    k = int(meta["k_components"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(summary) + 1)

    ax1.bar(x, summary["explained_variance_ratio"] * 100,
            color=[COLOR_CRISIS if i < k else "#B0B0B0" for i in range(len(summary))],
            label="Explained variance per komponen")
    ax1.set_xlabel("Komponen Utama (PC)")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])

    ax2 = ax1.twinx()
    ax2.plot(x, summary["cumulative_explained_variance"] * 100,
              color=COLOR_UCL, marker="o", linewidth=2, label="Kumulatif")
    ax2.axhline(95, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(k, color=COLOR_NORMAL, linestyle=":", linewidth=1.5)
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_ylim(0, 105)

    ax2.text(k + 0.1, 40, f"k = {k}\n({summary['cumulative_explained_variance'].iloc[k-1]*100:.1f}%)",
              color=COLOR_NORMAL, fontsize=9)

    fig.suptitle("Scree Plot - PCA pada Indikator Makroekonomi (dilatih dari data normal)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Scree plot disimpan ke: {out_path}")


def plot_control_chart(scores_path, out_path):
    df = pd.read_csv(scores_path).reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    ucl_spe = df["UCL_SPE"].iloc[0]
    ucl_t2 = df["UCL_T2"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, col, ucl, title in [
        (axes[0], "SPE", ucl_spe, "Control Chart - SPE (Q-statistic)"),
        (axes[1], "T2", ucl_t2, "Control Chart - Hotelling's T\u00b2"),
    ]:
        normal_mask = df[LABEL_COL] == 0
        crisis_mask = df[LABEL_COL] == 1
        ax.scatter(df.loc[normal_mask, "idx"], df.loc[normal_mask, col],
                   s=10, color=COLOR_NORMAL, alpha=0.6, label="Normal")
        ax.scatter(df.loc[crisis_mask, "idx"], df.loc[crisis_mask, col],
                   s=16, color=COLOR_CRISIS, alpha=0.85, label="Krisis (crisis_label=1)")
        ax.axhline(ucl, color=COLOR_UCL, linestyle="--", linewidth=1.3,
                   label=f"UCL (\u03b1=0.05) = {ucl:.2f}")
        ax.set_ylabel(col)
        ax.set_title(title, fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[1].set_xlabel("Indeks observasi (panel data: negara \u00d7 tahun, 1990-2024)")
    fig.suptitle("Control Chart Deteksi Anomali Berbasis PCA (UCL dihitung dari data normal saja)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Control chart disimpan ke: {out_path}")


def plot_contribution(contrib_path, out_path):
    df = pd.read_csv(contrib_path)
    crisis_anom = df[(df[LABEL_COL] == 1) & (df["is_anomaly_combined"] == 1)]
    normal_anom = df[(df[LABEL_COL] == 0) & (df["is_anomaly_combined"] == 1)]

    mean_contrib_crisis = crisis_anom[FEATURE_COLS].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(mean_contrib_crisis))
    ax.barh(y_pos, mean_contrib_crisis.values, color=COLOR_CRISIS, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mean_contrib_crisis.index)
    ax.set_xlabel("Rata-rata kontribusi terhadap SPE (residual\u00b2)")
    ax.set_title(f"Contribution Plot - Variabel Paling Berkontribusi pada\n"
                 f"Anomali Terdeteksi saat Krisis (n={len(crisis_anom)} observasi)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Contribution plot disimpan ke: {out_path}")
    return mean_contrib_crisis


def plot_effect_size(effect_path, out_path):
    df = pd.read_csv(effect_path)
    labels = df["score"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df["cohens_d"], width, label="Cohen's d", color=COLOR_NORMAL)
    ax.bar(x + width/2, df["cliffs_delta"], width, label="Cliff's delta", color=COLOR_CRISIS)

    for thresh, txt in [(0.2, "kecil"), (0.5, "sedang"), (0.8, "besar")]:
        ax.axhline(thresh, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Effect size (krisis vs normal)")
    ax.set_title("Effect Size Skor Anomali PCA: Krisis vs Normal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Effect size plot disimpan ke: {out_path}")


def main():
    scores_path = os.path.join(RESULTS_DIR, "pca_scores.csv")
    summary_path = os.path.join(RESULTS_DIR, "pca_model_summary.csv")
    meta_path = os.path.join(RESULTS_DIR, "pca_run_metadata.csv")
    contrib_path = os.path.join(RESULTS_DIR, "contribution_matrix.csv")
    effect_path = os.path.join(RESULTS_DIR, "effect_size.csv")

    plot_scree(summary_path, meta_path, os.path.join(FIGURES_DIR, "01_scree_plot.png"))
    plot_control_chart(scores_path, os.path.join(FIGURES_DIR, "02_control_chart.png"))
    plot_contribution(contrib_path, os.path.join(FIGURES_DIR, "03_contribution_plot.png"))
    plot_effect_size(effect_path, os.path.join(FIGURES_DIR, "04_effect_size_plot.png"))


if __name__ == "__main__":
    main()
