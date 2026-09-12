# =============================================================================
# laporan_akhir.py
# Laporan Akhir - DBSCAN Anomaly Detection pada Data Makroekonomi
# 49 Negara, 1990-2024
#
# Output:
#   output/laporan_dashboard.png   -> Visualisasi utama multi-panel
#   output/laporan_ringkasan.txt   -> Laporan teks lengkap
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
import os
import textwrap
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Path ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "output"
GT_PATH      = "ground_truth_imf.csv"
HASIL_G      = os.path.join(OUTPUT_DIR, "hasil_global.csv")
HASIL_P      = os.path.join(OUTPUT_DIR, "hasil_per_country.csv")
EVAL_R       = os.path.join(OUTPUT_DIR, "eval_ringkasan.csv")
EVAL_E       = os.path.join(OUTPUT_DIR, "evaluasi_per_country.csv")
DASHBOARD    = os.path.join(OUTPUT_DIR, "laporan_dashboard.png")
TXT_LAPORAN  = os.path.join(OUTPUT_DIR, "laporan_ringkasan.txt")

# Palet warna konsisten
C_ANOMALI = "#e63946"
C_NORMAL  = "#457b9d"
C_GOLD    = "#f4a261"
C_GREEN   = "#2a9d8f"
C_DARK    = "#1d3557"
C_LIGHT   = "#a8dadc"


# =============================================================================
# LOAD DATA
# =============================================================================
def load_all():
    gt       = pd.read_csv(GT_PATH)
    hasil_g  = pd.read_csv(HASIL_G)
    hasil_p  = pd.read_csv(HASIL_P)
    eval_r   = pd.read_csv(EVAL_R)
    eval_e   = pd.read_csv(EVAL_E)

    merged_g = hasil_g.merge(gt[["economy","year","is_crisis","crisis_name"]],
                              on=["economy","year"], how="inner")
    merged_p = hasil_p.merge(gt[["economy","year","is_crisis","crisis_name"]],
                              on=["economy","year"], how="inner")
    return gt, hasil_g, hasil_p, eval_r, eval_e, merged_g, merged_p


# =============================================================================
# PANEL 1: HEATMAP ANOMALI PER NEGARA PER TAHUN (Per-Country)
# =============================================================================
def panel_heatmap(ax, hasil_p):
    pivot = hasil_p.pivot_table(
        index="economy", columns="year",
        values="predicted_anomaly", aggfunc="max"
    ).fillna(0)

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#f1faee", C_ANOMALI], N=2
    )
    sns.heatmap(pivot, cmap=cmap, linewidths=0.15, linecolor="#ddd",
                ax=ax, cbar=False, vmin=0, vmax=1)

    # Garis vertikal krisis global
    years = list(pivot.columns)
    crisis_marks = {1997: "AFC", 2008: "GFC", 2020: "COVID-19"}
    for yr, lbl in crisis_marks.items():
        if yr in years:
            x = years.index(yr) + 0.5
            ax.axvline(x, color=C_DARK, linewidth=1.2, linestyle="--", alpha=0.7)
            ax.text(x + 0.1, -0.8, lbl, color=C_DARK, fontsize=7,
                    fontweight="bold", va="top")

    ax.set_title("Peta Anomali per Negara per Tahun (DBSCAN Per-Country)",
                 fontsize=11, fontweight="bold", color=C_DARK, pad=8)
    ax.set_xlabel("Tahun", fontsize=9)
    ax.set_ylabel("Negara", fontsize=9)
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)

    # Legend manual
    patch_a = mpatches.Patch(color=C_ANOMALI, label="Anomali")
    patch_n = mpatches.Patch(color="#f1faee", label="Normal", linewidth=0.5,
                              edgecolor="gray")
    ax.legend(handles=[patch_a, patch_n], loc="upper left",
               fontsize=7, framealpha=0.8)


# =============================================================================
# PANEL 2: TIMELINE ANOMALI PER TAHUN (dua metode)
# =============================================================================
def panel_timeline(ax, hasil_g, hasil_p, gt):
    years = sorted(hasil_p["year"].unique())

    # Per-country
    pc_timeline = (hasil_p.groupby("year")["predicted_anomaly"]
                           .sum().reindex(years, fill_value=0))
    # Global (scale x10 supaya terlihat)
    g_timeline = (hasil_g.groupby("year")["predicted_anomaly"]
                          .sum().reindex(years, fill_value=0))
    # GT crisis count
    gt_timeline = (gt.groupby("year")["is_crisis"]
                     .sum().reindex(years, fill_value=0))

    ax.fill_between(years, pc_timeline.values, alpha=0.25, color=C_ANOMALI,
                    label="_nolegend_")
    ax.plot(years, pc_timeline.values, color=C_ANOMALI, lw=2,
            marker="o", ms=3, label="Anomali Per-Country")
    ax.plot(years, gt_timeline.values, color=C_DARK, lw=1.5,
            linestyle="--", marker="s", ms=3, label="Krisis IMF (GT)")
    ax.plot(years, g_timeline.values * 10, color=C_GOLD, lw=1.5,
            linestyle=":", marker="^", ms=3,
            label="Anomali Global (x10)")

    # Anotasi krisis
    for yr, lbl in {1997:"AFC", 2008:"GFC", 2020:"COVID"}.items():
        ax.axvline(yr, color=C_DARK, linestyle="--", alpha=0.3, lw=1)
        ax.text(yr + 0.2, ax.get_ylim()[1] * 0.88, lbl,
                color=C_DARK, fontsize=7, fontweight="bold")

    ax.set_title("Timeline Anomali per Tahun vs Krisis IMF",
                 fontsize=11, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Tahun", fontsize=9)
    ax.set_ylabel("Jumlah Negara", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_xlim(min(years)-0.5, max(years)+0.5)


# =============================================================================
# PANEL 3: ROC CURVE (kedua metode)
# =============================================================================
def panel_roc(ax, merged_g, merged_p):
    for df, lbl, color, ls in [
        (merged_g, "Global",      C_GOLD,   "-"),
        (merged_p, "Per-Country", C_ANOMALI, "--"),
    ]:
        y_true  = df["is_crisis"].values
        y_score = df["anomaly_score"].values
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                label=f"{lbl} (AUC={roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.05, color=color)

    ax.plot([0,1],[0,1], color="gray", linestyle=":", lw=1,
            label="Random (AUC=0.5)")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=9)
    ax.set_title("ROC Curve — DBSCAN vs Ground Truth IMF",
                 fontsize=11, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")


# =============================================================================
# PANEL 4: METRIK EVALUASI PERBANDINGAN
# =============================================================================
def panel_metrics(ax, eval_r):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    colors_g = [C_GOLD]   * len(metrics)
    colors_p = [C_ANOMALI] * len(metrics)

    row_g = eval_r[eval_r["metode"] == "GLOBAL"].iloc[0]
    row_p = eval_r[eval_r["metode"] == "PER-COUNTRY"].iloc[0]

    vals_g = [row_g[m] for m in metrics]
    vals_p = [row_p[m] for m in metrics]

    bars_g = ax.bar(x - width/2, vals_g, width, label="DBSCAN Global",
                    color=C_GOLD, edgecolor="white", alpha=0.9)
    bars_p = ax.bar(x + width/2, vals_p, width, label="DBSCAN Per-Country",
                    color=C_ANOMALI, edgecolor="white", alpha=0.9)

    for bars in [bars_g, bars_p]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Perbandingan Metrik Evaluasi (DBSCAN vs IMF GT)",
                 fontsize=11, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, lw=1)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


# =============================================================================
# PANEL 5: F1 PER NEGARA (horizontal bar)
# =============================================================================
def panel_f1_country(ax, eval_e):
    df = eval_e.sort_values("f1", ascending=True)
    colors = [C_ANOMALI if f < 0.3 else
              C_GOLD if f < 0.6 else
              C_GREEN for f in df["f1"]]

    bars = ax.barh(df["economy"], df["f1"], color=colors,
                   edgecolor="white", linewidth=0.3, height=0.7)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, lw=1)
    ax.set_xlabel("F1-Score", fontsize=9)
    ax.set_title("F1-Score per Negara (DBSCAN Per-Country)",
                 fontsize=11, fontweight="bold", color=C_DARK)
    ax.set_xlim(0, 1.1)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Legend
    p1 = mpatches.Patch(color=C_GREEN,   label="F1 >= 0.6 (Baik)")
    p2 = mpatches.Patch(color=C_GOLD,    label="F1 0.3-0.6 (Sedang)")
    p3 = mpatches.Patch(color=C_ANOMALI, label="F1 < 0.3 (Rendah)")
    ax.legend(handles=[p1,p2,p3], fontsize=7, loc="lower right")


# =============================================================================
# PANEL 6: TOP 10 ANOMALI GLOBAL (tabel)
# =============================================================================
def panel_top_anomali(ax, hasil_g, merged_g):
    ax.axis("off")
    top = (merged_g[merged_g["predicted_anomaly"]==1]
           .sort_values("anomaly_score", ascending=False)
           [["economy","year","anomaly_score","is_crisis","crisis_name"]]
           .copy())
    top["anomaly_score"] = top["anomaly_score"].round(4)
    top["Terdeteksi GT"] = top["is_crisis"].map({1: "Ya", 0: "Tidak"})
    top["Nama Krisis"]   = top["crisis_name"].fillna("-")
    top = top[["economy","year","anomaly_score","Terdeteksi GT","Nama Krisis"]]
    top.columns = ["Negara","Tahun","Score","GT Krisis","Nama Krisis"]
    top["Nama Krisis"] = top["Nama Krisis"].apply(
        lambda x: x[:22]+"..." if len(str(x)) > 25 else x
    )

    table = ax.table(
        cellText=top.values,
        colLabels=top.columns,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif top.iloc[r-1]["GT Krisis"] == "Ya" if r > 0 else False:
            cell.set_facecolor("#d8f3dc")
        elif r % 2 == 0:
            cell.set_facecolor("#f8f9fa")
        cell.set_edgecolor("#dee2e6")

    ax.set_title("Top Anomali DBSCAN Global vs Ground Truth IMF",
                 fontsize=11, fontweight="bold", color=C_DARK, pad=10)


# =============================================================================
# BUAT DASHBOARD UTAMA
# =============================================================================
def create_dashboard(gt, hasil_g, hasil_p, eval_r, eval_e, merged_g, merged_p):
    fig = plt.figure(figsize=(24, 30), facecolor="#f8f9fa")
    fig.suptitle(
        "Laporan Akhir: Deteksi Anomali Makroekonomi dengan DBSCAN\n"
        "49 Negara | 1990-2024 | Unsupervised Learning",
        fontsize=17, fontweight="bold", color=C_DARK, y=0.99
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97,
        top=0.96, bottom=0.03
    )

    # Baris 1: Heatmap (full width)
    ax1 = fig.add_subplot(gs[0, :])
    panel_heatmap(ax1, hasil_p)

    # Baris 2: Timeline (2/3) + Metrics (1/3)
    ax2 = fig.add_subplot(gs[1, :2])
    panel_timeline(ax2, hasil_g, hasil_p, gt)

    ax3 = fig.add_subplot(gs[1, 2])
    panel_metrics(ax3, eval_r)

    # Baris 3: ROC (1/3) + F1 per economy (2/3)
    ax4 = fig.add_subplot(gs[2, 0])
    panel_roc(ax4, merged_g, merged_p)

    ax5 = fig.add_subplot(gs[2, 1:])
    panel_f1_country(ax5, eval_e)

    # Baris 4: Top anomali global table (full)
    ax6 = fig.add_subplot(gs[3, :])
    panel_top_anomali(ax6, hasil_g, merged_g)

    plt.savefig(DASHBOARD, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[DASHBOARD] Saved: {DASHBOARD}")
    plt.close()


# =============================================================================
# LAPORAN TEKS
# =============================================================================
def buat_laporan_teks(gt, hasil_g, hasil_p, eval_r, eval_e, merged_g, merged_p):
    lines = []
    SEP = "=" * 70
    SEP2 = "-" * 70

    def add(text=""):
        lines.append(text)

    row_g = eval_r[eval_r["metode"] == "GLOBAL"].iloc[0]
    row_p = eval_r[eval_r["metode"] == "PER-COUNTRY"].iloc[0]

    add(SEP)
    add("LAPORAN AKHIR: DETEKSI ANOMALI MAKROEKONOMI DENGAN DBSCAN")
    add(SEP)
    add(f"Tanggal      : {datetime.now().strftime('%d %B %Y, %H:%M')}")
    add("Metode       : DBSCAN (Density-Based Spatial Clustering of Applications")
    add("               with Noise) - Unsupervised Learning")
    add("Data         : Indikator Makroekonomi 49 Negara, 1990-2024")
    add("Ground Truth : IMF Historical Crisis Database")
    add()

    add(SEP)
    add("1. DESKRIPSI DATA")
    add(SEP)
    add(f"  - Total observasi        : {len(gt):,} baris")
    add(f"  - Jumlah negara          : {gt['economy'].nunique()} negara")
    add(f"  - Periode                : {gt['year'].min()} - {gt['year'].max()}")
    add(f"  - Fitur makroekonomi     : 14 indikator")
    add(f"  - Krisis historis (GT)   : {gt['is_crisis'].sum()} tahun-negara")
    add(f"  - Periode normal (GT)    : {(gt['is_crisis']==0).sum()} tahun-negara")
    add()
    add("  Indikator yang digunakan:")
    add("    GDP_Growth, Inflation_CPI, Unemployment, Current_Account_GDP,")
    add("    Reserves_Months_Imports, FDI_Inflows_GDP, Exports_GDP, Imports_GDP,")
    add("    Gross_Savings_GDP, Investment_GDP, Manufacturing_Value,")
    add("    Domestic_Credit_GDP, Broad_Money_Growth, Exchange_Depreciation")
    add()

    add(SEP)
    add("2. PREPROCESSING")
    add(SEP)
    add("  a) Feature Engineering:")
    add("     Exchange_Rate -> Exchange_Depreciation (% perubahan YoY per negara)")
    add("     Alasan: Skala Exchange_Rate berbeda jauh antar negara (IDR vs USD)")
    add()
    add("  b) Imputasi Missing Values:")
    add("     - Tahap 1: Median per economy (imputasi kontekstual per negara)")
    add("     - Tahap 2: Median global sebagai fallback")
    add("     Kolom dengan missing terbanyak: Broad_Money_Growth (25%), Domestic_Credit_GDP (18%)")
    add()
    add("  c) Standarisasi:")
    add("     - StandardScaler (z-score) pada 14 fitur")
    add("     - Wajib: DBSCAN berbasis jarak Euclidean")
    add()
    add("  d) Reduksi Dimensi (PCA):")
    add("     - Input: 14 fitur -> Output: 9 komponen (threshold 90% varians)")
    add("     - Varians tercakup: 92.59%")
    add("     - Re-standarisasi ulang komponen PCA sebelum DBSCAN Global")
    add()

    add(SEP)
    add("3. METODOLOGI DBSCAN")
    add(SEP)
    add("  DBSCAN: Density-Based Spatial Clustering of Applications with Noise")
    add("  Prinsip: Titik yang terisolasi (jauh dari klaster padat) = Noise = Anomali")
    add()
    add("  Definisi: Anomaly = DBSCAN label -1 (noise point)")
    add("  Prediksi : label -1 -> predicted_anomaly = 1 (Anomali/Krisis)")
    add("             label >= 0 -> predicted_anomaly = 0 (Normal)")
    add()
    add("  Anomaly Score: Jarak minimum ke core point terdekat, dinormalisasi [0,1]")
    add("                 Score = 1 -> paling jauh/paling anomali")
    add()
    add("  Tuning Parameter:")
    add("  - eps  : Kneedle Algorithm (max perpendicular distance ke diagonal)")
    add("  - min_samples = 9 (Global) / 3 (Per-Negara)")
    add()
    add("  [A] DBSCAN GLOBAL")
    add("      Input : data_pca.csv (9 komponen PCA, standarisasi ulang)")
    add(f"      eps   : 3.1631 (otomatis via Kneedle)")
    add("      Tujuan: Temukan pasangan (negara, tahun) yang anomali secara global")
    add()
    add("  [B] DBSCAN PER-NEGARA")
    add("      Input : data_scaled.csv (14 fitur, z-score)")
    add("      eps   : Adaptif per negara (Kneedle, min 1.0)")
    add("      Tujuan: Temukan tahun abnormal dalam sejarah setiap negara")
    add()

    add(SEP)
    add("4. HASIL DETEKSI ANOMALI")
    add(SEP)
    add()
    add("  [A] DBSCAN GLOBAL")
    add(f"      Total anomali    : {hasil_g['predicted_anomaly'].sum()} ({hasil_g['predicted_anomaly'].sum()/len(hasil_g)*100:.2f}%)")
    add()
    add("      Anomali terdeteksi:")
    for _, row in (hasil_g[hasil_g["predicted_anomaly"]==1]
                   .sort_values("anomaly_score", ascending=False).iterrows()):
        gt_row = merged_g[(merged_g["economy"]==row["economy"]) &
                          (merged_g["year"]==row["year"])].iloc[0]
        krisis = gt_row["crisis_name"] if pd.notna(gt_row["crisis_name"]) else "Tidak ada di GT"
        add(f"        {row['economy']} {row['year']}  score={row['anomaly_score']:.4f}  |  {krisis}")
    add()

    add("  [B] DBSCAN PER-NEGARA")
    add(f"      Total anomali    : {hasil_p['predicted_anomaly'].sum()} ({hasil_p['predicted_anomaly'].sum()/len(hasil_p)*100:.2f}%)")
    add()
    add("      Distribusi anomali per negara (top 15):")
    summ = (hasil_p.groupby("economy")["predicted_anomaly"].sum()
                   .sort_values(ascending=False).head(15))
    for eco, n in summ.items():
        add(f"        {eco:5s}: {n} tahun anomali")
    add()

    # Deteksi krisis besar
    add("      Deteksi Krisis Besar (Per-Country):")
    crises = {
        "Asian FC 1997-98": {"years": range(1997,2000),
                              "eco": ["IDN","THA","KOR","MYS","PHL"]},
        "GFC 2008-09":      {"years": range(2007,2012),
                              "eco": ["USA","GBR","DEU","FRA","ESP","IRL"]},
        "COVID-19 2020":    {"years": [2020],
                              "eco": list(hasil_p["economy"].unique())},
    }
    for crisis_name, info in crises.items():
        detected = hasil_p[
            (hasil_p["year"].isin(info["years"])) &
            (hasil_p["economy"].isin(info["eco"])) &
            (hasil_p["predicted_anomaly"] == 1)
        ]
        total_possible = len(info["eco"])
        add(f"        {crisis_name}: {len(detected)} terdeteksi dari {total_possible} negara terdampak")
    add()

    add(SEP)
    add("5. EVALUASI vs GROUND TRUTH IMF")
    add(SEP)
    add()
    add(f"  {'Metrik':<15} {'GLOBAL':>12} {'PER-NEGARA':>12}")
    add(f"  {'-'*40}")
    for m in ["accuracy","precision","recall","f1","roc_auc"]:
        add(f"  {m.upper():<15} {row_g[m]:>12.4f} {row_p[m]:>12.4f}")
    add()
    add(f"  Confusion Matrix:")
    add(f"  {'':20} {'GLOBAL':>10} {'PER-NEGARA':>12}")
    for k in ["TP","FP","TN","FN"]:
        add(f"  {k:<20} {int(row_g[k]):>10} {int(row_p[k]):>12}")
    add()

    add("  Interpretasi:")
    add("  - DBSCAN Global : Precision tinggi (0.615) tapi Recall sangat rendah")
    add("    (0.035). Model sangat selektif - hanya deteksi anomali yang benar-")
    add("    benar ekstrem secara global (hiperinflasi BRA, PER, dst).")
    add()
    add("  - DBSCAN Per-Negara : Lebih seimbang (F1=0.354, ROC-AUC=0.638).")
    add("    Recall 0.323 artinya ~1/3 krisis historis berhasil terdeteksi.")
    add("    Metode ini lebih sensitif terhadap anomali kontekstual per negara.")
    add()

    add(SEP)
    add("6. TEMUAN UTAMA")
    add(SEP)
    add()
    add("  a) Krisis Hiperinflasi Amerika Latin (1990-1994)")
    add("     Brazil, Peru, Argentina: Terdeteksi kuat di GLOBAL karena nilai")
    add("     inflasi dan pertumbuhan uang beredar yang sangat ekstrem.")
    add()
    add("  b) Asian Financial Crisis 1997-98")
    add("     Indonesia (IDN) terdeteksi di Global (1998). Per-country mendeteksi")
    add("     beberapa negara Asia yang terdampak langsung.")
    add()
    add("  c) Global Financial Crisis 2008-09")
    add("     Per-country berhasil mendeteksi beberapa negara Eropa (AUT, BEL,")
    add("     DEU, GBR, ITA) pada tahun 2009 sebagai anomali.")
    add()
    add("  d) COVID-19 2020")
    add("     31 dari 49 negara dideteksi anomali di tahun 2020 (Per-Country).")
    add("     Ini merupakan deteksi terbaik - guncangan COVID memang mempengaruhi")
    add("     hampir semua indikator makroekonomi secara bersamaan.")
    add()
    add("  e) Krisis Transisi Pasca-Komunis 1991-92")
    add("     Romania (ROU) terdeteksi di GLOBAL pada 1991-92.")
    add()
    add("  f) Greek Debt Crisis 2010-15")
    add("     Yunani (GRC) mendapat F1=0.556 dengan Recall=0.833 -")
    add("     sebagian besar tahun krisis Yunani berhasil teridentifikasi.")
    add()

    add(SEP)
    add("7. KETERBATASAN DAN CATATAN")
    add(SEP)
    add()
    add("  1. DBSCAN tidak dirancang untuk deteksi anomali berlabel - kinerja")
    add("     terbatas wajar untuk unsupervised learning.")
    add()
    add("  2. Ground truth IMF mencakup krisis yang berlangsung bertahun-tahun")
    add("     (multiyear), sehingga banyak tahun di tengah periode krisis")
    add("     mungkin tidak menunjukkan tanda ekstrem di indikator tahunan.")
    add()
    add("  3. Beberapa negara (SGP, IRL, NLD) memiliki struktur ekonomi unik")
    add("     sehingga krisis mereka tidak terpola di ruang fitur yang sama.")
    add()
    add("  4. Imputasi 25% missing pada Broad_Money_Growth dapat mempengaruhi")
    add("     deteksi untuk negara dengan data awal terbatas.")
    add()
    add("  5. eps dipilih otomatis - optimasi manual mungkin meningkatkan kinerja.")
    add()

    add(SEP)
    add("8. KESIMPULAN")
    add(SEP)
    add()
    add("  DBSCAN sebagai metode unsupervised learning mampu mendeteksi anomali")
    add("  makroekonomi dengan performa yang bervariasi antar pendekatan:")
    add()
    add("  * Global  : Sangat presisi (0.615) tapi recall rendah (0.035).")
    add("    Terbaik untuk menangkap krisis yang benar-benar ekstrem global.")
    add()
    add("  * Per-Negara : Lebih seimbang (F1=0.354, AUC=0.638).")
    add("    Mampu mendeteksi ~1/3 krisis historis tanpa label apapun.")
    add()
    add("  Deteksi COVID-19 2020 (31/49 negara) dan Hiperinflasi Amerika Latin")
    add("  merupakan keberhasilan paling menonjol dari model ini.")
    add()
    add(f"  File output: output/ | {datetime.now().strftime('%Y-%m-%d')}")
    add(SEP)

    laporan_text = "\n".join(lines)
    with open(TXT_LAPORAN, "w", encoding="utf-8") as f:
        f.write(laporan_text)
    print(f"[LAPORAN] Saved: {TXT_LAPORAN}")
    print()
    print(laporan_text)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  MEMBUAT LAPORAN AKHIR")
    print("=" * 65)

    gt, hasil_g, hasil_p, eval_r, eval_e, merged_g, merged_p = load_all()

    print("[1/2] Membuat dashboard visualisasi...")
    create_dashboard(gt, hasil_g, hasil_p, eval_r, eval_e, merged_g, merged_p)

    print("[2/2] Membuat laporan teks...")
    buat_laporan_teks(gt, hasil_g, hasil_p, eval_r, eval_e, merged_g, merged_p)

    print()
    print("=" * 65)
    print("  LAPORAN AKHIR SELESAI")
    print("=" * 65)
    print()
    print("  Output:")
    print(f"  - {DASHBOARD}")
    print(f"  - {TXT_LAPORAN}")


if __name__ == "__main__":
    main()
