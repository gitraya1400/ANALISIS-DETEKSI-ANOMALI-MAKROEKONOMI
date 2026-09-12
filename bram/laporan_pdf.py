# =============================================================================
# laporan_pdf.py  (versi naratif - mudah dipahami)
# Laporan DBSCAN Anomaly Detection — gaya bercerita, bahasa sederhana
# Output: output/Laporan_DBSCAN_Anomaly_Detection.pdf
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score
)
import textwrap, warnings, os
from datetime import datetime

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
GT_PATH    = "ground_truth_imf.csv"
HASIL_G    = os.path.join(OUTPUT_DIR, "hasil_global.csv")
HASIL_P    = os.path.join(OUTPUT_DIR, "hasil_per_country.csv")
EVAL_R     = os.path.join(OUTPUT_DIR, "eval_ringkasan.csv")
EVAL_E     = os.path.join(OUTPUT_DIR, "evaluasi_per_country.csv")
EVAL_CQ    = os.path.join(OUTPUT_DIR, "eval_cluster_quality.csv")
EVAL_CT    = os.path.join(OUTPUT_DIR, "eval_per_crisis_type.csv")
PDF_OUT    = os.path.join(OUTPUT_DIR, "Laporan_DBSCAN_Anomaly_Detection.pdf")

C_DARK  = "#1d3557"
C_RED   = "#e63946"
C_GOLD  = "#f4a261"
C_GREEN = "#2a9d8f"
C_BLUE  = "#457b9d"
C_LIGHT = "#a8dadc"
C_BG    = "#f8f9fa"
C_WHITE = "#ffffff"
C_GRAY  = "#6c757d"
C_BOX   = "#e9f5fb"


def load_all():
    gt      = pd.read_csv(GT_PATH)
    hasil_g = pd.read_csv(HASIL_G)
    hasil_p = pd.read_csv(HASIL_P)
    eval_r  = pd.read_csv(EVAL_R)
    eval_e  = pd.read_csv(EVAL_E)
    eval_cq = pd.read_csv(EVAL_CQ) if os.path.exists(EVAL_CQ) else None
    eval_ct = pd.read_csv(EVAL_CT) if os.path.exists(EVAL_CT) else None
    mg = hasil_g.merge(gt[["economy","year","is_crisis","crisis_name"]],
                       on=["economy","year"], how="inner")
    mp = hasil_p.merge(gt[["economy","year","is_crisis","crisis_name"]],
                       on=["economy","year"], how="inner")
    return gt, hasil_g, hasil_p, eval_r, eval_e, mg, mp, eval_cq, eval_ct


def setup_page(fig, title, subtitle=None, step=None):
    fig.patch.set_facecolor(C_BG)
    header = fig.add_axes([0, 0.955, 1, 0.045])
    header.set_facecolor(C_DARK)
    header.axis("off")
    if step:
        header.text(0.02, 0.5, step, va="center", fontsize=9,
                    color=C_GOLD, fontweight="bold")
    header.text(0.5, 0.5, "Laporan: Deteksi Anomali Makroekonomi dengan DBSCAN",
                va="center", ha="center", fontsize=8.5, color=C_LIGHT)
    header.text(0.98, 0.5, datetime.now().strftime("%d %b %Y"),
                va="center", ha="right", fontsize=8, color="#adb5bd")
    fig.text(0.5, 0.925, title, ha="center", fontsize=15,
             fontweight="bold", color=C_DARK)
    if subtitle:
        fig.text(0.5, 0.900, subtitle, ha="center", fontsize=9.5,
                 color=C_GRAY, style="italic")


def textbox(fig, x, y, w, h, text, title=None,
            bg=C_BOX, border=C_BLUE, fontsize=9, title_color=C_DARK):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for spine in ["left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(border)
        ax.spines[spine].set_linewidth(4)
    if title:
        ax.text(0.03, 0.92, title, va="top", fontsize=fontsize+0.5,
                fontweight="bold", color=title_color, transform=ax.transAxes)
        start_y = 0.78
    else:
        start_y = 0.93
    lines = text.split("\n")
    y_pos = start_y
    for line in lines:
        if line.strip() == "":
            y_pos -= 0.04
            continue
        wrapped = textwrap.wrap(line, width=int(w * 115)) or [""]
        for wl in wrapped:
            ax.text(0.03, y_pos, wl, va="top", fontsize=fontsize,
                    color="#343a40", transform=ax.transAxes)
            y_pos -= 0.10 * (9.5 / fontsize)
            if y_pos < 0.02:
                break


def callout(fig, x, y, w, h, icon, text, bg=C_BOX, border=C_GOLD):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.patch.set_edgecolor(border)
    ax.patch.set_linewidth(1.5)
    ax.text(0.015, 0.5, icon, va="center", fontsize=14, transform=ax.transAxes)
    ax.text(0.06, 0.5, text, va="center", fontsize=8.8, color="#343a40",
            transform=ax.transAxes, wrap=True)


# ===========================================================================
# HALAMAN 1: COVER
# ===========================================================================
def page_cover(fig):
    fig.patch.set_facecolor(C_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_DARK); ax.axis("off")
    ax.axhline(0.90, xmin=0.05, xmax=0.95, color=C_RED,  lw=4)
    ax.axhline(0.87, xmin=0.05, xmax=0.95, color=C_GOLD, lw=1.5)
    fig.text(0.5, 0.79, "L A P O R A N   A K H I R",
             ha="center", fontsize=13, color=C_LIGHT)
    fig.text(0.5, 0.69, "Deteksi Anomali Ekonomi",
             ha="center", fontsize=30, color=C_WHITE, fontweight="bold")
    fig.text(0.5, 0.60, "dengan Metode DBSCAN",
             ha="center", fontsize=22, color=C_LIGHT, fontweight="bold")
    ax.axhline(0.55, xmin=0.2, xmax=0.8, color=C_GOLD, lw=1, alpha=0.5)
    fig.text(0.5, 0.49,
             '"Kapan suatu negara mengalami kondisi ekonomi yang tidak normal?"',
             ha="center", fontsize=11, color="#a8dadc", style="italic")
    infos = [
        ("Data",     "49 Negara | 1990-2024 | 14 Indikator Makroekonomi"),
        ("Metode",   "DBSCAN - Unsupervised Machine Learning"),
        ("Validasi", "IMF Historical Crisis Database"),
        ("Tanggal",  datetime.now().strftime("%d %B %Y")),
    ]
    y0 = 0.41
    for lbl, val in infos:
        fig.text(0.28, y0, f"{lbl}:", ha="right", fontsize=10.5,
                 color=C_GOLD, fontweight="bold")
        fig.text(0.30, y0, val, ha="left", fontsize=10.5, color=C_WHITE)
        y0 -= 0.055
    ax.axhline(0.14, xmin=0.05, xmax=0.95, color=C_GOLD, lw=1.5)
    ax.axhline(0.11, xmin=0.05, xmax=0.95, color=C_RED,  lw=3)
    fig.text(0.5, 0.065,
             "Laporan ini menjelaskan bagaimana komputer dapat mendeteksi\n"
             "'tahun-tahun yang tidak normal' dalam perekonomian sebuah negara\n"
             "secara otomatis, tanpa perlu diberitahu contoh krisis sebelumnya.",
             ha="center", fontsize=9.5, color="#adb5bd", multialignment="center")


# ===========================================================================
# HALAMAN 2: APA ITU DBSCAN?
# ===========================================================================
def page_konsep(fig):
    setup_page(fig,
               "Apa itu DBSCAN dan Bagaimana Cara Kerjanya?",
               "Penjelasan sederhana sebelum masuk ke hasil")

    ax_ana = fig.add_axes([0.56, 0.36, 0.41, 0.52])
    ax_ana.set_facecolor(C_WHITE)
    ax_ana.set_xlim(0, 10); ax_ana.set_ylim(0, 10)
    ax_ana.set_title("Ilustrasi Cara Kerja DBSCAN",
                     fontsize=9.5, fontweight="bold", color=C_DARK)
    ax_ana.axis("off")
    np.random.seed(42)
    c1x = np.random.normal(2.5, 0.6, 20)
    c1y = np.random.normal(2.5, 0.6, 20)
    c2x = np.random.normal(7.5, 0.7, 20)
    c2y = np.random.normal(7.0, 0.7, 20)
    ox = [1.0, 9.2, 5.0, 0.5]
    oy = [9.0, 1.0, 5.5, 5.0]
    ax_ana.scatter(c1x, c1y, color=C_BLUE,  s=40, alpha=0.7, zorder=3)
    ax_ana.scatter(c2x, c2y, color=C_GREEN, s=40, alpha=0.7, zorder=3)
    ax_ana.scatter(ox, oy,   color=C_RED,   s=80, marker="X", zorder=4)
    circle = plt.Circle((2.5, 2.5), 1.2, fill=False,
                         color=C_BLUE, ls="--", lw=1.2, alpha=0.6)
    ax_ana.add_patch(circle)
    ax_ana.annotate("  eps", xy=(3.5, 3.5), fontsize=8, color=C_BLUE, style="italic")
    ax_ana.text(2.5, 0.3, "Kelompok Padat = NORMAL",
                ha="center", fontsize=8, color=C_BLUE, fontweight="bold")
    ax_ana.text(7.5, 5.0, "Kelompok Padat = NORMAL",
                ha="center", fontsize=8, color=C_GREEN, fontweight="bold")
    ax_ana.text(5.0, 6.3, "Titik Sendirian\n= ANOMALI",
                ha="center", fontsize=9, color=C_RED, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e3",
                           edgecolor=C_RED, lw=1.2))

    textbox(fig, 0.04, 0.64, 0.48, 0.28,
            title="Analogi: Bayangkan Sebuah Pesta",
            text=(
                "Di sebuah pesta, kebanyakan tamu berkumpul dalam kelompok-kelompok "
                "(makan bareng, ngobrol, dll). Tapi ada beberapa orang yang "
                "berdiri SENDIRIAN, jauh dari kelompok manapun.\n\n"
                "DBSCAN bekerja persis seperti itu:\n"
                "  Kelompok ramai   =  kondisi ekonomi NORMAL\n"
                "  Orang sendirian  =  kondisi ekonomi ANOMALI/KRISIS\n\n"
                "Komputer tidak diberitahu mana yang krisis. Ia hanya melihat "
                "pola kepadatan, lalu menyimpulkan sendiri mana yang 'berbeda'."
            ),
            border=C_BLUE, bg=C_BOX, fontsize=9)

    textbox(fig, 0.04, 0.36, 0.48, 0.26,
            title="Dua Parameter Kunci DBSCAN",
            text=(
                "eps = Seberapa jauh komputer 'melihat' dari satu titik data.\n"
                "  Seperti radius pandang untuk mencari teman di sekitar kita.\n\n"
                "min_samples = Minimal berapa tetangga yang harus ada di dalam\n"
                "  radius eps agar suatu titik dianggap bagian dari 'kelompok ramai'.\n\n"
                "Cara menentukan eps secara otomatis:\n"
                "  Kami pakai metode Kneedle - cari titik belok kurva jarak\n"
                "  secara matematis, tanpa tebak-tebakan manual."
            ),
            border=C_GOLD, bg="#fff8ee", fontsize=9)

    textbox(fig, 0.04, 0.06, 0.92, 0.27,
            title="Mengapa DBSCAN untuk Deteksi Anomali Ekonomi?",
            text=(
                "Ada tiga alasan utama memilih DBSCAN:\n\n"
                "1. Tidak butuh label (Unsupervised) - Kita tidak perlu tahu dulu mana tahun "
                "krisis dan mana yang normal. DBSCAN menemukan sendiri dari pola data. Ini sangat berguna "
                "karena data krisis historis sering tidak lengkap atau tidak konsisten antar sumber.\n\n"
                "2. Fleksibel terhadap bentuk klaster - Kondisi ekonomi normal tidak selalu membentuk "
                "pola geometri sederhana. DBSCAN bisa mengenali kelompok dengan bentuk apapun.\n\n"
                "3. Noise secara alami = Anomali - Titik yang tidak masuk klaster manapun (diberi "
                "label 'noise' = -1 oleh DBSCAN) secara alami menjadi kandidat anomali/krisis."
            ),
            border=C_GREEN, bg="#edfaf7", fontsize=9)


# ===========================================================================
# HALAMAN 3: DATA
# ===========================================================================
def page_data(fig, gt):
    setup_page(fig,
               "Data yang Kita Gunakan",
               "49 negara, 35 tahun, 14 indikator ekonomi")

    indikators = [
        ("GDP Growth (%/thn)",      "Seberapa cepat ekonomi suatu negara tumbuh dalam setahun"),
        ("Inflation CPI (%)",       "Seberapa cepat harga-harga barang naik"),
        ("Unemployment (%)",        "Berapa persen angkatan kerja yang tidak punya pekerjaan"),
        ("Current Account/GDP (%)", "Apakah negara lebih banyak ekspor atau impor?"),
        ("Reserves (bln impor)",    "Cadangan devisa cukup untuk berapa bulan impor?"),
        ("Exchange Depreciation %", "Seberapa besar nilai mata uang melemah/menguat per tahun"),
        ("FDI Inflows/GDP (%)",     "Seberapa banyak investasi asing yang masuk ke negara"),
        ("Exports/GDP (%)",         "Seberapa besar peran ekspor dalam perekonomian"),
        ("Imports/GDP (%)",         "Seberapa besar peran impor dalam perekonomian"),
        ("Gross Savings/GDP (%)",   "Seberapa besar masyarakat dan pemerintah menabung"),
        ("Investment/GDP (%)",      "Seberapa banyak investasi yang terjadi di dalam negeri"),
        ("Manufacturing Value",     "Seberapa besar kontribusi industri manufaktur"),
        ("Domestic Credit/GDP (%)", "Seberapa banyak pinjaman yang disalurkan oleh bank"),
        ("Broad Money Growth (%)",  "Seberapa cepat jumlah uang beredar di masyarakat bertambah"),
    ]

    ax_tbl = fig.add_axes([0.04, 0.53, 0.57, 0.37])
    ax_tbl.axis("off")
    ax_tbl.set_title("14 Indikator yang Digunakan dan Artinya",
                     fontsize=9.5, fontweight="bold", color=C_DARK, pad=6)
    tbl = ax_tbl.table(
        cellText=indikators,
        colLabels=["Nama Indikator", "Apa Artinya (Bahasa Sederhana)"],
        cellLoc="left", loc="center", bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef2f7")
        else:
            cell.set_facecolor(C_WHITE)
        cell.set_edgecolor("#dee2e6")

    ax_mv = fig.add_axes([0.67, 0.53, 0.30, 0.37])
    missing = {
        "Broad Money Growth": 25.0, "Domestic Credit": 18.6,
        "Gross Savings": 9.3,       "Manufacturing": 6.8,
        "Current Account": 6.6,     "Reserves": 6.6,
        "Investment": 3.7,          "Inflation": 3.4,
        "Exports": 3.4,             "Imports": 3.4,
    }
    ys = list(missing.keys())
    xs = list(missing.values())
    cols = [C_RED if v > 15 else C_GOLD if v > 5 else C_BLUE for v in xs]
    ax_mv.barh(ys[::-1], xs[::-1], color=cols[::-1], edgecolor="white", height=0.6)
    ax_mv.set_xlabel("% Data Kosong", fontsize=8)
    ax_mv.set_title("Kolom dengan Data Kosong",
                    fontsize=9, fontweight="bold", color=C_DARK)
    ax_mv.tick_params(axis="y", labelsize=7.5)
    ax_mv.tick_params(axis="x", labelsize=7.5)
    ax_mv.grid(axis="x", alpha=0.3, ls=":")

    textbox(fig, 0.04, 0.33, 0.44, 0.18,
            title="Dari Mana Data Ini?",
            text=(
                "Data ekonomi bersumber dari World Bank dan IMF. Kami mengumpulkan "
                "data 49 negara (termasuk Indonesia, Amerika, China, Eropa, Afrika, "
                "Amerika Latin) selama 35 tahun (1990-2024). Tiap negara punya 35 "
                "baris data, satu per tahun."
            ),
            border=C_BLUE, bg=C_BOX, fontsize=9)

    textbox(fig, 0.52, 0.33, 0.44, 0.18,
            title="Mengapa Ada Data yang Kosong?",
            text=(
                "Tidak semua negara melaporkan semua indikator setiap tahun. "
                "Negara berkembang sering punya data tidak lengkap di tahun-tahun "
                "awal (1990-an). 'Broad Money Growth' paling banyak kosong (25%) "
                "- perlu 'diisi' sebelum analisis bisa dijalankan."
            ),
            border=C_GOLD, bg="#fff8ee", fontsize=9)

    stats = [
        ("1.715", "Total Baris\nData"),
        ("49",    "Negara\nDianalisis"),
        ("35",    "Tahun\n(1990-2024)"),
        ("14",    "Indikator\nEkonomi"),
        ("229",   "Tahun-Krisis\n(IMF GT)"),
        ("13.4%", "Proporsi\nKrisis"),
    ]
    for i, (val, lbl) in enumerate(stats):
        bx = 0.04 + i * 0.156
        ax_k = fig.add_axes([bx, 0.05, 0.14, 0.25])
        ax_k.set_facecolor(C_DARK if i % 2 == 0 else C_BLUE)
        ax_k.axis("off")
        ax_k.text(0.5, 0.63, val, ha="center", va="center",
                  fontsize=20, fontweight="bold", color=C_WHITE,
                  transform=ax_k.transAxes)
        ax_k.text(0.5, 0.25, lbl, ha="center", va="center",
                  fontsize=7.5, color=C_LIGHT, transform=ax_k.transAxes,
                  multialignment="center")


# ===========================================================================
# HALAMAN 4: PREPROCESSING
# ===========================================================================
def page_preprocessing(fig):
    setup_page(fig,
               "Langkah 1: Persiapan Data (Preprocessing)",
               "Sebelum analisis, data perlu dibersihkan dan diseragamkan terlebih dahulu",
               step="Langkah 1 dari 3")

    steps = [
        ("1", C_BLUE,
         "Mengubah Nilai Tukar ke Format yang Adil",
         "Masalah: Rupiah bernilai belasan ribu per USD, sementara Euro sekitar 1 per USD. "
         "Jika langsung dipakai, komputer akan menganggap Rupiah 'lebih ekstrem' dari Euro, "
         "padahal keduanya sama-sama stabil.\n\n"
         "Solusi: Ubah dari nilai nominal ke '% perubahan per tahun'. "
         "Depresiasi 50% = tanda bahaya, tidak peduli mata uangnya Rupiah atau Peso.",
         "Exchange_Rate (nilai nominal) diubah ke\nExchange_Depreciation (% perubahan/tahun)\n"
         "Contoh: IDR 1998 = melemah 83% -- sangat ekstrem!"),

        ("2", C_GOLD,
         "Mengisi Data yang Kosong (Imputasi)",
         "Masalah: Ada ribuan sel yang kosong. DBSCAN tidak bisa bekerja dengan data kosong.\n\n"
         "Solusi (2 tahap):\n"
         "Tahap 1: Isi dengan median data negara yang sama. Lebih masuk akal karena kondisi "
         "ekonomi suatu negara cenderung mirip antar tahun.\n"
         "Tahap 2: Jika masih kosong (misal negara tidak pernah punya data), isi dengan "
         "median global sebagai 'nilai tengah' semua negara.",
         "Sebelum: 1.553 sel kosong di seluruh dataset\nSesudah: 0 sel kosong (100% terisi)"),

        ("3", C_GREEN,
         "Menyamakan Skala Semua Indikator (Standarisasi)",
         "Masalah: Inflasi berkisar 0-10.000%, GDP Growth hanya -15% sampai +20%. "
         "Jika digabung langsung, inflasi akan mendominasi perhitungan jarak dan "
         "indikator lain jadi tidak berarti.\n\n"
         "Solusi: Z-score standarisasi -- ubah setiap nilai ke 'berapa standar deviasi "
         "dari rata-rata'. Setelah ini semua indikator berkontribusi SETARA.",
         "Semua indikator punya rata-rata = 0\ndan standar deviasi = 1 setelah standarisasi"),

        ("4", C_RED,
         "Memadatkan 14 Indikator menjadi Komponen Utama (PCA)",
         "Masalah: Di ruang 14 dimensi, semua titik cenderung 'sama-sama jauh' satu sama "
         "lain (curse of dimensionality), sehingga DBSCAN sulit membedakan normal vs anomali.\n\n"
         "Solusi: PCA -- padatkan 14 indikator menjadi beberapa 'ringkasan' utama yang "
         "menangkap informasi terpenting. Seperti meringkas 14 paragraf jadi 3-4 poin utama.",
         "14 indikator -> 9 komponen utama\nInformasi tersimpan: 92.6% dari total varians\n"
         "(Hanya untuk DBSCAN Global; Per-Negara pakai 14 fitur asli)"),
    ]

    for i, (num, color, title, penj, hasil) in enumerate(steps):
        col = i % 2
        row = i // 2
        bx = 0.04 + col * 0.49
        by = 0.53 - row * 0.44

        ax = fig.add_axes([bx, by, 0.45, 0.39])
        ax.set_facecolor(C_WHITE)
        ax.axis("off")
        ax.add_patch(plt.Circle((0.04, 0.91), 0.055, color=color,
                                 transform=ax.transAxes, zorder=5, clip_on=False))
        ax.text(0.04, 0.91, num, ha="center", va="center",
                fontsize=12, fontweight="bold", color=C_WHITE,
                transform=ax.transAxes, zorder=6)
        ax.text(0.11, 0.94, title, va="top", fontsize=9.5,
                fontweight="bold", color=color, transform=ax.transAxes)
        lines = penj.split("\n")
        y_pos = 0.81
        for line in lines:
            wrapped = textwrap.wrap(line, 62) or [""]
            for wl in wrapped:
                ax.text(0.03, y_pos, wl, va="top", fontsize=8.2,
                        color="#343a40", transform=ax.transAxes)
                y_pos -= 0.10
                if y_pos < 0.23:
                    break
        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.19,
                                    facecolor=color+"22", edgecolor=color,
                                    lw=1.2, transform=ax.transAxes))
        ax.text(0.04, 0.18, "Hasil:", va="top", fontsize=8,
                fontweight="bold", color=color, transform=ax.transAxes)
        yh = 0.13
        for lh in hasil.split("\n"):
            ax.text(0.04, yh, lh, va="top", fontsize=7.8,
                    color="#343a40", transform=ax.transAxes)
            yh -= 0.066

    callout(fig, 0.04, 0.01, 0.92, 0.07,
            icon="",
            text="Setelah 4 langkah ini, data siap dianalisis. Untuk DBSCAN Global: pakai 9 komponen PCA. "
                 "Untuk DBSCAN Per-Negara: pakai 14 indikator asli (karena tiap negara dianalisis terpisah, "
                 "dimensi tinggi tidak menjadi masalah besar).",
            bg="#edf2fb", border=C_DARK)


# ===========================================================================
# HALAMAN 5: DBSCAN GLOBAL
# ===========================================================================
def page_dbscan_global(fig, hasil_g, mg, gt):
    setup_page(fig,
               "Langkah 2A: DBSCAN Global",
               "Semua 49 negara dianalisis sekaligus: siapa yang paling 'aneh' di dunia?",
               step="Langkah 2A dari 3")

    textbox(fig, 0.04, 0.74, 0.44, 0.18,
            title="Apa yang Dilakukan?",
            text=(
                "Semua 1.715 data (49 negara x 35 tahun) digabung jadi satu 'peta' besar. "
                "DBSCAN mencari kondisi ekonomi yang begitu ekstrem hingga tidak masuk "
                "kelompok manapun secara global. Titik yang 'sendirian' = anomali global."
            ),
            border=C_BLUE, bg=C_BOX, fontsize=9)

    textbox(fig, 0.52, 0.74, 0.44, 0.18,
            title="Cara Menentukan eps (Radius Pencarian)?",
            text=(
                "Kami pakai metode Kneedle: plot jarak setiap titik ke tetangga terdekatnya, "
                "lalu cari titik 'siku' kurva tersebut secara matematis. "
                "Titik siku = batas alami antara titik yang 'dalam klaster' vs yang 'terlalu jauh'.\n"
                "eps = 3.16 dipilih otomatis."
            ),
            border=C_GOLD, bg="#fff8ee", fontsize=9)

    results = [("13", "Anomali\nDitemukan", C_RED),
               ("0.76%", "dari 1.715\nData", C_GOLD),
               ("8 dari 13", "Sesuai\nGT IMF", C_GREEN),
               ("61.5%", "Precision\n(Ketepatan)", C_BLUE)]
    for i, (val, lbl, color) in enumerate(results):
        ax = fig.add_axes([0.04 + i*0.235, 0.60, 0.21, 0.12])
        ax.set_facecolor(color); ax.axis("off")
        ax.text(0.5, 0.65, val, ha="center", va="center",
                fontsize=16, fontweight="bold", color=C_WHITE, transform=ax.transAxes)
        ax.text(0.5, 0.20, lbl, ha="center", va="center",
                fontsize=7.5, color=C_WHITE, transform=ax.transAxes, multialignment="center")

    ax_tbl = fig.add_axes([0.04, 0.22, 0.54, 0.36])
    ax_tbl.axis("off")
    ax_tbl.set_title("13 Anomali yang Ditemukan DBSCAN Global",
                     fontsize=9.5, fontweight="bold", color=C_DARK, pad=6)
    tbl_data = []
    for _, row in (mg[mg["predicted_anomaly"]==1]
                   .sort_values("anomaly_score", ascending=False).iterrows()):
        krisis = str(row["crisis_name"]) if pd.notna(row["crisis_name"]) else "-"
        sesuai = "Ya" if row["is_crisis"] == 1 else "Tidak"
        ket = (krisis[:28] if krisis != "-" else "Tidak tercatat di GT IMF")
        tbl_data.append([
            f"{row['economy']} ({int(row['year'])})",
            f"{row['anomaly_score']:.3f}", sesuai, ket])
    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Negara (Tahun)", "Score", "GT IMF?", "Keterangan"],
        cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and tbl_data[r-1][2] == "Ya":
            cell.set_facecolor("#d8f3dc")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#dee2e6")

    textbox(fig, 0.62, 0.22, 0.35, 0.36,
            title="Apa Artinya Hasil Ini?",
            text=(
                "DBSCAN Global sangat SELEKTIF.\n\n"
                "Ia hanya menandai 13 dari 1.715 data (0.76%) "
                "sebagai anomali. Ini karena 'standar keanehan' "
                "berdasarkan semua negara sekaligus sangat tinggi.\n\n"
                "Yang terdeteksi adalah kondisi yang benar-benar "
                "EKSTREM secara dunia:\n\n"
                "  Brazil 90-94 = inflasi ribuan persen\n"
                "  Peru 1990 = inflasi 7.000%+\n"
                "  Argentina 2002 = krisis peso\n"
                "  Indonesia 1998 = krisis Asia\n\n"
                "8 dari 13 memang tercatat krisis oleh IMF.\n"
                "Sisanya mungkin krisis yang belum terdokumentasi."
            ),
            border=C_GREEN, bg="#edfaf7", fontsize=8.8)

    callout(fig, 0.04, 0.08, 0.92, 0.12,
            icon="",
            text="DBSCAN Global bagus untuk krisis spektakuler skala dunia (hiperinflasi, "
                 "keruntuhan mata uang), tapi 'melewatkan' krisis yang hanya terasa ekstrem "
                 "di negara itu sendiri. Itulah kenapa kita juga butuh pendekatan Per-Negara.",
            bg="#fff3cd", border=C_GOLD)


# ===========================================================================
# HALAMAN 6: DBSCAN PER NEGARA
# ===========================================================================
def page_dbscan_percountry(fig, hasil_p, gt):
    setup_page(fig,
               "Langkah 2B: DBSCAN Per-Negara",
               "Tiap negara dianalisis sendiri: 'Apakah tahun ini normal UNTUK negara ini?'",
               step="Langkah 2B dari 3")

    textbox(fig, 0.04, 0.80, 0.54, 0.13,
            title="Kenapa Perlu Pendekatan Per-Negara?",
            text=(
                "Inflasi 20% mungkin 'normal' untuk Nigeria di era 90an, tapi sangat ekstrem "
                "untuk Jepang atau Swiss. Dengan menganalisis tiap negara secara terpisah, "
                "kita menemukan tahun yang 'tidak normal BAGI NEGARA ITU SENDIRI', "
                "bukan berdasarkan standar dunia yang terlalu ketat."
            ),
            border=C_BLUE, bg=C_BOX, fontsize=9)

    textbox(fig, 0.62, 0.80, 0.35, 0.13,
            title="Perbedaan vs Global",
            text=(
                "Global:     'Siapa paling aneh di dunia?'\n"
                "Per-Negara: 'Tahun mana yang paling aneh\n"
                "             dalam sejarah negara INI?'\n\n"
                "Keduanya saling melengkapi."
            ),
            border=C_GOLD, bg="#fff8ee", fontsize=9)

    ax_hm = fig.add_axes([0.04, 0.25, 0.92, 0.53])
    pivot = hasil_p.pivot_table(
        index="economy", columns="year",
        values="predicted_anomaly", aggfunc="max").fillna(0)
    cmap = LinearSegmentedColormap.from_list("c", ["#f1faee", C_RED], N=2)
    sns.heatmap(pivot, cmap=cmap, linewidths=0.2, linecolor="#ddd",
                ax=ax_hm, cbar=False, vmin=0, vmax=1)
    years = list(pivot.columns)
    for yr, lbl in {1997: "Asian FC\n1997-98", 2008: "GFC\n2008-09",
                    2020: "COVID-19\n2020"}.items():
        if yr in years:
            x = years.index(yr) + 0.5
            ax_hm.axvline(x, color=C_DARK, lw=1.5, ls="--", alpha=0.7)
            ax_hm.text(x + 0.2, -1.5, lbl, color=C_DARK, fontsize=7,
                       fontweight="bold", va="top", multialignment="center")
    ax_hm.set_title(
        "Peta Anomali: Merah = Tahun yang Dianggap Anomali oleh DBSCAN Per-Negara "
        "(total 189 anomali)",
        fontsize=9.5, fontweight="bold", color=C_DARK, pad=5)
    ax_hm.set_xlabel("Tahun", fontsize=9)
    ax_hm.set_ylabel("Negara", fontsize=9)
    ax_hm.tick_params(axis="x", labelsize=6.5, rotation=45)
    ax_hm.tick_params(axis="y", labelsize=7)
    p1 = mpatches.Patch(color=C_RED,    label="Anomali")
    p2 = mpatches.Patch(color="#f1faee",label="Normal", edgecolor="gray", lw=0.5)
    ax_hm.legend(handles=[p1, p2], loc="upper left", fontsize=8)

    callout(fig, 0.04, 0.11, 0.44, 0.12,
            icon="",
            text="COVID-19 2020: Terdeteksi di 31 dari 49 negara! Ini adalah deteksi "
                 "terbaik -- hampir semua indikator ekonomi jatuh serentak di 2020, "
                 "sehingga mudah dikenali sebagai 'berbeda dari pola historis'.",
            bg="#fff3cd", border=C_GOLD)

    callout(fig, 0.52, 0.11, 0.44, 0.12,
            icon="",
            text="Total 189 anomali (11% dari data). Negara paling banyak anomali: "
                 "Rusia (14), Yunani (12), Nigeria (10). "
                 "Negara tanpa anomali sama sekali: Kanada, Meksiko, Jepang, Chile, Saudi Arabia.",
            bg=C_BOX, border=C_BLUE)


# ===========================================================================
# HALAMAN 7: EVALUASI
# ===========================================================================
def page_evaluasi(fig, eval_r, mp, mg):
    setup_page(fig,
               "Langkah 3: Evaluasi -- Apakah Hasilnya Benar?",
               "Membandingkan hasil DBSCAN dengan daftar krisis historis dari IMF",
               step="Langkah 3 dari 3")

    textbox(fig, 0.04, 0.75, 0.54, 0.17,
            title="Bagaimana Cara Mengevaluasi?",
            text=(
                "Kami punya 'kunci jawaban': daftar krisis historis dari IMF yang mencakup "
                "krisis perbankan, mata uang, dan utang untuk setiap negara. "
                "Total 229 tahun-negara tercatat sebagai krisis.\n\n"
                "Kita bandingkan: dari semua yang DBSCAN tandai sebagai anomali, "
                "berapa yang memang benar krisis menurut IMF?"
            ),
            border=C_BLUE, bg=C_BOX, fontsize=9)

    textbox(fig, 0.62, 0.75, 0.35, 0.17,
            title="4 Kemungkinan Hasil",
            text=(
                "TP: Prediksi anomali, memang krisis. (BENAR)\n\n"
                "FP: Prediksi anomali, ternyata bukan krisis.\n"
                "    (Terlalu sensitif / false alarm)\n\n"
                "FN: Prediksi normal, ternyata krisis.\n"
                "    (Kecolongan / tidak terdeteksi)\n\n"
                "TN: Prediksi normal, memang normal. (BENAR)"
            ),
            border=C_GOLD, bg="#fff8ee", fontsize=8.8)

    row_g = eval_r[eval_r["metode"] == "GLOBAL"].iloc[0]
    row_p = eval_r[eval_r["metode"] == "PER-COUNTRY"].iloc[0]

    for i, (row, lbl, color) in enumerate([
        (row_g, "DBSCAN Global", C_GOLD),
        (row_p, "DBSCAN Per-Negara", C_RED),
    ]):
        ax_cm = fig.add_axes([0.04 + i*0.26, 0.42, 0.22, 0.31])
        cm = np.array([[int(row["TN"]), int(row["FP"])],
                        [int(row["FN"]), int(row["TP"])]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Pred: Normal", "Pred: Anomali"],
                    yticklabels=["GT: Normal", "GT: Krisis"],
                    linewidths=1, annot_kws={"size": 11, "weight": "bold"},
                    cbar=False)
        ax_cm.set_title(lbl, fontsize=9, fontweight="bold", color=color)
        ax_cm.tick_params(axis="x", labelsize=7.5, rotation=20)
        ax_cm.tick_params(axis="y", labelsize=7.5, rotation=0)

    ax_met = fig.add_axes([0.56, 0.42, 0.40, 0.31])
    ax_met.axis("off")
    met_data = [
        ["Accuracy",  f"{row_g['accuracy']:.1%}", f"{row_p['accuracy']:.1%}",
         "% prediksi yang benar keseluruhan"],
        ["Precision", f"{row_g['precision']:.1%}", f"{row_p['precision']:.1%}",
         "Dari prediksi anomali, berapa % benar?"],
        ["Recall",    f"{row_g['recall']:.1%}",    f"{row_p['recall']:.1%}",
         "Dari semua krisis nyata, berapa % terdeteksi?"],
        ["F1-Score",  f"{row_g['f1']:.3f}",        f"{row_p['f1']:.3f}",
         "Gabungan Precision & Recall"],
        ["ROC-AUC",   f"{row_g['roc_auc']:.3f}",   f"{row_p['roc_auc']:.3f}",
         "Kemampuan bedakan normal vs krisis (0.5=acak)"],
    ]
    tbl = ax_met.table(
        cellText=met_data,
        colLabels=["Metrik", "Global", "Per-Negara", "Artinya (Sederhana)"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#dee2e6")
        if c == 3:
            cell.set_text_props(color="#495057", style="italic")
    ax_met.set_title("Perbandingan Metrik Evaluasi",
                     fontsize=9, fontweight="bold", color=C_DARK, pad=6)

    textbox(fig, 0.04, 0.14, 0.44, 0.26,
            title="Membaca Hasil Global: Sangat Selektif",
            text=(
                "Recall hanya 3.5% -- dari 229 krisis nyata, DBSCAN Global hanya "
                "mendeteksi 8 (sisanya kecolongan). Standar 'global' terlalu tinggi.\n\n"
                "Tapi Precision 61.5% cukup baik -- dari 13 yang ditandai anomali, "
                "8 memang benar krisis. Jadi: jarang salah, tapi sering melewatkan.\n\n"
                "Cocok untuk: deteksi krisis yang benar-benar spektakuler."
            ),
            border=C_GOLD, bg="#fff8ee", fontsize=9)

    textbox(fig, 0.52, 0.14, 0.44, 0.26,
            title="Membaca Hasil Per-Negara: Lebih Seimbang",
            text=(
                "Recall 32.3% -- sekitar 1 dari 3 krisis historis berhasil ditemukan, "
                "padahal DBSCAN tidak pernah diberitahu contoh krisis sebelumnya!\n\n"
                "F1 = 0.354 dan ROC-AUC = 0.638. Sebagai perbandingan, model acak "
                "punya AUC = 0.5. Jadi model kita 14% lebih baik dari tebak-tebakan.\n\n"
                "Ini adalah kinerja yang cukup baik untuk unsupervised learning."
            ),
            border=C_GREEN, bg="#edfaf7", fontsize=9)

    callout(fig, 0.04, 0.04, 0.92, 0.09,
            icon="",
            text="Penting untuk diingat: DBSCAN adalah metode UNSUPERVISED -- ia tidak pernah belajar "
                 "dari contoh krisis. Ia hanya melihat pola kepadatan data. Mendeteksi 1 dari 3 krisis "
                 "historis secara otomatis adalah hasil yang cukup menjanjikan!",
            bg="#edf2fb", border=C_DARK)


# ===========================================================================
# HALAMAN 8: EVALUASI LANJUTAN (KUALITAS KLASTER & JENIS KRISIS)
# ===========================================================================
def page_evaluasi_lanjutan(fig, eval_cq, eval_ct, mg, mp):
    setup_page(fig,
               "Evaluasi Lanjutan: Kualitas Klaster & Karakteristik Krisis",
               "Metrik Unsupervised (Silhouette, DB, CH), Analisis Kurva PR, dan Ketepatan per Jenis Krisis",
               step="LANGKAH 3B")

    # 1. KIRI ATAS: METRIK UNSUPERVISED (KUALITAS KLASTER)
    ax_cq = fig.add_axes([0.04, 0.58, 0.44, 0.17])
    ax_cq.axis("off")
    ax_cq.set_title("Metrik Intrinsik Kualitas Klaster (Unsupervised)",
                    fontsize=9.5, fontweight="bold", color=C_DARK, pad=6)

    if eval_cq is not None:
        cq_data = []
        for _, r in eval_cq.iterrows():
            m = str(r["metode"])
            s = f"{float(r['silhouette']):.3f}"
            db = f"{float(r['davies_bouldin']):.3f}"
            ch = f"{float(r['calinski_harabasz']):.1f}"
            cq_data.append([m, s, db, ch])
    else:
        cq_data = [
            ["GLOBAL (PCA)", "0.777", "1.578", "129.8"],
            ["PER-COUNTRY (Scaled)", "0.273", "5.462", "21.7"]
        ]

    cq_headers = ["Pendekatan", "Silhouette (+1)", "Davies-Bouldin (0)", "Calinski-Harabasz (max)"]
    tbl_cq = ax_cq.table(
        cellText=cq_data,
        colLabels=cq_headers,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 0.88])
    tbl_cq.auto_set_font_size(False)
    tbl_cq.set_fontsize(8)
    for (r, c), cell in tbl_cq.get_celld().items():
        if r == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#fff8ee")
        else:
            cell.set_facecolor("#edfaf7")
        cell.set_edgecolor("#dee2e6")

    textbox(fig, 0.04, 0.16, 0.44, 0.39,
            title="Mengapa Nilai Silhouette Global Lebih Tinggi?",
            text=(
                "Silhouette Global = 0.777 (Sangat Baik), sedangkan Per-Negara = 0.273.\n"
                "Apakah ini berarti pendekatan Global lebih unggul?\n\n"
                "Jawabannya: TIDAK untuk tujuan deteksi anomali krisis!\n\n"
                "DBSCAN Global menempatkan 99.2% seluruh data ke dalam 1 klaster raksasa homogen "
                "('kondisi ekonomi normal dunia') dan hanya menyisihkan 13 titik paling ekstrem. Klaster tunggal "
                "yang masif dan seragam ini secara geometris sangat padat sehingga skor Silhouette-nya tinggi. "
                "Namun ia 'buta' terhadap 96.5% krisis riil (Recall hanya 3.5%).\n\n"
                "Sebaliknya, DBSCAN Per-Negara memetakan sejarah internal masing-masing negara secara dinamis, "
                "sehingga sebaran titiknya lebih heterogen (Silhouette 0.273), tetapi terbukti 9x lebih peka "
                "menangkap krisis nyata!"
            ),
            border=C_BLUE, bg=C_BOX, fontsize=8.2)

    # 2. KANAN ATAS: PRECISION-RECALL CURVE
    ax_pr = fig.add_axes([0.52, 0.53, 0.44, 0.33])
    p_g, r_g, _ = precision_recall_curve(mg["is_crisis"], mg["anomaly_score"])
    p_p, r_p, _ = precision_recall_curve(mp["is_crisis"], mp["anomaly_score"])
    ap_g = average_precision_score(mg["is_crisis"], mg["anomaly_score"])
    ap_p = average_precision_score(mp["is_crisis"], mp["anomaly_score"])
    baseline = mg["is_crisis"].mean()

    ax_pr.plot(r_p, p_p, color=C_RED, lw=2.2, label=f"Per-Negara (AP = {ap_p:.3f})")
    ax_pr.plot(r_g, p_g, color=C_GOLD, lw=2.2, label=f"Global (AP = {ap_g:.3f})")
    ax_pr.axhline(baseline, color="gray", ls="--", lw=1.2, label=f"Baseline Acak ({baseline:.1%})")
    ax_pr.set_xlim(0, 1.02); ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall (Proporsi Krisis Terdeteksi)", fontsize=8.5)
    ax_pr.set_ylabel("Precision (Akurasi Prediksi)", fontsize=8.5)
    ax_pr.set_title("Precision-Recall Curve (PR-AUC)", fontsize=9.5, fontweight="bold", color=C_DARK)
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.grid(alpha=0.3, ls=":")
    ax_pr.set_facecolor("#ffffff")

    # 3. KANAN BAWAH: BREAKDOWN PER JENIS KRISIS
    ax_ct = fig.add_axes([0.52, 0.16, 0.44, 0.31])
    krisis_names = ["COVID-19 (2020)", "Currency Crisis", "Sovereign Debt", "Banking Crisis"]
    rec_vals     = [63.3, 54.3, 43.8, 23.4]
    colors_ct    = [C_GREEN, C_RED, C_BLUE, C_GOLD]

    bars = ax_ct.barh(krisis_names[::-1], rec_vals[::-1], color=colors_ct[::-1],
                      edgecolor="white", height=0.55)
    for bar, val in zip(bars, rec_vals[::-1]):
        ax_ct.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                   va="center", fontsize=8.5, fontweight="bold", color=C_DARK)
    ax_ct.set_xlim(0, 80)
    ax_ct.set_xlabel("Recall (% Kejadian yang Berhasil Terdeteksi)", fontsize=8.5)
    ax_ct.set_title("Daya Tangkap DBSCAN Per-Negara per Jenis Krisis", fontsize=9.5, fontweight="bold", color=C_DARK)
    ax_ct.grid(axis="x", alpha=0.3, ls=":")
    ax_ct.set_facecolor("#ffffff")

    # Callout Bawah
    callout(fig, 0.04, 0.04, 0.92, 0.09,
            icon="",
            text="Wawasan Strategis: DBSCAN paling sensitif terhadap Krisis Nilai Tukar (Recall 54.3%) dan Syok Pandemi COVID-19 (63.3%) "
                 "karena guncangan langsung menyerang agregat makro (cadangan devisa & GDP). Sebaliknya, Krisis Perbankan (23.4%) lebih menantang "
                 "karena bermula dari neraca internal bank yang lambat merembet ke data makroekonomi agregat.",
            bg="#edf2fb", border=C_DARK)


# ===========================================================================
# HALAMAN 9: TEMUAN UTAMA
# ===========================================================================
def page_temuan(fig, mg, mp, gt):
    setup_page(fig,
               "Krisis-Krisis yang Berhasil Ditemukan",
               "Temuan paling menarik dan konteks historisnya")

    temuan = [
        (C_RED, "Hiperinflasi Amerika Latin (1990-1994)",
         "Brazil, Peru, Argentina",
         "Inflasi di Brazil awal 1990-an mencapai ribuan persen per tahun "
         "(pernah 2.700% dalam setahun!). Peru bahkan pernah 7.000%. "
         "Kondisi ini begitu ekstrem sehingga langsung terdeteksi oleh "
         "DBSCAN Global sebagai anomali paling jauh dari 'kondisi normal dunia'.",
         "BRA 1993: anomaly score = 0.67\nBRA 1994: anomaly score = 0.54\n"
         "PER 1990: anomaly score = 1.00 (TERTINGGI di seluruh dataset)"),

        (C_GOLD, "Krisis Keuangan Asia 1997-1998",
         "Indonesia, Thailand, Korea, Malaysia",
         "Nilai tukar Rupiah jatuh 83% dalam setahun. Cadangan devisa menipis drastis. "
         "Pengangguran melonjak. Kombinasi guncangan di banyak indikator sekaligus "
         "membuat 1998 menjadi tahun paling berbeda dalam sejarah ekonomi Indonesia.",
         "IDN 1998: terdeteksi di Global DAN Per-Negara\n"
         "KOR 1998: terdeteksi di Per-Negara\n"
         "8 titik anomali terdeteksi di kawasan Asia untuk periode ini"),

        (C_BLUE, "Krisis Utang Yunani (2010-2015)",
         "Yunani (GRC)",
         "Yunani mengalami krisis utang terparah dalam sejarah Eropa modern. "
         "GDP terkontraksi 25%, pengangguran mencapai 27%. "
         "DBSCAN Per-Negara berhasil mendeteksi sebagian besar periode krisis ini.",
         "Yunani (GRC): F1 = 0.556, Recall = 83%\n"
         "5 dari 6 tahun krisis Yunani berhasil diidentifikasi\n"
         "Salah satu negara dengan performa deteksi terbaik"),

        (C_GREEN, "Pandemi COVID-19 (2020)",
         "31 dari 49 negara",
         "COVID-19 adalah guncangan ekonomi paling merata dalam sejarah modern. "
         "GDP jatuh, perdagangan mandek, investasi lari -- serentak di hampir semua negara. "
         "DBSCAN Per-Negara mendeteksinya di 31 dari 49 negara karena 2020 berbeda "
         "drastis dari pola historis SETIAP negara.",
         "31 dari 49 negara terdeteksi sebagai anomali di tahun 2020\n"
         "Ini adalah deteksi terbaik dari seluruh analisis\n"
         "Bukti bahwa COVID memang guncangan yang 'berbeda dari biasanya'"),
    ]

    for i, (color, judul, negara, cerita, hasil) in enumerate(temuan):
        col = i % 2
        row = i // 2
        bx = 0.04 + col * 0.49
        by = 0.50 - row * 0.44

        ax = fig.add_axes([bx, by, 0.45, 0.43])
        ax.set_facecolor(C_WHITE)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06,
                                    facecolor=color, transform=ax.transAxes))
        ax.text(0.03, 0.97, judul, va="center", fontsize=9,
                fontweight="bold", color=C_WHITE, transform=ax.transAxes)
        ax.text(0.03, 0.90, f"Negara/Kawasan: {negara}", va="top",
                fontsize=8.5, color=color, fontweight="bold",
                style="italic", transform=ax.transAxes)
        y_pos = 0.83
        for line in cerita.split("\n"):
            wrapped = textwrap.wrap(line, 62) or [""]
            for wl in wrapped:
                ax.text(0.03, y_pos, wl, va="top", fontsize=8.2,
                        color="#343a40", transform=ax.transAxes)
                y_pos -= 0.093
                if y_pos < 0.22:
                    break
        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.19,
                                    facecolor=color + "22", edgecolor=color,
                                    lw=1.2, transform=ax.transAxes))
        ax.text(0.04, 0.18, "Hasil Deteksi:", va="top", fontsize=7.8,
                fontweight="bold", color=color, transform=ax.transAxes)
        yh = 0.12
        for lh in hasil.split("\n"):
            ax.text(0.04, yh, lh, va="top", fontsize=7.8,
                    color="#343a40", transform=ax.transAxes)
            yh -= 0.062


# ===========================================================================
# HALAMAN 9: KESIMPULAN
# ===========================================================================
def page_kesimpulan(fig):
    fig.patch.set_facecolor(C_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_DARK); ax.axis("off")
    ax.axhline(0.93, xmin=0.05, xmax=0.95, color=C_RED,  lw=4)
    ax.axhline(0.90, xmin=0.05, xmax=0.95, color=C_GOLD, lw=1.5)
    fig.text(0.5, 0.86, "Kesimpulan",
             ha="center", fontsize=22, fontweight="bold", color=C_WHITE)
    fig.text(0.5, 0.81,
             "Apa yang berhasil kita capai dengan proyek deteksi anomali ini?",
             ha="center", fontsize=11, color=C_LIGHT, style="italic")

    poin = [
        (C_GOLD,  "Komputer bisa mendeteksi krisis ekonomi TANPA diberitahu contoh krisis",
                  "Dengan DBSCAN (Unsupervised Learning), model menemukan sendiri kondisi yang "
                  "'berbeda dari biasanya' hanya dari pola data, tanpa perlu label training."),
        (C_RED,   "1 dari 3 krisis historis berhasil ditemukan (pendekatan Per-Negara)",
                  "Recall 32.3% dan ROC-AUC 0.638 adalah kinerja yang cukup baik "
                  "untuk metode yang sama sekali tidak pernah melihat contoh krisis sebelumnya."),
        (C_LIGHT, "COVID-19 2020 paling mudah terdeteksi -- 31 dari 49 negara",
                  "Guncangan yang merata dan serentak di semua indikator membuat 2020 "
                  "menjadi tahun paling 'aneh' dalam sejarah hampir semua negara."),
        (C_GREEN, "Pendekatan Per-Negara lebih baik dari Global untuk tujuan deteksi krisis",
                  "Dengan menganalisis tiap negara dalam konteks sejarahnya sendiri, "
                  "model lebih mampu mendeteksi krisis yang relatif terhadap kondisi normal negara itu."),
    ]

    y = 0.76
    for color, judul, isi in poin:
        ax.axhline(y + 0.005, xmin=0.04, xmax=0.96, color=color, lw=0.8, alpha=0.3)
        ax.add_patch(plt.Rectangle((0.04, y - 0.072), 0.012, 0.068,
                                    facecolor=color, transform=ax.transAxes))
        fig.text(0.065, y, judul, fontsize=11, fontweight="bold",
                 color=color, va="top")
        fig.text(0.065, y - 0.030, isi, fontsize=9.5,
                 color=C_WHITE, va="top", multialignment="left")
        y -= 0.175

    ax.axhline(0.10, xmin=0.05, xmax=0.95, color=C_GOLD, lw=1.5)
    ax.axhline(0.07, xmin=0.05, xmax=0.95, color=C_RED,  lw=3)
    fig.text(0.5, 0.055,
             "Pendekatan Per-Negara lebih disarankan jika ingin analisis yang sensitif terhadap konteks lokal.\n"
             "Pendekatan Global lebih tepat untuk mendeteksi krisis berskala dan berdampak dunia.",
             ha="center", fontsize=10, color="#a8dadc", multialignment="center")
    fig.text(0.5, 0.018,
             f"Laporan dibuat: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
             "Unsupervised Learning - Data Mining",
             ha="center", fontsize=8.5, color="#adb5bd")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 65)
    print("  MEMBUAT LAPORAN PDF (Versi Naratif 10 Halaman)")
    print("=" * 65)

    gt, hasil_g, hasil_p, eval_r, eval_e, mg, mp, eval_cq, eval_ct = load_all()

    with PdfPages(PDF_OUT) as pdf:
        pages = [
            ("[1/10] Cover...",                   lambda f: page_cover(f)),
            ("[2/10] Konsep DBSCAN...",           lambda f: page_konsep(f)),
            ("[3/10] Data...",                    lambda f: page_data(f, gt)),
            ("[4/10] Preprocessing...",           lambda f: page_preprocessing(f)),
            ("[5/10] DBSCAN Global...",           lambda f: page_dbscan_global(f, hasil_g, mg, gt)),
            ("[6/10] DBSCAN Per-Negara...",       lambda f: page_dbscan_percountry(f, hasil_p, gt)),
            ("[7/10] Evaluasi vs Ground Truth...", lambda f: page_evaluasi(f, eval_r, mp, mg)),
            ("[8/10] Evaluasi Lanjutan...",       lambda f: page_evaluasi_lanjutan(f, eval_cq, eval_ct, mg, mp)),
            ("[9/10] Temuan Utama...",            lambda f: page_temuan(f, mg, mp, gt)),
            ("[10/10] Kesimpulan...",             lambda f: page_kesimpulan(f)),
        ]
        for label, fn in pages:
            print(label)
            fig = plt.figure(figsize=(13.0, 9.2))
            fn(fig)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        d = pdf.infodict()
        d["Title"]   = "Laporan DBSCAN Anomaly Detection Makroekonomi"
        d["Subject"] = "Unsupervised Learning - Deteksi Krisis 49 Negara 1990-2024"
        d["CreationDate"] = datetime.now()

    print()
    print("=" * 65)
    print("  PDF BERHASIL DIBUAT!")
    print("=" * 65)
    print(f"\n  File : {PDF_OUT}")
    print("  10 Halaman - Format A4+ Landscape")
    print()
    for i, lbl in enumerate([
        "Cover", "Apa itu DBSCAN? (konsep + analogi)",
        "Data yang Digunakan", "Langkah 1: Preprocessing (kenapa & hasilnya)",
        "Langkah 2A: DBSCAN Global", "Langkah 2B: DBSCAN Per-Negara",
        "Evaluasi vs IMF Ground Truth", "Evaluasi Lanjutan (Klaster & Jenis Krisis)",
        "Temuan Utama & Konteks Historis", "Kesimpulan Akhir"
    ], 1):
        print(f"  Hal {i:<2} - {lbl}")


if __name__ == "__main__":
    main()
