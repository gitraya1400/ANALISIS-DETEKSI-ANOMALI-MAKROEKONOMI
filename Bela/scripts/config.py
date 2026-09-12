"""
config.py
=========
Konfigurasi bersama untuk seluruh pipeline PCA-based Anomaly Detection.
Kelompok 4 - Analisis Deteksi Anomali pada Indikator Makroekonomi Global
sebagai Instrumen Early Warning System (EWS) Krisis Ekonomi.

Semua path dibuat relatif terhadap root folder proyek, sehingga script
bisa dijalankan dari mana saja selama struktur folder tidak diubah.
"""

import os

# ---------------------------------------------------------------------------
# PATH PROYEK
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
REPORT_DIR = os.path.join(ROOT_DIR, "report")

RAW_CSV = os.path.join(DATA_DIR, "data_cleaned.csv")

for _d in (RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# VARIABEL
# ---------------------------------------------------------------------------
# 14 variabel numerik yang digunakan sebagai fitur (identik dengan yang
# dipakai algoritma lain di kelompok -- economy, year, dan crisis_label
# TIDAK diikutsertakan sebagai fitur, hanya sebagai identifier/label).
ID_COLS = ["economy", "year"]
LABEL_COL = "crisis_label"

FEATURE_COLS = [
    "GDP_Growth",
    "GDP_PerCapita_Growth",
    "Inflation_CPI",
    "Total_Reserves",
    "Unemployment",
    "Current_Account_GDP",
    "Trade_GDP",
    "FDI_Inflows_GDP",
    "Exports_GDP",
    "Imports_GDP",
    "Gross_Savings_GDP",
    "Exchange_Rate",
    "Manufacturing_Value",
    "Investment_GDP",
]

# ---------------------------------------------------------------------------
# PARAMETER PCA
# ---------------------------------------------------------------------------
# Ambang batas cumulative explained variance untuk menentukan jumlah
# komponen utama (k) yang dipertahankan -- sesuai proposal Bab 3.2.6 (>=95%)
EXPLAINED_VARIANCE_THRESHOLD = 0.95

# Tingkat signifikansi untuk UCL (Upper Control Limit) SPE (chi-square) dan
# T^2 (F-distribution). alpha = 0.05 -> UCL pada persentil ke-95.
ALPHA = 0.05

RANDOM_STATE = 42
