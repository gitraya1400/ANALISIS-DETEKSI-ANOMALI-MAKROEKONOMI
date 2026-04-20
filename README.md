<p align="center">
  <h1 align="center">🌐 Deteksi Anomali pada Indikator Ekonomi Makro</h1>
  <h3 align="center"><em>Early Warning System</em> Krisis Ekonomi menggunakan Pendekatan <em>Unsupervised Learning</em></h3>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Data-World%20Bank%20API-00457C?logo=databricks&logoColor=white" alt="World Bank">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

<p align="center">
  <strong>UTS Data Mining — Semester 6 (2026)</strong>
</p>

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Latar Belakang](#-latar-belakang)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi & Setup](#-instalasi--setup)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil & Temuan](#-hasil--temuan)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Referensi](#-referensi)
- [Lisensi](#-lisensi)

---

## 🎯 Tentang Proyek

Proyek ini membangun **Early Warning System (EWS)** untuk deteksi dini krisis ekonomi menggunakan pendekatan **unsupervised anomaly detection**. Sistem ini menganalisis 14 indikator makroekonomi dari 49 negara dalam rentang waktu 35 tahun (1990–2024) untuk mengidentifikasi pola abnormal yang berpotensi mengindikasikan tekanan sistemik, ketidakseimbangan struktural, atau awal dari krisis ekonomi.

Pendekatan utama: **komparasi enam algoritma unsupervised lintas-paradigma** yang masing-masing memiliki dasar matematis berbeda, kemudian diagregasi melalui mekanisme **Majority Voting** untuk menghasilkan deteksi yang lebih robust.

### Algoritma yang Diimplementasikan

| # | Algoritma | Paradigma | Deskripsi Singkat |
|---|-----------|-----------|-------------------|
| 1 | **Isolation Forest** | Ensemble/Tree | Isolasi anomali via partisi acak — anomali lebih cepat terisolasi |
| 2 | **Local Outlier Factor (LOF)** | Density-based | Deteksi berdasarkan rasio kepadatan lokal tetangga |
| 3 | **Autoencoder** ✅ | Deep Learning | Kompresi non-linear encoder-bottleneck-decoder, anomali = reconstruction error tinggi |
| 4 | **One-Class SVM** | Kernel/Boundary | Transformasi RBF untuk membentuk batas keputusan di ruang dimensi tinggi |
| 5 | **PCA Reconstruction Error** | Linear Compression | Kompresi linear via komponen utama, anomali = error rekonstruksi besar |
| 6 | **DBSCAN** | Spatial Clustering | Pengelompokan berbasis densitas, titik di luar kluster = anomali |

> ✅ = Sudah diimplementasikan

---

## 🔍 Latar Belakang

Krisis ekonomi merupakan fenomena berulang yang memiliki dampak luas terhadap kesejahteraan masyarakat global. Beberapa krisis besar yang menjadi acuan dalam penelitian ini:

| Krisis | Tahun | Dampak |
|--------|-------|--------|
| 🌏 **Krisis Keuangan Asia** | 1997–1998 | Kolaps mata uang & perbankan di Asia Tenggara |
| 🇷🇺 **Krisis Rusia** | 1998 | Default utang negara & devaluasi Rubel |
| 🇦🇷 **Krisis Argentina** | 2001–2002 | Hyperinflasi & kolaps sistem perbankan |
| 🌍 **Global Financial Crisis** | 2008–2009 | Resesi global akibat krisis subprime mortgage |
| 🇪🇺 **Krisis Utang Eropa** | 2010–2012 | Krisis utang berdaulat Yunani, Portugal, Irlandia |
| 🦠 **Pandemi COVID-19** | 2020 | Kontraksi ekonomi global akibat pandemi |

Metode tradisional (ekonometrik) memiliki keterbatasan dalam menangkap pola non-linear. Pendekatan *unsupervised learning* menjadi solusi karena:
1. Krisis bersifat **langka** (*rare events*) → sulit mendapat data berlabel
2. Pola krisis **bervariasi** antar tipe, negara, dan era
3. Indikator ekonomi bersifat **multidimensi** dengan interaksi kompleks

---

## 📊 Dataset

### Sumber Data
Data diperoleh dari **[World Bank Open Data](https://data.worldbank.org/)** melalui API resmi menggunakan library Python `wbgapi`.

### Spesifikasi

| Aspek | Detail |
|-------|--------|
| **Observasi** | 1.714 baris |
| **Negara** | 49 negara dari 8 kawasan ekonomi |
| **Periode** | 1990–2024 (35 tahun) |
| **Fitur** | 14 indikator makroekonomi |
| **Preprocessing** | Z-score standardization |
| **Ground Truth** | 329 observasi krisis (19.2%) |

### 14 Indikator Makroekonomi

| No. | Kode World Bank | Indikator | Deskripsi |
|-----|----------------|-----------|-----------|
| 1 | `NY.GDP.MKTP.KD.ZG` | GDP Growth | Pertumbuhan PDB riil (%) |
| 2 | `NY.GDP.PCAP.KD.ZG` | GDP Per Capita Growth | Pertumbuhan PDB per kapita (%) |
| 3 | `FP.CPI.TOTL.ZG` | Inflation CPI | Inflasi berdasarkan CPI (%) |
| 4 | `FI.RES.TOTL.CD` | Total Reserves | Cadangan devisa total (USD) |
| 5 | `SL.UEM.TOTL.ZS` | Unemployment | Tingkat pengangguran (%) |
| 6 | `BN.CAB.XOKA.GD.ZS` | Current Account/GDP | Neraca transaksi berjalan/PDB (%) |
| 7 | `NE.TRD.GNFS.ZS` | Trade/GDP | Rasio perdagangan/PDB (%) |
| 8 | `BX.KLT.DINV.WD.GD.ZS` | FDI Inflows/GDP | FDI masuk/PDB (%) |
| 9 | `NE.EXP.GNFS.ZS` | Exports/GDP | Rasio ekspor/PDB (%) |
| 10 | `NE.IMP.GNFS.ZS` | Imports/GDP | Rasio impor/PDB (%) |
| 11 | `NY.GNS.ICTR.ZS` | Gross Savings/GDP | Tabungan bruto/PDB (%) |
| 12 | `PA.NUS.FCRF` | Exchange Rate | Nilai tukar resmi (LCU/USD) |
| 13 | `NV.IND.MANF.ZS` | Manufacturing Value | Kontribusi manufaktur/PDB (%) |
| 14 | `NE.GDI.TOTL.ZS` | Investment/GDP | Pembentukan modal tetap/PDB (%) |

### Cakupan 49 Negara (8 Kawasan)

| Kawasan | Negara |
|---------|--------|
| 🇺🇸 **G7** | USA, GBR, DEU, FRA, ITA, JPN, CAN |
| 🇧🇷 **BRICS** | BRA, RUS, IND, CHN, ZAF |
| 🇮🇩 **ASEAN-6** | IDN, THA, MYS, PHL, VNM, SGP |
| 🇰🇷 **Asia Lainnya** | KOR, PAK, BGD, LKA |
| 🇲🇽 **Amerika Latin** | MEX, ARG, COL, CHL, PER |
| 🇸🇦 **Timur Tengah & Afrika** | SAU, TUR, EGY, NGA, KEN, ARE |
| 🇪🇺 **Eropa** | ESP, NLD, SWE, NOR, POL, GRC, PRT, IRL, CHE, AUT, BEL, CZE, HUN, ROU |
| 🇦🇺 **Oseania** | AUS, NZL |

---

## 🧪 Metodologi

### Pipeline Data Mining

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  1. AKUISISI │────▶│ 2. PREPROCESS │────▶│ 3. REDUKSI DIMENSI│
│  World Bank  │     │  Interpolasi  │     │   PCA + t-SNE     │
│  API (wbgapi)│     │  Filtering    │     │                    │
│              │     │  Z-score      │     │                    │
└──────────────┘     └───────────────┘     └────────┬───────────┘
                                                     │
                                                     ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│ 6. INTERPRET │◀────│ 5. EVALUASI   │◀────│ 4. DETEKSI       │
│  Feature     │     │  Precision    │     │   ANOMALI        │
│  Importance  │     │  Recall       │     │   6 Algoritma    │
│  SHAP/Mean   │     │  F1-Score     │     │   Lintas-Paradigma│
└──────────────┘     └───────────────┘     └──────────────────┘
```

### Threshold & Evaluasi
- **Contamination rate:** ~8% (sesuai frekuensi historis krisis global)
- **Ground truth:** 6 periode krisis tervalidasi (Asia '97, Rusia '98, Argentina '01, GFC '08, Eropa '10, COVID '20)
- **Metrik:** Precision, Recall, F1-Score, AUC-ROC
- **Agregasi:** Majority Voting (≥3 dari 6 model) untuk konsensus final

---

## 📁 Struktur Proyek

```
ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/
│
├── 📄 README.md                          # Dokumentasi proyek (file ini)
├── 📄 Laporan_UTS.md                     # Proposal/laporan akademik lengkap
│
├── 📓 DETEKSI ANOMALI_DATA DAN EDA.ipynb # Notebook: Akuisisi data & EDA
├── 📓 Preprocessing_Data.ipynb           # Notebook: Preprocessing & cleaning
├── 📓 UTAMA_GABUNG.ipynb                 # Notebook: Gabungan semua metode (WIP)
├── 📓 nama_method.ipynb                  # Notebook: Template metode (WIP)
│
├── 📊 data_before clean.csv              # Data mentah dari World Bank (1.715 × 16)
├── 📊 data_with_labels.csv               # Data dengan crisis_label (1.714 × 17)
├── 📊 data_cleaned.csv                   # Data final: z-score + label (1.714 × 17)
│
├── 📂 deka/                              # Folder: Implementasi per-anggota (Deka)
│   ├── 📓 encoder.ipynb                  #   Autoencoder anomaly detection
│   ├── 📄 encoder.html                   #   Versi HTML notebook (siap cetak)
│   ├── 📊 hasil_autoencoder.csv          #   Output hasil deteksi (1.714 × 19)
│   └── 📄 laporan_autoencoder.md         #   Laporan singkat Autoencoder
│
└── 📂 .venv/                             # Virtual environment Python
```

### Deskripsi File Data

| File | Baris | Kolom | Deskripsi |
|------|-------|-------|-----------|
| `data_before clean.csv` | 1.715 | 16 | Data mentah dari World Bank API (14 indikator + economy + year) |
| `data_with_labels.csv` | 1.714 | 17 | Data + kolom `crisis_label` (ground truth krisis historis) |
| `data_cleaned.csv` | 1.714 | 17 | Data final ter-standardisasi (z-score) + crisis_label |
| `deka/hasil_autoencoder.csv` | 1.714 | 19 | Hasil Autoencoder: + `reconstruction_error` + `anomaly_ae` |

---

## ⚙️ Instalasi & Setup

### Prasyarat
- Python 3.10+ (dikembangkan dengan Python 3.13)
- pip (package manager)

### Langkah Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/gitraya1400/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI.git
cd ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependensi
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter wbgapi
```

### Dependensi Utama

| Package | Versi | Kegunaan |
|---------|-------|----------|
| `numpy` | ≥1.24 | Komputasi numerik |
| `pandas` | ≥2.0 | Manipulasi data |
| `matplotlib` | ≥3.8 | Visualisasi dasar |
| `seaborn` | ≥0.13 | Visualisasi statistik |
| `scikit-learn` | ≥1.4 | ML: IF, LOF, OCSVM, PCA, DBSCAN, metrik |
| `tensorflow` | ≥2.15 | Deep Learning: Autoencoder |
| `wbgapi` | ≥1.0 | Akses World Bank API |
| `jupyter` | ≥1.0 | Notebook environment |

---

## 🚀 Cara Menjalankan

### 1. Akuisisi & EDA
```bash
jupyter notebook "DETEKSI ANOMALI_DATA DAN EDA.ipynb"
```
Mengambil data dari World Bank API dan melakukan Exploratory Data Analysis.

### 2. Preprocessing
```bash
jupyter notebook "Preprocessing_Data.ipynb"
```
Membersihkan data, menangani missing values, dan melakukan standardisasi z-score.

### 3. Deteksi Anomali — Autoencoder
```bash
jupyter notebook "deka/encoder.ipynb"
```
Menjalankan Autoencoder untuk deteksi anomali. Output:
- `deka/hasil_autoencoder.csv` — Hasil deteksi
- `deka/encoder.html` — Versi HTML notebook

### 4. Melihat Laporan
- **Laporan akademik:** Buka `Laporan_UTS.md`
- **Laporan Autoencoder:** Buka `deka/laporan_autoencoder.md`
- **Notebook HTML:** Buka `deka/encoder.html` di browser

---

## 📈 Hasil & Temuan

### Autoencoder — Evaluasi vs Ground Truth

| Metrik | Nilai |
|--------|-------|
| **Precision** | 32.6% |
| **Recall** | 13.7% |
| **F1-Score** | 19.3% |
| **AUC-ROC** | 0.6066 |

### Deteksi per Krisis Historis

```
Krisis Asia 1997-98          ████████████████████████████████  50.0%  (5/10)
Krisis Argentina 2001-02     ████████████████████████████████  50.0%  (1/2)
Pandemi COVID-19 2020        ████████████                      18.4%  (9/49)
Global Financial Crisis      ████                               7.1%  (7/98)
Krisis Utang Eropa 2010-12   ████                               6.7%  (1/15)
Krisis Rusia 1998            ░                                  0.0%  (0/1)
```

### Interpretasi Kunci
- ✅ **Krisis regional yang intens** (Asia '97) paling baik terdeteksi oleh Autoencoder
- ⚠️ **Krisis global** (GFC '08) sulit dideteksi karena efek tersebar merata
- 📌 **Autoencoder tidak cukup sebagai EWS tunggal** — perlu dikombinasikan dengan algoritma lain melalui Majority Voting

### Top Negara dengan Anomali Terbanyak

| Negara | Kode | Anomali |
|--------|------|---------|
| Irlandia | IRL | 18 |
| Singapura | SGP | 16 |
| Belanda | NLD | 11 |
| Hongaria | HUN | 10 |
| Indonesia | IDN | 9 |

> Negara dengan ekonomi terbuka tinggi (trade-heavy) cenderung lebih sering ditandai sebagai anomali.

---

## 🛠️ Teknologi yang Digunakan

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Jupyter-F37726?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
</p>

---

## 📚 Referensi

1. Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.
2. Ahmed, M. et al. (2016). A survey of anomaly detection techniques in financial domain. *Future Generation Computer Systems*, 55.
3. Beutel, J. et al. (2019). Does machine learning help us predict banking crises? *Journal of Financial Stability*, 45.
4. Breunig, M. M. et al. (2000). LOF: Identifying density-based local outliers. *ACM SIGMOD*.
5. Chandola, V. et al. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3).
6. Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press.
7. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
8. Kaminsky, G. et al. (1998). Leading indicators of currency crises. *IMF Staff Papers*.
9. Laeven, L. & Valencia, F. (2018). Systemic banking crises revisited. *IMF Working Paper*.
10. Liu, F. T. et al. (2008). Isolation Forest. *IEEE ICDM*.
11. Reinhart, C. M. & Rogoff, K. S. (2009). *This Time Is Different*. Princeton University Press.
12. World Bank (2024). *World Bank Open Data*. https://data.worldbank.org/

> Referensi lengkap tersedia di [`Laporan_UTS.md`](Laporan_UTS.md#4-referensi)

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan akademik — **UTS Mata Kuliah Data Mining, Semester 6 (2026)**.

---

<p align="center">
  <sub>Dibuat dengan ❤️ menggunakan Python, TensorFlow, dan data dari World Bank Open Data</sub>
</p>
