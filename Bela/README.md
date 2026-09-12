# PCA-based Anomaly Detection — EWS Krisis Ekonomi (Kelompok 4)

Bagian individu dari proposal kelompok **"Analisis Deteksi Anomali pada Indikator
Makroekonomi Global sebagai Instrumen Early Warning System (EWS) Krisis Ekonomi"**
(Program Studi D-IV Komputasi Statistik, Politeknik Statistika STIS).

Repositori ini berisi implementasi **PCA-based Anomaly Detection** menggunakan
Squared Prediction Error (SPE) dan Hotelling's T², untuk dibandingkan dengan
algoritma lain milik anggota kelompok (Isolation Forest, LOF, Autoencoder,
One-Class SVM, DBSCAN) pada laporan gabungan.

## ⚠️ Catatan Revisi Metodologi (penting)

Versi awal bagian ini melatih PCA dengan skema *train/validation/test* (70/15/15,
mengikuti pola Autoencoder), sehingga hasilnya **tidak bisa dibandingkan secara
adil** dengan LOF — LOF di-*scoring* pada seluruh 1.714 baris, sedangkan PCA versi
lama hanya dievaluasi pada subset test.

Perbaikan yang diterapkan di repositori ini:

1. **StandardScaler dan PCA di-*fit* HANYA dari seluruh baris normal**
   (`crisis_label == 0`, 1.385 baris) — *bukan* hanya 70%-nya. Prinsip "PCA hanya
   belajar dari kondisi normal" tetap terjaga; tidak ada informasi krisis yang
   bocor ke tahap pelatihan.
2. **SPE dan T² dihitung untuk SELURUH 1.714 baris** (normal + krisis), persis
   seperti cara LOF di-*scoring* pada seluruh dataset.
3. **UCL (Upper Control Limit)** tetap dihitung **hanya dari sebaran SPE/T² data
   normal**, sehingga tidak ada label krisis yang bocor ke penentuan ambang batas
   anomali.

Dengan skema ini, metrik evaluasi PCA (AUC-ROC, F1, dll.) dapat dibandingkan
langsung dengan metrik LOF pada laporan gabungan kelompok.

## Struktur Folder

```
├── data/               → data_cleaned.csv (1.714 observasi, 49 negara, 1990–2024)
├── notebooks/          → notebook lengkap (penjelasan teori + kode + hasil, sudah dieksekusi)
├── scripts/            → versi .py modular, bisa dijalankan ulang sekali klik
├── results/            → CSV hasil deteksi, metrik evaluasi, effect size
├── figures/            → 4 PNG (scree plot, control chart, contribution plot, effect size)
├── report/             → laporan Word (.docx) + script Node.js pembuatnya
├── README.md
└── .gitignore
```

## Cara Menjalankan Ulang (Reproducibility)

### 1. Install dependency Python

```bash
pip install numpy pandas scipy scikit-learn matplotlib nbformat --break-system-packages
```

### 2. Jalankan seluruh pipeline dengan satu perintah

```bash
cd scripts
python run_all.py
```

Ini akan menjalankan berurutan:

| Script | Fungsi | Output |
|---|---|---|
| `config.py` | Konfigurasi path & parameter bersama | – |
| `01_pca_pipeline.py` | Fit scaler & PCA (data normal saja), hitung SPE/T²/UCL untuk semua baris | `results/pca_scores.csv`, `results/pca_model_summary.csv`, `results/pca_run_metadata.csv`, `results/contribution_matrix.csv` |
| `02_evaluate.py` | Hitung AUC-ROC, AP, Precision/Recall/F1, Cohen's d, Cliff's delta | `results/evaluation_metrics.csv`, `results/effect_size.csv` |
| `03_visualize.py` | Buat 4 figure | `figures/01_scree_plot.png` … `figures/04_effect_size_plot.png` |

### 3. (Opsional) Buka notebook untuk versi belajar + narasi lengkap

```bash
jupyter notebook notebooks/pca_anomaly_detection.ipynb
```

Notebook ini **self-contained** — berisi ulang seluruh kode di atas beserta
penjelasan rumus (PCA, SPE, T², UCL Jackson-Mudholkar, UCL distribusi F, Cohen's
d, Cliff's delta) dan sudah dieksekusi sehingga semua output/plot langsung
terlihat tanpa perlu dijalankan ulang.

### 4. (Opsional) Generate ulang laporan Word

```bash
cd report
node generate_report.js
```

Script ini membaca hasil terbaru dari `results/*.csv` dan gambar dari
`figures/*.png`, sehingga laporan otomatis konsisten dengan hasil run terakhir.
Membutuhkan package `docx` (npm) — sudah tersedia di environment pengembangan
skill dokumen; jika belum ada, jalankan `npm install docx`.

## Ringkasan Metodologi

- **Fitur:** 14 indikator makroekonomi numerik terstandardisasi (Z-score),
  identik dengan fitur yang dipakai algoritma lain di kelompok (economy & year
  hanya sebagai identifier).
- **Pemilihan k:** cumulative explained variance ≥ 95% → **k = 9** komponen dari
  14 fitur (95,41% variance dijelaskan).
- **SPE (Q-statistic):** reconstruction error di luar ruang k komponen utama.
- **Hotelling's T²:** jarak terbobot di dalam ruang k komponen utama.
- **UCL SPE:** pendekatan chi-square (Jackson & Mudholkar, 1979) dari eigenvalue
  komponen yang dibuang (data normal).
- **UCL T²:** distribusi F, dengan n = jumlah data normal yang dipakai melatih
  PCA (bukan seluruh dataset).
- **α = 0.05** (UCL pada persentil ke-95).

## Ringkasan Hasil Utama

| Skor | AUC-ROC | Average Precision | Cohen's d | Cliff's delta |
|---|---|---|---|---|
| SPE | 0.521 | 0.221 | 0.146 (negligible) | 0.042 (negligible) |
| Hotelling's T² | 0.675 | 0.350 | 0.454 (small) | 0.349 (medium) |
| Combined score | 0.665 | 0.338 | 0.451 (small) | 0.330 (medium) |

**Temuan utama:** Hotelling's T² memberikan diskriminasi krisis-vs-normal yang
lebih baik dibandingkan SPE pada dataset ini — periode krisis lebih sering
bermanifestasi sebagai kombinasi ekstrem *di dalam* ruang komponen utama
(ditangkap T²), bukan sebagai pola yang gagal direkonstruksi (ditangkap SPE).
Detail lengkap beserta pembahasan ada di `report/Laporan_PCA_Anomaly_Detection_Kelompok4.docx`.

## Sumber Data

`data_cleaned.csv` bersumber dari World Bank Global Economic Monitor,
dikombinasikan dengan katalog krisis historis IMF/World Bank untuk
`crisis_label` (lihat proposal kelompok, Bab 3.1).
