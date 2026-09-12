# PCA-based Anomaly Detection — Early Warning System Krisis Ekonomi

Bagian individu dari proyek kelompok **3SI1 Kelompok 4** — *Analisis Deteksi Anomali pada
Indikator Makroekonomi Global sebagai Instrumen Early Warning System Krisis Ekonomi*
(sesuai proposal Section 2.1.3.5 dan 3.2.6).

## Struktur Folder

```
.
├── data/
│   └── data_with_labels.csv          # Data hasil preprocessing tim (input utama)
├── notebooks/
│   └── PCA_Anomaly_Detection_EWS.ipynb   # Notebook step-by-step + penjelasan konsep + hasil
├── scripts/
│   └── pca_anomaly_detection.py      # Versi script non-interaktif (sekali jalan, hasilkan semua output)
├── results/
│   ├── test_result.csv               # Hasil deteksi per observasi (SPE, T2, is_anomaly, dst)
│   ├── metrics_summary.csv           # Ringkasan metrik evaluasi & confusion matrix
│   ├── effect_sizes.csv              # Effect size tiap variabel (anomali vs normal)
│   └── run_log.txt                   # Log lengkap output saat pipeline dijalankan
├── figures/
│   ├── 01_scree_plot.png
│   ├── 02_control_chart_spe.png
│   ├── 03_contribution_plot_top_anomaly.png
│   └── 04_effect_size.png
└── report/
    ├── Laporan_PCA_Anomaly_Detection.docx   # Laporan individu (Word), siap dibaca/diedit
    └── build_report.js                       # Script pembuat laporan (docx-js), untuk regenerasi otomatis
```

## Cara Menjalankan Ulang

**Opsi 1 — Notebook (disarankan untuk belajar step-by-step):**
Buka `notebooks/PCA_Anomaly_Detection_EWS.ipynb` di Jupyter/Colab, jalankan sel dari atas ke bawah.
Notebook ini sudah berisi hasil eksekusi dengan data asli di `data/data_with_labels.csv`.

**Opsi 2 — Script (untuk regenerasi cepat semua hasil/grafik):**
```bash
cd scripts
python pca_anomaly_detection.py
```
Akan menimpa ulang isi folder `results/` dan `figures/` dengan hasil terbaru.

## Ringkasan Hasil

| Metrik | Nilai |
|---|---|
| Jumlah komponen PCA (k) | 9 dari 14 (≥95% explained variance) |
| AUC-ROC | 0.6414 |
| F1-Score | 0.3984 |
| Precision | 0.6049 |
| Recall | 0.2970 |
| Average Precision | 0.5785 |

Detail interpretasi lengkap ada di `report/Laporan_PCA_Anomaly_Detection.docx` dan Section 12 notebook.

## Catatan untuk Integrasi Kelompok

- Preprocessing (fitur, split 70/15/15, StandardScaler) mengikuti kerangka komparatif seragam
  di proposal Section 3.2.1, supaya bisa dibandingkan apple-to-apple dengan 5 algoritma lain
  (Isolation Forest, LOF, Autoencoder, OCSVM, DBSCAN).
- File `results/test_result.csv` dan `results/metrics_summary.csv` bisa langsung dipakai
  teman yang mengerjakan Tahap 6 (Evaluasi & Seleksi Model) untuk digabung dengan hasil algoritma lain.
