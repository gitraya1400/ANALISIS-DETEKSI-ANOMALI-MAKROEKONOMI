# 📍 Laporan Singkat — Deteksi Anomali Ekonomi Makro dengan Local Outlier Factor (LOF)

**Disusun oleh:** Amir  
**Tanggal:** 21 April 2026  
**Metode:** Local Outlier Factor (Density-Based — Unsupervised Anomaly Detection)

---

## 1. Ringkasan Eksekutif

Notebook ini mengimplementasikan **Local Outlier Factor (LOF)** sebagai salah satu dari enam algoritma deteksi anomali dalam proyek *Early Warning System* (EWS) krisis ekonomi. LOF bekerja dengan prinsip **perbandingan kerapatan lokal**: suatu observasi dianggap anomali apabila kerapatan lokalnya secara signifikan lebih rendah dibandingkan kerapatan lokal tetangga-tetangganya.

**Metodologi:**
- Prinsip: Perbandingan kerapatan lokal (*local density comparison*) antar observasi
- Grid Search: 5 variasi `n_neighbors` x 4 variasi `contamination` = 20 kombinasi
- Evaluasi: Precision, Recall, F1-Score, ROC-AUC terhadap *ground truth* krisis historis
- Interpretasi: Analisis LOF Score per negara dan per krisis historis

---

## 2. Konsep Local Outlier Factor

### Prinsip Kerja LOF

LOF, yang diperkenalkan oleh Breunig et al. (2000), mengukur derajat anomali suatu titik data secara **relatif terhadap kerapatan lokal** tetangga-tetangganya:

1. **k-distance(p)**: Jarak dari titik p ke tetangga ke-k terdekatnya
2. **Reachability Distance**: `reach-dist_k(p,o) = max(k-dist(o), d(p,o))`
3. **Local Reachability Density (LRD)**: Kebalikan dari rata-rata *reachability distance* terhadap k-tetangga — semakin padat lingkungan, semakin tinggi LRD
4. **LOF Score**: Rasio rata-rata LRD tetangga / LRD titik itu sendiri — mengukur seberapa jarang titik tersebut relatif terhadap lingkungannya

- **LOF ~ 1** => Titik berada di area dengan kerapatan serupa => **Normal**
- **LOF >> 1** => Titik berada di area jauh lebih jarang dari tetangganya => **Anomali**

---

## 3. Dataset

| Aspek | Detail |
|-------|--------|
| **Sumber** | World Bank Open Data API |
| **Observasi** | 1.714 baris (49 negara × 35 tahun) |
| **Periode** | 1990–2024 |
| **Fitur** | 14 indikator makroekonomi (sudah z-score) |
| **Ground Truth** | 329 observasi krisis (19.2%) |

### 14 Indikator Makroekonomi

| No. | Fitur | Deskripsi |
|-----|-------|-----------|
| 1 | GDP_Growth | Pertumbuhan PDB riil (%) |
| 2 | GDP_PerCapita_Growth | Pertumbuhan PDB per kapita (%) |
| 3 | Inflation_CPI | Inflasi berdasarkan CPI (%) |
| 4 | Total_Reserves | Cadangan devisa total |
| 5 | Unemployment | Tingkat pengangguran (%) |
| 6 | Current_Account_GDP | Neraca transaksi berjalan/PDB (%) |
| 7 | Trade_GDP | Rasio perdagangan/PDB (%) |
| 8 | FDI_Inflows_GDP | FDI masuk/PDB (%) |
| 9 | Exports_GDP | Rasio ekspor/PDB (%) |
| 10 | Imports_GDP | Rasio impor/PDB (%) |
| 11 | Gross_Savings_GDP | Tabungan bruto/PDB (%) |
| 12 | Exchange_Rate | Nilai tukar LCU/USD |
| 13 | Manufacturing_Value | Kontribusi manufaktur/PDB (%) |
| 14 | Investment_GDP | Investasi bruto/PDB (%) |

---

## 4. Konfigurasi Model

**Spesifikasi:**
- **Algoritma:** Local Outlier Factor (LOF)
- **n_neighbors:** 10 (ukuran lingkungan lokal)
- **Contamination:** 0.1919 (sesuai rasio krisis aktual)
- **Metric:** Euclidean (Minkowski p=2)
- **Grid Search:** 5 variasi n_neighbors × 4 variasi contamination = 20 kombinasi

---

## 5. Hasil Deteksi Anomali

### 5.1 Ringkasan Deteksi

| Parameter | Nilai |
|-----------|-------|
| **n_neighbors** | 10 |
| **Contamination** | 0.1919 (rasio krisis aktual) |
| **Anomali terdeteksi** | 329 observasi (19.2%) |

### 5.2 Evaluasi vs Ground Truth

| Metrik | Nilai |
|--------|-------|
| **Precision** | 0.3435 (34.4%) |
| **Recall** | 0.3435 (34.4%) |
| **F1-Score** | 0.3435 (34.4%) |
| **ROC-AUC** | 0.6794 |
| **Average Precision** | 0.3373 |

### 5.3 Confusion Matrix

|  | **Prediksi: Normal** | **Prediksi: Anomali** |
|--|---------------------|-----------------------|
| **Aktual: Normal** | TN = 1.169 | FP = 216 |
| **Aktual: Krisis** | FN = 216 | TP = 113 |

---

## 6. Deteksi per Krisis Historis

| Krisis | Tahun | Observasi | Terdeteksi | Detection Rate |
|--------|-------|-----------|------------|----------------|
| **Krisis Asia** | 1997–1998 | 14 | 7 | **50.0%** |
| **Krisis Rusia** | 1998 | 1 | 0 | 0.0% |
| **Krisis Argentina** | 2001–2002 | 2 | 1 | **50.0%** |
| **Global Financial Crisis** | 2008–2009 | 98 | 22 | **22.4%** |
| **Krisis Utang Eropa** | 2010–2012 | 15 | 9 | **60.0%** |
| **Pandemi COVID-19** | 2020 | 49 | 22 | **44.9%** |

### Interpretasi:
- **Krisis Utang Eropa** mendapat detection rate tertinggi (60%), karena LOF efektif mendeteksi negara-negara Eropa Selatan yang menyimpang dari peer group regional mereka.
- **GFC** sulit dideteksi (22.4%) karena dampaknya merata secara global — kerapatan lokal setiap titik tidak turun signifikan relatif terhadap tetangganya.
- **COVID-19** terdeteksi 44.9%, lebih baik dari Autoencoder (18.4%) — guncangan eksogen ini menciptakan titik ekstrem dalam ruang fitur.

---

## 7. Top 10 Negara dengan Anomali Terbanyak

| No. | Negara | Kode | Jumlah Anomali |
|-----|--------|------|----------------|
| 1 | Irlandia | IRL | 27 |
| 2 | Arab Saudi | SAU | 23 |
| 3 | Nigeria | NGA | 21 |
| 4 | Yunani | GRC | 16 |
| 5 | Rumania | ROU | 15 |
| 6 | Hongaria | HUN | 15 |
| 7 | Swiss | CHE | 14 |
| 8 | Rusia | RUS | 10 |
| 9 | Belanda | NLD | 10 |
| 10 | Tiongkok | CHN | 10 |

> **Catatan:** Negara berbasis komoditas/migas (SAU, NGA) dan ekonomi yang sangat terbuka (IRL, NLD) cenderung memiliki profil indikator yang berbeda dari peer group mereka, sehingga LOF menandainya lebih sering sebagai anomali.

---

## 8. Analisis & Interpretasi

### Kekuatan LOF:
1. **Deteksi anomali lokal** — mampu mengidentifikasi negara/tahun yang anomali dalam konteks peer group mereka, bukan hanya outlier global.
2. **Hasil terbaik pada Krisis Utang Eropa** (60%) — paling unggul dibandingkan Autoencoder (6.7%) untuk krisis bersifat klaster regional.
3. **Tidak memerlukan label** — sepenuhnya *unsupervised*, sesuai dengan sifat deteksi anomali.

### Keterbatasan:
1. **Recall rendah pada mode konservatif** — mode `auto` hanya menghasilkan Recall 8.5%.
2. **Bias terhadap negara struktural outlier** — petrostate dan ekonomi terbuka sering ditandai anomali karena profil strukturalnya, bukan karena krisis.
3. **Tidak mendeteksi krisis global merata** — GFC hanya terdeteksi 22.4% karena semua negara terdampak bersamaan.
4. **Sensitif terhadap dimensionalitas tinggi** — Goldstein & Uchida (2016) menunjukkan bahwa performa LOF cenderung menurun pada dataset berdimensi tinggi (> 10 fitur) karena jarak antar titik menjadi kurang diskriminatif (*curse of dimensionality*), relevan mengingat dataset ini memiliki 14 fitur.

### Rekomendasi:
- LOF **tidak cukup kuat berdiri sendiri** sebagai satu-satunya EWS.
- Perlu dikombinasikan dengan algoritma lain (Isolation Forest, Autoencoder, OCSVM, PCA, DBSCAN) melalui **Majority Voting** untuk meningkatkan reliabilitas deteksi.
- Gunakan `contamination=0.1919` jika prioritas adalah coverage maksimal untuk EWS.
---

## 9. File Output

| File | Deskripsi |
|------|-----------|
| `lof.ipynb` | Notebook lengkap dengan kode dan visualisasi |
| `hasil_lof.csv` | Dataset hasil deteksi (1.714 baris) |
| `laporan_lof.md` | Laporan singkat ini |

### Kolom pada `hasil_lof.csv`:
- `economy`, `year` — Metadata negara dan tahun
- `lof_score` — Nilai LOF score (semakin tinggi = semakin anomali)
- `predicted_anomaly` — Prediksi LOF (0=Normal, 1=Anomali)
- `crisis_label` — Ground truth (0=Normal, 1=Krisis)

---

## Referensi

1. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 93–104.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Goldstein, M., & Uchida, S. (2016). A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data. *PLOS ONE*, 11(4).

---

*Laporan ini merupakan bagian dari proyek **Deteksi Anomali pada Indikator Ekonomi Makro sebagai Early Warning System Krisis Ekonomi menggunakan Pendekatan Unsupervised Learning** — UTS Data Mining 2026.*
