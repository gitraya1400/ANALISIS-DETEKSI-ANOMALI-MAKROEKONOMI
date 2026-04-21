# 🔍 Laporan Singkat — Deteksi Anomali Ekonomi Makro dengan DBSCAN

**Disusun oleh:** Bram  
**Tanggal:** 21 April 2026  
**Metode:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise — Unsupervised Anomaly Detection)

---

## 1. Ringkasan Eksekutif

Notebook ini mengimplementasikan **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) sebagai salah satu dari enam algoritma deteksi anomali dalam proyek *Early Warning System* (EWS) krisis ekonomi. DBSCAN bekerja dengan prinsip **pengelompokan berbasis kepadatan**: algoritma membentuk cluster dari titik-titik yang berdekatan secara spasial, sementara titik-titik yang tidak masuk ke cluster manapun dilabeli sebagai **noise (−1)** dan diperlakukan sebagai **anomali**.

**Metodologi:**
- Prinsip: Spatial clustering berbasis density — titik noise (−1) = anomali
- Grid Search: 8 variasi `eps` × 5 variasi `min_samples` = 40 kombinasi (26 valid)
- Evaluasi: Precision, Recall, F1-Score terhadap *ground truth* krisis historis
- Interpretasi: Analisis cluster, feature importance, dan deteksi per krisis

---

## 2. Konsep DBSCAN

### Prinsip Kerja DBSCAN

DBSCAN, yang diperkenalkan oleh Ester et al. (1996), mengelompokkan titik-titik data berdasarkan **kepadatan spasial** dalam ruang fitur berdimensi tinggi. Algoritma ini mengklasifikasikan setiap titik data menjadi salah satu dari tiga kategori:

1. **Core Point**: Titik yang memiliki setidaknya `min_samples` tetangga dalam radius `eps`. Titik ini menjadi inti dari sebuah cluster.
2. **Border Point**: Titik yang berada dalam radius `eps` dari core point, namun tidak memenuhi syarat sebagai core point. Titik ini menjadi anggota pinggiran cluster.
3. **Noise Point (−1)**: Titik yang tidak berada dalam radius `eps` dari core point manapun. Titik ini dianggap **anomali**.

### Mekanisme Kerja

1. **Inisialisasi**: Algoritma memilih titik data yang belum dikunjungi secara acak.
2. **Ekspansi cluster**: Jika titik tersebut merupakan *core point*, cluster baru dibentuk dan diperluas secara rekursif ke semua titik yang *density-reachable*.
3. **Labeling noise**: Titik yang tidak dapat dijangkau oleh cluster manapun dilabeli sebagai noise (−1).
4. **Iterasi**: Proses diulang hingga semua titik telah dikunjungi.

### Keunggulan untuk Deteksi Anomali

| Aspek | DBSCAN | Metode Lain |
|-------|--------|-------------|
| **Bentuk cluster** | Tidak mengasumsikan bentuk tertentu (bisa non-convex) | K-Means: mengasumsikan spherical |
| **Jumlah cluster** | Ditentukan otomatis oleh data | K-Means: perlu ditentukan a priori |
| **Deteksi anomali** | Native — noise point = anomali | Isolation Forest: berbasis path length |
| **Sensitivitas** | Efektif pada anomali yang terletak jauh dari cluster manapun | LOF: relatif terhadap tetangga lokal |

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

## 4. Konfigurasi Model & Grid Search

### 4.1 Parameter DBSCAN

DBSCAN memiliki dua hyperparameter utama:

- **`eps` (epsilon)**: Radius lingkungan (*neighbourhood*) — jarak maksimum antara dua titik agar masih dianggap bertetangga. Semakin kecil `eps`, semakin ketat definisi cluster, dan semakin banyak titik diklasifikasikan sebagai noise/anomali.
- **`min_samples`**: Jumlah minimum titik dalam radius `eps` agar sebuah titik dianggap sebagai *core point*. Semakin besar nilai ini, semakin ketat syarat pembentukan cluster.

### 4.2 Estimasi Awal Parameter eps — K-Distance Plot

Sebelum melakukan grid search, dilakukan analisis **K-distance plot** untuk mendapatkan estimasi awal parameter `eps` yang baik. Metode ini menghitung jarak ke tetangga ke-k dari setiap titik data, mengurutkannya, dan mencari titik "elbow" — di mana jarak meningkat tajam — sebagai indikasi nilai `eps` yang optimal.

### 4.3 Grid Search

| Parameter | Nilai yang Diuji |
|-----------|-----------------|
| **eps** | 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 |
| **min_samples** | 3, 5, 7, 10, 15 |
| **Metric** | Euclidean |
| **Total kombinasi** | 40, menghasilkan 26 kombinasi valid |
| **Kriteria seleksi** | F1-Score tertinggi terhadap ground truth |

### 4.4 Top 5 Hasil Grid Search

| Rank | eps | min_samples | Clusters | Noise (%) | Precision | Recall | F1-Score |
|------|-----|-------------|----------|-----------|-----------|--------|----------|
| 1 | 1.5 | 10 | 7 | 17.0% | 0.3390 | 0.3009 | **0.3188** |
| 2 | 1.5 | 15 | 3 | 23.6% | 0.2889 | 0.3556 | **0.3188** |
| 3 | 1.5 | 7 | 11 | 13.3% | 0.3816 | 0.2644 | 0.3124 |
| 4 | 1.5 | 5 | 10 | 11.1% | 0.4211 | 0.2432 | 0.3083 |
| 5 | 1.5 | 3 | 16 | 7.0% | 0.4583 | 0.1672 | 0.2450 |

> **Catatan**: `eps=1.5` secara konsisten menghasilkan F1-Score tertinggi, menunjukkan bahwa pada radius yang lebih kecil, DBSCAN mampu membentuk cluster yang lebih ketat dan menyisihkan titik-titik anomali dengan lebih efektif.

---

## 5. Hasil Deteksi Anomali

### 5.1 Parameter Terbaik

| Parameter | Nilai |
|-----------|-------|
| **eps** | 1.5 |
| **min_samples** | 10 |
| **Jumlah cluster** | 7 |
| **Anomali terdeteksi** | 292 observasi (17.0%) |

### 5.2 Evaluasi vs Ground Truth

| Metrik | Nilai |
|--------|-------|
| **Precision** | 0.3390 (33.9%) |
| **Recall** | 0.3009 (30.1%) |
| **F1-Score** | 0.3188 (31.9%) |

### 5.3 Confusion Matrix

|  | **Prediksi: Normal** | **Prediksi: Anomali** |
|--|---------------------|-----------------------|
| **Aktual: Normal** | TN = 1.192 | FP = 193 |
| **Aktual: Krisis** | FN = 230 | TP = 99 |

### 5.4 Distribusi Cluster

| Cluster | Jumlah Observasi | Persentase |
|---------|-----------------|------------|
| **Noise (−1) / Anomali** | 292 | 17.0% |
| **Cluster 0** (cluster utama) | 1.343 | 78.4% |
| **Cluster 1** | 17 | 1.0% |
| **Cluster 2** | 12 | 0.7% |
| **Cluster 3** | 14 | 0.8% |
| **Cluster 4** | 10 | 0.6% |
| **Cluster 5** | 14 | 0.8% |
| **Cluster 6** | 12 | 0.7% |

> **Catatan:** Cluster 0 mendominasi data (78.4%), merepresentasikan kondisi ekonomi "normal". Enam cluster minor (total 79 observasi / 4.6%) menangkap kelompok-kelompok ekonomi dengan karakteristik spesifik yang tetap terstruktur (bukan noise). Sisa 17.0% dianggap noise/anomali.

---

## 6. Deteksi per Krisis Historis

| Krisis | Tahun | Observasi | Terdeteksi | Detection Rate |
|--------|-------|-----------|------------|----------------|
| **Krisis Asia** | 1997–1998 | 10 | 6 | **60.0%** |
| **Krisis Rusia** | 1998 | 1 | 0 | 0.0% |
| **Krisis Argentina** | 2001–2002 | 2 | 2 | **100.0%** |
| **Global Financial Crisis** | 2008–2009 | 98 | 27 | **27.6%** |
| **Krisis Utang Eropa** | 2010–2012 | 15 | 7 | **46.7%** |
| **Pandemi COVID-19** | 2020 | 49 | 26 | **53.1%** |

### Interpretasi:
- **Krisis Argentina** mendapat detection rate tertinggi (**100%**) — kedua observasi (2001, 2002) terdeteksi sebagai noise. Hal ini karena krisis Argentina bersifat sangat mendalam dan terisolasi, menciptakan titik data yang sangat jauh dari cluster utama.
- **Krisis Asia** terdeteksi 60% — IDN (1998), KOR (1998), MYS (1997-1998), dan THA (1997-1998) berhasil diidentifikasi. Negara-negara ini menunjukkan guncangan simultan pada multiple indikator.
- **COVID-19** terdeteksi 53.1% (26 dari 49 negara) — detection rate tertinggi untuk krisis global. Guncangan eksogen pandemi menciptakan pola indikator yang ekstrem.
- **GFC** terdeteksi 27.6% — meskipun berdampak global, hanya negara-negara yang terkena sangat parah (seperti IRL, SGP, HUN) yang profil indikatornya cukup menyimpang untuk diklasifikasikan sebagai noise.
- **Krisis Utang Eropa** terdeteksi 46.7% — GRC (2010-2012), IRL (2010-2012), dan PRT (2012) teridentifikasi.
- **Krisis Rusia** tidak terdeteksi (0%) — hanya satu observasi, dan Rusia pada 1998 mungkin tidak cukup terisolasi dalam ruang fitur 14 dimensi.

---

## 7. Top 10 Negara dengan Anomali Terbanyak

| No. | Negara | Kode | Jumlah Anomali |
|-----|--------|------|----------------|
| 1 | Singapura | SGP | 35 |
| 2 | Irlandia | IRL | 32 |
| 3 | Vietnam | VNM | 23 |
| 4 | Malaysia | MYS | 20 |
| 5 | Belanda | NLD | 18 |
| 6 | Arab Saudi | SAU | 17 |
| 7 | Hongaria | HUN | 17 |
| 8 | Swiss | CHE | 13 |
| 9 | Rumania | ROU | 11 |
| 10 | Rusia | RUS | 9 |

> **Catatan:** Negara-negara dengan ekonomi sangat terbuka (rasio Trade/GDP, Exports/GDP tinggi) seperti SGP, IRL, NLD, dan MYS mendominasi daftar ini. DBSCAN menandai mereka karena profil indikator mereka secara spasial berada jauh dari cluster utama yang didominasi oleh ekonomi berukuran menengah hingga besar dengan keterbukaan moderat.

---

## 8. Feature Importance

Analisis feature importance berdasarkan **perbedaan rata-rata absolut** antara observasi anomali (noise) dan normal (cluster member):

| Rank | Fitur | |Mean Diff| | Mean Normal | Mean Anomali |
|------|-------|-----------:|------------:|-------------:|
| 1 | Exports_GDP | **1.3179** | −0.2272 | 1.0907 |
| 2 | Trade_GDP | **1.3066** | −0.2252 | 1.0814 |
| 3 | Imports_GDP | **1.2771** | −0.2200 | 1.0571 |
| 4 | FDI_Inflows_GDP | 0.8012 | −0.1412 | 0.6600 |
| 5 | Gross_Savings_GDP | 0.6584 | −0.1123 | 0.5461 |
| 6 | Current_Account_GDP | 0.6562 | −0.1112 | 0.5450 |
| 7 | Manufacturing_Value | 0.4245 | −0.0735 | 0.3510 |
| 8 | Exchange_Rate | 0.2802 | −0.0500 | 0.2302 |
| 9 | GDP_PerCapita_Growth | 0.2670 | 0.0440 | −0.2229 |
| 10 | Investment_GDP | 0.2529 | −0.0450 | 0.2079 |
| 11 | GDP_Growth | 0.1742 | 0.0277 | −0.1465 |
| 12 | Unemployment | 0.1393 | 0.0201 | −0.1192 |
| 13 | Inflation_CPI | 0.1390 | −0.0582 | 0.0808 |
| 14 | Total_Reserves | 0.0629 | −0.0140 | 0.0489 |

### Interpretasi Feature Importance:
- **Trade-related indicators** (Exports/GDP, Trade/GDP, Imports/GDP) merupakan tiga fitur terpenting, mengkonfirmasi bahwa DBSCAN menangkap pola anomali pada negara-negara dengan keterbukaan ekonomi ekstrem.
- **FDI dan Savings** juga berperan signifikan — anomali cenderung memiliki nilai FDI dan tabungan bruto yang lebih tinggi dari rata-rata.
- **GDP Growth dan Unemployment** memiliki perbedaan yang relatif kecil, menunjukkan bahwa DBSCAN lebih sensitif terhadap indikator struktural (trade, FDI) daripada indikator siklus ekonomi (growth, unemployment).

---

## 9. Analisis & Interpretasi

### Kekuatan DBSCAN:
1. **Deteksi anomali native** — noise point secara natural merepresentasikan observasi yang tidak cocok dengan pola mayoritas, tanpa perlu menentukan threshold atau contamination rate.
2. **Tidak memerlukan jumlah cluster a priori** — berbeda dengan K-Means, DBSCAN menentukan jumlah cluster secara otomatis berdasarkan struktur data.
3. **Robust terhadap bentuk cluster non-convex** — mampu mendeteksi formasi cluster yang kompleks, relevan untuk data ekonomi multivariat yang memiliki korelasi non-linear antar indikator.
4. **Tidak memerlukan label** — sepenuhnya *unsupervised*, sesuai dengan paradigma deteksi anomali.

### Keterbatasan:
1. **Sensitif terhadap parameter `eps` dan `min_samples`** — perubahan kecil pada parameter dapat mengubah hasil secara signifikan. Grid search menjadi kritis.
2. **Kesulitan pada densitas bervariasi** — jika cluster memiliki kepadatan yang sangat berbeda, satu nilai `eps` global mungkin tidak optimal untuk semua cluster. Beberapa cluster ekonomi (misalnya negara maju vs negara berkembang) memiliki karakteristik densitas yang berbeda.
3. **Tidak menghasilkan skor kontinu** — berbeda dengan Isolation Forest (anomaly score) atau Autoencoder (reconstruction error), DBSCAN hanya menghasilkan label biner (cluster vs noise). Hal ini membatasi kemampuan untuk melakukan *ranking* tingkat keparahan anomali.
4. **Curse of dimensionality** — performa DBSCAN cenderung menurun pada data berdimensi tinggi (14 fitur) karena konsep jarak Euclidean menjadi kurang diskriminatif, sesuai dengan temuan Aggarwal et al. (2001).
5. **Sensitivitas terhadap metrik jarak** — menggunakan jarak Euclidean pada data yang sudah di-standardisasi (z-score), namun korelasi antar fitur tidak dipertimbangkan. Metrik Mahalanobis atau penggunaan PCA pra-clustering dapat meningkatkan performa.

### Perbandingan dengan Metode Lain:

| Metrik | DBSCAN | LOF | Autoencoder |
|--------|--------|-----|-------------|
| **Precision** | 0.3390 | 0.3435 | 0.3261 |
| **Recall** | 0.3009 | 0.3435 | 0.1368 |
| **F1-Score** | 0.3188 | 0.3435 | 0.1927 |
| **Anomali (%)** | 17.0% | 19.2% | 8.1% |

> **Catatan:** DBSCAN memiliki performa yang kompetitif dengan LOF dan secara signifikan lebih baik dari Autoencoder dalam hal F1-Score. Precision DBSCAN (33.9%) sebanding dengan LOF (34.4%), namun recall-nya sedikit lebih rendah (30.1% vs 34.4%).

### Rekomendasi:
- DBSCAN **tidak cukup kuat berdiri sendiri** sebagai satu-satunya EWS — sama seperti metode lain.
- Perlu dikombinasikan dengan algoritma lain (Isolation Forest, LOF, Autoencoder, One-Class SVM, PCA Reconstruction Error) melalui **Majority Voting** untuk meningkatkan reliabilitas deteksi.
- Eksplorasi **HDBSCAN** (*Hierarchical DBSCAN*) sebagai alternatif yang lebih robust terhadap variasi densitas.
- Pertimbangkan reduksi dimensi (PCA) sebelum DBSCAN untuk mengurangi efek *curse of dimensionality*.

---

## 10. File Output

| File | Deskripsi |
|------|-----------|
| `dbscan.ipynb` | Notebook lengkap dengan kode dan visualisasi |
| `grid_search_dbscan.csv` | Hasil grid search 26 kombinasi parameter |
| `deteksi_per_krisis_dbscan.csv` | Detection rate per krisis historis |
| `top_negara_anomali_dbscan.csv` | Top 10 negara dengan anomali terbanyak |
| `feature_importance_dbscan.csv` | Feature importance (mean difference) |
| `distribusi_cluster_dbscan.csv` | Distribusi cluster DBSCAN |
| `hasil_dbscan.csv` | Dataset hasil deteksi (1.714 baris) |
| `ringkasan_model_dbscan.csv` | Ringkasan metrik model terbaik |
| `laporan_dbscan.md` | Laporan singkat ini |

### Kolom pada `hasil_dbscan.csv`:
- `economy`, `year` — Metadata negara dan tahun
- `crisis_label` — Ground truth (0=Normal, 1=Krisis)
- `dbscan_anomaly` — Prediksi DBSCAN (0=Normal/Cluster, 1=Anomali/Noise)
- `dbscan_cluster` — ID cluster DBSCAN (−1=Noise, 0,1,2,...=Cluster ID)
- 14 kolom fitur indikator ekonomi makro (z-score)

---

## Referensi

1. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96)*, 226–231.
2. Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). On the Surprising Behavior of Distance Metrics in High Dimensional Space. *Proceedings of the 8th International Conference on Database Theory (ICDT)*, 420–434.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN. *ACM Transactions on Database Systems*, 42(3), 1–21.

---

*Laporan ini merupakan bagian dari proyek **Deteksi Anomali pada Indikator Ekonomi Makro sebagai Early Warning System Krisis Ekonomi menggunakan Pendekatan Unsupervised Learning** — UTS Data Mining 2026.*
