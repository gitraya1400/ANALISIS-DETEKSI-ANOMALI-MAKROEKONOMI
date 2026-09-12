# 📍 Laporan — Deteksi Anomali Ekonomi Makro dengan Local Outlier Factor (LOF) v2

**Disusun oleh:** Amir
**Tanggal:** 12 September 2026
**Metode:** Local Outlier Factor (Density-Based — Unsupervised Anomaly Detection), **fully label-free hyperparameter selection**

---

## 1. Ringkasan Eksekutif

Laporan ini adalah revisi metodologis dari `laporan_lof.md` (versi lama), dibangun dari dataset mentah baru (`raw_data_master.csv`) dan ground truth eksternal baru (`ground_truth_imf.csv`), dengan memperbaiki masalah utama yang diidentifikasi pada versi lama: **klaim "unsupervised murni" tidak konsisten dengan implementasi**, karena `n_neighbors` dipilih lewat grid search yang dievaluasi memakai AUC-ROC terhadap `crisis_label`, dan `contamination` diset sama persis dengan rasio krisis aktual (0,1919) — keduanya bentuk penggunaan label, hanya bukan untuk *fitting* model.

**Perbaikan pada versi ini:**
1. `n_neighbors` **tidak dipilih** dari satu nilai lewat label — dipakai rentang `{5, 10, 15, 20, 30, 50}` (nilai yang sama seperti rentang grid search di proposal final kelompok Bab III §3.2.3), diagregasi lewat **maksimum skor LOF across rentang k** tersebut, mengikuti rekomendasi asli Breunig et al. (2000) sendiri — bukan diseleksi berdasarkan AUC-ROC terhadap label.
2. `contamination` diganti dengan threshold tetap `anomaly_score > 1,5`, yaitu aturan `contamination='auto'` bawaan scikit-learn (berbasis interpretasi skor LOF Breunig sendiri: LOF ≈ 1 normal, LOF ≫ 1 anomali) — bukan diturunkan dari rasio krisis.
3. `crisis_label` sekarang dibangun dari ground truth eksternal (`ground_truth_imf.csv`, style IMF/Laeven-Valencia: banking/currency/sovereign-debt crisis + guncangan eksogen COVID-19), bukan aturan enam-episode buatan sendiri yang menghasilkan angka tidak konsisten (179 vs 329) di versi lama.
4. Ditambahkan **robustness check** lewat stratified resampling (`random_state=42`) untuk membuktikan hasil utama bukan kebetulan satu kali fit.

Dengan perubahan ini, LOF sekarang berada pada level yang **sama dengan DBSCAN** dalam hal seberapa banyak informasi label yang dipakai sebelum evaluasi: **nol**. Ini menjawab langsung kritik bahwa perbandingan enam algoritma sebelumnya tidak apple-to-apple karena level penggunaan label yang berbeda-beda dan tidak dijelaskan eksplisit.

---

## 2. Konsep Local Outlier Factor

Tidak berubah dari versi lama — LOF (Breunig et al., 2000) mengukur derajat anomali suatu titik data secara **relatif terhadap kerapatan lokal** tetangga-tetangganya:

1. **k-distance(p)**: Jarak dari titik p ke tetangga ke-k terdekatnya
2. **Reachability Distance**: `reach-dist_k(p,o) = max(k-dist(o), d(p,o))`
3. **Local Reachability Density (LRD)**: Kebalikan dari rata-rata *reachability distance* terhadap k-tetangga
4. **LOF Score**: Rasio rata-rata LRD tetangga / LRD titik itu sendiri

- **LOF ~ 1** => Normal. **LOF >> 1** => Anomali.

---

## 3. Dataset

| Aspek | Detail |
|-------|--------|
| **Sumber fitur** | `Amir/raw_data_master.csv` (World Bank) |
| **Sumber ground truth** | `Amir/ground_truth_imf.csv` (banking/currency/sovereign-debt crisis + COVID-19) |
| **Observasi** | 1.715 baris (49 negara × 35 tahun), **tanpa baris yang di-drop** |
| **Periode** | 1990–2024 |
| **Fitur** | 14 indikator (13 asli + `Exchange_Depreciation` menggantikan `Exchange_Rate` nominal), distandardisasi `RobustScaler` |
| **Ground Truth** | 229 observasi krisis (13,4%) — bukan 19,2%/329 (versi lama) atau 10,2%/175 (versi antara sebelum ground truth eksternal ditemukan) |

14 fitur: `GDP_Growth`, `Inflation_CPI`, `Unemployment`, `Current_Account_GDP`, `Reserves_Months_Imports`, `FDI_Inflows_GDP`, `Exports_GDP`, `Imports_GDP`, `Gross_Savings_GDP`, `Investment_GDP`, `Manufacturing_Value`, `Domestic_Credit_GDP`, `Broad_Money_Growth`, `Exchange_Depreciation`. Detail preprocessing (interpolasi per negara, `KNNImputer` fallback untuk 14 pasangan negara-fitur yang datanya kosong total, `RobustScaler`) ada di `Amir/preproccesingDataAmir.ipynb`.

---

## 4. Konfigurasi Model — Label-Free

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| `n_neighbors` | Rentang `{5, 10, 15, 20, 30, 50}`, diagregasi via **MAX** | Breunig et al. (2000); rentang sama seperti proposal Bab III §3.2.3, agregasi berbeda |
| `contamination` | Threshold tetap `anomaly_score > 1,5` | `contamination='auto'` scikit-learn |
| `metric` | Euclidean (Minkowski p=2) | — |
| Mode | `novelty=False` (transductive, fit sekali di seluruh 1.715 baris) | Konsisten dengan cara Breunig menerapkan LOF dan cara lima algoritma lain di proyek kelompok dijalankan |
| `random_state` | 42 (dipakai di tahap robustness check, §7) | Aturan wajib kelompok |

`crisis_label` **tidak disentuh** di tahap manapun pada tabel ini — hanya dipakai di §5 dan §7, setelah skor dan prediksi sudah selesai dihitung.

---

## 5. Hasil Evaluasi Utama

| Metrik | Nilai |
|--------|-------|
| Precision | 0,2456 |
| Recall | 0,1834 |
| F1-Score | 0,2100 |
| ROC-AUC | 0,7099 |
| Average Precision | 0,2427 |

**Confusion Matrix:**

|  | Prediksi: Normal | Prediksi: Anomali |
|--|---|---|
| **Aktual: Normal** | TN = 1.357 | FP = 129 |
| **Aktual: Krisis** | FN = 187 | TP = 42 |

`predicted_anomaly = 1` untuk 171 baris (10,0% dari 1.715) — bukan diset sama dengan rasio krisis aktual seperti versi lama, murni konsekuensi dari threshold `1,5` yang label-free.

---

## 6. Deteksi per Krisis Historis

Dari `ground_truth_imf.csv`: 229 baris krisis terbagi ke **28 episode bernama** dan **22 baris krisis tanpa nama episode spesifik** (banking/currency/sovereign-debt crisis individual negara yang tidak diberi label naratif di sumber data). Episode dengan observasi terbanyak:

| Krisis | Observasi | Terdeteksi | Detection Rate |
|--------|-----------|------------|-----------------|
| COVID-19 Global Shock | 49 | 14 | 28,6% |
| Global Financial Crisis | 47 | 3 | 6,4% |
| Asian Financial Crisis | 19 | 6 | 31,6% |
| Nigerian Banking Crisis | 9 | 2 | 22,2% |
| Transition Banking Crisis | 8 | 0 | 0,0% |
| Argentine Crisis | 3 | 2 | 66,7% |
| Collor Plan / Banking Crisis | 4 | 4 | 100,0% |

Pola konsisten dengan versi lama: krisis regional/klaster (Asian Financial Crisis, Argentine Crisis, Collor Plan) terdeteksi lebih baik daripada krisis global merata (GFC 6,4%, COVID-19 28,6%) — sesuai prinsip kerja LOF yang menilai keanomalian relatif terhadap tetangga lokal, bukan populasi global.

---

## 7. Top 10 Negara dengan Anomali Terbanyak

| Negara | Jumlah Anomali |
|--------|-----------------|
| SAU | 14 |
| HUN | 11 |
| NLD | 9 |
| IRL | 9 |
| RUS | 9 |
| NGA | 7 |
| BRA | 6 |
| CHE | 6 |
| ARG | 6 |
| ROU | 6 |

Konsisten dengan versi lama: negara komoditas/migas (Arab Saudi) dan ekonomi sangat terbuka (Irlandia, Belanda) tetap dominan.

---

## 8. Robustness Check — Stratified Resampling

20 subsample stratified 80% (`random_state=42`), prosedur label-free yang identik (§4) diulang pada tiap subsample:

| Metrik | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| AUC-ROC | 0,7068 | 0,0106 | 0,6846 | 0,7264 |
| F1-Score | 0,2265 | 0,0153 | 0,2063 | 0,2620 |

Sebaran sangat ketat (std ≈ 0,01 pada AUC-ROC) — hasil utama di §5 **bukan kebetulan satu kali fit**, melainkan stabil across subsample data yang berbeda-beda.

---

## 9. Analisis dan Interpretasi

**Perbandingan jujur dengan versi lama:**

| | Versi Lama | Versi v2 (Label-Free) |
|---|---|---|
| F1-Score | 0,3435 | 0,2100 (primer) / 0,2265±0,0153 (resampled) |
| ROC-AUC | 0,6794 | 0,7099 |
| `n_neighbors` | Dipilih via AUC-ROC terhadap label | Rentang tetap, diagregasi MAX, tanpa label |
| `contamination` | = rasio krisis aktual (0,1919) | Threshold tetap 1,5 (`auto`) |

F1 versi lama **lebih tinggi**, tapi diukur dari parameter yang di-tuning pada label yang sama dengan yang dilaporkan — angka itu mengukur seberapa pas parameter di-fit ke label, bukan generalisasi. F1 versi ini **lebih rendah tapi jujur**: parameter dan threshold ditentukan sepenuhnya tanpa melihat `crisis_label`, dan ROC-AUC (metrik yang tidak bergantung pada satu ambang batas) justru **lebih tinggi** (0,7099 vs 0,6794) — menunjukkan skor kontinu LOF versi ini sebenarnya lebih diskriminatif, meskipun ambang batas biner yang label-free menghasilkan F1 yang lebih rendah.

**Kekuatan:**
1. Sepenuhnya label-free di tahap tuning dan threshold — level yang sama dengan DBSCAN.
2. Robustness check membuktikan hasil stabil, bukan artefak satu fit.
3. Ground truth eksternal (IMF-style) lebih kredibel dan lebih terperinci per negara dibanding aturan enam-episode buatan sendiri.

**Keterbatasan:**
1. F1/Precision/Recall primer lebih rendah dari versi lama — ini konsekuensi yang diharapkan dari menghapus label leakage, bukan regresi kualitas model.
2. GFC (krisis global merata) tetap sulit dideteksi (6,4%) — konsisten dengan sifat LOF sebagai detektor kerapatan lokal.
3. **Metodologi ini menyimpang dari proposal final kelompok Bab III** — §3.2.1 mendokumentasikan `StandardScaler` untuk keenam algoritma (di sini dipakai `RobustScaler` untuk LOF), dan §3.2.3 mendokumentasikan grid search `n_neighbors` via AUC-ROC terhadap `crisis_label` (di sini label-free). Ini keputusan sadar Amir untuk memperbaiki masalah metodologis yang teridentifikasi; proposal akan direvisi menyusul untuk mencerminkan pendekatan ini — bukan penyimpangan diam-diam.

---

## 10. Rekomendasi

- LOF label-free ini **lebih defensibel secara metodologis** untuk diklaim sebagai "unsupervised murni" dibanding versi lama, dan sejajar dengan DBSCAN dalam perbandingan enam algoritma.
- Hasil deteksi LOF versi ini yang dipakai untuk tahap **Majority Voting** tingkat proyek kelompok, bukan versi lama.
- Proposal Bab III §3.2.1 dan §3.2.3 perlu direvisi agar konsisten dengan metodologi yang benar-benar dipakai.

---

## 11. File Output

| File | Deskripsi |
|------|-----------|
| `preproccesingDataAmir.ipynb` | Preprocessing: `raw_data_master.csv` → `data_cleaned_v2.csv` |
| `data_cleaned_v2.csv` | 1.715 baris, 14 fitur `RobustScaler`, `crisis_label` |
| `lof_v2.ipynb` | Notebook modeling lengkap (kode dan hasil di atas) |
| `hasil_lof_v2.csv` | Kolom `['economy', 'year', 'anomaly_score', 'predicted_anomaly']` |
| `grid_search_lof_v2.csv` | Diagnostik sweep per-k (bukan grid search berbasis label) |
| `ringkasan_model_lof_v2.csv` | Ringkasan satu baris: konfigurasi + metrik utama + robustness |
| `laporan_lof_v2.md` | Laporan ini |

---

## Referensi

1. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 93–104.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Goldstein, M., & Uchida, S. (2016). A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data. *PLOS ONE*, 11(4).
4. Laeven, L., & Valencia, F. (2018). Systemic banking crises revisited. *IMF Working Paper*, WP/18/206. (dasar taksonomi `ground_truth_imf.csv`)

---

*Laporan ini merupakan revisi metodologis individu (metode Local Outlier Factor) dalam proyek kelompok **Deteksi Anomali pada Indikator Ekonomi Makro sebagai Early Warning System Krisis Ekonomi menggunakan Pendekatan Unsupervised Learning**.*
