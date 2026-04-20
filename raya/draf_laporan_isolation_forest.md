# Draf Laporan: Deteksi Anomali Menggunakan Algoritma Isolation Forest

## Bagian: Isolation Forest untuk Deteksi Anomali pada Indikator Ekonomi Makro

---

## 1. Pendahuluan Algoritma Isolation Forest

### 1.1 Konsep Dasar

Isolation Forest, yang diperkenalkan oleh Liu, Ting, dan Zhou (2008), merupakan algoritma deteksi anomali berbasis *ensemble learning* yang mengadopsi pendekatan fundamentally berbeda dari metode deteksi anomali konvensional. Alih-alih membangun profil dari data normal kemudian mengidentifikasi penyimpangan, Isolation Forest secara eksplisit mengisolasi observasi anomali melalui mekanisme partisi acak (*random partitioning*).

Prinsip kerja Isolation Forest didasarkan pada dua asumsi kunci mengenai sifat data anomali:

1. **Anomali berjumlah sedikit** (*minority*) — observasi anomali merepresentasikan proporsi kecil dari keseluruhan dataset.
2. **Anomali memiliki nilai atribut yang berbeda signifikan** (*different*) — observasi anomali memiliki karakteristik fitur yang menyimpang jauh dari sebagian besar data.

### 1.2 Mekanisme Kerja

Isolation Forest bekerja melalui konstruksi pohon isolasi (*isolation trees* atau *iTrees*) sebagai berikut:

1. **Pembangunan iTree**: Untuk setiap pohon isolasi, algoritma secara acak memilih satu fitur dan satu nilai *split* di antara rentang minimum dan maksimum fitur tersebut. Proses ini diulang secara rekursif hingga setiap observasi terisolasi atau kedalaman maksimum tercapai.

2. **Path Length (Panjang Jalur)**: Jumlah *split* yang diperlukan untuk mengisolasi suatu observasi disebut *path length*. Observasi anomali, karena memiliki nilai yang berbeda signifikan, cenderung terisolasi lebih cepat (path length pendek). Sebaliknya, observasi normal yang berdekatan satu sama lain memerlukan lebih banyak *split* untuk diisolasi (path length panjang).

3. **Ensemble dan Anomaly Score**: Sejumlah *t* pohon isolasi dibangun secara independen, dan *path length* rata-rata dari seluruh pohon dinormalisasi menjadi *anomaly score* dalam rentang [0, 1]. Skor mendekati 1 mengindikasikan anomali, sedangkan skor mendekati 0.5 atau lebih rendah mengindikasikan observasi normal.

### 1.3 Keunggulan untuk Deteksi Anomali Ekonomi Multivariat

Isolation Forest memiliki beberapa keunggulan signifikan yang menjadikannya sangat cocok untuk konteks deteksi anomali pada data indikator ekonomi makro multivariat, dibandingkan algoritma *clustering* biasa seperti K-Means atau DBSCAN:

| Aspek | Isolation Forest | Algoritma Clustering |
|-------|-----------------|---------------------|
| **Fokus** | Secara langsung mengarah pada isolasi anomali | Memprofilkan data normal terlebih dahulu |
| **Kompleksitas** | O(t × n × log n), sangat efisien | K-Means: O(n × k × d × i); DBSCAN: O(n²) untuk high-dim |
| **Dimensionalitas** | Robust terhadap *curse of dimensionality* | Performa degradasi signifikan pada dimensi tinggi |
| **Asumsi distribusi** | Tidak memerlukan asumsi distribusi | K-Means mengasumsikan cluster spherical |
| **Penentuan parameter** | Minimal (*contamination*, *n_estimators*) | Memerlukan penentuan jumlah cluster atau parameter epsilon |
| **Anomali global vs lokal** | Mendeteksi anomali global secara efektif | Bergantung pada struktur cluster |

Dalam konteks dataset indikator ekonomi makro yang memiliki 14 fitur numerik (GDP Growth, Inflation CPI, Unemployment, dll.), Isolation Forest menawarkan keunggulan berikut:

- **Toleransi terhadap heterojenitas ekonomi**: Data makroekonomi berasal dari berbagai negara dengan karakteristik ekonomi yang sangat beragam. Isolation Forest tidak memerlukan asumsi bahwa data normal membentuk cluster homogen.
- **Efektivitas pada fitur yang sudah distandardisasi**: Dataset telah melalui proses standardisasi (z-score), namun antar-fitur tetap memiliki korelasi dan distribusi yang bervariasi. Partisi acak pada Isolation Forest secara natural menangani variasi ini.
- **Deteksi krisis sebagai outlier multivariat**: Krisis ekonomi seringkali bermanifestasi sebagai kombinasi simultan dari penyimpangan pada beberapa indikator sekaligus — GDP kontraksi, inflasi tinggi, cadangan turun, pengangguran naik. Isolation Forest menangkap pola multivariat ini secara efisien tanpa perlu mendefinisikan threshold manual untuk setiap indikator.

---

## 2. Hyperparameter yang Disetel

### 2.1 `n_estimators` (Jumlah Pohon Isolasi)

Parameter ini menentukan jumlah pohon isolasi (*iTrees*) dalam ensemble. Semakin banyak pohon, semakin stabil estimasi anomaly score, namun dengan biaya komputasi yang meningkat. Dalam eksperimen ini, digunakan variasi nilai: 50, 100, 200, 300, dan 500 pohon. Berdasarkan literatur, nilai 100–200 pohon umumnya sudah cukup untuk mencapai konvergensi anomaly score (Liu et al., 2008). Nilai default `n_estimators=200` digunakan untuk perbandingan utama.

### 2.2 `contamination` (Proporsi Estimasi Anomali)

Hyperparameter kritis yang menentukan proporsi observasi yang diharapkan sebagai anomali dalam dataset. Parameter ini secara langsung memengaruhi threshold decision function yang memisahkan inlier dari outlier. Empat variasi contamination digunakan dalam eksperimen:

- **`auto`**: Mode default scikit-learn yang menggunakan offset threshold berdasarkan paper asli.
- **0.10**: Asumsi konservatif bahwa sekitar 10% data mengandung anomali.
- **0.15**: Nilai moderat di antara asumsi konservatif dan estimasi sebenarnya.
- **0.1919 (rasio krisis)**: Nilai yang diturunkan dari proporsi aktual label krisis dalam dataset (~19.19%). Meskipun informasi ini secara teknis tidak tersedia dalam skenario *unsupervised* murni, penggunaannya berfungsi sebagai batas atas (*upper bound*) atau referensi oracle untuk evaluasi.

### 2.3 `max_samples` dan `max_features`

- **`max_samples='auto'`**: Menggunakan min(256, n_samples) sebagai jumlah sub-sampel untuk membangun setiap pohon, sesuai rekomendasi paper asli.
- **`max_features=1.0`**: Seluruh fitur digunakan dalam pembangunan setiap pohon untuk memastikan anomali multivariat tertangkap secara komprehensif.

### 2.4 `random_state`

Nilai `random_state=42` digunakan secara konsisten untuk memastikan reprodusibilitas hasil eksperimen.

---

## 3. Metodologi Implementasi

### 3.1 Persiapan Data

Dari 17 kolom pada dataset `data_cleaned.csv`, tiga kolom dieksklusi dari proses training model:
- **`economy`**: Identitas negara (bukan fitur numerik).
- **`year`**: Penanda waktu (bukan indikator ekonomi).
- **`crisis_label`**: Ground truth yang disisihkan hanya untuk evaluasi pasca-prediksi.

Sehingga, 14 fitur numerik indikator ekonomi makro digunakan sebagai input model: GDP_Growth, GDP_PerCapita_Growth, Inflation_CPI, Total_Reserves, Unemployment, Current_Account_GDP, Trade_GDP, FDI_Inflows_GDP, Exports_GDP, Imports_GDP, Gross_Savings_GDP, Exchange_Rate, Manufacturing_Value, dan Investment_GDP.

### 3.2 Training dan Prediksi

Model Isolation Forest dilatih pada seluruh 1.714 observasi secara *unsupervised*. Untuk setiap variasi contamination, model menghasilkan:
- **Prediksi label**: −1 (anomali) atau +1 (inlier).
- **Anomaly score**: Skor kontinu dari `decision_function()`, di mana semakin negatif mengindikasikan semakin anomali.

### 3.3 Penyelarasan Label

Output prediksi Isolation Forest diselaraskan agar sejajar dengan konvensi `crisis_label`:

| Output Isolation Forest | crisis_label | Interpretasi |
|------------------------|-------------|-------------|
| −1 (outlier) | 1 | Anomali / Krisis |
| +1 (inlier) | 0 | Normal |

### 3.4 Evaluasi

Evaluasi dilakukan menggunakan:
- **Classification Report**: Precision, recall, dan F1-score per kelas.
- **Confusion Matrix**: Distribusi True Positive, False Positive, True Negative, dan False Negative.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve, mengukur kemampuan diskriminatif model secara keseluruhan.
- **Precision-Recall Curve**: Lebih informatif daripada ROC untuk dataset dengan ketidakseimbangan kelas.

---

## 4. Catatan Penting

Perlu ditekankan bahwa evaluasi terhadap `crisis_label` bersifat **post-hoc** dan bertujuan untuk mengukur sejauh mana pola anomali yang terdeteksi secara *unsupervised* bersesuaian dengan kejadian krisis yang telah diidentifikasi. Dalam aplikasi *Early Warning System* sesungguhnya, label krisis tidak tersedia secara real-time, dan anomaly score dari Isolation Forest berfungsi sebagai sinyal peringatan dini yang perlu divalidasi lebih lanjut oleh pakar ekonomi.

---

## Referensi

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM)*, 413–422.
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). Isolation-Based Anomaly Detection. *ACM Transactions on Knowledge Discovery from Data*, 6(1), 1–39.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
