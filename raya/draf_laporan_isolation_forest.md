# Draf Laporan: Deteksi Anomali Menggunakan Algoritma Isolation Forest

## Bagian: Isolation Forest untuk Deteksi Anomali pada Indikator Ekonomi Makro sebagai Early Warning System Krisis Ekonomi

---

## 1. Pendahuluan Algoritma Isolation Forest

### 1.1 Latar Belakang

Deteksi anomali (*anomaly detection*) merupakan cabang *machine learning* yang semakin berkembang pesat seiring meningkatnya volume dan kecepatan data (Chandola et al., 2009). Dalam konteks ekonomi, deteksi anomali pada indikator makroekonomi menjadi krusial sebagai komponen *Early Warning System* (EWS) untuk mengantisipasi krisis ekonomi sebelum terjadi dampak sistemik yang lebih luas.

Metode deteksi anomali tradisional seperti *density-based* (LOF), *distance-based* (k-NN), *neural network*, dan *spectral-based* pada mulanya dikembangkan untuk klasifikasi atau *clustering* dan baru kemudian dimodifikasi untuk deteksi anomali (Al Farizi et al., 2021). Pendekatan konvensional ini memiliki kelemahan fundamental: mereka dioptimalkan untuk **memprofilkan data normal**, bukan untuk **mendeteksi anomali** secara langsung. Konsekuensinya, hasil deteksi seringkali tidak optimal — menghasilkan terlalu banyak *false alarm* (data normal teridentifikasi sebagai anomali) atau terlalu sedikit anomali yang terdeteksi (Liu et al., 2008).

Isolation Forest (iForest), yang diperkenalkan oleh Liu, Ting, dan Zhou (2008), merupakan algoritma pertama yang secara eksplisit dirancang untuk deteksi anomali — bukan merupakan modifikasi dari algoritma lain. Pendekatan ini secara fundamental berbeda karena mengisolasi anomali secara langsung (*isolation-based*), bukan memprofilkan data normal terlebih dahulu.

### 1.2 Konsep Dasar: Isolasi

Dalam paper asli Liu et al. (2008), konsep **isolasi** didefinisikan sebagai "memisahkan suatu *instance* dari *instance* lainnya" (*separating an instance from the rest of the instances*). Prinsip kerja Isolation Forest didasarkan pada dua properti kuantitatif anomali:

1. **Anomali berjumlah sedikit** (*few*) — observasi anomali merepresentasikan proporsi kecil dari keseluruhan dataset, sehingga jumlah partisi yang diperlukan untuk mengisolasi mereka lebih sedikit.
2. **Anomali memiliki nilai atribut yang berbeda signifikan** (*different*) — observasi anomali memiliki karakteristik fitur yang menyimpang jauh dari sebagian besar data, sehingga lebih mudah dipisahkan pada tahap partisi awal.

Dengan kata lain, anomali bersifat "*few and different*" yang membuat mereka **lebih rentan terhadap isolasi** (*more susceptible to isolation*) dibanding data normal (Liu et al., 2008).

### 1.3 Mekanisme Kerja

#### 1.3.1 Konstruksi Isolation Tree (iTree)

Isolation Forest bekerja melalui konstruksi **pohon isolasi** (*isolation trees* atau *iTrees*) sebagai berikut:

**Definisi (Isolation Tree):** Sebuah *node* T pada iTree dapat berupa *external node* tanpa anak (*leaf*), atau *internal node* dengan satu *test* dan tepat dua *daughter nodes* (T_l, T_r). Setiap *test* terdiri dari atribut **q** dan nilai *split* **p** sehingga q < p membagi data ke T_l dan T_r (Liu et al., 2008).

Proses pembangunan iTree:
1. Pilih secara acak satu fitur **q** dari keseluruhan fitur.
2. Pilih secara acak satu nilai *split* **p** di antara nilai minimum dan maksimum fitur q pada data tersebut.
3. Partisi data: observasi dengan q < p masuk ke cabang kiri, sisanya ke cabang kanan.
4. Ulangi secara rekursif sampai: (a) kedalaman pohon mencapai batas *l* = ⌈log₂ψ⌉, (b) hanya tersisa 1 observasi, atau (c) semua data memiliki nilai identik.

#### 1.3.2 Path Length dan Anomaly Score

**Definisi (Path Length):** Path length *h(x)* dari sebuah titik *x* diukur berdasarkan jumlah *edge* (sisi) yang dilalui *x* dari *root node* hingga *terminating node* pada iTree.

Observasi anomali, karena memiliki nilai yang jauh berbeda dari mayoritas data, cenderung terisolasi lebih cepat — menghasilkan **path length pendek**. Sebaliknya, observasi normal yang memiliki karakteristik serupa satu sama lain memerlukan lebih banyak *split* untuk diisolasi — menghasilkan **path length panjang**.

Liu et al. (2008) mengilustrasikan bahwa untuk distribusi Gaussian dengan 135 titik data, sebuah titik normal x_i memerlukan 12 partisi untuk terisolasi, sedangkan anomali x_o hanya memerlukan 4 partisi. Rata-rata *path length* konvergen ke 12.82 (untuk x_i) dan 4.02 (untuk x_o) ketika menggunakan 1000 pohon.

#### 1.3.3 Normalisasi: Anomaly Score s(x, n)

Karena *path length* maksimum tumbuh dengan orde *n* sementara rata-rata tumbuh dengan orde log *n*, diperlukan normalisasi. Liu et al. (2008) menggunakan analogi dengan *Binary Search Tree* (BST):

**Rata-rata path length** pada BST untuk *n* instance:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

di mana H(i) = ln(i) + 0.5772156649 (konstanta Euler).

**Anomaly Score** didefinisikan sebagai:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

di mana E(h(x)) adalah rata-rata *path length* dari kumpulan iTrees.

Interpretasi anomaly score:
- Jika E(h(x)) → 0, maka **s → 1**: **pasti anomali**
- Jika E(h(x)) → c(n), maka **s → 0.5**: **tidak ada anomali yang jelas**
- Jika E(h(x)) → n − 1, maka **s → 0**: **pasti normal**

### 1.4 Penanganan Swamping dan Masking

Isolation Forest memiliki kemampuan unik dalam menangani dua masalah klasik dalam deteksi anomali (Liu et al., 2008):

- **Swamping**: Kesalahan mengidentifikasi data normal sebagai anomali. Terjadi ketika data normal terlalu dekat dengan anomali, sehingga jumlah partisi yang diperlukan meningkat.
- **Masking**: Keberadaan terlalu banyak anomali yang saling menutupi eksistensi mereka. Ketika kluster anomali besar dan padat, jumlah partisi untuk mengisolasi setiap anomali meningkat.

**Sub-sampling** menjadi kunci keunggulan Isolation Forest: dengan menggunakan sub-sampel yang kecil (ψ = 256), efek *swamping* dan *masking* berkurang secara signifikan. Liu et al. (2008) menunjukkan bahwa pada data Mulcross, penggunaan seluruh sampel (4096 instance) hanya menghasilkan AUC 0.67, sementara sub-sampling ψ = 128 mencapai AUC **0.91**.

### 1.5 Keunggulan untuk Deteksi Anomali Ekonomi Makro

Al Farizi et al. (2021) dalam *systematic literature review* menunjukkan bahwa IF merupakan algoritma pertama yang secara eksplisit dirancang untuk deteksi anomali. Evaluasi oleh Domingues et al. (2018) terhadap beberapa algoritma AD termasuk GMM, KDE, Mahalanobis Distance, LOF, One-class SVM, dan IF menunjukkan bahwa **IF unggul dalam akurasi dan waktu** sambil menunjukkan performa memuaskan pada dataset berdimensi tinggi.

Perbandingan Isolation Forest dengan metode deteksi anomali lainnya:

| Aspek | Isolation Forest | LOF / DBSCAN | K-Means | OCSVM |
|-------|-----------------|-------------|---------|-------|
| **Pendekatan** | Isolasi langsung | Profil densitas normal | Profil cluster normal | Profil batas normal |
| **Kompleksitas training** | O(t × ψ × log ψ) | O(n²) | O(n × k × d × i) | O(n² × d) |
| **Kompleksitas evaluasi** | O(n × t × log ψ) | O(n²) | O(n × k × d) | O(n × d) |
| **Dimensi tinggi** | Sangat baik | Menurun drastis | Menurun | Menurun |
| **Asumsi distribusi** | Tidak diperlukan | Densitas lokal | Cluster spherical | Distribusi tertentu |
| **Sub-sampling** | Meningkatkan performa | Tidak disarankan | Tidak disarankan | Tidak disarankan |
| **Parameter** | Minimal (2: t, ψ) | k (jumlah tetangga) | k (jumlah cluster) | ν, γ, kernel |

Liu et al. (2008) menunjukkan bahwa iForest secara konsisten mengungguli ORCA (distance-based) terutama pada dataset besar (n > 1000), dengan AUC lebih tinggi dan waktu eksekusi yang **jauh lebih cepat** — misalnya pada dataset HTTP (567.497 instance): iForest 15.58 detik vs ORCA 9.487 detik.

Dalam konteks dataset indikator ekonomi makro yang memiliki 14 fitur numerik, Isolation Forest menawarkan keunggulan khusus:

- **Toleransi terhadap heterojenitas ekonomi**: Data makroekonomi berasal dari berbagai negara (49 negara) dengan karakteristik ekonomi yang sangat beragam. Isolation Forest tidak memerlukan asumsi bahwa data normal membentuk cluster homogen.
- **Efektivitas pada fitur yang sudah distandardisasi**: Dataset telah melalui proses standardisasi (z-score), namun antar-fitur tetap memiliki korelasi dan distribusi yang bervariasi. Partisi acak pada Isolation Forest secara natural menangani variasi ini.
- **Deteksi krisis sebagai outlier multivariat**: Krisis ekonomi seringkali bermanifestasi sebagai kombinasi simultan dari penyimpangan pada beberapa indikator sekaligus — GDP kontraksi, inflasi tinggi, cadangan turun, pengangguran naik. Isolation Forest menangkap pola multivariat ini secara efisien tanpa perlu mendefinisikan threshold manual untuk setiap indikator.
- **Kompleksitas linier**: Dengan 1.714 observasi dan 14 fitur, Isolation Forest mampu membangun ensemble model dalam hitungan detik, memungkinkan eksplorasi *grid search* parameter secara menyeluruh.

---

## 2. Kelemahan Isolation Forest dan Mitigasi

Al Farizi et al. (2021) mengidentifikasi beberapa kelemahan utama Isolation Forest berdasarkan 17 studi yang ditinjau:

### 2.1 Akurasi Rendah pada Conditional Anomalies

Karena IF memilih fitur dan *split point* secara acak, performanya dapat menurun pada dataset di mana hanya beberapa fitur tertentu yang memengaruhi terjadinya anomali (Stripling et al., 2018). Pada dataset ekonomi makro, beberapa indikator mungkin lebih relevan untuk deteksi krisis (misalnya GDP_Growth dan Total_Reserves) dibanding lainnya.

**Mitigasi**: Dalam implementasi kami, seluruh 14 fitur digunakan (`max_features=1.0`) untuk memastikan anomali multivariat tertangkap. Analisis *feature importance* dilakukan pasca-training untuk mengidentifikasi indikator mana yang paling berkontribusi.

### 2.2 Bias Akibat Splitting Paralel

Hariri et al. (2019) menemukan bahwa splitting horizontal/vertikal standar pada IF dapat menyebabkan **ghost areas** — area dengan skor anomali tinggi yang sebenarnya bukan anomali. Solusi Extended Isolation Forest yang menggunakan arah splitting acak telah diusulkan.

**Mitigasi**: Penggunaan *ensemble* yang besar (n_estimators=200–500) membantu merata-ratakan efek bias ini.

### 2.3 Random Selection yang Tidak Efisien

Zou et al. (2019) menunjukkan bahwa setiap data memiliki bobot fitur yang berbeda, sehingga pemilihan fitur secara acak tidak selalu efektif. Beberapa peneliti mengusulkan metode Pre-IF (seleksi fitur/reduksi dimensi), Post-IF (reprocessing output IF), atau modifikasi algoritma IF itu sendiri.

**Mitigasi**: Dalam studi ini, kami menggunakan *grid search* pada `contamination` dan `n_estimators` untuk mengoptimalkan performa, serta mengevaluasi model dengan multiple metrics (F1-Score, ROC-AUC, Precision-Recall).

---

## 3. Hyperparameter yang Disetel

### 3.1 `n_estimators` — Jumlah Pohon Isolasi (t)

Parameter ini menentukan jumlah iTree dalam *ensemble*. Liu et al. (2008) menemukan bahwa "path lengths usually converge well before t = 100" dan merekomendasikan t = 100 sebagai nilai default. Namun, untuk memastikan stabilitas lebih tinggi, kami mengeksplorasi variasi: **50, 100, 200, 300, dan 500** pohon.

Studi empiris Liu et al. (2008) menggunakan dataset Arrhythmia (452 instance, 274 dimensi) dan Satellite (6.435 instance, 36 dimensi) menunjukkan bahwa AUC konvergen pada t yang kecil. Peningkatan t setelah konvergensi hanya menambah waktu komputasi tanpa peningkatan performa deteksi.

### 3.2 `contamination` — Proporsi Estimasi Anomali

Hyperparameter kritis yang menentukan proporsi observasi yang diharapkan sebagai anomali. Parameter ini secara langsung memengaruhi threshold *decision function* yang memisahkan *inlier* dari *outlier*. Empat variasi digunakan:

| Nilai | Rasionalisasi |
|-------|---------------|
| **`auto`** | Mode default scikit-learn berdasarkan paper asli |
| **0.10** | Asumsi konservatif (~10% anomali), standar umum dalam literatur AD |
| **0.15** | Nilai moderat antara asumsi konservatif dan estimasi sebenarnya |
| **~0.1919** | Diturunkan dari proporsi aktual crisis_label dalam dataset (~19.19%). Berfungsi sebagai *oracle reference* |

### 3.3 `max_samples` — Sub-sampling Size (ψ)

Liu et al. (2008) secara empiris menemukan bahwa **ψ = 256** umumnya memberikan detail yang cukup untuk deteksi anomali pada berbagai macam data. Performa deteksi mendekati optimal pada kisaran ini dan tidak sensitif terhadap variasi ψ yang lebar. Implementasi scikit-learn menggunakan `max_samples='auto'` yang menerapkan min(256, n_samples).

Batas kedalaman pohon otomatis dihitung sebagai l = ⌈log₂ψ⌉ ≈ 8 level. Rasionalnya: hanya data dengan *path length* lebih pendek dari rata-rata yang menarik untuk deteksi anomali (Liu et al., 2008).

### 3.4 `max_features`

Nilai `max_features=1.0` menggunakan seluruh 14 fitur pada setiap pohon untuk memastikan anomali multivariat (kombinasi penyimpangan pada beberapa indikator sekaligus) tertangkap secara komprehensif.

### 3.5 `random_state`

Nilai `random_state=42` digunakan secara konsisten untuk memastikan reprodusibilitas hasil.

---

## 4. Metodologi Implementasi

### 4.1 Persiapan Data

Dari 17 kolom pada dataset `data_cleaned.csv` (1.714 observasi), tiga kolom dieksklusi dari proses training:
- **`economy`**: Identitas negara (bukan fitur numerik).
- **`year`**: Penanda waktu (bukan indikator ekonomi).
- **`crisis_label`**: Ground truth yang disisihkan **hanya untuk evaluasi pasca-prediksi**.

Sehingga, **14 fitur numerik** indikator ekonomi makro yang sudah distandardisasi (z-score) digunakan: GDP_Growth, GDP_PerCapita_Growth, Inflation_CPI, Total_Reserves, Unemployment, Current_Account_GDP, Trade_GDP, FDI_Inflows_GDP, Exports_GDP, Imports_GDP, Gross_Savings_GDP, Exchange_Rate, Manufacturing_Value, dan Investment_GDP.

### 4.2 Pembagian Data

Data dibagi menjadi:
- **Training set (80%)**: Untuk melatih model Isolation Forest secara *unsupervised*.
- **Test set (20%)**: Untuk evaluasi *out-of-sample* yang lebih adil.
- Pembagian dilakukan dengan `stratify=crisis_label` untuk mempertahankan proporsi kelas.

### 4.3 Grid Search

Berbeda dari pendekatan *single model*, kami melakukan **grid search** pada kombinasi hyperparameter:
- 4 nilai `contamination` × 5 nilai `n_estimators` = **20 model** dilatih dan dievaluasi.
- Model terbaik dipilih berdasarkan **F1-Score** pada seluruh dataset.

### 4.4 Penyelarasan Label

Output prediksi Isolation Forest diselaraskan agar sejajar dengan konvensi `crisis_label`:

| Output Isolation Forest | crisis_label | Interpretasi |
|------------------------|-------------|-------------|
| −1 (outlier) | 1 | Anomali / Krisis |
| +1 (inlier) | 0 | Normal |

### 4.5 Evaluasi Komprehensif

Evaluasi dilakukan menggunakan:
- **Classification Report**: Precision, recall, dan F1-score per kelas.
- **Confusion Matrix**: Distribusi TP, FP, TN, dan FN untuk 4 model representatif.
- **ROC-AUC**: Area Under ROC Curve — mengukur kemampuan diskriminatif model. Liu et al. (2008) menggunakan AUC sebagai metrik utama dalam evaluasi mereka.
- **Precision-Recall Curve & Average Precision**: Lebih informatif untuk dataset imbalanced (krisis ≈ 19.2%).
- **Analisis Anomaly Score**: Distribusi skor untuk kelas normal vs krisis.
- **Feature Importance**: Identifikasi indikator yang paling membedakan anomali dari normal.
- **Timeline Temporal**: Analisis temporal untuk mendeteksi pola krisis historis.
- **Reduksi Dimensi (PCA & t-SNE)**: Visualisasi proyeksi 2D untuk validasi visual pemisahan anomali.

---

## 5. Analisis dan Temuan

### 5.1 Grid Search

Grid search pada 20 kombinasi hyperparameter menunjukkan bahwa:
- **Contamination** sangat memengaruhi *trade-off precision-recall*: nilai yang lebih tinggi meningkatkan *recall* (mendeteksi lebih banyak krisis) namun menurunkan *precision* (lebih banyak *false alarm*).
- **n_estimators** memberikan stabilitas setelah ~200 pohon, konsisten dengan temuan Liu et al. (2008) yang melaporkan konvergensi AUC pada t ≤ 100.
- Nilai `contamination` mendekati rasio krisis aktual (~0.1919) memberikan **F1-Score terbaik**, yang masuk akal karena *threshold* sesuai dengan prevalensi anomali sesungguhnya.

### 5.2 Feature Importance

Analisis *feature importance* berbasis *anomaly score differential* mengungkapkan indikator-indikator yang paling berkontribusi dalam membedakan pola anomali vs normal. Temuan ini memberikan *insight* mengenai indikator mana yang paling sensitif terhadap kondisi krisis.

### 5.3 Analisis Temporal

Timeline anomali menunjukkan **spike** anomaly score pada tahun-tahun krisis historis:
- **1997–1998**: Asian Financial Crisis
- **2008–2009**: Global Financial Crisis
- **2020**: COVID-19 Pandemic

Pola ini mengkonfirmasi bahwa Isolation Forest mampu menangkap sinyal krisis tanpa informasi label yang eksplisit.

### 5.4 Reduksi Dimensi

Proyeksi PCA dan t-SNE memvisualisasikan bahwa anomali yang terdeteksi cenderung berada di **daerah periferi** ruang fitur — konsisten dengan prinsip kerja Isolation Forest di mana titik-titik dengan path length pendek (anomali) terletak di area yang lebih terisolasi dari cluster utama data.

---

## 6. Catatan Penting

Perlu ditekankan bahwa evaluasi terhadap `crisis_label` bersifat **post-hoc** dan bertujuan untuk mengukur sejauh mana pola anomali yang terdeteksi secara *unsupervised* bersesuaian dengan kejadian krisis yang telah diidentifikasi. Dalam aplikasi *Early Warning System* sesungguhnya:

1. **Label krisis tidak tersedia secara real-time** — anomaly score dari Isolation Forest berfungsi sebagai sinyal peringatan dini yang perlu divalidasi lebih lanjut oleh pakar ekonomi.
2. **Deteksi bersifat unsupervised** — model dilatih tanpa informasi label, membuatnya applicable pada data makroekonomi baru tanpa perlu pelabelan manual.
3. **Anomali ≠ Krisis dengan pasti** — skor anomali yang tinggi mengindikasikan kondisi ekonomi yang **tidak biasa** dan memerlukan investigasi lebih lanjut, bukan prediksi definitif krisis.

---

## Referensi

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM)*, 413–422. DOI: 10.1109/ICDM.2008.17
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). Isolation-Based Anomaly Detection. *ACM Transactions on Knowledge Discovery from Data*, 6(1), 1–39. DOI: 10.1145/2133360.2133363
3. Al Farizi, W. S., Hidayah, I., & Rizal, M. N. (2021). Isolation Forest Based Anomaly Detection: A Systematic Literature Review. *Proceedings of the 8th International Conference on Information Technology, Computer and Electrical Engineering (ICITACEE)*, 118–122. DOI: 10.1109/ICITACEE53184.2021.9617498
4. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1–58. DOI: 10.1145/1541880.1541882
5. Domingues, R., Filippone, M., Michiardi, P., & Zouaoui, J. (2018). A Comparative Evaluation of Outlier Detection Algorithms. *Pattern Recognition*, 74, 406–421. DOI: 10.1016/j.patcog.2017.09.037
6. Hariri, S., Carrasco Kind, M., & Brunner, R. J. (2019). Extended Isolation Forest. *IEEE Transactions on Knowledge and Data Engineering*. DOI: 10.1109/TKDE.2019.2947676
7. Stripling, E., Baesens, B., Chizi, B., & vanden Broucke, S. (2018). Isolation-based Conditional Anomaly Detection on Mixed-attribute Data. *Decision Support Systems*, 111, 13–26. DOI: 10.1016/j.dss.2018.04.001
8. Zou, Z., Xie, Y., Huang, K., Xu, G., Feng, D., & Long, D. (2019). A Docker Container Anomaly Monitoring System Based on Optimized Isolation Forest. *IEEE Transactions on Cloud Computing*. DOI: 10.1109/TCC.2019.2935724
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
