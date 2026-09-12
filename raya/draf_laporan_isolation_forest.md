# Draf Laporan: Deteksi Anomali Menggunakan Algoritma Isolation Forest

## Bagian: Isolation Forest untuk Deteksi Anomali pada Indikator Ekonomi Makro sebagai Early Warning System Krisis Ekonomi

**Disusun oleh:** M. Rezky Raya Kilwouw (222313190)  
**Program Studi:** DIV Komputasi Statistik — Kelas 3SI1  
**Mata Kuliah:** Data Mining (Tugas UTS 2026)  

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

**Sub-sampling** menjadi kunci keunggulan Isolation Forest: dengan menggunakan sub-sampel yang kecil (ψ = 256), efek *swamping* dan *masking* berkurang secara signifikan.

---

## 2. Metodologi Implementasi & Data

### 2.1 Sumber Data dan Transformasi Fitur

Penelitian ini menggunakan **Master Raw Data** resmi yang ditarik langsung dari **World Bank Open Data API** (`raw_data_master.csv`) yang mencakup **49 negara** dalam rentang waktu **1990–2024** (35 tahun = 1.715 baris data panel).

Untuk menghindari distorsi skala dan redundansi, dilakukan rekayasa fitur (*feature engineering*):
1. **Transformasi Kurs Valuta**: Nilai tukar nominal resmi (LCU/USD) diubah menjadi **persentase depresiasi tahunan (% YoY)**:
   $$\text{Exchange\_Depreciation}_{i,t} = \frac{\text{Rate}_{i,t} - \text{Rate}_{i,t-1}}{\text{Rate}_{i,t-1}} \times 100\%$$
   sehingga nilai tukar bernilai komparabel lintas-negara tanpa bias nominal mata uang.
2. **Cadangan Devisa**: Menggunakan indikator ketahanan cadangan devisa dalam hitungan bulan impor (**`Reserves_Months_Imports`** / `FI.RES.TOTL.MO`), menggantikan nilai nominal dolar AS mentah.
3. **14 Fitur Makroekonomi**:
   `GDP_Growth`, `Inflation_CPI`, `Unemployment`, `Current_Account_GDP`, `Reserves_Months_Imports`, `Exchange_Depreciation`, `FDI_Inflows_GDP`, `Exports_GDP`, `Imports_GDP`, `Gross_Savings_GDP`, `Investment_GDP`, `Manufacturing_Value`, `Domestic_Credit_GDP`, `Broad_Money_Growth`.
4. **Imputasi Data Panel**: Interpolasi linear per-negara untuk mempertahankan aspek sekuensial temporal deret waktu, dilanjutkan `KNNImputer(k=5)` untuk sisa sel kosong tanpa menghapus satu pun baris (tepat 1.715 observasi utuh).
5. **Standarisasi**: Menggunakan `StandardScaler` (z-score) pada seluruh 14 indikator.

### 2.2 Tolak Ukur Evaluasi (Ground Truth IMF)

Model Isolation Forest dilatih secara **Murni Unsupervised** pada matriks fitur $X$ tanpa melihat label apa pun. Hasil prediksi model kemudian divalidasi secara eksternal (*post-hoc external benchmarking*) menggunakan database resmi **IMF Systemic Banking Crises Database (Laeven & Valencia, 2018)** yang mencakup krisis perbankan sistemik, krisis mata uang, dan *default* utang negara, ditambah peristiwa syok resesi global **Pandemi COVID-19 2020** (`ground_truth_imf.csv`). Total tercatat 229 kejadian krisis nyata (**13.4%** prevalensi krisis).

---

## 3. Hasil Eksperimen & Analisis

### 3.1 Grid Search Hyperparameter

Eksplorasi dilakukan terhadap kombinasi parameter `contamination` (tingkat anomali) dan `n_estimators` (jumlah pohon isolasi):

| Contamination | n_estimators | TP | FP | TN | FN | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **0.15** | **100** | **39** | **219** | **1.267** | **190** | **0.1512** | **0.1703** | **0.1602** | **0.5250** |
| 0.15 | 50 | 38 | 220 | 1.266 | 191 | 0.1473 | 0.1659 | 0.1561 | 0.5250 |
| 0.1335 (Rasio IMF) | 100 | 35 | 194 | 1.292 | 194 | 0.1528 | 0.1528 | 0.1528 | 0.5250 |
| 0.10 | 100 | 28 | 144 | 1.342 | 201 | 0.1628 | 0.1223 | 0.1397 | 0.5250 |
| 0.08 | 100 | 21 | 116 | 1.370 | 208 | 0.1533 | 0.0917 | 0.1148 | 0.5250 |
| auto | 100 | 22 | 150 | 1.336 | 207 | 0.1279 | 0.0961 | 0.1097 | 0.5250 |

Konfigurasi `contamination=0.15` dan `n_estimators=100` menghasilkan nilai F1-Score tertinggi (0.1602) dengan kemampuan menjaring 39 observasi krisis riil.

### 3.2 Deteksi pada Episode Krisis Historis Dunia

Pengujian kemampuan model terhadap 6 episode guncangan krisis besar yang tercatat dalam sejarah dunia:

| Episode Krisis Dunia | Periode Tahun | Negara Terdampak | Total Observasi | Terdeteksi Anomali | Detection Rate (%) | Mean Anomaly Score |
|---|---|---|---|---|---|---|
| **Krisis Keuangan Asia** | 1997–1998 | IDN, THA, MYS, KOR, PHL | 10 | 6 | **60.0%** | +0.0076 |
| **Krisis Keuangan Rusia** | 1998 | RUS | 1 | 1 | **100.0%** | +0.0009 |
| **Krisis Argentina** | 2001–2002 | ARG | 2 | 1 | **50.0%** | +0.0519 |
| **Krisis Utang Eropa** | 2010–2012 | GRC, PRT, IRL, ESP, ITA | 15 | 6 | **40.0%** | -0.0079 |
| **Pandemi COVID-19** | 2020 | Global (49 negara) | 49 | 14 | **28.6%** | -0.0202 |
| **Global Financial Crisis (GFC)** | 2008–2009 | Global (49 negara) | 98 | 15 | **15.3%** | -0.0358 |

**Temuan Kunci:**
* Isolation Forest sangat sensitif terhadap **krisis nilai tukar dan guncangan mendadak yang terkonsentrasi**: Krisis Asia 1997–1998 terdeteksi **60%**, dan Krisis Rusia 1998 terdeteksi **100%** (jauh lebih unggul daripada model DBSCAN sebelumnya yang gagal mendeteksi Rusia / 0%).
* Pada krisis global seperti GFC 2008–2009 dan COVID-19, model menangkap negara-negara dengan kontraksi ekonomi terdalam (seperti Inggris, Spanyol, dan Italia).

### 3.3 Analisis Feature Importance

Berdasarkan analisis selisih rata-rata nilai fitur antara observasi anomali vs observasi normal (*anomaly score differential*), terungkap indikator makroekonomi yang paling dominan memicu keterisolasian anomali oleh Isolation Forest:

| Peringkat | Indikator Makroekonomi | Rata-rata Normal | Rata-rata Anomali | Selisih (Anomali - Normal) | Interpretasi Ekonomi |
|:---:|---|:---:|:---:|:---:|---|
| **1** | **`Inflation_CPI`** | 6.18% | **96.79%** | **+90.61%** | Lonjakan inflasi ekstrem adalah sinyal anomali paling tajam. |
| **2** | **`Broad_Money_Growth`** | 12.04% | **81.78%** | **+69.73%** | Ekspansi likuiditas moneter/pencetakan uang tak terkendali. |
| **3** | **`Exchange_Depreciation`** | 2.81% | **37.34%** | **+34.53%** | Depresiasi tajam mata uang lokal terhadap USD. |
| **4** | **`Exports_GDP`** | 33.98% | **65.14%** | **+31.16%** | Keterpaparan tinggi pada sektor perdagangan eksternal. |
| **5** | **`Imports_GDP`** | 33.63% | **56.97%** | **+23.34%** | Tekanan neraca perdagangan/kebutuhan impor devisa. |

Hasil ini sangat sesuai dengan teori ekonomi makro (*macroeconomic balance-of-payments theory*): krisis ekonomi yang parah hampir selalu didahului atau diiringi oleh **inflasi yang tak terkendali**, **kejatuhan nilai tukar yang tajam**, dan **ledakan jumlah uang beredar**.

---

## 4. Kesimpulan dan Kontribusi untuk EWS

1. **Efektivitas Model**: Algoritma Isolation Forest terbukti mampu mendeteksi krisis regional dan devaluasi mata uang secara mandiri tanpa menggunakan label krisis saat pelatihan, dengan tingkat keberhasilan 60% pada Krisis Asia 1997–1998 dan 100% pada Krisis Rusia 1998.
2. **Keterbatasan**: Karena sifat partisi acak pada fitur yang berdistribusi tebal (*heavy-tailed*), negara-negara dengan rasio perdagangan internasional yang sangat masif (seperti Singapura dan Irlandia) cenderung sering terisolasi lebih cepat.
3. **Rekomendasi Integrasi EWS**: Hasil prediksi dari model Isolation Forest (`hasil_isolation_forest.csv`) sangat ideal digabungkan dengan model berbasis kepadatan (*LOF*) dan model non-linear (*Autoencoder*) melalui mekanisme **Majority Voting**. Kombinasi multi-paradigma ini akan saling menutupi kelemahan masing-masing model dan meminimalisir alarm palsu (*false alarms*) bagi pembuat kebijakan.

---

## Referensi

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM)*, 413–422. DOI: 10.1109/ICDM.2008.17
2. Laeven, L., & Valencia, F. (2018). Systemic Banking Crises Revisited. *IMF Working Paper*, WP/18/206. International Monetary Fund.
3. Kaminsky, G. L., & Reinhart, C. M. (1999). The Twin Crises: The Causes of Banking and Balance-of-Payments Problems. *American Economic Review*, 89(3), 473–500.
4. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1–58.
5. Al Farizi, W. S., Hidayah, I., & Rizal, M. N. (2021). Isolation Forest Based Anomaly Detection: A Systematic Literature Review. *Proceedings of the 8th International Conference on Information Technology, Computer and Electrical Engineering (ICITACEE)*, 118–122.
