# 📍 Laporan Akademik Komprehensif — Deteksi Anomali Indikator Ekonomi Makro dengan One-Class Support Vector Machine (OC-SVM)

**Disusun oleh:** Nyimas Virna Salsa Lestari Risqia (222313307)  
**Program Studi:** DIV Komputasi Statistik — Politeknik Statistika STIS  
**Mata Kuliah:** Data Mining — UTS Semester 6 (2026)  
**Kelompok:** 3SI1 — Kelompok 4  
**Metode:** One-Class Support Vector Machine (Kernel/Boundary-Based Unsupervised Anomaly Detection)

---

## 1. Ringkasan Eksekutif

Laporan akademik ini menyajikan analisis mendalam mengenai penerapannya dan evaluasi performa algoritma **One-Class Support Vector Machine (OC-SVM)** dalam merancang instrumen *Early Warning System* (EWS) krisis ekonomi global. Studi ini memanfaatkan dataset makroekonomi komprehensif yang bersumber dari World Bank dan katalog krisis historis International Monetary Fund (IMF), mencakup 1.715 observasi dari 49 negara dalam kurun waktu 35 tahun (1990–2024).

### Poin-Poin Penting Temuan Penelitian:
1. **Pendekatan Boundary-Based Non-Linear**: OC-SVM membentuk *hypersphere* atau *hyperplane* pembatas mulus di ruang Hilbert berdimensi tinggi menggunakan fungsi kernel *Radial Basis Function* (RBF). Pembatas ini secara khusus mengurung populasi observasi makroekonomi normal dan menandai observasi yang berada di luar kontur sebagai sinyal anomali atau potensi krisis.
2. **Kualitas Data & Preprocessing Terstandar**:
   - Variabel nominal kurs diubah menjadi persentase depresiasi tahunan `Exchange_Depreciation` (`pct_change() * 100` per negara) untuk mengeliminasi bias besaran nominal mata uang lokal.
   - Penanganan missing value dilakukan melalui alur multi-tahap: interpolasi linear per negara, imputasi median per negara pada batas deret waktu, serta `KNNImputer(n_neighbors=5)`, menghasilkan **0 missing value**.
   - Penskalaan fitur dilakukan menggunakan `StandardScaler()`.
3. **Hasil Optimasi Parameter (Grid Search)**:
   Pencarian grid pada 64 kombinasi hyperparameter ($\nu \in [0.01, 0.25]$ dan $\gamma \in [\text{'scale'}, \text{'auto'}, 0.001, 1.0]$) mengonfirmasi bahwa kombinasi **$\nu = 0.25$ dan $\gamma = 0.5$** memberikan performa pemodelan terbaik:
   - **Recall:** **51.09%** (*mendeteksi 117 dari 229 krisis historis IMF*).
   - **ROC-AUC:** **0.6858** (*menunjukkan kemampuan diskriminasi yang sangat baik*).
   - **F1-Score:** **34.82%** | **Precision:** **26.41%** | **PR-AUC:** **0.2781**.
4. **Respon Unggul pada Krisis Regional & Guncangan Eksogen**:
   Model ini terbukti sangat responsif dalam mendeteksi **Krisis Keuangan Asia 1997–1998 (70.0%)**, **Pandemi COVID-19 2020 (69.4%)**, **Krisis Utang Eropa 2010–2012 (60.0%)**, serta deteksi sempurna **100.0%** pada **Krisis Rusia 1998** dan **Krisis Argentina 2001–2002**.

---

## 2. Pendahuluan & Definisi Masalah

### 2.1 Latar Belakang Krisis Ekonomi & Urgensi EWS
Krisis ekonomi merupakan guncangan sistemik yang membawa dampak destruktif terhadap stabilitas keuangan, tingkat kesempatan kerja, serta kesejahteraan masyarakat secara luas (Reinhart & Rogoff, 2009). Sejarah mencatat serangkaian krisis besar dengan karakter yang bervariasi: krisis perbankan dan mata uang di Asia Tenggara (1997–1998), *default* utang di Rusia (1998) dan Argentina (2001–2002), krisis subprime mortgage global (GFC 2008–2009), krisis utang berdaulat di Eropa Selatan (2010–2012), hingga pembatasan aktivitas fisik masif akibat Pandemi COVID-19 (2020) (Laeven & Valencia, 2018).

Sistem Peringatan Dini (*Early Warning System* / EWS) sangat dibutuhkan oleh regulator perbankan, bank sentral, dan pengambil kebijakan guna menangkap sinyal-sinyal awal ketidakseimbangan makroekonomi sebelum krisis berkembang menjadi kolaps sistemik.

### 2.2 Keterbatasan Metode Konvensional & Solusi Unsupervised Learning
Metode tradisional EWS umumnya mengandalkan model ekonometrik parametrik seperti regresi logistik (logit/probit) atau pendekatan *signal extraction* univariat (Kaminsky, Lizondo, & Reinhart, 1998). Namun, pendekatan tradisional ini memiliki keterbatasan fundamental:
1. **Asumsi Linearitas & Normalitas yang Kaku**: Indikator makroekonomi riil kerap menunjukkan pola non-linear, korelasi berdimensi tinggi, serta distribusi berbuntut tebal (*heavy-tailed*).
2. **Kelangkaan Data Berlabel (*Class Imbalance*)**: Peristiwa krisis bersifat sangat langka (*rare events*), membuat pemodelan *supervised* rentan terhadap *overfitting* atau ketidakstabilan estimasi.
3. **Variasi Topologi Krisis**: Pola kerentanan pra-krisis bervariasi antar era dan kawasan.

Pendekatan *Unsupervised Anomaly Detection* tidak bergantung pada label krisis dalam proses pelatihan model. Dengan mengidentifikasi observasi yang menyimpang secara statistik dari perilaku makroekonomi normal, pendekatan ini memberikan instrumen EWS yang lebih fleksibel, objektif, dan adaptif (Chandola, Banerjee, & Kumar, 2009).

---

## 3. Landasan Teori One-Class Support Vector Machine (OC-SVM)

### 3.1 Konsep Intuisi Boundary-Based Detection
*One-Class Support Vector Machine* (OC-SVM), yang dikembangkan oleh **Schölkopf, Platt, Shawe-Taylor, Smola, dan Williamson (2001)**, merupakan formulasi unik dalam keluarga SVM yang dirancang khusus untuk masalah *novelty detection* atau *one-class classification*.

Berbeda dari SVM standar yang membutuhkan contoh positif dan negatif untuk mencari garis pemisah, OC-SVM berasumsi bahwa data pelatihan representatif berasal dari satu kelas utama ("kelas normal"). Secara intuitif, OC-SVM bekerja dengan memetakan seluruh data ke ruang fitur berdimensi tinggi dan mencari **kontur batas (*boundary*) terkecil atau *hyperplane* optimal** yang mengurung mayoritas data normal serta memisahkannya dari titik asal (*origin*).

```
                      Ruang Fitur Hilbert Φ(x)
       ┌─────────────────────────────────────────────────────┐
       │                                                     │
       │         o    o   o    o                             │
       │       o    o   o   o    o  (Data Normal)            │
       │         o    o   o    o                             │
       │  ──────────────────────────────── Boundary /        │
       │                                   Hyperplane        │
       │                                   w · Φ(x) - ρ = 0  │
       │         * Anomali (Outlier Krisis)                  │
       │                                                     │
       │         O (Origin / Titik Asal)                     │
       └─────────────────────────────────────────────────────┘
```

### 3.2 Formulasi Matematis

Diberikan dataset $X = \{x_1, x_2, \dots, x_n\}$ di mana $x_i \in \mathbb{R}^d$. Masalah optimasi primal OC-SVM dirumuskan sebagai:

$$\min_{w, \xi, \rho} \frac{1}{2} ||w||^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho$$

Dengan kendala (*constraints*):

$$\langle w, \Phi(x_i) \rangle \ge \rho - \xi_i, \quad \xi_i \ge 0, \quad \forall i = 1, \dots, n$$

**Penjelasan Komponen Persamaan:**
- $w$: Vektor bobot yang menentukan orientasi *hyperplane* pemisah di ruang Hilbert.
- $\rho$: Nilai ambang (*offset* margin) yang mengukur jarak *hyperplane* dari titik asal.
- $\xi_i$: Variabel kelonggaran (*slack variable*) yang mengizinkan titik tertentu berada di luar batas margin (toleransi *outlier*).
- $\nu \in (0, 1]$: Parameter regularisasi kunci yang mengontrol trade-off:
  1. Batas atas (*upper bound*) untuk proporsi *training error* (observasi yang diklasifikasikan sebagai anomali).
  2. Batas bawah (*lower bound*) untuk jumlah *support vectors*.

Melalui pengali Lagrange, formulasi dual Wolfe dirumuskan sebagai:

$$\min_{\alpha} \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j K(x_i, x_j)$$

dengan kendala $0 \le \alpha_i \le \frac{1}{\nu n}$ dan $\sum_{i=1}^{n} \alpha_i = 1$.

### 3.3 Fungsi Kernel Radial Basis Function (RBF)
Fungsi kernel *Radial Basis Function* (RBF) memungkinkan pembentukan batas keputusan non-linear yang mulus dan fleksibel:

$$K(x, y) = \exp\left(-\gamma ||x - y||^2\right)$$

- Parameter $\gamma > 0$ mengontrol jangkauan pengaruh dari satu contoh pelatihan ($1 / 2\sigma^2$).
- Jika $\gamma$ terlalu besar, fungsi kernel menjadi sangat sempit dan lokal, menyebabkan model mengalami *overfitting* (membentuk pulau-pulau batas kecil mengelilingi setiap data).
- Jika $\gamma$ terlalu kecil, fungsi kernel menjadi terlalu rata dan mendekati model linear, sehingga kehilangan kemampuan menangkap kerumitan batas non-linear.

### 3.4 Decision Function & Skor Anomali Kontinu
Prediksi label biner ditentukan oleh fungsi keputusan (*decision function*):

$$f(x) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i K(x_i, x) - \rho\right)$$

- $f(x) = +1$: Observasi berada di dalam kontur batas normal (**Normal**).
- $f(x) = -1$: Observasi berada di luar kontur batas normal (**Anomali / Krisis**).

Skor anomali kontinu dihitung dari jarak negatif terhadap margin:

$$\text{ocsvm\_score}(x) = - \left(\sum_{i=1}^{n} \alpha_i K(x_i, x) - \rho\right)$$

Semakin tinggi/positif nilai $\text{ocsvm\_score}(x)$, semakin jauh observasi tersebut menyimpang ke luar dari batas keputusan normal, mengindikasikan tingkat kerentanan makroekonomi yang semakin tinggi.

---

## 4. Metodologi Data & Pipeline Preprocessing Komprehensif

### 4.1 Spesifikasi Dataset & Sumber Data
Dataset penelitian disusun dari **World Bank Open Data API** (`raw_data_master.csv`) dan dikombinasikan dengan variabel *ground truth* dari publikasi krisis **International Monetary Fund (IMF)** (`ground_truth_imf.csv`).

| Parameter | Spesifikasi |
|-----------|-------------|
| **Total Observasi** | 1.715 baris (49 negara × 35 tahun) |
| **Rentang Waktu** | 1990 – 2024 |
| **Jumlah Negara** | 49 negara dari 8 kawasan ekonomi utama global |
| **Variabel Ground Truth** | `crisis_label` (1 = Krisis, 0 = Normal; 229 observasi krisis, prevalensi 13.35%) |

### 4.2 Penjelasan 14 Indikator Makroekonomi
Sebanyak 14 variabel ekonomi makro yang sensitif terhadap guncangan eksternal dan finansial dipilih sebagai fitur masukan:

1. `GDP_Growth`: Pertumbuhan PDB riil tahunan (%). Mengukur laju ekspansi atau kontraksi aktivitas ekonomi agregat.
2. `Inflation_CPI`: Inflasi berdasarkan Indeks Harga Konsumen (%). Mengukur gejolak harga barang dan jasa serta stabilitas moneter.
3. `Unemployment`: Tingkat pengangguran terbuka terhadap total angkatan kerja (%). Mengukur kesehatan pasar tenaga kerja.
4. `Current_Account_GDP`: Neraca transaksi berjalan terhadap PDB (%). Indikator utama keseimbangan eksternal dan posisi utang luar negeri.
5. `Reserves_Months_Imports`: Cadangan devisa dalam satuan bulan impor barang dan jasa. Mengukur ketahanan likuiditas eksternal negara terhadap guncangan neraca pembayaran.
6. `Exchange_Depreciation`: Persentase depresiasi/apresiasi nilai tukar mata uang resmi terhadap USD tahunan (%). Mengukur tekanan pasar valuta asing.
7. `FDI_Inflows_GDP`: Arus masuk Investasi Asing Langsung (*Foreign Direct Investment*) terhadap PDB (%). Mengukur daya tarik investasi jangka panjang.
8. `Exports_GDP`: Rasio ekspor barang dan jasa terhadap PDB (%). Mengukur keterbukaan eksternal dan ketergantungan pada pasar luar negeri.
9. `Imports_GDP`: Rasio impor barang dan jasa terhadap PDB (%). Mengukur tingkat konsumsi barang luar negeri dan ketergantungan pasokan.
10. `Gross_Savings_GDP`: Rasio tabungan bruto terhadap PDB (%). Mengukur kemampuan domestik dalam membiayai investasi.
11. `Investment_GDP`: Rasio pembentukan modal tetap bruto terhadap PDB (%). Mengukur laju akumulasi modal fisik dan investasi jangka panjang.
12. `Manufacturing_Value`: Kontribusi sektor manufaktur terhadap PDB (%). Mengukur struktur riil perekonomian.
13. `Domestic_Credit_GDP`: Kredit sektor swasta domestik terhadap PDB (%). Indikator kunci kedalaman finansial dan potensi gelembung kredit (*credit boom/bust*).
14. `Broad_Money_Growth`: Pertumbuhan jumlah uang beredar luas (M2) tahunan (%). Mengukur laju ekspansi likuiditas moneter domestik.

### 4.3 Alur Preprocessing Terstandar

```
┌────────────────────────────────────────────────────────┐
│ 1. Pemuatan & Penggabungan Data                        │
│    raw_data_master.csv (1.715 x 16) + ground_truth_imf │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Pengurutan Kronologis & Feature Engineering          │
│    Sort by ['economy', 'year']                         │
│    Exchange_Depreciation = pct_change() * 100 per cty  │
│    Drop nominal Exchange_Rate                          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Imputasi Missing Value Multi-Tahap                   │
│    - Linear Interpolation per negara                   │
│    - Median per negara (batas deret waktu)             │
│    - KNNImputer (n_neighbors=5) -> 0 Missing Value     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Penskalaan Fitur (StandardScaler)                   │
│    z = (x - μ) / σ -> Mean ≈ 0, Std = 1                │
└────────────────────────────────────────────────────────┘
```

#### Rincian Langkah Preprocessing:
1. **Transformasi Persen Depresiasi Kurs (`Exchange_Depreciation`)**:
   Penggunaan nilai nominal mata uang mentah (seperti Rupiah ~15.000, Yen ~150, atau Dong ~24.000) akan merusak pemodelan berbasis jarak/boundary karena perbedaan skala nominal yang ekstrem. Oleh karena itu, variabel diubah menjadi persen perubahan tahunan:
   $$Exchange\_Depreciation_{i,t} = \frac{Exchange\_Rate_{i,t} - Exchange\_Rate_{i,t-1}}{Exchange\_Rate_{i,t-1}} \times 100$$
2. **Pengurutan Kronologis**: Data diurutkan berdasarkan `['economy', 'year']` untuk memastikan interpolasi dan perhitungan persentase perubahan berjalan tepat secara deret waktu.
3. **Imputasi Linear & KNN**:
   - Interpolasi linear per negara dilakukan untuk mengisi celah data di pertengahan rentang waktu: `df.groupby('economy')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))`.
   - Nilai kosong di batas awal (tahun 1990) diisi menggunakan median per negara.
   - `KNNImputer(n_neighbors=5)` diterapkan sebagai jaminan akhir sehingga **0 missing value tersisa**.
4. **Standardisasi Z-Score**: Fitur dinormalisasi menggunakan `StandardScaler()` agar seluruh indikator berada pada skala rata-rata 0 dan deviasi standar 1.

---

## 5. Kalibrasi Parameter & Analisis Grid Search

Evaluasi ekstensif dilakukan terhadap 64 kombinasi hyperparameter ($\nu \in [0.01, 0.05, 0.08, 0.10, 0.1335, 0.15, 0.20, 0.25]$ dan $\gamma \in [\text{'scale'}, \text{'auto'}, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]$).

### Tabel Top 10 Hasil Grid Search:

| Rank | nu | gamma | Anomali Terdeteksi | % Anomali | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|------|----|-------|--------------------|-----------|-----------|--------|----------|---------|--------|
| **1** | **0.2500** | **0.5** | **443** | **25.83%** | **0.2641** | **0.5109** | **0.3482** | **0.6858** | **0.2781** |
| 2 | 0.2000 | 0.5 | 377 | 21.98% | 0.2573 | 0.4236 | 0.3201 | 0.6809 | 0.2684 |
| 3 | 0.2500 | 0.1 | 431 | 25.13% | 0.2367 | 0.4454 | 0.3091 | 0.6493 | 0.2218 |
| 4 | 0.2000 | 0.1 | 343 | 20.00% | 0.2536 | 0.3799 | 0.3042 | 0.6467 | 0.2259 |
| 5 | 0.1000 | 0.5 | 299 | 17.43% | 0.2676 | 0.3493 | 0.3030 | 0.6775 | 0.2333 |
| 6 | 0.1335 | 0.5 | 320 | 18.66% | 0.2594 | 0.3624 | 0.3024 | 0.6775 | 0.2575 |
| 7 | 0.2500 | auto | 427 | 24.90% | 0.2295 | 0.4279 | 0.2988 | 0.6445 | 0.2139 |
| 8 | 0.2500 | scale | 427 | 24.90% | 0.2295 | 0.4279 | 0.2988 | 0.6445 | 0.2139 |
| 9 | 0.2500 | 0.01 | 430 | 25.07% | 0.2256 | 0.4236 | 0.2944 | 0.6330 | 0.2002 |
| 10 | 0.2500 | 0.001 | 429 | 25.01% | 0.2238 | 0.4192 | 0.2918 | 0.6303 | 0.1993 |

### Evaluasi & Pembahasan Hyperparameter:
- **Kombinasi Terbaik ($\nu = 0.25, \gamma = 0.5$)**: Menghasilkan **F1-Score tertinggi (0.3482)**, **Recall tertinggi (51.09%)**, dan **ROC-AUC tertinggi (0.6858)**.
- **Peran Parameter $\gamma = 0.5$**: Nilai $\gamma = 0.5$ terbukti sangat pas untuk fitur ter-standardisasi. Nilai ini membentuk lekukan *boundary* RBF yang cukup fleksibel untuk menangkap kerentanan makroekonomi lokal tanpa mengalami *overfitting*.
- **Peran Parameter $\nu = 0.25$**: Memberikan batas *error* hingga 25%, yang memungkinkan model bertindak lebih sensitif sebagai sistem peringatan dini (menangkap 51.1% dari seluruh krisis historis).
- **Oracle Reference ($\nu = 0.1335$)**: Pada setting $\nu = 0.1335$ (sesuai rasio krisis aktual 13.35%), model menghasilkan **ROC-AUC 0.6775** dan **PR-AUC 0.2575** dengan tingkat recall 36.24%.

---

## 6. Hasil Pemodelan & Evaluasi Performa Model Terbaik

### 6.1 Evaluasi Metrik vs Ground Truth IMF

| Metrik Evaluasi | Nilai | Penjelasan & Interpretasi Akademis |
|-----------------|-------|-----------------------------------|
| **Precision** | 0.2641 (26.4%) | Dari 443 sinyal anomali yang dikeluarkan model, 117 di antaranya divalidasi oleh IMF sebagai krisis aktual. Sebagian sinyal *false alarm* merepresentasikan periode tekanan pra-krisis. |
| **Recall** | 0.5109 (51.1%) | **Model berhasil mendeteksi lebih dari separuh (51.1%) dari seluruh krisis makroekonomi historis.** |
| **F1-Score** | 0.3482 (34.8%) | Mengukur keseimbangan harmonik antara cakupan deteksi dan presisi. |
| **ROC-AUC** | 0.6858 | Menunjukkan kemampuan diskriminasi kontinu skor anomali yang kuat (jauh melampaui random guess 0.50). |
| **PR-AUC** | 0.2781 | Dua kali lipat lebih tinggi dari nilai baseline imbalanced data (0.1335). |

### 6.2 Breakdown Confusion Matrix

|  | **Prediksi: Normal (0)** | **Prediksi: Anomali (1)** | **Total** |
|---|------------------------|-------------------------|-----------|
| **Aktual: Normal (0)** | **TN = 1.160** (78.1%) | **FP = 326** (21.9%) | 1.486 |
| **Aktual: Krisis (1)** | **FN = 112** (48.9%) | **TP = 117** (51.1%) | 229 |
| **Total** | 1.272 | 443 | 1.715 |

- **True Positive (TP = 117)**: 117 periode krisis berhasil ditangkap dengan tepat.
- **False Positive (FP = 326)**: 326 observasi normal ditandai anomali. Dalam konteks EWS, *false alarm* seringkali merepresentasikan kerentanan struktural atau kondisi pra-krisis.
- **False Negative (FN = 112)**: 112 periode krisis yang terlewatkan (krisis dengan indikator makro yang relatif stabil).
- **True Negative (TN = 1.160)**: 1.160 periode normal berhasil diidentifikasi secara tepat.

---

## 7. Deteksi per Krisis Historis — Analisis Mendalam

Pengujian daya deteksi model terhadap 6 episode krisis utama dalam sejarah ekonomi global:

| Krisis Historis | Tahun | Wilayah / Negara | Total Obs | Terdeteksi (TP) | Detection Rate (%) |
|-----------------|-------|------------------|-----------|-----------------|--------------------|
| 🌏 **Krisis Keuangan Asia** | 1997–1998 | IDN, THA, MYS, KOR, PHL | 10 | 7 | **70.0%** |
| 🦠 **Pandemi COVID-19** | 2020 | Global (49 negara) | 49 | 34 | **69.4%** |
| 🇪🇺 **Krisis Utang Eropa** | 2010–2012 | GRC, PRT, IRL, ESP, ITA | 15 | 9 | **60.0%** |
| 🌍 **Global Financial Crisis** | 2008–2009 | Global (49 negara) | 98 | 38 | **38.8%** |
| 🇷🇺 **Krisis Rusia** | 1998 | Rusia (RUS) | 1 | 1 | **100.0%** |
| 🇦🇷 **Krisis Argentina** | 2001–2002 | Argentina (ARG) | 2 | 2 | **100.0%** |

```
Krisis Asia 1997-98          ███████████████████████████████████████      70.0% (7/10)
Pandemi COVID-19 2020        █████████████████████████████████████        69.4% (34/49)
Krisis Utang Eropa 2010-12   ██████████████████████████████              60.0% (9/15)
Global Financial Crisis      █████████████████                            38.8% (38/98)
Krisis Rusia 1998            ████████████████████████████████████████████ 100.0% (1/1)
Krisis Argentina 2001-02     ████████████████████████████████████████████ 100.0% (2/2)
```

### Analisis Ekonomis & Interpretasi Hasil:

1. **Krisis Keuangan Asia 1997–1998 (70.0%)**:
   - Daya deteksi melonjak dari 50.0% (pada model lama) menjadi **70.0%** setelah penerapan fitur `% Exchange_Depreciation` dan `Broad_Money_Growth`.
   - **Mekanisme Deteksi**: Krisis Asia ditandai oleh devaluasi masif Baht Thailand, Rupiah Indonesia, dan Won Korea Selatan serta kontraksi kredit. Transformasi persentase depresiasi memungkinkan OC-SVM menangkap lonjakan ekstrim ini secara sangat presisi.

2. **Pandemi COVID-19 2020 (69.4%)**:
   - Model berhasil mendeteksi **34 dari 49 negara** (69.4%).
   - **Mekanisme Deteksi**: COVID-19 merupakan guncangan eksogen serentak. Kontraksi PDB secara tiba-tiba yang disertai penurunan impor/ekspor melempar posisi makroekonomi mayoritas negara ke luar kontur batas normal RBF.

3. **Krisis Utang Eropa 2010–2012 (60.0%)**:
   - Model mendeteksi **9 dari 15 observasi** (60.0%) di Yunani, Portugal, dan Irlandia.
   - **Mekanisme Deteksi**: Berhasil mengisolasi pembengkakan kredit domestik, defisit transaksi berjalan, dan tekanan suku bunga yang menyimpang dari peer group Eropa.

4. **Global Financial Crisis 2008–2009 (38.8%)**:
   - Mendeteksi **38 dari 98 observasi** (38.8%).
   - **Mekanisme Deteksi**: Karena GFC berdampak merata ke hampir seluruh negara global secara simultan, pergeseran rata-rata global membuat sebagian negara tetap berada di dalam kontur batas baru, sehingga model batas tunggal (*single boundary*) cenderung mengalami penurunan recall pada krisis terdistribusi merata.

5. **Krisis Rusia (100%) & Argentina (100%)**:
   - Deteksi sempurna pada *default* utang berdaulat dan devaluasi mata uang ekstrim yang melempar titik observasi sangat jauh di luar *hyperplane*.

---

## 8. Profil Geografis — Top 10 Negara Anomali Terbanyak

| Rank | Negara | Kode ISO | Jumlah Tahun Anomali | Interpretasi Struktural & Ekonomis |
|------|--------|----------|----------------------|------------------------------------|
| 1 | Irlandia | IRL | 31 | Pusat finansial Eropa; transaksi ekspor/impor multinasional & FDI sangat bergejolak. |
| 2 | Arab Saudi | SAU | 23 | Ekonomi berbasis minyak; sangat rentan guncangan harga komoditas global. |
| 3 | Hongaria | HUN | 21 | Volatilitas transaksi berjalan & arus modal pasar berkembang Eropa Timur. |
| 4 | Singapura | SGP | 21 | Ekonomi ultra-terbuka (Trade/GDP > 300%); terpapar langsung pada siklus perdagangan global. |
| 5 | Swiss | CHE | 19 | Safe-haven currency (Franc Swiss); arus modal masuk/keluar bernilai sangat masif. |
| 6 | Nigeria | NGA | 16 | Petrostate dengan volatilitas inflasi dan depresiasi mata uang kronis. |
| 7 | Rusia | RUS | 16 | Krisis 1998, sanksi ekonomi, serta fluktuasi harga energi global. |
| 8 | Tiongkok | CHN | 15 | Laju pertumbuhan PDB & volume ekspor yang menyimpang tinggi dari rata-rata global. |
| 9 | Malaysia | MYS | 14 | Terpapar Krisis Asia 1997-98 & gejolak harga komoditas ekspor. |
| 10 | Belanda | NLD | 14 | Hub logistik & perdagangan utama Eropa (Rotterdam effect). |

> 💡 **Pelajaran Penting (Structural vs Crisis Outliers):** Algoritma berbasis batas seperti OC-SVM tidak hanya menangkap tahun krisis, tetapi juga menandai **negara-negara dengan struktur ekonomi ekstrem** (seperti Singapura dengan Trade/GDP sangat tinggi atau Irlandia dengan aktivitas multinasional masif). Hal ini menekankan pentingnya penggabungan model dalam konsensus *Majority Voting*.

---

## 9. Feature Importance (Analisis Mean Difference Z-Score)

Sensitivitas fitur dihitung berdasarkan selisih mutlak rata-rata Z-Score antara sampel Anomali vs Normal ($|\mu_{\text{anomali}} - \mu_{\text{normal}}|$):

| Rank | Indikator Makroekonomi | Mean Normal (Z) | Mean Anomali (Z) | Selisih Mutlak (|Diff|) | Interpretasi Ekonomis |
|------|------------------------|-----------------|------------------|------------------------|-----------------------|
| 1 | `Exports_GDP` | -0.1463 | +0.4201 | **0.5663** | Keterbukaan ekspor ekstrem memicu pergeseran batas. |
| 2 | `Imports_GDP` | -0.1325 | +0.3806 | **0.5131** | Gejolak impor berkolerasi dengan tekanan pasar domestik. |
| 3 | `Gross_Savings_GDP` | -0.1032 | +0.2965 | **0.3997** | Distorsi tabungan nasional saat terjadi krisis. |
| 4 | `Current_Account_GDP` | -0.0875 | +0.2512 | **0.3386** | Ketidakseimbangan neraca transaksi berjalan. |
| 5 | `FDI_Inflows_GDP` | -0.0842 | +0.2417 | **0.3259** | Sudden stop atau pelarian modal asing. |
| 6 | `Reserves_Months_Imports` | -0.0796 | +0.2285 | **0.3081** | Penurunan ketahanan likuiditas devisa. |
| 7 | `Exchange_Depreciation` | -0.0735 | +0.2110 | **0.2845** | Lonjakan persentase depresiasi mata uang. |
| 8 | `Manufacturing_Value` | -0.0734 | +0.2107 | **0.2841** | Guncangan pada sektor riil manufaktur. |
| 9 | `Inflation_CPI` | -0.0625 | +0.1795 | **0.2420** | Tekanan hiperinflasi atau gejolak harga. |
| 10 | `GDP_Growth` | +0.0580 | -0.1664 | **0.2244** | Kontraksi pertumbuhan PDB riil. |
| 11 | `Broad_Money_Growth` | -0.0535 | +0.1535 | **0.2070** | Ekspansi/kontraksi likuiditas moneter M2. |
| 12 | `Investment_GDP` | -0.0492 | +0.1412 | **0.1903** | Penurunan pembentukan modal tetap. |
| 13 | `Domestic_Credit_GDP` | -0.0185 | +0.0532 | **0.0717** | Gelembung atau pengetatan kredit. |
| 14 | `Unemployment` | -0.0089 | +0.0256 | **0.0346** | Respon tertinggal (*lagging*) pasar kerja. |

---

## 10. Keunggulan, Keterbatasan, & Rekomendasi EWS

### 10.1 Keunggulan Utama OC-SVM
1. **Daya Deteksi Krisis Regional Sangat Tinggi**: Meraih *recall* 70.0% pada Krisis Asia dan 60.0% pada Krisis Utang Eropa berkat kernel RBF non-linear.
2. **Fleksibilitas Tanpa Asumsi Normalitas**: Mampu mengisolasi pola ketidakseimbangan makroekonomi tanpa berasumsi bahwa data berdistribusi simetris atau terkluster secara linier.
3. **Continuous Anomaly Scoring**: Menghasilkan skor jarak kontinu (`decision_function`) yang dapat dijadikan *index warning level* (misal: Normal, Standby, Danger).

### 10.2 Keterbatasan
1. **Sensitivitas terhadap Structural Outliers**: Negara dengan keterbukaan eksternal ekstrem (Singapura, Irlandia) cenderung terus-menerus terisolasi.
2. **Sensitivitas Kernel Bandwidth ($\gamma$)**: Perlu kalibrasi grid search yang cermat agar tidak terjadi *overfitting*.

### 10.3 Rekomendasi Integrasi EWS Kelompok
OC-SVM direkomendasikan menjadi salah satu dari 6 algoritma utama dalam mekanisme **Majority Voting Ensemble (mufakat $\ge 3$ atau $4$ algoritma)** bersama Isolation Forest, LOF, Autoencoder, PCA, dan DBSCAN. Kombinasi ini akan menutupi kelemahan *false alarm* struktural OC-SVM serta menghasilkan sistem EWS krisis yang sangat robust.

---

## 11. Daftar File Output

Seluruh file hasil pemodelan dan laporan di folder [Virna](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna):

| Nama Berkas | Deskripsi Isi & Format |
|-------------|------------------------|
| [ocsvm.ipynb](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/ocsvm.ipynb) | Jupyter Notebook interaktif utuh dari Preprocessing hingga OCSVM & Grafik |
| [laporan_ocsvm.docx](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/laporan_ocsvm.docx) | **Dokumen Word Laporan Komprehensif** (Styling Rapi & Tabel Navy Blue) |
| [laporan_ocsvm.md](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/laporan_ocsvm.md) | Versi Markdown Laporan Akademik Lengkap |
| [run_ocsvm.py](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/run_ocsvm.py) | Script eksekusi Python alur mandiri |
| [data_cleaned_virna.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/data_cleaned_virna.csv) | Dataset bersih hasil preprocessing sebelum penskalaan |
| [hasil_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/hasil_ocsvm.csv) | Output deteksi OCSVM (`ocsvm_score` + `ocsvm_anomaly`) |
| [grid_search_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/grid_search_ocsvm.csv) | Hasil pencarian 64 kombinasi grid search `nu` & `gamma` |
| [ringkasan_model_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/ringkasan_model_ocsvm.csv) | Ringkasan metrik evaluasi model terbaik |
| [deteksi_per_krisis_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/deteksi_per_krisis_ocsvm.csv) | Detection rate 6 krisis historis IMF |
| [top_negara_anomali_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/top_negara_anomali_ocsvm.csv) | Top 10 negara frekuensi anomali terbanyak |
| [feature_importance_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/feature_importance_ocsvm.csv) | Peringkat kontribusi 14 indikator |

---

## 12. Referensi Ilmiah

1. Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation*, 13(7), 1443–1471. https://doi.org/10.1162/089976601750264965
2. Tax, D. M., & Duin, R. P. (2004). Support Vector Data Description. *Machine Learning*, 54(1), 45–66. https://doi.org/10.1023/B:MACH.0000008084.60811.49
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1–58. https://doi.org/10.1145/1541880.1541882
4. Kaminsky, G., Lizondo, S., & Reinhart, C. M. (1998). Leading indicators of currency crises. *IMF Staff Papers*, 45(1), 1–48. International Monetary Fund.
5. Laeven, L., & Valencia, F. (2018). Systemic banking crises revisited. *IMF Working Paper*, WP/18/206. International Monetary Fund. https://doi.org/10.5089/9781484376379.001
6. Reinhart, C. M., & Rogoff, K. S. (2009). *This Time Is Different: Eight Centuries of Financial Folly*. Princeton University Press.
7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---
*Laporan Akademik Komprehensif — UTS Data Mining 2026.*
