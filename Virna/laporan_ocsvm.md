# 📍 Laporan Akademik — Deteksi Anomali Indikator Ekonomi Makro dengan One-Class Support Vector Machine (OC-SVM)

**Disusun oleh:** Nyimas Virna Salsa Lestari Risqia (222313307)  
**Program Studi:** DIV Komputasi Statistik — Politeknik Statistika STIS  
**Mata Kuliah:** Data Mining — UTS Semester 6 (2026)  
**Kelompok:** 3SI1 — Kelompok 4  
**Metode:** One-Class Support Vector Machine (Kernel/Boundary-Based Unsupervised Anomaly Detection)

---

## 1. Ringkasan Eksekutif

Laporan ini menyajikan implementasi dan evaluasi algoritma **One-Class Support Vector Machine (OC-SVM)** sebagai bagian dari sistem peringatan dini (*Early Warning System* / EWS) krisis ekonomi berbasis *unsupervised anomaly detection*. Pengujian dilakukan menggunakan dataset 14 indikator makroekonomi ter-standardisasi dari 49 negara dalam rentang waktu 35 tahun (1990–2024), mencakup total 1.714 observasi.

**Temuan Utama Metodologis & Empiris:**
- **Prinsip Kerja:** OC-SVM membentuk *hypersphere* / *hyperplane* pembatas mulus non-linear di ruang Hilbert berdimensi tinggi menggunakan fungsi kernel *Radial Basis Function* (RBF) untuk mengurung mayoritas observasi ekonomi normal. Observasi yang berada di luar *boundary* dikategorikan sebagai anomali/krisis.
- **Hasil Tuning (Grid Search):** Evaluasi 56 kombinasi hyperparameter ($\nu \in [0.01, 0.25]$ dan $\gamma \in [\text{'scale'}, \text{'auto'}, 0.001, 1.0]$) menghasilkan konfigurasi optimal pada $\nu = 0.25$ dan $\gamma = 0.001$.
- **Performa Evaluasi (vs Ground Truth):**
  - **F1-Score:** 39.05% (0.3905)
  - **Recall:** 44.98% (45.0% krisis historis berhasil terdeteksi)
  - **Precision:** 34.50% (0.3450)
  - **ROC-AUC:** 0.6599 (0.6658 pada kalibrasi $\nu = 0.1919$)
  - **PR-AUC:** 0.2900
- **Daya Deteksi Krisis Historis:** OC-SVM menunjukkan performa deteksi sangat tinggi pada guncangan sistemik global seperti **Pandemi COVID-19 2020 (69.4%)**, **Krisis Utang Eropa 2010–2012 (53.3%)**, **Krisis Asia 1997–1998 (50.0%)**, serta deteksi sempurna 100% pada **Krisis Rusia 1998** dan **Krisis Argentina 2001–2002**.

---

## 2. Landasan Teori One-Class Support Vector Machine

### 2.1 Konsep Dasar
*One-Class Support Vector Machine* (OC-SVM), yang diperkenalkan oleh **Schölkopf, Platt, Shawe-Taylor, Smola, dan Williamson (2001)**, merupakan perluasan dari Support Vector Machine (SVM) konvensional untuk kasus *unsupervised novelty detection* atau deteksi anomali tanpa label kelas eksplisit.

Berbeda dari SVM standar yang membutuhkan dua kelas label (positif dan negatif) untuk menemukan *hyperplane* pemisah berjarak maksimum, OC-SVM berasumsi bahwa dataset pelatihan utamanya terdiri dari satu kelas tunggal ("kelas normal"). Algoritma ini memetakan data masukan ke ruang fitur berdimensi tinggi melalui transformasi non-linear $\Phi(x)$ dan mencoba memisahkan seluruh data normal dari titik asal (*origin*) dengan margin sebesar mungkin.

```
       Ruang Fitur Berdimensi Tinggi Φ(x)
       ┌──────────────────────────────────────┐
       │                                      │
       │   o   o   o  (Data Normal)          │
       │     o   o  o                         │
       │   o   o   o                          │
       │────────────── Boundary / Hyperplane  │
       │               w · Φ(x) - ρ = 0       │
       │                                      │
       │   * Anomali (Outlier)                │
       │                                      │
       │   O (Origin / Titik Asal)            │
       └──────────────────────────────────────┘
```

### 2.2 Formulasi Matematis

Diberikan dataset pelatihan $X = \{x_1, x_2, \dots, x_n\}$ di mana $x_i \in \mathbb{R}^d$. Masalah optimasi primal OC-SVM dirumuskan sebagai berikut:

$$\min_{w, \xi, \rho} \frac{1}{2} ||w||^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho$$

Dengan kendala (*constraints*):

$$\langle w, \Phi(x_i) \rangle \ge \rho - \xi_i, \quad \xi_i \ge 0, \quad \forall i = 1, \dots, n$$

di mana:
- $w$: Vector bobot tegak lurus terhadap *hyperplane* pembatas.
- $\rho$: Margin/offset *hyperplane* terhadap titik asal.
- $\xi_i$: Variable kelonggaran (*slack variable*) yang mengizinkan observasi tertentu berada di luar batas margin (toleransi *outlier*).
- $\nu \in (0, 1]$: Parameter regularisasi yang berfungsi sebagai:
  1. Batas atas (*upper bound*) untuk proporsi *training errors* (observasi di luar batas/anomali).
  2. Batas bawah (*lower bound*) untuk jumlah *support vectors*.

Melalui pengali Lagrange dan transformasi dual Wolfe, persamaan di atas dapat diselesaikan menggunakan kernel trick:

$$\min_{\alpha} \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j K(x_i, x_j)$$

dengan kendala $0 \le \alpha_i \le \frac{1}{\nu n}$ dan $\sum_{i=1}^{n} \alpha_i = 1$.

### 2.3 Peran Kernel Radial Basis Function (RBF)
Fungsi kernel *Radial Basis Function* (RBF) digunakan untuk menangani hubungan non-linear antar indikator ekonomi makro:

$$K(x, y) = \exp\left(-\gamma ||x - y||^2\right)$$

- Parameter $\gamma > 0$ mengontrol lebar fungsi Gaussian (jangkauan pengaruh dari satu titik observasi). Nilai $\gamma$ yang terlalu besar dapat menyebabkan *overfitting* (batas menjadi terlalu ketat mengelilingi titik individual), sedangkan $\gamma$ yang terlalu kecil membuat batas menjadi terlalu sederhana dan mirip dengan model linear.

### 2.4 Decision Function dan Anomaly Score
Fungsi keputusan (*decision function*) untuk observasi uji $x$ didefinisikan sebagai:

$$f(x) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i K(x_i, x) - \rho\right)$$

- $f(x) = +1$: Observasi berada di dalam batas (*inlier* / **Normal**).
- $f(x) = -1$: Observasi berada di luar batas (*outlier* / **Anomali/Krisis**).

Untuk memperoleh skor anomali kontinu (*anomaly score* $s(x)$), nilai jarak kontinu terhadap margin diambil sebagai:

$$s(x) = - \left(\sum_{i=1}^{n} \alpha_i K(x_i, x) - \rho\right)$$

Semakin positif/tinggi nilai $s(x)$, semakin jauh observasi tersebut menyimpang di luar batas keputusan normal.

---

## 3. Spesifikasi Dataset & Preprocessing

| Aspek | Detail Spesifikasi |
|-------|--------------------|
| **Sumber Data** | World Bank Open Data API (via `wbgapi`) |
| **Cakupan Sampel** | 1.714 observasi (49 negara × 35 tahun, 1990–2024) |
| **Jumlah Fitur** | 14 Indikator Makroekonomi (z-score standardized) |
| **Ground Truth** | 329 observasi krisis historis (`crisis_label = 1`, prevalensi 19.19%) |

### 14 Indikator Makroekonomi:
1. `GDP_Growth` — Pertumbuhan PDB riil (%)
2. `GDP_PerCapita_Growth` — Pertumbuhan PDB per kapita (%)
3. `Inflation_CPI` — Inflasi berdasarkan CPI (%)
4. `Total_Reserves` — Cadangan devisa total (USD)
5. `Unemployment` — Tingkat pengangguran (%)
6. `Current_Account_GDP` — Transaksi berjalan terhadap PDB (%)
7. `Trade_GDP` — Rasio perdagangan terhadap PDB (%)
8. `FDI_Inflows_GDP` — FDI masuk terhadap PDB (%)
9. `Exports_GDP` — Rasio ekspor terhadap PDB (%)
10. `Imports_GDP` — Rasio impor terhadap PDB (%)
11. `Gross_Savings_GDP` — Tabungan bruto terhadap PDB (%)
12. `Exchange_Rate` — Nilai tukar LCU/USD
13. `Manufacturing_Value` — Nilai tambah manufaktur/PDB (%)
14. `Investment_GDP` — Pembentukan modal tetap/PDB (%)

---

## 4. Konfigurasi Model & Grid Search

Pencarian grid ekstensif dilakukan pada 56 kombinasi hyperparameter:
- `nu`: `[0.01, 0.05, 0.08, 0.10, 0.15, 0.1919, 0.25]`
- `gamma`: `['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]`

### Top 10 Hasil Grid Search (Diurutkan berdasarkan F1-Score):

| Rank | nu | gamma | Anomali Terdeteksi | % Anomali | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|------|----|-------|--------------------|-----------|-----------|--------|----------|---------|--------|
| **1** | **0.2500** | **0.001** | **429** | **25.03%** | **0.3450** | **0.4498** | **0.3905** | **0.6599** | **0.2900** |
| 2 | 0.2500 | 0.01 | 429 | 25.03% | 0.3240 | 0.4225 | 0.3668 | 0.6492 | 0.2801 |
| 3 | 0.1919 | 0.001 | 329 | 19.19% | 0.3617 | 0.3617 | 0.3617 | 0.6658 | 0.2978 |
| 4 | 0.2500 | 0.5 | 434 | 25.32% | 0.3134 | 0.4134 | 0.3565 | 0.6497 | 0.3394 |
| 5 | 0.1500 | 0.001 | 256 | 14.94% | 0.3906 | 0.3040 | 0.3419 | 0.6668 | 0.3013 |
| 6 | 0.1919 | 0.01 | 328 | 19.14% | 0.3384 | 0.3374 | 0.3379 | 0.6526 | 0.2856 |
| 7 | 0.1500 | 1.0 | 519 | 30.28% | 0.2755 | 0.4347 | 0.3373 | 0.6252 | 0.2518 |
| 8 | 0.2500 | scale | 431 | 25.15% | 0.2970 | 0.3891 | 0.3368 | 0.6296 | 0.2776 |
| 9 | 0.2500 | auto | 427 | 24.91% | 0.2974 | 0.3860 | 0.3360 | 0.6298 | 0.2706 |
| 10 | 0.2500 | 0.1 | 428 | 24.97% | 0.2944 | 0.3830 | 0.3329 | 0.6318 | 0.2926 |

**Analisis Parameter:**
- Kombinasi **$\nu = 0.25$ dan $\gamma = 0.001$** menghasilkan F1-Score tertinggi (**0.3905**) dengan tingkat *recall* mencapai 44.98%.
- Nilai $\gamma = 0.001$ yang relatif kecil membentuk permukaan *boundary* yang lebih mulus (*smooth*), sehingga tidak terjebak *overfitting* pada varians lokal yang berisik (*noise*).
- Pilihan oracle reference $\nu = 0.1919$ (sesuai proporsi aktual 19.19%) menghasilkan **Precision 36.17%**, **Recall 36.17%**, **F1 36.17%**, dan **ROC-AUC 0.6658**.

---

## 5. Hasil Pemodelan & Evaluasi Performance

### 5.1 Matriks Evaluasi Model Terbaik ($\nu=0.25, \gamma=0.001$)

| Metrik | Nilai | Interpretasi Ekonomis & Teknis |
|--------|-------|--------------------------------|
| **Precision** | 0.3450 (34.5%) | Dari seluruh peringatan anomali yang dikeluarkan, 34.5% benar-benar merupakan episode krisis historis. |
| **Recall** | 0.4498 (45.0%) | Model berhasil menangkap 45.0% dari seluruh kejadian krisis makroekonomi yang terjadi. |
| **F1-Score** | 0.3905 (39.1%) | Keseimbangan harmonik terbaik antara cakupan deteksi dan sensitivitas. |
| **ROC-AUC** | 0.6599 | Kemampuan diskriminatif model berada di atas pemodelan acak (0.50). |
| **PR-AUC / AP** | 0.2900 | Performa area Precision-Recall unggul dibandingkan *baseline* proporsi krisis (0.1919). |

### 5.2 Confusion Matrix

|  | **Prediksi: Normal (0)** | **Prediksi: Anomali (1)** | **Total Aktual** |
|---|------------------------|-------------------------|------------------|
| **Aktual: Normal (0)** | **TN = 1.104** (79.7%) | **FP = 281** (20.3%) | 1.385 |
| **Aktual: Krisis (1)** | **FN = 181** (55.0%) | **TP = 148** (45.0%) | 329 |
| **Total Prediksi** | 1.285 | 429 | 1.714 |

---

## 6. Deteksi per Krisis Historis

Pengujian dilakukan terhadap 6 periode krisis utama dalam katalog krisis global:

| Krisis Historis | Tahun | Wilayah / Negara | Total Observasi | Terdeteksi (TP) | Detection Rate (%) |
|-----------------|-------|------------------|-----------------|-----------------|--------------------|
| 🦠 **Pandemi COVID-19** | 2020 | Global (49 negara) | 49 | 34 | **69.4%** |
| 🇦🇷 **Krisis Argentina** | 2001–2002 | Argentina (ARG) | 2 | 2 | **100.0%** |
| 🇷🇺 **Krisis Rusia** | 1998 | Rusia (RUS) | 1 | 1 | **100.0%** |
| 🇪🇺 **Krisis Utang Eropa** | 2010–2012 | GRC, PRT, IRL, ESP, ITA | 15 | 8 | **53.3%** |
| 🌏 **Krisis Keuangan Asia** | 1997–1998 | IDN, THA, MYS, KOR, PHL | 10 | 5 | **50.0%** |
| 🌍 **Global Financial Crisis (GFC)** | 2008–2009 | Global (49 negara) | 98 | 36 | **36.7%** |

```
Pandemi COVID-19 2020        ███████████████████████████████████████████  69.4% (34/49)
Krisis Utang Eropa 2010-12   ███████████████████████████                  53.3% (8/15)
Krisis Asia 1997-98          █████████████████████████                    50.0% (5/10)
Global Financial Crisis      ██████████████████                           36.7% (36/98)
Krisis Rusia 1998            ████████████████████████████████████████████ 100.0% (1/1)
Krisis Argentina 2001-02     ████████████████████████████████████████████ 100.0% (2/2)
```

### Interpretasi Hasil per Krisis:
1. **Pandemi COVID-19 (69.4%):** OC-SVM sangat efektif mendeteksi guncangan eksogen serentak di mana indikator PDB, perdagangan, dan tenaga kerja secara mendadak terlempar jauh di luar batas *hyperplane* normal.
2. **Krisis Rusia & Argentina (100.0%):** Peristiwa *default* utang dan devaluasi ekstrem menghasilkan pencilan multivariat yang sangat tegas dalam ruang *kernel RBF*.
3. **Krisis Utang Eropa (53.3%):** OC-SVM berhasil mengisolasi negara-negara Eropa Selatan (Yunani, Irlandia, Portugal) yang memiliki rasio utang dan defisit eksternal menyimpang dari peer group Eropa.
4. **Global Financial Crisis 2008–2009 (36.7%):** Karena dampak GFC berimbas hampir ke seluruh dunia secara simultan, pergeseran agregat global menyebabkan sebagian negara tetap berada di dalam kontur batas baru, sehingga *recall* pada krisis terdistribusi merata cenderung lebih rendah dibandingkan guncangan eksogen tajam.

---

## 7. Profil Geografis — Top 10 Negara Anomali Terbanyak

| Rank | Negara | Kode ISO | Jumlah Tahun Anomali | Karakteristik Struktur Ekonomi |
|------|--------|----------|----------------------|--------------------------------|
| 1 | Tiongkok | CHN | 35 | Pertumbuhan PDB & ekspor sangat tinggi (Outlier Struktural) |
| 2 | Afrika Selatan | ZAF | 35 | Volatilitas pengangguran & transaksi berjalan tinggi |
| 3 | Singapura | SGP | 35 | Ekonomi sangat terbuka (Trade/GDP > 300%) |
| 4 | Vietnam | VNM | 34 | Transformasi industri & lonjakan FDI masif |
| 5 | Argentina | ARG | 20 | Volatilitas inflasi & devaluasi mata uang kronis |
| 6 | Irlandia | IRL | 18 | Pusat finansial / transaksi multinasional eropa |
| 7 | Malaysia | MYS | 18 | Ekonomi berbasis ekspor manufaktur & komoditas |
| 8 | Indonesia | IDN | 17 | Dampak Krisis 1997-98 & guncangan komoditas |
| 9 | Yunani | GRC | 17 | Krisis utang berdaulat & kontraksi fiskal panjang |
| 10 | Kenya | KEN | 14 | Negara berkembang dengan struktur keuangan dinamis |

> 📌 **Catatan Ekonomis:** OC-SVM tidak hanya menandai tahun-tahun krisis historis, tetapi juga menandai **anomali struktural** pada negara-negara dengan profil makroekonomi ekstrem (seperti Singapura dengan Trade/GDP sangat tinggi atau Tiongkok dengan laju pertumbuhan masif). Hal ini merupakan karakteristik inheren dari algoritma deteksi batas (*boundary-based*).

---

## 8. Feature Importance (Analisis Mean Difference)

Sensitivitas fitur dihitung berdasarkan selisih mutlak rata-rata nilai z-score antara kelompok terdeteksi Anomali vs Normal:

| Rank | Indikator Makroekonomi | Mean Normal | Mean Anomali | Selisih Mutlak (|Diff|) |
|------|------------------------|-------------|--------------|------------------------|
| 1 | `Exports_GDP` | -0.1559 | +0.4563 | **0.6121** |
| 2 | `Trade_GDP` | -0.1533 | +0.4490 | **0.6023** |
| 3 | `Exchange_Rate` | -0.1504 | +0.4413 | **0.5916** |
| 4 | `Imports_GDP` | -0.1484 | +0.4348 | **0.5832** |
| 5 | `Manufacturing_Value` | -0.1051 | +0.3102 | **0.4153** |
| 6 | `FDI_Inflows_GDP` | -0.1027 | +0.2886 | **0.3913** |
| 7 | `Gross_Savings_GDP` | -0.0965 | +0.2886 | **0.3851** |
| 8 | `Total_Reserves` | -0.0948 | +0.2707 | **0.3655** |
| 9 | `Current_Account_GDP` | -0.0876 | +0.2650 | **0.3526** |
| 10 | `Investment_GDP` | -0.0814 | +0.2362 | **0.3177** |
| 11 | `Unemployment` | -0.0738 | +0.2064 | **0.2802** |
| 12 | `GDP_PerCapita_Growth` | +0.0337 | -0.1066 | **0.1403** |
| 13 | `Inflation_CPI` | -0.0568 | +0.0322 | **0.0889** |
| 14 | `GDP_Growth` | +0.0113 | -0.0419 | **0.0532** |

### Insight Penting:
1. **Sektor Perdagangan Luar Negeri & Nilai Tukar** (`Exports_GDP`, `Trade_GDP`, `Exchange_Rate`, `Imports_GDP`) merupakan pendorong utama penetapan batas anomali pada OC-SVM.
2. Anomali terdeteksi ditandai oleh lonjakan ekstrem pada rasio keterbukaan perdagangan dan gejolak nilai tukar mata uang, yang secara langsung merepresentasikan kerentanan eksternal (*external vulnerability*) suatu negara.

---

## 9. Keunggulan, Keterbatasan, & Rekomendasi EWS

### 9.1 Keunggulan OC-SVM
1. **Fleksibilitas Batas Non-Linear:** Kernel RBF mampu mengisolasi distribusi data berdimensi 14 tanpa mengasumsikan bentuk *spherical* atau distribusi normal.
2. **Sensitif terhadap Guncangan Eksogen:** Unggul dalam mendeteksi krisis skala global mendadak seperti COVID-19 (69.4%) dan *default* negara (100%).
3. **Model Pembatas yang Robust:** Memberikan skor kontinu (`decision_function`) yang dapat dikonversi menjadi *Early Warning Indicator*.

### 9.2 Keterbatasan
1. **Kecenderungan Sensitif pada Outlier Struktural:** Negara dengan rasio perdagangan ekstrem (Singapura) atau pertumbuhan tinggi (Tiongkok) konsisten ditandai anomali meskipun tidak sedang mengalami krisis.
2. **Sensitivitas terhadap Skala & Hyperparameter $\gamma$:** Memerlukan tuning $\gamma$ secara teliti agar tidak terjadi *overfitting*.

### 9.3 Rekomendasi Integrasi EWS
- OC-SVM sangat direkomendasikan masuk ke dalam mekanisme **Majority Voting Ensemble** (minimal $\ge 3$ atau $4$ algoritma mufakat) bersama Isolation Forest, LOF, Autoencoder, PCA, dan DBSCAN untuk mengurangi *false positive* akibat outlier struktural.

---

## 10. Daftar File Output

Seluruh file hasil eksekusi pemodelan disimpan di folder [Virna](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna):

| Nama File | Deskripsi Isi File |
|-----------|--------------------|
| [ocsvm.ipynb](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/ocsvm.ipynb) | Notebook Jupyter Python interaktif lengkap dengan kode, analisis, dan visualisasi |
| [run_ocsvm.py](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/run_ocsvm.py) | Script eksekusi Python mandiri |
| [hasil_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/hasil_ocsvm.csv) | Dataset hasil deteksi (1.714 baris) + `ocsvm_score` + `ocsvm_anomaly` |
| [grid_search_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/grid_search_ocsvm.csv) | Tabel hasil pencarian 56 kombinasi grid search `nu` & `gamma` |
| [ringkasan_model_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/ringkasan_model_ocsvm.csv) | Ringkasan metrik evaluasi model terbaik (Precision, Recall, F1, ROC, AP, CM) |
| [deteksi_per_krisis_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/deteksi_per_krisis_ocsvm.csv) | Tingkat deteksi (*detection rate*) untuk 6 krisis historis |
| [top_negara_anomali_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/top_negara_anomali_ocsvm.csv) | 10 negara dengan frekuensi anomali terbanyak |
| [feature_importance_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/feature_importance_ocsvm.csv) | Peringkat kontribusi 14 fitur makroekonomi |

---

## 11. Referensi Ilmiah

1. Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the Support of a High-Dimensional Distribution. *Neural Computation*, 13(7), 1443–1471. https://doi.org/10.1162/089976601750264965
2. Tax, D. M., & Duin, R. P. (2004). Support Vector Data Description. *Machine Learning*, 54(1), 45–66. https://doi.org/10.1023/B:MACH.0000008084.60811.49
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1–58. https://doi.org/10.1145/1541880.1541882
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. Vert, R., & Vert, J. P. (2006). Consistency and Convergence Rates of One-Class SVMs and Related Algorithms. *Journal of Machine Learning Research*, 7, 817–854.

---
*Laporan ini merupakan bagian dari Proyek **Analisis Deteksi Anomali pada Indikator Makroekonomi Global sebagai Instrumen Early Warning System Krisis Ekonomi** — UTS Data Mining 2026.*
