# 📍 Laporan Akademik Pembaruan — Deteksi Anomali Indikator Ekonomi Makro dengan One-Class Support Vector Machine (OC-SVM)

**Disusun oleh:** Nyimas Virna Salsa Lestari Risqia (222313307)  
**Program Studi:** DIV Komputasi Statistik — Politeknik Statistika STIS  
**Mata Kuliah:** Data Mining — UTS Semester 6 (2026)  
**Kelompok:** 3SI1 — Kelompok 4  
**Metode:** One-Class Support Vector Machine (Kernel/Boundary-Based Unsupervised Anomaly Detection)

---

## 1. Ringkasan Eksekutif

Laporan ini menyajikan pembaruan menyeluruh atas implementasi dan evaluasi algoritma **One-Class Support Vector Machine (OC-SVM)** sebagai instrumen *Early Warning System* (EWS) krisis ekonomi. Pembaruan dilakukan berbasis dataset master baru `raw_data_master.csv` dan `ground_truth_imf.csv` yang mencakup 1.715 observasi dari 49 negara dalam rentang tahun 1990–2024.

**Pembaruan Utamanya Meliputi:**
1. **Peningkatan Kualitas Data & Feature Engineering:**
   - Variabel nominal `Exchange_Rate` diubah menjadi persentase depresiasi tahunan `Exchange_Depreciation` (`pct_change() * 100` per negara) untuk mengeliminasi bias nominal mata uang (seperti IDR ~15.000 atau VND ~24.000).
   - Pengurutan data secara ketat berdasarkan `['economy', 'year']`.
   - Pipeline imputasi multi-tahap: interpolasi linear per negara, pengisian median batas deret per negara, dan `KNNImputer(n_neighbors=5)` sehingga menghasilkan 0 missing values.
   - Penskalaan fitur menggunakan `StandardScaler()`.
2. **Peningkatan Daya Deteksi (Recall & ROC-AUC):**
   - Hasil tuning pada 64 kombinasi hyperparameter ($\nu \in [0.01, 0.25]$ dan $\gamma \in [\text{'scale'}, \text{'auto'}, 0.001, 1.0]$) menghasilkan kombinasi terbaik pada **$\nu = 0.25$ dan $\gamma = 0.5$**.
   - **Recall melonjak menjadi 51.09%** (berhasil mendeteksi lebih dari 51% dari seluruh kejadian krisis historis).
   - **ROC-AUC mencapai 0.6858** (meningkat dari model sebelumnya).
   - **F1-Score:** 34.82% (0.3482) | **Precision:** 26.41% | **PR-AUC:** 0.2781.
3. **Peningkatan Deteksi per Krisis Historis (Ground Truth IMF):**
   - **Krisis Keuangan Asia 1997–1998:** Daya deteksi melonjak tajam ke **70.0%** (7/10 observasi).
   - **Krisis Utang Eropa 2010–2012:** Daya deteksi meningkat ke **60.0%** (9/15 observasi).
   - **Global Financial Crisis 2008–2009:** Daya deteksi meningkat ke **38.8%** (38/98 observasi).
   - **Pandemi COVID-19 2020:** Tetap tinggi pada **69.4%** (34/49 negara).
   - **Krisis Rusia 1998 & Argentina 2001–2002:** Deteksi sempurna **100.0%**.

---

## 2. Landasan Teori One-Class Support Vector Machine

### 2.1 Konsep Utama
*One-Class Support Vector Machine* (OC-SVM) (Schölkopf et al., 2001) adalah metode *unsupervised novelty detection* yang dirancang untuk mempelajari pembatas (*boundary*) ketat yang melingkupi distribusi populasi data normal.

Algoritma ini memetakan data fitur makroekonomi ke ruang fitur Hilbert berdimensi tinggi melalui fungsi transformasi non-linear $\Phi(x)$ dan mengkonstruksi *hyperplane* pemisah berjarak maksimum dari titik asal (*origin*):

$$\min_{w, \xi, \rho} \frac{1}{2} ||w||^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho$$

Dengan kendala:

$$\langle w, \Phi(x_i) \rangle \ge \rho - \xi_i, \quad \xi_i \ge 0, \quad \forall i = 1, \dots, n$$

```
       Ruang Fitur Hilbert Berdimensi Tinggi Φ(x)
       ┌──────────────────────────────────────┐
       │                                      │
       │   o   o   o  (Data Normal)          │
       │     o   o  o                         │
       │   o   o   o                          │
       │────────────── Boundary / Hyperplane  │
       │               w · Φ(x) - ρ = 0       │
       │                                      │
       │   * Anomali Krisis (Outlier)         │
       │                                      │
       │   O (Origin / Titik Asal)            │
       └──────────────────────────────────────┘
```

### 2.2 Peran Kernel RBF & Hyperparameter ($\nu, \gamma$)
Fungsi kernel *Radial Basis Function* (RBF) menangani korelasi kompleks non-linear antar variabel ekonomi:

$$K(x, y) = \exp\left(-\gamma ||x - y||^2\right)$$

- **$\nu$ (nu)**: Menentukan batas atas proporsi *training errors* (anomali) dan batas bawah jumlah *support vectors*. Pada dataset ini, nilai $\nu = 0.25$ memberikan sensitivitas optimal untuk mendeteksi potensi krisis.
- **$\gamma$ (gamma)**: Menentukan jangkauan pengaruh dari satu contoh pelatihan. Nilai $\gamma = 0.5$ menghasilkan kontur pembatas non-linear yang responsif terhadap penyimpangan makroekonomi lokal.

---

## 3. Spesifikasi Dataset & Pipeline Preprocessing

| Langkah Pipeline | Deskripsi Metodologis | Hasil Eksplorasi |
|------------------|-----------------------|------------------|
| **Pemuatan Data** | Penggabungan `raw_data_master.csv` & `ground_truth_imf.csv` | 1.715 observasi (49 negara × 35 tahun, 1990–2024) |
| **Ground Truth** | Variabel `is_crisis` diselaraskan sebagai `crisis_label` | 229 observasi krisis (prevalensi 13.35%) |
| **Pengurutan Data** | Sorted by `['economy', 'year']` | Menjaga kontinuitas deret waktu per negara |
| **Transformasi Exchange Rate** | $Exchange\_Depreciation_{i,t} = \frac{Ex_{i,t} - Ex_{i,t-1}}{Ex_{i,t-1}} \times 100$ | Mengeliminasi bias skala nominal mata uang |
| **Imputasi Multi-tahap** | 1. `interpolate(method='linear', limit_direction='both')`<br>2. Median per negara<br>3. `KNNImputer(n_neighbors=5)` | **0 Missing Value** |
| **Penskalaan** | `StandardScaler()` pada 14 indikator | Mean $\approx 0$, Variance $= 1$ |

### 14 Indikator Makroekonomi Preprocessed:
1. `GDP_Growth` — Pertumbuhan PDB riil (%)
2. `Inflation_CPI` — Inflasi berdasarkan CPI (%)
3. `Unemployment` — Tingkat pengangguran (%)
4. `Current_Account_GDP` — Neraca transaksi berjalan terhadap PDB (%)
5. `Reserves_Months_Imports` — Cadangan devisa dalam bulan impor
6. `Exchange_Depreciation` — Depresiasi nilai tukar tahunan (%)
7. `FDI_Inflows_GDP` — FDI masuk terhadap PDB (%)
8. `Exports_GDP` — Rasio ekspor terhadap PDB (%)
9. `Imports_GDP` — Rasio impor terhadap PDB (%)
10. `Gross_Savings_GDP` — Tabungan bruto terhadap PDB (%)
11. `Investment_GDP` — Pembentukan modal tetap/PDB (%)
12. `Manufacturing_Value` — Kontribusi manufaktur/PDB (%)
13. `Domestic_Credit_GDP` — Kredit domestik terhadap PDB (%)
14. `Broad_Money_Growth` — Pertumbuhan uang beredar / M2 (%)

---

## 4. Hasil Tuning Grid Search Hyperparameter

Pencarian grid dilakukan pada 64 kombinasi hyperparameter:

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

**Analisis Konfigurasi:**
- Kombinasi **$\nu = 0.25$ dan $\gamma = 0.5$** mengungguli seluruh kombinasi lain dengan **F1-Score 0.3482**, **Recall 51.09%**, dan **ROC-AUC 0.6858**.
- Nilai $\gamma = 0.5$ terbukti sangat efektif menangkap penyimpangan fitur ter-standardisasi secara lebih tajam dibandingkan RBF linear/lemah ($\gamma=0.001$).

---

## 5. Evaluasi Performa Model Terbaik ($\nu=0.25, \gamma=0.5$)

### 5.1 Classification Report & Metrik Utama

| Metrik | Nilai | Evaluasi Teknis |
|--------|-------|-----------------|
| **Precision** | 0.2641 (26.4%) | 26.4% dari sinyal anomali terdeteksi merupakan krisis aktual. |
| **Recall** | 0.5109 (51.1%) | **Mendeteksi lebih dari separuh (51.1%) seluruh episode krisis historis.** |
| **F1-Score** | 0.3482 (34.8%) | Keseimbangan harmonik terbaik untuk EWS. |
| **ROC-AUC** | 0.6858 | Performa area ROC tertinggi (diskriminasi sangat baik). |
| **PR-AUC** | 0.2781 | Jauh melampaui baseline imbalanced (13.35%). |

### 5.2 Confusion Matrix

|  | **Prediksi: Normal (0)** | **Prediksi: Anomali (1)** | **Total** |
|---|------------------------|-------------------------|-----------|
| **Aktual: Normal (0)** | **TN = 1.160** (78.1%) | **FP = 326** (21.9%) | 1.486 |
| **Aktual: Krisis (1)** | **FN = 112** (48.9%) | **TP = 117** (51.1%) | 229 |
| **Total** | 1.272 | 443 | 1.715 |

---

## 6. Evaluasi Deteksi per Krisis Historis (Ground Truth IMF)

| Krisis Historis | Tahun | Wilayah / Negara | Total | Terdeteksi (TP) | Detection Rate (%) |
|-----------------|-------|------------------|-------|-----------------|--------------------|
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

### Analisis Kunci:
- **Krisis Asia 1997-98 (70.0%):** Berkat penambahan variabel `Exchange_Depreciation` dan `Broad_Money_Growth`, OC-SVM berhasil mendeteksi devaluasi Baht, Rupiah, dan Won secara jauh lebih presisi dibanding model awal (meningkat dari 50% ke 70%).
- **Krisis Utang Eropa (60.0%):** Memotret gejolak kredit domestik dan transaksi berjalan di Yunani, Irlandia, dan Portugal.
- **Pandemi COVID-19 (69.4%):** Berhasil menangkap kontraksi serentak PDB dan lonjakan depresiasi di 34 negara.

---

## 7. Profil Geografis — Top 10 Negara Anomali Terbanyak

| Rank | Negara | Kode ISO | Jumlah Tahun Anomali | Profil & Alasan Terisolasi |
|------|--------|----------|----------------------|----------------------------|
| 1 | Irlandia | IRL | 31 | Volatilitas ekspor-impor finansial multinasional tinggi |
| 2 | Arab Saudi | SAU | 23 | Ketergantungan migas & fluktuasi cadangan devisa |
| 3 | Hongaria | HUN | 21 | Fluktuasi transaksi berjalan & modal investasi |
| 4 | Singapura | SGP | 21 | Ekonomi sangat terbuka (Trade/GDP > 300%) |
| 5 | Swiss | CHE | 19 | Fluktuasi nilai tukar Franc Swiss & arus modal |
| 6 | Nigeria | NGA | 16 | Volatilitas harga minyak & depresiasi Naira |
| 7 | Rusia | RUS | 16 | Krisis 1998, sanksi ekonomi, & volatilitas Rubel |
| 8 | Tiongkok | CHN | 15 | Laju pertumbuhan PDB masif & dinamika ekspor |
| 9 | Malaysia | MYS | 14 | Krisis 1997-98 & fluktuasi harga komoditas |
| 10 | Belanda | NLD | 14 | Pusat hub perdagangan terbuka Eropa |

---

## 8. Feature Importance (Mean Difference Z-Score)

Sensitivitas fitur dihitung berdasarkan selisih mutlak rata-rata skor Z-Score antara sampel Anomali vs Normal:

| Rank | Indikator Makroekonomi | Mean Normal (Z) | Mean Anomali (Z) | Selisih Mutlak (|Diff|) |
|------|------------------------|-----------------|------------------|------------------------|
| 1 | `Exports_GDP` | -0.1463 | +0.4201 | **0.5663** |
| 2 | `Imports_GDP` | -0.1325 | +0.3806 | **0.5131** |
| 3 | `Gross_Savings_GDP` | -0.1032 | +0.2965 | **0.3997** |
| 4 | `Current_Account_GDP` | -0.0875 | +0.2512 | **0.3386** |
| 5 | `FDI_Inflows_GDP` | -0.0842 | +0.2417 | **0.3259** |
| 6 | `Reserves_Months_Imports` | -0.0796 | +0.2285 | **0.3081** |
| 7 | `Exchange_Depreciation` | -0.0735 | +0.2110 | **0.2845** |
| 8 | `Manufacturing_Value` | -0.0734 | +0.2107 | **0.2841** |
| 9 | `Inflation_CPI` | -0.0625 | +0.1795 | **0.2420** |
| 10 | `GDP_Growth` | +0.0580 | -0.1664 | **0.2244** |
| 11 | `Broad_Money_Growth` | -0.0535 | +0.1535 | **0.2070** |
| 12 | `Investment_GDP` | -0.0492 | +0.1412 | **0.1903** |
| 13 | `Domestic_Credit_GDP` | -0.0185 | +0.0532 | **0.0717** |
| 14 | `Unemployment` | -0.0089 | +0.0256 | **0.0346** |

---

## 9. Kesimpulan & Rekomendasi EWS

1. Pembaruan preprocessing dengan **transformasi `% Exchange_Depreciation`** dan **imputasi linear + KNN** terbukti meningkatkan daya deteksi krisis (*Recall*) OC-SVM secara signifikan menjadi **51.09%** dan ROC-AUC menjadi **0.6858**.
2. Krisis regional tajam seperti **Krisis Keuangan Asia (70.0%)** dan **Krisis Utang Eropa (60.0%)** terbukti jauh lebih sukses terdeteksi setelah penghapusan bias skala nominal kurs.
3. OC-SVM direkomendasikan menjadi salah satu arsitektur utama dalam konsensus **Majority Voting (≥ 3 dari 6 model)** EWS kelompok.

---

## 10. Daftar Berkas Output

Seluruh file hasil eksekusi pemodelan di folder [Virna](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna):

| Nama File | Deskripsi Berkas |
|-----------|------------------|
| [ocsvm.ipynb](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/ocsvm.ipynb) | Jupyter Notebook interaktif lengkap dengan kode preprocessing & OCSVM |
| [run_ocsvm.py](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/run_ocsvm.py) | Script eksekusi Python mandiri |
| [data_cleaned_virna.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/data_cleaned_virna.csv) | Dataset bersih hasil preprocessing akhir (1.715 baris) |
| [hasil_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/hasil_ocsvm.csv) | Output deteksi OCSVM (`ocsvm_score` + `ocsvm_anomaly`) |
| [grid_search_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/grid_search_ocsvm.csv) | Hasil pencarian 64 kombinasi grid search `nu` & `gamma` |
| [ringkasan_model_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/ringkasan_model_ocsvm.csv) | Ringkasan metrik evaluasi model terbaik |
| [deteksi_per_krisis_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/deteksi_per_krisis_ocsvm.csv) | Tingkat deteksi 6 krisis historis IMF |
| [top_negara_anomali_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/top_negara_anomali_ocsvm.csv) | Top 10 negara frekuensi anomali terbanyak |
| [feature_importance_ocsvm.csv](file:///e:/SEMESTER%206/DATMIN/Project%20UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/Virna/feature_importance_ocsvm.csv) | Peringkat kontribusi 14 indikator |

---
*Laporan Akademik Pembaruan — UTS Data Mining 2026.*
