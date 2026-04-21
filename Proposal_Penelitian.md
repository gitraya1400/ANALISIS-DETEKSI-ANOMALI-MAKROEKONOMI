# Proposal Penelitian: Deteksi Anomali pada Indikator Makroekonomi sebagai *Early Warning System* Krisis Keuangan Menggunakan Pendekatan Agregasi *Unsupervised Learning* Lintas-Paradigma

---

## 1. Latar Belakang

Siklus boom-and-bust dalam perekonomian global telah berulang kali membuktikan bahwa krisis ekonomi bukan sekadar anomali statistik sesaat, melainkan **bencana sistemik berskala masif** yang mampu menyapu bersih puluhan tahun kemajuan suatu bangsa. Krisis Keuangan Asia (1997-1998), *Global Financial Crisis* (2008), hingga kejutan resesi akibat Pandemi COVID-19 (2020) telah memberikan pelajaran memilukan: hilangnya triliunan dolar dari PDB, kebangkrutan massal institusi keuangan, hiperinflasi absolut, hingga lonjakan pengangguran ekstrem yang memicu kerusuhan dan instabilitas sosial-politik global. **Keterlambatan sepersekian bulan saja dalam memitigasi alarm krisis dapat berarti jurang resesi berkepanjangan.** 

Oleh karena itu, membangun sebuah *Early Warning System* (EWS) yang proaktif, mutakhir, dan komprehensif adalah sebuah **urgensi absolut** yang tidak bisa ditawar lagi. EWS tradisional yang ada saat ini mayoritas masih ditopang oleh model ekonometrika konvensional (*probit/logit models*) yang terkekang pada asumsi linearitas dan sebaran data yang normal. Dalam kenyataan riilnya, keruntuhan ekonomi memiliki topologi volatilitas yang liar, memiliki kausalitas non-linear (*heavy-tailed distributions*), serta berantai antar-kawasan dengan *contagion effect* yang kompleks.

Ketika dunia akademik beralih ke ranah *Machine Learning*, seringkali riset kandas oleh absennya kesepakatan target (label krisis kualitatif) sehingga menghambat pelatihan algoritma *Supervised Learning*. Krisis adalah *rare-events* berdimensi tinggi yang bentuk kemunculannya selalu bermutasi. Oleh karenanya, pendekatan **Unsupervised Learning** hadir sebagai satu-satunya *Holy Grail* untuk memetakan titik-titik guncangan aneh dalam ekonomi makro tanpa perlu mendikte algoritma terkait seperti apa bentuk masa lalu krisis tersebut.

Namun, tidak ada satu algoritma matematis pun yang sempurna. Sebuah pergeseran indikator mungkin terlihat wajar secara kerapatan kompresi linier, tapi terlihat fatal dari perspektif kurungan batas (*boundary*). Oleh karena itu, penelitian ini sangat mendesak karena **mengajukan sebuah kontribusi kebaruan melalui sistem agregasi algoritma lintas-paradigma murni (Ensemble of Unsupervised Anomalies)**. Memanfaatkan data dari *World Bank Open Data* terhadap 49 negara dari tahun 1990—2024, penelitian ini akan mengadu dan menyelaraskan enam filosofi algoritma *(Isolation Forest, LOF, Autoencoder, One-Class SVM, PCA Reconstruction, dan DBSCAN)* untuk membentuk sinyal alarm EWS berkonsensus *Majority Voting* tertinggi yang bebas bias algoritma tunggal.

## 2. Rumusan Masalah

Berdasarkan kuatnya urgensi empiris dan teoretis di atas, maka rumusan masalah dalam penelitian ini adalah:
1. Bagaimana cara merancang arsitektur model *Early Warning System* tanpa label (*unsupervised*) yang mempu mengidentifikasi krisis makroekonomi secara otomatis?
2. Bagaimana kinerja (akurasi/sensitivitas deteksi) dari enam paradigma matematis anomalus (*Isolation Forest, LOF, Autoencoder, OCSVM, PCA, DBSCAN*) dalam mendeteksi anomali pada 14 indikator fundamental makro ekonomi?
3. Sejauh mana sistem agregasi konsensus lintas-paradigma memitigasi kesalahan *false positive* (alarm palsu) yang kerap dihasilkan ketika merilis EWS berbasis kecerdasan buatan dalam memutus kebijakan ekonomi vital?

## 3. Tujuan Penelitian

1. **Membangun arsitektur cerdas pendeteksi tekanan sistemik** berbahan dasar integrasi *World Bank API* berbasis *Unsupervised Learning* tanpa intervensi *labeling* subjektif pakar.
2. **Mendemonstrasikan komparasi performa dan ekstraksi fitur (SHAP)** untuk kelancaran interpretasi para pengambil kebijakan pada enam metodologi: Density (LOF), Threshold Ensembel (IForest), Deep Non-linear (Autoencoder), Hyperplane (OCSVM), Spatial Cluster (DBSCAN), dan Linear Variance (PCA).
3. **Menciptakan konsensus pendeteksian akhir (*Majority Voting*)** yang terbukti secara objektif lebih stabil, presisi, dan *robust* bila diadu melawan catatan *ground truth* krisis historis global.

## 4. Tinjauan Pustaka

Penelitian ini disandarkan pada fondasi yang amat kuat melalui sintesis puluhan kepustakaan bertaraf internasional. Deteksi krisis makroekonomi menuntut model untuk menganalisa tumpukan indikator di ruang dimensi tinggi. 

- **Density & Spatial-Based (LOF & DBSCAN):** Breunig et al. merumuskan prinsip metrik kepadatan relatif (LOF), yang dalam konteks ekonomi mampu mendeteksi suatu kawasan yang anjlok pertumbuhannya biarpun negara berkembang tetangganya stabil. Di satu sisi, konsep *Noise Spatial* DBSCAN terbukti relevan dalam riset *time series* fluktuasi tingkat parah bursa aset mata uang (bram_papers).
- **Ensemble Isolation (Isolation Forest):** Metode dari keluarga pohon acak ini (Raya_papers) diakui di pelbagai literatur mutakhir (misal: *Anomaly Detection through on-line Isolation*) sebagai peretas anomali sejati; algoritma ini tak fokus membentuk profil normalitas, namun langsung memanen keganjilan berdasarkan instrumen *path length* pendek.
- **Deep Learning Konseptual (Autoencoder) dan OCSVM:** Jaringan *Neural-Network* bertipe *bottleneck-decoder* digunakan banyak peneliti (Deka_papers) untuk mendulang persentase akurasi *financial distress* jauh melampaui regresi linear sederhana, mengingat efek panik investor bergerak fluktuatif ekstrem. Di samping itu, pengenalan pembatasan eksklusif OCSVM di jurnal fusi anomali multi-dimensi (Virna_papers) menunjukkan efisiensinya dalam mengeksekusi parameter margin error di bursa indikator.

## 5. Tabel Perbandingan Literatur

| No | Fokus / Paradigma Riset | Nama File Literatur Utama | Konteks Kontribusi Jurnal (Relevansi) | Algoritma Target |
|----|-------------------------|--------------------------------|--------------------------------------|------------------|
| 1 | **Density-based & Computer Networks/Broad Systems** | `LOF Identifying Density-Based Local Outliers.pdf` (oleh Breunig dkk.) & `Application of Local Outlier Factor Algorithm...` (Amir) | Konsepsi dasar LOF yang menggunakan skor anomali lokal, berguna untuk meretas pola krisis berskala regional. | LOF |
| 2 | **Deep Learning / Financial Time Series & Early Warning** | `A deep Learning framework for financial time series using stacked autoencoders...` & `Anomaly Detection Using Autoencoders` (Deka) | Menyajikan kegunaan sel neural untuk mengenkripsi fitur volatilitas *financial distress* bursa secara non-linear. | Autoencoder |
| 3 | **Spatial & Data Cryptocurrency / Finance Time Series** | `A Modified DBSCAN Algorithm for Anomaly Detection in Time Series Data.pdf` & `Original Paper for DBSCAN.pdf` (Bram) | Eksplorasi deteksi kelompok kepadatan titik-titik ekuilibrium waktu menggunakan e-radius (epsilon) DBSCAN pada indikator finansial. | DBSCAN |
| 4 | **Ensemble Tree / Streaming Data Anomaly** | `Isolation Forest Based Anomaly Detection: A Systematic Literature Review.pdf` & `Isolation Forest.pdf` (Raya) | Memvalidasi ketangguhan pohon isolasi demi mencari *outliers* mutlak dengan kerumitan kalkulasi yang termurah (*low computational cost*). | Isolation Forest |
| 5 | **Financial Shock & General Macro Warning System** | `Global Shocks and Local Fragilities A Financial Stress Index Approach...` & `Early Warning of Enterprise Financial Risk Based on Decision.pdf` (Bela) | Memberi pandangan fundamental mengenai urgensi EWS, indeks stres likuiditas pasar, dan betapa berbahayanya guncangan kejutan global terhadap negara. | Domain/Framework (Macro) |
| 6 | **Hyperplane Boundary & High-Dimensional Systems** | `Active Anomaly Detection Based on Deep One-Class Classification.pdf` & `Efficient Anomaly Detection for... One-Class SVM.pdf` (Virna) | Penjabaran pembungkusan limit data mayoritas melalui *kernel trick* RBF demi memerangkap batas indikator sehat (*normal-boundary*). | One-Class SVM |

---

## 6. Contoh Output Visualisasi Preliminer (Dari Eksperimentasi Laporan)

Di bawah ini dilampirkan bentuk uji pembuktian preliminer yang menunjukkan bahwa model-model arsitektur mandiri (*Unsupervised*) telah berhasil menunjukkan potensi ekstraksi sinyal krisis. *Catatan: Gambar hasil langsung ditarik dari artefak internal file notebook (`.ipynb`)*.

### 6.1 Anomali Tren Indikator Ekonomi Berbasis Isolation Forest
![Potensi Deteksi Anomali - Isolation Forest](file:///d:/Perkuliahan/SEMESTER%206/Data%20Mining/UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/extracted_images/raya_img_14.png)
*(Sumber: Hasil plot model Isolation Forest - folder raya)*

### 6.2 Anomali dan Kesalahan Rekonstruksi (Reconstruction Error) dengan Autoencoder
![Potensi Deteksi Anomali - Autoencoder Volatilitas](file:///d:/Perkuliahan/SEMESTER%206/Data%20Mining/UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/extracted_images/deka_img_14.png)
*(Sumber: Hasil plot kompresi dan dekompresi data Makro pada Autoencoder - folder deka)*

### 6.3 Analisis Keruangan (Spatial Boundary) Melalui LOF 
![Sebaran Skor Anomali Local - LOF](file:///d:/Perkuliahan/SEMESTER%206/Data%20Mining/UTS/ANALISIS-DETEKSI-ANOMALI-MAKROEKONOMI/extracted_images/Amir_img_13.png)
*(Sumber: Hasil plot skor LOF - folder Amir)*

**(Catatan untuk Tim OCSVM dan PCA: Lampiran visual OCSVM dan PCA akan disematkan di sini paska proses komputasi diselesaikan di sprint selanjutnya.**)

---

## 7. Daftar Pustaka

1. Al-Amri, R., dkk. (2021). *Application of Local Outlier Factor Algorithm to Detect Anomalies in Computer Network*. IEEE.
2. Bao, W., Yue, J., & Rao, Y. (2017). *A deep learning framework for financial time series using stacked autoencoders and long-short term memory*. PLoS ONE.
3. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD Record.
4. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD.
5. Kaminsky, G., Lizondo, S., & Reinhart, C. M. (1998). *Leading Indicators of Currency Crises*. IMF Staff Papers.
6. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. IEEE International Conference on Data Mining.
7. Qiao, Y., dkk. (2020). *Early Warning of Enterprise Financial Risk Based on Decision Tree*. IEEE.
8. Holopainen, M., & Sarlin, P. (2017). *Toward robust early-warning models: A horse race, ensembles and model uncertainty*. Quantitative Finance.
9. Razi, A., dkk. (2022). *Global Shocks and Local Fragilities A Financial Stress Index Approach to Pakistan’s Monetary and Asset Market Dynamics*. JEM.
10. Ruff, L., dkk. (2018). *Deep One-Class Classification for Anomaly Detection*. ICML.

---
*Draf Dokumen Proposal V-1.0 (Synthesized from Laporan_UTS.md & Collective Papers)*
