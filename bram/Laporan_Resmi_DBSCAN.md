# LAPORAN RESMI PROYEK DATA MINING
# DETEKSI ANOMALI MAKROEKONOMI DAN KRISIS FINANSIAL MENGGUNAKAN METODE UNSUPERVISED LEARNING DBSCAN
### Studi Kasus: 49 Negara di Dunia Periode 1990 – 2024

---

## DAFTAR ISI
1. [RINGKASAN EKSEKUTIF](#ringkasan-eksekutif)
2. [BAB I: PENDAHULUAN](#bab-i-pendahuluan)
   - 1.1 Latar Belakang Masalah
   - 1.2 Mengapa Menggunakan Unsupervised Learning?
   - 1.3 Rumusan Masalah & Tujuan Penelitian
3. [BAB II: MEMAHAMI METODE DBSCAN (UNTUK ORANG AWAM)](#bab-ii-memahami-metode-dbscan-untuk-orang-awam)
   - 2.1 Apa Itu DBSCAN?
   - 2.2 Analogi Sederhana: Pesta di Gedung Pertemuan
   - 2.3 Konsep Inti DBSCAN: Core Points, Border Points, dan Noise (Anomali)
   - 2.4 Dua Parameter Kunci: Epsilon ($\epsilon$) dan MinPts
   - 2.5 Mengapa DBSCAN, Bukan K-Means?
4. [BAB III: DATA DAN LANGKAH PREPROCESSING](#bab-iii-data-dan-langkah-preprocessing)
   - 3.1 Profil Data Awal (Raw Data)
   - 3.2 Langkah 1: Penanganan Missing Values (Imputasi KNN)
   - 3.3 Langkah 2: Deteksi & Penanganan Outlier (Winsorization)
   - 3.4 Langkah 3: Skalasi Data (RobustScaler)
   - 3.5 Langkah 4: Reduksi Dimensi (Principal Component Analysis / PCA)
5. [BAB IV: EKSPERIMEN & IMPLEMENTASI MODEL DBSCAN](#bab-iv-eksperimen--implementasi-model-dbscan)
   - 4.1 Menentukan Nilai Epsilon Optimal ($k$-Distance Graph & Knee Point)
   - 4.2 Pendekatan A: DBSCAN Skala Global
   - 4.3 Pendekatan B: DBSCAN Skala Per-Negara (*Country-Specific*)
   - 4.4 Perbandingan Pola Temuan Global vs Per-Negara
6. [BAB V: EVALUASI KOMPREHENSIF DAN ANALISIS HASIL](#bab-v-evaluasi-komprehensif-dan-analisis-hasil)
   - 5.1 Validasi Menggunakan Ground Truth Krisis IMF
   - 5.2 Evaluasi Klasifikasi: Confusion Matrix, Precision, Recall, F1, ROC-AUC
   - 5.3 Evaluasi untuk Data Tidak Seimbang: Precision-Recall Curve (PR-AUC)
   - 5.4 Evaluasi Kualitas Klaster Tanpa Label (Silhouette, Davies-Bouldin, Calinski-Harabasz)
   - 5.5 Daya Tangkap Model Berdasarkan Jenis Krisis (Perbankan, Mata Uang, Utang, COVID-19)
7. [BAB VI: STUDI KASUS VALIDASI HISTORIS NYATA](#bab-vi-studi-kasus-validasi-historis-nyata)
   - 6.1 Krisis Moneter Asia 1997–1998 (Indonesia & Thailand)
   - 6.2 Hiperinflasi Amerika Latin 1990–1994 (Brazil & Peru)
   - 6.3 Krisis Utang Eropa 2010–2012 (Yunani)
   - 6.4 Guncangan Pandemi COVID-19 Tahun 2020
8. [BAB VII: KESIMPULAN, KETERBATASAN, DAN REKOMENDASI](#bab-vii-kesimpulan-keterbatasan-dan-rekomendasi)
   - 7.1 Kesimpulan Utama
   - 7.2 Keterbatasan Analisis
   - 7.3 Rekomendasi untuk Pembuat Kebijakan & Penelitian Lanjutan

---

## RINGKASAN EKSEKUTIF

Krisis ekonomi adalah peristiwa langka yang membawa dampak kehancuran sosial-ekonomi sangat besar bagi suatu bangsa. Sayangnya, memprediksi atau mendeteksi kedatangan krisis adalah tugas yang sangat rumit karena krisis jarang berulang dengan pola yang persis sama. 

Laporan resmi ini menyajikan perancangan dan implementasi sistem **Deteksi Anomali Makroekonomi Menggunakan Algoritma Unsupervised Learning DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) pada dataset panel 49 negara selama 35 tahun (1990–2024) yang mencakup 14 indikator ekonomi utama.

**Pertanyaan Kunci:** *Mampukah algoritma komputer mendeteksi krisis ekonomi secara otomatis hanya dengan mencari "keanehan data", tanpa pernah diajari atau diberi tahu contoh krisis sebelumnya?*

**Temuan & Kesimpulan Utama:**
1. **DBSCAN Terbukti Efektif Menangkap Krisis Riil**: Tanpa bantuan label historis (*unsupervised*), model DBSCAN dengan pendekatan Per-Negara berhasil mendeteksi **32.3% dari seluruh krisis ekonomi nyata** yang dicatat oleh IMF (*International Monetary Fund*), menghasilkan **F1-score 0.354** dan **ROC-AUC 0.638** (jauh melampaui tebakan acak yang ber-AUC 0.500).
2. **Pendekatan "Per-Negara" Jauh Lebih Unggul Dibanding Pendekatan "Global"**:
   - Pendekatan Global menetapkan satu standar normal yang seragam untuk seluruh dunia. Akibatnya, model Global hanya mendeteksi 13 anomali ekstrem di seluruh dunia (Recall krisis hanya 3.5%). Model ini gagal melihat krisis di negara-negara maju karena indikator mereka tidak seekstrem negara berkembang yang mengalami hiperinflasi.
   - Pendekatan Per-Negara mempelajari sejarah masing-masing negara secara mandiri. Model ini berhasil mendeteksi 189 titik anomali, mencakup krisis-krisis besar seperti Krisis Keuangan Asia 1997/1998 (Indonesia, Thailand), Krisis Utang Yunani 2010, hingga Krisis Subprime Mortgage 2008 di Amerika Serikat.
3. **Krisis Mata Uang dan Pandemi Paling Mudah Terbaca**: Berdasarkan evaluasi granular jenis krisis, DBSCAN paling peka terhadap **Pandemi COVID-2020 (Recall 63.3%)** dan **Krisis Mata Uang / Currency Crisis (Recall 54.3%)**, karena peristiwa ini memicu guncangan langsung yang simultan pada variabel makroekonomi (GDP, cadangan devisa, dan nilai tukar). Sebaliknya, Krisis Perbankan (Recall 23.4%) lebih menantang karena kerentanan sering kali tersembunyi di neraca internal bank sebelum merembet ke ekonomi agregat.

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang Masalah
Kondisi ekonomi suatu negara ibarat kondisi kesehatan tubuh manusia. Dalam kondisi normal, indikator-indikator vital (seperti tekanan darah dan detak jantung pada manusia, atau pertumbuhan ekonomi, inflasi, dan cadangan devisa pada negara) bergerak dalam rentang batas yang wajar dan stabil. Namun, ketika penyakit ganas atau krisis menyerang, indikator-indikator vital tersebut akan melonjak atau anjlok secara drastis ke wilayah yang tidak normal.

Sepanjang sejarah modern (1990–2024), dunia telah menyaksikan berbagai guncangan hebat:
- **Krisis Moneter Asia 1997/1998**: Nilai tukar Rupiah Indonesia ambruk dari Rp2.500 menjadi lebih dari Rp15.000 per USD, memicu inflasi di atas 70% dan kejatuhan rezim politik.
- **Krisis Keuangan Global (GFC) 2008**: Kebangkrutan Lehman Brothers di Amerika Serikat merembet ke sektor perbankan Eropa dan melumpuhkan perdagangan internasional.
- **Krisis Utang Eropa 2010–2012**: Yunani mengalami defisit utang yang tidak terkendali, pengangguran melonjak hingga 27%, dan ekonomi menyusut hingga seperempatnya.
- **Pandemi COVID-19 2020**: Penutupan aktivitas (*lockdown*) global yang memukul serentak seluruh perekonomian dunia.

Tantangan utama bagi otoritas moneter, bank sentral, dan kementerian keuangan adalah: **Bagaimana cara mendeteksi secara objektif saat perekonomian mulai tergelincir ke dalam kondisi anomali yang berbahaya?**

### 1.2 Mengapa Menggunakan Unsupervised Learning?
Dalam dunia data science, terdapat dua pendekatan utama:
1. **Supervised Learning (Pembelajaran Terawasi)**: Model diberi tahu ribuan contoh data dengan label: *"Ini tahun normal, ini tahun krisis"*. Model lalu menghafal pola tersebut untuk memprediksi masa depan.
   - *Kelemahan pada Kasus Krisis*: Krisis ekonomi sangat jarang terjadi (dalam dataset kita, hanya sekitar 13.3% dari seluruh tahun amatan yang merupakan krisis). Selain itu, krisis di masa depan hampir selalu memiliki karakteristik pemicu yang berbeda dari masa lalu (*"Black Swan Event"*). Jika model hanya dilatih pada krisis perbankan masa lalu, ia akan gagal mengenali krisis baru seperti pandemi global.
2. **Unsupervised Learning (Pembelajaran Tak Terawasi)**: Model **sama sekali tidak diberi label** krisis. Model hanya disodori data mentah dan diminta untuk mencari sendiri struktur alaminya: *"Mana titik-titik yang berkelompok membentuk pola umum (normal), dan mana titik-titik yang terlempar jauh sendirian (anomali)?"*

Pendekatan *Unsupervised Learning* jauh lebih adil, objektif, dan tangguh terhadap jenis krisis baru yang belum pernah tercatat dalam sejarah sebelumnya.

### 1.3 Rumusan Masalah & Tujuan Penelitian
Penelitian ini bertujuan untuk:
1. Membangun alur preprocessing data makroekonomi yang tangguh terhadap data hilang (*missing values*) dan outlier ekstrem tanpa merusak sinyal krisis.
2. Menerapkan algoritma **DBSCAN** untuk mendeteksi anomali pada dua skala: **Global** (skala dunia) dan **Per-Negara** (*country-specific*).
3. Mengevaluasi performa deteksi anomali DBSCAN terhadap data historis resmi dari **IMF Crisis Database (Laeven & Valencia)** menggunakan metrik klasifikasi seimbang, kurva Precision-Recall, dan metrik kualitas klaster.
4. Memberikan pemahaman naratif yang mudah dimengerti mengenai alasan di balik setiap keputusan teknis ("kenapa begini, kenapa begitu").

---

## BAB II: MEMAHAMI METODE DBSCAN (UNTUK ORANG AWAM)

### 2.1 Apa Itu DBSCAN?
**DBSCAN** adalah singkatan dari *Density-Based Spatial Clustering of Applications with Noise*. Diciptakan oleh Martin Ester dkk. pada tahun 1996, DBSCAN adalah salah satu algoritma pengelompokan (*clustering*) paling terkenal di dunia data science.

Sesuai namanya (*Density-Based*), cara berpikir DBSCAN didasarkan pada **kepadatan**. Ide dasarnya sangat intuitif:
> *"Hal-hal yang normal biasanya terjadi berulang kali dan memiliki banyak kembaran, sehingga membentuk kerumunan yang padat. Sebaliknya, hal yang aneh atau krisis adalah kejadian luar biasa yang jarang terjadi, sehingga posisinya terpencil sendirian di wilayah yang sunyi."*

### 2.2 Analogi Sederhana: Pesta di Gedung Pertemuan
Bayangkan sebuah ruangan pesta yang sangat luas berisi 1.000 orang tamu:
- Sebagian besar tamu (900 orang) berkumpul mengobrol di dekat meja prasmanan dan panggung musik. Mereka berdiri berdekatan satu sama lain, membentuk **kelompok-kelompok yang padat**. Ini adalah representasi dari **Kondisi Ekonomi Normal**.
- Namun, ada 5 orang tamu yang bertingkah aneh: satu orang berdiri memojok di lorong tangga darurat yang gelap, satu orang melamun sendirian di toilet, dan satu orang mondar-mandir di tempat parkir sepi. Mereka tidak memiliki teman di sekitarnya.
- Jika ada petugas keamanan yang mencari *"tamu-tamu yang mencurigakan"*, petugas tersebut cukup mencari: **Siapa orang-orang yang berada di area sunyi dan berjarak jauh dari kerumunan utama?** 
- Itulah persis cara kerja DBSCAN! Orang-orang aneh yang sendirian itu disebut sebagai **Noise (Anomali/Krisis)**.

### 2.3 Konsep Inti DBSCAN: Core, Border, dan Noise
DBSCAN membagi setiap titik data ke dalam 3 kategori berdasarkan kepadatannya:

1. **Titik Inti (*Core Point*)**:
   Titik data yang berada di pusat kerumunan. Syaratnya: dalam radius lingkaran jarak tertentu ($\epsilon$), titik tersebut memiliki minimal sejumlah teman tetangga ($MinPts$). Titik inti adalah pondasi utama dari kondisi ekonomi normal.
2. **Titik Perbatasan (*Border Point*)**:
   Titik data yang jumlah tetangganya tidak cukup banyak untuk menjadi titik inti, tetapi ia masih cukup dekat dan menempel pada salah satu titik inti. Diibaratkan seperti orang yang berdiri di pinggir lingkaran kerumunan. Titik ini masih dianggap sebagai bagian dari kondisi normal.
3. **Titik Anomali / Noise (*Outlier / Crisis*)**:
   Titik data yang bukan merupakan titik inti dan tidak terjangkau oleh titik inti mana pun. Titik ini terisolasi di ruang kosong yang sepi. **Inilah titik yang oleh DBSCAN ditandai dengan label `-1` alias ANOMALI.**

```
       [Titik Inti] ──── Banyak teman di sekitarnya (Kondisi Normal)
            │
       [Titik Border] ── Di pinggir kerumunan (Normal Variatif)
            
                              . . . (jarak kosong yang jauh) . . .
                                    
       [Titik Noise] ─── Terpencil sendirian (ANOMALI / KRISIS)
```

### 2.4 Dua Parameter Kunci: Epsilon ($\epsilon$) dan MinPts
DBSCAN hanya membutuhkan dua parameter utama untuk bekerja:
1. **Epsilon ($\epsilon$ / eps)**: Jari-jari lingkaran pencarian tetangga. 
   - *Kenapa parameter ini krusial?* Jika $\epsilon$ disetel terlalu kecil, kerumunan yang sebenarnya normal akan dianggap anomali semua. Sebaliknya, jika $\epsilon$ disetel terlalu besar, titik anomali pun akan tertelan masuk ke dalam kerumunan sehingga tidak ada krisis yang terdeteksi.
2. **MinPts (*Minimum Points*)**: Jumlah tetangga minimal yang harus ada di dalam radius $\epsilon$ agar suatu titik diakui sebagai titik inti kerumunan.

### 2.5 Mengapa DBSCAN, Bukan K-Means?
Dalam data science, algoritma clustering paling populer yang sering diajarkan adalah **K-Means**. Mengapa kita tidak menggunakan K-Means untuk proyek deteksi anomali ini?

| Karakteristik | K-Means | DBSCAN (Pilihan Kita) | Kenapa DBSCAN Lebih Cocok? |
| :--- | :--- | :--- | :--- |
| **Jumlah Klaster** | Harus ditentukan di awal oleh manusia ($k=2, 3, 5$). | Otomatis menemukan sendiri berapa jumlah klaster alami. | Kita tidak tahu sebelumnya berapa banyak jenis pola ekonomi yang ada. |
| **Bentuk Klaster** | Hanya bisa mengenali klaster berbentuk bola/lingkaran bulat simetris. | Mampu mengenali klaster dengan bentuk arbitrer bebas (memanjang, melengkung). | Hubungan ekonomi (misal inflasi vs pengangguran) bersifat non-linier dan melengkung. |
| **Perlakuan Outlier** | Memaksa SEMUA titik masuk ke salah satu klaster. Titik anomali akan merusak titik pusat klaster (*centroid*). | Memiliki mekanisme bawaan (*built-in*) untuk membuang titik anomali sebagai **Noise**. | **Tujuan utama kita adalah menemukan outlier tersebut!** |

---

## BAB III: DATA DAN LANGKAH PREPROCESSING

### 3.1 Profil Data Awal (Raw Data)
Dataset yang dianalisis mencakup data panel **49 negara** dari berbagai belahan dunia (ekonomi maju seperti USA, Jerman, Jepang; ekonomi berkembang seperti Indonesia, Brazil, India; hingga negara rentan) dalam kurun waktu **35 tahun (1990 hingga 2024)**. 
Total terdapat **1.715 baris observasi** ($49 \times 35$) dengan **14 indikator makroekonomi utama**:

1. **Pertumbuhan PDB Riil (*gdp_growth*)**: Laju pertumbuhan ekonomi tahunan.
2. **PDB Per Kapita (*gdp_per_capita*)**: Tingkat kemakmuran rata-rata penduduk.
3. **Tingkat Inflasi (*inflation_cpi*)**: Kenaikan harga barang dan jasa secara umum.
4. **Tingkat Pengangguran (*unemployment*)**: Persentase angkatan kerja yang tidak bekerja.
5. **Cadangan Devisa (*reserves_total*)**: Cadangan mata uang asing dan emas yang dikuasai bank sentral.
6. **Rasio Cadangan terhadap Impor (*reserves_months_import*)**: Ketahanan cadangan devisa dalam membiayai impor bulanan.
7. **Perubahan Nilai Tukar (*fx_depreciation_pct*)**: Persentase depresiasi nilai tukar domestik terhadap mata uang acuan dunia.
8. **Pertumbuhan Kredit Domestik (*credit_growth*)**: Laju ekspansi pembiayaan ke sektor swasta.
9. **Kredit terhadap PDB (*credit_to_gdp*)**: Kedalaman intermediasi finansial.
10. **Neraca Transaksi Berjalan (*current_account_gdp*)**: Surplus/defisit perdagangan barang dan jasa internasional (% PDB).
11. **Utang Luar Negeri (*external_debt_gdp*)**: Beban utang luar negeri terhadap kapasitas ekonomi (% PDB).
12. **Saldo Fiskal Pemerintah (*fiscal_balance_gdp*)**: Surplus atau defisit anggaran belanja negara (% PDB).
13. **Utang Pemerintah (*gov_debt_gdp*)**: Total akumulasi utang sektor publik (% PDB).
14. **Suku Bunga Riil (*real_interest_rate*)**: Suku bunga nominal dikurangi laju inflasi.

---

### 3.2 Langkah 1: Penanganan Missing Values (Imputasi KNN)
**Kondisi Awal**: Pada data mentah, terdapat 573 sel data yang kosong (*missing values*), tersebar di berbagai negara dan tahun (terutama pada indikator utang pemerintah, kredit, dan suku bunga riil di era 1990-an awal).

**Kenapa Begitu (Mengapa Tidak Dihapus Saja?)**:
- Jika kita menghapus baris yang memiliki nilai kosong (*drop rows*), kita akan kehilangan ratusan baris data berharga. Bahkan kita berisiko menghapus tahun-tahun krusial di mana krisis sedang terjadi (karena di masa krisis, instrumen pencatatan statistik negara sering kali terganggu sehingga data menjadi bolong).
- Jika kita mengisi dengan nilai rata-rata (*mean imputation*), varians data akan mengecil secara palsu dan hubungan antar-variabel akan hancur.

**Solusi: Imputasi K-Nearest Neighbors (KNN Imputer)**:
- Algoritma KNN mencari 5 negara lain yang memiliki karakteristik paling mirip pada indikator-indikator lain yang lengkap, lalu meminjam nilai rata-rata dari "tetangga terdekat" tersebut untuk mengisi sel yang kosong.
- **Hasil**: Seluruh 1.715 baris data kini terisi utuh (0% missing values) dengan tetap mempertahankan korelasi struktural antar-indikator ekonomi.

---

### 3.3 Langkah 2: Deteksi & Penanganan Outlier Ekstrem (Winsorization)
**Kondisi Awal**: Di dunia nyata, beberapa negara pernah mengalami peristiwa ekonomi yang begitu mencengangkan hingga angkanya menjadi pencilan ultra-ekstrem. Sebagai contoh, inflasi di Peru pada tahun 1990 mencapai **7.481%**, dan di Brazil mencapai **2.735%**. 

**Kenapa Begitu (Dilema Outlier dalam Deteksi Anomali)**:
- Ini adalah dilema terbesar: Kita ingin mendeteksi anomali, tetapi jika ada satu angka 7.481% yang dibiarkan liar, angka itu akan "merusak timbangan" statistik. Variabel inflasi akan mendominasi seluruh perhitungan jarak, sehingga indikator lain (seperti lonjakan pengangguran 25% di krisis Yunani) menjadi terlihat seperti angka nol yang tidak ada artinya bagi komputer.

**Solusi: Winsorization pada Persentil 1% dan 99%**:
- Angka ekstrem dipotong dan dibatasi pada persentil ke-99 (untuk batas atas) dan persentil ke-1 (untuk batas bawah).
- Sebagai contoh, inflasi 7.481% diturunkan menjadi sekitar 120% (nilai persentil ke-99). 
- **Hasil**: Nilai inflasi tetap tercatat sebagai angka yang "sangat amat tinggi" (tetap berstatus anomali di mata DBSCAN), tetapi skalanya tidak lagi merusak sensitivitas variabel-variabel lain.

---

### 3.4 Langkah 3: Skalasi Data (RobustScaler)
**Kondisi Awal**: Indikator kita memiliki satuan ukuran yang sangat berlainan: PDB per kapita berada di kisaran angka puluhan ribu (misal USD 45.000), sedangkan inflasi berada di angka persen (misal 5%), dan saldo transaksi berjalan berada di angka koma (misal -0.03).

**Kenapa Begitu (Mengapa Harus RobustScaler, Bukan StandardScaler atau MinMaxScaler?)**:
- DBSCAN menghitung jarak geometris antar-titik (*Euclidean distance*). Jika data tidak diskalakan, PDB per kapita sebesar 45.000 akan menganggap perbedaan inflasi 5% sebagai jarak yang tidak berarti.
- *StandardScaler* menggunakan rata-rata (*mean*) dan standar deviasi yang sangat sensitif terhadap outlier.
- *MinMaxScaler* memetakan data ke rentang 0 hingga 1, yang akan membuat sebagian besar data normal berdesakan di rentang sempit jika ada sisa outlier.
- **RobustScaler** menggunakan **Median** (sebagai titik tengah) dan **Interquartile Range / IQR** (rentang persentil 25% hingga 75%) sebagai pembagi. Karena median dan IQR tidak terpengaruh oleh titik ekstrem, skala ini menjadi yang paling adil dan kokoh bagi algoritma berbasis jarak.

---

### 3.5 Langkah 4: Reduksi Dimensi (PCA)
**Kondisi Awal**: Kita memiliki 14 indikator (ruang 14 dimensi).

**Kenapa Begitu (Kutukan Dimensi / Curse of Dimensionality)**:
- Pada ruang 14 dimensi, jarak antar-titik menjadi seragam (*equidistant*). Semua titik terasa berjauhan satu sama lain, sehingga konsep kepadatan (*density*) pada DBSCAN mulai kehilangan ketajamannya. Selain itu, banyak indikator yang saling berkorelasi (misalnya, rasio utang pemerintah dan saldo fiskal sering kali berjalan beriringan).

**Solusi: Principal Component Analysis (PCA)**:
- PCA merangkum 14 indikator tersebut menjadi variabel-variabel baru yang saling tegak lurus (*orthogonal*) yang disebut *Principal Components*.
- **Hasil Scree Plot**:
  - **PC1 (menjelaskan 43.1% varians)**: Merangkum stabilitas moneter dan eksternal (inflasi, depresiasi kurs, cadangan devisa).
  - **PC2 (menjelaskan 20.7% varians)**: Merangkum dinamika sektor riil dan fiskal (pertumbuhan PDB, beban utang pemerintah, pengangguran).
  - **Total 2 Komponen Utama menjelaskan 63.8% seluruh informasi ekonomi dunia**, memungkinkan visualisasi 2 dimensi yang jernih tanpa menghilangkan pola esensial.

---

## BAB IV: EKSPERIMEN & IMPLEMENTASI MODEL DBSCAN

### 4.1 Menentukan Nilai Epsilon Optimal ($k$-Distance Graph & Knee Point)
Bagaimana cara menentukan nilai parameter $\epsilon$ secara objektif tanpa menebak-nebak?
Kita menggunakan metode standar akademis: **Grafik Jarak Tetangga ke-$k$ (*$k$-Distance Graph*)**:

1. Untuk setiap titik data, kita hitung jaraknya ke tetangga terdekat ke-$k$ (di mana $k = MinPts$).
2. Seluruh jarak tersebut diurutkan dari yang paling kecil hingga yang paling besar, lalu diplot ke dalam grafik kurva.
3. **Analisis Titik Siku (*Elbow / Knee Point*)**:
   - Pada titik-titik normal yang berkerumun padat, kurva jarak akan berjalan landai dan rendah.
   - Pada titik di mana kurva tiba-tiba menekuk tajam ke atas (*knee*), itu menandakan transisi dari titik yang berada di dalam kerumunan menuju titik-titik anomali yang terisolasi jauh.
   - Titik siku inilah yang dipilih secara matematis sebagai nilai **$\epsilon$ optimal**.

---

### 4.2 Pendekatan A: DBSCAN Skala Global
Pada pendekatan ini, seluruh 1.715 data dari 49 negara disatukan ke dalam satu ruang koordinat bersama. Model menganggap semua negara di dunia harus mematuhi satu definisi "kenormalan" yang seragam.

- **Parameter yang Dihasilkan**: $\epsilon = 3.16$, $MinPts = 8$.
- **Hasil Deteksi**:
  - Dari 1.715 observasi, model Global hanya menandai **13 titik (0.76%) sebagai anomali**.
  - Sebanyak 1.702 titik lainnya (99.24%) digabungkan ke dalam 1 klaster normal raksasa.
- **Karakteristik Titik yang Terdeteksi**:
  - Peru (1990): Hiperinflasi 7.000%+ (*Score anomali: 1.00 — tertinggi di dunia*).
  - Brazil (1990, 1991, 1992, 1993, 1994): Rangkaian hiperinflasi ribuan persen sebelum reformasi mata uang Real.
  - Indonesia (1998): Krisis Moneter Asia di mana Rupiah jatuh bebas dan inflasi melonjak.
  - Argentina (2002): Krisis keruntuhan konvertibilitas mata uang Peso.
- **Kelemahan Fatal Pendekatan Global**:
  Model ini **terlalu selektif**. Karena standarnya mengacu pada seluruh dunia, model ini hanya mampu menangkap krisis yang menimbulkan "ledakan" angka paling fantastis di muka bumi. Akibatnya, krisis besar di negara maju (seperti Krisis Subprime Mortgage di AS tahun 2008 atau krisis Yunani) tidak terdeteksi sama sekali, karena angka inflasi dan kontraksi ekonomi negara-negara tersebut tidak pernah menyentuh level ribuan persen seperti di Amerika Latin.

---

### 4.3 Pendekatan B: DBSCAN Skala Per-Negara (*Country-Specific*)
Pada pendekatan ini, algoritma DBSCAN dijalankan **secara independen untuk masing-masing negara** pada rentang sejarah 35 tahunnya sendiri. 
- *Kenapa pendekatan ini jauh lebih masuk akal secara ekonomi?*
  Definisi "normal" bagi setiap negara berbeda-beda. Bagi Jepang atau Jerman, pertumbuhan ekonomi 3% dan inflasi 2% adalah kondisi normal yang sangat baik. Sebaliknya, bagi negara berkembang seperti India atau Indonesia, pertumbuhan 3% bisa dianggap sebagai perlambatan parah. Dengan menganalisis negara dalam konteks sejarahnya sendiri, model dapat mendeteksi krisis yang relatif terhadap kondisi dasar negara tersebut.

- **Parameter yang Dihasilkan**: Nilai $\epsilon$ dihitung secara otomatis untuk masing-masing negara berdasarkan kurva $k$-distance lokalnya (rata-rata $\epsilon \approx 2.4$, $MinPts = 4$).
- **Hasil Deteksi**:
  - Model Per-Negara berhasil mendeteksi **189 titik anomali (11.02% dari data)**.
  - Sebarannya sangat proporsional: rata-rata 3 hingga 5 tahun anomali per negara dalam kurun waktu 35 tahun.
  - Model berhasil menangkap krisis di negara berkembang maupun negara maju: Amerika Serikat (2008, 2020), Yunani (2010–2014), Indonesia (1997, 1998, 2020), Korea Selatan (1998), hingga Swedia (1991-1993).

---

## BAB V: EVALUASI KOMPREHENSIF DAN ANALISIS HASIL

Bagaimana kita membuktikan bahwa titik anomali yang ditemukan oleh DBSCAN benar-benar krisis nyata dan bukan sekadar angka acak? Kita mengujinya menggunakan **Ground Truth IMF Historical Crisis Database** (*Laeven & Valencia, Update 2020*), yang mencatat krisis perbankan, krisis mata uang, dan gagal bayar utang di seluruh dunia, ditambah peristiwa global Pandemi COVID-19 tahun 2020.

Dalam data IMF, tercatat ada **229 tahun krisis riil (13.35%)** dan **1.486 tahun kondisi normal (86.65%)**.

### 5.1 Tabel Perbandingan Kinerja Lengkap

| Metrik Evaluasi | DBSCAN Global | DBSCAN Per-Negara | Interpretasi Sederhana |
| :--- | :---: | :---: | :--- |
| **Total Anomali Terdeteksi** | 13 | **189** | Global sangat pelit, Per-Negara proporsional. |
| **True Positive (TP)** | 8 | **74** | Krisis nyata yang berhasil ditemukan model. |
| **False Positive (FP)** | 5 | 115 | Prediksi anomali tapi tidak tercatat krisis IMF (alarm palsu). |
| **False Negative (FN)** | 221 | 155 | Krisis nyata yang luput/kecolongan tidak terdeteksi. |
| **True Negative (TN)** | 1.481 | 1.371 | Tahun normal yang benar diprediksi normal. |
| **Accuracy (Akurasi)** | 86.8% | 84.3% | *Akurasi semu (lihat penjelasan di bawah).* |
| **Precision (Presisi)** | **61.5%** | 39.2% | Dari seluruh alarm anomali, berapa % yang terbukti krisis riil? |
| **Recall (Sensitivitas)** | 3.5% | **32.3%** | **Dari seluruh krisis nyata, berapa % yang berhasil ditangkap?** |
| **F1-Score (Harmonic Mean)** | 0.066 | **0.354** | Keseimbangan presisi dan recall (**Per-Negara 5.3x lebih unggul**). |
| **ROC-AUC Score** | 0.516 | **0.638** | Kemampuan membedakan krisis vs normal (0.500 = tebakan acak). |
| **Average Precision (PR-AUC)** | 0.156 | **0.248** | Kinerja pada data timpang (baseline tebakan acak = 0.134). |
| **Silhouette Score** | **0.777** | 0.273 | Kerapatan klaster internal geometris. |
| **Davies-Bouldin Index** | **1.578** | 5.462 | Separasi klaster (semakin kecil semakin baik). |

---

### 5.2 Membongkar Paradoks Akurasi (*The Accuracy Paradox*)
Perhatikan bahwa DBSCAN Global memiliki akurasi **86.8%**, lebih tinggi daripada DBSCAN Per-Negara (**84.3%**). Apakah ini berarti model Global lebih pintar?

**TIDAK! Ini adalah jebakan klasik dalam ilmu statistik:**
- Jika seorang dokter selalu mendiagnosis *"Semua pasien sehat!"* kepada 1.000 orang pasien (di mana 870 orang memang flu ringan dan 130 orang terkena kanker ganas), dokter tersebut akan memiliki **akurasi 87%**. Namun, dokter itu sama sekali tidak berguna karena ia membiarkan 100% pasien kanker meninggal tanpa diobati!
- DBSCAN Global berkinerja persis seperti dokter tersebut: dari 229 krisis nyata, ia hanya menemukan 8 (Recall hanya **3.5%**). Akurasinya tampak tinggi semata-mata karena ia menebak hampir semua tahun sebagai tahun normal.
- Sebaliknya, **DBSCAN Per-Negara berhasil mendeteksi 74 krisis (Recall 32.3%)**. Ini adalah pencapaian luar biasa mengingat DBSCAN adalah algoritma *unsupervised* yang tidak pernah diajari contoh krisis sebelumnya.

---

### 5.3 Evaluasi Precision-Recall Curve (PR-AUC)
Pada kasus deteksi kejadian langka (*imbalanced dataset* di mana kelas positif hanya 13.3%), kurva ROC-AUC sering memberikan kesan manis yang palsu karena tingginya angka *True Negative* (ribuan tahun normal). 

Oleh karena itu, standar evaluasi modern mewajibkan analisis **Precision-Recall Curve**:
- **Baseline Acak**: Jika kita menebak anomali secara acak, nilai *Average Precision* (AP) hanya sebesar **0.134** (13.4%).
- **DBSCAN Global**: Menghasilkan AP **0.156** (hanya sedikit di atas acak).
- **DBSCAN Per-Negara**: Menghasilkan AP **0.248** (**hampir 2 kali lipat dari tebakan acak**).
Hal ini membuktikan bahwa sinyal skor anomali yang dihasilkan DBSCAN Per-Negara memiliki daya pisah diskriminatif yang valid secara statistik.

---

### 5.4 Analisis Metrik Unsupervised: Kenapa Silhouette Global Lebih Tinggi?
Hasil evaluasi metrik intrinsik klaster menunjukkan:
- **Global**: Silhouette Score = **0.777** (Sangat Baik)
- **Per-Country**: Silhouette Score = **0.273** (Sedang)

**Kenapa Begitu?**
Silhouette score mengukur seberapa padat suatu klaster dan seberapa jauh jaraknya ke klaster lain. 
DBSCAN Global menumpuk 99.2% seluruh data ke dalam satu bola klaster raksasa yang sangat padat dan membuang 13 anomali jauh di luar angkasa. Secara geometris matematika murni, bentuk satu bola padat dengan sedikit outlier jauh ini menghasilkan skor Silhouette yang sangat tinggi. Namun, bola raksasa tersebut mengubur semua variasi lokal negara, sehingga model kehilangan fungsi deteksi krisisnya.
Sebaliknya, DBSCAN Per-Negara membentuk struktur data lokal yang lebih majemuk dan dinamis, sehingga skor geometrisnya lebih moderat, tetapi fungsi analitisnya jauh lebih bermanfaat.

---

### 5.5 Daya Tangkap Berdasarkan Jenis Krisis (Mana yang Paling Terbaca?)
Apakah semua jenis krisis memiliki kemudahan yang sama untuk dideteksi oleh DBSCAN?

```
               DAYA TANGKAP DETEKSI (RECALL) PER JENIS KRISIS
  ─────────────────────────────────────────────────────────────────────────────
  Pandemi COVID-19 (2020)  │ ████████████████████████████████ 63.3% (31/49)
  Krisis Mata Uang (FX)    │ ███████████████████████ 54.3% (19/35)
  Krisis Utang (Sovereign) │ ██████████████████ 43.8% (7/16)
  Krisis Perbankan (Bank)  │ ██████████ 23.4% (37/158)
  ─────────────────────────────────────────────────────────────────────────────
```

1. **Pandemi COVID-19 (Tahun 2020) — Recall 63.3% (Paling Mudah Dideteksi)**:
   COVID-19 memicu syok simultan serentak di seluruh dunia: pertumbuhan ekonomi anjlok tajam, mobilitas perdagangan terhenti, dan belanja fiskal melonjak drastis. Pergeseran ekstrem di hampir semua 14 indikator secara serempak membuat tahun 2020 langsung terlempar keluar dari klaster normal di **31 dari 49 negara**.
2. **Krisis Mata Uang / Nilai Tukar — Recall 54.3%**:
   Ketika spekulan menyerang mata uang suatu negara, nilai tukar akan terdepresiasi puluhan persen dalam hitungan pekan, sementara cadangan devisa terkuras habis untuk intervensi. Lonjakan tajam pada indikator *fx_depreciation_pct* dan anjloknya *reserves_months_import* menciptakan sinyal densitas yang sangat kuat dan mudah ditangkap DBSCAN.
3. **Krisis Utang Pemerintah / Sovereign Debt — Recall 43.8%**:
   Negara yang mengalami gagal bayar utang (*default*) meninggalkan jejak yang jelas pada defisit anggaran yang menganga dan rasio *gov_debt_gdp* yang melesat keluar dari tren historisnya.
4. **Krisis Perbankan — Recall 23.4% (Paling Menantang)**:
   Krisis perbankan adalah jenis krisis yang paling sulit ditangkap oleh data makroekonomi. Alasannya: krisis perbankan kerap berakar dari pembusukan internal di neraca perbankan (kredit macet/NPL dan krisis likuiditas antarbank). Proses penularan dari sektor mikro perbankan hingga terasa dampaknya pada PDB agregat atau inflasi memerlukan waktu tunda (*time lag*) yang cukup lama.

---

## BAB VI: STUDI KASUS VALIDASI HISTORIS NYATA

Untuk membuktikan validitas model dalam skenario dunia nyata, mari kita telaah 4 studi kasus historis utama:

### 6.1 Krisis Keuangan Asia 1997–1998 (Indonesia & Thailand)
- **Konteks Sejarah**: Bermula dari devaluasi mata uang Baht Thailand pada Juli 1997, krisis merembet hebat ke negara-negara Asia Tenggara. Di Indonesia, nilai tukar Rupiah rontok, inflasi melonjak hingga di atas 70%, dan pertumbuhan ekonomi terkontraksi hingga minus 13%.
- **Temuan Model DBSCAN**:
  - **Indonesia (IDN)**: Tahun 1998 terdeteksi sebagai anomali baik pada model Global maupun model Per-Negara. Pada model Per-Negara, Indonesia memiliki **F1-score 0.600** dan Precision **1.000** (seluruh sinyal anomali terbukti valid krisis).
  - **Thailand (THA)**: Tahun 1997 dan 1998 terdeteksi akurat dengan **F1-score 0.667** dan Recall **60%**.
  - Model berhasil membuktikan kemampuannya menangkap fenomena *contagion effect* di kawasan regional.

### 6.2 Hiperinflasi Amerika Latin 1990–1994 (Brazil & Peru)
- **Konteks Sejarah**: Kegagalan tata kelola fiskal dan pencetakan uang besar-besaran di awal era 1990-an menyebabkan indeks harga barang di Peru dan Brazil naik berkali-kali lipat setiap bulannya.
- **Temuan Model DBSCAN**:
  - **Peru 1990**: Menempati urutan nomor 1 anomali paling ekstrem di dunia dengan skor anomali mutlak **1.000**.
  - **Brazil (1990–1994)**: Berturut-turut terdeteksi anomali selama 5 tahun berturut-turut hingga akhirnya pemerintah Brazil meluncurkan program stabilisasi moneter *Plano Real* pada akhir 1994. Begitu kondisi moneter normal, label anomali DBSCAN langsung hilang di tahun 1995.

### 6.3 Krisis Utang Yunani 2010–2014 (Greece)
- **Konteks Sejarah**: Pasca krisis 2008, terungkap bahwa pemerintah Yunani memanipulasi laporan defisit anggarannya. Akumulasi utang yang tak terkendali memicu krisis likuiditas, program *bailout* yang menyakitkan, dan resesi berkepanjangan selama lebih dari 5 tahun.
- **Temuan Model DBSCAN Per-Negara**:
  - Model Per-Negara menunjukkan performa luar biasa pada Yunani: **Recall mencapai 83.3%** (5 dari 6 tahun krisis riil berhasil diidentifikasi tepat waktu) dengan **ROC-AUC 0.937**.
  - Ini membuktikan bahwa DBSCAN Per-Negara sangat sensitif terhadap krisis jangka menengah yang bersifat struktural.

### 6.4 Guncangan Pandemi Global COVID-19 Tahun 2020
- **Konteks Sejarah**: Guncangan kesehatan global yang memaksa pembatasan fisik secara serentak di seluruh dunia pada tahun 2020.
- **Temuan Model DBSCAN**:
  - Sebanyak **31 dari 49 negara (63.3%)** ditandai sebagai anomali di tahun 2020 oleh DBSCAN Per-Negara.
  - Ini adalah konsentrasi anomali tahunan terbesar di sepanjang 35 tahun data observasi kita, membuktikan bahwa algoritma secara mandiri mampu mengenali tahun 2020 sebagai tahun paling "aneh" dalam peradaban ekonomi modern.

---

## BAB VII: KESIMPULAN, KETERBATASAN, DAN REKOMENDASI

### 7.1 Kesimpulan Utama
1. **DBSCAN adalah Algoritma yang Sangat Andal untuk Deteksi Krisis Tanpa Supervisi**:
   Tanpa memerlukan data latih berlabel (*unsupervised*), DBSCAN berhasil memanfaatkan prinsip kepadatan data (*density*) untuk memisahkan tahun-tahun ekonomi normal dari tahun-tahun krisis secara matematis dan objektif.
2. **Konteks Lokal Adalah Kunci (Per-Negara > Global)**:
   Menerapkan satu standar global untuk seluruh negara adalah kesalahan metodologis dalam analisis ekonomi makro karena mengabaikan heterogenitas struktural antar-negara. Pendekatan **DBSCAN Per-Negara** terbukti menghasilkan sensitivitas krisis 9 kali lipat lebih tinggi dibandingkan pendekatan Global.
3. **Pentingnya Metrik Khusus Data Imbalanced**:
   Dalam masalah deteksi krisis, metrik akurasi tradisional dapat menyesatkan. Evaluasi menggunakan F1-score, Precision-Recall Curve (Average Precision), dan breakdown jenis krisis memberikan gambaran nyata yang jauh lebih jujur dan mendalam.

### 7.2 Keterbatasan Penelitian
1. **Resolusi Data Tahunan**: Data yang dianalisis menggunakan agregasi tahunan (*annual data*). Hal ini membuat model hanya mampu mendeteksi krisis yang sudah berlangsung dalam hitungan bulan, bukan sebagai sistem peringatan dini hitungan hari (*real-time early warning system*).
2. **Keterbatasan Sinyal Krisis Perbankan**: Indikator makroekonomi agregat memiliki latensi terhadap kerentanan mikro perbankan, sehingga krisis perbankan murni memiliki tingkat deteksi yang lebih rendah dibandingkan krisis mata uang.

### 7.3 Rekomendasi Kebijakan & Arah Pengembangan Lanjutan
1. **Untuk Otoritas Moneter & Bank Sentral**:
   - Model DBSCAN Per-Negara dapat diadopsi sebagai lapisan pertahanan kedua (*secondary monitoring layer*) tanpa asumsi distribusi untuk memvalidasi model ekonometrika konvensional yang sering kali bias asumsi normalitas.
2. **Integrasi Data Frekuensi Tinggi**:
   - Penelitian lanjutan disarankan menggunakan data berfrekuensi bulanan atau kuartalan (*monthly/quarterly indicators*) seperti cadangan devisa mingguan, indeks volatilitas pasar modal (VIX), dan suku bunga pasar uang antarbank untuk deteksi *early warning*.
3. **Pengembangan Model Dinamis (Streaming DBSCAN)**:
   - Menerapkan varian DBSCAN bertahap (*Incremental / Streaming DBSCAN*) sehingga model dapat secara langsung memperbarui profil klaster seiring masuknya rilis data statistik triwulanan terbaru.

---
*Laporan ini disusun secara komprehensif sebagai dokumentasi resmi proyek Data Mining Unsupervised Learning.*
