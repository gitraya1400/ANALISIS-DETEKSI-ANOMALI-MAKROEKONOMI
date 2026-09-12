# Analisis Hasil — Autoencoder (Deka)

**Kelompok 4 — 3SI1 | Early Warning System Krisis Ekonomi**

Dokumen ini menganalisis hasil menjalankan `encoder_revisi.ipynb` di Google Colab. Notebook berjalan tanpa satu pun error, dan ketiga file keluaran terbentuk sesuai format standar kelompok.

---

## 1. Ringkasan Angka

| Metrik | Seluruh Data | Data Uji Saja | Tanpa 2020 |
|--------|--------------|---------------|------------|
| Precision | 0,368 | 0,796 | 0,308 |
| Recall | 0,188 | 0,188 | 0,183 |
| F1-Score | 0,249 | 0,304 | 0,230 |
| AUC-ROC | **0,720** | 0,725 | 0,686 |
| AUC-PR | 0,292 | 0,708 | 0,225 |

Konfigurasi: 1.715 observasi, 14 fitur, threshold 0,9933 dari persentil ke-95 error data latih, 117 observasi ditandai anomali, seed 42.

Training berhenti di epoch 119 lewat early stopping, dengan bobot terbaik dipulihkan dari epoch 99. Train loss 0,521 dan validation loss 0,341.

**Catatan penting soal loss.** Validation loss lebih rendah daripada train loss. Ini normal dan bukan tanda kesalahan, karena Dropout dan BatchNormalization aktif saat training tetapi nonaktif saat evaluasi.

---

## 2. Perbaikan Terbukti Berhasil

### 2.1 Bias struktural hilang sepenuhnya

Ini hasil paling meyakinkan dari seluruh revisi.

| Negara | Versi lama | Versi revisi |
|--------|-----------|--------------|
| Singapura | 35 dari 35 tahun ditandai anomali | **0 dari 35 tahun** |

Model tidak lagi mengacaukan struktur ekonomi terbuka dengan kondisi krisis. Ini membuktikan *country demeaning* bekerja persis seperti yang diharapkan.

### 2.2 Krisis Rusia 1998 kini terdeteksi

Pada versi lama tingkat deteksinya 0 persen. Sekarang 100 persen, dan alasannya terlihat jelas di data.

| Tahun | Depresiasi (%) | GDP Growth | Inflasi | Skor | Ditandai |
|-------|----------------|------------|---------|------|----------|
| 1997 | 12,97 | 1,4 | 14,76 | 0,51 | Tidak |
| 1998 | **67,77** | -5,3 | 27,69 | 1,02 | Ya |
| 1999 | **153,68** | 6,4 | 85,75 | 1,46 | Ya |

Versi lama memakai nilai tukar nominal sehingga lonjakan ini tidak terbaca. Setelah diubah menjadi persentase depresiasi, krisis rubel langsung tertangkap. Ini bukti langsung bahwa perbaikan nomor satu memang menyelesaikan masalah nyata.

### 2.3 Fitur terpenting masuk akal secara ekonomi

| Peringkat | Fitur | Selisih error |
|-----------|-------|---------------|
| 1 | Exchange_Depreciation | 6,005 |
| 2 | GDP_Growth | 1,732 |
| 3 | Broad_Money_Growth | 1,345 |
| 4 | FDI_Inflows_GDP | 0,868 |
| 5 | Domestic_Credit_GDP | 0,640 |

Depresiasi nilai tukar menjadi penanda krisis terkuat, jauh di atas fitur lain. Ini sejalan dengan literatur krisis mata uang, dan sekaligus menegaskan bahwa fitur yang dulu salah bentuk justru yang paling informatif setelah diperbaiki.

Perlu dicatat, fitur ini juga yang paling rentan. Nilainya sangat besar pada kasus hiperinflasi, sehingga sebagian keunggulannya berasal dari skala yang lebar, bukan semata daya pisahnya.

---

## 3. Pembacaan Metrik yang Perlu Kehati-hatian

### 3.1 Precision 0,796 pada data uji tidak boleh dipakai sebagai klaim utama

Angka ini terlihat mengesankan, tetapi menyesatkan jika dikutip tanpa konteks. Penyebabnya adalah komposisi data uji, bukan kemampuan model.

Data uji berisi 229 observasi krisis dan hanya 223 observasi normal, sehingga proporsi krisisnya 50,7 persen. Bandingkan dengan data penuh yang hanya 13,35 persen. Dengan basis sebesar itu, menebak krisis jadi jauh lebih mudah.

Hal yang sama berlaku untuk AUC-PR 0,708 pada data uji, karena baseline-nya 0,507 sementara pada data penuh hanya 0,134.

**Yang sebaiknya dilaporkan sebagai angka utama adalah hasil pada seluruh data**, yaitu precision 0,368 dan AUC-ROC 0,720. Metrik data uji tetap dicantumkan, tetapi dijelaskan bahwa basis kelasnya berbeda.

Satu angka yang tetap sah dibandingkan adalah AUC-ROC, karena relatif tidak terpengaruh proporsi kelas. Nilainya 0,720 dan 0,725, hampir sama, yang menunjukkan model tidak mengalami overfitting.

### 3.2 Recall 0,188 memang rendah, dan itu konsekuensi threshold

Model hanya menangkap 43 dari 229 observasi krisis dan melewatkan 186. Ini bukan kegagalan acak, melainkan akibat threshold ketat pada persentil ke-95.

Uji berbagai threshold menunjukkan pertukaran yang jelas:

| Threshold | Ditandai | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| P95 (dipakai) | 86 | 0,349 | 0,131 | 0,190 |
| P90 | 172 | 0,343 | 0,258 | 0,294 |
| P86,65 | 229 | 0,349 | 0,349 | 0,349 |
| **P84** | 275 | 0,335 | 0,402 | **0,365** |

F1 terbaik berada di sekitar persentil ke-84, hampir dua kali lipat dibanding sekarang. Menariknya, precision hampir tidak berubah di seluruh rentang, sehingga melonggarkan threshold menaikkan recall nyaris tanpa biaya.

**Namun angka ini tidak boleh langsung dipakai.** Pencarian tadi dilakukan di atas seluruh data berlabel, sehingga hasilnya terlalu optimis dan termasuk kebocoran data. Threshold sekarang justru dipilih tanpa melihat label, dan itulah yang membuatnya jujur.

Jika ingin memakai threshold lebih longgar, cara yang sah adalah menurunkan persentil pada data latih normal, misalnya ke persentil ke-90, lalu melaporkannya sebagai keputusan desain. Bukan dengan memilih angka yang memaksimalkan F1 pada data uji.

---

## 4. Keterbatasan yang Harus Masuk Laporan

### 4.1 Model masih bersifat coincident

| Horizon | AUC-ROC |
|---------|---------|
| Konkuren, t ke t | 0,720 |
| Lead 1 tahun | 0,625 |
| Lead 2 tahun | 0,596 |

Kemampuan prediksi menurun jelas seiring bertambahnya horizon. Model lebih tepat disebut pendeteksi tekanan yang sedang berlangsung daripada sistem peringatan dini penuh.

**Tetapi ada sinyal awal yang nyata dan layak dilaporkan.** Rata-rata skor satu tahun sebelum krisis dimulai lebih tinggi daripada tahun tenang.

| Kondisi | Rata-rata skor |
|---------|----------------|
| Tahun tenang | 0,344 |
| Satu tahun sebelum krisis | 0,488 |
| Saat krisis berlangsung | 1,274 |

Selisih antara tahun tenang dan tahun sebelum krisis diuji dengan Mann-Whitney U dan **signifikan secara statistik dengan p sebesar 0,0046**. Artinya tekanan memang mulai terbaca sebelum krisis, hanya saja belum cukup kuat untuk melewati threshold.

Ini temuan yang bagus untuk Bab IV. Model punya daya peringatan dini yang terukur, walau lemah.

### 4.2 Saudi Arabia menyumbang seperlima false positive

Dari 74 false positive, 16 berasal dari Saudi Arabia. Negara ini ditandai anomali 17 dari 35 tahun padahal hanya punya satu label krisis.

Penyebabnya terlihat dari volatilitas fiskalnya:

| Fitur | Simpangan baku Saudi | Median negara lain | Rasio |
|-------|---------------------|--------------------|-------|
| Reserves_Months_Imports | 12,80 | 1,42 | **9,0x** |
| Current_Account_GDP | 12,68 | 2,73 | 4,6x |
| Gross_Savings_GDP | 11,55 | 2,73 | 4,2x |

Tahun yang ditandai adalah 1990 sampai 1993, 2003 sampai 2005, 2015 sampai 2017, dan 2020. Pola ini mengikuti Perang Teluk dan siklus harga minyak, bukan krisis perbankan.

**Interpretasinya penting.** Model mendeteksi guncangan makroekonomi nyata pada ekonomi berbasis komoditas, tetapi guncangan itu tidak tercatat dalam katalog Laeven dan Valencia yang berfokus pada krisis perbankan. Ini keterbatasan ground truth, bukan sepenuhnya kesalahan model. Sebaiknya ditulis demikian di laporan.

### 4.3 Global Financial Crisis justru terdeteksi rendah

| Episode | N | Terdeteksi | Tingkat deteksi |
|---------|---|-----------|-----------------|
| Krisis Rusia 1998 | 1 | 1 | 100,0% |
| Krisis Asia 1997-1998 | 10 | 5 | 50,0% |
| Krisis Argentina 2001-2002 | 2 | 1 | 50,0% |
| Krisis Utang Eropa | 15 | 4 | 26,7% |
| COVID-19 | 49 | 10 | 20,4% |
| **Global Financial Crisis** | 98 | 7 | **7,1%** |

GFC adalah episode dengan deteksi terburuk, padahal krisis paling terkenal. Alasannya masuk akal secara metodologis. Karena *country demeaning* membandingkan tiap negara dengan rata-ratanya sendiri, guncangan yang menimpa hampir semua negara secara bersamaan justru sebagian terserap ke dalam rata-rata negara itu.

Dengan kata lain, transformasi yang memperbaiki bias Singapura punya efek samping melemahkan deteksi krisis global serentak. Ini pertukaran yang perlu disebut terbuka di bagian keterbatasan.

Perhatikan juga bahwa deteksi 100 persen pada Rusia hanya berbasis satu observasi, sehingga tidak dapat digeneralisasi.

### 4.4 Ketergantungan pada label COVID

AUC turun dari 0,720 menjadi 0,686 ketika tahun 2020 dikeluarkan, selisih 0,034. Penurunannya ada tetapi tidak besar, sehingga performa model tidak semata bertumpu pada COVID. Ini kabar baik untuk validitas hasil.

---

## 5. Penilaian Keseluruhan

AUC-ROC 0,720 tergolong sedang. Jauh di atas tebakan acak 0,50, tetapi belum termasuk kuat. Untuk konteks UTS dengan enam algoritma yang dibandingkan, angka ini wajar dan dapat dipertanggungjawabkan karena diperoleh tanpa kebocoran data.

Yang lebih penting daripada angkanya adalah bahwa hasil ini **jujur**. Threshold dikalibrasi tanpa melihat label, data uji tidak pernah dilihat model saat latih, dan ground truth berasal dari sumber eksternal. Bandingkan dengan versi lama yang AUC-nya tampak lebih baik tetapi dihitung dari label buatan sendiri.

Perbandingan menyeluruh:

| Aspek | Versi lama | Versi revisi |
|-------|-----------|--------------|
| AUC-ROC | 0,640 | 0,720 |
| Singapura ditandai | 35 dari 35 tahun | 0 dari 35 tahun |
| Rusia 1998 | Tidak terdeteksi | Terdeteksi |
| Sumber label | Buatan sendiri, tautologis | IMF eksternal |
| Kalibrasi threshold | Seluruh data | Data latih saja |

---

## 6. Saran Tindak Lanjut

**Untuk laporan, tiga hal berikut sebaiknya ditulis.** Pertama, gunakan metrik seluruh data sebagai angka utama dan jelaskan mengapa precision data uji terlihat tinggi. Kedua, laporkan sinyal pra-krisis yang signifikan dengan p sebesar 0,0046 sebagai temuan pendukung. Ketiga, sebut deteksi GFC yang rendah beserta penjelasan efek *demeaning* agar tidak terlihat sebagai kelemahan yang tidak dipahami.

**Untuk perbandingan antar model**, pastikan lima algoritma lain memakai `data_features_autoencoder.csv` dan skema threshold yang setara. Membandingkan model dengan threshold persentil berbeda akan menghasilkan kesimpulan yang keliru.

**Jika masih ada waktu**, satu eksperimen bernilai tinggi adalah menambahkan fitur selisih tahunan untuk seluruh variabel. Pada uji awal sebelumnya, skema dengan fitur selisih memberi AUC konkuren 0,666 dibanding 0,668 pada skema terpilih, tetapi lebih unggul saat tahun 2020 dikeluarkan. Kombinasi demeaning dan fitur selisih berpeluang memperbaiki deteksi krisis serentak seperti GFC.
