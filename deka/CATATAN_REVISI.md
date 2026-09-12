# Catatan Revisi — Autoencoder (Deka)

**Kelompok 4 — 3SI1 | Analisis Deteksi Anomali pada Indikator Makroekonomi Global**

Dokumen ini merangkum apa yang diperbaiki, mengapa, dan bukti empirisnya. Ditujukan sebagai lampiran teknis untuk revisi Bab III dan bahan diskusi kelompok.

---

## 1. Status File

| File | Keterangan |
|------|------------|
| `encoder_revisi.ipynb` | Notebook autoencoder versi revisi, siap jalan di Google Colab |
| `encoder.ipynb` | Versi lama, **jangan dipakai lagi** (disimpan sebagai arsip) |
| `raw_data_master.csv` | Input data mentah, 1.715 baris |
| `ground_truth_imf.csv` | Label krisis eksternal IMF, 1.715 baris |

Notebook revisi menghasilkan tiga output:

- `hasil_autoencoder.csv` dengan kolom `economy`, `year`, `anomaly_score`, `predicted_anomaly`
- `ringkasan_model_autoencoder.csv` berisi metrik untuk agregasi lintas model
- `data_features_autoencoder.csv` berisi fitur hasil rekayasa, bisa dipakai anggota lain

---

## 2. Delapan Perbaikan yang Diterapkan

### 2.1 Nilai tukar nominal menjadi persentase depresiasi

Versi lama memasukkan `Exchange_Rate` dalam satuan mata uang lokal per USD. Nilai ini tidak sebanding antar negara. Pada data master, rasio antara nilai tukar tertinggi dan terendah mencapai **25.127 kali lipat**.

| Negara | Median nominal | Kondisi sebenarnya |
|--------|----------------|--------------------|
| Inggris | 0,64 per USD | Normal |
| Irlandia | 0,80 per USD | Normal |
| Indonesia | 9.387 per USD | Normal |
| Vietnam | 16.105 per USD | Normal dan stabil |

Setelah standardisasi global, Vietnam memperoleh z-score ekstrem dan dibaca model sebagai krisis permanen. Perbaikannya mengikuti praktik baku pada literatur krisis mata uang, yaitu memakai laju depresiasi tahunan.

```python
df['Exchange_Depreciation'] = df.groupby('economy')['Exchange_Rate'].pct_change() * 100
df = df.drop(columns=['Exchange_Rate'])
```

### 2.2 Cadangan devisa sudah berbentuk rasio

Masalah ini sudah teratasi di file master. Kolom `Total_Reserves` yang dulu bernilai nominal USD kini diganti `Reserves_Months_Imports` dengan rentang 0,03 sampai 37,37 bulan. Tidak ada lagi perbandingan antara cadangan Tiongkok senilai triliunan dolar dengan Kenya senilai puluhan juta dolar.

### 2.3 Ground truth eksternal, bukan buatan sendiri

Versi lama membentuk `crisis_label` melalui aturan if-else atas empat variabel yang juga menjadi input model. Akibatnya evaluasi bersifat tautologis: model dinilai berdasarkan aturan yang dibuat dari variabel yang sama.

Versi revisi memakai `ground_truth_imf.csv` yang bersumber dari Laeven dan Valencia (2018). Label tidak diturunkan dari variabel input, sehingga evaluasi menjadi sah.

Proporsi krisis berubah dari 19,2 persen menjadi **13,35 persen**, yaitu 229 dari 1.715 observasi.

### 2.4 Country demeaning untuk menghapus bias struktural

Ini perbaikan dengan dampak terbesar. Standardisasi global membandingkan struktur ekonomi antar negara, bukan kondisi krisisnya.

Rata-rata nilai mutlak z-score tertinggi di bawah penskalaan global:

| Negara | Rata-rata absolut z |
|--------|---------------------|
| Singapura | 1,53 |
| Tiongkok | 1,05 |
| Irlandia | 0,94 |

Singapura menempati posisi teratas bukan karena krisis, melainkan karena rasio perdagangan terhadap PDB di atas 300 persen. Pada versi lama, Singapura dicap anomali selama 35 dari 35 tahun.

Solusinya adalah mengurangi setiap nilai dengan rata-rata negaranya sendiri sebelum standardisasi global. Dengan begitu tiap negara dinilai relatif terhadap kondisi normalnya sendiri, sementara skala antar fitur tetap sebanding.

**Hasil setelah perbaikan: Singapura terdeteksi anomali 0 persen dari 35 tahun.**

Empat skema penskalaan diuji secara empiris memakai PCA sebagai proksi:

| Skema | AUC konkuren | AUC lead 1 tahun |
|-------|--------------|------------------|
| Global murni | 0,6401 | 0,5573 |
| Robust scaler | 0,6135 | 0,5205 |
| Z-score per negara | 0,6341 | 0,5363 |
| **Demeaning lalu global** | **0,6676** | **0,5872** |

Skema demeaning unggul pada kedua horizon, sehingga dipilih.

### 2.5 Evaluasi early warning yang jujur

Judul penelitian menyebut *Early Warning System*, tetapi versi lama menilai data tahun t terhadap krisis tahun t. Itu deteksi bersamaan, bukan peringatan dini.

Notebook revisi melaporkan tiga horizon sekaligus: konkuren, lead satu tahun, dan lead dua tahun. Pada uji coba, AUC turun dari 0,71 pada horizon konkuren menjadi 0,60 pada lead satu tahun dan 0,57 pada lead dua tahun.

Penurunan ini **dilaporkan apa adanya sebagai keterbatasan**, bukan disembunyikan. Ini justru memperkuat kredibilitas laporan.

### 2.6 Istilah metodologi yang konsisten

Autoencoder dilatih hanya pada data non-krisis, sehingga memakai informasi label. Istilah yang tepat adalah **semi-supervised novelty detection**, bukan unsupervised murni. Notebook menyatakan ini secara eksplisit di bagian pembuka.

### 2.7 Threshold bebas kebocoran data

Versi lama menghitung persentil ke-92 dari reconstruction error seluruh data, termasuk data uji. Ini membocorkan informasi.

Versi revisi mengalibrasi threshold pada persentil ke-95 error **data latih normal saja**. Terdapat pemeriksaan otomatis yang memastikan tidak ada observasi krisis yang masuk ke data latih.

### 2.8 Imputasi per negara

Interpolasi linear kini dilakukan per negara dengan urutan waktu yang benar, sehingga tren masing-masing negara terjaga dan tidak tercampur antar negara.

---

## 3. Temuan Baru pada Ground Truth

Pemeriksaan `ground_truth_imf.csv` menemukan satu inkonsistensi yang perlu diputuskan bersama.

**Tahun 2020 melabeli seluruh 49 negara sebagai krisis, tetapi ketiga komponen krisis bernilai nol.**

| Aspek | Nilai |
|-------|-------|
| Baris tahun 2020 | 49 |
| Dilabeli krisis | 49 |
| `banking_crisis` | 0 |
| `currency_crisis` | 0 |
| `sovereign_debt_crisis` | 0 |
| Sumber label | "COVID-19 Global Shock" |

Label 2020 berasal dari kategori di luar taksonomi Laeven dan Valencia. Satu tahun ini menyumbang 49 dari 229 observasi krisis, atau sekitar 21 persen dari seluruh label positif.

Dampaknya terukur. Pada uji coba, AUC turun dari 0,707 menjadi 0,672 ketika tahun 2020 dikeluarkan. Artinya sebagian performa model memang bertumpu pada COVID.

**Rekomendasi:** pertahankan label 2020 pada analisis utama, tetapi selalu laporkan uji sensitivitas tanpa 2020. Notebook sudah menyediakan keduanya secara otomatis. Ini sejalan dengan rencana menempatkan COVID sebagai bukti keunggulan pendekatan tanpa definisi krisis apriori.

Perlu juga diselaraskan dengan komponen krisis. Jika COVID dianggap guncangan non-finansial, sebaiknya ditambahkan kolom penanda tersendiri agar tidak tampak seperti kesalahan pengisian data.

---

## 4. Hasil Uji Coba Pipeline

Logika notebook telah dijalankan penuh memakai PCA sebagai pengganti autoencoder, karena TensorFlow tidak tersedia di lingkungan lokal. Seluruh pemeriksaan otomatis lolos.

Angka di bawah ini adalah hasil proksi PCA. **Angka final akan berbeda** setelah notebook dijalankan dengan autoencoder sungguhan di Colab.

| Cakupan evaluasi | Precision | Recall | F1 | AUC-ROC |
|------------------|-----------|--------|-----|---------|
| Seluruh data | 0,400 | 0,210 | 0,275 | 0,707 |
| Data uji saja | 0,800 | 0,210 | 0,332 | 0,717 |
| Tanpa tahun 2020 | 0,301 | 0,172 | 0,219 | 0,672 |

Perbandingan terhadap versi lama:

| Indikator | Versi lama | Versi revisi |
|-----------|-----------|--------------|
| AUC-ROC | 0,640 | 0,707 |
| Singapura dicap anomali | 35 dari 35 tahun | 0 dari 35 tahun |
| Sumber label | Buatan sendiri, tautologis | IMF eksternal |
| Threshold | Dari seluruh data | Dari data latih saja |

---

## 5. Keterbatasan yang Harus Ditulis di Laporan

Tiga hal ini sebaiknya masuk bagian keterbatasan agar laporan tidak terlihat mengklaim berlebihan.

**Pertama, model bersifat coincident.** AUC pada horizon lead satu tahun turun mendekati 0,60. Model lebih tepat disebut pendeteksi tekanan makroekonomi yang sedang berlangsung daripada sistem peringatan dini penuh.

**Kedua, krisis Rusia 1998 tetap tidak terdeteksi.** Tingkat deteksi 0 persen. Penyebabnya adalah jumlah observasi yang sangat sedikit, hanya satu baris negara-tahun, sehingga sulit dinilai secara statistik.

**Ketiga, kualitas data beberapa negara terbatas.** Untuk Uni Emirat Arab, sejumlah variabel nyaris konstan sepanjang periode akibat imputasi berat. Khusus nilai tukar, kestabilan ini justru benar secara ekonomi karena dirham dipatok terhadap dolar sejak 1997. Hal ini perlu disebut agar pembaca tidak salah menduga adanya kesalahan data.

Variabel `Broad_Money_Growth` memiliki missing value 25 persen dan `Domestic_Credit_GDP` sebesar 18,6 persen sebelum imputasi. Proporsi ini cukup besar dan layak disebutkan.

---

## 6. Yang Perlu Dikoordinasikan dengan Anggota Lain

Agar perbandingan enam algoritma menjadi adil, beberapa hal harus seragam.

**Skema data harus identik.** Semua anggota memakai `raw_data_master.csv`, transformasi `Exchange_Depreciation`, imputasi per negara, dan country demeaning yang sama. Notebook ini menyimpan `data_features_autoencoder.csv` agar bisa dipakai langsung tanpa mengulang rekayasa fitur.

**Perbedaan paradigma harus dinyatakan.** Autoencoder, One-Class SVM, dan PCA dilatih pada data normal saja, sehingga tergolong semi-supervised. Isolation Forest dan LOF memakai label hanya untuk penyetelan hyperparameter. DBSCAN paling mendekati unsupervised murni. Perbedaan ini wajib ditulis di Bab III agar pembaca tahu bahwa ketiga kelompok model tidak berada pada kondisi yang setara.

**DBSCAN perlu dua perbaikan.** Unit analisis harus disebut observasi negara-tahun, bukan negara, karena satu baris adalah satu pasangan negara dan tahun. Selain itu, titik bernilai minus satu harus dikeluarkan sebelum menghitung Silhouette Index dan Davies-Bouldin Index, karena jika tidak, seluruh noise akan diperlakukan sebagai satu klaster tersendiri.

**Angka di laporan perlu disamakan.** Jumlah observasi adalah 1.715, jumlah fitur 14, dan neuron input autoencoder 14. Bab II yang menyebut 16 neuron perlu dikoreksi.

---

## 7. Langkah Menjalankan di Colab

1. Unggah `encoder_revisi.ipynb`, `raw_data_master.csv`, dan `ground_truth_imf.csv` ke Colab.
2. Jalankan seluruh sel secara berurutan. TensorFlow sudah tersedia secara bawaan.
3. Unduh tiga file keluaran dan letakkan di folder `deka/`.
4. Perbarui angka pada laporan memakai isi `ringkasan_model_autoencoder.csv`.

Seed dikunci pada nilai 42 sesuai standar kelompok, sehingga hasil dapat direproduksi.
