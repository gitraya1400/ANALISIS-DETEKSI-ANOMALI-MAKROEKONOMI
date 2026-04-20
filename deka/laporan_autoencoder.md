# 🧠 Laporan Singkat — Deteksi Anomali Ekonomi Makro dengan Autoencoder

**Disusun oleh:** Deka  
**Tanggal:** 20 April 2026  
**Metode:** Autoencoder (Deep Learning — Unsupervised Anomaly Detection)

---

## 1. Ringkasan Eksekutif

Notebook ini mengimplementasikan **Autoencoder** sebagai salah satu dari enam algoritma deteksi anomali dalam proyek *Early Warning System* (EWS) krisis ekonomi. Autoencoder bekerja dengan prinsip **kompresi non-linear**: model dilatih untuk merekonstruksi data normal, sehingga data anomali/krisis menghasilkan *reconstruction error* yang lebih tinggi.

---

## 2. Dataset

| Aspek | Detail |
|-------|--------|
| **Sumber** | World Bank Open Data API |
| **Observasi** | 1.714 baris (49 negara × 35 tahun) |
| **Periode** | 1990–2024 |
| **Fitur** | 14 indikator makroekonomi (sudah z-score) |
| **Ground Truth** | 329 observasi krisis (19.2%) |

### 14 Indikator Makroekonomi

| No. | Fitur | Deskripsi |
|-----|-------|-----------|
| 1 | GDP_Growth | Pertumbuhan PDB riil (%) |
| 2 | GDP_PerCapita_Growth | Pertumbuhan PDB per kapita (%) |
| 3 | Inflation_CPI | Inflasi berdasarkan CPI (%) |
| 4 | Total_Reserves | Cadangan devisa total |
| 5 | Unemployment | Tingkat pengangguran (%) |
| 6 | Current_Account_GDP | Neraca transaksi berjalan/PDB (%) |
| 7 | Trade_GDP | Rasio perdagangan/PDB (%) |
| 8 | FDI_Inflows_GDP | FDI masuk/PDB (%) |
| 9 | Exports_GDP | Rasio ekspor/PDB (%) |
| 10 | Imports_GDP | Rasio impor/PDB (%) |
| 11 | Gross_Savings_GDP | Tabungan bruto/PDB (%) |
| 12 | Exchange_Rate | Nilai tukar LCU/USD |
| 13 | Manufacturing_Value | Kontribusi manufaktur/PDB (%) |
| 14 | Investment_GDP | Investasi bruto/PDB (%) |

---

## 3. Arsitektur Model

```
Input (14 fitur)
    │
┌───▼───┐
│Dense 32│ + BatchNorm + LeakyReLU(0.1) + Dropout(0.2)
└───┬───┘
    │        ENCODER
┌───▼───┐
│Dense 16│ + BatchNorm + LeakyReLU(0.1) + Dropout(0.2)
└───┬───┘
    │
╔═══▼═══╗
║Dense  6║  ← BOTTLENECK (rasio kompresi: 2.3x)
╚═══╤═══╝
    │
┌───▼───┐
│Dense 16│ + BatchNorm + LeakyReLU(0.1) + Dropout(0.2)
└───┬───┘
    │        DECODER
┌───▼───┐
│Dense 32│ + BatchNorm + LeakyReLU(0.1) + Dropout(0.2)
└───┬───┘
    │
┌───▼───┐
│Dense 14│  ← Output (rekonstruksi)
└───────┘
```

**Spesifikasi:**
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate = 0.001)
- **Regularisasi:** BatchNormalization + Dropout(0.2) + EarlyStopping(patience=20)
- **Training Split:** 80% train / 20% validation
- **Max Epochs:** 200 (dengan EarlyStopping)

---

## 4. Hasil Deteksi Anomali

### 4.1 Threshold

| Parameter | Nilai |
|-----------|-------|
| **Metode threshold** | Persentil ke-92 dari reconstruction error |
| **Nilai threshold** | 0.453560 |
| **Anomali terdeteksi** | 138 observasi (8.1%) |
| **Contamination rate** | ~8% (sesuai frekuensi historis krisis) |

### 4.2 Evaluasi vs Ground Truth

| Metrik | Nilai |
|--------|-------|
| **Precision** | 0.3261 (32.6%) |
| **Recall** | 0.1368 (13.7%) |
| **F1-Score** | 0.1927 (19.3%) |
| **AUC-ROC** | 0.6066 |

### 4.3 Confusion Matrix

|  | **Prediksi: Normal** | **Prediksi: Anomali** |
|--|---------------------|-----------------------|
| **Aktual: Normal** | TN = 1.292 | FP = 93 |
| **Aktual: Krisis** | FN = 284 | TP = 45 |

---

## 5. Deteksi per Krisis Historis

| Krisis | Tahun | Observasi | Terdeteksi | Detection Rate |
|--------|-------|-----------|------------|----------------|
| **Krisis Asia** | 1997–1998 | 10 | 5 | **50.0%** |
| **Krisis Rusia** | 1998 | 1 | 0 | 0.0% |
| **Krisis Argentina** | 2001–2002 | 2 | 1 | **50.0%** |
| **Global Financial Crisis** | 2008–2009 | 98 | 7 | 7.1% |
| **Krisis Utang Eropa** | 2010–2012 | 15 | 1 | 6.7% |
| **Pandemi COVID-19** | 2020 | 49 | 9 | **18.4%** |

### Interpretasi:
- **Krisis Asia** mendapat detection rate tertinggi (50%), menunjukkan Autoencoder cukup sensitif terhadap guncangan regional yang ekstrem pada indikator makro.
- **GFC dan Krisis Eropa** sulit dideteksi karena efeknya tersebar secara global namun dengan intensitas yang beragam antar negara.
- **COVID-19** terdeteksi pada 18.4% negara — krisis ini bersifat eksogen (non-ekonomi) sehingga polanya berbeda dari krisis finansial murni.

---

## 6. Top 10 Negara dengan Anomali Terbanyak

| No. | Negara | Kode | Jumlah Anomali |
|-----|--------|------|----------------|
| 1 | Irlandia | IRL | 18 |
| 2 | Singapura | SGP | 16 |
| 3 | Belanda | NLD | 11 |
| 4 | Hongaria | HUN | 10 |
| 5 | Indonesia | IDN | 9 |
| 6 | Rumania | ROU | 6 |
| 7 | Tiongkok | CHN | 6 |
| 8 | Thailand | THA | 6 |
| 9 | Rusia | RUS | 6 |
| 10 | Swiss | CHE | 6 |

> **Catatan:** Negara-negara dengan ekonomi terbuka (trade/GDP tinggi seperti IRL, SGP, NLD) cenderung memiliki profil indikator yang lebih volatile, sehingga Autoencoder menandainya lebih sering sebagai anomali.

---

## 7. Analisis & Interpretasi

### Kekuatan Autoencoder:
1. **Kompresi non-linear** — mampu menangkap interaksi kompleks antar 14 indikator yang tidak bisa ditangkap oleh metode linear.
2. **Sensitivitas terhadap krisis regional** — detection rate 50% pada Krisis Asia menunjukkan kemampuan mendeteksi guncangan mendadak yang terkonsentrasi.
3. **Tidak memerlukan label** — sepenuhnya *unsupervised*, sesuai dengan sifat deteksi anomali.

### Keterbatasan:
1. **Recall rendah (13.7%)** — banyak krisis tidak terdeteksi, terutama krisis global yang efeknya tersebar merata.
2. **Precision moderat (32.6%)** — sekitar 2/3 anomali terdeteksi bukan merupakan krisis menurut ground truth → *false positive* cukup tinggi.
3. **Bias terhadap ekonomi terbuka** — negara dengan rasio Trade/GDP dan FDI tinggi lebih sering ditandai sebagai anomali.

### Rekomendasi:
- Autoencoder **tidak cukup kuat berdiri sendiri** sebagai satu-satunya EWS.
- Perlu dikombinasikan dengan algoritma lain (Isolation Forest, LOF, OCSVM, PCA, DBSCAN) melalui **Majority Voting** untuk meningkatkan reliabilitas deteksi.
- Threshold dapat di-tuning lebih lanjut (misalnya persentil 88–95) untuk mencari keseimbangan optimal Precision-Recall.

---

## 8. File Output

| File | Deskripsi |
|------|-----------|
| `encoder.ipynb` | Notebook lengkap dengan kode dan visualisasi |
| `encoder.html` | Versi HTML dari notebook (siap cetak) |
| `hasil_autoencoder.csv` | Dataset hasil deteksi (1.714 baris, 19 kolom) |
| `laporan_autoencoder.md` | Laporan singkat ini |

### Kolom pada `hasil_autoencoder.csv`:
- `economy`, `year` — Metadata negara dan tahun
- 14 kolom fitur indikator (z-score)
- `crisis_label` — Ground truth (0=Normal, 1=Krisis)
- `reconstruction_error` — Nilai reconstruction error MSE
- `anomaly_ae` — Prediksi Autoencoder (0=Normal, 1=Anomali)

---

*Laporan ini merupakan bagian dari proyek **Deteksi Anomali pada Indikator Ekonomi Makro sebagai Early Warning System Krisis Ekonomi menggunakan Pendekatan Unsupervised Learning** — UTS Data Mining 2026.*
