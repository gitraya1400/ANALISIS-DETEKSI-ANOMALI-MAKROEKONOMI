const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, AlignmentType, ImageRun, ShadingType, BorderStyle, PageBreak
} = require("docx");

const FIG_DIR = path.join(__dirname, "..", "figures");
const RESULTS_DIR = path.join(__dirname, "..", "results");

function img(filename, widthPx, heightPx) {
  return new ImageRun({
    type: "png",
    data: fs.readFileSync(path.join(FIG_DIR, filename)),
    transformation: { width: widthPx, height: heightPx },
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
    alignment: AlignmentType.JUSTIFIED,
  });
}
function pRich(runs, opts = {}) {
  return new Paragraph({ children: runs, spacing: { after: 160 }, alignment: AlignmentType.JUSTIFIED, ...opts });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 90 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 19 })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 260 },
  });
}
function figureCentered(filename, widthPx, heightPx, captionText) {
  return [
    new Paragraph({ children: [img(filename, widthPx, heightPx)], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 } }),
    caption(captionText),
  ];
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, color: "auto", fill: "16213E" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
    verticalAlign: "center",
  });
}

function simpleTable(headerRow, rows, colWidths) {
  const headerCells = headerRow.map((t, i) => cell(t, { header: true, width: colWidths[i] }));
  const bodyRows = rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, { width: colWidths[i] })) }));
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [new TableRow({ children: headerCells, tableHeader: true }), ...bodyRows],
  });
}

// ============ Baca hasil numerik dari CSV ============
function readCsv(file) {
  const raw = fs.readFileSync(path.join(RESULTS_DIR, file), "utf-8").trim().split("\n");
  const headers = raw[0].split(",");
  const values = raw[1].split(",");
  const obj = {};
  headers.forEach((h, i) => obj[h] = values[i]);
  return obj;
}
const metrics = readCsv("metrics_summary.csv");
const fmt = (x) => Number(x).toFixed(4);
const fmtPct = (x) => (Number(x) * 100).toFixed(1) + "%";

const effectRaw = fs.readFileSync(path.join(RESULTS_DIR, "effect_sizes.csv"), "utf-8").trim().split("\n").slice(1);
const effectRows = effectRaw.map(line => {
  const [feat, val] = line.split(",");
  return [feat, Number(val).toFixed(3), Number(val) > 0 ? "Lebih tinggi di kelompok anomali" : "Lebih rendah di kelompok anomali"];
}).sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])));

const doc = new Document({
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: [
      // ===== COVER =====
      new Paragraph({ text: "", spacing: { before: 800 } }),
      new Paragraph({
        children: [new TextRun({ text: "LAPORAN INDIVIDU", bold: true, size: 32 })],
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "PCA-based Anomaly Detection", bold: true, size: 40 })],
        alignment: AlignmentType.CENTER, spacing: { after: 100 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "sebagai Instrumen Early Warning System Krisis Ekonomi", bold: true, size: 26 })],
        alignment: AlignmentType.CENTER, spacing: { after: 400 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Bagian dari Proyek Kelompok:", italics: true, size: 22 })],
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({
          text: "Analisis Deteksi Anomali pada Indikator Makroekonomi Global sebagai Instrumen Early Warning System Krisis Ekonomi (3SI1 Kelompok 4)",
          italics: true, size: 22,
        })],
        alignment: AlignmentType.CENTER, spacing: { after: 600 },
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ===== 1. PENDAHULUAN =====
      h1("1. Pendahuluan"),
      p("Dokumen ini merupakan laporan individu untuk bagian PCA-based Anomaly Detection, " +
        "sesuai pembagian tugas proposal kelompok Section 2.1.3.5 (landasan teori) dan 3.2.6 (implementasi). " +
        "Metode ini merupakan salah satu dari enam algoritma unsupervised anomaly detection yang dibandingkan " +
        "dalam proyek kelompok untuk membangun Early Warning System (EWS) krisis ekonomi."),
      p("Prinsip dasar metode ini: Principal Component Analysis (PCA) dilatih hanya menggunakan data " +
        "kondisi ekonomi normal (crisis_label = 0), kemudian dipakai untuk mengukur seberapa jauh suatu " +
        "observasi baru menyimpang dari pola normal tersebut. Dua ukuran penyimpangan yang digunakan adalah " +
        "Squared Prediction Error (SPE) dan Hotelling's T\u00B2."),

      // ===== 2. METODOLOGI =====
      h1("2. Metodologi"),
      h2("2.1 Data"),
      p("Dataset bersumber dari World Bank Global Economic Monitor, mencakup 49 negara periode 1990\u20132024, " +
        "dengan total 1.714 observasi setelah preprocessing. Terdapat 16 kolom: economy dan year sebagai " +
        "identitas, 14 kolom indikator makroekonomi numerik, dan crisis_label sebagai ground truth."),
      h2("2.2 Preprocessing"),
      bullet("Fitur yang digunakan: 14 variabel numerik (GDP Growth, GDP Per Capita Growth, Inflation CPI, Total Reserves, Unemployment, Current Account to GDP, Trade to GDP, FDI Inflows to GDP, Exports to GDP, Imports to GDP, Gross Savings to GDP, Exchange Rate, Manufacturing Value Added, Investment to GDP)."),
      bullet("Pembagian data: 70% train (HANYA observasi normal), 15% validasi, 15% test (validasi & test berisi campuran normal dan anomali)."),
      bullet("Standardisasi Z-score menggunakan StandardScaler, di-fit hanya pada data train untuk menghindari data leakage."),
      h2("2.3 Penentuan Jumlah Komponen PCA"),
      p("Jumlah komponen utama (k) ditentukan dari titik di mana cumulative explained variance mencapai " +
        "minimal 95%. Hasil analisis pada data training menunjukkan bahwa " + metrics.k_components +
        " komponen sudah cukup untuk mencapai kriteria tersebut."),
      ...figureCentered("01_scree_plot.png", 500, 260, "Gambar 1. Scree Plot \u2014 Pemilihan Jumlah Komponen PCA"),
      h2("2.4 Statistik Deteksi Anomali"),
      p("SPE (Squared Prediction Error) mengukur seberapa besar residual rekonstruksi suatu observasi " +
        "terhadap ruang komponen utama, dengan Upper Control Limit (UCL) ditentukan melalui pendekatan " +
        "distribusi chi-square (Jackson & Mudholkar, 1979). Hotelling's T\u00B2 mengukur jarak suatu observasi " +
        "dari pusat data di dalam ruang komponen utama, dengan UCL ditentukan melalui distribusi F. " +
        "Suatu observasi diklasifikasikan sebagai anomali apabila SPE melebihi UCL-nya ATAU T\u00B2 melebihi UCL-nya."),
      simpleTable(
        ["Statistik", "Nilai UCL (\u03B1 = 0,05)"],
        [
          ["SPE (Q-statistic)", Number(metrics.ucl_spe).toFixed(4)],
          ["Hotelling's T\u00B2", Number(metrics.ucl_t2).toFixed(4)],
        ],
        [5200, 3600]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),

      // ===== 3. HASIL =====
      h1("3. Hasil"),
      h2("3.1 Metrik Evaluasi (Test Set)"),
      p("Evaluasi dilakukan dengan membandingkan hasil klasifikasi anomali terhadap crisis_label " +
        "aktual pada data test (373 observasi, di luar data training)."),
      simpleTable(
        ["Metrik", "Nilai"],
        [
          ["AUC-ROC", fmt(metrics["AUC-ROC"])],
          ["F1-Score", fmt(metrics["F1-Score"])],
          ["Precision", fmt(metrics["Precision"])],
          ["Recall", fmt(metrics["Recall"])],
          ["Average Precision", fmt(metrics["Average Precision"])],
        ],
        [5200, 3600]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      h2("3.2 Confusion Matrix"),
      simpleTable(
        ["Kategori", "Jumlah Observasi", "Keterangan"],
        [
          ["True Positive", metrics.true_positive, "Anomali & memang krisis"],
          ["False Positive", metrics.false_positive, "Anomali tapi tidak krisis (false alarm)"],
          ["False Negative", metrics.false_negative, "Tidak anomali tapi krisis (lolos deteksi)"],
          ["True Negative", metrics.true_negative, "Tidak anomali & memang normal"],
        ],
        [2600, 2000, 4200]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      ...figureCentered("02_control_chart_spe.png", 560, 224, "Gambar 2. Control Chart SPE pada Data Test (merah = crisis_label aktual)"),

      // ===== 4. INTERPRETASI =====
      h1("4. Interpretasi Hasil"),
      h2("4.1 Variabel Pembeda Utama (Effect Size)"),
      p("Untuk memahami variabel apa yang paling membedakan observasi anomali dari observasi normal, " +
        "dihitung effect size (standardized mean difference) pada data test. Effect size positif berarti " +
        "nilai variabel tersebut lebih tinggi pada kelompok anomali; negatif berarti lebih rendah."),
      simpleTable(
        ["Variabel", "Effect Size", "Arah"],
        effectRows,
        [3200, 1600, 4000]
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      ...figureCentered("04_effect_size.png", 500, 375, "Gambar 3. Effect Size Tiap Variabel (Anomali vs Normal)"),
      h2("4.2 Studi Kasus: Observasi dengan Anomaly Score Tertinggi"),
      p("Observasi dengan anomaly score tertinggi pada data test adalah Brasil (BRA) tahun 1990, " +
        "bertepatan dengan periode krisis hiperinflasi Brasil pada awal dekade 1990-an. Contribution plot " +
        "menunjukkan bahwa GDP Growth, GDP Per Capita Growth, dan Gross Savings to GDP adalah tiga variabel " +
        "penyumbang residual (SPE) terbesar pada observasi ini \u2014 mengindikasikan bahwa kombinasi pertumbuhan " +
        "ekonomi dan tabungan domestik yang ekstrem menjadi ciri utama anomali pada kasus ini."),
      ...figureCentered("03_contribution_plot_top_anomaly.png", 500, 300, "Gambar 4. Contribution Plot \u2014 Brasil 1990"),
      h2("4.3 Diskusi"),
      p("Model PCA menghasilkan AUC-ROC sebesar " + fmt(metrics["AUC-ROC"]) + " dan precision " +
        fmt(metrics["Precision"]) + ", namun recall relatif rendah (" + fmt(metrics["Recall"]) +
        " atau setara " + fmtPct(metrics["Recall"]) + "). Artinya, dari observasi krisis aktual di data test, " +
        "model hanya berhasil mendeteksi sekitar " + fmtPct(metrics["Recall"]) + "-nya, sementara sisanya " +
        "(" + metrics.false_negative + " observasi) tidak terdeteksi sebagai anomali. Karakteristik ini " +
        "menunjukkan bahwa PCA cenderung konservatif: ketika suatu observasi ditandai sebagai anomali, " +
        "kemungkinan besar itu memang krisis (precision " + fmt(metrics.Precision) + "), namun model ini " +
        "tidak cukup sensitif untuk menangkap seluruh kasus krisis, kemungkinan karena PCA hanya mampu " +
        "menangkap hubungan linier antar variabel, sementara sebagian pola krisis mungkin bersifat non-linier."),
      p("Variabel yang paling konsisten membedakan kelompok anomali dari kelompok normal adalah Gross " +
        "Savings to GDP, Exports to GDP, Trade to GDP, dan Imports to GDP (cenderung lebih tinggi pada " +
        "observasi anomali), serta GDP Per Capita Growth dan GDP Growth (cenderung lebih rendah pada " +
        "observasi anomali). Pola ini konsisten dengan literatur krisis eksternal, di mana keterbukaan " +
        "ekonomi (trade openness) yang tinggi berbarengan dengan perlambatan pertumbuhan sering menjadi " +
        "sinyal kerentanan makroekonomi."),

      // ===== 5. KESIMPULAN =====
      h1("5. Kesimpulan dan Keterbatasan"),
      bullet("PCA dengan " + metrics.k_components + " komponen utama berhasil menangkap 95,4% variasi dari 14 indikator makroekonomi yang digunakan."),
      bullet("Kombinasi SPE dan Hotelling's T\u00B2 menghasilkan precision " + fmt(metrics.Precision) + " namun recall hanya " + fmt(metrics.Recall) + ", menunjukkan model lebih konservatif (sedikit false alarm) dibanding sensitif (banyak krisis lolos deteksi)."),
      bullet("Nilai tambah utama PCA dibanding metode lain (Autoencoder, OCSVM) adalah interpretabilitas melalui contribution plot dan analisis effect size, yang memungkinkan identifikasi variabel penyebab anomali secara eksplisit."),
      bullet("Keterbatasan utama: PCA mengasumsikan hubungan linier antar variabel, sehingga pola anomali non-linier berpotensi tidak tertangkap \u2014 perlu dibandingkan dengan hasil algoritma non-linier (Autoencoder, One-Class SVM) dari anggota kelompok lain pada tahap evaluasi dan seleksi model."),
      bullet("Hasil ini menggunakan pipeline preprocessing yang seragam dengan algoritma lain dalam kelompok (14 fitur yang sama, split data yang sama, StandardScaler yang sama) sesuai kerangka komparatif pada proposal Section 3.2.1, sehingga siap diintegrasikan ke tahap perbandingan enam algoritma."),

      // ===== REFERENSI =====
      h1("Referensi"),
      p("Jackson, J. E., & Mudholkar, G. S. (1979). Control Procedures for Residuals Associated With Principal Component Analysis. Technometrics, 21(3), 341\u2013349."),
      p("World Bank. Global Economic Monitor. World Bank Open Data."),
      p("Proposal Kelompok 3SI1 Kelompok 4. Analisis Deteksi Anomali pada Indikator Makroekonomi Global sebagai Instrumen Early Warning System Krisis Ekonomi."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(__dirname, "Laporan_PCA_Anomaly_Detection.docx"), buffer);
  console.log("Laporan berhasil dibuat.");
});
