/**
 * generate_report.js
 * ===================
 * Membuat laporan Word (docx) untuk bagian PCA-based Anomaly Detection,
 * Kelompok 4 - Analisis Deteksi Anomali pada Indikator Makroekonomi Global
 * sebagai Instrumen Early Warning System (EWS) Krisis Ekonomi.
 *
 * Menjalankan:  node generate_report.js
 * Membutuhkan:  hasil terbaru di ../results/*.csv dan ../figures/*.png
 *               (jalankan scripts/run_all.py terlebih dahulu bila belum ada).
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ImageRun, AlignmentType, ShadingType, BorderStyle,
  PageOrientation, PageBreak,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const RESULTS = path.join(ROOT, "results");
const FIGURES = path.join(ROOT, "figures");

// ---------------------------------------------------------------------------
// Util: baca CSV sederhana (tanpa dependency tambahan)
// ---------------------------------------------------------------------------
function readCSV(filePath) {
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  const lines = raw.split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const row = {};
    headers.forEach((h, i) => (row[h] = vals[i]));
    return row;
  });
}

function fmt(numStr, digits = 3) {
  const n = parseFloat(numStr);
  if (Number.isNaN(n)) return numStr;
  return n.toFixed(digits);
}

function fmtInt(numStr) {
  const n = parseFloat(numStr);
  if (Number.isNaN(n)) return numStr;
  return String(Math.round(n));
}

const meta = readCSV(path.join(RESULTS, "pca_run_metadata.csv"));
const metaMap = {};
meta.forEach((r) => (metaMap[r.metric] = r.value));

const summary = readCSV(path.join(RESULTS, "pca_model_summary.csv"));
const evalMetrics = readCSV(path.join(RESULTS, "evaluation_metrics.csv"));
const effectSize = readCSV(path.join(RESULTS, "effect_size.csv"));

function getMetric(scoreLabel, metricLabel) {
  const row = evalMetrics.find((r) => r.score === scoreLabel && r.metric === metricLabel);
  return row ? row.value : "-";
}

function getEffect(scoreLabel, col) {
  const row = effectSize.find((r) => r.score === scoreLabel);
  return row ? row[col] : "-";
}

// ---------------------------------------------------------------------------
// Helper styling
// ---------------------------------------------------------------------------
const FONT = "Calibri";
const ACCENT = "C44E52";
const ACCENT_DARK = "8B2E33";

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, bold: true, color: ACCENT_DARK, font: FONT })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: FONT })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })],
  });
}

function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { after: 240 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, italics: true, font: FONT, size: 20, color: "555555" })],
  });
}

function imageParagraph(filename, width = 560) {
  const filePath = path.join(FIGURES, filename);
  const buffer = fs.readFileSync(filePath);
  // deteksi rasio asli agar tidak gepeng -- baca dimensi PNG dari header
  const { width: origW, height: origH } = pngDimensions(buffer);
  const height = Math.round((width * origH) / origW);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new ImageRun({
        data: buffer,
        transformation: { width, height },
        type: "png",
      }),
    ],
  });
}

function pngDimensions(buffer) {
  // PNG: width/height ada di IHDR chunk, offset 16-24
  const width = buffer.readUInt32BE(16);
  const height = buffer.readUInt32BE(20);
  return { width, height };
}

function simpleTable(headerRow, dataRows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const mkCell = (text, isHeader = false, width) =>
    new TableCell({
      width: { size: width, type: WidthType.DXA },
      shading: isHeader ? { type: ShadingType.CLEAR, color: "auto", fill: "C44E52" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [
        new Paragraph({
          children: [
            new TextRun({
              text: String(text),
              bold: isHeader,
              color: isHeader ? "FFFFFF" : "000000",
              font: FONT,
              size: 20,
            }),
          ],
        }),
      ],
    });

  const rows = [
    new TableRow({
      tableHeader: true,
      children: headerRow.map((h, i) => mkCell(h, true, colWidths[i])),
    }),
    ...dataRows.map(
      (r) => new TableRow({ children: r.map((c, i) => mkCell(c, false, colWidths[i])) })
    ),
  ];

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows,
  });
}

// ---------------------------------------------------------------------------
// Susun konten dokumen
// ---------------------------------------------------------------------------
const titlePage = [
  new Paragraph({ spacing: { before: 2000 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "LAPORAN BAGIAN INDIVIDU", bold: true, size: 28, font: FONT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200 },
    children: [
      new TextRun({
        text: "PCA-based Anomaly Detection",
        bold: true,
        size: 40,
        color: ACCENT_DARK,
        font: FONT,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 400 },
    children: [
      new TextRun({
        text: "Instrumen Early Warning System (EWS) Krisis Ekonomi",
        size: 28,
        font: FONT,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100 },
    children: [
      new TextRun({
        text: "Bagian dari Proposal Kelompok 4:",
        italics: true,
        size: 22,
        font: FONT,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 800 },
    children: [
      new TextRun({
        text: '"Analisis Deteksi Anomali pada Indikator Makroekonomi Global sebagai Instrumen Early Warning System (EWS) Krisis Ekonomi"',
        italics: true,
        size: 22,
        font: FONT,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1200 },
    children: [
      new TextRun({
        text: "PROGRAM STUDI D-IV KOMPUTASI STATISTIK",
        size: 22,
        font: FONT,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "POLITEKNIK STATISTIKA STIS", size: 22, font: FONT })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "2025/2026", size: 22, font: FONT })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

const revisionNote = [
  heading1("Catatan Revisi Metodologi"),
  body(
    "Draf awal bagian PCA pada tugas ini melatih model menggunakan skema pembagian data " +
      "70% latih / 15% validasi / 15% uji terhadap data normal, mengikuti pola yang dipakai " +
      "pada implementasi Autoencoder. Skema tersebut menyebabkan hasil PCA tidak dapat " +
      "dibandingkan secara adil (apple-to-apple) dengan hasil Local Outlier Factor (LOF) milik " +
      "anggota kelompok lain, karena LOF di-scoring pada seluruh 1.714 observasi sedangkan PCA " +
      "versi awal hanya dievaluasi pada subset uji."
  ),
  body("Laporan ini menggunakan skema yang telah diperbaiki agar konsisten dengan LOF, yaitu:"),
  bullet(
    "StandardScaler dan PCA di-fit HANYA dari seluruh baris data normal (crisis_label = 0, " +
      "1.385 baris) -- bukan hanya 70%-nya -- sehingga prinsip 'PCA hanya mempelajari kondisi " +
      "normal' tetap terjaga tanpa ada informasi krisis yang bocor ke tahap pelatihan."
  ),
  bullet(
    "SPE (Squared Prediction Error) dan Hotelling's T\u00b2 dihitung untuk SELURUH 1.714 " +
      "observasi (normal + krisis), persis seperti cara LOF di-scoring pada seluruh dataset."
  ),
  bullet(
    "Upper Control Limit (UCL) tetap dihitung hanya dari sebaran SPE/T\u00b2 pada data normal, " +
      "sehingga tidak ada label krisis yang bocor ke penentuan ambang batas anomali."
  ),
  body(
    "Dengan skema ini, seluruh metrik evaluasi pada laporan ini dapat dibandingkan langsung " +
      "dengan metrik LOF pada laporan gabungan kelompok."
  ),
];

const introSection = [
  heading1("1. Pendahuluan"),
  body(
    "Bagian ini merupakan implementasi individu dari algoritma PCA-based Anomaly Detection " +
      "sebagaimana dirancang dalam proposal Kelompok 4 (Bab 2.1.3.5 dan Bab 3.2.6). PCA " +
      "digunakan untuk mendeteksi anomali pada 14 indikator makroekonomi terstandardisasi dari " +
      "49 negara periode 1990-2024 (1.714 observasi panel), dengan crisis_label sebagai " +
      "referensi validasi eksternal yang disusun dari katalog krisis historis World Bank/IMF."
  ),
  body(
    "Prinsip metode: PCA membangun profil 'normalitas' makroekonomi dari data yang tidak " +
      "mengandung periode krisis, kemudian mengukur seberapa jauh setiap observasi (baik normal " +
      "maupun krisis) menyimpang dari profil tersebut melalui dua statistik komplementer: " +
      "Squared Prediction Error (SPE / Q-statistic) untuk mengukur reconstruction error di luar " +
      "ruang komponen utama, dan Hotelling's T\u00b2 untuk mengukur jarak di dalam ruang komponen " +
      "utama yang dipertahankan."
  ),
];

const dataSection = [
  heading1("2. Data dan Variabel"),
  body(
    `Model dilatih (fit) menggunakan ${fmtInt(metaMap["n_normal_train"])} baris data normal ` +
      `(crisis_label = 0), kemudian digunakan untuk men-scoring seluruh ${fmtInt(metaMap["n_total"])} ` +
      `observasi. Sebanyak ${fmtInt(metaMap["n_features"])} variabel numerik terstandardisasi (Z-score) ` +
      "digunakan sebagai fitur, identik dengan variabel yang dipakai algoritma lain di kelompok " +
      "ini (economy dan year hanya digunakan sebagai identifier, bukan fitur)."
  ),
  bullet("GDP_Growth, GDP_PerCapita_Growth \u2014 dinamika pertumbuhan ekonomi riil"),
  bullet("Inflation_CPI \u2014 tekanan harga konsumen"),
  bullet("Total_Reserves \u2014 ketahanan eksternal / cadangan devisa"),
  bullet("Unemployment \u2014 kondisi pasar tenaga kerja"),
  bullet("Current_Account_GDP, Trade_GDP, Exports_GDP, Imports_GDP \u2014 posisi eksternal & keterbukaan perdagangan"),
  bullet("FDI_Inflows_GDP \u2014 arus modal asing"),
  bullet("Gross_Savings_GDP, Investment_GDP \u2014 tabungan dan investasi domestik"),
  bullet("Exchange_Rate \u2014 stabilitas nilai tukar"),
  bullet("Manufacturing_Value \u2014 produktivitas sektor riil"),
];

const methodSection = [
  heading1("3. Metodologi"),
  heading2("3.1 Pemilihan Jumlah Komponen Utama (k)"),
  body(
    "PCA dilatih penuh (14 komponen) dari data normal, kemudian jumlah komponen yang " +
      "dipertahankan (k) ditentukan dari kriteria cumulative explained variance \u2265 95%, sesuai " +
      `proposal. Diperoleh k = ${fmtInt(metaMap["k_components"])} komponen dengan cumulative explained ` +
      `variance sebesar ${(parseFloat(metaMap["cumulative_var_at_k"]) * 100).toFixed(2)}%.`
  ),
  imageParagraph("01_scree_plot.png", 5600 / 10),
  caption("Gambar 1. Scree plot explained variance per komponen dan kumulatifnya."),
  simpleTable(
    ["Komponen", "Eigenvalue", "Explained Var. (%)", "Kumulatif (%)", "Dipertahankan"],
    summary.map((r) => [
      r.component,
      fmt(r.eigenvalue, 3),
      (parseFloat(r.explained_variance_ratio) * 100).toFixed(2),
      (parseFloat(r.cumulative_explained_variance) * 100).toFixed(2),
      r.retained === "True" ? "Ya" : "Tidak",
    ]),
    [1800, 1800, 2200, 2000, 1800]
  ),
  body(""),
  heading2("3.2 Perhitungan SPE dan Hotelling's T\u00b2"),
  body(
    "SPE mengukur reconstruction error (jumlah kuadrat residual di luar ruang k komponen " +
      "utama), sedangkan T\u00b2 mengukur jarak Mahalanobis-terbobot di dalam ruang k komponen " +
      "utama yang dipertahankan. Kedua statistik dihitung untuk seluruh 1.714 observasi " +
      "menggunakan scaler dan PCA yang telah di-fit dari data normal."
  ),
  heading2("3.3 Upper Control Limit (UCL)"),
  body(
    "UCL SPE dihitung dengan pendekatan chi-square (Jackson & Mudholkar, 1979) menggunakan " +
      "eigenvalue komponen yang dibuang dari data normal. UCL T\u00b2 dihitung dengan distribusi F " +
      "menggunakan n = jumlah data normal yang dipakai melatih PCA. Tingkat signifikansi " +
      `\u03b1 = ${metaMap["alpha"]} (UCL pada persentil ke-95). Observasi dengan SPE atau T\u00b2 di ` +
      "atas UCL masing-masing dikategorikan sebagai anomali."
  ),
  simpleTable(
    ["Statistik", "Nilai UCL (\u03b1 = 0.05)"],
    [
      ["SPE (Q-statistic)", fmt(metaMap["UCL_SPE"], 4)],
      ["Hotelling's T\u00b2", fmt(metaMap["UCL_T2"], 4)],
    ],
    [4000, 4000]
  ),
];

const resultSection = [
  heading1("4. Hasil"),
  heading2("4.1 Control Chart"),
  body(
    "Control chart berikut memplot SPE dan T\u00b2 untuk seluruh observasi (diurutkan sesuai " +
      "indeks panel data negara \u00d7 tahun), diwarnai berdasarkan crisis_label, dengan garis UCL " +
      "sebagai ambang batas anomali."
  ),
  imageParagraph("02_control_chart.png", 6200 / 10),
  caption("Gambar 2. Control chart SPE dan Hotelling's T\u00b2 terhadap UCL."),
  heading2("4.2 Metrik Evaluasi terhadap crisis_label"),
  body(
    "Berikut metrik evaluasi yang dihitung pada seluruh 1.714 observasi (normal + krisis), " +
      "konsisten dengan cara LOF dievaluasi di laporan kelompok."
  ),
  simpleTable(
    ["Skor", "AUC-ROC", "Average Precision"],
    [
      ["SPE", fmt(getMetric("SPE", "AUC-ROC"), 3), fmt(getMetric("SPE", "Average Precision (AP)"), 3)],
      ["T\u00b2", fmt(getMetric("T2", "AUC-ROC"), 3), fmt(getMetric("T2", "Average Precision (AP)"), 3)],
      [
        "Combined score",
        fmt(getMetric("combined_score", "AUC-ROC"), 3),
        fmt(getMetric("combined_score", "Average Precision (AP)"), 3),
      ],
    ],
    [3000, 2800, 2800]
  ),
  body(""),
  simpleTable(
    ["Kriteria Threshold (berbasis UCL)", "Precision", "Recall", "F1-Score"],
    [
      [
        "SPE > UCL_SPE",
        fmt(getMetric("SPE > UCL_SPE", "Precision"), 3),
        fmt(getMetric("SPE > UCL_SPE", "Recall"), 3),
        fmt(getMetric("SPE > UCL_SPE", "F1-Score"), 3),
      ],
      [
        "T\u00b2 > UCL_T2",
        fmt(getMetric("T2 > UCL_T2", "Precision"), 3),
        fmt(getMetric("T2 > UCL_T2", "Recall"), 3),
        fmt(getMetric("T2 > UCL_T2", "F1-Score"), 3),
      ],
      [
        "Combined (SPE atau T\u00b2)",
        fmt(getMetric("Combined (SPE atau T2)", "Precision"), 3),
        fmt(getMetric("Combined (SPE atau T2)", "Recall"), 3),
        fmt(getMetric("Combined (SPE atau T2)", "F1-Score"), 3),
      ],
    ],
    [3400, 2200, 2200, 2200]
  ),
  body(""),
  heading2("4.3 Effect Size: Krisis vs Normal"),
  body(
    "Selain metrik klasifikasi, dihitung ukuran effect size untuk melihat seberapa besar " +
      "perbedaan skor anomali antara periode krisis dan normal: Cohen's d (parametrik) dan " +
      "Cliff's delta (non-parametrik, lebih robust terhadap distribusi skor yang skewed)."
  ),
  imageParagraph("04_effect_size_plot.png", 5600 / 10),
  caption("Gambar 3. Effect size (Cohen's d dan Cliff's delta) skor SPE, T\u00b2, dan combined score."),
  simpleTable(
    ["Skor", "Cohen's d", "Interpretasi", "Cliff's delta", "Interpretasi"],
    [
      [
        "SPE",
        fmt(getEffect("SPE", "cohens_d"), 3),
        getEffect("SPE", "cohens_d_interpretation"),
        fmt(getEffect("SPE", "cliffs_delta"), 3),
        getEffect("SPE", "cliffs_delta_interpretation"),
      ],
      [
        "T\u00b2",
        fmt(getEffect("T2", "cohens_d"), 3),
        getEffect("T2", "cohens_d_interpretation"),
        fmt(getEffect("T2", "cliffs_delta"), 3),
        getEffect("T2", "cliffs_delta_interpretation"),
      ],
      [
        "Combined score",
        fmt(getEffect("combined_score", "cohens_d"), 3),
        getEffect("combined_score", "cohens_d_interpretation"),
        fmt(getEffect("combined_score", "cliffs_delta"), 3),
        getEffect("combined_score", "cliffs_delta_interpretation"),
      ],
    ],
    [2200, 1600, 1800, 1800, 1800]
  ),
  body(""),
  heading2("4.4 Contribution Plot"),
  body(
    "Untuk observasi krisis yang terdeteksi sebagai anomali, dihitung kontribusi rata-rata " +
      "tiap variabel terhadap SPE untuk mengidentifikasi indikator makroekonomi yang paling " +
      "menyimpang saat krisis terjadi."
  ),
  imageParagraph("03_contribution_plot.png", 5200 / 10),
  caption("Gambar 4. Rata-rata kontribusi tiap variabel terhadap SPE pada anomali saat krisis."),
];

const discussionSection = [
  heading1("5. Pembahasan"),
  body(
    "Hasil menunjukkan bahwa Hotelling's T\u00b2 memberikan diskriminasi krisis-vs-normal yang " +
      `lebih baik dibandingkan SPE pada dataset ini (AUC-ROC T\u00b2 = ${fmt(getMetric("T2", "AUC-ROC"), 3)} ` +
      `vs SPE = ${fmt(getMetric("SPE", "AUC-ROC"), 3)}, effect size T\u00b2 juga lebih besar). Hal ini ` +
      "mengindikasikan bahwa periode krisis pada data makroekonomi ini lebih sering " +
      "bermanifestasi sebagai kombinasi nilai ekstrem di dalam ruang komponen utama yang " +
      "menjelaskan variasi terbesar (ditangkap T\u00b2), dibandingkan sebagai pola yang gagal " +
      "direkonstruksi oleh komponen utama tersebut (ditangkap SPE)."
  ),
  body(
    "Skor gabungan (combined_score), yang mengambil rasio maksimum SPE dan T\u00b2 terhadap " +
      "UCL masing-masing, memberikan keseimbangan Precision-Recall-F1 yang wajar untuk dipakai " +
      "sebagai bagian dari sistem EWS, meskipun secara AUC-ROC sedikit lebih rendah dari T\u00b2 " +
      "saja -- menunjukkan bahwa penambahan SPE ke dalam skor gabungan sedikit menambah noise " +
      "pada kasus dataset ini."
  ),
  body(
    "Contribution plot menunjukkan variabel-variabel terkait cadangan devisa dan " +
      "investasi/tabungan domestik sebagai kontributor utama pada anomali yang terdeteksi saat " +
      "krisis, konsisten dengan literatur EWS klasik (Kaminsky et al., 1997) yang menekankan " +
      "peran cadangan devisa sebagai salah satu prediktor krisis mata uang terkuat."
  ),
  body(
    "Sebagai keterbatasan, UCL berbasis asumsi distribusi (chi-square approx. untuk SPE, F " +
      "untuk T\u00b2) dapat kurang presisi bila residual atau skor komponen utama tidak sepenuhnya " +
      "memenuhi asumsi tersebut. Penyertaan Cliff's delta sebagai pembanding non-parametrik " +
      "dimaksudkan untuk memitigasi keterbatasan ini."
  ),
];

const conclusionSection = [
  heading1("6. Kesimpulan"),
  bullet(
    `PCA dilatih dari ${fmtInt(metaMap["n_normal_train"])} observasi normal dan mempertahankan ` +
      `${fmtInt(metaMap["k_components"])} komponen utama (\u2265 95% cumulative explained variance).`
  ),
  bullet(
    `SPE dan T\u00b2 dihitung untuk seluruh ${fmtInt(metaMap["n_total"])} observasi, dengan UCL yang ` +
      "murni ditentukan dari sebaran data normal -- tidak ada leakage label krisis ke tahap " +
      "penentuan ambang batas."
  ),
  bullet(
    `Hotelling's T\u00b2 (AUC-ROC = ${fmt(getMetric("T2", "AUC-ROC"), 3)}) mengungguli SPE ` +
      `(AUC-ROC = ${fmt(getMetric("SPE", "AUC-ROC"), 3)}) dalam mendeteksi periode krisis pada ` +
      "dataset ini."
  ),
  bullet(
    "Skema evaluasi yang digunakan (fit dari seluruh data normal, scoring pada seluruh " +
      "dataset) memungkinkan perbandingan yang adil dengan hasil LOF dan algoritma lain di " +
      "laporan gabungan kelompok."
  ),
];

const doc = new Document({
  sections: [
    {
      properties: {
        page: { size: { width: 11907, height: 16840 } }, // A4 in DXA
      },
      children: [
        ...titlePage,
        ...revisionNote,
        ...introSection,
        ...dataSection,
        ...methodSection,
        ...resultSection,
        ...discussionSection,
        ...conclusionSection,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = path.join(__dirname, "Laporan_PCA_Anomaly_Detection_Kelompok4.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("Laporan berhasil dibuat:", outPath);
});
