# =============================================================================
# make_laporan_resmi_pdf.py
# Membuat Laporan Resmi Deteksi Anomali DBSCAN dalam format PDF Portrait (A4)
# Menggunakan ReportLab dengan typography elegan, cover resmi, dan grafik tersemat.
# Output: output/Laporan_Resmi_DBSCAN.pdf
# =============================================================================

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

OUTPUT_DIR = "output"
PDF_PATH = os.path.join(OUTPUT_DIR, "Laporan_Resmi_DBSCAN.pdf")

# Palette warna resmi
C_PRIMARY   = colors.HexColor("#1d3557") # Dark Navy
C_SECONDARY = colors.HexColor("#457b9d") # Medium Blue
C_ACCENT    = colors.HexColor("#e63946") # Crimson Red
C_GOLD      = colors.HexColor("#f4a261") # Gold
C_GREEN     = colors.HexColor("#2a9d8f") # Teal Green
C_LIGHT_BG  = colors.HexColor("#f8f9fa") # Light Gray
C_BOX_BG    = colors.HexColor("#edf6f9") # Soft Blue Gray
C_DARK_TEXT = colors.HexColor("#212529") # Dark Slate Text


class NumberedCanvas(canvas.Canvas):
    """
    Canvas kustom dua tahap (two-pass) untuk menghitung total halaman secara dinamis
    dan menambahkan running header serta running footer di setiap halaman.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Halaman Cover tidak diberi header & footer
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#6c757d"))

        # Header Atas
        self.drawString(54, 805, "LAPORAN RESMI: DETEKSI ANOMALI MAKROEKONOMI DENGAN DBSCAN")
        self.drawRightString(541, 805, "UNSUPERVISED LEARNING (1990-2024)")
        self.setStrokeColor(colors.HexColor("#dee2e6"))
        self.setLineWidth(0.75)
        self.line(54, 798, 541, 798)

        # Footer Bawah
        self.line(54, 45, 541, 45)
        page_text = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(541, 32, page_text)
        self.drawString(54, 32, "Dokumentasi Proyek Data Mining — Pascasarjana / Sarjana STIS")
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=C_PRIMARY,
        alignment=1, # Center
        spaceAfter=15
    )

    style_cover_sub = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=C_SECONDARY,
        alignment=1,
        spaceAfter=25
    )

    style_h1 = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=C_PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=C_SECONDARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h3 = ParagraphStyle(
        "Heading3_Custom",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=C_PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=C_DARK_TEXT,
        spaceAfter=6
    )

    style_body_bold = ParagraphStyle(
        "BodyBold_Custom",
        parent=style_body,
        fontName="Helvetica-Bold"
    )

    style_bullet = ParagraphStyle(
        "Bullet_Custom",
        parent=style_body,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    style_callout = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#1b4965")
    )

    style_caption = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6c757d"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    def make_callout(text, title=None, border_color=C_SECONDARY, bg_color=C_BOX_BG):
        story_box = []
        if title:
            story_box.append(Paragraph(f"<b>{title}</b>", style_h3))
            story_box.append(Spacer(1, 3))
        story_box.append(Paragraph(text, style_callout))
        t = Table([[story_box]], colWidths=[487])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), bg_color),
            ("BOX", (0,0), (-1,-1), 1.2, border_color),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ]))
        return t

    story = []

    # =========================================================================
    # HALAMAN COVER RESMI
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("LAPORAN RESMI DATA MINING", ParagraphStyle(
        "CoverHeader", fontName="Helvetica-Bold", fontSize=12,
        textColor=C_GOLD, alignment=1, spaceAfter=15
    )))
    story.append(HRFlowable(width="80%", thickness=2, color=C_ACCENT, spaceAfter=20, spaceBefore=5))
    story.append(Paragraph("DETEKSI ANOMALI MAKROEKONOMI DAN KRISIS FINANSIAL MENGGUNAKAN ALGORITMA DBSCAN", style_cover_title))
    story.append(Paragraph("Eksperimen Komparatif Pendekatan Global vs. Per-Negara pada Data Panel 49 Negara (1990–2024)", style_cover_sub))
    story.append(HRFlowable(width="40%", thickness=1, color=C_SECONDARY, spaceAfter=35, spaceBefore=5))

    # Box Informasi Penyusun & Proyek
    info_table_data = [
        [Paragraph("<b>Mata Kuliah</b>", style_body), Paragraph(": Data Mining / Pembelajaran Mesin", style_body)],
        [Paragraph("<b>Metode</b>", style_body), Paragraph(": Unsupervised Learning (DBSCAN)", style_body)],
        [Paragraph("<b>Dataset</b>", style_body), Paragraph(": 49 Negara, 14 Indikator Makro, 1990–2024 (1.715 baris)", style_body)],
        [Paragraph("<b>Validasi</b>", style_body), Paragraph(": IMF Historical Systemic Crisis Database", style_body)],
        [Paragraph("<b>Tanggal Rilis</b>", style_body), Paragraph(f": {datetime.now().strftime('%d %B %Y')}", style_body)],
    ]
    t_info = Table(info_table_data, colWidths=[120, 367])
    t_info.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_LIGHT_BG),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#ced4da")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e9ecef")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(t_info)

    story.append(Spacer(1, 70))
    story.append(make_callout(
        "<b>Intisari Pertanyaan Penelitian:</b><br/>"
        "<i>\"Mampukah algoritma komputer mendeteksi krisis ekonomi global secara objektif hanya dengan "
        "menganalisis pola kepadatan ruang data (density), tanpa pernah diajari atau diberi label krisis sebelumnya?\"</i>",
        title="Ringkasan Ide Utama",
        border_color=C_PRIMARY,
        bg_color=C_BOX_BG
    ))
    story.append(PageBreak())

    # =========================================================================
    # RINGKASAN EKSEKUTIF
    # =========================================================================
    story.append(Paragraph("RINGKASAN EKSEKUTIF", style_h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_PRIMARY, spaceAfter=10))

    story.append(Paragraph(
        "Krisis ekonomi adalah peristiwa langka yang membawa kehancuran sosial dan finansial sangat besar. "
        "Namun, mendeteksi kedatangan krisis adalah tugas yang sangat menantang karena krisis jarang sekali berulang "
        "dengan pola pemicu yang persis sama. Laporan resmi ini menyajikan hasil implementasi algoritma "
        "<b>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</b> pada data makroekonomi "
        "49 negara selama 35 tahun (1990 hingga 2024).",
        style_body
    ))
    story.append(Paragraph(
        "Secara garis besar, laporan ini menyimpulkan tiga temuan paling penting:",
        style_body
    ))
    story.append(Paragraph(
        "<b>1. Pembelajaran Tak Terawasi (Unsupervised) Berhasil:</b> Tanpa pernah diajari contoh krisis oleh manusia, "
        "model DBSCAN Per-Negara berhasil mendeteksi <b>32.3% dari seluruh krisis ekonomi nyata</b> yang dicatat oleh IMF, "
        "menghasilkan skor ROC-AUC sebesar <b>0.638</b> dan Average Precision sebesar <b>0.248</b> (jauh di atas tebakan acak 0.134).",
        style_bullet
    ))
    story.append(Paragraph(
        "<b>2. Pendekatan 'Per-Negara' Jauh Lebih Unggul dari 'Global':</b> Menerapkan satu standar dunia (DBSCAN Global) "
        "terbukti gagal karena hanya mendeteksi 13 anomali ekstrem (Recall hanya 3.5%). Model Global tidak peka terhadap krisis di "
        "negara maju karena data mereka tidak seekstrem negara berkembang yang mengalami hiperinflasi. Sebaliknya, pendekatan "
        "DBSCAN Per-Negara mengamati sejarah spesifik masing-masing negara, sehingga mampu menangkap 189 titik anomali yang proporsional.",
        style_bullet
    ))
    story.append(Paragraph(
        "<b>3. Krisis Nilai Tukar & COVID-19 Paling Mudah Terbaca:</b> Model ini paling sensitif menangkap <b>Krisis Pandemi COVID-19 (Recall 63.3%)</b> "
        "dan <b>Krisis Nilai Tukar / Mata Uang (Recall 54.3%)</b>, karena peristiwa tersebut memicu guncangan langsung serentak pada cadangan "
        "devisa dan PDB. Sebaliknya, Krisis Perbankan (Recall 23.4%) lebih menantang karena kerap bersembunyi di neraca internal bank "
        "sebelum merembet ke variabel makro agregat.",
        style_bullet
    ))

    story.append(Spacer(1, 10))

    # =========================================================================
    # BAB I: PENDAHULUAN
    # =========================================================================
    story.append(Paragraph("BAB I: PENDAHULUAN", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "Kondisi perekonomian suatu negara mirip dengan kondisi kesehatan tubuh manusia. Dalam keadaan normal dan sehat, "
        "indikator-indikator vital bergerak dalam batas yang wajar dan stabil. Namun saat krisis menyerang, indikator tersebut "
        "melonjak atau anjlok secara drastis ke wilayah yang tidak wajar. Sepanjang 1990–2024, dunia telah dihantam berbagai peristiwa "
        "katastropik, seperti Krisis Moneter Asia 1997/1998, Krisis Keuangan Global 2008, Krisis Utang Eropa 2010, dan Pandemi COVID-19 2020.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Mengapa Memilih Unsupervised Learning?</b> Pada metode supervised learning konvensional, model membutuhkan ribuan "
        "contoh krisis berlabel untuk belajar. Masalahnya, krisis ekonomi adalah peristiwa langka (hanya ~13.3% dari total data amatan). "
        "Lebih parah lagi, krisis masa depan kerap kali berkarakteristik baru (<i>Black Swan Event</i>) yang polanya tidak sama dengan masa lalu. "
        "Dengan Unsupervised Learning, algoritma tidak mendikte pola krisis masa lalu, melainkan secara mandiri mengidentifikasi "
        "setiap titik data yang 'menyimpang dari kerumunan normal'.",
        style_body
    ))

    # =========================================================================
    # BAB II: MEMAHAMI METODE DBSCAN (UNTUK ORANG AWAM)
    # =========================================================================
    story.append(Paragraph("BAB II: MEMAHAMI METODE DBSCAN", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "DBSCAN (<i>Density-Based Spatial Clustering of Applications with Noise</i>) adalah algoritma pengelompokan yang bekerja "
        "berdasarkan <b>kepadatan</b>. Intuisinya sangat sederhana dan manusiawi: hal-hal yang wajar atau normal akan terjadi berulang kali, "
        "sehingga titik-titik datanya berkumpul padat membentuk kerumunan. Sebaliknya, anomali atau krisis adalah peristiwa aneh yang jarang "
        "terjadi, sehingga posisinya terpencil sendirian di wilayah yang sunyi.",
        style_body
    ))

    story.append(make_callout(
        "<b>Analogi Pesta di Gedung:</b><br/>"
        "Bayangkan sebuah gedung pesta dengan 1.000 tamu. 900 tamu berkumpul di sekitar meja prasmanan dan panggung (berdiri berdekatan, "
        "membentuk kerumunan padat — ini adalah <i>Kondisi Ekonomi Normal</i>). Namun ada 5 tamu yang bertingkah mencurigakan: satu orang melamun "
        "sendirian di tangga darurat, satu orang memojok di sudut tempat parkir yang gelap. Petugas keamanan yang mencari orang aneh cukup "
        "mencari: <i>siapa yang berada di ruang sunyi tanpa teman di dekatnya?</i> Orang-orang terisolasi itulah yang disebut <b>Noise / Anomali</b>.",
        title="Analogi Pemahaman DBSCAN"
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DBSCAN membagi data ke dalam 3 jenis titik:",
        style_body
    ))
    story.append(Paragraph("• <b>Core Point (Titik Inti)</b>: Titik yang berada di pusat kerumunan padat (memiliki minimal <i>MinPts</i> tetangga dalam radius epsilon).", style_bullet))
    story.append(Paragraph("• <b>Border Point (Titik Batas)</b>: Titik di tepi kerumunan yang menempel pada titik inti.", style_bullet))
    story.append(Paragraph("• <b>Noise / Anomaly (Label -1)</b>: Titik yang tidak memiliki tetangga cukup dan tidak terjangkau kerumunan. <b>Inilah yang dideteksi sebagai krisis.</b>", style_bullet))

    story.append(Paragraph(
        "<b>Kenapa Memilih DBSCAN, Bukan K-Means?</b> K-Means memaksa seluruh titik masuk ke dalam salah satu klaster dan hanya bisa "
        "mengenali bentuk bulat simetris. Outlier pada K-Means akan merusak titik pusat (centroid). Sebaliknya, DBSCAN mampu menangani "
        "klaster dengan bentuk melengkung arbitrer dan memiliki mekanisme bawaan untuk langsung memisahkan outlier sebagai noise.",
        style_body
    ))

    # =========================================================================
    # BAB III: DATA DAN LANGKAH PREPROCESSING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB III: DATA DAN LANGKAH PREPROCESSING", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "Data yang digunakan mencakup <b>49 negara</b> selama <b>35 tahun (1990–2024)</b> dengan total <b>1.715 baris data</b> "
        "dan <b>14 indikator makroekonomi</b> (PDB, inflasi, pengangguran, cadangan devisa, depresiasi kurs, pertumbuhan kredit, "
        "saldo fiskal, utang pemerintah, dll.). Data mentah tidak bisa langsung dimasukkan ke DBSCAN tanpa 4 langkah preprocessing:",
        style_body
    ))

    # Tabel Langkah Preprocessing
    prep_table_data = [
        [Paragraph("<b>Tahap</b>", style_body_bold), Paragraph("<b>Masalah di Data Mentah</b>", style_body_bold), Paragraph("<b>Solusi & Kenapa Begitu?</b>", style_body_bold), Paragraph("<b>Hasil</b>", style_body_bold)],
        [
            Paragraph("<b>1. Imputasi</b>", style_body),
            Paragraph("Ada 573 sel data kosong (missing values).", style_body),
            Paragraph("Menggunakan <b>KNN Imputer (k=5)</b>. Menghapus data akan menghilangkan tahun krisis. KNN meminjam nilai rata-rata dari 5 negara yang karakteristiknya paling mirip.", style_body),
            Paragraph("100% data terisi utuh (0 missing).", style_body)
        ],
        [
            Paragraph("<b>2. Outlier</b>", style_body),
            Paragraph("Nilai hiperinflasi ekstrem (misal Peru 1990 inflasi 7.481%).", style_body),
            Paragraph("<b>Winsorization persentil 1% & 99%</b>. Jika dibiarkan liar, skala inflasi akan menenggelamkan variabel lain. Angka 7.481% dipotong ke ~120% agar tetap terbaca anomali tanpa merusak skala.", style_body),
            Paragraph("Varians terjaga, sinyal anomali tetap hidup.", style_body)
        ],
        [
            Paragraph("<b>3. Skalasi</b>", style_body),
            Paragraph("Satuan variabel beda jauh (PDB per kapita 40.000 vs inflasi 5%).", style_body),
            Paragraph("<b>RobustScaler</b> (menggunakan Median dan IQR). StandardScaler sangat sensitif terhadap outlier. RobustScaler membagi dengan rentang kuartil sehingga kebal terhadap distorsi pencilan.", style_body),
            Paragraph("Seluruh 14 fitur berada dalam skala berimbang.", style_body)
        ],
        [
            Paragraph("<b>4. Reduksi (PCA)</b>", style_body),
            Paragraph("14 dimensi memicu <i>Curse of Dimensionality</i> (jarak antar titik jadi seragam).", style_body),
            Paragraph("<b>Principal Component Analysis</b>. Merangkum 14 indikator menjadi komponen independen.<br/>• PC1 (43.1%): Stabilitas Moneter & Kurs<br/>• PC2 (20.7%): Riil, Utang & Fiskal", style_body),
            Paragraph("2 komponen menjelaskan <b>63.8% varians</b> informasi.", style_body)
        ]
    ]

    t_prep = Table(prep_table_data, colWidths=[65, 120, 212, 90])
    t_prep.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#ced4da")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_LIGHT_BG]),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t_prep)
    story.append(Spacer(1, 10))

    # Sematkan Gambar Preprocessing jika ada
    img_pca = os.path.join(OUTPUT_DIR, "pca_scatter.png")
    if os.path.exists(img_pca):
        story.append(Image(img_pca, width=420, height=210))
        story.append(Paragraph("Gambar 3.1: Proyeksi Data ke Ruang 2D PCA (PC1 vs PC2) memperlihatkan klaster normal vs titik ekstrem", style_caption))

    # =========================================================================
    # BAB IV: EKSPERIMEN & IMPLEMENTASI MODEL DBSCAN
    # =========================================================================
    story.append(Paragraph("BAB IV: EKSPERIMEN & IMPLEMENTASI DBSCAN", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "<b>Penentuan Epsilon Optimal:</b> Nilai radius epsilon ditentukan secara matematis menggunakan <b>Grafik Jarak Tetangga ke-$k$ ($k$-Distance Graph)</b>. "
        "Titik siku (knee/elbow) pada kurva jarak tetangga menandai batas alami di mana titik beralih dari kerumunan normal ke anomali yang terisolasi.",
        style_body
    ))

    story.append(Paragraph(
        "Kami membandingkan dua arsitektur pemodelan:",
        style_body
    ))
    story.append(Paragraph(
        "<b>1. DBSCAN Skala Global (eps=3.16, MinPts=8):</b> Seluruh 1.715 data dunia disatukan dalam satu ruang bersama. "
        "Hasilnya: model hanya mendeteksi <b>13 anomali (0.76%)</b>. Model ini terlalu kaku karena menuntut sebuah negara harus mengalami "
        "guncangan fantastis setingkat hiperinflasi ribuan persen di Amerika Latin agar dianggap 'aneh' secara skala dunia. "
        "Krisis di AS (2008) atau Yunani (2010) terlewatkan sama sekali.",
        style_body
    ))
    story.append(Paragraph(
        "<b>2. DBSCAN Skala Per-Negara (eps ~ 2.4, MinPts=4):</b> Model dijalankan mandiri untuk tiap negara berdasarkan "
        "sejarah 35 tahunnya sendiri. Hasilnya: model mendeteksi <b>189 anomali (11.0%)</b>. Model ini jauh lebih masuk akal secara ekonomi "
        "karena mampu mengapresiasi bahwa pertumbuhan ekonomi 1% bagi Indonesia adalah anomali perlambatan parah, sedangkan bagi Jepang adalah kondisi biasa.",
        style_body
    ))

    img_kdist = os.path.join(OUTPUT_DIR, "kdist_global.png")
    if os.path.exists(img_kdist):
        story.append(Image(img_kdist, width=380, height=180))
        story.append(Paragraph("Gambar 4.1: Penentuan Nilai Epsilon Optimal via k-Distance Graph Titik Siku", style_caption))

    # =========================================================================
    # BAB V: EVALUASI KOMPREHENSIF DAN ANALISIS HASIL
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB V: EVALUASI KOMPREHENSIF", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "Hasil prediksi anomali DBSCAN diverifikasi secara ketat terhadap database resmi <b>IMF Crisis Database</b> "
        "(229 tahun krisis riil vs 1.486 tahun normal).",
        style_body
    ))

    # Tabel Evaluasi Lengkap
    eval_table_data = [
        [Paragraph("<b>Metrik Evaluasi</b>", style_body_bold), Paragraph("<b>DBSCAN Global</b>", style_body_bold), Paragraph("<b>DBSCAN Per-Negara</b>", style_body_bold), Paragraph("<b>Makna Praktis</b>", style_body_bold)],
        [Paragraph("True Positive (TP)", style_body), Paragraph("8", style_body), Paragraph("<b>74</b>", style_body), Paragraph("Krisis nyata yang berhasil ditemukan model.", style_body)],
        [Paragraph("False Positive (FP)", style_body), Paragraph("5", style_body), Paragraph("115", style_body), Paragraph("Prediksi anomali tetapi bukan krisis menurut IMF.", style_body)],
        [Paragraph("False Negative (FN)", style_body), Paragraph("221", style_body), Paragraph("<b>155</b>", style_body), Paragraph("Krisis nyata yang luput/kecolongan.", style_body)],
        [Paragraph("Accuracy", style_body), Paragraph("86.8%", style_body), Paragraph("84.3%", style_body), Paragraph("Menyesatkan karena 86.7% data memang kondisi normal.", style_body)],
        [Paragraph("Precision", style_body), Paragraph("<b>61.5%</b>", style_body), Paragraph("39.2%", style_body), Paragraph("Dari semua prediksi anomali, berapa % yang terbukti krisis.", style_body)],
        [Paragraph("Recall (Daya Tangkap)", style_body), Paragraph("3.5%", style_body), Paragraph("<b>32.3%</b>", style_body), Paragraph("Dari semua krisis riil, berapa % yang tertangkap (Per-Negara 9x lipat).", style_body)],
        [Paragraph("F1-Score", style_body), Paragraph("0.066", style_body), Paragraph("<b>0.354</b>", style_body), Paragraph("Rata-rata harmonik Precision & Recall (Per-Negara unggul telak).", style_body)],
        [Paragraph("ROC-AUC", style_body), Paragraph("0.516", style_body), Paragraph("<b>0.638</b>", style_body), Paragraph("Kemampuan membedakan krisis vs normal (0.500 = tebakan acak).", style_body)],
        [Paragraph("Average Precision (AP)", style_body), Paragraph("0.156", style_body), Paragraph("<b>0.248</b>", style_body), Paragraph("Area di bawah kurva PR (baseline tebakan acak = 0.134).", style_body)],
        [Paragraph("Silhouette Score", style_body), Paragraph("<b>0.777</b>", style_body), Paragraph("0.273", style_body), Paragraph("Metrik kualitas klaster tanpa label (lihat penjelasan di bawah).", style_body)]
    ]

    t_eval = Table(eval_table_data, colWidths=[110, 80, 105, 192])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#ced4da")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_LIGHT_BG]),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 8))

    story.append(make_callout(
        "<b>Penjelasan Kritis: Mengapa Silhouette Global Lebih Tinggi (0.777 vs 0.273)?</b><br/>"
        "Silhouette mengukur kepadatan klaster secara geometris. DBSCAN Global menaruh 99.2% data ke dalam 1 bola raksasa "
        "yang sangat padat dan membuang 13 anomali jauh di luar angkasa. Secara geometris matematika murni, bentuk 1 bola padat "
        "menghasilkan Silhouette tinggi. Namun bola raksasa itu mengubur seluruh krisis lokal sehingga model Global gagal mendeteksi "
        "96.5% krisis nyata. DBSCAN Per-Negara lebih heterogen secara geometris, namun 9 kali lebih peka menangkap krisis di dunia nyata!",
        title="Wawasan Penting Mengenai Metrik Unsupervised"
    ))

    story.append(Spacer(1, 6))

    # Daya Tangkap per Jenis Krisis
    story.append(Paragraph("<b>Daya Tangkap Berdasarkan Jenis Krisis (Per-Country):</b>", style_h3))
    story.append(Paragraph("• <b>Pandemi COVID-19 (2020) — Recall 63.3% (31 dari 49 negara):</b> Guncangan serentak paling masif pada PDB dan belanja publik.", style_bullet))
    story.append(Paragraph("• <b>Krisis Nilai Tukar / Mata Uang — Recall 54.3% (19 dari 35 krisis):</b> Depresiasi tajam dan terkurasnya cadangan devisa langsung mencuat dari klaster normal.", style_bullet))
    story.append(Paragraph("• <b>Krisis Utang Pemerintah — Recall 43.8% (7 dari 16 krisis):</b> Gagal bayar utang meninggalkan jejak defisit dan rasio utang yang melonjak.", style_bullet))
    story.append(Paragraph("• <b>Krisis Perbankan — Recall 23.4% (37 dari 158 krisis):</b> Paling sulit karena krisis perbankan kerap bermula di neraca internal bank dan memiliki jeda waktu sebelum merembet ke makro agregat.", style_bullet))

    img_pr = os.path.join(OUTPUT_DIR, "eval_pr_curve.png")
    if os.path.exists(img_pr):
        story.append(Image(img_pr, width=400, height=180))
        story.append(Paragraph("Gambar 5.1: Kurva Precision-Recall memperlihatkan keunggulan DBSCAN Per-Negara atas baseline acak", style_caption))

    # =========================================================================
    # BAB VI: STUDI KASUS HISTORIS NYATA
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB VI: STUDI KASUS VALIDASI HISTORIS", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "Untuk membuktikan validitas model dalam konteks sejarah riil, empat peristiwa besar diuji:",
        style_body
    ))
    story.append(Paragraph(
        "<b>1. Krisis Moneter Asia 1997–1998 (Indonesia & Thailand):</b><br/>"
        "Di Indonesia, kejatuhan Rupiah 83% dan inflasi di atas 70% membuat tahun 1998 terdeteksi sebagai anomali baik pada model Global "
        "maupun Per-Negara. Pada model Per-Negara, Indonesia mencatatkan <b>Precision 1.000</b> (seluruh alarm anomali terbukti krisis IMF) "
        "dan F1-score 0.600. Thailand juga terdeteksi akurat di tahun 1997 dan 1998 (F1 = 0.667, Recall = 60%).",
        style_body
    ))
    story.append(Paragraph(
        "<b>2. Hiperinflasi Amerika Latin 1990–1994 (Brazil & Peru):</b><br/>"
        "Peru 1990 mencatat skor anomali mutlak tertinggi di dunia (1.000) akibat inflasi 7.481%. Brazil terdeteksi anomali berturut-turut "
        "selama 5 tahun (1990-1994), dan label anomali langsung hilang pada tahun 1995 setelah keberhasilan stabilisasi moneter <i>Plano Real</i>.",
        style_body
    ))
    story.append(Paragraph(
        "<b>3. Krisis Utang Yunani 2010–2014 (Greece):</b><br/>"
        "Yunani mengalami kontraksi PDB 25% dan pengangguran 27%. DBSCAN Per-Negara berhasil mendeteksi <b>5 dari 6 tahun krisis Yunani (Recall 83.3%)</b> "
        "dengan nilai ROC-AUC sebesar 0.937, membuktikan kepekaan model terhadap krisis struktural berkepanjangan.",
        style_body
    ))
    story.append(Paragraph(
        "<b>4. Pandemi COVID-19 Tahun 2020:</b><br/>"
        "Sebanyak <b>31 dari 49 negara (63.3%)</b> terdeteksi serempak sebagai anomali pada tahun 2020. Ini membuktikan bahwa tanpa perlu "
        "diberitahu adanya wabah virus, algoritma secara mandiri mampu melihat bahwa tahun 2020 adalah tahun paling menyimpang dalam sejarah modern.",
        style_body
    ))

    img_f1 = os.path.join(OUTPUT_DIR, "eval_f1_per_economy.png")
    if os.path.exists(img_f1):
        story.append(Image(img_f1, width=420, height=190))
        story.append(Paragraph("Gambar 6.1: Performa Deteksi F1-Score per Negara pada DBSCAN Per-Country", style_caption))

    # =========================================================================
    # BAB VII: KESIMPULAN & REKOMENDASI
    # =========================================================================
    story.append(Paragraph("BAB VII: KESIMPULAN & REKOMENDASI", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY, spaceAfter=8))

    story.append(Paragraph(
        "<b>Kesimpulan:</b><br/>"
        "1. Metode Unsupervised DBSCAN terbukti efektif dan objektif dalam mendeteksi anomali makroekonomi tanpa supervisi label.<br/>"
        "2. Pendekatan Per-Negara jauh lebih unggul daripada pendekatan Global dalam mendeteksi krisis lokal dan regional.<br/>"
        "3. Evaluasi menggunakan F1-score, Precision-Recall Curve, dan breakdown jenis krisis memberikan pemahaman yang utuh dan tidak terjebak paradoks akurasi.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Rekomendasi untuk Otoritas Moneter & Pembuat Kebijakan:</b><br/>"
        "• Adopsi DBSCAN Per-Negara sebagai alat pemantauan sekunder (<i>secondary surveillance layer</i>) tanpa asumsi sebaran normal.<br/>"
        "• Tingkatkan frekuensi data ke tingkat kuartalan atau bulanan untuk membangun sistem peringatan dini (<i>Early Warning System</i>).<br/>"
        "• Integrasikan indikator likuiditas mikro-perbankan agar deteksi krisis perbankan dapat ditingkatkan.",
        style_body
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Laporan Resmi PDF berhasil dibuat di: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
