import os
import json

def make_markdown_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    }

def make_code_cell(source_text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    }

cells = []

# Title & Metadata
cells.append(make_markdown_cell("""# 🌐 Deteksi Anomali Indikator Makroekonomi Global dengan One-Class Support Vector Machine (OC-SVM)

**Disusun oleh:** Nyimas Virna Salsa Lestari Risqia (222313307)  
**Mata Kuliah:** Data Mining — UTS Semester 6 (2026)  
**Kelompok:** 3SI1 — Kelompok 4  
**Metode:** One-Class Support Vector Machine (Kernel/Boundary-Based Unsupervised Anomaly Detection)

---

## 📌 Ringkasan Eksekutif & Prinsip Kerja

Notebook ini mengimplementasikan **One-Class Support Vector Machine (OC-SVM)** (Schölkopf et al., 2001) untuk deteksi anomali pada 14 indikator makroekonomi dari 49 negara periode 1990–2024 (1.714 observasi) sebagai komponen *Early Warning System* (EWS) krisis ekonomi.

### Prinsip Matematis OC-SVM:
1. **Pemetaan Ruang Dimensi Tinggi**: Menggunakan fungsi kernel Radial Basis Function ($RBF$) $K(x, y) = \\exp(-\\gamma ||x - y||^2)$ untuk memetakan data fitur ke ruang Hilbert (*feature space* $\\Phi(x)$).
2. **Maximal Margin Hyperplane**: Membentuk *hyperplane* optimal yang memisahkan mayoritas data normal dari titik asal (*origin*).
3. **Parameter $\\nu$ (nu)**: Mengontrol batas atas proporsi *training error* (anomali) dan batas bawah jumlah *support vectors*.
4. **Parameter $\\gamma$ (gamma)**: Mengontrol jangkauan pengaruh dari satu contoh pelatihan ($1 / 2\\sigma^2$).
"""))

# Cell 1: Setup Imports
cells.append(make_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
SEED = 42
np.random.seed(SEED)
print("Libraries loaded successfully.")
"""))

# Cell 2: Data Loading
cells.append(make_code_cell("""# Load dataset
data_path = '../data_cleaned.csv'
if not os.path.exists(data_path):
    data_path = 'data_cleaned.csv'

df = pd.read_csv(data_path)
print(f"Shape dataset: {df.shape}")
print(f"Jumlah negara: {df['economy'].nunique()}")
print(f"Rentang tahun: {df['year'].min()} - {df['year'].max()}")
print(f"Persentase crisis_label aktual: {df['crisis_label'].mean()*100:.2f}% ({df['crisis_label'].sum()} observasi)")

FEATURE_COLS = [
    'GDP_Growth', 'GDP_PerCapita_Growth', 'Inflation_CPI',
    'Total_Reserves', 'Unemployment', 'Current_Account_GDP',
    'Trade_GDP', 'FDI_Inflows_GDP', 'Exports_GDP', 'Imports_GDP',
    'Gross_Savings_GDP', 'Exchange_Rate', 'Manufacturing_Value',
    'Investment_GDP'
]

X = df[FEATURE_COLS].values
y_true = df['crisis_label'].values
"""))

# Cell 3: Grid Search Markdown
cells.append(make_markdown_cell("""## 1. ⚙️ Grid Search Hyperparameter (`nu` & `gamma`)

Proses eksplorasi parameter untuk menentukan kombinasi terbaik `nu` (0.01 – 0.25) dan `gamma` ('scale', 'auto', 0.001 – 1.0) dengan evaluasi metrik Precision, Recall, F1-Score, ROC-AUC, dan PR-AUC.
"""))

# Cell 4: Grid Search Execution
cells.append(make_code_cell("""nu_values = [0.01, 0.05, 0.08, 0.10, 0.15, 0.1919, 0.25]
gamma_values = ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

results = []
for nu in nu_values:
    for gamma in gamma_values:
        ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        preds = ocsvm.fit_predict(X)
        y_pred = (preds == -1).astype(int)
        scores = -ocsvm.decision_function(X).ravel()
        
        n_anom = int(y_pred.sum())
        pct_anom = n_anom / len(y_pred) * 100
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        
        results.append({
            'nu': nu,
            'gamma': str(gamma),
            'n_anomalies': n_anom,
            'pct_anomalies': round(pct_anom, 2),
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1_score': round(f, 4),
            'roc_auc': round(roc, 4),
            'pr_auc': round(pr_auc, 4)
        })

df_results = pd.DataFrame(results).sort_values('f1_score', ascending=False)
df_results.to_csv('grid_search_ocsvm.csv', index=False)
print("Top 10 Kombinasi Hyperparameter Terbaik:")
df_results.head(10)
"""))

# Cell 5: Best Model Fitting Markdown
cells.append(make_markdown_cell("""## 2. 🏆 Fitting Model Terbaik & Evaluasi Performa

Menggunakan kombinasi hyperparameter terbaik (`nu = 0.25`, `gamma = 0.001`) untuk melakukan pemodelan One-Class SVM final.
"""))

# Cell 6: Model Fitting & Evaluation Code
cells.append(make_code_cell("""best_nu = 0.25
best_gamma = 0.001

best_ocsvm = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
best_preds = best_ocsvm.fit_predict(X)
y_pred = (best_preds == -1).astype(int)
best_scores = -best_ocsvm.decision_function(X).ravel()

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, best_scores)
pr_auc = average_precision_score(y_true, best_scores)

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomali/Krisis'], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1-Score: {f1:.4f}")
"""))

# Cell 7: Confusion Matrix, ROC, PR Plots
cells.append(make_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred Normal', 'Pred Anomali'],
            yticklabels=['Actual Normal', 'Actual Krisis'])
axes[0].set_title('Confusion Matrix One-Class SVM', fontsize=12, fontweight='bold')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_true, best_scores)
axes[1].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'OC-SVM (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')

# 3. Precision-Recall Curve
prec_arr, rec_arr, _ = precision_recall_curve(y_true, best_scores)
axes[2].plot(rec_arr, prec_arr, color='#2ca02c', lw=2, label=f'OC-SVM (AP = {pr_auc:.4f})')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[2].legend(loc='lower left')

plt.tight_layout()
plt.show()
"""))

# Cell 8: Per-Crisis Detection Markdown
cells.append(make_markdown_cell("""## 3. 📉 Evaluasi Deteksi per-Krisis Historis

Pengujian daya deteksi model terhadap 6 periode krisis utama yang terdokumentasi dalam sejarah ekonomi global.
"""))

# Cell 9: Per-Crisis Detection Code
cells.append(make_code_cell("""GROUND_TRUTH = {
    'Krisis Asia (1997-98)': {'tahun': [1997, 1998], 'negara': ['IDN', 'THA', 'MYS', 'KOR', 'PHL']},
    'Krisis Rusia (1998)': {'tahun': [1998], 'negara': ['RUS']},
    'Krisis Argentina (2001-02)': {'tahun': [2001, 2002], 'negara': ['ARG']},
    'Global Financial Crisis (2008-09)': {'tahun': [2008, 2009], 'negara': list(df['economy'].unique())},
    'Krisis Utang Eropa (2010-12)': {'tahun': [2010, 2011, 2012], 'negara': ['GRC', 'PRT', 'IRL', 'ESP', 'ITA']},
    'Pandemi COVID-19 (2020)': {'tahun': [2020], 'negara': list(df['economy'].unique())}
}

df_eval = df[['economy', 'year', 'crisis_label']].copy()
df_eval['ocsvm_anomaly'] = y_pred

crisis_results = []
for nama, info in GROUND_TRUTH.items():
    mask = df['economy'].isin(info['negara']) & df['year'].isin(info['tahun'])
    crisis_obs = df_eval[mask]
    n_total = len(crisis_obs)
    n_detected = int(crisis_obs['ocsvm_anomaly'].sum())
    detection_rate = n_detected / n_total * 100 if n_total > 0 else 0
    crisis_results.append({
        'krisis': nama,
        'tahun': '-'.join(map(str, info['tahun'])),
        'total_observasi': n_total,
        'terdeteksi': n_detected,
        'detection_rate': round(detection_rate, 1)
    })

df_crisis = pd.DataFrame(crisis_results)
df_crisis.to_csv('deteksi_per_krisis_ocsvm.csv', index=False)

plt.figure(figsize=(10, 5))
barplot = sns.barplot(data=df_crisis, x='detection_rate', y='krisis', palette='viridis')
plt.title('Detection Rate per Krisis Historis — One-Class SVM', fontsize=13, fontweight='bold')
plt.xlabel('Detection Rate (%)')
plt.ylabel('')
for p in barplot.patches:
    width = p.get_width()
    plt.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%', ha='left', va='center', fontweight='bold')
plt.xlim(0, 110)
plt.tight_layout()
plt.show()

df_crisis
"""))

# Cell 10: Top Countries Markdown
cells.append(make_markdown_cell("""## 4. 🌍 Top 10 Negara dengan Frekuensi Anomali Terbanyak

Analisis profil geografis negara-negara yang paling sering diisolasi sebagai anomali oleh One-Class SVM.
"""))

# Cell 11: Top Countries Code
cells.append(make_code_cell("""top_countries = df_eval[df_eval['ocsvm_anomaly'] == 1].groupby('economy').size().sort_values(ascending=False).head(10)
df_top_countries = top_countries.reset_index()
df_top_countries.columns = ['economy', 'jumlah_anomali']
df_top_countries.to_csv('top_negara_anomali_ocsvm.csv', index=False)

plt.figure(figsize=(10, 4))
sns.barplot(data=df_top_countries, x='economy', y='jumlah_anomali', palette='Blues_r')
plt.title('Top 10 Negara dengan Frekuensi Anomali Terbanyak (OC-SVM)', fontsize=12, fontweight='bold')
plt.xlabel('Kode Negara')
plt.ylabel('Jumlah Tahun Anomali')
for idx, row in df_top_countries.iterrows():
    plt.text(idx, row['jumlah_anomali'] + 0.5, str(row['jumlah_anomali']), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Cell 12: Feature Importance Markdown
cells.append(make_markdown_cell("""## 5. 🔍 Feature Importance (Mean Difference Analysis)

Mengukur sensitivitas indikator makroekonomi dengan membandingkan nilai rata-rata sampel terdeteksi anomali versus sampel normal.
"""))

# Cell 13: Feature Importance Code
cells.append(make_code_cell("""df_fi = df[FEATURE_COLS].copy()
df_fi['anomaly'] = y_pred
normal_mean = df_fi[df_fi['anomaly'] == 0][FEATURE_COLS].mean()
anomaly_mean = df_fi[df_fi['anomaly'] == 1][FEATURE_COLS].mean()
mean_diff = (anomaly_mean - normal_mean).abs().sort_values(ascending=False)

fi_list = []
for rank, (feat, diff) in enumerate(mean_diff.items(), 1):
    fi_list.append({
        'rank': rank, 'feature': feat,
        'mean_diff': round(diff, 4),
        'mean_normal': round(normal_mean[feat], 4),
        'mean_anomali': round(anomaly_mean[feat], 4)
    })

df_fi_res = pd.DataFrame(fi_list)
df_fi_res.to_csv('feature_importance_ocsvm.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_fi_res, x='mean_diff', y='feature', palette='mako')
plt.title('Feature Importance One-Class SVM (Absolute Mean Difference)', fontsize=12, fontweight='bold')
plt.xlabel('Beda Rata-rata Z-Score (|Anomali - Normal|)')
plt.ylabel('Indikator Makroekonomi')
plt.tight_layout()
plt.show()

df_fi_res
"""))

# Cell 14: PCA & t-SNE Projections Markdown
cells.append(make_markdown_cell("""## 6. 🎨 Visualisasi Proyeksi Ruang Fitur 2D (PCA & t-SNE)

Visualisasi sebaran observasi normal vs anomali pada ruang dimensi tereduksi untuk memverifikasi pemisahan *boundary* OCSVM.
"""))

# Cell 15: PCA & t-SNE Projections Code
cells.append(make_code_cell("""pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
X_tsne = tsne.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA Plot
axes[0].scatter(X_pca[y_pred == 0, 0], X_pca[y_pred == 0, 1], c='#1f77b4', alpha=0.5, label='Normal', s=20)
axes[0].scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], c='#d62728', alpha=0.8, label='Anomali (OCSVM)', s=30, marker='x')
axes[0].set_title(f'Proyeksi PCA 2D (Explained Var: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()

# t-SNE Plot
axes[1].scatter(X_tsne[y_pred == 0, 0], X_tsne[y_pred == 0, 1], c='#1f77b4', alpha=0.5, label='Normal', s=20)
axes[1].scatter(X_tsne[y_pred == 1, 0], X_tsne[y_pred == 1, 1], c='#d62728', alpha=0.8, label='Anomali (OCSVM)', s=30, marker='x')
axes[1].set_title('Proyeksi t-SNE 2D', fontsize=12, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Cell 16: Export & Summary Markdown
cells.append(make_markdown_cell("""## 7. 💾 Menyimpan Hasil & Ringkasan Model

Menyimpan dataset akhir (`hasil_ocsvm.csv`) dan ringkasan metrik pemodelan (`ringkasan_model_ocsvm.csv`).
"""))

# Cell 17: Export Code
cells.append(make_code_cell("""df_output = df[['economy', 'year', 'crisis_label']].copy()
df_output['ocsvm_score'] = best_scores
df_output['ocsvm_anomaly'] = y_pred
for col in FEATURE_COLS:
    df_output[col] = df[col]
df_output.to_csv('hasil_ocsvm.csv', index=False)

summary = pd.DataFrame([{
    'model': 'One-Class SVM',
    'kernel': 'rbf',
    'nu': best_nu,
    'gamma': str(best_gamma),
    'n_anomalies': int(y_pred.sum()),
    'pct_anomalies': round(int(y_pred.sum()) / len(y_pred) * 100, 2),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'roc_auc': round(roc_auc, 4),
    'pr_auc': round(pr_auc, 4),
    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
}])
summary.to_csv('ringkasan_model_ocsvm.csv', index=False)
print("Seluruh file output telah berhasil disimpan di folder Virna/.")
summary
"""))

notebook_data = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

target_nb = os.path.join('Virna', 'ocsvm.ipynb')
with open(target_nb, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=1)

print(f"Jupyter Notebook Virna/ocsvm.ipynb successfully created!")
