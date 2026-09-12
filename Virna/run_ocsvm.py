"""
Script to run One-Class SVM anomaly detection and produce all results for Virna's section in EWS project.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

SEED = 42
np.random.seed(SEED)

# Determine path to data_cleaned.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data_cleaned.csv')
if not os.path.exists(data_path):
    data_path = 'data_cleaned.csv'

# ── 1. Load Data ──────────────────────────────────────────────────────────
df = pd.read_csv(data_path)
print(f'Shape dataset: {df.shape}')
print(f'Jumlah negara: {df["economy"].nunique()}')
print(f'Rentang tahun: {df["year"].min()} - {df["year"].max()}')
print(f'Persentase krisis: {df["crisis_label"].mean()*100:.2f}%')

FEATURE_COLS = [
    'GDP_Growth', 'GDP_PerCapita_Growth', 'Inflation_CPI',
    'Total_Reserves', 'Unemployment', 'Current_Account_GDP',
    'Trade_GDP', 'FDI_Inflows_GDP', 'Exports_GDP', 'Imports_GDP',
    'Gross_Savings_GDP', 'Exchange_Rate', 'Manufacturing_Value',
    'Investment_GDP'
]

X = df[FEATURE_COLS].values
y_true = df['crisis_label'].values

# ── 2. Grid Search for OCSVM ──────────────────────────────────────────────
nu_values = [0.01, 0.05, 0.08, 0.10, 0.15, 0.1919, 0.25]
gamma_values = ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

results = []
for nu in nu_values:
    for gamma in gamma_values:
        ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        # OCSVM output: 1 for normal (inlier), -1 for anomaly (outlier)
        preds = ocsvm.fit_predict(X)
        y_pred = (preds == -1).astype(int)
        
        # Decision function: lower values mean more anomalous
        scores = -ocsvm.decision_function(X).ravel()
        
        n_anomalies = int(y_pred.sum())
        pct_anomalies = n_anomalies / len(y_pred) * 100
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        
        results.append({
            'nu': nu,
            'gamma': str(gamma),
            'n_anomalies': n_anomalies,
            'pct_anomalies': round(pct_anomalies, 2),
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1_score': round(f, 4),
            'roc_auc': round(roc, 4),
            'pr_auc': round(pr_auc, 4)
        })

df_results = pd.DataFrame(results).sort_values('f1_score', ascending=False)
grid_search_csv = os.path.join(script_dir, 'grid_search_ocsvm.csv')
df_results.to_csv(grid_search_csv, index=False)
print(f'\nGrid search: {len(df_results)} kombinasi dievaluasi')
print(df_results.head(10).to_string(index=False))

# ── 3. Best Model Selection ───────────────────────────────────────────────
best = df_results.iloc[0]
best_nu = best['nu']
best_gamma = best['gamma']
if best_gamma not in ['scale', 'auto']:
    best_gamma = float(best_gamma)

print(f'\n=== PARAMETER TERBAIK ===')
print(f'nu = {best_nu}, gamma = {best_gamma}')

best_ocsvm = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
best_preds = best_ocsvm.fit_predict(X)
y_pred = (best_preds == -1).astype(int)
best_scores = -best_ocsvm.decision_function(X).ravel()

n_anomalies = int(y_pred.sum())
print(f'Jumlah anomali terdeteksi: {n_anomalies} ({n_anomalies/len(y_pred)*100:.2f}%)')

# ── 4. Evaluation Metrics ────────────────────────────────────────────────
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, best_scores)
pr_auc = average_precision_score(y_true, best_scores)

print(f'\n=== CLASSIFICATION REPORT ===')
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomali/Krisis'], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f'Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')
print(f'ROC-AUC Score: {roc_auc:.4f}')
print(f'PR-AUC Score: {pr_auc:.4f}')

# ── 5. Per-Crisis Analysis ───────────────────────────────────────────────
GROUND_TRUTH = {
    'Krisis Asia 1997-1998': {'tahun': [1997, 1998], 'negara': ['IDN', 'THA', 'MYS', 'KOR', 'PHL']},
    'Krisis Rusia 1998': {'tahun': [1998], 'negara': ['RUS']},
    'Krisis Argentina 2001-2002': {'tahun': [2001, 2002], 'negara': ['ARG']},
    'Global Financial Crisis 2008-2009': {'tahun': [2008, 2009], 'negara': list(df['economy'].unique())},
    'Krisis Utang Eropa 2010-2012': {'tahun': [2010, 2011, 2012], 'negara': ['GRC', 'PRT', 'IRL', 'ESP', 'ITA']},
    'Pandemi COVID-19 2020': {'tahun': [2020], 'negara': list(df['economy'].unique())}
}

df_eval = df[['economy', 'year', 'crisis_label']].copy()
df_eval['ocsvm_anomaly'] = y_pred
df_eval['ocsvm_score'] = best_scores

print(f'\n=== DETEKSI PER KRISIS HISTORIS ===')
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

    print(f'{nama}: {n_detected}/{n_total} ({detection_rate:.1f}%)')

pd.DataFrame(crisis_results).to_csv(os.path.join(script_dir, 'deteksi_per_krisis_ocsvm.csv'), index=False)

# ── 6. Top Countries with Most Anomalies ─────────────────────────────────
top_countries = df_eval[df_eval['ocsvm_anomaly'] == 1].groupby('economy').size().sort_values(ascending=False).head(10)
print(f'\n=== TOP 10 NEGARA ANOMALI TERBANYAK ===')
top_countries_list = []
for rank, (country, count) in enumerate(top_countries.items(), 1):
    print(f'  {rank}. {country}: {count}')
    top_countries_list.append({'rank': rank, 'economy': country, 'jumlah_anomali': count})

pd.DataFrame(top_countries_list).to_csv(os.path.join(script_dir, 'top_negara_anomali_ocsvm.csv'), index=False)

# ── 7. Feature Importance (Mean Difference) ──────────────────────────────
df_fi = df[FEATURE_COLS].copy()
df_fi['anomaly'] = y_pred
normal_mean = df_fi[df_fi['anomaly'] == 0][FEATURE_COLS].mean()
anomaly_mean = df_fi[df_fi['anomaly'] == 1][FEATURE_COLS].mean()
mean_diff = (anomaly_mean - normal_mean).abs().sort_values(ascending=False)

print(f'\n=== FEATURE IMPORTANCE (Mean Difference) ===')
fi_list = []
for rank, (feat, diff) in enumerate(mean_diff.items(), 1):
    print(f'  {rank}. {feat}: {diff:.4f} (Normal={normal_mean[feat]:.4f}, Anomali={anomaly_mean[feat]:.4f})')
    fi_list.append({
        'rank': rank, 'feature': feat,
        'mean_diff': round(diff, 4),
        'mean_normal': round(normal_mean[feat], 4),
        'mean_anomali': round(anomaly_mean[feat], 4)
    })

pd.DataFrame(fi_list).to_csv(os.path.join(script_dir, 'feature_importance_ocsvm.csv'), index=False)

# ── 8. Save Full Results ─────────────────────────────────────────────────
df_output = df[['economy', 'year', 'crisis_label']].copy()
df_output['ocsvm_score'] = best_scores
df_output['ocsvm_anomaly'] = y_pred
for col in FEATURE_COLS:
    df_output[col] = df[col]
df_output.to_csv(os.path.join(script_dir, 'hasil_ocsvm.csv'), index=False)

# ── 9. Save Model Summary ───────────────────────────────────────────────
summary = pd.DataFrame([{
    'model': 'One-Class SVM',
    'kernel': 'rbf',
    'nu': best_nu,
    'gamma': str(best_gamma),
    'n_anomalies': n_anomalies,
    'pct_anomalies': round(n_anomalies / len(y_pred) * 100, 2),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'roc_auc': round(roc_auc, 4),
    'pr_auc': round(pr_auc, 4),
    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
}])
summary.to_csv(os.path.join(script_dir, 'ringkasan_model_ocsvm.csv'), index=False)

print(f'\n=== ALL FILES SAVED SUCCESSFULLY IN Virna/ ===')
print('grid_search_ocsvm.csv')
print('deteksi_per_krisis_ocsvm.csv')
print('top_negara_anomali_ocsvm.csv')
print('feature_importance_ocsvm.csv')
print('hasil_ocsvm.csv')
print('ringkasan_model_ocsvm.csv')
print('\nDone!')
