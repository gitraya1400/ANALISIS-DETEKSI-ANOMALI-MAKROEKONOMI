"""
Script to run full preprocessing from raw_data_master.csv and ground_truth_imf.csv
all the way to One-Class SVM anomaly detection and result generation for Virna.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

SEED = 42
np.random.seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, 'raw_data_master.csv')
gt_path = os.path.join(script_dir, 'ground_truth_imf.csv')

# ── 1. Load Raw Data & Ground Truth ───────────────────────────────────────
raw_df = pd.read_csv(raw_path)
gt_df = pd.read_csv(gt_path)

df = pd.merge(raw_df, gt_df[['economy', 'year', 'is_crisis', 'crisis_name']], on=['economy', 'year'], how='left')
df = df.rename(columns={'is_crisis': 'crisis_label'})
df['crisis_label'] = df['crisis_label'].fillna(0).astype(int)

print(f"Dataset dimuat: {df.shape[0]} observasi, {df['economy'].nunique()} negara (1990-2024)")
print(f"Prevalensi crisis_label: {df['crisis_label'].mean()*100:.2f}% ({df['crisis_label'].sum()} observasi krisis)")

# ── 2. Sorting & Exchange Depreciation Transformation ──────────────────────
df = df.sort_values(['economy', 'year']).reset_index(drop=True)
df['Exchange_Depreciation'] = df.groupby('economy')['Exchange_Rate'].pct_change() * 100
df = df.drop(columns=['Exchange_Rate'])

feature_cols = [
    'GDP_Growth', 'Inflation_CPI', 'Unemployment', 'Current_Account_GDP',
    'Reserves_Months_Imports', 'Exchange_Depreciation', 'FDI_Inflows_GDP',
    'Exports_GDP', 'Imports_GDP', 'Gross_Savings_GDP', 'Investment_GDP',
    'Manufacturing_Value', 'Domestic_Credit_GDP', 'Broad_Money_Growth'
]

# ── 3. Missing Value Imputation Pipeline ──────────────────────────────────
# Linear interpolation per country
for col in feature_cols:
    df[col] = df.groupby('economy')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

# Fill remaining boundary NaNs using country median
for col in feature_cols:
    df[col] = df.groupby('economy')[col].transform(lambda x: x.fillna(x.median()))

# Global KNNImputer if any NaN remains
if df[feature_cols].isnull().sum().sum() > 0:
    imputer = KNNImputer(n_neighbors=5)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

print(f"Imputasi selesai. Sisa Missing Value: {df[feature_cols].isnull().sum().sum()}")

# Save preprocessed clean dataset (unscaled for inspection)
df_clean_unscaled = df.copy()
df_clean_unscaled.to_csv(os.path.join(script_dir, 'data_cleaned_virna.csv'), index=False)

# ── 4. Scaling (StandardScaler) ───────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
y_true = df['crisis_label'].values

# ── 5. Grid Search One-Class SVM ───────────────────────────────────────────
nu_values = [0.01, 0.05, 0.08, 0.10, 0.1335, 0.15, 0.20, 0.25]
gamma_values = ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

results = []
for nu in nu_values:
    for gamma in gamma_values:
        ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        preds = ocsvm.fit_predict(X_scaled)
        y_pred = (preds == -1).astype(int)
        scores = -ocsvm.decision_function(X_scaled).ravel()

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
df_results.to_csv(os.path.join(script_dir, 'grid_search_ocsvm.csv'), index=False)
print(f"\nGrid Search: {len(df_results)} kombinasi dievaluasi")
print(df_results.head(10).to_string(index=False))

# ── 6. Select Best Model & Evaluate ───────────────────────────────────────
best = df_results.iloc[0]
best_nu = best['nu']
best_gamma = best['gamma']
if best_gamma not in ['scale', 'auto']:
    best_gamma = float(best_gamma)

print(f"\n=== PARAMETER TERBAIK ===")
print(f"nu = {best_nu}, gamma = {best_gamma}")

best_ocsvm = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
best_preds = best_ocsvm.fit_predict(X_scaled)
y_pred = (best_preds == -1).astype(int)
best_scores = -best_ocsvm.decision_function(X_scaled).ravel()

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, best_scores)
pr_auc = average_precision_score(y_true, best_scores)

print(f"\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomali/Krisis'], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"ROC-AUC Score: {roc_auc:.4f} | PR-AUC Score: {pr_auc:.4f}")

# ── 7. Per-Crisis Detection Analysis ─────────────────────────────────────
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

print(f"\n=== DETEKSI PER KRISIS HISTORIS ===")
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

    print(f"{nama}: {n_detected}/{n_total} ({detection_rate:.1f}%)")

pd.DataFrame(crisis_results).to_csv(os.path.join(script_dir, 'deteksi_per_krisis_ocsvm.csv'), index=False)

# ── 8. Top Countries with Most Anomalies ─────────────────────────────────
top_countries = df_eval[df_eval['ocsvm_anomaly'] == 1].groupby('economy').size().sort_values(ascending=False).head(10)
print(f"\n=== TOP 10 NEGARA ANOMALI TERBANYAK ===")
top_countries_list = []
for rank, (country, count) in enumerate(top_countries.items(), 1):
    print(f"  {rank}. {country}: {count}")
    top_countries_list.append({'rank': rank, 'economy': country, 'jumlah_anomali': count})

pd.DataFrame(top_countries_list).to_csv(os.path.join(script_dir, 'top_negara_anomali_ocsvm.csv'), index=False)

# ── 9. Feature Importance (Mean Difference on Scaled Data) ──────────────
df_fi = pd.DataFrame(X_scaled, columns=feature_cols)
df_fi['anomaly'] = y_pred
normal_mean = df_fi[df_fi['anomaly'] == 0][feature_cols].mean()
anomaly_mean = df_fi[df_fi['anomaly'] == 1][feature_cols].mean()
mean_diff = (anomaly_mean - normal_mean).abs().sort_values(ascending=False)

print(f"\n=== FEATURE IMPORTANCE (Mean Difference Z-Score) ===")
fi_list = []
for rank, (feat, diff) in enumerate(mean_diff.items(), 1):
    print(f"  {rank}. {feat}: {diff:.4f} (Normal={normal_mean[feat]:.4f}, Anomali={anomaly_mean[feat]:.4f})")
    fi_list.append({
        'rank': rank, 'feature': feat,
        'mean_diff': round(diff, 4),
        'mean_normal': round(normal_mean[feat], 4),
        'mean_anomali': round(anomaly_mean[feat], 4)
    })

pd.DataFrame(fi_list).to_csv(os.path.join(script_dir, 'feature_importance_ocsvm.csv'), index=False)

# ── 10. Save Full Dataset Results ────────────────────────────────────────
df_output = df[['economy', 'year', 'crisis_label']].copy()
df_output['ocsvm_score'] = best_scores
df_output['ocsvm_anomaly'] = y_pred
for idx, col in enumerate(feature_cols):
    df_output[col] = X_scaled[:, idx]

df_output.to_csv(os.path.join(script_dir, 'hasil_ocsvm.csv'), index=False)

# ── 11. Save Model Summary ───────────────────────────────────────────────
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
summary.to_csv(os.path.join(script_dir, 'ringkasan_model_ocsvm.csv'), index=False)

print(f"\n=== SELURUH FILE BERHASIL DIPERBARUI DI Virna/ ===")
print("data_cleaned_virna.csv")
print("grid_search_ocsvm.csv")
print("deteksi_per_krisis_ocsvm.csv")
print("top_negara_anomali_ocsvm.csv")
print("feature_importance_ocsvm.csv")
print("hasil_ocsvm.csv")
print("ringkasan_model_ocsvm.csv")
print("\nDone!")
