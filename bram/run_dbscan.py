"""
Script to run DBSCAN anomaly detection and produce all results for the report.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SEED = 42
np.random.seed(SEED)

# ── 1. Load Data ──────────────────────────────────────────────────────────
df = pd.read_csv('../data_cleaned.csv')
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

# ── 2. Grid Search ────────────────────────────────────────────────────────
eps_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
min_samples_values = [3, 5, 7, 10, 15]

results = []
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        labels = dbscan.fit_predict(X)
        y_pred = (labels == -1).astype(int)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        pct_noise = n_noise / len(labels) * 100
        if n_noise == 0 or n_noise == len(labels):
            continue
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        results.append({
            'eps': eps, 'min_samples': min_samples,
            'n_clusters': n_clusters, 'n_noise': n_noise,
            'pct_noise': pct_noise, 'precision': p, 'recall': r, 'f1_score': f
        })

df_results = pd.DataFrame(results).sort_values('f1_score', ascending=False)
df_results.to_csv('grid_search_dbscan.csv', index=False)
print(f'\nGrid search: {len(df_results)} kombinasi valid')
print(df_results.head(10).to_string(index=False))

# ── 3. Best Model ─────────────────────────────────────────────────────────
best = df_results.iloc[0]
best_eps = best['eps']
best_min_samples = int(best['min_samples'])
print(f'\n=== PARAMETER TERBAIK ===')
print(f'eps = {best_eps}, min_samples = {best_min_samples}')

best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean', n_jobs=-1)
best_labels = best_dbscan.fit_predict(X)
y_pred = (best_labels == -1).astype(int)

n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
n_noise = (best_labels == -1).sum()

print(f'Jumlah cluster: {n_clusters}')
print(f'Jumlah noise/anomali: {n_noise} ({n_noise/len(best_labels)*100:.1f}%)')

# ── 4. Evaluation ─────────────────────────────────────────────────────────
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f'\n=== CLASSIFICATION REPORT ===')
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomali/Krisis'], zero_division=0))

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f'Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')

# ── 5. Cluster Distribution ──────────────────────────────────────────────
print(f'\n=== DISTRIBUSI CLUSTER ===')
cluster_dist = []
for label in sorted(set(best_labels)):
    count = (best_labels == label).sum()
    pct = count / len(best_labels) * 100
    label_name = 'Noise/Anomali' if label == -1 else f'Cluster {label}'
    print(f'  {label_name}: {count} ({pct:.1f}%)')
    cluster_dist.append({'cluster': label, 'label_name': label_name, 'count': count, 'pct': round(pct, 1)})

pd.DataFrame(cluster_dist).to_csv('distribusi_cluster_dbscan.csv', index=False)

# ── 6. Per-Crisis Analysis ───────────────────────────────────────────────
GROUND_TRUTH = {
    'Krisis Asia 1997-1998': {'tahun': [1997, 1998], 'negara': ['IDN', 'THA', 'MYS', 'KOR', 'PHL']},
    'Krisis Rusia 1998': {'tahun': [1998], 'negara': ['RUS']},
    'Krisis Argentina 2001-2002': {'tahun': [2001, 2002], 'negara': ['ARG']},
    'Global Financial Crisis 2008-2009': {'tahun': [2008, 2009], 'negara': list(df['economy'].unique())},
    'Krisis Utang Eropa 2010-2012': {'tahun': [2010, 2011, 2012], 'negara': ['GRC', 'PRT', 'IRL', 'ESP', 'ITA']},
    'Pandemi COVID-19 2020': {'tahun': [2020], 'negara': list(df['economy'].unique())}
}

df_eval = df[['economy', 'year', 'crisis_label']].copy()
df_eval['dbscan_anomaly'] = y_pred

print(f'\n=== DETEKSI PER KRISIS HISTORIS ===')
crisis_results = []
for nama, info in GROUND_TRUTH.items():
    mask = df['economy'].isin(info['negara']) & df['year'].isin(info['tahun'])
    crisis_obs = df_eval[mask]
    n_total = len(crisis_obs)
    n_detected = int(crisis_obs['dbscan_anomaly'].sum())
    detection_rate = n_detected / n_total * 100 if n_total > 0 else 0

    crisis_results.append({
        'krisis': nama,
        'tahun': '-'.join(map(str, info['tahun'])),
        'total_observasi': n_total,
        'terdeteksi': n_detected,
        'detection_rate': round(detection_rate, 1)
    })

    print(f'{nama}: {n_detected}/{n_total} ({detection_rate:.1f}%)')
    if n_detected > 0:
        detected = crisis_obs[crisis_obs['dbscan_anomaly'] == 1][['economy', 'year']]
        for _, row in detected.iterrows():
            print(f'    - {row["economy"]} ({int(row["year"])})')

pd.DataFrame(crisis_results).to_csv('deteksi_per_krisis_dbscan.csv', index=False)

# ── 7. Top Countries with Most Anomalies ─────────────────────────────────
top_countries = df_eval[df_eval['dbscan_anomaly'] == 1].groupby('economy').size().sort_values(ascending=False).head(10)
print(f'\n=== TOP 10 NEGARA ANOMALI TERBANYAK ===')
top_countries_list = []
for rank, (country, count) in enumerate(top_countries.items(), 1):
    print(f'  {rank}. {country}: {count}')
    top_countries_list.append({'rank': rank, 'economy': country, 'jumlah_anomali': count})

pd.DataFrame(top_countries_list).to_csv('top_negara_anomali_dbscan.csv', index=False)

# ── 8. Feature Importance ────────────────────────────────────────────────
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

pd.DataFrame(fi_list).to_csv('feature_importance_dbscan.csv', index=False)

# ── 9. Save Full Results ─────────────────────────────────────────────────
df_output = df[['economy', 'year', 'crisis_label']].copy()
df_output['dbscan_anomaly'] = y_pred
df_output['dbscan_cluster'] = best_labels
for col in FEATURE_COLS:
    df_output[col] = df[col]
df_output.to_csv('hasil_dbscan.csv', index=False)

# ── 10. Save Model Summary ───────────────────────────────────────────────
summary = pd.DataFrame([{
    'model': 'DBSCAN',
    'eps': best_eps,
    'min_samples': best_min_samples,
    'n_clusters': n_clusters,
    'n_anomalies': n_noise,
    'pct_anomalies': round(n_noise / len(y_pred) * 100, 2),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
}])
summary.to_csv('ringkasan_model_dbscan.csv', index=False)

print(f'\n=== FILES SAVED ===')
print('grid_search_dbscan.csv')
print('distribusi_cluster_dbscan.csv')
print('deteksi_per_krisis_dbscan.csv')
print('top_negara_anomali_dbscan.csv')
print('feature_importance_dbscan.csv')
print('hasil_dbscan.csv')
print('ringkasan_model_dbscan.csv')
print('\nDone!')
