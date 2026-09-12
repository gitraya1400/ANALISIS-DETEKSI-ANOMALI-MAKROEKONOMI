"""
run_all.py
==========
Jalankan seluruh pipeline PCA-based Anomaly Detection dengan SATU KALI KLIK
(satu perintah). Urutan eksekusi:

    1. 01_pca_pipeline.py  -> fit scaler & PCA (data normal saja), scoring semua baris
    2. 02_evaluate.py      -> hitung metrik evaluasi & effect size
    3. 03_visualize.py     -> buat 4 figure (scree, control chart, contribution, effect size)

Cara pakai:
    cd scripts
    python run_all.py

Semua output akan otomatis tersimpan ke folder results/ dan figures/ di
root proyek (lihat config.py untuk detail path).
"""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [
    "01_pca_pipeline.py",
    "02_evaluate.py",
    "03_visualize.py",
]


def run_step(script_name):
    print("\n" + "=" * 70)
    print(f"MENJALANKAN: {script_name}")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=SCRIPTS_DIR,
    )
    if result.returncode != 0:
        print(f"[GAGAL] {script_name} berhenti dengan kode error {result.returncode}")
        sys.exit(result.returncode)


def main():
    print("PIPELINE PCA-BASED ANOMALY DETECTION - EWS KRISIS EKONOMI (KELOMPOK 4)")
    for step in PIPELINE:
        run_step(step)
    print("\n" + "=" * 70)
    print("SELESAI. Cek folder results/ dan figures/ untuk melihat seluruh output.")
    print("=" * 70)


if __name__ == "__main__":
    main()
