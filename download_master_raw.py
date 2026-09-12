import sys
import time
import pandas as pd
import wbgapi as wb

# 1. Daftar 49 Negara Resmi (Sesuai Dokumen Kelompok)
countries = [
    # G7
    'USA', 'GBR', 'DEU', 'FRA', 'ITA', 'JPN', 'CAN',
    # BRICS
    'BRA', 'RUS', 'IND', 'CHN', 'ZAF',
    # ASEAN
    'IDN', 'THA', 'MYS', 'PHL', 'VNM', 'SGP',
    # Asia Lainnya
    'KOR', 'PAK', 'BGD', 'LKA',
    # Amerika Latin
    'MEX', 'ARG', 'COL', 'CHL', 'PER',
    # Timur Tengah & Afrika
    'SAU', 'TUR', 'EGY', 'NGA', 'KEN', 'ARE',
    # Eropa Non-G7
    'ESP', 'NLD', 'SWE', 'NOR', 'POL', 'GRC', 'PRT',
    'IRL', 'CHE', 'AUT', 'BEL', 'CZE', 'HUN', 'ROU',
    # Oseania
    'AUS', 'NZL'
]

# 2. Periode Observasi (1990 - 2024 = 35 Tahun)
years = list(range(1990, 2025))

# 3. Indikator Resmi World Bank (Bukan Dummy, Langsung via API Resmi)
indicators = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
    'FP.CPI.TOTL.ZG': 'Inflation_CPI',
    'SL.UEM.TOTL.ZS': 'Unemployment',
    'BN.CAB.XOKA.GD.ZS': 'Current_Account_GDP',
    'FI.RES.TOTL.MO': 'Reserves_Months_Imports',
    'PA.NUS.FCRF': 'Exchange_Rate',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_Inflows_GDP',
    'NE.EXP.GNFS.ZS': 'Exports_GDP',
    'NE.IMP.GNFS.ZS': 'Imports_GDP',
    'NY.GNS.ICTR.ZS': 'Gross_Savings_GDP',
    'NE.GDI.TOTL.ZS': 'Investment_GDP',
    'NV.IND.MANF.ZS': 'Manufacturing_Value',
    'FS.AST.PRVT.GD.ZS': 'Domestic_Credit_GDP',
    'FM.LBL.BMNY.ZG': 'Broad_Money_Growth'
}

print(f"Total Negara   : {len(countries)}")
print(f"Total Periode  : {min(years)} - {max(years)} ({len(years)} tahun)")
print(f"Total Indikator: {len(indicators)}")
print(f"Target Baris   : {len(countries) * len(years):,} baris")
print("=" * 60)
print("Mengunduh data riil langsung dari World Bank Open Data API...")
print("=" * 60)

all_data = []

for idx, (ind_code, ind_name) in enumerate(indicators.items(), 1):
    print(f"[{idx:2d}/{len(indicators)}] Mengambil {ind_name:.<26s} ({ind_code})...", end=" ", flush=True)
    success = False
    retries = 3
    while retries > 0 and not success:
        try:
            df_temp = wb.data.DataFrame(
                ind_code,
                economy=countries,
                time=years,
                numericTimeKeys=True
            )
            # Reshape wide ke long format (economy, year, value)
            df_temp = df_temp.reset_index()
            id_col = df_temp.columns[0]
            df_temp = df_temp.melt(id_vars=[id_col], var_name='year', value_name=ind_name)
            df_temp = df_temp.rename(columns={id_col: 'economy'})
            df_temp['year'] = df_temp['year'].astype(int)
            
            non_null = df_temp[ind_name].notna().sum()
            total_pts = len(df_temp)
            pct = (non_null / total_pts) * 100
            print(f"BERHASIL ({non_null:,}/{total_pts:,} terisi, {pct:.1f}%)")
            all_data.append(df_temp)
            success = True
        except Exception as e:
            retries -= 1
            print(f"\n   [Warning] Percobaan gagal: {e}. Mengulang (sisa {retries})...", flush=True)
            time.sleep(3)
    
    if not success:
        print(f"GAGAL setelah 3 percobaan.")

if not all_data:
    print("Error: Tidak ada data yang berhasil ditarik!")
    sys.exit(1)

# Menggabungkan seluruh indikator berdasarkan key (economy, year)
print("\n" + "=" * 60)
print("Menggabungkan seluruh indikator menjadi master table...")
df_master = all_data[0]
for d in all_data[1:]:
    df_master = df_master.merge(d, on=['economy', 'year'], how='outer')

# Mengurutkan berdasarkan negara dan tahun
df_master = df_master.sort_values(['economy', 'year']).reset_index(drop=True)

# Simpan ke CSV mentah murni (tanpa scaling, tanpa imputasi, tanpa dummy)
output_filename = 'raw_data_master.csv'
df_master.to_csv(output_filename, index=False)

print("=" * 60)
print(f"RAW DATA MASTER BERHASIL DIBUAT DAN DISIMPAN!")
print(f"File Output : {output_filename}")
print(f"Ukuran Data : {df_master.shape[0]:,} baris x {df_master.shape[1]} kolom")
print(f"Negara Unik : {df_master['economy'].nunique()} negara")
print(f"Rentang Thn : {df_master['year'].min()} - {df_master['year'].max()}")
print("\nRingkasan Missing Value Mentah per Indikator:")
print(df_master.isnull().sum())
print("\n5 Baris Pertama Data Mentah:")
print(df_master.head(5))
