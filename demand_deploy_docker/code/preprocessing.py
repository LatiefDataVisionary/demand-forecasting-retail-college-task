import pandas as pd
import numpy as np
import joblib
import json
import os

def load_preprocessing_assets(assets_path):
    scaler = joblib.load(os.path.join(assets_path, 'scaler.pkl'))
    with open(os.path.join(assets_path, 'capping_params.json'), 'r') as f:
        capping_params = json.load(f)
    with open(os.path.join(assets_path, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)
    print("Preprocessing assets loaded successfully.")
    return scaler, capping_params, feature_columns

def apply_preprocessing(df, preprocessing_assets):
    scaler, capping_params, feature_columns = preprocessing_assets

    processed_df = df.copy()

    if 'Date' not in processed_df.columns:
        raise KeyError("Input data must contain the 'Date' column for temporal feature extraction.")

    print("\n--- 1. PREPROCESSING STAGE: FEATURE ENGINEERING ---")

    # DAFTAR KOLOM NUMERIK MENTAH (yang harus diubah dari 'object' ke 'float')
    numerical_raw_cols = ['Units Sold', 'Price', 'Inventory Level', 'Units Ordered', 'Competitor Pricing', 'RETAIL SALES', 'price', 'discount_percentage', 'sales_revenue', 
                          'Promotion', 'Epidemic', 'holiday_season', 'promotion_applied', 'competitor_price_index', 'economic_index', 'weather_impact', 'Discount']
    
    # Konversi eksplisit kolom numerik (karena di app.py kita paksa jadi 'object')
    for col in numerical_raw_cols:
        if col in processed_df.columns:
            # Gunakan to_numeric dengan errors='coerce' untuk keamanan konversi dari string/object
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype(float)
            
    print("   -> Kolom numerik mentah dikonversi dari object ke float/numerik.")
    
    # Konversi Tipe Data
    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    processed_df['Product ID'] = processed_df['Product ID'].astype(str)
    if 'Store ID' in processed_df.columns: processed_df['Store ID'] = processed_df['Store ID'].astype(str)
    print("   -> Tipe data kategorikal dan Date dikonversi.")

    # Ekstraksi Fitur Temporal
    processed_df['year'] = processed_df['Date'].dt.year
    processed_df['month'] = processed_df['Date'].dt.month
    processed_df['day'] = processed_df['Date'].dt.day
    processed_df['dayofweek'] = processed_df['Date'].dt.dayofweek
    processed_df['dayofyear'] = processed_df['Date'].dt.dayofyear
    # Pastikan weekofyear menggunakan isocalendar yang aman di Pandas > 2.0
    processed_df['weekofyear'] = processed_df['Date'].dt.isocalendar().week.astype(int) 
    processed_df['quarter'] = processed_df['Date'].dt.quarter
    print("   -> 7 Fitur Temporal diekstrak.")

    # Inisiasi Lag Features (Jika tidak ada di input)
    if 'demand_lag_7' not in processed_df.columns: processed_df['demand_lag_7'] = 0 
    if 'demand_lag_28' not in processed_df.columns: processed_df['demand_lag_28'] = 0 
    if 'demand_rolling_mean_7' not in processed_df.columns: processed_df['demand_rolling_mean_7'] = 0
    if 'demand_rolling_mean_28' not in processed_df.columns: processed_df['demand_rolling_mean_28'] = 0

    
    # --- 2. Capping ---
    print("\n--- 2. PREPROCESSING STAGE: OUTLIER HANDLING (CAPPING) ---")
    
    for col, params in capping_params.items():
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].clip(lower=params['lower_bound'], upper=params['upper_bound'])
            processed_df[f'{col}_capped'] = processed_df[col] 
            
    print(f"   -> Capping diterapkan pada {len(capping_params)} kolom numerik.")

    # --- 3. Encoding dan Penyesuaian Kolom ---
    print("\n--- 3. PREPROCESSING STAGE: ENCODING & SCALING ---")
    
    # Pastikan kolom boolean diubah ke integer (seperti Epidemic, Promotion, dll)
    for col in ['Promotion', 'Epidemic', 'holiday_season', 'promotion_applied', 'weather_impact']:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(int)
    
    categorical_features_for_encoding = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality', 'SUPPLIER', 'ITEM TYPE']
    processed_df = pd.get_dummies(processed_df, columns=categorical_features_for_encoding, dummy_na=False)
    print(f"   -> One-Hot Encoding diterapkan pada {len(categorical_features_for_encoding)} fitur kategorikal.")

    # Menyesuaikan kolom agar sesuai dengan feature_columns.json (penting untuk model)
    missing_cols = set(feature_columns) - set(processed_df.columns)
    for c in missing_cols:
        processed_df[c] = 0

    extra_cols = set(processed_df.columns) - set(feature_columns)
    processed_df = processed_df.drop(columns=list(extra_cols))
    
    # Scaling (hanya kolom yang ada di processed_df dan feature_columns)
    
    # Ambil hanya kolom yang benar-benar digunakan model
    processed_df = processed_df[feature_columns]

    numerical_features_to_scale_for_deployment = scaler.feature_names_in_.tolist()
    
    cols_to_transform = [col for col in numerical_features_to_scale_for_deployment if col in processed_df.columns]
    
    processed_df[cols_to_transform] = scaler.transform(processed_df[cols_to_transform])
    print(f"   -> {len(cols_to_transform)} Fitur numerik di-Scaling menggunakan StandardScaler.")

    # Hapus kolom Date (yang tidak lagi diperlukan setelah fitur temporal diekstrak)
    if 'Date' in processed_df.columns: 
        processed_df = processed_df.drop(columns=['Date'])

    print("\nPreprocessing berhasil diterapkan dan data siap untuk prediksi.")
    return processed_df