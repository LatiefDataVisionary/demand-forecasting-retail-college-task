import joblib
import pandas as pd
from flask import Flask, request, jsonify
import os
import numpy as np # PENTING: Menambahkan import numpy

# Import preprocessing functions
from preprocessing import load_preprocessing_assets, apply_preprocessing

# --- Configuration ---
ASSETS_PATH = '../assets'

# --- Load Model and Preprocessing Assets ---
model = joblib.load(os.path.join(ASSETS_PATH, 'model.pkl'))
scaler, capping_params, feature_columns = load_preprocessing_assets(ASSETS_PATH)

# --- Flask App Setup ---
app = Flask(__name__)

# Fungsi untuk memberikan saran (Simulasi logika bisnis)
def generate_recommendations(predictions):
    avg_demand = np.mean(predictions)
    
    if avg_demand >= 150:
        recommendation = "🟢 **TINGKAT PERMINTAAN TINGGI (HIGH DEMAND)**: Direkomendasikan untuk segera meningkatkan level inventaris sebesar **15%** dan meluncurkan kampanye *pre-order* untuk memaksimalkan keuntungan."
    elif avg_demand >= 100:
        recommendation = "🟡 **TINGKAT PERMINTAAN SEDANG (MODERATE DEMAND)**: Pertahankan inventaris saat ini dan monitor tren. Pertimbangkan promosi minor untuk meningkatkan volume penjualan."
    else:
        recommendation = "🔴 **TINGKAT PERMINTAAN RENDAH (LOW DEMAND)**: Kurangi inventaris baru dan fokus pada membersihkan stok lama melalui penawaran diskon yang menarik."
        
    summary = f"\n[SUMMARY KEPUTUSAN BISNIS]: Rata-rata Prediksi Permintaan: {avg_demand:.2f} unit."
    
    return summary, recommendation

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json(force=True)
        
        # LOGIKA PEMBENTUKAN DATAFRAME PALING AMAN:
        # Menggunakan json_normalize dan memaksa semua kolom menjadi tipe 'object' (string)
        # untuk mencegah konflik DType DateTime64DType dan Float64DType saat inisiasi Pandas.
        if isinstance(json_data, list) and json_data:
            input_df = pd.json_normalize(json_data).astype('object')
        else:
            input_df = pd.json_normalize([json_data]).astype('object')

        # --- TAHAP 1: PREPROCESSING ---
        print("\n\n#####################################################")
        print("#               [STARTING PREDICTION]               #")
        print("#####################################################")
        
        # Output detail preprocessing akan muncul dari preprocessing.py
        processed_input_df = apply_preprocessing(input_df, (scaler, capping_params, feature_columns))

        # --- TAHAP 2: PREDIKSI ---
        predictions = model.predict(processed_input_df)
        
        # --- TAHAP 3: HASIL DAN SARAN (OUTPUT KEREN) ---
        summary, recommendation = generate_recommendations(predictions)
        
        print("\n\n=======================================================")
        print(f"  ✨ HASIL PREDIKSI PERMINTAAN UNTUK {len(predictions)} DATA POINT ✨")
        print("=======================================================")
        print(f"  * Total Data Input (setelah Preprocessing): {processed_input_df.shape[0]} baris, {processed_input_df.shape[1]} fitur")
        print(f"  * Model Digunakan: LightGBM Regressor")
        print("-------------------------------------------------------")
        print(summary)
        print(recommendation)
        print("=======================================================\n")

        # Mengembalikan prediksi ke user API
        return jsonify(predictions.tolist())

    except Exception as e:
        app.logger.error(f"Prediction Error: {e}") 
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)