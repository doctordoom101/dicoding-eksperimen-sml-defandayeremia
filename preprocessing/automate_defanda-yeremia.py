import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../credit_card_fraud_data_raw/card_transdata.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'fraud_detection_preprocessing')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'clean_card_transdata.csv')

def process_data():
    print("Memulai proses data cleaning...")

    # 1. Load Data
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Ukuran awal data: {df.shape}")

    # 2. Drop missing value
    df = df.dropna()
    print(f"Setelah drop NA: {df.shape}")

    # 3. Fix binary columns dtype
    binary_cols = [
        'repeat_retailer',
        'used_chip',
        'used_pin_number',
        'online_order',
        'fraud'
    ]

    df[binary_cols] = df[binary_cols].astype(int)

    # 4. Scaling numeric features (ROBUST)
    num_cols = [
        'distance_from_home',
        'distance_from_last_transaction',
        'ratio_to_median_purchase_price'
    ]

    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Save processed data & scaler
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "robust_scaler.joblib"))

    print(f"Data berhasil diproses.")
    print(f"- Data final: {OUTPUT_FILE}")
    print(f"- Scaler disimpan: robust_scaler.joblib")

if __name__ == "__main__":
    process_data()