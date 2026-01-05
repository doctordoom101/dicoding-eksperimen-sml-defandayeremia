import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../credit_card_fraud_data_raw/card_transdata.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'fraud_detection_preprocessing')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'clean_card_transdata.csv')

def process_data():
    print("Memulai proses data cleaning...")
    
    # 2. Load Data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 3. Preprocessing 
    df.dropna(inplace=True)
    
    print(f"Data diproses. Ukuran akhir: {df.shape}")
    
    # 4. Simpan Data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data berhasil disimpan di: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()