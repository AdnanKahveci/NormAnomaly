import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Dosya yolları
RAW_DATA_PATH = '../data/raw/creditcard.csv'  # Ham veri dosya yolu
PROCESSED_DATA_PATH = '../data/processed/processed_data.csv'  # İşlenmiş veri dosya yolu
MODEL_DIR = '../results/models'  # Modelin kaydedileceği dizin
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')  # Scaler kaydedileceği yol

def load_data(file_path):
    """
    Veriyi belirtilen dosya yolundan yükler ve bir DataFrame döndürür.
    """
    try:
        data = pd.read_csv(file_path)
        print("Veri başarıyla yüklendi.")
        return data
    except FileNotFoundError:
        print(f"{file_path} bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return None

def preprocess_data(df):
    """
    Veriyi işler, ölçekler ve X, y olarak ayırır.
    """
    # Zaman sütununu normalleştir
    df['Time'] = df['Time'].apply(lambda x: x / 3600)  # Saat cinsine çevir
    
    # İşlemleri sınıf dışındaki tüm sütunlar için uygula
    features = df.drop(['Class'], axis=1)
    labels = df['Class']
    
    # Standartlaştırma işlemi
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Scaler'ı kaydet
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(scaler, SCALER_PATH)
    
    # İşlenmiş veriyi dataframe'e çevir
    processed_df = pd.DataFrame(scaled_features, columns=features.columns)
    processed_df['Class'] = labels.values  # Etiketleri geri ekle
    
    return processed_df

def save_processed_data(df, processed_data_path):
    """
    İşlenmiş veriyi belirtilen yola kaydeder.
    """
    if not os.path.exists(os.path.dirname(processed_data_path)):
        os.makedirs(os.path.dirname(processed_data_path))
    df.to_csv(processed_data_path, index=False)
    print(f"İşlenmiş veri '{processed_data_path}' konumuna kaydedildi.")

def preprocess_pipeline(raw_data_path, processed_data_path):
    """
    Veri işleme hattı: Yükleme, işleme ve kaydetme.
    """
    # Veriyi yükle
    df = load_data(raw_data_path)
    
    if df is not None:
        # Veriyi işle
        processed_df = preprocess_data(df)
        
        # İşlenmiş veriyi kaydet
        save_processed_data(processed_df, processed_data_path)
        
        # Eğitim ve test setlerine ayır
        X = processed_df.drop('Class', axis=1)
        y = processed_df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Eğitim ve test verilerini kaydet
        pd.DataFrame(X_train).to_csv(os.path.join(os.path.dirname(processed_data_path), 'X_train.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(os.path.dirname(processed_data_path), 'X_test.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(os.path.dirname(processed_data_path), 'y_train.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(os.path.dirname(processed_data_path), 'y_test.csv'), index=False)
        
        return X_train, X_test, y_train, y_test
    else:
        print("Veri işlenemedi.")
        return None, None, None, None

if __name__ == "__main__":
    # Veri işleme işlemini çalıştır
    X_train, X_test, y_train, y_test = preprocess_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("Veri ön işleme tamamlandı ve işlendi.")
    print(f"Eğitim veri seti boyutu: {X_train.shape}, {y_train.shape}")