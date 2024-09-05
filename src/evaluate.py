# evaluate.py

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import load_data

# Klasör yapısı
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')  # Bir üst dizine çık
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')  # Bir üst dizine çık

# Veri setinin ve modelin dosya yolları
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')

# Modeli değerlendir fonksiyonu
def evaluate_model():
    # Veriyi yükleyin
    df = load_data(PROCESSED_DATA_PATH)
    
    # Özellikler ve etiketler
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Eğitim ve test setlerine ayırın
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Özellikleri standartlaştırın
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modeli yükleyin
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model başarıyla yüklendi.")
    else:
        print(f"Model dosyası '{MODEL_PATH}' bulunamadı.")
        return

    # Tahmin yapın
    y_pred = model.predict(X_test)

    # Performans değerlendirmesi
    print("Karışıklık Matrisi:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
