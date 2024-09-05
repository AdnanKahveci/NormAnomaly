import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from utils import load_data

# Klasör yapısı
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')  # Bir üst dizine çık
MODEL_DIR = os.path.join(BASE_DIR, '..', 'results', 'models')  # Bir üst dizine çık
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'results', 'figures')  # Bir üst dizine çık
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Veri setinin ve modelin dosya yolları
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
print(PROCESSED_DATA_PATH)
# Grafik kaydetme fonksiyonları
def save_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc='lower right')

    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_roc_curve.png'))
    plt.close()

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

# Eğitim fonksiyonu
def train_model():
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

    # Scaler'ı kaydet
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(scaler, SCALER_PATH)

    # Random Forest modelini oluşturun ve eğitin
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model Performansı
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Eğitim Performansı:")
    print(classification_report(y_train, y_train_pred))

    print("Test Performansı:")
    print(classification_report(y_test, y_test_pred))

    # Modeli kaydedin
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model '{MODEL_PATH}' konumuna kaydedildi.")

    # ROC ve Confusion Matrix Kaydet
    save_roc_curve(y_test, model.predict_proba(X_test)[:, 1], 'RandomForest')
    save_confusion_matrix(y_test, y_test_pred, 'RandomForest')

if __name__ == "__main__":
    train_model()
