# utils.py

import os
import torch
import pandas as pd

def save_model(model, path):
    """Modeli belirtilen yola kaydedin."""
    torch.save(model.state_dict(), path)
    print(f"Model '{path}' konumuna kaydedildi.")

def load_data(file_path):
    """
    Veriyi yükleyip döndüren yardımcı fonksiyon.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} bulunamadı.")
    return pd.read_csv(file_path)

def load_processed_data(data_dir):
    """İşlenmiş verileri yükleyin."""
    X_train_path = os.path.join(data_dir, 'X_train.csv')
    X_test_path = os.path.join(data_dir, 'X_test.csv')
    y_train_path = os.path.join(data_dir, 'y_train.csv')
    y_test_path = os.path.join(data_dir, 'y_test.csv')
    
    # Dosyaların mevcut olup olmadığını kontrol edin
    if not os.path.isfile(X_train_path):
        raise FileNotFoundError(f"{X_train_path} bulunamadı.")
    if not os.path.isfile(X_test_path):
        raise FileNotFoundError(f"{X_test_path} bulunamadı.")
    if not os.path.isfile(y_train_path):
        raise FileNotFoundError(f"{y_train_path} bulunamadı.")
    if not os.path.isfile(y_test_path):
        raise FileNotFoundError(f"{y_test_path} bulunamadı.")
    
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    
    return X_train, X_test, y_train, y_test
