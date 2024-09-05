# tests/test_utils.py
import os
import pandas as pd
import sys

# Proje kök dizinini Python yoluna ekleyin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import load_data, load_processed_data

def test_load_data():
    test_csv_path = 'data/processed/processed_data.csv'
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df.to_csv(test_csv_path, index=False)
    
    loaded_df = load_data(test_csv_path)
    assert not loaded_df.empty, "Yüklenen veri boş olmamalıdır"
    assert set(df.columns) == set(loaded_df.columns), "Sütun isimleri eşleşmelidir"
    
def test_load_processed_data():
    # İşlenmiş veriler için örnek veri
    data_dir = 'tests/test_data/'
    X_train = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    X_test = pd.DataFrame({'feature1': [5, 6], 'feature2': [7, 8]})
    y_train = pd.DataFrame({'Class': [0, 1]})
    y_test = pd.DataFrame({'Class': [1, 0]})
    
    X_train.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)
    
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = load_processed_data(data_dir)
    
    assert not X_train_loaded.empty, "X_train verisi boş olmamalıdır"
    assert not X_test_loaded.empty, "X_test verisi boş olmamalıdır"
    assert not y_train_loaded.empty, "y_train verisi boş olmamalıdır"
    assert not y_test_loaded.empty, "y_test verisi boş olmamalıdır"
