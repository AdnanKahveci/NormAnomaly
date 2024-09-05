import sys
import os
import pytest
import pandas as pd

# Proje kök dizinini Python yoluna ekleyin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import preprocess_pipeline

# Örnek veri seti için yol
TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data', 'processed_data.csv'))

@pytest.fixture
def setup_data():
    # Test verisini oluşturun
    df = pd.DataFrame({
        'V1': [0.1, 0.2],
        'V2': [0.2, 0.3],
        'Class': [0, 1]
    })
    df.to_csv(TEST_DATA_PATH, index=False)
    yield
    os.remove(TEST_DATA_PATH)

def test_preprocess_pipeline(setup_data):
    X_train, X_test, y_train, y_test = preprocess_pipeline(TEST_DATA_PATH)
    assert X_train.shape[0] > 0, "X_train verisi boş olmamalıdır"
    assert y_train.shape[0] > 0, "y_train verisi boş olmamalıdır"
    assert X_test.shape[0] > 0, "X_test verisi boş olmamalıdır"
    assert y_test.shape[0] > 0, "y_test verisi boş olmamalıdır"
