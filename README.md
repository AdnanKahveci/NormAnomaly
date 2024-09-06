# Credit Card Fraud Detection with Data Preprocessing Pipeline

Bu proje, kredi kartı dolandırıcılığını tespit etmek için bir veri işleme hattı geliştirmeyi amaçlamaktadır. Ham veri üzerinde çeşitli ön işleme adımları uygulanır ve model eğitimi için hazır hale getirilir.

## Proje Yapısı

```plaintext
project-root/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv   # Ham veri dosyası
│   ├── processed/           # İşlenmiş veri dosyalarının saklanacağı klasör
│
├── results/
│   └── models/              # Eğitim sırasında oluşturulan model ve scaler dosyalarının kaydedileceği klasör
│
├── src/
│   └── data_preprocessing.py # Veri işleme hattını tanımlayan Python betiği
│
├── README.md                # Proje hakkında bilgi içeren dosya
│
└── requirements.txt         # Gerekli Python kütüphanelerinin listesi
```

## Klasör ve Dosya Açıklamaları

- **`data/raw/`**: Ham veri dosyalarının saklandığı klasör. Örneğin, `creditcard.csv` dosyası burada bulunur.
- **`data/processed/`**: İşlenmiş veri dosyalarının saklanacağı klasör.
- **`results/models/`**: Eğitim sırasında oluşturulan model ve scaler dosyalarının kaydedileceği klasör.
- **`results/figures/`**: Model performans grafikleri gibi eğitim ve test süreçlerinde oluşturulan görsellerin kaydedileceği klasör.
- **`src/data_preprocessing.py`**: Veriyi işlemek için kullanılan Python betiği.
- **`src/train_model.py`**: Model eğitimi ve değerlendirmesi için kullanılan Python betiği.
- **`README.md`**: Proje hakkında genel bilgi, kullanım kılavuzu ve dokümantasyon içerir.
- **`requirements.txt`**: Projenin çalışması için gerekli olan Python kütüphanelerinin listesi.

## Kurulum ve Kullanım
Projeyi çalıştırmadan önce aşağıdaki bağımlılıkların kurulu olduğundan emin olun:

- Python 3.8+
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib

1. **Gerekli Kütüphaneleri Yükleyin**:
   ```bash
   pip install -r requirements.txt
2. **Ham Veri Dosyasını Yükleyin: data/raw/creditcard.csv dosyasının mevcut olduğundan emin olun. Bu dosya, ham kredi kartı işlemlerini içeren veri kümesini temsil eder**:

3. **Veri Ön İşleme: data_preprocessing.py betiğini çalıştırarak veriyi işleyin ve gerekli dosyaları oluşturun**:
   ```bash
    python src/data_preprocessing.py
4. **Modeli Eğitin: train_model.py betiğini çalıştırarak modeli eğitin ve sonuçları değerlendirin**:
   ```bash
   python src/train_model.py

## Veri Kümesi
Veri kümesi data/raw/creditcard.csv dosyasında bulunmalıdır. Veri seti, kredi kartı işlemlerini ve olası dolandırıcılık sınıflandırmalarını içerir

## Proje Klasör Yapısı

Yukarıdaki README dosyasında belirtilen klasör yapısını oluşturmak için terminal veya komut istemcisini kullanabilirsiniz. Aşağıda, bu klasör yapısını oluşturmak için gerekli komutlar verilmiştir:

```bash
mkdir -p project-root/data/raw
mkdir -p project-root/data/processed
mkdir -p project-root/results/models
mkdir -p project-root/scripts
touch project-root/data/raw/creditcard.csv
touch project-root/scripts/data_preprocessing.py
touch project-root/README.md
touch project-root/requirements.txt
```
## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.
