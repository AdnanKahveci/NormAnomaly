import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class CreditCardDataset(Dataset):
    """
    Kredi kartı dolandırıcılık veri seti için PyTorch veri seti sınıfı.
    """

    def __init__(self, data_dir, train=True, transform=None):
        """
        Veri setini yükleyin ve işleyin.
        
        Args:
            data_dir (str): İşlenmiş verilerin bulunduğu dizin.
            train (bool): Eğitim veya test veri seti olup olmadığını belirleyin.
            transform (callable, optional): Dönüştürme fonksiyonu.
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)  # Dosya yolunu dinamik oluşturma
        self.transform = transform
        self.train = train
        
        # Eğitim veya test setine göre dosyaları yükle
        if self.train:
            self.X = pd.read_csv(os.path.join(self.data_dir, 'processed/X_train.csv'))
            self.y = pd.read_csv(os.path.join(self.data_dir, 'processed/y_train.csv'))
        else:
            self.X = pd.read_csv(os.path.join(self.data_dir, 'processed/X_test.csv'))
            self.y = pd.read_csv(os.path.join(self.data_dir, 'processed/y_test.csv'))
        
        # Pandas DataFrame'i PyTorch Tensörüne dönüştür
        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.y = torch.tensor(self.y.values.squeeze(), dtype=torch.float32)

    def __len__(self):
        """Veri setinin uzunluğunu döndürür."""
        return len(self.y)

    def __getitem__(self, idx):
        """Bir veri örneği döndürür."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'features': self.X[idx], 'label': self.y[idx]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def get_dataloader(data_dir, batch_size=32, train=True):
    """
    PyTorch DataLoader nesnesini döndürür.
    
    Args:
        data_dir (str): Veri seti dizini.
        batch_size (int): Her bir batch için örnek sayısı.
        train (bool): Eğitim veya test veri seti olup olmadığını belirleyin.
    
    Returns:
        DataLoader: PyTorch DataLoader nesnesi.
    """
    dataset = CreditCardDataset(data_dir, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


if __name__ == "__main__":
    # Veri kümesi ve DataLoader testi
    data_dir = '../data/'  # Verinin bulunduğu ana dizin
    train_loader = get_dataloader(data_dir, batch_size=32, train=True)
    test_loader = get_dataloader(data_dir, batch_size=32, train=False)
    
    # İlk batch'i inceleyin
    for batch in train_loader:
        print(batch['features'].shape, batch['label'].shape)
        break
