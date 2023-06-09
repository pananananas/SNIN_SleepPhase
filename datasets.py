import torch as t
import numpy as np
from torch.utils.data import Dataset

"""Przykladowe dane"""
class BasicDataset(Dataset):

    def __init__(self) -> None:
        xy = np.loadtxt("dane.csv", delimiter=',', dtype=t.float32, skiprows=1)
        # skiprows=1 - pomiń pierwszy rząd, któy zwykle zawiera nagłówek
        self.x = t.from_numpy(xy[:, 1:]) # weź karzdy rząd, pomiń tylko pierwszą kolumnę
        self.y = t.from_numpy(xy[:, [0]]) # weź tylko pierwszą kolumnę karzdego rzędu
        self.n_samples = xy.shape[0]
        # pierwsza kolumna tego modelu zawiera nazwy klas

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

