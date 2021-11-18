import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
import os

DIAGNOSIS_MAP = {
    'NONE': 0,
    'LVHV': 1,
    'STTC': 2,
    'TWC': 3
}

class PublicECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_train, path_to_ecg, batch_size=8, val_split=0.2):
        n_samples = len(pd.read_csv(path_to_train))
        n_train = math.ceil((n_samples*(1-val_split)))
        train_seq = cls(path_to_train, path_to_ecg, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_train, path_to_ecg, batch_size, start_idx=n_train)

        return train_seq, valid_seq

    def __init__(self, path_to_train, path_to_ecg, batch_size, start_idx=0, end_idx=None, predict_data=False):
        train_df = pd.read_csv(path_to_train)
        sliced_df = train_df.iloc[start_idx : end_idx]
        #sliced_df.to_csv('debug.csv')
        if predict_data: 
            self.y = None
        else: 
            self.y = sliced_df.iloc[:, 1:].to_numpy()

        self.ecgFiles = sliced_df['FileName']
        ecg_list = []
        for ecg in self.ecgFiles:
            ecg_df = pd.read_csv(os.path.join(path_to_ecg, ecg + '.csv'))
            ecg_list.append(ecg_df.to_numpy())
            
        self.x = np.stack(ecg_list, axis=0)
        
        self.batch_size = batch_size
        self.start_idx = start_idx
        if end_idx is None:
            end_idx = sliced_df.index[-1] + 1
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
