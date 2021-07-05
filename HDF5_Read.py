import h5py
import cv2
import numpy as np
import argparse
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(self.h5_path, 'r') as record:
            keys = list(record.keys())   
 

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index, jitter=.2, hue=.1, sat=1.5, val=1.5):

        with h5py.File(self.h5_path, 'r') as record:
            keys = list(record.keys())
            train_data = np.array(record[keys[index]]['train']) 

            target_data = np.array(record[keys[index]]['target'])   

        return train_data, target_data, keys[index]

    def __len__(self):
        with h5py.File(self.h5_path,'r') as record:
            return len(record)

