import os
import  os.path
import  numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader

from datetime import datetime

class MyDataset(Dataset):
    def __init__(self, args, device_length=96, pml_thickness=40):
        self.data_folder = args.data_folder
        self.wl = args.wl
        self.dL = args.dL
        self.farfield_name = args.farfield_name

        self.eps = torch.from_numpy(np.load(self.data_folder + '/' + 'input_eps.npy', mmap_mode='r').astype(np.float32))
        self.eps = torch.unsqueeze(self.eps, dim=1)

        self.farfield = torch.from_numpy(np.load(self.data_folder + '/' + self.farfield_name + '.npy', mmap_mode='r').astype(np.complex64))
        self.farfield = self.farfield[:,:-1]

        self.farfield_mean_real = torch.mean(self.farfield.real, dim=0)
        self.farfield_mean_imag = torch.mean(self.farfield.imag, dim=0)
        self.farfield_mean = torch.complex(self.farfield_mean_real, self.farfield_mean_imag)
        self.farfield_mean_real = torch.mean(self.farfield.real, dim=0)
        self.farfield_mean_imag = torch.mean(self.farfield.imag, dim=0)

        if args.total_sample_number:
            random.seed(1234)
            indices = np.array(random.sample(list(range(len(self.directories))), total_sample_number))
            self.eps = eps[indices]
            self.farfield = self.farfield[indices]

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, idx):
    	return {"eps": self.eps[idx], 'farfield': self.farfield[idx], 'farfield_mean': self.farfield_mean}

