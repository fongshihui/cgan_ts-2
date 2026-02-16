import torch
from torch.utils.data import Dataset
import numpy as np

class ResidualDataset(Dataset):
    def __init__(self, npz):
        d = np.load(npz)
        self.x = d['eps']
        self.c = d['cond']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i]).unsqueeze(-1),
            torch.tensor(self.c[i], dtype=torch.long)
        )
