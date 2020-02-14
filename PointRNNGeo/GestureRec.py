

import torch
from torch.utils.data import Dataset
from torch_geometric.io import read_off
import numpy as np

class GestureRecdata(Dataset):


    def __init__(self, data,label):
        self.data = torch.tensor(data)
        self.label = torch.tensor(label,dtype=torch.int64)

    def __len__(self):
            return len(self.data)

    def __getitem__(self, index):
        target = self.label[index]
        data_val = self.data[index]
        return data_val, target



