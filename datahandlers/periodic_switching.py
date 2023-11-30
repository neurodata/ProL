import numpy as np
import torch
from torch.utils.data import Dataset

def sample(t, N=10):
    '''sample from the process'''
    if (t // (N/2)) % 2 == 0:
        mu = 1
    else:
        mu = -1
    y = np.random.binomial(1, 0.5)
    x = np.random.normal((-1)**(y+1)*mu, 0.5)
    return x, t, y

class PeriodicSwitchingDataset(Dataset):
    def __init__(self, cfg):
        np.random.seed(cfg.seed)
        data = np.array([sample(s, cfg.N) for s in range(0, cfg.t)])
        self.data = torch.from_numpy(data)
        self.contextlength = cfg.contextlength
        self.t = cfg.t

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ### check
        r = np.random.randint(0, len(self.data)-2*self.contextlength) # select the end of the subsequence
        s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' time beyond the subsequence
        z = torch.cat((self.data[r:r+self.contextlength], self.data[s:s+1]))
        y = z[-1, -1].clone()
        z[-1, -1] = 0.5
        return z, y