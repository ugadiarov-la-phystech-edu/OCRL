import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class FolderDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = self._parse_files()
    
    def __len__(self):
        return len(self.files)
        
    def _parse_files(self):
        obs = []
        for ep in sorted(os.listdir(self.root)):
            ep_folder = os.path.join(self.root, ep)
            for obs_step in sorted(os.listdir(ep_folder)):
                obs.append(os.path.join(ep_folder,obs_step))
        return obs
    def __getitem__(self, index):
        res = {"labels": []}
        res['obss'] = torch.Tensor(np.load(self.files[index])) /255.0
        
        return res

class DataSet(Dataset):
    def __init__(self, data):
        self._data = data
        self._num_samples = data["obss"].shape[0]

    def __getitem__(self, index):
        res = {}
        for key in self._data.keys():
            if key == "obss":
                res[key] = torch.Tensor(self._data[key][index]).permute(2,0,1)/255.0
            elif key == "labels":
                res[key] = torch.LongTensor([self._data[key][index]])
            else:
                if key == "num_objs":
                    continue
                res[key] = torch.Tensor(self._data[key][index])
        return res

    def __len__(self):
        return self._num_samples
