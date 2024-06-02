import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, data, obs_size, allow_resize=False):
        self._data = data
        self._allow_resize = allow_resize
        expected_shape = (obs_size, obs_size)
        actual_shape = data["obss"].shape[1:3]
        assert self._allow_resize or expected_shape == actual_shape, f'Expected shape={expected_shape}. Actual shape={actual_shape}'

        self._obs_size = obs_size
        self._need_resize = expected_shape != actual_shape
        self._num_samples = data["obss"].shape[0]
        if self._need_resize:
            self._resize_transform = torchvision.transforms.Resize(self._obs_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    def __getitem__(self, index):
        res = {}
        for key in self._data.keys():
            if key == "obss":
                obs = torch.Tensor(self._data[key][index]).permute(2,0,1)/255.0
                if self._need_resize:
                    obs = self._resize_transform(obs)
                res[key] = obs
            elif key == "labels":
                res[key] = torch.LongTensor([self._data[key][index]])
            else:
                if key == "num_objs":
                    continue
                res[key] = torch.Tensor(self._data[key][index])
        return res

    def __len__(self):
        return self._num_samples
