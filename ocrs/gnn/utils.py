"""Utility functions."""
import os
import h5py
import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname, use_rle, image_shape):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('use_rle', data=np.array(use_rle))
        hf.create_dataset('image_shape', data=np.array(image_shape))
        for episode in range(len(array_dict)):
            grp = hf.create_group(str(episode))
            for array_name, array_value in array_dict[episode].items():
                if use_rle and isinstance(array_value[0], np.ndarray):
                    # align sizes of rle arrays
                    max_len = len(max(array_value, key=len))
                    for j in range(len(array_value)):
                        element = array_value[j]
                        array_value[j] = np.pad(element, pad_width=(max_len - len(element), 0), constant_values=0)

                grp.create_dataset(array_name, data=array_value)


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        use_rle = 'use_rle' in hf and np.asarray(hf['use_rle']).item()
        image_shape = np.asarray(hf['image_shape']) if 'image_shape' in hf else None
        i = 0
        for grp in hf.keys():
            if grp in ('use_rle', 'image_shape'):
                continue

            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]

            i += 1

    return array_dict, use_rle, image_shape


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def make_node_mlp_layers(num_layers, input_dim, hidden_dim, output_dim, act_fn, layer_norm):
    layers = []

    for idx in range(num_layers):

        if idx == 0:
            # first layer, input_dim => hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_act_fn(act_fn))
        elif idx == num_layers - 2:
            # layer before the last, add layer norm
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_act_fn(act_fn))
        elif idx == num_layers - 1:
            # last layer, hidden_dim => output_dim and no activation
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # all other layers, hidden_dim => hidden_dim
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_act_fn(act_fn))

    return layers
