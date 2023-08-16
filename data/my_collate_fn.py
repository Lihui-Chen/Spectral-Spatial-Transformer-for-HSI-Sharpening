#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_collect_fn.py    
@Contact :   lihuichen@stu.scu.edu.cn
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
20-8-17 上午10:50   LihuiChen      1.0         None
'''
# import lib
import torch.nn.functional as F
import torch
from random import shuffle

def mask_collate_fn(data):
    '''
    :param data: batch of samples with different numbers of channels
    :return:
        data:
        data_length: [batch_size] refers to how many channel of each sample in the bacth
        mask:
    '''
    data, path = data[0][0], data[0][1]
    data.sort(key=lambda x: x['LR'].shape[0], reverse=True)
    data_length = [sq['LR'].shape[0] for sq in data]
    mask = []
    for i in range(data_length[0]):
        mask.append(sum(value>i for value in data_length))

    pad_size = [data_length[0]-sq for sq in data_length]
    for idx, train_data in enumerate(data):
        if pad_size[idx] != 0:
            train_data['LR'] = pad_tensor(train_data['LR'], data_length[0], dim=0)
            train_data['HR'] = pad_tensor(train_data['HR'], data_length[0], dim=0)
    batch_data = {}
    batch_data['LR'] = torch.stack([train_data['LR'] for train_data in data])
    batch_data['GT'] = torch.stack([train_data['GT'] for train_data in data])
    batch_data['PAN'] = torch.stack([train_data['PAN'] for train_data in data])
    return batch_data, data_length, mask


def rand_band_collate_fn(data):
    '''
    :param data: batch of samples with different numbers of channels
    :return:
        data:
        data_length: [batch_size] refers to how many channel of each sample in the bacth
        mask:
    '''
    data_length, mask = None, None
    for train_data in data:
        num_band = train_data['LR'].shape[0]
        band_idx = list()
        shuffle(band_idx)
        train_data['LR'] = train_data['LR'][band_idx,:,:]
        train_data['HR'] = train_data['HR'][band_idx,:,:]
    batch_data = {}
    batch_data['LR'] = torch.stack([train_data['LR'] for train_data in data])
    batch_data['HR'] = torch.stack([train_data['HR'] for train_data in data])
    batch_data['PAN'] = torch.stack([train_data['PAN'] for train_data in data])
    return batch_data, data_length, mask


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


# class PadCollate:
#     """
#     a variant of callate_fn that pads according to the longest sequence in
#     a batch of sequences
#     """
#
#     def __init__(self, dim=0):
#         """
#         args:
#             dim - the dimension to be padded (dimension of time in sequences)
#         """
#         self.dim = dim
#
#     def pad_collate(self, batch):
#         """
#         args:
#             batch - list of (tensor, label)
#
#         reutrn:
#             xs - a tensor of all examples in 'batch' after padding
#             ys - a LongTensor of all labels in batch
#         """
#         # find longest sequence
#         max_len = max(map(lambda x: x[0].shape[self.dim], batch))
#         # pad according to max_len
#         batch = map(lambda (x, y):(pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
#         # stack all
#         xs = torch.stack(map(lambda x: x[0], batch), dim=0)
#         ys = torch.LongTensor(map(lambda x: x[1], batch))
#         return xs, ys
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)




# class MyDataLoader():
#     def __init__(self):
#         pass
#     def
#
# class Wrapper(torch.utils.data.Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.sampling_size = sampling_size
#         self.num_classes = 1000
#
#         # create class2sample dict
#         self.class_idx_to_sample_ids = {i: [] for i in range(self.num_classes)}
#         for idx, (_, class_idx) in enumerate(dataset):
#             self.class_idx_to_sample_ids[class_idx].append(idx)
#
#     def __len__(self):
#         return self.sampling_size
#
#     def __getitem__(self, index):
#         if index < self.sampling_size:
#             class_idx = 1  # the class idx you want
#             sample_ids = np.random.choice(self.class_idx_to_sample_ids[class_idx], 1, replace=True)
#             return self.dataset[sample_ids[0]]