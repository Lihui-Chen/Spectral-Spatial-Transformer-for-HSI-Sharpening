# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023 Lihui Chen
All rights reserved. This work should only be used for nonprofit purposes.

@file        : trans_data.py
@Date        : 2023/08/16
@Author      : Lihui Chen
@version     : 1.0
@description : 
@reference   :
'''

import numpy as np
import torch


def multidata(func):
    def inner(input, *args, **kvargs):
        if isinstance(input, dict):
            return {t_key: func(tensor, *args, **kvargs) for t_key, tensor in input.items()}
        elif isinstance(input, (np.ndarray, torch.Tensor)):
            return func(input, *args)
        elif isinstance(input, (list, tuple)):
            return [func(tmp, *args, **kvargs) for tmp in input]
        else:
            raise(TypeError('Cannot transfer %s of np.ndarray to tensor'%type(input)))
    return inner

@multidata
def np2tensor(img, img_range, run_range=1):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))/1.0
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(run_range / img_range)
    return tensor

@multidata
def tensor2np(tensor, img_range, run_range=1, is_quantize=False):
    array = np.transpose(
        map2img_range(tensor, img_range, run_range, is_quantize).numpy(), (1, 2, 0)
    )
    return array

def map2img_range(img:torch.Tensor, img_range:float, run_range=1, is_quantize=False):
    if is_quantize:
        return img.mul(img_range / run_range).clamp(0, int(img_range)).round()
    else:
        return img.mul(img_range / run_range).clamp(0, img_range)
    
@multidata
def data2device(batch:torch.Tensor, device):
    return batch.to(device)