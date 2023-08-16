import torch
import numpy as np
import torch.nn.functional as F
# from utils.genMTF import GNyq2win
import random
EPS = 1e-7

def shuffle_channel(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C//groups, H, W)
    x = x.transpose(1, 2)
    x = x.contiguous()
    x = x.view(B, C, H, W)
    return x

def layer_norm(x, eps=1e-6):
    dim_len = len(x.shape)
    ori = x
    b = x.shape[0]
    x = x.view(b, -1)
    mean_x = x.mean(dim=-1).view(b, *([1]*(dim_len-1)))
    std_x = x.std(dim=-1, unbiased=False).view(b, *([1]*(dim_len-1)))
    std_x = torch.max(std_x, torch.ones_like(std_x)*EPS)
    return (ori-mean_x)/std_x, mean_x, std_x

def instance_norm(x): 
    "x: Tensor"
    B,C = x.shape[:2]
    ori = x
    x = x.view(B, C, -1)
    mean_x = x.mean(dim=-1).view(B,C, 1, 1)
    std_x = x.std(dim=-1).view(B, C, 1,1)
    std_x = torch.max(std_x, torch.ones_like(std_x)*EPS)
    return (ori-mean_x)/std_x, mean_x, std_x

def calc_mean_std(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def normalize(input):
    size = input.shape
    assert(len(size)==4)
    N, C =  size[:2]
    # input_tmp = input.view(N, C, -1)
    input_max = input.max()
    input_min = input.min()
    if input_max==input_min and input_max!=0:
        input_min=0
    elif input_max==input_min and input_max==0:
        input_max = 1
    # input_max = input_tmp.max(dim=2).values
    # input_min = input_tmp.min(dim=2).values
    # input_min = torch.where((input_min!=0)|(input_min!=input_max), input_min, torch.tensor(1.0, device=input.device))
    # input_min = torch.where(input_max!=input_min, input_min, torch.tensor(0.0, device=input.device))
    # input_max = input_max.view(N, C, 1, 1)
    # input_min = input_min.view(N, C, 1, 1)
    return (input-input_min)/(input_max-input_min), input_max, input_min

def denormalize(input, input_max, input_min):
    return input*(input_max-input_min) + input_min

def data_norm(ms, pan):
    norm_min = min(ms.min(), pan.min())
    norm_max = max(ms.max(), pan.max())
    ms = (ms-norm_min)/(norm_max-norm_min)
    pan = (pan-norm_min)/(norm_max-norm_min)
    return ms, pan, norm_min, norm_max

def get_filter_kernel(type, direction=None):
    if type=='Sobel':
        if direction=='y':
            kernel = torch.tensor([[1.0, 2, 1], [0, 0 ,0], [-1, -2, -1]]).div_(8)
        else:
            kernel = torch.tensor([[1.0, 0, -1], [2, 0 ,-2], [1, 0, -1]]).div_(8)

    return kernel


def reduce_mean(x, dim=[]):
    # x_shape = x.shape
    x = torch.flatten(x, dim[0], dim[1])
    x = x.mean(dim=dim[0], keepdim=True)
    return x

def reduce_sum(x, dim=[]):
    # x_shape = x.shape
    x = torch.flatten(x, dim[0], dim[1])
    x = x.sum(dim=dim[0], keepdim=True)
    return x
    
def modPad(x, mod, dim):
    
    n = x.shape[dim]
    if n%mod != 0:
        padsize = (n//mod+1)*mod - n
        x = F.pad(x, [0, padsize, 0, 0], mode='replicate')
    return x, padsize