import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-7

def linear_attn(q, k, x, EPS=EPS):
    l1, d1  = q.shape[-2:]
    l2, d2 = x.shape[-2:]
    k = k.transpose(-2, -1)
    if l1*d1*l2+l1*l2*d2<= d2*l2*d1+d2*d1*l1:
        q = q@k
        q = q/(q.sum(dim=-1, keepdim=True)+EPS)
        x = q@x
    else:
        x = q@(k@x)
        q = q@k.sum(dim=-1, keepdim=True) + EPS
        x = x/q
    return x


class ACT():
    r'''
    args:
        actType: chose one of 'relu', 'prelu', 'lrelu'
        negative_slope: for 'lrelu' and initial vlaue for 'prelu'
    return:
        activation function
    '''
    def __init__(self, actType, negative_slope=0.01):
        super().__init__()
        self.actType = actType
        self.negative_slope = negative_slope

    def get_act(self,):
        if self.actType.lower() == 'relu':
            act = nn.ReLU(True)
        elif self.actType.lower() == 'lreul':
            act = nn.LeakyReLU(self.negative_slope)
        elif self.actType.lower() == 'prelu':
            act = nn.PReLU(self.negative_slope)
        elif self.actType.lower() == 'gelu':
            act = nn.GELU()
        else: 
            raise('This type of %s activation is not added in ACT, please add it first.'%self.actType)
        return act


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inFe, outFe, kernel_size=3, stride=1, padding=1, actType=nn.ReLU()):
        super(ResBlock, self).__init__()
        self.is_linear = False
        if inFe != outFe:
            self.linear = nn.Conv2d(inFe, outFe, 1, 1, 0)
            self.is_linear = True
        self.conv1 = nn.Conv2d(inFe, outFe, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = actType
        self.conv2 = nn.Conv2d(outFe, outFe, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        if self.is_linear:
            x = self.linear(x)
        x = x + res
        return x


def conv2d(inCh, outCh, kSize, stride, pad,):
    return nn.Sequential(
        nn.Conv2d(inCh, outCh, kernel_size=kSize, stride=stride, padding=(kSize-1)//2),
        nn.ReLU(inplace=True),
    )

class transBlock(nn.Module):
    def __init__(self, dim, numHeads) -> None:
        super().__init__()
        self.multiattn = MultiAttn(dim, numHeads)
        self.ffn = Mlp(dim)
        
    def forward(self, x):
        x = x + self.multiattn(x)
        x = x + self.ffn(x)
        return x

class MultiAttn(nn.Module):
    def __init__(self, dim, numHeads) -> None:
        super().__init__()
        self.numHeads = numHeads
        self.qkv = nn.Linear(dim, dim*3)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, dim)

    def forward(self,x):
        B, L, dim = x.shape
        x = self.qkv(x)
        x = x.reshape(B, L, 3, self.numHeads, dim//self.numHeads)
        x = x.permute(2, 0, 3, 1, 4)
        q, k, x = x[0], x[1], x[2]
        q = self.relu(q)
        k = self.relu(k)
        x = linear_attn(q, k, x)
        x = x.transpose(1,2).contiguous()
        x = x.view(B, L, dim)
        x = self.proj(x)
        return x 
        
class speMultiAttn(nn.Module):
    def __init__(self, convDim, numHeads, poolSize, ksize=3) -> None:
        super().__init__()
        self.numHeads = numHeads
        self.convDim = convDim
        self.poolSize = poolSize
        self.avepool = nn.AdaptiveAvgPool2d(self.poolSize)
        self.relu = nn.ReLU()
        self.q = nn.Linear(self.poolSize**2, self.poolSize**2*self.numHeads)
        self.k = nn.Linear(self.poolSize**2, self.poolSize**2*self.numHeads)
        self.v = nn.Conv2d(self.convDim, self.convDim, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, ksize, 1, (ksize-1)//2, groups=self.convDim)
    

    def forward(self, x, padsize=0):
        B, C, H, W = x.shape
        q = self.avepool(x)
        q = q.view(B, C, -1)
        q = self.q(q)
        k = self.avepool(x)
        k = k.view(B, C, -1)
        k = self.k(k)
        x = self.v(x)
        q = self.relu(q).view(B, C, self.numHeads, -1)
        k = self.relu(k).view(B, C, self.numHeads, -1)
        
        x = x.view(B,C,-1)
        if x.shape[2]%self.numHeads!=0:
            padsize = (x.shape[2]//self.numHeads+1)*self.numHeads-x.shape[2]
            x = F.pad(x, [0, padsize, 0, 0], mode='replicate') 
        x = x.view(B, C, self.numHeads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        x = x.transpose(1, 2)
        x = linear_attn(q, k, x)
        x = x.transpose(1,2).contiguous().view(B, C, -1)
        if padsize:
            x = x[:,:,:-padsize]
        x = x.view(B, C, H, W)
        x = self.proj(x)
        return x
        # pass
        
class convMultiheadAttetionV2(nn.Module):
    r""" Patch based multihead attention.
    """
    def __init__(self, convDim, numHeads, patchSize, qkScale=None, qkvBias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.convDim = convDim
        self.numHeads = numHeads
        self.patchSize = patchSize
        self.relu = nn.ReLU()
        self.register_buffer('one', None)
        self.qkv = nn.Conv2d(self.convDim,  self.convDim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, 3, 1, 1)

    def forward(self, x, mask=None, padsize=0):
        """
        x: [B_, N, C] N=H*W
        """
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.convDim, H, W).transpose(0, 1)
        q, k, x = qkv[0], qkv[1], qkv[2]
        del qkv
        q, k = self.relu(q), self.relu(k)
        q = F.unfold(q, self.patchSize, padding=(self.patchSize-1)//2, stride=1)# [B, numHead, headDim, H, W]
        k = F.unfold(k, self.patchSize, padding=(self.patchSize-1)//2, stride=1)
        x = x.view(B, C, H*W)
        
        if q.shape[1]%self.numHeads!=0:
            padsize = (q.shape[1]//self.numHeads+1)*self.numHeads-q.shape[1]
            q = F.pad(q, [0, 0, 0, padsize], mode='replicate')
            k = F.pad(k, [0, 0, 0, padsize], mode='replicate')
            x = F.pad(x, [0, 0, 0, padsize], mode='replicate')  
        
        q = q.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        k = k.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        
        x = x.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        ################  {attention}  ################
        x = linear_attn(q, k, x)
        x = x.transpose(-1, -2).contiguous().view(B, -1, H*W)
        if padsize != 0:
            x = x[:, :-padsize, :]
        x = x.view(B, C, H, W)
        ################  { FFN }  ################
        x = self.proj(x)
        return x
        
    def flops(self, x):
        B, C, H, W = x.shape
        d1 = C*self.patchSize**2 #for q and k
        l1 = H*W      
        d2 = C
        flops = 0
        flops += l1 * C* 3 * C
        flops += min(l1*d1*l1+l1*l1*d2, l1*d1*d2+d1*l1*d2)
        flops += l1*C * 9*C # 
        return flops