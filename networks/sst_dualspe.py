# -*- encoding: utf-8 -*-
'''
Spectral and Patch Attention-based Vision Transformer for Hyperimage sharpening.
@File    :   baseline.py
@Contact :   lihuichen@stu.scu.edu.cn
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
xxxxxxxxx         LihuiChen      1.0         None
'''

# import lib

from .common_fn import instance_norm, shuffle_channel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .common_block import ACT, Mlp, linear_attn, speMultiAttn

EPS = 1e-7
# nn.GELU()

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False):
        super(FeedForward, self).__init__()
        act = ACT('gelu')
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gelu1 = act.get_act()
        self.depthConv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu2 = act.get_act()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.depthConv(x)
        x = self.gelu2(x)
        x = self.conv2(x)
        return x
        

class convMultiheadAttetion(nn.Module):
    r""" Patch based multihead attention.
    """
    def __init__(self, convDim, numHeads, patchSize, qkScale=None, qkvBias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.convDim = convDim
        self.numHeads = numHeads
        self.patchSize = patchSize
        # self.headDim = self.embedDim//self.numHeads
        # self.qkscale = qkScale if qkScale is not None else  (self.embedDim//self.numHeads)**-0.5
        # self.relu = nn.ReLU(True)
        self.relu = nn.ReLU()
        self.register_buffer('one', None)
        self.qkv = nn.Conv2d(self.convDim,  self.convDim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, 3, 1, 1)

    def forward(self, x, mask=None):
        """
        x: [B_, N, C] N=H*W
        """
        B, C, H, W = x.shape
        # B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.convDim, H, W).transpose(0, 1)
        # qkv = self.qkv(x).reshape(B, 3, self.numHeads, C//self.numHeads, H, W).transpose(0, 1)
        q, k, x = qkv[0], qkv[1], qkv[2]
        del qkv
        q, k = self.relu(q), self.relu(k)
        q = F.unfold(q, self.patchSize, padding=(self.patchSize-1)//2, stride=1)# [B, numHead, headDim, H, W]
        k = F.unfold(k, self.patchSize, padding=(self.patchSize-1)//2, stride=1)

        self.one = torch.ones_like(x)
        x = F.unfold(x, self.patchSize, padding=(self.patchSize-1)//2, stride=1)
        self.one = F.unfold(self.one, self.patchSize, padding=(self.patchSize-1)//2, stride=1)
        q = q.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        k = k.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        x = x.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        ################  {attention}  ################
        x = linear_attn(q, k, x)
        x = x.transpose(-1, -2).contiguous().view(B, -1, H*W)
        x = F.fold(x, [H,W], self.patchSize, padding=(self.patchSize-1)//2, stride=1)
        self.one = F.fold(self.one, [H,W], self.patchSize, padding=(self.patchSize-1)//2, stride=1)
        x = x/self.one
        x = x.view(B, C, H, W)
        ################  { FFN }  ################
        x = self.proj(x)
        return x

class convMultiheadAttetionV2(nn.Module):
    r""" Patch based multihead attention.
    """
    def __init__(self, convDim, numHeads, patchSize, qkScale=None, qkvBias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.convDim = convDim
        self.numHeads = numHeads
        self.patchSize = patchSize
        # self.headDim = self.embedDim//self.numHeads
        # self.qkscale = qkScale if qkScale is not None else  (self.embedDim//self.numHeads)**-0.5
        # self.relu = nn.ReLU(True)
        self.relu = nn.ReLU()
        self.register_buffer('one', None)
        self.qkv = nn.Conv2d(self.convDim,  self.convDim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(self.convDim, self.convDim, 3, 1, 1)

    def forward(self, x, mask=None):
        """
        x: [B_, N, C] N=H*W
        """
        B, C, H, W = x.shape
        # B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.convDim, H, W).transpose(0, 1)
        # qkv = self.qkv(x).reshape(B, 3, self.numHeads, C//self.numHeads, H, W).transpose(0, 1)
        q, k, x = qkv[0], qkv[1], qkv[2]
        del qkv
        q, k = self.relu(q), self.relu(k)
        q = F.unfold(q, self.patchSize, padding=(self.patchSize-1)//2, stride=1)# [B, numHead, headDim, H, W]
        k = F.unfold(k, self.patchSize, padding=(self.patchSize-1)//2, stride=1)
        x = x.view(B, C, H*W)
        
        q = q.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        k = k.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        x = x.view(B, self.numHeads, -1, H*W).transpose(-1, -2)
        ################  {attention}  ################
        x = linear_attn(q, k, x)
        x = x.transpose(-1, -2).contiguous().view(B, -1, H*W)
        x = x.view(B, C, H, W)
        ################  { FFN }  ################
        x = self.proj(x)
        return x

class convTransBlock(nn.Module):
    def __init__(self, convDim, numHeads, patchSize) -> None:
        super().__init__()
        self.multiAttn = convMultiheadAttetionV2(convDim, numHeads, patchSize)
        self.ffn = FeedForward(convDim)

    def forward(self, x):
        """ 
        input: x  shape [B, C, H, W]
        """
        x = x + self.multiAttn(x)
        x = x + self.ffn(x)
        return x


class speTransBlock(nn.Module):
    def __init__(self, convDim, numHeads, poolSzie) -> None:
        super().__init__()
        self.multiattn = speMultiAttn(convDim, numHeads, poolSzie)
        self.ffn = FeedForward(convDim)
        
    def forward(self, x):
        x = x + self.multiattn(x)
        x = x + self.ffn(x)
        return x
        # pass

class Net(nn.Module):
    r""" 
    args: 
        convDim: dim for spectral embedding
        dimScale: used to multiply convDim to get the dim for spatial embedding
        poolSize: size of adaptivepool2d in spectral embedding, combined with convDim,
            the embedding dim for spectral embedding is poolSize**2*convDim
        numHeads: the head number for multihead attention
        patchSize: the size of patch attention base transformer in spatial transformer
    """
    def __init__(self, opt, dimScale=1, ksize=3, stride=1, padding=1, attn_drop=0.0): #convDim, poolSize, numHeads, patchSize,
        super().__init__()
        ################  {network args}  ################
        self.useCheckpoint=True
        self.dimHs = opt['LRdim'] #dimHs
        self.dimMs = opt['REFdim'] # dimMs
        self.scale = opt['scale']
        self.numLayers = opt['numLayers']
        self.numHeads = opt['numHeads']
        self.convDim = opt['convDim']
        self.poolSize = opt['poolSize']
        self.patchSize = opt['patchSize']
        self.embedDim4spectral = self.convDim*self.poolSize**2
        self.embedDim4spatial = self.convDim*dimScale*self.patchSize**2
        # self.register_buffer('one', None)
        
        ################  {net module}  ################
        self.conv_head = nn.Conv2d(self.dimHs+self.dimMs, self.convDim, 3, 1, 1)
        self.spe_embed = nn.Conv2d(self.dimHs+self.dimMs, self.dimHs+self.dimMs, 3, 1, 1)
        self.spe_trans1 = nn.Sequential(*[
            speTransBlock(self.convDim, self.numHeads, self.poolSize) for _ in range(self.numLayers)
        ])
        self.spe_trans2 = nn.ModuleList([
           speTransBlock(self.dimHs+self.dimMs, self.numHeads, poolSzie=self.poolSize) for _ in range(self.numLayers)
        ])
        self.fuse= nn.ModuleList([
           nn.Conv2d(self.convDim+self.dimHs+self.dimMs, self.convDim, 1, 1, 0) for _ in range(self.numLayers)
        ])
        self.transfer= nn.ModuleList([
           nn.Conv2d(self.convDim, self.dimHs+self.dimMs, 1, 1, 0) for _ in range(self.numLayers-1)
        ])
        self.conv_tail = nn.Conv2d(self.convDim, self.dimHs, 3, 1, 1)
        
    def forward(self,batchData, mask=None):
        """
        inputs:
            hs: image with the size of [B, C1, h, w]
            hs: image with the size of [B, C2, H, W]
        mid variable:
            spectralEmbedding: [B, C1+C2, P**2*convDim]
            spatialEmbedding: [B*(C1+C2), convDim*dimScale, HW]
            spectralDependency: [B, C1, C1+C2] is the attention matrix
                to measure the similarity of bands in hs+ms to bands in ms
            patchDependency: [B*C2, H*W, H*W] is the attention matrix
                to measure the similarity of patchs to patchs in the ms image
        """
        hs = batchData['LR']
        ms = batchData['REF']
        del batchData
        # hs, mean_hs, std_hs = instance_norm(hs)
        # ms, mean_ms, std_ms = instance_norm((ms))
        

        interpHS = F.interpolate(hs, scale_factor=self.scale, mode='bicubic', align_corners=False)
        tmp = F.interpolate(ms, scale_factor=1.0/self.scale, mode='bicubic', align_corners=False)

        ms = torch.cat((ms, interpHS), dim=1)
        B, C, H, W  = ms.shape
        ms = self.conv_head(ms)
        
        # hs = torch.cat((tmp+mean_ms*std_ms, hs+mean_hs*std_hs), dim=1)
        hs = torch.cat((tmp, hs), dim=1)
        # hs = hs+mean_hs*std_hs
        B, c, h, w = hs.shape
        # hs = hs.view(B*c, 1, h, w)
        hs = self.spe_embed(hs)
      
       
        for idx in range(self.numLayers):
            ms = self.spe_trans1[idx](ms) 
            hs = self.spe_trans2[idx](hs)
            tmp = F.interpolate(hs, scale_factor=self.scale, mode='bicubic', align_corners=False)
            tmp = torch.cat((ms,tmp), dim=1)
            tmp = self.fuse[idx](tmp)
            if idx<self.numLayers-1:
                ms = ms+tmp
                tmp = F.interpolate(tmp, scale_factor=1.0/self.scale, mode='bicubic', align_corners=False)
                tmp = self.transfer[idx](tmp)
                hs = hs+tmp

        ms = self.conv_tail(tmp)
        ms = ms + interpHS
        # ms = ms*std_hs+mean_hs
        return ms
      
    def loss(self, rec=None, gt=None):
        return F.l1_loss(rec, gt)
    
    
