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
from .common_block import ACT, Mlp, linear_attn, convMultiheadAttetionV2, speMultiAttn

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
        
class spaTransBlock(nn.Module):
    def __init__(self, convDim, numHeads, patchSize, rezero=False) -> None:
        super().__init__()
        self.multiAttn = convMultiheadAttetionV2(convDim, numHeads, patchSize)
        self.ffn = FeedForward(convDim)
        self.reweight = nn.Parameter(torch.zeros(1)) if rezero else 1

    def forward(self, x):
        """ 
        input: x  shape [B, C, H, W]
        """
        x = x + self.multiAttn(x)*self.reweight
        x = x + self.ffn(x)*self.reweight
        return x


class speTransBlock(nn.Module):
    def __init__(self, convDim, numHeads, poolSzie, ksize=3, rezero=False) -> None:
        super().__init__()
        self.multiattn = speMultiAttn(convDim, numHeads, poolSzie, ksize)
        self.ffn = FeedForward(convDim)
        self.reweight = nn.Parameter(torch.zeros(1)) if rezero else 1
            
        
    def forward(self, x):
        x = x + self.multiattn(x)*self.reweight
        x = x + self.ffn(x)*self.reweight
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
        self.ksize = opt['ksize']
        self.embedDim4spectral = self.convDim*self.poolSize**2
        self.embedDim4spatial = self.convDim*dimScale*self.patchSize**2
        
        ################  {net module}  ################
        self.conv_head = nn.Conv2d(self.dimHs+self.dimMs, self.convDim, 3, 1, 1)
        self.spe_embed = nn.Conv2d(self.dimHs+self.dimMs, self.convDim, 3, 1, 1)
          
        self.spa_trans = nn.ModuleList([
            spaTransBlock(self.convDim, self.numHeads, self.patchSize, rezero=False) for _ in range(self.numLayers)
        ])
        self.spe_trans = nn.ModuleList([
           speTransBlock(self.convDim, self.numHeads, poolSzie=self.poolSize, ksize=self.ksize, rezero=False) for _ in range(self.numLayers)
        ])
        self.fuse= nn.ModuleList([
           nn.Conv2d(self.convDim*2, self.convDim, 1, 1, 0) for _ in range(self.numLayers)
        ])
        self.transfer= nn.ModuleList([
           nn.Conv2d(self.convDim, self.convDim, 1, 1, 0) for _ in range(self.numLayers-1)
        ])
        self.dense_fuse = nn.Sequential(
            nn.Conv2d(self.convDim*self.numLayers, self.convDim, 1, 1, 0),
            nn.GELU()
        )
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
      

        interpHS = F.interpolate(hs, scale_factor=self.scale, mode='bicubic', align_corners=False)
        tmp = F.interpolate(ms, scale_factor=1.0/self.scale, mode='bicubic', align_corners=False)

        ms = torch.cat((ms, interpHS), dim=1)
        B, C, H, W  = ms.shape
        ms = self.conv_head(ms)

        hs = torch.cat((tmp, hs), dim=1)
        B, c, h, w = hs.shape
        hs = self.spe_embed(hs)
      
        fuse_fe = []
        for idx in range(self.numLayers):
            ms = self.spa_trans[idx](ms) 
            hs = self.spe_trans[idx](hs)
            tmp = F.interpolate(hs, scale_factor=self.scale, mode='bicubic', align_corners=False)
            tmp = torch.cat((ms,tmp), dim=1)
            tmp = self.fuse[idx](tmp)
            fuse_fe.append(tmp)
            if idx<self.numLayers-1:
                ms = ms+tmp
                tmp = F.interpolate(tmp, scale_factor=1.0/self.scale, mode='bicubic', align_corners=False)
                tmp = self.transfer[idx](tmp)
                hs = hs+tmp

        tmp = self.dense_fuse(torch.cat(fuse_fe, dim=1))
        ms = self.conv_tail(tmp)
        ms = ms + interpHS
        return ms
      

    def loss(self, rec=None, gt=None):
        return F.l1_loss(rec, gt['GT'])
    
