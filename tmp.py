import cv2
import numpy as np
from scipy.stats import pearsonr

def SAM(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    [h,w,_] = img1.shape
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2).squeeze()
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2)).squeeze()
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2)).squeeze()
    img1_mul_img2 = img1_spectral_norm*img2_spectral_norm
    img12 = np.copy(img1_mul_img2)
    img1_mul_img2[np.where(img1_mul_img2==0)] = np.finfo(np.float64).eps
    sam_map = np.arccos(inner_product/img1_mul_img2)
    
    inner_product.resize(h*w)
    img12.resize(h*w)
    inner_product = np.delete(inner_product, np.where(img12==0))
    img12 = np.delete(img12, np.where(img12==0))
    sam = np.degrees(np.mean(np.arccos((inner_product/img12).clip(min=0, max=1))))
    # numerical stability
    # cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    # return np.mean(np.arccos(cos_theta))
    return sam_map, sam

def ERGAS(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        means_real = means_real**2
        means_real[np.where(means_real==0)] = np.finfo(np.float64).eps
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses/means_real).mean())
    else:
        raise ValueError('Wrong input image dimensions.')
    
    
def CC(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    h,w,c=img1.shape
    CC=0
    for i in range(c):
        img1_single=img1[:,:,i]
        img1_single=img1_single.reshape(-1,1).squeeze()
        img2_single = img2[:, :, i]
        img2_single = img2_single.reshape(-1, 1).squeeze()

        # img1_single = img1[..., i].flatten()
        # img2_single = img2[...,i].flatten()
        cc=pearsonr(img1_single,img2_single)[0]
        CC=CC+cc
    CC=CC/c
    return CC
def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
    #print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0 this not necessary, because if mu==0, the simga must be 0.
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq)* (mu1_sq + mu2_sq) != 0)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)

def Q_AVE(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    result = np.load('/home/ser606/Documents/LihuiChen/ArbRPN_20200916/results/SR/RNN_RESIDUAL_BI_PAN_FB_MASK/QB-FIX-4/x4/LR_00044_x4_RNN_RESIDUAL_BI_PAN_FB_MASK.npy')
    target = np.load('/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/HRMS_npy/HR_00044.npy')
    sam_map, sam = SAM(result, target)
    ergas = ERGAS(result, target)
    cc = CC(result, target)
    Q_ave = Q_AVE(result, target, 32)


    print('sam:%.4f, ergas:%.4f, cc:%.4f, Q_ave:%.4f'%(sam, ergas, cc, Q_ave))
