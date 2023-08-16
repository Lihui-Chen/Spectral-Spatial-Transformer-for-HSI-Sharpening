import os
import random
import numpy as np
from networks.common_block import Mlp, linear_attn
import scipy.misc as misc
import imageio
# from tqdm import tqdm
import torch
import glob
import torch.nn as nn
from data import trans_data

BENCHMARK = ['IK', 'WV2', 'P', 'SP', 'QB']
VALID_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
                    '.bmp', '.BMP', '.tiff', '.TIFF', '.tif', '.TIF', '.mat', '.npy']


def is_valid_file(filename):
    return any(filename.endswith(extension) for extension in VALID_EXTENSIONS)

def _get_paths_from_dataroot(path, ext):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = glob.glob(os.path.join(path, '*'+ext))
    assert images, '[%s] has no valid file' % path
    return images


def get_image_paths(data_type, dataroot, subset=None):
    paths = None
    if dataroot is not None:
        if 'npy' in data_type:
            old_dir = dataroot
            dataroot = old_dir #+ '_npy'
            if not os.path.exists(dataroot):
                print('===> Creating binary files in [%s]' % dataroot)
                os.makedirs(dataroot)
                img_paths = sorted(_get_paths_from_dataroot(old_dir, data_type))
                # path_bar = tqdm(img_paths)
                for v in img_paths:
                    img = read_img(v, data_type)
                    ext = os.path.splitext(os.path.basename(v))[-1]
                    name_sep = os.path.basename(v.replace(ext, '.npy'))
                    np.save(os.path.join(dataroot, name_sep), img)
            paths = sorted(_get_paths_from_dataroot(dataroot, data_type))
        else:
            paths = sorted(_get_paths_from_dataroot(dataroot, data_type))
    else:
        raise ValueError('dataroot of dataset is None.')
    
    if subset is None:
        return paths
    start = int(subset[0]*len(paths))
    end = -1 if subset[1] == 1 else int(subset[1]*len(paths))
    return paths[start:end]

# if 'mat' in data_type:
#     img = read_img(v, 'mat')
# elif 'img' in data_type:
#     img = read_img(v, 'img')
# elif 'tif' in data_type:
#     img = read_img(v, 'tif')
# else:
#     raise TypeError('Cannot process this data type.')

def read_img(path, data_type):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if 'npy' in path:
        img = np.load(path)
    elif 'img' in data_type:
        img = imageio.imread(path, pilmode='RGB')
    elif 'mat' in data_type:
        from scipy import io as sciio
        img = sciio.loadmat(file_name=path)
    elif 'tif' in data_type:
        from skimage import io as skimgio
        img = skimgio.imread(path)
    else:
        raise NotImplementedError('Cannot read this type (%s) of data'%data_type)
    if isinstance(img, np.ndarray) and img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


####################
# image processing
# process on numpy image
####################

def np2Tensor(l_dict, run_range, img_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))/1.0
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(run_range / img_range)
        return tensor
    return {t_key:_np2Tensor(_l) for t_key, _l in l_dict.items()}


def _Tensor2np(tensor_dict, run_range, img_range):
    def _Tensor2numpy(tensor, run_range):
        array = np.transpose(
            quantize(tensor, run_range, img_range).numpy(), (1, 2, 0)
        ).astype(np.uint16)
        return array
    return {t_key:_Tensor2numpy(tensor, run_range) for t_key, tensor in tensor_dict.items()}


def quantize(img, run_range, img_range):
    return img.mul(img_range / run_range).clamp(0, int(img_range)).round()

def get_patch(imgdict, scale_dict, patch_size):
    '''
    imgdict: a list of images whose resolution increase with index of the list
    scale_dict: list of scales for the corresponding images in imglist
    patch_size: the patch size for the fisrt image to be cropped.
    '''
    if 'LR' in imgdict.keys():
        ih, iw = imgdict['LR'].shape[:2]
    else:
        ih, iw = imgdict['GT'].shape[:2]
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)
    sizedict = {t_key:(ix*t_scale, iy*t_scale, patch_size*t_scale) for t_key, t_scale in scale_dict.items()}
    out_patch = {t_key: imgdict[t_key][ix:ix+t_psize, iy:iy+t_psize] for t_key, (ix, iy, t_psize) in sizedict.items()}
    return out_patch


def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(input, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    @trans_data.multidata
    def _augment(img, hflip, vflip, rot90):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img
    return _augment(input, hflip, vflip, rot90)


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img


