import torch.utils.data as data
import numpy as np
from data import common
import torch
import collections


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return self.opt['dataname'].split('_')[1]

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.repeat = opt['repeat'] if opt['repeat'] is not None else 1
        self.patchSize = opt['patch_size']
        self.runRange = opt['run_range']
        self.imgRange = opt['img_range']
        self.is_train = ('train' == opt['phase'])
        self.scaledict = opt['scaledict']
        self.img_paths = collections.OrderedDict()
        for t_key in self.scaledict.keys():
            self.img_paths[t_key] = common.get_image_paths(opt['data_type'], opt[t_key], opt['subset'])
        self.data_len = len(self.img_paths[t_key])

    def __getitem__(self, idx):
        imgdict, pathdict = self._load_file(idx)
        if self.is_train:
            imgdict= self._get_patch(imgdict)
        # else:
        #     imgdict = self._get_center_patch(imgdict, self.scaledict, self.patchSize)
        imgdict = common.np2Tensor(imgdict, self.runRange, self.imgRange)
        return (imgdict, pathdict)

    def __len__(self):
        if self.is_train:
            return self.data_len*self.repeat
        return self.data_len


    def _get_index(self, idx):
        if self.is_train:
            return idx % self.data_len
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        img_path_dict = {t_key: tmp_path[idx] for t_key, tmp_path in self.img_paths.items()}
        img_dict = dict()
        if self.opt['data_type'] == '.mat':
            tmp_img_dict = common.read_img(img_path_dict['LR'], self.opt['data_type']) 
            img_dict['LR'] = tmp_img_dict['HSLR']
            img_dict['GT'] = tmp_img_dict['HSHR']
            img_dict['MSHR'] = tmp_img_dict['MSHR']
        else:
            img_dict = {t_key:common.read_img(tmp_path, self.opt['data_type']) for t_key, tmp_path in img_path_dict.items()}
        return img_dict, img_path_dict

    def _get_patch(self, imgdict):
        patch_size = self.patchSize
        # random crop and augment
        imgdict = common.get_patch(imgdict,self.scaledict,patch_size)
        imgdict = common.augment(imgdict)
        # lr = common.add_noise(lr, self.opt['noise'])
        return imgdict
    
    def _get_center_patch(self, imgdict, scale_dict, patch_size):
        '''
        imgdict: a list of images whose resolution increase with index of the list
        scale_dict: list of scales for the corresponding images in imglist
        patch_size: the patch size for the fisrt image to be cropped.
        '''
        iw, ih = imgdict['LR'].shape[:2]
        ix , iy = (iw-patch_size)//2, (ih-patch_size)//2
        sizedict = {t_key:(ix*t_scale, iy*t_scale, patch_size*t_scale) for t_key, t_scale in scale_dict.items()}
        out_patch = {t_key: imgdict[t_key][ix:ix+t_psize, iy:iy+t_psize] for t_key, (ix, iy, t_psize) in sizedict.items()}
        return out_patch
