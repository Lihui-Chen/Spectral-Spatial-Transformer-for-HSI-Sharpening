import torch.utils.data as data
# from data import common as common
from data import fileio, preproc, trans_data
from options.options import dict_to_nonedict
import numpy as np
import torch
import os

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''
    def name(self):
        return self.opt['dataname']

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.is_train = ('train' in opt['phase'])
        self.scale = opt['scaledict']['REF']
        self.repeat = opt['repeat'] if self.opt['repeat'] else 1
        self.runRange = opt['run_range']
        self.imgRange = opt['img_range']
        self.patchSize = opt['patch_size']

        ### get lr/hr image paths
        self.scaledict = opt['scaledict']
        self.img_paths = dict()
        for t_key in self.scaledict.keys():
            tpath = os.path.join(opt['data_root'], t_key)
            self.img_paths[t_key] = fileio.get_image_paths(tpath, opt['data_type'])
        self.data_len = len(self.img_paths[t_key])
        
        ### load images to ram
        print('Loading images from %s'%opt['data_root'])
        self.imgdict = {t_key:self._load_file(t_value, opt['data_type']) 
                        for t_key, t_value in self.img_paths.items()}
        print('===> End Loading [%04d] images <===\n'%self.__len__())

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        pathdict = {t_key:t_value[idx] for t_key, t_value in self.img_paths.items()}
        imgbatch_dict = {t_key:t_value[idx] for t_key, t_value in self.imgdict.items()}
        if self.is_train:
            imgbatch_dict = self._get_patch(imgbatch_dict)
        imgbatch_dict = trans_data.np2tensor(imgbatch_dict, self.imgRange, self.runRange)
        return (imgbatch_dict, pathdict)

    def __len__(self):
        if self.is_train:
            return self.data_len * self.repeat
        else:
            return self.data_len

    def _get_index(self, idx):
        if self.is_train:
            return idx % self.data_len
        else:
            return idx

    def _load_file(self, file_dir, data_type, patch_size=None):
        file_list = [fileio.read_img(tmp, data_type) for tmp in file_dir]
        return file_list


    def _get_patch(self, imgdict):
        imgdict = preproc.get_patch(imgdict, self.scaledict, self.patchSize)
        imgdict = preproc.augment(imgdict)
        # lr = common.add_noise(lr, self.opt['noise'])
        return imgdict