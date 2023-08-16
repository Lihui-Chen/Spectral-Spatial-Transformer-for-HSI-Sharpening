from collections import OrderedDict

def set_dataset(dataname, phase, scale):
    '''
    dataname: Chikusei, CAVE, Harvard
    phase: train, valid, test
    '''
    data_setting = OrderedDict()
    data_setting['data_root'] = '../HypersharpDataset/%s/%s'%(dataname, phase)
    if dataname == 'Chikusei':
        data_setting['img_range'] = 4095
        data_setting['LRdim'] = 120
        data_setting['REFdim'] = 9
    elif dataname in ('CAVE', 'Harvard'):
        data_setting['img_range'] = 65535 if dataname=='CAVE' else 0.065
        data_setting['LRdim'] = 31
        data_setting['REFdim'] = 3
    # else:
        # return 
        # raise(ValueError('The dataset of %s is not suppport. Pls set it before runing.')%dataname)
    data_setting['scaledict'] = {'LR': 1, 'GT':scale, 'REF':scale}   
    return data_setting
    