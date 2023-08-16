import torch.utils.data
import torch.utils.data._utils.collate as collate
import data.my_collate_fn as my_collate_fn


def create_dataloader(dataset, dataset_opt, collate_fn=None):
    if collate_fn is None:
        collate_fn = collate.default_collate
    elif collate_fn == 'mask_collate_fn':
        collate_fn = my_collate_fn.mask_collate_fn
    elif collate_fn == 'rand_band_collate_fn':
        collate_fn = my_collate_fn.rand_band_collate_fn

    phase = dataset_opt['phase']
    if 'train' == phase :
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    elif 'valid' == phase:
        batch_size = dataset_opt['batch_size']
        shuffle = False
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1

    return torch.utils.data.DataLoader(\
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

def create_dataset(dataset_opt):
    mode = dataset_opt['setmode'].upper()
    if 'RAM' in mode:
        from data.dataset_ram import LRHRDataset as D
    else :
        from data.dataset_disk import LRHRDataset as D
    dataset = D(dataset_opt)
    return dataset

class data_prefetcher():
    def __init__(self, loader, img_range=255.0):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.img_range = torch.cuda.FloatTensor([img_range/255.0])
        # self.scale = scale
        self.preload()

    def preload(self):
        try:
            self.nextbatch = next(self.loader)
        except StopIteration:
            self.nextbatch = None
            return
        with torch.cuda.stream(self.stream):
            self.nextbatch[0]['LR'] = self.nextbatch[0]['LR'].cuda(non_blocking=True)#.mul_(self.img_range)
            self.nextbatch[0]['HR'] = self.nextbatch[0]['HR'].cuda(non_blocking=True)#.mul_(self.img_range)
            self.nextbatch[0]['PAN'] = self.nextbatch[0]['PAN'].cuda(non_blocking=True)#.mul_(self.img_range)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.nextbatch
        self.preload()
        return batch

