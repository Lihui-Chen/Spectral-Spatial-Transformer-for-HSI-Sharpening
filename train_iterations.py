import random
# from tqdm import tqdm
import time
# from torch.utils.tensorboard import SummaryWriter
import torch

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader, data_prefetcher
from data import create_dataset
import os
import numpy as np
# import data.my_collate_fn as my_collate

def pytorch_seed(seed=0):
    print("===> Random Seed: [%d]" %seed)
    seed=int(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def main():
    args = option.add_args()
    opt = option.parse(args)


    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    pytorch_seed(seed)

    # create train and val dataloader
    train_loader_list = []
    bm_names = []
    collate_fn = None
    if opt['collate_fn'] is not None:
        collate_fn = opt['collate_fn']
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'train' in phase:
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, collate_fn) #todo:
            train_loader_list.append(train_loader)
            print('===> Train Dataset: %s  Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")
            bm_names.append(train_set.name())
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s  Number of images: [%d]' % (val_set.name(), len(val_set)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    solver = create_solver(opt)

    scale = opt['scale']
    model_name = opt['networks']['net_arch'].upper()

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)"%(model_name, scale, start_epoch, NUM_EPOCH))
    # writer = SummaryWriter('runs/')
    lamda = 1
    idx_iter = 1
    all_iter = 99000
    iter_per_epoch = 99
    epoch = start_epoch
    data_prefetcher_list = [data_prefetcher(loader) for loader in train_loader_list]
    (batch, data_len, mask) = data_prefetcher_list[0].next()
    n_dataset = len(train_loader_list)
    train_loss_list = []
    training_start_time=time.time()
    while idx_iter <= all_iter:
        ## for training
        solver.feed_data(batch)
        iter_loss = solver.train_step(mask, lamda)
        batch_size = batch['LR'].size(0)
        train_loss_list.append(iter_loss*batch_size)
        
        idx_training_set = idx_iter % n_dataset
        (batch, data_len, mask)  = data_prefetcher_list[idx_training_set].next()
        idx_iter += 1
        
        # logging every epoch
        if idx_iter % iter_per_epoch ==0:
            training_end_time = time.time()
            solver_log['epoch'] = epoch
            solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
            solver_log['records']['lr'].append(solver.get_current_learning_rate())
            print('[Epoch]: [%d/%d] || No.Iter: %d || Avg Train Loss: %.6f || Time: %.1f' % (epoch,
                                                        NUM_EPOCH, iter_per_epoch,
                                                        sum(train_loss_list)/len(train_set), training_end_time-training_start_time))
            ## the validating stage

            psnr_list = []
            ssim_list = []
            val_loss_list = []

            for iter, val_batch in enumerate(val_loader):
                solver.feed_data(val_batch)
                iter_loss = solver.test()
                val_loss_list.append(iter_loss)

                # calculate evaluation metrics
                visuals = solver.get_current_visual()
                psnr, ssim = util.pan_calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale, img_range=opt['img_range'])
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                if opt["save_image"]:
                    solver.save_current_visual(epoch, iter)

            solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
            solver_log['records']['psnr'].append(sum(psnr_list)/len(psnr_list))
            solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))

            # record the best epoch
            epoch_is_best = False
            if solver_log['best_pred'] < (sum(psnr_list)/len(psnr_list)):
                solver_log['best_pred'] = (sum(psnr_list)/len(psnr_list))
                epoch_is_best = True
                solver_log['best_epoch'] = epoch

            print("[ val ]: [%d/%d] || CC: %.4f RMSE: %.4f Loss: %.6f  Best CC: %.4f in Epoch: [%d]" % (epoch,
                                                        NUM_EPOCH,sum(psnr_list)/len(psnr_list),
                                                                                                sum(ssim_list)/len(ssim_list),
                                                                                                sum(val_loss_list)/len(val_loss_list),
                                                                                                solver_log['best_pred'],
                                                                                                solver_log['best_epoch']))
            # writer.add_scalar('train_loss', sum(train_loss_list) / len(train_set))
            # writer.add_scalar('val_loss', sum(val_loss_list)/len(val_loss_list))
            # writer.add_scalar('CC', sum(psnr_list)/len(psnr_list))
            # writer.add_scalar('RMSE', sum(ssim_list)/len(ssim_list))

            solver.set_current_log(solver_log)
            solver.save_checkpoint(epoch, epoch_is_best)
            solver.save_current_log()

            # update lr
            solver.update_learning_rate(epoch)

            #update lmda
            lamda = max(1 - 0.01 * (epoch // 5), 0)
            epoch += 1
            training_start_time = time.time()

    # writer.close()
    print('===> Finished !')


if __name__ == '__main__':
    main()