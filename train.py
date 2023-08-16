import random
import time
import torch
import options.options as option
from solvers import create_solver
from data import create_dataloader, data_prefetcher
from data import create_dataset, trans_data
import os
import numpy as np
from networks import get_network_description
from test import validate
import atexit
from utils.stdio import print_to_markdwon_table



def main():
    args = option.add_train_args().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpuid)
    opt = option.parse(args)

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    pytorch_seed(seed)

    ##################### {dataset} #####################
    collate_fn = None
    if opt['mask_training'] is not None:
        collate_fn = opt['mask_training']
    for dataname, dataset_opt in sorted(opt['datasets'].items()):
        print('===> Dataset: %s <===' % (dataname))
        if 'train' in dataname:
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(
                train_set, dataset_opt, collate_fn)
            train_loader.dataname = dataname
            if train_loader is None:
                raise ValueError("[Error] The training data does not exist")
        elif 'val' in dataname:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            val_loader.dataname = dataname
        else:
            raise NotImplementedError(
                "[Error] Dataset phase [%s] in *.json is not recognized." % dataname)

    ##################  network, optimizer, loss #####################
    solver = create_solver(opt)
    run_device = next(solver.model.parameters()).device
    # solver.model, solver.optimizer = amp.initialize(solver.model, solver.optimizer, opt_level='O0')

    ################  { local logging }  ################
    solver_log = solver.get_current_log()
    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    
    ################  { Training Section }  ################
    # gradientScaler = GradScaler()
    last_loss = None
    last_tot_norm = None
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> [%d/%d] [Netx%d: %s] [Dataset: %s/%s] [lr: %.6f] <===' % (epoch, NUM_EPOCH,
                opt['scale'], opt['networks']['net_arch'], train_loader.dataname, val_loader.dataname, solver.get_current_learning_rate()))
        ###################  Training   ###################
        trainLoss = 0
        imgCount = 0
        start_time = time.time()
        acuGsteps = opt['solver']['acuSteps']
        solver.model.train()
        batch_idx = 0
        torch.cuda.empty_cache()
        for batch, batchpath in train_loader:
            batchdata = trans_data.data2device(batch, run_device)
            if opt['networks']['net_arch'] == 'cu_nets':
                out, trainLoss = solver.model.optimize_joint_parameters(
                    batchdata)
                trainLoss += trainLoss.item()*batch['LR'].shape[0]
                imgCount += batch['LR'].shape[0]
                if (batch_idx+1) == 100*acuGsteps:
                    break
            else:
                out = solver.model(batchdata)
                loss = solver.model.loss(out, batchdata)
                loss /= acuGsteps
                trainLoss += loss.item()*batch['LR'].shape[0]
                imgCount += batch['LR'].shape[0]
                if batch_idx == start_epoch:
                    last_loss = trainLoss
                # gradientScaler.scale(loss).backward()
                # if trainLoss < 3*last_loss:
                loss.backward()
                # if last_loss is None: last_loss = loss.item()
                # if loss > 3*last_loss and epoch>50: 
                #     print('Skip this iteration')
                #     solver.optimizer.zero_grad()
                #     continue
                # else:
                #     last_loss = loss.item()
                if last_tot_norm is None:
                    last_tot_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                            for p in solver.model.parameters()]))
                torch.nn.utils.clip_grad_norm_(solver.model.parameters(), max_norm=3*last_tot_norm, norm_type=2)
                last_tot_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                            for p in solver.model.parameters()]))
                # print(last_tot_norm)
                if (batch_idx+1) % acuGsteps == 0:
                    # gradientScaler.step(solver.optimizer)
                    # gradientScaler.update()
                    solver.optimizer.step()
                    solver.optimizer.zero_grad()
                    pass
                if (batch_idx+1) == 100*acuGsteps: break
            batch_idx += 1
        end_time = time.time()
        # del loss, out, batchdata
        trainLoss /= imgCount
        print('[Train]: No.Iter: %d | Avg Train Loss: %.6f | lossType: %s | Time: %.1f'
              % ((batch_idx+1) // acuGsteps, trainLoss, opt['solver']['loss_type'], end_time-start_time))

        ################ {validation} ################
        torch.cuda.empty_cache()
        s = time.time()
        testMetrics, SRout = validate(solver.model, val_loader, opt['datasets'][val_loader.dataname],
                                      is_visualize_out=(epoch%10==0), is_visGT=(epoch == 1))
        e = time.time()
        columns = ['Networks']
        rows = ['%8s'%opt['networks']['net_arch']]
        for key, value in testMetrics.items():
            columns +=  [key]
            rows += ['%.5f'%value]
        columns +=  ['Time']
        rows += ['%.5f'%(e-s)]
        print_to_markdwon_table(columns, [rows])
        
        # '[ val ]: {:^{width}}\n{:^{width}}'.fromat(key=tvalue for tkey, tvalue in testMetrics.items()), width=len('[ val ]: ')
        # print("[ val ]:%s" % (''.join([' %s: %.4f |' % (tkey, tvalue)
        #                                for tkey, tvalue in testMetrics.items()])))

        ################  { local logging }  ################
        solver_log['epoch'] = epoch
        epoch_is_best = False
        if solver_log['best_pred'] is None or solver_log['best_pred'] > testMetrics['ERGAS']:
            solver_log['best_pred'] = testMetrics['ERGAS']
            epoch_is_best = True
            solver_log['best_epoch'] = epoch
        solver_log['records']['train_loss'].append(trainLoss)
        solver_log['records']['lr'].append(solver.get_current_learning_rate())
        testMetrics.pop('time')
        for tkey, tvalue in testMetrics.items():
            if epoch == 1:
                solver_log['records'][tkey] = [tvalue]
            else:
                solver_log['records'][tkey].append(tvalue)

        print('Best Epoch [%d] [ERGAS: %.4f]' % (
            solver_log['best_epoch'], solver_log['best_pred']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch_is_best)
        solver.save_current_records()

        ####### { Updating params. rely on epochs }  #######
        solver.update_learning_rate()

        ################  { comet logging }  ################
        
    best_records = {key: value[solver_log['best_epoch']-1] for key, value
                    in solver_log['records'].items()}
    
    print('===> Finished !')
    ################  {end main}  ################


def pytorch_seed(seed=0):
    print("===> Random Seed: [%d]" % seed)
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()
