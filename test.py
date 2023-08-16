
import time
import os
import options.options as option
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import numpy as np
from data import fileio, trans_data
import torch
from utils.util import pan_calc_metrics_all as all_metrics
from utils.util import metrics_part
from utils.vistool import hist_line_stretch
from utils.stdio import print_to_markdwon_table

@trans_data.multidata
def data2device(batch:torch.Tensor, device):
    return batch.to(device)

def validate(net, dataloader, dataopt, is_saveImg=False, savedir= './results', is_visualize_out=False, is_visGT=False):
    run_range = dataopt['run_range']
    img_range = dataopt['img_range']
    scale = dataopt['scaledict']['REF']
    valLossTime = {'val_loss': 0, 'time':0}
    imageCount = 0
    net.eval()
    vis_tbloggin = {'Out':[], 'GT':[]} if is_visGT else {'Out':[]}
    for batchIdx, (dataBatch, batchPath) in enumerate(dataloader):
        with torch.no_grad():
            GT = dataBatch.get('GT', dataBatch.get('HR', None))
            dataBatch = data2device(dataBatch, next(net.parameters()).device)
            strat_time = time.time()
            out = net(dataBatch)
            end_time = time.time()
            valLossTime['time'] += (end_time-strat_time)
            valLossTime['val_loss'] += (net.loss(out, dataBatch).item())
        out = out.split(1, dim=0)
        out = trans_data.tensor2np([tmp.squeeze().cpu() for tmp in out], img_range, run_range, is_quantize=(img_range>=255))
        if GT is not None:
            GT = GT.split(1, dim=0)
            imageCount += len(GT)
            GT = trans_data.tensor2np([tmp.squeeze().cpu() for tmp in GT], img_range, run_range, is_quantize=(img_range>=255))
            for imgIdx, (tmpout, tmpGT) in enumerate(zip(out, GT)):
                tmpMetrics = metrics_part(tmpout, tmpGT, scale=scale, img_range=img_range)
                if imgIdx==0 and batchIdx==0:
                    test_metrics = tmpMetrics 
                else:
                    test_metrics={tkey:test_metrics[tkey]+tvalue
                                  for tkey, tvalue in tmpMetrics.items()}
        if is_visGT and batchIdx==0:
            GT = [tmp[:,:,(4,3,2)] for tmp in GT]
            GT = [hist_line_stretch(tmp.astype(np.float), nbins=255)[0] for tmp in GT]
            GT = np.concatenate(GT, axis=1)
            vis_tbloggin['GT'] = GT
            
        if is_visualize_out and batchIdx==0: 
            out = [tmp[:,:,(4,3,2)] for tmp in out]
            out = [hist_line_stretch(tmp.astype(np.float), nbins=255)[0] for tmp in out]
            out = np.concatenate(out, axis=1)
            vis_tbloggin['Out'] = out
        
    test_metrics = {**test_metrics, **valLossTime}
    test_metrics = {tkey: tvalue/len(dataloader.dataset) for tkey, tvalue in test_metrics.items()}
    
    return test_metrics, vis_tbloggin


def test(solver, dataloader, opt, is_saveImg=True, savedir= None):
    dataopt = opt['datasets'][dataloader.name]
    run_range = dataopt['run_range']
    img_range = dataopt['img_range']
    scale = dataopt['scaledict']['REF']
    dataname = dataopt['name']
    valTime = []
    net = solver.model
    net.eval()
    for imgIdx, (dataBatch, batchPath) in enumerate(dataloader):
        with torch.no_grad():
            strat_time = time.time()
            dataBatch = data2device(dataBatch, next(net.parameters()).device)
            GT = dataBatch.get('GT', dataBatch.get('HR', None))
            out = net(dataBatch, imgIdx)
            end_time = time.time()
            valTime.append(end_time-strat_time)
        out = trans_data.tensor2np(out.squeeze().cpu(), img_range, run_range, is_quantize=False)
        if GT is not None:
            GT = trans_data.tensor2np(GT.squeeze().cpu(), img_range, run_range, is_quantize=False)
            tmpMetric = all_metrics(out, GT, scale=scale, img_range=img_range)
            print(batchPath['LR'])
            if imgIdx==0:
                test_metrics = {tkey:[tvalue,] for tkey, tvalue in tmpMetric.items()}
            else:
                for tkey, tvalue in tmpMetric.items(): test_metrics[tkey].append(tvalue)
        if is_saveImg: 
            path = os.path.join('results', dataname, savedir)
            if not os.path.isdir(path): os.makedirs(path)
            path = os.path.join(path, os.path.basename(batchPath['LR'][0])[:-4])
            fileio.save_img(path, out, '.npy')
            
    test_metrics = {**test_metrics, **{'Time':valTime}}
    test_metrics = {tkey: np.mean(np.array(tvalue)) for tkey, tvalue in test_metrics.items()}
    return test_metrics



def main():
    args = option.add_test_args().parse_args()
    opt = option.parse(args)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['net_arch'].upper()
    # create test dataloader
    bm_names = []
    test_loaders = {}
    for dataname, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loader.name=dataname
        # test_loaders.append(test_loader)
        test_loaders[dataname] = test_loader
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (
            dataname, len(test_set)))

    # create solver (and load model)
    solver = create_solver(opt)
    # solver.cal_flops()
    print("==================== Start Test========================")
    print("Method: %s || Scale: %d || Degradation: %s" %(model_name, scale, degrad))
    for dataname, dataloader in test_loaders.items():
        testMetrics = test(solver, dataloader, opt, is_saveImg=True, savedir=opt['results_dir'])
        print('======= The results on %s are as following. ======='%dataname)
        columns = ['Networks']
        rows = ['%8s'%model_name]
        for key, value in testMetrics.items():
            columns +=  [key]
            rows += ['%.4f'%value]
        print_to_markdwon_table(columns, [rows])
    print("======================= END =======================")

if __name__ == '__main__':
    main()
