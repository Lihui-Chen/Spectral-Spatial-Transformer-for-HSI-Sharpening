import os
from collections import OrderedDict
from datetime import datetime
import yaml
import torch
from yaml.events import NodeEvent
from data.dataset_setting import set_dataset
import data.fileio as fileio
import shutil
import argparse

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def _common_args():
    # basic args
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-gpuid', type=str, default='0,', help='Define the gpu id to run code.')
    parser.add_argument('-net_arch', type=str, default=None, help='The network to run.')
    parser.add_argument('-pretrained_path', type=str, default=None)
    parser.add_argument('-scale', type=float, default=None, help='The upscale for the running network.')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')

    ########## network setting  ##########
    parser.add_argument('-convDim', type=int, default=None)
    parser.add_argument('-numHeads', type=int, default=None)
    parser.add_argument('-patchSize', type=int, default=None)
    parser.add_argument('-poolSize', type=int, default=None)
    parser.add_argument('-numLayers', type=int, default=None)
    parser.add_argument('-ksize', type=int, default=None)

    return parser

def add_train_args():
    parser = _common_args()

    # for logging
    parser.add_argument('-log_dir', type=str, default=None, help='The path of saved model.')
    parser.add_argument('-tags', type=str, nargs='+',default=None)
    parser.add_argument('-tag', type=str, default=None)

    ########### dataset-setting ##########
    parser.add_argument('-setmode', type=str, default=None)
    parser.add_argument('-repeat', type=int, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-patch_size', type=int, default=None)

    ########### optimizer-setting ##########
    parser.add_argument('-optimType', type=str, default=None)
    parser.add_argument('-learning_rate', type=float, default=None)
    parser.add_argument('-lr_scheme', type=str, default=None)
    parser.add_argument('-warmUpEpoch', type=int, default=None)
    parser.add_argument('-lrStepSize', type=int, default=None)
    parser.add_argument('-acuSteps', type=int, default=None)
    parser.add_argument('-num_epochs', type=float, default=None)
    
    return parser

def add_test_args():
    parser = _common_args()
    parser.add_argument('-results_dir', type=str, default=None)
    return parser


def parse(args):
    Loader, Dumper = OrderedYaml()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    
    if opt['mode']=='train' and opt['solver']['pretrain'] == 'resume':
        opt = opt['solver']['pretrained_path']
        assert os.path.isfile(opt), \
            'The models of %s does not exist.'%opt
        opt = os.path.join(os.path.dirname(opt), 'options.yml') 
        with open(opt, mode='r') as f:
            opt = yaml.load(f, Loader=Loader)
        return opt

    def add_args2yml(chars, optdic):
        if getattr(args, chars, None) is not None:
            optdic[chars] = getattr(args, chars)
        return optdic
    
    ################  { data Setting }  ################
    run_range = opt['run_range']
    for dataname, dataset in opt['datasets'].items():
        phase, name = dataname.split('_')
        dataset['phase']=phase
        dataset['name'] = name
        data_root = dataset.get('data_root', None)
        dataset_setting = set_dataset(name, phase, opt['scale'])
        dataset = {**dataset, **dataset_setting}
        if data_root is not None: dataset['data_root'] = data_root
        dataset['run_range'] = run_range
        dataset = add_args2yml('setmode', dataset)
        if 'train' == phase:
            dataset = add_args2yml('repeat', dataset)
            dataset = add_args2yml('batch_size', dataset)
            dataset = add_args2yml('patch_size', dataset)
        opt['datasets'][dataname]=dataset
    
    ################  { network Setting }  ################
    opt['networks']['scale'] = opt['scale']
    opt['networks'] = add_args2yml('net_arch', opt['networks'])
    opt['networks'] = add_args2yml('convDim', opt['networks'])
    opt['networks'] = add_args2yml('numHeads', opt['networks'])
    opt['networks'] = add_args2yml('numLayers', opt['networks'])
    opt['networks'] = add_args2yml('patchSize', opt['networks'])
    opt['networks'] = add_args2yml('poolSize', opt['networks'])
    opt['networks'] = add_args2yml('ksize', opt['networks'])
    opt['networks']['LRdim'] = opt['datasets'][dataname]['LRdim']
    opt['networks']['REFdim'] = opt['datasets'][dataname]['REFdim']
    
    
    ################  { optim&lr_rate Setting }  ################
    opt['solver'] = add_args2yml('pretrained_path', opt['solver'])
    if opt['mode']=='train':
        opt['solver'] = add_args2yml('optimType', opt['solver'])
        opt['solver'] = add_args2yml('learning_rate', opt['solver'])
        opt['solver'] = add_args2yml('lr_scheme', opt['solver'])
        opt['solver'] = add_args2yml('warmUpEpoch', opt['solver'])
        opt['solver'] = add_args2yml('lrStepSize', opt['solver'])
        opt['solver'] = add_args2yml('acuSteps', opt['solver'])
        opt['solver'] = add_args2yml('num_epochs', opt['solver'])
    
   
    ################  { Logging setting }  ################
    opt['timestamp'] = get_timestamp() # logging date and time
    if opt['mode']=='train': # train
        opt['logger'] = add_args2yml('tag', opt['logger'])
        opt['logger'] = add_args2yml('tags', opt['logger'])
        config_str = '%s' %(opt['networks']['net_arch'])
        if opt['logger'].get('tag', None) is not None: config_str = config_str + '_' + opt['logger']['tag']
        config_str = getattr(args, 'log_dir', '') + config_str
        opt = set_log_dir(opt, config_str, Dumper)
    else: # test
        opt = add_args2yml('results_dir', opt)
        opt['results_dir'] =  opt.get('results_dir', '') + opt['networks']['net_arch']
    opt = dict_to_nonedict(opt)
    return opt


def set_log_dir(opt, config_str, Dumper):
    if opt['solver']['pretrain'] == 'finetune': # finetune
        assert os.path.isfile(opt['solver']['pretrained_path']), \
            'The models of %s does not exist.'%opt
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        exp_path += '_finetune'
    else:
        exp_path = os.path.join('experiments', config_str)
        
    exp_path = os.path.relpath(exp_path)
    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['path'] = path_opt
    
    fileio.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
    fileio.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'))
    save_setting(opt, Dumper)
    print("===> Experimental DIR: [%s]" % exp_path)
    
    
    return opt

def save_setting(opt, Dumper):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.yml')
    network_file = opt["networks"]['net_arch'] + '.py'
    shutil.copy('./networks/' + network_file, os.path.join(dump_dir, network_file))
    with open(dump_path, 'w') as dump_file:
        yaml.dump(opt, dump_file, Dumper=Dumper)


class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')