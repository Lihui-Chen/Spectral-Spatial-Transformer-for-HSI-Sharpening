import functools
import importlib
import torch
import torch.nn as nn
from torch.nn import init
import imp
import os

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################


    
# choose one network
def create_model(opt, exp_root=None):

    which_model = opt['net_arch']
    print('===> Building network [%s]...'%which_model)
    Model = importlib.import_module('networks.'+which_model, which_model)
    net = Model.Net(opt)

    if torch.cuda.is_available() and which_model != 'cu_nets':
        net = net.cuda()
    print_network(net, exp_root)
    return net

def print_network(model, exp_root):
    """
    print network summary including module and number of parameters
    """
    s, n = get_network_description(model)
    if isinstance(model, nn.DataParallel):
        net_struc_str = '{} - {}'.format(model.__class__.__name__,
                                            model.module.__class__.__name__)
    else:
        net_struc_str = '{}'.format(model.__class__.__name__)
    print("==================================================")
    print("===> Network Summary\n")
    net_lines = []
    line = s + '\n'
    print(line)
    net_lines.append(line)
    line = 'Network structure: [{}], with parameters: [{:,d}]'.format(
        net_struc_str, n)
    print(line)
    net_lines.append(line)
    if exp_root is not None and os.path.isdir(exp_root):
        with open(os.path.join(exp_root, 'network_summary.txt'), 'w') as f:
            f.writelines(net_lines)
    print("==================================================")

def get_network_description(network):
    '''Get the string and total parameters of the network'''
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n