from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from inspect import isfunction
import copy

from ...links.noise import MCDropout
from ...functions import stride_pooling_nd
from ...initializers import bilinear_upsample
from ...links.connection import PixelShuffleUpsampler2D
from ...links.connection import PixelShuffleUpsampler3D


# supported functions

_supported_convs_2d = {
    'conv': nn.Conv2d,
    # 'deformable': DeformableConvolution2D, # NOTE: unsupported in PyTorch
}

_supported_convs_3d = {
    'conv': nn.Conv3d,
}

_supported_upconvs_2d = {
    'deconv': nn.ConvTranspose2d,
    'pixel_shuffle': PixelShuffleUpsampler2D,
}

_supported_upconvs_3d = {
    'deconv': nn.ConvTranspose3d,
    'pixel_shuffle': PixelShuffleUpsampler3D,
}

_supported_pools_2d = {
    'none': lambda x: x,
    'max': F.max_pool2d,
    'average': F.avg_pool2d,
    'stride': stride_pooling_nd,
}

_supported_pools_3d = {
    'none': lambda x: x,
    'max': F.max_pool3d,
    'average': F.avg_pool3d,
    'stride': stride_pooling_nd,
}

_supported_norms_2d = {
    'batch': nn.BatchNorm2d,
    'instance': nn.InstanceNorm2d,
}

_supported_norms_3d = {
    'batch': nn.BatchNorm3d,
    'instance': nn.InstanceNorm3d,
}

_supported_activations = {
    'none': lambda x: x,
    'identity': lambda x: x,
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    # 'clipped_relu': F.clipped_relu, # NOTE: unsupported in PyTorch
    # 'crelu': F.crelu,
    'elu': F.elu,
    # 'hard_sigmoid': F.hard_sigmoid,
    'softplus': F.softplus,
    'softmax': F.softmax,
    'log_softmax': F.log_softmax,
    # 'maxout': F.maxout,
    # 'swish': F.swish,
    'selu': F.selu,
    'rrelu': F.rrelu,
    'prelu': F.prelu,
}

_supported_dropouts = {
    'none': lambda x: x,
    'dropout': nn.Dropout,
    'mc_dropout': MCDropout,
}

_supported_initializers = {
    'zero': nn.init.zeros_,
    'identity': nn.init.eye_,
    'constant': nn.init.constant_,
    'one': nn.init.ones_,
    'normal': nn.init.normal_,
    # 'lecun_normal': , # NOTE: unsupported in PyTorch
    'glorot_normal': nn.init.xavier_normal_,
    'he_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
    'uniform': nn.init.uniform_,
    # 'lecun_uniform': ,
    'glorot_uniform': nn.init.xavier_uniform_,
    'he_uniform': nn.init.kaiming_uniform_,
    'bilinear': bilinear_upsample,
}

_supported_link_hooks = {
    'spectral_normalization': nn.utils.spectral_norm,
    'weight_standardization': nn.utils.weight_norm,
}

# default parameters

_default_conv_param = {
    'name':'conv',
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'padding_mode': 'reflect',
    'initialW': {'name': 'he_normal'},
    'initial_bias': {'name': 'zero'},
}

_default_upconv_param = {
    'name':'deconv',
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
    # 'padding_mode': 'reflect', # NOTE: unsupported in PyTorch
    'initialW': {'name': 'bilinear'},
    'initial_bias': {'name': 'zero'},
}

_default_pool_param = {
    'name': 'max',
    'kernel_size': 2,
    'stride': 2,
}

_default_norm_param = {
    'name': 'batch'
}

_default_activation_param = {
    'name': 'relu',
    # 'inplace': True
}

_default_dropout_param = {
    'name': 'dropout',
    'p': .5,
}


def _mapper(param, supported):
    assert isinstance(param, dict)
    assert isinstance(supported, dict)

    param = copy.deepcopy(param)

    if 'name' not in param.keys():
        raise ValueError('"name" must be in param.keys()..')

    name = param.pop('name')

    if name not in supported.keys():
        raise KeyError('"%s" is not supported.. Available: %s'
                            % (name, supported.keys()))

    func = supported[name]

    if isfunction(func):
        return partial(func, **param)
    elif issubclass(func, (nn.Module)):
        return func(**param)
    else:
        raise ValueError('unsupported class type.. <%s>' % func.__class__)


def pool(ndim, param):
    """ Return a function of the pool layer """
    if ndim == 2:
        supported_pools = _supported_pools_2d
    else:
        supported_pools = _supported_pools_3d

    return _mapper(param, supported_pools)

def activation(param):
    """ Return a function of the activation layer """
    return _mapper(param, _supported_activations)

def dropout(param):
    """ Return a function of the activation layer """
    return _mapper(param, _supported_dropouts)

def initializer(param):
    """ Return a function of the initializer """
    return _mapper(param, _supported_initializers)

def norm(ndim, size, param):
    """ Return a object of the normalization layer """
    param = copy.deepcopy(param)
    param['num_features'] = size

    if ndim == 2:
        supported_norms = _supported_norms_2d
    else:
        supported_norms = _supported_norms_3d

    return _mapper(param, supported_norms)

def link_hook(param):
    """ Return a function of the link hook """
    return _mapper(param, _supported_link_hooks)

def conv(ndim, in_channels, out_channels, param):
    """ Return a object of the convolution layer """
    conv_param = copy.deepcopy(param)

    initialW_param = conv_param.pop('initialW', None)
    initial_bias_param = conv_param.pop('initial_bias', None)
    hook_param = conv_param.pop('hook', None)

    conv_param['in_channels'] = in_channels
    conv_param['out_channels'] = out_channels

    if ndim == 2:
        supported_convs = _supported_convs_2d
    else:
        supported_convs = _supported_convs_3d

    link = _mapper(conv_param, supported_convs)

    if link.weight is not None \
            and initialW_param is not None:
        initialW = initializer(initialW_param)
        initialW(link.weight.data)

    if link.bias is not None \
            and initial_bias_param is not None:
        initial_bias = initializer(initial_bias_param)
        initial_bias(link.bias.data)

    if hook_param is not None:
        hook = link_hook(hook_param)
        link = hook(link)

    return link

def upconv(ndim, in_channels, out_channels, param):
    """ Return a object of the up-convolution layer """
    conv_param = copy.deepcopy(param)

    initialW_param = conv_param.pop('initialW', None)
    initial_bias_param = conv_param.pop('initial_bias', None)
    hook_param = conv_param.pop('hook', None)

    conv_param['in_channels'] = in_channels
    conv_param['out_channels'] = out_channels

    if ndim == 2:
        supported_upconvs = _supported_upconvs_2d
    else:
        supported_upconvs = _supported_upconvs_3d

    link = _mapper(conv_param, supported_upconvs)

    if link.weight is not None \
            and initialW_param is not None:
        initialW = initializer(initialW_param)
        initialW(link.weight.data)

    if link.bias is not None \
            and initial_bias_param is not None:
        initial_bias = initializer(initial_bias_param)
        initial_bias(link.bias.data)

    if hook_param is not None:
        hook = link_hook(hook_param)
        link = hook(link)

    return link
