from __future__ import absolute_import

import chainer
from chainer import initializers
import chainer.functions as F
import chainer.links as L

from functools import partial
from inspect import isfunction
import copy

from ...functions import mc_dropout
from ...functions import stride_pooling_nd
from ...links.connection import Convolution2D
from ...links.connection import Convolution3D
from ...links.connection import DeformableConvolution2D
from ...links.connection import DilatedConvolution2D
from ...links.connection import DepthwiseConvolution2D
from ...links.connection import Deconvolution2D
from ...links.connection import Deconvolution3D
from ...links.connection import PixelShuffleUpsampler2D
from ...links.connection import PixelShuffleUpsampler3D
from ...links.normalization import InstanceNormalization
from ...initializers import BilinearUpsample


# supported functions

_supported_convs_2d = {
    'conv': Convolution2D,
    'deformable': DeformableConvolution2D,
    'dilated': DilatedConvolution2D,
    'local': L.LocalConvolution2D,
    # 'depthwise': DepthwiseConvolution2D, # TODO
    # 'mlp': L.MLPConvolution2D, #TODO
}

_supported_convs_3d = {
    'conv': Convolution3D,
}

_supported_upconvs_2d = {
    'deconv': Deconvolution2D,
    'pixel_shuffle': PixelShuffleUpsampler2D,
}

_supported_upconvs_3d = {
    'deconv': Deconvolution3D,
    'pixel_shuffle': PixelShuffleUpsampler3D,
}

_supported_pools = {
    'none': lambda x: x,
    'max': F.max_pooling_nd,
    'average': F.average_pooling_nd,
    'stride': stride_pooling_nd,
}

_supported_norms = {
    'batch': L.BatchNormalization,
    'instance': InstanceNormalization,
    'decorrelated_batch': L.DecorrelatedBatchNormalization,
    'group': L.GroupNormalization,
    'layer': L.LayerNormalization,
}

_supported_activations = {
    'none': lambda x: x,
    'identity': F.identity,
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'clipped_relu': F.clipped_relu,
    'crelu': F.crelu,
    'elu': F.elu,
    'hard_sigmoid': F.hard_sigmoid,
    'softplus': F.softplus,
    'softmax': F.softmax,
    'log_softmax': F.log_softmax,
    'maxout': F.maxout,
    'swish': F.swish,
    'selu': F.selu,
    'rrelu': F.rrelu,
    'prelu': F.prelu,
}

_supported_dropouts = {
    'none': lambda x: x,
    'dropout': F.dropout,
    'mc_dropout': mc_dropout,
}

_supported_initializers = {
    'zero': initializers.Zero,
    'identity': initializers.Identity,
    'constant': initializers.Constant,
    'one': initializers.One,
    'nan': initializers.NaN,
    'normal': initializers.Normal,
    'lecun_normal': initializers.LeCunNormal,
    'glorot_normal': initializers.GlorotNormal,
    'he_normal': initializers.HeNormal,
    'orthogonal': initializers.Orthogonal,
    'uniform': initializers.Uniform,
    'lecun_uniform': initializers.LeCunUniform,
    'glorot_uniform': initializers.GlorotUniform,
    'he_uniform': initializers.HeUniform,
    'bilinear': BilinearUpsample,
}

# default parameters

_default_conv_param = {
    'name':'conv',
    'ksize': 3,
    'stride': 1,
    'pad': 1,
    'initialW': {'name': 'he_normal', 'scale': 1.0},
    'initial_bias': {'name': 'zero'},
}

_default_upconv_param = {
    'name':'deconv',
    'ksize': 3,
    'stride': 2,
    'pad': 0,
    'initialW': {'name': 'bilinear', 'scale': 1.0},
    'initial_bias': {'name': 'zero'},
}

_default_pool_param = {
    'name': 'max',
    'ksize': 2,
    'stride': 2,
}

_default_norm_param = {
    'name': 'batch'
}

_default_activation_param = {
    'name': 'relu'
}

_default_dropout_param = {
    'name': 'dropout',
    'ratio': .5,
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
    elif issubclass(func, chainer.Initializer):
        return func(**param)
    elif issubclass(func, (chainer.Link, chainer.Chain)):
        return func(**param)
    else:
        raise ValueError('unsupported class type.. <%s>' % func.__class__)


def pool(param):
    """ Return a function of the pool layer """
    return _mapper(param, _supported_pools)

def activation(param):
    """ Return a function of the activation layer """
    return _mapper(param, _supported_activations)

def dropout(param):
    """ Return a function of the activation layer """
    return _mapper(param, _supported_dropouts)

def initializer(param):
    """ Return a object of the initializer """
    return _mapper(param, _supported_initializers)

def norm(size, param):
    """ Return a object of the normalization layer """
    param = copy.deepcopy(param)
    param['size'] = size
    return _mapper(param, _supported_norms)

def conv(ndim, in_channels, out_channels, param):
    """ Return a object of the convolution layer """
    conv_param = copy.deepcopy(param)

    initialW_param = conv_param.get('initialW')
    initial_bias_param = conv_param.get('initial_bias')

    if initialW_param is not None:
        initialW = initializer(initialW_param)
        conv_param['initialW'] = initialW

    if initial_bias_param is not None:
        initial_bias = initializer(initial_bias_param)
        conv_param['initial_bias'] = initial_bias

    conv_param['in_channels'] = in_channels
    conv_param['out_channels'] = out_channels

    if ndim == 2:
        supported_convs = _supported_convs_2d
    else:
        supported_convs = _supported_convs_3d

    return _mapper(conv_param, supported_convs)

def upconv(ndim, in_channels, out_channels, param):
    """ Return a object of the up-convolution layer """
    conv_param = copy.deepcopy(param)

    initialW_param = conv_param.get('initialW')
    initial_bias_param = conv_param.get('initial_bias')

    if initialW_param is not None:
        initialW = initializer(initialW_param)
        conv_param['initialW'] = initialW

    if initial_bias_param is not None:
        initial_bias = initializer(initial_bias_param)
        conv_param['initial_bias'] = initial_bias

    conv_param['in_channels'] = in_channels
    conv_param['out_channels'] = out_channels

    if ndim == 2:
        supported_upconvs = _supported_upconvs_2d
    else:
        supported_upconvs = _supported_upconvs_3d

    return _mapper(conv_param, supported_upconvs)
