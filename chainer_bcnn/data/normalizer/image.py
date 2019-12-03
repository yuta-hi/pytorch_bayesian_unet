from __future__ import absolute_import

import numpy as np
from . import Operation

_row_axis = 1
_col_axis = 2
_channel_axis = 0

def quantize(x, n_bit, x_min=0., x_max=1., rescale=True):

    n_discrete_values = 1 << n_bit
    scale = (n_discrete_values - 1) / (x_max - x_min)
    quantized = np.round(x * scale) - np.round(x_min * scale)
    quantized = np.clip(quantized, 0., n_discrete_values)

    if not rescale:
        return quantized

    quantized /= scale
    return quantized + x_min

class Quantize(Operation):
    """ Quantize the given images to specific resolution.

    Non-linearity and overfitting make the neural networks sensitive to tiny noises in high-dimensional data [1].
    Quantizing it to a necessary and sufficient level may be effective, especially for medical images which have >16 bits information.
     [1] Goodfellow et al., "Explaining and harnessing adversarial examples.", 2014. https://arxiv.org/abs/1412.6572

    Args:
        n_bit (int): Number of bits.
        x_min (float, optional): Minimum value in the input domain. Defaults to 0.
        x_max (float, optional): Maximum value in the input domain. Defaults to 1.
        rescale (bool, optional): If True, output value is rescaled to input domain. Defaults to True.
    """
    def __init__(self, n_bit, x_min=0., x_max=1., rescale=True):
        self._args = locals()
        self._args.pop('self')
        self._n_bit = n_bit
        self._x_min = x_min
        self._x_max = x_max
        self._rescale = rescale
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x):
        x = [quantize(x_i, self._n_bit,
                      self._x_min, self._x_max, self._rescale)
                for x_i in x]
        return x

    def summary(self):
        return self._args


def clip(x, param, scale=1.):

    if isinstance(param, str):
        if param == 'minmax':
            param = (np.min(x), np.max(x))
        elif param == 'ch_minmax':
            tmp = np.swapaxes(x, _channel_axis, 0)
            tmp = np.reshape(tmp, (len(tmp), -1))
            tmp_shape = [len(tmp)] + [1] * (x.ndim - 1)
            param = (np.min(tmp, axis=1).reshape(tmp_shape),
                     np.max(tmp, axis=1).reshape(tmp_shape))
        else:
            raise NotImplementedError('unsupported parameters..')

    assert isinstance(param, (list, tuple))
    x = (x - param[0]) / (param[1] - param[0])  # [0, 1]
    x = np.clip(x, 0., 1.)

    return x * scale

class Clip(Operation):
    """ Clip (limit) the values in given images.

    Args:
        param (tuple or str): Tuple of minimum and maximum values.
            If 'minmax' or 'ch_minmax', minimum and maximum values are automatically estimated.
            'ch_minmax' is the channel-wise minmax normalization.
    """
    def __init__(self, param):
        self._args = locals()
        self._args.pop('self')
        self._param = param
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x):
        x = [clip(x_i, self._param) for x_i in x]
        return x

    def summary(self):
        return self._args


def subtract(x, param):

    if isinstance(param, str):
        if param == 'mean':  # NOTE: for z-score normalization
            param = np.mean(x)
        elif param == 'ch_mean':
            tmp = np.swapaxes(x, _channel_axis, 0)
            tmp = np.reshape(tmp, (len(tmp), -1))
            tmp_shape = [len(tmp)] + [1] * (x.ndim - 1)
            param = np.mean(tmp, axis=1).reshape(tmp_shape)
        else:
            raise NotImplementedError('unsupported parameters..')

    x -= param

    return x

class Subtract(Operation):
    """ Subtract a value or tensor from given images.

    Args:
        param (float, numpy.ndarray or str): A value or tensor.
            If 'mean' or 'ch_mean', subtracting values are automatically estimated.
            'ch_mean' is to subtract the channel-wise mean.
    """
    def __init__(self, param):
        self._args = locals()
        self._args.pop('self')
        self._param = param
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x):
        x = [subtract(x_i, self._param) for x_i in x]
        return x

    def summary(self):
        return self._args


def divide(x, param):

    if isinstance(param, str):
        if param == 'std':  # NOTE: for z-score normalization
            param = np.std(x)
        elif param == 'ch_std':
            tmp = np.swapaxes(x, _channel_axis, 0)
            tmp = np.reshape(tmp, (len(tmp), -1))
            tmp_shape = [len(tmp)] + [1] * (x.ndim - 1)
            param = np.std(tmp, axis=1).reshape(tmp_shape)
        else:
            raise NotImplementedError('unsupported parameters..')

    x /= param

    return x

class Divide(Operation):
    """ Divide the given images by a value or tensor

    Args:
        param (float, numpy.ndarray or str): A value or tensor.
            If 'std' or 'ch_std', deviding values are automatically estimated.
            'ch_std' is to divide the channel-wise standard deviation.
    """
    def __init__(self, param):
        self._args = locals()
        self._args.pop('self')
        self._param = param
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x):
        x = [divide(x_i, self._param) for x_i in x]
        return x

    def summary(self):
        return self._args
