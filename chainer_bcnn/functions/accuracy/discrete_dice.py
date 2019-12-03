from __future__ import absolute_import

import six
import numpy as np
import chainer
from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check

from ..loss._helper import to_onehot

class DiscreteDice(function.Function):
    """ Dice coefficient
    NOTE: This is not a differentiable function.
        See also: ~chainer_bcnn.functions.loss.dice
    """
    def __init__(self,
                n_class,
                normalize=True,
                ignore_label=-1,
                eps=1e-08,
                dtype=None):
        self.n_class = n_class
        self.normalize = normalize
        self.ignore_label = ignore_label
        self.eps = eps
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'i',
            t_type.dtype.kind == 'i'
        )

        type_check.expect(
            x_type.shape == t_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        b = y.shape[0]
        c = self.n_class

        t_onehot = to_onehot(t, n_class=c, dtype=self.dtype)
        t_onehot = t_onehot.reshape(b, c, -1)

        y_onehot = to_onehot(y, n_class=c, dtype=self.dtype)
        y_onehot = y_onehot.reshape(b, c, -1)

        if self.ignore_label != -1:
            t_onehot = xp.concatenate( (t_onehot[:, :self.ignore_label], t_onehot[:, self.ignore_label + 1:]), axis=1)
            y_onehot = xp.concatenate( (y_onehot[:, :self.ignore_label], y_onehot[:, self.ignore_label + 1:]), axis=1)

        intersection = y_onehot * t_onehot
        cardinality  = y_onehot + t_onehot

        if self.normalize: # NOTE: channel-wise
            intersection = xp.sum(intersection, axis=-1)
            cardinality  = xp.sum(cardinality, axis=-1)
            ret = (2. * intersection / (cardinality + self.eps))
            ret = xp.mean(ret, axis=1)

        else:
            intersection = xp.sum(intersection, axis=(0, 2))
            cardinality  = xp.sum(cardinality, axis=(0, 2))
            ret = (2. * intersection / (cardinality + self.eps))

        return xp.mean(ret) ,


def softmax_discrete_dice(y, t, normalize=True, ignore_label=-1, eps=1e-8):
    """ Dice coefficient with Softmax pre-activates.

    Args:
        y (~chainer.Variable): Logits
        t (~numpy.ndarray or ~cupy.ndarray): Ground-truth label
        normalize (bool, optional): If True, calculate the dice coefficients for each class and take the average. Defaults to True.
        ignore_label (int, optional): Defaults to -1.
        eps (float, optional): Defaults to 1e-08.

    NOTE: This is not a differentiable function.
          See also: ~chainer_bcnn.functions.loss.dice
    """
    dtype = y.dtype
    n_class = y.shape[1]
    y = chainer.functions.argmax(y, axis=1)
    return DiscreteDice(n_class, normalize=normalize,
                        ignore_label=ignore_label,
                        eps=eps, dtype=dtype)(y, t)

def sigmoid_discrete_dice(y, t, eps=1e-8):
    raise NotImplementedError()
