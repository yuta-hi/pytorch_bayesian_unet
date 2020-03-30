from __future__ import absolute_import

import six
import numpy as np
import torch

from ..loss._helper import to_onehot

def _check_type_forward(x, t):
    assert x.shape == t.shape, 'x.shape != t.shape..'


def _discrete_dice(y, t, n_class, normalize=True,
                ignore_label=-1, eps=1e-08):
    """ Dice coefficient
    NOTE: This is not a differentiable function.
        See also: ~pytorch_bcnn.functions.loss.dice
    """
    b = y.shape[0]
    c = n_class

    t_onehot = to_onehot(t, n_class=c)
    t_onehot = t_onehot.view(b, c, -1)

    y_onehot = to_onehot(y, n_class=c)
    y_onehot = y_onehot.view(b, c, -1)

    if ignore_label != -1:
        t_onehot = torch.cat( (t_onehot[:, :ignore_label], t_onehot[:, ignore_label + 1:]), dim=1)
        y_onehot = torch.cat( (y_onehot[:, :ignore_label], y_onehot[:, ignore_label + 1:]), dim=1)

    intersection = y_onehot * t_onehot
    cardinality  = y_onehot + t_onehot

    if normalize: # NOTE: channel-wise
        intersection = torch.sum(intersection, dim=-1)
        cardinality  = torch.sum(cardinality, dim=-1)
        ret = (2. * intersection / (cardinality + eps))
        ret = torch.mean(ret, dim=1)

    else:
        intersection = torch.sum(intersection, dim=(0, 2))
        cardinality  = torch.sum(cardinality, dim=(0, 2))
        ret = (2. * intersection / (cardinality + eps))

    return torch.mean(ret)


def softmax_discrete_dice(y, t, normalize=True, ignore_label=-1, eps=1e-8):
    """ Dice coefficient with Softmax pre-activates.

    Args:
        y (~torch.Tensor): Logits
        t (~torch.Tensor): Ground-truth label
        normalize (bool, optional): If True, calculate the dice coefficients for each class and take the average. Defaults to True.
        ignore_label (int, optional): Defaults to -1.
        eps (float, optional): Defaults to 1e-08.

    NOTE: This is not a differentiable function.
          See also: ~pytorch_bcnn.functions.loss.dice
    """
    n_class = y.shape[1]
    y = torch.argmax(y, dim=1)
    return _discrete_dice(y, t, n_class, normalize=normalize,
                        ignore_label=ignore_label,
                        eps=eps)(y, t)

def sigmoid_discrete_dice(y, t, eps=1e-8):
    raise NotImplementedError()
