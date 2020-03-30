from __future__ import absolute_import

import numpy
import torch
import torch.nn.functional as F

_reduce_table = {
    'mean': 'sum',
    'no': 'none',
}

def _check_type_forward(x, t):
    assert x.shape == t.shape, 'x.shape != t.shape..'

def sigmoid_cross_entropy(x, t, normalize=True, reduce='mean'):

    _check_type_forward(x, t)

    _reduce = _reduce_table[reduce]

    log1p_exp = torch.log1p(torch.exp(x))
    loss = t * (log1p_exp - x) + (1 - t) * log1p_exp

    if _reduce == 'sum':
        if normalize:
            count = t.numel()
        else:
            count = len(t)
        count = max(count, 1.)

        loss /= count

    return loss
