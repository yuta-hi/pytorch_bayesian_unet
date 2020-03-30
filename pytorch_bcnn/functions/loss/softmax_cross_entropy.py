from __future__ import absolute_import

import numpy
import torch
import torch.nn.functional as F

_reduce_table = {
    'mean': 'sum',
    'no': 'none',
}

def _check_type_forward(x, t):
    assert t.dim() == x.dim() - 1, 't.dim() != x.dim() - 1..'
    assert x.shape[0] == t.shape[0], 'x.shape[0] != t.shape[0]..'
    assert x.shape[2:] == t.shape[1:], 'x.shape[2:] != t.shape[1:]..'

def softmax_cross_entropy(x, t, normalize=True, class_weight=None,
        ignore_label=-1, reduce='mean'):

    _check_type_forward(x, t)

    _reduce = _reduce_table[reduce]

    log_p = F.log_softmax(x, dim=1)
    loss = F.nll_loss(log_p, t, class_weight, None, ignore_label, None, _reduce)

    if _reduce == 'sum':
        if normalize:
            count = t.numel()
        else:
            count = len(t)
        count = max(count, 1.)

        loss /= count

    return loss
