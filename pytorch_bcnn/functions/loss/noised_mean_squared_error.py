from __future__ import absolute_import

import numpy
import torch

def _check_type_forward(logits, log_var, t):
    assert logits.dim() == t.dim(), 'logits.dim() != t.dim()..'
    assert log_var.dim() == t.dim(), 'log_var.dim() != t.dim()..'
    assert logits.shape == t.shape, 'logits.shape != t.shape..'
    assert log_var.shape[0] == t.shape[0], 'log_var.shape[0] != t.shape[0]..'
    assert log_var.shape[2:] == t.shape[2:], 'log_var.shape[0] != t.shape[0]..'


def noised_squared_error(y, t, normalize=False):

    assert isinstance(y, (list,tuple))
    logits, log_var = y

    _check_type_forward(logits, log_var, t)

    loss = torch.exp(- log_var) * (logits - t)**2. + log_var

    if normalize:
        count = loss.numel()
    else:
        count = len(loss)

    loss = torch.sum(loss / count)

    return loss


def noised_mean_squared_error(y, t):
    """ Mean squared error for aleatoric uncertainty estimates.
    See: https://arxiv.org/pdf/1703.04977.pdf

    Args:
        y (list of ~torch.Tensor): logits and sigma
        t (~torch.Tensor): ground-truth

    Returns:
        [~torch.Tensor]: Loss value.
    """
    return noised_squared_error(y, t, normalize=True)
