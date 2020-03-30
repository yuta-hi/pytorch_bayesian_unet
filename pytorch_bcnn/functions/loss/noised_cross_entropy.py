from __future__ import absolute_import

import torch

from .softmax_cross_entropy import softmax_cross_entropy
from .sigmoid_cross_entropy import sigmoid_cross_entropy
from .sigmoid_soft_cross_entropy import sigmoid_soft_cross_entropy

def noised_softmax_cross_entropy(y, t, mc_iteration,
                                 normalize=True, class_weight=None,
                                 ignore_label=-1, reduce='mean'):
    """ Softmax Cross-entropy for aleatoric uncertainty estimates.
    See: https://arxiv.org/pdf/1703.04977.pdf

    Args:
        y (list of ~torch.Tensor): logits and sigma
        t (~torch.Tensor): ground-truth
        mc_iteration (int): number of iteration of MCMC.
        normalize (bool, optional): Defaults to True.
        reduce (str, optional): Defaults to 'mean'.

    Returns:
        [~torch.Tensor]: Loss value.
    """

    assert isinstance(y, (list, tuple))

    logits, log_std = y

    assert logits.shape[0]  == log_std.shape[0]
    assert log_std.shape[1] in (logits.shape[1], 1)
    assert logits.shape[2:] == log_std.shape[2:]

    dtype = logits.dtype
    device = logits.device

    ret = []

    # std = torch.sqrt(torch.exp(log_var))
    std = torch.exp(log_std)

    for _ in range(mc_iteration):
        noise = std * torch.empty(std.shape, dtype=dtype, device=device).normal_(0., 1.)
        loss = softmax_cross_entropy(logits + noise, t,
                                     normalize=normalize,
                                     class_weight=class_weight,
                                     ignore_label=ignore_label,
                                     reduce=reduce)
        ret.append(loss[None])

    ret = torch.cat(ret, dim=0)

    if reduce == 'mean':
        return torch.mean(ret)

    return ret


def noised_sigmoid_cross_entropy(y, t, mc_iteration, normalize=True, reduce='mean'):
    """ Sigmoid Cross-entropy for aleatoric uncertainty estimates.

    Args:
        y (list of ~torch.Tensor): logits and sigma
        t (~torch.Tensor): ground-truth
        mc_iteration (int): number of iteration of MCMC.
        normalize (bool, optional): Defaults to True.
        reduce (str, optional): Defaults to 'mean'.

    Returns:
        [~torch.Tensor]: Loss value.
    """
    assert isinstance(y, (list, tuple))

    logits, log_std = y

    assert logits.shape[0] == log_std.shape[0]
    assert log_std.shape[1] in (logits.shape[1], 1)
    assert logits.shape[2:] == log_std.shape[2:]
    assert logits.shape == t.shape

    dtype = logits.dtype
    device = logits.device

    ret = []

    # std = torch.sqrt(torch.exp(log_var))
    std = torch.exp(log_std)

    for _ in range(mc_iteration):
        noise = std * torch.empty(std.shape, dtype=dtype, device=device).normal_(0., 1.)
        loss = sigmoid_cross_entropy(logits + noise, t,
                                     normalize=normalize,
                                     reduce=reduce)
        ret.append(loss[None])

    ret = torch.cat(ret, dim=0)

    if reduce == 'mean':
        return torch.mean(ret)

    return ret


def noised_sigmoid_soft_cross_entropy(y, t, mc_iteration, normalize=True, reduce='mean'):
    """ Sigmoid Soft Cross-entropy for aleatoric uncertainty estimates.

    Args:
        y (list of ~torch.Tensor): logits and sigma
        t (~torch.Tensor): ground-truth
        mc_iteration (int): number of iteration of MCMC.
        normalize (bool, optional): Defaults to True.
        reduce (str, optional): Defaults to 'mean'.

    Returns:
        [~torch.Tensor]: Loss value.
    """
    assert isinstance(y, (list, tuple))

    logits, log_std = y

    assert logits.shape == log_std.shape
    assert logits.shape == t.shape

    dtype = logits.dtype
    device = logits.device

    ret = []

    # std = torch.sqrt(torch.exp(log_var))
    std = torch.exp(log_std)

    for _ in range(mc_iteration):
        noise = std * torch.empty(std.shape, dtype=dtype, device=device).normal_(0., 1.)
        loss = sigmoid_soft_cross_entropy(logits + noise, t,
                                          normalize=normalize,
                                          reduce=reduce)
        ret.append(loss[None])

    ret = torch.cat(ret, dim=0)

    if reduce == 'mean':
        return torch.mean(ret)

    return ret
