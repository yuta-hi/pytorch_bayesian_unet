from __future__ import absolute_import

import torch

from ._helper import to_onehot

def _check_type_forward(x, t):
    assert t.dim() == x.dim() - 1, 't.dim() != x.dim() - 1..'
    assert x.shape[0] == t.shape[0], 'x.shape[0] != t.shape[0]..'
    assert x.shape[2:] == t.shape[1:], 'x.shape[2:] != t.shape[1:]..'


def jaccard(y, t, normalize=True, class_weight=None,
            ignore_label=-1, reduce='mean', eps=1e-08):
    """ Differentable Jaccard index.

    Args:
        y (~torch.Tensor): Probability
        t (~torch.Tensor): Ground-truth label
        normalize (bool, optional): If True, calculate the jaccard indices for each class and take the average. Defaults to True.
        class_weight (list or ndarray, optional): Defaults to None.
        ignore_label (int, optional): Defaults to -1.
        reduce (str, optional): Defaults to 'mean'.
        eps (float, optional): Defaults to 1e-08.
    """
    _check_type_forward(y, t)

    device = y.device
    dtype = y.dtype

    if class_weight is not None:
        class_weight = torch.as_tensor(class_weight, dtype=dtype, device=device)

    b, c = y.shape[:2]
    t_onehot = to_onehot(t, n_class=c)

    y = y.view(b, c, -1)
    t_onehot = t_onehot.view(b, c, -1)

    if ignore_label != -1:
        t_onehot = torch.cat( (t_onehot[:, :ignore_label], t_onehot[:, ignore_label + 1:]), dim=1)
        y = torch.cat( (y[:, :ignore_label], y[:, ignore_label + 1:]), dim=1)

    intersection = y * t_onehot
    cardinality = y + t_onehot

    if normalize:  # NOTE: channel-wise
        intersection = torch.sum(intersection, dim=-1)
        cardinality = torch.sum(cardinality, dim=-1)
        union = cardinality - intersection
        ret = (2. * intersection / (union + eps))
        if class_weight is not None:
            ret *= class_weight
        ret = torch.mean(ret, dim=1)

    else:
        intersection = torch.sum(intersection, dim=(0, 2))
        cardinality = torch.sum(cardinality, dim=(0, 2))
        union = cardinality - intersection
        ret = (2. * intersection / (union + eps))
        if class_weight is not None:
            ret *= class_weight

    if reduce == 'mean':
        ret = torch.mean(ret)
    else:
        raise NotImplementedError('unsupported reduce type..')

    return ret


def softmax_jaccard(y, t, normalize=True, class_weight=None,
                   ignore_label=-1, reduce='mean', eps=1e-08):
    """ Differentable Jaccard index with Softmax pre-activates.

    Args:
        y (~torch.Tensor): Probability
        t (~torch.Tensor): Ground-truth label
        normalize (bool, optional): If True, calculate the jaccard indices for each class and take the average. Defaults to True.
        class_weight (list or ndarray, optional): Defaults to None.
        ignore_label (int, optional): Defaults to -1.
        reduce (str, optional): Defaults to 'mean'.
        eps (float, optional): Defaults to 1e-08.
    """
    y = torch.softmax(y, dim=1)
    return jaccard(y, t, normalize, class_weight,
                   ignore_label, reduce, eps)


def softmax_jaccard_loss(y, t, normalize=True, class_weight=None,
                         ignore_label=-1, reduce='mean', eps=1e-08):
    """ Differentable Jaccard-index loss with Softmax pre-activates.

    Args:
        y (~torch.Tensor): Probability
        t (~torch.Tensor): Ground-truth label
        normalize (bool, optional): If True, calculate the jaccard indices for each class and take the average. Defaults to True.
        class_weight (list or ndarray, optional): Defaults to None.
        ignore_label (int, optional): Defaults to -1.
        reduce (str, optional): Defaults to 'mean'.
        eps (float, optional): Defaults to 1e-08.
    """
    return 1.0 - softmax_jaccard(y, t, normalize, class_weight,
                                 ignore_label, reduce, eps)


def sigmoid_jaccard(y, t, *args, **kwards):
    raise NotImplementedError()


def sigmoid_jaccard_loss(y, t, *args, **kwards):
    raise NotImplementedError()
