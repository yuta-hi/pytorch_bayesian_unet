from __future__ import absolute_import

from chainer import cuda
from chainer import functions as F
from chainer.functions import sigmoid_cross_entropy
from chainer.functions import softmax_cross_entropy

from .sigmoid_soft_cross_entropy import sigmoid_soft_cross_entropy

def noised_softmax_cross_entropy(y, t, mc_iteration,
                                 normalize=True, cache_score=True, class_weight=None,
                                 ignore_label=-1, reduce='mean', enable_double_backprop=False):
    """ Softmax Cross-entropy for aleatoric uncertainty estimates.
    See: https://arxiv.org/pdf/1703.04977.pdf

    Args:
        y (list of ~chainer.Variable): logits and sigma
        t (~numpy.ndarray or ~cupy.ndarray): ground-truth
        mc_iteration (int): number of iteration of MCMC.
        normalize (bool, optional): Defaults to True.
        reduce (str, optional): Defaults to 'mean'.

    Returns:
        [~chainer.Variable]: Loss value.
    """

    assert isinstance(y, (list, tuple))

    logits, log_std = y

    assert logits.shape == log_std.shape

    xp = cuda.get_array_module(t)

    ret = []

    # std = F.sqrt(F.exp(log_var))
    std = F.exp(log_std)

    for _ in range(mc_iteration):
        noise = std * xp.random.normal(0., 1., std.shape)
        loss = softmax_cross_entropy(logits + noise, t,
                                     normalize=normalize,
                                     cache_score=cache_score,
                                     class_weight=class_weight,
                                     ignore_label=ignore_label,
                                     reduce=reduce,
                                     enable_double_backprop=enable_double_backprop)
        ret.append(loss[None])

    ret = F.concat(ret, axis=0)

    if reduce == 'mean':
        return F.mean(ret)

    return ret


def noised_sigmoid_cross_entropy(y, t, mc_iteration, normalize=True, reduce='mean'):
    """ Sigmoid Cross-entropy for aleatoric uncertainty estimates.

    Args:
        y (list of ~chainer.Variable): logits and sigma
        t (~numpy.ndarray or ~cupy.ndarray): ground-truth
        mc_iteration (int): number of iteration of MCMC.
        normalize (bool, optional): Defaults to True.
        reduce (str, optional): Defaults to 'mean'.

    Returns:
        [~chainer.Variable]: Loss value.
    """
    assert isinstance(y, (list, tuple))

    logits, log_std = y

    assert logits.shape == log_std.shape
    assert logits.shape == t.shape

    xp = cuda.get_array_module(t)

    ret = []

    # std = F.sqrt(F.exp(log_var))
    std = F.exp(log_std)

    for _ in range(mc_iteration):
        noise = std * xp.random.normal(0., 1., std.shape)
        loss = sigmoid_cross_entropy(logits + noise, t,
                                     normalize=normalize,
                                     reduce=reduce)
        ret.append(loss[None])

    ret = F.concat(ret, axis=0)

    if reduce == 'mean':
        return F.mean(ret)

    return ret


def noised_sigmoid_soft_cross_entropy(y, t, mc_iteration, normalize=True, reduce='mean'):
    """ Sigmoid Soft Cross-entropy for aleatoric uncertainty estimates.

    Args:
        y (list of ~chainer.Variable): logits and sigma
        t (~numpy.ndarray or ~cupy.ndarray): ground-truth
        mc_iteration (int): number of iteration of MCMC.
        normalize (bool, optional): Defaults to True.
        reduce (str, optional): Defaults to 'mean'.

    Returns:
        [~chainer.Variable]: Loss value.
    """
    assert isinstance(y, (list, tuple))

    logits, log_std = y

    assert logits.shape == log_std.shape
    assert logits.shape == t.shape

    xp = cuda.get_array_module(t)

    ret = []

    # std = F.sqrt(F.exp(log_var))
    std = F.exp(log_std)

    for _ in range(mc_iteration):
        noise = std * xp.random.normal(0., 1., std.shape)
        loss = sigmoid_soft_cross_entropy(logits + noise, t,
                                          normalize=normalize,
                                          reduce=reduce)
        ret.append(loss[None])

    ret = F.concat(ret, axis=0)

    if reduce == 'mean':
        return F.mean(ret)

    return ret
