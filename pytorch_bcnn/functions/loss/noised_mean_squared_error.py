from __future__ import absolute_import

from chainer import functions as F

def noised_squared_error(y, t, normalize=False):

    assert isinstance(y, (list,tuple))
    logits, log_var = y

    loss = F.exp(- log_var) * (logits - t)**2. + log_var

    if normalize:
        count = loss.size
    else:
        count = len(loss)

    loss = F.sum(loss / count)

    return loss

def noised_mean_squared_error(y, t):
    """ Mean squared error for aleatoric uncertainty estimates.
    See: https://arxiv.org/pdf/1703.04977.pdf

    Args:
        y (list of ~chainer.Variable): logits and sigma
        t (~numpy.ndarray or ~cupy.ndarray): ground-truth

    Returns:
        [~chainer.Variable]: Loss value.
    """
    return noised_squared_error(y, t, normalize=True)
