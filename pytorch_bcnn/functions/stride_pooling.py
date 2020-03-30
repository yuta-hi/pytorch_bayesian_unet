from __future__ import absolute_import

def contiguous(func):

    def wrap(*args, **kwards):

        ret = func(*args, **kwards)

        return ret.contiguous()

    return wrap


def _pair(x, ndim=2):
    if hasattr(x, '__getitem__'):
        return x
    return [x] * ndim


@contiguous
def stride_pooling_2d(x, stride):
    stride = _pair(stride, 2)
    return x[:, :, ::stride[0], ::stride[1]]


@contiguous
def stride_pooling_3d(x, stride):
    stride = _pair(stride, 3)
    return x[:, :, ::stride[0], ::stride[1], ::stride[2]]


def stride_pooling_nd(x, stride):
    """ Spatial pooling by stride.

    Args:
        x (ndarray or Variable): Input tensor
        stride (tuple or int): Stride length

    Returns:
        ndarray or Variable: Output tensor
    """

    ndim = x.dim() - 2

    if ndim == 2:
        return stride_pooling_2d(x, stride)
    elif ndim == 3:
        return stride_pooling_3d(x, stride)
    else:
        raise NotImplementedError('unsupported nd pooling..')
