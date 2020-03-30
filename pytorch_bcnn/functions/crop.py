from __future__ import absolute_import

def contiguous(func):

    def wrap(*args, **kwards):

        ret = func(*args, **kwards)

        return ret.contiguous()

    return wrap


@contiguous
def crop_2d(x, shape):
    left = (x.shape[2] - shape[2]) // 2
    top  = (x.shape[3] - shape[3]) // 2
    right  = left + shape[2]
    bottom = top  + shape[3]
    assert left >= 0 and top >= 0 and \
        right <= x.shape[2] and bottom <= x.shape[3], \
        'Cropping image is less shape than input shape.\n'\
        'Input shape:{}, Cropping shape:{}, (L,R,T,B):({},{},{},{})'.format(
            x.shape, shape, left, right, top, bottom)
    return x[:, :, left:right, top:bottom]


@contiguous
def crop_3d(x, shape):
    left = (x.shape[2] - shape[2]) // 2
    top  = (x.shape[3] - shape[3]) // 2
    near = (x.shape[4] - shape[4]) // 2
    right  = left + shape[2]
    bottom = top  + shape[3]
    far    = near + shape[4]
    assert left >= 0 and top >= 0 and near >= 0 and \
        right <= x.shape[2] and bottom <= x.shape[3] and far <= x.shape[4],\
        'Cropping image is less shape than input shape.\n' \
        'Input shape:{}, Cropping shape:{}, (L,R,T,B,N,F):({},{},{},{},{},{})'.format(
            x.shape, shape, left, right, top, bottom, near, far)
    return x[:, :, left:right, top:bottom, near:far]


@contiguous
def crop_nd(x, shape):
    slices = [slice(0, x.shape[0]), slice(0, x.shape[1])]
    for n in range(2, x.ndim):
        start = (x.shape[n] - shape[n]) // 2
        end = start + shape[n]
        assert start >= 0 and end <= x.shape[n], \
            'Cropping image is less shape than input shape.\n' \
            'Dimension: {}, Cropping shape: {}, (Start, End): ({},{})'.format(
                n, x.shape, start, end)
        slices.append(slice(start, end))
    return x[tuple(slices)]


def crop(x, shape, ndim=None):
    """ Spatial cropping x by given shape

    Args:
        x (ndarray or Variable): Input tensor
        shape (tuple): Desired spatial shape
        ndim (int, optional): Input dimensions. If None, this will be estimated automatically.
                              Defaults to None.

    Returns:
        ndarray or Variable: Cropped tensor
    """

    if ndim is None:
        ndim = x.dim() - 2

    if len(shape) == ndim:
        shape = (None, None, ) + tuple(shape)
    elif len(shape) == (ndim + 2):
        pass
    else:
        raise ValueError('`len(shape)` must be equal to `x.dim` or `x.dim-2`..')

    if x.shape[2:] == shape[2:]:
        return x

    if ndim == 2:
        return crop_2d(x, shape)
    elif ndim == 3:
        return crop_3d(x, shape)

    return crop_nd(x, shape)
