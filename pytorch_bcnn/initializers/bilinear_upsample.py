from __future__ import absolute_import

import numpy as np
import torch


def _kernel_center(ksize):

    center = [None] * len(ksize)
    factor = [None] * len(ksize)

    for i, s in enumerate(ksize):

        factor[i] = (s + 1) // 2
        if s % 2 == 1:
            center[i] = factor[i] - 1
        else:
            center[i] = factor[i] - 0.5

    return center, factor


def _bilinear_kernel_2d(ksize):
    """ Get a kernel upsampling by bilinear interpolation

    Args:
        ksize (list of int): Kernel size.

    Returns:
        numpy.ndarray: A kernel.

    See also:
        https://arxiv.org/pdf/1411.4038.pdf
        https://github.com/d2l-ai/d2l-en/blob/master/chapter_computer-vision/fcn.md#initialize-the-transposed-convolution-layer
    """

    assert len(ksize) == 2

    og = np.ogrid[:ksize[0], :ksize[1]]
    center, factor = _kernel_center(ksize)

    kernel = (1 - abs(og[0] - center[0]) / factor[0]) * \
             (1 - abs(og[1] - center[1]) / factor[1])

    return kernel


def _bilinear_kernel_3d(ksize):

    assert len(ksize) == 3

    og = np.ogrid[:ksize[0], :ksize[1], :ksize[2]]
    center, factor = _kernel_center(ksize)

    kernel = (1 - abs(og[0] - center[0]) / factor[0]) * \
             (1 - abs(og[1] - center[1]) / factor[1]) * \
             (1 - abs(og[2] - center[2]) / factor[2])

    return kernel


def _bilinear_kernel_nd(ksize, dtype=np.float32):

    if len(ksize) == 2:
        kernel = _bilinear_kernel_2d(ksize)
    elif len(ksize) == 3:
        kernel = _bilinear_kernel_3d(ksize)
    else:
        raise NotImplementedError()

    return kernel.astype(dtype)


def bilinear_upsample(tensor, gain=1.):
    """ Initializer of Bilinear upsampling kernel for convolutional weights.
    See also: https://arxiv.org/pdf/1411.4038.pdf
    """
    shape = list(tensor.shape)

    dtype = tensor.dtype
    device = tensor.device

    if shape[0] != shape[1]:
        raise ValueError(
            'The number of input and output channels are NOT same..')

    with torch.no_grad():

        ksize = shape[2:]
        kernel = gain * _bilinear_kernel_nd(ksize)

        weight = np.zeros(shape)
        weight[range(shape[0]), range(shape[1]), ...] = kernel

        tensor[...] = torch.as_tensor(weight, dtype=dtype, device=device)
