from __future__ import absolute_import

import chainer
import chainer.functions as F

from .convolution import ConvolutionND

def _pair(x, ndim=2):
    if hasattr(x, '__getitem__'):
        return x
    return [x]*ndim

class PixelShuffleUpsamplerND(chainer.Chain):
    """Pixel Shuffler for the super resolution.
    This upsampler is effective upsampling method compared with the deconvolution.
    The deconvolution has a problem of the checkerboard artifact.
    A detail of this problem shows the following.
    http://distill.pub/2016/deconv-checkerboard/

    See also:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, ndim, in_channels, out_channels, resolution,
                 ksize=None, stride=1, pad=0, pad_mode='reflect', nobias=False,
                 initialW=None, initial_bias=None):
        super(PixelShuffleUpsamplerND, self).__init__()

        self.ndim = ndim
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pad = _pair(pad, self.ndim)
        self.pad_mode = pad_mode

        with self.init_scope():
            m = self.resolution ** self.ndim
            self.conv = ConvolutionND(
                            ndim, in_channels, out_channels * m,
                            ksize, stride, self.pad, self.pad_mode, nobias,
                            initialW, initial_bias)

    def __call__(self, x):
        r = self.resolution
        out = self.conv(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = self.out_channels

        in_shape = out.shape[2:]
        out_shape = tuple(s * r for s in in_shape)

        r_tuple = tuple(self.resolution for _ in range(self.ndim))
        out = F.reshape(out, (batchsize, out_channels,) + r_tuple + in_shape)
        out = F.transpose(out, self.make_transpose_indices())
        out = F.reshape(out, (batchsize, out_channels, ) + out_shape)
        return out

    def make_transpose_indices(self):
        si = [0, 1]
        si.extend([2 * (i + 1) + 1 for i in range(self.ndim)])
        si.extend([2 * (i + 1) for i in range(self.ndim)])
        return si

class PixelShuffleUpsampler2D(PixelShuffleUpsamplerND):

    def __init__(self, in_channels, out_channels, resolution,
                 ksize=None, stride=1, pad=0, pad_mode='reflect', nobias=False,
                 initialW=None, initial_bias=None):

        super(PixelShuffleUpsampler2D, self).__init__(
            2, in_channels, out_channels, resolution,
            ksize, stride, pad, pad_mode, nobias,
            initialW, initial_bias)

class PixelShuffleUpsampler3D(PixelShuffleUpsamplerND):

    def __init__(self, in_channels, out_channels, resolution,
                 ksize=None, stride=1, pad=0, pad_mode='reflect', nobias=False,
                 initialW=None, initial_bias=None):

        super(PixelShuffleUpsampler3D, self).__init__(
            3, in_channels, out_channels, resolution,
            ksize, stride, pad, pad_mode, nobias,
            initialW, initial_bias)
