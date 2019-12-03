from __future__ import absolute_import

import chainer
from chainer import link
import chainer.functions as F
import chainer.links as L

def _pair(x, ndim=2):
    if hasattr(x, '__getitem__'):
        return x
    return [x]*ndim

class Deconvolution2D(L.Deconvolution2D):
    """ Deconvolution2D with reflection padding
    """
    # NOTE: out_size is different from original one

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, pad_mode='reflect',
                 nobias=False, initialW=None, initial_bias=None, **kwargs):

        outsize = (None, None)

        super(Deconvolution2D, self).__init__(in_channels, out_channels, ksize, stride, 0,
                                            nobias, outsize, initialW, initial_bias, **kwargs)
        self.pad = _pair(pad)
        self.pad_mode = pad_mode

    def forward(self, x):

        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        y = F.deconvolution_2d(
            x, self.W, self.b, self.stride, 0, self.outsize,
            dilate=self.dilate, groups=self.groups)

        return y

class DeconvolutionND(L.DeconvolutionND):
    """ DeconvolutionND with reflection padding
    """
    def __init__(self, ndim, in_channels, out_channels, ksize=None, stride=1,
                 pad=0, pad_mode='reflect', nobias=False, initialW=None,
                 initial_bias=None, dilate=1, groups=1):

        outsize = None

        super(DeconvolutionND, self).__init__(ndim, in_channels, out_channels, ksize, stride,
                                              0, nobias, outsize, initialW,
                                              initial_bias, dilate, groups)

        self.ndim = ndim
        self.pad = _pair(pad, ndim)
        self.pad_mode = pad_mode

    def forward(self, x):

        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        return F.deconvolution_nd(
            x, self.W, b=self.b, stride=self.stride, pad=0,
            outsize=self.outsize, dilate=self.dilate, groups=self.groups)

class Deconvolution3D(DeconvolutionND):
    """ Deconvolution3D with reflection padding
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, pad_mode='reflect',
                 nobias=False, outsize=None, initialW=None, initial_bias=None,
                 dilate=1, groups=1):
        super(Deconvolution3D, self).__init__(
            3, in_channels, out_channels, ksize, stride, pad, pad_mode, nobias,
            initialW, initial_bias, dilate, groups)
