from __future__ import absolute_import

import chainer
from chainer import link
import chainer.functions as F
import chainer.links as L

def _pair(x, ndim=2):
    if hasattr(x, '__getitem__'):
        return x
    return [x]*ndim

class Convolution2D(L.Convolution2D):
    """ Convolution2D with reflection padding
    """
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, pad_mode='reflect',
                 nobias=False, initialW=None, initial_bias=None, **kwargs):

        super(Convolution2D, self).__init__(in_channels, out_channels, ksize, stride, 0,
                                            nobias, initialW, initial_bias, **kwargs)
        self.pad = _pair(pad)
        self.pad_mode = pad_mode

    def forward(self, x):

        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        return F.convolution_2d(
            x, self.W, self.b, self.stride, 0, dilate=self.dilate,
            groups=self.groups)

class ConvolutionND(L.ConvolutionND):
    """ ConvolutionND with reflection padding
    """
    def __init__(self, ndim, in_channels, out_channels, ksize=None, stride=1,
                 pad=0, pad_mode='reflect', nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):

        super(ConvolutionND, self).__init__(ndim, in_channels, out_channels, ksize, stride, 0,
                                            nobias, initialW, initial_bias, cover_all, dilate, groups)
        self.ndim = ndim
        self.pad = _pair(pad, ndim)
        self.pad_mode = pad_mode

    def forward(self, x):

        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        return F.convolution_nd(
            x, self.W, self.b, self.stride, 0, cover_all=self.cover_all,
            dilate=self.dilate, groups=self.groups)

class Convolution3D(ConvolutionND):
    """ Convolution3D with reflection padding
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, pad_mode='reflect',
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):

        super(Convolution3D, self).__init__(
            3, in_channels, out_channels, ksize, stride, pad, pad_mode, nobias, initialW,
            initial_bias, cover_all, dilate, groups)

class DeformableConvolution2DSampler(chainer.links.connection.deformable_convolution_2d.DeformableConvolution2DSampler):
    """ DeformableConvolution2DSampler with reflection padding
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, pad_mode='reflect',
                 nobias=False, initialW=None, initial_bias=None):

        super(DeformableConvolution2DSampler, self).__init__(
            in_channels, out_channels, ksize, stride, 0,
            nobias, initialW, initial_bias)

        self.pad = _pair(pad)
        self.pad_mode = pad_mode

    def forward(self, x, offset):
        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        return F.deformable_convolution_2d_sampler(
            x, offset, self.W, self.b, self.stride, 0)

class DeformableConvolution2D(link.Chain):
    """ DeformableConvolution2D with reflection padding
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, pad_mode='reflect',
                 offset_nobias=False, offset_initialW=None,
                 offset_initial_bias=None,
                 deform_nobias=False,
                 deform_initialW=None, deform_initial_bias=None): # TODO: merge argment

        super(DeformableConvolution2D, self).__init__()

        self.pad = _pair(pad)
        self.pad_mode = pad_mode

        kh, kw = _pair(ksize)

        with self.init_scope():
            self.offset_conv = Convolution2D(
                in_channels, 2 * kh * kw, ksize, stride, self.pad, self.pad_mode,
                offset_nobias, offset_initialW, offset_initial_bias)
            self.deform_conv = DeformableConvolution2DSampler(
                in_channels, out_channels, ksize, stride, self.pad, self.pad_mode,
                deform_nobias, deform_initialW, deform_initial_bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class DilatedConvolution2D(L.DilatedConvolution2D):
    """ DilatedConvolution2D with reflection padding
    """
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, pad_mode='reflect',
                 dilate=1, nobias=False, initialW=None, initial_bias=None):

        super(DilatedConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, 0,
            dilate, nobias, initialW, initial_bias
        )

        self.pad = _pair(pad)
        self.pad_mode = pad_mode

    def forward(self, x):

        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        return F.dilated_convolution_2d(
            x, self.W, self.b, self.stride, 0, self.dilate)

class DepthwiseConvolution2D(L.DepthwiseConvolution2D):
    """ DepthwiseConvolution2D with reflection padding
    """
    def __init__(self, in_channels, channel_multiplier, ksize, stride=1, pad=0, pad_mode='reflect',
                 nobias=False, initialW=None, initial_bias=None):

        super(DepthwiseConvolution2D, self).__init__(
            in_channels, channel_multiplier, ksize, stride, 0,
            nobias, initialW, initial_bias
        )

        self.pad = _pair(pad)
        self.pad_mode = pad_mode

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])

        pad_width = [(0,0), (0,0)] + list(map(lambda x: (x, x), self.pad))
        x = F.pad(x, pad_width, self.pad_mode)

        return F.depthwise_convolution_2d(
            x, self.W, self.b, self.stride, 0)
