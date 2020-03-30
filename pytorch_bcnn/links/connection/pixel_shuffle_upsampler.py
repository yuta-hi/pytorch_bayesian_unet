from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleUpsampler2D(nn.Conv2d):
    """Pixel Shuffler for the super resolution.
    This upsampler is effective upsampling method compared with the deconvolution.
    The deconvolution has a problem of the checkerboard artifact.
    A detail of this problem shows the following.
    http://distill.pub/2016/deconv-checkerboard/

    See also:
        https://arxiv.org/abs/1609.05158
    """
    ndim = 2

    def __init__(self, in_channels, out_channels, resolution, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        m = resolution ** self.ndim

        super(PixelShuffleUpsampler2D, self).__init__(
            in_channels, out_channels * m, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

        self.resolution = resolution
        self.out_channels = out_channels

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, resolution={resolution}'
             ', kernel_size={kernel_size}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def forward(self, x):
        r = self.resolution
        out = super().forward(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = self.out_channels

        in_shape = out.shape[2:]
        out_shape = tuple(s * r for s in in_shape)

        r_tuple = tuple(self.resolution for _ in range(self.ndim))
        out = out.view((batchsize, out_channels,) + r_tuple + in_shape)
        out = out.permute(self.make_transpose_indices()).contiguous()
        out = out.view((batchsize, out_channels, ) + out_shape)
        return out

    def make_transpose_indices(self):
        si = [0, 1]
        si.extend([2 * (i + 1) + 1 for i in range(self.ndim)])
        si.extend([2 * (i + 1) for i in range(self.ndim)])
        return si


class PixelShuffleUpsampler3D(nn.Conv3d):
    """Pixel Shuffler for the super resolution.
    This upsampler is effective upsampling method compared with the deconvolution.
    The deconvolution has a problem of the checkerboard artifact.
    A detail of this problem shows the following.
    http://distill.pub/2016/deconv-checkerboard/

    See also:
        https://arxiv.org/abs/1609.05158
    """
    ndim = 3

    def __init__(self, in_channels, out_channels, resolution, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        m = resolution ** self.ndim

        super(PixelShuffleUpsampler3D, self).__init__(
            in_channels, out_channels * m, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

        self.resolution = resolution
        self.out_channels = out_channels

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, resolution={resolution}'
             ', kernel_size={kernel_size}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def forward(self, x):
        r = self.resolution
        out = super().forward(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = self.out_channels

        in_shape = out.shape[2:]
        out_shape = tuple(s * r for s in in_shape)

        r_tuple = tuple(self.resolution for _ in range(self.ndim))
        out = out.view((batchsize, out_channels,) + r_tuple + in_shape)
        out = out.permute(self.make_transpose_indices()).contiguous()
        out = out.view((batchsize, out_channels, ) + out_shape)
        return out

    def make_transpose_indices(self):
        si = [0, 1]
        si.extend([2 * (i + 1) + 1 for i in range(self.ndim)])
        si.extend([2 * (i + 1) for i in range(self.ndim)])
        return si
