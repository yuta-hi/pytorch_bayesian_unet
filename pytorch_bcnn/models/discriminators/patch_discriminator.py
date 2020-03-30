from __future__ import absolute_import

from .discriminator_base import DiscriminatorBase
from .discriminator_base import conv, crop
from .discriminator_base import _default_conv_param
from .discriminator_base import _default_pool_param
from .discriminator_base import _default_norm_param
from .discriminator_base import _default_activation_param
from .discriminator_base import _default_dropout_param


class PatchDiscriminator(DiscriminatorBase):
    """ Patch based discriminator (Markovian discriminator)

    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        nlayer (int, optional): Number of layers.
            Defaults to 4.
        nfilter (list or int, optional): Number of filters.
            Defaults to 64.
        ninner (list or int, optional): Number of layers in UNetBlock.
            Defaults to 1.
        conv_param (dict, optional): Hyperparameter of convolution layer.
            Defaults to {'name':'conv', 'ksize': 3, 'stride': 1, 'pad': 1,
             'initialW': {'name': 'he_normal', 'scale': 1.0}, 'initial_bias': {'name': 'zero'}}.
        pool_param (dict, optional): Hyperparameter of pooling layer.
            Defaults to {'name': 'max', 'ksize': 2, 'stride': 2}.
        norm_param (dict or None, optional): Hyperparameter of normalization layer.
            Defaults to {'name': 'batch'}.
        activation_param (dict, optional): Hyperparameter of activation layer.
            Defaults to {'name': 'relu'}.
        dropout_param (dict or None, optional): Hyperparameter of dropout layer.
            Defaults to {'name': 'dropout', 'ratio': .5}.
        dropout_enables (list or tuple, optional): Set whether to apply dropout for each layer.
            If None, apply the dropout in all layers.
            Defaults to None.
        residual (bool, optional): Enable the residual learning.
            Defaults to False.
        preserve_color (bool, optional): If True, the normalization will be discarded in the first layer.
            Defaults to False.

    See: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self,
                 ndim,
                 in_channels,
                 out_channels,
                 nlayer=4,
                 nfilter=64,
                 ninner=1,
                 conv_param=_default_conv_param,
                 pool_param=_default_pool_param,
                 norm_param=_default_norm_param,
                 activation_param=_default_activation_param,
                 dropout_param=_default_dropout_param,
                 dropout_enables=None,
                 residual=False,
                 preserve_color=False
                ):

        super(PatchDiscriminator, self).__init__(
                                ndim,
                                in_channels,
                                nlayer,
                                nfilter,
                                ninner,
                                conv_param,
                                pool_param,
                                norm_param,
                                activation_param,
                                dropout_param,
                                dropout_enables,
                                residual,
                                preserve_color)
        self._args = locals()

        self._out_channels = out_channels

        conv_out_param = {
            'name': 'conv',
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'padding_mode': conv_param.get('padding_mode', 'zeros'),
            'bias': conv_param.get('bias', True),
            'initialW': conv_param.get('initialW', None),
            'initial_bias': conv_param.get('initial_bias', None),
            'hook': conv_param.get('hook', None),
        }

        self.add_module('conv_out',
                    conv(ndim,
                         self._nfilter[-1],
                         out_channels,
                         conv_out_param))

    def forward(self, x):

        h = super().forward(x)

        out = self['conv_out'](h)
        out = crop(out, h.shape)

        return out
