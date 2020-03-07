from __future__ import absolute_import

from .discriminator_base import DiscriminatorBase
from .discriminator_base import conv, crop
from .discriminator_base import _default_conv_param
from .discriminator_base import _default_pool_param
from .discriminator_base import _default_norm_param
from .discriminator_base import _default_activation_param
from .discriminator_base import _default_dropout_param


class PatchDiscriminator(DiscriminatorBase):

    def __init__(self,
                 ndim,
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
            'ksize': 3,
            'stride': 1,
            'pad': 1,
            'nobias': conv_param.get('nobias', False),
            'initialW': conv_param.get('initialW', None),
            'initial_bias': conv_param.get('initial_bias', None),
            'hook': conv_param.get('hook', None),
        }

        with self.init_scope():
            self.add_link('conv_out', conv(ndim, None, out_channels, conv_out_param))


    def forward(self, x):

        h = super().forward(x)

        out = self['conv_out'](h)
        out = crop(out, h.shape)

        return out
