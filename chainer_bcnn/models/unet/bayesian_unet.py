from __future__ import absolute_import

import chainer
import chainer.functions as F
from chainer import reporter
import warnings

from .unet_base import UNetBase
from ._helper import conv
from ._helper import _default_conv_param
from ._helper import _default_norm_param
from ._helper import _default_upconv_param
from ._helper import _default_pool_param
from ._helper import _default_activation_param
from ._helper import _default_dropout_param
from ...functions import crop

class BayesianUNet(UNetBase):
    """ Bayesian U-Net

    Args:
        ndim (int): Number of spatial dimensions.
        out_channels (int): Number of output channels.
        nlayer (int, optional): Number of layers.
            Defaults to 5.
        nfilter (list or int, optional): Number of filters.
            Defaults to 32.
        ninner (list or int, optional): Number of layers in UNetBlock.
            Defaults to 2.
        sigma (bool, optional): If True, the network concurrently outputs the sigma.
            Defaults to False.
        sigma_channels (int or None, optional): Number of channels for the sigma.
            If None, this is set equal to number of output channels, automatically.
            Defaults to None.
        conv_param (dict, optional): Hyperparameter of convolution layer.
            Defaults to {'name':'conv', 'ksize': 3, 'stride': 1, 'pad': 1,
             'initialW': {'name': 'he_normal', 'scale': 1.0}, 'initial_bias': {'name': 'zero'}}.
        pool_param (dict, optional): Hyperparameter of pooling layer.
            Defaults to {'name': 'max', 'ksize': 2, 'stride': 2}.
        upconv_param (dict, optional): Hyperparameter of up-convolution layer.
            Defaults to {'name':'deconv', 'ksize': 3, 'stride': 2, 'pad': 0,
             'initialW': {'name': 'bilinear', 'scale': 1.0}, 'initial_bias': {'name': 'zero'}}.
        norm_param (dict or None, optional): Hyperparameter of normalization layer.
            Defaults to {'name': 'batch'}.
        activation_param (dict, optional): Hyperparameter of activation layer.
            Defaults to {'name': 'relu'}.
        dropout_param (dict or None, optional): Hyperparameter of dropout layer.
            Defaults to {'name': 'mc_dropout', 'ratio': .5,}.
        residual (bool, optional): Enable the residual learning.
            Defaults to False.

    See also: ~chainer_bcnn.links.mc_sampler
              ~chainer_bcnn.functions.mc_dropout
    """

    def __init__(self,
                 ndim,
                 out_channels,
                 nlayer=5,
                 nfilter=32,
                 ninner=2,
                 sigma=False,
                 sigma_channels=None,
                 conv_param=_default_conv_param,
                 pool_param=_default_pool_param,
                 upconv_param=_default_upconv_param,
                 norm_param=_default_norm_param,
                 activation_param=_default_activation_param,
                 dropout_param={'name': 'mc_dropout', 'ratio': .5,},
                 residual=False,
                ):

        dropout_name = dropout_param['name']
        if dropout_name != 'mc_dropout':
            warnings.warn('Currently, %s is not supported in BayesianUNet.. \
                           Use ``mc_dropout`` instead.' % dropout_name)
            dropout_param['name'] = 'mc_dropout'

        return_all_latent = False

        super(BayesianUNet, self).__init__(
                                ndim,
                                nlayer,
                                nfilter,
                                ninner,
                                conv_param,
                                pool_param,
                                upconv_param,
                                norm_param,
                                activation_param,
                                dropout_param,
                                residual,
                                return_all_latent)
        self._args = locals()

        if sigma_channels is None:
            sigma_channels = out_channels

        self._out_channels = out_channels
        self._sigma = sigma
        self._sigma_channels = sigma_channels

        conv_out_param = {
            'name': 'conv',
            'ksize': 3,
            'stride': 1,
            'pad': 1,
            'nobias': False,
            'initialW': conv_param.get('initialW'),
            'initial_bias': conv_param.get('initial_bias'),
        }

        with self.init_scope():
            self.add_link('conv_out', conv(ndim, None, out_channels, conv_out_param))


        if sigma:
            conv_sigma_param = {
                'name': 'conv',
                'ksize': 3,
                'stride': 1,
                'pad': 1,
                'nobias': True,
                'initialW': {'name': 'zero'},
            }

            with self.init_scope():
                self.add_link('conv_sigma', conv(ndim, None, sigma_channels, conv_sigma_param))

    def forward(self, x):

        h = super().forward(x)

        out = self['conv_out'](h)
        out = crop(out, x.shape)

        if not self._sigma:
            return out

        sigma = self['conv_sigma'](h)
        sigma = crop(sigma, x.shape)

        reporter.report({'sigma': F.mean(sigma)}, self)

        return out, sigma
