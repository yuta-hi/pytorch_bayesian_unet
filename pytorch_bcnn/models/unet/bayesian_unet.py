from __future__ import absolute_import

import torch.nn.functional as F
from pytorch_trainer import reporter
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


def _check_dropout_param(param):

    name = param['name']
    if name == 'dropout':
        warnings.warn('`%s` is not supported in BayesianUNet.. \
                        Use ``mc_dropout`` instead.' % name)
        param['name'] = 'mc_dropout'


class BayesianUNet(UNetBase):
    """ Bayesian U-Net

    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of input channels.
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
        dropout_enables (list or tuple, optional): Set whether to apply dropout for each layer.
            If None, apply the dropout in all layers.
            Defaults to None.
        residual (bool, optional): Enable the residual learning.
            Defaults to False.
        preserve_color (bool, optional): If True, the normalization will be discarded in the first layer.
            Defaults to False.
        exp_ninner (str, optional): Specify the number of layers in ExpansionBlock.
            If 'same', it is set to the same value as `ninner`.
            Defaults to 'same'.
        exp_norm_param (str, optional): Specify the hyperparameter of normalization layer in ExpansionBlock.
            If 'same', it is set to the same value as `norm_param`.
            Defaults to 'same'.
        exp_activation_param (str, optional): Specify the hyperparameter of normalization layer in ExpansionBlock.
            If 'same', it is set to the same value as `activation_param`.
            Defaults to 'same'.
        exp_dropout_param (str, optional): Specify the hyperparameter of normalization layer in ExpansionBlock.
            If 'same', it is set to the same value as `dropout_param`.
            Defaults to 'same'.

    See also: ~chainer_bcnn.links.mc_sampler
              ~chainer_bcnn.functions.mc_dropout
    """

    def __init__(self,
                 ndim,
                 in_channels,
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
                 dropout_param={'name': 'mc_dropout', 'p': .5,},
                 dropout_enables=None,
                 residual=False,
                 preserve_color=False,
                 exp_ninner='same',
                 exp_norm_param='same',
                 exp_activation_param='same',
                 exp_dropout_param='same',
                ):

        _check_dropout_param(dropout_param)
        if exp_dropout_param != 'same':
            _check_dropout_param(exp_dropout_param)

        return_all_latent = False

        super(BayesianUNet, self).__init__(
                                ndim,
                                in_channels,
                                nlayer,
                                nfilter,
                                ninner,
                                conv_param,
                                pool_param,
                                upconv_param,
                                norm_param,
                                activation_param,
                                dropout_param,
                                dropout_enables,
                                residual,
                                preserve_color,
                                exp_ninner,
                                exp_norm_param,
                                exp_activation_param,
                                exp_dropout_param,
                                return_all_latent)
        self._args = locals()

        if sigma_channels is None:
            sigma_channels = out_channels

        self._out_channels = out_channels
        self._sigma = sigma
        self._sigma_channels = sigma_channels

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

        conv_out_nfilter_in = self._nfilter[0]
        if self._exp_ninner[0] == 0:
            conv_out_nfilter_in += self._nfilter[1]

        self.add_module('conv_out',
                    conv(ndim,
                         conv_out_nfilter_in,
                         out_channels,
                         conv_out_param))


        if sigma:
            conv_sigma_param = {
                'name': 'conv',
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'padding_mode': conv_param.get('padding_mode', 'zeros'),
                'bias': False,
                'initialW': {'name': 'zero'},
                'hook': conv_param.get('hook', None),
            }

            self.add_module('conv_sigma',
                        conv(ndim,
                             conv_out_nfilter_in,
                             sigma_channels,
                             conv_sigma_param))

    def forward(self, x):

        h = super().forward(x)

        out = self['conv_out'](h)
        out = crop(out, x.shape)

        if not self._sigma:
            return out

        sigma = self['conv_sigma'](h)
        sigma = crop(sigma, x.shape)

        reporter.report({'sigma': torch.mean(sigma)}, self)

        return out, sigma
