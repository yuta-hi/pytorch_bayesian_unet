from __future__ import absolute_import

import copy
import warnings
import chainer
import chainer.functions as F

from .. import Model
from ..unet.unet_base import UNetBaseBlock
from ..unet._helper import conv, _default_conv_param
from ..unet._helper import norm, _default_norm_param
from ..unet._helper import pool, _default_pool_param
from ..unet._helper import activation, _default_activation_param
from ..unet._helper import dropout, _default_dropout_param
from ..unet._helper import initializer
from ...functions import crop


class Block(UNetBaseBlock):
    """ Convolution blocks """
    pass


class DiscriminatorBase(Model):

    def __init__(self,
                 ndim,
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

        super(DiscriminatorBase, self).__init__()

        self._args = locals()

        self._ndim = ndim
        self._nlayer = nlayer

        if isinstance(nfilter, int):
            nfilter = [nfilter*(2**i) for i in range(nlayer)]
        assert len(nfilter) == nlayer
        self._nfilter = nfilter

        if isinstance(ninner, int):
            ninner = [ninner]*nlayer
        assert len(ninner) == nlayer
        self._ninner = ninner

        self._conv_param = conv_param
        self._pool_param = pool_param
        self._norm_param = norm_param
        self._activation_param = activation_param,
        self._dropout_param = dropout_param

        if dropout_enables is None:
            dropout_enables = [True]*nlayer
        assert isinstance(dropout_enables, (list,tuple))
        self._dropout_enables = dropout_enables

        self._residual = residual
        self._preserve_color = preserve_color

        self._pool = pool(pool_param)
        self._activation = activation(activation_param)
        self._dropout = dropout(dropout_param)

        with self.init_scope():

            # down
            for i in range(nlayer):

                self.add_link('block_%d' % i,
                            Block(ndim,
                                  nfilter[i],
                                  conv_param,
                                  None if preserve_color and i == 0 else norm_param,
                                  activation_param,
                                  ninner[i],
                                  residual))

    def forward(self, x):

        h = x

        # down
        for i in range(self._nlayer):

            if i != 0:
                h = self._pool(h)

            h = self['block_%d' % (i)](h)

            if self._dropout_enables[i]:
                h = self._dropout(h)

        return h

