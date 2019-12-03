from __future__ import absolute_import

import numpy
import six

import chainer
from chainer import configuration
from chainer import functions
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable

from chainer.links import BatchNormalization

class InstanceNormalization(BatchNormalization):

    """Instance normalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.batch_normalization`

    It normalizes the input by *instance statistics*. It also
    maintains approximated population statistics by moving averages.

    See also :class`~chainer.links.BatchNormalization`

    Args:
        size (int, tuple of ints, or None): Size (or shape) of channel
            dimensions.  If ``None``, the size will be determined from
            dimension(s) of the input batch during the first forward pass.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        axis (int or tuple of int): Axis over which normalization is
            performed. When axis is ``None``, it is determined from input
            dimensions. For example, if ``x.ndim`` is 4, axis becomes (0, 2, 3)
            and normalization is performed over 0th, 2nd and 3rd axis of input.
            If it is 2, axis becomes (0) and normalization is performed
            over 0th axis of input. When a tuple of int is given to this
            option, numbers in the tuple must be being sorted in ascending
            order. For example, (0, 2) is OK, but (2, 0) is not.

        initial_gamma: Initializer of the scaling parameter. The default value
            is ``1``.
        initial_beta: Initializer of the shifting parameter. The default value
            is ``0``.
        initial_avg_mean: Initializer of the moving average of population mean.
            The default value is ``0``.
        initial_avg_var: Initializer of the moving average of population
            variance. The default value is ``1``.

    .. note::

        From v5.0.0, the initial value of the population variance is changed to
        1. It does not change the behavior of training, but the resulting model
        may have a slightly different behavior on inference. To emulate the
        old behavior, pass ``initial_avg_var=0`` for training.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ # TODO:

    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`

    Attributes:
        gamma (~chainer.Variable): Scaling parameter. In mixed16 mode, it is
            initialized as float32 variable.
        beta (~chainer.Variable): Shifting parameter. In mixed16 mode, it is
            initialized as float32 variable.
        avg_mean (:ref:`ndarray`): Population mean. In mixed16 mode, it is
            initialized as float32 array.
        avg_var (:ref:`ndarray`): Population variance. In mixed16 mode, it is
            initialized as float32 array.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
    """

    def __init__(self, size=None, decay=0.9, eps=2e-5, dtype=None,
                 use_gamma=False, use_beta=False,
                 initial_gamma=None, initial_beta=None, axis=None,
                 initial_avg_mean=None, initial_avg_var=None):

        super(InstanceNormalization, self).__init__(
            size, decay, eps, dtype,
            use_gamma, use_beta,
            initial_gamma, initial_beta, axis,
            initial_avg_mean, initial_avg_var,
        )

    def forward(self, x, **kwargs):
        """forward(self, x, finetune=False)

        Invokes the forward propagation of InstanceNormalization.

        The InstanceNormalization computes moving averages of
        mean and variance for evaluation, and normalizes the
        input using batch statistics.

        Args:
            x (~chainer.Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, InstanceNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.

        """
        finetune, = argument.parse_kwargs(
            kwargs, ('finetune', False),
            test='test argument is not supported anymore. '
                 'Use chainer.using_config')

        if self.avg_mean is None:
            param_shape = tuple([
                d
                for i, d in enumerate(x.shape)
                if i not in self.axis])
            self._initialize_params(param_shape)

        gamma = self.gamma
        if gamma is None:
            with chainer.using_device(self.device):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=self._highprec_dtype)

        beta = self.beta
        if beta is None:
            with chainer.using_device(self.device):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=self._highprec_dtype)

        # reshape
        b, ch = x.shape[:2]
        reshaped = functions.reshape(x, (1, b*ch,) + x.shape[2:])

        gamma = self.xp.tile(gamma, (b,))
        beta  = self.xp.tile(beta,  (b,))

        avg_mean = self.xp.tile(self.avg_mean, (b,))
        avg_var  = self.xp.tile(self.avg_var,  (b,))

        if finetune:
            self.N += 1
            decay = 1. - 1. / self.N
        else:
            decay = self.decay

        if chainer.config.in_recomputing:
            # Do not update statistics when extra forward computation is
            # called.
            if finetune:
                self.N -= 1  # Revert the count
            avg_mean = None
            avg_var = None

        ret = functions.batch_normalization(
            reshaped, gamma, beta, eps=self.eps, running_mean=avg_mean,
            running_var=avg_var, decay=decay, axis=self.axis)

        # reshape back
        self.avg_mean, self.avg_var = None, None
        if avg_mean is not None:
            self.avg_mean = avg_mean.reshape(b, ch).mean(axis=0)
        if avg_var is not None:
            self.avg_var = avg_var.reshape(b, ch).mean(axis=0)

        ret = functions.reshape(ret, x.shape)

        return ret
