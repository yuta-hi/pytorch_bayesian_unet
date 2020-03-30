from __future__ import absolute_import

import os
import six
import numpy

import cupy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.training import extensions

from tensorboardX import SummaryWriter


def _to_cpu(x):
    if isinstance(x, cupy.ndarray):
        x = cuda.to_cpu(x)
    return x


class ParameterStatisticsX(extensions.ParameterStatistics):
    """Trainer extension to report parameter statistics with tensorboardX.

    Statistics are collected and reported for a given :class:`~chainer.Link`
    or an iterable of :class:`~chainer.Link`\\ s. If a link contains child
    links, the statistics are reported separately for each child.

    Any function that takes a one-dimensional :class:`numpy.ndarray` or a
    :class:`cupy.ndarray` and outputs a single or multiple real numbers can be
    registered to handle the collection of statistics, e.g.
    :meth:`numpy.ndarray.mean`.

    The keys of reported statistics follow the convention of link name
    followed by parameter name, attribute name and function name, e.g.
    ``VGG16Layers/conv1_1/W/data/mean``. They are prepended with an optional
    prefix and appended with integer indices if the statistics generating
    function return multiple values.

    Args:
        links (~chainer.Link or iterable of ~chainer.Link): Link(s) containing
            the parameters to observe. The link is expected to have a ``name``
            attribute which is used as a part of the report key.
        statistics (dict or 'default'): Dictionary with function name to
            function mappings.
            The name is a string and is used as a part of the report key. The
            function is responsible for generating the statistics.
            If the special value ``'default'`` is specified, the default
            statistics functions will be used.
        report_params (bool): If ``True``, report statistics for parameter
            values such as weights and biases.
        report_grads (bool): If ``True``, report statistics for parameter
            gradients.
        prefix (str): Optional prefix to prepend to the report keys.
        histogram (bool): If ``True``, histogram are computed.
        trigger: Trigger that decides when to aggregate the results and report
            the values.
        skip_nan_params (bool): If ``True``, statistics are not computed for
            parameters including NaNs and a single NaN value is immediately
            reported instead. Otherwise, this extension will simply try to
            compute the statistics without performing any checks for NaNs.
        log_dir (str): Output directory. If ``None``, log_dir is automatically
            set to `.tensorboard`

    .. note::

       The default statistic functions are as follows:
       * ``'mean'`` (``xp.mean(x)``)
       * ``'std'`` (``xp.std(x)``)
       * ``'min'`` (``xp.min(x)``)
       * ``'max'`` (``xp.max(x)``)
       * ``'zeros'`` (``xp.count_nonzero(x == 0)``)
       * ``'percentile'`` (``xp.percentile(x, \
(0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87))``)

    """
    # prefix ends with a '/' and param_name is preceded by a '/'
    scalar_report_key_template = ('{prefix}{link_name}{param_name}/'
                                  '{attr_name}/{function_name}')

    histgram_report_key_template = (
        '{prefix}{link_name}{param_name}/{attr_name}')

    def __init__(self, links, statistics='default',
                 report_params=True, report_grads=True, prefix=None,
                 histogram=True, trigger=(1, 'epoch'), skip_nan_params=False,
                 log_dir=None,
                 ):

        super(ParameterStatisticsX, self).__init__(
            links, statistics,
            report_params, report_grads, prefix,
            trigger, skip_nan_params)

        self._histogram = histogram
        self._log_dir = log_dir
        self._logger = None

    def initialize(self, trainer):
        # setup log_dir and logger
        if self._log_dir is None:
            self._log_dir = '.tensorboard'
        self._log_dir = os.path.join(trainer.out, self._log_dir)
        self._logger = SummaryWriter(log_dir=self._log_dir)

    def __call__(self, trainer):
        """Execute the statistics extension.

        Collect statistics for the current state of parameters.

        Note that this method will merely update its statistic summary, unless
        the internal trigger is fired. If the trigger is fired, the summary
        will also be reported and then reset for the next accumulation.

        Args:
            trainer (~chainer.training.Trainer): Associated trainer that
                invoked this extension.
        """

        if not self._trigger(trainer):
            return

        statistics = {}

        for link in self._links:
            link_name = getattr(link, 'name', 'None')
            for param_name, param in link.namedparams():
                for attr_name in self._attrs:
                    for function_name, function in \
                            six.iteritems(self._statistics):
                        # Get parameters as a flattend one-dimensional array
                        # since the statistics function should make no
                        # assumption about the axes
                        params = getattr(param, attr_name).ravel()

                        # save as scalar
                        if (self._skip_nan_params and
                                (backend.get_array_module(params).isnan(params).any())):
                            value = numpy.nan
                        else:
                            value = function(params)
                        key = self.scalar_report_key_template.format(
                            prefix=self._prefix + '/' if self._prefix else '',
                            link_name=link_name,
                            param_name=param_name,
                            attr_name=attr_name,
                            function_name=function_name
                        )

                        if (isinstance(value, chainer.get_array_types())
                                and value.size > 1):
                            # Append integer indices to the keys if the
                            # statistic function return multiple values
                            statistics.update({'{}/{}'.format(key, i): v for
                                               i, v in enumerate(value)})
                        else:
                            statistics[key] = value

                        # save as histogram
                        if self._histogram:
                            key = self.histgram_report_key_template.format(
                                prefix=self._prefix + '/' if self._prefix else '',
                                link_name=link_name,
                                param_name=param_name,
                                attr_name=attr_name,
                            )
                            self._logger.add_histogram(
                                key, _to_cpu(params),
                                trainer.updater.iteration)

        for k, v in statistics.items():
            self._logger.add_scalar(k, _to_cpu(v), trainer.updater.iteration)
