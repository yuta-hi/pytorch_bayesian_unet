from __future__ import absolute_import

import os
import six
import copy
import tqdm
import warnings
import numpy as np
from chainer import function
from chainer import configuration
from chainer.dataset import convert
from chainer import reporter as reporter_module
from chainer.training.extensions import Evaluator

from ..visualizer import Visualizer
from ..visualizer import ImageVisualizer

_default_visualizer = ImageVisualizer()


class Validator(Evaluator):
    """ Trainer extension to evaluate and visualize outputs on a validation set.
    This extension evaluates the current models by a given evaluation function.
    It creates a :class:`~chainer.Reporter` object to store values observed in
    the evaluation function on each iteration. The report for all iterations
    are aggregated to :class:`~chainer.DictSummary`. The collected mean values
    are further reported to the reporter object of the trainer, where the name
    of each observation is prefixed by the evaluator name. See
    :class:`~chainer.Reporter` for details in naming rules of the reports.

    The main differences are:
    - There are no optimizers in an evaluator. Instead, it holds links
      to evaluate.
    - An evaluation loop function is used instead of an update function.
    - Preparation routine can be customized, which is called before each
      evaluation. It can be used, e.g., to initialize the state of stateful
      recurrent networks.
    There are two ways to modify the evaluation behavior besides setting a
    custom evaluation function. One is by setting a custom evaluation loop via
    the ``eval_func`` argument. The other is by inheriting this class and
    overriding the :meth:`evaluate` method. In latter case, users have to
    create and handle a reporter object manually. Users also have to copy the
    iterators before using them, in order to reuse them at the next time of
    evaluation. In both cases, the functions are called in testing mode
    (i.e., ``chainer.config.train`` is set to ``False``).
    This extension is called at the end of each epoch by default.
    Args:
        iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        filename (str, optional): Name of the visualization file under the output directory. It can
            be a format string. Defaults to 'validation_iter_{.updater.iteration}'.
        visualizer (~chainer_bcnn.Visualizer, optional): Visualizer for making output catalogs.
            Defaults to `chainer_bcnn.visualizer.ImageVisualizer()`.
        n_vis (int or None, optional): Number of samples for visualization.
            If None, all available samples will be visualized. Defaults to None.
        converter: Converter function to build input arrays.
            :func:`~chainer.dataset.concat_examples` is used by default.
        device: Device to which the validation data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
    Attributes:
        filename: visualization file.
        converter: Converter function.
        visualizer: Visualizer that make a catalog for each evaluation process.
        device: Device to which the validation data is sent.
        n_vis: Number of samples for visualization.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
    See also:
        :class:`~chainer.training.extensions.Evaluator`.
    """


    def __init__(self, iterator, target,
                 filename='validation_iter_{.updater.iteration}',
                 visualizer=_default_visualizer, n_vis=None,
                 converter=convert.concat_examples,
                 device=None,
                 eval_hook=None, eval_func=None):

        super(Validator, self).__init__(iterator, target,
                                        converter,
                                        device, eval_hook, eval_func)

        assert isinstance(visualizer, Visualizer)

        if n_vis is None:
            n_vis = np.inf

        self.filename = filename
        self.visualizer = visualizer
        self.n_vis = n_vis

    def initialize(self, trainer):
        # NOTE: visualize the activations of a model with initial weights
        reporter = reporter_module.Reporter()
        with reporter.scope(trainer.observation):
            self.report(trainer)

    def evaluate(self, trainer):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Note that this function assumes that the main iterator raises
        ``StopIteration`` or code in the evaluation loop raises an exception.
        So, if this assumption is not held, the function could be caught in
        an infinite loop.

        Users can override this method to customize the evaluation routine.

        .. note::

            This method encloses :attr:`eval_func` calls with
            :func:`function.no_backprop_mode` context, so all calculations
            using :class:`~chainer.FunctionNode`\\s inside
            :attr:`eval_func` do not make computational graphs. It is for
            reducing the memory consumption.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
            :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.visualizer.reset()

        desc = 'valid (iter=%08d)' % trainer.updater.iteration
        total = len(it.dataset) // it.batch_size

        for batch in tqdm.tqdm(it, total=total, desc=desc, ncols=80, leave=False):
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)

                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            if self.visualizer.n_examples < self.n_vis:
                if hasattr(eval_func, 'x') \
                        and hasattr(eval_func, 'y') \
                        and hasattr(eval_func, 't'):

                    self.visualizer.add_batch(eval_func.x,
                                              eval_func.y,
                                              eval_func.t)
                else:
                    warnings.warn('`eval_func` should have attributes'
                                  '`x`, `y` and `t` for visualization..')

            summary.add(observation)

        # save
        filename = self.filename
        if callable(filename):
            filename = filename(trainer)
        else:
            filename = filename.format(trainer)

        out = os.path.join(trainer.out, filename)
        self.visualizer.save(out)

        return summary.compute_mean()

    def report(self, trainer):

        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate(trainer)

        reporter_module.report(result)

        return result

    def __call__(self, trainer):
        return self.report(trainer)
