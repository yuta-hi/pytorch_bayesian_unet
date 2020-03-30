from __future__ import absolute_import

import numpy as np
import cupy
import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import reporter as reporter_module
from chainer import configuration
from chainer import function
import copy
import six
import tqdm
import sys
import traceback


def _concat_arrays(arrays):
    """Concat CPU and GPU array

    Args:
        arrays (numpy.array or cupy.array): CPU or GPU array
    """
    # cupy
    if isinstance(arrays[0], cupy.ndarray):
        return cupy.concatenate(arrays)

    # numpy
    if not isinstance(arrays[0], np.ndarray):
        arrays = np.asarray(arrays)

    return np.concatenate(arrays)


def _split_predictions(pred):
    """split preditions into list of array(s).
    Args:
        pred (list): A list of preditions.

    Returns:
        List of array(s)
    """
    if len(pred) == 0:
        raise ValueError('prediction is empty')

    first_elem = pred[0]

    if isinstance(first_elem, (tuple, list)):
        result = []

        for i in six.moves.range(len(first_elem)):
            result.append(_concat_arrays([example[i] for example in pred]))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}

        for key in first_elem:
            result[key] = _concat_arrays([example[key] for example in pred])

        return result

    else:
        return _concat_arrays(pred)


def _variable_to_array(var, to_cpu=True):

    if isinstance(var, (tuple, list)):
        array = [v.data if isinstance(v, chainer.Variable) else v for v in var]
        if to_cpu:
            array = [cuda.to_cpu(v) for v in array]

        return tuple(array)

    elif isinstance(var, dict):
        array = {}
        for key, v in var.items():
            arr = v.data if isinstance(v, chainer.Variable) else v
            if to_cpu:
                arr = cuda.to_cpu(arr)
            array[key] = arr

        return array
    else:
        array = var
        if isinstance(array, chainer.Variable):
            array = array.data
        if to_cpu:
            array = cuda.to_cpu(array)

        return array


class Inferencer(object):
    """ The inferencing loop for Chainer.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary that maps strings to iterators.
            If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        model: Model to predict outputs. It can also be a dictionary
            that maps strings to models.
            If this is just an model, then the model is
            registered by the name ``'main'``.
        converter (optional): Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`chainer.dataset.concat_examples` is used
            by default.
        device (int, optional): Device to which the training data is sent. Negative value
            indicates the host memory (CPU). Defaults to None.
        to_cpu (bool, optional): Allow the Cupy's output tensor to be converted to Numpy. Defaults to True.
    """

    def __init__(self, iterator, model,
                 converter=convert.concat_examples,
                 device=None, to_cpu=True):

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(model, dict):
            model = {'main': model}
        self._model = model

        self.observation = {}
        reporter = reporter_module.Reporter()
        for name, target in six.iteritems(self._model):
            reporter.add_observer(name, target)
            reporter.add_observers(
                name, target.namedlinks(skipself=True))
        self.reporter = reporter

        self.converter = converter
        self.device = device
        self.to_cpu = to_cpu

    def get_model(self, name):
        return self._model[name]

    def get_iterator(self, name):
        return self._iterators[name]

    def predict(self, model, batch):
        ret = self.predict_core(model, batch)
        return ret

    def predict_core(self, model, batch):
        in_arrays = self.converter(batch, self.device)

        with function.no_backprop_mode():
            with configuration.using_config('train', False):

                if isinstance(in_arrays, tuple):
                    y = model(*in_arrays)
                elif isinstance(in_arrays, dict):
                    y = model(**in_arrays)
                else:
                    y = model(in_arrays)

        return _variable_to_array(y, to_cpu=self.to_cpu)

    def finalize(self):
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def run(self):
        reporter = self.reporter

        iterator = self._iterators['main']
        model = self._model['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        rets = []

        try:
            for batch in tqdm.tqdm(it, desc='inference',
                                   total=len(it.dataset) // it.batch_size,
                                   ncols=80, leave=False):
                with reporter.scope(self.observation):
                    pred = self.predict(model, batch)
                    rets.append(pred)

        except Exception as e:
            print('Exception in main inference loop: {}'.format(e),
                  file=sys.stderr)
            print('Traceback (most recent call last):', file=sys.stderr)
            traceback.print_tb(sys.exc_info()[2])
            six.reraise(*sys.exc_info())

        finally:
            pass

        return _split_predictions(rets)

    def __del__(self):
        self.finalize()
