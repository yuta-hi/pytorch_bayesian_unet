from __future__ import absolute_import

import numpy as np
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import json

_channel_axis = 0


class DataAugmentor(object):
    """ Data augmentor for image and volume data.
    This class manages the operations.
    """

    def __init__(self, n_dim=None):

        assert n_dim is None or n_dim in [2, 3]

        self._n_dim = n_dim
        self._operations = []

    def add(self, op):

        assert isinstance(op, Operation)
        if self._n_dim is None:  # NOTE: auto set
            self._n_dim = op.ndim
        self._operations.append(op)

    def get(self):
        return self._operations

    def preprocess(self, x):

        is_expanded = False

        if x is not None:
            if isinstance(x, list):
                if x[0].ndim == self._n_dim:
                    x = [np.expand_dims(x_i, _channel_axis) for x_i in x]
                    is_expanded = True
                assert x[0].ndim == self._n_dim + 1, '`x[0].ndim` must be `self._n_dim + 1`'
            else:
                if x.ndim == self._n_dim:
                    x = np.expand_dims(x, _channel_axis)
                    is_expanded = True
                assert x.ndim == self._n_dim + 1, '`x.ndim` must be `self._n_dim + 1`'

        return x, is_expanded

    def postprocess(self, x, is_expanded):

        if not is_expanded:
            return x

        if x is not None:
            if isinstance(x, list):
                x = [np.rollaxis(x_i, _channel_axis, 0)[0] for x_i in x]
            else:
                x = np.rollaxis(x, _channel_axis, 0)[0]
        return x

    def apply(self, x=None, y=None):

        x, is_expanded_x = self.preprocess(x)
        y, is_expanded_y = self.preprocess(y)

        for op in self._operations:
            x, y = op.apply(x, y)

        x = self.postprocess(x, is_expanded_x)
        y = self.postprocess(y, is_expanded_y)

        assert(x is not None or y is not None)
        if x is None:
            return y
        if y is None:
            return x
        return x, y

    def __call__(self, x=None, y=None):
        return self.apply(x, y)

    def summary(self, out=None):

        ret = OrderedDict()

        for op in self._operations:
            name = op.__class__.__name__
            ret[name] = op.summary()

        if out is None:
            return ret

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(ret, f, ensure_ascii=False, indent=4)

        return ret


class Operation(metaclass=ABCMeta):
    """ Base class of operations
    """

    def preprocess_core(self, x):
        if x is None:
            return x
        elif isinstance(x, list):
            return x
        else:
            return [x]  # NOTE: to list

    def preprocess(self, x, y):
        x = self.preprocess_core(x)
        y = self.preprocess_core(y)
        return x, y

    def postprocess_core(self, x):
        if x is None:
            return x
        elif len(x) == 1:
            return x[0]
        else:
            return x

    def postprocess(self, x, y):
        x = self.postprocess_core(x)
        y = self.postprocess_core(y)
        return x, y

    @abstractmethod
    def apply_core(self, x, y):
        raise NotImplementedError()

    def apply(self, x=None, y=None):
        x, y = self.preprocess(x, y)
        x, y = self.apply_core(x, y)
        x, y = self.postprocess(x, y)
        return x, y

    @property
    @abstractmethod
    def ndim(self):
        raise NotImplementedError()

    @abstractmethod
    def summary(self):
        raise NotImplementedError()


from .image import Flip as Flip2D  # NOQA
from .image import Crop as Crop2D  # NOQA
from .image import Affine as Affine2D  # NOQA

from .volume import Flip as Flip3D  # NOQA
from .volume import Crop as Crop3D  # NOQA
from .volume import Affine as Affine3D  # NOQA
