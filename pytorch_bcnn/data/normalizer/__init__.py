from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

from .. import augmentor

class Normalizer(augmentor.DataAugmentor):

    def apply(self, x):

        x, is_expanded_x = self.preprocess(x)

        for op in self._operations:
            x = op.apply(x)

        x = self.postprocess(x, is_expanded_x)

        return x

    def __call__(self, x):
        return self.apply(x)

class Operation(augmentor.Operation):

    def preprocess(self, x):
        x = self.preprocess_core(x)
        return x

    def postprocess(self, x):
        x = self.postprocess_core(x)
        return x

    @abstractmethod
    def apply_core(self, x):
        raise NotImplementedError()

    def apply(self, x):
        x = self.preprocess(x)
        x = self.apply_core(x)
        x = self.postprocess(x)
        return x

from .image import Quantize as Quantize2D  # NOQA
from .image import Clip as Clip2D  # NOQA
from .image import Subtract as Subtract2D  # NOQA
from .image import Divide as Divide2D  # NOQA

from .volume import Quantize as Quantize3D  # NOQA
from .volume import Clip as Clip3D  # NOQA
from .volume import Subtract as Subtract3D  # NOQA
from .volume import Divide as Divide3D  # NOQA
