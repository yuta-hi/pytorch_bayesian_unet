from __future__ import absolute_import

from abc import ABCMeta, abstractmethod


class Visualizer(metaclass=ABCMeta):
    """ Base class of visualizers
    """

    def __init__(self, *args, **kwargs):
        self._examples = None
        self.reset()

    def reset(self):
        self._examples = []

    @property
    def n_examples(self):
        return len(self._examples)

    @abstractmethod
    def add_example(self, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def add_batch(self, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError()


from .image import ImageVisualizer  # NOQA
