from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

from chainer.training import StandardUpdater
from chainer import reporter
from chainer.dataset import convert


def _update(optimizer, in_arrays, loss_func):

    if isinstance(in_arrays, tuple):
        optimizer.update(loss_func, *in_arrays)
    elif isinstance(in_arrays, dict):
        optimizer.update(loss_func, **in_arrays)
    else:
        optimizer.update(loss_func, in_arrays)


class CGANUpdater(StandardUpdater, metaclass=ABCMeta):

    def __init__(self, iterator, optimizer, alpha,
                    converter=convert.concat_examples,
                    device=None, auto_new_epoch=True):

        assert isinstance(optimizer, dict)

        loss_func = loss_scale =None

        super(CGANUpdater, self).__init__(
            iterator, optimizer, converter,
            device, loss_func, loss_scale, auto_new_epoch)

        self.alpha = alpha

    @property
    def discriminator(self):
        optimizer = self._optimizers['dis']
        return optimizer.target

    @property
    def generator(self):
        optimizer = self._optimizers['gen']
        return optimizer.target

    def conditional_lossfun(self, y_fake, y_true):

        optimizer = self._optimizers['gen']
        model = optimizer.target

        if hasattr(model, 'lossfun'):
            lossfun = model.lossfun
        else:
            raise RuntimeError

        loss = lossfun(y_fake, y_true)
        reporter.report({'loss_cond': loss})
        return loss


    @abstractmethod
    def discriminative_lossfun(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def generative_lossfun(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def update_core(self):
        raise NotImplementedError()


from .dcgan import DCGANUpdater # NOQA
from .lsgan import LSGANUpdater # NOQA
