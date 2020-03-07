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
    """ Base class of updater for conditional GANs

    Args:
        iterator: Dataset iterator for the training dataset.
        optimizer (dict): Optimizers to update parameters. It should be a dictionary
            that has `gen` and `dis` keys. Note that `gen` and `dis` means the generator
            and discniminator, respectively.
        alpha (float): Loss scaling factor for balancing the conditional loss.
        converter (optional): Converter function to build input arrays. Defaults to `convert.concat_examples`.
        device (int, optional): Device to which the training data is sent. Negative value
            indicates the host memory (CPU). Defaults to None.
        loss_func: Conditional loss function. `lossfun` attribute of the optimizer's target link for
            the generator is used by default. Defaults to None.
        auto_new_epoch (bool, optional): If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of optimizers is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``. Defaults to True.
    """



    def __init__(self, iterator, optimizer, alpha,
                    converter=convert.concat_examples,
                    device=None, loss_func=None, auto_new_epoch=True):

        assert isinstance(optimizer, dict)

        loss_scale = None

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
            lossfun = self.loss_func

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
