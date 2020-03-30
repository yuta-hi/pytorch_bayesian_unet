from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

from pytorch_trainer.training import StandardUpdater
from pytorch_trainer import reporter
from pytorch_trainer.dataset import convert


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
        model (dict): Generative and discriminative models. It should be a dictionary
            that has `gen` and `dis` keys. Note that `gen` and `dis` means the generator
            and discniminator, respectively.
        alpha (float): Loss scaling factor for balancing the conditional loss.
        converter (optional): Converter function to build input arrays. Defaults to `convert.concat_examples`.
        device (int, optional): Device to which the training data is sent. Negative value
            indicates the host memory (CPU). Defaults to None.
        loss_func: Conditional loss function. `lossfun` attribute of the optimizer's target link for
            the generator is used by default. Defaults to None.
    """



    def __init__(self, iterator, optimizer, model, alpha,
                    converter=convert.concat_examples,
                    device=None, loss_func=None):

        assert isinstance(optimizer, dict)

        super(CGANUpdater, self).__init__(
            iterator, optimizer, model, converter,
            device, loss_func)

        self.alpha = alpha

    @property
    def discriminator(self):
        return self._models['dis']

    @property
    def generator(self):
        return self._models['gen']

    def conditional_lossfun(self, y_fake, y_true):

        model = self.generator

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
