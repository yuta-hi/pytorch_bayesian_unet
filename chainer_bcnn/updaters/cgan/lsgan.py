from __future__ import absolute_import

from chainer import backend
from chainer import reporter
import chainer.functions as F

from .dcgan import DCGANUpdater


class LSGANUpdater(DCGANUpdater):
    """ Updater for Least Square GAN (LSGAN)

    Args:
        iterator: Dataset iterator for the training dataset.
        optimizer (dict): Optimizers to update parameters. It should be a dictionary
            that has `gen` and `dis` keys. Note that `gen` and `dis` means the generator
            and discniminator, respectively.
        alpha (float): Loss scaling factor for balancing the conditional loss.
        buffer_size (int, optional): Size of buffer, which handles the experience replay. Defaults to 0.
        converter (optional): Converter function to build input arrays. Defaults to `convert.concat_examples`.
        device (int, optional): Device to which the training data is sent. Negative value
            indicates the host memory (CPU). Defaults to None.
        loss_func: Conditional loss function. `lossfun` attribute of the optimizer's target link for
            the generator is used by default. Defaults to None.
        auto_new_epoch (bool, optional): If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of optimizers is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``. Defaults to True.

    See also:
        https://arxiv.org/pdf/1611.04076.pdf
    """
    def discriminative_lossfun(self, p_real, p_fake):
        xp = backend.get_array_module(p_fake.array)
        t_1 = xp.ones(p_real.array.shape, dtype=p_real.array.dtype)
        t_0 = xp.zeros(p_fake.array.shape, dtype=p_fake.array.dtype)
        loss = (F.mean_squared_error(p_real, t_1) \
                 + F.mean_squared_error(p_fake, t_0)) * 0.5
        reporter.report({'loss_dis': loss})
        return loss

    def generative_lossfun(self, p_fake):
        xp = backend.get_array_module(p_fake.array)
        t_1 = xp.ones(p_fake.array.shape, dtype=p_fake.array.dtype)
        loss = F.mean_squared_error(p_fake, t_1)
        reporter.report({'loss_gen': loss})
        return loss
