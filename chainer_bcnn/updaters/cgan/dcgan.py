from __future__ import absolute_import

import chainer.functions as F
from chainer import reporter
from chainer.dataset import convert

from . import CGANUpdater
from ._replay_buffer import ReplayBuffer

class DCGANUpdater(CGANUpdater):
    """ Updater for DCGAN

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
        auto_new_epoch (bool, optional): If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of optimizers is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``. Defaults to True.

    See also:
        https://arxiv.org/pdf/1511.06434.pdf
    """
    def __init__(self, iterator, optimizer, alpha, buffer_size=0,
                    converter=convert.concat_examples,
                    device=None, auto_new_epoch=True):

        super(DCGANUpdater, self).__init__(
            iterator, optimizer, alpha, converter,
            device, auto_new_epoch)

        self._buffer = ReplayBuffer(buffer_size)
        self._buffer_size = buffer_size

    def discriminative_lossfun(self, p_real, p_fake):
        size = p_real.size // p_real.shape[1]
        loss = (F.sum(F.softplus(-p_real)) / size \
                + F.sum(F.softplus(p_fake)) / size) * 0.5 # NOTE: equivalent to binary cross entropy
        reporter.report({'loss_dis': loss})
        return loss

    def generative_lossfun(self, p_fake):
        size = p_fake.size // p_fake.shape[1]
        loss = F.sum(F.softplus(-p_fake)) / size
        reporter.report({'loss_gen': loss})
        return loss

    def update_core(self):

        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        opt_dis = self._optimizers['dis']
        opt_gen = self._optimizers['gen']

        x_real, y_real = in_arrays

        # generative
        y_fake = self.generator(x_real)
        xy_fake = F.concat((x_real, y_fake))
        p_fake = self.discriminator(xy_fake)

        loss_gen = self.generative_lossfun(p_fake) \
                    + self.alpha * self.conditional_lossfun(y_fake, y_real)

        self.generator.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        # discriminative
        # NOTE: deallocate intermediate variable nodes related to the generator
        #       with `array` method instead of `unchain_backward`
        y_fake_old = self._buffer(y_fake.array)

        xy_fake = F.concat((x_real, y_fake_old))
        p_fake = self.discriminator(xy_fake)

        xy_real = F.concat((x_real, y_real))
        p_real = self.discriminator(xy_real)

        loss_dis = self.discriminative_lossfun(p_real, p_fake)

        self.discriminator.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        if self.auto_new_epoch and iterator.is_new_epoch:
            opt_gen.new_epoch(auto=True)
            opt_dis.new_epoch(auto=True)
