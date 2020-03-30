from __future__ import absolute_import

import numpy
import torch
import torch.nn.functional as F
from pytorch_trainer import reporter
from pytorch_trainer.dataset import convert

from . import CGANUpdater
from ._replay_buffer import ReplayBuffer

class DCGANUpdater(CGANUpdater):
    """ Updater for DCGAN

    Args:
        iterator: Dataset iterator for the training dataset.
        optimizer (dict): Optimizers to update parameters. It should be a dictionary
            that has `gen` and `dis` keys. Note that `gen` and `dis` means the generator
            and discniminator, respectively.
        model (dict): Generative and discriminative models. It should be a dictionary
            that has `gen` and `dis` keys. Note that `gen` and `dis` means the generator
            and discniminator, respectively.
        alpha (float): Loss scaling factor for balancing the conditional loss.
        buffer_size (int, optional): Size of buffer, which handles the experience replay. Defaults to 0.
        converter (optional): Converter function to build input arrays. Defaults to `convert.concat_examples`.
        device (int, optional): Device to which the training data is sent. Negative value
            indicates the host memory (CPU). Defaults to None.
        loss_func: Conditional loss function. `lossfun` attribute of the optimizer's target link for
            the generator is used by default. Defaults to None.

    See also:
        https://arxiv.org/pdf/1511.06434.pdf
    """
    def __init__(self, iterator, optimizer, model, alpha, buffer_size=0,
                    converter=convert.concat_examples,
                    device=None, loss_func=None):

        super(DCGANUpdater, self).__init__(
            iterator, optimizer, model, alpha, converter,
            device, loss_func)

        self._buffer = ReplayBuffer(buffer_size)
        self._buffer_size = buffer_size

    def discriminative_lossfun(self, p_real, p_fake):
        size = p_real.numel() / p_real.shape[1]
        loss = (torch.sum(F.softplus(-p_real)) / size \
                + torch.sum(F.softplus(p_fake)) / size) * 0.5 # NOTE: equivalent to binary cross entropy
        reporter.report({'loss_dis': loss})
        return loss

    def generative_lossfun(self, p_fake):
        size = p_fake.numel() / p_fake.shape[1]
        loss = torch.sum(F.softplus(-p_fake)) / size
        reporter.report({'loss_gen': loss})
        return loss

    def update_core(self):

        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        opt_dis = self._optimizers['dis']
        opt_gen = self._optimizers['gen']

        for model in self._models.values():
            model.train()

        x_real, y_real = in_arrays

        # generative
        self.discriminator.requires_grad_(False)

        y_fake = self.generator(x_real)
        xy_fake = torch.cat((x_real, y_fake), dim=1)
        p_fake = self.discriminator(xy_fake)

        loss_gen = self.generative_lossfun(p_fake) \
                    + self.alpha * self.conditional_lossfun(y_fake, y_real)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # discriminative
        # NOTE: deallocate intermediate variable nodes related to the generator
        #       with `detach` method
        self.discriminator.requires_grad_(True)

        y_fake_old = self._buffer(y_fake.detach())

        xy_fake = torch.cat((x_real, y_fake_old), dim=1)
        p_fake = self.discriminator(xy_fake)

        xy_real = torch.cat((x_real, y_real), dim=1)
        p_real = self.discriminator(xy_real)

        loss_dis = self.discriminative_lossfun(p_real, p_fake)

        opt_dis.zero_grad()
        loss_dis.backward()
        opt_dis.step()
