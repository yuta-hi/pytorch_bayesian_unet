from __future__ import absolute_import

import torch
import torch.nn.functional as F
from pytorch_trainer import reporter

from .dcgan import DCGANUpdater


class LSGANUpdater(DCGANUpdater):
    """ Updater for Least Square GAN (LSGAN)

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
        https://arxiv.org/pdf/1611.04076.pdf
    """
    def discriminative_lossfun(self, p_real, p_fake):
        t_1 = torch.ones(p_real.shape, dtype=p_real.dtype, device=p_real.device)
        t_0 = torch.zeros(p_fake.shape, dtype=p_fake.dtype, device=p_fake.device)
        loss = (F.mse_loss(p_real, t_1) \
                 + F.mse_loss(p_fake, t_0)) * 0.5
        reporter.report({'loss_dis': loss})
        return loss

    def generative_lossfun(self, p_fake):
        t_1 = torch.ones(p_fake.shape, dtype=p_fake.dtype, device=p_fake.device)
        loss = F.mse_loss(p_fake, t_1)
        reporter.report({'loss_gen': loss})
        return loss
