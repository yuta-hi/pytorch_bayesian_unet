from __future__ import absolute_import

from chainer import backend
from chainer import reporter
import chainer.functions as F

from .dcgan import DCGANUpdater


class LSGANUpdater(DCGANUpdater):

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
