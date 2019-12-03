import numpy as np
import chainer
from chainer import configuration

from chainer_bcnn.links import MCSampler

def _calc_uncertanty_from_mc_samples(samples):

    mean_pred = samples.mean(axis=0, keepdims=False)
    var_pred = samples.var(axis=0, keepdims=True)

    return mean_pred, var_pred


def main():
    mc_iteration = 10
    mc_samples = np.random.rand(1, 10, 2).astype(np.float32)
    mc_samples = np.repeat(mc_samples, mc_iteration, axis=0)

    _mean, _var = _calc_uncertanty_from_mc_samples(mc_samples)

    print('numpy')
    print(_mean)
    print(_var)
    print('------')

    mean = chainer.functions.mean(mc_samples, axis=0)
    var = mc_samples - mean
    var = chainer.functions.mean(chainer.functions.square(var), axis=0)

    mean = mean.data
    var = var.data

    print('chainer')
    print(mean)
    print(var)

    print((np.abs(mean-_mean)))
    print((np.abs(var-_var)))
    print('------')

    sampler = MCSampler(lambda x: x, mc_iteration, lambda x: x, None, None)
    with configuration.using_config('train', False):
        mean, var = sampler(mc_samples[0])

    print('mc_sampler')
    print(mean)
    print(var)

    print((np.abs(mean-_mean)))
    print((np.abs(var-_var)))
    print('------')

if __name__ == '__main__':
    main()
