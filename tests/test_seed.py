from chainer_bcnn.utils import fixed_seed


def main():
    import random
    import numpy as np
    import cupy as cp
    import chainer

    print(np.random.rand(10))
    print(cp.random.rand(10))
    print(random.random())
    print(chainer.links.Convolution2D(2, 2, 3, 1, 0).W.data)
    print(chainer.config.cudnn_deterministic)

    print('------')

    with fixed_seed(0, True):
        print(np.random.rand(10))
        print(cp.random.rand(10))
        print(random.random())
        print(chainer.links.Convolution2D(2, 2, 3, 1, 0).W.data)
        print(chainer.config.cudnn_deterministic)


if __name__ == '__main__':
    main()
