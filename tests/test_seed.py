from pytorch_bcnn.utils import fixed_seed


def main():
    import random
    import numpy as np
    import torch
    import torch.nn as nn

    print(np.random.rand(10))
    print(random.random())
    print(nn.Conv2d(2, 2, 3, 1, 0).weight.data)
    print(torch.backends.cudnn.deterministic)
    print(torch.backends.cudnn.benchmark)

    print('------')

    with fixed_seed(0, True):
        print(np.random.rand(10))
        print(random.random())
        print(nn.Conv2d(2, 2, 3, 1, 0).weight.data)
        print(torch.backends.cudnn.deterministic)
        print(torch.backends.cudnn.benchmark)

    print(np.random.rand(10))
    print(random.random())
    print(nn.Conv2d(2, 2, 3, 1, 0).weight.data)

    print(torch.backends.cudnn.deterministic)
    print(torch.backends.cudnn.benchmark)

if __name__ == '__main__':
    main()
