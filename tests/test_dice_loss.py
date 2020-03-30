import numpy as np
import torch

from pytorch_bcnn.functions.accuracy.discrete_dice import _discrete_dice
from pytorch_bcnn.functions.loss.dice import dice
from pytorch_bcnn.functions.loss._helper import to_onehot


def _dice(y, t):
    y = y.astype(np.bool)
    t = t.astype(np.bool)

    return 2. * np.logical_and(y, t).sum() / (y.sum() + t.sum())

if __name__ == '__main__':

    n_class = 3

    y = np.random.randint(0, n_class, (10, 100,200)).astype(np.int64)
    t = np.random.randint(0, n_class, (10, 100,200)).astype(np.int64)

    d_all = []
    for i in range(1, n_class):
        d = _dice(y==i, t==i)
        print('class %d:' % i, d)
        d_all.append(d[np.newaxis])

    print('mean:', np.mean(np.concatenate(d_all)))

    y = torch.as_tensor(y)
    t = torch.as_tensor(t)
    d = _discrete_dice(y, t, n_class, normalize=False, ignore_label=0)
    print(d)

    y = to_onehot(y, n_class).float()
    d = dice(y, t, normalize=False, ignore_label=0)
    print(d)
