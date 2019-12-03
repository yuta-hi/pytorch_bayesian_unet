import cupy as cp
import chainer

from chainer_bcnn.functions.accuracy.discrete_dice import DiscreteDice
from chainer_bcnn.functions.loss.dice import dice
from chainer_bcnn.functions.loss._helper import to_onehot


def _dice(y, t):
    y = y.astype(cp.bool)
    t = t.astype(cp.bool)

    return 2. * cp.logical_and(y, t).sum() / (y.sum() + t.sum())

if __name__ == '__main__':

    n_class = 3

    y = cp.random.randint(0, n_class, (10, 100,200)).astype(cp.int32)
    t = cp.random.randint(0, n_class, (10, 100,200)).astype(cp.int32)

    d_all = []
    for i in range(1, n_class):
        d = _dice(y==i, t==i)
        print('class %d:' % i, d)
        d_all.append(d[cp.newaxis])

    print('mean:', cp.mean(cp.concatenate(d_all)))

    d = DiscreteDice(n_class, dtype=cp.float32, normalize=False, ignore_label=0)(y, t)
    print(d)

    y = to_onehot(y, n_class, cp.float32)
    d = dice(y, t, normalize=False, ignore_label=0)
    print(d)
