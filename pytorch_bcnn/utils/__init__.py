from __future__ import absolute_import

import os
import numpy as np
import contextlib
import warnings

@contextlib.contextmanager
def fixed_seed(seed, strict=False):
    """Fix random seed to improve the reproducibility.

    Args:
        seed (float): Random seed
        strict (bool, optional): If True, cuDNN works under deterministic mode.
            Defaults to False.

    TODO: Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`.
          If your dataset has stochastic behavior, such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.
    """

    import random
    import chainer

    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

    if strict:
        warnings.warn('Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`. \
          If your dataset has stochastic behavior such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.')

    with chainer.using_config('cudnn_deterministic', strict):
        yield

    pass

def find_latest_snapshot(fmt, path, return_fullpath=True):
    '''Alias of :func:`_find_latest_snapshot`
    '''
    from chainer.training.extensions._snapshot \
        import _find_latest_snapshot

    ret = _find_latest_snapshot(fmt, path)

    if ret is None:
        raise FileNotFoundError('cannot find snapshot for <%s>' %
                                    os.path.join(path, fmt))

    if return_fullpath:
        return os.path.join(path, ret)

    return ret
