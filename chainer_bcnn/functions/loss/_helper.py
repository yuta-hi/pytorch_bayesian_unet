from __future__ import absolute_import

from chainer import backends

def to_onehot(t, n_class, dtype=None):

    xp = backends.cuda.get_array_module(t)

    if dtype is None:
        dtype = t.dtype

    t_onehot = xp.eye(n_class)[t]
    t_onehot = xp.rollaxis(t_onehot, axis=-1, start=1)
    t_onehot = t_onehot.astype(dtype)

    return t_onehot
