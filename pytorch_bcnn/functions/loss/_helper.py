from __future__ import absolute_import

import torch

def to_onehot(t, n_class):

    dtype = t.dtype
    device = t.device

    t_onehot = torch.eye(n_class, dtype=dtype, device=device)[t]

    axes = tuple(range(t_onehot.dim()))
    axes = (axes[0], axes[-1],) + axes[1:-1]

    return t_onehot.permute(axes).contiguous()



