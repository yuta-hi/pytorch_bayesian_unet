import numpy as np
import chainer

import torch
from torch.nn.functional import cross_entropy

from pytorch_bcnn.functions.loss import softmax_cross_entropy

def test():

    b, c, w, h = 5, 10, 20, 30

    x = np.random.rand(b, c, w, h).astype(np.float32)
    t = np.random.randint(0, c, (b, w, h)).astype(np.int32)

    ret = chainer.functions.softmax_cross_entropy(x, t, normalize=False)
    print(ret.data)

    t = t.astype(np.int64)
    ret = softmax_cross_entropy(torch.as_tensor(x), torch.as_tensor(t), normalize=False)
    print(ret)

    ret = cross_entropy(torch.as_tensor(x), torch.as_tensor(t), reduction='sum')
    print(ret)

if __name__ == '__main__':
    test()
