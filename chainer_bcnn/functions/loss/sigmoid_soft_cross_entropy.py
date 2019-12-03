from __future__ import absolute_import

import numpy as np
from chainer.functions.loss import sigmoid_cross_entropy
from chainer.utils import type_check


class SigmoidSoftCrossEntropy(sigmoid_cross_entropy.SigmoidCrossEntropy):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            t_type.dtype == np.float32,
            x_type.shape == t_type.shape
        )


def sigmoid_soft_cross_entropy(x, t, normalize=True, reduce='mean'):
    return SigmoidSoftCrossEntropy(normalize, reduce).apply((x, t))[0]
