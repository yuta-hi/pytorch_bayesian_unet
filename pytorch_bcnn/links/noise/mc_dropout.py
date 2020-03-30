from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Dropout):
    """
    Drops elements of input variable randomly.
    This module drops input elements randomly with probability ``p`` and
    scales the remaining elements by factor ``1 / (1 - p)``.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = MCDropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    See the paper by Y. Gal, and G. Zoubin: `Dropout as a bayesian approximation: \
    Representing model uncertainty in deep learning .\
    <https://arxiv.org/abs/1506.02142>`

    See also: A. Kendall: `Bayesian SegNet: Model Uncertainty \
    in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding \
    <https://arxiv.org/abs/1511.02680>`_.
    """

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)
