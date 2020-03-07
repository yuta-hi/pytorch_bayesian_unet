from __future__ import absolute_import

from chainer.functions.noise.dropout import Dropout
from chainer.utils import argument

def mc_dropout(x, ratio=.5, **kwargs):
    """mc_dropout(x, ratio=.5)

    Drops elements of input variable randomly.
    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``.
    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        ratio (float):
            Dropout ratio. The ``ratio`` must be ``0.0 <= ratio < 1.0``.
        mask (:ref:`ndarray` or None):
            The mask to be used for dropout.
            You do not have to specify this value, unless you need to make
            results deterministic.
            If ``mask`` is not specified or set to ``None``, a mask will be
            generated randomly according to the given ``ratio``.
            If ``mask`` is specified, ``ratio`` will be ignored.
            The shape and dtype must be the same as ``x`` and should be on the
            same device.
            Note that iDeep and cuDNN will not be used for this function if
            mask is specified, as iDeep and cuDNN do not support it.
        return_mask (bool):
            If ``True``, the mask used for dropout is returned together with
            the output variable.
            The returned mask can later be reused by passing it to ``mask``
            argument.
    Returns:
        ~chainer.Variable or tuple:
            When ``return_mask`` is ``False`` (default), returns the output
            variable.
            When ``True``, returns the tuple of the output variable and
            mask (:ref:`ndarray`). The mask will be on the same device as the
            input.

    See the paper by Y. Gal, and G. Zoubin: `Dropout as a bayesian approximation: \
    Representing model uncertainty in deep learning .\
    <https://arxiv.org/abs/1506.02142>`

    See also: A. Kendall: `Bayesian SegNet: Model Uncertainty \
    in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding \
    <https://arxiv.org/abs/1511.02680>`_.
    """

    mask = None
    return_mask = False
    if kwargs:
        mask, return_mask = argument.parse_kwargs(
            kwargs, ('mask', mask), ('return_mask', return_mask),
            train='train argument is not supported anymore. '
                  'Use chainer.using_config')

    func = Dropout(ratio, mask, return_mask)
    out, = func.apply((x,))
    mask = func.mask

    if return_mask:
        return out, mask
    return out
