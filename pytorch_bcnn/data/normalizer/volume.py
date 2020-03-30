from __future__ import absolute_import

from . import image

_row_axis = 1
_col_axis = 2
_depth_axis = 3
_channel_axis = 0

assert image._channel_axis == _channel_axis

class Quantize(image.Quantize):
    """ Quantize the given images to specific resolution.

    Non-linearity and overfitting make the neural networks sensitive to tiny noises in high-dimensional data [1].
    Quantizing it to a necessary and sufficient level may be effective, especially for medical images which have >16 bits information.
     [1] Goodfellow et al., "Explaining and harnessing adversarial examples.", 2014. https://arxiv.org/abs/1412.6572

    Args:
        n_bit (int): Number of bits.
        x_min (float, optional): Minimum value in the input domain. Defaults to 0..
        x_max (float, optional): Maximum value in the input domain. Defaults to 1..
        rescale (bool, optional): If True, output value is rescaled to input domain. Defaults to True.
    """
    def __init__(self, n_bit, x_min=0., x_max=1., rescale=True):
        super().__init__(n_bit, x_min, x_max, rescale)
        self._ndim = 3

class Clip(image.Clip):
    """ Clip (limit) the values in given images.

    Args:
        param (tuple or str): Tuple of minimum and maximum values.
            If 'minmax' or 'ch_minmax', minimum and maximum values are automatically estimated.
            'ch_minmax' is the channel-wise minmax normalization.
    """
    def __init__(self, param):
        super().__init__(param)
        self._ndim = 3

class Subtract(image.Subtract):
    """ Subtract a value or tensor from given images.

    Args:
        param (float, numpy.ndarray or str): A value or tensor.
            If 'mean' or 'ch_mean', subtracting values are automatically estimated.
            'ch_mean' is to subtract the channel-wise mean.
    """
    def __init__(self, param):
        super().__init__(param)
        self._ndim = 3

class Divide(image.Divide):
    """ Divide the given images by a value or tensor

    Args:
        param (float, numpy.ndarray or str): A value or tensor.
            If 'std' or 'ch_std', deviding values are automatically estimated.
            'ch_std' is to divide the channel-wise standard deviation.
    """
    def __init__(self, param):
        super().__init__(param)
        self._ndim = 3

