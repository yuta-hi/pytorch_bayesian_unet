from __future__ import absolute_import

from .unet_base import UNetBase
from .unet import UNet
from .bayesian_unet import BayesianUNet

_supported_models = {
    'unet_base': UNetBase,
    'unet': UNet,
    'bayesian_unet': BayesianUNet,
}
