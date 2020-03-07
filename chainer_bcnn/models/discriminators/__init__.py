from __future__ import absolute_import

from .discriminator_base import DiscriminatorBase
from .patch_discriminator import PatchDiscriminator

_supported_models = {
    'discriminator_base': DiscriminatorBase,
    'patch_discriminator': PatchDiscriminator,
}
