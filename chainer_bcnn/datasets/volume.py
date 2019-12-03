from __future__ import absolute_import

import numpy as np

from .image import ImageDataset
from .image import _inspect_n_args

class VolumeDataset(ImageDataset):
    """ Dataset for three-dimensional images

    Args:
        root (str): Directory to the dataset
        patients (list, optional): List of patient names. Defaults to [].
        classes (None or list, optional): List of class names. Defaults to None.
        dtypes (dict, optional): An dictionary of data types.
            Defaults to {'image': np.float32, 'label': np.int32, 'mask': np.uint8}.
        filenames (dict, optional): An dictionary of wildcard to filenames.
            Each filename can be a format string using '{root}' and '{patient}'.
            Defaults to {'image': '{root}/{patient}/image.mha',
             'label': '{root}/{patient}/label.mha', 'mask': '{root}/{patient}/mask.mha'}.
        normalizer (callable, optional): An callable function for normalization. Defaults to None.
        augmentor (callable, optional): An callable function for data augmentation. Defaults to None.
        mask_cvals (dict, optional): Value used for points outside the mask.
            Defaults to {'image': 0, 'label': 0}
    """

    def normalize(self, x, y=None):

        # reshape
        if x.ndim == 3:
            x = x[np.newaxis]
        elif x.ndim == 4:
            x = np.transpose(x, (3, 0, 1, 2))  # [c, w, h, d]

        if y is not None:
            # NOTE: assume that `y` is categorical label
            if y.dtype in [np.int32, np.int64]:
                if y.ndim == 4:
                    if y.shape[-1] == 1:
                        y = y[:, :, :, 0]
                    else:
                        pass

            # NOTE: assume that `y` is continuous label (e.g., heatmap)
            elif y.dtype in [np.float32, np.float64]:
                if y.ndim == 3:
                    y = y[np.newaxis]
                elif y.ndim == 4:
                    y = np.transpose(y, (3, 0, 1, 2))  # [c, w, h, d]

            else:
                raise NotImplementedError('unsupported dtype..')

        # normalizer
        if self.normalizer is not None:
            if _inspect_n_args(self.normalizer) == 2:
                x, y = self.normalizer(x, y)
            else:
                x = self.normalizer(x)

        return x, y

    def denormalize(self, x, y=None):
        raise NotImplementedError()


class VolumeSliceDataset(ImageDataset):
    pass
