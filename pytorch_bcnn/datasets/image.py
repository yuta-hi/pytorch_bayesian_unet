from __future__ import absolute_import

import numpy as np
from collections import OrderedDict
from inspect import signature

from . import BaseDataset
from . import convert_to_tensor
from ..data import load_image

_supported_filetypes = [
    'image',
    'label',
    'mask',
]

_default_dtypes = OrderedDict({
    'image': np.float32,
    'label': np.int32,
    'mask': np.uint8,
})

_default_filenames = OrderedDict({
    'image': '{root}/{patient}/image.mha',
    'label': '{root}/{patient}/label.mha',
    'mask': '{root}/{patient}/mask.mha',
})

_default_mask_cvals = OrderedDict({
    'image': 0,
    'label': 0,
})

_channel_axis = 0


def _inspect_n_args(func):
    sig = signature(func)
    return len(sig.parameters)


class ImageDataset(BaseDataset):
    """ Dataset for two-dimensional images

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
    def __init__(self,
                 root,
                 patients=[],
                 classes=None,
                 dtypes=_default_dtypes,
                 filenames=_default_filenames,
                 normalizer=None,
                 augmentor=None,
                 mask_cvals=_default_mask_cvals):

        for key in filenames.keys():
            if key not in _supported_filetypes:
                raise KeyError('unsupported filetype.. <%s>' % key)

        super(ImageDataset, self).__init__(
            root, patients, classes, dtypes,
            filenames, normalizer, augmentor)

        self._mask_cvals = mask_cvals

    def normalize(self, x, y=None):

        # reshape
        if x.ndim == 2:
            x = x[np.newaxis]
        elif x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))  # [c, w, h]

        if y is not None:
            # NOTE: assume that `y` is categorical label
            if y.dtype in [np.int32, np.int64]:
                if y.ndim == 3:
                    if y.shape[-1] in [1, 3]:
                        y = y[:, :, 0]  # NOTE: ad-hoc
                    else:
                        pass

            # NOTE: assume that `y` is continuous label (e.g., heatmap)
            elif y.dtype in [np.float32, np.float64]:
                if y.ndim == 2:
                    y = y[np.newaxis]
                elif y.ndim == 3:
                    y = np.transpose(y, (2, 0, 1))  # [c, w, h]

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

    def masking(self, x, y, mask):

        if x.ndim -1 != mask.ndim:
            mask = np.squeeze(mask, -1)

        x[:, mask==0] = self._mask_cvals['image']
        if y is not None:
            y[mask==0] = self._mask_cvals['label']

        return x, y

    def load_images(self, i):

        images, spacings = {}, {}

        for key in self.files.keys():

            images[key], spacings[key] = \
                load_image(self.files[key][i])

            images[key] = images[key].astype(self.dtypes[key])

        return images, spacings

    @convert_to_tensor
    def get_example(self, i):

        # load
        images, _ = self.load_images(i)

        image = images['image']
        label = images.get('label')
        mask = images.get('mask')

        # transfrom
        image, label = self.normalize(image, label)

        # masking
        if mask is not None:
            image, label = self.masking(image, label, mask)

        # augment
        if self.augmentor is not None:
            if _inspect_n_args(self.augmentor) == 2:
                image, label = self.augmentor(image, label)
            else:
                image = self.augmentor(image)

        # return
        if label is None:
            return image

        return image, label

    def __copy__(self):

        return ImageDataset(
            self._root,
            self._patients,
            self._classes,
            self._dtypes,
            self._filenames,
            self._normalizer,
            self._augmentor,
            self._mask_cvals)
