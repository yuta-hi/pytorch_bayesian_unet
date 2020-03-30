from __future__ import absolute_import

import os
import cv2
import numpy as np
from matplotlib import cm
from functools import partial

from . import Visualizer
from ..inference.inferencer import _variable_to_array

_default_cmap = np.asarray([cm.Set1(i)[:3] for i in range(9)]) # NOTE: RGB format

_default_cmaps = {
    'x': None,
    'y': _default_cmap,
    't': _default_cmap,
}

_default_clims = {
    'x': 'minmax',
    'y': None,
    't': None,
}

_default_transforms = {
    'x': lambda x: x,
    'y': lambda x: np.argmax(x, axis=0),  # NOTE: assume that `y` is logits
    't': lambda x: x,
}


def lut(label, cmap):
    assert np.max(label) <= len(cmap)
    cmap = 255.*cmap.copy()
    cmap = cmap.astype(np.uint8)
    cmap256 = np.zeros((256, 3), np.uint8)
    cmap256[:len(cmap)] = cmap
    im_r = cv2.LUT(label, cmap256[:, 2])  # NOTE: opencv's BGR format
    im_g = cv2.LUT(label, cmap256[:, 1])
    im_b = cv2.LUT(label, cmap256[:, 0])
    im_color = cv2.merge((im_r, im_g, im_b))
    return im_color


def clim(x, param, scale=255.):

    if isinstance(param, str):
        if param == 'minmax':
            param = (np.min(x), np.max(x))
        else:
            raise NotImplementedError('unsupported clim type..')

    assert isinstance(param, (list, tuple))
    norm = (x.astype(np.float32) - param[0]) / (param[1] - param[0])
    return np.clip(norm, 0.0, 1.0, out=norm) * scale


def cast(x):
    return x.astype(np.uint8)


def boundary(image, pad=1, color=(0, 255, 255)):
    image[:pad, :] = color
    image[-pad:, :] = color
    image[:, :pad] = color
    image[:, -pad:] = color
    return image


class ImageVisualizer(Visualizer):
    """ Visualier for two-dimensional images.

    Make the catalog of the input `x`, output `y`, and ground-truth `t`.
    You can set the transform, clip, and clip arguments to help each visualization.
    The default configuration is for segmentation task.

    Args:
        transforms (dict): An dictionary of functions for converting images.
        cmaps (dict): An dictionary of colormap for label images. Note that color format must be RGB order.
        clims (dict): An dictionary of clim values for adjusting the window level and width.
        overlay (bool): If True, the output and ground-truth are overlayed on the input.

    .. note::

        The following are examples in the scenario of segmentation and heatmap regression.

        1. Segmentation

            import numpy as np
            from matplotlib import cm

            cmap = np.asarray([cm.Set1(i)[:3] for i in range(9)])

            visualizer = ImageVisualizer(
                transforms = {
                    'x': lambda x: x,
                    'y': lambda x: np.argmax(x, axis=0),  # NOTE: assume that `y` is logits
                    't': lambda x: x,
                },
                cmaps = {
                    'x': None,
                    'y': cmap,
                    't': cmap,
                },
                clims = {
                    'x': [0, 255],
                    'y': None,
                    't': None,
                }
            )

        2. Heatmap regression

            import numpy as np
            import chainer.functions as F
            import matplotlib.pyplot as plt

            def alpha_blend(heatmaps, cmap='jet'):
                assert heatmaps.ndim == 3

                ch, w, h = heatmaps.shape
                ret = np.zeros((3, w, h))
                mapper = plt.get_cmap(cmap, ch)

                for i in range(ch):
                    color = np.ones((3, w, h)) \
                                * np.asarray(mapper(i)[:3]).reshape(-1,1,1)
                    ret += (color * heatmaps[i])

                return ret

            visualizer = ImageVisualizer(
                transforms = {
                    'x': lambda x: x,
                    'y': lambda x: alpha_blend(F.sigmoid(x[0]).data),
                    't': lambda x: alpha_blend(x),
                },
                cmap = None,
                clims = {
                    'x': 'minmax',
                    'y': (0., 1.),
                    't': (0., 1.),
                }
            )

    """
    def __init__(self,
                 transforms=_default_transforms,
                 cmaps=_default_cmaps,
                 clims=_default_clims,
                 overlay=True):

        super(ImageVisualizer, self).__init__()

        if transforms is None:
            transforms = {}
        if cmaps is None:
            cmaps = {}
        if clims is None:
            clims = {}

        assert isinstance(transforms, dict)
        assert isinstance(cmaps, dict)
        assert isinstance(clims, dict)

        self._cmaps = cmaps
        self._clims = clims
        self._transforms = transforms
        self._overlay = overlay

        self._alpha = 0.2
        self._examples = None

        self.reset()

    def add_example(self, x, y, t):
        x = _variable_to_array(x, True)
        y = _variable_to_array(y, True)
        t = _variable_to_array(t, True)
        self._examples.append([x, y, t])

    def add_batch(self, x, y, t):
        if isinstance(x, tuple):
            x = list(map(list, zip(*x)))
        if isinstance(y, tuple):
            y = list(map(list, zip(*y)))
        if isinstance(t, tuple):
            t = list(map(list, zip(*t)))

        for i in range(x.shape[0]):
            self.add_example(x[i], y[i], t[i])

    def save(self, filename):

        image = self._make_catalog()

        dirname = os.path.dirname(filename)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)
        cv2.imwrite(filename, image)

    def _make_x(self, x):

        # transfrom
        trans = self._transforms.get('x')
        if trans is not None:
            if callable(trans):
                x = trans(x)
            else:
                raise ValueError('transfrom function should be callable..')

        assert x.ndim == 3, x.shape
        x = x.transpose(1, 2, 0)

        # cmap
        param = self._cmaps.get('x')
        if param is not None:
            raise ValueError('cmap for x is unsupported..')

        # clim
        param = self._clims.get('x')
        if param is not None:
            x = clim(x, param)

        # rgb
        if x.shape[-1] not in [1, 3]:
            x = np.mean(x, axis=-1, keepdims=True)
        if x.shape[-1] != 3:
            x = np.repeat(x, 3, axis=2)  # NOTE: to rgb

        x = cast(x)

        return x

    def _make_y(self, y):

        # transfrom
        trans = self._transforms.get('y')
        if trans is not None:
            if callable(trans):
                y = trans(y)
            else:
                raise ValueError('transfrom function should be callable..')

        assert y.ndim in [2, 3], y.shape

        # cmap
        param = self._cmaps.get('y')
        if param is not None:
            y = lut(cast(y), param)
        else:
            # clim
            param = self._clims.get('y')
            if param is not None:
                y = clim(y, param)

            if y.ndim == 2:
                y = np.expand_dims(y, axis=2)
            elif y.ndim == 3:
                y = y.transpose(1, 2, 0)

            if y.shape[-1] not in [1, 3]:
                y = np.mean(y, axis=-1, keepdims=True)
            if y.shape[-1] != 3:
                y = np.repeat(y, 3, axis=2)  # NOTE: to rgb

            y = cast(y)

        return y

    def _make_t(self, t):

        # transfrom
        trans = self._transforms.get('t')
        if trans is not None:
            if callable(trans):
                t = trans(t)
            else:
                raise ValueError('transfrom function should be callable..')

        assert t.ndim in [2, 3], t.shape

        # cmap
        param = self._cmaps.get('t')
        if param is not None:
            t = lut(cast(t), param)
        else:
            # clim
            param = self._clims.get('t')
            if param is not None:
                t = clim(t, param)

            if t.ndim == 2:
                t = np.expand_dims(t, axis=2)
            elif t.ndim == 3:
                t = t.transpose(1, 2, 0)

            if t.shape[-1] not in [1, 3]:
                t = np.mean(t, axis=-1, keepdims=True)
            if t.shape[-1] != 3:
                t = np.repeat(t, 3, axis=2)  # NOTE: to rgb

            t = cast(t)

        return t

    def _make_catalog(self):

        catalog = []

        for (x, y, t) in self._examples:

            x = self._make_x(x)
            y = self._make_y(y)
            t = self._make_t(t)

            assert y.shape == t.shape

            if x.ndim == y.ndim:
                y, t = [y], [t]

            page = []
            for c in range(len(y)):

                if self._overlay:
                    xy = cv2.addWeighted(x, 1. - self._alpha, y[c], self._alpha, 0)
                    xt = cv2.addWeighted(x, 1. - self._alpha, t[c], self._alpha, 0)
                    page.append(
                        np.concatenate([x, y[c], xy, t[c], xt], axis=1))
                else:
                    page.append(
                        np.concatenate([x, y[c], t[c]], axis=1))

            catalog.append(
                boundary(np.concatenate(page, axis=0)))

        return np.concatenate(catalog, axis=0)
