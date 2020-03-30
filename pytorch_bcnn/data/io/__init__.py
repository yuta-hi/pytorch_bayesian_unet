from __future__ import absolute_import

import numpy as np
import os
import cv2

from . import mhd


def load_image(filename):
    """ Load a two/three dimensional image from given filename

    Args:
        filename (str)

    Returns:
        numpy.ndarray: An image
        list of float: Spacing
    """

    _, ext = os.path.splitext(os.path.basename(filename))

    if ext in ('.mha', '.mhd'):
        [img, img_header] = mhd.read(filename)
        spacing = img_header['ElementSpacing']
        img.flags.writeable = True
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))

    elif ext in ('.png', '.jpg', '.bmp'):
        img = cv2.imread(filename)
        spacing = None

    else:
        raise NotImplementedError()

    return img, spacing


def save_image(filename, image, spacing=None):
    """ Save a two/three dimensional image

    Args:
        filename (str)
        image (numpy.ndarray): A two/three dimensional image
        spacing (list of float, optional): Spacing. Defaults to None.
    """

    dirname = os.path.dirname(filename)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    _, ext = os.path.splitext(os.path.basename(filename))

    if ext in ('.mha', '.mhd'):
        header = {}
        if spacing is not None:
            header['ElementSpacing'] = spacing
        if image.ndim == 2:
            header['TransformMatrix'] = '1 0 0 1'
            header['Offset'] = '0 0'
            header['CenterOfRotation'] = '0 0'
        elif image.ndim == 3:
            image = image.transpose((2, 0, 1))
            header['TransformMatrix'] = '1 0 0 0 1 0 0 0 1'
            header['Offset'] = '0 0 0'
            header['CenterOfRotation'] = '0 0 0'
        else:
            raise NotImplementedError()
        mhd.write(filename, image, header)

    elif ext in ('.png', '.jpg', '.bmp'):
        cv2.imwrite(filename, image)

    else:
        raise NotImplementedError()
