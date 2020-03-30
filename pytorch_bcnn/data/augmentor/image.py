from __future__ import absolute_import

import numpy as np
import scipy.ndimage as ndi
import cv2

from . import Operation

_row_axis = 1
_col_axis = 2
_channel_axis = 0


def flip(x, axis):
    if x is None:
        return x
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class Flip(Operation):
    """ Reverse the order of elements in given images along the specificed axis, stochastically.

    Args:
        axis (int): An axis
    """
    def __init__(self, axis):
        self._args = locals()
        self._axis = axis
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x, y):

        if np.random.random() > 0.5:
            if x is not None:
                x = [flip(x_i, self._axis) for x_i in x]
            if y is not None:
                y = [flip(y_i, self._axis) for y_i in y]
        return x, y


def crop(x, x_s, x_e, y_s, y_e):
    if x is None:
        return x
    x = np.asarray(x).swapaxes(_channel_axis, 0)
    x = x[:, x_s:x_e, y_s:y_e]
    x = x.swapaxes(0, _channel_axis)
    return x


class Crop(Operation):
    """ Crop given images to the specified size at the random location.

    Args:
        size (list or tuple): Cropping size
    """
    def __init__(self, size):
        self._args = locals()
        assert(isinstance(size, (list, tuple)))
        self._size = size
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x, y):

        if x is not None:
            h, w = x[0].shape[_row_axis], x[0].shape[_col_axis]
        elif y is not None:
            h, w = y[0].shape[_row_axis], y[0].shape[_col_axis]
        else:
            return x, y

        x_s = np.random.randint(0, h - self._size[0] + 1)
        x_e = x_s + self._size[0]
        y_s = np.random.randint(0, w - self._size[1] + 1)
        y_e = y_s + self._size[1]

        if x is not None:
            x = [crop(x_i, x_s, x_e, y_s, y_e) for x_i in x]
        if y is not None:
            y = [crop(y_i, x_s, x_e, y_s, y_e) for y_i in y]
        return x, y


def resize(x, size, interp_order=0):

    if x is None:
        return x

    if interp_order == 0:
        interpolation = cv2.INTER_NEAREST
    elif interp_order == 1:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_CUBIC

    x = np.asarray(x).swapaxes(_channel_axis, 2) # NOTE: opencv's format
    x = cv2.resize(x, size, interpolation=interpolation)
    if x.ndim == 2:
        x = x[:,:,np.newaxis]
    x = x.swapaxes(2, _channel_axis)

    return x


class ResizeCrop(Crop):
    """ Resize given images to the random size and,
        crop them to the specified size at the random location.

    Args:
        resize_size (list or tuple): Resizing size
        crop_size (list or tuple): Cropping size, which should be smaller than resizing size.
    """
    def __init__(self, resize_size, crop_size, interp_order=(0, 0)):

        super(ResizeCrop, self).__init__(crop_size)

        self._args = locals()

        assert(isinstance(resize_size, (list, tuple)))
        self._resize_size = resize_size

        assert all([src >= dst for src, dst in zip(self._resize_size, self._size)]), \
            'Cropping size should be smaller than resizing size..'

        if isinstance(interp_order, int):
            interp_order = [interp_order] * 2
        self._interp_order = interp_order

    def apply_core(self, x, y):

        size = (np.random.randint(self._size[0], self._resize_size[0] + 1), \
                np.random.randint(self._size[1], self._resize_size[1] + 1))

        if x is not None:
            x = [resize(x_i, size, self._interp_order[0]) for x_i in x]
        if y is not None:
            y = [resize(y_i, size, self._interp_order[1]) for y_i in y]

        return super().apply_core(x, y)


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    interp_order=0):
    """ Apply a transfrom matrix to an image

    Args:
        x (numpy.ndarray): An image (3D tensor).
        transform_matrix (numpy.ndarray): A 3x3 transformation matrix.
        channel_axis (int, optional): Index of axis for channels. Defaults to 0.
        fill_mode (str, optional): Points outside the boundaries of the image
                                   are filled according to the given mode. Defaults to 'nearest'
                                    (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval (float, optional): Value used for points outside the boundaries of the image. Defaults to 0.
        interp_order (int, optional): The order of the spline interpolation. Defaults to 0.

    Returns:
        numpy.ndarray: A transformed image.

    See also:
        :class:`~keras.preprocessing.image.ImageDataGenerator`
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=interp_order,  # NOTE: The order of the spline interpolation
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def zoom_matrix(zx, zy):
    matrix = np.array([[zx, 0, 0],
                       [0, zy, 0],
                       [0, 0, 1]])
    return matrix


def rotation_matrix(theta):
    matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
    return matrix


def translate_matrix(tx, ty):
    matrix = np.array([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0, 1]])
    return matrix


def shear_matrix(shear):
    matrix = np.array([[1, -np.sin(shear), 0],
                       [0, np.cos(shear), 0],
                       [0, 0, 1]])
    return matrix


def affine(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
           fill_mode='nearest', cval=0., interp_order=0):

    shape = x.shape

    matrix = np.eye(3)
    if theta != 0:
        matrix = np.dot(matrix, rotation_matrix(theta))
    if tx != 0 or ty != 0:
        matrix = np.dot(matrix, translate_matrix(tx, ty))
    if shear != 0:
        matrix = np.dot(matrix, shear_matrix(shear))
    if zx != 1 or zy != 1:
        matrix = np.dot(matrix, zoom_matrix(zx, zy))

    if np.any(matrix != np.eye(3)):
        h, w = shape[_row_axis], shape[_col_axis]
        matrix = transform_matrix_offset_center(matrix, h, w)
        x = apply_transform(x, matrix, _channel_axis,
                            fill_mode, cval, interp_order)
    return x


class Affine(Operation):
    """ Apply a randomly generated affine transform matrix to given images

    Args:
        rotation (float, optional): Rotation angle. Defaults to 0.
        translate (tuple of float, optional): Translation ratios. Defaults to (0., 0.).
        shear (float, optional): Shear angle. Defaults to 0.
        zoom (tuple of float, optional): Enlarge and shrinkage ratios. Defaults to (1., 1.).
        keep_aspect_ratio (bool, optional): Keep the aspect ratio. Defaults to True.
        fill_mode (tuple of str, optional): Points outside the boundaries of the image are filled according to the given mode
                                            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`). Defaults to ('nearest', 'nearest').
        cval (tuple of float, optional): Values used for points outside the boundaries of the image. Defaults to (0., 0.).
        interp_order (tuple of int, optional): Spline interpolation orders. Defaults to (0, 0).
    """
    def __init__(self,
                 rotation=0.,
                 translate=(0., 0.),
                 shear=0.,
                 zoom=(1., 1.),
                 keep_aspect_ratio=True,
                 fill_mode=('nearest', 'nearest'),
                 cval=(0., 0.),
                 interp_order=(0, 0),
                 ):

        self._args = locals()

        if isinstance(translate, (float, int)):
            translate = [translate] * 2
        if isinstance(fill_mode, str):
            fill_mode = [fill_mode] * 2
        if isinstance(cval, (float, int)):
            cval = [cval] * 2
        if isinstance(interp_order, int):
            interp_order = [interp_order] * 2

        assert len(zoom) == 2

        self._rotation = rotation
        self._translate = translate
        self._shear = shear
        self._zoom = zoom
        self._keep_aspect_ratio = keep_aspect_ratio
        self._fill_mode = fill_mode
        self._cval = cval
        self._interp_order = interp_order
        self._ndim = 2

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x, y):

        if self._rotation:
            theta = np.pi / 180 * \
                np.random.uniform(-self._rotation, self._rotation)
        else:
            theta = 0

        if self._translate[0]:
            tx = np.random.uniform(-self._translate[0], self._translate[0])
        else:
            tx = 0

        if self._translate[1]:
            ty = np.random.uniform(-self._translate[1], self._translate[1])
        else:
            ty = 0

        if self._shear:
            shear = np.random.uniform(-self._shear, self._shear)
        else:
            shear = 0

        if self._zoom[0] == 1 and self._zoom[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self._zoom[0], self._zoom[1], 2)

        if self._keep_aspect_ratio:
            zy = zx

        fill_mode = self._fill_mode
        cval = self._cval
        interp_order = self._interp_order

        if x is not None:
            x = [affine(x_i, theta, tx, ty, shear, zx, zy,
                        fill_mode[0], cval[0], interp_order[0]) for x_i in x]
        if y is not None:
            y = [affine(y_i, theta, tx, ty, shear, zx, zy,
                        fill_mode[1], cval[1], interp_order[1]) for y_i in y]

        return x, y
