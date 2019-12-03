from __future__ import absolute_import

import numpy as np
import scipy.ndimage as ndi

from . import Operation

_row_axis = 1
_col_axis = 2
_depth_axis = 3
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
        self._args.pop('self')
        self._axis = axis
        self._ndim = 3

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

    def summary(self):
        return self._args


def crop(x, x_s, x_e, y_s, y_e, z_s, z_e):
    if x is None:
        return x
    x = np.asarray(x).swapaxes(_channel_axis, 0)
    x = x[:, x_s:x_e, y_s:y_e, z_s:z_e]
    x = x.swapaxes(0, _channel_axis)
    return x


class Crop(Operation):
    """ Crop given images to the specified size at the random location.

    Args:
        size (list or tuple): Cropping size
    """
    def __init__(self, size):
        self._args = locals()
        self._args.pop('self')
        assert isinstance(size, (list, tuple))
        self._size = size
        self._ndim = 3

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x, y):

        if x is not None:
            h, w, d = (x[0].shape[axis]
                       for axis in (_row_axis, _col_axis, _depth_axis))
        elif y is not None:
            h, w, d = (y[0].shape[axis]
                       for axis in (_row_axis, _col_axis, _depth_axis))
        else:
            return x, y

        y_s = np.random.randint(0, h - self._size[0])
        y_e = y_s + self._size[0]
        x_s = np.random.randint(0, w - self._size[1])
        x_e = x_s + self._size[1]
        z_s = np.random.randint(0, d - self._size[2])
        z_e = z_s + self._size[2]

        if x is not None:
            x = [crop(x_i, x_s, x_e, y_s, y_e, z_s, z_e) for x_i in x]
        if y is not None:
            y = [crop(y_i, x_s, x_e, y_s, y_e, z_s, z_e) for y_i in y]
        return x, y

    def summary(self):
        return self._args


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    interp_order=0):
    """ Apply a transfrom matrix to an volume

    Args:
        x (numpy.ndarray): A volume (4D tensor).
        transform_matrix (numpy.ndarray): A 4x4 transformation matrix.
        channel_axis (int, optional): Index of axis for channels. Defaults to 0.
        fill_mode (str, optional): Points outside the boundaries of the image
                                   are filled according to the given mode. Defaults to 'nearest'
                                    (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval (float, optional): Value used for points outside the boundaries of the image. Defaults to 0.
        interp_order (int, optional): The order of the spline interpolation. Defaults to 0.

    Returns:
        numpy.ndarray: A transformed volume.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, -1]
    channel_volumes = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=interp_order,  # NOTE: The order of the spline interpolation
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_volumes, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array(
        [[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array(
        [[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def zoom_matrix(zx, zy, zz):
    matrix = np.array([[zx, 0, 0, 0],
                       [0, zy, 0, 0],
                       [0, 0, zz, 0],
                       [0, 0, 0, 1]])
    return matrix


def rotation_matrix(px, py, pz):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(px), np.sin(px)],
                   [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                   [0, 1, 0],
                   [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                   [-np.sin(pz), np.cos(pz), 0],
                   [0, 0, 1]])

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = Rz.dot(Ry).dot(Rx)  # z-y-x
    matrix[-1, -1] = 1
    return matrix


def translate_matrix(tx, ty, tz):
    matrix = np.array([[1, 0, 0, tx],
                       [0, 1, 0, ty],
                       [0, 0, 1, tz],
                       [0, 0, 0, 1]])
    return matrix


def shear_matrix(x, y, z):
    Sx = np.array([[1, 0, 0],
                   [0, np.cos(x), 0],
                   [0, -np.sin(x), 1]])

    Sy = np.array([[1, 0, -np.sin(y)],
                   [0, 1, 0],
                   [0, 0, np.cos(y)]])

    Sz = np.array([[np.cos(z), 0, 0],
                   [-np.sin(z), 1, 0],
                   [0, 0, 1]])

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = Sz.dot(Sy).dot(Sx)  # z-y-x
    matrix[-1, -1] = 1
    return matrix


def affine(x,
           px=0, py=0, pz=0,
           tx=0, ty=0, tz=0,
           sx=0, sy=0, sz=0,
           zx=1, zy=1, zz=1,
           fill_mode='nearest',
           cval=0.,
           interp_order=0):

    shape = x.shape

    matrix = np.eye(4)
    if px != 0 or py != 0 or pz != 0:
        matrix = np.dot(matrix, rotation_matrix(px, py, pz))
    if tx != 0 or ty != 0 or tz != 0:
        matrix = np.dot(matrix, translate_matrix(tx, ty, tz))
    if sx != 0 or sy != 0 or sz != 0:
        matrix = np.dot(matrix, shear_matrix(sx, sy, sz))
    if zx != 1 or zy != 1 or zz != 1:
        matrix = np.dot(matrix, zoom_matrix(zx, zy, zz))

    if np.any(matrix != np.eye(4)):
        h, w, d = shape[_row_axis], shape[_col_axis], shape[_depth_axis]
        matrix = transform_matrix_offset_center(matrix, h, w, d)
        x = apply_transform(x, matrix, _channel_axis,
                            fill_mode, cval, interp_order)
    return x


class Affine(Operation):
    """ Apply a randomly generated affine transform matrix to given volumes

    Args:
        rotation (tuple of float, optional): Rotation angles. Defaults to (0., 0., 0.)
        translate (tuple of float, optional): Translation ratios. Defaults to (0., 0., 0.)
        shear (tuple of float, optional): Shear angles. Defaults to (0., 0., 0.)
        zoom (tuple of float, optional): Enlarge and shrinkage ratios. Defaults to (1., 1.).
        keep_aspect_ratio (bool, optional): Keep the aspect ratio. Defaults to True.
        fill_mode (tuple of str, optional): Points outside the boundaries of the volume are filled according to the given mode
                                            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`). Defaults to ('nearest', 'nearest').
        cval (tuple of float, optional): Values used for points outside the boundaries of the volume. Defaults to (0., 0.).
        interp_order (tuple of int, optional): Spline interpolation orders. Defaults to (0, 0).
    """
    def __init__(self,
                 rotation=(0., 0., 0.), # TODO: might be better to use the quaternion in 3D.
                 translate=(0., 0., 0.),
                 shear=(0., 0., 0.),
                 zoom=(1., 1.),
                 keep_aspect_ratio=True,
                 fill_mode=('nearest', 'nearest'),
                 cval=(0., 0.),
                 interp_order=(0, 0),
                 ):

        self._args = locals()
        self._args.pop('self')

        if isinstance(rotation, (float, int)):
            rotation = [rotation] * 3
        if isinstance(translate, (float, int)):
            translate = [translate] * 3
        if isinstance(shear, (float, int)):
            shear = [shear] * 3
        if isinstance(fill_mode, str):
            fill_mode = [fill_mode] * 2
        if isinstance(cval, (float, int)):
            cval = [cval] * 2
        if isinstance(interp_order, int):
            interp_order = [interp_order] * 2

        assert(len(zoom) == 2)

        self._rotation = rotation
        self._translate = translate
        self._shear = shear
        self._zoom = zoom
        self._keep_aspect_ratio = keep_aspect_ratio
        self._fill_mode = fill_mode
        self._cval = cval
        self._interp_order = interp_order
        self._ndim = 3

    @property
    def ndim(self):
        return self._ndim

    def apply_core(self, x, y):

        if self._rotation[0]:
            px = np.pi / 180 * \
                np.random.uniform(-self._rotation[0], self._rotation[0])
        else:
            px = 0

        if self._rotation[1]:
            py = np.pi / 180 * \
                np.random.uniform(-self._rotation[1], self._rotation[1])
        else:
            py = 0

        if self._rotation[2]:
            pz = np.pi / 180 * \
                np.random.uniform(-self._rotation[2], self._rotation[2])
        else:
            pz = 0

        if self._translate[0]:
            tx = np.random.uniform(-self._translate[0], self._translate[0])
        else:
            tx = 0

        if self._translate[1]:
            ty = np.random.uniform(-self._translate[1], self._translate[1])
        else:
            ty = 0

        if self._translate[2]:
            tz = np.random.uniform(-self._translate[2], self._translate[2])
        else:
            tz = 0

        if self._shear[0]:
            sx = np.random.uniform(-self._shear[0], self._shear[0])
        else:
            sx = 0

        if self._shear[1]:
            sy = np.random.uniform(-self._shear[1], self._shear[1])
        else:
            sy = 0

        if self._shear[2]:
            sz = np.random.uniform(-self._shear[2], self._shear[2])
        else:
            sz = 0

        if self._zoom[0] == 1 and self._zoom[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(self._zoom[0], self._zoom[1], 3)

        if self._keep_aspect_ratio:
            zy = zx
            zz = zx

        fill_mode = self._fill_mode
        cval = self._cval
        interp_order = self._interp_order

        if x is not None:
            x = [affine(x_i, px, py, pz, tx, ty, tz, sx, sy, sz, zx, zy,
                        zz, fill_mode[0], cval[0], interp_order[0]) for x_i in x]
        if y is not None:
            y = [affine(y_i, px, py, pz, tx, ty, tz, sx, sy, sz, zx, zy,
                        zz, fill_mode[1], cval[1], interp_order[1]) for y_i in y]

        return x, y

    def summary(self):
        return self._args
