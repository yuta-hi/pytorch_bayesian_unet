import numpy as np
from chainer_bcnn.data.augmentor import DataAugmentor, Crop2D, Flip2D, Affine2D

import cv2
import time
import matplotlib.pyplot as plt


def main():
    augmentor = DataAugmentor()
    augmentor.add(Crop2D(size=(300, 400)))
    augmentor.add(Flip2D(axis=2))
    augmentor.add(Affine2D(rotation=15.,
                           translate=(10., 10.),
                           shear=0.25,
                           zoom=(0.8, 1.2),
                           keep_aspect_ratio=True,
                           fill_mode=('nearest', 'constant'),
                           cval=(0., 0.),
                           interp_order=(3, 0)))

    augmentor.summary('augment.json')

    x_in = cv2.imread('lenna.png').astype(np.float32) / 255.
    x_in = x_in[:, :, ::-1]
    x_in = np.transpose(x_in, (2, 0, 1))
    y_in = x_in[0, ...]
    print(x_in.shape)
    print(y_in.shape)

    tic = time.time()
    x_out, y_out = augmentor.apply(x_in, y_in)
    print('time: %f [sec]' % (time.time()-tic))

    print(x_out.shape)
    print(y_out.shape)

    plt.subplot(2, 2, 1)
    plt.imshow(np.transpose(x_in, (1, 2, 0)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(2, 2, 2)
    plt.imshow(np.transpose(x_out, (1, 2, 0)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(2, 2, 3)
    plt.imshow(y_in)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(2, 2, 4)
    plt.imshow(y_out)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
