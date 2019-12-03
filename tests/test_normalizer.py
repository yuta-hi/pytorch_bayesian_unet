import numpy as np
from chainer_bcnn.data.normalizer import Normalizer, Quantize2D, Clip2D, Subtract2D, Divide2D

import cv2
import time
import matplotlib.pyplot as plt


def main():
    normalizer = Normalizer()
    normalizer.add(Clip2D('ch_minmax'))
    # normalizer.add(Quantize2D(n_bit=3))
    # normalizer.add(Subtract2D('ch_mean'))
    # normalizer.add(Divide2D('ch_std'))
    normalizer.add(Subtract2D(0.5))
    normalizer.add(Divide2D(0.5))
    normalizer.summary('norm.json')

    x_in = cv2.imread('lenna.png').astype(np.float32)
    x_in = x_in[:, :, ::-1]
    x_in = np.transpose(x_in, (2, 0, 1))
    print(x_in.shape)

    tic = time.time()
    x_out = normalizer.apply(x_in)
    print('time: %f [sec]' % (time.time()-tic))

    print(x_out.shape)

    plt.imshow(np.transpose(x_out, (1, 2, 0))[:,:,0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
