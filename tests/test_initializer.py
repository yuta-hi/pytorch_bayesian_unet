import cv2
import numpy as np
import chainer.links as L
from chainer_bcnn.links.connection import Convolution2D
from chainer_bcnn.links.connection import Deconvolution2D
from chainer_bcnn.initializers import BilinearUpsample
from chainer_bcnn.functions import crop
import matplotlib.pyplot as plt

def test_compare_model(x, model_1, model_2):

    print('x.shape:', x.shape)
    y_1 = model_1(x).data
    y_2 = model_2(x).data

    y_2 = crop(y_2, y_1.shape, ndim=2)

    print('y_1.shape:', y_1.shape)
    print('y_2.shape:', y_2.shape)

    plt.subplot(141)
    plt.imshow(x[0,0,:,:], cmap='gray')
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(y_1[0,0,:,:], cmap='gray')
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(y_2[0,0,:,:], cmap='gray')
    plt.colorbar()
    plt.subplot(144)
    plt.imshow(np.abs(y_2[0,0,:,:]-y_1[0,0,:,:]), cmap='jet')
    plt.colorbar()
    plt.show()

def main():
    x = cv2.imread('lenna.png')
    x = cv2.resize(x, (64, 64))
    x = np.transpose(x, (2,0,1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    c = x.shape[1]

    conv_default = L.Convolution2D(c,c,ksize=(3,3), stride=1, pad=(1,1), initialW=BilinearUpsample(), nobias=True)
    conv_reflect =   Convolution2D(c,c,ksize=(3,3), stride=1, pad=(1,1), initialW=BilinearUpsample(), nobias=True)

    deconv_default = L.Deconvolution2D(c,c,ksize=(3,3), stride=2, pad=(0,0), initialW=BilinearUpsample(), nobias=True)
    deconv_reflect =   Deconvolution2D(c,c,ksize=(3,3), stride=2, pad=(1,1), initialW=BilinearUpsample(), nobias=True)

    test_compare_model(x, conv_default, conv_reflect)
    test_compare_model(x, deconv_default, deconv_reflect)

if __name__ == '__main__':
    main()
