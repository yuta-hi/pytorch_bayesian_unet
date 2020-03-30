import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_bcnn.initializers import bilinear_upsample
from pytorch_bcnn.functions import crop
import matplotlib.pyplot as plt

def main():
    x = cv2.imread('lenna.png')
    x = cv2.resize(x, (64, 64))
    x = np.transpose(x, (2,0,1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    c = x.shape[1]

    deconv = nn.ConvTranspose2d(c, c, kernel_size=(3,3), stride=2, padding=(0,0), bias=False)
    bilinear_upsample(deconv.weight)

    y = deconv(torch.as_tensor(x))
    y = y.detach().numpy()

    plt.subplot(1,2,1)
    plt.imshow(x[0,0], cmap='gray')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(y[0,0], cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
