import os
import numpy as np
import subprocess
import warnings

import torch
from torchviz import make_dot
from pytorch_bcnn.models import UNet, BayesianUNet

def main():
    model = BayesianUNet(ndim=2, in_channels=1, out_channels=10)
    x = np.random.rand(2, 1, 200, 300).astype(np.float32)
    y = model(torch.Tensor(x))

    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render('graph_2d_unet', format='pdf')

    print(y.shape)
    model.save_args('2d_unet.json')
    model.show_statistics()
    print(model.count_params())

    print('-----')

    model = BayesianUNet(ndim=3, in_channels=1, out_channels=10, nlayer=3)
    x = np.random.rand(2, 1, 20, 30, 10).astype(np.float32)
    y = model(torch.Tensor(x))

    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render('graph_3d_unet', format='pdf')

    print(y.shape)
    model.save_args('3d_unet.json')
    model.show_statistics()
    print(model.count_params())

    print('-----')

    model.freeze_layers('upconv', verbose=True)
    print('-----')
    model.freeze_layers('upconv', recursive=False, verbose=True)
    print('-----')
    model.freeze_layers(startwith='upconv', verbose=True)
    print('-----')
    model.freeze_layers(endwith='norm', verbose=True)

    model.show_statistics()


if __name__ == '__main__':
    main()
