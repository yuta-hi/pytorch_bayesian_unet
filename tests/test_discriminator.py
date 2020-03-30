import os
import numpy as np
import subprocess
import warnings
import torch
from torchviz import make_dot
from pytorch_bcnn.models import PatchDiscriminator

def main():

    conv_param = {
        'name':'conv',
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'initialW': {'name': 'normal', 'std': 0.02},
        'initial_bias': {'name': 'zero'},
        'hook': {'name': 'spectral_normalization'}
    }

    pool_param = {
        'name': 'stride',
        'stride': 2
    }

    norm_param = {
        'name': 'batch'
    }

    activation_param = {
        'name': 'leaky_relu'
    }

    dropout_param = {
        'name': 'none'
    }

    model = PatchDiscriminator(
                        ndim=2,
                        in_channels=1,
                        out_channels=1,
                        nlayer=5,
                        nfilter=32,
                        conv_param=conv_param,
                        pool_param=pool_param,
                        norm_param=norm_param,
                        activation_param=activation_param,
                        dropout_param=dropout_param)

    x = np.random.rand(2, 1, 256, 256).astype(np.float32)
    y = model(torch.as_tensor(x))
    print(y.shape)

    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render('graph_2d_discriminator', format='pdf')

    model.save_args('2d_discriminator.json')
    model.show_statistics()
    print('-----')

    print(model.count_params())
    print('-----')

    # check the shape of the first left singular vector.
    vector_name = 'weight_u'
    print(vector_name)
    print(getattr(model.block_0.conv_0, vector_name).shape)
    print('-----')


if __name__ == '__main__':
    main()
