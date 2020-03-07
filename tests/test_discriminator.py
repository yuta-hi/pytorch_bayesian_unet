import os
import numpy as np
import subprocess
import warnings
from chainer_bcnn.extensions import log_report
from chainer_bcnn.models import PatchDiscriminator
import chainer.computational_graph as c


def dump_graph(variable, out):

    g = c.build_computational_graph(variable)
    with open(out, 'w') as o:
        o.write(g.dump())

    try:
        out_png, _ = os.path.splitext(out)
        out_png += '.png'
        subprocess.call('dot -T png %s -o %s' % (out, out_png), shell=False)
    except:
        warnings.warn('please install graphviz and set your environment.')


def main():

    conv_param = {
        'name':'conv',
        'ksize': 3,
        'stride': 1,
        'pad': 1,
        'initialW': {'name': 'normal', 'scale': 0.02},
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
                        out_channels=1,
                        nlayer=5,
                        nfilter=32,
                        conv_param=conv_param,
                        pool_param=pool_param,
                        norm_param=norm_param,
                        activation_param=activation_param,
                        dropout_param=dropout_param)

    x = np.random.rand(2, 1, 256, 256).astype(np.float32)
    y = model(x)
    print(y.shape)
    dump_graph(y, 'graph_2d_discriminator.dot')
    model.save_args('2d_discriminator.json')
    model.show_statistics()
    print('-----')

    print(model.count_params())
    print('-----')

    # check the shape of the first left singular vector.
    vector_name = 'W_u'
    print(vector_name)
    print(getattr(model.block_0.conv_0, vector_name).shape)
    print('-----')


if __name__ == '__main__':
    main()
