import os
import numpy as np
import subprocess
import warnings
from chainer_bcnn.extensions import log_report
from chainer_bcnn.models import UNet, BayesianUNet
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
    model = BayesianUNet(ndim=2, out_channels=10)
    x = np.random.rand(2, 1, 200, 300).astype(np.float32)
    y = model(x)
    print(y.shape)
    dump_graph(y, 'graph_2d_unet.dot')
    model.save_args('2d_unet.json')
    model.show_statistics()
    print(model.count_params())

    print('-----')

    model = BayesianUNet(ndim=3, out_channels=10, nlayer=3)
    x = np.random.rand(2, 1, 20, 30, 10).astype(np.float32)
    y = model(x)
    print(y.shape)
    dump_graph(y, 'graph_3d_unet.dot')
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


if __name__ == '__main__':
    main()
