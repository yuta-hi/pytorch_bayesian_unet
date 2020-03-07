import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from functools import partial
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
import chainer.functions as F
from chainerui.utils import save_args
from chainer_bcnn.datasets import ImageDataset
from chainer_bcnn.data.augmentor import DataAugmentor, Flip2D, Affine2D, Crop2D, ResizeCrop2D
from chainer_bcnn.data.normalizer import Normalizer, Clip2D, Subtract2D, Divide2D
from chainer_bcnn.models import BayesianUNet, UNet
from chainer_bcnn.links import Regressor
from chainer_bcnn.extensions import LogReport
from chainer_bcnn.extensions import PrintReport
from chainer_bcnn.extensions import Validator
from chainer_bcnn.visualizer import ImageVisualizer
from chainer_bcnn.links import MCSampler
from chainer_bcnn.inference import Inferencer
from chainer_bcnn.data import load_image, save_image
from chainer_bcnn.datasets import train_valid_split
from chainer_bcnn.utils import fixed_seed
from chainer_bcnn.utils import find_latest_snapshot

from train_and_test_pix2pix import get_dataset
from train_and_test_pix2pix import get_normalizer
from train_and_test_pix2pix import get_augmentor
from train_and_test_pix2pix import build_generator
from train_and_test_pix2pix import test_phase


def train_phase(generator, train, valid, args):

    print('# samples:')
    print('-- train:', len(train))
    print('-- valid:', len(valid))

    # setup dataset iterators
    train_batchsize = min(args.batchsize*len(args.gpu), len(train))
    valid_batchsize = args.batchsize
    train_iter = chainer.iterators.MultiprocessIterator(train, train_batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, valid_batchsize,
                                                repeat=False, shuffle=True)

    # setup a model
    model = Regressor(generator,
                      activation=F.tanh,
                      lossfun=F.mean_absolute_error,
                      accfun=F.mean_absolute_error)

    if args.gpu[0] >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu[0]).use()
        if len(args.gpu) == 1: model.to_gpu()

    # setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.lr, beta1=args.beta, beta2=0.999, eps=1e-08, amsgrad=False)
    optimizer.setup(model)
    if args.decay > 0:
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.decay))


    # setup a trainer
    if len(args.gpu) == 1:
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=args.gpu[0])
    else:
        devices = {'main':args.gpu[0]}
        for idx, g in enumerate(args.gpu[1:]):
            devices['slave_%d' % idx] = g
        updater = training.updaters.ParallelUpdater(
            train_iter, optimizer, devices=devices)


    frequency = max(args.iteration//80, 1) if args.frequency == -1 else max(1, args.frequency)

    stop_trigger = triggers.EarlyStoppingTrigger(monitor='validation/main/loss',
                        max_trigger=(args.iteration, 'iteration'),
                        check_trigger=(frequency, 'iteration'),
                        patients=np.inf if args.pinfall == -1 else max(1, args.pinfall))

    trainer = training.Trainer(updater, stop_trigger, out=args.out)


    # shift lr
    trainer.extend(
        extensions.LinearShift('alpha', (args.lr, 0.0),
                        (args.iteration//2, args.iteration),
                        optimizer=optimizer))

    # setup a visualizer

    transforms = {'x': lambda x: x, 'y': lambda x: x, 't': lambda x: x}
    clims = {'x': (-1., 1.), 'y': (-1., 1.), 't': (-1., 1.)}

    visualizer = ImageVisualizer(transforms=transforms,
                                 cmaps=None,
                                 clims=clims)

    # setup a validator
    valid_file = os.path.join('validation', 'iter_{.updater.iteration:08}.png')
    trainer.extend(Validator(valid_iter, model, valid_file,
                             visualizer=visualizer, n_vis=20,
                             device=args.gpu[0]),
                             trigger=(frequency, 'iteration'))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration:08}.npz'),
                                       trigger=(frequency, 'iteration'))
    trainer.extend(extensions.snapshot_object(generator, 'generator_iter_{.updater.iteration:08}.npz'),
                                              trigger=(frequency, 'iteration'))

    log_keys = ['main/loss', 'validation/main/loss',
                'main/accuracy',  'validation/main/accuracy']

    trainer.extend(LogReport(keys=log_keys))

    # setup log ploter
    if extensions.PlotReport.available():
        for plot_key in ['loss', 'accuracy']:
            plot_keys = [key for key in log_keys if key.split('/')[-1].startswith(plot_key)]
            trainer.extend(
                extensions.PlotReport(plot_keys,
                                     'iteration', file_name=plot_key + '.png',
                                     trigger=(frequency, 'iteration')) )

    trainer.extend(PrintReport(['iteration'] + log_keys + ['elapsed_time'], n_step=100))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)


    # train
    trainer.run()


def main():

    parser = argparse.ArgumentParser(description='Example: Uncertainty estimates in image synthesis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', '-d', type=str, default='./preprocessed',
                        help='Directory to dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=5,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=200000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, nargs='+', default=[0],
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--valid_augment', action='store_true',
                        help='Enable data augmentation during validation')
    parser.add_argument('--valid_split_ratio', type=float, default=0.1,
                        help='Ratio of validation data to training data')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Exponential decay rate of the first order moment in Adam')
    parser.add_argument('--decay', type=float, default=-1,
                        help='Weight of L2 regularization')
    parser.add_argument('--mc_iteration', type=int, default=15,
                        help='Number of iteration of MCMC')
    parser.add_argument('--pinfall', type=int, default=-1,
                        help='Countdown for early stopping of training.')
    parser.add_argument('--freeze_upconv', action='store_true',
                        help='Disables updating the up-convolutional weights. If weights are initialized with \
                            bilinear kernels, up-conv acts as bilinear upsampler.')
    parser.add_argument('--test_on_test', action='store_true',
                        help='Switch to the testing phase on test dataset')
    parser.add_argument('--test_on_valid', action='store_true',
                        help='Switch to the testing phase on valid dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Fix the random seed')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    # setup output directory
    os.makedirs(args.out, exist_ok=True)

    # NOTE: ad-hoc
    normalizer = get_normalizer()
    augmentor = get_augmentor()

    # setup a generator
    with fixed_seed(args.seed, strict=False):

        generator = build_generator()

        if args.freeze_upconv:
            generator.freeze_layers(name='upconv',
                                    recursive=True,
                                    verbose=True)

        # setup dataset
        train, valid, test = get_dataset(args.data_root,
                                         args.valid_split_ratio,
                                         args.valid_augment,
                                         normalizer, augmentor)

        # run
        if args.test_on_test:
            raise RuntimeError('This example is under construction. Please tune the hyperparameters first..')
            test_phase(generator, test, args)
        elif args.test_on_valid:
            test_phase(generator, valid, args)
        else:
            save_args(args, args.out)
            generator.save_args(os.path.join(args.out, 'model.json'))
            normalizer.summary(os.path.join(args.out, 'norm.json'))
            augmentor.summary(os.path.join(args.out, 'augment.json'))

            train_phase(generator, train, valid, args)


if __name__ == '__main__':
    main()
