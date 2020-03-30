import os
import argparse
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from functools import partial
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_bcnn.datasets import ImageDataset
from pytorch_bcnn.data.augmentor import DataAugmentor, Flip2D, Affine2D
from pytorch_bcnn.data.normalizer import Normalizer, Clip2D, Subtract2D, Divide2D
from pytorch_bcnn.models import BayesianUNet
from pytorch_bcnn.links import Classifier
from pytorch_bcnn.links import MCSampler
from pytorch_bcnn.functions.loss import softmax_cross_entropy
from pytorch_bcnn.inference import Inferencer
from pytorch_bcnn.visualizer import ImageVisualizer
from pytorch_bcnn.data import load_image, save_image
from pytorch_bcnn.datasets import train_valid_split
from pytorch_bcnn.extensions import LogReport
from pytorch_bcnn.extensions import PrintReport
from pytorch_bcnn.extensions import Validator
from pytorch_bcnn.utils import save_args
from pytorch_bcnn.utils import fixed_seed
from pytorch_bcnn.utils import find_latest_snapshot
from pytorch_trainer import iterators
from pytorch_trainer import dataset
from pytorch_trainer import training
from pytorch_trainer.training import extensions
from pytorch_trainer.training import triggers
from scipy.stats import pearsonr

def eval_metric(y, t):
    def dice(y, t):
        y = y.astype(np.bool)
        t = t.astype(np.bool)
        return 2. * np.logical_and(y, t).sum() / (y.sum() + t.sum())
    return dice(y, t)


def train_phase(predictor, train, valid, args):

    print('# classes:', train.n_classes)
    print('# samples:')
    print('-- train:', len(train))
    print('-- valid:', len(valid))

    # setup dataset iterators
    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize,
                                                repeat=False, shuffle=True)

    # setup a model
    class_weight = None # NOTE: please set if you have..

    lossfun = partial(softmax_cross_entropy,
                      normalize=False, class_weight=class_weight)

    device = torch.device(args.gpu)

    model = Classifier(predictor, lossfun=lossfun)
    model.to(device)

    # setup an optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=max(args.decay, 0))


    # setup a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, model, device=device)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    frequency = max(args.iteration//20, 1) if args.frequency == -1 else max(1, args.frequency)

    stop_trigger = triggers.EarlyStoppingTrigger(monitor='validation/main/loss',
                        max_trigger=(args.iteration, 'iteration'),
                        check_trigger=(frequency, 'iteration'),
                        patients=np.inf if args.pinfall == -1 else max(1, args.pinfall))

    trainer = training.Trainer(updater, stop_trigger, out=args.out)


    # setup a visualizer
    transforms = {'x': lambda x: x, 'y': lambda x: np.argmax(x, axis=0), 't': lambda x: x}

    cmap = np.array([[0,0,0],[0,0,1]])
    cmaps = {'x': None, 'y': cmap, 't': cmap}

    clims = {'x': 'minmax', 'y': None, 't': None}

    visualizer = ImageVisualizer(transforms=transforms,
                                 cmaps=cmaps, clims=clims)

    # setup a validator
    valid_file = os.path.join('validation', 'iter_{.updater.iteration:08}.png')
    trainer.extend(Validator(valid_iter, model, valid_file,
                             visualizer=visualizer, n_vis=20,
                             device=args.gpu),
                             trigger=(frequency, 'iteration'))

    # trainer.extend(DumpGraph(model, 'main/loss'))

    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration:08}.pth'),
                                       trigger=(frequency, 'iteration'))
    trainer.extend(extensions.snapshot_object(predictor, 'predictor_iter_{.updater.iteration:08}.pth'),
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
        trainer.load_state_dict(torch.load(args.resume))


    # train
    trainer.run()


def test_phase(predictor, test, args):

    print('# samples:')
    print('-- test:', len(test))

    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # setup a inferencer
    snapshot_file = find_latest_snapshot('predictor_iter_{.updater.iteration:08}.pth', args.out)
    predictor.load_state_dict(torch.load(snapshot_file))

    print('Loaded a snapshot:', snapshot_file)

    model = MCSampler(predictor,
                      mc_iteration=args.mc_iteration,
                      activation=partial(torch.softmax, dim=1),
                      reduce_mean=partial(torch.argmax, dim=1),
                      reduce_var=partial(torch.mean, dim=1))

    device = torch.device(args.gpu)
    model.to(device)

    infer = Inferencer(test_iter, model, device=args.gpu)

    pred, uncert = infer.run()


    # evaluate
    os.makedirs(os.path.join(args.out, 'test'), exist_ok=True)

    acc_values = []
    uncert_values = []

    uncert_clim = (0, np.percentile(uncert, 95))


    files = test.files['image']
    if isinstance(files, np.ndarray): files = files.tolist()
    commonpath = os.path.commonpath(files)

    plt.rcParams['font.size'] = 14

    for i, (p, u, imf, lbf) in enumerate(zip(pred, uncert,
                                             test.files['image'],
                                             test.files['label'])):
        im, _ = load_image(imf)
        im = im[:,:,::-1]
        lb, _ = load_image(lbf)
        if lb.ndim == 3: lb = lb[:,:,0]

        acc_values.append( eval_metric(p,lb) )
        uncert_values.append( np.mean(u[p==1]) ) # NOTE: instrument class


        plt.figure(figsize=(20,4))

        for j, (pic, cmap, clim, title) in enumerate(zip(
                                        [im, p, lb, u, (p!=lb).astype(np.uint8)],
                                        [None, None, None, 'jet', 'jet'],
                                        [None, None, None, uncert_clim, None],
                                        ['Input image\n%s' % os.path.relpath(imf, commonpath),
                                             'Predicted label\n(DC=%.3f)' % acc_values[-1],
                                             'Ground-truth label',
                                             'Predicted variance\n(PV=%.4f)' % uncert_values[-1],
                                             'Error'])):
            plt.subplot(1,5, j+1)
            plt.imshow(pic, cmap=cmap)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(title)
            plt.clim(clim)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'test/%03d.png' % i))
        plt.close()


    c = pearsonr(uncert_values, acc_values)

    plt.figure(figsize=(11,11))
    ax = sns.scatterplot(x=uncert_values,
                         y=acc_values, color='blue', s=50)
    ax.set_xlabel('Predicted variance')
    ax.set_ylabel('Dice coefficient')
    plt.grid()
    plt.title('r=%.3f' % c[0])
    plt.savefig(os.path.join(args.out, 'eval.png'))
    plt.close()


def get_dataset(data_root,
                valid_split_type,
                valid_split_ratio,
                valid_augment,
                normalizer=None,
                augmentor=None):

    class_list = ['background', 'instrument']
    dtypes = OrderedDict({'image': np.float32, 'label': np.int64})

    getter = partial(ImageDataset, root=data_root, classes=class_list,
                        dtypes=dtypes, normalizer=normalizer)

    # train and valid dataset
    train_patients = ['OP1', 'OP2', 'OP3', 'OP4']

    train_filenames = OrderedDict({
        'image': '{root}/train/{patient}/Raw/*_raw.png',
        'label': '{root}/train/{patient}/Masks/*_class.png',
    })

    if valid_split_type == 'slice':
        dataset = getter(patients=train_patients, filenames=train_filenames, augmentor=augmentor)
        train, valid = train_valid_split(dataset, valid_split_ratio)

    elif valid_split_type == 'patient':
        train_patients, valid_patients = train_valid_split(train_patients, valid_split_ratio)
        train = getter(patients=train_patients, filenames=train_filenames, augmentor=augmentor)
        valid = getter(patients=valid_patients, filenames=train_filenames, augmentor=augmentor)

    else:
        raise NotImplementedError('unsupported validation split type..')

    if not valid_augment:
        del valid.augmentor

    # test dataset
    test_filenames = OrderedDict({
        'image': '{root}/test/{patient}/*_raw.png',
        'label': '{root}/test/{patient}/*_class.png',
    })

    test_patients = ['OP*'] # NOTE: wildcard

    test = getter(patients=test_patients, filenames=test_filenames, augmentor=None)

    return train, valid, test



def main():

    parser = argparse.ArgumentParser(description='Example: Uncertainty estimates in segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', '-d', type=str, default='./preprocessed',
                        help='Directory to dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=2,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=50000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=str, default='cuda:0',
                        help='GPU Device')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--valid_augment', action='store_true',
                        help='Enable data augmentation during validation')
    parser.add_argument('--valid_split_ratio', type=float, default=0.1,
                        help='Ratio of validation data to training data')
    parser.add_argument('--valid_split_type', type=str, default='slice', choices=['slice', 'patient'],
                        help='How to choice validation data from training data')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
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
    # setup a normalizer
    normalizer = Normalizer()
    normalizer.add(Clip2D('minmax'))
    normalizer.add(Subtract2D(0.5))
    normalizer.add(Divide2D(1./255.))

    # setup an augmentor
    augmentor = DataAugmentor(n_dim=2)
    augmentor.add(Flip2D(axis=2))
    augmentor.add(Affine2D(rotation=15.,
                        translate=(10.,10.),
                        shear=0.25,
                        zoom=(0.8, 1.2),
                        keep_aspect_ratio=True,
                        fill_mode=('reflect', 'reflect'),
                        cval=(0.,0.),
                        interp_order=(1,0)))


    with fixed_seed(args.seed, strict=False):

        # setup a predictor
        conv_param = { # NOTE: you can change layer type if you want..
            'name':'conv',
            'kernel_size': 3,
            'stride': 1,
            'padding': 2,
            'padding_mode': 'reflect',
            'dilation': 2,
            'initialW': {'name': 'he_normal'},
            'initial_bias': {'name': 'zero'},
        }

        upconv_param = { # NOTE: you can change layer type if you want..
            'name':'deconv',
            'kernel_size': 3,
            'stride': 2,
            'padding': 0,
            'initialW': {'name': 'bilinear'},
            'initial_bias': {'name': 'zero'},
        }

        norm_param = {'name': 'batch'}

        predictor = BayesianUNet(ndim=2,
                                 in_channels=3,
                                 out_channels=2,
                                 nlayer=4,
                                 nfilter=32,
                                 conv_param=conv_param,
                                 upconv_param=upconv_param,
                                 norm_param=norm_param)

        if args.freeze_upconv:
            predictor.freeze_layers(name='upconv',
                                    recursive=True,
                                    verbose=True)

        # setup dataset
        train, valid, test = get_dataset(args.data_root,
                                         args.valid_split_type,
                                         args.valid_split_ratio,
                                         args.valid_augment,
                                         normalizer, augmentor)

        # run
        if args.test_on_test:
            test_phase(predictor, test, args)
        elif args.test_on_valid:
            test_phase(predictor, valid, args)
        else:
            save_args(args, args.out)
            predictor.save_args(os.path.join(args.out, 'model.json'))
            normalizer.summary(os.path.join(args.out, 'norm.json'))
            augmentor.summary(os.path.join(args.out, 'augment.json'))

            train_phase(predictor, train, valid, args)


if __name__ == '__main__':
    main()
