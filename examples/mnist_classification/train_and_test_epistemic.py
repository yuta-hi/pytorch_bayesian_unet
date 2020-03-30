import os
import argparse
import numpy as np
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from chainer.datasets import get_mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_bcnn.links.noise import MCDropout
from pytorch_bcnn.links import Classifier
from pytorch_bcnn.links import MCSampler
from pytorch_bcnn.inference import Inferencer
from pytorch_bcnn.utils import fixed_seed
from pytorch_trainer import iterators
from pytorch_trainer import dataset
from pytorch_trainer import training
from pytorch_trainer.training import extensions


class Dataset(dataset.DatasetMixin):

    def __init__(self, phase, indices=None, withlabel=True, ndim=3, scale=1.,
                 dtype=np.float32, label_dtype=np.int32, rgb_format=False):

        super(Dataset, self).__init__()

        train, test = get_mnist(withlabel, ndim, scale, dtype, label_dtype, rgb_format)

        if phase == 'train':
            dataset = train
        elif phase == 'test':
            dataset = test
        else:
            raise KeyError('`phase` should be `train` or `test`..')

        if indices is not None:
            if isinstance(indices, list):
                indices = np.asarray(indices)
        else:
            indices = np.arange(len(dataset))

        assert len(indices) <= len(dataset)

        dataset = dataset[indices]

        if withlabel:
            images, labels = dataset
        else:
            images, labels = dataset, None

        self._phase=phase
        self._indices=indices
        self._ndim=ndim
        self._scale=scale
        self._dtype=dtype
        self._label_dtype=label_dtype
        self._rgb_format=rgb_format

        self._images = images
        self._labels = labels

    @property
    def indices(self):
        return self._indices

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self._indices)

    @dataset.convert_to_tensor
    def get_example(self, i):

        if self._labels is None:
            return self._images[i]

        return self._images[i], int(self._labels[i])


class BayesianConvNet(nn.Module):

    _in_size = (1, 28, 28)

    def __init__(self,
                 n_in=1,
                 conv_size=3, n_filter=32,
                 pool_size=5,
                 n_units=128, n_out=10):
        super(BayesianConvNet, self).__init__()

        self._n_in = n_in
        self._n_filter = n_filter
        self._conv_size = conv_size
        self._pool_size = pool_size
        self._n_units = n_units
        self._n_out = n_out

        padding = (conv_size-1) // 2

        self.conv_1 = nn.Conv2d(n_in, n_filter, conv_size, padding=padding)
        self.conv_2 = nn.Conv2d(n_filter, n_filter, conv_size, padding=padding)

        _h_size = self._in_size[0] * n_filter * \
                        (self._in_size[1]//pool_size) * (self._in_size[2]//pool_size)

        self.l1 = nn.Linear(_h_size, n_units)
        self.l2 = nn.Linear(n_units, n_out)

        self.dropout1 = MCDropout(0.25)
        self.dropout2 = MCDropout(0.5)

        self._initialize_params()

    def _initialize_params(self):
        initialW = nn.init.kaiming_normal_
        initialW(self.conv_1.weight)
        initialW(self.conv_2.weight)
        initialW(self.l1.weight)
        initialW(self.l2.weight)

    def forward(self, x):

        h = F.relu(self.conv_1(x))
        h = F.relu(self.conv_2(h))
        h = F.max_pool2d(h, self._pool_size)
        h = self.dropout1(h)

        h = torch.flatten(h, 1)

        h = F.relu(self.l1(h))
        h = self.dropout2(h)
        h = self.l2(h)

        return h


def train_phase(predictor, train, valid, args):

    # setup iterators
    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    # setup a model
    device = torch.device(args.gpu)

    model = Classifier(predictor)
    model.to(device)

    # setup an optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=max(args.decay, 0))

    # setup a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, model, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))

    # trainer.extend(DumpGraph(model, 'main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        trainer.load_state_dict(torch.load(args.resume))

    trainer.run()

    torch.save(predictor.state_dict(), os.path.join(args.out, 'predictor.pth'))


def test_phase(predictor, test, args):

    # setup an iterator
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # setup an inferencer
    predictor.load_state_dict(torch.load(os.path.join(args.out, 'predictor.pth')))

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
    os.makedirs(args.out, exist_ok=True)

    match = pred == test.labels
    accuracy = np.sum(match) / len(match)

    arr = [uncert[match], uncert[np.logical_not(match)]]

    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(13,5))
    ax = sns.violinplot(data=arr, inner='quartile', palette='Blues', orient='h', cut=0)
    ax.set_xlabel('Predicted variance')
    ax.set_yticklabels(['Correct prediction\n(n=%d)' % len(arr[0]), 'Wrong prediction\n(n=%d)' % len(arr[1])])
    plt.title('Accuracy=%.3f' % accuracy)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'eval.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Example: Uncertainty estimates in classification')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the log files')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--test_on_test', action='store_true',
                        help='Switch to the testing phase on test dataset')
    parser.add_argument('--test_on_valid', action='store_true',
                        help='Switch to the testing phase on valid dataset')
    parser.add_argument('--mc_iteration', type=int, default=50,
                        help='Number of iteration of MCMC')
    parser.add_argument('--decay', type=float, default=-1,
                        help='Weight of L2 regularization')
    parser.add_argument('--seed', type=int, default=0,
                        help='Fix the random seed')
    args = parser.parse_args()


    os.makedirs(args.out, exist_ok=True)

    with fixed_seed(args.seed, strict=False):

        # setup a predictor
        predictor = BayesianConvNet(n_units=args.unit, n_out=10)

        # setup dataset
        train = Dataset(phase='train', indices=np.arange(0, 1000))
        valid = Dataset(phase='train', indices=np.arange(1000, 2000))
        test = Dataset(phase='test')

        # run
        if args.test_on_test:
            test_phase(predictor, test, args)
        elif args.test_on_valid:
            test_phase(predictor, valid, args)
        else:
            train_phase(predictor, train, valid, args)

if __name__ == '__main__':
    main()
