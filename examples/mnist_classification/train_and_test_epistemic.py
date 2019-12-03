import os
import argparse
import numpy as np
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import get_mnist
from chainer_bcnn.functions import mc_dropout
from chainer_bcnn.links import Classifier
from chainer_bcnn.links import MCSampler
from chainer_bcnn.inference import Inferencer
from chainer_bcnn.utils import fixed_seed

class Dataset(chainer.dataset.DatasetMixin):

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

    def get_example(self, i):

        if self._labels is None:
            return self._images[i]

        return self._images[i], self._labels[i]


class BayesianConvNet(chainer.Chain):

    def __init__(self,
                 conv_size=3, n_filter=32,
                 pool_size=5,
                 n_units=128, n_out=10):
        super(BayesianConvNet, self).__init__()

        self._n_filter = n_filter
        self._conv_size = conv_size
        self._pool_size = pool_size
        self._n_units = n_units
        self._n_out = n_out

        initialW = chainer.initializers.HeNormal()

        with self.init_scope():

            self.conv_1 = L.Convolution2D(None, n_filter, conv_size, initialW=initialW)
            self.conv_2 = L.Convolution2D(None, n_filter, conv_size, initialW=initialW)

            self.l1 = L.Linear(None, n_units, initialW=initialW)
            self.l2 = L.Linear(None, n_out, initialW=initialW)

    def forward(self, x):

        h = F.relu(self.conv_1(x))
        h = F.relu(self.conv_2(h))
        h = F.max_pooling_2d(h, self._pool_size)
        h = mc_dropout(h, 0.25)

        h = F.relu(self.l1(h))
        h = mc_dropout(h, 0.5)
        h = self.l2(h)

        return h


def train_phase(predictor, train, valid, args):

    # setup iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    model = Classifier(predictor)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if args.decay > 0:
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.decay))

    # setup a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

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
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    chainer.serializers.save_npz(os.path.join(args.out, 'predictor.npz'), predictor)


def test_phase(predictor, test, args):

    # setup an iterator
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # setup an inferencer
    chainer.serializers.load_npz(os.path.join(args.out, 'predictor.npz'), predictor)

    model = MCSampler(predictor,
                      mc_iteration=args.mc_iteration,
                      activation=partial(F.softmax, axis=1),
                      reduce_mean=partial(F.argmax, axis=1),
                      reduce_var=partial(F.mean, axis=1))

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

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
