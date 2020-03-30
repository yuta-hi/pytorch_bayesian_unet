import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_bcnn.links.noise import MCDropout
from pytorch_bcnn.links import Regressor
from pytorch_bcnn.links import MCSampler
from pytorch_bcnn.inference import Inferencer
from pytorch_bcnn.utils import fixed_seed
from pytorch_trainer import iterators
from pytorch_trainer import dataset
from pytorch_trainer import training
from pytorch_trainer.training import extensions


class Dataset(dataset.DatasetMixin):

    def __init__(self,
                 func=lambda x: x*np.sin(x),
                 n_samples=100,
                 x_lim=(-5, 5),
                 dtype=np.float32):

        assert callable(func)
        assert isinstance(x_lim, (list, tuple))

        x = np.random.rand(n_samples, 1)*(x_lim[1] - x_lim[0]) + x_lim[0]
        x = np.sort(x, axis=0)
        t = func(x)

        self._func = func
        self._x_lim = x_lim
        self._x = x.astype(dtype)
        self._t = t.astype(dtype)

    @property
    def x(self): # NOTE: input
        return self._x

    @property
    def y(self): # NOTE: observation
        return self._t

    @property
    def t(self): # NOTE: ground-truth
        return self._t

    def __len__(self):
        return len(self._x)

    @dataset.convert_to_tensor
    def get_example(self, i):
        return self.x[i], self.y[i]


class BayesianMLP(nn.Module):

    def __init__(self, n_in, n_units, n_out, drop_ratio):
        super(BayesianMLP, self).__init__()

        self.n_in = n_in
        self.n_units = n_units
        self.n_out = n_out
        self.drop_ratio = drop_ratio

        self.l1 = nn.Linear(n_in, n_units)
        self.l2 = nn.Linear(n_units, n_units)
        self.l3 = nn.Linear(n_units, n_out)

        self.dropout = MCDropout(drop_ratio)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = self.dropout(h)
        h = F.relu(self.l2(h))
        h = self.dropout(h)
        return self.l3(h)


def train_phase(predictor, train, valid, args):

    # visualize
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(13,5))
    ax = sns.scatterplot(x=train.x.ravel(), y=train.y.ravel(), color='blue', s=55, alpha=0.3)
    ax.plot(train.x.ravel(), train.t.ravel(), color='red', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 15)
    plt.legend(['Ground-truth', 'Observation'])
    plt.title('Training data set')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'train_dataset.png'))
    plt.close()

    # setup iterators
    train_iter = iterators.SerialIterator(train, args.batchsize, shuffle=True)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    # setup a model
    device = torch.device(args.gpu)

    model = Regressor(predictor)
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
                      activation=None,
                      reduce_mean=None,
                      reduce_var=None)

    device = torch.device(args.gpu)
    model.to(device)

    infer = Inferencer(test_iter, model, device=args.gpu)

    pred, uncert = infer.run()


    # visualize
    x = test.x.ravel()
    t = test.t.ravel()
    pred = pred.ravel()
    uncert = uncert.ravel()

    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(13,5))
    ax = sns.scatterplot(x=x, y=pred, color='blue', s=75)
    ax.errorbar(x, pred, yerr=uncert, fmt='none', capsize=10, ecolor='gray', linewidth=1.5)
    ax.plot(x, t, color='red', linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 15)
    plt.legend(['Ground-truth', 'Prediction', 'Predicted variance'])
    plt.title('Result on testing data set')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'eval.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Example: Uncertainty estimates in regression')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=str, default='cuda:0',
                        help='GPU Device')
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
        predictor = BayesianMLP(n_in=1, n_units=args.unit, n_out=1, drop_ratio=0.1)

        # setup dataset
        train = Dataset(x_lim=(-5, 5), n_samples=1000)
        valid = Dataset(x_lim=(-5, 5), n_samples=1000)
        test  = Dataset(x_lim=(-10, 10), n_samples=500)

        # run
        if args.test_on_test:
            test_phase(predictor, test, args)
        elif args.test_on_valid:
            test_phase(predictor, valid, args)
        else:
            train_phase(predictor, train, valid, args)

if __name__ == '__main__':
    main()
