import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import reporter
from chainer.training import extensions
from chainer_bcnn.functions import mc_dropout
from chainer_bcnn.functions.loss import noised_mean_squared_error
from chainer_bcnn.links import Regressor
from chainer_bcnn.links import MCSampler
from chainer_bcnn.inference import Inferencer
from chainer_bcnn.utils import fixed_seed

def noise(mask, scale):
    ret = np.random.rand(*mask.shape) - 0.5
    ret *= (2.*scale*mask)
    return ret

class Dataset(chainer.dataset.DatasetMixin):

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

        mask = np.logical_and(-3 < x, x < -1)
        y = t + noise(mask, scale=2)

        self._func = func
        self._x_lim = x_lim
        self._x = x.astype(dtype)
        self._y = y.astype(dtype)
        self._t = t.astype(dtype)

    @property
    def x(self): # NOTE: input
        return self._x

    @property
    def y(self): # NOTE: observation
        return self._y

    @property
    def t(self): # NOTE: ground-truth
        return self._t

    def __len__(self):
        return len(self._x)

    def get_example(self, i):
        return self.x[i], self.y[i]


class BayesianMLP(chainer.Chain):

    def __init__(self, n_units, n_out, drop_ratio):
        super(BayesianMLP, self).__init__()

        self.n_units = n_units
        self.n_out = n_out
        self.drop_ratio = drop_ratio

        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)
            self.l3_sigma = L.Linear(None, n_out, nobias=True, initialW=chainer.initializers.Zero()) # NOTE: log_sigma

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = mc_dropout(h, self.drop_ratio)
        h = F.relu(self.l2(h))
        h = mc_dropout(h, self.drop_ratio)

        out = self.l3(h)
        sigma = self.l3_sigma(h)

        reporter.report({'sigma': F.mean(sigma)}, self)

        return out, sigma


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
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    # setup a model
    lossfun = noised_mean_squared_error
    accfun = lambda y, t: F.mean_absolute_error(y[0], t)

    model = Regressor(predictor, lossfun=lossfun, accfun=accfun)

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

        trainer.extend(
            extensions.PlotReport(
                ['main/predictor/sigma', 'validation/main/predictor/sigma'],
                'epoch', file_name='sigma.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'main/predictor/sigma', 'validation/main/predictor/sigma',
         'elapsed_time']))

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
                      activation=[F.identity, F.exp],
                      reduce_mean=None,
                      reduce_var=None)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    infer = Inferencer(test_iter, model, device=args.gpu)

    pred, epistemic_uncert, aleatory_uncert, _ = infer.run()

    # visualize
    x = test.x.ravel()
    t = test.t.ravel()
    pred = pred.ravel()
    epistemic_uncert = epistemic_uncert.ravel()
    aleatory_uncert = aleatory_uncert.ravel()

    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(13,5))
    ax = sns.scatterplot(x=x, y=pred, color='blue', s=75)
    ax.errorbar(x, pred, yerr=epistemic_uncert, fmt='none', capsize=10, ecolor='gray', linewidth=1.5)
    ax.plot(x, t, color='red', linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 15)
    plt.legend(['Ground-truth', 'Prediction', 'Epistemic uncertainty'])
    plt.title('Result on testing data set')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'eval_epistemic.png'))
    plt.close()

    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(13,5))
    ax = sns.scatterplot(x=x, y=pred, color='blue', s=75)
    ax.errorbar(x, pred, yerr=aleatory_uncert, fmt='none', capsize=10, ecolor='gray', linewidth=1.5)
    ax.plot(x, t, color='red', linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 15)
    plt.legend(['Ground-truth', 'Prediction', 'Aleatoric uncertainty'])
    plt.title('Result on testing data set')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'eval_aleatoric.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Example: Uncertainty estimates in regression')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
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
        predictor = BayesianMLP(n_units=args.unit, n_out=1, drop_ratio=0.1)

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
