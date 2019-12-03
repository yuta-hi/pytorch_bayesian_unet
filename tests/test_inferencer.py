import numpy as np
import chainer
from chainer.dataset import DatasetMixin
from chainer.iterators import SerialIterator
from chainer_bcnn.models import BayesianUNet
from chainer_bcnn.links import MCSampler
from chainer_bcnn.inference import Inferencer


class Dataset(DatasetMixin):

    def __init__(self, n_samples, shape, dtype=np.float32):
        self._n_samples = n_samples
        self._shape = shape
        self._dtype = dtype

    def __len__(self):
        return self._n_samples

    def get_example(self, i):
        return np.random.rand(*self._shape).astype(self._dtype)


def test(predictor, shape, batch_size, gpu, to_cpu):

    print('------')

    n_samples = 10
    dataset = Dataset(n_samples, shape)

    model = MCSampler(predictor, mc_iteration=5)

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    iterator = SerialIterator(dataset, batch_size, repeat=False)

    infer = Inferencer(iterator, model, device=gpu, to_cpu=to_cpu)

    ret = infer.run()

    if isinstance(ret, (list, tuple)):
        for r in ret:
            print(r.shape)
            print(r.__class__)
    else:
        print(ret.shape)
        print(ret.__class__)


def main():
    test(BayesianUNet(ndim=2, out_channels=5),
         (1, 200, 300),
         batch_size=2,
         gpu=0,
         to_cpu=True)

    test(BayesianUNet(ndim=3, out_channels=5, nlayer=3),
         (1, 64, 64, 64),
         batch_size=2,
         gpu=0,
         to_cpu=True)


if __name__ == '__main__':
    main()
