import numpy as np
import torch
from pytorch_trainer.iterators import SerialIterator
from pytorch_bcnn.models import UNet, BayesianUNet
from pytorch_bcnn.links import MCSampler
from pytorch_bcnn.inference import Inferencer

from pytorch_trainer.dataset import DatasetMixin
from pytorch_trainer.dataset import convert_to_tensor


class Dataset(DatasetMixin):

    def __init__(self, n_samples, shape, dtype=np.float32):
        self._n_samples = n_samples
        self._shape = shape
        self._dtype = dtype

    def __len__(self):
        return self._n_samples

    @convert_to_tensor
    def get_example(self, i):
        return np.random.rand(*self._shape).astype(self._dtype)


def test(predictor, shape, batch_size, gpu, to_numpy):

    print('------')

    n_samples = 10
    dataset = Dataset(n_samples, shape)

    model = MCSampler(predictor, mc_iteration=5)
    model.eval()

    device = torch.device(gpu)
    model.to(device)

    iterator = SerialIterator(dataset, batch_size, repeat=False)

    infer = Inferencer(iterator, model, device=gpu, to_numpy=to_numpy)

    ret = infer.run()

    if isinstance(ret, (list, tuple)):
        for r in ret:
            print(r.shape)
            print(r.__class__)
    else:
        print(ret.shape)
        print(ret.__class__)


def main():
    test(BayesianUNet(ndim=2, in_channels=1, out_channels=5),
         (1, 200, 300),
         batch_size=2,
         gpu='cuda',
         to_numpy=True)

    test(BayesianUNet(ndim=3, in_channels=1, out_channels=5, nlayer=3),
         (1, 64, 64, 64),
         batch_size=2,
         gpu='cuda',
         to_numpy=True)


if __name__ == '__main__':
    main()
