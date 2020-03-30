from __future__ import absolute_import

import numpy as np
import torch

class ReplayBuffer(object):
    """ Buffer for handling the experience replay.
    Args:
        size (int): buffer size
        p (float): probability to evoke the past experience

    See also:
     https://arxiv.org/pdf/1612.07828.pdf
     https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p
        self._buffer = []


    @property
    def buffer(self):
        if len(self._buffer) == 0:
            return None
        return self._buffer

    def __call__(self, samples):

        if not isinstance(samples, torch.Tensor):
            samples = torch.as_tensor(samples)

        n_samples = len(samples)

        if self.size == 0:
            return samples

        if len(self._buffer) < self.size:
            if len(self._buffer) == 0:
                self._buffer = samples
            self._buffer = torch.cat((self._buffer, samples))
            return samples

        # evoke the memory
        random_bool = np.random.rand(n_samples) < self.p
        replay_indices = np.random.randint(0, len(self._buffer), size=n_samples)[random_bool]
        sample_indices = np.random.randint(0, n_samples, size=n_samples)[random_bool]

        self._buffer[replay_indices], samples[sample_indices] \
            = samples[sample_indices], self._buffer[replay_indices] # swap

        return samples


if __name__ == '__main__':

    import numpy as np
    import torch

    buffer = ReplayBuffer(10)
    print(buffer.buffer)

    for i in range(20):
        a = buffer(torch.as_tensor(np.zeros((2,3,4,5)) + i))
        print(i, a)
    print(a.shape)
    print(a.__class__)

    print(len(buffer.buffer))
    print(buffer.buffer.shape)
