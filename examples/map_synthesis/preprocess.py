import os
import glob
import cv2
import numpy as np
import tqdm
import urllib.request
from shutil import copyfile
import tarfile

from chainer_bcnn.data import load_image, save_image


def my_hook(t): # https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def download(url, out):

    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not os.path.exists(out):
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, ncols=80) as t:
            urllib.request.urlretrieve (url, out, reporthook=my_hook(t))


def preprocess_map(root, size, out):

    os.makedirs(out, exist_ok=True)

    files = glob.glob(os.path.join(root, '*.jpg'))

    for i, f in enumerate(files):

        img, _ = load_image(f)
        _, w, _ = img.shape

        img_a = img[:,:w//2,:].astype(np.float32)
        img_b = img[:,w//2:,:].astype(np.float32)

        img_a = cv2.resize(img_a, size)
        img_b = cv2.resize(img_b, size)

        img_a /= 127.5
        img_b /= 127.5

        img_a -= 1.
        img_b -= 1.

        save_image(os.path.join(out, '%04d_a.mha' % i), img_a)
        save_image(os.path.join(out, '%04d_b.mha' % i), img_b)


if __name__ == '__main__':

    url = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz'
    temp = os.path.join('./temp', os.path.basename(url))
    download(url, temp)

    with tarfile.open(temp, 'r:*') as tar:
        tar.extractall('./temp')

    preprocess_map('./temp/maps/train', (286,286), './preprocessed/train')
    preprocess_map('./temp/maps/val',   (256,256), './preprocessed/val')
