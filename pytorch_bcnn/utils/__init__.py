from __future__ import absolute_import

import os
import numpy as np
import contextlib
import warnings
import tempfile
import shutil
import argparse
import json


@contextlib.contextmanager
def fixed_seed(seed, strict=False):
    """Fix random seed to improve the reproducibility.

    Args:
        seed (float): Random seed
        strict (bool, optional): If True, cuDNN works under deterministic mode.
            Defaults to False.

    TODO: Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`.
          If your dataset has stochastic behavior, such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.
    """

    import random
    import torch
    import copy

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if strict:
        warnings.warn('Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`. \
          If your dataset has stochastic behavior such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.')

        _deterministic = copy.copy(torch.backends.cudnn.deterministic)
        _benchmark = copy.copy(torch.backends.cudnn.benchmark)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yield

    if strict:
        torch.backends.cudnn.deterministic = _deterministic
        torch.backends.cudnn.benchmark = _benchmark


# https://github.com/chainer/chainerui/blob/master/chainerui/utils/tempdir.py
@contextlib.contextmanager
def tempdir(**kwargs):
    # A context manager that defines a lifetime of a temporary directory.
    ignore_errors = kwargs.pop('ignore_errors', False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)

# https://github.com/chainer/chainerui/blob/master/chainerui/utils/save_args.py
def convert_dict(conditions):
    if isinstance(conditions, argparse.Namespace):
        return vars(conditions)
    return conditions

# https://github.com/chainer/chainerui/blob/master/chainerui/utils/save_args.py
def save_args(conditions, out_path):
    """A util function to save experiment condition for job table.

    Args:
        conditions (:class:`argparse.Namespace` or dict): Experiment conditions
            to show on a job table. Keys are show as table header and values
            are show at a job row.
        out_path (str): Output directory name to save conditions.

    """

    args = convert_dict(conditions)

    try:
        os.makedirs(out_path)
    except OSError:
        pass

    with tempdir(prefix='args', dir=out_path) as tempd:
        path = os.path.join(tempd, 'args.json')
        with open(path, 'w') as f:
            json.dump(args, f, indent=4)

        new_path = os.path.join(out_path, 'args')
        shutil.move(path, new_path)


# https://github.com/chainer/chainer/blob/v7.1.0/chainer/training/extensions/_snapshot.py
def _find_snapshot_files(fmt, path):
    '''Only prefix and suffix match
    TODO(kuenishi): currently clean format string such as
    "snapshot{.iteration}.npz" can only be parsed, but tricky (or
    invalid) formats like "snapshot{{.iteration}}.npz" are hard to
    detect and to properly show errors, just ignored or fails so far.
    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.
    Returns:
        A sorted list of pair of ``mtime, filename``, whose file
        name that matched the format ``fmt`` directly under ``path``.
    '''
    prefix = fmt.split('{')[0]
    suffix = fmt.split('}')[-1]

    matched_files = (file for file in os.listdir(path)
                     if file.startswith(prefix) and file.endswith(suffix))

    def _prepend_mtime(f):
        t = os.stat(os.path.join(path, f)).st_mtime
        return (t, f)

    return sorted(_prepend_mtime(file) for file in matched_files)

# https://github.com/chainer/chainer/blob/v7.1.0/chainer/training/extensions/_snapshot.py
def _find_latest_snapshot(fmt, path):
    """Finds the latest snapshots in a directory
    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.
    Returns:
        Latest snapshot file, in terms of a file that has newest
        ``mtime`` that matches format ``fmt`` directly under
        ``path``. If no such file found, it returns ``None``.
    """
    snapshot_files = _find_snapshot_files(fmt, path)

    if len(snapshot_files) > 0:
        _, filename = snapshot_files[-1]
        return filename

    return None


def find_latest_snapshot(fmt, path, return_fullpath=True):
    '''Alias of :func:`_find_latest_snapshot`
    '''
    ret = _find_latest_snapshot(fmt, path)

    if ret is None:
        raise FileNotFoundError('cannot find snapshot for <%s>' %
                                    os.path.join(path, fmt))

    if return_fullpath:
        return os.path.join(path, ret)

    return ret
