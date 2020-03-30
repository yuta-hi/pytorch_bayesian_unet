from __future__ import absolute_import

import numpy as np
from pytorch_trainer.dataset import DatasetMixin
from pytorch_trainer.dataset import convert_to_tensor
import tqdm
import glob
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from ..data import load_image  # NOQA

class BaseDataset(DatasetMixin, metaclass=ABCMeta):
    """ Base class of dataset

    Args:
        root (str): Directory to the dataset
        patients (list, optional): List of patient names. Defaults to [].
        classes (None or list, optional): List of class names. Defaults to None.
        dtypes (dict, optional): An dictionary of data types. Defaults to {}.
        filenames (dict, optional): An dictionary of wildcard to filenames.
            Each filename can be a format string using '{root}' and '{patient}'. Defaults to {}.
        normalizer (callable, optional): An callable function for normalization. Defaults to None.
        augmentor (callable, optional): An callable function for data augmentation. Defaults to None.
    """
    def __init__(self,
                 root,
                 patients=[],
                 classes=None,
                 dtypes={},
                 filenames={},
                 normalizer=None,
                 augmentor=None):

        super(BaseDataset, self).__init__()

        assert isinstance(patients, (list, np.ndarray)), \
            'please specify the patient names..'
        if classes is not None:
            if isinstance(classes, list):
                classes = np.asarray(classes)
            assert isinstance(classes, np.ndarray), \
                'class names should be list or np.ndarray..'
        assert isinstance(dtypes, dict), \
            'please specify the dtype per each file..'
        assert isinstance(filenames, dict), \
            'please specify the filename per each file..'
        if normalizer is not None:
            assert callable(normalizer), 'normalizer should be callable..'
        if augmentor is not None:
            assert callable(augmentor), 'augmentor should be callable..'

        # initialize
        files = OrderedDict()
        file_sizes = []

        for key in filenames.keys():

            files[key] = []
            for p in tqdm.tqdm(patients, desc='Collecting %s files' % key, ncols=80):
                files[key].extend(
                    glob.glob(filenames[key].format(root=root, patient=p)))

            if len(files[key]) == 0:
                warnings.warn('%s files are not found.. ' % key)
            file_sizes.append(len(files[key]))

        assert all(file_sizes[0] == s for s in file_sizes), \
            'the number of files must be the same..'

        self._root = root
        self._patients = patients
        self._classes = classes
        self._dtypes = dtypes
        self._filenames = filenames
        self._files = files
        self._normalizer = normalizer
        self._augmentor = augmentor

    def __len__(self):
        key = list(self._files.keys())[0]
        return len(self._files[key])

    @property
    def classes(self):
        return self._classes

    @property
    def n_classes(self):
        if self.classes is None:
            return None
        return len(self.classes)

    @property
    def files(self):
        return self._files

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def normalizer(self):
        return self._normalizer

    @property
    def augmentor(self):
        return self._augmentor

    @augmentor.deleter
    def augmentor(self):
        self._augmentor = None

    @classmethod
    @abstractmethod
    def normalize(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def denormalize(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    @convert_to_tensor
    def get_example(self, i):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def __copy__(self):
        """Copy the class instance"""
        raise NotImplementedError()

from .volume import VolumeDataset  # NOQA
from .image import ImageDataset  # NOQA

def train_valid_split(train, valid_ratio):

    if isinstance(train, BaseDataset):

        valid = train.__copy__()

        n_samples = len(train)

        valid_indices = np.random.choice(np.arange(n_samples),
                                         int(valid_ratio * n_samples),
                                         replace=False)
        files = train.files

        for key in files.keys():
            valid._files[key] = np.asarray(files[key])[valid_indices]
            train._files[key] = np.delete(
                np.asarray(files[key]), valid_indices)

    elif isinstance(train, (list, np.ndarray)):

        valid = np.asarray(train)

        n_samples = len(train)

        valid_indices = np.random.choice(np.arange(n_samples),
                                         int(valid_ratio * n_samples),
                                         replace=False)

        valid = valid[valid_indices]
        train = np.delete(train, valid_indices)

    assert len(train) + len(valid) == n_samples

    return train, valid


def load_crossval_list(xls_file, index):
    import pandas as pd
    from distutils.version import LooseVersion

    if LooseVersion(pd.__version__) >= LooseVersion('0.21.0'):
        df = pd.read_excel(xls_file, sheet_name=index)
    else:
        df = pd.read_excel(xls_file, sheetname=index)

    train = df['train'].dropna().tolist()
    valid = df['valid'].dropna().tolist()
    test = df['test'].dropna().tolist()

    return {'train': train, 'valid': valid, 'test': test}
