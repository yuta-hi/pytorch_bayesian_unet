import numpy as np
from chainer_bcnn.visualizer import ImageVisualizer
from chainer_bcnn.visualizer.image import _default_cmap


def test_classification_sparse():

    _categorical_cmaps = {
        'y': _default_cmap,
        't': _default_cmap,
    }

    _categorical_clims = {
        'x': (0., 1.),
    }

    _categorical_transforms = {
        'x': lambda x: x,
        'y': lambda x: x,
        't': lambda x: x,
    }

    # NOTE: out is an unexpected argument.
    visualizer = ImageVisualizer(transforms=_categorical_transforms,
                                 cmaps=_categorical_cmaps,
                                 clims=_categorical_clims)

    x = np.random.rand(3, 100, 200)
    y = np.random.randint(0, 10, (100, 200))
    t = np.random.randint(0, 10, (100, 200))

    for _ in range(3):
        visualizer.add_example(x, y, t)
    visualizer.save('test_classification_sparse.png')


def test_classification():

    _categorical_cmaps = {
        'y': _default_cmap,
        't': _default_cmap,
    }

    _categorical_clims = {
        'x': (0., 1.),
    }

    _categorical_transforms = {
        'x': lambda x: x,
        'y': lambda x: np.argmax(x, axis=0),
        't': lambda x: np.argmax(x, axis=0),
    }

    # NOTE: out is an unexpected argument.
    visualizer = ImageVisualizer(transforms=_categorical_transforms,
                                 cmaps=_categorical_cmaps,
                                 clims=_categorical_clims)

    x = np.random.rand(3, 100, 200)
    y = np.random.rand(10, 100, 200)
    t = np.random.rand(10, 100, 200)

    for _ in range(3):
        visualizer.add_example(x, y, t)
    visualizer.save('test_classification.png')


def test_regression():

    _regression_cmaps = None

    _regression_clims = {
        'x': (0., 1.),
        'y': (0., 1.),
        't': (0., 1.),
    }

    _regression_transforms = None

    visualizer = ImageVisualizer(transforms=_regression_transforms,
                                 cmaps=_regression_cmaps,
                                 clims=_regression_clims)

    x = np.random.rand(3, 100, 200)
    y = np.random.rand(5, 100, 200)
    t = np.random.rand(5, 100, 200)

    for _ in range(3):
        visualizer.add_example(x, y, t)
    visualizer.save('test_regression.png')


def main():
    test_classification_sparse()
    test_classification()
    test_regression()


if __name__ == '__main__':
    main()
