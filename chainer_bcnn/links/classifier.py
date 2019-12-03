from __future__ import absolute_import

import chainer
from chainer import configuration
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from functools import partial

def _get_value(args, kwargs, key):

    if not (isinstance(key, (int, str))):
        raise TypeError('key must be int or str, but is %s' %
                        type(key))

    if isinstance(key, int):
        if not (-len(args) <= key < len(args)):
            msg = 'key %d is out of bounds' % key
            raise ValueError(msg)
        value = args[key]

    elif isinstance(key, str):
        if key not in kwargs:
            msg = 'key "%s" is not found' % key
            raise ValueError(msg)
        value = kwargs[key]

    return value

def get_values(args, kwargs, keys):

    getter = partial(_get_value,
                     args=args, kwargs=kwargs)

    if isinstance(keys, (list,tuple)):
        return [getter(key=key) for key in keys]

    return getter(key=keys)


class Classifier(link.Chain):

    """A simple classifier model.
    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.
    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable):
            Loss function.
            You can specify one of loss functions from
            :doc:`built-in loss functions </reference/functions>`, or
            your own loss function (see the example below).
            It should not be an
            :doc:`loss functions with parameters </reference/links>`
            (i.e., :class:`~chainer.Link` instance).
            The function must accept two argument (an output from predictor
            and its ground truth labels), and return a loss.
            Returned value must be a Variable derived from the input Variable
            to perform backpropagation on the variable.
        accfun (callable):
            Function that computes accuracy.
            You can specify one of evaluation functions from
            :doc:`built-in evaluation functions </reference/functions>`, or
            your own evaluation function.
            The signature of the function is the same as ``lossfun``.
        x_keys (tuple, int or str): Key to specify input variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.
            If you use multiple variables, please specify ``tuple`` of ``int`` or ``str``.
        t_keys (tuple, int or str): Key to specify label variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.
            If you use multiple variables, please specify ``tuple`` of ``int`` or ``str``.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable):
            Loss function.
            See the description in the arguments for details.
        accfun (callable):
            Function that computes accuracy.
            See the description in the arguments for details.
        x (~chainer.Variable or tuple): Inputs for the last minibatch.
        y (~chainer.Variable or tuple): Predictions for the last minibatch.
        t (~chainer.Variable or tuple): Labels for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.

    .. note::
        This link uses :func:`chainer.softmax_cross_entropy` with
        default arguments as a loss function (specified by ``lossfun``),
        if users do not explicitly change it. In particular, the loss function
        does not support double backpropagation.
        If you need second or higher order differentiation, you need to turn
        it on with ``enable_double_backprop=True``:
          >>> import chainer.functions as F
          >>> import chainer.links as L
          >>>
          >>> def lossfun(x, t):
          ...     return F.softmax_cross_entropy(
          ...         x, t, enable_double_backprop=True)
          >>>
          >>> predictor = L.Linear(10)
          >>> model = L.Classifier(predictor, lossfun=lossfun)
    """

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 x_keys=(0), t_keys=(-1)):

        super(Classifier, self).__init__()

        assert callable(predictor), 'predictor should be callable..'
        if lossfun is not None:
            assert callable(lossfun), 'lossfun should be callable..'
        if accfun is not None:
            assert callable(accfun), 'accfun should be callable..'


        with self.init_scope():
            self.predictor = predictor

        self.lossfun = lossfun
        self.accfun = accfun

        self.x_keys = x_keys
        self.t_keys = t_keys

        self._reset()

    def _reset(self):

        self.x = None
        self.y = None
        self.t = None

        self.loss = None
        self.accuracy = None

    def forward(self, *args, **kwargs):
        """Computes the loss value for input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.
        When ``label_key`` is ``int``, the corresponding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground truth
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.
        .. note::
            We set ``None`` to the attributes ``y``, ``loss`` and ``accuracy``
            each time before running the predictor, to avoid unnecessary memory
            consumption. Note that the variables set on those attributes hold
            the whole computation graph when they are computed. The graph
            stores interim values on memory required for back-propagation.
            We need to clear the attributes to free those values.
        Returns:
            ~chainer.Variable: Loss value.
        """

        self._reset()

        x = get_values(args, kwargs, self.x_keys)
        t = get_values(args, kwargs, self.t_keys)

        y = self.predictor(x)

        if t is not None:

            if self.lossfun is not None:
                self.loss = self.lossfun(y, t)
                reporter.report({'loss': self.loss}, self)

            if self.accfun is not None:
                self.accuracy = self.accfun(y, t)
                reporter.report({'accuracy': self.accuracy}, self)

        self.x = x
        self.y = y
        self.t = t

        if configuration.config.train:
            if self.loss is None:
                raise ValueError('loss is None..')
            return self.loss

        else:
            return self.y
