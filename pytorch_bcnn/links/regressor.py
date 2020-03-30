from __future__ import absolute_import

import torch.nn as nn

from .classifier import Classifier

class Regressor(Classifier):
    """ A simple regressor model.
    It computes the loss and accuracy based on given input/label pair.

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
        activation (callable):
            Function that apply final activation functions to preditions.
            You can specify one of evaluation functions from
            :doc:`built-in activation functions </reference/functions>`, or
            your own activation function.
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
        activation (callable):
            Activation function after the predictor output.
            See the description in the arguments for details.
        x (~chainer.Variable or tuple): Inputs for the last minibatch.
        y (~chainer.Variable or tuple): Predictions for the last minibatch.
        t (~chainer.Variable or tuple): Labels for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.

    See also: ~chainer_bcnn.links.Classifier
    """

    def __init__(self, predictor,
                 lossfun=nn.MSELoss(reduction='mean'),
                 accfun=nn.L1Loss(reduction='mean'),
                 activation=None,
                 x_keys=(0), t_keys=(-1)):

        super(Regressor, self).__init__(
            predictor, lossfun, accfun, activation,
            x_keys, t_keys
        )
