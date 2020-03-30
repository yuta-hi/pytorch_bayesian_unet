from __future__ import absolute_import

import torch
from abc import ABCMeta, abstractmethod
import json

def _len_children(chain):
    if not hasattr(chain, 'children'):
        return None
    return len([l for l in chain.children()])

def _freeze_layers(chain, name=None, startwith=None, endwith=None,
                   recursive=True, verbose=False):

    for l_name, l in chain.named_children():

        flag = False
        if name is not None:
            flag = flag or (l_name == name)
        if startwith is not None:
            flag = flag or l_name.startswith(startwith)
        if endwith is not None:
            flag = flag or l_name.endswith(endwith)

        if flag:
            l = getattr(chain, l_name)
            l.requires_grad_(False)
            if verbose == True:
                print('disabled update:', l_name)

        if recursive and hasattr(l, 'children'):
            _freeze_layers(l, name,
                           startwith, endwith,
                           recursive, verbose)

def _show_statistics(chain):

    def _show_statistics_depth(chain, depth):

            depth += 1

            for name, l in chain.named_children():
                l = getattr(chain, name)
                print('--'*depth, name)

                if hasattr(l, 'children'):
                    _show_statistics_depth(l, depth)

            if not hasattr(chain, 'children') or _len_children(chain) == 0:

                # parameters
                print('  '*depth, '(params)')
                for name, p in chain.named_parameters():
                    summary = ['  '*depth + '    %s:' % name]
                    if p.data is not None:
                        summary.append('%.3e +- %.3e' % ((p.data.mean()), (p.data.std())))
                        summary.append(list(p.data.shape))
                        if hasattr(p, 'requires_grad'):
                            if not p.requires_grad: summary.append('freeze')
                    else:
                        summary.append(None)
                    print(*summary)


    for name, l in chain.named_children():
        print(name)
        _show_statistics_depth(l, depth=0)


class Model(torch.nn.Module, metaclass=ABCMeta):
    """ Base class of Models (e.g., U-Net)
    """

    def freeze_layers(self, name=None,
                      startwith=None, endwith=None,
                      recursive=True, verbose=False):
        _freeze_layers(self, name,
                       startwith, endwith,
                       recursive, verbose)

    def show_statistics(self):
        _show_statistics(self)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_freezed_params(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def save_args(self, out):
        args = self._args.copy()
        ignore_keys = ['__class__', 'self']
        for key in ignore_keys:
            if key in args.keys():
                args.pop(key)

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(args, f, ensure_ascii=False, indent=4)

    def __getitem__(self, name):
        return getattr(self, name)

    @abstractmethod
    def forward(self, x, **kwargs):
        '''
        Args:
            x (~chainer.Variable)
            kwargs: Optional arguments will be contained.
        Return:
            o (~chainer.Variable)
        '''
        raise NotImplementedError()


from .unet import UNetBase # NOQA
from .unet import UNet # NOQA
from .unet import BayesianUNet # NOQA

from .discriminators import DiscriminatorBase # NOQA
from .discriminators import PatchDiscriminator # NOQA
