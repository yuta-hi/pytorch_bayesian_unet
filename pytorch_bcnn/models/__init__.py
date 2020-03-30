from __future__ import absolute_import

import chainer
from abc import ABCMeta, abstractmethod
import json

def _len_children(chain):
    if not hasattr(chain, 'children'):
        return None
    return len([l for l in chain.children()])

def _freeze_layers(chain, name=None, startwith=None, endwith=None,
                   recursive=True, verbose=False):

    for l in chain.children():

        flag = False
        if name is not None:
            flag = flag or (l.name == name)
        if startwith is not None:
            flag = flag or l.name.startswith(startwith)
        if endwith is not None:
            flag = flag or l.name.endswith(endwith)

        if flag:
            l = getattr(chain, l.name)
            l.disable_update()
            if verbose == True:
                print('disabled update:', l.name)

        if recursive and hasattr(l, 'children'):
            _freeze_layers(l, name,
                           startwith, endwith,
                           recursive, verbose)

def _show_statistics(chain):

    def _show_statistics_depth(chain, depth):

            xp = chain.xp
            depth += 1

            for l in chain.children():
                l = getattr(chain, l.name)
                print('--'*depth, l.name)

                if hasattr(l, 'children'):
                    _show_statistics_depth(l, depth)

            if not hasattr(chain, 'children') or _len_children(chain) == 0:

                # parameters
                print('  '*depth, '(params)')
                for p in chain.params():
                    summary = ['  '*depth + '    %s:' % p.name]
                    if p.data is not None:
                        summary.append('%.3e +- %.3e' % (xp.mean(p.data), xp.std(p.data)))
                        summary.append(p.data.shape)
                        if hasattr(p.update_rule, 'enabled'):
                            if not p.update_rule.enabled: summary.append('freeze')
                    else:
                        summary.append(None)
                    print(*summary)

                # hooks
                if len(chain.local_link_hooks) > 0:
                    print('  '*depth, '(hooks)')
                    for name, hook in chain.local_link_hooks.items():
                        print('  '*depth + '    %s' % name)


    for l in chain.children():
        print(l.name)
        _show_statistics_depth(l, depth=0)

class Model(chainer.Chain, metaclass=ABCMeta):
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

    def save_args(self, out):
        args = self._args.copy()
        ignore_keys = ['__class__', 'self']
        for key in ignore_keys:
            if key in args.keys():
                args.pop(key)

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(args, f, ensure_ascii=False, indent=4)

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
