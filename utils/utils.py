import os
import torch
from torch import nn

__all__ = [
    'AverageMeter', 'freeze_model', 'cuda_to_cpu', 'load_module_state_dict',
    'check_mkdir', 'check_makedirs'
]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def freeze_model(model):
    for m in model.modules():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def cuda_to_cpu(filename):
    checkpoint = torch.load(filename + '.pth')
    state_dict = checkpoint['state_dict'].copy()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()

    if 'optimizer' in checkpoint.keys():
        optimizer = checkpoint['optimizer'].copy()
        for k0, _ in optimizer['state'].items():
            for k, v in optimizer['state'][k0].items():
                if torch.is_tensor(v):
                    optimizer['state'][k0][k] = v.cpu()

    new_state = {
        'state_dict': state_dict,
        'optimizer': optimizer,
        'epoch': checkpoint['epoch']
    }
    torch.save(new_state, filename + '_cpu.pth')
    print("CPU verision: " + filename + '_cpu.pth')


def load_module_state_dict(net, state_dict, add=None, strict=False):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if add is not None:
            name = add + name
        if name in own_state:
            print(name)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
