"""
Various utility functions used across several files.
"""

from contextlib import contextmanager
import sys
import time
import os
import collections
import hashlib
import random

import numpy as np
import torch


def _next_seeds(n):
    # deterministically generate seeds for envs
    # not perfect due to correlation between generators,
    # but we can't use urandom here to have replicable experiments
    # https://stats.stackexchange.com/questions/233061
    mt_state_size = 624
    seeds = []
    for _ in range(n):
        state = np.random.randint(2**31, size=mt_state_size)
        digest = hashlib.sha224(state.tobytes()).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)[0]
        seeds.append(int(seed))
        if seeds[-1] is None:
            seeds[-1] = int(state.sum())
    return seeds


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    print('seeding with seed {}'.format(seed))
    np.random.seed(seed)
    rand_seed, torch_cpu_seed, torch_gpu_seed = _next_seeds(3)
    random.seed(rand_seed)
    torch.manual_seed(torch_cpu_seed)
    torch.cuda.manual_seed_all(torch_gpu_seed)


def gpus():
    """Retrieve gpus from env var CUDA_VISIBLE_DEVICES"""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpulist = list(range(torch.cuda.device_count()))
    else:
        gpulist = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpulist = list(map(int, filter(None, gpulist)))
    gpulist.sort()
    return gpulist


class RollingAverageWindow:
    """Creates an automatically windowed rolling average."""

    def __init__(self, window_size):
        self._window_size = window_size
        self._items = collections.deque([], window_size)
        self._total = 0

    def update(self, value):
        """updates the rolling window"""
        if len(self._items) < self._window_size:
            self._total += value
            self._items.append(value)
        else:
            self._total -= self._items.popleft()
            self._total += value
            self._items.append(value)

    def value(self):
        """returns the current windowed avg"""
        return self._total / len(self._items)


def import_matplotlib():
    """import and return the matplotlib module in a way that uses
    a display-independent backend (import when generating images on
    servers"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


@contextmanager
def timeit(name, preprint=True):
    """Enclose a with-block with to debug-print out block runtime"""
    t = time.time()
    if preprint:
        print(name, end='')
        sys.stdout.flush()
    yield
    t = time.time() - t
    if not preprint:
        print(name, end='')
    print(' took {:0.1f} seconds'.format(t))


def toflat(model):
    """convert a pytorch module to its flattened parameter vector
    (put onto CPU)"""
    return torch.cat(
        tuple(p.data.view(-1).cpu().detach() for p in model.parameters()))


def fromflat(model, flat):
    """set the pytorch module to the parameters determined by the given
    flattened value"""
    param_counts = [p.numel() for p in model.parameters()]
    idx_ends = list(np.cumsum(param_counts))
    idx_begins = [0] + idx_ends[:-1]
    for begin, end, p in zip(idx_begins, idx_ends, model.parameters()):
        p.data[:] = flat[begin:end].view(*p.data.shape)


def num_parameters(model):
    """number of parameters in pytorch model"""
    return sum(p.numel() for p in model.parameters())
