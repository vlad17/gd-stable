"""
Accepts the trace of one or more training computations and plots their
loss curves.

If multiple curves are supplied, they should all share the same step
indices (the time at training when the loss was sampled).
All curves are assumed to come from the same architecture.
"""

import itertools
import os

from absl import app
from absl import flags
import torch

from gd_stable.utils import seed_all, import_matplotlib

flags.DEFINE_multi_string(
    'file',
    None,
    'A file containing iterates and loss '
    'curves from training',
    short_name='f')
flags.DEFINE_string('outfile', './data/out.pdf', 'output file name')


def _main(_):
    seed_all(1)

    steps = None
    names = []
    train_losses = []
    test_losses = []
    input_size, hiddens, output_size = None, None, None
    for i, filename in enumerate(flags.FLAGS.file):
        state_dict = torch.load(filename)

        if steps is None:
            steps = state_dict['steps']
        else:
            assert _issorted(state_dict['steps']), filename
            assert set(state_dict['steps']) == set(steps), filename

        if input_size is None:
            input_size = state_dict['input_size']
        else:
            assert input_size == state_dict['input_size'], filename

        if output_size is None:
            output_size = state_dict['output_size']
        else:
            assert output_size == state_dict['output_size'], filename

        if hiddens is None:
            hiddens = state_dict['hiddens']
        else:
            assert hiddens == state_dict['hiddens'], filename

        train_losses.append(state_dict['train_losses'])
        test_losses.append(state_dict['test_losses'])
        names.append('{} - samples {}'.format(i, state_dict['samples']))

        del state_dict  # iterates take up a lot of space, kill it

    network_str = '-'.join(map(str, [input_size] + hiddens + [output_size]))
    plt = import_matplotlib()
    plt.clf()
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.title('{} loss curves'.format(network_str))
    for train, test, name, c in zip(train_losses, test_losses, names, colors):
        plt.semilogy(steps, train, label='train ' + name, color=c)
        plt.semilogy(steps, test, label='test ' + name, ls='--', color=c)
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    os.makedirs(os.path.dirname(flags.FLAGS.outfile), exist_ok=True)
    plt.savefig(flags.FLAGS.outfile)


def _issorted(ls):
    for x, y in zip(ls, ls[1:]):
        if x > y:
            return False
    return True


if __name__ == '__main__':
    flags.mark_flag_as_required('file')
    app.run(_main)
