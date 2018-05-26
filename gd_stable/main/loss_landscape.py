"""
Accepts the trace of two training computations differing only in the number of
samples the had, as well as the original loss function.

Plots the two loss curves along with the loss function along the principle
component of the updates.
"""

import itertools
import os

from absl import app
from absl import flags
import torch

from gd_stable.mlp import MLP
from gd_stable.utils import seed_all, import_matplotlib

flags.DEFINE_string('first', None, 'first iterate training file')
flags.DEFINE_string('second', None, 'second iterate training file')
flags.DEFINE_string('true_network', None, 'original network that generates '
                    'the true labels')

def _main(_):
    seed_all(1)

    state_dict = torch.load(flags.FLAGS.true_network)
    input_size = state_dict['input_size']
    hiddens = state_dict['hiddens']
    output_size = state_dict['output_size']
    mlp_true = MLP(input_size, hiddens, output_size)
    mlp_true.load_state_dict(state_dict['network'])

    first = torch.load(flags.FLAGS.first)
    assert input_size == first['input_size'], input_size
    assert hiddens == first['hiddens'], hiddens
    assert output_size == first['output_size'], output_size
    second = torch.load(flags.FLAGS.second)
    assert input_size == second['input_size'], input_size
    assert hiddens == second['hiddens'], hiddens
    assert output_size == second['output_size'], output_size

    ns_first = first['samples']
    ns_second = second['samples']

    # TODO: use loss fn from the one with more samples? plot both loss functions?

    print('loaded both networks')

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
        names.append('{} - samples {}'.format(
            i, state_dict['samples']))

        del state_dict # iterates take up a lot of space, kill it


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

if __name__ == '__main__':
    flags.mark_flag_as_required('first')
    flags.mark_flag_as_required('second')
    flags.mark_flag_as_required('true_network')
    app.run(_main)
