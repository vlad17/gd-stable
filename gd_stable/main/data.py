"""
Take in a network and then export a bunch of samples from it.

For example,

    python gd_stable/main/data.py --depth 5 --width 32 --nsamples 10000

will read in the network in ./data/mlp-5-32.pth and generate 10000
input-output pairs according to the network's current weights, writing
them out to ./data/data-5-32-10000.pth (which then gets 'inputs' and 'outputs'
tensors).

We fix input sizes to 64 and output sizes to be 1.

Samples generate are standard normal distributed.
"""

from absl import app
from absl import flags
import torch
import numpy as np

from gd_stable.mlp import MLP
from gd_stable.utils import seed_all

flags.DEFINE_integer('depth', 5, 'depth of generated network')
flags.DEFINE_integer('width', 32, 'width of generated network')
flags.DEFINE_integer('nsamples', 10000, 'number of samples to generate')


def _main(_):
    seed_all(1)

    input_size = 64
    hiddens = [flags.FLAGS.width] * flags.FLAGS.depth
    output_size = 1
    mlp = MLP(input_size, hiddens, output_size)
    savefile = './data/mlp-{}-{}.pth'.format(flags.FLAGS.depth,
                                             flags.FLAGS.width)
    mlp.load_state_dict(torch.load(savefile))
    print('loaded MLP {} from {}'.format(
        ' -> '.join(map(str, [input_size] + hiddens + [output_size])),
        savefile))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('yes using GPU')
    else:
        device = torch.device('cpu')
        print('not using GPU')
    mlp = mlp.to(device)

    batch_size = 1024
    ns = flags.FLAGS.nsamples
    inputs = torch.randn((ns, input_size))
    outputs = torch.zeros((ns, output_size))
    for i in range(0, ns, batch_size):
        low = i
        high = min(ns, i + batch_size)
        batch = inputs[low:high].to(device)
        outputs[low:high] = mlp(batch)

    datafile = './data/data-{}-{}-{}.pth'.format(flags.FLAGS.depth,
                                                 flags.FLAGS.width, ns)
    torch.save({'inputs': inputs, 'outputs': outputs}, datafile)


if __name__ == '__main__':
    app.run(_main)
