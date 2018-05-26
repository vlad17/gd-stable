"""
Generate a random neural network with weights on the unit ball.

For example,

    python gd_stable/main/generate_network.py --depth 5 --width 32

will generate a 5-by-32 MLP (with parameters on the unit ball) and write
it out to ./data/mlp-5-32.pth (overwriting existing files).

We fix input sizes and output sizes (output size is just the width).

"""

from absl import app
from absl import flags
import numpy as np
import torch

from gd_stable.mlp import MLP
from gd_stable.utils import seed_all, num_parameters, fromflat

flags.DEFINE_integer('depth', 5, 'depth of generated network')
flags.DEFINE_integer('width', 32, 'width of generated network')


def _main(_):
    seed_all(1234)

    input_size = 64
    hiddens = [flags.FLAGS.width] * flags.FLAGS.depth
    output_size = flags.FLAGS.width
    print('generating MLP {}'.format(' -> '.join(
        map(str, [input_size] + hiddens + [output_size]))))
    mlp = MLP(input_size, hiddens, output_size)

    n = num_parameters(mlp)
    print('num parameters', n)

    new_params = np.random.randn(n)
    # new_params has l2 norm of order sqrt(n)
    # but that results in output values that are way too large
    # square-rooting the norm seems to give interesting behavior
    new_params /= np.sqrt(np.linalg.norm(new_params))
    fromflat(mlp, torch.from_numpy(new_params))
    state_dict = {
        'input_size': input_size,
        'hiddens': hiddens,
        'output_size': output_size,
        'network': mlp.state_dict(),
    }
    torch.save(
        state_dict, './data/mlp-{}-{}.pth'.format(flags.FLAGS.depth,
                                                  flags.FLAGS.width))


if __name__ == '__main__':
    app.run(_main)
