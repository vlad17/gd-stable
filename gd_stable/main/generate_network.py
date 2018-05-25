"""
Generate a random neural network with weights on the unit ball.

For example,

    python gd_stable/main/generate_network.py --depth 5 --width 32

will generate a 5-by-32 MLP (with parameters on the unit ball) and write
it out to ./data/mlp-5-32.pth (overwriting existing files).

We fix input sizes to 64 and output sizes to be 1.

"""

from absl import app
from absl import flags
import numpy as np
import torch

from gd_stable.mlp import MLP
from gd_stable.utils import seed_all

flags.DEFINE_integer('depth', 5, 'depth of generated network')
flags.DEFINE_integer('width', 32, 'width of generated network')


def _main(_):
    seed_all(1)

    input_size = 64
    hiddens = [flags.FLAGS.width] * flags.FLAGS.depth
    output_size = 1
    print('generating MLP {}'.format(' -> '.join(
        map(str, [input_size] + hiddens + [output_size]))))
    mlp = MLP(input_size, hiddens, output_size)

    param_counts = [p.numel() for p in mlp.parameters()]
    num_parameters = sum(param_counts)
    print('num parameters', num_parameters)

    new_params = np.random.randn(num_parameters)
    new_params /= np.linalg.norm(new_params)

    idx_ends = list(np.cumsum(param_counts))
    idx_begins = [0] + idx_ends[:-1]
    for begin, end, p in zip(idx_begins, idx_ends, mlp.parameters()):
        p.data[:] = torch.from_numpy(new_params[begin:end].reshape(
            p.data.shape))

    torch.save(
        mlp.state_dict(), './data/mlp-{}-{}.pth'.format(
            flags.FLAGS.depth, flags.FLAGS.width))


if __name__ == '__main__':
    app.run(_main)
