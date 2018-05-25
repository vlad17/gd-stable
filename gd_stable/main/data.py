"""
Generate a fake train/test set according to a randomly-created neural
network.

    python gd_stable/main/data.py --depth 5 --width 32 --nsamples 10000

Will generate a 5-by-32 MLP (with parameters on the unit ball)
"""

from absl import app
from absl import flags

from lbs.utils import seed_all

flags.DEFINE_integer('seed', 1, 'random seed')


def _main(_):
    pass


if __name__ == '__main__':
    app.run(_main)
