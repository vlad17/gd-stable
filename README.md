# Stability of GD in NNs [![Build Status](https://travis-ci.com/vlad17/gd-stable.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/gd-stable)

## Setup

See `setup.py` for necessary python packages. Requires a linux x64 box.

```
# MKL on RISE machines messes things up for some reason
conda create -y -n gd-stable-env python=3.5 numpy nomkl
source activate gd-stable-env
./scripts/install-pytorch.sh
pip install --no-cache-dir --editable .
```

## Scripts

All scripts are available in `scripts/`, and should be run from the repo root in the `lbs-env`.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `tests.sh` | runs tests |
| `install-pytorch.sh` | infer python and cuda versions, use them to install pytorch |
| `format.sh` | auto-format the entire `gd_stable` directory |

## Example

All mainfiles are documented. Run `python gd_stable/main/*.py --help` for any `*` for details.

```
DEPTH=5
WIDTH=32
# generate the true test net
python gd_stable/main/generate_network.py --depth ${DEPTH} --width ${WIDTH}
# train a new net on a sampled dataset (usually needs grad norm clipping)
python gd_stable/main/train.py --depth ${DEPTH} --width ${WIDTH} --learning_rate 0.01 --grad_norm_clip 1 --samples 1024 --steps 1000
# view corresponding result
xdg-open ./data/plot-${DEPTH}-${WIDTH}.pdf
```
