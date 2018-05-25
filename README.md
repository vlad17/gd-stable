# Stability of GD in NNs [![Build Status](https://travis-ci.com/vlad17/lbs.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/gd-stable)

## Setup

See `setup.py` for necessary python packages. Requires a linux x64 box.

```
conda create -y -n gd-stable-env python=3.5
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
