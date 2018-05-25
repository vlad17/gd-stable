#! /usr/bin/env bash

# Lints code:
#
#   # Lint gd_stable by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh gd_stable/somefile/*.py

set -euo pipefail

lint() {
    pylint "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint gd_stable
    else
        lint "$@"
    fi
}

main "$@"
