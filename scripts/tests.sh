#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this.
#
#   ./scripts/tests.sh
#   ./scripts/tests.sh --dry-run

set -eo pipefail

set -u

if [ $# -gt 1 ] || [ $# -eq 1 ] && [ "$1" != "--dry-run" ] ; then
    echo 'usage: ./scripts/tests.sh [--dry-run]' 1>&2
    exit 1
fi

if [ $# -eq 1 ] ; then
    DRY_RUN="true"
else
    DRY_RUN="false"
fi

box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}

main() {
    cmd=""
    function note_failure {
        box "${cmd}"
    }
    trap note_failure EXIT

    cmds=()
    cmds+=("rm -rf ./data/mlp-1-113.pth")
    cmds+=("python gd_stable/main/generate_network.py --depth 1 --width 113")
    cmds+=("test -f ./data/mlp-1-113.pth")
    cmds+=("rm -rf ./data/plot-1-113.pdf")
    cmds+=("python gd_stable/main/train.py --depth 1 --width 113 --samples 1000")
    cmds+=("test -f ./data/plot-1-113.pdf")

    for cmd in "${cmds[@]}"; do
        box "${cmd}"
        if [ "$DRY_RUN" != "true" ] ; then
            $cmd
        fi
    done

    trap '' EXIT
}

main
