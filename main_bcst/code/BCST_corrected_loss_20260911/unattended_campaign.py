"""BCST angle-initialization and optimization adapters.

The command-line interface runs one baseline cell through full_depth_worker.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import campaign as c

METHODS = ('xy_separate', 'xy_combined', 'no_feasibility', 'warm_xy', 'uniform_projector')
INITIAL64 = {m: c.initial(m, 64) for m in METHODS}


def initial(method, depth):
    """Slice the archived p64 arrays, independently for each phase and restart."""
    assert method in METHODS and 1 <= depth <= 64
    gamma, beta = INITIAL64[method]
    phases = gamma.shape[1]//64
    return (gamma.reshape(4, phases, 64)[:, :, :depth].reshape(4, phases*depth).copy(),
            beta[:, :depth].copy())


def resource(method, depth):
    """Evaluate the occupation-product resource model at an integer depth."""
    assert method in METHODS and 0 <= depth <= 64
    return c.resource(method, depth)


def optimize(ctx, d, method, depth, evaluations, progress=None):
    # Only the grid-restricted initialization adapter changes; the Adam loop is identical.
    with patch.object(c, 'initial', initial):
        return c.optimize(ctx, d, method, depth, evaluations, progress)


def score(ctx, d, state, method, depth):
    family_to_method = {c.METHODS[m][1]: m for m in METHODS}
    with patch.object(c.extension, 'resource', lambda family, p: resource(family_to_method[family], p)):
        return c.score(ctx, d, state, method, depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--depth', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.depth <= 64:
        parser.error('--depth must be between 1 and 64')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'BCST_full_depth_20260913'))
    from full_depth_worker import run_cell
    run_cell(args.seed, args.method, args.depth)


if __name__ == '__main__':
    main()
