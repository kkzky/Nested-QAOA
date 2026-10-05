"""Run the CPU BCST numerical demonstration and print its results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Tiny linear-algebra operations benefit from one CPU thread.
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=Path, default=root / 'instance.json')
    args = parser.parse_args()
    import numpy as np
    import scipy
    from simulate import build_problem, run_demo
    settings = json.loads(args.instance.read_text(encoding='utf-8'))
    problem = build_problem(settings)
    print(f'Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {scipy.__version__} | CPU', flush=True)
    print(f'BCST: {len(settings["cardinalities"]) * settings["channels"]} qubits, '
          f'{problem.dimension} fixed-cardinality states, {int(problem.feasible.sum())} feasible states.', flush=True)
    print(f'LP stage depths: {settings["lp_depths"]}; block-XY depth: {settings["xy_depth"]}.', flush=True)
    print(f'{settings["restarts"]} starts per optimization; up to {settings["max_iterations"]} iterations per start.', flush=True)
    targets = np.flatnonzero(problem.optimum)
    for index in targets:
        assignments = [' '.join(str(c + 1) for c in range(settings['channels']) if mask >> c & 1)
                       for mask in problem.configurations[index]]
        print('Optimal assignment: ' + '; '.join(f'{site}: [{channels}]' for site, channels in
              zip(settings['site_names'], assignments)) + f'; O = {problem.objective[index]}.', flush=True)
    print('', flush=True)
    results = run_demo(problem, lambda message: print(message, flush=True))
    summary = [{'method': name, **value} for name, value in [
        ('Initial state', results['initial_metrics']), ('3-stage LP-QAOA', results['lp']['metrics']),
        ('Block-XY QAOA', results['xy']['metrics'])]]
    print('\nMethod                 Optimum probability    Feasible probability    Expected full energy')
    for row in summary:
        print(f'{row["method"]:<23}{row["success_probability"]:>17.2%}'
              f'{row["feasible_probability"]:>23.2%}{row["expected_hamiltonian"]:>24.6f}')
    print(f'\nOptimization time: {results["elapsed_seconds"]:.2f} s')


if __name__ == '__main__':
    main()
