"""Run a standard-QAOA N=18 cell with the saved coefficient table and budget."""
from pathlib import Path
import argparse
import json
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-seed', type=int, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'code' / 'BCST_corrected_loss_20260911' / 'original_source'))
    import torch
    import targeted_coherent_im3_vanilla_qaoa as numerical
    inputs = json.loads((root / 'data' / 'n18' / 'inputs' / numerical.MANIFEST_NAME).read_text())
    table = numerical.table(inputs, args.table_seed)
    h, objective, penalty, feasible = numerical.runtime_diagonals(table, torch.device(args.device))
    optimized = numerical.optimize(h, args.depth, numerical.OBJECTIVE_EVALUATIONS)
    metrics = numerical._metrics(optimized['state'], objective, penalty, feasible)
    mask = feasible & (objective == objective[feasible].min())
    probabilities = torch.abs(optimized['state'][:, mask]).square().sum(dim=1).cpu().tolist()
    ru = numerical.resource(args.depth)
    repeats = [None if p <= 0 else (1 if p >= 1 else math.ceil(math.log(.01) / math.log1p(-p))) for p in probabilities]
    result = dict(table_seed=args.table_seed, depth=args.depth, resource_RU=ru,
        gamma_by_restart=optimized['gamma'].cpu().tolist(),
        beta_by_restart=optimized['beta'].cpu().tolist(),
        metrics=metrics, probability_by_restart=probabilities, RTS99_by_restart=repeats,
        total_cost_by_restart=[None if n is None else n * ru for n in repeats],
        optimizer_trace=optimized['trace'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
