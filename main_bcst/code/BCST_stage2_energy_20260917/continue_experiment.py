"""Continue requested depths, then compare one frozen reference using full-H phases."""
from pathlib import Path
import argparse
import json
import math
import time
import traceback

import numpy as np
import torch
import run_experiment as r

ROOT = Path(__file__).resolve().parent
NEW_DEPTHS = [16, 14, 18, 20]


def initial_stage2(depth):
    # Extending the same RNG streams preserves every previously used prefix.
    size = max(20, depth)
    gamma, beta = [], []
    for seed in r.RESTARTS:
        gamma.append(np.random.default_rng(seed + 1_000_003*(2*95101)).uniform(-math.pi, math.pi, size)[:depth])
        beta.append(np.random.default_rng(seed + 1_000_003*(2*95101+1)).uniform(-math.pi, math.pi, size)[:depth])
    return np.asarray(gamma), np.asarray(beta)


def load_reference(ctx, d, separator, depth, progress, require_saved=False):
    path = ROOT/'stage2'/f'{separator}_p2_{depth}.json'
    if not path.exists():
        path = ROOT.parents[1] / 'data' / 'inputs' / f'{separator}_p2_{depth}.json'
    phase = d.c_tensor+d.q_tensor if separator == 'C_plus_Q' else d.q_tensor
    if path.exists():
        record = json.loads(path.read_text())
        gamma = torch.tensor(record['gamma'], dtype=ctx.screen.real_dtype, device=ctx.screen.device)
        beta = torch.tensor(record['beta'], dtype=ctx.screen.real_dtype, device=ctx.screen.device)
        with torch.no_grad():
            _, state = r.c.kernels.history_circuit(ctx.screen, ctx.c12_state, (phase,), phase, gamma, beta, depth)
        print('REUSE_STAGE2', separator, depth, flush=True)
        return state.detach(), record, True
    if require_saved:
        raise FileNotFoundError(path)
    state, record = r.optimize(ctx.screen, ctx.c12_state, phase, phase,
        initial_stage2(depth), 800, progress, f'{separator} p2={depth} stage2')
    return state, record, False


def finish_stage3(ctx, d, baseline, separator, depth, state, stage2, progress, full_h=False):
    arrays = tuple(a[:, :r.P3].copy() for a in r.c.old._initial_angle_arrays(r.c.old.FULL_LP, 64))
    phase = d.h_tensor if full_h else d.o_tensor
    phase_name = '(O+1024(C+Q))/2048' if full_h else 'O/2048'
    final_state, stage3 = r.optimize(ctx.screen, state, phase, d.h_tensor, arrays, 2400,
        progress, f'{separator} p2={depth} stage3 {"full_H" if full_h else "O_only"}')
    measurements = r.measure(ctx.screen, d, final_state, baseline['score']['ground_indices'])
    probabilities = measurements['success_probability']
    repeats = [r.c.rts99(p) for p in probabilities]
    ledger = r.ru(depth)
    ledger['stage3_separator'] = 15480 if full_h else 15420
    ledger['terminal_evaluation'] = 15420
    if full_h:
        ledger['per_shot'] += 60*r.P3
    costs = [None if count is None else count*ledger['per_shot'] for count in repeats]
    p_mean, cost_mean = r.gm(probabilities), r.gm(costs)
    result = dict(status='COMPLETE', seed=r.SEED, separator=separator,
        stage1_depth=12, stage2_depth=depth, stage3_depth=r.P3,
        stage2_phase='C_raw+Q_raw' if separator == 'C_plus_Q' else 'Q_raw',
        stage2_loss='expectation of stage2 phase Hamiltonian',
        stage3_phase=phase_name, stage3_loss='(O+1024(C+Q))/2048',
        restart_seeds=r.RESTARTS, stage2=stage2, stage3=stage3,
        coefficient_table=baseline['coefficient_table'], measurements=measurements,
        probability_by_restart=probabilities, probability_geomean=p_mean,
        RTS99_by_restart=repeats, RU=ledger, total_cost_by_restart=costs,
        total_cost_geomean=cost_mean)
    if separator == 'C_plus_Q':
        unmerged = r.ru(depth, True)
        if full_h:
            unmerged['per_shot'] += 60*r.P3
        result['unmerged_C_plus_Q_sensitivity'] = dict(RU=unmerged,
            total_cost_geomean=r.gm([None if n is None else n*unmerged['per_shot'] for n in repeats]))
    return result


def continue_depths(separator):
    ctx, d, baseline = r.setup()
    progress = ROOT/'progress'/f'{separator}.json'
    for depth in NEW_DEPTHS:
        output = ROOT/'results'/f'{separator}_p2_{depth}_p3_{r.P3}.json'
        if output.exists():
            print('SKIP_COMPLETE', separator, depth, flush=True)
            continue
        started = time.time()
        state, record, reused = load_reference(ctx, d, separator, depth, progress)
        if not reused:
            record['measurements'] = r.measure(ctx.screen, d, state, baseline['score']['ground_indices'])
            r.write(ROOT/'stage2'/f'{separator}_p2_{depth}.json', record)
        result = finish_stage3(ctx, d, baseline, separator, depth, state, record, progress)
        result['elapsed_sec'] = time.time()-started
        result['stage2_reused_from_saved_parameters'] = reused
        r.write(output, result)
        r.summarize(separator, baseline)
        r.write(progress, dict(status='POINT_COMPLETE', separator=separator, stage2_depth=depth,
            success_probability=result['probability_geomean'], total_cost=result['total_cost_geomean'],
            updated_unix=time.time()))
        print('COMPLETE', separator, depth, result['probability_geomean'], result['total_cost_geomean'], flush=True)
        del state
    r.write(progress, dict(status='COMPLETE', separator=separator,
        continuation_stage2_depths=NEW_DEPTHS, updated_unix=time.time()))


def full_h_best():
    output = ROOT/'full_H_result.json'
    if output.exists():
        print('SKIP_COMPLETE_FULL_H', flush=True)
        return
    candidates = []
    for path in (ROOT/'results').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('status') == 'COMPLETE' and record.get('stage3_phase') == 'O/2048' and record.get('separator') == 'Q':
            if record.get('total_cost_geomean') is not None:
                candidates.append((record, path))
    chosen, chosen_path = min(candidates, key=lambda item: (
        item[0]['total_cost_geomean'], item[0]['stage2_depth'], item[0]['separator']))
    separator, depth = chosen['separator'], chosen['stage2_depth']
    selection = dict(selection_rule='Minimum geometric-mean RTS99*RU over four restarts, across Q-only Stage-2 configurations and all completed O-phase depths; smaller depth breaks an exact tie.',
        separator=separator, stage2_depth=depth, stage3_depth=r.P3,
        reference_result=chosen_path.name, reference_probability=chosen['probability_geomean'],
        reference_total_cost=chosen['total_cost_geomean'], candidates=len(candidates),
        stage2_parameters_frozen=True, stage3_initialization='Same original angles as the matched O-phase point',
        selected_unix=time.time())
    r.write(ROOT/'full_H_selection.json', selection)
    print('SELECT_FULL_H', json.dumps(selection), flush=True)
    started = time.time()
    ctx, d, baseline = r.setup()
    progress = ROOT/'progress'/'full_H.json'
    state, record, _ = load_reference(ctx, d, separator, depth, progress, require_saved=True)
    result = finish_stage3(ctx, d, baseline, separator, depth, state, record, progress, full_h=True)
    result['selection'] = selection
    result['elapsed_sec'] = time.time()-started
    result['stage2_reused_from_saved_parameters'] = True
    result['comparison_to_matched_O_phase'] = dict(
        probability_ratio=result['probability_geomean']/chosen['probability_geomean'],
        total_cost_ratio=result['total_cost_geomean']/chosen['total_cost_geomean'] if result['total_cost_geomean'] is not None else None,
        RU_difference=result['RU']['per_shot']-chosen['RU']['per_shot'])
    r.write(output, result)
    r.write(progress, dict(status='COMPLETE', separator=separator, stage2_depth=depth,
        success_probability=result['probability_geomean'], total_cost=result['total_cost_geomean'],
        updated_unix=time.time()))
    print('COMPLETE_FULL_H', separator, depth, result['probability_geomean'], result['total_cost_geomean'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--separator', choices=['C_plus_Q','Q'])
    parser.add_argument('--full-h-best', action='store_true')
    args = parser.parse_args()
    try:
        if args.full_h_best:
            full_h_best()
        else:
            continue_depths(args.separator)
    except Exception:
        key = 'full_H' if args.full_h_best else args.separator
        r.write(ROOT/'progress'/f'{key}.json', dict(status='FAILED', error=traceback.format_exc(), updated_unix=time.time()))
        raise
