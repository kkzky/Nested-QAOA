"""Direct Stage-2 energy-separator experiments using the existing BCST kernels."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'BCST_corrected_loss_20260911'
sys.path.insert(0, str(BASE))
import campaign as c

DEPTHS = [3, 1, 2, 4, 6, 8, 12, 16]
SEED = 2026091100
P3 = 14
RESTARTS = list(c.old.RESTART_SEEDS)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def gm(values):
    if any(v is None for v in values):
        return None
    if any(v == 0 for v in values):
        return 0.0
    return math.exp(math.fsum(math.log(float(v)) for v in values)/len(values))


def ru(p2, separate_c=False):
    phase_cost = 150 if separate_c else 120
    p1 = 5 + 12*(30+75)
    p2_cost = (1+2*p2)*p1 + p2*(phase_cost+900)
    return dict(stage1=p1, stage2_separator=phase_cost, stage2_preparation=p2_cost,
                stage3_depth=P3, per_shot=(1+2*P3)*p2_cost+P3*(15420+900)+15420)


def setup():
    torch.set_num_threads(4)
    config = replace(c.engine.valid_config(), restarts=4, restart_batch=4,
                     initialization_seeds=tuple(RESTARTS), device=os.environ.get('LPQAOA_DEVICE', 'cuda'),
                     dtype='complex64', learning_rate=0.035, activation_checkpointing=False)
    screen = c.engine.StructuralScreen(config)
    saved = c.read(BASE/'inputs/original_manifest.json')
    with torch.no_grad():
        stage1 = c.common.replay_c12(screen, saved['embedded_C12_row'])
    raw_c = np.rint(screen.c.cpu().numpy()*10).astype(np.int64)
    raw_q = np.rint(screen.q.cpu().numpy()*48).astype(np.int64)
    ctx = SimpleNamespace(screen=screen, c12_state=stage1, c=raw_c, q=raw_q)
    baseline = c.read(BASE/'results/refinement/2026091100_lp_p14.json')
    table = baseline['coefficient_table']
    c.objective.validated_coefficient_table = lambda t: np.asarray(t['service_cost'], dtype=np.int64)
    d = c.diagonals(ctx, table, 1024)
    return ctx, d, baseline


def initial_stage2(depth):
    gamma, beta = [], []
    for seed in RESTARTS:
        gamma.append(np.random.default_rng(seed+1_000_003*(2*95101)).uniform(-math.pi, math.pi, 16)[:depth])
        beta.append(np.random.default_rng(seed+1_000_003*(2*95101+1)).uniform(-math.pi, math.pi, 16)[:depth])
    return np.asarray(gamma), np.asarray(beta)


def optimize(screen, reference, phase, loss, arrays, evaluations, progress, label):
    gamma, beta = [torch.tensor(a, dtype=screen.real_dtype, device=screen.device,
                               requires_grad=True) for a in arrays]
    depth = beta.shape[1]
    optimizer = torch.optim.Adam((gamma, beta), lr=0.035)
    best_energy = torch.full((4,), math.inf, dtype=screen.real_dtype, device=screen.device)
    best_gamma, best_beta = torch.zeros_like(gamma), torch.zeros_like(beta)
    best_index = torch.zeros(4, dtype=torch.int64, device=screen.device)
    best_state = reference.detach().clone()
    trace = []
    started = time.time()
    for i in range(1, evaluations+1):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(i < evaluations):
            energies, states = c.kernels.history_circuit(screen, reference, (phase,), loss,
                                                       gamma, beta, depth)
        with torch.no_grad():
            improved = energies.detach() < best_energy
            best_state[improved] = states.detach()[improved]
            best_gamma[improved] = gamma.detach()[improved]
            best_beta[improved] = beta.detach()[improved]
            best_index[improved] = i
            best_energy = torch.minimum(best_energy, energies.detach())
        if i == 1 or i == evaluations or i % 200 == 0:
            row = dict(evaluation=i, energies=energies.detach().cpu().tolist(),
                       best_energies=best_energy.cpu().tolist(), elapsed_sec=time.time()-started)
            trace.append(row)
            write(progress, dict(status='RUNNING', phase=label, evaluations=evaluations,
                                 updated_unix=time.time(), **row))
            print(label, i, row['best_energies'], flush=True)
        if i < evaluations:
            energies.sum().backward()
            optimizer.step()
    return best_state.detach(), dict(evaluations_per_restart=evaluations,
                                    updates_per_restart=evaluations-1,
                                    selected_energy=best_energy.cpu().tolist(),
                                    best_evaluation=best_index.cpu().tolist(),
                                    gamma=best_gamma.detach().cpu().tolist(),
                                    beta=best_beta.detach().cpu().tolist(),
                                    trace=trace, elapsed_sec=time.time()-started)


def measure(screen, d, state, ground_indices):
    with torch.no_grad():
        probability = state.to(torch.complex128).abs().square()
        probability /= probability.sum(1, keepdim=True)
        mean = lambda a: (probability*a.to(torch.float64)[None,:]).sum(1).cpu().tolist()
        return dict(success_probability=probability[:,ground_indices].sum(1).cpu().tolist(),
                    feasible_probability=probability[:,screen.final_mask].sum(1).cpu().tolist(),
                    adjacency_probability=probability[:,screen.c_mask].sum(1).cpu().tolist(),
                    quota_probability=probability[:,screen.q_mask].sum(1).cpu().tolist(),
                    expected_C=mean(d.c_tensor), expected_Q=mean(d.q_tensor),
                    corrected_loss=mean(d.h_tensor))


def summarize(separator, baseline):
    records = [json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    rows = []
    for r in records:
        if r.get('status') != 'COMPLETE':
            continue
        rows.append(dict(separator=r['separator'], stage2_depth=r['stage2_depth'],
                         stage3_depth=P3, success_probability=r['probability_geomean'],
                         per_shot_RU=r['RU']['per_shot'], total_cost=r['total_cost_geomean'],
                         elapsed_sec=r['elapsed_sec']))
    best = {}
    for name in ['C_plus_Q','Q']:
        available = [r for r in rows if r['separator']==name and r['total_cost'] is not None]
        if available:
            best[name] = min(available,key=lambda r:r['total_cost'])
    write(ROOT/f'summary_{separator}.json', dict(completed=len(rows), results=rows,
          best_by_total_cost=best))


def run(separator):
    ctx, d, baseline = setup()
    screen = ctx.screen
    phase = d.c_tensor+d.q_tensor if separator=='C_plus_Q' else d.q_tensor
    stage3_arrays = tuple(a[:,:P3].copy() for a in c.old._initial_angle_arrays(c.old.FULL_LP,64))
    for p2 in DEPTHS:
        result_path = ROOT/'results'/f'{separator}_p2_{p2}_p3_{P3}.json'
        if result_path.exists():
            continue
        started = time.time()
        progress = ROOT/'progress'/f'{separator}.json'
        stage2_state, stage2_record = optimize(screen,ctx.c12_state,phase,phase,
            initial_stage2(p2),800,progress,f'{separator} p2={p2} stage2')
        stage2_record['measurements'] = measure(screen,d,stage2_state,baseline['score']['ground_indices'])
        write(ROOT/'stage2'/f'{separator}_p2_{p2}.json',stage2_record)
        final_state, stage3_record = optimize(screen,stage2_state,d.o_tensor,d.h_tensor,
            stage3_arrays,2400,progress,f'{separator} p2={p2} stage3')
        metrics = measure(screen,d,final_state,baseline['score']['ground_indices'])
        p = metrics['success_probability']
        repetitions = [c.rts99(x) for x in p]
        ledger = ru(p2)
        costs = [None if r is None else r*ledger['per_shot'] for r in repetitions]
        p_mean, cost_mean = gm(p), gm(costs)
        result = dict(status='COMPLETE', seed=SEED, separator=separator,
                      stage1_depth=12,stage2_depth=p2,stage3_depth=P3,
                      stage2_phase='C_raw+Q_raw' if separator=='C_plus_Q' else 'Q_raw',
                      stage2_loss='expectation of stage2 phase Hamiltonian',
                      stage3_phase='O/2048',stage3_loss='(O+1024(C+Q))/2048',
                      restart_seeds=RESTARTS,stage2=stage2_record,stage3=stage3_record,
                      coefficient_table=baseline['coefficient_table'],measurements=metrics,
                      probability_by_restart=p,probability_geomean=p_mean,RTS99_by_restart=repetitions,
                      RU=ledger,total_cost_by_restart=costs,total_cost_geomean=cost_mean,
                      elapsed_sec=time.time()-started)
        if separator=='C_plus_Q':
            unmerged=ru(p2,True)
            result['unmerged_C_plus_Q_sensitivity']=dict(RU=unmerged,
                total_cost_geomean=gm([None if r is None else r*unmerged['per_shot'] for r in repetitions]))
        write(result_path,result)
        summarize(separator,baseline)
        write(progress,dict(status='POINT_COMPLETE',separator=separator,stage2_depth=p2,
                            success_probability=p_mean,total_cost=cost_mean,updated_unix=time.time()))
        print('COMPLETE',separator,p2,'P=',p_mean,'cost=',cost_mean,flush=True)
        del stage2_state,final_state
    summarize(separator,baseline)
    write(ROOT/'progress'/f'{separator}.json',dict(status='COMPLETE',separator=separator,
          stage2_depths=DEPTHS,updated_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--separator',choices=['C_plus_Q','Q'],required=True)
    args=parser.parse_args()
    try:
        run(args.separator)
    except Exception:
        write(ROOT/'progress'/f'{args.separator}.json',dict(status='FAILED',error=traceback.format_exc(),
              updated_unix=time.time()))
        raise
