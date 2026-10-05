"""Full-depth BCST jobs using the existing numerical kernels and Adam loop."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'BCST_corrected_loss_20260911'
sys.path.insert(0, str(BASE))
import campaign as c

engine, old, common, stage2 = c.engine, c.old, c.common, c.stage2
energy_fn = c.energy_fn
raw_constraints = c.raw_constraints
read = c.read

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def initial(method, depth):
    gamma, beta = c.initial(method, 64)
    phases = gamma.shape[1] // 64
    return (gamma.reshape(4, phases, 64)[:, :, :depth].reshape(4, phases*depth).copy(),
            beta[:, :depth].copy())

def resource(method, depth):
    return c.resource(method, depth)

def diagonals(ctx, table):
    # Reuse the objective arithmetic without coefficient-table seals.
    c.objective.validated_coefficient_table = lambda t: np.asarray(t['service_cost'], dtype=np.int64)
    return c.diagonals(ctx, table, 1024)

def make_context(device='cuda', dtype='complex64'):
    torch.set_num_threads(4)
    config = replace(engine.valid_config(), restarts=4, restart_batch=4,
                     initialization_seeds=old.RESTART_SEEDS, device=device, dtype=dtype,
                     learning_rate=0.035, activation_checkpointing=False)
    screen = engine.StructuralScreen(config)
    saved = read(BASE / 'inputs' / 'original_manifest.json')
    with torch.no_grad():
        c12 = common.replay_c12(screen, saved['embedded_C12_row'])
        f = saved['embedded_service_independent_VF3']
        fg = torch.tensor(f['gamma_by_restart'], dtype=screen.real_dtype, device=screen.device)
        fb = torch.tensor(f['beta_by_restart'], dtype=screen.real_dtype, device=screen.device)
        floss, vf3 = stage2._feasibility_energy(screen, c12)(fg, fb)
    c, q = raw_constraints(screen.masks_np)
    return SimpleNamespace(screen=screen, c12_state=c12, vf3_state=vf3,
                           native_state=screen.native[None, :].expand(4, -1).clone(), c=c, q=q,
                           replay_error=None)

def metrics(ctx, d, state):
    with torch.no_grad():
        prob = state.detach().to(torch.complex128).abs().square()
        norm = prob.sum(dim=1)
        prob = prob/norm[:, None]
        expect = lambda x: (prob*x.to(torch.float64)).sum(dim=1).cpu().tolist()
        return dict(norm=norm.cpu().tolist(), corrected_loss=expect(d.h_tensor),
                    old_loss=expect(d.old_tensor), raw_C=expect(d.c_tensor), raw_Q=expect(d.q_tensor),
                    raw_O=(np.asarray(expect(d.o_tensor))*2048).tolist(),
                    P_C=prob[:, ctx.screen.c_mask].sum(dim=1).cpu().tolist(),
                    P_Q=prob[:, ctx.screen.q_mask].sum(dim=1).cpu().tolist(),
                    P_F=prob[:, ctx.screen.final_mask].sum(dim=1).cpu().tolist())

def optimize(ctx, d, method, depth, evaluations, progress_path=None):
    """Original Adam updates; no separate output replay or acceptance procedure."""
    energy = energy_fn(ctx, d, method, depth)
    arrays = initial(method, depth)
    gamma, beta = [torch.tensor(a, device=ctx.screen.device, dtype=ctx.screen.real_dtype,
                               requires_grad=True) for a in arrays]
    opt = torch.optim.Adam((gamma, beta), lr=0.035)
    best_e = torch.full((4,), math.inf, device=gamma.device, dtype=gamma.dtype)
    best_g, best_b = torch.zeros_like(gamma), torch.zeros_like(beta)
    best_i = torch.zeros(4, dtype=torch.int64, device=gamma.device)
    best_s = None
    trace = []
    started = time.time()
    for i in range(1, evaluations+1):
        endpoint = i == evaluations
        opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(not endpoint):
            values, states = energy(gamma, beta)
        assert torch.isfinite(values).all(), f'Nonfinite loss at {i}'
        with torch.no_grad():
            improved = values.detach() < best_e
            if best_s is None:
                best_s = states.detach().clone()
            elif torch.any(improved):
                best_s[improved] = states.detach()[improved]
            best_e = torch.where(improved, values.detach(), best_e)
            best_g[improved] = gamma.detach()[improved]
            best_b[improved] = beta.detach()[improved]
            best_i[improved] = i
        record = i == 1 or endpoint or i % 200 == 0
        row = None
        if record:
            row = dict(evaluation=i, endpoint=endpoint, loss_by_restart=values.detach().cpu().tolist(),
                       components=metrics(ctx, d, states),
                       gamma_by_restart=gamma.detach().cpu().tolist(),
                       beta_by_restart=beta.detach().cpu().tolist(), elapsed_sec=time.time()-started)
        if not endpoint:
            values.sum().backward()
            assert torch.isfinite(gamma.grad).all() and torch.isfinite(beta.grad).all(), f'Nonfinite gradient at {i}'
            if record:
                row['gradient_l2_by_restart'] = torch.sqrt(
                    gamma.grad.square().sum(1)+beta.grad.square().sum(1)).detach().cpu().tolist()
            opt.step()
        if record:
            trace.append(row)
            if progress_path:
                write(progress_path, dict(method=method, depth=depth, evaluation=i,
                                          evaluations=evaluations, elapsed_sec=time.time()-started,
                                          updated_unix=time.time(), process_id=os.getpid()))
            print(f'{method} p={depth} eval={i}/{evaluations} loss={row["loss_by_restart"]}', flush=True)
    selected = metrics(ctx, d, best_s)
    return SimpleNamespace(state=best_s, metrics=selected, gamma=best_g, beta=best_b,
                           best_evaluation=best_i, trace=trace, elapsed_sec=time.time()-started)

def score(ctx, d, state, method, depth):
    feasible = (ctx.c == 0) & (ctx.q == 0)
    ground_energy = int(d.o[feasible].min())
    mask = feasible & (d.o == ground_energy)
    with torch.no_grad():
        prob = state.detach().to(torch.complex128).abs().square()
        prob /= prob.sum(dim=1, keepdim=True)
        p = prob[:, torch.as_tensor(mask, device=state.device)].sum(dim=1).cpu().tolist()
    ru = resource(method, depth)
    repetitions = [c.rts99(x) for x in p]
    costs = {k: [None if n is None else n*v for n in repetitions] for k, v in ru.items()}
    return dict(spectral_contract=dict(ground_energy=ground_energy, degeneracy=int(mask.sum())),
                ground_indices=np.flatnonzero(mask).tolist(),
                probability_by_restart=p, probability_geomean=c.geomean(p), rts99_by_restart=repetitions,
                RU=ru, total_cost_by_restart=costs,
                total_cost_geomean={k: c.geomean(v) for k, v in costs.items()},
                score_computed_after_checkpoint_selection=True)

def run_cell(seed, method, depth):
    dest = ROOT/'results'/'validation'/f'{seed}_{method}_p{depth}.json'
    if dest.exists():
        print(f'ALREADY_COMPLETE {seed} {method} p={depth}', flush=True)
        return
    ctx = make_context()
    matrix = np.random.Generator(np.random.PCG64(seed)).integers(1, 21, (5, 6), dtype=np.int64)
    table = dict(service_cost=matrix.tolist(), service_cost_seed=seed,
                 service_cost_generator='numpy.random.PCG64', coherent_risk_weight=256, phase_scale=2048)
    d = diagonals(ctx, table)
    torch.cuda.reset_peak_memory_stats()
    progress = ROOT/'progress'/f'{seed}_{method}_p{depth}.json'
    out = optimize(ctx, d, method, depth, 2400, progress)
    scored = score(ctx, d, out.state, method, depth)
    ref = c.circuit_spec(ctx, d, method)[0]
    baseline = dict(metrics=metrics(ctx, d, ref), score=score(ctx, d, ref, method, 0))
    body = dict(status='COMPLETE', cohort='validation', seed=seed, method=method,
                method_label=c.METHODS[method][0], depth=depth, coefficient_table=table,
                restart_seeds=list(old.RESTART_SEEDS), phase_names=c.PHASES[method],
                loss='(O_raw+1024*(C_raw+Q_raw))/2048', checkpoint='minimum corrected loss per start',
                evaluations_per_restart=2400, updates_per_restart=2399, endpoint_evaluations_per_restart=1,
                best_evaluation_by_restart=out.best_evaluation.cpu().tolist(),
                gamma_by_restart=out.gamma.detach().cpu().tolist(), beta_by_restart=out.beta.detach().cpu().tolist(),
                selected_metrics=out.metrics, score=scored, p0=baseline, optimizer_trace=out.trace,
                elapsed_sec=out.elapsed_sec, peak_GPU_memory_bytes=int(torch.cuda.max_memory_allocated()),
                environment=dict(torch=torch.__version__, numpy=np.__version__, dtype='complex64',
                                 device='cuda', GPU=torch.cuda.get_device_name()))
    write(dest, body)
    print(f'COMPLETE {seed} {method} p={depth}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--method', choices=['xy_separate','xy_combined','no_feasibility','warm_xy','uniform_projector'], required=True)
    parser.add_argument('--depth', type=int, required=True)
    args = parser.parse_args()
    run_cell(args.seed, args.method, args.depth)
