"""Corrected BCST experiment, retaining archived kernels and optimizer rules."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
DATA_ROOT = ROOT.parents[1] / 'data'
sys.path.insert(0, str(ROOT / 'original_source'))
import targeted_coherent_im3_depth_sweep as old
import targeted_coherent_im3_seed10_extension as extension
import targeted_coherent_im3_objective as objective
import targeted_coherent_im3_variational_f3 as stage2
import targeted_three_stage_common as common
import stage3_objective_common as kernels
import target_free_structural_screen as engine
from corrected_loss import corrected_diagonal, raw_constraints, spectral_contract
sys.path.insert(0, str(ROOT.parent))
from resource_accounting import ledger as resource_ledger


def resource(method, depth):
    names = {'xy_separate': 'block_xy_separate', 'xy_combined': 'block_xy_combined',
             'no_feasibility': 'omit_feasibility', 'warm_xy': 'warm_start_block_xy',
             'uniform_projector': 'uniform_projector'}
    if method not in names:
        raise ValueError('Use BCST_Q_native2_complete_20260917/worker.py for LP-QAOA.')
    return {'occupation_products': resource_ledger(names[method], depth)['per_shot']}

METHODS = {
    'lp': ('LP-QAOA', old.FULL_LP),
    'two_stage': ('Two-stage LP-QAOA', old.COLLAPSE_SEPARATE),
    'xy_separate': ('Block-XY QAOA (separate phases)', old.DIRECT_SEPARATE),
    'xy_combined': ('Block-XY QAOA (combined phase)', old.DIRECT_COMBINED),
    'no_feasibility': ('LP-QAOA without the joint-feasibility stage', old.SKIP_F3),
    'warm_xy': ('Warm-start block-XY QAOA', old.WARM_COMBINED),
    'uniform_projector': ('Uniform-state projector QAOA', old.NATIVE_PROJECTOR_SEPARATE),
}
PHASES = {
    'lp': ['O/2048'], 'two_stage': ['Pi_F', 'O/2048'],
    'xy_separate': ['Q/48', 'C/10', 'O/2048'],
    'xy_combined': ['H'], 'no_feasibility': ['O/2048'],
    'warm_xy': ['H'], 'uniform_projector': ['Q/48', 'C/10', 'O/2048'],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    if not path.exists():
        candidates = [DATA_ROOT / 'inputs' / path.name,
                      DATA_ROOT / 'runtime_inputs' / path.name,
                      DATA_ROOT / 'runtime_inputs' / ROOT.name / path.name]
        if path.name.endswith('.json'):
            candidates.append(DATA_ROOT / 'raw' / 'baselines' / path.name)
        path = next((p for p in candidates if p.exists()), path)
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != data:
            raise FileExistsError(f'Refusing to replace {path}')
        return
    kernels.atomic_write_json(path, data)


def source_hashes():
    paths = list((ROOT / 'original_source').glob('*.py'))
    paths += [ROOT / p for p in ('campaign.py', 'corrected_loss.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def grid(method):
    return [1, 2, 4, 8, 16, 32, 64] + ([96] if method in ('xy_separate', 'xy_combined') else [])


def freeze():
    protocol = dict(
        schema='bcst-corrected-loss-20260911-v1', penalty=1024,
        loss='(O_raw+1024*(C_raw+Q_raw))/2048', checkpoint='minimum corrected loss per start',
        phase_scale=2048, lp_phase='O_raw/2048', method_phases=PHASES,
        test_seed=2026091100, validation_seeds=list(range(2026091101, 2026091111)),
        seeds_accepted_without_filtering=True,
        matrix_generator='numpy.Generator(PCG64(seed)).integers(1,21,(5,6),dtype=int64)',
        restart_seeds=list(old.RESTART_SEEDS), methods={k: v[0] for k, v in METHODS.items()},
        depth_grids={k: grid(k) for k in METHODS}, evaluations=2400, adam_updates=2399,
        learning_rate=0.035, dtype='complex64', activation_checkpointing=False,
        initialization='Archived independent prefix-consistent arrays; deterministic tails to p=96; warm-start +/-0.001',
        training_and_checkpoint_selection_use_target_probability=False,
        depth_selection='Minimize geometric mean RTS99*RU over four test starts; smaller depth breaks exact ties; p=0 excluded',
        validation_depth_selection=False,
        success='Unconditional probability of the entire constrained ground space; no postselection',
        rts99='ceil(log(0.01)/log1p(-P)); P=0 -> infinity; P=1 -> 1',
        aggregation='Geometric mean over starts, then geometric mean over instances',
        primary_resource='Occupation-product phases, quadratic terms 1 RU, including full replay and terminal O evaluation',
        training_resources='Evaluation budget and wall time recorded separately; not included in RTS99*RU',
        fixed_phase_AA_included=False,
        phase_hierarchy_note='O-only LP phase is retained. Correct loss does not imply full-X2 phase-ground-space nesting.',
        sources=source_hashes(),
    )
    protocol['protocol_sha256'] = digest(protocol)
    write(ROOT / 'protocol.json', protocol, immutable=True)
    return protocol


def protocol():
    p = read(ROOT / 'protocol.json')
    return p


def initial(method, depth):
    family = METHODS[method][1]
    if depth > 64:
        return extension._high_initial_arrays(family, depth)
    return old._initial_angle_arrays(family, depth)


def make_context(device='cuda', dtype='complex64'):
    torch.set_num_threads(4)
    config = replace(engine.valid_config(), restarts=4, restart_batch=4,
                     initialization_seeds=old.RESTART_SEEDS, device=device, dtype=dtype,
                     learning_rate=0.035, activation_checkpointing=False)
    screen = engine.StructuralScreen(config)
    saved = read(ROOT / 'inputs' / 'original_manifest.json')
    with torch.no_grad():
        c12 = common.replay_c12(screen, saved['embedded_C12_row'])
        f = saved['embedded_service_independent_VF3']
        fg = torch.tensor(f['gamma_by_restart'], dtype=screen.real_dtype, device=screen.device)
        fb = torch.tensor(f['beta_by_restart'], dtype=screen.real_dtype, device=screen.device)
        floss, vf3 = stage2._feasibility_energy(screen, c12)(fg, fb)
    expected = torch.tensor(f['selected_feasibility_loss'], dtype=screen.real_dtype, device=screen.device)
    assert torch.allclose(floss, expected, atol=2e-5, rtol=2e-5)
    c, q = raw_constraints(screen.masks_np)
    assert np.array_equal(c, np.rint(screen.c.cpu().numpy()*10).astype(np.int64))
    assert np.array_equal(q, np.rint(screen.q.cpu().numpy()*48).astype(np.int64))
    return SimpleNamespace(screen=screen, c12_state=c12, vf3_state=vf3,
                           native_state=screen.native[None, :].expand(4, -1).clone(), c=c, q=q,
                           replay_error=float((floss-expected).abs().max().cpu()))


def diagonals(ctx, table, penalty=1024):
    o, risk, service = objective.native_shell_components(ctx.screen, table)
    raw, normalized = corrected_diagonal(o, ctx.c, ctx.q, penalty)
    to = lambda a: torch.as_tensor(a, dtype=ctx.screen.real_dtype, device=ctx.screen.device)
    return SimpleNamespace(o=o, risk=risk, service=service, h_raw=raw,
                           o_tensor=to(o/2048.), h_tensor=to(normalized),
                           c_tensor=to(ctx.c), q_tensor=to(ctx.q),
                           old_tensor=ctx.screen.common+to(o/2048.))


def circuit_spec(ctx, d, method):
    s = ctx.screen
    if method == 'lp':
        return ctx.vf3_state, (d.o_tensor,), 'projector'
    if method == 'two_stage':
        return ctx.c12_state, (s.final_mask.to(s.real_dtype), d.o_tensor), 'projector'
    if method == 'no_feasibility':
        return ctx.c12_state, (d.o_tensor,), 'projector'
    if method == 'xy_separate':
        return ctx.native_state, (s.q, s.c, d.o_tensor), 'xy'
    if method == 'xy_combined':
        return ctx.native_state, (d.h_tensor,), 'xy'
    if method == 'warm_xy':
        return ctx.c12_state, (d.h_tensor,), 'xy'
    if method == 'uniform_projector':
        return ctx.native_state, (s.q, s.c, d.o_tensor), 'projector'
    raise ValueError(method)


def energy_fn(ctx, d, method, depth):
    ref, phases, kind = circuit_spec(ctx, d, method)
    def energy(gamma, beta):
        if kind == 'projector':
            return kernels.history_circuit(ctx.screen, ref, phases, d.h_tensor, gamma, beta, depth)
        return ctx.screen.block_energy(ref, phases, d.h_tensor, gamma, beta, depth)
    return energy


def metrics(ctx, d, state):
    with torch.no_grad():
        prob = state.detach().to(torch.complex128).abs().square()
        norm = prob.sum(dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=2e-5, rtol=2e-5)
        prob = prob/norm[:, None]
        expect = lambda x: (prob*x.to(torch.float64)).sum(dim=1).cpu().tolist()
        return dict(norm=norm.cpu().tolist(), corrected_loss=expect(d.h_tensor),
                    old_loss=expect(d.old_tensor), raw_C=expect(d.c_tensor), raw_Q=expect(d.q_tensor),
                    raw_O=(np.asarray(expect(d.o_tensor))*2048).tolist(),
                    P_C=prob[:, ctx.screen.c_mask].sum(dim=1).cpu().tolist(),
                    P_Q=prob[:, ctx.screen.q_mask].sum(dim=1).cpu().tolist(),
                    P_F=prob[:, ctx.screen.final_mask].sum(dim=1).cpu().tolist())


def optimize(ctx, d, method, depth, evaluations, progress_path=None):
    """Original Adam loop, with finite checks and loss-component diagnostics added."""
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
    assert np.allclose(selected['corrected_loss'], best_e.detach().cpu().numpy(), atol=2e-5, rtol=2e-5)
    assert np.all(np.asarray(selected['corrected_loss']) <= np.asarray(trace[0]['loss_by_restart'])+2e-5)
    # Recompute the selected circuit as an independent output/checkpoint check.
    with torch.no_grad():
        replay_e, replay_s = energy(best_g, best_b)
    assert torch.allclose(best_s, replay_s, atol=2e-5, rtol=2e-5)
    assert torch.allclose(best_e, replay_e, atol=2e-5, rtol=2e-5)
    return SimpleNamespace(state=best_s, metrics=selected, gamma=best_g, beta=best_b,
                           best_evaluation=best_i, trace=trace, elapsed_sec=time.time()-started,
                           initial_sha256=digest([a.tolist() for a in arrays]))


def rts99(p):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(p)
    if p == 0:
        return None  # JSON null means infinite cost, never a discarded observation.
    return 1 if p == 1 else int(math.ceil(math.log(0.01)/math.log1p(-p)))


def geomean(values):
    if any(v is None for v in values):
        return None
    if any(v == 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values)/len(values))


def score(ctx, d, state, method, depth):
    """Ground-space construction is outside the training/checkpoint code."""
    spectrum = spectral_contract(d.o, ctx.c, ctx.q, 1024)
    mask = (ctx.c == 0) & (ctx.q == 0) & (d.o == spectrum['ground_energy'])
    with torch.no_grad():
        prob = state.detach().to(torch.complex128).abs().square()
        prob /= prob.sum(dim=1, keepdim=True)
        p = prob[:, torch.as_tensor(mask, device=state.device)].sum(dim=1).cpu().tolist()
    ru = resource(method, depth)
    repetitions = [rts99(x) for x in p]
    costs = {k: [None if n is None else n*v for n in repetitions] for k, v in ru.items()}
    return dict(spectral_contract=spectrum, ground_indices=np.flatnonzero(mask).tolist(),
                ground_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                probability_by_restart=p, probability_geomean=geomean(p), rts99_by_restart=repetitions,
                RU=ru, total_cost_by_restart=costs,
                total_cost_geomean={k: geomean(v) for k, v in costs.items()},
                score_computed_after_checkpoint_selection=True)


def result_path(cohort, seed, method, depth):
    return ROOT/'results'/cohort/f'{seed}_{method}_p{depth}.json'


def load_result(cohort, seed, method, depth, p):
    r = read(result_path(cohort, seed, method, depth))
    assert r['protocol_sha256'] == p['protocol_sha256']
    assert r['result_sha256'] == digest({k: v for k, v in r.items() if k != 'result_sha256'})
    assert (r['cohort'], r['seed'], r['method'], r['depth']) == (cohort, seed, method, depth)
    assert r['evaluations_per_restart'] == p['evaluations']
    return r


def run_cell(cohort, seed, method, depth, device='cuda'):
    p = protocol()
    assert method in METHODS and depth in grid(method)
    if cohort == 'test':
        assert seed == p['test_seed']
    else:
        assert cohort == 'validation' and seed in p['validation_seeds']
        locked = read(ROOT/'depth_lock.json')
        assert locked['protocol_sha256'] == p['protocol_sha256']
        assert depth == locked['depths'][method]
    dest = result_path(cohort, seed, method, depth)
    if dest.exists():
        return load_result(cohort, seed, method, depth, p)
    ctx = make_context(device)
    table = objective.coefficient_table_from_pcg64(seed)
    d = diagonals(ctx, table, p['penalty'])
    torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None
    progress = ROOT/'progress'/f'{cohort}_{seed}_{method}_p{depth}.json'
    out = optimize(ctx, d, method, depth, p['evaluations'], progress)
    # Success is first computed here, after the loss-selected state is finalized.
    scored = score(ctx, d, out.state, method, depth)
    ref = circuit_spec(ctx, d, method)[0]
    baseline = dict(metrics=metrics(ctx, d, ref), score=score(ctx, d, ref, method, 0))
    body = dict(status='COMPLETE', protocol_sha256=p['protocol_sha256'], cohort=cohort,
                seed=seed, method=method, method_label=METHODS[method][0], depth=depth,
                coefficient_table=table, restart_seeds=p['restart_seeds'], phase_names=PHASES[method],
                loss=p['loss'], checkpoint=p['checkpoint'], evaluations_per_restart=p['evaluations'],
                updates_per_restart=p['evaluations']-1, endpoint_evaluations_per_restart=1,
                output_replay_checks_not_training_evaluations=1,
                initial_angles_sha256=out.initial_sha256,
                best_evaluation_by_restart=out.best_evaluation.cpu().tolist(),
                gamma_by_restart=out.gamma.detach().cpu().tolist(), beta_by_restart=out.beta.detach().cpu().tolist(),
                selected_metrics=out.metrics, score=scored, p0=baseline, optimizer_trace=out.trace,
                state_sha256=common.state_sha256(out.state), elapsed_sec=out.elapsed_sec,
                frozen_reference_replay_error=ctx.replay_error,
                peak_GPU_memory_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else None,
                environment=dict(torch=torch.__version__, numpy=np.__version__, dtype=p['dtype'],
                                 device=device, GPU=torch.cuda.get_device_name() if device.startswith('cuda') else None))
    body['result_sha256'] = digest(body)
    write(dest, body, immutable=True)
    print(f'COMPLETE {cohort} {seed} {method} p={depth}', flush=True)
    return body


def lock_depths():
    p = protocol()
    rows, depths = [], {}
    for method in METHODS:
        choices = []
        for depth in grid(method):
            r = load_result('test', p['test_seed'], method, depth, p)
            cost = r['score']['total_cost_geomean']['occupation_products']
            choices.append((math.inf if cost is None else cost, depth))
            rows.append(dict(method=method, depth=depth, result_sha256=r['result_sha256'], total_cost=cost))
        depths[method] = min(choices)[1]
    body = dict(protocol_sha256=p['protocol_sha256'], test_seed=p['test_seed'], depths=depths,
                selection=p['depth_selection'], test_results=rows)
    body['lock_sha256'] = digest(body)
    write(ROOT/'depth_lock.json', body, immutable=True)
    return body


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('lock-depths')
    c = sub.add_parser('cell')
    c.add_argument('--cohort', choices=('test', 'validation'), required=True)
    c.add_argument('--seed', type=int, required=True)
    c.add_argument('--method', choices=METHODS, required=True)
    c.add_argument('--depth', type=int, required=True)
    c.add_argument('--device', default='cuda')
    a = parser.parse_args()
    if a.command == 'freeze':
        print(freeze()['protocol_sha256'])
    elif a.command == 'lock-depths':
        print(json.dumps(lock_depths()['depths']))
    else:
        run_cell(a.cohort, a.seed, a.method, a.depth, a.device)


if __name__ == '__main__':
    main()
