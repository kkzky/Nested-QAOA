"""User-authorized unattended continuation; the original numerical core is unchanged."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

import numpy as np
import torch
import campaign as c

METHODS = ('lp', 'two_stage', 'xy_separate', 'xy_combined',
           'no_feasibility', 'warm_xy', 'uniform_projector')
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
EXTRA = {'lp': [6, 10, 12, 14], 'two_stage': [24, 40], 'xy_separate': [6, 10, 12, 14]}
SOURCE_FILES = ('unattended_campaign.py', 'verify_unattended.py', 'launch_unattended.py',
                'prepare_unattended_handoff.py', 'audit_loss_contract.py')
INITIAL64 = {m: c.initial(m, 64) for m in METHODS}
ARCHIVED_RESOURCE = c.extension.resource


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


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(body, field):
    return dict(body, **{field: c.digest(body)})


def verify_seal(body, field):
    assert body[field] == c.digest({k: v for k, v in body.items() if k != field})
    return body


def source_hashes():
    return {name: file_sha(c.ROOT/name) for name in SOURCE_FILES}


def plan():
    p = c.protocol()
    a = verify_seal(c.read(c.ROOT/'unattended_plan.json'), 'amendment_sha256')
    assert a['base_protocol_sha256'] == p['protocol_sha256']
    assert a['sources'] == source_hashes(), 'Unattended code changed after freeze'
    assert a['methods'] == list(METHODS)
    assert a['coarse_depths'] == list(DEPTHS)
    assert a['fixed_added_depths'] == EXTRA
    assert a['validation_seeds'] == p['validation_seeds']
    assert a['validation_authorization'] == 'automatic_after_integrity_checks_no_performance_gate'
    return a


def freeze():
    p = c.protocol()
    manuscript = c.ROOT/'inputs'/'manuscript_method_scope.json'
    evidence = c.read(manuscript)
    assert evidence['retained_methods'] == list(METHODS)
    body = dict(schema='bcst-unattended-20260911-v1', base_protocol_sha256=p['protocol_sha256'],
                user_instruction='Finish warm-start p64, cancel both p96 cells, refine, then directly validate ten seeds; no monitoring',
                methods=list(METHODS), coarse_depths=list(DEPTHS), original_test_cells=49,
                excluded_original_cells=[['xy_separate', 96], ['xy_combined', 96]],
                fixed_added_depths=EXTRA, added_test_cells=10, test_seed=p['test_seed'],
                validation_seeds=p['validation_seeds'], validation_cells=70,
                validation_authorization='automatic_after_integrity_checks_no_performance_gate',
                report_cost_advantage_without_using_it_as_gate=True,
                loss_optimizer_initialization_and_RU_changes='No numerical changes; added-depth adapters slice archived p64 initial arrays and evaluate unchanged affine RU formulas', evaluations_per_restart=2400,
                restarts=4, workers=2, fixed_angle_ablation_included=False, fixed_phase_AA_included=False,
                manuscript_scope_sha256=file_sha(manuscript), manuscript_scope=evidence,
                sources=source_hashes())
    a = sealed(body, 'amendment_sha256')
    c.write(c.ROOT/'unattended_plan.json', a, immutable=True)
    jobs = dict(amendment_sha256=a['amendment_sha256'],
                stages=[dict(name='finish_coarse', tasks=[dict(cohort='test', seed=p['test_seed'],
                             method=m, depth=d) for m in METHODS for d in DEPTHS]),
                        dict(name='refinement', tasks=[dict(cohort='refinement', seed=p['test_seed'],
                             method=m, depth=d) for m, ds in EXTRA.items() for d in ds]),
                        dict(name='lock_depths_and_audit', source='49 coarse and 10 refined test results'),
                        dict(name='validation', tasks=[dict(cohort='validation', seed=s, method=m,
                             depth_source='refined_depth_lock.json:'+m) for s in p['validation_seeds'] for m in METHODS]),
                        dict(name='audit_and_package', output='delivery/bcst_corrected_loss_completed.zip')])
    c.write(c.ROOT/'unattended_queue_plan.json', sealed(jobs, 'queue_sha256'), immutable=True)
    return a


def coarse_tasks(p):
    return [('test', p['test_seed'], m, d) for m in METHODS for d in DEPTHS]


def native_ru(method, depth):
    base, step = dict(lp=(28235, 41950), two_stage=(16685, 19270), xy_separate=(15425, 15945),
                     xy_combined=(15425, 15855), no_feasibility=(16685, 18850),
                     warm_xy=(16685, 15855), uniform_projector=(15425, 16780))[method]
    return base+step*depth


def verify_result(r, p):
    verify_seal(r, 'result_sha256')
    assert r['status'] == 'COMPLETE' and r['protocol_sha256'] == p['protocol_sha256']
    assert r['method'] in METHODS and r['depth'] <= 64
    assert r['evaluations_per_restart'] == p['evaluations'] == 2400
    assert r['updates_per_restart'] == 2399 and r['endpoint_evaluations_per_restart'] == 1
    assert r['restart_seeds'] == p['restart_seeds'] and r['phase_names'] == c.PHASES[r['method']]
    assert r['loss'] == p['loss'] and r['checkpoint'] == p['checkpoint']
    assert r['coefficient_table'] == c.objective.coefficient_table_from_pcg64(r['seed'])
    assert r['initial_angles_sha256'] == c.digest([x.tolist() for x in initial(r['method'], r['depth'])])
    assert [row['evaluation'] for row in r['optimizer_trace']] == [1]+list(range(200, 2400, 200))+[2400]
    assert all(1 <= i <= 2400 for i in r['best_evaluation_by_restart'])
    assert r['score']['RU'] == resource(r['method'], r['depth'])
    assert r['score']['RU']['occupation_products'] == native_ru(r['method'], r['depth'])
    probabilities = r['score']['probability_by_restart']
    assert len(probabilities) == 4 and all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
    assert np.allclose(r['selected_metrics']['norm'], 1, atol=4e-5, rtol=0)
    assert all(math.isfinite(x) for x in r['selected_metrics']['corrected_loss'])
    assert r['score']['rts99_by_restart'] == [c.rts99(x) for x in probabilities]
    assert math.isclose(c.geomean(probabilities), r['score']['probability_geomean'], rel_tol=1e-12, abs_tol=1e-15)
    for key, ru in r['score']['RU'].items():
        costs = [None if n is None else n*ru for n in r['score']['rts99_by_restart']]
        assert costs == r['score']['total_cost_by_restart'][key]
        gm = c.geomean(costs)
        if gm is None:
            assert r['score']['total_cost_geomean'][key] is None
        else:
            assert math.isclose(gm, r['score']['total_cost_geomean'][key], rel_tol=1e-12)
    return r


def load_result(task, p, a=None):
    r = verify_result(c.load_result(*task, p), p)
    if task[0] != 'test':
        a = a or plan()
        assert r['execution_amendment_sha256'] == a['amendment_sha256']
        manifest = verify_seal(c.read(c.ROOT/'refinement_manifest.json'), 'manifest_sha256')
        assert manifest['amendment_sha256'] == a['amendment_sha256']
        assert r['refinement_manifest_sha256'] == manifest['manifest_sha256']
        if task[0] == 'validation':
            lock = verify_seal(c.read(c.ROOT/'refined_depth_lock.json'), 'lock_sha256')
            assert lock['amendment_sha256'] == a['amendment_sha256']
            assert r['refined_depth_lock_sha256'] == lock['lock_sha256']
            assert r['depth'] == lock['depths'][r['method']]
    return r


def choose_depths(rows):
    return {m: min((x for x in rows if x['method'] == m),
                  key=lambda x: (math.inf if x['total_cost'] is None else x['total_cost'], x['depth']))['depth']
            for m in METHODS}


def result_rows(tasks, p, a):
    rows = []
    for task in tasks:
        r = load_result(task, p, a)
        rows.append(dict(cohort=task[0], seed=task[1], method=task[2], depth=task[3],
                         total_cost=r['score']['total_cost_geomean']['occupation_products'],
                         success_probability=r['score']['probability_geomean'], result_sha256=r['result_sha256']))
    return rows


def make_refinement_manifest(p, a):
    rows = result_rows(coarse_tasks(p), p, a)
    for m in ('lp', 'xy_separate'):
        ranked = sorted((x for x in rows if x['method'] == m),
                        key=lambda x: (math.inf if x['total_cost'] is None else x['total_cost'], x['depth']))
        lo, hi = sorted(x['depth'] for x in ranked[:2])
        candidates = sorted({math.floor(lo+(hi-lo)*q+.5) for q in (.25, .5, .75)}-set(DEPTHS))
        assert sorted(set(candidates+[6])) == EXTRA[m], 'Unexpected coarse ranking; refuse to silently alter frozen job list'
    body = dict(amendment_sha256=a['amendment_sha256'], protocol_sha256=p['protocol_sha256'],
                test_seed=p['test_seed'], added_depths=EXTRA, original_test_results=rows,
                selection='LP and separate XY interior quartiles plus user-requested p6; two-stage explicitly 24 and 40',
                validation_results_used=False)
    manifest = sealed(body, 'manifest_sha256')
    c.write(c.ROOT/'refinement_manifest.json', manifest, immutable=True)
    coarse = sealed(dict(amendment_sha256=a['amendment_sha256'], protocol_sha256=p['protocol_sha256'],
                         depths=choose_depths(rows), test_results=rows), 'lock_sha256')
    c.write(c.ROOT/'coarse_depth_lock_64.json', coarse, immutable=True)
    return manifest


def run_cell(cohort, seed, method, depth, device='cuda'):
    p, a = c.protocol(), plan()
    assert method in METHODS and cohort in ('refinement', 'validation')
    manifest = verify_seal(c.read(c.ROOT/'refinement_manifest.json'), 'manifest_sha256')
    assert manifest['amendment_sha256'] == a['amendment_sha256']
    lock_sha = None
    if cohort == 'refinement':
        assert seed == p['test_seed'] and method in EXTRA and depth in EXTRA[method]
    else:
        lock = verify_seal(c.read(c.ROOT/'refined_depth_lock.json'), 'lock_sha256')
        review = verify_seal(c.read(c.ROOT/'refined_test_review_decision.json'), 'review_sha256')
        assert seed in p['validation_seeds'] and depth == lock['depths'][method]
        assert lock['amendment_sha256'] == a['amendment_sha256']
        assert review['amendment_sha256'] == a['amendment_sha256'] and review['authorize_validation'] is True
        assert review['lock_sha256'] == lock['lock_sha256'] and review['integrity_checks_passed'] is True
        lock_sha = lock['lock_sha256']
    task = (cohort, seed, method, depth)
    if c.result_path(*task).exists():
        return load_result(task, p, a)
    ctx = c.make_context(device)
    table = c.objective.coefficient_table_from_pcg64(seed)
    d = c.diagonals(ctx, table, p['penalty'])
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    out = optimize(ctx, d, method, depth, p['evaluations'],
                     c.ROOT/'progress'/f'{cohort}_{seed}_{method}_p{depth}.json')
    scored = score(ctx, d, out.state, method, depth)
    ref = c.circuit_spec(ctx, d, method)[0]
    baseline = dict(metrics=c.metrics(ctx, d, ref), score=score(ctx, d, ref, method, 0))
    body = dict(status='COMPLETE', protocol_sha256=p['protocol_sha256'],
                execution_amendment_sha256=a['amendment_sha256'],
                refinement_manifest_sha256=manifest['manifest_sha256'], refined_depth_lock_sha256=lock_sha,
                cohort=cohort, seed=seed, method=method, method_label=c.METHODS[method][0], depth=depth,
                coefficient_table=table, restart_seeds=p['restart_seeds'], phase_names=c.PHASES[method],
                loss=p['loss'], checkpoint=p['checkpoint'], evaluations_per_restart=p['evaluations'],
                updates_per_restart=p['evaluations']-1, endpoint_evaluations_per_restart=1,
                output_replay_checks_not_training_evaluations=1, initial_angles_sha256=out.initial_sha256,
                best_evaluation_by_restart=out.best_evaluation.cpu().tolist(),
                gamma_by_restart=out.gamma.detach().cpu().tolist(), beta_by_restart=out.beta.detach().cpu().tolist(),
                selected_metrics=out.metrics, score=scored, p0=baseline, optimizer_trace=out.trace,
                state_sha256=c.common.state_sha256(out.state), elapsed_sec=out.elapsed_sec,
                frozen_reference_replay_error=ctx.replay_error,
                peak_GPU_memory_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else None,
                environment=dict(torch=torch.__version__, numpy=np.__version__, dtype=p['dtype'],
                                 device=device, GPU=torch.cuda.get_device_name() if device.startswith('cuda') else None))
    result = sealed(body, 'result_sha256')
    verify_result(result, p)
    c.write(c.result_path(*task), result, immutable=True)
    return result


def status(phase, **fields):
    c.write(c.ROOT/'status.json', dict(phase=phase, controller_pid=os.getpid(),
                                     updated_unix=time.time(), **fields))


def task_command(task):
    cohort, seed, method, depth = task
    return [sys.executable, '-u', str(c.ROOT/'unattended_campaign.py'), 'cell',
            '--cohort', cohort, '--seed', str(seed), '--method', method, '--depth', str(depth)]


def run_task(task):
    p, a = c.protocol(), plan()
    if c.result_path(*task).exists():
        load_result(task, p, a)
        return
    dest = c.ROOT/'logs'/f'{task[0]}_{task[1]}_{task[2]}_p{task[3]}.log'
    dest.parent.mkdir(exist_ok=True)
    with dest.open('a', encoding='utf-8') as log:
        outcome = subprocess.run(task_command(task), cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT)
    if outcome.returncode:
        raise RuntimeError(f'{task} exited {outcome.returncode}; see {dest}')
    load_result(task, p, a)


def run_phase(tasks, workers, phase):
    p, a = c.protocol(), plan()
    pending = []
    completed = 0
    for task in tasks:
        if c.result_path(*task).exists():
            load_result(task, p, a)
            completed += 1
        else:
            pending.append(task)
    def update():
        status(phase, total_cells=len(tasks), completed_cells=completed, workers=workers)
    update()
    # Bounded submission: no new queued cells start after a worker failure.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(pending)
        active = {}
        for _ in range(workers):
            task = next(iterator, None)
            if task is not None:
                active[pool.submit(run_task, task)] = task
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                del active[future]
                completed += 1
            update()
            for _ in done:
                task = next(iterator, None)
                if task is not None:
                    active[pool.submit(run_task, task)] = task


def audit_replays(tasks, p, a, filename):
    from audit_loss_contract import enumerate_diagonals
    ctx = c.make_context('cuda', dtype='complex128')
    bits, conflict, quota, risk = enumerate_diagonals()
    assert np.array_equal(ctx.c, conflict) and np.array_equal(ctx.q, quota)
    feasible = (conflict == 0) & (quota == 0)
    rows = []
    for task in tasks:
        r = load_result(task, p, a)
        matrix = np.asarray(r['coefficient_table']['service_cost'], dtype=np.int64)
        objective = 256*risk+np.einsum('nvc,vc->n', bits.astype(np.int64), matrix)
        raw = objective+1024*(conflict+quota)
        mask = feasible & (objective == objective[feasible].min())
        assert np.array_equal(raw == raw.min(), mask)
        assert np.flatnonzero(mask).tolist() == r['score']['ground_indices']
        d = c.diagonals(ctx, r['coefficient_table'])
        assert np.array_equal(d.o, objective)
        g = torch.tensor(r['gamma_by_restart'], dtype=torch.float64, device='cuda')
        b = torch.tensor(r['beta_by_restart'], dtype=torch.float64, device='cuda')
        with torch.no_grad():
            loss, state = c.energy_fn(ctx, d, task[2], task[3])(g, b)
            probs = state.abs().square()
            probs /= probs.sum(dim=1, keepdim=True)
            success = probs[:, torch.as_tensor(mask, device='cuda')].sum(1).cpu().numpy()
            summed = (probs*torch.as_tensor(raw/2048., device='cuda')).sum(1).cpu().numpy()
        assert np.allclose(loss.cpu().numpy(), summed, atol=1e-10, rtol=1e-10)
        assert np.allclose(success, r['score']['probability_by_restart'], atol=1e-10, rtol=3e-3)
        assert np.allclose(summed, r['selected_metrics']['corrected_loss'], atol=2e-5, rtol=2e-5)
        rows.append(dict(task=list(task), result_sha256=r['result_sha256'],
                         max_success_abs_error=float(np.max(np.abs(success-r['score']['probability_by_restart'])))))
        print('REPLAY_PASS '+str(task), flush=True)
    report = sealed(dict(status='PASS', amendment_sha256=a['amendment_sha256'],
                         checked_cells=len(rows), rows=rows), 'audit_sha256')
    c.write(c.ROOT/'audit'/filename, report, immutable=True)
    return report


def export(tasks, p, a, label):
    rows, starts = result_rows(tasks, p, a), []
    for task in tasks:
        r = load_result(task, p, a)
        for i, restart in enumerate(p['restart_seeds']):
            starts.append(dict(cohort=task[0], instance_seed=task[1], method=task[2], depth=task[3],
                               restart_seed=restart, probability=r['score']['probability_by_restart'][i],
                               RU=r['score']['RU']['occupation_products'], RTS99=r['score']['rts99_by_restart'][i],
                               total_cost=r['score']['total_cost_by_restart']['occupation_products'][i],
                               corrected_loss=r['selected_metrics']['corrected_loss'][i],
                               P_F=r['selected_metrics']['P_F'][i]))
    root = c.ROOT/'analysis'
    root.mkdir(exist_ok=True)
    c.write(root/f'{label}_per_instance.json', dict(amendment_sha256=a['amendment_sha256'], rows=rows))
    with (root/f'{label}_per_start.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(starts[0]))
        writer.writeheader()
        writer.writerows(starts)
    return rows


def package_results(p, a, validation_rows, lock):
    aggregate = {}
    for m in METHODS:
        rows = [r for r in validation_rows if r['method'] == m]
        assert {r['seed'] for r in rows} == set(p['validation_seeds']) and len(rows) == 10
        aggregate[m] = dict(depth=lock['depths'][m], instances=10,
                            probability_geomean=c.geomean([r['success_probability'] for r in rows]),
                            total_cost_geomean=c.geomean([r['total_cost'] for r in rows]))
    c.write(c.ROOT/'analysis'/'validation_summary.json', dict(amendment_sha256=a['amendment_sha256'], methods=aggregate))
    directory = c.ROOT/'delivery'
    directory.mkdir(exist_ok=True)
    destination = directory/'bcst_corrected_loss_completed.zip'
    included = []
    for folder in ('results', 'analysis', 'audit', 'inputs', 'original_source', 'logs'):
        included.extend(x for x in (c.ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    included.extend(x for x in c.ROOT.iterdir() if x.is_file() and x.suffix in ('.py', '.json', '.md'))
    # Credentials are outside this isolated experiment directory and are never collected.
    assert all(c.ROOT in x.parents for x in included)
    hashes = {x.relative_to(c.ROOT).as_posix(): file_sha(x) for x in sorted(set(included))}
    c.write(directory/'package_manifest.json', dict(amendment_sha256=a['amendment_sha256'], files=hashes))
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(c.ROOT/name, arcname=name)
        z.write(directory/'package_manifest.json', arcname='package_manifest.json')
        z.writestr('COMPLETED.json', json.dumps(dict(status='COMPLETE', original_test_cells=49,
                   refinement_cells=10, validation_cells=70, final_double_replay_audit='PASS',
                   amendment_sha256=a['amendment_sha256']), indent=2))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
    c.write(directory/'COMPLETED.json', dict(status='COMPLETE', original_test_cells=49,
            refinement_cells=10, validation_cells=70, package=destination.name,
            package_sha256=file_sha(destination), amendment_sha256=a['amendment_sha256'], time_unix=time.time()))
    return str(destination)


def main_controller():
    import fcntl
    from prepare_unattended_handoff import alive
    guard = (c.ROOT/'controller.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p, a = c.protocol(), plan()
    checks = c.read(c.ROOT/'audit'/'unattended_verification.json')
    assert checks['status'] == 'PASS' and checks['sources'] == a['sources']
    assert c.read(c.ROOT/'audit'/'campaign_verification.json')['allowed_workers'] >= a['workers']
    try:
        handoff = c.read(c.ROOT/'unattended_handoff.json')
        assert all(x['method'] == 'warm_xy' and x['depth'] == 64 for x in handoff['preserved'])
        status('WAITING_FOR_WARM_START_P64', workers=2, refinement_cells_queued=10,
               validation_cells_queued=70, automatic_validation=True)
        while any(alive(x) for x in handoff['preserved']):
            time.sleep(10)
        # A missing or failed warm-start result halts here; no fabricated replacement.
        load_result(('test', p['test_seed'], 'warm_xy', 64), p, a)
        manifest = make_refinement_manifest(p, a)
        refinement = [('refinement', p['test_seed'], m, d) for m, ds in EXTRA.items() for d in ds]
        run_phase(refinement, a['workers'], 'depth_refinement')
        test = coarse_tasks(p)+refinement
        rows = export(test, p, a, 'combined_test')
        lock = sealed(dict(protocol_sha256=p['protocol_sha256'], amendment_sha256=a['amendment_sha256'],
                          refinement_manifest_sha256=manifest['manifest_sha256'], depths=choose_depths(rows),
                          selection=p['depth_selection'], test_results=rows), 'lock_sha256')
        c.write(c.ROOT/'refined_depth_lock.json', lock, immutable=True)
        selected = [next((r['cohort'], p['test_seed'], m, r['depth']) for r in rows
                         if r['method'] == m and r['depth'] == lock['depths'][m]) for m in METHODS]
        status('AUDITING_SELECTED_TEST_OUTPUTS', refinement_complete=True)
        audit = audit_replays(selected, p, a, 'selected_test_double_replay.json')
        selected_rows = result_rows(selected, p, a)
        comparison = {x['method']: x for x in selected_rows}
        lp_cost = comparison['lp']['total_cost']
        positive = lp_cost is not None and all(comparison[m]['total_cost'] is None or
                    lp_cost < comparison[m]['total_cost'] for m in ('xy_separate', 'xy_combined'))
        review = sealed(dict(amendment_sha256=a['amendment_sha256'], protocol_sha256=p['protocol_sha256'],
                      lock_sha256=lock['lock_sha256'], audit_sha256=audit['audit_sha256'],
                      integrity_checks_passed=True, positive_cost_only=positive,
                      authorization='User requested automatic validation after refinement; no performance gate',
                      authorize_validation=True, selected_results=selected_rows), 'review_sha256')
        c.write(c.ROOT/'refined_test_review_decision.json', review, immutable=True)
        validation = [('validation', seed, m, lock['depths'][m]) for seed in p['validation_seeds'] for m in METHODS]
        c.write(c.ROOT/'validation_queue.json', sealed(dict(amendment_sha256=a['amendment_sha256'],
                     lock_sha256=lock['lock_sha256'], tasks=[list(t) for t in validation]), 'queue_sha256'), immutable=True)
        run_phase(validation, a['workers'], 'fixed_depth_validation')
        validation_rows = export(validation, p, a, 'validation')
        status('FINAL_AUDIT', validation_cells=70)
        audit_replays(refinement+validation, p, a, 'refinement_and_validation_double_replay.json')
        status('PACKAGING', original_test_cells=49, refinement_cells=10, validation_cells=70)
        package = package_results(p, a, validation_rows, lock)
        status('COMPLETE', original_test_cells=49, refinement_cells=10, validation_cells=70, package=package)
        print('CAMPAIGN_COMPLETE '+package, flush=True)
    except Exception as exc:
        status('FAILED', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('controller')
    cell = sub.add_parser('cell')
    cell.add_argument('--cohort', choices=('refinement', 'validation'), required=True)
    cell.add_argument('--seed', type=int, required=True)
    cell.add_argument('--method', choices=METHODS, required=True)
    cell.add_argument('--depth', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze()['amendment_sha256'])
    elif args.command == 'controller':
        main_controller()
    else:
        run_cell(args.cohort, args.seed, args.method, args.depth)


if __name__ == '__main__':
    main()
