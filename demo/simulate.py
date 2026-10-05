"""Small CPU-only BCST simulation with exact mixers and adjoint gradients."""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Problem:
    specification: dict
    configurations: np.ndarray
    occupations: np.ndarray
    conflict: np.ndarray
    quota: np.ndarray
    assignment: np.ndarray
    intermodulation: np.ndarray
    objective: np.ndarray
    hamiltonian: np.ndarray
    feasible: np.ndarray
    optimum: np.ndarray
    xy_generator: np.ndarray
    xy_eigenvalues: np.ndarray
    xy_eigenvectors: np.ndarray

    @property
    def dimension(self):
        return len(self.configurations)

    def initial_state(self):
        return np.ones(self.dimension, dtype=np.complex128) / math.sqrt(self.dimension)


def build_problem(specification):
    channels = int(specification['channels'])
    cardinalities = specification['cardinalities']
    local_masks = [[sum(1 << c for c in combination)
                    for combination in itertools.combinations(range(channels), k)]
                   for k in cardinalities]
    configurations = np.asarray(list(itertools.product(*local_masks)), dtype=np.int64)
    occupations = ((configurations[:, :, None] >> np.arange(channels)) & 1)
    conflict = np.zeros(len(configurations), dtype=np.int64)
    for u, v in specification['adjacency_edges']:
        conflict += np.sum(occupations[:, u] * occupations[:, v], axis=1)
    quota = np.sum((occupations.sum(axis=1) - np.asarray(specification['channel_quotas'])) ** 2, axis=1)
    pairs = [[(a, b) for a, b in itertools.combinations(range(channels), 2)
              if 2 * a - b == c or 2 * b - a == c] for c in range(channels)]
    interference = np.zeros_like(occupations)
    for v in range(len(cardinalities)):
        for c, source_pairs in enumerate(pairs):
            for u in range(len(cardinalities)):
                if u != v:
                    for a, b in source_pairs:
                        interference[:, v, c] += occupations[:, u, a] * occupations[:, u, b]
    intermodulation = np.sum(occupations * interference ** 2, axis=(1, 2))
    assignment = np.sum(occupations * np.asarray(specification['assignment_costs']), axis=(1, 2))
    objective = assignment + specification['intermodulation_weight'] * intermodulation
    feasible = (conflict == 0) & (quota == 0)
    if not np.any(feasible):
        raise ValueError('The instance has no feasible configuration.')
    optimum = feasible & (objective == objective[feasible].min())
    hamiltonian = (objective + specification['constraint_weight'] * (conflict + quota)) / specification['phase_scale']
    # Restricted matrix of sum_{v,a<b} (X_va X_vb + Y_va Y_vb)/2.
    # Each allowed within-block exchange has matrix element one.
    generator = np.zeros((len(configurations), len(configurations)), dtype=np.float64)
    for i, a in enumerate(configurations):
        for j in range(i):
            b = configurations[j]
            changed = np.flatnonzero(a != b)
            if len(changed) == 1 and int(a[changed[0]] ^ b[changed[0]]).bit_count() == 2:
                generator[i, j] = generator[j, i] = 1
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    return Problem(specification, configurations, occupations, conflict, quota,
                   assignment, intermodulation, objective, hamiltonian, feasible,
                   optimum, generator, eigenvalues, eigenvectors)


def mixer(problem, state, beta, reference=None):
    if reference is None:
        return problem.xy_eigenvectors @ (
            np.exp(-1j * beta * problem.xy_eigenvalues) *
            (problem.xy_eigenvectors.T @ state))
    # Exactly exp[-i beta (I - |reference><reference|)], including its global phase.
    phase = np.exp(-1j * beta)
    return phase * state + (1 - phase) * reference * np.vdot(reference, state)


def generator_action(problem, state, reference):
    if reference is None:
        return problem.xy_generator @ state
    return state - reference * np.vdot(reference, state)


def circuit(problem, angles, initial, separator, reference=None):
    state = initial.copy()
    states = [state]
    for gamma, beta in np.asarray(angles).reshape(-1, 2):
        state = np.exp(-1j * gamma * separator) * state
        states.append(state)
        state = mixer(problem, state, beta, reference)
        states.append(state)
    return state, states


def loss_and_gradient(angles, problem, initial, separator, loss, reference=None):
    state, states = circuit(problem, angles, initial, separator, reference)
    adjoint = loss * state
    value = float(np.vdot(state, adjoint).real)
    gradient = np.empty_like(angles)
    for layer in reversed(range(len(angles) // 2)):
        gamma, beta = angles[2 * layer:2 * layer + 2]
        gradient[2 * layer + 1] = 2 * np.vdot(
            adjoint, -1j * generator_action(problem, states[2 * layer + 2], reference)).real
        adjoint = mixer(problem, adjoint, -beta, reference)
        gradient[2 * layer] = 2 * np.vdot(
            adjoint, -1j * separator * states[2 * layer + 1]).real
        adjoint = np.exp(1j * gamma * separator) * adjoint
    return value, gradient


def metrics(problem, state):
    probability = np.abs(state) ** 2
    return dict(norm=float(probability.sum()),
                success_probability=float(probability[problem.optimum].sum()),
                feasible_probability=float(probability[problem.feasible].sum()),
                conflict_free_probability=float(probability[problem.conflict == 0].sum()),
                quota_satisfied_probability=float(probability[problem.quota == 0].sum()),
                expected_conflict=float(probability @ problem.conflict),
                expected_quota=float(probability @ problem.quota),
                expected_objective=float(probability @ problem.objective),
                expected_hamiltonian=float(probability @ problem.hamiltonian))


def optimize(problem, depth, initial, separator, loss, reference, seed, progress):
    settings = problem.specification
    rng = np.random.default_rng(seed)
    candidates = []
    selected = None
    for restart in range(settings['restarts']):
        start = rng.uniform(-math.pi, math.pi, 2 * depth)
        result = minimize(loss_and_gradient, start,
                          args=(problem, initial, separator, loss, reference),
                          method='L-BFGS-B', jac=True,
                          options=dict(maxiter=settings['max_iterations'], ftol=1e-12,
                                       gtol=1e-8, maxls=30))
        candidate = dict(restart=restart, angles=result.x.reshape(-1, 2).tolist(),
                         loss=float(result.fun), iterations=int(result.nit),
                         function_evaluations=int(result.nfev),
                         converged=bool(result.success), stopping_message=str(result.message))
        candidates.append(candidate)
        if selected is None or candidate['loss'] < selected['loss']:
            selected = candidate
        progress(f"  restart {restart + 1}/{settings['restarts']}: loss={result.fun:.6g}")
    state, _ = circuit(problem, np.asarray(selected['angles']).ravel(), initial, separator, reference)
    return state, dict(depth=depth, selected_restart=selected['restart'],
                       loss=selected['loss'], angles=selected['angles'],
                       candidates=candidates, metrics=metrics(problem, state))


def run_demo(problem, progress=print):
    started = time.perf_counter()
    settings = problem.specification
    initial = problem.initial_state()
    state = initial
    lp_stages = []
    phase_scale = settings['phase_scale']
    full_loss_name = f"(O + {settings['constraint_weight']}(C + Q))/{phase_scale}"
    stages = [
        ('Adjacency constraint', problem.conflict / 2, problem.conflict, 'C/2', 'C', None),
        ('Global channel quotas', problem.quota, problem.quota, 'Q', 'Q', 'learned'),
        ('Final objective', problem.objective / phase_scale, problem.hamiltonian,
         f'O/{phase_scale}', full_loss_name, 'learned')]
    for j, ((name, separator, loss, phase_name, loss_name, ref_kind), depth) in enumerate(zip(stages, settings['lp_depths'])):
        progress(f'LP-QAOA stage {j + 1}: {name}, depth {depth}')
        reference = None if ref_kind is None else state.copy()
        state, result = optimize(problem, depth, state, separator, loss,
                                 reference, settings['optimizer_seed'] + j, progress)
        result.update(name=name, separator=phase_name, loss_operator=loss_name,
                      mixer='XY' if reference is None else 'learned projector')
        lp_stages.append(result)
        stage_metrics = result['metrics']
        progress(f"  feasible probability={stage_metrics['feasible_probability']:.2%}; "
                 f"optimum probability={stage_metrics['success_probability']:.2%}")
    lp_state = state
    progress(f"Block-XY QAOA: depth {settings['xy_depth']}")
    xy_state, xy_result = optimize(
        problem, settings['xy_depth'], initial, problem.hamiltonian, problem.hamiltonian,
        None, settings['optimizer_seed'] + 100, progress)
    return dict(
        instance=settings, dimension=problem.dimension,
        initial_metrics=metrics(problem, initial),
        lp=dict(stages=lp_stages, metrics=metrics(problem, lp_state)),
        xy=dict(optimization=xy_result, metrics=metrics(problem, xy_state)),
        elapsed_seconds=time.perf_counter() - started)
