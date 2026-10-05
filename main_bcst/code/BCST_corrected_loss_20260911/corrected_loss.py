"""Shared raw-diagonal loss definition for the corrected BCST experiment."""
from __future__ import annotations

import numpy as np

PHASE_SCALE = 2048.0


def raw_constraints(masks):
    masks = np.asarray(masks, dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != 5:
        raise ValueError('Expected the unchanged five-block basis.')
    bits = (masks[:, :, None] >> np.arange(6)[None, None, :]) & 1
    if not np.all(bits.sum(axis=2) == 2):
        raise ValueError('The loss contract applies to the two-occupied-spin sector.')
    c = sum(np.sum(bits[:, u] * bits[:, v], axis=1, dtype=np.int64)
            for u, v in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)))
    q = np.sum((bits.sum(axis=1) - np.array((2, 2, 2, 2, 1, 1))) ** 2, axis=1)
    return c, q


def corrected_diagonal(objective_raw, conflict_raw, quota_raw, penalty):
    """Return the integer score and its single, shared normalized diagonal."""
    if isinstance(penalty, bool) or int(penalty) != penalty or penalty <= 0:
        raise ValueError('An explicitly selected positive integer penalty is required.')
    arrays = [np.asarray(x) for x in (objective_raw, conflict_raw, quota_raw)]
    if not all(x.shape == arrays[0].shape and x.ndim == 1 for x in arrays):
        raise ValueError('All diagonals must have the same one-dimensional basis ordering.')
    if not all(np.issubdtype(x.dtype, np.integer) for x in arrays):
        raise TypeError('Pass raw integer diagonals, not separately normalized components.')
    if any(np.any(x < 0) for x in arrays):
        raise ValueError('This problem uses nonnegative raw diagonals.')
    o, c, q = (x.astype(np.int64, copy=False) for x in arrays)
    score = o + int(penalty) * (c + q)
    return score, score.astype(np.float64) / PHASE_SCALE


def spectral_contract(objective_raw, c, q, penalty):
    """Exact integer check; this diagnostic is never a training target."""
    score, _ = corrected_diagonal(objective_raw, c, q, penalty)
    feasible = (c == 0) & (q == 0)
    ground_energy = int(np.min(objective_raw[feasible]))
    target = feasible & (objective_raw == ground_energy)
    actual = score == score.min()
    if not np.array_equal(target, actual):
        raise AssertionError('Penalty Hamiltonian does not encode the constrained target.')
    if not np.array_equal(score[feasible], objective_raw[feasible]):
        raise AssertionError('The objective within the feasible sector changed.')
    return dict(ground_energy=ground_energy, degeneracy=int(target.sum()),
                infeasible_margin=int(score[~feasible].min()) - ground_energy,
                exact_target_mask_matches=True)
