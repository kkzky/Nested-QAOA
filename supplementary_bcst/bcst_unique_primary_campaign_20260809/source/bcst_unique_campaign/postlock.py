"""Unique-ground-primary target evaluator used only behind an immutable lock."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np

from bcst_v2.instance_core import (
    build_training_instance,
    derive_targets,
    rts99,
)

from appendix_v3 import target_evaluator as legacy_target_evaluator

from .contract import canonical_json_bytes


class UniqueTargetError(ValueError):
    """Raised when a post-lock target is not the contracted unique optimum."""


_LEGACY_EVALUATOR = legacy_target_evaluator.evaluate_bcst_targets


def _exact_terminal_metrics(
    probability: float, terminal_ru: int
) -> tuple[int | float, int | float, dict[str, str], dict[str, str]]:
    """Encode exact integer RTS/cost without ever coercing a huge int to float."""

    repetitions = rts99(probability)
    if isinstance(repetitions, float):
        if not math.isinf(repetitions) or repetitions < 0.0:
            raise UniqueTargetError("RTS99 returned an invalid extended-real value")
        cost: int | float = math.inf
        repetitions_record = {"kind": "positive_infinity"}
        cost_record = {"kind": "positive_infinity"}
    else:
        repetitions = int(repetitions)
        cost = repetitions * int(terminal_ru)
        repetitions_record = {
            "kind": "finite",
            "integer_decimal": str(repetitions),
        }
        cost_record = {"kind": "finite", "integer_decimal": str(cost)}
    return repetitions, cost, repetitions_record, cost_record


def evaluate_unique_targets(
    *,
    job_record: Mapping[str, Any],
    state: np.ndarray,
    cohort_lock_path: object,
    expected_lock_sha256: str,
) -> Mapping[str, Any]:
    """Replay the existing strict evaluator and add unique-primary fields."""

    base = dict(
        _LEGACY_EVALUATOR(
            job_record=job_record,
            state=state,
            cohort_lock_path=cohort_lock_path,
            expected_lock_sha256=expected_lock_sha256,
        )
    )
    N = int(job_record["N"])
    seed = int(job_record["problem_seed"])
    instance = build_training_instance(N, seed, 1, 2, 1, 16)
    targets = derive_targets(instance)
    if targets.ground_degeneracy != 1 or targets.unique_ground.size != 1:
        raise UniqueTargetError(
            f"frozen instance N={N}, seed={seed} is not uniquely grounded"
        )
    amplitudes = np.asarray(state, dtype="<c16")
    probabilities = np.square(np.abs(amplitudes), dtype=np.float64)
    norm = float(np.sum(probabilities))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise UniqueTargetError("locked state is not normalized")
    probabilities /= norm
    feasible_mass = float(np.sum(probabilities[instance.feasible_indices]))
    unique_probability = float(probabilities[int(targets.unique_ground[0])])
    if not (0.0 <= unique_probability <= feasible_mass + 1e-14 <= 1.0 + 1e-10):
        raise UniqueTargetError("unique/feasible probability relation is invalid")
    conditional = unique_probability / feasible_mass if feasible_mass > 0.0 else 0.0
    ground_metric = dict(base["targets"]["ground"])
    base_probability = float(ground_metric.get("probability"))
    if not math.isclose(
        unique_probability, base_probability, rel_tol=0.0, abs_tol=1e-15
    ):
        raise UniqueTargetError(
            "normalized unique probability differs from the normalized ground target"
        )
    (
        _expected_rts,
        _expected_cost,
        expected_rts_record,
        expected_cost_record,
    ) = _exact_terminal_metrics(
        unique_probability,
        int(base["terminal_RU"]),
    )
    if (
        ground_metric.get("RTS99") != expected_rts_record
        or ground_metric.get("total_logical_cost") != expected_cost_record
    ):
        raise UniqueTargetError(
            "unique probability does not reproduce the reported RTS99/logical cost"
        )
    base["targets"] = dict(base["targets"])
    base["targets"]["unique_ground"] = ground_metric
    base["primary_target"] = "unique_ground_state"
    base["secondary_targets"] = ["best2", "best8"]
    base["ground_degeneracy"] = 1
    base["unique_ground_applicable"] = True
    base["ground_energy_exact"] = {
        "numerator_decimal": str(
            int(instance.exact_raw_energy_numerators[int(targets.unique_ground[0])])
        ),
        "denominator_decimal": str(1 << 54),
    }
    base["ground_state_sha256"] = targets.hashes["ground"]
    base["feasible_mass"] = feasible_mass
    base["unique_probability_unconditional"] = unique_probability
    base["unique_probability_conditional_on_feasibility"] = conditional
    construction_hashes = {str(k): str(v).upper() for k, v in instance.hashes.items()}
    base["construction_hashes"] = construction_hashes
    base["instance_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "N": N,
                "problem_seed": seed,
                "setting": "d4_1of2_r6_1of16",
                "counts": dict(instance.counts),
                "resources": {
                    key: int(getattr(instance.resources, key))
                    for key in ("B", "C", "X", "O", "S", "P1")
                },
                "hashes": construction_hashes,
            }
        )
    ).hexdigest().upper()
    return base


__all__ = ["UniqueTargetError", "evaluate_unique_targets"]
