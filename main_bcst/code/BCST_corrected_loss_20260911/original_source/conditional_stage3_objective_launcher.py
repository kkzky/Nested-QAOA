"""Dormant launcher for the conditional three-stage BCST objective campaign.

This module deliberately contains no target construction or target-probability
code.  It can validate the target-free Stage-A and final structural handoffs,
freeze the twelve coefficient tables, and enumerate the complete paper-facing
method/resource matrix.  It does *not* run a variational outcome by itself.

Two independent gates are enforced:

1. the complete 16-restart Stage-A aggregate must say
   ``PROCEED_TO_EXPENSIVE_STAGES``; and
2. the later target-free structural handoff must document both structural
   orders, the complete finite-control envelope, and a saved-angle complex128
   ``FINAL_STRUCTURAL_PASS``.

The objective-table generator and method-manifest builder accept only the
private activation token returned by the second validator.  Consequently the
CLI cannot materialize an objective table from Stage A alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent

STAGE_A_SCHEMA = "nonredundant-bcst-target-free-structural-screen-v2"
STRUCTURAL_HANDOFF_SCHEMA = "nonredundant-bcst-final-structural-handoff-v1"
COEFFICIENT_SCHEMA = "nonredundant-bcst-objective-coefficients-v1"
METHOD_MATRIX_SCHEMA = "nonredundant-bcst-stage3-method-matrix-v1"
CAMPAIGN_SCHEMA = "nonredundant-bcst-stage3-objective-campaign-v1"
POST_DEVELOPMENT_SCHEMA = "nonredundant-bcst-stage3-post-development-tasks-v1"

BLOCKS = 5
LABELS = 6
EXPECTED_SUPPORTS = {"native": 759_375, "quota": 4_530, "conflict": 6_570, "final": 390}
EXPECTED_RESTART_SEEDS = tuple(range(2_026_081_100, 2_026_081_116))
STAGE_A_CHECK_KEYS = {
    "median_final_mass_at_least_0p25",
    "median_stage_gain_at_least_2",
    "coarse_cost_margin_at_least_1p25",
    "fixed_uniform_cost_margin_at_least_1p25",
    "coarse_paired_wins",
    "fixed_uniform_paired_wins",
    "all_costs_finite_positive",
}
REPLAY_CHECK_KEYS = {
    "source_preliminary_pass",
    "no_optimization_or_reselection",
    "median_final_mass_at_least_0p25",
    "median_stage_gain_at_least_2",
    "geometric_mean_cost_margin_at_least_1p25",
    "paired_wins_at_least_75_percent",
    "complex128_norm_tolerance",
    "all_costs_finite_positive",
}
STRUCTURAL_FIRST_DEPTHS = (4, 8, 12)
STRUCTURAL_SECOND_DEPTHS = (1, 2, 3, 4, 6, 8, 12)
STRUCTURAL_DIRECT_PRESET_DEPTHS = (32, 48, 64, 96, 128, 192, 256)
DEVELOPMENT_SEEDS = (20_260_811, 20_260_812, 20_260_813, 20_260_814)
HELD_OUT_SEEDS = tuple(range(20_260_815, 20_260_823))
OBJECTIVE_SEEDS = DEVELOPMENT_SEEDS + HELD_OUT_SEEDS
OBJECTIVE_DEPTHS = (16, 32, 64, 96, 128)
PENALTY_MULTIPLIERS = (0.5, 1.0, 2.0)
LOCAL_SWEEPS = (1, 2, 4)
RESIDUAL_EDGES = ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4))
SPECTRAL_PAIRS = tuple(
    (left, right)
    for left in range(LABELS)
    for right in range(LABELS)
    if abs(left - right) <= 1
)
CANONICAL_BASIS_STARTS = {
    "capacity": {"native_shell_index": 2_174, "local_pair_digits": (0, 0, 9, 9, 14),
                 "bit_masks": (3, 3, 12, 12, 48), "sector": "Q=0"},
    "conflict": {"native_shell_index": 30_524, "local_pair_digits": (0, 9, 0, 9, 14),
                 "bit_masks": (3, 12, 3, 12, 48), "sector": "C=0"},
    "exact-final": {"native_shell_index": 30_524, "local_pair_digits": (0, 9, 0, 9, 14),
                    "bit_masks": (3, 12, 3, 12, 48), "sector": "Q=C=0"},
}

# Manuscript-native complete-replay resource units.
P_RU = 5
X_RU = 75
Q_RU = 120
C_NATIVE_RU = 30
C_PAULI_RU = 30
O_RU = 540
S_RU = 900
T_O_RU = 540
T_ALL_RU = 720
BASIS_PREP_RU = 10

CAPACITY_PROXY_RU = 1_050
CAPACITY_SWEEP_RU = 8_400
CONFLICT_PROXY_RU = 1_650
CONFLICT_SWEEP_RU = 33_600
EXACT_FINAL_PROXY_RU = 1_850
EXACT_FINAL_SWEEP_RU = 108_080
FULL_GUARD_PROXY_RU = 2_250
FULL_GUARD_SWEEP_RU = 211_200


class ActivationError(RuntimeError):
    """A required target-free activation artifact failed closed."""


class ProtocolBlocker(RuntimeError):
    """The frozen protocol cannot yet be launched without weakening it."""


@dataclass(frozen=True)
class StageAGate:
    manifest_id: str
    selected_optimistic_method: str
    status: str = "PROCEED_TO_EXPENSIVE_STAGES"


@dataclass(frozen=True)
class OrderSelection:
    order: str
    first: str
    second: str
    depth_first: int
    depth_second: int
    complex64_algorithm: str

    @property
    def p_q(self) -> int:
        return self.depth_first if self.first == "Q" else self.depth_second

    @property
    def p_c(self) -> int:
        return self.depth_first if self.first == "C" else self.depth_second


@dataclass(frozen=True)
class _ActivatedStructuralHandoff:
    """Capability object; construct only through ``activate_structural_handoff``."""

    primary_order: str
    q_then_c: OrderSelection
    c_then_q: OrderSelection
    stage_a_manifest_id: str
    structural_handoff_id: str


@dataclass(frozen=True)
class ResourceLedger:
    primary: int | None
    native_c: int | None
    pauli_c: int | None
    terminal_all_sensitivity: int | None
    generator_proxy: int | None = None
    finite_rank: bool = True


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    role: str
    order: str | None
    p_q: int | None
    p_c: int | None
    p_o: int | None
    phase_depth: int | None
    phase_variant: str
    mixer: str
    trotter_sweeps: int | None
    penalty_multiplier: float
    parameter_count: int
    forward_evaluations: tuple[int, ...]
    gradient_updates: tuple[int, ...]
    endpoint_evaluations: tuple[int, ...]
    resource: ResourceLedger
    primary_finite_rank_eligible: bool
    development_only: bool = False
    paired_snapshot: bool = False
    matched_against: tuple[str, ...] = ()
    initial_reference: str = "family-defined target-blind reference"
    training_loss: str = "E[lambda*(Qtilde+Ctilde)+Otilde]"
    terminal_verification: str = "T_O=540 primary; T_all=720 sensitivity"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActivationError(f"{path} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActivationError(message)


def _all_exactly_false(guard: object) -> bool:
    return isinstance(guard, dict) and bool(guard) and all(value is False for value in guard.values())


def validate_stage_a_aggregate(payload: Mapping[str, object]) -> StageAGate:
    """Validate the scientific Stage-A aggregate, not a shard or smoke file."""

    _require(payload.get("schema") == STAGE_A_SCHEMA, "Stage-A schema mismatch")
    _require(payload.get("profile") == "valid", "smoke/deviant Stage A cannot activate")
    _require(
        payload.get("execution_stage") == "A_NECESSITY_COMPLETE_16_RESTARTS",
        "Stage-A artifact is not the complete 16-restart aggregate",
    )
    _require(payload.get("scientific_decision_allowed") is True, "Stage-A decision is excluded")
    _require(payload.get("excluded_smoke") is False, "excluded smoke cannot activate")
    _require(payload.get("protocol_deviation") is False, "Stage-A protocol deviation")
    _require(payload.get("stage_a_shards_complete") is True, "Stage-A shards are incomplete")
    _require(payload.get("early_promotion_used") is False, "early Stage-A promotion is forbidden")
    _require(_all_exactly_false(payload.get("target_guard")), "Stage A touched objective/target data")
    _require(payload.get("support_counts") == EXPECTED_SUPPORTS, "Stage-A support counts mismatch")

    accounting = payload.get("evaluation_accounting")
    _require(isinstance(accounting, dict), "Stage-A evaluation accounting is absent")
    _require(
        accounting.get("all_non_diagnostic_rows_match_budget") is True,
        "Stage-A optimizer budgets are unequal",
    )
    gate = payload.get("gate")
    _require(isinstance(gate, dict), "Stage-A gate is absent")
    _require(gate.get("status") == "PROCEED_TO_EXPENSIVE_STAGES", "Stage A did not PROCEED")
    _require(gate.get("pass") is True, "Stage-A pass flag is not true")
    checks = gate.get("checks")
    _require(isinstance(checks, dict) and set(checks) == STAGE_A_CHECK_KEYS,
             "Stage-A check set does not match the frozen engine")
    _require(all(value is True for value in checks.values()), "one or more Stage-A checks failed")

    rows = payload.get("methods")
    _require(isinstance(rows, list) and len(rows) == 6, "Stage-A coarse matrix must contain six rows")
    expected_names = {f"Coarse-only-{kind}-p{depth}" for depth in (4, 8, 12) for kind in ("Q", "C")}
    _require({row.get("algorithm") for row in rows if isinstance(row, dict)} == expected_names,
             "Stage-A coarse method set is incomplete or contains extras")
    for row in rows:
        _require(isinstance(row, dict), "invalid Stage-A row")
        _require(tuple(row.get("initialization_seeds", ())) == EXPECTED_RESTART_SEEDS,
                 "Stage-A restart seed/order mismatch")
        _require(row.get("aggregate_pipeline_evaluations_per_restart") == 4_800,
                 "Stage-A row does not use 4,800 forward evaluations")
        costs = row.get("feasibility_RTS99_x_primary_RU")
        _require(isinstance(costs, list) and len(costs) == 16, "Stage-A cost vector is incomplete")
        _require(all(isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0 for x in costs),
                 "Stage-A contains a missing/nonfinite/nonpositive cost")

    selected = gate.get("selected_optimistic_lp_bound")
    _require(isinstance(selected, dict) and isinstance(selected.get("algorithm"), str),
             "Stage-A optimistic selection is missing")
    manifest_id = str(payload.get("manifest_sha256", ""))
    _require(bool(manifest_id), "Stage-A aggregate lacks its frozen manifest identifier")
    return StageAGate(manifest_id, str(selected["algorithm"]))


def _parse_order_selection(name: str, raw: object) -> OrderSelection:
    _require(name in {"Q-C", "C-Q"}, f"invalid structural order {name}")
    _require(isinstance(raw, dict), f"missing {name} structural selection")
    first, second = name.split("-")
    _require(raw.get("first") == first and raw.get("second") == second,
             f"{name} selection has inconsistent stage labels")
    p_first = raw.get("depth_first")
    p_second = raw.get("depth_second")
    _require(isinstance(p_first, int) and p_first in {4, 8, 12}, f"invalid {name} first depth")
    _require(isinstance(p_second, int) and p_second in {1, 2, 3, 4, 6, 8, 12},
             f"invalid {name} learned-projector depth")
    _require(raw.get("selected_by_frozen_2pct_mass_then_ru_rule") is True,
             f"{name} was not selected by the frozen target-free rule")
    algorithm = raw.get("complex64_algorithm")
    _require(isinstance(algorithm, str) and algorithm, f"{name} selected row is missing")
    return OrderSelection(name, first, second, p_first, p_second, algorithm)


def _finite_positive_vector(value: object, size: int, label: str) -> np.ndarray:
    _require(isinstance(value, list) and len(value) == size, f"{label} vector is incomplete")
    array = np.asarray(value, dtype=np.float64)
    _require(bool(np.all(np.isfinite(array))) and bool(np.all(array > 0)),
             f"{label} vector is nonfinite or nonpositive")
    return array


def _validate_structural_method_row(row: object) -> dict[str, object]:
    _require(isinstance(row, dict), "invalid structural method row")
    _require(isinstance(row.get("algorithm"), str) and bool(row["algorithm"]),
             "structural row lacks an algorithm ID")
    _require(tuple(row.get("initialization_seeds", ())) == EXPECTED_RESTART_SEEDS,
             "structural row restart seed/order mismatch")
    _require(row.get("aggregate_pipeline_evaluations_per_restart") == 4_800,
             "structural row does not use 4,800 forward evaluations")
    _finite_positive_vector(row.get("feasibility_RTS99_x_primary_RU"), 16,
                            f"{row['algorithm']} cost")
    final = row.get("final_mass")
    _require(isinstance(final, list) and len(final) == 16,
             f"{row['algorithm']} final-mass vector is incomplete")
    final_array = np.asarray(final, dtype=np.float64)
    _require(bool(np.all(np.isfinite(final_array))) and bool(np.all((final_array >= 0) & (final_array <= 1))),
             f"{row['algorithm']} has invalid final mass")
    norms = row.get("state_norm")
    _require(isinstance(norms, list) and len(norms) == 16,
             f"{row['algorithm']} state-norm vector is incomplete")
    norm_array = np.asarray(norms, dtype=np.float64)
    _require(bool(np.all(np.isfinite(norm_array)))
             and float(np.max(np.abs(norm_array - 1.0))) <= 2e-5,
             f"{row['algorithm']} complex64 norm evidence failed")
    ledger = row.get("RU")
    _require(isinstance(ledger, dict) and isinstance(ledger.get("primary"), int)
             and int(ledger["primary"]) > 0, f"{row['algorithm']} lacks a positive primary RU")
    return row


def _structural_gate_metrics(
    winner: Mapping[str, object], controls: Sequence[Mapping[str, object]], *, norm_tolerance: float
) -> tuple[dict[str, bool], set[str]]:
    _require(bool(controls), "structural gate lacks finite controls")
    winner_cost = _finite_positive_vector(winner.get("feasibility_RTS99_x_primary_RU"), 16,
                                          "selected LP cost")
    final = np.asarray(winner.get("final_mass"), dtype=np.float64)
    stage1 = winner.get("stage1_final_mass")
    _require(isinstance(stage1, list) and len(stage1) == 16,
             "selected LP lacks its Stage-1 mass vector")
    stage1_array = np.asarray(stage1, dtype=np.float64)
    _require(bool(np.all(np.isfinite(stage1_array))) and bool(np.all(stage1_array >= 0)),
             "selected LP has invalid Stage-1 mass")
    control_costs = np.asarray([
        _finite_positive_vector(row.get("feasibility_RTS99_x_primary_RU"), 16,
                                f"{row.get('algorithm')} cost")
        for row in controls
    ])
    envelope_indices = np.argmin(control_costs, axis=0)
    strongest = control_costs[envelope_indices, np.arange(16)]
    ratios = strongest / winner_cost
    ratio_geomean = float(np.exp(np.mean(np.log(ratios))))
    norms = np.asarray(winner.get("state_norm"), dtype=np.float64)
    checks = {
        "median_final_mass_at_least_0p25": float(np.median(final)) >= 0.25,
        "median_stage_gain_at_least_2": float(
            np.median(final / np.maximum(stage1_array, 1e-30))
        ) >= 2.0,
        "geometric_mean_cost_margin_at_least_1p25": ratio_geomean >= 1.25,
        "paired_wins_at_least_75_percent": int(np.sum(ratios > 1.0)) >= 12,
        "norm_tolerance": float(np.max(np.abs(norms - 1.0))) <= norm_tolerance,
        "all_costs_finite_positive": True,
    }
    envelope = {str(controls[int(index)]["algorithm"]) for index in envelope_indices}
    return checks, envelope


def _select_structural_full(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    _require(bool(rows), "no structural full-LP rows")
    medians = np.asarray([np.median(np.asarray(row["final_mass"], dtype=np.float64)) for row in rows])
    _require(bool(np.all(np.isfinite(medians))), "nonfinite full-LP mass")
    maximum = float(np.max(medians))
    eligible = [row for row, median in zip(rows, medians) if float(median) >= 0.98 * maximum]
    return min(
        eligible,
        key=lambda row: (
            int(row["RU"]["primary"]),
            -float(np.median(np.asarray(row["final_mass"], dtype=np.float64))),
            str(row["algorithm"]),
        ),
    )


def _structural_comparator_total_ru(family: str, depth: int) -> int:
    if family == "direct_joint":
        return P_RU + depth * (420 + X_RU) + 420
    if family == "direct_separate":
        return P_RU + depth * (Q_RU + C_NATIVE_RU + X_RU) + 420
    if family == "fixed_native_joint":
        return P_RU + depth * (420 + 2 * P_RU + S_RU) + 420
    if family == "fixed_native_separate":
        return P_RU + depth * (Q_RU + C_NATIVE_RU + 2 * P_RU + S_RU) + 420
    raise ValueError(family)


def _required_structural_depths(selected_ru: int) -> dict[str, tuple[int, ...]]:
    candidates = sorted(
        set(STRUCTURAL_DIRECT_PRESET_DEPTHS)
        | {left + right for left in STRUCTURAL_FIRST_DEPTHS for right in STRUCTURAL_SECOND_DEPTHS}
    )
    return {
        family: tuple(depth for depth in candidates
                      if _structural_comparator_total_ru(family, depth) <= selected_ru)
        for family in ("direct_joint", "direct_separate", "fixed_native_joint", "fixed_native_separate")
    }


def activate_structural_handoff(
    stage_a_payload: Mapping[str, object], handoff: Mapping[str, object]
) -> _ActivatedStructuralHandoff:
    """Return the only token accepted by objective-dependent builders."""

    stage_a = validate_stage_a_aggregate(stage_a_payload)
    _require(handoff.get("schema") == STRUCTURAL_HANDOFF_SCHEMA, "structural handoff schema mismatch")
    _require(handoff.get("status") == "FINAL_STRUCTURAL_PASS", "final structural handoff did not pass")
    _require(handoff.get("pass") is True, "final structural pass flag is not true")
    _require(handoff.get("profile") == "valid", "structural handoff is not valid-profile")
    _require(handoff.get("stage_a_manifest_id") == stage_a.manifest_id,
             "structural handoff does not descend from this Stage-A aggregate")
    _require(_all_exactly_false(handoff.get("target_guard")),
             "structural handoff touched objective/target data")
    _require(handoff.get("all_16_restarts_retained") is True, "structural handoff lost restarts")
    _require(handoff.get("all_unfiltered_rows_retained") is True, "structural handoff filtered rows")
    _require(handoff.get("nonfinite_costs_present") is False,
             "structural handoff contains invalid costs")

    methods_raw = handoff.get("complex64_methods")
    _require(isinstance(methods_raw, list), "structural handoff lacks its complex64 rows")
    methods = [_validate_structural_method_row(row) for row in methods_raw]
    full = [row for row in methods if row.get("family") == "full_lp"]
    expected_full = {
        (first, p1, p2)
        for first in ("Q", "C")
        for p1 in STRUCTURAL_FIRST_DEPTHS
        for p2 in STRUCTURAL_SECOND_DEPTHS
    }
    observed_full = {
        (row.get("first"), row.get("depth_first"), row.get("depth_second")) for row in full
    }
    _require(observed_full == expected_full and len(full) == len(expected_full),
             "structural full-LP grid is incomplete or contains duplicates")
    for row in full:
        first = str(row["first"])
        p1, p2 = int(row["depth_first"]), int(row["depth_second"])
        first_circuit = P_RU + p1 * ((Q_RU if first == "Q" else C_NATIVE_RU) + X_RU)
        second_phase = C_NATIVE_RU if first == "Q" else Q_RU
        expected_ru = (1 + 2 * p2) * first_circuit + p2 * (second_phase + S_RU) + 420
        _require(int(row["RU"]["primary"]) == expected_ru,
                 f"structural full-LP RU mismatch for {row['algorithm']}")

    replay = handoff.get("saved_angle_complex128_replay")
    _require(isinstance(replay, dict), "saved-angle complex128 replay evidence is absent")
    _require(replay.get("status") == "FINAL_STRUCTURAL_PASS" and replay.get("pass") is True,
             "saved-angle complex128 replay did not pass")
    _require(replay.get("saved_angles_only") is True, "complex128 replay did not use saved angles")
    _require(replay.get("optimization_performed") is False, "complex128 replay reoptimized")
    _require(replay.get("configuration_reselected") is False,
             "complex128 replay reselected a configuration")
    _require(replay.get("max_norm_error") is not None and
             math.isfinite(float(replay["max_norm_error"])) and
             float(replay["max_norm_error"]) <= 1e-10,
             "complex128 replay norm tolerance failed")
    replay_checks = replay.get("checks")
    _require(isinstance(replay_checks, dict) and set(replay_checks) == REPLAY_CHECK_KEYS
             and all(value is True for value in replay_checks.values()),
             "complex128 replay check matrix is incomplete or failed")

    selections = handoff.get("selected_orders")
    _require(isinstance(selections, dict), "both-order structural selections are absent")
    q_then_c = _parse_order_selection("Q-C", selections.get("Q-C"))
    c_then_q = _parse_order_selection("C-Q", selections.get("C-Q"))
    selected_qc = _select_structural_full([row for row in full if row.get("first") == "Q"])
    selected_cq = _select_structural_full([row for row in full if row.get("first") == "C"])
    for declared, recomputed in ((q_then_c, selected_qc), (c_then_q, selected_cq)):
        _require(declared.complex64_algorithm == recomputed.get("algorithm"),
                 f"{declared.order} declared selection differs from recomputed 2%-mass/lower-RU rule")
        _require((declared.depth_first, declared.depth_second) ==
                 (recomputed.get("depth_first"), recomputed.get("depth_second")),
                 f"{declared.order} declared depths differ from recomputed selection")
    primary = handoff.get("primary_order")
    _require(primary in {"Q-C", "C-Q"}, "primary structural order is missing")
    _require(handoff.get("primary_order_selected_target_free") is True,
             "primary order was not frozen target-free")
    recomputed_primary = _select_structural_full(full)
    expected_primary = "Q-C" if recomputed_primary.get("first") == "Q" else "C-Q"
    _require(primary == expected_primary, "declared primary order differs from recomputed target-free winner")
    required_families = {
        "coarse", "warm", "local", "direct_joint", "direct_separate",
        "fixed_native_joint", "fixed_native_separate",
    }
    finite_controls = [row for row in methods if row.get("family") in required_families]
    complex64_checks, required_replay_controls = _structural_gate_metrics(
        recomputed_primary, finite_controls, norm_tolerance=2e-5
    )
    _require(all(complex64_checks.values()),
             "complex64 structural winner fails recomputed mass/gain/cost/norm gate")
    replay_rows = replay.get("replayed_rows")
    _require(isinstance(replay_rows, list) and len(replay_rows) >= 2,
             "complex128 replay lacks winner/control row evidence")
    replayed_algorithms: set[str] = set()
    for row in replay_rows:
        _require(isinstance(row, dict) and isinstance(row.get("algorithm"), str),
                 "invalid complex128 replay row")
        replayed_algorithms.add(str(row["algorithm"]))
        _require(row.get("optimization_performed") is False
                 and row.get("configuration_reselected") is False
                 and row.get("angles_reselected") is False,
                 "a complex128 replay row reoptimized or reselected")
        norms = row.get("state_norm")
        _require(isinstance(norms, list) and len(norms) == 16,
                 "complex128 replay row norm vector is incomplete")
        norm_array = np.asarray(norms, dtype=np.float64)
        _require(bool(np.all(np.isfinite(norm_array)))
                 and float(np.max(np.abs(norm_array - 1.0))) <= 1e-10,
                 "complex128 replay row norm evidence failed")
        _finite_positive_vector(row.get("feasibility_RTS99_x_primary_RU"), 16,
                                f"{row['algorithm']} replay cost")
        final_mass = row.get("final_mass")
        stage1_mass = row.get("stage1_final_mass")
        _require(isinstance(final_mass, list) and len(final_mass) == 16,
                 "complex128 replay final-mass vector is incomplete")
        # Only the learned-history winner has a Stage-1 precursor.  A one-stage
        # coarse/direct/fixed control correctly records ``None`` here; forcing
        # it to fabricate a precursor vector would make a real saved-angle
        # handoff fail even though the vector is never used for control rows.
        if str(row["algorithm"]) == str(recomputed_primary["algorithm"]):
            _require(isinstance(stage1_mass, list) and len(stage1_mass) == 16,
                     "complex128 replay winner Stage-1-mass vector is incomplete")
        else:
            _require(stage1_mass is None or
                     (isinstance(stage1_mass, list) and len(stage1_mass) == 16),
                     "complex128 replay control has malformed Stage-1 metadata")
    expected_replay_algorithms = {str(recomputed_primary["algorithm"])} | required_replay_controls
    _require(replayed_algorithms == expected_replay_algorithms,
             "complex128 replay rows do not exactly equal winner plus complex64 control envelope")
    declared_controls = replay.get("selected_control_algorithms")
    _require(isinstance(declared_controls, list)
             and set(map(str, declared_controls)) == required_replay_controls,
             "complex128 control-envelope replay evidence is incomplete")
    replay_by_algorithm = {str(row["algorithm"]): row for row in replay_rows}
    complex128_checks, _ = _structural_gate_metrics(
        replay_by_algorithm[str(recomputed_primary["algorithm"])],
        [replay_by_algorithm[name] for name in sorted(required_replay_controls)],
        norm_tolerance=1e-10,
    )
    _require(all(complex128_checks.values()),
             "complex128 saved-angle rows fail recomputed mass/gain/cost/norm gate")

    # Recompute the finite-control coverage from rows rather than trusting a
    # self-attested `complete` boolean.
    observed_families = {str(row.get("family")) for row in methods}
    _require(required_families.issubset(observed_families),
             "finite structural-control family matrix is incomplete")
    coarse_rows = [row for row in methods if row.get("family") == "coarse"]
    _require({(row.get("first"), row.get("depth_first")) for row in coarse_rows}
             == {(first, depth) for first in ("Q", "C") for depth in STRUCTURAL_FIRST_DEPTHS}
             and len(coarse_rows) == 6,
             "structural coarse Q/C grid is incomplete or duplicated")
    for row in coarse_rows:
        phase = Q_RU if row["first"] == "Q" else C_NATIVE_RU
        expected_ru = P_RU + int(row["depth_first"]) * (phase + X_RU) + 420
        _require(int(row["RU"]["primary"]) == expected_ru,
                 f"coarse RU mismatch for {row['algorithm']}")
    for selected in (q_then_c, c_then_q):
        matching_warm = [row for row in methods if row.get("family") == "warm"
                         and row.get("first") == selected.first
                         and row.get("depth_first") == selected.depth_first
                         and row.get("depth_second") == selected.depth_second]
        _require(bool(matching_warm), f"missing selected-depth warm control for {selected.order}")
        first_phase = Q_RU if selected.first == "Q" else C_NATIVE_RU
        first_circuit = P_RU + selected.depth_first * (first_phase + X_RU)
        warm_expected = first_circuit + selected.depth_second * (
            Q_RU + C_NATIVE_RU + X_RU
        ) + 420
        _require(all(row.get("mixer") == "complete block-XY"
                     and int(row["RU"]["primary"]) == warm_expected
                     for row in matching_warm),
                 f"warm mixer/RU mismatch for {selected.order}")
        matching_local = [row for row in methods if row.get("family") == "local"
                          and row.get("first") == selected.first
                          and row.get("depth_first") == selected.depth_first
                          and row.get("depth_second") == selected.depth_second]
        sweeps = {
            row.get("trotter_sweeps", row.get("replay_spec", {}).get("trotter_sweeps"))
            for row in matching_local
        }
        _require(sweeps == {1, 2, 4}, f"selected-depth local sweep matrix incomplete for {selected.order}")
        expected_mixer = "capacity_switch" if selected.first == "Q" else "conflict_guarded"
        local_phase = C_NATIVE_RU if selected.first == "Q" else Q_RU
        sweep_ru = CAPACITY_SWEEP_RU if selected.first == "Q" else CONFLICT_SWEEP_RU
        _require(all(
            row.get("local_mixer") == expected_mixer
            and int(row["RU"]["primary"]) == first_circuit
            + selected.depth_second * (local_phase + int(row.get("trotter_sweeps")) * sweep_ru)
            + 420
            for row in matching_local
        ), f"local mixer/RU mismatch for {selected.order}")

    required_depths = _required_structural_depths(int(recomputed_primary["RU"]["primary"]))
    completed_depths = {
        family: tuple(sorted({int(row["depth_second"]) for row in methods
                              if row.get("family") == family}))
        for family in required_depths
    }
    for family, depths in required_depths.items():
        _require(set(depths).issubset(set(completed_depths[family])),
                 f"structural comparator ladder incomplete for {family}")
        for row in methods:
            if row.get("family") == family:
                _require(int(row["RU"]["primary"]) ==
                         _structural_comparator_total_ru(family, int(row["depth_second"])),
                         f"structural comparator RU mismatch for {row['algorithm']}")
    declared_required = handoff.get("required_comparator_depth_manifest")
    declared_completed = handoff.get("completed_comparator_depth_manifest")
    _require(isinstance(declared_required, dict) and
             {key: tuple(value) for key, value in declared_required.items()} == required_depths,
             "declared structural comparator requirements differ from recomputation")
    _require(isinstance(declared_completed, dict) and all(
        set(required_depths[key]).issubset(set(declared_completed.get(key, ())))
        for key in required_depths
    ), "declared structural comparator completion is insufficient")

    handoff_id = str(handoff.get("handoff_id", ""))
    _require(bool(handoff_id), "structural handoff lacks an identifier")
    return _ActivatedStructuralHandoff(
        str(primary), q_then_c, c_then_q, stage_a.manifest_id, handoff_id
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _require_token(token: object) -> _ActivatedStructuralHandoff:
    if not isinstance(token, _ActivatedStructuralHandoff):
        raise ActivationError("a validated final structural handoff is required")
    return token


def _coefficient_table(token: _ActivatedStructuralHandoff, seed: int) -> dict[str, object]:
    """Draw raw coefficients only; never apply them to a shell state."""

    _require_token(token)
    if seed not in OBJECTIVE_SEEDS:
        raise ValueError("only the twelve frozen objective seeds are allowed")
    rng = np.random.default_rng(seed)
    assignment = rng.integers(1, 21, size=(BLOCKS, LABELS), dtype=np.int64)
    residual = rng.integers(
        5, 31, size=(len(RESIDUAL_EDGES), len(SPECTRAL_PAIRS)), dtype=np.int64
    )
    raw = {
        "seed": seed,
        "assignment_cost": assignment.tolist(),
        "residual_interference_cost": residual.tolist(),
    }
    return {**raw, "table_sha256": _sha256(raw)}


def coefficient_manifest(
    token: _ActivatedStructuralHandoff,
    independently_audited_seed_table: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze all unfiltered raw tables without evaluating an objective state."""

    _require_token(token)
    tables = [_coefficient_table(token, seed) for seed in OBJECTIVE_SEEDS]
    if independently_audited_seed_table is not None:
        first = tables[0]
        _require(independently_audited_seed_table.get("seed") == DEVELOPMENT_SEEDS[0],
                 "independent seed table has the wrong seed")
        _require(independently_audited_seed_table.get("assignment_cost") == first["assignment_cost"],
                 "seed 20260811 assignment table mismatch")
        _require(independently_audited_seed_table.get("residual_interference_cost") ==
                 first["residual_interference_cost"],
                 "seed 20260811 residual table mismatch")
    body = {
        "schema": COEFFICIENT_SCHEMA,
        "stage_a_manifest_id": token.stage_a_manifest_id,
        "structural_handoff_id": token.structural_handoff_id,
        "generation": {
            "rng": "numpy.random.default_rng / PCG64",
            "draw_order": "5x6 assignment then 5x16 residual table",
            "assignment_integers": "[1,21)",
            "residual_integers": "[5,31)",
            "residual_edges": RESIDUAL_EDGES,
            "spectral_pairs": SPECTRAL_PAIRS,
            "objective_scale": 800,
            "evaluated_on_any_state": False,
            "target_constructed": False,
        },
        "development_seeds": DEVELOPMENT_SEEDS,
        "held_out_seeds": HELD_OUT_SEEDS,
        "tables": tables,
        "all_tables_retained": True,
    }
    return {**body, "manifest_sha256": _sha256(body)}


def phase_ru(name: str, *, pauli_c: bool = False) -> int:
    c = C_PAULI_RU if pauli_c else C_NATIVE_RU
    table = {
        "Q": Q_RU,
        "C": c,
        "O": O_RU,
        "QC_joint": 420,
        "QC_separate": Q_RU + c,
        "CO_joint": c + O_RU,
        "CO_separate": c + O_RU,
        "QO_joint": 720,
        "QO_separate": Q_RU + O_RU,
        "QCO_joint": 720,
        "QCO_separate": Q_RU + c + O_RU,
    }
    if name not in table:
        raise ValueError(name)
    return table[name]


def block_ru(phase: int, depth: int) -> int:
    return P_RU + depth * (phase + X_RU)


def history_ru(prior: int, phase: int, depth: int) -> int:
    return (1 + 2 * depth) * prior + depth * (phase + S_RU)


def warm_ru(prior: int, phase: int, mixer: int, depth: int) -> int:
    return prior + depth * (phase + mixer)


def fixed_ru(phase: int, depth: int) -> int:
    return P_RU + depth * (phase + 2 * P_RU + S_RU)


def report_ru(circuit: int, terminal: int = T_O_RU) -> int:
    return circuit + terminal


def _ledger(
    native_circuit: int | None,
    pauli_circuit: int | None = None,
    *,
    generator_proxy_circuit: int | None = None,
    finite_rank: bool = True,
) -> ResourceLedger:
    if not finite_rank:
        return ResourceLedger(None, None, None, None,
                              None if generator_proxy_circuit is None else report_ru(generator_proxy_circuit),
                              False)
    if native_circuit is None:
        raise ValueError("finite ledger requires a native-C circuit cost")
    pauli = native_circuit if pauli_circuit is None else pauli_circuit
    return ResourceLedger(
        report_ru(native_circuit),
        report_ru(native_circuit),
        report_ru(pauli),
        native_circuit + T_ALL_RU,
        None if generator_proxy_circuit is None else report_ru(generator_proxy_circuit),
        True,
    )


def _optimization_accounting(plan: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    evaluations = tuple(int(value) for value in plan)
    if any(value < 2 for value in evaluations):
        raise ValueError("each optimized call requires an update and endpoint")
    updates = tuple(value - 1 for value in evaluations)
    endpoints = tuple(1 for _ in evaluations)
    return updates, endpoints


def _spec(
    *,
    method_id: str,
    family: str,
    role: str,
    phase_variant: str,
    mixer: str,
    resource: ResourceLedger,
    parameter_count: int,
    evaluation_plan: Sequence[int] = (9_600,),
    order: str | None = None,
    p_q: int | None = None,
    p_c: int | None = None,
    p_o: int | None = None,
    phase_depth: int | None = None,
    sweeps: int | None = None,
    penalty: float = 1.0,
    finite_control: bool = True,
    development_only: bool = False,
    paired_snapshot: bool = False,
    matched_against: Sequence[str] = (),
    initial_reference: str = "family-defined target-blind reference",
    training_loss: str = "E[lambda*(Qtilde+Ctilde)+Otilde]",
    terminal_verification: str = "T_O=540 primary; T_all=720 sensitivity",
) -> MethodSpec:
    evaluations = tuple(int(x) for x in evaluation_plan)
    if not paired_snapshot and sum(evaluations) != 9_600:
        raise ValueError(f"primary method {method_id} is not allocated 9,600 evaluations")
    if paired_snapshot and sum(evaluations) != 4_800:
        raise ValueError("paired precursor must inherit exactly 2,400+2,400 evaluations")
    updates, endpoints = _optimization_accounting([x for x in evaluations if x > 0])
    if paired_snapshot:
        # The trailing zero denotes sampling without a fictitious optimizer.
        updates = updates + (0,)
        endpoints = endpoints + (0,)
    return MethodSpec(
        method_id, family, role, order, p_q, p_c, p_o, phase_depth,
        phase_variant, mixer, sweeps, penalty, parameter_count, evaluations,
        updates, endpoints, resource, finite_control, development_only,
        paired_snapshot, tuple(matched_against), initial_reference,
        training_loss, terminal_verification,
    )


def _order_circuits(selection: OrderSelection, p_o: int) -> tuple[int, int, int, int]:
    """Return native/Pauli structural precursor and full circuit costs."""

    if selection.first == "Q":
        a_native = block_ru(Q_RU, selection.depth_first)
        a_pauli = a_native
        structural_native = history_ru(a_native, C_NATIVE_RU, selection.depth_second)
        structural_pauli = history_ru(a_pauli, C_PAULI_RU, selection.depth_second)
    else:
        a_native = block_ru(C_NATIVE_RU, selection.depth_first)
        a_pauli = block_ru(C_PAULI_RU, selection.depth_first)
        structural_native = history_ru(a_native, Q_RU, selection.depth_second)
        structural_pauli = history_ru(a_pauli, Q_RU, selection.depth_second)
    return (
        structural_native,
        structural_pauli,
        history_ru(structural_native, O_RU, p_o),
        history_ru(structural_pauli, O_RU, p_o),
    )


def _geometric_grid_up_to(cap: int) -> tuple[int, ...]:
    if cap < 1:
        return ()
    values: set[int] = set()
    k = 2
    while True:
        value = 8 * int(math.ceil(2 ** (k / 2)))
        if value > cap:
            break
        values.add(value)
        k += 1
    return tuple(sorted(values))


def _method_group_id(configuration_id: str) -> str:
    """Remove only tunable depth fields; retain order, variant, sweep and lambda."""

    parts = configuration_id.split("__")
    retained = [
        part for part in parts
        if not part.startswith("pO-") and not part.startswith("d-")
    ]
    return "__".join(retained)


def _affine_depths(
    targets: Sequence[tuple[str, int]], base: int, slope: int, phase_depths: Sequence[int]
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    if base < 0 or slope <= 0:
        raise ValueError("invalid affine RU recurrence")
    depths = set(int(d) for d in phase_depths)
    matched: dict[int, set[str]] = {int(d): set() for d in phase_depths}
    for comparator_id, target in targets:
        cap = max(0, (int(target) - base) // slope)
        required = set(_geometric_grid_up_to(cap))
        if cap >= 1:
            required.add(cap)
        required.update(int(d) for d in phase_depths)
        for depth in required:
            depths.add(depth)
            matched.setdefault(depth, set()).add(comparator_id)
    return tuple(sorted(d for d in depths if d >= 1)), {
        depth: tuple(sorted(names)) for depth, names in matched.items()
    }


def build_method_matrix(token: _ActivatedStructuralHandoff) -> dict[str, object]:
    """Enumerate every mandatory quantum configuration before objective outcomes."""

    token = _require_token(token)
    selections = (token.q_then_c, token.c_then_q)
    specs: list[MethodSpec] = []

    def add(item: MethodSpec) -> None:
        if any(existing.method_id == item.method_id for existing in specs):
            raise AssertionError(f"duplicate method id {item.method_id}")
        specs.append(item)

    full_targets: list[tuple[str, int]] = []
    phase_matched_depths: set[int] = set()
    for selection in selections:
        for p_o in OBJECTIVE_DEPTHS:
            structural_native, structural_pauli, full_native, full_pauli = _order_circuits(selection, p_o)
            full_id = f"full__{selection.order}__pQ-{selection.p_q}__pC-{selection.p_c}__pO-{p_o}"
            add(_spec(
                method_id=full_id, family="full_three_stage", role="confirmatory full LP",
                order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                phase_depth=selection.p_q + selection.p_c + p_o,
                phase_variant=f"new-phase-only {selection.first}->{selection.second}->O",
                mixer="block-XY; learned-global; learned-global",
                resource=_ledger(full_native, full_pauli),
                parameter_count=2 * (selection.p_q + selection.p_c + p_o),
                evaluation_plan=(2_400, 2_400, 4_800), finite_control=True,
                initial_reference="native-uniform shell; learned structural histories",
                training_loss="Stage1 new constraint; Stage2 E[Qtilde+Ctilde]; Stage3 E[Qtilde+Ctilde+Otilde]",
            ))
            full_targets.append((full_id, report_ru(full_native)))
            d_sum = selection.p_q + selection.p_c + p_o
            phase_matched_depths.add(d_sum)

            # Frozen penalty and objective-only Stage-3 loss sensitivities use
            # the same circuit and resource ledger but can never replace the
            # lambda=1 cumulative-loss primary row after outcomes are known.
            for penalty in (0.5, 2.0):
                add(_spec(
                    method_id=f"full-penalty-sensitivity__{selection.order}__pQ-{selection.p_q}__pC-{selection.p_c}__pO-{p_o}__lambda-{penalty:g}",
                    family="full_penalty_sensitivity", role="prespecified penalty robustness sensitivity",
                    order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                    phase_depth=d_sum,
                    phase_variant=f"new-phase-only {selection.first}->{selection.second}->O",
                    mixer="block-XY; learned-global; learned-global",
                    resource=_ledger(full_native, full_pauli),
                    parameter_count=2 * d_sum, evaluation_plan=(2_400, 2_400, 4_800),
                    penalty=penalty, finite_control=False,
                    initial_reference="native-uniform shell; learned structural histories",
                    training_loss=f"Stage3 E[{penalty:g}*(Qtilde+Ctilde)+Otilde]",
                    matched_against=(full_id,),
                ))
            add(_spec(
                method_id=f"full-objective-only-sensitivity__{selection.order}__pQ-{selection.p_q}__pC-{selection.p_c}__pO-{p_o}",
                family="full_objective_only_sensitivity",
                role="prespecified Stage-3 objective-only-loss sensitivity; never a rescue row",
                order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                phase_depth=d_sum,
                phase_variant=f"new-phase-only {selection.first}->{selection.second}->O",
                mixer="block-XY; learned-global; learned-global",
                resource=_ledger(full_native, full_pauli), parameter_count=2 * d_sum,
                evaluation_plan=(2_400, 2_400, 4_800), finite_control=False,
                initial_reference="native-uniform shell; learned structural histories",
                training_loss="Stage3 E[Otilde] only; upstream stages retain target-free structural losses",
                matched_against=(full_id,),
            ))

            add(_spec(
                method_id=f"precursor-snapshot__{selection.order}__pO-{p_o}",
                family="paired_precursor_snapshot", role="within-pipeline causal diagnostic",
                order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                phase_depth=selection.p_q + selection.p_c,
                phase_variant="Q,C structural phases only", mixer="block-XY then learned-global",
                resource=_ledger(structural_native, structural_pauli),
                parameter_count=2 * (selection.p_q + selection.p_c),
                evaluation_plan=(2_400, 2_400, 0), finite_control=False, paired_snapshot=True,
                matched_against=(full_id,),
                initial_reference="paired state inside the named full pipeline",
                training_loss="upstream 2,400+2,400 structural training only; no dummy Stage-3 optimization",
            ))

            # All three adjacent collapses, joint and separate.
            for variant in ("joint", "separate"):
                qc = f"QC_{variant}"
                first_depth = selection.p_q + selection.p_c
                native0 = block_ru(phase_ru(qc), first_depth)
                pauli0 = block_ru(phase_ru(qc, pauli_c=True), first_depth)
                native = history_ru(native0, O_RU, p_o)
                pauli = history_ru(pauli0, O_RU, p_o)
                add(_spec(
                    method_id=f"collapse-QC-O__{variant}__{selection.order}__pO-{p_o}",
                    family="adjacent_collapse", role="strongest adjacent-collapse gate candidate",
                    order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                    phase_depth=first_depth + p_o, phase_variant=f"{qc}->O",
                    mixer="block-XY then learned-global", resource=_ledger(native, pauli),
                    parameter_count=(2 if variant == "joint" else 3) * first_depth + 2 * p_o,
                    evaluation_plan=(4_800, 4_800), finite_control=True,
                    matched_against=(full_id,),
                ))

                co = f"CO_{variant}"
                q0 = block_ru(Q_RU, selection.p_q)
                second_depth = selection.p_c + p_o
                native = history_ru(q0, phase_ru(co), second_depth)
                pauli = history_ru(q0, phase_ru(co, pauli_c=True), second_depth)
                add(_spec(
                    method_id=f"collapse-Q-CO__{variant}__{selection.order}__pO-{p_o}",
                    family="adjacent_collapse", role="strongest adjacent-collapse gate candidate",
                    order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                    phase_depth=selection.p_q + second_depth, phase_variant=f"Q->{co}",
                    mixer="block-XY then learned-global", resource=_ledger(native, pauli),
                    parameter_count=2 * selection.p_q + (2 if variant == "joint" else 3) * second_depth,
                    evaluation_plan=(4_800, 4_800), finite_control=True,
                    matched_against=(full_id,),
                ))

                qo = f"QO_{variant}"
                c0_native = block_ru(C_NATIVE_RU, selection.p_c)
                c0_pauli = block_ru(C_PAULI_RU, selection.p_c)
                second_depth = selection.p_q + p_o
                native = history_ru(c0_native, phase_ru(qo), second_depth)
                pauli = history_ru(c0_pauli, phase_ru(qo, pauli_c=True), second_depth)
                add(_spec(
                    method_id=f"collapse-C-QO__{variant}__{selection.order}__pO-{p_o}",
                    family="adjacent_collapse", role="strongest adjacent-collapse gate candidate",
                    order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                    phase_depth=selection.p_c + second_depth, phase_variant=f"C->{qo}",
                    mixer="block-XY then learned-global", resource=_ledger(native, pauli),
                    parameter_count=2 * selection.p_c + (2 if variant == "joint" else 3) * second_depth,
                    evaluation_plan=(4_800, 4_800), finite_control=True,
                    matched_against=(full_id,),
                ))

            # Warm refinements from the complete structural precursor.
            warm_variants = (
                ("objective-only", "O", 2 * p_o),
                ("cumulative-separate", "QCO_separate", 4 * p_o),
                ("cumulative-joint", "QCO_joint", 2 * p_o),
            )
            for label, phase_name, refinement_params in warm_variants:
                native = warm_ru(structural_native, phase_ru(phase_name), X_RU, p_o)
                pauli = warm_ru(structural_pauli, phase_ru(phase_name, pauli_c=True), X_RU, p_o)
                add(_spec(
                    method_id=f"warm-final__{label}__{selection.order}__pO-{p_o}",
                    family="warm_block_xy", role="learned-precursor warm control",
                    order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                    phase_depth=p_o, phase_variant=phase_name, mixer="complete block-XY",
                    resource=_ledger(native, pauli),
                    parameter_count=2 * (selection.p_q + selection.p_c) + refinement_params,
                    evaluation_plan=(2_400, 2_400, 4_800), finite_control=True,
                    matched_against=(full_id,),
                    initial_reference="approximate learned final-structural precursor",
                    training_loss=("E[Otilde]" if label == "objective-only" else
                                   "E[Qtilde+Ctilde+Otilde]"),
                ))

            # Local objective mixer from an approximate structural precursor:
            # four full guards are mandatory, never the cheaper exact-sector ledger.
            for sweeps in LOCAL_SWEEPS:
                mixer = sweeps * FULL_GUARD_SWEEP_RU
                proxy_native = warm_ru(structural_native, O_RU, FULL_GUARD_PROXY_RU, p_o)
                native = warm_ru(structural_native, O_RU, mixer, p_o)
                pauli = warm_ru(structural_pauli, O_RU, mixer, p_o)
                add(_spec(
                    method_id=f"local-final-full-guard__{selection.order}__pO-{p_o}__s-{sweeps}",
                    family="learned_local", role="digitized finite local Stage-3 control",
                    order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                    phase_depth=p_o, phase_variant="O", mixer="75-template four-full-guard switch",
                    sweeps=sweeps,
                    resource=_ledger(native, pauli, generator_proxy_circuit=proxy_native),
                    parameter_count=2 * (selection.p_q + selection.p_c + p_o),
                evaluation_plan=(2_400, 2_400, 4_800), finite_control=True,
                matched_against=(full_id,),
                initial_reference="approximate learned final-structural precursor",
                ))
            add(_spec(
                method_id=f"analog-local-final-full-guard__{selection.order}__pO-{p_o}",
                family="learned_local_analog", role="optimistic probability-only locality sensitivity",
                order=selection.order, p_q=selection.p_q, p_c=selection.p_c, p_o=p_o,
                phase_depth=p_o, phase_variant="O", mixer="exact analog full-guard adjacency exponential",
                resource=_ledger(None, finite_rank=False,
                                 generator_proxy_circuit=warm_ru(structural_native, O_RU, FULL_GUARD_PROXY_RU, p_o)),
                parameter_count=2 * (selection.p_q + selection.p_c + p_o),
                evaluation_plan=(2_400, 2_400, 4_800), finite_control=False,
                matched_against=(full_id,),
                initial_reference="approximate learned final-structural precursor",
            ))

        # Standalone equal-budget precursor control is independent of pO.
        structural_native, structural_pauli, _, _ = _order_circuits(selection, 1)
        add(_spec(
            method_id=f"standalone-precursor__{selection.order}", family="standalone_precursor",
            role="equal-budget Stage-2-only finite control", order=selection.order,
            p_q=selection.p_q, p_c=selection.p_c, phase_depth=selection.p_q + selection.p_c,
            phase_variant="structural Q,C only", mixer="block-XY then learned-global",
            resource=_ledger(structural_native, structural_pauli),
            parameter_count=2 * (selection.p_q + selection.p_c),
            evaluation_plan=(4_800, 4_800), finite_control=True,
            initial_reference="native-uniform shell",
            training_loss="Stage1 new constraint; Stage2 E[Qtilde+Ctilde]",
        ))

    # Learned-Q / learned-C warm and local controls.  These use the unique pQ
    # and pC supplied by each corresponding order; they are not allowed to
    # substitute depths from the other order.
    q_selection = token.q_then_c
    c_selection = token.c_then_q
    for p_o in OBJECTIVE_DEPTHS:
        q_prior = block_ru(Q_RU, q_selection.p_q)
        c_prior_native = block_ru(C_NATIVE_RU, c_selection.p_c)
        c_prior_pauli = block_ru(C_PAULI_RU, c_selection.p_c)
        for start, prior_native, prior_pauli, depth, start_selection in (
            ("learned-Q", q_prior, q_prior, q_selection.p_c + p_o, q_selection),
            ("learned-C", c_prior_native, c_prior_pauli, c_selection.p_q + p_o, c_selection),
        ):
            for label, phase_name, per_layer_params in (
                ("objective-only", "O", 2),
                ("cumulative-separate", "QCO_separate", 4),
                ("cumulative-joint", "QCO_joint", 2),
            ):
                native = warm_ru(prior_native, phase_ru(phase_name), X_RU, depth)
                pauli = warm_ru(prior_pauli, phase_ru(phase_name, pauli_c=True), X_RU, depth)
                prep_depth = q_selection.p_q if start == "learned-Q" else c_selection.p_c
                add(_spec(
                    method_id=f"warm-{start}__{label}__pO-{p_o}", family="warm_block_xy",
                    role="single-constraint learned-state warm control", order=None,
                    p_q=start_selection.p_q, p_c=start_selection.p_c,
                    p_o=p_o, phase_depth=depth,
                    phase_variant=phase_name, mixer="complete block-XY", resource=_ledger(native, pauli),
                    parameter_count=2 * prep_depth + per_layer_params * depth,
                    evaluation_plan=(4_800, 4_800), finite_control=True,
                    initial_reference=start,
                    training_loss=("E[Otilde]" if label == "objective-only" else
                                   "E[Qtilde+Ctilde+Otilde]"),
                ))

        # Local structural replacement followed by learned-global O.
        for start in ("Q", "C"):
            if start == "Q":
                prior_native = q_prior
                prior_pauli = q_prior
                structural_depth = q_selection.p_c
                structural_phase = "C"
                mixer_name = "75-template capacity switch"
                sweep_cost = CAPACITY_SWEEP_RU
                proxy_cost = CAPACITY_PROXY_RU
                order = "Q-C"
                start_selection = q_selection
            else:
                prior_native = c_prior_native
                prior_pauli = c_prior_pauli
                structural_depth = c_selection.p_q
                structural_phase = "Q"
                mixer_name = "75-template symmetric conflict-guarded XY"
                sweep_cost = CONFLICT_SWEEP_RU
                proxy_cost = CONFLICT_PROXY_RU
                order = "C-Q"
                start_selection = c_selection
            for sweeps in LOCAL_SWEEPS:
                local_native = warm_ru(prior_native, phase_ru(structural_phase), sweeps * sweep_cost,
                                       structural_depth)
                local_pauli = warm_ru(prior_pauli, phase_ru(structural_phase, pauli_c=True),
                                      sweeps * sweep_cost, structural_depth)
                proxy_struct = warm_ru(prior_native, phase_ru(structural_phase), proxy_cost,
                                       structural_depth)
                native = history_ru(local_native, O_RU, p_o)
                pauli = history_ru(local_pauli, O_RU, p_o)
                proxy = history_ru(proxy_struct, O_RU, p_o)
                add(_spec(
                    method_id=f"local-structural-{start}__pO-{p_o}__s-{sweeps}",
                    family="learned_local", role="local Stage-2 replacement plus global O",
                    order=order, p_q=start_selection.p_q, p_c=start_selection.p_c,
                    p_o=p_o,
                    phase_depth=structural_depth + p_o, phase_variant=f"{structural_phase}->O",
                    mixer=mixer_name + "; learned-global", sweeps=sweeps,
                    resource=_ledger(native, pauli, generator_proxy_circuit=proxy),
                    parameter_count=2 * ((q_selection.p_q if start == "Q" else c_selection.p_c)
                                         + structural_depth + p_o),
                    evaluation_plan=(2_400, 2_400, 4_800), finite_control=True,
                    initial_reference=f"learned {start}=0 preparation",
                ))
            proxy_struct = warm_ru(
                prior_native, phase_ru(structural_phase), proxy_cost, structural_depth
            )
            proxy_total = history_ru(proxy_struct, O_RU, p_o)
            add(_spec(
                method_id=f"local-structural-{start}-analog__pO-{p_o}",
                family="learned_local_analog",
                role="optimistic probability-only local Stage-2 replacement plus global O",
                order=order, p_q=start_selection.p_q, p_c=start_selection.p_c,
                p_o=p_o,
                phase_depth=structural_depth + p_o, phase_variant=f"{structural_phase}->O",
                mixer=mixer_name + " exact analog adjacency; learned-global",
                resource=_ledger(None, finite_rank=False, generator_proxy_circuit=proxy_total),
                parameter_count=2 * ((q_selection.p_q if start == "Q" else c_selection.p_c)
                                     + structural_depth + p_o),
                evaluation_plan=(2_400, 2_400, 4_800), finite_control=False,
                initial_reference=f"learned {start}=0 preparation",
            ))

            # Adjacent-collapse local variants, joint and separate phases.
            phase_base = "CO" if start == "Q" else "QO"
            collapse_depth = structural_depth + p_o
            for variant in ("joint", "separate"):
                phase_name = f"{phase_base}_{variant}"
                for sweeps in LOCAL_SWEEPS:
                    native = warm_ru(prior_native, phase_ru(phase_name), sweeps * sweep_cost,
                                     collapse_depth)
                    pauli = warm_ru(prior_pauli, phase_ru(phase_name, pauli_c=True),
                                    sweeps * sweep_cost, collapse_depth)
                    proxy = warm_ru(prior_native, phase_ru(phase_name), proxy_cost, collapse_depth)
                    add(_spec(
                        method_id=f"local-collapse-{start}__{variant}__pO-{p_o}__s-{sweeps}",
                        family="learned_local_collapse", role="finite local adjacent-collapse control",
                        order=order, p_q=start_selection.p_q, p_c=start_selection.p_c,
                        p_o=p_o,
                        phase_depth=collapse_depth, phase_variant=phase_name, mixer=mixer_name,
                        sweeps=sweeps, resource=_ledger(native, pauli, generator_proxy_circuit=proxy),
                        parameter_count=2 * (q_selection.p_q if start == "Q" else c_selection.p_c)
                        + (2 if variant == "joint" else 3) * collapse_depth,
                        evaluation_plan=(4_800, 4_800), finite_control=True,
                        initial_reference=f"learned {start}=0 preparation",
                    ))
                proxy = warm_ru(prior_native, phase_ru(phase_name), proxy_cost, collapse_depth)
                add(_spec(
                    method_id=f"local-collapse-{start}-analog__{variant}__pO-{p_o}",
                    family="learned_local_collapse_analog",
                    role="optimistic probability-only local adjacent-collapse sensitivity",
                    order=order, p_q=start_selection.p_q, p_c=start_selection.p_c,
                    p_o=p_o,
                    phase_depth=collapse_depth, phase_variant=phase_name,
                    mixer=mixer_name + " exact analog adjacency",
                    resource=_ledger(None, finite_rank=False, generator_proxy_circuit=proxy),
                    parameter_count=2 * (q_selection.p_q if start == "Q" else c_selection.p_c)
                    + (2 if variant == "joint" else 3) * collapse_depth,
                    evaluation_plan=(4_800, 4_800), finite_control=False,
                    initial_reference=f"learned {start}=0 preparation",
                ))

    # Direct/fixed global RU-matched ladders.  Each family gets its own exact
    # integer cap and the phase-depth-matched points even when those exceed a
    # particular comparator's RU.
    global_families = (
        ("direct-block-joint", "direct_block_xy", "QCO_joint", False),
        ("direct-block-separate", "direct_block_xy", "QCO_separate", False),
        ("fixed-native-joint", "fixed_native_global", "QCO_joint", True),
        ("fixed-native-separate", "fixed_native_global", "QCO_separate", True),
    )
    for label, family, phase_name, fixed in global_families:
        native_phase = phase_ru(phase_name)
        pauli_phase = phase_ru(phase_name, pauli_c=True)
        if fixed:
            base = P_RU + T_O_RU
            slope = native_phase + 2 * P_RU + S_RU
        else:
            base = P_RU + T_O_RU
            slope = native_phase + X_RU
        depths, matched = _affine_depths(full_targets, base, slope, sorted(phase_matched_depths))
        for depth in depths:
            native = fixed_ru(native_phase, depth) if fixed else block_ru(native_phase, depth)
            pauli = fixed_ru(pauli_phase, depth) if fixed else block_ru(pauli_phase, depth)
            penalties = (1.0,) if fixed else PENALTY_MULTIPLIERS
            for penalty in penalties:
                add(_spec(
                    method_id=f"{label}__d-{depth}__lambda-{penalty:g}", family=family,
                    role="prescribed-reference global control" if fixed else "strong direct block-XY control",
                    phase_depth=depth, phase_variant=phase_name,
                    mixer="prescribed native rank-one" if fixed else "complete block-XY",
                    resource=_ledger(native, pauli),
                    parameter_count=(2 if phase_name.endswith("joint") else 4) * depth,
                    evaluation_plan=(9_600,), penalty=penalty,
                    finite_control=(penalty == 1.0),
                    matched_against=matched.get(depth, ()),
                    initial_reference="prescribed native-uniform shell",
                ))

    # Direct local controls: target-blind 10-X basis preparation, family- and
    # sweep-specific RU caps, with analog rows phase-matched but unranked.
    direct_local_defs = (
        ("capacity", "CO", CAPACITY_SWEEP_RU, CAPACITY_PROXY_RU),
        ("conflict", "QO", CONFLICT_SWEEP_RU, CONFLICT_PROXY_RU),
        ("exact-final", "O", EXACT_FINAL_SWEEP_RU, EXACT_FINAL_PROXY_RU),
    )
    for label, phase_base, sweep_cost, proxy_cost in direct_local_defs:
        variants = ("joint", "separate") if phase_base != "O" else ("objective",)
        for variant in variants:
            phase_name = phase_base if phase_base == "O" else f"{phase_base}_{variant}"
            for sweeps in LOCAL_SWEEPS:
                native_phase = phase_ru(phase_name)
                pauli_phase = phase_ru(phase_name, pauli_c=True)
                base = BASIS_PREP_RU + T_O_RU
                slope = native_phase + sweeps * sweep_cost
                depths, matched = _affine_depths(full_targets, base, slope, sorted(phase_matched_depths))
                for depth in depths:
                    native = BASIS_PREP_RU + depth * (native_phase + sweeps * sweep_cost)
                    pauli = BASIS_PREP_RU + depth * (pauli_phase + sweeps * sweep_cost)
                    proxy = BASIS_PREP_RU + depth * (native_phase + proxy_cost)
                    add(_spec(
                        method_id=f"direct-local-{label}__{variant}__d-{depth}__s-{sweeps}",
                        family="direct_local", role="target-blind finite direct-local control",
                        phase_depth=depth, phase_variant=phase_name,
                        mixer=f"{label} fixed-order digitized local mixer", sweeps=sweeps,
                        resource=_ledger(native, pauli, generator_proxy_circuit=proxy),
                        parameter_count=(2 if variant in {"joint", "objective"} else 3) * depth,
                        evaluation_plan=(9_600,), finite_control=True,
                        matched_against=matched.get(depth, ()),
                        initial_reference=(
                            "lexicographically first Q=0 basis state; 10-X ledger"
                            if label == "capacity" else
                            "lexicographically first C=0 basis state; 10-X ledger"
                            if label == "conflict" else
                            "lexicographically first Q=C=0 basis state; 10-X ledger"
                        ),
                    ))
            for depth in sorted(phase_matched_depths):
                proxy = BASIS_PREP_RU + depth * (phase_ru(phase_name) + proxy_cost)
                add(_spec(
                    method_id=f"direct-local-{label}-analog__{variant}__d-{depth}",
                    family="direct_local_analog", role="optimistic probability-only direct-local sensitivity",
                    phase_depth=depth, phase_variant=phase_name,
                    mixer=f"exact analog {label} adjacency exponential",
                    resource=_ledger(None, finite_rank=False, generator_proxy_circuit=proxy),
                    parameter_count=(2 if variant in {"joint", "objective"} else 3) * depth,
                    evaluation_plan=(9_600,), finite_control=False,
                    initial_reference=(
                        "lexicographically first Q=0 basis state; 10-X ledger"
                        if label == "capacity" else
                        "lexicographically first C=0 basis state; 10-X ledger"
                        if label == "conflict" else
                        "lexicographically first Q=C=0 basis state; 10-X ledger"
                    ),
                ))

    # Static and optimized ideal references never enter finite-RU rankings.
    for sector, support in (("native", 759_375), ("quota", 4_530), ("conflict", 6_570), ("final", 390)):
        add(MethodSpec(
            f"ideal-uniform-{sector}", "ideal_static", "probability reference only", None,
            None, None, None, None, "none", f"uniform {sector}", None, 1.0, 0,
            (), (), (), ResourceLedger(None, None, None, None, None, False), False,
        ))
    for p_o in OBJECTIVE_DEPTHS:
        add(_spec(
            method_id=f"ideal-uniform-final-global-O__pO-{p_o}", family="ideal_optimized",
            role="unknown-preparation optimistic ceiling", p_o=p_o, phase_depth=p_o,
            phase_variant="O", mixer="rank-one about ideal uniform-final reference",
            resource=_ledger(None, finite_rank=False), parameter_count=2 * p_o,
            evaluation_plan=(9_600,), finite_control=False,
            initial_reference="ideal uniform-final state; preparation cost unknown",
            training_loss="E[Otilde]",
        ))

    primary_control_families = {
        "adjacent_collapse", "standalone_precursor", "warm_block_xy",
        "learned_local", "learned_local_collapse", "fixed_native_global",
        "direct_block_xy", "direct_local",
    }
    sensitivity_families = {
        "full_penalty_sensitivity", "full_objective_only_sensitivity",
        "learned_local_analog", "learned_local_collapse_analog",
        "direct_local_analog",
    }
    serialized = []
    for spec in specs:
        row = {**asdict(spec), "resource": asdict(spec.resource)}
        if spec.family == "full_three_stage":
            row["gate_category"] = "candidate"
        elif spec.family == "direct_block_xy" and spec.penalty_multiplier != 1.0:
            row["gate_category"] = "sensitivity"
        elif spec.family in primary_control_families:
            row["gate_category"] = "finite_control"
        elif spec.family in sensitivity_families:
            row["gate_category"] = "sensitivity"
        elif spec.family == "paired_precursor_snapshot":
            row["gate_category"] = "diagnostic"
        else:
            row["gate_category"] = "ceiling_or_reference"
        row["configuration_id"] = spec.method_id
        row["method_group_id"] = _method_group_id(spec.method_id)
        row["development_selection_eligible"] = row["gate_category"] in {
            "candidate", "finite_control"
        }
        serialized.append(row)
    primary_rows = [
        row for row in serialized
        if row["gate_category"] in {"candidate", "finite_control"}
    ]
    for candidate in serialized:
        if candidate["gate_category"] != "sensitivity":
            candidate["sensitivity_parent_configuration_ids"] = ()
            continue
        parents = set(map(str, candidate.get("matched_against", ())))
        family = candidate["family"]
        if family == "direct_block_xy":
            parents.update(
                str(row["configuration_id"])
                for row in primary_rows
                if row["family"] == "direct_block_xy"
                and row["phase_variant"] == candidate["phase_variant"]
                and row["phase_depth"] == candidate["phase_depth"]
            )
        elif family == "direct_local_analog":
            parents.update(
                str(row["configuration_id"])
                for row in primary_rows
                if row["family"] == "direct_local"
                and row["phase_variant"] == candidate["phase_variant"]
                and row["phase_depth"] == candidate["phase_depth"]
            )
        elif family == "learned_local_analog":
            parents.update(
                str(row["configuration_id"])
                for row in primary_rows
                if row["family"] == "learned_local"
                and row["order"] == candidate["order"]
                and row["p_o"] == candidate["p_o"]
            )
        elif family == "learned_local_collapse_analog":
            parents.update(
                str(row["configuration_id"])
                for row in primary_rows
                if row["family"] == "learned_local_collapse"
                and row["order"] == candidate["order"]
                and row["p_o"] == candidate["p_o"]
                and row["phase_variant"] == candidate["phase_variant"]
            )
        candidate["sensitivity_parent_configuration_ids"] = tuple(sorted(parents))
        if not parents:
            raise AssertionError(
                f"sensitivity lacks an explicit primary parent: {candidate['configuration_id']}"
            )
    ids = [row["method_id"] for row in serialized]
    if len(ids) != len(set(ids)):
        raise AssertionError("method identifiers are not unique")
    finite = [row for row in serialized if row["primary_finite_rank_eligible"]]
    if any(sum(row["forward_evaluations"]) != 9_600 for row in finite):
        raise AssertionError("finite control has unequal primary optimizer budget")
    if any(row["resource"]["finite_rank"] is not True for row in finite):
        raise AssertionError("a primary finite-ranked row lacks a finite RU ledger")
    if not any(row["family"] == "adjacent_collapse" for row in finite):
        raise AssertionError("adjacent-collapse matrix is absent")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in serialized:
        grouped.setdefault(str(row["method_group_id"]), []).append(row)
    for group_id, rows in grouped.items():
        if len({str(row["family"]) for row in rows}) != 1:
            raise AssertionError(f"method-group family collision: {group_id}")

    body = {
        "schema": METHOD_MATRIX_SCHEMA,
        "primary_order": token.primary_order,
        "selected_orders": {
            "Q-C": asdict(token.q_then_c),
            "C-Q": asdict(token.c_then_q),
        },
        "objective_depths": OBJECTIVE_DEPTHS,
        "restart_seeds": EXPECTED_RESTART_SEEDS,
        "primary_forward_evaluations_per_configuration_restart": 9_600,
        "budget_sensitivity_forward_evaluations": 19_200,
        "development_selection": {
            "tables": DEVELOPMENT_SEEDS,
            "metric": "geometric mean integer best-two RTS99 x primary complete-replay RU",
            "ties": "lower primary RU, lower depth, canonical configuration ID",
            "restart_selection_before_target_lock": "lowest expected training loss; initialization-seed tie",
            "double_budget_wave": (
                "after development selection, rerun selected direct block-XY, fixed-native global, "
                "and strongest finite direct-local configurations at 19,200 evaluations"
            ),
            "selection_unit": "exactly one configuration_id per method_group_id",
        },
        "held_out_evaluation": {
            "tables": HELD_OUT_SEEDS,
            "configuration_frozen_from_development": True,
            "held_out_target_may_change_configuration": False,
        },
        "precision_replay": {
            "saved_complex64_angles_only": True,
            "complex128_reoptimization": False,
            "complex128_configuration_reselection": False,
            "required_rows": (
                "every selected held-out method, primary full, strongest adjacent collapse, "
                "and every row entering strongest finite-control envelope"
            ),
        },
        "optimizer_contract": {
            "optimizer": "Adam",
            "learning_rate": 0.035,
            "forward_evaluation_definition": "each training loss call, including exactly one post-update endpoint per optimized call",
            "angle_initialization": "independent uniform [-pi,pi]",
            "stream_derivation": "SHA-256(objective_seed,method_id,configuration_id,restart_seed,parameter_block)",
            "complex64_max_norm_error": 2e-5,
            "complex128_saved_angle_max_norm_error": 1e-10,
            "restart_checkpoint_selection": "minimum expected training loss; lower initialization seed on exact tie",
            "target_metrics_available_during_optimization": False,
        },
        "target_boundary": {
            "optimizer_imports_target_artifact": False,
            "optimizer_allowed_target_operations": (),
            "target_lock_after_complete_optimizer_seal": True,
            "target_lock_sets": "stable (raw O_s, canonical native-shell index) sort; exact-cardinality best 1, 2 and 8",
            "development_locks": "development selection only",
            "held_out_locks": "evaluation only; cannot change methods, restarts, depths, penalties or reporting",
        },
        "direct_local_freeze": {
            "canonical_basis_starts": CANONICAL_BASIS_STARTS,
            "basis_preparation": "ten deterministic X gates; 10 RU",
            "capacity_template_order": "C5 edges (0,1),(1,2),(2,3),(3,4),(0,4), then lexicographic label pairs",
            "conflict_template_order": "blocks 0..4, then lexicographic label pairs",
            "final_template_order": "C5 edges (0,1),(1,2),(2,3),(3,4),(0,4), then lexicographic label pairs",
            "product_formula_order": "repeat the same frozen 75-template order for each of 1,2,4 sweeps",
        },
        "classical_post_lock_controls": {
            "exact_enumeration": {
                "states": 390,
                "canonical_order": True,
                "records_objective_calls_and_wall_time": True,
            },
            "exact_integer_solver": {
                "variables": "30 x bits plus McCormick y auxiliaries",
                "constraints": "hard demand, quota, C5; y<=xi, y<=xj, y>=xi+xj-1",
                "required_status": "proved optimum, zero gap, objective cross-check with enumeration",
            },
            "connected_switch_steepest_descent": {
                "starts": "lexicographic first plus first 15 PCG64-permuted states; seed 2026081100",
                "move": "largest strict decrease over every neighbour; canonical tie",
                "target_stop": False,
            },
            "tabu": {
                "tenure": 7,
                "calls_per_restart": 9_600,
                "aspiration": "strict incumbent improvement only",
                "target_stop": False,
            },
            "simulated_annealing": {
                "calls_per_restart": 9_600,
                "schedule": "geometric 0.1 at proposal 1 to 0.001 at proposal 9600",
                "proposal": "uniform graph neighbour; Metropolis",
                "target_stop": False,
            },
        },
        "paper_facing_success_gate": {
            "held_out_tables": 8,
            "primary_metric": "unconditional exact-cardinality best-two RTS99 x complete replay primary RU",
            "strongest_finite_control_advantage_geomean_at_least": 1.20,
            "strongest_finite_control_paired_wins_at_least": 7,
            "strongest_adjacent_collapse_advantage_geomean_at_least": 1.20,
            "strongest_adjacent_collapse_paired_wins_at_least": 7,
            "standalone_precursor_cost_over_full_geomean_strictly_greater_than": 1.0,
            "standalone_precursor_paired_wins_at_least": 5,
            "all_direct_ladders_complete": True,
            "required_robustness": (
                "saved-angle complex128", "Pauli-C", "T_all terminal",
                "1/2/4-sweep local", "19,200-evaluation direct", "lambda 1/2,1,2"
            ),
            "any_failed_clause_is_negative": True,
        },
        "method_count": len(serialized),
        "primary_finite_rank_eligible_count": len(finite),
        "methods": serialized,
    }
    return {**body, "matrix_sha256": _sha256(body)}


def build_post_development_task_manifest(
    token: _ActivatedStructuralHandoff,
    matrix: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Freeze held-out and 19,200-evaluation tasks after development selection.

    This is a planner only.  It consumes a separately produced, post-lock
    development-selection artifact and never reads target identities or target
    probabilities itself.
    """

    token = _require_token(token)
    _require(matrix.get("schema") == METHOD_MATRIX_SCHEMA, "method matrix schema mismatch")
    _require(selection.get("schema") == "nonredundant-bcst-stage3-development-selection-v1",
             "development-selection schema mismatch")
    _require(selection.get("structural_handoff_id") == token.structural_handoff_id,
             "development selection belongs to another structural handoff")
    _require(tuple(selection.get("development_tables", ())) == DEVELOPMENT_SEEDS,
             "development selection does not use exactly the four frozen tables")
    _require(selection.get("all_development_optimizer_rows_sealed") is True,
             "development optimizers are not completely sealed")
    _require(selection.get("all_development_target_locks_complete") is True,
             "development target locks are incomplete")
    _require(selection.get("held_out_target_constructed") is False,
             "held-out target information existed during development selection")
    _require(
        selection.get("selection_metric")
        == "geometric mean integer best-two RTS99 x primary complete-replay RU",
        "development selection metric mismatch",
    )
    _require(
        selection.get("tie_breaks")
        == "lower primary RU, lower depth, canonical configuration ID",
        "development selection tie rule mismatch",
    )

    rows_raw = matrix.get("methods")
    _require(isinstance(rows_raw, list), "method matrix rows are absent")
    rows = [row for row in rows_raw if isinstance(row, dict)]
    by_configuration = {str(row.get("configuration_id")): row for row in rows}
    _require(len(by_configuration) == len(rows), "configuration IDs are missing or duplicated")
    eligible_groups = {
        str(row["method_group_id"])
        for row in rows
        if row.get("development_selection_eligible") is True
    }
    raw_choices = selection.get("selected_configurations")
    _require(isinstance(raw_choices, list), "development selected configurations are absent")
    choices: dict[str, str] = {}
    for item in raw_choices:
        _require(isinstance(item, dict), "invalid development selection row")
        group = str(item.get("method_group_id", ""))
        configuration = str(item.get("configuration_id", ""))
        _require(group and group not in choices, "missing or duplicate method-group selection")
        _require(configuration in by_configuration, "selected configuration is not in the frozen matrix")
        row = by_configuration[configuration]
        _require(row.get("method_group_id") == group, "configuration selected for the wrong method group")
        _require(row.get("development_selection_eligible") is True,
                 "a sensitivity/diagnostic/reference was selected as a primary method")
        choices[group] = configuration
    _require(set(choices) == eligible_groups,
             "development selection is not exactly one configuration per eligible method group")

    # Recompute Section-9 selection from every sealed development cost rather
    # than trusting the declared winners.  Cost rows contain no target identity.
    raw_costs = selection.get("development_configuration_costs")
    _require(isinstance(raw_costs, list), "development configuration costs are absent")
    cost_map: dict[tuple[str, int], float] = {}
    eligible_configurations = {
        str(row["configuration_id"])
        for row in rows
        if row.get("development_selection_eligible") is True
    }
    for item in raw_costs:
        _require(isinstance(item, dict), "invalid development cost row")
        configuration = str(item.get("configuration_id", ""))
        objective_seed = item.get("objective_seed")
        cost = item.get("integer_best_two_RTS99_x_primary_RU")
        _require(configuration in eligible_configurations and objective_seed in DEVELOPMENT_SEEDS,
                 "development cost row references an ineligible configuration/table")
        key = (configuration, int(objective_seed))
        _require(key not in cost_map, "duplicate development configuration cost")
        _require(isinstance(cost, (int, float)) and math.isfinite(float(cost)) and float(cost) > 0,
                 "development configuration cost is nonfinite or nonpositive")
        cost_map[key] = float(cost)
    expected_cost_keys = {
        (configuration, seed)
        for configuration in eligible_configurations
        for seed in DEVELOPMENT_SEEDS
    }
    _require(set(cost_map) == expected_cost_keys,
             "development configuration-cost matrix is incomplete or contains extras")
    recomputed: dict[str, str] = {}
    for group in eligible_groups:
        candidates = [row for row in rows if row.get("method_group_id") == group
                      and row.get("development_selection_eligible") is True]
        def selection_key(row: dict[str, object]) -> tuple[float, int, int, str]:
            configuration = str(row["configuration_id"])
            costs = [cost_map[(configuration, seed)] for seed in DEVELOPMENT_SEEDS]
            geomean = float(np.exp(np.mean(np.log(np.asarray(costs, dtype=np.float64)))))
            resource = row.get("resource")
            primary_ru = int(resource["primary"]) if isinstance(resource, dict) else 2**63 - 1
            depth = int(row.get("phase_depth") or 0)
            return geomean, primary_ru, depth, configuration
        recomputed[group] = str(min(candidates, key=selection_key)["configuration_id"])
    _require(choices == recomputed,
             "declared development selections differ from recomputed four-table geometric-cost rule")

    selected_rows = [by_configuration[choices[group]] for group in sorted(choices)]
    selected_ids = {str(row["configuration_id"]) for row in selected_rows}
    mapped_sensitivities = [
        candidate for candidate in rows
        if candidate.get("gate_category") == "sensitivity"
        and bool(
            set(map(str, candidate.get("sensitivity_parent_configuration_ids", ())))
            & selected_ids
        )
    ]
    ideal_heldout_rows = [row for row in rows if row.get("family") == "ideal_optimized"]
    frozen_heldout_rows = selected_rows + sorted(
        mapped_sensitivities, key=lambda row: str(row["configuration_id"])
    ) + sorted(ideal_heldout_rows, key=lambda row: str(row["configuration_id"]))
    held_out_tasks: list[dict[str, object]] = []
    for objective_seed in HELD_OUT_SEEDS:
        for row in frozen_heldout_rows:
            for restart_seed in EXPECTED_RESTART_SEEDS:
                held_out_tasks.append({
                    "task_id": f"heldout__o-{objective_seed}__{row['configuration_id']}__r-{restart_seed}",
                    "wave": "held_out_primary",
                    "objective_seed": objective_seed,
                    "method_group_id": row["method_group_id"],
                    "configuration_id": row["configuration_id"],
                    "restart_seed": restart_seed,
                    "forward_evaluations": row["forward_evaluations"],
                    "gradient_updates": row["gradient_updates"],
                    "endpoint_evaluations": row["endpoint_evaluations"],
                    "frozen_from_development": True,
                    "sensitivity_not_primary": row.get("gate_category") == "sensitivity",
                    "held_out_target_available_to_worker": False,
                })

    sensitivity_families = {"direct_block_xy", "fixed_native_global", "direct_local"}
    sensitivity_rows = [row for row in selected_rows if row.get("family") in sensitivity_families]
    double_budget_tasks: list[dict[str, object]] = []
    for objective_seed in OBJECTIVE_SEEDS:
        for row in sensitivity_rows:
            for restart_seed in EXPECTED_RESTART_SEEDS:
                double_budget_tasks.append({
                    "task_id": f"budget19200__o-{objective_seed}__{row['configuration_id']}__r-{restart_seed}",
                    "wave": "development_selected_direct_19200_sensitivity",
                    "objective_seed": objective_seed,
                    "method_group_id": row["method_group_id"],
                    "configuration_id": row["configuration_id"],
                    "restart_seed": restart_seed,
                    "forward_evaluations": (19_200,),
                    "gradient_updates": (19_199,),
                    "endpoint_evaluations": (1,),
                    "replaces_primary_row": False,
                    "target_available_to_worker": False,
                })

    body = {
        "schema": POST_DEVELOPMENT_SCHEMA,
        "stage_a_manifest_id": token.stage_a_manifest_id,
        "structural_handoff_id": token.structural_handoff_id,
        "matrix_sha256": matrix.get("matrix_sha256"),
        "selected_configurations": [
            {"method_group_id": group, "configuration_id": choices[group]}
            for group in sorted(choices)
        ],
        "mapped_sensitivity_configurations": [
            str(row["configuration_id"]) for row in sorted(
                mapped_sensitivities, key=lambda row: str(row["configuration_id"])
            )
        ],
        "held_out_ideal_ceiling_configurations": [
            str(row["configuration_id"]) for row in sorted(
                ideal_heldout_rows, key=lambda row: str(row["configuration_id"])
            )
        ],
        "held_out_tasks": held_out_tasks,
        "double_budget_tasks": double_budget_tasks,
        "held_out_task_count": len(held_out_tasks),
        "double_budget_task_count": len(double_budget_tasks),
        "frozen_heldout_configuration_count": len(frozen_heldout_rows),
        "all_eight_held_out_tables_retained": True,
        "double_budget_is_sensitivity_not_replacement": True,
        "target_identity_exposed_to_optimizer_tasks": False,
    }
    return {**body, "manifest_sha256": _sha256(body)}


def build_campaign_manifest(
    token: _ActivatedStructuralHandoff,
    coefficients: Mapping[str, object],
    matrix: Mapping[str, object],
) -> dict[str, object]:
    """Bind frozen tables and controls; still performs no objective/state call."""

    token = _require_token(token)
    _require(coefficients.get("schema") == COEFFICIENT_SCHEMA, "coefficient manifest schema mismatch")
    _require(matrix.get("schema") == METHOD_MATRIX_SCHEMA, "method matrix schema mismatch")
    _require(coefficients.get("structural_handoff_id") == token.structural_handoff_id,
             "coefficient manifest belongs to another handoff")
    methods = matrix.get("methods")
    _require(isinstance(methods, list) and len(methods) == matrix.get("method_count"),
             "method matrix is incomplete")
    body = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "FROZEN_CONDITIONALLY_LAUNCHABLE_NO_OBJECTIVE_OUTCOME",
        "stage_a_manifest_id": token.stage_a_manifest_id,
        "structural_handoff_id": token.structural_handoff_id,
        "coefficient_manifest_sha256": coefficients.get("manifest_sha256"),
        "method_matrix_sha256": matrix.get("matrix_sha256"),
        "development_seeds": DEVELOPMENT_SEEDS,
        "held_out_seeds": HELD_OUT_SEEDS,
        "restart_seeds": EXPECTED_RESTART_SEEDS,
        "execution_waves": (
            "development optimization and sealing",
            "development target locks and configuration selection",
            "development-selected 19,200-evaluation sensitivities",
            "held-out optimization and sealing",
            "held-out target locks and read-only evaluation",
            "saved-angle complex128 replay",
            "post-lock classical controls and unfiltered aggregation",
        ),
        "resume_contract": "atomic configuration/restart rows; never overwrite a complete row",
        "required_worker_capabilities": (
            "exact checkpointed/adjoint statevector gradients at every RU-matched depth",
            "block-XY, learned/fixed rank-one, digitized local and analog adjacency mixers",
            "one-way optimizer seal -> target lock -> read-only evaluator separation",
            "classical exact/ILP/local/tabu/annealing controls",
        ),
        # Reaching this builder already requires the private capability issued
        # only after Stage A and the final target-free B--E handoff pass.  The
        # workers still start no outcome automatically; this flag says the
        # frozen campaign is internally executable rather than contradicting
        # its implemented worker contract.
        "launchable": True,
        "both_target_free_activation_gates_validated": True,
        "worker_implementation_contract": "nonredundant-bcst-stage3-workers-v1",
        "implemented_worker_capabilities": {
            "checkpointed_all_literal_ru_matched_depths": True,
            "digitized_and_analog_local_mixers": True,
            "four_full_guard_and_exact_final_mixers": True,
            "process_separated_optimizer_and_postlock": True,
            "development_heldout_and_19200_waves": True,
            "saved_angle_complex128_replay": True,
            "exact_and_heuristic_classical_controls": True,
        },
        "implementation_blockers": (),
        "automatic_outcome_launch": False,
        "operator_scheduling_required": True,
        "objective_evaluated_on_state": False,
        "target_constructed": False,
        "variational_outcome_run": False,
    }
    return {**body, "campaign_manifest_sha256": _sha256(body)}


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _audited_seed_table() -> dict[str, object]:
    audit = _load_object(HERE / "structural_audit.json")
    value = audit.get("frozen_objective")
    if not isinstance(value, dict):
        raise ActivationError("structural audit lacks the independently frozen seed table")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("readiness", "freeze"), required=True)
    parser.add_argument("--stage-a-aggregate", type=Path, required=True)
    parser.add_argument("--structural-handoff", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_a_payload = _load_object(args.stage_a_aggregate)
    stage_a = validate_stage_a_aggregate(stage_a_payload)
    if args.structural_handoff is None:
        if args.stage == "freeze":
            raise ActivationError("freeze requires the final target-free structural handoff")
        print(json.dumps({
            "status": "STAGE_A_PROCEED_WAITING_FOR_FINAL_STRUCTURAL_HANDOFF",
            "stage_a_manifest_id": stage_a.manifest_id,
            "objective_table_materialized": False,
            "target_constructed": False,
        }, indent=2))
        return
    handoff_payload = _load_object(args.structural_handoff)
    token = activate_structural_handoff(stage_a_payload, handoff_payload)
    if args.stage == "readiness":
        print(json.dumps({
            "status": "READY_TO_FREEZE_DORMANT_OBJECTIVE_CAMPAIGN",
            "primary_order": token.primary_order,
            "objective_table_materialized": False,
            "target_constructed": False,
        }, indent=2))
        return
    if args.campaign_root is None:
        raise ActivationError("freeze requires --campaign-root")
    coefficients = coefficient_manifest(token, _audited_seed_table())
    matrix = build_method_matrix(token)
    campaign = build_campaign_manifest(token, coefficients, matrix)
    destinations = {
        "coefficients": args.campaign_root / "frozen_objective_coefficients.json",
        "method_matrix": args.campaign_root / "frozen_objective_method_matrix.json",
        "campaign": args.campaign_root / "frozen_objective_campaign.json",
    }
    for key, path in destinations.items():
        payload = {"coefficients": coefficients, "method_matrix": matrix, "campaign": campaign}[key]
        if path.exists():
            raise ActivationError(f"refusing to overwrite frozen artifact {path}")
        _atomic_write_json(path, payload)
    print(json.dumps({
        "status": campaign["status"],
        "outputs": {key: str(path) for key, path in destinations.items()},
        "method_count": matrix["method_count"],
        "objective_evaluated_on_state": False,
        "target_constructed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
