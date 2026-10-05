"""Conditional target-free structural worker for the nonredundant BCST case.

This executable starts only from a complete, passing Stage-A aggregate.  It
implements the remaining target-free structural campaign:

* Stage B: the complete 2 x 3 x 7 learned-projector grid;
* Stage C: selected-checkpoint warm, prescribed-reference, and 1/2/4-sweep
  local controls, plus the unranked analog-adjacency sensitivity;
* Stage D: every phase-matched and comparator-specific RU-matched direct row;
* Stage E: saved-angle complex128 replay and the final structural handoff.

No objective coefficients, soft-cost table, optimum, or target set are present
in this module.  Valid work is checkpointed after every completed method/shard.
The existing frozen manifest remains authoritative; this worker does not edit
it or the Stage-A artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

import target_free_structural_screen as engine


STRUCTURAL_WORKER_SCHEMA = "nonredundant-bcst-post-stage-a-worker-v1"
STAGE_B_SCHEMA = "nonredundant-bcst-structural-stage-b-v1"
STAGE_C_SCHEMA = "nonredundant-bcst-structural-stage-c-v1"
STAGE_D_SCHEMA = "nonredundant-bcst-structural-stage-d-v1"
STRUCTURAL_HANDOFF_SCHEMA = "nonredundant-bcst-final-structural-handoff-v1"

EXPECTED_STAGE_A_SCHEMA = "nonredundant-bcst-target-free-structural-screen-v2"
EXPECTED_RESTART_SEEDS = tuple(range(2_026_081_100, 2_026_081_116))
FINITE_CONTROL_FAMILIES = {
    "coarse",
    "warm",
    "local",
    "direct_joint",
    "direct_separate",
    "fixed_native_joint",
    "fixed_native_separate",
}
COMPARATOR_FAMILIES = (
    "direct_joint",
    "direct_separate",
    "fixed_native_joint",
    "fixed_native_separate",
)


class StructuralActivationError(RuntimeError):
    """The conditional structural campaign failed closed before execution."""


class KrylovConvergenceError(RuntimeError):
    """The analog exponential did not meet its declared numerical tolerance."""


@dataclass(frozen=True)
class StructuralSelection:
    first: str
    second: str
    depth_first: int
    depth_second: int
    algorithm: str
    primary_ru: int

    @property
    def order(self) -> str:
        return f"{self.first}-{self.second}"

    @property
    def phase_depth(self) -> int:
        return self.depth_first + self.depth_second


@dataclass(frozen=True)
class AnalogConfig:
    tolerance_complex64: float = 2e-6
    max_krylov_dimension: int = 256
    check_interval: int = 16

    def validate(self) -> None:
        if not (0 < self.tolerance_complex64 < 1e-3):
            raise ValueError("invalid analog Krylov tolerance")
        if self.max_krylov_dimension < 32:
            raise ValueError("analog Krylov dimension is too small")
        if self.check_interval < 4 or self.max_krylov_dimension % self.check_interval:
            raise ValueError("Krylov check interval must divide the maximum dimension")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralActivationError(message)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StructuralActivationError(f"{path} is not a JSON object")
    return value


def _target_guard_is_false(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "objective_table_loaded",
            "target_constructed",
            "target_probability_computed",
        }
        and all(item is False for item in value.values())
    )


def validate_passing_stage_a(
    payload: Mapping[str, object], manifest: Mapping[str, object]
) -> None:
    """Fail closed unless this is the live complete valid Stage-A aggregate."""

    _require(payload.get("schema") == EXPECTED_STAGE_A_SCHEMA, "Stage-A schema mismatch")
    _require(payload.get("profile") == "valid", "Stage A is not valid-profile")
    _require(
        payload.get("execution_stage") == "A_NECESSITY_COMPLETE_16_RESTARTS",
        "Stage A is not the complete 16-restart aggregate",
    )
    _require(payload.get("scientific_decision_allowed") is True, "Stage A is excluded")
    _require(payload.get("excluded_smoke") is False, "smoke Stage A cannot activate")
    _require(payload.get("protocol_deviation") is False, "Stage A has a protocol deviation")
    _require(payload.get("stage_a_shards_complete") is True, "Stage-A shards are incomplete")
    _require(payload.get("early_promotion_used") is False, "early promotion is forbidden")
    _require(_target_guard_is_false(payload.get("target_guard")), "Stage A touched target data")
    _require(
        payload.get("manifest_sha256") == manifest.get("manifest_sha256"),
        "Stage A and frozen manifest identifiers differ",
    )
    gate = payload.get("gate")
    _require(isinstance(gate, dict), "Stage-A gate is absent")
    _require(
        gate.get("status") == "PROCEED_TO_EXPENSIVE_STAGES" and gate.get("pass") is True,
        "Stage A did not pass",
    )
    checks = gate.get("checks")
    _require(isinstance(checks, dict) and checks and all(x is True for x in checks.values()),
             "a Stage-A check failed")
    rows = payload.get("methods")
    _require(isinstance(rows, list) and len(rows) == 6, "Stage A must contain six coarse rows")
    expected = {
        f"Coarse-only-{kind}-p{depth}"
        for kind in ("Q", "C")
        for depth in engine.valid_config().depths
    }
    _require(
        {row.get("algorithm") for row in rows if isinstance(row, dict)} == expected,
        "Stage-A coarse grid is incomplete",
    )
    for row in rows:
        _require(isinstance(row, dict), "invalid Stage-A row")
        _require(tuple(row.get("initialization_seeds", ())) == EXPECTED_RESTART_SEEDS,
                 "Stage-A seed order mismatch")
        _require(row.get("aggregate_pipeline_evaluations_per_restart") == 4_800,
                 "Stage-A budget mismatch")
        costs = row.get("feasibility_RTS99_x_primary_RU")
        _require(
            isinstance(costs, list)
            and len(costs) == 16
            and all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > 0
                for value in costs
            ),
            "Stage-A costs are incomplete or invalid",
        )


def _canonical_pipeline_stream(first: str, p1: int, p2: int) -> int:
    """The Stage-B stream that defines a selected row's Stage-1 checkpoint."""

    task = {
        "family": "full_lp",
        "first": first,
        "depth_first": p1,
        "depth_second": p2,
        "trotter_sweeps": None,
    }
    return engine._task_stream(task)


def normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    """Expose order/local metadata required by the dormant objective launcher."""

    result = dict(row)
    spec = result.get("replay_spec")
    if not isinstance(spec, dict):
        spec = {}
    family = str(result.get("family"))
    first: str | None = None
    second: str | None = None
    if family in {"coarse", "stage1"}:
        value = spec.get("constraint")
        if value in {"Q", "C"}:
            first = str(value)
        else:
            name = str(result.get("algorithm", ""))
            if "-Q-" in name:
                first = "Q"
            elif "-C-" in name:
                first = "C"
    elif spec.get("kind") == "pipeline":
        first = str(spec.get("first"))
        second = str(spec.get("second"))
    result["first"] = first
    result["second"] = second
    if family in {"local", "local_analog"}:
        result["local_mixer"] = spec.get("local_mixer")
        if family == "local":
            result["trotter_sweeps"] = int(spec.get("trotter_sweeps", 0))
        else:
            result["finite_RU_rank"] = False
    return result


def _finite_positive(values: object, length: int = 16) -> bool:
    if not isinstance(values, list) or len(values) != length:
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(array)) and np.all(array > 0))


def _row_selection(row: Mapping[str, object]) -> StructuralSelection:
    first = str(row.get("first"))
    _require(first in {"Q", "C"}, "selected LP row lacks its first constraint")
    second = "C" if first == "Q" else "Q"
    ledger = row.get("RU")
    _require(isinstance(ledger, dict) and isinstance(ledger.get("primary"), int),
             "selected LP row lacks primary RU")
    return StructuralSelection(
        first=first,
        second=second,
        depth_first=int(row["depth_first"]),
        depth_second=int(row["depth_second"]),
        algorithm=str(row["algorithm"]),
        primary_ru=int(ledger["primary"]),
    )


def select_both_orders(
    full_rows: Sequence[dict[str, object]],
) -> tuple[StructuralSelection, StructuralSelection, StructuralSelection]:
    q_rows = [row for row in full_rows if row.get("first") == "Q"]
    c_rows = [row for row in full_rows if row.get("first") == "C"]
    _require(len(q_rows) == 21 and len(c_rows) == 21, "full LP grid is incomplete")
    q_choice = _row_selection(engine.select_full_lp_by_mass_ru(q_rows))
    c_choice = _row_selection(engine.select_full_lp_by_mass_ru(c_rows))
    primary = _row_selection(engine.select_full_lp_by_mass_ru(full_rows))
    return q_choice, c_choice, primary


def _selection_record(selection: StructuralSelection) -> dict[str, object]:
    return {
        "first": selection.first,
        "second": selection.second,
        "depth_first": selection.depth_first,
        "depth_second": selection.depth_second,
        "selected_by_frozen_2pct_mass_then_ru_rule": True,
        "complex64_algorithm": selection.algorithm,
        "primary_RU": selection.primary_ru,
    }


def _stage_b_gate(
    winner: Mapping[str, object],
    full_rows: Sequence[Mapping[str, object]],
    profile: str,
) -> dict[str, object]:
    final = np.asarray(winner.get("final_mass"), dtype=np.float64)
    first = np.asarray(winner.get("stage1_final_mass"), dtype=np.float64)
    norms = np.asarray(winner.get("state_norm"), dtype=np.float64)
    all_rows_valid = True
    for row in full_rows:
        row_final = np.asarray(row.get("final_mass", ()), dtype=np.float64)
        row_norms = np.asarray(row.get("state_norm", ()), dtype=np.float64)
        all_rows_valid &= (
            row_final.shape == (16,)
            and bool(np.all(np.isfinite(row_final)))
            and bool(np.all((row_final >= 0) & (row_final <= 1)))
            and row_norms.shape == (16,)
            and bool(np.all(np.isfinite(row_norms)))
            and float(np.max(np.abs(row_norms - 1.0))) <= 2e-5
            and _finite_positive(row.get("feasibility_RTS99_x_primary_RU"))
        )
    checks = {
        "all_42_full_rows_numerically_valid": bool(
            len(full_rows) == 42 and all_rows_valid
        ),
        "all_16_restarts_present": final.shape == (16,) and first.shape == (16,),
        "median_final_mass_at_least_0p25": final.shape == (16,)
        and bool(np.all(np.isfinite(final)))
        and float(np.median(final)) >= 0.25,
        "median_stage_gain_at_least_2": final.shape == (16,)
        and first.shape == (16,)
        and bool(np.all(np.isfinite(first)))
        and float(np.median(final / np.maximum(first, 1e-30))) >= 2.0,
        "complex64_norm_tolerance": norms.shape == (16,)
        and bool(np.all(np.isfinite(norms)))
        and float(np.max(np.abs(norms - 1.0))) <= 2e-5,
        "finite_positive_sampling_cost": _finite_positive(
            winner.get("feasibility_RTS99_x_primary_RU")
        ),
    }
    passed = profile == "valid" and all(checks.values())
    return {
        "status": "PROCEED_TO_FINITE_CONTROLS" if passed else "STOP_AFTER_FULL_LP_GRID",
        "pass": passed,
        "checks": checks,
    }


def evaluate_control_envelope(
    winner: Mapping[str, object], controls: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Recompute the monotone finite-control gate at an intermediate wave."""

    winner_cost = np.asarray(
        winner.get("feasibility_RTS99_x_primary_RU", ()), dtype=np.float64
    )
    valid = winner_cost.shape == (16,) and np.all(np.isfinite(winner_cost)) and np.all(
        winner_cost > 0
    )
    matrices: list[np.ndarray] = []
    for row in controls:
        values = np.asarray(row.get("feasibility_RTS99_x_primary_RU", ()), dtype=np.float64)
        if values.shape != (16,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            valid = False
        else:
            matrices.append(values)
    if not matrices:
        valid = False
    if valid:
        envelope = np.min(np.stack(matrices), axis=0)
        ratios = envelope / winner_cost
        ratio_gm = engine.geometric_mean(ratios.tolist())
        paired = int(np.sum(ratios > 1.0))
    else:
        ratios = np.full(16, np.nan)
        ratio_gm = math.nan
        paired = 0
    checks = {
        "all_costs_finite_positive": bool(valid),
        "geometric_mean_cost_margin_at_least_1p25": bool(
            math.isfinite(ratio_gm) and ratio_gm >= 1.25
        ),
        "paired_wins_at_least_12_of_16": paired >= 12,
    }
    return {
        "status": "CONTINUE" if all(checks.values()) else "STOP_STRUCTURAL_NO_GO",
        "pass": all(checks.values()),
        "checks": checks,
        "geometric_mean_strongest_control_over_lp_cost": ratio_gm,
        "paired_cost_wins": paired,
        "paired_cost_wins_required": 12,
        "ratios_by_restart": ratios.tolist(),
    }


def frozen_candidate_depths(config: engine.ScreenConfig) -> tuple[int, ...]:
    return tuple(
        sorted(
            set(config.ru_matched_direct_depths)
            | {
                first + second
                for first in config.depths
                for second in config.learned_projector_depths
            }
        )
    )


def active_comparator_depths(
    primary_ru: int,
    selections: Sequence[StructuralSelection],
    config: engine.ScreenConfig,
) -> dict[str, tuple[int, ...]]:
    """Full RU-capped ladders plus both selected phase-depth-matched points."""

    candidates = frozen_candidate_depths(config)
    required = engine.required_comparator_depths(primary_ru, candidates)
    phase_matched = {selection.phase_depth for selection in selections}
    result = {
        family: tuple(sorted(set(required[family]) | phase_matched))
        for family in COMPARATOR_FAMILIES
    }
    for family in COMPARATOR_FAMILIES:
        if not set(required[family]).issubset(result[family]):
            raise AssertionError(f"RU ladder was truncated for {family}")
    return result


def _batch_config(
    base: engine.ScreenConfig, task: Mapping[str, object], *, sweeps: int | None = None
) -> engine.ScreenConfig:
    seeds = tuple(int(value) for value in task["seeds"])
    return replace(
        base,
        restarts=len(seeds),
        initialization_seeds=seeds,
        restart_batch=len(seeds),
        trotter_sweeps=base.trotter_sweeps if sweeps is None else sweeps,
        output=Path(str(task["output"])),
    )


def _new_shard_body(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    rows: Sequence[dict[str, object]],
    traces: Mapping[str, object],
    planned_algorithms: Sequence[str],
) -> dict[str, object]:
    completed = sorted(str(row["algorithm"]) for row in rows)
    body = {
        "schema": f"{STRUCTURAL_WORKER_SCHEMA}-shard-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "task_id": task["task_id"],
        "batch_index": task["batch_index"],
        "seeds": list(task["seeds"]),
        "target_guard": {
            "objective_table_loaded": False,
            "target_constructed": False,
            "target_probability_computed": False,
        },
        "formal_decision_performed": False,
        "planned_algorithms": sorted(planned_algorithms),
        "completed_algorithms": completed,
        "complete": completed == sorted(planned_algorithms),
        "rows": list(rows),
        "traces": dict(traces),
    }
    return engine._json_safe(body)


def _save_shard(
    output: Path,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    rows: Sequence[dict[str, object]],
    traces: Mapping[str, object],
    planned_algorithms: Sequence[str],
) -> dict[str, object]:
    body = _new_shard_body(manifest, task, rows, traces, planned_algorithms)
    payload = {**body, "shard_sha256": engine.sha256_json(body)}
    engine.atomic_write_json(output, payload)
    return payload


def _open_resumable_shard(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    planned_algorithms: Sequence[str],
    *,
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output = Path(str(task["output"]))
    if not output.exists():
        return [], {}
    if not resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = engine._load_and_validate_shard(output, dict(manifest), dict(task))
    _require(
        sorted(payload.get("planned_algorithms", ())) == sorted(planned_algorithms),
        f"resume plan differs for {task['task_id']}",
    )
    rows = payload.get("rows")
    traces = payload.get("traces")
    _require(isinstance(rows, list) and isinstance(traces, dict), "invalid partial shard")
    return list(rows), dict(traces)


def _pipeline_algorithm(family: str, first: str, p1: int, p2: int, sweeps: int) -> str:
    second = "C" if first == "Q" else "Q"
    if family == "full_lp":
        return f"LP-{first}-to-{second}-p{p1}-{p2}"
    if family == "warm":
        return f"Warm-{first}-separate-QC-p{p1}-{p2}"
    if family == "local":
        return f"Local-{first}-to-{second}-p{p1}-{p2}-s{sweeps}"
    raise ValueError(family)


def execute_full_lp_task(
    base_config: engine.ScreenConfig,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    *,
    resume: bool,
) -> dict[str, object]:
    _require(task.get("stage") == "B" and task.get("family") == "full_lp",
             "not a Stage-B full-LP task")
    first = str(task["first"])
    p1, p2 = int(task["depth_first"]), int(task["depth_second"])
    algorithm = _pipeline_algorithm("full_lp", first, p1, p2, 1)
    rows, traces = _open_resumable_shard(
        manifest, task, [algorithm], resume=resume
    )
    if algorithm not in {row.get("algorithm") for row in rows}:
        config = _batch_config(base_config, task)
        screen = engine.StructuralScreen(config)
        screen.rows = rows
        screen.traces = traces
        screen._run_pipeline_family(
            first,
            p1,
            p2,
            _canonical_pipeline_stream(first, p1, p2),
            "full_lp",
            emit_stage1=False,
        )
        rows, traces = screen.rows, screen.traces
        _save_shard(Path(str(task["output"])), manifest, task, rows, traces, [algorithm])
    return _save_shard(
        Path(str(task["output"])), manifest, task, rows, traces, [algorithm]
    )


def _slice_row_for_task(
    row: Mapping[str, object], task: Mapping[str, object]
) -> dict[str, object]:
    positions = {int(seed): index for index, seed in enumerate(row["initialization_seeds"])}
    indices = [positions[int(seed)] for seed in task["seeds"]]
    result = dict(row)
    for field in engine.PER_RESTART_ROW_FIELDS:
        value = row.get(field)
        if isinstance(value, list):
            result[field] = [value[index] for index in indices]
    return result


def _replay_selected_stage1(
    screen: engine.StructuralScreen,
    selected_row: Mapping[str, object],
    task: Mapping[str, object],
) -> engine.OptimizationResult:
    sliced = _slice_row_for_task(selected_row, task)
    first = str(selected_row["first"])
    p1 = int(selected_row["depth_first"])
    gamma = torch.as_tensor(
        sliced["stage1_gamma_by_restart"], dtype=screen.real_dtype, device=screen.device
    )
    beta = torch.as_tensor(
        sliced["stage1_beta_by_restart"], dtype=screen.real_dtype, device=screen.device
    )
    phase = screen.q if first == "Q" else screen.c
    with torch.no_grad():
        energy, state = screen.block_energy(
            screen.native, (phase,), phase, gamma, beta, p1
        )
    observed_mass = np.asarray(screen.metrics(state)["final_mass"], dtype=np.float64)
    expected_mass = np.asarray(sliced["stage1_final_mass"], dtype=np.float64)
    tolerance = 2e-5 if screen.config.dtype == "complex64" else 1e-10
    if observed_mass.shape != expected_mass.shape or np.max(
        np.abs(observed_mass - expected_mass)
    ) > tolerance:
        raise StructuralActivationError(
            "saved Stage-1 angles do not replay the selected precursor"
        )
    first_evaluations = engine.budget_plan(screen.config, 2)[0]
    return engine.OptimizationResult(
        label=f"Saved-Stage1-{first}-p{p1}",
        state=state,
        energy=energy,
        gamma=gamma,
        beta=beta,
        best_evaluation=torch.zeros(
            screen.config.restarts, dtype=torch.int64, device=screen.device
        ),
        evaluations_per_restart=first_evaluations,
        updates_per_restart=first_evaluations - 1,
        endpoint_evaluations_per_restart=1,
        trace=[{"saved_checkpoint_replay": True, "optimization_performed": False}],
        elapsed_sec=0.0,
    )


def _run_selected_pipeline_control(
    screen: engine.StructuralScreen,
    selected_row: Mapping[str, object],
    task: Mapping[str, object],
    family: str,
) -> dict[str, object]:
    first = str(selected_row["first"])
    second = str(selected_row["second"])
    p1, p2 = int(selected_row["depth_first"]), int(selected_row["depth_second"])
    phase_second = screen.c if second == "C" else screen.q
    stage1 = _replay_selected_stage1(screen, selected_row, task)
    stage1_metrics = screen.metrics(stage1.state)
    first_evals, second_evals = engine.budget_plan(screen.config, 2)
    task_stream = engine._task_stream(dict(task))
    stage1_native = screen.first_ru(first, p1, engine.C_NATIVE_RU)
    stage1_pauli = screen.first_ru(first, p1, engine.C_PAULI_RU)

    if family == "warm":
        result = engine.optimize(
            lambda gamma, beta: screen.block_energy(
                stage1.state, (screen.q, screen.c), screen.common, gamma, beta, p2
            ),
            2 * p2,
            p2,
            second_evals,
            _pipeline_algorithm("warm", first, p1, p2, 1),
            screen.config,
            screen.real_dtype,
            screen.device,
            task_stream + 20,
        )
        native = stage1_native + p2 * (
            engine.Q_RU + engine.C_NATIVE_RU + engine.XY_RU
        )
        pauli = stage1_pauli + p2 * (
            engine.Q_RU + engine.C_PAULI_RU + engine.XY_RU
        )
        row = screen._public_result(
            result,
            _pipeline_algorithm("warm", first, p1, p2, 1),
            "warm",
            p1,
            p2,
            "separate Q and C angles; common cumulative loss",
            "complete block-XY",
            {
                "primary": native + engine.STRUCTURAL_TERMINAL_RU,
                "native_C": native + engine.STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": pauli + engine.STRUCTURAL_TERMINAL_RU,
            },
            first_evals + second_evals,
            (first_evals - 1) + result.updates_per_restart,
            stage1_metrics["final_mass"],
            stage1,
            {"kind": "pipeline", "family": "warm", "first": first, "second": second},
        )
        row["stage1_checkpoint_source_algorithm"] = selected_row["algorithm"]
        row["stage1_checkpoint_replayed_without_optimization"] = True
        return row

    if family != "local":
        raise ValueError(family)
    mixer_kind = "capacity_switch" if first == "Q" else "conflict_guarded"
    result = engine.optimize(
        lambda gamma, beta: screen.template_energy(
            stage1.state,
            phase_second,
            screen.local_mixer(mixer_kind),
            gamma,
            beta,
            p2,
        ),
        p2,
        p2,
        second_evals,
        _pipeline_algorithm("local", first, p1, p2, screen.config.trotter_sweeps),
        screen.config,
        screen.real_dtype,
        screen.device,
        task_stream + 30,
    )
    ledger = engine.local_control_ru_ledger(
        first, p1, p2, screen.config.trotter_sweeps
    )
    description = (
        "75 adjacent-agent four-bit switches"
        if first == "Q"
        else "75 symmetric six-local guarded XY templates"
    )
    row = screen._public_result(
        result,
        _pipeline_algorithm("local", first, p1, p2, screen.config.trotter_sweeps),
        "local",
        p1,
        p2,
        f"new {second} phase; common cumulative loss",
        (
            f"{description}; fixed {screen.config.trotter_sweeps}-sweep "
            "first-order product formula"
        ),
        {
            **ledger,
            f"{screen.config.trotter_sweeps}_sweep_native_C": ledger["native_C"],
            f"{screen.config.trotter_sweeps}_sweep_pauli_C": ledger[
                "pauli_C_sensitivity"
            ],
        },
        first_evals + second_evals,
        (first_evals - 1) + result.updates_per_restart,
        stage1_metrics["final_mass"],
        stage1,
        {
            "kind": "pipeline",
            "family": "local",
            "first": first,
            "second": second,
            "local_mixer": mixer_kind,
            "trotter_sweeps": screen.config.trotter_sweeps,
        },
    )
    row["stage1_checkpoint_source_algorithm"] = selected_row["algorithm"]
    row["stage1_checkpoint_replayed_without_optimization"] = True
    return row


def execute_selected_pipeline_task(
    base_config: engine.ScreenConfig,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    selected_row: Mapping[str, object],
    family: str,
    *,
    resume: bool,
) -> dict[str, object]:
    expected_family = "warm" if family == "warm" else "local_digitized"
    _require(task.get("family") == expected_family, f"not a {family} task")
    sweeps = int(task.get("trotter_sweeps") or 1)
    algorithm = _pipeline_algorithm(
        family,
        str(selected_row["first"]),
        int(selected_row["depth_first"]),
        int(selected_row["depth_second"]),
        sweeps,
    )
    rows, traces = _open_resumable_shard(
        manifest, task, [algorithm], resume=resume
    )
    if algorithm not in {row.get("algorithm") for row in rows}:
        config = _batch_config(base_config, task, sweeps=sweeps)
        screen = engine.StructuralScreen(config)
        screen.rows = rows
        screen.traces = traces
        _run_selected_pipeline_control(screen, selected_row, task, family)
        rows, traces = screen.rows, screen.traces
    return _save_shard(
        Path(str(task["output"])), manifest, task, rows, traces, [algorithm]
    )


def _one_stage_algorithm(family: str, depth: int) -> str:
    return {
        "direct_joint": f"Collapsed-joint-QC-p{depth}",
        "direct_separate": f"Direct-separate-Q-C-p{depth}",
        "fixed_native_joint": f"Fixed-native-rank1-joint-QC-p{depth}",
        "fixed_native_separate": f"Fixed-native-rank1-separate-Q-C-p{depth}",
    }[family]


def _run_one_stage_family(
    screen: engine.StructuralScreen, family: str, depth: int, stream: int
) -> dict[str, object]:
    (evaluations,) = engine.budget_plan(screen.config, 1)
    if family == "direct_joint":
        result = engine.optimize(
            lambda gamma, beta: screen.block_energy(
                screen.native, (screen.common,), screen.common, gamma, beta, depth
            ),
            depth,
            depth,
            evaluations,
            _one_stage_algorithm(family, depth),
            screen.config,
            screen.real_dtype,
            screen.device,
            stream,
        )
        phase = "single Q+C union angle; common cumulative loss"
        mixer = "complete block-XY"
        spec = {"kind": "one_stage", "family": family}
    elif family == "direct_separate":
        result = engine.optimize(
            lambda gamma, beta: screen.block_energy(
                screen.native, (screen.q, screen.c), screen.common, gamma, beta, depth
            ),
            2 * depth,
            depth,
            evaluations,
            _one_stage_algorithm(family, depth),
            screen.config,
            screen.real_dtype,
            screen.device,
            stream + 1,
        )
        phase = "separate Q and C angles; common cumulative loss"
        mixer = "complete block-XY"
        spec = {"kind": "one_stage", "family": family}
    elif family == "fixed_native_joint":
        result = engine.optimize(
            lambda gamma, beta: screen.fixed_rank_one_energy(
                (screen.common,), gamma, beta, depth
            ),
            depth,
            depth,
            evaluations,
            _one_stage_algorithm(family, depth),
            screen.config,
            screen.real_dtype,
            screen.device,
            stream,
        )
        phase = "single Q+C union angle; common cumulative loss"
        mixer = "prescribed native-uniform rank-one projector"
        spec = {"kind": "one_stage", "family": family}
    elif family == "fixed_native_separate":
        result = engine.optimize(
            lambda gamma, beta: screen.fixed_rank_one_energy(
                (screen.q, screen.c), gamma, beta, depth
            ),
            2 * depth,
            depth,
            evaluations,
            _one_stage_algorithm(family, depth),
            screen.config,
            screen.real_dtype,
            screen.device,
            stream + 1,
        )
        phase = "separate Q and C angles; common cumulative loss"
        mixer = "prescribed native-uniform rank-one projector"
        spec = {"kind": "one_stage", "family": family}
    else:
        raise ValueError(family)
    primary = engine.comparator_total_ru(family, depth, pauli_c=False)
    pauli = engine.comparator_total_ru(family, depth, pauli_c=True)
    return screen._public_result(
        result,
        _one_stage_algorithm(family, depth),
        family,
        0,
        depth,
        phase,
        mixer,
        {"primary": primary, "native_C": primary, "pauli_C_sensitivity": pauli},
        evaluations,
        result.updates_per_restart,
        replay_spec=spec,
    )


def execute_comparator_task(
    base_config: engine.ScreenConfig,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    active_families: Sequence[str],
    *,
    resume: bool,
) -> dict[str, object]:
    expected_task_family = (
        "fixed_native_pair"
        if all(family.startswith("fixed_native") for family in active_families)
        else "direct_pair"
    )
    _require(task.get("family") == expected_task_family, "comparator task-family mismatch")
    depth = int(task["depth_second"])
    planned = [_one_stage_algorithm(family, depth) for family in active_families]
    rows, traces = _open_resumable_shard(
        manifest, task, planned, resume=resume
    )
    completed = {str(row["algorithm"]) for row in rows}
    config = _batch_config(base_config, task)
    screen = engine.StructuralScreen(config)
    screen.rows = rows
    screen.traces = traces
    base_stream = engine._task_stream(dict(task))
    for family in active_families:
        algorithm = _one_stage_algorithm(family, depth)
        if algorithm in completed:
            continue
        _run_one_stage_family(screen, family, depth, base_stream)
        completed.add(algorithm)
        _save_shard(
            Path(str(task["output"])),
            manifest,
            task,
            screen.rows,
            screen.traces,
            planned,
        )
    return _save_shard(
        Path(str(task["output"])),
        manifest,
        task,
        screen.rows,
        screen.traces,
        planned,
    )


def _merge_tasks(
    manifest: Mapping[str, object],
    tasks: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        shard = engine._load_and_validate_shard(
            Path(str(task["output"])), dict(manifest), dict(task)
        )
        _require(shard.get("complete") is True, f"incomplete shard {task['task_id']}")
        _require(_target_guard_is_false(shard.get("target_guard")),
                 f"target guard failed in {task['task_id']}")
        for row in shard["rows"]:
            groups.setdefault(str(row["algorithm"]), []).append(row)
    merged = [
        normalize_row(engine._merge_row_shards(rows, expected_seeds))
        for _, rows in sorted(groups.items())
    ]
    return merged


def _full_task_matrix(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    tasks = [
        task
        for task in manifest["tasks"]
        if task.get("stage") == "B" and task.get("family") == "full_lp"
    ]
    expected = 2 * 3 * 7 * len(manifest["seed_batches"])
    _require(len(tasks) == expected, "frozen Stage-B task matrix is incomplete")
    return tasks


def _selected_tasks(
    manifest: Mapping[str, object],
    selection: StructuralSelection,
    family: str,
    sweeps: int | None = None,
) -> list[dict[str, object]]:
    task_family = "warm" if family == "warm" else "local_digitized"
    tasks = [
        task
        for task in manifest["tasks"]
        if task.get("family") == task_family
        and task.get("first") == selection.first
        and int(task.get("depth_first") or 0) == selection.depth_first
        and int(task.get("depth_second") or 0) == selection.depth_second
        and (sweeps is None or int(task.get("trotter_sweeps") or 0) == sweeps)
    ]
    _require(len(tasks) == len(manifest["seed_batches"]),
             f"selected {family} task matrix is incomplete")
    return tasks


def _comparator_tasks(
    manifest: Mapping[str, object], task_family: str, depths: Iterable[int]
) -> list[dict[str, object]]:
    depth_set = set(int(value) for value in depths)
    tasks = [
        task
        for task in manifest["tasks"]
        if task.get("family") == task_family
        and int(task.get("depth_second") or 0) in depth_set
    ]
    observed = {
        int(task["depth_second"]): sum(
            int(other["depth_second"]) == int(task["depth_second"])
            for other in tasks
        )
        for task in tasks
    }
    expected_batches = len(manifest["seed_batches"])
    _require(
        set(observed) == depth_set and all(count == expected_batches for count in observed.values()),
        f"{task_family} frozen depth grid is incomplete",
    )
    return tasks


def _all_finite_rows(
    stage_a: Mapping[str, object],
    stage_b: Mapping[str, object],
    stage_c_rows: Sequence[dict[str, object]],
    stage_d_rows: Sequence[dict[str, object]] = (),
) -> list[dict[str, object]]:
    return [
        *(normalize_row(row) for row in stage_a["methods"]),
        *stage_b["full_lp_rows"],
        *stage_c_rows,
        *stage_d_rows,
    ]


def _decision_from_rows(
    config: engine.ScreenConfig, rows: Sequence[dict[str, object]]
) -> dict[str, object]:
    screen = object.__new__(engine.StructuralScreen)
    screen.config = config
    screen.rows = list(rows)
    return screen.gate_decision()


def _tridiagonal_coefficients(
    alphas: Sequence[torch.Tensor],
    off_diagonals: Sequence[torch.Tensor],
    beta: torch.Tensor,
) -> torch.Tensor:
    batch = beta.shape[0]
    dimension = len(alphas)
    matrix = torch.zeros(
        (batch, dimension, dimension), dtype=alphas[0].dtype, device=beta.device
    )
    index = torch.arange(dimension, device=beta.device)
    matrix[:, index, index] = torch.stack(alphas, dim=1)
    if dimension > 1:
        off = torch.stack(off_diagonals[: dimension - 1], dim=1)
        idx = torch.arange(dimension - 1, device=beta.device)
        matrix[:, idx, idx + 1] = off
        matrix[:, idx + 1, idx] = off
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    phase = torch.exp(-1j * beta[:, None] * eigenvalues)
    spectral = phase * eigenvectors[:, 0, :].to(phase.dtype)
    return torch.bmm(
        eigenvectors.to(phase.dtype), spectral[:, :, None]
    )[:, :, 0]


def krylov_expm_action(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    beta: torch.Tensor,
    *,
    tolerance: float,
    max_dimension: int,
    check_interval: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Adaptive Hermitian Lanczos action with a posteriori residual checks."""

    if state.ndim != 2 or beta.shape != (state.shape[0],):
        raise ValueError("invalid Krylov batch shapes")
    # Match the engine's probability-sum normalization.  On a 759,375-entry
    # complex64 row, `torch.linalg.vector_norm` can lose several 1e-4 solely in
    # its reduction even when `sum(abs(state)**2)` is accurate to 1e-6.
    norm = torch.sqrt(torch.sum(torch.abs(state) ** 2, dim=1))
    # A zero row is legitimate in the custom backward pass when one restart
    # contributes no adjoint signal.  Its exponential action is exactly zero.
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    q_previous = torch.zeros_like(state)
    q = state / safe_norm[:, None]
    previous_off = torch.zeros_like(norm.real)
    basis: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    off_diagonals: list[torch.Tensor] = []
    final_coefficients: torch.Tensor | None = None
    final_residual: torch.Tensor | None = None
    used = 0

    for step in range(max_dimension):
        basis.append(q)
        work = matvec(q)
        alpha = torch.real(torch.sum(q.conj() * work, dim=1))
        work = work - alpha[:, None] * q
        if step:
            work = work - previous_off[:, None] * q_previous
        # A second local orthogonalization is inexpensive and suppresses the
        # dominant finite-precision drift without an O(m^2 N) full reorthogonalization.
        correction = torch.sum(q.conj() * work, dim=1)
        work = work - correction[:, None] * q
        next_off = torch.sqrt(torch.sum(torch.abs(work) ** 2, dim=1)).real
        alphas.append(alpha)
        used = step + 1

        should_check = used % check_interval == 0 or used == max_dimension
        if should_check:
            coefficients = _tridiagonal_coefficients(alphas, off_diagonals, beta)
            residual = next_off * torch.abs(coefficients[:, -1])
            final_coefficients = coefficients
            final_residual = residual
            if bool(torch.all(residual <= tolerance)):
                break
        if used == max_dimension:
            break
        safe = next_off.clamp_min(torch.finfo(next_off.dtype).eps)
        q_previous, q = q, work / safe[:, None]
        previous_off = next_off
        off_diagonals.append(next_off)

    if final_coefficients is None or final_residual is None:
        raise RuntimeError("Krylov action failed to produce coefficients")
    maximum_residual = float(torch.max(final_residual).detach().cpu())
    if maximum_residual > tolerance:
        raise KrylovConvergenceError(
            f"Lanczos residual {maximum_residual:.3e} exceeds {tolerance:.3e} "
            f"at dimension {used}"
        )
    answer = torch.zeros_like(state)
    for index, vector in enumerate(basis[:used]):
        answer = answer + final_coefficients[:, index : index + 1].to(state.dtype) * vector
    answer = answer * norm[:, None]
    return answer, {
        "krylov_dimension": used,
        "maximum_relative_residual_estimate": maximum_residual,
        "tolerance": tolerance,
    }


class _AnalogExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, beta: torch.Tensor, mixer: "AnalogAdjacencyMixer"):
        with torch.no_grad():
            output, diagnostic = mixer.action(state, beta, record=True)
        ctx.mixer = mixer
        ctx.save_for_backward(output, beta)
        mixer.last_diagnostic = diagnostic
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, beta = ctx.saved_tensors
        mixer: AnalogAdjacencyMixer = ctx.mixer
        with torch.no_grad():
            grad_state, _ = mixer.action(grad_output, -beta, record=False)
            derivative = -1j * mixer.matvec(output)
            grad_beta = torch.real(torch.sum(grad_output.conj() * derivative, dim=1))
        return grad_state, grad_beta, None


class AnalogAdjacencyMixer:
    """Numerical exponential of the sum of the 75 audited local generators."""

    def __init__(
        self,
        screen: engine.StructuralScreen,
        kind: str,
        analog_config: AnalogConfig,
    ) -> None:
        analog_config.validate()
        template = screen.local_mixer(kind)
        rows = torch.cat(
            [torch.cat((left, right)) for left, right in template.templates]
        )
        columns = torch.cat(
            [torch.cat((right, left)) for left, right in template.templates]
        )
        values = torch.ones(rows.numel(), dtype=screen.real_dtype, device=screen.device)
        coo = torch.sparse_coo_tensor(
            torch.stack((rows, columns)),
            values,
            (engine.DIMENSION, engine.DIMENSION),
            device=screen.device,
        ).coalesce()
        self.matrix = coo.to_sparse_csr()
        self.kind = kind
        self.config = analog_config
        self.last_diagnostic: dict[str, object] = {}

    def matvec(self, state: torch.Tensor) -> torch.Tensor:
        real = torch.sparse.mm(self.matrix, state.real.T).T
        imag = torch.sparse.mm(self.matrix, state.imag.T).T
        return torch.complex(real, imag).to(state.dtype)

    def action(
        self, state: torch.Tensor, beta: torch.Tensor, *, record: bool
    ) -> tuple[torch.Tensor, dict[str, object]]:
        answer, diagnostic = krylov_expm_action(
            self.matvec,
            state,
            beta,
            tolerance=self.config.tolerance_complex64,
            max_dimension=self.config.max_krylov_dimension,
            check_interval=self.config.check_interval,
        )
        if record:
            self.last_diagnostic = diagnostic
        return answer, diagnostic

    def apply(self, state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return _AnalogExponential.apply(state, beta, self)


def _analog_energy(
    screen: engine.StructuralScreen,
    reference: torch.Tensor,
    phase: torch.Tensor,
    mixer: AnalogAdjacencyMixer,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = reference.clone()

    def layer(current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor):
        current = current * torch.exp(
            -1j * layer_gamma[:, None] * phase[None, :]
        ).to(screen.complex_dtype)
        current = mixer.apply(current, layer_beta)
        return engine.renormalize(current)

    for index in range(depth):
        if screen.config.activation_checkpointing and torch.is_grad_enabled():
            state = engine.checkpoint(
                layer,
                state,
                gamma[:, index],
                beta[:, index],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            state = layer(state, gamma[:, index], beta[:, index])
    return engine.expectation(state, screen.common), state


def _analog_task_path(campaign_root: Path, selection: StructuralSelection, batch: int) -> Path:
    return (
        campaign_root
        / "analog_shards"
        / f"analog__{selection.order}__p1-{selection.depth_first}__p2-{selection.depth_second}"
        f"__batch-{batch:02d}.json"
    )


def execute_analog_task(
    base_config: engine.ScreenConfig,
    manifest: Mapping[str, object],
    campaign_root: Path,
    selected_row: Mapping[str, object],
    batch_index: int,
    analog_config: AnalogConfig,
    *,
    resume: bool,
) -> dict[str, object]:
    selection = _row_selection(selected_row)
    seeds = tuple(int(value) for value in manifest["seed_batches"][batch_index])
    output = _analog_task_path(campaign_root, selection, batch_index)
    task = {
        "task_id": (
            f"C__local_analog__{selection.first}__p1-{selection.depth_first}"
            f"__p2-{selection.depth_second}__batch-{batch_index:02d}"
        ),
        "batch_index": batch_index,
        "seeds": list(seeds),
        "output": str(output),
    }
    algorithm = (
        f"Analog-local-{selection.first}-to-{selection.second}"
        f"-p{selection.depth_first}-{selection.depth_second}"
    )
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        payload = _load_json(output)
        body = {key: value for key, value in payload.items() if key != "shard_sha256"}
        _require(payload.get("shard_sha256") == engine.sha256_json(body),
                 "analog shard checksum mismatch")
        _require(payload.get("manifest_sha256") == manifest.get("manifest_sha256"),
                 "analog shard manifest mismatch")
        _require(payload.get("task_id") == task["task_id"], "analog task mismatch")
        _require(payload.get("seeds") == list(seeds), "analog shard seed mismatch")
        _require(
            payload.get("krylov_protocol") == asdict(analog_config),
            "analog resume settings differ from the completed shard",
        )
        if payload.get("complete") is True:
            return payload

    config = replace(
        base_config,
        restarts=len(seeds),
        initialization_seeds=seeds,
        restart_batch=len(seeds),
        output=output,
    )
    screen = engine.StructuralScreen(config)
    stage1 = _replay_selected_stage1(screen, selected_row, task)
    phase = screen.c if selection.second == "C" else screen.q
    kind = "capacity_switch" if selection.first == "Q" else "conflict_guarded"
    mixer = AnalogAdjacencyMixer(screen, kind, analog_config)
    _, second_evals = engine.budget_plan(config, 2)
    result = engine.optimize(
        lambda gamma, beta: _analog_energy(
            screen, stage1.state, phase, mixer, gamma, beta, selection.depth_second
        ),
        selection.depth_second,
        selection.depth_second,
        second_evals,
        algorithm,
        config,
        screen.real_dtype,
        screen.device,
        600_000_000
        + 10_000 * selection.depth_first
        + 100 * selection.depth_second
        + (0 if selection.first == "Q" else 1),
    )
    stage1_metrics = screen.metrics(stage1.state)
    first_circuit = screen.first_ru(
        selection.first, selection.depth_first, engine.C_NATIVE_RU
    )
    phase_ru = engine.C_NATIVE_RU if selection.second == "C" else engine.Q_RU
    proxy = (
        engine.CAPACITY_SWITCH_PROXY_RU
        if selection.first == "Q"
        else engine.CONFLICT_GUARDED_PROXY_RU
    )
    row = screen._public_result(
        result,
        algorithm,
        "local_analog",
        selection.depth_first,
        selection.depth_second,
        f"new {selection.second} phase; common cumulative loss",
        "exact exponential of the sum of 75 audited adjacency generators",
        {
            "primary": None,
            "finite_RU_rank": None,
            "generator_proxy_native_C": first_circuit
            + selection.depth_second * (phase_ru + proxy)
            + engine.STRUCTURAL_TERMINAL_RU,
        },
        sum(engine.budget_plan(config, 2)),
        2 * (second_evals - 1),
        stage1_metrics["final_mass"],
        stage1,
        {
            "kind": "pipeline_analog",
            "family": "local_analog",
            "first": selection.first,
            "second": selection.second,
            "local_mixer": kind,
            "finite_RU_rank": False,
        },
    )
    row["stage1_checkpoint_source_algorithm"] = selected_row["algorithm"]
    row["stage1_checkpoint_replayed_without_optimization"] = True
    row = normalize_row(row)
    body = engine._json_safe(
        {
            "schema": f"{STRUCTURAL_WORKER_SCHEMA}-analog-shard-v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "task_id": task["task_id"],
            "batch_index": batch_index,
            "seeds": list(seeds),
            "target_guard": {
                "objective_table_loaded": False,
                "target_constructed": False,
                "target_probability_computed": False,
            },
            "complete": True,
            "rows": [row],
            "traces": screen.traces,
            "krylov_protocol": asdict(analog_config),
            "last_krylov_diagnostic": mixer.last_diagnostic,
        }
    )
    payload = {**body, "shard_sha256": engine.sha256_json(body)}
    engine.atomic_write_json(output, payload)
    return payload


def _merge_analog_shards(
    manifest: Mapping[str, object],
    campaign_root: Path,
    selections: Sequence[StructuralSelection],
) -> list[dict[str, object]]:
    rows_by_algorithm: dict[str, list[dict[str, object]]] = {}
    for selection in selections:
        for batch in range(len(manifest["seed_batches"])):
            payload = _load_json(_analog_task_path(campaign_root, selection, batch))
            _require(payload.get("complete") is True, "analog shard is incomplete")
            body = {key: value for key, value in payload.items() if key != "shard_sha256"}
            _require(payload.get("shard_sha256") == engine.sha256_json(body),
                     "analog shard checksum mismatch")
            for row in payload["rows"]:
                rows_by_algorithm.setdefault(str(row["algorithm"]), []).append(row)
    return [
        normalize_row(engine._merge_row_shards(rows, EXPECTED_RESTART_SEEDS))
        for _, rows in sorted(rows_by_algorithm.items())
    ]


class PostStageAStructuralWorker:
    def __init__(
        self,
        campaign_root: Path,
        stage_a_path: Path,
        execution_target: str,
        device: str,
        *,
        resume: bool = True,
        analog_config: AnalogConfig = AnalogConfig(),
    ) -> None:
        self.campaign_root = campaign_root.resolve()
        self.stage_a_path = stage_a_path.resolve()
        self.execution_target = execution_target
        self.resume = resume
        self.analog_config = analog_config
        self.analog_config.validate()
        self.config = replace(
            engine.valid_config(),
            device=device,
            restart_batch=4 if execution_target == "remote80" else 1,
        )
        self.manifest_path = self.campaign_root / engine.MANIFEST_FILENAME
        self.manifest = _load_json(self.manifest_path)
        engine.validate_manifest(self.manifest, self.config, execution_target)
        self.stage_a = _load_json(self.stage_a_path)
        validate_passing_stage_a(self.stage_a, self.manifest)

    @property
    def stage_b_path(self) -> Path:
        return self.campaign_root / "stage_B_full_lp_aggregate.json"

    @property
    def stage_c_path(self) -> Path:
        return self.campaign_root / "stage_C_finite_controls_aggregate.json"

    @property
    def stage_d_path(self) -> Path:
        return self.campaign_root / "stage_D_preliminary_structural_aggregate.json"

    @property
    def stage_e_path(self) -> Path:
        return self.campaign_root / "stage_E_saved_angle_complex128_replay.json"

    @property
    def handoff_path(self) -> Path:
        return self.campaign_root / "final_structural_handoff.json"

    def run_stage_b(self) -> dict[str, object]:
        started = time.time()
        tasks = _full_task_matrix(self.manifest)
        for task in tasks:
            execute_full_lp_task(
                self.config, self.manifest, task, resume=self.resume
            )
        rows = _merge_tasks(self.manifest, tasks, EXPECTED_RESTART_SEEDS)
        expected = {
            (first, p1, p2)
            for first in ("Q", "C")
            for p1 in self.config.depths
            for p2 in self.config.learned_projector_depths
        }
        observed = {
            (row.get("first"), row.get("depth_first"), row.get("depth_second"))
            for row in rows
        }
        _require(observed == expected and len(rows) == len(expected),
                 "aggregated full-LP grid is incomplete or duplicated")
        q_choice, c_choice, primary = select_both_orders(rows)
        primary_row = next(row for row in rows if row["algorithm"] == primary.algorithm)
        gate = _stage_b_gate(primary_row, rows, self.config.profile)
        payload = engine._json_safe(
            {
                "schema": STAGE_B_SCHEMA,
                "profile": self.config.profile,
                "stage_a_manifest_id": self.manifest["manifest_sha256"],
                "target_guard": {
                    "objective_table_loaded": False,
                    "target_constructed": False,
                    "target_probability_computed": False,
                },
                "full_grid_complete": True,
                "all_16_restarts_retained": True,
                "full_lp_rows": rows,
                "selected_orders": {
                    q_choice.order: _selection_record(q_choice),
                    c_choice.order: _selection_record(c_choice),
                },
                "primary_order": primary.order,
                "primary_selection": _selection_record(primary),
                "gate": gate,
                "elapsed_sec": time.time() - started,
            }
        )
        engine.atomic_write_json(self.stage_b_path, payload)
        return payload

    def _load_stage_b_pass(self) -> tuple[
        dict[str, object], dict[str, object], dict[str, object], StructuralSelection
    ]:
        stage_b = _load_json(self.stage_b_path)
        _require(stage_b.get("schema") == STAGE_B_SCHEMA, "Stage-B schema mismatch")
        _require(stage_b.get("gate", {}).get("pass") is True, "Stage B did not pass")
        rows = stage_b["full_lp_rows"]
        selections = stage_b["selected_orders"]
        q_record = selections["Q-C"]
        c_record = selections["C-Q"]
        by_name = {str(row["algorithm"]): row for row in rows}
        q_row = by_name[str(q_record["complex64_algorithm"])]
        c_row = by_name[str(c_record["complex64_algorithm"])]
        primary = _row_selection(by_name[str(stage_b["primary_selection"]["complex64_algorithm"])])
        return stage_b, q_row, c_row, primary

    def _aggregate_stage_c_finite_rows(
        self,
        warm_tasks: Sequence[Mapping[str, object]],
        fixed_tasks: Sequence[Mapping[str, object]],
        local_tasks: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        return [
            *_merge_tasks(self.manifest, warm_tasks, EXPECTED_RESTART_SEEDS),
            *_merge_tasks(self.manifest, fixed_tasks, EXPECTED_RESTART_SEEDS),
            *_merge_tasks(self.manifest, local_tasks, EXPECTED_RESTART_SEEDS),
        ]

    def run_stage_c(self) -> dict[str, object]:
        started = time.time()
        stage_b, q_row, c_row, primary = self._load_stage_b_pass()
        selected_rows = (q_row, c_row)
        selections = (_row_selection(q_row), _row_selection(c_row))
        by_order = {selection.order: row for selection, row in zip(selections, selected_rows)}

        warm_tasks: list[dict[str, object]] = []
        for selection in selections:
            tasks = _selected_tasks(self.manifest, selection, "warm")
            warm_tasks.extend(tasks)
            for task in tasks:
                execute_selected_pipeline_task(
                    self.config,
                    self.manifest,
                    task,
                    by_order[selection.order],
                    "warm",
                    resume=self.resume,
                )
        warm_rows = _merge_tasks(self.manifest, warm_tasks, EXPECTED_RESTART_SEEDS)
        coarse = [normalize_row(row) for row in self.stage_a["methods"]]
        winner = next(
            row for row in stage_b["full_lp_rows"] if row["algorithm"] == primary.algorithm
        )
        warm_gate = evaluate_control_envelope(winner, [*coarse, *warm_rows])
        if not warm_gate["pass"]:
            payload = self._stage_c_payload(
                stage_b, warm_rows, [], [], [], warm_gate, started, "warm"
            )
            engine.atomic_write_json(self.stage_c_path, payload)
            return payload

        depth_plan = active_comparator_depths(
            primary.primary_ru, selections, self.config
        )
        fixed_depths = sorted(
            set(depth_plan["fixed_native_joint"])
            | set(depth_plan["fixed_native_separate"])
        )
        fixed_tasks = _comparator_tasks(
            self.manifest, "fixed_native_pair", fixed_depths
        )
        for task in fixed_tasks:
            depth = int(task["depth_second"])
            active = [
                family
                for family in ("fixed_native_joint", "fixed_native_separate")
                if depth in depth_plan[family]
            ]
            execute_comparator_task(
                self.config, self.manifest, task, active, resume=self.resume
            )
        fixed_rows = _merge_tasks(self.manifest, fixed_tasks, EXPECTED_RESTART_SEEDS)
        fixed_gate = evaluate_control_envelope(
            winner, [*coarse, *warm_rows, *fixed_rows]
        )
        if not fixed_gate["pass"]:
            payload = self._stage_c_payload(
                stage_b, warm_rows, fixed_rows, [], [], fixed_gate, started, "fixed"
            )
            engine.atomic_write_json(self.stage_c_path, payload)
            return payload

        local_tasks: list[dict[str, object]] = []
        for selection in selections:
            for sweeps in self.config.local_sweep_sensitivities:
                tasks = _selected_tasks(
                    self.manifest, selection, "local", sweeps=sweeps
                )
                local_tasks.extend(tasks)
                for task in tasks:
                    execute_selected_pipeline_task(
                        self.config,
                        self.manifest,
                        task,
                        by_order[selection.order],
                        "local",
                        resume=self.resume,
                    )
        local_rows = _merge_tasks(self.manifest, local_tasks, EXPECTED_RESTART_SEEDS)
        final_gate = evaluate_control_envelope(
            winner, [*coarse, *warm_rows, *fixed_rows, *local_rows]
        )
        if not final_gate["pass"]:
            payload = self._stage_c_payload(
                stage_b, warm_rows, fixed_rows, local_rows, [], final_gate, started, "local"
            )
            engine.atomic_write_json(self.stage_c_path, payload)
            return payload

        analog_rows: list[dict[str, object]] = []
        for row in selected_rows:
            for batch in range(len(self.manifest["seed_batches"])):
                execute_analog_task(
                    self.config,
                    self.manifest,
                    self.campaign_root,
                    row,
                    batch,
                    self.analog_config,
                    resume=self.resume,
                )
        analog_rows = _merge_analog_shards(
            self.manifest, self.campaign_root, selections
        )
        payload = self._stage_c_payload(
            stage_b,
            warm_rows,
            fixed_rows,
            local_rows,
            analog_rows,
            final_gate,
            started,
            "complete",
        )
        payload["comparator_depth_plan"] = {
            key: list(value) for key, value in depth_plan.items()
        }
        engine.atomic_write_json(self.stage_c_path, payload)
        return payload

    def _stage_c_payload(
        self,
        stage_b: Mapping[str, object],
        warm: Sequence[dict[str, object]],
        fixed: Sequence[dict[str, object]],
        local: Sequence[dict[str, object]],
        analog: Sequence[dict[str, object]],
        gate: Mapping[str, object],
        started: float,
        completed_wave: str,
    ) -> dict[str, object]:
        return engine._json_safe(
            {
                "schema": STAGE_C_SCHEMA,
                "profile": self.config.profile,
                "stage_a_manifest_id": self.manifest["manifest_sha256"],
                "target_guard": {
                    "objective_table_loaded": False,
                    "target_constructed": False,
                    "target_probability_computed": False,
                },
                "selected_orders": stage_b["selected_orders"],
                "primary_order": stage_b["primary_order"],
                "completed_wave": completed_wave,
                "finite_rows": [*warm, *fixed, *local],
                "warm_rows": list(warm),
                "fixed_native_rows": list(fixed),
                "local_digitized_rows": list(local),
                "analog_local_sensitivity_rows": list(analog),
                "analog_finite_RU_rank": False,
                "gate": dict(gate),
                "elapsed_sec": time.time() - started,
            }
        )

    def run_stage_d(self) -> dict[str, object]:
        started = time.time()
        stage_b, q_row, c_row, primary = self._load_stage_b_pass()
        stage_c = _load_json(self.stage_c_path)
        _require(stage_c.get("schema") == STAGE_C_SCHEMA, "Stage-C schema mismatch")
        _require(
            stage_c.get("gate", {}).get("pass") is True
            and stage_c.get("completed_wave") == "complete",
            "Stage C did not complete and pass",
        )
        selections = (_row_selection(q_row), _row_selection(c_row))
        depth_plan = active_comparator_depths(
            primary.primary_ru, selections, self.config
        )
        direct_depths = sorted(
            set(depth_plan["direct_joint"]) | set(depth_plan["direct_separate"])
        )
        tasks = _comparator_tasks(self.manifest, "direct_pair", direct_depths)
        for task in tasks:
            depth = int(task["depth_second"])
            active = [
                family
                for family in ("direct_joint", "direct_separate")
                if depth in depth_plan[family]
            ]
            execute_comparator_task(
                self.config, self.manifest, task, active, resume=self.resume
            )
        direct_rows = _merge_tasks(self.manifest, tasks, EXPECTED_RESTART_SEEDS)
        all_rows = _all_finite_rows(
            self.stage_a,
            stage_b,
            stage_c["finite_rows"],
            direct_rows,
        )
        decision = _decision_from_rows(self.config, all_rows)
        payload = engine._json_safe(
            {
                "schema": STAGE_D_SCHEMA,
                "profile": self.config.profile,
                "stage_a_manifest_id": self.manifest["manifest_sha256"],
                "target_guard": {
                    "objective_table_loaded": False,
                    "target_constructed": False,
                    "target_probability_computed": False,
                },
                "selected_orders": stage_b["selected_orders"],
                "primary_order": stage_b["primary_order"],
                "required_comparator_depth_manifest": {
                    key: list(value)
                    for key, value in engine.required_comparator_depths(
                        primary.primary_ru, frozen_candidate_depths(self.config)
                    ).items()
                },
                "executed_comparator_depth_plan": {
                    key: list(value) for key, value in depth_plan.items()
                },
                "direct_rows": direct_rows,
                "complex64_methods": all_rows,
                "analog_local_sensitivity_rows": stage_c[
                    "analog_local_sensitivity_rows"
                ],
                "gate": decision,
                "elapsed_sec": time.time() - started,
            }
        )
        engine.atomic_write_json(self.stage_d_path, payload)
        return payload

    def run_stage_e(self) -> tuple[dict[str, object], dict[str, object]]:
        stage_b, _, _, primary = self._load_stage_b_pass()
        stage_d = _load_json(self.stage_d_path)
        _require(stage_d.get("schema") == STAGE_D_SCHEMA, "Stage-D schema mismatch")
        _require(
            stage_d.get("gate", {}).get("preliminary_pass") is True,
            "Stage D did not earn saved-angle replay",
        )
        source = {
            "gate": stage_d["gate"],
            "methods": stage_d["complex64_methods"],
        }
        replay = engine.run_saved_angle_precision_replay(
            source, replace(self.config, dtype="complex128")
        )
        engine.atomic_write_json(self.stage_e_path, replay)
        handoff = self._build_handoff(stage_b, stage_d, replay, primary)
        engine.atomic_write_json(self.handoff_path, handoff)
        return replay, handoff

    def _build_handoff(
        self,
        stage_b: Mapping[str, object],
        stage_d: Mapping[str, object],
        replay: Mapping[str, object],
        primary: StructuralSelection,
    ) -> dict[str, object]:
        methods = list(stage_d["complex64_methods"])
        replay_rows = list(replay["replayed_rows"])
        max_norm_error = max(
            abs(float(value) - 1.0)
            for row in replay_rows
            for value in row["state_norm"]
        )
        required = engine.required_comparator_depths(
            primary.primary_ru, frozen_candidate_depths(self.config)
        )
        completed = {
            family: sorted(
                {
                    int(row["depth_second"])
                    for row in methods
                    if row.get("family") == family
                }
            )
            for family in COMPARATOR_FAMILIES
        }
        for family, depths in required.items():
            _require(set(depths).issubset(completed[family]),
                     f"Stage-D ladder incomplete for {family}")
        all_costs_finite = all(
            _finite_positive(row.get("feasibility_RTS99_x_primary_RU"))
            for row in methods
        )
        replay_evidence = {
            "status": replay["gate"]["status"],
            "pass": replay["gate"]["pass"],
            "saved_angles_only": True,
            "optimization_performed": False,
            "configuration_reselected": False,
            "max_norm_error": max_norm_error,
            "checks": replay["gate"]["checks"],
            "replayed_rows": replay_rows,
            "selected_control_algorithms": replay["selected_control_algorithms"],
        }
        body = engine._json_safe(
            {
                "schema": STRUCTURAL_HANDOFF_SCHEMA,
                "status": replay["gate"]["status"],
                "pass": replay["gate"]["pass"],
                "profile": self.config.profile,
                "stage_a_manifest_id": self.manifest["manifest_sha256"],
                "target_guard": {
                    "objective_table_loaded": False,
                    "target_constructed": False,
                    "target_probability_computed": False,
                },
                "all_16_restarts_retained": True,
                "all_unfiltered_rows_retained": True,
                "nonfinite_costs_present": not all_costs_finite,
                "complex64_methods": methods,
                "analog_local_sensitivity_rows": stage_d[
                    "analog_local_sensitivity_rows"
                ],
                "analog_finite_RU_rank": False,
                "saved_angle_complex128_replay": replay_evidence,
                "selected_orders": stage_b["selected_orders"],
                "primary_order": stage_b["primary_order"],
                "primary_order_selected_target_free": True,
                "required_comparator_depth_manifest": {
                    key: list(value) for key, value in required.items()
                },
                "completed_comparator_depth_manifest": completed,
            }
        )
        return {**body, "handoff_id": engine.sha256_json(body)}

    def run_all(self) -> dict[str, object]:
        stage_b = self.run_stage_b()
        if not stage_b["gate"]["pass"]:
            return stage_b
        stage_c = self.run_stage_c()
        if not stage_c["gate"]["pass"]:
            return stage_c
        stage_d = self.run_stage_d()
        if not stage_d["gate"]["preliminary_pass"]:
            return stage_d
        _, handoff = self.run_stage_e()
        return handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("stage-b", "stage-c", "stage-d", "stage-e", "all"), required=True
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--stage-a-aggregate", type=Path, required=True)
    parser.add_argument(
        "--execution-target", choices=("remote80", "local8"), default="remote80"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--analog-tolerance", type=float, default=2e-6)
    parser.add_argument("--analog-max-krylov", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analog = AnalogConfig(
        tolerance_complex64=args.analog_tolerance,
        max_krylov_dimension=args.analog_max_krylov,
    )
    worker = PostStageAStructuralWorker(
        args.campaign_root,
        args.stage_a_aggregate,
        args.execution_target,
        args.device,
        resume=not args.no_resume,
        analog_config=analog,
    )
    if args.stage == "stage-b":
        payload = worker.run_stage_b()
        output = worker.stage_b_path
    elif args.stage == "stage-c":
        payload = worker.run_stage_c()
        output = worker.stage_c_path
    elif args.stage == "stage-d":
        payload = worker.run_stage_d()
        output = worker.stage_d_path
    elif args.stage == "stage-e":
        _, payload = worker.run_stage_e()
        output = worker.handoff_path
    else:
        payload = worker.run_all()
        output = (
            worker.handoff_path
            if payload.get("schema") == STRUCTURAL_HANDOFF_SCHEMA
            else worker.stage_b_path
            if payload.get("schema") == STAGE_B_SCHEMA
            else worker.stage_c_path
            if payload.get("schema") == STAGE_C_SCHEMA
            else worker.stage_d_path
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "schema": payload.get("schema"),
                "status": payload.get("status", payload.get("gate", {}).get("status")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
