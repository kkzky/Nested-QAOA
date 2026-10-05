#!/usr/bin/env python3
"""Target-blind seed-10 extension and high-depth direct diagnostics.

This campaign never writes into the completed three-table depth sweep.  It
adds seven fresh, unfiltered PCG64 service tables to every original curve and
adds a p=96 diagnostic point for both direct block-XY baselines on
all ten tables.  The old three-table results are merged only by the post-seal
scorer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

import stage3_objective_common as objective_engine
import target_free_structural_screen as engine
import targeted_coherent_im3_depth_sweep as source_campaign
import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_screen as discovery
import targeted_coherent_im3_variational_f3 as variational
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-seed10-extension-high-direct-v1"
IMPLEMENTATION_REVISION = "fresh7-plus-all10-direct-p96-r3-20260819"
MANIFEST_NAME = "targeted_coherent_im3_seed10_extension_manifest.json"
PLAN_NAME = "targeted_coherent_im3_seed10_extension_two_pod_plan.json"
RESULT_DIRECTORY = "targeted_coherent_im3_seed10_extension_results"
GLOBAL_SEAL_NAME = "targeted_coherent_im3_seed10_extension_global_seal.json"
QUEUE_SOURCE_NAME = "targeted_coherent_im3_seed10_extension_queue.py"
SCORER_SOURCE_NAME = "targeted_coherent_im3_seed10_extension_score.py"
VANILLA_SOURCE_NAME = "targeted_coherent_im3_vanilla_qaoa.py"
SUPERVISOR_SOURCE_NAME = "targeted_coherent_im3_seed10_extension_supervisor.py"
PROTOCOL_SOURCE_NAME = "COHERENT_IM3_SEED10_EXTENSION_PROTOCOL_20260819.md"

SOURCE_TABLE_SEEDS = tuple(source_campaign.TABLE_SEEDS)
ADDITIONAL_TABLE_SEEDS = tuple(range(2_026_081_804, 2_026_081_811))
ALL_TABLE_SEEDS = SOURCE_TABLE_SEEDS + ADDITIONAL_TABLE_SEEDS
RESTART_SEEDS = tuple(source_campaign.RESTART_SEEDS)

FAMILIES = tuple(source_campaign.FAMILIES)
STANDARD_DEPTHS = tuple(source_campaign.POSITIVE_DEPTHS)
HIGH_DIRECT_FAMILIES = (
    source_campaign.DIRECT_SEPARATE,
    source_campaign.DIRECT_COMBINED,
)
HIGH_DIRECT_DEPTHS = (96,)
ALL_RESULT_DEPTHS = tuple(sorted(set(STANDARD_DEPTHS + HIGH_DIRECT_DEPTHS)))

TASK_CURVE_EXTENSION = "fresh7_standard_depth_curve"
TASK_HIGH_DIRECT = "all10_high_depth_direct_diagnostic"
TASK_KINDS = (TASK_CURVE_EXTENSION, TASK_HIGH_DIRECT)

OBJECTIVE_EVALUATIONS = int(source_campaign.OBJECTIVE_EVALUATIONS)
LEARNING_RATE = float(source_campaign.LEARNING_RATE)
VF3_REPLAY_ATOL = float(source_campaign.VF3_REPLAY_ATOL)
VF3_REPLAY_RTOL = float(source_campaign.VF3_REPLAY_RTOL)


@dataclass(frozen=True)
class Task:
    task_kind: str
    table_seed: int
    family: str
    depth: int


def expected_tasks() -> tuple[Task, ...]:
    standard = (
        Task(TASK_CURVE_EXTENSION, seed, family, depth)
        for family in FAMILIES
        for depth in STANDARD_DEPTHS
        for seed in ADDITIONAL_TABLE_SEEDS
    )
    high = (
        Task(TASK_HIGH_DIRECT, seed, family, depth)
        for family in HIGH_DIRECT_FAMILIES
        for depth in HIGH_DIRECT_DEPTHS
        for seed in ALL_TABLE_SEEDS
    )
    tasks = tuple((*standard, *high))
    common._require(len(tasks) == 412 and len(set(tasks)) == len(tasks), "task set changed")
    return tasks


def task_allowed(task_kind: str, seed: int, family: str, depth: int) -> bool:
    return Task(str(task_kind), int(seed), str(family), int(depth)) in set(expected_tasks())


def table_id(seed: int) -> str:
    common._require(seed in ALL_TABLE_SEEDS, f"unknown table seed {seed}")
    return f"pcg64_{int(seed)}"


def method_id(task_kind: str, family: str, depth: int) -> str:
    common._require(task_kind in TASK_KINDS, f"unknown task kind {task_kind}")
    common._require(family in FAMILIES, f"unknown family {family}")
    return f"coherent_im3_seed10_extension__{task_kind}__{family}__p{int(depth)}"


def result_path(
    root: Path, task_kind: str, seed: int, family: str, depth: int
) -> Path:
    common._require(task_allowed(task_kind, seed, family, depth), "invalid result cell")
    return (
        root / RESULT_DIRECTORY / table_id(seed)
        / f"{method_id(task_kind, family, depth)}.json"
    )


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _plan_path(root: Path) -> Path:
    return root / PLAN_NAME


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_source_sha256() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = (
        Path(__file__).name,
        QUEUE_SOURCE_NAME,
        SCORER_SOURCE_NAME,
        VANILLA_SOURCE_NAME,
        SUPERVISOR_SOURCE_NAME,
        PROTOCOL_SOURCE_NAME,
        "targeted_coherent_im3_depth_sweep.py",
        "targeted_coherent_im3_depth_sweep_queue.py",
        "targeted_coherent_im3_depth_sweep_score.py",
        "COHERENT_IM3_DEPTH_SWEEP_PROTOCOL_20260818.md",
        "stage3_objective_common.py",
        "target_free_structural_screen.py",
        "targeted_three_stage_common.py",
        "targeted_coherent_im3_variational_f3.py",
        "targeted_coherent_im3_fixed_uniform_control.py",
        "targeted_coherent_im3_objective.py",
        "targeted_coherent_im3_screen.py",
        "targeted_coherent_im3_score.py",
        "targeted_coherent_im3_heldout.py",
    )
    paths = tuple(source_root / name for name in names)
    for path in paths:
        common._require(path.is_file(), f"missing frozen source {path}")
    return {path.name: _file_sha256(path) for path in paths}


def _verified_source(source_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest, plan = source_campaign.validated_plan(source_root.resolve())
    seal_path = source_root.resolve() / source_campaign.GLOBAL_SEAL_NAME
    seal = common.load_json(seal_path)
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    common._require(
        seal.get("schema") == f"{source_campaign.SCHEMA}-global-seal"
        and seal.get("status") == "SEALED_COMPLETE_TARGET_BLIND_DEPTH_SWEEP"
        and seal.get("manifest_sha256") == manifest["manifest_sha256"]
        and seal.get("plan_sha256") == plan["plan_sha256"]
        and seal.get("seal_sha256") == common.sha256_json(body)
        and seal.get("complete_optimizer_result_count")
        == len(SOURCE_TABLE_SEEDS) * len(FAMILIES) * len(STANDARD_DEPTHS),
        "source three-table depth sweep is not a valid immutable seal",
    )
    expected = {
        (seed, family, depth)
        for seed in SOURCE_TABLE_SEEDS
        for family in FAMILIES
        for depth in STANDARD_DEPTHS
    }
    sealed_rows = seal.get("completed_cells", ())
    sealed = {
        (int(row["table_seed"]), str(row["family"]), int(row["depth"])):
        str(row["result_sha256"])
        for row in sealed_rows
    }
    expected_paths = {
        source_campaign.result_path(source_root, seed, family, depth).resolve()
        for seed, family, depth in expected
    }
    actual_paths = {
        path.resolve()
        for path in (source_root / source_campaign.RESULT_DIRECTORY).glob("*/*.json")
    }
    common._require(
        len(sealed_rows) == len(expected)
        and set(sealed) == expected
        and actual_paths == expected_paths,
        "source seal result-key/path set is not exact",
    )
    for seed, family, depth in expected:
        result = common.load_json(
            source_campaign.result_path(source_root, seed, family, depth)
        )
        source_campaign.validate_result(
            manifest, result, seed, family, depth
        )
        common._require(
            result["result_sha256"] == sealed[(seed, family, depth)],
            f"source result bytes no longer match seal {seed}/{family}/p{depth}",
        )
    return manifest, seal


def resource(family: str, depth: int) -> dict[str, int]:
    if depth in source_campaign.DEPTHS:
        return source_campaign.resource(family, depth)
    common._require(
        family in HIGH_DIRECT_FAMILIES and depth in HIGH_DIRECT_DEPTHS,
        "invalid high-depth resource cell",
    )

    def one(c_ru: int) -> int:
        if family == source_campaign.DIRECT_SEPARATE:
            return int(
                source_campaign.PREP_RU
                + depth
                * (
                    source_campaign.Q_RU
                    + c_ru
                    + source_campaign.O_RU
                    + source_campaign.XY_RU
                )
                + source_campaign.O_RU
            )
        return int(
            source_campaign.PREP_RU
            + depth * (source_campaign.QCO_UNION_RU + source_campaign.XY_RU)
            + source_campaign.O_RU
        )

    native = one(source_campaign.C_NATIVE_RU)
    pauli = (
        one(source_campaign.C_PAULI_RU)
        if family == source_campaign.DIRECT_SEPARATE
        else native
    )
    return {"native_C": native, "uniform_Pauli_C_sensitivity": pauli}


def _standard_initial_arrays(family: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
    # Byte-for-byte the frozen initialization used by the completed curves.
    return source_campaign._initial_angle_arrays(family, depth)


def _high_initial_arrays(family: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Preserve each p64 phase prefix and append independent frozen tails."""

    common._require(family in HIGH_DIRECT_FAMILIES and depth in HIGH_DIRECT_DEPTHS, "bad high cell")
    spec = source_campaign.family_specs()[family]
    base_gamma, base_beta = _standard_initial_arrays(family, 64)
    base_gamma = base_gamma.reshape(len(RESTART_SEEDS), spec.gamma_multiplier, 64)
    tail_length = max(HIGH_DIRECT_DEPTHS) - 64
    gamma_rows: list[np.ndarray] = []
    beta_rows: list[np.ndarray] = []
    for restart_index, seed in enumerate(RESTART_SEEDS):
        phase_rows: list[np.ndarray] = []
        for phase_index in range(spec.gamma_multiplier):
            rng = np.random.default_rng(
                int(seed)
                + 1_000_003 * int(400_000 + 31 * spec.stream + phase_index)
            )
            tail = rng.uniform(-math.pi, math.pi, size=tail_length)
            phase_rows.append(
                np.concatenate((base_gamma[restart_index, phase_index], tail[: depth - 64]))
            )
        gamma_rows.append(np.concatenate(phase_rows))
        beta_rng = np.random.default_rng(
            int(seed) + 1_000_003 * int(500_000 + 31 * spec.stream)
        )
        beta_tail = beta_rng.uniform(-math.pi, math.pi, size=tail_length)
        beta_rows.append(
            np.concatenate((base_beta[restart_index], beta_tail[: depth - 64]))
        )
    gamma = np.asarray(gamma_rows, dtype=np.float64)
    beta = np.asarray(beta_rows, dtype=np.float64)
    common._require(
        gamma.shape == (len(RESTART_SEEDS), spec.gamma_multiplier * depth)
        and beta.shape == (len(RESTART_SEEDS), depth)
        and np.array_equal(
            gamma.reshape(len(RESTART_SEEDS), spec.gamma_multiplier, depth)[:, :, :64],
            base_gamma,
        )
        and np.array_equal(beta[:, :64], base_beta),
        "high-depth prefix contract failed",
    )
    return gamma, beta


def initial_arrays(
    task_kind: str, family: str, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    if task_kind == TASK_CURVE_EXTENSION:
        return _standard_initial_arrays(family, depth)
    return _high_initial_arrays(family, depth)


def initial_angles_sha256(task_kind: str, family: str, depth: int) -> str:
    gamma, beta = initial_arrays(task_kind, family, depth)
    return common.sha256_json(
        {"gamma_by_restart": gamma.tolist(), "beta_by_restart": beta.tolist()}
    )


def initialization_policy(task_kind: str, family: str, depth: int) -> dict[str, object]:
    if task_kind == TASK_CURVE_EXTENSION:
        return {
            **source_campaign.initialization_policy(family),
            "exactly_reuses_completed_curve_initialization_contract": True,
        }
    return {
        "independent_at_every_positive_depth": True,
        "continuation_or_optimized_angle_transfer": False,
        "distribution": "uniform[-pi,pi]",
        "p64_random_initialization_prefix_byte_identical_to_completed_curve": True,
        "tail_rng": "numpy PCG64 with phase-separated frozen extension streams",
        "append_only_tail_preserves_every_p64_phase_prefix": True,
        "gamma_layout": "phase-major",
        "identity_padding": False,
    }


def _task_contract(task: Task) -> dict[str, object]:
    return {
        "task_kind": task.task_kind,
        "table_seed": task.table_seed,
        "table_id": table_id(task.table_seed),
        "family": task.family,
        "depth": task.depth,
        "method_id": method_id(task.task_kind, task.family, task.depth),
        "resource": resource(task.family, task.depth),
        "initial_angles_sha256": initial_angles_sha256(
            task.task_kind, task.family, task.depth
        ),
    }


def freeze(root: Path, source_root: Path) -> dict[str, object]:
    root = root.resolve()
    source_manifest, source_seal = _verified_source(source_root)
    tables = [coherent.coefficient_table_from_pcg64(seed) for seed in ALL_TABLE_SEEDS]
    stage2 = source_manifest["embedded_service_independent_VF3"]
    c12 = source_manifest["embedded_C12_row"]
    tasks = expected_tasks()
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_TARGET_BLIND_SEED10_EXTENSION_AND_DIRECT_DIAGNOSTICS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "frozen_source_sha256": frozen_source_sha256(),
        "source_three_table_depth_sweep": {
            "manifest_sha256": source_manifest["manifest_sha256"],
            "seal_sha256": source_seal["seal_sha256"],
            "table_seeds": list(SOURCE_TABLE_SEEDS),
            "immutable_and_not_written_by_this_campaign": True,
            "merge_only_after_this_campaign_global_seal": True,
        },
        "cohort": {
            "generator": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "draw": "integers(low=1, high=21, size=(5,6), dtype=int64)",
            "source_table_seeds": list(SOURCE_TABLE_SEEDS),
            "additional_table_seeds": list(ADDITIONAL_TABLE_SEEDS),
            "all_ten_table_seeds": list(ALL_TABLE_SEEDS),
            "tables_in_seed_order": tables,
            "seven_new_tables_unfiltered_no_rejection_or_replacement": True,
            "only_service_cost_matrix_varies": True,
        },
        "standard_curve_extension": {
            "table_seeds": list(ADDITIONAL_TABLE_SEEDS),
            "families": list(FAMILIES),
            "positive_depths": list(STANDARD_DEPTHS),
            "old_three_p0_and_positive_depth_cells_reused_only_post_seal": True,
        },
        "high_depth_direct_diagnostic": {
            "table_seeds": list(ALL_TABLE_SEEDS),
            "families": list(HIGH_DIRECT_FAMILIES),
            "depths": list(HIGH_DIRECT_DEPTHS),
            "diagnostic_not_claim_confirmation": True,
            "outcome_not_preclaimed": True,
            "memory_policy": (
                "activation checkpointing remains disabled exactly as in p<=64; intended "
                "two-worker concurrency requires excluded A800 preflight, else one worker"
            ),
        },
        "restart_seeds": list(RESTART_SEEDS),
        "restart_count_per_cell": len(RESTART_SEEDS),
        "optimization": {
            "algorithm": "Adam",
            "learning_rate": LEARNING_RATE,
            "forward_evaluations_per_restart_including_endpoint": OBJECTIVE_EVALUATIONS,
            "gradient_updates_per_restart": OBJECTIVE_EVALUATIONS - 1,
            "independent_at_every_depth": True,
            "checkpoint_rule": "minimum target-blind expected cumulative D loss",
            "dtype": "complex64",
        },
        "embedded_C12_row": c12,
        "embedded_C12_row_sha256": common.sha256_json(c12),
        "embedded_service_independent_VF3": stage2,
        "precursor_replay_validation": source_manifest["precursor_replay_validation"],
        "task_count": len(tasks),
        "task_contracts_sha256": common.sha256_json([_task_contract(task) for task in tasks]),
        "post_completion_target_definition": {
            "target_cardinality": "all exactly minimum-energy feasible states",
            "domain": "jointly feasible native-shell states",
            "ranking": "minimum raw coherent objective",
            "scorer_only_after_global_completion": True,
        },
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "manifest_sha256": common.sha256_json(body)}
    root.mkdir(parents=True, exist_ok=True)
    (root / RESULT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    destination = _manifest_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace manifest {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_manifest(root: Path) -> dict[str, object]:
    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    cohort = manifest.get("cohort", {})
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("status")
        == "FROZEN_TARGET_BLIND_SEED10_EXTENSION_AND_DIRECT_DIAGNOSTICS"
        and manifest.get("implementation_revision") == IMPLEMENTATION_REVISION
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and manifest.get("frozen_source_sha256") == frozen_source_sha256()
        and tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS
        and manifest.get("task_count") == len(expected_tasks())
        and manifest.get("task_contracts_sha256")
        == common.sha256_json([_task_contract(task) for task in expected_tasks()])
        and manifest.get("target_constructed") is False,
        "invalid extension manifest",
    )
    common._require(
        tuple(cohort.get("source_table_seeds", ())) == SOURCE_TABLE_SEEDS
        and tuple(cohort.get("additional_table_seeds", ())) == ADDITIONAL_TABLE_SEEDS
        and tuple(cohort.get("all_ten_table_seeds", ())) == ALL_TABLE_SEEDS
        and cohort.get("tables_in_seed_order")
        == [coherent.coefficient_table_from_pcg64(seed) for seed in ALL_TABLE_SEEDS],
        "extension cohort changed",
    )
    stage2 = manifest.get("embedded_service_independent_VF3", {})
    common._require(
        manifest.get("embedded_C12_row_sha256")
        == common.sha256_json(manifest.get("embedded_C12_row"))
        and tuple(stage2.get("restart_seeds", ())) == RESTART_SEEDS
        and np.asarray(stage2.get("gamma_by_restart")).shape
        == (len(RESTART_SEEDS), source_campaign.F3_DEPTH),
        "invalid embedded precursors",
    )
    return manifest


def table(manifest: Mapping[str, object], seed: int) -> dict[str, object]:
    common._require(seed in ALL_TABLE_SEEDS, f"unknown seed {seed}")
    row = manifest["cohort"]["tables_in_seed_order"][ALL_TABLE_SEEDS.index(seed)]
    common._require(row == coherent.coefficient_table_from_pcg64(seed), "table changed")
    return row


def _make_screen(*, device: str, checkpointing: bool) -> engine.StructuralScreen:
    config = replace(
        engine.valid_config(),
        restarts=len(RESTART_SEEDS),
        initialization_seeds=RESTART_SEEDS,
        restart_batch=len(RESTART_SEEDS),
        device=device,
        dtype="complex64",
        learning_rate=LEARNING_RATE,
        activation_checkpointing=checkpointing,
    )
    return engine.StructuralScreen(config)


_RUNTIME_CACHE: dict[tuple[str, str, bool], source_campaign.RuntimeContext] = {}


def runtime_context(
    manifest: Mapping[str, object], device: str, *, checkpointing: bool
) -> source_campaign.RuntimeContext:
    key = (str(device), str(manifest["manifest_sha256"]), bool(checkpointing))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    screen = _make_screen(device=device, checkpointing=checkpointing)
    c12 = common.replay_c12(screen, manifest["embedded_C12_row"])
    stage2 = manifest["embedded_service_independent_VF3"]
    f_gamma = torch.as_tensor(
        stage2["gamma_by_restart"], dtype=screen.real_dtype, device=screen.device
    )
    f_beta = torch.as_tensor(
        stage2["beta_by_restart"], dtype=screen.real_dtype, device=screen.device
    )
    with torch.no_grad():
        f_loss, vf3 = variational._feasibility_energy(screen, c12)(f_gamma, f_beta)
    saved = torch.as_tensor(
        stage2["selected_feasibility_loss"],
        dtype=screen.real_dtype,
        device=screen.device,
    )
    common._require(
        torch.allclose(f_loss, saved, atol=VF3_REPLAY_ATOL, rtol=VF3_REPLAY_RTOL),
        "VF3 replay mismatch",
    )
    fixed = common.fixed_uniform_precursor(screen)
    native = screen.native[None, :].expand(len(RESTART_SEEDS), -1).clone()
    evidence = {
        "C12_state_sha256": common.state_sha256(c12),
        "VF3_saved_state_sha256": stage2["state_sha256"],
        "VF3_replay_state_sha256": common.state_sha256(vf3),
        "VF3_replay_loss_max_abs_error": float(torch.max(torch.abs(f_loss - saved)).cpu()),
        "VF3_replay_absolute_tolerance": VF3_REPLAY_ATOL,
        "VF3_replay_relative_tolerance": VF3_REPLAY_RTOL,
        "fixed_F32_state_sha256": common.state_sha256(fixed),
        "native_state_sha256": common.state_sha256(native),
        "activation_checkpointing": bool(checkpointing),
    }
    context = source_campaign.RuntimeContext(screen, c12, vf3, fixed, native, evidence)
    _RUNTIME_CACHE[key] = context
    return context


def energy_fn(
    family: str,
    depth: int,
    context: source_campaign.RuntimeContext,
    objective: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    common._require(family in FAMILIES and depth in ALL_RESULT_DEPTHS, "bad energy cell")
    screen = context.screen
    cumulative = screen.common + objective
    if family == source_campaign.FULL_LP:
        reference, phases, kernel = context.vf3_state, (objective,), "history"
    elif family == source_campaign.SKIP_F3:
        reference, phases, kernel = context.c12_state, (objective,), "history"
    elif family == source_campaign.FIXED_UNIFORM:
        reference, phases, kernel = context.fixed_f32_state, (objective,), "history"
    elif family == source_campaign.COLLAPSE_SEPARATE:
        reference = context.c12_state
        phases, kernel = (screen.final_mask.to(screen.real_dtype), objective), "history"
    elif family == source_campaign.DIRECT_SEPARATE:
        reference, phases, kernel = context.native_state, (screen.q, screen.c, objective), "block_xy"
    elif family == source_campaign.DIRECT_COMBINED:
        reference, phases, kernel = context.native_state, (cumulative,), "block_xy"
    elif family == source_campaign.WARM_COMBINED:
        reference, phases, kernel = context.c12_state, (cumulative,), "block_xy"
    elif family == source_campaign.NATIVE_PROJECTOR_SEPARATE:
        reference, phases, kernel = context.native_state, (screen.q, screen.c, objective), "history"
    else:  # pragma: no cover
        raise ValueError(family)

    def energy(gamma: torch.Tensor, beta: torch.Tensor):
        if kernel == "history":
            return objective_engine.history_circuit(
                screen, reference, phases, cumulative, gamma, beta, depth
            )
        return screen.block_energy(reference, phases, cumulative, gamma, beta, depth)

    return energy


def _optimize(
    energy: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    task_kind: str,
    family: str,
    depth: int,
    context: source_campaign.RuntimeContext,
    evaluations: int,
) -> engine.OptimizationResult:
    gamma_np, beta_np = initial_arrays(task_kind, family, depth)
    gamma = torch.as_tensor(
        gamma_np, dtype=context.screen.real_dtype, device=context.screen.device
    ).clone().requires_grad_(True)
    beta = torch.as_tensor(
        beta_np, dtype=context.screen.real_dtype, device=context.screen.device
    ).clone().requires_grad_(True)
    optimizer = torch.optim.Adam((gamma, beta), lr=LEARNING_RATE)
    best_energy = torch.full(
        (len(RESTART_SEEDS),), math.inf,
        dtype=context.screen.real_dtype, device=context.screen.device,
    )
    best_state: torch.Tensor | None = None
    best_gamma, best_beta = torch.zeros_like(gamma), torch.zeros_like(beta)
    best_eval = torch.zeros(len(RESTART_SEEDS), dtype=torch.int64, device=context.screen.device)
    trace: list[dict[str, object]] = []
    started = time.time()

    def inspect(values: torch.Tensor, states: torch.Tensor, evaluation: int, endpoint: bool) -> None:
        nonlocal best_energy, best_state, best_gamma, best_beta, best_eval
        detached = values.detach()
        improved = detached < best_energy
        if best_state is None:
            best_state = states.detach().clone()
        elif torch.any(improved):
            best_state[improved] = states.detach()[improved]
        best_energy = torch.where(improved, detached, best_energy)
        best_gamma[improved] = gamma.detach()[improved]
        best_beta[improved] = beta.detach()[improved]
        best_eval[improved] = evaluation
        if evaluation == 1 or endpoint or evaluation % context.screen.config.trace_every == 0:
            trace.append({
                "evaluation": evaluation,
                "post_update_endpoint": endpoint,
                "loss_by_restart": [float(value) for value in detached.cpu()],
            })

    for evaluation in range(1, evaluations):
        optimizer.zero_grad(set_to_none=True)
        values, states = energy(gamma, beta)
        inspect(values, states, evaluation, False)
        values.sum().backward()
        if trace and trace[-1]["evaluation"] == evaluation:
            with torch.no_grad():
                gradient_l2 = torch.sqrt(
                    torch.sum(gamma.grad * gamma.grad, dim=1)
                    + torch.sum(beta.grad * beta.grad, dim=1)
                )
            trace[-1]["gradient_l2_by_restart"] = [
                float(value) for value in gradient_l2.detach().cpu()
            ]
        optimizer.step()
    with torch.no_grad():
        values, states = energy(gamma, beta)
        inspect(values, states, evaluations, True)
    if best_state is None:  # pragma: no cover
        raise RuntimeError("optimizer produced no state")
    return engine.OptimizationResult(
        label=method_id(task_kind, family, depth),
        state=best_state,
        energy=best_energy,
        gamma=best_gamma,
        beta=best_beta,
        best_evaluation=best_eval,
        evaluations_per_restart=evaluations,
        updates_per_restart=evaluations - 1,
        endpoint_evaluations_per_restart=1,
        trace=trace,
        elapsed_sec=time.time() - started,
    )


def _trace_ok(result: Mapping[str, object], evaluations: int) -> bool:
    trace = result.get("optimizer_trace")
    if not isinstance(trace, list):
        return False
    every = int(engine.valid_config().trace_every)
    expected = [value for value in range(1, evaluations) if value == 1 or value % every == 0]
    expected.append(evaluations)
    rows_ok = True
    for index, row in enumerate(trace):
        if not isinstance(row, Mapping):
            rows_ok = False
            break
        endpoint = index == len(trace) - 1
        if (row.get("post_update_endpoint") is True) != endpoint:
            rows_ok = False
            break
        losses = np.asarray(row.get("loss_by_restart"), dtype=np.float64)
        if losses.shape != (len(RESTART_SEEDS),) or not np.all(np.isfinite(losses)):
            rows_ok = False
            break
        if not endpoint:
            gradients = np.asarray(
                row.get("gradient_l2_by_restart"), dtype=np.float64
            )
            if (
                gradients.shape != (len(RESTART_SEEDS),)
                or not np.all(np.isfinite(gradients))
            ):
                rows_ok = False
                break
    return (
        [row.get("evaluation") for row in trace] == expected
        and rows_ok
        and result.get("optimizer_trace_sha256") == common.sha256_json(trace)
    )


def validate_result(
    manifest: Mapping[str, object],
    result: Mapping[str, object],
    task_kind: str,
    seed: int,
    family: str,
    depth: int,
) -> None:
    common._require(task_allowed(task_kind, seed, family, depth), "unexpected result")
    spec = source_campaign.family_specs()[family]
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    gamma = np.asarray(result.get("gamma_by_restart"), dtype=np.float64)
    beta = np.asarray(result.get("beta_by_restart"), dtype=np.float64)
    selected = np.asarray(result.get("selected_expected_loss"), dtype=np.float64)
    initial = np.asarray(result.get("initial_expected_loss"), dtype=np.float64)
    final = np.asarray(result.get("final_expected_loss"), dtype=np.float64)
    best_evaluation = np.asarray(
        result.get("best_evaluation_by_restart"), dtype=np.float64
    )
    trace = result.get("optimizer_trace")
    trace_boundary_ok = (
        isinstance(trace, list)
        and len(trace) >= 2
        and initial.shape == final.shape == (len(RESTART_SEEDS),)
        and np.all(np.isfinite(initial))
        and np.all(np.isfinite(final))
        and np.allclose(
            initial,
            np.asarray(trace[0].get("loss_by_restart"), dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        and np.allclose(
            final,
            np.asarray(trace[-1].get("loss_by_restart"), dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
    )
    metric_expected_loss = np.asarray(
        result.get("metrics", {}).get("expected_cumulative_loss"), dtype=np.float64
    )
    metric_state_norm = np.asarray(
        result.get("metrics", {}).get("state_norm"), dtype=np.float64
    )
    probability_metrics = [
        np.asarray(result.get("metrics", {}).get(key), dtype=np.float64)
        for key in ("quota_mass", "conflict_mass", "final_mass")
    ]
    checkpoint_consistency_ok = (
        selected.shape == metric_expected_loss.shape == (len(RESTART_SEEDS),)
        and np.allclose(selected, metric_expected_loss, atol=2e-5, rtol=2e-5)
        and np.all(selected <= initial + 2e-5)
        and np.all(selected <= final + 2e-5)
    )
    expected_table = table(manifest, seed)
    checkpointing = False
    common._require(
        result.get("schema") == f"{SCHEMA}-optimizer-result"
        and result.get("status") == "COMPLETED_TARGET_BLIND_EXTENSION_CELL"
        and result.get("manifest_sha256") == manifest["manifest_sha256"]
        and result.get("task_kind") == task_kind
        and result.get("table_seed") == seed
        and result.get("table_id") == table_id(seed)
        and result.get("coefficient_table_sha256")
        == expected_table["service_cost_compact_json_sha256"]
        and result.get("family") == family
        and result.get("depth") == depth
        and result.get("method_id") == method_id(task_kind, family, depth)
        and tuple(result.get("restart_seeds", ())) == RESTART_SEEDS
        and result.get("resource") == resource(family, depth)
        and result.get("initialization_policy") == initialization_policy(task_kind, family, depth)
        and result.get("initial_angles_sha256") == initial_angles_sha256(task_kind, family, depth)
        and result.get("forward_evaluations_per_restart") == OBJECTIVE_EVALUATIONS
        and result.get("gradient_updates_per_restart") == OBJECTIVE_EVALUATIONS - 1
        and result.get("endpoint_evaluations_per_restart") == 1
        and best_evaluation.shape == (len(RESTART_SEEDS),)
        and np.all(np.isfinite(best_evaluation))
        and np.all(best_evaluation == np.floor(best_evaluation))
        and np.all((best_evaluation >= 1) & (best_evaluation <= OBJECTIVE_EVALUATIONS))
        and selected.shape == (len(RESTART_SEEDS),)
        and np.all(np.isfinite(selected))
        and trace_boundary_ok
        and checkpoint_consistency_ok
        and gamma.shape == (len(RESTART_SEEDS), spec.gamma_multiplier * depth)
        and beta.shape == (len(RESTART_SEEDS), depth)
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(beta))
        and source_campaign._metrics_ok(result.get("metrics", {}))
        and metric_state_norm.shape == (len(RESTART_SEEDS),)
        and np.allclose(metric_state_norm, 1.0, atol=2e-5, rtol=2e-5)
        and all(
            values.shape == (len(RESTART_SEEDS),)
            and np.all((values >= -2e-5) & (values <= 1.0 + 2e-5))
            for values in probability_metrics
        )
        and _trace_ok(result, OBJECTIVE_EVALUATIONS)
        and result.get("activation_checkpointing") is checkpointing
        and result.get("checkpoint_selected_by_target_blind_loss_only") is True
        and result.get("independent_depth_optimization_no_continuation") is True
        and result.get("target_artifact_loaded") is False
        and result.get("target_constructed") is False
        and result.get("target_probability_computed") is False
        and result.get("result_sha256") == common.sha256_json(body),
        f"invalid extension result {task_kind}/{seed}/{family}/p{depth}",
    )


def run_cell(
    root: Path,
    task_kind: str,
    seed: int,
    family: str,
    depth: int,
    *,
    device: str,
    evaluations: int = OBJECTIVE_EVALUATIONS,
    production: bool = True,
) -> dict[str, object]:
    manifest = validated_manifest(root)
    common._require(task_allowed(task_kind, seed, family, depth), "invalid task")
    common._require(
        evaluations == OBJECTIVE_EVALUATIONS or not production,
        "production must use the frozen evaluation budget",
    )
    destination = result_path(root, task_kind, seed, family, depth)
    if production and destination.exists():
        existing = common.load_json(destination)
        validate_result(manifest, existing, task_kind, seed, family, depth)
        return existing
    checkpointing = False
    context = runtime_context(manifest, device, checkpointing=checkpointing)
    table_row = table(manifest, seed)
    objective = discovery.objective_tensor(context.screen, table_row)
    if context.screen.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.screen.device)
    optimized = _optimize(
        energy_fn(family, depth, context, objective),
        task_kind, family, depth, context, evaluations,
    )
    metrics = discovery._metrics(context.screen, optimized.state, objective)
    spec = source_campaign.family_specs()[family]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-optimizer-result",
        "status": "COMPLETED_TARGET_BLIND_EXTENSION_CELL",
        "manifest_sha256": manifest["manifest_sha256"],
        "task_kind": task_kind,
        "table_seed": seed,
        "table_id": table_id(seed),
        "coefficient_table_sha256": table_row["service_cost_compact_json_sha256"],
        "family": family,
        "method_id": method_id(task_kind, family, depth),
        "depth": depth,
        "kernel": spec.kernel,
        "reference": spec.reference,
        "phase_names": list(spec.phases),
        "mixer": spec.mixer,
        "restart_seeds": list(RESTART_SEEDS),
        "initialization_policy": initialization_policy(task_kind, family, depth),
        "initial_angles_sha256": initial_angles_sha256(task_kind, family, depth),
        "resource": resource(family, depth),
        "forward_evaluations_per_restart": optimized.evaluations_per_restart,
        "gradient_updates_per_restart": optimized.updates_per_restart,
        "endpoint_evaluations_per_restart": optimized.endpoint_evaluations_per_restart,
        "best_evaluation_by_restart": [int(value) for value in optimized.best_evaluation.cpu()],
        "selected_expected_loss": [float(value) for value in optimized.energy.cpu()],
        "initial_expected_loss": list(optimized.trace[0]["loss_by_restart"]),
        "final_expected_loss": list(optimized.trace[-1]["loss_by_restart"]),
        "gamma_by_restart": optimized.gamma.cpu().tolist(),
        "beta_by_restart": optimized.beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(optimized.state),
        "optimizer_trace": optimized.trace,
        "optimizer_trace_sha256": common.sha256_json(optimized.trace),
        "precursor_replay_evidence": context.replay_evidence,
        "elapsed_sec": optimized.elapsed_sec,
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(context.screen.device))
            if context.screen.device.type == "cuda" else None
        ),
        "activation_checkpointing": checkpointing,
        "checkpoint_selected_by_target_blind_loss_only": True,
        "independent_depth_optimization_no_continuation": True,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
        "execution_environment": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "requested_device": device,
            "resolved_device": str(context.screen.device),
            "cuda_device_name": (
                torch.cuda.get_device_name(context.screen.device)
                if context.screen.device.type == "cuda" else None
            ),
        },
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    if production:
        destination.parent.mkdir(parents=True, exist_ok=True)
        common.atomic_write_json(destination, payload)
        validate_result(manifest, payload, task_kind, seed, family, depth)
    return payload


def _estimated_seconds(task: Task) -> float:
    baseline = source_campaign.family_specs()[task.family].p64_estimated_seconds
    multiplier = task.depth / 64.0
    return max(20.0, float(baseline) * multiplier)


def plan_shards(root: Path) -> dict[str, object]:
    manifest = validated_manifest(root)
    rows = [{**_task_contract(task), "estimated_seconds": _estimated_seconds(task)} for task in expected_tasks()]
    standard = [row for row in rows if row["task_kind"] == TASK_CURVE_EXTENSION]
    separate = [
        row for row in rows
        if row["task_kind"] == TASK_HIGH_DIRECT
        and row["family"] == source_campaign.DIRECT_SEPARATE
    ]
    combined = [
        row for row in rows
        if row["task_kind"] == TASK_HIGH_DIRECT
        and row["family"] == source_campaign.DIRECT_COMBINED
    ]
    standard.sort(key=lambda row: (-float(row["estimated_seconds"]), str(row["family"]), -int(row["depth"]), int(row["table_seed"])))
    separate.sort(key=lambda row: int(row["table_seed"]))
    combined.sort(key=lambda row: int(row["table_seed"]))
    shards: list[list[dict[str, object]]] = [[], []]
    loads = [0.0, 0.0]
    for row in standard:
        eligible = [index for index in range(2) if len(shards[index]) < 196]
        index = min(eligible, key=lambda value: (loads[value], len(shards[value]), value))
        shards[index].append(row)
        loads[index] += float(row["estimated_seconds"])
    common._require([len(shard) for shard in shards] == [196, 196], "standard 196/196 split failed")
    for index, selected in enumerate((separate[:6], separate[6:])):
        shards[index].extend(selected)
        loads[index] += sum(float(row["estimated_seconds"]) for row in selected)
    for index, selected in enumerate((combined[:4], combined[4:])):
        shards[index].extend(selected)
        loads[index] += sum(float(row["estimated_seconds"]) for row in selected)
    for shard in shards:
        shard.sort(
            key=lambda row: (
                0 if row["task_kind"] == TASK_CURVE_EXTENSION else 1,
                -float(row["estimated_seconds"]), str(row["family"]),
                int(row["table_seed"]),
            )
        )
    flat = [
        Task(str(row["task_kind"]), int(row["table_seed"]), str(row["family"]), int(row["depth"]))
        for shard in shards for row in shard
    ]
    common._require(
        len(flat) == len(expected_tasks())
        and set(flat) == set(expected_tasks())
        and [len(shard) for shard in shards] == [206, 206],
        "bad stage-aware shard union",
    )
    shard_rows = [
        {
            "shard_index": index,
            "platform_label": ("gpu_worker_1", "gpu_worker_2")[index],
            "task_count": len(shards[index]),
            "task_count_by_stage": {
                "standard": sum(
                    row["task_kind"] == TASK_CURVE_EXTENSION for row in shards[index]
                ),
                "p96_direct_separate": sum(
                    row["task_kind"] == TASK_HIGH_DIRECT
                    and row["family"] == source_campaign.DIRECT_SEPARATE
                    for row in shards[index]
                ),
                "p96_direct_combined": sum(
                    row["task_kind"] == TASK_HIGH_DIRECT
                    and row["family"] == source_campaign.DIRECT_COMBINED
                    for row in shards[index]
                ),
            },
            "estimated_serial_seconds": loads[index],
            "recommended_concurrent_workers": 2,
            "high_depth_two_worker_preflight_required": True,
            "tasks_in_execution_order": shards[index],
        }
        for index in range(2)
    ]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-two-pod-plan",
        "status": "FROZEN_TWO_DISJOINT_A800_EXTENSION_QUEUES_NOT_LAUNCHED",
        "manifest_sha256": manifest["manifest_sha256"],
        "assignment_rule": (
            "standard LPT 196/196; p96 cross-split q0 separate6+combined4, "
            "q1 separate4+combined6 to minimize two-worker stage waves"
        ),
        "optimizer_task_count": len(flat),
        "shards": shard_rows,
        "exact_expected_task_set_complete": True,
        "duplicate_task_count": 0,
        "optimization_launched": False,
    }
    payload = {**body, "plan_sha256": common.sha256_json(body)}
    destination = _plan_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace plan {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_plan(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = validated_manifest(root)
    plan = common.load_json(_plan_path(root))
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    tasks = [
        Task(str(row["task_kind"]), int(row["table_seed"]), str(row["family"]), int(row["depth"]))
        for shard in plan.get("shards", ()) for row in shard.get("tasks_in_execution_order", ())
    ]
    common._require(
        plan.get("schema") == f"{SCHEMA}-two-pod-plan"
        and plan.get("manifest_sha256") == manifest["manifest_sha256"]
        and plan.get("plan_sha256") == common.sha256_json(body)
        and len(plan.get("shards", ())) == 2
        and len(tasks) == len(expected_tasks())
        and set(tasks) == set(expected_tasks())
        and all(
            sum(int(value) for value in shard.get("task_count_by_stage", {}).values())
            == int(shard.get("task_count", -1))
            for shard in plan.get("shards", ())
        )
        and [
            shard.get("task_count_by_stage") for shard in plan.get("shards", ())
        ]
        == [
            {"standard": 196, "p96_direct_separate": 6, "p96_direct_combined": 4},
            {"standard": 196, "p96_direct_separate": 4, "p96_direct_combined": 6},
        ],
        "invalid two-pod extension plan",
    )
    return manifest, plan


def seal(root: Path) -> dict[str, object]:
    manifest, plan = validated_plan(root)
    completed: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    for task in expected_tasks():
        path = result_path(root, task.task_kind, task.table_seed, task.family, task.depth)
        expected_paths.add(path.resolve())
        result = common.load_json(path)
        validate_result(manifest, result, task.task_kind, task.table_seed, task.family, task.depth)
        completed.append({**_task_contract(task), "result_sha256": result["result_sha256"]})
    actual_paths = {path.resolve() for path in (root / RESULT_DIRECTORY).glob("*/*.json")}
    common._require(actual_paths == expected_paths, "extension result set is not exact")
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-global-seal",
        "status": "SEALED_COMPLETE_TARGET_BLIND_SEED10_EXTENSION",
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "frozen_source_sha256": manifest["frozen_source_sha256"],
        "complete_optimizer_result_count": len(completed),
        "expected_optimizer_result_count": len(expected_tasks()),
        "completed_cells": completed,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "seal_sha256": common.sha256_json(body)}
    destination = root / GLOBAL_SEAL_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace seal {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def smoke(
    root: Path, *, device: str, family: str, depth: int, evaluations: int
) -> dict[str, object]:
    common._require(family in HIGH_DIRECT_FAMILIES and depth in HIGH_DIRECT_DEPTHS, "bad smoke cell")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    payload = run_cell(
        root, TASK_HIGH_DIRECT, ALL_TABLE_SEEDS[0], family, depth,
        device=device, evaluations=evaluations, production=False,
    )
    trace = payload["optimizer_trace"]
    selected = np.asarray(payload["selected_expected_loss"], dtype=np.float64)
    gamma = np.asarray(payload["gamma_by_restart"], dtype=np.float64)
    beta = np.asarray(payload["beta_by_restart"], dtype=np.float64)
    metric_arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["metrics"].items()
    }
    finite_audit = {
        "expected_evaluations_observed": [row["evaluation"] for row in trace]
        == [1, evaluations],
        "all_trace_losses_finite": all(
            np.all(np.isfinite(np.asarray(row["loss_by_restart"], dtype=np.float64)))
            for row in trace
        ),
        "nonendpoint_gradient_finite": bool(
            np.asarray(trace[0].get("gradient_l2_by_restart"), dtype=np.float64).shape
            == (len(RESTART_SEEDS),)
            and np.all(
                np.isfinite(
                    np.asarray(trace[0].get("gradient_l2_by_restart"), dtype=np.float64)
                )
            )
        ),
        "selected_loss_finite": selected.shape == (len(RESTART_SEEDS),)
        and bool(np.all(np.isfinite(selected))),
        "angles_finite": bool(np.all(np.isfinite(gamma)) and np.all(np.isfinite(beta))),
        "all_metric_vectors_finite": all(
            values.shape == (len(RESTART_SEEDS),) and np.all(np.isfinite(values))
            for values in metric_arrays.values()
        ),
        "state_norm_close_to_one": bool(
            np.allclose(metric_arrays["state_norm"], 1.0, atol=2e-5, rtol=2e-5)
        ),
    }
    common._require(all(finite_audit.values()), "high-depth smoke finite-output audit failed")
    scientific_evidence = {
        "selected_loss": selected.tolist(),
        "initial_loss": list(trace[0]["loss_by_restart"]),
        "endpoint_loss": list(trace[-1]["loss_by_restart"]),
        "gradient_l2": list(trace[0]["gradient_l2_by_restart"]),
        "state_norm": metric_arrays["state_norm"].tolist(),
        "gamma_l2": np.sqrt(np.sum(gamma * gamma, axis=1)).tolist(),
        "beta_l2": np.sqrt(np.sum(beta * beta, axis=1)).tolist(),
        "metric_vectors": {
            key: values.tolist() for key, values in metric_arrays.items()
        },
    }
    return {
        "status": "EXCLUDED_HIGH_DEPTH_DIRECT_SMOKE_PASS",
        "production_result_written": False,
        "family": family,
        "depth": depth,
        "forward_evaluations": evaluations,
        "restart_batch": len(RESTART_SEEDS),
        "finite_output_audit": finite_audit,
        "scientific_evidence": scientific_evidence,
        "elapsed_sec": payload["elapsed_sec"],
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device)))
            if device.startswith("cuda") else None
        ),
        "cuda_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(torch.device(device)))
            if device.startswith("cuda") else None
        ),
    }


def estimate() -> dict[str, object]:
    tasks = expected_tasks()
    serial = sum(_estimated_seconds(task) for task in tasks)
    return {
        "additional_standard_curve_task_count": sum(task.task_kind == TASK_CURVE_EXTENSION for task in tasks),
        "all10_high_direct_task_count": sum(task.task_kind == TASK_HIGH_DIRECT for task in tasks),
        "total_optimizer_task_count": len(tasks),
        "projected_serial_GPU_hours": serial / 3600.0,
        "projected_two_A800_two_worker_wall_hours": serial / (4 * 3600.0),
        "warning": "p96 calibration is conservative; excluded two-worker A800 smoke is mandatory",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--source-campaign-root", type=Path, required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--source-campaign-root", type=Path, required=True)
    command = sub.add_parser("run-cell")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--task-kind", choices=TASK_KINDS, required=True)
    command.add_argument("--table-seed", type=int, choices=ALL_TABLE_SEEDS, required=True)
    command.add_argument("--family", choices=FAMILIES, required=True)
    command.add_argument("--depth", type=int, choices=ALL_RESULT_DEPTHS, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("plan-shards")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--family", choices=HIGH_DIRECT_FAMILIES, default=source_campaign.DIRECT_SEPARATE)
    command.add_argument("--depth", type=int, choices=HIGH_DIRECT_DEPTHS, default=max(HIGH_DIRECT_DEPTHS))
    command.add_argument("--evaluations", type=int, default=2)
    command.add_argument("--device", default="cuda")
    sub.add_parser("estimate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.source_campaign_root)
    elif args.command == "prepare":
        freeze(args.campaign_root, args.source_campaign_root)
        payload = plan_shards(args.campaign_root)
    elif args.command == "run-cell":
        payload = run_cell(
            args.campaign_root, args.task_kind, args.table_seed, args.family,
            args.depth, device=args.device,
        )
    elif args.command == "plan-shards":
        payload = plan_shards(args.campaign_root)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    elif args.command == "smoke":
        payload = smoke(
            args.campaign_root, device=args.device, family=args.family,
            depth=args.depth, evaluations=args.evaluations,
        )
    elif args.command == "estimate":
        payload = estimate()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload.get("status", payload))


if __name__ == "__main__":
    main()
