#!/usr/bin/env python3
"""Target-blind coherent-IM3 depth sweep on three fresh service tables.

The campaign is an isolated, post-confirmation characterization study.  It
freezes eight scientifically distinct curve families on a common log2 depth
grid.  Every positive-depth cell is optimized independently for the same
endpoint-inclusive budget.  Depth zero is a derived, family-specific precursor
anchor and is deliberately absent from the GPU task plan.

Target construction and success-probability scoring live only in
``targeted_coherent_im3_depth_sweep_score.py`` and require a complete global
optimizer seal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np
import torch

import stage3_objective_common as objective_engine
import target_free_structural_screen as engine
import targeted_coherent_im3_fixed_uniform_control as fixed_control
import targeted_coherent_im3_heldout as prior_campaign
import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_screen as discovery
import targeted_coherent_im3_variational_f3 as variational
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-depth-sweep-eight-family-v1"
IMPLEMENTATION_REVISION = "fresh3-independent-log2-depth-r1-20260818"
MANIFEST_NAME = "targeted_coherent_im3_depth_sweep_manifest.json"
PLAN_NAME = "targeted_coherent_im3_depth_sweep_two_pod_plan.json"
RESULT_DIRECTORY = "targeted_coherent_im3_depth_sweep_results"
GLOBAL_SEAL_NAME = "targeted_coherent_im3_depth_sweep_global_seal.json"
QUEUE_SOURCE_NAME = "targeted_coherent_im3_depth_sweep_queue.py"
SCORER_SOURCE_NAME = "targeted_coherent_im3_depth_sweep_score.py"
PROTOCOL_SOURCE_NAME = "COHERENT_IM3_DEPTH_SWEEP_PROTOCOL_20260818.md"

# Fresh PCG64 streams.  They are generated once and accepted without filtering.
TABLE_SEEDS = tuple(range(2_026_081_801, 2_026_081_804))
RESTART_SEEDS = tuple(range(2_026_081_100, 2_026_081_104))
DEPTHS = (0, 1, 2, 4, 8, 16, 32, 64)
POSITIVE_DEPTHS = DEPTHS[1:]

OBJECTIVE_EVALUATIONS = 2_400
LEARNING_RATE = 0.035
ACTIVATION_CHECKPOINTING = False
VF3_REPLAY_ATOL = 2e-5
VF3_REPLAY_RTOL = 2e-5
OPTIMIZED_REPLAY_ATOL = 2e-5
OPTIMIZED_REPLAY_RTOL = 2e-5

# The order is paper-facing and frozen.  Same-oracle joint is omitted because
# the separate-phase row is the stronger/nonredundant collapse.  The old exact
# F3 row is omitted because it was numerically near-identical to VF3 and does
# not replace the decisive removal ablation.
FULL_LP = "full_C12_VF3_O"
SKIP_F3 = "ablation_C12_skipF3_O"
FIXED_UNIFORM = "control_native_fixedF32_O"
COLLAPSE_SEPARATE = "ablation_same_oracle_separate_F_O_history"
DIRECT_SEPARATE = "standard_native_direct_separate_Q_C_O_blockXY"
DIRECT_COMBINED = "standard_native_direct_combined_D_blockXY"
WARM_COMBINED = "standard_warm_C12_combined_D_blockXY"
NATIVE_PROJECTOR_SEPARATE = (
    "standard_native_reference_projector_separate_Q_C_O"
)
FAMILIES = (
    FULL_LP,
    SKIP_F3,
    FIXED_UNIFORM,
    COLLAPSE_SEPARATE,
    DIRECT_SEPARATE,
    DIRECT_COMBINED,
    WARM_COMBINED,
    NATIVE_PROJECTOR_SEPARATE,
)

STREAMS = {
    FULL_LP: 95_001,
    # Identical initialization streams pair the clean Stage-2 removal with LP.
    SKIP_F3: 95_001,
    FIXED_UNIFORM: 96_032,
    COLLAPSE_SEPARATE: 95_003,
    DIRECT_SEPARATE: 95_064,
    DIRECT_COMBINED: 97_064,
    WARM_COMBINED: 97_141,
    NATIVE_PROJECTOR_SEPARATE: 97_224,
}

PREP_RU = int(engine.PREP_RU)
XY_RU = int(engine.XY_RU)
SELECTIVE_RU = int(engine.SELECTIVE_RU)
C12_DEPTH = 12
F3_DEPTH = int(variational.P_F)
F32_DEPTH = int(fixed_control.FEASIBILITY_DEPTH)
Q_RU = 120
C_NATIVE_RU = 30
C_PAULI_RU = 30
QC_UNION_RU = int(coherent.support_ledger()["Q_union_C_RU"])
QCO_UNION_RU = int(coherent.support_ledger()["Q_union_C_union_O_RU"])
SEPARATE_QC_PLUS_O_RU = int(coherent.support_ledger()["separate_QC_plus_O_RU"])
O_RU = int(coherent.support_ledger()["objective_RU"])


@dataclass(frozen=True)
class FamilySpec:
    family: str
    kernel: str
    reference: str
    phases: tuple[str, ...]
    mixer: str
    gamma_multiplier: int
    stream: int
    p64_estimated_seconds: float


def family_specs() -> dict[str, FamilySpec]:
    rows = (
        FamilySpec(
            FULL_LP, "history", "saved_C12_then_saved_VF3", ("O",),
            "learned_projector_about_VF3", 1, STREAMS[FULL_LP], 586.0,
        ),
        FamilySpec(
            SKIP_F3, "history", "saved_C12", ("O",),
            "learned_projector_about_C12", 1, STREAMS[SKIP_F3], 586.0,
        ),
        FamilySpec(
            FIXED_UNIFORM, "history", "native_then_fixed_F32", ("O",),
            "learned_projector_about_fixed_F32", 1,
            STREAMS[FIXED_UNIFORM], 581.0,
        ),
        FamilySpec(
            COLLAPSE_SEPARATE, "history", "saved_C12", ("F_indicator", "O"),
            "learned_projector_about_C12", 2,
            STREAMS[COLLAPSE_SEPARATE], 744.0,
        ),
        FamilySpec(
            DIRECT_SEPARATE, "block_xy", "native_uniform", ("Q", "C", "O"),
            "complete_block_XY", 3, STREAMS[DIRECT_SEPARATE], 10_765.0,
        ),
        FamilySpec(
            DIRECT_COMBINED, "block_xy", "native_uniform", ("D",),
            "complete_block_XY", 1, STREAMS[DIRECT_COMBINED], 10_486.0,
        ),
        FamilySpec(
            WARM_COMBINED, "block_xy", "saved_C12", ("D",),
            "complete_block_XY", 1, STREAMS[WARM_COMBINED], 10_469.0,
        ),
        FamilySpec(
            NATIVE_PROJECTOR_SEPARATE, "history", "native_uniform",
            ("Q", "C", "O"), "native_reference_rank_one_projector", 3,
            STREAMS[NATIVE_PROJECTOR_SEPARATE], 889.0,
        ),
    )
    return {row.family: row for row in rows}


def table_id(seed: int) -> str:
    return f"pcg64_{int(seed)}"


def method_id(family: str, depth: int) -> str:
    common._require(family in FAMILIES, f"unknown depth-sweep family {family}")
    common._require(depth in DEPTHS, f"unknown depth {depth}")
    return f"coherent_im3_depth_sweep__{family}__p{int(depth)}"


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_source_sha256() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = (
        Path(__file__).name,
        QUEUE_SOURCE_NAME,
        SCORER_SOURCE_NAME,
        PROTOCOL_SOURCE_NAME,
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
    paths = [source_root / name for name in names]
    for path in paths:
        common._require(path.is_file(), f"missing frozen depth-sweep source: {path}")
    return {path.name: _file_sha256(path) for path in paths}


def _plan_path(root: Path) -> Path:
    return root / PLAN_NAME


def result_path(root: Path, seed: int, family: str, depth: int) -> Path:
    common._require(depth > 0, "p0 is a derived anchor and has no result file")
    return root / RESULT_DIRECTORY / table_id(seed) / f"{method_id(family, depth)}.json"


def _history_ru(prior: int, depth: int, phase_ru: int) -> int:
    return int((1 + 2 * depth) * prior + depth * (phase_ru + SELECTIVE_RU))


def _c12_ru(c_ru: int) -> int:
    return int(PREP_RU + C12_DEPTH * (c_ru + XY_RU))


def _vf3_ru(c_ru: int) -> int:
    return _history_ru(_c12_ru(c_ru), F3_DEPTH, QC_UNION_RU)


def _fixed_f32_ru() -> int:
    return int(
        PREP_RU
        + F32_DEPTH * (QC_UNION_RU + 2 * PREP_RU + SELECTIVE_RU)
    )


def resource(family: str, depth: int) -> dict[str, int]:
    """Exact per-shot replay RU, including the terminal O measurement."""

    common._require(family in FAMILIES and depth in DEPTHS, "invalid resource cell")

    def one(c_ru: int) -> int:
        c12 = _c12_ru(c_ru)
        if family == FULL_LP:
            return _history_ru(_vf3_ru(c_ru), depth, O_RU) + O_RU
        if family == SKIP_F3:
            return _history_ru(c12, depth, O_RU) + O_RU
        if family == FIXED_UNIFORM:
            return _history_ru(_fixed_f32_ru(), depth, O_RU) + O_RU
        if family == COLLAPSE_SEPARATE:
            return _history_ru(c12, depth, SEPARATE_QC_PLUS_O_RU) + O_RU
        if family == DIRECT_SEPARATE:
            return (
                PREP_RU
                + depth * (Q_RU + c_ru + O_RU + XY_RU)
                + O_RU
            )
        if family == DIRECT_COMBINED:
            return PREP_RU + depth * (QCO_UNION_RU + XY_RU) + O_RU
        if family == WARM_COMBINED:
            return c12 + depth * (QCO_UNION_RU + XY_RU) + O_RU
        if family == NATIVE_PROJECTOR_SEPARATE:
            return (
                PREP_RU
                + depth
                * (Q_RU + c_ru + O_RU + 2 * PREP_RU + SELECTIVE_RU)
                + O_RU
            )
        raise ValueError(family)  # pragma: no cover

    native = one(C_NATIVE_RU)
    pauli = one(C_PAULI_RU)
    # Combined phases and the fixed exact indicator do not expose a separate C
    # implementation, so the convention can differ only through a C12 replay.
    if family in (FIXED_UNIFORM, DIRECT_COMBINED):
        pauli = native
    return {
        "native_C": int(native),
        "uniform_Pauli_C_sensitivity": int(pauli),
    }






WARM_PERTURBATION = 1e-3


def _initial_angle_arrays(
    family: str, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen, cross-depth prefix-consistent initial angles.

    Gamma is sampled as ``(phase, max_depth)`` and then sliced along depth,
    rather than sampling a fresh flattened vector per cell.  Consequently the
    first p layers of every phase are byte-identical across deeper points.
    Warm-C12 is initialized near the identity with deterministic Rademacher
    perturbations; all other families use the established full-range uniform
    initialization.
    """

    common._require(family in FAMILIES and depth in POSITIVE_DEPTHS, "invalid initialization cell")
    spec = family_specs()[family]
    gamma_rows: list[np.ndarray] = []
    beta_rows: list[np.ndarray] = []
    max_depth = max(DEPTHS)
    for seed in RESTART_SEEDS:
        gamma_rng = np.random.default_rng(
            int(seed) + 1_000_003 * int(2 * spec.stream)
        )
        beta_rng = np.random.default_rng(
            int(seed) + 1_000_003 * int(2 * spec.stream + 1)
        )
        if family == WARM_COMBINED:
            gamma_max = WARM_PERTURBATION * (
                2 * gamma_rng.integers(
                    0, 2, size=(spec.gamma_multiplier, max_depth), dtype=np.int8
                ).astype(np.float64)
                - 1.0
            )
            beta_max = WARM_PERTURBATION * (
                2 * beta_rng.integers(0, 2, size=max_depth, dtype=np.int8).astype(np.float64)
                - 1.0
            )
        else:
            gamma_max = gamma_rng.uniform(
                -math.pi, math.pi, size=(spec.gamma_multiplier, max_depth)
            )
            beta_max = beta_rng.uniform(-math.pi, math.pi, size=max_depth)
        gamma_rows.append(gamma_max[:, :depth].reshape(-1))
        beta_rows.append(beta_max[:depth])
    gamma = np.asarray(gamma_rows, dtype=np.float64)
    beta = np.asarray(beta_rows, dtype=np.float64)
    common._require(
        gamma.shape == (len(RESTART_SEEDS), spec.gamma_multiplier * depth)
        and beta.shape == (len(RESTART_SEEDS), depth)
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(beta)),
        "invalid frozen initialization arrays",
    )
    return gamma, beta


def initial_angles_sha256(family: str, depth: int) -> str:
    gamma, beta = _initial_angle_arrays(family, depth)
    return common.sha256_json(
        {"gamma_by_restart": gamma.tolist(), "beta_by_restart": beta.tolist()}
    )


def initialization_policy(family: str) -> dict[str, object]:
    base: dict[str, object] = {
        "independent_at_every_positive_depth": True,
        "continuation_or_angle_transfer": False,
        "prefix_consistent_through_max_depth": max(DEPTHS),
        "gamma_layout": "phase-major; slice first p layers independently for every phase",
        "rng": "numpy PCG64(seed + 1000003*derived_stream)",
    }
    if family == WARM_COMBINED:
        return {
            **base,
            "distribution": "deterministic Rademacher signs times 1e-3",
            "perturbation_magnitude": WARM_PERTURBATION,
            "identity_preserving_warm_start": True,
        }
    return {
        **base,
        "distribution": "uniform[-pi,pi]",
        "identity_preserving_warm_start": False,
    }


def _optimize_independent(
    energy: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    family: str,
    depth: int,
    evaluations: int,
    label: str,
    context: "RuntimeContext",
) -> engine.OptimizationResult:
    """Engine-equivalent Adam loop with immutable supplied initial arrays."""

    if evaluations < 2:
        raise ValueError("at least one update plus one endpoint is required")
    gamma_np, beta_np = _initial_angle_arrays(family, depth)
    gamma = torch.as_tensor(
        gamma_np, dtype=context.screen.real_dtype, device=context.screen.device
    ).clone().requires_grad_(True)
    beta = torch.as_tensor(
        beta_np, dtype=context.screen.real_dtype, device=context.screen.device
    ).clone().requires_grad_(True)
    optimizer = torch.optim.Adam((gamma, beta), lr=LEARNING_RATE)
    restarts = len(RESTART_SEEDS)
    best_energy = torch.full(
        (restarts,), math.inf,
        dtype=context.screen.real_dtype,
        device=context.screen.device,
    )
    best_state: torch.Tensor | None = None
    best_gamma = torch.zeros_like(gamma)
    best_beta = torch.zeros_like(beta)
    best_eval = torch.zeros(restarts, dtype=torch.int64, device=context.screen.device)
    trace: list[dict[str, object]] = []
    updates = evaluations - 1
    started = time.time()

    def inspect(
        energies: torch.Tensor,
        states: torch.Tensor,
        evaluation: int,
        endpoint: bool,
    ) -> None:
        nonlocal best_energy, best_state, best_gamma, best_beta, best_eval
        detached = energies.detach()
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
            trace.append(
                {
                    "evaluation": evaluation,
                    "post_update_endpoint": endpoint,
                    "loss_by_restart": [float(value) for value in detached.cpu()],
                }
            )

    for evaluation in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        energies, states = energy(gamma, beta)
        inspect(energies, states, evaluation, False)
        energies.sum().backward()
        optimizer.step()
    with torch.no_grad():
        energies, states = energy(gamma, beta)
        inspect(energies, states, evaluations, True)
    if best_state is None:  # pragma: no cover
        raise RuntimeError(label)
    return engine.OptimizationResult(
        label=label,
        state=best_state,
        energy=best_energy,
        gamma=best_gamma,
        beta=best_beta,
        best_evaluation=best_eval,
        evaluations_per_restart=evaluations,
        updates_per_restart=updates,
        endpoint_evaluations_per_restart=1,
        trace=trace,
        elapsed_sec=time.time() - started,
    )


def _compact_stage2(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "method_id": result["method_id"],
        "restart_seeds": result["restart_seeds"],
        "depth": result["depth"],
        "gamma_by_restart": result["gamma_by_restart"],
        "beta_by_restart": result["beta_by_restart"],
        "selected_feasibility_loss": result["selected_feasibility_loss"],
        "best_evaluation_by_restart": result["best_evaluation_by_restart"],
        "fixed_pi_initial_feasibility_loss": result[
            "fixed_pi_initial_feasibility_loss"
        ],
        "metrics": result["metrics"],
        "state_sha256": result["state_sha256"],
        "source_result_sha256": result["result_sha256"],
    }


def _family_contracts() -> dict[str, object]:
    rows: dict[str, object] = {}
    for family, spec in family_specs().items():
        rows[family] = {
            "kernel": spec.kernel,
            "reference": spec.reference,
            "phase_names": list(spec.phases),
            "mixer": spec.mixer,
            "gamma_count_at_depth_p": f"{spec.gamma_multiplier}*p",
            "beta_count_at_depth_p": "p",
            "optimizer_stream_same_at_every_depth": spec.stream,
            "initialization_policy": initialization_policy(family),
            "initial_angles_sha256_by_positive_depth": {
                str(depth): initial_angles_sha256(family, depth)
                for depth in POSITIVE_DEPTHS
            },
            "positive_depth_optimizer": "Adam",
            "positive_depth_learning_rate": LEARNING_RATE,
            "positive_depth_forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
            "checkpoint_rule": "minimum target-blind expected cumulative D loss",
            "p0": {
                "derived_without_optimization": True,
                "reference": spec.reference,
                "resource": resource(family, 0),
            },
            "resource_by_depth": {
                str(depth): resource(family, depth) for depth in DEPTHS
            },
        }
    return rows


def _optimization_contract() -> dict[str, object]:
    return {
        "independent_at_every_positive_depth": True,
        "continuation_or_angle_transfer": False,
        "algorithm": "Adam",
        "learning_rate": LEARNING_RATE,
        "forward_evaluations_per_restart_including_endpoint": OBJECTIVE_EVALUATIONS,
        "gradient_updates_per_restart": OBJECTIVE_EVALUATIONS - 1,
        "dtype": "complex64",
        "activation_checkpointing": ACTIVATION_CHECKPOINTING,
        "trace_every_evaluations": int(engine.valid_config().trace_every),
        "checkpoint_rule": "minimum expected cumulative D loss only",
    }


def _replay_validation_contract() -> dict[str, object]:
    return {
        "VF3_loss_absolute_tolerance": VF3_REPLAY_ATOL,
        "VF3_loss_relative_tolerance": VF3_REPLAY_RTOL,
        "optimized_loss_and_metric_absolute_tolerance": OPTIMIZED_REPLAY_ATOL,
        "optimized_loss_and_metric_relative_tolerance": OPTIMIZED_REPLAY_RTOL,
        "reason": "covers measured CPU/GPU complex64 reduction-order roundoff",
    }


def freeze(root: Path, validation_root: Path) -> dict[str, object]:
    """Freeze three new unfiltered tables and embed audited precursor angles."""

    root = root.resolve()
    validation_root = validation_root.resolve()
    source_manifest, source_seal, source_score, stage_a = (
        prior_campaign._verified_validation_sources(validation_root)
    )
    c12_row = common.stage_a_c12_row(stage_a)
    stage2 = variational._stage2_result(validation_root, source_manifest)
    common._require(
        tuple(int(value) for value in stage2["restart_seeds"]) == RESTART_SEEDS,
        "audited VF3 restart columns changed",
    )
    tables = [coherent.coefficient_table_from_pcg64(seed) for seed in TABLE_SEEDS]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_FRESH3_TARGET_BLIND_DEPTH_CHARACTERIZATION",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "study_scope": (
            "supplemental post-confirmation depth characterization; not a new "
            "claim-confirmation cohort"
        ),
        "frozen_source_sha256": frozen_source_sha256(),
        "cohort": {
            "generator": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "draw": "integers(low=1, high=21, size=(5,6), dtype=int64)",
            "table_seeds": list(TABLE_SEEDS),
            "table_count": len(TABLE_SEEDS),
            "tables_in_seed_order": tables,
            "fresh_relative_to_unique20": True,
            "unfiltered_no_rejection_or_replacement": True,
            "only_service_cost_matrix_varies": True,
            "analyst_prelaunch_check": {
                "all_three_have_nondegenerate_feasible_ground_energy": True,
                "purpose": "sanity check only; no table was rejected, replaced, or adapted",
                "optimizer_target_metric_blind_despite_prior_identity_inspection": True,
            },
        },
        "depths_in_frozen_order": list(DEPTHS),
        "positive_depths_optimized": list(POSITIVE_DEPTHS),
        "p0_semantics": "derived family-specific precursor anchor; no optimizer task",
        "families_in_frozen_order": list(FAMILIES),
        "family_count": len(FAMILIES),
        "families": _family_contracts(),
        "restart_seeds": list(RESTART_SEEDS),
        "restart_count_per_cell": len(RESTART_SEEDS),
        "optimization": _optimization_contract(),
        "embedded_C12_row": c12_row,
        "embedded_C12_row_sha256": common.sha256_json(c12_row),
        "embedded_service_independent_VF3": _compact_stage2(stage2),
        "precursor_reuse": {
            "C12": "audited saved-angle replay; never reoptimized",
            "VF3": "audited service-independent saved-angle replay; never reoptimized",
            "fixed_F32": "deterministic fixed-pi exact-indicator replay",
        },
        "precursor_replay_validation": _replay_validation_contract(),
        "source_validation": {
            "manifest_sha256": source_manifest["manifest_sha256"],
            "optimizer_seal_sha256": source_seal["seal_sha256"],
            "post_seal_score_sha256": source_score["score_sha256"],
        },
        "post_completion_target_definition": {
            "target_cardinality": "all exactly minimum-energy feasible states",
            "domain": "jointly feasible native-shell states",
            "ranking": "minimum raw coherent objective; score total ground-space probability if tied",
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
            raise FileExistsError(f"refusing to replace depth-sweep manifest: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_manifest(root: Path) -> dict[str, object]:
    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    cohort = manifest.get("cohort", {})
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("status") == "FROZEN_FRESH3_TARGET_BLIND_DEPTH_CHARACTERIZATION"
        and manifest.get("implementation_revision") == IMPLEMENTATION_REVISION
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and tuple(manifest.get("depths_in_frozen_order", ())) == DEPTHS
        and tuple(manifest.get("families_in_frozen_order", ())) == FAMILIES
        and tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS
        and manifest.get("families") == _family_contracts()
        and manifest.get("optimization") == _optimization_contract()
        and manifest.get("precursor_replay_validation") == _replay_validation_contract()
        and manifest.get("frozen_source_sha256") == frozen_source_sha256()
        and manifest.get("target_constructed") is False,
        "invalid depth-sweep manifest",
    )
    common._require(
        tuple(cohort.get("table_seeds", ())) == TABLE_SEEDS
        and cohort.get("table_count") == len(TABLE_SEEDS)
        and cohort.get("unfiltered_no_rejection_or_replacement") is True
        and cohort.get("tables_in_seed_order")
        == [coherent.coefficient_table_from_pcg64(seed) for seed in TABLE_SEEDS],
        "depth-sweep cohort changed",
    )
    c12 = manifest.get("embedded_C12_row")
    stage2 = manifest.get("embedded_service_independent_VF3", {})
    gamma = np.asarray(stage2.get("gamma_by_restart"), dtype=np.float64)
    beta = np.asarray(stage2.get("beta_by_restart"), dtype=np.float64)
    common._require(
        isinstance(c12, dict)
        and manifest.get("embedded_C12_row_sha256") == common.sha256_json(c12)
        and tuple(stage2.get("restart_seeds", ())) == RESTART_SEEDS
        and gamma.shape == beta.shape == (len(RESTART_SEEDS), F3_DEPTH)
        and bool(np.all(np.isfinite(gamma)))
        and bool(np.all(np.isfinite(beta))),
        "invalid embedded depth-sweep precursors",
    )
    return manifest


def table(manifest: Mapping[str, object], seed: int) -> dict[str, object]:
    common._require(seed in TABLE_SEEDS, f"unknown depth-sweep table seed {seed}")
    row = manifest["cohort"]["tables_in_seed_order"][TABLE_SEEDS.index(seed)]
    common._require(
        row == coherent.coefficient_table_from_pcg64(seed),
        f"service table changed for seed {seed}",
    )
    return row


def make_screen(*, device: str) -> engine.StructuralScreen:
    config = replace(
        engine.valid_config(),
        restarts=len(RESTART_SEEDS),
        initialization_seeds=RESTART_SEEDS,
        restart_batch=len(RESTART_SEEDS),
        device=device,
        dtype="complex64",
        learning_rate=LEARNING_RATE,
        activation_checkpointing=ACTIVATION_CHECKPOINTING,
    )
    return engine.StructuralScreen(config)


@dataclass
class RuntimeContext:
    screen: engine.StructuralScreen
    c12_state: torch.Tensor
    vf3_state: torch.Tensor
    fixed_f32_state: torch.Tensor
    native_state: torch.Tensor
    replay_evidence: dict[str, object]


_RUNTIME_CACHE: dict[tuple[str, str], RuntimeContext] = {}


def runtime_context(manifest: Mapping[str, object], device: str) -> RuntimeContext:
    key = (str(device), str(manifest["manifest_sha256"]))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    screen = make_screen(device=device)
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
    saved_f_loss = torch.as_tensor(
        stage2["selected_feasibility_loss"],
        dtype=screen.real_dtype,
        device=screen.device,
    )
    common._require(
        torch.allclose(
            f_loss,
            saved_f_loss,
            atol=VF3_REPLAY_ATOL,
            rtol=VF3_REPLAY_RTOL,
        ),
        "service-independent VF3 replay loss mismatch",
    )
    fixed_f32 = common.fixed_uniform_precursor(screen)
    native = screen.native[None, :].expand(len(RESTART_SEEDS), -1).clone()
    evidence = {
        "C12_state_sha256": common.state_sha256(c12),
        "VF3_saved_state_sha256": stage2["state_sha256"],
        "VF3_replay_state_sha256": common.state_sha256(vf3),
        "VF3_replay_loss_max_abs_error": float(
            torch.max(torch.abs(f_loss - saved_f_loss)).cpu()
        ),
        "VF3_replay_absolute_tolerance": VF3_REPLAY_ATOL,
        "VF3_replay_relative_tolerance": VF3_REPLAY_RTOL,
        "resolved_device": str(screen.device),
        "fixed_F32_state_sha256": common.state_sha256(fixed_f32),
        "native_state_sha256": common.state_sha256(native),
    }
    context = RuntimeContext(screen, c12, vf3, fixed_f32, native, evidence)
    _RUNTIME_CACHE[key] = context
    return context


def p0_state(family: str, context: RuntimeContext) -> torch.Tensor:
    if family == FULL_LP:
        return context.vf3_state
    if family in (SKIP_F3, COLLAPSE_SEPARATE, WARM_COMBINED):
        return context.c12_state
    if family == FIXED_UNIFORM:
        return context.fixed_f32_state
    if family in (DIRECT_SEPARATE, DIRECT_COMBINED, NATIVE_PROJECTOR_SEPARATE):
        return context.native_state
    raise ValueError(family)


def energy_fn(
    family: str,
    depth: int,
    context: RuntimeContext,
    objective: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    common._require(family in FAMILIES and depth in POSITIVE_DEPTHS, "invalid energy cell")
    screen = context.screen
    cumulative = screen.common + objective
    if family == FULL_LP:
        reference, phases, kernel = context.vf3_state, (objective,), "history"
    elif family == SKIP_F3:
        reference, phases, kernel = context.c12_state, (objective,), "history"
    elif family == FIXED_UNIFORM:
        reference, phases, kernel = context.fixed_f32_state, (objective,), "history"
    elif family == COLLAPSE_SEPARATE:
        reference = context.c12_state
        phases = (screen.final_mask.to(screen.real_dtype), objective)
        kernel = "history"
    elif family == DIRECT_SEPARATE:
        reference, phases, kernel = context.native_state, (screen.q, screen.c, objective), "block_xy"
    elif family == DIRECT_COMBINED:
        reference, phases, kernel = context.native_state, (cumulative,), "block_xy"
    elif family == WARM_COMBINED:
        reference, phases, kernel = context.c12_state, (cumulative,), "block_xy"
    elif family == NATIVE_PROJECTOR_SEPARATE:
        reference, phases, kernel = context.native_state, (screen.q, screen.c, objective), "history"
    else:  # pragma: no cover
        raise ValueError(family)

    def energy(gamma: torch.Tensor, beta: torch.Tensor):
        if kernel == "history":
            return objective_engine.history_circuit(
                screen, reference, phases, cumulative, gamma, beta, depth
            )
        return screen.block_energy(
            reference, phases, cumulative, gamma, beta, depth
        )

    return energy


def _metrics_ok(metrics: Mapping[str, object]) -> bool:
    for key in (
        "state_norm", "quota_mass", "conflict_mass", "final_mass",
        "expected_Qtilde", "expected_Ctilde", "common_loss",
        "expected_Otilde", "expected_cumulative_loss",
    ):
        values = np.asarray(metrics.get(key), dtype=np.float64)
        if values.shape != (len(RESTART_SEEDS),) or not np.all(np.isfinite(values)):
            return False
    return True


def _trace_ok(result: Mapping[str, object]) -> bool:
    trace = result.get("optimizer_trace")
    if not isinstance(trace, list):
        return False
    trace_every = int(engine.valid_config().trace_every)
    expected = [
        evaluation
        for evaluation in range(1, OBJECTIVE_EVALUATIONS)
        if evaluation == 1 or evaluation % trace_every == 0
    ] + [OBJECTIVE_EVALUATIONS]
    if [row.get("evaluation") for row in trace] != expected:
        return False
    for index, row in enumerate(trace):
        values = np.asarray(row.get("loss_by_restart"), dtype=np.float64)
        if values.shape != (len(RESTART_SEEDS),) or not np.all(np.isfinite(values)):
            return False
        if bool(row.get("post_update_endpoint")) != (index == len(trace) - 1):
            return False
    return result.get("optimizer_trace_sha256") == common.sha256_json(trace)


def validate_result(
    manifest: Mapping[str, object],
    result: Mapping[str, object],
    seed: int,
    family: str,
    depth: int,
) -> None:
    common._require(depth in POSITIVE_DEPTHS, "p0 cannot be a result artifact")
    spec = family_specs()[family]
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    expected_table = table(manifest, seed)
    gamma = np.asarray(result.get("gamma_by_restart"), dtype=np.float64)
    beta = np.asarray(result.get("beta_by_restart"), dtype=np.float64)
    common._require(
        result.get("schema") == f"{SCHEMA}-optimizer-result"
        and result.get("status") == "COMPLETED_TARGET_BLIND_DEPTH_CELL"
        and result.get("manifest_sha256") == manifest["manifest_sha256"]
        and result.get("table_seed") == seed
        and result.get("table_id") == table_id(seed)
        and result.get("family") == family
        and result.get("depth") == depth
        and result.get("method_id") == method_id(family, depth)
        and result.get("coefficient_table_sha256")
        == expected_table["service_cost_compact_json_sha256"]
        and tuple(result.get("restart_seeds", ())) == RESTART_SEEDS
        and result.get("optimizer_stream") == spec.stream
        and result.get("initialization_policy") == initialization_policy(family)
        and result.get("initial_angles_sha256") == initial_angles_sha256(family, depth)
        and result.get("resource") == resource(family, depth)
        and result.get("forward_evaluations_per_restart") == OBJECTIVE_EVALUATIONS
        and result.get("gradient_updates_per_restart") == OBJECTIVE_EVALUATIONS - 1
        and result.get("endpoint_evaluations_per_restart") == 1
        and gamma.shape == (len(RESTART_SEEDS), spec.gamma_multiplier * depth)
        and beta.shape == (len(RESTART_SEEDS), depth)
        and bool(np.all(np.isfinite(gamma)))
        and bool(np.all(np.isfinite(beta)))
        and _metrics_ok(result.get("metrics", {}))
        and _trace_ok(result)
        and isinstance(result.get("precursor_replay_evidence"), dict)
        and float(result["precursor_replay_evidence"].get("VF3_replay_loss_max_abs_error", math.inf))
        <= VF3_REPLAY_ATOL
        and result["precursor_replay_evidence"].get("VF3_replay_absolute_tolerance")
        == VF3_REPLAY_ATOL
        and result.get("checkpoint_selected_by_target_blind_loss_only") is True
        and result.get("independent_depth_optimization_no_continuation") is True
        and result.get("target_artifact_loaded") is False
        and result.get("target_constructed") is False
        and result.get("target_probability_computed") is False
        and result.get("result_sha256") == common.sha256_json(body),
        f"invalid depth-sweep result {seed}/{family}/p{depth}",
    )


def run_cell(
    root: Path, seed: int, family: str, depth: int, *, device: str
) -> dict[str, object]:
    manifest = validated_manifest(root)
    common._require(depth in POSITIVE_DEPTHS, "p0 is derived and must not be optimized")
    destination = result_path(root, seed, family, depth)
    if destination.exists():
        existing = common.load_json(destination)
        validate_result(manifest, existing, seed, family, depth)
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    table_row = table(manifest, seed)
    context = runtime_context(manifest, device)
    objective = discovery.objective_tensor(context.screen, table_row)
    spec = family_specs()[family]
    result = _optimize_independent(
        energy_fn(family, depth, context, objective),
        family,
        depth,
        OBJECTIVE_EVALUATIONS,
        method_id(family, depth),
        context,
    )
    metrics = discovery._metrics(context.screen, result.state, objective)
    common._require(_metrics_ok(metrics), f"nonfinite result {seed}/{family}/p{depth}")
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-optimizer-result",
        "status": "COMPLETED_TARGET_BLIND_DEPTH_CELL",
        "manifest_sha256": manifest["manifest_sha256"],
        "table_seed": seed,
        "table_id": table_id(seed),
        "coefficient_table_sha256": table_row["service_cost_compact_json_sha256"],
        "family": family,
        "method_id": method_id(family, depth),
        "depth": depth,
        "kernel": spec.kernel,
        "reference": spec.reference,
        "phase_names": list(spec.phases),
        "mixer": spec.mixer,
        "restart_seeds": list(RESTART_SEEDS),
        "optimizer_stream": spec.stream,
        "initialization_policy": initialization_policy(family),
        "initial_angles_sha256": initial_angles_sha256(family, depth),
        "resource": resource(family, depth),
        "forward_evaluations_per_restart": result.evaluations_per_restart,
        "gradient_updates_per_restart": result.updates_per_restart,
        "endpoint_evaluations_per_restart": result.endpoint_evaluations_per_restart,
        "best_evaluation_by_restart": [int(value) for value in result.best_evaluation.cpu()],
        "selected_expected_loss": [float(value) for value in result.energy.cpu()],
        "gamma_by_restart": result.gamma.cpu().tolist(),
        "beta_by_restart": result.beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(result.state),
        "optimizer_trace": result.trace,
        "optimizer_trace_sha256": common.sha256_json(result.trace),
        "precursor_replay_evidence": context.replay_evidence,
        "elapsed_sec": result.elapsed_sec,
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
            "activation_checkpointing": ACTIVATION_CHECKPOINTING,
        },
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    validate_result(manifest, payload, seed, family, depth)
    return payload


def _estimated_seconds(family: str, depth: int) -> float:
    return max(20.0, family_specs()[family].p64_estimated_seconds * depth / 64.0)


def plan_shards(root: Path) -> dict[str, object]:
    """Freeze two exact, disjoint, approximately load-balanced A800 queues."""

    manifest = validated_manifest(root)
    tasks = [
        {
            "table_seed": seed,
            "table_id": table_id(seed),
            "family": family,
            "depth": depth,
            "method_id": method_id(family, depth),
            "estimated_seconds": _estimated_seconds(family, depth),
        }
        for family in FAMILIES
        for depth in POSITIVE_DEPTHS
        for seed in TABLE_SEEDS
    ]
    tasks.sort(
        key=lambda row: (
            -float(row["estimated_seconds"]),
            FAMILIES.index(str(row["family"])),
            -int(row["depth"]),
            int(row["table_seed"]),
        )
    )
    shard_tasks: list[list[dict[str, object]]] = [[], []]
    shard_loads = [0.0, 0.0]
    per_shard = len(tasks) // 2
    common._require(len(tasks) % 2 == 0, "depth-sweep task count must split evenly")
    for task in tasks:
        eligible = [index for index in range(2) if len(shard_tasks[index]) < per_shard]
        index = min(eligible, key=lambda value: (shard_loads[value], len(shard_tasks[value]), value))
        shard_tasks[index].append(task)
        shard_loads[index] += float(task["estimated_seconds"])
    shards = [
        {
            "shard_index": index,
            "platform_label": ("gpu_worker_1", "gpu_worker_2")[index],
            "task_count": len(shard_tasks[index]),
            "estimated_serial_seconds": shard_loads[index],
            "recommended_concurrent_workers": 2,
            "tasks_in_execution_order": shard_tasks[index],
        }
        for index in range(2)
    ]
    actual = [
        (int(row["table_seed"]), str(row["family"]), int(row["depth"]))
        for shard in shards for row in shard["tasks_in_execution_order"]
    ]
    expected = {
        (seed, family, depth)
        for seed in TABLE_SEEDS for family in FAMILIES for depth in POSITIVE_DEPTHS
    }
    common._require(
        len(actual) == len(expected)
        and set(actual) == expected
        and [len(rows) for rows in shard_tasks] == [per_shard, per_shard],
        "depth-sweep queues are not an exact disjoint Cartesian product",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-two-pod-plan",
        "status": "FROZEN_TWO_DISJOINT_A800_DEPTH_QUEUES_NOT_LAUNCHED",
        "manifest_sha256": manifest["manifest_sha256"],
        "assignment_rule": "greedy descending-time balance across two disjoint queues",
        "positive_depth_task_count": len(actual),
        "derived_p0_cell_count": len(TABLE_SEEDS) * len(FAMILIES),
        "total_curve_cell_count_including_p0": len(TABLE_SEEDS) * len(FAMILIES) * len(DEPTHS),
        "shards": shards,
        "exact_positive_depth_cartesian_product_complete": True,
        "duplicate_task_count": 0,
        "optimization_launched": False,
    }
    payload = {**body, "plan_sha256": common.sha256_json(body)}
    destination = _plan_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace depth-sweep plan: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_plan(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = validated_manifest(root)
    plan = common.load_json(_plan_path(root))
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    tasks = [
        (int(row["table_seed"]), str(row["family"]), int(row["depth"]))
        for shard in plan.get("shards", ())
        for row in shard.get("tasks_in_execution_order", ())
    ]
    expected = {
        (seed, family, depth)
        for seed in TABLE_SEEDS for family in FAMILIES for depth in POSITIVE_DEPTHS
    }
    common._require(
        plan.get("schema") == f"{SCHEMA}-two-pod-plan"
        and plan.get("manifest_sha256") == manifest["manifest_sha256"]
        and plan.get("plan_sha256") == common.sha256_json(body)
        and len(plan.get("shards", ())) == 2
        and [int(shard.get("task_count", -1)) for shard in plan.get("shards", ())]
        == [len(expected) // 2, len(expected) // 2]
        and len(tasks) == len(expected)
        and set(tasks) == expected,
        "invalid depth-sweep two-pod plan",
    )
    return manifest, plan


def seal(root: Path) -> dict[str, object]:
    """Seal only the exact set of independently optimized positive-depth cells."""

    manifest, plan = validated_plan(root)
    completed: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    for seed in TABLE_SEEDS:
        for family in FAMILIES:
            for depth in POSITIVE_DEPTHS:
                path = result_path(root, seed, family, depth)
                expected_paths.add(path.resolve())
                result = common.load_json(path)
                validate_result(manifest, result, seed, family, depth)
                completed.append(
                    {
                        "table_seed": seed,
                        "family": family,
                        "depth": depth,
                        "result_sha256": result["result_sha256"],
                    }
                )
    actual_paths = {path.resolve() for path in (root / RESULT_DIRECTORY).glob("*/*.json")}
    common._require(actual_paths == expected_paths, "depth-sweep result set is not exact")
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-global-seal",
        "status": "SEALED_COMPLETE_TARGET_BLIND_DEPTH_SWEEP",
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "frozen_source_sha256": manifest["frozen_source_sha256"],
        "table_seeds_in_frozen_order": list(TABLE_SEEDS),
        "families_in_frozen_order": list(FAMILIES),
        "depths_in_frozen_order": list(DEPTHS),
        "optimized_positive_depths": list(POSITIVE_DEPTHS),
        "complete_optimizer_result_count": len(completed),
        "expected_optimizer_result_count": len(TABLE_SEEDS) * len(FAMILIES) * len(POSITIVE_DEPTHS),
        "derived_p0_cell_count": len(TABLE_SEEDS) * len(FAMILIES),
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
            raise FileExistsError(f"refusing to replace depth-sweep seal: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def smoke(root: Path, *, device: str) -> dict[str, object]:
    """Excluded p64/four-restart direct-separate forward/backward smoke."""

    manifest = validated_manifest(root)
    context = runtime_context(manifest, device)
    table_row = table(manifest, TABLE_SEEDS[0])
    objective = discovery.objective_tensor(context.screen, table_row)
    spec = family_specs()[DIRECT_SEPARATE]
    if context.screen.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.screen.device)
    result = _optimize_independent(
        energy_fn(DIRECT_SEPARATE, 64, context, objective),
        DIRECT_SEPARATE,
        64,
        2,
        f"EXCLUDED_SMOKE_{method_id(DIRECT_SEPARATE, 64)}",
        context,
    )
    peak = (
        int(torch.cuda.max_memory_allocated(context.screen.device))
        if context.screen.device.type == "cuda" else None
    )
    common._require(bool(torch.all(torch.isfinite(result.energy))), "p64 smoke failed")
    return {
        "status": "EXCLUDED_DEPTH_SWEEP_P64_FOUR_RESTART_SMOKE_PASS",
        "production_result_written": False,
        "family": DIRECT_SEPARATE,
        "depth": 64,
        "restart_batch": len(RESTART_SEEDS),
        "forward_evaluations": 2,
        "gradient_updates": 1,
        "elapsed_sec": result.elapsed_sec,
        "cuda_peak_memory_allocated_bytes": peak,
        "resolved_device": str(context.screen.device),
        "cuda_device_name": (
            torch.cuda.get_device_name(context.screen.device)
            if context.screen.device.type == "cuda" else None
        ),
    }


def estimate() -> dict[str, object]:
    serial = sum(
        _estimated_seconds(family, depth)
        for family in FAMILIES for depth in POSITIVE_DEPTHS for _ in TABLE_SEEDS
    )
    return {
        "table_count": len(TABLE_SEEDS),
        "family_count": len(FAMILIES),
        "depth_count_including_p0": len(DEPTHS),
        "positive_depth_optimizer_task_count": len(TABLE_SEEDS) * len(FAMILIES) * len(POSITIVE_DEPTHS),
        "derived_p0_cell_count": len(TABLE_SEEDS) * len(FAMILIES),
        "total_curve_cell_count": len(TABLE_SEEDS) * len(FAMILIES) * len(DEPTHS),
        "projected_serial_GPU_hours": serial / 3600.0,
        "projected_two_A800_two_worker_wall_hours": serial / (2 * 2 * 3600.0),
        "note": "rough calibration; use excluded parallel p64 smoke before launch",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--validation-root", type=Path, required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--validation-root", type=Path, required=True)
    command = sub.add_parser("run-cell")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--table-seed", type=int, choices=TABLE_SEEDS, required=True)
    command.add_argument("--family", choices=FAMILIES, required=True)
    command.add_argument("--depth", type=int, choices=POSITIVE_DEPTHS, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("plan-shards")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    sub.add_parser("estimate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.validation_root)
    elif args.command == "prepare":
        freeze(args.campaign_root, args.validation_root)
        payload = plan_shards(args.campaign_root)
    elif args.command == "run-cell":
        payload = run_cell(
            args.campaign_root, args.table_seed, args.family, args.depth,
            device=args.device,
        )
    elif args.command == "plan-shards":
        payload = plan_shards(args.campaign_root)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    elif args.command == "smoke":
        payload = smoke(args.campaign_root, device=args.device)
    elif args.command == "estimate":
        payload = estimate()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload.get("status", payload))


if __name__ == "__main__":
    main()
