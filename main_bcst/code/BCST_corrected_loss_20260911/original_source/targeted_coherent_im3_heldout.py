#!/usr/bin/env python3
"""Frozen, unfiltered 12-table held-out coherent-IM3 campaign.

Only the synthetic 5x6 service-cost matrix varies.  The IM3 physics, hard
constraints, circuit families, depths, optimizer, budgets, and four paired
restart streams are copied from the completed validation campaign.  Optimizer
workers never construct or score an optimum target.  A global target-blind
seal is required before the separate scorer may rank any table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import stage3_objective_common as objective_engine
import target_free_structural_screen as engine
import targeted_coherent_im3_fixed_uniform_control as fixed_control
import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_screen as discovery
import targeted_coherent_im3_variational_f3 as variational
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-true-three-stage-heldout-v1"
IMPLEMENTATION_REVISION = "prelaunch-audit-fixes-20260812-r2"
MANIFEST_NAME = "targeted_coherent_im3_heldout_manifest.json"
RESULT_DIRECTORY = "targeted_coherent_im3_heldout_results"
GLOBAL_SEAL_NAME = "targeted_coherent_im3_heldout_global_optimizer_seal.json"
SHARD_PLAN_NAME = "targeted_coherent_im3_heldout_two_gpu_shard_plan.json"

# Frozen before any held-out outcomes.  PCG64 is instantiated independently
# for each seed; all 12 generated matrices must be retained without rejection.
TABLE_SEEDS = tuple(range(2_026_081_401, 2_026_081_413))
RESTART_SEEDS = variational.RESTART_SEEDS

FULL = variational.FULL_METHOD
JOINT = discovery.COLLAPSE_JOINT
SEPARATE = discovery.COLLAPSE_SEPARATE
DIRECT = discovery.DIRECT
FIXED = fixed_control.METHOD_ID
METHODS = (FULL, JOINT, SEPARATE, DIRECT, FIXED)

VF3_EVALUATIONS = variational.STAGE2_EVALUATIONS
OBJECTIVE_EVALUATIONS = variational.OBJECTIVE_EVALUATIONS
STREAMS = {
    FULL: discovery.specs()[discovery.FULL].stream,
    JOINT: discovery.specs()[discovery.COLLAPSE_JOINT].stream,
    SEPARATE: discovery.specs()[discovery.COLLAPSE_SEPARATE].stream,
    DIRECT: discovery.specs()[discovery.DIRECT].stream,
    FIXED: fixed_control.STREAM,
}


def table_id(seed: int) -> str:
    return f"pcg64_{int(seed)}"


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _result_path(root: Path, seed: int, method: str) -> Path:
    return root / RESULT_DIRECTORY / table_id(seed) / f"{method}.json"


def _verified_validation_sources(
    validation_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Bind the campaign to the passed true-three-stage validation artifacts."""

    manifest = common.load_json(validation_root / variational.MANIFEST_NAME)
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    seal = common.load_json(validation_root / variational.SEAL_NAME)
    seal_body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    score_path = validation_root / "targeted_coherent_im3_variational_f3_post_seal_score.json"
    score = common.load_json(score_path)
    score_body = {key: value for key, value in score.items() if key != "score_sha256"}
    common._require(
        manifest.get("schema") == f"{variational.SCHEMA}-manifest"
        and manifest.get("manifest_sha256") == common.sha256_json(manifest_body)
        and seal.get("schema") == f"{variational.SCHEMA}-seal"
        and seal.get("manifest_sha256") == manifest["manifest_sha256"]
        and seal.get("seal_sha256") == common.sha256_json(seal_body)
        and score.get("manifest_sha256") == manifest["manifest_sha256"]
        and score.get("optimizer_seal_sha256") == seal["seal_sha256"]
        and score.get("score_sha256") == common.sha256_json(score_body)
        and score.get("gate", {}).get("pass") is True,
        "held-out freeze requires the passed true-three-stage validation",
    )
    stage_a = common.load_json(Path(str(manifest["stage_a_path"])))
    common.stage_a_c12_row(stage_a)
    return manifest, seal, score, stage_a


def _method_contract() -> dict[str, object]:
    return {
        FULL: {
            "pipeline": "C12-VF3-O16",
            "stage2_forward_evaluations_per_restart": VF3_EVALUATIONS,
            "objective_forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
            "initialization_stream": STREAMS[FULL],
            "resource": coherent.resource_ledger()["rows"]["full_C12_fixedF3_O16"],
        },
        JOINT: {
            "pipeline": "C12-(F+O)19 joint phase",
            "forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
            "initialization_stream": STREAMS[JOINT],
            "resource": discovery.resource_row(JOINT),
        },
        SEPARATE: {
            "pipeline": "C12-(F,O)19 separate phases",
            "forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
            "initialization_stream": STREAMS[SEPARATE],
            "resource": discovery.resource_row(SEPARATE),
        },
        DIRECT: {
            "pipeline": "uniform two-hot-(Q,C,O)64",
            "forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
            "initialization_stream": STREAMS[DIRECT],
            "resource": discovery.resource_row(DIRECT),
        },
        FIXED: {
            "pipeline": "uniform-F32-O16",
            "forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
            "initialization_stream": STREAMS[FIXED],
            "resource": fixed_control.resource_ledger()["rows"],
        },
    }


def freeze(root: Path, validation_root: Path) -> dict[str, object]:
    """Write all 12 raw matrices and the complete target-blind protocol."""

    root = root.resolve()
    validation_root = validation_root.resolve()
    source_manifest, source_seal, source_score, stage_a = _verified_validation_sources(
        validation_root
    )
    c12_row = common.stage_a_c12_row(stage_a)
    fixed_score_path = (
        validation_root.parent
        / "coherent_im3_fixed_uniform_validation_seed20260813"
        / "targeted_coherent_im3_fixed_uniform_post_seal_score_v2.json"
    )
    exact_score_path = (
        validation_root.parent
        / "coherent_im3_exact_final_local_validation_seed20260813"
        / "targeted_coherent_im3_exact_final_local_post_seal_score.json"
    )
    fixed_score = common.load_json(fixed_score_path)
    exact_score = common.load_json(exact_score_path)
    fixed_score_body = {
        key: value for key, value in fixed_score.items() if key != "score_sha256"
    }
    exact_score_body = {
        key: value for key, value in exact_score.items() if key != "score_sha256"
    }
    common._require(
        fixed_score.get("score_sha256") == common.sha256_json(fixed_score_body)
        and exact_score.get("score_sha256") == common.sha256_json(exact_score_body),
        "completed validation-control scores are invalid",
    )
    exact_literal_gms = [
        float(
            exact_score["comparisons_to_validated_full_by_full_C_convention"][
                convention
            ]["literal_primary"]["geometric_mean_control_over_full"]
        )
        for convention in ("native_C", "uniform_Pauli_C_sensitivity")
    ]
    common._require(
        min(exact_literal_gms) > 1.20,
        "exact-local p4 became competitive and must be included before freeze",
    )
    tables = [coherent.coefficient_table_from_pcg64(seed) for seed in TABLE_SEEDS]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_UNFILTERED_TARGET_BLIND_HELDOUT_COHORT",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "validation_root": str(validation_root),
        "validation_manifest_sha256": source_manifest["manifest_sha256"],
        "validation_optimizer_seal_sha256": source_seal["seal_sha256"],
        "validation_post_seal_score_sha256": source_score["score_sha256"],
        "embedded_target_blind_C12_row": c12_row,
        "embedded_target_blind_C12_row_sha256": common.sha256_json(c12_row),
        "cohort": {
            "experimental_unit": "one independently generated 5x6 service-cost matrix",
            "generator": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "draw": "integers(low=1, high=21, size=(5,6), dtype=int64)",
            "table_seeds": list(TABLE_SEEDS),
            "table_count": len(TABLE_SEEDS),
            "unfiltered_no_rejection_or_replacement": True,
            "tables_in_seed_order": tables,
            "physics_and_hierarchy_fixed": True,
            "only_service_cost_matrix_varies": True,
        },
        "restart_seeds": list(RESTART_SEEDS),
        "restart_count_per_table_method": len(RESTART_SEEDS),
        "paired_restart_rule": (
            "Within every table, rows are compared in matched restart-seed "
            "columns; method-specific initialization streams remain validation-fixed, "
            "so numerical angle starts are not shared across methods."
        ),
        "depths": {"C": 12, "variational_F": 3, "O": 16, "collapse": 19, "direct": 64, "fixed_F": 32},
        "optimizer": {"name": "Adam", "learning_rate": 0.035},
        "budgets": {
            "variational_F3_forward_evaluations_per_restart": VF3_EVALUATIONS,
            "O16_and_each_control_forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
        },
        "methods": _method_contract(),
        "completed_validation_controls": {
            "fixed_uniform_score_path": str(fixed_score_path.resolve()),
            "fixed_uniform_score_sha256": fixed_score["score_sha256"],
            "exact_local_score_path": str(exact_score_path.resolve()),
            "exact_local_score_sha256": exact_score["score_sha256"],
        },
        "global_target_blind_seal_required_before_any_target_scoring": True,
        "prespecified_table_level_analysis": {
            "primary_full_comparator": (
                "per table and restart, minimum RTS99*RU over joint collapse, "
                "separate collapse, direct p64, and fixed-uniform F32-O16"
            ),
            "primary_ledgers": ["native_C", "uniform_Pauli_C_sensitivity"],
            "equal_weight_per_table": True,
            "within_table_summary": "geometric mean over four paired restart cost ratios",
            "cohort_summary": "geometric mean of the 12 within-table geometric means",
            "paired_win_definition": "full within-table geometric-mean cost is strictly below its comparator envelope",
            "inference": {
                "sign_test": "exact two-sided binomial sign test on 12 table wins, ties count as non-wins for the gate",
                "effect_interval": (
                    "two-sided 95% percentile bootstrap CI of cohort geometric mean; "
                    "resample 12 tables with replacement using PCG64 seed 2026081499 and 200000 replicates"
                ),
            },
            "success_gate": {
                "minimum_table_wins_each_ledger": 10,
                "minimum_cohort_geometric_mean_each_ledger": 1.20,
                "minimum_full_final_feasible_mass_each_restart": 0.90,
                "minimum_tables_with_all_four_VF3_improvements": 10,
                "VF3_precursor_improvement": "selected VF3 feasibility loss is strictly below exact-pi F3 loss",
            },
        },
        "exact_local_p4": {
            "included_in_primary_methods": False,
            "decision_recorded_before_heldout_launch": True,
            "decision": "excluded: completed validation control was decisively noncompetitive",
            "literal_primary_control_over_full_geometric_mean": {
                "native_C": exact_literal_gms[0],
                "uniform_Pauli_C_sensitivity": exact_literal_gms[1],
            },
            "independent_audit_may_require_reopening_before_launch": True,
        },
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
            raise FileExistsError(f"refusing to replace held-out manifest: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def _validated_embedded_c12_row(manifest: dict[str, object]) -> dict[str, object]:
    row = manifest.get("embedded_target_blind_C12_row")
    common._require(
        isinstance(row, dict)
        and manifest.get("embedded_target_blind_C12_row_sha256")
        == common.sha256_json(row)
        and row.get("algorithm") == "Coarse-only-C-p12"
        and tuple(int(value) for value in row.get("initialization_seeds", ()))
        == common.ALL_STAGE_A_RESTART_SEEDS,
        "embedded target-blind C12 row is invalid",
    )
    for name in ("gamma_by_restart", "beta_by_restart"):
        values = np.asarray(row.get(name), dtype=np.float64)
        common._require(
            values.shape == (16, common.P_C) and np.all(np.isfinite(values)),
            f"invalid embedded C12 {name}",
        )
    return row


def validated_manifest(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("status") == "FROZEN_UNFILTERED_TARGET_BLIND_HELDOUT_COHORT"
        and manifest.get("implementation_revision") == IMPLEMENTATION_REVISION
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS
        and manifest.get("budgets", {}).get("variational_F3_forward_evaluations_per_restart") == VF3_EVALUATIONS
        and manifest.get("budgets", {}).get("O16_and_each_control_forward_evaluations_per_restart") == OBJECTIVE_EVALUATIONS,
        "held-out manifest is invalid",
    )
    cohort = manifest.get("cohort", {})
    common._require(
        tuple(cohort.get("table_seeds", ())) == TABLE_SEEDS
        and cohort.get("table_count") == len(TABLE_SEEDS)
        and cohort.get("unfiltered_no_rejection_or_replacement") is True
        and cohort.get("tables_in_seed_order")
        == [coherent.coefficient_table_from_pcg64(seed) for seed in TABLE_SEEDS],
        "held-out table cohort changed",
    )
    # Workers deliberately do not reopen the target-aware development score or
    # any machine-specific source path.  The manifest already binds the passed
    # validation hashes and embeds only the target-blind C12 saved-angle row.
    return manifest, _validated_embedded_c12_row(manifest)


def _table(manifest: dict[str, object], seed: int) -> dict[str, object]:
    common._require(seed in TABLE_SEEDS, f"unknown held-out table seed {seed}")
    index = TABLE_SEEDS.index(seed)
    table = manifest["cohort"]["tables_in_seed_order"][index]
    common._require(table == coherent.coefficient_table_from_pcg64(seed), "held-out table changed")
    return table


def _problem(seed: int, table: dict[str, object], c12_row: dict[str, object], device: str):
    screen = common.make_screen(device=device, dtype="complex64", seeds=RESTART_SEEDS)
    c_state = common.replay_c12(screen, c12_row)
    objective = discovery.objective_tensor(screen, table)
    return screen, c_state, objective


def _result_body(
    manifest: dict[str, object], seed: int, method: str, result: engine.OptimizationResult,
    metrics: dict[str, object], gamma: torch.Tensor, beta: torch.Tensor,
    resource: dict[str, object], *, stage2: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-method-result",
        "status": "COMPLETED_TARGET_BLIND_HELDOUT_METHOD",
        "manifest_sha256": manifest["manifest_sha256"],
        "table_seed": seed,
        "table_id": table_id(seed),
        "coefficient_table_sha256": manifest["cohort"]["tables_in_seed_order"][TABLE_SEEDS.index(seed)]["service_cost_compact_json_sha256"],
        "method_id": method,
        "restart_seeds": list(RESTART_SEEDS),
        "optimizer_stream": STREAMS[method],
        "resource": resource,
        "forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
        "gradient_updates_per_restart": result.updates_per_restart,
        "endpoint_evaluations_per_restart": result.endpoint_evaluations_per_restart,
        "best_evaluation_by_restart": [int(value) for value in result.best_evaluation.cpu()],
        "selected_expected_loss": [float(value) for value in result.energy.cpu()],
        "gamma_by_restart": gamma.cpu().tolist(),
        "beta_by_restart": beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(result.state),
        "elapsed_sec": result.elapsed_sec,
        "checkpoint_selected_by_expected_loss_only": True,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    if stage2 is not None:
        body["stage2"] = stage2
    return body


def _method_angle_counts(method: str) -> tuple[int, int]:
    if method in (FULL, FIXED):
        return variational.P_O, variational.P_O
    spec = discovery.specs()[method]
    if method in (JOINT, SEPARATE):
        return len(spec.phases) * spec.depth, spec.depth
    if method == DIRECT:
        return 3 * spec.depth, spec.depth
    raise ValueError(method)


def run_method(root: Path, seed: int, method: str, *, device: str) -> dict[str, object]:
    manifest, c12_row = validated_manifest(root)
    common._require(method in METHODS, f"unknown held-out method {method}")
    destination = _result_path(root, seed, method)
    if destination.exists():
        return common.load_json(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = _table(manifest, seed)
    screen, c_state, objective = _problem(seed, table, c12_row, device)

    stage2_payload: dict[str, object] | None = None
    if method == FULL:
        stage2_result, fixed_state = variational._optimize_from_pi(
            variational._feasibility_energy(screen, c_state), screen, VF3_EVALUATIONS
        )
        reference = stage2_result.state
        energy = variational._objective_energy(screen, reference, objective)
        gamma_count = beta_count = variational.P_O
        fixed_mass = screen.metrics(fixed_state)["final_mass"]
        selected_mass = screen.metrics(reference)["final_mass"]
        stage2_payload = {
            "method_id": variational.STAGE2_METHOD,
            "forward_evaluations_per_restart": VF3_EVALUATIONS,
            "exact_pi_feasibility_loss": [1.0 - float(value) for value in fixed_mass],
            "selected_feasibility_loss": [float(value) for value in stage2_result.energy.cpu()],
            "selected_final_mass": selected_mass,
            "best_evaluation_by_restart": [int(value) for value in stage2_result.best_evaluation.cpu()],
            "gamma_by_restart": stage2_result.gamma.cpu().tolist(),
            "beta_by_restart": stage2_result.beta.cpu().tolist(),
            "state_sha256": common.state_sha256(reference),
            "elapsed_sec": stage2_result.elapsed_sec,
            "checkpoint_selected_by_stage_local_feasibility_loss_only": True,
        }
        resource = coherent.resource_ledger()["rows"]["full_C12_fixedF3_O16"]
    elif method in (JOINT, SEPARATE, DIRECT):
        f_state = None
        if method != DIRECT:
            f_state = common.exact_grover_state(c_state, screen.final_mask, common.P_FEASIBILITY)
        energy, gamma_count, beta_count = discovery.energy_fn(
            screen, method, c_state if method != DIRECT else None, f_state, objective
        )
        resource = discovery.resource_row(method)
    elif method == FIXED:
        reference = common.fixed_uniform_precursor(screen)
        energy, gamma_count, beta_count = fixed_control._energy(screen, objective, reference)
        resource = fixed_control.resource_ledger()["rows"]
    else:  # pragma: no cover
        raise ValueError(method)

    result = engine.optimize(
        energy, gamma_count, beta_count, OBJECTIVE_EVALUATIONS, method,
        screen.config, screen.real_dtype, screen.device, STREAMS[method]
    )
    metrics = discovery._metrics(screen, result.state, objective)
    common._require(np.all(np.isfinite(metrics["expected_cumulative_loss"])), "nonfinite held-out result")
    body = _result_body(
        manifest, seed, method, result, metrics, result.gamma, result.beta,
        resource, stage2=stage2_payload,
    )
    payload = {**body, "result_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    return payload


def seal(root: Path) -> dict[str, object]:
    """Globally seal every target-blind optimizer row before target scoring."""

    manifest, _ = validated_manifest(root)
    hashes: dict[str, dict[str, str]] = {}
    for seed in TABLE_SEEDS:
        table_hashes: dict[str, str] = {}
        for method in METHODS:
            result = common.load_json(_result_path(root, seed, method))
            body = {key: value for key, value in result.items() if key != "result_sha256"}
            table = _table(manifest, seed)
            expected_table_hash = table["service_cost_compact_json_sha256"]
            expected_contract = manifest["methods"][method]
            expected_resource = expected_contract["resource"]
            gamma_count, beta_count = _method_angle_counts(method)
            gamma = np.asarray(result.get("gamma_by_restart"), dtype=np.float64)
            beta = np.asarray(result.get("beta_by_restart"), dtype=np.float64)
            best_evaluation = np.asarray(
                result.get("best_evaluation_by_restart"), dtype=np.int64
            )
            selected_loss = np.asarray(
                result.get("selected_expected_loss"), dtype=np.float64
            )
            common._require(
                result.get("schema") == f"{SCHEMA}-method-result"
                and result.get("status") == "COMPLETED_TARGET_BLIND_HELDOUT_METHOD"
                and result.get("manifest_sha256") == manifest["manifest_sha256"]
                and result.get("table_seed") == seed
                and result.get("method_id") == method
                and result.get("table_id") == table_id(seed)
                and result.get("coefficient_table_sha256") == expected_table_hash
                and tuple(result.get("restart_seeds", ())) == RESTART_SEEDS
                and result.get("optimizer_stream") == STREAMS[method]
                and result.get("resource") == expected_resource
                and result.get("forward_evaluations_per_restart") == OBJECTIVE_EVALUATIONS
                and result.get("gradient_updates_per_restart") == OBJECTIVE_EVALUATIONS - 1
                and result.get("endpoint_evaluations_per_restart") == 1
                and best_evaluation.shape == (len(RESTART_SEEDS),)
                and bool(np.all((best_evaluation >= 1) & (best_evaluation <= OBJECTIVE_EVALUATIONS)))
                and selected_loss.shape == (len(RESTART_SEEDS),)
                and bool(np.all(np.isfinite(selected_loss)))
                and gamma.shape == (len(RESTART_SEEDS), gamma_count)
                and beta.shape == (len(RESTART_SEEDS), beta_count)
                and bool(np.all(np.isfinite(gamma)))
                and bool(np.all(np.isfinite(beta)))
                and result.get("result_sha256") == common.sha256_json(body)
                and result.get("target_artifact_loaded") is False
                and result.get("target_constructed") is False
                and result.get("target_probability_computed") is False,
                f"invalid held-out result {seed}/{method}",
            )
            if method == FULL:
                stage2 = result.get("stage2", {})
                f_gamma = np.asarray(stage2.get("gamma_by_restart"), dtype=np.float64)
                f_beta = np.asarray(stage2.get("beta_by_restart"), dtype=np.float64)
                f_best = np.asarray(stage2.get("best_evaluation_by_restart"), dtype=np.int64)
                f_exact = np.asarray(stage2.get("exact_pi_feasibility_loss"), dtype=np.float64)
                f_selected = np.asarray(stage2.get("selected_feasibility_loss"), dtype=np.float64)
                common._require(
                    stage2.get("method_id") == variational.STAGE2_METHOD
                    and stage2.get("forward_evaluations_per_restart") == VF3_EVALUATIONS
                    and f_exact.shape == (len(RESTART_SEEDS),)
                    and f_selected.shape == (len(RESTART_SEEDS),)
                    and bool(np.all(np.isfinite(f_exact)))
                    and bool(np.all(np.isfinite(f_selected)))
                    and f_best.shape == (len(RESTART_SEEDS),)
                    and bool(np.all((f_best >= 1) & (f_best <= VF3_EVALUATIONS)))
                    and f_gamma.shape == (len(RESTART_SEEDS), variational.P_F)
                    and f_beta.shape == (len(RESTART_SEEDS), variational.P_F)
                    and bool(np.all(np.isfinite(f_gamma)))
                    and bool(np.all(np.isfinite(f_beta)))
                    and stage2.get("checkpoint_selected_by_stage_local_feasibility_loss_only") is True,
                    f"invalid VF3 budget for {seed}",
                )
            table_hashes[method] = result["result_sha256"]
        hashes[table_id(seed)] = table_hashes
    body = {
        "schema": f"{SCHEMA}-global-seal",
        "status": "SEALED_COMPLETE_UNFILTERED_TARGET_BLIND_HELDOUT_COHORT",
        "manifest_sha256": manifest["manifest_sha256"],
        "table_seeds_in_frozen_order": list(TABLE_SEEDS),
        "table_count": len(TABLE_SEEDS),
        "methods_in_frozen_order": list(METHODS),
        "method_count_per_table": len(METHODS),
        "restart_count_per_table_method": len(RESTART_SEEDS),
        "method_result_sha256_by_table": hashes,
        "all_tables_retained_without_rejection": True,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "seal_sha256": common.sha256_json(body)}
    destination = root / GLOBAL_SEAL_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace global seal: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def estimate() -> dict[str, object]:
    """Conservative measured-runtime projection; performs no optimization."""

    # Seconds per 12-table task, projected from completed 4-restart validation
    # runs on the local RTX 4070 / primary GPU class of devices.
    seconds = {FULL: 507.0, JOINT: 760.0, SEPARATE: 875.0, DIRECT: 6950.0, FIXED: 136.0}
    serial_hours = sum(seconds.values()) * len(TABLE_SEEDS) / 3600.0
    return {
        "table_count": len(TABLE_SEEDS),
        "method_task_count": len(TABLE_SEEDS) * len(METHODS),
        "projected_GPU_hours_serial": serial_hours,
        "projected_wall_hours_two_GPU_balanced": serial_hours / 2.0,
        "projected_wall_hours_four_GPU_balanced": serial_hours / 4.0,
        "dominant_method": DIRECT,
        "direct_fraction_of_projected_GPU_hours": seconds[DIRECT] / sum(seconds.values()),
        "planning_note": "Run one 4-restart table-method per process; direct p64 dominates and should be distributed first.",
    }


def plan_shards(root: Path) -> dict[str, object]:
    """Freeze a balanced two-GPU task queue without executing any task."""

    manifest, _ = validated_manifest(root)
    seconds = {
        FULL: 507.0,
        JOINT: 3_220.0,
        SEPARATE: 3_630.0,
        DIRECT: 6_950.0,
        FIXED: 136.0,
    }
    tasks = [
        {
            "table_seed": seed,
            "table_id": table_id(seed),
            "method": method,
            "estimated_seconds": seconds[method],
        }
        for method in METHODS
        for seed in TABLE_SEEDS
    ]
    tasks.sort(
        key=lambda item: (
            -float(item["estimated_seconds"]),
            int(item["table_seed"]),
            str(item["method"]),
        )
    )
    labels = ("primary_worker", "local_secondary")
    shard_tasks: list[list[dict[str, object]]] = [[], []]
    loads = [0.0, 0.0]
    for task in tasks:
        index = min(range(2), key=lambda value: (loads[value], value))
        shard_tasks[index].append(task)
        loads[index] += float(task["estimated_seconds"])
    body = {
        "schema": f"{SCHEMA}-two-gpu-shard-plan",
        "status": "FROZEN_TARGET_BLIND_TWO_GPU_QUEUE_NOT_LAUNCHED",
        "manifest_sha256": manifest["manifest_sha256"],
        "assignment_rule": "deterministic longest-processing-time greedy assignment; primary GPU wins exact ties",
        "execution_rule": "one task at a time per GPU; all four paired restarts batched within a task",
        "platform_priority": "start and keep primary_worker occupied; local_secondary runs its independent queue concurrently",
        "shards": [
            {
                "shard_index": index,
                "platform_label": labels[index],
                "estimated_seconds": loads[index],
                "estimated_hours": loads[index] / 3600.0,
                "task_count": len(shard_tasks[index]),
                "tasks_in_execution_order": shard_tasks[index],
            }
            for index in range(2)
        ],
        "task_count": len(tasks),
        "exact_cartesian_product_complete": True,
        "duplicate_task_count": 0,
        "optimization_launched": False,
    }
    payload = {**body, "plan_sha256": common.sha256_json(body)}
    destination = root / SHARD_PLAN_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace shard plan: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--validation-root", type=Path, required=True)
    command = sub.add_parser("run-method")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--table-seed", type=int, choices=TABLE_SEEDS, required=True)
    command.add_argument("--method", choices=METHODS, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("plan-shards")
    command.add_argument("--campaign-root", type=Path, required=True)
    sub.add_parser("estimate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.validation_root)
    elif args.command == "run-method":
        payload = run_method(args.campaign_root, args.table_seed, args.method, device=args.device)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    elif args.command == "plan-shards":
        payload = plan_shards(args.campaign_root)
    elif args.command == "estimate":
        payload = estimate()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload.get("status", payload))


if __name__ == "__main__":
    main()
