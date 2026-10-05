#!/usr/bin/env python3
"""Frozen finite fixed-uniform comparator for coherent-IM3 validation.

The circuit starts from the prescribed uniform state on the native two-hot
shell, applies 32 literal Grover iterations using only the exact joint
feasibility indicator, and then optimizes an O16 learned-projector history.
The optimizer never loads or constructs the post-seal best-two target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import target_free_structural_screen as engine
import targeted_coherent_im3_continuation as validation
import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_screen as discovery
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-fixed-uniform-F32-O16-validation-v1"
MANIFEST_NAME = "targeted_coherent_im3_fixed_uniform_manifest.json"
RESULT_NAME = "targeted_coherent_im3_fixed_uniform_result.json"
SEAL_NAME = "targeted_coherent_im3_fixed_uniform_seal.json"
METHOD_ID = "coherent_im3_fixed_uniform_F32_O16"

RESTART_SEEDS = validation.ALL_SEEDS
FEASIBILITY_DEPTH = 32
OBJECTIVE_DEPTH = 16
EVALUATIONS = validation.TOTAL_EVALUATIONS
STREAM = 96_032


def resource_ledger() -> dict[str, object]:
    """Return the exact replay ledger for this implementable circuit."""

    support = coherent.support_ledger()
    feasibility_indicator_ru = int(support["Q_union_C_RU"])
    objective_ru = int(support["objective_RU"])
    precursor = engine.PREP_RU + FEASIBILITY_DEPTH * (
        feasibility_indicator_ru + 2 * engine.PREP_RU + engine.SELECTIVE_RU
    )
    history = engine.StructuralScreen.history_ru(
        precursor, OBJECTIVE_DEPTH, objective_ru
    )
    reported = history + objective_ru
    common._require(precursor == 42_565, "fixed-uniform precursor RU changed")
    common._require(reported == 1_681_185, "fixed-uniform reported RU changed")
    return {
        "finite_implementable_control": True,
        "state_preparation": "native two-hot uniform preparation",
        "feasibility_oracle": "exact joint Q=0 and C=0 indicator",
        "feasibility_oracle_RU": feasibility_indicator_ru,
        "feasibility_depth": FEASIBILITY_DEPTH,
        "fixed_phase_and_reflection_angles": "pi",
        "precursor_circuit_RU": precursor,
        "objective_RU": objective_ru,
        "objective_depth": OBJECTIVE_DEPTH,
        "selective_reflection_RU": engine.SELECTIVE_RU,
        "terminal_objective_RU": objective_ru,
        "rows": {
            "native_C": {"primary_RU_with_O_terminal": reported},
            "uniform_Pauli_C_sensitivity": {
                "primary_RU_with_O_terminal": reported
            },
        },
        "convention_note": (
            "The two C conventions coincide because F32 calls the frozen exact "
            "joint feasibility-indicator oracle; it does not synthesize a "
            "separate native-C or Pauli-expanded C phase."
        ),
    }


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def freeze(root: Path, validation_root: Path) -> dict[str, object]:
    """Freeze provenance and all optimizer choices before this control runs."""

    root.mkdir(parents=True, exist_ok=True)
    source = common.load_json(validation_root / validation.MANIFEST_NAME)
    source_body = {
        key: value for key, value in source.items() if key != "manifest_sha256"
    }
    common._require(
        source.get("schema") == f"{validation.SCHEMA}-manifest"
        and source.get("manifest_sha256") == common.sha256_json(source_body)
        and tuple(source.get("all_restart_seeds", ())) == RESTART_SEEDS
        and source.get("first_three_semantics", {}).get(
            "aggregate_forward_evaluations_per_restart"
        )
        == EVALUATIONS
        and source.get("fourth_restart_semantics", {}).get(
            "forward_evaluations_including_endpoint"
        )
        == EVALUATIONS,
        "source coherent-IM3 validation manifest is invalid",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_TARGET_BLIND_FIXED_UNIFORM_VALIDATION_CONTROL",
        "source_validation_manifest_sha256": source["manifest_sha256"],
        "source_discovery_manifest_sha256": source[
            "discovery_manifest_sha256"
        ],
        "coefficient_table": coherent.coefficient_table(),
        "phase_scale": coherent.PHASE_SCALE,
        "method_id": METHOD_ID,
        "restart_seeds": list(RESTART_SEEDS),
        "restart_count": len(RESTART_SEEDS),
        "optimizer_stream": STREAM,
        "forward_evaluations_per_restart_including_endpoint": EVALUATIONS,
        "gradient_updates_per_restart": EVALUATIONS - 1,
        "optimizer": {"name": "Adam", "learning_rate": 0.035},
        "checkpoint_selection": "minimum expected cumulative loss only",
        "circuit": {
            "initial_state": "prescribed native two-hot uniform shell",
            "feasibility_stage": (
                "32 fixed-pi Grover iterations using only the exact joint "
                "Q=0 and C=0 indicator"
            ),
            "objective_stage": "O16 learned-projector history refinement",
            "objective_phase": "frozen coherent-IM3 concentration-risk diagonal",
        },
        "resource_ledger": resource_ledger(),
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "manifest_sha256": common.sha256_json(body)}
    destination = _manifest_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace manifest: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_manifest(root: Path) -> dict[str, object]:
    manifest = common.load_json(_manifest_path(root))
    body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("status")
        == "FROZEN_TARGET_BLIND_FIXED_UNIFORM_VALIDATION_CONTROL"
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS
        and manifest.get("forward_evaluations_per_restart_including_endpoint")
        == EVALUATIONS
        and manifest.get("coefficient_table") == coherent.coefficient_table()
        and manifest.get("phase_scale") == coherent.PHASE_SCALE
        and manifest.get("resource_ledger") == resource_ledger()
        and manifest.get("target_artifact_loaded") is False
        and manifest.get("target_constructed") is False,
        "fixed-uniform manifest is invalid",
    )
    return manifest


def _make_problem(device: str):
    screen = common.make_screen(
        device=device, dtype="complex64", seeds=RESTART_SEEDS
    )
    objective = discovery.objective_tensor(screen, coherent.coefficient_table())
    reference = common.fixed_uniform_precursor(screen)
    return screen, objective, reference


def _energy(screen, objective, reference):
    spec = common.MethodSpec(
        method_id=METHOD_ID,
        depth=OBJECTIVE_DEPTH,
        phase_names=("O",),
        evaluations=EVALUATIONS,
        stream=STREAM,
        reference_kind="fixed_uniform",
        primary_ru=int(
            resource_ledger()["rows"]["native_C"][
                "primary_RU_with_O_terminal"
            ]
        ),
    )
    return common.history_energy_fn(screen, reference, objective, spec)


def excluded_smoke(*, device: str) -> dict[str, object]:
    """Run exactly two endpoint-inclusive evaluations and write no artifact."""

    screen, objective, reference = _make_problem(device)
    energy, gamma_count, beta_count = _energy(screen, objective, reference)
    result = engine.optimize(
        energy,
        gamma_count,
        beta_count,
        2,
        f"EXCLUDED_SMOKE_{METHOD_ID}",
        screen.config,
        screen.real_dtype,
        screen.device,
        STREAM,
    )
    common._require(
        bool(torch.all(torch.isfinite(result.energy))), "nonfinite smoke result"
    )
    return {
        "status": "EXCLUDED_TWO_EVALUATION_SMOKE_PASS",
        "evaluations_per_restart_including_endpoint": 2,
        "artifact_written": False,
        "target_constructed": False,
    }


def run(root: Path, *, device: str) -> dict[str, object]:
    manifest = validated_manifest(root)
    destination = root / RESULT_NAME
    if destination.exists():
        return common.load_json(destination)

    screen, objective, reference = _make_problem(device)
    feasible_count = int(screen.final_mask.sum().detach().cpu())
    initial_mass = feasible_count / int(engine.DIMENSION)
    analytic_mass = float(common.grover_success([initial_mass], FEASIBILITY_DEPTH)[0])
    precursor_metrics = screen.metrics(reference)
    materialized_mass = np.asarray(
        precursor_metrics["final_mass"], dtype=np.float64
    )
    max_mass_error = float(np.max(np.abs(materialized_mass - analytic_mass)))
    common._require(feasible_count == 390, "exact feasibility indicator changed")
    common._require(max_mass_error <= 2e-5, "fixed-uniform F32 replay failed")

    energy, gamma_count, beta_count = _energy(screen, objective, reference)
    result = engine.optimize(
        energy,
        gamma_count,
        beta_count,
        EVALUATIONS,
        METHOD_ID,
        screen.config,
        screen.real_dtype,
        screen.device,
        STREAM,
    )
    metrics = discovery._metrics(screen, result.state, objective)
    common._require(
        np.all(np.isfinite(np.asarray(metrics["expected_cumulative_loss"]))),
        "fixed-uniform optimizer returned nonfinite loss",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-result",
        "status": "COMPLETED_TARGET_BLIND_FIXED_UNIFORM_VALIDATION_CONTROL",
        "manifest_sha256": manifest["manifest_sha256"],
        "method_id": METHOD_ID,
        "restart_seeds": list(RESTART_SEEDS),
        "optimizer_stream": STREAM,
        "reference": {
            "kind": "prescribed native-uniform exact-feasibility Grover",
            "native_shell_dimension": int(engine.DIMENSION),
            "exact_feasible_count": feasible_count,
            "grover_depth": FEASIBILITY_DEPTH,
            "fixed_phase_and_reflection_angles": "pi",
            "analytic_final_mass": analytic_mass,
            "materialized_final_mass": precursor_metrics["final_mass"],
            "max_mass_error": max_mass_error,
        },
        "objective_depth": OBJECTIVE_DEPTH,
        "forward_evaluations_per_restart_including_endpoint": EVALUATIONS,
        "gradient_updates_per_restart": result.updates_per_restart,
        "endpoint_evaluations_per_restart": result.endpoint_evaluations_per_restart,
        "resource_ledger": resource_ledger(),
        "best_evaluation_by_restart": [
            int(value) for value in result.best_evaluation.cpu()
        ],
        "selected_expected_loss": [float(value) for value in result.energy.cpu()],
        "gamma_by_restart": result.gamma.cpu().tolist(),
        "beta_by_restart": result.beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(result.state),
        "elapsed_sec": result.elapsed_sec,
        "checkpoint_selected_by_expected_loss_only": True,
        "finite_implementable_control": True,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_identity_available": False,
        "target_probability_computed": False,
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    return payload


def seal(root: Path) -> dict[str, object]:
    manifest = validated_manifest(root)
    result = common.load_json(root / RESULT_NAME)
    result_body = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    common._require(
        result.get("schema") == f"{SCHEMA}-result"
        and result.get("status")
        == "COMPLETED_TARGET_BLIND_FIXED_UNIFORM_VALIDATION_CONTROL"
        and result.get("manifest_sha256") == manifest["manifest_sha256"]
        and result.get("result_sha256") == common.sha256_json(result_body)
        and tuple(result.get("restart_seeds", ())) == RESTART_SEEDS
        and result.get("forward_evaluations_per_restart_including_endpoint")
        == EVALUATIONS
        and result.get("target_artifact_loaded") is False
        and result.get("target_constructed") is False
        and result.get("target_probability_computed") is False,
        "fixed-uniform result is invalid",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-seal",
        "status": "SEALED_COMPLETE_TARGET_BLIND_FIXED_UNIFORM_CONTROL",
        "manifest_sha256": manifest["manifest_sha256"],
        "result_sha256": result["result_sha256"],
        "method_id": METHOD_ID,
        "restart_count": len(RESTART_SEEDS),
        "evaluations_per_restart_including_endpoint": EVALUATIONS,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "seal_sha256": common.sha256_json(body)}
    destination = root / SEAL_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace seal: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--validation-root", type=Path, required=True)
    command = sub.add_parser("excluded-smoke")
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("run")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.validation_root)
    elif args.command == "excluded-smoke":
        payload = excluded_smoke(device=args.device)
    elif args.command == "run":
        payload = run(args.campaign_root, device=args.device)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload["status"])


if __name__ == "__main__":
    main()
