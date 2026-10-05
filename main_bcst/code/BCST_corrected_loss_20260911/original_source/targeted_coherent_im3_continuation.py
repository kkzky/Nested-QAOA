#!/usr/bin/env python3
"""Frozen continuation of the sealed coherent-IM3 discovery screen.

Rows 0--2 start from the sealed best angles at 800 evaluations.  No Adam
moments were stored, so they explicitly use fresh Adam state for 1,600 new
endpoint-inclusive forward evaluations.  The prespecified fourth restart uses
the original method initialization stream for one uninterrupted 2,400-
evaluation run.  No target artifact is loaded by this executable.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import target_free_structural_screen as engine
import targeted_coherent_im3_screen as discovery
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-four-method-validation-continuation-v1"
MANIFEST_NAME = "targeted_coherent_im3_continuation_manifest.json"
SEAL_NAME = "targeted_coherent_im3_continuation_seal.json"
RESULT_DIRECTORY = "targeted_coherent_im3_continuation_results"
METHODS = discovery.METHODS
FIRST_THREE_SEEDS = discovery.RESTART_SEEDS
FOURTH_SEED = common.PILOT_RESTART_SEEDS[3]
ALL_SEEDS = (*FIRST_THREE_SEEDS, FOURTH_SEED)
PARENT_EVALUATIONS = 800
CONTINUATION_EVALUATIONS = 1_600
FOURTH_EVALUATIONS = 2_400
TOTAL_EVALUATIONS = 2_400


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _result_path(root: Path, method_id: str) -> Path:
    return root / RESULT_DIRECTORY / f"{method_id}.json"


def _validate_discovery(discovery_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest, _ = discovery.validated_manifest(discovery_root)
    seal = common.load_json(discovery_root / discovery.SEAL_NAME)
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    common._require(
        seal.get("schema") == f"{discovery.SCHEMA}-seal"
        and seal.get("status") == "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_SCREEN"
        and seal.get("manifest_sha256") == manifest["manifest_sha256"]
        and seal.get("seal_sha256") == common.sha256_json(body)
        and seal.get("method_count") == 4
        and seal.get("restart_count_per_method") == 3
        and seal.get("evaluations_per_restart_including_endpoint") == PARENT_EVALUATIONS
        and seal.get("target_constructed") is False,
        "sealed coherent-IM3 discovery campaign is invalid",
    )
    return manifest, seal


def freeze(root: Path, discovery_root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    (root / RESULT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    discovery_root = discovery_root.resolve()
    manifest, seal = _validate_discovery(discovery_root)
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_TARGET_BLIND_COHERENT_IM3_VALIDATION_CONTINUATION",
        "discovery_root": str(discovery_root),
        "discovery_manifest_sha256": manifest["manifest_sha256"],
        "discovery_optimizer_seal_sha256": seal["seal_sha256"],
        "methods": list(METHODS),
        "first_three_restart_seeds": list(FIRST_THREE_SEEDS),
        "prespecified_fourth_restart_seed": FOURTH_SEED,
        "all_restart_seeds": list(ALL_SEEDS),
        "first_three_semantics": {
            "initial_angles": "sealed best angles after the 800-evaluation discovery screen",
            "optimizer_state": "fresh Adam; parent optimizer moments were not stored",
            "new_forward_evaluations_per_restart_including_endpoint": CONTINUATION_EVALUATIONS,
            "new_gradient_updates_per_restart": CONTINUATION_EVALUATIONS - 1,
            "aggregate_forward_evaluations_per_restart": TOTAL_EVALUATIONS,
            "checkpoint_selection": "minimum target-blind cumulative loss over sealed parent and continuation",
        },
        "fourth_restart_semantics": {
            "initial_angles": "original prespecified method stream and fourth seed",
            "optimizer_state": "one uninterrupted fresh Adam run",
            "forward_evaluations_including_endpoint": FOURTH_EVALUATIONS,
            "gradient_updates": FOURTH_EVALUATIONS - 1,
        },
        "optimizer": {"name": "Adam", "learning_rate": 0.035},
        "post_lock_gate": {
            "minimum_full_wins": 3,
            "minimum_geometric_mean_control_over_full": 1.20,
            "resource_conventions": ["native_C", "uniform_Pauli_C_sensitivity"],
            "strongest_comparator": "per-restart minimum cost envelope of all three controls",
            "minimum_full_median_final_mass": 0.90,
            "minimum_full_improvements_over_fixed_F3": 3,
        },
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "manifest_sha256": common.sha256_json(body)}
    destination = _manifest_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace continuation manifest: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_manifest(root: Path) -> tuple[dict[str, object], Path, dict[str, object], dict[str, object]]:
    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and tuple(manifest.get("all_restart_seeds", ())) == ALL_SEEDS
        and manifest.get("target_artifact_loaded") is False
        and manifest.get("target_constructed") is False,
        "coherent-IM3 continuation manifest is invalid",
    )
    discovery_root = Path(str(manifest["discovery_root"]))
    discovery_manifest, discovery_seal = _validate_discovery(discovery_root)
    common._require(
        manifest.get("discovery_manifest_sha256") == discovery_manifest["manifest_sha256"]
        and manifest.get("discovery_optimizer_seal_sha256") == discovery_seal["seal_sha256"],
        "continuation belongs to a different discovery campaign",
    )
    return manifest, discovery_root, discovery_manifest, discovery_seal


def _optimize_from_angles(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    gamma_initial: torch.Tensor,
    beta_initial: torch.Tensor,
    evaluations: int,
    label: str,
    config: engine.ScreenConfig,
) -> engine.OptimizationResult:
    """Fresh-Adam continuation from saved angles with endpoint accounting."""

    if evaluations < 2:
        raise ValueError("continuation needs at least two evaluations")
    gamma = gamma_initial.detach().clone().requires_grad_(True)
    beta = beta_initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((gamma, beta), lr=config.learning_rate)
    restarts = gamma.shape[0]
    best_energy = torch.full(
        (restarts,), math.inf, dtype=gamma.dtype, device=gamma.device
    )
    best_state: torch.Tensor | None = None
    best_gamma = torch.zeros_like(gamma)
    best_beta = torch.zeros_like(beta)
    best_eval = torch.zeros(restarts, dtype=torch.int64, device=gamma.device)
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
        if evaluation == 1 or endpoint or evaluation % config.trace_every == 0:
            trace.append(
                {
                    "evaluation_within_continuation": evaluation,
                    "aggregate_evaluation": PARENT_EVALUATIONS + evaluation,
                    "post_update_endpoint": endpoint,
                    "loss_by_restart": [float(value) for value in detached.cpu()],
                }
            )

    for evaluation in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        energies, states = energy_fn(gamma, beta)
        inspect(energies, states, evaluation, False)
        energies.sum().backward()
        optimizer.step()
    with torch.no_grad():
        energies, states = energy_fn(gamma, beta)
        inspect(energies, states, evaluations, True)
    if best_state is None:
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


def _energy_for(
    screen: engine.StructuralScreen,
    method_id: str,
    stage_a: dict[str, object],
    objective_table: dict[str, object],
):
    c_state = None
    f_state = None
    if method_id != discovery.DIRECT:
        c_state, f_state, _ = common.materialize_structural_states(
            screen, common.stage_a_c12_row(stage_a)
        )
    objective = discovery.objective_tensor(screen, objective_table)
    energy, gamma_count, beta_count = discovery.energy_fn(
        screen, method_id, c_state, f_state, objective
    )
    return energy, objective, gamma_count, beta_count


def run_method(root: Path, method_id: str, *, device: str) -> dict[str, object]:
    manifest, discovery_root, discovery_manifest, discovery_seal = validated_manifest(root)
    if method_id not in METHODS:
        raise ValueError(method_id)
    destination = _result_path(root, method_id)
    if destination.exists():
        return common.load_json(destination)

    parent = common.load_json(
        discovery_root / discovery.RESULT_DIRECTORY / f"{method_id}.json"
    )
    common._require(
        parent.get("result_sha256")
        == discovery_seal["method_result_sha256"].get(method_id)
        and tuple(parent.get("restart_seeds", ())) == FIRST_THREE_SEEDS
        and parent.get("forward_evaluations_per_restart") == PARENT_EVALUATIONS,
        f"invalid parent result for {method_id}",
    )
    stage_a = common.load_json(Path(str(discovery_manifest["stage_a_path"])))
    table = discovery_manifest["coefficient_table"]

    screen3 = common.make_screen(
        device=device, dtype="complex64", seeds=FIRST_THREE_SEEDS
    )
    energy3, _, gamma_count, beta_count = _energy_for(
        screen3, method_id, stage_a, table
    )
    parent_gamma = torch.as_tensor(
        parent["gamma_by_restart"], dtype=screen3.real_dtype, device=screen3.device
    )
    parent_beta = torch.as_tensor(
        parent["beta_by_restart"], dtype=screen3.real_dtype, device=screen3.device
    )
    common._require(
        parent_gamma.shape == (3, gamma_count)
        and parent_beta.shape == (3, beta_count),
        "parent saved-angle shape changed",
    )
    with torch.no_grad():
        parent_energy_replay, parent_state = energy3(parent_gamma, parent_beta)
    common._require(
        common.state_sha256(parent_state) == parent.get("state_sha256"),
        "parent saved-angle replay mismatch",
    )
    expected_parent = torch.as_tensor(
        parent["selected_expected_loss"],
        dtype=screen3.real_dtype,
        device=screen3.device,
    )
    common._require(
        bool(torch.allclose(parent_energy_replay, expected_parent, atol=2e-6, rtol=2e-6)),
        "parent saved losses do not replay",
    )

    continued = _optimize_from_angles(
        energy3,
        parent_gamma,
        parent_beta,
        CONTINUATION_EVALUATIONS,
        f"continuation_{method_id}",
        screen3.config,
    )
    use_continuation = continued.energy < parent_energy_replay
    selected_gamma3 = torch.where(
        use_continuation[:, None], continued.gamma, parent_gamma
    )
    selected_beta3 = torch.where(
        use_continuation[:, None], continued.beta, parent_beta
    )
    parent_best_eval = torch.as_tensor(
        parent["best_evaluation_by_restart"],
        dtype=torch.int64,
        device=screen3.device,
    )
    aggregate_best3 = torch.where(
        use_continuation,
        PARENT_EVALUATIONS + continued.best_evaluation,
        parent_best_eval,
    )

    screen1 = common.make_screen(
        device=device, dtype="complex64", seeds=(FOURTH_SEED,)
    )
    energy1, _, gamma_count1, beta_count1 = _energy_for(
        screen1, method_id, stage_a, table
    )
    common._require(
        gamma_count1 == gamma_count and beta_count1 == beta_count,
        "fourth-restart angle shape changed",
    )
    spec = discovery.specs()[method_id]
    fourth = engine.optimize(
        energy1,
        gamma_count,
        beta_count,
        FOURTH_EVALUATIONS,
        f"fourth_restart_{method_id}",
        screen1.config,
        screen1.real_dtype,
        screen1.device,
        spec.stream,
    )

    selected_gamma = torch.cat((selected_gamma3, fourth.gamma), dim=0)
    selected_beta = torch.cat((selected_beta3, fourth.beta), dim=0)
    screen4 = common.make_screen(device=device, dtype="complex64", seeds=ALL_SEEDS)
    energy4, objective4, gamma_count4, beta_count4 = _energy_for(
        screen4, method_id, stage_a, table
    )
    common._require(
        selected_gamma.shape == (4, gamma_count4)
        and selected_beta.shape == (4, beta_count4),
        "combined continuation angle shape changed",
    )
    with torch.no_grad():
        selected_energy, selected_state = energy4(selected_gamma, selected_beta)
    metrics = discovery._metrics(screen4, selected_state, objective4)
    common._require(
        np.all(np.isfinite(metrics["expected_cumulative_loss"])),
        "nonfinite coherent-IM3 continuation result",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-method-result",
        "status": "COMPLETED_TARGET_BLIND_VALIDATION_CONTINUATION",
        "manifest_sha256": manifest["manifest_sha256"],
        "discovery_parent_result_sha256": parent["result_sha256"],
        "method_id": method_id,
        "restart_seeds": list(ALL_SEEDS),
        "optimizer_stream": spec.stream,
        "depth": spec.depth,
        "phase_names": list(spec.phases),
        "reference_kind": spec.reference_kind,
        "resource": discovery.resource_row(method_id),
        "first_three": {
            "parent_forward_evaluations_per_restart": PARENT_EVALUATIONS,
            "new_forward_evaluations_per_restart_including_endpoint": CONTINUATION_EVALUATIONS,
            "new_gradient_updates_per_restart": continued.updates_per_restart,
            "new_endpoint_evaluations_per_restart": continued.endpoint_evaluations_per_restart,
            "aggregate_forward_evaluations_per_restart": TOTAL_EVALUATIONS,
            "optimizer_state_semantics": "fresh Adam from sealed best angles; no parent moments stored",
            "starting_parent_best_evaluation": [int(value) for value in parent_best_eval.cpu()],
            "continuation_best_evaluation": [int(value) for value in continued.best_evaluation.cpu()],
            "selected_from_continuation": [bool(value) for value in use_continuation.cpu()],
            "aggregate_selected_best_evaluation": [int(value) for value in aggregate_best3.cpu()],
            "elapsed_sec": continued.elapsed_sec,
            "trace": continued.trace,
        },
        "fourth_restart": {
            "seed": FOURTH_SEED,
            "forward_evaluations_including_endpoint": FOURTH_EVALUATIONS,
            "gradient_updates": fourth.updates_per_restart,
            "endpoint_evaluations": fourth.endpoint_evaluations_per_restart,
            "best_evaluation": int(fourth.best_evaluation[0].cpu()),
            "elapsed_sec": fourth.elapsed_sec,
            "trace": fourth.trace,
        },
        "aggregate_forward_evaluations_per_restart": [TOTAL_EVALUATIONS] * 4,
        "best_evaluation_by_restart": [
            *[int(value) for value in aggregate_best3.cpu()],
            int(fourth.best_evaluation[0].cpu()),
        ],
        "selected_expected_loss": [float(value) for value in selected_energy.cpu()],
        "gamma_by_restart": selected_gamma.cpu().tolist(),
        "beta_by_restart": selected_beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(selected_state),
        "checkpoint_selected_by_expected_loss_only": True,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_identity_available": False,
        "target_probability_computed": False,
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    return payload


def seal(root: Path) -> dict[str, object]:
    manifest, _, _, _ = validated_manifest(root)
    results: dict[str, dict[str, object]] = {}
    for method_id in METHODS:
        result = common.load_json(_result_path(root, method_id))
        body = {key: value for key, value in result.items() if key != "result_sha256"}
        common._require(
            result.get("schema") == f"{SCHEMA}-method-result"
            and result.get("status") == "COMPLETED_TARGET_BLIND_VALIDATION_CONTINUATION"
            and result.get("method_id") == method_id
            and result.get("manifest_sha256") == manifest["manifest_sha256"]
            and result.get("result_sha256") == common.sha256_json(body)
            and tuple(result.get("restart_seeds", ())) == ALL_SEEDS
            and result.get("aggregate_forward_evaluations_per_restart") == [TOTAL_EVALUATIONS] * 4
            and result.get("target_artifact_loaded") is False
            and result.get("target_constructed") is False,
            f"invalid coherent-IM3 continuation result {method_id}",
        )
        results[method_id] = result
    body = {
        "schema": f"{SCHEMA}-seal",
        "status": "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_VALIDATION",
        "manifest_sha256": manifest["manifest_sha256"],
        "method_result_sha256": {
            method_id: results[method_id]["result_sha256"] for method_id in METHODS
        },
        "method_count": 4,
        "restart_count_per_method": 4,
        "aggregate_evaluations_per_restart": TOTAL_EVALUATIONS,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "seal_sha256": common.sha256_json(body)}
    destination = root / SEAL_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace continuation seal: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--discovery-root", type=Path, required=True)
    command = sub.add_parser("run-method")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--method", choices=METHODS, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.discovery_root)
    elif args.command == "run-method":
        payload = run_method(args.campaign_root, args.method, device=args.device)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload["status"])


if __name__ == "__main__":
    main()

