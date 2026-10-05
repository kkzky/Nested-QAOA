#!/usr/bin/env python3
"""Frozen four-method, target-blind coherent-IM3 discovery screen."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import stage3_objective_common as objective_engine
import target_free_structural_screen as engine
import targeted_coherent_im3_objective as coherent
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-four-method-discovery-v1"
MANIFEST_NAME = "targeted_coherent_im3_manifest.json"
PRECURSOR_NAME = "targeted_coherent_im3_precursor.json"
SEAL_NAME = "targeted_coherent_im3_optimizer_seal.json"
RESULT_DIRECTORY = "targeted_coherent_im3_method_results"

FULL = "coherent_im3_full_C12_fixedF3_O16"
COLLAPSE_JOINT = "coherent_im3_same_oracle_joint_p19"
COLLAPSE_SEPARATE = "coherent_im3_same_oracle_separate_p19"
DIRECT = "coherent_im3_direct_separate_Q_C_O_p64"
METHODS = (FULL, COLLAPSE_JOINT, COLLAPSE_SEPARATE, DIRECT)

RESTART_SEEDS = common.PILOT_RESTART_SEEDS[:3]
EVALUATIONS = 800


@dataclass(frozen=True)
class Method:
    method_id: str
    depth: int
    phases: tuple[str, ...]
    reference_kind: str
    stream: int
    resource_key: str


def specs() -> dict[str, Method]:
    rows = (
        Method(FULL, 16, ("O",), "fixed_F3_after_saved_C12", 95_001, "full_C12_fixedF3_O16"),
        Method(
            COLLAPSE_JOINT,
            19,
            ("F_indicator+O",),
            "saved_C12",
            95_002,
            "same_oracle_collapse_joint_p19",
        ),
        Method(
            COLLAPSE_SEPARATE,
            19,
            ("F_indicator", "O"),
            "saved_C12",
            95_003,
            "same_oracle_collapse_separate_p19",
        ),
        Method(
            DIRECT,
            64,
            ("Q", "C", "O"),
            "uniform_native_two_hot_shell",
            95_064,
            "direct_block_XY_separate_Q_C_O_p64",
        ),
    )
    return {row.method_id: row for row in rows}


def resource_row(method_id: str) -> dict[str, int]:
    spec = specs()[method_id]
    row = coherent.resource_ledger()["rows"][spec.resource_key]
    return {key: int(value) for key, value in row.items()}


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _result_path(root: Path, method_id: str) -> Path:
    return root / RESULT_DIRECTORY / f"{method_id}.json"


def freeze(root: Path, stage_a_path: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    (root / RESULT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    stage_a_path = stage_a_path.resolve()
    common.stage_a_c12_row(common.load_json(stage_a_path))
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_TARGET_BLIND_COHERENT_IM3_DISCOVERY",
        "stage_a_path": str(stage_a_path),
        "freeze_source": str(
            (Path(__file__).resolve().parent / "FROZEN_COHERENT_IM3_RISK_OBJECTIVE_20260811.md")
        ),
        "coefficient_table": coherent.coefficient_table(),
        "phase_scale_fixed_before_optimizer_outcomes": coherent.PHASE_SCALE,
        "restart_seeds": list(RESTART_SEEDS),
        "restart_count": 3,
        "evaluations_per_restart_including_endpoint": EVALUATIONS,
        "depths": {"C": 12, "fixed_F": 3, "O": 16, "collapse": 19, "direct": 64},
        "fixed_F_angles": [float(common.GROVER_FIXED_ANGLES)] * 3,
        "methods": {
            method_id: {
                "depth": spec.depth,
                "phase_names": list(spec.phases),
                "reference_kind": spec.reference_kind,
                "initialization_stream": spec.stream,
                "resource": resource_row(method_id),
            }
            for method_id, spec in specs().items()
        },
        "support_ledger": coherent.support_ledger(),
        "resource_ledger": coherent.resource_ledger(),
        "optimizer": {"name": "Adam", "learning_rate": 0.035},
        "checkpoint_selection": "minimum expected cumulative loss only",
        "post_lock_gate": {
            "resource_conventions": ["native_C", "uniform_Pauli_C_sensitivity"],
            "strongest_comparator": "per-restart minimum cost envelope of all three controls",
            "minimum_full_wins": 2,
            "minimum_geometric_mean_control_over_full": 1.10,
            "minimum_full_median_final_mass": 0.90,
            "minimum_full_improvements_over_fixed_F3": 2,
        },
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "manifest_sha256": common.sha256_json(body)}
    destination = _manifest_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace a different manifest: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_manifest(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("manifest_sha256") == common.sha256_json(body),
        "coherent-IM3 manifest is invalid",
    )
    common._require(
        manifest.get("coefficient_table") == coherent.coefficient_table()
        and manifest.get("phase_scale_fixed_before_optimizer_outcomes") == coherent.PHASE_SCALE,
        "frozen coherent-IM3 objective changed",
    )
    common._require(
        tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS
        and manifest.get("restart_count") == 3
        and manifest.get("evaluations_per_restart_including_endpoint") == EVALUATIONS,
        "coherent-IM3 restart/evaluation contract changed",
    )
    stage_a = common.load_json(Path(str(manifest["stage_a_path"])))
    common.stage_a_c12_row(stage_a)
    return manifest, stage_a


def materialize(root: Path, *, device: str) -> dict[str, object]:
    manifest, stage_a = validated_manifest(root)
    screen = common.make_screen(device=device, dtype="complex64", seeds=RESTART_SEEDS)
    _, _, record = common.materialize_structural_states(
        screen, common.stage_a_c12_row(stage_a)
    )
    body = {**record, "manifest_sha256": manifest["manifest_sha256"]}
    payload = {**body, "record_sha256": common.sha256_json(body)}
    destination = root / PRECURSOR_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace precursor: {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def objective_tensor(screen: engine.StructuralScreen, table: dict[str, object]) -> torch.Tensor:
    raw = coherent.native_shell_diagonal(screen, table)
    return torch.as_tensor(raw, dtype=screen.real_dtype, device=screen.device) / coherent.PHASE_SCALE


def energy_fn(
    screen: engine.StructuralScreen,
    method_id: str,
    c_state: torch.Tensor | None,
    f_state: torch.Tensor | None,
    objective: torch.Tensor,
):
    spec = specs()[method_id]
    cumulative = screen.common + objective
    if method_id == FULL:
        common._require(f_state is not None, "full method has no fixed-F3 reference")
        reference, phases = f_state, (objective,)
        gamma_count, beta_count = spec.depth, spec.depth

        def energy(gamma: torch.Tensor, beta: torch.Tensor):
            return objective_engine.history_circuit(
                screen, reference, phases, cumulative, gamma, beta, spec.depth
            )

    elif method_id in (COLLAPSE_JOINT, COLLAPSE_SEPARATE):
        common._require(c_state is not None, "collapse method has no saved-C12 reference")
        indicator = screen.final_mask.to(screen.real_dtype)
        phases = (
            (indicator + objective,)
            if method_id == COLLAPSE_JOINT
            else (indicator, objective)
        )
        gamma_count, beta_count = len(phases) * spec.depth, spec.depth

        def energy(gamma: torch.Tensor, beta: torch.Tensor):
            return objective_engine.history_circuit(
                screen, c_state, phases, cumulative, gamma, beta, spec.depth
            )

    elif method_id == DIRECT:
        gamma_count, beta_count = 3 * spec.depth, spec.depth

        def energy(gamma: torch.Tensor, beta: torch.Tensor):
            return screen.block_energy(
                screen.native,
                (screen.q, screen.c, objective),
                cumulative,
                gamma,
                beta,
                spec.depth,
            )

    else:  # pragma: no cover
        raise ValueError(method_id)
    return energy, gamma_count, beta_count


def _metrics(
    screen: engine.StructuralScreen, state: torch.Tensor, objective: torch.Tensor
) -> dict[str, object]:
    base = screen.metrics(state)
    with torch.no_grad():
        expected_o = engine.expectation(state, objective)
        expected_total = engine.expectation(state, screen.common + objective)
    return {
        **base,
        "expected_Otilde": [float(value) for value in expected_o.cpu()],
        "expected_cumulative_loss": [float(value) for value in expected_total.cpu()],
    }


def run_method(root: Path, method_id: str, *, device: str) -> dict[str, object]:
    manifest, stage_a = validated_manifest(root)
    if method_id not in METHODS:
        raise ValueError(method_id)
    destination = _result_path(root, method_id)
    if destination.exists():
        return common.load_json(destination)

    screen = common.make_screen(device=device, dtype="complex64", seeds=RESTART_SEEDS)
    c_state: torch.Tensor | None = None
    f_state: torch.Tensor | None = None
    precursor_sha: str | None = None
    if method_id != DIRECT:
        c_state, f_state, precursor = common.materialize_structural_states(
            screen, common.stage_a_c12_row(stage_a)
        )
        precursor_sha = common.sha256_json(precursor)
    objective = objective_tensor(screen, manifest["coefficient_table"])
    energy, gamma_count, beta_count = energy_fn(
        screen, method_id, c_state, f_state, objective
    )
    spec = specs()[method_id]
    result = engine.optimize(
        energy,
        gamma_count,
        beta_count,
        EVALUATIONS,
        method_id,
        screen.config,
        screen.real_dtype,
        screen.device,
        spec.stream,
    )
    metrics = _metrics(screen, result.state, objective)
    common._require(
        np.all(np.isfinite(metrics["expected_cumulative_loss"])),
        "nonfinite coherent-IM3 outcome",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-method-result",
        "status": "COMPLETED_TARGET_BLIND",
        "manifest_sha256": manifest["manifest_sha256"],
        "method_id": method_id,
        "restart_seeds": list(RESTART_SEEDS),
        "paired_restart_streams": [
            {
                "restart_index": index,
                "seed": int(seed),
                "gamma_initialization_stream": 2 * spec.stream,
                "beta_initialization_stream": 2 * spec.stream + 1,
            }
            for index, seed in enumerate(RESTART_SEEDS)
        ],
        "optimizer_stream": spec.stream,
        "depth": spec.depth,
        "phase_names": list(spec.phases),
        "reference_kind": spec.reference_kind,
        "resource": resource_row(method_id),
        "forward_evaluations_per_restart": EVALUATIONS,
        "gradient_updates_per_restart": result.updates_per_restart,
        "endpoint_evaluations_per_restart": result.endpoint_evaluations_per_restart,
        "best_evaluation_by_restart": [int(value) for value in result.best_evaluation.cpu()],
        "selected_expected_loss": [float(value) for value in result.energy.cpu()],
        "gamma_by_restart": result.gamma.cpu().tolist(),
        "beta_by_restart": result.beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(result.state),
        "trace": result.trace,
        "elapsed_sec": result.elapsed_sec,
        "precursor_record_sha256": precursor_sha,
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
    manifest, _ = validated_manifest(root)
    precursor = common.load_json(root / PRECURSOR_NAME)
    precursor_body = {key: value for key, value in precursor.items() if key != "record_sha256"}
    common._require(
        precursor.get("manifest_sha256") == manifest["manifest_sha256"]
        and precursor.get("record_sha256") == common.sha256_json(precursor_body),
        "invalid coherent-IM3 precursor",
    )
    results: dict[str, dict[str, object]] = {}
    for method_id, spec in specs().items():
        result = common.load_json(_result_path(root, method_id))
        body = {key: value for key, value in result.items() if key != "result_sha256"}
        common._require(
            result.get("schema") == f"{SCHEMA}-method-result"
            and result.get("status") == "COMPLETED_TARGET_BLIND"
            and result.get("method_id") == method_id
            and result.get("manifest_sha256") == manifest["manifest_sha256"]
            and result.get("result_sha256") == common.sha256_json(body),
            f"invalid coherent-IM3 result {method_id}",
        )
        common._require(
            tuple(result.get("restart_seeds", ())) == RESTART_SEEDS
            and result.get("optimizer_stream") == spec.stream
            and result.get("forward_evaluations_per_restart") == EVALUATIONS
            and result.get("target_constructed") is False
            and result.get("target_probability_computed") is False,
            f"{method_id} violated the frozen target-blind protocol",
        )
        results[method_id] = result
    body = {
        "schema": f"{SCHEMA}-seal",
        "status": "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_SCREEN",
        "manifest_sha256": manifest["manifest_sha256"],
        "precursor_record_sha256": precursor["record_sha256"],
        "method_result_sha256": {
            method_id: results[method_id]["result_sha256"] for method_id in METHODS
        },
        "method_count": 4,
        "restart_count_per_method": 3,
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


def smoke(stage_a_path: Path, *, device: str) -> dict[str, object]:
    """One excluded, two-evaluation, one-restart wiring smoke; writes nothing."""

    stage_a = common.load_json(stage_a_path)
    screen = common.make_screen(
        device=device, dtype="complex64", seeds=(RESTART_SEEDS[0],)
    )
    c_state, f_state, _ = common.materialize_structural_states(
        screen, common.stage_a_c12_row(stage_a)
    )
    objective = objective_tensor(screen, coherent.coefficient_table())
    rows = {}
    for method_id, spec in specs().items():
        energy, gamma_count, beta_count = energy_fn(
            screen, method_id, c_state, f_state, objective
        )
        result = engine.optimize(
            energy,
            gamma_count,
            beta_count,
            2,
            f"excluded_smoke_{method_id}",
            screen.config,
            screen.real_dtype,
            screen.device,
            spec.stream,
        )
        rows[method_id] = {
            "evaluations": 2,
            "finite_loss": bool(torch.all(torch.isfinite(result.energy)).item()),
        }
    common._require(all(row["finite_loss"] for row in rows.values()), "smoke failed")
    return {
        "schema": f"{SCHEMA}-excluded-smoke",
        "status": "PASS_EXCLUDED_TWO_EVALUATION_SMOKE",
        "restart_count": 1,
        "rows": rows,
        "artifact_written": False,
        "excluded_from_analysis": True,
        "target_constructed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--stage-a", type=Path, required=True)
    command = sub.add_parser("materialize")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("run-method")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--method", choices=METHODS, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--stage-a", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.stage_a)
    elif args.command == "materialize":
        payload = materialize(args.campaign_root, device=args.device)
    elif args.command == "run-method":
        payload = run_method(args.campaign_root, args.method, device=args.device)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    elif args.command == "smoke":
        payload = smoke(args.stage_a, device=args.device)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload.get("status", payload.get("schema")))


if __name__ == "__main__":
    main()

