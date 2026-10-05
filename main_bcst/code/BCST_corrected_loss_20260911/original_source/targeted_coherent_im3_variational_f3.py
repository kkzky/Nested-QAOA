"""Contingent genuine three-optimized-stage coherent-IM3 follow-up.

This executable is intentionally isolated from the coherent-IM3 discovery and
validation campaign roots.  ``freeze`` is enabled only after the sealed
four-restart validation passes its prespecified post-lock gate.  The target-
aware validation score is inspected only by ``freeze``; neither optimizer
worker loads a target identity or computes a target probability.

Stage 1 replays the saved C12 circuit.  Stage 2 evaluates the exact all-pi F3
schedule, then optimizes three independent feasibility phases and three
independent history-mixer angles from a deterministic target-independent
1e-3-radian symmetry break.  The exact all-pi point remains eligible for
checkpoint selection.  Stage 3 optimizes O16 for 2,400 endpoint-inclusive
forward evaluations per restart with the validated full method's four paired
seeds and initialization stream.  Variational angles do not change sampling
RU, so both resource ledgers are identical to the validated fixed-F3 full row.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import stage3_objective_common as objective_engine
import target_free_structural_screen as engine
import targeted_coherent_im3_continuation as validation
import targeted_coherent_im3_continuation_score as validation_score
import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_screen as discovery
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-variational-f3-followup-v2"
MANIFEST_NAME = "targeted_coherent_im3_variational_f3_manifest.json"
RESULT_DIRECTORY = "targeted_coherent_im3_variational_f3_results"
STAGE2_NAME = "coherent_im3_variational_F3.json"
OBJECTIVE_NAME = "coherent_im3_full_C12_VF3_O16.json"
SEAL_NAME = "targeted_coherent_im3_variational_f3_optimizer_seal.json"

STAGE2_METHOD = "coherent_im3_variational_F3"
FULL_METHOD = "coherent_im3_full_C12_VF3_O16"
P_F = 3
P_O = 16
STAGE2_EVALUATIONS = 800
OBJECTIVE_EVALUATIONS = 2_400
LEARNING_RATE = 0.035
STAGE2_PERTURBATION_MAGNITUDE = 1e-3
STAGE2_PERTURBATION_STREAM = 95_101
RESTART_SEEDS = validation.ALL_SEEDS


def _stage2_perturbation_signs(
    seeds: tuple[int, ...] = RESTART_SEEDS,
) -> list[list[int]]:
    """Return one deterministic six-component Rademacher row per restart."""

    rows: list[list[int]] = []
    for seed in seeds:
        rng = np.random.default_rng(
            int(seed) + 1_000_003 * STAGE2_PERTURBATION_STREAM
        )
        bits = rng.integers(0, 2, size=2 * P_F, dtype=np.int8)
        rows.append([int(2 * value - 1) for value in bits])
    return rows


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _result_path(root: Path, name: str) -> Path:
    return root / RESULT_DIRECTORY / name


def _validated_parent_seal(
    parent_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    dict[str, object],
    dict[str, object],
]:
    """Validate the target-blind coherent validation without reading its score."""

    (
        manifest,
        discovery_root,
        discovery_manifest,
        discovery_seal,
    ) = validation.validated_manifest(parent_root)
    seal = common.load_json(parent_root / validation.SEAL_NAME)
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    common._require(
        seal.get("schema") == f"{validation.SCHEMA}-seal"
        and seal.get("status")
        == "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_VALIDATION"
        and seal.get("manifest_sha256") == manifest.get("manifest_sha256")
        and seal.get("seal_sha256") == common.sha256_json(body)
        and seal.get("method_count") == 4
        and seal.get("restart_count_per_method") == 4
        and seal.get("aggregate_evaluations_per_restart")
        == validation.TOTAL_EVALUATIONS
        and seal.get("target_constructed") is False
        and seal.get("target_probability_computed") is False,
        "sealed coherent-IM3 validation campaign is invalid or incomplete",
    )
    return manifest, seal, discovery_root, discovery_manifest, discovery_seal


def _validated_parent_score(
    parent_root: Path,
    manifest: dict[str, object],
    seal: dict[str, object],
    discovery_manifest: dict[str, object],
    discovery_seal: dict[str, object],
) -> dict[str, object]:
    """Require the validation pass and a complete collapse/direct envelope."""

    score = common.load_json(parent_root / validation_score.OUTPUT_NAME)
    body = {key: value for key, value in score.items() if key != "score_sha256"}
    comparisons = score.get("comparisons_by_resource_convention", {})
    rows = score.get("rows", {})
    common._require(
        score.get("schema") == f"{validation.SCHEMA}-post-seal-score"
        and score.get("status") == "COHERENT_IM3_VALIDATION_GATE_PASS"
        and score.get("manifest_sha256") == manifest.get("manifest_sha256")
        and score.get("optimizer_seal_sha256") == seal.get("seal_sha256")
        and score.get("discovery_score_sha256") is not None
        and score.get("score_sha256") == common.sha256_json(body)
        and score.get("gate", {}).get("pass") is True
        and set(rows) == set(discovery.METHODS),
        "variational-F3 follow-up requires a passed coherent-IM3 validation",
    )
    for convention in ("native_C", "uniform_Pauli_C_sensitivity"):
        comparison = comparisons.get(convention, {})
        envelope = comparison.get(
            "strongest_comparator_envelope_RTS99_x_RU"
        )
        common._require(
            isinstance(envelope, list)
            and len(envelope) == 4
            and comparison.get("full_paired_wins", 0) >= 3
            and comparison.get("geometric_mean_control_over_full") is not None
            and float(comparison["geometric_mean_control_over_full"]) >= 1.20,
            f"validated {convention} collapse/direct envelope is incomplete",
        )
        expected: list[int | None] = []
        controls = (
            discovery.COLLAPSE_JOINT,
            discovery.COLLAPSE_SEPARATE,
            discovery.DIRECT,
        )
        for restart in range(4):
            finite = [
                int(rows[method]["resource_conventions"][convention][
                    "RTS99_x_RU"
                ][restart])
                for method in controls
                if rows[method]["resource_conventions"][convention][
                    "RTS99_x_RU"
                ][restart]
                is not None
            ]
            expected.append(min(finite) if finite else None)
        common._require(
            expected == envelope,
            f"validated {convention} comparator envelope is inconsistent",
        )
    common._require(
        manifest.get("discovery_manifest_sha256")
        == discovery_manifest.get("manifest_sha256")
        and manifest.get("discovery_optimizer_seal_sha256")
        == discovery_seal.get("seal_sha256"),
        "validation/discovery binding changed",
    )
    return score


def freeze(root: Path, parent_root: Path) -> dict[str, object]:
    """Freeze the contingent target-blind follow-up in a new campaign root."""

    root = root.resolve()
    parent_root = parent_root.resolve()
    common._require(root != parent_root, "follow-up and parent roots must differ")
    (
        parent_manifest,
        parent_seal,
        discovery_root,
        discovery_manifest,
        discovery_seal,
    ) = _validated_parent_seal(parent_root)
    parent_score = _validated_parent_score(
        parent_root,
        parent_manifest,
        parent_seal,
        discovery_manifest,
        discovery_seal,
    )
    paired_stream = int(
        discovery_manifest["methods"][discovery.FULL]["initialization_stream"]
    )
    common._require(
        paired_stream == discovery.specs()[discovery.FULL].stream,
        "validated full-method initialization stream changed",
    )
    resource = discovery.resource_row(discovery.FULL)
    common._require(
        resource
        == coherent.resource_ledger()["rows"]["full_C12_fixedF3_O16"],
        "coherent-IM3 fixed-F3 full resource row changed",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_TARGET_BLIND_COHERENT_IM3_VARIATIONAL_F3_FOLLOWUP",
        "artifact_namespace": SCHEMA,
        "parent_validation_root": str(parent_root),
        "parent_validation_manifest_sha256": parent_manifest["manifest_sha256"],
        "parent_validation_optimizer_seal_sha256": parent_seal["seal_sha256"],
        "parent_validation_score_sha256": parent_score["score_sha256"],
        "parent_validation_gate_passed_before_freeze": True,
        "discovery_root": str(discovery_root),
        "discovery_manifest_sha256": discovery_manifest["manifest_sha256"],
        "discovery_optimizer_seal_sha256": discovery_seal["seal_sha256"],
        "stage_a_path": discovery_manifest["stage_a_path"],
        "coefficient_table": discovery_manifest["coefficient_table"],
        "phase_scale": discovery_manifest[
            "phase_scale_fixed_before_optimizer_outcomes"
        ],
        "restart_seeds": list(RESTART_SEEDS),
        "restart_count": 4,
        "depths": {"C": 12, "variational_F": P_F, "O": P_O},
        "stage2": {
            "method_id": STAGE2_METHOD,
            "phase_hamiltonian": "I - exact joint-feasibility projector",
            "loss": "1 - exact joint-feasible probability",
            "mixer": "history-state mixer about frozen C12 state",
            "independent_gamma_count": P_F,
            "independent_beta_count": P_F,
            "evaluation_1_gamma": [float(math.pi)] * P_F,
            "evaluation_1_beta": [float(math.pi)] * P_F,
            "perturbation": {
                "kind": (
                    "one deterministic target-independent Rademacher "
                    "direction per restart"
                ),
                "stream": STAGE2_PERTURBATION_STREAM,
                "magnitude_radians": STAGE2_PERTURBATION_MAGNITUDE,
                "signed_direction_gamma_then_beta_by_restart": (
                    _stage2_perturbation_signs()
                ),
                "optimizer_start": "pi plus signed magnitude",
            },
            "evaluations_per_restart_including_exact_pi_and_endpoint": (
                STAGE2_EVALUATIONS
            ),
            "evaluation_accounting": {
                "exact_pi_candidate_without_update": 1,
                "perturbed_branch_pre_update_evaluations": (
                    STAGE2_EVALUATIONS - 2
                ),
                "perturbed_branch_post_update_endpoint": 1,
                "gradient_updates": STAGE2_EVALUATIONS - 2,
            },
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
            "checkpoint_selection": (
                "minimum stage-local feasibility loss over exact-pi candidate "
                "and perturbed optimization trajectory"
            ),
        },
        "stage3": {
            "method_id": FULL_METHOD,
            "phase_names": ["coherent_IM3_O"],
            "mixer": "history-state mixer about frozen variational-F3 state",
            "evaluations_per_restart_including_endpoint": OBJECTIVE_EVALUATIONS,
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
            "paired_validation_initialization_stream": paired_stream,
            "paired_validation_restart_seeds": list(RESTART_SEEDS),
            "checkpoint_selection": "minimum expected cumulative loss only",
        },
        "resource": {
            **resource,
            "same_sampling_RU_as_validated_fixed_F3_full": True,
            "stage2_classical_forward_evaluations_per_restart": (
                STAGE2_EVALUATIONS
            ),
            "stage3_classical_forward_evaluations_per_restart": (
                OBJECTIVE_EVALUATIONS
            ),
        },
        "comparison_contract": {
            "source": "passed four-restart coherent-IM3 validation score",
            "strongest_control": (
                "per-restart minimum RTS99xRU over joint collapse, separate "
                "collapse, and direct p64"
            ),
            "resource_conventions": [
                "native_C",
                "uniform_Pauli_C_sensitivity",
            ],
        },
        "optimization_scope": {
            "pipeline_stages_optimized": 3,
            "preoptimized_C12_replayed_in_followup": True,
            "stages_optimized_within_followup": 2,
            "training_mode": (
                "sequential stage-local optimization; not joint end-to-end training"
            ),
            "new_forward_evaluations_per_restart": (
                STAGE2_EVALUATIONS + OBJECTIVE_EVALUATIONS
            ),
        },
        "development_instance_only": True,
        "selected_after_validation_pass": True,
        "target_identity_stored_in_manifest": False,
        "target_available_to_optimizer_workers": False,
        "target_probability_computed_by_optimizer_workers": False,
    }
    payload = {**body, "manifest_sha256": common.sha256_json(body)}
    root.mkdir(parents=True, exist_ok=True)
    (root / RESULT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    destination = _manifest_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different manifest: {destination}"
            )
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def _validated_manifest(
    root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Validate target-blind inputs; deliberately do not read the parent score."""

    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and manifest.get("parent_validation_gate_passed_before_freeze") is True,
        "coherent-IM3 variational-F3 manifest is invalid",
    )
    parent_root = Path(str(manifest["parent_validation_root"]))
    common._require(
        root.resolve() != parent_root.resolve(),
        "follow-up and parent roots must remain isolated",
    )
    (
        parent_manifest,
        parent_seal,
        discovery_root,
        discovery_manifest,
        discovery_seal,
    ) = _validated_parent_seal(parent_root)
    common._require(
        manifest.get("parent_validation_manifest_sha256")
        == parent_manifest.get("manifest_sha256")
        and manifest.get("parent_validation_optimizer_seal_sha256")
        == parent_seal.get("seal_sha256")
        and manifest.get("discovery_root") == str(discovery_root)
        and manifest.get("discovery_manifest_sha256")
        == discovery_manifest.get("manifest_sha256")
        and manifest.get("discovery_optimizer_seal_sha256")
        == discovery_seal.get("seal_sha256")
        and manifest.get("coefficient_table")
        == discovery_manifest.get("coefficient_table")
        and float(manifest.get("phase_scale", math.nan)) == coherent.PHASE_SCALE
        and tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS,
        "frozen coherent-IM3 follow-up inputs changed",
    )
    stage2 = manifest.get("stage2", {})
    perturbation = stage2.get("perturbation", {})
    accounting = stage2.get("evaluation_accounting", {})
    common._require(
        stage2.get("evaluation_1_gamma") == [float(math.pi)] * P_F
        and stage2.get("evaluation_1_beta") == [float(math.pi)] * P_F
        and perturbation.get("stream") == STAGE2_PERTURBATION_STREAM
        and float(perturbation.get("magnitude_radians", math.nan))
        == STAGE2_PERTURBATION_MAGNITUDE
        and perturbation.get("signed_direction_gamma_then_beta_by_restart")
        == _stage2_perturbation_signs()
        and stage2.get("evaluations_per_restart_including_exact_pi_and_endpoint")
        == STAGE2_EVALUATIONS
        and accounting.get("exact_pi_candidate_without_update") == 1
        and accounting.get("perturbed_branch_pre_update_evaluations")
        == STAGE2_EVALUATIONS - 2
        and accounting.get("perturbed_branch_post_update_endpoint") == 1
        and accounting.get("gradient_updates") == STAGE2_EVALUATIONS - 2,
        "frozen Stage-2 symmetry break or evaluation accounting changed",
    )
    stage_a = common.load_json(Path(str(discovery_manifest["stage_a_path"])))
    common.stage_a_c12_row(stage_a)
    return manifest, parent_manifest, parent_seal, discovery_manifest, stage_a


def _feasibility_energy(
    screen: engine.StructuralScreen, c_state: torch.Tensor
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    infeasible = (~screen.final_mask).to(screen.real_dtype)

    def energy(gamma: torch.Tensor, beta: torch.Tensor):
        return objective_engine.history_circuit(
            screen,
            c_state,
            (infeasible,),
            infeasible,
            gamma,
            beta,
            P_F,
        )

    return energy


def _optimize_from_pi(
    energy_fn: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    screen: engine.StructuralScreen,
    evaluations: int,
) -> tuple[engine.OptimizationResult, torch.Tensor]:
    """Retain exact pi, then optimize from the frozen signed perturbation."""

    if evaluations < 3:
        raise ValueError("pi, at least one perturbed update, and endpoint required")
    shape = (screen.config.restarts, P_F)
    pi_gamma = torch.full(
        shape, math.pi, dtype=screen.real_dtype, device=screen.device
    )
    pi_beta = torch.full(
        shape, math.pi, dtype=screen.real_dtype, device=screen.device
    )
    signs = torch.as_tensor(
        _stage2_perturbation_signs(
            tuple(int(seed) for seed in screen.config.initialization_seeds)
        ),
        dtype=screen.real_dtype,
        device=screen.device,
    )
    common._require(
        signs.shape == (screen.config.restarts, 2 * P_F)
        and bool(torch.all(torch.abs(signs) == 1).item()),
        "invalid frozen Stage-2 perturbation directions",
    )
    gamma = (
        pi_gamma + STAGE2_PERTURBATION_MAGNITUDE * signs[:, :P_F]
    ).requires_grad_(True)
    beta = (
        pi_beta + STAGE2_PERTURBATION_MAGNITUDE * signs[:, P_F:]
    ).requires_grad_(True)
    optimizer = torch.optim.Adam((gamma, beta), lr=LEARNING_RATE)
    with torch.no_grad():
        pi_energy, pi_state = energy_fn(pi_gamma, pi_beta)
    common._require(
        bool(torch.all(torch.isfinite(pi_energy)).item()),
        "nonfinite exact-pi feasibility candidate",
    )
    best_energy = pi_energy.detach().clone()
    best_state = pi_state.detach().clone()
    best_gamma = pi_gamma.clone()
    best_beta = pi_beta.clone()
    best_evaluation = torch.ones(
        screen.config.restarts, dtype=torch.int64, device=screen.device
    )
    trace: list[dict[str, object]] = [
        {
            "evaluation": 1,
            "candidate_kind": "exact_pi_no_update",
            "post_update_endpoint": False,
            "feasibility_loss_by_restart": [
                float(value) for value in pi_energy.detach().cpu()
            ],
        }
    ]
    started = time.time()

    def inspect(
        energies: torch.Tensor,
        states: torch.Tensor,
        evaluation: int,
        endpoint: bool,
    ) -> None:
        nonlocal best_energy, best_state
        detached = energies.detach()
        common._require(
            bool(torch.all(torch.isfinite(detached)).item()),
            f"nonfinite perturbed feasibility loss at evaluation {evaluation}",
        )
        improved = detached < best_energy
        if torch.any(improved):
            best_state[improved] = states.detach()[improved]
        best_energy = torch.where(improved, detached, best_energy)
        best_gamma[improved] = gamma.detach()[improved]
        best_beta[improved] = beta.detach()[improved]
        best_evaluation[improved] = evaluation
        if evaluation == 2 or endpoint or evaluation % screen.config.trace_every == 0:
            trace.append(
                {
                    "evaluation": evaluation,
                    "candidate_kind": (
                        "perturbed_post_update_endpoint"
                        if endpoint
                        else "perturbed_pre_update"
                    ),
                    "post_update_endpoint": endpoint,
                    "feasibility_loss_by_restart": [
                        float(value) for value in detached.cpu()
                    ],
                }
            )

    updates = evaluations - 2
    for evaluation in range(2, evaluations):
        optimizer.zero_grad(set_to_none=True)
        energies, states = energy_fn(gamma, beta)
        inspect(energies, states, evaluation, False)
        energies.sum().backward()
        optimizer.step()
    with torch.no_grad():
        energies, states = energy_fn(gamma, beta)
        inspect(energies, states, evaluations, True)
    return (
        engine.OptimizationResult(
            label=STAGE2_METHOD,
            state=best_state,
            energy=best_energy,
            gamma=best_gamma,
            beta=best_beta,
            best_evaluation=best_evaluation,
            evaluations_per_restart=evaluations,
            updates_per_restart=updates,
            endpoint_evaluations_per_restart=1,
            trace=trace,
            elapsed_sec=time.time() - started,
        ),
        pi_state,
    )


def _stage2_result(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    result = common.load_json(_result_path(root, STAGE2_NAME))
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    common._require(
        result.get("schema") == f"{SCHEMA}-stage2-result"
        and result.get("status")
        == "COMPLETED_TARGET_BLIND_COHERENT_VARIATIONAL_F3"
        and result.get("manifest_sha256") == manifest.get("manifest_sha256")
        and result.get("result_sha256") == common.sha256_json(body)
        and result.get("optimization_performed") is True
        and result.get("forward_evaluations_per_restart") == STAGE2_EVALUATIONS
        and result.get("gradient_updates_per_restart")
        == STAGE2_EVALUATIONS - 2
        and result.get("exact_pi_candidate_evaluations_per_restart") == 1
        and result.get("perturbed_branch_evaluations_per_restart")
        == STAGE2_EVALUATIONS - 1
        and result.get("perturbation_stream") == STAGE2_PERTURBATION_STREAM
        and float(result.get("perturbation_magnitude_radians", math.nan))
        == STAGE2_PERTURBATION_MAGNITUDE
        and result.get("perturbation_sign_gamma_then_beta_by_restart")
        == _stage2_perturbation_signs()
        and result.get("target_constructed") is False
        and result.get("target_probability_computed") is False,
        "coherent-IM3 variational-F3 result is invalid",
    )
    return result


def _replay_stage2(
    screen: engine.StructuralScreen,
    c_state: torch.Tensor,
    result: dict[str, object],
) -> torch.Tensor:
    gamma = torch.as_tensor(
        result["gamma_by_restart"], dtype=screen.real_dtype, device=screen.device
    )
    beta = torch.as_tensor(
        result["beta_by_restart"], dtype=screen.real_dtype, device=screen.device
    )
    common._require(
        gamma.shape == (screen.config.restarts, P_F)
        and beta.shape == (screen.config.restarts, P_F),
        "saved coherent variational-F3 angle shape mismatch",
    )
    with torch.no_grad():
        _, state = _feasibility_energy(screen, c_state)(gamma, beta)
    common._require(
        common.state_sha256(state) == result.get("state_sha256"),
        "saved coherent variational-F3 replay mismatch",
    )
    return state


def run_stage2(root: Path, *, device: str) -> dict[str, object]:
    manifest, _, _, _, stage_a = _validated_manifest(root)
    destination = _result_path(root, STAGE2_NAME)
    if destination.exists():
        return _stage2_result(root, manifest)
    screen = common.make_screen(device=device, dtype="complex64", seeds=RESTART_SEEDS)
    c_state = common.replay_c12(screen, common.stage_a_c12_row(stage_a))
    result, fixed_state = _optimize_from_pi(
        _feasibility_energy(screen, c_state), screen, STAGE2_EVALUATIONS
    )
    fixed_metrics = screen.metrics(fixed_state)
    c_metrics = screen.metrics(c_state)
    analytic_pi_mass = common.grover_success(c_metrics["final_mass"], P_F)
    common._require(
        float(
            np.max(
                np.abs(
                    np.asarray(fixed_metrics["final_mass"])
                    - np.asarray(analytic_pi_mass)
                )
            )
        )
        <= 2e-5,
        "evaluated all-pi candidate no longer matches exact Grover amplification",
    )
    metrics = screen.metrics(result.state)
    selected_loss = [float(value) for value in result.energy.cpu()]
    common._require(np.all(np.isfinite(selected_loss)), "nonfinite F3 loss")
    initial_loss = [1.0 - float(value) for value in fixed_metrics["final_mass"]]
    first_loss = result.trace[0]["feasibility_loss_by_restart"]
    common._require(
        float(np.max(np.abs(np.asarray(initial_loss) - np.asarray(first_loss))))
        <= 2e-5,
        "evaluation 1 no longer matches the fixed F3 schedule",
    )
    signs = _stage2_perturbation_signs()
    selected_source = [
        "exact_pi" if int(value) == 1 else "perturbed_trajectory"
        for value in result.best_evaluation.cpu()
    ]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-stage2-result",
        "status": "COMPLETED_TARGET_BLIND_COHERENT_VARIATIONAL_F3",
        "manifest_sha256": manifest["manifest_sha256"],
        "method_id": STAGE2_METHOD,
        "restart_seeds": list(RESTART_SEEDS),
        "depth": P_F,
        "phase_hamiltonian": "I - exact joint-feasibility projector",
        "loss": "1 - exact joint-feasible probability",
        "initialization": (
            "evaluation 1 exact all-pi candidate; Adam branch starts at pi "
            "plus one deterministic target-independent signed 1e-3 perturbation"
        ),
        "exact_pi_candidate_gamma_by_restart": [
            [float(math.pi)] * P_F for _ in RESTART_SEEDS
        ],
        "exact_pi_candidate_beta_by_restart": [
            [float(math.pi)] * P_F for _ in RESTART_SEEDS
        ],
        "perturbation_stream": STAGE2_PERTURBATION_STREAM,
        "perturbation_magnitude_radians": STAGE2_PERTURBATION_MAGNITUDE,
        "perturbation_sign_gamma_then_beta_by_restart": signs,
        "perturbed_optimizer_start_gamma_by_restart": [
            [
                float(math.pi + STAGE2_PERTURBATION_MAGNITUDE * value)
                for value in row[:P_F]
            ]
            for row in signs
        ],
        "perturbed_optimizer_start_beta_by_restart": [
            [
                float(math.pi + STAGE2_PERTURBATION_MAGNITUDE * value)
                for value in row[P_F:]
            ]
            for row in signs
        ],
        "fixed_pi_initial_metrics": fixed_metrics,
        "fixed_pi_initial_feasibility_loss": initial_loss,
        "forward_evaluations_per_restart": STAGE2_EVALUATIONS,
        "exact_pi_candidate_evaluations_per_restart": 1,
        "perturbed_branch_evaluations_per_restart": STAGE2_EVALUATIONS - 1,
        "perturbed_branch_pre_update_evaluations_per_restart": (
            STAGE2_EVALUATIONS - 2
        ),
        "gradient_updates_per_restart": result.updates_per_restart,
        "endpoint_evaluations_per_restart": result.endpoint_evaluations_per_restart,
        "best_evaluation_by_restart": [
            int(value) for value in result.best_evaluation.cpu()
        ],
        "selected_source_by_restart": selected_source,
        "selected_feasibility_loss": selected_loss,
        "gamma_by_restart": result.gamma.cpu().tolist(),
        "beta_by_restart": result.beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(result.state),
        "trace": result.trace,
        "elapsed_sec": result.elapsed_sec,
        "optimization_performed": True,
        "exact_pi_candidate_retained_if_perturbation_worsens": True,
        "checkpoint_selected_by_stage_local_feasibility_loss_only": True,
        "coherent_objective_present_in_manifest_but_not_used_by_stage2": True,
        "target_constructed": False,
        "target_identity_available": False,
        "target_probability_computed": False,
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    return payload


def _objective_energy(
    screen: engine.StructuralScreen,
    reference: torch.Tensor,
    objective: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    cumulative = screen.common + objective

    def energy(gamma: torch.Tensor, beta: torch.Tensor):
        return objective_engine.history_circuit(
            screen, reference, (objective,), cumulative, gamma, beta, P_O
        )

    return energy


def _objective_result(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    result = common.load_json(_result_path(root, OBJECTIVE_NAME))
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    common._require(
        result.get("schema") == f"{SCHEMA}-objective-result"
        and result.get("status") == "COMPLETED_TARGET_BLIND_COHERENT_O16"
        and result.get("manifest_sha256") == manifest.get("manifest_sha256")
        and result.get("result_sha256") == common.sha256_json(body)
        and result.get("forward_evaluations_per_restart")
        == OBJECTIVE_EVALUATIONS
        and result.get("target_constructed") is False
        and result.get("target_probability_computed") is False,
        "coherent-IM3 variational-F3 O16 result is invalid",
    )
    return result


def run_objective(root: Path, *, device: str) -> dict[str, object]:
    manifest, _, _, discovery_manifest, stage_a = _validated_manifest(root)
    destination = _result_path(root, OBJECTIVE_NAME)
    if destination.exists():
        return _objective_result(root, manifest)
    stage2 = _stage2_result(root, manifest)
    screen = common.make_screen(device=device, dtype="complex64", seeds=RESTART_SEEDS)
    c_state = common.replay_c12(screen, common.stage_a_c12_row(stage_a))
    variational_f_state = _replay_stage2(screen, c_state, stage2)
    objective = discovery.objective_tensor(screen, manifest["coefficient_table"])
    paired_stream = int(
        manifest["stage3"]["paired_validation_initialization_stream"]
    )
    common._require(
        paired_stream
        == int(
            discovery_manifest["methods"][discovery.FULL][
                "initialization_stream"
            ]
        )
        == discovery.specs()[discovery.FULL].stream,
        "O16 pairing stream changed",
    )
    result = engine.optimize(
        _objective_energy(screen, variational_f_state, objective),
        P_O,
        P_O,
        OBJECTIVE_EVALUATIONS,
        FULL_METHOD,
        screen.config,
        screen.real_dtype,
        screen.device,
        paired_stream,
    )
    metrics = discovery._metrics(screen, result.state, objective)
    common._require(
        np.all(np.isfinite(metrics["expected_cumulative_loss"])),
        "nonfinite coherent-IM3 variational-F3 O16 outcome",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-objective-result",
        "status": "COMPLETED_TARGET_BLIND_COHERENT_O16",
        "manifest_sha256": manifest["manifest_sha256"],
        "stage2_result_sha256": stage2["result_sha256"],
        "method_id": FULL_METHOD,
        "restart_seeds": list(RESTART_SEEDS),
        "depth": P_O,
        "phase_names": ["coherent_IM3_O"],
        "phase_scale": manifest["phase_scale"],
        "reference_kind": "optimized_coherent_variational_F3",
        "paired_validation_initialization_stream": paired_stream,
        "resource": manifest["resource"],
        "forward_evaluations_per_restart": OBJECTIVE_EVALUATIONS,
        "gradient_updates_per_restart": result.updates_per_restart,
        "endpoint_evaluations_per_restart": result.endpoint_evaluations_per_restart,
        "best_evaluation_by_restart": [
            int(value) for value in result.best_evaluation.cpu()
        ],
        "selected_expected_loss": [float(value) for value in result.energy.cpu()],
        "gamma_by_restart": result.gamma.cpu().tolist(),
        "beta_by_restart": result.beta.cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(result.state),
        "trace": result.trace,
        "elapsed_sec": result.elapsed_sec,
        "checkpoint_selected_by_expected_loss_only": True,
        "target_constructed": False,
        "target_identity_available": False,
        "target_probability_computed": False,
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    return payload


def seal(root: Path) -> dict[str, object]:
    manifest, _, parent_seal, _, _ = _validated_manifest(root)
    stage2 = _stage2_result(root, manifest)
    objective = _objective_result(root, manifest)
    common._require(
        objective.get("stage2_result_sha256") == stage2.get("result_sha256"),
        "O16 was not built from the sealed variational-F3 state",
    )
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-seal",
        "status": (
            "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_VARIATIONAL_F3_FOLLOWUP"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_validation_optimizer_seal_sha256": parent_seal["seal_sha256"],
        "stage2_result_sha256": stage2["result_sha256"],
        "objective_result_sha256": objective["result_sha256"],
        "restart_count": 4,
        "stage_count": 3,
        "optimized_pipeline_stage_count": 3,
        "preoptimized_replayed_stage_count": 1,
        "optimized_within_followup_stage_count": 2,
        "target_constructed": False,
        "target_identity_available_to_optimizer_workers": False,
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
    command.add_argument("--parent-validation-root", type=Path, required=True)
    command = sub.add_parser("run-stage2")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("run-objective")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.parent_validation_root)
    elif args.command == "run-stage2":
        payload = run_stage2(args.campaign_root, device=args.device)
    elif args.command == "run-objective":
        payload = run_objective(args.campaign_root, device=args.device)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload["status"])


if __name__ == "__main__":
    main()
