"""Immutable job-spec and queue generation for the staged campaign."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from appendix_v3 import adam_protocol, cell_cli, queue_controller, runner, selection
from appendix_v3.method_registry import parameter_layout_for_method

from .contract import CampaignContract, OPTIMIZED_METHODS, canonical_json_bytes
from .stage1_import import certificate_row


SPEC_REGISTRY_SCHEMA = "bcst-unique-all-possible-tuning-spec-registry-v1"
VALIDATION_TEMPLATE_SCHEMA = "bcst-unique-validation-template-registry-v1"


class SpecError(ValueError):
    """Raised when an immutable spec cannot be reproduced exactly."""


@dataclass(frozen=True)
class FrozenSpec:
    manifest: runner.JobManifest
    path: Path
    sha256: str
    initializer_sha256: tuple[str, str, str]


def publish_json(path: Path | str, value: Mapping[str, Any]) -> str:
    """Publish once, or verify byte-for-byte identity on a resumed run."""

    destination = Path(path).resolve()
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest().upper()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise SpecError(f"immutable artifact differs on resume: {destination}")
        return digest
    return runner.atomic_write_json(destination, value)


def resources_for(contract: CampaignContract, N: int) -> runner.LogicalResources:
    row = contract.resources[N]
    return runner.LogicalResources(**{key: int(value) for key, value in row.items()})


def stage1_evidence(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    N: int,
) -> selection.Stage1RegenerationEvidence:
    binding = contract.stage1_imports[N]
    certified = certificate_row(certificate, N)
    if certified.get("status") != "compatible_imported_immutable_evidence":
        raise SpecError(f"Stage-1 import N={N} is not certified")
    return selection.Stage1RegenerationEvidence(
        N=N,
        phase="stage1_generation",
        adopted_budget=binding.adopted_budget,
        kdf_checkpoint_zero_sha256=binding.kdf_checkpoint_zero_sha256,
        selected_amplitude_sha256=binding.amplitude_sha256,
        plan_sha256=contract.sha256,
        record_sha256=binding.record_sha256,
    )


def initializer_hashes(manifest: runner.JobManifest) -> tuple[str, str, str]:
    if manifest.cell_type == "stage1_only":
        evidence = manifest.stage1_regeneration_evidence
        if evidence is None:
            raise SpecError("Stage1Only lacks source restart hashes")
        return tuple(value.upper() for value in evidence.kdf_checkpoint_zero_sha256)
    if manifest.depth is None:
        raise SpecError("optimized manifest lacks a depth")
    domain = adam_protocol.InitializerDomain(
        phase=manifest.phase,
        method=manifest.method,
        N=manifest.N,
        problem_seed=manifest.problem_seed,
        depth=manifest.depth,
        setting=manifest.setting,
        arm=manifest.initializer_arm,
    )
    vectors = adam_protocol.derive_three_initializers(
        domain,
        parameter_layout_for_method(manifest.method, manifest.depth),
    )
    hashes = tuple(
        adam_protocol.parameter_sha256(vector).upper() for vector in vectors
    )
    if len(hashes) != 3 or len(set(hashes)) != 3:
        raise SpecError("initializer KDF did not produce three distinct starts")
    return hashes  # type: ignore[return-value]


def _manifest(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    *,
    cell_id: str,
    phase: str,
    N: int,
    problem_seed: int,
    method: str,
    depth: int | None,
) -> runner.JobManifest:
    binding = contract.stage1_imports[N]
    needs_phi = method in {
        "learned_projector",
        "same_stage1_state_direct",
        "stage1_only",
    }
    kwargs: dict[str, Any] = {}
    if needs_phi:
        kwargs = {
            "stage1_state_path": str(binding.state_path.resolve()),
            "stage1_record_path": str(binding.record_path.resolve()),
            "expected_stage1_amplitude_sha256": binding.amplitude_sha256,
            "expected_stage1_record_sha256": binding.record_sha256,
            "stage1_regeneration_evidence": stage1_evidence(
                contract, certificate, N
            ),
        }
    return runner.JobManifest(
        cell_id=cell_id,
        cell_type="stage1_only" if method == "stage1_only" else "method",
        phase=phase,
        N=N,
        problem_seed=problem_seed,
        method=method,
        depth=depth,
        adopted_budget=(
            None if method == "stage1_only" else contract.budgets[(method, N)]
        ),
        resources=resources_for(contract, N),
        plan_sha256=contract.sha256,
        **kwargs,
    )


def _write_spec(
    manifest: runner.JobManifest,
    path: Path,
    *,
    device: str,
    activation_checkpointing: bool,
) -> FrozenSpec:
    payload = {
        "schema": cell_cli.CELL_JOB_SPEC_SCHEMA,
        "plan_sha256": runner.EXPECTED_PLAN_SHA256,
        "manifest": manifest.to_record(),
        "device": device,
        "activation_checkpointing": activation_checkpointing,
    }
    digest = publish_json(path, payload)
    return FrozenSpec(manifest, path.resolve(), digest, initializer_hashes(manifest))


def tuning_cell_id(method: str, N: int, seed: int, depth: int, budget: int) -> str:
    return f"{method}_N{N}_seed{seed}_p{depth}_b{budget}_unique"


def validation_cell_id(
    method: str, N: int, seed: int, depth: int | None, budget: int | None
) -> str:
    if method == "stage1_only":
        return f"stage1_only_N{N}_seed{seed}_unique"
    return f"{method}_N{N}_seed{seed}_p{depth}_b{budget}_unique_validation"


def freeze_all_tuning_specs(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    spec_root: Path | str,
    registry_path: Path | str,
    *,
    device: str = "cuda",
) -> tuple[Mapping[tuple[str, int, int, int], FrozenSpec], str]:
    root = Path(spec_root).resolve()
    specs: dict[tuple[str, int, int, int], FrozenSpec] = {}
    rows: list[dict[str, Any]] = []
    for N in (25, 30):
        for method in OPTIMIZED_METHODS:
            budget = contract.budgets[(method, N)]
            for seed in contract.tuning_seeds[N]:
                for depth in contract.maximum_tuning_depths[(method, N)]:
                    cell_id = tuning_cell_id(method, N, seed, depth, budget)
                    manifest = _manifest(
                        contract,
                        certificate,
                        cell_id=cell_id,
                        phase="depth_reselection",
                        N=N,
                        problem_seed=seed,
                        method=method,
                        depth=depth,
                    )
                    frozen = _write_spec(
                        manifest,
                        root / f"{cell_id}.json",
                        device=device,
                        activation_checkpointing=True,
                    )
                    key = (method, N, seed, depth)
                    specs[key] = frozen
                    rows.append(
                        {
                            "method": method,
                            "N": N,
                            "problem_seed": seed,
                            "depth": depth,
                            "Adam_updates": budget,
                            "cell_id": cell_id,
                            "job_spec_path": str(frozen.path),
                            "job_spec_sha256": frozen.sha256,
                            "restart_initial_vector_sha256s": list(
                                frozen.initializer_sha256
                            ),
                            "initializer_freeze_representation": (
                                "KDF namespace, exact inputs, generation rule, and "
                                "reproduced vector SHA-256"
                            ),
                        }
                    )
    if len(rows) != 244 or len(specs) != 244:
        raise SpecError(f"all-possible tuning registry must contain 244 cells, got {len(rows)}")
    record = {
        "schema": SPEC_REGISTRY_SCHEMA,
        "campaign_id": contract.campaign_id,
        "plan_sha256": contract.sha256,
        "generated_before_target_oracle_open": True,
        "cell_count": len(rows),
        "cells": rows,
    }
    registry_hash = publish_json(registry_path, record)
    return specs, registry_hash


def freeze_validation_template(
    contract: CampaignContract, path: Path | str
) -> str:
    rows = []
    for N in (25, 30):
        for method in OPTIMIZED_METHODS:
            rows.append(
                {
                    "N": N,
                    "method": method,
                    "problem_seeds": list(contract.validation_seeds[N]),
                    "depth_source": "frozen unique-primary tuning decision",
                    "Adam_updates": contract.budgets[(method, N)],
                    "phase": "external_validation",
                    "restart_count": 3,
                    "continuation": False,
                }
            )
        rows.append(
            {
                "N": N,
                "method": "stage1_only",
                "problem_seeds": list(contract.validation_seeds[N]),
                "depth_source": None,
                "Adam_updates": None,
                "phase": "external_validation",
                "restart_count": 0,
            }
        )
    return publish_json(
        path,
        {
            "schema": VALIDATION_TEMPLATE_SCHEMA,
            "campaign_id": contract.campaign_id,
            "plan_sha256": contract.sha256,
            "materialize_only_after_depth_selection": True,
            "rows": rows,
        },
    )


def write_queue(
    path: Path | str,
    queue_id: str,
    specs: Sequence[FrozenSpec],
) -> str:
    if not specs:
        raise SpecError("cannot publish an empty optimizer queue")
    if len({spec.manifest.cell_id for spec in specs}) != len(specs):
        raise SpecError("queue contains duplicate cell IDs")
    return publish_json(
        path,
        {
            "schema": queue_controller.QUEUE_SCHEMA,
            "plan_sha256": runner.EXPECTED_PLAN_SHA256,
            "queue_id": queue_id,
            "cells": [
                {
                    "cell_id": spec.manifest.cell_id,
                    "job_spec_path": str(spec.path),
                    "job_spec_sha256": spec.sha256,
                }
                for spec in specs
            ],
        },
    )


def select_tuning_specs(
    registry: Mapping[tuple[str, int, int, int], FrozenSpec],
    contract: CampaignContract,
    depths_by_stratum: Mapping[tuple[str, int], Iterable[int]],
) -> list[FrozenSpec]:
    selected: list[FrozenSpec] = []
    for N in (25, 30):
        for method in OPTIMIZED_METHODS:
            for depth in sorted(set(depths_by_stratum.get((method, N), ()))):
                for seed in contract.tuning_seeds[N]:
                    selected.append(registry[(method, N, seed, depth)])
    return selected


def materialize_validation_specs(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    selected_depths: Mapping[tuple[str, int], int],
    spec_root: Path | str,
    registry_path: Path | str,
    *,
    device: str = "cuda",
) -> tuple[list[FrozenSpec], str]:
    root = Path(spec_root).resolve()
    specs: list[FrozenSpec] = []
    rows: list[dict[str, Any]] = []
    for N in (25, 30):
        for seed in contract.validation_seeds[N]:
            for method in OPTIMIZED_METHODS + ("stage1_only",):
                depth = None if method == "stage1_only" else selected_depths[(method, N)]
                budget = None if method == "stage1_only" else contract.budgets[(method, N)]
                cell_id = validation_cell_id(method, N, seed, depth, budget)
                manifest = _manifest(
                    contract,
                    certificate,
                    cell_id=cell_id,
                    phase="external_validation",
                    N=N,
                    problem_seed=seed,
                    method=method,
                    depth=depth,
                )
                frozen = _write_spec(
                    manifest,
                    root / f"{cell_id}.json",
                    device=device,
                    activation_checkpointing=True,
                )
                specs.append(frozen)
                rows.append(
                    {
                        "N": N,
                        "problem_seed": seed,
                        "method": method,
                        "depth": depth,
                        "Adam_updates": budget,
                        "cell_id": cell_id,
                        "job_spec_path": str(frozen.path),
                        "job_spec_sha256": frozen.sha256,
                        "restart_initial_vector_sha256s": list(
                            frozen.initializer_sha256
                        ),
                    }
                )
    if len(specs) != 120:
        raise SpecError(f"validation registry must contain 120 cells, got {len(specs)}")
    digest = publish_json(
        registry_path,
        {
            "schema": "bcst-unique-heldout-validation-spec-registry-v1",
            "campaign_id": contract.campaign_id,
            "plan_sha256": contract.sha256,
            "selected_depths": {
                f"{method}.N{N}": depth
                for (method, N), depth in sorted(selected_depths.items())
            },
            "target_blind": True,
            "cell_count": len(rows),
            "cells": rows,
        },
    )
    return specs, digest


def build_stage1_only_smoke_spec(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    *,
    device: str,
    path: Path | str,
) -> FrozenSpec:
    """Build one real CLI smoke cell without launching any optimizer.

    The smoke uses the first frozen N=25 heldout instance and the nonoptimized
    Stage1Only readout.  It therefore exercises instance construction, strict
    legacy-phi import, the requested CPU/CUDA device, job-spec parsing, the
    cell entry point, state serialization, and COMPLETE verification without
    consuming a tuning target or adding an Adam run.
    """

    if device not in {"cpu", "cuda"}:
        raise SpecError("smoke device must be cpu or cuda")
    N = 25
    seed = contract.validation_seeds[N][0]
    cell_id = f"preflight_{device}_stage1_only_N{N}_seed{seed}"
    manifest = _manifest(
        contract,
        certificate,
        cell_id=cell_id,
        phase="external_validation",
        N=N,
        problem_seed=seed,
        method="stage1_only",
        depth=None,
    )
    return _write_spec(
        manifest,
        Path(path),
        device=device,
        activation_checkpointing=False,
    )


__all__ = [
    "FrozenSpec",
    "SpecError",
    "build_stage1_only_smoke_spec",
    "freeze_all_tuning_specs",
    "freeze_validation_template",
    "materialize_validation_specs",
    "publish_json",
    "select_tuning_specs",
    "stage1_evidence",
    "write_queue",
]
