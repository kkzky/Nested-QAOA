"""Immutable fixed-configuration validation specs and queue generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from appendix_v3 import adam_protocol, cell_cli, queue_controller, runner, selection
from appendix_v3.method_registry import parameter_layout_for_method

from .contract import CampaignContract, FixedConfiguration, canonical_json_bytes
from .stage1_import import certificate_row


VALIDATION_REGISTRY_SCHEMA = "bcst-unique-strengthened-validation-spec-registry-v1"


class SpecError(ValueError):
    """Raised when an immutable fixed spec cannot be reproduced exactly."""


@dataclass(frozen=True)
class FrozenSpec:
    manifest: runner.JobManifest
    path: Path
    sha256: str
    initializer_sha256: tuple[str, str, str]
    configuration_id: str


def publish_json(path: Path | str, value: Mapping[str, Any]) -> str:
    destination = Path(path).resolve()
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest().upper()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise SpecError(f"immutable artifact differs on resume: {destination}")
        return digest
    return runner.atomic_write_json(destination, value)


def resources_for(contract: CampaignContract, N: int) -> runner.LogicalResources:
    return runner.LogicalResources(
        **{key: int(value) for key, value in contract.resources[N].items()}
    )


def stage1_evidence(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    N: int,
) -> selection.Stage1RegenerationEvidence:
    binding = contract.stage1_imports[N]
    certified = certificate_row(certificate, N)
    if certified.get("status") != "compatible_imported_immutable_evidence":
        raise SpecError(f"Stage1 import N={N} is not certified")
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
        raise SpecError("optimized manifest lacks depth")
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
        domain, parameter_layout_for_method(manifest.method, manifest.depth)
    )
    hashes = tuple(adam_protocol.parameter_sha256(vector).upper() for vector in vectors)
    if len(hashes) != 3 or len(set(hashes)) != 3:
        raise SpecError("initializer KDF did not produce three distinct starts")
    return hashes  # type: ignore[return-value]


def validation_cell_id(
    configuration: FixedConfiguration, seed: int
) -> str:
    return f"{configuration.configuration_id}_seed{seed}_unique_validation"


def _manifest(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    configuration: FixedConfiguration,
    seed: int,
) -> runner.JobManifest:
    N = configuration.N
    method = configuration.method
    binding = contract.stage1_imports[N]
    kwargs: dict[str, Any] = {}
    if method in {"learned_projector", "same_stage1_state_direct", "stage1_only"}:
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
        cell_id=validation_cell_id(configuration, seed),
        cell_type=("stage1_only" if method == "stage1_only" else "method"),
        phase="external_validation",
        N=N,
        problem_seed=seed,
        method=method,
        depth=configuration.depth,
        adopted_budget=configuration.Adam_updates,
        resources=resources_for(contract, N),
        plan_sha256=contract.sha256,
        **kwargs,
    )


def _write_spec(
    manifest: runner.JobManifest,
    path: Path,
    *,
    configuration_id: str,
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
    return FrozenSpec(
        manifest=manifest,
        path=path.resolve(),
        sha256=digest,
        initializer_sha256=initializer_hashes(manifest),
        configuration_id=configuration_id,
    )


def materialize_validation_specs(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    spec_root: Path | str,
    registry_path: Path | str,
    *,
    device: str = "cuda",
) -> tuple[list[FrozenSpec], str]:
    root = Path(spec_root).resolve()
    specs: list[FrozenSpec] = []
    rows: list[dict[str, Any]] = []
    for configuration in sorted(
        contract.configurations,
        key=lambda row: (row.N, row.configuration_id),
    ):
        for seed in contract.validation_seeds[configuration.N]:
            cell_id = validation_cell_id(configuration, seed)
            manifest = _manifest(contract, certificate, configuration, seed)
            frozen = _write_spec(
                manifest,
                root / f"{cell_id}.json",
                configuration_id=configuration.configuration_id,
                device=device,
                activation_checkpointing=True,
            )
            specs.append(frozen)
            rows.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "categories": list(configuration.categories),
                    "N": configuration.N,
                    "problem_seed": seed,
                    "method": configuration.method,
                    "depth": configuration.depth,
                    "Adam_updates": configuration.Adam_updates,
                    "terminal_RU": configuration.terminal_RU,
                    "cell_id": cell_id,
                    "job_spec_path": str(frozen.path),
                    "job_spec_sha256": frozen.sha256,
                    "restart_initial_vector_sha256s": list(
                        frozen.initializer_sha256
                    ),
                }
            )
    optimized_count = sum(row["method"] != "stage1_only" for row in rows)
    stage1_count = len(rows) - optimized_count
    if len(rows) != 345 or optimized_count != 315 or stage1_count != 30:
        raise SpecError(
            "validation registry must contain 345 physical cells "
            "(315 optimized and 30 Stage1Only)"
        )
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise SpecError("validation registry contains duplicate cell IDs")
    digest = publish_json(
        registry_path,
        {
            "schema": VALIDATION_REGISTRY_SCHEMA,
            "campaign_id": contract.campaign_id,
            "plan_sha256": contract.sha256,
            "target_blind": True,
            "configuration_count": len(contract.configurations),
            "cell_count": len(rows),
            "optimized_cell_count": optimized_count,
            "stage1_only_cell_count": stage1_count,
            "physical_deduplication_key": [
                "N", "problem_seed", "method", "depth", "Adam_updates"
            ],
            "cells": rows,
            "status": "frozen_before_execution",
        },
    )
    return specs, digest


def write_queue(
    path: Path | str,
    queue_id: str,
    specs: Sequence[FrozenSpec],
) -> str:
    if not specs:
        raise SpecError("cannot publish an empty queue")
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


def build_stage1_only_smoke_spec(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
    *,
    device: str,
    path: Path | str,
) -> FrozenSpec:
    if device not in {"cpu", "cuda"}:
        raise SpecError("smoke device must be cpu or cuda")
    configuration = next(
        row
        for row in contract.configurations
        if row.N == 25 and row.method == "stage1_only"
    )
    seed = contract.validation_seeds[25][0]
    manifest = _manifest(contract, certificate, configuration, seed)
    return _write_spec(
        manifest,
        Path(path),
        configuration_id=configuration.configuration_id,
        device=device,
        activation_checkpointing=False,
    )


__all__ = [
    "FrozenSpec",
    "SpecError",
    "VALIDATION_REGISTRY_SCHEMA",
    "build_stage1_only_smoke_spec",
    "initializer_hashes",
    "materialize_validation_specs",
    "publish_json",
    "resources_for",
    "stage1_evidence",
    "validation_cell_id",
    "write_queue",
]
