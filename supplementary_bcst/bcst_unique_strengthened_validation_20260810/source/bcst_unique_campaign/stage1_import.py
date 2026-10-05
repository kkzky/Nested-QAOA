"""Strict compatibility audit for the immutable paper-campaign Stage-1 states."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bcst_v2.instance_core import build_training_instance

from .contract import (
    CampaignContract,
    ContractError,
    Stage1Import,
    canonical_json_bytes,
    sha256_file,
)


IMPORT_CERTIFICATE_SCHEMA = "bcst-unique-stage1-import-certificate-v1"
LEGACY_STAGE1_SCHEMA = "lp-qaoa-appendix-stage1-target-blind-v3"
LEGACY_COMPLETE_SCHEMA = "lp-qaoa-appendix-cell-complete-v3"


class Stage1ImportError(ContractError):
    """Raised when a legacy Stage-1 artifact fails any compatibility check."""


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1ImportError(f"cannot read {label} at {path}") from exc
    if not isinstance(value, dict):
        raise Stage1ImportError(f"{label} must contain a JSON object")
    if canonical_json_bytes(value) != raw:
        raise Stage1ImportError(f"{label} is not canonical immutable JSON")
    return value, raw


def _hash_matches(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected.upper():
        raise Stage1ImportError(
            f"{label} SHA-256 mismatch: expected {expected.upper()}, observed {observed}"
        )
    return observed


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1ImportError(f"{label} must be an object")
    return value


def _verify_decision(binding: Stage1Import) -> Mapping[str, Any]:
    _hash_matches(binding.decision_path, binding.decision_sha256, "Stage-1 decision")
    decision, _ = _read_json(binding.decision_path, "Stage-1 decision")
    adopted = _mapping(decision.get("decision"), "Stage-1 adopted decision")
    evidence = _mapping(decision.get("regeneration_evidence"), "legacy Stage-1 evidence")
    if (
        decision.get("N") != binding.N
        or str(decision.get("plan_sha256", "")).upper()
        != binding.legacy_plan_sha256
        or decision.get("source_manifest_sha256") != binding.source_manifest_sha256
        or adopted.get("method") != "stage1"
        or adopted.get("role") != "stage1"
        or adopted.get("finalized") is not True
        or adopted.get("adopted_budget") != binding.adopted_budget
        or adopted.get("stage1_regeneration_required") is not False
        or str(decision.get("adopted_stage1_record_sha256", "")).upper()
        != binding.record_sha256
        or str(decision.get("adopted_stage1_state_file_sha256", "")).upper()
        != binding.state_file_sha256
        or str(decision.get("adopted_stage1_amplitude_sha256", "")).upper()
        != binding.amplitude_sha256
    ):
        raise Stage1ImportError("legacy Stage-1 decision does not bind the declared import")
    declared_kdf = evidence.get("kdf_checkpoint_zero_sha256")
    if not isinstance(declared_kdf, list) or tuple(
        str(value).upper() for value in declared_kdf
    ) != binding.kdf_checkpoint_zero_sha256:
        raise Stage1ImportError("legacy Stage-1 decision has different KDF start hashes")
    return decision


def _verify_record_and_state(
    contract: CampaignContract, binding: Stage1Import
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _hash_matches(
        binding.source_manifest_path,
        binding.source_manifest_sha256,
        "legacy source manifest",
    )
    _hash_matches(binding.record_path, binding.record_sha256, "Stage-1 record")
    _hash_matches(binding.state_path, binding.state_file_sha256, "Stage-1 state")
    _hash_matches(binding.complete_path, binding.complete_sha256, "Stage-1 COMPLETE")
    record, _ = _read_json(binding.record_path, "Stage-1 record")
    complete, _ = _read_json(binding.complete_path, "Stage-1 COMPLETE")
    if binding.complete_path.resolve() != (binding.record_path.parent / "COMPLETE.json").resolve():
        raise Stage1ImportError("Stage-1 COMPLETE is not adjacent to its record")
    if (
        record.get("schema") != LEGACY_STAGE1_SCHEMA
        or record.get("cell_type") != "stage1"
        or record.get("phase") != "stage1_generation"
        or record.get("method") != "stage1"
        or record.get("N") != binding.N
        or record.get("problem_seed") != binding.source_problem_seed
        or record.get("depth") != 12
        or record.get("adopted_budget") != binding.adopted_budget
        or record.get("setting") != "d4_1of2_r6_1of16"
        or record.get("target_blind") is not True
        or str(record.get("plan_sha256", "")).upper()
        != binding.legacy_plan_sha256
    ):
        raise Stage1ImportError("legacy Stage-1 record identity is incompatible")
    if (
        complete.get("schema") != LEGACY_COMPLETE_SCHEMA
        or str(complete.get("plan_sha256", "")).upper()
        != binding.legacy_plan_sha256
        or str(complete.get("record_sha256", "")).upper()
        != binding.record_sha256
        or str(complete.get("state_file_sha256", "")).upper()
        != binding.state_file_sha256
        or str(complete.get("state_amplitude_sha256", "")).upper()
        != binding.amplitude_sha256
        or complete.get("written_last") is not True
    ):
        raise Stage1ImportError("legacy Stage-1 COMPLETE has incompatible bindings")
    selected_state = _mapping(record.get("selected_state"), "Stage-1 selected state")
    if (
        str(selected_state.get("npy_file_sha256", "")).upper()
        != binding.state_file_sha256
        or str(selected_state.get("amplitude_sha256", "")).upper()
        != binding.amplitude_sha256
    ):
        raise Stage1ImportError("legacy Stage-1 record has different state hashes")

    try:
        state = np.load(binding.state_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise Stage1ImportError("legacy Stage-1 state is not a valid NPY array") from exc
    if state.dtype != np.dtype("<c16") or state.ndim != 1:
        raise Stage1ImportError("legacy Stage-1 state must be one-dimensional <c16")
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise Stage1ImportError("legacy Stage-1 state contains nonfinite amplitudes")
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise Stage1ImportError("legacy Stage-1 state is not normalized")
    amplitude_hash = hashlib.sha256(
        np.ascontiguousarray(state, dtype="<c16").tobytes(order="C")
    ).hexdigest().upper()
    if amplitude_hash != binding.amplitude_sha256:
        raise Stage1ImportError("legacy Stage-1 amplitude bytes differ from the contract")

    instance = build_training_instance(binding.N, binding.source_problem_seed, 1, 2, 1, 16)
    if instance.targets is not None:
        raise Stage1ImportError("target-blind instance builder unexpectedly exposed targets")
    if state.shape != (instance.dimension,):
        raise Stage1ImportError("legacy Stage-1 state dimension is incompatible")
    expected_resources = dict(contract.resources[binding.N])
    record_resources = _mapping(record.get("resource_record"), "Stage-1 resource record")
    record_instance = _mapping(record.get("instance"), "Stage-1 instance record")
    if dict(record_instance.get("resources", {})) != expected_resources:
        raise Stage1ImportError("legacy Stage-1 instance resource constants differ")
    # A Stage-1 optimizer records P1 as its preparation resource, not a terminal O.
    if record_resources.get("terminal_RU") != expected_resources["P1"]:
        raise Stage1ImportError("legacy Stage-1 resource accounting differs from P1")
    observed_hashes = {str(k): str(v).upper() for k, v in instance.hashes.items()}
    recorded_hashes = {
        str(k): str(v).upper()
        for k, v in dict(record_instance.get("hashes", {})).items()
    }
    if recorded_hashes != observed_hashes:
        raise Stage1ImportError("legacy Stage-1 construction hashes do not replay")
    probabilities = np.square(np.abs(state), dtype=np.float64)
    conflict_energy = float(np.dot(probabilities, instance.H_C))
    selected_energy = float(record.get("selected_energy"))
    if not math.isfinite(selected_energy) or abs(conflict_energy - selected_energy) > 1e-9:
        raise Stage1ImportError("legacy Stage-1 conflict energy does not replay")
    feasible_mass = float(np.sum(probabilities[instance.feasible_indices]))
    return record, {
        "dtype": state.dtype.str,
        "dimension": int(state.size),
        "normalization": norm,
        "feasible_mass": feasible_mass,
        "conflict_energy": conflict_energy,
        "construction_hashes": observed_hashes,
        "resources": expected_resources,
    }


def audit_one(contract: CampaignContract, binding: Stage1Import) -> dict[str, Any]:
    decision = _verify_decision(binding)
    record, compatibility = _verify_record_and_state(contract, binding)
    return {
        "N": binding.N,
        "source_problem_seed": binding.source_problem_seed,
        "adopted_budget": binding.adopted_budget,
        "depth_p1": 12,
        "legacy_plan_sha256": binding.legacy_plan_sha256,
        "source_manifest_path": str(binding.source_manifest_path.resolve()),
        "source_manifest_sha256": binding.source_manifest_sha256,
        "decision_path": str(binding.decision_path.resolve()),
        "decision_sha256": binding.decision_sha256,
        "record_path": str(binding.record_path.resolve()),
        "record_sha256": binding.record_sha256,
        "state_path": str(binding.state_path.resolve()),
        "state_file_sha256": binding.state_file_sha256,
        "amplitude_sha256": binding.amplitude_sha256,
        "complete_path": str(binding.complete_path.resolve()),
        "complete_sha256": binding.complete_sha256,
        "kdf_checkpoint_zero_sha256": list(binding.kdf_checkpoint_zero_sha256),
        "compatibility": compatibility,
        "legacy_record_cell_id": record.get("cell_id"),
        "legacy_decision_schema": decision.get("schema"),
        "status": "compatible_imported_immutable_evidence",
    }


def build_import_certificate(contract: CampaignContract) -> dict[str, Any]:
    rows = [audit_one(contract, contract.stage1_imports[size]) for size in (25, 30)]
    if len({row["amplitude_sha256"] for row in rows}) != 2:
        raise Stage1ImportError("N=25 and N=30 Stage-1 states unexpectedly share a hash")
    return {
        "schema": IMPORT_CERTIFICATE_SCHEMA,
        "campaign_id": contract.campaign_id,
        "plan_sha256": contract.sha256,
        "reuse_only": True,
        "regeneration_permitted": False,
        "sizes": rows,
        "status": "passed",
    }


def load_import_certificate(
    path: Path | str,
    *,
    contract: CampaignContract,
    expected_sha256: str,
    revalidate_sources: bool = False,
) -> dict[str, Any]:
    certificate_path = Path(path).resolve()
    _hash_matches(certificate_path, expected_sha256, "Stage-1 import certificate")
    certificate, _ = _read_json(certificate_path, "Stage-1 import certificate")
    if (
        certificate.get("schema") != IMPORT_CERTIFICATE_SCHEMA
        or certificate.get("campaign_id") != contract.campaign_id
        or certificate.get("plan_sha256") != contract.sha256
        or certificate.get("reuse_only") is not True
        or certificate.get("regeneration_permitted") is not False
        or certificate.get("status") != "passed"
    ):
        raise Stage1ImportError("Stage-1 import certificate has the wrong identity")
    rows = certificate.get("sizes")
    if not isinstance(rows, list) or len(rows) != 2:
        raise Stage1ImportError("Stage-1 import certificate must bind exactly two sizes")
    for size in (25, 30):
        row = certificate_row(certificate, size)
        binding = contract.stage1_imports[size]
        expected_identity = {
            "source_problem_seed": binding.source_problem_seed,
            "adopted_budget": binding.adopted_budget,
            "legacy_plan_sha256": binding.legacy_plan_sha256,
            "source_manifest_sha256": binding.source_manifest_sha256,
            "decision_sha256": binding.decision_sha256,
            "record_sha256": binding.record_sha256,
            "state_file_sha256": binding.state_file_sha256,
            "amplitude_sha256": binding.amplitude_sha256,
            "complete_sha256": binding.complete_sha256,
            "kdf_checkpoint_zero_sha256": list(
                binding.kdf_checkpoint_zero_sha256
            ),
        }
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise Stage1ImportError(
                f"Stage-1 import certificate identity differs at N={size}"
            )
    if revalidate_sources:
        expected = build_import_certificate(contract)
        if certificate != expected:
            raise Stage1ImportError("Stage-1 source artifacts changed after certification")
    return certificate


def certificate_row(certificate: Mapping[str, Any], N: int) -> Mapping[str, Any]:
    rows = certificate.get("sizes")
    if not isinstance(rows, list):
        raise Stage1ImportError("Stage-1 import certificate lacks size records")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("N") == N]
    if len(matches) != 1:
        raise Stage1ImportError(f"Stage-1 import certificate does not uniquely bind N={N}")
    return matches[0]


__all__ = [
    "IMPORT_CERTIFICATE_SCHEMA",
    "Stage1ImportError",
    "audit_one",
    "build_import_certificate",
    "certificate_row",
    "load_import_certificate",
]
