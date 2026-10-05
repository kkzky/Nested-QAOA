"""Bind the copied V3 simulation engine to the new frozen plan at runtime.

Only campaign identity, seed domains, and the narrow imported-Stage-1 loader
are adapted.  Method dynamics, Adam, resource formulae, state replay, queue
integrity, and post-lock evidence checks remain the copied, tested V3 code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from appendix_v3 import adam_protocol, method_registry, provenance, runner, selection
from appendix_v3 import target_evaluator

from .contract import CampaignContract, load_contract, sha256_file
from .postlock import evaluate_unique_targets
from .stage1_import import (
    Stage1ImportError,
    certificate_row,
    load_import_certificate,
)


class RuntimeBindingError(ValueError):
    """Raised when the copied engine cannot be narrowly bound to this plan."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeBindingError(f"cannot read {label} at {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeBindingError(f"{label} must contain a JSON object")
    return value


def _strict_stage1_loader(
    contract: CampaignContract,
    certificate: Mapping[str, Any],
):
    """Return a loader that accepts only the two exact certified legacy states."""

    def load_stage1_artifact(
        state_path: Path | str,
        record_path: Path | str,
        *,
        expected_N: int,
        expected_amplitude_sha256: str,
        expected_record_sha256: str,
        expected_regeneration_evidence: selection.Stage1RegenerationEvidence,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        if expected_N not in (25, 30):
            raise Stage1ImportError("imported Stage-1 N must be 25 or 30")
        binding = contract.stage1_imports[expected_N]
        certified = certificate_row(certificate, expected_N)
        resolved_state = Path(state_path).resolve()
        resolved_record = Path(record_path).resolve()
        if (
            resolved_state != binding.state_path.resolve()
            or resolved_record != binding.record_path.resolve()
            or str(certified.get("state_path")) != str(resolved_state)
            or str(certified.get("record_path")) != str(resolved_record)
            or expected_amplitude_sha256.upper() != binding.amplitude_sha256
            or expected_record_sha256.upper() != binding.record_sha256
        ):
            raise Stage1ImportError("job attempts to substitute a different Stage-1 artifact")
        if not isinstance(
            expected_regeneration_evidence, selection.Stage1RegenerationEvidence
        ):
            raise Stage1ImportError("new-plan Stage-1 evidence is mandatory")
        evidence = expected_regeneration_evidence
        if (
            evidence.N != expected_N
            or evidence.adopted_budget != binding.adopted_budget
            or evidence.plan_sha256.upper() != contract.sha256
            or evidence.record_sha256.upper() != binding.record_sha256
            or evidence.selected_amplitude_sha256.upper() != binding.amplitude_sha256
            or tuple(value.upper() for value in evidence.kdf_checkpoint_zero_sha256)
            != binding.kdf_checkpoint_zero_sha256
        ):
            raise Stage1ImportError("job Stage-1 evidence differs from the import certificate")
        # Hash source bytes again in every worker; the expensive construction replay
        # was performed once when the import certificate was issued.
        if sha256_file(resolved_state) != binding.state_file_sha256:
            raise Stage1ImportError("imported Stage-1 state file changed")
        if sha256_file(resolved_record) != binding.record_sha256:
            raise Stage1ImportError("imported Stage-1 record changed")
        if sha256_file(binding.complete_path) != binding.complete_sha256:
            raise Stage1ImportError("imported Stage-1 COMPLETE changed")
        record = _load_json(resolved_record, "imported Stage-1 record")
        complete = _load_json(binding.complete_path, "imported Stage-1 COMPLETE")
        if (
            record.get("schema") != runner.STAGE1_RECORD_SCHEMA
            or str(record.get("plan_sha256", "")).upper()
            != binding.legacy_plan_sha256
            or record.get("N") != expected_N
            or record.get("problem_seed") != binding.source_problem_seed
            or record.get("depth") != 12
            or record.get("adopted_budget") != binding.adopted_budget
            or str(complete.get("plan_sha256", "")).upper()
            != binding.legacy_plan_sha256
            or str(complete.get("record_sha256", "")).upper()
            != binding.record_sha256
            or str(complete.get("state_file_sha256", "")).upper()
            != binding.state_file_sha256
        ):
            raise Stage1ImportError("imported Stage-1 legacy identity changed")
        state = np.load(resolved_state, allow_pickle=False)
        if state.dtype != np.dtype("<c16") or state.ndim != 1:
            raise Stage1ImportError("imported Stage-1 state dtype or rank changed")
        norm = float(np.vdot(state, state).real)
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-10:
            raise Stage1ImportError("imported Stage-1 state normalization changed")
        amplitude_hash = runner.amplitude_sha256(state)
        if amplitude_hash != binding.amplitude_sha256:
            raise Stage1ImportError("imported Stage-1 amplitudes changed")
        return np.array(state, dtype="<c16", order="C", copy=True), record, {
            "record_file_sha256": binding.record_sha256,
            "complete_file_sha256": binding.complete_sha256,
            "npy_file_sha256": binding.state_file_sha256,
            "amplitude_sha256": binding.amplitude_sha256,
            "import_certificate_sha256": sha256_file(
                Path(str(certificate["certificate_path"]))
            )
            if "certificate_path" in certificate
            else "",
        }

    return load_stage1_artifact


def activate(
    plan_path: Path | str,
    *,
    import_certificate_path: Path | str | None = None,
    import_certificate_sha256: str | None = None,
    revalidate_stage1_sources: bool = False,
) -> CampaignContract:
    """Apply the exact plan binding to every copied module used by a worker."""

    contract = load_contract(plan_path)
    plan_hash = contract.sha256
    tuning = {
        size: frozenset(contract.tuning_seeds[size]) for size in (25, 30)
    }
    validation = {
        size: frozenset(contract.validation_seeds[size]) for size in (25, 30)
    }
    stage1_seeds = dict(contract.stage1_seeds)

    runner.EXPECTED_PLAN_SHA256 = plan_hash
    runner.PLAN_SHA256 = plan_hash.lower()
    runner.PLAN_PATH = contract.path
    runner.DEFAULT_REPO_ROOT = contract.path.parent
    runner.TUNING_SEEDS = tuning
    runner.EXTERNAL_VALIDATION_SEEDS = validation
    runner.STAGE1_SEEDS = stage1_seeds

    selection.PLAN_SHA256 = plan_hash
    selection.TUNING_SEEDS = {
        size: tuple(contract.tuning_seeds[size]) for size in (25, 30)
    }
    selection.STAGE1_SEEDS = {
        size: (contract.stage1_seeds[size],) for size in (25, 30)
    }
    adam_protocol.PLAN_SHA256 = plan_hash.lower()
    provenance.PLAN_SHA256 = plan_hash.lower()
    provenance.APPENDIX_PLAN_SHA256 = plan_hash.lower()
    method_registry.EXPERIMENT_PLAN_SHA256 = plan_hash
    target_evaluator.PLAN_SHA256 = plan_hash.lower()

    def verify_plan_binding(*, plan_path=contract.path, repo_root=None):  # noqa: ANN001
        observed = sha256_file(Path(plan_path))
        if Path(plan_path).resolve() != contract.path or observed != plan_hash:
            raise RuntimeBindingError("worker plan path/hash differs from its frozen contract")
        current = load_contract(contract.path)
        if current.sha256 != plan_hash or current.campaign_id != contract.campaign_id:
            raise RuntimeBindingError("experiment plan changed during execution")
        return {
            "plan": dict(current.record),
            "provenance": {
                "binding": "new-plan runtime adapter over copied immutable V3 engine",
                "plan_sha256": plan_hash,
            },
        }

    runner.verify_plan_binding = verify_plan_binding
    # runner imported the provenance function by name; replace only that local alias.
    runner.assert_frozen_provenance = lambda _root: {
        "plan_sha256": plan_hash,
        "adapter": "bcst_unique_campaign.runtime",
    }

    if import_certificate_path is not None or import_certificate_sha256 is not None:
        if import_certificate_path is None or import_certificate_sha256 is None:
            raise RuntimeBindingError("Stage-1 certificate path and SHA-256 are both required")
        certificate = load_import_certificate(
            import_certificate_path,
            contract=contract,
            expected_sha256=import_certificate_sha256,
            revalidate_sources=revalidate_stage1_sources,
        )
        certificate = dict(certificate)
        certificate["certificate_path"] = str(Path(import_certificate_path).resolve())
        runner.load_stage1_artifact = _strict_stage1_loader(contract, certificate)

    # The copied evaluator still performs the complete lock evidence audit; this
    # replacement changes only post-lock scientific fields and enforces degeneracy=1.
    target_evaluator.evaluate_bcst_targets = evaluate_unique_targets
    return contract


__all__ = ["RuntimeBindingError", "activate"]
