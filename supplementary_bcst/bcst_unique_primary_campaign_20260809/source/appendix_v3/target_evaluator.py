"""Read-only BCST target evaluation for already locked optimizer states.

This module is intentionally separate from the optimizer runner.  Its callable
is passed to :func:`appendix_v3.runner.evaluate_targets_after_lock`, which first
verifies the immutable cohort lock and every selected-state hash.  The callable
then independently re-verifies that on-disk evidence chain before deriving a
target, so a caller-supplied assertion cannot stand in for a real cohort lock.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bcst_v2.instance_core import (
    build_training_instance,
    clip_probability,
    derive_targets,
    rts99,
    total_sampling_cost,
)

from .provenance import PLAN_SHA256


FROZEN_SETTING = "d4_1of2_r6_1of16"
SETTING_ARGUMENTS = (1, 2, 1, 16)
JOB_MANIFEST_SCHEMA = "lp-qaoa-appendix-job-manifest-v3"
COHORT_LOCK_SCHEMA = "lp-qaoa-appendix-target-blind-cohort-lock-v3"
COMPLETE_SCHEMA = "lp-qaoa-appendix-cell-complete-v3"


class TargetEvaluationError(ValueError):
    """Raised when a locked record cannot be evaluated exactly as frozen."""


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TargetEvaluationError(f"{field} must be a SHA-256 digest")
    digest = value.upper()
    if len(digest) != 64 or any(character not in "0123456789ABCDEF" for character in digest):
        raise TargetEvaluationError(f"{field} must be a SHA-256 digest")
    return digest


def _read_bytes(path_value: object, *, field: str) -> tuple[Path, bytes]:
    if not isinstance(path_value, (str, Path)):
        raise TargetEvaluationError(f"{field} must be an on-disk path")
    path = Path(path_value).resolve()
    try:
        payload = path.read_bytes()
    except (OSError, ValueError) as exc:
        raise TargetEvaluationError(f"{field} is not a readable on-disk file") from exc
    return path, payload


def _json_object(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetEvaluationError(f"{field} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TargetEvaluationError(f"{field} must contain a JSON object")
    return value


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _amplitude_sha256(state: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(state, dtype="<c16"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest().upper()


def _verify_lock_evidence(
    *,
    job_record: Mapping[str, Any],
    state: np.ndarray,
    cohort_lock_path: Path | str,
    expected_lock_sha256: str,
) -> None:
    """Verify the complete on-disk evidence chain before deriving targets."""

    expected_lock_hash = _sha256(
        expected_lock_sha256, field="expected cohort-lock SHA-256"
    )
    _, lock_payload = _read_bytes(cohort_lock_path, field="cohort lock")
    if _payload_sha256(lock_payload) != expected_lock_hash:
        raise TargetEvaluationError("cohort-lock SHA-256 mismatch")
    lock_record = _json_object(lock_payload, field="cohort lock")
    cells = lock_record.get("cells")
    if (
        lock_record.get("schema") != COHORT_LOCK_SCHEMA
        or str(lock_record.get("plan_sha256", "")).upper() != str(PLAN_SHA256).upper()
        or lock_record.get("target_blind") is not True
        or lock_record.get("atomic_immutable_lock") is not True
        or not isinstance(cells, list)
        or not cells
        or lock_record.get("cell_count") != len(cells)
    ):
        raise TargetEvaluationError("cohort lock has invalid integrity fields")

    cell_id = job_record.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise TargetEvaluationError("locked job record lacks a cell_id")
    if not all(isinstance(cell, Mapping) for cell in cells):
        raise TargetEvaluationError("cohort lock contains a malformed cell entry")
    matches = [cell for cell in cells if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise TargetEvaluationError("requested cell is not uniquely present in the cohort lock")
    locked = matches[0]

    record_path, record_payload = _read_bytes(
        locked.get("record_path"), field="locked target-blind record"
    )
    locked_record_hash = _sha256(
        locked.get("record_sha256"), field="locked record SHA-256"
    )
    if _payload_sha256(record_payload) != locked_record_hash:
        raise TargetEvaluationError("locked target-blind record SHA-256 mismatch")
    disk_record = _json_object(record_payload, field="locked target-blind record")
    if disk_record != dict(job_record):
        raise TargetEvaluationError("job record does not equal the locked on-disk record")

    for field in ("N", "problem_seed", "method", "adopted_budget", "depth"):
        if locked.get(field) != disk_record.get(field):
            raise TargetEvaluationError(f"cohort-lock cell {field} disagrees with its record")
    if locked.get("resource_record") != disk_record.get("resource_record"):
        raise TargetEvaluationError("cohort-lock resource record disagrees with its record")

    manifest_path, manifest_payload = _read_bytes(
        locked.get("manifest_path"), field="locked job manifest"
    )
    manifest_hash = _sha256(
        locked.get("manifest_sha256"), field="locked manifest SHA-256"
    )
    if _payload_sha256(manifest_payload) != manifest_hash:
        raise TargetEvaluationError("locked job-manifest SHA-256 mismatch")
    manifest = _json_object(manifest_payload, field="locked job manifest")
    if (
        manifest.get("schema") != JOB_MANIFEST_SCHEMA
        or str(manifest.get("plan_sha256", "")).upper()
        != str(PLAN_SHA256).upper()
        or manifest.get("target_blind") is not True
        or manifest.get("cell_id") != cell_id
    ):
        raise TargetEvaluationError("locked job manifest has invalid bindings")
    manifest_record = disk_record.get("manifest")
    if (
        not isinstance(manifest_record, Mapping)
        or str(manifest_record.get("sha256", "")).upper() != manifest_hash
        or Path(str(manifest_record.get("path", ""))).resolve() != manifest_path
    ):
        raise TargetEvaluationError("locked record does not bind its job manifest")

    complete_path, complete_payload = _read_bytes(
        locked.get("complete_path"), field="locked COMPLETE marker"
    )
    complete_hash = _sha256(
        locked.get("complete_sha256"), field="locked COMPLETE SHA-256"
    )
    if _payload_sha256(complete_payload) != complete_hash:
        raise TargetEvaluationError("locked COMPLETE marker SHA-256 mismatch")
    complete = _json_object(complete_payload, field="locked COMPLETE marker")
    if (
        complete.get("schema") != COMPLETE_SCHEMA
        or str(complete.get("plan_sha256", "")).upper() != str(PLAN_SHA256).upper()
        or complete.get("cell_id") != cell_id
        or str(complete.get("record_sha256", "")).upper() != locked_record_hash
        or str(complete.get("manifest_sha256", "")).upper() != manifest_hash
        or Path(str(complete.get("record_path", ""))).resolve() != record_path
        or Path(str(complete.get("manifest_path", ""))).resolve() != manifest_path
        or complete.get("written_last") is not True
        or complete_path != (record_path.parent / "COMPLETE.json").resolve()
    ):
        raise TargetEvaluationError("locked COMPLETE marker has invalid bindings")

    state_path, state_payload = _read_bytes(
        locked.get("selected_state_path"), field="locked selected state"
    )
    state_file_hash = _sha256(
        locked.get("selected_state_file_sha256"),
        field="locked selected-state file SHA-256",
    )
    if _payload_sha256(state_payload) != state_file_hash:
        raise TargetEvaluationError("locked selected-state file SHA-256 mismatch")
    selected_state_record = disk_record.get("selected_state")
    if (
        not isinstance(selected_state_record, Mapping)
        or str(selected_state_record.get("npy_file_sha256", "")).upper()
        != state_file_hash
        or Path(str(selected_state_record.get("path", ""))).resolve() != state_path
        or str(complete.get("state_file_sha256", "")).upper() != state_file_hash
        or Path(str(complete.get("state_path", ""))).resolve() != state_path
    ):
        raise TargetEvaluationError("locked record does not bind its selected state")
    try:
        disk_state = np.load(io.BytesIO(state_payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TargetEvaluationError("locked selected-state file is invalid") from exc
    if (
        disk_state.dtype != np.dtype("<c16")
        or disk_state.ndim != 1
        or not np.all(np.isfinite(disk_state.real))
        or not np.all(np.isfinite(disk_state.imag))
    ):
        raise TargetEvaluationError("locked selected-state array is invalid")
    disk_norm = float(np.vdot(disk_state, disk_state).real)
    if not math.isfinite(disk_norm) or abs(disk_norm - 1.0) > 1e-10:
        raise TargetEvaluationError("locked selected-state normalization is invalid")
    locked_amplitude_hash = _sha256(
        locked.get("selected_state_amplitude_sha256"),
        field="locked selected-state amplitude SHA-256",
    )
    if (
        _amplitude_sha256(disk_state) != locked_amplitude_hash
        or str(selected_state_record.get("amplitude_sha256", "")).upper()
        != locked_amplitude_hash
        or str(complete.get("state_amplitude_sha256", "")).upper()
        != locked_amplitude_hash
        or _amplitude_sha256(np.asarray(state)) != locked_amplitude_hash
    ):
        raise TargetEvaluationError("selected state does not match the locked amplitudes")


def _extended_integer(value: int | float) -> dict[str, str]:
    if isinstance(value, bool):
        raise TargetEvaluationError("boolean is not an extended integer")
    if isinstance(value, (int, np.integer)):
        return {"kind": "finite", "integer_decimal": str(int(value))}
    number = float(value)
    if math.isinf(number) and number > 0.0:
        return {"kind": "positive_infinity"}
    raise TargetEvaluationError("expected a nonnegative integer or positive infinity")


def _probability(probabilities: np.ndarray, indices: np.ndarray) -> float:
    raw = float(np.sum(probabilities[np.asarray(indices, dtype=np.int64)]))
    return clip_probability(raw)


def evaluate_bcst_targets(
    *,
    job_record: Mapping[str, Any],
    state: np.ndarray,
    cohort_lock_path: Path | str,
    expected_lock_sha256: str,
) -> Mapping[str, Any]:
    """Derive best-2, best-8, and ground metrics after the cohort lock."""

    if not isinstance(job_record, Mapping):
        raise TargetEvaluationError("job_record must be a mapping")
    _verify_lock_evidence(
        job_record=job_record,
        state=state,
        cohort_lock_path=cohort_lock_path,
        expected_lock_sha256=expected_lock_sha256,
    )
    try:
        N = int(job_record["N"])
        problem_seed = int(job_record["problem_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetEvaluationError("locked record lacks N/problem_seed") from exc
    if N not in (25, 30):
        raise TargetEvaluationError("N must be 25 or 30")
    if job_record.get("setting") != FROZEN_SETTING:
        raise TargetEvaluationError("locked record has the wrong BCST setting")

    instance = build_training_instance(
        N,
        problem_seed,
        *SETTING_ARGUMENTS,
    )
    if instance.targets is not None:
        raise TargetEvaluationError("training-instance builder exposed targets early")
    observed_hashes = {
        str(key): str(value).upper() for key, value in instance.hashes.items()
    }
    instance_record = job_record.get("instance")
    if not isinstance(instance_record, Mapping):
        raise TargetEvaluationError("locked record lacks its target-blind instance record")
    setting_record = instance_record.get("setting")
    if not isinstance(setting_record, Mapping) or dict(setting_record) != {
        "label": FROZEN_SETTING,
        "d4_num": 1,
        "d4_den": 2,
        "r6_num": 1,
        "r6_den": 16,
    }:
        raise TargetEvaluationError("locked instance setting is inconsistent")
    if (
        instance_record.get("N") != N
        or instance_record.get("problem_seed") != problem_seed
    ):
        raise TargetEvaluationError("locked instance identity is inconsistent")
    recorded_hashes = {
        str(key): str(value).upper()
        for key, value in dict(instance_record.get("hashes", {})).items()
    }
    if observed_hashes != recorded_hashes:
        raise TargetEvaluationError("rebuilt instance hashes differ from the locked record")

    amplitudes = np.asarray(state)
    if amplitudes.dtype != np.dtype("<c16") or amplitudes.shape != (instance.dimension,):
        raise TargetEvaluationError("selected state has the wrong dtype or dimension")
    if not np.all(np.isfinite(amplitudes.real)) or not np.all(
        np.isfinite(amplitudes.imag)
    ):
        raise TargetEvaluationError("selected state contains nonfinite amplitudes")
    probabilities = np.square(np.abs(amplitudes), dtype=np.float64)
    probability_norm = float(np.sum(probabilities))
    if not math.isfinite(probability_norm) or probability_norm <= 0.0:
        raise TargetEvaluationError("selected state has an invalid probability norm")
    if abs(probability_norm - 1.0) > 1e-10:
        raise TargetEvaluationError("selected state is not normalized")
    probabilities /= probability_norm

    resource_record = job_record.get("resource_record")
    if not isinstance(resource_record, Mapping):
        raise TargetEvaluationError("locked record lacks a resource record")
    terminal_ru = resource_record.get("terminal_RU")
    if isinstance(terminal_ru, bool) or not isinstance(terminal_ru, int) or terminal_ru <= 0:
        raise TargetEvaluationError("terminal_RU must be a positive integer")

    targets = derive_targets(instance)
    result: dict[str, Any] = {
        "evaluation_role": "post_lock_read_only",
        "primary_target": "best2",
        "secondary_targets": ["ground", "best8"],
        "target_hashes": dict(targets.hashes),
        "ground_degeneracy": int(targets.ground_degeneracy),
        "unique_ground_applicable": bool(targets.ground_degeneracy == 1),
        "terminal_RU": int(terminal_ru),
        "probability_norm": probability_norm,
        "targets": {},
    }
    for label, indices in (
        ("ground", targets.ground),
        ("best2", targets.best2),
        ("best8", targets.best8),
    ):
        probability = _probability(probabilities, indices)
        repetitions = rts99(probability)
        total_cost = total_sampling_cost(probability, terminal_ru)
        result["targets"][label] = {
            "configuration_count": int(indices.size),
            "probability": probability,
            "RTS99": _extended_integer(repetitions),
            "total_logical_cost": _extended_integer(total_cost),
        }
    return result


__all__ = [
    "FROZEN_SETTING",
    "SETTING_ARGUMENTS",
    "TargetEvaluationError",
    "evaluate_bcst_targets",
]
