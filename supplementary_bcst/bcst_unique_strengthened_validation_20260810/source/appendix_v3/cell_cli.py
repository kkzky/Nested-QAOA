"""Strict one-process entry point for one target-blind appendix cell.

The job-spec file is immutable input.  It contains a canonical
``JobManifest.to_record()`` payload plus the frozen execution settings.  The
output directory must not already exist: a failed invocation is a permanent
attempt and a controller must allocate a new directory before retrying.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import runner


CELL_JOB_SPEC_SCHEMA = "lp-qaoa-appendix-cell-job-spec-v1"
CELL_COMPLETION_SCHEMA = "lp-qaoa-appendix-cell-cli-completion-v1"

_JOB_SPEC_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "manifest",
        "device",
        "activation_checkpointing",
    }
)
_MANIFEST_DERIVED_KEYS = frozenset(
    {
        "schema",
        "optimizer",
        "early_stopping",
        "continuation",
        "initializer_rule",
    }
)
_RESOURCE_KEYS = frozenset({"B", "C", "X", "O", "S", "P1"})
_COMPLETE_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "cell_id",
        "manifest_path",
        "manifest_sha256",
        "record_path",
        "record_sha256",
        "state_path",
        "state_file_sha256",
        "state_amplitude_sha256",
        "written_last",
    }
)
_ALLOWED_RECORD_SCHEMAS = frozenset(
    {
        runner.STAGE1_RECORD_SCHEMA,
        runner.METHOD_RECORD_SCHEMA,
        runner.STAGE1_ONLY_RECORD_SCHEMA,
    }
)
_FORBIDDEN_TARGET_KEY_PATTERNS = (
    "best2",
    "best8",
    "ground",
    "target_identity",
    "target_identities",
    "target_index",
    "target_indices",
    "target_probability",
    "target_probabilities",
    "target_metrics",
    "target_hashes",
    "targets",
    "success_probability",
    "rts99",
)


class CellCLIError(ValueError):
    """Raised when a job spec or completed attempt violates the CLI contract."""


@dataclass(frozen=True)
class LoadedJobSpec:
    path: Path
    sha256: str
    manifest: runner.JobManifest
    device: str
    activation_checkpointing: bool
    record: Mapping[str, Any]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _load_json_object(path: Path | str, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CellCLIError(f"cannot read {label} JSON at {resolved}") from exc
    if not isinstance(payload, dict):
        raise CellCLIError(f"{label} must be a JSON object")
    return payload


def _assert_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    observed = frozenset(payload)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise CellCLIError(
            f"{label} fields differ from the frozen schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _assert_plain_int(value: Any, *, label: str, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if isinstance(value, bool) or type(value) is not int:
        raise CellCLIError(f"{label} must be a plain JSON integer")


def _assert_target_blind(value: Any, *, path: str = "job") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized != "target_blind" and any(
                pattern in normalized for pattern in _FORBIDDEN_TARGET_KEY_PATTERNS
            ):
                raise CellCLIError(f"target-bearing field is forbidden at {path}.{key}")
            _assert_target_blind(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_target_blind(nested, path=f"{path}[{index}]")


def manifest_from_record(payload: Mapping[str, Any]) -> runner.JobManifest:
    """Reconstruct and round-trip a strict canonical runner manifest record."""

    if not isinstance(payload, Mapping):
        raise CellCLIError("manifest must be a JSON object")
    init_names = frozenset(field.name for field in fields(runner.JobManifest))
    _assert_exact_keys(
        payload,
        init_names.union(_MANIFEST_DERIVED_KEYS),
        label="manifest",
    )
    if payload.get("schema") != runner.JOB_MANIFEST_SCHEMA:
        raise CellCLIError("manifest has the wrong schema")
    if payload.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256:
        raise CellCLIError("manifest has the wrong plan SHA-256")

    for name in (
        "cell_id",
        "cell_type",
        "phase",
        "method",
        "setting",
        "initializer_arm",
        "plan_sha256",
    ):
        if not isinstance(payload.get(name), str):
            raise CellCLIError(f"manifest.{name} must be a string")
    for name in ("N", "problem_seed", "restart_count"):
        _assert_plain_int(payload.get(name), label=f"manifest.{name}")
    for name in ("depth", "adopted_budget"):
        _assert_plain_int(
            payload.get(name), label=f"manifest.{name}", allow_none=True
        )
    if type(payload.get("target_blind")) is not bool:
        raise CellCLIError("manifest.target_blind must be a JSON boolean")
    for name in ("stage1_state_path", "stage1_record_path"):
        value = payload.get(name)
        if value is not None:
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise CellCLIError(f"manifest.{name} must be an absolute path or null")
    for name in (
        "expected_stage1_amplitude_sha256",
        "expected_stage1_record_sha256",
    ):
        value = payload.get(name)
        if value is not None and not _is_sha256(value):
            raise CellCLIError(f"manifest.{name} must be a SHA-256 or null")

    resources_payload = payload.get("resources")
    if not isinstance(resources_payload, Mapping):
        raise CellCLIError("manifest.resources must be a JSON object")
    _assert_exact_keys(resources_payload, _RESOURCE_KEYS, label="manifest.resources")
    for name in sorted(_RESOURCE_KEYS):
        _assert_plain_int(
            resources_payload.get(name), label=f"manifest.resources.{name}"
        )
    resources = runner.LogicalResources(
        **{name: resources_payload[name] for name in _RESOURCE_KEYS}
    )
    kwargs = {
        name: payload[name]
        for name in init_names
        if name not in {"resources", "stage1_regeneration_evidence"}
    }
    kwargs["resources"] = resources
    if "stage1_regeneration_evidence" in init_names:
        raw_evidence = payload.get("stage1_regeneration_evidence")
        if raw_evidence is None:
            kwargs["stage1_regeneration_evidence"] = None
        else:
            try:
                kwargs["stage1_regeneration_evidence"] = (
                    runner.Stage1RegenerationEvidence.from_record(raw_evidence)
                )
            except (TypeError, ValueError) as exc:
                raise CellCLIError(
                    f"invalid Stage-1 regeneration evidence: {exc}"
                ) from exc
    try:
        manifest = runner.JobManifest(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CellCLIError(f"invalid frozen runner manifest: {exc}") from exc
    if manifest.to_record() != dict(payload):
        raise CellCLIError("manifest is not the canonical JobManifest record")
    _assert_target_blind(payload, path="manifest")
    return manifest


def load_job_spec(
    path: Path | str,
    *,
    expected_sha256: str,
) -> LoadedJobSpec:
    """Load an immutable job spec and reconstruct its strict manifest."""

    resolved = Path(path).resolve()
    if not _is_sha256(expected_sha256):
        raise CellCLIError("expected job-spec SHA-256 is required")
    observed_hash = runner.sha256_file(resolved)
    if observed_hash != expected_sha256.upper():
        raise CellCLIError("job-spec SHA-256 mismatch")
    payload = _load_json_object(resolved, label="job spec")
    _assert_exact_keys(payload, _JOB_SPEC_KEYS, label="job spec")
    if payload.get("schema") != CELL_JOB_SPEC_SCHEMA:
        raise CellCLIError("job spec has the wrong schema")
    if payload.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256:
        raise CellCLIError("job spec has the wrong plan SHA-256")
    device = payload.get("device")
    if device not in {"cpu", "cuda"}:
        raise CellCLIError("job spec device must be exactly 'cpu' or 'cuda'")
    checkpointing = payload.get("activation_checkpointing")
    if type(checkpointing) is not bool:
        raise CellCLIError("activation_checkpointing must be a JSON boolean")
    _assert_target_blind(payload, path="job_spec")
    manifest = manifest_from_record(payload["manifest"])
    return LoadedJobSpec(
        path=resolved,
        sha256=observed_hash,
        manifest=manifest,
        device=device,
        activation_checkpointing=checkpointing,
        record=payload,
    )


def _load_state(path: Path) -> np.ndarray:
    try:
        with open(path, "rb") as handle:
            loaded = np.load(handle, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CellCLIError("cannot load completed selected state") from exc
    if loaded.dtype != np.dtype("<c16") or loaded.ndim != 1:
        raise CellCLIError("completed selected state must be one-dimensional <c16")
    state = np.ascontiguousarray(loaded)
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise CellCLIError("completed selected state contains nonfinite amplitudes")
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > runner.STATE_NORM_TOLERANCE:
        raise CellCLIError("completed selected state is not normalized")
    return state


def verify_complete_directory(
    output_dir: Path | str,
    *,
    expected_cell_id: str,
    expected_manifest_record: Mapping[str, Any],
    expected_job_spec_sha256: str,
) -> dict[str, Any]:
    """Independently verify all files bound by a cell's COMPLETE marker."""

    if not isinstance(expected_cell_id, str) or not expected_cell_id:
        raise CellCLIError("expected cell ID is required")
    if not _is_sha256(expected_job_spec_sha256):
        raise CellCLIError("expected job-spec SHA-256 is required")
    output = Path(output_dir).resolve()
    complete_path = output / "COMPLETE.json"
    complete = _load_json_object(complete_path, label="COMPLETE marker")
    if complete_path.read_bytes() != runner.canonical_json_bytes(complete):
        raise CellCLIError("COMPLETE marker is not the exact canonical immutable record")
    _assert_exact_keys(complete, _COMPLETE_KEYS, label="COMPLETE marker")
    if (
        complete.get("schema") != runner.COMPLETE_SCHEMA
        or complete.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256
        or complete.get("cell_id") != expected_cell_id
        or complete.get("written_last") is not True
    ):
        raise CellCLIError("COMPLETE marker identity fields are invalid")

    expected_paths = {
        "manifest_path": (output / "job_manifest.json").resolve(),
        "record_path": (output / "target_blind_record.json").resolve(),
        "state_path": (output / "selected_state.npy").resolve(),
    }
    for field_name, expected_path in expected_paths.items():
        raw = complete.get(field_name)
        if not isinstance(raw, str) or Path(raw).resolve() != expected_path:
            raise CellCLIError(f"COMPLETE marker has the wrong {field_name}")
        if not expected_path.is_file():
            raise CellCLIError(f"completed artifact is missing: {expected_path}")

    manifest_path = expected_paths["manifest_path"]
    record_path = expected_paths["record_path"]
    state_path = expected_paths["state_path"]
    manifest_hash = runner.sha256_file(manifest_path)
    record_hash = runner.sha256_file(record_path)
    state_hash = runner.sha256_file(state_path)
    for observed, field_name in (
        (manifest_hash, "manifest_sha256"),
        (record_hash, "record_sha256"),
        (state_hash, "state_file_sha256"),
    ):
        if observed != str(complete.get(field_name, "")).upper():
            raise CellCLIError(f"COMPLETE marker does not bind {field_name}")

    manifest_record = _load_json_object(manifest_path, label="job manifest")
    if manifest_path.read_bytes() != runner.canonical_json_bytes(manifest_record):
        raise CellCLIError("job manifest is not the exact canonical immutable record")
    manifest = manifest_from_record(manifest_record)
    if manifest.cell_id != expected_cell_id:
        raise CellCLIError("job manifest cell ID does not match its attempt")
    if manifest_record != dict(expected_manifest_record):
        raise CellCLIError("completed manifest differs from the immutable job spec")

    record = _load_json_object(record_path, label="target-blind record")
    if record_path.read_bytes() != runner.canonical_json_bytes(record):
        raise CellCLIError(
            "target-blind record is not the exact canonical immutable record"
        )
    _assert_target_blind(record, path="target_blind_record")
    if (
        record.get("schema") not in _ALLOWED_RECORD_SCHEMAS
        or record.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256
        or record.get("target_blind") is not True
        or record.get("cell_id") != expected_cell_id
    ):
        raise CellCLIError("target-blind record identity fields are invalid")
    record_manifest = record.get("manifest")
    if not isinstance(record_manifest, Mapping) or (
        Path(str(record_manifest.get("path", ""))).resolve() != manifest_path
        or str(record_manifest.get("sha256", "")).upper() != manifest_hash
    ):
        raise CellCLIError("target-blind record does not bind its manifest")
    try:
        from .campaign_records import validate_target_blind_record_against_manifest

        validate_target_blind_record_against_manifest(
            record,
            manifest,
            validate_optimizer_evidence=False,
        )
    except ValueError as exc:
        raise CellCLIError(
            f"target-blind record differs from its validated manifest: {exc}"
        ) from exc

    selected_state = record.get("selected_state")
    if not isinstance(selected_state, Mapping) or (
        Path(str(selected_state.get("path", ""))).resolve() != state_path
        or str(selected_state.get("npy_file_sha256", "")).upper() != state_hash
    ):
        raise CellCLIError("target-blind record does not bind its selected state")
    state = _load_state(state_path)
    amplitude_hash = runner.amplitude_sha256(state)
    if amplitude_hash != str(selected_state.get("amplitude_sha256", "")).upper():
        raise CellCLIError("target-blind record has the wrong amplitude hash")
    if amplitude_hash != str(complete.get("state_amplitude_sha256", "")).upper():
        raise CellCLIError("COMPLETE marker does not bind selected amplitudes")

    return {
        "schema": CELL_COMPLETION_SCHEMA,
        "plan_sha256": runner.EXPECTED_PLAN_SHA256,
        "cell_id": expected_cell_id,
        "job_spec_sha256": expected_job_spec_sha256.upper(),
        "complete_path": str(complete_path),
        "complete_sha256": runner.sha256_file(complete_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "record_path": str(record_path),
        "record_sha256": record_hash,
        "state_path": str(state_path),
        "state_file_sha256": state_hash,
        "state_amplitude_sha256": amplitude_hash,
        "restart_count": manifest.restart_count,
        "restart_execution": "inside_one_cell_process_sequentially",
    }


def execute_job_spec(
    job_spec_path: Path | str,
    output_dir: Path | str,
    *,
    expected_job_spec_sha256: str,
) -> dict[str, Any]:
    """Execute exactly one production runner call in a fresh output directory."""

    loaded = load_job_spec(
        job_spec_path,
        expected_sha256=expected_job_spec_sha256,
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise CellCLIError(f"refusing to reuse output directory {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    manifest = loaded.manifest
    if manifest.cell_type == "stage1":
        runner.run_stage1_cell(
            manifest,
            output,
            device=loaded.device,
            activation_checkpointing=loaded.activation_checkpointing,
        )
    elif manifest.cell_type == "method":
        runner.run_method_cell(
            manifest,
            output,
            device=loaded.device,
            activation_checkpointing=loaded.activation_checkpointing,
        )
    elif manifest.cell_type == "stage1_only":
        runner.run_stage1_only_cell(
            manifest,
            output,
            device=loaded.device,
        )
    else:  # JobManifest currently makes this unreachable; keep the boundary closed.
        raise CellCLIError(f"unsupported cell type {manifest.cell_type!r}")

    return verify_complete_directory(
        output,
        expected_cell_id=manifest.cell_id,
        expected_manifest_record=manifest.to_record(),
        expected_job_spec_sha256=loaded.sha256,
    )


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", required=True, type=Path)
    parser.add_argument("--expected-job-spec-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Any incidental library output is diagnostic output.  Stdout remains a
        # one-line machine interface containing only the completion record.
        with contextlib.redirect_stdout(sys.stderr):
            completion = execute_job_spec(
                args.job_spec,
                args.output_dir,
                expected_job_spec_sha256=args.expected_job_spec_sha256,
            )
    except Exception as exc:  # noqa: BLE001 - CLI must convert failures to status.
        error = {"error": type(exc).__name__, "message": str(exc)}
        sys.stderr.write(_compact_json(error) + "\n")
        return 1
    sys.stdout.write(_compact_json(completion) + "\n")
    return 0


__all__ = [
    "CELL_COMPLETION_SCHEMA",
    "CELL_JOB_SPEC_SCHEMA",
    "CellCLIError",
    "LoadedJobSpec",
    "execute_job_spec",
    "load_job_spec",
    "main",
    "manifest_from_record",
    "verify_complete_directory",
]


if __name__ == "__main__":
    raise SystemExit(main())
