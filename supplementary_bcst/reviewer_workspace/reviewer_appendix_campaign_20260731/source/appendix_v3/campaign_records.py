"""Strict record adapters for live reviewer-appendix campaign decisions.

This module performs no simulation and never reads target results.  It turns
completed target-blind optimizer records into the evidence objects consumed by
``selection`` while preserving the runner's immutable artifact boundaries.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import runner, selection
from .adam_protocol import (
    InitializerDomain,
    derive_three_initializers,
    parameter_sha256,
)
from .method_registry import canonical_method_id, parameter_layout_for_method


_TARGET_KEY_PATTERNS = (
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
_LOGICAL_CELL_ID = re.compile(r"[A-Za-z0-9_.-]+")


class CampaignRecordError(ValueError):
    """Raised when a live-campaign record is incomplete or inconsistent."""


def _plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise CampaignRecordError(f"{field} must be a plain integer")
    if value < minimum:
        raise CampaignRecordError(f"{field} must be at least {minimum}")
    return value


def _finite_real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignRecordError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignRecordError(f"{field} must be a finite real number")
    return result


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise CampaignRecordError(f"{field} must be a SHA-256 digest")
    digest = value.upper()
    if len(digest) != 64 or any(
        character not in "0123456789ABCDEF" for character in digest
    ):
        raise CampaignRecordError(f"{field} must be a SHA-256 digest")
    return digest


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignRecordError(f"{field} must be a mapping")
    return value


def _float_hex_vector(value: object, *, field: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or not value:
        raise CampaignRecordError(f"{field} must be a nonempty float-hex vector")
    parsed: list[float] = []
    for index, raw_component in enumerate(value):
        if not isinstance(raw_component, str):
            raise CampaignRecordError(
                f"{field}[{index}] must be a canonical float-hex string"
            )
        try:
            component = float.fromhex(raw_component)
        except ValueError as exc:
            raise CampaignRecordError(
                f"{field}[{index}] must be a canonical float-hex string"
            ) from exc
        if not math.isfinite(component) or component.hex() != raw_component:
            raise CampaignRecordError(
                f"{field}[{index}] must be a canonical finite float-hex string"
            )
        parsed.append(component)
    return np.asarray(parsed, dtype=np.float64)


def _bound_logical_cell_id(
    value: object,
    *,
    depth: int,
    initializer_sha256: Sequence[str],
) -> str:
    """Bind a caller label to the exact cross-budget optimizer identity."""

    if not isinstance(value, str) or _LOGICAL_CELL_ID.fullmatch(value) is None:
        raise CampaignRecordError(
            "logical_cell_id must contain only letters, digits, dot, underscore, and dash"
        )
    kdf_payload = runner.canonical_json_bytes(
        {
            "depth": depth,
            "initializer_sha256": list(initializer_sha256),
        }
    )
    kdf_identity = hashlib.sha256(kdf_payload).hexdigest().upper()
    return f"{value}@depth-{depth}@kdf-{kdf_identity}"


def _assert_target_blind(value: object, *, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized != "target_blind" and any(
                pattern in normalized for pattern in _TARGET_KEY_PATTERNS
            ):
                raise CampaignRecordError(
                    f"target-derived field is forbidden at {path}.{key}"
                )
            _assert_target_blind(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_target_blind(child, path=f"{path}[{index}]")


def _exact_method_identity(
    job_record: Mapping[str, Any],
) -> tuple[str, int, int, int, str]:
    if job_record.get("target_blind") is not True:
        raise CampaignRecordError("optimizer job record must be target blind")
    if job_record.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256:
        raise CampaignRecordError("optimizer job record has the wrong plan hash")
    if job_record.get("setting") != runner.FIXED_SETTING:
        raise CampaignRecordError("optimizer job record has the wrong setting")

    size = _plain_int(job_record.get("N"), field="job N", minimum=1)
    if size not in runner.ALLOWED_SIZES:
        raise CampaignRecordError("job N must be exactly 25 or 30")
    problem_seed = _plain_int(
        job_record.get("problem_seed"), field="job problem_seed"
    )
    raw_method = job_record.get("method")
    if not isinstance(raw_method, str) or not raw_method:
        raise CampaignRecordError("job method must be a nonempty canonical string")
    try:
        method = canonical_method_id(raw_method)
    except ValueError as exc:
        raise CampaignRecordError(f"unknown optimizer method {raw_method!r}") from exc
    if raw_method != method:
        raise CampaignRecordError("job method must use its exact canonical identifier")

    cell_type = job_record.get("cell_type")
    phase = job_record.get("phase")
    schema = job_record.get("schema")
    depth = _plain_int(job_record.get("depth"), field="job depth", minimum=1)
    if cell_type == "stage1":
        if (
            schema != runner.STAGE1_RECORD_SCHEMA
            or phase != "stage1_generation"
            or method != "stage1"
            or depth != runner.STAGE1_DEPTH
            or problem_seed != runner.STAGE1_SEEDS[size]
        ):
            raise CampaignRecordError("Stage-1 optimizer identity is inconsistent")
    elif cell_type == "method":
        if schema != runner.METHOD_RECORD_SCHEMA:
            raise CampaignRecordError("Stage-2 optimizer identity is inconsistent")
        if phase in {"depth_tuning", "budget_audit", "depth_reselection"}:
            allowed_methods = (
                frozenset({"native_grover_d"})
                if phase == "depth_tuning"
                else runner.EXTERNAL_METHODS
                if phase == "depth_reselection"
                else runner.OPTIMIZED_AUDIT_METHODS
            )
            valid_phase_identity = (
                problem_seed in runner.TUNING_SEEDS[size]
                and method in allowed_methods
            )
        elif phase == "external_validation":
            valid_phase_identity = (
                problem_seed in runner.EXTERNAL_VALIDATION_SEEDS[size]
                and method in runner.EXTERNAL_METHODS
            )
        elif phase == "causal_ablation":
            valid_phase_identity = (
                problem_seed in runner.CAUSAL_ABLATION_SEEDS[size]
                and method in runner.CAUSAL_METHODS
            )
        else:
            valid_phase_identity = False
        if not valid_phase_identity:
            raise CampaignRecordError("Stage-2 optimizer identity is inconsistent")
    else:
        raise CampaignRecordError("budget evidence requires an optimized job record")
    return method, size, problem_seed, depth, str(phase)


def _validated_restart(
    value: object,
    *,
    restart_index: int,
    budget: int,
) -> tuple[tuple[float, ...], int, float, str, str]:
    restart = _mapping(value, field=f"restart {restart_index}")
    if (
        _plain_int(
            restart.get("restart_index"),
            field=f"restart {restart_index} index",
        )
        != restart_index
        or _plain_int(
            restart.get("execution_order"),
            field=f"restart {restart_index} execution order",
        )
        != restart_index
        or restart.get("role") != "independent_random"
        or restart.get("continuation") is not False
        or _plain_int(
            restart.get("executed_updates"),
            field=f"restart {restart_index} executed updates",
        )
        != budget
    ):
        raise CampaignRecordError(
            f"restart {restart_index} identity or optimizer budget is inconsistent"
        )

    initializer_hash = _sha256(
        restart.get("initializer_sha256"),
        field=f"restart {restart_index} initializer hash",
    )
    initializer = _float_hex_vector(
        restart.get("initializer_float_hex"),
        field=f"restart {restart_index} initializer",
    )
    if parameter_sha256(initializer).upper() != initializer_hash:
        raise CampaignRecordError(
            f"restart {restart_index} initializer bytes disagree with its hash "
            "and the frozen KDF domain"
        )
    raw_trace = restart.get("trace")
    if not isinstance(raw_trace, (list, tuple)) or len(raw_trace) != budget + 1:
        raise CampaignRecordError(
            f"restart {restart_index} lacks complete checkpoint coverage 0..{budget}"
        )
    energies: list[float] = []
    checkpoint_zero_hash: str | None = None
    for expected_checkpoint, raw_point in enumerate(raw_trace):
        point = _mapping(
            raw_point,
            field=f"restart {restart_index} checkpoint {expected_checkpoint}",
        )
        checkpoint = _plain_int(
            point.get("checkpoint"),
            field=f"restart {restart_index} trace checkpoint",
        )
        if checkpoint != expected_checkpoint:
            raise CampaignRecordError(
                f"restart {restart_index} lacks complete checkpoint coverage 0..{budget}"
            )
        energies.append(
            _finite_real(
                point.get("energy"),
                field=f"restart {restart_index} checkpoint {checkpoint} energy",
            )
        )
        if expected_checkpoint == 0:
            checkpoint_zero_hash = _sha256(
                point.get("parameter_sha256"),
                field=f"restart {restart_index} checkpoint-zero parameter hash",
            )
            checkpoint_zero_parameters = _float_hex_vector(
                point.get("parameters_float_hex"),
                field=f"restart {restart_index} checkpoint-zero parameters",
            )
            if (
                parameter_sha256(checkpoint_zero_parameters).upper()
                != checkpoint_zero_hash
                or not np.array_equal(checkpoint_zero_parameters, initializer)
            ):
                raise CampaignRecordError(
                    f"restart {restart_index} checkpoint-zero bytes differ from the KDF initializer"
                )
    if checkpoint_zero_hash != initializer_hash:
        raise CampaignRecordError(
            f"restart {restart_index} KDF initializer differs from checkpoint zero"
        )

    selected_checkpoint = _plain_int(
        restart.get("selected_checkpoint"),
        field=f"restart {restart_index} selected checkpoint",
    )
    if selected_checkpoint > budget:
        raise CampaignRecordError(
            f"restart {restart_index} selected checkpoint exceeds its budget"
        )
    selected_energy = _finite_real(
        restart.get("selected_energy"),
        field=f"restart {restart_index} selected energy",
    )
    expected_checkpoint = min(
        range(budget + 1), key=lambda checkpoint: (energies[checkpoint], checkpoint)
    )
    if (
        selected_checkpoint != expected_checkpoint
        or selected_energy != energies[expected_checkpoint]
    ):
        raise CampaignRecordError(
            f"restart {restart_index} incumbent disagrees with its full trace"
        )
    restart_record_hash = _sha256(
        restart.get("restart_record_sha256"),
        field=f"restart {restart_index} record hash",
    )
    restart_without_hash = dict(restart)
    restart_without_hash.pop("restart_record_sha256", None)
    try:
        observed_restart_hash = hashlib.sha256(
            runner.canonical_json_bytes(restart_without_hash)
        ).hexdigest().upper()
    except (TypeError, ValueError, runner.RunnerContractError) as exc:
        raise CampaignRecordError(
            f"restart {restart_index} is not canonical JSON data"
        ) from exc
    if observed_restart_hash != restart_record_hash:
        raise CampaignRecordError(
            f"restart {restart_index} record hash does not bind its contents"
        )
    return (
        tuple(energies),
        selected_checkpoint,
        selected_energy,
        initializer_hash,
        restart_record_hash,
    )


def budget_record_from_target_blind_job(
    job_record: Mapping[str, Any],
    *,
    logical_cell_id: str,
    h_min: float,
    h_max: float,
) -> selection.BudgetAuditRecord:
    """Convert one complete optimizer trace into frozen budget-audit evidence."""

    if not isinstance(job_record, Mapping):
        raise CampaignRecordError("job_record must be a mapping")
    _assert_target_blind(job_record)
    method, size, problem_seed, depth, phase = _exact_method_identity(job_record)
    budget = _plain_int(
        job_record.get("adopted_budget"), field="job adopted budget", minimum=1
    )
    if budget not in runner.ALLOWED_BUDGETS:
        raise CampaignRecordError("job optimizer budget is not a frozen candidate")

    optimization = _mapping(job_record.get("optimization"), field="optimization")
    if (
        optimization.get("optimizer") != "Adam"
        or optimization.get("execution") != "strictly_sequential_0_1_2"
        or optimization.get("early_stopping") is not False
        or optimization.get("continuation") is not False
        or _plain_int(
            optimization.get("restart_count"), field="optimizer restart count"
        )
        != runner.RESTART_COUNT
        or _plain_int(
            optimization.get("executed_updates_per_restart"),
            field="optimizer executed updates per restart",
        )
        != budget
        or _plain_int(
            optimization.get("objective_evaluations"),
            field="optimizer objective evaluations",
        )
        != runner.RESTART_COUNT * (budget + 1)
    ):
        raise CampaignRecordError("optimizer identity or budget is inconsistent")

    raw_restarts = optimization.get("restarts")
    if not isinstance(raw_restarts, (list, tuple)) or len(raw_restarts) != 3:
        raise CampaignRecordError("optimizer must contain exactly three restarts")
    restart_data = tuple(
        _validated_restart(restart, restart_index=index, budget=budget)
        for index, restart in enumerate(raw_restarts)
    )
    if len({data[3] for data in restart_data}) != runner.RESTART_COUNT:
        raise CampaignRecordError("optimizer KDF restart hashes must be distinct")
    raw_restart_hashes = optimization.get("restart_hashes")
    if not isinstance(raw_restart_hashes, (list, tuple)):
        raise CampaignRecordError("optimizer restart hashes must be a sequence")
    declared_restart_hashes = tuple(
        _sha256(value, field=f"optimizer restart hash {index}")
        for index, value in enumerate(raw_restart_hashes)
    )
    if declared_restart_hashes != tuple(data[4] for data in restart_data):
        raise CampaignRecordError(
            "optimizer restart hashes do not bind the restart records"
        )
    try:
        domain = InitializerDomain(
            phase=phase,
            method=method,
            N=size,
            problem_seed=problem_seed,
            depth=depth,
            setting=runner.FIXED_SETTING,
            arm=runner.FIXED_INITIALIZER_ARM,
        )
        expected_initializers = derive_three_initializers(
            domain,
            parameter_layout_for_method(method, depth),
        )
        expected_initializer_hashes = tuple(
            parameter_sha256(initializer).upper()
            for initializer in expected_initializers
        )
    except ValueError as exc:
        raise CampaignRecordError(
            "optimizer KDF domain or parameter layout is invalid"
        ) from exc
    if tuple(data[3] for data in restart_data) != expected_initializer_hashes:
        raise CampaignRecordError(
            "optimizer restart hashes differ from the frozen KDF domain"
        )
    paired_cell_id = _bound_logical_cell_id(
        logical_cell_id,
        depth=depth,
        initializer_sha256=expected_initializer_hashes,
    )

    selected_restart = _plain_int(
        optimization.get("selected_restart"), field="globally selected restart"
    )
    expected_restart = min(
        range(runner.RESTART_COUNT),
        key=lambda index: (restart_data[index][2], restart_data[index][1], index),
    )
    if selected_restart != expected_restart:
        raise CampaignRecordError("globally selected restart disagrees with traces")
    _, selected_checkpoint, selected_energy, _, _ = restart_data[selected_restart]
    if (
        _plain_int(
            optimization.get("selected_checkpoint"),
            field="global selected checkpoint",
        )
        != selected_checkpoint
        or _finite_real(
            optimization.get("selected_energy"), field="global selected energy"
        )
        != selected_energy
        or _plain_int(
            job_record.get("selected_checkpoint"),
            field="job selected checkpoint",
        )
        != selected_checkpoint
    ):
        raise CampaignRecordError("global optimizer incumbent is inconsistent")

    replay_energy = _finite_real(
        job_record.get("selected_energy"), field="job selected replay energy"
    )
    if abs(replay_energy - selected_energy) > runner.ENERGY_REPLAY_TOLERANCE:
        raise CampaignRecordError("job selected replay energy disagrees with optimizer")

    energies = restart_data[selected_restart][0]
    tail_start_checkpoint = (4 * budget) // 5
    best_at_tail_start = min(energies[: tail_start_checkpoint + 1])
    best_at_end = min(energies)
    if best_at_end != selected_energy:
        raise CampaignRecordError("full-trace best-so-far end differs from selected energy")

    return selection.BudgetAuditRecord(
        cell_id=paired_cell_id,
        method=method,
        N=size,
        problem_seed=problem_seed,
        budget=budget,
        selected_energy=selected_energy,
        h_min=h_min,
        h_max=h_max,
        selected_checkpoint=selected_checkpoint,
        best_so_far_at_tail_start=best_at_tail_start,
        best_so_far_at_end=best_at_end,
    )


def validate_target_blind_record_against_manifest(
    job_record: Mapping[str, Any],
    manifest: runner.JobManifest,
    *,
    validate_optimizer_evidence: bool = True,
) -> None:
    """Validate serialized V3 record identity and optimizer evidence."""

    if not isinstance(job_record, Mapping):
        raise CampaignRecordError("job_record must be a mapping")
    if not isinstance(manifest, runner.JobManifest):
        raise CampaignRecordError("manifest must be a validated JobManifest")
    _assert_target_blind(job_record)
    expected_schema = {
        "stage1": runner.STAGE1_RECORD_SCHEMA,
        "method": runner.METHOD_RECORD_SCHEMA,
        "stage1_only": runner.STAGE1_ONLY_RECORD_SCHEMA,
    }[manifest.cell_type]
    if job_record.get("schema") != expected_schema:
        raise CampaignRecordError("target-blind record schema differs from its manifest")
    if (
        job_record.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256
        or job_record.get("target_blind") is not True
    ):
        raise CampaignRecordError("target-blind record has the wrong V3 identity")

    for field in (
        "cell_id",
        "cell_type",
        "phase",
        "N",
        "problem_seed",
        "setting",
        "method",
        "depth",
        "adopted_budget",
    ):
        expected = getattr(manifest, field)
        if field == "method":
            expected = canonical_method_id(expected)
        if job_record.get(field) != expected:
            raise CampaignRecordError(
                f"target-blind record {field} differs from its manifest"
            )

    if not validate_optimizer_evidence:
        return

    if manifest.cell_type in {"stage1", "method"}:
        # Bounds do not participate in identity/KDF validation.  The returned
        # selection record is intentionally discarded at this admission gate.
        budget_record_from_target_blind_job(
            job_record,
            logical_cell_id=manifest.cell_id,
            h_min=0.0,
            h_max=0.0,
        )
        return

    if (
        job_record.get("optimizer") != "not_applicable"
        or job_record.get("stage2_restart_count") != 0
    ):
        raise CampaignRecordError("Stage1Only optimizer identity is inconsistent")
    raw_hashes = job_record.get("restart_hashes")
    if not isinstance(raw_hashes, (list, tuple)) or len(raw_hashes) != runner.RESTART_COUNT:
        raise CampaignRecordError("Stage1Only lacks exactly three source restart hashes")
    restart_hashes = tuple(
        _sha256(value, field=f"Stage1Only source restart hash {index}")
        for index, value in enumerate(raw_hashes)
    )
    boundary = _mapping(
        job_record.get("stage1_boundary"), field="Stage1Only Stage-1 boundary"
    )
    boundary_hashes = boundary.get("source_restart_hashes")
    if not isinstance(boundary_hashes, (list, tuple)) or tuple(
        _sha256(value, field=f"Stage1Only boundary restart hash {index}")
        for index, value in enumerate(boundary_hashes)
    ) != restart_hashes:
        raise CampaignRecordError("Stage1Only source restart hashes are inconsistent")
    evidence = manifest.stage1_regeneration_evidence
    if evidence is None or boundary.get("regeneration_evidence") != evidence.to_record():
        raise CampaignRecordError("Stage1Only regeneration evidence differs from its manifest")
    selected_state = _mapping(
        job_record.get("selected_state"), field="Stage1Only selected state"
    )
    if (
        selected_state.get("amplitude_sha256", "").lower()
        != manifest.expected_stage1_amplitude_sha256.lower()
    ):
        raise CampaignRecordError("Stage1Only selected state differs from its manifest")


def stage1_decision_and_evidence(
    records_by_budget: Mapping[
        int, Sequence[selection.BudgetAuditRecord]
    ],
    *,
    stage1_record: Mapping[str, Any],
    stage1_record_sha256: str,
) -> tuple[selection.BudgetDecision, selection.Stage1RegenerationEvidence]:
    """Finalize the Stage-1 budget and bind it to the adopted Stage-1 job."""

    if not isinstance(stage1_record, Mapping):
        raise CampaignRecordError("stage1_record must be a mapping")
    _assert_target_blind(stage1_record)
    _exact_method_identity(stage1_record)
    expected_record_hash = _sha256(
        stage1_record_sha256, field="Stage-1 record hash"
    )
    try:
        canonical_record = runner.canonical_json_bytes(dict(stage1_record))
    except (TypeError, ValueError) as exc:
        raise CampaignRecordError("Stage-1 record is not canonical JSON data") from exc
    observed_record_hash = hashlib.sha256(canonical_record).hexdigest().upper()
    if observed_record_hash != expected_record_hash:
        raise CampaignRecordError(
            "Stage-1 record SHA-256 does not bind the supplied record"
        )
    decision = selection.decide_method_size_budget(
        records_by_budget,
        role="stage1",
    )
    evidence = selection.bind_stage1_regeneration_evidence(
        decision,
        stage1_record,
        record_sha256=observed_record_hash,
    )
    return decision, evidence


def _read_canonical_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignRecordError(f"{field} is not a readable JSON object") from exc
    if not isinstance(value, dict):
        raise CampaignRecordError(f"{field} must contain a JSON object")
    try:
        canonical_payload = runner.canonical_json_bytes(value)
    except (TypeError, ValueError, runner.RunnerContractError) as exc:
        raise CampaignRecordError(f"{field} is not canonical JSON data") from exc
    if payload != canonical_payload:
        raise CampaignRecordError(f"{field} is not canonical immutable JSON")
    return value, payload


def load_completed_target_blind_record(
    record_path: Path | str,
    *,
    expected_record_sha256: str,
) -> dict[str, Any]:
    """Load a record only after verifying its complete artifact hash chain."""

    expected_hash = _sha256(
        expected_record_sha256, field="expected target-blind record hash"
    )
    record_file = Path(record_path).resolve()
    if record_file.name != "target_blind_record.json":
        raise CampaignRecordError(
            "completed record path must name target_blind_record.json"
        )
    record, record_payload = _read_canonical_json(
        record_file, field="target-blind record"
    )
    observed_record_hash = hashlib.sha256(record_payload).hexdigest().upper()
    if observed_record_hash != expected_hash:
        raise CampaignRecordError("target-blind record SHA-256 mismatch")
    _assert_target_blind(record)
    expected_schema_by_cell_type = {
        "stage1": runner.STAGE1_RECORD_SCHEMA,
        "method": runner.METHOD_RECORD_SCHEMA,
        "stage1_only": runner.STAGE1_ONLY_RECORD_SCHEMA,
    }
    if (
        record.get("schema")
        != expected_schema_by_cell_type.get(record.get("cell_type"))
        or record.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256
        or record.get("target_blind") is not True
        or not isinstance(record.get("cell_id"), str)
        or not record.get("cell_id")
    ):
        raise CampaignRecordError("target-blind record identity is invalid")

    output_dir = record_file.parent
    complete_path = output_dir / "COMPLETE.json"
    complete, _ = _read_canonical_json(complete_path, field="COMPLETE marker")
    complete_keys = {
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
    if (
        set(complete) != complete_keys
        or complete.get("schema") != runner.COMPLETE_SCHEMA
        or complete.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256
        or complete.get("cell_id") != record.get("cell_id")
        or complete.get("written_last") is not True
        or Path(str(complete.get("record_path", ""))).resolve() != record_file
        or str(complete.get("record_sha256", "")).upper() != observed_record_hash
    ):
        raise CampaignRecordError("COMPLETE marker does not bind the record")

    manifest_path = (output_dir / "job_manifest.json").resolve()
    state_path = (output_dir / "selected_state.npy").resolve()
    if (
        Path(str(complete.get("manifest_path", ""))).resolve() != manifest_path
        or Path(str(complete.get("state_path", ""))).resolve() != state_path
    ):
        raise CampaignRecordError("COMPLETE marker has noncanonical artifact paths")
    manifest, manifest_payload = _read_canonical_json(
        manifest_path, field="job manifest"
    )
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest().upper()
    if manifest_hash != str(complete.get("manifest_sha256", "")).upper():
        raise CampaignRecordError("COMPLETE marker does not bind the job manifest")
    try:
        from .cell_cli import manifest_from_record

        parsed_manifest = manifest_from_record(manifest)
    except ValueError as exc:
        raise CampaignRecordError(f"job manifest is invalid: {exc}") from exc
    record_manifest = _mapping(record.get("manifest"), field="record manifest binding")
    if (
        Path(str(record_manifest.get("path", ""))).resolve() != manifest_path
        or str(record_manifest.get("sha256", "")).upper() != manifest_hash
    ):
        raise CampaignRecordError("target-blind record does not bind its manifest")
    for field in (
        "cell_id",
        "cell_type",
        "phase",
        "N",
        "problem_seed",
        "setting",
        "method",
        "depth",
        "adopted_budget",
    ):
        manifest_value = getattr(parsed_manifest, field)
        if field == "method":
            manifest_value = canonical_method_id(manifest_value)
        if record.get(field) != manifest_value:
            raise CampaignRecordError(
                f"target-blind record {field} disagrees with its manifest"
            )
    validate_target_blind_record_against_manifest(record, parsed_manifest)

    try:
        state_payload = state_path.read_bytes()
    except OSError as exc:
        raise CampaignRecordError("selected state is not readable") from exc
    state_file_hash = hashlib.sha256(state_payload).hexdigest().upper()
    selected_state = _mapping(
        record.get("selected_state"), field="selected-state binding"
    )
    if (
        Path(str(selected_state.get("path", ""))).resolve() != state_path
        or str(selected_state.get("npy_file_sha256", "")).upper()
        != state_file_hash
        or str(complete.get("state_file_sha256", "")).upper()
        != state_file_hash
    ):
        raise CampaignRecordError("record or COMPLETE marker does not bind the state")
    try:
        state = np.load(io.BytesIO(state_payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CampaignRecordError("selected state is not a valid NPY array") from exc
    if (
        state.dtype != np.dtype("<c16")
        or state.ndim != 1
        or not np.all(np.isfinite(state.real))
        or not np.all(np.isfinite(state.imag))
    ):
        raise CampaignRecordError("selected state has an invalid representation")
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > runner.STATE_NORM_TOLERANCE:
        raise CampaignRecordError("selected state is not normalized")
    amplitude_hash = runner.amplitude_sha256(state)
    if (
        selected_state.get("dtype") != "<c16"
        or _plain_int(
            selected_state.get("dimension"), field="selected-state dimension"
        )
        != state.size
        or abs(
            _finite_real(
                selected_state.get("probability_norm"),
                field="selected-state probability norm",
            )
            - norm
        )
        > runner.STATE_NORM_TOLERANCE
        or abs(
            _finite_real(
                selected_state.get("normalization_error"),
                field="selected-state normalization error",
            )
            - abs(norm - 1.0)
        )
        > runner.STATE_NORM_TOLERANCE
        or amplitude_hash != str(selected_state.get("amplitude_sha256", "")).upper()
        or amplitude_hash
        != str(complete.get("state_amplitude_sha256", "")).upper()
    ):
        raise CampaignRecordError("record or COMPLETE marker has the wrong amplitudes")
    return record


__all__ = [
    "CampaignRecordError",
    "budget_record_from_target_blind_job",
    "load_completed_target_blind_record",
    "stage1_decision_and_evidence",
    "validate_target_blind_record_against_manifest",
]
