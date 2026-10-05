"""Single-cell, target-blind runner for the reviewer appendix campaign.

This module deliberately is *not* a campaign controller.  It provides small,
restartable operations for:

* one fresh Stage-1 optimization cell;
* one optimized Stage-2 method cell;
* one Stage1Only record (with no Stage-2 optimizer);
* an immutable cohort lock over completed target-blind records; and
* target evaluation that is impossible through this API until that lock exists.

Scientific dynamics, method definitions, Adam, and provenance are owned by
their dedicated modules.  The interfaces consumed here are intentionally
small: ``AppendixDynamics``, the method-registry helpers, ``run_adam`` and its
dataclasses, and ``assert_frozen_provenance``.  No target identities or target
probabilities enter a job manifest, objective, optimizer call, state replay,
or target-blind record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from bcst_v2.dynamics import canonicalize_global_phase
from bcst_v2.instance_core import build_training_instance
from bcst_v2.torch_adapter import (
    configure_deterministic_device,
    to_reduced_problem,
)

from .adam_protocol import (
    AdamConfig,
    InitializerDomain,
    derive_three_initializers,
    run_adam,
)
from .dynamics import AppendixDynamics
from .method_registry import (
    NATIVE_INITIAL_METHODS,
    PHI_INITIAL_METHODS,
    canonical_method_id,
    get_method_spec,
    parameter_layout_for_method,
    terminal_ru,
)
from .provenance import PLAN_SHA256, assert_frozen_provenance
from .selection import (
    SelectionError,
    Stage1RegenerationEvidence,
    validate_stage1_record_against_evidence,
)


EXPECTED_PLAN_SHA256 = (
    "20476043ADF7233C5CD721E620EC4CD620CA58C5BECFD1A192BD0495C25BAF08"
)
PLAN_PATH = Path(__file__).resolve().parents[2] / "EXPERIMENT_PLAN_V3.json"
DEFAULT_REPO_ROOT = PLAN_PATH.parent.parent

JOB_MANIFEST_SCHEMA = "lp-qaoa-appendix-job-manifest-v3"
STAGE1_RECORD_SCHEMA = "lp-qaoa-appendix-stage1-target-blind-v3"
METHOD_RECORD_SCHEMA = "lp-qaoa-appendix-method-target-blind-v3"
STAGE1_ONLY_RECORD_SCHEMA = "lp-qaoa-appendix-stage1-only-target-blind-v3"
COHORT_LOCK_SCHEMA = "lp-qaoa-appendix-target-blind-cohort-lock-v3"
TARGET_RESULT_SCHEMA = "lp-qaoa-appendix-post-lock-target-results-v3"
COMPLETE_SCHEMA = "lp-qaoa-appendix-cell-complete-v3"

ALLOWED_SIZES = frozenset({25, 30})
ALLOWED_BUDGETS = frozenset({400, 800, 1600})
FIXED_SETTING = "d4_1of2_r6_1of16"
FIXED_INITIALIZER_ARM = "adam_lr_0x1.1eb851eb851ecp-5"
STAGE1_DEPTH = 12
RESTART_COUNT = 3
STATE_NORM_TOLERANCE = 1e-10
ENERGY_REPLAY_TOLERANCE = 1e-9

STAGE1_SEEDS = {25: 840025000, 30: 840030000}
TUNING_SEEDS = {
    25: frozenset({840025011, 840025012}),
    30: frozenset({840030011, 840030012}),
}
EXTERNAL_VALIDATION_SEEDS = {
    25: frozenset(range(840025102, 840025112)),
    30: frozenset(range(840030101, 840030111)),
}
CAUSAL_ABLATION_SEEDS = {
    25: frozenset(range(840025201, 840025204)),
    30: frozenset(range(840030201, 840030204)),
}
EXTERNAL_METHODS = frozenset(
    {
        "learned_projector",
        "ordinary_native_direct",
        "decoupled_native_direct",
        "same_stage1_state_direct",
        "native_grover_d",
    }
)
CAUSAL_METHODS = frozenset(
    {
        "learned_projector",
        "projector_hd",
        "known_projector_hd",
        "xy_hf",
        "same_stage1_state_direct",
    }
)
OPTIMIZED_AUDIT_METHODS = EXTERNAL_METHODS.union(CAUSAL_METHODS)

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
    "success_probability",
    "rts99",
)


class RunnerContractError(ValueError):
    """Raised when a cell departs from the prospectively frozen contract."""


class TargetEvaluator(Protocol):
    """Minimal post-lock target-evaluation interface.

    The callable receives a verified target-blind job record, its verified
    selected state, and the real cohort-lock path/hash so it can independently
    verify the evidence chain before deriving target identities.  It returns a
    JSON-safe mapping containing target hashes and metrics.
    """

    def __call__(
        self,
        *,
        job_record: Mapping[str, Any],
        state: np.ndarray,
        cohort_lock_path: Path | str,
        expected_lock_sha256: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LogicalResources:
    """Exact integer primitives consumed by ``method_registry.terminal_ru``."""

    B: int
    C: int
    X: int
    O: int
    S: int
    P1: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RunnerContractError(
                    f"logical resource {name} must be a nonnegative integer"
                )


@dataclass(frozen=True)
class JobManifest:
    """Frozen input contract for one independently runnable cell."""

    cell_id: str
    cell_type: str
    phase: str
    N: int
    problem_seed: int
    method: str
    depth: int | None
    adopted_budget: int | None
    resources: LogicalResources
    setting: str = FIXED_SETTING
    stage1_state_path: str | None = None
    stage1_record_path: str | None = None
    expected_stage1_amplitude_sha256: str | None = None
    expected_stage1_record_sha256: str | None = None
    stage1_regeneration_evidence: Stage1RegenerationEvidence | None = None
    initializer_arm: str = FIXED_INITIALIZER_ARM
    plan_sha256: str = EXPECTED_PLAN_SHA256
    restart_count: int = RESTART_COUNT
    target_blind: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", self.cell_id
        ):
            raise RunnerContractError(
                "cell_id may contain only letters, digits, dot, underscore, and dash"
            )
        if not isinstance(self.cell_type, str) or self.cell_type not in {
            "stage1",
            "method",
            "stage1_only",
        }:
            raise RunnerContractError(f"unsupported cell_type {self.cell_type!r}")
        if not isinstance(self.method, str) or not self.method:
            raise RunnerContractError("method must be a nonempty string")
        if isinstance(self.N, bool) or type(self.N) is not int:
            raise RunnerContractError("N must be a plain integer")
        if self.N not in ALLOWED_SIZES:
            raise RunnerContractError(f"N must be one of {sorted(ALLOWED_SIZES)}")
        if (
            isinstance(self.problem_seed, bool)
            or type(self.problem_seed) is not int
            or self.problem_seed < 0
        ):
            raise RunnerContractError("problem_seed must be a nonnegative plain integer")
        if not isinstance(self.phase, str) or not self.phase:
            raise RunnerContractError("phase must be a nonempty string")
        if self.setting != FIXED_SETTING:
            raise RunnerContractError(
                f"setting must be the frozen {FIXED_SETTING!r}"
            )
        if self.initializer_arm != FIXED_INITIALIZER_ARM:
            raise RunnerContractError(
                "initializer_arm must be the frozen Adam learning-rate hex label"
            )
        if (
            not isinstance(self.plan_sha256, str)
            or self.plan_sha256.upper() != EXPECTED_PLAN_SHA256
        ):
            raise RunnerContractError("job manifest is bound to the wrong plan hash")
        if not isinstance(self.resources, LogicalResources):
            raise RunnerContractError("resources must be a LogicalResources record")
        if isinstance(self.restart_count, bool) or type(self.restart_count) is not int:
            raise RunnerContractError("restart_count must be a plain integer")
        if self.restart_count != RESTART_COUNT:
            raise RunnerContractError("the campaign requires exactly three restarts")
        if not self.target_blind:
            raise RunnerContractError("job manifests must be target blind")

        canonical = canonical_method_id(self.method)
        self._validate_phase_seed_and_method(canonical)
        if self.cell_type == "stage1":
            if isinstance(self.depth, bool) or type(self.depth) is not int:
                raise RunnerContractError("Stage-1 depth must be a plain integer")
            if canonical != "stage1" or self.depth != STAGE1_DEPTH:
                raise RunnerContractError("Stage-1 cells require method=Stage1 and depth=12")
            if isinstance(self.adopted_budget, bool) or type(self.adopted_budget) is not int:
                raise RunnerContractError("Stage-1 budget must be a plain integer")
            if self.adopted_budget not in ALLOWED_BUDGETS:
                raise RunnerContractError("Stage-1 budget is not a frozen candidate")
            self._require_no_stage1_binding()
        elif self.cell_type == "stage1_only":
            if canonical != "stage1_only":
                raise RunnerContractError("Stage1Only cell has the wrong method")
            if self.depth is not None or self.adopted_budget is not None:
                raise RunnerContractError("Stage1Only has no Stage-2 depth or budget")
            self._require_stage1_binding()
        else:
            if canonical in {"stage1", "stage1_only"}:
                raise RunnerContractError("optimized method cell has an invalid method")
            if isinstance(self.depth, bool) or type(self.depth) is not int or self.depth < 1:
                raise RunnerContractError("optimized method depth must be a positive plain integer")
            if isinstance(self.adopted_budget, bool) or type(self.adopted_budget) is not int:
                raise RunnerContractError("method budget must be a plain integer")
            if self.adopted_budget not in ALLOWED_BUDGETS:
                raise RunnerContractError("method budget is not a frozen candidate")
            spec = get_method_spec(canonical)
            if spec.initial_state == "phi":
                self._require_stage1_binding()
            else:
                self._require_no_stage1_binding()

    def _validate_phase_seed_and_method(self, canonical: str) -> None:
        if self.phase == "stage1_generation":
            if self.cell_type != "stage1" or canonical != "stage1":
                raise RunnerContractError("stage1_generation is reserved for Stage1")
            if self.problem_seed != STAGE1_SEEDS[self.N]:
                raise RunnerContractError("Stage1 uses only its frozen per-N seed")
            return
        if self.cell_type == "stage1":
            raise RunnerContractError("Stage1 must use phase='stage1_generation'")
        if self.phase in {"depth_tuning", "budget_audit", "depth_reselection"}:
            if self.cell_type != "method" or self.problem_seed not in TUNING_SEEDS[self.N]:
                raise RunnerContractError(
                    f"{self.phase} requires an optimized method and a frozen tuning seed"
                )
            allowed = (
                frozenset({"native_grover_d"})
                if self.phase == "depth_tuning"
                else EXTERNAL_METHODS
                if self.phase == "depth_reselection"
                else OPTIMIZED_AUDIT_METHODS
            )
            if canonical not in allowed:
                raise RunnerContractError(
                    f"method {canonical} is not permitted in phase {self.phase}"
                )
            return
        if self.phase == "external_validation":
            if self.problem_seed not in EXTERNAL_VALIDATION_SEEDS[self.N]:
                raise RunnerContractError("external validation seed is outside its frozen cohort")
            if canonical not in EXTERNAL_METHODS.union({"stage1_only"}):
                raise RunnerContractError("method is not in the external-validation cohort")
            return
        if self.phase == "causal_ablation":
            if self.problem_seed not in CAUSAL_ABLATION_SEEDS[self.N]:
                raise RunnerContractError("causal-ablation seed is outside its frozen cohort")
            if canonical not in CAUSAL_METHODS.union({"stage1_only"}):
                raise RunnerContractError("method is not in the causal-ablation design")
            return
        raise RunnerContractError(f"unknown frozen phase {self.phase!r}")

    def _require_stage1_binding(self) -> None:
        if not (
            self.stage1_state_path
            and self.stage1_record_path
            and _is_sha256(self.expected_stage1_amplitude_sha256)
            and _is_sha256(self.expected_stage1_record_sha256)
            and isinstance(
                self.stage1_regeneration_evidence,
                Stage1RegenerationEvidence,
            )
        ):
            raise RunnerContractError(
                f"{self.method} requires Stage-1 paths, identity hashes, and bound regeneration evidence"
            )
        evidence = self.stage1_regeneration_evidence
        if evidence.N != self.N:
            raise RunnerContractError("Stage-1 evidence N differs from the job manifest")
        if (
            evidence.selected_amplitude_sha256
            != self.expected_stage1_amplitude_sha256.lower()
        ):
            raise RunnerContractError(
                "Stage-1 evidence amplitude hash differs from the job manifest"
            )
        if evidence.record_sha256 != self.expected_stage1_record_sha256.lower():
            raise RunnerContractError(
                "Stage-1 evidence record hash differs from the job manifest"
            )
        if evidence.plan_sha256 != self.plan_sha256.lower():
            raise RunnerContractError(
                "Stage-1 evidence plan hash differs from the job manifest"
            )

    def _require_no_stage1_binding(self) -> None:
        if any(
            value is not None
            for value in (
                self.stage1_state_path,
                self.stage1_record_path,
                self.expected_stage1_amplitude_sha256,
                self.expected_stage1_record_sha256,
                self.stage1_regeneration_evidence,
            )
        ):
            raise RunnerContractError(
                f"{self.method} prohibits a Stage-1 artifact binding"
            )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["stage1_regeneration_evidence"] = (
            None
            if self.stage1_regeneration_evidence is None
            else self.stage1_regeneration_evidence.to_record()
        )
        record.update(
            {
                "schema": JOB_MANIFEST_SCHEMA,
                "method": canonical_method_id(self.method),
                "resources": asdict(self.resources),
                "optimizer": (
                    "not_applicable" if self.cell_type == "stage1_only" else "Adam"
                ),
                "early_stopping": False,
                "continuation": False,
                "initializer_rule": "three independent SHA-256 KDF starts",
            }
        )
        _assert_target_blind(record)
        return record


@dataclass(frozen=True)
class CellArtifacts:
    manifest_path: Path
    manifest_sha256: str
    record_path: Path
    record_sha256: str
    state_path: Path
    state_file_sha256: str
    amplitude_sha256: str
    complete_path: Path
    complete_sha256: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class LockArtifacts:
    lock_path: Path
    lock_sha256: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _RuntimeBundle:
    """Private, verified target-blind runtime used by production entry points."""

    dynamics: AppendixDynamics
    instance_record: Mapping[str, Any]
    device_record: Mapping[str, Any]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise RunnerContractError("scientific records cannot contain NaN or infinity")
        return number
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            flat = np.asarray(value, dtype=np.complex128).reshape(-1)
            return [[float(item.real), float(item.imag)] for item in flat]
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise RunnerContractError(f"{type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(temp_path: Path, destination: Path) -> None:
    """Publish a complete temporary file without an overwrite race."""

    if destination.exists():
        raise RunnerContractError(f"refusing to overwrite immutable artifact {destination}")
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise RunnerContractError(
            f"refusing to overwrite immutable artifact {destination}"
        ) from exc
    except OSError as exc:
        raise RunnerContractError(
            "atomic immutable publication requires a same-filesystem hard link"
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()
    _fsync_directory(destination.parent)


def atomic_write_json(path: Path | str, value: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    with tempfile.NamedTemporaryFile(
        "wb",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_publish(temp_path, destination)
    return hashlib.sha256(payload).hexdigest().upper()


def atomic_write_npy(path: Path | str, state: np.ndarray) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        np.save(handle, np.asarray(state, dtype="<c16"), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256_file(temp_path)
    _atomic_publish(temp_path, destination)
    return digest


def verify_plan_binding(
    *,
    plan_path: Path | str = PLAN_PATH,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    observed = sha256_file(plan_path)
    if observed != EXPECTED_PLAN_SHA256:
        raise RunnerContractError(
            f"plan SHA-256 mismatch: expected {EXPECTED_PLAN_SHA256}, observed {observed}"
        )
    if str(PLAN_SHA256).upper() != EXPECTED_PLAN_SHA256:
        raise RunnerContractError("runner and provenance module disagree on plan SHA-256")
    with open(plan_path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("schema") != "lp-qaoa-reviewer-appendix-layered-plan-v3":
        raise RunnerContractError("unrecognized experiment-plan schema")
    effective_seeds = plan.get("effective_seed_domains", {})
    expected_tuning = {
        "depth_and_budget_tuning_N25": sorted(TUNING_SEEDS[25]),
        "depth_and_budget_tuning_N30": sorted(TUNING_SEEDS[30]),
    }
    if any(effective_seeds.get(key) != value for key, value in expected_tuning.items()):
        raise RunnerContractError("V3 plan and runner disagree on fresh tuning seeds")
    gate = plan.get("optimizer_gate", {})
    if (
        gate.get("terminal_normalized_drop_threshold") != 0.01
        or gate.get("budget_normalized_mean_improvement_threshold") != 0.02
        or gate.get("warning_affects_decision") is not False
    ):
        raise RunnerContractError("V3 optimizer-gate contract is inconsistent")
    fresh = plan.get("fresh_execution_requirement", {})
    if (
        fresh.get("reuse_any_v2_optimizer_record") is not False
        or fresh.get("reuse_v2_native_target_file") is not False
    ):
        raise RunnerContractError("V3 plan must fail closed against V2 evidence")
    provenance = assert_frozen_provenance(Path(repo_root))
    return {"plan": plan, "provenance": provenance}


def _assert_target_blind(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized != "target_blind" and any(
                token in normalized for token in _TARGET_KEY_PATTERNS
            ):
                raise RunnerContractError(
                    f"target-derived key {key!r} is forbidden before cohort lock at {path}"
                )
            _assert_target_blind(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_target_blind(child, f"{path}[{index}]")


def _as_float(value: Any, *, label: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RunnerContractError(f"{label} must be scalar")
        number = float(value.detach().cpu())
    else:
        number = float(value)
    if not math.isfinite(number):
        raise RunnerContractError(f"{label} is nonfinite")
    return number


def _as_numpy_parameters(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8").reshape(-1))
    if not np.all(np.isfinite(array)):
        raise RunnerContractError("optimizer parameters are nonfinite")
    return array


def _parameter_sha256(value: Any) -> str:
    array = _as_numpy_parameters(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest().upper()


def _parameter_hex(value: Any) -> list[str]:
    return [float(item).hex() for item in _as_numpy_parameters(value)]


def _canonical_state(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    state = np.ascontiguousarray(np.asarray(value, dtype=np.complex128).reshape(-1))
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise RunnerContractError("state contains nonfinite amplitudes")
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or norm <= 0.0:
        raise RunnerContractError("state norm is invalid")
    state = np.ascontiguousarray(canonicalize_global_phase(state))
    normalization_error = abs(float(np.vdot(state, state).real) - 1.0)
    if normalization_error > STATE_NORM_TOLERANCE:
        raise RunnerContractError("canonical state normalization failed")
    state.setflags(write=False)
    return state


def amplitude_sha256(state: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(state, dtype="<c16"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest().upper()


def _load_serialized_state(path: Path | str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype != np.dtype("<c16") or loaded.ndim != 1:
        raise RunnerContractError("serialized state must be one-dimensional <c16")
    state = np.ascontiguousarray(loaded)
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise RunnerContractError("serialized state contains nonfinite amplitudes")
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > STATE_NORM_TOLERANCE:
        raise RunnerContractError("serialized state normalization check failed")
    return state


def _load_json(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RunnerContractError(f"{path} does not contain a JSON object")
    return value


def _verify_complete_marker(
    record_path: Path | str,
    *,
    expected_record_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    record = Path(record_path).resolve()
    marker_path = record.parent / "COMPLETE.json"
    if not marker_path.is_file():
        raise RunnerContractError(f"cell is incomplete; missing {marker_path}")
    marker = _load_json(marker_path)
    if marker.get("schema") != COMPLETE_SCHEMA:
        raise RunnerContractError("cell COMPLETE marker has the wrong schema")
    if marker.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise RunnerContractError("cell COMPLETE marker has the wrong plan hash")
    observed_record_hash = sha256_file(record)
    if expected_record_sha256 is not None and observed_record_hash != expected_record_sha256.upper():
        raise RunnerContractError("cell record hash violates the manifest boundary")
    if observed_record_hash != str(marker.get("record_sha256", "")).upper():
        raise RunnerContractError("cell COMPLETE marker does not bind its record")
    return marker, sha256_file(marker_path)


def write_job_manifest(
    manifest: JobManifest,
    output_dir: Path | str,
    *,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> tuple[Path, str]:
    verify_plan_binding(repo_root=repo_root)
    output = Path(output_dir)
    path = output / "job_manifest.json"
    digest = atomic_write_json(path, manifest.to_record())
    return path.resolve(), digest


def _build_target_blind_runtime(
    manifest: JobManifest,
    *,
    device: str,
) -> _RuntimeBundle:
    """Build the only production runtime from the frozen target-blind instance."""

    instance = build_training_instance(
        manifest.N,
        manifest.problem_seed,
        1,
        2,
        1,
        16,
    )
    if instance.targets is not None:
        raise RunnerContractError("production optimization received target identities")
    target_hash_keys = sorted(
        key for key in instance.hashes if "target" in str(key).lower()
    )
    if target_hash_keys:
        raise RunnerContractError(
            f"production optimization received target hashes: {target_hash_keys}"
        )
    if int(instance.N) != manifest.N or int(instance.master_seed) != manifest.problem_seed:
        raise RunnerContractError("built instance identity disagrees with manifest")
    if (
        int(instance.d4_numerator),
        int(instance.d4_denominator),
        int(instance.r6_numerator),
        int(instance.r6_denominator),
    ) != (1, 2, 1, 16):
        raise RunnerContractError("built instance setting disagrees with frozen setting")
    observed_resources = LogicalResources(
        B=int(instance.resources.B),
        C=int(instance.resources.C),
        X=int(instance.resources.X),
        O=int(instance.resources.O),
        S=int(instance.resources.S),
        P1=int(instance.resources.P1),
    )
    if observed_resources != manifest.resources:
        raise RunnerContractError(
            "manifest logical resources do not exactly match the built instance"
        )
    configuration = configure_deterministic_device(device)
    problem = to_reduced_problem(instance)
    dynamics = AppendixDynamics(problem, device=configuration.device)
    if dynamics.real_dtype != torch.float64 or dynamics.complex_dtype != torch.complex128:
        raise RunnerContractError("runtime dynamics is not float64/complex128")
    instance_hashes = {str(key): str(value).upper() for key, value in instance.hashes.items()}
    instance_record = {
        "builder": "bcst_v2.instance_core.build_training_instance",
        "target_blind": True,
        "N": manifest.N,
        "problem_seed": manifest.problem_seed,
        "setting": {
            "label": FIXED_SETTING,
            "d4_num": 1,
            "d4_den": 2,
            "r6_num": 1,
            "r6_den": 16,
        },
        "dimension": int(instance.dimension),
        "counts": {str(key): int(value) for key, value in instance.counts.items()},
        "resources": asdict(observed_resources),
        "hashes": instance_hashes,
    }
    _assert_target_blind(instance_record)
    device_record = _json_safe(configuration)
    return _RuntimeBundle(dynamics, instance_record, device_record)


def load_stage1_artifact(
    state_path: Path | str,
    record_path: Path | str,
    *,
    expected_N: int,
    expected_amplitude_sha256: str,
    expected_record_sha256: str,
    expected_regeneration_evidence: Stage1RegenerationEvidence,
) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    if not isinstance(expected_regeneration_evidence, Stage1RegenerationEvidence):
        raise RunnerContractError("bound Stage-1 regeneration evidence is mandatory")
    if expected_regeneration_evidence.N != expected_N:
        raise RunnerContractError("Stage-1 evidence N violates the job boundary")
    if (
        expected_regeneration_evidence.selected_amplitude_sha256
        != expected_amplitude_sha256.lower()
    ):
        raise RunnerContractError(
            "Stage-1 evidence amplitude hash violates the job boundary"
        )
    if (
        expected_regeneration_evidence.record_sha256
        != expected_record_sha256.lower()
    ):
        raise RunnerContractError(
            "Stage-1 evidence record hash violates the job boundary"
        )
    observed_record_hash = sha256_file(record_path)
    if observed_record_hash != expected_record_sha256.upper():
        raise RunnerContractError("Stage-1 record SHA-256 violates the job boundary")
    complete, complete_hash = _verify_complete_marker(
        record_path, expected_record_sha256=expected_record_sha256
    )
    record = _load_json(record_path)
    if record.get("schema") != STAGE1_RECORD_SCHEMA:
        raise RunnerContractError("Stage-1 record has the wrong schema")
    if record.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise RunnerContractError("Stage-1 artifact is bound to the wrong plan")
    if record.get("N") != expected_N:
        raise RunnerContractError("Stage-1 N does not match the method cell")
    try:
        validate_stage1_record_against_evidence(
            record,
            record_sha256=observed_record_hash,
            evidence=expected_regeneration_evidence,
        )
    except SelectionError as exc:
        raise RunnerContractError(str(exc)) from exc
    state_record = record.get("selected_state", {})
    if not isinstance(state_record, Mapping):
        raise RunnerContractError("Stage-1 selected-state record is missing")
    observed_file_hash = sha256_file(state_path)
    if observed_file_hash != state_record.get("npy_file_sha256"):
        raise RunnerContractError("Stage-1 NPY file hash mismatch")
    if observed_file_hash != str(complete.get("state_file_sha256", "")).upper():
        raise RunnerContractError("Stage-1 COMPLETE marker does not bind its state")
    state = _load_serialized_state(state_path)
    observed_amplitude_hash = amplitude_sha256(state)
    recorded_amplitude_hash = str(state_record.get("amplitude_sha256", "")).upper()
    if observed_amplitude_hash != recorded_amplitude_hash:
        raise RunnerContractError("Stage-1 amplitude hash mismatch")
    if observed_amplitude_hash != str(
        complete.get("state_amplitude_sha256", "")
    ).upper():
        raise RunnerContractError("Stage-1 COMPLETE marker does not bind its amplitudes")
    if observed_amplitude_hash != expected_amplitude_sha256.upper():
        raise RunnerContractError("Stage-1 amplitude hash violates the job boundary")
    return np.array(state, dtype="<c16", order="C", copy=True), record, {
        "record_file_sha256": observed_record_hash,
        "complete_file_sha256": complete_hash,
        "npy_file_sha256": observed_file_hash,
        "amplitude_sha256": observed_amplitude_hash,
    }


def _resource_record(
    resources: LogicalResources,
    method: str,
    depth: int | None,
) -> dict[str, Any]:
    canonical = canonical_method_id(method)
    if canonical == "stage1":
        total = resources.P1
        formula_role = "fresh Stage-1 preparation P1"
    else:
        total = terminal_ru(resources, canonical, depth)
        formula_role = get_method_spec(canonical).resource_kind if canonical != "stage1_only" else "stage1_only"
    if isinstance(total, bool) or int(total) < 0:
        raise RunnerContractError("terminal_RU must be a nonnegative integer")
    return {
        "method": canonical,
        "depth": depth,
        "symbols": asdict(resources),
        "formula_role": formula_role,
        "terminal_RU": int(total),
        "terminal_objective_verification": canonical != "stage1",
        "optimizer_evaluations_included": False,
    }


def _initializer_domain(manifest: JobManifest) -> InitializerDomain:
    return InitializerDomain(
        phase=manifest.phase,
        method=canonical_method_id(manifest.method),
        N=manifest.N,
        setting=manifest.setting,
        problem_seed=manifest.problem_seed,
        depth=int(manifest.depth),
        arm=manifest.initializer_arm,
    )


def _validate_and_record_adam(
    result: Any,
    expected_initializers: Sequence[np.ndarray],
    *,
    maximum_steps: int,
) -> dict[str, Any]:
    restarts = tuple(result.restarts)
    if len(restarts) != RESTART_COUNT or len(expected_initializers) != RESTART_COUNT:
        raise RunnerContractError("Adam must execute exactly three restarts")
    restart_records: list[dict[str, Any]] = []
    restart_hashes: list[str] = []
    for expected_index, (restart, expected_initializer) in enumerate(
        zip(restarts, expected_initializers)
    ):
        if int(restart.restart_index) != expected_index:
            raise RunnerContractError("Adam restarts were not returned in sequential order")
        if str(restart.role) != "independent_random":
            raise RunnerContractError("Adam restart has a forbidden role")
        observed_initializer = _as_numpy_parameters(restart.initializer)
        expected_initializer = _as_numpy_parameters(expected_initializer)
        if not np.array_equal(observed_initializer, expected_initializer):
            raise RunnerContractError("Adam checkpoint zero violates the KDF initializer")
        observed_initializer_hash = _parameter_sha256(observed_initializer)
        if str(restart.initializer_sha256).upper() != observed_initializer_hash:
            raise RunnerContractError("Adam initializer hash mismatch")
        trace = tuple(restart.trace)
        checkpoints = [int(point.checkpoint) for point in trace]
        if checkpoints != list(range(maximum_steps + 1)):
            raise RunnerContractError(
                "every restart must record checkpoint zero and every exact Adam update"
            )
        trace_records: list[dict[str, Any]] = []
        for point in trace:
            parameter_hash = _parameter_sha256(point.parameters)
            if str(point.parameter_sha256).upper() != parameter_hash:
                raise RunnerContractError("Adam trace parameter hash mismatch")
            trace_records.append(
                {
                    "checkpoint": int(point.checkpoint),
                    "energy": _as_float(point.energy, label="trace energy"),
                    "gradient_norm": _as_float(
                        point.gradient_norm, label="trace gradient norm"
                    ),
                    "gradient_sha256": _parameter_sha256(point.gradient),
                    "gradient_float_hex": _parameter_hex(point.gradient),
                    "parameter_sha256": parameter_hash,
                    "parameters_float_hex": _parameter_hex(point.parameters),
                }
            )
        selected_parameter_hash = _parameter_sha256(restart.selected_parameters)
        if selected_parameter_hash != str(restart.selected_parameter_sha256).upper():
            raise RunnerContractError("restart selected-parameter hash mismatch")
        selected_checkpoint = int(restart.selected_checkpoint)
        if selected_checkpoint < 0 or selected_checkpoint > maximum_steps:
            raise RunnerContractError("restart selected checkpoint is outside its trace")
        selected_point = trace_records[selected_checkpoint]
        selected_energy_value = _as_float(
            restart.selected_energy, label="restart selected energy"
        )
        selected_gradient_hash = _parameter_sha256(restart.selected_gradient)
        if (
            selected_point["parameter_sha256"] != selected_parameter_hash
            or not math.isclose(
                selected_point["energy"],
                selected_energy_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or selected_point["gradient_sha256"] != selected_gradient_hash
        ):
            raise RunnerContractError("restart incumbent disagrees with its trace")
        restart_record = {
            "restart_index": expected_index,
            "execution_order": expected_index,
            "role": "independent_random",
            "continuation": False,
            "initializer_sha256": observed_initializer_hash,
            "initializer_float_hex": _parameter_hex(observed_initializer),
            "executed_updates": maximum_steps,
            "selected_checkpoint": selected_checkpoint,
            "selected_energy": selected_energy_value,
            "selected_gradient_norm": _as_float(
                restart.selected_gradient_norm,
                label="restart selected gradient norm",
            ),
            "selected_parameter_sha256": selected_parameter_hash,
            "selected_gradient_sha256": selected_gradient_hash,
            "trace": trace_records,
        }
        restart_hash = hashlib.sha256(canonical_json_bytes(restart_record)).hexdigest().upper()
        restart_record["restart_record_sha256"] = restart_hash
        restart_records.append(restart_record)
        restart_hashes.append(restart_hash)

    selected_restart = int(result.selected_restart)
    if selected_restart not in range(RESTART_COUNT):
        raise RunnerContractError("Adam selected an invalid restart")
    selected_checkpoint = int(result.selected_checkpoint)
    selected_hash = _parameter_sha256(result.selected_parameters)
    if selected_hash != str(result.selected_parameter_sha256).upper():
        raise RunnerContractError("global selected-parameter hash mismatch")
    chosen = restart_records[selected_restart]
    if (
        selected_checkpoint != chosen["selected_checkpoint"]
        or selected_hash != chosen["selected_parameter_sha256"]
    ):
        raise RunnerContractError("global Adam incumbent disagrees with restart record")
    selected_energy = _as_float(result.selected_energy, label="selected energy")
    selected_gradient_norm = _as_float(
        result.selected_gradient_norm, label="selected gradient norm"
    )
    if (
        not math.isclose(
            selected_energy,
            chosen["selected_energy"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _parameter_sha256(result.selected_gradient)
        != chosen["selected_gradient_sha256"]
    ):
        raise RunnerContractError("global Adam incumbent disagrees with selected restart")
    return {
        "optimizer": "Adam",
        "restart_count": RESTART_COUNT,
        "execution": "strictly_sequential_0_1_2",
        "early_stopping": False,
        "continuation": False,
        "executed_updates_per_restart": maximum_steps,
        "restarts": restart_records,
        "restart_hashes": restart_hashes,
        "selected_restart": selected_restart,
        "selected_checkpoint": selected_checkpoint,
        "selected_energy": selected_energy,
        "selected_gradient_norm": selected_gradient_norm,
        "selected_parameter_sha256": selected_hash,
        "selected_parameters_float_hex": _parameter_hex(result.selected_parameters),
        "selected_gradient_sha256": _parameter_sha256(result.selected_gradient),
        "selected_gradient_float_hex": _parameter_hex(result.selected_gradient),
        "objective_evaluations": int(result.objective_evaluations),
    }


def _run_adam_cell(
    manifest: JobManifest,
    dynamics: AppendixDynamics,
    *,
    phi: np.ndarray | None,
    device: str,
    activation_checkpointing: bool,
) -> tuple[dict[str, Any], np.ndarray, float, float]:
    method = canonical_method_id(manifest.method)
    depth = int(manifest.depth)
    budget = int(manifest.adopted_budget)
    layout = parameter_layout_for_method(method, depth)
    count = sum(int(size) for _, size in layout)
    domain = _initializer_domain(manifest)
    expected_initializers = derive_three_initializers(domain, layout)
    if len(expected_initializers) != RESTART_COUNT:
        raise RunnerContractError("initializer derivation did not return exactly three starts")

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        energy, _ = dynamics.energy_from_parameters(
            method,
            parameters,
            depth=depth,
            phi=phi,
            activation_checkpointing=activation_checkpointing,
        )
        return energy

    config = AdamConfig(maximum_steps=budget)
    result = run_adam(
        objective,
        domain=domain,
        parameter_layout=layout,
        config=config,
        device=device,
    )
    if result.domain != domain or result.config != config:
        raise RunnerContractError("Adam result domain/config does not match the job")
    if int(result.objective_evaluations) != RESTART_COUNT * (budget + 1):
        raise RunnerContractError("Adam objective-evaluation count is inconsistent")
    optimization = _validate_and_record_adam(
        result, expected_initializers, maximum_steps=budget
    )
    if tuple(result.parameter_layout) != tuple(layout):
        raise RunnerContractError("Adam result parameter layout does not match the registry")
    selected_parameters = torch.tensor(
        _as_numpy_parameters(result.selected_parameters),
        dtype=dynamics.real_dtype,
        device=dynamics.device,
        requires_grad=True,
    )
    replay_energy_tensor, replay_state_tensor = dynamics.energy_from_parameters(
        method,
        selected_parameters,
        depth=depth,
        phi=phi,
        activation_checkpointing=False,
    )
    replay_energy = _as_float(replay_energy_tensor, label="selected replay energy")
    if abs(replay_energy - optimization["selected_energy"]) > ENERGY_REPLAY_TOLERANCE:
        raise RunnerContractError("selected-state replay energy disagrees with Adam")
    replay_gradient_tensor = torch.autograd.grad(
        replay_energy_tensor,
        selected_parameters,
        create_graph=False,
        retain_graph=False,
    )[0]
    if not bool(torch.all(torch.isfinite(replay_gradient_tensor))):
        raise RunnerContractError("selected-point replay gradient is nonfinite")
    replay_gradient = _as_numpy_parameters(replay_gradient_tensor)
    stored_gradient = _as_numpy_parameters(result.selected_gradient)
    if replay_gradient.shape != stored_gradient.shape or not np.allclose(
        replay_gradient,
        stored_gradient,
        rtol=1e-11,
        atol=1e-11,
    ):
        raise RunnerContractError("selected-point replay gradient disagrees with Adam")
    replay_gradient_norm = float(np.linalg.norm(replay_gradient))
    if not math.isfinite(replay_gradient_norm):
        raise RunnerContractError("selected-point replay gradient norm is nonfinite")
    if not math.isclose(
        replay_gradient_norm,
        optimization["selected_gradient_norm"],
        rel_tol=1e-11,
        abs_tol=1e-11,
    ):
        raise RunnerContractError("selected-point replay gradient norm disagrees with Adam")
    state = _canonical_state(replay_state_tensor)
    zero_state = dynamics.zero_angle_state(method, depth=depth, phi=phi)
    zero_energy = _as_float(
        dynamics.expected_energy(method, zero_state), label="zero-angle energy"
    )
    feasible_mass = _as_float(
        dynamics.feasible_mass(replay_state_tensor), label="feasible mass"
    )
    optimization["zero_angle_diagnostic"] = {
        "optimizer_eligible": False,
        "energy": zero_energy,
    }
    optimization["selected_replay_energy"] = replay_energy
    raw_state = np.ascontiguousarray(
        replay_state_tensor.detach().cpu().numpy().astype("<c16", copy=False)
    )
    optimization["selected_replay"] = {
        "objective_evaluations": 1,
        "gradient_evaluations": 1,
        "excluded_from_optimizer_selection": True,
        "energy": replay_energy,
        "energy_matches_optimizer": True,
        "gradient_norm": replay_gradient_norm,
        "gradient_sha256": _parameter_sha256(replay_gradient),
        "gradient_matches_optimizer": True,
        "raw_state_sha256": hashlib.sha256(raw_state.tobytes(order="C")).hexdigest().upper(),
        "canonical_state_amplitude_sha256": amplitude_sha256(state),
        "raw_probability_norm": float(np.vdot(raw_state, raw_state).real),
        "canonical_probability_norm": float(np.vdot(state, state).real),
    }
    optimization["evaluation_accounting"] = {
        "optimizer_objective_and_gradient_evaluations": int(
            result.objective_evaluations
        ),
        "selected_point_replay_objective_and_gradient_evaluations": 1,
        "zero_angle_diagnostic_objective_evaluations": 1,
    }
    optimization["health"] = {
        "finite_energy_and_gradient": True,
        "complete_three_restarts": True,
        "selected_no_worse_than_checkpoint_zero": all(
            optimization["selected_energy"] <= row["trace"][0]["energy"] + 1e-12
            for row in optimization["restarts"]
            if row["restart_index"] == optimization["selected_restart"]
        ),
        "selected_no_worse_than_zero_angle_within_1e-9": (
            optimization["selected_energy"] <= zero_energy + 1e-9
        ),
    }
    return optimization, state, replay_energy, feasible_mass


def _state_record(state: np.ndarray, state_path: Path, state_file_hash: str) -> dict[str, Any]:
    probability_norm = float(np.vdot(state, state).real)
    return {
        "path": str(state_path.resolve()),
        "npy_file_sha256": state_file_hash,
        "amplitude_sha256": amplitude_sha256(state),
        "dtype": "<c16",
        "dimension": int(state.size),
        "probability_norm": probability_norm,
        "normalization_error": abs(probability_norm - 1.0),
    }


def _write_cell_record(
    output_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    state: np.ndarray,
    record: dict[str, Any],
) -> CellArtifacts:
    state_path = output_dir / "selected_state.npy"
    state_file_hash = atomic_write_npy(state_path, state)
    record["manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": manifest_sha256,
    }
    record["selected_state"] = _state_record(state, state_path, state_file_hash)
    _assert_target_blind(record)
    record_path = output_dir / "target_blind_record.json"
    record_hash = atomic_write_json(record_path, record)
    complete_record = {
        "schema": COMPLETE_SCHEMA,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "cell_id": record["cell_id"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "record_path": str(record_path.resolve()),
        "record_sha256": record_hash,
        "state_path": str(state_path.resolve()),
        "state_file_sha256": state_file_hash,
        "state_amplitude_sha256": record["selected_state"]["amplitude_sha256"],
        "written_last": True,
    }
    complete_path = output_dir / "COMPLETE.json"
    complete_hash = atomic_write_json(complete_path, complete_record)
    return CellArtifacts(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        record_path=record_path.resolve(),
        record_sha256=record_hash,
        state_path=state_path.resolve(),
        state_file_sha256=state_file_hash,
        amplitude_sha256=record["selected_state"]["amplitude_sha256"],
        complete_path=complete_path.resolve(),
        complete_sha256=complete_hash,
        record=record,
    )


def _run_stage1_cell_with_runtime(
    manifest: JobManifest,
    dynamics: AppendixDynamics,
    output_dir: Path | str,
    *,
    instance_record: Mapping[str, Any],
    device_record: Mapping[str, Any],
    device: str = "cuda",
    activation_checkpointing: bool = True,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> CellArtifacts:
    """Run and freeze one fresh Stage-1 cell from exactly three starts."""

    if manifest.cell_type != "stage1":
        raise RunnerContractError("run_stage1_cell requires a Stage-1 manifest")
    output = Path(output_dir)
    manifest_path, manifest_hash = write_job_manifest(
        manifest, output, repo_root=repo_root
    )
    optimization, state, replay_energy, feasible_mass = _run_adam_cell(
        manifest,
        dynamics,
        phi=None,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )
    record = {
        "schema": STAGE1_RECORD_SCHEMA,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "target_blind": True,
        "cell_id": manifest.cell_id,
        "cell_type": "stage1",
        "phase": manifest.phase,
        "N": manifest.N,
        "problem_seed": manifest.problem_seed,
        "setting": manifest.setting,
        "method": "stage1",
        "depth": STAGE1_DEPTH,
        "adopted_budget": manifest.adopted_budget,
        "optimization": optimization,
        "selected_energy": replay_energy,
        "selected_gradient_norm": optimization["selected_gradient_norm"],
        "selected_checkpoint": optimization["selected_checkpoint"],
        "feasible_mass": feasible_mass,
        "instance": dict(instance_record),
        "deterministic_device": dict(device_record),
        "resource_record": _resource_record(
            manifest.resources, "stage1", STAGE1_DEPTH
        ),
    }
    return _write_cell_record(output, manifest_path, manifest_hash, state, record)


def run_stage1_cell(
    manifest: JobManifest,
    output_dir: Path | str,
    *,
    device: str = "cuda",
    activation_checkpointing: bool = True,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> CellArtifacts:
    """Production Stage-1 entry point; builds its own target-blind instance."""

    if manifest.cell_type != "stage1":
        raise RunnerContractError("run_stage1_cell requires a Stage-1 manifest")
    verify_plan_binding(repo_root=repo_root)
    runtime = _build_target_blind_runtime(manifest, device=device)
    return _run_stage1_cell_with_runtime(
        manifest,
        runtime.dynamics,
        output_dir,
        instance_record=runtime.instance_record,
        device_record=runtime.device_record,
        device=str(runtime.dynamics.device),
        activation_checkpointing=activation_checkpointing,
        repo_root=repo_root,
    )


def _run_method_cell_with_runtime(
    manifest: JobManifest,
    dynamics: AppendixDynamics,
    output_dir: Path | str,
    *,
    instance_record: Mapping[str, Any],
    device_record: Mapping[str, Any],
    device: str = "cuda",
    activation_checkpointing: bool = True,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> CellArtifacts:
    """Run one target-blind optimized method cell from exactly three starts."""

    if manifest.cell_type != "method":
        raise RunnerContractError("run_method_cell requires a method manifest")
    method = canonical_method_id(manifest.method)
    output = Path(output_dir)
    manifest_path, manifest_hash = write_job_manifest(
        manifest, output, repo_root=repo_root
    )
    phi: np.ndarray | None = None
    stage1_boundary: dict[str, Any] | None = None
    if method in PHI_INITIAL_METHODS:
        phi, stage1_record, stage1_hashes = load_stage1_artifact(
            manifest.stage1_state_path,
            manifest.stage1_record_path,
            expected_N=manifest.N,
            expected_amplitude_sha256=manifest.expected_stage1_amplitude_sha256,
            expected_record_sha256=manifest.expected_stage1_record_sha256,
            expected_regeneration_evidence=manifest.stage1_regeneration_evidence,
        )
        stage1_boundary = {
            "state_path": str(Path(manifest.stage1_state_path).resolve()),
            "record_path": str(Path(manifest.stage1_record_path).resolve()),
            "expected_record_sha256": manifest.expected_stage1_record_sha256,
            "regeneration_evidence": manifest.stage1_regeneration_evidence.to_record(),
            **stage1_hashes,
            "source_cell_id": stage1_record["cell_id"],
            "source_restart_hashes": stage1_record["optimization"]["restart_hashes"],
            "identity_boundary_passed": True,
        }
    elif method in NATIVE_INITIAL_METHODS:
        if manifest.stage1_state_path is not None:
            raise RunnerContractError(f"native method {method} prohibits phi")
    else:  # pragma: no cover - registry partitions optimized methods.
        raise RunnerContractError(f"method {method} has no initial-state class")

    optimization, state, replay_energy, feasible_mass = _run_adam_cell(
        manifest,
        dynamics,
        phi=phi,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )
    record = {
        "schema": METHOD_RECORD_SCHEMA,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "target_blind": True,
        "cell_id": manifest.cell_id,
        "cell_type": "method",
        "phase": manifest.phase,
        "N": manifest.N,
        "problem_seed": manifest.problem_seed,
        "setting": manifest.setting,
        "method": method,
        "depth": manifest.depth,
        "adopted_budget": manifest.adopted_budget,
        "stage1_boundary": stage1_boundary,
        "optimization": optimization,
        "selected_energy": replay_energy,
        "selected_gradient_norm": optimization["selected_gradient_norm"],
        "selected_checkpoint": optimization["selected_checkpoint"],
        "feasible_mass": feasible_mass,
        "instance": dict(instance_record),
        "deterministic_device": dict(device_record),
        "resource_record": _resource_record(
            manifest.resources, method, manifest.depth
        ),
    }
    return _write_cell_record(output, manifest_path, manifest_hash, state, record)


def run_method_cell(
    manifest: JobManifest,
    output_dir: Path | str,
    *,
    device: str = "cuda",
    activation_checkpointing: bool = True,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> CellArtifacts:
    """Production method entry point; builds its own target-blind instance."""

    if manifest.cell_type != "method":
        raise RunnerContractError("run_method_cell requires a method manifest")
    verify_plan_binding(repo_root=repo_root)
    runtime = _build_target_blind_runtime(manifest, device=device)
    return _run_method_cell_with_runtime(
        manifest,
        runtime.dynamics,
        output_dir,
        instance_record=runtime.instance_record,
        device_record=runtime.device_record,
        device=str(runtime.dynamics.device),
        activation_checkpointing=activation_checkpointing,
        repo_root=repo_root,
    )


def _run_stage1_only_cell_with_runtime(
    manifest: JobManifest,
    dynamics: AppendixDynamics,
    output_dir: Path | str,
    *,
    instance_record: Mapping[str, Any],
    device_record: Mapping[str, Any],
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> CellArtifacts:
    """Freeze a target-blind Stage1Only cell; no Stage-2 restart is run."""

    if manifest.cell_type != "stage1_only":
        raise RunnerContractError("run_stage1_only_cell requires Stage1Only manifest")
    output = Path(output_dir)
    manifest_path, manifest_hash = write_job_manifest(
        manifest, output, repo_root=repo_root
    )
    phi, stage1_record, stage1_hashes = load_stage1_artifact(
        manifest.stage1_state_path,
        manifest.stage1_record_path,
        expected_N=manifest.N,
        expected_amplitude_sha256=manifest.expected_stage1_amplitude_sha256,
        expected_record_sha256=manifest.expected_stage1_record_sha256,
        expected_regeneration_evidence=manifest.stage1_regeneration_evidence,
    )
    state_tensor = torch.as_tensor(
        phi, dtype=dynamics.complex_dtype, device=dynamics.device
    )
    h_c = _as_float(
        dynamics.expected_energy("stage1", state_tensor),
        label="Stage1Only H_C",
    )
    feasible_mass = _as_float(
        dynamics.feasible_mass(state_tensor), label="Stage1Only feasible mass"
    )
    source_optimization = stage1_record["optimization"]
    record = {
        "schema": STAGE1_ONLY_RECORD_SCHEMA,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "target_blind": True,
        "cell_id": manifest.cell_id,
        "cell_type": "stage1_only",
        "phase": manifest.phase,
        "N": manifest.N,
        "problem_seed": manifest.problem_seed,
        "setting": manifest.setting,
        "method": "stage1_only",
        "depth": None,
        "adopted_budget": None,
        "optimizer": "not_applicable",
        "stage2_restart_count": 0,
        "stage2_health": "not_applicable",
        "stage1_boundary": {
            "state_path": str(Path(manifest.stage1_state_path).resolve()),
            "record_path": str(Path(manifest.stage1_record_path).resolve()),
            "expected_record_sha256": manifest.expected_stage1_record_sha256,
            "regeneration_evidence": manifest.stage1_regeneration_evidence.to_record(),
            **stage1_hashes,
            "source_cell_id": stage1_record["cell_id"],
            "source_restart_hashes": source_optimization["restart_hashes"],
            "identity_boundary_passed": True,
        },
        "restart_hashes": source_optimization["restart_hashes"],
        "restart_hash_scope": "frozen_stage1_source",
        "selected_energy": h_c,
        "selected_gradient_norm": source_optimization["selected_gradient_norm"],
        "selected_checkpoint": source_optimization["selected_checkpoint"],
        "feasible_mass": feasible_mass,
        "instance": dict(instance_record),
        "deterministic_device": dict(device_record),
        "health": {
            "finite_normalized_frozen_stage1": True,
            "verified_amplitude_hash": True,
            "finite_H_C": True,
            "exact_resource_accounting": True,
            "stage2_optimizer_checks": "not_applicable",
        },
        "resource_record": _resource_record(
            manifest.resources, "stage1_only", None
        ),
    }
    # Store an immutable local copy so every locked cell has a self-contained,
    # hash-verified selected-state artifact while preserving exact amplitudes.
    return _write_cell_record(output, manifest_path, manifest_hash, phi, record)


def run_stage1_only_cell(
    manifest: JobManifest,
    output_dir: Path | str,
    *,
    device: str = "cuda",
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> CellArtifacts:
    """Production Stage1Only entry point with a verified objective instance."""

    if manifest.cell_type != "stage1_only":
        raise RunnerContractError("run_stage1_only_cell requires Stage1Only manifest")
    verify_plan_binding(repo_root=repo_root)
    runtime = _build_target_blind_runtime(manifest, device=device)
    return _run_stage1_only_cell_with_runtime(
        manifest,
        runtime.dynamics,
        output_dir,
        instance_record=runtime.instance_record,
        device_record=runtime.device_record,
        repo_root=repo_root,
    )


def _record_restart_hashes(record: Mapping[str, Any]) -> list[str]:
    if record.get("cell_type") in {"stage1", "method"}:
        hashes = record.get("optimization", {}).get("restart_hashes", [])
    else:
        hashes = record.get("restart_hashes", [])
    hashes = [str(item).upper() for item in hashes]
    if len(hashes) != RESTART_COUNT or not all(_is_sha256(item) for item in hashes):
        raise RunnerContractError("locked cell lacks exactly three valid restart hashes")
    return hashes


def lock_target_blind_cohort(
    manifest_paths: Sequence[Path | str],
    record_paths: Sequence[Path | str],
    lock_path: Path | str,
    *,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> LockArtifacts:
    """Atomically freeze all expected target-blind cell records.

    The exact set of manifests must match the exact set of records by cell ID.
    The lock contains the fields required by the amended plan and immutable
    hashes of every source artifact.  Re-running against an existing lock is
    rejected rather than overwriting it.
    """

    verify_plan_binding(repo_root=repo_root)
    # Local imports avoid a module-import cycle while making this public lock
    # boundary independently enforce the same strict V3 admission contract as
    # the queue completion verifier.
    from .campaign_records import validate_target_blind_record_against_manifest
    from .cell_cli import manifest_from_record

    manifests: dict[str, tuple[Path, dict[str, Any], str, JobManifest]] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        payload = _load_json(path)
        if payload.get("schema") != JOB_MANIFEST_SCHEMA:
            raise RunnerContractError(f"{path} is not a job manifest")
        _assert_target_blind(payload)
        try:
            parsed_manifest = manifest_from_record(payload)
        except ValueError as exc:
            raise RunnerContractError(f"{path} has an invalid V3 job manifest: {exc}") from exc
        cell_id = str(payload.get("cell_id"))
        if cell_id in manifests:
            raise RunnerContractError(f"duplicate expected manifest {cell_id}")
        manifests[cell_id] = (path, payload, sha256_file(path), parsed_manifest)

    records: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for raw_path in record_paths:
        path = Path(raw_path).resolve()
        payload = _load_json(path)
        if payload.get("schema") not in {
            STAGE1_RECORD_SCHEMA,
            METHOD_RECORD_SCHEMA,
            STAGE1_ONLY_RECORD_SCHEMA,
        }:
            raise RunnerContractError(f"{path} is not a target-blind cell record")
        _assert_target_blind(payload)
        cell_id = str(payload.get("cell_id"))
        if cell_id in records:
            raise RunnerContractError(f"duplicate target-blind record {cell_id}")
        records[cell_id] = (path, payload, sha256_file(path))

    if set(manifests) != set(records):
        missing = sorted(set(manifests).difference(records))
        unexpected = sorted(set(records).difference(manifests))
        raise RunnerContractError(
            f"cohort is incomplete or unexpected; missing={missing}, unexpected={unexpected}"
        )
    if not manifests:
        raise RunnerContractError("cannot lock an empty cohort")

    rows: list[dict[str, Any]] = []
    for cell_id in sorted(manifests):
        manifest_path, manifest, manifest_hash, parsed_manifest = manifests[cell_id]
        record_path, record, record_hash = records[cell_id]
        complete, complete_hash = _verify_complete_marker(
            record_path, expected_record_sha256=record_hash
        )
        if record.get("plan_sha256") != EXPECTED_PLAN_SHA256:
            raise RunnerContractError(f"record {cell_id} has the wrong plan hash")
        recorded_manifest = record.get("manifest", {})
        if str(recorded_manifest.get("sha256", "")).upper() != manifest_hash:
            raise RunnerContractError(f"record {cell_id} has the wrong manifest hash")
        if str(complete.get("manifest_sha256", "")).upper() != manifest_hash:
            raise RunnerContractError(f"COMPLETE marker {cell_id} has the wrong manifest hash")
        try:
            validate_target_blind_record_against_manifest(record, parsed_manifest)
        except ValueError as exc:
            raise RunnerContractError(
                f"record {cell_id} violates its V3 manifest or optimizer/KDF contract: {exc}"
            ) from exc
        state_record = record.get("selected_state", {})
        state_path = Path(str(state_record.get("path", ""))).resolve()
        if not state_path.is_file():
            raise RunnerContractError(f"record {cell_id} selected-state file is missing")
        state_file_hash = sha256_file(state_path)
        if state_file_hash != str(state_record.get("npy_file_sha256", "")).upper():
            raise RunnerContractError(f"record {cell_id} selected-state hash mismatch")
        if state_file_hash != str(complete.get("state_file_sha256", "")).upper():
            raise RunnerContractError(f"COMPLETE marker {cell_id} has the wrong state hash")
        state = _load_serialized_state(state_path)
        amp_hash = amplitude_sha256(state)
        if amp_hash != str(state_record.get("amplitude_sha256", "")).upper():
            raise RunnerContractError(f"record {cell_id} amplitude hash mismatch")
        restart_hashes = _record_restart_hashes(record)
        resource_record = record.get("resource_record")
        if not isinstance(resource_record, Mapping) or not isinstance(
            resource_record.get("terminal_RU"), int
        ):
            raise RunnerContractError(f"record {cell_id} lacks exact resource accounting")
        rows.append(
            {
                "cell_id": cell_id,
                "N": int(record["N"]),
                "problem_seed": int(record["problem_seed"]),
                "method": str(record["method"]),
                "adopted_budget": record.get("adopted_budget"),
                "depth": record.get("depth"),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "record_path": str(record_path),
                "record_sha256": record_hash,
                "complete_path": str((record_path.parent / "COMPLETE.json").resolve()),
                "complete_sha256": complete_hash,
                "restart_hashes": restart_hashes,
                "selected_checkpoint": record.get("selected_checkpoint"),
                "selected_state_path": str(state_path),
                "selected_state_file_sha256": state_file_hash,
                "selected_state_amplitude_sha256": amp_hash,
                "selected_energy": _as_float(
                    record["selected_energy"], label="locked selected energy"
                ),
                "selected_gradient_norm": _as_float(
                    record["selected_gradient_norm"],
                    label="locked selected gradient norm",
                ),
                "stage1_amplitude_sha256": (
                    record.get("stage1_boundary", {}) or {}
                ).get("amplitude_sha256"),
                "resource_record": dict(resource_record),
            }
        )
    lock_record = {
        "schema": COHORT_LOCK_SCHEMA,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "target_blind": True,
        "atomic_immutable_lock": True,
        "cell_count": len(rows),
        "cells": rows,
    }
    _assert_target_blind(lock_record)
    destination = Path(lock_path).resolve()
    lock_hash = atomic_write_json(destination, lock_record)
    return LockArtifacts(destination, lock_hash, lock_record)


def evaluate_targets_after_lock(
    lock_path: Path | str,
    evaluator: TargetEvaluator,
    output_path: Path | str,
    *,
    expected_lock_sha256: str,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> tuple[Path, str, dict[str, Any]]:
    """Evaluate target identities and probabilities only after a valid lock."""

    verify_plan_binding(repo_root=repo_root)
    locked_path = Path(lock_path).resolve()
    observed_lock_hash = sha256_file(locked_path)
    if not _is_sha256(expected_lock_sha256):
        raise RunnerContractError("expected cohort-lock SHA-256 is required")
    if observed_lock_hash != expected_lock_sha256.upper():
        raise RunnerContractError("cohort lock SHA-256 mismatch")
    lock_record = _load_json(locked_path)
    if lock_record.get("schema") != COHORT_LOCK_SCHEMA:
        raise RunnerContractError("target evaluation requires a valid cohort lock")
    locked_cells = lock_record.get("cells")
    if (
        lock_record.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or lock_record.get("atomic_immutable_lock") is not True
        or not isinstance(locked_cells, list)
        or not locked_cells
        or lock_record.get("cell_count") != len(locked_cells)
    ):
        raise RunnerContractError("cohort lock integrity fields are invalid")
    _assert_target_blind(lock_record)

    verified_cells: list[tuple[Mapping[str, Any], dict[str, Any], np.ndarray]] = []
    for locked in locked_cells:
        if not isinstance(locked, Mapping):
            raise RunnerContractError("cohort lock contains a malformed cell entry")
        complete_path = Path(locked["complete_path"]).resolve()
        record_path = Path(locked["record_path"]).resolve()
        if sha256_file(record_path) != locked["record_sha256"]:
            raise RunnerContractError("a target-blind record changed after cohort lock")
        _, complete_hash = _verify_complete_marker(
            record_path,
            expected_record_sha256=str(locked["record_sha256"]),
        )
        if complete_path != (record_path.parent / "COMPLETE.json").resolve():
            raise RunnerContractError("cohort lock points to the wrong COMPLETE marker")
        if complete_hash != locked["complete_sha256"]:
            raise RunnerContractError("a COMPLETE marker changed after cohort lock")
        job_record = _load_json(record_path)
        _assert_target_blind(job_record)
        if job_record.get("cell_id") != locked.get("cell_id"):
            raise RunnerContractError("cohort lock cell identity disagrees with its record")
        state_path = Path(locked["selected_state_path"]).resolve()
        if sha256_file(state_path) != locked["selected_state_file_sha256"]:
            raise RunnerContractError("a selected state changed after cohort lock")
        state = _load_serialized_state(state_path)
        if amplitude_sha256(state) != locked["selected_state_amplitude_sha256"]:
            raise RunnerContractError("selected amplitudes changed after cohort lock")
        verified_cells.append((locked, job_record, state))

    results: list[dict[str, Any]] = []
    for locked, job_record, state in verified_cells:
        metrics = evaluator(
            job_record=job_record,
            state=state,
            cohort_lock_path=locked_path,
            expected_lock_sha256=observed_lock_hash,
        )
        if not isinstance(metrics, Mapping):
            raise RunnerContractError("target evaluator must return a mapping")
        results.append(
            {
                "cell_id": locked["cell_id"],
                "N": locked["N"],
                "problem_seed": locked["problem_seed"],
                "method": locked["method"],
                "locked_record_sha256": locked["record_sha256"],
                "metrics": _json_safe(metrics),
            }
        )
    result_record = {
        "schema": TARGET_RESULT_SCHEMA,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "cohort_lock_path": str(locked_path),
        "cohort_lock_sha256": observed_lock_hash,
        "evaluation_role": "read_only_post_lock",
        "cell_count": len(results),
        "cells": results,
    }
    destination = Path(output_path).resolve()
    result_hash = atomic_write_json(destination, result_record)
    return destination, result_hash, result_record


__all__ = [
    "ALLOWED_BUDGETS",
    "COHORT_LOCK_SCHEMA",
    "COMPLETE_SCHEMA",
    "CellArtifacts",
    "EXPECTED_PLAN_SHA256",
    "FIXED_INITIALIZER_ARM",
    "FIXED_SETTING",
    "JOB_MANIFEST_SCHEMA",
    "JobManifest",
    "LockArtifacts",
    "LogicalResources",
    "METHOD_RECORD_SCHEMA",
    "PLAN_PATH",
    "RESTART_COUNT",
    "RunnerContractError",
    "STAGE1_ONLY_RECORD_SCHEMA",
    "STAGE1_RECORD_SCHEMA",
    "Stage1RegenerationEvidence",
    "TARGET_RESULT_SCHEMA",
    "TargetEvaluator",
    "amplitude_sha256",
    "atomic_write_json",
    "atomic_write_npy",
    "canonical_json_bytes",
    "evaluate_targets_after_lock",
    "load_stage1_artifact",
    "lock_target_blind_cohort",
    "run_method_cell",
    "run_stage1_cell",
    "run_stage1_only_cell",
    "sha256_file",
    "verify_plan_binding",
    "write_job_manifest",
]
