"""Lean single-job production runner for the corrected BCST v2 campaign.

The CLI intentionally runs one independently restartable artifact at a time:

``stage1``
    Generate the single frozen Stage-1 state for one N from exactly three
    KDF-derived Adam starts.

``job``
    Run one method/optimizer/depth/master-seed/setting cell from exactly three
    KDF-derived starts.  Optimization remains target blind.  Optional target
    metrics are computed only after the optimizer and selected-state replay
    are complete.

``stage1-only``
    Evaluate the frozen Stage-1 state on one objective instance without a
    Stage-2 optimizer.

Every artifact is bound to ``LEAN_EXECUTION_PLAN_20260730.json`` SHA-256
``172d2267...e7f6``.  State arrays are canonical little-endian complex128;
JSON is canonical, finite, and uses explicit strings for infinity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .dynamics import canonicalize_global_phase
from .instance_core import (
    BCSTInstance,
    TargetSets,
    build_training_instance,
    clip_probability,
    derive_targets,
    rts99,
    total_sampling_cost,
)
from .optimizer_protocol import (
    ADAM_BASE_STEPS,
    LEAN_EXECUTION_PLAN_SHA256 as OPTIMIZER_LEAN_EXECUTION_PLAN_SHA256,
    PHASE_TAGS,
    SPSA_BASE_UPDATES,
    AdamSettings,
    OptimizerCallCounts,
    ProtocolResult,
    RestartContext,
    SPSASettings,
    evaluate_zero_angle_diagnostic,
    generate_three_random_initializers,
    make_random_initializer,
    parameter_layout_for_method,
    parameter_sha256,
    run_adam_protocol,
    run_spsa_protocol,
    stage1_adam_settings,
    standard_adam_settings,
    standard_spsa_settings,
    validate_restart_batch,
)
from .torch_adapter import (
    ApplicationCounts,
    ReplayDiagnostics,
    TorchMethodAdapter,
    build_method_adapter,
    configure_deterministic_device,
)


LEAN_EXECUTION_PLAN_SHA256 = (
    "172d2267b18745b3518e4efccc58b367aeff9a2ec9169ff5ef322dd04762e7f6"
)
LEAN_PLAN_FILENAME = "LEAN_EXECUTION_PLAN_20260730.json"
LEAN_PLAN_PATH = Path(__file__).resolve().parents[2] / LEAN_PLAN_FILENAME
RUNNER_SCHEMA = "bcst-lean-production-runner-v1"
STAGE1_SCHEMA = "bcst-lean-stage1-artifact-v1"
JOB_SCHEMA = "bcst-lean-single-job-artifact-v1"
STAGE1_ONLY_SCHEMA = "bcst-lean-stage1-only-artifact-v1"
METHODS = (
    "learned_projector",
    "ordinary_native_direct",
    "decoupled_native_direct",
    "same_stage1_state_direct",
)
PHI_METHODS = frozenset(
    {"learned_projector", "same_stage1_state_direct"}
)
NATIVE_METHODS = frozenset(
    {"ordinary_native_direct", "decoupled_native_direct"}
)
STAGE1_SEEDS = {30: 830000030, 25: 830000025}
SETTINGS: Mapping[str, tuple[int, int, int, int]] = {
    "d4_1of2_r6_1of16": (1, 2, 1, 16),
    "d4_1of2_r6_2of16": (1, 2, 2, 16),
    "d4_3of4_r6_1of16": (3, 4, 1, 16),
    "d4_3of4_r6_2of16": (3, 4, 2, 16),
}
ADAM_ARMS = (0.01, 0.02, 0.035)
SPSA_ARMS = (0.02, 0.05, 0.1)


class RunnerError(ValueError):
    """Raised when a CLI job departs from the frozen lean execution plan."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lean_execution_plan(
    path: Path | str = LEAN_PLAN_PATH,
) -> dict[str, Any]:
    """Verify the authoritative plan bytes and its minimal identity fields."""

    if OPTIMIZER_LEAN_EXECUTION_PLAN_SHA256 != LEAN_EXECUTION_PLAN_SHA256:
        raise RunnerError(
            "runner and optimizer protocol disagree on the lean plan SHA-256"
        )
    source = Path(path)
    observed = sha256_file(source)
    if observed.lower() != LEAN_EXECUTION_PLAN_SHA256:
        raise RunnerError(
            "lean execution plan SHA-256 mismatch: "
            f"expected {LEAN_EXECUTION_PLAN_SHA256}, observed {observed.lower()}"
        )
    with open(source, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "bcst-lean-execution-plan-v1":
        raise RunnerError("lean execution plan schema is not recognized")
    if payload.get("stage1", {}).get("depth") != 12:
        raise RunnerError("lean execution plan Stage-1 depth is not 12")
    if payload.get("authoritative_user_choices", {}).get("restarts", "").split(
        " ", 1
    )[0] != "Exactly":
        raise RunnerError("lean execution plan restart rule is missing")
    return payload


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            raise RunnerError("NaN cannot be serialized")
        if math.isinf(number):
            return "positive_infinity" if number > 0 else "negative_infinity"
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise RunnerError(f"value of type {type(value).__name__} is not JSON-safe")


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


def atomic_write_json(
    path: Path | str,
    value: Any,
    *,
    overwrite: bool = False,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise RunnerError(f"refusing to overwrite existing artifact {destination}")
    payload = canonical_json_bytes(value)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_npy(
    path: Path | str,
    values: np.ndarray,
    *,
    overwrite: bool = False,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise RunnerError(f"refusing to overwrite existing artifact {destination}")
    array = np.ascontiguousarray(np.asarray(values))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        file_hash = sha256_file(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return file_hash


def _canonical_state(state: np.ndarray) -> np.ndarray:
    canonical = canonicalize_global_phase(
        np.asarray(state, dtype=np.complex128)
    )
    canonical = np.ascontiguousarray(canonical.astype("<c16", copy=False))
    if canonical.ndim != 1:
        raise RunnerError("state must be one dimensional")
    if abs(float(np.vdot(canonical, canonical).real) - 1.0) > 1e-10:
        raise RunnerError("canonical state normalization failed")
    canonical.setflags(write=False)
    return canonical


def _amplitude_sha256(state: np.ndarray) -> str:
    canonical_bytes = np.ascontiguousarray(
        np.asarray(state).astype("<c16", copy=False)
    ).tobytes(order="C")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _float_hex(values: np.ndarray | Sequence[float]) -> list[str]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(value).hex() for value in array]


def _same_float64_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare two parameter vectors as literal native binary64 bytes."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        left_array.dtype != np.dtype(np.float64)
        or right_array.dtype != np.dtype(np.float64)
        or left_array.shape != right_array.shape
    ):
        return False
    return (
        np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
    )


def _setting_record(label: str) -> dict[str, Any]:
    try:
        d4_num, d4_den, r6_num, r6_den = SETTINGS[label]
    except KeyError as exc:
        raise RunnerError(f"unknown frozen setting {label!r}") from exc
    return {
        "label": label,
        "d4_num": d4_num,
        "d4_den": d4_den,
        "r6_num": r6_num,
        "r6_den": r6_den,
    }


def _build_instance(
    N: int, master_seed: int, setting_label: str
) -> BCSTInstance:
    setting = _setting_record(setting_label)
    return build_training_instance(
        N,
        master_seed,
        setting["d4_num"],
        setting["d4_den"],
        setting["r6_num"],
        setting["r6_den"],
    )


def _make_context(
    *,
    phase: str,
    N: int,
    setting: Any,
    method: str,
    optimizer: str,
    master_seed: int,
    depth: int,
    scalar_arm: float,
) -> RestartContext:
    if optimizer == "Adam":
        arm_domain = {"learning_rate": float(scalar_arm)}
    else:
        arm_domain = {"c": float(scalar_arm)}
    return RestartContext(
        phase=phase,
        N=N,
        setting=setting,
        method=method,
        optimizer=optimizer,
        problem_master_seed=master_seed,
        depth=depth,
        hyperparameter_arm=arm_domain,
        lean_execution_plan_hash=LEAN_EXECUTION_PLAN_SHA256,
    )


def _initializers_record(
    context: RestartContext,
    layout: Sequence[tuple[str, int]],
) -> tuple[Any, list[dict[str, Any]]]:
    initializers = generate_three_random_initializers(context, layout)
    validate_restart_batch(
        initializers,
        expected_context=context,
        expected_layout=layout,
    )
    rows: list[dict[str, Any]] = []
    for initializer in initializers:
        reconstructed = make_random_initializer(
            context,
            restart_index=initializer.restart_index,
            layout=layout,
        )
        byte_equal = _same_float64_bytes(
            initializer.parameters, reconstructed.parameters
        )
        if not byte_equal:
            raise RunnerError("checkpoint-zero vector failed KDF reconstruction")
        rows.append(
            {
                "restart_index": initializer.restart_index,
                "role": initializer.role,
                "supplied_initializer": initializer.supplied_initializer,
                "layout": [list(item) for item in initializer.layout],
                "domain": initializer.domain_metadata,
                "parameter_sha256": initializer.parameter_hash,
                "reconstructed_parameter_sha256": (
                    reconstructed.parameter_hash
                ),
                "byte_equal_to_reconstruction": byte_equal,
                "parameters_float_hex": _float_hex(initializer.parameters),
            }
        )
    return initializers, rows


def _protocol_record(
    result: ProtocolResult,
    initializer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    restart_rows: list[dict[str, Any]] = []
    for restart, initializer_row in zip(result.restarts, initializer_rows):
        if not _same_float64_bytes(
            restart.checkpoints[0].parameters,
            restart.initializer.parameters,
        ):
            raise RunnerError(
                "actual checkpoint zero differs from the KDF initializer"
            )
        restart_rows.append(
            {
                **initializer_row,
                "actual_checkpoint_zero_parameter_sha256": parameter_sha256(
                    restart.checkpoints[0].parameters
                ),
                "checkpoint_zero_byte_equal_to_initializer": True,
                "initial_energy": restart.initial_energy,
                "best_checkpoint": restart.best_checkpoint,
                "best_energy": restart.best_energy,
                "terminal_energy": restart.terminal_energy,
                "executed_updates": restart.executed_updates,
                "calls": restart.calls.to_contract_schema(),
                "metadata": dict(restart.metadata),
                "checkpoints": [
                    {
                        "index": checkpoint.index,
                        "energy": checkpoint.energy,
                        "parameter_sha256": parameter_sha256(
                            checkpoint.parameters
                        ),
                        "parameters_float_hex": _float_hex(
                            checkpoint.parameters
                        ),
                    }
                    for checkpoint in restart.checkpoints
                ],
            }
        )
    return {
        "optimizer": result.optimizer,
        "restart_count": len(result.restarts),
        "restarts": restart_rows,
        "selected_restart": result.selected_restart,
        "selected_checkpoint": result.selected_checkpoint,
        "selected_energy": result.selected_energy,
        "selected_replay_energy": result.selected_replay_energy,
        "selected_parameter_sha256": parameter_sha256(
            result.selected_parameters
        ),
        "selected_parameters_float_hex": _float_hex(
            result.selected_parameters
        ),
        "calls": result.calls.to_contract_schema(),
    }


def _run_optimizer(
    adapter: TorchMethodAdapter,
    context: RestartContext,
    settings: AdamSettings | SPSASettings,
) -> tuple[ProtocolResult, dict[str, Any]]:
    layout = parameter_layout_for_method(adapter.method, adapter.depth)
    initializers, initializer_rows = _initializers_record(context, layout)
    if context.optimizer == "Adam":
        if not isinstance(settings, AdamSettings):
            raise RunnerError("Adam context requires AdamSettings")
        result = run_adam_protocol(
            adapter.objective,
            adapter.value_and_gradient,
            initializers,
            expected_context=context,
            settings=settings,
            spectral_range=1.0,
        )
    else:
        if not isinstance(settings, SPSASettings):
            raise RunnerError("SPSA context requires SPSASettings")
        result = run_spsa_protocol(
            adapter.objective,
            initializers,
            expected_context=context,
            settings=settings,
        )
    adapter.validate_optimizer_accounting(result.calls)
    return result, _protocol_record(result, initializer_rows)


def _measure(
    operation: Callable[[], Any],
    *,
    device: str | torch.device,
) -> tuple[Any, dict[str, Any]]:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    python_before, _ = tracemalloc.get_traced_memory()
    wall_start = time.perf_counter()
    process_start = time.process_time()
    result = operation()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    wall_seconds = time.perf_counter() - wall_start
    process_seconds = time.process_time() - process_start
    _, python_peak = tracemalloc.get_traced_memory()
    if not tracing_before:
        tracemalloc.stop()
    if resolved_device.type == "cuda":
        peak_allocated = int(
            torch.cuda.max_memory_allocated(resolved_device)
        )
        peak_reserved = int(torch.cuda.max_memory_reserved(resolved_device))
    else:
        peak_allocated = 0
        peak_reserved = 0
    return result, {
        "wall_seconds": wall_seconds,
        "process_seconds": process_seconds,
        "peak_device_allocated_bytes": peak_allocated,
        "peak_device_reserved_bytes": peak_reserved,
        "peak_python_tracemalloc_delta_bytes": max(
            0, int(python_peak - python_before)
        ),
        "device": str(resolved_device),
    }


def _timing_memory_record(
    total_start: float,
    **phases: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine phase measurements and expose one job-level peak summary."""

    allocated = max(
        (
            int(phase.get("peak_device_allocated_bytes", 0))
            for phase in phases.values()
        ),
        default=0,
    )
    reserved = max(
        (
            int(phase.get("peak_device_reserved_bytes", 0))
            for phase in phases.values()
        ),
        default=0,
    )
    python_peak = max(
        (
            int(phase.get("peak_python_tracemalloc_delta_bytes", 0))
            for phase in phases.values()
        ),
        default=0,
    )
    return {
        **{name: dict(phase) for name, phase in phases.items()},
        "total_wall_seconds": time.perf_counter() - total_start,
        "peak_device_allocated_bytes": allocated,
        "peak_device_reserved_bytes": reserved,
        "peak_python_tracemalloc_delta_bytes": python_peak,
    }


def _counter_record(counts: ApplicationCounts) -> dict[str, int]:
    return counts.to_dict()


def _replay_record(replay: ReplayDiagnostics) -> dict[str, Any]:
    return {
        "method": replay.method,
        "depth": replay.depth,
        "parameter_sha256": replay.parameter_sha256,
        "state_sha256": replay.state_sha256,
        "canonical_state_sha256": replay.canonical_state_sha256,
        "vector_norm": replay.vector_norm,
        "probability_norm": replay.probability_norm,
        "normalization_error": replay.normalization_error,
        "feasible_mass": replay.feasible_mass,
        "expected_energy": replay.expected_energy,
        "training_hamiltonian": replay.training_hamiltonian,
        "application_counts": _counter_record(replay.application_counts),
    }


def _probability(
    probabilities: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    raw = float(np.sum(probabilities[np.asarray(indices, dtype=np.int64)]))
    return raw, clip_probability(raw)


def _target_metrics(
    instance: BCSTInstance,
    state: np.ndarray,
    targets: TargetSets,
    *,
    method: str,
    depth: int | None,
) -> dict[str, Any]:
    amplitudes = np.asarray(state, dtype=np.complex128)
    probabilities = np.square(np.abs(amplitudes), dtype=np.float64)
    probability_norm = float(probabilities.sum())
    probabilities /= probability_norm
    if method == "stage1_only":
        terminal_ru = instance.terminal_ru("Stage1_only")
    else:
        terminal_ru = instance.terminal_ru(method, depth)
    metrics: dict[str, Any] = {
        "target_hashes": dict(targets.hashes),
        "ground_degeneracy": targets.ground_degeneracy,
        "unique_ground_applicable": targets.ground_degeneracy == 1,
        "terminal_RU": terminal_ru,
        "targets": {},
    }
    for label, indices in (
        ("best2", targets.best2),
        ("best8", targets.best8),
        ("ground", targets.ground),
    ):
        raw, probability = _probability(probabilities, indices)
        repetitions = rts99(probability)
        cost = total_sampling_cost(probability, terminal_ru)
        metrics["targets"][label] = {
            "configuration_count": int(indices.size),
            "raw_probability": raw,
            "probability": probability,
            "RTS99": repetitions,
            "total_logical_cost": cost,
        }
    metrics["primary_target"] = "best2"
    metrics["primary_RTS99"] = metrics["targets"]["best2"]["RTS99"]
    metrics["primary_total_logical_cost"] = metrics["targets"]["best2"][
        "total_logical_cost"
    ]
    return metrics


def _optimizer_settings(
    optimizer: str,
    scalar_arm: float,
    budget: int | None,
) -> AdamSettings | SPSASettings:
    if optimizer == "Adam":
        if scalar_arm not in ADAM_ARMS:
            raise RunnerError(f"Adam arm must be one of {ADAM_ARMS}")
        resolved_budget = ADAM_BASE_STEPS if budget is None else int(budget)
        if resolved_budget not in {ADAM_BASE_STEPS, 2 * ADAM_BASE_STEPS}:
            raise RunnerError("Adam budget must be the base or doubled budget")
        return standard_adam_settings(scalar_arm, budget=resolved_budget)
    if optimizer == "SPSA":
        if scalar_arm not in SPSA_ARMS:
            raise RunnerError(f"SPSA arm must be one of {SPSA_ARMS}")
        resolved_budget = SPSA_BASE_UPDATES if budget is None else int(budget)
        if resolved_budget not in {SPSA_BASE_UPDATES, 2 * SPSA_BASE_UPDATES}:
            raise RunnerError("SPSA budget must be the base or doubled budget")
        return standard_spsa_settings(scalar_arm, budget=resolved_budget)
    raise RunnerError("optimizer must be Adam or SPSA")


def _prepare_output_paths(
    output_dir: Path | str,
    state_name: str | None,
    record_name: str,
    *,
    overwrite: bool,
) -> tuple[Path | None, Path]:
    root = Path(output_dir)
    state_path = root / state_name if state_name is not None else None
    record_path = root / record_name
    paths = [record_path] + ([] if state_path is None else [state_path])
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise RunnerError(
                "refusing to overwrite existing artifacts: "
                + ", ".join(existing)
            )
    root.mkdir(parents=True, exist_ok=True)
    return state_path, record_path


def generate_stage1_artifact(
    N: int,
    output_dir: Path | str,
    *,
    device: str | torch.device = "cuda",
    activation_checkpointing: bool = True,
    overwrite: bool = False,
    settings_override: AdamSettings | None = None,
    minimum_feasible_mass: float = 0.5,
) -> dict[str, Any]:
    """Generate and atomically serialize the single frozen Stage-1 state."""

    verify_lean_execution_plan()
    N = int(N)
    if N not in STAGE1_SEEDS:
        raise RunnerError("Stage-1 N must be 25 or 30")
    state_path, record_path = _prepare_output_paths(
        output_dir,
        "stage1_state.npy",
        "stage1_metadata.json",
        overwrite=overwrite,
    )
    assert state_path is not None
    total_start = time.perf_counter()
    seed = STAGE1_SEEDS[N]
    reference_setting = "d4_1of2_r6_1of16"
    instance = _build_instance(N, seed, reference_setting)
    adapter = build_method_adapter(
        instance,
        "stage1",
        12,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )
    settings = (
        stage1_adam_settings()
        if settings_override is None
        else settings_override
    )
    if not isinstance(settings, AdamSettings):
        raise RunnerError("Stage-1 settings override must be AdamSettings")
    context = _make_context(
        phase="stage1_generation",
        N=N,
        setting={"role": "stage1_conflict_only"},
        method="stage1",
        optimizer="Adam",
        master_seed=seed,
        depth=12,
        scalar_arm=settings.learning_rate,
    )

    adapter.reset_counters()
    zero, zero_measurement = _measure(
        lambda: evaluate_zero_angle_diagnostic(
            adapter.objective,
            parameter_layout_for_method("stage1", 12),
        ),
        device=adapter.device,
    )
    zero_counters = adapter.counters
    adapter.reset_counters()

    (result, protocol), optimizer_measurement = _measure(
        lambda: _run_optimizer(adapter, context, settings),
        device=adapter.device,
    )
    optimizer_counters = adapter.counters
    adapter.reset_counters()
    replay, replay_measurement = _measure(
        lambda: adapter.replay(result.selected_parameters),
        device=adapter.device,
    )
    if replay.feasible_mass < float(minimum_feasible_mass):
        raise RunnerError(
            "Stage-1 feasible mass is below the required minimum: "
            f"{replay.feasible_mass} < {minimum_feasible_mass}"
        )
    canonical_state = _canonical_state(replay.state)
    amplitude_hash = _amplitude_sha256(canonical_state)
    state_file_hash = atomic_write_npy(
        state_path, canonical_state, overwrite=overwrite
    )
    record: dict[str, Any] = {
        "schema": STAGE1_SCHEMA,
        "status": "complete",
        "runner_schema": RUNNER_SCHEMA,
        "lean_execution_plan_sha256": LEAN_EXECUTION_PLAN_SHA256,
        "N": N,
        "stage1_optimizer_seed": seed,
        "depth": 12,
        "optimizer": "Adam",
        "settings": settings.to_contract_schema(),
        "production_settings": settings_override is None,
        "restart_policy": {
            "count": 3,
            "roles": ["independent_random"] * 3,
            "continuation": False,
            "supplied_initializer_count": 0,
        },
        "instance_hashes": dict(instance.hashes),
        "zero_angle_diagnostic": {
            "role": zero.role,
            "optimizer_eligible": zero.optimizer_eligible,
            "energy": zero.energy,
            "parameter_count": zero.parameter_count,
            "parameters_float_hex": _float_hex(zero.parameters),
            "calls": zero.objective_calls,
            "application_counts": _counter_record(zero_counters),
        },
        "optimization": protocol,
        "optimizer_application_counts": _counter_record(
            optimizer_counters
        ),
        "selected_state_replay": _replay_record(replay),
        "state": {
            "path": state_path.name,
            "dtype": "<c16",
            "shape": [instance.dimension],
            "basis_order": (
                "lexicographic Cartesian local pairs; last block fastest"
            ),
            "canonical_global_phase": True,
            "amplitude_sha256": amplitude_hash,
            "npy_file_sha256": state_file_hash,
        },
        "health": {
            "minimum_feasible_mass": float(minimum_feasible_mass),
            "feasible_mass_pass": True,
            "normalization_error_at_most_1e-10": (
                replay.normalization_error <= 1e-10
            ),
        },
        "timing_memory": _timing_memory_record(
            total_start,
            zero_angle_diagnostic=zero_measurement,
            optimizer=optimizer_measurement,
            selected_replay=replay_measurement,
        ),
        "device_configuration": asdict(adapter.device_configuration),
    }
    record_hash = atomic_write_json(
        record_path, record, overwrite=overwrite
    )
    return {
        "record": record,
        "record_path": str(record_path),
        "record_sha256": record_hash,
        "state_path": str(state_path),
        "state_npy_sha256": state_file_hash,
        "amplitude_sha256": amplitude_hash,
    }


def load_stage1_artifact(
    state_path: Path | str,
    metadata_path: Path | str,
    *,
    expected_N: int,
    expected_dimension: int,
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Load and verify a frozen canonical Stage-1 state byte-for-byte."""

    state_source = Path(state_path)
    metadata_source = Path(metadata_path)
    with open(metadata_source, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema") != STAGE1_SCHEMA:
        raise RunnerError("Stage-1 metadata schema is invalid")
    if metadata.get("lean_execution_plan_sha256") != LEAN_EXECUTION_PLAN_SHA256:
        raise RunnerError("Stage-1 metadata uses another lean execution plan")
    if int(metadata.get("N", -1)) != int(expected_N):
        raise RunnerError("Stage-1 state N does not match the job N")
    state = np.load(state_source, allow_pickle=False)
    if state.dtype != np.dtype("<c16") or state.shape != (
        int(expected_dimension),
    ):
        raise RunnerError("Stage-1 state has the wrong dtype or shape")
    state = np.ascontiguousarray(state.astype("<c16", copy=False))
    if not np.all(np.isfinite(state.real)) or not np.all(
        np.isfinite(state.imag)
    ):
        raise RunnerError("Stage-1 state contains nonfinite amplitudes")
    norm_error = abs(float(np.vdot(state, state).real) - 1.0)
    if norm_error > 1e-10:
        raise RunnerError("Stage-1 state normalization check failed")
    amplitude_hash = _amplitude_sha256(state)
    expected_hash = metadata.get("state", {}).get("amplitude_sha256")
    if amplitude_hash != expected_hash:
        raise RunnerError("Stage-1 amplitude SHA-256 mismatch")
    file_hash = sha256_file(state_source)
    expected_file_hash = metadata.get("state", {}).get("npy_file_sha256")
    if file_hash != expected_file_hash:
        raise RunnerError("Stage-1 NPY file SHA-256 mismatch")
    state.setflags(write=False)
    return state, metadata, amplitude_hash


def run_method_job(
    *,
    N: int,
    master_seed: int,
    setting_label: str,
    method: str,
    optimizer: str,
    depth: int,
    phase: str,
    scalar_arm: float,
    output_dir: Path | str,
    budget: int | None = None,
    stage1_state_path: Path | str | None = None,
    stage1_metadata_path: Path | str | None = None,
    post_target_metrics: bool = False,
    device: str | torch.device = "cuda",
    activation_checkpointing: bool = True,
    overwrite: bool = False,
    settings_override: AdamSettings | SPSASettings | None = None,
) -> dict[str, Any]:
    """Run one complete optimized method cell and atomically save its record."""

    verify_lean_execution_plan()
    if method not in METHODS:
        raise RunnerError(f"method must be one of {METHODS}")
    if phase not in PHASE_TAGS:
        raise RunnerError(f"unknown phase {phase!r}")
    if method in PHI_METHODS:
        if stage1_state_path is None or stage1_metadata_path is None:
            raise RunnerError(f"{method} requires Stage-1 state and metadata")
    elif stage1_state_path is not None or stage1_metadata_path is not None:
        raise RunnerError("native methods prohibit every Stage-1 input")
    state_path, record_path = _prepare_output_paths(
        output_dir,
        "selected_state.npy",
        "job_record.json",
        overwrite=overwrite,
    )
    assert state_path is not None
    total_start = time.perf_counter()
    instance = _build_instance(N, master_seed, setting_label)
    phi: np.ndarray | None = None
    stage1_hash: str | None = None
    stage1_metadata_hash: str | None = None
    if method in PHI_METHODS:
        assert stage1_state_path is not None
        assert stage1_metadata_path is not None
        phi, _stage1_metadata, stage1_hash = load_stage1_artifact(
            stage1_state_path,
            stage1_metadata_path,
            expected_N=N,
            expected_dimension=instance.dimension,
        )
        stage1_metadata_hash = sha256_file(stage1_metadata_path)

    settings = (
        _optimizer_settings(optimizer, scalar_arm, budget)
        if settings_override is None
        else settings_override
    )
    if optimizer == "Adam" and not isinstance(settings, AdamSettings):
        raise RunnerError("Adam job settings must be AdamSettings")
    if optimizer == "SPSA" and not isinstance(settings, SPSASettings):
        raise RunnerError("SPSA job settings must be SPSASettings")
    adapter = build_method_adapter(
        instance,
        method,
        depth,
        phi=phi,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )
    if method in PHI_METHODS:
        if adapter.phi_record is None:
            raise RunnerError("phi metadata is unexpectedly absent")
        if adapter.phi_record.raw_sha256 != stage1_hash:
            raise RunnerError(
                "LP/Warm adapter did not load the exact Stage-1 amplitudes"
            )
    context = _make_context(
        phase=phase,
        N=N,
        setting=_setting_record(setting_label),
        method=method,
        optimizer=optimizer,
        master_seed=master_seed,
        depth=depth,
        scalar_arm=scalar_arm,
    )

    adapter.reset_counters()
    zero, zero_measurement = _measure(
        lambda: evaluate_zero_angle_diagnostic(
            adapter.objective,
            parameter_layout_for_method(method, depth),
        ),
        device=adapter.device,
    )
    zero_counters = adapter.counters
    adapter.reset_counters()
    (result, protocol), optimizer_measurement = _measure(
        lambda: _run_optimizer(adapter, context, settings),
        device=adapter.device,
    )
    optimizer_counters = adapter.counters
    adapter.reset_counters()
    replay, replay_measurement = _measure(
        lambda: adapter.replay(result.selected_parameters),
        device=adapter.device,
    )
    canonical_state = _canonical_state(replay.state)
    selected_amplitude_hash = _amplitude_sha256(canonical_state)
    state_file_hash = atomic_write_npy(
        state_path, canonical_state, overwrite=overwrite
    )

    metrics: dict[str, Any] | None = None
    metrics_measurement: dict[str, Any] | None = None
    if post_target_metrics:
        # This occurs strictly after target-blind optimization and replay.
        def calculate_target_metrics() -> dict[str, Any]:
            targets = derive_targets(instance)
            return _target_metrics(
                instance,
                canonical_state,
                targets,
                method=method,
                depth=depth,
            )

        metrics, metrics_measurement = _measure(
            calculate_target_metrics,
            device=adapter.device,
        )
    phase_measurements = {
        "zero_angle_diagnostic": zero_measurement,
        "optimizer": optimizer_measurement,
        "selected_replay": replay_measurement,
    }
    if metrics_measurement is not None:
        phase_measurements["post_optimization_target_metrics"] = (
            metrics_measurement
        )
    record: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "status": "complete",
        "runner_schema": RUNNER_SCHEMA,
        "lean_execution_plan_sha256": LEAN_EXECUTION_PLAN_SHA256,
        "N": int(N),
        "problem_master_seed": int(master_seed),
        "setting": _setting_record(setting_label),
        "phase": phase,
        "method": method,
        "optimizer": optimizer,
        "depth": int(depth),
        "scalar_arm": float(scalar_arm),
        "settings": settings.to_contract_schema(),
        "production_settings": settings_override is None,
        "restart_policy": {
            "count": 3,
            "roles": ["independent_random"] * 3,
            "continuation": False,
            "supplied_initializer_count": 0,
        },
        "instance_hashes": dict(instance.hashes),
        "stage1": (
            None
            if method in NATIVE_METHODS
            else {
                "amplitude_sha256": stage1_hash,
                "metadata_file_sha256": stage1_metadata_hash,
                "adapter_raw_phi_sha256": adapter.phi_record.raw_sha256
                if adapter.phi_record is not None
                else None,
                "shared_identity_required_for": [
                    "learned_projector",
                    "same_stage1_state_direct",
                    "stage1_only",
                ],
            }
        ),
        "zero_angle_diagnostic": {
            "role": zero.role,
            "optimizer_eligible": zero.optimizer_eligible,
            "energy": zero.energy,
            "parameter_count": zero.parameter_count,
            "parameters_float_hex": _float_hex(zero.parameters),
            "calls": zero.objective_calls,
            "application_counts": _counter_record(zero_counters),
        },
        "optimization": protocol,
        "optimizer_application_counts": _counter_record(
            optimizer_counters
        ),
        "selected_state_replay": _replay_record(replay),
        "selected_state": {
            "path": state_path.name,
            "dtype": "<c16",
            "shape": [instance.dimension],
            "canonical_global_phase": True,
            "amplitude_sha256": selected_amplitude_hash,
            "npy_file_sha256": state_file_hash,
        },
        "post_optimization_target_metrics_enabled": bool(
            post_target_metrics
        ),
        "post_optimization_target_metrics": metrics,
        "timing_memory": _timing_memory_record(
            total_start, **phase_measurements
        ),
        "device_configuration": asdict(adapter.device_configuration),
    }
    record_hash = atomic_write_json(
        record_path, record, overwrite=overwrite
    )
    return {
        "record": record,
        "record_path": str(record_path),
        "record_sha256": record_hash,
        "state_path": str(state_path),
        "state_npy_sha256": state_file_hash,
        "selected_amplitude_sha256": selected_amplitude_hash,
    }


def run_stage1_only_metrics(
    *,
    N: int,
    master_seed: int,
    setting_label: str,
    stage1_state_path: Path | str,
    stage1_metadata_path: Path | str,
    output_dir: Path | str,
    device: str | torch.device = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate one frozen Stage-1 state on one objective instance."""

    verify_lean_execution_plan()
    _state_path, record_path = _prepare_output_paths(
        output_dir,
        None,
        "stage1_only_record.json",
        overwrite=overwrite,
    )
    total_start = time.perf_counter()
    instance = _build_instance(N, master_seed, setting_label)
    state, _metadata, amplitude_hash = load_stage1_artifact(
        stage1_state_path,
        stage1_metadata_path,
        expected_N=N,
        expected_dimension=instance.dimension,
    )
    configuration = configure_deterministic_device(device)
    probability_norm = float(np.vdot(state, state).real)
    probabilities = np.square(np.abs(state), dtype=np.float64)
    normalized = probabilities / probabilities.sum()
    feasible_mass = float(normalized[instance.feasible_mask].sum())
    expected_conflict_energy = float(np.dot(normalized, instance.H_C))
    def calculate_target_metrics() -> dict[str, Any]:
        targets = derive_targets(instance)
        return _target_metrics(
            instance,
            state,
            targets,
            method="stage1_only",
            depth=None,
        )

    metrics, metrics_measurement = _measure(
        calculate_target_metrics,
        device=configuration.device,
    )
    record: dict[str, Any] = {
        "schema": STAGE1_ONLY_SCHEMA,
        "status": "complete",
        "runner_schema": RUNNER_SCHEMA,
        "lean_execution_plan_sha256": LEAN_EXECUTION_PLAN_SHA256,
        "N": int(N),
        "problem_master_seed": int(master_seed),
        "setting": _setting_record(setting_label),
        "method": "stage1_only",
        "optimizer": "not_applicable",
        "stage1_amplitude_sha256": amplitude_hash,
        "stage1_metadata_file_sha256": sha256_file(stage1_metadata_path),
        "probability_norm": probability_norm,
        "normalization_error": abs(probability_norm - 1.0),
        "feasible_mass": feasible_mass,
        "expected_H_C": expected_conflict_energy,
        "target_metrics": metrics,
        "instance_hashes": dict(instance.hashes),
        "timing_memory": _timing_memory_record(
            total_start,
            target_metrics=metrics_measurement,
        ),
        "device_configuration": asdict(configuration),
    }
    record_hash = atomic_write_json(
        record_path, record, overwrite=overwrite
    )
    return {
        "record": record,
        "record_path": str(record_path),
        "record_sha256": record_hash,
        "stage1_amplitude_sha256": amplitude_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bcst_v2.production_runner",
        description="Run one lean BCST v2 production artifact.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser(
        "stage1", help="generate the frozen Stage-1 state for one N"
    )
    stage1.add_argument("--N", type=int, choices=(30, 25), required=True)
    stage1.add_argument("--output-dir", type=Path, required=True)
    stage1.add_argument("--device", default="cuda")
    stage1.add_argument(
        "--no-activation-checkpointing",
        action="store_true",
    )
    stage1.add_argument("--overwrite", action="store_true")

    job = subparsers.add_parser(
        "job", help="run one method/optimizer/depth/seed/setting cell"
    )
    job.add_argument("--N", type=int, choices=(30, 25), required=True)
    job.add_argument("--master-seed", type=int, required=True)
    job.add_argument(
        "--setting", choices=tuple(SETTINGS), required=True
    )
    job.add_argument("--method", choices=METHODS, required=True)
    job.add_argument("--optimizer", choices=("Adam", "SPSA"), required=True)
    job.add_argument("--depth", type=int, required=True)
    job.add_argument("--phase", choices=PHASE_TAGS, required=True)
    job.add_argument("--arm", type=float, required=True)
    job.add_argument("--budget", type=int)
    job.add_argument("--stage1-state", type=Path)
    job.add_argument("--stage1-metadata", type=Path)
    job.add_argument("--post-target-metrics", action="store_true")
    job.add_argument("--output-dir", type=Path, required=True)
    job.add_argument("--device", default="cuda")
    job.add_argument(
        "--no-activation-checkpointing",
        action="store_true",
    )
    job.add_argument("--overwrite", action="store_true")

    stage1_only = subparsers.add_parser(
        "stage1-only", help="evaluate Stage-1-only metrics for one instance"
    )
    stage1_only.add_argument(
        "--N", type=int, choices=(30, 25), required=True
    )
    stage1_only.add_argument("--master-seed", type=int, required=True)
    stage1_only.add_argument(
        "--setting", choices=tuple(SETTINGS), required=True
    )
    stage1_only.add_argument("--stage1-state", type=Path, required=True)
    stage1_only.add_argument(
        "--stage1-metadata", type=Path, required=True
    )
    stage1_only.add_argument("--output-dir", type=Path, required=True)
    stage1_only.add_argument("--device", default="cuda")
    stage1_only.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stage1":
        result = generate_stage1_artifact(
            args.N,
            args.output_dir,
            device=args.device,
            activation_checkpointing=not args.no_activation_checkpointing,
            overwrite=args.overwrite,
        )
    elif args.command == "job":
        result = run_method_job(
            N=args.N,
            master_seed=args.master_seed,
            setting_label=args.setting,
            method=args.method,
            optimizer=args.optimizer,
            depth=args.depth,
            phase=args.phase,
            scalar_arm=args.arm,
            budget=args.budget,
            stage1_state_path=args.stage1_state,
            stage1_metadata_path=args.stage1_metadata,
            post_target_metrics=args.post_target_metrics,
            output_dir=args.output_dir,
            device=args.device,
            activation_checkpointing=not args.no_activation_checkpointing,
            overwrite=args.overwrite,
        )
    else:
        result = run_stage1_only_metrics(
            N=args.N,
            master_seed=args.master_seed,
            setting_label=args.setting,
            stage1_state_path=args.stage1_state,
            stage1_metadata_path=args.stage1_metadata,
            output_dir=args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
    summary = {
        key: value
        for key, value in result.items()
        if key != "record"
    }
    print(canonical_json_bytes(summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ADAM_ARMS",
    "JOB_SCHEMA",
    "LEAN_EXECUTION_PLAN_SHA256",
    "LEAN_PLAN_PATH",
    "METHODS",
    "RUNNER_SCHEMA",
    "RunnerError",
    "SETTINGS",
    "SPSA_ARMS",
    "STAGE1_ONLY_SCHEMA",
    "STAGE1_SCHEMA",
    "atomic_write_json",
    "atomic_write_npy",
    "build_parser",
    "canonical_json_bytes",
    "generate_stage1_artifact",
    "load_stage1_artifact",
    "main",
    "run_method_job",
    "run_stage1_only_metrics",
    "sha256_file",
    "verify_lean_execution_plan",
]
