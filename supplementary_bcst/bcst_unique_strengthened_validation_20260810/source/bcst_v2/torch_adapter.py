"""Production Torch adapter for target-blind BCST v2 optimization.

The instance generator is NumPy-only and the optimizer protocol consumes
NumPy callbacks.  This module is the narrow bridge between them:

* reject any instance that already exposes target identities or target hashes;
* copy the training diagonals into :class:`~bcst_v2.dynamics.ReducedProblem`;
* configure deterministic float64/complex128 Torch execution;
* validate and bind the frozen Stage-1 state for LP and Warm-XY;
* expose NumPy energy and value-plus-exact-gradient callbacks; and
* retain independently checkable operator-application counters and replay
  diagnostics.

No target derivation or target probability is imported into this module.
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Iterator, Literal

import numpy as np
import torch

from .dynamics import (
    Method,
    ReducedDynamics,
    ReducedProblem,
    canonical_state_sha256,
    parameter_count,
)
from .instance_core import BCSTInstance
from .optimizer_protocol import OptimizerCallCounts, parameter_sha256


SUPPORTED_METHODS: tuple[Method, ...] = (
    "stage1",
    "learned_projector",
    "ordinary_native_direct",
    "decoupled_native_direct",
    "same_stage1_state_direct",
)
PHI_METHODS: frozenset[Method] = frozenset(
    {"learned_projector", "same_stage1_state_direct"}
)
NATIVE_INITIAL_METHODS: frozenset[Method] = frozenset(
    {"stage1", "ordinary_native_direct", "decoupled_native_direct"}
)
STATE_NORM_TOLERANCE = 1e-10
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class TorchAdapterError(ValueError):
    """Raised when an adapter input violates the frozen execution boundary."""


class TargetExposureError(TorchAdapterError):
    """Raised when target-aware data reaches a target-blind runner."""


def _plain_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TorchAdapterError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise TorchAdapterError(f"{name} must be an integer") from exc
    if result < 1:
        raise TorchAdapterError(f"{name} must be positive")
    return result


def _validated_method(method: str) -> Method:
    if method not in SUPPORTED_METHODS:
        raise TorchAdapterError(f"unsupported dynamics method {method!r}")
    return method  # type: ignore[return-value]


def _ensure_target_blind(instance: BCSTInstance) -> None:
    if not isinstance(instance, BCSTInstance):
        raise TorchAdapterError("instance must be a BCSTInstance")
    if instance.targets is not None:
        raise TargetExposureError(
            "target-aware BCSTInstance cannot enter target-blind optimization"
        )
    leaked_keys = tuple(
        sorted(key for key in instance.hashes if "target" in key.lower())
    )
    if leaked_keys:
        raise TargetExposureError(
            "target hashes cannot enter target-blind optimization: "
            + ", ".join(leaked_keys)
        )


@dataclass(frozen=True)
class DeterministicDevice:
    """Recorded deterministic-device settings applied before tensor creation."""

    device: str
    torch_version: str
    deterministic_algorithms: bool
    cublas_workspace_config: str
    tf32_matmul_enabled: bool
    tf32_cudnn_enabled: bool
    cudnn_benchmark: bool
    real_dtype: str = "torch.float64"
    complex_dtype: str = "torch.complex128"


def configure_deterministic_device(
    device: str | torch.device = "cpu",
) -> DeterministicDevice:
    """Apply and report the contract's deterministic Torch settings."""

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise TorchAdapterError("only CPU validation or CUDA production is supported")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise TorchAdapterError("CUDA was requested but is not available")

    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.are_deterministic_algorithms_enabled():
        raise TorchAdapterError("Torch deterministic algorithms did not enable")

    return DeterministicDevice(
        device=str(resolved),
        torch_version=str(torch.__version__),
        deterministic_algorithms=True,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        tf32_matmul_enabled=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn_enabled=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
    )


def _float64_copy(values: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype != np.float64 or source.ndim != 1:
        raise TorchAdapterError(f"{name} must be a one-dimensional float64 array")
    if not np.all(np.isfinite(source)):
        raise TorchAdapterError(f"{name} contains nonfinite values")
    return np.array(source, dtype=np.float64, order="C", copy=True)


def to_reduced_problem(instance: BCSTInstance) -> ReducedProblem:
    """Convert one target-blind instance to an independent training problem."""

    _ensure_target_blind(instance)
    feasible = np.asarray(instance.feasible_mask)
    if feasible.dtype != np.bool_ or feasible.shape != (instance.dimension,):
        raise TorchAdapterError("instance feasible mask has the wrong shape or dtype")
    return ReducedProblem(
        layout=(instance.K,) * 5,
        h_c=_float64_copy(instance.H_C, "H_C"),
        h_f=_float64_copy(instance.H_F, "H_F"),
        h_d=_float64_copy(instance.H_D, "H_D"),
        h_d_obj=_float64_copy(instance.H_D_obj, "H_D_obj"),
        h_d_conf=_float64_copy(instance.H_D_conf, "H_D_conf"),
        feasible_mask=np.array(feasible, dtype=np.bool_, order="C", copy=True),
    )


def build_reduced_dynamics(
    instance: BCSTInstance,
    *,
    device: str | torch.device = "cpu",
) -> ReducedDynamics:
    """Build plain deterministic dynamics from a target-blind instance."""

    problem = to_reduced_problem(instance)
    configuration = configure_deterministic_device(device)
    dynamics = ReducedDynamics(problem, device=configuration.device)
    if dynamics.real_dtype != torch.float64:
        raise TorchAdapterError("dynamics real dtype is not float64")
    if dynamics.complex_dtype != torch.complex128:
        raise TorchAdapterError("dynamics complex dtype is not complex128")
    for values in (
        problem.h_c,
        problem.h_f,
        problem.h_d,
        problem.h_d_obj,
        problem.h_d_conf,
        problem.feasible_mask,
    ):
        values.setflags(write=False)
    return dynamics


@dataclass(frozen=True)
class PhiRecord:
    """Integrity and normalization metadata for one frozen Stage-1 state."""

    raw_sha256: str
    canonical_state_sha256: str
    effective_state_sha256: str
    probability_norm: float
    normalization_error: float
    amplitude_count: int
    dtype: str = "complex128"


def _prepare_phi(
    phi: np.ndarray | torch.Tensor | None,
    *,
    method: Method,
    dimension: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, PhiRecord | None]:
    if method in PHI_METHODS and phi is None:
        raise TorchAdapterError(f"{method} requires the frozen Stage-1 state phi")
    if method in NATIVE_INITIAL_METHODS and phi is not None:
        raise TorchAdapterError(f"{method} prohibits a Stage-1 state phi")
    if phi is None:
        return None, None

    if isinstance(phi, torch.Tensor):
        if phi.requires_grad:
            raise TorchAdapterError("the frozen Stage-1 state cannot require gradients")
        if phi.dtype != torch.complex128:
            raise TorchAdapterError("phi must have dtype complex128")
        source = phi.detach().cpu().numpy()
    else:
        source = np.asarray(phi)
        if source.dtype != np.complex128:
            raise TorchAdapterError("phi must have dtype complex128")
    if source.shape != (dimension,):
        raise TorchAdapterError(
            f"phi shape must be ({dimension},), received {source.shape}"
        )
    if not np.all(np.isfinite(source.real)) or not np.all(np.isfinite(source.imag)):
        raise TorchAdapterError("phi contains nonfinite amplitudes")

    canonical_bytes = np.ascontiguousarray(source.astype("<c16", copy=False))
    probability_norm = float(np.vdot(canonical_bytes, canonical_bytes).real)
    if not math.isfinite(probability_norm) or probability_norm <= 0.0:
        raise TorchAdapterError("phi has invalid probability norm")
    normalization_error = abs(probability_norm - 1.0)
    if normalization_error > STATE_NORM_TOLERANCE:
        raise TorchAdapterError(
            "phi normalization error exceeds "
            f"{STATE_NORM_TOLERANCE:.1e}: {normalization_error}"
        )
    raw_sha256 = hashlib.sha256(
        canonical_bytes.tobytes(order="C")
    ).hexdigest()
    canonical_hash = canonical_state_sha256(canonical_bytes)

    tensor = torch.tensor(
        canonical_bytes,
        dtype=torch.complex128,
        device=device,
    )
    tensor = (tensor / torch.linalg.vector_norm(tensor)).detach()
    effective_hash = canonical_state_sha256(tensor)
    record = PhiRecord(
        raw_sha256=raw_sha256,
        canonical_state_sha256=canonical_hash,
        effective_state_sha256=effective_hash,
        probability_norm=probability_norm,
        normalization_error=normalization_error,
        amplitude_count=dimension,
    )
    return tensor, record


@dataclass(frozen=True)
class ApplicationCounts:
    """Non-overlapping callback and explicit forward-operator counts."""

    energy_function_calls: int = 0
    value_and_gradient_calls: int = 0
    state_replay_calls: int = 0
    exact_gradient_calls: int = 0
    layer_executions: int = 0
    checkpoint_recomputed_layers: int = 0
    h_c_phase_applications: int = 0
    h_f_phase_applications: int = 0
    h_d_phase_applications: int = 0
    h_d_conf_phase_applications: int = 0
    h_d_obj_phase_applications: int = 0
    training_hamiltonian_expectation_applications: int = 0
    xy_mixer_applications: int = 0
    projector_mixer_applications: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TorchAdapterError(
                    f"counter {item.name} must be a nonnegative integer"
                )

    @property
    def objective_evaluations(self) -> int:
        return (
            self.energy_function_calls
            + self.value_and_gradient_calls
            + self.state_replay_calls
        )

    @property
    def hamiltonian_phase_applications(self) -> int:
        return (
            self.h_c_phase_applications
            + self.h_f_phase_applications
            + self.h_d_phase_applications
            + self.h_d_conf_phase_applications
            + self.h_d_obj_phase_applications
        )

    @property
    def hamiltonian_applications(self) -> int:
        return (
            self.hamiltonian_phase_applications
            + self.training_hamiltonian_expectation_applications
        )

    @property
    def mixer_applications(self) -> int:
        return self.xy_mixer_applications + self.projector_mixer_applications

    def __sub__(self, earlier: "ApplicationCounts") -> "ApplicationCounts":
        values = {
            item.name: getattr(self, item.name) - getattr(earlier, item.name)
            for item in fields(self)
        }
        return ApplicationCounts(**values)

    def to_dict(self) -> dict[str, int]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result.update(
            {
                "objective_evaluations": self.objective_evaluations,
                "hamiltonian_phase_applications": (
                    self.hamiltonian_phase_applications
                ),
                "hamiltonian_applications": self.hamiltonian_applications,
                "mixer_applications": self.mixer_applications,
            }
        )
        return result


def expected_application_counts(
    method: str,
    depth: int,
    *,
    energy_function_calls: int = 0,
    value_and_gradient_calls: int = 0,
    state_replay_calls: int = 0,
    activation_checkpointing: bool = False,
) -> ApplicationCounts:
    """Algebraically derive every counter from public callback counts."""

    resolved_method = _validated_method(method)
    p = _plain_positive_int(depth, "depth")
    raw_counts = {
        "energy_function_calls": energy_function_calls,
        "value_and_gradient_calls": value_and_gradient_calls,
        "state_replay_calls": state_replay_calls,
    }
    for name, value in raw_counts.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TorchAdapterError(f"{name} must be an integer")
        raw_counts[name] = int(value)
        if raw_counts[name] < 0:
            raise TorchAdapterError(f"{name} must be nonnegative")
    if not isinstance(activation_checkpointing, (bool, np.bool_)):
        raise TorchAdapterError("activation_checkpointing must be bool")

    simulations = sum(raw_counts.values())
    recomputed_layers = (
        p * raw_counts["value_and_gradient_calls"]
        if activation_checkpointing
        else 0
    )
    layer_executions = p * simulations + recomputed_layers
    values: dict[str, int] = {
        **raw_counts,
        "exact_gradient_calls": raw_counts["value_and_gradient_calls"],
        "layer_executions": layer_executions,
        "checkpoint_recomputed_layers": recomputed_layers,
        "training_hamiltonian_expectation_applications": simulations,
    }
    if resolved_method == "stage1":
        values["h_c_phase_applications"] = layer_executions
        values["xy_mixer_applications"] = layer_executions
    elif resolved_method == "learned_projector":
        values["h_f_phase_applications"] = layer_executions
        values["projector_mixer_applications"] = layer_executions
    elif resolved_method in {
        "ordinary_native_direct",
        "same_stage1_state_direct",
    }:
        values["h_d_phase_applications"] = layer_executions
        values["xy_mixer_applications"] = layer_executions
    else:
        values["h_d_conf_phase_applications"] = layer_executions
        values["h_d_obj_phase_applications"] = layer_executions
        values["xy_mixer_applications"] = layer_executions
    return ApplicationCounts(**values)


class _ApplicationLedger:
    _FIELD_NAMES = tuple(item.name for item in fields(ApplicationCounts))

    def __init__(self) -> None:
        self._values = {name: 0 for name in self._FIELD_NAMES}
        self._phase = "idle"
        self._lock = threading.RLock()

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @contextmanager
    def execution_phase(self, phase: str) -> Iterator[None]:
        with self._lock:
            if self._phase != "idle":
                raise TorchAdapterError(
                    "Torch callback execution must be sequential and non-reentrant"
                )
            self._phase = phase
        try:
            yield
        finally:
            with self._lock:
                self._phase = "idle"

    def add(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            if field_name not in self._values:
                raise AssertionError(f"unknown counter field {field_name}")
            self._values[field_name] += int(amount)

    def snapshot(self) -> ApplicationCounts:
        with self._lock:
            return ApplicationCounts(**dict(self._values))

    def reset(self) -> None:
        with self._lock:
            if self._phase != "idle":
                raise TorchAdapterError("cannot reset counters during execution")
            self._values = {name: 0 for name in self._FIELD_NAMES}


class _CountingReducedDynamics(ReducedDynamics):
    def __init__(
        self,
        problem: ReducedProblem,
        *,
        device: str | torch.device,
        ledger: _ApplicationLedger,
    ) -> None:
        self._ledger = ledger
        super().__init__(problem, device=device)

    def _apply_layer(
        self,
        method: Method,
        state: torch.Tensor,
        parameter_parts: tuple[torch.Tensor, ...],
        layer: int,
        phi: torch.Tensor | None,
    ) -> torch.Tensor:
        self._ledger.add("layer_executions")
        if self._ledger.phase == "gradient_backward":
            self._ledger.add("checkpoint_recomputed_layers")
        if method == "stage1":
            self._ledger.add("h_c_phase_applications")
            self._ledger.add("xy_mixer_applications")
        elif method == "learned_projector":
            self._ledger.add("h_f_phase_applications")
            self._ledger.add("projector_mixer_applications")
        elif method in {
            "ordinary_native_direct",
            "same_stage1_state_direct",
        }:
            self._ledger.add("h_d_phase_applications")
            self._ledger.add("xy_mixer_applications")
        elif method == "decoupled_native_direct":
            self._ledger.add("h_d_conf_phase_applications")
            self._ledger.add("h_d_obj_phase_applications")
            self._ledger.add("xy_mixer_applications")
        else:  # pragma: no cover - ReducedDynamics rejects this as well.
            raise TorchAdapterError(f"unsupported dynamics method {method!r}")
        return super()._apply_layer(
            method, state, parameter_parts, layer, phi
        )

    def expected_energy(self, method: Method, state: torch.Tensor) -> torch.Tensor:
        self._ledger.add("training_hamiltonian_expectation_applications")
        return super().expected_energy(method, state)


@dataclass(frozen=True)
class ReplayDiagnostics:
    """Deterministic state and training-only diagnostics from one replay."""

    method: str
    depth: int
    parameter_sha256: str
    state_sha256: str
    canonical_state_sha256: str
    state: np.ndarray
    vector_norm: float
    probability_norm: float
    normalization_error: float
    feasible_mass: float
    expected_energy: float
    training_hamiltonian: str
    application_counts: ApplicationCounts

    def __post_init__(self) -> None:
        state = np.asarray(self.state)
        if state.dtype != np.complex128 or state.ndim != 1:
            raise TorchAdapterError("replay state must be one-dimensional complex128")
        copied = np.array(state, dtype=np.complex128, order="C", copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "state", copied)


def _training_hamiltonian_label(method: Method) -> str:
    if method == "stage1":
        return "H_C"
    if method == "learned_projector":
        return "H_F"
    return "H_D"


class TorchMethodAdapter:
    """Bound callbacks, replay, and counters for one method/depth/phi tuple."""

    def __init__(
        self,
        instance: BCSTInstance,
        method: str,
        depth: int,
        *,
        phi: np.ndarray | torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        activation_checkpointing: bool = False,
    ) -> None:
        _ensure_target_blind(instance)
        self.instance = instance
        self.method = _validated_method(method)
        self.depth = _plain_positive_int(depth, "depth")
        if not isinstance(activation_checkpointing, (bool, np.bool_)):
            raise TorchAdapterError("activation_checkpointing must be bool")
        self.activation_checkpointing = bool(activation_checkpointing)
        self.problem = to_reduced_problem(instance)
        self.device_configuration = configure_deterministic_device(device)
        self.device = torch.device(self.device_configuration.device)
        self._ledger = _ApplicationLedger()
        self.dynamics = _CountingReducedDynamics(
            self.problem,
            device=self.device,
            ledger=self._ledger,
        )
        self._phi, self.phi_record = _prepare_phi(
            phi,
            method=self.method,
            dimension=self.problem.dimension,
            device=self.device,
        )
        self.parameter_count = parameter_count(self.method, self.depth)
        if self.dynamics.real_dtype != torch.float64:
            raise TorchAdapterError("adapter dynamics must use float64")
        if self.dynamics.complex_dtype != torch.complex128:
            raise TorchAdapterError("adapter dynamics must use complex128")
        for values in (
            self.problem.h_c,
            self.problem.h_f,
            self.problem.h_d,
            self.problem.h_d_obj,
            self.problem.h_d_conf,
            self.problem.feasible_mask,
        ):
            values.setflags(write=False)

    @property
    def phi(self) -> torch.Tensor | None:
        """Return a detached copy; the bound frozen state cannot be mutated."""

        return None if self._phi is None else self._phi.detach().clone()

    def _parameters(
        self, parameters: np.ndarray, *, requires_grad: bool
    ) -> torch.Tensor:
        values = np.asarray(parameters)
        if values.dtype != np.float64:
            raise TorchAdapterError("callback parameters must have dtype float64")
        if values.shape != (self.parameter_count,):
            raise TorchAdapterError(
                f"expected {self.parameter_count} parameters, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise TorchAdapterError("callback parameters contain nonfinite values")
        return torch.tensor(
            np.ascontiguousarray(values),
            dtype=torch.float64,
            device=self.device,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _finite_energy(energy: torch.Tensor) -> float:
        if energy.dtype != torch.float64 or energy.ndim != 0:
            raise TorchAdapterError("training energy must be a scalar float64")
        result = float(energy.detach().cpu())
        if not math.isfinite(result):
            raise TorchAdapterError("training energy is nonfinite")
        return result

    def objective(self, parameters: np.ndarray) -> float:
        """NumPy-compatible training-energy callback for SPSA and replay."""

        tensor = self._parameters(parameters, requires_grad=False)
        with self._ledger.execution_phase("energy_function"):
            energy, state = self.dynamics.energy_from_parameters(
                self.method,
                tensor,
                depth=self.depth,
                phi=self._phi,
                activation_checkpointing=False,
            )
        if state.dtype != torch.complex128:
            raise TorchAdapterError("evolved state is not complex128")
        result = self._finite_energy(energy)
        self._ledger.add("energy_function_calls")
        return result

    def value_and_gradient(
        self, parameters: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """NumPy-compatible energy and exact reverse-mode gradient callback."""

        tensor = self._parameters(parameters, requires_grad=True)
        with self._ledger.execution_phase("gradient_forward"):
            energy, state = self.dynamics.energy_from_parameters(
                self.method,
                tensor,
                depth=self.depth,
                phi=self._phi,
                activation_checkpointing=self.activation_checkpointing,
            )
        if state.dtype != torch.complex128:
            raise TorchAdapterError("evolved state is not complex128")
        result = self._finite_energy(energy)
        with self._ledger.execution_phase("gradient_backward"):
            (gradient,) = torch.autograd.grad(
                energy,
                (tensor,),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
        if gradient.dtype != torch.float64:
            raise TorchAdapterError("exact gradient is not float64")
        gradient_np = np.array(
            gradient.detach().cpu().numpy(),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if gradient_np.shape != (self.parameter_count,) or not np.all(
            np.isfinite(gradient_np)
        ):
            raise TorchAdapterError("exact gradient has invalid shape or values")
        self._ledger.add("value_and_gradient_calls")
        self._ledger.add("exact_gradient_calls")
        return result, gradient_np

    # Explicit names make the optimizer-facing ABI self-documenting.
    numpy_objective = objective
    numpy_value_and_gradient = value_and_gradient

    def replay(self, parameters: np.ndarray) -> ReplayDiagnostics:
        """Replay one parameter vector and return target-free state diagnostics."""

        before = self.counters
        tensor = self._parameters(parameters, requires_grad=False)
        with self._ledger.execution_phase("state_replay"):
            energy, state = self.dynamics.energy_from_parameters(
                self.method,
                tensor,
                depth=self.depth,
                phi=self._phi,
                activation_checkpointing=False,
            )
        self._ledger.add("state_replay_calls")
        expected_energy = self._finite_energy(energy)
        probabilities = torch.abs(state) ** 2
        probability_norm = float(torch.sum(probabilities).detach().cpu())
        vector_norm = math.sqrt(probability_norm)
        normalization_error = abs(probability_norm - 1.0)
        if not math.isfinite(probability_norm):
            raise TorchAdapterError("replayed state norm is nonfinite")
        if normalization_error > STATE_NORM_TOLERANCE:
            raise TorchAdapterError(
                "replayed state normalization error exceeds "
                f"{STATE_NORM_TOLERANCE:.1e}"
            )
        feasible_mass = float(
            (
                torch.sum(probabilities[self.dynamics.feasible_mask])
                / torch.sum(probabilities)
            )
            .detach()
            .cpu()
        )
        if (
            not math.isfinite(feasible_mass)
            or feasible_mass < -1e-12
            or feasible_mass > 1.0 + 1e-12
        ):
            raise TorchAdapterError("replayed feasible mass is invalid")

        state_np = np.array(
            state.detach().cpu().numpy(),
            dtype=np.complex128,
            order="C",
            copy=True,
        )
        state_bytes = np.ascontiguousarray(
            state_np.astype("<c16", copy=False)
        ).tobytes(order="C")
        after = self.counters
        return ReplayDiagnostics(
            method=self.method,
            depth=self.depth,
            parameter_sha256=parameter_sha256(parameters),
            state_sha256=hashlib.sha256(state_bytes).hexdigest(),
            canonical_state_sha256=canonical_state_sha256(state_np),
            state=state_np,
            vector_norm=vector_norm,
            probability_norm=probability_norm,
            normalization_error=normalization_error,
            feasible_mass=feasible_mass,
            expected_energy=expected_energy,
            training_hamiltonian=_training_hamiltonian_label(self.method),
            application_counts=after - before,
        )

    @property
    def counters(self) -> ApplicationCounts:
        return self._ledger.snapshot()

    def reset_counters(self) -> None:
        self._ledger.reset()

    def expected_counters(self) -> ApplicationCounts:
        observed = self.counters
        return expected_application_counts(
            self.method,
            self.depth,
            energy_function_calls=observed.energy_function_calls,
            value_and_gradient_calls=observed.value_and_gradient_calls,
            state_replay_calls=observed.state_replay_calls,
            activation_checkpointing=self.activation_checkpointing,
        )

    def validate_counters(self) -> ApplicationCounts:
        observed = self.counters
        expected = self.expected_counters()
        if observed != expected:
            differences = {
                item.name: (
                    getattr(observed, item.name),
                    getattr(expected, item.name),
                )
                for item in fields(ApplicationCounts)
                if getattr(observed, item.name) != getattr(expected, item.name)
            }
            raise TorchAdapterError(
                f"operator application counters disagree with algebra: {differences}"
            )
        return observed

    def validate_optimizer_accounting(
        self, optimizer_counts: OptimizerCallCounts
    ) -> ApplicationCounts:
        """Cross-check aggregate callback counts against optimizer bookkeeping."""

        if not isinstance(optimizer_counts, OptimizerCallCounts):
            raise TorchAdapterError(
                "optimizer_counts must be an OptimizerCallCounts record"
            )
        observed = self.validate_counters()
        if observed.objective_evaluations != optimizer_counts.objective_calls:
            raise TorchAdapterError(
                "adapter objective evaluations disagree with optimizer accounting"
            )
        if observed.exact_gradient_calls != optimizer_counts.exact_gradient_calls:
            raise TorchAdapterError(
                "adapter exact-gradient calls disagree with optimizer accounting"
            )
        return observed


def build_method_adapter(
    instance: BCSTInstance,
    method: str,
    depth: int,
    *,
    phi: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    activation_checkpointing: bool = False,
) -> TorchMethodAdapter:
    """Construct the production callback bundle for one frozen trial cell."""

    return TorchMethodAdapter(
        instance,
        method,
        depth,
        phi=phi,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )


__all__ = [
    "ApplicationCounts",
    "CUBLAS_WORKSPACE_CONFIG",
    "DeterministicDevice",
    "PHI_METHODS",
    "PhiRecord",
    "ReplayDiagnostics",
    "STATE_NORM_TOLERANCE",
    "SUPPORTED_METHODS",
    "TargetExposureError",
    "TorchAdapterError",
    "TorchMethodAdapter",
    "build_method_adapter",
    "build_reduced_dynamics",
    "configure_deterministic_device",
    "expected_application_counts",
    "to_reduced_problem",
]
