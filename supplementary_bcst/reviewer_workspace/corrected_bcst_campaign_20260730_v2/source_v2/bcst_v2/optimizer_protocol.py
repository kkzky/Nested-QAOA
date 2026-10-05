"""Lean optimizer and restart protocol for the corrected BCST v2 campaign.

The module is intentionally independent of the campaign's state-vector
dynamics.  A campaign runner supplies NumPy-callable energy and gradient
functions; this module supplies deterministic initial angles, restart
validation, optimizer bookkeeping, incumbent selection, and budget decisions.

The implementation is bound to ``LEAN_EXECUTION_PLAN_20260730.json`` with
SHA-256
``172D2267B18745B3518E4EFCCC58B367AEFF9A2EC9169FF5EF322DD04762E7F6``.
Every context must supply that hash explicitly; there is no fallback plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


LEAN_EXECUTION_PLAN_SHA256 = (
    "172d2267b18745b3518e4efccc58b367aeff9a2ec9169ff5ef322dd04762e7f6"
)
CAMPAIGN_ID = "corrected_bcst_campaign_20260730_v2"
RNG_KDF_VERSION = "bcst-lean-sha256-independent-component-v1"

RESTART_COUNT = 3
RESTART_ROLE = "independent_random"
RESTART_ROLES_EXACT = (RESTART_ROLE,) * RESTART_COUNT
PHASE_TAGS = (
    "stage1_generation",
    "optimizer_calibration",
    "setting_selection",
    "depth_selection_N30",
    "depth_selection_N25",
    "confirmation_N30",
    "confirmation_N25",
    "budget_double_audit",
    "engineering_capacity_test",
)
PROHIBITED_RESTART_ROLES = (
    "deterministic_linear_ramp",
    "zero_or_identity_restart",
    "continuation",
    "interpolation",
    "history_append",
    "cross_method_transfer",
    "shared_stream",
    "unknown",
)
OPTIMIZED_METHODS = (
    "learned_projector",
    "ordinary_native_direct",
    "decoupled_native_direct",
    "same_stage1_state_direct",
)

ADAM_LEARNING_RATE_ARMS = (0.01, 0.02, 0.035)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
ADAM_BASE_STEPS = 200
ADAM_PATIENCE = 80
ADAM_IMPROVEMENT_TOLERANCE_FRACTION = 1e-10

SPSA_C_ARMS = (0.02, 0.05, 0.1)
SPSA_BASE_UPDATES = 160
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101
SPSA_A = 16.0
SPSA_CALIBRATION_DIRECTIONS = 8
SPSA_DIRECTIONS_PER_UPDATE = 1
SPSA_TARGET_FIRST_UPDATE_RMS = 0.05
SPSA_MAX_UPDATE_RMS = 0.10

REPLAY_ABSOLUTE_TOLERANCE = 1e-9
MIN_SPECTRAL_RANGE = 1e-12

EnergyFunction = Callable[[np.ndarray], float]
ValueAndGradientFunction = Callable[[np.ndarray], tuple[float, np.ndarray]]
ParameterLayoutInput = Mapping[str, int] | Sequence[tuple[str, int]]


class ProtocolViolation(ValueError):
    """Raised when an optimizer record violates the frozen protocol."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the KDF domain as sorted, compact, finite UTF-8 JSON."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(f"domain is not canonical-JSON compatible: {exc}") from exc
    return text.encode("utf-8")


def _require_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ProtocolViolation(f"{field_name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ProtocolViolation(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


def _require_plain_int(value: Any, *, field_name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ProtocolViolation(f"{field_name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ProtocolViolation(f"{field_name} must be at least {minimum}")
    return result


def _finite_energy(value: Any, *, where: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolViolation(f"{where} returned a non-finite energy")
    return result


def _call_objective(
    objective: EnergyFunction,
    parameters: np.ndarray,
    *,
    where: str,
) -> float:
    # Optimizer state is never shared with an external callback.  A buggy
    # callback may mutate its private copy, but cannot replace a KDF start or
    # a later central iterate.
    callback_parameters = np.array(parameters, dtype=np.float64, copy=True, order="C")
    return _finite_energy(objective(callback_parameters), where=where)


def _call_value_and_gradient(
    value_and_gradient: ValueAndGradientFunction,
    parameters: np.ndarray,
    *,
    where: str,
) -> tuple[float, np.ndarray]:
    callback_parameters = np.array(parameters, dtype=np.float64, copy=True, order="C")
    value, gradient = value_and_gradient(callback_parameters)
    isolated_gradient = np.array(
        gradient, dtype=np.float64, copy=True, order="C"
    )
    return _finite_energy(value, where=where), isolated_gradient


def _readonly_float64(values: Any, *, one_dimensional: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if one_dimensional and array.ndim != 1:
        raise ProtocolViolation("parameter arrays must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ProtocolViolation("parameter arrays must contain only finite values")
    output = np.array(array, dtype=np.float64, copy=True, order="C")
    output.setflags(write=False)
    return output


def normalize_parameter_layout(layout: ParameterLayoutInput) -> tuple[tuple[str, int], ...]:
    """Return a validated, ordered parameter-family layout."""

    items = tuple(layout.items()) if isinstance(layout, Mapping) else tuple(layout)
    if not items:
        raise ProtocolViolation("at least one parameter family is required")
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ProtocolViolation("each parameter layout entry must be (family, size)")
        family = str(item[0])
        if not family:
            raise ProtocolViolation("parameter-family names must be nonempty")
        if family in seen:
            raise ProtocolViolation(f"duplicate parameter family {family!r}")
        size = _require_plain_int(item[1], field_name=f"size of {family}", minimum=1)
        seen.add(family)
        normalized.append((family, size))
    return tuple(normalized)


def parameter_layout_for_method(method: str, depth: int) -> tuple[tuple[str, int], ...]:
    """Return the contract parameter ordering without importing circuit dynamics."""

    p = _require_plain_int(depth, field_name="depth", minimum=1)
    layouts = {
        "learned_projector": (("objective_gamma", p), ("projector_beta", p)),
        "ordinary_native_direct": (("shared_cost_gamma", p), ("xy_beta", p)),
        "decoupled_native_direct": (
            ("conflict_gamma", p),
            ("objective_gamma", p),
            ("xy_beta", p),
        ),
        "same_stage1_state_direct": (("shared_cost_gamma", p), ("xy_beta", p)),
        "stage1": (("conflict_gamma", p), ("xy_beta", p)),
    }
    try:
        return layouts[str(method)]
    except KeyError as exc:
        raise ProtocolViolation(f"unknown optimizer method {method!r}") from exc


@dataclass(frozen=True)
class RestartContext:
    """All non-component fields in one initializer RNG domain."""

    phase: str
    N: int
    setting: Any
    method: str
    optimizer: Literal["Adam", "SPSA"] | str
    problem_master_seed: int
    depth: int
    hyperparameter_arm: Any
    lean_execution_plan_hash: str
    campaign: str = CAMPAIGN_ID

    def __post_init__(self) -> None:
        if self.phase not in PHASE_TAGS:
            raise ProtocolViolation(f"unsupported phase tag {self.phase!r}")
        N = _require_plain_int(self.N, field_name="N", minimum=1)
        if N not in (25, 30):
            raise ProtocolViolation("the frozen campaign supports only N=25 and N=30")
        depth = _require_plain_int(self.depth, field_name="depth", minimum=1)
        seed = _require_plain_int(
            self.problem_master_seed, field_name="problem_master_seed", minimum=0
        )
        if not str(self.method):
            raise ProtocolViolation("method must be nonempty")
        if self.optimizer not in ("Adam", "SPSA"):
            raise ProtocolViolation("optimizer must be exactly 'Adam' or 'SPSA'")
        if not str(self.campaign):
            raise ProtocolViolation("campaign must be nonempty")
        normalized_hash = _require_sha256(
            self.lean_execution_plan_hash, field_name="lean_execution_plan_hash"
        )
        if normalized_hash != LEAN_EXECUTION_PLAN_SHA256:
            raise ProtocolViolation(
                "lean_execution_plan_hash does not match the authoritative lean plan"
            )
        # This also rejects NaN, non-string mapping keys, and unsupported objects.
        canonical_json_bytes(self.setting)
        canonical_json_bytes(self.hyperparameter_arm)
        if self.phase.endswith("_N30") and N != 30:
            raise ProtocolViolation(f"phase {self.phase!r} requires N=30")
        if self.phase.endswith("_N25") and N != 25:
            raise ProtocolViolation(f"phase {self.phase!r} requires N=25")
        if self.phase in (
            "optimizer_calibration",
            "setting_selection",
            "budget_double_audit",
        ) and N != 30:
            raise ProtocolViolation(f"phase {self.phase!r} is frozen to N=30")
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "problem_master_seed", seed)
        object.__setattr__(self, "lean_execution_plan_hash", normalized_hash)

    def base_domain(
        self,
        restart_index: int,
        *,
        initializer_kdf: bool = False,
    ) -> dict[str, Any]:
        """Return the exact domain fields shared by components of one restart."""

        index = _require_plain_int(
            restart_index, field_name="restart_index", minimum=0
        )
        domain = {
            "campaign": self.campaign,
            "lean_execution_plan_hash": self.lean_execution_plan_hash,
            "phase": self.phase,
            "N": self.N,
            "setting": self.setting,
            "method": self.method,
            "optimizer": self.optimizer,
            "problem_master_seed": self.problem_master_seed,
            "depth": self.depth,
            "restart_index": index,
        }
        # Calibration arm comparisons must use common theta_0 vectors.  This
        # exception applies only to initializer components: SPSA perturbations
        # remain arm-domain-separated.
        if not (initializer_kdf and self.phase == "optimizer_calibration"):
            domain["hyperparameter_arm"] = self.hyperparameter_arm
        return domain

    def component_domain(
        self,
        *,
        restart_index: int,
        parameter_family: str,
        component_index: int,
        initializer_kdf: bool = False,
    ) -> dict[str, Any]:
        family = str(parameter_family)
        if not family:
            raise ProtocolViolation("parameter_family must be nonempty")
        index = _require_plain_int(
            component_index, field_name="component_index", minimum=0
        )
        return {
            **self.base_domain(
                restart_index, initializer_kdf=initializer_kdf
            ),
            "parameter_family": family,
            "component_index": index,
        }


def sha256_uniform_angle(component_domain: Mapping[str, Any]) -> float:
    """Map one SHA-256 component domain to an interior Uniform[-pi, pi] point."""

    digest = hashlib.sha256(canonical_json_bytes(dict(component_domain))).digest()
    z = int.from_bytes(digest[:8], "big") >> 11
    u = (z + 0.5) / float(1 << 53)
    return -math.pi + (2.0 * math.pi * u)


def derive_uniform_angle(
    context: RestartContext,
    *,
    restart_index: int,
    parameter_family: str,
    component_index: int,
) -> float:
    """Derive one independently hashed initializer component."""

    domain = context.component_domain(
        restart_index=restart_index,
        parameter_family=parameter_family,
        component_index=component_index,
        initializer_kdf=True,
    )
    return sha256_uniform_angle(domain)


def _canonical_parameter_bytes(
    parameters: np.ndarray | Sequence[float],
) -> bytes:
    array = np.asarray(parameters, dtype=np.float64)
    if array.ndim != 1:
        raise ProtocolViolation("parameters must be one-dimensional for hashing")
    canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
    return canonical.tobytes(order="C")


def _same_parameter_bytes(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
) -> bool:
    return _canonical_parameter_bytes(left) == _canonical_parameter_bytes(right)


def parameter_sha256(parameters: np.ndarray | Sequence[float]) -> str:
    """Hash the canonical little-endian binary64 flattened parameter array."""

    return hashlib.sha256(_canonical_parameter_bytes(parameters)).hexdigest()


@dataclass(frozen=True)
class RestartInitializer:
    """A KDF-created initializer; arbitrary supplied initializers are not accepted."""

    context: RestartContext
    restart_index: int
    role: str
    layout: tuple[tuple[str, int], ...]
    parameters: np.ndarray = field(repr=False, compare=False)
    parameter_hash: str
    supplied_initializer: bool = False

    def __post_init__(self) -> None:
        index = _require_plain_int(
            self.restart_index, field_name="restart_index", minimum=0
        )
        layout = normalize_parameter_layout(self.layout)
        parameters = _readonly_float64(self.parameters)
        expected_size = sum(size for _, size in layout)
        if parameters.size != expected_size:
            raise ProtocolViolation(
                f"initializer has {parameters.size} values; layout requires {expected_size}"
            )
        digest = _require_sha256(self.parameter_hash, field_name="parameter_hash")
        if not isinstance(self.supplied_initializer, (bool, np.bool_)):
            raise ProtocolViolation("supplied_initializer must be boolean")
        object.__setattr__(self, "restart_index", index)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "parameter_hash", digest)
        object.__setattr__(self, "supplied_initializer", bool(self.supplied_initializer))

    @property
    def initial_parameters(self) -> np.ndarray:
        """A read-only flattened binary64 initializer array."""

        return self.parameters

    @property
    def domain_metadata(self) -> dict[str, Any]:
        return {
            "rng_kdf_version": RNG_KDF_VERSION,
            **self.context.base_domain(
                self.restart_index, initializer_kdf=True
            ),
            "scalar_distribution": "independent_uniform[-pi,pi]",
            "calibration_arm_excluded": (
                self.context.phase == "optimizer_calibration"
            ),
        }

    def component_domain(self, parameter_family: str, component_index: int) -> dict[str, Any]:
        return self.context.component_domain(
            restart_index=self.restart_index,
            parameter_family=parameter_family,
            component_index=component_index,
            initializer_kdf=True,
        )

    def family_arrays(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        offset = 0
        for family, size in self.layout:
            values = np.array(self.parameters[offset : offset + size], copy=True)
            values.setflags(write=False)
            output[family] = values
            offset += size
        return output


def make_random_initializer(
    context: RestartContext,
    *,
    restart_index: int,
    layout: ParameterLayoutInput,
) -> RestartInitializer:
    """Construct one and only one allowed restart type from its SHA-256 domain."""

    index = _require_plain_int(restart_index, field_name="restart_index", minimum=0)
    normalized_layout = normalize_parameter_layout(layout)
    values = [
        derive_uniform_angle(
            context,
            restart_index=index,
            parameter_family=family,
            component_index=component,
        )
        for family, size in normalized_layout
        for component in range(size)
    ]
    parameters = np.asarray(values, dtype=np.float64)
    return RestartInitializer(
        context=context,
        restart_index=index,
        role=RESTART_ROLE,
        layout=normalized_layout,
        parameters=parameters,
        parameter_hash=parameter_sha256(parameters),
        supplied_initializer=False,
    )


def generate_three_random_initializers(
    context: RestartContext,
    layout: ParameterLayoutInput,
) -> tuple[RestartInitializer, RestartInitializer, RestartInitializer]:
    """Generate the exact three isolated independent-random starts."""

    normalized_layout = normalize_parameter_layout(layout)
    records = tuple(
        make_random_initializer(
            context, restart_index=index, layout=normalized_layout
        )
        for index in range(RESTART_COUNT)
    )
    # Kept as a runtime assertion because a digest collision is a protocol failure.
    validate_restart_batch(
        records, expected_context=context, expected_layout=normalized_layout
    )
    return records  # type: ignore[return-value]


def _validate_random_initializer(
    record: RestartInitializer,
    *,
    expected_context: RestartContext,
    expected_layout: tuple[tuple[str, int], ...],
    expected_index: int | None = None,
) -> RestartInitializer:
    if not isinstance(record, RestartInitializer):
        raise ProtocolViolation("optimizer restart must be a RestartInitializer")
    if record.role != RESTART_ROLE:
        raise ProtocolViolation(
            f"restart role must be {RESTART_ROLE!r}, observed {record.role!r}"
        )
    if record.supplied_initializer:
        raise ProtocolViolation("supplied_initializer_count must be zero")
    if expected_index is not None and record.restart_index != expected_index:
        raise ProtocolViolation(
            f"restart index must be {expected_index}, observed {record.restart_index}"
        )
    if record.restart_index not in range(RESTART_COUNT):
        raise ProtocolViolation("restart index must be one of 0, 1, 2")
    if record.context != expected_context:
        raise ProtocolViolation(
            "RNG domain mismatch: phase/method/optimizer/seed/depth/arm "
            "must equal the expected restart context"
        )
    if record.layout != expected_layout:
        raise ProtocolViolation("initializer parameter layout mismatch")
    observed_hash = parameter_sha256(record.parameters)
    if record.parameter_hash != observed_hash:
        raise ProtocolViolation(
            f"initializer parameter hash mismatch for restart {record.restart_index}"
        )
    expected = make_random_initializer(
        expected_context,
        restart_index=record.restart_index,
        layout=expected_layout,
    )
    if not _same_parameter_bytes(record.parameters, expected.parameters):
        raise ProtocolViolation(
            f"initializer bytes do not match the KDF domain for restart "
            f"{record.restart_index}"
        )
    return record


def validate_restart_batch(
    restarts: Sequence[RestartInitializer],
    *,
    expected_context: RestartContext | None = None,
    expected_layout: ParameterLayoutInput | None = None,
) -> tuple[RestartInitializer, RestartInitializer, RestartInitializer]:
    """Hard-fail any departure from the frozen three-random-start protocol."""

    records = tuple(restarts)
    if len(records) != RESTART_COUNT:
        raise ProtocolViolation(
            f"restart_count must be exactly {RESTART_COUNT}, observed {len(records)}"
        )
    if any(not isinstance(record, RestartInitializer) for record in records):
        raise ProtocolViolation("every optimizer restart must be a RestartInitializer")
    roles = tuple(record.role for record in records)
    if roles != RESTART_ROLES_EXACT:
        forbidden = [role for role in roles if role in PROHIBITED_RESTART_ROLES]
        detail = f"; forbidden roles={forbidden}" if forbidden else ""
        raise ProtocolViolation(
            f"restart roles must equal {RESTART_ROLES_EXACT}, observed {roles}{detail}"
        )
    if any(record.supplied_initializer for record in records):
        raise ProtocolViolation("supplied_initializer_count must be zero")
    if tuple(record.restart_index for record in records) != tuple(range(RESTART_COUNT)):
        raise ProtocolViolation("restart indices must be ordered exactly as 0, 1, 2")

    context = expected_context if expected_context is not None else records[0].context
    layout = (
        normalize_parameter_layout(expected_layout)
        if expected_layout is not None
        else records[0].layout
    )
    for expected_index, record in enumerate(records):
        _validate_random_initializer(
            record,
            expected_context=context,
            expected_layout=layout,
            expected_index=expected_index,
        )
    hashes = tuple(record.parameter_hash for record in records)
    if len(set(hashes)) != RESTART_COUNT:
        raise ProtocolViolation("all three initializer parameter hashes must be distinct")
    return records  # type: ignore[return-value]


# Explicit name used by campaign validators.
validate_three_random_restarts = validate_restart_batch


@dataclass(frozen=True)
class ZeroAngleDiagnostic:
    """A separate, never-eligible zero-angle diagnostic record."""

    parameters: np.ndarray = field(repr=False, compare=False)
    energy: float
    parameter_count: int
    role: str = "zero_angle_diagnostic"
    optimizer_eligible: bool = False
    objective_calls: int = 1

    def __post_init__(self) -> None:
        count = _require_plain_int(
            self.parameter_count, field_name="parameter_count", minimum=1
        )
        parameters = _readonly_float64(self.parameters)
        if parameters.size != count or np.any(parameters != 0.0):
            raise ProtocolViolation("zero-angle diagnostic must contain exactly all zeros")
        if self.role != "zero_angle_diagnostic":
            raise ProtocolViolation("zero diagnostic role is immutable")
        if self.optimizer_eligible is not False:
            raise ProtocolViolation("zero diagnostic can never be optimizer eligible")
        if self.objective_calls != 1:
            raise ProtocolViolation("zero diagnostic is evaluated exactly once")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "energy", _finite_energy(self.energy, where="diagnostic"))
        object.__setattr__(self, "parameter_count", count)


def evaluate_zero_angle_diagnostic(
    objective: EnergyFunction,
    parameter_count_or_layout: int | ParameterLayoutInput,
) -> ZeroAngleDiagnostic:
    """Evaluate, but never promote, the method/depth zero-angle state."""

    if isinstance(parameter_count_or_layout, (int, np.integer)) and not isinstance(
        parameter_count_or_layout, (bool, np.bool_)
    ):
        count = _require_plain_int(
            parameter_count_or_layout, field_name="parameter_count", minimum=1
        )
    else:
        count = sum(
            size for _, size in normalize_parameter_layout(parameter_count_or_layout)
        )
    parameters = np.zeros(count, dtype=np.float64)
    energy = _call_objective(
        objective, parameters, where="zero-angle diagnostic"
    )
    return ZeroAngleDiagnostic(parameters, energy, count)


@dataclass(frozen=True)
class Checkpoint:
    """One optimizer-eligible central iterate."""

    index: int
    energy: float
    parameters: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        index = _require_plain_int(self.index, field_name="checkpoint index", minimum=0)
        object.__setattr__(self, "index", index)
        object.__setattr__(
            self, "energy", _finite_energy(self.energy, where=f"checkpoint {index}")
        )
        object.__setattr__(self, "parameters", _readonly_float64(self.parameters))


def select_incumbent(checkpoints: Sequence[Checkpoint]) -> Checkpoint:
    """Select lowest energy, with exact ties resolved by earlier checkpoint."""

    points = tuple(checkpoints)
    if not points:
        raise ProtocolViolation("at least one eligible checkpoint is required")
    if len({point.index for point in points}) != len(points):
        raise ProtocolViolation("checkpoint indices must be unique")
    return min(points, key=lambda point: (point.energy, point.index))


@dataclass(frozen=True)
class OptimizerCallCounts:
    """Typed, non-overlapping optimizer-call accounting."""

    optimizer: Literal["Adam", "SPSA"] | str
    restart_count: int
    executed_update_steps: int
    calibration_directions_per_restart: int
    initial_objective_calls: int
    calibration_perturbation_calls: int
    update_perturbation_calls: int
    central_iterate_objective_calls: int
    gradient_forward_objective_calls: int
    terminal_objective_calls: int
    selected_point_replay_calls: int
    exact_gradient_calls: int

    def __post_init__(self) -> None:
        if self.optimizer not in ("Adam", "SPSA"):
            raise ProtocolViolation("count schema optimizer must be Adam or SPSA")
        for name in (
            "restart_count",
            "executed_update_steps",
            "calibration_directions_per_restart",
            "initial_objective_calls",
            "calibration_perturbation_calls",
            "update_perturbation_calls",
            "central_iterate_objective_calls",
            "gradient_forward_objective_calls",
            "terminal_objective_calls",
            "selected_point_replay_calls",
            "exact_gradient_calls",
        ):
            normalized = _require_plain_int(
                getattr(self, name), field_name=name, minimum=0
            )
            object.__setattr__(self, name, normalized)

    @property
    def perturbation_objective_calls(self) -> int:
        return self.calibration_perturbation_calls + self.update_perturbation_calls

    @property
    def objective_calls(self) -> int:
        return (
            self.initial_objective_calls
            + self.calibration_perturbation_calls
            + self.update_perturbation_calls
            + self.central_iterate_objective_calls
            + self.gradient_forward_objective_calls
            + self.terminal_objective_calls
            + self.selected_point_replay_calls
        )

    def to_contract_schema(self) -> dict[str, Any]:
        if self.optimizer == "SPSA":
            gradient_status = "valid_zero_not_applicable"
        else:
            gradient_status = "applicable"
        return {
            "optimizer": self.optimizer,
            "restart_count": self.restart_count,
            "executed_update_steps": self.executed_update_steps,
            "calibration_directions_per_restart": (
                self.calibration_directions_per_restart
            ),
            "objective_calls": self.objective_calls,
            "initial_objective_calls": self.initial_objective_calls,
            "perturbation_objective_calls": self.perturbation_objective_calls,
            "calibration_perturbation_calls": self.calibration_perturbation_calls,
            "update_perturbation_calls": self.update_perturbation_calls,
            "central_iterate_objective_calls": self.central_iterate_objective_calls,
            "gradient_forward_objective_calls": self.gradient_forward_objective_calls,
            "terminal_objective_calls": self.terminal_objective_calls,
            "selected_point_replay_calls": self.selected_point_replay_calls,
            "exact_gradient_calls": self.exact_gradient_calls,
            "exact_gradient_calls_status": gradient_status,
        }

    def validate(self) -> "OptimizerCallCounts":
        if self.restart_count < 1:
            raise ProtocolViolation("call accounting requires at least one restart")
        if self.initial_objective_calls != self.restart_count:
            raise ProtocolViolation("every restart requires exactly one initial evaluation")
        if self.optimizer == "SPSA":
            if self.exact_gradient_calls != 0:
                raise ProtocolViolation("SPSA exact_gradient_calls must be valid zero")
            if (
                self.calibration_directions_per_restart
                != SPSA_CALIBRATION_DIRECTIONS
            ):
                raise ProtocolViolation(
                    "SPSA requires exactly eight calibration directions per restart"
                )
            expected_calibration_calls = (
                2 * self.restart_count * self.calibration_directions_per_restart
            )
            if self.calibration_perturbation_calls != expected_calibration_calls:
                raise ProtocolViolation(
                    "SPSA calibration calls must equal two times restart count "
                    "times calibration directions"
                )
            if self.gradient_forward_objective_calls or self.terminal_objective_calls:
                raise ProtocolViolation("SPSA cannot report Adam-only call categories")
            if self.update_perturbation_calls != 2 * self.executed_update_steps:
                raise ProtocolViolation(
                    "SPSA requires two update perturbation calls per update"
                )
            if self.central_iterate_objective_calls != self.executed_update_steps:
                raise ProtocolViolation(
                    "SPSA requires one updated-central-iterate evaluation per update"
                )
            if self.perturbation_objective_calls <= 0:
                raise ProtocolViolation("SPSA perturbation calls cannot be missing or zero")
        else:
            if self.calibration_directions_per_restart != 0:
                raise ProtocolViolation("Adam calibration directions must be zero")
            if self.calibration_perturbation_calls or self.update_perturbation_calls:
                raise ProtocolViolation("Adam cannot report SPSA perturbation calls")
            if self.central_iterate_objective_calls:
                raise ProtocolViolation("Adam cannot report SPSA central-iterate calls")
            if self.gradient_forward_objective_calls != self.executed_update_steps:
                raise ProtocolViolation(
                    "Adam requires one gradient-forward objective call per update"
                )
            if self.exact_gradient_calls != self.executed_update_steps:
                raise ProtocolViolation(
                    "Adam exact-gradient calls must equal executed update steps"
                )
            if self.exact_gradient_calls <= 0:
                raise ProtocolViolation("zero Adam exact-gradient calls are a hard failure")
            if self.terminal_objective_calls != self.restart_count:
                raise ProtocolViolation(
                    "Adam requires one terminal/final evaluation per restart"
                )
        return self


def expected_spsa_call_counts(
    updates: int,
    *,
    restarts: int = RESTART_COUNT,
    calibration_directions: int = SPSA_CALIBRATION_DIRECTIONS,
    include_selected_replay: bool = True,
) -> OptimizerCallCounts:
    """Return exact SPSA accounting, including central-iterate evaluations."""

    B = _require_plain_int(updates, field_name="updates", minimum=1)
    R = _require_plain_int(restarts, field_name="restarts", minimum=1)
    directions = _require_plain_int(
        calibration_directions, field_name="calibration_directions", minimum=1
    )
    counts = OptimizerCallCounts(
        optimizer="SPSA",
        restart_count=R,
        executed_update_steps=R * B,
        calibration_directions_per_restart=directions,
        initial_objective_calls=R,
        calibration_perturbation_calls=R * 2 * directions,
        update_perturbation_calls=R * 2 * B,
        central_iterate_objective_calls=R * B,
        gradient_forward_objective_calls=0,
        terminal_objective_calls=0,
        selected_point_replay_calls=int(bool(include_selected_replay)),
        exact_gradient_calls=0,
    )
    return counts.validate()


def spsa_objective_call_count(
    updates: int,
    *,
    restarts: int = RESTART_COUNT,
    calibration_directions: int = SPSA_CALIBRATION_DIRECTIONS,
    include_selected_replay: bool = True,
) -> int:
    """Convenience scalar for the frozen ``R*(1+2D+3B)+replay`` formula."""

    return expected_spsa_call_counts(
        updates,
        restarts=restarts,
        calibration_directions=calibration_directions,
        include_selected_replay=include_selected_replay,
    ).objective_calls


def expected_adam_call_counts(
    steps_per_restart: int | Sequence[int],
    *,
    restarts: int = RESTART_COUNT,
    include_selected_replay: bool = True,
) -> OptimizerCallCounts:
    """Return the standard Adam count schema for actual executed steps."""

    if isinstance(steps_per_restart, (int, np.integer)) and not isinstance(
        steps_per_restart, (bool, np.bool_)
    ):
        R = _require_plain_int(restarts, field_name="restarts", minimum=1)
        steps = (
            _require_plain_int(
                steps_per_restart, field_name="steps_per_restart", minimum=1
            ),
        ) * R
    else:
        steps = tuple(
            _require_plain_int(step, field_name="executed Adam step", minimum=1)
            for step in steps_per_restart
        )
        if not steps:
            raise ProtocolViolation("at least one Adam restart count is required")
        R = len(steps)
        if restarts != RESTART_COUNT and restarts != R:
            raise ProtocolViolation("restarts disagrees with steps_per_restart")
        if restarts == RESTART_COUNT and R != RESTART_COUNT:
            # A sequence is authoritative, but silently changing the frozen count is unsafe.
            raise ProtocolViolation("the frozen Adam protocol requires three restart counts")
    total_steps = sum(steps)
    counts = OptimizerCallCounts(
        optimizer="Adam",
        restart_count=R,
        executed_update_steps=total_steps,
        calibration_directions_per_restart=0,
        initial_objective_calls=R,
        calibration_perturbation_calls=0,
        update_perturbation_calls=0,
        central_iterate_objective_calls=0,
        gradient_forward_objective_calls=total_steps,
        terminal_objective_calls=R,
        selected_point_replay_calls=int(bool(include_selected_replay)),
        exact_gradient_calls=total_steps,
    )
    return counts.validate()


# Short schema-oriented aliases for callers that do not execute the toy runners.
adam_count_schema = expected_adam_call_counts
spsa_count_schema = expected_spsa_call_counts


@dataclass(frozen=True)
class AdamSettings:
    """The lean-plan Adam algorithm and frozen defaults."""

    learning_rate: float
    maximum_steps: int = ADAM_BASE_STEPS
    beta1: float = ADAM_BETA1
    beta2: float = ADAM_BETA2
    epsilon: float = ADAM_EPSILON
    patience: int = ADAM_PATIENCE
    improvement_tolerance_fraction: float = ADAM_IMPROVEMENT_TOLERANCE_FRACTION
    replay_absolute_tolerance: float = REPLAY_ABSOLUTE_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "improvement_tolerance_fraction",
            "replay_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProtocolViolation(f"Adam {name} must be finite")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0.0:
            raise ProtocolViolation("Adam learning_rate must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ProtocolViolation("Adam beta values must lie in [0,1)")
        if self.epsilon <= 0.0:
            raise ProtocolViolation("Adam epsilon must be positive")
        if self.improvement_tolerance_fraction < 0.0:
            raise ProtocolViolation("Adam improvement tolerance cannot be negative")
        if self.replay_absolute_tolerance < 0.0:
            raise ProtocolViolation("Adam replay tolerance cannot be negative")
        object.__setattr__(
            self,
            "maximum_steps",
            _require_plain_int(
                self.maximum_steps, field_name="maximum_steps", minimum=1
            ),
        )
        object.__setattr__(
            self,
            "patience",
            _require_plain_int(self.patience, field_name="patience", minimum=1),
        )

    def to_contract_schema(self) -> dict[str, Any]:
        return {
            "name": "Adam",
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "maximum_steps": self.maximum_steps,
            "patience": self.patience,
            "improvement_tolerance_fraction_of_spectral_range": (
                self.improvement_tolerance_fraction
            ),
            "gradient": "exact",
            "angle_wrapping": False,
        }


def standard_adam_settings(
    learning_rate: float,
    *,
    budget: int = ADAM_BASE_STEPS,
) -> AdamSettings:
    """Build the standard Stage-2 Adam tuple (beta1=.9, beta2=.999, eps=1e-8)."""

    return AdamSettings(learning_rate=learning_rate, maximum_steps=budget)


def stage1_adam_settings() -> AdamSettings:
    """Return the separately frozen Stage-1 Adam tuple."""

    return AdamSettings(
        learning_rate=0.035,
        maximum_steps=400,
        patience=160,
    )


@dataclass(frozen=True)
class SPSASettings:
    """The frozen one-direction, two-sided SPSA protocol."""

    c: float
    updates: int = SPSA_BASE_UPDATES
    alpha: float = SPSA_ALPHA
    gamma: float = SPSA_GAMMA
    calibration_directions: int = SPSA_CALIBRATION_DIRECTIONS
    directions_per_update: int = SPSA_DIRECTIONS_PER_UPDATE
    target_first_update_rms: float = SPSA_TARGET_FIRST_UPDATE_RMS
    max_update_rms: float = SPSA_MAX_UPDATE_RMS
    replay_absolute_tolerance: float = REPLAY_ABSOLUTE_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "c",
            "alpha",
            "gamma",
            "target_first_update_rms",
            "max_update_rms",
            "replay_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProtocolViolation(f"SPSA {name} must be finite")
            object.__setattr__(self, name, value)
        if self.c <= 0.0 or self.alpha <= 0.0 or self.gamma <= 0.0:
            raise ProtocolViolation("SPSA c, alpha, and gamma must be positive")
        if self.target_first_update_rms <= 0.0 or self.max_update_rms <= 0.0:
            raise ProtocolViolation("SPSA RMS scales must be positive")
        if self.replay_absolute_tolerance < 0.0:
            raise ProtocolViolation("SPSA replay tolerance cannot be negative")
        object.__setattr__(
            self,
            "updates",
            _require_plain_int(self.updates, field_name="updates", minimum=1),
        )
        object.__setattr__(
            self,
            "calibration_directions",
            _require_plain_int(
                self.calibration_directions,
                field_name="calibration_directions",
                minimum=1,
            ),
        )
        directions = _require_plain_int(
            self.directions_per_update,
            field_name="directions_per_update",
            minimum=1,
        )
        if directions != SPSA_DIRECTIONS_PER_UPDATE:
            raise ProtocolViolation("the lean plan freezes one SPSA direction per update")
        object.__setattr__(self, "directions_per_update", directions)
        if self.calibration_directions != SPSA_CALIBRATION_DIRECTIONS:
            raise ProtocolViolation(
                "the lean plan freezes exactly eight SPSA calibration directions"
            )

    @property
    def A(self) -> float:
        # The lean plan freezes A to the base-budget value.  Changing only the
        # execution horizon from B to 2B must not alter any prefix update.
        return SPSA_A

    def calibrated_a(self, gradient_rms: float) -> float:
        rms = float(gradient_rms)
        if not math.isfinite(rms) or rms < 0.0:
            raise ProtocolViolation("SPSA gradient RMS must be finite and nonnegative")
        return (
            self.target_first_update_rms
            * (1.0 + self.A) ** self.alpha
            / max(rms, MIN_SPECTRAL_RANGE)
        )

    def to_contract_schema(self) -> dict[str, Any]:
        return {
            "name": "SPSA",
            "c": self.c,
            "updates": self.updates,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "A": self.A,
            "calibration_directions": self.calibration_directions,
            "directions_per_update": self.directions_per_update,
            "target_first_update_rms": self.target_first_update_rms,
            "max_update_rms": self.max_update_rms,
            "angle_wrapping": False,
            "early_stopping": False,
        }


def standard_spsa_settings(
    c: float,
    *,
    budget: int = SPSA_BASE_UPDATES,
) -> SPSASettings:
    return SPSASettings(c=c, updates=budget)


def _validate_context_arm_matches_settings(
    context: RestartContext,
    settings: AdamSettings | SPSASettings,
) -> None:
    arm = context.hyperparameter_arm
    if not isinstance(arm, Mapping):
        raise ProtocolViolation("hyperparameter_arm must be a one-field mapping")
    if isinstance(settings, AdamSettings):
        expected_key = "learning_rate"
        expected_value = settings.learning_rate
        allowed_values = ADAM_LEARNING_RATE_ARMS
        expected_optimizer = "Adam"
    else:
        expected_key = "c"
        expected_value = settings.c
        allowed_values = SPSA_C_ARMS
        expected_optimizer = "SPSA"
    if context.optimizer != expected_optimizer:
        raise ProtocolViolation(
            f"{expected_optimizer} settings require an {expected_optimizer} context"
        )
    if set(arm) != {expected_key}:
        raise ProtocolViolation(
            f"{expected_optimizer} hyperparameter_arm must contain exactly "
            f"{expected_key!r}"
        )
    if isinstance(arm[expected_key], (bool, np.bool_)) or not isinstance(
        arm[expected_key], (int, float, np.integer, np.floating)
    ):
        raise ProtocolViolation(
            f"context arm {expected_key} must be a numeric scalar"
        )
    arm_value = float(arm[expected_key])
    if not math.isfinite(arm_value) or arm_value != expected_value:
        raise ProtocolViolation(
            f"context arm {expected_key}={arm_value!r} does not match "
            f"optimizer settings {expected_value!r}"
        )
    if expected_value not in allowed_values:
        raise ProtocolViolation(
            f"{expected_optimizer} scalar arm {expected_value!r} is not in "
            "the lean-plan arm set"
        )


def _method_layout_for_context(
    context: RestartContext,
) -> tuple[tuple[str, int], ...]:
    return parameter_layout_for_method(context.method, context.depth)


def _spsa_direction_domain(
    initializer: RestartInitializer,
    *,
    kind: Literal["calibration", "update"],
    iteration_index: int,
    direction_index: int,
    parameter_index: int,
) -> dict[str, Any]:
    iteration = _require_plain_int(
        iteration_index, field_name="iteration_index", minimum=0
    )
    direction = _require_plain_int(
        direction_index, field_name="direction_index", minimum=0
    )
    parameter = _require_plain_int(
        parameter_index, field_name="parameter_index", minimum=0
    )
    if kind not in ("calibration", "update"):
        raise ProtocolViolation("SPSA direction kind must be calibration or update")
    return {
        **initializer.context.component_domain(
            restart_index=initializer.restart_index,
            parameter_family="spsa_perturbation",
            component_index=parameter,
        ),
        "calibration_or_update": kind,
        "iteration_index": iteration,
        "direction_index": direction,
        "parameter_index": parameter,
    }


def spsa_rademacher_direction(
    initializer: RestartInitializer,
    *,
    kind: Literal["calibration", "update"],
    iteration_index: int,
    direction_index: int = 0,
) -> np.ndarray:
    """Derive one independent Rademacher vector from SHA-256 component domains."""

    signs = np.empty(initializer.parameters.size, dtype=np.float64)
    for parameter_index in range(signs.size):
        domain = _spsa_direction_domain(
            initializer,
            kind=kind,
            iteration_index=iteration_index,
            direction_index=direction_index,
            parameter_index=parameter_index,
        )
        digest = hashlib.sha256(canonical_json_bytes(domain)).digest()
        signs[parameter_index] = 1.0 if (digest[0] & 1) else -1.0
    signs.setflags(write=False)
    return signs


@dataclass(frozen=True)
class RestartResult:
    optimizer: Literal["Adam", "SPSA"] | str
    restart_index: int
    initializer: RestartInitializer
    checkpoints: tuple[Checkpoint, ...]
    best_checkpoint: int
    best_energy: float
    best_parameters: np.ndarray = field(repr=False, compare=False)
    terminal_energy: float
    executed_updates: int
    calls: OptimizerCallCounts
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.optimizer not in ("Adam", "SPSA"):
            raise ProtocolViolation("restart result optimizer must be Adam or SPSA")
        if self.optimizer != self.initializer.context.optimizer:
            raise ProtocolViolation(
                "restart result optimizer disagrees with initializer context"
            )
        if self.restart_index != self.initializer.restart_index:
            raise ProtocolViolation("restart result index disagrees with initializer")
        _validate_random_initializer(
            self.initializer,
            expected_context=self.initializer.context,
            expected_layout=_method_layout_for_context(self.initializer.context),
            expected_index=self.restart_index,
        )
        points = tuple(self.checkpoints)
        if not points or points[0].index != 0:
            raise ProtocolViolation("restart trace must include initial checkpoint zero")
        if not _same_parameter_bytes(
            points[0].parameters, self.initializer.parameters
        ):
            raise ProtocolViolation(
                "checkpoint-zero bytes do not equal the reconstructed initializer"
            )
        if tuple(point.index for point in points) != tuple(range(len(points))):
            raise ProtocolViolation("restart checkpoint indices must be contiguous")
        incumbent = select_incumbent(points)
        if (
            self.best_checkpoint != incumbent.index
            or float(self.best_energy) != incumbent.energy
        ):
            raise ProtocolViolation("restart result does not preserve its true incumbent")
        best_parameters = _readonly_float64(self.best_parameters)
        if not _same_parameter_bytes(best_parameters, incumbent.parameters):
            raise ProtocolViolation("best_parameters do not match best_checkpoint")
        if float(self.terminal_energy) != points[-1].energy:
            raise ProtocolViolation("terminal_energy does not match the trace")
        updates = _require_plain_int(
            self.executed_updates, field_name="executed_updates", minimum=1
        )
        if len(points) != updates + 1:
            raise ProtocolViolation("trace must contain initial plus every iterate")
        self.calls.validate()
        if self.calls.restart_count != 1:
            raise ProtocolViolation("per-restart call schema must have restart_count=1")
        if self.calls.optimizer != self.optimizer:
            raise ProtocolViolation(
                "per-restart call schema optimizer disagrees with result"
            )
        if self.calls.executed_update_steps != updates:
            raise ProtocolViolation(
                "per-restart call count does not match executed_updates"
            )
        if self.calls.selected_point_replay_calls != 0:
            raise ProtocolViolation(
                "selected-point replay is a protocol-level call, not per restart"
            )
        object.__setattr__(self, "checkpoints", points)
        object.__setattr__(self, "best_parameters", best_parameters)
        object.__setattr__(self, "best_energy", incumbent.energy)
        object.__setattr__(self, "executed_updates", updates)

    @property
    def initial_energy(self) -> float:
        return self.checkpoints[0].energy

    @property
    def trace_energies(self) -> tuple[float, ...]:
        return tuple(point.energy for point in self.checkpoints)


def select_restart_incumbent(results: Sequence[RestartResult]) -> RestartResult:
    """Select energy, then lower restart index, then earlier checkpoint."""

    records = tuple(results)
    if not records:
        raise ProtocolViolation("at least one restart result is required")
    indices = tuple(record.restart_index for record in records)
    if len(set(indices)) != len(indices):
        raise ProtocolViolation("restart result indices must be unique")
    return min(
        records,
        key=lambda record: (
            record.best_energy,
            record.restart_index,
            record.best_checkpoint,
        ),
    )


@dataclass(frozen=True)
class ProtocolResult:
    optimizer: Literal["Adam", "SPSA"] | str
    restarts: tuple[RestartResult, RestartResult, RestartResult]
    selected_restart: int
    selected_checkpoint: int
    selected_energy: float
    selected_parameters: np.ndarray = field(repr=False, compare=False)
    selected_replay_energy: float
    calls: OptimizerCallCounts

    def __post_init__(self) -> None:
        records = tuple(self.restarts)
        if len(records) != RESTART_COUNT:
            raise ProtocolViolation("protocol result must contain exactly three restarts")
        if tuple(record.restart_index for record in records) != tuple(
            range(RESTART_COUNT)
        ):
            raise ProtocolViolation(
                "protocol restart results must be ordered exactly as 0, 1, 2"
            )
        if any(record.optimizer != self.optimizer for record in records):
            raise ProtocolViolation(
                "every restart result must use the protocol optimizer"
            )
        selected = select_restart_incumbent(records)
        if (
            self.selected_restart != selected.restart_index
            or self.selected_checkpoint != selected.best_checkpoint
            or float(self.selected_energy) != selected.best_energy
        ):
            raise ProtocolViolation("protocol result selection violates incumbent ties")
        parameters = _readonly_float64(self.selected_parameters)
        if not _same_parameter_bytes(parameters, selected.best_parameters):
            raise ProtocolViolation("selected parameters disagree with selected restart")
        direction_counts = {
            record.calls.calibration_directions_per_restart for record in records
        }
        if len(direction_counts) != 1:
            raise ProtocolViolation(
                "all restarts must report one calibration-direction count"
            )
        expected_calls = OptimizerCallCounts(
            optimizer=self.optimizer,
            restart_count=RESTART_COUNT,
            executed_update_steps=sum(
                record.calls.executed_update_steps for record in records
            ),
            calibration_directions_per_restart=direction_counts.pop(),
            initial_objective_calls=sum(
                record.calls.initial_objective_calls for record in records
            ),
            calibration_perturbation_calls=sum(
                record.calls.calibration_perturbation_calls for record in records
            ),
            update_perturbation_calls=sum(
                record.calls.update_perturbation_calls for record in records
            ),
            central_iterate_objective_calls=sum(
                record.calls.central_iterate_objective_calls for record in records
            ),
            gradient_forward_objective_calls=sum(
                record.calls.gradient_forward_objective_calls for record in records
            ),
            terminal_objective_calls=sum(
                record.calls.terminal_objective_calls for record in records
            ),
            selected_point_replay_calls=1,
            exact_gradient_calls=sum(
                record.calls.exact_gradient_calls for record in records
            ),
        ).validate()
        if self.calls != expected_calls:
            raise ProtocolViolation(
                "protocol call counts do not reconcile with restart counts plus "
                "exactly one selected-point replay"
            )
        object.__setattr__(self, "restarts", records)
        object.__setattr__(self, "selected_parameters", parameters)
        object.__setattr__(
            self,
            "selected_replay_energy",
            _finite_energy(self.selected_replay_energy, where="selected replay"),
        )
        self.calls.validate()


def _validate_replay(expected: float, replay: float, tolerance: float) -> None:
    if abs(float(expected) - float(replay)) > float(tolerance):
        raise ProtocolViolation(
            f"selected-point energy replay mismatch: expected {expected}, observed {replay}"
        )


def _run_adam_restart(
    objective: EnergyFunction,
    value_and_gradient: ValueAndGradientFunction,
    initializer: RestartInitializer,
    settings: AdamSettings,
    *,
    spectral_range: float,
) -> RestartResult:
    """Run one standard Adam restart against caller-supplied CPU/GPU callables.

    The initial objective is evaluated separately.  Each executed update has
    one value-and-exact-gradient forward call, and the terminal iterate is
    evaluated once.  No iterate is wrapped or clipped.
    """

    if initializer.context.optimizer != "Adam":
        raise ProtocolViolation("Adam restart requires an Adam RNG context")
    expected_layout = _method_layout_for_context(initializer.context)
    _validate_random_initializer(
        initializer,
        expected_context=initializer.context,
        expected_layout=expected_layout,
        expected_index=initializer.restart_index,
    )
    _validate_context_arm_matches_settings(initializer.context, settings)
    supplied_range = float(spectral_range)
    if not math.isfinite(supplied_range) or supplied_range < 0.0:
        raise ProtocolViolation("spectral_range must be finite and nonnegative")
    energy_range = max(supplied_range, MIN_SPECTRAL_RANGE)
    theta = np.array(initializer.parameters, dtype=np.float64, copy=True)
    initial_energy = _call_objective(
        objective, theta, where="Adam initial objective"
    )
    checkpoints: list[Checkpoint] = [Checkpoint(0, initial_energy, theta)]
    first_forward_energy: float | None = None
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    significant_best = initial_energy
    last_significant_checkpoint = 0
    improvement_tolerance = settings.improvement_tolerance_fraction * energy_range
    executed = 0

    for update_index in range(1, settings.maximum_steps + 1):
        value, gradient = _call_value_and_gradient(
            value_and_gradient, theta, where="Adam gradient forward"
        )
        if gradient.shape != theta.shape or not np.all(np.isfinite(gradient)):
            raise ProtocolViolation("Adam gradient has wrong shape or non-finite values")

        checkpoint_index = update_index - 1
        if checkpoint_index == 0:
            first_forward_energy = value
            _validate_replay(
                initial_energy, value, settings.replay_absolute_tolerance
            )
        else:
            checkpoints.append(Checkpoint(checkpoint_index, value, theta))
            if value < significant_best - improvement_tolerance:
                significant_best = value
                last_significant_checkpoint = checkpoint_index

        m = settings.beta1 * m + (1.0 - settings.beta1) * gradient
        v = settings.beta2 * v + (1.0 - settings.beta2) * np.square(gradient)
        m_hat = m / (1.0 - settings.beta1**update_index)
        v_hat = v / (1.0 - settings.beta2**update_index)
        theta = theta - settings.learning_rate * m_hat / (
            np.sqrt(v_hat) + settings.epsilon
        )
        executed = update_index

        # The current gradient has already been used, so the update is retained;
        # this preserves exact-gradient-calls == executed update steps.
        if (
            checkpoint_index > 0
            and checkpoint_index - last_significant_checkpoint >= settings.patience
        ):
            break

    terminal_energy = _call_objective(
        objective, theta, where="Adam terminal objective"
    )
    checkpoints.append(Checkpoint(executed, terminal_energy, theta))
    incumbent = select_incumbent(checkpoints)
    counts = expected_adam_call_counts(
        (executed,), restarts=1, include_selected_replay=False
    )
    metadata = {
        "settings": settings.to_contract_schema(),
        "first_gradient_forward_energy": first_forward_energy,
        "improvement_tolerance": improvement_tolerance,
        "angle_wrapping": False,
    }
    return RestartResult(
        optimizer="Adam",
        restart_index=initializer.restart_index,
        initializer=initializer,
        checkpoints=tuple(checkpoints),
        best_checkpoint=incumbent.index,
        best_energy=incumbent.energy,
        best_parameters=incumbent.parameters,
        terminal_energy=terminal_energy,
        executed_updates=executed,
        calls=counts,
        metadata=metadata,
    )


def run_adam_protocol(
    objective: EnergyFunction,
    value_and_gradient: ValueAndGradientFunction,
    initializers: Sequence[RestartInitializer],
    *,
    expected_context: RestartContext,
    settings: AdamSettings,
    spectral_range: float,
) -> ProtocolResult:
    """Run exactly three Adam restarts sequentially and replay the selected point."""

    expected_layout = _method_layout_for_context(expected_context)
    _validate_context_arm_matches_settings(expected_context, settings)
    records = validate_restart_batch(
        initializers,
        expected_context=expected_context,
        expected_layout=expected_layout,
    )
    if expected_context.optimizer != "Adam":
        raise ProtocolViolation("run_adam_protocol requires expected optimizer Adam")
    results = tuple(
        _run_adam_restart(
            objective,
            value_and_gradient,
            initializer,
            settings,
            spectral_range=spectral_range,
        )
        for initializer in records
    )
    selected = select_restart_incumbent(results)
    replay = _call_objective(
        objective,
        selected.best_parameters,
        where="Adam selected-point replay",
    )
    _validate_replay(
        selected.best_energy, replay, settings.replay_absolute_tolerance
    )
    counts = expected_adam_call_counts(
        tuple(result.executed_updates for result in results),
        include_selected_replay=True,
    )
    return ProtocolResult(
        optimizer="Adam",
        restarts=results,  # type: ignore[arg-type]
        selected_restart=selected.restart_index,
        selected_checkpoint=selected.best_checkpoint,
        selected_energy=selected.best_energy,
        selected_parameters=selected.best_parameters,
        selected_replay_energy=replay,
        calls=counts,
    )


def _run_spsa_restart(
    objective: EnergyFunction,
    initializer: RestartInitializer,
    settings: SPSASettings,
) -> RestartResult:
    """Run one two-sided SPSA restart with explicit updated-central checkpoints."""

    if initializer.context.optimizer != "SPSA":
        raise ProtocolViolation("SPSA restart requires an SPSA RNG context")
    expected_layout = _method_layout_for_context(initializer.context)
    _validate_random_initializer(
        initializer,
        expected_context=initializer.context,
        expected_layout=expected_layout,
        expected_index=initializer.restart_index,
    )
    _validate_context_arm_matches_settings(initializer.context, settings)
    theta = np.array(initializer.parameters, dtype=np.float64, copy=True)
    initial_energy = _call_objective(
        objective, theta, where="SPSA initial objective"
    )
    checkpoints: list[Checkpoint] = [Checkpoint(0, initial_energy, theta)]

    calibration_gradients: list[np.ndarray] = []
    for direction_index in range(settings.calibration_directions):
        delta = spsa_rademacher_direction(
            initializer,
            kind="calibration",
            iteration_index=0,
            direction_index=direction_index,
        )
        plus = _call_objective(
            objective,
            theta + settings.c * delta,
            where="SPSA calibration plus",
        )
        minus = _call_objective(
            objective,
            theta - settings.c * delta,
            where="SPSA calibration minus",
        )
        calibration_gradients.append(
            ((plus - minus) / (2.0 * settings.c)) * delta
        )
    stacked = np.stack(calibration_gradients, axis=0)
    gradient_rms = float(np.sqrt(np.mean(np.square(stacked), dtype=np.float64)))
    a = settings.calibrated_a(gradient_rms)

    clipped_update_count = 0
    for update_index in range(1, settings.updates + 1):
        c_k = settings.c / (update_index**settings.gamma)
        a_k = a / ((settings.A + update_index) ** settings.alpha)
        delta = spsa_rademacher_direction(
            initializer,
            kind="update",
            iteration_index=update_index,
            direction_index=0,
        )
        plus = _call_objective(
            objective,
            theta + c_k * delta,
            where="SPSA update plus",
        )
        minus = _call_objective(
            objective,
            theta - c_k * delta,
            where="SPSA update minus",
        )
        gradient = ((plus - minus) / (2.0 * c_k)) * delta
        parameter_update = a_k * gradient
        update_rms = float(np.sqrt(np.mean(np.square(parameter_update))))
        if update_rms > settings.max_update_rms:
            parameter_update *= settings.max_update_rms / update_rms
            clipped_update_count += 1
        theta = theta - parameter_update
        # This central evaluation is mandatory and is the only post-update
        # checkpoint; neither plus/minus nor calibration probes are eligible.
        central_energy = _call_objective(
            objective, theta, where="SPSA updated central iterate"
        )
        checkpoints.append(Checkpoint(update_index, central_energy, theta))

    incumbent = select_incumbent(checkpoints)
    counts = expected_spsa_call_counts(
        settings.updates,
        restarts=1,
        calibration_directions=settings.calibration_directions,
        include_selected_replay=False,
    )
    metadata = {
        "settings": settings.to_contract_schema(),
        "calibration_gradient_rms": gradient_rms,
        "calibrated_a": a,
        "clipped_update_count": clipped_update_count,
        "calibration_and_perturbation_probes_checkpoint_eligible": False,
    }
    return RestartResult(
        optimizer="SPSA",
        restart_index=initializer.restart_index,
        initializer=initializer,
        checkpoints=tuple(checkpoints),
        best_checkpoint=incumbent.index,
        best_energy=incumbent.energy,
        best_parameters=incumbent.parameters,
        terminal_energy=checkpoints[-1].energy,
        executed_updates=settings.updates,
        calls=counts,
        metadata=metadata,
    )


def run_spsa_protocol(
    objective: EnergyFunction,
    initializers: Sequence[RestartInitializer],
    *,
    expected_context: RestartContext,
    settings: SPSASettings,
) -> ProtocolResult:
    """Run three SPSA restarts sequentially, then one selected-point replay."""

    expected_layout = _method_layout_for_context(expected_context)
    _validate_context_arm_matches_settings(expected_context, settings)
    records = validate_restart_batch(
        initializers,
        expected_context=expected_context,
        expected_layout=expected_layout,
    )
    if expected_context.optimizer != "SPSA":
        raise ProtocolViolation("run_spsa_protocol requires expected optimizer SPSA")
    results = tuple(
        _run_spsa_restart(objective, initializer, settings)
        for initializer in records
    )
    selected = select_restart_incumbent(results)
    replay = _call_objective(
        objective,
        selected.best_parameters,
        where="SPSA selected-point replay",
    )
    _validate_replay(
        selected.best_energy, replay, settings.replay_absolute_tolerance
    )
    counts = expected_spsa_call_counts(
        settings.updates,
        restarts=RESTART_COUNT,
        calibration_directions=settings.calibration_directions,
        include_selected_replay=True,
    )
    return ProtocolResult(
        optimizer="SPSA",
        restarts=results,  # type: ignore[arg-type]
        selected_restart=selected.restart_index,
        selected_checkpoint=selected.best_checkpoint,
        selected_energy=selected.best_energy,
        selected_parameters=selected.best_parameters,
        selected_replay_energy=replay,
        calls=counts,
    )


@dataclass(frozen=True)
class BudgetTailDiagnostic:
    budget: int
    tail_start_checkpoint: int
    late_checkpoint_threshold: int
    selected_checkpoint: int
    best_through_tail_start: float
    best_through_budget: float
    normalized_tail_drop: float
    late_checkpoint: bool
    base_budget_limited: bool


def late_checkpoint_threshold(budget: int) -> int:
    B = _require_plain_int(budget, field_name="budget", minimum=1)
    return math.ceil(0.90 * B)


def tail_start_checkpoint(budget: int) -> int:
    B = _require_plain_int(budget, field_name="budget", minimum=1)
    return math.floor(0.80 * B)


def budget_tail_diagnostic(
    checkpoint_energies: Sequence[float],
    *,
    selected_checkpoint: int,
    budget: int,
    h_min: float,
    h_max: float,
) -> BudgetTailDiagnostic:
    """Apply the exact final-10%/final-20% energy-only saturation rule.

    A short Adam trace is extended with its final best-so-far value through the
    requested budget.  Its actual selected checkpoint is not moved, so early
    stopping cannot manufacture a late checkpoint.
    """

    B = _require_plain_int(budget, field_name="budget", minimum=1)
    energies = tuple(
        _finite_energy(value, where="budget-tail checkpoint")
        for value in checkpoint_energies
    )
    if not energies:
        raise ProtocolViolation("budget-tail diagnostics require checkpoint zero")
    if len(energies) > B + 1:
        raise ProtocolViolation("checkpoint trace exceeds the requested budget")
    selected = _require_plain_int(
        selected_checkpoint, field_name="selected_checkpoint", minimum=0
    )
    if selected >= len(energies):
        raise ProtocolViolation("selected_checkpoint is absent from the actual trace")
    minimum = float(h_min)
    maximum = float(h_max)
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum:
        raise ProtocolViolation("Hamiltonian extrema must be finite with h_max >= h_min")
    denominator = max(maximum - minimum, MIN_SPECTRAL_RANGE)
    tail_start = tail_start_checkpoint(B)
    start_observed_index = min(tail_start, len(energies) - 1)
    best_at_start = min(energies[: start_observed_index + 1])
    best_at_end = min(energies)
    normalized_drop = (best_at_start - best_at_end) / denominator
    late_threshold = late_checkpoint_threshold(B)
    late = selected >= late_threshold
    limited = late and normalized_drop >= 0.01
    return BudgetTailDiagnostic(
        budget=B,
        tail_start_checkpoint=tail_start,
        late_checkpoint_threshold=late_threshold,
        selected_checkpoint=selected,
        best_through_tail_start=best_at_start,
        best_through_budget=best_at_end,
        normalized_tail_drop=normalized_drop,
        late_checkpoint=late,
        base_budget_limited=limited,
    )


def method_is_base_budget_limited(
    seed_diagnostics: Sequence[BudgetTailDiagnostic],
) -> bool:
    """Both and exactly two calibration seeds must satisfy both tail tests."""

    diagnostics = tuple(seed_diagnostics)
    if len(diagnostics) != 2:
        raise ProtocolViolation(
            "method budget trigger requires exactly two calibration seeds"
        )
    if diagnostics[0].budget != diagnostics[1].budget:
        raise ProtocolViolation("calibration seed diagnostics must use one base budget")
    return all(diagnostic.base_budget_limited for diagnostic in diagnostics)


def _validate_four_method_mapping(
    values: Mapping[str, Any], *, field_name: str
) -> None:
    observed = set(values)
    expected = set(OPTIMIZED_METHODS)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProtocolViolation(
            f"{field_name} must cover exactly four optimized methods; "
            f"missing={missing}, extra={extra}"
        )


def optimizer_double_audit_trigger(
    method_limited: Mapping[str, bool],
) -> bool:
    """Trigger the one doubled audit iff at least three of four methods qualify."""

    _validate_four_method_mapping(method_limited, field_name="method_limited")
    if any(not isinstance(value, (bool, np.bool_)) for value in method_limited.values()):
        raise ProtocolViolation("method_limited values must be boolean")
    return sum(bool(value) for value in method_limited.values()) >= 3


@dataclass(frozen=True)
class BudgetEnergyPair:
    """One calibration seed's base/doubled energy-only comparison."""

    base_energy: float
    doubled_energy: float
    h_min: float
    h_max: float

    def __post_init__(self) -> None:
        for name in ("base_energy", "doubled_energy", "h_min", "h_max"):
            value = _finite_energy(getattr(self, name), where=name)
            object.__setattr__(self, name, value)
        if self.h_max < self.h_min:
            raise ProtocolViolation("BudgetEnergyPair requires h_max >= h_min")

    @property
    def normalized_improvement(self) -> float:
        return (self.base_energy - self.doubled_energy) / max(
            self.h_max - self.h_min, MIN_SPECTRAL_RANGE
        )


def method_doubled_budget_improvement(
    calibration_seed_pairs: Sequence[BudgetEnergyPair],
) -> float:
    """Mean normalized expected-energy improvement over exactly two seeds."""

    pairs = tuple(calibration_seed_pairs)
    if len(pairs) != 2:
        raise ProtocolViolation(
            "doubled-budget adoption requires exactly two calibration seed pairs"
        )
    return math.fsum(pair.normalized_improvement for pair in pairs) / 2.0


def adopt_doubled_budget(method_improvements: Mapping[str, float]) -> bool:
    """Adopt 2B uniformly iff at least three methods improve by at least 2%."""

    _validate_four_method_mapping(
        method_improvements, field_name="method_improvements"
    )
    values: list[float] = []
    for method in OPTIMIZED_METHODS:
        value = float(method_improvements[method])
        if not math.isfinite(value):
            raise ProtocolViolation(f"non-finite improvement for method {method}")
        values.append(value)
    return sum(value >= 0.02 for value in values) >= 3


def adopted_budget(
    base_budget: int,
    method_improvements: Mapping[str, float],
) -> int:
    """Return B or 2B from the uniform four-method adoption rule."""

    B = _require_plain_int(base_budget, field_name="base_budget", minimum=1)
    return 2 * B if adopt_doubled_budget(method_improvements) else B


__all__ = [
    "ADAM_BASE_STEPS",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPSILON",
    "ADAM_IMPROVEMENT_TOLERANCE_FRACTION",
    "ADAM_LEARNING_RATE_ARMS",
    "ADAM_PATIENCE",
    "AdamSettings",
    "BudgetEnergyPair",
    "BudgetTailDiagnostic",
    "CAMPAIGN_ID",
    "Checkpoint",
    "LEAN_EXECUTION_PLAN_SHA256",
    "OPTIMIZED_METHODS",
    "OptimizerCallCounts",
    "PHASE_TAGS",
    "PROHIBITED_RESTART_ROLES",
    "ProtocolResult",
    "ProtocolViolation",
    "RESTART_COUNT",
    "RESTART_ROLE",
    "RESTART_ROLES_EXACT",
    "RNG_KDF_VERSION",
    "RestartContext",
    "RestartInitializer",
    "RestartResult",
    "SPSA_BASE_UPDATES",
    "SPSA_A",
    "SPSA_CALIBRATION_DIRECTIONS",
    "SPSA_C_ARMS",
    "SPSASettings",
    "ZeroAngleDiagnostic",
    "adam_count_schema",
    "adopt_doubled_budget",
    "adopted_budget",
    "budget_tail_diagnostic",
    "canonical_json_bytes",
    "derive_uniform_angle",
    "evaluate_zero_angle_diagnostic",
    "expected_adam_call_counts",
    "expected_spsa_call_counts",
    "generate_three_random_initializers",
    "late_checkpoint_threshold",
    "make_random_initializer",
    "method_doubled_budget_improvement",
    "method_is_base_budget_limited",
    "normalize_parameter_layout",
    "optimizer_double_audit_trigger",
    "parameter_layout_for_method",
    "parameter_sha256",
    "run_adam_protocol",
    "run_spsa_protocol",
    "select_incumbent",
    "select_restart_incumbent",
    "sha256_uniform_angle",
    "spsa_count_schema",
    "spsa_objective_call_count",
    "spsa_rademacher_direction",
    "stage1_adam_settings",
    "standard_adam_settings",
    "standard_spsa_settings",
    "tail_start_checkpoint",
    "validate_restart_batch",
    "validate_three_random_restarts",
]
