"""Independent Adam protocol for the frozen reviewer-appendix campaign.

This implementation intentionally does not import or wrap the predecessor
campaign's restart or optimizer machinery.  It owns the SHA-256 initializer
domain, constructs exactly three starts internally, creates a fresh Adam
instance for each start, and executes the requested budget without early
stopping.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .provenance import (
    CAMPAIGN_ID,
    PLAN_SHA256,
    PREDECESSOR_SOURCE_MANIFEST_SHA256,
    canonical_json_bytes,
)


RESTART_COUNT = 3
RESTART_ROLE = "independent_random"
ALLOWED_BUDGETS = (400, 800, 1600)
KDF_SCHEMA = "lp-qaoa-reviewer-appendix-sha256-component-v3"
FROZEN_SETTING = "d4_1of2_r6_1of16"
FROZEN_LEARNING_RATE = 0.035
FROZEN_BETA1 = 0.9
FROZEN_BETA2 = 0.999
FROZEN_EPSILON = 1e-8
FROZEN_ARM = "adam_lr_0x1.1eb851eb851ecp-5"

Objective = Callable[[torch.Tensor], torch.Tensor]
ParameterLayout = tuple[tuple[str, int], ...]


class AdamProtocolError(ValueError):
    """Raised when a run would violate the frozen Adam protocol."""


def _plain_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise AdamProtocolError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise AdamProtocolError(f"{name} must be at least {minimum}")
    return result


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdamProtocolError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class InitializerDomain:
    """Every initializer-domain field except restart/component indices.

    The optimizer budget is structurally absent.  Reusing this value for
    400-, 800-, and 1600-step runs therefore yields byte-identical starts.
    """

    phase: str
    method: str
    N: int
    problem_seed: int
    depth: int
    setting: str = FROZEN_SETTING
    arm: str = FROZEN_ARM

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _nonempty_text(self.phase, name="phase"))
        object.__setattr__(self, "method", _nonempty_text(self.method, name="method"))
        n_value = _plain_int(self.N, name="N", minimum=1)
        if n_value not in (25, 30):
            raise AdamProtocolError("N must be exactly 25 or 30")
        object.__setattr__(self, "N", n_value)
        object.__setattr__(
            self,
            "problem_seed",
            _plain_int(self.problem_seed, name="problem_seed", minimum=0),
        )
        object.__setattr__(
            self, "depth", _plain_int(self.depth, name="depth", minimum=1)
        )
        setting = _nonempty_text(self.setting, name="setting")
        if setting != FROZEN_SETTING:
            raise AdamProtocolError(
                f"setting must be exactly {FROZEN_SETTING!r}"
            )
        object.__setattr__(self, "setting", setting)
        arm = _nonempty_text(self.arm, name="arm")
        if arm != FROZEN_ARM:
            raise AdamProtocolError(f"arm must be exactly {FROZEN_ARM!r}")
        object.__setattr__(self, "arm", arm)

    def kdf_record(
        self,
        *,
        parameter_layout: ParameterLayout,
        restart_index: int,
        parameter_family: str,
        component_index: int,
    ) -> dict[str, object]:
        layout = normalize_parameter_layout(parameter_layout)
        restart = _plain_int(restart_index, name="restart_index", minimum=0)
        if restart >= RESTART_COUNT:
            raise AdamProtocolError("restart_index must be 0, 1, or 2")
        family = _nonempty_text(parameter_family, name="parameter_family")
        family_sizes = dict(layout)
        if family not in family_sizes:
            raise AdamProtocolError("parameter_family is absent from parameter_layout")
        component = _plain_int(component_index, name="component_index", minimum=0)
        if component >= family_sizes[family]:
            raise AdamProtocolError("component_index is outside parameter_family")
        return {
            "schema": KDF_SCHEMA,
            "campaign": CAMPAIGN_ID,
            "plan_sha256": PLAN_SHA256,
            "predecessor_source_manifest_sha256": (
                PREDECESSOR_SOURCE_MANIFEST_SHA256
            ),
            "optimizer": "Adam",
            "phase": self.phase,
            "setting": self.setting,
            "method": self.method,
            "N": self.N,
            "problem_seed": self.problem_seed,
            "depth": self.depth,
            "arm": self.arm,
            "parameter_layout": [list(item) for item in layout],
            "restart_index": restart,
            "parameter_family": family,
            "component_index": component,
        }


@dataclass(frozen=True)
class AdamConfig:
    maximum_steps: int
    learning_rate: float = FROZEN_LEARNING_RATE
    beta1: float = FROZEN_BETA1
    beta2: float = FROZEN_BETA2
    epsilon: float = FROZEN_EPSILON

    def __post_init__(self) -> None:
        steps = _plain_int(self.maximum_steps, name="maximum_steps", minimum=1)
        if steps not in ALLOWED_BUDGETS:
            raise AdamProtocolError(
                f"maximum_steps must be one of {ALLOWED_BUDGETS}"
            )
        object.__setattr__(self, "maximum_steps", steps)
        for name in ("learning_rate", "beta1", "beta2", "epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise AdamProtocolError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        frozen = {
            "learning_rate": FROZEN_LEARNING_RATE,
            "beta1": FROZEN_BETA1,
            "beta2": FROZEN_BETA2,
            "epsilon": FROZEN_EPSILON,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise AdamProtocolError(
                    f"{name} must equal the frozen value {expected!r}"
                )


def normalize_parameter_layout(layout: object) -> ParameterLayout:
    """Validate the exact ordered ``((family, size), ...)`` layout."""

    if not isinstance(layout, tuple) or not layout:
        raise AdamProtocolError(
            "parameter_layout must be a nonempty tuple of (family, size) tuples"
        )
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in layout:
        if not isinstance(item, tuple) or len(item) != 2:
            raise AdamProtocolError(
                "each parameter_layout entry must be a (family, size) tuple"
            )
        family = _nonempty_text(item[0], name="parameter family")
        if family in seen:
            raise AdamProtocolError(f"duplicate parameter family {family!r}")
        size = _plain_int(item[1], name=f"size of {family}", minimum=1)
        seen.add(family)
        normalized.append((family, size))
    return tuple(normalized)


def _canonical_float64_bytes(values: np.ndarray | tuple[float, ...]) -> bytes:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise AdamProtocolError("parameter vectors must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise AdamProtocolError("parameter vectors must be finite")
    return np.ascontiguousarray(array.astype("<f8", copy=False)).tobytes(order="C")


def parameter_sha256(values: np.ndarray | tuple[float, ...]) -> str:
    return hashlib.sha256(_canonical_float64_bytes(values)).hexdigest()


def _uniform_angle(record: dict[str, object]) -> float:
    digest = hashlib.sha256(canonical_json_bytes(record)).digest()
    z_value = int.from_bytes(digest[:8], "big") >> 11
    unit = (z_value + 0.5) / float(1 << 53)
    return -math.pi + (2.0 * math.pi * unit)


def derive_three_initializers(
    domain: InitializerDomain,
    parameter_layout: ParameterLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the only optimizer-eligible starts: three KDF vectors."""

    if not isinstance(domain, InitializerDomain):
        raise AdamProtocolError("domain must be an InitializerDomain")
    layout = normalize_parameter_layout(parameter_layout)
    starts: list[np.ndarray] = []
    hashes: set[str] = set()
    for restart in range(RESTART_COUNT):
        vector = np.asarray(
            [
                _uniform_angle(
                    domain.kdf_record(
                        parameter_layout=layout,
                        restart_index=restart,
                        parameter_family=family,
                        component_index=component,
                    )
                )
                for family, size in layout
                for component in range(size)
            ],
            dtype=np.float64,
        )
        digest = parameter_sha256(vector)
        if digest in hashes:
            raise AdamProtocolError("KDF restart collision detected")
        hashes.add(digest)
        vector.setflags(write=False)
        starts.append(vector)
    if len(starts) != RESTART_COUNT:
        raise AdamProtocolError("exactly three initializers are required")
    return starts[0], starts[1], starts[2]


@dataclass(frozen=True)
class TracePoint:
    checkpoint: int
    energy: float
    parameters: tuple[float, ...]
    parameter_sha256: str
    gradient: tuple[float, ...]
    gradient_norm: float


@dataclass(frozen=True)
class RestartResult:
    restart_index: int
    role: str
    initializer: tuple[float, ...]
    initializer_sha256: str
    trace: tuple[TracePoint, ...]
    selected_checkpoint: int
    selected_energy: float
    selected_parameters: tuple[float, ...]
    selected_parameter_sha256: str
    selected_gradient: tuple[float, ...]
    selected_gradient_norm: float


@dataclass(frozen=True)
class AdamResult:
    domain: InitializerDomain
    config: AdamConfig
    parameter_layout: ParameterLayout
    restarts: tuple[RestartResult, RestartResult, RestartResult]
    objective_evaluations: int
    selected_restart: int
    selected_checkpoint: int
    selected_energy: float
    selected_parameters: tuple[float, ...]
    selected_parameter_sha256: str
    selected_gradient: tuple[float, ...]
    selected_gradient_norm: float


def _evaluate(
    objective: Objective,
    parameters: torch.Tensor,
    *,
    checkpoint: int,
) -> TracePoint:
    output = objective(parameters)
    if not isinstance(output, torch.Tensor) or output.numel() != 1:
        raise AdamProtocolError("objective must return one scalar torch.Tensor")
    if torch.is_complex(output):
        raise AdamProtocolError("objective energy must be real")
    if output.dtype != torch.float64:
        raise AdamProtocolError("objective energy must have dtype torch.float64")
    scalar = output.reshape(())
    if not scalar.requires_grad:
        raise AdamProtocolError(
            "objective energy must depend differentiably on the parameters"
        )
    try:
        gradient_tensor = torch.autograd.grad(
            scalar,
            parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
    except RuntimeError as exc:
        raise AdamProtocolError(
            "objective energy is detached from or does not use the parameters"
        ) from exc
    if gradient_tensor is None:
        raise AdamProtocolError("objective did not produce a parameter gradient")
    if gradient_tensor.dtype != torch.float64:
        raise AdamProtocolError("objective gradient must have dtype torch.float64")

    energy = float(scalar.detach().cpu().item())
    gradient_array = np.asarray(
        gradient_tensor.detach().cpu().numpy(), dtype=np.float64
    ).reshape(-1)
    parameter_array = np.asarray(
        parameters.detach().cpu().numpy(), dtype=np.float64
    ).reshape(-1)
    if not math.isfinite(energy):
        raise AdamProtocolError("objective returned a non-finite energy")
    if not np.all(np.isfinite(gradient_array)):
        raise AdamProtocolError("objective returned a non-finite gradient")
    if not np.all(np.isfinite(parameter_array)):
        raise AdamProtocolError("Adam produced non-finite parameters")
    return TracePoint(
        checkpoint=checkpoint,
        energy=energy,
        parameters=tuple(float(value) for value in parameter_array),
        parameter_sha256=parameter_sha256(parameter_array),
        gradient=tuple(float(value) for value in gradient_array),
        gradient_norm=float(np.linalg.norm(gradient_array)),
    )


def _run_restart(
    objective: Objective,
    *,
    initializer: np.ndarray,
    restart_index: int,
    config: AdamConfig,
    device: str | torch.device,
) -> RestartResult:
    # A new leaf tensor and optimizer are constructed for every restart.  No
    # moments, checkpoints, or parameter objects cross this boundary.
    parameters = torch.tensor(
        np.array(initializer, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [parameters],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )

    trace: list[TracePoint] = [_evaluate(objective, parameters, checkpoint=0)]
    for checkpoint in range(1, config.maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        parameters.grad = torch.tensor(
            trace[-1].gradient,
            dtype=torch.float64,
            device=parameters.device,
        )
        optimizer.step()
        trace.append(_evaluate(objective, parameters, checkpoint=checkpoint))

    # Strict comparison retains the earlier checkpoint on an exact tie.
    selected = min(trace, key=lambda point: (point.energy, point.checkpoint))
    initial = np.asarray(initializer, dtype=np.float64)
    return RestartResult(
        restart_index=restart_index,
        role=RESTART_ROLE,
        initializer=tuple(float(value) for value in initial),
        initializer_sha256=parameter_sha256(initial),
        trace=tuple(trace),
        selected_checkpoint=selected.checkpoint,
        selected_energy=selected.energy,
        selected_parameters=selected.parameters,
        selected_parameter_sha256=selected.parameter_sha256,
        selected_gradient=selected.gradient,
        selected_gradient_norm=selected.gradient_norm,
    )


def run_adam(
    objective: Objective,
    *,
    domain: InitializerDomain,
    parameter_layout: ParameterLayout,
    config: AdamConfig,
    device: str | torch.device = "cpu",
) -> AdamResult:
    """Run exactly three sequential, independent Adam restarts.

    There is intentionally no supplied-initializer parameter.  Every run
    derives checkpoint zero internally from the frozen KDF domain.
    """

    if not callable(objective):
        raise AdamProtocolError("objective must be callable")
    if not isinstance(domain, InitializerDomain):
        raise AdamProtocolError("domain must be an InitializerDomain")
    if not isinstance(config, AdamConfig):
        raise AdamProtocolError("config must be an AdamConfig")
    layout = normalize_parameter_layout(parameter_layout)
    initializers = derive_three_initializers(domain, layout)
    if len(initializers) != RESTART_COUNT:
        raise AdamProtocolError("exactly three KDF starts are mandatory")

    restart_results = tuple(
        _run_restart(
            objective,
            initializer=initializer,
            restart_index=restart_index,
            config=config,
            device=device,
        )
        for restart_index, initializer in enumerate(initializers)
    )
    if len(restart_results) != RESTART_COUNT:
        raise AdamProtocolError("exactly three sequential restarts are mandatory")

    selected_restart_result = min(
        restart_results,
        key=lambda restart: (
            restart.selected_energy,
            restart.selected_checkpoint,
            restart.restart_index,
        ),
    )
    return AdamResult(
        domain=domain,
        config=config,
        parameter_layout=layout,
        restarts=(restart_results[0], restart_results[1], restart_results[2]),
        objective_evaluations=RESTART_COUNT * (config.maximum_steps + 1),
        selected_restart=selected_restart_result.restart_index,
        selected_checkpoint=selected_restart_result.selected_checkpoint,
        selected_energy=selected_restart_result.selected_energy,
        selected_parameters=selected_restart_result.selected_parameters,
        selected_parameter_sha256=(
            selected_restart_result.selected_parameter_sha256
        ),
        selected_gradient=selected_restart_result.selected_gradient,
        selected_gradient_norm=selected_restart_result.selected_gradient_norm,
    )


__all__ = [
    "ALLOWED_BUDGETS",
    "AdamConfig",
    "AdamProtocolError",
    "AdamResult",
    "InitializerDomain",
    "KDF_SCHEMA",
    "FROZEN_SETTING",
    "FROZEN_ARM",
    "ParameterLayout",
    "RESTART_COUNT",
    "RESTART_ROLE",
    "RestartResult",
    "TracePoint",
    "derive_three_initializers",
    "normalize_parameter_layout",
    "parameter_sha256",
    "run_adam",
]
