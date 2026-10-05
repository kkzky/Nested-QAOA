"""Exact reduced-basis dynamics for the corrected BCST campaign.

The optimizer never sees targets.  This module accepts only training
Hamiltonian diagonals, the fixed-cardinality layout, and an optional frozen
Stage-1 state.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


Method = Literal["stage1", "learned_projector", "ordinary_native_direct",
                 "decoupled_native_direct", "same_stage1_state_direct"]


def local_weight_two_patterns(block_size: int) -> np.ndarray:
    if block_size < 3:
        raise ValueError("block size must exceed the fixed weight two")
    patterns = []
    for left in range(block_size):
        for right in range(left + 1, block_size):
            row = np.zeros(block_size, dtype=np.int8)
            row[left] = 1
            row[right] = 1
            patterns.append(row)
    return np.stack(patterns)


def johnson_weight_two_adjacency(block_size: int) -> np.ndarray:
    patterns = local_weight_two_patterns(block_size)
    local_dim = len(patterns)
    adjacency = np.zeros((local_dim, local_dim), dtype=np.float64)
    for left in range(local_dim):
        for right in range(left + 1, local_dim):
            if np.count_nonzero(patterns[left] != patterns[right]) == 2:
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
    return adjacency


@dataclass(frozen=True)
class ReducedProblem:
    layout: tuple[int, ...]
    h_c: np.ndarray
    h_f: np.ndarray
    h_d: np.ndarray
    h_d_obj: np.ndarray
    h_d_conf: np.ndarray
    feasible_mask: np.ndarray

    def __post_init__(self) -> None:
        if len(self.layout) != 5:
            raise ValueError("the frozen campaign requires five blocks")
        dimensions = tuple(math.comb(int(size), 2) for size in self.layout)
        expected = math.prod(dimensions)
        for name in ("h_c", "h_f", "h_d", "h_d_obj", "h_d_conf"):
            values = np.asarray(getattr(self, name))
            if values.shape != (expected,):
                raise ValueError(f"{name} shape {values.shape} != {(expected,)}")
            if values.dtype != np.float64:
                raise ValueError(f"{name} must be float64")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains a nonfinite value")
        feasible = np.asarray(self.feasible_mask)
        if feasible.shape != (expected,) or feasible.dtype != np.bool_:
            raise ValueError("feasible_mask has the wrong shape or dtype")
        for name in ("h_c", "h_f", "h_d"):
            values = np.asarray(getattr(self, name))
            spectral_range = float(values.max() - values.min())
            if not math.isclose(spectral_range, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{name} spectral range is not one: {spectral_range}")
        if not np.allclose(
            self.h_d_obj + self.h_d_conf - self.h_d,
            np.full(expected, (self.h_d_obj + self.h_d_conf - self.h_d)[0]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("decoupled components differ from h_d by more than identity")

    @property
    def N(self) -> int:
        return int(sum(self.layout))

    @property
    def local_dimensions(self) -> tuple[int, ...]:
        return tuple(math.comb(int(size), 2) for size in self.layout)

    @property
    def dimension(self) -> int:
        return math.prod(self.local_dimensions)


def parameter_count(method: Method, depth: int) -> int:
    if depth < 1:
        raise ValueError("depth must be positive")
    if method == "decoupled_native_direct":
        return 3 * int(depth)
    if method in {
        "stage1",
        "learned_projector",
        "ordinary_native_direct",
        "same_stage1_state_direct",
    }:
        return 2 * int(depth)
    raise ValueError(f"unknown method {method!r}")


class ReducedDynamics:
    def __init__(
        self,
        problem: ReducedProblem,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.problem = problem
        self.device = torch.device(device)
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128
        self.h_c = torch.as_tensor(
            problem.h_c, dtype=self.real_dtype, device=self.device
        )
        self.h_f = torch.as_tensor(
            problem.h_f, dtype=self.real_dtype, device=self.device
        )
        self.h_d = torch.as_tensor(
            problem.h_d, dtype=self.real_dtype, device=self.device
        )
        self.h_d_obj = torch.as_tensor(
            problem.h_d_obj, dtype=self.real_dtype, device=self.device
        )
        self.h_d_conf = torch.as_tensor(
            problem.h_d_conf, dtype=self.real_dtype, device=self.device
        )
        self.feasible_mask = torch.as_tensor(
            problem.feasible_mask, dtype=torch.bool, device=self.device
        )
        local_eigensystems = []
        for block_size in problem.layout:
            adjacency = torch.as_tensor(
                johnson_weight_two_adjacency(int(block_size)),
                dtype=self.real_dtype,
                device=self.device,
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(adjacency)
            local_eigensystems.append(
                (eigenvalues, eigenvectors.to(self.complex_dtype))
            )
        self.local_eigensystems = tuple(local_eigensystems)
        self.xy_range = float(2 * (problem.N - 5))
        if self.xy_range <= 0:
            raise ValueError("invalid normalized XY spectral range")
        amplitude = 1.0 / math.sqrt(problem.dimension)
        self.native_state = torch.full(
            (problem.dimension,),
            complex(amplitude, 0.0),
            dtype=self.complex_dtype,
            device=self.device,
        )

    def _normalized(self, state: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(state)
        if not bool(torch.isfinite(norm)) or float(norm.detach().cpu()) <= 0.0:
            raise ValueError("state has invalid norm")
        return state / norm

    def apply_xy(self, state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply exp(-i beta H_XY) exactly."""

        if state.shape != (self.problem.dimension,):
            raise ValueError("state has the wrong reduced-basis shape")
        shaped = state.reshape(self.problem.local_dimensions)
        # H_XY=(B_raw+10I)/(2(N-5)); the identity part is retained so
        # serialized states follow the mathematical convention exactly.
        local_angle = beta / self.xy_range
        for block, (eigenvalues, eigenvectors) in enumerate(
            self.local_eigensystems
        ):
            phases = torch.exp(
                -1j * local_angle.to(self.real_dtype) * eigenvalues
            ).to(self.complex_dtype)
            unitary = eigenvectors @ torch.diag(phases) @ eigenvectors.T.conj()
            moved = torch.movedim(shaped, block, 0)
            original_shape = moved.shape
            moved = unitary @ moved.reshape(original_shape[0], -1)
            shaped = torch.movedim(moved.reshape(original_shape), 0, block)
        global_phase = torch.exp(
            -1j
            * beta.to(self.real_dtype)
            * torch.tensor(10.0 / self.xy_range, dtype=self.real_dtype, device=self.device)
        ).to(self.complex_dtype)
        return global_phase * shaped.reshape(-1)

    def apply_history(
        self,
        state: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        phi = self._normalized(phi)
        phase = torch.exp(-1j * beta.to(self.real_dtype)).to(self.complex_dtype)
        overlap = torch.sum(phi.conj() * state)
        return phase * state + (1.0 - phase) * overlap * phi

    def _split_parameters(
        self,
        method: Method,
        parameters: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, ...]:
        expected = parameter_count(method, depth)
        if parameters.shape != (expected,):
            raise ValueError(
                f"{method} depth {depth} expects {expected} parameters, "
                f"received {tuple(parameters.shape)}"
            )
        if parameters.dtype != self.real_dtype:
            raise ValueError("parameters must be float64")
        if method == "decoupled_native_direct":
            return (
                parameters[:depth],
                parameters[depth : 2 * depth],
                parameters[2 * depth :],
            )
        return parameters[:depth], parameters[depth:]

    def training_diagonal(self, method: Method) -> torch.Tensor:
        if method == "stage1":
            return self.h_c
        if method == "learned_projector":
            return self.h_f
        if method in {
            "ordinary_native_direct",
            "decoupled_native_direct",
            "same_stage1_state_direct",
        }:
            return self.h_d
        raise ValueError(f"unknown method {method!r}")

    def _initial_state(self, method: Method, phi: torch.Tensor | None) -> torch.Tensor:
        if method in {"stage1", "ordinary_native_direct", "decoupled_native_direct"}:
            if phi is not None:
                raise ValueError(f"{method} must not receive a Stage-1 initial state")
            return self.native_state
        if method in {"learned_projector", "same_stage1_state_direct"}:
            if phi is None:
                raise ValueError(f"{method} requires the frozen Stage-1 state")
            if phi.shape != (self.problem.dimension,):
                raise ValueError("Stage-1 state has the wrong shape")
            return self._normalized(phi.to(self.device, dtype=self.complex_dtype))
        raise ValueError(f"unknown method {method!r}")

    def _apply_layer(
        self,
        method: Method,
        state: torch.Tensor,
        parameter_parts: tuple[torch.Tensor, ...],
        layer: int,
        phi: torch.Tensor | None,
    ) -> torch.Tensor:
        if method == "decoupled_native_direct":
            gamma_conf, gamma_obj, betas = parameter_parts
            state = state * torch.exp(
                -1j * gamma_conf[layer] * self.h_d_conf
            ).to(self.complex_dtype)
            state = state * torch.exp(
                -1j * gamma_obj[layer] * self.h_d_obj
            ).to(self.complex_dtype)
            return self.apply_xy(state, betas[layer])
        gammas, betas = parameter_parts
        diagonal = self.training_diagonal(method)
        state = state * torch.exp(-1j * gammas[layer] * diagonal).to(
            self.complex_dtype
        )
        if method == "learned_projector":
            if phi is None:
                raise ValueError("learned projector requires phi")
            return self.apply_history(state, betas[layer], phi)
        return self.apply_xy(state, betas[layer])

    def evolve(
        self,
        method: Method,
        parameters: torch.Tensor,
        *,
        depth: int,
        phi: torch.Tensor | None = None,
        activation_checkpointing: bool = False,
    ) -> torch.Tensor:
        parts = self._split_parameters(method, parameters, depth)
        state = self._initial_state(method, phi).clone()
        if not activation_checkpointing or not parameters.requires_grad:
            for layer in range(depth):
                state = self._apply_layer(method, state, parts, layer, phi)
            return state

        segment_length = max(1, math.ceil(math.sqrt(depth)))
        for start in range(0, depth, segment_length):
            stop = min(depth, start + segment_length)

            def run_segment(
                input_state: torch.Tensor,
                all_parameters: torch.Tensor,
                *,
                segment_start: int = start,
                segment_stop: int = stop,
            ) -> torch.Tensor:
                local_parts = self._split_parameters(
                    method, all_parameters, depth
                )
                output = input_state
                for local_layer in range(segment_start, segment_stop):
                    output = self._apply_layer(
                        method, output, local_parts, local_layer, phi
                    )
                return output

            state = checkpoint(
                run_segment,
                state,
                parameters,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return state

    def expected_energy(self, method: Method, state: torch.Tensor) -> torch.Tensor:
        probabilities = torch.abs(state) ** 2
        norm = torch.sum(probabilities)
        return torch.sum(probabilities * self.training_diagonal(method)) / norm

    def energy_from_parameters(
        self,
        method: Method,
        parameters: torch.Tensor,
        *,
        depth: int,
        phi: torch.Tensor | None = None,
        activation_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.evolve(
            method,
            parameters,
            depth=depth,
            phi=phi,
            activation_checkpointing=activation_checkpointing,
        )
        return self.expected_energy(method, state), state

    def feasible_mass(self, state: torch.Tensor) -> float:
        probabilities = torch.abs(state) ** 2
        mass = torch.sum(probabilities[self.feasible_mask]) / torch.sum(probabilities)
        return float(mass.detach().cpu())

    def zero_angle_state(
        self,
        method: Method,
        *,
        depth: int,
        phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parameters = torch.zeros(
            parameter_count(method, depth),
            dtype=self.real_dtype,
            device=self.device,
        )
        return self.evolve(method, parameters, depth=depth, phi=phi)


def canonicalize_global_phase(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype="<c16").reshape(-1).copy()
    if values.size == 0:
        raise ValueError("cannot canonicalize an empty state")
    magnitudes = np.abs(values)
    pivot = int(np.argmax(magnitudes))
    if magnitudes[pivot] == 0.0:
        raise ValueError("cannot canonicalize the zero vector")
    values /= np.linalg.norm(values)
    phase = values[pivot] / abs(values[pivot])
    values /= phase
    values.real[values.real == 0.0] = 0.0
    values.imag[values.imag == 0.0] = 0.0
    return values.astype("<c16", copy=False)


def canonical_state_sha256(state: np.ndarray | torch.Tensor) -> str:
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    canonical = canonicalize_global_phase(np.asarray(state))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

