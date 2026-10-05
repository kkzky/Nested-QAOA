"""Registry-driven exact dynamics for the reviewer appendix campaign.

Only the frozen predecessor's reduced-basis mathematical engine is imported.
No predecessor campaign protocol, runner, target, or analysis state is reused.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from bcst_v2.dynamics import ReducedDynamics, ReducedProblem

from .method_registry import get_method_spec, parameter_count


StateInput: TypeAlias = np.ndarray | torch.Tensor
STATE_NORM_TOLERANCE = 1e-10


class AppendixDynamics(ReducedDynamics):
    """Exact float64/complex128 dynamics configured by immutable method specs."""

    def _bound_phi(self, method: str, phi: StateInput | None) -> torch.Tensor | None:
        spec = get_method_spec(method)
        if spec.initial_state == "phi" and phi is None:
            raise ValueError(f"{spec.method_id} requires the frozen Stage-1 state phi")
        if spec.initial_state == "native" and phi is not None:
            raise ValueError(f"{spec.method_id} prohibits a Stage-1 state phi")
        if phi is None:
            return None
        if isinstance(phi, torch.Tensor) and phi.requires_grad:
            raise ValueError("the frozen Stage-1 state cannot require gradients")
        bound = torch.as_tensor(phi, dtype=self.complex_dtype, device=self.device)
        if bound.shape != (self.problem.dimension,):
            raise ValueError("Stage-1 state has the wrong shape")
        if not bool(torch.all(torch.isfinite(bound))):
            raise ValueError("Stage-1 state contains nonfinite amplitudes")
        probability_norm = torch.sum(torch.abs(bound) ** 2)
        if not bool(torch.isfinite(probability_norm)):
            raise ValueError("Stage-1 state has a nonfinite norm")
        normalization_error = abs(float(probability_norm.detach().cpu()) - 1.0)
        if normalization_error > STATE_NORM_TOLERANCE:
            raise ValueError("Stage-1 state normalization check failed")
        return bound.detach()

    def _split_parameters(
        self,
        method: str,
        parameters: torch.Tensor,
        depth: int,
    ) -> dict[str, torch.Tensor]:
        spec = get_method_spec(method)
        expected = parameter_count(spec.method_id, depth)
        if parameters.shape != (expected,):
            raise ValueError(
                f"{spec.method_id} depth {depth} expects {expected} parameters, "
                f"received {tuple(parameters.shape)}"
            )
        if parameters.dtype != self.real_dtype:
            raise ValueError("parameters must be float64")
        # ``torch.device("cuda")`` is intentionally index-free, while a tensor
        # allocated through it reports the resolved device (normally
        # ``cuda:0``).  Compare against an actually allocated model tensor so
        # the contract checks the physical device rather than two differently
        # spelled aliases for it.
        if parameters.device != self.h_c.device:
            raise ValueError("parameters and dynamics must use the same device")
        parts: dict[str, torch.Tensor] = {}
        offset = 0
        for family in spec.parameter_families:
            parts[family] = parameters[offset : offset + depth]
            offset += depth
        return parts

    def training_diagonal(self, method: str) -> torch.Tensor:
        loss = get_method_spec(method).loss
        if loss == "h_c":
            return self.h_c
        if loss == "h_f":
            return self.h_f
        if loss == "h_d":
            return self.h_d
        raise AssertionError(f"unknown loss kind {loss!r}")

    def _initial_state(
        self, method: str, bound_phi: torch.Tensor | None
    ) -> torch.Tensor:
        spec = get_method_spec(method)
        if spec.initial_state == "native":
            return self.native_state
        if bound_phi is None:  # pragma: no cover - guarded by _bound_phi.
            raise ValueError(f"{spec.method_id} requires phi")
        return self._normalized(bound_phi)

    def _projector_reference(
        self, method: str, bound_phi: torch.Tensor | None
    ) -> torch.Tensor:
        reference = get_method_spec(method).projector_reference
        if reference == "native":
            return self.native_state
        if reference == "phi":
            if bound_phi is None:  # pragma: no cover - guarded by _bound_phi.
                raise ValueError(f"{method} requires phi as projector reference")
            return bound_phi
        raise ValueError(f"{method} does not use a projector mixer")

    def _apply_layer(
        self,
        method: str,
        state: torch.Tensor,
        parameter_parts: dict[str, torch.Tensor],
        layer: int,
        bound_phi: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = get_method_spec(method)
        if spec.phase == "h_d_split":
            state = state * torch.exp(
                -1j * parameter_parts["conflict_gamma"][layer] * self.h_d_conf
            ).to(self.complex_dtype)
            state = state * torch.exp(
                -1j * parameter_parts["objective_gamma"][layer] * self.h_d_obj
            ).to(self.complex_dtype)
        else:
            phase_family = spec.parameter_families[0]
            if spec.phase == "h_c":
                diagonal = self.h_c
            elif spec.phase == "h_f":
                diagonal = self.h_f
            elif spec.phase == "h_d":
                diagonal = self.h_d
            else:  # pragma: no cover - MethodSpec constrains the phase enum.
                raise AssertionError(f"unknown phase kind {spec.phase!r}")
            state = state * torch.exp(
                -1j * parameter_parts[phase_family][layer] * diagonal
            ).to(self.complex_dtype)

        beta = parameter_parts[spec.parameter_families[-1]][layer]
        if spec.mixer == "xy":
            return self.apply_xy(state, beta)
        reference = self._projector_reference(spec.method_id, bound_phi)
        return self.apply_history(state, beta, reference)

    def evolve(
        self,
        method: str,
        parameters: torch.Tensor,
        *,
        depth: int,
        phi: StateInput | None = None,
        activation_checkpointing: bool = False,
    ) -> torch.Tensor:
        spec = get_method_spec(method)
        # parameter_count validates depth before it is used in slicing/ranges.
        parameter_count(spec.method_id, depth)
        bound_phi = self._bound_phi(spec.method_id, phi)
        parts = self._split_parameters(spec.method_id, parameters, depth)
        state = self._initial_state(spec.method_id, bound_phi).clone()
        if not activation_checkpointing or not parameters.requires_grad:
            for layer in range(depth):
                state = self._apply_layer(
                    spec.method_id, state, parts, layer, bound_phi
                )
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
                    spec.method_id, all_parameters, depth
                )
                output = input_state
                for local_layer in range(segment_start, segment_stop):
                    output = self._apply_layer(
                        spec.method_id,
                        output,
                        local_parts,
                        local_layer,
                        bound_phi,
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

    def expected_energy(self, method: str, state: torch.Tensor) -> torch.Tensor:
        if state.shape != (self.problem.dimension,):
            raise ValueError("state has the wrong reduced-basis shape")
        probabilities = torch.abs(state) ** 2
        norm = torch.sum(probabilities)
        if not bool(torch.isfinite(norm)) or float(norm.detach().cpu()) <= 0.0:
            raise ValueError("state has invalid norm")
        return torch.sum(probabilities * self.training_diagonal(method)) / norm

    def energy_from_parameters(
        self,
        method: str,
        parameters: torch.Tensor,
        *,
        depth: int,
        phi: StateInput | None = None,
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
        norm = torch.sum(probabilities)
        if not bool(torch.isfinite(norm)) or float(norm.detach().cpu()) <= 0.0:
            raise ValueError("state has invalid norm")
        mass = torch.sum(probabilities[self.feasible_mask]) / norm
        return float(mass.detach().cpu())

    def zero_angle_state(
        self,
        method: str,
        *,
        depth: int,
        phi: StateInput | None = None,
    ) -> torch.Tensor:
        parameters = torch.zeros(
            parameter_count(method, depth),
            dtype=self.real_dtype,
            device=self.device,
        )
        return self.evolve(method, parameters, depth=depth, phi=phi)


__all__ = ["AppendixDynamics", "STATE_NORM_TOLERANCE", "StateInput"]
