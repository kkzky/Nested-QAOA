"""Immutable method contracts for the prospective reviewer appendix campaign.

This module is deliberately independent of optimizer and target-analysis code.
It is the single source of truth for the circuit ingredients, parameter-family
ordering, and terminal logical-resource formula of every optimized method in
``EXPERIMENT_PLAN_V3.json``.  The predecessor campaign remains read-only.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Protocol, TypeAlias


InitialState: TypeAlias = Literal["native", "phi"]
PhaseKind: TypeAlias = Literal["h_c", "h_f", "h_d", "h_d_split"]
LossKind: TypeAlias = Literal["h_c", "h_f", "h_d"]
MixerKind: TypeAlias = Literal["xy", "projector"]
ProjectorReference: TypeAlias = Literal["none", "native", "phi"]

# Exact campaign contract consumed by this registry.  Tests verify the bytes of
# EXPERIMENT_PLAN_V3.json against this digest so that a silent plan edit cannot
# change the meaning of any method or resource formula.
EXPERIMENT_PLAN_SHA256 = (
    "20476043ADF7233C5CD721E620EC4CD620CA58C5BECFD1A192BD0495C25BAF08"
)


class LogicalResourceLike(Protocol):
    """Structural type accepted by :func:`terminal_ru`."""

    B: int
    C: int
    X: int
    O: int
    S: int
    P1: int


@dataclass(frozen=True)
class MethodSpec:
    """Complete target-blind circuit contract for one optimized method."""

    method_id: str
    paper_label: str
    initial_state: InitialState
    phase: PhaseKind
    loss: LossKind
    mixer: MixerKind
    projector_reference: ProjectorReference
    parameter_families: tuple[str, ...]
    resource_kind: str

    def __post_init__(self) -> None:
        if not self.method_id or not self.paper_label:
            raise ValueError("method identifiers and labels must be nonempty")
        if not self.parameter_families:
            raise ValueError("optimized methods require parameter families")
        if len(set(self.parameter_families)) != len(self.parameter_families):
            raise ValueError("parameter-family names must be unique")
        if self.phase == "h_d_split":
            expected_prefix = ("conflict_gamma", "objective_gamma")
            if self.parameter_families[:2] != expected_prefix:
                raise ValueError(
                    "split H_D parameters must begin with conflict then objective gamma"
                )
        elif len(self.parameter_families) != 2:
            raise ValueError("single-phase methods require one gamma and one beta family")
        expected_beta = "projector_beta" if self.mixer == "projector" else "xy_beta"
        if self.parameter_families[-1] != expected_beta:
            raise ValueError("the final parameter family must match the mixer")
        if self.mixer == "projector":
            if self.projector_reference == "none":
                raise ValueError("projector mixers require a reference state")
        elif self.projector_reference != "none":
            raise ValueError("XY methods cannot declare a projector reference")
        if self.projector_reference == "phi" and self.initial_state != "phi":
            raise ValueError("a learned projector requires the frozen Stage-1 state")

    def parameter_layout(self, depth: int) -> tuple[tuple[str, int], ...]:
        p = _positive_depth(depth)
        return tuple((family, p) for family in self.parameter_families)

    def parameter_count(self, depth: int) -> int:
        p = _positive_depth(depth)
        return len(self.parameter_families) * p


def _positive_depth(depth: int) -> int:
    if isinstance(depth, bool):
        raise ValueError("depth must be a positive integer")
    try:
        value = int(operator.index(depth))
    except TypeError as exc:
        raise ValueError("depth must be a positive integer") from exc
    if value < 1:
        raise ValueError("depth must be a positive integer")
    return value


_SPECS = {
    "stage1": MethodSpec(
        method_id="stage1",
        paper_label="Stage1",
        initial_state="native",
        phase="h_c",
        loss="h_c",
        mixer="xy",
        projector_reference="none",
        parameter_families=("conflict_gamma", "xy_beta"),
        resource_kind="stage1",
    ),
    "learned_projector": MethodSpec(
        method_id="learned_projector",
        paper_label="Projector_HF",
        initial_state="phi",
        phase="h_f",
        loss="h_f",
        mixer="projector",
        projector_reference="phi",
        parameter_families=("objective_gamma", "projector_beta"),
        resource_kind="projector_hf",
    ),
    "ordinary_native_direct": MethodSpec(
        method_id="ordinary_native_direct",
        paper_label="StdXY",
        initial_state="native",
        phase="h_d",
        loss="h_d",
        mixer="xy",
        projector_reference="none",
        parameter_families=("shared_cost_gamma", "xy_beta"),
        resource_kind="native_xy_hd",
    ),
    "decoupled_native_direct": MethodSpec(
        method_id="decoupled_native_direct",
        paper_label="dXY",
        initial_state="native",
        phase="h_d_split",
        loss="h_d",
        mixer="xy",
        projector_reference="none",
        parameter_families=("conflict_gamma", "objective_gamma", "xy_beta"),
        resource_kind="native_xy_hd",
    ),
    "same_stage1_state_direct": MethodSpec(
        method_id="same_stage1_state_direct",
        paper_label="XY_HD",
        initial_state="phi",
        phase="h_d",
        loss="h_d",
        mixer="xy",
        projector_reference="none",
        parameter_families=("shared_cost_gamma", "xy_beta"),
        resource_kind="xy_hd",
    ),
    "projector_hd": MethodSpec(
        method_id="projector_hd",
        paper_label="Projector_HD",
        initial_state="phi",
        phase="h_d",
        loss="h_d",
        mixer="projector",
        projector_reference="phi",
        parameter_families=("shared_cost_gamma", "projector_beta"),
        resource_kind="projector_hd",
    ),
    "xy_hf": MethodSpec(
        method_id="xy_hf",
        paper_label="XY_HF",
        initial_state="phi",
        phase="h_f",
        loss="h_f",
        mixer="xy",
        projector_reference="none",
        parameter_families=("objective_gamma", "xy_beta"),
        resource_kind="xy_hf",
    ),
    "known_projector_hd": MethodSpec(
        method_id="known_projector_hd",
        paper_label="KnownProjector_HD",
        initial_state="phi",
        phase="h_d",
        loss="h_d",
        mixer="projector",
        projector_reference="native",
        parameter_families=("shared_cost_gamma", "projector_beta"),
        resource_kind="known_projector_hd",
    ),
    "native_grover_d": MethodSpec(
        method_id="native_grover_d",
        paper_label="NativeGrover-d",
        initial_state="native",
        phase="h_d_split",
        loss="h_d",
        mixer="projector",
        projector_reference="native",
        parameter_families=(
            "conflict_gamma",
            "objective_gamma",
            "projector_beta",
        ),
        resource_kind="native_grover_d",
    ),
}

METHOD_SPECS: Mapping[str, MethodSpec] = MappingProxyType(_SPECS)

_ALIASES = {
    **{method_id: method_id for method_id in _SPECS},
    "Stage1": "stage1",
    "LP": "learned_projector",
    "Projector_HF": "learned_projector",
    "StdXY": "ordinary_native_direct",
    "Std-XY": "ordinary_native_direct",
    "dXY": "decoupled_native_direct",
    "d-XY": "decoupled_native_direct",
    "WarmXY": "same_stage1_state_direct",
    "Warm-XY": "same_stage1_state_direct",
    "XY_HD": "same_stage1_state_direct",
    "Projector_HD": "projector_hd",
    "XY_HF": "xy_hf",
    "KnownProjector_HD": "known_projector_hd",
    "NativeGrover-d": "native_grover_d",
    "stage1_only": "stage1_only",
    "Stage1Only": "stage1_only",
    "Stage1_only": "stage1_only",
}
METHOD_ALIASES: Mapping[str, str] = MappingProxyType(_ALIASES)


def canonical_method_id(method: str) -> str:
    """Return the canonical internal identifier for a paper or code label."""

    try:
        return METHOD_ALIASES[str(method)]
    except KeyError as exc:
        raise ValueError(f"unknown appendix method {method!r}") from exc


def get_method_spec(method: str) -> MethodSpec:
    canonical = canonical_method_id(method)
    if canonical == "stage1_only":
        raise ValueError("Stage1Only has no optimized circuit specification")
    return METHOD_SPECS[canonical]


def parameter_layout_for_method(
    method: str, depth: int
) -> tuple[tuple[str, int], ...]:
    return get_method_spec(method).parameter_layout(depth)


def parameter_count(method: str, depth: int) -> int:
    return get_method_spec(method).parameter_count(depth)


def terminal_ru(
    resources: LogicalResourceLike,
    method: str,
    depth: int | None = None,
) -> int:
    """Return the exact plan-v2 terminal logical-resource count."""

    canonical = canonical_method_id(method)
    if canonical == "stage1_only":
        if depth is not None:
            raise ValueError("Stage1Only has no Stage-2 depth")
        return int(resources.P1 + resources.O)
    if canonical == "stage1":
        raise ValueError("Stage1 is a preparation optimizer; use Stage1Only for terminal RU")
    if depth is None:
        raise ValueError(f"{method} requires a Stage-2 depth")
    p = _positive_depth(depth)
    spec = METHOD_SPECS[canonical]
    B, C, X, O, S, P1 = (
        int(resources.B),
        int(resources.C),
        int(resources.X),
        int(resources.O),
        int(resources.S),
        int(resources.P1),
    )
    if min(B, C, X, O, S, P1) < 0:
        raise ValueError("logical-resource primitives must be nonnegative")

    if spec.resource_kind == "projector_hf":
        return P1 + p * (O + 2 * P1 + S) + O
    if spec.resource_kind == "projector_hd":
        return P1 + p * (C + O + 2 * P1 + S) + O
    if spec.resource_kind == "known_projector_hd":
        return P1 + p * (C + O + 2 * B + S) + O
    if spec.resource_kind == "xy_hf":
        return P1 + p * (O + X) + O
    if spec.resource_kind == "xy_hd":
        return P1 + p * (C + O + X) + O
    if spec.resource_kind == "native_xy_hd":
        return B + p * (C + O + X) + O
    if spec.resource_kind == "native_grover_d":
        return B + p * (C + O + 2 * B + S) + O
    raise AssertionError(f"unhandled resource kind {spec.resource_kind!r}")


PHI_INITIAL_METHODS = frozenset(
    method_id
    for method_id, spec in METHOD_SPECS.items()
    if spec.initial_state == "phi"
)
NATIVE_INITIAL_METHODS = frozenset(METHOD_SPECS).difference(PHI_INITIAL_METHODS)


__all__ = [
    "EXPERIMENT_PLAN_SHA256",
    "METHOD_ALIASES",
    "METHOD_SPECS",
    "NATIVE_INITIAL_METHODS",
    "PHI_INITIAL_METHODS",
    "LogicalResourceLike",
    "MethodSpec",
    "canonical_method_id",
    "get_method_spec",
    "parameter_count",
    "parameter_layout_for_method",
    "terminal_ru",
]
