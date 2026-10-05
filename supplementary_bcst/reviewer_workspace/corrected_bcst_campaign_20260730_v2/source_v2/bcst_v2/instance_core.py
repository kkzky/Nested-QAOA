"""Exact instance construction for the reviewed BCST v2 campaign.

This module implements the instance-facing portion of contract candidate 2
(SHA-256 D6D6A955FB20DD3786E903D5A28103BE11DD1D0F20E998C487631E3E0CB09C3C).
It is intentionally NumPy-only.  Production dynamics may copy the immutable
arrays to another backend, while :class:`ProductShellXY` preserves the exact
factorized XY convention without materializing a D-by-D matrix.

The public entry point is::

    build_instance(
        N, master_seed, d4_num, d4_den, r6_num, r6_den,
        include_targets=False,
    )

The returned :class:`BCSTInstance` exposes the reduced product-shell basis,
cycle conflicts, feasible-projection pools, frozen support orders and dyadic
coefficients, all Hamiltonian diagonals, the factorized ``H_XY`` operator,
logical-resource accounting, and integrity hashes.  Exact target ordering is
available only through the explicit post-lock path described below.

Target derivation is phase-separated.  ``build_instance`` defaults to a
target-blind training object (``targets is None`` and no target hashes).
Only post-lock analysis should call :func:`derive_targets`, or explicitly
construct an analysis instance with ``include_targets=True``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_SHA256 = (
    "D6D6A955FB20DD3786E903D5A28103BE11DD1D0F20E998C487631E3E0CB09C3C"
)
HASH_SCHEMA = "bcst-v2-canonical-hash-v1"
BLOCK_COUNT = 5
LOCAL_WEIGHT = 2
RAW_COEFFICIENT_DENOMINATOR = 1 << 54
CONFLICT_SPECTRAL_MAXIMUM = 10
PENALTY_COEFFICIENT = 11
SUPPORTED_LAYOUTS: Mapping[int, Mapping[str, int]] = MappingProxyType(
    {
        25: MappingProxyType(
            {
                "K": 5,
                "dimension": 100_000,
                "feasible_count": 120,
                "P4_count": 450,
                "P6_count": 900,
            }
        ),
        30: MappingProxyType(
            {
                "K": 6,
                "dimension": 759_375,
                "feasible_count": 6_570,
                "P4_count": 1_575,
                "P6_count": 9_450,
            }
        ),
    }
)

Support = tuple[int, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single JSON encoding used by all structured hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _structured_hash(domain: str, payload: Any) -> str:
    envelope = {
        "schema": HASH_SCHEMA,
        "domain": str(domain),
        "payload": payload,
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def _array_hash(domain: str, values: np.ndarray, dtype: str) -> str:
    """Hash a numeric array with an explicit little-endian wire encoding."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    header = canonical_json_bytes(
        {
            "schema": HASH_SCHEMA,
            "domain": str(domain),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _as_index(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def round_half_up_product(count: int, numerator: int, denominator: int) -> int:
    """Return ``floor(count*numerator/denominator + 1/2)`` exactly."""

    count = _as_index("count", count)
    numerator = _as_index("numerator", numerator)
    denominator = _as_index("denominator", denominator)
    if count < 0:
        raise ValueError("count must be nonnegative")
    if numerator < 0:
        raise ValueError("numerator must be nonnegative")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (2 * count * numerator + denominator) // (2 * denominator)


def support_order_text(
    N: int, master_seed: int, degree: int, support: Sequence[int]
) -> str:
    """Canonical contract text used to rank one support."""

    suffix = ",".join(str(_as_index("qubit", q)) for q in support)
    return (
        f"bcst-v2|support-order|{_as_index('N', N)}|"
        f"{_as_index('master_seed', master_seed)}|"
        f"{_as_index('degree', degree)}|{suffix}"
    )


def support_order_digest(
    N: int, master_seed: int, degree: int, support: Sequence[int]
) -> bytes:
    return hashlib.sha256(
        support_order_text(N, master_seed, degree, support).encode("utf-8")
    ).digest()


def coefficient_key_text(
    N: int, master_seed: int, degree: int, support: Sequence[int]
) -> str:
    """Canonical contract text used to assign one dyadic coefficient."""

    suffix = ",".join(str(_as_index("qubit", q)) for q in support)
    return (
        f"bcst-v2|coefficient|{_as_index('N', N)}|"
        f"{_as_index('master_seed', master_seed)}|"
        f"{_as_index('degree', degree)}|{suffix}"
    )


def coefficient_numerator(
    N: int, master_seed: int, degree: int, support: Sequence[int]
) -> int:
    """Return the exact odd signed numerator ``a_S`` from the contract."""

    digest = hashlib.sha256(
        coefficient_key_text(N, master_seed, degree, support).encode("utf-8")
    ).digest()
    z = int.from_bytes(digest[:8], "big") >> 11
    numerator = 2 * z + 1 - (1 << 53)
    if numerator == 0 or numerator % 2 == 0:
        raise AssertionError("coefficient KDF did not produce a nonzero odd integer")
    return numerator


def dyadic_coefficient(
    N: int, master_seed: int, degree: int, support: Sequence[int]
) -> float:
    """Return ``a_S/2**54``, exactly representable as binary64."""

    return math.ldexp(
        float(coefficient_numerator(N, master_seed, degree, support)), -54
    )


@dataclass(frozen=True)
class ProductShellBasis:
    """Canonical reduced basis with the last block varying fastest."""

    N: int
    K: int
    block_count: int
    local_weight: int
    local_choices: np.ndarray
    local_bits: np.ndarray
    local_indices: np.ndarray
    packed_bits: np.ndarray

    @property
    def local_dimension(self) -> int:
        return int(self.local_choices.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.packed_bits.size)

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return (self.local_dimension,) * self.block_count

    def uniform_state(self) -> np.ndarray:
        """Return the native positive-amplitude product-shell state."""

        result = np.full(
            self.dimension,
            1.0 / math.sqrt(self.dimension),
            dtype=np.complex128,
        )
        return result


@dataclass(frozen=True)
class ProductShellXY:
    """Factorized exact representation of the normalized XY generator."""

    basis: ProductShellBasis
    local_adjacency: np.ndarray
    local_eigenvalues: np.ndarray
    local_eigenvectors: np.ndarray

    @property
    def raw_shift(self) -> int:
        return 10

    @property
    def raw_minimum(self) -> int:
        return -10

    @property
    def raw_maximum(self) -> int:
        return 10 * (self.basis.K - 2)

    @property
    def spectral_range(self) -> int:
        return 2 * (self.basis.N - 5)

    def _check_state(self, state: np.ndarray) -> np.ndarray:
        values = np.asarray(state)
        if values.ndim < 1 or values.shape[-1] != self.basis.dimension:
            raise ValueError(
                "state trailing dimension must equal the reduced-basis "
                f"dimension {self.basis.dimension}"
            )
        return values

    @staticmethod
    def _apply_local(matrix: np.ndarray, tensor: np.ndarray, axis: int) -> np.ndarray:
        moved = np.moveaxis(tensor, axis, -1)
        transformed = moved @ matrix.T
        return np.moveaxis(transformed, -1, axis)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply ``H_XY=(B_raw+10 I)/(2(N-5))`` without a global matrix."""

        values = self._check_state(state)
        leading = values.shape[:-1]
        tensor = values.reshape(leading + self.basis.tensor_shape)
        block_axis_start = len(leading)
        result = self.raw_shift * tensor
        for block in range(self.basis.block_count):
            result = result + self._apply_local(
                self.local_adjacency, tensor, block_axis_start + block
            )
        return (result / self.spectral_range).reshape(values.shape)

    def evolve(self, state: np.ndarray, beta: float) -> np.ndarray:
        """Apply ``exp(-i*beta*H_XY)`` by commuting local eigendecompositions."""

        values = self._check_state(state)
        beta = float(beta)
        if not math.isfinite(beta):
            raise ValueError("beta must be finite")
        leading = values.shape[:-1]
        result = np.asarray(values, dtype=np.complex128).reshape(
            leading + self.basis.tensor_shape
        )
        phases = np.exp(
            -1j * beta * self.local_eigenvalues / self.spectral_range
        )
        local_unitary = (
            self.local_eigenvectors * phases[np.newaxis, :]
        ) @ self.local_eigenvectors.T
        block_axis_start = len(leading)
        for block in range(self.basis.block_count):
            result = self._apply_local(
                local_unitary, result, block_axis_start + block
            )
        result = result * np.exp(
            -1j * beta * self.raw_shift / self.spectral_range
        )
        return result.reshape(values.shape)


@dataclass(frozen=True)
class ProblemStructure:
    """Seed-independent exact shell and conflict structure for one N."""

    basis: ProductShellBasis
    conflict_edges: tuple[tuple[int, int], ...]
    C_conf: np.ndarray
    H_C: np.ndarray
    feasible_mask: np.ndarray
    feasible_indices: np.ndarray
    P4: tuple[Support, ...]
    P6: tuple[Support, ...]
    H_XY: ProductShellXY
    hashes: Mapping[str, str]

    @property
    def N(self) -> int:
        return self.basis.N

    @property
    def K(self) -> int:
        return self.basis.K

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @property
    def feasible_count(self) -> int:
        return int(self.feasible_indices.size)


@dataclass(frozen=True)
class TargetSets:
    """Conflict-feasible configurations in exact contract target order."""

    ordered_feasible_indices: np.ndarray
    ordered_energy_numerators: tuple[int, ...]
    best2: np.ndarray
    best8: np.ndarray
    ground: np.ndarray
    unique_ground: np.ndarray
    ground_degeneracy: int
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class ObjectiveNormalization:
    raw_denominator: int
    raw_min_numerator: int
    raw_max_numerator: int
    raw_range_numerator: int
    g_common_denominator: int
    g_min_numerator: int
    g_max_numerator: int
    g_range_numerator: int
    g_min: float
    g_max: float
    delta_G: float


@dataclass(frozen=True)
class LogicalResources:
    """Primitive and derived logical terminal resource counts."""

    N: int
    m4: int
    m6: int
    B: int
    C: int
    X: int
    O: int
    S: int
    P1: int

    def terminal_ru(self, method: str, depth: int | None = None) -> int:
        canonical = {
            "stage1_only": "stage1_only",
            "Stage1_only": "stage1_only",
            "learned_projector": "learned_projector",
            "LP": "learned_projector",
            "ordinary_native_direct": "ordinary_native_direct",
            "StdXY": "ordinary_native_direct",
            "Std-XY": "ordinary_native_direct",
            "decoupled_native_direct": "decoupled_native_direct",
            "dXY": "decoupled_native_direct",
            "d-XY": "decoupled_native_direct",
            "same_stage1_state_direct": "same_stage1_state_direct",
            "WarmXY": "same_stage1_state_direct",
            "Warm-XY": "same_stage1_state_direct",
        }.get(str(method))
        if canonical is None:
            raise ValueError(f"unknown method {method!r}")
        if canonical == "stage1_only":
            if depth is not None:
                raise ValueError("Stage1_only has no Stage-2 depth")
            return self.P1 + self.O
        if depth is None:
            raise ValueError(f"{method} requires a Stage-2 depth")
        p = _as_index("depth", depth)
        if p < 1:
            raise ValueError("Stage-2 depth must be positive")
        if canonical == "learned_projector":
            return self.P1 + p * (self.O + 2 * self.P1 + self.S) + self.O
        if canonical in {"ordinary_native_direct", "decoupled_native_direct"}:
            return self.B + p * (self.C + self.O + self.X) + self.O
        return self.P1 + p * (self.C + self.O + self.X) + self.O


@dataclass(frozen=True)
class BCSTInstance:
    """Complete deterministic v2 instance returned by :func:`build_instance`."""

    structure: ProblemStructure
    master_seed: int
    d4_numerator: int
    d4_denominator: int
    r6_numerator: int
    r6_denominator: int
    P4_order: tuple[Support, ...]
    P6_order: tuple[Support, ...]
    m4: int
    m6: int
    supports4: tuple[Support, ...]
    supports6: tuple[Support, ...]
    supports: tuple[Support, ...]
    coefficient_numerators: np.ndarray
    coefficients: np.ndarray
    exact_raw_energy_numerators: np.ndarray
    C_raw: np.ndarray
    H_F: np.ndarray
    G: np.ndarray
    H_D: np.ndarray
    H_D_obj: np.ndarray
    H_D_conf: np.ndarray
    normalization: ObjectiveNormalization
    targets: TargetSets | None
    resources: LogicalResources
    hashes: Mapping[str, str]

    @property
    def N(self) -> int:
        return self.structure.N

    @property
    def K(self) -> int:
        return self.structure.K

    @property
    def basis(self) -> ProductShellBasis:
        return self.structure.basis

    @property
    def basis_local_choices(self) -> np.ndarray:
        return self.basis.local_choices

    @property
    def basis_bits(self) -> np.ndarray:
        return self.basis.packed_bits

    @property
    def dimension(self) -> int:
        return self.structure.dimension

    @property
    def feasible_count(self) -> int:
        return self.structure.feasible_count

    @property
    def conflict_edges(self) -> tuple[tuple[int, int], ...]:
        return self.structure.conflict_edges

    @property
    def C_conf(self) -> np.ndarray:
        return self.structure.C_conf

    @property
    def H_C(self) -> np.ndarray:
        return self.structure.H_C

    @property
    def feasible_mask(self) -> np.ndarray:
        return self.structure.feasible_mask

    @property
    def feasible_indices(self) -> np.ndarray:
        return self.structure.feasible_indices

    @property
    def P4(self) -> tuple[Support, ...]:
        return self.structure.P4

    @property
    def P6(self) -> tuple[Support, ...]:
        return self.structure.P6

    @property
    def H_XY(self) -> ProductShellXY:
        return self.structure.H_XY

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "dimension": self.dimension,
                "conflict_feasible": self.feasible_count,
                "P4": len(self.P4),
                "P6": len(self.P6),
                "m4": self.m4,
                "m6": self.m6,
            }
        )

    def terminal_ru(self, method: str, depth: int | None = None) -> int:
        return self.resources.terminal_ru(method, depth)

    def require_targets(self) -> TargetSets:
        """Return targets or fail closed when this is a target-blind instance."""

        if self.targets is None:
            raise RuntimeError(
                "targets were not derived; confirmation optimization must "
                "remain target-blind until the cohort lock"
            )
        return self.targets


def _build_basis(N: int, K: int) -> ProductShellBasis:
    choices = np.asarray(
        list(itertools.combinations(range(K), LOCAL_WEIGHT)), dtype=np.uint8
    )
    local_bits = np.asarray(
        [(1 << int(a)) | (1 << int(b)) for a, b in choices], dtype=np.uint8
    )
    local_dimension = int(choices.shape[0])
    dimension = local_dimension**BLOCK_COUNT
    flat = np.arange(dimension, dtype=np.int64)
    local_indices = np.empty((dimension, BLOCK_COUNT), dtype=np.uint8)
    for block in range(BLOCK_COUNT):
        stride = local_dimension ** (BLOCK_COUNT - block - 1)
        local_indices[:, block] = (flat // stride) % local_dimension
    packed_bits = np.zeros(dimension, dtype=np.uint64)
    for block in range(BLOCK_COUNT):
        packed_bits |= (
            local_bits[local_indices[:, block]].astype(np.uint64)
            << np.uint64(block * K)
        )
    return ProductShellBasis(
        N=N,
        K=K,
        block_count=BLOCK_COUNT,
        local_weight=LOCAL_WEIGHT,
        local_choices=_readonly(choices),
        local_bits=_readonly(local_bits),
        local_indices=_readonly(local_indices),
        packed_bits=_readonly(packed_bits),
    )


def _feasible_projection_pool(
    basis: ProductShellBasis,
    feasible_indices: np.ndarray,
    block_order: int,
) -> tuple[Support, ...]:
    feasible_rows = basis.local_indices[feasible_indices]
    supports: set[Support] = set()
    for blocks in itertools.combinations(range(BLOCK_COUNT), block_order):
        projections = np.unique(feasible_rows[:, blocks], axis=0)
        for projection in projections:
            support: list[int] = []
            for block, local_index in zip(blocks, projection):
                local_pair = basis.local_choices[int(local_index)]
                support.extend(
                    (
                        int(block * basis.K + int(local_pair[0])),
                        int(block * basis.K + int(local_pair[1])),
                    )
                )
            canonical = tuple(support)
            if tuple(sorted(canonical)) != canonical:
                raise AssertionError("support canonicalization failed")
            supports.add(canonical)
    return tuple(sorted(supports))


def _build_xy(basis: ProductShellBasis) -> ProductShellXY:
    choices = basis.local_choices
    shared = np.equal(
        choices[:, np.newaxis, :, np.newaxis],
        choices[np.newaxis, :, np.newaxis, :],
    ).any(axis=(2, 3))
    adjacency = (shared & ~np.eye(basis.local_dimension, dtype=bool)).astype(
        np.float64
    )
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    expected = np.asarray(
        [-2] * (basis.K * (basis.K - 3) // 2)
        + [basis.K - 4] * (basis.K - 1)
        + [2 * (basis.K - 2)],
        dtype=np.float64,
    )
    if not np.allclose(eigenvalues, expected, rtol=0.0, atol=1e-12):
        raise AssertionError("local Johnson-graph XY spectrum is incorrect")
    return ProductShellXY(
        basis=basis,
        local_adjacency=_readonly(adjacency),
        local_eigenvalues=_readonly(eigenvalues),
        local_eigenvectors=_readonly(eigenvectors),
    )


@lru_cache(maxsize=2)
def build_product_shell_structure(N: int) -> ProblemStructure:
    """Build and cache the seed-independent N=25 or N=30 structure."""

    N = _as_index("N", N)
    if N not in SUPPORTED_LAYOUTS:
        raise ValueError(f"supported N values are {tuple(SUPPORTED_LAYOUTS)}, got {N}")
    expected = SUPPORTED_LAYOUTS[N]
    K = int(expected["K"])
    basis = _build_basis(N, K)
    if basis.dimension != int(expected["dimension"]):
        raise AssertionError("product-shell dimension differs from the contract")

    lookup = np.empty(
        (basis.local_dimension, basis.local_dimension), dtype=np.uint8
    )
    for left, left_mask in enumerate(basis.local_bits):
        for right, right_mask in enumerate(basis.local_bits):
            lookup[left, right] = (
                int(int(left_mask) & int(right_mask)).bit_count()
            )
    conflicts = np.zeros(basis.dimension, dtype=np.uint8)
    for block in range(BLOCK_COUNT):
        conflicts += lookup[
            basis.local_indices[:, block],
            basis.local_indices[:, (block + 1) % BLOCK_COUNT],
        ]
    if int(conflicts.min()) != 0 or int(conflicts.max()) != 10:
        raise AssertionError("cycle-conflict shell extrema differ from [0,10]")
    feasible_mask = conflicts == 0
    feasible_indices = np.flatnonzero(feasible_mask).astype(np.int64)
    if feasible_indices.size != int(expected["feasible_count"]):
        raise AssertionError("conflict-feasible count differs from the contract")

    P4 = _feasible_projection_pool(basis, feasible_indices, 2)
    P6 = _feasible_projection_pool(basis, feasible_indices, 3)
    if len(P4) != int(expected["P4_count"]):
        raise AssertionError("P4 count differs from the frozen contract count")
    if len(P6) != int(expected["P6_count"]):
        raise AssertionError("P6 count differs from the frozen contract count")
    if len(set(P4)) != len(P4) or any(len(term) != 4 for term in P4):
        raise AssertionError("P4 is not a unique canonical degree-4 pool")
    if len(set(P6)) != len(P6) or any(len(term) != 6 for term in P6):
        raise AssertionError("P6 is not a unique canonical degree-6 pool")

    conflict_edges = tuple(
        (block * K + label, ((block + 1) % BLOCK_COUNT) * K + label)
        for block in range(BLOCK_COUNT)
        for label in range(K)
    )
    hashes = MappingProxyType(
        {
            "basis": _array_hash(
                "reduced-basis-packed-bits", basis.packed_bits, "<u8"
            ),
            "conflict_graph": _structured_hash(
                "cycle-conflict-graph",
                [{"u": u, "v": v, "weight": 1} for u, v in conflict_edges],
            ),
            "conflict_diagonal": _array_hash(
                "C-conf-shell-diagonal", conflicts, "|u1"
            ),
            "P4_lexicographic_pool": _structured_hash(
                "P4-lexicographic-pool", [list(term) for term in P4]
            ),
            "P6_lexicographic_pool": _structured_hash(
                "P6-lexicographic-pool", [list(term) for term in P6]
            ),
        }
    )
    return ProblemStructure(
        basis=basis,
        conflict_edges=conflict_edges,
        C_conf=_readonly(conflicts),
        H_C=_readonly(conflicts.astype(np.float64) / 10.0),
        feasible_mask=_readonly(feasible_mask),
        feasible_indices=_readonly(feasible_indices),
        P4=P4,
        P6=P6,
        H_XY=_build_xy(basis),
        hashes=hashes,
    )


def _rank_supports(
    pool: Sequence[Support], N: int, master_seed: int, degree: int
) -> tuple[Support, ...]:
    return tuple(
        sorted(
            pool,
            key=lambda support: (
                support_order_digest(N, master_seed, degree, support),
                support,
            ),
        )
    )


def _support_local_coordinates(
    basis: ProductShellBasis, support: Support
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for qubit in support:
        block, label = divmod(int(qubit), basis.K)
        grouped.setdefault(block, []).append(label)
    blocks = tuple(sorted(grouped))
    if len(blocks) not in (2, 3):
        raise AssertionError("selected support has the wrong block order")
    pair_to_index = {
        tuple(int(value) for value in pair): index
        for index, pair in enumerate(basis.local_choices)
    }
    local_coordinates = tuple(
        pair_to_index[tuple(sorted(grouped[block]))] for block in blocks
    )
    return blocks, local_coordinates


def _exact_objective_diagonal(
    basis: ProductShellBasis,
    supports: Sequence[Support],
    numerators: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[tuple[int, ...], np.ndarray], ...]]:
    max_active_terms = math.comb(BLOCK_COUNT, 2) + math.comb(BLOCK_COUNT, 3)
    max_numerator = (1 << 53) - 1
    if max_active_terms * max_numerator > np.iinfo(np.int64).max:
        raise AssertionError("the exact objective no longer fits guarded int64")

    tables: dict[tuple[int, ...], np.ndarray] = {}
    L = basis.local_dimension
    for support, numerator in zip(supports, numerators):
        blocks, coordinates = _support_local_coordinates(basis, support)
        table = tables.setdefault(blocks, np.zeros((L,) * len(blocks), dtype=np.int64))
        if table[coordinates] != 0:
            raise AssertionError("two coefficients map to the same support coordinate")
        table[coordinates] = int(numerator)

    exact = np.zeros(basis.dimension, dtype=np.int64)
    components: list[tuple[tuple[int, ...], np.ndarray]] = []
    for blocks in sorted(tables):
        table = tables[blocks]
        indices = tuple(basis.local_indices[:, block] for block in blocks)
        exact -= table[indices]
        components.append((blocks, _readonly(table)))
    return _readonly(exact), tuple(components)


def _target_sets_from_rows(
    structure: ProblemStructure, rows: list[tuple[int, int, int]]
) -> TargetSets:
    rows.sort(key=lambda row: (row[0], row[1]))
    ordered_indices = np.asarray([row[2] for row in rows], dtype=np.int64)
    ordered_energies = tuple(row[0] for row in rows)
    minimum = ordered_energies[0]
    ground_degeneracy = sum(energy == minimum for energy in ordered_energies)
    best2 = ordered_indices[:2].copy()
    best8 = ordered_indices[: min(8, len(rows))].copy()
    ground = ordered_indices[:ground_degeneracy].copy()
    unique_ground = ground.copy() if ground_degeneracy == 1 else np.empty(0, np.int64)

    def target_hash(domain: str, indices: np.ndarray) -> str:
        packed = [
            int(structure.basis.packed_bits[int(index)]) for index in indices
        ]
        return _structured_hash(domain, packed)

    hashes = MappingProxyType(
        {
            "best2": target_hash("target-best2-packed-bits", best2),
            "best8": target_hash("target-best8-packed-bits", best8),
            "ground": target_hash("target-ground-packed-bits", ground),
        }
    )
    return TargetSets(
        ordered_feasible_indices=_readonly(ordered_indices),
        ordered_energy_numerators=ordered_energies,
        best2=_readonly(best2),
        best8=_readonly(best8),
        ground=_readonly(ground),
        unique_ground=_readonly(unique_ground),
        ground_degeneracy=ground_degeneracy,
        hashes=hashes,
    )


def _exact_targets(
    structure: ProblemStructure,
    objective_components: Sequence[tuple[tuple[int, ...], np.ndarray]],
    vectorized_exact: np.ndarray,
) -> TargetSets:
    rows: list[tuple[int, int, int]] = []
    for basis_index in structure.feasible_indices:
        index = int(basis_index)
        local_row = structure.basis.local_indices[index]
        # Python ``sum`` is arbitrary precision.  This is the normative target
        # energy path; the guarded vectorized int64 diagonal is cross-checked.
        exact_energy = -sum(
            int(table[tuple(int(local_row[block]) for block in blocks)])
            for blocks, table in objective_components
        )
        if exact_energy != int(vectorized_exact[index]):
            raise AssertionError("exact target energy disagrees with shell diagonal")
        rows.append(
            (exact_energy, int(structure.basis.packed_bits[index]), index)
        )
    return _target_sets_from_rows(structure, rows)


def derive_targets(instance: BCSTInstance) -> TargetSets:
    """Derive canonical targets deterministically in post-lock analysis.

    The exact diagonal is guarded against overflow during construction.  Each
    value is converted to a Python arbitrary-precision integer before sorting
    by ``(e_num, packed_unsigned_bitstring)``.
    """

    rows = [
        (
            int(instance.exact_raw_energy_numerators[int(index)]),
            int(instance.basis.packed_bits[int(index)]),
            int(index),
        )
        for index in instance.feasible_indices
    ]
    return _target_sets_from_rows(instance.structure, rows)


def _normalizations(
    exact: np.ndarray, conflicts: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ObjectiveNormalization,
]:
    raw_min = int(exact.min())
    raw_max = int(exact.max())
    raw_range = raw_max - raw_min
    if raw_range <= 0:
        raise ValueError("objective raw range must be positive")
    if math.ldexp(float(raw_range), -54) <= 1e-12:
        raise ValueError("objective raw range must exceed 1e-12")
    shifted = exact - np.int64(raw_min)
    H_F = shifted.astype(np.float64) / float(raw_range)
    H_C = conflicts.astype(np.float64) / 10.0

    # G has common exact denominator 10*raw_range and numerator
    # 10*(e-e_min) + 11*C_conf*raw_range.  Determine its exact extrema
    # with Python integers using only the extrema within each conflict sector.
    sector_extrema: list[tuple[int, int, int]] = []
    for conflict in range(CONFLICT_SPECTRAL_MAXIMUM + 1):
        sector = exact[conflicts == conflict]
        if sector.size:
            sector_extrema.append(
                (conflict, int(sector.min()), int(sector.max()))
            )
    g_min_numerator = min(
        10 * (sector_min - raw_min)
        + PENALTY_COEFFICIENT * conflict * raw_range
        for conflict, sector_min, _sector_max in sector_extrema
    )
    g_max_numerator = max(
        10 * (sector_max - raw_min)
        + PENALTY_COEFFICIENT * conflict * raw_range
        for conflict, _sector_min, sector_max in sector_extrema
    )
    g_range_numerator = g_max_numerator - g_min_numerator
    if g_range_numerator <= 0:
        raise ValueError("direct Hamiltonian G range must be positive")
    g_denominator = 10 * raw_range
    g_min = g_min_numerator / g_denominator
    g_max = g_max_numerator / g_denominator
    delta_G = g_range_numerator / g_denominator
    if delta_G <= 1e-12:
        raise ValueError("direct Hamiltonian G range must exceed 1e-12")

    H_D_obj = (
        10.0 * shifted.astype(np.float64) / float(g_range_numerator)
    )
    H_D_conf = (
        PENALTY_COEFFICIENT
        * conflicts.astype(np.float64)
        * float(raw_range)
        / float(g_range_numerator)
    )
    H_D = H_D_obj + H_D_conf - (
        g_min_numerator / float(g_range_numerator)
    )
    G = H_F + PENALTY_COEFFICIENT * H_C
    normalization = ObjectiveNormalization(
        raw_denominator=RAW_COEFFICIENT_DENOMINATOR,
        raw_min_numerator=raw_min,
        raw_max_numerator=raw_max,
        raw_range_numerator=raw_range,
        g_common_denominator=g_denominator,
        g_min_numerator=g_min_numerator,
        g_max_numerator=g_max_numerator,
        g_range_numerator=g_range_numerator,
        g_min=g_min,
        g_max=g_max,
        delta_G=delta_G,
    )
    return (
        _readonly(H_F),
        _readonly(G),
        _readonly(H_D),
        _readonly(H_D_obj),
        _readonly(H_D_conf),
        normalization,
    )


def _logical_resources(N: int, K: int, m4: int, m6: int) -> LogicalResources:
    B = BLOCK_COUNT
    C = N
    X = BLOCK_COUNT * math.comb(K, 2)
    O = 14 * m4 + 22 * m6
    S = N * N
    P1 = B + 12 * (C + X)
    return LogicalResources(
        N=N, m4=m4, m6=m6, B=B, C=C, X=X, O=O, S=S, P1=P1
    )


def build_instance(
    N: int,
    master_seed: int,
    d4_num: int,
    d4_den: int,
    r6_num: int,
    r6_den: int,
    *,
    include_targets: bool = False,
) -> BCSTInstance:
    """Build one complete deterministic contract-candidate-2 instance.

    ``d4_num/d4_den`` is the degree-4 pool density and
    ``r6_num/r6_den`` is the degree-6-to-degree-4 count ratio.  Counts use
    exact round-half-up arithmetic, and selected supports are prefixes of
    independently SHA-256-ranked P4 and P6 orders.  The safe default is
    target-blind.  Set ``include_targets=True`` only for post-lock analysis.
    """

    N = _as_index("N", N)
    master_seed = _as_index("master_seed", master_seed)
    d4_num = _as_index("d4_num", d4_num)
    d4_den = _as_index("d4_den", d4_den)
    r6_num = _as_index("r6_num", r6_num)
    r6_den = _as_index("r6_den", r6_den)
    if not isinstance(include_targets, (bool, np.bool_)):
        raise TypeError("include_targets must be bool")
    include_targets = bool(include_targets)
    if master_seed < 0:
        raise ValueError("master_seed must be nonnegative")
    if d4_den <= 0 or r6_den <= 0:
        raise ValueError("density and ratio denominators must be positive")
    if d4_num <= 0 or d4_num > d4_den:
        raise ValueError("degree-4 density must lie in (0,1]")
    if r6_num <= 0:
        raise ValueError("degree-6 ratio must be positive")

    structure = build_product_shell_structure(N)
    P4_order = _rank_supports(structure.P4, N, master_seed, 4)
    P6_order = _rank_supports(structure.P6, N, master_seed, 6)
    m4 = round_half_up_product(len(P4_order), d4_num, d4_den)
    m6 = round_half_up_product(m4, r6_num, r6_den)
    if m4 <= 0 or m4 > len(P4_order):
        raise ValueError("degree-4 term count is outside its eligible pool")
    if m6 <= 0 or m6 > len(P6_order):
        raise ValueError("degree-6 term count is outside its eligible pool")

    supports4 = P4_order[:m4]
    supports6 = P6_order[:m6]
    supports = supports4 + supports6
    degrees = (4,) * m4 + (6,) * m6
    coefficient_values = [
        coefficient_numerator(N, master_seed, degree, support)
        for degree, support in zip(degrees, supports)
    ]
    numerators = np.asarray(coefficient_values, dtype=np.int64)
    coefficients = np.ldexp(numerators.astype(np.float64), -54)
    if np.any(coefficients == 0.0):
        raise AssertionError("the coefficient family unexpectedly contains zero")
    if not all(
        float(value).hex()
        == math.ldexp(float(numerator), -54).hex()
        for value, numerator in zip(coefficients, coefficient_values)
    ):
        raise AssertionError("dyadic coefficient conversion was not exact")
    numerators = _readonly(numerators)
    coefficients = _readonly(coefficients)

    exact, objective_components = _exact_objective_diagonal(
        structure.basis, supports, numerators
    )
    raw = _readonly(
        np.ldexp(exact.astype(np.float64), -54).astype(np.float64, copy=False)
    )
    H_F, G, H_D, H_D_obj, H_D_conf, normalization = _normalizations(
        exact, structure.C_conf
    )
    targets = (
        _exact_targets(structure, objective_components, exact)
        if include_targets
        else None
    )
    resources = _logical_resources(N, structure.K, m4, m6)

    support_payload = [
        {"degree": degree, "qubits": list(support)}
        for degree, support in zip(degrees, supports)
    ]
    coefficient_payload = [
        {
            "degree": degree,
            "qubits": list(support),
            "numerator": int(numerator),
            "denominator": RAW_COEFFICIENT_DENOMINATOR,
            "float_hex": float(coefficient).hex(),
        }
        for degree, support, numerator, coefficient in zip(
            degrees, supports, numerators, coefficients
        )
    ]
    hashes = dict(structure.hashes)
    hashes.update(
        {
            "P4_order": _structured_hash(
                "P4-complete-SHA256-order", [list(term) for term in P4_order]
            ),
            "P6_order": _structured_hash(
                "P6-complete-SHA256-order", [list(term) for term in P6_order]
            ),
            "selected_supports": _structured_hash(
                "selected-degree4-degree6-supports", support_payload
            ),
            "selected_P4": _structured_hash(
                "selected-degree4-supports",
                [list(support) for support in supports4],
            ),
            "selected_P6": _structured_hash(
                "selected-degree6-supports",
                [list(support) for support in supports6],
            ),
            "coefficients": _structured_hash(
                "selected-coefficient-numerator-and-float-hex",
                coefficient_payload,
            ),
            "coefficient_numerators": _array_hash(
                "selected-coefficient-numerators", numerators, "<i8"
            ),
            "coefficient_float_hex": _structured_hash(
                "selected-coefficient-float-hex",
                [float(value).hex() for value in coefficients],
            ),
            "raw_objective_numerators": _array_hash(
                "raw-objective-exact-numerators", exact, "<i8"
            ),
            "raw_objective": _array_hash(
                "raw-objective-binary64", raw, "<f8"
            ),
            "H_F": _array_hash("normalized-objective-H_F", H_F, "<f8"),
            "H_D": _array_hash("normalized-direct-H_D", H_D, "<f8"),
        }
    )
    if targets is not None:
        hashes.update(
            {
                "target_best2": targets.hashes["best2"],
                "target_best8": targets.hashes["best8"],
                "target_ground": targets.hashes["ground"],
            }
        )
    return BCSTInstance(
        structure=structure,
        master_seed=master_seed,
        d4_numerator=d4_num,
        d4_denominator=d4_den,
        r6_numerator=r6_num,
        r6_denominator=r6_den,
        P4_order=P4_order,
        P6_order=P6_order,
        m4=m4,
        m6=m6,
        supports4=supports4,
        supports6=supports6,
        supports=supports,
        coefficient_numerators=numerators,
        coefficients=coefficients,
        exact_raw_energy_numerators=exact,
        C_raw=raw,
        H_F=H_F,
        G=G,
        H_D=H_D,
        H_D_obj=H_D_obj,
        H_D_conf=H_D_conf,
        normalization=normalization,
        targets=targets,
        resources=resources,
        hashes=MappingProxyType(hashes),
    )


def build_training_instance(
    N: int,
    master_seed: int,
    d4_num: int,
    d4_den: int,
    r6_num: int,
    r6_den: int,
) -> BCSTInstance:
    """Build a fail-closed target-blind instance for optimization jobs."""

    result = build_instance(
        N,
        master_seed,
        d4_num,
        d4_den,
        r6_num,
        r6_den,
        include_targets=False,
    )
    if result.targets is not None or any(
        key.startswith("target_") for key in result.hashes
    ):
        raise AssertionError("training instance unexpectedly exposes targets")
    return result


def clip_probability(probability: float, tolerance: float = 1e-12) -> float:
    """Validate a binary64 probability and apply only the contract endpoint clip."""

    raw = float(probability)
    tolerance = float(tolerance)
    if not math.isfinite(raw):
        raise ValueError("probability must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if raw < 0.0:
        if raw >= -tolerance:
            return 0.0
        raise ValueError("probability is below zero by more than tolerance")
    if raw > 1.0:
        if raw <= 1.0 + tolerance:
            return 1.0
        raise ValueError("probability is above one by more than tolerance")
    return raw


def rts99(probability: float, tolerance: float = 1e-12) -> int | float:
    """Return the exact contract integer RTS99 rule or positive infinity."""

    q = clip_probability(probability, tolerance)
    if q == 0.0:
        return math.inf
    if q == 1.0:
        return 1
    ratio = math.log(0.01) / math.log1p(-q)
    if math.isfinite(ratio):
        return max(1, int(math.ceil(ratio)))

    # The mathematical RTS is finite for every positive binary64 q, even when
    # the binary64 division overflows.  Decimal.from_float preserves q exactly;
    # 1,200 digits comfortably resolve subtraction at the smallest subnormal.
    with localcontext() as context:
        context.prec = 1_200
        decimal_q = Decimal.from_float(q)
        decimal_ratio = Decimal("0.01").ln() / (Decimal(1) - decimal_q).ln()
        return max(
            1,
            int(decimal_ratio.to_integral_value(rounding=ROUND_CEILING)),
        )


def total_sampling_cost(
    probability: float, terminal_ru: int, tolerance: float = 1e-12
) -> int | float:
    """Return ``RTS99 * terminal_RU`` with extended-real zero handling."""

    ru = _as_index("terminal_ru", terminal_ru)
    if ru <= 0:
        raise ValueError("terminal_ru must be positive")
    repetitions = rts99(probability, tolerance)
    if math.isinf(repetitions):
        return math.inf
    return int(repetitions) * ru


__all__ = [
    "BCSTInstance",
    "BLOCK_COUNT",
    "CONTRACT_SHA256",
    "LOCAL_WEIGHT",
    "LogicalResources",
    "ObjectiveNormalization",
    "ProductShellBasis",
    "ProductShellXY",
    "ProblemStructure",
    "RAW_COEFFICIENT_DENOMINATOR",
    "SUPPORTED_LAYOUTS",
    "TargetSets",
    "build_instance",
    "build_product_shell_structure",
    "build_training_instance",
    "canonical_json_bytes",
    "clip_probability",
    "coefficient_key_text",
    "coefficient_numerator",
    "derive_targets",
    "dyadic_coefficient",
    "round_half_up_product",
    "rts99",
    "support_order_digest",
    "support_order_text",
    "total_sampling_cost",
]
