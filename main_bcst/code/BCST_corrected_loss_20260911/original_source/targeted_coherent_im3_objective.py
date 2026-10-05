#!/usr/bin/env python3
"""Frozen coherent-IM3 concentration-risk objective for the 5x6 BCST shell.

The engineering definition and the service table in this module are copied
verbatim from ``FROZEN_COHERENT_IM3_RISK_OBJECTIVE_20260811.md``.  This module
contains no optimizer and does not rank states.  Optimization uses the same
fixed phase normalization as the earlier IM3 screen; that convention is part
of the target-blind campaign manifest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np


BLOCKS = 5
CHANNELS = 6
PHASE_SCALE = 2_048.0

PAIR_LISTS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 2), (2, 4)),
    ((2, 3), (3, 5)),
    ((0, 1), (3, 4)),
    ((1, 2), (4, 5)),
    ((0, 2), (2, 3)),
    ((1, 3), (3, 4)),
)

SERVICE_COST = np.asarray(
    [
        [12, 20, 9, 8, 9, 5],
        [3, 15, 7, 18, 9, 1],
        [8, 11, 17, 14, 18, 11],
        [19, 1, 3, 9, 9, 17],
        [7, 9, 6, 4, 19, 9],
    ],
    dtype=np.int64,
)
SERVICE_COST_COMPACT_JSON_SHA256 = (
    "ccb2b7956c6629ad1acf50b8b11dc3af39cf1b2f1a31e0b9a39de5ecb03a0940"
)


def _compact_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _table_for_matrix(
    matrix: Sequence[Sequence[int]], *, service_cost_seed: int | None = None
) -> dict[str, object]:
    """Build coefficient metadata without applying it to any search state."""

    array = np.asarray(matrix, dtype=np.int64)
    if array.shape != (BLOCKS, CHANNELS):
        raise ValueError("coherent-IM3 service table must have shape (5, 6)")
    if np.any(array < 1) or np.any(array > 20):
        raise ValueError("coherent-IM3 service costs must be integers in [1, 20]")
    values = array.tolist()
    payload: dict[str, object] = {
        "definition": "256*sum_{v,c} L[v,c]^2 + sum_{v,c} A[v,c] x[v,c]",
        "pair_lists_by_product_channel": [
            [list(pair) for pair in pairs] for pairs in PAIR_LISTS
        ],
        "service_cost": values,
        "service_cost_compact_json_sha256": _compact_sha256(values),
        "coherent_risk_weight": 256,
        "directed_inter_site_links": BLOCKS * (BLOCKS - 1),
        "phase_scale": PHASE_SCALE,
    }
    if service_cost_seed is not None:
        payload["service_cost_generator"] = "numpy.random.PCG64"
        payload["service_cost_seed"] = int(service_cost_seed)
        payload["service_cost_draw"] = "integers(low=1, high=21, size=(5,6))"
    return payload


def coefficient_table() -> dict[str, object]:
    """Return the prespecified development coefficients without ranking states."""

    payload = _table_for_matrix(SERVICE_COST)
    if payload["service_cost_compact_json_sha256"] != SERVICE_COST_COMPACT_JSON_SHA256:
        raise AssertionError("frozen coherent-IM3 service table hash changed")
    return payload


def coefficient_table_from_pcg64(seed: int) -> dict[str, object]:
    """Draw one unfiltered held-out service table from an explicit PCG64 seed."""

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    matrix = rng.integers(1, 21, size=(BLOCKS, CHANNELS), dtype=np.int64)
    return _table_for_matrix(matrix, service_cost_seed=int(seed))


def validated_coefficient_table(table: dict[str, object]) -> np.ndarray:
    """Validate fixed physics and return the table's service-cost matrix."""

    matrix = np.asarray(table.get("service_cost"), dtype=np.int64)
    expected = _table_for_matrix(
        matrix,
        service_cost_seed=(
            int(table["service_cost_seed"])
            if "service_cost_seed" in table
            else None
        ),
    )
    if table != expected:
        raise ValueError("coherent-IM3 coefficient table is invalid or changed")
    if "service_cost_seed" in table:
        regenerated = coefficient_table_from_pcg64(int(table["service_cost_seed"]))
        if table != regenerated:
            raise ValueError("service table does not match its frozen PCG64 seed")
    return matrix


def native_shell_components(
    screen: object, table: dict[str, object] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw ``(total, squared-risk, service)`` on the native shell."""

    masks = np.asarray(screen.masks_np, dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != BLOCKS:
        raise ValueError("native-shell mask table has the wrong shape")
    bits = (
        (masks[:, :, None] >> np.arange(CHANNELS, dtype=np.int64)[None, None, :])
        & 1
    ).astype(np.int64, copy=False)
    service_cost = (
        SERVICE_COST if table is None else validated_coefficient_table(table)
    )
    service = np.einsum("sbc,bc->s", bits, service_cost, dtype=np.int64)
    risk = np.zeros(masks.shape[0], dtype=np.int64)
    for victim in range(BLOCKS):
        for product, generators in enumerate(PAIR_LISTS):
            load = np.zeros(masks.shape[0], dtype=np.int64)
            for source in range(BLOCKS):
                if source == victim:
                    continue
                for left, right in generators:
                    load += bits[:, source, left] * bits[:, source, right]
            risk += bits[:, victim, product] * load * load
    total = 256 * risk + service
    return total, risk, service


def native_shell_diagonal(screen: object, table: dict[str, object] | None = None) -> np.ndarray:
    """Return one validated coherent-IM3 raw diagonal on the native shell."""

    frozen = coefficient_table() if table is None else table
    validated_coefficient_table(frozen)
    total, _, _ = native_shell_components(screen, frozen)
    return total


def objective_value(state: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    """Return raw ``(total, squared-risk, service)`` for one assignment."""

    occupied = tuple(frozenset(int(channel) for channel in pair) for pair in state)
    if len(occupied) != BLOCKS:
        raise ValueError("a coherent-IM3 state must contain five sites")
    service = sum(
        int(SERVICE_COST[site, channel])
        for site, pair in enumerate(occupied)
        for channel in pair
    )
    risk = 0
    for victim in range(BLOCKS):
        for product, generators in enumerate(PAIR_LISTS):
            if product not in occupied[victim]:
                continue
            load = sum(
                left in occupied[source] and right in occupied[source]
                for source in range(BLOCKS)
                if source != victim
                for left, right in generators
            )
            risk += load * load
    return 256 * risk + service, risk, service


def support_ledger() -> dict[str, object]:
    """Return the exact native-shell support ledger from the frozen document."""

    return {
        "unary_service_supports": 30,
        "cubic_single_hit_supports": 240,
        "quintic_cross_source_co_hit_supports": 720,
        "maximum_locality": 5,
        "objective_RU": 15_420,
        "Q_union_C_RU": 120,
        "Q_union_C_union_O_RU": 15_480,
        "separate_QC_plus_O_RU": 15_540,
        "uniform_diagonal_RU_by_locality": {"k1": 2, "k3": 10, "k5": 18},
    }


def resource_ledger() -> dict[str, object]:
    """Occupation-product resource components."""
    return {"initial": 5, "C": 30, "Q": 120, "O": 15420, "XY": 75, "selective_phase": 900}
