"""Canonical resource-unit accounting for the nested-QAOA manuscript.

One resource unit (RU) is charged for each implemented diagonal phase term,
each transverse-field term, and each physical XY pair operation.  A history
mixer is synthesized as preparation, selective phase, and inverse preparation.
The BCST campaign includes its easy initial-state preparation in each
complete preparation and in every replay of the frozen preparation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence


TARGET_PROBABILITY = 0.99
ACCOUNTING_VERSION = "phi0-full-xy-history-replay-v2"


def repetitions_to_solution(
    success_probability: float,
    target_probability: float = TARGET_PROBABILITY,
) -> int | None:
    """Return the integer number of independent samples needed for the target."""
    p = float(success_probability)
    if not math.isfinite(p) or p <= 0.0:
        return None
    if p >= 1.0:
        return 1
    return int(math.ceil(math.log1p(-target_probability) / math.log1p(-p)))


def parse_depths(params: str) -> tuple[int, ...]:
    """Extract p, p1/p2, or a parenthesized depth tuple from a label."""
    label = str(params).strip()
    tuple_match = re.search(r"\(([^()]*)\)", label)
    if tuple_match:
        values = tuple(int(x) for x in re.findall(r"\d+", tuple_match.group(1)))
        if values:
            return values
    p1 = re.search(r"\bp1\s*=\s*(\d+)", label)
    p2 = re.search(r"\bp2\s*=\s*(\d+)", label)
    if p1 and p2:
        return int(p1.group(1)), int(p2.group(1))
    if p1:
        return (int(p1.group(1)),)
    p = re.search(r"\bp\s*=\s*(\d+)", label)
    if p:
        return (int(p.group(1)),)
    values = tuple(int(x) for x in re.findall(r"\d+", label))
    if values:
        return values
    raise ValueError(f"cannot parse circuit depth from {params!r}")


def parse_layout(layout: str | Sequence[int], n: int | None = None) -> tuple[int, ...]:
    if isinstance(layout, str):
        values = tuple(int(x) for x in re.findall(r"\d+", layout))
    else:
        values = tuple(int(x) for x in layout)
    if not values and n is not None:
        values = (int(n),)
    if not values:
        raise ValueError("empty block layout")
    if n is not None and sum(values) != int(n):
        raise ValueError(f"layout {values} does not sum to N={n}")
    return values


def xy_mixer_ru(layout: str | Sequence[int]) -> int:
    """Count all physical pair rotations in a complete within-block XY mixer."""
    return sum(k * (k - 1) // 2 for k in parse_layout(layout))


def direct_ru(depth: int, phase_ru: int, mixer_ru: int, p0_ru: int = 0) -> int:
    return int(p0_ru + depth * (phase_ru + mixer_ru))


def fixed_stage_ru(
    previous_preparation_ru: int,
    depth: int,
    phase_ru: int,
    mixer_ru: int,
) -> int:
    """Append a stage with a fixed mixer to an existing preparation."""
    return int(previous_preparation_ru + depth * (phase_ru + mixer_ru))


def history_stage_ru(
    previous_preparation_ru: int,
    depth: int,
    phase_ru: int,
    selective_phase_ru: int,
) -> int:
    """Append a history-mixer stage using U, a selective phase, and U inverse."""
    mixer = 2 * int(previous_preparation_ru) + int(selective_phase_ru)
    return int(previous_preparation_ru + depth * (int(phase_ru) + mixer))


def two_stage_history_ru(
    p1: int,
    p2: int,
    stage1_phase_ru: int,
    stage1_mixer_ru: int,
    stage2_phase_ru: int,
    selective_phase_ru: int,
    p0_ru: int = 0,
) -> int:
    preparation = direct_ru(p1, stage1_phase_ru, stage1_mixer_ru, p0_ru)
    return history_stage_ru(preparation, p2, stage2_phase_ru, selective_phase_ru)


def warm_start_ru(
    p1: int,
    p2: int,
    stage1_phase_ru: int,
    stage1_mixer_ru: int,
    stage2_phase_ru: int,
    stage2_mixer_ru: int,
    p0_ru: int = 0,
) -> int:
    preparation = direct_ru(p1, stage1_phase_ru, stage1_mixer_ru, p0_ru)
    return fixed_stage_ru(preparation, p2, stage2_phase_ru, stage2_mixer_ru)


def recursive_history_ru(
    depths: Sequence[int],
    phase_rus: Sequence[int],
    first_mixer_ru: int,
    selective_phase_ru: int,
    p0_ru: int = 0,
) -> int:
    """Count a first fixed-mixer stage followed by history-mixer stages."""
    if len(depths) != len(phase_rus) or not depths:
        raise ValueError("depths and phase_rus must have the same nonzero length")
    preparation = direct_ru(depths[0], phase_rus[0], first_mixer_ru, p0_ru)
    for depth, phase_ru in zip(depths[1:], phase_rus[1:]):
        preparation = history_stage_ru(preparation, depth, phase_ru, selective_phase_ru)
    return int(preparation)


def bcst_ru(
    algorithm: str,
    params: str,
    *,
    n: int,
    layout: str | Sequence[int],
    conflict_ru: int,
    objective_ru: int,
    p0_ru: int = 0,
) -> int:
    """Resource count for the two-stage block-constrained spin-tiling runs."""
    name = algorithm.lower()
    depths = parse_depths(params)
    mixer = xy_mixer_ru(layout)
    if "stage1" in name or "shell-only" in name:
        return direct_ru(depths[0], conflict_ru, mixer, p0_ru)
    if "warm-same-state" in name:
        if len(depths) != 2:
            raise ValueError(f"expected two depths for {algorithm} {params}")
        return warm_start_ru(
            depths[0],
            depths[1],
            conflict_ru,
            mixer,
            conflict_ru + objective_ru,
            mixer,
            p0_ru,
        )
    if "nested" in name:
        if len(depths) != 2:
            raise ValueError(f"expected two depths for {algorithm} {params}")
        stage2_phase = objective_ru + (conflict_ru if "guard" in name else 0)
        return two_stage_history_ru(
            depths[0],
            depths[1],
            conflict_ru,
            mixer,
            stage2_phase,
            n * n,
            p0_ru,
        )
    if name in {"std-xy", "d-xy"} or name.endswith("-xy"):
        return direct_ru(depths[0], conflict_ru + objective_ru, mixer, p0_ru)
    raise ValueError(f"unsupported BCST algorithm {algorithm!r}")


def soft_problem_ru(
    algorithm: str,
    params: str,
    *,
    n: int,
    coarse_phase_ru: int,
    fine_phase_ru: int,
    p0_ru: int = 0,
) -> int:
    """Resource count for the soft SBM and RNA coarse-to-fine experiments."""
    name = algorithm.lower()
    depths = parse_depths(params)
    if "coarse-only" in name:
        return direct_ru(depths[0], coarse_phase_ru, n, p0_ru)
    if "warm-coarse" in name:
        if len(depths) != 2:
            raise ValueError(f"expected two depths for {algorithm} {params}")
        return warm_start_ru(
            depths[0], depths[1], coarse_phase_ru, n, fine_phase_ru, n, p0_ru
        )
    if "nested" in name:
        if len(depths) != 2:
            raise ValueError(f"expected two depths for {algorithm} {params}")
        return two_stage_history_ru(
            depths[0], depths[1], coarse_phase_ru, n, fine_phase_ru, n * n, p0_ru
        )
    if any(token in name for token in ("qaoa", "decoupled", "grouped")):
        return direct_ru(depths[0], fine_phase_ru, n, p0_ru)
    raise ValueError(f"unsupported soft-problem algorithm {algorithm!r}")


def legacy_n18_ru(algorithm: str, params: str, p0_ru: int = 0) -> int:
    """Unified count for the recovered N=18 edge-objective benchmark."""
    n = 18
    one_hot_ru = 18
    conflict_ru = 18
    objective_ru = 153
    xy_ru = 18
    name = algorithm.lower()
    depths = parse_depths(params)
    if name == "xy-nested":
        return two_stage_history_ru(
            depths[0], depths[1], conflict_ru, xy_ru, objective_ru, n * n, p0_ru
        )
    if name == "nested":
        return recursive_history_ru(
            depths,
            (one_hot_ru, one_hot_ru + conflict_ru, objective_ru),
            n,
            n * n,
            p0_ru,
        )
    if name in {"std-xy", "d-xy"}:
        return direct_ru(depths[0], conflict_ru + objective_ru, xy_ru, p0_ru)
    if name == "std":
        return direct_ru(depths[0], one_hot_ru + conflict_ru + objective_ru, n, p0_ru)
    if name == "true-rqaoa":
        p = depths[0]
        final_match = re.search(r"final\s*=\s*(\d+)", params)
        final_variables = int(final_match.group(1)) if final_match else 6
        calls = max(1, n - final_variables)
        return calls * direct_ru(p, one_hot_ru + conflict_ru + objective_ru, n, p0_ru)
    raise ValueError(f"unsupported legacy N=18 algorithm {algorithm!r}")


def population_std(values: Iterable[float]) -> float:
    xs = [float(x) for x in values]
    if not xs:
        return math.nan
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
