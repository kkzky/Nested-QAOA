"""Pure, fail-closed selection rules for the reviewer appendix campaign.

The functions in this module consume already-computed, target-aware tuning
costs or target-blind optimizer audit summaries.  They perform no simulation,
file I/O, random-number generation, or mutation.  Integer products are kept
exact throughout depth selection; floating-point arithmetic is confined to
the prospectively specified normalized optimizer-energy tests.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Mapping, Sequence, TypeAlias


BASE_DEPTH_GRID = (1, 2, 4, 8, 12, 16, 24, 36)
FIRST_BOUNDARY_EXTENSION = (48, 64)
SECOND_BOUNDARY_EXTENSION = (80, 96)
ALLOWED_BUDGETS = (400, 800, 1600)
PLAN_SHA256 = "20476043ADF7233C5CD721E620EC4CD620CA58C5BECFD1A192BD0495C25BAF08"
STAGE1_REGENERATION_EVIDENCE_SCHEMA = (
    "lp-qaoa-appendix-stage1-regeneration-evidence-v1"
)
STAGE1_SEEDS = {25: (840_025_000,), 30: (840_030_000,)}
TUNING_SEEDS = {
    25: (840_025_011, 840_025_012),
    30: (840_030_011, 840_030_012),
}
CAUSAL_SEEDS = {
    25: (840_025_201, 840_025_202, 840_025_203),
    30: (840_030_201, 840_030_202, 840_030_203),
}
CAUSAL_DEPTHS = {25: (1, 12, 24), 30: (1, 24, 80)}

EXTERNAL_METHODS = frozenset(
    {
        "learned_projector",
        "ordinary_native_direct",
        "decoupled_native_direct",
        "same_stage1_state_direct",
        "native_grover_d",
    }
)
CAUSAL_METHODS = frozenset(
    {
        "learned_projector",
        "projector_hd",
        "known_projector_hd",
        "xy_hf",
        "same_stage1_state_direct",
    }
)

NORMALIZATION_FLOOR = 1e-12
BUDGET_IMPROVEMENT_THRESHOLD = 0.02
TERMINAL_DROP_THRESHOLD = 0.01

AuditRole: TypeAlias = Literal[
    "stage1", "external_stage2", "causal_stage2"
]


class SelectionError(ValueError):
    """Raised when selection evidence violates the frozen protocol."""


def _plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SelectionError(f"{field} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise SelectionError(f"{field} must be an integer") from exc
    if result < minimum:
        raise SelectionError(f"{field} must be at least {minimum}")
    return result


def _finite_real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelectionError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionError(f"{field} must be finite")
    return result


def _canonical_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
    ):
        raise SelectionError(f"{field} must be a nonempty canonical string")
    return value


def _sha256(value: object, *, field: str) -> str:
    digest = _canonical_text(value, field=field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SelectionError(f"{field} must be a 64-character SHA-256 digest")
    return digest


@dataclass(frozen=True)
class DepthCostRecord:
    """One locked tuning or causal cell used as selection evidence."""

    cell_id: str
    method: str
    N: int
    budget: int
    depth: int
    problem_seed: int
    best2_logical_cost: int | None
    locked_target_result_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cell_id", _canonical_text(self.cell_id, field="cell_id")
        )
        object.__setattr__(
            self, "method", _canonical_text(self.method, field="method")
        )
        size = _plain_int(self.N, field="N", minimum=1)
        if size not in (25, 30):
            raise SelectionError("N must be exactly 25 or 30")
        budget = _plain_int(self.budget, field="budget", minimum=1)
        if budget not in ALLOWED_BUDGETS:
            raise SelectionError(f"budget must be one of {ALLOWED_BUDGETS}")
        object.__setattr__(self, "N", size)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(
            self, "depth", _plain_int(self.depth, field="depth", minimum=1)
        )
        object.__setattr__(
            self,
            "problem_seed",
            _plain_int(self.problem_seed, field="problem_seed", minimum=0),
        )
        if self.best2_logical_cost is not None:
            object.__setattr__(
                self,
                "best2_logical_cost",
                _plain_int(
                    self.best2_logical_cost,
                    field="best2 logical cost",
                    minimum=1,
                ),
            )
        object.__setattr__(
            self,
            "locked_target_result_sha256",
            _sha256(
                self.locked_target_result_sha256,
                field="locked_target_result_sha256",
            ),
        )


@dataclass(frozen=True)
class DepthDecision:
    """Boundary-aware disposition of one method-size depth scan."""

    method: str
    N: int
    budget: int
    problem_seeds: tuple[int, int]
    source_cell_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    provisional_depth: int | None
    selected_depth: int | None
    products: tuple[tuple[int, int | None], ...]
    extension_48_64_required: bool
    extension_80_96_required: bool
    finalized: bool
    unresolved: bool
    reason: str | None

    def product_at(self, depth: int) -> int | None:
        requested = _plain_int(depth, field="depth", minimum=1)
        for observed_depth, product in self.products:
            if observed_depth == requested:
                return product
        raise SelectionError(f"depth {requested} is absent from this decision")


def _materialize_depth_records(
    raw_records: Sequence[DepthCostRecord],
    *,
    field: str,
) -> tuple[DepthCostRecord, ...]:
    if isinstance(raw_records, Mapping):
        raise SelectionError(f"{field} requires bound DepthCostRecord evidence")
    try:
        records = tuple(raw_records)
    except TypeError as exc:
        raise SelectionError(f"{field} must be a record sequence") from exc
    if not records or not all(isinstance(record, DepthCostRecord) for record in records):
        raise SelectionError(f"{field} must contain DepthCostRecord values")
    return records


def _validate_depth_grid(
    raw_records: Sequence[DepthCostRecord],
    *,
    expected_depths: tuple[int, ...],
    field: str,
    expected_metadata: tuple[str, int, int] | None = None,
    expected_seeds: tuple[int, ...] | None = None,
) -> tuple[dict[int, int | None], tuple[DepthCostRecord, ...]]:
    records = _materialize_depth_records(raw_records, field=field)
    first = records[0]
    metadata = (first.method, first.N, first.budget)
    if expected_metadata is not None and metadata != expected_metadata:
        raise SelectionError(f"{field} method/N/budget does not match the base scan")
    seeds = expected_seeds if expected_seeds is not None else TUNING_SEEDS[first.N]
    if len(records) != len(expected_depths) * len(seeds):
        raise SelectionError(
            f"{field} requires every planned depth/tuning-seed cell"
        )

    by_key: dict[tuple[int, int], DepthCostRecord] = {}
    cell_ids: set[str] = set()
    for record in records:
        if (record.method, record.N, record.budget) != metadata:
            raise SelectionError(f"{field} must have uniform method/N/budget")
        key = (record.depth, record.problem_seed)
        if key in by_key:
            raise SelectionError(f"duplicate {field} depth/seed cell {key}")
        if record.cell_id in cell_ids:
            raise SelectionError(f"duplicate {field} cell_id {record.cell_id!r}")
        by_key[key] = record
        cell_ids.add(record.cell_id)

    expected_keys = {
        (depth, problem_seed)
        for depth in expected_depths
        for problem_seed in seeds
    }
    if set(by_key) != expected_keys:
        raise SelectionError(
            f"{field} must contain exactly depths {list(expected_depths)} and seeds {list(seeds)}"
        )

    products: dict[int, int | None] = {}
    for depth in expected_depths:
        costs = tuple(
            by_key[(depth, problem_seed)].best2_logical_cost
            for problem_seed in seeds
        )
        products[depth] = (
            None if any(cost is None for cost in costs) else math.prod(costs)  # type: ignore[arg-type]
        )
    return products, tuple(
        by_key[key] for key in sorted(by_key)
    )


def _decision_evidence(
    records: Sequence[DepthCostRecord],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(records, key=lambda row: (row.depth, row.problem_seed)))
    return (
        tuple(record.cell_id for record in ordered),
        tuple(record.locked_target_result_sha256 for record in ordered),
    )


def _depth_decision(
    *,
    metadata: tuple[str, int, int],
    records: Sequence[DepthCostRecord],
    provisional_depth: int | None,
    selected_depth: int | None,
    products: Mapping[int, int | None],
    extension_48_64_required: bool,
    extension_80_96_required: bool,
    finalized: bool,
    unresolved: bool,
    reason: str | None,
) -> DepthDecision:
    method, N, budget = metadata
    cell_ids, hashes = _decision_evidence(records)
    return DepthDecision(
        method=method,
        N=N,
        budget=budget,
        problem_seeds=TUNING_SEEDS[N],
        source_cell_ids=cell_ids,
        source_hashes=hashes,
        provisional_depth=provisional_depth,
        selected_depth=selected_depth,
        products=tuple(sorted(products.items())),
        extension_48_64_required=extension_48_64_required,
        extension_80_96_required=extension_80_96_required,
        finalized=finalized,
        unresolved=unresolved,
        reason=reason,
    )


def _finite_minimum(
    products: Mapping[int, int | None],
) -> tuple[int | None, int | None]:
    finite = tuple(
        (int(product), int(depth))
        for depth, product in products.items()
        if product is not None
    )
    if not finite:
        return None, None
    product, depth = min(finite, key=lambda row: (row[0], row[1]))
    return depth, product


def _boundary_trigger(
    products: Mapping[int, int | None],
    *,
    previous_depth: int,
    boundary_depth: int,
) -> bool:
    selected, minimum = _finite_minimum(products)
    if selected is None or minimum is None:
        return False
    boundary = products.get(boundary_depth)
    previous = products.get(previous_depth)
    if boundary is None:
        return False

    # Exact cross-multiplication implements boundary <= 1.10 * minimum
    # without a floating-point rounding decision.
    within_ten_percent = 10 * boundary <= 11 * minimum
    # A finite boundary following an infinite preceding point is a decrease.
    nonincreasing = previous is None or boundary <= previous
    return selected == boundary_depth or (
        within_ten_percent and nonincreasing
    )


def select_depth_with_boundary(
    base_records: Sequence[DepthCostRecord],
    *,
    first_extension_records: Sequence[DepthCostRecord] | None = None,
    second_extension_records: Sequence[DepthCostRecord] | None = None,
) -> DepthDecision:
    """Select a depth from two-seed exact products and frozen extensions.

    This is the common rule for the provisional NativeGrover-d scan and for
    mandatory reselection of an externally validated method after its budget
    changes.  Supplying untriggered, partial, or out-of-order extension data is
    a protocol error rather than ignorable extra evidence.
    """

    base_products, base_evidence = _validate_depth_grid(
        base_records,
        expected_depths=BASE_DEPTH_GRID,
        field="base depth scan",
    )
    metadata = (
        base_evidence[0].method,
        base_evidence[0].N,
        base_evidence[0].budget,
    )
    provisional, base_minimum = _finite_minimum(base_products)
    if provisional is None or base_minimum is None:
        if (
            first_extension_records is not None
            or second_extension_records is not None
        ):
            raise SelectionError(
                "extension evidence is invalid when every base product is infinite"
            )
        return _depth_decision(
            metadata=metadata,
            records=base_evidence,
            provisional_depth=None,
            selected_depth=None,
            products=base_products,
            extension_48_64_required=False,
            extension_80_96_required=False,
            finalized=True,
            unresolved=True,
            reason="all_products_infinite",
        )

    first_required = _boundary_trigger(
        base_products, previous_depth=24, boundary_depth=36
    )
    if not first_required:
        if (
            first_extension_records is not None
            or second_extension_records is not None
        ):
            raise SelectionError("p48/p64 evidence supplied without the p36 trigger")
        return _depth_decision(
            metadata=metadata,
            records=base_evidence,
            provisional_depth=provisional,
            selected_depth=provisional,
            products=base_products,
            extension_48_64_required=False,
            extension_80_96_required=False,
            finalized=True,
            unresolved=False,
            reason=None,
        )

    if first_extension_records is None:
        if second_extension_records is not None:
            raise SelectionError("p80/p96 evidence cannot precede p48/p64 evidence")
        return _depth_decision(
            metadata=metadata,
            records=base_evidence,
            provisional_depth=provisional,
            selected_depth=None,
            products=base_products,
            extension_48_64_required=True,
            extension_80_96_required=False,
            finalized=False,
            unresolved=False,
            reason="extension_48_64_required",
        )

    first_products, first_evidence = _validate_depth_grid(
        first_extension_records,
        expected_depths=FIRST_BOUNDARY_EXTENSION,
        field="first extension",
        expected_metadata=metadata,
    )
    if set(record.cell_id for record in base_evidence).intersection(
        record.cell_id for record in first_evidence
    ):
        raise SelectionError("depth evidence reuses a cell_id across scan stages")
    evidence_64 = base_evidence + first_evidence
    through_64 = {**base_products, **first_products}
    selected_64, _ = _finite_minimum(through_64)
    if selected_64 is None:
        raise AssertionError("a finite base minimum disappeared")
    second_required = _boundary_trigger(
        through_64, previous_depth=48, boundary_depth=64
    )
    if not second_required:
        if second_extension_records is not None:
            raise SelectionError("p80/p96 evidence supplied without the p64 trigger")
        return _depth_decision(
            metadata=metadata,
            records=evidence_64,
            provisional_depth=provisional,
            selected_depth=selected_64,
            products=through_64,
            extension_48_64_required=True,
            extension_80_96_required=False,
            finalized=True,
            unresolved=False,
            reason=None,
        )

    if second_extension_records is None:
        return _depth_decision(
            metadata=metadata,
            records=evidence_64,
            provisional_depth=provisional,
            selected_depth=None,
            products=through_64,
            extension_48_64_required=True,
            extension_80_96_required=True,
            finalized=False,
            unresolved=False,
            reason="extension_80_96_required",
        )

    second_products, second_evidence = _validate_depth_grid(
        second_extension_records,
        expected_depths=SECOND_BOUNDARY_EXTENSION,
        field="second extension",
        expected_metadata=metadata,
    )
    prior_ids = {record.cell_id for record in evidence_64}
    if prior_ids.intersection(record.cell_id for record in second_evidence):
        raise SelectionError("depth evidence reuses a cell_id across scan stages")
    evidence_96 = evidence_64 + second_evidence
    through_96 = {**through_64, **second_products}
    selected_96, _ = _finite_minimum(through_96)
    if selected_96 is None:
        raise AssertionError("a finite base minimum disappeared")
    unresolved = _boundary_trigger(
        through_96, previous_depth=80, boundary_depth=96
    )
    return _depth_decision(
        metadata=metadata,
        records=evidence_96,
        provisional_depth=provisional,
        selected_depth=selected_96,
        products=through_96,
        extension_48_64_required=True,
        extension_80_96_required=True,
        finalized=True,
        unresolved=unresolved,
        reason="final_depth_boundary_p96" if unresolved else None,
    )


@dataclass(frozen=True)
class BudgetAuditRecord:
    """Sufficient statistics for one budget and one frozen audit cell.

    ``best_so_far_at_tail_start`` and ``best_so_far_at_end`` refer to the
    selected restart at checkpoints 80% and 100% of ``budget`` respectively.
    The latter must equal the selected energy for that restart.
    """

    cell_id: str
    method: str
    N: int
    problem_seed: int
    budget: int
    selected_energy: float
    h_min: float
    h_max: float
    selected_checkpoint: int
    best_so_far_at_tail_start: float
    best_so_far_at_end: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cell_id", _canonical_text(self.cell_id, field="cell_id")
        )
        object.__setattr__(
            self, "method", _canonical_text(self.method, field="method")
        )
        size = _plain_int(self.N, field="N", minimum=1)
        if size not in (25, 30):
            raise SelectionError("N must be exactly 25 or 30")
        object.__setattr__(self, "N", size)
        object.__setattr__(
            self,
            "problem_seed",
            _plain_int(self.problem_seed, field="problem_seed", minimum=0),
        )
        budget = _plain_int(self.budget, field="budget", minimum=1)
        if budget not in ALLOWED_BUDGETS:
            raise SelectionError(f"budget must be one of {ALLOWED_BUDGETS}")
        checkpoint = _plain_int(
            self.selected_checkpoint, field="selected_checkpoint", minimum=0
        )
        if checkpoint > budget:
            raise SelectionError("selected_checkpoint exceeds budget")
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "selected_checkpoint", checkpoint)

        for field in (
            "selected_energy",
            "h_min",
            "h_max",
            "best_so_far_at_tail_start",
            "best_so_far_at_end",
        ):
            object.__setattr__(
                self,
                field,
                _finite_real(getattr(self, field), field=field),
            )
        if self.h_max < self.h_min:
            raise SelectionError("h_max must be at least h_min")
        if not math.isfinite(self.h_max - self.h_min):
            raise SelectionError("Hamiltonian range must be finite")
        if self.best_so_far_at_end > self.best_so_far_at_tail_start:
            raise SelectionError("best-so-far energy cannot increase")
        if self.selected_energy != self.best_so_far_at_end:
            raise SelectionError(
                "selected_energy must equal best_so_far_at_end for the selected restart"
            )


def _normalization(record: BudgetAuditRecord) -> float:
    return max(record.h_max - record.h_min, NORMALIZATION_FLOOR)


def terminal_progress(record: BudgetAuditRecord) -> bool:
    """Apply the V3 substantive final-20%-tail-drop predicate."""

    if not isinstance(record, BudgetAuditRecord):
        raise SelectionError("terminal_progress requires a BudgetAuditRecord")
    normalized_tail_drop = (
        record.best_so_far_at_tail_start - record.best_so_far_at_end
    ) / _normalization(record)
    return normalized_tail_drop >= TERMINAL_DROP_THRESHOLD


def terminal_checkpoint_warning(record: BudgetAuditRecord) -> bool:
    """Report a final-10%-checkpoint warning without affecting decisions."""

    if not isinstance(record, BudgetAuditRecord):
        raise SelectionError(
            "terminal_checkpoint_warning requires a BudgetAuditRecord"
        )
    return 10 * record.selected_checkpoint >= 9 * record.budget


def normalized_selected_energy_improvement(
    short: BudgetAuditRecord,
    long: BudgetAuditRecord,
) -> float:
    """Return the frozen normalized selected-energy improvement for one cell."""

    if not isinstance(short, BudgetAuditRecord) or not isinstance(
        long, BudgetAuditRecord
    ):
        raise SelectionError("improvement requires two BudgetAuditRecord values")
    if (short.budget, long.budget) not in ((400, 800), (800, 1600)):
        raise SelectionError("budget comparison must be 400-to-800 or 800-to-1600")
    if short.cell_id != long.cell_id:
        raise SelectionError("budget records must identify the same audit cell")
    if (
        short.method,
        short.N,
        short.problem_seed,
    ) != (
        long.method,
        long.N,
        long.problem_seed,
    ):
        raise SelectionError("paired budget records must have identical method/N/seed")
    if short.h_min != long.h_min or short.h_max != long.h_max:
        raise SelectionError("Hamiltonian bounds must match across paired budgets")
    value = (short.selected_energy - long.selected_energy) / _normalization(short)
    if not math.isfinite(value):
        raise SelectionError("normalized improvement must be finite")
    return value


@dataclass(frozen=True)
class BudgetDecision:
    """Prospective optimizer-budget disposition for one method-size stratum."""

    role: AuditRole
    method: str
    N: int
    problem_seeds: tuple[int, ...]
    adopted_budget: int | None
    mean_improvement_400_to_800: float
    mean_improvement_800_to_1600: float | None
    terminal_progress_800: tuple[tuple[str, bool], ...]
    terminal_progress_1600: tuple[tuple[str, bool], ...]
    terminal_checkpoint_warnings_800: tuple[tuple[str, bool], ...]
    terminal_checkpoint_warnings_1600: tuple[tuple[str, bool], ...]
    triggered_1600: bool
    finalized: bool
    optimizer_inconclusive: bool
    stage1_regeneration_required: bool
    full_depth_reselection_required: bool
    causal_grid_replay_required: bool
    reason: str


def _validated_budget_records(
    records_by_budget: Mapping[int, Sequence[BudgetAuditRecord]],
    *,
    role: AuditRole,
) -> tuple[
    dict[int, dict[str, BudgetAuditRecord]],
    str,
    int,
    tuple[int, ...],
]:
    if role not in ("stage1", "external_stage2", "causal_stage2"):
        raise SelectionError(
            "role must be stage1, external_stage2, or causal_stage2"
        )
    if not isinstance(records_by_budget, Mapping):
        raise SelectionError("records_by_budget must be a mapping")
    normalized: dict[int, dict[str, BudgetAuditRecord]] = {}
    for raw_budget, raw_records in records_by_budget.items():
        budget = _plain_int(raw_budget, field="budget key", minimum=1)
        if budget not in ALLOWED_BUDGETS:
            raise SelectionError(f"budget key must be one of {ALLOWED_BUDGETS}")
        if budget in normalized:
            raise SelectionError(f"duplicate budget {budget}")
        try:
            records = tuple(raw_records)
        except TypeError as exc:
            raise SelectionError("each budget must map to a record sequence") from exc
        expected_count = 1 if role == "stage1" else 2
        if len(records) != expected_count:
            noun = "cell" if expected_count == 1 else "tuning-seed cells"
            raise SelectionError(
                f"{role} budget {budget} requires exactly {expected_count} {noun}"
            )
        by_id: dict[str, BudgetAuditRecord] = {}
        for record in records:
            if not isinstance(record, BudgetAuditRecord):
                raise SelectionError("audit entries must be BudgetAuditRecord values")
            if record.budget != budget:
                raise SelectionError("record budget does not match its mapping key")
            if record.cell_id in by_id:
                raise SelectionError(f"duplicate audit cell {record.cell_id!r}")
            by_id[record.cell_id] = record
        normalized[budget] = by_id

    if 400 not in normalized or 800 not in normalized:
        raise SelectionError("complete 400- and 800-step audit records are mandatory")
    if set(normalized) not in ({400, 800}, {400, 800, 1600}):
        raise SelectionError("unexpected budget evidence")
    expected_ids = set(normalized[400])
    for budget, records in normalized.items():
        if set(records) != expected_ids:
            raise SelectionError(
                f"audit cell identifiers do not pair at budget {budget}"
            )
    first = next(iter(normalized[400].values()))
    method, size = first.method, first.N
    expected_seeds = STAGE1_SEEDS[size] if role == "stage1" else TUNING_SEEDS[size]
    if role == "stage1" and method != "stage1":
        raise SelectionError("Stage1 audit records must use method='stage1'")
    if role == "external_stage2" and method not in EXTERNAL_METHODS:
        raise SelectionError("external Stage2 audit uses an ineligible method")
    if role == "causal_stage2" and method not in CAUSAL_METHODS:
        raise SelectionError("causal Stage2 audit uses an ineligible method")

    base_id_seed = {
        cell_id: record.problem_seed
        for cell_id, record in normalized[400].items()
    }
    for budget, records in normalized.items():
        if any(record.method != method or record.N != size for record in records.values()):
            raise SelectionError("budget audit must have uniform method and N")
        observed_seeds = tuple(sorted(record.problem_seed for record in records.values()))
        if observed_seeds != expected_seeds:
            raise SelectionError(
                f"budget {budget} must use frozen audit seeds {list(expected_seeds)}"
            )
        if {
            cell_id: record.problem_seed for cell_id, record in records.items()
        } != base_id_seed:
            raise SelectionError("audit cell IDs must retain the same seed pairing")
    return normalized, method, size, expected_seeds


def _mean_improvement(
    short_records: Mapping[str, BudgetAuditRecord],
    long_records: Mapping[str, BudgetAuditRecord],
) -> float:
    identifiers = tuple(sorted(short_records))
    values = tuple(
        normalized_selected_energy_improvement(
            short_records[cell_id], long_records[cell_id]
        )
        for cell_id in identifiers
    )
    return math.fsum(values) / len(values)


def _depth_flags(
    role: AuditRole, adopted_budget: int | None
) -> tuple[bool, bool]:
    changed = adopted_budget is not None and adopted_budget != 400
    return (
        role == "external_stage2" and changed,
        role == "causal_stage2" and changed,
    )


def _stage1_regeneration_flag(
    role: AuditRole, adopted_budget: int | None
) -> bool:
    return (
        role == "stage1"
        and adopted_budget is not None
        and adopted_budget != 400
    )


@dataclass(frozen=True)
class Stage1RegenerationEvidence:
    """Cryptographic binding from a budget decision to the frozen phi record."""

    N: int
    phase: str
    adopted_budget: int
    kdf_checkpoint_zero_sha256: tuple[str, str, str]
    selected_amplitude_sha256: str
    plan_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        size = _plain_int(self.N, field="Stage1 evidence N", minimum=1)
        if size not in (25, 30):
            raise SelectionError("Stage1 evidence N must be exactly 25 or 30")
        if self.phase != "stage1_generation":
            raise SelectionError("Stage1 evidence phase must be stage1_generation")
        budget = _plain_int(
            self.adopted_budget, field="Stage1 evidence adopted budget", minimum=1
        )
        if budget not in ALLOWED_BUDGETS:
            raise SelectionError(
                f"Stage1 evidence budget must be one of {ALLOWED_BUDGETS}"
            )
        hashes = tuple(self.kdf_checkpoint_zero_sha256)
        if len(hashes) != 3:
            raise SelectionError("Stage1 evidence requires exactly three KDF hashes")
        normalized_hashes = tuple(
            _sha256(value, field=f"Stage1 KDF checkpoint-zero hash {index}")
            for index, value in enumerate(hashes)
        )
        if len(set(normalized_hashes)) != 3:
            raise SelectionError("Stage1 KDF checkpoint-zero hashes must be distinct")
        object.__setattr__(self, "N", size)
        object.__setattr__(self, "adopted_budget", budget)
        object.__setattr__(self, "kdf_checkpoint_zero_sha256", normalized_hashes)
        object.__setattr__(
            self,
            "selected_amplitude_sha256",
            _sha256(
                self.selected_amplitude_sha256,
                field="Stage1 selected amplitude hash",
            ),
        )
        plan_hash = _sha256(self.plan_sha256, field="Stage1 evidence plan hash")
        if plan_hash != PLAN_SHA256.lower():
            raise SelectionError("Stage1 evidence is bound to the wrong plan hash")
        object.__setattr__(self, "plan_sha256", plan_hash)
        object.__setattr__(
            self,
            "record_sha256",
            _sha256(self.record_sha256, field="Stage1 record hash"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema": STAGE1_REGENERATION_EVIDENCE_SCHEMA,
            "N": self.N,
            "phase": self.phase,
            "adopted_budget": self.adopted_budget,
            "kdf_checkpoint_zero_sha256": list(
                self.kdf_checkpoint_zero_sha256
            ),
            "selected_amplitude_sha256": self.selected_amplitude_sha256,
            "plan_sha256": self.plan_sha256,
            "record_sha256": self.record_sha256,
        }

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "Stage1RegenerationEvidence":
        if not isinstance(value, Mapping):
            raise SelectionError("Stage1 regeneration evidence must be a mapping")
        expected = {
            "schema",
            "N",
            "phase",
            "adopted_budget",
            "kdf_checkpoint_zero_sha256",
            "selected_amplitude_sha256",
            "plan_sha256",
            "record_sha256",
        }
        if set(value) != expected:
            raise SelectionError("Stage1 regeneration evidence has incorrect fields")
        if value.get("schema") != STAGE1_REGENERATION_EVIDENCE_SCHEMA:
            raise SelectionError("Stage1 regeneration evidence has the wrong schema")
        raw_hashes = value.get("kdf_checkpoint_zero_sha256")
        if not isinstance(raw_hashes, (list, tuple)):
            raise SelectionError("Stage1 KDF hashes must be a sequence")
        return cls(
            N=value.get("N"),  # type: ignore[arg-type]
            phase=value.get("phase"),  # type: ignore[arg-type]
            adopted_budget=value.get("adopted_budget"),  # type: ignore[arg-type]
            kdf_checkpoint_zero_sha256=tuple(raw_hashes),  # type: ignore[arg-type]
            selected_amplitude_sha256=value.get(  # type: ignore[arg-type]
                "selected_amplitude_sha256"
            ),
            plan_sha256=value.get("plan_sha256"),  # type: ignore[arg-type]
            record_sha256=value.get("record_sha256"),  # type: ignore[arg-type]
        )


def _stage1_evidence_from_record(
    record: Mapping[str, object],
    *,
    record_sha256: str,
) -> Stage1RegenerationEvidence:
    if not isinstance(record, Mapping):
        raise SelectionError("Stage1 job record must be a mapping")
    if record.get("cell_type") != "stage1" or record.get("method") != "stage1":
        raise SelectionError("Stage1 evidence requires an actual Stage1 job record")
    if record.get("target_blind") is not True:
        raise SelectionError("Stage1 job record must remain target blind")
    size = _plain_int(record.get("N"), field="Stage1 record N", minimum=1)
    if size not in STAGE1_SEEDS:
        raise SelectionError("Stage1 record N must be exactly 25 or 30")
    if record.get("problem_seed") != STAGE1_SEEDS[size][0]:
        raise SelectionError("Stage1 record uses the wrong frozen seed")
    phase = record.get("phase")
    if phase != "stage1_generation":
        raise SelectionError("Stage1 record has the wrong phase")
    if record.get("depth") != 12:
        raise SelectionError("Stage1 record depth must be 12")
    budget = _plain_int(
        record.get("adopted_budget"), field="Stage1 record adopted budget", minimum=1
    )
    if budget not in ALLOWED_BUDGETS:
        raise SelectionError("Stage1 record has a noncandidate optimizer budget")
    plan_hash = _sha256(record.get("plan_sha256"), field="Stage1 record plan hash")
    if plan_hash != PLAN_SHA256.lower():
        raise SelectionError("Stage1 record is bound to the wrong plan")

    optimization = record.get("optimization")
    if not isinstance(optimization, Mapping):
        raise SelectionError("Stage1 record lacks its optimization evidence")
    if (
        optimization.get("optimizer") != "Adam"
        or optimization.get("restart_count") != 3
        or optimization.get("executed_updates_per_restart") != budget
        or optimization.get("early_stopping") is not False
        or optimization.get("continuation") is not False
    ):
        raise SelectionError("Stage1 optimizer record disagrees with adopted budget")
    raw_restarts = optimization.get("restarts")
    if not isinstance(raw_restarts, (list, tuple)) or len(raw_restarts) != 3:
        raise SelectionError("Stage1 record requires exactly three restart records")
    kdf_hashes: list[str] = []
    for expected_index, restart in enumerate(raw_restarts):
        if not isinstance(restart, Mapping):
            raise SelectionError("Stage1 restart evidence must be a mapping")
        if (
            restart.get("restart_index") != expected_index
            or restart.get("role") != "independent_random"
            or restart.get("executed_updates") != budget
        ):
            raise SelectionError("Stage1 restart ordering or budget is invalid")
        initializer_hash = _sha256(
            restart.get("initializer_sha256"),
            field=f"Stage1 restart {expected_index} initializer hash",
        )
        trace = restart.get("trace")
        if not isinstance(trace, (list, tuple)) or not trace:
            raise SelectionError("Stage1 restart trace is missing checkpoint zero")
        checkpoint_zero = trace[0]
        if not isinstance(checkpoint_zero, Mapping):
            raise SelectionError("Stage1 checkpoint-zero evidence must be a mapping")
        if checkpoint_zero.get("checkpoint") != 0:
            raise SelectionError("Stage1 restart trace does not begin at checkpoint zero")
        checkpoint_hash = _sha256(
            checkpoint_zero.get("parameter_sha256"),
            field=f"Stage1 restart {expected_index} checkpoint-zero hash",
        )
        if checkpoint_hash != initializer_hash:
            raise SelectionError("Stage1 KDF initializer and checkpoint zero differ")
        kdf_hashes.append(initializer_hash)

    selected_state = record.get("selected_state")
    if not isinstance(selected_state, Mapping):
        raise SelectionError("Stage1 record lacks selected-state evidence")
    amplitude_hash = _sha256(
        selected_state.get("amplitude_sha256"),
        field="Stage1 selected amplitude hash",
    )
    return Stage1RegenerationEvidence(
        N=size,
        phase=phase,
        adopted_budget=budget,
        kdf_checkpoint_zero_sha256=tuple(kdf_hashes),  # type: ignore[arg-type]
        selected_amplitude_sha256=amplitude_hash,
        plan_sha256=plan_hash,
        record_sha256=record_sha256,
    )


def bind_stage1_regeneration_evidence(
    decision: BudgetDecision,
    stage1_record: Mapping[str, object],
    *,
    record_sha256: str,
) -> Stage1RegenerationEvidence:
    """Bind a finalized Stage1 budget choice to the actual serialized phi job."""

    evidence = _stage1_evidence_from_record(
        stage1_record, record_sha256=record_sha256
    )
    validate_stage1_regeneration_evidence(decision, evidence)
    return evidence


def validate_stage1_regeneration_evidence(
    decision: BudgetDecision,
    evidence: Stage1RegenerationEvidence,
) -> None:
    """Require evidence—not a Boolean—for the adopted Stage1/phi boundary."""

    if not isinstance(decision, BudgetDecision):
        raise SelectionError("decision must be a BudgetDecision")
    if decision.role != "stage1" or decision.method != "stage1":
        raise SelectionError("Stage1 evidence requires a Stage1 budget decision")
    if (
        not decision.finalized
        or decision.optimizer_inconclusive
        or decision.adopted_budget is None
    ):
        raise SelectionError("Stage1 budget must be finalized before phi freeze")
    if not isinstance(evidence, Stage1RegenerationEvidence):
        raise SelectionError("bound Stage1RegenerationEvidence is mandatory")
    if evidence.N != decision.N:
        raise SelectionError("Stage1 evidence N differs from the budget decision")
    if evidence.adopted_budget != decision.adopted_budget:
        raise SelectionError(
            "Stage1 record optimizer budget differs from the adopted budget"
        )
    if evidence.plan_sha256 != PLAN_SHA256.lower():
        raise SelectionError("Stage1 evidence has the wrong plan hash")


def validate_stage1_record_against_evidence(
    stage1_record: Mapping[str, object],
    *,
    record_sha256: str,
    evidence: Stage1RegenerationEvidence,
) -> None:
    """Recompute every Stage1 evidence field from a loaded job record."""

    if not isinstance(evidence, Stage1RegenerationEvidence):
        raise SelectionError("bound Stage1RegenerationEvidence is mandatory")
    observed = _stage1_evidence_from_record(
        stage1_record, record_sha256=record_sha256
    )
    if observed != evidence:
        if observed.adopted_budget != evidence.adopted_budget:
            raise SelectionError(
                "Stage1 record optimizer budget differs from adopted evidence"
            )
        raise SelectionError("Stage1 job record differs from regeneration evidence")


def decide_method_size_budget(
    records_by_budget: Mapping[int, Sequence[BudgetAuditRecord]],
    *,
    role: AuditRole,
) -> BudgetDecision:
    """Choose 400/800/1600 or conservatively mark the stratum unresolved."""

    records, method, size, problem_seeds = _validated_budget_records(
        records_by_budget, role=role
    )
    improvement_400_800 = _mean_improvement(records[400], records[800])
    progress_800 = tuple(
        (cell_id, terminal_progress(records[800][cell_id]))
        for cell_id in sorted(records[800])
    )
    warnings_800 = tuple(
        (cell_id, terminal_checkpoint_warning(records[800][cell_id]))
        for cell_id in sorted(records[800])
    )
    any_progress_800 = any(flag for _, flag in progress_800)
    trigger_1600 = (
        improvement_400_800 >= BUDGET_IMPROVEMENT_THRESHOLD
        or any_progress_800
    )

    if not trigger_1600:
        if 1600 in records:
            raise SelectionError("1600-step evidence supplied without its trigger")
        full_reselection, causal_replay = _depth_flags(role, 400)
        return BudgetDecision(
            role=role,
            method=method,
            N=size,
            problem_seeds=problem_seeds,
            adopted_budget=400,
            mean_improvement_400_to_800=improvement_400_800,
            mean_improvement_800_to_1600=None,
            terminal_progress_800=progress_800,
            terminal_progress_1600=(),
            terminal_checkpoint_warnings_800=warnings_800,
            terminal_checkpoint_warnings_1600=(),
            triggered_1600=False,
            finalized=True,
            optimizer_inconclusive=False,
            stage1_regeneration_required=False,
            full_depth_reselection_required=full_reselection,
            causal_grid_replay_required=causal_replay,
            reason="adopt_400",
        )

    if 1600 not in records:
        return BudgetDecision(
            role=role,
            method=method,
            N=size,
            problem_seeds=problem_seeds,
            adopted_budget=None,
            mean_improvement_400_to_800=improvement_400_800,
            mean_improvement_800_to_1600=None,
            terminal_progress_800=progress_800,
            terminal_progress_1600=(),
            terminal_checkpoint_warnings_800=warnings_800,
            terminal_checkpoint_warnings_1600=(),
            triggered_1600=True,
            finalized=False,
            optimizer_inconclusive=False,
            stage1_regeneration_required=False,
            full_depth_reselection_required=False,
            causal_grid_replay_required=False,
            reason="run_exact_1600",
        )

    improvement_800_1600 = _mean_improvement(records[800], records[1600])
    progress_1600 = tuple(
        (cell_id, terminal_progress(records[1600][cell_id]))
        for cell_id in sorted(records[1600])
    )
    warnings_1600 = tuple(
        (cell_id, terminal_checkpoint_warning(records[1600][cell_id]))
        for cell_id in sorted(records[1600])
    )
    any_progress_1600 = any(flag for _, flag in progress_1600)

    # The final budget is deliberately not declared adequate while any 1600
    # cell has substantive tail progress.  Checkpoint-position warnings are
    # carried separately and have no role in this claim gate.
    if any_progress_1600:
        return BudgetDecision(
            role=role,
            method=method,
            N=size,
            problem_seeds=problem_seeds,
            adopted_budget=None,
            mean_improvement_400_to_800=improvement_400_800,
            mean_improvement_800_to_1600=improvement_800_1600,
            terminal_progress_800=progress_800,
            terminal_progress_1600=progress_1600,
            terminal_checkpoint_warnings_800=warnings_800,
            terminal_checkpoint_warnings_1600=warnings_1600,
            triggered_1600=True,
            finalized=True,
            optimizer_inconclusive=True,
            stage1_regeneration_required=False,
            full_depth_reselection_required=False,
            causal_grid_replay_required=False,
            reason="terminal_progress_at_1600",
        )

    adopted = (
        800
        if (
            improvement_800_1600 < BUDGET_IMPROVEMENT_THRESHOLD
            and not any_progress_800
        )
        else 1600
    )
    full_reselection, causal_replay = _depth_flags(role, adopted)
    return BudgetDecision(
        role=role,
        method=method,
        N=size,
        problem_seeds=problem_seeds,
        adopted_budget=adopted,
        mean_improvement_400_to_800=improvement_400_800,
        mean_improvement_800_to_1600=improvement_800_1600,
        terminal_progress_800=progress_800,
        terminal_progress_1600=progress_1600,
        terminal_checkpoint_warnings_800=warnings_800,
        terminal_checkpoint_warnings_1600=warnings_1600,
        triggered_1600=True,
        finalized=True,
        optimizer_inconclusive=False,
        stage1_regeneration_required=_stage1_regeneration_flag(role, adopted),
        full_depth_reselection_required=full_reselection,
        causal_grid_replay_required=causal_replay,
        reason=f"adopt_{adopted}",
    )


def validate_full_depth_reselection_evidence(
    decision: BudgetDecision,
    depth_decision: DepthDecision | None,
) -> None:
    """Bind an external budget change to a complete adopted-budget scan."""

    if not isinstance(decision, BudgetDecision):
        raise SelectionError("decision must be a BudgetDecision")
    if not decision.finalized or decision.adopted_budget is None:
        raise SelectionError("an adopted final budget is required before depth freeze")
    if not decision.full_depth_reselection_required:
        if depth_decision is not None:
            raise SelectionError("full depth reselection evidence was not required")
        return
    if not isinstance(depth_decision, DepthDecision):
        raise SelectionError("bound DepthDecision evidence is mandatory")
    if (
        depth_decision.method != decision.method
        or depth_decision.N != decision.N
    ):
        raise SelectionError("depth evidence method/N does not match budget decision")
    if depth_decision.budget != decision.adopted_budget:
        raise SelectionError(
            "depth evidence budget does not equal the adopted method-size budget"
        )
    if depth_decision.budget == 400:
        raise SelectionError("Adam-400 depth evidence cannot satisfy a budget-change replay")
    if (
        not depth_decision.finalized
        or depth_decision.unresolved
        or depth_decision.selected_depth is None
    ):
        raise SelectionError("depth reselection must be finalized and resolved")
    if depth_decision.problem_seeds != TUNING_SEEDS[decision.N]:
        raise SelectionError("depth decision does not carry the frozen tuning seeds")
    if len(depth_decision.source_cell_ids) != 2 * len(depth_decision.products):
        raise SelectionError("depth decision lacks one source cell per depth/seed")
    if len(depth_decision.source_hashes) != len(depth_decision.source_cell_ids):
        raise SelectionError("depth decision source hashes are incomplete")


@dataclass(frozen=True)
class CausalGridEvidence:
    """Complete adopted-budget causal depth-by-seed replay evidence."""

    method: str
    N: int
    budget: int
    depths: tuple[int, ...]
    problem_seeds: tuple[int, ...]
    source_cell_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]


def validate_causal_grid_replay_evidence(
    decision: BudgetDecision,
    records: Sequence[DepthCostRecord] | None,
) -> CausalGridEvidence | None:
    """Require every frozen causal depth/seed point after a budget change."""

    if not isinstance(decision, BudgetDecision):
        raise SelectionError("decision must be a BudgetDecision")
    if not decision.finalized or decision.adopted_budget is None:
        raise SelectionError("an adopted final budget is required before causal replay")
    if not decision.causal_grid_replay_required:
        if records is not None:
            raise SelectionError("causal-grid replay evidence was not required")
        return None
    if records is None:
        raise SelectionError("complete causal-grid replay evidence is mandatory")
    _, evidence = _validate_depth_grid(
        records,
        expected_depths=CAUSAL_DEPTHS[decision.N],
        expected_seeds=CAUSAL_SEEDS[decision.N],
        expected_metadata=(
            decision.method,
            decision.N,
            decision.adopted_budget,
        ),
        field="causal-grid replay",
    )
    if decision.adopted_budget == 400:
        raise SelectionError("Adam-400 cells cannot satisfy a budget-change causal replay")
    cell_ids, hashes = _decision_evidence(evidence)
    return CausalGridEvidence(
        method=decision.method,
        N=decision.N,
        budget=decision.adopted_budget,
        depths=CAUSAL_DEPTHS[decision.N],
        problem_seeds=CAUSAL_SEEDS[decision.N],
        source_cell_ids=cell_ids,
        source_hashes=hashes,
    )


__all__ = [
    "ALLOWED_BUDGETS",
    "BASE_DEPTH_GRID",
    "BUDGET_IMPROVEMENT_THRESHOLD",
    "BudgetAuditRecord",
    "BudgetDecision",
    "CAUSAL_DEPTHS",
    "CAUSAL_SEEDS",
    "CausalGridEvidence",
    "DepthCostRecord",
    "DepthDecision",
    "FIRST_BOUNDARY_EXTENSION",
    "NORMALIZATION_FLOOR",
    "SECOND_BOUNDARY_EXTENSION",
    "STAGE1_REGENERATION_EVIDENCE_SCHEMA",
    "STAGE1_SEEDS",
    "Stage1RegenerationEvidence",
    "SelectionError",
    "TERMINAL_DROP_THRESHOLD",
    "TUNING_SEEDS",
    "bind_stage1_regeneration_evidence",
    "decide_method_size_budget",
    "normalized_selected_energy_improvement",
    "select_depth_with_boundary",
    "terminal_checkpoint_warning",
    "terminal_progress",
    "validate_causal_grid_replay_evidence",
    "validate_full_depth_reselection_evidence",
    "validate_stage1_record_against_evidence",
    "validate_stage1_regeneration_evidence",
]
