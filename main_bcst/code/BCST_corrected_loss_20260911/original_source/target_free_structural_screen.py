"""Frozen, outcome-blind structural screen for the 30-bit C5 instance.

The executable contains only the two hard-constraint Hamiltonians.  It never
loads the frozen soft-cost table and never constructs an optimum or ranked
target.  The native simulation space is the exact product of five two-hot
six-resource blocks (15**5 basis states).

The valid profile is intentionally expensive and is not run by this module's
tests.  The smoke profile is explicitly excluded from scientific decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


HERE = Path(__file__).resolve().parent
BLOCKS = 5
LABELS = 6
LOCAL_PAIRS = tuple(itertools.combinations(range(LABELS), 2))
LOCAL_MASKS = np.asarray(
    [sum(1 << label for label in pair) for pair in LOCAL_PAIRS], dtype=np.uint8
)
LOCAL_DIM = len(LOCAL_PAIRS)
DIMENSION = LOCAL_DIM**BLOCKS
QUOTA = (2, 2, 2, 2, 1, 1)
CYCLE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))
STRIDES = np.asarray([LOCAL_DIM ** (BLOCKS - 1 - b) for b in range(BLOCKS)], dtype=np.int64)

# Manuscript-native structural RU constants.
PREP_RU = 5
XY_RU = 75
Q_RU = 120
C_NATIVE_RU = 30
C_PAULI_RU = 30
QC_UNION_RU = 120
SELECTIVE_RU = 900
STRUCTURAL_TERMINAL_RU = 120
# The first number is the audited optimistic generator proxy.  The second is
# one literal first-order Pauli-product Trotter sweep.
CAPACITY_SWITCH_PROXY_RU = 1_050
CAPACITY_SWITCH_PAULI_SWEEP_RU = 8_400
CONFLICT_GUARDED_PROXY_RU = 1_650
CONFLICT_GUARDED_PAULI_SWEEP_RU = 33_600

EXPECTED_SUPPORTS = {
    "native": 759_375,
    "quota": 4_530,
    "conflict": 6_570,
    "final": 390,
}

CAMPAIGN_SCHEMA = "nonredundant-bcst-target-free-campaign-v1"
MANIFEST_FILENAME = "frozen_manifest.json"


@dataclass(frozen=True)
class ScreenConfig:
    profile: str = "valid"
    depths: tuple[int, ...] = (4, 8, 12)
    learned_projector_depths: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12)
    ru_matched_direct_depths: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256)
    restarts: int = 16
    evaluation_budget: int = 4_800
    learning_rate: float = 0.035
    initialization_seeds: tuple[int, ...] = tuple(range(2_026_081_100, 2_026_081_116))
    device: str = "cuda"
    dtype: str = "complex64"
    trotter_sweeps: int = 1
    local_sweep_sensitivities: tuple[int, ...] = (1, 2, 4)
    stage_a_grover_depths: tuple[int, ...] = (
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128
    )
    restart_batch: int = 4
    activation_checkpointing: bool = True
    trace_every: int = 200
    output: Path = HERE / "results" / "target_free_structural_valid.json"

    @classmethod
    def smoke(cls, output: Path | None = None) -> "ScreenConfig":
        return cls(
            profile="smoke",
            depths=(1,),
            learned_projector_depths=(1,),
            ru_matched_direct_depths=(),
            restarts=1,
            # Two stages receive two evaluations each: one update plus the
            # mandatory post-update endpoint.  One-stage controls receive four
            # evaluations, keeping aggregate forward calls exactly equal.
            evaluation_budget=4,
            initialization_seeds=(2_026_081_100,),
            device="cpu",
            dtype="complex64",
            local_sweep_sensitivities=(1,),
            stage_a_grover_depths=(1, 2, 3, 4),
            restart_batch=1,
            activation_checkpointing=False,
            trace_every=1,
            output=(
                HERE / "results" / "EXCLUDED_smoke_r1_s1.json"
                if output is None
                else output
            ),
        )

    def validate(self) -> None:
        if self.profile not in {"valid", "smoke"}:
            raise ValueError(self.profile)
        if not self.depths or any(depth < 1 for depth in self.depths):
            raise ValueError("depths must be positive")
        if tuple(sorted(set(self.depths))) != self.depths:
            raise ValueError("first-stage depths must be sorted and unique")
        if not self.learned_projector_depths or any(
            depth < 1 for depth in self.learned_projector_depths
        ):
            raise ValueError("learned-projector depths must be positive")
        if tuple(sorted(set(self.learned_projector_depths))) != self.learned_projector_depths:
            raise ValueError("learned-projector depths must be sorted and unique")
        if any(depth < 1 for depth in self.ru_matched_direct_depths):
            raise ValueError("direct depths must be positive")
        if self.restarts < 1 or len(self.initialization_seeds) != self.restarts:
            raise ValueError("one explicit initialization seed is required per restart")
        if self.evaluation_budget < 4 or self.evaluation_budget % 2:
            raise ValueError("evaluation budget must be even and at least four")
        if self.learning_rate <= 0 or self.trotter_sweeps < 1:
            raise ValueError("invalid optimizer or Trotter setting")
        if tuple(sorted(set(self.local_sweep_sensitivities))) != self.local_sweep_sensitivities:
            raise ValueError("local sweep sensitivities must be sorted and unique")
        if any(sweep not in {1, 2, 4} for sweep in self.local_sweep_sensitivities):
            raise ValueError("only the frozen 1/2/4-sweep sensitivities are allowed")
        if not self.stage_a_grover_depths or any(p < 1 for p in self.stage_a_grover_depths):
            raise ValueError("invalid Stage-A Grover ladder")
        if self.restart_batch < 1 or self.restart_batch > self.restarts:
            raise ValueError("restart batch must lie in [1,restarts]")
        if self.dtype not in {"complex64", "complex128"}:
            raise ValueError(self.dtype)


def valid_config() -> ScreenConfig:
    return ScreenConfig()


def budget_plan(config: ScreenConfig, stages: int) -> tuple[int, ...]:
    """Return exact forward-evaluation allocations, including endpoints."""
    if stages == 1:
        return (config.evaluation_budget,)
    if stages == 2:
        half = config.evaluation_budget // 2
        return (half, half)
    raise ValueError("the structural screen has only one- and two-stage methods")


def rts99(probability: float) -> int | None:
    if probability <= 0:
        return None
    if probability >= 1:
        return 1
    return int(math.ceil(math.log(0.01) / math.log1p(-probability)))


def geometric_mean(values: Sequence[float]) -> float:
    positive = np.asarray(values, dtype=np.float64)
    if positive.size == 0 or np.any(~np.isfinite(positive)) or np.any(positive <= 0):
        return math.nan
    return float(np.exp(np.mean(np.log(positive))))


def grover_success(initial_probability: float | np.ndarray, depth: int) -> np.ndarray:
    """Exact success for a feasibility-indicator phase and rank-one reflection."""
    probability = np.clip(np.asarray(initial_probability, dtype=np.float64), 0.0, 1.0)
    angle = np.arcsin(np.sqrt(probability))
    return np.sin((2 * depth + 1) * angle) ** 2


def comparator_total_ru(family: str, depth: int, *, pauli_c: bool = False) -> int:
    c_ru = C_PAULI_RU if pauli_c else C_NATIVE_RU
    if family == "direct_joint":
        return PREP_RU + depth * (QC_UNION_RU + XY_RU) + STRUCTURAL_TERMINAL_RU
    if family == "direct_separate":
        return PREP_RU + depth * (Q_RU + c_ru + XY_RU) + STRUCTURAL_TERMINAL_RU
    if family == "fixed_native_joint":
        return (
            PREP_RU
            + depth * (QC_UNION_RU + 2 * PREP_RU + SELECTIVE_RU)
            + STRUCTURAL_TERMINAL_RU
        )
    if family == "fixed_native_separate":
        return (
            PREP_RU
            + depth * (Q_RU + c_ru + 2 * PREP_RU + SELECTIVE_RU)
            + STRUCTURAL_TERMINAL_RU
        )
    raise ValueError(family)


def local_control_ru_ledger(
    first: str, depth_first: int, depth_second: int, sweeps: int
) -> dict[str, int]:
    """Consistent native-C primary and separately named local sensitivities."""
    if first not in {"Q", "C"} or min(depth_first, depth_second, sweeps) < 1:
        raise ValueError((first, depth_first, depth_second, sweeps))
    first_native = StructuralScreen.first_ru(first, depth_first, C_NATIVE_RU)
    first_pauli = StructuralScreen.first_ru(first, depth_first, C_PAULI_RU)
    if first == "Q":
        phase_native, phase_pauli = C_NATIVE_RU, C_PAULI_RU
        sweep_ru = CAPACITY_SWITCH_PAULI_SWEEP_RU
        proxy_ru = CAPACITY_SWITCH_PROXY_RU
    else:
        phase_native = phase_pauli = Q_RU
        sweep_ru = CONFLICT_GUARDED_PAULI_SWEEP_RU
        proxy_ru = CONFLICT_GUARDED_PROXY_RU
    return {
        "primary": first_native
        + depth_second * (phase_native + sweeps * sweep_ru)
        + STRUCTURAL_TERMINAL_RU,
        "native_C": first_native
        + depth_second * (phase_native + sweeps * sweep_ru)
        + STRUCTURAL_TERMINAL_RU,
        "pauli_C_sensitivity": first_pauli
        + depth_second * (phase_pauli + sweeps * sweep_ru)
        + STRUCTURAL_TERMINAL_RU,
        "generator_proxy_native_C": first_native
        + depth_second * (phase_native + proxy_ru)
        + STRUCTURAL_TERMINAL_RU,
    }


def required_comparator_depths(
    selected_lp_ru: int, candidates: Sequence[int]
) -> dict[str, tuple[int, ...]]:
    """Comparator-specific deterministic native-RU caps."""
    families = (
        "direct_joint",
        "direct_separate",
        "fixed_native_joint",
        "fixed_native_separate",
    )
    return {
        family: tuple(
            depth
            for depth in sorted(set(int(value) for value in candidates))
            if comparator_total_ru(family, depth) <= selected_lp_ru
        )
        for family in families
    }


def select_full_lp_by_mass_ru(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Frozen 2%-mass window followed by the lower-primary-RU tie rule."""
    if not rows:
        raise ValueError("no full-LP rows")
    medians = np.asarray(
        [float(np.median(np.asarray(row["final_mass"], dtype=float))) for row in rows],
        dtype=float,
    )
    if np.any(~np.isfinite(medians)):
        raise ValueError("nonfinite LP mass")
    maximum = float(np.max(medians))
    eligible = [row for row, median in zip(rows, medians) if median >= 0.98 * maximum]
    if any(row.get("RU", {}).get("primary") is None for row in eligible):
        raise ValueError("missing primary RU")
    return min(
        eligible,
        key=lambda row: (
            int(row["RU"]["primary"]),
            -float(np.median(np.asarray(row["final_mass"], dtype=float))),
            str(row["algorithm"]),
        ),
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def config_record(config: ScreenConfig) -> dict[str, object]:
    record = asdict(config)
    record["output"] = str(config.output)
    return record


def scientific_config_record(config: ScreenConfig) -> dict[str, object]:
    """Parameters affecting scientific results; excludes location and batching."""
    record = config_record(config)
    record.pop("output")
    record.pop("restart_batch")
    return record


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    """Durably replace one JSON artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def restart_seed_batches(
    config: ScreenConfig, execution_target: str
) -> tuple[tuple[int, ...], ...]:
    if execution_target not in {"remote80", "local8"}:
        raise ValueError(execution_target)
    batch_size = 4 if execution_target == "remote80" else 1
    if config.profile == "smoke":
        batch_size = 1
    seeds = config.initialization_seeds
    return tuple(
        tuple(seeds[start : start + batch_size])
        for start in range(0, len(seeds), batch_size)
    )


def frozen_campaign_manifest(
    config: ScreenConfig, campaign_root: Path, execution_target: str
) -> dict[str, object]:
    """Build the outcome-blind task universe before any valid optimization."""
    batches = restart_seed_batches(config, execution_target)
    tasks: list[dict[str, object]] = []

    def add_task(
        stage: str,
        family: str,
        batch_index: int,
        *,
        first: str | None = None,
        p1: int = 0,
        p2: int = 0,
        sweep: int | None = None,
        conditional: str,
    ) -> None:
        fields = [stage, family]
        if first is not None:
            fields.append(first)
        if p1:
            fields.append(f"p1-{p1}")
        if p2:
            fields.append(f"p2-{p2}")
        if sweep is not None:
            fields.append(f"sweep-{sweep}")
        fields.append(f"batch-{batch_index:02d}")
        task_id = "__".join(fields)
        tasks.append(
            {
                "task_id": task_id,
                "stage": stage,
                "family": family,
                "first": first,
                "second": None if first is None else ("C" if first == "Q" else "Q"),
                "depth_first": p1,
                "depth_second": p2,
                "trotter_sweeps": sweep,
                "batch_index": batch_index,
                "seeds": list(batches[batch_index]),
                "evaluation_budget_per_restart": config.evaluation_budget,
                "conditional": conditional,
                "output": str(campaign_root / "shards" / f"{task_id}.json"),
            }
        )

    for batch_index in range(len(batches)):
        add_task(
            "A",
            "coarse_bundle",
            batch_index,
            conditional=(
                "always; all frozen Q/C coarse depths for one ordered restart shard"
            ),
        )
        for first in ("Q", "C"):
            for p1 in config.depths:
                for p2 in config.learned_projector_depths:
                    add_task(
                        "B",
                        "full_lp",
                        batch_index,
                        first=first,
                        p1=p1,
                        p2=p2,
                        conditional="only if completed Stage A passes",
                    )
                    add_task(
                        "C",
                        "warm",
                        batch_index,
                        first=first,
                        p1=p1,
                        p2=p2,
                        conditional="only after Stage B selects a structurally eligible LP row",
                    )
                    for sweep in config.local_sweep_sensitivities:
                        add_task(
                            "C",
                            "local_digitized",
                            batch_index,
                            first=first,
                            p1=p1,
                            p2=p2,
                            sweep=sweep,
                            conditional=(
                                "only after Stage B; finite control, primary native-C ledger "
                                "and literal sweep cost"
                            ),
                        )
        candidate_depths = tuple(
            sorted(
                set(config.ru_matched_direct_depths)
                | {
                    left + right
                    for left in config.depths
                    for right in config.learned_projector_depths
                }
            )
        )
        for depth in candidate_depths:
            add_task(
                "D",
                "direct_pair",
                batch_index,
                p2=depth,
                conditional=(
                    "only if Stage C passes; execute a comparator family only through "
                    "its deterministic selected-LP primary-RU cap"
                ),
            )
            add_task(
                "C",
                "fixed_native_pair",
                batch_index,
                p2=depth,
                conditional=(
                    "only after Stage B; execute each fixed-native family only through "
                    "its deterministic selected-LP primary-RU cap"
                ),
            )

    body: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "profile": config.profile,
        "execution_target": execution_target,
        "restart_batch": 4 if execution_target == "remote80" else 1,
        "activation_checkpointing": config.activation_checkpointing,
        "scientific_config": scientific_config_record(config),
        "scientific_config_sha256": sha256_json(scientific_config_record(config)),
        "code_sha256": code_sha256(),
        "campaign_root": str(campaign_root),
        "seed_batches": [list(batch) for batch in batches],
        "aggregation_contract": {
            "formal_stage_a_decision_requires_all_restarts": config.restarts,
            "early_promotion_allowed": False,
            "early_stop_allowed": "failure only, and only when no remaining seeds can restore the final thresholds",
            "all_shards_retained": True,
            "checksum_required": True,
        },
        "stage_a_selection": (
            "eligible median final mass >=0.25 and median stage gain >=2; then minimum "
            "geometric-mean optimistic RTS99 x primary RU, lower RU/depth ties"
        ),
        "full_lp_selection": (
            "within 2% of maximum median final mass, then lowest primary RU"
        ),
        "analog_local_protocol": {
            "conditional_after_stage_a": True,
            "generator": "exact exponential of the sum of the audited local adjacency templates",
            "optimizer_budget": config.evaluation_budget,
            "initialization_seeds": list(config.initialization_seeds),
            "loss": "Q/48+C/10",
            "finite_RU_rank": False,
            "role": "optimistic locality sensitivity; never part of the finite-RU winning envelope",
        },
        "precision_replay": {
            "conditional": "only after all complex64 finite controls pass",
            "dtype": "complex128",
            "saved_angles_only": True,
            "reoptimization": False,
            "configuration_reselection": False,
        },
        "tasks": tasks,
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def validate_manifest(
    manifest: dict[str, object], config: ScreenConfig, execution_target: str
) -> None:
    claimed = str(manifest.get("manifest_sha256", ""))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if claimed != sha256_json(body):
        raise RuntimeError("manifest checksum mismatch")
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise RuntimeError("manifest schema mismatch")
    if manifest.get("execution_target") != execution_target:
        raise RuntimeError("execution-target mismatch")
    if manifest.get("code_sha256") != code_sha256():
        raise RuntimeError("code changed after manifest freeze")
    expected_config_hash = sha256_json(scientific_config_record(config))
    if manifest.get("scientific_config_sha256") != expected_config_hash:
        raise RuntimeError("scientific configuration changed after manifest freeze")


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def enumerate_shell() -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(DIMENSION, dtype=np.int64)
    digits = np.empty((DIMENSION, BLOCKS), dtype=np.uint8)
    remainder = indices.copy()
    for block in range(BLOCKS - 1, -1, -1):
        digits[:, block] = remainder % LOCAL_DIM
        remainder //= LOCAL_DIM
    return digits, LOCAL_MASKS[digits]


def structural_arrays() -> dict[str, np.ndarray]:
    digits, masks = enumerate_shell()
    counts = np.zeros((DIMENSION, LABELS), dtype=np.uint8)
    for label in range(LABELS):
        counts[:, label] = np.sum((masks >> label) & 1, axis=1)
    q_raw = np.sum(
        (counts.astype(np.int16) - np.asarray(QUOTA, dtype=np.int16)[None, :]) ** 2,
        axis=1,
    ).astype(np.float32)
    c_raw = np.zeros(DIMENSION, dtype=np.uint8)
    for left, right in CYCLE_EDGES:
        common = masks[:, left] & masks[:, right]
        for label in range(LABELS):
            c_raw += ((common >> label) & 1).astype(np.uint8)
    q_mask = q_raw == 0
    c_mask = c_raw == 0
    final_mask = q_mask & c_mask
    observed = {
        "native": DIMENSION,
        "quota": int(np.sum(q_mask)),
        "conflict": int(np.sum(c_mask)),
        "final": int(np.sum(final_mask)),
    }
    if observed != EXPECTED_SUPPORTS:
        raise AssertionError(f"structural support mismatch: {observed}")
    if float(np.max(q_raw)) != 48.0 or int(np.max(c_raw)) != 10:
        raise AssertionError("frozen normalization mismatch")
    return {
        "digits": digits,
        "masks": masks,
        "q_norm": q_raw / 48.0,
        "c_norm": c_raw.astype(np.float32) / 10.0,
        "q_mask": q_mask,
        "c_mask": c_mask,
        "final_mask": final_mask,
    }


def local_xy_eigensystem(
    device: torch.device, real_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    adjacency = torch.zeros((LOCAL_DIM, LOCAL_DIM), dtype=real_dtype, device=device)
    for i, left in enumerate(LOCAL_PAIRS):
        for j, right in enumerate(LOCAL_PAIRS):
            if len(set(left) ^ set(right)) == 2:
                adjacency[i, j] = 1
    return torch.linalg.eigh(adjacency)


def apply_block_xy(
    state: torch.Tensor,
    beta: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    blocks: int = BLOCKS,
) -> torch.Tensor:
    if state.ndim != 2 or state.shape[1] != LOCAL_DIM**blocks:
        raise ValueError("state does not match the product two-hot shell")
    batch = state.shape[0]
    phases = torch.exp(-1j * beta[:, None] * eigenvalues[None, :]).to(state.dtype)
    vectors = eigenvectors.to(state.dtype)
    unitary = (vectors[None, :, :] * phases[:, None, :]) @ vectors.T[None, :, :]
    shaped = state.reshape((batch,) + (LOCAL_DIM,) * blocks)
    for block in range(blocks):
        moved = shaped.movedim(block + 1, -1)
        original_shape = moved.shape
        flat = moved.reshape(batch, -1, LOCAL_DIM)
        flat = torch.bmm(flat, unitary.transpose(1, 2))
        shaped = flat.reshape(original_shape).movedim(-1, block + 1)
    return shaped.reshape(batch, LOCAL_DIM**blocks)


def apply_history(
    state: torch.Tensor, beta: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if reference.ndim == 1:
        reference = reference[None, :].expand(state.shape[0], -1)
    reference = reference / torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    overlap = torch.sum(reference.conj() * state, dim=1, keepdim=True)
    phase = torch.exp(-1j * beta).to(state.dtype)[:, None]
    return phase * state + (1.0 - phase) * overlap * reference


def expectation(state: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    probability = torch.abs(state) ** 2
    norm = probability.sum(dim=1).clamp_min(1e-12)
    return (probability * diagonal[None, :]).sum(dim=1) / norm


def renormalize(state: torch.Tensor) -> torch.Tensor:
    """Remove accumulated complex64 round-off after an analytically unitary layer."""
    norm = torch.sqrt(torch.sum(torch.abs(state) ** 2, dim=1, keepdim=True).clamp_min(1e-24))
    return state / norm


def _swap_table() -> np.ndarray:
    answer = np.full((LOCAL_DIM, LABELS, LABELS), -1, dtype=np.int16)
    lookup = {int(mask): i for i, mask in enumerate(LOCAL_MASKS)}
    for local_index, mask_value in enumerate(LOCAL_MASKS):
        mask = int(mask_value)
        for a, b in LOCAL_PAIRS:
            if ((mask >> a) & 1) == ((mask >> b) & 1):
                continue
            answer[local_index, a, b] = lookup[mask ^ (1 << a) ^ (1 << b)]
            answer[local_index, b, a] = answer[local_index, a, b]
    return answer


SWAP_TABLE = _swap_table()


class TemplateMixer:
    """Fixed-order first-order product of physical two-level generators."""

    def __init__(
        self,
        kind: str,
        digits: np.ndarray,
        masks: np.ndarray,
        device: torch.device,
        sweeps: int,
    ) -> None:
        if kind not in {"capacity_switch", "conflict_guarded"}:
            raise ValueError(kind)
        self.kind = kind
        self.sweeps = sweeps
        self.templates: list[tuple[torch.Tensor, torch.Tensor]] = []
        indices = np.arange(DIMENSION, dtype=np.int64)
        if kind == "capacity_switch":
            iterator: Iterable[tuple[int, int, int, int]] = (
                (left, right, a, b)
                for left, right in CYCLE_EDGES
                for a, b in LOCAL_PAIRS
            )
            for left, right, a, b in iterator:
                dl = digits[:, left]
                dr = digits[:, right]
                nl = SWAP_TABLE[dl, a, b]
                nr = SWAP_TABLE[dr, a, b]
                left_has_a = ((masks[:, left] >> a) & 1).astype(bool)
                right_has_a = ((masks[:, right] >> a) & 1).astype(bool)
                applicable = (nl >= 0) & (nr >= 0) & (left_has_a != right_has_a)
                changed = indices + (nl.astype(np.int64) - dl) * STRIDES[left]
                changed += (nr.astype(np.int64) - dr) * STRIDES[right]
                source = np.flatnonzero(applicable & (indices < changed)).astype(np.int64)
                partner = changed[source].astype(np.int64)
                self.templates.append(
                    (
                        torch.from_numpy(source).to(device),
                        torch.from_numpy(partner).to(device),
                    )
                )
        else:
            for block in range(BLOCKS):
                neighbors = ((block - 1) % BLOCKS, (block + 1) % BLOCKS)
                for a, b in LOCAL_PAIRS:
                    current = digits[:, block]
                    changed_local = SWAP_TABLE[current, a, b]
                    guard = (1 << a) | (1 << b)
                    applicable = changed_local >= 0
                    applicable &= (masks[:, neighbors[0]] & guard) == 0
                    applicable &= (masks[:, neighbors[1]] & guard) == 0
                    changed = indices + (
                        changed_local.astype(np.int64) - current
                    ) * STRIDES[block]
                    source = np.flatnonzero(applicable & (indices < changed)).astype(np.int64)
                    partner = changed[source].astype(np.int64)
                    self.templates.append(
                        (
                            torch.from_numpy(source).to(device),
                            torch.from_numpy(partner).to(device),
                        )
                    )
        if len(self.templates) != 75 or any(left.numel() == 0 for left, _ in self.templates):
            raise AssertionError(f"unexpected {kind} template construction")

    def apply(self, state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        scaled = beta / self.sweeps
        cosine = torch.cos(scaled).to(state.real.dtype)[:, None]
        sine = torch.sin(scaled).to(state.real.dtype)[:, None]
        for _ in range(self.sweeps):
            for left, right in self.templates:
                amp_left = torch.index_select(state, 1, left)
                amp_right = torch.index_select(state, 1, right)
                delta_left = (cosine - 1.0) * amp_left - 1j * sine * amp_right
                delta_right = (cosine - 1.0) * amp_right - 1j * sine * amp_left
                joined_indices = torch.cat((left, right))
                joined_delta = torch.cat((delta_left, delta_right), dim=1)
                state = torch.index_add(state, 1, joined_indices, joined_delta)
        return state


@dataclass
class OptimizationResult:
    label: str
    state: torch.Tensor
    energy: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    best_evaluation: torch.Tensor
    evaluations_per_restart: int
    updates_per_restart: int
    endpoint_evaluations_per_restart: int
    trace: list[dict[str, object]]
    elapsed_sec: float


def initial_angles(
    seeds: Sequence[int], count: int, stream: int, real_dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 1_000_003 * int(stream))
        rows.append(rng.uniform(-math.pi, math.pi, size=count))
    return torch.as_tensor(np.asarray(rows), dtype=real_dtype, device=device)


def optimize(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    gamma_count: int,
    beta_count: int,
    evaluations: int,
    label: str,
    config: ScreenConfig,
    real_dtype: torch.dtype,
    device: torch.device,
    stream: int,
) -> OptimizationResult:
    """Optimize with the final post-update endpoint included in `evaluations`."""
    if evaluations < 2:
        raise ValueError("at least one update plus one endpoint is required")
    gamma = initial_angles(
        config.initialization_seeds, gamma_count, 2 * stream, real_dtype, device
    ).requires_grad_(True)
    beta = initial_angles(
        config.initialization_seeds, beta_count, 2 * stream + 1, real_dtype, device
    ).requires_grad_(True)
    optimizer = torch.optim.Adam((gamma, beta), lr=config.learning_rate)
    restarts = config.restarts
    best_energy = torch.full((restarts,), math.inf, dtype=real_dtype, device=device)
    best_state: torch.Tensor | None = None
    best_gamma = torch.zeros_like(gamma)
    best_beta = torch.zeros_like(beta)
    best_eval = torch.zeros(restarts, dtype=torch.int64, device=device)
    trace: list[dict[str, object]] = []
    updates = evaluations - 1
    started = time.time()

    def inspect(energies: torch.Tensor, states: torch.Tensor, evaluation: int, endpoint: bool) -> None:
        nonlocal best_energy, best_state, best_gamma, best_beta, best_eval
        detached_energy = energies.detach()
        improved = detached_energy < best_energy
        if best_state is None:
            best_state = states.detach().clone()
        elif torch.any(improved):
            best_state[improved] = states.detach()[improved]
        best_energy = torch.where(improved, detached_energy, best_energy)
        best_gamma[improved] = gamma.detach()[improved]
        best_beta[improved] = beta.detach()[improved]
        best_eval[improved] = evaluation
        if evaluation == 1 or endpoint or evaluation % config.trace_every == 0:
            trace.append(
                {
                    "evaluation": evaluation,
                    "post_update_endpoint": endpoint,
                    "loss_by_restart": [float(x) for x in detached_energy.cpu()],
                }
            )

    for evaluation in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        energies, states = energy_fn(gamma, beta)
        inspect(energies, states, evaluation, False)
        energies.sum().backward()
        optimizer.step()
    with torch.no_grad():
        energies, states = energy_fn(gamma, beta)
        inspect(energies, states, evaluations, True)
    if best_state is None:
        raise RuntimeError(label)
    return OptimizationResult(
        label=label,
        state=best_state,
        energy=best_energy,
        gamma=best_gamma,
        beta=best_beta,
        best_evaluation=best_eval,
        evaluations_per_restart=evaluations,
        updates_per_restart=updates,
        endpoint_evaluations_per_restart=1,
        trace=trace,
        elapsed_sec=time.time() - started,
    )


class StructuralScreen:
    def __init__(self, config: ScreenConfig) -> None:
        config.validate()
        self.config = config
        self.device = select_device(config.device)
        self.real_dtype = torch.float32 if config.dtype == "complex64" else torch.float64
        self.complex_dtype = torch.complex64 if config.dtype == "complex64" else torch.complex128
        arrays = structural_arrays()
        self.digits = arrays.pop("digits")
        self.masks_np = arrays.pop("masks")
        self.q = torch.as_tensor(arrays["q_norm"], dtype=self.real_dtype, device=self.device)
        self.c = torch.as_tensor(arrays["c_norm"], dtype=self.real_dtype, device=self.device)
        self.common = self.q + self.c
        self.q_mask = torch.as_tensor(arrays["q_mask"], dtype=torch.bool, device=self.device)
        self.c_mask = torch.as_tensor(arrays["c_mask"], dtype=torch.bool, device=self.device)
        self.final_mask = torch.as_tensor(arrays["final_mask"], dtype=torch.bool, device=self.device)
        self.native = torch.full(
            (DIMENSION,),
            complex(1 / math.sqrt(DIMENSION), 0),
            dtype=self.complex_dtype,
            device=self.device,
        )
        self.xy_eigenvalues, self.xy_eigenvectors = local_xy_eigensystem(
            self.device, self.real_dtype
        )
        self._local_mixers: dict[str, TemplateMixer] = {}
        self.rows: list[dict[str, object]] = []
        self.traces: dict[str, object] = {}
        self.objective_table_loaded = False
        self.target_constructed = False

    def local_mixer(self, kind: str) -> TemplateMixer:
        if kind not in self._local_mixers:
            self._local_mixers[kind] = TemplateMixer(
                kind, self.digits, self.masks_np, self.device, self.config.trotter_sweeps
            )
        return self._local_mixers[kind]

    def block_energy(
        self,
        initial: torch.Tensor,
        phases: Sequence[torch.Tensor],
        evaluation: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial.ndim == 1:
            state = initial[None, :].expand(gamma.shape[0], -1).clone()
        else:
            state = initial.clone()
        if gamma.shape[1] != len(phases) * depth:
            raise ValueError("phase-angle shape mismatch")

        def layer_step(
            current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor
        ) -> torch.Tensor:
            for phase_index, phase in enumerate(phases):
                current = current * torch.exp(
                    -1j * layer_gamma[:, phase_index : phase_index + 1] * phase[None, :]
                ).to(self.complex_dtype)
            current = apply_block_xy(
                current, layer_beta, self.xy_eigenvalues, self.xy_eigenvectors
            )
            return renormalize(current)

        for layer in range(depth):
            layer_gamma = torch.stack(
                [gamma[:, phase_index * depth + layer] for phase_index in range(len(phases))],
                dim=1,
            )
            layer_beta = beta[:, layer]
            if self.config.activation_checkpointing and torch.is_grad_enabled():
                state = checkpoint(
                    layer_step,
                    state,
                    layer_gamma,
                    layer_beta,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state = layer_step(state, layer_gamma, layer_beta)
        return expectation(state, evaluation), state

    def history_energy(
        self,
        reference: torch.Tensor,
        phase: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = reference.clone()

        def layer_step(
            current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor
        ) -> torch.Tensor:
            current = current * torch.exp(
                -1j * layer_gamma[:, None] * phase[None, :]
            ).to(self.complex_dtype)
            current = apply_history(current, layer_beta, reference)
            return renormalize(current)

        for layer in range(depth):
            if self.config.activation_checkpointing and torch.is_grad_enabled():
                state = checkpoint(
                    layer_step,
                    state,
                    gamma[:, layer],
                    beta[:, layer],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state = layer_step(state, gamma[:, layer], beta[:, layer])
        return expectation(state, self.common), state

    def fixed_rank_one_energy(
        self,
        phases: Sequence[torch.Tensor],
        gamma: torch.Tensor,
        beta: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized prescribed-native rank-one/Grover control."""
        reference = self.native[None, :].expand(gamma.shape[0], -1)
        state = reference.clone()
        if gamma.shape[1] != len(phases) * depth:
            raise ValueError("fixed-reference phase-angle shape mismatch")

        def layer_step(
            current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor
        ) -> torch.Tensor:
            for phase_index, phase in enumerate(phases):
                current = current * torch.exp(
                    -1j * layer_gamma[:, phase_index : phase_index + 1] * phase[None, :]
                ).to(self.complex_dtype)
            current = apply_history(current, layer_beta, reference)
            return renormalize(current)

        for layer in range(depth):
            layer_gamma = torch.stack(
                [gamma[:, phase_index * depth + layer] for phase_index in range(len(phases))],
                dim=1,
            )
            if self.config.activation_checkpointing and torch.is_grad_enabled():
                state = checkpoint(
                    layer_step,
                    state,
                    layer_gamma,
                    beta[:, layer],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state = layer_step(state, layer_gamma, beta[:, layer])
        return expectation(state, self.common), state

    def template_energy(
        self,
        reference: torch.Tensor,
        phase: torch.Tensor,
        mixer: TemplateMixer,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = reference.clone()

        def layer_step(
            current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor
        ) -> torch.Tensor:
            current = current * torch.exp(
                -1j * layer_gamma[:, None] * phase[None, :]
            ).to(self.complex_dtype)
            current = mixer.apply(current, layer_beta)
            return renormalize(current)

        for layer in range(depth):
            if self.config.activation_checkpointing and torch.is_grad_enabled():
                state = checkpoint(
                    layer_step,
                    state,
                    gamma[:, layer],
                    beta[:, layer],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                state = layer_step(state, gamma[:, layer], beta[:, layer])
        return expectation(state, self.common), state

    def metrics(self, state: torch.Tensor) -> dict[str, list[float]]:
        with torch.no_grad():
            probability = torch.abs(state) ** 2
            norm = probability.sum(dim=1).clamp_min(1e-12)
            result = {
                "state_norm": norm,
                "quota_mass": probability[:, self.q_mask].sum(dim=1) / norm,
                "conflict_mass": probability[:, self.c_mask].sum(dim=1) / norm,
                "final_mass": probability[:, self.final_mask].sum(dim=1) / norm,
                "expected_Qtilde": (probability * self.q[None, :]).sum(dim=1) / norm,
                "expected_Ctilde": (probability * self.c[None, :]).sum(dim=1) / norm,
                "common_loss": (probability * self.common[None, :]).sum(dim=1) / norm,
            }
        return {key: [float(x) for x in value.cpu()] for key, value in result.items()}

    def replay_saved_row(self, row: dict[str, object]) -> dict[str, object]:
        """Evaluate saved angles exactly in complex128; never optimize or reselect."""
        if self.config.dtype != "complex128":
            raise ValueError("saved-angle precision replay requires complex128")
        seeds = tuple(int(value) for value in row["initialization_seeds"])
        if seeds != self.config.initialization_seeds:
            raise ValueError("replay seed/order mismatch")
        spec = row.get("replay_spec")
        if not isinstance(spec, dict):
            raise ValueError("row lacks a replay specification")
        gamma = torch.as_tensor(
            row["gamma_by_restart"], dtype=self.real_dtype, device=self.device
        )
        beta = torch.as_tensor(
            row["beta_by_restart"], dtype=self.real_dtype, device=self.device
        )
        p1 = int(row["depth_first"])
        p2 = int(row["depth_second"])
        family = str(spec["family"])
        replay_stage1_final_mass: list[float] | None = None

        with torch.no_grad():
            if spec["kind"] == "one_stage":
                if family == "coarse":
                    constraint = str(spec["constraint"])
                    phase = self.q if constraint == "Q" else self.c
                    _, state = self.block_energy(
                        self.native, (phase,), phase, gamma, beta, p1
                    )
                elif family == "direct_joint":
                    _, state = self.block_energy(
                        self.native, (self.common,), self.common, gamma, beta, p2
                    )
                elif family == "direct_separate":
                    _, state = self.block_energy(
                        self.native, (self.q, self.c), self.common, gamma, beta, p2
                    )
                elif family == "fixed_native_joint":
                    _, state = self.fixed_rank_one_energy((self.common,), gamma, beta, p2)
                elif family == "fixed_native_separate":
                    _, state = self.fixed_rank_one_energy((self.q, self.c), gamma, beta, p2)
                else:
                    raise ValueError(f"unsupported one-stage replay family {family}")
            elif spec["kind"] == "pipeline":
                first = str(spec["first"])
                second = str(spec["second"])
                first_phase = self.q if first == "Q" else self.c
                second_phase = self.q if second == "Q" else self.c
                stage1_gamma = torch.as_tensor(
                    row["stage1_gamma_by_restart"],
                    dtype=self.real_dtype,
                    device=self.device,
                )
                stage1_beta = torch.as_tensor(
                    row["stage1_beta_by_restart"],
                    dtype=self.real_dtype,
                    device=self.device,
                )
                _, reference = self.block_energy(
                    self.native,
                    (first_phase,),
                    first_phase,
                    stage1_gamma,
                    stage1_beta,
                    p1,
                )
                replay_stage1_final_mass = self.metrics(reference)["final_mass"]
                if family == "full_lp":
                    _, state = self.history_energy(reference, second_phase, gamma, beta, p2)
                elif family == "warm":
                    _, state = self.block_energy(
                        reference, (self.q, self.c), self.common, gamma, beta, p2
                    )
                elif family == "local":
                    expected_sweeps = int(spec["trotter_sweeps"])
                    if expected_sweeps != self.config.trotter_sweeps:
                        raise ValueError("local replay Trotter-sweep mismatch")
                    _, state = self.template_energy(
                        reference,
                        second_phase,
                        self.local_mixer(str(spec["local_mixer"])),
                        gamma,
                        beta,
                        p2,
                    )
                else:
                    raise ValueError(f"unsupported pipeline replay family {family}")
            else:
                raise ValueError(spec["kind"])

        metrics = self.metrics(state)
        primary_ru = row["RU"].get("primary")
        rts = [rts99(value) for value in metrics["final_mass"]]
        costs = [
            None if shots is None or primary_ru is None else int(shots) * int(primary_ru)
            for shots in rts
        ]
        return {
            "schema": "nonredundant-bcst-saved-angle-complex128-replay-v1",
            "algorithm": row["algorithm"],
            "source_initialization_seeds": list(seeds),
            "dtype": "complex128",
            "optimization_performed": False,
            "configuration_reselected": False,
            "angles_reselected": False,
            "target_free": True,
            "replay_spec": spec,
            **metrics,
            "stage1_final_mass": replay_stage1_final_mass,
            "feasibility_RTS99": rts,
            "feasibility_RTS99_x_primary_RU": costs,
            "RU": row["RU"],
        }

    def _public_result(
        self,
        result: OptimizationResult,
        method: str,
        family: str,
        depth_first: int,
        depth_second: int,
        phase: str,
        mixer: str,
        ru: dict[str, int | None],
        aggregate_evaluations: int,
        aggregate_updates: int,
        stage1_final_mass: Sequence[float] | None = None,
        stage1_result: OptimizationResult | None = None,
        replay_spec: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metrics = self.metrics(result.state)
        primary_ru = ru.get("primary")
        final_mass = metrics["final_mass"]
        rts = [rts99(value) for value in final_mass]
        cost = [
            None if shots is None or primary_ru is None else int(shots) * int(primary_ru)
            for shots in rts
        ]
        row = {
            "algorithm": method,
            "family": family,
            "target_free": True,
            "depth_first": depth_first,
            "depth_second": depth_second,
            "phase": phase,
            "mixer": mixer,
            "training_loss": (
                phase if family in {"stage1", "coarse"} else "Q/48 + C/10"
            ),
            "loss_by_restart": [float(x) for x in result.energy.cpu()],
            **metrics,
            "stage1_final_mass": None if stage1_final_mass is None else list(stage1_final_mass),
            "feasibility_RTS99": rts,
            "feasibility_RTS99_x_primary_RU": cost,
            "RU": ru,
            "evaluations_this_call_per_restart": result.evaluations_per_restart,
            "updates_this_call_per_restart": result.updates_per_restart,
            "endpoint_evaluations_this_call_per_restart": result.endpoint_evaluations_per_restart,
            "aggregate_pipeline_evaluations_per_restart": aggregate_evaluations,
            "aggregate_pipeline_updates_per_restart": aggregate_updates,
            "aggregate_pipeline_state_evaluations": aggregate_evaluations * self.config.restarts,
            "best_evaluation_by_restart": [int(x) for x in result.best_evaluation.cpu()],
            "gamma_by_restart": result.gamma.cpu().tolist(),
            "beta_by_restart": result.beta.cpu().tolist(),
            "stage1_gamma_by_restart": (
                None if stage1_result is None else stage1_result.gamma.cpu().tolist()
            ),
            "stage1_beta_by_restart": (
                None if stage1_result is None else stage1_result.beta.cpu().tolist()
            ),
            "initialization_seeds": list(self.config.initialization_seeds),
            "replay_spec": replay_spec,
            "elapsed_sec_this_call": result.elapsed_sec,
        }
        if aggregate_evaluations != self.config.evaluation_budget and family != "stage1":
            raise AssertionError(f"budget mismatch for {method}")
        self.rows.append(row)
        self.traces[method] = result.trace
        return row

    @staticmethod
    def first_ru(kind: str, depth: int, c_ru: int) -> int:
        phase = Q_RU if kind == "Q" else c_ru
        return PREP_RU + depth * (phase + XY_RU)

    @staticmethod
    def history_ru(prior: int, depth: int, phase_ru: int) -> int:
        return (1 + 2 * depth) * prior + depth * (phase_ru + SELECTIVE_RU)

    def _stage1(self, kind: str, depth: int, evaluations: int, stream: int) -> OptimizationResult:
        diagonal = self.q if kind == "Q" else self.c
        label = f"Stage1-{kind}-p{depth}-stream{stream}"
        return optimize(
            lambda gamma, beta: self.block_energy(
                self.native, (diagonal,), diagonal, gamma, beta, depth
            ),
            depth,
            depth,
            evaluations,
            label,
            self.config,
            self.real_dtype,
            self.device,
            stream,
        )

    def _run_coarse(self, kind: str, depth: int, stream: int) -> dict[str, object]:
        """Standalone full-budget coarse control, not a pipeline checkpoint."""
        (evaluations,) = budget_plan(self.config, 1)
        result = self._stage1(kind, depth, evaluations, stream)
        native = self.first_ru(kind, depth, C_NATIVE_RU)
        pauli = self.first_ru(kind, depth, C_PAULI_RU)
        return self._public_result(
            result,
            f"Coarse-only-{kind}-p{depth}",
            "coarse",
            depth,
            0,
            f"{kind} normalized",
            "complete block-XY",
            {
                "primary": native + STRUCTURAL_TERMINAL_RU,
                "native_C": native + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": pauli + STRUCTURAL_TERMINAL_RU,
            },
            evaluations,
            result.updates_per_restart,
            replay_spec={"kind": "one_stage", "family": "coarse", "constraint": kind},
        )

    def _run_fixed_native(self, depth: int, stream: int) -> None:
        """Optimized prescribed-native rank-one controls required by the re-audit."""
        (evaluations,) = budget_plan(self.config, 1)
        joint = optimize(
            lambda gamma, beta: self.fixed_rank_one_energy(
                (self.common,), gamma, beta, depth
            ),
            depth,
            depth,
            evaluations,
            f"Fixed-native-rank1-joint-QC-p{depth}",
            self.config,
            self.real_dtype,
            self.device,
            stream,
        )
        joint_circuit = PREP_RU + depth * (
            QC_UNION_RU + 2 * PREP_RU + SELECTIVE_RU
        )
        self._public_result(
            joint,
            f"Fixed-native-rank1-joint-QC-p{depth}",
            "fixed_native_joint",
            0,
            depth,
            "single Q+C union angle; common cumulative loss",
            "prescribed native-uniform rank-one projector",
            {
                "primary": joint_circuit + STRUCTURAL_TERMINAL_RU,
                "native_C": joint_circuit + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": joint_circuit + STRUCTURAL_TERMINAL_RU,
            },
            evaluations,
            joint.updates_per_restart,
            replay_spec={"kind": "one_stage", "family": "fixed_native_joint"},
        )

        separate = optimize(
            lambda gamma, beta: self.fixed_rank_one_energy(
                (self.q, self.c), gamma, beta, depth
            ),
            2 * depth,
            depth,
            evaluations,
            f"Fixed-native-rank1-separate-Q-C-p{depth}",
            self.config,
            self.real_dtype,
            self.device,
            stream + 1,
        )
        separate_native = PREP_RU + depth * (
            Q_RU + C_NATIVE_RU + 2 * PREP_RU + SELECTIVE_RU
        )
        separate_pauli = PREP_RU + depth * (
            Q_RU + C_PAULI_RU + 2 * PREP_RU + SELECTIVE_RU
        )
        self._public_result(
            separate,
            f"Fixed-native-rank1-separate-Q-C-p{depth}",
            "fixed_native_separate",
            0,
            depth,
            "separate Q and C angles; common cumulative loss",
            "prescribed native-uniform rank-one projector",
            {
                "primary": separate_native + STRUCTURAL_TERMINAL_RU,
                "native_C": separate_native + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": separate_pauli + STRUCTURAL_TERMINAL_RU,
            },
            evaluations,
            separate.updates_per_restart,
            replay_spec={"kind": "one_stage", "family": "fixed_native_separate"},
        )

    def _run_order_controls(self, first: str, p1: int, p2: int, stream: int) -> None:
        second = "C" if first == "Q" else "Q"
        phase_second = self.c if second == "C" else self.q
        first_evals, second_evals = budget_plan(self.config, 2)
        stage1 = self._stage1(first, p1, first_evals, stream)
        stage1_metrics = self.metrics(stage1.state)
        stage1_native = self.first_ru(first, p1, C_NATIVE_RU)
        stage1_pauli = self.first_ru(first, p1, C_PAULI_RU)
        self._public_result(
            stage1,
            f"Stage1-{first}-p{p1}-for-{second}-p{p2}",
            "stage1",
            p1,
            0,
            f"{first} normalized",
            "complete block-XY",
            {
                "primary": stage1_native + STRUCTURAL_TERMINAL_RU,
                "native_C": stage1_native + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": stage1_pauli + STRUCTURAL_TERMINAL_RU,
            },
            first_evals,
            stage1.updates_per_restart,
        )

        full = optimize(
            lambda gamma, beta: self.history_energy(
                stage1.state, phase_second, gamma, beta, p2
            ),
            p2,
            p2,
            second_evals,
            f"LP-{first}-to-{second}-p{p1}-{p2}",
            self.config,
            self.real_dtype,
            self.device,
            stream + 10,
        )
        full_native = self.history_ru(
            stage1_native, p2, C_NATIVE_RU if second == "C" else Q_RU
        )
        full_pauli = self.history_ru(
            stage1_pauli, p2, C_PAULI_RU if second == "C" else Q_RU
        )
        self._public_result(
            full,
            f"LP-{first}-to-{second}-p{p1}-{p2}",
            "full_lp",
            p1,
            p2,
            f"new {second} phase; common cumulative loss",
            "learned history projector",
            {
                "primary": full_native + STRUCTURAL_TERMINAL_RU,
                "native_C": full_native + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": full_pauli + STRUCTURAL_TERMINAL_RU,
            },
            first_evals + second_evals,
            stage1.updates_per_restart + full.updates_per_restart,
            stage1_metrics["final_mass"],
            stage1,
            {"kind": "pipeline", "family": "full_lp", "first": first, "second": second},
        )

        warm = optimize(
            lambda gamma, beta: self.block_energy(
                stage1.state, (self.q, self.c), self.common, gamma, beta, p2
            ),
            2 * p2,
            p2,
            second_evals,
            f"Warm-{first}-separate-QC-p{p1}-{p2}",
            self.config,
            self.real_dtype,
            self.device,
            stream + 20,
        )
        warm_native = stage1_native + p2 * (Q_RU + C_NATIVE_RU + XY_RU)
        warm_pauli = stage1_pauli + p2 * (Q_RU + C_PAULI_RU + XY_RU)
        self._public_result(
            warm,
            f"Warm-{first}-separate-QC-p{p1}-{p2}",
            "warm",
            p1,
            p2,
            "separate Q and C angles; common cumulative loss",
            "complete block-XY",
            {
                "primary": warm_native + STRUCTURAL_TERMINAL_RU,
                "native_C": warm_native + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": warm_pauli + STRUCTURAL_TERMINAL_RU,
            },
            first_evals + second_evals,
            stage1.updates_per_restart + warm.updates_per_restart,
            stage1_metrics["final_mass"],
            stage1,
            {"kind": "pipeline", "family": "warm", "first": first, "second": second},
        )

        if first == "Q":
            mixer_kind = "capacity_switch"
            mixer_name = (
                f"75 adjacent-agent four-bit switches; fixed {self.config.trotter_sweeps}-sweep Trotter"
            )
            proxy = CAPACITY_SWITCH_PROXY_RU
            digital = CAPACITY_SWITCH_PAULI_SWEEP_RU * self.config.trotter_sweeps
            new_phase_native, new_phase_pauli = C_NATIVE_RU, C_PAULI_RU
        else:
            mixer_kind = "conflict_guarded"
            mixer_name = (
                f"75 symmetric six-local guarded XY templates; fixed {self.config.trotter_sweeps}-sweep Trotter"
            )
            proxy = CONFLICT_GUARDED_PROXY_RU
            digital = CONFLICT_GUARDED_PAULI_SWEEP_RU * self.config.trotter_sweeps
            new_phase_native = new_phase_pauli = Q_RU
        local = optimize(
            lambda gamma, beta: self.template_energy(
                stage1.state,
                phase_second,
                self.local_mixer(mixer_kind),
                gamma,
                beta,
                p2,
            ),
            p2,
            p2,
            second_evals,
            f"Local-{first}-to-{second}-p{p1}-{p2}",
            self.config,
            self.real_dtype,
            self.device,
            stream + 30,
        )
        local_proxy = stage1_native + p2 * (new_phase_native + proxy)
        local_digital_native = stage1_native + p2 * (new_phase_native + digital)
        local_digital_pauli = stage1_pauli + p2 * (new_phase_pauli + digital)
        self._public_result(
            local,
            f"Local-{first}-to-{second}-p{p1}-{p2}",
            "local",
            p1,
            p2,
            f"new {second} phase; common cumulative loss",
            mixer_name,
            {
                # The physical mixer uses the literal Pauli ledger, while the
                # hard C phase retains the same native primary convention as LP.
                "primary": local_digital_native + STRUCTURAL_TERMINAL_RU,
                "generator_proxy_native_C": local_proxy + STRUCTURAL_TERMINAL_RU,
                f"{self.config.trotter_sweeps}_sweep_native_C": local_digital_native
                + STRUCTURAL_TERMINAL_RU,
                f"{self.config.trotter_sweeps}_sweep_pauli_C": local_digital_pauli
                + STRUCTURAL_TERMINAL_RU,
            },
            first_evals + second_evals,
            stage1.updates_per_restart + local.updates_per_restart,
            stage1_metrics["final_mass"],
            stage1,
            {
                "kind": "pipeline",
                "family": "local",
                "first": first,
                "second": second,
                "local_mixer": mixer_kind,
                "trotter_sweeps": self.config.trotter_sweeps,
            },
        )

    def _run_pipeline_family(
        self,
        first: str,
        p1: int,
        p2: int,
        stream: int,
        family: str,
        *,
        emit_stage1: bool,
    ) -> list[dict[str, object]]:
        """Run one independently shardable pipeline family.

        The deterministic stage-1 optimization is repeated inside a shard so
        that a resumed task has no hidden in-memory dependency.  Since its
        angles depend only on the frozen seed and stream, this is equivalent to
        loading the corresponding saved checkpoint and preserves the exact
        2400+2400 forward-evaluation contract.
        """
        if first not in {"Q", "C"} or family not in {"full_lp", "warm", "local"}:
            raise ValueError((first, family))
        before = len(self.rows)
        second = "C" if first == "Q" else "Q"
        phase_second = self.c if second == "C" else self.q
        first_evals, second_evals = budget_plan(self.config, 2)
        stage1 = self._stage1(first, p1, first_evals, stream)
        stage1_metrics = self.metrics(stage1.state)
        stage1_native = self.first_ru(first, p1, C_NATIVE_RU)
        stage1_pauli = self.first_ru(first, p1, C_PAULI_RU)
        if emit_stage1:
            self._public_result(
                stage1,
                f"Stage1-{first}-p{p1}-for-{second}-p{p2}",
                "stage1",
                p1,
                0,
                f"{first} normalized",
                "complete block-XY",
                {
                    "primary": stage1_native + STRUCTURAL_TERMINAL_RU,
                    "native_C": stage1_native + STRUCTURAL_TERMINAL_RU,
                    "pauli_C_sensitivity": stage1_pauli + STRUCTURAL_TERMINAL_RU,
                },
                first_evals,
                stage1.updates_per_restart,
            )

        if family == "full_lp":
            result = optimize(
                lambda gamma, beta: self.history_energy(
                    stage1.state, phase_second, gamma, beta, p2
                ),
                p2,
                p2,
                second_evals,
                f"LP-{first}-to-{second}-p{p1}-{p2}",
                self.config,
                self.real_dtype,
                self.device,
                stream + 10,
            )
            native = self.history_ru(
                stage1_native, p2, C_NATIVE_RU if second == "C" else Q_RU
            )
            pauli = self.history_ru(
                stage1_pauli, p2, C_PAULI_RU if second == "C" else Q_RU
            )
            self._public_result(
                result,
                f"LP-{first}-to-{second}-p{p1}-{p2}",
                "full_lp",
                p1,
                p2,
                f"new {second} phase; common cumulative loss",
                "learned history projector",
                {
                    "primary": native + STRUCTURAL_TERMINAL_RU,
                    "native_C": native + STRUCTURAL_TERMINAL_RU,
                    "pauli_C_sensitivity": pauli + STRUCTURAL_TERMINAL_RU,
                },
                first_evals + second_evals,
                stage1.updates_per_restart + result.updates_per_restart,
                stage1_metrics["final_mass"],
                stage1,
                {"kind": "pipeline", "family": "full_lp", "first": first, "second": second},
            )
        elif family == "warm":
            result = optimize(
                lambda gamma, beta: self.block_energy(
                    stage1.state, (self.q, self.c), self.common, gamma, beta, p2
                ),
                2 * p2,
                p2,
                second_evals,
                f"Warm-{first}-separate-QC-p{p1}-{p2}",
                self.config,
                self.real_dtype,
                self.device,
                stream + 20,
            )
            native = stage1_native + p2 * (Q_RU + C_NATIVE_RU + XY_RU)
            pauli = stage1_pauli + p2 * (Q_RU + C_PAULI_RU + XY_RU)
            self._public_result(
                result,
                f"Warm-{first}-separate-QC-p{p1}-{p2}",
                "warm",
                p1,
                p2,
                "separate Q and C angles; common cumulative loss",
                "complete block-XY",
                {
                    "primary": native + STRUCTURAL_TERMINAL_RU,
                    "native_C": native + STRUCTURAL_TERMINAL_RU,
                    "pauli_C_sensitivity": pauli + STRUCTURAL_TERMINAL_RU,
                },
                first_evals + second_evals,
                stage1.updates_per_restart + result.updates_per_restart,
                stage1_metrics["final_mass"],
                stage1,
                {"kind": "pipeline", "family": "warm", "first": first, "second": second},
            )
        else:
            if first == "Q":
                mixer_kind = "capacity_switch"
                template_description = "75 adjacent-agent four-bit switches"
            else:
                mixer_kind = "conflict_guarded"
                template_description = "75 symmetric six-local guarded XY templates"
            result = optimize(
                lambda gamma, beta: self.template_energy(
                    stage1.state,
                    phase_second,
                    self.local_mixer(mixer_kind),
                    gamma,
                    beta,
                    p2,
                ),
                p2,
                p2,
                second_evals,
                f"Local-{first}-to-{second}-p{p1}-{p2}-s{self.config.trotter_sweeps}",
                self.config,
                self.real_dtype,
                self.device,
                stream + 30,
            )
            local_ru = local_control_ru_ledger(
                first, p1, p2, self.config.trotter_sweeps
            )
            self._public_result(
                result,
                f"Local-{first}-to-{second}-p{p1}-{p2}-s{self.config.trotter_sweeps}",
                "local",
                p1,
                p2,
                f"new {second} phase; common cumulative loss",
                (
                    f"{template_description}; fixed "
                    f"{self.config.trotter_sweeps}-sweep first-order product formula"
                ),
                {
                    **local_ru,
                    f"{self.config.trotter_sweeps}_sweep_native_C": local_ru["native_C"],
                    f"{self.config.trotter_sweeps}_sweep_pauli_C": local_ru[
                        "pauli_C_sensitivity"
                    ],
                },
                first_evals + second_evals,
                stage1.updates_per_restart + result.updates_per_restart,
                stage1_metrics["final_mass"],
                stage1,
                {
                    "kind": "pipeline",
                    "family": "local",
                    "first": first,
                    "second": second,
                    "local_mixer": mixer_kind,
                    "trotter_sweeps": self.config.trotter_sweeps,
                },
            )
        return self.rows[before:]

    def _run_direct(self, depth: int, stream: int) -> None:
        (evaluations,) = budget_plan(self.config, 1)
        joint = optimize(
            lambda gamma, beta: self.block_energy(
                self.native, (self.common,), self.common, gamma, beta, depth
            ),
            depth,
            depth,
            evaluations,
            f"Collapsed-joint-QC-p{depth}",
            self.config,
            self.real_dtype,
            self.device,
            stream,
        )
        joint_circuit = PREP_RU + depth * (QC_UNION_RU + XY_RU)
        self._public_result(
            joint,
            f"Collapsed-joint-QC-p{depth}",
            "direct_joint",
            0,
            depth,
            "single Q+C union angle; common cumulative loss",
            "complete block-XY",
            {
                "primary": joint_circuit + STRUCTURAL_TERMINAL_RU,
                "native_C": joint_circuit + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": joint_circuit + STRUCTURAL_TERMINAL_RU,
            },
            evaluations,
            joint.updates_per_restart,
            replay_spec={"kind": "one_stage", "family": "direct_joint"},
        )
        separate = optimize(
            lambda gamma, beta: self.block_energy(
                self.native, (self.q, self.c), self.common, gamma, beta, depth
            ),
            2 * depth,
            depth,
            evaluations,
            f"Direct-separate-Q-C-p{depth}",
            self.config,
            self.real_dtype,
            self.device,
            stream + 1,
        )
        separate_native = PREP_RU + depth * (Q_RU + C_NATIVE_RU + XY_RU)
        separate_pauli = PREP_RU + depth * (Q_RU + C_PAULI_RU + XY_RU)
        self._public_result(
            separate,
            f"Direct-separate-Q-C-p{depth}",
            "direct_separate",
            0,
            depth,
            "separate Q and C angles; common cumulative loss",
            "complete block-XY",
            {
                "primary": separate_native + STRUCTURAL_TERMINAL_RU,
                "native_C": separate_native + STRUCTURAL_TERMINAL_RU,
                "pauli_C_sensitivity": separate_pauli + STRUCTURAL_TERMINAL_RU,
            },
            evaluations,
            separate.updates_per_restart,
            replay_spec={"kind": "one_stage", "family": "direct_separate"},
        )

    def stage_a_necessity_bounds(
        self, coarse_rows: Sequence[dict[str, object]]
    ) -> dict[str, object]:
        """Optimistic exact-feasibility Grover bounds; no learned stage 2 is run."""
        uniform_probability = EXPECTED_SUPPORTS["final"] / EXPECTED_SUPPORTS["native"]
        fixed_rows: list[dict[str, object]] = []
        for depth in self.config.stage_a_grover_depths:
            probability = float(grover_success(uniform_probability, depth))
            ru = comparator_total_ru("fixed_native_joint", depth)
            shots = rts99(probability)
            fixed_rows.append(
                {
                    "algorithm": f"Analytic-fixed-uniform-exact-feasibility-Grover-p{depth}",
                    "depth": depth,
                    "initial_mass": uniform_probability,
                    "final_mass": probability,
                    "RTS99": shots,
                    "primary_RU": ru,
                    "RTS99_x_RU": None if shots is None else shots * ru,
                    "phase_oracle": "exact Q=0 and C=0 indicator",
                    "oracle_RU_assumption": "optimistic Q-union-C phase lower bound (420 RU)",
                }
            )
        finite_fixed = [row for row in fixed_rows if row["RTS99_x_RU"] is not None]
        best_fixed = min(finite_fixed, key=lambda row: int(row["RTS99_x_RU"]))

        learned_rows: list[dict[str, object]] = []
        for coarse in coarse_rows:
            initial = np.asarray(coarse["final_mass"], dtype=np.float64)
            prior_circuit_ru = int(coarse["RU"]["primary"]) - STRUCTURAL_TERMINAL_RU
            coarse_cost = coarse["feasibility_RTS99_x_primary_RU"]
            for depth in self.config.stage_a_grover_depths:
                final = grover_success(initial, depth)
                total_ru = (
                    (1 + 2 * depth) * prior_circuit_ru
                    + depth * (QC_UNION_RU + SELECTIVE_RU)
                    + STRUCTURAL_TERMINAL_RU
                )
                shots = [rts99(float(value)) for value in final]
                costs = [None if value is None else int(value) * total_ru for value in shots]
                learned_rows.append(
                    {
                        "algorithm": f"Optimistic-{coarse['algorithm']}-exact-feasibility-Grover-p{depth}",
                        "coarse_algorithm": coarse["algorithm"],
                        "coarse_depth": int(coarse["depth_first"]),
                        "grover_depth": depth,
                        "initial_mass": initial.tolist(),
                        "final_mass": final.tolist(),
                        "RTS99": shots,
                        "primary_RU": total_ru,
                        "RTS99_x_RU": costs,
                        "paired_coarse_cost": coarse_cost,
                        "phase_oracle": "exact Q=0 and C=0 indicator",
                        "status": "optimistic learned-projector necessity bound, not a variational outcome",
                    }
                )

        # Stage A is a permissive *necessity* bound rather than the actual LP
        # model-selection gate.  Applying the final gate's 2%-of-maximum-mass
        # rule here could incorrectly stop a much cheaper, slightly lower-mass
        # optimistic setting.  First enforce only the two structural thresholds,
        # then select the lowest geometric-mean optimistic sampling cost.  The
        # actual complex64 LP grid below retains the frozen 2%/lower-RU rule.
        eligible: list[dict[str, object]] = []
        for row in learned_rows:
            final_candidate = np.asarray(row["final_mass"], dtype=float)
            initial_candidate = np.asarray(row["initial_mass"], dtype=float)
            candidate_cost = [
                math.nan if value is None else float(value)
                for value in row["RTS99_x_RU"]
            ]
            row["median_final_mass"] = float(np.median(final_candidate))
            row["median_stage_gain"] = float(
                np.median(final_candidate / np.maximum(initial_candidate, 1e-30))
            )
            row["geometric_mean_optimistic_cost"] = geometric_mean(candidate_cost)
            if (
                row["median_final_mass"] >= 0.25
                and row["median_stage_gain"] >= 2.0
                and math.isfinite(float(row["geometric_mean_optimistic_cost"]))
            ):
                eligible.append(row)
        if not eligible:
            return {
                "status": (
                    "EXCLUDED_SMOKE_NO_SCIENTIFIC_DECISION"
                    if self.config.profile == "smoke"
                    else "STOP_STRUCTURAL_NO_GO"
                ),
                "pass": None if self.config.profile == "smoke" else False,
                "selection_rule": (
                    "among optimistic rows meeting median mass >=0.25 and median "
                    "stage gain >=2, minimize geometric-mean RTS99 x primary RU; "
                    "then lower RU/depth"
                ),
                "selected_optimistic_lp_bound": None,
                "best_fixed_uniform_exact_feasibility_grover": best_fixed,
                "checks": {
                    "any_structurally_eligible_optimistic_lp_bound": False,
                    "all_costs_finite_positive": False,
                },
                "fixed_uniform_ladder": fixed_rows,
                "learned_reference_ladder": learned_rows,
            }
        selected = min(
            eligible,
            key=lambda row: (
                float(row["geometric_mean_optimistic_cost"]),
                int(row["primary_RU"]),
                int(row["grover_depth"]),
                int(row["coarse_depth"]),
                str(row["algorithm"]),
            ),
        )
        selected_cost = np.asarray(
            [math.nan if value is None else float(value) for value in selected["RTS99_x_RU"]]
        )
        coarse_cost = np.asarray(
            [math.nan if value is None else float(value) for value in selected["paired_coarse_cost"]]
        )
        fixed_cost = float(best_fixed["RTS99_x_RU"])
        coarse_ratio = coarse_cost / selected_cost
        fixed_ratio = np.full_like(selected_cost, fixed_cost) / selected_cost
        coarse_gm = geometric_mean(coarse_ratio.tolist())
        fixed_gm = geometric_mean(fixed_ratio.tolist())
        final = np.asarray(selected["final_mass"], dtype=float)
        initial = np.asarray(selected["initial_mass"], dtype=float)
        paired_required = math.ceil(0.75 * self.config.restarts)
        checks = {
            "median_final_mass_at_least_0p25": float(np.median(final)) >= 0.25,
            "median_stage_gain_at_least_2": float(
                np.median(final / np.maximum(initial, 1e-30))
            )
            >= 2.0,
            "coarse_cost_margin_at_least_1p25": bool(
                np.isfinite(coarse_gm) and coarse_gm >= 1.25
            ),
            "fixed_uniform_cost_margin_at_least_1p25": bool(
                np.isfinite(fixed_gm) and fixed_gm >= 1.25
            ),
            "coarse_paired_wins": int(np.sum(coarse_ratio > 1.0)) >= paired_required,
            "fixed_uniform_paired_wins": int(np.sum(fixed_ratio > 1.0)) >= paired_required,
            "all_costs_finite_positive": bool(
                np.all(np.isfinite(selected_cost))
                and np.all(selected_cost > 0)
                and np.all(np.isfinite(coarse_cost))
                and np.all(coarse_cost > 0)
                and math.isfinite(fixed_cost)
                and fixed_cost > 0
            ),
        }
        mathematical_pass = all(checks.values())
        return {
            "status": (
                "EXCLUDED_SMOKE_NO_SCIENTIFIC_DECISION"
                if self.config.profile == "smoke"
                else "PROCEED_TO_EXPENSIVE_STAGES" if mathematical_pass else "STOP_STRUCTURAL_NO_GO"
            ),
            "pass": None if self.config.profile == "smoke" else mathematical_pass,
            "selection_rule": (
                "among optimistic rows meeting median mass >=0.25 and median stage gain >=2, "
                "minimize geometric-mean RTS99 x primary RU; then lower RU/depth"
            ),
            "selected_optimistic_lp_bound": selected,
            "best_fixed_uniform_exact_feasibility_grover": best_fixed,
            "coarse_geometric_mean_cost_ratio": coarse_gm,
            "fixed_uniform_geometric_mean_cost_ratio": fixed_gm,
            "coarse_paired_wins": int(np.sum(coarse_ratio > 1.0)),
            "fixed_uniform_paired_wins": int(np.sum(fixed_ratio > 1.0)),
            "paired_wins_required": paired_required,
            "checks": checks,
            "fixed_uniform_ladder": fixed_rows,
            "learned_reference_ladder": learned_rows,
        }

    def run_stage_a(self) -> dict[str, object]:
        """Standalone coarse preparations plus analytic necessity bounds only."""
        started = time.time()
        coarse_rows = []
        stream = 50_000
        for depth in self.config.depths:
            coarse_rows.append(self._run_coarse("Q", depth, stream))
            stream += 10
            coarse_rows.append(self._run_coarse("C", depth, stream))
            stream += 10
        bounds = self.stage_a_necessity_bounds(coarse_rows)
        return self._payload(
            stage="A_NECESSITY",
            decision=bounds,
            ideal=self._ideal_rows(),
            started=started,
        )

    def _ideal_rows(self) -> list[dict[str, object]]:
        return [
            {
                "algorithm": "Ideal-uniform-native",
                "family": "ideal",
                "target_free": True,
                "final_mass": [EXPECTED_SUPPORTS["final"] / EXPECTED_SUPPORTS["native"]],
                "RU": {"primary": None},
                "oracle_preparation": False,
            },
            {
                "algorithm": "Ideal-uniform-capacity",
                "family": "ideal",
                "target_free": True,
                "final_mass": [EXPECTED_SUPPORTS["final"] / EXPECTED_SUPPORTS["quota"]],
                "RU": {"primary": None},
                "oracle_preparation": True,
            },
            {
                "algorithm": "Ideal-uniform-conflict",
                "family": "ideal",
                "target_free": True,
                "final_mass": [EXPECTED_SUPPORTS["final"] / EXPECTED_SUPPORTS["conflict"]],
                "RU": {"primary": None},
                "oracle_preparation": True,
            },
            {
                "algorithm": "Ideal-uniform-final",
                "family": "ideal",
                "target_free": True,
                "final_mass": [1.0],
                "RU": {"primary": None},
                "oracle_preparation": True,
            },
        ]

    def gate_decision(self) -> dict[str, object]:
        if self.config.profile != "valid":
            return {
                "status": "EXCLUDED_SMOKE_NO_SCIENTIFIC_DECISION",
                "pass": None,
            }
        full = [row for row in self.rows if row["family"] == "full_lp"]
        controls = [
            row
            for row in self.rows
            if row["family"]
            in {
                "coarse",
                "warm",
                "local",
                "direct_joint",
                "direct_separate",
                "fixed_native_joint",
                "fixed_native_separate",
            }
        ]
        if not full or not controls:
            return {"status": "FAIL_CLOSED_INCOMPLETE_METHOD_MATRIX", "pass": False}
        try:
            winner = select_full_lp_by_mass_ru(full)
        except (TypeError, ValueError):
            return {"status": "FAIL_CLOSED_INVALID_LP_SELECTION", "pass": False}
        winner_cost = np.asarray(
            [
                math.nan if value is None else float(value)
                for value in winner["feasibility_RTS99_x_primary_RU"]
            ]
        )
        control_costs = np.asarray(
            [
                [
                    math.nan if value is None else float(value)
                    for value in row["feasibility_RTS99_x_primary_RU"]
                ]
                for row in controls
            ],
            dtype=float,
        )
        costs_valid = bool(
            np.all(np.isfinite(winner_cost))
            and np.all(winner_cost > 0)
            and np.all(np.isfinite(control_costs))
            and np.all(control_costs > 0)
        )
        strongest = np.min(control_costs, axis=0) if costs_valid else np.full_like(winner_cost, math.nan)
        ratios = strongest / winner_cost
        stage1 = np.asarray(winner["stage1_final_mass"], dtype=float)
        final = np.asarray(winner["final_mass"], dtype=float)
        absolute = float(np.median(final)) >= 0.25
        stage_gain = float(np.median(final / np.maximum(stage1, 1e-30))) >= 2.0
        cost_ratio_gm = geometric_mean(ratios.tolist())
        cost_advantage = bool(np.isfinite(cost_ratio_gm) and cost_ratio_gm >= 1.25)
        paired_wins_required = math.ceil(0.75 * self.config.restarts)
        paired_wins = int(np.sum(ratios > 1.0)) >= paired_wins_required
        norms = np.asarray(winner["state_norm"], dtype=float)
        norm_pass = bool(np.max(np.abs(norms - 1.0)) <= (2e-5 if self.config.dtype == "complex64" else 1e-10))
        candidate_depths = tuple(
            sorted(
                set(self.config.ru_matched_direct_depths)
                | {
                    left + right
                    for left in self.config.depths
                    for right in self.config.learned_projector_depths
                }
            )
        )
        required = required_comparator_depths(int(winner["RU"]["primary"]), candidate_depths)
        completed = {
            family: tuple(
                sorted(
                    int(row["depth_second"])
                    for row in self.rows
                    if row["family"] == family
                )
            )
            for family in required
        }
        manifest_complete = all(
            set(depths).issubset(set(completed[family]))
            for family, depths in required.items()
        )
        preliminary = (
            absolute
            and stage_gain
            and cost_advantage
            and paired_wins
            and norm_pass
            and costs_valid
            and manifest_complete
        )
        return {
            "status": "SAVED_ANGLE_COMPLEX128_REPLAY_REQUIRED" if preliminary else "FAILED_PRELIMINARY",
            "selected_full_lp": winner["algorithm"],
            "selection_rule": "within 2% of maximum median final mass, then lowest primary RU",
            "median_final_mass": float(np.median(final)),
            "median_stage_gain": float(np.median(final / np.maximum(stage1, 1e-30))),
            "geometric_mean_strongest_control_over_lp_cost": cost_ratio_gm,
            "paired_cost_wins": int(np.sum(ratios > 1.0)),
            "paired_cost_wins_required": paired_wins_required,
            "absolute_mass_pass": absolute,
            "stage_gain_pass": stage_gain,
            "cost_advantage_pass": cost_advantage,
            "paired_wins_pass": paired_wins,
            "norm_pass": norm_pass,
            "all_costs_finite_positive": costs_valid,
            "required_comparator_depth_manifest": required,
            "completed_comparator_depth_manifest": completed,
            "comparator_manifest_complete": manifest_complete,
            "preliminary_pass": preliminary,
            "pass": None,
            "complex128_replay_required_before_final_claim": True,
        }

    def _payload(
        self,
        *,
        stage: str,
        decision: dict[str, object],
        ideal: list[dict[str, object]],
        started: float,
    ) -> dict[str, object]:
        return {
            "schema": "nonredundant-bcst-target-free-structural-screen-v2",
            "scope": "two hard constraints only; no soft-cost table or optimum target",
            "execution_stage": stage,
            "profile": self.config.profile,
            "scientific_decision_allowed": self.config.profile == "valid",
            "excluded_smoke": self.config.profile == "smoke",
            "config": {**asdict(self.config), "output": str(self.config.output)},
            "frozen_valid_config": {
                **asdict(valid_config()),
                "output": str(valid_config().output),
            },
            "protocol_deviation": (
                scientific_config_record(self.config)
                != scientific_config_record(valid_config())
            ),
            "execution_batch_deviation": (
                self.config.restart_batch != valid_config().restart_batch
            ),
            "support_counts": EXPECTED_SUPPORTS,
            "normalization": {
                "Q_shell_max": 48,
                "C_shell_max": 10,
                "common_loss": "Q/48+C/10",
            },
            "target_guard": {
                "objective_table_loaded": self.objective_table_loaded,
                "target_constructed": self.target_constructed,
                "target_probability_computed": False,
            },
            "evaluation_accounting": {
                "budget_definition": "forward loss evaluations per restart including one post-update endpoint per optimized call",
                "one_stage_plan": budget_plan(self.config, 1),
                "two_stage_plan": budget_plan(self.config, 2),
                "all_non_diagnostic_rows_match_budget": all(
                    row["aggregate_pipeline_evaluations_per_restart"]
                    == self.config.evaluation_budget
                    for row in self.rows
                    if row["family"] != "stage1"
                ),
            },
            "ru_provenance": {
                "primary_convention": "native C=30 for every family",
                "native": {
                    "P": PREP_RU,
                    "X": XY_RU,
                    "Q": Q_RU,
                    "C": C_NATIVE_RU,
                    "Q_union_C": QC_UNION_RU,
                    "S": SELECTIVE_RU,
                    "terminal": STRUCTURAL_TERMINAL_RU,
                },
                "sensitivities": {
                    "C_pauli": C_PAULI_RU,
                    "capacity_switch_generator_proxy": CAPACITY_SWITCH_PROXY_RU,
                    "capacity_switch_one_pauli_sweep": CAPACITY_SWITCH_PAULI_SWEEP_RU,
                    "conflict_guarded_generator_proxy": CONFLICT_GUARDED_PROXY_RU,
                    "conflict_guarded_one_pauli_sweep": CONFLICT_GUARDED_PAULI_SWEEP_RU,
                },
                "local_mixer_note": "primary local cost uses literal Pauli mixer sweeps but retains native C; generator proxy and Pauli-C are separate sensitivities",
            },
            "frozen_conditional_controls": {
                "local_digitized_sweeps": self.config.local_sweep_sensitivities,
                "analog_exact_adjacency": {
                    "required_after_stage_a": True,
                    "finite_RU_rank": False,
                    "label": "optimistic analog exact matrix exponential",
                },
                "direct_candidate_depths": tuple(
                    sorted(
                        set(self.config.ru_matched_direct_depths)
                        | {
                            left + right
                            for left in self.config.depths
                            for right in self.config.learned_projector_depths
                        }
                    )
                ),
            },
            "methods": self.rows,
            "ideal_references": ideal,
            "gate": decision,
            "traces": self.traces,
            "elapsed_sec": time.time() - started,
        }

    def run(self) -> dict[str, object]:
        """Excluded integration run; valid execution must begin with Stage A."""
        if self.config.profile == "valid":
            raise RuntimeError("valid execution is staged; call run_stage_a() first")
        started = time.time()
        stream = 1
        direct_depths: set[int] = set()
        for depth in self.config.depths:
            self._run_coarse("Q", depth, stream)
            stream += 10
            self._run_coarse("C", depth, stream)
            stream += 10
        for p1 in self.config.depths:
            for p2 in self.config.learned_projector_depths:
                self._run_order_controls("Q", p1, p2, stream)
                stream += 100
                self._run_order_controls("C", p1, p2, stream)
                stream += 100
                direct_depths.add(p1 + p2)
        direct_depths.update(self.config.ru_matched_direct_depths)
        for depth in sorted(direct_depths):
            self._run_direct(depth, stream)
            self._run_fixed_native(depth, stream + 10)
            stream += 100
        ideal = self._ideal_rows()
        decision = self.gate_decision()
        return self._payload(stage="EXCLUDED_INTEGRATION", decision=decision, ideal=ideal, started=started)


PER_RESTART_ROW_FIELDS = {
    "loss_by_restart",
    "state_norm",
    "quota_mass",
    "conflict_mass",
    "final_mass",
    "expected_Qtilde",
    "expected_Ctilde",
    "common_loss",
    "stage1_final_mass",
    "feasibility_RTS99",
    "feasibility_RTS99_x_primary_RU",
    "best_evaluation_by_restart",
    "gamma_by_restart",
    "beta_by_restart",
    "stage1_gamma_by_restart",
    "stage1_beta_by_restart",
    "initialization_seeds",
}


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _task_stream(task: dict[str, object]) -> int:
    """Deterministic initialization stream independent of execution order."""
    family_codes = {
        "coarse_bundle": 50,
        "full_lp": 100,
        "warm": 200,
        "local_digitized": 300,
        "fixed_native_pair": 400,
        "direct_pair": 500,
    }
    first_code = 0 if task.get("first") in {None, "Q"} else 1
    return (
        1_000_000 * family_codes[str(task["family"])]
        + 10_000 * int(task.get("depth_first") or 0)
        + 100 * int(task.get("depth_second") or 0)
        + 10 * int(task.get("trotter_sweeps") or 0)
        + first_code
    )


def _load_and_validate_shard(
    path: Path, manifest: dict[str, object], task: dict[str, object]
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.get("shard_sha256", ""))
    body = {key: value for key, value in payload.items() if key != "shard_sha256"}
    if claimed != sha256_json(body):
        raise RuntimeError(f"shard checksum mismatch: {path}")
    if payload.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError(f"shard manifest mismatch: {path}")
    if payload.get("task_id") != task.get("task_id"):
        raise RuntimeError(f"shard task mismatch: {path}")
    if payload.get("seeds") != task.get("seeds"):
        raise RuntimeError(f"shard seed mismatch: {path}")
    return payload


def execute_stage_a_shard(
    config: ScreenConfig,
    manifest: dict[str, object],
    task: dict[str, object],
    *,
    resume: bool,
) -> dict[str, object]:
    """Run one frozen restart shard containing every Stage-A coarse row."""
    if task.get("stage") != "A" or task.get("family") != "coarse_bundle":
        raise ValueError("not a Stage-A coarse bundle")
    output = Path(str(task["output"]))
    existing: dict[str, object] | None = None
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing shard: {output}")
        existing = _load_and_validate_shard(output, manifest, task)
        if existing.get("complete") is True:
            return existing

    seeds = tuple(int(value) for value in task["seeds"])
    batch_config = replace(
        config,
        restarts=len(seeds),
        initialization_seeds=seeds,
        restart_batch=len(seeds),
        output=output,
    )
    screen = StructuralScreen(batch_config)
    if existing is not None:
        screen.rows = list(existing.get("rows", []))
        screen.traces = dict(existing.get("traces", {}))
    stream = _task_stream(task)
    expected_algorithms = {
        f"Coarse-only-{kind}-p{depth}"
        for depth in config.depths
        for kind in ("Q", "C")
    }
    completed_algorithms = {str(row["algorithm"]) for row in screen.rows}
    if not completed_algorithms.issubset(expected_algorithms):
        raise RuntimeError("partial Stage-A shard contains an unexpected method")

    def save_progress() -> dict[str, object]:
        completed = {str(row["algorithm"]) for row in screen.rows}
        body = {
            "schema": f"{CAMPAIGN_SCHEMA}-shard-v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "task_id": task["task_id"],
            "batch_index": task["batch_index"],
            "seeds": list(seeds),
            "profile": config.profile,
            "scientific_decision_allowed": config.profile == "valid",
            "target_guard": {
                "objective_table_loaded": False,
                "target_constructed": False,
                "target_probability_computed": False,
            },
            "formal_decision_performed": False,
            "complete": completed == expected_algorithms,
            "completed_algorithms": sorted(completed),
            "rows": screen.rows,
            "traces": screen.traces,
        }
        body = _json_safe(body)
        payload = {**body, "shard_sha256": sha256_json(body)}
        atomic_write_json(output, payload)
        return payload

    for depth in config.depths:
        q_name = f"Coarse-only-Q-p{depth}"
        if q_name not in completed_algorithms:
            screen._run_coarse("Q", depth, stream + 10 * depth)
            completed_algorithms.add(q_name)
            save_progress()
        c_name = f"Coarse-only-C-p{depth}"
        if c_name not in completed_algorithms:
            screen._run_coarse("C", depth, stream + 10 * depth + 1)
            completed_algorithms.add(c_name)
            save_progress()
    payload = save_progress()
    if payload["complete"] is not True:
        raise RuntimeError("Stage-A shard failed its row-completeness check")
    return payload


def _merge_row_shards(
    rows: Sequence[dict[str, object]], expected_seeds: Sequence[int]
) -> dict[str, object]:
    if not rows:
        raise RuntimeError("cannot merge an empty row group")
    seed_position = {int(seed): index for index, seed in enumerate(expected_seeds)}
    ordered = sorted(
        rows,
        key=lambda row: seed_position[int(row["initialization_seeds"][0])],
    )
    observed_seeds = [
        int(seed) for row in ordered for seed in row["initialization_seeds"]
    ]
    if observed_seeds != [int(seed) for seed in expected_seeds]:
        raise RuntimeError("row shards are incomplete, duplicated, or out of seed order")

    merged = dict(ordered[0])
    for key in PER_RESTART_ROW_FIELDS:
        values = [row.get(key) for row in ordered]
        if all(value is None for value in values):
            merged[key] = None
        elif any(value is None for value in values):
            raise RuntimeError(f"inconsistent per-restart field {key}")
        else:
            merged[key] = [item for value in values for item in value]
    merged["elapsed_sec_this_call"] = float(
        sum(float(row["elapsed_sec_this_call"]) for row in ordered)
    )
    merged["aggregate_pipeline_state_evaluations"] = int(
        merged["aggregate_pipeline_evaluations_per_restart"] * len(expected_seeds)
    )
    for row in ordered[1:]:
        for key in (
            "algorithm",
            "family",
            "depth_first",
            "depth_second",
            "phase",
            "mixer",
            "training_loss",
            "RU",
            "evaluations_this_call_per_restart",
            "aggregate_pipeline_evaluations_per_restart",
            "replay_spec",
        ):
            if row.get(key) != merged.get(key):
                raise RuntimeError(f"static row metadata mismatch for {key}")
    return merged


def aggregate_stage_a(
    config: ScreenConfig, manifest: dict[str, object]
) -> dict[str, object]:
    """Completeness-check all Stage-A shards before any formal decision."""
    tasks = [task for task in manifest["tasks"] if task["stage"] == "A"]
    expected_batches = restart_seed_batches(
        config, str(manifest["execution_target"])
    )
    if len(tasks) != len(expected_batches):
        raise RuntimeError("Stage-A manifest has the wrong number of restart shards")
    shards = [
        _load_and_validate_shard(Path(str(task["output"])), manifest, task)
        for task in tasks
    ]
    if any(shard.get("complete") is not True for shard in shards):
        raise RuntimeError("formal Stage-A aggregation requires every complete shard")
    if any(shard["formal_decision_performed"] for shard in shards):
        raise RuntimeError("a restart shard improperly performed a formal decision")

    by_algorithm: dict[str, list[dict[str, object]]] = {}
    for shard in shards:
        for row in shard["rows"]:
            by_algorithm.setdefault(str(row["algorithm"]), []).append(row)
    expected_algorithms = {
        f"Coarse-only-{kind}-p{depth}"
        for depth in config.depths
        for kind in ("Q", "C")
    }
    if set(by_algorithm) != expected_algorithms:
        raise RuntimeError("Stage-A method matrix is incomplete or contains extras")
    rows = [
        _merge_row_shards(by_algorithm[name], config.initialization_seeds)
        for name in sorted(by_algorithm)
    ]

    decision_engine = object.__new__(StructuralScreen)
    decision_engine.config = config
    decision_engine.rows = rows
    decision_engine.traces = {
        "shards": [
            {
                "task_id": shard["task_id"],
                "seeds": shard["seeds"],
                "shard_sha256": shard["shard_sha256"],
            }
            for shard in shards
        ]
    }
    decision_engine.objective_table_loaded = False
    decision_engine.target_constructed = False
    decision = decision_engine.stage_a_necessity_bounds(rows)
    payload = decision_engine._payload(
        stage="A_NECESSITY_COMPLETE_16_RESTARTS",
        decision=decision,
        ideal=decision_engine._ideal_rows(),
        started=time.time(),
    )
    payload["manifest_sha256"] = manifest["manifest_sha256"]
    payload["stage_a_shards_complete"] = True
    payload["stage_a_shard_count"] = len(shards)
    payload["early_promotion_used"] = False
    return _json_safe(payload)


def run_stage_a_campaign(
    config: ScreenConfig,
    campaign_root: Path,
    execution_target: str,
    *,
    resume: bool,
) -> dict[str, object]:
    """Create/freeze a manifest, run atomic shards, and formally aggregate."""
    if config.profile != "valid":
        raise ValueError("the staged campaign runner is only for the frozen valid profile")
    manifest_path = campaign_root / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, config, execution_target)
    else:
        manifest = frozen_campaign_manifest(config, campaign_root, execution_target)
        atomic_write_json(manifest_path, manifest)
    for task in manifest["tasks"]:
        if task["stage"] == "A":
            execute_stage_a_shard(config, manifest, task, resume=resume)
    payload = aggregate_stage_a(config, manifest)
    aggregate_path = campaign_root / "stage_A_aggregate.json"
    atomic_write_json(aggregate_path, payload)
    write_summary(payload, campaign_root / "stage_A_aggregate.md")
    return payload


def select_saved_angle_replay_rows(
    source_payload: dict[str, object]
) -> list[dict[str, object]]:
    """Freeze the complex64 winner and the controls realizing its cost envelope."""
    gate = source_payload.get("gate")
    if not isinstance(gate, dict) or not gate.get("preliminary_pass"):
        raise RuntimeError("precision replay requires a complete preliminary complex64 pass")
    winner_name = gate.get("selected_full_lp")
    methods = source_payload.get("methods")
    if not isinstance(methods, list):
        raise RuntimeError("source payload lacks method rows")
    winners = [row for row in methods if row.get("algorithm") == winner_name]
    if len(winners) != 1:
        raise RuntimeError("selected complex64 LP row is missing or ambiguous")
    winner = winners[0]
    controls = [
        row
        for row in methods
        if row.get("family")
        in {
            "coarse",
            "warm",
            "local",
            "direct_joint",
            "direct_separate",
            "fixed_native_joint",
            "fixed_native_separate",
        }
    ]
    if not controls:
        raise RuntimeError("no finite controls are available for precision replay")
    cost_matrix = np.asarray(
        [
            [
                math.nan if value is None else float(value)
                for value in row["feasibility_RTS99_x_primary_RU"]
            ]
            for row in controls
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(cost_matrix)) or np.any(cost_matrix <= 0):
        raise RuntimeError("invalid complex64 control cost fails closed before replay")
    selected_indices = sorted(set(int(index) for index in np.argmin(cost_matrix, axis=0)))
    selected = [winner, *(controls[index] for index in selected_indices)]
    if any(not isinstance(row.get("replay_spec"), dict) for row in selected):
        raise RuntimeError("a frozen replay row lacks saved-angle metadata")
    return selected


def run_saved_angle_precision_replay(
    source_payload: dict[str, object], config: ScreenConfig
) -> dict[str, object]:
    """Replay fixed complex64 configurations in complex128 without optimization."""
    selected_rows = select_saved_angle_replay_rows(source_payload)
    replayed: list[dict[str, object]] = []
    for row in selected_rows:
        spec = row["replay_spec"]
        sweeps = int(spec.get("trotter_sweeps", config.trotter_sweeps))
        row_config = replace(config, dtype="complex128", trotter_sweeps=sweeps)
        screen = StructuralScreen(row_config)
        replay = screen.replay_saved_row(row)
        replayed.append({**row, **replay})

    winner = replayed[0]
    controls = replayed[1:]
    winner_cost = np.asarray(
        [
            math.nan if value is None else float(value)
            for value in winner["feasibility_RTS99_x_primary_RU"]
        ],
        dtype=float,
    )
    control_costs = np.asarray(
        [
            [
                math.nan if value is None else float(value)
                for value in row["feasibility_RTS99_x_primary_RU"]
            ]
            for row in controls
        ],
        dtype=float,
    )
    finite = bool(
        np.all(np.isfinite(winner_cost))
        and np.all(winner_cost > 0)
        and np.all(np.isfinite(control_costs))
        and np.all(control_costs > 0)
    )
    ratios = (
        np.min(control_costs, axis=0) / winner_cost
        if finite
        else np.full_like(winner_cost, math.nan)
    )
    final = np.asarray(winner["final_mass"], dtype=float)
    stage1 = np.asarray(winner["stage1_final_mass"], dtype=float)
    norms = np.asarray(winner["state_norm"], dtype=float)
    ratio_gm = geometric_mean(ratios.tolist())
    paired_required = math.ceil(0.75 * config.restarts)
    checks = {
        "source_preliminary_pass": bool(source_payload["gate"]["preliminary_pass"]),
        "no_optimization_or_reselection": all(
            row["optimization_performed"] is False
            and row["configuration_reselected"] is False
            and row["angles_reselected"] is False
            for row in replayed
        ),
        "median_final_mass_at_least_0p25": float(np.median(final)) >= 0.25,
        "median_stage_gain_at_least_2": float(
            np.median(final / np.maximum(stage1, 1e-30))
        )
        >= 2.0,
        "geometric_mean_cost_margin_at_least_1p25": bool(
            math.isfinite(ratio_gm) and ratio_gm >= 1.25
        ),
        "paired_wins_at_least_75_percent": int(np.sum(ratios > 1.0))
        >= paired_required,
        "complex128_norm_tolerance": bool(np.max(np.abs(norms - 1.0)) <= 1e-10),
        "all_costs_finite_positive": finite,
    }
    passed = all(checks.values())
    return _json_safe(
        {
            "schema": "nonredundant-bcst-saved-angle-complex128-replay-v1",
            "profile": config.profile,
            "target_free": True,
            "source_selected_algorithm": winner["algorithm"],
            "configuration_selection_performed": False,
            "optimization_performed": False,
            "selected_control_algorithms": [row["algorithm"] for row in controls],
            "replayed_rows": replayed,
            "gate": {
                "status": "FINAL_STRUCTURAL_PASS" if passed else "FAILED_COMPLEX128_REPLAY",
                "pass": passed,
                "checks": checks,
                "geometric_mean_strongest_control_over_lp_cost": ratio_gm,
                "paired_cost_wins": int(np.sum(ratios > 1.0)),
                "paired_cost_wins_required": paired_required,
            },
        }
    )


def write_summary(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Target-free structural screen",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Scientific decision allowed: `{payload['scientific_decision_allowed']}`",
        f"- Gate: `{payload['gate']['status']}`",
        f"- Target guard: `{payload['target_guard']}`",
        "",
        "| method | median final mass | primary RU | median RTS99 x RU |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["methods"]:
        final = float(np.median(np.asarray(row["final_mass"], dtype=float)))
        ru = row["RU"].get("primary")
        finite = [value for value in row["feasibility_RTS99_x_primary_RU"] if value is not None]
        cost = float(np.median(finite)) if finite else None
        lines.append(f"| {row['algorithm']} | {final:.8g} | {ru} | {cost} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("valid", "smoke"), default="smoke")
    parser.add_argument(
        "--stage", choices=("smoke", "manifest", "stage-a", "replay"), default="smoke"
    )
    parser.add_argument(
        "--execution-target", choices=("remote80", "local8"), default="remote80"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--replay-input", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ScreenConfig:
    config = valid_config() if args.profile == "valid" else ScreenConfig.smoke()
    config = replace(
        config,
        restart_batch=(
            1 if args.profile == "smoke" else 4 if args.execution_target == "remote80" else 1
        ),
    )
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.dtype is not None:
        config = replace(config, dtype=args.dtype)
    if args.output is not None:
        config = replace(config, output=args.output)
    config.validate()
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.stage in {"manifest", "stage-a"} and config.dtype != "complex64":
        raise RuntimeError("the frozen optimization stages require complex64")
    campaign_root = (
        args.campaign_root
        if args.campaign_root is not None
        else config.output.with_suffix("")
    )
    if args.stage == "smoke":
        if config.profile != "smoke":
            raise RuntimeError("valid execution requires an explicit staged command")
        payload = StructuralScreen(config).run()
        atomic_write_json(config.output, _json_safe(payload))
        write_summary(payload, config.output.with_suffix(".md"))
        destination = config.output
    elif args.stage == "manifest":
        if config.profile != "valid":
            raise RuntimeError("the frozen campaign manifest is defined for the valid profile")
        manifest = frozen_campaign_manifest(config, campaign_root, args.execution_target)
        destination = campaign_root / MANIFEST_FILENAME
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            validate_manifest(existing, config, args.execution_target)
            payload = existing
        else:
            atomic_write_json(destination, manifest)
            payload = manifest
    elif args.stage == "stage-a":
        if config.profile != "valid":
            raise RuntimeError("Stage A requires --profile valid")
        payload = run_stage_a_campaign(
            config,
            campaign_root,
            args.execution_target,
            resume=not args.no_resume,
        )
        destination = campaign_root / "stage_A_aggregate.json"
    else:
        if config.profile != "valid" or args.replay_input is None:
            raise RuntimeError("replay requires --profile valid and --replay-input")
        source = json.loads(args.replay_input.read_text(encoding="utf-8"))
        payload = run_saved_angle_precision_replay(source, replace(config, dtype="complex128"))
        destination = (
            args.output
            if args.output is not None
            else campaign_root / "stage_E_saved_angle_complex128_replay.json"
        )
        atomic_write_json(destination, payload)
    print(
        json.dumps(
            {
                "output": str(destination),
                "stage": args.stage,
                "status": payload.get("gate", {}).get("status"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
