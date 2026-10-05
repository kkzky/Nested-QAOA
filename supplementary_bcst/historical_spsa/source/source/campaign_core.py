from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from hyper4_continuous_runner import (
    HeteroBlockWDSHCCInstance,
    HeteroConfig,
    HeteroQAOARunner,
)


CAMPAIGN = "bcst_signed_continuous_hyper4_20260726_v1"
ACCOUNTING_TAG = "phi0-full-xy-history-replay-variable-k-v1"
SUPPORT_ALGORITHM = "eligible-mixed4-shuffled-per-composition-weighted-deficit-v1"
RNG_ALGORITHM = "sha256-domain-separated-seed-pcg64dxsm-v1"
TARGET_ALGORITHM = "feasible-energy-stable-packed-bitstring-v1"

LAYOUTS: dict[int, tuple[int, ...]] = {
    20: (7, 7, 6),
    22: (6, 6, 5, 5),
    24: (6, 6, 6, 6),
    26: (7, 7, 6, 6),
    28: (6, 6, 6, 5, 5),
    30: (6, 6, 6, 6, 6),
    32: (7, 7, 6, 6, 6),
    34: (6, 6, 6, 6, 5, 5),
    36: (6, 6, 6, 6, 6, 6),
}

EXPECTED_STRUCTURE: dict[int, dict[str, int]] = {
    20: {"D": 6_615, "F": 450, "C": 19, "X": 57, "H": 1_530, "1111": 0, "211": 1_020, "22": 510},
    22: {"D": 22_500, "F": 570, "C": 21, "X": 50, "H": 3_140, "1111": 420, "211": 2_240, "22": 480},
    24: {"D": 50_625, "F": 1_710, "C": 24, "X": 60, "H": 4_800, "1111": 630, "211": 3_360, "22": 810},
    26: {"D": 99_225, "F": 4_950, "C": 25, "X": 72, "H": 7_260, "1111": 930, "211": 5_100, "22": 1_230},
    28: {"D": 337_500, "F": 2_190, "C": 27, "X": 65, "H": 11_205, "1111": 2_750, "211": 7_370, "22": 1_085},
    30: {"D": 759_375, "F": 6_570, "C": 30, "X": 75, "H": 15_375, "1111": 3_750, "211": 10_050, "22": 1_575},
    32: {"D": 1_488_375, "F": 22_770, "C": 31, "X": 87, "H": 21_075, "1111": 5_070, "211": 13_830, "22": 2_175},
    34: {"D": 5_062_500, "F": 16_770, "C": 33, "X": 80, "H": 29_085, "1111": 9_800, "211": 17_360, "22": 1_925},
    36: {"D": 11_390_625, "F": 50_310, "C": 36, "X": 90, "H": 37_485, "1111": 12_600, "211": 22_320, "22": 2_565},
}

TERM_SCHEDULES: dict[str, dict[int, int]] = {
    "L8": {n: 8 * n for n in LAYOUTS},
    "L12": {n: 12 * n for n in LAYOUTS},
    "L18": {n: 18 * n for n in LAYOUTS},
    "Q05": {n: int(round(n * n / 2)) for n in LAYOUTS},
    "F2": {n: int(round(0.02 * EXPECTED_STRUCTURE[n]["H"])) for n in LAYOUTS},
    "F4": {n: int(round(0.04 * EXPECTED_STRUCTURE[n]["H"])) for n in LAYOUTS},
    "F6": {n: int(round(0.06 * EXPECTED_STRUCTURE[n]["H"])) for n in LAYOUTS},
    "F10": {n: int(round(0.10 * EXPECTED_STRUCTURE[n]["H"])) for n in LAYOUTS},
    "F20": {n: int(round(0.20 * EXPECTED_STRUCTURE[n]["H"])) for n in LAYOUTS},
    "F40": {n: int(round(0.40 * EXPECTED_STRUCTURE[n]["H"])) for n in LAYOUTS},
    "P3": {30: 6150},
}

COMPOSITION_PROBABILITIES: dict[str, float] = {
    "1111": 0.45,
    "211": 0.35,
    "22": 0.20,
}


@dataclass(frozen=True)
class StreamSeeds:
    support_seed: int
    magnitude_seed: int
    sign_seed: int
    optimizer_seed: int


@dataclass
class ObjectiveBundle:
    terms: list[tuple[int, int, int, int]]
    weights: np.ndarray
    support_order: list[tuple[int, int, int, int]]
    support_compositions: list[str]
    magnitude_quantiles: np.ndarray
    sign_bits: np.ndarray
    seeds: StreamSeeds
    manifest: dict[str, Any]
    targets: dict[str, np.ndarray]
    target_manifest: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_csv_lossless(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV without an explicit schema")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def stream_seed(n: int, instance_seed: int, domain: str) -> int:
    payload = f"{CAMPAIGN}|N={n}|instance={instance_seed}|domain={domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def separated_streams(n: int, instance_seed: int, optimizer_domain: str = "optimizer") -> StreamSeeds:
    return StreamSeeds(
        support_seed=stream_seed(n, instance_seed, "support"),
        magnitude_seed=stream_seed(n, instance_seed, "magnitude"),
        sign_seed=stream_seed(n, instance_seed, "sign"),
        optimizer_seed=stream_seed(n, instance_seed, optimizer_domain),
    )


def make_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(seed))


def xy_pair_cost(layout: Sequence[int]) -> int:
    return int(sum(math.comb(int(k), 2) for k in layout))


def build_structure(
    n: int,
    *,
    device: str | torch.device = "cpu",
    dtype: str = "complex128",
    optimizer_seed: int | None = None,
    opt_restarts: int = 3,
    opt_steps: int = 85,
    opt_lr: float = 0.055,
    optimizer_name: str = "adam",
    spsa_c: float = 0.10,
    spsa_target_step: float = 0.10,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    spsa_stability_fraction: float = 0.10,
    spsa_calibration_directions: int = 8,
    spsa_directions_per_update: int = 1,
    spsa_max_step_norm: float = 0.50,
    patience: int = 18,
    continuation: bool = True,
    sequential_restarts: bool = False,
    trace_every: int = 0,
    penalty: float = 1.0,
) -> tuple[HeteroConfig, HeteroBlockWDSHCCInstance, HeteroQAOARunner]:
    if n not in LAYOUTS:
        raise ValueError(f"unsupported N={n}")
    resolved_device = torch.device(device)
    raw_seed = int(optimizer_seed if optimizer_seed is not None else stream_seed(n, 0, "stage1-optimizer"))
    # The frozen optimizer multiplies cfg.seed while deriving label-specific
    # Torch seeds, so constrain the domain-separated seed before handing it in.
    seed = raw_seed % 1_000_000_007
    cfg = HeteroConfig(
        layout=LAYOUTS[n],
        weights=tuple(2 for _ in LAYOUTS[n]),
        seed=seed,
        device=str(resolved_device),
        dtype=dtype,
        conflict_topology="cycle",
        conflict_weight_scale=1.0,
        penalty_scale=float(penalty),
        nested_stage2_guard_scale=0.0,
        opt_restarts=int(opt_restarts),
        opt_steps=int(opt_steps),
        opt_lr=float(opt_lr),
        optimizer_name=str(optimizer_name),
        spsa_c=float(spsa_c),
        spsa_target_step=float(spsa_target_step),
        spsa_alpha=float(spsa_alpha),
        spsa_gamma=float(spsa_gamma),
        spsa_stability_fraction=float(spsa_stability_fraction),
        spsa_calibration_directions=int(spsa_calibration_directions),
        spsa_directions_per_update=int(spsa_directions_per_update),
        spsa_max_step_norm=float(spsa_max_step_norm),
        early_stop_patience=int(patience),
        log_every=max(1, int(trace_every) if trace_every else 30),
        stage1_sequential_restarts=bool(sequential_restarts),
        direct_sequential_restarts=bool(sequential_restarts),
        stage2_sequential_restarts=bool(sequential_restarts),
        stage2_sequential_min_depth=1,
        continuation_init=bool(continuation),
        trace_every=int(trace_every),
    )
    inst = HeteroBlockWDSHCCInstance(cfg, resolved_device, seed)
    return cfg, inst, HeteroQAOARunner(cfg, inst)


def structural_summary(inst: HeteroBlockWDSHCCInstance) -> dict[str, Any]:
    n = inst.N
    pools = enumerate_eligible_supports(inst)
    counts = {label: len(values) for label, values in pools.items()}
    return {
        "N": n,
        "layout": list(inst.block_sizes),
        "block_weights": list(inst.block_weights),
        "product_shell_dimension": int(inst.dim),
        "conflict_feasible_count": int(inst.feasible_count),
        "conflict_edges": int(len(inst.conflict_edges)),
        "xy_pair_cost": xy_pair_cost(inst.block_sizes),
        "eligible_support_count": int(sum(counts.values())),
        "eligible_by_composition": counts,
    }


def composition_label(inst: HeteroBlockWDSHCCInstance, term: Sequence[int]) -> str | None:
    counts = sorted(
        (
            int(value)
            for value in np.bincount(
                np.array([inst.block_for_qubit(int(q)) for q in term], dtype=np.int16),
                minlength=inst.M,
            )
            if value
        ),
        reverse=True,
    )
    mapping = {
        (1, 1, 1, 1): "1111",
        (2, 1, 1): "211",
        (2, 2): "22",
    }
    return mapping.get(tuple(counts))


def enumerate_eligible_supports(
    inst: HeteroBlockWDSHCCInstance,
) -> dict[str, list[tuple[int, int, int, int]]]:
    cached = getattr(inst, "_eligible_support_pools", None)
    if cached is not None:
        return cached
    conflict_pairs = {
        tuple(sorted((int(u), int(v))))
        for u, v, _weight in inst.conflict_edges
    }
    pools: dict[str, list[tuple[int, int, int, int]]] = {
        "1111": [],
        "211": [],
        "22": [],
    }
    for raw_term in itertools.combinations(range(inst.N), 4):
        label = composition_label(inst, raw_term)
        if label is None:
            continue
        selected = set(raw_term)
        if any(u in selected and v in selected for u, v in conflict_pairs):
            continue
        term = tuple(int(q) for q in raw_term)
        pools[label].append(term)
    feasible = inst.feasible_state_bits_np
    for label, candidates in list(pools.items()):
        eligible: list[tuple[int, int, int, int]] = []
        for start in range(0, len(candidates), 256):
            batch = candidates[start : start + 256]
            masks = np.array([term_mask(term) for term in batch], dtype=np.uint64)
            active = np.any(
                (feasible[:, None] & masks[None, :]) == masks[None, :],
                axis=0,
            )
            eligible.extend(term for term, keep in zip(batch, active) if bool(keep))
        pools[label] = eligible
    inst._eligible_support_pools = pools
    return pools


def verify_support_activity(
    inst: HeteroBlockWDSHCCInstance,
    terms: Sequence[Sequence[int]],
    *,
    batch_size: int = 512,
) -> None:
    feasible = inst.feasible_state_bits_np
    for start in range(0, len(terms), batch_size):
        batch = terms[start : start + batch_size]
        masks = np.array([term_mask(term) for term in batch], dtype=np.uint64)
        active = np.any(
            (feasible[:, None] & masks[None, :]) == masks[None, :],
            axis=0,
        )
        if not bool(np.all(active)):
            failed = [tuple(batch[i]) for i in np.flatnonzero(~active)]
            raise AssertionError(f"conflict-feasible-inactive supports found: {failed[:5]}")


def term_mask(term: Sequence[int]) -> np.uint64:
    mask = np.uint64(0)
    for q in term:
        mask |= np.uint64(1) << np.uint64(int(q))
    return mask


def weighted_support_order(
    inst: HeteroBlockWDSHCCInstance,
    support_seed: int,
) -> tuple[list[tuple[int, int, int, int]], list[str], dict[str, int]]:
    pools = enumerate_eligible_supports(inst)
    active_labels = [label for label in ("1111", "211", "22") if pools[label]]
    probabilities = np.array(
        [COMPOSITION_PROBABILITIES[label] for label in active_labels],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()
    shuffled: dict[str, list[tuple[int, int, int, int]]] = {}
    for label in active_labels:
        rng = make_rng(stream_seed(inst.N, support_seed, f"support-order-{label}"))
        indices = rng.permutation(len(pools[label]))
        shuffled[label] = [pools[label][int(i)] for i in indices]

    used = {label: 0 for label in active_labels}
    order: list[tuple[int, int, int, int]] = []
    labels: list[str] = []
    total = sum(len(pools[label]) for label in active_labels)
    for position in range(total):
        available = [
            i
            for i, label in enumerate(active_labels)
            if used[label] < len(shuffled[label])
        ]
        if not available:
            break
        available_mass = float(probabilities[available].sum())
        deficits: list[tuple[float, int]] = []
        for i in available:
            label = active_labels[i]
            renorm_probability = float(probabilities[i] / available_mass)
            available_used = sum(used[active_labels[j]] for j in available)
            desired_after = renorm_probability * (available_used + 1)
            deficits.append((desired_after - used[label], -i))
        _deficit, neg_index = max(deficits)
        chosen_index = -neg_index
        chosen = active_labels[chosen_index]
        order.append(shuffled[chosen][used[chosen]])
        labels.append(chosen)
        used[chosen] += 1
    if len(order) != total or len(set(order)) != total:
        raise AssertionError("support allocator failed to produce a complete unique order")
    return order, labels, {label: len(pools[label]) for label in ("1111", "211", "22")}


def map_weights(
    family: str,
    magnitude_quantiles: np.ndarray,
    sign_bits: np.ndarray,
    *,
    range_a: float | None = None,
) -> np.ndarray:
    u = np.asarray(magnitude_quantiles, dtype=np.float64)
    signs = np.where(np.asarray(sign_bits, dtype=np.uint8) > 0, 1.0, -1.0)
    if family == "pos_disc_rms1":
        magnitudes = (np.floor(np.minimum(u, np.nextafter(1.0, 0.0)) * 3.0) + 1.0) / math.sqrt(14.0 / 3.0)
        return magnitudes
    if family == "signed_disc_rms1":
        magnitudes = (np.floor(np.minimum(u, np.nextafter(1.0, 0.0)) * 3.0) + 1.0) / math.sqrt(14.0 / 3.0)
        return signs * magnitudes
    if family == "pos_cont_rms1":
        return math.sqrt(3.0) * u
    if family == "signed_cont_rms1":
        return signs * math.sqrt(3.0) * u
    if family in {"signed_cont_uniform", "signed_cont_u1"}:
        a = float(1.0 if family == "signed_cont_u1" and range_a is None else range_a)
        if not math.isfinite(a) or a <= 0:
            raise ValueError("signed continuous range must be positive and finite")
        return signs * a * u
    raise ValueError(f"unknown weight family {family!r}")


def support_hash(terms: Sequence[Sequence[int]]) -> str:
    payload = [
        {"term_index": i, "qubits": [int(q) for q in term]}
        for i, term in enumerate(terms)
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def objective_hash(terms: Sequence[Sequence[int]], weights: Sequence[float]) -> str:
    payload = [
        {
            "term_index": i,
            "qubits": [int(q) for q in term],
            "weight_hex": float(weight).hex(),
        }
        for i, (term, weight) in enumerate(zip(terms, weights))
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def objective_cost_numpy(
    state_bits: np.ndarray,
    terms: Sequence[Sequence[int]],
    weights: Sequence[float],
) -> np.ndarray:
    cost = np.zeros(len(state_bits), dtype=np.float64)
    for term, weight in zip(terms, weights):
        mask = term_mask(term)
        cost[(state_bits & mask) == mask] -= float(weight)
    return cost


def objective_cost_torch(
    state_bits: np.ndarray,
    terms: Sequence[Sequence[int]],
    weights: Sequence[float],
    *,
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    if device.type != "cuda" or len(state_bits) < 100_000:
        values = objective_cost_numpy(state_bits, terms, weights)
        return torch.tensor(values, dtype=real_dtype, device=device)
    states = torch.tensor(state_bits.view(np.int64), dtype=torch.int64, device=device)
    result = torch.zeros(len(state_bits), dtype=real_dtype, device=device)
    masks = np.array([term_mask(term) for term in terms], dtype=np.uint64).view(np.int64)
    weights_np = np.asarray(weights, dtype=np.float64)
    term_batch = 64 if real_dtype == torch.float64 else 128
    for start in range(0, len(terms), term_batch):
        stop = min(len(terms), start + term_batch)
        mask_batch = torch.tensor(masks[start:stop], dtype=torch.int64, device=device)
        weight_batch = torch.tensor(weights_np[start:stop], dtype=real_dtype, device=device)
        active = (torch.bitwise_and(states[:, None], mask_batch[None, :]) == mask_batch[None, :])
        result -= active.to(real_dtype) @ weight_batch
    return result


def target_count_sqrt(feasible_count: int) -> int:
    return max(2, int(round(2.0 * math.sqrt(float(feasible_count) / 450.0))))


def samples_to_confidence(probability: float, confidence: float = 0.99) -> int | None:
    if probability >= confidence:
        return 1
    if probability <= 0.0 or not math.isfinite(probability):
        return None
    return int(math.ceil(math.log1p(-confidence) / math.log1p(-probability)))


def rts_99(probability: float) -> int | None:
    return samples_to_confidence(float(probability), 0.99)


def build_targets(
    inst: HeteroBlockWDSHCCInstance,
    feasible_energies: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feasible_indices = np.flatnonzero(inst.feasible_mask_np)
    feasible_bits = inst.state_bits_np[feasible_indices]
    energies = np.asarray(feasible_energies, dtype=np.float64)
    if energies.shape != (inst.feasible_count,):
        raise ValueError("feasible-energy array has the wrong shape")
    order = np.lexsort((feasible_bits, energies))
    sorted_indices = feasible_indices[order]
    sorted_bits = feasible_bits[order]
    sorted_energies = energies[order]
    k_a = target_count_sqrt(inst.feasible_count)
    k_b = 2
    k_manuscript = max(1, min(inst.feasible_count, int(round(6.17e-4 * inst.feasible_count))))
    ground_tolerance = max(1e-12, 32.0 * np.finfo(np.float64).eps * max(1.0, abs(float(sorted_energies[0]))))
    ground_count = int(np.count_nonzero(np.abs(sorted_energies - sorted_energies[0]) <= ground_tolerance))
    targets = {
        "target_a_sqrt": sorted_indices[:k_a].astype(np.int64),
        "target_b_k2": sorted_indices[:k_b].astype(np.int64),
        "exact_ground": sorted_indices[:ground_count].astype(np.int64),
        "manuscript_fraction": sorted_indices[:k_manuscript].astype(np.int64),
    }
    target_rows: dict[str, Any] = {}
    for label, indices in targets.items():
        bits = inst.state_bits_np[indices]
        bitstrings = [format(int(value), f"0{inst.N}b") for value in bits]
        payload = {
            "label": label,
            "indices": [int(v) for v in indices],
            "bitstrings": bitstrings,
        }
        count = len(indices)
        p_feasible = count / inst.feasible_count
        p_product = count / inst.dim
        target_rows[label] = {
            **payload,
            "count": count,
            "sha256": sha256_bytes(canonical_json_bytes(payload)),
            "uniform_feasible_probability": p_feasible,
            "uniform_product_probability": p_product,
            "uniform_feasible_samples_99": samples_to_confidence(p_feasible),
            "uniform_product_samples_99": samples_to_confidence(p_product),
            "threshold_energy": float(inst.Cobj_np[int(indices[-1])]),
        }
    manifest = {
        "algorithm": TARGET_ALGORITHM,
        "tie_break": "ascending zero-padded packed bitstring",
        "ground_tolerance": ground_tolerance,
        "ground_degeneracy": ground_count,
        "minimum_feasible_energy": float(sorted_energies[0]),
        "targets": target_rows,
    }
    return targets, manifest


def install_objective(
    inst: HeteroBlockWDSHCCInstance,
    terms: Sequence[Sequence[int]],
    weights: Sequence[float],
) -> tuple[np.ndarray, torch.Tensor]:
    tensor = objective_cost_torch(
        inst.state_bits_np,
        terms,
        weights,
        device=inst.device,
        real_dtype=inst.real_dtype,
    )
    values = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    inst.objective_terms = {
        tuple(int(q) for q in term): float(weight)
        for term, weight in zip(terms, weights)
    }
    inst.Cobj_np = values
    inst.Cobj = tensor
    return values, tensor


def build_objective_bundle(
    inst: HeteroBlockWDSHCCInstance,
    *,
    instance_seed: int,
    schedule: str,
    family: str,
    range_a: float | None = None,
    optimizer_domain: str = "optimizer",
) -> ObjectiveBundle:
    n = inst.N
    if schedule not in TERM_SCHEDULES:
        raise ValueError(f"unknown term schedule {schedule!r}")
    m = int(TERM_SCHEDULES[schedule][n])
    seeds = separated_streams(n, instance_seed, optimizer_domain)
    order, labels, pool_counts = weighted_support_order(inst, seeds.support_seed)
    magnitude_rng = make_rng(seeds.magnitude_seed)
    sign_rng = make_rng(seeds.sign_seed)
    magnitude_quantiles = magnitude_rng.random(len(order), dtype=np.float64)
    sign_bits = sign_rng.integers(0, 2, size=len(order), dtype=np.uint8)
    all_weights = map_weights(family, magnitude_quantiles, sign_bits, range_a=range_a)
    terms = order[:m]
    weights = all_weights[:m].copy()
    selected_labels = labels[:m]
    support_sha = support_hash(terms)
    objective_sha = objective_hash(terms, weights)
    full_support_sha = support_hash(order)
    values, _tensor = install_objective(inst, terms, weights)
    feasible_energies = values[inst.feasible_mask_np]
    targets, target_manifest = build_targets(inst, feasible_energies)
    primary_mask = np.zeros(inst.dim, dtype=bool)
    primary_mask[targets["target_a_sqrt"]] = True
    inst.opt_mask_np = primary_mask
    inst.opt_mask = torch.tensor(primary_mask, dtype=torch.bool, device=inst.device)
    inst.min_feasible_obj = float(np.min(feasible_energies))

    empirical = {
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "rms": float(np.sqrt(np.mean(np.square(weights)))),
        "positive_count": int(np.count_nonzero(weights > 0)),
        "negative_count": int(np.count_nonzero(weights < 0)),
        "zero_count": int(np.count_nonzero(weights == 0)),
    }
    composition_counts = {
        label: int(selected_labels.count(label))
        for label in ("1111", "211", "22")
    }
    manifest = {
        "schema": "bcst-signed-continuous-objective-v1",
        "campaign": CAMPAIGN,
        "N": n,
        "layout": list(inst.block_sizes),
        "instance_seed": int(instance_seed),
        "stream_seeds": asdict(seeds),
        "rng_algorithm": RNG_ALGORITHM,
        "support_algorithm": SUPPORT_ALGORITHM,
        "support_schedule": schedule,
        "term_count": m,
        "weight_family": family,
        "range_a": None if range_a is None else float(range_a),
        "composition_probabilities": COMPOSITION_PROBABILITIES,
        "eligible_pool_counts": pool_counts,
        "selected_composition_counts": composition_counts,
        "rejections": {
            "duplicate": 0,
            "product_shell_inactive": 0,
            "conflict_feasible_inactive": 0,
            "note": "eligible pool is exhaustively enumerated before shuffling",
        },
        "support_sha256": support_sha,
        "full_support_order_sha256": full_support_sha,
        "objective_sha256": objective_sha,
        "weight_empirical": empirical,
        "target_manifest_sha256": sha256_bytes(canonical_json_bytes(target_manifest)),
    }
    return ObjectiveBundle(
        terms=terms,
        weights=weights,
        support_order=order,
        support_compositions=labels,
        magnitude_quantiles=magnitude_quantiles,
        sign_bits=sign_bits,
        seeds=seeds,
        manifest=manifest,
        targets=targets,
        target_manifest=target_manifest,
    )


def save_objective_bundle(directory: Path | str, bundle: ObjectiveBundle) -> dict[str, str]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    support_rows = [
        {
            "order_index": i,
            "composition": bundle.support_compositions[i],
            "q0": term[0],
            "q1": term[1],
            "q2": term[2],
            "q3": term[3],
            "selected": int(i < len(bundle.terms)),
            "magnitude_quantile_hex": float(bundle.magnitude_quantiles[i]).hex(),
            "sign_bit": int(bundle.sign_bits[i]),
        }
        for i, term in enumerate(bundle.support_order)
    ]
    objective_rows = [
        {
            "term_index": i,
            "composition": bundle.support_compositions[i],
            "q0": term[0],
            "q1": term[1],
            "q2": term[2],
            "q3": term[3],
            "weight": format(float(weight), ".17g"),
            "weight_hex": float(weight).hex(),
        }
        for i, (term, weight) in enumerate(zip(bundle.terms, bundle.weights))
    ]
    write_csv_lossless(root / "support_order.csv", support_rows)
    write_csv_lossless(root / "objective_terms.csv", objective_rows)
    atomic_write_json(root / "objective_manifest.json", bundle.manifest)
    atomic_write_json(root / "target_manifest.json", bundle.target_manifest)
    outputs = {}
    for name in ("support_order.csv", "objective_terms.csv", "objective_manifest.json", "target_manifest.json"):
        outputs[name] = sha256_file(root / name)
    atomic_write_json(root / "file_hashes.json", outputs)
    return outputs


def replay_probabilities(
    state: torch.Tensor,
    inst: HeteroBlockWDSHCCInstance,
    bundle: ObjectiveBundle,
) -> dict[str, float]:
    with torch.no_grad():
        probability = torch.abs(state) ** 2
        result = {
            label: float(torch.sum(probability[torch.tensor(indices, dtype=torch.long, device=state.device)]).cpu())
            for label, indices in bundle.targets.items()
        }
        result["feasible_probability"] = float(torch.sum(probability[inst.feasible_mask]).cpu())
        result["state_norm"] = float(torch.sum(probability).cpu())
        result["expected_objective"] = float(torch.sum(probability * inst.Cobj).cpu())
        result["expected_conflict"] = float(torch.sum(probability * inst.Cconf).cpu())
    return result


def feasible_diversity(
    state: torch.Tensor,
    inst: HeteroBlockWDSHCCInstance,
) -> dict[str, float]:
    with torch.no_grad():
        probabilities = (torch.abs(state) ** 2)[inst.feasible_mask]
        mass = float(torch.sum(probabilities).cpu())
        if mass <= 0:
            return {
                "feasible_mass": mass,
                "feasible_cond_entropy": 0.0,
                "feasible_effective_support": 0.0,
                "feasible_ipr_effective_support": 0.0,
            }
        conditional = probabilities / mass
        positive = conditional > 0
        entropy = float((-torch.sum(conditional[positive] * torch.log(conditional[positive]))).cpu())
        inverse_ipr = float((1.0 / torch.sum(conditional * conditional)).cpu())
        return {
            "feasible_mass": mass,
            "feasible_cond_entropy": entropy,
            "feasible_effective_support": float(math.exp(entropy)),
            "feasible_ipr_effective_support": inverse_ipr,
        }


def resources(
    *,
    algorithm: str,
    n: int,
    layout: Sequence[int],
    conflict_edges: int,
    m: int,
    objective_degree: int = 4,
    p1: int | None = None,
    depth: int | None = None,
) -> dict[str, int]:
    c = int(conflict_edges)
    x = xy_pair_cost(layout)
    degree = int(objective_degree)
    if degree < 1:
        raise ValueError("objective degree must be positive")
    objective_ru_per_term = 1 if degree == 2 else 4 * degree - 2
    o = objective_ru_per_term * int(m)
    s = int(n * n)
    stage1_depth = int(p1 or 0)
    p1_cost = len(layout) + stage1_depth * (c + x)
    p = int(depth or 0)
    if algorithm == "XY-Stage1-Only":
        circuit = p1_cost
    elif algorithm == "XY-LP-QAOA":
        circuit = p1_cost + p * (o + 2 * p1_cost + s)
    elif algorithm in {"Std-XY", "d-XY"}:
        circuit = len(layout) + p * (c + o + x)
    elif algorithm == "Warm-XY":
        circuit = p1_cost + p * (c + o + x)
    else:
        raise ValueError(f"no RU rule for {algorithm!r}")
    return {
        "C": c,
        "X": x,
        "m": int(m),
        "objective_degree": degree,
        "objective_RU_per_term": objective_ru_per_term,
        "O": o,
        "S": s,
        "P1": p1_cost,
        "circuit_RU": int(circuit),
        "terminal_RU": int(circuit + o),
    }


def target_costs(
    replays: Mapping[str, float],
    terminal_ru: int,
) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for label in ("target_a_sqrt", "target_b_k2", "exact_ground", "manuscript_fraction"):
        repetitions = rts_99(float(replays[label]))
        out[f"{label}_RTS99"] = repetitions
        out[f"{label}_cost"] = None if repetitions is None else int(repetitions * terminal_ru)
    return out


def state_checkpoint(
    path: Path | str,
    state: torch.Tensor,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = state.detach().cpu().numpy()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, destination)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    manifest = {
        **dict(metadata),
        "path": str(destination),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_file(destination),
    }
    atomic_write_json(sidecar, manifest)
    return manifest


def environment_manifest(device: torch.device) -> dict[str, Any]:
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = int(torch.cuda.max_memory_allocated(device))
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": cuda_name,
        "peak_cuda_memory_bytes": cuda_memory,
    }


def load_stage1_checkpoint(
    runner: HeteroQAOARunner,
    checkpoint_json: Path | str,
    *,
    trial_index: int = 0,
) -> tuple[int, torch.Tensor, dict[str, Any]]:
    with open(checkpoint_json, encoding="utf-8") as handle:
        data = json.load(handle)
    p1 = int(data["p1"])
    gammas = torch.tensor(data["gammas"], dtype=runner.real_dtype, device=runner.device)
    betas = torch.tensor(data["betas"], dtype=runner.real_dtype, device=runner.device)
    with torch.no_grad():
        energies, states = runner.energy_xy(runner.inst.Cconf, gammas[None, :], betas[None, :], p1)
        state = states[0]
        state /= torch.linalg.vector_norm(state)
        energy = float(energies[0].cpu())
    if runner.complex_dtype == torch.complex64:
        relative_tolerance, absolute_tolerance = 2e-5, 2e-6
    else:
        relative_tolerance, absolute_tolerance = 2e-8, 2e-9
    if not math.isclose(
        energy,
        float(data["conflict_energy"]),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(
            f"stage-1 checkpoint reconstruction mismatch: reconstructed={energy}, saved={data['conflict_energy']}"
        )
    info = {
        "best_energy": energy,
        "selected_restart": int(data.get("selected_restart", -1)),
        "restart_energies": list(data.get("restart_energies", [])),
        "gammas": gammas,
        "betas": betas,
        "stage1_init_source": f"frozen_checkpoint:{Path(checkpoint_json).name}",
        "stage1_state_reused": True,
    }
    runner._stage1_cache[(trial_index, p1)] = (state, int(data.get("iterations", 0)), info)
    runner._stage1_param_cache[p1] = (gammas, betas)
    return p1, state, data


def reconstruct_state_from_angles(
    runner: HeteroQAOARunner,
    *,
    algorithm: str,
    depth: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    stage1_state: torch.Tensor | None = None,
) -> torch.Tensor:
    g = torch.tensor(gammas, dtype=runner.real_dtype, device=runner.device)[None, :]
    b = torch.tensor(betas, dtype=runner.real_dtype, device=runner.device)[None, :]
    with torch.no_grad():
        if algorithm == "XY-LP-QAOA":
            if stage1_state is None:
                raise ValueError("LP reconstruction requires the Stage-1 state")
            _energy, states = runner.energy_history(stage1_state, runner.inst.Cobj, g, b, depth)
        elif algorithm == "Std-XY":
            h = runner.penalty * runner.inst.Cconf + runner.inst.Cobj
            _energy, states = runner.energy_xy(h, g, b, depth)
        elif algorithm == "d-XY":
            h_pen = runner.penalty * runner.inst.Cconf
            _energy, states = runner.energy_xy(
                h_pen + runner.inst.Cobj,
                g,
                b,
                depth,
                split=(h_pen, runner.inst.Cobj),
            )
        elif algorithm == "Warm-XY":
            if stage1_state is None:
                raise ValueError("warm reconstruction requires the Stage-1 state")
            h = runner.penalty * runner.inst.Cconf + runner.inst.Cobj
            _energy, states = runner.energy_xy(h, g, b, depth, initial_state=stage1_state)
        else:
            raise ValueError(f"unsupported reconstruction algorithm {algorithm!r}")
        state = states[0]
        state /= torch.linalg.vector_norm(state)
    return state


def elapsed_timer() -> tuple[float, callable]:
    started = time.perf_counter()
    return started, lambda: time.perf_counter() - started
