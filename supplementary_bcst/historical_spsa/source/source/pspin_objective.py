from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import campaign_core as core


OBJECTIVE_MODEL = "block_state_p3"
INTERACTION_ORDER_BLOCKS = 3
POLYNOMIAL_DEGREE = 6
BLOCK_COUNT = 5
BLOCK_WEIGHT = 2
MIN_SCALING_N = 26
MAX_SCALING_N = 32
REFERENCE_POOL_COUNT = 9450
REFERENCE_TERM_COUNT = 6150
TERM_DENSITY_NUMERATOR = 41
TERM_DENSITY_DENOMINATOR = 63
RNG_ALGORITHM = "sha256-domain-separated-seed-pcg64dxsm-v1"
RNG_PREFIX = "bcst-block-state-pspin-v1"


def _domain_seed(instance_seed: int, domain: str) -> int:
    digest = hashlib.sha256(
        (
            f"{RNG_PREFIX}|instance-seed={instance_seed}|domain={domain}"
        ).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _rng(instance_seed: int, domain: str) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64DXSM(_domain_seed(instance_seed, domain))
    )


def layout_for_n(n: int) -> tuple[int, ...]:
    """Return the balanced five-block continuation of the N=30 layout."""
    n = int(n)
    if n < MIN_SCALING_N or n > MAX_SCALING_N:
        raise ValueError(
            f"the frozen size-scaling campaign supports "
            f"{MIN_SCALING_N} <= N <= {MAX_SCALING_N}, got N={n}"
        )
    quotient, remainder = divmod(n, BLOCK_COUNT)
    layout = tuple(
        [quotient + 1] * remainder
        + [quotient] * (BLOCK_COUNT - remainder)
    )
    if len(layout) != BLOCK_COUNT or sum(layout) != n:
        raise AssertionError(f"invalid balanced layout for N={n}: {layout}")
    if min(layout) <= BLOCK_WEIGHT:
        raise AssertionError(
            f"block size must exceed block weight {BLOCK_WEIGHT}: {layout}"
        )
    return layout


def term_count_for_pool(pool_count: int) -> int:
    """Preserve the exact N=30 support density, 6150/9450 = 41/63."""
    pool_count = int(pool_count)
    selected = (
        TERM_DENSITY_NUMERATOR * pool_count
        + TERM_DENSITY_DENOMINATOR // 2
    ) // TERM_DENSITY_DENOMINATOR
    if selected <= 0 or selected > pool_count:
        raise AssertionError(
            f"invalid selected support count {selected}/{pool_count}"
        )
    return selected


def _mask_to_term(mask: int, n: int) -> tuple[int, ...]:
    return tuple(
        qubit for qubit in range(int(n)) if (int(mask) >> qubit) & 1
    )


def eligible_support_pool(inst) -> list[tuple[int, ...]]:
    expected_layout = layout_for_n(int(inst.N))
    if tuple(inst.block_sizes) != expected_layout:
        raise ValueError(
            f"expected balanced five-block layout {expected_layout} for "
            f"N={inst.N}, got {tuple(inst.block_sizes)}"
        )
    masks = np.asarray(inst.feasible_state_bits_np, dtype=np.uint64)
    support_masks: set[int] = set()
    block_masks = tuple(
        ((1 << int(size)) - 1) << int(offset)
        for offset, size in zip(inst.offsets, inst.block_sizes)
    )
    for blocks in itertools.combinations(
        range(BLOCK_COUNT), INTERACTION_ORDER_BLOCKS
    ):
        block_mask = sum(
            block_masks[block] for block in blocks
        )
        support_masks.update(
            int(mask) & block_mask for mask in masks
        )
    if int(inst.N) == 30 and len(support_masks) != REFERENCE_POOL_COUNT:
        raise AssertionError(
            f"expected {REFERENCE_POOL_COUNT:,} eligible degree-6 supports "
            f"at N=30, "
            f"found {len(support_masks):,}"
        )
    terms = [
        _mask_to_term(mask, int(inst.N)) for mask in sorted(support_masks)
    ]
    if any(len(term) != POLYNOMIAL_DEGREE for term in terms):
        raise AssertionError("eligible support has the wrong degree")
    core.verify_support_activity(inst, terms)
    return terms


def build_bundle(
    inst,
    *,
    instance_seed: int,
    optimizer_domain: str,
) -> core.ObjectiveBundle:
    pool = eligible_support_pool(inst)
    term_count = term_count_for_pool(len(pool))
    if int(inst.N) == 30 and term_count != REFERENCE_TERM_COUNT:
        raise AssertionError(
            f"N=30 identity requires {REFERENCE_TERM_COUNT} terms, "
            f"computed {term_count}"
        )
    support_rng = _rng(
        instance_seed,
        f"order-{INTERACTION_ORDER_BLOCKS}-support-order",
    )
    permutation = support_rng.permutation(len(pool))
    order = [pool[int(index)] for index in permutation]
    weight_rng = _rng(
        instance_seed,
        f"order-{INTERACTION_ORDER_BLOCKS}-weights",
    )
    all_weights = weight_rng.uniform(
        -0.5, 0.5, size=len(order)
    ).astype(np.float64)
    terms = order[:term_count]
    weights = all_weights[:term_count].copy()

    values, _tensor = core.install_objective(inst, terms, weights)
    feasible_energies = values[inst.feasible_mask_np]
    targets, target_manifest = core.build_targets(
        inst, feasible_energies
    )
    primary_mask = np.zeros(inst.dim, dtype=bool)
    primary_mask[targets["target_a_sqrt"]] = True
    inst.opt_mask_np = primary_mask
    inst.opt_mask = torch.tensor(
        primary_mask,
        dtype=torch.bool,
        device=inst.device,
    )
    inst.min_feasible_obj = float(np.min(feasible_energies))

    magnitude_quantiles = 2.0 * np.abs(all_weights)
    sign_bits = (all_weights >= 0.0).astype(np.uint8)
    labels = ["222"] * len(order)
    raw_selected_masks = np.asarray(
        [int(core.term_mask(term)) for term in terms],
        dtype="<u8",
    )
    raw_selected_weights = weights.astype("<f8", copy=False)
    optimizer_seed = core.separated_streams(
        inst.N,
        instance_seed,
        optimizer_domain,
    ).optimizer_seed
    streams = core.StreamSeeds(
        support_seed=_domain_seed(
            instance_seed,
            f"order-{INTERACTION_ORDER_BLOCKS}-support-order",
        ),
        magnitude_seed=_domain_seed(
            instance_seed,
            f"order-{INTERACTION_ORDER_BLOCKS}-weights",
        ),
        sign_seed=_domain_seed(
            instance_seed,
            f"order-{INTERACTION_ORDER_BLOCKS}-weights",
        ),
        optimizer_seed=optimizer_seed,
    )
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
    manifest: dict[str, Any] = {
        "schema": "bcst-signed-continuous-block-state-pspin-objective-v1",
        "campaign": core.CAMPAIGN,
        "N": int(inst.N),
        "layout": list(inst.block_sizes),
        "instance_seed": int(instance_seed),
        "objective_model": OBJECTIVE_MODEL,
        "interaction_order_in_blocks": INTERACTION_ORDER_BLOCKS,
        "polynomial_degree": POLYNOMIAL_DEGREE,
        "term_count": term_count,
        "term_density": (
            TERM_DENSITY_NUMERATOR / TERM_DENSITY_DENOMINATOR
        ),
        "term_density_fraction": (
            f"{TERM_DENSITY_NUMERATOR}/{TERM_DENSITY_DENOMINATOR}"
        ),
        "weight_family": "signed_cont_uniform",
        "range_a": 0.5,
        "rng_algorithm": RNG_ALGORITHM,
        "rng_domain_prefix": RNG_PREFIX,
        "stream_seeds": {
            "support_seed": streams.support_seed,
            "magnitude_seed": streams.magnitude_seed,
            "sign_seed": streams.sign_seed,
            "optimizer_seed": streams.optimizer_seed,
        },
        "support_algorithm": (
            "all-conflict-feasible-active-three-block-state-indicators-"
            "sorted-then-frozen-shuffle-v1"
        ),
        "eligible_support_pool_count": len(order),
        "selected_term_count": term_count,
        "support_sha256": core.support_hash(terms),
        "full_support_order_sha256": core.support_hash(order),
        "objective_sha256": core.objective_hash(terms, weights),
        "scout_raw_support_u64_sha256": hashlib.sha256(
            raw_selected_masks.tobytes()
        ).hexdigest(),
        "scout_raw_weight_f64_sha256": hashlib.sha256(
            raw_selected_weights.tobytes()
        ).hexdigest(),
        "weight_empirical": empirical,
        "objective_sign_convention": (
            "energy equals minus saved coefficient times active monomial"
        ),
        "target_manifest_sha256": core.sha256_bytes(
            core.canonical_json_bytes(target_manifest)
        ),
        "selection_provenance": (
            "lowest polynomial degree passing the predeclared method-neutral "
            "landscape rule; no QAOA outcome used"
        ),
    }
    return core.ObjectiveBundle(
        terms=terms,
        weights=weights,
        support_order=order,
        support_compositions=labels,
        magnitude_quantiles=magnitude_quantiles,
        sign_bits=sign_bits,
        seeds=streams,
        manifest=manifest,
        targets=targets,
        target_manifest=target_manifest,
    )


def save_bundle(
    directory: Path | str,
    bundle: core.ObjectiveBundle,
) -> dict[str, str]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    support_rows = []
    for index, term in enumerate(bundle.support_order):
        row: dict[str, Any] = {
            "order_index": index,
            "composition": "222",
            "selected": int(index < len(bundle.terms)),
            "magnitude_quantile_hex": float(
                bundle.magnitude_quantiles[index]
            ).hex(),
            "sign_bit": int(bundle.sign_bits[index]),
        }
        row.update(
            {f"q{position}": int(qubit) for position, qubit in enumerate(term)}
        )
        support_rows.append(row)
    objective_rows = []
    for index, (term, weight) in enumerate(
        zip(bundle.terms, bundle.weights)
    ):
        row = {
            "term_index": index,
            "composition": "222",
            "weight": format(float(weight), ".17g"),
            "weight_hex": float(weight).hex(),
        }
        row.update(
            {f"q{position}": int(qubit) for position, qubit in enumerate(term)}
        )
        objective_rows.append(row)
    core.write_csv_lossless(root / "support_order.csv", support_rows)
    core.write_csv_lossless(root / "objective_terms.csv", objective_rows)
    core.atomic_write_json(
        root / "objective_manifest.json", bundle.manifest
    )
    core.atomic_write_json(
        root / "target_manifest.json", bundle.target_manifest
    )
    outputs = {}
    for name in (
        "support_order.csv",
        "objective_terms.csv",
        "objective_manifest.json",
        "target_manifest.json",
    ):
        outputs[name] = core.sha256_file(root / name)
    core.atomic_write_json(root / "file_hashes.json", outputs)
    return outputs
