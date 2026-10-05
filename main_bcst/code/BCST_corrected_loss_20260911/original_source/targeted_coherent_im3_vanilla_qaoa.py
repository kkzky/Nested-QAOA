#!/usr/bin/env python3
"""Exact 18-qubit vanilla-QAOA diagnostic for the coherent-IM3 family.

The production 30-qubit unconstrained differentiable statevector is rejected
by an explicit memory lower bound.  The preregistered diagnostic keeps six
labels and the exact coherent objective, but uses the induced three-site path,
a balanced nonuniform quota, and a fixed exact penalty Lambda=377.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np
import torch

import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_seed10_extension as extension
import targeted_three_stage_common as common


SCHEMA = "targeted-coherent-im3-vanilla-qaoa-b3l6-nonredundant-v2"
IMPLEMENTATION_REVISION = "induced-path-balanced-quota-penalty377-r3-20260819"
MANIFEST_NAME = "targeted_coherent_im3_vanilla_qaoa_manifest.json"
PLAN_NAME = "targeted_coherent_im3_vanilla_qaoa_two_pod_plan.json"
RESULT_DIRECTORY = "targeted_coherent_im3_vanilla_qaoa_results"
GLOBAL_SEAL_NAME = "targeted_coherent_im3_vanilla_qaoa_global_seal.json"
QUEUE_SOURCE_NAME = "targeted_coherent_im3_vanilla_qaoa_queue.py"
SCORER_SOURCE_NAME = "targeted_coherent_im3_vanilla_qaoa_score.py"
PROTOCOL_SOURCE_NAME = extension.PROTOCOL_SOURCE_NAME

BLOCKS = 3
LABELS = 6
QUBITS = BLOCKS * LABELS
DIMENSION = 1 << QUBITS
QUOTA = (2, 1, 1, 1, 1, 0)
EDGES = ((0, 1), (1, 2))
WITNESS_PAIRS = ((0, 2), (1, 3), (0, 4))
PENALTY_LAMBDA = 377
PHASE_AND_LOSS_SCALE = 377.0

TABLE_SEEDS = tuple(extension.ALL_TABLE_SEEDS)
RESTART_SEEDS = tuple(extension.RESTART_SEEDS)
DEPTHS = (0, 1, 2, 4, 8, 16, 32, 64)
POSITIVE_DEPTHS = DEPTHS[1:]
OBJECTIVE_EVALUATIONS = 2_400
LEARNING_RATE = 0.035
INITIALIZATION_STREAM = 812_096

OBJECTIVE_SUPPORTS = {"k1": 18, "k3": 72, "k4": 24, "k5": 84}
HAMILTONIAN_SUPPORTS = {"k1": 18, "k2": 63, "k3": 72, "k4": 24, "k5": 84}
OBJECTIVE_RU = 2_604
HAMILTONIAN_PHASE_RU = 2_667
PREP_RU = 18
MIXER_RU = 18


def table_id(seed: int) -> str:
    common._require(seed in TABLE_SEEDS, f"unknown vanilla table seed {seed}")
    return f"pcg64_{int(seed)}"


def method_id(depth: int) -> str:
    common._require(depth in DEPTHS, f"unknown vanilla depth {depth}")
    return f"coherent_im3_vanilla_qaoa_B3L6__p{int(depth)}"


def resource(depth: int) -> int:
    common._require(depth in DEPTHS, "invalid vanilla resource depth")
    return int(PREP_RU + depth * (HAMILTONIAN_PHASE_RU + MIXER_RU) + OBJECTIVE_RU)


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _plan_path(root: Path) -> Path:
    return root / PLAN_NAME


def result_path(root: Path, seed: int, depth: int) -> Path:
    common._require(seed in TABLE_SEEDS and depth in POSITIVE_DEPTHS, "bad result cell")
    return root / RESULT_DIRECTORY / table_id(seed) / f"{method_id(depth)}.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_source_sha256() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = (
        Path(__file__).name,
        QUEUE_SOURCE_NAME,
        SCORER_SOURCE_NAME,
        PROTOCOL_SOURCE_NAME,
        "targeted_coherent_im3_seed10_extension.py",
        "targeted_coherent_im3_objective.py",
        "targeted_three_stage_common.py",
    )
    paths = tuple(source_root / name for name in names)
    for path in paths:
        common._require(path.is_file(), f"missing vanilla frozen source {path}")
    return {path.name: _file_sha256(path) for path in paths}


_ARRAY_CACHE: dict[int, dict[str, np.ndarray]] = {}


def structural_arrays() -> dict[str, np.ndarray]:
    if 0 in _ARRAY_CACHE:
        return _ARRAY_CACHE[0]
    indices = np.arange(DIMENSION, dtype=np.uint32)
    bits = (
        (indices[:, None] >> np.arange(QUBITS, dtype=np.uint32)[None, :]) & 1
    ).astype(np.int8).reshape(DIMENSION, BLOCKS, LABELS)
    site_counts = np.sum(bits, axis=2, dtype=np.int16)
    label_counts = np.sum(bits, axis=1, dtype=np.int16)
    site_penalty = np.sum((site_counts - 2) ** 2, axis=1, dtype=np.int32)
    quota_penalty = np.sum(
        (label_counts - np.asarray(QUOTA, dtype=np.int16)[None, :]) ** 2,
        axis=1,
        dtype=np.int32,
    )
    conflict_penalty = np.zeros(DIMENSION, dtype=np.int32)
    for left, right in EDGES:
        conflict_penalty += np.sum(
            bits[:, left, :] * bits[:, right, :], axis=1, dtype=np.int32
        )
    penalty = site_penalty + quota_penalty + conflict_penalty
    payload = {
        "bits": bits,
        "site_penalty": site_penalty,
        "quota_penalty": quota_penalty,
        "conflict_penalty": conflict_penalty,
        "penalty": penalty,
        "feasible_mask": penalty == 0,
    }
    common._require(
        int(np.sum(site_penalty == 0)) == 3_375
        and int(np.sum(quota_penalty == 0)) == 243
        and int(np.sum((site_penalty == 0) & (quota_penalty == 0))) == 36
        and int(np.sum((site_penalty == 0) & (conflict_penalty == 0))) == 540
        and int(np.sum((quota_penalty == 0) & (conflict_penalty == 0))) == 81
        and int(np.sum(payload["feasible_mask"])) == 12
        and int(np.min(penalty[~payload["feasible_mask"]])) >= 1,
        "B3 structural support changed",
    )
    _ARRAY_CACHE[0] = payload
    return payload


def objective_arrays(table_row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = structural_arrays()["bits"].astype(np.int64, copy=False)
    service_cost = coherent.validated_coefficient_table(dict(table_row))[:BLOCKS, :]
    service = np.einsum("sbc,bc->s", bits, service_cost, dtype=np.int64)
    risk = np.zeros(DIMENSION, dtype=np.int64)
    for victim in range(BLOCKS):
        for product, generators in enumerate(coherent.PAIR_LISTS):
            load = np.zeros(DIMENSION, dtype=np.int64)
            for source in range(BLOCKS):
                if source == victim:
                    continue
                for left, right in generators:
                    load += bits[:, source, left] * bits[:, source, right]
            risk += bits[:, victim, product] * load * load
    objective = 256 * risk + service
    return objective, risk, service


def _witness_index() -> int:
    answer = 0
    for block, pair in enumerate(WITNESS_PAIRS):
        for label in pair:
            answer |= 1 << (block * LABELS + label)
    return answer


def exact_penalty_evidence(table_row: Mapping[str, object]) -> dict[str, object]:
    arrays = structural_arrays()
    objective, risk, service = objective_arrays(table_row)
    index = _witness_index()
    common._require(
        int(arrays["penalty"][index]) == 0
        and int(risk[index]) == 1
        and int(service[index]) <= 120
        and int(objective[index]) == 256 + int(service[index])
        and int(objective[index]) <= 376
        and int(np.min(objective)) >= 0,
        "fixed exact-penalty witness proof failed",
    )
    hamiltonian = objective + PENALTY_LAMBDA * arrays["penalty"].astype(np.int64)
    infeasible_min = int(np.min(hamiltonian[~arrays["feasible_mask"]]))
    common._require(
        infeasible_min >= PENALTY_LAMBDA > int(objective[index]),
        "Lambda=377 no longer separates infeasible states from a feasible witness",
    )
    return {
        "lambda": PENALTY_LAMBDA,
        "witness_pairs": [list(pair) for pair in WITNESS_PAIRS],
        "witness_basis_index": index,
        "witness_penalty": int(arrays["penalty"][index]),
        "witness_risk": int(risk[index]),
        "witness_service_and_objective": int(objective[index]),
        "infeasible_H_lower_bound": PENALTY_LAMBDA,
        "actual_minimum_infeasible_H": infeasible_min,
        "actual_minimum_used_for_proof_audit_only_not_target_selection": True,
        "proof": "P is nonnegative integer; infeasible P>=1 and O>=0; witness P=0,risk=1,O<=256+120=376",
    }


def _add_term(poly: defaultdict[frozenset[int], int], coefficient: int, *variables: int) -> None:
    poly[frozenset(int(value) for value in variables)] += int(coefficient)


def symbolic_support_audit() -> dict[str, object]:
    objective: defaultdict[frozenset[int], int] = defaultdict(int)
    for block in range(BLOCKS):
        for label in range(LABELS):
            _add_term(objective, 1, block * LABELS + label)
    for victim in range(BLOCKS):
        for product, generators in enumerate(coherent.PAIR_LISTS):
            monomials = [
                (source * LABELS + left, source * LABELS + right)
                for source in range(BLOCKS)
                if source != victim
                for left, right in generators
            ]
            victim_bit = victim * LABELS + product
            for left_index, left in enumerate(monomials):
                for right_index, right in enumerate(monomials):
                    _add_term(
                        objective,
                        256,
                        victim_bit,
                        *left,
                        *right,
                    )
    objective = defaultdict(int, {key: value for key, value in objective.items() if value})
    objective_counts = {
        f"k{degree}": sum(len(key) == degree for key in objective)
        for degree in range(1, 6)
        if any(len(key) == degree for key in objective)
    }

    hamiltonian = defaultdict(int, objective)
    # (sum x - target)^2 = sum x + 2 sum_{i<j}x_ix_j -2t sum x + t^2.
    for block in range(BLOCKS):
        variables = [block * LABELS + label for label in range(LABELS)]
        _add_term(hamiltonian, PENALTY_LAMBDA * (1 - 4), *variables[:1])
        for variable in variables[1:]:
            _add_term(hamiltonian, PENALTY_LAMBDA * (1 - 4), variable)
        for i, left in enumerate(variables):
            for right in variables[i + 1 :]:
                _add_term(hamiltonian, 2 * PENALTY_LAMBDA, left, right)
    for label in range(LABELS):
        variables = [block * LABELS + label for block in range(BLOCKS)]
        target = int(QUOTA[label])
        for variable in variables:
            _add_term(
                hamiltonian,
                PENALTY_LAMBDA * (1 - 2 * target),
                variable,
            )
        for i, left in enumerate(variables):
            for right in variables[i + 1 :]:
                _add_term(hamiltonian, 2 * PENALTY_LAMBDA, left, right)
    for left, right in EDGES:
        for label in range(LABELS):
            _add_term(
                hamiltonian,
                PENALTY_LAMBDA,
                left * LABELS + label,
                right * LABELS + label,
            )
    hamiltonian = defaultdict(
        int, {key: value for key, value in hamiltonian.items() if key and value}
    )
    h_counts = {
        f"k{degree}": sum(len(key) == degree for key in hamiltonian)
        for degree in range(1, 6)
        if any(len(key) == degree for key in hamiltonian)
    }
    common._require(
        objective_counts == OBJECTIVE_SUPPORTS
        and h_counts == HAMILTONIAN_SUPPORTS,
        f"symbolic support counts changed: {objective_counts}, {h_counts}",
    )
    weights = {1: 2, 2: 1, 3: 10, 4: 14, 5: 18}
    objective_ru = sum(count * weights[int(key[1:])] for key, count in objective_counts.items())
    h_ru = sum(count * weights[int(key[1:])] for key, count in h_counts.items())
    common._require(
        objective_ru == OBJECTIVE_RU and h_ru == HAMILTONIAN_PHASE_RU,
        "vanilla RU ledger changed",
    )
    return {
        "objective_supports": objective_counts,
        "hamiltonian_union_supports": h_counts,
        "uniform_diagonal_RU_by_locality": {f"k{k}": value for k, value in weights.items()},
        "objective_terminal_RU": objective_ru,
        "hamiltonian_phase_RU": h_ru,
        "preparation_RU": PREP_RU,
        "transverse_X_mixer_RU": MIXER_RU,
        "total_RU_formula": "2622 + 2685*p",
    }


def symbolic_runtime_diagonal_equality(
    table_row: Mapping[str, object]
) -> dict[str, object]:
    """Expand H independently and compare every one of its 2^18 values."""

    service_cost = coherent.validated_coefficient_table(dict(table_row))[:BLOCKS, :]
    polynomial: defaultdict[frozenset[int], int] = defaultdict(int)
    for block in range(BLOCKS):
        for label in range(LABELS):
            _add_term(
                polynomial,
                int(service_cost[block, label]),
                block * LABELS + label,
            )
    for victim in range(BLOCKS):
        for product, generators in enumerate(coherent.PAIR_LISTS):
            monomials = [
                (source * LABELS + left, source * LABELS + right)
                for source in range(BLOCKS)
                if source != victim
                for left, right in generators
            ]
            victim_bit = victim * LABELS + product
            for left in monomials:
                for right in monomials:
                    _add_term(polynomial, 256, victim_bit, *left, *right)

    # Site cardinalities, including constants.
    for block in range(BLOCKS):
        target = 2
        variables = [block * LABELS + label for label in range(LABELS)]
        _add_term(polynomial, PENALTY_LAMBDA * target * target)
        for variable in variables:
            _add_term(
                polynomial, PENALTY_LAMBDA * (1 - 2 * target), variable
            )
        for i, left in enumerate(variables):
            for right in variables[i + 1 :]:
                _add_term(polynomial, 2 * PENALTY_LAMBDA, left, right)
    # Global quota, including constants.
    for label in range(LABELS):
        target = int(QUOTA[label])
        variables = [block * LABELS + label for block in range(BLOCKS)]
        _add_term(polynomial, PENALTY_LAMBDA * target * target)
        for variable in variables:
            _add_term(
                polynomial, PENALTY_LAMBDA * (1 - 2 * target), variable
            )
        for i, left in enumerate(variables):
            for right in variables[i + 1 :]:
                _add_term(polynomial, 2 * PENALTY_LAMBDA, left, right)
    for left, right in EDGES:
        for label in range(LABELS):
            _add_term(
                polynomial,
                PENALTY_LAMBDA,
                left * LABELS + label,
                right * LABELS + label,
            )
    polynomial = defaultdict(int, {key: value for key, value in polynomial.items() if value})

    flat_bits = structural_arrays()["bits"].reshape(DIMENSION, QUBITS).astype(
        np.int64, copy=False
    )
    symbolic = np.full(DIMENSION, polynomial.get(frozenset(), 0), dtype=np.int64)
    for support, coefficient in polynomial.items():
        if not support:
            continue
        variables = np.fromiter(sorted(support), dtype=np.int64)
        symbolic += int(coefficient) * np.prod(flat_bits[:, variables], axis=1)
    objective, _, _ = objective_arrays(table_row)
    runtime = objective + PENALTY_LAMBDA * structural_arrays()["penalty"].astype(np.int64)
    common._require(np.array_equal(symbolic, runtime), "symbolic/runtime H mismatch")
    return {
        "service_cost_compact_json_sha256": table_row[
            "service_cost_compact_json_sha256"
        ],
        "basis_state_count": DIMENSION,
        "polynomial_term_count_including_constant": len(polynomial),
        "exact_all_basis_states_equal": True,
        "raw_H_int64_sha256": hashlib.sha256(runtime.tobytes()).hexdigest(),
    }


def full30_memory_certificate() -> dict[str, object]:
    dimension = 1 << 30
    state_bytes = dimension * 8
    restart_state_bytes = len(RESTART_SEEDS) * state_bytes
    diagonal_bytes = dimension * 4
    p1_differentiable_lower_bound = diagonal_bytes + 3 * restart_state_bytes
    transverse_x_layer_work_ratio_vs_n18 = (30 * (1 << 30)) / (18 * (1 << 18))
    common._require(
        state_bytes == 8 * 1024**3
        and restart_state_bytes == 32 * 1024**3
        and diagonal_bytes == 4 * 1024**3
        and p1_differentiable_lower_bound == 100 * 1024**3,
        "full-30 memory arithmetic changed",
    )
    return {
        "full_problem_qubits": 30,
        "full_dimension": dimension,
        "one_complex64_state_bytes": state_bytes,
        "four_restart_state_batch_bytes": restart_state_bytes,
        "one_float32_cost_diagonal_bytes": diagonal_bytes,
        "p1_noncheckpointed_autograd_lower_bound_bytes": p1_differentiable_lower_bound,
        "A800_nominal_bytes": 80 * 1024**3,
        "transverse_X_layer_work_ratio_N30_over_N18": transverse_x_layer_work_ratio_vs_n18,
        "work_ratio_semantics": (
            "exact O(N*2^N) amplitude-touch ratio, not a hardware benchmark or exact wall-time ratio"
        ),
        "decision": "FULL30_FOUR_RESTART_BATCHED_NONCHECKPOINTED_PYTORCH_INADMISSIBLE",
        "scope_note": (
            "this is not a proof that every exact simulator is memory-infeasible; "
            "sequential restarts or specialized checkpointing can reduce memory but "
            "remain computationally prohibitive for the frozen 10x7x4 campaign"
        ),
        "downsizing_rule": (
            "use the preregistered tractable analogue B=3,L=6,N=18, preserving six "
            "labels, the exact coherent objective, and nontrivial global quota/conflict "
            "constraints while completing all 10x7x4 optimized cells"
        ),
    }


def freeze(root: Path, extension_root: Path) -> dict[str, object]:
    root = root.resolve()
    extension_manifest, extension_plan = extension.validated_plan(extension_root.resolve())
    tables = [coherent.coefficient_table_from_pcg64(seed) for seed in TABLE_SEEDS]
    penalty_evidence = [exact_penalty_evidence(row) for row in tables]
    support = symbolic_support_audit()
    diagonal_equalities = [symbolic_runtime_diagonal_equality(row) for row in tables]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-manifest",
        "status": "FROZEN_TARGET_BLIND_VANILLA_QAOA_DIAGNOSTIC",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "frozen_source_sha256": frozen_source_sha256(),
        "source_extension_manifest_sha256": extension_manifest["manifest_sha256"],
        "source_extension_plan_sha256": extension_plan["plan_sha256"],
        "full30_memory_certificate": full30_memory_certificate(),
        "diagnostic_problem": {
            "blocks": BLOCKS,
            "labels": LABELS,
            "qubits": QUBITS,
            "dimension": DIMENSION,
            "site_constraint": "exactly two selected labels per block",
            "quota": list(QUOTA),
            "quota_rule": (
                "canonical closest-to-uniform nonuniform quota; among permutations "
                "of (2,1,1,1,1,0), repeat lowest and omit highest by label-order tie-break"
            ),
            "conflict_edges": [list(edge) for edge in EDGES],
            "conflict_graph_rule": "induced first-three-site path of the original five-cycle",
            "objective": "same coherent IM3 load-square formula restricted to three sites",
            "penalty": "H=O+377*(site_two_hot_penalty+quota_penalty+conflict_penalty)",
            "phase_and_loss_diagonal": "H/377",
            "initial_state": "uniform |+>^18",
            "mixer": "standard transverse sum_j X_j",
            "not_direct_quantitative_evidence_for_30_qubits": True,
        },
        "table_seeds": list(TABLE_SEEDS),
        "tables_in_seed_order": tables,
        "all_ten_tables_unfiltered": True,
        "exact_penalty_evidence_in_seed_order": penalty_evidence,
        "symbolic_support_and_resource_audit": support,
        "symbolic_runtime_diagonal_equality_in_seed_order": diagonal_equalities,
        "depths_in_frozen_order": list(DEPTHS),
        "p0_semantics": "derived uniform |+>^18 anchor",
        "restart_seeds": list(RESTART_SEEDS),
        "optimization": {
            "algorithm": "Adam",
            "learning_rate": LEARNING_RATE,
            "forward_evaluations_per_restart_including_endpoint": OBJECTIVE_EVALUATIONS,
            "gradient_updates_per_restart": OBJECTIVE_EVALUATIONS - 1,
            "independent_at_every_positive_depth": True,
            "continuation_or_angle_transfer": False,
            "checkpoint_rule": "minimum expected H/377 only",
            "gamma_initialization": "uniform[-pi,pi]",
            "beta_initialization": "uniform[-pi/2,pi/2]",
            "prefix_consistent_through_p64": True,
            "dtype": "complex64",
        },
        "resource_by_depth": {str(depth): resource(depth) for depth in DEPTHS},
        "post_completion_target_definition": (
            "all minimum raw-O states among exact P=0 feasible bitstrings"
        ),
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "manifest_sha256": common.sha256_json(body)}
    root.mkdir(parents=True, exist_ok=True)
    (root / RESULT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    destination = _manifest_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace vanilla manifest {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_manifest(root: Path) -> dict[str, object]:
    manifest = common.load_json(_manifest_path(root))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    common._require(
        manifest.get("schema") == f"{SCHEMA}-manifest"
        and manifest.get("status") == "FROZEN_TARGET_BLIND_VANILLA_QAOA_DIAGNOSTIC"
        and manifest.get("implementation_revision") == IMPLEMENTATION_REVISION
        and manifest.get("manifest_sha256") == common.sha256_json(body)
        and manifest.get("frozen_source_sha256") == frozen_source_sha256()
        and tuple(manifest.get("table_seeds", ())) == TABLE_SEEDS
        and manifest.get("tables_in_seed_order")
        == [coherent.coefficient_table_from_pcg64(seed) for seed in TABLE_SEEDS]
        and tuple(manifest.get("depths_in_frozen_order", ())) == DEPTHS
        and tuple(manifest.get("restart_seeds", ())) == RESTART_SEEDS
        and manifest.get("full30_memory_certificate") == full30_memory_certificate()
        and manifest.get("symbolic_support_and_resource_audit") == symbolic_support_audit()
        and manifest.get("target_constructed") is False,
        "invalid vanilla manifest",
    )
    for row, evidence in zip(
        manifest["tables_in_seed_order"], manifest["exact_penalty_evidence_in_seed_order"]
    ):
        common._require(evidence == exact_penalty_evidence(row), "penalty proof changed")
    return manifest


def table(manifest: Mapping[str, object], seed: int) -> dict[str, object]:
    common._require(seed in TABLE_SEEDS, f"unknown seed {seed}")
    row = manifest["tables_in_seed_order"][TABLE_SEEDS.index(seed)]
    common._require(row == coherent.coefficient_table_from_pcg64(seed), "table changed")
    return row


def initial_arrays(depth: int) -> tuple[np.ndarray, np.ndarray]:
    common._require(depth in POSITIVE_DEPTHS, "bad initialization depth")
    gamma_rows, beta_rows = [], []
    for seed in RESTART_SEEDS:
        gamma_rng = np.random.default_rng(
            int(seed) + 1_000_003 * (2 * INITIALIZATION_STREAM)
        )
        beta_rng = np.random.default_rng(
            int(seed) + 1_000_003 * (2 * INITIALIZATION_STREAM + 1)
        )
        gamma_rows.append(gamma_rng.uniform(-math.pi, math.pi, size=max(DEPTHS))[:depth])
        beta_rows.append(beta_rng.uniform(-math.pi / 2, math.pi / 2, size=max(DEPTHS))[:depth])
    return np.asarray(gamma_rows), np.asarray(beta_rows)


def initial_angles_sha256(depth: int) -> str:
    gamma, beta = initial_arrays(depth)
    return common.sha256_json({"gamma_by_restart": gamma.tolist(), "beta_by_restart": beta.tolist()})


def runtime_diagonals(
    table_row: Mapping[str, object], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = structural_arrays()
    objective, _, _ = objective_arrays(table_row)
    penalty = arrays["penalty"].astype(np.int64)
    h_raw = objective + PENALTY_LAMBDA * penalty
    return (
        torch.as_tensor(h_raw / PHASE_AND_LOSS_SCALE, dtype=torch.float32, device=device),
        torch.as_tensor(objective, dtype=torch.float32, device=device),
        torch.as_tensor(penalty, dtype=torch.float32, device=device),
        torch.as_tensor(arrays["feasible_mask"], dtype=torch.bool, device=device),
    )


def apply_x_mixer(state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    batch = state.shape[0]
    cosine = torch.cos(beta).to(state.real.dtype)[:, None, None]
    sine = torch.sin(beta).to(state.real.dtype)[:, None, None]
    current = state
    for qubit in range(QUBITS):
        low = 1 << qubit
        high = DIMENSION // (2 * low)
        shaped = current.reshape(batch, high, 2, low)
        zero, one = shaped[:, :, 0, :], shaped[:, :, 1, :]
        current = torch.stack(
            (cosine * zero - 1j * sine * one, cosine * one - 1j * sine * zero),
            dim=2,
        ).reshape(batch, DIMENSION)
    return current


def energy_fn(
    diagonal: torch.Tensor, depth: int
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    device = diagonal.device
    initial = torch.full(
        (DIMENSION,), complex(1 / math.sqrt(DIMENSION), 0),
        dtype=torch.complex64, device=device,
    )

    def energy(gamma: torch.Tensor, beta: torch.Tensor):
        state = initial[None, :].expand(gamma.shape[0], -1).clone()
        for layer in range(depth):
            state = state * torch.exp(
                -1j * gamma[:, layer : layer + 1] * diagonal[None, :]
            ).to(torch.complex64)
            state = apply_x_mixer(state, beta[:, layer])
            state = state / torch.sqrt(
                torch.sum(torch.abs(state) ** 2, dim=1, keepdim=True).clamp_min(1e-24)
            )
        probability = torch.abs(state) ** 2
        loss = torch.sum(probability * diagonal[None, :], dim=1)
        return loss, state

    return energy


def optimize(
    diagonal: torch.Tensor, depth: int, evaluations: int
) -> dict[str, object]:
    gamma_np, beta_np = initial_arrays(depth)
    gamma = torch.as_tensor(gamma_np, dtype=torch.float32, device=diagonal.device).clone().requires_grad_(True)
    beta = torch.as_tensor(beta_np, dtype=torch.float32, device=diagonal.device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam((gamma, beta), lr=LEARNING_RATE)
    circuit = energy_fn(diagonal, depth)
    best_loss = torch.full((len(RESTART_SEEDS),), math.inf, dtype=torch.float32, device=diagonal.device)
    best_state: torch.Tensor | None = None
    best_gamma, best_beta = torch.zeros_like(gamma), torch.zeros_like(beta)
    best_eval = torch.zeros(len(RESTART_SEEDS), dtype=torch.int64, device=diagonal.device)
    trace: list[dict[str, object]] = []
    started = time.time()

    def inspect(loss: torch.Tensor, state: torch.Tensor, evaluation: int, endpoint: bool) -> None:
        nonlocal best_loss, best_state, best_gamma, best_beta, best_eval
        detached = loss.detach()
        improved = detached < best_loss
        if best_state is None:
            best_state = state.detach().clone()
        elif torch.any(improved):
            best_state[improved] = state.detach()[improved]
        best_loss = torch.where(improved, detached, best_loss)
        best_gamma[improved], best_beta[improved] = gamma.detach()[improved], beta.detach()[improved]
        best_eval[improved] = evaluation
        if evaluation == 1 or endpoint or evaluation % 200 == 0:
            trace.append({
                "evaluation": evaluation,
                "post_update_endpoint": endpoint,
                "loss_by_restart": [float(value) for value in detached.cpu()],
            })

    for evaluation in range(1, evaluations):
        optimizer.zero_grad(set_to_none=True)
        loss, state = circuit(gamma, beta)
        inspect(loss, state, evaluation, False)
        loss.sum().backward()
        if trace and trace[-1]["evaluation"] == evaluation:
            gradient = torch.sqrt(torch.sum(gamma.grad**2 + beta.grad**2, dim=1))
            trace[-1]["gradient_l2_by_restart"] = [float(value) for value in gradient.detach().cpu()]
        optimizer.step()
    with torch.no_grad():
        loss, state = circuit(gamma, beta)
        inspect(loss, state, evaluations, True)
    common._require(best_state is not None, "vanilla optimizer produced no state")
    return {
        "state": best_state,
        "loss": best_loss,
        "gamma": best_gamma,
        "beta": best_beta,
        "best_evaluation": best_eval,
        "trace": trace,
        "elapsed_sec": time.time() - started,
    }


def _metrics(
    state: torch.Tensor,
    objective: torch.Tensor,
    penalty: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> dict[str, list[float]]:
    probability = torch.abs(state) ** 2
    return {
        "state_norm": [float(value) for value in torch.sum(probability, dim=1).cpu()],
        "feasible_mass": [float(value) for value in torch.sum(probability[:, feasible_mask], dim=1).cpu()],
        "expected_raw_objective": [float(value) for value in torch.sum(probability * objective[None, :], dim=1).cpu()],
        "expected_integer_penalty": [float(value) for value in torch.sum(probability * penalty[None, :], dim=1).cpu()],
    }


def validate_result(
    manifest: Mapping[str, object], result: Mapping[str, object], seed: int, depth: int
) -> None:
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    gamma = np.asarray(result.get("gamma_by_restart"), dtype=np.float64)
    beta = np.asarray(result.get("beta_by_restart"), dtype=np.float64)
    trace = result.get("optimizer_trace")
    metrics = result.get("metrics", {})
    expected_trace_evaluations = [
        value
        for value in range(1, OBJECTIVE_EVALUATIONS)
        if value == 1 or value % 200 == 0
    ] + [OBJECTIVE_EVALUATIONS]
    selected = np.asarray(result.get("selected_expected_H_over_377"), dtype=np.float64)
    initial = np.asarray(result.get("initial_expected_H_over_377"), dtype=np.float64)
    final = np.asarray(result.get("final_expected_H_over_377"), dtype=np.float64)
    best_evaluation = np.asarray(
        result.get("best_evaluation_by_restart"), dtype=np.float64
    )
    expected_o = np.asarray(metrics.get("expected_raw_objective"), dtype=np.float64)
    expected_p = np.asarray(metrics.get("expected_integer_penalty"), dtype=np.float64)
    algebra_ok = (
        selected.shape == expected_o.shape == expected_p.shape == (len(RESTART_SEEDS),)
        and np.allclose(
            selected,
            expected_o / PHASE_AND_LOSS_SCALE + expected_p,
            atol=2e-5,
            rtol=2e-5,
        )
    )
    trace_ok = isinstance(trace, list) and len(trace) >= 2
    if trace_ok:
        trace_ok = [row.get("evaluation") for row in trace] == expected_trace_evaluations
    if trace_ok:
        for index, row in enumerate(trace):
            if not isinstance(row, Mapping):
                trace_ok = False
                break
            endpoint = index == len(trace) - 1
            losses = np.asarray(row.get("loss_by_restart"), dtype=np.float64)
            if (
                (row.get("post_update_endpoint") is True) != endpoint
                or losses.shape != (len(RESTART_SEEDS),)
                or not np.all(np.isfinite(losses))
            ):
                trace_ok = False
                break
            if not endpoint:
                gradients = np.asarray(
                    row.get("gradient_l2_by_restart"), dtype=np.float64
                )
                if (
                    gradients.shape != (len(RESTART_SEEDS),)
                    or not np.all(np.isfinite(gradients))
                ):
                    trace_ok = False
                    break
    trace_boundary_ok = (
        trace_ok
        and initial.shape == final.shape == (len(RESTART_SEEDS),)
        and np.all(np.isfinite(initial))
        and np.all(np.isfinite(final))
        and np.allclose(
            initial,
            np.asarray(trace[0].get("loss_by_restart"), dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        and np.allclose(
            final,
            np.asarray(trace[-1].get("loss_by_restart"), dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
    )
    state_norm = np.asarray(metrics.get("state_norm"), dtype=np.float64)
    feasible_mass = np.asarray(metrics.get("feasible_mass"), dtype=np.float64)
    checkpoint_consistency_ok = (
        selected.shape == initial.shape == final.shape == (len(RESTART_SEEDS),)
        and np.all(selected <= initial + 2e-5)
        and np.all(selected <= final + 2e-5)
    )
    common._require(
        result.get("schema") == f"{SCHEMA}-optimizer-result"
        and result.get("status") == "COMPLETED_TARGET_BLIND_VANILLA_QAOA_CELL"
        and result.get("manifest_sha256") == manifest["manifest_sha256"]
        and result.get("table_seed") == seed
        and result.get("table_id") == table_id(seed)
        and result.get("coefficient_table_sha256")
        == table(manifest, seed)["service_cost_compact_json_sha256"]
        and result.get("method_id") == method_id(depth)
        and result.get("depth") == depth
        and tuple(result.get("restart_seeds", ())) == RESTART_SEEDS
        and result.get("initial_angles_sha256") == initial_angles_sha256(depth)
        and result.get("resource_RU") == resource(depth)
        and result.get("forward_evaluations_per_restart") == OBJECTIVE_EVALUATIONS
        and result.get("gradient_updates_per_restart") == OBJECTIVE_EVALUATIONS - 1
        and result.get("endpoint_evaluations_per_restart") == 1
        and best_evaluation.shape == (len(RESTART_SEEDS),)
        and np.all(np.isfinite(best_evaluation))
        and np.all(best_evaluation == np.floor(best_evaluation))
        and np.all((best_evaluation >= 1) & (best_evaluation <= OBJECTIVE_EVALUATIONS))
        and selected.shape == (len(RESTART_SEEDS),)
        and np.all(np.isfinite(selected))
        and checkpoint_consistency_ok
        and gamma.shape == beta.shape == (len(RESTART_SEEDS), depth)
        and np.all(np.isfinite(gamma)) and np.all(np.isfinite(beta))
        and trace_ok
        and trace_boundary_ok
        and result.get("optimizer_trace_sha256") == common.sha256_json(trace)
        and algebra_ok
        and all(
            np.asarray(metrics.get(key), dtype=np.float64).shape == (len(RESTART_SEEDS),)
            and np.all(np.isfinite(np.asarray(metrics.get(key), dtype=np.float64)))
            for key in ("state_norm", "feasible_mass", "expected_raw_objective", "expected_integer_penalty")
        )
        and state_norm.shape == (len(RESTART_SEEDS),)
        and np.allclose(state_norm, 1.0, atol=2e-5, rtol=2e-5)
        and feasible_mass.shape == (len(RESTART_SEEDS),)
        and np.all((feasible_mass >= -2e-5) & (feasible_mass <= 1.0 + 2e-5))
        and result.get("checkpoint_selected_by_target_blind_loss_only") is True
        and result.get("independent_depth_optimization_no_continuation") is True
        and result.get("target_artifact_loaded") is False
        and result.get("target_constructed") is False
        and result.get("target_probability_computed") is False
        and result.get("result_sha256") == common.sha256_json(body),
        f"invalid vanilla result {seed}/p{depth}",
    )


def run_cell(
    root: Path, seed: int, depth: int, *, device: str,
    evaluations: int = OBJECTIVE_EVALUATIONS, production: bool = True,
) -> dict[str, object]:
    manifest = validated_manifest(root)
    common._require(seed in TABLE_SEEDS and depth in POSITIVE_DEPTHS, "invalid cell")
    common._require(evaluations == OBJECTIVE_EVALUATIONS or not production, "budget changed")
    destination = result_path(root, seed, depth)
    if production and destination.exists():
        existing = common.load_json(destination)
        validate_result(manifest, existing, seed, depth)
        return existing
    resolved = torch.device(device)
    common._require(resolved.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    h, objective, penalty, feasible_mask = runtime_diagonals(table(manifest, seed), resolved)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    optimized = optimize(h, depth, evaluations)
    metrics = _metrics(optimized["state"], objective, penalty, feasible_mask)
    trace = optimized["trace"]
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-optimizer-result",
        "status": "COMPLETED_TARGET_BLIND_VANILLA_QAOA_CELL",
        "manifest_sha256": manifest["manifest_sha256"],
        "table_seed": seed,
        "table_id": table_id(seed),
        "coefficient_table_sha256": table(manifest, seed)["service_cost_compact_json_sha256"],
        "method_id": method_id(depth),
        "depth": depth,
        "restart_seeds": list(RESTART_SEEDS),
        "initial_angles_sha256": initial_angles_sha256(depth),
        "resource_RU": resource(depth),
        "forward_evaluations_per_restart": evaluations,
        "gradient_updates_per_restart": evaluations - 1,
        "endpoint_evaluations_per_restart": 1,
        "best_evaluation_by_restart": [int(value) for value in optimized["best_evaluation"].cpu()],
        "selected_expected_H_over_377": [float(value) for value in optimized["loss"].cpu()],
        "initial_expected_H_over_377": list(trace[0]["loss_by_restart"]),
        "final_expected_H_over_377": list(trace[-1]["loss_by_restart"]),
        "gamma_by_restart": optimized["gamma"].cpu().tolist(),
        "beta_by_restart": optimized["beta"].cpu().tolist(),
        "metrics": metrics,
        "state_sha256": common.state_sha256(optimized["state"]),
        "optimizer_trace": trace,
        "optimizer_trace_sha256": common.sha256_json(trace),
        "elapsed_sec": optimized["elapsed_sec"],
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(resolved)) if resolved.type == "cuda" else None
        ),
        "cuda_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(resolved)) if resolved.type == "cuda" else None
        ),
        "checkpoint_selected_by_target_blind_loss_only": True,
        "independent_depth_optimization_no_continuation": True,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
        "execution_environment": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "requested_device": device,
            "resolved_device": str(resolved),
            "cuda_device_name": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None,
        },
    }
    payload = {**body, "result_sha256": common.sha256_json(body)}
    if production:
        destination.parent.mkdir(parents=True, exist_ok=True)
        common.atomic_write_json(destination, payload)
        validate_result(manifest, payload, seed, depth)
    return payload


def plan_shards(root: Path) -> dict[str, object]:
    manifest = validated_manifest(root)
    tasks = [
        {
            "table_seed": seed,
            "table_id": table_id(seed),
            "depth": depth,
            "method_id": method_id(depth),
            "estimated_seconds": max(10.0, 2_500.0 * depth / 64.0),
        }
        for depth in POSITIVE_DEPTHS for seed in TABLE_SEEDS
    ]
    tasks.sort(key=lambda row: (-float(row["estimated_seconds"]), -int(row["depth"]), int(row["table_seed"])))
    shards: list[list[dict[str, object]]] = [[], []]
    loads = [0.0, 0.0]
    for task in tasks:
        index = min(range(2), key=lambda value: (loads[value], len(shards[value]), value))
        shards[index].append(task)
        loads[index] += float(task["estimated_seconds"])
    actual = {(int(row["table_seed"]), int(row["depth"])) for shard in shards for row in shard}
    expected = {(seed, depth) for seed in TABLE_SEEDS for depth in POSITIVE_DEPTHS}
    common._require(len(tasks) == 70 and actual == expected, "vanilla task union changed")
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-two-pod-plan",
        "status": "FROZEN_TWO_DISJOINT_A800_VANILLA_QUEUES_NOT_LAUNCHED",
        "manifest_sha256": manifest["manifest_sha256"],
        "assignment_rule": "greedy descending-depth calibrated-time balance",
        "optimizer_task_count": len(tasks),
        "derived_p0_cell_count": len(TABLE_SEEDS),
        "shards": [
            {
                "shard_index": index,
                "platform_label": ("gpu_worker_1", "gpu_worker_2")[index],
                "task_count": len(shards[index]),
                "estimated_serial_seconds": loads[index],
                "recommended_concurrent_workers": 2,
                "tasks_in_execution_order": shards[index],
            }
            for index in range(2)
        ],
        "exact_expected_task_set_complete": True,
        "duplicate_task_count": 0,
        "optimization_launched": False,
    }
    payload = {**body, "plan_sha256": common.sha256_json(body)}
    destination = _plan_path(root)
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace vanilla plan {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def validated_plan(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = validated_manifest(root)
    plan = common.load_json(_plan_path(root))
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    tasks = [(int(row["table_seed"]), int(row["depth"])) for shard in plan.get("shards", ()) for row in shard.get("tasks_in_execution_order", ())]
    expected = {(seed, depth) for seed in TABLE_SEEDS for depth in POSITIVE_DEPTHS}
    common._require(
        plan.get("schema") == f"{SCHEMA}-two-pod-plan"
        and plan.get("manifest_sha256") == manifest["manifest_sha256"]
        and plan.get("plan_sha256") == common.sha256_json(body)
        and len(tasks) == len(expected) and set(tasks) == expected,
        "invalid vanilla plan",
    )
    return manifest, plan


def seal(root: Path) -> dict[str, object]:
    manifest, plan = validated_plan(root)
    completed, expected_paths = [], set()
    for seed in TABLE_SEEDS:
        for depth in POSITIVE_DEPTHS:
            path = result_path(root, seed, depth)
            expected_paths.add(path.resolve())
            result = common.load_json(path)
            validate_result(manifest, result, seed, depth)
            completed.append({"table_seed": seed, "depth": depth, "result_sha256": result["result_sha256"]})
    actual = {path.resolve() for path in (root / RESULT_DIRECTORY).glob("*/*.json")}
    common._require(actual == expected_paths, "vanilla result set not exact")
    body: dict[str, object] = {
        "schema": f"{SCHEMA}-global-seal",
        "status": "SEALED_COMPLETE_TARGET_BLIND_VANILLA_QAOA_DIAGNOSTIC",
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "complete_optimizer_result_count": len(completed),
        "expected_optimizer_result_count": len(TABLE_SEEDS) * len(POSITIVE_DEPTHS),
        "derived_p0_cell_count": len(TABLE_SEEDS),
        "completed_cells": completed,
        "target_artifact_loaded": False,
        "target_constructed": False,
        "target_probability_computed": False,
    }
    payload = {**body, "seal_sha256": common.sha256_json(body)}
    destination = root / GLOBAL_SEAL_NAME
    if destination.exists():
        existing = common.load_json(destination)
        if existing != payload:
            raise FileExistsError(f"refusing to replace vanilla seal {destination}")
        return existing
    common.atomic_write_json(destination, payload)
    return payload


def smoke(root: Path, *, device: str, depth: int = 64, evaluations: int = 2) -> dict[str, object]:
    common._require(depth in POSITIVE_DEPTHS and evaluations >= 2, "bad smoke")
    payload = run_cell(
        root, TABLE_SEEDS[0], depth, device=device,
        evaluations=evaluations, production=False,
    )
    trace = payload["optimizer_trace"]
    selected = np.asarray(payload["selected_expected_H_over_377"], dtype=np.float64)
    gamma = np.asarray(payload["gamma_by_restart"], dtype=np.float64)
    beta = np.asarray(payload["beta_by_restart"], dtype=np.float64)
    metric_arrays = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["metrics"].items()
    }
    finite_audit = {
        "expected_evaluations_observed": [row["evaluation"] for row in trace]
        == [1, evaluations],
        "all_trace_losses_finite": all(
            np.all(np.isfinite(np.asarray(row["loss_by_restart"], dtype=np.float64)))
            for row in trace
        ),
        "nonendpoint_gradient_finite": bool(
            np.asarray(trace[0].get("gradient_l2_by_restart"), dtype=np.float64).shape
            == (len(RESTART_SEEDS),)
            and np.all(
                np.isfinite(
                    np.asarray(trace[0].get("gradient_l2_by_restart"), dtype=np.float64)
                )
            )
        ),
        "selected_loss_finite": selected.shape == (len(RESTART_SEEDS),)
        and bool(np.all(np.isfinite(selected))),
        "angles_finite": bool(np.all(np.isfinite(gamma)) and np.all(np.isfinite(beta))),
        "all_metric_vectors_finite": all(
            values.shape == (len(RESTART_SEEDS),) and np.all(np.isfinite(values))
            for values in metric_arrays.values()
        ),
        "state_norm_close_to_one": bool(
            np.allclose(metric_arrays["state_norm"], 1.0, atol=2e-5, rtol=2e-5)
        ),
    }
    common._require(all(finite_audit.values()), "vanilla smoke finite-output audit failed")
    scientific_evidence = {
        "selected_loss": selected.tolist(),
        "initial_loss": list(trace[0]["loss_by_restart"]),
        "endpoint_loss": list(trace[-1]["loss_by_restart"]),
        "gradient_l2": list(trace[0]["gradient_l2_by_restart"]),
        "state_norm": metric_arrays["state_norm"].tolist(),
        "gamma_l2": np.sqrt(np.sum(gamma * gamma, axis=1)).tolist(),
        "beta_l2": np.sqrt(np.sum(beta * beta, axis=1)).tolist(),
        "metric_vectors": {
            key: values.tolist() for key, values in metric_arrays.items()
        },
    }
    return {
        "status": "EXCLUDED_VANILLA_QAOA_SMOKE_PASS",
        "production_result_written": False,
        "depth": depth,
        "restart_batch": len(RESTART_SEEDS),
        "evaluations": evaluations,
        "finite_output_audit": finite_audit,
        "scientific_evidence": scientific_evidence,
        "elapsed_sec": payload["elapsed_sec"],
        "cuda_peak_memory_allocated_bytes": payload["cuda_peak_memory_allocated_bytes"],
        "cuda_peak_memory_reserved_bytes": payload["cuda_peak_memory_reserved_bytes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--extension-campaign-root", type=Path, required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--extension-campaign-root", type=Path, required=True)
    command = sub.add_parser("run-cell")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--table-seed", type=int, choices=TABLE_SEEDS, required=True)
    command.add_argument("--depth", type=int, choices=POSITIVE_DEPTHS, required=True)
    command.add_argument("--device", default="cuda")
    command = sub.add_parser("plan-shards")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("seal")
    command.add_argument("--campaign-root", type=Path, required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--depth", type=int, choices=POSITIVE_DEPTHS, default=64)
    command.add_argument("--evaluations", type=int, default=2)
    command.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        payload = freeze(args.campaign_root, args.extension_campaign_root)
    elif args.command == "prepare":
        freeze(args.campaign_root, args.extension_campaign_root)
        payload = plan_shards(args.campaign_root)
    elif args.command == "run-cell":
        payload = run_cell(args.campaign_root, args.table_seed, args.depth, device=args.device)
    elif args.command == "plan-shards":
        payload = plan_shards(args.campaign_root)
    elif args.command == "seal":
        payload = seal(args.campaign_root)
    elif args.command == "smoke":
        payload = smoke(
            args.campaign_root, device=args.device,
            depth=args.depth, evaluations=args.evaluations,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(payload.get("status", payload))


if __name__ == "__main__":
    main()
