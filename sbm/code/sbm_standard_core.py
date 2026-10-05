"""Efficient exact-statevector kernels for the standard-loss SBM campaign.

Every variational objective in this module is an expected energy.  Exact-ground
probability is evaluated only after optimization and is never used to choose a
checkpoint or restart.
"""

from __future__ import annotations

import json
import math
import re
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


RUNNER_SCHEMA_VERSION = 2
RUNNER_IMPLEMENTATION_ID = "sbm_standard_loss_v2_pcg64_queuebound_20260826"
INITIALIZATION_SCHEME = "per_restart_numpy_pcg64_v1_with_linear_restart_zero"
SPSA_PERTURBATION_SCHEME = "per_restart_numpy_pcg64_rademacher_v1"
METHOD_SPEC_KEYS = (
    "stage1_only",
    "lp_qaoa",
    "warm_start_qaoa",
    "standard_qaoa",
)
QUEUE_TASK_SPEC_KEYS = {
    "runner_schema_version",
    "runner_implementation_id",
    "name",
    "runner",
    "args",
}
SUPPORTED_RUNNERS = {
    "run_cell.py",
    "run_depth_scan.py",
    "run_natural_sbm_adam_direct_microbatch.py",
}


@dataclass(frozen=True)
class InstanceSpec:
    n: int
    graph_seed: int
    group_size: int
    p_in: float | None
    p_out: float | None
    coarse_inter_weight: float
    p_micro: float | None = None
    macro_assignment: str = "alternating_contiguous_microblocks"
    coarse_intra_weight: float = 1.0
    coarse_topology: str = "full"
    coarse_forest_edges: int = 0
    centered_threshold: float = 0.5
    feasible_hamming_weight: int | None = None
    mixer_type: str = "x"
    edge_weight_distribution: str = "unit"
    edge_weight_coefficient_of_variation: float = 0.0
    edge_sign_distribution: str = "all_positive"
    edge_sign_reliability: float = 1.0
    fine_support_model: str = "independent_pair_bernoulli"
    intra_edge_sign_policy: str = "sampled"
    intra_edge_scale: float = 1.0
    precision: str = "single"
    cost_chunk_size: int = 2_097_152


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    steps: int
    restarts: int
    adam_lr: float = 0.04
    spsa_a: float = 0.04
    spsa_c: float = 0.12
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101


def resolve_method_optimizer_specs(arguments: Any) -> dict[str, OptimizerSpec]:
    """Resolve method-specific overrides while preserving the original CLI defaults."""
    base = OptimizerSpec(
        name=arguments.optimizer,
        steps=arguments.steps,
        restarts=arguments.restarts,
        adam_lr=arguments.adam_lr,
        spsa_a=arguments.spsa_a,
        spsa_c=arguments.spsa_c,
        spsa_alpha=arguments.spsa_alpha,
        spsa_gamma=arguments.spsa_gamma,
    )
    prefixes = {
        "stage1_only": "stage1",
        "lp_qaoa": "lp",
        "warm_start_qaoa": "warm",
        "standard_qaoa": "direct",
    }
    fields = (
        "steps", "restarts", "adam_lr", "spsa_a", "spsa_c", "spsa_alpha", "spsa_gamma",
    )
    resolved: dict[str, OptimizerSpec] = {}
    for method, prefix in prefixes.items():
        values = asdict(base)
        for field in fields:
            override = getattr(arguments, f"{prefix}_{field}", None)
            if override is not None:
                values[field] = override
        spec = OptimizerSpec(**values)
        if spec.name not in {"adam", "spsa"}:
            raise ValueError(f"unknown optimizer family: {spec.name}")
        if spec.steps <= 0 or spec.restarts <= 0:
            raise ValueError(f"{method} steps and restarts must be positive")
        if spec.adam_lr <= 0.0 or spec.spsa_a <= 0.0 or spec.spsa_c <= 0.0:
            raise ValueError(f"{method} optimizer scales must be positive")
        if spec.spsa_alpha < 0.0 or spec.spsa_gamma < 0.0:
            raise ValueError(f"{method} SPSA exponents must be nonnegative")
        resolved[method] = spec
    if len({spec.name for spec in resolved.values()}) != 1:
        raise ValueError("all methods in one cell must use the same optimizer family")
    return resolved


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0x7FFFFFFF)


def instance_spec_payload(instance: InstanceSpec) -> dict[str, Any]:
    """Serialize an instance while omitting the opt-in field at its legacy default."""
    payload = asdict(instance)
    if instance.macro_assignment == "alternating_contiguous_microblocks":
        payload.pop("macro_assignment")
    if instance.feasible_hamming_weight is None:
        payload.pop("feasible_hamming_weight")
    if instance.mixer_type == "x":
        payload.pop("mixer_type")
    if instance.edge_weight_distribution == "unit":
        payload.pop("edge_weight_distribution")
    if instance.edge_weight_coefficient_of_variation == 0.0:
        payload.pop("edge_weight_coefficient_of_variation")
    if instance.edge_sign_distribution == "all_positive":
        payload.pop("edge_sign_distribution")
    if instance.edge_sign_reliability == 1.0:
        payload.pop("edge_sign_reliability")
    if instance.fine_support_model == "independent_pair_bernoulli":
        payload.pop("fine_support_model")
    if instance.intra_edge_sign_policy == "sampled":
        payload.pop("intra_edge_sign_policy")
    if instance.intra_edge_scale == 1.0:
        payload.pop("intra_edge_scale")
    return payload


def target_definition(instance: InstanceSpec) -> str:
    if instance.feasible_hamming_weight is None:
        return "complete_exact_ground_set"
    return "complete_exact_ground_set_within_fixed_hamming_weight_sector"


def target_evaluation_rule(instance: InstanceSpec) -> str:
    if instance.feasible_hamming_weight is None:
        return "complete_exact_ground_set_after_optimization"
    return "complete_exact_feasible_sector_ground_set_after_optimization"


def mixer_protocol_metadata(instance: InstanceSpec) -> dict[str, Any]:
    """Describe the opt-in mixer without allocating statevector-sized tensors."""
    if instance.mixer_type != "ring_xy":
        raise ValueError("mixer metadata is only emitted for the opt-in ring_xy branch")
    n = int(instance.n)
    even_matching = [[left, left + 1] for left in range(0, n, 2)]
    odd_matching = [[left, (left + 1) % n] for left in range(1, n, 2)]
    return {
        "mixer_type": "ring_xy",
        "hamiltonian_weight_constraint": int(instance.feasible_hamming_weight),
        "initial_state": "uniform_dicke_state_in_fixed_hamming_weight_sector",
        "initial_state_preparation_ru": n * n,
        "initial_state_preparation_ru_formula": "N^2",
        "initial_state_preparation_ru_interpretation": (
            "experiment_resource_accounting_convention_for_deterministic_Dicke_state_"
            "preparation;proxy_not_exact_physical_gate_count"
        ),
        "gate_convention": "exp[-i*beta*(XX+YY)/2]",
        "qubit_labels": "graph_node_indices_equal_most_significant_to_least_significant_bit_order",
        "product_formula_order": "even_edge_matching_then_odd_edge_matching",
        "even_edge_matching": even_matching,
        "odd_edge_matching": odd_matching,
        "exchange_gates_per_mixer_layer": n,
        "resource_accounting": (
            "one_resource_unit_per_exchange_gate;N_units_per_even_N_ring_mixer_layer"
        ),
        "methods_using_ring_xy": [
            "stage1_only", "warm_start_qaoa", "standard_qaoa",
        ],
        "lp_stage2_mixer": "learned_reference_projector_reflection_unchanged",
        "standard_qaoa_label": "fixed_weight_ring_xy_qaoa_not_transverse_field_qaoa",
    }


def canonical_queue_task_spec(task: dict[str, Any]) -> dict[str, Any]:
    """Return the complete, unhashed queue identity embedded in every result."""
    if not isinstance(task, dict):
        raise ValueError("each queue task must be a mapping")
    unexpected = set(task) - {"name", "runner", "args"}
    if unexpected:
        raise ValueError(f"unsupported queue task fields: {sorted(unexpected)}")
    name = task.get("name")
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None
        or name in {".", "..", "queue_status", "summary"}
    ):
        raise ValueError("queue task name must be safe and cannot use a reserved result name")
    runner = task.get("runner", "run_cell.py")
    if runner not in SUPPORTED_RUNNERS:
        raise ValueError(f"unsupported runner {runner}")
    arguments = task.get("args")
    if not isinstance(arguments, dict):
        raise ValueError(f"queue task {name} args must be a mapping")
    normalized_arguments: dict[str, Any] = {}
    for raw_key, value in arguments.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"queue task {name} argument names must be strings")
        normalized_key = raw_key.replace("_", "-")
        # Keep one canonical serialized spelling for the shared edge-count slot.
        # The top-K spelling is a clearer CLI alias, while the established key
        # preserves compatibility with existing queue/result identities.
        if normalized_key == "coarse-top-k-edges":
            normalized_key = "coarse-forest-edges"
        if normalized_key in {"output", "queue-task-spec-json"}:
            raise ValueError(f"queue task {name} contains a reserved runner argument")
        if normalized_key in normalized_arguments:
            raise ValueError(
                f"queue task {name} has duplicate arguments after CLI normalization: {normalized_key}"
            )
        normalized_arguments[normalized_key] = value
    # A JSON round trip makes the embedded structure independent of Python aliases
    # while preserving every JSON-visible argument without hashes or digests.
    canonical_arguments = json.loads(json.dumps(
        normalized_arguments, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ))
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "runner_implementation_id": RUNNER_IMPLEMENTATION_ID,
        "name": name,
        "runner": runner,
        "args": canonical_arguments,
    }


def queue_cli_arguments(values: dict[str, Any]) -> list[str]:
    """Translate canonical queue arguments to runner CLI tokens."""
    arguments: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif isinstance(value, list):
            if value and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
                serialized = ",".join(f"{item[0]}:{item[1]}" for item in value)
            else:
                serialized = ",".join(str(item) for item in value)
            arguments.extend((flag, serialized))
        else:
            arguments.extend((flag, str(value)))
    return arguments


def validate_queue_runtime_binding(arguments: Any, parser: Any, runner: str) -> None:
    """Bind an embedded queue task to every realized argument, including defaults."""
    embedded = getattr(arguments, "queue_task_spec_json", None)
    if embedded is None:
        return
    embedded = validate_embedded_queue_task_spec(embedded)
    if embedded["runner"] != runner:
        raise ValueError("embedded queue task selected a different runner")
    expected = parser.parse_args([
        "--output", str(getattr(arguments, "output")),
        "--queue-task-spec-json", json.dumps(embedded, separators=(",", ":")),
        *queue_cli_arguments(embedded["args"]),
    ])
    ignored = {"output", "queue_task_spec_json"}
    mismatches = {
        key: {"embedded": value, "realized": getattr(arguments, key, None)}
        for key, value in vars(expected).items()
        if key not in ignored and getattr(arguments, key, None) != value
    }
    if mismatches:
        raise ValueError(
            "embedded queue arguments differ from realized runner arguments: "
            + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
        )


def validate_embedded_queue_task_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) != QUEUE_TASK_SPEC_KEYS:
        raise ValueError("invalid embedded queue task spec")
    if spec.get("runner_schema_version") != RUNNER_SCHEMA_VERSION:
        raise ValueError("embedded queue task schema mismatch")
    if spec.get("runner_implementation_id") != RUNNER_IMPLEMENTATION_ID:
        raise ValueError("embedded queue task implementation mismatch")
    canonical = canonical_queue_task_spec({
        "name": spec.get("name"),
        "runner": spec.get("runner"),
        "args": spec.get("args"),
    })
    if spec != canonical:
        raise ValueError("embedded queue task spec is not canonical")
    return canonical


def make_stage1_reference_key(instance: InstanceSpec, depth: int,
                              optimizer: OptimizerSpec, initialization_seed: int) -> str:
    instance_payload = instance_spec_payload(instance)
    # The optional hierarchical-SBM parameter must not alter any established
    # two-community reference key when the new branch is inactive.
    if instance.p_micro is None:
        instance_payload.pop("p_micro")
    # Preserve every existing full-topology reference key byte for byte.  Forest
    # parameters remain part of the key whenever the new branch is active.
    if instance.coarse_topology == "full" and instance.coarse_forest_edges == 0:
        instance_payload.pop("coarse_topology")
        instance_payload.pop("coarse_forest_edges")
    descriptor = {
        "schema": RUNNER_SCHEMA_VERSION,
        "instance": instance_payload,
        "stage1_depth": int(depth),
        "stage1_optimizer": asdict(optimizer),
        "initialization_seed": int(initialization_seed),
    }
    return "stage1_reference:" + json.dumps(descriptor, sort_keys=True, separators=(",", ":"))


def exact_rts99(probability: float) -> int | None:
    if not math.isfinite(probability) or probability <= 0.0:
        return None
    if probability >= 1.0:
        return 1
    return int(math.ceil(math.log(0.01) / math.log1p(-probability)))


def normalize_state(state: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        norm = torch.linalg.vector_norm(state)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
            raise FloatingPointError("cannot normalize an invalid state")
        return (state / norm).detach().clone()


def resource_units(n: int, method: str, p1: int | None, p2: int | None,
                   coarse_two_body_term_count: int | None = None,
                   fine_two_body_term_count: int | None = None,
                   initial_state_preparation_ru: int = 0) -> int:
    """Current-paper resource model, including terminal fine-energy evaluation."""
    pairs = n * (n - 1) // 2
    coarse_pairs = pairs if coarse_two_body_term_count is None else int(coarse_two_body_term_count)
    fine_pairs = pairs if fine_two_body_term_count is None else int(fine_two_body_term_count)
    if coarse_pairs < 0 or coarse_pairs > pairs:
        raise ValueError("coarse two-body term count must lie between zero and C(n,2)")
    if fine_pairs < 0 or fine_pairs > pairs:
        raise ValueError("fine two-body term count must lie between zero and C(n,2)")
    preparation = int(initial_state_preparation_ru)
    if preparation < 0 or preparation != initial_state_preparation_ru:
        raise ValueError("initial-state preparation resource units must be a nonnegative integer")
    terminal = fine_pairs
    stage1 = preparation + (0 if p1 is None else p1 * (coarse_pairs + n))
    if method == "stage1_only":
        assert p1 is not None
        return int(stage1 + terminal)
    if method == "standard_qaoa":
        assert p2 is not None
        return int(preparation + p2 * (fine_pairs + n) + terminal)
    assert p1 is not None and p2 is not None
    if method == "warm_start_qaoa":
        return int(stage1 + p2 * (fine_pairs + n) + terminal)
    if method == "lp_qaoa":
        learned_mixer = 2 * stage1 + n * n
        return int(stage1 + p2 * (fine_pairs + learned_mixer) + terminal)
    raise ValueError(f"unknown method: {method}")


def coarse_topology_metadata(instance: InstanceSpec) -> dict[str, Any]:
    """Describe the topology branch without allocating any simulation tensors."""
    if instance.fine_support_model == "quotient_k22_plus_intra":
        return {
            "topology": "fine_support_quotient_reused_exactly",
            "requested_forest_edges": int(instance.coarse_forest_edges),
            "candidate_statistic": "not_applicable_no_data_dependent_edge_selection",
            "edge_order": "ascending_lexicographic_quotient_pair",
            "selection": (
                "single_child_1_connected_simple_random_3_regular_quotient_reused_for_"
                "both_fine_support_and_coarse_projection"
            ),
            "topology_sampling_stream": "numpy_seedsequence_child_1_pcg64",
            "edge_weights": (
                "literal_float64_block_projection;realized_intra_coupling_retained;"
                "each_quotient_K2,2_quartet_replaced_by_its_realized_signed_mean"
            ),
            "edge_selection_uses_sampled_weights_signs_labels_or_fine_oracle": False,
            "resource_term_count": "same_fixed_84_edge_register_as_fine_support",
            "simulation_memory": "no_additional_statevector_or_cost_tensor;small_host_metadata_only",
        }
    if instance.edge_sign_distribution != "all_positive":
        block_statistic = "empirical_mean_centered_signed_weighted_adjacency"
    elif instance.edge_weight_distribution != "unit":
        block_statistic = "empirical_mean_centered_weighted_adjacency"
    else:
        block_statistic = "empirical_mean_centered_adjacency"
    if instance.coarse_topology == "independent_random_3_regular":
        return {
            "topology": instance.coarse_topology,
            "requested_forest_edges": int(instance.coarse_forest_edges),
            "candidate_statistic": (
                "not_applicable_topology_sampled_independently_without_consulting_adjacency_values"
            ),
            "edge_order": "ascending_lexicographic_group_pair_after_topology_sampling",
            "selection": (
                "independent_configuration_model_rejection_until_simple_and_connected;"
                "10000_attempt_cap;fail_if_no_valid_graph"
            ),
            "topology_sampling_stream": "numpy_seedsequence_child_1_pcg64",
            "edge_weights": (
                f"{block_statistic}_on_each_fixed_group_pair"
            ),
            "edge_selection_uses_sampled_adjacency": False,
            "edge_selection_uses_planted_labels_or_fine_target": False,
            "resource_term_count": "fixed_selected_pair_register",
            "simulation_memory": "no_additional_statevector_or_cost_tensor; small_host_metadata_only",
        }
    if instance.coarse_topology == "connected_top_k":
        return {
            "topology": instance.coarse_topology,
            # coarse_forest_edges is the established serialized storage slot.
            "requested_forest_edges": int(instance.coarse_forest_edges),
            "requested_top_k_edges": int(instance.coarse_forest_edges),
            "candidate_statistic": (
                f"{block_statistic}_for_each_group_pair"
            ),
            "edge_order": "ascending_lexicographic_tuple(-abs(block_mean),left_group,right_group)",
            "selection": (
                "deterministic_union_find_maximum_abs_block_mean_spanning_tree_"
                "then_remaining_pairs_in_edge_order"
            ),
            "edge_selection_uses_planted_labels_or_fine_target": False,
            "resource_term_count": "retained_nonzero_pair_terms",
            "simulation_memory": "no_additional_statevector_or_cost_tensor; small_host_metadata_only",
        }
    forest = instance.coarse_topology == "maximum_spanning_forest"
    return {
        "topology": instance.coarse_topology,
        "requested_forest_edges": int(instance.coarse_forest_edges),
        "candidate_statistic": (
            f"{block_statistic}_for_each_group_pair"
        ),
        "edge_order": (
            "ascending_lexicographic_tuple(-abs(block_mean),left_group,right_group)"
            if forest else "not_applicable_all_group_pairs_retained"
        ),
        "selection": "deterministic_union_find_acyclic_prefix" if forest else "all_group_pairs",
        "edge_selection_uses_planted_labels_or_fine_target": False,
        "resource_term_count": (
            "retained_nonzero_pair_terms" if forest else "legacy_complete_pair_register"
        ),
        "simulation_memory": "no_additional_statevector_or_cost_tensor; small_host_metadata_only",
    }


def sbm_model_metadata(instance: InstanceSpec) -> dict[str, Any]:
    """Describe the graph distribution and the meaning of each probability."""
    if instance.fine_support_model == "quotient_k22_plus_intra":
        return {
            "model_type": "hierarchical_random_regular_block_model",
            "family_description": "random_regular_censored_hierarchical_spin_glass",
            "standard_iid_sbm_claimed": False,
            "maximum_likelihood_claimed": False,
            "microblock_size": int(instance.group_size),
            "macro_assignment": instance.macro_assignment,
            "macro_assignment_stream": "numpy_seedsequence_child_0_pcg64",
            "macro_balance": "equal_number_of_positive_and_negative_microblocks",
            "fine_support_model": instance.fine_support_model,
            "support_rule": (
                "all_intra_K2_edges_plus_all_four_K2,2_edges_for_every_edge_of_one_"
                "connected_simple_random_3_regular_microblock_quotient"
            ),
            "support_topology_stream": "numpy_seedsequence_child_1_pcg64",
            "adjacency_sampling_stream": "unused_child_2_reserved_for_cross_version_identity",
            "edge_weight_model": {
                "distribution": instance.edge_weight_distribution,
                "mean": 1.0,
                "coefficient_of_variation": float(
                    instance.edge_weight_coefficient_of_variation
                ),
                "sampling_stream": "numpy_seedsequence_child_3_pcg64",
                "sampled_for": "all_unordered_node_pairs_before_support_is_applied",
                "independent_of_labels_support_and_signs": True,
                "intra_edge_scale": float(instance.intra_edge_scale),
            },
            "edge_sign_model": {
                "distribution": instance.edge_sign_distribution,
                "inter_edge_desired_sign_rule": (
                    "+1_within_planted_macro_community;-1_across_planted_macro_communities"
                ),
                "inter_edge_desired_sign_reliability": float(
                    instance.edge_sign_reliability
                ),
                "intra_edge_sign_policy": instance.intra_edge_sign_policy,
                "sampling_stream": "numpy_seedsequence_child_4_pcg64",
                "sampled_for": "all_unordered_node_pairs_before_support_is_applied",
                "intra_pair_uniforms_recorded_but_sign_overridden_positive": True,
            },
        }
    if instance.p_micro is None:
        metadata = {
            "model_type": "two_community_sbm",
            "edge_probabilities": {
                "same_macro_community": float(instance.p_in),
                "different_macro_communities": float(instance.p_out),
            },
        }
        if instance.coarse_topology == "independent_random_3_regular":
            metadata["adjacency_sampling_stream"] = "numpy_seedsequence_child_2_pcg64"
        if instance.edge_weight_distribution != "unit":
            metadata["edge_weight_model"] = {
                "distribution": instance.edge_weight_distribution,
                "mean": 1.0,
                "coefficient_of_variation": float(
                    instance.edge_weight_coefficient_of_variation
                ),
                "sampling_stream": "numpy_seedsequence_child_3_pcg64",
                "sampled_for": "all_unordered_node_pairs_before_adjacency_is_applied",
                "independent_of_labels_adjacency_and_coarse_topology": True,
            }
        if instance.edge_sign_distribution != "all_positive":
            metadata["edge_sign_model"] = {
                "distribution": instance.edge_sign_distribution,
                "desired_sign_rule": (
                    "+1_within_planted_macro_community;-1_across_planted_macro_communities"
                ),
                "intra_edge_sign_policy": instance.intra_edge_sign_policy,
                "configured_reliability_applies_to": (
                    "inter_microblock_pairs_only"
                    if instance.intra_edge_sign_policy == "deterministic_positive"
                    else "all_unordered_pairs"
                ),
                "desired_sign_reliability": float(instance.edge_sign_reliability),
                "sampling_stream": "numpy_seedsequence_child_4_pcg64",
                "sampled_for": "all_unordered_node_pairs_before_adjacency_is_applied",
                "random_draws_independent_of_labels_adjacency_topology_and_magnitudes": True,
            }
        return metadata
    metadata = {
        "model_type": "hierarchical_two_level_sbm",
        "edge_probabilities": {
            "same_microblock": float(instance.p_micro),
            "different_microblocks_same_macro_community": float(instance.p_in),
            "different_macro_communities": float(instance.p_out),
        },
        "microblock_size": int(instance.group_size),
        "macro_assignment": instance.macro_assignment,
    }
    if float(instance.centered_threshold) != 0.5:
        metadata["centered_adjacency_threshold"] = float(instance.centered_threshold)
    if instance.macro_assignment == "random_balanced_microblocks":
        metadata.update({
            "macro_assignment_stream": "numpy_seedsequence_child_0_pcg64",
            "macro_balance": "equal_number_of_positive_and_negative_microblocks",
        })
    if (
        instance.macro_assignment == "random_balanced_microblocks"
        or instance.coarse_topology == "independent_random_3_regular"
        or instance.edge_weight_distribution != "unit"
    ):
        metadata["adjacency_sampling_stream"] = "numpy_seedsequence_child_2_pcg64"
    if instance.edge_weight_distribution != "unit":
        metadata["edge_weight_model"] = {
            "distribution": instance.edge_weight_distribution,
            "mean": 1.0,
            "coefficient_of_variation": float(
                instance.edge_weight_coefficient_of_variation
            ),
            "sampling_stream": "numpy_seedsequence_child_3_pcg64",
            "sampled_for": "all_unordered_node_pairs_before_adjacency_is_applied",
            "independent_of_labels_adjacency_and_coarse_topology": True,
        }
    if instance.edge_sign_distribution != "all_positive":
        metadata["edge_sign_model"] = {
            "distribution": instance.edge_sign_distribution,
            "desired_sign_rule": (
                "+1_within_planted_macro_community;-1_across_planted_macro_communities"
            ),
            "desired_sign_reliability": float(instance.edge_sign_reliability),
            "sampling_stream": "numpy_seedsequence_child_4_pcg64",
            "sampled_for": "all_unordered_node_pairs_before_adjacency_is_applied",
            "random_draws_independent_of_labels_adjacency_topology_and_magnitudes": True,
        }
    return metadata


class SBMInstance:
    """Centered-adjacency two-community SBM and optional hierarchical variant."""

    def __init__(self, spec: InstanceSpec, device: torch.device) -> None:
        if spec.n <= 0 or spec.group_size <= 0:
            raise ValueError("n and group_size must be positive")
        if spec.n % spec.group_size != 0:
            raise ValueError("n must be divisible by group_size")
        if spec.edge_sign_distribution not in {
            "all_positive", "planted_macro_reliability",
        }:
            raise ValueError(
                "edge_sign_distribution must be all_positive or planted_macro_reliability"
            )
        edge_sign_reliability = float(spec.edge_sign_reliability)
        if not math.isfinite(edge_sign_reliability) or not (
            0.0 <= edge_sign_reliability <= 1.0
        ):
            raise ValueError("edge-sign reliability must lie in the closed interval [0,1]")
        if (
            spec.edge_sign_distribution == "all_positive"
            and edge_sign_reliability != 1.0
        ):
            raise ValueError("all-positive edge signs require reliability one")
        signed_branch = spec.edge_sign_distribution != "all_positive"
        if spec.fine_support_model not in {
            "independent_pair_bernoulli", "quotient_k22_plus_intra",
        }:
            raise ValueError(
                "fine_support_model must be independent_pair_bernoulli or "
                "quotient_k22_plus_intra"
            )
        regular_block_support = spec.fine_support_model == "quotient_k22_plus_intra"
        if spec.intra_edge_sign_policy not in {"sampled", "deterministic_positive"}:
            raise ValueError(
                "intra_edge_sign_policy must be sampled or deterministic_positive"
            )
        intra_edge_scale = float(spec.intra_edge_scale)
        if not math.isfinite(intra_edge_scale) or intra_edge_scale <= 0.0:
            raise ValueError("intra_edge_scale must be finite and strictly positive")
        if regular_block_support:
            if spec.group_size != 2:
                raise ValueError("quotient K2,2 support requires two-site microblocks")
            if spec.p_micro is not None or spec.p_in is not None or spec.p_out is not None:
                raise ValueError(
                    "quotient K2,2 support has no Bernoulli probabilities; require "
                    "p_micro=p_in=p_out=None"
                )
            if not signed_branch:
                raise ValueError("quotient K2,2 support requires the censored sign model")
            if spec.intra_edge_sign_policy != "deterministic_positive":
                raise ValueError(
                    "quotient K2,2 support requires deterministic-positive intra signs"
                )
            if spec.coarse_topology != "independent_random_3_regular":
                raise ValueError(
                    "quotient K2,2 support requires the same random 3-regular coarse topology"
                )
        elif spec.intra_edge_sign_policy != "sampled" or intra_edge_scale != 1.0:
            raise ValueError(
                "nondefault intra sign/scale is available only with quotient K2,2 support"
            )
        if signed_branch and float(spec.centered_threshold) != 0.0:
            raise ValueError("the signed weighted-cut branch requires centered_threshold=0")
        if regular_block_support:
            probability_valid = True
        elif spec.p_in is None or spec.p_out is None:
            raise ValueError("Bernoulli support requires finite p_in and p_out")
        elif spec.p_micro is None:
            probability_valid = (
                0.0 <= spec.p_out <= spec.p_in <= 1.0
                if signed_branch else 0.0 <= spec.p_out < spec.p_in <= 1.0
            )
            if not probability_valid:
                raise ValueError(
                    "require 0 <= p_out <= p_in <= 1 for signed instances and "
                    "0 <= p_out < p_in <= 1 otherwise"
                )
        elif signed_branch:
            if not (0.0 <= spec.p_out <= spec.p_in < spec.p_micro <= 1.0):
                raise ValueError(
                    "signed hierarchical SBM requires 0 <= p_out <= p_in < p_micro <= 1"
                )
        elif not (
            0.0 <= spec.p_out < spec.centered_threshold
            < spec.p_in < spec.p_micro <= 1.0
        ):
            raise ValueError(
                "hierarchical SBM requires 0 <= p_out < centered_threshold "
                "< p_in < p_micro <= 1"
            )
        if spec.precision.lower() not in {"single", "float32", "fp32"}:
            raise ValueError("this campaign intentionally uses complex64/float32")
        if spec.edge_weight_distribution not in {"unit", "lognormal_mean_one"}:
            raise ValueError("edge_weight_distribution must be unit or lognormal_mean_one")
        edge_weight_cv = float(spec.edge_weight_coefficient_of_variation)
        if not math.isfinite(edge_weight_cv) or edge_weight_cv < 0.0:
            raise ValueError("edge-weight coefficient of variation must be finite and nonnegative")
        if spec.edge_weight_distribution == "unit" and edge_weight_cv != 0.0:
            raise ValueError("unit edge weights require coefficient of variation zero")
        if spec.mixer_type not in {"x", "ring_xy"}:
            raise ValueError("mixer_type must be x or ring_xy")
        if spec.mixer_type == "x" and spec.feasible_hamming_weight is not None:
            raise ValueError("a fixed Hamming-weight sector requires mixer_type=ring_xy")
        if spec.mixer_type == "ring_xy" and spec.feasible_hamming_weight is None:
            raise ValueError("mixer_type=ring_xy requires feasible_hamming_weight")
        if spec.mixer_type == "ring_xy":
            weight = int(spec.feasible_hamming_weight)
            if spec.feasible_hamming_weight != weight:
                raise ValueError("feasible_hamming_weight must be an integer")
            if spec.n < 4 or spec.n % 2 != 0:
                raise ValueError("ring_xy requires an even n >= 4")
            if not (0 < weight < spec.n):
                raise ValueError("feasible_hamming_weight must lie strictly between 0 and n")
            if 2 * weight != spec.n:
                raise ValueError("the balanced ring_xy branch requires feasible_hamming_weight=n/2")

        self.spec = spec
        self.device = device
        self.N = int(spec.n)
        self.dim = 1 << self.N
        self.initial_state_preparation_ru = (
            self.N * self.N if spec.feasible_hamming_weight is not None else 0
        )
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.energy_tolerance = 1e-5
        self.groups = [
            list(range(start, start + spec.group_size))
            for start in range(0, self.N, spec.group_size)
        ]
        num_groups = len(self.groups)
        self.uses_independent_instance_streams = (
            spec.macro_assignment == "random_balanced_microblocks"
            or spec.coarse_topology == "independent_random_3_regular"
            or spec.edge_weight_distribution != "unit"
            or spec.edge_sign_distribution != "all_positive"
            or regular_block_support
        )
        if spec.macro_assignment not in {
            "alternating_contiguous_microblocks", "random_balanced_microblocks",
        }:
            raise ValueError(
                "macro_assignment must be alternating_contiguous_microblocks or "
                "random_balanced_microblocks"
            )
        if spec.macro_assignment == "random_balanced_microblocks":
            if spec.p_micro is None and not regular_block_support:
                raise ValueError("random balanced macro assignment requires the hierarchical SBM")
            if num_groups % 2 != 0:
                raise ValueError("random balanced macro assignment requires an even number of microblocks")
        if spec.coarse_topology not in {
            "full", "maximum_spanning_forest", "connected_top_k", "independent_random_3_regular",
        }:
            raise ValueError(
                "coarse_topology must be full, maximum_spanning_forest, connected_top_k, "
                "or independent_random_3_regular"
            )
        if spec.coarse_topology == "full" and spec.coarse_forest_edges != 0:
            raise ValueError("full coarse topology requires coarse_forest_edges=0")
        if spec.coarse_topology == "maximum_spanning_forest" and not (
            1 <= spec.coarse_forest_edges <= num_groups - 1
        ):
            raise ValueError("forest coarse topology requires 1..num_groups-1 edges")
        total_group_pairs = num_groups * (num_groups - 1) // 2
        if spec.coarse_topology == "connected_top_k" and not (
            num_groups - 1 <= spec.coarse_forest_edges <= total_group_pairs
        ):
            raise ValueError(
                "connected top-K coarse topology requires num_groups-1..C(num_groups,2) edges"
            )
        if spec.coarse_topology == "independent_random_3_regular":
            if spec.coarse_forest_edges != 0:
                raise ValueError("independent random 3-regular topology requires coarse_forest_edges=0")
            if num_groups < 4 or num_groups % 2 != 0:
                raise ValueError(
                    "independent random 3-regular topology requires an even number of microblocks >= 4"
                )
        self.planted = self._make_planted()
        self.coarse_topology_vertex_order: list[int] | None = None
        self.coarse_topology_sampling_attempts: int | None = None
        self.coarse_topology_fallback_used: bool | None = None
        self._cached_random_3_regular_pairs: tuple[tuple[int, int], ...] | None = None
        if regular_block_support:
            # v5 samples the quotient exactly once.  Fine support and the
            # literal coarse projection both consume this cached child-1 graph.
            self._connected_random_3_regular_pairs()
        # The complete unordered-pair weight field is sampled before adjacency
        # is applied, from a stream disjoint from labels, topology, and edges.
        self.edge_strengths = self._sample_edge_strengths()
        # The complete sign-flip field is likewise sampled before adjacency.
        # Its uniforms are independent; only the deterministic desired-sign
        # mapping refers to the already sampled planted macro labels.
        self.edge_signs = self._sample_edge_signs()
        self.adjacency, binary_centered_adjacency = self._sample_graph()
        if regular_block_support:
            interaction_scales = np.ones((self.N, self.N), dtype=np.float64)
            for group in self.groups:
                left, right = group
                interaction_scales[left, right] = interaction_scales[right, left] = (
                    intra_edge_scale
                )
            self.interaction_scales = interaction_scales
            self.weighted_adjacency = (
                self.adjacency.astype(np.float64, copy=False)
                * self.edge_strengths.astype(np.float64, copy=False)
                * interaction_scales
            )
        else:
            self.interaction_scales = None
            self.weighted_adjacency = self.adjacency * self.edge_strengths
        if spec.edge_sign_distribution == "all_positive":
            self.signed_weighted_adjacency = self.weighted_adjacency
        else:
            self.signed_weighted_adjacency = (
                self.weighted_adjacency.astype(np.float64, copy=False)
                * self.edge_signs.astype(np.float64, copy=False)
            )
        if (
            spec.edge_weight_distribution == "unit"
            and spec.edge_sign_distribution == "all_positive"
        ):
            # Preserve legacy arrays bit for bit when the opt-in branch is off.
            self.canonical_weighted_adjacency_float64 = None
            self.canonical_signed_weighted_adjacency_float64 = None
            self.canonical_centered_adjacency_float64 = None
            self.centered_adjacency = binary_centered_adjacency
        else:
            self.canonical_weighted_adjacency_float64 = (
                self.weighted_adjacency.astype(np.float64, copy=True)
            )
            self.canonical_signed_weighted_adjacency_float64 = (
                self.signed_weighted_adjacency.astype(np.float64, copy=True)
            )
            canonical_centered = np.zeros((self.N, self.N), dtype=np.float64)
            upper = np.triu_indices(self.N, k=1)
            centered_values = (
                self.canonical_signed_weighted_adjacency_float64[upper]
                - float(spec.centered_threshold)
            )
            canonical_centered[upper] = centered_values
            canonical_centered[(upper[1], upper[0])] = centered_values
            self.canonical_centered_adjacency_float64 = canonical_centered
            # Statevector simulation remains the campaign's declared float32
            # target.  The unrounded coefficients above define the canonical
            # weighted or signed-weighted oracle used by opt-in screens.
            self.centered_adjacency = canonical_centered.astype(np.float32)
        self.fine_pair_coefficients = -self.centered_adjacency
        self.fine_observed_edge_count = int(np.count_nonzero(np.triu(self.adjacency, k=1)))
        if regular_block_support:
            degrees = np.sum(self.adjacency, axis=1, dtype=np.float64)
            expected_edges = len(self.groups) + 4 * (3 * len(self.groups) // 2)
            if (
                self.fine_observed_edge_count != expected_edges
                or not bool(np.all(degrees == 7.0))
            ):
                raise RuntimeError(
                    "quotient K2,2 support must be exactly 7-regular with the expected edge count"
                )
        self.fine_resource_two_body_term_count = (
            self.fine_observed_edge_count
            if spec.feasible_hamming_weight is not None
            else self.N * (self.N - 1) // 2
        )
        self.coarse_pair_coefficients = self._coarse_pair_matrix()
        self.coarse_two_body_term_count = int(np.count_nonzero(
            np.triu(self.coarse_pair_coefficients, k=1),
        ))
        # The established full-topology protocol charges the complete pair
        # register even when an empirical block mean happens to be exactly zero.
        # Data-selected sparse topologies charge retained nonzero terms; the
        # independent fixed topology charges its selected pair register.
        if spec.coarse_topology == "full":
            self.coarse_resource_two_body_term_count = self.N * (self.N - 1) // 2
        elif regular_block_support:
            self.coarse_resource_two_body_term_count = (
                len(self.groups) + 4 * len(self.selected_coarse_block_edges)
            )
        elif spec.coarse_topology == "independent_random_3_regular":
            intra_terms = (
                num_groups * spec.group_size * (spec.group_size - 1) // 2
                if float(spec.coarse_intra_weight) != 0.0 else 0
            )
            cross_terms = (
                len(self.selected_coarse_block_edges) * spec.group_size * spec.group_size
                if float(spec.coarse_inter_weight) != 0.0 else 0
            )
            self.coarse_resource_two_body_term_count = intra_terms + cross_terms
        else:
            self.coarse_resource_two_body_term_count = self.coarse_two_body_term_count
        self.fine_cost, self.coarse_cost = self._enumerate_costs()
        if spec.feasible_hamming_weight is None:
            self.feasible_sector_mask = None
            self.feasible_indices = None
            self.feasible_sector_dimension = self.dim
            self.uniform_state = torch.full(
                (self.dim,), 1.0 / math.sqrt(self.dim),
                dtype=self.complex_dtype, device=self.device,
            )
        else:
            self.feasible_sector_mask, self.feasible_indices = self._make_feasible_sector(
                int(spec.feasible_hamming_weight),
            )
            self.feasible_sector_dimension = int(self.feasible_indices.numel())
            expected_dimension = math.comb(self.N, int(spec.feasible_hamming_weight))
            if self.feasible_sector_dimension != expected_dimension:
                raise RuntimeError("fixed-Hamming-weight sector dimension mismatch")
            self.uniform_state = torch.zeros(
                (self.dim,), dtype=self.complex_dtype, device=self.device,
            )
            self.uniform_state[self.feasible_indices] = 1.0 / math.sqrt(
                self.feasible_sector_dimension
            )
        self._diagnostics()

    def _make_feasible_sector(self, weight: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.empty(self.dim, dtype=torch.bool, device=self.device)
        chunk_size = max(1, int(self.spec.cost_chunk_size))
        with torch.no_grad():
            for start in range(0, self.dim, chunk_size):
                stop = min(start + chunk_size, self.dim)
                indices = torch.arange(start, stop, dtype=torch.int64, device=self.device)
                hamming_weights = torch.zeros_like(indices)
                for shift in range(self.N):
                    hamming_weights.add_(torch.bitwise_and(
                        torch.bitwise_right_shift(indices, shift), 1,
                    ))
                mask[start:stop] = hamming_weights == weight
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        return mask, indices

    def _make_planted(self) -> np.ndarray:
        if self.spec.macro_assignment == "random_balanced_microblocks":
            rng = self._independent_rng(0)
            order = [int(value) for value in rng.permutation(len(self.groups))]
            group_labels = np.full(len(self.groups), -1, dtype=np.int8)
            group_labels[order[:len(order) // 2]] = 1
            self.macro_group_labels = group_labels
            labels = np.empty(self.N, dtype=np.int8)
            for group_index, group in enumerate(self.groups):
                labels[group] = group_labels[group_index]
            return labels

        labels = np.ones(self.N, dtype=np.int8)
        group_labels = np.ones(len(self.groups), dtype=np.int8)
        for group_index, group in enumerate(self.groups):
            group_labels[group_index] = 1 if group_index % 2 == 0 else -1
            labels[group] = group_labels[group_index]
        self.macro_group_labels = group_labels
        return labels

    def _independent_rng(self, child_index: int) -> np.random.Generator:
        if child_index not in {0, 1, 2, 3, 4}:
            raise ValueError("instance RNG child index must be 0, 1, 2, 3, or 4")
        root = np.random.SeedSequence(int(self.spec.graph_seed))
        # SeedSequence children 0--3 are unchanged by requesting a fifth child.
        return np.random.default_rng(root.spawn(5)[child_index])

    def _connected_random_3_regular_pairs(self) -> tuple[tuple[int, int], ...]:
        """Return one cached child-1 simple connected 3-regular quotient."""
        if self._cached_random_3_regular_pairs is not None:
            return self._cached_random_3_regular_pairs
        rng = self._independent_rng(1)
        pairs: set[tuple[int, int]] | None = None
        attempt_cap = 10_000
        for attempt in range(1, attempt_cap + 1):
            stubs = np.repeat(np.arange(len(self.groups), dtype=np.int64), 3)
            rng.shuffle(stubs)
            candidate_pairs: set[tuple[int, int]] = set()
            valid = True
            for position in range(0, len(stubs), 2):
                left = int(stubs[position])
                right = int(stubs[position + 1])
                pair = (min(left, right), max(left, right))
                if left == right or pair in candidate_pairs:
                    valid = False
                    break
                candidate_pairs.add(pair)
            if not valid:
                continue
            neighbors = [set() for _ in self.groups]
            for left, right in candidate_pairs:
                neighbors[left].add(right)
                neighbors[right].add(left)
            reached = {0}
            frontier = [0]
            while frontier:
                current = frontier.pop()
                for neighbor in neighbors[current]:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        frontier.append(neighbor)
            if len(reached) == len(self.groups):
                pairs = candidate_pairs
                self.coarse_topology_sampling_attempts = attempt
                self.coarse_topology_fallback_used = False
                break
        if pairs is None:
            raise RuntimeError(
                "could not sample a simple connected random 3-regular topology "
                f"within {attempt_cap} attempts"
            )
        expected_edge_count = 3 * len(self.groups) // 2
        if len(pairs) != expected_edge_count:
            raise RuntimeError(
                "independent random 3-regular construction produced wrong edge count"
            )
        self._cached_random_3_regular_pairs = tuple(sorted(pairs))
        return self._cached_random_3_regular_pairs

    def _sample_edge_strengths(self) -> np.ndarray:
        upper = np.triu_indices(self.N, k=1)
        if self.spec.edge_weight_distribution == "unit":
            strengths = np.zeros((self.N, self.N), dtype=np.float32)
            self.edge_weight_latent_normals = None
            values = np.ones(len(upper[0]), dtype=np.float32)
        else:
            strengths = np.zeros((self.N, self.N), dtype=np.float64)
            rng = self._independent_rng(3)
            latent_normals = rng.standard_normal(len(upper[0]))
            # Keep the pre-transform draws so paired CV cells can be audited
            # without reconstructing or advancing any RNG stream.
            self.edge_weight_latent_normals = latent_normals.astype(
                np.float64, copy=True,
            )
            coefficient_of_variation = float(
                self.spec.edge_weight_coefficient_of_variation
            )
            sigma = math.sqrt(math.log1p(coefficient_of_variation ** 2))
            mu = -0.5 * sigma * sigma
            values = np.exp(mu + sigma * latent_normals).astype(np.float64)
            if not bool(np.all(np.isfinite(values))) or not bool(np.all(values > 0.0)):
                raise RuntimeError("sampled edge strengths must be finite and positive")
        strengths[upper] = values
        strengths[(upper[1], upper[0])] = values
        return strengths

    def _sample_edge_signs(self) -> np.ndarray:
        signs = np.zeros((self.N, self.N), dtype=np.int8)
        upper = np.triu_indices(self.N, k=1)
        if self.spec.edge_sign_distribution == "all_positive":
            self.edge_sign_flip_uniforms = None
            values = np.ones(len(upper[0]), dtype=np.int8)
        else:
            rng = self._independent_rng(4)
            uniforms = rng.random(len(upper[0])).astype(np.float64)
            self.edge_sign_flip_uniforms = uniforms.copy()
            desired = (
                self.planted[upper[0]].astype(np.int8)
                * self.planted[upper[1]].astype(np.int8)
            )
            reliable = uniforms < float(self.spec.edge_sign_reliability)
            values = np.where(reliable, desired, -desired).astype(np.int8)
            if self.spec.intra_edge_sign_policy == "deterministic_positive":
                same_microblock = (
                    upper[0] // int(self.spec.group_size)
                    == upper[1] // int(self.spec.group_size)
                )
                values[same_microblock] = np.int8(1)
            if not bool(np.all(np.isin(values, (-1, 1)))):
                raise RuntimeError("sampled edge signs must be exactly -1 or +1")
        signs[upper] = values
        signs[(upper[1], upper[0])] = values
        return signs

    def _edge_probability(self, i: int, j: int) -> float:
        if self.spec.fine_support_model == "quotient_k22_plus_intra":
            raise RuntimeError("fixed quotient support has no Bernoulli edge probability")
        if self.spec.p_micro is not None and i // self.spec.group_size == j // self.spec.group_size:
            return float(self.spec.p_micro)
        return float(self.spec.p_in if self.planted[i] == self.planted[j] else self.spec.p_out)

    def _sample_graph(self) -> tuple[np.ndarray, np.ndarray]:
        if self.spec.fine_support_model == "quotient_k22_plus_intra":
            adjacency = np.zeros((self.N, self.N), dtype=np.float32)
            for group in self.groups:
                left, right = group
                adjacency[left, right] = adjacency[right, left] = 1.0
            for left_group, right_group in self._connected_random_3_regular_pairs():
                for left in self.groups[left_group]:
                    for right in self.groups[right_group]:
                        adjacency[left, right] = adjacency[right, left] = 1.0
            self.adjacency_uniforms = None
            centered = adjacency.copy()
            return adjacency, centered
        rng = (
            self._independent_rng(2)
            if self.uses_independent_instance_streams
            else np.random.default_rng(self.spec.graph_seed)
        )
        adjacency = np.zeros((self.N, self.N), dtype=np.float32)
        centered = np.zeros_like(adjacency)
        adjacency_uniforms: list[float] | None = (
            [] if self.spec.edge_sign_distribution != "all_positive" else None
        )
        for i in range(self.N):
            for j in range(i + 1, self.N):
                probability = self._edge_probability(i, j)
                uniform = float(rng.random())
                if adjacency_uniforms is not None:
                    adjacency_uniforms.append(uniform)
                edge = float(uniform < probability)
                adjacency[i, j] = adjacency[j, i] = edge
                value = edge - self.spec.centered_threshold
                centered[i, j] = centered[j, i] = value
        self.adjacency_uniforms = (
            np.asarray(adjacency_uniforms, dtype=np.float64)
            if adjacency_uniforms is not None else None
        )
        return adjacency, centered

    def _coarse_pair_matrix(self) -> np.ndarray:
        self.canonical_coarse_pair_coefficients_float64 = None
        if self.spec.fine_support_model == "quotient_k22_plus_intra":
            source = self.canonical_signed_weighted_adjacency_float64
            if source is None or source.dtype != np.float64:
                raise RuntimeError("literal quotient projection requires canonical float64 couplings")
            matrix64 = np.zeros((self.N, self.N), dtype=np.float64)
            for group in self.groups:
                left, right = group
                value = -float(source[left, right])
                matrix64[left, right] = matrix64[right, left] = value
            selected: list[tuple[int, int, float]] = []
            for left_index, right_index in self._connected_random_3_regular_pairs():
                left = self.groups[left_index]
                right = self.groups[right_index]
                realized_mean = float(np.mean(
                    source[np.ix_(left, right)], dtype=np.float64,
                ))
                selected.append((left_index, right_index, realized_mean))
                coefficient = -realized_mean
                for i in left:
                    for j in right:
                        matrix64[i, j] = matrix64[j, i] = coefficient
            self.selected_coarse_block_edges = [
                {
                    "left_group": left,
                    "right_group": right,
                    "block_mean": block_mean,
                    "coarse_pair_coefficient": -block_mean,
                }
                for left, right, block_mean in selected
            ]
            self.canonical_coarse_pair_coefficients_float64 = matrix64.copy()
            return matrix64.astype(np.float32)

        matrix = np.zeros((self.N, self.N), dtype=np.float32)
        for group in self.groups:
            for offset, i in enumerate(group):
                for j in group[offset + 1:]:
                    value = -float(self.spec.coarse_intra_weight)
                    matrix[i, j] = matrix[j, i] = value

        candidates: list[tuple[int, int, float]] = []
        centered_source = (
            self.canonical_centered_adjacency_float64
            if (
                self.spec.edge_weight_distribution != "unit"
                or self.spec.edge_sign_distribution != "all_positive"
            )
            else self.centered_adjacency
        )
        for left_index, left in enumerate(self.groups):
            for right_index in range(left_index + 1, len(self.groups)):
                right = self.groups[right_index]
                block_mean = float(np.mean(centered_source[np.ix_(left, right)]))
                candidates.append((left_index, right_index, block_mean))

        if self.spec.coarse_topology == "full":
            selected = candidates
        elif self.spec.coarse_topology == "maximum_spanning_forest":
            parent = list(range(len(self.groups)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            selected = []
            ordered = sorted(candidates, key=lambda item: (-abs(item[2]), item[0], item[1]))
            for left_index, right_index, block_mean in ordered:
                left_root = find(left_index)
                right_root = find(right_index)
                if left_root == right_root:
                    continue
                parent[right_root] = left_root
                selected.append((left_index, right_index, block_mean))
                if len(selected) == self.spec.coarse_forest_edges:
                    break
            if len(selected) != self.spec.coarse_forest_edges:
                raise RuntimeError("could not construct the requested coarse forest")
        elif self.spec.coarse_topology == "connected_top_k":
            parent = list(range(len(self.groups)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            ordered = sorted(candidates, key=lambda item: (-abs(item[2]), item[0], item[1]))
            selected = []
            selected_pairs: set[tuple[int, int]] = set()
            for left_index, right_index, block_mean in ordered:
                left_root = find(left_index)
                right_root = find(right_index)
                if left_root == right_root:
                    continue
                parent[right_root] = left_root
                selected.append((left_index, right_index, block_mean))
                selected_pairs.add((left_index, right_index))
                if len(selected) == len(self.groups) - 1:
                    break
            if len(selected) != len(self.groups) - 1:
                raise RuntimeError("could not construct the connected top-K spanning tree")
            if len(selected) < self.spec.coarse_forest_edges:
                for candidate in ordered:
                    pair = (candidate[0], candidate[1])
                    if pair in selected_pairs:
                        continue
                    selected.append(candidate)
                    if len(selected) == self.spec.coarse_forest_edges:
                        break
            if len(selected) != self.spec.coarse_forest_edges:
                raise RuntimeError("could not construct the requested connected top-K topology")
        else:
            # Child 1 is sampled once and cached.  The v5 opt-in branch reuses
            # exactly this quotient for fine support as well as the projection.
            pairs = self._connected_random_3_regular_pairs()
            candidate_by_pair = {
                (left, right): block_mean for left, right, block_mean in candidates
            }
            selected = [
                (left, right, candidate_by_pair[(left, right)])
                for left, right in sorted(pairs)
            ]

        self.selected_coarse_block_edges = [
            {"left_group": left, "right_group": right, "block_mean": block_mean}
            for left, right, block_mean in selected
        ]
        for left_index, right_index, block_mean in selected:
            value = -float(self.spec.coarse_inter_weight) * block_mean
            for i in self.groups[left_index]:
                for j in self.groups[right_index]:
                    matrix[i, j] = matrix[j, i] = value
        return matrix

    def _enumerate_costs(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Matrix products are substantially faster than one kernel per Ising pair.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        fine_matrix = torch.tensor(self.fine_pair_coefficients, dtype=self.real_dtype, device=self.device)
        coarse_matrix = torch.tensor(self.coarse_pair_coefficients, dtype=self.real_dtype, device=self.device)
        fine = torch.empty(self.dim, dtype=self.real_dtype, device=self.device)
        coarse = torch.empty_like(fine)
        shifts = torch.arange(self.N - 1, -1, -1, dtype=torch.int64, device=self.device)
        chunk_size = max(1, int(self.spec.cost_chunk_size))
        with torch.no_grad():
            for start in range(0, self.dim, chunk_size):
                stop = min(start + chunk_size, self.dim)
                indices = torch.arange(start, stop, dtype=torch.int64, device=self.device)
                bits = torch.bitwise_and(torch.bitwise_right_shift(indices[:, None], shifts[None, :]), 1)
                spins = 1.0 - 2.0 * bits.to(self.real_dtype)
                fine[start:stop] = 0.5 * torch.sum((spins @ fine_matrix) * spins, dim=1)
                coarse[start:stop] = 0.5 * torch.sum((spins @ coarse_matrix) * spins, dim=1)
        return fine, coarse

    @staticmethod
    def _label_index(labels: np.ndarray) -> int:
        index = 0
        for label in labels:
            index = (index << 1) | int(label < 0)
        return index

    def _diagnostics(self) -> None:
        with torch.no_grad():
            if self.feasible_indices is None:
                fine_domain = self.fine_cost
                coarse_domain = self.coarse_cost
                self.ground_energy = float(torch.min(self.fine_cost).item())
                ground_mask = torch.abs(self.fine_cost - self.ground_energy) <= self.energy_tolerance
                self.ground_indices = torch.nonzero(ground_mask, as_tuple=False).reshape(-1)
            else:
                fine_domain = self.fine_cost[self.feasible_indices]
                coarse_domain = self.coarse_cost[self.feasible_indices]
                self.ground_energy = float(torch.min(fine_domain).item())
                feasible_ground_mask = (
                    torch.abs(fine_domain - self.ground_energy) <= self.energy_tolerance
                )
                self.ground_indices = self.feasible_indices[feasible_ground_mask]
            self.num_ground_states = int(self.ground_indices.numel())
            if self.num_ground_states == 0:
                raise RuntimeError("exact ground set is empty")
            target_coarse_min = torch.min(self.coarse_cost[self.ground_indices])
            rank = torch.sum(coarse_domain < target_coarse_min - self.energy_tolerance)
            self.target_coarse_rank = int(rank.item())
            self.target_coarse_quantile = (
                self.target_coarse_rank / self.feasible_sector_dimension
            )
            coarse_ground = torch.min(coarse_domain)
            self.coarse_ground_energy = float(coarse_ground.item())
            coarse_ground_mask = torch.abs(coarse_domain - coarse_ground) <= self.energy_tolerance
            self.num_coarse_ground_states = int(torch.sum(coarse_ground_mask).item())
            self.target_is_coarse_ground = bool(
                torch.any(torch.abs(self.coarse_cost[self.ground_indices] - coarse_ground) <= self.energy_tolerance).item()
            )
            fine_centered = fine_domain - torch.mean(fine_domain)
            coarse_centered = coarse_domain - torch.mean(coarse_domain)
            denominator = torch.sqrt(torch.mean(fine_centered.square()) * torch.mean(coarse_centered.square()))
            self.coarse_fine_correlation = float(
                (torch.mean(fine_centered * coarse_centered) / denominator).item()
            ) if float(denominator.item()) > 0.0 else float("nan")
            planted_index = self._label_index(self.planted)
            flipped_index = self._label_index(-self.planted)
            ground_index_set = set(int(value) for value in self.ground_indices.detach().cpu().tolist())
            self.planted_pair_is_ground = planted_index in ground_index_set and flipped_index in ground_index_set

            ground_indices = sorted(ground_index_set)
            self.exact_ground_bitstrings = [
                format(index, f"0{self.N}b") for index in ground_indices
            ]
            num_groups = len(self.groups)
            block_uniform_indices: list[int] = []
            for logical_index in range(1 << num_groups):
                physical_index = 0
                for group_index, group in enumerate(self.groups):
                    bit = (logical_index >> (num_groups - 1 - group_index)) & 1
                    for _ in group:
                        physical_index = (physical_index << 1) | bit
                block_uniform_indices.append(physical_index)
            block_uniform_tensor = torch.tensor(
                block_uniform_indices, dtype=torch.int64, device=self.device,
            )
            if self.feasible_sector_mask is not None:
                block_uniform_tensor = block_uniform_tensor[
                    self.feasible_sector_mask[block_uniform_tensor]
                ]
                block_uniform_indices = [
                    int(value) for value in block_uniform_tensor.detach().cpu().tolist()
                ]
            block_uniform_index_set = set(block_uniform_indices)
            self.ground_state_block_uniform_count = sum(
                index in block_uniform_index_set for index in ground_indices
            )
            self.all_ground_states_block_uniform = (
                self.ground_state_block_uniform_count == self.num_ground_states
            )
            if not block_uniform_indices:
                self.best_block_uniform_energy = None
                self.best_block_uniform_gap_to_exact_ground = None
                self.next_distinct_block_uniform_energy = None
                self.next_block_uniform_energy_gap = None
                self.best_block_uniform_rank_within_feasible_sector = None
            else:
                block_uniform_energies = self.fine_cost[block_uniform_tensor]
                best = torch.min(block_uniform_energies)
                self.best_block_uniform_energy = float(best.item())
                gap_to_exact = self.best_block_uniform_energy - self.ground_energy
                if gap_to_exact < -self.energy_tolerance:
                    raise RuntimeError("block-uniform minimum lies below the exact ground energy")
                self.best_block_uniform_gap_to_exact_ground = float(max(0.0, gap_to_exact))
                higher = block_uniform_energies[
                    block_uniform_energies > best + self.energy_tolerance
                ]
                if int(higher.numel()) == 0:
                    self.next_distinct_block_uniform_energy = None
                    self.next_block_uniform_energy_gap = None
                else:
                    next_energy = float(torch.min(higher).item())
                    self.next_distinct_block_uniform_energy = next_energy
                    self.next_block_uniform_energy_gap = next_energy - self.best_block_uniform_energy
                self.best_block_uniform_rank_within_feasible_sector = int(torch.sum(
                    fine_domain < best - self.energy_tolerance
                ).item())
            self.block_uniform_basis_state_count = len(block_uniform_indices)
            self.block_uniform_effective_qubit_count = num_groups
            self.block_uniform_fraction_of_full_basis = (
                self.block_uniform_basis_state_count / self.dim
            )
            self.block_uniform_fraction_of_feasible_sector = (
                self.block_uniform_basis_state_count / self.feasible_sector_dimension
            )

    def summary(self) -> dict[str, Any]:
        spec_payload = instance_spec_payload(self.spec)
        summary = {
            **spec_payload,
            **sbm_model_metadata(self.spec),
            "dimension": self.dim,
            "ground_energy": self.ground_energy,
            "num_ground_states": self.num_ground_states,
            "ground_indices": [int(value) for value in self.ground_indices.detach().cpu().tolist()],
            "target_definition": target_definition(self.spec),
            "global_spin_flip_symmetry": True,
            "planted_pair_is_ground": self.planted_pair_is_ground,
            "target_coarse_rank": self.target_coarse_rank,
            "target_coarse_quantile": self.target_coarse_quantile,
            "target_is_coarse_ground": self.target_is_coarse_ground,
            "coarse_fine_correlation": self.coarse_fine_correlation,
            "coarse_two_body_term_count": self.coarse_two_body_term_count,
            "coarse_resource_two_body_term_count": self.coarse_resource_two_body_term_count,
            "coarse_ground_energy": self.coarse_ground_energy,
            "num_coarse_ground_states": self.num_coarse_ground_states,
            "selected_coarse_block_edges": self.selected_coarse_block_edges,
        }
        if (
            self.spec.macro_assignment != "alternating_contiguous_microblocks"
            or self.spec.coarse_topology == "independent_random_3_regular"
            or self.spec.feasible_hamming_weight is not None
        ):
            summary.update({
                "macro_group_labels": [int(value) for value in self.macro_group_labels.tolist()],
                "coarse_topology_vertex_order": self.coarse_topology_vertex_order,
                "coarse_topology_sampling_attempts": self.coarse_topology_sampling_attempts,
                "coarse_topology_fallback_used": self.coarse_topology_fallback_used,
                "exact_ground_bitstrings": self.exact_ground_bitstrings,
                "ground_state_block_uniform_count": self.ground_state_block_uniform_count,
                "all_ground_states_block_uniform": self.all_ground_states_block_uniform,
                "best_block_uniform_energy": self.best_block_uniform_energy,
                "best_block_uniform_gap_to_exact_ground": (
                    self.best_block_uniform_gap_to_exact_ground
                ),
                "next_distinct_block_uniform_energy": self.next_distinct_block_uniform_energy,
                "next_block_uniform_energy_gap": self.next_block_uniform_energy_gap,
                "block_uniform_basis_state_count": self.block_uniform_basis_state_count,
                "block_uniform_effective_qubit_count": self.block_uniform_effective_qubit_count,
                "block_uniform_fraction_of_full_basis": self.block_uniform_fraction_of_full_basis,
            })
        if self.spec.feasible_hamming_weight is not None:
            weight = int(self.spec.feasible_hamming_weight)
            sector_shift = float(self.spec.centered_threshold) * (
                ((self.N - 2 * weight) ** 2 - self.N) / 2
            )
            weighted = self.spec.edge_weight_distribution != "unit"
            signed = self.spec.edge_sign_distribution != "all_positive"
            if self.spec.fine_support_model == "quotient_k22_plus_intra":
                numerical_hamiltonian = (
                    "-sum_{inter_support}s_ij*w_ij*Z_iZ_j-"
                    "sum_{intra_support}lambda*w_ij*Z_iZ_j"
                )
                compiled_hamiltonian = numerical_hamiltonian
            elif signed and weighted:
                numerical_hamiltonian = "-sum_{i<j}(A_ij*s_ij*w_ij-threshold)Z_iZ_j"
                compiled_hamiltonian = (
                    "-sum_{(i,j)_in_observed_edges}s_ij*w_ij Z_iZ_j"
                )
            elif signed:
                numerical_hamiltonian = "-sum_{i<j}(A_ij*s_ij-threshold)Z_iZ_j"
                compiled_hamiltonian = "-sum_{(i,j)_in_observed_edges}s_ij Z_iZ_j"
            elif weighted:
                numerical_hamiltonian = "-sum_{i<j}(A_ij*w_ij-threshold)Z_iZ_j"
                compiled_hamiltonian = "-sum_{(i,j)_in_observed_edges}w_ij Z_iZ_j"
            else:
                numerical_hamiltonian = "-sum_{i<j}(A_ij-threshold)Z_iZ_j"
                compiled_hamiltonian = "-sum_{(i,j)_in_observed_edges}Z_iZ_j"
            summary.update({
                "diagnostic_domain": "fixed_hamming_weight_feasible_sector",
                "feasible_sector_dimension": self.feasible_sector_dimension,
                "feasible_sector_fraction_of_full_basis": (
                    self.feasible_sector_dimension / self.dim
                ),
                "initial_state_definition": (
                    "uniform_dicke_state_in_fixed_hamming_weight_sector"
                ),
                "initial_state_preparation_ru": self.initial_state_preparation_ru,
                "mixer_metadata": mixer_protocol_metadata(self.spec),
                "fine_resource_two_body_term_count": self.fine_resource_two_body_term_count,
                "fine_observed_edge_count": self.fine_observed_edge_count,
                "fine_hamiltonian_compilation": {
                    "numerical_hamiltonian": numerical_hamiltonian,
                    "compiled_hamiltonian": compiled_hamiltonian,
                    "relation_on_feasible_sector": (
                        "numerical_hamiltonian_equals_compiled_hamiltonian_plus_constant"
                    ),
                    "constant_energy_shift": sector_shift,
                    "constant_formula": (
                        "threshold*(((N-2K)^2-N)/2);equals-threshold*N/2-at-K=N/2"
                    ),
                    "nonedge_phase_terms_compiled_out": True,
                    "target_and_expected_energy_gradients_unchanged": True,
                    "unitary_dynamics_differ_only_by_global_phase": True,
                },
                "block_uniform_fraction_of_feasible_sector": (
                    self.block_uniform_fraction_of_feasible_sector
                ),
                "best_block_uniform_rank_within_feasible_sector": (
                    self.best_block_uniform_rank_within_feasible_sector
                ),
            })
            if self.spec.fine_support_model == "quotient_k22_plus_intra":
                summary["fine_hamiltonian_compilation"].update({
                    "relation_on_feasible_sector": "numerical_and_compiled_hamiltonians_identical",
                    "constant_energy_shift": 0.0,
                    "constant_formula": "not_applicable_no_centering_or_tau_term",
                    "support_is_exact_connected_7_regular_graph": True,
                    "canonical_fine_coefficient_convention": "C_f(i,j)=-J_ij",
                    "canonical_coarse_coefficient_convention": (
                        "intra_C_c=-lambda*w;inter_C_c=mean_of_four_C_f=-mean_of_four_J"
                    ),
                    "canonical_coarse_pair_coefficients_attribute": (
                        "canonical_coarse_pair_coefficients_float64"
                    ),
                    "nonedge_phase_terms_compiled_out": False,
                    "unitary_dynamics_differ_only_by_global_phase": False,
                })
        if self.spec.edge_weight_distribution != "unit":
            upper = np.triu_indices(self.N, k=1)
            all_weights = self.edge_strengths[upper].astype(np.float64)
            observed_mask = self.adjacency[upper] > 0.5
            observed_weights = all_weights[observed_mask]
            observed_interaction_magnitudes = (
                np.abs(self.canonical_signed_weighted_adjacency_float64[upper][observed_mask])
                if self.spec.fine_support_model == "quotient_k22_plus_intra"
                else observed_weights
            )
            latent = self.edge_weight_latent_normals
            if latent is None or len(latent) != len(all_weights):
                raise RuntimeError("weighted branch lacks its complete latent-normal audit trail")

            def distribution_stats(values: np.ndarray) -> dict[str, Any]:
                if values.size == 0:
                    return {
                        "count": 0, "minimum": None, "maximum": None,
                        "mean": None, "population_standard_deviation": None,
                        "population_coefficient_of_variation": None,
                        "sum": 0.0, "sum_of_squares": 0.0,
                        "effective_edge_count": 0.0,
                    }
                total = float(np.sum(values, dtype=np.float64))
                sum_squares = float(np.sum(values * values, dtype=np.float64))
                mean = float(np.mean(values, dtype=np.float64))
                standard_deviation = float(np.std(values, dtype=np.float64))
                return {
                    "count": int(values.size),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "mean": mean,
                    "population_standard_deviation": standard_deviation,
                    "population_coefficient_of_variation": (
                        standard_deviation / mean if mean > 0.0 else None
                    ),
                    "sum": total,
                    "sum_of_squares": sum_squares,
                    "effective_edge_count": (
                        total * total / sum_squares if sum_squares > 0.0 else 0.0
                    ),
                }

            all_stats = distribution_stats(all_weights)
            observed_stats = distribution_stats(observed_weights)
            observed_interaction_stats = distribution_stats(
                observed_interaction_magnitudes
            )
            summary["edge_weight_diagnostics"] = {
                "distribution": self.spec.edge_weight_distribution,
                "configured_mean": 1.0,
                "configured_coefficient_of_variation": float(
                    self.spec.edge_weight_coefficient_of_variation
                ),
                "sampling_stream": "numpy_seedsequence_child_3_pcg64",
                "independent_of_label_topology_and_adjacency_streams": True,
                "pre_sampled_for_all_unordered_node_pairs": True,
                "pre_sample_order": "ascending_lexicographic_(left_node,right_node)",
                "sampled_before_binary_adjacency_is_applied": True,
                "per_instance_normalization_applied": False,
                "conditional_resampling_applied": False,
                "canonical_coefficient_precision": "numpy_float64_before_simulation_cast",
                "simulation_coefficient_precision": "numpy_float32_then_torch_float32",
                "canonical_weighted_adjacency_attribute": (
                    "canonical_weighted_adjacency_float64"
                ),
                "all_unordered_pair_strengths": all_stats,
                "observed_edge_strengths": observed_stats,
                "observed_edge_interaction_magnitudes": observed_interaction_stats,
                "maximum_observed_weight_over_total_observed_weight": (
                    float(observed_stats["maximum"]) / float(observed_stats["sum"])
                    if observed_stats["count"] and float(observed_stats["sum"]) > 0.0
                    else None
                ),
                "pair_records": [
                    {
                        "left_node": int(left),
                        "right_node": int(right),
                        "latent_standard_normal": float(latent[offset]),
                        "edge_strength": float(all_weights[offset]),
                        "binary_edge_observed": bool(observed_mask[offset]),
                        "weighted_adjacency": (
                            float(self.canonical_weighted_adjacency_float64[
                                int(left), int(right)
                            ]) if observed_mask[offset] else 0.0
                        ),
                        **({
                            "interaction_scale": float(self.interaction_scales[
                                int(left), int(right)
                            ]),
                            "absolute_interaction_magnitude": float(abs(
                                self.canonical_signed_weighted_adjacency_float64[
                                    int(left), int(right)
                                ]
                            )),
                        } if self.spec.fine_support_model == "quotient_k22_plus_intra" else {}),
                    }
                    for offset, (left, right) in enumerate(zip(upper[0], upper[1]))
                ],
            }
        if self.spec.edge_sign_distribution != "all_positive":
            upper = np.triu_indices(self.N, k=1)
            observed_actual = self.adjacency[upper] > 0.5
            if self.spec.fine_support_model == "quotient_k22_plus_intra":
                quotient_pairs = set(self._connected_random_3_regular_pairs())
                summary["support_sampling_diagnostics"] = {
                    "support_model": self.spec.fine_support_model,
                    "quotient_sampling_stream": "numpy_seedsequence_child_1_pcg64",
                    "adjacency_child_2_status": "reserved_and_unused",
                    "quotient_reused_identically_for_fine_support_and_coarse_projection": True,
                    "quotient_edges": [list(pair) for pair in sorted(quotient_pairs)],
                    "support_rule": (
                        "all_12_intra_K2_edges_plus_all_four_K2,2_edges_for_each_"
                        "of_18_quotient_edges"
                    ),
                    "pair_records": [
                        {
                            "left_node": int(left),
                            "right_node": int(right),
                            "left_group": int(left) // int(self.spec.group_size),
                            "right_group": int(right) // int(self.spec.group_size),
                            "same_microblock": bool(
                                int(left) // int(self.spec.group_size)
                                == int(right) // int(self.spec.group_size)
                            ),
                            "quotient_edge": bool(
                                (
                                    min(int(left) // int(self.spec.group_size),
                                        int(right) // int(self.spec.group_size)),
                                    max(int(left) // int(self.spec.group_size),
                                        int(right) // int(self.spec.group_size)),
                                ) in quotient_pairs
                            ),
                            "binary_edge_observed": bool(observed_actual[offset]),
                        }
                        for offset, (left, right) in enumerate(zip(upper[0], upper[1]))
                    ],
                }
            else:
                adjacency_uniforms = self.adjacency_uniforms
                if adjacency_uniforms is None or len(adjacency_uniforms) != len(upper[0]):
                    raise RuntimeError("signed branch lacks its complete adjacency-uniform audit trail")
                adjacency_probabilities = np.asarray([
                    self._edge_probability(int(left), int(right))
                    for left, right in zip(upper[0], upper[1])
                ], dtype=np.float64)
                observed_from_uniform = adjacency_uniforms < adjacency_probabilities
                if not bool(np.array_equal(observed_from_uniform, observed_actual)):
                    raise RuntimeError("stored adjacency uniforms do not reconstruct the graph")
                summary["adjacency_sampling_diagnostics"] = {
                    "sampling_stream": "numpy_seedsequence_child_2_pcg64",
                    "pre_sample_order": "ascending_lexicographic_(left_node,right_node)",
                    "all_unordered_pair_uniforms_recorded": True,
                    "edge_rule": "A_ij=1_if_and_only_if_u_ij<pair_probability",
                    "nonmicro_probability_independent_of_planted_macro_label": bool(
                        float(self.spec.p_in) == float(self.spec.p_out)
                    ),
                    "conditional_resampling_applied": False,
                    "pair_records": [
                        {
                            "left_node": int(left),
                            "right_node": int(right),
                            "adjacency_uniform": float(adjacency_uniforms[offset]),
                            "pair_probability": float(adjacency_probabilities[offset]),
                            "same_microblock": bool(
                                int(left) // int(self.spec.group_size)
                                == int(right) // int(self.spec.group_size)
                            ),
                            "same_planted_macro_community": bool(
                                int(self.planted[int(left)]) == int(self.planted[int(right)])
                            ),
                            "binary_edge_observed": bool(observed_actual[offset]),
                        }
                        for offset, (left, right) in enumerate(zip(upper[0], upper[1]))
                    ],
                }
            uniforms = self.edge_sign_flip_uniforms
            if uniforms is None or len(uniforms) != len(upper[0]):
                raise RuntimeError("signed branch lacks its complete sign-uniform audit trail")
            desired = (
                self.planted[upper[0]].astype(np.int8)
                * self.planted[upper[1]].astype(np.int8)
            )
            actual = self.edge_signs[upper].astype(np.int8)
            observed_mask = self.adjacency[upper] > 0.5
            desired_match = actual == desired
            observed_count = int(np.sum(observed_mask))
            summary["edge_sign_diagnostics"] = {
                "distribution": self.spec.edge_sign_distribution,
                "configured_desired_sign_reliability": float(
                    self.spec.edge_sign_reliability
                ),
                "desired_sign_rule": (
                    "+1_within_planted_macro_community;-1_across_planted_macro_communities"
                ),
                "intra_edge_sign_policy": self.spec.intra_edge_sign_policy,
                "configured_reliability_applies_to": (
                    "inter_microblock_pairs_only"
                    if self.spec.intra_edge_sign_policy == "deterministic_positive"
                    else "all_unordered_pairs"
                ),
                "sampling_stream": "numpy_seedsequence_child_4_pcg64",
                "random_draws_independent_of_label_topology_adjacency_and_magnitude_streams": True,
                "pre_sampled_for_all_unordered_node_pairs": True,
                "pre_sample_order": "ascending_lexicographic_(left_node,right_node)",
                "sampled_before_binary_adjacency_is_applied": (
                    self.spec.fine_support_model == "independent_pair_bernoulli"
                ),
                "sampled_before_fixed_support_is_applied": (
                    self.spec.fine_support_model == "quotient_k22_plus_intra"
                ),
                "conditional_resampling_applied": False,
                "all_unordered_pair_desired_sign_match_fraction": float(np.mean(
                    desired_match.astype(np.float64)
                )),
                "observed_edge_desired_sign_match_fraction": (
                    float(np.mean(desired_match[observed_mask].astype(np.float64)))
                    if observed_count else None
                ),
                "all_unordered_pair_positive_sign_count": int(np.sum(actual > 0)),
                "all_unordered_pair_negative_sign_count": int(np.sum(actual < 0)),
                "observed_edge_positive_sign_count": int(np.sum(
                    (actual > 0) & observed_mask
                )),
                "observed_edge_negative_sign_count": int(np.sum(
                    (actual < 0) & observed_mask
                )),
                "canonical_signed_weighted_adjacency_attribute": (
                    "canonical_signed_weighted_adjacency_float64"
                ),
                "pair_records": [
                    {
                        "left_node": int(left),
                        "right_node": int(right),
                        "sign_flip_uniform": float(uniforms[offset]),
                        "desired_edge_sign": int(desired[offset]),
                        "actual_edge_sign": int(actual[offset]),
                        "actual_matches_desired": bool(desired_match[offset]),
                        "same_microblock": bool(
                            int(left) // int(self.spec.group_size)
                            == int(right) // int(self.spec.group_size)
                        ),
                        "sign_uniform_used_for_actual_sign": bool(
                            self.spec.intra_edge_sign_policy != "deterministic_positive"
                            or int(left) // int(self.spec.group_size)
                            != int(right) // int(self.spec.group_size)
                        ),
                        "binary_edge_observed": bool(observed_mask[offset]),
                        "signed_weighted_adjacency": (
                            float(self.canonical_signed_weighted_adjacency_float64[
                                int(left), int(right)
                            ])
                            if observed_mask[offset] else 0.0
                        ),
                    }
                    for offset, (left, right) in enumerate(zip(upper[0], upper[1]))
                ],
            }
        return summary


class StatevectorEngine:
    def __init__(self, instance: SBMInstance) -> None:
        self.instance = instance
        self.device = instance.device
        if instance.spec.mixer_type == "ring_xy":
            self.ring_xy_even_edges = tuple(
                (left, left + 1) for left in range(0, instance.N, 2)
            )
            self.ring_xy_odd_edges = tuple(
                (left, (left + 1) % instance.N) for left in range(1, instance.N, 2)
            )
            self.ring_xy_edges = self.ring_xy_even_edges + self.ring_xy_odd_edges
        else:
            self.ring_xy_even_edges = ()
            self.ring_xy_odd_edges = ()
            self.ring_xy_edges = ()

    @staticmethod
    def phase(angle: torch.Tensor) -> torch.Tensor:
        return torch.polar(torch.ones_like(angle), -angle)

    @staticmethod
    def _mixer_beta_vector(states: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2:
            raise ValueError("mixer states must have shape (batch, dimension)")
        batch = states.shape[0]
        if betas.ndim == 0:
            return betas.expand(batch)
        if betas.ndim == 1 and betas.numel() == 1:
            return betas.expand(batch)
        if betas.ndim != 1 or betas.shape[0] != batch:
            raise ValueError("mixer beta must be scalar or contain one value per batch row")
        return betas

    def apply_x_mixer(self, states: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        betas = self._mixer_beta_vector(states, betas)
        cosine = torch.cos(betas).reshape(batch, 1, 1)
        sine = torch.sin(betas).reshape(batch, 1, 1)
        for qubit in range(self.instance.N):
            view = states.reshape(batch, -1, 2, 1 << qubit)
            zero = view[:, :, 0, :]
            one = view[:, :, 1, :]
            states = torch.stack(
                (cosine * zero - 1j * sine * one, cosine * one - 1j * sine * zero),
                dim=2,
            ).reshape(batch, self.instance.dim)
        return states

    def _apply_ring_xy_edge(self, states: torch.Tensor, betas: torch.Tensor,
                            edge: tuple[int, int]) -> torch.Tensor:
        """Apply exp[-i beta (XX+YY)/2] using graph-node/MSB bit ordering."""
        batch = states.shape[0]
        betas = self._mixer_beta_vector(states, betas)
        cosine = torch.cos(betas)
        sine = torch.sin(betas)
        left, right = edge
        if right == left + 1:
            view = states.reshape(batch, 1 << left, 4, 1 << (self.instance.N - right - 1))
            cosine = cosine.reshape(batch, 1, 1)
            sine = sine.reshape(batch, 1, 1)
            zero_zero = view[:, :, 0, :]
            zero_one = view[:, :, 1, :]
            one_zero = view[:, :, 2, :]
            one_one = view[:, :, 3, :]
            return torch.stack((
                zero_zero,
                cosine * zero_one - 1j * sine * one_zero,
                cosine * one_zero - 1j * sine * zero_one,
                one_one,
            ), dim=2).reshape(batch, self.instance.dim)
        if {left, right} == {0, self.instance.N - 1}:
            view = states.reshape(batch, 2, 1 << (self.instance.N - 2), 2)
            cosine = cosine.reshape(batch, 1)
            sine = sine.reshape(batch, 1)
            zero_zero = view[:, 0, :, 0]
            zero_one = view[:, 0, :, 1]
            one_zero = view[:, 1, :, 0]
            one_one = view[:, 1, :, 1]
            row_zero = torch.stack((
                zero_zero,
                cosine * zero_one - 1j * sine * one_zero,
            ), dim=2)
            row_one = torch.stack((
                cosine * one_zero - 1j * sine * zero_one,
                one_one,
            ), dim=2)
            return torch.stack((row_zero, row_one), dim=1).reshape(batch, self.instance.dim)
        raise ValueError(f"ring_xy received a non-ring edge: {edge}")

    def _ring_xy_edge_generator(self, states: torch.Tensor,
                                edge: tuple[int, int]) -> torch.Tensor:
        """Apply (XX+YY)/2 for one ring edge, without the exponential."""
        batch = states.shape[0]
        left, right = edge
        if right == left + 1:
            view = states.reshape(batch, 1 << left, 4, 1 << (self.instance.N - right - 1))
            zero = torch.zeros_like(view[:, :, 0, :])
            return torch.stack((
                zero, view[:, :, 2, :], view[:, :, 1, :], zero,
            ), dim=2).reshape(batch, self.instance.dim)
        if {left, right} == {0, self.instance.N - 1}:
            view = states.reshape(batch, 2, 1 << (self.instance.N - 2), 2)
            zero = torch.zeros_like(view[:, 0, :, 0])
            row_zero = torch.stack((zero, view[:, 1, :, 0]), dim=2)
            row_one = torch.stack((view[:, 0, :, 1], zero), dim=2)
            return torch.stack((row_zero, row_one), dim=1).reshape(batch, self.instance.dim)
        raise ValueError(f"ring_xy received a non-ring edge: {edge}")

    def apply_ring_xy_mixer(self, states: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        betas = self._mixer_beta_vector(states, betas)
        for edge in self.ring_xy_edges:
            states = self._apply_ring_xy_edge(states, betas, edge)
        return states

    def apply_problem_mixer(self, states: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        if self.instance.spec.mixer_type == "x":
            return self.apply_x_mixer(states, betas)
        if self.instance.spec.mixer_type == "ring_xy":
            return self.apply_ring_xy_mixer(states, betas)
        raise ValueError(f"unsupported mixer_type: {self.instance.spec.mixer_type}")

    @staticmethod
    def apply_projector_mixer(states: torch.Tensor, betas: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        overlap = torch.matmul(states, reference.conj()).unsqueeze(1)
        phase = torch.exp(-1j * betas).unsqueeze(1)
        return phase * states - (phase - 1.0) * overlap * reference.unsqueeze(0)

    def _setup(self, method: str, batch: int, reference: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if method in {"stage1", "standard_qaoa"}:
            initial = self.instance.uniform_state
        elif method in {"warm_start_qaoa", "lp_qaoa"} and reference is not None:
            initial = reference
        else:
            raise ValueError(f"invalid method/reference combination: {method}")
        hamiltonian = self.instance.coarse_cost if method == "stage1" else self.instance.fine_cost
        return initial.unsqueeze(0).expand(batch, self.instance.dim).clone(), hamiltonian

    def states(self, method: str, gammas: torch.Tensor, betas: torch.Tensor,
               reference: torch.Tensor | None = None) -> torch.Tensor:
        batch, depth = gammas.shape
        states, hamiltonian = self._setup(method, batch, reference)
        for layer in range(depth):
            states = states * self.phase(gammas[:, layer:layer + 1] * hamiltonian.unsqueeze(0))
            if method == "lp_qaoa":
                assert reference is not None
                states = self.apply_projector_mixer(states, betas[:, layer], reference)
            else:
                states = self.apply_problem_mixer(states, betas[:, layer])
        return states

    def checkpointed_states(self, method: str, gammas: torch.Tensor, betas: torch.Tensor,
                            reference: torch.Tensor | None = None) -> torch.Tensor:
        batch, depth = gammas.shape
        states, hamiltonian = self._setup(method, batch, reference)

        def layer_step(current: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
            updated = current * self.phase(gamma.unsqueeze(1) * hamiltonian.unsqueeze(0))
            if method == "lp_qaoa":
                assert reference is not None
                return self.apply_projector_mixer(updated, beta, reference)
            if self.instance.spec.mixer_type == "x":
                return _BatchedSharedAngleXMixer.apply(updated, beta, self)
            return _BatchedSharedAngleRingXYMixer.apply(updated, beta, self)

        for layer in range(depth):
            states = checkpoint(
                layer_step, states, gammas[:, layer], betas[:, layer], use_reentrant=False,
            )
        return states

    @staticmethod
    def expected_energies(states: torch.Tensor, hamiltonian: torch.Tensor) -> torch.Tensor:
        return torch.sum(states.abs().square() * hamiltonian.unsqueeze(0), dim=1)

    def objective(self, method: str, states: torch.Tensor) -> torch.Tensor:
        hamiltonian = self.instance.coarse_cost if method == "stage1" else self.instance.fine_cost
        return self.expected_energies(states, hamiltonian)


class _BatchedSharedAngleXMixer(torch.autograd.Function):
    """Low-memory derivative for exp(-i beta_b sum_j X_j), one beta per batch row."""

    @staticmethod
    def forward(ctx, states: torch.Tensor, betas: torch.Tensor,
                engine: StatevectorEngine) -> torch.Tensor:
        with torch.no_grad():
            output = engine.apply_x_mixer(states, betas)
        ctx.engine = engine
        ctx.save_for_backward(output, betas)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, betas = ctx.saved_tensors
        engine: StatevectorEngine = ctx.engine
        with torch.no_grad():
            generator_output = torch.zeros_like(output)
            batch = output.shape[0]
            for qubit in range(engine.instance.N):
                view = output.reshape(batch, -1, 2, 1 << qubit)
                generator_output.add_(view.flip(2).reshape_as(output))
            derivative = -1j * generator_output
            grad_betas = torch.real(torch.sum(grad_output.conj() * derivative, dim=1))
            grad_states = engine.apply_x_mixer(grad_output, -betas)
        return grad_states, grad_betas, None


class _BatchedSharedAngleRingXYMixer(torch.autograd.Function):
    """Adjoint derivative of the ordered two-matching ring-XY product."""

    @staticmethod
    def forward(ctx, states: torch.Tensor, betas: torch.Tensor,
                engine: StatevectorEngine) -> torch.Tensor:
        beta_vector = engine._mixer_beta_vector(states, betas)
        with torch.no_grad():
            output = engine.apply_ring_xy_mixer(states, beta_vector)
        ctx.engine = engine
        ctx.beta_input_shape = tuple(betas.shape)
        ctx.save_for_backward(output, beta_vector)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, betas = ctx.saved_tensors
        engine: StatevectorEngine = ctx.engine
        with torch.no_grad():
            current = output
            adjoint = grad_output
            grad_beta_rows = torch.zeros_like(betas)
            for edge in reversed(engine.ring_xy_edges):
                generator_output = engine._ring_xy_edge_generator(current, edge)
                derivative = -1j * generator_output
                grad_beta_rows.add_(torch.real(torch.sum(
                    adjoint.conj() * derivative, dim=1,
                )))
                current = engine._apply_ring_xy_edge(current, -betas, edge)
                adjoint = engine._apply_ring_xy_edge(adjoint, -betas, edge)
            if ctx.beta_input_shape == ():
                grad_betas = torch.sum(grad_beta_rows)
            elif ctx.beta_input_shape == (1,) and grad_beta_rows.numel() != 1:
                grad_betas = torch.sum(grad_beta_rows).reshape(1)
            else:
                grad_betas = grad_beta_rows
        return adjoint, grad_betas, None


def initial_angle_rows(restarts: int, depth: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate restart-indexed starts whose prefixes do not depend on batch size."""
    if restarts <= 0 or depth <= 0:
        raise ValueError("restarts and depth must be positive")
    gammas = np.empty((restarts, depth), dtype=np.float32)
    betas = np.empty_like(gammas)
    stream_seeds: list[int] = []
    for restart in range(restarts):
        stream_seed = stable_seed(seed, "initial_angles", restart)
        stream_seeds.append(stream_seed)
        generator = np.random.Generator(np.random.PCG64(stream_seed))
        values = generator.uniform(-math.pi, math.pi, size=2 * depth).astype(np.float32)
        gammas[restart] = values[:depth]
        betas[restart] = values[depth:]
    # The first row is a fixed smooth schedule; every other row has its own PCG64 stream.
    gammas[0] = np.linspace(0.03, 0.30, depth, dtype=np.float32)
    betas[0] = np.linspace(0.30, 0.03, depth, dtype=np.float32)
    return gammas, betas, stream_seeds


def spsa_perturbation_rows(restarts: int, depth: int, steps: int,
                           seed: int) -> tuple[np.ndarray, list[int]]:
    """Generate independent per-restart Rademacher streams for SPSA."""
    if restarts <= 0 or depth <= 0 or steps <= 0:
        raise ValueError("restarts, depth, and steps must be positive")
    perturbations = np.empty((steps, restarts, 2 * depth), dtype=np.float32)
    stream_seeds: list[int] = []
    for restart in range(restarts):
        stream_seed = stable_seed(seed, "spsa_perturbations", restart)
        stream_seeds.append(stream_seed)
        generator = np.random.Generator(np.random.PCG64(stream_seed))
        draws = generator.integers(0, 2, size=(steps, 2 * depth), dtype=np.int8)
        perturbations[:, restart, :] = 2.0 * draws.astype(np.float32) - 1.0
    return perturbations, stream_seeds


def _initial_angles(restarts: int, depth: int, dtype: torch.dtype,
                    device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    gamma_rows, beta_rows, stream_seeds = initial_angle_rows(restarts, depth, seed)
    gammas = torch.as_tensor(gamma_rows, dtype=dtype, device=device).clone()
    betas = torch.as_tensor(beta_rows, dtype=dtype, device=device).clone()
    return gammas, betas, stream_seeds


def _capture_best(losses: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor,
                  best_losses: torch.Tensor, best_gammas: torch.Tensor,
                  best_betas: torch.Tensor, best_steps: torch.Tensor, step: int) -> None:
    improved = losses.detach() < best_losses
    best_losses.copy_(torch.where(improved, losses.detach(), best_losses))
    best_gammas.copy_(torch.where(improved.unsqueeze(1), gammas.detach(), best_gammas))
    best_betas.copy_(torch.where(improved.unsqueeze(1), betas.detach(), best_betas))
    best_steps.copy_(torch.where(improved, torch.full_like(best_steps, int(step)), best_steps))


def _optimize_adam_batch(
    engine: StatevectorEngine,
    method: str,
    depth: int,
    reference: torch.Tensor | None,
    spec: OptimizerSpec,
    seed: int,
    *,
    gamma_rows: np.ndarray | None = None,
    beta_rows: np.ndarray | None = None,
    initialization_stream_seeds: list[int] | None = None,
    restart_offset: int = 0,
) -> dict[str, Any]:
    """Optimize one contiguous batch of globally indexed Adam restarts."""
    supplied = (
        gamma_rows is not None,
        beta_rows is not None,
        initialization_stream_seeds is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError("Adam restart rows and their stream seeds must be supplied together")
    if all(supplied):
        assert gamma_rows is not None
        assert beta_rows is not None
        assert initialization_stream_seeds is not None
        if (
            gamma_rows.ndim != 2
            or beta_rows.shape != gamma_rows.shape
            or gamma_rows.shape[1] != depth
            or gamma_rows.shape[0] <= 0
            or len(initialization_stream_seeds) != gamma_rows.shape[0]
        ):
            raise ValueError("invalid pre-generated Adam restart rows")
        gammas = torch.as_tensor(
            gamma_rows, dtype=engine.instance.real_dtype, device=engine.device,
        ).clone()
        betas = torch.as_tensor(
            beta_rows, dtype=engine.instance.real_dtype, device=engine.device,
        ).clone()
        initialization_stream_seeds = list(initialization_stream_seeds)
    else:
        if restart_offset != 0:
            raise ValueError("a nonzero restart offset requires pre-generated Adam restart rows")
        gammas, betas, initialization_stream_seeds = _initial_angles(
            spec.restarts, depth, engine.instance.real_dtype, engine.device, seed,
        )
    batch_restarts = int(gammas.shape[0])
    initial_gamma = gammas.detach().cpu().tolist()
    initial_beta = betas.detach().cpu().tolist()
    gammas.requires_grad_(True)
    betas.requires_grad_(True)
    optimizer = torch.optim.Adam((gammas, betas), lr=spec.adam_lr)
    best_losses = torch.full((batch_restarts,), float("inf"), dtype=engine.instance.real_dtype, device=engine.device)
    best_gammas = gammas.detach().clone()
    best_betas = betas.detach().clone()
    best_steps = torch.zeros(batch_restarts, dtype=torch.int64, device=engine.device)
    trace_steps = sorted(set((0, spec.steps // 4, spec.steps // 2, 3 * spec.steps // 4, spec.steps)))
    traces: list[list[dict[str, float | int]]] = [[] for _ in range(batch_restarts)]
    started = time.time()

    for step in range(spec.steps):
        optimizer.zero_grad(set_to_none=True)
        states = engine.checkpointed_states(method, gammas, betas, reference)
        losses = engine.objective(method, states)
        _capture_best(losses, gammas, betas, best_losses, best_gammas, best_betas, best_steps, step)
        if step in trace_steps:
            if not bool(torch.all(torch.isfinite(losses)).item()):
                raise FloatingPointError("Adam produced a nonfinite expected energy")
            values = losses.detach().cpu().tolist()
            best_values = best_losses.detach().cpu().tolist()
            for restart in range(batch_restarts):
                traces[restart].append({"step": step, "loss": float(values[restart]), "best_loss": float(best_values[restart])})
        torch.sum(losses).backward()
        if gammas.grad is None or betas.grad is None:
            raise FloatingPointError("Adam gradient is missing")
        if step in trace_steps:
            if not bool(torch.all(torch.isfinite(gammas.grad)).item() and torch.all(torch.isfinite(betas.grad)).item()):
                raise FloatingPointError("Adam produced a nonfinite gradient")
        optimizer.step()

    with torch.no_grad():
        terminal_states = engine.states(method, gammas, betas, reference)
        terminal_losses = engine.objective(method, terminal_states)
        if not bool(torch.all(torch.isfinite(terminal_losses)).item()):
            raise FloatingPointError("Adam produced a nonfinite terminal expected energy")
        _capture_best(
            terminal_losses, gammas, betas, best_losses, best_gammas, best_betas, best_steps, spec.steps,
        )
    terminal_values = terminal_losses.detach().cpu().tolist()
    best_values = best_losses.detach().cpu().tolist()
    best_step_values = best_steps.detach().cpu().tolist()
    for restart in range(batch_restarts):
        traces[restart].append({"step": spec.steps, "loss": float(terminal_values[restart]), "best_loss": float(best_values[restart])})
    selected_local = int(torch.argmin(best_losses).item())
    selected_global = restart_offset + selected_local
    return {
        "optimizer": "adam",
        "initialization_seed": int(seed),
        "initialization_scheme": INITIALIZATION_SCHEME,
        "initialization_stream_seeds": initialization_stream_seeds,
        "initial_gamma": [[float(value) for value in row] for row in initial_gamma],
        "initial_beta": [[float(value) for value in row] for row in initial_beta],
        "objective": "expected_coarse_energy" if method == "stage1" else "expected_fine_energy",
        "steps": spec.steps,
        "restarts": batch_restarts,
        "learning_rate": spec.adam_lr,
        "selected_restart": selected_global,
        "selected_loss": float(best_losses[selected_local].item()),
        "gamma": [float(value) for value in best_gammas[selected_local].detach().cpu().tolist()],
        "beta": [float(value) for value in best_betas[selected_local].detach().cpu().tolist()],
        "restart_results": [
            {
                "restart": restart_offset + restart,
                "best_loss": float(best_values[restart]),
                "best_step": int(best_step_values[restart]),
                "trace": traces[restart],
            }
            for restart in range(batch_restarts)
        ],
        "elapsed_seconds": time.time() - started,
        "objective_evaluations_per_restart": spec.steps + 1,
    }


def optimize_adam(engine: StatevectorEngine, method: str, depth: int,
                  reference: torch.Tensor | None, spec: OptimizerSpec,
                  seed: int, restart_batch_size: int | None = None) -> dict[str, Any]:
    """Run Adam, optionally trading time for memory without changing logical restarts."""
    if restart_batch_size is None or restart_batch_size >= spec.restarts:
        if restart_batch_size is not None and restart_batch_size <= 0:
            raise ValueError("Adam restart batch size must be positive")
        return _optimize_adam_batch(engine, method, depth, reference, spec, seed)
    if restart_batch_size <= 0:
        raise ValueError("Adam restart batch size must be positive")

    # Generate all logical restart rows once.  In particular, only global
    # restart zero receives the fixed smooth schedule; later microbatches must
    # not create another local restart zero or reuse an earlier PCG64 stream.
    gamma_rows, beta_rows, stream_seeds = initial_angle_rows(
        spec.restarts, depth, seed,
    )
    batches: list[dict[str, Any]] = []
    started = time.time()
    for restart_offset in range(0, spec.restarts, restart_batch_size):
        stop = min(spec.restarts, restart_offset + restart_batch_size)
        batch = _optimize_adam_batch(
            engine,
            method,
            depth,
            reference,
            spec,
            seed,
            gamma_rows=gamma_rows[restart_offset:stop],
            beta_rows=beta_rows[restart_offset:stop],
            initialization_stream_seeds=stream_seeds[restart_offset:stop],
            restart_offset=restart_offset,
        )
        batches.append(batch)
        if stop < spec.restarts and engine.device.type == "cuda":
            # Every value retained in ``batch`` is already on the host.  Return
            # unused allocator blocks before the next statevector microbatch.
            torch.cuda.empty_cache()

    winner = min(
        batches,
        key=lambda batch: (float(batch["selected_loss"]), int(batch["selected_restart"])),
    )
    restart_results = [
        restart_result
        for batch in batches
        for restart_result in batch["restart_results"]
    ]
    if [row["restart"] for row in restart_results] != list(range(spec.restarts)):
        raise RuntimeError("Adam microbatch merge lost the global restart order")
    return {
        "optimizer": "adam",
        "initialization_seed": int(seed),
        "initialization_scheme": INITIALIZATION_SCHEME,
        "initialization_stream_seeds": [
            stream_seed
            for batch in batches
            for stream_seed in batch["initialization_stream_seeds"]
        ],
        "initial_gamma": [
            row for batch in batches for row in batch["initial_gamma"]
        ],
        "initial_beta": [
            row for batch in batches for row in batch["initial_beta"]
        ],
        "objective": "expected_coarse_energy" if method == "stage1" else "expected_fine_energy",
        "steps": spec.steps,
        "restarts": spec.restarts,
        "learning_rate": spec.adam_lr,
        "selected_restart": int(winner["selected_restart"]),
        "selected_loss": float(winner["selected_loss"]),
        "gamma": list(winner["gamma"]),
        "beta": list(winner["beta"]),
        "restart_results": restart_results,
        "elapsed_seconds": time.time() - started,
        "objective_evaluations_per_restart": spec.steps + 1,
    }


def optimize_spsa(engine: StatevectorEngine, method: str, depth: int,
                  reference: torch.Tensor | None, spec: OptimizerSpec,
                  seed: int) -> dict[str, Any]:
    gammas, betas, initialization_stream_seeds = _initial_angles(
        spec.restarts, depth, engine.instance.real_dtype, engine.device, seed,
    )
    initial_gamma = gammas.detach().cpu().tolist()
    initial_beta = betas.detach().cpu().tolist()
    parameters = torch.cat((gammas, betas), dim=1)
    perturbation_rows, perturbation_stream_seeds = spsa_perturbation_rows(
        spec.restarts, depth, spec.steps, seed,
    )
    perturbations_by_step = torch.as_tensor(
        perturbation_rows, dtype=engine.instance.real_dtype, device=engine.device,
    )
    with torch.no_grad():
        states = engine.states(method, gammas, betas, reference)
        initial_losses = engine.objective(method, states)
        if not bool(torch.all(torch.isfinite(initial_losses)).item()):
            raise FloatingPointError("SPSA produced a nonfinite initial expected energy")
    best_losses = initial_losses.clone()
    best_parameters = parameters.clone()
    best_steps = torch.zeros(spec.restarts, dtype=torch.int64, device=engine.device)
    trace_steps = sorted(set((0, spec.steps // 4, spec.steps // 2, 3 * spec.steps // 4, spec.steps)))
    traces: list[list[dict[str, float | int]]] = [
        [{"step": 0, "loss": float(initial_losses[index].item()), "best_loss": float(initial_losses[index].item())}]
        for index in range(spec.restarts)
    ]
    stability_offset = max(1.0, 0.1 * spec.steps)
    started = time.time()

    with torch.no_grad():
        for step in range(1, spec.steps + 1):
            perturbations = perturbations_by_step[step - 1]
            ck = spec.spsa_c / (step ** spec.spsa_gamma)
            ak = spec.spsa_a / ((step + stability_offset) ** spec.spsa_alpha)
            plus = parameters + ck * perturbations
            minus = parameters - ck * perturbations
            combined = torch.cat((plus, minus), dim=0)
            states = engine.states(method, combined[:, :depth], combined[:, depth:], reference)
            losses = engine.objective(method, states)
            plus_losses = losses[:spec.restarts]
            minus_losses = losses[spec.restarts:]
            for candidates, candidate_losses in ((plus, plus_losses), (minus, minus_losses)):
                improved = candidate_losses < best_losses
                best_losses.copy_(torch.where(improved, candidate_losses, best_losses))
                best_parameters.copy_(torch.where(improved.unsqueeze(1), candidates, best_parameters))
                best_steps.copy_(torch.where(improved, torch.full_like(best_steps, step), best_steps))
            gradient_scale = ((plus_losses - minus_losses) / (2.0 * ck)).unsqueeze(1)
            parameters = parameters - ak * gradient_scale * perturbations
            if step in trace_steps:
                if not bool(torch.all(torch.isfinite(losses)).item()):
                    raise FloatingPointError("SPSA produced a nonfinite expected energy")
                observed = torch.minimum(plus_losses, minus_losses).detach().cpu().tolist()
                current_best = best_losses.detach().cpu().tolist()
                for restart in range(spec.restarts):
                    traces[restart].append({"step": step, "loss": float(observed[restart]), "best_loss": float(current_best[restart])})

        terminal_states = engine.states(method, parameters[:, :depth], parameters[:, depth:], reference)
        terminal_losses = engine.objective(method, terminal_states)
        if not bool(torch.all(torch.isfinite(terminal_losses)).item()):
            raise FloatingPointError("SPSA produced a nonfinite terminal expected energy")
        improved = terminal_losses < best_losses
        best_losses.copy_(torch.where(improved, terminal_losses, best_losses))
        best_parameters.copy_(torch.where(improved.unsqueeze(1), parameters, best_parameters))
        best_steps.copy_(torch.where(improved, torch.full_like(best_steps, spec.steps), best_steps))

    selected = int(torch.argmin(best_losses).item())
    best_values = best_losses.detach().cpu().tolist()
    best_step_values = best_steps.detach().cpu().tolist()
    return {
        "optimizer": "spsa",
        "initialization_seed": int(seed),
        "initialization_scheme": INITIALIZATION_SCHEME,
        "initialization_stream_seeds": initialization_stream_seeds,
        "initial_gamma": [[float(value) for value in row] for row in initial_gamma],
        "initial_beta": [[float(value) for value in row] for row in initial_beta],
        "perturbation_scheme": SPSA_PERTURBATION_SCHEME,
        "perturbation_stream_seeds": perturbation_stream_seeds,
        "objective": "expected_coarse_energy" if method == "stage1" else "expected_fine_energy",
        "steps": spec.steps,
        "restarts": spec.restarts,
        "learning_rate_scale": spec.spsa_a,
        "perturbation_scale": spec.spsa_c,
        "learning_rate_exponent": spec.spsa_alpha,
        "perturbation_exponent": spec.spsa_gamma,
        "selected_restart": selected,
        "selected_loss": float(best_losses[selected].item()),
        "gamma": [float(value) for value in best_parameters[selected, :depth].detach().cpu().tolist()],
        "beta": [float(value) for value in best_parameters[selected, depth:].detach().cpu().tolist()],
        "restart_results": [
            {
                "restart": restart,
                "best_loss": float(best_values[restart]),
                "best_step": int(best_step_values[restart]),
                "trace": traces[restart],
            }
            for restart in range(spec.restarts)
        ],
        "elapsed_seconds": time.time() - started,
        "objective_evaluations_per_restart": 2 * spec.steps + 2,
    }


def optimize(engine: StatevectorEngine, method: str, depth: int,
             reference: torch.Tensor | None, spec: OptimizerSpec,
             seed_parts: Iterable[Any],
             restart_batch_size: int | None = None) -> dict[str, Any]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if spec.steps <= 0 or spec.restarts <= 0:
        raise ValueError("steps and restarts must be positive")
    # Callers choose the seed-family label.  LP and warm start deliberately pass
    # the same label so their initial angle rows are paired.
    seed = stable_seed(*seed_parts, depth, spec.name)
    if restart_batch_size is not None and method != "standard_qaoa":
        raise ValueError("restart microbatching is only enabled for direct standard QAOA")
    if spec.name == "adam":
        result = optimize_adam(
            engine, method, depth, reference, spec, seed, restart_batch_size,
        )
    elif spec.name == "spsa":
        if restart_batch_size is not None:
            raise ValueError("restart microbatching is currently implemented only for Adam")
        result = optimize_spsa(engine, method, depth, reference, spec, seed)
    else:
        raise ValueError(f"unknown optimizer: {spec.name}")
    return result


def replay(engine: StatevectorEngine, method: str, result: dict[str, Any],
           reference: torch.Tensor | None) -> torch.Tensor:
    gamma = torch.tensor(result["gamma"], dtype=engine.instance.real_dtype, device=engine.device).unsqueeze(0)
    beta = torch.tensor(result["beta"], dtype=engine.instance.real_dtype, device=engine.device).unsqueeze(0)
    with torch.no_grad():
        return engine.states(method, gamma, beta, reference)[0].detach().clone()


def state_metrics(instance: SBMInstance, state: torch.Tensor, ru: int) -> dict[str, Any]:
    with torch.no_grad():
        probabilities = state.abs().square()
        exact_probability = float(torch.sum(probabilities[instance.ground_indices]).item())
        norm = float(torch.sum(probabilities).item())
        fine_energy = float(torch.sum(probabilities * instance.fine_cost).item())
        coarse_energy = float(torch.sum(probabilities * instance.coarse_cost).item())
        outside_sector_probability = (
            None if instance.feasible_sector_mask is None
            else float(torch.sum(probabilities[~instance.feasible_sector_mask]).item())
        )
    if not all(math.isfinite(value) for value in (exact_probability, norm, fine_energy, coarse_energy)):
        raise FloatingPointError("nonfinite state metric")
    if abs(norm - 1.0) > 5e-4:
        raise FloatingPointError(f"state norm drifted to {norm}")
    if exact_probability < -1e-7 or exact_probability > 1.0 + 5e-4:
        raise FloatingPointError(f"invalid exact-ground probability {exact_probability}")
    if outside_sector_probability is not None and (
        not math.isfinite(outside_sector_probability)
        or outside_sector_probability < -1e-7
        or outside_sector_probability > 5e-5
    ):
        raise FloatingPointError(
            f"state leaked outside the fixed-Hamming-weight sector: {outside_sector_probability}"
        )
    repetitions = exact_rts99(exact_probability)
    metrics = {
        "target": target_definition(instance.spec),
        "exact_ground_probability": exact_probability,
        "state_norm": norm,
        "expected_fine_energy": fine_energy,
        "expected_coarse_energy": coarse_energy,
        "resource_units_per_run": int(ru),
        "repetitions_to_99_percent": repetitions,
        "total_cost_to_99_percent": None if repetitions is None else int(repetitions * ru),
    }
    if outside_sector_probability is not None:
        metrics.update({
            "outside_feasible_sector_probability": outside_sector_probability,
            "feasible_sector_probability": norm - outside_sector_probability,
        })
    return metrics
