"""Depth scan for one SBM graph with Stage-1 states cached by p1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sbm_standard_core import (
    INITIALIZATION_SCHEME,
    RUNNER_IMPLEMENTATION_ID,
    RUNNER_SCHEMA_VERSION,
    InstanceSpec,
    SBMInstance,
    StatevectorEngine,
    coarse_topology_metadata,
    instance_spec_payload,
    make_stage1_reference_key,
    mixer_protocol_metadata,
    normalize_state,
    optimize,
    replay,
    resolve_method_optimizer_specs,
    resource_units,
    sbm_model_metadata,
    stable_seed,
    state_metrics,
    target_evaluation_rule,
    validate_embedded_queue_task_spec,
    validate_queue_runtime_binding,
)


RUNNER_NAME = "run_depth_scan.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def parse_depths(text: str) -> list[int]:
    values = sorted(set(int(value.strip()) for value in text.split(",") if value.strip()))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("depths must be positive integers")
    return values


def parse_configs(text: str) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for item in text.split(","):
        if not item.strip():
            continue
        left, right = item.replace("-", ":").split(":", 1)
        pair = (int(left), int(right))
        if pair[0] <= 0 or pair[1] <= 0:
            raise argparse.ArgumentTypeError("depths must be positive")
        values.append(pair)
    values = sorted(set(values))
    if not values:
        raise argparse.ArgumentTypeError("at least one p1:p2 configuration is required")
    return values


def parse_queue_task_spec(text: str) -> dict[str, Any]:
    try:
        return validate_embedded_queue_task_spec(json.loads(text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def record(method: str, p1: int | None, p2: int | None,
           optimizer_name: str, optimization: dict[str, Any],
           metrics: dict[str, Any],
           stage1_reference_key: str | None) -> dict[str, Any]:
    return {
        "method": method,
        "p1": p1,
        "p2": p2,
        "stage1_reference_key": stage1_reference_key,
        "optimizer": optimizer_name,
        "training_objective": optimization["objective"],
        "optimization": optimization,
        "metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    device = torch.device("cuda")
    torch.manual_seed(args.graph_seed)
    torch.cuda.manual_seed_all(args.graph_seed)
    validate_queue_runtime_binding(args, build_parser(), RUNNER_NAME)
    queue_task_spec = args.queue_task_spec_json
    instance_spec = InstanceSpec(
        n=args.n,
        graph_seed=args.graph_seed,
        group_size=args.group_size,
        p_in=args.p_in,
        p_out=args.p_out,
        coarse_inter_weight=args.coarse_inter_weight,
        p_micro=args.p_micro,
        macro_assignment=args.macro_assignment,
        coarse_intra_weight=args.coarse_intra_weight,
        coarse_topology=args.coarse_topology,
        coarse_forest_edges=args.coarse_forest_edges,
        centered_threshold=args.centered_threshold,
        feasible_hamming_weight=args.feasible_hamming_weight,
        mixer_type=args.mixer_type,
        edge_weight_distribution=args.edge_weight_distribution,
        edge_weight_coefficient_of_variation=(
            args.edge_weight_coefficient_of_variation
        ),
        edge_sign_distribution=args.edge_sign_distribution,
        edge_sign_reliability=args.edge_sign_reliability,
        fine_support_model=args.fine_support_model,
        intra_edge_sign_policy=args.intra_edge_sign_policy,
        intra_edge_scale=args.intra_edge_scale,
        cost_chunk_size=args.cost_chunk_size,
    )
    optimizer_specs = resolve_method_optimizer_specs(args)
    stage1_optimizer_spec = optimizer_specs["stage1_only"]
    lp_optimizer_spec = optimizer_specs["lp_qaoa"]
    warm_optimizer_spec = optimizer_specs["warm_start_qaoa"]
    direct_optimizer_spec = optimizer_specs["standard_qaoa"]
    if args.direct_restart_batch_size is not None:
        if args.direct_restart_batch_size <= 0:
            raise ValueError("direct restart batch size must be positive")
        if args.optimizer != "adam":
            raise ValueError("direct restart microbatching is currently implemented only for Adam")
    common_seed_parts = (args.graph_seed, args.n, args.group_size)
    p1_values = sorted(set(pair[0] for pair in args.lp_configs))
    reference_keys = {
        str(p1): make_stage1_reference_key(
            instance_spec,
            p1,
            stage1_optimizer_spec,
            stable_seed(*common_seed_parts, "shared_stage1", p1, args.optimizer),
        )
        for p1 in p1_values
    }
    payload: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "protocol": {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "runner_implementation_id": RUNNER_IMPLEMENTATION_ID,
            "runner": RUNNER_NAME,
            "queue_task_spec": queue_task_spec,
            "loss_rule": "expected_energy_only",
            "checkpoint_selection": "minimum_expected_energy",
            "target_evaluation": target_evaluation_rule(instance_spec),
            "stage1_cache": "one_frozen_state_per_p1_reused_by_stage1_only_warm_start_and_lp_qaoa",
            "stage1_optimization_count": len(p1_values),
            "stage1_normalization": "unit_norm_before_evaluation_and_reuse",
            "fine_stage_initializations": "paired_between_warm_start_and_lp_qaoa",
            "initialization_scheme": INITIALIZATION_SCHEME,
            "initialization_pairing": "restart_indexed_streams; lp_and_warm_share_overlapping_starts_within_each_depth",
            "instance": instance_spec_payload(instance_spec),
            "graph_model": sbm_model_metadata(instance_spec),
            "coarse_construction": coarse_topology_metadata(instance_spec),
            "optimizer_family": args.optimizer,
            "resolved_optimizer_specs": {
                method: asdict(spec) for method, spec in optimizer_specs.items()
            },
            "stage1_reference_key": reference_keys,
            "lp_configs": [list(pair) for pair in args.lp_configs],
            "direct_depths": args.direct_depths,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "records": [],
    }
    if args.direct_restart_batch_size is not None:
        payload["protocol"]["direct_restart_execution"] = {
            "logical_restarts": direct_optimizer_spec.restarts,
            "restart_batch_size": min(
                args.direct_restart_batch_size, direct_optimizer_spec.restarts,
            ),
            "global_restart_indices": list(range(direct_optimizer_spec.restarts)),
            "initialization_streams": "identical_to_the_unsplit_logical_restart_set",
            "optimizer_updates": "independent_restart_trajectories_merged_by_minimum_expected_energy",
        }
    if instance_spec.feasible_hamming_weight is not None:
        payload["protocol"]["mixer"] = mixer_protocol_metadata(instance_spec)
        payload["protocol"]["fine_hamiltonian_resource_convention"] = (
            "within_the_fixed_weight_sector_the_centering_term_is_a_constant;"
            "compile_fine_phases_and_terminal_measurement_to_observed_edges_only"
        )
    output = Path(args.output)
    save(output, payload)
    started = time.time()

    instance = SBMInstance(instance_spec, device)
    engine = StatevectorEngine(instance)
    payload["instance"] = instance.summary()
    save(output, payload)
    for p1 in p1_values:
        reference_key = reference_keys[str(p1)]
        expected_stage1_seed = stable_seed(
            *common_seed_parts, "shared_stage1", p1, args.optimizer,
        )
        stage1_optimization = optimize(
            engine, "stage1", p1, None, stage1_optimizer_spec,
            (*common_seed_parts, "shared_stage1"),
        )
        if stage1_optimization["initialization_seed"] != expected_stage1_seed:
            raise RuntimeError("Stage-1 reference key does not match the realized initialization stream")
        stage1_state = normalize_state(replay(engine, "stage1", stage1_optimization, None))
        payload["records"].append(record(
            "stage1_only", p1, None, args.optimizer, stage1_optimization,
            state_metrics(instance, stage1_state, resource_units(
                args.n, "stage1_only", p1, None, instance.coarse_resource_two_body_term_count,
                instance.fine_resource_two_body_term_count,
                instance.initial_state_preparation_ru,
            )),
            reference_key,
        ))
        save(output, payload)

        for _, p2 in (pair for pair in args.lp_configs if pair[0] == p1):
            for method, method_optimizer_spec in (
                ("lp_qaoa", lp_optimizer_spec),
                ("warm_start_qaoa", warm_optimizer_spec),
            ):
                optimization = optimize(
                    engine, method, p2, stage1_state, method_optimizer_spec,
                    (*common_seed_parts, p1, "fine_stage"),
                )
                state = replay(engine, method, optimization, stage1_state)
                payload["records"].append(record(
                    method, p1, p2, args.optimizer, optimization,
                    state_metrics(instance, state, resource_units(
                        args.n, method, p1, p2, instance.coarse_resource_two_body_term_count,
                        instance.fine_resource_two_body_term_count,
                        instance.initial_state_preparation_ru,
                    )),
                    reference_key,
                ))
                save(output, payload)
                del state
        del stage1_state

    for depth in args.direct_depths:
        optimization = optimize(
            engine, "standard_qaoa", depth, None, direct_optimizer_spec,
            (*common_seed_parts, "direct"),
            restart_batch_size=args.direct_restart_batch_size,
        )
        state = replay(engine, "standard_qaoa", optimization, None)
        payload["records"].append(record(
            "standard_qaoa", None, depth, args.optimizer, optimization,
            state_metrics(instance, state, resource_units(
                args.n, "standard_qaoa", None, depth, instance.coarse_resource_two_body_term_count,
                instance.fine_resource_two_body_term_count,
                instance.initial_state_preparation_ru,
            )),
            None,
        ))
        save(output, payload)
        del state

    payload["status"] = "complete"
    payload["completed_at"] = utc_now()
    payload["elapsed_seconds"] = time.time() - started
    save(output, payload)
    return payload


def add_method_optimizer_overrides(parser: argparse.ArgumentParser, prefix: str) -> None:
    label = prefix.replace("_", "-")
    parser.add_argument(f"--{label}-steps", type=int)
    parser.add_argument(f"--{label}-restarts", type=int)
    parser.add_argument(f"--{label}-adam-lr", type=float)
    parser.add_argument(f"--{label}-spsa-a", type=float)
    parser.add_argument(f"--{label}-spsa-c", type=float)
    parser.add_argument(f"--{label}-spsa-alpha", type=float)
    parser.add_argument(f"--{label}-spsa-gamma", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--queue-task-spec-json", type=parse_queue_task_spec)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--p-in", type=float)
    parser.add_argument("--p-out", type=float)
    parser.add_argument("--p-micro", type=float)
    parser.add_argument(
        "--macro-assignment",
        choices=("alternating_contiguous_microblocks", "random_balanced_microblocks"),
        default="alternating_contiguous_microblocks",
    )
    parser.add_argument("--coarse-inter-weight", type=float, required=True)
    parser.add_argument("--coarse-intra-weight", type=float, default=1.0)
    parser.add_argument("--centered-threshold", type=float, default=0.5)
    parser.add_argument(
        "--coarse-topology",
        choices=(
            "full", "maximum_spanning_forest", "connected_top_k",
            "independent_random_3_regular",
        ),
        default="full",
    )
    parser.add_argument(
        "--coarse-forest-edges", "--coarse-top-k-edges",
        dest="coarse_forest_edges", type=int, default=0,
    )
    parser.add_argument("--feasible-hamming-weight", type=int)
    parser.add_argument("--mixer-type", choices=("x", "ring_xy"), default="x")
    parser.add_argument(
        "--edge-weight-distribution",
        choices=("unit", "lognormal_mean_one"), default="unit",
    )
    parser.add_argument(
        "--edge-weight-coefficient-of-variation", type=float, default=0.0,
    )
    parser.add_argument(
        "--edge-sign-distribution",
        choices=("all_positive", "planted_macro_reliability"),
        default="all_positive",
    )
    parser.add_argument("--edge-sign-reliability", type=float, default=1.0)
    parser.add_argument(
        "--fine-support-model",
        choices=("independent_pair_bernoulli", "quotient_k22_plus_intra"),
        default="independent_pair_bernoulli",
    )
    parser.add_argument(
        "--intra-edge-sign-policy",
        choices=("sampled", "deterministic_positive"), default="sampled",
    )
    parser.add_argument("--intra-edge-scale", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adam", "spsa"), required=True)
    parser.add_argument("--lp-configs", type=parse_configs, required=True)
    parser.add_argument("--direct-depths", type=parse_depths, required=True)
    parser.add_argument("--direct-restart-batch-size", type=int)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--adam-lr", type=float, default=0.04)
    parser.add_argument("--spsa-a", type=float, default=0.05)
    parser.add_argument("--spsa-c", type=float, default=0.12)
    parser.add_argument("--spsa-alpha", type=float, default=0.602)
    parser.add_argument("--spsa-gamma", type=float, default=0.101)
    for prefix in ("stage1", "lp", "warm", "direct"):
        add_method_optimizer_overrides(parser, prefix)
    parser.add_argument("--cost-chunk-size", type=int, default=2_097_152)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output)
    try:
        result = run(args)
        print(json.dumps({
            "status": result["status"],
            "output": str(output),
            "elapsed_seconds": result["elapsed_seconds"],
        }))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                existing.update(failure)
                failure = existing
            except Exception:
                pass
        save(output, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
