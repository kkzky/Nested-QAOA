"""Run one direct Adam QAOA depth with restart microbatching.

This runner exists only to evaluate the high-depth direct baseline without
materializing the Stage-1, LP-QAOA, or warm-start computations.  Logical
restarts are generated once and split only for statevector execution, so the
result is the same four-start optimization protocol used by the main queue.
"""

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
    OptimizerSpec,
    SBMInstance,
    StatevectorEngine,
    canonical_queue_task_spec,
    coarse_topology_metadata,
    instance_spec_payload,
    optimize,
    replay,
    resource_units,
    sbm_model_metadata,
    stable_seed,
    state_metrics,
    target_evaluation_rule,
    validate_embedded_queue_task_spec,
    validate_queue_runtime_binding,
)


RUNNER_NAME = "run_natural_sbm_adam_direct_microbatch.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_queue_task_spec(text: str) -> dict[str, Any]:
    try:
        return validate_embedded_queue_task_spec(json.loads(text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_instance_spec(args: argparse.Namespace) -> InstanceSpec:
    return InstanceSpec(
        n=args.n,
        graph_seed=args.graph_seed,
        group_size=args.group_size,
        p_in=args.p_in,
        p_out=args.p_out,
        coarse_inter_weight=args.coarse_inter_weight,
        macro_assignment=args.macro_assignment,
        coarse_intra_weight=args.coarse_intra_weight,
        coarse_topology=args.coarse_topology,
        coarse_forest_edges=args.coarse_forest_edges,
        centered_threshold=args.centered_threshold,
        feasible_hamming_weight=None,
        mixer_type="x",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    if args.optimizer != "adam":
        raise ValueError("the direct microbatch runner supports Adam only")
    if args.depth != 8:
        raise ValueError("this sidecar is frozen to direct QAOA depth p=8")
    if args.restarts != 4 or args.direct_restart_batch_size != 2:
        raise ValueError("the sidecar requires four logical restarts in batches of two")
    if args.steps <= 0:
        raise ValueError("Adam steps must be positive")

    torch.manual_seed(args.graph_seed)
    torch.cuda.manual_seed_all(args.graph_seed)
    validate_queue_runtime_binding(args, build_parser(), RUNNER_NAME)
    instance_spec = build_instance_spec(args)
    optimizer_spec = OptimizerSpec(
        name="adam",
        steps=args.steps,
        restarts=args.restarts,
        adam_lr=args.adam_lr,
    )
    queue_task_spec = args.queue_task_spec_json
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
            "initialization_scheme": INITIALIZATION_SCHEME,
            "instance": instance_spec_payload(instance_spec),
            "graph_model": sbm_model_metadata(instance_spec),
            "coarse_construction": coarse_topology_metadata(instance_spec),
            "optimizer_family": "adam",
            "method": "standard_qaoa",
            "direct_depth": args.depth,
            "optimizer_spec": asdict(optimizer_spec),
            "direct_restart_execution": {
                "logical_restarts": args.restarts,
                "restart_batch_size": args.direct_restart_batch_size,
                "global_restart_indices": list(range(args.restarts)),
                "restart_batches": [[0, 1], [2, 3]],
                "initialization_streams": (
                    "identical_to_the_unsplit_logical_restart_set"
                ),
                "optimizer_updates": (
                    "independent_restart_trajectories_merged_by_minimum_expected_energy"
                ),
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "records": [],
    }
    output = Path(args.output)
    save(output, payload)
    started = time.time()

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    instance = SBMInstance(instance_spec, device)
    engine = StatevectorEngine(instance)
    payload["instance"] = instance.summary()
    payload["instance"].update({
        "fine_resource_two_body_term_count": int(
            instance.fine_resource_two_body_term_count
        ),
        "initial_state_preparation_ru": int(instance.initial_state_preparation_ru),
    })
    save(output, payload)

    common_seed_parts = (args.graph_seed, args.n, args.group_size)
    expected_seed = stable_seed(
        *common_seed_parts, "direct", args.depth, "adam",
    )
    optimization = optimize(
        engine,
        "standard_qaoa",
        args.depth,
        None,
        optimizer_spec,
        (*common_seed_parts, "direct"),
        restart_batch_size=args.direct_restart_batch_size,
    )
    if int(optimization["initialization_seed"]) != expected_seed:
        raise RuntimeError("direct p8 initialization stream differs from the main protocol")
    state = replay(engine, "standard_qaoa", optimization, None)
    metrics = state_metrics(
        instance,
        state,
        resource_units(
            args.n,
            "standard_qaoa",
            None,
            args.depth,
            instance.coarse_resource_two_body_term_count,
            instance.fine_resource_two_body_term_count,
            instance.initial_state_preparation_ru,
        ),
    )
    payload["records"].append({
        "method": "standard_qaoa",
        "p1": None,
        "p2": args.depth,
        "stage1_reference_key": None,
        "optimizer": "adam",
        "training_objective": optimization["objective"],
        "optimization": optimization,
        "metrics": metrics,
    })
    payload["execution_diagnostics"] = {
        "logical_restarts": args.restarts,
        "restart_batch_size": args.direct_restart_batch_size,
        "restart_result_ids": [
            int(row["restart"]) for row in optimization["restart_results"]
        ],
        "device_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    payload["status"] = "complete"
    payload["completed_at"] = utc_now()
    payload["elapsed_seconds"] = time.time() - started
    save(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--queue-task-spec-json", type=parse_queue_task_spec)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--p-in", type=float, required=True)
    parser.add_argument("--p-out", type=float, required=True)
    parser.add_argument(
        "--macro-assignment",
        choices=("alternating_contiguous_microblocks",),
        required=True,
    )
    parser.add_argument("--coarse-inter-weight", type=float, required=True)
    parser.add_argument("--coarse-intra-weight", type=float, required=True)
    parser.add_argument("--centered-threshold", type=float, required=True)
    parser.add_argument("--coarse-topology", choices=("full",), required=True)
    parser.add_argument("--coarse-forest-edges", type=int, required=True)
    parser.add_argument("--mixer-type", choices=("x",), required=True)
    parser.add_argument("--edge-weight-distribution", choices=("unit",), required=True)
    parser.add_argument(
        "--edge-weight-coefficient-of-variation", type=float, required=True,
    )
    parser.add_argument("--edge-sign-distribution", choices=("all_positive",), required=True)
    parser.add_argument("--edge-sign-reliability", type=float, required=True)
    parser.add_argument(
        "--fine-support-model", choices=("independent_pair_bernoulli",), required=True,
    )
    parser.add_argument("--intra-edge-sign-policy", choices=("sampled",), required=True)
    parser.add_argument("--intra-edge-scale", type=float, required=True)
    parser.add_argument("--optimizer", choices=("adam",), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--adam-lr", type=float, required=True)
    parser.add_argument("--direct-restart-batch-size", type=int, required=True)
    parser.add_argument("--cost-chunk-size", type=int, required=True)
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
        }, sort_keys=True, allow_nan=False))
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
