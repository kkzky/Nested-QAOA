from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SOURCE = Path(__file__).resolve().parent
CONTRACT_PATH = SOURCE / "depth_coverage_contract.json"
PAPER_CONTRACT_PATH = SOURCE / "paper_facing_contract.json"
IDENTITY_REFERENCE_PATH = SOURCE / "N30_OBJECTIVE_IDENTITY_REFERENCE.json"
RUNNER = SOURCE / "paper_facing_campaign_runner.py"
DEPTH_COVERAGE_ROOT = "depth_coverage_v2"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"Invalid geometric-mean values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def one_trial_by_depth(case_root: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    result: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((case_root / "trials").glob("*.json")):
        row = read_json(path)
        depth = int(row["depth"])
        if depth in result:
            raise RuntimeError(f"Duplicate depth {depth} under {case_root}")
        result[depth] = (path, row)
    return result


def validate_existing_completion(
    *,
    case_root: Path,
    depths: list[int],
) -> None:
    """Reject a stale completion marker before the underlying runner can skip.

    A completion marker is reusable only when both its trial files and its
    saved scan contract cover exactly the requested depth set.  This prevents
    a completed narrower scan from silently satisfying a later wider scan.
    """
    complete_path = case_root / "COMPLETE.json"
    if not complete_path.exists():
        return
    complete = read_json(complete_path)
    scan_contract_path = case_root / "scan_contract.json"
    observed_depths = sorted(one_trial_by_depth(case_root))
    contract_depths = (
        sorted(int(value) for value in read_json(scan_contract_path)["depths"])
        if scan_contract_path.exists()
        else []
    )
    expected_depths = sorted(depths)
    mismatches: dict[str, Any] = {}
    if complete.get("status") != "complete":
        mismatches["status"] = complete.get("status")
    if not complete.get("all_trials_validated", False):
        mismatches["all_trials_validated"] = complete.get(
            "all_trials_validated"
        )
    if observed_depths != expected_depths:
        mismatches["trial_depths"] = {
            "expected": expected_depths,
            "observed": observed_depths,
        }
    if contract_depths != expected_depths:
        mismatches["scan_contract_depths"] = {
            "expected": expected_depths,
            "observed": contract_depths,
        }
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse a stale or incompatible completion marker "
            f"under {case_root}: {mismatches}"
        )


def identity_reference() -> dict[int, dict[str, str]]:
    reference = read_json(IDENTITY_REFERENCE_PATH)
    return {
        int(row["seed"]): {
            "support_sha256": row["support_sha256"],
            "objective_sha256": row["objective_sha256"],
            "target_manifest_sha256": row["target_manifest_sha256"],
        }
        for row in reference["rows"]
    }


def build_command(
    *,
    campaign_root: Path,
    seed: int,
    name: str,
    config: dict[str, Any],
    depths: list[int],
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "scan",
        "--N",
        "30",
        "--seed",
        str(seed),
        "--schedule",
        "P3",
        "--objective-model",
        "block_state_p3",
        "--family",
        "signed_cont_uniform",
        "--range-a",
        "0.5",
        "--algorithm",
        str(config["algorithm"]),
        "--depths",
        ",".join(str(depth) for depth in depths),
        "--penalty-kappa",
        str(config["penalty_kappa"]),
        "--device",
        "cuda",
        "--dtype",
        "complex64",
        "--restarts",
        "5",
        "--steps",
        "120",
        "--optimizer",
        "spsa",
        "--spsa-c",
        str(config["spsa_c"]),
        "--spsa-target-step",
        str(config["spsa_target_step"]),
        "--spsa-alpha",
        "0.602",
        "--spsa-gamma",
        "0.101",
        "--spsa-stability-fraction",
        "0.10",
        "--spsa-calibration-directions",
        "8",
        "--spsa-directions-per-update",
        "1",
        "--spsa-max-step-norm",
        str(config["spsa_max_step_norm"]),
        "--patience",
        "60",
        "--trace-every",
        "5",
        "--sequential-restarts",
        "--budget-tag",
        "paper_facing_depth_coverage_correction",
        "--output",
        str(output),
    ]
    if config["algorithm"] in {"XY-LP-QAOA", "Warm-XY"}:
        stage1 = campaign_root / "stage1" / "N30" / "stage1_p12.json"
        if not stage1.exists():
            raise FileNotFoundError(f"Missing exact N=30 Stage-1 checkpoint: {stage1}")
        command.extend(["--stage1-checkpoint", str(stage1)])
    return command


def validate_case(
    *,
    campaign_root: Path,
    case_root: Path,
    seed: int,
    name: str,
    config: dict[str, Any],
    depths: list[int],
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    complete = read_json(case_root / "COMPLETE.json")
    if complete.get("status") != "complete" or not complete.get(
        "all_trials_validated", False
    ):
        raise RuntimeError(f"Invalid completion marker: {case_root / 'COMPLETE.json'}")
    trials = one_trial_by_depth(case_root)
    if sorted(trials) != sorted(depths):
        raise RuntimeError(
            f"Depth mismatch for seed={seed}, config={name}: "
            f"{sorted(trials)} versus {sorted(depths)}"
        )
    expected_calls = 5 * (2 + 2 * 8 + 2 * 1 * 120) + 1
    stage1 = campaign_root / "stage1" / "N30" / "stage1_p12.json"
    expected_stage1 = (
        sha256(stage1)
        if config["algorithm"] in {"XY-LP-QAOA", "Warm-XY"}
        else None
    )
    summaries: list[dict[str, Any]] = []
    for depth in sorted(trials):
        path, row = trials[depth]
        expected = {
            "N": 30,
            "seed": seed,
            "algorithm": config["algorithm"],
            "depth": depth,
            "penalty_kappa": float(config["penalty_kappa"]),
            "objective_model": "block_state_p3",
            "objective_degree": 6,
            "validation_status": "passed",
            "completion_status": "complete",
            "support_sha256": expected_identity["support_sha256"],
            "objective_sha256": expected_identity["objective_sha256"],
            "target_manifest_sha256": expected_identity[
                "target_manifest_sha256"
            ],
            "stage1_checkpoint_sha256": expected_stage1,
        }
        mismatches = {
            key: {"expected": value, "observed": row.get(key)}
            for key, value in expected.items()
            if row.get(key) != value
        }
        optimizer = row["optimizer_result"]
        optimizer_expected = {
            "name": "spsa",
            "objective_calls_total": expected_calls,
            "objective_call_accounting_validated": True,
            "incumbent_preservation_validated": True,
        }
        for key, value in optimizer_expected.items():
            if optimizer.get(key) != value:
                mismatches[f"optimizer_result.{key}"] = {
                    "expected": value,
                    "observed": optimizer.get(key),
                }
        if mismatches:
            raise RuntimeError(
                f"Depth-coverage mismatch for {path}: {mismatches}"
            )
        summaries.append(
            {
                "seed": seed,
                "configuration": name,
                "algorithm": config["algorithm"],
                "penalty_kappa": float(config["penalty_kappa"]),
                "spsa_c": float(config["spsa_c"]),
                "depth": depth,
                "target_a_probability": float(
                    row["replay_probabilities"]["target_a_sqrt"]
                ),
                "target_b_probability": float(
                    row["replay_probabilities"]["target_b_k2"]
                ),
                "target_a_cost": int(
                    row["target_costs"]["target_a_sqrt_cost"]
                ),
                "target_b_cost": int(
                    row["target_costs"]["target_b_k2_cost"]
                ),
                "terminal_RU": int(row["resources"]["terminal_RU"]),
                "objective_calls": int(
                    row["optimizer_result"]["objective_calls_total"]
                ),
                "trial_path": str(path),
                "trial_sha256": sha256(path),
            }
        )
    return {
        "seed": seed,
        "configuration": name,
        "algorithm": config["algorithm"],
        "case_root": str(case_root),
        "complete_sha256": sha256(case_root / "COMPLETE.json"),
        "trials": summaries,
    }


def run_case(
    *,
    campaign_root: Path,
    seed: int,
    name: str,
    config: dict[str, Any],
    depths: list[int],
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    case_root = (
        campaign_root
        / DEPTH_COVERAGE_ROOT
        / "tuning"
        / f"seed{seed}"
        / name
    )
    print(
        json.dumps(
            {
                "event": "depth_coverage_case_start",
                "seed": seed,
                "configuration": name,
                "depth_count": len(depths),
            }
        ),
        flush=True,
    )
    validate_existing_completion(case_root=case_root, depths=depths)
    subprocess.run(
        build_command(
            campaign_root=campaign_root,
            seed=seed,
            name=name,
            config=config,
            depths=depths,
            output=case_root,
        ),
        cwd=SOURCE,
        check=True,
    )
    result = validate_case(
        campaign_root=campaign_root,
        case_root=case_root,
        seed=seed,
        name=name,
        config=config,
        depths=depths,
        expected_identity=expected_identity,
    )
    print(
        json.dumps(
            {
                "event": "depth_coverage_case_complete",
                "seed": seed,
                "configuration": name,
            }
        ),
        flush=True,
    )
    return result


def initial_summary(
    *,
    results: list[dict[str, Any]],
    configurations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        trial for result in results for trial in result["trials"]
    ]
    seeds = sorted({int(row["seed"]) for row in rows})
    depths = sorted({int(row["depth"]) for row in rows})
    aggregate: list[dict[str, Any]] = []
    for name in configurations:
        for depth in depths:
            subset = [
                row
                for row in rows
                if row["configuration"] == name and row["depth"] == depth
            ]
            if not subset:
                continue
            aggregate.append(
                {
                    "configuration": name,
                    "algorithm": subset[0]["algorithm"],
                    "depth": depth,
                    "seed_count": len(subset),
                    "seeds": sorted(int(row["seed"]) for row in subset),
                    "target_a_geomean_probability": geometric_mean(
                        [row["target_a_probability"] for row in subset]
                    ),
                    "target_b_geomean_probability": geometric_mean(
                        [row["target_b_probability"] for row in subset]
                    ),
                    "target_a_geomean_cost": geometric_mean(
                        [float(row["target_a_cost"]) for row in subset]
                    ),
                    "target_b_geomean_cost": geometric_mean(
                        [float(row["target_b_cost"]) for row in subset]
                    ),
                }
            )
    per_configuration: dict[str, Any] = {}
    for name in configurations:
        config_rows = [
            row
            for row in aggregate
            if row["configuration"] == name
            and int(row["seed_count"]) == len(seeds)
        ]
        if not config_rows:
            raise RuntimeError(
                f"No two-seed retained depths for configuration {name}"
            )
        per_configuration[name] = {}
        for target in ("a", "b"):
            selected = min(
                config_rows,
                key=lambda row: (
                    float(row[f"target_{target}_geomean_cost"]),
                    int(row["depth"]),
                ),
            )
            per_configuration[name][f"target_{target}"] = {
                "selected_depth": int(selected["depth"]),
                "selected_geomean_cost": float(
                    selected[f"target_{target}_geomean_cost"]
                ),
                "selected_geomean_probability": float(
                    selected[f"target_{target}_geomean_probability"]
                ),
                "at_upper_boundary": int(selected["depth"]) == max(depths),
            }
    return {
        "rows": rows,
        "aggregate": aggregate,
        "per_configuration_minima": per_configuration,
    }


def seed900_pruning_decision(
    *,
    seed900_results: list[dict[str, Any]],
    configurations: dict[str, dict[str, Any]],
    depths: list[int],
) -> dict[str, Any]:
    rows = [
        trial for result in seed900_results for trial in result["trials"]
    ]
    decisions: dict[str, Any] = {}
    for name in configurations:
        config_rows = sorted(
            [
                row
                for row in rows
                if row["configuration"] == name and int(row["seed"]) == 900
            ],
            key=lambda row: int(row["depth"]),
        )
        if [int(row["depth"]) for row in config_rows] != depths:
            raise RuntimeError(
                f"Incomplete seed-900 depth grid for {name}: "
                f"{[int(row['depth']) for row in config_rows]} versus {depths}"
            )
        best_a = min(float(row["target_a_cost"]) for row in config_rows)
        best_b = min(float(row["target_b_cost"]) for row in config_rows)
        best_depth_a = min(
            (
                int(row["depth"])
                for row in config_rows
                if float(row["target_a_cost"]) == best_a
            )
        )
        best_depth_b = min(
            (
                int(row["depth"])
                for row in config_rows
                if float(row["target_b_cost"]) == best_b
            )
        )
        protected = {best_depth_a, best_depth_b}
        for selected in (best_depth_a, best_depth_b):
            lower = [depth for depth in depths if depth < selected]
            higher = [depth for depth in depths if depth > selected]
            if lower:
                protected.add(max(lower))
            if higher:
                protected.add(min(higher))
        rows_decision: list[dict[str, Any]] = []
        retained: list[int] = []
        pruned: list[int] = []
        for row in config_rows:
            depth = int(row["depth"])
            a_multiple = float(row["target_a_cost"]) / best_a
            b_multiple = float(row["target_b_cost"]) / best_b
            obviously_bad = (
                a_multiple >= 5.0
                and b_multiple >= 5.0
                and depth not in protected
            )
            (pruned if obviously_bad else retained).append(depth)
            rows_decision.append(
                {
                    "depth": depth,
                    "target_a_cost": int(row["target_a_cost"]),
                    "target_b_cost": int(row["target_b_cost"]),
                    "target_a_multiple_of_best": a_multiple,
                    "target_b_multiple_of_best": b_multiple,
                    "protected_as_minimum_or_neighbor": depth in protected,
                    "seed901_action": "prune" if obviously_bad else "retain",
                }
            )
        decisions[name] = {
            "algorithm": configurations[name]["algorithm"],
            "best_target_a_cost": int(best_a),
            "best_target_a_depth": best_depth_a,
            "best_target_b_cost": int(best_b),
            "best_target_b_depth": best_depth_b,
            "protected_depths": sorted(protected),
            "retained_depths_for_seed901": retained,
            "pruned_depths_for_seed901": pruned,
            "rows": rows_decision,
        }
    return {
        "schema": "n30-degree6-seed900-depth-pruning-decision-v1",
        "status": "frozen_before_seed901_launch",
        "seed": 900,
        "rule": (
            "Prune a configuration-depth pair only if both target costs are "
            "at least five times that configuration's seed-900 best, while "
            "protecting each target minimum and its nearest tested neighbors."
        ),
        "configurations": decisions,
        "contract_sha256": sha256(CONTRACT_PATH),
        "local_simulation_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    correction = read_json(CONTRACT_PATH)
    paper_contract = read_json(PAPER_CONTRACT_PATH)
    configurations = paper_contract["standalone_configurations"]
    requested_names = correction["tuning"]["configurations"]
    missing = sorted(set(requested_names) - set(configurations))
    if missing:
        raise RuntimeError(f"Missing frozen configurations: {missing}")
    depths = [
        int(value) for value in correction["tuning"]["initial_depth_grid"]
    ]
    seeds = [int(value) for value in correction["tuning"]["seeds"]]
    identities = identity_reference()
    seed900_tasks = [
        {
            "campaign_root": campaign_root,
            "seed": 900,
            "name": name,
            "config": configurations[name],
            "depths": depths,
            "expected_identity": identities[900],
        }
        for name in requested_names
    ]
    started = time.perf_counter()
    seed900_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(run_case, **task) for task in seed900_tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            seed900_results.append(future.result())
    pruning = seed900_pruning_decision(
        seed900_results=seed900_results,
        configurations={
            name: configurations[name] for name in requested_names
        },
        depths=depths,
    )
    pruning_path = (
        campaign_root
        / DEPTH_COVERAGE_ROOT
        / "SEED900_DEPTH_PRUNING_DECISION.json"
    )
    write_json(pruning_path, pruning)
    print(
        json.dumps(
            {
                "event": "seed900_depth_pruning_frozen",
                "path": str(pruning_path),
                "sha256": sha256(pruning_path),
                "retained_counts": {
                    name: len(
                        value["retained_depths_for_seed901"]
                    )
                    for name, value in pruning["configurations"].items()
                },
                "pruned_counts": {
                    name: len(value["pruned_depths_for_seed901"])
                    for name, value in pruning["configurations"].items()
                },
            }
        ),
        flush=True,
    )
    seed901_tasks = [
        {
            "campaign_root": campaign_root,
            "seed": 901,
            "name": name,
            "config": configurations[name],
            "depths": pruning["configurations"][name][
                "retained_depths_for_seed901"
            ],
            "expected_identity": identities[901],
        }
        for name in requested_names
    ]
    seed901_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(run_case, **task) for task in seed901_tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            seed901_results.append(future.result())
    results = seed900_results + seed901_results
    summary = initial_summary(
        results=results,
        configurations={name: configurations[name] for name in requested_names},
    )
    complete = {
        "schema": "n30-degree6-paper-facing-depth-coverage-initial-complete-v1",
        "status": "complete",
        "phase": "initial_wide_depth_grid_before_refinement",
        "seeds": seeds,
        "seed900_full_depths": depths,
        "seed901_depths_by_configuration": {
            name: value["retained_depths_for_seed901"]
            for name, value in pruning["configurations"].items()
        },
        "seed900_pruning_decision": str(pruning_path),
        "seed900_pruning_decision_sha256": sha256(pruning_path),
        "configurations": requested_names,
        "cases": sorted(
            results, key=lambda row: (row["seed"], row["configuration"])
        ),
        "aggregate": summary["aggregate"],
        "per_configuration_minima": summary["per_configuration_minima"],
        "refinement_required": True,
        "contract": str(CONTRACT_PATH),
        "contract_sha256": sha256(CONTRACT_PATH),
        "paper_contract_sha256": sha256(PAPER_CONTRACT_PATH),
        "identity_reference_sha256": sha256(IDENTITY_REFERENCE_PATH),
        "runner_sha256": sha256(RUNNER),
        "workers": args.workers,
        "elapsed_sec": time.perf_counter() - started,
        "local_simulation_used": False,
    }
    write_json(
        campaign_root
        / DEPTH_COVERAGE_ROOT
        / "INITIAL_WIDE_DEPTH_COMPLETE.json",
        complete,
    )


if __name__ == "__main__":
    main()
