from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "paper_facing_contract.json"
RUNNER = ROOT / "paper_facing_campaign_runner.py"
STAGE1 = ROOT / "stage1_p12.json"
INITIAL_SELECTION = ROOT / "SPSA_CALIBRATION_SELECTION_INITIAL.json"
EXTENDED_SELECTION = ROOT / "SPSA_CALIBRATION_SELECTION_EXTENDED.json"
ULTRASMALL_SELECTION = ROOT / "SPSA_CALIBRATION_SELECTION_ULTRASMALL_C.json"


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


def one_trial(case_root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted((case_root / "trials").glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one trial under {case_root}, found {len(paths)}"
        )
    return paths[0], read_json(paths[0])


def validate_frozen_sources(contract: dict[str, Any]) -> None:
    frozen = contract["frozen_evidence"]
    expected = {
        STAGE1: frozen["stage1_checkpoint_sha256"],
        INITIAL_SELECTION: frozen["initial_calibration_selection_sha256"],
        EXTENDED_SELECTION: frozen["extended_calibration_selection_sha256"],
        ULTRASMALL_SELECTION: frozen["ultrasmall_c_selection_sha256"],
        ROOT / "campaign_runner.py": frozen["base_runner_sha256"],
        RUNNER: frozen["paper_facing_wrapper_sha256"],
        ROOT / "independent_validator.py": frozen[
            "independent_validator_sha256"
        ],
    }
    mismatches = {
        str(path): {"expected": wanted, "observed": sha256(path)}
        for path, wanted in expected.items()
        if not path.exists() or sha256(path) != wanted
    }
    if mismatches:
        raise RuntimeError(f"Frozen source mismatch: {mismatches}")


def build_command(
    *,
    phase: str,
    seed: int,
    depth: int,
    config: dict[str, Any],
    case_root: Path,
    device: str,
    initial_angle_trial: Path | None,
    angle_assisted: bool,
) -> list[str]:
    algorithm = str(config["algorithm"])
    tag_suffix = "lp_angle_assisted" if angle_assisted else "history"
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
        algorithm,
        "--depths",
        str(depth),
        "--penalty-kappa",
        str(config["penalty_kappa"]),
        "--device",
        device,
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
        f"paper_facing_{phase}_{tag_suffix}",
        "--output",
        str(case_root),
    ]
    if algorithm in {"XY-LP-QAOA", "Warm-XY"}:
        command.extend(["--stage1-checkpoint", str(STAGE1)])
    if initial_angle_trial is not None:
        command.extend(
            ["--initial-angle-trial", str(initial_angle_trial)]
        )
    return command


def validate_case(
    *,
    case_root: Path,
    config: dict[str, Any],
    seed: int,
    depth: int,
    contract: dict[str, Any],
    initial_angle_trial: Path | None,
    angle_assisted: bool,
) -> dict[str, Any]:
    complete_path = case_root / "COMPLETE.json"
    complete = read_json(complete_path)
    if complete.get("status") != "complete" or not complete.get(
        "all_trials_validated", False
    ):
        raise RuntimeError(f"Invalid completion marker: {complete_path}")
    trial_path, row = one_trial(case_root)
    expected_calls = int(
        contract["optimizer"]["expected_objective_calls_per_trial"]
    )
    checks = {
        "N": (row.get("N"), 30),
        "seed": (row.get("seed"), seed),
        "depth": (row.get("depth"), depth),
        "algorithm": (row.get("algorithm"), config["algorithm"]),
        "objective_model": (row.get("objective_model"), "block_state_p3"),
        "objective_degree": (row.get("objective_degree"), 6),
        "validation_status": (row.get("validation_status"), "passed"),
        "optimizer_name": (
            row["optimizer_result"].get("name"),
            "spsa",
        ),
        "objective_calls_total": (
            row["optimizer_result"].get("objective_calls_total"),
            expected_calls,
        ),
        "penalty_kappa": (
            float(row.get("penalty_kappa")),
            float(config["penalty_kappa"]),
        ),
        "stage1_checkpoint_sha256": (
            row.get("stage1_checkpoint_sha256"),
            (
                contract["frozen_evidence"]["stage1_checkpoint_sha256"]
                if config["algorithm"]
                in {"XY-LP-QAOA", "Warm-XY"}
                else None
            ),
        ),
        "base_runner_sha256": (
            row["source_hashes"].get("campaign_runner"),
            contract["frozen_evidence"]["base_runner_sha256"],
        ),
        "paper_facing_wrapper_sha256": (
            row["source_hashes"].get("paper_facing_wrapper"),
            contract["frozen_evidence"][
                "paper_facing_wrapper_sha256"
            ],
        ),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    if not row["optimizer_result"].get(
        "objective_call_accounting_validated", False
    ):
        mismatches["objective_call_accounting_validated"] = {
            "observed": False,
            "expected": True,
        }
    if not row["optimizer_result"].get(
        "incumbent_preservation_validated", False
    ):
        mismatches["incumbent_preservation_validated"] = {
            "observed": False,
            "expected": True,
        }
    scan = read_json(case_root / "scan_contract.json")
    angle_seed = scan.get("initial_angle_seed")
    if initial_angle_trial is None:
        if angle_seed is not None:
            mismatches["initial_angle_seed"] = {
                "observed": angle_seed,
                "expected": None,
            }
    else:
        expected_sha = sha256(initial_angle_trial)
        if not isinstance(angle_seed, dict) or angle_seed.get(
            "sha256"
        ) != expected_sha:
            mismatches["initial_angle_seed"] = {
                "observed": angle_seed,
                "expected_sha256": expected_sha,
            }
        if angle_assisted and (
            not isinstance(angle_seed, dict)
            or angle_seed.get("transfer")
            != "XY-LP-QAOA_angles_to_same_stage1_Warm-XY"
        ):
            mismatches["angle_transfer"] = {
                "observed": angle_seed,
                "expected": (
                    "XY-LP-QAOA_angles_to_same_stage1_Warm-XY"
                ),
            }
    if mismatches:
        raise RuntimeError(
            f"Trial validation mismatch at {trial_path}: {mismatches}"
        )
    return {
        "trial": str(trial_path),
        "trial_sha256": sha256(trial_path),
        "complete_sha256": sha256(complete_path),
        "support_sha256": row["support_sha256"],
        "objective_sha256": row["objective_sha256"],
        "target_manifest_sha256": row["target_manifest_sha256"],
        "target_a_probability": row["replay_probabilities"][
            "target_a_sqrt"
        ],
        "target_b_probability": row["replay_probabilities"][
            "target_b_k2"
        ],
        "target_a_cost": row["target_costs"]["target_a_sqrt_cost"],
        "target_b_cost": row["target_costs"]["target_b_k2_cost"],
    }


def run_case(
    *,
    phase_root: Path,
    phase: str,
    seed: int,
    depth: int,
    config_name: str,
    config: dict[str, Any],
    contract: dict[str, Any],
    device: str,
    initial_angle_trial: Path | None,
    angle_assisted: bool,
    dry_run: bool,
) -> dict[str, Any]:
    case_root = (
        phase_root
        / f"seed{seed}"
        / config_name
        / f"p{depth}"
    )
    command = build_command(
        phase=phase,
        seed=seed,
        depth=depth,
        config=config,
        case_root=case_root,
        device=device,
        initial_angle_trial=initial_angle_trial,
        angle_assisted=angle_assisted,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "case_root": str(case_root),
        }
    complete_path = case_root / "COMPLETE.json"
    if not complete_path.exists():
        log_path = (
            phase_root
            / "logs"
            / f"seed{seed}"
            / config_name
            / f"p{depth}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                + " ".join(command)
                + "\n"
            )
            log.flush()
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"Case failed ({process.returncode}): "
                f"{phase}/seed{seed}/{config_name}/p{depth}"
            )
    return validate_case(
        case_root=case_root,
        config=config,
        seed=seed,
        depth=depth,
        contract=contract,
        initial_angle_trial=initial_angle_trial,
        angle_assisted=angle_assisted,
    )


def run_chain(
    *,
    phase_root: Path,
    phase: str,
    seed: int,
    config_name: str,
    config: dict[str, Any],
    contract: dict[str, Any],
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    history_depth = int(contract["stages"]["history_depth"])
    reported_depths = [
        int(value) for value in contract["stages"]["reported_depths"]
    ]
    previous_trial: Path | None = None
    results: dict[str, Any] = {}
    for depth in (history_depth, *reported_depths):
        result = run_case(
            phase_root=phase_root,
            phase=phase,
            seed=seed,
            depth=depth,
            config_name=config_name,
            config=config,
            contract=contract,
            device=device,
            initial_angle_trial=previous_trial,
            angle_assisted=False,
            dry_run=dry_run,
        )
        results[str(depth)] = result
        if dry_run:
            previous_trial = (
                phase_root
                / f"seed{seed}"
                / config_name
                / f"p{depth}"
                / "trials"
                / "DRY_RUN.json"
            )
        else:
            previous_trial = Path(result["trial"])
    return {
        "seed": seed,
        "config": config_name,
        "results": results,
    }


def run_phase(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    validate_frozen_sources(contract)
    phase_root = Path(args.output_root) / args.phase
    complete_path = phase_root / "SUPERVISOR_COMPLETE.json"
    if complete_path.exists() and not args.dry_run:
        return read_json(complete_path)
    phase_root.mkdir(parents=True, exist_ok=True)
    seeds = [
        int(value)
        for value in contract["stages"][args.phase]["seeds"]
    ]
    configs = contract["standalone_configurations"]
    started = time.time()
    chains: dict[str, Any] = {}
    try:
        jobs = [
            (seed, name, config)
            for seed in seeds
            for name, config in configs.items()
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    run_chain,
                    phase_root=phase_root,
                    phase=args.phase,
                    seed=seed,
                    config_name=name,
                    config=config,
                    contract=contract,
                    device=args.device,
                    dry_run=args.dry_run,
                ): (seed, name)
                for seed, name, config in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                seed, name = futures[future]
                chains[f"seed{seed}/{name}"] = future.result()

        angle_config = {
            "algorithm": contract["angle_transfer_control"]["algorithm"],
            "penalty_kappa": contract["angle_transfer_control"][
                "penalty_kappa"
            ],
            "spsa_c": contract["angle_transfer_control"]["spsa_c"],
            "spsa_target_step": contract["angle_transfer_control"][
                "spsa_target_step"
            ],
            "spsa_max_step_norm": contract["angle_transfer_control"][
                "spsa_max_step_norm"
            ],
        }
        angle_results: dict[str, Any] = {}
        angle_jobs = []
        for seed in seeds:
            for depth in contract["stages"]["reported_depths"]:
                if args.dry_run:
                    lp_trial = (
                        phase_root
                        / f"seed{seed}"
                        / "lp_selected"
                        / f"p{depth}"
                        / "trials"
                        / "DRY_RUN.json"
                    )
                else:
                    lp_trial = Path(
                        chains[f"seed{seed}/lp_selected"]["results"][
                            str(depth)
                        ]["trial"]
                    )
                angle_jobs.append((seed, int(depth), lp_trial))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    run_case,
                    phase_root=phase_root,
                    phase=args.phase,
                    seed=seed,
                    depth=depth,
                    config_name="same_state_warm_lp_angle_assisted",
                    config=angle_config,
                    contract=contract,
                    device=args.device,
                    initial_angle_trial=lp_trial,
                    angle_assisted=True,
                    dry_run=args.dry_run,
                ): (seed, depth)
                for seed, depth, lp_trial in angle_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                seed, depth = futures[future]
                angle_results[f"seed{seed}/p{depth}"] = future.result()

        if args.dry_run:
            return {
                "schema": "n30-degree6-paper-facing-dry-run-v1",
                "phase": args.phase,
                "standalone_chains": chains,
                "angle_transfer": angle_results,
            }

        identities_by_seed: dict[str, list[str]] = {}
        for seed in seeds:
            identities = set()
            for name in configs:
                for depth in contract["stages"]["reported_depths"]:
                    row = chains[f"seed{seed}/{name}"]["results"][
                        str(depth)
                    ]
                    identities.add(
                        "|".join(
                            [
                                row["support_sha256"],
                                row["objective_sha256"],
                                row["target_manifest_sha256"],
                            ]
                        )
                    )
            for depth in contract["stages"]["reported_depths"]:
                row = angle_results[f"seed{seed}/p{depth}"]
                identities.add(
                    "|".join(
                        [
                            row["support_sha256"],
                            row["objective_sha256"],
                            row["target_manifest_sha256"],
                        ]
                    )
                )
            if len(identities) != 1:
                raise RuntimeError(
                    f"Objective identity mismatch within seed {seed}: "
                    f"{sorted(identities)}"
                )
            identities_by_seed[str(seed)] = sorted(identities)

        payload = {
            "schema": "n30-degree6-three-block-paper-facing-phase-v1",
            "status": "complete",
            "phase": args.phase,
            "contract_sha256": sha256(CONTRACT),
            "runner_sha256": sha256(RUNNER),
            "supervisor_sha256": sha256(Path(__file__)),
            "seeds": seeds,
            "reported_depths": contract["stages"]["reported_depths"],
            "history_depth": contract["stages"]["history_depth"],
            "standalone_chain_count": len(chains),
            "angle_transfer_trial_count": len(angle_results),
            "reported_trial_count": len(seeds)
            * len(contract["stages"]["reported_depths"])
            * (len(configs) + 1),
            "history_trial_count": len(seeds) * len(configs),
            "objective_identities_by_seed": identities_by_seed,
            "standalone_chains": chains,
            "angle_transfer": angle_results,
            "elapsed_sec": time.time() - started,
            "completed_unix": time.time(),
        }
        write_json(complete_path, payload)
        return payload
    except Exception as error:
        write_json(
            phase_root / "SUPERVISOR_FAILED.json",
            {
                "schema": "n30-degree6-paper-facing-phase-failure-v1",
                "status": "failed",
                "phase": args.phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_sec": time.time() - started,
                "failed_unix": time.time(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("pilot", "confirmation"), required=True
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_phase(args)
    print(
        json.dumps(
            {
                "status": result.get("status", "dry_run"),
                "phase": args.phase,
                "output_root": args.output_root,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
