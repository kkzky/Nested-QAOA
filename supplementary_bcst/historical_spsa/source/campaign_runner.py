from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import campaign_core as core
import pspin_objective
from independent_validator import sha256_file as validator_sha256_file
from independent_validator import validate_trial


SOURCE_FILES = {
    "campaign_runner": Path(__file__),
    "campaign_core": SOURCE / "campaign_core.py",
    "base_runner": SOURCE / "hyper4_continuous_runner.py",
    "resource_model": SOURCE / "ru_model.py",
    "pspin_objective": SOURCE / "pspin_objective.py",
    "engine_helper": SOURCE / "wdshcc_edge25_blockxy_nested.py",
    "independent_validator": ROOT / "independent_validator.py",
}


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def _read_source_hashes() -> dict[str, str]:
    return {name: core.sha256_file(path) for name, path in SOURCE_FILES.items()}


EXECUTION_SOURCE_HASHES = _read_source_hashes()


def source_hashes() -> dict[str, str]:
    """Return hashes captured when this process imported its executable sources."""
    return dict(EXECUTION_SOURCE_HASHES)


def trial_identifier(payload: dict[str, Any]) -> str:
    return hashlib.sha256(core.canonical_json_bytes(payload)).hexdigest()[:24]


def stage1_depth_scan(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    complete = output / "COMPLETE.json"
    if complete.exists() and not args.force:
        with open(complete, encoding="utf-8") as handle:
            result = json.load(handle)
        print(json.dumps({"status": "skipped_complete", "output": str(output)}, indent=2))
        return result
    output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    core.LAYOUTS[int(args.N)] = pspin_objective.layout_for_n(args.N)
    optimizer_seed = core.stream_seed(args.N, 0, f"stage1-{args.budget_tag}")
    _cfg, inst, runner = core.build_structure(
        args.N,
        device=args.device,
        dtype=args.dtype,
        optimizer_seed=optimizer_seed,
        opt_restarts=args.restarts,
        opt_steps=args.steps,
        opt_lr=args.lr,
        patience=args.patience,
        continuation=True,
        sequential_restarts=args.sequential_restarts,
        trace_every=args.trace_every,
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for p1 in parse_ints(args.depths):
        depth_started = time.perf_counter()
        if inst.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _success, iterations, state, info = runner.run_stage1_only(p1, 0)
        diversity = core.feasible_diversity(state, inst)
        with torch.no_grad():
            norm = float(torch.linalg.vector_norm(state).cpu())
            fidelity_feasible = float(torch.abs(torch.sum(inst.psi_feasible.conj() * state)).square().cpu())
            probability = torch.abs(state) ** 2
            conflict = float(torch.sum(probability * inst.Cconf).cpu())
        restart_diagnostics = build_stage1_restart_diagnostics(
            runner,
            inst,
            p1=p1,
            initializer_source=str(info["stage1_init_source"]),
        )
        selected_restart_replay = restart_diagnostics[int(info["stage1_selected_restart"])]
        if abs(float(selected_restart_replay["conflict_energy"]) - conflict) > 2e-5:
            raise AssertionError(
                "selected Stage-1 restart conflict-energy replay mismatch: "
                f"{selected_restart_replay['conflict_energy']} versus {conflict}"
            )
        if abs(float(selected_restart_replay["final_success"]) - float(diversity["feasible_mass"])) > 2e-5:
            raise AssertionError(
                "selected Stage-1 restart feasible-mass replay mismatch: "
                f"{selected_restart_replay['final_success']} versus {diversity['feasible_mass']}"
            )
        if inst.device.type == "cuda":
            torch.cuda.synchronize()
            peak_gpu_memory_allocated = int(torch.cuda.max_memory_allocated())
            peak_gpu_memory_reserved = int(torch.cuda.max_memory_reserved())
        else:
            peak_gpu_memory_allocated = None
            peak_gpu_memory_reserved = None
        row = {
            "schema": "bcst-stage1-depth-checkpoint-v1",
            "campaign": core.CAMPAIGN,
            "budget_tag": args.budget_tag,
            "N": args.N,
            "layout": list(inst.block_sizes),
            "p1": p1,
            "optimizer_seed": optimizer_seed,
            "optimizer_seed_effective": int(runner.cfg.seed),
            "dtype": args.dtype,
            "device": str(inst.device),
            "optimizer": {
                "name": "Adam",
                "restarts": args.restarts,
                "steps": args.steps,
                "learning_rate": args.lr,
                "patience": args.patience,
                "sequential_restarts": bool(args.sequential_restarts),
                "continuation": True,
            },
            "iterations": int(iterations),
            "conflict_energy": conflict,
            "state_norm": norm,
            "fidelity_to_exact_feasible_uniform": fidelity_feasible,
            **diversity,
            "selected_restart": int(info["stage1_selected_restart"]),
            "restart_energies": json.loads(info["stage1_restart_energies"]),
            "restart_diagnostics": restart_diagnostics,
            "initializer": info["stage1_init_source"],
            "gammas": json.loads(info["stage1_gammas"]),
            "betas": json.loads(info["stage1_betas"]),
            "wall_time_sec": time.perf_counter() - depth_started,
            "peak_gpu_memory_bytes_allocated": peak_gpu_memory_allocated,
            "peak_gpu_memory_bytes_reserved": peak_gpu_memory_reserved,
            "source_hashes": source_hashes(),
        }
        path = output / f"stage1_p{p1}.json"
        core.atomic_write_json(path, row)
        row["checkpoint_path"] = str(path)
        row["checkpoint_sha256"] = core.sha256_file(path)
        rows.append(row)
        print(
            json.dumps(
                {
                    "N": args.N,
                    "p1": p1,
                    "conflict_energy": conflict,
                    "feasible_mass": diversity["feasible_mass"],
                    "effective_support": diversity["feasible_effective_support"],
                }
            ),
            flush=True,
        )
    pareto: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = (
                other["conflict_energy"] <= candidate["conflict_energy"] + 1e-12
                and other["feasible_mass"] >= candidate["feasible_mass"] - 1e-12
                and other["feasible_effective_support"] >= candidate["feasible_effective_support"] - 1e-12
            )
            strictly_better = (
                other["conflict_energy"] < candidate["conflict_energy"] - 1e-12
                or other["feasible_mass"] > candidate["feasible_mass"] + 1e-12
                or other["feasible_effective_support"] > candidate["feasible_effective_support"] + 1e-12
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    max_feasible = max(row["feasible_mass"] for row in rows)
    eligible = [row for row in rows if row["feasible_mass"] >= 0.95 * max_feasible]
    provisional = max(
        eligible,
        key=lambda row: (
            row["feasible_effective_support"],
            -row["conflict_energy"],
            -row["p1"],
        ),
    )
    selection = {
        "schema": "bcst-stage1-provisional-selection-v1",
        "status": "requires_downstream_pareto_pilot",
        "N": args.N,
        "pareto_depths": [row["p1"] for row in pareto],
        "provisional_p1": provisional["p1"],
        "provisional_checkpoint": provisional["checkpoint_path"],
        "selection_rule": "max feasible effective support among depths within 95% of maximum feasible mass; downstream pilot required",
        "all_rows": rows,
    }
    core.atomic_write_json(output / "stage1_selection.json", selection)
    if runner.trace_rows:
        core.write_csv_lossless(output / "optimizer_trace.csv", runner.trace_rows)
    result = {
        "schema": "bcst-stage1-scan-complete-v1",
        "status": "complete",
        "N": args.N,
        "elapsed_sec": time.perf_counter() - started,
        "selection_path": str(output / "stage1_selection.json"),
        "source_hashes": source_hashes(),
        "environment": core.environment_manifest(inst.device),
    }
    core.atomic_write_json(complete, result)
    return result


def read_provisional_checkpoint(path: Path) -> Path:
    if path.name == "stage1_selection.json":
        with open(path, encoding="utf-8") as handle:
            selection = json.load(handle)
        return Path(selection["provisional_checkpoint"])
    return path


def seed_angle_cache(
    runner,
    args: argparse.Namespace,
    *,
    p1: int,
) -> dict[str, Any] | None:
    """Seed continuation with a validated selected-angle record from an earlier scan."""
    source_value = getattr(args, "initial_angle_trial", None)
    if not source_value:
        return None
    source = Path(source_value)
    with open(source, encoding="utf-8") as handle:
        row = json.load(handle)
    expected = {
        "N": int(args.N),
        "seed": int(args.seed),
        "schedule": args.schedule,
        "family": args.family,
        "algorithm": args.algorithm,
        "objective_model": args.objective_model,
    }
    mismatches = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    observed_range = row.get("range_a")
    if args.range_a is None:
        if observed_range is not None:
            mismatches["range_a"] = {"expected": None, "observed": observed_range}
    elif observed_range is None or abs(float(observed_range) - float(args.range_a)) > 1e-12:
        mismatches["range_a"] = {"expected": args.range_a, "observed": observed_range}
    if abs(float(row.get("penalty_kappa", float("nan"))) - float(args.penalty_kappa)) > 1e-12:
        mismatches["penalty_kappa"] = {
            "expected": args.penalty_kappa,
            "observed": row.get("penalty_kappa"),
        }
    if args.algorithm in {"XY-LP-QAOA", "Warm-XY"} and int(row.get("p1", -1)) != int(p1):
        mismatches["p1"] = {"expected": p1, "observed": row.get("p1")}
    if mismatches:
        raise ValueError(f"initial-angle trial identity mismatch: {mismatches}")
    depth = int(row["depth"])
    gammas = torch.tensor(
        row["optimizer_gammas"],
        dtype=runner.real_dtype,
        device=runner.device,
    )
    betas = torch.tensor(
        row["optimizer_betas"],
        dtype=runner.real_dtype,
        device=runner.device,
    )
    expected_gamma_count = 2 * depth if args.algorithm == "d-XY" else depth
    if gammas.numel() != expected_gamma_count or betas.numel() != depth:
        raise ValueError(
            "initial-angle trial has invalid angle dimensions: "
            f"depth={depth}, gamma_count={gammas.numel()}, beta_count={betas.numel()}"
        )
    if args.algorithm in {"Std-XY", "d-XY"}:
        runner._direct_param_cache[args.algorithm][depth] = (gammas, betas)
    elif args.algorithm == "XY-LP-QAOA":
        runner._stage2_param_cache.setdefault(p1, {})[depth] = (gammas, betas)
    elif args.algorithm == "Warm-XY":
        runner._warm_param_cache.setdefault(p1, {})[depth] = (gammas, betas)
    else:
        raise ValueError(f"unsupported initial-angle algorithm {args.algorithm!r}")
    return {
        "path": str(source),
        "sha256": core.sha256_file(source),
        "trial_id": row["trial_id"],
        "depth": depth,
        "selected_restart": int(row["optimizer_result"]["selected_restart"]),
        "source_runner_hash": row["runner_hash"],
    }


def restore_completed_trial_angles(
    runner,
    args: argparse.Namespace,
    row: dict[str, Any],
    *,
    p1: int,
) -> None:
    """Restore continuation state when resuming a partially completed scan."""
    expected_identity = {
        "N": int(args.N),
        "seed": int(args.seed),
        "schedule": args.schedule,
        "family": args.family,
        "algorithm": args.algorithm,
        "p1": int(p1),
    }
    mismatches = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected_identity.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"partial-scan trial identity mismatch: {mismatches}")
    depth = int(row["depth"])
    gammas = torch.tensor(
        row["optimizer_gammas"],
        dtype=runner.real_dtype,
        device=runner.device,
    )
    betas = torch.tensor(
        row["optimizer_betas"],
        dtype=runner.real_dtype,
        device=runner.device,
    )
    expected_gamma_count = 2 * depth if args.algorithm == "d-XY" else depth
    if gammas.numel() != expected_gamma_count or betas.numel() != depth:
        raise ValueError(
            "partial-scan trial has invalid angle dimensions: "
            f"depth={depth}, gamma_count={gammas.numel()}, beta_count={betas.numel()}"
        )
    if args.algorithm in {"Std-XY", "d-XY"}:
        runner._direct_param_cache[args.algorithm][depth] = (gammas, betas)
    elif args.algorithm == "XY-LP-QAOA":
        runner._stage2_param_cache.setdefault(p1, {})[depth] = (gammas, betas)
    elif args.algorithm == "Warm-XY":
        runner._warm_param_cache.setdefault(p1, {})[depth] = (gammas, betas)
    else:
        raise ValueError(f"unsupported partial-scan algorithm {args.algorithm!r}")


def optimize_one(
    runner,
    inst,
    bundle,
    *,
    algorithm: str,
    p1: int,
    depth: int,
    trial_index: int,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    if algorithm == "XY-LP-QAOA":
        _success, iterations, state, info = runner.run_xy_nested(p1, depth, trial_index)
    elif algorithm == "Std-XY":
        _success, iterations, state, info = runner.run_std_xy(depth, trial_index)
    elif algorithm == "d-XY":
        _success, iterations, state, info = runner.run_decoupled_xy(depth, trial_index)
    elif algorithm == "Warm-XY":
        _success, iterations, state, info = runner.run_xy_warm(p1, depth, trial_index)
    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    return state, int(iterations), info


def extract_angles(algorithm: str, info: dict[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    if algorithm == "XY-LP-QAOA":
        gammas = json.loads(info["stage2_gammas"])
        betas = json.loads(info["stage2_betas"])
        optimizer = {
            "selected_restart": int(info["stage2_selected_restart"]),
            "restart_energies": json.loads(info["stage2_restart_energies"]),
            "initializer": info["stage2_init_source"],
            "expected_energy": float(info["stage2_energy"]),
            "restart_execution": info["stage2_restart_execution"],
        }
    else:
        gammas = json.loads(info["optimizer_gammas"])
        betas = json.loads(info["optimizer_betas"])
        optimizer = {
            "selected_restart": int(info["optimizer_selected_restart"]),
            "restart_energies": json.loads(info["optimizer_restart_energies"]),
            "initializer": info["optimizer_init_source"],
            "expected_energy": float(info["energy"]),
            "restart_execution": info["optimizer_restart_execution"],
        }
    return gammas, betas, optimizer


def replay_restart_state(
    runner,
    inst,
    *,
    algorithm: str,
    depth: int,
    stage1_state: torch.Tensor | None,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> torch.Tensor:
    g = torch.tensor([list(gammas)], dtype=runner.real_dtype, device=inst.device)
    b = torch.tensor([list(betas)], dtype=runner.real_dtype, device=inst.device)
    with torch.no_grad():
        if algorithm == "XY-LP-QAOA":
            if stage1_state is None:
                raise ValueError("restart replay for XY-LP-QAOA requires Stage-1 state")
            _energy, states = runner.energy_history(stage1_state, inst.Cobj, g, b, depth)
        elif algorithm == "Std-XY":
            h = runner.penalty * inst.Cconf + inst.Cobj
            _energy, states = runner.energy_xy(h, g, b, depth)
        elif algorithm == "d-XY":
            h_pen = runner.penalty * inst.Cconf
            _energy, states = runner.energy_xy(
                h_pen + inst.Cobj,
                g,
                b,
                depth,
                split=(h_pen, inst.Cobj),
            )
        elif algorithm == "Warm-XY":
            if stage1_state is None:
                raise ValueError("restart replay for Warm-XY requires Stage-1 state")
            h = runner.penalty * inst.Cconf + inst.Cobj
            _energy, states = runner.energy_xy(h, g, b, depth, initial_state=stage1_state)
        else:
            raise ValueError(f"unsupported restart replay algorithm {algorithm!r}")
        state = states[0].detach().clone()
        state /= torch.clamp(torch.linalg.vector_norm(state), min=1e-20)
    return state


def replay_stage1_restart_state(
    runner,
    inst,
    *,
    p1: int,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> torch.Tensor:
    g = torch.tensor([list(gammas)], dtype=runner.real_dtype, device=inst.device)
    b = torch.tensor([list(betas)], dtype=runner.real_dtype, device=inst.device)
    with torch.no_grad():
        _energy, states = runner.energy_xy(inst.Cconf, g, b, p1)
        state = states[0].detach().clone()
        state /= torch.clamp(torch.linalg.vector_norm(state), min=1e-20)
    return state


def build_stage1_restart_diagnostics(
    runner,
    inst,
    *,
    p1: int,
    initializer_source: str,
) -> list[dict[str, Any]]:
    info = getattr(runner, "_last_optimizer_info", None)
    if not isinstance(info, dict):
        raise AssertionError("Stage-1 optimizer did not expose restart provenance")
    restart_gammas = info.get("restart_gammas", [])
    restart_betas = info.get("restart_betas", [])
    restart_energies = info.get("restart_energies", [])
    restart_best_steps = info.get("restart_best_steps", [])
    initializer_roles = info.get("restart_initializer_roles", [])
    expected = int(runner.cfg.opt_restarts)
    lengths = {
        len(restart_gammas),
        len(restart_betas),
        len(restart_energies),
        len(restart_best_steps),
        len(initializer_roles),
    }
    if lengths != {expected}:
        raise AssertionError(
            f"incomplete Stage-1 restart provenance: expected {expected}, got {sorted(lengths)}"
        )
    seed = int(info["restart_seed"])
    stop_reason = str(info["stopping_reason"])
    diagnostics: list[dict[str, Any]] = []
    for restart in range(expected):
        state = replay_stage1_restart_state(
            runner,
            inst,
            p1=p1,
            gammas=restart_gammas[restart],
            betas=restart_betas[restart],
        )
        with torch.no_grad():
            probability = torch.abs(state) ** 2
            probability /= torch.clamp(torch.sum(probability), min=1e-20)
            conflict_energy = float(torch.sum(probability * inst.Cconf).detach().cpu())
            fidelity_feasible = float(
                torch.abs(torch.sum(inst.psi_feasible.conj() * state)).square().detach().cpu()
            )
        diversity = core.feasible_diversity(state, inst)
        role = str(initializer_roles[restart])
        diagnostics.append(
            {
                "restart_index": restart,
                "optimizer_seed": seed,
                "seed_scope": "shared_optimizer_stream_with_restart_index",
                "initializer_role": role,
                "initializer_source": initializer_source if role == "continuation" else role,
                "best_step": int(restart_best_steps[restart]),
                "stopping_reason": stop_reason,
                "final_energy": float(restart_energies[restart]),
                "final_success": float(diversity["feasible_mass"]),
                "conflict_energy": conflict_energy,
                "fidelity_to_exact_feasible_uniform": fidelity_feasible,
                "feasible_diversity": diversity,
                "gammas": [float(value) for value in restart_gammas[restart]],
                "betas": [float(value) for value in restart_betas[restart]],
            }
        )
    return diagnostics


def build_restart_diagnostics(
    runner,
    inst,
    bundle,
    *,
    algorithm: str,
    depth: int,
    stage1_state: torch.Tensor | None,
    initializer_source: str,
) -> list[dict[str, Any]]:
    info = getattr(runner, "_last_optimizer_info", None)
    if not isinstance(info, dict):
        raise AssertionError("optimizer did not expose restart provenance")
    restart_gammas = info.get("restart_gammas", [])
    restart_betas = info.get("restart_betas", [])
    restart_energies = info.get("restart_energies", [])
    restart_best_steps = info.get("restart_best_steps", [])
    initializer_roles = info.get("restart_initializer_roles", [])
    expected = int(runner.cfg.opt_restarts)
    lengths = {
        len(restart_gammas),
        len(restart_betas),
        len(restart_energies),
        len(restart_best_steps),
        len(initializer_roles),
    }
    if lengths != {expected}:
        raise AssertionError(f"incomplete restart provenance: expected {expected}, got {sorted(lengths)}")
    seed = int(info["restart_seed"])
    stop_reason = str(info["stopping_reason"])
    diagnostics: list[dict[str, Any]] = []
    for restart in range(expected):
        state = replay_restart_state(
            runner,
            inst,
            algorithm=algorithm,
            depth=depth,
            stage1_state=stage1_state,
            gammas=restart_gammas[restart],
            betas=restart_betas[restart],
        )
        role = str(initializer_roles[restart])
        replay_probabilities = core.replay_probabilities(state, inst, bundle)
        diagnostics.append(
            {
                "restart_index": restart,
                "optimizer_seed": seed,
                "seed_scope": "shared_optimizer_stream_with_restart_index",
                "initializer_role": role,
                "initializer_source": initializer_source if role == "continuation" else role,
                "best_step": int(restart_best_steps[restart]),
                "stopping_reason": stop_reason,
                "final_energy": float(restart_energies[restart]),
                "final_success": {
                    key: float(value)
                    for key, value in replay_probabilities.items()
                    if key
                    in {
                        "target_a_sqrt",
                        "target_b_k2",
                        "exact_ground",
                        "manuscript_fraction",
                        "feasible_probability",
                    }
                },
                "gammas": [float(value) for value in restart_gammas[restart]],
                "betas": [float(value) for value in restart_betas[restart]],
                "replay_probabilities": replay_probabilities,
            }
        )
    return diagnostics


def algorithm_scan(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    complete = output / "COMPLETE.json"
    if complete.exists() and not args.force:
        with open(complete, encoding="utf-8") as handle:
            result = json.load(handle)
        print(json.dumps({"status": "skipped_complete", "output": str(output)}, indent=2))
        return result
    output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.objective_model == pspin_objective.OBJECTIVE_MODEL:
        # The original source hard-coded N=30.  The paper-facing scaling
        # extension keeps five balanced blocks and exactly two selected
        # variables per block for every consecutive N.
        core.LAYOUTS[int(args.N)] = pspin_objective.layout_for_n(args.N)
    streams = core.separated_streams(args.N, args.seed, f"{args.algorithm}-{args.budget_tag}")
    _cfg, inst, runner = core.build_structure(
        args.N,
        device=args.device,
        dtype=args.dtype,
        optimizer_seed=streams.optimizer_seed,
        opt_restarts=args.restarts,
        opt_steps=args.steps,
        opt_lr=args.lr,
        optimizer_name=args.optimizer,
        spsa_c=args.spsa_c,
        spsa_target_step=args.spsa_target_step,
        spsa_alpha=args.spsa_alpha,
        spsa_gamma=args.spsa_gamma,
        spsa_stability_fraction=args.spsa_stability_fraction,
        spsa_calibration_directions=args.spsa_calibration_directions,
        spsa_directions_per_update=args.spsa_directions_per_update,
        spsa_max_step_norm=args.spsa_max_step_norm,
        patience=args.patience,
        continuation=True,
        sequential_restarts=args.sequential_restarts,
        trace_every=args.trace_every,
        penalty=1.0,
    )
    if args.objective_model == pspin_objective.OBJECTIVE_MODEL:
        if args.schedule != "P3":
            raise ValueError("block_state_p3 requires the P3 schedule")
        if args.family != "signed_cont_uniform" or args.range_a != 0.5:
            raise ValueError(
                "block_state_p3 requires signed_cont_uniform with "
                "--range-a 0.5"
            )
        bundle = pspin_objective.build_bundle(
            inst,
            instance_seed=args.seed,
            optimizer_domain=f"{args.algorithm}-{args.budget_tag}",
        )
    else:
        if args.schedule == "P3":
            raise ValueError(
                "P3 schedule requires --objective-model block_state_p3"
            )
        bundle = core.build_objective_bundle(
            inst,
            instance_seed=args.seed,
            schedule=args.schedule,
            family=args.family,
            range_a=args.range_a,
            optimizer_domain=f"{args.algorithm}-{args.budget_tag}",
        )
    objective_dir = output / "objective"
    if args.objective_model == pspin_objective.OBJECTIVE_MODEL:
        pspin_objective.save_bundle(objective_dir, bundle)
    else:
        core.save_objective_bundle(objective_dir, bundle)
    weight_rms = float(bundle.manifest["weight_empirical"]["rms"])
    penalty = float(args.penalty_kappa * weight_rms)
    runner.penalty = penalty
    p1 = 0
    stage1_state = None
    stage1_checkpoint = None
    if args.algorithm in {"XY-LP-QAOA", "Warm-XY"}:
        if not args.stage1_checkpoint:
            raise ValueError(f"{args.algorithm} requires --stage1-checkpoint")
        stage1_checkpoint = read_provisional_checkpoint(Path(args.stage1_checkpoint))
        p1, stage1_state, _stage1_data = core.load_stage1_checkpoint(runner, stage1_checkpoint)
    initial_angle_seed = seed_angle_cache(runner, args, p1=p1)
    trace_path = output / "optimizer_trace.csv"
    if trace_path.exists() and not args.force:
        with trace_path.open(encoding="utf-8", newline="") as handle:
            runner.trace_rows = list(csv.DictReader(handle))

    scan_contract = {
        "schema": "bcst-algorithm-scan-contract-v1",
        "campaign": core.CAMPAIGN,
        "N": args.N,
        "seed": args.seed,
        "schedule": args.schedule,
        "family": args.family,
        "objective_model": args.objective_model,
        "objective_degree": int(
            bundle.manifest.get("polynomial_degree", 4)
        ),
        "range_a": args.range_a,
        "algorithm": args.algorithm,
        "depths": parse_ints(args.depths),
        "stage1_checkpoint": None if stage1_checkpoint is None else str(stage1_checkpoint),
        "penalty_kappa": args.penalty_kappa,
        "penalty": penalty,
        "budget_tag": args.budget_tag,
        "initial_angle_seed": initial_angle_seed,
        "optimizer": {
            "name": args.optimizer,
            "restarts": args.restarts,
            "steps": args.steps,
            "learning_rate": args.lr,
            "patience": args.patience,
            "sequential_restarts": bool(args.sequential_restarts),
            "trace_every": args.trace_every,
            "spsa": (
                {
                    "c": args.spsa_c,
                    "target_step": args.spsa_target_step,
                    "alpha": args.spsa_alpha,
                    "gamma": args.spsa_gamma,
                    "stability_fraction": args.spsa_stability_fraction,
                    "calibration_directions": args.spsa_calibration_directions,
                    "directions_per_update": args.spsa_directions_per_update,
                    "max_step_norm": args.spsa_max_step_norm,
                    "incumbent_preserved": True,
                    "gamma_wrapping": False,
                }
                if args.optimizer == "spsa"
                else None
            ),
        },
        "source_hashes": source_hashes(),
    }
    core.atomic_write_json(output / "scan_contract.json", scan_contract)
    started = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(parse_ints(args.depths)):
        identity = {
            "N": args.N,
            "seed": args.seed,
            "schedule": args.schedule,
            "family": args.family,
            "objective_model": args.objective_model,
            "range_a": args.range_a,
            "algorithm": args.algorithm,
            "p1": p1,
            "depth": depth,
            "penalty_kappa": args.penalty_kappa,
            "budget_tag": args.budget_tag,
        }
        trial_id = trial_identifier(identity)
        trial_path = output / "trials" / f"{trial_id}.json"
        if trial_path.exists() and not args.force:
            with open(trial_path, encoding="utf-8") as handle:
                completed_row = json.load(handle)
            restore_completed_trial_angles(
                runner,
                args,
                completed_row,
                p1=p1,
            )
            trial_rows.append(completed_row)
            continue
        trial_started = time.perf_counter()
        if inst.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        state, iterations, info = optimize_one(
            runner,
            inst,
            bundle,
            algorithm=args.algorithm,
            p1=p1,
            depth=depth,
            trial_index=0,
        )
        replays = core.replay_probabilities(state, inst, bundle)
        diversity = core.feasible_diversity(state, inst)
        resource = core.resources(
            algorithm=args.algorithm,
            n=args.N,
            layout=inst.block_sizes,
            conflict_edges=len(inst.conflict_edges),
            m=len(bundle.terms),
            objective_degree=int(
                bundle.manifest.get("polynomial_degree", 4)
            ),
            p1=p1,
            depth=depth,
        )
        costs = core.target_costs(replays, resource["terminal_RU"])
        gammas, betas, optimizer = extract_angles(args.algorithm, info)
        optimizer_runtime = getattr(runner, "_last_optimizer_info", {})
        optimizer.update(
            {
                "name": str(
                    optimizer_runtime.get("optimizer_name", args.optimizer)
                ),
                "objective_calls_total": int(
                    optimizer_runtime.get("objective_calls_total", 0)
                ),
                "gradient_calls_total": int(
                    optimizer_runtime.get("gradient_calls_total", 0)
                ),
                "hardware_circuit_proxy_total": int(
                    optimizer_runtime.get("hardware_circuit_proxy_total", 0)
                ),
                "restart_objective_calls": optimizer_runtime.get(
                    "restart_objective_calls", []
                ),
                "restart_initial_energies": optimizer_runtime.get(
                    "restart_initial_energies", []
                ),
                "spsa_calibration": optimizer_runtime.get(
                    "spsa_calibration", []
                ),
                "spsa_settings": optimizer_runtime.get("spsa_settings"),
            }
        )
        if args.optimizer == "spsa":
            initial_energies = [
                float(value)
                for value in optimizer_runtime.get(
                    "restart_initial_energies", []
                )
            ]
            best_energies = [
                float(value)
                for value in optimizer_runtime.get("restart_energies", [])
            ]
            if len(initial_energies) != args.restarts or len(
                best_energies
            ) != args.restarts:
                raise AssertionError(
                    "SPSA did not record every restart incumbent"
                )
            violations = [
                best - initial
                for best, initial in zip(best_energies, initial_energies)
                if best > initial + 1e-6
            ]
            if violations:
                raise AssertionError(
                    f"SPSA lost an initial incumbent: {violations}"
                )
            expected_calls = (
                args.restarts
                * (
                    2
                    + 2 * args.spsa_calibration_directions
                    + 2
                    * args.spsa_directions_per_update
                    * args.steps
                )
                + 1
            )
            observed_calls = int(
                optimizer_runtime.get("objective_calls_total", -1)
            )
            if observed_calls != expected_calls:
                raise AssertionError(
                    "SPSA circuit-evaluation accounting mismatch: "
                    f"expected {expected_calls}, observed {observed_calls}"
                )
            optimizer["incumbent_preservation_validated"] = True
            optimizer["objective_call_accounting_validated"] = True
        optimizer["restart_diagnostics"] = build_restart_diagnostics(
            runner,
            inst,
            bundle,
            algorithm=args.algorithm,
            depth=depth,
            stage1_state=stage1_state,
            initializer_source=str(optimizer["initializer"]),
        )
        selected_restart_replay = optimizer["restart_diagnostics"][
            int(optimizer["selected_restart"])
        ]["replay_probabilities"]
        for key, value in replays.items():
            if abs(float(selected_restart_replay[key]) - float(value)) > 2e-5:
                raise AssertionError(
                    f"selected restart replay mismatch for {key}: "
                    f"{selected_restart_replay[key]} versus {value}"
                )
        if inst.device.type == "cuda":
            torch.cuda.synchronize()
            peak_gpu_memory_allocated = int(torch.cuda.max_memory_allocated())
            peak_gpu_memory_reserved = int(torch.cuda.max_memory_reserved())
        else:
            peak_gpu_memory_allocated = None
            peak_gpu_memory_reserved = None
        row = {
            "schema": "bcst-signed-continuous-trial-v1",
            "trial_id": trial_id,
            "campaign": core.CAMPAIGN,
            "runner_hash": source_hashes()["campaign_runner"],
            "validator_hash": source_hashes()["independent_validator"],
            "accounting_tag": core.ACCOUNTING_TAG,
            **identity,
            "layout": list(inst.block_sizes),
            "block_weights": list(inst.block_weights),
            "product_shell_dimension": int(inst.dim),
            "conflict_feasible_count": int(inst.feasible_count),
            "conflict_edge_list": [[int(u), int(v)] for u, v, _weight in inst.conflict_edges],
            "stream_seeds": bundle.manifest["stream_seeds"],
            "optimizer_seed_effective": int(runner.cfg.seed),
            "term_count": len(bundle.terms),
            "objective_model": args.objective_model,
            "objective_degree": int(
                bundle.manifest.get("polynomial_degree", 4)
            ),
            "support_sha256": bundle.manifest["support_sha256"],
            "objective_sha256": bundle.manifest["objective_sha256"],
            "target_manifest_sha256": bundle.manifest["target_manifest_sha256"],
            "objective_manifest": bundle.manifest,
            "circuit_definition": (
                "actual optimized Stage-1 state; objective-only Stage 2; full-register rank-one history projector"
                if args.algorithm == "XY-LP-QAOA"
                else (
                    "exact same optimized Stage-1 state as LP; fixed nonzero conflict penalty plus objective; full within-block XY"
                    if args.algorithm == "Warm-XY"
                    else "native product block-weight-two state; fixed nonzero conflict penalty plus objective; full within-block XY"
                )
            ),
            "p1": p1,
            "depth": depth,
            "penalty_kappa": args.penalty_kappa,
            "penalty": penalty,
            "phases": "decoupled" if args.algorithm == "d-XY" else "coupled",
            "optimizer_settings": scan_contract["optimizer"],
            "optimizer_result": optimizer,
            "optimizer_gammas": gammas,
            "optimizer_betas": betas,
            "iterations": iterations,
            "stage1_checkpoint": None if stage1_checkpoint is None else str(stage1_checkpoint),
            "stage1_checkpoint_sha256": None if stage1_checkpoint is None else core.sha256_file(stage1_checkpoint),
            "stage1_conflict_energy": None if p1 == 0 else float(info["stage1_energy"]),
            "stage1_feasible_mass": None if p1 == 0 else float(info["stage1_feasible_mass"]),
            "replay_probabilities": replays,
            "feasible_diversity": diversity,
            "resources": resource,
            "target_costs": costs,
            "wall_time_sec": time.perf_counter() - trial_started,
            "peak_gpu_memory_bytes_allocated": peak_gpu_memory_allocated,
            "peak_gpu_memory_bytes_reserved": peak_gpu_memory_reserved,
            "dtype": args.dtype,
            "device": str(inst.device),
            "source_hashes": source_hashes(),
            "completion_status": "complete",
            "validation_status": "pending",
        }
        if args.save_states:
            checkpoint = core.state_checkpoint(
                output / "states" / f"{trial_id}.npy",
                state,
                metadata={"trial_id": trial_id, "algorithm": args.algorithm},
            )
            row["state_checkpoint"] = checkpoint
        audit = validate_trial(row)
        row["validation_status"] = audit["status"]
        if audit["status"] != "passed":
            raise AssertionError(audit)
        core.atomic_write_json(trial_path, row)
        core.atomic_write_json(output / "audits" / f"{trial_id}.json", audit)
        trial_rows.append(row)
        print(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "algorithm": args.algorithm,
                    "depth": depth,
                    "target_a": replays["target_a_sqrt"],
                    "target_b": replays["target_b_k2"],
                    "target_a_cost": costs["target_a_sqrt_cost"],
                    "target_b_cost": costs["target_b_k2_cost"],
                }
            ),
            flush=True,
        )
    if runner.trace_rows:
        core.write_csv_lossless(output / "optimizer_trace.csv", runner.trace_rows)
    flattened: list[dict[str, Any]] = []
    for row in trial_rows:
        flattened.append(
            {
                "trial_id": row["trial_id"],
                "N": row["N"],
                "seed": row["seed"],
                "schedule": row["schedule"],
                "family": row["family"],
                "objective_model": row["objective_model"],
                "objective_degree": row["objective_degree"],
                "range_a": row["range_a"],
                "algorithm": row["algorithm"],
                "p1": row["p1"],
                "depth": row["depth"],
                "penalty_kappa": row["penalty_kappa"],
                "penalty": row["penalty"],
                "target_a_probability": row["replay_probabilities"]["target_a_sqrt"],
                "target_b_probability": row["replay_probabilities"]["target_b_k2"],
                "exact_probability": row["replay_probabilities"]["exact_ground"],
                "manuscript_probability": row["replay_probabilities"]["manuscript_fraction"],
                "feasible_probability": row["replay_probabilities"]["feasible_probability"],
                "circuit_RU": row["resources"]["circuit_RU"],
                "terminal_RU": row["resources"]["terminal_RU"],
                "target_a_RTS99": row["target_costs"]["target_a_sqrt_RTS99"],
                "target_a_cost": row["target_costs"]["target_a_sqrt_cost"],
                "target_b_RTS99": row["target_costs"]["target_b_k2_RTS99"],
                "target_b_cost": row["target_costs"]["target_b_k2_cost"],
                "validation_status": row["validation_status"],
            }
        )
    core.write_csv_lossless(output / "trial_rows.csv", flattened)
    result = {
        "schema": "bcst-algorithm-scan-complete-v1",
        "status": "complete",
        "trial_count": len(trial_rows),
        "elapsed_sec": time.perf_counter() - started,
        "scan_contract_sha256": core.sha256_file(output / "scan_contract.json"),
        "trial_rows_sha256": core.sha256_file(output / "trial_rows.csv"),
        "environment": core.environment_manifest(inst.device),
        "all_trials_validated": all(row["validation_status"] == "passed" for row in trial_rows),
    }
    core.atomic_write_json(complete, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1-scan")
    stage1.add_argument("--N", type=int, required=True)
    stage1.add_argument("--depths", required=True)
    stage1.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    stage1.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    stage1.add_argument("--restarts", type=int, default=3)
    stage1.add_argument("--steps", type=int, default=85)
    stage1.add_argument("--lr", type=float, default=0.055)
    stage1.add_argument("--patience", type=int, default=18)
    stage1.add_argument("--trace-every", type=int, default=5)
    stage1.add_argument("--sequential-restarts", action="store_true")
    stage1.add_argument("--budget-tag", default="screening")
    stage1.add_argument("--output", required=True)
    stage1.add_argument("--force", action="store_true")

    scan = subparsers.add_parser("scan")
    scan.add_argument("--N", type=int, required=True)
    scan.add_argument("--seed", type=int, required=True)
    scan.add_argument("--schedule", choices=sorted(core.TERM_SCHEDULES), required=True)
    scan.add_argument(
        "--objective-model",
        choices=("mixed_quartic", pspin_objective.OBJECTIVE_MODEL),
        default="mixed_quartic",
    )
    scan.add_argument(
        "--family",
        choices=(
            "pos_disc_rms1",
            "signed_disc_rms1",
            "pos_cont_rms1",
            "signed_cont_rms1",
            "signed_cont_uniform",
            "signed_cont_u1",
        ),
        required=True,
    )
    scan.add_argument("--range-a", type=float)
    scan.add_argument("--algorithm", choices=("XY-LP-QAOA", "Std-XY", "d-XY", "Warm-XY"), required=True)
    scan.add_argument("--depths", required=True)
    scan.add_argument("--stage1-checkpoint")
    scan.add_argument("--initial-angle-trial")
    scan.add_argument("--penalty-kappa", type=float, default=4.0)
    scan.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    scan.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    scan.add_argument("--restarts", type=int, default=3)
    scan.add_argument("--steps", type=int, default=85)
    scan.add_argument("--lr", type=float, default=0.055)
    scan.add_argument(
        "--optimizer",
        choices=("adam", "spsa"),
        default="adam",
    )
    scan.add_argument("--spsa-c", type=float, default=0.10)
    scan.add_argument("--spsa-target-step", type=float, default=0.10)
    scan.add_argument("--spsa-alpha", type=float, default=0.602)
    scan.add_argument("--spsa-gamma", type=float, default=0.101)
    scan.add_argument(
        "--spsa-stability-fraction",
        type=float,
        default=0.10,
    )
    scan.add_argument(
        "--spsa-calibration-directions",
        type=int,
        default=8,
    )
    scan.add_argument(
        "--spsa-directions-per-update",
        type=int,
        default=1,
    )
    scan.add_argument(
        "--spsa-max-step-norm",
        type=float,
        default=0.50,
    )
    scan.add_argument("--patience", type=int, default=18)
    scan.add_argument("--trace-every", type=int, default=5)
    scan.add_argument("--sequential-restarts", action="store_true")
    scan.add_argument("--save-states", action="store_true")
    scan.add_argument("--budget-tag", default="screening")
    scan.add_argument("--output", required=True)
    scan.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "stage1-scan":
        result = stage1_depth_scan(args)
    else:
        result = algorithm_scan(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
