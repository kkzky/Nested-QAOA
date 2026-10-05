from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "trainability_contract.json"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def domain_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def load_modules(paper_source_root: Path):
    source = paper_source_root / "source"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(paper_source_root))
    import campaign_core as core  # type: ignore
    import pspin_objective  # type: ignore

    return core, pspin_objective


def load_stage1_checkpoint_precision_compatible(
    core,
    runner,
    checkpoint_path: Path,
    *,
    dtype: str,
) -> tuple[int, torch.Tensor, dict[str, Any]]:
    if dtype != "complex128":
        return core.load_stage1_checkpoint(runner, checkpoint_path)
    data = read_json(checkpoint_path)
    p1 = int(data["p1"])
    gammas = torch.tensor(
        data["gammas"], dtype=runner.real_dtype, device=runner.device
    )
    betas = torch.tensor(
        data["betas"], dtype=runner.real_dtype, device=runner.device
    )
    with torch.no_grad():
        energies, states = runner.energy_xy(
            runner.inst.Cconf, gammas[None, :], betas[None, :], p1
        )
        state = states[0]
        state /= torch.linalg.vector_norm(state)
        energy = float(energies[0].cpu())
    # The frozen checkpoint's reference energy was evaluated in complex64.
    # A complex128 reconstruction from the same saved angles should therefore
    # be judged against the original complex64 checkpoint tolerance, not the
    # tighter tolerance used for a checkpoint natively produced in complex128.
    if not math.isclose(
        energy,
        float(data["conflict_energy"]),
        rel_tol=2e-5,
        abs_tol=2e-6,
    ):
        raise ValueError(
            "complex128 reconstruction of the complex64 Stage-1 checkpoint "
            f"mismatched: reconstructed={energy}, "
            f"saved={data['conflict_energy']}"
        )
    info = {
        "best_energy": energy,
        "selected_restart": int(data.get("selected_restart", -1)),
        "restart_energies": list(data.get("restart_energies", [])),
        "gammas": gammas,
        "betas": betas,
        "stage1_init_source": (
            f"complex128_reconstruction_of_complex64_checkpoint:"
            f"{checkpoint_path.name}"
        ),
        "stage1_state_reused": True,
    }
    runner._stage1_cache[(0, p1)] = (
        state,
        int(data.get("iterations", 0)),
        info,
    )
    runner._stage1_param_cache[p1] = (gammas, betas)
    return p1, state, data


def resize(values: np.ndarray, depth: int) -> np.ndarray:
    if len(values) == depth:
        return values.copy()
    if len(values) == 1:
        return np.repeat(values, depth)
    return np.interp(
        np.linspace(0.0, 1.0, depth),
        np.linspace(0.0, 1.0, len(values)),
        values,
    )


def previous_depth(depth: int) -> int:
    mapping = {26: 20, 29: 28, 32: 31}
    if depth not in mapping:
        raise ValueError(f"no frozen paper-history predecessor for p={depth}")
    return mapping[depth]


def one_trial(case_root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted((case_root / "trials").glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one trial under {case_root}, found {len(paths)}")
    return paths[0], read_json(paths[0])


def paper_trial(
    paper_results_root: Path,
    *,
    paper_phase: str,
    seed: int,
    config: str,
    depth: int,
) -> tuple[Path, dict[str, Any]]:
    return one_trial(
        paper_results_root
        / paper_phase
        / f"seed{seed}"
        / config
        / f"p{depth}"
    )


def point_arrays(
    *,
    seed: int,
    depth: int,
    algorithm: str,
    point_kind: str,
    point_index: int | None,
    paper_results_root: Path,
    paper_phase: str,
    config_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gamma_count = 2 * depth if algorithm == "d-XY" else depth
    if point_kind == "deterministic_linear_ramp":
        return (
            np.linspace(0.05, 0.65, gamma_count, dtype=np.float64),
            np.linspace(0.55, 0.05, depth, dtype=np.float64),
            {"source": "optimizer_deterministic_linear_ramp"},
        )
    if point_kind == "random":
        assert point_index is not None
        shared = np.random.Generator(
            np.random.PCG64DXSM(
                domain_seed(
                    "n30-degree6-trainability-v1",
                    seed,
                    depth,
                    point_index,
                    "shared-objective-and-beta",
                )
            )
        )
        objective_gamma = shared.uniform(-math.pi, math.pi, size=depth)
        beta = shared.uniform(-math.pi, math.pi, size=depth)
        if algorithm == "d-XY":
            conflict = np.random.Generator(
                np.random.PCG64DXSM(
                    domain_seed(
                        "n30-degree6-trainability-v1",
                        seed,
                        depth,
                        point_index,
                        "separated-conflict-angle",
                    )
                )
            ).uniform(-math.pi, math.pi, size=depth)
            gamma = np.concatenate((conflict, objective_gamma))
        else:
            gamma = objective_gamma
        return gamma, beta, {
            "source": "method_neutral_random_stream",
            "point_index": point_index,
        }

    final_path, final_row = paper_trial(
        paper_results_root,
        paper_phase=paper_phase,
        seed=seed,
        config=config_name,
        depth=depth,
    )
    if point_kind == "paper_final":
        return (
            np.asarray(final_row["optimizer_gammas"], dtype=np.float64),
            np.asarray(final_row["optimizer_betas"], dtype=np.float64),
            {
                "source": "completed_paper_facing_spsa_selected_angles",
                "trial_path": str(final_path),
                "trial_sha256": sha256_file(final_path),
                "trial_id": final_row["trial_id"],
            },
        )

    source_depth = previous_depth(depth)
    source_path, source_row = paper_trial(
        paper_results_root,
        paper_phase=paper_phase,
        seed=seed,
        config=config_name,
        depth=source_depth,
    )
    source_gamma = np.asarray(source_row["optimizer_gammas"], dtype=np.float64)
    source_beta = np.asarray(source_row["optimizer_betas"], dtype=np.float64)
    if algorithm == "d-XY":
        conflict = source_gamma[:source_depth]
        objective = source_gamma[source_depth:]
        if point_kind == "paper_history_identity":
            extra = depth - source_depth
            gamma = np.concatenate(
                (
                    conflict,
                    np.zeros(extra),
                    objective,
                    np.zeros(extra),
                )
            )
            beta = np.concatenate((source_beta, np.zeros(extra)))
        elif point_kind == "paper_history_interpolation":
            gamma = np.concatenate((resize(conflict, depth), resize(objective, depth)))
            beta = resize(source_beta, depth)
        else:
            raise ValueError(point_kind)
    else:
        if point_kind == "paper_history_identity":
            extra = depth - source_depth
            gamma = np.concatenate((source_gamma, np.zeros(extra)))
            beta = np.concatenate((source_beta, np.zeros(extra)))
        elif point_kind == "paper_history_interpolation":
            gamma = resize(source_gamma, depth)
            beta = resize(source_beta, depth)
        else:
            raise ValueError(point_kind)
    return gamma, beta, {
        "source": point_kind,
        "source_depth": source_depth,
        "source_trial_path": str(source_path),
        "source_trial_sha256": sha256_file(source_path),
        "source_trial_id": source_row["trial_id"],
        "final_trial_identity_checked_against": final_row["trial_id"],
    }


def expectation(
    state: torch.Tensor,
    diagonal: torch.Tensor,
) -> torch.Tensor:
    probability = torch.abs(state) ** 2
    norm = torch.clamp(torch.sum(probability), min=1e-30)
    return torch.sum(probability * diagonal) / norm


def probability(
    state: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = torch.abs(state) ** 2
    norm = torch.clamp(torch.sum(values), min=1e-30)
    return torch.sum(values[mask]) / norm


def flatten_gradients(
    gradients: Iterable[torch.Tensor | None],
    parameters: Iterable[torch.Tensor],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    return torch.cat(pieces)


def gradient_metrics(
    gradient: torch.Tensor,
    *,
    gamma_count: int,
    scale: float,
    prefix: str,
) -> dict[str, float]:
    detached = gradient.detach()
    parameter_count = int(detached.numel())
    l2 = float(torch.linalg.vector_norm(detached).cpu())
    rms = l2 / math.sqrt(parameter_count)
    gamma = detached[:gamma_count]
    beta = detached[gamma_count:]
    safe_scale = max(float(scale), 1e-30)
    return {
        f"{prefix}_gradient_l2": l2,
        f"{prefix}_gradient_rms": rms,
        f"{prefix}_gradient_rms_normalized": rms / safe_scale,
        f"{prefix}_gradient_abs_mean": float(torch.mean(torch.abs(detached)).cpu()),
        f"{prefix}_gradient_parameter_variance": float(
            torch.var(detached, unbiased=False).cpu()
        ),
        f"{prefix}_gamma_gradient_rms": float(
            torch.linalg.vector_norm(gamma).cpu()
        )
        / math.sqrt(max(1, gamma_count)),
        f"{prefix}_beta_gradient_rms": float(
            torch.linalg.vector_norm(beta).cpu()
        )
        / math.sqrt(max(1, parameter_count - gamma_count)),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    value = float(denominator.detach().cpu())
    if not math.isfinite(value) or value <= 1e-30:
        return None
    observed = float(
        torch.dot(left, right).detach().cpu() / denominator.detach().cpu()
    )
    return max(-1.0, min(1.0, observed))


def evaluate_point(
    *,
    runner,
    inst,
    bundle,
    stage1_state: torch.Tensor,
    algorithm: str,
    penalty_kappa: float,
    depth: int,
    gamma_values: np.ndarray,
    beta_values: np.ndarray,
) -> dict[str, Any]:
    weight_rms = float(bundle.manifest["weight_empirical"]["rms"])
    penalty = float(penalty_kappa * weight_rms)
    gamma = torch.tensor(
        gamma_values,
        dtype=runner.real_dtype,
        device=runner.device,
    ).requires_grad_(True)
    beta = torch.tensor(
        beta_values,
        dtype=runner.real_dtype,
        device=runner.device,
    ).requires_grad_(True)
    g = gamma[None, :]
    b = beta[None, :]
    native_diagonal = penalty * inst.Cconf + inst.Cobj
    if algorithm == "XY-LP-QAOA":
        _native, states = runner.energy_history(
            stage1_state, inst.Cobj, g, b, depth
        )
    elif algorithm == "Std-XY":
        _native, states = runner.energy_xy(native_diagonal, g, b, depth)
    elif algorithm == "d-XY":
        _native, states = runner.energy_xy(
            native_diagonal,
            g,
            b,
            depth,
            split=(penalty * inst.Cconf, inst.Cobj),
        )
    elif algorithm == "Warm-XY":
        _native, states = runner.energy_xy(
            native_diagonal,
            g,
            b,
            depth,
            initial_state=stage1_state,
        )
    else:
        raise ValueError(algorithm)
    state = states[0]
    objective_energy = expectation(state, inst.Cobj)
    conflict_energy = expectation(state, inst.Cconf)
    target_a = probability(
        state, inst.opt_mask
    )
    target_b_mask = getattr(inst, "_trainability_target_b_mask", None)
    if target_b_mask is None:
        target_b_mask = torch.zeros(
            inst.dim, dtype=torch.bool, device=inst.device
        )
        target_b_mask[
            torch.as_tensor(
                bundle.targets["target_b_k2"],
                dtype=torch.int64,
                device=inst.device,
            )
        ] = True
        inst._trainability_target_b_mask = target_b_mask
    target_b = probability(
        state, target_b_mask
    )
    feasible_mass = probability(state, inst.feasible_mask)
    parameters = (gamma, beta)
    objective_gradient = flatten_gradients(
        torch.autograd.grad(
            objective_energy,
            parameters,
            retain_graph=True,
            allow_unused=True,
        ),
        parameters,
    )
    conflict_gradient = flatten_gradients(
        torch.autograd.grad(
            conflict_energy,
            parameters,
            retain_graph=True,
            allow_unused=True,
        ),
        parameters,
    )
    log_target_a = torch.log(torch.clamp(target_a, min=1e-30))
    target_gradient = flatten_gradients(
        torch.autograd.grad(
            log_target_a,
            parameters,
            retain_graph=False,
            allow_unused=True,
        ),
        parameters,
    )
    if algorithm == "XY-LP-QAOA":
        native_gradient = objective_gradient
        native_values = inst.Cobj_np
    else:
        native_gradient = objective_gradient + penalty * conflict_gradient
        native_values = inst.Cobj_np + penalty * inst.Cconf_np
    objective_scale = float(np.std(inst.Cobj_np.astype(np.float64)))
    native_scale = float(np.std(native_values.astype(np.float64)))
    gamma_count = int(gamma.numel())
    metrics: dict[str, Any] = {
        "parameter_count": int(gamma.numel() + beta.numel()),
        "gamma_parameter_count": gamma_count,
        "beta_parameter_count": int(beta.numel()),
        "penalty": penalty,
        "weight_rms": weight_rms,
        "native_hamiltonian_full_shell_std": native_scale,
        "objective_hamiltonian_full_shell_std": objective_scale,
        "training_loss": float(
            (
                objective_energy
                if algorithm == "XY-LP-QAOA"
                else objective_energy + penalty * conflict_energy
            )
            .detach()
            .cpu()
        ),
        "objective_energy": float(objective_energy.detach().cpu()),
        "conflict_energy": float(conflict_energy.detach().cpu()),
        "target_a_probability": float(target_a.detach().cpu()),
        "target_b_probability": float(target_b.detach().cpu()),
        "feasible_probability": float(feasible_mass.detach().cpu()),
        **gradient_metrics(
            native_gradient,
            gamma_count=gamma_count,
            scale=native_scale,
            prefix="training_loss",
        ),
        **gradient_metrics(
            objective_gradient,
            gamma_count=gamma_count,
            scale=objective_scale,
            prefix="objective",
        ),
        **gradient_metrics(
            conflict_gradient,
            gamma_count=gamma_count,
            scale=float(np.std(inst.Cconf_np.astype(np.float64))),
            prefix="conflict",
        ),
        **gradient_metrics(
            target_gradient,
            gamma_count=gamma_count,
            scale=1.0,
            prefix="target_a_log_probability",
        ),
        "cosine_training_loss_vs_objective": cosine(
            native_gradient, objective_gradient
        ),
        "cosine_conflict_vs_objective": cosine(
            conflict_gradient, objective_gradient
        ),
        "cosine_training_descent_vs_target_a_ascent": cosine(
            -native_gradient, target_gradient
        ),
    }
    numeric_values = [
        value
        for value in metrics.values()
        if isinstance(value, (float, int)) and value is not None
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise FloatingPointError("non-finite trainability metric")
    del (
        states,
        state,
        objective_energy,
        conflict_energy,
        target_a,
        target_b,
        feasible_mass,
        objective_gradient,
        conflict_gradient,
        target_gradient,
        native_gradient,
        gamma,
        beta,
        g,
        b,
    )
    if inst.device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def iter_cases(
    contract: dict[str, Any],
    *,
    phase: str,
    precision_audit: bool,
) -> Iterable[tuple[int, int, str, dict[str, Any], str, int | None]]:
    if precision_audit:
        seeds = [900]
        depths = [26, 32]
        configs = contract["primary_configurations"]
        points = [
            ("deterministic_linear_ramp", None),
            ("random", 0),
            ("random", 1),
        ]
    else:
        seeds = contract["phases"][phase]["seeds"]
        depths = contract["depths"]
        configs = contract["primary_configurations"]
        points = [
            ("random", index)
            for index in range(
                contract[
                    "random_initializations_per_seed_depth_configuration"
                ]
            )
        ] + [
            ("deterministic_linear_ramp", None),
            ("paper_history_identity", None),
            ("paper_history_interpolation", None),
            ("paper_final", None),
        ]
    for seed in seeds:
        for depth in depths:
            all_configs = dict(configs)
            if not precision_audit and depth == 29:
                all_configs.update(
                    contract["penalty_sensitivity_at_depth_29"]
                )
            for config_name, config in all_configs.items():
                for point_kind, point_index in points:
                    yield (
                        int(seed),
                        int(depth),
                        str(config_name),
                        dict(config),
                        str(point_kind),
                        point_index,
                    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    paper_source_root = Path(args.paper_source_root).resolve()
    paper_results_root = Path(args.paper_results_root).resolve()
    stage1_path = paper_source_root / "stage1_p12.json"
    core, pspin_objective = load_modules(paper_source_root)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dtype = "complex128" if args.precision_audit else "complex64"
    cases = list(
        iter_cases(
            contract,
            phase=args.phase,
            precision_audit=args.precision_audit,
        )
    )
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    started = time.time()
    completed = 0
    peak_memory = 0
    current_seed: int | None = None
    context = None
    for case_index, (
        seed,
        depth,
        config_name,
        config,
        point_kind,
        point_index,
    ) in enumerate(cases):
        point_label = (
            f"{point_kind}_{point_index:03d}"
            if point_index is not None
            else point_kind
        )
        row_path = (
            output_root
            / "rows"
            / f"seed{seed}"
            / config_name
            / f"p{depth}"
            / f"{point_label}.json"
        )
        if row_path.exists() and not args.force:
            completed += 1
            continue
        if current_seed != seed:
            if context is not None:
                del context
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            streams = core.separated_streams(
                30, seed, "paper-facing-trainability-v1"
            )
            _cfg, inst, runner = core.build_structure(
                30,
                device=args.device,
                dtype=dtype,
                optimizer_seed=streams.optimizer_seed,
                opt_restarts=1,
                opt_steps=1,
                continuation=False,
                sequential_restarts=True,
                trace_every=0,
                penalty=1.0,
            )
            bundle = pspin_objective.build_bundle(
                inst,
                instance_seed=seed,
                optimizer_domain="paper-facing-trainability-v1",
            )
            p1, stage1_state, _stage1 = (
                load_stage1_checkpoint_precision_compatible(
                    core,
                    runner,
                    stage1_path,
                    dtype=dtype,
                )
            )
            if p1 != 12:
                raise AssertionError(f"unexpected Stage-1 depth {p1}")
            expected_path, expected_trial = paper_trial(
                paper_results_root,
                paper_phase=(
                    "pilot"
                    if args.phase == "pilot" or args.precision_audit
                    else "confirmation"
                ),
                seed=seed,
                config="lp_selected",
                depth=26,
            )
            observed_identity = (
                bundle.manifest["support_sha256"],
                bundle.manifest["objective_sha256"],
                bundle.manifest["target_manifest_sha256"],
            )
            expected_identity = (
                expected_trial["support_sha256"],
                expected_trial["objective_sha256"],
                expected_trial["target_manifest_sha256"],
            )
            if observed_identity[:2] != expected_identity[:2]:
                raise AssertionError(
                    f"objective identity mismatch for seed {seed}: "
                    f"{observed_identity} != {expected_identity}"
                )
            expected_target_path = (
                expected_path.parent.parent
                / "objective"
                / "target_manifest.json"
            )
            expected_targets = read_json(expected_target_path)["targets"]
            target_index_mismatches = {
                label: {
                    "observed": [
                        int(value) for value in bundle.targets[label]
                    ],
                    "expected": [
                        int(value)
                        for value in expected_targets[label]["indices"]
                    ],
                }
                for label in bundle.targets
                if [
                    int(value) for value in bundle.targets[label]
                ]
                != [
                    int(value)
                    for value in expected_targets[label]["indices"]
                ]
            }
            if target_index_mismatches:
                raise AssertionError(
                    f"target-index identity mismatch for seed {seed}: "
                    f"{target_index_mismatches}"
                )
            if dtype == "complex64" and observed_identity != expected_identity:
                raise AssertionError(
                    f"complex64 target-manifest mismatch for seed {seed}: "
                    f"{observed_identity} != {expected_identity}"
                )
            context = (
                inst,
                runner,
                bundle,
                stage1_state,
                expected_path,
                expected_identity,
                observed_identity,
                expected_target_path,
            )
            current_seed = seed
        assert context is not None
        (
            inst,
            runner,
            bundle,
            stage1_state,
            identity_trial,
            identity,
            precision_identity,
            expected_target_path,
        ) = context
        gamma, beta, point_source = point_arrays(
            seed=seed,
            depth=depth,
            algorithm=config["algorithm"],
            point_kind=point_kind,
            point_index=point_index,
            paper_results_root=paper_results_root,
            paper_phase=(
                "pilot"
                if args.phase == "pilot" or args.precision_audit
                else "confirmation"
            ),
            config_name=config_name,
        )
        if len(gamma) != (
            2 * depth if config["algorithm"] == "d-XY" else depth
        ):
            raise AssertionError("gamma dimension mismatch")
        if len(beta) != depth:
            raise AssertionError("beta dimension mismatch")
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        case_started = time.time()
        metrics = evaluate_point(
            runner=runner,
            inst=inst,
            bundle=bundle,
            stage1_state=stage1_state,
            algorithm=config["algorithm"],
            penalty_kappa=float(config["penalty_kappa"]),
            depth=depth,
            gamma_values=gamma,
            beta_values=beta,
        )
        case_peak = (
            int(torch.cuda.max_memory_allocated())
            if str(args.device).startswith("cuda")
            else 0
        )
        peak_memory = max(peak_memory, case_peak)
        row = {
            "schema": "n30-degree6-paper-facing-trainability-row-v1",
            "phase": args.phase,
            "precision_audit": bool(args.precision_audit),
            "N": 30,
            "seed": seed,
            "depth": depth,
            "configuration": config_name,
            "algorithm": config["algorithm"],
            "penalty_kappa": float(config["penalty_kappa"]),
            "point_kind": point_kind,
            "point_index": point_index,
            "dtype": dtype,
            "device": str(inst.device),
            "support_sha256": identity[0],
            "objective_sha256": identity[1],
            "target_manifest_sha256": identity[2],
            "precision_recomputed_target_manifest_sha256": (
                precision_identity[2]
            ),
            "target_index_identity_reference": str(expected_target_path),
            "target_index_identity_reference_sha256": sha256_file(
                expected_target_path
            ),
            "identity_trial_path": str(identity_trial),
            "identity_trial_sha256": sha256_file(identity_trial),
            "stage1_checkpoint_sha256": sha256_file(stage1_path),
            "stage1_checkpoint_reference_dtype": "complex64",
            "stage1_reconstruction_dtype": dtype,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "point_source": point_source,
            "gamma_sha256": hashlib.sha256(
                np.asarray(gamma, dtype="<f8").tobytes()
            ).hexdigest(),
            "beta_sha256": hashlib.sha256(
                np.asarray(beta, dtype="<f8").tobytes()
            ).hexdigest(),
            "wall_time_sec": time.time() - case_started,
            "peak_gpu_memory_bytes": case_peak,
            **metrics,
            "validation_status": "passed",
        }
        write_json(row_path, row)
        completed += 1
        write_json(
            output_root / "PROGRESS.json",
            {
                "schema": "n30-degree6-trainability-progress-v1",
                "status": "running",
                "phase": args.phase,
                "precision_audit": bool(args.precision_audit),
                "completed_cases": completed,
                "planned_cases": len(cases),
                "last_case": str(row_path),
                "elapsed_sec": time.time() - started,
                "peak_gpu_memory_bytes": peak_memory,
                "updated_unix": time.time(),
            },
        )
        print(
            json.dumps(
                {
                    "case": case_index + 1,
                    "planned": len(cases),
                    "seed": seed,
                    "depth": depth,
                    "config": config_name,
                    "point": point_label,
                    "training_gradient_rms": row[
                        "training_loss_gradient_rms"
                    ],
                    "objective_gradient_rms": row[
                        "objective_gradient_rms"
                    ],
                    "target_gradient_rms": row[
                        "target_a_log_probability_gradient_rms"
                    ],
                    "wall_sec": row["wall_time_sec"],
                    "peak_gib": case_peak / (1024**3),
                }
            ),
            flush=True,
        )
    payload = {
        "schema": "n30-degree6-paper-facing-trainability-complete-v1",
        "status": "complete",
        "phase": args.phase,
        "precision_audit": bool(args.precision_audit),
        "dtype": dtype,
        "completed_cases": completed,
        "planned_cases": len(cases),
        "elapsed_sec": time.time() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "paper_source_hashes": {
            "campaign_core.py": sha256_file(
                paper_source_root / "source" / "campaign_core.py"
            ),
            "hyper4_continuous_runner.py": sha256_file(
                paper_source_root
                / "source"
                / "hyper4_continuous_runner.py"
            ),
            "pspin_objective.py": sha256_file(
                paper_source_root / "source" / "pspin_objective.py"
            ),
            "stage1_p12.json": sha256_file(stage1_path),
        },
        "completed_unix": time.time(),
    }
    write_json(output_root / "COMPLETE.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("pilot", "confirmation"), required=True
    )
    parser.add_argument("--paper-source-root", required=True)
    parser.add_argument("--paper-results-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-audit", action="store_true")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
