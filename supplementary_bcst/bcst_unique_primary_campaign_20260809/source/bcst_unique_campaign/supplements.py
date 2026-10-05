"""Validation gates for the frozen-angle panel and eligible historical data."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .contract import CampaignContract, canonical_json_bytes, sha256_file
from .specs import publish_json


class SupplementError(ValueError):
    """Raised when a required supplement is missing or substitutes new settings."""


EXPECTED_ANGLE_MANIFEST_SHA256 = (
    "5FDEC751C9D6845AE0DEAC575769857F8C86955D41A1DCC665E663C7085BCB77"
)
EXPECTED_EXISTING_EVIDENCE_MANIFEST_SHA256 = (
    "3FE475FD05BF7E9AF401E3837F640EC5349D3367259162B6E4C42D9A80A3CFA2"
)
GRADIENT_METHODS = (
    "learned_projector",
    "ordinary_native_direct",
    "decoupled_native_direct",
    "same_stage1_state_direct",
)
METHOD_CONFIG = {
    "learned_projector": ("lp_selected", "XY-LP-QAOA", 4.0),
    "ordinary_native_direct": ("ordinary_selected", "Std-XY", 6.0),
    "decoupled_native_direct": ("separated_selected", "d-XY", 8.0),
    "same_stage1_state_direct": ("same_state_warm_selected", "Warm-XY", 3.0),
}
SHA256_RE = re.compile(r"[0-9A-Fa-f]{64}")
GRADIENT_ROW_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "input_manifest_sha256",
        "N",
        "problem_seed",
        "method",
        "depth",
        "angle_index",
        "angle_vector_sha256",
        "gamma_sha256",
        "beta_sha256",
        "source_row_sha256",
        "source_configuration",
        "support_sha256",
        "objective_sha256",
        "stage1_checkpoint_sha256",
        "historical_contract_sha256",
        "historical_runner_sha256",
        "historical_target_manifest_sha256",
        "ground_state_sha256",
        "ground_index_decimal",
        "ground_bitstring",
        "ground_energy_float64_hex",
        "ground_degeneracy",
        "historical_state_amplitude_sha256",
        "parameter_count",
        "training_loss",
        "training_loss_gradient_rms",
        "training_loss_gradient_rms_normalized",
        "common_objective_gradient_rms",
        "common_objective_gradient_rms_normalized",
        "unique_probability_unclipped",
        "feasible_probability",
        "log_probability_floor",
        "log_probability_clipped",
        "log_unique_probability_gradient_l2",
        "log_unique_probability_gradient_rms",
        "training_descent_alignment_unique",
        "alignment_defined",
        "alignment_interpretation",
        "optimization_runs",
        "resampled",
        "selected_by_target",
        "historical_training_gradient_reproduced",
        "status",
        "row_content_sha256",
    }
)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupplementError(f"cannot read supplement {path}") from exc
    if not isinstance(value, dict):
        raise SupplementError(f"supplement {path} must contain an object")
    return value


def _within(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise SupplementError(f"{label} escapes the campaign package")
    return resolved


def _angle_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        gamma = np.asarray(
            [float.fromhex(str(value)) for value in row["gamma_float64_hex"]],
            dtype="<f8",
        )
        beta = np.asarray(
            [float.fromhex(str(value)) for value in row["beta_float64_hex"]],
            dtype="<f8",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SupplementError("fixed-angle row has invalid float64-hex values") from exc
    if gamma.ndim != 1 or beta.ndim != 1 or not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(beta)):
        raise SupplementError("fixed-angle vectors must be finite one-dimensional float64")
    gamma_bytes = np.ascontiguousarray(gamma, dtype="<f8").tobytes(order="C")
    beta_bytes = np.ascontiguousarray(beta, dtype="<f8").tobytes(order="C")
    if (
        len(gamma) != row.get("gamma_parameter_count")
        or len(beta) != row.get("beta_parameter_count")
        or hashlib.sha256(gamma_bytes).hexdigest()
        != str(row.get("gamma_sha256", "")).lower()
        or hashlib.sha256(beta_bytes).hexdigest()
        != str(row.get("beta_sha256", "")).lower()
    ):
        raise SupplementError("fixed-angle component hashes do not reproduce")
    vector_bytes = (
        struct.pack("<Q", len(gamma))
        + gamma_bytes
        + struct.pack("<Q", len(beta))
        + beta_bytes
    )
    if hashlib.sha256(vector_bytes).hexdigest() != str(
        row.get("angle_vector_sha256", "")
    ).lower():
        raise SupplementError("fixed-angle vector hash does not reproduce")
    return gamma, beta


def audit_gradient_inputs(
    contract: CampaignContract,
    input_manifest_path: Path,
) -> tuple[dict[str, Any], list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]]]:
    """Verify exact source, row, and explicit-vector bytes before GPU replay."""

    campaign_root = contract.path.parent.resolve()
    manifest_path = _within(campaign_root, input_manifest_path, "angle manifest")
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != EXPECTED_ANGLE_MANIFEST_SHA256:
        raise SupplementError(
            "fixed-angle manifest differs from the recovered frozen 864-vector registry"
        )
    manifest = _read(manifest_path)
    if (
        manifest.get("schema") != "bcst-fixed-angle-panel-input-manifest-v1"
        or manifest.get("status") != "complete"
        or manifest.get("setting_count") != 864
        or manifest.get("optimization_runs") != 0
        or manifest.get("resampled") is not False
        or manifest.get("source_panel")
        != "n30-degree6-paper-facing-trainability-v1"
    ):
        raise SupplementError("fixed-angle manifest header is incompatible")
    copied = manifest.get("copied_source_files")
    if not isinstance(copied, Mapping) or not copied:
        raise SupplementError("fixed-angle manifest lacks copied-source bindings")
    for relative, binding in copied.items():
        if not isinstance(relative, str) or not isinstance(binding, Mapping):
            raise SupplementError("copied-source binding is malformed")
        path = _within(campaign_root, campaign_root / relative, "copied source")
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("bytes")
            or sha256_file(path) != str(binding.get("sha256", "")).upper()
        ):
            raise SupplementError(f"copied historical source changed: {relative}")

    exact_identity = manifest.get("exact_target_identity")
    if not isinstance(exact_identity, Mapping):
        raise SupplementError("fixed-angle manifest lacks exact unique-target identity")
    reference_relative = exact_identity.get("objective_identity_reference_relative_path")
    if not isinstance(reference_relative, str):
        raise SupplementError("objective identity reference path is missing")
    reference_path = _within(
        campaign_root,
        campaign_root / reference_relative,
        "objective identity reference",
    )
    if sha256_file(reference_path) != str(
        exact_identity.get("objective_identity_reference_sha256", "")
    ).upper():
        raise SupplementError("objective identity reference hash changed")
    target_bindings = exact_identity.get("seeds")
    if not isinstance(target_bindings, Mapping) or set(target_bindings) != {
        str(seed) for seed in (101, 102, 103, 104, 105, 106)
    }:
        raise SupplementError("exact unique-target bindings lack six confirmation seeds")
    for seed in (101, 102, 103, 104, 105, 106):
        binding = target_bindings[str(seed)]
        if not isinstance(binding, Mapping):
            raise SupplementError(f"exact target binding is malformed for seed {seed}")
        relative = binding.get("relative_path")
        if not isinstance(relative, str):
            raise SupplementError(f"exact target path is missing for seed {seed}")
        target_path = _within(campaign_root, campaign_root / relative, "exact target")
        if (
            not target_path.is_file()
            or target_path.stat().st_size != binding.get("bytes")
            or sha256_file(target_path) != str(binding.get("sha256", "")).upper()
        ):
            raise SupplementError(f"exact target manifest changed for seed {seed}")
        target = _read(target_path)
        exact = target.get("targets", {}).get("exact_ground", {})
        if (
            target.get("ground_degeneracy") != 1
            or binding.get("ground_degeneracy") != 1
            or exact.get("count") != 1
            or exact.get("indices") != [binding.get("exact_ground_index")]
            or exact.get("bitstrings") != [binding.get("exact_ground_bitstring")]
            or exact.get("sha256") != binding.get("exact_ground_sha256")
            or target.get("minimum_feasible_energy")
            != binding.get("minimum_feasible_energy")
        ):
            raise SupplementError(f"exact target identity mismatch for seed {seed}")

    settings = manifest.get("settings")
    if not isinstance(settings, list) or len(settings) != 864:
        raise SupplementError("fixed-angle manifest must list exactly 864 settings")
    expected = {
        (30, seed, method, depth, angle_index)
        for seed in (101, 102, 103, 104, 105, 106)
        for method in GRADIENT_METHODS
        for depth in (26, 29, 32)
        for angle_index in range(12)
    }
    observed: set[tuple[int, int, str, int, int]] = set()
    audited: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = []
    for row in settings:
        if not isinstance(row, Mapping):
            raise SupplementError("fixed-angle setting is malformed")
        key = (
            int(row.get("N")),
            int(row.get("problem_seed")),
            str(row.get("method")),
            int(row.get("depth")),
            int(row.get("angle_index")),
        )
        if key not in expected or key in observed:
            raise SupplementError(f"duplicate/out-of-contract angle setting {key}")
        observed.add(key)
        config, algorithm, kappa = METHOD_CONFIG[key[2]]
        expected_gamma_count = 2 * key[3] if algorithm == "d-XY" else key[3]
        if (
            row.get("source_configuration") != config
            or row.get("gamma_parameter_count") != expected_gamma_count
            or row.get("beta_parameter_count") != key[3]
        ):
            raise SupplementError(f"method/depth angle shape mismatch at {key}")
        gamma, beta = _angle_arrays(row)
        source_relative = row.get("source_row_relative_path")
        if not isinstance(source_relative, str):
            raise SupplementError(f"source row path is missing at {key}")
        source_path = _within(
            campaign_root, campaign_root / source_relative, "historical source row"
        )
        if sha256_file(source_path) != str(row.get("source_row_sha256", "")).upper():
            raise SupplementError(f"historical source row hash changed at {key}")
        source = _read(source_path)
        if (
            source.get("schema")
            != "n30-degree6-paper-facing-trainability-row-v1"
            or source.get("phase") != "confirmation"
            or source.get("precision_audit") is not False
            or source.get("dtype") != "complex64"
            or source.get("N") != 30
            or source.get("seed") != key[1]
            or source.get("depth") != key[3]
            or source.get("configuration") != config
            or source.get("algorithm") != algorithm
            or float(source.get("penalty_kappa")) != kappa
            or source.get("point_kind") != "random"
            or source.get("point_index") != key[4]
            or str(source.get("gamma_sha256", "")).lower()
            != str(row.get("gamma_sha256", "")).lower()
            or str(source.get("beta_sha256", "")).lower()
            != str(row.get("beta_sha256", "")).lower()
            or str(source.get("support_sha256", "")).lower()
            != str(row.get("support_sha256", "")).lower()
            or str(source.get("objective_sha256", "")).lower()
            != str(row.get("objective_sha256", "")).lower()
            or str(source.get("stage1_checkpoint_sha256", "")).lower()
            != str(row.get("stage1_checkpoint_sha256", "")).lower()
            or str(source.get("contract_sha256", "")).lower()
            != str(row.get("historical_contract_sha256", "")).lower()
            or str(source.get("runner_sha256", "")).lower()
            != str(row.get("historical_runner_sha256", "")).lower()
            or source.get("validation_status") != "passed"
        ):
            raise SupplementError(f"historical row identity mismatch at {key}")
        audited.append((row, gamma, beta))
    if observed != expected:
        raise SupplementError("fixed-angle input coverage is incomplete")
    return manifest, audited


def _load_historical_modules(campaign_root: Path):
    historical = campaign_root / "gradient_inputs" / "historical_source"
    runner_path = historical / "trainability_runner.py"
    module_name = "_bcst_frozen_trainability_runner_20260729"
    specification = importlib.util.spec_from_file_location(module_name, runner_path)
    if specification is None or specification.loader is None:
        raise SupplementError("cannot load the frozen historical trainability runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    paper_root = historical / "paper_source"
    core, objective = module.load_modules(paper_root)
    expected_core = (paper_root / "source" / "campaign_core.py").resolve()
    expected_objective = (paper_root / "source" / "pspin_objective.py").resolve()
    if (
        Path(core.__file__).resolve() != expected_core
        or Path(objective.__file__).resolve() != expected_objective
    ):
        raise SupplementError("historical runtime imported a module outside the package")
    return module, core, objective, paper_root


def _build_gradient_context(
    *,
    module,
    core,
    objective,
    paper_root: Path,
    seed: int,
    device: str,
    target_binding: Mapping[str, Any],
) -> dict[str, Any]:
    streams = core.separated_streams(30, seed, "paper-facing-trainability-v1")
    _cfg, instance, dynamics = core.build_structure(
        30,
        device=device,
        dtype="complex64",
        optimizer_seed=streams.optimizer_seed,
        opt_restarts=1,
        opt_steps=1,
        continuation=False,
        sequential_restarts=True,
        trace_every=0,
        penalty=1.0,
    )
    bundle = objective.build_bundle(
        instance,
        instance_seed=seed,
        optimizer_domain="paper-facing-trainability-v1",
    )
    stage1_path = paper_root / "stage1_p12.json"
    p1, stage1_state, _stage1 = module.load_stage1_checkpoint_precision_compatible(
        core,
        dynamics,
        stage1_path,
        dtype="complex64",
    )
    if p1 != 12:
        raise SupplementError("historical gradient Stage-1 depth is not p1=12")
    objective_values = np.asarray(instance.Cobj_np)
    feasible = np.asarray(
        instance.feasible_mask.detach().cpu().numpy(), dtype=np.bool_
    )
    if objective_values.ndim != 1 or feasible.shape != objective_values.shape:
        raise SupplementError("historical objective/feasible shell has the wrong shape")
    feasible_indices = np.flatnonzero(feasible)
    feasible_values = objective_values[feasible_indices]
    ground_energy = np.min(feasible_values)
    ground_indices = feasible_indices[feasible_values == ground_energy]
    if len(ground_indices) != 1:
        raise SupplementError(
            f"historical seed {seed} unique-ground degeneracy is {len(ground_indices)}"
        )
    ground_index = int(ground_indices[0])
    if (
        target_binding.get("ground_degeneracy") != 1
        or ground_index != target_binding.get("exact_ground_index")
        or not math.isclose(
            float(ground_energy),
            float(target_binding.get("minimum_feasible_energy")),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or str(bundle.manifest["support_sha256"]).lower()
        != str(target_binding.get("support_sha256", "")).lower()
        or str(bundle.manifest["objective_sha256"]).lower()
        != str(target_binding.get("objective_sha256", "")).lower()
        or str(bundle.manifest["target_manifest_sha256"]).lower()
        != str(target_binding.get("canonical_target_manifest_sha256", "")).lower()
    ):
        raise SupplementError(f"reconstructed exact target differs at seed {seed}")
    return {
        "seed": seed,
        "instance": instance,
        "dynamics": dynamics,
        "bundle": bundle,
        "stage1_state": stage1_state,
        "stage1_path": stage1_path,
        "ground_index": ground_index,
        "ground_energy": float(ground_energy),
        "ground_state_sha256": str(target_binding["exact_ground_sha256"]),
        "ground_bitstring": str(target_binding["exact_ground_bitstring"]),
        "feasible_mask": instance.feasible_mask,
        "support_sha256": str(bundle.manifest["support_sha256"]),
        "objective_sha256": str(bundle.manifest["objective_sha256"]),
        "historical_target_manifest_sha256": str(
            bundle.manifest["target_manifest_sha256"]
        ),
        "stage1_checkpoint_sha256": sha256_file(stage1_path).lower(),
    }


def _flatten(module, gradients, parameters) -> torch.Tensor:
    return module.flatten_gradients(gradients, parameters)


def _evaluate_unique_setting(
    *,
    module,
    context: Mapping[str, Any],
    input_row: Mapping[str, Any],
    gamma_values: np.ndarray,
    beta_values: np.ndarray,
    source_row: Mapping[str, Any],
    plan_sha256: str,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    method = str(input_row["method"])
    _config, algorithm, penalty_kappa = METHOD_CONFIG[method]
    depth = int(input_row["depth"])
    dynamics = context["dynamics"]
    instance = context["instance"]
    bundle = context["bundle"]
    stage1_state = context["stage1_state"]
    weight_rms = float(bundle.manifest["weight_empirical"]["rms"])
    penalty = penalty_kappa * weight_rms
    gamma = torch.tensor(
        gamma_values,
        dtype=dynamics.real_dtype,
        device=dynamics.device,
        requires_grad=True,
    )
    beta = torch.tensor(
        beta_values,
        dtype=dynamics.real_dtype,
        device=dynamics.device,
        requires_grad=True,
    )
    g = gamma[None, :]
    b = beta[None, :]
    native_diagonal = penalty * instance.Cconf + instance.Cobj
    if algorithm == "XY-LP-QAOA":
        _energy, states = dynamics.energy_history(
            stage1_state, instance.Cobj, g, b, depth
        )
    elif algorithm == "Std-XY":
        _energy, states = dynamics.energy_xy(native_diagonal, g, b, depth)
    elif algorithm == "d-XY":
        _energy, states = dynamics.energy_xy(
            native_diagonal,
            g,
            b,
            depth,
            split=(penalty * instance.Cconf, instance.Cobj),
        )
    elif algorithm == "Warm-XY":
        _energy, states = dynamics.energy_xy(
            native_diagonal,
            g,
            b,
            depth,
            initial_state=stage1_state,
        )
    else:
        raise SupplementError(f"unknown historical gradient algorithm {algorithm}")
    state = states[0]
    parameters = (gamma, beta)
    objective_energy = module.expectation(state, instance.Cobj)
    objective_gradient = _flatten(
        module,
        torch.autograd.grad(
            objective_energy,
            parameters,
            retain_graph=True,
            allow_unused=True,
        ),
        parameters,
    )
    if algorithm == "XY-LP-QAOA":
        training_energy = objective_energy
        training_gradient = objective_gradient
    else:
        conflict_energy = module.expectation(state, instance.Cconf)
        conflict_gradient = _flatten(
            module,
            torch.autograd.grad(
                conflict_energy,
                parameters,
                retain_graph=True,
                allow_unused=True,
            ),
            parameters,
        )
        training_energy = objective_energy + penalty * conflict_energy
        training_gradient = objective_gradient + penalty * conflict_gradient

    # Accumulate probabilities in float64 even though the historical method
    # state is complex64.  This implements log(max(P,1e-300)) without changing
    # the state or angles used by the original panel.
    probabilities = state.real.to(torch.float64).square() + state.imag.to(
        torch.float64
    ).square()
    norm = torch.sum(probabilities)
    unique_probability = probabilities[int(context["ground_index"])] / norm
    feasible_probability = torch.sum(probabilities[context["feasible_mask"]]) / norm
    unclipped = float(unique_probability.detach().cpu())
    clipped = unclipped < 1e-300
    log_unique = torch.log(torch.clamp(unique_probability, min=1e-300))
    unique_gradient = _flatten(
        module,
        torch.autograd.grad(
            log_unique,
            parameters,
            retain_graph=False,
            allow_unused=True,
        ),
        parameters,
    )
    parameter_count = int(unique_gradient.numel())
    unique_l2 = float(torch.linalg.vector_norm(unique_gradient).detach().cpu())
    unique_rms = unique_l2 / math.sqrt(parameter_count)
    training_l2 = float(torch.linalg.vector_norm(training_gradient).detach().cpu())
    training_rms = training_l2 / math.sqrt(parameter_count)
    objective_l2 = float(torch.linalg.vector_norm(objective_gradient).detach().cpu())
    objective_rms = objective_l2 / math.sqrt(parameter_count)
    alignment_value = module.cosine(-training_gradient, unique_gradient)
    alignment_defined = alignment_value is not None
    alignment = 0.0 if alignment_value is None else float(alignment_value)
    native_values = (
        instance.Cobj_np
        if algorithm == "XY-LP-QAOA"
        else instance.Cobj_np + penalty * instance.Cconf_np
    )
    native_scale = float(np.std(np.asarray(native_values, dtype=np.float64)))
    objective_scale = float(
        np.std(np.asarray(instance.Cobj_np, dtype=np.float64))
    )
    source_training_rms = float(source_row["training_loss_gradient_rms"])
    if not math.isclose(
        training_rms, source_training_rms, rel_tol=3e-5, abs_tol=3e-6
    ):
        raise SupplementError(
            "reconstructed fixed-angle method state does not reproduce the "
            f"historical training gradient at seed={context['seed']}, method={method}, "
            f"p={depth}, angle={input_row['angle_index']}"
        )
    state_bytes = np.ascontiguousarray(
        state.detach().cpu().numpy(), dtype="<c8"
    ).tobytes(order="C")
    numeric = (
        unclipped,
        float(feasible_probability.detach().cpu()),
        float(training_energy.detach().cpu()),
        training_rms,
        objective_rms,
        unique_rms,
        alignment,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise SupplementError("unique fixed-angle replay produced a nonfinite metric")
    row = {
        "schema": "bcst-unique-fixed-angle-gradient-row-v1",
        "plan_sha256": plan_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "N": 30,
        "problem_seed": int(context["seed"]),
        "method": method,
        "depth": depth,
        "angle_index": int(input_row["angle_index"]),
        "angle_vector_sha256": str(input_row["angle_vector_sha256"]),
        "gamma_sha256": str(input_row["gamma_sha256"]),
        "beta_sha256": str(input_row["beta_sha256"]),
        "source_row_sha256": str(input_row["source_row_sha256"]),
        "source_configuration": str(input_row["source_configuration"]),
        "support_sha256": str(context["support_sha256"]),
        "objective_sha256": str(context["objective_sha256"]),
        "stage1_checkpoint_sha256": str(context["stage1_checkpoint_sha256"]),
        "historical_contract_sha256": str(input_row["historical_contract_sha256"]),
        "historical_runner_sha256": str(input_row["historical_runner_sha256"]),
        "historical_target_manifest_sha256": str(
            input_row["historical_target_manifest_sha256"]
        ),
        "ground_state_sha256": str(context["ground_state_sha256"]),
        "ground_index_decimal": str(context["ground_index"]),
        "ground_bitstring": str(context["ground_bitstring"]),
        "ground_energy_float64_hex": float(context["ground_energy"]).hex(),
        "ground_degeneracy": 1,
        "historical_state_amplitude_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "parameter_count": parameter_count,
        "training_loss": float(training_energy.detach().cpu()),
        "training_loss_gradient_rms": training_rms,
        "training_loss_gradient_rms_normalized": training_rms
        / max(native_scale, 1e-300),
        "common_objective_gradient_rms": objective_rms,
        "common_objective_gradient_rms_normalized": objective_rms
        / max(objective_scale, 1e-300),
        "unique_probability_unclipped": unclipped,
        "feasible_probability": float(feasible_probability.detach().cpu()),
        "log_probability_floor": 1e-300,
        "log_probability_clipped": clipped,
        "log_unique_probability_gradient_l2": unique_l2,
        "log_unique_probability_gradient_rms": unique_rms,
        "training_descent_alignment_unique": alignment,
        "alignment_defined": alignment_defined,
        "alignment_interpretation": (
            "exact_cosine"
            if alignment_defined and not clipped
            else "descriptive_floor_or_zero_norm"
        ),
        "optimization_runs": 0,
        "resampled": False,
        "selected_by_target": False,
        "historical_training_gradient_reproduced": True,
        "status": "complete",
    }
    row["row_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(row)
    ).hexdigest().upper()
    return row


def _validate_gradient_row(
    record: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    contract: CampaignContract,
    input_manifest_sha256: str,
    target_binding: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> None:
    if set(record) != GRADIENT_ROW_FIELDS:
        raise SupplementError("existing unique-gradient row has a changed field schema")
    config, algorithm, _kappa = METHOD_CONFIG[str(input_row.get("method"))]
    expected_parameter_count = int(input_row.get("gamma_parameter_count")) + int(
        input_row.get("beta_parameter_count")
    )
    identity = (
        record.get("schema") == "bcst-unique-fixed-angle-gradient-row-v1"
        and record.get("plan_sha256") == contract.sha256
        and record.get("input_manifest_sha256") == input_manifest_sha256
        and record.get("N") == 30
        and record.get("problem_seed") == input_row.get("problem_seed")
        and record.get("method") == input_row.get("method")
        and record.get("depth") == input_row.get("depth")
        and record.get("angle_index") == input_row.get("angle_index")
        and record.get("angle_vector_sha256")
        == input_row.get("angle_vector_sha256")
        and record.get("gamma_sha256") == input_row.get("gamma_sha256")
        and record.get("beta_sha256") == input_row.get("beta_sha256")
        and record.get("source_row_sha256") == input_row.get("source_row_sha256")
        and record.get("source_configuration") == config
        and record.get("support_sha256") == input_row.get("support_sha256")
        and record.get("support_sha256") == target_binding.get("support_sha256")
        and record.get("objective_sha256") == input_row.get("objective_sha256")
        and record.get("objective_sha256") == target_binding.get("objective_sha256")
        and record.get("stage1_checkpoint_sha256")
        == input_row.get("stage1_checkpoint_sha256")
        and record.get("historical_contract_sha256")
        == input_row.get("historical_contract_sha256")
        and record.get("historical_runner_sha256")
        == input_row.get("historical_runner_sha256")
        and record.get("historical_target_manifest_sha256")
        == input_row.get("historical_target_manifest_sha256")
        and record.get("historical_target_manifest_sha256")
        == target_binding.get("canonical_target_manifest_sha256")
        and record.get("ground_state_sha256")
        == target_binding.get("exact_ground_sha256")
        and record.get("ground_index_decimal")
        == str(target_binding.get("exact_ground_index"))
        and record.get("ground_bitstring")
        == target_binding.get("exact_ground_bitstring")
        and record.get("ground_energy_float64_hex")
        == float(target_binding.get("minimum_feasible_energy")).hex()
        and record.get("ground_degeneracy") == 1
        and record.get("parameter_count") == expected_parameter_count
        and record.get("optimization_runs") == 0
        and record.get("resampled") is False
        and record.get("selected_by_target") is False
        and record.get("historical_training_gradient_reproduced") is True
        and record.get("status") == "complete"
    )
    hashes = (
        record.get("angle_vector_sha256"),
        record.get("gamma_sha256"),
        record.get("beta_sha256"),
        record.get("source_row_sha256"),
        record.get("support_sha256"),
        record.get("objective_sha256"),
        record.get("stage1_checkpoint_sha256"),
        record.get("historical_contract_sha256"),
        record.get("historical_runner_sha256"),
        record.get("historical_target_manifest_sha256"),
        record.get("ground_state_sha256"),
        record.get("historical_state_amplitude_sha256"),
        record.get("row_content_sha256"),
    )
    if not identity or not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
        for value in hashes
    ):
        raise SupplementError("existing unique-gradient row identity is incompatible")
    numeric_fields = (
        "training_loss",
        "training_loss_gradient_rms",
        "training_loss_gradient_rms_normalized",
        "common_objective_gradient_rms",
        "common_objective_gradient_rms_normalized",
        "unique_probability_unclipped",
        "feasible_probability",
        "log_probability_floor",
        "log_unique_probability_gradient_l2",
        "log_unique_probability_gradient_rms",
        "training_descent_alignment_unique",
    )
    if not all(
        not isinstance(record.get(key), bool)
        and isinstance(record.get(key), (int, float))
        and math.isfinite(float(record[key]))
        for key in numeric_fields
    ):
        raise SupplementError("existing unique-gradient row has nonfinite metrics")
    probability = float(record["unique_probability_unclipped"])
    feasible = float(record["feasible_probability"])
    unique_l2 = float(record["log_unique_probability_gradient_l2"])
    unique_rms = float(record["log_unique_probability_gradient_rms"])
    alignment = float(record["training_descent_alignment_unique"])
    if (
        not 0.0 <= probability <= feasible <= 1.0
        or unique_l2 < 0.0
        or unique_rms < 0.0
        or not math.isclose(
            unique_rms,
            unique_l2 / math.sqrt(expected_parameter_count),
            rel_tol=1e-14,
            abs_tol=1e-15,
        )
        or not -1.0 - 1e-7 <= alignment <= 1.0 + 1e-7
        or record.get("log_probability_floor") != 1e-300
        or record.get("log_probability_clipped") is not (probability < 1e-300)
        or not isinstance(record.get("alignment_defined"), bool)
        or (
            record.get("alignment_defined") is False
            and not math.isclose(alignment, 0.0, rel_tol=0.0, abs_tol=0.0)
        )
        or record.get("alignment_interpretation")
        != (
            "exact_cosine"
            if record.get("alignment_defined") is True
            and record.get("log_probability_clipped") is False
            else "descriptive_floor_or_zero_norm"
        )
    ):
        raise SupplementError("existing unique-gradient row metric invariants changed")
    source_checks = (
        ("training_loss", "training_loss"),
        ("training_loss_gradient_rms", "training_loss_gradient_rms"),
        (
            "training_loss_gradient_rms_normalized",
            "training_loss_gradient_rms_normalized",
        ),
        ("common_objective_gradient_rms", "objective_gradient_rms"),
        (
            "common_objective_gradient_rms_normalized",
            "objective_gradient_rms_normalized",
        ),
        ("feasible_probability", "feasible_probability"),
    )
    for record_key, source_key in source_checks:
        source_value = source_row.get(source_key)
        if (
            isinstance(source_value, bool)
            or not isinstance(source_value, (int, float))
            or not math.isclose(
                float(record[record_key]),
                float(source_value),
                rel_tol=3e-5,
                abs_tol=3e-6,
            )
        ):
            raise SupplementError(
                f"existing unique-gradient row does not reproduce source {source_key}"
            )
    if (
        source_row.get("algorithm") != algorithm
        or source_row.get("parameter_count") != expected_parameter_count
        or source_row.get("support_sha256") != record.get("support_sha256")
        or source_row.get("objective_sha256") != record.get("objective_sha256")
        or source_row.get("stage1_checkpoint_sha256")
        != record.get("stage1_checkpoint_sha256")
    ):
        raise SupplementError("existing unique-gradient source identity changed")
    content = dict(record)
    claimed_content_hash = str(content.pop("row_content_sha256"))
    if hashlib.sha256(canonical_json_bytes(content)).hexdigest().upper() != claimed_content_hash:
        raise SupplementError("existing unique-gradient row content hash changed")


def _gradient_aggregation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    medians: dict[tuple[int, str, int], tuple[float, float, float]] = {}
    for seed in (101, 102, 103, 104, 105, 106):
        for method in GRADIENT_METHODS:
            for depth in (26, 29, 32):
                selected = [
                    row
                    for row in rows
                    if row["problem_seed"] == seed
                    and row["method"] == method
                    and row["depth"] == depth
                ]
                if len(selected) != 12:
                    raise SupplementError("gradient aggregation cell is incomplete")
                probability = float(
                    np.median([row["unique_probability_unclipped"] for row in selected])
                )
                gradient = float(
                    np.median(
                        [row["log_unique_probability_gradient_rms"] for row in selected]
                    )
                )
                alignment = float(
                    np.median(
                        [row["training_descent_alignment_unique"] for row in selected]
                    )
                )
                medians[(seed, method, depth)] = (probability, gradient, alignment)
                cells.append(
                    {
                        "problem_seed": seed,
                        "method": method,
                        "depth": depth,
                        "angle_count": 12,
                        "median_unique_probability": probability,
                        "median_log_unique_gradient_rms": gradient,
                        "median_training_descent_alignment_unique": alignment,
                        "clip_count": sum(
                            bool(row["log_probability_clipped"]) for row in selected
                        ),
                    }
                )
    comparisons: list[dict[str, Any]] = []
    for depth in (26, 29, 32):
        for baseline in GRADIENT_METHODS[1:]:
            ratios = []
            differences = []
            paired = []
            for seed in (101, 102, 103, 104, 105, 106):
                _lp_probability, lp_gradient, lp_alignment = medians[
                    (seed, "learned_projector", depth)
                ]
                _base_probability, base_gradient, base_alignment = medians[
                    (seed, baseline, depth)
                ]
                ratio = None if lp_gradient == 0.0 else base_gradient / lp_gradient
                difference = base_alignment - lp_alignment
                if ratio is not None and ratio > 0.0 and math.isfinite(ratio):
                    ratios.append(ratio)
                differences.append(difference)
                paired.append(
                    {
                        "problem_seed": seed,
                        "baseline_over_LP_gradient_ratio": ratio,
                        "baseline_minus_LP_alignment": difference,
                    }
                )
            gm_ratio = (
                None
                if len(ratios) != 6
                else math.exp(math.fsum(math.log(value) for value in ratios) / 6)
            )
            comparisons.append(
                {
                    "depth": depth,
                    "baseline": baseline,
                    "paired_seed_count": 6,
                    "per_seed": paired,
                    "geometric_mean_baseline_over_LP_gradient_ratio": gm_ratio,
                    "mean_baseline_minus_LP_alignment": math.fsum(differences) / 6,
                }
            )
    return {
        "aggregation_order": "median_over_12_angles_then_paired_over_6_instances",
        "instance_method_depth_medians": cells,
        "paired_baseline_vs_LP": comparisons,
    }


def recompute_unique_gradient_panel(
    contract: CampaignContract,
    input_manifest_path: Path,
    result_path: Path,
    *,
    device: str = "cuda",
) -> Path:
    """Resumably replay all 864 exact vectors and compute unique-target gradients."""

    manifest, audited = audit_gradient_inputs(contract, input_manifest_path)
    input_sha = sha256_file(input_manifest_path)
    destination = Path(result_path).resolve()
    if destination.exists():
        validate_gradient_panel(contract, input_manifest_path, destination)
        return destination
    if device != "cuda" or not torch.cuda.is_available():
        raise SupplementError("the fixed-angle panel requires the remote CUDA A800")
    device_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if "A800" not in device_name:
        raise SupplementError(f"fixed-angle replay requires A800, observed {device_name}")
    torch.use_deterministic_algorithms(True)
    module, core, objective, paper_root = _load_historical_modules(
        contract.path.parent
    )
    row_root = destination.parent / "rows"
    by_seed: dict[int, list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]]] = {
        seed: [] for seed in (101, 102, 103, 104, 105, 106)
    }
    for item in audited:
        by_seed[int(item[0]["problem_seed"])].append(item)
    final_rows: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    for seed in (101, 102, 103, 104, 105, 106):
        seed_items = sorted(
            by_seed[seed],
            key=lambda item: (
                GRADIENT_METHODS.index(str(item[0]["method"])),
                int(item[0]["depth"]),
                int(item[0]["angle_index"]),
            ),
        )
        paths = [
            row_root
            / f"seed{seed}"
            / str(item[0]["method"])
            / f"p{item[0]['depth']}"
            / f"angle_{int(item[0]['angle_index']):02d}.json"
            for item in seed_items
        ]
        missing = [path for path in paths if not path.exists()]
        context = None
        if missing:
            context = _build_gradient_context(
                module=module,
                core=core,
                objective=objective,
                paper_root=paper_root,
                seed=seed,
                device=device,
                target_binding=manifest["exact_target_identity"]["seeds"][str(seed)],
            )
            first_input = seed_items[0][0]
            if (
                str(context["support_sha256"]).lower()
                != str(first_input["support_sha256"]).lower()
                or str(context["objective_sha256"]).lower()
                != str(first_input["objective_sha256"]).lower()
                or str(context["historical_target_manifest_sha256"]).lower()
                != str(first_input["historical_target_manifest_sha256"]).lower()
                or str(context["stage1_checkpoint_sha256"]).lower()
                != str(first_input["stage1_checkpoint_sha256"]).lower()
            ):
                raise SupplementError(f"historical instance reconstruction mismatch at seed {seed}")
        for (input_row, gamma, beta), path in zip(seed_items, paths):
            if path.exists():
                row = _read(path)
                source_path = _within(
                    contract.path.parent,
                    contract.path.parent / str(input_row["source_row_relative_path"]),
                    "historical source row",
                )
                _validate_gradient_row(
                    row,
                    input_row,
                    contract=contract,
                    input_manifest_sha256=input_sha,
                    target_binding=manifest["exact_target_identity"]["seeds"][
                        str(seed)
                    ],
                    source_row=_read(source_path),
                )
            else:
                assert context is not None
                source_path = _within(
                    contract.path.parent,
                    contract.path.parent / str(input_row["source_row_relative_path"]),
                    "historical source row",
                )
                row = _evaluate_unique_setting(
                    module=module,
                    context=context,
                    input_row=input_row,
                    gamma_values=gamma,
                    beta_values=beta,
                    source_row=_read(source_path),
                    plan_sha256=contract.sha256,
                    input_manifest_sha256=input_sha,
                )
                publish_json(path, row)
            final_rows.append(row)
        if context is None:
            # Recover immutable ground identity from one completed row when an
            # entire seed was already complete before this resume.
            sample = final_rows[-len(seed_items)]
            instances.append(
                {
                    "problem_seed": seed,
                    "support_sha256": sample["support_sha256"],
                    "objective_sha256": sample["objective_sha256"],
                    "ground_index_decimal": sample["ground_index_decimal"],
                    "ground_bitstring": sample["ground_bitstring"],
                    "ground_energy_float64_hex": sample["ground_energy_float64_hex"],
                    "ground_state_sha256": sample["ground_state_sha256"],
                    "ground_degeneracy": 1,
                }
            )
        else:
            instances.append(
                {
                    "problem_seed": seed,
                    "support_sha256": context["support_sha256"],
                    "objective_sha256": context["objective_sha256"],
                    "ground_index_decimal": str(context["ground_index"]),
                    "ground_bitstring": context["ground_bitstring"],
                    "ground_energy_float64_hex": float(context["ground_energy"]).hex(),
                    "ground_state_sha256": context["ground_state_sha256"],
                    "ground_degeneracy": 1,
                }
            )
            del context
            torch.cuda.empty_cache()
    if len(final_rows) != 864:
        raise SupplementError("fixed-angle replay did not produce 864 rows")
    final_rows.sort(
        key=lambda row: (
            int(row["problem_seed"]),
            GRADIENT_METHODS.index(str(row["method"])),
            int(row["depth"]),
            int(row["angle_index"]),
        )
    )
    result = {
        "schema": "bcst-unique-fixed-angle-gradient-panel-v1",
        "campaign_id": contract.campaign_id,
        "plan_sha256": contract.sha256,
        "input_manifest_path": str(input_manifest_path.resolve()),
        "input_manifest_sha256": input_sha,
        "historical_input_manifest_pinned_sha256": EXPECTED_ANGLE_MANIFEST_SHA256,
        "device": device,
        "device_name": device_name,
        "dtype": "complex64_historical_state_with_float64_probability_accumulation",
        "setting_count": 864,
        "settings": final_rows,
        "instances": instances,
        "aggregation": _gradient_aggregation(final_rows),
        "optimization_runs": 0,
        "resampled": False,
        "target_selected_angles": False,
        "status": "complete",
    }
    publish_json(destination, result)
    return destination


def validate_gradient_panel(
    contract: CampaignContract,
    input_manifest_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Admit only a 1:1 recomputation of the exact 864 frozen angle settings."""

    if not input_manifest_path.is_file() or not result_path.is_file():
        raise SupplementError(
            "unique-gradient panel requires FROZEN_ANGLE_INPUT_MANIFEST.json and "
            "UNIQUE_FIXED_ANGLE_PANEL.json; the 864 vectors cannot be reconstructed "
            "from the local median-only source table and must not be resampled"
        )
    inputs, audited_inputs = audit_gradient_inputs(contract, input_manifest_path)
    result = _read(result_path)
    input_hash = sha256_file(input_manifest_path)
    if (
        input_hash != EXPECTED_ANGLE_MANIFEST_SHA256
        or inputs.get("schema") != "bcst-fixed-angle-panel-input-manifest-v1"
        or inputs.get("status") != "complete"
        or inputs.get("setting_count") != 864
        or inputs.get("resampled") is not False
        or result.get("schema") != "bcst-unique-fixed-angle-gradient-panel-v1"
        or result.get("campaign_id") != contract.campaign_id
        or result.get("plan_sha256") != contract.sha256
        or result.get("input_manifest_sha256") != input_hash
        or result.get("historical_input_manifest_pinned_sha256")
        != EXPECTED_ANGLE_MANIFEST_SHA256
        or result.get("input_manifest_path") != str(input_manifest_path.resolve())
        or result.get("device") != "cuda"
        or "A800" not in str(result.get("device_name", ""))
        or result.get("dtype")
        != "complex64_historical_state_with_float64_probability_accumulation"
        or result.get("setting_count") != 864
        or result.get("optimization_runs") != 0
        or result.get("resampled") is not False
        or result.get("target_selected_angles") is not False
        or result.get("status") != "complete"
    ):
        raise SupplementError("fixed-angle panel identity is incompatible")
    input_rows = [item[0] for item in audited_inputs]
    rows = result.get("settings")
    if not isinstance(input_rows, list) or not isinstance(rows, list):
        raise SupplementError("fixed-angle settings must be lists")
    if len(input_rows) != 864 or len(rows) != 864:
        raise SupplementError("fixed-angle panel must contain exactly 864 settings")
    expected = {
        (
            30,
            seed,
            method,
            depth,
            angle_index,
        )
        for seed in (101, 102, 103, 104, 105, 106)
        for method in (
            "learned_projector",
            "ordinary_native_direct",
            "decoupled_native_direct",
            "same_stage1_state_direct",
        )
        for depth in (26, 29, 32)
        for angle_index in range(12)
    }
    input_by_key: dict[tuple[int, int, str, int, int], Mapping[str, Any]] = {}
    target_bindings = inputs.get("exact_target_identity", {}).get("seeds", {})
    if not isinstance(target_bindings, Mapping):
        raise SupplementError("fixed-angle inputs lack exact-target bindings")
    for row in input_rows:
        if not isinstance(row, Mapping):
            raise SupplementError("frozen-angle input row is malformed")
        key = (
            int(row.get("N")),
            int(row.get("problem_seed")),
            str(row.get("method")),
            int(row.get("depth")),
            int(row.get("angle_index")),
        )
        if key in input_by_key:
            raise SupplementError(f"duplicate frozen-angle input {key}")
        input_by_key[key] = row
    if set(input_by_key) != expected:
        raise SupplementError("frozen-angle input coverage differs from the contract")
    seen: set[tuple[int, int, str, int, int]] = set()
    clipped = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise SupplementError("unique-gradient result row is malformed")
        key = (
            int(row.get("N")),
            int(row.get("problem_seed")),
            str(row.get("method")),
            int(row.get("depth")),
            int(row.get("angle_index")),
        )
        if key in seen or key not in expected:
            raise SupplementError(f"duplicate/out-of-contract unique-gradient row {key}")
        seen.add(key)
        source = input_by_key[key]
        target_binding = target_bindings.get(str(key[1]))
        if not isinstance(target_binding, Mapping):
            raise SupplementError(f"exact target binding is missing at {key}")
        source_path = _within(
            contract.path.parent,
            contract.path.parent / str(source["source_row_relative_path"]),
            "historical source row",
        )
        _validate_gradient_row(
            row,
            source,
            contract=contract,
            input_manifest_sha256=input_hash,
            target_binding=target_binding,
            source_row=_read(source_path),
        )
        clipped += int(row.get("log_probability_clipped") is True)
    if seen != expected:
        raise SupplementError("unique-gradient output is incomplete")
    expected_instances = [
        {
            "problem_seed": seed,
            "support_sha256": target_bindings[str(seed)]["support_sha256"],
            "objective_sha256": target_bindings[str(seed)]["objective_sha256"],
            "ground_index_decimal": str(
                target_bindings[str(seed)]["exact_ground_index"]
            ),
            "ground_bitstring": target_bindings[str(seed)]["exact_ground_bitstring"],
            "ground_energy_float64_hex": float(
                target_bindings[str(seed)]["minimum_feasible_energy"]
            ).hex(),
            "ground_state_sha256": target_bindings[str(seed)]["exact_ground_sha256"],
            "ground_degeneracy": 1,
        }
        for seed in (101, 102, 103, 104, 105, 106)
    ]
    if result.get("instances") != expected_instances:
        raise SupplementError("unique-gradient instance identity ledger changed")
    expected_aggregation = _gradient_aggregation(list(rows))
    if result.get("aggregation") != expected_aggregation:
        raise SupplementError("unique-gradient aggregation does not reproduce its rows")
    return {
        "schema": "bcst-unique-fixed-angle-gradient-admission-v1",
        "campaign_id": contract.campaign_id,
        "plan_sha256": contract.sha256,
        "input_manifest_path": str(input_manifest_path.resolve()),
        "input_manifest_sha256": input_hash,
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "setting_count": 864,
        "clip_count": clipped,
        "optimization_runs": 0,
        "status": "passed",
    }


def historical_reanalysis_inventory(campaign_root: Path) -> dict[str, Any]:
    """Strictly admit only the packaged and hash-bound existing-evidence set."""

    root = campaign_root.resolve()
    manifest_path = root / "existing_evidence" / "EXISTING_EVIDENCE_MANIFEST.json"
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != EXPECTED_EXISTING_EVIDENCE_MANIFEST_SHA256:
        raise SupplementError("existing-evidence manifest differs from the frozen package")
    manifest = _read(manifest_path)
    expected_roles = {
        "matched_first_five",
        "matched_second_five",
        "lp_p80_second_five_sensitivity",
        "n30_lp_capped_depth_grid",
        "adam400_fixed_budget",
        "causal_common_depth_ablation",
        "legacy_spsa_exact_ground",
        "historical_gradient_summary",
    }
    groups = manifest.get("groups")
    if (
        manifest.get("schema") != "bcst-unique-existing-evidence-manifest-v1"
        or manifest.get("status") != "complete"
        or manifest.get("group_count") != 8
        or manifest.get("fresh_prospective_substitution_permitted") is not False
        or set(manifest.get("coverage_roles", [])) != expected_roles
        or not isinstance(groups, list)
        or len(groups) != 8
    ):
        raise SupplementError("existing-evidence manifest header is incompatible")
    admitted_groups: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[Path] = set()
    file_count = 0
    for group in groups:
        if not isinstance(group, Mapping):
            raise SupplementError("existing-evidence group is malformed")
        role = str(group.get("role"))
        files = group.get("files")
        if (
            role not in expected_roles
            or role in seen_roles
            or group.get("status") != "verified_and_packaged"
            or group.get("fresh_prospective_substitution_permitted") is not False
            or not isinstance(files, list)
            or not files
        ):
            raise SupplementError(f"existing-evidence group is incompatible: {role}")
        seen_roles.add(role)
        admitted_files = []
        for binding in files:
            if not isinstance(binding, Mapping):
                raise SupplementError(f"existing-evidence file is malformed: {role}")
            relative = binding.get("packaged_relative_path")
            if not isinstance(relative, str):
                raise SupplementError(f"existing-evidence path is missing: {role}")
            path = _within(root, root / relative, "existing evidence")
            if path in seen_paths:
                raise SupplementError(f"existing-evidence path is duplicated: {relative}")
            seen_paths.add(path)
            if (
                not path.is_file()
                or path.stat().st_size != binding.get("bytes")
                or sha256_file(path) != str(binding.get("sha256", "")).upper()
            ):
                raise SupplementError(f"existing-evidence bytes changed: {relative}")
            admitted_files.append(
                {
                    "path": str(path),
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            file_count += 1
        admitted_groups.append(
            {
                "role": role,
                "interpretation": group.get("interpretation"),
                "validated_coverage": group.get("validated_coverage"),
                "files": admitted_files,
                "status": "admitted_fixed_historical_evidence",
            }
        )
    if seen_roles != expected_roles:
        raise SupplementError("existing-evidence role coverage is incomplete")
    return {
        "schema": "bcst-unique-existing-data-reanalysis-inventory-v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": observed_manifest_sha,
        "group_count": len(admitted_groups),
        "file_count": file_count,
        "fresh_campaign_substitution_permitted": False,
        "groups": admitted_groups,
        "status": "inventory_complete",
    }


__all__ = [
    "SupplementError",
    "audit_gradient_inputs",
    "historical_reanalysis_inventory",
    "recompute_unique_gradient_panel",
    "validate_gradient_panel",
]
