#!/usr/bin/env python3
"""Post-seal exact ground-space scorer for the 18-qubit vanilla diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import targeted_coherent_im3_score as score_common
import targeted_coherent_im3_vanilla_qaoa as campaign
import targeted_three_stage_common as common


SCHEMA = f"{campaign.SCHEMA}-post-global-seal-score-v2"
OUTPUT_NAME = "targeted_coherent_im3_vanilla_qaoa_post_global_seal_score.json"
CELL_CSV_NAME = "targeted_coherent_im3_vanilla_qaoa_curve_cells.csv"
SUMMARY_CSV_NAME = "targeted_coherent_im3_vanilla_qaoa_curve_summary.csv"
CELL_FIELDS = (
    "depth", "table_seed", "table_id", "restart_index", "restart_seed",
    "p0_derived_anchor", "ground_energy", "ground_degeneracy",
    "ground_space_success_probability", "feasible_mass",
    "conditional_ground_probability_given_feasible", "expected_raw_objective",
    "expected_integer_penalty", "expected_H_over_377", "RTS99", "RU",
    "RTS99_x_RU", "amplification_over_full_uniform_ground_probability",
    "feasible_mass_amplification_over_full_uniform",
    "conditional_ground_amplification_over_uniform_feasible",
    "optimizer_result_sha256",
)
SUMMARY_FIELDS = (
    "cohort", "claim_role", "depth", "independent_table_count",
    "nested_optimizer_attempts_per_table", "nested_optimizer_attempt_count",
    "success_probability_arithmetic_mean", "success_probability_geometric_mean",
    "table_equal_success_probability_geometric_mean", "success_probability_median",
    "success_probability_min", "success_probability_max",
    "feasible_mass_arithmetic_mean", "feasible_mass_geometric_mean",
    "conditional_ground_probability_geometric_mean",
    "ground_amplification_geometric_mean", "feasible_amplification_geometric_mean",
    "conditional_amplification_geometric_mean", "RU", "RTS99_x_RU_geometric_mean",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_bytes(payload)
    if path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _gm(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)) or np.any(array < 0):
        return None
    if np.any(array == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _verified_seal(
    root: Path, manifest: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    seal = common.load_json(root / campaign.GLOBAL_SEAL_NAME)
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    common._require(
        seal.get("schema") == f"{campaign.SCHEMA}-global-seal"
        and seal.get("status")
        == "SEALED_COMPLETE_TARGET_BLIND_VANILLA_QAOA_DIAGNOSTIC"
        and seal.get("manifest_sha256") == manifest["manifest_sha256"]
        and seal.get("plan_sha256") == plan["plan_sha256"]
        and seal.get("complete_optimizer_result_count")
        == len(campaign.TABLE_SEEDS) * len(campaign.POSITIVE_DEPTHS)
        and seal.get("seal_sha256") == common.sha256_json(body)
        and seal.get("target_constructed") is False,
        "complete target-blind vanilla seal required",
    )
    expected = {
        (seed, depth)
        for seed in campaign.TABLE_SEEDS for depth in campaign.POSITIVE_DEPTHS
    }
    sealed_rows = seal.get("completed_cells", ())
    sealed = {
        (int(row["table_seed"]), int(row["depth"])): str(row["result_sha256"])
        for row in sealed_rows
    }
    expected_paths = {
        campaign.result_path(root, seed, depth).resolve()
        for seed, depth in expected
    }
    actual_paths = {
        path.resolve() for path in (root / campaign.RESULT_DIRECTORY).glob("*/*.json")
    }
    common._require(
        len(sealed_rows) == len(expected)
        and set(sealed) == expected
        and actual_paths == expected_paths,
        "vanilla seal key/path set is not exact",
    )
    for seed, depth in expected:
        result = common.load_json(campaign.result_path(root, seed, depth))
        campaign.validate_result(manifest, result, seed, depth)
        common._require(
            result["result_sha256"] == sealed[(seed, depth)],
            f"vanilla result no longer matches seal {seed}/p{depth}",
        )
    return seal


def _target(
    table_row: Mapping[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    arrays = campaign.structural_arrays()
    objective, risk, service = campaign.objective_arrays(table_row)
    feasible = np.flatnonzero(arrays["feasible_mask"])
    minimum = int(np.min(objective[feasible]))
    target = np.sort(feasible[objective[feasible] == minimum]).astype(np.int64)
    common._require(target.size > 0, "empty vanilla ground space")
    return target, {
        "ground_indices": target.tolist(),
        "ground_energy": minimum,
        "ground_energy_degeneracy": int(target.size),
        "risk_by_ground_index": [int(risk[index]) for index in target],
        "service_by_ground_index": [int(service[index]) for index in target],
        "feasible_support_size": int(feasible.size),
    }


def _p0_state(device: torch.device) -> torch.Tensor:
    return torch.full(
        (len(campaign.RESTART_SEEDS), campaign.DIMENSION),
        complex(1 / math.sqrt(campaign.DIMENSION), 0),
        dtype=torch.complex64,
        device=device,
    )


def _replay(
    result: Mapping[str, object], diagonal: torch.Tensor, depth: int
) -> torch.Tensor:
    gamma = torch.as_tensor(
        result["gamma_by_restart"], dtype=torch.float32, device=diagonal.device
    )
    beta = torch.as_tensor(
        result["beta_by_restart"], dtype=torch.float32, device=diagonal.device
    )
    with torch.no_grad():
        loss, state = campaign.energy_fn(diagonal, depth)(gamma, beta)
    saved = torch.as_tensor(
        result["selected_expected_H_over_377"],
        dtype=torch.float32,
        device=diagonal.device,
    )
    common._require(
        torch.allclose(loss, saved, atol=2e-5, rtol=2e-5),
        f"vanilla saved-angle replay mismatch p{depth}",
    )
    return state


def _summary(
    cohort: str,
    claim_role: str,
    depth: int,
    table_seeds: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row[key] not in (None, "")]

    probability = values("ground_space_success_probability")
    feasible = values("feasible_mass")
    conditional = values("conditional_ground_probability_given_feasible")
    common._require(
        {int(row["table_seed"]) for row in rows} == set(table_seeds)
        and len(rows) == len(table_seeds) * len(campaign.RESTART_SEEDS),
        f"invalid nested-restart coverage for {cohort}/p{depth}",
    )
    table_gms = [
        _gm([
            float(row["ground_space_success_probability"])
            for row in rows if int(row["table_seed"]) == seed
        ])
        for seed in table_seeds
    ]
    common._require(all(value is not None for value in table_gms), "undefined table GM")
    costs = [row["RTS99_x_RU"] for row in rows]
    return {
        "cohort": cohort,
        "claim_role": claim_role,
        "depth": depth,
        "independent_table_count": len(table_seeds),
        "nested_optimizer_attempts_per_table": len(campaign.RESTART_SEEDS),
        "nested_optimizer_attempt_count": len(rows),
        "success_probability_arithmetic_mean": float(np.mean(probability)),
        "success_probability_geometric_mean": _gm(probability),
        "table_equal_success_probability_geometric_mean": _gm([float(v) for v in table_gms]),
        "success_probability_median": float(np.median(probability)),
        "success_probability_min": float(np.min(probability)),
        "success_probability_max": float(np.max(probability)),
        "feasible_mass_arithmetic_mean": float(np.mean(feasible)),
        "feasible_mass_geometric_mean": _gm(feasible),
        "conditional_ground_probability_geometric_mean": _gm(conditional),
        "ground_amplification_geometric_mean": _gm(values("amplification_over_full_uniform_ground_probability")),
        "feasible_amplification_geometric_mean": _gm(values("feasible_mass_amplification_over_full_uniform")),
        "conditional_amplification_geometric_mean": _gm(values("conditional_ground_amplification_over_uniform_feasible")),
        "RU": campaign.resource(depth),
        "RTS99_x_RU_geometric_mean": (
            _gm([float(value) for value in costs])
            if all(value is not None for value in costs) else None
        ),
    }


def score(root: Path, *, device: str) -> dict[str, object]:
    root = root.resolve()
    manifest, plan = campaign.validated_plan(root)
    seal = _verified_seal(root, manifest, plan)
    destination = root / OUTPUT_NAME
    if destination.exists():
        existing = common.load_json(destination)
        body = {key: value for key, value in existing.items() if key != "score_sha256"}
        common._require(
            existing.get("score_sha256") == common.sha256_json(body)
            and existing.get("global_seal_sha256") == seal["seal_sha256"]
            and existing["csv_exports"]["cell_csv_sha256"] == _file_sha256(root / CELL_CSV_NAME)
            and existing["csv_exports"]["summary_csv_sha256"] == _file_sha256(root / SUMMARY_CSV_NAME),
            "existing vanilla score invalid",
        )
        return existing

    resolved = torch.device(device)
    common._require(resolved.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    targets = {
        seed: _target(campaign.table(manifest, seed)) for seed in campaign.TABLE_SEEDS
    }
    cell_rows: list[dict[str, object]] = []
    for seed in campaign.TABLE_SEEDS:
        table_row = campaign.table(manifest, seed)
        diagonal, objective, penalty, feasible_mask = campaign.runtime_diagonals(
            table_row, resolved
        )
        target, target_lock = targets[seed]
        uniform_ground = len(target) / campaign.DIMENSION
        uniform_feasible = int(torch.sum(feasible_mask).cpu()) / campaign.DIMENSION
        uniform_conditional = len(target) / int(torch.sum(feasible_mask).cpu())
        for depth in campaign.DEPTHS:
            result: Mapping[str, object] | None = None
            if depth == 0:
                state = _p0_state(resolved)
            else:
                result = common.load_json(campaign.result_path(root, seed, depth))
                campaign.validate_result(manifest, result, seed, depth)
                state = _replay(result, diagonal, depth)
            metrics = campaign._metrics(state, objective, penalty, feasible_mask)
            if result is not None:
                for key in metrics:
                    common._require(
                        np.allclose(metrics[key], result["metrics"][key], atol=2e-5, rtol=2e-5),
                        f"vanilla metric replay mismatch {seed}/p{depth}/{key}",
                    )
            success = score_common._target_mass(state, target)
            ru = campaign.resource(depth)
            costs = score_common._costs(success, ru)
            for index, restart_seed in enumerate(campaign.RESTART_SEEDS):
                feasible = float(metrics["feasible_mass"][index])
                conditional = float(success[index]) / feasible if feasible > 0 else None
                expected_h = (
                    float(metrics["expected_raw_objective"][index])
                    / campaign.PHASE_AND_LOSS_SCALE
                    + float(metrics["expected_integer_penalty"][index])
                )
                cell_rows.append({
                    "depth": depth,
                    "table_seed": seed,
                    "table_id": campaign.table_id(seed),
                    "restart_index": index,
                    "restart_seed": restart_seed,
                    "p0_derived_anchor": depth == 0,
                    "ground_energy": target_lock["ground_energy"],
                    "ground_degeneracy": target_lock["ground_energy_degeneracy"],
                    "ground_space_success_probability": float(success[index]),
                    "feasible_mass": feasible,
                    "conditional_ground_probability_given_feasible": conditional,
                    "expected_raw_objective": float(metrics["expected_raw_objective"][index]),
                    "expected_integer_penalty": float(metrics["expected_integer_penalty"][index]),
                    "expected_H_over_377": expected_h,
                    "RTS99": score_common.rts99(float(success[index])),
                    "RU": ru,
                    "RTS99_x_RU": costs[index],
                    "amplification_over_full_uniform_ground_probability": float(success[index]) / uniform_ground,
                    "feasible_mass_amplification_over_full_uniform": feasible / uniform_feasible,
                    "conditional_ground_amplification_over_uniform_feasible": (
                        conditional / uniform_conditional if conditional is not None else None
                    ),
                    "optimizer_result_sha256": "" if result is None else result["result_sha256"],
                })

    summary_rows = []
    cohorts = (
        (
            "pooled10",
            "descriptive_only_because_first3_tables_were_inspected_before_extension",
            campaign.TABLE_SEEDS,
        ),
        (
            "fresh7",
            "unfiltered_replication_cohort_for_any_confirmatory_diagnostic_statement",
            campaign.extension.ADDITIONAL_TABLE_SEEDS,
        ),
    )
    for cohort, claim_role, table_seeds in cohorts:
        for depth in campaign.DEPTHS:
            rows = [
                row for row in cell_rows
                if row["depth"] == depth and int(row["table_seed"]) in table_seeds
            ]
            summary_rows.append(
                _summary(cohort, claim_role, depth, table_seeds, rows)
            )
    common._require(len(cell_rows) == 320 and len(summary_rows) == 16, "vanilla score coverage changed")
    cell_bytes = _csv_bytes(CELL_FIELDS, cell_rows)
    summary_bytes = _csv_bytes(SUMMARY_FIELDS, summary_rows)
    _atomic_write(root / CELL_CSV_NAME, cell_bytes)
    _atomic_write(root / SUMMARY_CSV_NAME, summary_bytes)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "status": "COMPLETE_VANILLA_QAOA_B3L6_TEN_TABLE_CURVES",
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "global_seal_sha256": seal["seal_sha256"],
        "full30_memory_certificate": manifest["full30_memory_certificate"],
        "target_locks_in_seed_order": [
            {"table_seed": seed, "table_id": campaign.table_id(seed), **targets[seed][1]}
            for seed in campaign.TABLE_SEEDS
        ],
        "cell_row_count": len(cell_rows),
        "summary_row_count": len(summary_rows),
        "curve_summary_rows": summary_rows,
        "statistical_unit_guardrail": {
            "independent_replication_unit": "service_coefficient_table",
            "pooled_independent_table_count": len(campaign.TABLE_SEEDS),
            "fresh_replication_independent_table_count": len(campaign.extension.ADDITIONAL_TABLE_SEEDS),
            "optimizer_restarts_per_table": len(campaign.RESTART_SEEDS),
            "optimizer_restarts_are_nested_attempts_not_independent_replicates": True,
            "uncertainty_or_significance_must_aggregate_within_table_first": True,
        },
        "chronology_guardrail": {
            "source_three_tables_inspected_before_extension_choice": True,
            "pooled10_curves_are_descriptive": True,
            "fresh7_tables_are_unfiltered_replication_cohort": True,
            "confirmatory_diagnostic_statements_must_be_supported_on_fresh7_separately": True,
        },
        "csv_exports": {
            "cell_csv": CELL_CSV_NAME,
            "cell_csv_sha256": hashlib.sha256(cell_bytes).hexdigest(),
            "summary_csv": SUMMARY_CSV_NAME,
            "summary_csv_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        },
        "claim_boundary": (
            "B3 is a structurally nonredundant diagnostic, not quantitative N30 evidence. "
            "Four optimizer restarts per table are nested attempts, not independent replicates."
        ),
    }
    payload = {**body, "score_sha256": common.sha256_json(body)}
    common.atomic_write_json(destination, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(score(args.campaign_root, device=args.device)["status"])


if __name__ == "__main__":
    main()
