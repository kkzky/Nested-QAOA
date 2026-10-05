#!/usr/bin/env python3
"""Post-seal target lock and frozen discovery gate for coherent IM3."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import targeted_coherent_im3_objective as coherent
import targeted_coherent_im3_screen as campaign
import targeted_three_stage_common as common


OUTPUT_NAME = "targeted_coherent_im3_post_seal_score.json"
CONVENTIONS = {
    "native_C": "primary_RU_with_O_terminal",
    "uniform_Pauli_C_sensitivity": "uniform_Pauli_C_sensitivity_primary_RU",
}


def rts99(probability: float) -> int | None:
    if not math.isfinite(probability) or probability <= 0.0:
        return None
    if probability >= 1.0:
        return 1
    return int(math.ceil(math.log(0.01) / math.log1p(-probability)))


def _target_mass(state: torch.Tensor, indices: np.ndarray) -> list[float]:
    selected = torch.as_tensor(indices, dtype=torch.int64, device=state.device)
    with torch.no_grad():
        probability = torch.abs(state) ** 2
        norm = probability.sum(dim=1).clamp_min(1e-24)
        value = torch.index_select(probability, 1, selected).sum(dim=1) / norm
    return [float(item) for item in value.cpu()]


def _costs(probabilities: Sequence[float], ru: int) -> list[int | None]:
    return [
        None if (shots := rts99(float(value))) is None else shots * int(ru)
        for value in probabilities
    ]


def _comparison(
    comparator: Sequence[int | None], full: Sequence[int | None]
) -> dict[str, object]:
    common._require(
        len(comparator) == len(full) == 3,
        "discovery comparison must contain three paired restarts",
    )
    ratios: list[float | None] = []
    full_wins = 0
    for control, lp in zip(comparator, full):
        if control is None or lp is None or lp <= 0:
            ratios.append(None)
            continue
        ratio = float(control) / float(lp)
        ratios.append(ratio)
        full_wins += int(ratio > 1.0)
    finite = [value for value in ratios if value is not None and value > 0]
    geometric = (
        float(np.exp(np.mean(np.log(np.asarray(finite, dtype=np.float64)))))
        if len(finite) == 3
        else None
    )
    return {
        "paired_cost_ratio_control_over_full": ratios,
        "geometric_mean_control_over_full": geometric,
        "full_paired_wins": full_wins,
        "paired_count": 3,
    }


def _validated_seal(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest, _ = campaign.validated_manifest(root)
    seal = common.load_json(root / campaign.SEAL_NAME)
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    common._require(
        seal.get("schema") == f"{campaign.SCHEMA}-seal"
        and seal.get("status") == "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_SCREEN"
        and seal.get("manifest_sha256") == manifest["manifest_sha256"]
        and seal.get("seal_sha256") == common.sha256_json(body)
        and seal.get("method_count") == 4
        and seal.get("restart_count_per_method") == 3
        and seal.get("target_constructed") is False,
        "coherent-IM3 optimizer seal is invalid",
    )
    return manifest, seal


def lock_and_score(root: Path, *, device: str) -> dict[str, object]:
    destination = root / OUTPUT_NAME
    if destination.exists():
        return common.load_json(destination)
    manifest, seal = _validated_seal(root)
    stage_a = common.load_json(Path(str(manifest["stage_a_path"])))
    screen = common.make_screen(
        device=device, dtype="complex64", seeds=campaign.RESTART_SEEDS
    )
    c_state, f_state, _ = common.materialize_structural_states(
        screen, common.stage_a_c12_row(stage_a)
    )
    raw, risk, service = coherent.native_shell_components(screen)
    final_indices = np.flatnonzero(screen.final_mask.detach().cpu().numpy())
    common._require(final_indices.size == 390, "final sector changed")
    order = np.lexsort((final_indices, raw[final_indices]))
    best_two = final_indices[order[:2]]
    objective = torch.as_tensor(
        raw, dtype=screen.real_dtype, device=screen.device
    ) / coherent.PHASE_SCALE

    sealed_ids = seal["method_result_sha256"]
    common._require(isinstance(sealed_ids, dict), "invalid coherent-IM3 seal map")
    rows: dict[str, dict[str, object]] = {}
    for method_id, spec in campaign.specs().items():
        result = common.load_json(
            root / campaign.RESULT_DIRECTORY / f"{method_id}.json"
        )
        common._require(
            result.get("result_sha256") == sealed_ids.get(method_id),
            f"sealed coherent-IM3 result changed for {method_id}",
        )
        reference_c = None if method_id == campaign.DIRECT else c_state
        reference_f = None if method_id == campaign.DIRECT else f_state
        energy, gamma_count, beta_count = campaign.energy_fn(
            screen, method_id, reference_c, reference_f, objective
        )
        gamma = torch.as_tensor(
            result["gamma_by_restart"], dtype=screen.real_dtype, device=screen.device
        )
        beta = torch.as_tensor(
            result["beta_by_restart"], dtype=screen.real_dtype, device=screen.device
        )
        common._require(
            gamma.shape == (3, gamma_count) and beta.shape == (3, beta_count),
            f"saved coherent-IM3 angle shape mismatch for {method_id}",
        )
        with torch.no_grad():
            _, state = energy(gamma, beta)
        common._require(
            common.state_sha256(state) == result.get("state_sha256"),
            f"saved-angle replay mismatch for {method_id}",
        )
        probability = _target_mass(state, best_two)
        resources = result["resource"]
        resource_rows: dict[str, object] = {}
        for convention, resource_key in CONVENTIONS.items():
            ru = int(resources[resource_key])
            resource_rows[convention] = {"RU": ru, "RTS99_x_RU": _costs(probability, ru)}
        rows[method_id] = {
            "best_two_probability": probability,
            "RTS99": [rts99(value) for value in probability],
            "final_mass": screen.metrics(state)["final_mass"],
            "resource_conventions": resource_rows,
            "best_evaluation_by_restart": result["best_evaluation_by_restart"],
        }

    comparisons: dict[str, object] = {}
    checks: dict[str, bool] = {}
    for convention in CONVENTIONS:
        full_cost = rows[campaign.FULL]["resource_conventions"][convention]["RTS99_x_RU"]
        control_costs = [
            rows[method_id]["resource_conventions"][convention]["RTS99_x_RU"]
            for method_id in campaign.METHODS
            if method_id != campaign.FULL
        ]
        envelope = []
        for restart in range(3):
            finite = [
                int(cost[restart])
                for cost in control_costs
                if cost[restart] is not None
            ]
            envelope.append(min(finite) if finite else None)
        comparison = _comparison(envelope, full_cost)
        comparison["strongest_comparator_envelope_RTS99_x_RU"] = envelope
        comparisons[convention] = comparison
        gm = comparison["geometric_mean_control_over_full"]
        checks[f"{convention}_envelope_gm_at_least_1p10"] = (
            gm is not None and float(gm) >= 1.10
        )
        checks[f"{convention}_full_wins_at_least_2_of_3"] = (
            int(comparison["full_paired_wins"]) >= 2
        )

    precursor_probability = _target_mass(f_state, best_two)
    full_probability = rows[campaign.FULL]["best_two_probability"]
    checks["full_improves_fixed_F3_in_at_least_2_of_3"] = (
        sum(left > right for left, right in zip(full_probability, precursor_probability))
        >= 2
    )
    checks["full_median_final_mass_at_least_0p90"] = (
        float(np.median(rows[campaign.FULL]["final_mass"])) >= 0.90
    )
    passed = all(checks.values())
    body: dict[str, object] = {
        "schema": f"{campaign.SCHEMA}-post-seal-score",
        "status": (
            "COHERENT_IM3_DISCOVERY_GATE_PASS"
            if passed
            else "COHERENT_IM3_DISCOVERY_GATE_FAIL"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "optimizer_seal_sha256": seal["seal_sha256"],
        "target_lock": {
            "constructed_only_after_four_method_seal": True,
            "best_two_native_shell_indices": best_two.tolist(),
            "best_two_objective_values": raw[best_two].tolist(),
            "best_two_squared_risk": risk[best_two].tolist(),
            "best_two_service_cost": service[best_two].tolist(),
            "tie_rule": "raw objective then canonical native-shell index",
        },
        "precursor_best_two_probability": precursor_probability,
        "rows": rows,
        "comparisons_by_resource_convention": comparisons,
        "gate": {
            "pass": passed,
            "checks": checks,
            "rule": (
                "Under both native-C and Pauli-C ledgers, full needs >=2/3 "
                "paired wins and >=1.10 geometric-mean advantage over the "
                "per-restart envelope of all three controls; it must also "
                "improve fixed F3 in >=2/3 and retain >=0.90 median final mass."
            ),
        },
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
    print(lock_and_score(args.campaign_root, device=args.device)["status"])


if __name__ == "__main__":
    main()

