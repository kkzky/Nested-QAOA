#!/usr/bin/env python3
"""Post-seal score for the coherent-IM3 2,400-evaluation validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import targeted_coherent_im3_continuation as continuation
import targeted_coherent_im3_score as discovery_score
import targeted_coherent_im3_screen as discovery
import targeted_three_stage_common as common


OUTPUT_NAME = "targeted_coherent_im3_continuation_post_seal_score.json"


def lock_and_score(root: Path, *, device: str) -> dict[str, object]:
    destination = root / OUTPUT_NAME
    if destination.exists():
        return common.load_json(destination)
    manifest, discovery_root, discovery_manifest, discovery_seal = (
        continuation.validated_manifest(root)
    )
    seal = common.load_json(root / continuation.SEAL_NAME)
    seal_body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    common._require(
        seal.get("schema") == f"{continuation.SCHEMA}-seal"
        and seal.get("status") == "SEALED_COMPLETE_TARGET_BLIND_COHERENT_IM3_VALIDATION"
        and seal.get("manifest_sha256") == manifest["manifest_sha256"]
        and seal.get("seal_sha256") == common.sha256_json(seal_body)
        and seal.get("method_count") == 4
        and seal.get("restart_count_per_method") == 4
        and seal.get("target_constructed") is False,
        "coherent-IM3 continuation seal is invalid",
    )
    locked = common.load_json(discovery_root / discovery_score.OUTPUT_NAME)
    locked_body = {key: value for key, value in locked.items() if key != "score_sha256"}
    common._require(
        locked.get("schema") == f"{discovery.SCHEMA}-post-seal-score"
        and locked.get("score_sha256") == common.sha256_json(locked_body)
        and locked.get("manifest_sha256") == discovery_manifest["manifest_sha256"]
        and locked.get("optimizer_seal_sha256") == discovery_seal["seal_sha256"],
        "discovery target lock is invalid",
    )
    target = np.asarray(
        locked["target_lock"]["best_two_native_shell_indices"], dtype=np.int64
    )
    common._require(target.shape == (2,), "locked coherent-IM3 target is invalid")

    stage_a = common.load_json(Path(str(discovery_manifest["stage_a_path"])))
    screen = common.make_screen(
        device=device, dtype="complex64", seeds=continuation.ALL_SEEDS
    )
    c_state, f_state, _ = common.materialize_structural_states(
        screen, common.stage_a_c12_row(stage_a)
    )
    objective = discovery.objective_tensor(
        screen, discovery_manifest["coefficient_table"]
    )
    rows: dict[str, dict[str, object]] = {}
    sealed_ids = seal["method_result_sha256"]
    for method_id, spec in discovery.specs().items():
        result = common.load_json(
            root / continuation.RESULT_DIRECTORY / f"{method_id}.json"
        )
        common._require(
            result.get("result_sha256") == sealed_ids.get(method_id),
            f"sealed continuation result changed for {method_id}",
        )
        reference_c = None if method_id == discovery.DIRECT else c_state
        reference_f = None if method_id == discovery.DIRECT else f_state
        energy, gamma_count, beta_count = discovery.energy_fn(
            screen, method_id, reference_c, reference_f, objective
        )
        gamma = torch.as_tensor(
            result["gamma_by_restart"], dtype=screen.real_dtype, device=screen.device
        )
        beta = torch.as_tensor(
            result["beta_by_restart"], dtype=screen.real_dtype, device=screen.device
        )
        common._require(
            gamma.shape == (4, gamma_count) and beta.shape == (4, beta_count),
            f"saved continuation angle shape mismatch for {method_id}",
        )
        with torch.no_grad():
            _, state = energy(gamma, beta)
        common._require(
            common.state_sha256(state) == result.get("state_sha256"),
            f"continuation saved-angle replay mismatch for {method_id}",
        )
        probability = discovery_score._target_mass(state, target)
        resource_rows: dict[str, object] = {}
        for convention, resource_key in discovery_score.CONVENTIONS.items():
            ru = int(result["resource"][resource_key])
            resource_rows[convention] = {
                "RU": ru,
                "RTS99_x_RU": discovery_score._costs(probability, ru),
            }
        rows[method_id] = {
            "best_two_probability": probability,
            "RTS99": [discovery_score.rts99(value) for value in probability],
            "final_mass": screen.metrics(state)["final_mass"],
            "best_evaluation_by_restart": result["best_evaluation_by_restart"],
            "resource_conventions": resource_rows,
        }

    comparisons: dict[str, object] = {}
    checks: dict[str, bool] = {}
    for convention in discovery_score.CONVENTIONS:
        full_cost = rows[discovery.FULL]["resource_conventions"][convention]["RTS99_x_RU"]
        controls = [
            rows[method_id]["resource_conventions"][convention]["RTS99_x_RU"]
            for method_id in continuation.METHODS
            if method_id != discovery.FULL
        ]
        envelope = []
        for restart in range(4):
            finite = [int(cost[restart]) for cost in controls if cost[restart] is not None]
            envelope.append(min(finite) if finite else None)
        ratios: list[float | None] = []
        full_wins = 0
        for control, lp in zip(envelope, full_cost):
            if control is None or lp is None or lp <= 0:
                ratios.append(None)
            else:
                ratio = float(control) / float(lp)
                ratios.append(ratio)
                full_wins += int(ratio > 1.0)
        finite_ratios = [value for value in ratios if value is not None and value > 0]
        gm = (
            float(np.exp(np.mean(np.log(np.asarray(finite_ratios, dtype=np.float64)))))
            if len(finite_ratios) == 4
            else None
        )
        comparisons[convention] = {
            "strongest_comparator_envelope_RTS99_x_RU": envelope,
            "paired_cost_ratio_control_over_full": ratios,
            "geometric_mean_control_over_full": gm,
            "full_paired_wins": full_wins,
            "paired_count": 4,
        }
        checks[f"{convention}_envelope_gm_at_least_1p20"] = (
            gm is not None and gm >= 1.20
        )
        checks[f"{convention}_full_wins_at_least_3_of_4"] = full_wins >= 3

    precursor_probability = discovery_score._target_mass(f_state, target)
    full_probability = rows[discovery.FULL]["best_two_probability"]
    checks["full_improves_fixed_F3_in_at_least_3_of_4"] = (
        sum(left > right for left, right in zip(full_probability, precursor_probability)) >= 3
    )
    checks["full_median_final_mass_at_least_0p90"] = (
        float(np.median(rows[discovery.FULL]["final_mass"])) >= 0.90
    )
    passed = all(checks.values())
    body: dict[str, object] = {
        "schema": f"{continuation.SCHEMA}-post-seal-score",
        "status": (
            "COHERENT_IM3_VALIDATION_GATE_PASS"
            if passed
            else "COHERENT_IM3_VALIDATION_GATE_FAIL"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "optimizer_seal_sha256": seal["seal_sha256"],
        "discovery_score_sha256": locked["score_sha256"],
        "target_lock": {
            "reused_unchanged_after_continuation_seal": True,
            **locked["target_lock"],
        },
        "precursor_best_two_probability": precursor_probability,
        "rows": rows,
        "comparisons_by_resource_convention": comparisons,
        "gate": {
            "pass": passed,
            "checks": checks,
            "rule": (
                "Under both native-C and Pauli-C ledgers, full requires >=3/4 "
                "paired wins and >=1.20 geometric-mean advantage over the "
                "per-restart envelope of all three controls; full also "
                "improves fixed F3 in >=3/4 and retains >=0.90 median final mass."
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
