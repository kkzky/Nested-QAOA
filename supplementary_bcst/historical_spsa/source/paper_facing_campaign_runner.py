from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import campaign_runner as base


def _range_mismatch(expected: float | None, observed: Any) -> bool:
    if expected is None:
        return observed is not None
    if observed is None:
        return True
    return abs(float(expected) - float(observed)) > 1e-12


def seed_angle_cache(
    runner,
    args,
    *,
    p1: int,
) -> dict[str, Any] | None:
    """Permit the preregistered LP-angle transfer into the warm block-XY control.

    All ordinary within-method history transfers continue to use the base
    validator unchanged.  The only cross-method transfer allowed here is from
    a validated learned-projector trial at the exact same instance and depth
    into a Warm-XY trial that uses the identical frozen Stage-1 checkpoint.
    """
    source_value = getattr(args, "initial_angle_trial", None)
    if not source_value:
        return None
    source = Path(source_value)
    with source.open(encoding="utf-8") as handle:
        row = json.load(handle)
    if not (
        args.algorithm == "Warm-XY"
        and row.get("algorithm") == "XY-LP-QAOA"
    ):
        return base._ORIGINAL_SEED_ANGLE_CACHE(runner, args, p1=p1)

    requested_depths = base.parse_ints(args.depths)
    if len(requested_depths) != 1:
        raise ValueError(
            "LP-angle-assisted Warm-XY must run exactly one depth per scan"
        )
    depth = int(row.get("depth", -1))
    expected = {
        "N": int(args.N),
        "seed": int(args.seed),
        "schedule": args.schedule,
        "family": args.family,
        "objective_model": args.objective_model,
        "p1": int(p1),
        "depth": int(requested_depths[0]),
        "algorithm": "XY-LP-QAOA",
        "validation_status": "passed",
        "completion_status": "complete",
    }
    mismatches = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if _range_mismatch(args.range_a, row.get("range_a")):
        mismatches["range_a"] = {
            "expected": args.range_a,
            "observed": row.get("range_a"),
        }
    target_stage1_sha = base.core.sha256_file(
        base.read_provisional_checkpoint(Path(args.stage1_checkpoint))
    )
    if row.get("stage1_checkpoint_sha256") != target_stage1_sha:
        mismatches["stage1_checkpoint_sha256"] = {
            "expected": target_stage1_sha,
            "observed": row.get("stage1_checkpoint_sha256"),
        }
    if mismatches:
        raise ValueError(
            f"LP-angle-assisted Warm-XY source mismatch: {mismatches}"
        )

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
    if gammas.numel() != depth or betas.numel() != depth:
        raise ValueError(
            "LP-angle source has invalid angle dimensions: "
            f"depth={depth}, gamma_count={gammas.numel()}, "
            f"beta_count={betas.numel()}"
        )
    runner._warm_param_cache.setdefault(p1, {})[depth] = (gammas, betas)
    return {
        "path": str(source),
        "sha256": base.core.sha256_file(source),
        "trial_id": row["trial_id"],
        "depth": depth,
        "selected_restart": int(
            row["optimizer_result"]["selected_restart"]
        ),
        "source_runner_hash": row["runner_hash"],
        "transfer": "XY-LP-QAOA_angles_to_same_stage1_Warm-XY",
        "source_penalty_kappa": float(row["penalty_kappa"]),
        "target_penalty_kappa": float(args.penalty_kappa),
        "support_sha256": row["support_sha256"],
        "objective_sha256": row["objective_sha256"],
        "target_manifest_sha256": row["target_manifest_sha256"],
        "stage1_checkpoint_sha256": target_stage1_sha,
    }


base._ORIGINAL_SEED_ANGLE_CACHE = base.seed_angle_cache
base.seed_angle_cache = seed_angle_cache
base.EXECUTION_SOURCE_HASHES["paper_facing_wrapper"] = (
    base.core.sha256_file(Path(__file__))
)


if __name__ == "__main__":
    base.main()
