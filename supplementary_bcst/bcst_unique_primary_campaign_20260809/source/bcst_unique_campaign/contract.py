"""Load and validate the frozen unique-ground campaign contract.

The experiment plan is data, not a collection of defaults.  This module keeps
all controller decisions tied to its exact bytes and rejects a partially
compatible plan instead of silently filling missing fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PLAN_SCHEMA = "bcst-unique-primary-experiment-plan-v1"
OPTIMIZED_METHODS = (
    "learned_projector",
    "same_stage1_state_direct",
    "decoupled_native_direct",
    "ordinary_native_direct",
    "native_grover_d",
)
SIZES = (25, 30)
SHA256_RE = re.compile(r"[0-9A-Fa-f]{64}")


class ContractError(ValueError):
    """Raised when the frozen plan is missing or internally inconsistent."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _plain_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be a plain integer >= {minimum}")
    return value


def _int_list(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a nonempty integer list")
    result = tuple(_plain_int(item, f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{label} contains duplicate values")
    return result


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a SHA-256 digest")
    return value.upper()


@dataclass(frozen=True)
class Stage1Import:
    N: int
    source_problem_seed: int
    adopted_budget: int
    legacy_plan_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    decision_path: Path
    decision_sha256: str
    state_path: Path
    state_file_sha256: str
    amplitude_sha256: str
    record_path: Path
    record_sha256: str
    complete_path: Path
    complete_sha256: str
    kdf_checkpoint_zero_sha256: tuple[str, str, str]


@dataclass(frozen=True)
class CampaignContract:
    path: Path
    sha256: str
    record: Mapping[str, Any]
    campaign_id: str
    tuning_seeds: Mapping[int, tuple[int, ...]]
    validation_seeds: Mapping[int, tuple[int, ...]]
    stage1_seeds: Mapping[int, int]
    resources: Mapping[int, Mapping[str, int]]
    budgets: Mapping[tuple[str, int], int]
    base_depths: tuple[int, ...]
    first_extension: tuple[int, ...]
    second_extension: tuple[int, ...]
    lp_n30_extension: tuple[int, ...]
    stage1_imports: Mapping[int, Stage1Import]

    @property
    def all_fresh_seeds(self) -> tuple[int, ...]:
        return tuple(
            seed
            for size in SIZES
            for seed in self.tuning_seeds[size] + self.validation_seeds[size]
        )

    @property
    def maximum_tuning_depths(self) -> Mapping[tuple[str, int], tuple[int, ...]]:
        rows: dict[tuple[str, int], tuple[int, ...]] = {}
        common = self.base_depths + self.first_extension + self.second_extension
        for size in SIZES:
            for method in OPTIMIZED_METHODS:
                depths = common
                if method == "learned_projector" and size == 30:
                    depths += self.lp_n30_extension
                rows[(method, size)] = depths
        return rows


def _load_stage1_imports(
    plan_path: Path, stage1: Mapping[str, Any], stage1_seeds: Mapping[int, int]
) -> Mapping[int, Stage1Import]:
    raw_imports = stage1.get("legacy_import_contract")
    if not isinstance(raw_imports, Mapping):
        raise ContractError(
            "stage1.legacy_import_contract with exact legacy paths and hashes is required; "
            "Stage-1 regeneration is forbidden by this plan"
        )
    result: dict[int, Stage1Import] = {}
    for size in SIZES:
        row = _mapping(
            raw_imports.get(str(size)), f"stage1.legacy_import_contract.{size}"
        )
        required = {
            "cell_id",
            "problem_seed",
            "source_problem_seed",
            "depth",
            "adopted_budget",
            "legacy_plan_sha256",
            "source_manifest_path",
            "source_manifest_sha256",
            "decision_path",
            "decision_sha256",
            "state_path",
            "state_file_sha256",
            "state_amplitude_sha256",
            "record_path",
            "record_sha256",
            "complete_path",
            "complete_sha256",
            "kdf_checkpoint_zero_sha256",
        }
        if not required.issubset(row):
            raise ContractError(
                f"stage1.legacy_import_contract.{size} is incomplete; "
                f"missing={sorted(required - set(row))}"
            )
        raw_kdf = row["kdf_checkpoint_zero_sha256"]
        if not isinstance(raw_kdf, list) or len(raw_kdf) != 3:
            raise ContractError(
                f"stage1.legacy_import_contract.{size} requires exactly three KDF hashes"
            )
        kdf = tuple(
            _sha(value, f"stage1.legacy_import_contract.{size}.kdf[{index}]")
            for index, value in enumerate(raw_kdf)
        )
        if len(set(kdf)) != 3:
            raise ContractError(
                f"stage1.legacy_import_contract.{size} KDF hashes must be distinct"
            )

        def source_path(field: str) -> Path:
            value = row[field]
            if not isinstance(value, str) or not value:
                raise ContractError(
                    f"stage1.legacy_import_contract.{size}.{field} must be a path"
                )
            path = Path(value)
            if not path.is_absolute():
                path = (plan_path.parent / path).resolve()
            return path

        source_seed = _plain_int(
            row["source_problem_seed"],
            f"stage1.legacy_import_contract.{size}.source_problem_seed",
        )
        if (
            source_seed != stage1_seeds[size]
            or row.get("problem_seed") != source_seed
            or row.get("depth") != 12
            or row.get("cell_id")
            != f"stage1_N{size}_seed{source_seed}_b{row.get('adopted_budget')}"
        ):
            raise ContractError(f"Stage-1 import seed mismatch at N={size}")
        adopted_budget = _plain_int(
            row["adopted_budget"],
            f"stage1.legacy_import_contract.{size}.adopted_budget",
            1,
        )
        result[size] = Stage1Import(
            N=size,
            source_problem_seed=source_seed,
            adopted_budget=adopted_budget,
            legacy_plan_sha256=_sha(
                row["legacy_plan_sha256"],
                f"stage1.legacy_import_contract.{size}.legacy_plan_sha256",
            ),
            source_manifest_path=source_path("source_manifest_path"),
            source_manifest_sha256=_sha(
                row["source_manifest_sha256"],
                f"stage1.legacy_import_contract.{size}.source_manifest_sha256",
            ),
            decision_path=source_path("decision_path"),
            decision_sha256=_sha(
                row["decision_sha256"],
                f"stage1.legacy_import_contract.{size}.decision_sha256",
            ),
            state_path=source_path("state_path"),
            state_file_sha256=_sha(
                row["state_file_sha256"],
                f"stage1.legacy_import_contract.{size}.state_file_sha256",
            ),
            amplitude_sha256=_sha(
                row["state_amplitude_sha256"],
                f"stage1.legacy_import_contract.{size}.state_amplitude_sha256",
            ),
            record_path=source_path("record_path"),
            record_sha256=_sha(
                row["record_sha256"],
                f"stage1.legacy_import_contract.{size}.record_sha256",
            ),
            complete_path=source_path("complete_path"),
            complete_sha256=_sha(
                row["complete_sha256"],
                f"stage1.legacy_import_contract.{size}.complete_sha256",
            ),
            kdf_checkpoint_zero_sha256=kdf,  # type: ignore[arg-type]
        )
    return result


def load_contract(path: Path | str) -> CampaignContract:
    plan_path = Path(path).resolve()
    try:
        raw = plan_path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read experiment plan at {plan_path}") from exc
    if not isinstance(record, Mapping):
        raise ContractError("experiment plan must be a JSON object")
    if record.get("schema") != PLAN_SCHEMA:
        raise ContractError("experiment plan has the wrong schema")
    if record.get("status") != "frozen_before_execution":
        raise ContractError("experiment plan is not frozen before execution")
    campaign_id = record.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ContractError("campaign_id must be a nonempty string")
    plan_hash = hashlib.sha256(raw).hexdigest().upper()

    scope = _mapping(record.get("scope"), "scope")
    if (
        scope.get("sizes") != [25, 30]
        or scope.get("primary_target") != "unique_ground_state"
        or scope.get("primary_metric") != "integer_RTS99_times_terminal_RU"
        or scope.get("paper_editing_authorized") is not False
    ):
        raise ContractError("scope is not the frozen unique-ground numerical-only scope")

    construction = _mapping(record.get("construction"), "construction")
    expected_objective = (
        "C_obj^(s)(x)=-sum_{T in T4^(s)} w_T product_{u in T} x_u - "
        "sum_{T in T6^(s)} w_T product_{u in T} x_u"
    )
    sign_convention = str(construction.get("objective_sign_convention", ""))
    if (
        construction.get("blocks") != 5
        or construction.get("selected_labels_per_block") != 2
        or construction.get("fine_objective") != expected_objective
        or "diagonal cost Hamiltonian is their negative" not in sign_convention
        or "minimizing C_obj" not in sign_convention
        or construction.get("ground_definition")
        != "the single lowest-C_obj conflict-feasible computational-basis state"
    ):
        raise ContractError(
            "construction/objective sign differs from the exact negative-reward "
            "Hamiltonian implemented by bcst_v2.instance_core"
        )

    optimizer = _mapping(record.get("optimizer"), "optimizer")
    if (
        optimizer.get("name") != "Adam"
        or optimizer.get("learning_rate") != 0.035
        or optimizer.get("independent_restarts_per_optimized_cell") != 3
        or optimizer.get("continuation") is not False
        or optimizer.get("layerwise_continuation") is not False
        or optimizer.get("cross_depth_parameter_transfer") is not False
        or optimizer.get("cross_method_parameter_transfer") is not False
        or optimizer.get("optimizer_state_transfer") is not False
        or optimizer.get("early_stopping") is not False
    ):
        raise ContractError("optimizer contract is not three fresh Adam restarts")

    execution = _mapping(record.get("execution_contract"), "execution_contract")
    if execution.get("maximum_concurrent_optimizer_cells") != 4:
        raise ContractError("maximum concurrent optimizer cells must be exactly four")

    seed_domains = _mapping(record.get("seed_domains"), "seed_domains")
    stage1_source = _mapping(
        seed_domains.get("stage1_state_sources"), "seed_domains.stage1_state_sources"
    )
    tuning_raw = _mapping(
        seed_domains.get("fresh_depth_tuning"), "seed_domains.fresh_depth_tuning"
    )
    validation_raw = _mapping(
        seed_domains.get("fresh_heldout_validation"),
        "seed_domains.fresh_heldout_validation",
    )
    stage1_seeds = {
        size: _plain_int(stage1_source.get(str(size)), f"Stage-1 seed N={size}")
        for size in SIZES
    }
    tuning = {
        size: _int_list(tuning_raw.get(str(size)), f"tuning seeds N={size}")
        for size in SIZES
    }
    validation = {
        size: _int_list(validation_raw.get(str(size)), f"validation seeds N={size}")
        for size in SIZES
    }
    if any(len(tuning[size]) != 2 or len(validation[size]) != 10 for size in SIZES):
        raise ContractError("each size requires two tuning and ten validation seeds")
    all_fresh = [seed for size in SIZES for seed in tuning[size] + validation[size]]
    if len(set(all_fresh)) != 24:
        raise ContractError("fresh tuning and validation seeds are not globally disjoint")
    if set(all_fresh).intersection(stage1_seeds.values()):
        raise ContractError("fresh seeds collide with a Stage-1 source seed")

    resource_model = _mapping(record.get("resource_model"), "resource_model")
    constants = _mapping(resource_model.get("constants"), "resource_model.constants")
    resources: dict[int, Mapping[str, int]] = {}
    for size in SIZES:
        row = _mapping(constants.get(str(size)), f"resource constants N={size}")
        if set(row) != {"B", "C", "X", "O", "S", "P1"}:
            raise ContractError(f"resource constants N={size} have incorrect fields")
        resources[size] = {
            key: _plain_int(value, f"resource {key} N={size}")
            for key, value in row.items()
        }

    methods = _mapping(record.get("methods"), "methods")
    if not set(OPTIMIZED_METHODS).issubset(methods):
        raise ContractError("method registry is incomplete")
    budgets: dict[tuple[str, int], int] = {}
    for method in OPTIMIZED_METHODS:
        row = _mapping(methods[method], f"methods.{method}")
        updates = _mapping(row.get("Adam_updates"), f"methods.{method}.Adam_updates")
        for size in SIZES:
            budgets[(method, size)] = _plain_int(
                updates.get(str(size)), f"{method} N={size} Adam updates", 1
            )

    tuning_contract = _mapping(record.get("depth_tuning"), "depth_tuning")
    if tuning_contract.get("applies_to") != list(OPTIMIZED_METHODS):
        raise ContractError("depth tuning method order differs from the method registry")
    base = _int_list(tuning_contract.get("base_grid"), "base depth grid")
    first = _int_list(
        tuning_contract.get("first_boundary_extension"), "first extension"
    )
    second = _int_list(
        tuning_contract.get("second_boundary_extension"), "second extension"
    )
    lp_cap = _int_list(
        tuning_contract.get("lp_N30_capped_extension"), "LP N30 capped extension"
    )
    if base != (1, 2, 4, 8, 12, 16, 24, 36):
        raise ContractError("base depth grid differs from the frozen contract")
    if first != (48, 64) or second != (80, 96) or lp_cap != (112, 128):
        raise ContractError("depth extension grids differ from the frozen contract")
    expected_extension_rules = {
        "base_to_first_rule": "Run [48,64] iff p=36 is the finite base-grid minimum or cost(p=36)<=1.10*base_min and cost(p=36)<=cost(p=24).",
        "first_to_second_rule": "Run [80,96] iff p=64 is the finite tested-grid minimum or cost(p=64)<=1.10*tested_min and cost(p=64)<=cost(p=48).",
        "lp_N30_second_to_capped_rule": "For learned_projector at N=30 only, run [112,128] iff p=96 is the finite tested-grid minimum or cost(p=96)<=1.10*tested_min and cost(p=96)<=cost(p=80).",
    }
    if any(tuning_contract.get(key) != value for key, value in expected_extension_rules.items()):
        raise ContractError("depth boundary tie rule must be exactly nonincreasing (<=)")
    if tuning_contract.get("optimizer_inconclusive_candidate_rule") != (
        "For learned_projector only, a depth is selection- and extension-ineligible "
        "if either tuning seed has optimizer_inconclusive health; if no healthy finite "
        "LP depth exists, block that stratum. For same_stage1_state_direct, "
        "decoupled_native_direct, ordinary_native_direct, and native_grover_d, every "
        "finite cost remains selection- and extension-eligible regardless of optimizer "
        "health. Always disclose every candidate's per-seed health, all inconclusive "
        "candidate depths, and whether the selected depth passed health; never censor "
        "or block a finite baseline solely because its optimizer health is inconclusive."
    ):
        raise ContractError("optimizer-inconclusive tuning eligibility rule changed")

    stage1 = _mapping(record.get("stage1"), "stage1")
    if stage1.get("depth_p1") != 12 or "Reuse" not in str(stage1.get("reuse_rule", "")):
        raise ContractError("Stage-1 must be p1=12 and imported, not regenerated")
    imports = _load_stage1_imports(plan_path, stage1, stage1_seeds)

    return CampaignContract(
        path=plan_path,
        sha256=plan_hash,
        record=record,
        campaign_id=campaign_id,
        tuning_seeds=tuning,
        validation_seeds=validation,
        stage1_seeds=stage1_seeds,
        resources=resources,
        budgets=budgets,
        base_depths=base,
        first_extension=first,
        second_extension=second,
        lp_n30_extension=lp_cap,
        stage1_imports=imports,
    )


__all__ = [
    "CampaignContract",
    "ContractError",
    "OPTIMIZED_METHODS",
    "PLAN_SCHEMA",
    "SIZES",
    "Stage1Import",
    "canonical_json_bytes",
    "load_contract",
    "sha256_file",
]
