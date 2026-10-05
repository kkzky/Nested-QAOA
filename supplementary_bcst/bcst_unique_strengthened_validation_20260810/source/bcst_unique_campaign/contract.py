"""Strict loader for the fixed-depth strengthened BCST validation campaign."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PLAN_SCHEMA = "bcst-unique-strengthened-validation-plan-v1"
OPTIMIZED_METHODS = (
    "learned_projector",
    "same_stage1_state_direct",
    "decoupled_native_direct",
    "ordinary_native_direct",
    "native_grover_d",
)
METHODS = OPTIMIZED_METHODS + ("stage1_only",)
SIZES = (25, 30)
CATEGORIES = frozenset({"core_selected", "direct_depth_scan", "matched_ru"})
SHA256_RE = re.compile(r"[0-9A-Fa-f]{64}")


class ContractError(ValueError):
    """Raised when immutable validation inputs are incomplete or inconsistent."""


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
        raise ContractError(f"{label} contains duplicates")
    return result


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a SHA-256 digest")
    return value.upper()


def _bound_path(plan_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a path")
    path = Path(value)
    if not path.is_absolute():
        path = plan_path.parent / path
    return path.resolve()


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
class FixedConfiguration:
    configuration_id: str
    N: int
    method: str
    depth: int | None
    Adam_updates: int | None
    terminal_RU: int
    categories: tuple[str, ...]

    @property
    def physical_key(self) -> tuple[int, str, int | None, int | None]:
        return (self.N, self.method, self.depth, self.Adam_updates)


@dataclass(frozen=True)
class CampaignContract:
    path: Path
    sha256: str
    record: Mapping[str, Any]
    campaign_id: str
    validation_seeds: Mapping[int, tuple[int, ...]]
    tuning_seeds: Mapping[int, tuple[int, ...]]
    stage1_seeds: Mapping[int, int]
    resources: Mapping[int, Mapping[str, int]]
    budgets: Mapping[tuple[str, int], int]
    stage1_imports: Mapping[int, Stage1Import]
    configurations: tuple[FixedConfiguration, ...]
    source_bindings: Mapping[str, tuple[Path, str]]

    @property
    def all_fresh_seeds(self) -> tuple[int, ...]:
        return tuple(
            seed for N in SIZES for seed in self.validation_seeds[N]
        )

    @property
    def configuration_by_id(self) -> Mapping[str, FixedConfiguration]:
        return {row.configuration_id: row for row in self.configurations}


def terminal_ru(resources: Mapping[str, int], method: str, depth: int | None) -> int:
    B, C, X, O, S, P1 = (
        int(resources[key]) for key in ("B", "C", "X", "O", "S", "P1")
    )
    if method == "stage1_only":
        if depth is not None:
            raise ContractError("Stage1Only depth must be null")
        return P1 + O
    if depth is None:
        raise ContractError(f"optimized method {method} lacks depth")
    p = int(depth)
    if method == "learned_projector":
        return P1 + p * (O + 2 * P1 + S) + O
    if method == "same_stage1_state_direct":
        return P1 + p * (C + O + X) + O
    if method in {"ordinary_native_direct", "decoupled_native_direct"}:
        return B + p * (C + O + X) + O
    if method == "native_grover_d":
        return B + p * (C + O + 2 * B + S) + O
    raise ContractError(f"unknown method {method}")


def _load_stage1_imports(
    plan_path: Path,
    stage1: Mapping[str, Any],
    stage1_seeds: Mapping[int, int],
) -> Mapping[int, Stage1Import]:
    raw_imports = _mapping(
        stage1.get("legacy_import_contract"), "stage1.legacy_import_contract"
    )
    result: dict[int, Stage1Import] = {}
    required = {
        "cell_id", "problem_seed", "source_problem_seed", "depth",
        "adopted_budget", "legacy_plan_sha256", "source_manifest_path",
        "source_manifest_sha256", "decision_path", "decision_sha256",
        "state_path", "state_file_sha256", "state_amplitude_sha256",
        "record_path", "record_sha256", "complete_path", "complete_sha256",
        "kdf_checkpoint_zero_sha256",
    }
    for N in SIZES:
        row = _mapping(raw_imports.get(str(N)), f"Stage1 import N={N}")
        if not required.issubset(row):
            raise ContractError(
                f"Stage1 import N={N} missing {sorted(required - set(row))}"
            )
        source_seed = _plain_int(row["source_problem_seed"], "Stage1 source seed")
        budget = _plain_int(row["adopted_budget"], "Stage1 budget", 1)
        if (
            source_seed != stage1_seeds[N]
            or row.get("problem_seed") != source_seed
            or row.get("depth") != 12
            or row.get("cell_id") != f"stage1_N{N}_seed{source_seed}_b{budget}"
        ):
            raise ContractError(f"Stage1 import identity mismatch at N={N}")
        raw_kdf = row["kdf_checkpoint_zero_sha256"]
        if not isinstance(raw_kdf, list) or len(raw_kdf) != 3:
            raise ContractError("Stage1 import requires three KDF hashes")
        kdf = tuple(_sha(value, "Stage1 KDF hash") for value in raw_kdf)
        if len(set(kdf)) != 3:
            raise ContractError("Stage1 KDF hashes must be distinct")
        result[N] = Stage1Import(
            N=N,
            source_problem_seed=source_seed,
            adopted_budget=budget,
            legacy_plan_sha256=_sha(row["legacy_plan_sha256"], "legacy plan"),
            source_manifest_path=_bound_path(
                plan_path, row["source_manifest_path"], "source manifest"
            ),
            source_manifest_sha256=_sha(
                row["source_manifest_sha256"], "source manifest"
            ),
            decision_path=_bound_path(plan_path, row["decision_path"], "decision"),
            decision_sha256=_sha(row["decision_sha256"], "decision"),
            state_path=_bound_path(plan_path, row["state_path"], "state"),
            state_file_sha256=_sha(row["state_file_sha256"], "state file"),
            amplitude_sha256=_sha(
                row["state_amplitude_sha256"], "state amplitudes"
            ),
            record_path=_bound_path(plan_path, row["record_path"], "record"),
            record_sha256=_sha(row["record_sha256"], "record"),
            complete_path=_bound_path(plan_path, row["complete_path"], "COMPLETE"),
            complete_sha256=_sha(row["complete_sha256"], "COMPLETE"),
            kdf_checkpoint_zero_sha256=kdf,  # type: ignore[arg-type]
        )
    return result


def _load_source_bindings(
    plan_path: Path, strengthening: Mapping[str, Any]
) -> Mapping[str, tuple[Path, str]]:
    row = _mapping(
        strengthening.get("source_campaign_binding"), "source_campaign_binding"
    )
    fields = {
        "plan": ("plan_path", "plan_sha256"),
        "final_depth_selection": (
            "final_depth_selection_path", "final_depth_selection_sha256"
        ),
        "final_results": ("final_results_path", "final_results_sha256"),
        "coverage_certificate": (
            "coverage_certificate_path", "coverage_certificate_sha256"
        ),
        "complete": ("complete_path", "complete_sha256"),
    }
    return {
        name: (
            _bound_path(plan_path, row[path_key], f"source {name}"),
            _sha(row[sha_key], f"source {name}"),
        )
        for name, (path_key, sha_key) in fields.items()
    }


def load_contract(path: Path | str) -> CampaignContract:
    plan_path = Path(path).resolve()
    try:
        raw = plan_path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read validation plan at {plan_path}") from exc
    if not isinstance(record, Mapping):
        raise ContractError("validation plan must be an object")
    if record.get("schema") != PLAN_SCHEMA or record.get("status") != "frozen_before_execution":
        raise ContractError("validation plan schema/status is incompatible")
    campaign_id = record.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ContractError("campaign_id must be nonempty")
    plan_hash = hashlib.sha256(raw).hexdigest().upper()

    scope = _mapping(record.get("scope"), "scope")
    if (
        scope.get("sizes") != [25, 30]
        or scope.get("primary_target") != "unique_ground_state"
        or scope.get("primary_metric") != "integer_RTS99_times_terminal_RU"
        or scope.get("paper_editing_authorized") is not False
    ):
        raise ContractError("scope differs from unique-ground numerical-only validation")
    construction = _mapping(record.get("construction"), "construction")
    if (
        construction.get("blocks") != 5
        or construction.get("selected_labels_per_block") != 2
        or construction.get("fine_objective")
        != "C_obj^(s)(x)=-sum_{T in T4^(s)} w_T product_{u in T} x_u - sum_{T in T6^(s)} w_T product_{u in T} x_u"
        or construction.get("ground_definition")
        != "the single lowest-C_obj conflict-feasible computational-basis state"
    ):
        raise ContractError("BCST construction identity changed")

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
        raise ContractError("optimizer is not three fresh fixed-budget Adam restarts")
    execution = _mapping(record.get("execution_contract"), "execution_contract")
    if execution.get("maximum_concurrent_optimizer_cells") != 4:
        raise ContractError("maximum concurrency must be four")

    seed_domains = _mapping(record.get("seed_domains"), "seed_domains")
    stage1_raw = _mapping(seed_domains.get("stage1_state_sources"), "Stage1 seeds")
    validation_raw = _mapping(
        seed_domains.get("fresh_heldout_validation"), "validation seeds"
    )
    stage1_seeds = {
        N: _plain_int(stage1_raw.get(str(N)), f"Stage1 seed N={N}") for N in SIZES
    }
    validation = {
        N: _int_list(validation_raw.get(str(N)), f"validation seeds N={N}")
        for N in SIZES
    }
    if any(len(validation[N]) != 15 for N in SIZES):
        raise ContractError("exactly fifteen validation seeds per size are required")
    all_validation = [seed for N in SIZES for seed in validation[N]]
    if len(set(all_validation)) != 30 or set(all_validation).intersection(stage1_seeds.values()):
        raise ContractError("validation seeds collide globally or with Stage1")

    constants = _mapping(
        _mapping(record.get("resource_model"), "resource_model").get("constants"),
        "resource constants",
    )
    resources: dict[int, Mapping[str, int]] = {}
    for N in SIZES:
        row = _mapping(constants.get(str(N)), f"resources N={N}")
        if set(row) != {"B", "C", "X", "O", "S", "P1"}:
            raise ContractError(f"resource fields differ at N={N}")
        resources[N] = {
            key: _plain_int(value, f"resource {key} N={N}")
            for key, value in row.items()
        }

    methods = _mapping(record.get("methods"), "methods")
    budgets: dict[tuple[str, int], int] = {}
    for method in OPTIMIZED_METHODS:
        updates = _mapping(
            _mapping(methods.get(method), f"methods.{method}").get("Adam_updates"),
            f"methods.{method}.Adam_updates",
        )
        for N in SIZES:
            budgets[(method, N)] = _plain_int(
                updates.get(str(N)), f"{method} budget N={N}", 1
            )

    strengthening = _mapping(
        record.get("validation_only_strengthening"), "validation_only_strengthening"
    )
    if (
        strengthening.get("no_depth_tuning") is not True
        or strengthening.get("no_adaptive_arm_activation") is not True
        or strengthening.get("no_target_based_selection") is not True
        or strengthening.get("categories") != sorted(CATEGORIES)
    ):
        raise ContractError("validation-only no-selection contract is incomplete")
    raw_configs = strengthening.get("fixed_configurations")
    if not isinstance(raw_configs, list) or len(raw_configs) != 23:
        raise ContractError("fixed configuration registry must contain 23 rows")
    configurations: list[FixedConfiguration] = []
    ids: set[str] = set()
    physical: set[tuple[int, str, int | None, int | None]] = set()
    for index, raw_row in enumerate(raw_configs):
        row = _mapping(raw_row, f"fixed configuration {index}")
        config_id = row.get("configuration_id")
        if (
            not isinstance(config_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", config_id)
            or config_id in ids
        ):
            raise ContractError("configuration_id is invalid or duplicated")
        N = _plain_int(row.get("N"), "configuration N")
        method = row.get("method")
        if N not in SIZES or method not in METHODS:
            raise ContractError(f"unknown configuration method/size {method}/{N}")
        depth_raw = row.get("depth")
        budget_raw = row.get("Adam_updates")
        if method == "stage1_only":
            if depth_raw is not None or budget_raw is not None:
                raise ContractError("Stage1Only configuration must not optimize")
            depth = None
            budget = None
        else:
            depth = _plain_int(depth_raw, "configuration depth", 1)
            budget = _plain_int(budget_raw, "configuration budget", 1)
            if budget != budgets[(str(method), N)]:
                raise ContractError(f"configuration budget differs for {config_id}")
        categories_raw = row.get("categories")
        if (
            not isinstance(categories_raw, list)
            or not categories_raw
            or any(category not in CATEGORIES for category in categories_raw)
            or len(set(categories_raw)) != len(categories_raw)
        ):
            raise ContractError(f"invalid categories for {config_id}")
        ru = _plain_int(row.get("terminal_RU"), "terminal_RU", 1)
        if ru != terminal_ru(resources[N], str(method), depth):
            raise ContractError(f"terminal RU does not reproduce for {config_id}")
        config = FixedConfiguration(
            configuration_id=config_id,
            N=N,
            method=str(method),
            depth=depth,
            Adam_updates=budget,
            terminal_RU=ru,
            categories=tuple(str(value) for value in categories_raw),
        )
        if config.physical_key in physical:
            raise ContractError(
                f"physical configuration duplicated instead of role-mapped: {config_id}"
            )
        ids.add(config_id)
        physical.add(config.physical_key)
        configurations.append(config)
    if (
        sum(row.N == 25 for row in configurations) != 11
        or sum(row.N == 30 for row in configurations) != 12
    ):
        raise ContractError("configuration counts must be N25=11 and N30=12")
    if sum(row.method != "stage1_only" for row in configurations) != 21:
        raise ContractError("fixed registry must contain twenty-one optimized configurations")
    expected_counts = strengthening.get("expected_physical_counts")
    if expected_counts != {
        "configurations": {"25": 11, "30": 12},
        "optimized_cells": 315,
        "stage1_only_readouts": 30,
        "total_cells": 345,
    }:
        raise ContractError("physical count contract differs from the 345-cell matrix")

    source_bindings = _load_source_bindings(plan_path, strengthening)
    stage1 = _mapping(record.get("stage1"), "stage1")
    imports = _load_stage1_imports(plan_path, stage1, stage1_seeds)
    return CampaignContract(
        path=plan_path,
        sha256=plan_hash,
        record=record,
        campaign_id=campaign_id,
        validation_seeds=validation,
        tuning_seeds={N: tuple() for N in SIZES},
        stage1_seeds=stage1_seeds,
        resources=resources,
        budgets=budgets,
        stage1_imports=imports,
        configurations=tuple(configurations),
        source_bindings=source_bindings,
    )


__all__ = [
    "CATEGORIES",
    "CampaignContract",
    "ContractError",
    "FixedConfiguration",
    "METHODS",
    "OPTIMIZED_METHODS",
    "PLAN_SCHEMA",
    "SIZES",
    "Stage1Import",
    "canonical_json_bytes",
    "load_contract",
    "sha256_file",
    "terminal_ru",
]
