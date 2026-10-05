"""Shared primitives for the targeted three-stage BCST development check.

This module intentionally contains no target construction or ranking.  It
replays the already completed ``Coarse-only-C-p12`` preparation, materializes
the exact three-iteration feasibility-amplification circuit selected in Stage
A, and defines the small objective-method set used by the target-blind
optimizer executable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import conditional_stage3_objective_launcher as launcher
import stage3_objective_common as objective_engine
import target_free_structural_screen as engine


SCHEMA = "nonredundant-bcst-targeted-three-stage-development-v1"
MANIFEST_SCHEMA = f"{SCHEMA}-manifest"
PRECURSOR_SCHEMA = f"{SCHEMA}-precursors"
METHOD_RESULT_SCHEMA = f"{SCHEMA}-method-result"
SEAL_SCHEMA = f"{SCHEMA}-optimizer-seal"
SCORE_SCHEMA = f"{SCHEMA}-post-seal-score"

OBJECTIVE_SEED = 20_260_811
PILOT_RESTART_SEEDS = tuple(range(2_026_081_100, 2_026_081_104))
ALL_STAGE_A_RESTART_SEEDS = tuple(range(2_026_081_100, 2_026_081_116))

P_C = 12
P_FEASIBILITY = 3
# Use the lowest objective depth in the already frozen Stage-3 protocol.  The
# pilot deliberately tests one point rather than opening another depth grid.
P_OBJECTIVE = 16
P_COLLAPSE = P_FEASIBILITY + P_OBJECTIVE
P_TOTAL = P_C + P_FEASIBILITY + P_OBJECTIVE
GROVER_FIXED_ANGLES = math.pi

COARSE_METHOD = "coarse_static_C12"
PRECURSOR_METHOD = "precursor_C12_F3"
FULL_METHOD = "full_C12_F3_O16"
COLLAPSE_JOINT = "collapse_C12_QO_joint_p19"
COLLAPSE_SEPARATE = "collapse_C12_QO_separate_p19"
COARSE_REFINE = "coarse_refine_C12_O16"
FIXED_UNIFORM = "fixed_uniform_F32_O16"
WAVE1_METHODS = (
    FULL_METHOD,
    COLLAPSE_JOINT,
    COLLAPSE_SEPARATE,
)


class TargetedValidationError(RuntimeError):
    """A scientific contract of the targeted development check was violated."""


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    depth: int
    phase_names: tuple[str, ...]
    evaluations: int
    stream: int
    reference_kind: str
    primary_ru: int


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TargetedValidationError(f"{path} is not a JSON object")
    return value


def atomic_write_json(path: Path, payload: object) -> None:
    engine.atomic_write_json(path, payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetedValidationError(message)


def stage_a_c12_row(stage_a: Mapping[str, object]) -> dict[str, object]:
    """Extract the unique formal ``Coarse-only-C-p12`` saved-angle row."""

    _require(
        stage_a.get("schema") == "nonredundant-bcst-target-free-structural-screen-v2",
        "Stage-A schema mismatch",
    )
    _require(
        stage_a.get("execution_stage") == "A_NECESSITY_COMPLETE_16_RESTARTS",
        "Stage A is not the complete formal run",
    )
    gate = stage_a.get("gate")
    _require(
        isinstance(gate, dict)
        and gate.get("pass") is True
        and gate.get("status") == "PROCEED_TO_EXPENSIVE_STAGES",
        "Stage A did not pass",
    )
    rows = stage_a.get("methods")
    _require(isinstance(rows, list), "Stage A has no method rows")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("algorithm") == "Coarse-only-C-p12"
    ]
    _require(len(matches) == 1, "Stage A must contain one Coarse-only-C-p12 row")
    row = dict(matches[0])
    _require(
        tuple(int(value) for value in row.get("initialization_seeds", ()))
        == ALL_STAGE_A_RESTART_SEEDS,
        "C12 row does not contain the formal 16 seeds in order",
    )
    for name in ("gamma_by_restart", "beta_by_restart"):
        values = np.asarray(row.get(name), dtype=np.float64)
        _require(values.shape == (16, P_C), f"invalid C12 {name} shape")
        _require(np.all(np.isfinite(values)), f"nonfinite C12 {name}")
    mass = np.asarray(row.get("final_mass"), dtype=np.float64)
    _require(mass.shape == (16,) and np.all(np.isfinite(mass)), "invalid C12 mass")
    return row


def make_screen(
    *, device: str, dtype: str, seeds: Sequence[int] = PILOT_RESTART_SEEDS
) -> engine.StructuralScreen:
    selected = tuple(int(seed) for seed in seeds)
    config = replace(
        engine.valid_config(),
        restarts=len(selected),
        initialization_seeds=selected,
        restart_batch=len(selected),
        device=device,
        dtype=dtype,
        activation_checkpointing=True,
    )
    return engine.StructuralScreen(config)


def _slice_saved_angles(
    row: Mapping[str, object], seeds: Sequence[int], key: str
) -> np.ndarray:
    positions = {
        int(seed): index for index, seed in enumerate(row["initialization_seeds"])
    }
    try:
        indices = [positions[int(seed)] for seed in seeds]
    except KeyError as error:
        raise TargetedValidationError(f"unknown Stage-A restart seed {error.args[0]}") from error
    return np.asarray(row[key], dtype=np.float64)[indices]


def replay_c12(
    screen: engine.StructuralScreen,
    c12_row: Mapping[str, object],
) -> torch.Tensor:
    """Replay the saved C12 angles once; never reoptimize Stage 1."""

    seeds = screen.config.initialization_seeds
    gamma = torch.as_tensor(
        _slice_saved_angles(c12_row, seeds, "gamma_by_restart"),
        dtype=screen.real_dtype,
        device=screen.device,
    )
    beta = torch.as_tensor(
        _slice_saved_angles(c12_row, seeds, "beta_by_restart"),
        dtype=screen.real_dtype,
        device=screen.device,
    )
    with torch.no_grad():
        _, state = screen.block_energy(
            screen.native, (screen.c,), screen.c, gamma, beta, P_C
        )
    return state


def exact_grover_state(
    reference: torch.Tensor,
    good_mask: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Apply exact indicator phase flips and reflections about ``reference``.

    With every phase and reflection angle fixed to :math:`\\pi`, this is the
    literal state-vector realization of the Stage-A analytic formula
    ``sin((2p+1) asin(sqrt(m)))**2``.
    """

    if reference.ndim != 2 or good_mask.ndim != 1:
        raise TargetedValidationError("Grover reference/mask rank mismatch")
    if reference.shape[1] != good_mask.shape[0] or depth < 0:
        raise TargetedValidationError("Grover reference/mask/depth mismatch")
    state = reference.clone()
    signs = torch.where(
        good_mask,
        torch.tensor(-1.0, dtype=reference.real.dtype, device=reference.device),
        torch.tensor(1.0, dtype=reference.real.dtype, device=reference.device),
    ).to(reference.dtype)
    beta = torch.full(
        (reference.shape[0],),
        GROVER_FIXED_ANGLES,
        dtype=reference.real.dtype,
        device=reference.device,
    )
    for _ in range(int(depth)):
        state = state * signs[None, :]
        state = engine.apply_history(state, beta, reference)
        state = engine.renormalize(state)
    return state


def grover_success(initial_mass: np.ndarray | Sequence[float], depth: int) -> np.ndarray:
    value = np.clip(np.asarray(initial_mass, dtype=np.float64), 0.0, 1.0)
    return np.sin((2 * int(depth) + 1) * np.arcsin(np.sqrt(value))) ** 2


def state_sha256(state: torch.Tensor) -> str:
    return objective_engine.state_sha256(state)


def materialize_structural_states(
    screen: engine.StructuralScreen,
    c12_row: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    c_state = replay_c12(screen, c12_row)
    f_state = exact_grover_state(c_state, screen.final_mask, P_FEASIBILITY)
    c_metrics = screen.metrics(c_state)
    f_metrics = screen.metrics(f_state)
    expected_c_mass = _slice_saved_angles(c12_row, screen.config.initialization_seeds, "final_mass")
    expected_f_mass = grover_success(expected_c_mass, P_FEASIBILITY)
    replay_error = float(
        np.max(np.abs(np.asarray(c_metrics["final_mass"]) - expected_c_mass))
    )
    grover_error = float(
        np.max(np.abs(np.asarray(f_metrics["final_mass"]) - expected_f_mass))
    )
    _require(replay_error <= 2e-5, f"C12 replay error {replay_error:.3e}")
    _require(grover_error <= 2e-5, f"p3 Grover error {grover_error:.3e}")
    record = {
        "schema": PRECURSOR_SCHEMA,
        "restart_seeds": list(screen.config.initialization_seeds),
        "dtype": screen.config.dtype,
        "stage1": {
            "method_id": COARSE_METHOD,
            "depth": P_C,
            "saved_angle_replay": True,
            "optimization_performed": False,
            "metrics": c_metrics,
            "state_sha256": state_sha256(c_state),
            "max_mass_replay_error": replay_error,
        },
        "stage2": {
            "method_id": PRECURSOR_METHOD,
            "depth": P_FEASIBILITY,
            "phase": "exact Q=0 and C=0 indicator",
            "phase_angles": [GROVER_FIXED_ANGLES] * P_FEASIBILITY,
            "reflection_angles": [GROVER_FIXED_ANGLES] * P_FEASIBILITY,
            "optimization_performed": False,
            "metrics": f_metrics,
            "analytic_expected_final_mass": expected_f_mass.tolist(),
            "state_sha256": state_sha256(f_state),
            "max_analytic_mass_error": grover_error,
        },
        "objective_table_loaded": False,
        "target_constructed": False,
    }
    return c_state, f_state, record


def coefficient_table(seed: int = OBJECTIVE_SEED) -> dict[str, object]:
    """Generate the single prespecified development table without applying it."""

    if int(seed) != OBJECTIVE_SEED:
        raise TargetedValidationError("the pilot accepts only objective seed 20260811")
    rng = np.random.default_rng(int(seed))
    assignment = rng.integers(1, 21, size=(engine.BLOCKS, engine.LABELS), dtype=np.int64)
    residual = rng.integers(
        5,
        31,
        size=(len(launcher.RESIDUAL_EDGES), len(launcher.SPECTRAL_PAIRS)),
        dtype=np.int64,
    )
    body = {
        "seed": int(seed),
        "assignment_cost": assignment.tolist(),
        "residual_interference_cost": residual.tolist(),
    }
    return {**body, "table_sha256": sha256_json(body)}


def objective_diagonal_raw(
    screen: engine.StructuralScreen, table: Mapping[str, object]
) -> np.ndarray:
    """Apply a frozen coefficient table to the full native shell, without ranking."""

    raw = {
        "seed": int(table["seed"]),
        "assignment_cost": table["assignment_cost"],
        "residual_interference_cost": table["residual_interference_cost"],
    }
    _require(table.get("table_sha256") == sha256_json(raw), "coefficient table changed")
    assignment = np.asarray(table["assignment_cost"], dtype=np.int64)
    residual = np.asarray(table["residual_interference_cost"], dtype=np.int64)
    _require(
        assignment.shape == (engine.BLOCKS, engine.LABELS),
        "assignment coefficient shape mismatch",
    )
    _require(
        residual.shape
        == (len(launcher.RESIDUAL_EDGES), len(launcher.SPECTRAL_PAIRS)),
        "residual coefficient shape mismatch",
    )
    masks = screen.masks_np
    objective = np.zeros(engine.DIMENSION, dtype=np.int64)
    for block in range(engine.BLOCKS):
        for label in range(engine.LABELS):
            objective += assignment[block, label] * ((masks[:, block] >> label) & 1)
    for edge_index, (left, right) in enumerate(launcher.RESIDUAL_EDGES):
        for pair_index, (a, b) in enumerate(launcher.SPECTRAL_PAIRS):
            active = ((masks[:, left] >> a) & 1) & ((masks[:, right] >> b) & 1)
            objective += residual[edge_index, pair_index] * active
    return objective


def objective_tensor(
    screen: engine.StructuralScreen, table: Mapping[str, object]
) -> torch.Tensor:
    raw = objective_diagonal_raw(screen, table)
    return torch.as_tensor(raw, dtype=screen.real_dtype, device=screen.device) / 800.0


def c12_circuit_ru() -> int:
    return engine.PREP_RU + P_C * (engine.C_NATIVE_RU + engine.XY_RU)


def feasibility_precursor_circuit_ru() -> int:
    return engine.StructuralScreen.history_ru(
        c12_circuit_ru(), P_FEASIBILITY, engine.QC_UNION_RU
    )


def method_primary_ru(method_id: str) -> int:
    c12 = c12_circuit_ru()
    if method_id == COARSE_METHOD:
        return c12 + launcher.T_O_RU
    if method_id == PRECURSOR_METHOD:
        return feasibility_precursor_circuit_ru() + launcher.T_O_RU
    if method_id == FULL_METHOD:
        circuit = launcher.history_ru(
            feasibility_precursor_circuit_ru(), launcher.O_RU, P_OBJECTIVE
        )
        return launcher.report_ru(circuit)
    if method_id == COLLAPSE_JOINT:
        circuit = launcher.history_ru(
            c12, launcher.phase_ru("QO_joint"), P_COLLAPSE
        )
        return launcher.report_ru(circuit)
    if method_id == COLLAPSE_SEPARATE:
        circuit = launcher.history_ru(
            c12, launcher.phase_ru("QO_separate"), P_COLLAPSE
        )
        return launcher.report_ru(circuit)
    if method_id == COARSE_REFINE:
        circuit = launcher.history_ru(c12, launcher.O_RU, P_OBJECTIVE)
        return launcher.report_ru(circuit)
    if method_id == FIXED_UNIFORM:
        fixed_precursor = engine.PREP_RU + 32 * (
            engine.QC_UNION_RU + 2 * engine.PREP_RU + engine.SELECTIVE_RU
        )
        circuit = launcher.history_ru(fixed_precursor, launcher.O_RU, P_OBJECTIVE)
        return launcher.report_ru(circuit)
    raise TargetedValidationError(f"unknown method {method_id}")


def method_specs() -> dict[str, MethodSpec]:
    definitions = (
        (FULL_METHOD, P_OBJECTIVE, ("O",), 4_800, 91_001, "learned_feasible"),
        (
            COLLAPSE_JOINT,
            P_COLLAPSE,
            ("QO_joint",),
            4_800,
            91_002,
            "learned_coarse",
        ),
        (
            COLLAPSE_SEPARATE,
            P_COLLAPSE,
            ("Q", "O"),
            4_800,
            91_003,
            "learned_coarse",
        ),
    )
    return {
        method_id: MethodSpec(
            method_id,
            depth,
            phases,
            evaluations,
            stream,
            reference,
            method_primary_ru(method_id),
        )
        for method_id, depth, phases, evaluations, stream, reference in definitions
    }


def history_energy_fn(
    screen: engine.StructuralScreen,
    reference: torch.Tensor,
    objective: torch.Tensor,
    spec: MethodSpec,
):
    cumulative = screen.common + objective
    if spec.phase_names == ("O",):
        phases = (objective,)
    elif spec.phase_names == ("QO_joint",):
        phases = (screen.q + objective,)
    elif spec.phase_names == ("Q", "O"):
        phases = (screen.q, objective)
    else:
        raise TargetedValidationError(f"unsupported phases {spec.phase_names}")

    def energy(gamma: torch.Tensor, beta: torch.Tensor):
        return objective_engine.history_circuit(
            screen,
            reference,
            phases,
            cumulative,
            gamma,
            beta,
            spec.depth,
        )

    return energy, len(phases) * spec.depth, spec.depth


def fixed_uniform_precursor(screen: engine.StructuralScreen) -> torch.Tensor:
    reference = screen.native[None, :].expand(screen.config.restarts, -1).clone()
    return exact_grover_state(reference, screen.final_mask, 32)


def reference_for_method(
    screen: engine.StructuralScreen,
    spec: MethodSpec,
    c_state: torch.Tensor,
    f_state: torch.Tensor,
) -> torch.Tensor:
    if spec.reference_kind == "learned_feasible":
        return f_state
    if spec.reference_kind == "learned_coarse":
        return c_state
    if spec.reference_kind == "fixed_uniform":
        return fixed_uniform_precursor(screen)
    raise TargetedValidationError(f"unknown reference {spec.reference_kind}")


def replay_optimized_state(
    screen: engine.StructuralScreen,
    spec: MethodSpec,
    reference: torch.Tensor,
    objective: torch.Tensor,
    gamma_values: Sequence[Sequence[float]],
    beta_values: Sequence[Sequence[float]],
) -> torch.Tensor:
    energy, gamma_count, beta_count = history_energy_fn(
        screen, reference, objective, spec
    )
    gamma = torch.as_tensor(
        gamma_values, dtype=screen.real_dtype, device=screen.device
    )
    beta = torch.as_tensor(beta_values, dtype=screen.real_dtype, device=screen.device)
    _require(
        gamma.shape == (screen.config.restarts, gamma_count),
        f"saved gamma shape mismatch for {spec.method_id}",
    )
    _require(
        beta.shape == (screen.config.restarts, beta_count),
        f"saved beta shape mismatch for {spec.method_id}",
    )
    with torch.no_grad():
        _, state = energy(gamma, beta)
    return state
