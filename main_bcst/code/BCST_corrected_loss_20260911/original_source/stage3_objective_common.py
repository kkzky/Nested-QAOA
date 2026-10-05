"""Shared, gate-checked machinery for the conditional BCST objective campaign.

This module intentionally contains no target construction, ranking, or target
membership code.  Objective diagonals can only be materialized through an
``ActivatedCampaign`` returned by :func:`activate_campaign`, which independently
revalidates both target-free gates and byte-compares the three frozen launcher
artifacts with a fresh reconstruction.

The optimizer and post-lock executables share this module so that saved-angle
replay uses exactly the same circuit implementation as optimization.  The
post-lock executable is the only module that is allowed to rank final-sector
states.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import conditional_stage3_objective_launcher as launch
import post_stage_a_structural_worker as structural_worker
import target_free_structural_screen as engine


DEVELOPMENT_TASK_SCHEMA = "nonredundant-bcst-stage3-development-tasks-v1"
OPTIMIZER_TASK_SCHEMA = "nonredundant-bcst-stage3-optimizer-task-v1"
OPTIMIZER_TABLE_SEAL_SCHEMA = "nonredundant-bcst-stage3-optimizer-table-seal-v1"

FROZEN_COEFFICIENTS = "frozen_objective_coefficients.json"
FROZEN_MATRIX = "frozen_objective_method_matrix.json"
FROZEN_CAMPAIGN = "frozen_objective_campaign.json"
FROZEN_DEVELOPMENT_TASKS = "frozen_development_tasks.json"
FROZEN_POST_DEVELOPMENT_TASKS = "frozen_post_development_tasks.json"


class ObjectiveActivationError(RuntimeError):
    """The final structural gates or a frozen campaign artifact failed closed."""


class ObjectiveExecutionError(RuntimeError):
    """A frozen objective task cannot be executed without protocol deviation."""


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ObjectiveActivationError(f"{path} is not a JSON object")
    return value


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".json-{os.getpid()}-{uuid.uuid4().hex[:16]}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_create_json(path: Path, payload: object) -> None:
    """Publish a complete JSON file atomically without replacing any file.

    The temporary file is fully written before a same-directory hard link
    makes it visible at ``path``.  Hard-link creation is atomic and fails when
    the destination already exists, unlike ``os.replace``.  This is the
    completion-preserving publication primitive for distributed task outputs.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".json-{os.getpid()}-{uuid.uuid4().hex[:16]}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_write_json(path: Path, payload: object) -> None:
    """Create ``path`` once, or accept an existing byte-equivalent JSON value."""

    if path.exists():
        if json_bytes(load_json(path)) != json_bytes(payload):
            raise ObjectiveExecutionError(f"immutable JSON already differs: {path}")
        return
    try:
        atomic_create_json(path, payload)
    except FileExistsError:
        # Another process may have won the create race.  It is reusable only
        # when it froze exactly the same value.
        if json_bytes(load_json(path)) != json_bytes(payload):
            raise ObjectiveExecutionError(f"immutable JSON create race differed: {path}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ObjectiveActivationError(message)


@dataclass(frozen=True)
class ActivatedCampaign:
    """Private capability produced only after both target-free gates validate."""

    token: launch._ActivatedStructuralHandoff
    campaign_root: Path
    coefficients: Mapping[str, object]
    matrix: Mapping[str, object]
    campaign: Mapping[str, object]
    stage_a_path: Path
    structural_handoff_path: Path

    @property
    def rows_by_id(self) -> dict[str, dict[str, object]]:
        rows = self.matrix.get("methods")
        if not isinstance(rows, list):
            raise ObjectiveActivationError("frozen method rows are absent")
        return {
            str(row["configuration_id"]): row
            for row in rows
            if isinstance(row, dict)
        }

    @property
    def tables_by_seed(self) -> dict[int, dict[str, object]]:
        tables = self.coefficients.get("tables")
        if not isinstance(tables, list):
            raise ObjectiveActivationError("frozen coefficient tables are absent")
        return {
            int(table["seed"]): table
            for table in tables
            if isinstance(table, dict)
        }


def _fresh_frozen_artifacts(
    token: launch._ActivatedStructuralHandoff,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    audited = json.loads(
        (Path(__file__).resolve().parent / "structural_audit.json").read_text(
            encoding="utf-8"
        )
    )["frozen_objective"]
    coefficients = launch.coefficient_manifest(token, audited)
    matrix = launch.build_method_matrix(token)
    campaign = launch.build_campaign_manifest(token, coefficients, matrix)
    return coefficients, matrix, campaign


def freeze_after_both_gates(
    stage_a_path: Path,
    structural_handoff_path: Path,
    campaign_root: Path,
) -> ActivatedCampaign:
    """Freeze objective tables/matrix after, and only after, both gates pass."""

    stage_a = load_json(stage_a_path)
    handoff = load_json(structural_handoff_path)
    token = launch.activate_structural_handoff(stage_a, handoff)
    coefficients, matrix, campaign = _fresh_frozen_artifacts(token)
    campaign_root.mkdir(parents=True, exist_ok=True)
    for name, value in (
        (FROZEN_COEFFICIENTS, coefficients),
        (FROZEN_MATRIX, matrix),
        (FROZEN_CAMPAIGN, campaign),
    ):
        destination = campaign_root / name
        if destination.exists():
            _require(
                json_bytes(load_json(destination)) == json_bytes(value),
                f"existing {name} differs",
            )
        else:
            atomic_write_json(destination, value)
    return ActivatedCampaign(
        token,
        campaign_root.resolve(),
        coefficients,
        matrix,
        campaign,
        stage_a_path.resolve(),
        structural_handoff_path.resolve(),
    )


def activate_campaign(
    stage_a_path: Path,
    structural_handoff_path: Path,
    campaign_root: Path,
) -> ActivatedCampaign:
    """Revalidate gates and exact frozen launcher artifacts for every worker."""

    stage_a = load_json(stage_a_path)
    handoff = load_json(structural_handoff_path)
    token = launch.activate_structural_handoff(stage_a, handoff)
    expected = _fresh_frozen_artifacts(token)
    paths = (
        campaign_root / FROZEN_COEFFICIENTS,
        campaign_root / FROZEN_MATRIX,
        campaign_root / FROZEN_CAMPAIGN,
    )
    for path in paths:
        _require(path.is_file(), f"required frozen artifact is absent: {path.name}")
    observed = tuple(load_json(path) for path in paths)
    for path, actual, fresh in zip(paths, observed, expected):
        _require(
            json_bytes(actual) == json_bytes(fresh),
            f"frozen artifact differs from reconstruction: {path.name}",
        )
    coefficients, matrix, campaign = observed
    required_capabilities = {
        "checkpointed_all_literal_ru_matched_depths",
        "digitized_and_analog_local_mixers",
        "four_full_guard_and_exact_final_mixers",
        "process_separated_optimizer_and_postlock",
        "development_heldout_and_19200_waves",
        "saved_angle_complex128_replay",
        "exact_and_heuristic_classical_controls",
    }
    capabilities = campaign.get("implemented_worker_capabilities")
    _require(
        campaign.get("launchable") is True
        and campaign.get("both_target_free_activation_gates_validated") is True
        and campaign.get("worker_implementation_contract")
        == "nonredundant-bcst-stage3-workers-v1"
        and campaign.get("implementation_blockers") == []
        and isinstance(capabilities, dict)
        and set(capabilities) == required_capabilities
        and all(capabilities.values()),
        "frozen campaign does not declare the implemented post-gate worker contract",
    )
    _require(
        campaign.get("objective_evaluated_on_state") is False
        and campaign.get("target_constructed") is False,
        "frozen campaign already claims objective/target access",
    )
    return ActivatedCampaign(
        token,
        campaign_root.resolve(),
        coefficients,
        matrix,
        campaign,
        stage_a_path.resolve(),
        structural_handoff_path.resolve(),
    )


def require_activation(value: object) -> ActivatedCampaign:
    if not isinstance(value, ActivatedCampaign):
        raise ObjectiveActivationError(
            "a freshly validated Stage-A plus final-structural activation is required"
        )
    return value


def objective_values_numpy(
    activation: ActivatedCampaign,
    objective_seed: int,
    native_shell_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Evaluate ``O_s`` only at the explicitly requested native-shell indices."""

    activation = require_activation(activation)
    table = activation.tables_by_seed.get(int(objective_seed))
    if table is None:
        raise ObjectiveExecutionError("objective seed is not in the frozen twelve-table set")
    raw = {
        "seed": int(table["seed"]),
        "assignment_cost": table["assignment_cost"],
        "residual_interference_cost": table["residual_interference_cost"],
    }
    _require(sha256_json(raw) == table.get("table_sha256"), "coefficient table changed")
    indices = np.asarray(native_shell_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= engine.DIMENSION):
        raise ObjectiveExecutionError("native-shell objective indices are invalid")
    digits = np.empty((indices.size, engine.BLOCKS), dtype=np.int16)
    remainder = indices.copy()
    for block in range(engine.BLOCKS - 1, -1, -1):
        digits[:, block] = remainder % engine.LOCAL_DIM
        remainder //= engine.LOCAL_DIM
    masks = engine.LOCAL_MASKS[digits]
    assignment = np.asarray(table["assignment_cost"], dtype=np.int64)
    residual = np.asarray(table["residual_interference_cost"], dtype=np.int64)
    if assignment.shape != (launch.BLOCKS, launch.LABELS):
        raise ObjectiveExecutionError("assignment coefficient shape mismatch")
    if residual.shape != (len(launch.RESIDUAL_EDGES), len(launch.SPECTRAL_PAIRS)):
        raise ObjectiveExecutionError("residual coefficient shape mismatch")
    objective = np.zeros(indices.size, dtype=np.int64)
    for block in range(launch.BLOCKS):
        for label in range(launch.LABELS):
            objective += assignment[block, label] * ((masks[:, block] >> label) & 1)
    for edge_index, (left, right) in enumerate(launch.RESIDUAL_EDGES):
        for pair_index, (a, b) in enumerate(launch.SPECTRAL_PAIRS):
            active = ((masks[:, left] >> a) & 1) & ((masks[:, right] >> b) & 1)
            objective += residual[edge_index, pair_index] * active
    return objective


def objective_diagonal_numpy(
    activation: ActivatedCampaign, objective_seed: int
) -> np.ndarray:
    """Evaluate the full optimizer diagonal, without any target/rank operation."""

    return objective_values_numpy(
        activation, objective_seed, np.arange(engine.DIMENSION, dtype=np.int64)
    )


def objective_tensor(
    activation: ActivatedCampaign,
    objective_seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = objective_diagonal_numpy(activation, objective_seed)
    return torch.as_tensor(raw, dtype=dtype, device=device) / 800.0


def deterministic_angles(
    objective_seed: int,
    method_id: str,
    configuration_id: str,
    restart_seed: int,
    parameter_block: str,
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Frozen SHA-256 stream derivation; independent for every angle block."""

    payload = [
        int(objective_seed),
        str(method_id),
        str(configuration_id),
        int(restart_seed),
        str(parameter_block),
    ]
    digest = hashlib.sha256(json_bytes(payload)).digest()
    seed = int.from_bytes(digest[:16], "big", signed=False)
    rng = np.random.default_rng(seed)
    values = rng.uniform(-math.pi, math.pi, size=(1, int(count)))
    return torch.as_tensor(values, dtype=dtype, device=device)


def state_sha256(state: torch.Tensor) -> str:
    array = state.detach().cpu().numpy()
    canonical = array.astype("<c8" if array.dtype == np.complex64 else "<c16", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def method_phase_plan(
    screen: engine.StructuralScreen,
    objective: torch.Tensor,
    name: str,
    penalty: float,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Return physical phase diagonals and the prespecified expected loss."""

    cumulative = float(penalty) * screen.common + objective
    if name == "Q":
        return (screen.q,), screen.q
    if name == "C":
        return (screen.c,), screen.c
    if name == "O":
        return (objective,), objective
    if name == "QC_joint":
        return (float(penalty) * screen.common,), screen.common
    if name == "QC_separate":
        return (float(penalty) * screen.q, float(penalty) * screen.c), screen.common
    if name == "CO_joint":
        return (float(penalty) * screen.c + objective,), cumulative
    if name == "CO_separate":
        return (float(penalty) * screen.c, objective), cumulative
    if name == "QO_joint":
        return (float(penalty) * screen.q + objective,), cumulative
    if name == "QO_separate":
        return (float(penalty) * screen.q, objective), cumulative
    if name == "QCO_joint":
        return (cumulative,), cumulative
    if name == "QCO_separate":
        return (
            float(penalty) * screen.q,
            float(penalty) * screen.c,
            objective,
        ), cumulative
    raise ObjectiveExecutionError(f"unsupported phase variant {name}")


def block_circuit(
    screen: engine.StructuralScreen,
    initial: torch.Tensor,
    phases: Sequence[torch.Tensor],
    loss: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return screen.block_energy(initial, phases, loss, gamma, beta, depth)


def history_circuit(
    screen: engine.StructuralScreen,
    reference: torch.Tensor,
    phases: Sequence[torch.Tensor],
    loss: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = reference.clone()
    if gamma.shape[1] != len(phases) * depth or beta.shape[1] != depth:
        raise ObjectiveExecutionError("history angle shape mismatch")

    def layer_step(
        current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor
    ) -> torch.Tensor:
        for phase_index, phase in enumerate(phases):
            current = current * torch.exp(
                -1j * layer_gamma[:, phase_index : phase_index + 1] * phase[None, :]
            ).to(screen.complex_dtype)
        current = engine.apply_history(current, layer_beta, reference)
        return engine.renormalize(current)

    for layer in range(depth):
        layer_gamma = torch.stack(
            [gamma[:, index * depth + layer] for index in range(len(phases))], dim=1
        )
        if screen.config.activation_checkpointing and torch.is_grad_enabled():
            state = checkpoint(
                layer_step,
                state,
                layer_gamma,
                beta[:, layer],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            state = layer_step(state, layer_gamma, beta[:, layer])
    return engine.expectation(state, loss), state


class PairTemplateMixer:
    """Fixed-order product formula for an explicitly supplied template set."""

    def __init__(
        self,
        templates: Sequence[tuple[torch.Tensor, torch.Tensor]],
        sweeps: int,
    ) -> None:
        if sweeps not in launch.LOCAL_SWEEPS:
            raise ObjectiveExecutionError("only the frozen 1/2/4 sweeps are permitted")
        if len(templates) != 75:
            raise ObjectiveExecutionError("local mixer must contain 75 templates")
        self.templates = tuple(templates)
        self.sweeps = int(sweeps)

    def apply(self, state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        scaled = beta / self.sweeps
        cosine = torch.cos(scaled).to(state.real.dtype)[:, None]
        sine = torch.sin(scaled).to(state.real.dtype)[:, None]
        for _ in range(self.sweeps):
            for left, right in self.templates:
                amp_left = torch.index_select(state, 1, left)
                amp_right = torch.index_select(state, 1, right)
                delta_left = (cosine - 1.0) * amp_left - 1j * sine * amp_right
                delta_right = (cosine - 1.0) * amp_right - 1j * sine * amp_left
                state = torch.index_add(
                    state,
                    1,
                    torch.cat((left, right)),
                    torch.cat((delta_left, delta_right), dim=1),
                )
        return state


def _external_neighbors(left: int, right: int) -> tuple[int, int]:
    left_external = next(
        value
        for value in ((left - 1) % engine.BLOCKS, (left + 1) % engine.BLOCKS)
        if value != right
    )
    right_external = next(
        value
        for value in ((right - 1) % engine.BLOCKS, (right + 1) % engine.BLOCKS)
        if value != left
    )
    return left_external, right_external


def final_switch_templates(
    screen: engine.StructuralScreen, *, full_guard: bool
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build the 75 exact-sector or four-full-guard final switch templates."""

    indices = np.arange(engine.DIMENSION, dtype=np.int64)
    templates: list[tuple[torch.Tensor, torch.Tensor]] = []
    for left, right in engine.CYCLE_EDGES:
        left_external, right_external = _external_neighbors(left, right)
        for a, b in engine.LOCAL_PAIRS:
            dl = screen.digits[:, left]
            dr = screen.digits[:, right]
            nl = engine.SWAP_TABLE[dl, a, b]
            nr = engine.SWAP_TABLE[dr, a, b]
            left_has_a = ((screen.masks_np[:, left] >> a) & 1).astype(bool)
            right_has_a = ((screen.masks_np[:, right] >> a) & 1).astype(bool)
            applicable = (nl >= 0) & (nr >= 0) & (left_has_a != right_has_a)
            for label in (a, b):
                if full_guard or engine.QUOTA[label] == 2:
                    bit = 1 << label
                    applicable &= (screen.masks_np[:, left_external] & bit) == 0
                    applicable &= (screen.masks_np[:, right_external] & bit) == 0
            changed = indices + (nl.astype(np.int64) - dl) * engine.STRIDES[left]
            changed += (nr.astype(np.int64) - dr) * engine.STRIDES[right]
            source = np.flatnonzero(applicable & (indices < changed)).astype(np.int64)
            partner = changed[source].astype(np.int64)
            if source.size == 0:
                raise ObjectiveExecutionError("empty final-switch template")
            templates.append(
                (
                    torch.from_numpy(source).to(screen.device),
                    torch.from_numpy(partner).to(screen.device),
                )
            )
    if len(templates) != 75:
        raise ObjectiveExecutionError("final-switch template count mismatch")
    return tuple(templates)


def template_circuit(
    screen: engine.StructuralScreen,
    reference: torch.Tensor,
    phases: Sequence[torch.Tensor],
    loss: torch.Tensor,
    mixer: object,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = reference.clone()
    if gamma.shape[1] != len(phases) * depth or beta.shape[1] != depth:
        raise ObjectiveExecutionError("local-mixer angle shape mismatch")

    def layer_step(
        current: torch.Tensor, layer_gamma: torch.Tensor, layer_beta: torch.Tensor
    ) -> torch.Tensor:
        for phase_index, phase in enumerate(phases):
            current = current * torch.exp(
                -1j * layer_gamma[:, phase_index : phase_index + 1] * phase[None, :]
            ).to(screen.complex_dtype)
        current = mixer.apply(current, layer_beta)
        return engine.renormalize(current)

    for layer in range(depth):
        layer_gamma = torch.stack(
            [gamma[:, index * depth + layer] for index in range(len(phases))], dim=1
        )
        if screen.config.activation_checkpointing and torch.is_grad_enabled():
            state = checkpoint(
                layer_step,
                state,
                layer_gamma,
                beta[:, layer],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            state = layer_step(state, layer_gamma, beta[:, layer])
    return engine.expectation(state, loss), state


class _SparseAnalogExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, beta: torch.Tensor, mixer: "SparseAnalogMixer"):
        with torch.no_grad():
            output, diagnostic = mixer.action(state, beta)
        ctx.mixer = mixer
        ctx.save_for_backward(output, beta)
        mixer.last_diagnostic = diagnostic
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, beta = ctx.saved_tensors
        mixer: SparseAnalogMixer = ctx.mixer
        with torch.no_grad():
            grad_state, _ = mixer.action(grad_output, -beta)
            derivative = -1j * mixer.matvec(output)
            grad_beta = torch.real(torch.sum(grad_output.conj() * derivative, dim=1))
        return grad_state, grad_beta, None


class SparseAnalogMixer:
    """Exact sparse-adjacency exponential evaluated by adaptive Lanczos."""

    def __init__(
        self,
        screen: engine.StructuralScreen,
        templates: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        tolerance: float | None = None,
        max_dimension: int = 256,
        check_interval: int = 16,
    ) -> None:
        rows = torch.cat([torch.cat((left, right)) for left, right in templates])
        columns = torch.cat([torch.cat((right, left)) for left, right in templates])
        values = torch.ones(rows.numel(), dtype=screen.real_dtype, device=screen.device)
        self.matrix = torch.sparse_coo_tensor(
            torch.stack((rows, columns)),
            values,
            (engine.DIMENSION, engine.DIMENSION),
            device=screen.device,
        ).coalesce().to_sparse_csr()
        self.tolerance = (
            float(tolerance)
            if tolerance is not None
            else (2e-6 if screen.complex_dtype == torch.complex64 else 1e-12)
        )
        self.max_dimension = int(max_dimension)
        self.check_interval = int(check_interval)
        self.last_diagnostic: dict[str, object] = {}

    def matvec(self, state: torch.Tensor) -> torch.Tensor:
        real = torch.sparse.mm(self.matrix, state.real.T).T
        imag = torch.sparse.mm(self.matrix, state.imag.T).T
        return torch.complex(real, imag).to(state.dtype)

    def action(
        self, state: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object]]:
        return structural_worker.krylov_expm_action(
            self.matvec,
            state,
            beta,
            tolerance=self.tolerance,
            max_dimension=self.max_dimension,
            check_interval=self.check_interval,
        )

    def apply(self, state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return _SparseAnalogExponential.apply(state, beta, self)


class ObjectiveScreen:
    """Objective-aware state engine constructed only from an activation token."""

    def __init__(
        self,
        activation: ActivatedCampaign,
        objective_seed: int,
        restart_seed: int,
        *,
        device: str,
        dtype: str = "complex64",
    ) -> None:
        self.activation = require_activation(activation)
        if restart_seed not in launch.EXPECTED_RESTART_SEEDS:
            raise ObjectiveExecutionError("restart seed is outside the frozen 16-seed set")
        config = replace(
            engine.valid_config(),
            restarts=1,
            initialization_seeds=(int(restart_seed),),
            restart_batch=1,
            device=device,
            dtype=dtype,
            activation_checkpointing=True,
        )
        self.screen = engine.StructuralScreen(config)
        self.objective_seed = int(objective_seed)
        self.restart_seed = int(restart_seed)
        self.objective = objective_tensor(
            activation,
            objective_seed,
            device=self.screen.device,
            dtype=self.screen.real_dtype,
        )
        self.cumulative = self.screen.common + self.objective
        self._mixers: dict[tuple[str, int | None], object] = {}

    def basis_state(self, sector: str) -> torch.Tensor:
        record = launch.CANONICAL_BASIS_STARTS[sector]
        index = int(record["native_shell_index"])
        actual_masks = tuple(int(value) for value in self.screen.masks_np[index])
        expected_masks = tuple(int(value) for value in record["bit_masks"])
        if actual_masks != expected_masks:
            raise ObjectiveExecutionError(f"canonical {sector} basis index/mask mismatch")
        if sector == "capacity" and not bool(self.screen.q_mask[index]):
            raise ObjectiveExecutionError("canonical capacity start is outside Q=0")
        if sector == "conflict" and not bool(self.screen.c_mask[index]):
            raise ObjectiveExecutionError("canonical conflict start is outside C=0")
        if sector == "exact-final" and not bool(self.screen.final_mask[index]):
            raise ObjectiveExecutionError("canonical exact-final start is infeasible")
        state = torch.zeros(
            (1, engine.DIMENSION),
            dtype=self.screen.complex_dtype,
            device=self.screen.device,
        )
        state[0, index] = 1.0
        return state

    def mixer(self, kind: str, sweeps: int | None, *, analog: bool) -> object:
        key = (f"analog:{kind}" if analog else kind, sweeps)
        if key in self._mixers:
            return self._mixers[key]
        if kind == "capacity":
            templates = tuple(self.screen.local_mixer("capacity_switch").templates)
        elif kind == "conflict":
            templates = tuple(self.screen.local_mixer("conflict_guarded").templates)
        elif kind == "exact-final":
            templates = final_switch_templates(self.screen, full_guard=False)
        elif kind == "full-guard":
            templates = final_switch_templates(self.screen, full_guard=True)
        else:
            raise ObjectiveExecutionError(f"unknown local mixer {kind}")
        value: object
        if analog:
            value = SparseAnalogMixer(self.screen, templates)
        else:
            if sweeps is None:
                raise ObjectiveExecutionError("digitized mixer lacks its frozen sweep count")
            value = PairTemplateMixer(templates, int(sweeps))
        self._mixers[key] = value
        return value

    def metrics(self, state: torch.Tensor) -> dict[str, float]:
        base = self.screen.metrics(state)
        with torch.no_grad():
            expected_o = engine.expectation(state, self.objective)
            expected_total = engine.expectation(state, self.cumulative)
        return {
            "state_norm": float(base["state_norm"][0]),
            "quota_mass": float(base["quota_mass"][0]),
            "conflict_mass": float(base["conflict_mass"][0]),
            "final_mass": float(base["final_mass"][0]),
            "expected_Qtilde": float(base["expected_Qtilde"][0]),
            "expected_Ctilde": float(base["expected_Ctilde"][0]),
            "expected_Otilde": float(expected_o[0].detach().cpu()),
            "expected_cumulative_loss": float(expected_total[0].detach().cpu()),
        }


def development_task_manifest(activation: ActivatedCampaign) -> dict[str, object]:
    """Expand every frozen optimizable configuration over 4x16 dev runs."""

    activation = require_activation(activation)
    tasks: list[dict[str, object]] = []
    excluded_families = {"ideal_static", "paired_precursor_snapshot"}
    for objective_seed in launch.DEVELOPMENT_SEEDS:
        for configuration_id, row in sorted(activation.rows_by_id.items()):
            if row.get("family") in excluded_families:
                continue
            evaluations = tuple(int(value) for value in row.get("forward_evaluations", ()))
            if not evaluations:
                continue
            for restart_seed in launch.EXPECTED_RESTART_SEEDS:
                tasks.append(
                    {
                        "task_id": (
                            f"development__o-{objective_seed}__{configuration_id}"
                            f"__r-{restart_seed}"
                        ),
                        "wave": "development_primary",
                        "objective_seed": int(objective_seed),
                        "method_group_id": row["method_group_id"],
                        "configuration_id": configuration_id,
                        "restart_seed": int(restart_seed),
                        "forward_evaluations": list(evaluations),
                        "gradient_updates": list(row["gradient_updates"]),
                        "endpoint_evaluations": list(row["endpoint_evaluations"]),
                        "target_available_to_worker": False,
                    }
                )
    expected_configurations = sum(
        1
        for row in activation.rows_by_id.values()
        if row.get("family") not in excluded_families and row.get("forward_evaluations")
    )
    expected_count = (
        len(launch.DEVELOPMENT_SEEDS)
        * expected_configurations
        * len(launch.EXPECTED_RESTART_SEEDS)
    )
    if len(tasks) != expected_count:
        raise AssertionError("development task matrix was truncated")
    body = {
        "schema": DEVELOPMENT_TASK_SCHEMA,
        "stage_a_manifest_id": activation.token.stage_a_manifest_id,
        "structural_handoff_id": activation.token.structural_handoff_id,
        "matrix_sha256": activation.matrix["matrix_sha256"],
        "coefficient_manifest_sha256": activation.coefficients["manifest_sha256"],
        "development_seeds": list(launch.DEVELOPMENT_SEEDS),
        "restart_seeds": list(launch.EXPECTED_RESTART_SEEDS),
        "configuration_count_per_table": expected_configurations,
        "task_count": len(tasks),
        "all_frozen_development_configurations_retained": True,
        "target_identity_exposed_to_optimizer_tasks": False,
        "tasks": tasks,
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def freeze_development_tasks(activation: ActivatedCampaign) -> dict[str, object]:
    manifest = development_task_manifest(activation)
    path = activation.campaign_root / FROZEN_DEVELOPMENT_TASKS
    if path.exists():
        _require(load_json(path) == manifest, "existing development task manifest differs")
    else:
        atomic_write_json(path, manifest)
    return manifest


def validate_task_manifest(
    activation: ActivatedCampaign,
    manifest: Mapping[str, object],
    *,
    post_development: bool,
) -> None:
    expected_schema = (
        launch.POST_DEVELOPMENT_SCHEMA if post_development else DEVELOPMENT_TASK_SCHEMA
    )
    _require(manifest.get("schema") == expected_schema, "task-manifest schema mismatch")
    _require(
        manifest.get("structural_handoff_id") == activation.token.structural_handoff_id,
        "task manifest belongs to another structural handoff",
    )
    _require(
        manifest.get("matrix_sha256") == activation.matrix.get("matrix_sha256"),
        "task manifest belongs to another method matrix",
    )
    key = "manifest_sha256"
    body = {name: value for name, value in manifest.items() if name != key}
    _require(manifest.get(key) == sha256_json(body), "task manifest content changed")
    if post_development:
        _require(
            manifest.get("target_identity_exposed_to_optimizer_tasks") is False,
            "post-development task manifest exposes targets",
        )
        selection_path = activation.campaign_root / "development_selection.json"
        _require(selection_path.is_file(), "sealed development selection is absent")
        selection = load_json(selection_path)
        selection_body = {
            name: value for name, value in selection.items() if name != "selection_sha256"
        }
        _require(
            selection.get("selection_sha256") == sha256_json(selection_body),
            "development selection changed",
        )
        fresh_post = launch.build_post_development_task_manifest(
            activation.token, activation.matrix, selection
        )
        _require(
            json_bytes(dict(manifest)) == json_bytes(fresh_post),
            "post-development tasks differ from the independently recomputed selection",
        )
    else:
        fresh = development_task_manifest(activation)
        _require(dict(manifest) == fresh, "development task manifest differs from frozen matrix")


def task_lookup(
    manifest: Mapping[str, object], task_id: str
) -> dict[str, object]:
    candidates: list[object]
    if manifest.get("schema") == DEVELOPMENT_TASK_SCHEMA:
        raw = manifest.get("tasks")
        candidates = raw if isinstance(raw, list) else []
    elif manifest.get("schema") == launch.POST_DEVELOPMENT_SCHEMA:
        held = manifest.get("held_out_tasks")
        budget = manifest.get("double_budget_tasks")
        candidates = (
            (held if isinstance(held, list) else [])
            + (budget if isinstance(budget, list) else [])
        )
    else:
        raise ObjectiveActivationError("unknown task manifest schema")
    matches = [item for item in candidates if isinstance(item, dict) and item.get("task_id") == task_id]
    if len(matches) != 1:
        raise ObjectiveExecutionError("task ID is absent or duplicated")
    task = matches[0]
    if task.get("target_available_to_worker", task.get("held_out_target_available_to_worker")) is not False:
        raise ObjectiveExecutionError("optimizer task does not explicitly exclude target access")
    return task


def optimizer_task_path(campaign_root: Path, task: Mapping[str, object]) -> Path:
    wave = str(task["wave"])
    seed = int(task["objective_seed"])
    return campaign_root / "optimizer_tasks" / wave / f"o-{seed}" / f"{task['task_id']}.json"


def make_screen(
    activation: ActivatedCampaign,
    objective_seed: int,
    restart_seed: int,
    *,
    device: str,
    dtype: str,
) -> ObjectiveScreen:
    return ObjectiveScreen(
        activation,
        objective_seed,
        restart_seed,
        device=device,
        dtype=dtype,
    )
