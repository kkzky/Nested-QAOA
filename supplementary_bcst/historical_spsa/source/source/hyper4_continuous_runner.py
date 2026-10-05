from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch

from ru_model import ACCOUNTING_VERSION, bcst_ru, direct_ru, warm_start_ru, xy_mixer_ru
from wdshcc_edge25_blockxy_nested import choose_device, write_csv


Term = Tuple[int, ...]


@dataclass(frozen=True)
class HeteroConfig:
    layout: Tuple[int, ...]
    weights: Tuple[int, ...]
    seed: int = 531
    device: str = "auto"
    dtype: str = "complex64"
    conflict_topology: str = "cycle"
    conflict_weight_scale: float = 1.0
    penalty_scale: float = 7.0
    nested_stage2_guard_scale: float = 7.0
    opt_restarts: int = 3
    opt_steps: int = 85
    opt_lr: float = 0.055
    optimizer_name: str = "adam"
    spsa_c: float = 0.10
    spsa_target_step: float = 0.10
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_stability_fraction: float = 0.10
    spsa_calibration_directions: int = 8
    spsa_directions_per_update: int = 1
    spsa_max_step_norm: float = 0.50
    early_stop_patience: int = 18
    early_stop_tol: float = 1e-6
    log_every: int = 30
    stage1_sequential_restarts: bool = False
    direct_sequential_restarts: bool = False
    stage2_sequential_restarts: bool = False
    stage2_sequential_min_depth: int = 1
    continuation_init: bool = False
    trace_every: int = 0
    target_prob: float = 0.99

    @property
    def n_qubits(self) -> int:
        return int(sum(self.layout))

    @property
    def n_blocks(self) -> int:
        return len(self.layout)


def parse_ints(text: str, default: Sequence[int] = ()) -> Tuple[int, ...]:
    text = (text or "").strip()
    if not text:
        return tuple(default)
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_configs(text: str, default: Sequence[Tuple[int, int]] = ()) -> Tuple[Tuple[int, int], ...]:
    text = (text or "").strip()
    if not text:
        return tuple(default)
    out: List[Tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sep = "-" if "-" in chunk else ":"
        a, b = chunk.split(sep)
        out.append((int(a), int(b)))
    return tuple(out)


def load_preload_params(path: str, seed: int) -> Tuple[Dict[int, dict], Dict[int, Dict[int, dict]]]:
    stage1: Dict[int, dict] = {}
    nested: Dict[int, Dict[int, dict]] = {}
    if not path:
        return stage1, nested
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) != seed:
                continue
            if row["algorithm"] == "XY-Stage1-Only":
                p1 = int(row["params"].split("=", 1)[1])
                stage1[p1] = {
                    "gammas": json.loads(row["stage1_gammas"]),
                    "betas": json.loads(row["stage1_betas"]),
                    "best_energy": float(row["stage1_energy"]),
                    "iters": int(float(row["iters"])),
                    "selected_restart": int(float(row.get("stage1_selected_restart", -1))),
                    "restart_energies": json.loads(row.get("stage1_restart_energies", "[]") or "[]"),
                }
            elif row["algorithm"] == "XY-Nested":
                p1_text, p2_text = row["params"].strip("()").split(",", 1)
                p1, p2 = int(p1_text), int(p2_text)
                nested.setdefault(p1, {})[p2] = {
                    "gammas": json.loads(row["stage2_gammas"]),
                    "betas": json.loads(row["stage2_betas"]),
                }
    if not stage1:
        raise ValueError(f"no seed-{seed} stage-1 rows found in preload CSV: {path}")
    if not nested:
        raise ValueError(f"no seed-{seed} nested rows found in preload CSV: {path}")
    return stage1, nested


def load_direct_preload_params(
    path: str,
    seed: int,
) -> Dict[str, Dict[int, Tuple[List[float], List[float]]]]:
    direct: Dict[str, Dict[int, Tuple[List[float], List[float]]]] = {
        "Std-XY": {},
        "d-XY": {},
    }
    if not path:
        return direct
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) != seed or row["algorithm"] not in direct:
                continue
            depth = int(row["params"].split("=", 1)[1])
            direct[row["algorithm"]][depth] = (
                json.loads(row["optimizer_gammas"]),
                json.loads(row["optimizer_betas"]),
            )
    if not any(direct.values()):
        raise ValueError(f"no seed-{seed} direct rows found in preload CSV: {path}")
    return direct


def default_weights(layout: Sequence[int]) -> Tuple[int, ...]:
    return tuple(min(2, int(k) - 1) for k in layout)


def rts(prob: float, target_prob: float = 0.99) -> float:
    if prob >= target_prob:
        return 1.0
    if prob <= 1e-14 or not math.isfinite(prob):
        return float("inf")
    return float(math.ceil(math.log(1.0 - target_prob) / math.log(1.0 - prob)))


class HeteroBlockWDSHCCInstance:
    """Exact product-of-cardinality-block WDS-HCC shell with unequal K blocks."""

    def __init__(self, cfg: HeteroConfig, device: torch.device, seed: int) -> None:
        if len(cfg.layout) != len(cfg.weights):
            raise ValueError("layout and weights must have the same length")
        for k, r in zip(cfg.layout, cfg.weights):
            if not (0 < int(r) < int(k)):
                raise ValueError(f"bad block weight r={r} for block size K={k}")
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.real_dtype = torch.float32 if cfg.dtype == "complex64" else torch.float64
        self.complex_dtype = torch.complex64 if cfg.dtype == "complex64" else torch.complex128

        self.N = cfg.n_qubits
        self.M = cfg.n_blocks
        self.block_sizes = tuple(int(x) for x in cfg.layout)
        self.block_weights = tuple(int(x) for x in cfg.weights)
        self.offsets = np.cumsum([0] + list(self.block_sizes[:-1])).astype(np.int32)
        self.qubit_to_block = np.zeros(self.N, dtype=np.int16)
        for b, (start, size) in enumerate(zip(self.offsets, self.block_sizes)):
            self.qubit_to_block[int(start):int(start + size)] = int(b)

        self.local_patterns = [self._build_local_patterns(k, r) for k, r in zip(self.block_sizes, self.block_weights)]
        self.Ls = tuple(int(p.shape[0]) for p in self.local_patterns)
        self.dim = int(np.prod(self.Ls, dtype=np.int64))
        if self.N > 64:
            raise ValueError("packed shell representation supports at most 64 qubits")
        self.state_bits_np = self._build_state_bits()

        self.path_conflict_edges = self._build_conflict_edges("path")
        self.cycle_conflict_edges = self._build_conflict_edges("cycle")
        self.conflict_edges = self._build_conflict_edges(cfg.conflict_topology)
        self.Cpath_np = self._build_conflict_cost_np(self.path_conflict_edges)
        self.Ccycle_np = self._build_conflict_cost_np(self.cycle_conflict_edges)
        self.Cconf_np = self._build_conflict_cost_np(self.conflict_edges)
        self.Cpath = torch.tensor(self.Cpath_np, dtype=self.real_dtype, device=self.device)
        self.Ccycle = torch.tensor(self.Ccycle_np, dtype=self.real_dtype, device=self.device)
        self.Cconf = torch.tensor(self.Cconf_np, dtype=self.real_dtype, device=self.device)

        self.feasible_mask_np = self.Cconf_np <= 1e-9
        self.path_feasible_mask_np = self.Cpath_np <= 1e-9
        self.cycle_feasible_mask_np = self.Ccycle_np <= 1e-9
        if int(self.feasible_mask_np.sum()) == 0:
            raise RuntimeError("no conflict-feasible block states found")
        self.feasible_state_bits_np = self.state_bits_np[self.feasible_mask_np]
        self.Cobj_np = np.zeros(self.dim, dtype=np.float32)
        self.Cobj = torch.tensor(self.Cobj_np, dtype=self.real_dtype, device=self.device)
        self.min_feasible_obj = 0.0
        self.opt_mask_np = self.feasible_mask_np.copy()
        self.feasible_mask = torch.tensor(self.feasible_mask_np, dtype=torch.bool, device=self.device)
        self.path_feasible_mask = torch.tensor(self.path_feasible_mask_np, dtype=torch.bool, device=self.device)
        self.cycle_feasible_mask = torch.tensor(self.cycle_feasible_mask_np, dtype=torch.bool, device=self.device)
        self.opt_mask = torch.tensor(self.opt_mask_np, dtype=torch.bool, device=self.device)

        self.psi_block = torch.ones(self.dim, dtype=self.complex_dtype, device=self.device) / math.sqrt(self.dim)
        self.psi_feasible = torch.zeros(self.dim, dtype=self.complex_dtype, device=self.device)
        self.psi_feasible[self.feasible_mask] = 1.0 / math.sqrt(self.feasible_count)
        self.local_eigs = self._build_local_xy_eigendecomps()

    @staticmethod
    def _build_local_patterns(k: int, r: int) -> np.ndarray:
        rows = []
        for comb in itertools.combinations(range(k), r):
            row = np.zeros(k, dtype=np.int8)
            row[list(comb)] = 1
            rows.append(row)
        return np.array(rows, dtype=np.int8)

    def _build_state_bits(self) -> np.ndarray:
        state_bits = np.zeros(self.dim, dtype=np.uint64)
        for block, patterns in enumerate(self.local_patterns):
            local_masks = np.zeros(patterns.shape[0], dtype=np.uint64)
            offset = int(self.offsets[block])
            for local in range(patterns.shape[1]):
                if np.any(patterns[:, local]):
                    local_masks |= patterns[:, local].astype(np.uint64) << np.uint64(offset + local)
            repeat_each = int(np.prod(self.Ls[block + 1 :], dtype=np.int64))
            tile_count = self.dim // (patterns.shape[0] * repeat_each)
            state_bits |= np.tile(np.repeat(local_masks, repeat_each), tile_count)
        return state_bits

    def _qubit(self, block: int, local: int) -> int:
        return int(self.offsets[block]) + int(local)

    def _build_conflict_edges(self, topology: str) -> List[Tuple[int, int, float]]:
        edges: Dict[Tuple[int, int], float] = {}
        if topology == "none":
            return []
        if topology not in ("cycle", "path"):
            raise ValueError(f"unknown conflict topology {topology!r}")
        block_range = range(self.M) if topology == "cycle" else range(self.M - 1)
        for b in block_range:
            nb = (b + 1) % self.M
            if topology == "path" and nb == 0:
                continue
            for local in range(min(self.block_sizes[b], self.block_sizes[nb])):
                u = self._qubit(b, local)
                v = self._qubit(nb, local)
                edges[tuple(sorted((u, v)))] = float(self.cfg.conflict_weight_scale)
        return [(u, v, w) for (u, v), w in sorted(edges.items())]

    def _build_conflict_cost_np(self, edges: Sequence[Tuple[int, int, float]]) -> np.ndarray:
        c = np.zeros(self.dim, dtype=np.float32)
        for u, v, w in edges:
            mask = (np.uint64(1) << np.uint64(u)) | (np.uint64(1) << np.uint64(v))
            c[(self.state_bits_np & mask) == mask] += np.float32(w)
        return c

    def _build_local_xy_eigendecomps(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for patterns in self.local_patterns:
            ldim = patterns.shape[0]
            h = np.zeros((ldim, ldim), dtype=np.float32)
            for a, pa in enumerate(patterns):
                for b, pb in enumerate(patterns):
                    if a < b and int(np.sum(np.abs(pa - pb))) == 2:
                        h[a, b] = 1.0
                        h[b, a] = 1.0
            ht = torch.tensor(h, dtype=self.real_dtype, device=self.device)
            evals, evecs = torch.linalg.eigh(ht)
            out.append((evals, evecs.to(self.complex_dtype)))
        return out

    @property
    def conflict_ru(self) -> int:
        return len(self.conflict_edges)

    @property
    def mcz_ru(self) -> int:
        return self.N * self.N

    @property
    def feasible_count(self) -> int:
        return int(self.feasible_mask_np.sum())

    @property
    def optimum_count(self) -> int:
        return int(self.opt_mask_np.sum())

    def block_for_qubit(self, q: int) -> int:
        return int(self.qubit_to_block[int(q)])

    def local_index(self, q: int) -> int:
        b = self.block_for_qubit(q)
        return int(q) - int(self.offsets[b])

    def summary(self) -> dict:
        return {
            "N": self.N,
            "M": self.M,
            "layout": ",".join(str(x) for x in self.block_sizes),
            "weights": ",".join(str(x) for x in self.block_weights),
            "local_basis_sizes": ",".join(str(x) for x in self.Ls),
            "subspace_dim": self.dim,
            "full_hilbert_dim": 2 ** self.N,
            "compression_factor": float((2 ** self.N) / self.dim),
            "conflict_topology": self.cfg.conflict_topology,
            "conflict_edges": self.conflict_ru,
            "conflict_RU": self.conflict_ru,
            "MCZ_RU": self.mcz_ru,
            "feasible_count": self.feasible_count,
            "feasible_fraction": self.feasible_count / self.dim,
            "path_feasible_count": int(self.path_feasible_mask_np.sum()),
            "cycle_feasible_count": int(self.cycle_feasible_mask_np.sum()),
            "optimum_count": self.optimum_count,
            "min_feasible_obj": self.min_feasible_obj,
        }


class HeteroQAOARunner:
    def __init__(self, cfg: HeteroConfig, inst: HeteroBlockWDSHCCInstance) -> None:
        self.cfg = cfg
        self.inst = inst
        self.device = inst.device
        self.real_dtype = inst.real_dtype
        self.complex_dtype = inst.complex_dtype
        self.penalty = cfg.penalty_scale
        self._stage1_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, int, dict]] = {}
        self._stage1_param_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._stage2_param_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._warm_param_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._direct_param_cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {
            "Std-XY": {},
            "d-XY": {},
        }
        self.trace_rows: List[dict] = []

    @staticmethod
    def _stable_label_seed(label: str) -> int:
        return sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % 1000003

    @staticmethod
    def _resize_angles(values: torch.Tensor, new_depth: int) -> torch.Tensor:
        old_depth = int(values.numel())
        if old_depth == new_depth:
            return values.detach().clone()
        if old_depth == 1:
            return values.detach().expand(new_depth).clone()
        source = torch.linspace(0.0, 1.0, old_depth, dtype=values.dtype, device=values.device)
        target = torch.linspace(0.0, 1.0, new_depth, dtype=values.dtype, device=values.device)
        right = torch.searchsorted(source, target, right=False).clamp(1, old_depth - 1)
        left = right - 1
        weight = (target - source[left]) / (source[right] - source[left])
        return values[left] + weight * (values[right] - values[left])

    def _continuation_initializers(
        self,
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        depth: int,
        split_gamma: bool = False,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], str]:
        if not self.cfg.continuation_init:
            return (), "independent"

        initializers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        sources: List[str] = []
        exact = cache.get(depth)

        lower = [candidate for candidate in cache if candidate < depth]
        if not lower:
            if exact is not None:
                exact_gamma, exact_beta = exact
                initializers.append((exact_gamma.detach().clone(), exact_beta.detach().clone()))
                sources.append(f"exact_p={depth}")
            return tuple(initializers), "+".join(sources) if sources else "independent"
        source_depth = max(lower)
        source_gamma, source_beta = cache[source_depth]
        extra = depth - source_depth
        if split_gamma:
            gamma_identity = torch.cat(
                (
                    source_gamma[:source_depth].detach(),
                    torch.zeros(extra, dtype=source_gamma.dtype, device=source_gamma.device),
                    source_gamma[source_depth:].detach(),
                    torch.zeros(extra, dtype=source_gamma.dtype, device=source_gamma.device),
                )
            )
        else:
            gamma_identity = torch.cat(
                (
                    source_gamma.detach(),
                    torch.zeros(extra, dtype=source_gamma.dtype, device=source_gamma.device),
                )
            )
        beta_identity = torch.cat(
            (
                source_beta.detach(),
                torch.zeros(extra, dtype=source_beta.dtype, device=source_beta.device),
                )
        )
        # The paper-facing protocol allows only one of five starts to inherit
        # the preceding shallower solution.  Use the identity-preserving
        # append so that this one start exactly retains the shallower
        # incumbent, leaving the deterministic ramp and three random starts
        # independent.
        initializers.append((gamma_identity, beta_identity))
        sources.append(f"identity_preserving_append_p={source_depth}")
        if exact is not None:
            exact_gamma, exact_beta = exact
            initializers.append((exact_gamma.detach().clone(), exact_beta.detach().clone()))
            sources.append(f"exact_p={depth}")
        return tuple(initializers), "+".join(sources)

    def _identity_initializer(
        self,
        p_gamma: int,
        p_beta: int,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Return an exact no-op circuit so a supplied state is never lost."""
        return (
            (
                torch.zeros(
                    p_gamma,
                    dtype=self.real_dtype,
                    device=self.device,
                ),
                torch.zeros(
                    p_beta,
                    dtype=self.real_dtype,
                    device=self.device,
                ),
            ),
        )

    def apply_xy_mixer(self, psi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        batch = psi.shape[0]
        shaped = psi.reshape([batch] + list(self.inst.Ls))
        letters = "cdefghijklmnopqrstuvwxyz"
        if self.inst.M > len(letters):
            raise ValueError("too many blocks for einsum labels")
        dims = list(letters[: self.inst.M])
        for block, (evals, evecs) in enumerate(self.inst.local_eigs):
            phases = torch.exp(-1j * beta[:, None].to(self.real_dtype) * evals[None, :]).to(self.complex_dtype)
            u_local = evecs[None, :, :] @ torch.diag_embed(phases) @ evecs.T.conj()[None, :, :]
            in_dims = dims.copy()
            out_dims = dims.copy()
            in_dims[block] = "y"
            out_dims[block] = "x"
            shaped = torch.einsum(f"bxy,b{''.join(in_dims)}->b{''.join(out_dims)}", u_local, shaped)
        return shaped.reshape(batch, self.inst.dim)

    def apply_history_mixer(self, psi: torch.Tensor, beta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        phase = torch.exp(-1j * beta).to(self.complex_dtype)[:, None]
        overlap = torch.sum(phi.conj()[None, :] * psi, dim=1, keepdim=True)
        return phase * psi + (1.0 - phase) * overlap * phi[None, :]

    def energy_xy(
        self,
        h_cost: torch.Tensor,
        gammas: torch.Tensor,
        betas: torch.Tensor,
        p: int,
        split: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = gammas.shape[0]
        phi = self.inst.psi_block if initial_state is None else initial_state
        phi = phi / torch.clamp(torch.linalg.vector_norm(phi), min=1e-20)
        psi = phi[None, :].expand(batch, -1).clone()
        if split is None:
            for d in range(p):
                psi = psi * torch.exp(-1j * gammas[:, d : d + 1].to(self.real_dtype) * h_cost[None, :]).to(self.complex_dtype)
                psi = self.apply_xy_mixer(psi, betas[:, d])
            h_eval = h_cost
        else:
            h_pen, h_obj = split
            for d in range(p):
                psi = psi * torch.exp(-1j * gammas[:, d : d + 1].to(self.real_dtype) * h_pen[None, :]).to(self.complex_dtype)
                psi = psi * torch.exp(-1j * gammas[:, p + d : p + d + 1].to(self.real_dtype) * h_obj[None, :]).to(self.complex_dtype)
                psi = self.apply_xy_mixer(psi, betas[:, d])
            h_eval = h_pen + h_obj
        norm_sq = torch.sum(torch.abs(psi) ** 2, dim=1)
        energy = torch.real(torch.sum(psi.conj() * h_eval[None, :] * psi, dim=1)) / torch.clamp(
            norm_sq, min=1e-20
        )
        return energy, psi

    def energy_history(
        self,
        phi: torch.Tensor,
        h_cost: torch.Tensor,
        gammas: torch.Tensor,
        betas: torch.Tensor,
        p: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = gammas.shape[0]
        phi = phi / torch.clamp(torch.linalg.vector_norm(phi), min=1e-20)
        psi = phi[None, :].expand(batch, -1).clone()
        for d in range(p):
            psi = psi * torch.exp(-1j * gammas[:, d : d + 1].to(self.real_dtype) * h_cost[None, :]).to(self.complex_dtype)
            psi = self.apply_history_mixer(psi, betas[:, d], phi)
        norm_sq = torch.sum(torch.abs(psi) ** 2, dim=1)
        energy = torch.real(torch.sum(psi.conj() * h_cost[None, :] * psi, dim=1)) / torch.clamp(
            norm_sq, min=1e-20
        )
        return energy, psi

    def optimize(
        self,
        energy_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        p_gamma: int,
        p_beta: int,
        label: str,
        trial_idx: int,
        initializers: Sequence[Tuple[torch.Tensor, torch.Tensor]] = (),
    ) -> Tuple[torch.Tensor, int, dict]:
        if self.cfg.optimizer_name == "spsa":
            return self.optimize_spsa_sequential_restarts(
                energy_fn,
                p_gamma,
                p_beta,
                label,
                trial_idx,
                initializers,
            )
        if self.cfg.optimizer_name != "adam":
            raise ValueError(f"unsupported optimizer {self.cfg.optimizer_name!r}")
        seed = int(self.inst.seed * 100000 + self._stable_label_seed(label))
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        g = torch.empty((self.cfg.opt_restarts, p_gamma), dtype=self.real_dtype, device=self.device).uniform_(-math.pi, math.pi)
        b = torch.empty((self.cfg.opt_restarts, p_beta), dtype=self.real_dtype, device=self.device).uniform_(-math.pi, math.pi)
        if self.cfg.opt_restarts > 0:
            g[0] = torch.linspace(0.05, 0.65, p_gamma, dtype=self.real_dtype, device=self.device)
            b[0] = torch.linspace(0.55, 0.05, p_beta, dtype=self.real_dtype, device=self.device)
        initializer_slots = max(0, self.cfg.opt_restarts - 1)
        usable_initializers = list(initializers)[-initializer_slots:] if initializer_slots else []
        initializer_start = max(1, self.cfg.opt_restarts - len(usable_initializers))
        for restart, (g_init, b_init) in enumerate(usable_initializers, start=initializer_start):
            if g_init.numel() != p_gamma or b_init.numel() != p_beta:
                raise ValueError(f"initializer shape mismatch for {label}")
            g[restart] = g_init.to(device=self.device, dtype=self.real_dtype)
            b[restart] = b_init.to(device=self.device, dtype=self.real_dtype)
        g.requires_grad_(True)
        b.requires_grad_(True)
        opt = torch.optim.Adam([g, b], lr=self.cfg.opt_lr)
        restart_best_energies = torch.full(
            (self.cfg.opt_restarts,), float("inf"), dtype=self.real_dtype, device=self.device
        )
        restart_best_g = g.detach().clone()
        restart_best_b = b.detach().clone()
        restart_best_steps = [0] * self.cfg.opt_restarts
        initializer_roles = [
            (
                "deterministic_linear_ramp"
                if restart == 0
                else "continuation"
                if restart >= initializer_start
                else "random_shared_stream"
            )
            for restart in range(self.cfg.opt_restarts)
        ]
        best_energy = float("inf")
        best_step = 0
        stale = 0
        t0 = time.time()
        step = 0
        for step in range(self.cfg.opt_steps):
            opt.zero_grad(set_to_none=True)
            energies, states = energy_fn(g, b)
            trace_this_step = self.cfg.trace_every > 0 and (
                step == 0 or (step + 1) % self.cfg.trace_every == 0
            )
            trace_row: Optional[dict] = None
            with torch.no_grad():
                improved = energies.detach() < restart_best_energies
                restart_best_energies[improved] = energies.detach()[improved]
                restart_best_g[improved] = g.detach()[improved]
                restart_best_b[improved] = b.detach()[improved]
                for restart in torch.nonzero(improved, as_tuple=False).reshape(-1).detach().cpu().tolist():
                    restart_best_steps[int(restart)] = step + 1
                cur = float(torch.min(energies).detach().cpu())
                if cur < best_energy - self.cfg.early_stop_tol:
                    best_energy = cur
                    best_step = step + 1
                    stale = 0
                else:
                    stale += 1
                log_this_step = self.cfg.log_every > 0 and (
                    step == 0 or (step + 1) % self.cfg.log_every == 0
                )
                if log_this_step or trace_this_step:
                    idx = int(torch.argmin(energies).detach().cpu())
                    state = states[idx]
                    state_prob = torch.abs(state) ** 2
                    state_norm = torch.clamp(torch.sum(state_prob), min=1e-20)
                    opt_mass = float((torch.sum(state_prob[self.inst.opt_mask]) / state_norm).detach().cpu())
                    feas_mass = float((torch.sum(state_prob[self.inst.feasible_mask]) / state_norm).detach().cpu())
                    if log_this_step:
                        print(
                            f"    {label} step={step + 1:>3}/{self.cfg.opt_steps} "
                            f"E={cur: .6g} feas={100 * feas_mass:5.1f}% opt={100 * opt_mass:7.4f}%",
                            flush=True,
                        )
                    if trace_this_step:
                        detached_energies = energies.detach()
                        trace_row = {
                            "seed": self.inst.seed,
                            "label": label,
                            "trial": trial_idx,
                            "optimizer_execution": "batched",
                            "step": step + 1,
                            "max_steps": self.cfg.opt_steps,
                            "restart_count": self.cfg.opt_restarts,
                            "parameter_count": int(g.numel() + b.numel()),
                            "current_best_restart": idx,
                            "energy_min": cur,
                            "energy_mean": float(torch.mean(detached_energies).cpu()),
                            "energy_variance": float(torch.var(detached_energies, unbiased=False).cpu()),
                            "loss_sum": float(torch.sum(detached_energies).cpu()),
                            "feasible_mass": feas_mass,
                            "target_mass": opt_mass,
                            "stale_steps": stale,
                            "elapsed_sec": time.time() - t0,
                            "restart_energies": json.dumps(
                                [float(value) for value in detached_energies.cpu().tolist()]
                            ),
                        }
            loss = energies.sum()
            loss.backward()
            if trace_row is not None:
                with torch.no_grad():
                    grad = torch.cat((g.grad.reshape(-1), b.grad.reshape(-1)))
                    trace_row.update(
                        {
                            "gradient_l2": float(torch.linalg.vector_norm(grad).cpu()),
                            "gradient_abs_mean": float(torch.mean(torch.abs(grad)).cpu()),
                            "gradient_variance": float(torch.var(grad, unbiased=False).cpu()),
                            "gamma_gradient_variance": float(
                                torch.var(g.grad.reshape(-1), unbiased=False).cpu()
                            ),
                            "beta_gradient_variance": float(
                                torch.var(b.grad.reshape(-1), unbiased=False).cpu()
                            ),
                        }
                    )
                    self.trace_rows.append(trace_row)
            opt.step()
            if stale >= self.cfg.early_stop_patience:
                break
        with torch.no_grad():
            final_energies, _final_states = energy_fn(g, b)
            improved = final_energies.detach() < restart_best_energies
            restart_best_energies[improved] = final_energies.detach()[improved]
            restart_best_g[improved] = g.detach()[improved]
            restart_best_b[improved] = b.detach()[improved]
            for restart in torch.nonzero(improved, as_tuple=False).reshape(-1).detach().cpu().tolist():
                restart_best_steps[int(restart)] = step + 1
            del final_energies, _final_states
            idx = int(torch.argmin(restart_best_energies).detach().cpu())
            selected_gamma = restart_best_g[idx].detach().clone()
            selected_beta = restart_best_b[idx].detach().clone()
            selected_energy, selected_state = energy_fn(selected_gamma[None, :], selected_beta[None, :])
            best = selected_state[0].detach().clone()
            best /= torch.clamp(torch.linalg.vector_norm(best), min=1e-20)
            final_energy = float(selected_energy[0].detach().cpu())
            restart_energies = [float(value) for value in restart_best_energies.detach().cpu().tolist()]
        return best, step + 1, {
            "best_energy": final_energy,
            "best_step": best_step,
            "elapsed_sec": time.time() - t0,
            "selected_restart": idx,
            "restart_energies": restart_energies,
            "restart_gammas": restart_best_g.detach().cpu().tolist(),
            "restart_betas": restart_best_b.detach().cpu().tolist(),
            "restart_best_steps": restart_best_steps,
            "restart_initializer_roles": initializer_roles,
            "restart_seed": seed,
            "stopping_reason": (
                "early_stop_patience" if stale >= self.cfg.early_stop_patience else "max_steps"
            ),
            "gammas": selected_gamma,
            "betas": selected_beta,
        }

    def optimize_sequential_restarts(
        self,
        energy_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        p_gamma: int,
        p_beta: int,
        label: str,
        trial_idx: int,
        initializers: Sequence[Tuple[torch.Tensor, torch.Tensor]] = (),
    ) -> Tuple[torch.Tensor, int, dict]:
        """Run the same seeded restarts one at a time to bound autograd memory."""
        if self.cfg.optimizer_name == "spsa":
            return self.optimize_spsa_sequential_restarts(
                energy_fn,
                p_gamma,
                p_beta,
                label,
                trial_idx,
                initializers,
            )
        if self.cfg.optimizer_name != "adam":
            raise ValueError(f"unsupported optimizer {self.cfg.optimizer_name!r}")
        seed = int(self.inst.seed * 100000 + self._stable_label_seed(label))
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        g_init = torch.empty(
            (self.cfg.opt_restarts, p_gamma), dtype=self.real_dtype, device=self.device
        ).uniform_(-math.pi, math.pi)
        b_init = torch.empty(
            (self.cfg.opt_restarts, p_beta), dtype=self.real_dtype, device=self.device
        ).uniform_(-math.pi, math.pi)
        if self.cfg.opt_restarts > 0:
            g_init[0] = torch.linspace(0.05, 0.65, p_gamma, dtype=self.real_dtype, device=self.device)
            b_init[0] = torch.linspace(0.55, 0.05, p_beta, dtype=self.real_dtype, device=self.device)
        initializer_slots = max(0, self.cfg.opt_restarts - 1)
        usable_initializers = list(initializers)[-initializer_slots:] if initializer_slots else []
        initializer_start = max(1, self.cfg.opt_restarts - len(usable_initializers))
        for restart, (g_value, b_value) in enumerate(usable_initializers, start=initializer_start):
            if g_value.numel() != p_gamma or b_value.numel() != p_beta:
                raise ValueError(f"initializer shape mismatch for {label}")
            g_init[restart] = g_value.to(device=self.device, dtype=self.real_dtype)
            b_init[restart] = b_value.to(device=self.device, dtype=self.real_dtype)

        gammas = []
        betas = []
        optimizers = []
        restart_best_energies = [float("inf")] * self.cfg.opt_restarts
        restart_best_g: List[torch.Tensor] = []
        restart_best_b: List[torch.Tensor] = []
        restart_best_steps = [0] * self.cfg.opt_restarts
        initializer_roles = [
            (
                "deterministic_linear_ramp"
                if restart == 0
                else "continuation"
                if restart >= initializer_start
                else "random_shared_stream"
            )
            for restart in range(self.cfg.opt_restarts)
        ]
        for restart in range(self.cfg.opt_restarts):
            g = g_init[restart : restart + 1].clone().detach().requires_grad_(True)
            b = b_init[restart : restart + 1].clone().detach().requires_grad_(True)
            gammas.append(g)
            betas.append(b)
            optimizers.append(torch.optim.Adam([g, b], lr=self.cfg.opt_lr))
            restart_best_g.append(g[0].detach().clone())
            restart_best_b.append(b[0].detach().clone())
        del g_init, b_init

        best_energy = float("inf")
        best_step = 0
        stale = 0
        t0 = time.time()
        step = 0
        for step in range(self.cfg.opt_steps):
            cur = float("inf")
            cur_state: Optional[torch.Tensor] = None
            log_this_step = self.cfg.log_every > 0 and (step == 0 or (step + 1) % self.cfg.log_every == 0)
            trace_this_step = self.cfg.trace_every > 0 and (
                step == 0 or (step + 1) % self.cfg.trace_every == 0
            )
            report_this_step = log_this_step or trace_this_step
            trace_energies: List[float] = []
            trace_gradients: List[torch.Tensor] = []
            trace_gamma_gradients: List[torch.Tensor] = []
            trace_beta_gradients: List[torch.Tensor] = []
            for restart, (g, b, opt) in enumerate(zip(gammas, betas, optimizers)):
                opt.zero_grad(set_to_none=True)
                energies, states = energy_fn(g, b)
                energy = float(energies[0].detach().cpu())
                if trace_this_step:
                    trace_energies.append(energy)
                if energy < restart_best_energies[restart]:
                    restart_best_energies[restart] = energy
                    restart_best_g[restart] = g[0].detach().clone()
                    restart_best_b[restart] = b[0].detach().clone()
                    restart_best_steps[restart] = step + 1
                if energy < cur:
                    cur = energy
                    if report_this_step:
                        cur_state = states[0].detach().clone()
                loss = energies.sum()
                loss.backward()
                if trace_this_step:
                    gamma_gradient = g.grad.detach().reshape(-1).clone()
                    beta_gradient = b.grad.detach().reshape(-1).clone()
                    trace_gamma_gradients.append(gamma_gradient)
                    trace_beta_gradients.append(beta_gradient)
                    trace_gradients.append(torch.cat((gamma_gradient, beta_gradient)))
                opt.step()
                del loss, energies, states

            with torch.no_grad():
                if cur < best_energy - self.cfg.early_stop_tol:
                    best_energy = cur
                    best_step = step + 1
                    stale = 0
                else:
                    stale += 1
                if report_this_step:
                    assert cur_state is not None
                    state_prob = torch.abs(cur_state) ** 2
                    state_norm = torch.clamp(torch.sum(state_prob), min=1e-20)
                    opt_mass = float((torch.sum(state_prob[self.inst.opt_mask]) / state_norm).detach().cpu())
                    feas_mass = float((torch.sum(state_prob[self.inst.feasible_mask]) / state_norm).detach().cpu())
                    if log_this_step:
                        print(
                            f"    {label} step={step + 1:>3}/{self.cfg.opt_steps} "
                            f"E={cur: .6g} feas={100 * feas_mass:5.1f}% opt={100 * opt_mass:7.4f}%",
                            flush=True,
                        )
                    if trace_this_step:
                        all_gradients = torch.cat(trace_gradients)
                        all_gamma_gradients = torch.cat(trace_gamma_gradients)
                        all_beta_gradients = torch.cat(trace_beta_gradients)
                        energy_values = np.asarray(trace_energies, dtype=np.float64)
                        self.trace_rows.append(
                            {
                                "seed": self.inst.seed,
                                "label": label,
                                "trial": trial_idx,
                                "optimizer_execution": "sequential",
                                "step": step + 1,
                                "max_steps": self.cfg.opt_steps,
                                "restart_count": self.cfg.opt_restarts,
                                "parameter_count": self.cfg.opt_restarts * (p_gamma + p_beta),
                                "active_parameter_count_per_restart": p_gamma + p_beta,
                                "current_best_restart": int(np.argmin(energy_values)),
                                "energy_min": cur,
                                "energy_mean": float(np.mean(energy_values)),
                                "energy_variance": float(np.var(energy_values)),
                                "loss_sum": float(np.sum(energy_values)),
                                "feasible_mass": feas_mass,
                                "target_mass": opt_mass,
                                "stale_steps": stale,
                                "elapsed_sec": time.time() - t0,
                                "gradient_l2": float(torch.linalg.vector_norm(all_gradients).cpu()),
                                "gradient_abs_mean": float(torch.mean(torch.abs(all_gradients)).cpu()),
                                "gradient_variance": float(
                                    torch.var(all_gradients, unbiased=False).cpu()
                                ),
                                "gamma_gradient_variance": float(
                                    torch.var(all_gamma_gradients, unbiased=False).cpu()
                                ),
                                "beta_gradient_variance": float(
                                    torch.var(all_beta_gradients, unbiased=False).cpu()
                                ),
                                "restart_energies": json.dumps(trace_energies),
                                "restart_gradient_l2": json.dumps(
                                    [
                                        float(torch.linalg.vector_norm(gradient).cpu())
                                        for gradient in trace_gradients
                                    ]
                                ),
                                "restart_gradient_abs_mean": json.dumps(
                                    [
                                        float(torch.mean(torch.abs(gradient)).cpu())
                                        for gradient in trace_gradients
                                    ]
                                ),
                                "restart_gradient_variance": json.dumps(
                                    [
                                        float(torch.var(gradient, unbiased=False).cpu())
                                        for gradient in trace_gradients
                                    ]
                                ),
                                "restart_initializer_roles": json.dumps(initializer_roles),
                            }
                        )
            del cur_state
            if stale >= self.cfg.early_stop_patience:
                break

        with torch.no_grad():
            for restart, (g, b) in enumerate(zip(gammas, betas)):
                energies, _states = energy_fn(g, b)
                energy = float(energies[0].detach().cpu())
                if energy < restart_best_energies[restart]:
                    restart_best_energies[restart] = energy
                    restart_best_g[restart] = g[0].detach().clone()
                    restart_best_b[restart] = b[0].detach().clone()
                    restart_best_steps[restart] = step + 1
                del energies, _states
            selected_restart = int(np.argmin(restart_best_energies))
            selected_energy, selected_state = energy_fn(
                restart_best_g[selected_restart][None, :],
                restart_best_b[selected_restart][None, :],
            )
            final_energy = float(selected_energy[0].detach().cpu())
            best = selected_state[0].detach().clone()
            best /= torch.clamp(torch.linalg.vector_norm(best), min=1e-20)
        return best, step + 1, {
            "best_energy": final_energy,
            "best_step": best_step,
            "elapsed_sec": time.time() - t0,
            "restart_execution": "sequential",
            "selected_restart": selected_restart,
            "restart_energies": restart_best_energies,
            "restart_gammas": [
                [float(value) for value in gamma.detach().cpu().tolist()]
                for gamma in restart_best_g
            ],
            "restart_betas": [
                [float(value) for value in beta.detach().cpu().tolist()]
                for beta in restart_best_b
            ],
            "restart_best_steps": restart_best_steps,
            "restart_initializer_roles": initializer_roles,
            "restart_seed": seed,
            "stopping_reason": (
                "early_stop_patience" if stale >= self.cfg.early_stop_patience else "max_steps"
            ),
            "gammas": restart_best_g[selected_restart].detach().clone(),
            "betas": restart_best_b[selected_restart].detach().clone(),
        }

    def optimize_spsa_sequential_restarts(
        self,
        energy_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        p_gamma: int,
        p_beta: int,
        label: str,
        trial_idx: int,
        initializers: Sequence[Tuple[torch.Tensor, torch.Tensor]] = (),
    ) -> Tuple[torch.Tensor, int, dict]:
        """Incumbent-preserving, calibrated first-order SPSA.

        Every supplied initializer is evaluated before perturbation.  The
        calibration follows the standard SPSA idea of estimating a typical
        gradient scale and choosing ``a`` to attain a requested first-step
        magnitude.  The best point among the initializer, calibration probes,
        update probes, and final iterate is retained.  Gamma angles are not
        wrapped because continuous objective coefficients need not be
        2-pi-periodic.
        """
        if self.cfg.spsa_c <= 0.0:
            raise ValueError("spsa_c must be positive")
        if self.cfg.spsa_target_step <= 0.0:
            raise ValueError("spsa_target_step must be positive")
        if self.cfg.spsa_calibration_directions < 1:
            raise ValueError("spsa_calibration_directions must be positive")
        if self.cfg.spsa_directions_per_update < 1:
            raise ValueError("spsa_directions_per_update must be positive")
        if self.cfg.spsa_max_step_norm <= 0.0:
            raise ValueError("spsa_max_step_norm must be positive")

        seed = int(self.inst.seed * 100000 + self._stable_label_seed(label))
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        g_init = torch.empty(
            (self.cfg.opt_restarts, p_gamma),
            dtype=self.real_dtype,
            device=self.device,
        ).uniform_(-math.pi, math.pi)
        b_init = torch.empty(
            (self.cfg.opt_restarts, p_beta),
            dtype=self.real_dtype,
            device=self.device,
        ).uniform_(-math.pi, math.pi)
        if self.cfg.opt_restarts > 0:
            g_init[0] = torch.linspace(
                0.05, 0.65, p_gamma, dtype=self.real_dtype, device=self.device
            )
            b_init[0] = torch.linspace(
                0.55, 0.05, p_beta, dtype=self.real_dtype, device=self.device
            )
        initializer_slots = max(0, self.cfg.opt_restarts - 1)
        usable_initializers = (
            list(initializers)[-initializer_slots:] if initializer_slots else []
        )
        initializer_start = max(
            1, self.cfg.opt_restarts - len(usable_initializers)
        )
        for restart, (g_value, b_value) in enumerate(
            usable_initializers, start=initializer_start
        ):
            if g_value.numel() != p_gamma or b_value.numel() != p_beta:
                raise ValueError(f"initializer shape mismatch for {label}")
            g_init[restart] = g_value.to(
                device=self.device, dtype=self.real_dtype
            )
            b_init[restart] = b_value.to(
                device=self.device, dtype=self.real_dtype
            )
        initializer_roles = [
            (
                "deterministic_linear_ramp"
                if restart == 0
                else "continuation"
                if restart >= initializer_start
                else "random_shared_stream"
            )
            for restart in range(self.cfg.opt_restarts)
        ]

        restart_best_energies = [float("inf")] * self.cfg.opt_restarts
        restart_best_theta: List[torch.Tensor] = []
        restart_best_steps = [0] * self.cfg.opt_restarts
        restart_initial_energies: List[float] = []
        restart_objective_calls = [0] * self.cfg.opt_restarts
        restart_calibration: List[dict] = []
        best_energy = float("inf")
        best_step = 0
        t0 = time.time()
        parameter_count = p_gamma + p_beta
        stability = max(
            1.0,
            self.cfg.spsa_stability_fraction * float(self.cfg.opt_steps),
        )

        for restart in range(self.cfg.opt_restarts):
            theta = torch.cat((g_init[restart], b_init[restart])).detach().clone()
            best_theta = theta.detach().clone()
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed + 1_000_003 * (restart + 1))

            def rademacher() -> torch.Tensor:
                bits = torch.randint(
                    0,
                    2,
                    theta.shape,
                    generator=generator,
                    device=self.device,
                )
                return bits.to(self.real_dtype).mul_(2).sub_(1)

            def observe_batch(
                candidates: torch.Tensor,
                observed_step: int,
            ) -> torch.Tensor:
                nonlocal best_theta, best_energy, best_step
                with torch.no_grad():
                    energies, states = energy_fn(
                        candidates[:, :p_gamma],
                        candidates[:, p_gamma:],
                    )
                    detached = energies.detach()
                restart_objective_calls[restart] += int(candidates.shape[0])
                for candidate_index in range(int(candidates.shape[0])):
                    value = float(detached[candidate_index].cpu())
                    if value < restart_best_energies[restart]:
                        restart_best_energies[restart] = value
                        best_theta = candidates[candidate_index].detach().clone()
                        restart_best_steps[restart] = observed_step
                    if value < best_energy:
                        best_energy = value
                        best_step = observed_step
                del states
                return detached

            # Preserve the unperturbed incumbent before any SPSA probe.
            initial_values = observe_batch(theta[None, :], 0)
            restart_initial_energies.append(float(initial_values[0].cpu()))

            calibration_gradients: List[torch.Tensor] = []
            for _calibration_index in range(
                self.cfg.spsa_calibration_directions
            ):
                delta = rademacher()
                candidates = torch.stack(
                    (
                        theta + self.cfg.spsa_c * delta,
                        theta - self.cfg.spsa_c * delta,
                    )
                )
                values = observe_batch(candidates, 0)
                scalar = (values[0] - values[1]) / (
                    2.0 * self.cfg.spsa_c
                )
                calibration_gradients.append(scalar * delta)
            calibration_stack = torch.stack(calibration_gradients)
            gradient_abs_mean = float(
                torch.mean(torch.abs(calibration_stack)).cpu()
            )
            gradient_rms = float(
                torch.sqrt(torch.mean(calibration_stack.square())).cpu()
            )
            safe_gradient_scale = max(gradient_abs_mean, 1e-12)
            a_base = (
                self.cfg.spsa_target_step
                * ((1.0 + stability) ** self.cfg.spsa_alpha)
                / safe_gradient_scale
            )
            restart_calibration.append(
                {
                    "restart": restart,
                    "gradient_abs_mean": gradient_abs_mean,
                    "gradient_rms": gradient_rms,
                    "a_base": a_base,
                    "c_base": self.cfg.spsa_c,
                    "target_step": self.cfg.spsa_target_step,
                    "stability": stability,
                    "directions": self.cfg.spsa_calibration_directions,
                }
            )

            for update in range(1, self.cfg.opt_steps + 1):
                c_k = self.cfg.spsa_c / (
                    float(update) ** self.cfg.spsa_gamma
                )
                gradient = torch.zeros_like(theta)
                for _direction_index in range(
                    self.cfg.spsa_directions_per_update
                ):
                    delta = rademacher()
                    candidates = torch.stack(
                        (theta + c_k * delta, theta - c_k * delta)
                    )
                    values = observe_batch(candidates, update)
                    scalar = (values[0] - values[1]) / (2.0 * c_k)
                    gradient.add_(scalar * delta)
                gradient.div_(float(self.cfg.spsa_directions_per_update))
                a_k = a_base / (
                    (float(update) + stability) ** self.cfg.spsa_alpha
                )
                update_vector = a_k * gradient
                update_norm = float(
                    torch.linalg.vector_norm(update_vector).cpu()
                )
                if update_norm > self.cfg.spsa_max_step_norm:
                    update_vector.mul_(
                        self.cfg.spsa_max_step_norm / update_norm
                    )
                    update_norm = self.cfg.spsa_max_step_norm
                theta = theta - update_vector

                log_this_step = self.cfg.log_every > 0 and (
                    update == 1 or update % self.cfg.log_every == 0
                )
                trace_this_step = self.cfg.trace_every > 0 and (
                    update == 1 or update % self.cfg.trace_every == 0
                )
                if log_this_step:
                    print(
                        f"    {label} SPSA restart={restart + 1}/"
                        f"{self.cfg.opt_restarts} step={update:>3}/"
                        f"{self.cfg.opt_steps} "
                        f"Ebest={restart_best_energies[restart]: .6g} "
                        f"|g_hat|={float(torch.linalg.vector_norm(gradient).cpu()):.5g} "
                        f"|dtheta|={update_norm:.5g}",
                        flush=True,
                    )
                if trace_this_step:
                    gamma_gradient = gradient[:p_gamma]
                    beta_gradient = gradient[p_gamma:]
                    self.trace_rows.append(
                        {
                            "seed": self.inst.seed,
                            "label": label,
                            "trial": trial_idx,
                            "optimizer": "spsa",
                            "optimizer_execution": "sequential",
                            "restart": restart,
                            "initializer_role": initializer_roles[restart],
                            "step": update,
                            "max_steps": self.cfg.opt_steps,
                            "restart_count": self.cfg.opt_restarts,
                            "parameter_count": parameter_count,
                            "energy_min": restart_best_energies[restart],
                            "objective_calls": restart_objective_calls[restart],
                            "gradient_l2": float(
                                torch.linalg.vector_norm(gradient).cpu()
                            ),
                            "gradient_abs_mean": float(
                                torch.mean(torch.abs(gradient)).cpu()
                            ),
                            "gradient_variance": float(
                                torch.var(gradient, unbiased=False).cpu()
                            ),
                            "gamma_gradient_variance": float(
                                torch.var(
                                    gamma_gradient, unbiased=False
                                ).cpu()
                            ),
                            "beta_gradient_variance": float(
                                torch.var(
                                    beta_gradient, unbiased=False
                                ).cpu()
                            ),
                            "spsa_a_k": a_k,
                            "spsa_c_k": c_k,
                            "spsa_update_l2": update_norm,
                            "elapsed_sec": time.time() - t0,
                        }
                    )

            # Score the final iterate even if it was never a +/- probe.
            observe_batch(theta[None, :], self.cfg.opt_steps)
            restart_best_theta.append(best_theta)

        del g_init, b_init
        selected_restart = int(np.argmin(restart_best_energies))
        selected_theta = restart_best_theta[selected_restart]
        with torch.no_grad():
            selected_energy, selected_state = energy_fn(
                selected_theta[:p_gamma][None, :],
                selected_theta[p_gamma:][None, :],
            )
            best = selected_state[0].detach().clone()
            best /= torch.clamp(
                torch.linalg.vector_norm(best), min=1e-20
            )
            final_energy = float(selected_energy[0].detach().cpu())
        restart_objective_calls[selected_restart] += 1
        return best, self.cfg.opt_steps, {
            "optimizer_name": "spsa",
            "best_energy": final_energy,
            "best_step": best_step,
            "elapsed_sec": time.time() - t0,
            "restart_execution": "sequential",
            "selected_restart": selected_restart,
            "restart_energies": restart_best_energies,
            "restart_gammas": [
                [
                    float(value)
                    for value in theta_value[:p_gamma].detach().cpu().tolist()
                ]
                for theta_value in restart_best_theta
            ],
            "restart_betas": [
                [
                    float(value)
                    for value in theta_value[p_gamma:].detach().cpu().tolist()
                ]
                for theta_value in restart_best_theta
            ],
            "restart_best_steps": restart_best_steps,
            "restart_initial_energies": restart_initial_energies,
            "restart_initializer_roles": initializer_roles,
            "restart_seed": seed,
            "restart_objective_calls": restart_objective_calls,
            "objective_calls_total": int(sum(restart_objective_calls)),
            "gradient_calls_total": 0,
            "hardware_circuit_proxy_total": int(
                sum(restart_objective_calls)
            ),
            "spsa_calibration": restart_calibration,
            "spsa_settings": {
                "a_calibration": "first-step target divided by mean absolute SPSA gradient",
                "c": self.cfg.spsa_c,
                "target_step": self.cfg.spsa_target_step,
                "alpha": self.cfg.spsa_alpha,
                "gamma": self.cfg.spsa_gamma,
                "stability_fraction": self.cfg.spsa_stability_fraction,
                "calibration_directions": self.cfg.spsa_calibration_directions,
                "directions_per_update": self.cfg.spsa_directions_per_update,
                "max_step_norm": self.cfg.spsa_max_step_norm,
                "incumbent_preserved": True,
                "gamma_wrapping": False,
            },
            "stopping_reason": "max_steps",
            "gammas": selected_theta[:p_gamma].detach().clone(),
            "betas": selected_theta[p_gamma:].detach().clone(),
        }

    def state_stats(self, state: Optional[torch.Tensor]) -> dict:
        nan_stats = {
            "success_prob": float("nan"),
            "feasible_mass": float("nan"),
            "ipr": float("nan"),
            "entropy": float("nan"),
            "feasible_cond_entropy": float("nan"),
            "feasible_cond_ipr": float("nan"),
            "feasible_eff_support_entropy": float("nan"),
            "feasible_eff_support_ipr": float("nan"),
            "feasible_uniformity_entropy": float("nan"),
            "feasible_uniformity_ipr": float("nan"),
            "feasible_max_cond_prob": float("nan"),
            "opt_cond_prob": float("nan"),
            "opt_over_uniform_cond": float("nan"),
        }
        if state is None:
            return nan_stats
        p = torch.abs(state) ** 2
        p = p / torch.clamp(torch.sum(p), min=1e-20)
        success = float(torch.sum(p[self.inst.opt_mask]).detach().cpu())
        feasible = float(torch.sum(p[self.inst.feasible_mask]).detach().cpu())
        ipr = float(torch.sum(p * p).detach().cpu())
        entropy = float(torch.sum(-p * torch.log(torch.clamp(p, min=1e-30))).detach().cpu())
        out = {
            "success_prob": success,
            "feasible_mass": feasible,
            "ipr": ipr,
            "entropy": entropy,
            **{k: v for k, v in nan_stats.items() if k not in ("success_prob", "feasible_mass", "ipr", "entropy")},
        }
        if feasible > 1e-30 and self.inst.feasible_count > 0:
            pf = p[self.inst.feasible_mask]
            qf = pf / torch.clamp(torch.sum(pf), min=1e-30)
            cond_entropy = float(torch.sum(-qf * torch.log(torch.clamp(qf, min=1e-30))).detach().cpu())
            cond_ipr = float(torch.sum(qf * qf).detach().cpu())
            eff_entropy = float(math.exp(cond_entropy)) if cond_entropy < 700.0 else float("inf")
            eff_ipr = float(1.0 / cond_ipr) if cond_ipr > 0.0 else float("inf")
            opt_cond = success / feasible
            feasible_count = float(self.inst.feasible_count)
            out.update(
                {
                    "feasible_cond_entropy": cond_entropy,
                    "feasible_cond_ipr": cond_ipr,
                    "feasible_eff_support_entropy": eff_entropy,
                    "feasible_eff_support_ipr": eff_ipr,
                    "feasible_uniformity_entropy": eff_entropy / feasible_count,
                    "feasible_uniformity_ipr": eff_ipr / feasible_count,
                    "feasible_max_cond_prob": float(torch.max(qf).detach().cpu()),
                    "opt_cond_prob": opt_cond,
                    "opt_over_uniform_cond": opt_cond * feasible_count / max(float(self.inst.optimum_count), 1.0),
                }
            )
        return out

    def run_stage1_only(self, p1: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        key = (trial_idx, p1)
        if key not in self._stage1_cache:
            initializers, init_source = self._continuation_initializers(self._stage1_param_cache, p1)
            optimize_stage1 = self.optimize_sequential_restarts if self.cfg.stage1_sequential_restarts else self.optimize
            state, iters, info = optimize_stage1(
                lambda g, b: self.energy_xy(self.inst.Cconf, g, b, p1),
                p_gamma=p1,
                p_beta=p1,
                label=f"XY-Stage1-Only p1={p1}",
                trial_idx=trial_idx,
                initializers=initializers,
            )
            self._last_optimizer_info = info
            info["stage1_init_source"] = init_source
            self._stage1_param_cache[p1] = (info["gammas"], info["betas"])
            self._stage1_cache[key] = (state, iters, info)
        state, iters, info = self._stage1_cache[key]
        stats = self.state_stats(state)
        return stats["success_prob"], iters, state, {
            "stage1_feasible_mass": stats["feasible_mass"],
            "stage1_success_prob": stats["success_prob"],
            "stage1_energy": info["best_energy"],
            "stage1_selected_restart": info.get("selected_restart", -1),
            "stage1_restart_energies": json.dumps(info.get("restart_energies", [])),
            "stage1_init_source": info.get("stage1_init_source", "shared_fixed_state"),
            "stage1_gammas": json.dumps([float(value) for value in info["gammas"].detach().cpu().tolist()]),
            "stage1_betas": json.dumps([float(value) for value in info["betas"].detach().cpu().tolist()]),
            "stage1_state_reused": bool(info.get("stage1_state_reused", False)),
            "stage1_restart_execution": "sequential" if self.cfg.stage1_sequential_restarts else "batched",
            "stage2_energy": float("nan"),
            "stage2_guard_scale": 0.0,
        }

    def run_xy_nested(self, p1: int, p2: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        _p, it1, phi1, extra1 = self.run_stage1_only(p1, trial_idx)
        use_sequential = self.cfg.stage2_sequential_restarts and p2 >= self.cfg.stage2_sequential_min_depth
        optimize_stage2 = self.optimize_sequential_restarts if use_sequential else self.optimize
        p1_cache = self._stage2_param_cache.setdefault(p1, {})
        initializers, init_source = self._continuation_initializers(p1_cache, p2)
        if not initializers:
            initializers = self._identity_initializer(p2, p2)
            init_source = "identity_on_stage1_state"
        phi2, it2, info2 = optimize_stage2(
            lambda g, b: self.energy_history(phi1, self.inst.Cobj, g, b, p2),
            p_gamma=p2,
            p_beta=p2,
            label=f"XY-Nested S2 p2={p2}",
            trial_idx=trial_idx,
            initializers=initializers,
        )
        self._last_optimizer_info = info2
        p1_cache[p2] = (info2["gammas"], info2["betas"])
        stats = self.state_stats(phi2)
        return stats["success_prob"], it1 + it2, phi2, {
            "stage1_feasible_mass": extra1["stage1_feasible_mass"],
            "stage1_success_prob": extra1["stage1_success_prob"],
            "stage1_energy": extra1["stage1_energy"],
            "stage1_selected_restart": extra1["stage1_selected_restart"],
            "stage1_restart_energies": extra1["stage1_restart_energies"],
            "stage1_init_source": extra1["stage1_init_source"],
            "stage1_gammas": extra1["stage1_gammas"],
            "stage1_betas": extra1["stage1_betas"],
            "stage1_state_reused": extra1["stage1_state_reused"],
            "stage1_restart_execution": extra1["stage1_restart_execution"],
            "stage2_energy": info2["best_energy"],
            "stage2_guard_scale": 0.0,
            "stage2_restart_execution": "sequential" if use_sequential else "batched",
            "stage2_init_source": init_source,
            "stage2_selected_restart": info2.get("selected_restart", -1),
            "stage2_restart_energies": json.dumps(info2.get("restart_energies", [])),
            "stage2_gammas": json.dumps([float(value) for value in info2["gammas"].detach().cpu().tolist()]),
            "stage2_betas": json.dumps([float(value) for value in info2["betas"].detach().cpu().tolist()]),
        }

    def run_xy_warm(self, p1: int, p2: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        """Use the exact stage-1 reference state with an ordinary block-XY mixer."""
        _p, it1, phi1, extra1 = self.run_stage1_only(p1, trial_idx)
        h = self.penalty * self.inst.Cconf + self.inst.Cobj
        p1_cache = self._warm_param_cache.setdefault(p1, {})
        initializers, init_source = self._continuation_initializers(p1_cache, p2)
        if not initializers:
            initializers = self._identity_initializer(p2, p2)
            init_source = "identity_on_stage1_state"
        optimize_warm = self.optimize_sequential_restarts if self.cfg.direct_sequential_restarts else self.optimize
        phi2, it2, info2 = optimize_warm(
            lambda g, b: self.energy_xy(h, g, b, p2, initial_state=phi1),
            p_gamma=p2,
            p_beta=p2,
            label=f"XY-Warm-Same-State S2 p2={p2}",
            trial_idx=trial_idx,
            initializers=initializers,
        )
        self._last_optimizer_info = info2
        p1_cache[p2] = (info2["gammas"], info2["betas"])
        stats = self.state_stats(phi2)
        return stats["success_prob"], it1 + it2, phi2, {
            "stage1_feasible_mass": extra1["stage1_feasible_mass"],
            "stage1_success_prob": extra1["stage1_success_prob"],
            "stage1_energy": extra1["stage1_energy"],
            "stage1_selected_restart": extra1["stage1_selected_restart"],
            "stage1_restart_energies": extra1["stage1_restart_energies"],
            "stage1_init_source": extra1["stage1_init_source"],
            "stage1_gammas": extra1["stage1_gammas"],
            "stage1_betas": extra1["stage1_betas"],
            "stage1_state_reused": extra1["stage1_state_reused"],
            "stage1_restart_execution": extra1["stage1_restart_execution"],
            "energy": info2["best_energy"],
            "optimizer_init_source": init_source,
            "optimizer_selected_restart": info2.get("selected_restart", -1),
            "optimizer_restart_energies": json.dumps(info2.get("restart_energies", [])),
            "optimizer_gammas": json.dumps(
                [float(value) for value in info2["gammas"].detach().cpu().tolist()]
            ),
            "optimizer_betas": json.dumps(
                [float(value) for value in info2["betas"].detach().cpu().tolist()]
            ),
            "optimizer_restart_execution": (
                "sequential" if self.cfg.direct_sequential_restarts else "batched"
            ),
            "warm_reference": "exact_same_stage1_state_as_XY-Nested",
        }

    def run_std_xy(self, p: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        h = self.penalty * self.inst.Cconf + self.inst.Cobj
        cache = self._direct_param_cache["Std-XY"]
        initializers, init_source = self._continuation_initializers(cache, p)
        if not initializers:
            initializers = self._identity_initializer(p, p)
            init_source = "identity_on_uniform_state"
        optimize_direct = self.optimize_sequential_restarts if self.cfg.direct_sequential_restarts else self.optimize
        state, iters, info = optimize_direct(
            lambda g, b: self.energy_xy(h, g, b, p),
            p_gamma=p,
            p_beta=p,
            label=f"Std-XY p={p}",
            trial_idx=trial_idx,
            initializers=initializers,
        )
        self._last_optimizer_info = info
        cache[p] = (info["gammas"], info["betas"])
        return self.state_stats(state)["success_prob"], iters, state, {
            "energy": info["best_energy"],
            "optimizer_init_source": init_source,
            "optimizer_selected_restart": info.get("selected_restart", -1),
            "optimizer_restart_energies": json.dumps(info.get("restart_energies", [])),
            "optimizer_gammas": json.dumps([float(value) for value in info["gammas"].detach().cpu().tolist()]),
            "optimizer_betas": json.dumps([float(value) for value in info["betas"].detach().cpu().tolist()]),
            "optimizer_restart_execution": "sequential" if self.cfg.direct_sequential_restarts else "batched",
        }

    def run_decoupled_xy(self, p: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        h_pen = self.penalty * self.inst.Cconf
        cache = self._direct_param_cache["d-XY"]
        initializers, init_source = self._continuation_initializers(cache, p, split_gamma=True)
        if not initializers:
            initializers = self._identity_initializer(2 * p, p)
            init_source = "identity_on_uniform_state"
        optimize_direct = self.optimize_sequential_restarts if self.cfg.direct_sequential_restarts else self.optimize
        state, iters, info = optimize_direct(
            lambda g, b: self.energy_xy(h_pen + self.inst.Cobj, g, b, p, split=(h_pen, self.inst.Cobj)),
            p_gamma=2 * p,
            p_beta=p,
            label=f"d-XY p={p}",
            trial_idx=trial_idx,
            initializers=initializers,
        )
        self._last_optimizer_info = info
        cache[p] = (info["gammas"], info["betas"])
        return self.state_stats(state)["success_prob"], iters, state, {
            "energy": info["best_energy"],
            "optimizer_init_source": init_source,
            "optimizer_selected_restart": info.get("selected_restart", -1),
            "optimizer_restart_energies": json.dumps(info.get("restart_energies", [])),
            "optimizer_gammas": json.dumps([float(value) for value in info["gammas"].detach().cpu().tolist()]),
            "optimizer_betas": json.dumps([float(value) for value in info["betas"].detach().cpu().tolist()]),
            "optimizer_restart_execution": "sequential" if self.cfg.direct_sequential_restarts else "batched",
        }


def term_indicator(inst: HeteroBlockWDSHCCInstance, term: Term) -> np.ndarray:
    mask = np.uint64(0)
    for q in term:
        mask |= np.uint64(1) << np.uint64(q)
    return (inst.state_bits_np & mask) == mask


def term_support(inst: HeteroBlockWDSHCCInstance, term: Term) -> Tuple[int, int]:
    selected_per_block = np.bincount(
        np.array([inst.block_for_qubit(q) for q in term], dtype=np.int16),
        minlength=inst.M,
    )
    block_count = 1
    for size, weight, selected in zip(inst.block_sizes, inst.block_weights, selected_per_block):
        selected = int(selected)
        if selected > weight:
            block_count = 0
            break
        block_count *= math.comb(size - selected, weight - selected)

    mask = np.uint64(0)
    for q in term:
        mask |= np.uint64(1) << np.uint64(q)
    feasible_count = int(np.count_nonzero((inst.feasible_state_bits_np & mask) == mask))
    return int(block_count), feasible_count


def classify_terms(inst: HeteroBlockWDSHCCInstance, terms: Dict[Term, float]) -> Dict[str, int]:
    stats = {
        "total": len(terms),
        "block_active": 0,
        "block_zero": 0,
        "feasible_active": 0,
        "feasible_zero": 0,
        "feasible_constant_one": 0,
        "infeasible_only": 0,
    }
    for term in terms:
        block_count, feasible_count = term_support(inst, term)
        stats["block_active" if block_count > 0 else "block_zero"] += 1
        stats["feasible_active" if feasible_count > 0 else "feasible_zero"] += 1
        if feasible_count == inst.feasible_count:
            stats["feasible_constant_one"] += 1
        if block_count > 0 and feasible_count == 0:
            stats["infeasible_only"] += 1
    return stats


def term_cost(inst: HeteroBlockWDSHCCInstance, terms: Dict[Term, float]) -> np.ndarray:
    c = np.zeros(inst.dim, dtype=np.float32)
    for term, weight in terms.items():
        mask = np.uint64(0)
        for q in term:
            mask |= np.uint64(1) << np.uint64(q)
        c[(inst.state_bits_np & mask) == mask] -= np.float32(weight)
    return c


def draw_weight(rng: np.random.Generator, mode: str, scale: float, integer_max: int) -> float:
    if mode == "positive_integer":
        return float(rng.integers(1, integer_max + 1))
    if mode == "positive_exp":
        return float(scale * math.exp(0.35 * float(rng.normal())))
    if mode == "signed_integer":
        mag = int(rng.integers(1, integer_max + 1))
        return float(mag if rng.random() < 0.5 else -mag)
    if mode == "signed_normal":
        return float(scale * rng.normal())
    raise ValueError(f"unknown weight mode {mode!r}")


def choose_blocks_for_counts(inst: HeteroBlockWDSHCCInstance, rng: np.random.Generator, counts: Sequence[int]) -> Optional[List[int]]:
    remaining = list(range(inst.M))
    chosen: List[int] = []
    for count in sorted((int(c) for c in counts), reverse=True):
        compatible = [b for b in remaining if inst.block_sizes[b] >= count and inst.block_weights[b] >= count]
        if not compatible:
            return None
        b = int(rng.choice(compatible))
        remaining.remove(b)
        chosen.append(b)
    rng.shuffle(chosen)
    return chosen


def sample_term(inst: HeteroBlockWDSHCCInstance, rng: np.random.Generator, mode: str, arity: int) -> Term:
    if mode in ("cross", "cross4"):
        counts = [1] * arity
    elif mode == "pair2":
        if arity % 2:
            raise ValueError("pair2 mode requires even arity")
        counts = [2] * (arity // 2)
    elif mode == "mixed":
        if arity == 4:
            compositions = [(1, 1, 1, 1), (2, 1, 1), (2, 2)]
            probs = np.array([0.45, 0.35, 0.20], dtype=float)
        elif arity == 5:
            compositions = [(1, 1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1)]
            probs = np.array([0.40, 0.35, 0.25], dtype=float)
        elif arity == 6:
            compositions = [(1, 1, 1, 1, 1, 1), (2, 1, 1, 1, 1), (2, 2, 1, 1)]
            probs = np.array([0.34, 0.33, 0.33], dtype=float)
        else:
            compositions = [tuple([1] * min(arity, inst.M))]
            probs = np.array([1.0], dtype=float)
        valid = [(comp, prob) for comp, prob in zip(compositions, probs) if len(comp) <= inst.M]
        if not valid:
            raise ValueError("no valid mixed compositions")
        comps = [v[0] for v in valid]
        probs = np.array([v[1] for v in valid], dtype=float)
        probs = probs / probs.sum()
        counts = list(comps[int(rng.choice(len(comps), p=probs))])
    else:
        raise ValueError(f"unknown term mode {mode!r}")

    blocks = choose_blocks_for_counts(inst, rng, counts)
    if blocks is None:
        raise ValueError(f"cannot sample arity={arity} counts={counts} on layout={inst.block_sizes} weights={inst.block_weights}")
    bits: List[int] = []
    for b, count in zip(blocks, counts):
        local = rng.choice(inst.block_sizes[b], size=int(count), replace=False)
        bits.extend([inst._qubit(b, int(q)) for q in local])
    return tuple(sorted(bits))


def make_random_terms(
    inst: HeteroBlockWDSHCCInstance,
    count: int,
    seed: int,
    arity: int,
    term_mode: str,
    weight_mode: str,
    weight_scale: float,
    integer_max: int,
    min_feasible_support: int,
) -> Tuple[Dict[Term, float], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    terms: Dict[Term, float] = {}
    candidates = 0
    rejected_duplicate = 0
    rejected_block_zero = 0
    rejected_feasible_low = 0
    max_attempts = max(1000, 100 * count)
    while len(terms) < count and candidates < max_attempts:
        candidates += 1
        term = sample_term(inst, rng, term_mode, arity)
        if term in terms:
            rejected_duplicate += 1
            continue
        block_count, feasible_count = term_support(inst, term)
        if block_count <= 0:
            rejected_block_zero += 1
            continue
        if feasible_count < min_feasible_support:
            rejected_feasible_low += 1
            continue
        terms[term] = draw_weight(rng, weight_mode, weight_scale, integer_max)
    if len(terms) < count:
        raise RuntimeError(f"accepted only {len(terms)} random {arity}-body terms out of requested {count}")
    stats = {
        "candidate_terms": candidates,
        "accepted_terms": len(terms),
        "rejected_duplicate": rejected_duplicate,
        "rejected_block_zero": rejected_block_zero,
        "rejected_feasible_low": rejected_feasible_low,
    }
    stats.update({f"term_{k}": v for k, v in classify_terms(inst, terms).items()})
    return terms, stats


def enumerate_feasible_mixed4_terms(inst: HeteroBlockWDSHCCInstance) -> List[Term]:
    """Enumerate all feasible-active (2,1,1) and (2,2) four-body terms."""
    if inst.M != 3 or any(weight < 2 for weight in inst.block_weights):
        raise ValueError("uniform_mixed4 requires exactly three blocks with block weight at least two")

    conflict_pairs = {tuple(sorted((int(u), int(v)))) for u, v, _ in inst.conflict_edges}

    def admissible(term: Term) -> bool:
        selected = set(term)
        return not any(u in selected and v in selected for u, v in conflict_pairs)

    pool: set[Term] = set()
    blocks = range(inst.M)

    for doubled in blocks:
        singles = [block for block in blocks if block != doubled]
        for pair in itertools.combinations(range(inst.block_sizes[doubled]), 2):
            pair_bits = [inst._qubit(doubled, local) for local in pair]
            for local_a in range(inst.block_sizes[singles[0]]):
                for local_b in range(inst.block_sizes[singles[1]]):
                    term = tuple(
                        sorted(
                            pair_bits
                            + [
                                inst._qubit(singles[0], local_a),
                                inst._qubit(singles[1], local_b),
                            ]
                        )
                    )
                    if admissible(term):
                        pool.add(term)

    for block_a, block_b in itertools.combinations(blocks, 2):
        for pair_a in itertools.combinations(range(inst.block_sizes[block_a]), 2):
            for pair_b in itertools.combinations(range(inst.block_sizes[block_b]), 2):
                term = tuple(
                    sorted(
                        [inst._qubit(block_a, local) for local in pair_a]
                        + [inst._qubit(block_b, local) for local in pair_b]
                    )
                )
                if admissible(term):
                    pool.add(term)

    return sorted(pool)


def make_uniform_mixed4_terms(
    inst: HeteroBlockWDSHCCInstance,
    density: float,
    seed: int,
    weight_mode: str,
    weight_scale: float,
    integer_max: int,
) -> Tuple[Dict[Term, float], Dict[str, object]]:
    if not (0.0 < density <= 1.0):
        raise ValueError("term density must lie in (0, 1]")
    pool = enumerate_feasible_mixed4_terms(inst)
    count = max(1, min(len(pool), int(round(density * len(pool)))))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(pool), size=count, replace=False)
    terms = {
        pool[int(index)]: draw_weight(rng, weight_mode, weight_scale, integer_max)
        for index in selected
    }
    realized_density = len(terms) / len(pool)
    return terms, {
        "candidate_terms": len(pool),
        "candidate_pool_size": len(pool),
        "accepted_terms": len(terms),
        "requested_term_density": float(density),
        "realized_term_density": float(realized_density),
        "expected_active_terms_per_feasible_state": float(15.0 * realized_density),
        "rejected_duplicate": 0,
        "rejected_block_zero": 0,
        "rejected_feasible_low": 0,
        "term_total": len(terms),
        "term_block_active": len(terms),
        "term_block_zero": 0,
        "term_feasible_active": len(terms),
        "term_feasible_zero": 0,
        "term_feasible_constant_one": 0,
        "term_infeasible_only": 0,
    }


def set_target_mask(inst: HeteroBlockWDSHCCInstance, target_mode: str, topk: int, top_fraction: float) -> Dict[str, float]:
    feasible_idx = np.where(inst.feasible_mask_np)[0]
    feasible_obj = inst.Cobj_np[feasible_idx]
    order = np.argsort(feasible_obj, kind="stable")
    min_feasible = float(feasible_obj[order[0]])
    if target_mode == "exact":
        chosen = feasible_idx[np.abs(feasible_obj - min_feasible) <= 1e-7]
        threshold = min_feasible
    elif target_mode == "topk":
        k = max(1, min(int(topk), len(feasible_idx)))
        chosen = feasible_idx[order[:k]]
        threshold = float(feasible_obj[order[k - 1]])
    elif target_mode == "topfrac":
        k = max(1, min(len(feasible_idx), int(round(float(top_fraction) * len(feasible_idx)))))
        chosen = feasible_idx[order[:k]]
        threshold = float(feasible_obj[order[k - 1]])
    elif target_mode == "topfrac_ties":
        k = max(1, min(len(feasible_idx), int(round(float(top_fraction) * len(feasible_idx)))))
        threshold = float(feasible_obj[order[k - 1]])
        chosen = feasible_idx[feasible_obj <= threshold]
    else:
        raise ValueError(f"unknown target mode {target_mode!r}")
    inst.min_feasible_obj = min_feasible
    opt = np.zeros(inst.dim, dtype=bool)
    opt[chosen] = True
    inst.opt_mask_np = opt
    inst.opt_mask = torch.tensor(opt, dtype=torch.bool, device=inst.device)
    return {
        "min_feasible_obj": min_feasible,
        "target_threshold": threshold,
        "target_count": int(opt.sum()),
        "target_fraction_feasible": float(opt.sum() / max(1, inst.feasible_count)),
    }


def install_random_objective(inst: HeteroBlockWDSHCCInstance, args: argparse.Namespace, seed: int) -> Tuple[int, Dict[str, object]]:
    if args.term_mode == "uniform_mixed4":
        if int(args.arity) != 4:
            raise ValueError("uniform_mixed4 requires arity=4")
        terms, term_stats = make_uniform_mixed4_terms(
            inst=inst,
            density=args.term_density,
            seed=seed + 19001,
            weight_mode=args.weight_mode,
            weight_scale=args.weight_scale,
            integer_max=args.integer_max,
        )
    else:
        terms, term_stats = make_random_terms(
            inst=inst,
            count=args.term_count,
            seed=seed + 19001,
            arity=args.arity,
            term_mode=args.term_mode,
            weight_mode=args.weight_mode,
            weight_scale=args.weight_scale,
            integer_max=args.integer_max,
            min_feasible_support=args.min_feasible_support,
        )
    c = term_cost(inst, terms)
    inst.objective_terms = dict(terms)
    inst.Cobj_np = c
    inst.Cobj = torch.tensor(c, dtype=inst.real_dtype, device=inst.device)
    target_stats = set_target_mask(inst, args.target_mode, args.topk, args.top_fraction)
    ru_per_term = int(args.ru_per_term) if int(args.ru_per_term) > 0 else int(2 ** int(args.arity) - 2)
    obj_ru = ru_per_term * len(terms)
    weights = np.array(list(terms.values()), dtype=float)
    return obj_ru, {
        "hyper_arity": args.arity,
        "objective_mode": f"hetero_random_weighted_{args.arity}body",
        "term_mode": args.term_mode,
        "weight_mode": args.weight_mode,
        "term_count": len(terms),
        "objective_RU": obj_ru,
        "ru_per_term": ru_per_term,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        **term_stats,
        **target_stats,
    }


def resource_stage1(inst: HeteroBlockWDSHCCInstance, p1: int) -> Tuple[int, int]:
    legacy_ru = int(p1 * inst.conflict_ru)
    corrected_ru = direct_ru(
        p1,
        inst.conflict_ru,
        xy_mixer_ru(inst.block_sizes),
    )
    return corrected_ru, legacy_ru


def resource_xy(inst: HeteroBlockWDSHCCInstance, p: int, objective_ru: int) -> Tuple[int, int]:
    legacy_ru = int(p * (inst.conflict_ru + objective_ru))
    corrected_ru = bcst_ru(
        "d-XY",
        f"p={p}",
        n=inst.N,
        layout=inst.block_sizes,
        conflict_ru=inst.conflict_ru,
        objective_ru=objective_ru,
    )
    return corrected_ru, legacy_ru


def resource_nested(inst: HeteroBlockWDSHCCInstance, p1: int, p2: int, objective_ru: int) -> Tuple[int, int]:
    stage1 = p1 * inst.conflict_ru
    history = 2 * stage1 + inst.mcz_ru
    legacy_ru = int(stage1 + p2 * (objective_ru + history))
    corrected_ru = bcst_ru(
        "XY-Nested",
        f"({p1},{p2})",
        n=inst.N,
        layout=inst.block_sizes,
        conflict_ru=inst.conflict_ru,
        objective_ru=objective_ru,
    )
    return corrected_ru, legacy_ru


def resource_warm(inst: HeteroBlockWDSHCCInstance, p1: int, p2: int, objective_ru: int) -> Tuple[int, int]:
    mixer_ru = xy_mixer_ru(inst.block_sizes)
    corrected_ru = warm_start_ru(
        p1,
        p2,
        inst.conflict_ru,
        mixer_ru,
        inst.conflict_ru + objective_ru,
        mixer_ru,
    )
    legacy_ru = int(
        p1 * inst.conflict_ru + p2 * (inst.conflict_ru + objective_ru)
    )
    return corrected_ru, legacy_ru


def append_trial(
    rows: List[dict],
    runner: HeteroQAOARunner,
    seed: int,
    algorithm: str,
    params: str,
    corrected_ru: int,
    legacy_ru: int,
    fn,
    target_meta: dict,
    target_prob: float,
) -> None:
    t0 = time.time()
    p_succ, iters, state, extra = fn()
    elapsed = time.time() - t0
    stats = runner.state_stats(state)
    trial_rts = rts(float(p_succ), target_prob)
    rows.append(
        {
            "seed": seed,
            "algorithm": algorithm,
            "params": params,
            "trial": 0,
            "accounting_tag": ACCOUNTING_VERSION,
            "RU": corrected_ru,
            "legacy_phase_RU": legacy_ru,
            "success_prob": float(p_succ),
            "RTS": trial_rts,
            "RTS_x_RU": trial_rts * corrected_ru if np.isfinite(trial_rts) else float("inf"),
            "legacy_phase_RTS_x_RU": trial_rts * legacy_ru if np.isfinite(trial_rts) else float("inf"),
            "iters": iters,
            "elapsed_sec": elapsed,
            **target_meta,
            **stats,
            **extra,
        }
    )
    print(
        f"<<< {algorithm} {params} P={100 * p_succ:.5f}% "
        f"feasible={100 * stats['feasible_mass']:.2f}% RU={corrected_ru} "
        f"legacyRU={legacy_ru} "
        f"RTSxRU={rows[-1]['RTS_x_RU']:.3g} elapsed={elapsed:.1f}s",
        flush=True,
    )


def aggregate(rows: Sequence[dict], target_prob: float) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["params"]), []).append(row)
    out: List[dict] = []
    for (algo, params), vals in grouped.items():
        probs = np.array([float(v["success_prob"]) for v in vals], dtype=float)
        ru = np.array([float(v["RU"]) for v in vals], dtype=float)
        legacy_ru = np.array([float(v["legacy_phase_RU"]) for v in vals], dtype=float)
        seed_costs = np.array([float(v["RTS_x_RU"]) for v in vals], dtype=float)
        legacy_seed_costs = np.array([float(v["legacy_phase_RTS_x_RU"]) for v in vals], dtype=float)
        p = float(np.nanmean(probs))
        row_rts = rts(p, target_prob)
        corrected_ru = float(np.nanmean(ru))
        legacy_phase_ru = float(np.nanmean(legacy_ru))
        row = {
            "algorithm": algo,
            "params": params,
            "accounting_tag": ACCOUNTING_VERSION,
            "P_succ_mean": p,
            "P_succ_std": float(np.nanstd(probs)),
            "RU": corrected_ru,
            "legacy_phase_RU": legacy_phase_ru,
            "RTS": row_rts,
            "RTS_x_RU": float(np.nanmean(seed_costs)),
            "RTS_x_RU_std": float(np.nanstd(seed_costs)),
            "RTS_x_RU_from_mean": row_rts * corrected_ru if np.isfinite(row_rts) else float("inf"),
            "legacy_phase_RTS_x_RU": float(np.nanmean(legacy_seed_costs)),
            "legacy_phase_RTS_x_RU_from_mean": row_rts * legacy_phase_ru if np.isfinite(row_rts) else float("inf"),
            "iters_mean": float(np.nanmean([float(v.get("iters", np.nan)) for v in vals])),
            "runs": len(vals),
        }
        for key in (
            "feasible_mass",
            "feasible_uniformity_entropy",
            "feasible_uniformity_ipr",
            "feasible_eff_support_entropy",
            "feasible_eff_support_ipr",
            "feasible_max_cond_prob",
            "opt_cond_prob",
            "opt_over_uniform_cond",
            "stage1_feasible_mass",
            "stage1_success_prob",
        ):
            values = [
                float(v.get(key, np.nan))
                if v.get(key, np.nan) not in (None, "")
                else float("nan")
                for v in vals
            ]
            if any(np.isfinite(values)):
                row[f"{key}_mean"] = float(np.nanmean(values))
        out.append(row)
    return sorted(out, key=lambda row: (float("inf") if not np.isfinite(float(row["RTS_x_RU"])) else float(row["RTS_x_RU"]), row["algorithm"]))


def write_report(path: str, summary: dict, agg: Sequence[dict]) -> None:
    arity = int(summary.get("hyper_arity", 4))
    lines = [
        f"# Heterogeneous Random Hyper{arity} WDS-HCC Scale-Up",
        "",
        "## Instance",
        "",
    ]
    for key in (
        "N",
        "M",
        "layout",
        "weights",
        "local_basis_sizes",
        "subspace_dim",
        "feasible_count",
        "target_count",
        "target_fraction_feasible",
        "term_count",
        "candidate_pool_size",
        "requested_term_density",
        "realized_term_density",
        "expected_active_terms_per_feasible_state",
        "objective_RU",
        "ru_per_term",
        "term_mode",
        "weight_mode",
        "accounting_tag",
        "xy_mixer_RU",
    ):
        if key in summary:
            lines.append(f"- `{key}`: `{summary[key]}`")
    lines += [
        "",
        "## Aggregated Results",
        "",
        "| algorithm | params | P_success | corrected RU | mean seed RTS x RU | cost from mean P | legacy phase cost | feasible mass | amp/uniform |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['algorithm']} | {row['params']} | {100 * float(row['P_succ_mean']):.5f}% | "
            f"{float(row['RU']):.0f} | {float(row['RTS_x_RU']):.0f} | "
            f"{float(row['RTS_x_RU_from_mean']):.0f} | {float(row['legacy_phase_RTS_x_RU']):.0f} | "
            f"{100 * float(row.get('feasible_mass_mean', float('nan'))):.2f}% | "
            f"{float(row.get('opt_over_uniform_cond_mean', float('nan'))):.3g} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def update_outputs(outdir: str, summary: dict, rows: List[dict], target_prob: float) -> List[dict]:
    agg = aggregate(rows, target_prob)
    write_csv(os.path.join(outdir, "trial_results.csv"), rows)
    write_csv(os.path.join(outdir, "aggregated_results.csv"), agg)
    with open(os.path.join(outdir, "instance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    write_report(os.path.join(outdir, "HETERO_RANDOM_HYPERK_RESULTS.md"), summary, agg)
    return agg


def run(args: argparse.Namespace) -> str:
    run_started = time.time()
    layout = parse_ints(args.layout)
    weights = parse_ints(args.weights, default_weights(layout))
    seeds = parse_ints(args.seeds, (531,))
    nested_configs = parse_configs(args.nested_configs, ((5, 7), (7, 7)))
    nested_configs_after_first = parse_configs(args.nested_configs_after_first)
    warm_configs = parse_configs(args.warm_configs)
    stage1_depths = parse_ints(args.stage1_depths, (3, 5))
    baseline_depths = parse_ints(args.baseline_depths, (2, 3, 6))
    baseline_depths_after_first = parse_ints(args.baseline_depths_after_first)
    baseline_algorithms = tuple(x.strip().lower() for x in args.baseline_algorithms.split(",") if x.strip())
    unknown_baselines = set(baseline_algorithms) - {"stdxy", "dxy"}
    if unknown_baselines:
        raise ValueError(f"unknown baseline algorithms: {sorted(unknown_baselines)}")
    preloaded_stage1, preloaded_nested = load_preload_params(
        args.preload_params_csv,
        args.preload_seed,
    )
    preloaded_direct = load_direct_preload_params(
        args.preload_direct_params_csv,
        args.preload_direct_seed,
    )
    trace_seeds = set(parse_ints(args.trace_seeds, seeds[:1]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layout_tag = "x".join(str(k) for k in layout)
    outdir = os.path.join(os.getcwd(), f"bcst_corrected_ru_hyper{args.arity}_N{sum(layout)}_{layout_tag}_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"[outdir] {outdir}", flush=True)

    if args.share_stage1_state and not args.reuse_structural_instance:
        raise ValueError("--share-stage1-state requires --reuse-structural-instance")
    if args.share_direct_params and not args.reuse_structural_instance:
        raise ValueError("--share-direct-params requires --reuse-structural-instance")
    if args.share_nested_params and not args.reuse_structural_instance:
        raise ValueError("--share-nested-params requires --reuse-structural-instance")
    if args.preload_params_csv and not args.share_stage1_state:
        raise ValueError("--preload-params-csv requires --share-stage1-state")
    if args.preload_params_csv and not args.share_nested_params:
        raise ValueError("--preload-params-csv requires --share-nested-params")
    if args.preload_direct_params_csv and not args.share_direct_params:
        raise ValueError("--preload-direct-params-csv requires --share-direct-params")

    rows: List[dict] = []
    summaries: List[dict] = []
    latest_summary: dict = {}
    structural_inst: Optional[HeteroBlockWDSHCCInstance] = None
    shared_stage1_cache: Dict[int, Tuple[torch.Tensor, int, dict]] = {}
    shared_nested_param_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {
        p1: {
            p2: (
                torch.tensor(values["gammas"], dtype=torch.float32),
                torch.tensor(values["betas"], dtype=torch.float32),
            )
            for p2, values in depth_cache.items()
        }
        for p1, depth_cache in preloaded_nested.items()
    }
    shared_direct_param_cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {
        algorithm: {
            depth: (
                torch.tensor(values[0], dtype=torch.float32),
                torch.tensor(values[1], dtype=torch.float32),
            )
            for depth, values in depth_cache.items()
        }
        for algorithm, depth_cache in preloaded_direct.items()
    }
    for seed in seeds:
        seed_started = time.time()
        cfg = HeteroConfig(
            layout=layout,
            weights=weights,
            seed=seed,
            device=args.device,
            dtype=args.dtype,
            conflict_topology=args.conflict_topology,
            penalty_scale=args.penalty_scale,
            nested_stage2_guard_scale=args.nested_guard_scale,
            opt_restarts=args.opt_restarts,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            early_stop_patience=args.early_stop_patience,
            log_every=args.log_every,
            stage1_sequential_restarts=args.stage1_sequential_restarts,
            direct_sequential_restarts=args.direct_sequential_restarts,
            stage2_sequential_restarts=args.stage2_sequential_restarts,
            stage2_sequential_min_depth=args.stage2_sequential_min_depth,
            continuation_init=args.continuation_init,
            trace_every=args.trace_every if seed in trace_seeds else 0,
        )
        device = choose_device(cfg.device)
        print(f"\n[seed {seed}] N={cfg.n_qubits} layout={layout} weights={weights} device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
        structural_reused = bool(args.reuse_structural_instance and structural_inst is not None)
        structural_started = time.time()
        if structural_reused:
            assert structural_inst is not None
            inst = structural_inst
            inst.cfg = cfg
            inst.seed = seed
            inst.rng = np.random.default_rng(seed)
            print("[instance] reusing seed-independent structural shell", flush=True)
        else:
            inst = HeteroBlockWDSHCCInstance(cfg, device=device, seed=seed)
            if args.reuse_structural_instance:
                structural_inst = inst
        structural_elapsed = time.time() - structural_started
        objective_started = time.time()
        obj_ru, obj_meta = install_random_objective(inst, args, seed)
        objective_elapsed = time.time() - objective_started
        objective_term_rows = [
            {
                "seed": seed,
                "term_id": index,
                "qubits": ",".join(str(qubit) for qubit in term),
                "weight": float(weight),
            }
            for index, (term, weight) in enumerate(sorted(inst.objective_terms.items()))
        ]
        objective_terms_file = f"objective_terms_seed{seed}.csv"
        write_csv(os.path.join(outdir, objective_terms_file), objective_term_rows)
        objective_terms_canonical = json.dumps(
            [
                {"term": list(term), "weight": float(weight)}
                for term, weight in sorted(inst.objective_terms.items())
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        obj_meta["objective_terms_file"] = objective_terms_file
        obj_meta["objective_terms_sha256"] = hashlib.sha256(
            objective_terms_canonical
        ).hexdigest()
        runner = HeteroQAOARunner(cfg, inst)
        if preloaded_stage1 and not shared_stage1_cache:
            for p1, values in preloaded_stage1.items():
                gammas = torch.tensor(values["gammas"], dtype=runner.real_dtype, device=device)
                betas = torch.tensor(values["betas"], dtype=runner.real_dtype, device=device)
                with torch.no_grad():
                    energies, states = runner.energy_xy(
                        inst.Cconf,
                        gammas[None, :],
                        betas[None, :],
                        p1,
                    )
                    state = states[0].detach().clone()
                    state /= torch.clamp(torch.linalg.vector_norm(state), min=1e-20)
                    reconstructed_energy = float(energies[0].detach().cpu())
                if not math.isclose(
                    reconstructed_energy,
                    float(values["best_energy"]),
                    rel_tol=2e-5,
                    abs_tol=2e-6,
                ):
                    raise ValueError(
                        f"preloaded stage-1 energy mismatch at p1={p1}: "
                        f"reconstructed {reconstructed_energy}, CSV {values['best_energy']}"
                    )
                shared_stage1_cache[p1] = (
                    state,
                    int(values["iters"]),
                    {
                        "best_energy": reconstructed_energy,
                        "selected_restart": int(values["selected_restart"]),
                        "restart_energies": list(values["restart_energies"]),
                        "gammas": gammas.detach().clone(),
                        "betas": betas.detach().clone(),
                        "stage1_init_source": f"preloaded_seed_{args.preload_seed}_fixed_angles",
                        "stage1_state_reused": False,
                    },
                )
        if args.share_stage1_state:
            for p1, (state, iters, info) in shared_stage1_cache.items():
                runner._stage1_cache[(0, p1)] = (
                    state,
                    iters,
                    {**info, "stage1_state_reused": True},
                )
                runner._stage1_param_cache[p1] = (info["gammas"], info["betas"])
        if args.share_direct_params:
            for algorithm, depth_cache in shared_direct_param_cache.items():
                runner._direct_param_cache[algorithm].update(depth_cache)
        if args.share_nested_params:
            for p1, depth_cache in shared_nested_param_cache.items():
                runner._stage2_param_cache.setdefault(p1, {}).update(depth_cache)
        latest_summary = {
            **inst.summary(),
            "seeds": ",".join(str(s) for s in seeds),
            "target_mode": args.target_mode,
            "topk": args.topk,
            "top_fraction": args.top_fraction,
            "accounting_tag": ACCOUNTING_VERSION,
            "xy_mixer_RU": xy_mixer_ru(inst.block_sizes),
            "ru_model": "full within-block XY pair count with complete history-preparation replay; P0 RU = 0",
            "resource_aggregation": "mean of seed-level integer-RTS times corrected RU",
            "structural_instance_reused": structural_reused,
            "stage1_state_shared_across_seeds": bool(args.share_stage1_state),
            "direct_params_shared_across_seeds": bool(args.share_direct_params),
            "preload_direct_params_csv": (
                os.path.abspath(args.preload_direct_params_csv)
                if args.preload_direct_params_csv
                else ""
            ),
            "preload_direct_seed": (
                args.preload_direct_seed if args.preload_direct_params_csv else None
            ),
            "nested_params_shared_across_seeds": bool(args.share_nested_params),
            "preload_params_csv": os.path.abspath(args.preload_params_csv) if args.preload_params_csv else "",
            "preload_seed": args.preload_seed if args.preload_params_csv else None,
            "preloaded_stage1_depths": ",".join(str(value) for value in sorted(preloaded_stage1)),
            "preloaded_nested_configs": ",".join(
                f"{p1}-{p2}"
                for p1 in sorted(preloaded_nested)
                for p2 in sorted(preloaded_nested[p1])
            ),
            "nested_configs_after_first": ",".join(
                f"{p1}-{p2}" for p1, p2 in nested_configs_after_first
            ),
            "baseline_depths_after_first": ",".join(str(value) for value in baseline_depths_after_first),
            "continuation_initialization": bool(args.continuation_init),
            "optimizer_protocol_tag": "expected-energy-best-checkpoint-config-stable-seed-v4",
            "optimizer_checkpoint_selection": "lowest expected energy encountered per restart",
            "continuation_modes": "identity-preserving append and interpolation",
            "shell_representation": "exact uint64 packed basis in C-order",
            "optimizer_restarts": cfg.opt_restarts,
            "optimizer_max_steps": cfg.opt_steps,
            "optimizer_lr": cfg.opt_lr,
            "optimizer_early_stop_patience": cfg.early_stop_patience,
            "optimizer_trace_every": cfg.trace_every,
            "optimizer_trace_seeds": ",".join(str(value) for value in sorted(trace_seeds)),
            "direct_penalty_scale": cfg.penalty_scale,
            "preload_direct_params_csv": (
                os.path.abspath(args.preload_direct_params_csv)
                if args.preload_direct_params_csv
                else ""
            ),
            "preload_direct_seed": (
                args.preload_direct_seed if args.preload_direct_params_csv else None
            ),
            "structural_build_elapsed_sec": structural_elapsed,
            "objective_build_elapsed_sec": objective_elapsed,
            "stage1_restart_execution": "sequential" if cfg.stage1_sequential_restarts else "batched",
            "direct_restart_execution": "sequential" if cfg.direct_sequential_restarts else "batched",
            "baseline_algorithms": ",".join(baseline_algorithms),
            "stage2_restart_execution": (
                f"sequential_at_p2_ge_{cfg.stage2_sequential_min_depth}"
                if cfg.stage2_sequential_restarts
                else "batched"
            ),
            **obj_meta,
        }
        summaries.append(latest_summary)
        print("[instance]", json.dumps(inst.summary(), allow_nan=True), flush=True)
        print("[objective]", json.dumps(obj_meta, allow_nan=True), flush=True)
        if args.target_mode in ("topfrac", "topfrac_ties"):
            target_policy_tag = f"{args.target_mode}_f{args.top_fraction:.9g}"
        elif args.target_mode == "topk":
            target_policy_tag = f"topk_k{args.topk}"
        else:
            target_policy_tag = args.target_mode
        objective_size_tag = (
            f"density{args.term_density:.9g}"
            if args.term_mode == "uniform_mixed4"
            else f"m{args.term_count}"
        )
        target_meta = {
            "target_tag": f"hetero_random{args.arity}_{args.term_mode}_{args.weight_mode}_{objective_size_tag}_{target_policy_tag}",
            "structural_instance_reused": structural_reused,
            "optimizer_protocol_tag": "expected-energy-best-checkpoint-config-stable-seed-v4",
            "optimizer_checkpoint_selection": "lowest expected energy encountered per restart",
            "continuation_modes": "identity-preserving append and interpolation",
            "shell_representation": "exact uint64 packed basis in C-order",
            "optimizer_restarts": cfg.opt_restarts,
            "optimizer_max_steps": cfg.opt_steps,
            "optimizer_lr": cfg.opt_lr,
            "optimizer_early_stop_patience": cfg.early_stop_patience,
            "direct_penalty_scale": cfg.penalty_scale,
            "continuation_initialization": bool(args.continuation_init),
            "nested_params_shared_across_seeds": bool(args.share_nested_params),
            "preload_params_csv": os.path.abspath(args.preload_params_csv) if args.preload_params_csv else "",
            "preload_seed": args.preload_seed if args.preload_params_csv else None,
            "nested_configs_after_first": ",".join(
                f"{p1}-{p2}" for p1, p2 in nested_configs_after_first
            ),
            "stage1_restart_execution": "sequential" if cfg.stage1_sequential_restarts else "batched",
            "direct_restart_execution": "sequential" if cfg.direct_sequential_restarts else "batched",
            "stage2_restart_execution": (
                f"sequential_at_p2_ge_{cfg.stage2_sequential_min_depth}"
                if cfg.stage2_sequential_restarts
                else "batched"
            ),
            **obj_meta,
            "structural_build_elapsed_sec": structural_elapsed,
            "objective_build_elapsed_sec": objective_elapsed,
        }
        update_outputs(outdir, {**latest_summary, "seed_summaries": summaries}, rows, cfg.target_prob)

        def record(algo: str, params: str, resource_counts: Tuple[int, int], fn) -> None:
            corrected_ru, legacy_ru = resource_counts
            append_trial(rows, runner, seed, algo, params, corrected_ru, legacy_ru, fn, target_meta, cfg.target_prob)
            agg = update_outputs(outdir, {**latest_summary, "seed_summaries": summaries}, rows, cfg.target_prob)
            if runner.trace_rows:
                write_csv(os.path.join(outdir, "optimization_trace.csv"), runner.trace_rows)
            print("[current best]", json.dumps(agg[0], allow_nan=True), flush=True)

        if not args.skip_stage1_output:
            for p1 in stage1_depths:
                record("XY-Stage1-Only", f"p1={p1}", resource_stage1(inst, p1), lambda p1=p1: runner.run_stage1_only(p1, 0))
        active_nested_configs = (
            nested_configs_after_first
            if seed != seeds[0] and nested_configs_after_first
            else nested_configs
        )
        if not args.skip_nested:
            for p1, p2 in active_nested_configs:
                record("XY-Nested", f"({p1},{p2})", resource_nested(inst, p1, p2, obj_ru), lambda p1=p1, p2=p2: runner.run_xy_nested(p1, p2, 0))
        for p1, p2 in warm_configs:
            record(
                "XY-Warm-Same-State",
                f"({p1},{p2})",
                resource_warm(inst, p1, p2, obj_ru),
                lambda p1=p1, p2=p2: runner.run_xy_warm(p1, p2, 0),
            )
        if args.share_stage1_state:
            for (_trial_idx, p1), (state, iters, info) in runner._stage1_cache.items():
                if p1 not in shared_stage1_cache:
                    shared_stage1_cache[p1] = (
                        state,
                        iters,
                        {**info, "stage1_state_reused": False},
                    )
        if args.share_nested_params:
            for p1, depth_cache in runner._stage2_param_cache.items():
                shared_nested_param_cache.setdefault(p1, {}).update(
                    {
                        depth: (gammas.detach().clone(), betas.detach().clone())
                        for depth, (gammas, betas) in depth_cache.items()
                    }
                )
        if not args.skip_baselines:
            active_baseline_depths = (
                baseline_depths_after_first
                if seed != seeds[0] and baseline_depths_after_first
                else baseline_depths
            )
            for p in active_baseline_depths:
                if "stdxy" in baseline_algorithms:
                    record("Std-XY", f"p={p}", resource_xy(inst, p, obj_ru), lambda p=p: runner.run_std_xy(p, 0))
                if "dxy" in baseline_algorithms:
                    record("d-XY", f"p={p}", resource_xy(inst, p, obj_ru), lambda p=p: runner.run_decoupled_xy(p, 0))
            if args.share_direct_params:
                for algorithm, depth_cache in runner._direct_param_cache.items():
                    shared_direct_param_cache[algorithm] = {
                        depth: (gammas.detach().clone(), betas.detach().clone())
                        for depth, (gammas, betas) in depth_cache.items()
                    }
        latest_summary["seed_wall_elapsed_sec"] = time.time() - seed_started
        update_outputs(outdir, {**latest_summary, "seed_summaries": summaries}, rows, cfg.target_prob)
        if args.reuse_structural_instance:
            del runner
            if device.type == "cuda":
                torch.cuda.empty_cache()

    final_summary = {
        **latest_summary,
        "seed_summaries": summaries,
        "validation_note": "Rows are written incrementally; manuscript cost is the mean of seed-level integer-RTS costs.",
        "run_wall_elapsed_sec": time.time() - run_started,
    }
    agg = update_outputs(outdir, final_summary, rows, HeteroConfig(layout, weights).target_prob)
    print("\n[final best]")
    if agg:
        print(json.dumps(agg[0], indent=2, allow_nan=True))
    else:
        print("no algorithm rows requested")
    print(f"[done] {outdir}", flush=True)
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True, help="Comma-separated block sizes, e.g. 6,6,6,6,6,3")
    parser.add_argument("--weights", default="", help="Comma-separated block weights; default min(2,K-1) per block")
    parser.add_argument("--seeds", default=os.environ.get("HETERO_RAND_SEEDS", "531"))
    parser.add_argument("--device", default=os.environ.get("HETERO_DEVICE", "auto"))
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default=os.environ.get("HETERO_DTYPE", "complex64"))
    parser.add_argument("--conflict-topology", choices=("cycle", "path", "none"), default=os.environ.get("HETERO_CONFLICT_TOPOLOGY", "cycle"))
    parser.add_argument("--arity", type=int, default=int(os.environ.get("HETERO_RAND_ARITY", "4")))
    parser.add_argument("--term-count", type=int, default=int(os.environ.get("HETERO_RAND_TERM_COUNT", "360")))
    parser.add_argument("--term-mode", choices=("cross", "cross4", "pair2", "mixed", "uniform_mixed4"), default=os.environ.get("HETERO_RAND_TERM_MODE", "mixed"))
    parser.add_argument("--term-density", type=float, default=float(os.environ.get("HETERO_RAND_TERM_DENSITY", "0.0")))
    parser.add_argument("--weight-mode", choices=("positive_exp", "positive_integer", "signed_normal", "signed_integer"), default=os.environ.get("HETERO_RAND_WEIGHT_MODE", "positive_integer"))
    parser.add_argument("--weight-scale", type=float, default=float(os.environ.get("HETERO_RAND_WEIGHT_SCALE", "1.0")))
    parser.add_argument("--integer-max", type=int, default=int(os.environ.get("HETERO_RAND_INTEGER_MAX", "3")))
    parser.add_argument("--min-feasible-support", type=int, default=int(os.environ.get("HETERO_RAND_MIN_FEASIBLE_SUPPORT", "1")))
    parser.add_argument(
        "--target-mode",
        choices=("exact", "topk", "topfrac", "topfrac_ties"),
        default=os.environ.get("HETERO_RAND_TARGET_MODE", "topk"),
    )
    parser.add_argument("--topk", type=int, default=int(os.environ.get("HETERO_RAND_TOPK", "64")))
    parser.add_argument("--top-fraction", type=float, default=float(os.environ.get("HETERO_RAND_TOP_FRACTION", "0.01")))
    parser.add_argument("--ru-per-term", type=int, default=int(os.environ.get("HETERO_RAND_RU_PER_TERM", "0")))
    parser.add_argument("--nested-configs", default=os.environ.get("HETERO_RAND_NESTED_CONFIGS", "5-7,7-7"))
    parser.add_argument(
        "--nested-configs-after-first",
        default=os.environ.get("HETERO_RAND_NESTED_CONFIGS_AFTER_FIRST", ""),
        help="Optional frozen config list for later seeds after seed-one continuation tuning",
    )
    parser.add_argument("--warm-configs", default=os.environ.get("HETERO_RAND_WARM_CONFIGS", ""))
    parser.add_argument("--stage1-depths", default=os.environ.get("HETERO_RAND_STAGE1_DEPTHS", "3,5"))
    parser.add_argument("--baseline-depths", default=os.environ.get("HETERO_RAND_BASELINE_DEPTHS", "2,3,6"))
    parser.add_argument(
        "--baseline-depths-after-first",
        default=os.environ.get("HETERO_RAND_BASELINE_DEPTHS_AFTER_FIRST", ""),
        help="Optional reduced depth list for later seeds when transferred direct parameters are available",
    )
    parser.add_argument(
        "--baseline-algorithms",
        default=os.environ.get("HETERO_RAND_BASELINE_ALGORITHMS", "stdxy,dxy"),
        help="Comma-separated subset of stdxy,dxy",
    )
    parser.add_argument("--opt-restarts", type=int, default=int(os.environ.get("HETERO_RAND_OPT_RESTARTS", "3")))
    parser.add_argument("--opt-steps", type=int, default=int(os.environ.get("HETERO_RAND_OPT_STEPS", "85")))
    parser.add_argument("--opt-lr", type=float, default=float(os.environ.get("HETERO_RAND_OPT_LR", "0.055")))
    parser.add_argument("--early-stop-patience", type=int, default=int(os.environ.get("HETERO_RAND_EARLY_STOP_PATIENCE", "18")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("HETERO_RAND_LOG_EVERY", "30")))
    parser.add_argument("--trace-every", type=int, default=int(os.environ.get("HETERO_RAND_TRACE_EVERY", "0")))
    parser.add_argument("--trace-seeds", default=os.environ.get("HETERO_RAND_TRACE_SEEDS", ""))
    parser.add_argument("--stage2-sequential-restarts", action="store_true")
    parser.add_argument("--stage2-sequential-min-depth", type=int, default=1)
    parser.add_argument("--stage1-sequential-restarts", action="store_true")
    parser.add_argument("--direct-sequential-restarts", action="store_true")
    parser.add_argument("--reuse-structural-instance", action="store_true")
    parser.add_argument("--share-stage1-state", action="store_true")
    parser.add_argument("--share-nested-params", action="store_true")
    parser.add_argument("--share-direct-params", action="store_true")
    parser.add_argument(
        "--preload-params-csv",
        default="",
        help="Seed-tuning trial_results.csv used to freeze stage-1 states and initialize nested angles",
    )
    parser.add_argument("--preload-seed", type=int, default=100)
    parser.add_argument(
        "--preload-direct-params-csv",
        default="",
        help="Seed-tuning trial_results.csv used to initialize fixed direct depths",
    )
    parser.add_argument("--preload-direct-seed", type=int, default=100)
    parser.add_argument("--continuation-init", action="store_true")
    parser.add_argument("--skip-stage1-output", action="store_true")
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--penalty-scale", type=float, default=float(os.environ.get("HETERO_RAND_PENALTY_SCALE", "7.0")))
    parser.add_argument("--nested-guard-scale", type=float, default=float(os.environ.get("HETERO_RAND_NESTED_GUARD_SCALE", "7.0")))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
