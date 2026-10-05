#!/usr/bin/env python3
"""25-qubit generalized-block edge-WDS-HCC XY-nested QAOA experiment.

This experiment replaces one-hot constraints with local r-of-K block
cardinality constraints.  The simulator never materializes the full 2**N
Hilbert space.  It enumerates the product of local fixed-weight block bases and
therefore supports the default 25 physical qubits:

    M = 5 blocks, K = 5 qubits per block, block_weight = 2
    subspace dimension = binom(5, 2)**5 = 100000

The first-stage XY mixer is the Johnson-graph XY mixer inside each fixed-weight
block.  Stage 2 is a history-state nested-QAOA mixer around the optimized
conflict-feasible state, following the Nested QAOA template paper.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def ncr(n: int, r: int) -> int:
    return math.comb(n, r)


@dataclass
class Config:
    # Generalized local block constraint: exactly block_weight selected in each block.
    n_qubits: int = 25
    n_blocks: int = 5
    block_size: int = 5
    block_weight: int = 2

    seed: int = 531
    seeds: Tuple[int, ...] = (531, 532, 533, 534)
    run_label: str = "N25_r2of5_edge"
    outdir: Optional[str] = None
    device: str = "auto"
    dtype: str = "complex64"

    # Instance construction.
    objective: str = "edge"
    edge_density: float = 1.0
    edge_weight_scale: float = 1.0
    edge_trap_multiplier: float = 2.5
    edge_trap_block_a: Tuple[int, ...] = (0,)
    edge_trap_block_b: Tuple[int, ...] = (1,)
    edge_planted_boost: float = 0.0
    edge_decoy_count: int = 0
    edge_decoy_boost: float = 0.0
    edge_decoy_pair_prob: float = 0.35
    hyperedge_sparsity: float = 0.22
    planted_strength: float = 2.4
    noise_strength: float = 0.30
    decoy_strength: float = 3.4
    decoy_count: int = 18
    conflict_weight_scale: float = 1.0
    penalty_scale: float = 7.0
    nested_stage2_guard_scale: float = 7.0
    warm_final_guard_scale: float = 7.0
    cyclic_conflicts: bool = True
    conflict_topology: str = "cycle"
    extra_conflict_prob: float = 0.0
    structured_extra_offsets: Tuple[int, ...] = ()

    # Optimizer.
    opt_restarts: int = 8
    opt_steps: int = 110
    opt_lr: float = 0.055
    early_stop_patience: int = 14
    early_stop_tol: float = 1e-6
    log_every: int = 25
    trials: int = 2
    target_prob: float = 0.99

    # Depth grids.
    xy_nested_configs: Tuple[Tuple[int, int], ...] = ((2, 3), (3, 3), (4, 3), (5, 3), (5, 4))
    xy_3stage_configs: Tuple[Tuple[int, int, int], ...] = ((3, 2, 2), (4, 2, 2), (5, 2, 2), (5, 3, 2))
    xy_3stage_warm_configs: Tuple[Tuple[int, int, int], ...] = ((2, 3, 2), (2, 3, 3), (4, 3, 2), (4, 3, 3))
    stage1_depths: Tuple[int, ...] = ()
    std_xy_depths: Tuple[int, ...] = (2, 4, 6, 8)
    decoupled_xy_depths: Tuple[int, ...] = (2, 4, 6, 8)
    grover_depths: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    # Classical sanity baselines.
    random_samples: int = 20000
    local_search_restarts: int = 250
    local_search_steps: int = 160

    # Runtime switches.
    run_xy_nested: bool = True
    run_xy_3stage: bool = False
    run_xy_3stage_warm: bool = False
    run_stage1_only: bool = True
    run_guarded_xy_nested: bool = True
    run_std_xy: bool = True
    run_decoupled_xy: bool = True
    # This is an oracle-style upper-bound diagnostic, not a realistic baseline.
    run_grover_feasible: bool = False
    run_classical_baselines: bool = True
    no_plots: bool = False


def apply_mode(cfg: Config, mode: str) -> Config:
    if mode == "pilot":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=1,
            opt_restarts=4,
            opt_steps=45,
            early_stop_patience=8,
            log_every=15,
            xy_nested_configs=((1, 2), (2, 2), (2, 3)),
            std_xy_depths=(1, 2, 3),
            decoupled_xy_depths=(1, 2, 3),
            grover_depths=(1, 2, 3),
            random_samples=6000,
            local_search_restarts=80,
            local_search_steps=90,
        )
    if mode == "sweep":
        return cfg
    if mode == "tri":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=2,
            opt_restarts=8,
            opt_steps=140,
            early_stop_patience=24,
            log_every=35,
            xy_nested_configs=((5, 3), (7, 3), (7, 4)),
            xy_3stage_configs=((2, 2, 2), (3, 2, 2), (4, 2, 2), (5, 2, 2), (5, 3, 2), (5, 3, 3), (7, 2, 2), (7, 3, 2)),
            stage1_depths=(3, 5, 7),
            std_xy_depths=(2, 4),
            decoupled_xy_depths=(2, 4),
            run_xy_3stage=True,
            run_guarded_xy_nested=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "tri_edge3":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=2,
            opt_restarts=8,
            opt_steps=120,
            early_stop_patience=24,
            log_every=35,
            xy_nested_configs=(),
            xy_3stage_configs=((2, 2, 3), (2, 3, 3), (3, 2, 3), (3, 3, 3), (4, 2, 3), (4, 3, 3)),
            stage1_depths=(2, 3, 4),
            std_xy_depths=(),
            decoupled_xy_depths=(),
            run_xy_nested=False,
            run_xy_3stage=True,
            run_guarded_xy_nested=False,
            run_std_xy=False,
            run_decoupled_xy=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "tri_cheap":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=2,
            opt_restarts=8,
            opt_steps=120,
            early_stop_patience=24,
            log_every=35,
            xy_nested_configs=(),
            xy_3stage_configs=((2, 2, 1), (2, 3, 1), (3, 2, 1), (3, 3, 1), (2, 2, 2), (2, 3, 2), (3, 2, 2), (3, 3, 2)),
            stage1_depths=(2, 3),
            std_xy_depths=(),
            decoupled_xy_depths=(),
            run_xy_nested=False,
            run_xy_3stage=True,
            run_guarded_xy_nested=False,
            run_std_xy=False,
            run_decoupled_xy=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "tri_warm":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=2,
            opt_restarts=10,
            opt_steps=150,
            early_stop_patience=28,
            log_every=35,
            xy_nested_configs=(),
            xy_3stage_configs=(),
            xy_3stage_warm_configs=((2, 2, 2), (2, 3, 2), (2, 3, 3), (3, 3, 2), (4, 2, 2), (4, 3, 2), (4, 3, 3), (4, 3, 4)),
            stage1_depths=(2, 3, 4),
            std_xy_depths=(2, 4, 6),
            decoupled_xy_depths=(2, 4, 6),
            run_xy_nested=False,
            run_xy_3stage=False,
            run_xy_3stage_warm=True,
            run_guarded_xy_nested=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "focused":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=3,
            opt_restarts=12,
            opt_steps=150,
            early_stop_patience=18,
            xy_nested_configs=((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)),
            std_xy_depths=(2, 4, 6, 8),
            decoupled_xy_depths=(2, 4, 6, 8),
            grover_depths=(2, 3, 4, 5, 6, 7),
        )
    if mode == "ridge":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=3,
            opt_restarts=14,
            opt_steps=240,
            early_stop_patience=36,
            log_every=40,
            xy_nested_configs=((5, 4), (6, 4), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (9, 3), (9, 4)),
            run_guarded_xy_nested=False,
            run_std_xy=False,
            run_decoupled_xy=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "ridge_hi":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=2,
            opt_restarts=16,
            opt_steps=260,
            early_stop_patience=42,
            log_every=40,
            xy_nested_configs=((9, 4), (9, 5), (10, 4), (10, 5), (11, 4), (11, 5)),
            run_guarded_xy_nested=False,
            run_std_xy=False,
            run_decoupled_xy=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "validate":
        return replace(
            cfg,
            seeds=(531, 532, 533),
            trials=2,
            opt_restarts=10,
            opt_steps=180,
            early_stop_patience=30,
            log_every=45,
            xy_nested_configs=((7, 4), (10, 4), (11, 4)),
            std_xy_depths=(2, 4, 6),
            decoupled_xy_depths=(2, 4, 6),
            run_guarded_xy_nested=False,
            run_grover_feasible=False,
            run_classical_baselines=False,
        )
    if mode == "deep":
        return replace(
            cfg,
            seeds=(cfg.seed,),
            trials=3,
            opt_restarts=18,
            opt_steps=260,
            early_stop_patience=30,
            log_every=40,
            xy_nested_configs=((2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (5, 4), (6, 4), (8, 4)),
            std_xy_depths=(3, 4, 5, 6, 7, 8),
            decoupled_xy_depths=(3, 4, 5, 6, 7, 8),
            grover_depths=(3, 4, 5, 6, 7, 8),
            local_search_restarts=250,
            local_search_steps=160,
        )
    raise ValueError(f"unknown mode {mode!r}")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def write_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class BlockWDSHCCInstance:
    """WDS-HCC instance restricted to the exact local r-of-K block subspace."""

    def __init__(self, cfg: Config, device: torch.device, seed: int) -> None:
        if cfg.n_qubits != cfg.n_blocks * cfg.block_size:
            raise ValueError("n_qubits must equal n_blocks * block_size")
        if not (0 < cfg.block_weight < cfg.block_size):
            raise ValueError("block_weight must be between 1 and block_size - 1")
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.real_dtype = torch.float32 if cfg.dtype == "complex64" else torch.float64
        self.complex_dtype = torch.complex64 if cfg.dtype == "complex64" else torch.complex128

        self.N = cfg.n_qubits
        self.M = cfg.n_blocks
        self.K = cfg.block_size
        self.R = cfg.block_weight
        self.local_patterns = self._build_local_patterns()
        self.L = self.local_patterns.shape[0]
        self.dim = self.L ** self.M

        self.choice_table = self._build_choice_table()
        self.x_np = self._build_x_matrix()
        self.x = torch.tensor(self.x_np, dtype=self.real_dtype, device=self.device)

        self.path_conflict_edges = self._build_conflict_edges("path", include_extra=False)
        self.cycle_conflict_edges = self._build_conflict_edges("cycle", include_extra=False)
        self.conflict_edges = self._build_conflict_edges()
        self.Cpath_np = self._build_conflict_cost_np(self.path_conflict_edges)
        self.Ccycle_np = self._build_conflict_cost_np(self.cycle_conflict_edges)
        self.Cconf_np = self._build_conflict_cost_np(self.conflict_edges)
        self.Cpath = torch.tensor(self.Cpath_np, dtype=self.real_dtype, device=self.device)
        self.Ccycle = torch.tensor(self.Ccycle_np, dtype=self.real_dtype, device=self.device)
        self.Cconf = torch.tensor(self.Cconf_np, dtype=self.real_dtype, device=self.device)

        self.planted_bits = self._choose_planted_feasible_bits()
        self.decoy_bits: List[np.ndarray] = self._choose_decoy_bits(int(cfg.edge_decoy_count))
        self.edge_weights = self._build_edge_weights()
        self.hyperedge_weights: Dict[Tuple[int, int, int], float] = {}
        self.Cobj_np = self._build_objective_cost_np()
        self.Cobj = torch.tensor(self.Cobj_np, dtype=self.real_dtype, device=self.device)

        self.feasible_mask_np = self.Cconf_np <= 1e-9
        self.path_feasible_mask_np = self.Cpath_np <= 1e-9
        self.cycle_feasible_mask_np = self.Ccycle_np <= 1e-9
        if int(self.feasible_mask_np.sum()) == 0:
            raise RuntimeError("no conflict-feasible block states found")
        feasible_obj = self.Cobj_np[self.feasible_mask_np]
        self.min_feasible_obj = float(feasible_obj.min())
        self.opt_mask_np = self.feasible_mask_np & (np.abs(self.Cobj_np - self.min_feasible_obj) <= 1e-7)
        self.feasible_mask = torch.tensor(self.feasible_mask_np, dtype=torch.bool, device=self.device)
        self.path_feasible_mask = torch.tensor(self.path_feasible_mask_np, dtype=torch.bool, device=self.device)
        self.cycle_feasible_mask = torch.tensor(self.cycle_feasible_mask_np, dtype=torch.bool, device=self.device)
        self.opt_mask = torch.tensor(self.opt_mask_np, dtype=torch.bool, device=self.device)

        self.psi_block = torch.ones(self.dim, dtype=self.complex_dtype, device=self.device)
        self.psi_block /= math.sqrt(self.dim)
        self.psi_feasible = torch.zeros(self.dim, dtype=self.complex_dtype, device=self.device)
        feasible_count = int(self.feasible_mask_np.sum())
        self.psi_feasible[self.feasible_mask] = 1.0 / math.sqrt(feasible_count)

        self.local_evals, self.local_evecs = self._build_local_xy_eigendecomp()

    def _build_local_patterns(self) -> np.ndarray:
        patterns = []
        for comb in itertools.combinations(range(self.K), self.R):
            row = np.zeros(self.K, dtype=np.int8)
            row[list(comb)] = 1
            patterns.append(row)
        return np.array(patterns, dtype=np.int8)

    def _build_choice_table(self) -> np.ndarray:
        grids = np.meshgrid(*([np.arange(self.L, dtype=np.int16)] * self.M), indexing="ij")
        return np.stack([g.reshape(-1) for g in grids], axis=1)

    def _build_x_matrix(self) -> np.ndarray:
        blocks = [self.local_patterns[self.choice_table[:, b]] for b in range(self.M)]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _build_conflict_edges(self, topology: Optional[str] = None, include_extra: bool = True) -> List[Tuple[int, int, float]]:
        edges: Dict[Tuple[int, int], float] = {}
        topology = self.cfg.conflict_topology if topology is None else topology
        if topology == "cycle" and not self.cfg.cyclic_conflicts:
            topology = "none"
        if topology in ("cycle", "path"):
            block_range = range(self.M) if topology == "cycle" else range(self.M - 1)
            for b in block_range:
                nb = (b + 1) % self.M
                for local in range(self.K):
                    u = b * self.K + local
                    v = nb * self.K + local
                    edges[tuple(sorted((u, v)))] = self.cfg.conflict_weight_scale
        elif topology != "none":
            raise ValueError(f"unknown conflict_topology {topology!r}")
        if include_extra:
            for raw_offset in self.cfg.structured_extra_offsets:
                offset = int(raw_offset) % self.K
                for b in range(self.M):
                    nb = (b + 2) % self.M
                    if nb == b:
                        continue
                    for i in range(self.K):
                        u = b * self.K + i
                        v = nb * self.K + ((i + offset) % self.K)
                        edges[tuple(sorted((u, v)))] = self.cfg.conflict_weight_scale
            for b1 in range(self.M):
                for b2 in range(b1 + 1, self.M):
                    if (b2 - b1) % self.M in (1, self.M - 1):
                        continue
                    for i in range(self.K):
                        for j in range(self.K):
                            if self.rng.random() < self.cfg.extra_conflict_prob:
                                u = b1 * self.K + i
                                v = b2 * self.K + j
                                edges[tuple(sorted((u, v)))] = self.cfg.conflict_weight_scale
        return [(u, v, w) for (u, v), w in sorted(edges.items())]

    def _build_conflict_cost_np(self, edges: Sequence[Tuple[int, int, float]]) -> np.ndarray:
        c = np.zeros(self.dim, dtype=np.float32)
        for u, v, w in edges:
            c += np.float32(w) * self.x_np[:, u] * self.x_np[:, v]
        return c

    def _choose_planted_feasible_bits(self) -> np.ndarray:
        feasible_idx = np.where(self.Cconf_np <= 1e-9)[0]
        # Prefer a state with low conflict neighborhood but otherwise random.
        idx = int(self.rng.choice(feasible_idx))
        return self.x_np[idx].astype(np.int8)

    def _sample_block_state(self) -> np.ndarray:
        choices = self.rng.integers(0, self.L, size=self.M)
        return np.concatenate([self.local_patterns[c] for c in choices]).astype(np.int8)

    def _choose_decoy_bits(self, count: int) -> List[np.ndarray]:
        decoys: List[np.ndarray] = []
        if count <= 0:
            return decoys
        attempts = 0
        planted = tuple(self.planted_bits.tolist())
        while len(decoys) < count and attempts < 20000:
            attempts += 1
            z = self._sample_block_state()
            if tuple(z.tolist()) == planted:
                continue
            conflict = 0.0
            for u, v, w in self.conflict_edges:
                conflict += w * float(z[u] * z[v])
            if conflict > 0:
                decoys.append(z)
        return decoys

    def _triples_from_selected(self, bits: np.ndarray) -> List[Tuple[int, int, int]]:
        selected = np.where(bits > 0)[0].tolist()
        return [tuple(t) for t in itertools.combinations(selected, 3)]

    def _build_hyperedge_weights(self) -> Dict[Tuple[int, int, int], float]:
        weights: Dict[Tuple[int, int, int], float] = {}
        for triple in itertools.combinations(range(self.N), 3):
            if self.rng.random() < self.cfg.hyperedge_sparsity:
                weights[triple] = float(self.cfg.noise_strength * self.rng.normal())

        for triple in self._triples_from_selected(self.planted_bits):
            weights[triple] = weights.get(triple, 0.0) + float(self.cfg.planted_strength)

        for decoy in self.decoy_bits:
            for triple in self._triples_from_selected(decoy):
                if self.rng.random() < 0.45:
                    weights[triple] = weights.get(triple, 0.0) + float(self.cfg.decoy_strength)
        return weights

    def _build_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """Paper-style 2-local weighted edge objective.

        We draw random weights on all qubit pairs and optionally strengthen a
        block-pair trap, analogous to the paper's 18-qubit false-minimum block.
        The objective minimized later is -sum w_ij x_i x_j.
        """
        weights: Dict[Tuple[int, int], float] = {}
        scale = float(self.cfg.edge_weight_scale)
        density = float(self.cfg.edge_density)
        if not (0.0 <= density <= 1.0):
            raise ValueError("edge_density must be in [0, 1]")
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if density >= 1.0 or self.rng.random() < density:
                    w = float(scale * self.rng.uniform(-1.0, 1.0))
                    if abs(w) > 1e-12:
                        weights[(i, j)] = w

        trap_a = set()
        trap_b = set()
        for b in self.cfg.edge_trap_block_a:
            trap_a.update(range(int(b) * self.K, (int(b) + 1) * self.K))
        for b in self.cfg.edge_trap_block_b:
            trap_b.update(range(int(b) * self.K, (int(b) + 1) * self.K))
        mult = float(self.cfg.edge_trap_multiplier)
        if mult != 1.0:
            for i in trap_a:
                for j in trap_b:
                    if i == j:
                        continue
                    key = tuple(sorted((i, j)))
                    weights[key] = weights.get(key, 0.0) * mult

        boost = float(self.cfg.edge_planted_boost)
        if boost != 0.0:
            selected = np.where(self.planted_bits > 0)[0].tolist()
            for a, i in enumerate(selected):
                for j in selected[a + 1:]:
                    key = tuple(sorted((i, j)))
                    weights[key] = weights.get(key, 0.0) + boost

        decoy_boost = float(self.cfg.edge_decoy_boost)
        decoy_pair_prob = float(self.cfg.edge_decoy_pair_prob)
        if decoy_boost != 0.0 and self.decoy_bits:
            for decoy in self.decoy_bits:
                selected = np.where(decoy > 0)[0].tolist()
                for a, i in enumerate(selected):
                    for j in selected[a + 1:]:
                        if self.rng.random() <= decoy_pair_prob:
                            key = tuple(sorted((i, j)))
                            weights[key] = weights.get(key, 0.0) + decoy_boost
        weights = {key: val for key, val in weights.items() if abs(val) > 1e-12}
        return weights

    def _build_objective_cost_np(self) -> np.ndarray:
        c = np.zeros(self.dim, dtype=np.float32)
        for (i, j), w in self.edge_weights.items():
            c -= np.float32(w) * self.x_np[:, i] * self.x_np[:, j]
        return c

    def _build_local_xy_eigendecomp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h = np.zeros((self.L, self.L), dtype=np.float32)
        for a, pa in enumerate(self.local_patterns):
            for b, pb in enumerate(self.local_patterns):
                if a < b and int(np.sum(np.abs(pa - pb))) == 2:
                    h[a, b] = 1.0
                    h[b, a] = 1.0
        ht = torch.tensor(h, dtype=self.real_dtype, device=self.device)
        evals, evecs = torch.linalg.eigh(ht)
        return evals, evecs.to(self.complex_dtype)

    @property
    def objective_ru(self) -> int:
        return len(self.edge_weights)

    @property
    def conflict_ru(self) -> int:
        return len(self.conflict_edges)

    @property
    def path_conflict_ru(self) -> int:
        return len(self.path_conflict_edges)

    @property
    def cycle_conflict_ru(self) -> int:
        return len(self.cycle_conflict_edges)

    @property
    def mcz_ru(self) -> int:
        return self.N * self.N

    @property
    def feasible_count(self) -> int:
        return int(self.feasible_mask_np.sum())

    @property
    def optimum_count(self) -> int:
        return int(self.opt_mask_np.sum())

    def resource_xy(self, p: int, decoupled: bool = False) -> int:
        if decoupled:
            return p * (self.conflict_ru + self.objective_ru)
        return p * (self.conflict_ru + self.objective_ru)

    def resource_xy_nested(self, p1: int, p2: int) -> int:
        # Stage-2 history mixer uses preparation + inverse overhead from stage 1.
        stage1 = p1 * self.conflict_ru
        history = 2 * stage1 + self.mcz_ru
        return int(stage1 + p2 * (self.objective_ru + history))

    def resource_xy_3stage(self, p_path: int, p_cycle: int, p_obj: int) -> int:
        stage1 = p_path * self.path_conflict_ru
        history1 = 2 * stage1 + self.mcz_ru
        stage2 = p_cycle * (self.cycle_conflict_ru + history1)
        prep2 = stage1 + stage2
        history2 = 2 * prep2 + self.mcz_ru
        stage3 = p_obj * (self.objective_ru + history2)
        return int(prep2 + stage3)

    def resource_xy_3stage_warm(self, p_path: int, p_cycle: int, p_obj: int) -> int:
        stage1 = p_path * self.path_conflict_ru
        history1 = 2 * stage1 + self.mcz_ru
        stage2 = p_cycle * (self.cycle_conflict_ru + history1)
        prep2 = stage1 + stage2
        guard = self.cycle_conflict_ru if self.cfg.warm_final_guard_scale != 0.0 else 0
        stage3 = p_obj * (self.objective_ru + guard)
        return int(prep2 + stage3)

    def resource_stage1_only(self, p1: int) -> int:
        return int(p1 * self.conflict_ru)

    def resource_guarded_xy_nested(self, p1: int, p2: int) -> int:
        stage1 = p1 * self.conflict_ru
        history = 2 * stage1 + self.mcz_ru
        return int(stage1 + p2 * (self.objective_ru + self.conflict_ru + history))

    def resource_grover_feasible(self, p: int) -> int:
        # Idealized known feasible-state Grover mixer baseline.
        return int(p * (self.objective_ru + self.mcz_ru))

    def summary(self) -> dict:
        planted_idx = self.index_of_bits(self.planted_bits)
        planted_obj = float(self.Cobj_np[planted_idx]) if planted_idx is not None else float("nan")
        return {
            "seed": self.seed,
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "block_weight": self.R,
            "local_basis_size": self.L,
            "subspace_dim": self.dim,
            "full_hilbert_dim": 2 ** self.N,
            "compression_factor": float((2 ** self.N) / self.dim),
            "conflict_edges": len(self.conflict_edges),
            "conflict_topology": self.cfg.conflict_topology,
            "structured_extra_offsets": list(self.cfg.structured_extra_offsets),
            "path_conflict_edges": len(self.path_conflict_edges),
            "cycle_conflict_edges": len(self.cycle_conflict_edges),
            "objective_edges": len(self.edge_weights),
            "edge_density": self.cfg.edge_density,
            "edge_trap_multiplier": self.cfg.edge_trap_multiplier,
            "edge_planted_boost": self.cfg.edge_planted_boost,
            "edge_decoy_count": self.cfg.edge_decoy_count,
            "edge_decoy_boost": self.cfg.edge_decoy_boost,
            "edge_decoy_pair_prob": self.cfg.edge_decoy_pair_prob,
            "objective_RU": self.objective_ru,
            "conflict_RU": self.conflict_ru,
            "path_conflict_RU": self.path_conflict_ru,
            "cycle_conflict_RU": self.cycle_conflict_ru,
            "MCZ_RU": self.mcz_ru,
            "feasible_count": self.feasible_count,
            "path_feasible_count": int(self.path_feasible_mask_np.sum()),
            "cycle_feasible_count": int(self.cycle_feasible_mask_np.sum()),
            "optimum_count": self.optimum_count,
            "feasible_fraction": self.feasible_count / self.dim,
            "min_feasible_obj": self.min_feasible_obj,
            "planted_obj": planted_obj,
            "planted_is_optimal": bool(self.opt_mask_np[planted_idx]) if planted_idx is not None else False,
            "planted_bitstring": "".join(str(int(x)) for x in self.planted_bits.tolist()),
            "decoy_count": len(self.decoy_bits),
        }

    def index_of_bits(self, bits: np.ndarray) -> Optional[int]:
        choices: List[int] = []
        for b in range(self.M):
            block = bits[b * self.K:(b + 1) * self.K]
            matches = np.where((self.local_patterns == block).all(axis=1))[0]
            if len(matches) != 1:
                return None
            choices.append(int(matches[0]))
        idx = 0
        for c in choices:
            idx = idx * self.L + c
        return idx


class QAOARunner:
    def __init__(self, cfg: Config, inst: BlockWDSHCCInstance) -> None:
        self.cfg = cfg
        self.inst = inst
        self.device = inst.device
        self.real_dtype = inst.real_dtype
        self.complex_dtype = inst.complex_dtype
        self.penalty = cfg.penalty_scale
        self._xy_stage1_cache: Dict[Tuple[object, ...], Tuple[torch.Tensor, int, dict]] = {}

    @staticmethod
    def _stable_label_seed(label: str) -> int:
        return sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % 1000003

    def rts(self, p: float) -> float:
        if p >= self.cfg.target_prob:
            return 1.0
        if p <= 1e-14 or not np.isfinite(p):
            return float("inf")
        return float(math.ceil(math.log(1.0 - self.cfg.target_prob) / math.log(1.0 - p)))

    def apply_xy_mixer(self, psi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        batch = psi.shape[0]
        phases = torch.exp(-1j * beta[:, None].to(self.real_dtype) * self.inst.local_evals[None, :])
        u_local = self.inst.local_evecs[None, :, :] @ torch.diag_embed(phases.to(self.complex_dtype)) @ self.inst.local_evecs.T.conj()[None, :, :]
        shaped = psi.reshape([batch] + [self.inst.L] * self.inst.M)
        letters = "cdefghijklmnopqrstuvw"
        dims = list(letters[:self.inst.M])
        for b in range(self.inst.M):
            in_dims = dims.copy()
            out_dims = dims.copy()
            in_dims[b] = "y"
            out_dims[b] = "x"
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = gammas.shape[0]
        psi = self.inst.psi_block[None, :].expand(batch, -1).clone()
        if split is None:
            for d in range(p):
                psi = psi * torch.exp(-1j * gammas[:, d:d + 1].to(self.real_dtype) * h_cost[None, :]).to(self.complex_dtype)
                psi = self.apply_xy_mixer(psi, betas[:, d])
            h_eval = h_cost
        else:
            h_pen, h_obj = split
            for d in range(p):
                psi = psi * torch.exp(-1j * gammas[:, d:d + 1].to(self.real_dtype) * h_pen[None, :]).to(self.complex_dtype)
                psi = psi * torch.exp(-1j * gammas[:, p + d:p + d + 1].to(self.real_dtype) * h_obj[None, :]).to(self.complex_dtype)
                psi = self.apply_xy_mixer(psi, betas[:, d])
            h_eval = h_pen + h_obj
        energy = torch.real(torch.sum(psi.conj() * h_eval[None, :] * psi, dim=1))
        return energy, psi

    def energy_xy_from(
        self,
        phi: torch.Tensor,
        h_phase: torch.Tensor,
        h_eval: torch.Tensor,
        gammas: torch.Tensor,
        betas: torch.Tensor,
        p: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = gammas.shape[0]
        psi = phi[None, :].expand(batch, -1).clone()
        for d in range(p):
            psi = psi * torch.exp(-1j * gammas[:, d:d + 1].to(self.real_dtype) * h_phase[None, :]).to(self.complex_dtype)
            psi = self.apply_xy_mixer(psi, betas[:, d])
        energy = torch.real(torch.sum(psi.conj() * h_eval[None, :] * psi, dim=1))
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
        psi = phi[None, :].expand(batch, -1).clone()
        for d in range(p):
            psi = psi * torch.exp(-1j * gammas[:, d:d + 1].to(self.real_dtype) * h_cost[None, :]).to(self.complex_dtype)
            psi = self.apply_history_mixer(psi, betas[:, d], phi)
        energy = torch.real(torch.sum(psi.conj() * h_cost[None, :] * psi, dim=1))
        return energy, psi

    def optimize(
        self,
        energy_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        p_gamma: int,
        p_beta: int,
        label: str,
        trial_idx: int,
    ) -> Tuple[torch.Tensor, int, dict]:
        torch.manual_seed(int(self.inst.seed * 100000 + trial_idx * 1009 + self._stable_label_seed(label)))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.inst.seed * 100000 + trial_idx * 1009 + self._stable_label_seed(label)))
        g = torch.empty((self.cfg.opt_restarts, p_gamma), dtype=self.real_dtype, device=self.device).uniform_(-math.pi, math.pi)
        b = torch.empty((self.cfg.opt_restarts, p_beta), dtype=self.real_dtype, device=self.device).uniform_(-math.pi, math.pi)
        if self.cfg.opt_restarts > 0:
            g[0] = torch.linspace(0.05, 0.65, p_gamma, dtype=self.real_dtype, device=self.device)
            b[0] = torch.linspace(0.55, 0.05, p_beta, dtype=self.real_dtype, device=self.device)
        g.requires_grad_(True)
        b.requires_grad_(True)
        opt = torch.optim.Adam([g, b], lr=self.cfg.opt_lr)
        best_energy = float("inf")
        best_step = 0
        stale = 0
        trace: List[dict] = []
        t0 = time.time()
        for step in range(self.cfg.opt_steps):
            opt.zero_grad(set_to_none=True)
            energies, states = energy_fn(g, b)
            loss = energies.sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur = float(torch.min(energies).detach().cpu())
                if cur < best_energy - self.cfg.early_stop_tol:
                    best_energy = cur
                    best_step = step + 1
                    stale = 0
                else:
                    stale += 1
                if self.cfg.log_every > 0 and (step == 0 or (step + 1) % self.cfg.log_every == 0):
                    idx = int(torch.argmin(energies).detach().cpu())
                    state = states[idx]
                    opt_mass = float(torch.sum(torch.abs(state[self.inst.opt_mask]) ** 2).detach().cpu())
                    feas_mass = float(torch.sum(torch.abs(state[self.inst.feasible_mask]) ** 2).detach().cpu())
                    print(
                        f"    {label} trial={trial_idx} step={step + 1:>3}/{self.cfg.opt_steps} "
                        f"E={cur: .6g} feas={100 * feas_mass:5.1f}% opt={100 * opt_mass:7.4f}%"
                    )
                    trace.append({"step": step + 1, "energy": cur, "feasible_mass": feas_mass, "opt_mass": opt_mass})
                if stale >= self.cfg.early_stop_patience:
                    break
        with torch.no_grad():
            energies, states = energy_fn(g, b)
            idx = int(torch.argmin(energies).detach().cpu())
            best = states[idx].detach().clone()
            final_energy = float(energies[idx].detach().cpu())
        info = {
            "best_energy": final_energy,
            "best_step": best_step,
            "actual_steps": step + 1,
            "elapsed_sec": time.time() - t0,
            "trace": trace,
        }
        return best, step + 1, info

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
            feasible_count = float(self.inst.feasible_count)
            opt_cond = success / feasible
            out.update({
                "feasible_cond_entropy": cond_entropy,
                "feasible_cond_ipr": cond_ipr,
                "feasible_eff_support_entropy": eff_entropy,
                "feasible_eff_support_ipr": eff_ipr,
                "feasible_uniformity_entropy": eff_entropy / feasible_count,
                "feasible_uniformity_ipr": eff_ipr / feasible_count,
                "feasible_max_cond_prob": float(torch.max(qf).detach().cpu()),
                "opt_cond_prob": opt_cond,
                "opt_over_uniform_cond": opt_cond * feasible_count / max(float(self.inst.optimum_count), 1.0),
            })
        return out

    def mask_mass(self, state: torch.Tensor, mask: torch.Tensor) -> float:
        return float(torch.sum(torch.abs(state[mask]) ** 2).detach().cpu())

    def run_xy_nested(self, p1: int, p2: int, trial_idx: int, guard: bool = False) -> Tuple[float, int, torch.Tensor, dict]:
        key = ("full", trial_idx, p1)
        if key not in self._xy_stage1_cache:
            phi1, it1, info1 = self.optimize(
                lambda g, b: self.energy_xy(self.inst.Cconf, g, b, p1),
                p_gamma=p1,
                p_beta=p1,
                label=f"XY-Nested S1 p1={p1}",
                trial_idx=trial_idx,
            )
            self._xy_stage1_cache[key] = (phi1, it1, info1)
        phi1, it1, info1 = self._xy_stage1_cache[key]
        stage2_cost = self.inst.Cobj
        if guard:
            stage2_cost = stage2_cost + self.cfg.nested_stage2_guard_scale * self.inst.Cconf
        phi2, it2, info2 = self.optimize(
            lambda g, b: self.energy_history(phi1, stage2_cost, g, b, p2),
            p_gamma=p2,
            p_beta=p2,
            label=f"{'Guarded ' if guard else ''}XY-Nested S2 p2={p2}",
            trial_idx=trial_idx,
        )
        stats = self.state_stats(phi2)
        extra = {
            "stage1_feasible_mass": self.state_stats(phi1)["feasible_mass"],
            "stage1_success_prob": self.state_stats(phi1)["success_prob"],
            "stage1_energy": info1["best_energy"],
            "stage2_energy": info2["best_energy"],
            "stage2_guard_scale": self.cfg.nested_stage2_guard_scale if guard else 0.0,
        }
        return stats["success_prob"], it1 + it2, phi2, extra

    def run_stage1_only(self, p1: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        key = ("full", trial_idx, p1)
        if key not in self._xy_stage1_cache:
            phi1, it1, info1 = self.optimize(
                lambda g, b: self.energy_xy(self.inst.Cconf, g, b, p1),
                p_gamma=p1,
                p_beta=p1,
                label=f"XY-Stage1-Only p1={p1}",
                trial_idx=trial_idx,
            )
            self._xy_stage1_cache[key] = (phi1, it1, info1)
        phi1, it1, info1 = self._xy_stage1_cache[key]
        stats = self.state_stats(phi1)
        extra = {
            "stage1_feasible_mass": stats["feasible_mass"],
            "stage1_success_prob": stats["success_prob"],
            "stage1_energy": info1["best_energy"],
            "stage2_energy": float("nan"),
            "stage2_guard_scale": 0.0,
        }
        return stats["success_prob"], it1, phi1, extra

    def run_xy_3stage(self, p_path: int, p_cycle: int, p_obj: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        key1 = ("path", trial_idx, p_path)
        if key1 not in self._xy_stage1_cache:
            phi1, it1, info1 = self.optimize(
                lambda g, b: self.energy_xy(self.inst.Cpath, g, b, p_path),
                p_gamma=p_path,
                p_beta=p_path,
                label=f"XY-3Stage S1 path p={p_path}",
                trial_idx=trial_idx,
            )
            self._xy_stage1_cache[key1] = (phi1, it1, info1)
        phi1, it1, info1 = self._xy_stage1_cache[key1]

        key2 = ("path-cycle", trial_idx, p_path, p_cycle)
        if key2 not in self._xy_stage1_cache:
            phi2, it2, info2 = self.optimize(
                lambda g, b: self.energy_history(phi1, self.inst.Ccycle, g, b, p_cycle),
                p_gamma=p_cycle,
                p_beta=p_cycle,
                label=f"XY-3Stage S2 cycle p={p_cycle}",
                trial_idx=trial_idx,
            )
            self._xy_stage1_cache[key2] = (phi2, it2, info2)
        phi2, it2, info2 = self._xy_stage1_cache[key2]

        phi3, it3, info3 = self.optimize(
            lambda g, b: self.energy_history(phi2, self.inst.Cobj, g, b, p_obj),
            p_gamma=p_obj,
            p_beta=p_obj,
            label=f"XY-3Stage S3 edge p={p_obj}",
            trial_idx=trial_idx,
        )
        stats = self.state_stats(phi3)
        extra = {
            "stage1_feasible_mass": self.mask_mass(phi1, self.inst.cycle_feasible_mask),
            "stage1_success_prob": self.state_stats(phi1)["success_prob"],
            "stage1_path_feasible_mass": self.mask_mass(phi1, self.inst.path_feasible_mask),
            "stage1_cycle_feasible_mass": self.mask_mass(phi1, self.inst.cycle_feasible_mask),
            "stage2_path_feasible_mass": self.mask_mass(phi2, self.inst.path_feasible_mask),
            "stage2_cycle_feasible_mass": self.mask_mass(phi2, self.inst.cycle_feasible_mask),
            "stage2_success_prob": self.state_stats(phi2)["success_prob"],
            "stage1_energy": info1["best_energy"],
            "stage2_energy": info2["best_energy"],
            "stage3_energy": info3["best_energy"],
            "stage2_guard_scale": 0.0,
        }
        return stats["success_prob"], it1 + it2 + it3, phi3, extra

    def run_xy_3stage_warm(self, p_path: int, p_cycle: int, p_obj: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        key1 = ("path", trial_idx, p_path)
        if key1 not in self._xy_stage1_cache:
            phi1, it1, info1 = self.optimize(
                lambda g, b: self.energy_xy(self.inst.Cpath, g, b, p_path),
                p_gamma=p_path,
                p_beta=p_path,
                label=f"XY-3Stage S1 path p={p_path}",
                trial_idx=trial_idx,
            )
            self._xy_stage1_cache[key1] = (phi1, it1, info1)
        phi1, it1, info1 = self._xy_stage1_cache[key1]

        key2 = ("path-cycle", trial_idx, p_path, p_cycle)
        if key2 not in self._xy_stage1_cache:
            phi2, it2, info2 = self.optimize(
                lambda g, b: self.energy_history(phi1, self.inst.Ccycle, g, b, p_cycle),
                p_gamma=p_cycle,
                p_beta=p_cycle,
                label=f"XY-3Stage S2 cycle p={p_cycle}",
                trial_idx=trial_idx,
            )
            self._xy_stage1_cache[key2] = (phi2, it2, info2)
        phi2, it2, info2 = self._xy_stage1_cache[key2]

        guard = float(self.cfg.warm_final_guard_scale)
        h_phase = self.inst.Cobj + guard * self.inst.Ccycle
        h_eval = self.inst.Cobj + guard * self.inst.Ccycle
        phi3, it3, info3 = self.optimize(
            lambda g, b: self.energy_xy_from(phi2, h_phase, h_eval, g, b, p_obj),
            p_gamma=p_obj,
            p_beta=p_obj,
            label=f"XY-3Stage WarmFinal p={p_obj}",
            trial_idx=trial_idx,
        )
        stats = self.state_stats(phi3)
        extra = {
            "stage1_feasible_mass": self.mask_mass(phi1, self.inst.cycle_feasible_mask),
            "stage1_success_prob": self.state_stats(phi1)["success_prob"],
            "stage1_path_feasible_mass": self.mask_mass(phi1, self.inst.path_feasible_mask),
            "stage1_cycle_feasible_mass": self.mask_mass(phi1, self.inst.cycle_feasible_mask),
            "stage2_path_feasible_mass": self.mask_mass(phi2, self.inst.path_feasible_mask),
            "stage2_cycle_feasible_mass": self.mask_mass(phi2, self.inst.cycle_feasible_mask),
            "stage2_success_prob": self.state_stats(phi2)["success_prob"],
            "warm_final_guard_scale": guard,
            "stage1_energy": info1["best_energy"],
            "stage2_energy": info2["best_energy"],
            "stage3_energy": info3["best_energy"],
            "stage2_guard_scale": 0.0,
        }
        return stats["success_prob"], it1 + it2 + it3, phi3, extra

    def run_std_xy(self, p: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        h = self.penalty * self.inst.Cconf + self.inst.Cobj
        state, it, info = self.optimize(
            lambda g, b: self.energy_xy(h, g, b, p),
            p_gamma=p,
            p_beta=p,
            label=f"Std-XY p={p}",
            trial_idx=trial_idx,
        )
        return self.state_stats(state)["success_prob"], it, state, {"energy": info["best_energy"]}

    def run_decoupled_xy(self, p: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        h_pen = self.penalty * self.inst.Cconf
        state, it, info = self.optimize(
            lambda g, b: self.energy_xy(h_pen + self.inst.Cobj, g, b, p, split=(h_pen, self.inst.Cobj)),
            p_gamma=2 * p,
            p_beta=p,
            label=f"d-XY p={p}",
            trial_idx=trial_idx,
        )
        return self.state_stats(state)["success_prob"], it, state, {"energy": info["best_energy"]}

    def run_grover_feasible(self, p: int, trial_idx: int) -> Tuple[float, int, torch.Tensor, dict]:
        # Idealized baseline: assumes exact feasible-state preparation is given.
        state, it, info = self.optimize(
            lambda g, b: self.energy_history(self.inst.psi_feasible, self.inst.Cobj, g, b, p),
            p_gamma=p,
            p_beta=p,
            label=f"Ideal-Grover-Feasible p={p}",
            trial_idx=trial_idx,
        )
        return self.state_stats(state)["success_prob"], it, state, {"energy": info["best_energy"]}


def classical_baselines(cfg: Config, inst: BlockWDSHCCInstance) -> List[dict]:
    rows: List[dict] = []
    rng = np.random.default_rng(inst.seed + 9001)
    feasible_idx = np.where(inst.feasible_mask_np)[0]
    opt_idx = set(np.where(inst.opt_mask_np)[0].tolist())
    eval_ru = inst.objective_ru + inst.conflict_ru

    if cfg.random_samples > 0:
        draws = rng.choice(feasible_idx, size=min(cfg.random_samples, len(feasible_idx)), replace=len(feasible_idx) < cfg.random_samples)
        hit = sum(1 for i in draws if int(i) in opt_idx)
        p = hit / len(draws)
        rows.append({
            "algorithm": "Classical-Random-Feasible",
            "params": f"samples={len(draws)}",
            "success_prob": p,
            "RU": len(draws) * eval_ru,
            "RTS": 1.0 if p >= cfg.target_prob else (float("inf") if p <= 0 else math.ceil(math.log(1 - cfg.target_prob) / math.log(1 - p))),
            "note": "Diagnostic Monte Carlo over exact feasible states; RU counts conflict+objective term queries per sample",
        })

    best_hits = 0
    evals = 0
    best_obj = float("inf")
    for _ in range(cfg.local_search_restarts):
        idx = int(rng.choice(feasible_idx))
        choices = inst.choice_table[idx].copy()
        cur_idx = idx
        cur_obj = float(inst.Cobj_np[cur_idx])
        for _step in range(cfg.local_search_steps):
            evals += 1
            b = int(rng.integers(0, inst.M))
            proposal = choices.copy()
            proposal[b] = int(rng.integers(0, inst.L))
            prop_idx = 0
            for c in proposal:
                prop_idx = prop_idx * inst.L + int(c)
            if not inst.feasible_mask_np[prop_idx]:
                continue
            prop_obj = float(inst.Cobj_np[prop_idx])
            if prop_obj <= cur_obj or rng.random() < math.exp(-(prop_obj - cur_obj) / 0.75):
                choices = proposal
                cur_idx = prop_idx
                cur_obj = prop_obj
            best_obj = min(best_obj, cur_obj)
        if cur_idx in opt_idx:
            best_hits += 1
    p_ls = best_hits / max(1, cfg.local_search_restarts)
    rows.append({
        "algorithm": "Classical-Local-Search",
        "params": f"restarts={cfg.local_search_restarts},steps={cfg.local_search_steps}",
        "success_prob": p_ls,
        "RU": evals * eval_ru,
        "RTS": 1.0 if p_ls >= cfg.target_prob else (float("inf") if p_ls <= 0 else math.ceil(math.log(1 - cfg.target_prob) / math.log(1 - p_ls))),
        "best_obj_seen": best_obj,
        "note": "Feasible-neighborhood annealed local search; RU counts conflict+objective term queries per evaluated move",
    })
    return rows


def aggregate(rows: Sequence[dict], cfg: Config) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["params"]), []).append(row)
    out: List[dict] = []
    for (algo, params), vals in grouped.items():
        probs = np.array([float(v["success_prob"]) for v in vals], dtype=float)
        ru = np.array([float(v["RU"]) for v in vals], dtype=float)
        iters = np.array([float(v.get("iters", np.nan)) for v in vals], dtype=float)
        finite_iters = iters[np.isfinite(iters)]
        p = float(np.nanmean(probs))
        if p >= cfg.target_prob:
            rts = 1.0
        elif p <= 1e-14 or not np.isfinite(p):
            rts = float("inf")
        else:
            rts = float(math.ceil(math.log(1 - cfg.target_prob) / math.log(1 - p)))
        ru_mean = float(np.nanmean(ru))
        row = {
            "algorithm": algo,
            "params": params,
            "P_succ_mean": p,
            "P_succ_std": float(np.nanstd(probs)),
            "RU": ru_mean,
            "RTS": rts,
            "RTS_x_RU": rts * ru_mean if np.isfinite(rts) else float("inf"),
            "iters_mean": float(np.mean(finite_iters)) if finite_iters.size else float("nan"),
            "runs": len(vals),
        }
        for metric in (
            "feasible_mass",
            "feasible_uniformity_entropy",
            "feasible_uniformity_ipr",
            "feasible_eff_support_entropy",
            "feasible_eff_support_ipr",
            "feasible_max_cond_prob",
            "opt_cond_prob",
            "opt_over_uniform_cond",
        ):
            arr = np.array([float(v.get(metric, np.nan)) for v in vals], dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")
        out.append(row)
    return sorted(out, key=lambda r: (float("inf") if not np.isfinite(float(r["RTS_x_RU"])) else float(r["RTS_x_RU"]), r["algorithm"]))


def comparable_quantum(rows: Sequence[dict]) -> List[dict]:
    excluded = {"Ideal-Grover-Feasible"}
    return [r for r in rows if r.get("algorithm") not in excluded]


def comparable_all(rows: Sequence[dict]) -> List[dict]:
    excluded = {"Ideal-Grover-Feasible", "Classical-Random-Feasible"}
    return [r for r in rows if r.get("algorithm") not in excluded]


def make_plots(outdir: str, rows: Sequence[dict]) -> None:
    finite = [r for r in rows if np.isfinite(float(r["RTS_x_RU"]))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for algo in sorted({r["algorithm"] for r in rows}):
        sub = [r for r in rows if r["algorithm"] == algo]
        ax.scatter([float(r["RU"]) for r in sub], [float(r["P_succ_mean"]) for r in sub], label=algo, s=42)
    ax.set_xscale("log")
    ax.set_xlabel("single-run resource units (RU)")
    ax.set_ylabel("success probability")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "success_vs_ru.png"))
    plt.close(fig)

    if finite:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
        for algo in sorted({r["algorithm"] for r in finite}):
            sub = [r for r in finite if r["algorithm"] == algo]
            ax.scatter([float(r["RU"]) for r in sub], [float(r["RTS_x_RU"]) for r in sub], label=algo, s=42)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("single-run resource units (RU)")
        ax.set_ylabel("RTS x RU")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "rts_x_ru_vs_ru.png"))
        plt.close(fig)


def run_experiment(cfg: Config) -> str:
    device = choose_device(cfg.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = cfg.outdir or os.path.join(os.getcwd(), f"outputs_{cfg.run_label}_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"[device] {device}")
    print(f"[outdir] {outdir}")

    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=list)

    trial_rows: List[dict] = []
    instance_rows: List[dict] = []
    classical_rows_all: List[dict] = []

    for seed in cfg.seeds:
        inst = BlockWDSHCCInstance(cfg, device=device, seed=int(seed))
        summary = inst.summary()
        instance_rows.append(summary)
        print("\n" + "=" * 88)
        print(json.dumps(summary, indent=2))
        print("=" * 88)

        if cfg.run_classical_baselines:
            for row in classical_baselines(cfg, inst):
                row = {"seed": seed, **row}
                classical_rows_all.append(row)

        runner = QAOARunner(cfg, inst)

        def record(name: str, params: str, ru: int, fn: Callable[[int], Tuple[float, int, Optional[torch.Tensor], dict]]) -> None:
            for trial in range(cfg.trials):
                print(f"\n>>> seed={seed} {name} {params} trial {trial + 1}/{cfg.trials} RU={ru}")
                t0 = time.time()
                p_succ, iters, state, extra = fn(trial)
                elapsed = time.time() - t0
                stats = runner.state_stats(state)
                row = {
                    "seed": seed,
                    "algorithm": name,
                    "params": params,
                    "trial": trial,
                    "RU": ru,
                    "success_prob": float(p_succ),
                    "iters": iters,
                    "elapsed_sec": elapsed,
                    **stats,
                    **extra,
                }
                trial_rows.append(row)
                print(
                    f"<<< {name} {params} P={100 * p_succ:.5f}% "
                    f"feasible={100 * stats['feasible_mass']:.2f}% iters={iters} elapsed={elapsed:.1f}s"
                )

        if cfg.run_xy_nested:
            for p1, p2 in cfg.xy_nested_configs:
                record(
                    "XY-Nested",
                    f"({p1},{p2})",
                    inst.resource_xy_nested(p1, p2),
                    lambda trial, p1=p1, p2=p2: runner.run_xy_nested(p1, p2, trial),
                )
        if cfg.run_xy_3stage:
            for p_path, p_cycle, p_obj in cfg.xy_3stage_configs:
                record(
                    "XY-3Stage-Nested",
                    f"({p_path},{p_cycle},{p_obj})",
                    inst.resource_xy_3stage(p_path, p_cycle, p_obj),
                    lambda trial, p_path=p_path, p_cycle=p_cycle, p_obj=p_obj: runner.run_xy_3stage(p_path, p_cycle, p_obj, trial),
                )
        if cfg.run_xy_3stage_warm:
            for p_path, p_cycle, p_obj in cfg.xy_3stage_warm_configs:
                record(
                    "XY-3Stage-WarmFinal",
                    f"({p_path},{p_cycle},{p_obj})",
                    inst.resource_xy_3stage_warm(p_path, p_cycle, p_obj),
                    lambda trial, p_path=p_path, p_cycle=p_cycle, p_obj=p_obj: runner.run_xy_3stage_warm(p_path, p_cycle, p_obj, trial),
                )
        if cfg.run_stage1_only:
            stage1_depths = cfg.stage1_depths or tuple(sorted({p1 for p1, _p2 in cfg.xy_nested_configs}))
            for p1 in stage1_depths:
                record(
                    "XY-Stage1-Only",
                    f"p1={p1}",
                    inst.resource_stage1_only(p1),
                    lambda trial, p1=p1: runner.run_stage1_only(p1, trial),
                )
        if cfg.run_guarded_xy_nested:
            for p1, p2 in cfg.xy_nested_configs:
                record(
                    "Guarded-XY-Nested",
                    f"({p1},{p2})",
                    inst.resource_guarded_xy_nested(p1, p2),
                    lambda trial, p1=p1, p2=p2: runner.run_xy_nested(p1, p2, trial, guard=True),
                )
        if cfg.run_std_xy:
            for p in cfg.std_xy_depths:
                record(
                    "Std-XY",
                    f"p={p}",
                    inst.resource_xy(p),
                    lambda trial, p=p: runner.run_std_xy(p, trial),
                )
        if cfg.run_decoupled_xy:
            for p in cfg.decoupled_xy_depths:
                record(
                    "d-XY",
                    f"p={p}",
                    inst.resource_xy(p, decoupled=True),
                    lambda trial, p=p: runner.run_decoupled_xy(p, trial),
                )
        if cfg.run_grover_feasible:
            for p in cfg.grover_depths:
                record(
                    "Ideal-Grover-Feasible",
                    f"p={p}",
                    inst.resource_grover_feasible(p),
                    lambda trial, p=p: runner.run_grover_feasible(p, trial),
                )

    quantum_agg = aggregate(trial_rows, cfg)
    comparable_quantum_agg = comparable_quantum(quantum_agg)
    classical_agg = aggregate(classical_rows_all, cfg) if classical_rows_all else []
    combined_agg = aggregate(trial_rows + classical_rows_all, cfg)
    comparable_all_agg = comparable_all(combined_agg)

    write_csv(os.path.join(outdir, "instance_summaries.csv"), instance_rows)
    write_csv(os.path.join(outdir, "trial_results.csv"), trial_rows)
    write_csv(os.path.join(outdir, "classical_baselines.csv"), classical_rows_all)
    write_csv(os.path.join(outdir, "aggregated_quantum_results.csv"), quantum_agg)
    write_csv(os.path.join(outdir, "aggregated_comparable_quantum_results.csv"), comparable_quantum_agg)
    write_csv(os.path.join(outdir, "aggregated_all_results.csv"), combined_agg)
    write_csv(os.path.join(outdir, "aggregated_comparable_all_results.csv"), comparable_all_agg)

    if not cfg.no_plots:
        make_plots(outdir, combined_agg)

    best_quantum = quantum_agg[0] if quantum_agg else {}
    best_comparable_quantum = comparable_quantum_agg[0] if comparable_quantum_agg else {}
    best_all = combined_agg[0] if combined_agg else {}
    best_comparable_all = comparable_all_agg[0] if comparable_all_agg else {}
    summary = {
        "outdir": outdir,
        "best_quantum": best_quantum,
        "best_comparable_quantum": best_comparable_quantum,
        "best_all": best_all,
        "best_comparable_all": best_comparable_all,
        "xy_nested_best": [r for r in comparable_quantum_agg if r["algorithm"] == "XY-Nested"][:3],
        "timestamp": timestamp,
    }
    with open(os.path.join(outdir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print("\n[best quantum]")
    print(json.dumps(best_quantum, indent=2, allow_nan=True))
    print("[best comparable quantum]")
    print(json.dumps(best_comparable_quantum, indent=2, allow_nan=True))
    print("[best all]")
    print(json.dumps(best_all, indent=2, allow_nan=True))
    print("[best comparable all]")
    print(json.dumps(best_comparable_all, indent=2, allow_nan=True))
    print(f"[done] {outdir}")
    return outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "focused", "deep", "ridge", "ridge_hi", "validate", "tri", "tri_edge3", "tri_cheap", "tri_warm", "sweep"), default="pilot")
    parser.add_argument("--n-blocks", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--block-weight", type=int, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--penalty-scale", type=float, default=None)
    parser.add_argument("--nested-stage2-guard-scale", type=float, default=None)
    parser.add_argument("--warm-final-guard-scale", type=float, default=None)
    parser.add_argument("--planted-strength", type=float, default=None)
    parser.add_argument("--decoy-strength", type=float, default=None)
    parser.add_argument("--noise-strength", type=float, default=None)
    parser.add_argument("--hyperedge-sparsity", type=float, default=None)
    parser.add_argument("--edge-trap-multiplier", type=float, default=None)
    parser.add_argument("--edge-density", type=float, default=None)
    parser.add_argument("--edge-planted-boost", type=float, default=None)
    parser.add_argument("--edge-decoy-count", type=int, default=None)
    parser.add_argument("--edge-decoy-boost", type=float, default=None)
    parser.add_argument("--edge-decoy-pair-prob", type=float, default=None)
    parser.add_argument("--edge-weight-scale", type=float, default=None)
    parser.add_argument("--conflict-topology", choices=("cycle", "path", "none"), default=None)
    parser.add_argument("--extra-conflict-prob", type=float, default=None)
    parser.add_argument("--structured-extra-offsets", type=str, default=None)
    parser.add_argument("--opt-steps", type=int, default=None)
    parser.add_argument("--opt-restarts", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_mode(Config(), args.mode)
    updates = {}
    for name in (
        "seed",
        "n_blocks",
        "block_size",
        "block_weight",
        "n_qubits",
        "device",
        "outdir",
        "penalty_scale",
        "nested_stage2_guard_scale",
        "warm_final_guard_scale",
        "planted_strength",
        "decoy_strength",
        "noise_strength",
        "hyperedge_sparsity",
        "edge_trap_multiplier",
        "edge_density",
        "edge_planted_boost",
        "edge_decoy_count",
        "edge_decoy_boost",
        "edge_decoy_pair_prob",
        "edge_weight_scale",
        "conflict_topology",
        "extra_conflict_prob",
        "opt_steps",
        "opt_restarts",
        "trials",
    ):
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    if ("n_blocks" in updates or "block_size" in updates) and "n_qubits" not in updates:
        updates["n_qubits"] = int(updates.get("n_blocks", cfg.n_blocks)) * int(updates.get("block_size", cfg.block_size))
    if args.seed is not None:
        updates["seeds"] = (args.seed,)
    if args.structured_extra_offsets is not None:
        text = args.structured_extra_offsets.strip()
        updates["structured_extra_offsets"] = tuple(
            int(part.strip()) for part in text.split(",") if part.strip()
        )
    if args.no_plots:
        updates["no_plots"] = True
    if updates:
        cfg = replace(cfg, **updates)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
