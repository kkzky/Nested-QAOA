from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ACCOUNTING_TAG = "phi0-full-xy-history-replay-variable-k-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rts_99(probability: float) -> int | None:
    value = float(probability)
    if value >= 0.99:
        return 1
    if value <= 0.0 or not math.isfinite(value):
        return None
    return int(math.ceil(math.log1p(-0.99) / math.log1p(-value)))


def xy_pairs(layout: Sequence[int]) -> int:
    return int(sum(int(k) * (int(k) - 1) // 2 for k in layout))


def reconstruct_resources(row: Mapping[str, Any]) -> dict[str, int]:
    layout = [int(value) for value in row["layout"]]
    conflict_edges = row["conflict_edge_list"]
    canonical_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in conflict_edges
    }
    if len(canonical_edges) != len(conflict_edges):
        raise AssertionError("conflict-edge manifest contains a duplicate")
    c = len(canonical_edges)
    x = xy_pairs(layout)
    m = int(row["term_count"])
    degree = int(row.get("objective_degree", 4))
    if degree < 1:
        raise AssertionError("objective degree must be positive")
    objective_ru_per_term = 4 * degree - 2
    o = objective_ru_per_term * m
    n = int(row["N"])
    s = n * n
    p1 = int(row.get("p1") or 0)
    depth = int(row.get("depth") or 0)
    p1_cost = p1 * (c + x)
    algorithm = row["algorithm"]
    if algorithm == "XY-Stage1-Only":
        circuit = p1_cost
    elif algorithm == "XY-LP-QAOA":
        circuit = p1_cost + depth * (o + 2 * p1_cost + s)
    elif algorithm in {"Std-XY", "d-XY"}:
        circuit = depth * (c + o + x)
    elif algorithm == "Warm-XY":
        circuit = p1_cost + depth * (c + o + x)
    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    return {
        "C": c,
        "X": x,
        "m": m,
        "objective_degree": degree,
        "objective_RU_per_term": objective_ru_per_term,
        "O": o,
        "S": s,
        "P1": p1_cost,
        "circuit_RU": circuit,
        "terminal_RU": circuit + o,
    }


def validate_trial(row: Mapping[str, Any]) -> dict[str, Any]:
    reconstructed = reconstruct_resources(row)
    recorded = row["resources"]
    mismatches = {
        key: {"recorded": int(recorded[key]), "reconstructed": int(value)}
        for key, value in reconstructed.items()
        if int(recorded[key]) != int(value)
    }
    replay = row["replay_probabilities"]
    target_audits: dict[str, Any] = {}
    for label in (
        "target_a_sqrt",
        "target_b_k2",
        "exact_ground",
        "manuscript_fraction",
    ):
        probability = float(replay[label])
        repetitions = rts_99(probability)
        cost = None if repetitions is None else repetitions * reconstructed["terminal_RU"]
        recorded_rts = row["target_costs"][f"{label}_RTS99"]
        recorded_cost = row["target_costs"][f"{label}_cost"]
        if repetitions != recorded_rts or cost != recorded_cost:
            mismatches[label] = {
                "recorded_RTS99": recorded_rts,
                "reconstructed_RTS99": repetitions,
                "recorded_cost": recorded_cost,
                "reconstructed_cost": cost,
            }
        target_audits[label] = {
            "probability": probability,
            "RTS99": repetitions,
            "cost": cost,
        }
    if row.get("accounting_tag") != ACCOUNTING_TAG:
        mismatches["accounting_tag"] = {
            "recorded": row.get("accounting_tag"),
            "expected": ACCOUNTING_TAG,
        }
    return {
        "schema": "bcst-independent-ru-audit-v1",
        "status": "passed" if not mismatches else "failed",
        "trial_id": row.get("trial_id"),
        "reconstructed_resources": reconstructed,
        "target_audits": target_audits,
        "mismatches": mismatches,
    }


def hand_calculated_tests() -> list[dict[str, Any]]:
    cases = [
        {
            "row": {
                "N": 20,
                "layout": [7, 7, 6],
                "conflict_edge_list": [[i, i + 1] for i in range(19)],
                "term_count": 160,
                "algorithm": "XY-LP-QAOA",
                "p1": 3,
                "depth": 8,
            },
            "expected": {
                "C": 19,
                "X": 57,
                "m": 160,
                "objective_degree": 4,
                "objective_RU_per_term": 14,
                "O": 2240,
                "S": 400,
                "P1": 228,
                "circuit_RU": 24_996,
                "terminal_RU": 27_236,
            },
        },
        {
            "row": {
                "N": 26,
                "layout": [7, 7, 6, 6],
                "conflict_edge_list": [[i, i + 1] for i in range(25)],
                "term_count": 312,
                "algorithm": "Std-XY",
                "p1": 0,
                "depth": 12,
            },
            "expected": {
                "C": 25,
                "X": 72,
                "m": 312,
                "objective_degree": 4,
                "objective_RU_per_term": 14,
                "O": 4368,
                "S": 676,
                "P1": 0,
                "circuit_RU": 53_580,
                "terminal_RU": 57_948,
            },
        },
        {
            "row": {
                "N": 30,
                "layout": [6, 6, 6, 6, 6],
                "conflict_edge_list": [[i, i + 1] for i in range(30)],
                "term_count": 450,
                "algorithm": "Warm-XY",
                "p1": 4,
                "depth": 18,
            },
            "expected": {
                "C": 30,
                "X": 75,
                "m": 450,
                "objective_degree": 4,
                "objective_RU_per_term": 14,
                "O": 6300,
                "S": 900,
                "P1": 420,
                "circuit_RU": 115_710,
                "terminal_RU": 122_010,
            },
        },
        {
            "row": {
                "N": 30,
                "layout": [6, 6, 6, 6, 6],
                "conflict_edge_list": [[i, i + 1] for i in range(30)],
                "term_count": 100,
                "objective_degree": 6,
                "algorithm": "Std-XY",
                "p1": 0,
                "depth": 2,
            },
            "expected": {
                "C": 30,
                "X": 75,
                "m": 100,
                "objective_degree": 6,
                "objective_RU_per_term": 22,
                "O": 2200,
                "S": 900,
                "P1": 0,
                "circuit_RU": 4610,
                "terminal_RU": 6810,
            },
        },
    ]
    results = []
    for case in cases:
        observed = reconstruct_resources(case["row"])
        if observed != case["expected"]:
            raise AssertionError({"observed": observed, "expected": case["expected"]})
        results.append({"status": "passed", **case})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    payload: dict[str, Any] = {
        "validator_sha256": sha256_file(Path(__file__)),
    }
    if args.self_test:
        payload["self_tests"] = hand_calculated_tests()
    if args.trial_json:
        with open(args.trial_json, encoding="utf-8") as handle:
            row = json.load(handle)
        payload["trial_audit"] = validate_trial(row)
    if not args.self_test and not args.trial_json:
        parser.error("request --self-test and/or --trial-json")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
