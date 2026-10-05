"""Durable bounded-concurrency controller for immutable appendix cell queues.

The controller schedules cells, never optimizer restarts.  Every scheduled
cell is one child process invoking :mod:`appendix_v3.cell_cli`; the runner's
three independent restarts remain sequential inside that process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import runner
from .cell_cli import (
    CELL_COMPLETION_SCHEMA,
    LoadedJobSpec,
    load_job_spec,
    verify_complete_directory,
)


QUEUE_SCHEMA = "lp-qaoa-appendix-immutable-cell-queue-v1"
PROGRESS_SCHEMA = "lp-qaoa-appendix-queue-progress-v1"
QUEUE_RESULT_SCHEMA = "lp-qaoa-appendix-queue-result-v1"
FAILED_ATTEMPT_SCHEMA = "lp-qaoa-appendix-failed-attempt-v1"

_QUEUE_KEYS = frozenset({"schema", "plan_sha256", "queue_id", "cells"})
_QUEUE_CELL_KEYS = frozenset({"cell_id", "job_spec_path", "job_spec_sha256"})
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_ATTEMPT_NAME = re.compile(r"attempt-(\d{4,})")
_SHA256 = re.compile(r"[0-9A-Fa-f]{64}")
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


class QueueControllerError(ValueError):
    """Raised when queue execution cannot preserve the frozen contract."""


@dataclass(frozen=True)
class QueueCell:
    cell_id: str
    job_spec_path: Path
    job_spec_sha256: str
    loaded_spec: LoadedJobSpec


@dataclass(frozen=True)
class LoadedQueue:
    path: Path
    sha256: str
    queue_id: str
    cells: tuple[QueueCell, ...]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class AttemptAllocation:
    cell: QueueCell
    attempt_number: int
    attempt_dir: Path
    output_dir: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class WorkerOutcome:
    cell_id: str
    status: str
    attempt_number: int
    attempt_dir: Path
    exit_code: int | None
    completion: Mapping[str, Any] | None
    error: str | None


def _compact_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_safe_identifier(value: Any) -> bool:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        return False
    if value in {".", ".."} or value.endswith("."):
        return False
    return value.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED_STEMS


def _assert_plain_worker_count(value: Any) -> int:
    if isinstance(value, bool) or type(value) is not int or not 1 <= value <= 4:
        raise QueueControllerError("max_workers must be a plain integer from 1 through 4")
    return value


def _read_hashed_json(path: Path | str, *, label: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueueControllerError(f"cannot read {label} JSON at {resolved}") from exc
    if not isinstance(payload, dict):
        raise QueueControllerError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest().upper()


def _assert_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    observed = frozenset(payload)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise QueueControllerError(
            f"{label} fields differ from the frozen schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_queue(
    queue_path: Path | str,
    *,
    expected_queue_sha256: str,
) -> LoadedQueue:
    """Load and fully validate a queue before any output directory is created."""

    if not _is_sha256(expected_queue_sha256):
        raise QueueControllerError("expected immutable queue SHA-256 is required")
    path = Path(queue_path).resolve()
    payload, observed_hash = _read_hashed_json(path, label="queue")
    if observed_hash != expected_queue_sha256.upper():
        raise QueueControllerError("immutable queue SHA-256 mismatch")
    _assert_exact_keys(payload, _QUEUE_KEYS, label="queue")
    if payload.get("schema") != QUEUE_SCHEMA:
        raise QueueControllerError("queue has the wrong schema")
    if payload.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256:
        raise QueueControllerError("queue has the wrong plan SHA-256")
    queue_id = payload.get("queue_id")
    if not _is_safe_identifier(queue_id):
        raise QueueControllerError("queue_id is not a safe nonempty identifier")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise QueueControllerError("queue cells must be a nonempty JSON array")

    cells: list[QueueCell] = []
    seen_ids: set[str] = set()
    seen_output_names: set[str] = set()
    seen_paths: set[Path] = set()
    seen_scientific_jobs: set[bytes] = set()
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            raise QueueControllerError(f"queue cell {index} must be a JSON object")
        _assert_exact_keys(raw_cell, _QUEUE_CELL_KEYS, label=f"queue cell {index}")
        cell_id = raw_cell.get("cell_id")
        if not _is_safe_identifier(cell_id):
            raise QueueControllerError(f"queue cell {index} has an unsafe cell_id")
        if cell_id in seen_ids:
            raise QueueControllerError(f"duplicate queue cell_id {cell_id}")
        seen_ids.add(cell_id)
        output_name = cell_id.casefold()
        if output_name in seen_output_names:
            raise QueueControllerError(
                f"queue cell {cell_id} would reuse a case-insensitive output path"
            )
        seen_output_names.add(output_name)
        raw_spec_path = raw_cell.get("job_spec_path")
        if not isinstance(raw_spec_path, str) or not raw_spec_path:
            raise QueueControllerError(f"queue cell {cell_id} lacks a job-spec path")
        spec_path = Path(raw_spec_path)
        if not spec_path.is_absolute():
            spec_path = path.parent / spec_path
        spec_path = spec_path.resolve()
        if spec_path in seen_paths:
            raise QueueControllerError(f"duplicate job-spec path {spec_path}")
        seen_paths.add(spec_path)
        expected_spec_hash = raw_cell.get("job_spec_sha256")
        if not _is_sha256(expected_spec_hash):
            raise QueueControllerError(f"queue cell {cell_id} lacks a job-spec SHA-256")
        try:
            loaded_spec = load_job_spec(
                spec_path,
                expected_sha256=expected_spec_hash,
            )
        except Exception as exc:
            raise QueueControllerError(f"invalid job spec for {cell_id}: {exc}") from exc
        if loaded_spec.manifest.cell_id != cell_id:
            raise QueueControllerError(
                f"queue cell ID {cell_id} differs from its immutable job manifest"
            )
        scientific_record = dict(loaded_spec.manifest.to_record())
        scientific_record.pop("cell_id", None)
        scientific_identity = _compact_json_bytes(scientific_record)
        if scientific_identity in seen_scientific_jobs:
            raise QueueControllerError(
                f"duplicate scientific job detected at queue cell {cell_id}"
            )
        seen_scientific_jobs.add(scientific_identity)
        cells.append(
            QueueCell(
                cell_id=cell_id,
                job_spec_path=spec_path,
                job_spec_sha256=loaded_spec.sha256,
                loaded_spec=loaded_spec,
            )
        )
    return LoadedQueue(path, observed_hash, queue_id, tuple(cells), payload)


def _assert_queue_unchanged(queue: LoadedQueue) -> None:
    if runner.sha256_file(queue.path) != queue.sha256:
        raise QueueControllerError("immutable queue changed during execution")


def _assert_job_specs_unchanged(queue: LoadedQueue) -> None:
    for cell in queue.cells:
        if runner.sha256_file(cell.job_spec_path) != cell.job_spec_sha256:
            raise QueueControllerError(
                f"immutable job spec changed during execution for {cell.cell_id}"
            )


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace mutable progress with one atomic same-filesystem rename."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_compact_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _validate_existing_progress(
    progress_path: Path,
    queue: LoadedQueue,
    run_root: Path,
) -> int:
    if not progress_path.exists():
        return 0
    payload, _ = _read_hashed_json(progress_path, label="progress")
    if (
        payload.get("schema") != PROGRESS_SCHEMA
        or payload.get("plan_sha256") != runner.EXPECTED_PLAN_SHA256
        or payload.get("queue_sha256") != queue.sha256
        or payload.get("queue_id") != queue.queue_id
        or payload.get("queue_path") != str(queue.path)
        or payload.get("run_root") != str(run_root)
        or not isinstance(payload.get("cells"), list)
    ):
        raise QueueControllerError("existing progress is not bound to this queue")
    revision = payload.get("revision")
    if isinstance(revision, bool) or type(revision) is not int or revision < 1:
        raise QueueControllerError("existing progress has an invalid revision")
    return revision


def _attempt_directories(cell_root: Path) -> list[tuple[int, Path]]:
    if not cell_root.exists():
        return []
    attempts: list[tuple[int, Path]] = []
    for child in cell_root.iterdir():
        match = _ATTEMPT_NAME.fullmatch(child.name)
        if match is not None and child.is_dir():
            attempts.append((int(match.group(1)), child.resolve()))
    attempts.sort(key=lambda item: item[0])
    if len({number for number, _ in attempts}) != len(attempts):
        raise QueueControllerError(f"duplicate attempt numbers under {cell_root}")
    return attempts


def _verified_existing_attempt(
    cell: QueueCell,
    cell_root: Path,
) -> tuple[dict[str, Any] | None, int]:
    valid: list[dict[str, Any]] = []
    attempts = _attempt_directories(cell_root)
    for _, attempt_dir in attempts:
        if (attempt_dir / "FAILED.json").exists():
            continue
        output_dir = attempt_dir / "artifacts"
        if not (output_dir / "COMPLETE.json").is_file():
            continue
        try:
            completion = verify_complete_directory(
                output_dir,
                expected_cell_id=cell.cell_id,
                expected_manifest_record=cell.loaded_spec.manifest.to_record(),
                expected_job_spec_sha256=cell.job_spec_sha256,
            )
        except Exception:
            continue
        completion = dict(completion)
        completion["attempt_dir"] = str(attempt_dir)
        valid.append(completion)
    if len(valid) > 1:
        raise QueueControllerError(
            f"cell {cell.cell_id} has multiple verified complete attempts"
        )
    maximum = max((number for number, _ in attempts), default=0)
    return (valid[0] if valid else None), maximum


def _allocate_attempt(
    cell: QueueCell,
    cell_root: Path,
    after_number: int,
) -> AttemptAllocation:
    cell_root.mkdir(parents=True, exist_ok=True)
    number = after_number + 1
    while True:
        if number > 999_999:
            raise QueueControllerError(f"attempt counter exhausted for {cell.cell_id}")
        attempt_dir = (cell_root / f"attempt-{number:04d}").resolve()
        if attempt_dir.parent != cell_root.resolve():
            raise QueueControllerError("attempt path escaped its cell directory")
        try:
            os.mkdir(attempt_dir)
            break
        except FileExistsError:
            number += 1
    return AttemptAllocation(
        cell=cell,
        attempt_number=number,
        attempt_dir=attempt_dir,
        output_dir=attempt_dir / "artifacts",
        stdout_path=attempt_dir / "stdout.log",
        stderr_path=attempt_dir / "stderr.log",
    )


def _worker_environment(extra: Mapping[str, str] | None) -> dict[str, str]:
    environment = dict(os.environ)
    if extra is not None:
        for key, value in extra.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise QueueControllerError("worker environment keys and values must be strings")
            environment[key] = value
    appendix_source = Path(__file__).resolve().parents[1]
    predecessor_source = (
        runner.DEFAULT_REPO_ROOT / "corrected_bcst_campaign_20260730_v2" / "source_v2"
    )
    required_paths = [str(appendix_source)]
    if predecessor_source.is_dir():
        required_paths.append(str(predecessor_source))
    existing = environment.get("PYTHONPATH")
    if existing:
        required_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(required_paths)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _parse_worker_completion(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QueueControllerError("cannot read worker stdout log") from exc
    stripped = text.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise QueueControllerError("worker stdout must contain exactly one compact JSON line")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise QueueControllerError("worker stdout is not valid completion JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CELL_COMPLETION_SCHEMA:
        raise QueueControllerError("worker stdout has the wrong completion schema")
    return payload


def _write_failed_attempt(allocation: AttemptAllocation, outcome: WorkerOutcome) -> None:
    failure_path = allocation.attempt_dir / "FAILED.json"
    if failure_path.exists():
        return
    runner.atomic_write_json(
        failure_path,
        {
            "schema": FAILED_ATTEMPT_SCHEMA,
            "plan_sha256": runner.EXPECTED_PLAN_SHA256,
            "cell_id": allocation.cell.cell_id,
            "attempt_number": allocation.attempt_number,
            "exit_code": outcome.exit_code,
            "error": outcome.error,
            "stdout_path": str(allocation.stdout_path),
            "stderr_path": str(allocation.stderr_path),
            "retry_policy": "allocate_a_new_attempt_directory",
        },
    )


def _run_worker(
    allocation: AttemptAllocation,
    *,
    worker_command: Sequence[str],
    environment: Mapping[str, str],
    cwd: Path,
) -> WorkerOutcome:
    command = [
        *worker_command,
        "--job-spec",
        str(allocation.cell.job_spec_path),
        "--expected-job-spec-sha256",
        allocation.cell.job_spec_sha256,
        "--output-dir",
        str(allocation.output_dir),
    ]
    exit_code: int | None = None
    error: str | None = None
    completion: Mapping[str, Any] | None = None
    try:
        with open(allocation.stdout_path, "xb") as stdout_handle, open(
            allocation.stderr_path, "xb"
        ) as stderr_handle:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                creationflags=creationflags,
            )
            exit_code = process.wait()
        if runner.sha256_file(allocation.cell.job_spec_path) != allocation.cell.job_spec_sha256:
            raise QueueControllerError("immutable job spec changed during execution")
        if exit_code != 0:
            raise QueueControllerError(f"worker exited with status {exit_code}")
        reported = _parse_worker_completion(allocation.stdout_path)
        verified = verify_complete_directory(
            allocation.output_dir,
            expected_cell_id=allocation.cell.cell_id,
            expected_manifest_record=allocation.cell.loaded_spec.manifest.to_record(),
            expected_job_spec_sha256=allocation.cell.job_spec_sha256,
        )
        if reported != verified:
            raise QueueControllerError(
                "worker completion JSON differs from independently verified artifacts"
            )
        completion = verified
    except Exception as exc:  # noqa: BLE001 - child failures become durable outcomes.
        error = f"{type(exc).__name__}: {exc}"

    outcome = WorkerOutcome(
        cell_id=allocation.cell.cell_id,
        status="succeeded" if error is None else "failed",
        attempt_number=allocation.attempt_number,
        attempt_dir=allocation.attempt_dir,
        exit_code=exit_code,
        completion=completion,
        error=error,
    )
    if error is not None:
        try:
            _write_failed_attempt(allocation, outcome)
        except Exception as marker_exc:  # Progress still records an unmarked partial attempt.
            outcome = WorkerOutcome(
                cell_id=outcome.cell_id,
                status=outcome.status,
                attempt_number=outcome.attempt_number,
                attempt_dir=outcome.attempt_dir,
                exit_code=outcome.exit_code,
                completion=outcome.completion,
                error=(
                    f"{outcome.error}; failed to write FAILED marker: "
                    f"{type(marker_exc).__name__}: {marker_exc}"
                ),
            )
    return outcome


def _progress_record(
    queue: LoadedQueue,
    run_root: Path,
    max_workers: int,
    revision: int,
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PROGRESS_SCHEMA,
        "plan_sha256": runner.EXPECTED_PLAN_SHA256,
        "queue_id": queue.queue_id,
        "queue_path": str(queue.path),
        "queue_sha256": queue.sha256,
        "run_root": str(run_root),
        "max_workers": max_workers,
        "restart_scheduling": "three_restarts_sequential_inside_each_cell_process",
        "revision": revision,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells": [dict(cells[cell.cell_id]) for cell in queue.cells],
    }


def _result_record(
    queue: LoadedQueue,
    run_root: Path,
    max_workers: int,
    *,
    dry_run: bool,
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(cells[cell.cell_id]) for cell in queue.cells]
    return {
        "schema": QUEUE_RESULT_SCHEMA,
        "plan_sha256": runner.EXPECTED_PLAN_SHA256,
        "queue_id": queue.queue_id,
        "queue_path": str(queue.path),
        "queue_sha256": queue.sha256,
        "run_root": str(run_root),
        "dry_run": dry_run,
        "max_workers": max_workers,
        "restart_scheduling": "three_restarts_sequential_inside_each_cell_process",
        "cell_count": len(rows),
        "succeeded": sum(row["status"] == "succeeded" for row in rows),
        "skipped": sum(row["status"] == "skipped_verified_complete" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "would_run": sum(row["status"] == "would_run" for row in rows),
        "would_skip": sum(row["status"] == "would_skip_verified_complete" for row in rows),
        "cells": rows,
    }


def run_queue(
    queue_path: Path | str,
    run_root: Path | str,
    *,
    expected_queue_sha256: str,
    max_workers: int,
    dry_run: bool = False,
    worker_command: Sequence[str] | None = None,
    worker_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run or inspect an immutable queue without ever scheduling restarts."""

    worker_count = _assert_plain_worker_count(max_workers)
    if type(dry_run) is not bool:
        raise QueueControllerError("dry_run must be a boolean")
    queue = load_queue(
        queue_path,
        expected_queue_sha256=expected_queue_sha256,
    )
    command = tuple(worker_command) if worker_command is not None else (
        sys.executable,
        "-m",
        "appendix_v3.cell_cli",
    )
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise QueueControllerError("worker_command must contain nonempty argument strings")
    environment = _worker_environment(worker_env)
    root = Path(run_root).resolve()
    cell_records: dict[str, dict[str, Any]] = {}
    pending: list[tuple[QueueCell, int]] = []
    allocated_seen: set[Path] = set()

    for cell in queue.cells:
        cell_root = root / "cells" / cell.cell_id
        completed, maximum_attempt = _verified_existing_attempt(cell, cell_root)
        if completed is not None:
            status = "would_skip_verified_complete" if dry_run else "skipped_verified_complete"
            cell_records[cell.cell_id] = {
                "cell_id": cell.cell_id,
                "status": status,
                "job_spec_path": str(cell.job_spec_path),
                "job_spec_sha256": cell.job_spec_sha256,
                "attempt_dir": completed["attempt_dir"],
                "completion": {key: value for key, value in completed.items() if key != "attempt_dir"},
            }
        else:
            status = "would_run" if dry_run else "pending"
            cell_records[cell.cell_id] = {
                "cell_id": cell.cell_id,
                "status": status,
                "job_spec_path": str(cell.job_spec_path),
                "job_spec_sha256": cell.job_spec_sha256,
                "next_attempt_number": maximum_attempt + 1,
            }
            pending.append((cell, maximum_attempt))

    if dry_run:
        _assert_queue_unchanged(queue)
        _assert_job_specs_unchanged(queue)
        return _result_record(
            queue,
            root,
            worker_count,
            dry_run=True,
            cells=cell_records,
        )

    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    revision = _validate_existing_progress(progress_path, queue, root) + 1
    allocations: list[AttemptAllocation] = []
    for cell, maximum_attempt in pending:
        _assert_queue_unchanged(queue)
        _assert_job_specs_unchanged(queue)
        allocation = _allocate_attempt(
            cell,
            root / "cells" / cell.cell_id,
            maximum_attempt,
        )
        if allocation.attempt_dir in allocated_seen:
            raise QueueControllerError("controller allocated a duplicate attempt path")
        allocated_seen.add(allocation.attempt_dir)
        allocations.append(allocation)
        cell_records[cell.cell_id] = {
            "cell_id": cell.cell_id,
            "status": "queued",
            "job_spec_path": str(cell.job_spec_path),
            "job_spec_sha256": cell.job_spec_sha256,
            "attempt_number": allocation.attempt_number,
            "attempt_dir": str(allocation.attempt_dir),
            "output_dir": str(allocation.output_dir),
            "stdout_path": str(allocation.stdout_path),
            "stderr_path": str(allocation.stderr_path),
        }
    _atomic_replace_json(
        progress_path,
        _progress_record(queue, root, worker_count, revision, cell_records),
    )

    futures: dict[Future[WorkerOutcome], AttemptAllocation] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for allocation in allocations:
            future = executor.submit(
                _run_worker,
                allocation,
                worker_command=command,
                environment=environment,
                cwd=queue.path.parent,
            )
            futures[future] = allocation
        for future in as_completed(futures):
            allocation = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:  # Defensive: _run_worker normally contains failures.
                outcome = WorkerOutcome(
                    cell_id=allocation.cell.cell_id,
                    status="failed",
                    attempt_number=allocation.attempt_number,
                    attempt_dir=allocation.attempt_dir,
                    exit_code=None,
                    completion=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
                try:
                    _write_failed_attempt(allocation, outcome)
                except Exception as marker_exc:
                    outcome = WorkerOutcome(
                        cell_id=outcome.cell_id,
                        status=outcome.status,
                        attempt_number=outcome.attempt_number,
                        attempt_dir=outcome.attempt_dir,
                        exit_code=outcome.exit_code,
                        completion=outcome.completion,
                        error=(
                            f"{outcome.error}; failed to write FAILED marker: "
                            f"{type(marker_exc).__name__}: {marker_exc}"
                        ),
                    )
            row = dict(cell_records[outcome.cell_id])
            row.update(
                {
                    "status": outcome.status,
                    "exit_code": outcome.exit_code,
                    "error": outcome.error,
                    "completion": dict(outcome.completion) if outcome.completion else None,
                }
            )
            cell_records[outcome.cell_id] = row
            revision += 1
            _assert_queue_unchanged(queue)
            _assert_job_specs_unchanged(queue)
            _atomic_replace_json(
                progress_path,
                _progress_record(queue, root, worker_count, revision, cell_records),
            )

    _assert_queue_unchanged(queue)
    _assert_job_specs_unchanged(queue)
    return _result_record(
        queue,
        root,
        worker_count,
        dry_run=False,
        cells=cell_records,
    )


def _compact_json(value: Mapping[str, Any]) -> str:
    return _compact_json_bytes(value).decode("utf-8").rstrip("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--expected-queue-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--max-workers", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_queue(
            args.queue,
            args.run_root,
            expected_queue_sha256=args.expected_queue_sha256,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - command failures are status codes.
        sys.stderr.write(
            _compact_json({"error": type(exc).__name__, "message": str(exc)}) + "\n"
        )
        return 2
    sys.stdout.write(_compact_json(result) + "\n")
    return 1 if result["failed"] else 0


__all__ = [
    "FAILED_ATTEMPT_SCHEMA",
    "LoadedQueue",
    "PROGRESS_SCHEMA",
    "QUEUE_RESULT_SCHEMA",
    "QUEUE_SCHEMA",
    "QueueCell",
    "QueueControllerError",
    "load_queue",
    "main",
    "run_queue",
]


if __name__ == "__main__":
    raise SystemExit(main())
