"""Creation-exclusive staging, finalization, and immutable run receipts."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .research_protocol import ProtocolViolation, sha256_file


class ExclusiveFileLock:
    """Fail-fast inter-process lock created with an atomic exclusive file open."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None
        self._token = secrets.token_hex(16)

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ProtocolViolation(f"Research resource is locked: {self.path}") from exc
        self._descriptor = descriptor
        payload = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "token": self._token,
        }
        os.write(descriptor, json.dumps(payload, sort_keys=True).encode("utf-8"))
        os.fsync(descriptor)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("token") != self._token:
                raise ProtocolViolation("Research lock ownership changed unexpectedly")
            self.path.unlink()
        except FileNotFoundError:
            pass


def _file_inventory(root: Path, *, excluded: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def build_run_receipt(
    run_dir: Path,
    *,
    project_root: Path,
    runtime_sources: Sequence[Path],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every staged output and every declared runtime source."""

    if not run_dir.is_dir():
        raise ProtocolViolation(f"Run directory does not exist: {run_dir}")
    root = project_root.resolve()
    code: list[dict[str, Any]] = []
    for source in sorted({path.resolve() for path in runtime_sources}):
        if not source.is_file():
            raise ProtocolViolation(f"Runtime source is missing: {source}")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProtocolViolation("Runtime sources must be inside project_root") from exc
        code.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return {
        "schema_version": "immutable_research_run_receipt_v2",
        "state": "prepared_for_atomic_finalization",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata),
        "outputs": _file_inventory(run_dir, excluded=frozenset({"run_receipt.json"})),
        "runtime_sources": code,
    }


def validate_run_receipt(run_dir: Path, *, project_root: Path) -> dict[str, Any]:
    receipt_path = run_dir / "run_receipt.json"
    if not receipt_path.is_file():
        raise ProtocolViolation("Run receipt is missing")
    parsed = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ProtocolViolation("Run receipt must be a JSON object")
    receipt: dict[str, Any] = dict(parsed)
    expected_outputs = {item["path"]: item for item in receipt.get("outputs", [])}
    actual_outputs = {
        item["path"]: item
        for item in _file_inventory(run_dir, excluded=frozenset({"run_receipt.json"}))
    }
    if set(expected_outputs) != set(actual_outputs):
        raise ProtocolViolation("Run output inventory was mutated")
    for path, expected in expected_outputs.items():
        if expected != actual_outputs[path]:
            raise ProtocolViolation(f"Run output was mutated: {path}")
    root = project_root.resolve()
    for expected in receipt.get("runtime_sources", []):
        source = root / str(expected["path"])
        if not source.is_file() or sha256_file(source) != expected["sha256"]:
            raise ProtocolViolation(f"Runtime source was mutated: {expected['path']}")
    return receipt


def finalize_staged_run(staged_dir: Path, destination: Path) -> Path:
    """Atomically rename one complete staged tree into its canonical run location."""

    staged = staged_dir.resolve()
    final = destination.resolve()
    if staged == final or final.exists():
        raise ProtocolViolation(f"Final run already exists: {final}")
    if not staged.is_dir():
        raise ProtocolViolation(f"Staged run does not exist: {staged}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)
    return final


def write_failed_run_receipt(
    failed_root: Path,
    *,
    batch_id: str,
    error_type: str,
    error_message: str,
    staging_path: Path,
) -> Path:
    """Persist an exclusive failure receipt without deleting partial evidence."""

    path = failed_root / f"{batch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "failed_research_run_receipt_v2",
        "batch_id": batch_id,
        "state": "failed",
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "error_type": error_type,
        "error_message": error_message,
        "staging_path": staging_path.resolve().as_posix(),
    }
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolViolation(f"Refusing to overwrite failed-run receipt: {path}") from exc
    return path
