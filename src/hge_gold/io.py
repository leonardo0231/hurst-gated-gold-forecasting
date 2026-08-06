from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decision_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "decision_hash"}
    return sha256_bytes(canonical_json(clean))


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dirs(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    ensure_dirs(path.parent)
    frame.to_parquet(path, index=False)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dirs(path.parent)
    frame.to_csv(path, index=False)
