from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import ThesisConfig
from .data import REQUIRED_COLUMNS, load_mt5_tab_export

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def code_tree_sha256(project_root: Path) -> str:
    """Fingerprint research code/config, independent of filesystem timestamps."""
    paths = [
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        *sorted((project_root / "src" / "hge_gold").glob("*.py")),
        *sorted((project_root / "configs").glob("*.yaml")),
        *sorted((project_root / "scripts").glob("*.py")),
        *sorted((project_root / "mt5").glob("*.mq5")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
                digest.update(block)
    return digest.hexdigest()


def resolved_config_payload(config: ThesisConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["project_root"] = config.project_root.as_posix()
    return payload


def runtime_metadata(project_root: Path) -> dict[str, Any]:
    revision: str | None = None
    dirty: bool | None = None
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        revision = revision_result.stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(dirty_result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_commit": revision,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _optional_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    return (
        (project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    )


def _display_path(project_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_sidecar(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source manifest root must be an object")
    return payload


def _sidecar_file(sidecar: dict[str, Any] | None, source_path: Path) -> dict[str, Any]:
    if sidecar is None:
        return {}
    files = sidecar.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Source manifest 'files' must be a list")
    matches = [
        row for row in files if isinstance(row, dict) and row.get("file") == source_path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"Source manifest must contain exactly one entry for {source_path.name}")
    entry = matches[0]
    declared_hash = entry.get("sha256")
    actual_hash = sha256_file(source_path)
    if declared_hash is not None and declared_hash != actual_hash:
        raise ValueError("Source CSV SHA-256 does not match its sidecar manifest")
    return entry


def source_metadata(
    source: pd.DataFrame,
    source_path: Path | None,
    config: ThesisConfig,
) -> dict[str, Any]:
    source_kind = config.data.source_kind
    is_market = source_kind == "market_evidence"
    sidecar_path = _optional_path(config.project_root, config.data.source_manifest_path)
    raw_path = _optional_path(config.project_root, config.data.raw_source_path)
    sidecar = _load_sidecar(sidecar_path)
    entry = _sidecar_file(sidecar, source_path) if source_path is not None else {}
    account = sidecar.get("account", {}) if sidecar else {}
    if not isinstance(account, dict):
        account = {}

    sidecar_values: dict[str, Any] = {
        "symbol": entry.get("symbol"),
        "timeframe": sidecar.get("timeframe") if sidecar else None,
        "source_type": sidecar.get("source_type") if sidecar else None,
        "broker": account.get("company"),
        "server": account.get("server"),
        "timezone": "UTC" if sidecar else None,
        "export_date": sidecar.get("downloaded_at_utc") if sidecar else None,
        "candle_boundary": sidecar.get("daily_candle_policy") if sidecar else None,
    }

    def choose(field: str) -> Any:
        configured = getattr(config.data, field)
        declared = sidecar_values.get(field)
        if configured is not None and declared is not None and str(configured) != str(declared):
            raise ValueError(f"Configured {field!r} conflicts with source sidecar")
        return configured if configured is not None else declared

    if is_market and source_path is None:
        raise ValueError("market_evidence requires an actual source CSV")
    if source_path is None:
        source_display = "research_sample"
    else:
        try:
            source_display = source_path.relative_to(config.project_root).as_posix()
        except ValueError:
            source_display = source_path.as_posix()

    raw_hash = sha256_file(raw_path) if raw_path is not None and raw_path.is_file() else None
    raw_declaration = sidecar.get("raw_file", {}) if sidecar else {}
    if raw_path is not None and raw_declaration:
        if not isinstance(raw_declaration, dict):
            raise ValueError("Source manifest 'raw_file' must be an object")
        if raw_declaration.get("file") not in {None, raw_path.name}:
            raise ValueError("Configured raw source conflicts with source manifest")
        declared_raw_hash = raw_declaration.get("sha256")
        if declared_raw_hash is not None and declared_raw_hash != raw_hash:
            raise ValueError("Raw source SHA-256 does not match its sidecar manifest")
    transformation_id: str | None = None
    raw_lineage_verified = False
    if raw_path is not None and raw_path.is_file() and is_market:
        reconstructed = load_mt5_tab_export(raw_path, min_rows=config.data.min_rows)
        comparable_source = source.loc[:, ("row_id", *REQUIRED_COLUMNS)].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            reconstructed.loc[:, comparable_source.columns],
            comparable_source,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
        transformation_id = "mt5_tab_export_to_canonical_ohlcv_v1"
        raw_lineage_verified = True
    return {
        "source": source_display,
        "source_kind": source_kind,
        "source_is_market_evidence": is_market,
        "source_sha256": sha256_file(source_path) if source_path is not None else None,
        "source_rows": int(len(source)),
        "source_start": source["date"].iloc[0].isoformat(),
        "source_end": source["date"].iloc[-1].isoformat(),
        "symbol": choose("symbol"),
        "timeframe": choose("timeframe"),
        "source_type": choose("source_type")
        or ("synthetic research sample" if not is_market else None),
        "broker": choose("broker"),
        "server": choose("server"),
        "timezone": choose("timezone"),
        "candle_boundary": choose("candle_boundary"),
        "session_definition": config.data.session_definition,
        "export_date": choose("export_date"),
        "decision_hour_utc": config.data.decision_hour_utc,
        "decision_timestamp_convention": config.data.decision_timestamp_convention,
        "volume_type": config.data.volume_type,
        "source_manifest": _display_path(config.project_root, sidecar_path),
        "source_manifest_sha256": None if sidecar_path is None else sha256_file(sidecar_path),
        "metadata_evidence_status": (sidecar.get("metadata_evidence_status") if sidecar else None),
        "source_manifest_created_at_utc": (
            sidecar.get("manifest_created_at_utc") if sidecar else None
        ),
        "raw_source": _display_path(config.project_root, raw_path),
        "raw_source_sha256": raw_hash,
        "transformation_id": transformation_id,
        "raw_to_canonical_lineage_verified": raw_lineage_verified,
    }
