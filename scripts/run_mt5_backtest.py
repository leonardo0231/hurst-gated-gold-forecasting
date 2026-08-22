"""Compile the HGE signal-replay EA and run a fixed MT5 Strategy Tester pass."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--signals", required=True, type=Path)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument(
        "--model",
        type=int,
        default=2,
        help="MT5 tester tick model; 2=open prices (EA default), 4=real ticks",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--close-running-terminal",
        action="store_true",
        help="Gracefully close this installation after metadata collection so /config can run",
    )
    return parser.parse_args()


def _public(value: Any) -> dict[str, Any]:
    return {
        key: item
        for key, item in value._asdict().items()
        if isinstance(item, str | int | float | bool) or item is None
    }


def _validate_signals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["entry_time", "exit_time", "direction", "signal_id"]
    if list(frame.columns) != required:
        raise ValueError(f"Signal columns must be exactly {required}")
    entry = pd.to_datetime(frame["entry_time"], errors="raise")
    exit_ = pd.to_datetime(frame["exit_time"], errors="raise")
    if not entry.is_monotonic_increasing or entry.duplicated().any():
        raise ValueError("Signal entry times must be sorted and unique")
    if not (exit_ > entry).all():
        raise ValueError("Every signal exit must be later than entry")
    if not frame["direction"].isin([-1, 1]).all():
        raise ValueError("Directions must be -1 or 1; no-trade rows must be omitted")
    if frame["signal_id"].duplicated().any():
        raise ValueError("Signal IDs must be unique")
    return frame


def _parse_report_summary(path: Path) -> dict[str, str | None]:
    document = path.read_text(encoding="utf-16", errors="replace")
    labels = {
        "history_quality": "History Quality:",
        "bars": "Bars:",
        "ticks": "Ticks:",
        "total_net_profit": "Total Net Profit:",
        "profit_factor": "Profit Factor:",
        "expected_payoff": "Expected Payoff:",
        "sharpe_ratio": "Sharpe Ratio:",
        "balance_drawdown_maximal": "Balance Drawdown Maximal:",
        "equity_drawdown_maximal": "Equity Drawdown Maximal:",
        "total_trades": "Total Trades:",
        "profit_trades": "Profit Trades (% of total):",
    }
    summary: dict[str, str | None] = {}
    for key, label in labels.items():
        match = re.search(
            rf">\s*{re.escape(label)}\s*</td>\s*<td[^>]*>\s*<b>(.*?)</b>",
            document,
            flags=re.DOTALL | re.IGNORECASE,
        )
        value = re.sub(r"<[^>]+>", "", match.group(1)) if match else None
        summary[key] = html.unescape(value).strip() if value is not None else None
    return summary


def _close_terminal_installation(terminal: Path) -> None:
    """Gracefully close only terminal64 processes launched from the requested path."""
    environment = os.environ.copy()
    environment["HGE_MT5_TERMINAL"] = str(terminal)
    command = (
        "$targets = Get-Process terminal64 -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Path -eq $env:HGE_MT5_TERMINAL }; "
        "foreach ($target in $targets) { "
        "$requested = $false; "
        "for ($attempt = 0; $attempt -lt 10 -and -not $requested; $attempt++) { "
        "$requested = $target.CloseMainWindow(); "
        "if (-not $requested) { Start-Sleep -Milliseconds 500 } }; "
        "if ($requested) { $target.WaitForExit(15000) }; "
        "if (-not $target.HasExited) { Stop-Process -Id $target.Id; $target.WaitForExit(5000) }; "
        "if (-not $target.HasExited) { exit 3 } }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        timeout=30,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not gracefully close the requested MT5 terminal installation")


def _write_mt5_index(root: Path, pipeline_manifest: Path) -> None:
    pipeline = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    indexed_files = sorted(
        path
        for path in root.glob("h*/*")
        if path.is_file() and path.name != "mt5_evidence_index.json"
    )
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_run_id": pipeline["run_id"],
        "pipeline_manifest": pipeline_manifest.relative_to(root.parents[1]).as_posix(),
        "pipeline_manifest_sha256": _sha256(pipeline_manifest),
        "files": {
            path.relative_to(root.parents[1]).as_posix(): _sha256(path) for path in indexed_files
        },
    }
    destination = root / "mt5_evidence_index.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    args = _arguments()
    terminal = args.terminal.resolve()
    signals = args.signals.resolve()
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    ea_source = project_root / "mt5" / "HGE_SignalReplay.mq5"
    if not terminal.is_file() or not ea_source.is_file() or not signals.is_file():
        raise FileNotFoundError("Terminal, EA source, and signals must all exist")
    signal_frame = _validate_signals(signals)

    if not mt5.initialize(path=str(terminal), timeout=60_000):
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        terminal_info = mt5.terminal_info()
        account = mt5.account_info()
        symbol = mt5.symbol_info(args.symbol)
        if terminal_info is None or account is None or symbol is None:
            raise RuntimeError(f"MT5 metadata unavailable: {mt5.last_error()}")
        data_path = Path(terminal_info.data_path)
        common_path = Path(terminal_info.commondata_path)
    finally:
        mt5.shutdown()

    if args.close_running_terminal:
        _close_terminal_installation(terminal)

    output_dir.mkdir(parents=True, exist_ok=True)
    expert_dir = data_path / "MQL5" / "Experts" / "HGE"
    expert_dir.mkdir(parents=True, exist_ok=True)
    deployed_source = expert_dir / ea_source.name
    shutil.copy2(ea_source, deployed_source)

    signal_name = f"hge_{_sha256(signals)[:16]}.csv"
    common_files = common_path / "Files"
    common_files.mkdir(parents=True, exist_ok=True)
    deployed_signals = common_files / signal_name
    shutil.copy2(signals, deployed_signals)

    compile_log = output_dir / "compile.log"
    metaeditor = terminal.parent / "MetaEditor64.exe"
    compiled = deployed_source.with_suffix(".ex5")
    prior_compiled_mtime = compiled.stat().st_mtime_ns if compiled.is_file() else None
    compile_result = subprocess.run(
        [str(metaeditor), f"/compile:{deployed_source}", f"/log:{compile_log}"],
        check=False,
        timeout=120,
    )
    compiled_was_refreshed = compiled.is_file() and (
        prior_compiled_mtime is None or compiled.stat().st_mtime_ns > prior_compiled_mtime
    )
    # MetaEditor64 returns 1 on this installation even after a successful compile.
    # A freshly written EX5 plus a clean compiler log (when emitted) is the evidence.
    if not compiled_was_refreshed:
        raise RuntimeError(f"EA compilation failed; inspect {compile_log}")
    if compile_log.is_file() and "0 errors" not in compile_log.read_text(
        encoding="utf-16-le", errors="ignore"
    ):
        raise RuntimeError(f"EA compilation reported errors; inspect {compile_log}")
    metaeditor_log = data_path / "logs" / "metaeditor.log"
    metaeditor_text = metaeditor_log.read_text(encoding="utf-16", errors="ignore")
    compile_lines = [line for line in metaeditor_text.splitlines() if str(deployed_source) in line]
    compile_evidence_line = compile_lines[-1] if compile_lines else ""
    if "0 errors, 0 warnings" not in compile_evidence_line:
        raise RuntimeError("MetaEditor log does not contain a clean compilation receipt")
    compile_evidence = output_dir / "compile_evidence.txt"
    compile_evidence.write_text(compile_evidence_line + "\n", encoding="utf-8")
    compiled_archive = output_dir / compiled.name
    shutil.copy2(compiled, compiled_archive)

    preset_dir = data_path / "MQL5" / "Profiles" / "Tester"
    preset_dir.mkdir(parents=True, exist_ok=True)
    preset_name = f"HGE_{_sha256(signals)[:16]}.set"
    preset = preset_dir / preset_name
    preset.write_text(
        "\n".join(
            [
                f"InpSignalFile={signal_name}",
                "InpLots=0.01",
                "InpMagic=26082026",
                "InpMaxDeviationPoints=20",
            ]
        ),
        encoding="utf-16",
    )
    installation_preset_dir = terminal.parent / "MQL5" / "Profiles" / "Tester"
    installation_preset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preset, installation_preset_dir / preset_name)
    preset_archive = output_dir / preset.name
    shutil.copy2(preset, preset_archive)

    report_name = f"HGE_strategy_tester_{_sha256(signals)[:16]}.htm"
    config_path = output_dir / "tester.ini"
    config_path.write_text(
        "\n".join(
            [
                "[Tester]",
                "Expert=HGE\\HGE_SignalReplay",
                f"ExpertParameters={preset_name}",
                f"Symbol={args.symbol}",
                "Period=D1",
                "Deposit=10000",
                "Currency=USD",
                "Leverage=1:100",
                f"Model={args.model}",
                "ExecutionMode=0",
                "Optimization=0",
                f"FromDate={args.from_date}",
                f"ToDate={args.to_date}",
                "ForwardMode=0",
                "Visual=0",
                "UseLocal=1",
                "UseRemote=0",
                "UseCloud=0",
                f"Report={report_name}",
                "ReplaceReport=1",
                "ShutdownTerminal=1",
            ]
        ),
        # MT5 terminal configuration files are UTF-16 text. UTF-8 is silently
        # ignored by current Windows terminal builds.
        encoding="utf-16",
    )

    started = datetime.now(UTC)
    tester_result = subprocess.run(
        [str(terminal), f"/config:{config_path}"],
        check=False,
        timeout=args.timeout,
    )
    generated_reports = [
        candidate
        for candidate in (terminal.parent / report_name, data_path / report_name)
        if candidate.is_file()
    ]
    reports: list[Path] = []
    for generated_report in generated_reports:
        archived_report = output_dir / generated_report.name
        shutil.copy2(generated_report, archived_report)
        reports.append(archived_report)
    evidence = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started.isoformat(),
        "scientific_status": "historical_holdout_v1_previously_revealed_signal_replay",
        "optimization_on_test": False,
        "signal_rows": len(signal_frame),
        "signal_sha256": _sha256(signals),
        "ea_source_sha256": _sha256(ea_source),
        "ea_compiled_sha256": _sha256(compiled),
        "compile_evidence": {
            "file": compile_evidence.name,
            "sha256": _sha256(compile_evidence),
            "receipt": compile_evidence_line,
        },
        "metaeditor_return_code": compile_result.returncode,
        "preset_sha256": _sha256(preset),
        "tester_config_sha256": _sha256(config_path),
        "terminal_return_code": tester_result.returncode,
        "terminal": _public(terminal_info),
        "account": {"server": account.server, "company": terminal_info.company},
        "symbol": _public(symbol),
        "tester": {
            "symbol": args.symbol,
            "period": "D1",
            "model": args.model,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "lots": 0.01,
            "model_rationale": (
                "Open-prices mode exactly matches this D1 bar-open-only EA."
                if args.model == 2
                else "User-selected tester model; required tick history must be locally available."
            ),
        },
        "reports": {path.name: _sha256(path) for path in reports if path.is_file()},
        "archived_inputs": {
            compiled_archive.name: _sha256(compiled_archive),
            preset_archive.name: _sha256(preset_archive),
            config_path.name: _sha256(config_path),
        },
        "report_summaries": {
            path.name: _parse_report_summary(path) for path in reports if path.is_file()
        },
        "limitations": [
            "This is frozen-signal replay, not native MQL5 model inference.",
            "The historical holdout was previously revealed and is not pristine confirmation.",
            "No optimization is permitted on the tester date range.",
        ],
    }
    evidence_path = output_dir / "mt5_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    if tester_result.returncode != 0 or not reports:
        raise RuntimeError(f"MT5 tester did not produce a report; inspect {output_dir}")
    pipeline_manifest = project_root / "artifacts" / "v3" / "market" / "execution_manifest.json"
    if pipeline_manifest.is_file():
        _write_mt5_index(output_dir.parent, pipeline_manifest)
    print(evidence_path)


if __name__ == "__main__":
    main()
