"""Download daily MT5 market data with provenance retained alongside the CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", default="2011-01-03", help="Inclusive date (YYYY-MM-DD)")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["XAUUSD", "XAGUSD", "US500"],
        help="MT5 symbol names to download",
    )
    parser.add_argument(
        "--volume-type",
        choices=("tick_volume", "real_volume"),
        default="tick_volume",
        help="Explicit MT5 field to expose as canonical volume; no fallback is performed",
    )
    return parser.parse_args()


def _public_fields(value: Any) -> dict[str, Any]:
    return {
        key: item
        for key, item in value._asdict().items()
        if isinstance(item, str | int | float | bool) or item is None
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    terminal = args.terminal.resolve()
    if not terminal.is_file():
        raise FileNotFoundError(f"MT5 terminal not found: {terminal}")

    started_at = datetime.now(UTC)
    if not mt5.initialize(path=str(terminal), timeout=60_000):
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        account = mt5.account_info()
        terminal_info = mt5.terminal_info()
        if account is None or terminal_info is None:
            raise RuntimeError(f"MT5 connection information unavailable: {mt5.last_error()}")

        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
        # UTC bounds constrain the requested history.  Completeness is handled
        # separately by conservatively dropping the newest fetched broker bar.
        end = datetime.combine(started_at.date(), datetime.min.time(), tzinfo=UTC)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []

        for symbol in args.symbols:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Unable to select {symbol}: {mt5.last_error()}")
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"No information for {symbol}: {mt5.last_error()}")
            # Reading the terminal's local history cache is markedly more reliable than
            # requesting a long date range from a demo server in one API call.
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 10_000)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"No daily history for {symbol}: {mt5.last_error()}")

            frame = pd.DataFrame(rates)
            frame["date"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
            frame = frame.sort_values("date").reset_index(drop=True)
            if frame["date"].duplicated().any():
                raise RuntimeError(f"Duplicate MT5 daily bars returned for {symbol}")
            if len(frame) < 2:
                raise RuntimeError(f"Insufficient daily history to verify completion for {symbol}")
            # Position zero may be the currently forming D1 bar.  MT5 does not
            # expose a portable cross-broker session-close timestamp, so the
            # newest fetched bar is always excluded rather than guessed from
            # the workstation's UTC calendar date.
            newest_unverifiable_bar_open = frame["date"].iloc[-1]
            frame = frame.iloc[:-1].copy()
            frame = frame.loc[
                (frame["date"] >= pd.Timestamp(start)) & (frame["date"] < pd.Timestamp(end))
            ].copy()
            if frame.empty:
                raise RuntimeError(f"No completed daily candles returned for {symbol}")
            frame["volume"] = frame[args.volume_type]
            if args.volume_type == "real_volume" and (frame["volume"] <= 0).all():
                raise RuntimeError(
                    f"{symbol} has no positive real_volume; choose tick_volume explicitly instead"
                )
            frame = frame.loc[
                :,
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "tick_volume",
                    "spread",
                    "real_volume",
                ],
            ]
            first_date = frame["date"].iloc[0]
            last_date = frame["date"].iloc[-1]
            file_name = f"{symbol}_D1_{first_date:%Y%m%d}_{last_date:%Y%m%d}.csv"
            destination = args.output_dir / file_name
            frame.to_csv(destination, index=False)
            files.append(
                {
                    "symbol": symbol,
                    "description": info.description,
                    "path": info.path,
                    "digits": info.digits,
                    "currency_base": info.currency_base,
                    "currency_profit": info.currency_profit,
                    "rows": len(frame),
                    "first_timestamp_utc": frame["date"].iloc[0].isoformat(),
                    "last_timestamp_utc": frame["date"].iloc[-1].isoformat(),
                    "newest_unverifiable_bar_open_utc": newest_unverifiable_bar_open.isoformat(),
                    "columns": list(frame.columns),
                    "volume_type": args.volume_type,
                    "file": destination.name,
                    "sha256": _sha256(destination),
                }
            )

        manifest = {
            "source_type": "MetaTrader5 Python API",
            "downloaded_at_utc": started_at.isoformat(),
            "requested_start_utc": start.isoformat(),
            "requested_end_exclusive_utc": end.isoformat(),
            "timeframe": "D1",
            "timezone": "UTC timestamps converted from MT5 epoch seconds",
            "export_date": started_at.date().isoformat(),
            "volume_type": args.volume_type,
            "candle_boundary_definition": (
                "Broker-defined MT5 D1 session; emitted timestamp is the bar-open epoch converted "
                "to UTC. Exact trading-session boundary must be interpreted for the named server."
            ),
            "decision_timestamp_convention": (
                "Features are computed only after a D1 bar is complete; earliest entry is the next "
                "observed D1 bar open."
            ),
            "availability_policy": "drop_newest_fetched_bar_as_unverifiable",
            "daily_candle_policy": (
                "The newest fetched broker D1 bar is always excluded because it may be incomplete."
            ),
            "terminal": _public_fields(terminal_info),
            "account": {
                "server": account.server,
                "company": terminal_info.company,
                "currency": account.currency,
            },
            "broker": terminal_info.company,
            "server": account.server,
            "files": files,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
