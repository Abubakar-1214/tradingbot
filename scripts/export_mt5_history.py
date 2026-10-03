from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import REPO_ROOT, load_config


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def export_history(
    start: datetime,
    end: datetime,
    output: Path,
    timeframe: str = "H1",
    utc_offset_hours: float | None = None,
) -> Path:
    cfg = load_config()
    offset_hours = (
        cfg.broker.utc_offset_hours
        if utc_offset_hours is None
        else float(utc_offset_hours)
    )
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 export requires a Windows MT5 installation") from exc

    tf_str = timeframe.upper()
    tf_const = getattr(mt5, f"TIMEFRAME_{tf_str}", None)
    if tf_const is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    initialize_args = {}
    if cfg.broker.mt5_path:
        initialize_args["path"] = str(cfg.broker.mt5_path)
    if cfg.broker.mt5_login is not None:
        initialize_args["login"] = int(cfg.broker.mt5_login)
    if cfg.broker.mt5_password:
        initialize_args["password"] = cfg.broker.mt5_password
    if cfg.broker.mt5_server:
        initialize_args["server"] = cfg.broker.mt5_server
    if not mt5.initialize(**initialize_args):
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")

    is_fine_tf = tf_str in ("M1", "M5", "M15")

    try:
        chunks = []
        cursor = start
        while cursor < end:
            if is_fine_tf:
                if cursor.month == 12:
                    month_end = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
                else:
                    month_end = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
                chunk_end = min(month_end, end)
            else:
                year_end = datetime(
                    cursor.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc
                )
                chunk_end = min(year_end, end)

            rates = mt5.copy_rates_range(
                cfg.broker.symbol,
                tf_const,
                cursor,
                chunk_end,
            )
            if rates is None:
                raise RuntimeError(
                    f"copy_rates_range failed for {cursor}..{chunk_end}: "
                    f"{mt5.last_error()}"
                )
            if len(rates):
                chunk = pd.DataFrame(rates)
                chunks.append(chunk)

            if is_fine_tf:
                if cursor.month == 12:
                    cursor = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
            else:
                cursor = datetime(
                    cursor.year + 1, 1, 1, tzinfo=timezone.utc
                )
    finally:
        mt5.shutdown()

    if chunks:
        bars = pd.concat(chunks, ignore_index=True)
        bars["time"] = (
            pd.to_datetime(bars["time"], unit="s", utc=True)
            - pd.to_timedelta(offset_hours, unit="h")
        ).dt.tz_localize(None)
        bars = bars.rename(columns={"tick_volume": "volume"})
        bars = bars[["time", "open", "high", "low", "close", "volume"]]
        bars = bars.drop_duplicates("time").sort_values("time")
    else:
        bars = pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume"]
        )
    output = output if output.is_absolute() else REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output, index=False)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export MT5 history to CSV")
    parser.add_argument("--from", dest="start", required=True, help="UTC start date/time")
    parser.add_argument("--to", dest="end", required=True, help="UTC end date/time")
    parser.add_argument(
        "--timeframe",
        default="H1",
        help="timeframe to export (e.g. M1, M5, M15, H1, H4, D1; default: H1)",
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=float,
        help="server time offset from UTC (default: BROKER_UTC_OFFSET_HOURS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output CSV path (default: data/xauusd_<timeframe>.csv)",
    )
    args = parser.parse_args(argv)
    start, end = _utc_datetime(args.start), _utc_datetime(args.end)
    if end <= start:
        parser.error("--to must be later than --from")
    
    tf = args.timeframe.upper()
    output_path = args.output
    if output_path is None:
        output_path = Path(f"data/xauusd_{tf.lower()}.csv")

    path = export_history(
        start, end, output_path, timeframe=tf, utc_offset_hours=args.utc_offset_hours
    )
    print(f"Exported MT5 {tf} bars to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
