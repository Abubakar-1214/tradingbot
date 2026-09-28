from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from live.live_trade_mt5 import Mt5BarSource
from scripts import export_mt5_history


class FakeMT5:
    TIMEFRAME_H1 = 1

    @staticmethod
    def initialize(**_kwargs):
        return True

    @staticmethod
    def copy_rates_range(_symbol, _timeframe, _start, _end):
        return [{
            "time": 1704067200,
            "open": 2000.0,
            "high": 2001.0,
            "low": 1999.0,
            "close": 2000.5,
            "tick_volume": 12,
        }]

    @staticmethod
    def copy_rates_from_pos(_symbol, _timeframe, _start_pos, _count):
        return [{
            "time": 1704067200,
            "open": 2000.0,
            "high": 2001.0,
            "low": 1999.0,
            "close": 2000.5,
            "tick_volume": 12,
        }]

    @staticmethod
    def shutdown():
        return None


def test_export_uses_configured_or_cli_server_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        export_mt5_history,
        "load_config",
        lambda: SimpleNamespace(
            broker=SimpleNamespace(
                symbol="XAUUSD",
                utc_offset_hours=2.0,
                mt5_path=None,
                mt5_login=None,
                mt5_password=None,
                mt5_server=None,
            )
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "MetaTrader5", FakeMT5)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    configured_output = tmp_path / "configured.csv"
    cli_output = tmp_path / "cli.csv"

    export_mt5_history.export_history(start, end, configured_output)
    export_mt5_history.main([
        "--from",
        "2024-01-01",
        "--to",
        "2024-01-02",
        "--output",
        str(cli_output),
        "--utc-offset-hours",
        "3",
    ])

    configured_time = pd.read_csv(configured_output, parse_dates=["time"]).iloc[0]["time"]
    cli_time = pd.read_csv(cli_output, parse_dates=["time"]).iloc[0]["time"]
    live_time = Mt5BarSource(
        FakeMT5, "XAUUSD", "H1", utc_offset_hours=2.0
    ).last_closed_bars(1).iloc[0]["time"]
    assert configured_time == live_time.tz_localize(None)
    assert cli_time == pd.Timestamp("2023-12-31 21:00:00")
