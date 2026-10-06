"""Subtask 6 verification: TradeExecutor model-mode SL/TP attachment.

- test_model_mode_entry_attaches_sltp: SLTP_MODE=model + sl_frac/tp_frac ->
  order carries entry*(1-sl_frac) / entry*(1+tp_frac) (buy); short mirrored.
- test_model_mode_clamps_out_of_bounds: fractions outside configured bounds
  are clamped to [sl_tp_min_frac, sl_tp_max_frac] / [tp_min_frac, tp_max_frac].
- test_rules_mode_identical_to_atr: default SLTP_MODE=rules keeps the ATR
  entry_sl_tp path (entry -+ atr*mult) — unchanged behavior.
- test_order_always_carries_sl_tp: both modes attach sl and tp on the order.
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.config import AppConfig, BrokerConfig, TradingBehaviorConfig  # noqa: E402
from tests.helpers import (  # noqa: E402
    EQUITY,
    FakeClock,
    build_executor,
    calm_md,
    make_config,
    sample_df,
)

BROKER_KW = dict(symbol="XAUUSD", allow_short=True, sl_atr_mult=2.0, tp_atr_mult=3.0)


def _model_cfg(bounds=(0.005, 0.05, 0.005, 0.05)):
    base = make_config()
    sl_min, sl_max, tp_min, tp_max = bounds
    return AppConfig(
        risk=base.risk,
        broker=BrokerConfig(**BROKER_KW),
        behavior=TradingBehaviorConfig(
            sl_tp_mode="model",
            sl_tp_min_frac=sl_min,
            sl_tp_max_frac=sl_max,
            tp_min_frac=tp_min,
            tp_max_frac=tp_max,
        ),
    )


def _rules_cfg():
    base = make_config()
    return AppConfig(risk=base.risk, broker=BrokerConfig(**BROKER_KW))


def _build(cfg, tmp_path):
    clock = FakeClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    ex = build_executor(clock, cfg, tmp_path, equity=EQUITY)
    ex.broker.set_mid(2000.0)
    return ex


def _last_submit_sl_tp(ex):
    """MockBroker logs TWO lines per order: 'submit ... sl=.. tp=..' then
    '  -> filled ticket=.. at ..'.  Scan for the submit line (the fill line
    has no sl/tp fields)."""
    for line in reversed(ex.broker.order_log()):
        if line.startswith("submit "):
            m = re.search(r"sl=(\S+) tp=(\S+)$", line)
            assert m, f"submit line missing sl/tp: {line!r}"
            return float(m.group(1)), float(m.group(2))
    raise AssertionError("no submit line found in order log")


def test_model_mode_entry_attaches_sltp(tmp_path):
    ex = _build(_model_cfg(), tmp_path)
    df = sample_df()
    md = calm_md(ex, df)
    ok, reason, result = ex.execute_entry(
        1, EQUITY, df, market_data=md, bar_time="2024-01-01T00:00:00",
        size_multiplier=1.0, sl_frac=0.02, tp_frac=0.03,
    )
    assert ok, reason
    entry = ex.broker.get_tick("XAUUSD")["ask"]  # buy fills at ask
    exp_sl, exp_tp = entry * (1 - 0.02), entry * (1 + 0.03)
    sl, tp = _last_submit_sl_tp(ex)
    assert abs(sl - exp_sl) < 1e-9, f"SL {sl} != expected {exp_sl}"
    assert abs(tp - exp_tp) < 1e-9, f"TP {tp} != expected {exp_tp}"
    # PositionInfo carries the same levels.
    pos = ex.broker.get_positions("XAUUSD")[0]
    assert abs(float(pos.sl) - exp_sl) < 1e-9
    assert abs(float(pos.tp) - exp_tp) < 1e-9


def test_model_mode_short_mirrors(tmp_path):
    ex = _build(_model_cfg(), tmp_path)
    df = sample_df()
    md = calm_md(ex, df)
    ok, reason, result = ex.execute_entry(
        2, EQUITY, df, market_data=md, bar_time="2024-01-01T00:00:00",
        size_multiplier=1.0, sl_frac=0.02, tp_frac=0.03,
    )
    assert ok, reason
    entry = ex.broker.get_tick("XAUUSD")["bid"]  # sell fills at bid
    exp_sl, exp_tp = entry * (1 + 0.02), entry * (1 - 0.03)
    sl, tp = _last_submit_sl_tp(ex)
    assert abs(sl - exp_sl) < 1e-9
    assert abs(tp - exp_tp) < 1e-9


def test_model_mode_clamps_out_of_bounds(tmp_path):
    ex = _build(_model_cfg(), tmp_path)  # bounds 0.005..0.05 both
    df = sample_df()
    md = calm_md(ex, df)
    ok, reason, result = ex.execute_entry(
        1, EQUITY, df, market_data=md, bar_time="2024-01-01T00:00:00",
        size_multiplier=1.0, sl_frac=0.50, tp_frac=0.001,  # both out of bounds
    )
    assert ok, reason
    entry = ex.broker.get_tick("XAUUSD")["ask"]
    exp_sl, exp_tp = entry * (1 - 0.05), entry * (1 + 0.005)  # clamped
    sl, tp = _last_submit_sl_tp(ex)
    assert abs(sl - exp_sl) < 1e-9, f"SL {sl} != clamped {exp_sl}"
    assert abs(tp - exp_tp) < 1e-9, f"TP {tp} != clamped {exp_tp}"


def test_rules_mode_identical_to_atr(tmp_path):
    ex = _build(_rules_cfg(), tmp_path)
    df = sample_df()
    md = calm_md(ex, df)
    ok, reason, result = ex.execute_entry(
        1, EQUITY, df, market_data=md, bar_time="2024-01-01T00:00:00",
        size_multiplier=1.0,
    )
    assert ok, reason
    entry = ex.broker.get_tick("XAUUSD")["ask"]
    atr = ex.compute_atr(df)
    exp_sl, exp_tp = entry - atr * 2.0, entry + atr * 3.0  # ATR rule path
    sl, tp = _last_submit_sl_tp(ex)
    assert abs(sl - exp_sl) < 1e-9
    assert abs(tp - exp_tp) < 1e-9
    # Explicit fractions are IGNORED in rules mode (ATR wins).
    ok2, reason2, _ = ex.execute_entry(
        2, EQUITY, df, market_data=md, bar_time="2024-01-01T01:00:00",
        size_multiplier=1.0, sl_frac=0.02, tp_frac=0.03,
    )
    assert ok2, reason2
    entry2 = ex.broker.get_tick("XAUUSD")["bid"]
    exp_sl2, exp_tp2 = entry2 + atr * 2.0, entry2 - atr * 3.0
    sl2, tp2 = _last_submit_sl_tp(ex)
    assert abs(sl2 - exp_sl2) < 1e-9
    assert abs(tp2 - exp_tp2) < 1e-9


def test_order_always_carries_sl_tp(tmp_path):
    for mode in ("rules", "model"):
        cfg = _model_cfg() if mode == "model" else _rules_cfg()
        ex = _build(cfg, tmp_path)
        df = sample_df()
        md = calm_md(ex, df)
        kw = dict(market_data=md, bar_time="2024-01-01T00:00:00", size_multiplier=1.0)
        if mode == "model":
            kw.update(sl_frac=0.01, tp_frac=0.02)
        ok, reason, _ = ex.execute_entry(1, EQUITY, df, **kw)
        assert ok, reason
        sl, tp = _last_submit_sl_tp(ex)
        assert sl > 0 and tp > 0, f"{mode}: order must carry sl={sl} tp={tp}"


if __name__ == "__main__":
    import pytest as pt

    sys.exit(pt.main([__file__, "-v"]))
