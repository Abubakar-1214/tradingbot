"""
Python MT5 Historical Backtester
Fetches historical M15, M5, and M1 candles directly from MT5 terminal
and runs the Tri-Timeframe SMA Scalping strategy with complete metrics.
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

def fetch_mt5_bars(symbol: str, timeframe_str: str, count: int = 5000) -> pd.DataFrame:
    if mt5 is None or not mt5.initialize():
        raise RuntimeError("MetaTrader 5 could not be initialized.")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    tf = tf_map.get(timeframe_str.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe_str}")

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch rates for {symbol} {timeframe_str}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    return df

def run_historical_backtest(
    symbol: str = "BTCUSD",
    days: int = 30,
    fast_sma: int = 20,
    slow_sma: int = 50,
    target_profit_usd: float = 0.40,
    reenter_bars_wait: int = 2,
    initial_balance: float = 100.0,
    lot_size: float = 0.01
):
    print(f"\n=======================================================")
    print(f"  MT5 HISTORICAL STRATEGY BACKTEST: {symbol}")
    print(f"=======================================================")
    print(f"Strategy: Tri-Timeframe SMA (M15 + M5 + M1)")
    print(f"Fast SMA: {fast_sma} | Slow SMA: {slow_sma}")
    print(f"Target Profit: +${target_profit_usd:.2f} | Skip Bars: {reenter_bars_wait}")
    print(f"Initial Balance: ${initial_balance:.2f} | Lot Size: {lot_size}")
    print(f"Connecting to MT5 to fetch real historical data...")

    if not mt5.initialize():
        print(f"❌ Failed to connect to MT5: {mt5.last_error()}")
        return

    # 1. Fetch M15, M5, M1 data
    # 30 days of M15 is ~2880 bars
    count_m15 = max(500, days * 96)
    count_m5 = max(1500, days * 288)
    count_m1 = max(5000, days * 1440)

    try:
        df_m15 = fetch_mt5_bars(symbol, "M15", count=count_m15)
        df_m5 = fetch_mt5_bars(symbol, "M5", count=count_m5)
        df_m1 = fetch_mt5_bars(symbol, "M1", count=count_m1)
    finally:
        mt5.shutdown()

    print(f"✅ Historical Data Fetched:")
    print(f"   M15 Bars: {len(df_m15)} ({df_m15['time'].iloc[0]} to {df_m15['time'].iloc[-1]})")
    print(f"   M5  Bars: {len(df_m5)}")
    print(f"   M1  Bars: {len(df_m1)}")

    # Calculate SMAs
    df_m15["fast"] = df_m15["close"].rolling(fast_sma).mean()
    df_m15["slow"] = df_m15["close"].rolling(slow_sma).mean()

    df_m5["fast"] = df_m5["close"].rolling(fast_sma).mean()
    df_m5["slow"] = df_m5["close"].rolling(slow_sma).mean()

    df_m1["fast"] = df_m1["close"].rolling(fast_sma).mean()
    df_m1["slow"] = df_m1["close"].rolling(slow_sma).mean()

    # Pre-index by time for fast lookups
    df_m5_indexed = df_m5.set_index("time")
    df_m1_indexed = df_m1.set_index("time")

    balance = initial_balance
    equity_curve = [balance]
    trades = []
    
    in_position = False
    pos_side = 0 # 1=Buy, 2=Sell
    entry_price = 0.0
    entry_time = None
    bars_since_close = 999
    last_close_bar = None

    contract_size = 1.0 if "BTC" in symbol or "ETH" in symbol else 100.0 if "XAU" in symbol else 100000.0

    print("\nRunning bar-by-bar simulation...")
    for i in range(slow_sma + 2, len(df_m15)):
        current_bar = df_m15.iloc[i]
        bar_time = current_bar["time"]
        bars_since_close += 1

        # Check if in position -> Evaluate profit
        if in_position:
            # Check high and low of the candle to see if profit target was achieved
            if pos_side == 1: # BUY
                max_profit = (current_bar["high"] - entry_price) * lot_size * contract_size
                close_profit = (current_bar["close"] - entry_price) * lot_size * contract_size
            else: # SELL
                max_profit = (entry_price - current_bar["low"]) * lot_size * contract_size
                close_profit = (entry_price - current_bar["close"]) * lot_size * contract_size

            if max_profit >= target_profit_usd:
                # Target profit locked!
                pnl = target_profit_usd
                balance += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": bar_time,
                    "side": "BUY" if pos_side == 1 else "SELL",
                    "entry_price": entry_price,
                    "pnl": pnl,
                    "balance": balance,
                    "status": "WIN"
                })
                in_position = False
                pos_side = 0
                bars_since_close = 0
                last_close_bar = bar_time
                equity_curve.append(balance)
                continue

        # If not in position -> check entry rules
        if not in_position:
            if last_close_bar == bar_time:
                continue
            if bars_since_close < reenter_bars_wait:
                continue

            # M15 Trend
            fast_15 = current_bar["fast"]
            slow_15 = current_bar["slow"]
            if pd.isna(fast_15) or pd.isna(slow_15):
                continue
            trend_m15 = 1 if fast_15 > slow_15 else 2 if fast_15 < slow_15 else 0

            # Find matching M5 and M1 trend at this exact timestamp
            recent_m5 = df_m5[df_m5["time"] <= bar_time]
            recent_m1 = df_m1[df_m1["time"] <= bar_time]
            if recent_m5.empty or recent_m1.empty:
                continue

            m5_row = recent_m5.iloc[-1]
            m1_row = recent_m1.iloc[-1]

            trend_m5 = 1 if m5_row["fast"] > m5_row["slow"] else 2 if m5_row["fast"] < m5_row["slow"] else 0
            trend_m1 = 1 if m1_row["fast"] > m1_row["slow"] else 2 if m1_row["fast"] < m1_row["slow"] else 0

            # Confluence
            if trend_m15 == 1 and trend_m5 == 1 and trend_m1 == 1:
                in_position = True
                pos_side = 1 # BUY
                entry_price = current_bar["open"]
                entry_time = bar_time
            elif trend_m15 == 2 and trend_m5 == 2 and trend_m1 == 2:
                in_position = True
                pos_side = 2 # SELL
                entry_price = current_bar["open"]
                entry_time = bar_time

    # Results calculation
    total_trades = len(trades)
    wins = len([t for t in trades if t["pnl"] > 0])
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    net_profit = balance - initial_balance
    return_pct = (net_profit / initial_balance) * 100.0

    print("\n" + "=" * 55)
    print("           BACKTEST RESULTS SUMMARY")
    print("=" * 55)
    print(f"Total Trades Taken  : {total_trades}")
    print(f"Winning Trades      : {wins}")
    print(f"Win Rate            : {win_rate:.1f}%")
    print(f"Initial Balance     : ${initial_balance:.2f}")
    print(f"Final Balance       : ${balance:.2f}")
    print(f"Net Realized Profit : +${net_profit:.2f} (+{return_pct:.1f}%)")
    print("=" * 55)

    if total_trades > 0:
        print("\nLast 5 Closed Trades:")
        for t in trades[-5:]:
            print(f"  {t['exit_time']} | {t['side']} @ {t['entry_price']:.2f} -> PnL: +${t['pnl']:.2f} | Balance: ${t['balance']:.2f}")

    return {
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "final_balance": balance,
        "trades": trades
    }

if __name__ == "__main__":
    run_historical_backtest()

