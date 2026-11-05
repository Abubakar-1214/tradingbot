import MetaTrader5 as mt5
from datetime import datetime

if mt5.initialize():
    symbols = ['XAUUSDm', 'DXYm', 'US500m', 'USOILm', 'XAGUSDm', 'EURUSDm']
    for sym in symbols:
        mt5.symbol_select(sym, True)
        rates_d1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 10000)
        rates_h1 = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 100000)
        if rates_d1 is not None and len(rates_d1) > 0:
            d1_start = datetime.fromtimestamp(rates_d1[0]['time']).strftime("%Y-%m-%d")
            d1_end = datetime.fromtimestamp(rates_d1[-1]['time']).strftime("%Y-%m-%d")
            d1_str = f"{len(rates_d1)} D1 bars ({d1_start} to {d1_end})"
        else:
            d1_str = "None"
        if rates_h1 is not None and len(rates_h1) > 0:
            h1_start = datetime.fromtimestamp(rates_h1[0]['time']).strftime("%Y-%m-%d")
            h1_end = datetime.fromtimestamp(rates_h1[-1]['time']).strftime("%Y-%m-%d")
            h1_str = f"{len(rates_h1)} H1 bars ({h1_start} to {h1_end})"
        else:
            h1_str = "None"
        print(f"{sym:<10} | {d1_str} | {h1_str}")
    mt5.shutdown()
