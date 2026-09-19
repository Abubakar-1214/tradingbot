---
kind: external_dependency
name: MetaTrader 5 — Local live trading execution platform
slug: metatrader-5
category: external_dependency
category_hints:
    - vendor_identity
    - client_constraint
scope:
    - '**'
---

Local MetaTrader 5 terminal used for live XAUUSD order execution. The bot connects via the `MetaTrader5` Python package, pulls real-time H1 candles with `copy_rates_from_pos`, and sends buy/close orders through `mt5.order_send` using IOC filling and a magic number (`234000`) to identify its positions. Requires an MT5 terminal logged into a broker account on the same machine; it is the primary live-trading path when a local MT5 instance is available.

Key constraints: runs only on a host with MT5 installed; uses H1 timeframe by default; long-only in this codebase (action 0=Flat, 1=Long); minimum lot size 0.01.