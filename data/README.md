# DreamerV3 & Autonomous Trader - Curated Dataset (data/new_data/)

Generated: 2026-10-06 16:04:27 UTC
Scope: Continuous Gold & Global Macro Dataset (2000–2026)

## 1. Dataset Catalog

| Dataset File | Timeframe | Rows | Coverage Period | Primary Features / Content |
| :--- | :--- | :--- | :--- | :--- |
| `xauusd_h1.csv` | **H1 (1 Hour)** | 112,467 | 2004-06-11 07:00:00 → 2026-10-06 00:00:00 | Primary RL execution candles (`open, high, low, close, volume`) |
| `xauusd_h4.csv` | **H4 (4 Hours)** | 30,526 | 2004-06-11 08:00:00 → 2026-10-06 00:00:00 | Multi-timeframe trend & swing levels |
| `xauusd_d1.csv` | **D1 (Daily)** | 6,402 | 2004-06-12 00:00:00 → 2026-10-06 00:00:00 | Macro trend filter & regime classifier |
| `xauusd_m15.csv` | **M15 (15 Min)** | 141,891 | 2020-01-01 23:00:00 → 2025-12-31 21:45:00 | High-resolution intraday structure |
| `xauusd_m5.csv` | **M5 (5 Min)** | 283,657 | 2022-01-02 23:00:00 → 2025-12-31 21:55:00 | Microstructure & order flow entry precision |
| `macro_daily.csv` | **Daily** | 3,698 | 2015-11-17 → 2025-12-31 | Intermarket drivers (`dxy_close, spx_close, vix_close, oil_close, btc_close`) |
| `economic_calendar.csv` | **Event-based** | 1,188 | 2000-01-07 → 2026-12-31 | High-impact events: NFP, CPI, FOMC rate decisions, GDP |

## 2. Integrity & Quality Standards
- **Zero Lookahead Leakage:** Resampling uses `label='right', closed='right'` so current bars only know past closed bars.
- **Strict Sorting & No Duplicates:** All timestamps are unique, strictly monotonically increasing, and clean.
- **Valid OHLC Geometry:** Guaranteed `high >= max(open, close)` and `low <= min(open, close)`.
