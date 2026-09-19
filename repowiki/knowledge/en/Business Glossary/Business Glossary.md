---
kind: business_term
name: Business Glossary
category: business_term
scope:
    - '**'
---

### XAUUSD
- Definition：Gold quoted against the US Dollar; the sole tradable symbol in this project. All data files, environments, feature pipelines, and live execution targets this instrument.
- Aliases：Gold、XAU/USD

### God Mode Features
- Definition：The set of 63 technical indicators (trend, momentum, volatility, volume, price action) computed per bar and fed into the model as part of the 140+ feature observation vector.
- Aliases：god_mode、63 indicators

### Ultimate 150 Features
- Definition：The full observation vector combining multi-timeframe signals, macro data, economic calendar events, and microstructure indicators into ~150 features per step, used by the main training script `train_ultimate_150.py`.
- Aliases：150 features、ultimate features

### Macro Data
- Definition：External market drivers appended to the XAUUSD observation: DXY (US Dollar Index), SPX (S&P 500), US10Y (Treasury yields), VIX, Oil (WTI), Bitcoin, EURUSD, Silver (XAGUSD), GLD. Downloaded via Yahoo Finance and merged into the feature pipeline.
- Aliases：macro、macro features

### Economic Calendar
- Definition：A time-indexed list of high-impact macro events (NFP, CPI, FOMC, GDP, etc.) generated for 2015–2025 and embedded as features so the model can anticipate or reduce risk around known news releases.
- Aliases：calendar、economic events

### Paper Trading
- Definition：Demo-account run of the trained model against live MT5 prices before risking real capital; the recommended validation step between backtesting and live deployment.
- Aliases：demo trading、paper trade

### Crisis Validation
- Definition：Evaluation protocol that tests model behavior during known market stress periods (2020 COVID crash, 2022 inflation spike, 2023 banking crisis) to verify drawdown control rather than normal-period performance.
- Aliases：crisis test、stress test

### Dreamer V3
- Definition：World-model-based reinforcement learning algorithm implemented in this repo; builds a latent dynamics model of markets and plans in imagination, trained separately from PPO via `train_dreamer.py`.
- Aliases：dreamerv3、DreamerV3

### PPO
- Definition：Proximal Policy Optimization, the on-policy actor-critic RL algorithm used as the primary strategy for training the XAUUSD trading policy via Stable-Baselines3.
- Aliases：PPO、Proximal Policy Optimization

### Magic Number
- Definition：Arbitrary integer tag (`234000`) attached to every order placed by the bot so it can distinguish its own positions from others when querying MT5 or MetaAPI.
- Aliases：magic
