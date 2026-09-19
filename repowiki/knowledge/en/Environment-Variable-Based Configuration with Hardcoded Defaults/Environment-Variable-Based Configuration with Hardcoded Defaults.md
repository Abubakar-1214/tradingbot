---
kind: configuration_system
name: Environment-Variable-Based Configuration with Hardcoded Defaults
category: configuration_system
scope:
    - '**'
source_files:
    - .env.example
    - live/live_trade_metaapi.py
    - train/train_ppo.py
    - train/train_ultimate_150.py
    - env/xauusd_env.py
    - env/realistic_execution.py
    - backtest/backtest_engine.py
---

## What system/approach is used

The repository uses a minimal, ad-hoc configuration approach centered on **`.env` files loaded via `python-dotenv`** and **Python module-level constants**. There is no centralized configuration loader, schema validation, or structured config file format (YAML/JSON/TOML) for the application logic. Configuration is split between two layers:

1. **Secrets and external service credentials** — loaded from `.env` at runtime via `load_dotenv()` in the live trading entry point.
2. **Algorithmic/training parameters** — defined as Python module-level constants directly inside training scripts and environment classes.

## Key files and packages

- `.env.example` — template listing all supported environment variables (`METAAPI_TOKEN`, `METAAPI_ACCOUNT_ID`, `SYMBOL`, `TIMEFRAME`, `VOLUME`, `MODEL_PATH`, `MAX_RISK_PER_TRADE`, `MAX_DAILY_LOSS`, `MAX_POSITIONS`, `NEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY`).
- `live/live_trade_metaapi.py` — the only place that calls `from dotenv import load_dotenv; load_dotenv()` and reads secrets via `os.getenv("METAAPI_TOKEN", ...)`, `os.getenv("METAAPI_ACCOUNT_ID", ...)`.
- `train/train_ppo.py` — hardcodes training hyperparameters as module-level constants: `WINDOW = 64`, `COST = 0.0001`, `N_ENVS = 8`, `TRAIN_END_DATE = "2022-01-01"`, `CHUNK_STEPS = 50_000`, `N_CHUNKS = 10`, `SAVE_DIR = "train"`, `SAVE_PREFIX = "ppo_xauusd"`.
- `train/train_ultimate_150.py` — similarly hardcodes DreamerV3 hyperparameters (`BATCH_SIZE = 16`, `PREFILL_STEPS = 5_000`, `TRAIN_STEPS = 1_000_000`, `SAVE_EVERY = 10_000`) plus one argparse-based override for `--device`, `--steps`, `--batch-size`, `--resume`, `--base-tf`.
- `env/xauusd_env.py` — defines default hyperparameters as constructor defaults (`window=64`, `cost_per_trade=0.0001`, `turnover_coef=0.0002`, `flat_penalty=0.00002`, `hold_bonus=0.00002`, `max_episode_steps=None`).
- `env/realistic_execution.py` — accepts an optional `config` dict and falls back to `get_default_config()` when none is provided, using `.get(key, default)` for spread/slippage/commission/volatility multipliers.
- `backtest/backtest_engine.py` — same pattern: `_default_config()` returns conservative defaults, then `config.get('spread', ...)`, etc., are read per instance.

## Architecture and conventions

- **No shared config module.** Each script owns its own constants. The comment in `train/train_ppo.py` explicitly enforces consistency across modules by stating `# KEEP CONSISTENT everywhere` next to `COST = 0.0001`.
- **Training vs. live divergence:** Training scripts hardcode paths like `MODEL_PATH = "train/ppo_xauusd_latest.zip"` and data paths like `data/xauusd_1h.csv`; live trading reads `MODEL_PATH` from `.env` but still hardcodes `SYMBOL`, `TIMEFRAME`, `VOLUME`, `WINDOW`, and `MAGIC_NUMBER` inline.
- **Argparse is limited to a few knobs.** Only `train_ultimate_150.py` exposes CLI flags (`--steps`, `--batch-size`, `--device`, `--resume`, `--base-tf`); other training scripts have no argument parsing.
- **Default-config pattern in components.** Both `XAUUSDTradingEnv.__init__` and `RealisticExecution.__init__` accept keyword arguments with sensible defaults so callers can override individual parameters without passing a full config object.
- **Environment variable names are documented, not enforced.** `.env.example` lists expected keys, but there is no validation, type coercion, or error if a required key is missing — missing values fall through to hardcoded fallback strings like `"YOUR_METAAPI_TOKEN_HERE"` which cause a runtime check later in `run_step`.

## Conventions and constraints

- Secrets must be placed in a local `.env` file alongside `.env.example`; the live script explicitly instructs users to create it based on the example.
- All training hyperparameters are module-level constants in `train/*.py` and must be manually edited to change behavior — there is no config file or CLI flag for most of them.
- The `COST` value is intentionally duplicated across training and environment code with a cross-file consistency comment, indicating a manual synchronization convention rather than a single source of truth.
- Risk-related variables (`MAX_RISK_PER_TRADE`, `MAX_DAILY_LOSS`, `MAX_POSITIONS`) appear in `.env.example` but are not currently read anywhere in the visible codebase, suggesting they are intended future hooks rather than active configuration.
- Deployment documentation (`DEPLOYMENT_GUIDE.md`) describes running via systemd or `nohup`/`screen` and expects the `.env` file to be present on the server — there is no containerized or cloud-native config injection mechanism shown.