---
kind: error_handling
name: Python Standard Exceptions, Logging, and Risk-Supervisor Circuit Breakers
category: error_handling
scope:
    - '**'
source_files:
    - data/load_data.py
    - features/calendar_features.py
    - features/cross_timeframe.py
    - features/god_mode_features.py
    - env/realistic_execution.py
    - backtest/backtest_engine.py
    - live/live_trade_mt5.py
    - models/risk_supervisor.py
---

## What system/approach is used

The repository does not define a custom error hierarchy or centralized exception framework. Instead it follows idiomatic Python error handling:
- **Built-in exceptions** are raised for invalid inputs and missing data (e.g. `FileNotFoundError`, `ValueError`).
- **`try`/`except Exception as e`** blocks are used in data ingestion / feature pipelines to swallow transient failures (network calls, external API fetches) so that one bad event or symbol does not abort the whole batch.
- **Structured logging via `logging.getLogger(__name__)`** is the primary mechanism for reporting warnings, rejections, and operational status; no dedicated error-reporting service is wired up.
- A **deterministic risk-supervisor circuit-breaker** (`RiskSupervisor`) acts as an explicit error-handling layer over trading decisions — instead of raising exceptions on unsafe trades, it returns `(approved: bool, reason: str)` tuples and logs a rejection with a human-readable tag (e.g. `CIRCUIT_BREAKER`, `MAX_DRAWDOWN`, `HIGH_VOLATILITY`, `SPREAD_TOO_WIDE`).

There is no middleware, no global exception handler, no `panic`/`recover` equivalent, and no custom exception classes anywhere in the codebase.

## Key files and packages

| Area | File | Error-handling role |
|---|---|---|
| Data loading | `data/load_data.py` | Raises `FileNotFoundError` when CSV is missing; raises `ValueError` when required columns are absent or OHLC sanity checks fail |
| Feature pipelines | `features/calendar_features.py`, `features/cross_timeframe.py`, `features/god_mode_features.py` | Wrap network / I/O calls in `try`/`except Exception as e` and log the failure, allowing the pipeline to continue |
| Execution model | `env/realistic_execution.py` | Uses `logging` to record initialization and cost breakdowns; no exceptions raised |
| Backtester | `backtest/backtest_engine.py` | Uses `logging` for lifecycle events and result summaries; defensive guards return `0.0` when input arrays are empty |
| Live trading | `live/live_trade_mt5.py` | Checks MT5 connection state (`mt5.initialize()`), retries on failed data fetch, catches `KeyboardInterrupt` to shut down cleanly |
| Risk supervisor | `models/risk_supervisor.py` | Central decision-level error handling: rejects trades deterministically and tracks rejection reasons; provides `emergency_shutdown()` |
| Wrapper | `models/risk_supervisor.py` (`SafeTradingAgent`) | Bridges AI agent output through the risk supervisor, overriding risky actions to flat and returning approval metadata |

## Architecture and conventions

1. **Input validation at boundaries.** The data loader (`load_ohlc_csv`) validates file existence, column presence, numeric coercion, and OHLC consistency, raising `FileNotFoundError` / `ValueError` immediately. This pushes malformed-data errors back to callers rather than propagating NaNs downstream.

2. **Fail-open in feature extraction.** Feature modules wrap expensive or flaky operations (e.g. fetching economic calendar data) in `try`/`except Exception as e` blocks. On failure they log the exception and either skip the feature or return defaults, keeping the rest of the pipeline running. This is a deliberate convention for robustness during training.

3. **Logging-first diagnostics.** Nearly every module sets up `logger = logging.getLogger(__name__)` and emits `info`/`warning`/`critical` messages for lifecycle events, trade approvals/rejections, and backtest results. There is no structured JSON logger configuration beyond `logging.basicConfig(level=logging.INFO)` in a few entry-point modules.

4. **Decision-level error handling via policy override.** Rather than throwing exceptions from the RL agent or environment when a trade would be unsafe, `RiskSupervisor.check_trade` evaluates 12 rules (daily loss limit, max drawdown, position size, consecutive losses, volatility filter, correlation guard, event risk, daily trade cap, cooldown, spread width, market hours) and returns `(True, "APPROVED")` or `(False, <reason>)`. `SafeTradingAgent.act` then overrides the action to flat when rejected. This makes safety violations a *controlled flow* rather than an exception path.

5. **Live loop resilience.** `live_trade_mt5.py` uses a `while True` loop with `time.sleep(10)` between iterations, retrying on failed MT5 data fetches, and catching `KeyboardInterrupt` to call `mt5.shutdown()` before exiting.

6. **No custom exception types.** A grep across all `.py` files finds zero definitions of `class ...Exception` or `class ...Error`. All domain-specific failures are expressed as standard Python exceptions or boolean-returning checks.

## Conventions and constraints

- **Data-layer failures raise**: `FileNotFoundError` for missing CSVs; `ValueError` for schema mismatches and OHLC invariant violations (`high < max(open, close, low)`, `low > min(open, close, high)`). Callers are expected to handle these.
- **Feature-layer failures are swallowed**: Network / I/O calls inside feature modules are wrapped in `try`/`except Exception as e`; failures are logged and the pipeline continues. This is the observed pattern, not a documented rule.
- **Trading decisions never raise**: Unsafe trades are rejected by `RiskSupervisor` with a string reason; the caller receives a boolean + reason tuple. No exceptions are thrown from `check_trade`.
- **Live execution relies on MT5 error codes**: Connection and order-send failures are reported via `mt5.last_error()` and printed; there is no retry/backoff logic beyond the outer loop's sleep-and-retry.
- **Logging level is set per module** via `logging.basicConfig(level=logging.INFO)` in several top-level scripts; there is no central logger configuration, so log formatting is inconsistent across modules.
- **Emergency shutdown is explicit**: `RiskSupervisor.emergency_shutdown()` sets a 365-day halt and logs critical messages; this is the only mechanism to force-stop live trading programmatically.