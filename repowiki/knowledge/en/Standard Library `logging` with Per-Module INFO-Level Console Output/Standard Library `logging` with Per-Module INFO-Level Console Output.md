---
kind: logging_system
name: Standard Library `logging` with Per-Module INFO-Level Console Output
category: logging_system
scope:
    - '**'
source_files:
    - backtest/backtest_engine.py
    - data/economic_calendar.py
    - data/sentiment_analysis.py
    - env/realistic_execution.py
    - features/calendar_features.py
    - features/cross_timeframe.py
    - features/god_mode_features.py
    - eval/crisis_validation.py
    - evaluate_model.py
---

## What system/approach is used

The repository uses Python's built-in `logging` module exclusively — no third-party logging framework (e.g. `loguru`, `structlog`, `python-json-logger`) is present in `requirements.txt` or imported anywhere. Each module that needs to emit logs performs its own initialization by calling `logging.basicConfig(level=logging.INFO)` followed by `logger = logging.getLogger(__name__)`. There is no centralized logger configuration, no log rotation, no file sink, and no structured JSON formatting.

## Key files and packages

The pattern is repeated across many modules:
- `backtest/backtest_engine.py` — initializes logger at module top, emits backtest lifecycle and results via `logger.info(...)`
- `data/economic_calendar.py` — same pattern; logs calendar load, missing-file fallbacks, and feature computation
- `data/sentiment_analysis.py` — same pattern
- `env/realistic_execution.py` — same pattern; logs execution model parameters
- `features/calendar_features.py`, `features/cross_timeframe.py`, `features/god_mode_features.py` — same pattern
- `eval/crisis_validation.py`, `evaluate_model.py` — same pattern

Every one of these files follows the identical two-line setup:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Architecture and conventions

- **Per-module logger**: Each module creates its own logger via `getLogger(__name__)`, so log records carry a module-qualified name (e.g. `backtest.backtest_engine`). This is the only structural convention beyond the standard library defaults.
- **Log level**: All modules hard-code `level=logging.INFO`. No module configures DEBUG, WARNING, ERROR, or CRITICAL as a default; those levels are still available through the standard hierarchy but are not set at the root handler.
- **Output sink**: Because `basicConfig` is called without arguments, output goes to `sys.stderr` using the default `%(message)s` format. There is no formatter, no timestamp, no filename/line-number prefix, and no file handler.
- **Message style**: Log messages are human-readable console strings decorated with emoji-style markers (`🧪`, `✅`, `⚠️`, `❌`, `📅`, `💰`, etc.) rather than machine-parseable structured records. They are not JSON and contain no fixed schema of fields.
- **Level usage**: The codebase predominantly uses `logger.info(...)`. `logger.warning(...)` is used for recoverable issues (missing calendar file, missing data path, NaN counts). `logger.error(...)` is reserved for fatal conditions such as "No data loaded - cannot validate" or missing checkpoints. `logger.debug(...)` is not observed in any of the scanned files.
- **No centralization**: There is no `config/logging.py`, no `setup_logging()` function, and no shared base class that configures logging once. Each module independently calls `basicConfig`, which means the first import wins and subsequent calls are effectively no-ops on the root handler.

## Conventions and constraints

Observed conventions (descriptive):
- Every module that produces logs imports `logging` and sets up `basicConfig(level=logging.INFO)` plus a `logger = logging.getLogger(__name__)` pair at the top of the file.
- Informational progress and result messages go through `logger.info`; warnings about missing optional resources go through `logger.warning`; unrecoverable failures go through `logger.error`.
- Messages are formatted as plain text with embedded emoji icons and f-string interpolation — there is no structured field extraction, no correlation IDs, and no log-level filtering at runtime.

Enforced rules (as evidenced by the codebase):
- There is no alternative logging framework; adding one would require changing every module that currently uses `logging` directly.
- Because each module calls `basicConfig`, importing a module solely to configure logging elsewhere will not propagate settings — the first module to import `logging` controls the root handler.
- No module reads log configuration from environment variables, config files, or CLI arguments; the log level is permanently pinned to `INFO`.
- There is no log aggregation, rotation, or persistence to disk — all output is printed to stderr.