# Training, evaluation, and deployment guide

> **Roman Urdu mein:** Is guide mein environment se deployment tak repeatable workflow hai. Har artifact ko apne data par evaluate karein; koi trained model repo ke saath nahin aata.

This is an engineering workflow, not a performance claim. No trained model is
shipped. The current strategies have not beaten buy-and-hold net of costs.
Passing tests, producing an artifact, or passing the configured promotion gate
does not establish future performance.

## 0. Environment setup

> **Roman Urdu mein:** Python 3.12 virtual environment banayein; CPU ya CUDA ke mutabiq PyTorch install karein. MetaTrader5 Python package Windows-only hai.

Use Python 3.12 and a virtual environment. The training and MockBroker flows
run without a MetaTrader terminal. The `MetaTrader5` Python package is
Windows-only and is imported lazily by the live MT5 adapter/exporter.

Linux/macOS example with `uv`:

```bash
uv venv --python 3.12 .venv
.venv/bin/python -c "from pathlib import Path; src=Path('requirements.txt'); dst=Path('/tmp/tradingbot-requirements.txt'); dst.write_text(''.join(line for line in src.read_text().splitlines(keepends=True) if not line.lower().startswith('metatrader5')))"
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/tradingbot-requirements.txt
uv pip install --python .venv/bin/python ruff
.venv/bin/python -m pytest tests env -q
```

The requirements-list command above omits only the Windows-only
`MetaTrader5` package from the temporary Linux/macOS install list; it does
not edit the repository requirements. On a Windows host, use the complete
`requirements.txt` instead.

On a CPU-only host, select the PyTorch CPU wheel index as above. For CUDA,
install the PyTorch build matching the host's CUDA runtime using the official
PyTorch selector, then install the other pinned project requirements. A
CPU-only PyTorch wheel cannot use a GPU. PPO and Transformer training currently
select CPU in their training code; Dreamer accepts `--device auto|cpu|cuda|mps`
and can use CUDA when the installed PyTorch build and host support it.

On Windows, create/activate `.venv` with `py -3.12 -m venv .venv` and
`.venv\Scripts\activate`, then install dependencies from `requirements.txt`.
Install the MT5 terminal and Windows `MetaTrader5` package only when using
MT5 export or the real MT5 broker. Never put credentials in committed files.

## 1. Export and prepare market data

> **Roman Urdu mein:** MT5 history export mein server offset set karein; exported `time` UTC-naive hota hai aur OHLCV columns training ke liye use hote hain.

Export hourly bars from an initialized MT5 terminal:

```bash
.venv/bin/python scripts/export_mt5_history.py \
  --from 2020-01-01 --to 2024-01-01 \
  --output data/xauusd_h1.csv
```

The exporter accepts exact ISO dates or date-times. `--utc-offset-hours`
overrides `BROKER_UTC_OFFSET_HOURS`; if omitted, the config value is used.
The value means **MT5 server clock minus UTC**: for a UTC+2 server, set `2`;
the conversion subtracts two hours to obtain UTC. A server one hour behind
UTC uses `-1`. Live `Mt5BarSource` applies the same subtraction. Exported CSV
timestamps are written without a timezone suffix but represent UTC.

Expected output columns are `time,open,high,low,close,volume`. Put the base
series at `data/xauusd_h1.csv` or pass another path with `--data`. The loader
also accepts `tick_volume` and renames it to `volume`.

Macro data is optional. If the model is trained with macro inputs, pass the
same daily CSV at training, evaluation, backtest, and live time using
`--macro` or `MACRO_CSV`. It must contain `time` plus one or more
`*_close` columns such as `dxy_close` or `vix_close`; the feature pipeline
shifts daily data to avoid using an unfinished daily candle.

Training fits its feature contract on the training period only, then applies
that contract to train and test rows. Choose `--train-end` as a timestamp
inside the CSV and leave enough bars on both sides for the 200-bar default
feature warm-up, the model window, and evaluation.

## 2. Train models in a controlled order

> **Roman Urdu mein:** Pehle baseline PPO banayein, phir Transformer aur Dreamer; uske baad MCTS, ensemble, aur optional adaptation/fine-tuning karein.

The commands below are smoke-scale examples verified on synthetic data. For
research runs, keep the same data split, feature window, cost assumptions, and
artifact structure while raising the step counts. PPO runs are CPU-based;
Transformer is currently CPU-based; Dreamer may use CUDA with a compatible
PyTorch install. Runtime depends on bar count, network size, rollout length,
and hardware: tiny runs take seconds, typical 100k-step training may take
minutes to hours on CPU, and default-scale Dreamer can take hours or longer.
These are planning ranges, not measured benchmarks.

Set paths and a split time that falls within your CSV:

```bash
DATA=data/xauusd_h1.csv
TRAIN_END=2022-01-01
ARTIFACTS=artifacts/models
```

### 2.1 PPO baseline

> **Roman Urdu mein:** PPO baseline ko pehle train karein; chhota smoke run pipeline check karta hai, jabke research run mein zyada timesteps chahiye.

PPO uses `--timesteps`, `--n-steps`, and `--batch-size`. A research run often
starts in the hundreds of thousands of timesteps and should be increased only
after inspecting runtime and evaluation stability. This runnable small-scale
command exercises artifact creation:

```bash
.venv/bin/python train/train_ppo.py \
  --data "$DATA" --train-end "$TRAIN_END" --window 8 \
  --timesteps 16 --n-steps 8 --batch-size 4 --chunk-steps 8 \
  --max-episode-steps 16 --hidden-sizes 8 --long-only \
  --artifact-root "$ARTIFACTS" --run-name ppo_baseline
```

PPO checkpoints are saved at chunk boundaries. To continue from an SB3
checkpoint, pass `--resume "$ARTIFACTS/ppo_baseline/model.zip"` with compatible
data dimensions and a new `--run-name`. The trainer writes a fresh artifact
directory rather than modifying the source artifact.

### 2.2 Transformer-PPO

> **Roman Urdu mein:** Transformer-PPO ko PPO ke baad sequence encoder ke saath train karein; is implementation mein training CPU par hoti hai.

Use a window and action configuration compatible with the other members
before assembling an ensemble. Research runs may start around 100k steps;
increase `--steps` after measuring your machine. The following small run is
reproducible on the sample CSV:

```bash
.venv/bin/python train/train_transformer.py \
  --data "$DATA" --train-end "$TRAIN_END" --window 8 \
  --steps 4 --rollout-steps 4 --batch-size 4 --update-epochs 1 \
  --num-minibatches 1 --hidden-dim 8 --num-heads 2 --num-layers 1 \
  --long-only --artifact-root "$ARTIFACTS" --run-name transformer_baseline
```

The CLI does not currently expose checkpoint or optimizer-state resume
options; start a new `--run-name` for another training run and keep the prior
artifact.

### 2.3 DreamerV3

> **Roman Urdu mein:** Dreamer ke liye pehle replay prefill hota hai, phir world model aur actor-critic train hote hain; checkpoints se resume mumkin hai.

For a research run, a starting budget around 100k steps with a replay prefill
of several thousand transitions is more informative than a tiny smoke run.
Choose `--seq-len`, `--batch-size`, and network widths to fit available memory.
This tiny command exercises save, manifest, and evaluation:

```bash
.venv/bin/python train/train_dreamer.py \
  --data "$DATA" --train-end "$TRAIN_END" --window 8 \
  --steps 2 --prefill 8 --train-every 1 --batch-size 1 --seq-len 4 \
  --embed-dim 8 --hidden-dim 16 --stoch-dim 2 --num-categories 3 \
  --horizon 1 --device cpu --long-only \
  --artifact-root "$ARTIFACTS" --run-name dreamer_base
```

Resume from a saved checkpoint with `--resume
"$ARTIFACTS/dreamer_base/model.pt"` and a new `--run-name`; prepared
observation/action dimensions must match. `--device auto` selects CUDA when
available.

### 2.4 Dreamer + MCTS manifest

> **Roman Urdu mein:** MCTS command Dreamer checkpoint ko reuse karke planning manifest banata hai; yeh naya Dreamer train nahin karta.

Create a search wrapper around the Dreamer artifact, then evaluate it so an
`evaluation.json` is present:

```bash
.venv/bin/python train/make_mcts_manifest.py \
  --dreamer "$ARTIFACTS/dreamer_base/manifest.json" \
  --simulations 8 --c-puct 1.0 \
  --artifact-root "$ARTIFACTS" --run-name dreamer_mcts
.venv/bin/python train/evaluate.py \
  --manifest "$ARTIFACTS/dreamer_mcts/manifest.json" \
  --data "$DATA" --start "$TRAIN_END"
```

The MCTS manifest references the Dreamer checkpoint by a relative path; retain
the original artifact directory. It does not expose a training-resume option
because MCTS manifest creation is not a training step.

### 2.5 Ensemble

> **Roman Urdu mein:** Ensemble mein sirf same feature contract aur observation width wale members jorein; weights aur vote mode manifest mein save hote hain.

An ensemble can either train members itself or assemble existing compatible
manifests with `--from`. This example uses the PPO baseline plus a second PPO
artifact trained with the same data split, window, and action count:

```bash
.venv/bin/python train/train_ppo.py \
  --data "$DATA" --train-end "$TRAIN_END" --window 8 \
  --timesteps 16 --n-steps 8 --batch-size 4 --chunk-steps 8 \
  --max-episode-steps 16 --hidden-sizes 8 --long-only \
  --artifact-root "$ARTIFACTS" --run-name ppo_member_2 --seed 43
.venv/bin/python train/train_ensemble.py \
  --from "$ARTIFACTS/ppo_baseline/manifest.json" \
        "$ARTIFACTS/ppo_member_2/manifest.json" \
  --data "$DATA" --train-end "$TRAIN_END" --eval-start "$TRAIN_END" \
  --window 8 --vote soft --min-agreement 0.6 \
  --artifact-root "$ARTIFACTS" --run-name ppo_ensemble
```

An assembly run creates `ensemble.json`, member references, a manifest, and an
evaluation record. There is no in-place ensemble resume; retrain or assemble
into a new `--run-name`.

### 2.6 Optional MAML/meta-training and recent adaptation

> **Roman Urdu mein:** MAML aur recent adaptation dono source Dreamer artifact ko nahin badalte; dono naya artifact aur evaluation banate hain.

Meta-training derives causal regime tasks from the training period. Lower
`--min-len` only when the dataset has no sufficiently long contiguous regimes;
verify the split and avoid adapting on evaluation rows.

```bash
.venv/bin/python train/meta_train_dreamer.py \
  --manifest "$ARTIFACTS/dreamer_base/manifest.json" \
  --data "$DATA" --min-len 32 --epochs 1 --tasks-per-batch 1 \
  --batch-size 1 --adapt-steps 1 \
  --artifact-root "$ARTIFACTS" --run-name dreamer_meta
.venv/bin/python train/adapt_recent.py \
  --manifest "$ARTIFACTS/dreamer_meta/manifest.json" \
  --data "$DATA" --bars 128 --steps 1 --batch-size 1 \
  --artifact-root "$ARTIFACTS" --run-name dreamer_recent
```

These scripts produce new Dreamer artifacts. They do not provide an
optimizer-state resume flag; rerun from a chosen source manifest into a new
directory.

### 2.7 Optional adversarial robustness fine-tune

> **Roman Urdu mein:** Adversarial fine-tune bounded self-play perturbations use karta hai aur output ko clean evaluation environment par check karta hai.

Use the Dreamer base artifact (or another evaluated Dreamer artifact) and
write the fine-tuned model separately:

```bash
.venv/bin/python train/train_adversarial.py \
  --manifest "$ARTIFACTS/dreamer_base/manifest.json" \
  --data "$DATA" --epochs 1 --steps-per-epoch 8 --train-every 1 \
  --batch-size 1 --mm-hidden-dim 8 --max-manipulation-rate 0.2 \
  --artifact-root "$ARTIFACTS" --run-name dreamer_adversarial
```

This command does not overwrite its source. There is no optimizer-state
resume flag; use a new run name and inspect the clean evaluation record.

### Artifact layout

> **Roman Urdu mein:** Har training run apna isolated folder rakhta hai: weights, feature contract, manifest aur evaluation alag files hain.

Typical paths under `artifacts/models/<run-name>/`:

```text
model.zip or model.pt
checkpoint_*.zip or checkpoint_*.pt   # where enabled by the trainer
feature_contract.json
manifest.json
evaluation.json
```

Ensembles also store `ensemble.json` and relative member-manifest paths.
Dreamer MCTS stores a manifest and feature contract that reference the
original Dreamer model file. Keep source artifacts available when moving an
ensemble or MCTS artifact. `artifacts/` is ignored by Git; do not commit model
weights, local market data, or account state.

## 3. Evaluation and promotion

> **Roman Urdu mein:** Evaluation out-of-sample metrics aur thresholds save karti hai; live promotion ke liye passed=true aur contract hash match dono zaroori hain.

Training scripts evaluate on rows at or after `--train-end` and write
`evaluation.json`. Its top-level fields are `metrics`, `thresholds`, `passed`,
`evaluated_at`, `test_start`, `test_end`, and `contract_hash`. The metrics
include `total_return_pct`, `cagr_pct`, `sharpe`, `max_dd_pct`, `trades`,
`win_rate`, `exposure_pct`, `costs`, `buy_hold_return_pct`, `final_equity`,
`bars`, and `bars_per_year`.

Default promotion thresholds in `train/evaluate.py` are:

| Field | Default |
|---|---:|
| `sharpe` | at least `0.5` |
| `max_dd_pct` | at most `20.0` |
| `trades` | at least `20` |
| `total_return_pct` | greater than `0.0` |

The gate is intentionally configurable in code: pass a `thresholds` mapping
to `write_evaluation(...)` (or change `DEFAULT_THRESHOLDS` for a controlled
project-wide policy), then re-evaluate and preserve the resulting record.
Do not edit `passed` manually to bypass a failed evaluation. Record the
data range, contract hash, threshold change, and reviewer approval. A live
model requires a readable manifest and `evaluation.json` with `passed: true`
and the same contract hash. In demo mode, a failed or missing evaluation logs
a warning unless `REQUIRE_PROMOTED_MODEL=true`; live mode always refuses an
unpromoted model.

## 4. Backtest a manifest

> **Roman Urdu mein:** Backtest manifest ke policy signals ko historical OHLC par costs aur baselines ke saath chalata hai; yeh live promotion ka badal nahin.

The backtest runner uses the repository's D1 and H1 data files by default.
Optional input/output overrides are available for isolated or synthetic runs:

```bash
.venv/bin/python scripts/run_backtest_eval.py \
  --manifest "$ARTIFACTS/ppo_baseline/manifest.json" \
  --d1-data "$DATA" --h1-data "$DATA" \
  --output-dir /tmp/tradingbot-backtests \
  --report-path /tmp/tradingbot-backtests/report.md
```

For standard repository data, `--d1-data`, `--h1-data`, `--output-dir`, and
`--report-path` may be omitted. The script compares rule strategies and
baselines on the same inputs, runs the supplied model only for a matching
manifest timeframe, and writes result files plus an honest report. A
backtest is not a substitute for walk-forward evaluation, paper trading, or
the promotion gate.

## 5. Walk-forward and retraining

> **Roman Urdu mein:** Time ke order mein train/test windows banayein, leakage se bachne ke liye purge/embargo rakhein, aur retraining ko fixed review process se jorein.

Keep a final untouched holdout period and use chronological walk-forward
windows with purge/embargo. Do not select settings from the same period used
for final reporting. Re-fit the feature scaler only on each training window,
then apply that window's contract to its test slice. Preserve each manifest,
evaluation record, source-data version, and cost configuration.

Retraining cadence should follow the instrument, timeframe, data-quality
checks, and documented review policy rather than an automatic schedule.
After a data, feature, environment-cost, model, or threshold change, create a
new artifact and repeat evaluation and demo checks. Compare against
buy-and-hold and the rule baselines using the same period and costs.

## 6. Demo, MT5, and live checklist

> **Roman Urdu mein:** Pehle MockBroker demo, phir MT5 demo account; real live mode tabhi jab promotion, risk, credential, reconciliation aur operational checks pass hon.

### Model-backed MockBroker demo

Use a small or fully trained PPO artifact. Set an explicit manifest path and
bounded history; disable the promotion requirement only for a deliberate
unpromoted **demo** smoke run:

```bash
TRADING_MODE=demo \
SIGNAL_SOURCE=model \
MODEL_MANIFEST="$ARTIFACTS/ppo_baseline/manifest.json" \
ALLOW_SHORT=false \
REQUIRE_PROMOTED_MODEL=false \
LIVE_HISTORY_BARS=300 \
RISK_STATE_DB=/tmp/tradingbot-demo-state/risk_state.db \
LOCAL_STATE_FILE=/tmp/tradingbot-demo-state/bot_state.json \
LOG_DIR=/tmp/tradingbot-demo-state/logs \
KILL_SWITCH_PATH=/tmp/tradingbot-demo-state/KILL_SWITCH \
.venv/bin/python live/live_trade_mt5.py --smoke --max-bars 300
```

`--smoke` uses synthetic bars and `MockBroker`; it never connects to a live
account. An evaluation failure is still logged, and this setting must not be
copied to a live deployment. Create the temporary state directory first with
`mkdir -p /tmp/tradingbot-demo-state`, or use dedicated writable state paths
for a local demo.

### MT5 demo account

On Windows with an installed MT5 terminal and `MetaTrader5` package, set
`TRADING_MODE=live` only when the account is a broker demo account and all live
gates pass. Exercise reconciliation, spread/SL/TP behavior, session boundary
handling, reconnects, and kill-switch handling before considering any real
account. Never treat a successful connection as a promotion decision.

### Live checklist and relevant environment variables

> **Roman Urdu mein:** Live se pehle har risk limit, session, state path, MT5 credential aur emergency kill switch ko verify karein; secrets logs ya Git mein na rakhein.

| Area | Variables / files | Review |
|---|---|---|
| Mode and signal | `TRADING_MODE`, `SIGNAL_SOURCE`, `MODEL_MANIFEST`, `REQUIRE_PROMOTED_MODEL` | Keep rule mode default until an evaluated model is explicitly selected. Live model mode requires `evaluation.json` passed and matching `contract_hash`; `ppo` is an alias for `model`. |
| Model input | `ALLOW_SHORT`, `FEATURE_WINDOW`, `FEATURE_DROP_WARMUP`, `FEATURE_NORMALIZE`, `LIVE_HISTORY_BARS`, `MACRO_CSV`, `MACRO_SHIFT` | Action count, observation width, feature contract, history warm-up, and macro columns must match the artifact. Legacy `MODEL_PATH` and `FEATURE_CONTRACT_PATH` are not used by the shared artifact registry. |
| Decision and sessions | `MIN_CONFIDENCE`, `MIN_ENSEMBLE_AGREEMENT`, `SESSION_FILTER`, `NO_TRADE_HOURS_UTC`, `COOLDOWN_BARS_AFTER_LOSS`, `USE_KELLY`, `KELLY_FRACTION`, `KELLY_MIN_TRADES` | Session/cooldown filters block entries, not exits. Keep Kelly disabled or conservatively configured until trade history is meaningful. |
| Position management | `BREAKEVEN_AT_R`, `TRAIL_START_R`, `TRAIL_ATR_MULT`, `PARTIAL_CLOSE_AT_R`, `PARTIAL_CLOSE_FRACTION`, `MAX_BARS_IN_TRADE`, `SL_ATR_MULT`, `TP_ATR_MULT` | Check broker lot minimum/step and confirm SL/TP and time-stop behavior on demo. |
| Risk breakers | `MAX_DAILY_LOSS`, `MAX_DRAWDOWN`, `MAX_CONSECUTIVE_LOSSES`, `MAX_RISK_PER_TRADE`, `MAX_POSITION`, `MAX_TRADES_PER_DAY`, `MIN_TRADE_INTERVAL_SEC`, `VOL_THRESHOLD`, `MAX_SPREAD`, `MARKET_HOURS_ONLY`, `CORRELATION_GUARD_ENABLED`, `CORRELATION_ASSET`, `CORRELATION_BLOCK_LONG_ON_UP` | RiskSupervisor is independent of model confidence. Verify limits in account currency and broker contract units. |
| Costs and broker | `COST_SPREAD`, `COST_COMMISSION`, `COST_SLIPPAGE`, `COST_SWAP_LONG`, `COST_SWAP_SHORT`, `SYMBOL`, `TIMEFRAME`, `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`, `MT5_MAGIC`, `MT5_DEVIATION`, `BROKER_UTC_OFFSET_HOURS`, `MAX_REQUOTE_RETRIES`, `ORDER_RETRY_DELAY_SEC` | Keep credentials in local environment only. `BROKER_UTC_OFFSET_HOURS` is server-time minus UTC; conversion subtracts it. |
| State and operations | `RISK_STATE_DB`, `LOCAL_STATE_FILE`, `LOG_DIR`, `KILL_SWITCH_PATH`, `BACKTEST_RESULTS_DIR`, `LOG_LEVEL` | Check disk permissions, retention, backups, and alerts. Create the configured kill-switch file to stop new bar processing. |

Also inspect `NEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `METAAPI_TOKEN`, and
`METAAPI_ACCOUNT_ID` only if using their optional integrations; they are not
required for the core MT5 model flow.

Before any real-money deployment, require a passed promotion record on the
intended data/contract, independent review of costs and drawdown, a sustained
MT5 demo period, reconciliation and restart tests, a tested kill switch,
monitoring/alert ownership, and a documented rollback plan. The code is
production-structured; no model is validated for real money until it passes
the gate on your data and a demo period.

## 7. Troubleshooting

> **Roman Urdu mein:** Error ko pehle artifact manifest, feature contract, history aur environment settings ke khilaf check karein; dimensions ko andazay se change na karein.

| Symptom | Check |
|---|---|
| `feature contract hash mismatch` | Reuse the same CSV schema, macro CSV, feature settings, and training contract. Do not hand-edit hashes; retrain or evaluate with the original contract. |
| `INSUFFICIENT_HISTORY` | Supply enough closed bars for `FEATURE_DROP_WARMUP`, the model window, and source warm-up. Check `LIVE_HISTORY_BARS`, missing rows, and timestamps. |
| Macro-feature error or missing macro values | Use the same `MACRO_CSV`/`--macro` with `time` and required `*_close` columns for training, evaluation, backtest, and live. |
| `action_dim` / `ALLOW_SHORT` mismatch | `allow_short=false` artifacts have two actions; three-action artifacts require `ALLOW_SHORT=true` to permit shorts. The live layer maps action 2 to flat when shorting is disabled. |
| `obs_dim mismatch` | The manifest must satisfy `obs_dim = window * n_features + 5`. Compare its feature-contract width and window; do not pad or truncate observations. |
| Missing or failed promotion record | Evaluate the artifact on a disjoint chronological period. Confirm `evaluation.json` exists, `passed` is true, and its `contract_hash` equals the manifest. |
| MCTS/ensemble member file not found | Keep relative member/checkpoint paths intact or recreate the artifact with valid references; do not copy only the top-level manifest. |
| Exported timestamps differ | Set `BROKER_UTC_OFFSET_HOURS` to server time minus UTC or override with `--utc-offset-hours`; exported timestamps represent UTC without a timezone suffix. |
