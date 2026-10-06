

---

## 4. What to show during trading (MT5-style operations view)

Trading runs as a **separate child process** (`live/live_trade_mt5.py`), not inside the dashboard. The dashboard reads persisted state (`state/*.json`, `state/risk_state.db`), watches the kill-switch file, and subscribes to decision events over WebSocket. It **never** sends orders itself (Hard Safety Invariant, Section 5.7). **[FACT + ENGINEERING RECOMMENDATION]**

### 4.1 Account summary panel (MT5-style)

| Stat | Data source (verified) |
|---|---|
| Balance | `MockBroker.account_info()` → `balance` (persisted in `mock_state.json` / `bot_state.json`) **[FACT]** |
| Equity | `RiskSupervisor.get_statistics()['current_equity']` and/or broker account `equity` **[FACT `models/risk_supervisor.py`]** |
| Drawdown | `get_statistics()['daily_loss_ratio']` vs `max_daily_loss`, plus env/report `max_drawdown_pct` **[FACT]** |
| Daily PnL | `get_statistics()['daily_pnl']` **[FACT]** |
| Consecutive losses | `get_statistics()['consecutive_losses']` **[FACT]** |
| Halt remaining | `halt_remaining_seconds()` / `get_statistics()['halt_remaining_sec']` — 365-day halt after `emergency_shutdown()` **[FACT]** |

Also show `get_statistics()` totals: `total_checks`, `approved`, `rejected`, `approval_rate`, `rejection_rate`, `trades_today`. **[FACT]**

### 4.2 XAUUSD candlestick chart with entry / SL / TP overlays

Chart the live (or replayed/smoke) price feed with candles (open/high/low/close + volume). Overlay, per active position:

- **Entry** line (from `mock_state.json` `closed[].entry` and live position `entry`),
- **SL / TP** lines (from `live/trade_executor.py::entry_sl_tp` — `sl_atr_mult=2.0`, `tp_atr_mult=3.0` defaults from `BrokerConfig`) **[FACT]**.

**[ENGINEERING RECOMMENDATION]** Use a canvas charting library (e.g. lightweight-charts) for decimation of thousands of candles. Data source: broker history/bars (H1 default `TIMEFRAME=H1` **[FACT `core/config.py`]**); the smoke-state `mid` (`mock_state.json`) is the last known price.

### 4.3 Positions table + closed-trades table

**Exact shape: `artifacts/_smoke_state/mock_state.json` [FACT]:**

```json
{
  "balance": 99999.96,
  "mid": 1943.90,
  "next_ticket": 11,
  "positions": [ ],
  "closed": [
    { "ticket": 1, "side": "buy", "volume": 0.05,
      "entry": 1985.17, "exit": 1980.79, "pnl": -0.22, "reason": "SL" },
    { "ticket": 4, "side": "sell", "volume": 0.05,
      "entry": 1966.70, "exit": 1959.38, "pnl": 0.36, "reason": "TP" },
    { "ticket": 10, "side": "sell", "volume": 0.05,
      "entry": 1949.17, "exit": 1944.14, "pnl": 0.25, "reason": "manual_close" }
  ]
}
```

Positions table columns: `ticket, side, volume, entry, sl, tp, current price, pnl`. Closed-trades columns: `ticket, side, volume, entry, exit, pnl, reason` with reason chips (`SL`/`TP`/`manual_close`/...). **[ENGINEERING RECOMMENDATION]** `bot_state.json` adds `last_bar_time` (UTC ISO) and `positions` map for live-loop progress. **[FACT]**

### 4.4 AI decision feed

The heart of the operations view. Each decision event, pushed over `/ws/live/decisions`, shows the full chain:

| Stage | Fields | Data source (verified) |
|---|---|---|
| 1. Inputs | window features + `[position, trade_pnl, bars_in_trade, drawdown, equity_ratio]` account vector; obs dim = `window*n_features + 5` | `env/dreamer_trading_env.py` observation; `core/model_artifacts.py` manifest check **[FACT]** |
| 2. Model probs | per-action probability vector | `models/registry.py` policy; `policy_probs(state)` / softmax **[FACT]** |
| 3. Confidence / agreement | `confidence`; `agreement` | `live/model_signal.py`; `models/ensemble.py` `info['agreement']` **[FACT]** |
| 4. Uncertainty | `uncertainty` (normalized entropy), `epistemic_uncertainty` | `models/ensemble.py` `act()` info **[FACT]** |
| 5. Decision filter | `reason` ∈ `NO_SIGNAL, HOLD, LOW_CONFIDENCE, NO_CONSENSUS, NO_EDGE, ENTRY_BLOCKED...` | `live/decision_engine.py::filter` **[FACT]** |
| 6. Risk check | `approved` + `reason` (risk reason strings below) | `RiskSupervisor.check_trade` **[FACT]** |
| 7. Order | `side, lot (volume), sl, tp` | `TradeExecutor.execute_entry` — volume floored to `LOT_STEP=0.01`, `MIN_LOT=0.01` **[FACT]** |
| 8. Fill | `retcode` + `retcode_name` + fill price | `live/broker.py` `OrderResult` + `RETCODE_NAMES` **[FACT]** |
| 9. Equity / reward | equity after fill; reward signal | `mock_state.json` balance / `bot_state.json`; env info `equity` **[FACT]** |

Retcode rendering (from `live/broker.py` **[FACT]**): success = `10008 ORDER_PLACED`, `10009 DONE`, `10010 DONE_PARTIAL`; transient = `10004 REQUOTE`, `10020 PRICE_CHANGED`, `10021 PRICE_OFF` (auto-retry, `max_requote_retries=3`); failure = `10006 REJECT`, `10016 INVALID`, `10018 MARKET_CLOSED`, `10019 NO_MONEY`, `10017 TRADE_DISABLED`, etc.

### 4.5 Brain panel (per model type)

- **Dreamer** (`models/dreamer_agent.py`): render RSSM latent stats (`stoch_dim=32`, `num_categories=32`), the imagination rollout (horizon `15`) as a small multi-branch tree/timeline, critic value estimates, and world-model reconstruction loss live during training. **[FACT]** `compute_world_model_loss` returns `(loss, metrics)` incl. `states` — bind to the panel.
- **Transformer** (`models/transformer_policy.py`): `attention_weights` → heatmap of heads x positions; plus `train_step` metrics `{loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction}`. **[FACT]**
- **Ensemble** (`models/ensemble.py`): member votes, `agreement`, `consensus` (bool vs `min_agreement=0.6`), `uncertainty`, `epistemic_uncertainty`. **[FACT]**
- **MCTS** (`models/mcts.py`): per-action `visit_counts`, `q_values`, `visit_distribution`, `root_visits`, `probs` from `MCTS.search()`. **[FACT]**
- **Adversarial** (`models/adversarial_training.py`): `manipulation_rate` and per-episode `manipulation_count`; env perturbation type per step (`manipulation_type` from `action_names`); `SelfPlayTrainer.history` (`trader_wins[]`, `mm_profits[]`). **[FACT]**
- **Meta (MAML)** (`models/meta_learning.py`): regime task list `{name, label, start, end, support, query}` and per-epoch `meta_loss`/`support_loss`. **[FACT]**

**[UNPROVEN]** — note: "imagination rollout quality implies alpha" is unproven (FINAL study Q2). The panel shows internals for audit/explainability, not as a performance claim.

### 4.6 Risk circuit-breaker panel

**Every rejection reason from `RiskSupervisor` rendered as a status chip, with the exact strings [FACT `models/risk_supervisor.py`]:**

| Chip | Meaning |
|---|---|
| `APPROVED` | trade passed all checks |
| `CIRCUIT_BREAKER: daily loss …` | daily-loss breaker |
| `HALTED until …` | emergency halt (365 d) active |
| `MAX_DRAWDOWN: …` | drawdown > `max_drawdown` |
| `POSITION_TOO_LARGE: …` | size > `max_position` |
| `TOO_MANY_LOSSES: …` | ≥ `max_consecutive_losses` |
| `HIGH_VOLATILITY: …` | vol > `vol_threshold` |
| `CORRELATION_GUARD: …` | DXY correlation block |
| `EVENT_RISK: …` | news event scale-down |
| `MAX_TRADES: …` | daily trade cap |
| `COOLDOWN: …` | min interval not elapsed |
| `SPREAD_TOO_WIDE: …` | spread > `max_spread` |
| `MARKET_CLOSED: …` | outside market hours |

Panel also shows `get_statistics()` counters (`rejection_reasons` dict) as a per-reason bar chart, plus `emergency_shutdown()`/halt countdown. **[FACT]**

### 4.7 Economic calendar event feed

**Data source: `features/calendar_features.py` (verified).** Events arrive with `time, currency, impact, forecast, previous` shape; the feature pipeline shifts them into the model input (`macro_shift=1`). The feed panel colors by impact (high/medium/low) and flags `EVENT_RISK`-style near-event trades. **[ENGINEERING RECOMMENDATION]** The calendar is advisory input to the model, never an order source.

### 4.8 Kill switch + TRADING_MODE badge + REQUIRE_PROMOTED_MODEL

- **Kill switch**: a big red button that writes the **kill-switch file** (`KILL_SWITCH` path from `PathConfig`, default `KILL_SWITCH`) — the live loop (`live/live_trade_mt5.py`) polls this file and exits promptly. **[FACT]** The dashboard should also surface whether the file exists (armed state).
- **TRADING_MODE badge**: `demo` (green, MockBroker) vs `live` (red, MT5), from `core/config.py::TradingMode`. **[FACT]**
- **REQUIRE_PROMOTED_MODEL badge**: show the honest gate state — since no model has passed promotion, the badge reads **"BLOCKED — no promoted model"** and the live loop refuses model-driven trades. **[FACT]** (from `core/config.py` `require_promoted_model=True` default + honest `evaluation.json` facts.)

### 4.9 Reconciliation status

`live/trade_executor.py::startup_reconcile()` returns a `ReconResult` (`live/broker.py`): matched/mismatched positions, drift, `ok` flag; drift → `halt_reason = "startup reconciliation drift: ..."` and the executor halts. **[FACT]** Render: status pill (OK / DRIFT), drift magnitude, timestamp, and the halt banner. This feeds `check_live_gates(..., reconciliation_passed=...)`. **[FACT]**

### 4.10 Log tail (live loop)

Tail of the live child's captured stdout/stderr (supervisor tee + `/ws/logs`), with follow-toggle. Display-only; no metric fabrication. **[ENGINEERING RECOMMENDATION]**
