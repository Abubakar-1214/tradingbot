# Models and live architecture

> **Roman Urdu mein:** Is repo mein model code aur live safety wiring maujood hai, lekin koi trained model ship nahin hota. Har model ko apne data par train, evaluate, aur promotion gate se guzarna zaroori hai.

The repository contains training and inference implementations, not pre-trained
trading recommendations. No trained artifact is shipped. Existing strategies
have not beaten buy-and-hold net of costs; passing a software test or the
promotion gate is not evidence of future results.

## Shared observation and policy interfaces

> **Roman Urdu mein:** Har policy ko ek hi feature window aur account state milti hai, is liye training aur live inputs ka contract match rehna chahiye.

The observation is a `float32` vector with
`window * n_features + 5` values. The first `window * n_features` values are
the feature matrix flattened in row-major order: each bar's feature row in
time order, oldest to newest. The last five values are the scaled account
state, in this exact order:

1. `position`: `-1` short, `0` flat, `1` long.
2. `trade_pnl`: leveraged unrealized trade return, multiplied by 100 and
   clipped to `[-10, 10]` for the observation.
3. `bars_in_trade`: `log1p(bars_in_trade) / 5`.
4. `drawdown`: fractional drawdown from the equity peak.
5. `equity_ratio`: `log(equity / reference_equity)` with a small positive
   floor.

`AccountState` stores `trade_pnl` as a return before the vector scaling; for
example, a 1% return with default leverage `1.0` is stored as `0.01` and
becomes `1.0` in the observation vector. `obs_dim(window, n_features)` is
`window * n_features + 5`. The ordered feature names and fitted train-only
normalization values are kept in `feature_contract.json`; loading a model
checks that contract hash and observation width.

All learned policies implement the `TradingPolicy` interface in
`models/policy.py`: `reset()`, `act(obs) -> PolicyOutput`, and
`observe_executed(action)`. `PolicyOutput` contains an integer action, action
probabilities, confidence, and an `info` dictionary. Live signal sources
return `Decision(action, confidence, size_multiplier, reason, info)` after the
policy output is filtered.

### Action semantics

> **Roman Urdu mein:** Model ka action 0 position ko flat karta hai; rule source mein no-signal ka matlab hold hota hai.

| Index | Policy target | Model-source live mapping | Rule-source mapping |
|---:|---|---|---|
| `0` | Flat | Close an open position (`MODEL_EXIT`); stay flat otherwise | No crossover means hold; `NO_SIGNAL` is not an exit |
| `1` | Long | Hold if already long; otherwise close an opposing position and attempt a gated long entry | Fast SMA crosses above slow SMA |
| `2` | Short | Hold if already short; otherwise close an opposing position and attempt a gated short entry; maps to flat when `ALLOW_SHORT=false` | Fast SMA crosses below slow SMA |

An entry rejected by any filter or by `TradeExecutor` leaves the account flat
when an opposite position was first closed. No pyramiding is performed by the
model signal mapping. Confidence or consensus rejection holds the current
position instead of converting the rejected action into an exit.

## PPO (MLP)

> **Roman Urdu mein:** PPO ek MLP policy hai jo shared observation se flat, long ya short action choose karti hai.

PPO uses Stable-Baselines3's clipped Proximal Policy Optimization objective.
An MLP actor and value function learn from on-policy rollouts and
generalized-advantage estimates. The discrete policy produces a categorical
distribution over two or three actions.

- **Role:** Simple neural baseline for the shared trading environment.
- **Input:** The shared flattened feature window followed by the exact five
  account-state values defined above.
- **Output:** `PpoPolicy.act()` returns `PolicyOutput`; the live
  `ModelSignalSource` converts that output to a `Decision`.
- **Use:** Training: `train/train_ppo.py` and `env/gym_env.py`. Evaluation:
  `train/evaluate.py`. Backtest: `backtest/model_signals.py` and
  `scripts/run_backtest_eval.py`. Live: `live/model_signal.py` through
  `live/live_trade_mt5.py`.
- **Artifacts:** `model.zip`, chunk checkpoints (`checkpoint_*.zip`),
  `feature_contract.json`, `manifest.json`, and `evaluation.json`.
- **Manifest type:** `ppo`.
- **Limitations:** PPO is trained on the selected historical sample and
  environment cost assumptions. A small smoke run only tests the pipeline;
  it does not establish predictive value or suitability for deployment.

## Transformer-PPO

> **Roman Urdu mein:** Transformer-PPO har bar ko token banata hai aur attention se sequence ke context ko process karta hai.

The Transformer policy applies a PPO actor-critic update to a causal sequence
encoder. The observation is reshaped into `window` bar tokens; each token
contains its `n_features` values plus the same five account-state values.
Positional encoding preserves bar order, and the actor emits categorical
action probabilities.

- **Role:** Sequence-model alternative to the MLP PPO baseline.
- **Input:** Shared observation of width `window * n_features + 5`; token
  tensor shape is `(window, n_features + 5)`.
- **Output:** `TransformerPolicy` returns `PolicyOutput`; live wraps it in a
  `Decision`.
- **Use:** Training: `train/train_transformer.py`. Attention inspection and
  updates: `models/transformer_policy.py`. Evaluation, backtest, and live use
  the shared interfaces described above.
- **Artifacts:** `model.pt`, optional `checkpoint_*.pt`,
  `feature_contract.json`, `manifest.json`, and `evaluation.json`.
- **Manifest type:** `transformer`.
- **Limitations:** The training wrapper currently uses CPU. Attention weights
  expose model attention, not a causal explanation of market behavior.

## DreamerV3

> **Roman Urdu mein:** Dreamer world model mein market ke latent states seekhta hai aur imagined rollouts se policy train karta hai.

Dreamer learns an observation representation and recurrent state-space model
(RSSM), then trains its actor and critic on imagined trajectories. The replay
buffer stores transitions for sequence-based world-model updates. Checkpoints
include model configuration and training state for resume.

- **Role:** Model-based reinforcement-learning policy and the base artifact
  used by the MCTS, MAML, and adversarial workflows.
- **Input:** The shared observation vector; the encoder consumes its flattened
  values.
- **Output:** `DreamerPolicy` returns `PolicyOutput` with value and latent
  diagnostics in `info`.
- **Use:** Training: `train/train_dreamer.py`. Implementation:
  `models/dreamer_agent.py` and `models/dreamer_components.py`. Evaluation,
  backtest, and live use the shared policy interfaces.
- **Artifacts:** `model.pt`, optional `checkpoint_*.pt`,
  `feature_contract.json`, `manifest.json`, and `evaluation.json`.
- **Manifest type:** `dreamer`.
- **Limitations:** World-model prediction error and replay coverage can limit
  imagined-policy quality. A checkpoint is not evidence that the learned
  dynamics match future market conditions.

## Dreamer + MCTS

> **Roman Urdu mein:** MCTS Dreamer ke latent model mein mumkin actions ko search karta hai; yeh Dreamer policy ke upar planning layer hai.

Dreamer MCTS uses the Dreamer actor as an action prior and its latent dynamics
and critic to evaluate a search tree. It selects actions with PUCT, normalizes
observed Q values with min/max statistics, and applies the configured number
of simulations. Optional Dirichlet noise is available for search exploration;
inference defaults to deterministic root selection without root noise.

- **Role:** Planning wrapper over an existing Dreamer checkpoint; it does not
  train a new world model.
- **Input:** The same shared observation and action space as its Dreamer model.
- **Output:** `DreamerMCTSPolicy` returns `PolicyOutput`, with search
  statistics in `info`.
- **Use:** Manifest creation: `train/make_mcts_manifest.py`. Search:
  `models/mcts.py`. Evaluation: `train/evaluate.py`. Backtest/live use the
  model registry and shared interfaces.
- **Artifacts:** `manifest.json` and `feature_contract.json`; the manifest
  points to the existing Dreamer `model.pt` by a relative path. Evaluation
  writes `evaluation.json` alongside the MCTS manifest. Keep the referenced
  Dreamer artifact at its expected relative path.
- **Manifest type:** `dreamer_mcts`.
- **Limitations:** Search quality depends on the underlying Dreamer model,
  simulation count, and reward/value calibration. MCTS manifest creation
  alone does not evaluate or promote the artifact.

MCTS `PolicyOutput.info` includes `visit_counts`, `q_values`, `probs`,
`visit_counts_by_action`, `q_values_by_action`, `visit_distribution`, and
`root_visits`. `visit_distribution` is the root visit distribution used for
the policy probabilities; dictionary forms are keyed by action index.

## Ensemble

> **Roman Urdu mein:** Ensemble kai compatible policies ke votes ko weights ke mutabiq jama karta hai aur disagreement par flat target deta hai.

`EnsemblePolicy` combines member probability distributions using either soft
probability averaging or hard weighted voting. It pads two-action members with
zero short probability when mixed with three-action members. Weighted
agreement below `min_agreement` produces a flat target and marks consensus
false.

- **Role:** Aggregates compatible policy artifacts and exposes disagreement
  for live filtering.
- **Input:** Shared observation; every member must use the same observation
  width and exact feature-contract hash.
- **Output:** `PolicyOutput`; live `DecisionEngine` can reject non-consensus
  outputs.
- **Use:** Assembly or member training: `train/train_ensemble.py`. Inference:
  `models/ensemble.py`, recursively loaded by `models/registry.py`. Available
  to evaluation, backtest, and live through that registry.
- **Artifacts:** `ensemble.json`, `feature_contract.json`, `manifest.json`
  with relative member-manifest references, and `evaluation.json`.
- **Manifest type:** `ensemble`.
- **Limitations:** Members must share the same contract and observation
  dimensions. Agreement is not a probability calibration guarantee; member
  errors can be correlated.

`EnsemblePolicy.info` reports `member_actions`, `agreement`, `consensus`,
`uncertainty`, `epistemic`, `epistemic_uncertainty`, and
`epistemic_disagreement`. The last three epistemic names currently carry the
same KL-based disagreement value.

## MAML adaptation

> **Roman Urdu mein:** MAML Dreamer ke world model ko mukhtalif historical regimes ke tasks par meta-train karta hai; recent adaptation alag artifact banati hai.

The MAML workflow builds causal market-regime segments from past returns and
splits each accepted segment into support and query portions. First-order
adaptation updates Dreamer's world-model parameters on task support data and
uses task query loss for the meta-update. Recent-history adaptation operates
on a copy of a Dreamer agent and writes a separate artifact.

- **Role:** Optional initialization and short-horizon adaptation workflow for
  Dreamer; it does not alter the source artifact in place.
- **Input:** The shared feature matrix and returns, plus an existing Dreamer
  checkpoint and matching contract.
- **Output:** Adapted Dreamer weights, loaded as `DreamerPolicy` and exposed
  through `PolicyOutput`.
- **Use:** Meta-training: `train/meta_train_dreamer.py`. Recent adaptation:
  `train/adapt_recent.py`. Core implementation:
  `models/meta_learning.py`. Resulting Dreamer artifacts can be evaluated,
  backtested, or loaded live.
- **Artifacts:** New `model.pt`, `feature_contract.json`, `manifest.json`,
  and `evaluation.json`; metadata marks meta-training or adaptation.
- **Manifest type:** `dreamer`.
- **Limitations:** Only first-order MAML is implemented. Regimes shorter than
  the configured minimum are skipped; adaptation can overfit a short recent
  window and must be evaluated independently.

## Adversarial robustness fine-tune

> **Roman Urdu mein:** Adversarial fine-tune Dreamer ko temporary spread, slippage aur stop-hunt perturbations ke khilaf train karta hai, phir clean data par evaluate karta hai.

The self-play workflow pairs a Dreamer trader with a learned market-maker
policy. The market maker chooses no perturbation, spread widening, increased
slippage, or a stop-hunt gap, subject to a manipulation-rate cap and cost.
Perturbations affect one environment step; the trader records replay
transitions and receives Dreamer training updates.

- **Role:** Optional fine-tuning against bounded execution perturbations.
- **Input:** The shared observation and a Dreamer manifest with a matching
  training-data contract.
- **Output:** A newly saved Dreamer artifact; inference returns
  `PolicyOutput`.
- **Use:** Training: `train/train_adversarial.py`. Environment and agents:
  `models/adversarial_training.py` and `env/dreamer_trading_env.py`.
  Evaluation uses the clean environment; the artifact can then be backtested
  or loaded live.
- **Artifacts:** New `model.pt`, `feature_contract.json`, `manifest.json`,
  and clean-environment `evaluation.json`.
- **Manifest type:** `dreamer`, with adversarial-fine-tuning metadata in
  `extra`.
- **Limitations:** Perturbations are stylized, one-step simulations and do
  not cover all broker or market failures. Self-play does not establish
  robustness outside the specified perturbation model.

## Non-neural decision and safety layers

> **Roman Urdu mein:** Yeh deterministic layers signal ko filter, size aur execute karti hain; neural model risk checks ko bypass nahin kar sakta.

### Rule SMA source

> **Roman Urdu mein:** Rule source fast aur slow SMA crossover par signal deta hai; crossover na ho to position hold hoti hai.

`RuleSignalSource` uses the latest two closed-bar fast/slow SMA values
(defaults 20 and 50). A cross produces action 1 or 2; no cross returns
`Decision(reason="NO_SIGNAL", info={"hold": True})`.

- **Role/input/output:** Non-neural live signal source; closed OHLC bars and
  current `AccountState` in, `Decision` out.
- **Use:** `live/live_trade_mt5.py::RuleSignalSource`; no train or
  model-manifest path.
- **Artifacts / manifest type:** None / not applicable.
- **Limitations:** A simple crossover signal with no fitted model or learned
  feature contract; no signal is deliberately a hold, not an exit.

### DecisionEngine filters and Kelly scaling

> **Roman Urdu mein:** DecisionEngine confidence, consensus, session aur cooldown rules lagata hai; Kelly size ko risk limit ke andar rakhta hai.

`DecisionEngine` preserves holds for low confidence, absent consensus, and
rule-source no-signal decisions. It blocks only new entries during configured
UTC sessions and post-loss cooldowns; exits remain available. Optional
fractional Kelly uses recent realized P&L as a fraction of closing equity and
scales against `MAX_RISK_PER_TRADE`, clamped to a size multiplier in `[0, 1]`.

- **Role/input/output:** Filters `PolicyOutput` or a rule `Decision`, UTC bar
  close time, current side, cooldown state, and recent trades into `Decision`.
- **Use:** `live/decision_engine.py`, invoked by `live/live_trade_mt5.py`.
- **Artifacts / manifest type:** None / not applicable.
- **Limitations:** Kelly estimates depend on a small, changing trade history;
  fewer than `KELLY_MIN_TRADES` uses multiplier `1.0`. It is not a guarantee
  of risk control; the independent risk supervisor still applies.

### ATR position sizing

> **Roman Urdu mein:** ATR sizing stop distance aur equity risk se lot volume nikalti hai, phir broker ke lot step ke mutabiq neeche round karti hai.

The live `ATRPositionSizer` derives a volume from equity, configured risk per
trade, entry price, and causal ATR-based stop distance. `TradeExecutor` floors
volume to `LOT_STEP`, applies the bounded decision multiplier, caps aggregate
notional exposure, and rejects a result below `MIN_LOT`.

- **Role/input/output:** Converts a requested side and market/account context
  into a broker volume and SL/TP request.
- **Use:** `models/position_sizing.py`, called from `live/trade_executor.py`.
- **Artifacts / manifest type:** None / not applicable.
- **Limitations:** Actual exposure depends on broker contract size, lot rules,
  fill price, and costs; a lower-than-minimum risk size is rejected rather
  than rounded up.

### TradeManager

> **Roman Urdu mein:** TradeManager khuli position par breakeven, tightening trailing stop, ek martaba partial close, aur time stop manage karta hai.

`TradeManager` compares the close against initial risk in R units. It can
raise a long stop or lower a short stop, request a one-time partial close
after lot-step flooring, or close a position after the configured bar limit.

- **Role/input/output:** Open `PositionInfo`, entry ATR, bars held, latest bar,
  current ATR, and trade metadata in; typed close/SL/partial-close actions out.
- **Use:** `live/trade_manager.py`, applied before the policy decision in
  `live/live_trade_mt5.py`; all resulting broker operations use the executor.
- **Artifacts / manifest type:** Trade metadata is persisted with local state;
  no model artifact / not applicable.
- **Limitations:** Stop triggers depend on broker execution and bar data.
  Partial closes can be skipped when the remaining volume would violate the
  minimum lot.

### RiskSupervisor

> **Roman Urdu mein:** RiskSupervisor daily loss, drawdown, trade count aur market conditions par circuit breakers chalata hai.

`RiskSupervisor` is a deterministic, SQLite-backed circuit-breaker layer. It
tracks account and trade history and reviews new exposure against configured
daily-loss, drawdown, consecutive-loss, trade-rate, spread, volatility, and
related gates.

- **Role/input/output:** Current signal, requested risk fraction, account
  state, and market data in; an approval decision and reason out.
- **Use:** `models/risk_supervisor.py`; every live entry or scale is routed
  through it by `live/trade_executor.py`.
- **Artifacts / manifest type:** `state/risk_state.db`; no model artifact /
  not applicable.
- **Limitations:** It depends on accurate broker/account data and local
  persistence. It does not predict losses or replace broker-side controls.

### TradeExecutor

> **Roman Urdu mein:** TradeExecutor har order ko size, SL/TP, lot limits aur RiskSupervisor ke through broker tak bhejta hai.

`TradeExecutor` owns the live order path, risk review, ATR sizing, SL/TP,
idempotent intent handling, and local position/trade metadata. Entries and
scales require risk approval; close operations are treated as de-risking and
are not blocked by an entry rejection.

- **Role/input/output:** Decision, account/market context, and closed bars in;
  broker order result and persisted state out.
- **Use:** `live/trade_executor.py`, wired into the closed-bar loop in
  `live/live_trade_mt5.py`.
- **Artifacts / manifest type:** `state/bot_state.json` and adjacent
  idempotency state; no model artifact / not applicable.
- **Limitations:** Broker rules and real fill conditions may differ from the
  mock broker. A successful software-side approval does not guarantee a fill.

## Live processing order

> **Roman Urdu mein:** Har band candle par pehle safety checks aur stops reconcile hote hain; policy sirf uske baad decision leti hai aur executed action feedback hota hai.

The live loop processes one new closed bar at a time. A kill switch or halted
executor stops processing before decisions. Broker stop/take-profit fills are
reconciled before `TradeManager` actions. The signal source then transforms
features under the saved contract, evaluates the policy, and returns the
actual post-filter/post-execution side through `executed()`.

```text
new closed bar
    |
    v
kill switch -> risk halt check
    |
    v
broker SL/TP fill reconciliation
    |
    v
TradeManager actions on existing positions
    |
    v
feature pipeline + saved feature-contract validation
    |
    v
policy / signal source
    |
    v
DecisionEngine confidence, consensus, session, cooldown, Kelly filters
    |
    v
TradeExecutor -> RiskSupervisor approval and position sizing
    |
    v
broker order / close / stop modification
    |
    v
signal_source.executed(actual broker position action)
    |
    v
persist bot state, trade metadata, and risk state
```

Rule mode retains its crossover semantics (`NO_SIGNAL` means hold). Model mode
interprets flat as an exit. Opposite-side decisions close first and only enter
the new side if all decision and execution gates approve it.
