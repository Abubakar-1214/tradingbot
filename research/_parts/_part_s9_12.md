# Section 9 — Market Regime Detection

## 9.1 Why regime matters for XAUUSD

**[FACT]** Financial time series exhibit regimes: periods of trending, ranging, high-volatility, low-volatility behavior with different statistical properties. Ignoring regimes means a policy averages over contradictory dynamics and risk parameters that are wrong in at least one regime. (Definitional; standard regime-switching literature — e.g., Hamilton's regime-switching models; HMM applications in finance.)

**[FACT]** The project already has explicit, label-based regime machinery in `models/meta_learning.py::MarketRegimeGenerator`: rolling statistics (trend_return, trend_volatility, volatility_48, expanding median) classify windows as high_vol (>1.5× median), low_vol (<0.5× median), trend_up/trend_down, range, and unknown; MAML (`MAMLTrader`) segments data into support/query tasks per regime. (Source: direct file reads.)

## 9.2 Regime detection options compared

| Option | What it is | Training | Strengths | Weaknesses | Verdict |
|---|---|---|---|---|---|
| **Separate regime model (e.g., HMM)** | Explicit latent-state model of market regime (HMM/regime-switching) | Unsupervised EM on returns/features | Interpretable states; standard in finance; RegimeRL uses explicit regime models (github sahilapage); strong evidence base (HMM regime literature) | Regime boundaries are fuzzy; labels are noisy; separate model adds a failure point | Adopt as the *risk/context* layer (project's MAML labels are the seed) |
| **Shared representation (implicit)** | Regime emerges from the learned latent (RSSM z_t) | World-model loss | No extra supervision; latent already encodes dynamics | Not auditable as "regime"; can't report regime to humans/risk | Keep as the *market-state* latent, but don't call it regime |
| **World-model conditioned** | Regime id conditions the world model (per-regime dynamics) | Joint world-model + regime loss | Most faithful dynamics per regime | More complex; needs enough data per regime | **Experiment** (Section 20; Proposed Experiments) |
| **Implicit in policy only** | Policy learns to act differently per regime without naming it | RL | Simple | No risk-layer visibility; policy can't explain itself | Not sufficient alone |
| **Ensemble of regime experts** | MoE-style per-regime experts (Section 19) | Mixture training | Forgetting-resistant (Section 7.4); specialized competence | Training complexity; gate training needs labels | Later phase experiment |

## 9.3 Evidence and recommendation

**[RESEARCH FINDING]** Regime-aware RL is a documented direction: RegimeRL (github sahilapage) explicitly combines a market regime model with risk-aware RL for trading, positioning regime detection *before* the RL policy rather than learning it implicitly. The HMM/regime-switching literature is mature and interpretable, which is why explicit regime layers dominate in production-adjacent quant work. (Verified via internet_search.)

**[ENGINEERING RECOMMENDATION]** Single evidence-based choice: **keep an explicit, interpretable regime module (HMM-style or the project's rolling-statistics labels) as the risk/context layer, and feed its output (regime id + regime statistics) into (a) the risk supervisor, (b) position sizing, and (c) as conditioning context for the world model.** Do not build a separate learned black-box regime model in parallel — the project's `MarketRegimeGenerator` already produces the labels; the upgrade is (i) HMM smoothing for stable boundaries, (ii) persistence of regime records into the long-term memory store (Section 8), and (iii) regime-conditioned risk parameters.

**[ENGINEERING RECOMMENDATION]** Map to the project: `RiskSupervisor` today gates on vol/spread/correlation deterministically; the regime module should upgrade those thresholds from static config to regime-conditional values (e.g., spread filter scaled by regime volatility; event halving only in high-vol regimes). MAML's regime tasks should continue to be the meta-training segmentation — this is evidence-aligned (MAML meta-learning across regimes is a documented approach for non-stationarity).

---

# Section 10 — Action Space Design

## 10.1 The current action space and its limits

**[FACT]** The project's RL agents act in a 3-discrete action space: flat (0), long (1), short (2) — `RealisticTradingEnv` and all policy wrappers (`PpoPolicy`, `DreamerPolicy`, `TransformerPolicy`, `DreamerMCTSPolicy`) expose `action_dim=3`. Position size, SL/TP, and holding period are handled by *separate deterministic* layers: `ATRPositionSizer` (risk_per_trade=0.02, sl_atr_mult/tp_atr_mult), `KellyPositionSizer` (kelly_fraction=0.25, capped at 0.10), and the live executor's `entry_sl_tp`. (Source: direct file reads.)

**[RESEARCH FINDING]** Multi-agent LLM frameworks (QuantAgent — github THU-MIG/QuantAgent; TradingAgents — arXiv 2412.20138) represent trading as a *full workflow* — analysis, plan formation (direction, entry, sizing, stop/take-profit, risk assessment), and execution — rather than a raw action. TradingAgents reports that structured multi-agent debates over such plans improved cumulative/risk-adjusted return in its test setups. This is evidence that *structured plans* are a useful action representation, even if the multi-agent execution path is not recommended wholesale (Section 2.6).

## 10.2 Options for the action space

| Option | What the agent outputs | Pros | Cons | Verdict |
|---|---|---|---|---|
| **BUY/SELL/HOLD (3 discrete)** | Direction only | Simple; current project state | Sizing/SL/TP/exit are outside the learned policy; suboptimal risk behavior; cannot learn "scale in" or "exit early" | Baseline only |
| **Direction + confidence** | Direction + confidence value | Lets risk layer scale by confidence (project `PolicyOutput.confidence`) | Confidence must be calibrated (Section 16); still no SL/TP/exit learning | Keep as *input* to sizing, not the whole answer |
| **Full trading plan (joint)** | (direction, entry threshold, size, SL, TP, holding period, exit trigger) as a structured action | Policy learns all risk decisions jointly; most expressive | High-dimensional; harder credit assignment (Section 7); needs a structured action encoder/decoder; risk of unstable gradients | **Recommended direction** (with a pragmatic two-stage rollout) |
| **Hierarchical (macro/micro)** | High-level (regime-based stance) + low-level (execution details) | Decomposes credit assignment; mirrors how humans trade; QuantAgent-style plan→execute | More components; coordination overhead | **Adopted in a specific form**: regime module sets stance; policy sets plan; deterministic risk/execution layer refines |
| **Continuous action** | Continuous vectors (position fraction, etc.) | Fine-grained | Needs clipping/interpretation for discrete FX conventions (lot steps) | Used only for sizing parameters, not direction |

## 10.3 Joint vs hierarchical learning — evidence and recommendation

**[RESEARCH FINDING]** The DRL-trading literature and agent-workflow research both converge on *structured, staged decision-making*: separate the decision (what to do) from execution details (how to do it), because joint end-to-end optimization of full order plans from raw prices is sample-hungry and hard to validate (Section 2.5; DRL-trading survey skepticism). QuantAgent/TradingAgents demonstrate the *workflow* benefit in LLM settings; the same staging principle applies to numeric RL.

**[ENGINEERING RECOMMENDATION]** Single evidence-based design — a **structured plan with staged learning**:
1. **Policy outputs a compact plan:** direction ∈ {flat, long, short} + confidence + *desired holding horizon* (a small set of classes: e.g., scalp ≤5 candles, swing ≤24 candles, position ≥24) — all discrete, easy to learn and audit.
2. **Deterministic risk/sizing layer converts the plan to orders:** size from risk budget + volatility (existing `ATRPositionSizer`/`KellyPositionSizer` logic, upgraded with model-based vol from Section 3.6); SL/TP from ATR multiples (existing `entry_sl_tp`) adjusted by regime; event windows halve size (existing `RiskSupervisor` event_position_scale).
3. **Holding-period learning:** the horizon class lets the policy learn "this signal is short-lived" — a genuine learned exit decision without exposing the policy to raw continuous exit timing. The `Advance`-style exit logic in the live loop (`on_closed_bar` → manage_open_positions) stays deterministic around it.

**[ENGINEERING RECOMMENDATION]** This keeps the learned surface small (discrete, interpretable), preserves the deterministic safety shell, and directly upgrades the project from 3-raw-actions to plan-based actions without a risky full-plan end-to-end jump. The project implication: `RealisticTradingEnv` action space becomes (direction, horizon_class); `DecisionEngine` consumes the plan; `TradeExecutor` already has all the deterministic machinery (`entry_sl_tp`, `fraction_and_volume`, `BELOW_MIN_LOT` handling).

---

# Section 11 — Reward Function Design

## 11.1 Why reward is the single most important design choice

**[FACT]** In RL, the reward function *is* the objective: whatever it measures, the agent will optimize it, including its loopholes (reward hacking). In trading, the reward must encode net-of-cost, risk-adjusted performance — anything less trains an agent to optimize something other than surviving and growing equity. (Definitional RL; DRL-trading literature.)

**[FACT]** The project's current reward is scaled log-return (`env/dreamer_trading_env.py`), with costs (spread/commission/slippage/swap) applied in the environment — cost-aware but *not* risk-adjusted, drawdown-aware, or multi-objective. (Source: direct file reads.)

## 11.2 Reward candidates compared

| Reward | What it optimizes | Strengths | Weaknesses | Verdict for XAUUSD |
|---|---|---|---|---|
| **Raw P&L / log-return** | Total profit | Simple; cost-aware if env charges costs | No risk penalty; can train high-volatility, high-drawdown behavior; variance-blinded | Reject as sole reward (project's current state — upgrade) |
| **Risk-adjusted (Sharpe/Sortino contribution)** | Return per unit of risk | Standard; aligns with survivability | Sharpe needs return *samples* (requires episode/segment bucketing); Sortino better for downside | **Adopt** as the core reward component (per-segment risk-adjusted return) |
| **Drawdown-penalized** | Avoid large equity drawdowns | Directly controls ruin risk; pairs with `RiskSupervisor` max_drawdown breaker | Can be overly conservative if weight too high | **Adopt** as a penalty term (project already has drawdown breakers — make it part of the reward) |
| **Profit factor** | Wins/losses ratio | Intuitive; resists overtrading | Ignores magnitude of tail risk; noisy on few trades | Auxiliary metric (Section 16), not primary reward |
| **Costs/slippage/execution quality** | Net-of-cost behavior | Critical honesty component; penalizes overtrading | Must be measured carefully (spread model) | **Adopt** — already in env; add explicit execution-cost penalty terms |
| **Multi-objective composite** | Weighted combination of the above | Flexible; can balance return, downside, differential return, Treynor | Weight tuning; reward hacking between terms | **Adopt** with the evidence-based composite below |

## 11.3 Evidence for risk-aware rewards

**[RESEARCH FINDING]** A risk-aware RL trading reward (arXiv 2506.04358) proposes a *composite differentiable reward* combining: base return, downside risk penalty, differential return (vs a baseline — directly relevant to the project's buy-and-hold honesty check), and a Treynor-like risk-adjusted term. The design goal is to make the RL objective risk-aware *within the gradient signal*, not just as a post-hoc filter.

**[RESEARCH FINDING]** TorchTrade (github ai4finance/torchtrade) implements RL trading environments/frameworks in PyTorch with several risk-aware reward/observation options, providing a reproducible baseline for risk-aware RL rewards. Pro-Trader-RL (github comach/Pro-Trader-RL) similarly shows RL trading policies trained on reward structures including risk/drawdown terms. RegimeRL (github sahilapage) adds regime-conditioned rewards. (All verified via internet_search; details as reported in the papers/repos.)

## 11.4 The recommended reward for XAUUSD

**[ENGINEERING RECOMMENDATION]** Single evidence-based reward design — a **per-trade-segment composite**:
```
R_t = α·Sharpe_segment + β·(differential_return vs buy-and-hold baseline)
      − γ·downside_deviation_segment − δ·(spread+commission+slippage paid)
      − ε·drawdown_penalty(equity_curve)
```
with per-segment (e.g., 24-candle) risk-adjusted return as the backbone (aligns with the critic's distributional value, Section 6.2.3), a differential term against the buy-and-hold baseline (honest: the project must beat buy-and-hold to be worth running — the README shows rule strategies do not), explicit cost penalties, and a drawdown penalty mirroring the deterministic `RiskSupervisor` breaker thresholds (max_drawdown 0.15, daily_loss 0.05, consecutive_losses 5) so the *learned* objective and the *deterministic* shell agree.

**[ENGINEERING RECOMMENDATION]** Calibration guardrails: (a) reward weights must be validated in a small grid with purged walk-forward (never tuned on the test path); (b) the reward must be invariant to the broker's lot-step rounding (i.e., measured on *executed* fills, not theoretical sizes — the env already returns realistic fills; keep that); (c) episode termination for the RL loop should coincide with risk-state resets (daily loss breaker) so drawdown penalties propagate through the episode boundary.

---

# Section 12 — The Realistic Trading Environment

## 12.1 Backtester vs market simulator vs world model

| Layer | What it is | Used for | Evidence/status in project |
|---|---|---|---|
| **Backtester** | Deterministic replay of historical OHLC with a fill model | Strategy validation, honesty checks | `backtest/engine.py` on kernc/backtesting.py with CostModel (one-way spread 0.000175, commission 0.00003, round-trip 0.000410); walk-forward purge+embargo; README: rule strategies do NOT beat buy-and-hold net of costs |
| **Market simulator** | Stochastic generative model of price/order-flow (LOB simulators, synthetic series) | RL training at scale; stress testing | `SyntheticBarSource` (seed 7) in the live stack for demo; no LOB-level simulator yet |
| **World model** | Learned dynamics (RSSM) for imagination/planning | RL training + what-if (Section 17) | `models/dreamer_agent.py` RSSM (faithful) — trained on indicator features, not raw OHLCV yet |

**[RESEARCH FINDING]** Multi-agent LOB reinforcement learning (arXiv 2006.05574) models a limit order book as a multi-agent environment and trains RL agents at LOB level — evidence that *order-flow-level* simulation is tractable and that microstructure realism changes learned behavior. JAX-LOB (arXiv 2308.13289) is the first GPU-accelerated LOB simulator, processing thousands of books in parallel (≥5× speedup vs comparable CPU), used as a gymnax RL environment for large-scale RL trading (repo KangOxford/AlphaTrade). TRADES (arXiv 2502.07071) targets realistic LOB generation — evidence that generative market models are an active frontier (Section 19).

## 12.2 What a realistic FX environment must include (and the project's coverage)

| Element | Required for realism | Project status (RealisticTradingEnv + live stack) |
|---|---|---|
| Bid/ask with spread | Long pays ask, short receives bid | ✅ `env/dreamer_trading_env.py`; `backtest/costs.py` documents verified fill semantics (fill = price·(1±spread)) |
| Commission | Fixed + relative per side | ✅ CostModel commission 0.00003/side |
| Slippage | Adverse, vol-scaled | ✅ env vol-scaled slippage; costs.py folds slippage into one-way spread (conservative) |
| Swap/overnight | Triple Wednesday for FX/metals | ✅ env swap, Wed triple |
| Partial fills | Realistic for size>liquidity | ❌ absent (discrete fill model) — acceptable for the current sizing cap (10%) but must be added before larger scale |
| Latency/queue | Execution delay and queue position | ❌ absent — out of scope for candle-close cadence; needed only for intra-candle strategies (not recommended) |
| Liquidity limits | Notional caps, market impact | ⚠️ partial: aggregate notional cap in `TradeExecutor`, MIN_LOT/floor handling; no impact model |
| Margin/leverage | Contract size, margin calls | ⚠️ partial: XAUUSD_CONTRACT_SIZE=100, risk cap 10%, BELOW_MIN_LOT rejection; no broker-margin-call simulation |
| Position state/equity/drawdown | Episode state | ✅ `AccountState` (position, trade_pnl, bars_in_trade, drawdown, equity_ratio) in `core/observation.py`; episode max-DD breaker in env |
| Trading hours | Market close/weekend gaps | ⚠️ `RiskSupervisor` market_hours_only config (default False); weekend gap risk not modeled |
| News/events | High-impact windows | ✅ event_position_scale=0.5 halving; `is_high_impact_event` placeholder in `build_market_data` (needs real calendar feed) |
| Costs during imagination | RL must dream *with* costs | ⚠️ env has costs; world-model imagination must include them (Section 6.3) |

## 12.3 Backtester vs market simulator — which for what

**[RESEARCH FINDING]** The backtest-overfitting literature (Lopez de Prado: CPCV/PBO/PSR/DSR, SSRN 2460551; purged k-fold; "How To Backtest Correctly" repo) establishes that *evaluation* must be done on multiple purged/embargoed historical paths, not a single walk-forward path — the project's `walk_forward` already implements the purge+embargo structure, though it tests fixed rule parameters only.

**[ENGINEERING RECOMMENDATION]** Single evidence-based division of labor:
1. **Evaluation (honesty):** the deterministic backtester with realistic costs (`backtest/engine.py`) — never the simulator — because evaluation must be reproducible and conservative. Extend it to CPCV-style combinatorial paths when computational budget allows.
2. **Training (scale + safety):** the world model (RSSM imagination) as the primary RL environment, supplemented by (a) a JAX-LOB-style GPU LOB simulator for microstructure experiments and (b) generative synthetic series (existing `SyntheticBarSource` pattern, upgraded with regime-conditioned generation) for stress/coverage. Training on cheap simulated experience, evaluating on the honest backtester, promoting through the gate.
3. **Live shadow:** the demo broker (`MockBroker` on real closed bars) as a continuous out-of-sample monitor before any LIVE promotion.

**[ENGINEERING RECOMMENDATION]** The world model must never *replace* the honest backtester for promotion decisions — a learned simulator can silently drift (non-stationarity, Section 6.3) and would flatter policies that exploit its artifacts. This is the project's existing stance (evaluation.json + contract_hash promotion gate) extended to require backtester-verified candidate policies.

## 12.4 What the project environment still lacks (priority list)

**[ENGINEERING RECOMMENDATION]** Priority upgrades, in order of importance for production safety (not for research novelty):
1. **Real news/economic calendar feed** into `build_market_data.is_high_impact_event` (currently a placeholder) — event windows are the biggest XAUUSD tail risk.
2. **Partial-fill and liquidity impact model** for sizes above minimum-lot multiples (before scaling beyond 10% exposure).
3. **Weekend/market-hours handling** with gap risk (spread widening at open; overnight gaps in gold).
4. **Margin/leverage simulation** (margin-call logic) so the RL agent learns leverage limits rather than assuming infinite credit.
5. **Raw OHLCV into observations** (Section 6.2.1) — the environment feeds the *model*, so representation upgrades start here.
6. **CPCV-style evaluation paths** in `backtest/engine.py` for multi-regime validation.
