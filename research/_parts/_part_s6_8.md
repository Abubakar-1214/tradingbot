# Section 6 — DreamerV3, World Models, and Model-Based RL: Deep Dive

## 6.1 Why world models are the load-bearing decision

**[FACT]** Model-based RL learns a model of environment dynamics — p(next state, reward | state, action) — and uses it for planning, imagination, or value estimation, instead of (only) learning directly from environment interaction. (Definitional; the Dreamer line of work, arXiv 2301.04104, operationalizes this for continuous-control and game tasks.)

**[RESEARCH FINDING]** DreamerV3 (arXiv 2301.04104) demonstrates that a single fixed world-model configuration, with no per-task tuning, outperforms specialized methods across more than 150 tasks, including sparse-reward and long-horizon settings; the results were published in Nature in 2025. Its core components are: the RSSM (Recurrent State-Space Model) latent dynamics, symlog encoding/decoding of observations and rewards, a two-hot distributional critic, free-nats KL balancing, and an imagination-based actor-critic trained on dreamed rollouts. (Nature 2025 publication; arXiv 2301.04104.)

**[RESEARCH FINDING]** TransDreamer (arXiv 2209.14153) replaces the RSSM's recurrent backbone with an (optional) Transformer for sequence modeling, improving representation of long-range dependencies in complex environments — evidence that the *latent dynamics core* can be swapped for Transformer/SSM backbones (see also Mamba-based dynamics, Section 19).

**[RESEARCH FINDING]** 251-dreamer-trading (github suenot/251-dreamer-trading) applies DreamerV2/V3 to trading, motivated by the RL sample-inefficiency problem, the cost/risk of real-market exploration, and delayed reward feedback — precisely the reasons model-based RL is attractive for FX (delayed trade outcomes, costly exploration, sparse high-quality feedback).

**[FACT]** The project already implements a faithful DreamerV3 agent: `models/dreamer_agent.py` contains the RSSM, symlog/symexp, two-hot-free-nats KL, imagination-based actor-critic with return normalization, and a `ReplayBuffer` with episode-boundary-safe sampling; `models/dreamer_components.py` provides the network blocks; `models/mcts.py` adds PUCT planning over the world model. (Source: direct file reads.)

## 6.2 The DreamerV3 components explained and their role in a trading brain

### 6.2.1 RSSM (Recurrent State-Space Model)

**[FACT]** RSSM maintains a deterministic recurrent state h_t and a stochastic latent z_t; it learns p(z_t | h_t, z_{t-1}, a_{t-1}) (prior) and q(z_t | observation history) (posterior), trained by KL + reconstruction losses. The latent state is a compressed, action-conditioned representation of market + account dynamics. (arXiv 2301.04104; project `models/dreamer_agent.py`.)

**[ENGINEERING RECOMMENDATION]** The RSSM latent is the system's *market state representation* (Section 20 block REPRESENTATION→WORLD MODEL). In the project today, observations fed to the RSSM are precomputed indicator features + account state (`env/dreamer_trading_env.py` observations built from `core/feature_pipeline.py`). The recommendation is to *extend* the observation so the RSSM also sees raw OHLCV (tokenized/patched) — the world model should learn market dynamics from prices, not only from derived indicators. This is a targeted upgrade of `dreamer_agent.py`, not a new parallel system.

### 6.2.2 Symlog encoding/decoding

**[FACT]** Symlog maps scalar observations/rewards through sign(x)·log(1+|x|) before prediction, and symexp maps back, normalizing the large dynamic range of prices/returns. (arXiv 2301.04104; project uses symlog/symexp in `models/dreamer_agent.py`.)

**[ENGINEERING RECOMMENDATION]** Keep symlog for reward/return prediction, but note its role for XAUUSD: raw gold price levels are large (e.g., 2000–3000) while returns are small; the *representation* of prices should additionally be scale-invariant (z-scored, log-returns, or Kronos-style normalized OHLCVA) before entering the encoder. The feature pipeline already z-scores on train-only fit (`core/feature_pipeline.py`) — keep that discipline.

### 6.2.3 Two-hot critic

**[FACT]** The critic predicts the distribution of returns over symlog-scaled bins (two-hot categorical), which is more robust than a scalar MSE critic, especially for skewed return distributions. (arXiv 2301.04104; project implements it.)

**[ENGINEERING RECOMMENDATION]** The two-hot distributional critic is a *risk* asset: the full return distribution enables drawdown-aware value estimation (Section 11). The recommended risk-aware reward design (arXiv 2506.04358 composite reward) should be scored in the *imagination* — i.e., the critic should value risk-adjusted return sequences, not raw discounted returns. This is a change to the reward passed to the critic and the actor objective, not a new network.

### 6.2.4 Free-nats KL balancing

**[FACT]** Free-nats thresholds the KL term so the model is not forced to compress observations when information is irrelevant — stabilizes training and prevents posterior collapse. (arXiv 2301.04104; project free_nats=1.0, kl_dyn_scale=0.5, kl_rep_scale=0.1.)

**[ENGINEERING RECOMMENDATION]** Keep free-nats; for financial data (high noise, low signal) it prevents the world model from "memorizing" noise. The more important tuning is *what* the world model must reconstruct: with risk-aware rewards, the reward decoder matters more than exact price reconstruction. Partial reconstruction (predict only next OHLC summary + reward) is a viable experiment (Proposed Experiments).

### 6.2.5 Imagination and return normalization

**[FACT]** The actor and critic are trained on *dreamed* rollouts from the RSSM (imagination), with return normalization by percentiles. This is how DreamerV3 scales to diverse reward scales. (arXiv 2301.04104; project `models/dreamer_agent.py` uses horizon=15, max_imag_starts=8192, slow_critic_tau=0.02.)

**[ENGINEERING RECOMMENDATION]** The imagination horizon is a *holding-period* prior: horizon 15 at the decision cadence (candle-close) means the policy values ~15 candles of consequence. For XAUUSD with news spikes, the horizon should be configurable per regime (Section 9). Keep imagination as the training engine; do not replace it with PPO-on-replay alone.

## 6.3 Adapting world models to Forex/XAUUSD — specific considerations

**[RESEARCH FINDING]** The DRL-trading literature (survey of 167 papers; arXiv 2307.11685; S0927538X25002136) uniformly flags: overfitting to the simulated environment, variable transaction costs, and non-stationarity. A world model trained on historical XAUUSD *must* therefore be treated as an approximation of a non-stationary process, not a fixed law of motion.

**[ENGINEERING RECOMMENDATION]** Forex adaptation of the world model:
1. **Non-stationarity handling:** retrain/continue-train the world model on rolling windows (Section 7), with purged/embargoed validation (project `backtest/engine.py` walk_forward already uses purge+embargo — mirror this for world-model training).
2. **Costs in the dream:** the reward decoder and imagination must include spread/commission/slippage/swap (project `env/dreamer_trading_env.py` already models these — keep them in the loop when rolling out imagined trajectories; never imagine cost-free trades).
3. **Microstructure realism:** intraday XAUUSD has spread widening around events and illiquid overnight gaps; the world model should see spread/vol features (project `RiskSupervisor` vol/spread filters and `trade_executor.build_market_data` volatility z-score/spread fraction are the deterministic mirrors of this).
4. **Regime-conditioned dynamics:** an HMM/regime-conditioned world model family (RegimeRL's explicit regime model) is an *experiment*; start with the simpler evidence-based step — train separate world models per detected regime or condition the RSSM on regime id (Section 9).
5. **News/events:** the world model cannot predict news content; treat event windows as *epistemic risk states* — reduce position size and widen stops (project already halves size on high-impact events via `RiskSupervisor` event_position_scale=0.5). The world model should mark event windows as high-uncertainty in its latent, not attempt to predict them.

**[UNPROVEN/HYPOTHETICAL]** Whether a world model can learn exploitable short-horizon XAUUSD dynamics that survive costs is unproven — the project's own evidence (rule strategies do not beat buy-and-hold net of costs) and the efficiency literature (Section 16) argue for skepticism. The world model's primary *defensible* value is safe RL training, counterfactual what-if analysis, and risk simulation — not guaranteed alpha.

## 6.4 Comparison with other RL approaches

| Approach | Training signal | Exploration cost | Strengths | Weaknesses | Verdict for this project |
|---|---|---|---|---|---|
| **DQN / value-based model-free** | TD errors on environment transitions | Real/simulated only | Simple; discrete actions fit flat/long/short | Sample-inefficient; no dynamics model; known instability in continuous control | Not the brain; project already moved past it |
| **PPO / policy-gradient model-free** | Clipped surrogate on environment/simulated rollouts | Real/simulated | Robust, widely used (project PPO, `models/policy.py::PpoPolicy`; Transformer-PPO `models/transformer_policy.py`) | No dynamics model; no what-if; sample-hungry | Keep as a *policy optimizer* inside the world-model loop (actor-critic on dreamed rollouts is DreamerV3-style; PPO-style clipping is compatible) |
| **Offline RL (CQL, IQL)** | Conservative value learning on fixed datasets (arXiv 2006.04779; IQL arXiv 2110.06169) | None (batch) | Learns from historical data without live exploration; conservative against distribution shift | Needs good coverage in the dataset; value overestimation still a risk | **Adopt as the training-time safeguard**: pre-train/regularize the policy on historical data with conservative value estimates before any imagination-based fine-tuning |
| **Decision Transformer (DT) / Trajectory Transformer** | Sequence modeling conditioned on desired return (arXiv 2106.01345) | None (batch) | Elegant; return-conditioned behavior; easy to validate | Ignores dynamics unless paired with a model; needs return-to-go labels | **Experiment**: DT as an offline pretraining objective (imitation/behavior cloning on good trajectories) then fine-tune with world-model RL |
| **Model-based RL (DreamerV3)** | World-model loss + imagination actor-critic | Imagined (cheap) | Safe exploration; what-if; credit assignment; project already implements it | Dynamics drift in non-stationary markets; needs re-fitting | **Core brain** |

**[ENGINEERING RECOMMENDATION]** Single evidence-based decision: the core remains **DreamerV3-style model-based RL**, with (a) CQL-style conservative regularization during offline pretraining and (b) an optional DT-style return-conditioned pretraining experiment, both before live-safe imagination fine-tuning. This is one integrated stack — offline pretraining → world-model fine-tuning → safe live use — not five parallel agents.

---

# Section 7 — Self-Learning and Self-Improvement

## 7.1 What "self-improving" can and cannot mean

**[FACT]** A self-improving trading system must close the loop: every completed trade produces an outcome; the outcome must produce a learning signal; the learning signal must update the system; the update must be validated before it affects live capital. Each link can fail silently, and the project's honest stance ("ML path refuses to fabricate results") is the correct discipline for this loop.

**[FACT]** The project today has **no production self-improvement loop**: the replay buffer is training-only; `live/live_trade_mt5.py` acts on closed bars and reconciles, but nothing in the live path updates the policy from live outcomes. (Source: direct file reads.)

## 7.2 Learning from every completed trade: reward signal and credit assignment

**[RESEARCH FINDING]** Delayed rewards are the fundamental credit-assignment problem in trading: a trade's entry decision pays off (or not) only at exit, possibly many bars later; sparse terminal rewards make per-step learning hard. World-model imagination addresses this by rolling value targets backward through dreamed trajectories (arXiv 2301.04104); return-conditioned sequence models (DT, arXiv 2106.01345) sidestep it by conditioning on target returns.

**[RESEARCH FINDING]** Counterfactual credit assignment (arXiv 2607.16999, counterfactual Shapley-based credit assignment for RL) attributes a *joint* outcome to individual decisions by asking "what would have happened if this decision had differed" — a principled answer to "which of my last 10 decisions caused this loss?" This requires a model of the counterfactual world, which is exactly what the world model provides (Section 17).

**[ENGINEERING RECOMMENDATION]** Reward design (Section 11) is the primary credit-assignment instrument: use per-*trade* risk-adjusted outcomes (Sharpe-contribution, profit-factor, drawdown-penalized) attributed to the trade's decision sequence, plus per-step auxiliary rewards (spread cost paid, position maintained) where they are causally unambiguous. Then let the world-model critic distribute credit over imagined rollouts. The project's `position_sizing.dynamic_sizing` already converts critic advantages into win-probabilities — a useful heuristic, but it must become a *trained* quantity with a proper calibration check before being trusted for sizing (Section 16).

## 7.3 Update timing: when should parameters change?

**[ENGINEERING RECOMMENDATION]** Three update tiers with different cadences:
1. **Slow (model maintenance):** world-model retraining on rolling windows (weekly/monthly), fully offline, purged+embargoed validation (mirror `backtest/engine.py::walk_forward` train 800 / embargo 25 / test 300 discipline).
2. **Medium (policy improvement):** offline RL/DT fine-tuning on accumulated experience every N completed trades, with conservative regularization (CQL-style), validated on held-out recent data before promotion.
3. **Fast (within-day):** *no* parameter updates during live trading in the first production phase; only deterministic adaptation (risk state, sizing) is allowed intraday. Online SGD during live trading is an experiment, not a default (see 7.6).

**[FACT]** The project already has the *promotion gate* that fits this model: `live/live_trade_mt5.py::enforce_model_promotion_gate` requires a manifest + evaluation.json passed==true + contract_hash match before LIVE start, and `model_promotion_failures` reports failures. The recommendation is to make promotion *automatic but gated*: a candidate policy trained on new experience enters the gate, is evaluated purged/embargoed, and is promoted only if it beats the incumbent with statistical significance (Section 16: DSR/PSR logic).

## 7.4 Catastrophic forgetting and stability-plasticity

**[RESEARCH FINDING]** Continual learning surveys (arXiv 2403.05175) locate forgetting in the stability-plasticity dilemma and organize methods into three families: regularization (EWC, arXiv 1612.00796, slows updates on Fisher-important weights; SI), replay-based (rehearsing old data), and dynamic-architecture (growing/modular networks).

**[ENGINEERING RECOMMENDATION]** Adopt all three families *at their natural scales*:
1. **Regularization (EWC):** when fine-tuning the policy on recent experience, penalize changes to weights important to past regimes (Fisher-based). This is the cheapest forgetting defense and fits the project's single-agent design.
2. **Replay:** the project's `ReplayBuffer` already stores experience; *extend its lifetime* from training-only to a managed experience store with stratified sampling across regimes (never sample only recent trades — that overfits recent behavior, Section 7.5).
3. **Dynamic architecture (modular experts):** the regime-conditioned world model (Section 9) can train *per-regime experts* (MoE-style, Section 19) so that learning a new regime does not overwrite old ones. Start as an experiment.

## 7.5 Overfitting to recent behavior

**[RESEARCH FINDING]** Backtest overfitting — selecting strategies that fit noise — is quantified by Lopez de Prado's methods: Combinatorial Purged Cross-Validation (CPCV), Probability of Backtest Overfitting (PBO), Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) correct performance estimates for selection bias and non-normality (SSRN 2460551; "How To Backtest Correctly" repo). Walk-forward testing on a single path is dangerous — it tests only one sequence of events; purged k-fold tests multiple "regimes" (Lopez de Prado; quantstrategy.io blog).

**[FACT]** The project's `backtest/engine.py::walk_forward` uses purge+embargo (train 800 / embargo 25 / test 300) with *fixed* rule parameters — honest by construction (no in-sample optimization in this release). The README honestly reports that rule strategies do **not** beat buy-and-hold net of costs. (Source: direct file reads + README.)

**[ENGINEERING RECOMMENDATION]** Self-improvement is precisely the place where overfitting-to-recent-behavior becomes catastrophic: a policy that chases last month's regime will adapt to noise. Guardrails: (a) minimum sample size for any update (e.g., ≥50 completed trades and ≥2 regime windows), (b) DSR/PSR-gated promotion against the incumbent *and* against buy-and-hold (Section 16), (c) EWC penalty on old-regime weights, (d) a "frozen champion + challenger" protocol where the challenger must beat the champion on purged/embargoed data with a deflated significance threshold.

## 7.6 Safe continual / online learning in production

**[RESEARCH FINDING]** The general continual-learning literature and the DRL-trading reviews agree on the direction: batch/offline updates with validation gates are safer than fully online SGD; online learning without such gates compounds non-stationarity and feedback loops. (arXiv 2403.05175; DRL-trading survey findings on overfitting/non-stationarity.)

**[ENGINEERING RECOMMENDATION]** The production-safe self-improvement loop (the "SAFE UPDATE" block of Section 20):
1. **Collect:** every completed trade + full observation/trajectory logged (audit-friendly; the live stack already logs idempotency tokens and fills).
2. **Buffer:** store in a stratified experience DB (Section 8), balanced across regimes and recency.
3. **Train (offline):** world-model maintenance + policy fine-tuning with EWC + CQL regularization; DT-style return-conditioned pretraining as an experiment.
4. **Validate:** purged/embargoed walk-forward + DSR/PSR gate + cost-inclusive evaluation + out-of-regime stability checks.
5. **Promote:** only through the existing promotion gate (`enforce_model_promotion_gate`), extended with statistical-significance criteria.
6. **Rollback:** keep the incumbent checkpoint; if the promoted policy degrades on live-monitored metrics (equity, risk-state circuit breakers), auto-rollback. The deterministic `RiskSupervisor` (SQLite-persisted breakers) is the emergency backstop independent of the model.

**[FACT]** The project has most of the *skeleton* for this loop: promotion gate, risk-state persistence, reconciliation, kill switch, honest evaluation. What is missing is (a) the experience DB, (b) the offline training job, (c) the significance-gated promotion, and (d) automatic rollback. (Source: direct file reads.)

## 7.7 Is "self-improving" realistic?

**[UNPROVEN/HYPOTHETICAL]** Whether any ML-based XAUUSD strategy can sustainably improve *net of costs* beyond buy-and-hold is unproven — the project's own honest backtests and the efficiency literature (Section 16) cut against it. What *is* evidence-based is the loop's value for: risk management (learning drawdown patterns), execution quality (learning spread/slippage behavior), and *not degrading* (conservative offline updates that preserve past competence). The report therefore defines "self-improving" as **continuously better calibrated and no-worse-than-incumbent risk-aware decisions** — not "guaranteed alpha compounding". This honest framing keeps the ML path's no-fabrication discipline intact.

---

# Section 8 — Memory Taxonomy

## 8.1 The full taxonomy

| Memory kind | What it holds | Persistence | Cost to use | Trading role |
|---|---|---|---|---|
| **Model parameters** | Learned weights (policy, world model, critic) | Disk checkpoints (project `save/load/from_checkpoint`) | Retraining to change | The "skill" memory — slowest to change, most compressed |
| **Context window** | Recent tokens/observations in the transformer/RSSM (project `TransformerActor` seq_len=64; RSSM h/z state) | Ephemeral (forward pass) | Cheap | Short-horizon market context (last 64 candles / latent history) |
| **Recurrent state** | Deterministic recurrent state h_t (RSSM) | Ephemeral per episode | Cheap | Encodes dynamics history into the representation |
| **Latent state** | Stochastic latent z_t (RSSM posterior) | Ephemeral per step | Cheap | Compressed belief about current market/account state |
| **External memory** | Explicitly stored vectors/experiences (e.g., memory-augmented networks) | Persistent, structured | Retrieval cost | Regime prototypes, rare-event patterns, news-event memory |
| **Replay buffer** | Raw (state, action, reward, next) transitions | Persistent (training) — project `ReplayBuffer` | Sampling cost | Offline RL/DT training data |
| **Historical DB** | Full market + trade history (OHLCV, fills, states) | Persistent (SQLite/CSV in project) | Query cost | Audit, evaluation, retraining, backtesting |
| **Retrieval** | Retrieve relevant past experience/features for the current state (RAG-style) | Index over external memory/DB | Search cost | Regime-matched analog retrieval; news/sentiment context |
| **Long-term representations** | Summaries/embeddings of long-horizon patterns (FinMem layered memory) | Persistent | Write/merge cost | Macro/regime memory; slow-moving context that must not be re-learned per window |

**[FACT]** The project currently uses: model parameters (all `models/`), context window (Transformer seq_len=64), recurrent + latent state (RSSM h/z), replay buffer (training-only `ReplayBuffer`), and a historical DB for audit/backtest data. It has **no external memory, no retrieval, and no long-term representation layer** in the decision path. (Source: direct file reads.)

## 8.2 FinMem layered memory as the design pattern

**[RESEARCH FINDING]** FinMem (arXiv 2311.13743) designs an LLM trading agent with **layered memory** — working, consolidated, and long-term layers — plus character/profile design and self-evolving professional knowledge, and reports leading performance vs algorithmic agents in its paper's setups. FinRobot (github AI4Finance-Foundation/FinRobot) demonstrates multi-agent + memory compositions for finance.

**[ENGINEERING RECOMMENDATION]** Adapt FinMem's *layering* to the numerical core, not its LLM-only decision logic:
- **Working memory = context window + RSSM latent** (last ~64 candles + current belief) — already present.
- **Consolidated memory = experience DB** stratified by regime and recency (the offline-training dataset, Section 7.3) — to be built; replaces the training-only replay buffer.
- **Long-term memory = regime/macro representation store**: per-regime summaries, event outcomes, macro-feature states (DXY, yields — Section 13) — to be built; retrieved by the regime module (Section 9) and the world model as conditioning context.

## 8.3 RAG / retrieval for trading

**[RESEARCH FINDING]** RAG (retrieval-augmented generation) is the standard method for grounding LLM outputs in external documents; in finance, retrieval over news/filings/sentiment is well established in the FinGPT/FinMem ecosystem. (arXiv 2306.06031; arXiv 2311.13743.)

**[ENGINEERING RECOMMENDATION]** Use retrieval in exactly two places, both non-decision-critical:
1. **LLM research desk:** retrieval over news/calendar/sentiment to ground analyst summaries (Section 18) — advisory only, never order-generating.
2. **Regime-conditioned experience retrieval:** when the current latent state matches a stored regime prototype, retrieve that regime's statistics (win-rate, vol, event outcomes) to inform risk parameters (Section 9). This is cheap and auditable.

**[UNPROVEN/HYPOTHETICAL]** Whether *numeric* RAG (retrieving historical price-pattern analogues) improves trading decisions is unproven — nearest-neighbor price patterns are a known fragile heuristic. Keep it as an experiment only (Proposed Experiments), never a decision driver.

## 8.4 The recommended memory architecture for this system

**[ENGINEERING RECOMMENDATION]** Single evidence-based design:
- **Short-term:** context window + RSSM h/z (unchanged, `models/transformer_policy.py` seq_len=64 / `models/dreamer_agent.py` latent).
- **Mid-term:** stratified experience DB (new module — SQLite or columnar store with regime/recency indexes) feeding offline training.
- **Long-term:** regime + macro representation store (new module — embeddings/summaries per regime window, retrieved by the regime module and world-model conditioning).
- **Audit:** the existing historical DB/fills/idempotency logs stay untouched and authoritative.
All memories are *inputs to training or risk*, never order sources themselves. This keeps the decision authority in the RL core while giving it the context a genuinely long-lived system needs.

## 8.5 Memory implications for the project codebase

**[FACT]** Concretely, the gaps to close in `autonoumuse_trader`: (1) extend `ReplayBuffer` lifetime/logging into a persistent stratified experience DB; (2) add a regime/macro representation store (`models/meta_learning.py` regime labels are the natural seed — promote them from training-time labels to persistent, retrievable regime records); (3) add retrieval hooks for the LLM research desk; (4) keep `models/position_sizing.py` and `models/risk_supervisor.py` deterministic and external to the learned memory (safety must not depend on memory retrieval). (Source: direct file reads.)
