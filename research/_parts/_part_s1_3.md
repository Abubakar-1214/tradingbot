# Section 1 — The ML/DL/NN/RL/DRL/Time-Series/Foundation-Model/World-Model/Policy-Value Landscape for Trading

## 1.1 Why a vocabulary map matters

Before choosing an architecture, one must be precise about what the terms mean and how they relate. Trading AI discussions routinely conflate *supervised forecasting*, *decision-making*, *model-based simulation*, and *parameter adaptation*. Each has a different role, a different training signal, and a different failure mode. This section defines the families and gives a trading example for each; Section 2 then compares whole-system architectures built from them.

## 1.2 The family tree

### 1.2.1 Machine Learning (ML) and Deep Learning (DL) / Neural Networks (NN)

**[FACT]** ML is the broad field of algorithms that improve from data; DL is the subfamily using multi-layer neural networks trained by gradient-based optimization; NNs are the function class (composed layers) that DL trains. (Definitional; standard textbook usage.)

**[FACT]** In this project, the ML/DL layer is present in `models/` as: a PPO policy from Stable-Baselines3 (`models/policy.py::PpoPolicy`), a Transformer policy with positional encodings and GAE (`models/transformer_policy.py`), a DreamerV3 agent with RSSM/actor/critic networks (`models/dreamer_agent.py`, `models/dreamer_components.py`), ensembles (`models/ensemble.py`), MAML meta-learning (`models/meta_learning.py`), and adversarial training. (Source: direct file reads.)

**[ENGINEERING RECOMMENDATION]** The ML layer of the future system should remain what it is today — a set of trainable function approximators — but the *training signal* and *objectives* must be rethought (Sections 7, 11, 15). Neural networks are not the question; what they are trained to do is the question.

### 1.2.2 Reinforcement Learning (RL) and Deep RL (DRL)

**[FACT]** RL trains an agent to maximize cumulative reward through interaction with an environment: at each step the agent observes a state, takes an action, receives a reward, and transitions. DRL is RL where the policy/value/world-model functions are deep networks. (Definitional; standard reference: Sutton & Barto, *Reinforcement Learning: An Introduction*.)

**[RESEARCH FINDING]** A review of DRL trading systems (survey of 167 papers) reports persistent problems: overfitting, sensitivity to transaction costs, and non-stationarity of financial markets; it concludes DRL models must be carefully designed and validated to avoid overfitting and ensure robustness in real-world markets. (Verified via `internet_search`; consistent with "Algorithmic Trading and Reinforcement Learning: Robust methodologies for AI in finance", ResearchGate 356833146, and the DRL-trading survey.) **[RESEARCH FINDING]** RL applied to *execution* specifically (selling/buying a given amount with lowest cost) shows that many existing RL methods are not robust when assumptions are relaxed (arXiv 2307.11685), while properly benchmarked DRL can learn effective execution strategies (S0927538X25002136). The picture is balanced: RL works when the reward, costs, and environment are modeled honestly.

**[FACT]** In this project, RL is the primary ML paradigm: `RealisticTradingEnv` (`env/dreamer_trading_env.py`) is the environment; PPO, Transformer-PPO, DreamerV3, Dreamer+MCTS, and MAML wrap it. (Source: direct file reads.)

**[ENGINEERING RECOMMENDATION]** RL remains the right *decision* layer for trading (decisions under uncertainty are the core problem), but it must be combined with (a) a world model or simulator for safe exploration (Section 6), (b) risk-aware rewards (Section 11), and (c) offline/off-policy training so the agent does not need to lose money live in order to learn (Sections 7, 15).

### 1.2.3 Time-Series (TS) Modeling

**[FACT]** Time-series modeling predicts or represents sequences indexed by time. Classical tools (ARIMA, GARCH, HMM) model autocorrelation, volatility clustering, and regime structure; modern tools are neural sequence models. (Definitional.)

**[RESEARCH FINDING]** Time-series foundation models (TSFMs) — pretrained sequence models specialized for forecasting — have shown strong zero-shot generalization on diverse forecasting tasks. GIFT-Eval (Salesforce, arXiv 2410.10393) is a benchmark of 23 dataset groups across 7 domains and 10 frequencies designed to test exactly that zero-shot, universal-forecasting claim. Moirai 2.0 ranks among the top pretrained models on GIFT-Eval (arXiv 2511.11698).

**[FACT]** The project currently uses *hand-computed* time-series features (RSI, ATR, BB position, MACD diff, momentum, volume ratio, vol, higher-TF features) via `core/feature_pipeline.py`, rather than end-to-end learned representations from raw series. (Source: direct file reads.)

**[ENGINEERING RECOMMENDATION]** Time-series modeling belongs in the *representation + market-model* blocks (Section 20), not as the decision layer. The feature pipeline is a legitimate, auditable first representation, but the study recommends adding an end-to-end learned temporal encoder (e.g., patched-transformer or state-space encoder) so the market model learns from OHLCV directly — see Sections 5, 14.

### 1.2.4 Foundation Models (FM)

**[FACT]** A foundation model is a large model pretrained on broad data at scale and then adapted (fine-tuned, prompted, or used as a feature extractor) to many downstream tasks. (Definitional; see Bommasani et al., *On the Opportunities and Risks of Foundation Models*.)

**[RESEARCH FINDING]** For *language*, frontier FMs (ChatGPT/Gemini/Claude/Grok) are the substrate on which reasoning, tool use, and agent behavior are built via post-training (SFT/RLHF/GRPO/DPO) and orchestration — the model itself is not the whole system (arXiv 2407.16216; arXiv 2502.21321; arXiv 2402.02716). For *finance*, FinGPT (arXiv 2306.06031) and FinMem (arXiv 2311.13743) demonstrate that LLM FMs can be adapted for sentiment/decision tasks; FinRobot (github AI4Finance-Foundation/FinRobot) combines FMs with quant models and multi-agent workflows.

**[RESEARCH FINDING]** For *numerical markets*, Kronos (arXiv 2508.02739) is a true financial FM: pretrained autoregressively on 12B K-line records from 45 global exchanges; in zero-shot price forecasting it reports a +93% RankIC boost over the leading TSFM and +87% over the best non-pretrained baseline; it also reports a 9% lower MAE on volatility forecasting and 22% better synthetic K-line fidelity. (Parsed in full from arXiv 2508.02739 during this study.)

**[ENGINEERING RECOMMENDATION]** Foundation models enter the trading system in two distinct roles and must not be conflated: (a) a *financial/TS FM* as the market model / representation encoder (Kronos-style tokenizer + autoregressive Transformer), and (b) an *LLM FM* as a decision-support tool (news interpretation, research, plan drafting) that never directly places orders. Section 5 details both.

### 1.2.5 World Models

**[FACT]** A world model learns a compressed, predictive model of environment dynamics: given state and action it predicts the next state and reward, enabling *imagination* (planning in a learned simulator) instead of trial-and-error in the real environment. (Definitional; core idea from Ha & Schmidhuber, *World Models*, 2018; operationalized by Dreamer.)

**[RESEARCH FINDING]** DreamerV3 (arXiv 2301.04104; Nature 2025 publication) learns a world model and improves behavior by imagining futures; with a single fixed configuration it outperforms specialized methods across 150+ tasks. Its components — RSSM latent state, symlog scaling, two-hot critic, free-nats KL, imagination horizon — are the current standard for scalable model-based RL.

**[FACT]** The project already implements a faithful DreamerV3 agent (`models/dreamer_agent.py`): RSSM, symlog/symexp, two-hot-free-nats KL, imagination + actor-critic with return normalization, and a `ReplayBuffer` with episode-boundary-safe sampling. `models/mcts.py` adds PUCT planning over the world model. (Source: direct file reads.)

**[ENGINEERING RECOMMENDATION]** The world model is the single most strategically important component for *safe* autonomous trading, because it converts "exploration costs real money" into "exploration happens in imagination/simulation". The study's final architecture keeps and upgrades the DreamerV3-style world model (Section 20), but changes what it observes (raw OHLCV + account state rather than precomputed indicator features only) and how it is trained (Section 6).

### 1.2.6 Policy and Value Models

**[FACT]** In RL, a *policy* maps states to actions (or action distributions); a *value function* maps states (or state-action pairs) to expected return; both can be separate networks or share a backbone. (Definitional.)

**[FACT]** The project has both: actor networks (`Actor` in DreamerV3, `TransformerActor`, PPO policy) and critic/value networks (`Critic`, `TransformerCritic`); `models/position_sizing.py::KellyPositionSizer.dynamic_sizing` even converts critic advantages into win-probability estimates for Kelly sizing. (Source: direct file reads.)

**[ENGINEERING RECOMMENDATION]** Keep actor-critic separation (it enables GAE, imagination-based value targets, and risk-aware critics), but upgrade the critic's objective to value risk-adjusted returns (Section 11) and make the policy's output a *structured trading plan* rather than 3 raw actions (Section 10).

## 1.3 How the families relate in a trading system (the integration diagram in words)

**[FACT]** The families compose into a layered system: **Data → Representation (TS/DL encoders) → Market Model / World Model (TS-FM, RSSM) → Policy & Value (RL/DRL) → Decision & Risk (deterministic) → Execution (MT5) → Outcome → Reward → Memory → Training → Validation → Safe Update.** (This is the structure recommended in Section 20 and it mirrors the project's existing layering: `core/feature_pipeline.py` → `env/dreamer_trading_env.py` observations → `models/` policies → `models/risk_supervisor.py` → `live/trade_executor.py` → MT5.)

**[ENGINEERING RECOMMENDATION]** Every future component change should be evaluated against this single question: *which block does it belong to, and does it improve that block's training signal without corrupting another block's?* This discipline is what separates an integrated system from five parallel systems (the anti-pattern this study explicitly rejects).

## 1.4 Trading examples per family

| Family | Trading example | Project component today | Role in final architecture |
|---|---|---|---|
| ML/DL/NN | Feature-based classifiers, encoders | `core/feature_pipeline.py` features; NN policies | Representation + market model |
| RL/DRL | PPO/Dreamer trading agents | `models/dreamer_agent.py`, `models/transformer_policy.py`, PPO | Policy + value (decision) |
| Time-series | ARIMA/GARCH/HMM forecasting, TS-FMs | Feature windows only; no forecasting head | Market model input; optional forecast auxiliary |
| Foundation models | FinGPT sentiment, Kronos K-line FM | None (no FM in the repo) | Market model encoder + decision-support LLM |
| World models | DreamerV3 imagination of future equity paths | `models/dreamer_agent.py` RSSM (faithful) | Core: simulation-based planning and safe RL |
| Policy/Value | Actor-critic, GAE, critic-based sizing | `models/dreamer_agent.py`, `models/policy.py`, Kelly sizer | Decision + risk-aware critic |

**[ENGINEERING RECOMMENDATION]** The system should be thought of as a **world-model-centered RL system with a learned representation layer and a deterministic safety shell** — not as a "forecast-the-price ML model" and not as a "pure RL policy". Section 2 justifies this at the architecture level; Sections 3–6 justify it at the model level.

---

# Section 2 — System Architectures Compared

## 2.1 The design space

This section compares whole-system architectures along information flow, training, and joint-training feasibility. The candidates:

1. **Independent models / ensemble** — several models each predict or decide independently; outputs combined by voting/averaging.
2. **Sequential pipeline** — rigid chain: forecast → signal → position sizing → risk.
3. **Shared-backbone multi-task** — one encoder feeding several heads (forecast, direction, volatility, value).
4. **Unified single model** — one model consumes everything and outputs the decision (end-to-end).
5. **Hierarchical agent** — multiple specialized agents (analyst, trader, risk manager) with structured communication.
6. **World-model + policy** — learned dynamics model + RL policy planning through imagination.
7. **Hybrid** — combinations of the above with deterministic layers.

## 2.2 Independent models / ensembles

**[FACT]** The project already implements ensembles: `models/ensemble.py` supports soft/hard voting, weighted members, consensus thresholds (`min_agreement=0.6` default, `consensus_threshold=3`), entropy-based uncertainty, and epistemic disagreement (KL) estimates. `core/config.py::TradingBehaviorConfig` sets `min_ensemble_agreement=0.6`. (Source: direct file reads.)

**[RESEARCH FINDING]** Ensembles reduce variance and improve robustness in supervised ML generally, and in RL-trading contexts disagreement-based uncertainty (epistemic) is a useful risk signal; however, an ensemble of individually-overfit models can be confidently wrong together (a shared-bias problem), and ensembling does not fix a reward that optimizes the wrong thing. (Synthesis of the DRL-trading skepticism literature, e.g., ResearchGate 356833146, and standard ensemble theory.)

**[ENGINEERING RECOMMENDATION]** Keep the ensemble as an *uncertainty + safety mechanism* (block trades when members disagree, expose epistemic uncertainty to the risk layer), but do **not** make the ensemble the whole architecture. Independent policies without a shared market representation duplicate learning and multiply training cost. The final system uses an ensemble of *critics/policies on a shared representation* for calibrated uncertainty, not a bag of independent pipelines.

## 2.3 Sequential pipeline (forecast → signal → size → risk)

**[FACT]** The project's current rule path is a sequential pipeline: causal features → rule signal (`RuleSignalSource` in `live/live_trade_mt5.py`) → `DecisionEngine.filter` → ATR sizing → RiskSupervisor → executor. (Source: direct file reads.)

**[RESEARCH FINDING]** Sequential pipelines that pass hard decisions from one stage to the next accumulate errors: a forecast error propagates into signal, size, and risk decisions with no feedback to correct upstream stages. In trading specifically, forecasting-first systems are vulnerable because *forecast accuracy does not imply trading profitability* after costs and adverse selection (Section 16; Lopez de Prado's critique of raw-accuracy evaluation, DSR/PSR, SSRN 2460551).

**[ENGINEERING RECOMMENDATION]** The sequential pipeline is the *safety skeleton* (deterministic risk must stay an outer layer), but the *learning* core must not be a rigid chain. Recommendation: end-to-end trainable decision path with the deterministic risk layer as a final gate, and an explicit feedback loop from outcomes to the decision model (Sections 7, 20).

## 2.4 Shared-backbone multi-task

**[RESEARCH FINDING]** Multi-task learning (shared encoder, multiple heads) is a standard DL pattern; in time series, PatchTST demonstrates that one patched-transformer backbone with self-supervised pretraining transfers to multiple forecasting tasks (arXiv 2211.14730), and iTransformer shows a single inverted-attention backbone handling multivariate series well (arXiv 2310.06625).

**[RESEARCH FINDING]** Sharing a backbone across tasks is efficient and can regularize via task diversity; the failure mode is *task interference* when heads need conflicting representations. For trading, a shared market-representation encoder (price/microstructure/macro) with heads for value, risk, and (optionally) auxiliary forecasts is a well-supported design. (Synthesis of multi-task learning literature; supported by PatchTST/iTransformer evidence.)

**[ENGINEERING RECOMMENDATION]** The final architecture's **representation block is a shared backbone** (Section 20): one encoder learns the market state; the value/critic, the policy, and the (optional, non-blocking) forecast auxiliary all consume its output. This directly mirrors how DreamerV3 shares the RSSM latent state across reward/value/policy heads. The project's existing RSSM already implements this sharing — the recommendation is to keep it and add auxiliary heads deliberately.

## 2.5 Unified single model (end-to-end)

**[FACT]** End-to-end training (observations → actions through one differentiable stack) optimizes exactly the decision objective and removes intermediate labels. It is the default in many DRL trading papers (PPO/DQN variants trained end-to-end on the trading env). (Definitional + DRL-trading literature.)

**[RESEARCH FINDING]** The DRL-trading review literature repeatedly warns that end-to-end RL policies trained on historical environments overfit the simulator: they exploit artifacts, ignore costs unless the reward explicitly encodes them, and generalize poorly across regimes (survey of 167 papers; arXiv 2307.11685 for execution-specific caveats).

**[ENGINEERING RECOMMENDATION]** Reject a fully unified single model as the production answer. The reasons: (a) safety cannot be gradient-optimized — circuit breakers must be deterministic and auditable (the project's `RiskSupervisor` is exactly this and should stay non-neural); (b) interpretability and validation of a pure black-box is insufficient for live capital (Section 16); (c) the honest project evidence shows rule policies don't beat buy-and-hold net of costs — a single black box trained on the same data is unlikely to do better without structural priors. End-to-end training is used *within* the decision stack (representation→value→policy jointly), but the outer shell remains deterministic.

## 2.6 Hierarchical agent

**[RESEARCH FINDING]** Multi-agent LLM trading frameworks exist and report gains: TradingAgents (arXiv 2412.20138) simulates a trading firm — fundamental/sentiment/technical analysts, bull/bear researchers, traders with different risk tolerances, and a risk-management team — with structured debates, and reports improvements in cumulative and risk-adjusted return on its test setups. FinRobot (github AI4Finance-Foundation/FinRobot) is an open-source agentic platform combining foundation models, financial tools, quant models, and multi-agent workflows.

**[ENGINEERING RECOMMENDATION]** Use hierarchy *sparingly and correctly*: an LLM "research desk" (news/sentiment/macro interpretation) reporting *into* a numerical decision core is useful; a hierarchy of LLM agents *competing to place orders* is a research experiment, not a production architecture. The project should not adopt TradingAgents-style agent debates for order generation. The numerical core (world model + policy + risk) remains the decision authority; LLM agents are advisors with strictly limited authority (Section 18). **[UNPROVEN/HYPOTHETICAL]** Whether LLM-agent hierarchies produce sustained net-of-cost alpha in live FX is unproven; the TradingAgents results are on limited backtest-style setups and have not been independently replicated at scale.

## 2.7 World-model + policy

**[RESEARCH FINDING]** DreamerV3-style world-model RL (arXiv 2301.04104; Nature 2025) is the strongest evidence-based argument for model-based RL as a general decision engine: one configuration, 150+ tasks, imagination-based learning. The 251-dreamer-trading project (github suenot/251-dreamer-trading) applies DreamerV2/V3 to trading with explicit motivation: RL sample inefficiency, the cost/risk of real-market exploration, and delayed feedback loops.

**[FACT]** The project already has a faithful DreamerV3 implementation (`models/dreamer_agent.py`) with imagination, and `models/mcts.py` adds PUCT planning. This is the closest of all candidate architectures to the project's current state.

**[ENGINEERING RECOMMENDATION]** **The world-model + policy architecture is the recommended core.** It gives: (a) safe exploration through imagination (critical when exploration = real P&L); (b) a learned dynamics model that can be used for planning, counterfactual "what-if" simulation (Section 17), and risk scenario analysis; (c) credit assignment through imagined rollouts (addressing delayed feedback, Section 7); (d) a natural place for the representation layer. The recommendation is not "replace DreamerV3" but "complete it": raw-OHLCV observations, risk-aware reward, structured actions, external memory, and safe continual update (Sections 6, 8, 10, 11, 15).

## 2.8 Hybrid — the recommended shape

**[FACT]** "Hybrid" is a dirty word unless pinned down. This study's hybrid is precisely: **learned representation (shared backbone) + learned world model (RSSM/Transformer dynamics) + learned policy & risk-aware critic (RL) + deterministic risk shell + optional LLM research desk (advisory) + external memory + safe-continual training loop.** Every block has a single role, a single training mode, and defined neighbors (Section 20).

**[ENGINEERING RECOMMENDATION]** Adopt this single hybrid architecture. Reject: (a) five parallel systems (independent pipelines duplicated per idea); (b) pure end-to-end black box (no safety/validation); (c) pure LLM-agent trading (no numerical market model); (d) pure forecasting pipeline (no decision feedback). This is the "one evidence-based answer per decision" the study commits to.

## 2.9 Correcting wrong assumptions

1. **Wrong:** "A more accurate forecast model makes a better trader." **[RESEARCH FINDING]** Forecast accuracy and net trading performance decouple once costs, adverse selection, and position sizing enter (Lopez de Prado DSR/PSR methodology, SSRN 2460551; GIFT-Eval measures forecasting, not trading). Sections 3 and 16 expand this.
2. **Wrong:** "RL needs the real market to explore." **[RESEARCH FINDING]** World models and offline RL (CQL, arXiv 2006.04779; Decision Transformer, arXiv 2106.01345) learn from simulation and historical data respectively, avoiding costly real exploration. DreamerV3 is evidence that imagination-based learning scales.
3. **Wrong:** "A bigger LLM, prompted cleverly, can trade." **[UNPROVEN/HYPOTHETICAL]** No evidence shows LLM-only trading produces sustained net-of-cost alpha; FMs lack calibrated uncertainty over prices and are prone to hallucination in high-stakes decisions. LLMs are advisors, not traders (Section 18).
4. **Wrong:** "Our backtest said 800% so the strategy works." **[FACT]** The project's own honest backtest retracted such claims; the legacy fake engine produced exactly this illusion and was archived. Evaluation methodology is the whole ballgame (Section 16).

---

# Section 3 — Prediction vs. Decision-Making

## 3.1 The core question

Does an autonomous trading AI need to explicitly predict the next candle (price/return forecast), or can it learn Market State → Action → Reward directly? This is the fork that determines whether the system is "forecasting-first" or "decision-first" (or a hybrid), and it drives the entire model-selection story of Sections 4–5.

## 3.2 The three candidate framings

1. **Forecasting-first:** predict future returns/price (point or distributional); convert forecast to position (e.g., sign of expected return scaled by confidence). Requires a labeled supervised objective (next-step return) and a separate decision rule.
2. **Direct policy (decision-first):** learn π(action | state) directly from rewards (RL) or from demonstrations (imitation/offline RL). No explicit forecast is produced or required internally; the policy implicitly values future outcomes through the value function.
3. **World-model hybrid:** learn p(next latent state, reward | state, action) — a *conditional* model of dynamics — and derive decisions via planning/imagination. The model predicts the *consequences of actions*, not the price path unconditionally.

## 3.3 Can the agent learn Market State → Action → Reward without predicting the next candle?

**[FACT]** Yes, formally: an RL agent with a value function never needs an explicit next-price forecast. The Q/actor-critic objective only requires (state, action, reward, next state) tuples and a reward definition; the value function summarizes expected future return. (Standard RL theory, Sutton & Barto.)

**[RESEARCH FINDING]** Decision Transformer (arXiv 2106.01345) goes further: it frames RL as sequence modeling conditioned on *desired return*, states, and actions — no dynamic-programming bootstrap, and no explicit forecast of the environment; it learns to produce actions that achieve target returns. Offline RL methods (CQL, arXiv 2006.04779) learn conservative policies directly from historical (state, action, reward, next state) data without any next-price model. RegimeRL (github sahilapage) explicitly positions risk-aware RL trading with **no price prediction**, instead focusing on position sizing and trade timing under an explicit market regime model.

**[ENGINEERING RECOMMENDATION]** The answer to the heading question is **yes** — the decision core should not require an explicit next-candle forecast. The project already embodies this: all its models are direct policies (flat/long/short) and no forecasting component exists. This is not a weakness to "fix" by bolting on a forecaster; it is the correct decision-first foundation. **[FACT]** (Project status: no explicit next-candle forecasting component anywhere; all models are direct policy — verified in project inspection.)

## 3.4 Forecasting-first: where it helps, where it hurts

**[RESEARCH FINDING]** Where explicit forecasting *does* help: (a) volatility forecasting — Kronos reports 9% lower MAE than baselines on volatility forecasting, and volatility is directly actionable for sizing/risk (arXiv 2508.02739); (b) regime/state forecasting for risk overlays; (c) synthetic-data generation for training the simulator (Kronos reports 22% better synthetic K-line fidelity); (d) interpretability/validation — a forecast head gives a human-auditable check on the market model (GIFT-Eval methodology exists precisely to benchmark such models, arXiv 2410.10393).

**[RESEARCH FINDING]** Where forecasting-first hurts: forecast accuracy is decoupled from net profitability after costs (Lopez de Prado; SSRN 2460551), and end-to-end direct policies can learn cost-aware behavior that a forecast+rule conversion cannot (the DRL-trading literature's cost-sensitivity findings). A forecast that is "right" in MAE terms can still produce losing trades if the forecast error distribution aligns with adverse selection.

**[ENGINEERING RECOMMENDATION]** Use forecasts as **auxiliary supervision and risk inputs, never as the decision path**:
- An auxiliary *volatility* head on the market model (trained with a proper probabilistic loss) feeding the risk/position-sizing block (project: `RiskSupervisor` vol filter + `ATRPositionSizer` — upgrade these to model-based vol forecasts).
- An auxiliary *return-distribution* head used for regime detection and what-if simulation, gated so it cannot override the policy (Section 20).
- The policy remains the decision authority.

## 3.5 The world-model framing: conditional prediction as the missing bridge

**[FACT]** A world model does not predict the price unconditionally; it predicts p(next latent state, reward | state, action). That is *conditional* prediction — "if I take action a here, what happens next?" — which is strictly more useful for decisions than unconditional next-candle forecasting. (Definitional; core of DreamerV3, arXiv 2301.04104.)

**[RESEARCH FINDING]** Action-conditioned world models underpin the strongest generalist RL results (DreamerV3: one config, 150+ tasks, Nature 2025). In trading, a market world model lets the agent answer counterfactuals ("what would equity have done if I had been flat vs long vs short?") — the foundation of credit assignment and what-if risk analysis (Section 17; counterfactual credit assignment literature).

**[ENGINEERING RECOMMENDATION]** The recommended architecture is therefore the **world-model hybrid**: conditional dynamics model (RSSM/Transformer state-space) + policy/value trained by imagination + auxiliary forecast heads for risk inputs + deterministic risk shell. This is decision-first with a forecasting *sidecar* that is action-conditioned and risk-directed.

## 3.6 When explicit forecasting is useful (summary table)

| Use case | Explicit forecast? | Why / evidence |
|---|---|---|
| Decision (long/short/flat) | No — policy | RL learns decisions without price forecasts (DT, CQL; project's direct-policy stack) |
| Position sizing / stop distance | Yes — volatility | Kronos vol forecast −9% MAE; ATR is the crude version; vol is directly actionable |
| Regime detection | Yes — state/regime features | HMM regime literature; RegimeRL's explicit regime model |
| Risk overlay / circuit breakers | Yes — spread, vol, event | Deterministic; project `RiskSupervisor` already consumes vol/spread/event |
| Training the simulator | Yes — generative | Kronos synthetic K-line fidelity +22%; TRADES LOB generation (arXiv 2502.07071) |
| Interpretability / audit | Yes — auxiliary head | GIFT-Eval-style benchmarking; human review of market-model sanity |
| What-if / counterfactual | Yes — action-conditioned | World-model rollouts; counterfactual credit assignment (Section 17) |

**[ENGINEERING RECOMMENDATION]** Summary of Section 3: **decision-first with an action-conditioned world model, plus volatility/regime forecast sidecars feeding risk; no unconditional price-forecast head in the decision path.** The project's current state (direct policies, no forecast component) is the right skeleton; the upgrades are the world-model observation space, reward, memory, and safe-continual loop — not a forecasting bolt-on.
