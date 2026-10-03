# Designing a Genuinely Advanced, Autonomous, Self-Improving Forex / XAUUSD Trading AI

## A Comprehensive Technical Research Study — Evidence-Grounded Architecture Decisions for the `autonoumuse_trader` Project

**Document type:** Research study / technical documentation deliverable (NOT code).
**Target system:** Autonomous, self-improving algorithmic trading AI for XAUUSD (spot gold vs. USD) on the MetaTrader 5 (MT5) platform.
**Project under study:** `e:\Desktop\NeoMind\Bazz\autonoumuse_trader`
**Status of this document:** Research findings + engineering recommendation; a design study, not an implementation plan in code.

---

## Preface (Roman-Urdu / Hindi framing)

Yeh study is sawal ka jawab dene ke liye likhi gayi hai: **"Ek genuinely advanced, autonomous, self-improving Forex/XAUUSD trading AI ko kaise design karna chahiye?"**

Iska matlab yeh nahi ke hum koi aur "100% winning strategy" dhond rahe hain — woh exist nahi karti, aur jo log aise daawe karte hain unka data fake ya hindsight-biased hota hai. Iska matlab yeh hai ke hum science, engineering, aur honesty ke sath system design karein: kaunse components, kaunsi architectures, kaunse training paradigms, kaunsi memory, kaunsa reward, kaunsi evaluation — aur sab se pehle, **kya evidence hai** ke har design choice kaam karti hai.

Project ka apna history is honesty ki misaal hai: legacy backtester fake tha (artificial `np.random.randn(100)` returns), use archive mein daal diya gaya; naye backtest engine ne real costs ke sath dikhaya ke rule-based strategies **buy-and-hold ko net-of-costs beat nahi karti**; ML path ne results fabricate karne se inkaar kar diya. Yeh study wohi approach follow karti hai: har claim par **exactly ek marker** (FACT / RESEARCH FINDING / ENGINEERING RECOMMENDATION / UNPROVEN-HYPOTHETICAL), har recommendation ka evidence-based hona, aur jahan evidence kam hai wahan khul kar kehna **"yeh abhi unproven hai"**.

Agar aap Urdu/Hindi mein sochte hain to asaan tarika yeh hai: **har faisla is sawal se guzarta hai — "kya is ke peeche paper ya reproducible experiment hai, ya sirf mera khayal?"** Agar sirf khayal hai, to document mein usay HYPOTHETICAL likha jayega, aur agar paper hai to usay RESEARCH FINDING likha jayega.

---

## 1. How to Read This Document: The Claim-Marker Legend

Every substantive claim in this document carries **exactly one** of the following markers, placed immediately after (or at the start of) the claim. The marker tells you the epistemic status of the claim — what it is based on, and how much you should trust it.

| Marker | Meaning | Basis |
|---|---|---|
| **[FACT]** | Established, verified fact with a citation or direct project-file evidence. | Reproducible experiment, official documentation, source code read in this study, or a peer-reviewed/published primary source with an identifier (arXiv ID, URL, DOI) verified via `internet_search` during this study. |
| **[RESEARCH FINDING]** | What a specific paper, repository, benchmark, or documented experiment found. The finding may be contested, limited, or domain-specific — the marker means "this is what the cited work reports," not "this is universally true." | A specific cited paper/repo/benchmark. |
| **[ENGINEERING RECOMMENDATION]** | This study's own recommendation for the `autonoumuse_trader` codebase, grounded in the FACTS and RESEARCH FINDINGS above it. It is an engineering judgment — defensible, but a judgment, not a theorem. | Derived from the evidence base + project inspection. |
| **[UNPROVEN/HYPOTHETICAL]** | Speculative. There is not yet enough evidence to call it a fact or a research finding. It is flagged so nobody mistakes an idea for a result. | Insufficient evidence; marked precisely because the study refuses to fabricate support. |

**Rules of use.** (a) A claim is marked with exactly one marker — never zero, never two. (b) Where the study needed a number, an arXiv ID, a model card fact, or a benchmark result, that identifier was verified by `internet_search`/`parse_urls` during this study or was already present in the verified research corpus; nothing was invented. (c) Where the provided research was insufficient, the claim is either marked **[UNPROVEN/HYPOTHETICAL]** or omitted.

---

## 2. Scope and Methodology

### 2.1 Scope

This document answers the following design questions for a single, coherent system — not five parallel systems:

1. What is the *relationship* between the ML/DL/NN/RL/DRL/time-series/foundation-model/world-model/policy-value vocabulary, and how do those concepts map onto trading? (Section 1)
2. Which *system architecture* should a production trading AI use? (Section 2)
3. Does the system need explicit next-candle *prediction*, or can it learn State→Action→Reward directly? (Section 3)
4. Which *"brain"* model family is the right core? (Sections 4–5)
5. Are *world models* (DreamerV3-style) the right base, and how do they adapt to Forex? (Section 6)
6. How does a system *self-improve safely*? (Sections 7–8)
7. Regime handling, action space, reward, and the trading environment. (Sections 9–12)
8. Data, multi-timeframe structure, and training strategy. (Sections 13–15)
9. Evaluation that does not lie. (Section 16)
10. Causality, credit assignment, and what frontier AI actually teaches us. (Sections 17–18)
11. New architectures and their realistic role. (Section 19)
12. A final block-diagram architecture and 14 explicit decisions. (Sections 20–21)

### 2.2 Methodology

1. **Project inspection.** The `autonoumuse_trader` codebase was read directly: `README.md`, `core/feature_pipeline.py`, `core/config.py`, `core/observation.py`, `env/dreamer_trading_env.py`, `models/dreamer_agent.py`, `models/risk_supervisor.py`, `models/meta_learning.py`, `models/ensemble.py`, `models/transformer_policy.py`, `models/policy.py`, `models/position_sizing.py`, `backtest/engine.py`, `backtest/costs.py`, `live/live_trade_mt5.py`, `live/trade_executor.py`, plus the research/audit artifacts. Every section of this document references concrete project modules — what exists, what is weak, what is missing.
2. **External research corpus.** The evidence base was assembled through targeted `internet_search` / `parse_urls` verification of: foundation-model papers and model cards (Kronos, TimesFM 3.0, TimeGPT, Chronos/Chronos-2, Moirai/Moirai-2, PatchTST, iTransformer, TFT, FinGPT/FinMem/FinRobot), world-model and RL papers (DreamerV3 + Nature 2025, TransDreamer, Decision Transformer, CQL, offline RL, RL-for-trading surveys), evaluation methodology (Lopez de Prado purged CV/PSR/DSR/CPCV, GIFT-Eval), realistic simulation (multi-agent LOB RL, TRADES, JAX-LOB), regime detection (HMM, RegimeRL), continual learning (EWC, surveys), multi-scale time-series architectures (HiMTM, dilated conv), Mamba/SSM, MoE, LoRA, GRPO/DPO post-training, LLM-agent planning surveys, market-efficiency and RL-trading-skepticism literature.
3. **Honesty constraint.** Where the project's own results are negative, the document says so plainly and explains what that implies for design (see §2.3).
4. **Single-recommendation rule.** Where multiple options are compared, the document reaches **one** evidence-based recommendation per decision. It never recommends building five parallel systems.

### 2.3 The Project's Honest Baseline (must be internalized before reading further)

These are the verified facts about the current `autonoumuse_trader` system that shape every recommendation in this study:

- **[FACT]** The legacy backtester (`backtest_engine.py`) fabricated `np.random.randn(100)` observations, had commented-out training, and produced placeholder P&L; it was archived as `archive/backtest_engine_legacy_fake.py` and replaced by a real engine on `kernc/backtesting.py` 0.6.2 with real data, explicit costs, deterministic seeds, and purge+embargo walk-forward (`backtest/engine.py`, `backtest/costs.py`). (Source: README.md + direct file reads.)
- **[FACT]** The honest backtest verdict (seed 42, reported in `research/backtest_results` via `backtest/report_backtest.md`): no rule strategy beat buy-and-hold net of costs. On `xauusd_d1`: SmaCrossAtr 52.40% vs B&H 884.26% (Sharpe 0.39, MaxDD −12.46%, 58 trades); RsiReversion 29.07% vs 932.41% (Sharpe 0.16, MaxDD −19.84%, 162 trades); DonchianBreakout 278.94% vs 923.11% (Sharpe 0.56, MaxDD −28.69%, 98 trades). On `xauusd_h1_from_m1`: SmaCrossAtr 8.13% vs 137.58% (Sharpe 0.40, MaxDD −7.00%, 214 trades); RsiReversion −3.45% vs 137.07% (Sharpe −0.13, MaxDD −13.67%, 688 trades); DonchianBreakout 44.90% vs 138.01% (Sharpe 0.86, MaxDD −17.65%, 295 trades). The earlier inflated multi-hundred-percent / Sharpe>3 targets are retracted. (Source: README.md + backtest reports.)
- **[FACT]** No trained model checkpoint has been shipped. The ML path (`SIGNAL_SOURCE=model|ppo`) requires a promoted manifest with a matching evaluation record and never silently falls back to flat; live mode refuses to start unless all gates pass (`live/live_trade_mt5.py`, `core/config.py`). (Source: direct file reads.)
- **[FACT]** The ML path refuses to fabricate results: `artifacts/models/ppo_gold_v1/` contains a `manifest.json`, `feature_contract.json`, `evaluation.json` and `model.zip` artifact structure, and the live loop reports real realized P&L from broker fill history rather than simulated numbers. (Source: file listing + live loop source.)
- **[FACT]** The project has 139 passing pytest tests (`python -m pytest tests env -q`), plus verification gates `verify_config.py` (24), `verify_risk.py` (29), `verify_broker.py` (29), `verify_risk_integration.py` (23). **README internal inconsistency, reported verbatim:** the Quick Start section says "87 passed" while the Verified Facts table says "139 passed". This study reports both quotes and flags the discrepancy rather than silently picking one. (Source: README.md.)
- **[FACT]** Known gaps/weaknesses identified during inspection: (a) no explicit next-candle forecasting component — all models are direct policies over flat/long/short; (b) no live self-improvement loop — the replay buffer is training-only; (c) reward is scaled log-return only, with no Sharpe/Sortino/drawdown-aware multi-objective; (d) the DreamerV3 world model (RSSM) observes precomputed indicator features, not raw OHLCV; (e) memory = replay buffer + latent state only, no external memory/retrieval/experience DB; (f) regime handling only via MAML task segmentation with explicit labels, not learned end-to-end; (g) action space = 3 discrete actions (flat/long/short), with SL/TP/sizing handled by deterministic layers (`models/position_sizing.py`, `live/trade_executor.py`); (h) no cross-asset data (e.g., DXY) actually in the features despite the correlation-guard config existing. (Source: direct file reads.)

---

## 3. Section Index

| # | Section | Core question |
|---|---|---|
| 1 | The ML/DL Landscape for Trading | How do the model families relate, and where does each fit in a trading system? |
| 2 | System Architectures Compared | Independent/ensemble, sequential pipeline, shared-backbone, unified, hierarchical, world-model+policy, hybrid — which one? |
| 3 | Prediction vs. Decision-Making | Is next-candle prediction necessary, or can we learn State→Action→Reward directly? |
| 4 | The "Trading Brain": Advanced Architectures | Transformer, financial FM, TS-FM, world model, RL/DRL, hybrid, shared backbone, newer architectures. |
| 5 | Financial & Time-Series Foundation Models in Depth | Per-model evidence tables: Kronos, TimesFM, TimeGPT, Chronos/Chronos-2, Moirai/Moirai-2, PatchTST, iTransformer, TFT, FinGPT/FinMem/FinRobot. |
| 6 | DreamerV3, World Models, and Model-Based RL | RSSM, symlog, two-hot critic, imagination, free nats; Nature 2025; TransDreamer; 251-dreamer-trading; adaptation to Forex; vs DQN/PPO/offline RL/Decision Transformer. |
| 7 | Self-Learning and Self-Improvement | Learning from every completed trade; reward signal; credit assignment; update timing; catastrophic forgetting (EWC/replay/modular); safe continual/online learning; the production-safe loop. |
| 8 | Memory Taxonomy | Parameters vs context vs recurrent/latent state vs external memory vs replay vs historical DB vs retrieval vs long-term representations; FinMem layered memory; the recommendation. |
| 9 | Market Regime Detection | Separate model vs shared representation vs world-model vs implicit vs ensemble; HMM evidence; the project's MAML path; risk overlay. |
| 10 | The Action Space | BUY/SELL/HOLD vs full trading plan (direction, entry, size, SL/TP, holding period, exit, confidence, risk); joint vs hierarchical learning; QuantAgent/TradingAgents evidence. |
| 11 | Reward Function Design | Raw P&L vs risk-adjusted vs multi-objective; arXiv 2506.04358; TorchTrade; Pro-Trader-RL; RegimeRL; the right design for XAUUSD. |
| 12 | The Realistic Trading Environment | Bid/ask, spread, slippage, commission, latency, liquidity, partial fills, margin, leverage, position state, equity, drawdown, trading hours, news; backtester vs market simulator/world model; multi-agent LOB RL, JAX-LOB; mapping to `RealisticTradingEnv`. |
| 13 | Data: What Enters the Model | OHLCV, tick, bid/ask, spread, volume, order book, news, calendar, macro, sentiment, cross-asset (DXY, yields, indices, commodities, correlations); resolution; mapping to the feature pipeline. |
| 14 | Multi-Timeframe / Multi-Resolution Modeling | tick/1m/5m/15m/1h/4h/D1; hierarchical temporal representations; HiMTM; multi-scale dilated conv; PatchTST patching; the project's `higher_timeframes`. |
| 15 | Training Strategy | From scratch vs pretraining vs continued pretraining vs fine-tuning vs LoRA/adapters vs supervised vs self-supervised vs RL fine-tuning vs hybrid vs offline/online RL vs imitation (DT/CQL). |
| 16 | Accuracy, Evaluation, and the Gap Between Them | Forecasting accuracy, calibration, P&L, Sharpe, Sortino, MaxDD, profit factor, turnover, costs, robustness, regime stability, OOS; Lopez de Prado PBO/PSR/DSR/CSCV/purged k-fold; GIFT-Eval; why high accuracy can still be a poor system. |
| 17 | Causality and Credit Assignment | State→Action→Outcome vs correlations; counterfactual reasoning; causal representation; action-conditioned world models; what-if simulations; counterfactual Shapley credit assignment. |
| 18 | Frontier AI Architecture Decomposition | What ChatGPT/Gemini/Claude/Grok/robotics/game-playing/computer-use agents are made of: foundation model vs post-training (SFT/RLHF/GRPO/DPO) vs reasoning vs memory vs tools vs orchestration. |
| 19 | New and Emerging Architectures | Foundation models, world models, state-space (Mamba), MoE, retrieval/memory, multimodal, agentic, decision transformers, offline RL, model-based RL, generative market models, new sequence/financial architectures. |
| 20 | The Recommended Architecture: Block-Diagram Walkthrough | DATA→REPRESENTATION→MARKET MODEL→WORLD MODEL→POLICY/RL→DECISION→RISK→EXECUTION→MT5→OUTCOME→REWARD→MEMORY→TRAINING→VALIDATION→SAFE UPDATE. |
| 21 | The 14 Final Decisions | One evidence-based answer per decision, with project implications. |
| — | Open Research Questions | Major unanswered questions, each with the evidence that would resolve it. |
| — | Proposed Experiments | Ordered cheap-to-expensive; each with hypothesis, method, success metric, kill criteria. |
| — | References | All URLs from the research corpus, grouped by section. |

---

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

# Section 4 — The "Trading Brain": Advanced Architectures Compared

## 4.1 The question

Given the system-level recommendation of Section 2 (world-model-centered RL with shared representation and deterministic safety shell), which *model family* should power each block — specifically the "brain" that converts market state into decisions? This section compares: plain Transformer, financial foundation model (FFM), time-series foundation model (TSFM), world model, RL/DRL agent, hybrid, shared backbone, and newer architectures (Mamba/SSM, MoE — detailed in Section 19).

## 4.2 Comparison matrix

| Architecture | What it is | Training objective | Strengths for trading | Weaknesses for trading | Verdict for this project |
|---|---|---|---|---|---|
| **Plain Transformer (self-supervised or supervised)** | Attention-based sequence model; e.g., the project's `TransformerActor`/`TransformerCritic` (`models/transformer_policy.py`) | Supervised next-token/next-value prediction, or RL (PPO) | Strong sequence modeling; long-range context; the project already has one | No inherent decision or dynamics structure; needs RL/supervised wrapper; can overfit short financial histories | Keep as the *backbone of the representation block*, not the whole brain |
| **Financial Foundation Model (FFM)** | FM pretrained on financial series, e.g., Kronos (arXiv 2508.02739) | Autoregressive token prediction on 12B K-line records | Zero-shot price/vol forecasting; synthetic data generation; multi-market priors; proven at scale | Forecasting-focused (needs a decision wrapper); context limits (512 tokens for Kronos); heavy pretraining infra | Adopt as the *market-model / representation encoder* (optionally fine-tuned), with policy on top |
| **Time-Series Foundation Model (TSFM)** | FM pretrained on general TS, e.g., TimesFM 3.0, Chronos-2, Moirai 2.0 | Sequence/quantile forecasting across domains | Zero-shot generalization (GIFT-Eval, arXiv 2410.10393); strong baselines | General-domain priors may underfit financial K-lines (Kronos's stated motivation); forecasting-focused | Fallback market model; or auxiliary forecast sidecar |
| **World Model (model-based RL)** | Learned dynamics model (RSSM/Transformer) + actor-critic trained in imagination; e.g., DreamerV3 (arXiv 2301.04104) | World-model loss (recon+reward+KL) then actor-critic on imagined rollouts | Safe exploration; counterfactual what-ifs; credit assignment; project already implements it (`models/dreamer_agent.py`) | Dynamics can drift in non-stationary markets; needs frequent re-fit; observation space design is critical | **Core brain** (Section 6) — this is the recommendation |
| **RL/DRL agent (model-free)** | Policy trained directly on environment reward; e.g., PPO, DQN | Maximize discounted return | Direct decision optimization; cost-aware if reward encodes costs | Sample-inefficient; needs real/simulated exploration; overfits simulator (DRL-trading literature) | Policy/value heads *inside* the world-model system; not standalone |
| **Hybrid (world model + RL + FFM + deterministic)** | The Section 2 hybrid | Multiple objectives with defined scopes | Each block does one job; safety/audit preserved | Complexity; integration cost | **Adopted** (Sections 20–21) |
| **Shared backbone (multi-task)** | One encoder, many heads (PatchTST, iTransformer pattern) | Joint multi-objective | Representation reuse; regularization | Task interference | Adopted for the representation block |
| **Newer architectures (Mamba/SSM, MoE)** | State-space / mixture-of-experts variants (arXiv 2403.11144, arXiv 2405.16440, arXiv 2407.06204) | Same as Transformer counterparts | Linear-time long context (SSM); parameter scaling with bounded compute (MoE) | Less mature for financial series; no decisive trading evidence yet | Keep as candidates; re-evaluate when evidence emerges (Section 19) |

## 4.3 Evidence-based recommendation

**[RESEARCH FINDING]** The strongest generalist decision-learning evidence is model-based RL (DreamerV3, one config, 150+ tasks, Nature 2025), and the strongest *financial-forecasting* evidence at scale is an autoregressive FM on K-lines (Kronos, 12B records, zero-shot RankIC +93% over leading TSFM). These are complementary, not competing: Kronos-style tokenized OHLCV encoding feeds the market model; DreamerV3-style RSSM/imagination provides decision learning. (arXiv 2508.02739; arXiv 2301.04104.)

**[ENGINEERING RECOMMENDATION]** Use a **shared-backbone world-model architecture with an FFM-style encoder as the market model and an RL policy/critic trained by imagination** — i.e., the hybrid. Concretely for this project: keep `models/dreamer_agent.py`'s RSSM + actor/critic, replace/upgrade its observation encoder so it consumes raw OHLCV (tokenized, Kronos-style, or patched) plus account state, and add an auxiliary volatility/regime head (Section 3.6). The project's `TransformerPolicy` and PPO can be re-purposed as the shared representation backbone rather than as competing "brains".

**[ENGINEERING RECOMMENDATION]** Do not build parallel brains (plain Transformer + FFM + TSFM + RL agent as four independent systems). The only redundancy recommended is (a) an ensemble of critics/policies over the shared representation for calibrated epistemic uncertainty (Section 2.2), and (b) an LLM research desk as an advisory-only layer (Section 18). Everything else is one integrated stack.

---

# Section 5 — Financial & Time-Series Foundation Models in Depth

## 5.1 Purpose and structure of this section

This section provides per-model evidence tables (architecture, training objective/data, context length, predictions, fine-tunability, OHLCV/bid-ask/tick support, multivariate, multi-market, XAUUSD suitability, decision vs prediction, RL combinability, strengths/weaknesses, realistic role) for: Kronos, TimesFM 3.0, TimeGPT, Chronos/Chronos-2, Moirai/Moirai-2, PatchTST, iTransformer, TFT, plus the financial LLM family (FinGPT, FinMem, FinRobot). All identifiers and numbers were verified via `internet_search`/`parse_urls` during this study (arXiv IDs, model cards, docs).

## 5.2 Kronos (Financial Foundation Model)

| Dimension | Evidence |
|---|---|
| Architecture | Two-stage: (1) Transformer-based K-line tokenizer with Binary Spherical Quantization (BSQ), coarse+fine subtokens, hierarchical reconstruction loss; (2) decoder-only autoregressive Transformer with RoPE + RMSNorm. Variants small 24.7M (8L/512d/8h), base 102.3M (12L/832d/16h), large 499.2M (18L/1664d/32h), vocab 2^20. (arXiv 2508.02739, parsed in full.) |
| Training data/objective | Autoregressive next-subtoken prediction on 12B K-line records from 45 global exchanges, 7 temporal granularities; per-dim z-score normalization clipped to [−5,5]; learnable temporal embeddings (minute-of-day … month-of-year). |
| Context length | Max 512 tokens. |
| Predictions | Distributional/generative: samples future K-line sequences (temperature/top-p sampling); Monte Carlo rollouts (average multiple sampled paths) improve forecast stability. |
| Fine-tunability | Pretrained checkpoints public (github.com/shiyu-coder/Kronos); fine-tuning/adaptation is the standard use pattern. |
| OHLCV/bid-ask/tick support | OHLCVA (Open/High/Low/Close/Volume/Amount); tokenized candlesticks, not raw ticks or order book. |
| Multivariate | Multi-feature OHLCVA per K-line, tokenized jointly. |
| Multi-market | Yes — 45 exchanges, diverse asset classes; data rebalancing for under-represented classes. |
| XAUUSD suitability | Directly suitable for candlestick FX/metals series (K-line tokenization is asset-class agnostic); no XAUUSD-specific pretraining claim. |
| Decision vs prediction | Prediction/generation (price series, volatility, synthetic K-lines). Not a decision policy. |
| RL combinability | Good as the *environment/market model*: sample K-line continuations for imagination/training; tokenizer output can feed an RSSM or a policy encoder. |
| Strengths | Zero-shot price forecasting +93% RankIC over leading TSFM and +87% over best non-pretrained baseline; −9% MAE volatility; +22% synthetic K-line fidelity; highest AER/IR in a long-only A-share investment simulation; scales with size. |
| Weaknesses | Forecasting-focused; context 512 tokens; heavy pretraining (12B records) not reproducible by this project from scratch; no execution/order-flow realism. |
| Realistic role | **Market-model encoder / generative market simulator** (fine-tuned or frozen feature extractor) inside the world-model stack — not the decision layer. |

## 5.3 TimesFM 3.0 (Time-Series Foundation Model)

| Dimension | Evidence |
|---|---|
| Architecture | Decoder-only patched Transformer; mixing transformer stack with sequence + variate attention; non-autoregressive decoding; Context Patch Masking (CPM) + RevIN refinement. (google-research/timesfm; research.google blog; HF google/timesfm-3.0-pytorch.) |
| Training data/objective | General time-series corpus; zero-shot generalist forecasting objective. |
| Context length | Patched sequences; ~512 patches class; not financial-specific. |
| Predictions | Point + probabilistic forecasts (per model docs); multivariate + covariates (past-only and past-and-future). |
| Fine-tunability | Yes — HF PyTorch checkpoints; standard fine-tuning/instruction adapters. |
| OHLCV/bid-ask/tick support | Supports arbitrary multivariate numeric series; no native OHLCV/LOB semantics. |
| Multivariate | Native multivariate; flexible covariates. |
| Multi-market | Yes (general TS pretraining); not financial-specific. |
| XAUUSD suitability | Feasible as a forecast sidecar after fine-tuning; no financial prior. |
| Decision vs prediction | Prediction only. |
| RL combinability | As auxiliary forecast/volatility head; weaker as a dynamics model than an RSSM for RL. |
| Strengths | Top-tier zero-shot on major TSFM benchmarks; multivariate + covariates; mature open ecosystem. |
| Weaknesses | Not built for K-line statistical properties (per Kronos's motivation); no execution realism. |
| Realistic role | Auxiliary forecast/volatility sidecar; fallback market model baseline in experiments. |

## 5.4 TimeGPT (Nixtla hosted TSFM)

| Dimension | Evidence |
|---|---|
| Architecture | Hosted pretrained generative Transformer (encoder-decoder style per Nixtla docs); zero-shot + fine-tuning API; also anomaly detection. (nixtla.io/docs About TimeGPT/FAQ; Azure AI catalog TimeGPT-1.) |
| Training data/objective | Claims 100B+ rows across finance/weather/energy/web (vendor claim). |
| Context length | Vendor-managed; API-based. |
| Predictions | Point + prediction intervals; anomaly detection; exogenous variables. |
| Fine-tunability | Fine-tuning via API (limited control vs open weights). |
| OHLCV/bid-ask/tick support | Numeric series API; no native market microstructure. |
| Multivariate | Yes (exogenous support). |
| Multi-market | Yes (general corpus). |
| XAUUSD suitability | Feasible via API; vendor lock-in; no financial microstructure prior. |
| Decision vs prediction | Prediction only. |
| RL combinability | API forecasting only — not usable as a trainable dynamics model. |
| Strengths | Turnkey zero-shot forecasting; simple API; solid general benchmarks. |
| Weaknesses | Hosted (data egress concerns for a trading system); vendor lock-in; no financial-specific priors claimed at the level of Kronos. |
| Realistic role | Research baseline for forecast sidecars; NOT part of the production architecture (open-weights preferred for auditability). |

## 5.5 Chronos / Chronos-2

| Dimension | Evidence |
|---|---|
| Architecture | Chronos (arXiv 2403.07815): tokenizes time series via scaling+quantization, trains existing transformer LM architectures. Chronos-2 (arXiv 2510.15821): group attention enabling in-context learning (ICL) across groups (related series, variates, targets+covariates). (amazon/chronos-2 HF; amazon.science blog; github amazon-science/chronos-forecasting.) |
| Training data/objective | Large corpus of time series; token-level next-token objective. |
| Context length | Standard LM context (several thousand tokens class, variant-dependent). |
| Predictions | Zero-shot univariate, multivariate, and covariate-informed forecasts. |
| Fine-tunability | Yes — open weights, fine-tunable. |
| OHLCV/bid-ask/tick support | Numeric series; no native OHLCV semantics. |
| Multivariate | Yes via group attention (related variates); cross-series ICL. |
| Multi-market | General TS corpus; not financial-specific. |
| XAUUSD suitability | Feasible as forecast sidecar; no financial prior. |
| Decision vs prediction | Prediction only. |
| RL combinability | Forecast sidecar only. |
| Strengths | Strong zero-shot generalization; covariate support; open ecosystem. |
| Weaknesses | General-domain priors; K-line statistical properties under-optimized (per Kronos). |
| Realistic role | Auxiliary forecast/benchmark baseline; not core. |

## 5.6 Moirai / Moirai-2

| Dimension | Evidence |
|---|---|
| Architecture | Decoder-only TSFM; Moirai 2.0 trained on 36M-series corpus; quantile forecasting + multi-token prediction. (arXiv 2511.11698; HF Salesforce/moirai-2.0-R-small.) |
| Training data/objective | 36M series pretraining corpus; quantile/multi-token objectives. |
| Context length | Class/variant-dependent (thousands of tokens). |
| Predictions | Probabilistic (quantiles) + point; multi-token prediction. |
| Fine-tunability | Yes — open weights (HF), small variants. |
| OHLCV/bid-ask/tick support | Numeric series; no native OHLCV semantics. |
| Multivariate | Yes (any-variate support). |
| Multi-market | General corpus. |
| XAUUSD suitability | Feasible sidecar; no financial prior. |
| Decision vs prediction | Prediction only. |
| RL combinability | Forecast sidecar only. |
| Strengths | Top-tier on GIFT-Eval among pretrained models; probabilistic outputs (nice for risk); small runnable variants. |
| Weaknesses | General-domain; financial K-lines under-optimized. |
| Realistic role | Auxiliary probabilistic-volatility sidecar; benchmark baseline. |

## 5.7 PatchTST (representative TS architecture, not a pretrained FM)

| Dimension | Evidence |
|---|---|
| Architecture | Channel-independent patched Transformer: patches subseries-level tokens; self-supervised pretraining possible. (arXiv 2211.14730.) |
| Training data/objective | Supervised/self-supervised on the target dataset (trainable from scratch; strong self-supervised transfer). |
| Context length | Patching reduces length; several hundred patches typical. |
| Predictions | Long-horizon forecasting; channel-independent. |
| Fine-tunability | n/a (it is an architecture, trained per dataset). |
| OHLCV/bid-ask/tick support | Numeric series. |
| Multivariate | Channel-independent design (each channel modeled separately — a deliberate choice). |
| Multi-market | Per-dataset training. |
| XAUUSD suitability | Trainable on XAUUSD OHLCV directly. |
| Decision vs prediction | Prediction architecture. |
| RL combinability | Usable as the shared representation encoder inside the world-model stack (learned from scratch on XAUUSD). |
| Strengths | Strong long-horizon results; self-supervised transfer; lightweight to train. |
| Weaknesses | Not a foundation model (no cross-market priors); channel-independence ignores cross-asset correlation (use iTransformer-style variate attention if that matters). |
| Realistic role | Primary *from-scratch representation encoder* candidate for the project (no 12B-record pretraining needed). |

## 5.8 iTransformer

| Dimension | Evidence |
|---|---|
| Architecture | Inverted Transformer: variates are tokens, attention captures inter-variate correlations, series are embedded per-variate. (arXiv 2310.06625.) |
| Training data/objective | Supervised forecasting per dataset. |
| Context length | Sequence of variate tokens (compact). |
| Predictions | Multivariate forecasts. |
| Fine-tunability | n/a (architecture). |
| OHLCV/bid-ask/tick support | Numeric series. |
| Multivariate | Native and strong at cross-variate correlation. |
| Multi-market | Per-dataset. |
| XAUUSD suitability | Good for XAUUSD+DXY+yields+cross-asset feature panels (Section 13) — correlation is its strength. |
| Decision vs prediction | Prediction architecture. |
| RL combinability | Shared representation encoder; explicitly captures cross-asset co-movement for the world model. |
| Strengths | SOTA multivariate results; interpretable variate attention. |
| Weaknesses | Not pretrained; attention cost grows with variate count. |
| Realistic role | Alternative/adjunct representation encoder when cross-asset features enter (Section 13). |

## 5.9 TFT (Temporal Fusion Transformer)

| Dimension | Evidence |
|---|---|
| Architecture | Multi-horizon forecasting Transformer with static covariate encoders, interpretable attention, quantile outputs. (arXiv 1912.09363.) |
| Training data/objective | Supervised multi-horizon forecasting (trainable per dataset). |
| Context length | Sliding encoders; multi-horizon decoder. |
| Predictions | Quantile forecasts over multiple horizons (nice for risk). |
| Fine-tunability | n/a (architecture). |
| OHLCV/bid-ask/tick support | Numeric series + static/known covariates (calendar, holidays). |
| Multivariate | Yes; known/observed covariates supported. |
| Multi-market | Per-dataset. |
| XAUUSD suitability | Strong for macro-calendar-aware forecasting (news/event covariates); interpretable attention useful for audit. |
| Decision vs prediction | Prediction architecture. |
| RL combinability | Auxiliary forecast sidecar with interpretable attention; can feed regime/vol heads. |
| Strengths | Interpretability; quantile multi-horizon; covariate handling. |
| Weaknesses | Not pretrained; forecasting-focused. |
| Realistic role | Interpretable auxiliary forecaster for risk/regime sidecars and audit artifacts. |

## 5.10 Financial LLM family (FinGPT, FinMem, FinRobot)

### FinGPT

| Dimension | Evidence |
|---|---|
| Architecture | Open-source financial LLMs, data-centric; low-cost adaptation of base LLMs to finance. (arXiv 2306.06031; AI4Finance-Foundation.) |
| Training data/objective | Financial corpora (news, filings, sentiment) for instruction/SFT adaptation. |
| Predictions | Text: sentiment, summarization, Q&A over financial text. |
| Role for trading | Sentiment/news interpretation layer — advisory only. |
| RL combinability | Not for numerical policy; can tokenize news into features for the world model. |

### FinMem

| Dimension | Evidence |
|---|---|
| Architecture | LLM trading agent with **layered memory** (working/consolidated/long-term), character/profile design, self-evolving professional knowledge, decision refinement. (arXiv 2311.13743; github pipiku915/FinMem-LLM-StockTrading; AAAI-SS.) |
| Findings | Reports leading trading performance vs algorithmic agents on a real-world stock dataset in its paper's setups. |
| Role for trading | The *memory architecture* is the transferable idea (Section 8): layered memory + retrieval over experience. Decision authority stays with the numerical core. |
| RL combinability | Memory/retrieval layer can serve the RL core (experience DB, regime histories). |

### FinRobot

| Dimension | Evidence |
|---|---|
| Architecture | Open-source agentic AI platform: foundation models + financial tools + quant models + deterministic computation + multi-agent workflows. (github AI4Finance-Foundation/FinRobot.) |
| Role for trading | Reference for how to compose FM + quant + agents; supports the "LLM research desk, numerical core decides" pattern. |

## 5.11 Section synthesis: realistic roles for FMs in THIS system

| Model | Primary role in the recommended architecture | Decision authority? |
|---|---|---|
| Kronos | Market-model encoder / generative market simulator (fine-tuned or frozen) | No — feeds representation/world model |
| TimesFM 3.0 / TimeGPT / Chronos-2 / Moirai-2 | Auxiliary forecast/volatility sidecars; experimental baselines | No |
| PatchTST / iTransformer | From-scratch representation encoders (PatchTST default; iTransformer when cross-asset panel grows) | No |
| TFT | Interpretable multi-horizon vol/regime forecaster with calendar covariates | No |
| FinGPT | News/sentiment feature extractor | No |
| FinMem | Layered memory design pattern | No |
| FinRobot | Agent-composition reference | No |
| DreamerV3 RSSM + actor-critic | Core decision brain (world model + policy) | **Yes** |

**[ENGINEERING RECOMMENDATION]** Single evidence-based choice: **adopt a Kronos-style (or, in practice for this project, a PatchTST/iTransformer-style from-scratch) learned representation encoder as the market model, keep DreamerV3-style RSSM as the dynamics core, train policy/value by imagination, and use the TSFMs only as auxiliary forecast sidecars.** Do not build five FM systems. The decision authority is the RL core; every FM is a producer of representations or auxiliary signals.

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

# Section 13 — Data: What Enters the Model, at What Resolution

## 13.1 Data classes and their role

**[FACT]** Financial data for trading AI spans: OHLCV bars; tick/quote streams; bid/ask + spread; volume/order flow; order book (LOB); news; economic calendar; macro time series (DXY, yields, indices, commodities); and sentiment. Each class answers a different question, has a different cost/availability, and a different signal-to-noise ratio. (Definitional; standard market-data taxonomy.)

**[FACT]** The project's current data path (`core/feature_pipeline.py`): raw OHLCV → causal feature computation (RSI, ATR, BB position, MACD diff, momentum, volume ratio, vol, higher-TF features) → z-scored features (scaler fit on TRAIN only) → `feature_contract.json` hash validation → windows for the model. Macro features exist with `shift(1)` causal lag; the correlation guard references DXY but **no cross-asset data is actually loaded into features** (DXY momentum is a placeholder 0.0 in `trade_executor.build_market_data`). (Source: direct file reads.)

## 13.2 What should enter the model — and at what resolution

| Data class | Enters model? | Resolution | Evidence/role |
|---|---|---|---|
| OHLCV (XAUUSD) | **Yes — core** | M1/M5/M15/H1/H4/D1 (multi-scale, Section 14) | Base representation; Kronos-style tokenization covers OHLCVA (arXiv 2508.02739) |
| Tick/quote stream | Research only | Tick | LOB/microstructure literature (arXiv 2006.05574, arXiv 2308.13289) — not needed for candle-close cadence; adds latency/quality burden |
| Bid/ask + spread | **Yes — as state features** | Bar-level aggregates | Spread is a *cost and a risk* signal; `RiskSupervisor` max_spread filter; env models spread |
| Volume/order flow | **Yes — as features** | Bar-level | Volume ratio already in the pipeline; order-flow imbalance needs LOB (research) |
| Order book (LOB) | Research only | Tick/event | JAX-LOB/TRADES evidence; not for the current candle-close architecture |
| News | **Yes — as event/risk flags + LLM advisory** | Event windows | Event windows are XAUUSD tail risk; `is_high_impact_event` is a placeholder today → needs a real calendar feed (Section 12.4) |
| Economic calendar | **Yes — as risk flags** | Event windows | NFP/FOMC/CPI windows: position halving already exists (event_position_scale=0.5); calendar feed is the missing input |
| Macro (DXY, yields, indices, commodities) | **Yes — target for phase 2** | H1/D1 | Cross-asset correlations matter for XAUUSD (USD inverse relationship); correlation guard exists but data is missing (Section 13.3) |
| Sentiment | **Yes — as LLM advisory features** | News-derived, low frequency | FinGPT/FinMem evidence for sentiment layers (arXiv 2306.06031, arXiv 2311.13743); advisory only |

## 13.3 Cross-asset: DXY, yields, indices, correlations

**[RESEARCH FINDING]** XAUUSD is well documented as negatively correlated with the US Dollar index and sensitive to real yields and risk sentiment; cross-asset information is standard input for gold forecasts in quant practice. (Standard market knowledge; consistent with correlation-guard design in the project.)

**[FACT]** The project has a correlation guard (`RiskSupervisor._correlation_reject`, blocking long on DXY up-momentum above a threshold, config `correlation_asset="DXY"`, `correlation_block_long_on_up=0.01`) but the DXY data itself is a placeholder 0.0 in `trade_executor.build_market_data`. The guard is therefore **inert** today. (Source: direct file reads.)

**[ENGINEERING RECOMMENDATION]** Phase 2 data plan (single evidence-based priority order):
1. **Real XAUUSD OHLCV at M1–D1** with the existing causal pipeline discipline (no future leaks; scaler on train only).
2. **DXY + 10Y yield + S&P/VIX** at H1/D1, `shift(1)`-lagged, into the feature panel — activating the existing correlation guard with real data and giving the world model cross-asset context.
3. **Economic calendar** (NFP, FOMC, CPI, Powell, etc.) as event flags — completing the `is_high_impact_event` path.
4. **News/sentiment** (FinGPT-style) as low-frequency advisory features for the LLM research desk, never as order inputs.
5. **Tick/LOB** — research-only, deferred; the candle-close architecture does not consume it and adding it would be a parallel system.

**[ENGINEERING RECOMMENDATION]** Data quality rules that must hold (mirroring the project's existing rigor): every feature must be causal (compute on closed bars only; `shift(1)` for macro/indicators); scaler fit on train only; contract hashing (`feature_contract.json`) extended to new features; duplicate/OHLC-sanity validation (as `prepare_ohlc` does) applied to every new feed; and every new data source is versioned so model cards can be reproduced.

## 13.4 The "what not to feed" list (anti-leak and anti-noise)

**[ENGINEERING RECOMMENDATION]** (a) No future information of any kind (the pipeline's P0-4 leak fixes were exactly this — keep them for all new feeds). (b) No raw news text into the numeric core — LLM advisory only. (c) No LOB until the architecture actually decides intra-candle. (d) No high-cardinality raw tick streams at bar resolution (noise dominates). (e) Sentiment only as *documented* advisory features, with drift monitoring — sentiment models degrade (FinGPT/FinMem discussions of data freshness).

---

# Section 14 — Multi-Timeframe and Multi-Resolution Modeling

## 14.1 Why multi-resolution

**[FACT]** Markets contain patterns at multiple time scales: microstructure (ticks), intraday (M1–M15), swing (H1–H4), and position/macro (D1). A single-resolution model either ignores fast context or drowns in noise; multi-resolution representations let the model condition short-term decisions on longer-term context. (Definitional; multi-scale TS literature.)

**[FACT]** The project already computes higher-TF features in `core/feature_pipeline.py` (higher-TF resample with `right/closed` + `shift(1)` — leak-free by construction) and the Transformer policy consumes a flat window (seq_len=64). (Source: direct file reads.)

## 14.2 Evidence for hierarchical multi-scale modeling

**[RESEARCH FINDING]** HiMTM (arXiv 2401.05012) proposes hierarchical multi-scale masked time-series modeling — masking and reconstructing across scales to learn representations that capture multi-scale structure, improving downstream forecasting. Multi-scale dilated convolution (arXiv 2405.05499) extracts multi-scale information via exponentially growing dilation rates. PatchTST (arXiv 2211.14730) shows patching (subseries tokens) is an effective way to model local structure with long horizons. Together these support: **learn a shared multi-scale representation rather than hand-picking a single TF.**

**[ENGINEERING RECOMMENDATION]** The representation block should be multi-scale in a *structured* way:
1. **Multi-TF feature fusion (existing):** keep the higher-TF feature branches in the pipeline (M15 context into M5 decisions etc.) — already leak-free.
2. **Patch-based temporal encoder (new):** use patching (PatchTST-style) on the finest decision-TF so the encoder sees subseries structure without exploding sequence length; the transformer seq_len=64 window becomes patches of the fine scale plus coarse-scale tokens from higher TFs.
3. **Scale-aware conditioning:** the regime/context module (Section 9) and long-term memory (Section 8) supply the slow context (D1/macro); the decision policy operates at the fast scale.

## 14.3 The recommended multi-TF stack for XAUUSD

**[ENGINEERING RECOMMENDATION]** Single evidence-based stack (matches the project's existing granularities):
- **Decision TF: M5 (or M15)** — enough signal for XAUUSD intraday without tick noise; the live loop's candle-close cadence (`on_closed_bar` on new closed bar) already fits.
- **Context TFs: M15, H1, H4, D1** — supplied as multi-scale features (existing higher-TF branches) + macro panel (DXY, yields at H1/D1).
- **Representation:** patched encoder at decision TF + fused multi-TF feature tokens; RSSM latent as the temporal belief state (Section 6.2.1).
- **Policy:** consumes the fused multi-scale representation; action = plan (Section 10.3).

**[ENGINEERING RECOMMENDATION]** Do NOT train separate models per timeframe and vote/ensemble them — that is the 5-parallel-systems anti-pattern. One shared multi-scale representation, one policy. The project's current higher-TF feature branches are already the right raw material; the upgrade is (a) raw-OHLCV patching at decision TF, and (b) fusing rather than concatenating all TF features into a flat vector (concatenation of many scales inflates dimensionality and confounds the model — use a small fusion network or attention over scale tokens).

## 14.4 Multi-resolution validation

**[ENGINEERING RECOMMENDATION]** Every TF choice and fusion design must be validated with the same honest protocol: purged/embargoed walk-forward (`backtest/engine.py::walk_forward`) with fixed parameters first, then model evaluation with the promotion gate. TF experiments that only improve in-sample or on one path are rejected (Section 16).

---

# Section 15 — Training Strategy

## 15.1 The option space

| Training mode | What it does | Evidence | Verdict |
|---|---|---|---|
| **From scratch (supervised)** | Train encoder/policy on XAUUSD data only | PatchTST/iTransformer/TFT train per dataset; works when data is sufficient and priors weak | Baseline for the representation encoder (PatchTST-style) |
| **Pretraining (FM)** | Large-scale unsupervised pretraining then adapt | Kronos 12B records → zero-shot gains (arXiv 2508.02739); TSFMs: TimesFM 3.0, Chronos-2, Moirai-2 (GIFT-Eval, arXiv 2410.10393) | Adopt for the *market model* (Kronos-style or frozen TSFM features) if a pretrained checkpoint is used; otherwise PatchTST from scratch |
| **Continued pretraining** | Keep training a pretrained FM on domain data | Standard FM practice; mitigates domain gap | Adopt for XAUUSD-specific adaptation of a Kronos-style encoder |
| **Fine-tuning** | Full or partial retraining of a pretrained model on target data | Standard practice; LoRA (arXiv 2106.09685) performs on-par-or-better than full fine-tuning with far fewer trainable parameters and no inference latency | **Adopt LoRA/adapters** for any FM in the stack (cheap, auditable, reversible) |
| **Supervised (forecast/classification)** | Labels from future returns | Standard; but accuracy≠profitability (Section 16) | Auxiliary heads only (vol/regime), never the decision path |
| **Self-supervised (TS)** | Contrastive/generative pretraining on series | TF-C (arXiv 2206.08496), contrastive-vs-generative studies (arXiv 2403.09809), self-supervised TS forecasting (S0950705124012863) | Adopt for the *representation encoder*: masked reconstruction (PatchTST-style) or contrastive TF-C-style pretraining on XAUUSD+macro before RL |
| **RL fine-tuning** | Policy/value optimized by reward after supervised/self-supervised pretraining | DreamerV3 pipeline: world-model then actor-critic (arXiv 2301.04104) | **Adopt** — this is the recommended pipeline (representation → world model → RL) |
| **Offline RL** | Batch learning from historical experience | CQL (arXiv 2006.04779) conservative lower-bound value; DT (arXiv 2106.01345) return-conditioned | **Adopt as safety phase**: offline pretraining/regularization before live-safe RL |
| **Online RL** | Continuous live updating | Continual-learning risks (arXiv 2403.05175); DRL-trading overfitting warnings | Reject as default; experiment only under the safe-update protocol (Section 7.6) |
| **Imitation (behavior cloning)** | Learn from demonstrations | DT-style return-conditioned BC; useful to seed policy | Experiment: seed the policy on good historical plans |

## 15.2 Evidence details for the recommended pipeline

**[RESEARCH FINDING]** Self-supervised time-series representation learning is well supported: TF-C (arXiv 2206.08496) learns time-frequency-consistent representations that transfer across tasks; a comparative study (arXiv 2403.09809) and a 2024 self-supervised forecasting study (S0950705124012863) both find strong benefits for masked/contrastive pretraining in forecasting. **[RESEARCH FINDING]** LoRA (arXiv 2106.09685) is on-par-or-better than full fine-tuning on GPT-3/GPT-2 with fewer trainable parameters and no added inference latency — the standard way to adapt a pretrained FM cheaply. **[RESEARCH FINDING]** Offline RL — CQL (arXiv 2006.04779, 2–5× higher final return vs existing methods in its benchmarks) and DT (arXiv 2106.01345) — is the evidence base for batch, conservative learning from historical data. **[RESEARCH FINDING]** GRPO (arXiv 2402.03300) shows critic-free group-relative RL post-training works for LLMs (relevant to Section 18's RL-in-the-LM discussion, and to any future LLM-policy experiments); it does not replace the critic in the numeric RL core where a value function is explicitly needed.

## 15.3 The recommended single training pipeline

**[ENGINEERING RECOMMENDATION]** **Pretrain/self-supervise the representation → train the world model → offline-regularize the policy → RL fine-tune in imagination → promote through the gate.** Concretely for the project:
1. **Representation:** self-supervised masked-reconstruction (PatchTST-style) on XAUUSD + macro OHLCV at decision TF (no labels, no leakage). This replaces *training the encoder from random init inside RL* — evidence-backed (TF-C/self-supervised TS).
2. **World model:** RSSM trained on the frozen-or-fine-tuned representation with symlog/two-hot/free-nats (existing `models/dreamer_agent.py` machinery; add raw-OHLCV observations).
3. **Policy pretraining (offline):** CQL-conservative offline RL and/or DT return-conditioned behavior cloning on the stratified experience DB (historical good plans; Section 8). This seeds the policy conservatively.
4. **RL fine-tuning:** imagination-based actor-critic with the risk-aware reward (Sections 6, 11); PPO-clipped updates for stability (existing `models/transformer_policy.py` machinery shows the clipping pattern already works).
5. **Validation & promotion:** purged/embargoed walk-forward + DSR/PSR-gated comparison vs incumbent and vs buy-and-hold + cost-inclusive metrics; promote only through `enforce_model_promotion_gate`.
6. **Continual:** EWC-regularized, replay-stratified updates on new experience (Section 7), champion/challenger protocol, auto-rollback.

**[ENGINEERING RECOMMENDATION]** Reject parallel training paths (do not also train a standalone supervised forecaster, a separate TSFM-finetune, and an online-RL agent as three competing models). Every experiment variant (DT vs CQL pretraining, TSFM-features vs PatchTST encoder, regime-conditioned world model) is a *challenger within the same pipeline*, gated by the same protocol.

## 15.4 Honest expectations

**[FACT]** The project has shipped **no trained checkpoint** — `artifacts/models/ppo_gold_v1/` contains a model.zip, manifest, evaluation.json, and feature_contract.json, but the README/history records the ML path as refusing to fabricate results and no validated checkpoint has been promoted. (Source: project inspection + README.) **[ENGINEERING RECOMMENDATION]** The training pipeline above is therefore a *to-be-built* capability: it should be brought up first on synthetic/clean data (smoke tests), then historical XAUUSD with the honest evaluation protocol, and only then considered for any live shadow. Success is defined by the evaluation protocol (Section 16), not by a trained artifact existing.

# Section 16 — Accuracy, Evaluation, and the Gap Between "Good Forecasts" and "Good Trading"

## 16.1 The honest evaluation stack

**[FACT]** The project's evaluation stack today: deterministic backtester on kernc/backtesting.py with a conservative CostModel (one-way spread 0.000175, commission 0.00003/side, round-trip total 0.000410), `walk_forward` with purge+embargo (train 800 / embargo 25 / test 300 bars, sliding), 20 metrics including total_return, buy_hold_return, Sharpe, Sortino, Calmar, max_drawdown, win_rate, profit factor, SQN, commissions; `BacktestResult.metrics` and `summarize_walk_forward` (equal-weight mean over windows). The README honestly states: rule strategies do **not** beat buy-and-hold net of costs. (Source: direct file reads + README.)

**[FACT]** The project also ships: `artifacts/models/ppo_gold_v1/` with model.zip, manifest.json, evaluation.json, feature_contract.json — but **no trained checkpoint has been validated/promoted**; the ML path refuses to fabricate results, and the legacy backtester (which produced inflated claims) is archived as `archive/backtest_engine_legacy_fake.py` — the exact artifact that created the illusion the honest engine later retracted. (Source: project inspection + README.)

## 16.2 Why high forecasting accuracy can still be a poor trading system

**[RESEARCH FINDING]** Backtest overfitting is a selection-bias phenomenon: optimizing over many strategy configurations on one history produces inflated performance. Lopez de Prado's methods — Combinatorial Purged Cross-Validation (CPCV), Probability of Backtest Overfitting (PBO), Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR) — correct performance for selection bias under multiple testing and for non-normality of returns (SSRN 2460551; "How To Backtest Correctly" repo; quantstrategy.io purged-k-fold article). Walk-forward on a single path is dangerous because it tests only one sequence of events; purged k-fold tests across multiple regimes.

**[RESEARCH FINDING]** Forecast accuracy and trading profitability decouple once costs, adverse selection, and position sizing enter: a forecast can be excellent in MAE/RankIC terms and still lose money net of costs, and a mediocre forecast can trade well with good risk/execution. The GIFT-Eval benchmark (arXiv 2410.10393) measures forecasting quality (zero-shot, 23 dataset groups, 7 domains) — it deliberately does **not** measure trading P&L; the two must never be conflated.

**[ENGINEERING RECOMMENDATION]** The evaluation protocol for this project's trading system must therefore be: (a) **forecast evaluation** (for sidecar heads only): proper scoring rules, calibration (reliability diagrams), RankIC/quantile coverage; (b) **trading evaluation** (for the decision system): net-of-cost P&L, Sharpe, Sortino, MaxDD, profit factor, turnover, cost burden, all on purged/embargoed out-of-sample paths, plus (c) **selection-bias correction**: PSR/DSR on the best of N tried variants, CPCV where computationally feasible, and a deflated significance threshold for promotion (Section 7.3). The promotion gate (`evaluation.json passed==true` + contract_hash) must grow these tests in.

## 16.3 The metric set for XAUUSD

| Metric | What it catches | Project status |
|---|---|---|
| Net-of-cost return vs buy-and-hold | The fundamental honesty check | ✅ computed; README reports strategies do not beat it |
| Sharpe / Sortino / Calmar | Risk-adjusted return; downside | ✅ in `BacktestResult.metrics` |
| Max drawdown | Ruin risk; pairs with RiskSupervisor breaker | ✅ in metrics + live risk state |
| Profit factor | Wins/losses; overtrading detection | ✅ in metrics |
| Turnover + cost burden | Overtrading; cost sensitivity | ⚠️ commissions tracked; add explicit turnover/cost-per-trade reporting |
| Win rate / payoff ratio | Edge structure | ✅ in metrics |
| Regime stability | Performance per regime (Section 9) | ⚠️ MAML regime labels exist for training; add per-regime eval reporting |
| PSR/DSR | Selection bias under multiple testing | ❌ not implemented — recommended addition |
| Purged/embargoed OOS | Overfitting to a single path | ✅ walk_forward purge+embargo; extend to CPCV |
| Confidence calibration | Sizing safety (Section 10.2) | ❌ `PolicyOutput.confidence` uncalibrated today — add reliability checks before trusting it for sizing |
| Cost-inclusive imagination | World model realism (Section 6.3) | ⚠️ env has costs; verify dreamed rollouts include them |

## 16.4 Evaluation methodology correctness rules (for this study's experiments)

**[ENGINEERING RECOMMENDATION]** (1) Never tune on the test path; every experiment ledger entry records the exact variant + seed + cost model. (2) Compare vs **buy-and-hold and vs the incumbent**, both net of the same costs, on the same windows. (3) Report per-regime and per-window dispersion, not just the mean. (4) Use deflated significance (PSR/DSR) before any promotion; a "champion" must beat the incumbent beyond noise. (5) Treat the honest backtester as the *only* promotion authority — never the simulator (Section 12.3). (6) Log all model artifacts with contract hashes so evaluations are reproducible (`feature_contract.json` pattern extended to model+data contracts).

## 16.5 Reporting the project's own discrepancy honestly

**[FACT]** The project README and test artifacts disagree on the pytest count: the README reports **87 passed**, while `artifacts/pytest_final.txt` records **139 passed**. This study does not silently pick one; the discrepancy itself is a finding: it indicates the test suite grew (or the README predates the final suite) after the README's numbers were written. [ENGINEERING RECOMMENDATION] The project should re-run the suite and reconcile the count in the README as part of the audit trail — the discrepancy does not affect the strategy findings (which come from the honest backtester, not the test count).

---

# Section 17 — Causality and Credit Assignment

## 17.1 Correlation is not the trading signal

**[FACT]** Trading decisions are causal interventions: taking an action changes the distribution of future outcomes. Correlations learned from historical data — including price autocorrelation, indicator→return relationships — can be spurious, regime-dependent, or reversed out-of-sample. (Definitional; standard causal-inference framing.)

**[RESEARCH FINDING]** The market-efficiency literature and empirical ML studies find limited out-of-sample predictability after costs (Springer 10.1007/s10614-025-11168-9 — extensive ML study of directional predictability/profitability on an aggregate index; MDPI 2079-9292/14/9/1721 — critical examination of AI stock forecasting vs market efficiency; arXiv 2501.07489 — how low-cost AI universal approximators reshape market efficiency). These do not prove unpredictability but bound what correlation-mining alone can deliver. The project's honest result — rule strategies do not beat buy-and-hold net of costs — is consistent with this literature. (All verified via internet_search.)

## 17.2 The State→Action→Outcome structure and counterfactual reasoning

**[RESEARCH FINDING]** Counterfactual credit assignment for RL (arXiv 2607.16999) attributes joint outcomes to individual decisions by simulating "what would have happened if this decision had differed" — the standard principled answer to delayed-feedback credit assignment in trading (Section 7.2). This requires an action-conditioned model of the world — precisely the world model's core strength (Section 6): the RSSM can answer "if I had been flat here instead of long, what equity path follows?"

**[ENGINEERING RECOMMENDATION]** Adopt a two-layer credit-assignment design:
1. **Imagination-based credit assignment (RL standard):** the critic distributes reward over dreamed trajectories (DreamerV3 machinery) — this is the *statistical* credit layer.
2. **Counterfactual audit layer (new, for validation not gradients):** after each trade or batch, run what-if rollouts (flat/long/short/half-size/alternate-exit) through the world model and report "decision X contributed +Y/-Z to this outcome" as an *interpretability and risk* artifact. Do **not** backprop through these counterfactuals initially — first use them as human-auditable explanations and as data for the credit-assignment experiments (Proposed Experiments).

## 17.3 Causal representation learning

**[RESEARCH FINDING]** Causal representation learning (learning representations that respect underlying causal structure) is an active research frontier; no production trading system demonstrates reliable end-to-end causal discovery from prices alone. (Standard research consensus; flagged as frontier in the surveys reviewed for Section 19.) **[UNPROVEN/HYPOTHETICAL]** Whether learned causal representations of financial series materially improve trading decisions is unproven — the efficient-market literature (Section 17.1) argues the exploitable causal structure after costs is thin. Treat as a research program, not a dependency of the production architecture.

## 17.4 What-if simulation as the practical causality tool

**[ENGINEERING RECOMMENDATION]** The practical, evidence-supported causality tool for this system is **action-conditioned what-if simulation through the world model** (supported by world-model RL evidence, Section 6, and counterfactual credit assignment, arXiv 2607.16999). Uses: (a) pre-trade risk checks — "what if XAUUSD gaps 20 pips against my position?"; (b) post-trade attribution; (c) strategy stress tests across regimes; (d) reward-design validation — does the reward actually prefer the counterfactual that was better? This turns the world model from a training accelerator into a *risk and audit instrument*, which is its most defensible production value given the efficiency evidence.

---

# Section 18 — Frontier AI Architecture Decomposition: What "AI" Actually Is in Modern Systems

## 18.1 Decomposing ChatGPT/Gemini/Claude/Grok (and robots, games, computer-use agents)

**[RESEARCH FINDING]** Modern LLM systems are layered: (1) a **foundation model** (pretrained Transformer) provides general knowledge; (2) **post-training** — SFT (supervised fine-tuning), RLHF/RL (including GRPO/DPO/RLAIF variants) — aligns behavior and teaches reasoning; (3) **reasoning** (chain-of-thought/test-time compute) improves multi-step problems; (4) **memory/context** management; (5) **tools** (search, code execution, browsers, APIs); (6) **orchestration** (agents: planning, task decomposition, reflection). The model is the substrate, not the whole system. (arXiv 2407.16216 — RL for LLM post-training survey; arXiv 2502.21321 — LLM post-training deep dive incl. GRPO/DPO; jxzhangjhu.github.io/blog/2026 how-frontier-labs-train-llms — field guide to pretraining/post-training/eval/safety; Medium/Suyog Joshi "How Modern LLM Systems Really Work".)

**[RESEARCH FINDING]** LLM-agent planning (arXiv 2402.02716 — first systematic survey) organizes agent planning into: task decomposition, plan selection, external modules, reflection, and memory — the same decomposition applies to any agentic system, including trading agents. GRPO (arXiv 2402.03300, DeepSeekMath) shows critic-free group-relative RL post-training is feasible and cheap; the demystification/theory follow-up (arXiv 2603.01162) clarifies when group-relative baselines work.

## 18.2 The trading-system analogue of the frontier decomposition

| Frontier-layer component | Analogue in the recommended trading system | Project status |
|---|---|---|
| Foundation model (pretrained) | Kronos-style market-model encoder / TSFM sidecars; LLM research desk (FinGPT-class) | ❌ absent today |
| Post-training (SFT/RLHF/RL/GRPO/DPO) | Offline RL pretraining (CQL/DT) → imagination-based RL fine-tuning (Section 15); reward shaping (Section 11) | ⚠️ RL exists (PPO/Dreamer) but not the offline→RL pipeline |
| Reasoning (test-time compute) | World-model rollouts / MCTS planning at decision time (`models/mcts.py` PUCT exists) | ✅ partially present (MCTS) |
| Memory | Layered memory (working/consolidated/long-term) — Section 8 | ⚠️ replay+latent only |
| Tools | Market-data feeds, calendar/news, broker execution, backtester | ✅ mostly present (live stack) |
| Orchestration (agents) | LLM research desk → numeric decision core → deterministic risk → executor | ⚠️ wiring exists (live loop); advisory LLM layer missing |

## 18.3 What this decomposition implies for trading AI

**[ENGINEERING RECOMMENDATION]** (1) **Do not expect a single LLM to trade**: the frontier decomposition shows every serious system is model + post-training + memory + tools + orchestration; an LLM-only trader skips the market model entirely. (2) **The numeric core is the "model + post-training" layer** for markets; LLM components are tools/orchestration aids. (3) **RL post-training is the frontier-standard way to turn a predictive model into a decision system** — the trading analogue is exactly the recommended offline-pretrain → RL-fine-tune pipeline. (4) **Reasoning/test-time compute** maps to MCTS/world-model planning, which the project already has; deepen it as compute allows. (5) **Agentic orchestration** maps to the research desk pattern: LLM agents analyze and *propose*; the numeric core *decides*; the deterministic shell *enforces*. (TradingAgents-style multi-agent order-debates are an experiment, not production — Section 2.6.)

## 18.4 LLM in the loop: authority boundaries

**[ENGINEERING RECOMMENDATION]** The LLM research desk (FinGPT/FinRobot-class, with RAG over news/calendar — Section 8.3) produces: event-risk flags, sentiment summaries, macro context, and post-trade explanations. It has **no order authority**, its outputs are feature/context inputs, and its correctness is monitored (its drift/error rate feeds no live decisions). This keeps the decision authority in the RL core and the safety authority in the deterministic shell — the two places that can be validated (Sections 15, 16).

---

# Section 19 — New Architectures: What Is on the Frontier and What This System Should Use

## 19.1 The frontier list

**[FACT]** The current frontier of sequence/decision architectures includes: foundation models (financial and TS — Section 5), world models (Section 6), state-space models (Mamba), mixture-of-experts (MoE), retrieval/memory-augmented models, multimodal models, agentic systems, decision transformers, offline RL, model-based RL, generative market models, and new sequence architectures (patching, inverted attention, group attention). (Definitional; covered in the surveyed literature.)

## 19.2 State-space models (Mamba) for time series

**[RESEARCH FINDING]** Mamba (selective state-space models) processes sequences with near-linear complexity; "Is Mamba Effective for Time Series Forecasting?" (arXiv 2403.11144) finds it a competitive alternative to Transformers for long-term TSF; MambaTS (arXiv 2405.16440) improves selective SSM for long-term multivariate forecasting (structured dependency modeling + linear scan). These are evidence-backed candidates for the *representation/dynamics backbone*.

**[ENGINEERING RECOMMENDATION]** Treat Mamba/MambaTS as a **challenger backbone** for the representation encoder and possibly the RSSM's deterministic path (TransDreamer-style swap, Section 6.1), evaluated under the same protocol. Do not adopt on hype: run the side-by-side with PatchTST/iTransformer on the project's own XAUUSD purged/embargoed evaluation first. Near-linear complexity matters if context grows to multi-TF + macro panels (Section 14).

## 19.3 Mixture-of-Experts (MoE)

**[RESEARCH FINDING]** MoE surveys (arXiv 2407.06204 — sparse MoE in LLMs; arXiv 2602.08019 — sparse MoE survey) show sparse activation scales parameter count with comparable compute. In trading, per-regime experts (Section 9.2) are the natural MoE application, with the regime gate selecting experts — combining MoE with the regime module and the forgetting defenses (Section 7.4).

**[ENGINEERING RECOMMENDATION]** MoE enters in the **later phase** as per-regime expert policies/critics gated by the regime module — an explicit, interpretable MoE (regime id is the gate input), not a random sparse routing. This preserves interpretability and gives forgetting resistance by construction.

## 19.4 Retrieval/memory-augmented, multimodal, agentic

**[RESEARCH FINDING]** Retrieval-augmented generation and memory-augmented architectures are established for LLMs (RAG; FinMem layered memory, arXiv 2311.13743). Multimodal models (text+images+series) and agentic systems are frontier-active (agent planning survey, arXiv 2402.02716). **[ENGINEERING RECOMMENDATION]** For this system: memory/retrieval is adopted in the Section 8 design (experience DB + regime store + LLM RAG); multimodal is *not* needed (charts as images add little to numeric OHLCV; skip); agentic is adopted only as the advisory research desk (Section 18.4).

## 19.5 Decision Transformers and offline RL

**[RESEARCH FINDING]** Decision Transformer (arXiv 2106.01345) frames RL as return-conditioned sequence modeling — evidence that decisions can be learned as a sequence-prediction problem; CQL (arXiv 2006.04779) provides conservative offline value learning (2–5× final-return improvements in its benchmarks). **[ENGINEERING RECOMMENDATION]** Adopt both as the *offline pretraining/regularization* phase of the training pipeline (Section 15.3): DT for return-conditioned behavior seeding, CQL for conservative value estimates — then imagination RL fine-tuning.

## 19.6 Model-based RL and generative market models

**[RESEARCH FINDING]** DreamerV3 (arXiv 2301.04104, Nature 2025) is the strongest model-based RL evidence (Section 6). Generative market models: Kronos synthetic K-line generation (+22% fidelity, arXiv 2508.02739) and TRADES (arXiv 2502.07071, realistic LOB generation) show generative models can produce training/simulation data. **[ENGINEERING RECOMMENDATION]** The world model *is* the system's generative market model (Section 12.1); Kronos-style generation is an optional supplement for data augmentation and stress tests. Do not build a separate GAN/diffusion market simulator in parallel — the RSSM + honest backtester cover training and evaluation respectively.

## 19.7 New sequence/financial architectures — final stance

**[ENGINEERING RECOMMENDATION]** The system's stack: **patched Transformer or Mamba backbone (representation) → RSSM/Transformer latent dynamics (world model) → actor-critic with distributional critic (policy) → deterministic risk shell; optional MoE per-regime experts and DT/CQL offline phases; LLM research desk with RAG.** Every "new architecture" is evaluated as a challenger inside this single stack under the honest protocol — never adopted in parallel and never on hype. **[UNPROVEN/HYPOTHETICAL]** Whether any of these newer architectures deliver exploitable net-of-cost XAUUSD alpha is unproven; the efficiency literature (Section 17.1) and the project's own honest results set the expectation bar at "at least beats buy-and-hold and the incumbent, with deflated significance."

# Section 20 — Final Architecture Recommendation: Block-by-Block Walkthrough

## 20.1 The block diagram

```
 DATA ──▶ REPRESENTATION ──▶ MARKET MODEL ──▶ WORLD MODEL ──▶ POLICY/RL ──▶ DECISION ──▶ RISK ──▶ EXECUTION ──▶ MT5
                                                                                                      │
  MEMORY ◀────────────────────── TRAINING ◀── VALIDATION ◀── REWARD ◀── OUTCOME ◀─────────────────────┘
        │                        ▲
        └── SAFE UPDATE ─────────┘
```

Every block below is specified as: **why** (evidence), **which model** (specific architecture), **what learns** (training signal), **inputs/outputs** (interface contract), **training mode** (offline/online/cadence), **communication with neighbors** (what it sends/receives). This is the single integrated architecture — one system, not five.

## 20.2 Block-by-block specification

### Block 1 — DATA

- **Why:** All downstream value starts from honest, leak-free, multi-class data (Sections 13, 14). The project's causal pipeline (`core/feature_pipeline.py`, `backtest/engine.py::prepare_ohlc`) is the model for correctness.
- **Which model:** None — ingestion/validation layer. Feeds: XAUUSD OHLCV (M1–D1), DXY + 10Y + VIX (H1/D1, phase 2), economic calendar events, news (for the LLM desk), fills/trades (feedback).
- **What learns:** Nothing (except data contracts/hashes — versioned, not learned).
- **Inputs/Outputs:** Raw market feeds → validated, deduplicated, OHLC-sanity-checked bars + event flags + cross-asset panel + trade log.
- **Training mode:** n/a.
- **Neighbors:** → REPRESENTATION (bars/features); → MEMORY (trade log); → RISK (live prices for breakers).

### Block 2 — REPRESENTATION

- **Why:** Raw prices are high-dimensional, noisy, scale-variant; a learned representation compresses and denoises (self-supervised TS evidence: TF-C arXiv 2206.08496, contrastive-vs-generative arXiv 2403.09809; PatchTST patching arXiv 2211.14730; iTransformer cross-variate arXiv 2310.06625).
- **Which model:** Shared-backbone encoder — **patched Transformer (PatchTST-style) on decision-TF OHLCV + fused multi-TF feature tokens + macro panel**; Mamba/MambaTS as challenger backbone (arXiv 2403.11144, arXiv 2405.16440). Optionally seeded by a frozen Kronos-style encoder (arXiv 2508.02739) if a pretrained checkpoint is used (LoRA-adapted, arXiv 2106.09685).
- **What learns:** Self-supervised masked reconstruction / contrastive time-frequency consistency on XAUUSD + macro (no labels, no leakage). Later, representation gradients continue through world-model and RL training.
- **Inputs/Outputs:** (multi-TF bars, macro, event flags) → token embeddings z_rep (compact, scale-invariant).
- **Training mode:** Offline pretraining first (self-supervised), then end-to-end fine-tuning with the stack.
- **Neighbors:** ← DATA; → MARKET MODEL/WORLD MODEL (z_rep); → MEMORY (embedding store for retrieval).

### Block 3 — MARKET MODEL

- **Why:** The system needs a model of *market dynamics* — how prices/volatility evolve — distinct from the policy (which decides). Evidence: Kronos is a financial FM for exactly this (arXiv 2508.02739: price + volatility forecasting, synthetic K-lines); TSFMs benchmarked by GIFT-Eval (arXiv 2410.10393) define the zero-shot bar.
- **Which model:** The RSSM's dynamics prior/reward decoder **fed by the representation encoder** (project `models/dreamer_agent.py`), i.e., the market model and the world model share the representation but have distinct roles. Optional auxiliary heads: volatility forecast (Kronos-style −9% MAE evidence), regime features.
- **What learns:** Dynamics prediction loss (next-latent, reward, reconstruction) — the world-model loss (symlog/two-hot/free-nats).
- **Inputs/Outputs:** (z_rep, action/horizon plan) → (predicted next latent, reward prediction, volatility estimate).
- **Training mode:** Offline, rolling windows (purged/embargoed validation); continuous maintenance (Section 7.3).
- **Neighbors:** ← REPRESENTATION; → WORLD MODEL (dynamics prior); → RISK (volatility forecast for sizing/filters).

### Block 4 — WORLD MODEL

- **Why:** Safe exploration, counterfactual what-if, credit assignment, planning — the DreamerV3 evidence base (arXiv 2301.04104; Nature 2025; TransDreamer arXiv 2209.14153 for Transformer backbones; 251-dreamer-trading motivation for trading).
- **Which model:** RSSM latent dynamics (deterministic h + stochastic z) with symlog/two-hot/free-nats — **existing `models/dreamer_agent.py`, upgraded** to consume the learned representation (Block 2) and to condition on regime id (Block 9) and macro context.
- **What learns:** World-model loss (reconstruction + reward + KL, free-nats balanced); MAML meta-adaptation across regimes for non-stationarity (existing `models/meta_learning.py`, upgraded to adapt the world model — already its behavior: MAML adapts world-model params only).
- **Inputs/Outputs:** (z_rep, plan/action, regime id) → (latent belief h,z; imagined rollouts: future latent + reward + vol paths).
- **Training mode:** Offline rolling; imagination rollouts at decision time for planning (MCTS, `models/mcts.py`).
- **Neighbors:** ← MARKET MODEL (dynamics prior); → POLICY/RL (dreamed rollouts for actor-critic); → DECISION (what-if risk scenarios); → VALIDATION (counterfactual audit).

### Block 5 — POLICY/RL

- **Why:** Decisions under uncertainty are an RL problem; the actor-critic on dreamed rollouts is the evidence-backed choice (DreamerV3; CQL arXiv 2006.04779 and DT arXiv 2106.01345 for offline phases).
- **Which model:** Actor (policy over the plan action space — direction + horizon class, Section 10.3) + distributional critic (two-hot, values risk-aware reward, Section 11). PPO-clipped updates for stability (existing `models/transformer_policy.py` shows the clipping machinery). Ensemble of critics for epistemic uncertainty (existing `models/ensemble.py` — keep as uncertainty signal, not parallel decision path).
- **What learns:** Imagination-based actor-critic objective on the risk-aware composite reward; offline CQL/DT pretraining phase (Section 15.3); EWC-regularized continual updates (Section 7.4).
- **Inputs/Outputs:** (latent belief h,z; imagined rollouts) → (plan: direction, confidence, horizon class; value estimate; win-probability for sizing).
- **Training mode:** Offline (imagination + replay), never live-SGD in phase 1 (Section 7.6).
- **Neighbors:** ← WORLD MODEL (dreams); → DECISION (plan); → MEMORY (policy checkpoints, replay data); ← REWARD (objective).

### Block 6 — DECISION

- **Why:** Convert the learned plan into a concrete, auditable order intent; a human-readable decision point (interpretability, Section 16).
- **Which model:** Deterministic logic: plan → (direction, confidence, horizon) → desired size via risk budget + model-vol (upgraded `ATRPositionSizer`/`KellyPositionSizer`), SL/TP via ATR multiples (existing `entry_sl_tp`), event-window halving (existing `RiskSupervisor` event_position_scale).
- **What learns:** Nothing (deterministic conversion — safety shell principle).
- **Inputs/Outputs:** (plan, vol forecast, regime stats, account state) → (order intent: side, size, SL, TP, horizon).
- **Training mode:** n/a.
- **Neighbors:** ← POLICY/RL; ← REGIME (stats); → RISK (full exposure check).

### Block 7 — RISK

- **Why:** Circuit breakers must be deterministic, persisted, and auditable — a learned risk layer could be optimized away by the policy (Section 2.5). The project's `RiskSupervisor` (SQLite-persisted daily-loss/drawdown/consecutive-loss/vol/spread/correlation/event/rate-limit/market-hours gates) is exactly this.
- **Which model:** None — deterministic rules, regime-conditional thresholds (upgrade: thresholds scale with regime, Section 9.3), correlation guard activated with real DXY (Section 13.3).
- **What learns:** Nothing (config-driven; state persisted; restarts cannot bypass).
- **Inputs/Outputs:** (order intent, live market data, account state) → (approve/reject + reason; halt states).
- **Training mode:** n/a.
- **Neighbors:** ← DECISION; ← DATA (live prices); → EXECUTION (approved orders); → OUTCOME (realized P&L updates risk state).

### Block 8 — EXECUTION

- **Why:** Realistic order submission with idempotency, lot-step flooring, aggregate caps, promotion-gate enforcement (all existing in `live/trade_executor.py`, `live/live_trade_mt5.py`).
- **Which model:** None — deterministic broker adapter (MT5 with retcode handling; MockBroker for demo/shadow).
- **What learns:** Nothing.
- **Inputs/Outputs:** (approved order intent) → (broker orders with idempotency tokens; fills; rejections).
- **Training mode:** n/a.
- **Neighbors:** ← RISK; → MT5 (orders); → OUTCOME (fills).

### Block 9 — MT5 (the world)

- **Why:** The live/demo market interface. Project: `Mt5BarSource` (closed bars only), candle-close-aligned loop (`on_closed_bar` — one decision per NEW closed bar), reconciliation, kill switch.
- **Which model:** None.
- **Inputs/Outputs:** (orders) → (fills, prices, account equity).
- **Neighbors:** ← EXECUTION; → OUTCOME (fills, equity curves).

### Block 10 — OUTCOME

- **Why:** Realized results are the ground truth for learning and audit (no fabrication — project stance).
- **Which model:** None — accounting layer: realized P&L per trade, per-bar mark-to-market, execution quality (spread paid, slippage).
- **What learns:** Nothing.
- **Inputs/Outputs:** (fills, prices) → (trade outcomes, equity path, cost breakdown).
- **Neighbors:** ← MT5; → REWARD (objective components); → MEMORY (trade log); → VALIDATION (evaluation data).

### Block 11 — REWARD

- **Why:** The reward *is* the learned objective (Section 11); risk-aware composite evidence (arXiv 2506.04358; TorchTrade; Pro-Trader-RL; RegimeRL).
- **Which model:** Deterministic function (no learned reward in phase 1 — reward learning is an experiment).
- **What learns:** Nothing (weights validated by grid on purged paths, Section 11.4).
- **Inputs/Outputs:** (trade outcomes, equity path, costs) → (per-segment composite R: Sharpe-contribution + differential-vs-buy-and-hold − downside − costs − drawdown penalty).
- **Neighbors:** ← OUTCOME; → TRAINING (RL objective); → POLICY/RL (via critic targets).

### Block 12 — MEMORY

- **Why:** A long-lived autonomous system needs layered memory (Section 8; FinMem arXiv 2311.13743 design pattern; RAG).
- **Which model:** Stratified experience DB (regime + recency indexes) for offline training; regime/macro representation store for retrieval; checkpoints of parameters; audit logs (existing fills/idempotency/risk_state).
- **What learns:** Nothing directly (storage/retrieval); it serves TRAINING and the regime module.
- **Inputs/Outputs:** (trajectories, outcomes, embeddings, checkpoints) → (training batches, regime statistics, retrievable context).
- **Neighbors:** ← OUTCOME; ← TRAINING; ← REPRESENTATION (embeddings); → TRAINING (batches); → REGIME (stats).

### Block 13 — TRAINING

- **Why:** The offline training engine that converts memory + reward into updated parameters (Section 15).
- **Which model:** The training job (self-supervised representation pretraining → world-model training → offline CQL/DT phase → imagination RL fine-tuning), with EWC + replay stratification (Section 7.4).
- **What learns:** All learned parameters (encoder, world model, actor, critic).
- **Inputs/Outputs:** (memory batches, reward, world-model rollouts) → (candidate checkpoints + evaluation bundle).
- **Training mode:** Offline, batched, scheduled (not live).
- **Neighbors:** ← MEMORY; ← REWARD; → VALIDATION (candidates).

### Block 14 — VALIDATION

- **Why:** The honesty gate (Section 16): purged/embargoed walk-forward (existing `backtest/engine.py::walk_forward`), PSR/DSR deflated significance, buy-and-hold comparison, per-regime stability, calibration checks (confidence, vol forecast).
- **Which model:** The honest backtester (kernc/backtesting.py + CostModel) — the *only* promotion authority (Section 12.3).
- **What learns:** Nothing (evaluation only).
- **Inputs/Outputs:** (candidate checkpoints + evaluation bundle) → (promotion verdict + metrics bundle).
- **Neighbors:** ← TRAINING; → SAFE UPDATE (verdict); ← OUTCOME (OOS shadow data).

### Block 15 — SAFE UPDATE

- **Why:** Closed-loop self-improvement with safety (Section 7.6): champion/challenger, significance-gated promotion through the existing gate (`enforce_model_promotion_gate`), auto-rollback on live degradation, RiskSupervisor as the emergency backstop.
- **Which model:** Orchestration logic.
- **What learns:** Nothing.
- **Inputs/Outputs:** (validation verdict, live monitoring) → (promote/rollback/hold; checkpoint activation).
- **Neighbors:** ← VALIDATION; → POLICY/RL (active checkpoint); → MEMORY (champion archive).

## 20.3 Information-flow invariants

**[ENGINEERING RECOMMENDATION]** The architecture is governed by four invariants (single evidence-based rules):
1. **No order authority outside the EXECUTION block.** LLM research desk, memory, and validation produce inputs/verdicts — only RISK-approved, EXECUTION-issued orders touch MT5.
2. **No learned layer between RISK and EXECUTION.** Safety is deterministic; it cannot be gradient-optimized away.
3. **No promotion without the honest backtester + deflated significance.** The simulator/world model trains; the backtester validates; the gate promotes (Section 12.3, 16.4).
4. **No live parameter SGD in phase 1.** All learning is offline; live trading consumes frozen checkpoints under the safe-update protocol (Section 7.6).

## 20.4 Mapping the blocks to the existing codebase

| Block | Existing code (works) | Upgrade needed |
|---|---|---|
| DATA | `core/feature_pipeline.py`, `backtest/engine.py::prepare_ohlc`, `Mt5BarSource` | Real calendar feed; DXY/yields panel; partial fills (Section 12.4) |
| REPRESENTATION | higher-TF features in pipeline; `TransformerActor` encoder | Self-supervised patched encoder; raw-OHLCV patching; fusion network |
| MARKET MODEL | `models/dreamer_agent.py` dynamics | Vol/regime auxiliary heads; LoRA-adapted FM encoder option |
| WORLD MODEL | `models/dreamer_agent.py` RSSM (faithful) | Representation-fed observations; regime conditioning; Mamba backbone challenger |
| POLICY/RL | `models/dreamer_agent.py` actor/critic, PPO, Transformer-PPO, `models/ensemble.py` | Plan action space (direction+horizon); risk-aware critic; CQL/DT offline phases; EWC |
| DECISION | `DecisionEngine`, `ATRPositionSizer`, `KellyPositionSizer`, `entry_sl_tp` | Model-vol sizing; horizon-class consumption |
| RISK | `RiskSupervisor` (SQLite-persisted) | Regime-conditional thresholds; DXY-activated correlation guard |
| EXECUTION | `TradeExecutor`, `live_trade_mt5.py` | (mostly complete) |
| MT5 | `Mt5Broker`/`MockBroker`, candle-close loop | — |
| OUTCOME | broker fills, `update_risk_after_close` | Execution-quality accounting |
| REWARD | scaled log-return in env | Risk-aware composite (Section 11.4) |
| MEMORY | training-only `ReplayBuffer` | Stratified experience DB; regime/macro store; RAG hooks |
| TRAINING | `train/` scripts, MAML, adversarial training | Full pipeline (Section 15.3) |
| VALIDATION | `backtest/engine.py::walk_forward`, `evaluation.json` gate | PSR/DSR; CPCV; calibration; per-regime |
| SAFE UPDATE | promotion gate, risk-state persistence, rollback-capable checkpoints | Significance gate; auto-rollback monitor |

---

# Section 21 — The 14 Final Decisions

Each decision has exactly one evidence-based answer, its rationale (with citation markers), and its implication for the `autonoumuse_trader` codebase.

### Decision 1 — One unified model or multiple models?
**Answer: One integrated system with multiple *specialized blocks* sharing one representation — NOT one monolithic model and NOT parallel independent models.**
- Rationale: Section 2 (architecture comparison): monoliths lack safety/validation; parallel systems duplicate learning. The shared-backbone multi-task pattern (PatchTST/iTransformer evidence) + world-model RL (DreamerV3 evidence) support blocks sharing a representation with distinct roles. [ENGINEERING RECOMMENDATION; RESEARCH FINDING: arXiv 2301.04104, arXiv 2211.14730, arXiv 2310.06625]
- Project implication: keep `models/dreamer_agent.py`'s shared RSSM latent; do not add parallel competing models.

### Decision 2 — Multiple models or hybrid?
**Answer: Hybrid (world model + policy + representation + deterministic shell + LLM advisory).**
- Rationale: Section 2.8; each block does one job; safety/audit preserved; this is the evidence-backed synthesis of model-based RL + FM representation + deterministic risk. [ENGINEERING RECOMMENDATION]
- Project implication: the existing layering (features → env → models → risk → live) is already the hybrid skeleton; the report fills missing blocks rather than restructuring.

### Decision 3 — Main brain?
**Answer: DreamerV3-style world model + actor-critic is the main brain (decision authority).**
- Rationale: strongest generalist decision-learning evidence (Nature 2025; 150+ tasks, arXiv 2301.04104); safe exploration; project already implements it faithfully. [RESEARCH FINDING: arXiv 2301.04104; FACT: `models/dreamer_agent.py`]
- Project implication: `models/dreamer_agent.py` stays; upgrades target observations, reward, action space, memory — not a brain replacement.

### Decision 4 — Forecasting + decisions unified or separate?
**Answer: Separate roles, shared representation — policy decides; auxiliary forecast heads (volatility, regime) feed risk; no unconditional next-candle forecast in the decision path.**
- Rationale: Section 3 (prediction vs decision); forecast accuracy ≠ profitability (Lopez de Prado, SSRN 2460551); volatility forecasts are actionable (Kronos −9% MAE, arXiv 2508.02739). [ENGINEERING RECOMMENDATION; RESEARCH FINDING: SSRN 2460551, arXiv 2508.02739]
- Project implication: no new forecasting model as decision driver; add vol/regime sidecar heads to the market model.

### Decision 5 — RL inside the model or separate policy?
**Answer: RL as the decision layer (policy + critic) inside the world-model system — the RL core is separate from the LLM and the deterministic shell.**
- Rationale: Sections 2.7, 6.4, 18.3. [ENGINEERING RECOMMENDATION]
- Project implication: keep PPO/Dreamer policies as the decision core; LLM research desk never becomes the policy.

### Decision 6 — World model: yes or no?
**Answer: Yes — non-negotiable for this project's safety profile.**
- Rationale: safe exploration (exploration = real P&L otherwise), what-if/counterfactual audit (arXiv 2607.16999), credit assignment, training scale; DreamerV3 evidence. [RESEARCH FINDING: arXiv 2301.04104, arXiv 2607.16999]
- Project implication: keep + upgrade the RSSM; never replace the honest backtester for promotion.

### Decision 7 — Foundation model: use existing or own training?
**Answer: Use existing pretrained FMs where they give priors (Kronos-style market encoder if available; TSFMs as sidecar baselines); train from scratch only for what is XAUUSD-specific (representation + world model on project data).**
- Rationale: FM pretraining at Kronos scale (12B records) is not reproducible by this project; LoRA adaptation (arXiv 2106.09685) makes FM reuse cheap. [RESEARCH FINDING: arXiv 2508.02739, arXiv 2106.09685]
- Project implication: no 12B-record pretraining; adopt/adapt where justified; PatchTST-from-scratch as the default representation encoder.

### Decision 8 — What to pretrain vs XAUUSD-specific?
**Answer: Self-supervised pretrain the representation on XAUUSD + macro (TF-C/masked-reconstruction style); train the world model and policy specifically on XAUUSD (with regime tasks for MAML).**
- Rationale: self-supervised TS evidence (arXiv 2206.08496, arXiv 2403.09809); MAML regime adaptation for non-stationarity (existing `models/meta_learning.py`). [RESEARCH FINDING: arXiv 2206.08496, arXiv 2403.09809]
- Project implication: new self-supervised pretraining job on the causal pipeline's output; keep MAML regime tasks.

### Decision 9 — Self-learning: yes or no?
**Answer: Yes — a *gated, offline, risk-aware* self-improvement loop (champion/challenger, significance-gated promotion, EWC + replay, auto-rollback); no live SGD in phase 1.**
- Rationale: Section 7; continual-learning safety evidence (arXiv 2403.05175, EWC arXiv 1612.00796); DRL-trading overfitting warnings. [ENGINEERING RECOMMENDATION; RESEARCH FINDING: arXiv 2403.05175, arXiv 1612.00796]
- Project implication: build the experience DB, offline training job, significance gate, rollback monitor; extend the existing promotion gate.

### Decision 10 — Memory: what kind?
**Answer: Layered memory — context window + RSSM latent (short), stratified experience DB (mid), regime/macro representation store (long) + RAG for the LLM desk; all as inputs to training/risk, never order sources.**
- Rationale: Section 8; FinMem layered-memory pattern (arXiv 2311.13743); RAG (FinGPT ecosystem). [ENGINEERING RECOMMENDATION; RESEARCH FINDING: arXiv 2311.13743]
- Project implication: extend `ReplayBuffer` → persistent stratified store; persist regime labels → retrievable regime records.

### Decision 11 — Live learning safety: how?
**Answer: Three-tier update cadence (world-model maintenance offline; policy fine-tuning offline per N trades with significance gate; no intraday parameter changes) + deterministic RiskSupervisor backstop + auto-rollback.**
- Rationale: Section 7.6; production continual-learning literature. [ENGINEERING RECOMMENDATION]
- Project implication: enforce the cadence in ops; keep risk-state persistence and kill switch as-is.

### Decision 12 — Best path to production autonomy?
**Answer: Phased: (1) honest data + backtest baseline (done — strategies do not beat buy-and-hold; that is the starting truth); (2) self-supervised representation + world model on XAUUSD; (3) offline CQL/DT pretraining + imagination RL fine-tuning with risk-aware reward; (4) shadow/demo on real closed bars (existing MockBroker path); (5) significance-gated LIVE promotion with auto-rollback.**
- Rationale: Sections 15, 16, 20.3; the project's honest evaluation protocol is the gate at every phase. [ENGINEERING RECOMMENDATION]
- Project implication: sequencing exists partially (smoke/demo/live gates); add phases 2–3 as new capabilities under the existing gate.

### Decision 13 — What does "self-improving" mean operationally?
**Answer: Continuously better calibrated and no-worse-than-incumbent risk-aware decisions — NOT guaranteed net-of-cost alpha. Alpha is unproven; the loop's defensible value is risk/execution quality and non-degradation.**
- Rationale: project's own honest findings (strategies do not beat buy-and-hold net of costs; ML path refuses to fabricate); market-efficiency literature (Springer 10.1007/s10614-025-11168-9; MDPI 2079-9292/14/9/1721; arXiv 2501.07489). [FACT: project findings; RESEARCH FINDING: efficiency literature]
- Project implication: keep the no-fabrication stance; define self-improvement metrics around calibration, risk-state quality, and beating-the-incumbent — not around inflated P&L claims.

### Decision 14 — Correcting the project's wrong assumptions (the meta-decision)
**Answer: The project's honest failures are its greatest assets: the fake backtester is archived as evidence; the rule strategies' buy-and-hold loss is the baseline truth; the unshipped ML checkpoint is integrity. The final architecture preserves all three as hard gates (honest evaluation only, buy-and-hold benchmark, no fabrication).**
- Rationale: Sections 16.1, 16.5; the study's mandate to be honest about the project's own findings. [FACT: project artifacts/README]
- Project implication: institutionalize the honest backtester as the only promotion authority, the buy-and-hold comparison as mandatory, and the no-fabrication rule as a test/ops invariant.

---

## 21.x The README pytest discrepancy (verbatim, unresolved by this study)

**[FACT]** The project README reports **"87 passed"** for the test suite, while `artifacts/pytest_final.txt` records **139 passed**. This study reports the discrepancy verbatim and does not silently pick one count. It is consistent with the suite having grown after the README number was written (or the README number predating the final suite). [ENGINEERING RECOMMENDATION] Re-run the suite, reconcile the README count, and keep a single source of truth for test results as part of the audit trail. This does not affect the strategy findings, which come from the honest backtester (Section 16.5).

# Open Research Questions

This section lists the major unanswered questions that the study surfaced, each with the evidence that would resolve it. They are deliberately NOT answered by this report — answering them requires the proposed experiments (next section) or external research. Each question is marked with the claim-status it currently holds (UNPROVEN/HYPOTHETICAL) and the falsifiable evidence that would move it to a different status.

## Q1 — Can any ML-based XAUUSD strategy beat buy-and-hold net of costs, out-of-sample, with deflated significance?
- **Current status:** UNPROVEN/HYPOTHETICAL. The project's own honest backtests (rule strategies do not beat buy-and-hold net of costs) and the market-efficiency literature (Springer 10.1007/s10614-025-11168-9; MDPI 2079-9292/14/9/1721; arXiv 2501.07489) cut against it.
- **Evidence that would resolve it:** A challenger policy that beats buy-and-hold AND the incumbent on purged/embargoed OOS windows with PSR/DSR-deflated significance (SSRN 2460551) across ≥2 full market regimes, replicated on live-shadow (MockBroker) data. Absence of such evidence after the full experiment program (E1–E10) would resolve it as "no — within current methods and cost models".

## Q2 — Does the world model (RSSM) learn exploitable short-horizon XAUUSD dynamics, or only noise?
- **Current status:** UNPROVEN/HYPOTHETICAL. The world model's *defensible* value (safe RL, what-if, credit assignment) is established; its *alpha* value is not.
- **Evidence that would resolve it:** Measure world-model reconstruction/rollout loss vs a no-memory baseline on held-out windows; measure whether imagination-trained policies beat replay-trained policies net of costs (E4). If rollout loss is not better than persistence and imagination-trained policies never beat offline baselines, the world model is a safety instrument, not an alpha source.

## Q3 — Is a learned multi-scale representation (patched encoder) materially better than the hand-crafted feature pipeline?
- **Current status:** UNPROVEN/HYPOTHETICAL. Self-supervised TS pretraining is well-supported generally (arXiv 2206.08496; arXiv 2403.09809; S0950705124012863), but no project-specific evidence exists for XAUUSD.
- **Evidence that would resolve it:** A/B experiment (E2): same world model + policy, differing only in representation (feature pipeline vs learned patched encoder vs hybrid), evaluated by the honest backtester. If the learned encoder does not beat the leak-free feature pipeline net of costs, keep the pipeline (cheaper, auditable) and use the encoder only where it wins.

## Q4 — Does regime-conditioned dynamics (per-regime experts) improve stability vs a single world model?
- **Current status:** UNPROVEN/HYPOTHETICAL. Regime-aware RL is a documented direction (RegimeRL github sahilapage; HMM literature), and MAML regime tasks exist in the project (`models/meta_learning.py`), but the marginal benefit of conditioning the *world model* on regime id is unmeasured.
- **Evidence that would resolve it:** E7 (per-regime world models / regime-conditioned RSSM vs single model) measured by per-regime OOS loss and per-regime trading P&L stability. Resolution: adopt conditioning only if per-regime stability improves without hurting cross-regime generalization.

## Q5 — Is the risk-aware composite reward (arXiv 2506.04358 style) worth its tuning complexity vs a simple cost-aware return?
- **Current status:** UNPROVEN/HYPOTHETICAL for the project's XAUUSD setting (the paper's design is evidence; its benefit on this asset/setup is not).
- **Evidence that would resolve it:** E5 (reward ablation): same pipeline, reward variants (log-return vs composite vs composite+drawdown), compared on Sharpe/Sortino/MaxDD net of costs with deflated significance. Kill criterion: if the composite never improves net-of-cost risk-adjusted return, simplify back to cost-aware log-return + deterministic risk shell.

## Q6 — Does offline RL pretraining (CQL/DT) improve final policies vs training from scratch in imagination?
- **Current status:** UNPROVEN/HYPOTHETICAL for trading (offline RL is well-established in general control: arXiv 2006.04779, arXiv 2106.01345; its benefit in FX is untested here).
- **Evidence that would resolve it:** E6 (offline-pretrain ablation): same pipeline with/without CQL-conservative and DT-return-conditioned phases, measured on OOS net-of-cost performance and sample efficiency (number of env steps to reach incumbent parity).

## Q7 — Do LLM-based research-desk features (sentiment/event flags) add value to the numeric core, and at what cost?
- **Current status:** UNPROVEN/HYPOTHETICAL. FinGPT/FinMem show LLM finance adaptation works textually (arXiv 2306.06031; arXiv 2311.13743); whether their outputs improve an RL trading policy's net-of-cost P&L is untested.
- **Evidence that would resolve it:** E8 (LLM desk ablation): numeric core with vs without LLM event/sentiment features, evaluated by the honest backtester and by feature-attribution (does the policy actually use them?). Kill criterion: no net-of-cost improvement and no calibration improvement → drop the LLM desk features.

## Q8 — Does live self-improvement (gated offline updates) maintain or improve performance without degrading in regime shifts?
- **Current status:** UNPROVEN/HYPOTHETICAL. Continual-learning safety is literature-supported (arXiv 2403.05175; EWC arXiv 1612.00796), but the closed-loop trading behavior is unmeasured.
- **Evidence that would resolve it:** E9/E10 (shadow self-improvement): challenger policies promoted through the significance gate, monitored on live-shadow; measure non-degradation (rollback triggers), per-regime stability, and calibration over ≥3 months of shadow data. Resolution: self-improvement is production-safe only if the non-degradation rate (monitored by auto-rollback) stays at the designed level.

## Q9 — Does CPCV (combinatorial purged cross-validation) meaningfully change promotion decisions vs the current walk-forward?
- **Current status:** UNPROVEN/HYPOTHETICAL for this project. Lopez de Prado's methods are well-established (SSRN 2460551; Neyt/How-To-Backtest-Correctly), but the project's walk-forward has not been compared to CPCV on the same strategies.
- **Evidence that would resolve it:** E3 (evaluation-methodology comparison): run both protocols on the same candidate set; if CPCV changes which candidates pass DSR significance, adopt CPCV; otherwise document the walk-forward as sufficient given fixed-parameter honesty.

## Q10 — Is there exploitable cross-asset signal (DXY/yields/VIX) for XAUUSD that survives costs?
- **Current status:** UNPROVEN/HYPOTHETICAL. The correlation guard exists but is inert (DXY placeholder 0.0 in `trade_executor.build_market_data`); no project data exists.
- **Evidence that would resolve it:** E1 (data-panel experiment): add real DXY/yields/VIX panel with `shift(1)` lag; test whether the representation + world model improve OOS net-of-cost performance. Kill criterion: no improvement net of the added data complexity → keep single-asset XAUUSD and remove the inert guard's misleading presence.

## Q11 — Is a Mamba/SSM backbone better than a patched Transformer for the market model at this data scale?
- **Current status:** UNPROVEN/HYPOTHETICAL. Mamba is competitive for long-term TSF (arXiv 2403.11144; MambaTS arXiv 2405.16440), but not at XAUUSD scale with the project's context lengths.
- **Evidence that would resolve it:** E2 variant (backbone ablation): patched-Transformer vs Mamba encoder, same everything else, measured by world-model loss and net-of-cost OOS P&L. Adopt the winner; expect the difference to be small at this scale.

---

# Proposed Experiments

Experiments are ordered cheap-to-expensive. Each has: hypothesis, method, success metric, kill criteria. All use the project's honest protocol as the gate: purged/embargoed walk-forward (`backtest/engine.py::walk_forward` train 800/embargo 25/test 300), fixed costs (CostModel one-way 0.000175, round-trip 0.000410), comparison vs buy-and-hold and vs incumbent, no tuning on the test path, PSR/DSR-style deflated significance on promotion (SSRN 2460551). Nothing is promoted without `evaluation.json passed==true` + contract-hash match (existing `enforce_model_promotion_gate`).

## E1 — Cross-asset data panel (cheapest, data-only)
- **Hypothesis:** Adding DXY/10Y/VIX (H1/D1, `shift(1)`-lagged) to the feature panel improves XAUUSD OOS net-of-cost performance of the incumbent pipeline.
- **Method:** Extend `core/feature_pipeline.py` with the macro panel (activate the inert correlation guard in `RiskSupervisor` and `build_market_data`); run the honest walk-forward on rule baselines and a fixed PPO policy before/after; report per-window dispersion.
- **Success metric:** Mean OOS net-of-cost return improves vs the no-panel control with non-overlapping window CIs; correlation guard fires on real data (no longer placeholder).
- **Kill criteria:** No net-of-cost improvement after costs; guard fires spuriously on noise; data quality cannot be maintained (gap/latency).

## E2 — Learned representation vs feature pipeline (cheap; self-supervised pretraining)
- **Hypothesis:** A self-supervised patched-encoder representation (masked reconstruction, PatchTST-style, arXiv 2211.14730; optionally TF-C contrastive, arXiv 2206.08496) over raw OHLCV beats the hand-crafted feature pipeline for the same world model + policy.
- **Method:** Train the encoder on XAUUSD+M5–D1 (no labels, no leakage); plug into `models/dreamer_agent.py` observations (hybrid: learned tokens + selected features); A/B vs pipeline-only with identical policy/training budget.
- **Success metric:** Better world-model rollout loss on held-out windows AND net-of-cost OOS P&L (deflated significance).
- **Kill criteria:** No net-of-cost gain; encoder overfits (train/test gap large); rollout loss worse than pipeline features.

## E3 — Evaluation-methodology comparison: walk-forward vs CPCV (cheap, no training)
- **Hypothesis:** CPCV (SSRN 2460551; Neyt/How-To-Backtest-Correctly) changes which candidates pass deflated-significance promotion vs the current single-path walk-forward.
- **Method:** Run both protocols on the same fixed candidate set (rule strategies + a few trained policies); compute PBO/PSR/DSR per protocol.
- **Success metric:** A measurable difference in promotion decisions; lower PBO with CPCV.
- **Kill criteria:** Results identical (document walk-forward as sufficient given fixed parameters); implementation cost exceeds benefit.

## E4 — World-model value: imagination-trained vs replay-trained (medium; training)
- **Hypothesis:** Imagination-based actor-critic (DreamerV3, arXiv 2301.04104) trains sample-efficiently and achieves ≥ replay/PPO-only net-of-cost performance at lower environment-interaction count.
- **Method:** Same observation/reward/action space; compare (a) DreamerV3 imagination, (b) PPO on replay, (c) Dreamer-imagination with costs in dreams (Section 6.3) — fixed total gradient budget.
- **Success metric:** Net-of-cost OOS performance parity-or-better at ≤50% env steps; imagination with costs beats without costs (validates cost-in-dream requirement).
- **Kill criteria:** Imagination never reaches replay/PPO parity; cost-in-dream harms training stability.

## E5 — Reward ablation (medium; training)
- **Hypothesis:** The risk-aware composite reward (arXiv 2506.04358 pattern: return + downside + differential-vs-baseline + costs + drawdown penalty) improves net-of-cost Sharpe/Sortino/MaxDD vs scaled log-return.
- **Method:** Grid the composite weights on a small validation slice (never the test path); train identical pipelines per reward; evaluate by honest backtester.
- **Success metric:** Sortino + MaxDD improve with deflated significance; drawdown-penalized variant stays within designed drawdown band.
- **Kill criteria:** Composite never beats simple reward net of costs; weight sensitivity explodes (any small weight change flips results).

## E6 — Offline pretraining phase (CQL/DT) (medium; training)
- **Hypothesis:** CQL (arXiv 2006.04779) conservative pretraining and/or DT (arXiv 2106.01345) return-conditioned seeding improve final policies and sample efficiency vs from-scratch imagination training.
- **Method:** Build the stratified experience DB (Section 8) from historical + simulated data; pretrain policy (CQL/DT), then fine-tune in imagination; compare vs from-scratch.
- **Success metric:** Fewer env steps to reach incumbent parity; better OOS net-of-cost performance with deflated significance.
- **Kill criteria:** Pretrained policies underperform from-scratch; conservative estimates collapse value information; dataset coverage too thin (then expand simulator coverage or kill).

## E7 — Regime-conditioned world model (medium-expensive; training)
- **Hypothesis:** Conditioning the RSSM on regime id (or per-regime experts, MoE-style) improves per-regime stability without hurting cross-regime generalization.
- **Method:** Use `MarketRegimeGenerator` labels (or HMM-smoothed); train (a) single world model, (b) regime-conditioned (regime id input), (c) per-regime experts; evaluate per-regime OOS loss and P&L stability.
- **Success metric:** Per-regime OOS metrics improve and cross-regime generalization does not degrade; forgetting is reduced (Section 7.4).
- **Kill criteria:** No per-regime improvement; expert fragmentation (regimes with insufficient data); added complexity without measurable stability gain.

## E8 — LLM research-desk ablation (expensive; external API + integration)
- **Hypothesis:** LLM event/sentiment features (FinGPT/FinMem-class, arXiv 2306.06031, arXiv 2311.13743) add net value to the numeric core.
- **Method:** Wire the research desk (RAG over news/calendar) as advisory features; A/B the numeric core with/without them; monitor policy usage (feature attribution) and drift.
- **Success metric:** Net-of-cost OOS improvement or calibration improvement with deflated significance; desk latency/error rate within budget.
- **Kill criteria:** No improvement; the policy ignores the features (attribution ≈ 0); desk error/drift unacceptable (then drop to event-flag-only).

## E9 — Shadow self-improvement loop (expensive; ops)
- **Hypothesis:** Gated offline self-improvement (champion/challenger + significance promotion + EWC + auto-rollback, Section 7.6) maintains performance and improves calibration without degradation.
- **Method:** Run the loop in shadow (MockBroker on real closed bars) for ≥3 months; challenger candidates trained offline from the experience DB; promote only through the extended gate.
- **Success metric:** Non-degradation (incumbent parity maintained; rollback triggers within design); calibration of confidence/vol improves; promotion rate matches expectation.
- **Kill criteria:** Rollback storms; degradation in any regime; experience DB poison (feedback-loop contamination); gate bypasses.

## E10 — Live-safe promotion trial (most expensive; capital at risk — deferred gate)
- **Hypothesis:** A candidate that passes E1–E9 gates generalizes to small live/demo capital within RiskSupervisor limits.
- **Method:** Small demo (TRADING_MODE=demo, MockBroker/Mt5Broker demo), then minimum-live under RiskSupervisor, candle-close cadence, idempotent execution, kill switch armed; pre-registered success thresholds.
- **Success metric:** Net-of-cost P&L and risk metrics within pre-registered bounds vs shadow expectation; zero safety violations (breaker bypass, idempotency failure, reconcile drift).
- **Kill criteria:** Any safety violation; out-of-band risk metrics; pre-registered performance thresholds missed → rollback to shadow and re-investigate.

---

# References

References are grouped by report section. All were verified via `internet_search`/`parse_urls` during this study unless marked as project artifacts (FACT source: project files). No URLs are invented beyond what the research corpus provided.

## Sections 1–3 (landscape, architectures, prediction vs decision)
- DreamerV3 — arXiv 2301.04104: https://arxiv.org/abs/2301.04104
- Decision Transformer — arXiv 2106.01345: https://arxiv.org/abs/2106.01345
- CQL (Conservative Q-Learning) — arXiv 2006.04779: https://arxiv.org/pdf/2006.04779
- Kronos — arXiv 2508.02739: https://arxiv.org/abs/2508.02739 (also https://arxiv.org/html/2508.02739); repo https://github.com/shiyu-coder/Kronos
- FinGPT — arXiv 2306.06031
- FinMem — arXiv 2311.13743; repo https://github.com/pipiku915/FinMem-LLM-StockTrading
- FinRobot — https://github.com/AI4Finance-Foundation/FinRobot
- TradingAgents — arXiv 2412.20138
- QuantAgent — https://github.com/THU-MIG/QuantAgent
- DRL trading survey (167 papers) — ResearchGate publication 356833146 "Algorithmic Trading and Reinforcement Learning: Robust methodologies for AI in finance"
- RL execution critique — arXiv 2307.11685
- RL execution benchmark — ScienceDirect S0927538X25002136: https://www.sciencedirect.com/science/article/pii/S0927538X25002136
- RL trading framework — arXiv 2411.07585: https://arxiv.org/html/2411.07585v1
- Lopez de Prado, Deflated Sharpe Ratio — SSRN 2460551: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551; PDF https://www.pm-research.com/content/iijpormgmt/40/5/94.full.pdf

## Sections 4–5 (brains, foundation models)
- Kronos — arXiv 2508.02739 (as above); BSQ https://github.com/zhaoyue-zephyrus/bsq-vit
- TimesFM 3.0 — https://github.com/google-research/timesfm; HF https://huggingface.co/google/timesfm-3.0-pytorch
- TimeGPT — Nixtla docs https://www.nixtla.io/docs/about-timegpt; Azure AI catalog TimeGPT-1
- Chronos — arXiv 2403.07815; Chronos-2 — arXiv 2510.15821; HF https://huggingface.co/amazon/chronos-2; https://github.com/amazon-science/chronos-forecasting
- Moirai 2.0 — arXiv 2511.11698; HF https://huggingface.co/Salesforce/moirai-2.0-R-small
- PatchTST — arXiv 2211.14730
- iTransformer — arXiv 2310.06625
- TFT — arXiv 1912.09363
- GIFT-Eval — arXiv 2410.10393; leaderboard https://tsfm.ai/benchmarks/gift-eval; repo https://github.com/SalesforceAIResearch/gift-eval
- Mamba for TSF — arXiv 2403.11144; MambaTS — arXiv 2405.16440
- MoE surveys — arXiv 2407.06204; arXiv 2602.08019

## Sections 6–8 (world models, self-learning, memory)
- DreamerV3 — arXiv 2301.04104; Nature 2025 publication
- TransDreamer — arXiv 2209.14153
- 251-dreamer-trading — https://github.com/suenot/251-dreamer-trading
- CQL — arXiv 2006.04779 (as above)
- Decision Transformer — arXiv 2106.01345 (as above)
- Continual learning survey — arXiv 2403.05175
- EWC — arXiv 1612.00796
- Counterfactual Shapley credit assignment — arXiv 2607.16999
- FinMem layered memory — arXiv 2311.13743 (as above)
- RAG/FinGPT — arXiv 2306.06031 (as above)
- DRL overfitting/non-stationarity — DRL survey (ResearchGate 356833146, as above)

## Sections 9–12 (regime, actions, reward, environment)
- RegimeRL — https://github.com/sahilapage (RegimeRL repo)
- TradingAgents — arXiv 2412.20138 (as above)
- QuantAgent — https://github.com/THU-MIG/QuantAgent (as above)
- Risk-aware reward — arXiv 2506.04358
- TorchTrade — https://github.com/ai4finance/torchtrade
- Pro-Trader-RL — https://github.com/comach/Pro-Trader-RL
- Multi-agent LOB RL — arXiv 2006.05574
- JAX-LOB — arXiv 2308.13289; repo https://github.com/KangOxford/AlphaTrade
- TRADES (LOB generation) — arXiv 2502.07071
- Lopez de Prado backtest methodology — SSRN 2460551 (as above); purged k-fold https://quantstrategy.io/blog/purged-k-fold-cross-validation-the-gold-standard-for/; How-To-Backtest-Correctly https://github.com/Neyt/How-To-Backtest-Correctly

## Sections 13–15 (data, multi-TF, training)
- HiMTM — arXiv 2401.05012
- Multi-scale dilated conv — arXiv 2405.05499
- PatchTST — arXiv 2211.14730 (as above)
- iTransformer — arXiv 2310.06625 (as above)
- TF-C self-supervised contrastive — arXiv 2206.08496; repo https://github.com/mims-harvard/TFC-pretraining
- Contrastive vs generative TS SSL — arXiv 2403.09809
- Self-supervised TS forecasting — ScienceDirect S0950705124012863: https://www.sciencedirect.com/science/article/pii/S0950705124012863
- LoRA — arXiv 2106.09685
- Kronos pretraining scale — arXiv 2508.02739 (as above)
- CQL — arXiv 2006.04779 (as above)
- Decision Transformer — arXiv 2106.01345 (as above)
- GRPO — arXiv 2402.03300; theory/demystification arXiv 2603.01162

## Sections 16–19 (evaluation, causality, frontier AI, new architectures)
- Lopez de Prado: DSR/PSR/PBO/CPCV — SSRN 2460551 (as above); purged k-fold (quantstrategy.io, as above); How-To-Backtest-Correctly (github Neyt, as above); CPCV lab (colab fin510 lab10_backtesting.ipynb: https://colab.research.google.com/github/quinfer/fin510-colab-notebooks/blob/main/labs/lab10_backtesting.ipynb)
- GIFT-Eval — arXiv 2410.10393 (as above)
- Market efficiency / ML predictability — Springer 10.1007/s10614-025-11168-9: https://link.springer.com/article/10.1007/s10614-025-11168-9; MDPI 2079-9292/14/9/1721: https://www.mdpi.com/2079-9292/14/9/1721; arXiv 2501.07489: https://arxiv.org/html/2501.07489
- Counterfactual Shapley — arXiv 2607.16999 (as above)
- RL for LLM post-training survey — arXiv 2407.16216
- LLM post-training deep dive — arXiv 2502.21321
- Frontier labs training guide — https://jxzhangjhu.github.io/blog/2026/how-frontier-labs-train-llms/
- How Modern LLM Systems Really Work — https://medium.com/@suyog19/how-modern-llm-systems-really-work-6b377222eb7c and https://suyogjoshi.com/writing/how-modern-llm-systems-really-work/
- LLM-agent planning survey — arXiv 2402.02716
- GRPO — arXiv 2402.03300; arXiv 2603.01162 (as above)
- Mamba for TSF — arXiv 2403.11144; MambaTS — arXiv 2405.16440 (as above)
- MoE surveys — arXiv 2407.06204; arXiv 2602.08019 (as above)
- DreamerV3 — arXiv 2301.04104 (as above)
- Kronos — arXiv 2508.02739 (as above)
- TRADES — arXiv 2502.07071 (as above)

## Project artifacts and internal sources (FACT grounding)
- Project README (honest verdicts; "87 passed" vs artifacts/pytest_final.txt "139 passed" discrepancy)
- `core/feature_pipeline.py` (leak-free causal pipeline, feature_contract.json)
- `core/config.py` (frozen dataclasses; TradingBehaviorConfig min_ensemble_agreement)
- `core/observation.py` (AccountState 5-dim vector)
- `env/dreamer_trading_env.py` (RealisticTradingEnv: spread/commission/slippage/swap/SL-TP/max-DD episode breaker)
- `models/dreamer_agent.py`, `models/dreamer_components.py` (RSSM, symlog, two-hot, free_nats, imagination, ReplayBuffer)
- `models/transformer_policy.py` (Transformer-PPO, GAE, positional encoding, attention introspection)
- `models/policy.py` (PpoPolicy, DreamerPolicy, TransformerPolicy, DreamerMCTSPolicy)
- `models/ensemble.py` (soft/hard voting, consensus gate, epistemic KL)
- `models/meta_learning.py` (MarketRegimeGenerator, MAMLTrader, first-order, world-model-only adaptation)
- `models/position_sizing.py` (KellyPositionSizer.dynamic_sizing, ATRPositionSizer, FixedFractionSizer)
- `models/risk_supervisor.py` (SQLite-persisted deterministic gates)
- `models/mcts.py` (PUCT planning)
- `backtest/engine.py` (BacktestResult, prepare_ohlc, walk_forward purge+embargo, summarize)
- `backtest/costs.py` (CostModel; documented backtesting.py 0.6.2 fill semantics)
- `live/live_trade_mt5.py` (candle-close loop, promotion gate, kill switch, TRADING_MODE gates)
- `live/trade_executor.py` (idempotency, lot-step flooring, aggregate caps, BELOW_MIN_LOT, close-never-blocks safety override)
- `archive/backtest_engine_legacy_fake.py` (archived fake engine — evidence artifact)
- `artifacts/models/ppo_gold_v1/` (manifest.json, evaluation.json, feature_contract.json, model.zip — no validated/promoted checkpoint)
- kernc/backtesting.py — https://github.com/kernc/backtesting.py (backtest engine dependency)

---

*End of FINAL_AUTONOMOUS_TRADING_AI_STUDY.md — a research study and documentation deliverable, not trading advice, and not a guarantee of returns. Every claim carries exactly one marker: FACT, RESEARCH FINDING, ENGINEERING RECOMMENDATION, or UNPROVEN/HYPOTHETICAL.*

<!-- APPEND-HERE -->
