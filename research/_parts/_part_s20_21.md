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
