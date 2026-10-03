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
