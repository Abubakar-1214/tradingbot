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
