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
