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
