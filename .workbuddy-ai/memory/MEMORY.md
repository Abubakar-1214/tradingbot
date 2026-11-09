# Project Memory - autonoumuse_trader

## Architecture: 15-Layer Advanced Autonomous Forex Trading AI

Research completed on 2026-11-08. Proposed 15-layer architecture for the trading AI.

### Current System Status (as of Nov 2026):
- **6 layers fully exist:** L1 Data, L2 Features, L7 World Model (DreamerV3), L8 Prediction (Ensemble), L10 Policy (PPO+Dreamer), L11 Risk (RiskSupervisor), L12 Execution (TradeExecutor)
- **3 layers partial:** L3 Microstructure (12 features only), L6 Regime (MAML regime generator), L13 Adversarial (MarketMakerAgent), L14 Meta-learning (MAML)
- **4 layers missing (2 novel):** L4 Sentiment, L5 Causal Inference (NOVEL), L9 Uncertainty Quantification (NOVEL), L15 LLM Orchestration (NOVEL)

### Priority Build Order:
1. L9 Uncertainty Quantification (BNN, deep ensembles, conformal prediction)
2. L15 LLM Orchestration (dual-layer reflection)
3. L4 Sentiment & News Intelligence (HAD agents, vector memory)
4. L5 Causal Inference (SCM, DAG, do-calculus) - NOVEL
5. Upgrade L3 (DeepLOB, PIN, Hawkes)
6. Upgrade L6 (HMM, ADWIN)
7. Upgrade L13 (MAGAT-style adversarial, 4 stress scenarios)
8. Upgrade L14 (Meta-SGD bi-level, NAS, EWC continual learning)

### Key Research References:
- MAGAT (Nature 2026): adversarial training, MAPPO, LARS, Sharpe 1.97-2.31
- MarS (Microsoft 2024): Large Market Model, order-level simulation
- FinAgent (KDD 2024): LLM agent with dual-layer reflection
- Meta-SGD (2025): bi-level optimization, learnable learning rates
- DreamerV3 (Nature 2025): world models
- Full doc: research/ADVANCED_15_LAYER_ARCHITECTURE.md
