# Audit v2 — Wiring Map / Import Graph (verified in this audit)

## Registry dispatch (`models/registry.py::load_policy`, lines 1-91)

```
load_policy(manifest_path)  →  _load_policy(Path, device, seen_set)
  ├─ load_manifest(path)  [core/model_artifacts.py]
  │    • validates manifest_version == 1
  │    • model file + feature_contract.json must exist
  │    • contract["hash"] == manifest.contract_hash  (else ModelArtifactError "hash mismatch")
  │    • manifest.obs_dim == window*n_features + 5  (else "obs_dim mismatch")
  ├─ dispatch by manifest.model_type:
  │   ├─ "ppo"           → PpoPolicy(model_file)                      [models/policy.py ~25]  (SB3 PPO.load)
  │   ├─ "dreamer"       → DreamerPolicy(DreamerV3Agent.from_checkpoint(model_file, device))  [lazy import models.dreamer_agent]
  │   ├─ "transformer"   → TransformerPolicy(TransformerAgentWrapper.from_checkpoint(model_file, map_location=device))  [lazy import models.transformer_policy]
  │   ├─ "dreamer_mcts"  → DreamerMCTSPolicy(DreamerV3Agent.from_checkpoint(...), num_simulations=extra.mcts_simulations, c_puct=extra.c_puct)  [lazy import models.mcts]
  │   └─ "ensemble"      → for each member: _member_manifest_path → recursive _load_policy (cycle-detect via seen set)
  │                          • REQUIRES all member contract_hash == ensemble manifest contract_hash  (else "same contract hash" error)
  │                          • builds EnsemblePolicy(min_agreement, vote, weights)
  └─ FINAL validation: policy.obs_dim == manifest.obs_dim AND policy.action_dim == manifest.action_dim  (else ModelArtifactError)
  Returns (policy, manifest)
```

## Live path import chain (production entry `live/live_trade_mt5.py`)

```
live/live_trade_mt5.py (952 lines)
  ├─ core/config         (AppConfig, TradingMode, load_config)
  ├─ core/model_artifacts (ModelArtifactError, load_manifest)
  ├─ core/observation    (AccountState)
  ├─ live/broker         (BaseBroker, BrokerError, PositionInfo)
  ├─ live/decision_engine (Decision, DecisionEngine)
  ├─ live/mock_broker    (MockBroker)
  ├─ live/model_signal   (ModelSignalSource)          → models/registry.load_policy → models/policy.* → all wrappers
  ├─ live/mt5_broker     (Mt5Broker)
  ├─ live/trade_executor (TradeExecutor)
  ├─ live/trade_manager  (Close, ModifySL, PartialClose, TradeManager)
  └─ models/position_sizing (ATRPositionSizer, KellyPositionSizer)
  └─ models/risk_supervisor (RiskSupervisor)
       │
       └── signal_source.build_signal_source(cfg):
             "model"/"ppo" → ModelSignalSource(cfg)  → load_policy(cfg.model.manifest_path)
             else          → RuleSignalSource (heuristic SMA, no model)

live/model_signal.py (130 lines)
  ├─ core/feature_pipeline (load_macro_daily, transform_feature_pipeline)
  ├─ core/model_artifacts  (load_manifest)
  ├─ core/observation      (AccountState, build_observation)
  ├─ live/decision_engine  (Decision)
  └─ models/registry       (load_policy)   ← THE hub that reaches every model module

live/decision_engine.py (151 lines)
  ├─ models.policy (PolicyOutput)
  └─ models.position_sizing (KellyPositionSizer)

live/trade_executor.py (601 lines)
  ├─ core.config (AppConfig)
  ├─ live/broker (BaseBroker, BrokerError, OrderRequest, OrderResult, PositionInfo, ReconResult)
  ├─ models.position_sizing (ATRPositionSizer)
  └─ models.risk_supervisor (RiskSupervisor)   ← check_trade on EVERY entry/scale path

live/trade_manager.py (105 lines)
  └─ live.broker (PositionInfo)
```

## Model-module consumers (full reachability table)

| models/* module | Imported by | Wired? |
|---|---|---|
| models/policy.py | models/registry.py, live/decision_engine.py, live/model_signal.py (via registry), train/evaluate.py, train/train_transformer.py, train/train_adversarial.py, train/meta_train_dreamer.py, train/adapt_recent.py, backtest/model_signals.py, tests | **WIRED (live + train + test)** |
| models/registry.py | live/model_signal.py, train/evaluate.py, train/train_ensemble.py, backtest/model_signals.py, tests | **WIRED** |
| models/ensemble.py | models/registry.py (ensemble branch), train/train_ensemble.py, tests | **WIRED** |
| models/mcts.py | models/policy.py (DreamerMCTSPolicy lazy import), models/registry.py (dreamer_mcts branch), tests | **WIRED** |
| models/meta_learning.py | train/meta_train_dreamer.py, train/adapt_recent.py, tests | **WIRED (training path)** |
| models/adversarial_training.py | train/train_adversarial.py, tests | **WIRED (training path)** |
| models/transformer_policy.py | models/policy.py (TransformerPolicy), train/train_transformer.py, train/train_ensemble.py (member training), tests | **WIRED** |
| models/dreamer_agent.py | models/policy.py, models/mcts.py (via agent), train/train_dreamer.py, train/train_adversarial.py, train/meta_train_dreamer.py, train/adapt_recent.py, tests | **WIRED** |
| models/dreamer_components.py | models/dreamer_agent.py, models/mcts.py (symexp), tests | **WIRED** |
| models/position_sizing.py | live/decision_engine.py, live/trade_executor.py, live/live_trade_mt5.py, tests | **WIRED** |
| models/risk_supervisor.py | live/trade_executor.py, live/live_trade_mt5.py, tests | **WIRED** |

## No orphans remain

Every `models.*` module is reachable from at least one of: the live entry point, the registry, or a training script. The previous audit's five orphans (ensemble, mcts, meta_learning, adversarial_training, transformer_policy) are all now imported by registry/train scripts/live chain.

## Deployment gate (live-only enforcement, `live/live_trade_mt5.py` ~line 790)

```
model_promotion_failures(cfg):
  load_manifest(cfg.model.manifest_path)                       # fails if manifest/weights absent
  requires sibling evaluation.json where evaluation["passed"] is True
        AND evaluation["contract_hash"] == manifest.contract_hash
  → returns list of failure strings; enforce_model_promotion_gate raises RuntimeError in LIVE mode
    (warns in demo unless REQUIRE_PROMOTED_MODEL=true)
_build_broker(cfg):
  demo → MockBroker (seed 42, state/mock_state.json)
  live → promotion gate + RiskSupervisor constructible + MT5_LOGIN/PASSWORD/SERVER present → Mt5Broker
```

**Conclusion: the wiring is complete end-to-end from `live/live_trade_mt5.py` down to every model wrapper via the registry, with hard validation (contract hash + obs_dim + action_dim) at every boundary. The only missing link to actual production is the trained artifact + evaluation record, which the promotion gate enforces.**
