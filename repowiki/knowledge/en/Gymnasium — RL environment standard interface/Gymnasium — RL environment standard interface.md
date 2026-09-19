---
kind: external_dependency
name: Gymnasium — RL environment standard interface
slug: gymnasium
category: external_dependency
category_hints:
    - framework_behavior
scope:
    - '**'
---


Note: the DreamerV3 training loop in `train_dreamer.py` defines its own `TradingEnvironment` class that does NOT inherit from `gymnasium.Env`, so it is not compatible with SB3 without adaptation.