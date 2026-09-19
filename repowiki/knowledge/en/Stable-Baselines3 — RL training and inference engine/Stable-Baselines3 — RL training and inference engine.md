---
kind: external_dependency
name: Stable-Baselines3 — RL training and inference engine
slug: stable-baselines3
category: external_dependency
category_hints:
    - framework_behavior
    - sdk_real_api
scope:
    - '**'
---


Framework behavior: models are trained with `MlpPolicy` on custom Gymnasium envs and deployed unchanged against the same observation shape `(window * features + 1)`.