---
kind: external_dependency
name: PyTorch — Deep learning backend
slug: pytorch
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
---

Deep learning framework powering the Dreamer V3 world-model agent and any custom neural components under `models/`. Device selection is automatic (CUDA > MPS > CPU) and supports Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU fallback. Used exclusively by the Dreamer V3 training pipeline; PPO training goes through Stable-Baselines3's internal PyTorch usage.