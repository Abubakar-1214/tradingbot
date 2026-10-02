# Models Audit Report — Autonomous XAUUSD Trading Bot

## Status update

The current model catalog and implemented interfaces are documented in
[`docs/MODELS.md`](docs/MODELS.md). The historical audit below is retained as
written.

**Project:** `e:\Desktop\NeoMind\Bazz\autonoumuse_trader`
**Question answered:** *Are the models added to this project FULLY implemented?*
**Method:** Code reading (every model, training script, environment, live path, eval path, and doc) **combined with online research** of the official papers/repos for each model family, so that "what the model is supposed to be" is verified against the canonical source — not just against the local code.
**Date of audit:** 2025 (evidence collected from the repository as-is).

---

## 1. Executive Summary (short version)

This repository contains **9 model-related modules**. The honest bottom line:

| Verdict class | Models |
|---|---|
| **FULLY IMPLEMENTED** (logic + wiring + tests all real and connected) | `risk_supervisor.py` (deterministic risk layer, not a neural net) |
| **PARTIALLY IMPLEMENTED** (real algorithm code + real training entry points, but **no trained checkpoint exists** and/or **not wired to live trading**) | DreamerV3 (`dreamer_agent.py` + `dreamer_components.py`), SB3 PPO (training scripts + live signal source), `position_sizing.py` (ATR sizer wired; Kelly/fixed-fraction unwired) |
| **STUB-ONLY / NOT IMPLEMENTED** (skeleton code with critical functions stubbed out, non-runnable) | `transformer_policy.py`, `meta_learning.py`, `adversarial_training.py` |
| **PARTIALLY IMPLEMENTED / NOT WIRED** (logic present but buggy or incompatible, zero callers anywhere) | `ensemble.py`, `mcts.py` |

**The single most important finding:** there is **zero trained model artifacts** in this repository (no `.pt`, no `.zip`, no `.pth`, no `.onnx`). Every path that would load a trained model — the live PPO signal source, the PPO evaluator, the Dreamer evaluator, the legacy MetaApi loop — **refuses to start or crashes** because the checkpoint does not exist. In practice, the **live trading system runs a purely heuristic SMA 20/50 crossover** (`RuleSignalSource`), and **no neural-network model is live in any path**.

---

## 2. Verdict Legend

- **FULLY IMPLEMENTED** — the algorithm logic is complete and correct, it is wired into the runnable system (live / backtest / train / tests), and it can actually run end-to-end with a trained artifact (or needs none, e.g. a deterministic rule layer).
- **PARTIALLY IMPLEMENTED** — substantial, real algorithm code exists and is wired into at least one path (usually training/eval), but one or more of the following is missing: a trained checkpoint, live-trading wiring, or complete training coverage.
- **STUB-ONLY** — the file/skeleton exists but core functions are empty `pass`, `return None`, `return []`, or commented-out calls; it cannot learn or run.
- **NOT WIRED** — the code is implemented (or partially implemented) but **no other module in the repository imports or calls it**; it is orphaned research code.
- **NOT IMPLEMENTED** — no code exists at all (only wishlist/docs mention).

---

## 3. Overall Verdict Table

| # | Model / Module | What it is (canonical) | Implementation quality | Wired? | Trained artifact? | **Verdict** |
|---|---|---|---|---|---|---|
| 1 | **DreamerV3** — `models/dreamer_agent.py` + `models/dreamer_components.py` | World-model RL agent (RSSM + imagination actor-critic) | Substantial real algorithm (encoder/RSSM/decoder/reward/critic/actor, symlog, KL balancing, λ-returns, full save/load) | Train + eval only (`train/*.py`, `evaluate_model.py`, `eval/analyze_dreamer.py`); **NOT in `live/`** | ❌ None exists | **PARTIALLY IMPLEMENTED** (algorithm real, never trained, not wired to live) |
| 2 | **SB3 PPO** — `train/train_ppo.py`, `train/train_ppo_aggressive.py`, `live/live_trade_mt5.py::PpoSignalSource`, `live/live_trade_metaapi.py`, `eval/eval_ppo.py` | Proximal Policy Optimization (library-based) | Real SB3 training scripts (correct API usage) + live signal gate | Train + legacy eval + optional live path (gated) | ❌ None exists (`train/ppo_xauusd_latest.zip` absent) | **PARTIALLY IMPLEMENTED** (training path real & runnable, but no checkpoint → live ML path refuses to start) |
| 3 | **Transformer policy** — `models/transformer_policy.py` | Transformer encoder policy (DT-style sequence policy) | Forward pass real; **`train_step` is bare `pass`**, `get_attention_weights` returns `None`; **no Decision Transformer class exists** | Orphan (no importers) | ❌ | **STUB-ONLY / NOT WIRED** |
| 4 | **Ensemble** — `models/ensemble.py` | Deep ensemble / consensus voting | Consensus logic present but **incompatible with Dreamer tuple `act()`** (hashes numpy arrays as dict keys; assumes scalar return) | Orphan (no importers) | ❌ | **PARTIALLY IMPLEMENTED / NOT WIRED** (buggy) |
| 5 | **MCTS** — `models/mcts.py` | Monte-Carlo tree search over latent world model (AlphaZero/MuZero style) | Search implemented, but action set lacks SHORT, and `DreamerMCTSAgent.act` has a `prev_action` AttributeError bug | Orphan (no importers) | ❌ | **PARTIALLY IMPLEMENTED / NOT WIRED** (buggy) |
| 6 | **Meta-learning (MAML)** — `models/meta_learning.py` | Model-Agnostic Meta-Learning (fast regime adaptation) | **Stubs:** `_sample_batch()` returns `None`; regime finders return `[]`; requires nonexistent `base_agent.compute_loss` → not runnable | Orphan (no importers) | ❌ | **STUB-ONLY / NOT IMPLEMENTED** |
| 7 | **Adversarial training** — `models/adversarial_training.py` | Self-play / adversarial market-maker training | **Stubs:** MM `learn()` is stats-only; trader `learn()` commented out; env market state is random noise | Orphan (no importers) | ❌ | **STUB-ONLY / NOT IMPLEMENTED** |
| 8 | **Position sizing** — `models/position_sizing.py` | Kelly / fixed-fraction / ATR position sizing | Real math (Kelly f*, fractional, capped; ATR risk-based) | `ATRPositionSizer` **wired into live executor**; Kelly/FixedFraction only tests/verify scripts | N/A (no training) | **FULLY IMPLEMENTED (ATR, wired)** / Kelly & FixedFraction **implemented but effectively unwired** |
| 9 | **Risk supervision** — `models/risk_supervisor.py` | Deterministic circuit-breaker risk layer (SQLite-persisted) | Complete: 11 rejection gates, daily reset, emergency shutdown, state persistence | **Wired into live** (`trade_executor.py`, `live_trade_mt5.py`) + tests + verify gates | N/A (deterministic, state DB exists at `state/risk_state.db`) | **FULLY IMPLEMENTED and WIRED** |

> **No model in this project is fully implemented in the sense of "logic + wiring + trained & connected to live".** Only the deterministic risk layer qualifies as fully implemented-and-wired. Everything neural-network-based is either untrained, unwired, or stubbed.

---

## 4. Per-Model Analysis

---

### 4.1 DreamerV3 — `models/dreamer_agent.py` (520 lines) + `models/dreamer_components.py` (351 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

**DreamerV3** (Hafner et al., 2023 — *"Mastering Diverse Domains through World Models"*, **arXiv:2301.04104**; official code **github.com/danijar/dreamerv3**) is a model-based reinforcement learning agent. It learns a **world model** of the environment from past experience, then **"imagines"** futures inside that learned latent model and trains an actor-critic policy entirely on the imagined trajectories ("learning by imagination").

The canonical architecture (from the paper and official repo):
1. **Sensory encoder + RSSM (Recurrent State-Space Model)** — compresses observations into a discrete latent state `(h_t, z_t)`; the RSSM predicts the next latent state given the previous state and action (prior) and corrects it with the new observation (posterior). DreamerV3 uses **discrete categorical latents** (32 classes × 32 categories by default) with straight-through gradient estimation.
2. **Decoder / Reward predictor** — reconstruct observations and predict reward from the latent state (trained with **symlog** targets).
3. **KL balancing** between prior and posterior with **free nats** so the world model keeps learning when predictions are already good.
4. **Actor + Critic trained in imagination** — the actor maximizes predicted returns via REINFORCE with a baseline from the critic; the critic is regressed toward **λ-returns**; **return normalization** (Symlog returns, percentile normalization) keeps scale stable; slow/EMA target critic for stability.
5. Replay buffer of experience sequences; training alternates world-model updates and actor-critic updates.

DreamerV3 is famous for being a single, general algorithm that works across many domains without per-domain tuning.

#### (b) Actual implementation in this codebase (file:line evidence)

- `models/dreamer_components.py`:
  - `symlog` / `symexp` (symlog targets, line ~30s), `unimix_logits` (unimix=0.01, line ~40s) — canonical DreamerV3 tricks ✅
  - `RMSNorm`, `GRUCell` (recurrent cell for RSSM)
  - `Encoder` — MLP 512/512 → embed_dim, RMSNorm+SiLU, applies symlog to the raw observation window
  - `RSSM` — prior/posterior networks, GRU over `concat(z_onehot, action)`, `observe()` (4-tuple), `imagine()`, `_sample_categorical` (straight-through), `get_state`, `kl_loss` (free-nats + dyn/rep scales)
  - `Decoder`, `RewardPredictor`, `Actor` (Categorical with unimix), `Critic` (symlog value)
- `models/dreamer_agent.py`:
  - `ReplayBuffer` — numpy ring buffer, capacity 100,000, `seq_len=64`, episode-boundary guard via done-mask prefix-sum (returns `None` on degenerate batch)
  - `DreamerV3Agent` — holds encoder/RSSM/decoder/reward/actor/critic/slow_critic (EMA tau=0.02), **3 optimizers** (world_model, actor, critic)
  - `train_step(batch_size)` — Phase 1: world-model loss (symlog reconstruction + reward MSE + KL); Phase 2: imagination rollout (horizon=15), critic regression to λ-returns, actor REINFORCE + entropy
  - `_imagine_trajectory`, `_lambda_returns`, `_normalize_returns`, `_update_slow_critic`
  - `act(obs, h, z, deterministic)` — one-hot categorical output, maintains `prev_action`
  - `save` / `load` — persists all 6 networks + 3 optimizers + return normalizer + RNG state + training_step (`weights_only=False`, backward compatible). This is a *genuinely complete* checkpoint format.
  - Hyperparameters match the paper's spirit: stoch 32×32, gamma 0.99, lambda 0.95, horizon 15, free_nats 1.0, kl scales, entropy 3e-4, bf16 autocast on CUDA.

#### (c) Complete vs stub

The **algorithm is substantially, genuinely implemented** — this is not a stub. Every canonical DreamerV3 component (RSSM, discrete latents, symlog, KL balancing, imagination actor-critic, λ-returns, EMA critic, full checkpoint save/load) exists with real math. **However**, it has **never been trained in this repository** (no checkpoint anywhere — see §7), so its correctness at scale is unproven here.

#### (d) Wiring — who imports/calls it

- `train/train_dreamer.py:18`, `train/train_god_mode.py:24`, `train/train_ultimate_150.py:28` (training entry points)
- `evaluate_model.py:22` (eval-only; default checkpoint `train/dreamer_ultimate/ultimate_150_xauusd_final.pt` — **absent** → exits)
- `eval/analyze_dreamer.py:20` (analysis; default `train/dreamer/dreamer_xauusd_final.pt` — **absent** → exits)
- `ReplayBuffer` also used by `scripts/_verify_p1_fixes.py:45`
- **NOT imported anywhere in `live/`.** The live system never references DreamerV3 (`live/live_trade_mt5.py` signal sources are `RuleSignalSource`/`PpoSignalSource`).

#### (e) Setup, use, train

```bash
# Train (requires data/xauusd_1h_macro.csv or data/xauusd_1h.csv)
python train/train_dreamer.py --steps 100000 --batch-size 16 --device cpu
#   → checkpoints saved to train/dreamer/dreamer_xauusd_{step}k.pt + dreamer_xauusd_final.pt
# God-mode variant (needs macro CSV; supports --resume)
python train/train_god_mode.py
# Ultimate-150 variant (1M default steps; supports --resume)
python train/train_ultimate_150.py
# Evaluate once a checkpoint exists
python evaluate_model.py --checkpoint train/dreamer/dreamer_xauusd_final.pt
```
Expected output of a full run: periodic checkpoints every 10k steps + a `_final.pt`. **Note:** `train/train_dreamer.py` has **no `--resume` flag** (despite what COLAB_TRAINING_GUIDE claims; see §9). Also note the RAM/GPU reality: DreamerV3 1M-step training needs a GPU (Colab T4 ≈ 24–30 h per the project's own guide); this machine has no GPU.

#### (f) Verdict

**PARTIALLY IMPLEMENTED** — real, complete algorithm code + real training/eval wiring, but **(1) no trained checkpoint has ever been produced** in this repo and **(2) DreamerV3 is not wired into live trading at all**. It cannot trade anything today.

---

### 4.2 SB3 PPO — training scripts + live signal source

#### (a) What it is & how it is SUPPOSED to work (online-verified)

**PPO — Proximal Policy Optimization** (Schulman et al., 2017, **arXiv:1707.06347**) is a policy-gradient RL algorithm that updates the policy while **clipping** the probability ratio so the new policy never moves too far from the old one (a "trust region" without TRPO's complexity). Canonical PPO: collect rollout trajectories with multiple workers, compute **Generalized Advantage Estimation** (GAE, λ≈0.95), then do several minibatch epochs of clipped surrogate loss + value-function loss + entropy bonus.

**Stable-Baselines3 (SB3)** — the standard library wrapper (`stable-baselines3.readthedocs.io/en/master/modules/ppo.html`): `PPO("MlpPolicy", env, n_steps=..., batch_size=..., gamma=..., learning_rate=..., ent_coef=...)`, then `model.learn(total_timesteps=...)`, `model.save(...)`, `PPO.load(path)` and `model.predict(obs, deterministic=True)`.

#### (b) Actual implementation in this codebase

- `train/train_ppo.py` (92 lines) — correct SB3 usage: `SubprocVecEnv` of 8× `XAUUSDTradingEnv` (2-action long-only, window 64, cost 1 bp, max_episode_steps=20,000); data `data/xauusd_1h.csv`; `PPO("MlpPolicy", n_steps=1024, batch_size=256, gamma=0.99, lr=3e-4)`; 10×50k = 500k steps; saves `train/ppo_xauusd_{k}k.zip` + `_latest.zip`. Uses `mp.set_start_method("spawn")` (Windows-safe).
- `train/train_ppo_aggressive.py` (123 lines) — 16× `XAUUSDTradingEnvAggressive` (3-action incl. short, window 120, stop-loss, turnover penalty off); data `data/xauusd_1h_macro.csv`; `PPO(n_steps=2048, batch_size=512, ent_coef=0.01)`; 500k steps; saves `train/ppo_xauusd_macro_{k}k.zip` + `_latest.zip`.
- `live/live_trade_mt5.py::PpoSignalSource` — lazy `PPO.load(cfg.paths.model_path)` + feature-contract JSON load; **raises `FileNotFoundError` if the model/contract are missing** ("Train a model or switch SIGNAL_SOURCE=rule"); builds the feature frame, validates the contract, `model.predict(obs, deterministic=True)`.
- `live/live_trade_metaapi.py` (231 lines, **LEGACY**) — hardcoded `MODEL_PATH = "train/ppo_xauusd_latest.zip"`, `VOLUME = 0.01`, `MAGIC_NUMBER = 234000`, `PPO.load(...)` at startup (crashes if absent), long-only open/close. No risk supervisor, no ATR sizing, no idempotency.
- `eval/eval_ppo.py` (93 lines) — `PPO.load("train/ppo_xauusd_latest.zip")` + deterministic predict loop vs Buy&Hold and MA baselines. Requires the absent checkpoint.

#### (c) Complete vs stub

The **training scripts are real and correctly use the SB3 library** (right API, right vectorized envs, proper save paths). The **inference path is real**. What is missing is the **artifact**: `train/ppo_xauusd_latest.zip` and `train/feature_contract.json` **do not exist** in this repo.

#### (d) Wiring

- Training: self-contained scripts (runnable today if data exists).
- Live: `build_signal_source(cfg)` in `live/live_trade_mt5.py` — `SIGNAL_SOURCE=ppo` selects `PpoSignalSource`; the live `_build_broker` gate refuses LIVE mode when SIGNAL_SOURCE=ppo unless the model + feature contract are present. Default is `SIGNAL_SOURCE=rule` (heuristic SMA).
- Eval: `eval/eval_ppo.py` (blocked on absent zip). Legacy: `live_trade_metaapi.py`.

#### (e) Setup, use, train

```bash
# Train (needs data/xauusd_1h.csv)
python train/train_ppo.py                 # → train/ppo_xauusd_500k.zip, train/ppo_xauusd_latest.zip
# Aggressive variant (needs data/xauusd_1h_macro.csv)
python train/train_ppo_aggressive.py      # → train/ppo_xauusd_macro_500k.zip, train/ppo_xauusd_macro_latest.zip
# Live with ML signals (only after a checkpoint + feature_contract.json exist)
# set SIGNAL_SOURCE=ppo in config
python live/live_trade_mt5.py
# Evaluate
python eval/eval_ppo.py
```

#### (f) Verdict

**PARTIALLY IMPLEMENTED** — real, library-correct training and inference code, wired into live as the *only* ML signal option, but the required trained checkpoint does not exist, so the live ML path **refuses to start** and today the live bot runs the heuristic rule instead.

---

### 4.3 Transformer policy — `models/transformer_policy.py` (441 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

A **Transformer-based policy** for RL over time-series (e.g., **Decision Transformer**, Chen et al. 2021, **arXiv:2106.01345** / **github.com/kzl/decision-transformer** — which casts RL as return-conditioned sequence modeling with a GPT-style causal transformer; or general transformer encoder policies that attend over a window of observations). In a trading context the idea is: attend over the last `seq_len` steps of features to decide the next action, capturing long-range temporal dependencies that RNNs handle worse.

#### (b) Actual implementation

- `PositionalEncoding`; `TransformerActor(state_dim, action_dim, hidden_dim=256, num_heads=8, num_layers=4, seq_len=64, dropout=0.1)` — Linear projection → positional encoding → `nn.TransformerEncoder` (batch_first, gelu) → last-token → action head (greedy argmax in `act`).
- `TransformerCritic` — same stack → scalar value.
- `TransformerAgentWrapper` — holds actor+critic + 2 Adam optimizers.
- **CRITICAL STUB:** `train_step(batch)` body is literally `pass` — the comment says *"In full implementation, would train on sequences"*. So **nothing ever learns**.
- `get_attention_weights()` returns `None`.
- **No Decision Transformer class exists** anywhere in the codebase — no return-conditioning, no causal masking, no DT training.

#### (c) Complete vs stub

**STUB-ONLY** — the forward pass is real, but the training step is empty, so this can never learn. It is also not a Decision Transformer despite the docs mentioning DT.

#### (d) Wiring

**Orphan.** No file imports `transformer_policy.py`. It is referenced only in documentation (`.kiro` design doc, `.qoder`/`repowiki` pages). `FIXES.md` P1-10 confirms the research-only modules are "imported by no wired path".

#### (e) Setup / use / train

There is **no training script** for it. You could only use it by writing a new training loop yourself (`agent = TransformerAgentWrapper(...)`, implement `train_step`, save/load checkpoints).

#### (f) Verdict

**STUB-ONLY / NOT WIRED / NOT IMPLEMENTED** (as a Decision Transformer).

---

### 4.4 Ensemble — `models/ensemble.py` (289 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

**Deep Ensembles** (Lakshminarayanan et al., 2017, **arXiv:1612.01474** — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles") train N networks with different seeds/initializations and **average their predictions** (or majority-vote the actions) to improve robustness and get uncertainty estimates from the disagreement (entropy). In trading bots, an "ensemble of 5 agents, consensus ≥ 3" is the design used in this project's `.kiro` spec.

#### (b) Actual implementation

- `EnsembleAgent(agent_class, num_models=5)` — seeds `42+i`, `hidden_dim += i*16`.
- `act(obs, use_consensus=True, consensus_threshold=3)` — majority vote; `get_uncertainty()` entropy.
- `train/save/load` with `prefix_model{i}.pt`.
- `MockAgent` demo.

**Bugs / incompatibilities:** `act` calls `model.act(obs)` (ensemble.py line 86) and hashes the returned action as a dict key (`action_counts[action]`, line 97) — but DreamerV3's `act` returns `(action_array, (h, z))` where `action_array` is a **numpy array** (`action.cpu().numpy()[0]`, `models/dreamer_agent.py:233`), and numpy arrays are **unhashable** → `TypeError: unhashable type: 'numpy.ndarray'` the moment the vote counter runs. The vote code also assumes a scalar action in {0,1,2}. So even a "correct" Dreamer ensemble would crash.

#### (c) Complete vs stub

**PARTIALLY IMPLEMENTED** — consensus logic exists and is conceptually sound, but it is incompatible with the only real agent it was designed to wrap (DreamerV3) and has never been executed against it.

#### (d) Wiring

**Orphan** — zero importers in code. Only `.kiro`/`.qoder` docs mention it.

#### (e) Setup / use / train

No training script. If you fixed `act()` you would still need to train 5 separate agents (`ensemble.train(...)` per model) and then `ensemble.act(obs)`.

#### (f) Verdict

**PARTIALLY IMPLEMENTED / NOT WIRED** (broken integration with the Dreamer interface).

---

### 4.5 MCTS — `models/mcts.py` (330 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

**Monte-Carlo Tree Search over a learned world model** is the *planning* idea behind **AlphaZero** (Silver et al., **arXiv:1712.01815**) and **MuZero** (Schrittwieser et al., **arXiv:1911.08265**): from a state, run many simulated rollouts — select nodes by UCB, expand children with the policy prior, evaluate leaves with the value function, backprop rewards — then pick the most-visited action. In Dreamer-style agents, MCTS plans **in latent space** using the RSSM to imagine transitions and the reward/critic heads to score them.

#### (b) Actual implementation

- `MCTSNode` — UCB `select_child(c_puct=1.0)`, `expand` uses `agent.rssm.imagine` + `symexp(reward_predictor(...))`, `backup`.
- `MCTS(agent, num_simulations=100, c_puct, gamma)` — search over latent states.
- **Bug 1:** `self.actions` contains only **two** entries — `[1,0]` (flat) and `[0,1]` (long). **There is no short action**, even though the trading envs support 3 actions (0/1/2) and the .kiro spec demands long/short/flat.
- **Bug 2:** `DreamerMCTSAgent.__init__` (lines 251–253) **never initializes `self.prev_action`** — it is only assigned at the end of `act` (line 289). If the **first** call to `act` passes non-`None` `h, z` (latent-state injection, e.g. resuming from a saved state), the `else` branch reads `self.prev_action` (line 275) → `AttributeError`. (In the standard flow where the first call uses `h=None, z=None`, `prev_action` is set on that first call, so the error manifests on the latent-injection path — but the attribute is genuinely uninitialized.)

#### (c) Complete vs stub

**PARTIALLY IMPLEMENTED (buggy)** — the search core is real (selection/expansion/backup), but the action space is wrong and the agent wrapper has a definite runtime bug.

#### (d) Wiring

**Orphan** — no importers. The `.kiro` design.md lists MCTS in the "already exists" set but the missing `DecisionCore`/`AutonomousOrchestrator` that would call it **do not exist**.

#### (e) Setup / use / train

No integration path. Even manual use hits the `prev_action` bug and the missing short action.

#### (f) Verdict

**PARTIALLY IMPLEMENTED / NOT WIRED** (buggy, incomplete action space).

---

### 4.6 Meta-learning (MAML) — `models/meta_learning.py` (344 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

**Model-Agnostic Meta-Learning (MAML)** (Finn et al., 2017, **arXiv:1703.03400**) trains a model's *initial weights* such that a few gradient steps on a new task's data ("fast adaptation") yields good performance — i.e., "learning to learn". In this project's design (`.kiro` spec) a **MAML adapter** would quickly adapt the trading policy to a new market regime.

#### (b) Actual implementation

- `MAMLTrader(base_agent, meta_lr=1e-3, adapt_lr=1e-2, adapt_steps=5)` — `meta_train` deep-copies the base agent, does inner SGD and outer meta-optimizer steps (correct MAML skeleton).
- **Stub 1:** `_sample_batch` body is `return None` → meta_train cannot run.
- **Stub 2:** `MarketRegimeGenerator._find_trending/_ranging/_volatile_periods` all `return []` → no task data ever generated.
- **Conceptual bug:** `fast_adapt` uses the *meta*-optimizer instead of fresh inner-loop parameters.
- **Hard dependency:** requires `base_agent.compute_loss`, which `DreamerV3Agent` does **not** define → would `AttributeError` even if the stubs were filled.

#### (c) Complete vs stub

**STUB-ONLY / NOT IMPLEMENTED** — skeleton only; nothing runnable.

#### (d) Wiring

**Orphan** — no importers.

#### (e) Setup / use / train

No training script exists. It cannot be invoked as-is.

#### (f) Verdict

**STUB-ONLY / NOT IMPLEMENTED**.

---

### 4.7 Adversarial training / self-play — `models/adversarial_training.py` (555 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

**Adversarial training** in the ML sense (Madry et al., 2018, **arXiv:1706.06083**) is min-max robust optimization against adversarial examples. In a **trading/market** context, the project's design uses **self-play**: a *trader agent* learns against an adversarial *market maker* agent that tries to manipulate/detriment the trader — so the trader learns to be robust to hostile market conditions (see e.g. **github.com/kayuksel/market-self-play**, **github.com/Aurovind7/MARL-Market-Maker**). A GAN-style alternating update (train trader → train MM → repeat) is canonical.

#### (b) Actual implementation

- `MarketMakerAgent` — policy MLP, `respond` = argmax over softmax.
- **Stub:** `_detect_trader_pattern` returns hardcoded `momentum_score = 0.0, reversion_score = 0.0`.
- **Stub:** `learn(reward)` — the body is a comment: *"Training would happen here — For now, just track statistics"* → **no gradient training at all**.
- **Stub:** `AdversarialTradingEnv._get_market_state()` returns `np.random.randn(100)` — **random noise**, not market data.
- **Broken:** `_apply_manipulation` touches `base_env` attributes that don't exist on the real envs.
- **Stub:** `SelfPlayTrainer._train_trader` — the trader-learn call is **commented out**; `_train_mm` never calls `mm.learn()`.

#### (c) Complete vs stub

**STUB-ONLY / NOT IMPLEMENTED** — no component actually learns.

#### (d) Wiring

**Orphan** — no importers.

#### (e) Setup / use / train

No training script. Cannot be trained as-is.

#### (f) Verdict

**STUB-ONLY / NOT IMPLEMENTED**.

---

### 4.8 Position sizing — `models/position_sizing.py` (404 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

Position sizing decides **how large** each trade is. Canonical methods:
- **Kelly criterion** (Kelly, 1956, *"A New Interpretation of Information Rate"*, Bell System Technical Journal 35:917–926): optimal fraction `f* = (p·b − q) / b` where p = win probability, b = win/loss odds ratio; usually used **fractional** (e.g., ¼ Kelly) to cut variance. **Wikipedia: Kelly criterion.**
- **Fixed-fractional**: risk a fixed fraction of equity per trade.
- **ATR-based**: size so that a stop of `k×ATR` risks at most `R%` of equity — the standard professional FX/commodity method (dollar risk ÷ (ATR × multiplier)).

#### (b) Actual implementation

- `KellyPositionSizer(max_position=0.10, kelly_fraction=0.25)` — full formula `f*`, fractional scaling, capped at max_position, returns 0 on no edge. ✅ Real math.
- `dynamic_sizing(agent, current_state, obs)` — uses the Dreamer world model: 1-step RSSM `observe` (4-tuple unpack), counterfactual `imagine`, win-prob clamped `sigmoid(advantage*5) ∈ [0.3, 0.7]`. ✅ Real (post P1-03 fix).
- `volatility_adjusted_sizing`, `update_statistics` (100-trade window), `FixedFractionSizer(0.02)`.
- `ATRPositionSizer(account_risk=0.02, atr_multiplier=2.0)` — `dollar_risk / (ATR × mult)`, capped at 10%, floored to LOT_STEP in the executor.

#### (c) Complete vs stub

**Implemented and correct.** The Kelly math and ATR sizing are genuine.

#### (d) Wiring

- `ATRPositionSizer` — **wired into live**: `live/live_trade_mt5.py:78` (passed to `TradeExecutor`), `live/trade_executor.py:64` (used for every entry/scale order), plus `scripts/verify_risk_integration.py:45`, `tests/helpers.py:14`, `tests/test_position_sizing.py:4`.
- `KellyPositionSizer` — `scripts/_verify_p1_fixes.py:44` + tests only (not in live).
- `FixedFractionSizer` — tests only.

#### (e) Setup / use / train

No training needed (statistics-based). Configurable via `cfg.risk.risk_per_trade` and `cfg.broker.sl_atr_mult`; used automatically in the live loop.

#### (f) Verdict

**FULLY IMPLEMENTED and WIRED (ATR sizer).** Kelly/fixed-fraction are implemented but only exercised by tests/verify scripts — mark those as **implemented but effectively NOT WIRED into production**.

---

### 4.9 Risk supervision — `models/risk_supervisor.py` (616 lines)

#### (a) What it is & how it is SUPPOSED to work (online-verified)

A **risk-management / circuit-breaker layer** — the standard institutional practice of hard, non-negotiable limits: daily loss limits, max drawdown, max position size, volatility/spread filters, trade cooldowns, consecutive-loss halts, event-window restrictions, kill switches (see e.g. tradewink.com algorithmic-trading risk management; eliteforextrading.com setting risk limits; oyamori.com circuit breakers). **It is not a machine-learning model** — it is deterministic safety logic that sits *in front of* the AI.

#### (b) Actual implementation

- `RiskSupervisor` — defaults: max_daily_loss 5%, max_position 10%, max_drawdown 15%, vol_threshold 3.0, max_spread 0.0005, max 20 trades/day, min interval 300 s, max 5 consecutive losses, event scale 0.5, initial equity 10,000.
- `check_trade(signal, requested_size, state, market_data)` — **11 rejection gates** incl. `POSITION_TOO_LARGE`, `HIGH_VOLATILITY`, `COOLDOWN`, `SPREAD_TOO_WIDE`, `EVENT_RISK`, `HALTED`; explicit fraction sizing.
- `update_state`, `emergency_shutdown`, `get_statistics`, `reset_daily`, `_maybe_reset_daily` (injectable UTC clock).
- `_db()` context manager — **SQLite persistence** with a Windows-safe WinError-32 fix → state survives restarts (`state/risk_state.db` exists in the repo).
- `_direction` normalization (int/str/dict), `_correlation_reject` (guard, off by default).
- `SafeTradingAgent(ai_agent, risk_supervisor)` wrapper.

#### (c) Complete vs stub

**Complete.** Verified by tests (`tests/test_risk.py`, `tests/test_executor.py`) and multiple verify gates.

#### (d) Wiring

- `live/live_trade_mt5.py:79` (constructed from config, consulted every decision cycle)
- `live/trade_executor.py:65` — **every ENTRY/SCALE/MODIFY order path consults `check_trade`**; CLOSE is never blocked (de-risk override)
- `scripts/verify_risk.py:29`, `scripts/verify_risk_integration.py:46`, `tests/helpers.py:15`, `tests/test_executor.py:27`, `tests/test_risk.py:6`

#### (e) Setup / use / train

No training. Configure via config/env; state auto-persists to `state/risk_state.db`.

#### (f) Verdict

**FULLY IMPLEMENTED and WIRED** — the only model in this project that is genuinely complete, tested, and connected to the live path.

---

## 5. Wiring Map (import graph)

```
models/risk_supervisor.py      ← live/live_trade_mt5.py:79, live/trade_executor.py:65,
                                  scripts/verify_risk.py:29, scripts/verify_risk_integration.py:46,
                                  tests/helpers.py:15, tests/test_executor.py:27, tests/test_risk.py:6
models/position_sizing.py
   ├─ ATRPositionSizer         ← live/live_trade_mt5.py:78, live/trade_executor.py:64,
   │                              scripts/verify_risk_integration.py:45, tests/helpers.py:14,
   │                              tests/test_position_sizing.py:4
   ├─ KellyPositionSizer       ← scripts/_verify_p1_fixes.py:44, tests
   └─ FixedFractionSizer       ← tests only
models/dreamer_agent.py
   ├─ DreamerV3Agent           ← train/train_dreamer.py:18, train/train_god_mode.py:24,
   │                              train/train_ultimate_150.py:28, evaluate_model.py:22,
   │                              eval/analyze_dreamer.py:20        (NO live/ import)
   └─ ReplayBuffer             ← scripts/_verify_p1_fixes.py:45
models/dreamer_components.py   ← dreamer_agent.py; symexp also eval/analyze_dreamer.py:94, models/mcts.py:91

SB3 PPO (library)              ← train/train_ppo.py, train/train_ppo_aggressive.py,
                                  live/live_trade_mt5.py::PpoSignalSource (gated),
                                  live/live_trade_metaapi.py (legacy), eval/eval_ppo.py

ORPHANS (zero importers):
   models/ensemble.py          ← referenced only in .kiro/design.md + .qoder/repowiki docs
   models/mcts.py              ← same
   models/meta_learning.py     ← same
   models/adversarial_training.py ← same (FIXES.md P1-10: "imported by no wired path")
   models/transformer_policy.py ← same
```

**Live decision chain (production entry point `live/live_trade_mt5.py`):**
`BarSource (MT5/synthetic) → on_closed_bar → RiskSupervisor.check_trade → TradeExecutor (ATRPositionSizer) → broker (Mock/MT5)`
with signal from `SIGNAL_SOURCE=rule` (**default: SMA 20/50 crossover heuristic**) or `SIGNAL_SOURCE=ppo` (SB3 PPO — **refuses to start without a checkpoint**).

**DreamerV3, Ensemble, MCTS, MAML, Adversarial, Transformer appear NOWHERE in the live, backtest, or gate/test paths.**

---

## 6. Training Scripts — entry commands and expected outputs

| Script | Entry command | Requires | Saves | Resume? | Runnable here? |
|---|---|---|---|---|---|
| `train/train_dreamer.py` | `python train/train_dreamer.py --steps 100000` | `data/xauusd_1h_macro.csv` or `data/xauusd_1h.csv` | `train/dreamer/dreamer_xauusd_{k}k.pt`, `_final.pt` (every 10k) | ❌ **no --resume flag** | Yes (CPU, slow) |
| `train/train_god_mode.py` | `python train/train_god_mode.py` | `data/xauusd_1h_macro.csv` | `train/dreamer/god_mode_xauusd_step{N}.pt`, `_final.pt` | ✅ `--resume` | Yes |
| `train/train_ultimate_150.py` | `python train/train_ultimate_150.py` | feature CSV(s) for M5/M15/H1/H4/D1/W1 + macro + calendar | `train/dreamer_ultimate/ultimate_150_xauusd_step{N}.pt`, `_final.pt` | ✅ `--resume` | Yes (1M steps default; huge) |
| `job_train_ultimate_150.py` | `python job_train_ultimate_150.py` | GPU (auto-batch: H100 128 / A100 64 / V100 32 / T4 32 / CPU 16) | same as ultimate | ✅ auto-resume scan | Launcher only |
| `train/train_ppo.py` | `python train/train_ppo.py` | `data/xauusd_1h.csv` | `train/ppo_xauusd_{k}k.zip`, `_latest.zip` | ❌ | Yes |
| `train/train_ppo_aggressive.py` | `python train/train_ppo_aggressive.py` | `data/xauusd_1h_macro.csv` | `train/ppo_xauusd_macro_{k}k.zip`, `_latest.zip` | ❌ | Yes |

**Expected output of any successful training run: periodic checkpoints.** **Reality: the save directories (`train/dreamer/`, `train/dreamer_ultimate/`) do not even exist** — i.e., **no training run has ever completed in this repository**.

**Env files (supporting infrastructure, all fully implemented):**
- `env/dreamer_trading_env.py` — `RealisticTradingEnv` (392 lines): window 64, short allowed, spread/commission/slippage/adverse-selection costs, event multipliers, swap, SL 0.01, max_drawdown 0.30, reward_scale 100, random start; `DEFAULT_ENV_KWARGS` + `EVAL_ENV_OVERRIDES`; `episode_stats()`.
- `env/xauusd_env.py` — `XAUUSDTradingEnv` (117 lines, 2-action long-only, used by train_ppo.py).
- `env/xauusd_env_aggressive.py` — `XAUUSDTradingEnvAggressive` (144 lines, 3-action, leverage, forced SL close; used by train_ppo_aggressive.py).
- `env/realistic_execution.py` — `RealisticExecutionModel` + `SlippageSimulator` (355 lines) — **NOT imported by any train script** (helper/demo only).

---

## 7. Model Artifact Search (subtask 4 result)

Searched the entire repository (excluding `.venv`, `archive/`, `__pycache__`) for `.zip`, `.pt`, `.pth`, `.onnx`, `.pkl`, `.safetensors`, `.joblib`, `.h5`:

**Result: ZERO model artifacts.** Files found were only project JSON/state files (`artifacts/_smoke_state/*`, `state/risk_state.db`, `state/mock_state.json`). `train/` contains only the 5 training scripts. `train/dreamer/` and `train/dreamer_ultimate/` do not exist. README.md itself states: *"MODEL_PATH=train/ppo_xauusd_latest.zip and FEATURE_CONTRACT_PATH=train/feature_contract.json do not exist in this repo."*

**Conclusion: no model has ever produced a checkpoint in this repository.** All "trained model" claims in guides are aspirational (see §9).

---

## 8. The 100-Model CSV (`Advanced_100_AI_Models_List.csv`)

The CSV lists 100 external AI models in 4 categories (ML 1–25, RL 26–50, DL 51–75, NN 76–100). Cross-referencing every row against the codebase:

| Presence | Rows |
|---|---|
| **Has real code in this repo** | **2:** #26 **Stable-Baselines3 PPO** (train_ppo.py, live PpoSignalSource, eval), #31 **DreamerV3** (models/dreamer_agent.py, train scripts, evaluate_model.py) |
| **External wishlist only** (no code, no import, no artifact) | **98:** XGBoost, LightGBM, CatBoost, Prophet, tsfresh, Optuna, RLlib DQN/A3C, Tianshou IMPALA, CleanRL, FinRL, TF-Agents, ElegantRL, PettingZoo, Pearl, ResNet-50, YOLOv8/v10, SAM, ViT, BERT, RoBERTa, T5, Whisper, Llama 3 8B, Mistral 7B, Phi-3, Qwen2, Gemma 2, Stable Diffusion v1.5/XL, CLIP, DINOv2, ConvNeXt, wav2vec2, Bark, CodeLlama, DeepSeek Coder, HRM, TFT, LSTM, GRU, Inception-v4, MobileNetV3, EfficientNetV2, DenseNet, VGG-16, U-Net, N-BEATS, PatchTST, Informer, Autoformer, GAT, GCN, Node2Vec, SNN, Pix2Pix, CycleGAN, StyleGAN3, SRGAN, Chronos, LSTM-FCN, Neural ODE, … (rows match only the CSV itself and doc pages) |
| **Repo's own custom models in the CSV** | **None** — ensemble, MCTS, MAML, adversarial, transformer policy, risk supervisor, position sizing never appear in the CSV |

**The CSV is a curated list of external models/libraries (a wishlist of "advanced AI models"), NOT an implementation manifest.** Claiming "we have 100 AI models" based on this file would be incorrect: only 2 of the 100 have any code in this project, and both of those are untrained (no checkpoints).

---

## 9. Documentation vs Reality (subtask 6)

| Doc | Claim | Reality (code evidence) |
|---|---|---|
| `DREAMER_IMPLEMENTATION_GUIDE.md` | "Phase 1 Complete: DreamerV3 World Model + Basic Policy" | **Unsupported** — ZERO checkpoints exist; the world model was never trained. Code exists but has never run to completion. |
| `DREAMER_IMPLEMENTATION_GUIDE.md` | `kl_balance = 0.8` | Code uses `kl_dyn_scale=0.5`, `kl_rep_scale=0.1` (different but equivalent-style balancing; minor doc/code mismatch). |
| `DREAMER_IMPLEMENTATION_GUIDE.md` | MCTS Phase 3 / adversarial Phase 4 "pending" | Matches reality: those modules are orphan stubs. |
| `COLAB_TRAINING_GUIDE.md` | "Training auto-resumes from last checkpoint" | **False for the main path** — `train/train_dreamer.py` has **no `--resume` flag** (only `train_god_mode.py` and `train_ultimate_150.py` do). |
| `COLAB_TRAINING_GUIDE.md` | Validate via `eval/crisis_validation.py` | `crisis_validation.py` is a **legacy placeholder** (MockAgent, simplified P&L ≈ 0.001; FIXES.md confirms `__NO_REFERENCES__`). |
| `DEPLOYMENT_GUIDE.md` / `FREE_DEPLOYMENT.md` | Deploy `live_trade_metaapi.py` + `train/ppo_xauusd_latest.zip` | Deploys the **legacy no-risk stack** (hardcoded VOLUME=0.01, no RiskSupervisor, no ATR) with an **absent** model zip → crashes at `PPO.load`. Applies to the old architecture; the new entry point is `live/live_trade_mt5.py`. |
| `README.md` | "STATUS: RESEARCH / NOT PRODUCTION READY"; honest artifact absence; backtest verdicts (all strategies lose to buy & hold) | ✅ **Honest and matches reality.** |
| `FIXES.md` | P0-05 fake backtester archived; P0-12 Kelly/ATR sizing wired; P1-01 full Dreamer save; P1-02 replay boundary guard; P1-03 4-tuple fix; **P1-10 research modules are orphans**; known limitation "trained checkpoint (absent)" | ✅ **Matches reality** — this file is the project's own confirmation of the orphan/untrained status. |
| `plans/plan.md` | 12 P0 fixes DONE with evidence (87 tests, 12/12 gates); "no GPU → no Dreamer/RL retraining"; "ML strategy path requires a trained checkpoint (absent) and refuses to fabricate results" | ✅ **Matches reality.** |
| `.kiro/specs/autonomous-trading-ai/design.md` | Blueprint for `AutonomousOrchestrator`, `DecisionCore`, `RegimeDetector`, `RiskLayer`, `ContinualLearner`… and explicitly: modules "exist karte hain, lekin inhe ek unified autonomous loop mein connect karna abhi baaki hai" (they exist but connecting them into one autonomous loop is still pending) | ✅ **The design doc itself confirms the models are NOT connected.** The orchestrator/decision-core/regime-detector/continual-learner it specifies **do not exist in code**. |

---

## 10. Legacy / Stale Paths (be careful not to be misled by them)

- **`live/live_trade_metaapi.py`** — LEGACY live loop (MetaApi + SB3 PPO, hardcoded volume 0.01, no risk layer). Superseded by `live/live_trade_mt5.py`. Running it today crashes at `PPO.load("train/ppo_xauusd_latest.zip")` (file absent). Deployment guides still point at it.
- **`eval/crisis_validation.py`** — LEGACY placeholder validator (MockAgent = random coin-flip; simplified P&L). No wired references.
- **`evaluate_model.py`** / **`eval/analyze_dreamer.py`** / **`eval/eval_ppo.py`** — real eval code but **exit immediately** because their default checkpoints are absent.
- **`models/__init__.py`** — empty (1-line comment); `models.*` are imported by explicit path, not package exports.

---

## 11. Online Sources Cited

**Model-family research (official papers / repos):**
1. DreamerV3 — Hafner, Pasukonis, Ba, Lillicrap. *"Mastering Diverse Domains through World Models."* **arXiv:2301.04104**; official code **https://github.com/danijar/dreamerv3**; https://danijar.com/dreamerv3
2. World Models — Ha & Schmidhuber. **arXiv:1809.01999** (foundation for model-based agents).
3. PPO — Schulman et al. *"Proximal Policy Optimization Algorithms."* **arXiv:1707.06347**; SB3 docs **https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html** (GAE λ=0.95, clip range, vf_coef, ent_coef, normalize_advantage).
4. Decision Transformer — Chen et al. **arXiv:2106.01345**; code **https://github.com/kzl/decision-transformer**.
5. AlphaZero — Silver et al. **arXiv:1712.01815** (MCTS + policy/value nets).
6. MuZero — Schrittwieser et al. **arXiv:1911.08265** (MCTS over a learned model; Nature version s41586-020-03051-4).
7. MAML — Finn, Abbeel, Levine. *"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."* **arXiv:1703.03400**.
8. Deep Ensembles — Lakshminarayanan, Pritzel, Blundell. **arXiv:1612.01474**.
9. Adversarial robustness / robust optimization — Madry et al. *"Towards Deep Learning Models Resistant to Adversarial Attacks."* **arXiv:1706.06083**; trading self-play references: **https://github.com/kayuksel/market-self-play**, **https://github.com/Aurovind7/MARL-Market-Maker**.
10. Kelly criterion — J. L. Kelly Jr. *"A New Interpretation of Information Rate."* Bell System Technical Journal **35 (1956): 917–926** (https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf); Wikipedia: *Kelly criterion* (fractional Kelly).
11. Risk management / circuit breakers — tradewink.com/learn/algorithmic-trading-risk-management; eliteforextrading.com/setting-risk-limits/; oyamori.com (daily loss limits, drawdown limits, kill switches).

**Project-internal evidence:** README.md, FIXES.md, plans/plan.md, DREAMER_IMPLEMENTATION_GUIDE.md, COLAB_TRAINING_GUIDE.md, DEPLOYMENT_GUIDE.md, FREE_DEPLOYMENT.md, `.kiro/specs/autonomous-trading-ai/design.md`, `Advanced_100_AI_Models_List.csv`, artifacts/*.txt (gate/pytest/smoke evidence).

---

## 12. Plain-Language Summary (Roman-Urdu friendly, simple English)

> **Sawal ka jawab: kya is project ke models FULLY implemented hain? → Nahee. (No.)**

**Ek line mein:** Is project mein models ki **code** likhi hui hai (kuch zyada, kuch adhoori), lekin **koi bhi model trained nahi hai** — aur jo model live trading mein chalta hai wo **AI nahi, ek simple formula** hai.

**Asaan alfaaz mein:**

1. **RiskSupervisor (risk rokne wala hissa)** — ✅ **Poora ban chuka hai aur chalta hai.** Ye AI nahi hai; ye ek "safety guard" hai jo rules lagata hai (ek din mein kitna loss ho sakta hai, kitna bada trade allowed hai, etc.). Ye live trading mein laga hua hai aur test bhi pass karta hai. **Yeh hi sirf ek cheez hai jo 100% complete hai.**

2. **Position sizing (trade kitna bada karna hai)** — ✅ ATR wala hissa **live mein laga hua hai**. Kelly wala hissa bana to hai lekin use nahi hota.

3. **DreamerV3 (jo "soch ke" trade karta hai)** — ⚠️ **Algorithm ka code sach mein bana hua hai** (world model, imagination, sab kuch), aur iske **training scripts bhi hain**. Lekin: **(a) isko kabhi train nahi kiya gaya** (koi trained file exist nahi karti), aur **(b) live trading mein iska connection bilkul nahi hai** — live system DreamerV3 ko bulata hi nahi.

4. **PPO (Stable-Baselines3)** — ⚠️ Training script **sahi bana hua hai**, aur live mein bhi iska "signal source" bana hua hai — **lekin** wo sirf tab chalta hai jab ek trained model file (`train/ppo_xauusd_latest.zip`) mojood ho. **Wo file exist nahi karti.** Is liye live mein ML path shuru hi nahi hota; default mein bot ek simple SMA 20/50 crossover (do moving averages ka formula) use karta hai.

5. **Transformer policy, MAML (meta-learning), Adversarial training (self-play)** — ❌ **Sirf skeleton hai.** Inke andar zaroori functions khaali hain (`pass`, `return None`). **Ye kabhi seekh hi nahi sakte.** Koi code inhe use bhi nahi karta.

6. **Ensemble aur MCTS** — ⚠️ Logic kuch likha hai lekin **bugs hain** (MCTS mein short trade option hi nahi; ensemble Dreamer ke saath crash karega). **Koi inhe use nahi karta** — ye sirf research files hain.

7. **100 AI models wali list (CSV)** — Ye ek **"wishlist" hai** (bahar ki websites ke models), **implementation nahi**. Is 100 mein se sirf **2** (PPO aur DreamerV3) ka code is project mein hai — aur wo bhi trained nahi.

8. **Docs (guides) ka daawa vs haqeeqat:** Guide likhti hai "Phase 1 Complete" aur "auto-resume from checkpoint" — **lekin koi checkpoint exist hi nahi karta**, aur `train_dreamer.py` mein resume ka option hai hi nahi. README.md aur FIXES.md khud **sach bolte hain**: "no trained checkpoint exists; strategies don't beat buy-and-hold."

**Aakhri nateeja (bottom line):**
- **FULLY IMPLEMENTED + WIRED:** sirf `RiskSupervisor` (aur ATR position sizing).
- **Baaki sab:** ya to **untrained** (Dreamer, PPO), ya **unwired/buggy** (Ensemble, MCTS), ya **sirf stub** (Transformer, MAML, Adversarial).
- **Is project mein koi bhi trained AI model nahi hai jo live trade kar raha ho.** Live bot abhi ek simple formula (SMA crossover) chala raha hai — AI nahi.

---

*Report generated by repository audit combining full code inspection with online verification of canonical papers/repos. Verdicts are evidence-based: "implemented" was only awarded where logic + wiring + training (or a deterministic design) are all real and connected.*
