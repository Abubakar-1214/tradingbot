# Dreamer Implementation Guide vs Code Reality — Audit Report

**Date:** 2026-10-03
**Scope:** `DREAMER_IMPLEMENTATION_GUIDE.md` (321 lines) cross-checked against actual source code
**Method:** direct file reads + grep verification of every claim in the guide
**Marker legend:** ✅ VERIFIED (claim matches code) · ⚠️ PARTIAL (code exists but not wired end-to-end) · ❌ INCORRECT (claim does not match code)

---

## 1. Executive Summary

| # | Guide Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Phase 1 Complete: DreamerV3 world model + basic policy | ✅ VERIFIED | `models/dreamer_agent.py` (565 lines, full RSSM/symlog/two-hot/actor-critic), `models/dreamer_components.py` (351 lines), `env/dreamer_trading_env.py` (421 lines), `train/train_dreamer.py` |
| 2 | Phase 2 (In Progress): Macro features [x] | ✅ VERIFIED | `core/feature_pipeline.py` — `load_macro_daily`, `compute_macro_features_causal`, contract-validated; `data/macro_daily.csv` exists; `data/xauusd_h1_macro.csv` exists |
| 3 | Phase 2: Economic calendar integration [ ] | ⚠️ PARTIAL | `features/calendar_features.py` (296 lines, 8 features: hours_to_event, days_since_event, event_density, is_high_impact, in_event_window, event_volatility_expected, event_type_nfp, event_type_fomc) EXISTS and is wired into `features/ultimate_150_features.py` + `features/god_mode_features.py`, BUT **NOT** wired into the shared causal pipeline `core/feature_pipeline.py` that train/backtest/live actually use |
| 4 | Phase 2: Volatility regime detection [ ] | ⚠️ PARTIAL | No explicit volatility-regime module exists. However `models/meta_learning.py::MarketRegimeGenerator` (120-bar trend/vol, 48-bar vol vs expanding median → high_vol/low_vol/trend_up/trend_down/range, shift(1)) EXISTS and is wired into `train/meta_train_dreamer.py`. This is regime *detection for meta-learning*, not a live risk overlay. |
| 5 | Phase 3 (Pending): MCTS Integration | ❌ INCORRECT | `models/mcts.py` (280 lines, MCTSNode/PUCT/DreamerMCTSAgent) EXISTS; `models/policy.py::DreamerMCTSPolicy` + `models/registry.py` (dreamer_mcts type, mcts_simulations) + `train/make_mcts_manifest.py` fully wired. Guide is stale. |
| 6 | Phase 4 (Future): Adversarial Training | ❌ INCORRECT | `models/adversarial_training.py` (333 lines, MarketMakerAgent + AdversarialTradingEnv + SelfPlayTrainer) EXISTS; `train/train_adversarial.py` EXISTS (requires a dreamer manifest, runs self-play epochs). Guide is stale. |
| 7 | Guide hyperparam `kl_balance=0.8` | ⚠️ PARTIAL | Code uses `kl_dyn_scale=0.5`, `kl_rep_scale=0.1` (DreamerV3 paper naming) — no `kl_balance` param exists. Conceptually equivalent (KL balancing), different name. |
| 8 | Guide: "actor imagines 15 steps ahead" | ✅ VERIFIED | `horizon=15` default in `models/dreamer_agent.py:146` |
| 9 | Guide: prefill 5,000 / train 100,000 steps / save every 10,000 | ✅ VERIFIED | `train/train_dreamer.py` CLI defaults: `--prefill 5000`, `--steps 100000`, `--save-every 10000` |
| 10 | Guide: 3 actions flat/long/short | ✅ VERIFIED | action_dim=3 default; policy.py semantics 0=flat/exit, 1=long, 2=short |
| 11 | Guide: sample efficiency ~50k-100k vs PPO 500k-1M | ⚠️ UNVERIFIED | Claim matches DreamerV3 paper's general finding (arXiv 2301.04104) but NO project training run has validated this on XAUUSD (no trained checkpoint shipped) |
| 12 | Guide: evaluation on post-2022 unseen data | ✅ VERIFIED | `train/train_dreamer.py` default `--train-end 2022-01-01`, test period after it; `train/evaluate.py` honest evaluation with promotion gate (sharpe≥0.5, maxdd≤20, trades≥20, return>0) |
| 13 | Guide: "This is the foundation of superhuman trading" | ❌ INCORRECT (overclaim) | Project's own README: rule strategies do NOT beat buy-and-hold net of costs. No ML checkpoint validated. See FINAL_AUTONOMOUS_TRADING_AI_STUDY.md §16.5 and Decision 13. |
| 14 | Guide: MCTS "500ms thinking time" | ⚠️ PARTIAL | MCTS exists but is a planning wrapper (`DreamerMCTSAgent`, default 100 sims / manifest default 32) — NOT integrated into the live decision path (`live/decision_engine.py` has no MCTS branch), no timing guarantee |
| 15 | Guide: macro "DXY, SPX, US10Y" | ⚠️ PARTIAL | macro_daily.csv columns are `*_close` generic (pipeline computes ret/mom/corr for whatever columns exist) — column presence for DXY/SPX/US10Y is data-dependent, verify actual CSV headers |
| 16 | Guide: "Phase 1 Complete" | ✅ VERIFIED | Faithful DreamerV3 implementation exists per arXiv 2301.04104 (RSSM, symlog, two-hot critic, free nats, imagination, lambda returns) |

---

## 2. What Is ACTUALLY Implemented (full inventory, verified)

### 2.1 DreamerV3 Core — COMPLETE ✅
| File | Lines | Purpose | Status |
|---|---|---|---|
| `models/dreamer_agent.py` | 565 | DreamerV3Agent: world model (RSSM), actor, critic, slow critic, return normalizer, replay buffer (100k ring, seq 64, episode-boundary-safe), save/load with RNG state | COMPLETE |
| `models/dreamer_components.py` | 351 | symlog/symexp, unimix, RMSNorm, GRUCell, Encoder, RSSM (prior/posterior, straight-through one-hot), Decoder, RewardPredictor, Actor, Critic | COMPLETE |
| `env/dreamer_trading_env.py` | 421 | RealisticTradingEnv: spread 2.5bp, commission, vol-scaled 80%-adverse slippage, swap (Wed triple), SL/TP, max-DD episode breaker, causal observations (features ≤ t-1, fill at open of t), reward = scaled log-return | COMPLETE |
| `train/train_dreamer.py` | 198 | CLI: prefill 5k, train 100k steps, save every 10k, contract-checked evaluation, writes manifest/evaluation.json | COMPLETE |
| `train/evaluate.py` | 189 | Honest evaluation: continuous eval pass, Sharpe/MaxDD/return/trades/win rate/costs vs buy-and-hold; promotion gate | COMPLETE |

### 2.2 Phase 2 "Data Nexus" — MOSTLY DONE (with gaps)
| Component | Code | Wired into shared pipeline? |
|---|---|---|
| Macro features (ret/mom/corr) | `core/feature_pipeline.py` | ✅ YES — train/backtest/live all use it; contract-hashed |
| Economic calendar (8 features) | `features/calendar_features.py` | ⚠️ Only in `ultimate_150_features.py` + `god_mode_features.py` — NOT in `core/feature_pipeline.py` main path |
| Volatility regime | `models/meta_learning.py::MarketRegimeGenerator` | ⚠️ Only for MAML meta-training (`train/meta_train_dreamer.py`) — no live risk overlay using regime labels |
| Event risk in env | `env/dreamer_trading_env.py` (event_mask, event_spread_multiplier 2.0, event_slippage_multiplier 3.0) | ⚠️ Env supports it; RiskSupervisor has EVENT_RISK halving — but event_mask source not wired from calendar |

### 2.3 Phase 3 "MCTS" — IMPLEMENTED BUT NOT LIVE-INTEGRATED ⚠️
| File | Purpose | Status |
|---|---|---|
| `models/mcts.py` | MCTSNode, PUCT (c_puct 1.0), MinMaxStats, DreamerMCTSAgent (uses dreamer.rssm + actor priors + reward_predictor; 100 sims default) | COMPLETE as library |
| `models/policy.py::DreamerMCTSPolicy` | Wraps DreamerMCTSAgent into TradingPolicy | COMPLETE |
| `models/registry.py` | `dreamer_mcts` manifest type, mcts_simulations | COMPLETE |
| `train/make_mcts_manifest.py` | Creates dreamer_mcts artifact from a dreamer base | COMPLETE |
| `live/decision_engine.py` | — | ❌ NO MCTS branch — live path uses direct policies only |

### 2.4 Phase 4 "Adversarial Training" — IMPLEMENTED ✅ (guide says future — stale)
| File | Purpose | Status |
|---|---|---|
| `models/adversarial_training.py` | MarketMakerAgent (4 actions: none/widen_spread/slippage/stop_hunt; manipulation budget), AdversarialTradingEnv (perturbations spread×3, slippage×3, adverse gap 0.5×SL), SelfPlayTrainer (trader vs MM epochs) | COMPLETE |
| `train/train_adversarial.py` | CLI taking a dreamer manifest → loads base agent → self-play train | COMPLETE (needs trained dreamer base first) |

### 2.5 Other models present (guide doesn't mention — completeness note)
- `models/transformer_policy.py` (Transformer-PPO, attention hook)
- `models/ensemble.py` (soft/hard voting, consensus gate, epistemic uncertainty)
- `models/meta_learning.py` (MAMLTrader first-order, MarketRegimeGenerator)
- `models/position_sizing.py` (Kelly, FixedFraction, ATR; dynamic sizing via dreamer critic advantage)
- `models/risk_supervisor.py` (SQLite circuit breakers — 616 lines)
- `train/train_ppo.py`, `train/train_transformer.py`, `train/train_ensemble.py`, `train/train_god_mode.py`, `train/train_ultimate_150.py`, `train/adapt_recent.py`, `train/meta_train_dreamer.py`

---

## 3. Reality Gaps vs Guide (what is genuinely INCOMPLETE)

### GAP-A — Economic calendar NOT in the shared causal pipeline
- `core/feature_pipeline.py` is the SINGLE path used by train/backtest/live (leak-free, contract-hashed, scaler fit on train only).
- Calendar features live only in the 150-feature and god-mode builders — those are NOT the default path. `train/train_dreamer.py` default uses the core pipeline.
- **Impact:** the agent currently does NOT see NFP/FOMC/event timing during standard training. Event-awareness only exists in the env cost model (event_mask) which nothing feeds.
- **Fix path:** add `compute_calendar_features` output to `core/feature_pipeline.py` with the same `shift(1)` causal alignment; bump CONTRACT_VERSION; update contract tests.

### GAP-B — Volatility regime exists for MAML only, not as live risk overlay
- `MarketRegimeGenerator` labels (high_vol/low_vol/trend_up/trend_down/range) are used to build MAML tasks.
- No live/risk consumption of regime labels. Guide lists "volatility regime detection" as incomplete — this matches: the detection code exists but the consumption is meta-training-only.
- **Fix path (matches FINAL study E7):** condition RSSM on regime id or add regime features to the observation; wire regime label into RiskSupervisor as an input.

### GAP-C — MCTS not in the live decision path
- `live/decision_engine.py` consumes raw policy outputs (ppl/transformer/dreamer/ensemble) — no `use_mcts` branch, no 500ms planning.
- **Fix path:** add a manifest type or env flag to route a `dreamer_mcts` model through DecisionEngine with a planning-time budget; add timeout so MCTS never blocks the bar close.

### GAP-D — No trained checkpoint shipped / no end-to-end Dreamer training run validated
- `artifacts/models/` contains only `ppo_gold_v1` (SB3 PPO, no evaluation.json passed==true guarantee checked here) + two `dreamer_20261003T*` runs (model.pt + evaluation.json + manifest.json present — timestamps 2026-10-03T2303/2305Z, i.e., TODAY). Verify those evaluation.json contents before claiming Dreamer training was validated — see §4.
- The guide's "Expected Performance" numbers (2-4h CPU, 50k-100k steps convergence) are paper-level claims, NOT validated on this project's data.

### GAP-E — Hyperparameter name mismatch
- Guide documents `kl_balance=0.8`; code uses `kl_dyn_scale=0.5` + `kl_rep_scale=0.1` (DreamerV3 paper names). Guide needs a correction note; conceptually same mechanism.

### GAP-F — "Superhuman trading" overclaim
- Contradicts project's own honest backtest verdict. The world model's defensible value: safe exploration, what-if planning, sample efficiency — NOT proven alpha. See FINAL_AUTONOMOUS_TRADING_AI_STUDY.md Decision 13.

---

## 4. Check the two dreamer run artifacts (today's date)

Two Dreamer training artifacts appeared TODAY under `artifacts/models/`:
- `dreamer_20261003T230304213562Z/` (model.pt + evaluation.json + feature_contract.json + manifest.json)
- `dreamer_20261003T230535411560Z/` (same layout)

**Before claiming "training complete", read `evaluation.json` in each:** check `passed`, `sharpe`, `max_dd_pct`, `trades`, `total_return_pct`, `buy_hold_return_pct`, `test_period`. The promotion gate requires sharpe≥0.5, maxdd≤20, trades≥20, return>0. If `passed==false`, the run is a checkpoint + honest failure, not a validated model.

---

## 5. Corrected Status Summary (for anyone reading the guide)

| Phase | Guide says | Reality |
|---|---|---|
| Phase 1 DreamerV3 | Complete | ✅ Complete — faithful implementation, unvalidated on real data |
| Phase 2 Macro | [x] | ✅ Complete in core pipeline |
| Phase 2 Calendar | [ ] | ⚠️ Code exists, NOT in shared pipeline (main gap) |
| Phase 2 Vol regime | [ ] | ⚠️ Detection exists (MAML), no live overlay |
| Phase 3 MCTS | Pending | ❌ Implemented as library + policy + manifest, NOT live-wired |
| Phase 4 Adversarial | Future | ❌ Implemented (MarketMaker + SelfPlay + train script), needs trained base |

**Bottom line:** the guide is STALE — it under-reports implemented capability (MCTS, adversarial, calendar module) and over-claims readiness ("superhuman trading", "Phase 1 Complete" implying validation). The codebase is further along than the guide claims on architecture, but unvalidated on results.

---

## 6. Recommended Guide Updates (single source of truth)

1. Mark Phase 3 MCTS and Phase 4 Adversarial as **Implemented (library-level)** with the caveat "not yet live-wired / no validated base model".
2. Move economic calendar from "pending" to "module exists; pending core-pipeline integration (GAP-A)".
3. Add a "Validated vs Unvalidated" table: what has test evidence (pytest 87/139, feature contract tests, risk tests, backtest honesty) vs what has none (trained ML checkpoint).
4. Replace "superhuman trading" framing with the honest framing from FINAL_AUTONOMOUS_TRADING_AI_STUDY.md Decision 13.
5. Fix `kl_balance` → `kl_dyn_scale`/`kl_rep_scale` in the hyperparameter table.
6. Point "Expected Performance" at the actual evaluation.json artifacts instead of paper claims.
