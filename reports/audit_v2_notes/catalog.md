# Audit v2 — File/Line Catalog of Model & Wiring Modules

Audit date: re-audit after commits 9fb216c, cd134c8, 07c2472, d8fdd6b, c21f2b2, 79157b7, 3daf65c.
All paths relative to repo root. Line numbers verified by full file reads in this audit session.

## 1. models/policy.py (170 lines) — shared policy interface

| Symbol | Lines | Notes |
|---|---|---|
| `PolicyOutput` dataclass | ~10 | frozen; action:int, probs:np.ndarray, confidence:float, info:dict |
| `TradingPolicy` Protocol | ~16 | runtime_checkable; action_dim, obs_dim, reset(), act(obs)->PolicyOutput, observe_executed(action) |
| `PpoPolicy` | ~25 | wraps SB3 PPO.load(model_path, device="cpu"); act uses get_distribution probs argmax; reset/observe_executed no-ops |
| `DreamerPolicy` | ~45 | wraps DreamerV3Agent; reset sets h/z/prev_action None + eval mode; act: encoder→initial_state fallback→rssm.observe 4-tuple→get_state→agent.policy_probs→symexp(critic); observe_executed sets one-hot prev_action |
| `TransformerPolicy` | ~95 | wraps TransformerAgentWrapper; act → agent.act(deterministic=True) + agent.policy_probs; observe_executed range-checks |
| `DreamerMCTSPolicy` | ~118 | lazy `from models.mcts import DreamerMCTSAgent` in __init__; act → (one_hot, _) = agent.act(obs), action=argmax, probs from last_stats visit_distribution |

## 2. models/registry.py (91 lines) — artifact loader

| Symbol | Lines | Notes |
|---|---|---|
| `load_policy(manifest_path, device="cpu")` | ~1 | entry point → _load_policy(Path, device, set()) |
| `_member_manifest_path(directory, member)` | ~20 | resolves member dir→manifest.json |
| `_load_policy` | ~27 | resolve+cycle-detect (seen set, ModelArtifactError on recursion); load_manifest; dispatch ppo/dreamer/transformer/dreamer_mcts/ensemble; final obs_dim/action_dim validation |

## 3. models/ensemble.py (266 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `EnsemblePolicy` | ~7 | __init__ validates members non-empty, vote∈{soft,hard}, min_agreement∈[0,1], action_dim=max member capped ≤3, same obs_dim, weights normalized |
| `EnsemblePolicy.reset` | ~55 | delegates to members |
| `EnsemblePolicy.observe_executed(action)` | ~58 | clamps to member action_dim before delegating |
| `EnsemblePolicy._member_prediction(member, obs)` | ~64 | HANDLES PolicyOutput / tuple ([0]) / scalar (one-hot) / probs array; validates width/validity; pads to ensemble action_dim |
| `EnsemblePolicy.act` | ~99 | soft avg or hard weighted counts; agreement; consensus gate → action 0 below min_agreement; entropy/KL uncertainty; returns PolicyOutput |
| `EnsembleAgent` | ~137 | seed 42+index, hidden_dim+=index*16; act collects results, lazily builds _members via _LegacyMemberPolicy; hard vote |
| `EnsembleAgent.train/save/load` | ~176 | save `f"{prefix}_model{index}.pt"` |
| `_LegacyMemberPolicy` | ~243 | wraps a model; pending_result consumed once on first act |

## 4. models/mcts.py (280 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `MinMaxStats` | ~7 | min/max normalize |
| `MCTSNode` | ~19 | h, z, prior, parent, action, reward; select_child PUCT; backup |
| `MCTS` | ~58 | __init__(agent, num_simulations, c_puct, gamma, dirichlet_alpha, dirichlet_fraction, root_noise, seed); actions=np.eye(action_dim) — FULL 3-action set |
| `MCTS._set_eval` | ~67 | encoder/rssm/actor/critic/reward_predictor eval |
| `MCTS._expand` | ~77 | rssm.get_state, actor softmax priors, rssm.imagine, symexp(reward) |
| `MCTS._add_root_noise` | ~100 | dirichlet |
| `MCTS.backup` | ~110 | reward+gamma accumulation |
| `MCTS.search(h, z)` | ~120 | → (action, stats) incl visit_counts/q_values/probs/visit_distribution/root_visits |
| `MCTS.search_with_stats` | ~140 | alias |
| `DreamerMCTSAgent` | ~186 | __init__ stores mcts and calls self.reset() (~199) setting h/z/prev_action/last_stats None + mcts._set_eval() |
| `DreamerMCTSAgent.act` | ~208 | encoder; rssm.initial_state fallback; previous_action zeros fallback; rssm.observe 4-tuple; use_mcts→search else actor.sample deterministic + synthetic stats; stores h/z/prev_action |
| `DreamerMCTSAgent.observe_executed` | ~258 | validates range; sets one-hot prev_action |

## 5. models/meta_learning.py (306 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `build_task_buffer(agent, env_segment, steps, seed)` | ~9 | RealisticTradingEnv, swaps agent.replay_buffer to larger, runs act/step |
| `MarketRegimeGenerator._labels(returns)` | ~52 | rolling 120 trend/vol, rolling 48 vol, expanding median → 5 labels |
| `MarketRegimeGenerator._find_label_periods` | ~88 | REAL run-length impl → [(start,end)] |
| `_find_trending/_ranging/_volatile_periods` | ~112-142 | delegate to _find_label_periods |
| `MarketRegimeGenerator.generate_regimes` | ~145 | tasks with 70/30 support/query split; min_len=512 default; validates dims |
| `MAMLTrader` | ~184 | meta_lr/adapt_lr/adapt_steps/first_order; raises NotImplementedError only for second-order |
| `MAMLTrader._sample_batch(buffer, batch_size)` | ~212 | ReplayBuffer sample→dict; dict data: validates episodes, valid non-done-crossing starts, returns numpy arrays; RAISES ValueError on empty (never None) |
| `MAMLTrader._adapt_copy` | ~233 | autograd.grad over world_model_params, manual add_ alpha=-adapt_lr |
| `MAMLTrader.meta_train` | ~248 | deepcopy base, build_task_buffer, adapt, query loss, accumulate grads, clip 1000, optimizer.step |
| `MAMLTrader.fast_adapt` | ~295 | → adapted deepcopy |

## 6. models/adversarial_training.py (333 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `MarketMakerAgent` | ~8 | requires action_dim==4; MLP(state+10→128→128→4); 10 pattern features from deque(maxlen=100) |
| `MarketMakerAgent.select_action` | ~95 | appends trader_action, builds obs, softmax, max_manipulation_rate cap, Categorical sample, counters |
| `MarketMakerAgent.respond` | ~147 | alias |
| `MarketMakerAgent.observe_reward` | ~150 | stores log_prob/reward |
| `MarketMakerAgent.finish_episode` | ~158 | REINFORCE with baseline; gamma 0.99; advantage=returns-baseline; clip 1.0; baseline EMA 0.9 — REAL MM training |
| `MarketMakerAgent.learn(reward)` | ~195 | STAT-ONLY (total_profit += reward, successful_traps += 1) — residual misnomer; real training is finish_episode() |
| `AdversarialTradingEnv` | ~208 | wraps base_env+mm; step applies perturbation {spread_mult, slippage_mult, adverse_gap}; base_env.set_perturbation + step; mm_reward = -trader_reward - manip_cost; reset calls mm.begin_episode |
| `SelfPlayTrainer` | ~260 | trader_agent + mm_agent + env; train_every/batch_size |
| `SelfPlayTrainer._trader_action` | ~290 | trader.act(obs,h,z,deterministic) unpack tuple |
| `SelfPlayTrainer._train_trader` | ~300 | ACTIVE: adv_env.reset, act/step, replay_buffer.add, calls `metrics = self.trader.train_step(batch_size=...)` every train_every, increments trader_updates; mm.finish_episode at episode end |
| `SelfPlayTrainer._train_mm` | ~330 | deterministic trader; collects info["mm_profit"]; finish_episode per end |
| `SelfPlayTrainer.train` | ~346 | alternates trader/mm; history {trader_wins, mm_profits, epochs, trader_updates} |

## 7. models/transformer_policy.py (404 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `tokenize_observation(observation, window, n_features)` | ~15 | validates flat width window*n_features+5; tokens (window, n_features+5) with account 5-vec broadcast |
| `compute_gae(rewards, values, dones, last_value, gamma, lam)` | ~49 | full GAE |
| `PositionalEncoding` | ~62 | |
| `_TransformerEncoder` | ~98 | with attention_weights extraction |
| `TransformerActor` | ~137 | last-token head → logits |
| `TransformerCritic` | ~157 | scalar value |
| `TransformerAgentWrapper` | ~173 | obs_dim = window*n_features+5 validated; seq_len=window; PPO hyperparams; config dict persisted |
| `TransformerAgentWrapper._tokens` | ~197 | |
| `TransformerAgentWrapper.evaluate_actions` | ~200 | log_prob, entropy, value |
| `TransformerAgentWrapper.policy_probs(obs_batch)` | ~204 | eval-mode softmax |
| `TransformerAgentWrapper.act(obs, deterministic=True)` | ~217 | (int action, float logp, float value) |
| `TransformerAgentWrapper.get_attention_weights` | ~245 | real (2, layers, seq, seq) from _TransformerEncoder |
| `TransformerAgentWrapper.train_step(batch)` | ~286 | FULL PPO: standardized advantages, minibatch loop, ratio=exp(logp-old_logp), clipped surrogate, value MSE, entropy bonus, grad clip; returns metrics dict |
| `TransformerAgentWrapper.save/load` | ~350 | {config, actor, critic, optimizer, training_step} |
| `TransformerAgentWrapper.from_checkpoint` | ~392 | classmethod; torch.load weights_only=False; cls(**config) |

## 8. models/dreamer_agent.py (565 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `ReplayBuffer` | ~18 | numpy ring buffer cap 100k, seq_len 64; sample→None if size<seq_len+1; episode-boundary guard (prefix-sum done counts); vectorized fancy-index gather → {obs,action,reward,done} |
| `DreamerV3Agent.__init__` | ~118 | full networks (encoder/rssm/decoder/reward_predictor/actor/critic/slow_critic); 3 optimizers; replay_buffer; return_low/high None |
| `DreamerV3Agent.act(obs,h,z,deterministic)` | ~177 | resets to initial_state when None; rssm.observe 4-tuple; actor.sample; stores prev_action; returns (action.cpu().numpy()[0], (h,z)) |
| `DreamerV3Agent.policy_probs(state)` | ~207 | torch.softmax(self.actor(state), dim=-1) — matches policy.py assumption |
| `DreamerV3Agent.compute_world_model_loss` | ~210 | NotImplementedError guard ONLY for params_override; T-loop reset on done; rssm.observe; KL; decoder recon MSE vs symlog; reward MSE |
| `DreamerV3Agent.train_step(batch_size)` | ~260 | PHASE1 world-model backward clip 1000; PHASE2 imagined trajectories, λ-returns, advantages normalize, critic MSE + slow-critic EMA, actor REINFORCE + entropy; returns metrics |
| `_imagine_trajectory` | ~330 | |
| `_lambda_returns` | ~370 | |
| `_normalize_returns` | ~388 | EMA decay 0.99, quantile 5/95, scale clamp min 1.0 |
| `_update_slow_critic` | ~398 | lerp tau |
| `DreamerV3Agent.save` | ~405 | config + all 6 nets + 3 optimizers + return norm + training_step + torch/numpy RNG |
| `DreamerV3Agent.from_checkpoint(cls, path, device)` | ~455 | torch.load weights_only=False; requires config; cls(device=device, **config) then load |
| `DreamerV3Agent.load` | ~468 | restores nets, optimizers if present, RNGs, prints "Loaded checkpoint from step N" |

## 9. models/dreamer_components.py (351 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `symlog` / `symexp` | ~7-30 | symexp exp clamp max 20.0 |
| `unimix_logits(unimix=0.01)` | ~32 | |
| `RMSNorm(dim)` | ~45 | |
| `GRUCell(input_size, hidden_size)` | ~60 | 3 gates W_ir/hr, norm_r |
| `Encoder(obs_dim, embed_dim)` | ~95 | symlog → MLP 512/RMSNorm/SiLU x2 → embed_dim |
| `RSSM` | ~120 | prior_net, posterior_net, gru |
| `RSSM.initial_state(batch_size, device)` | ~145 | (h zeros, z zeros) |
| `RSSM.observe(embed, action, h_prev, z_prev)` | ~155 | h=gru(cat[z,action]); posterior logits unimixed; z = one-hot categorical (argmax in eval); returns (h, z, prior_logits, posterior_logits) — 4-TUPLE |
| `RSSM.imagine(action, h_prev, z_prev)` | ~190 | (h, z, prior_logits) — 3-TUPLE |
| `RSSM.get_state(h,z)` | ~200 | cat([h,z]) |
| `RSSM.kl_loss` | ~205 | free nats + dyn/rep scales |
| `Decoder(state_dim, obs_dim)` | ~260 | symlog-space recon |
| `RewardPredictor(state_dim)` | ~275 | 1, symlog |
| `Actor(state_dim, action_dim)` | ~290 | logits; dist unimix Categorical; sample one-hot |
| `Critic(state_dim)` | ~335 | 1, symlog |

## 10. core/model_artifacts.py (94 lines)

| Symbol | Lines | Notes |
|---|---|---|
| `ModelArtifactError(RuntimeError)` | ~9 | bare pass (legit exception class) |
| `ModelManifest` dataclass | ~13 | all fields incl members=[], extra={} |
| `artifact_dir(root, model_type, run_name)` | ~35 | |
| `_read_and_validate(path)` | ~45 | pops manifest_version, validates version, model+contract file existence, contract["hash"]==manifest.contract_hash, obs_dim==window*n_features+5 |
| `save_manifest` | ~65 | writes then re-reads to validate |
| `load_manifest` | ~85 | |

## 11. core/observation.py (39 lines)

`AccountState` dataclass + `to_vector()` (5 floats) + `build_observation(features_window, account)` + `obs_dim(window, n_features)` = window*n_features+5.

## 12. core/config.py (561 lines)

FeatureConfig, RiskConfig (with __post_init__ validation), CostConfig, BrokerConfig, PathConfig (model_path legacy train/ppo_xauusd_latest.zip; manifest_path artifacts/models/production/manifest.json), ModelConfig, TradingBehaviorConfig, AppConfig, load_config(_env), check_live_gates, ensure_trading_allowed.

## 13. live/ modules

| File | Key symbols |
|---|---|
| live/model_signal.py (130) | ModelSignalSource: load_policy→manifest; contract hash check; action_dim∈(2,3); macro requirement; burn-in; decide() with INSUFFICIENT_HISTORY guards; executed(action) |
| live/decision_engine.py (151) | Decision dataclass; DecisionEngine.filter() → confidence/consensus/session/cooldown/Kelly scaling |
| live/trade_executor.py (601) | TradeExecutor: local state, reconciliation, ATR sizing, SL/TP, check_trade on every entry, idempotent entry token |
| live/trade_manager.py (105) | TradeManager: breakeven/trailing/partial close/time stop |
| live/live_trade_mt5.py (952) | LiveTrader entry point; RuleSignalSource; ModelSignalSource path; _build_broker gate (promotion); model_promotion_failures requires evaluation.json passed==True + contract_hash match |
| live/idempotency.py (108) | IdempotencyGuard persisted token→ticket; bare passes are exception handlers only |

## 14. train/ scripts

| File | Purpose |
|---|---|
| train/common.py (71) | write_model_artifact — contract + manifest + save |
| train/data.py (89) | load_bars, prepare_data (train-only scaler fit), PreparedData |
| train/evaluate.py (189) | evaluate_policy, promotion_gate (sharpe≥0.5, maxDD≤20, trades≥20, return>0), write_evaluation, main |
| train/train_ppo.py | PPO baseline trainer (SB3) |
| train/train_dreamer.py | Dreamer trainer (+resume) |
| train/train_transformer.py (219) | Transformer trainer; rollouts, GAE, train_step, artifact+eval |
| train/train_ensemble.py (190) | trains/assembles ensemble members; contract-hash equality check; writes ensemble.json + manifest + eval |
| train/train_adversarial.py (156) | SelfPlayTrainer fine-tune of a Dreamer artifact |
| train/meta_train_dreamer.py (148) | MarketRegimeGenerator + MAMLTrader.meta_train; writes new artifact |
| train/adapt_recent.py (144) | build_task_buffer + MAMLTrader.fast_adapt on recent bars; writes new artifact |
| train/make_mcts_manifest.py (72) | dreamer_mcts manifest referencing existing Dreamer checkpoint |
| train/load_bars via train/data | shared loader |

## Catalog entry count

~90 class/function entries across the 9 model files + supporting core/live/train modules.
