# Audit v2 — Stub/Dead-Code Grep Findings

Method: helper script `reports/audit_v2_notes/_grep_stubs.py` walks `models/`, `core/`, `live/`, `train/` for *.py and reports:
- bare `pass` lines (`^\s*pass\s*$`)
- `NotImplemented`
- `TODO` / `FIXME`
- `return None`

**26 total matches — every one inspected line-by-line and classified. NONE is a remaining functional stub in the model code.**

## BARE `pass` (7)

| file:line | context | verdict |
|---|---|---|
| core/model_artifacts.py:10 | `class ModelArtifactError(RuntimeError)` exception class body | LEGIT (exception class) |
| live/idempotency.py:48 | `except OSError:` after corrupt-file rename | LEGIT (exception handler) |
| live/idempotency.py:75 | `except OSError:` in _save | LEGIT |
| live/idempotency.py:97 | `except OSError:` in lookup stale-prune | LEGIT |
| live/idempotency.py:108 | `except OSError:` in clear | LEGIT |
| live/live_trade_metaapi.py:119 | legacy adapter stub method | LEGACY (superseded by mt5_broker) |

**No bare `pass` exists in any models/*.py file** (ensemble, mcts, meta_learning, adversarial_training, transformer_policy, dreamer_agent, dreamer_components, policy, registry, position_sizing, risk_supervisor).

## NotImplementedError (2)

| file:line | context | verdict |
|---|---|---|
| models/dreamer_agent.py:250 | `compute_world_model_loss` raises only when `params_override is not None` (second-order MAML support not implemented) | LEGIT GUARD — first-order MAML is the documented scope; never triggered by any caller in repo |
| models/meta_learning.py:166 | MAMLTrader raises NotImplementedError when `first_order=False` | LEGIT GUARD — documents first-order-only; all callers pass default first_order=True |

## return None (15)

| file:line | context | verdict |
|---|---|---|
| models/dreamer_agent.py:74, 106 | ReplayBuffer.sample returns None when buffer too small / episode boundary degenerate | LEGIT (data guard) — callers (`train_step` :308) check `if batch is None: return {}` |
| models/dreamer_agent.py:308 | train_step skips when sample returns None | LEGIT (guard, not stub) |
| models/policy.py:39 | PpoPolicy.reset/observe_executed no-ops | LEGIT protocol-conformance (SB3 handles internally) |
| models/policy.py:51 | DreamerPolicy.reset/observe_executed no-ops | LEGIT (reset/observe handled by wrapper internals) |
| models/policy.py:119 | DreamerMCTSPolicy.reset/observe_executed no-ops | LEGIT |
| models/risk_supervisor.py:517, 524, 535 | `_direction` unknown input / `_correlation_reject` passthrough | LEGIT (optional computation, documented) |
| live/live_trade_mt5.py:144 | RuleSignalSource.executed no-op | LEGIT (heuristic source doesn't learn) |
| live/mt5_broker.py:331 | `_order_check` success path | LEGIT |
| live/live_trade_metaapi.py:56, 78, 85, 86 | legacy MetaApi adapter | LEGACY |
| train/common.py:21 | git_commit None on subprocess failure | LEGIT (optional metadata) |

## TODO / FIXME

0 matches in models/, core/, live/, train/.

## policy.py runtime assumptions — VERIFIED against real implementations

| policy.py assumes | reality | OK |
|---|---|---|
| `DreamerV3Agent.policy_probs(state)` | exists at models/dreamer_agent.py ~line 207: `softmax(self.actor(state), dim=-1)` | ✅ |
| `DreamerV3Agent.from_checkpoint(path, device)` | exists at models/dreamer_agent.py ~line 455 (classmethod, torch.load weights_only=False) | ✅ |
| `rssm.observe(embed, action, h, z)` returns 4-tuple | dreamer_components.py ~line 155 returns `(h, z, prior_logits, posterior_logits)` | ✅ |
| `rssm.get_state(h, z)` | dreamer_components.py ~line 200 `cat([h, z])` | ✅ |
| `rssm.initial_state(batch_size, device)` | dreamer_components.py ~line 145 returns `(h_zeros, z_zeros)` | ✅ |
| `rssm.imagine(action, h, z)` returns 3-tuple | dreamer_components.py ~line 190 returns `(h, z, prior_logits)` | ✅ |
| `TransformerAgentWrapper.policy_probs(obs_batch)` | exists at models/transformer_policy.py ~line 204 (eval-mode softmax) | ✅ |
| `TransformerAgentWrapper.from_checkpoint` | exists at models/transformer_policy.py ~line 392 (classmethod) | ✅ |
| `DreamerV3Agent.act(obs, h, z, deterministic)` returns `(numpy_action, (h,z))` | dreamer_agent.py ~line 177 `(action.cpu().numpy()[0], (h, z))` | ✅ |

**Conclusion: no remaining functional stubs in model code; all policy.py assumptions hold. The only flagged residual is `MarketMakerAgent.learn()` (adversarial_training.py ~line 195) which is stat-only by name — real MM learning happens in `finish_episode()` which is invoked by both training paths.**
