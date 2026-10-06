# SL/TP Dual-Mode Implementation — Model-Decides vs Rule-Based

## Goal
Let the trading AI models control SL/TP (not just direction), while keeping a rule-based (ATR) SL/TP option — with a manifest/config compatibility guard so a model-trained-on-SL/TP is never run with discarded SL/TP output.

## Research Summary
- Dynamic SL/TP RL exists in literature: saeedzou/dynamic-stoploss-take-profit-rl-agent (FinRL + SB3) learns position sizing + SL delta + TP delta as a continuous Box action space, with reward penalties for SL triggers and low-profit sales, incentives for high-profit sales. Confirms SL/TP-as-action is a valid, implemented approach (RESEARCH FINDING).
- This project's DreamerV3 Actor head is a single discrete Categorical (models/dreamer_components.py Actor, action_dim=3) — continuous Box would require new actor head + distribution changes across all policies (Dreamer/PPO/Transformer/MCTS). Discrete composite action (3 directions × 5 SL buckets × 5 TP buckets = 75) keeps every policy's scalar action_dim + one-hot contract intact (ENGINEERING RECOMMENDATION, verified against models/policy.py PpoPolicy `action_space.n`, DreamerPolicy `agent.action_dim`, env `_decode_action`).
- Env semantics today: stop_loss=0.01 / take_profit=None are fixed equity-fraction params; SL/TP are checked against the bar's move in `step()` (verified env/dreamer_trading_env.py lines 48-49, 300-370). Model-decides mode will use the SAME semantics (price-fraction buckets) so training and live agree.
- User concern (verified as real): if a model trained with SL/TP actions is run in rule-based mode, its trained SL/TP output would be silently discarded → train/live behavior mismatch and degraded edge. Therefore a hard compatibility guard is required: SL/TP-trained artifacts may ONLY run in model mode; rule-trained artifacts may ONLY run in rules mode for SL/TP sourcing (config-level `SLTP_MODE`).

## Approach
- **Composite discrete action space**: 75 actions = direction (3) × SL bucket (5) × TP bucket (5). Buckets are price-fraction levels {0.5%, 1%, 2%, 3%, 5%} (SL) and {0.5%, 1%, 2%, 3%, 5%} (TP), matching the env's existing fraction semantics and live price conversion entry×(1∓frac).
- **Backward compatibility**: `sl_tp_action=False` (default) keeps the existing 3-action space, fixed env SL/TP, and current live ATR rule path untouched. Existing artifacts (action_dim 2/3) load with `sl_tp_mode="rules"` default.
- **Guard**: config `SLTP_MODE` (rules|model) + manifest `sl_tp_mode` (rules|model). Startup validation rejects mismatch (model artifact in rules mode, or rules artifact in model mode) with a clear error — this is the user's "if issue, keep only model-decides" safety answer, implemented as a hard gate rather than silent fallback.
- Live model mode: model outputs sl_frac/tp_frac → TradeExecutor clamps to [min,max] bounds and attaches SL/TP to the order (entry×(1∓frac)); rules mode: current ATR logic unchanged.

## Subtasks
1. **Env**: add `sl_tp_action: bool = False` to RealisticTradingEnv. When True: action_space = 75; `_decode_action` decodes composite idx → (direction, sl_frac, tp_frac); per-trade SL/TP from buckets replaces fixed params; `step()` uses decoded sl_frac/tp_frac for the trade (sl_hits/tp_hits stats unchanged). When False: existing behavior byte-for-byte. (verify: python unit script — backward-compat 3-action episode identical metrics; model-mode episode runs, sl/tp_hits counted, done on SL/TP hit)
2. **Dreamer agent**: Actor → composite head supporting action_dim=75 (single Categorical 75; keep unimix). ReplayBuffer action shape (75,) one-hot flows through unchanged. `act()`/`policy_probs()`/`train_step()`/`_imagine_trajectory()` operate on scalar idx as today. (verify: train 100 steps on a small random slice; losses finite, sampled actions decode to valid dir/sl/tp)
3. **Other policies**: PpoPolicy (`action_space.n` = 75) and TransformerPolicy/`DreamerMCTSPolicy` work on scalar action_dim — no changes required beyond manifest guards; add a smoke test loading a 75-action PPO stub if train_ppo supports Discrete naturally. (verify: PpoPolicy.act on 75-dim Discrete works)
4. **Manifest/artifacts**: add `sl_tp_mode: str = "rules"` to ModelManifest + write_model_artifact param; validation: sl_tp_mode=model ⇒ action_dim==75; obs_dim check unchanged. Existing manifests default to "rules". (verify: load_manifest on existing ppo_gold_v1/dreamer artifacts succeeds with sl_tp_mode=rules; a model-mode manifest round-trips)
5. **Config**: add `SLTP_MODE` (rules|model, default rules) + `sl_tp_min_frac`/`sl_tp_max_frac`/`tp_min_frac`/`tp_max_frac` bounds to BehaviorConfig; startup guard: require_promoted_model + SLTP_MODE=model ⇒ manifest.sl_tp_mode=="model" else clear ValueError; SLTP_MODE=rules + manifest.sl_tp_mode=="model" ⇒ ValueError (never silently discard). (verify: config unit test — both mismatch directions raise; match passes)
6. **TradeExecutor**: `sl_tp_mode` from cfg; model mode uses signal-provided sl_frac/tp_frac (clamped) → `entry_sl_tp`; rules mode unchanged ATR path. Order always carries sl+tp. (verify: unit test with MockBroker — model-mode entry attaches clamped SL/TP prices; rules-mode entry identical to today)
7. **ModelSignalSource / Decision**: Decision gains `sl_frac/tp_frac: float|None`; model-mode artifact decode → Decision carries bucket fractions; rules-mode artifact → None (executor uses ATR). DecisionEngine passes them through (no action change). (verify: decide() on a model-mode stub returns valid fractions; rules-mode stub returns None)
8. **train_dreamer**: add `--sl-tp-model` flag → env sl_tp_action=True, action_dim=75, manifest sl_tp_mode="model"; evaluate_policy infers sl_tp_action from policy.action_dim>3. (verify: train 200 steps on small slice with flag → artifact manifest sl_tp_mode=model, action_dim=75; eval on tiny test slice runs)
9. **Integration tests**: (a) backward-compat smoke — existing 3-action dreamer artifact + live_trade dry run in rules mode (MockBroker) unchanged; (b) model-mode end-to-end — train stub artifact → load in ModelSignalSource with SLTP_MODE=model → Decision with fractions → TradeExecutor order sl/tp attached; (c) guard tests — both mismatch combos raise at startup. (verify: all pass, exit 0)
10. **Frontend report update**: frontend_dashboard_design.md — add SL/TP dual-mode to training view (hyperparameter panel: sl_tp_action checkbox + bucket presets) and trading view (decision feed step: SL/TP source badge MODEL_DECIDED vs ATR_RULES; risk panel shows bounds clamp), with honest note that guard prevents silent discard. (verify: grep new markers present; report still passes existing checks)

## Deliverables
| File Path | Description |
|-----------|-------------|
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\env\dreamer_trading_env.py | sl_tp_action + composite decode + per-trade SL/TP |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\models\dreamer_agent.py | Actor action_dim 75 support (verify no regressions) |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\core\model_artifacts.py | manifest sl_tp_mode + validation |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\core\config.py | SLTP_MODE + bounds + guards |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\live\trade_executor.py | model-mode SL/TP attachment |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\live\decision_engine.py | Decision sl_frac/tp_frac passthrough |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\live\model_signal.py | model-mode decode |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\train\train_dreamer.py | --sl-tp-model flag |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\_parts\tests_sltp\*.py | unit + integration tests |
| e:\Desktop\NeoMind\Bazz\autonoumuse_trader\frontend_dashboard_design.md | SL/TP dual-mode spec update |

## Evaluation Criteria
- Backward compatibility: existing 3-action env episode metrics byte-identical before/after change; existing artifacts load (sl_tp_mode=rules); live rules-mode dry run unchanged.
- Model mode end-to-end: train stub (action_dim=75) → eval → ModelSignalSource decode → TradeExecutor order with clamped SL/TP attached (MockBroker verify).
- Guard: SLTP_MODE=model + rules artifact raises; SLTP_MODE=rules + model artifact raises; both clear errors.
- All new unit tests pass (exit 0); frontend report checks still pass.

## Notes
- No GPU (torch.cuda unavailable — FACT); all tests CPU, small slices only.
- Windows paths; use plain system python (E:\python_3.11.9_installed\python.exe) or .venv\Scripts\python.exe for test scripts; write part files then python-append on this path (edit_file append fails on large .md).
- Existing train scripts (train_ppo.py, train_transformer.py, etc.) are OUT OF SCOPE for model-mode training except where scalar action_dim already supports 75; documented in report as follow-up.
- Buckets: SL {0.005, 0.01, 0.02, 0.03, 0.05}; TP {0.005, 0.01, 0.02, 0.03, 0.05}; idx = dir*25 + sl_idx*5 + tp_idx.
