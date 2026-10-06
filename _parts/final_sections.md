
---

## 8. Non-functional requirements

### 8.1 Deployment shape

- **Single-user local application.** The dashboard binds to `127.0.0.1` on a fixed port; the SPA is served by the same FastAPI process. Optional bearer token (generated at first launch, stored in `.env`) for browser access. **[ENGINEERING RECOMMENDATION]**
- **Two processes total at runtime:** (1) the FastAPI backend (supervisor + readers + WS hub), (2) zero-or-one child (training job or live loop) — never both simultaneously (the concurrency guard covers training; the supervisor also refuses `POST /api/live/start` while a training job runs). **[ENGINEERING RECOMMENDATION]**

### 8.2 WebSocket vs polling decision

- **WebSocket for live feeds:** loss-curve ticks (per train step), env ticker, decision feed, risk deltas, log tail, checkpoint events — high-frequency or push-on-change; polling would lag or hammer the readers. **[ENGINEERING RECOMMENDATION]**
- **Polling fallback (REST) for slow panels:** model list, evaluation timeline, prep leak-check summary, reconciliation result. Refresh every 5–10 s or on demand. **[ENGINEERING RECOMMENDATION]**
- Client reconnection: WS auto-reconnect with exponential backoff; on reconnect the client re-requests a full snapshot (`GET /api/status`, `GET /api/risk/state`, `GET /api/state/{panel}`) before resuming the stream, so no event is lost silently. **[ENGINEERING RECOMMENDATION]**

### 8.3 Resilience to child-process crash

- If a child (training or live) exits unexpectedly, the supervisor: marks the job `failed` with exit code, keeps the log tail visible, raises a non-blocking banner, and — for training only — offers **Resume** if a checkpoint exists (`--resume` + checkpoint path, with the dimension pre-check). **[ENGINEERING RECOMMENDATION]**
- The backend never auto-restarts the **live** loop; restart requires explicit user action (safety first). **[ENGINEERING RECOMMENDATION — HARD RULE]**

### 8.4 Latency budgets

| Feed | Budget (p95) |
|---|---|
| Loss-curve tick → rendered | < 1 s |
| Decision feed item → rendered | < 250 ms |
| Log tail line → rendered | < 500 ms |
| REST snapshot | < 200 ms |
| Control action acknowledged | < 1 s |

**[ENGINEERING RECOMMENDATION]** These budgets are achievable only with WS push + canvas rendering; the Streamlit fallback meets only the REST snapshot and control-action budgets.

### 8.5 Windows process management

- Children are spawned with `subprocess.CREATE_NEW_PROCESS_GROUP` so the whole tree can be terminated by group. **[ENGINEERING RECOMMENDATION]**
- Clean stop: sentinel file (`.STOP` / `.PAUSE`) + graceful `terminate()`; hard kill: `taskkill /PID <pid> /T /F`. **[ENGINEERING RECOMMENDATION]**
- **Sentinel kill-switch pattern (already in the live loop — FACT):** `live/live_trade_mt5.py` checks the `KILL_SWITCH` file every loop iteration and halts with `logger.critical("KILL_SWITCH present — halting")`. The dashboard arms/disarms this file and always displays its state. **[FACT + ENGINEERING RECOMMENDATION]**

### 8.6 HONEST-STATE RULE (non-negotiable)

> **The dashboard never fabricates metrics.**
> - `evaluation.json` with `passed: false` renders **"NOT PROMOTABLE"** — never a green promotable badge. Every inspected artifact currently has `passed: false`. **[FACT]**
> - Missing `artifacts/models/production/manifest.json` renders **"none trained"** for the production slot — never an empty-but-active look. **[FACT]**
> - A killed/crashed job shows partial progress with a `failed`/`killed` chip — no implied completion.
> - Every number shown on screen must be traceable to a file path or metrics-dict key listed in Section 7.4; any panel that cannot resolve its source shows `—` (not a fabricated 0).

---

## 9. Risks & mitigations

| Risk | Impact | Mitigation (dashboard design) |
|---|---|---|
| Subprocess orphans after dashboard crash/close | Zombie training/live processes keep trading or consuming CPU | Process-group spawn (`CREATE_NEW_PROCESS_GROUP`); on startup the supervisor scans for orphan PIDs from its registry file and offers to terminate them; kill-switch file always honored by the live loop **[ENGINEERING RECOMMENDATION + FACT]** |
| Heavy DOM churn from per-step metrics | UI jank, high RAM in browser | Canvas/WebGL charts with downsampling (max ~2k points/series, aggregate beyond); batch WS frames (e.g. 10 steps/frame); virtualized log view (render last N lines only) **[ENGINEERING RECOMMENDATION]** |
| Timezone confusion (bars are UTC) | Misread signal times, wrong daily-loss boundaries | Display **all** timestamps in UTC (ISO-8601 with `Z`), honor `BROKER_UTC_OFFSET_HOURS` only inside live-account rendering; `mock_state.json` `last_bar_time` is `+00:00` **[FACT]** |
| Venv path resolution failure | Spawn fails with confusing error | Preflight check #6 verifies `.venv\Scripts\python.exe`; supervisor logs the resolved interpreter path at startup **[FACT: venv verified present]** |
| Contract drift (contract vs model mismatch) | Silent dimension mismatch, NaN signals | Manifest hash check + `obs_dim == window*n_features+5` validation on load and resume; UI shows CONTRACT HASH MISMATCH banner **[FACT `core/model_artifacts.py`]** |
| Data prep leakage (validation bars in fit) | Optimistic eval, false promotion | `train/data.py` fits on train window only; leak-check summary shown after every prep job **[FACT]** |
| Live trade while no promoted model | Unauthorized risk | `REQUIRE_PROMOTED_MODEL` badge (currently BLOCKED) + `check_live_gates` enforced at spawn **[FACT]** |

---

## 10. Roadmap mapping to the FINAL study experiments (E1–E10)

Each experiment from `research/FINAL_AUTONOMOUS_TRADING_AI_STUDY.md` (Proposed Experiments section) maps to dashboard features that **enable** or **verify** it:

| Exp | Experiment (FINAL study) | Dashboard features that enable/verify it |
|---|---|---|
| E1 | Cross-asset data panel (DXY/10Y/VIX, shift(1)) | Data-prep form (macro CSV selector + lag display); feature panel showing macro columns; correlation-guard chip in risk panel fires on real data |
| E2 | Learned representation vs feature pipeline | Prep form (contract build); job list with A/B runs; loss curves + eval timeline compare rollout loss and OOS P&L per variant |
| E3 | Walk-forward vs CPCV evaluation methodology | Eval timeline renders both protocols' promotion chips side-by-side; no training UI change |
| E4 | World-model value (imagination vs replay) | Job list with env-steps counter; loss curves (world_model_loss, entropy); checkpoint timeline shows which policy type reached parity |
| E5 | Reward ablation | Hyperparameter panel exposes reward-type/composite-weight flags per run; job comparison table maps reward variant → evaluation.json metrics |
| E6 | Offline pretraining (CQL/DT) | Prep form + base-manifest selector for staged runs; job list shows phase markers (pretrain→finetune); sample-efficiency counter (env steps to parity) on eval timeline |
| E7 | Regime-conditioned world model | Regime-label strip (3.5) colors per-regime OOS metrics; meta-train preset exposes regime-conditioning flags |
| E8 | LLM research-desk ablation | Economic calendar feed (4.7) + advisory-feature toggle in prep form; feature-attribution panel (brain panel) shows whether the policy uses event flags |
| E9 | Shadow self-improvement loop | Champion/challenger comparison on eval timeline; auto-rollback events feed; job list shows offline challenger training; risk panel shows non-degradation counters |
| E10 | Live-safe promotion trial | TRADING_MODE badge (demo→live), kill-switch arm state, reconciliation status, account panel vs shadow expectation; zero-safety-violation log |

---

## 11. Verifiable acceptance criteria

1. **File & structure:** `frontend_dashboard_design.md` exists at repo root, 600–900 lines, with all section headers: Overview, Tech-Stack, Training, Trading, Control, Preparation (AI), API, NFR, Risks, Roadmap, Acceptance. **[THIS DOCUMENT]**
2. **Tech-stack:** comparison table with ≥ 6 rows (FastAPI+WS+SPA vs Streamlit), decision rationale references the FACT that no backend exists today.
3. **Training view:** all 11 sub-panels listed, each with a concrete data source; the 7 `train_step` metric keys (`world_model_loss`, `recon_loss`, `reward_loss`, `kl_loss`, `value_loss`, `policy_loss`, `entropy`) appear verbatim.
4. **Trading view:** every panel has a concrete data source; the 13 risk reason strings appear verbatim (APPROVED, CIRCUIT_BREAKER, HALTED, MAX_DRAWDOWN, POSITION_TOO_LARGE, TOO_MANY_LOSSES, HIGH_VOLATILITY, CORRELATION_GUARD, EVENT_RISK, MAX_TRADES, COOLDOWN, SPREAD_TOO_WIDE, MARKET_CLOSED).
5. **Control:** HARD SAFETY INVARIANT present as a bolded rule (frontend never calls broker/order functions); resume dimension check (`obs_dim == window*n_features+5`); honest no-promoted-model statement.
6. **Preparation:** all 8 presets named exactly (dreamer, transformer, ppo, ensemble, god-mode, ultimate-150, adversarial, meta-train); all train-script names appear (train_dreamer.py, train_transformer.py, train_ppo.py, train_ensemble.py, train_dreamer_mcts.py, train_adversarial.py, train_meta.py); preflight checklist ≥ 6 items.
7. **API surface:** ≥ 12 REST endpoints, ≥ 4 WebSocket channels, 4 JSON payload examples (job start, job status, decision feed, risk state), and a state-source mapping table covering every panel from the training and trading sections.
8. **Honesty:** HONEST-STATE RULE present; `passed=false` renders "NOT PROMOTABLE"; missing production manifest renders "none trained"; no fabricated metric path exists in the spec.
9. **Roadmap:** E1–E10 all mapped (10 rows) to dashboard features.
10. **Risks:** ≥ 5 rows in the risks table.
11. **Acceptance:** this list itself has ≥ 8 items, each tied to a section.
12. **Claim markers:** every major claim tagged FACT / RESEARCH FINDING / ENGINEERING RECOMMENDATION / UNPROVEN.
