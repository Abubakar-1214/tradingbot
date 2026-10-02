# Trading AI — Models Report (kaise sochte hain, kya karte hain, kaise jude hain)

Repo: Abubakar-1214/tradingbot, branch `devin/1790614662-production-models` (6 commits, ~47 files).

> **Seedhi baat:** Code, wiring, tests aur docs complete hain. Koi model abhi **trained nahi** hai — training aap karenge (`docs/TRAINING_GUIDE.md`). Kisi model ka profit guaranteed nahi hai; purani backtests mein costs ke baad strategies buy-and-hold se behtar nahi thi. Har model ko promotion gate aur demo period pass karna hoga, tab hi real account.

---

## 1. Poora system ek nazar mein

Socho ek trading desk hai:

| Role | Kaun | Kaam |
|---|---|---|
| Aankhein (data) | Feature pipeline | Candles ko numbers (features) mein badalta hai |
| Dimaagh (brain) | PPO / Transformer / Dreamer / MCTS / Ensemble | Faisla: flat, long ya short |
| Manager (filter) | DecisionEngine | Kamzor ya risky signal rokta hai, size kam/zyada karta hai |
| Risk officer | RiskSupervisor | Daily loss, drawdown, streak — limit tootay to trading band |
| Trader ke haath | TradeExecutor + Broker (MT5 / Mock) | Lot size, SL/TP, order bhejna |
| Trade ka nigraan | TradeManager | Khuli trade: breakeven, trailing stop, partial close, time stop |

**Aham usool:** koi bhi neural model seedha broker ko order nahi bhej sakta. Har order DecisionEngine, TradeExecutor aur RiskSupervisor se guzarta hai.

---

## 2. Models kya "dekhte" hain (input)

Har model ko **ek hi tarah ka input** milta hai (`core/observation.py`):

**A) Market features — pichli 64 candles (H1)**, har candle ke liye (`core/feature_pipeline.py`):
- `ret` (return), `vol` (24-bar volatility), `mom` (24-bar momentum), `ma_diff` (fast-slow MA ka farq)
- `rsi`, `macd_diff`, `bb_position` (Bollinger band mein jagah), `atr_pct`, `volume_ratio`
- H4 aur D1 ka context: `h4_ret/mom/ma_diff`, `d1_ret/mom/ma_diff` (sirf band ho chuki candles)
- Optional macro (DXY waghera): `macro_*_ret/mom/corr` (ek din shift kiya hua)

Sab features **causal** hain, yani future ka data kabhi use nahi hota. Scaling (z-score) sirf training period par fit hoti hai aur `feature_contract.json` mein save hoti hai. Live mein bhi wahi contract lagta hai; hash match na ho to model load hi nahi hota.

**B) Account state — 5 numbers:**
1. `position`: -1 short, 0 flat, 1 long
2. `trade_pnl`: khuli trade ka return (x100, clip ±10)
3. `bars_in_trade`: trade kitni der se khuli hai (log scale)
4. `drawdown`: peak equity se kitna neeche
5. `equity_ratio`: log(equity / starting equity)

Is tarah model sirf market nahi dekhta, **apni halat** bhi dekhta hai: "main trade mein hoon? nuqsaan mein hoon? kitni der se?" Asal trader bhi aise hi sochta hai.

Input size = 64 × features + 5.

## 3. Models kya "bolte" hain (output)

Har model `PolicyOutput` deta hai:
- `action`: 0 = flat (bahar), 1 = long (buy), 2 = short (sell)
- `probs`: har action ki probability, maslan [0.2, 0.7, 0.1]
- `confidence`: chune gaye action ki probability (maslan 0.7)
- `info`: extra, maslan Ensemble ka agreement ya MCTS ke visit counts

## 4. Models kaise seekhte hain — training "duniya"

Sab models ek simulated market mein seekhte hain (`env/dreamer_trading_env.py`):
- Asli costs lagti hain: spread (2.5bp), commission, slippage (aksar khilaf), overnight swap. Volatile bars aur news par costs aur barh jaati hain.
- 1% stop-loss, 30% drawdown par episode khatam.
- **Reward** = equity ka log-return. Yani model ko sirf **costs ke baad asli munafa** ka inaam milta hai. Bar bar trade karega to costs usay khud saza dengi.

---

## 5. Har model detail mein

### 5.1 PPO (MLP) — "tajurbe se seekhne wala trader"
- **Kaise sochta hai:** 64 candles + account state ko ek flat list bana kar ek neural network (MLP) mein daalta hai. Network do cheezein nikalta hai: (a) har action ki probability, (b) "value" — yahan se aage kitna munafa expected hai.
- **Kaise seekhta hai:** Hazaron baar simulated market mein trade karta hai. Jo action umeed se behtar nikla (advantage, GAE se nikala jata hai), uski probability thori barha deta hai; jo bura nikla, use ghata deta hai. "Clip" (0.2) har update chhota rakhta hai taake model achanak pagal na ho jaye.
- **Kamzori:** Candles ka order "mehsoos" nahi karta; sab ko ek list samajhta hai. Sab se simple aur tez hai. **Baseline** hai: baqi models ko isay beat karna chahiye.
- Files: `train/train_ppo.py`, `models/policy.py::PpoPolicy`. Manifest type: `ppo`.

### 5.2 Transformer-PPO — "pattern pehchanne wala trader"
- **Kaise sochta hai:** Har candle ek "token" (lafz) hai. 64 tokens ek jumla hain. Har token mein us candle ke features + account state hote hain. Positional encoding batati hai kaunsi candle pehle thi. **Attention** ke zariye model khud faisla karta hai ke kaunsi pichli candles abhi ke liye zaroori hain (maslan kal ka breakout ya 20 candles pehle ka spike).
- **Kaise seekhta hai:** Wahi PPO tareeqa (clipped objective, GAE, 4 epochs per rollout), bas dimaagh Transformer hai.
- **Extra:** `get_attention_weights()` dikhata hai model ne kin candles par dhyan diya. Yeh debugging ke liye hai, market ka "sabab" nahi.
- Files: `models/transformer_policy.py`, `train/train_transformer.py`. Type: `transformer`.

### 5.3 DreamerV3 — "khwab mein practice karne wala trader"
Sab se advanced model. Do hisse hain:
1. **World model (market ka andaza):**
   - Encoder observation ko compress karta hai.
   - RSSM (recurrent memory `h` + random latent `z`) market ki "halat" yaad rakhta hai.
   - Model predict karta hai: agle step ka latent, observation (reconstruction) aur reward.
   - Loss = reconstruction + reward + KL.
2. **Actor-Critic (khwab mein practice):** World model ke andar 15 steps aage ki **imagined** trajectories banata hai. Actor (faisla karne wala) aur Critic (andaza lagane wala ke halat kitni achhi hai) asli market ke bajaye in khwabon par seekhte hain. Is liye data-efficient hai.
- **Live mein kaise sochta hai:** Har nayi candle par apni memory (h, z) update karta hai, sath mein **jo action asal mein execute hua** (`observe_executed`). Agar model ne long kaha lekin filter ne rok diya, model ko pata hota hai ke wo flat hai.
- **Kamzori:** Agar world model ka andaza ghalat ho to khwab bhi ghalat honge. Training sab se mehngi hai (GPU behtar).
- Files: `models/dreamer_agent.py`, `models/dreamer_components.py`, `train/train_dreamer.py`. Type: `dreamer`.

### 5.4 Dreamer + MCTS — "chess engine ki tarah aage sochne wala"
- **Kaise sochta hai:** Faisla karne se pehle Dreamer ke world model ke andar **search tree** banata hai: "Agar abhi long karun, phir flat... ya short karun, phir..." Har branch par:
  - Actor ki probability **prior** hai (kis raste ko pehle dekhna hai).
  - World model predicted reward deta hai, Critic us halat ki value deta hai.
  - PUCT formula exploration aur exploitation ko balance karta hai (AlphaZero/MuZero jaisa).
  - Default 32 simulations. Jis action ko sab se zyada visits milen, wohi faisla. Visit distribution hi `probs` / confidence ban jati hai.
- **Training:** Alag training nahi hoti. Trained Dreamer checkpoint par manifest banta hai (`train/make_mcts_manifest.py`), phir evaluate hota hai.
- **Kamzori:** Search utna hi achha hai jitna world model. Har candle par zyada CPU lagta hai (H1 par koi masla nahi).
- Files: `models/mcts.py`. Type: `dreamer_mcts`.

### 5.5 Ensemble — "committee / panchayat"
- **Kaise sochta hai:** Kai trained models (maslan PPO + Transformer + Dreamer) se ek hi candle par rai li jati hai.
  - `soft` vote: sab ki probabilities (weights ke saath) average.
  - `hard` vote: har model ka ek vote.
- **Agreement:** kitne (weighted) members ne jeetne wala action chuna. `min_agreement` (default 0.6) se kam ho to **consensus = false**, target flat, aur DecisionEngine usay "NO_CONSENSUS" keh kar hold karta hai (nayi trade nahi).
- **Uncertainty:** normalized entropy (committee kitni confused hai) aur epistemic disagreement (KL: members aapas mein kitna ikhtilaf karte hain).
- **Shart:** sab members ka feature contract hash ek jaisa ho, warna load reject.
- Files: `models/ensemble.py`, `train/train_ensemble.py`. Type: `ensemble`.

### 5.6 MAML (Meta-learning) — "naye market mood ke saath jaldi dhalne wala"
- **Idea:** Market ke mood (regimes) badalte hain. Model ko aisi starting setting sikhao jo har mood mein thore se data se jaldi adjust ho jaye.
- **Regimes (causal, sirf pichla data):** `trend_up`, `trend_down`, `range`, `high_vol`, `low_vol`. Yeh 48-bar aur 120-bar return/volatility se label hote hain. Chhote tukre (<512 bars) skip hote hain.
- **Kaise seekhta hai (first-order MAML):** Har regime ka support/query split banta hai. Dreamer world model ki **copy** support par thori der train hoti hai, query par test hoti hai. Us query loss se asal starting weights update hote hain.
- **Recent adaptation (`train/adapt_recent.py`):** Haal ke data par copy adapt hoti hai aur **naya artifact** banta hai. Live model ke andar khud-ba-khud kuch nahi badalta. Naya artifact bhi evaluate aur promote hona chahiye.
- Kamzori: sirf first-order implemented hai. Chhota recent data overfit kar sakta hai.
- Files: `models/meta_learning.py`, `train/meta_train_dreamer.py`. Type: `dreamer`.

### 5.7 Adversarial fine-tune — "mushkil broker ke khilaf practice"
- **Idea:** Ek "Market Maker" agent trader ko tang karne ki koshish karta hai. Trader ko aisi haalat mein bhi survive karna seekhna hai.
- **Market Maker ke 4 actions:** kuch nahi / spread x3 / slippage x3 / stop-hunt gap (aadhe stop-loss ke barabar adverse jhatka). Har perturbation sirf 1 step ke liye hota hai. Har manipulation ki cost hai, aur max rate cap 20% hai.
- **Seekhna:** Market Maker REINFORCE se seekhta hai (inaam = trader ka nuqsaan − cost). Trader (Dreamer) replay buffer aur `train_step` se sach mein train hota hai.
- **Result:** naya Dreamer artifact, jo **saaf (clean) market** par evaluate hota hai.
- Yeh robustness simulation hai, asal market manipulation ka nizaam nahi.
- Files: `models/adversarial_training.py`, `train/train_adversarial.py`. Type: `dreamer`.

### 5.8 Rule SMA (non-AI fallback)
- Fast SMA(20) slow SMA(50) ko upar cross kare to long, neeche cross kare to short. Cross na ho to **hold** (exit nahi).
- Default `SIGNAL_SOURCE=rule` hai, taake bina trained model ke bhi pura system test ho sake.

---

## 6. Faisle ke baad ki layers (non-neural, deterministic)

### DecisionEngine (`live/decision_engine.py`) — defaults
| Rule | Default | Asar |
|---|---|---|
| Min confidence | 0.55 | Is se kam ho to HOLD (na entry, na exit) |
| Ensemble agreement | 0.6 | Kam ho to NO_CONSENSUS, HOLD |
| Session filter | Juma 20:00 UTC ke baad, Itwar/Peer 01:00 se pehle, 21–22 UTC | Sirf **nayi entry** band; exit hamesha allowed |
| Loss ke baad cooldown | 3 bars | Nayi entry band |
| Confidence scaling | 0.25–1.0 | Kam confidence = chhota lot |
| Kelly (fractional 0.25) | 30 trades ke baad | Edge manfi ho to NO_EDGE (entry nahi); multiplier kabhi 1 se zyada nahi |

### TradeExecutor + ATR sizing
- Risk per trade (default 2%) aur ATR-based stop distance se lot nikalta hai, phir Kelly/confidence multiplier (≤1) lagta hai.
- Lot step par neeche round hota hai, MIN_LOT se kam ho to reject. Aggregate position cap bhi hai.
- Har order par SL/TP, idempotency token, startup reconciliation aur persisted state.

### RiskSupervisor (SQLite par persisted)
- Daily loss 5%, max drawdown 15%, 5 lagataar losses, 20 trades/day, max spread, volatility gate. Koi bhi toota to nayi entries band.
- Kill-switch file hai. Live mode tab tak start nahi hota jab tak saare gates pass na hon.

### TradeManager (`live/trade_manager.py`)
- 1R munafe par SL breakeven (+spread).
- 1.5R par trailing stop (2×ATR); stop sirf tight hota hai, kabhi dheela nahi.
- 1R par 50% partial close, ek dafa (lot limits ke saath).
- 72 bars ke baad agar trade 0.5R se kam par hai to TIME_STOP close.

---

## 7. Kaun kis se juda hai (connections)

```
               TRAINING (aapka kaam)
 MT5 export -> data/xauusd_h1.csv
      |
 train/data.py: time split, scaler SIRF train par fit -> feature_contract.json
      |
      +--> PPO ----------+
      +--> Transformer --+
      +--> Dreamer ------+--> Dreamer+MCTS (usi checkpoint par)
      |        |         +--> MAML meta-train / adapt_recent (naya Dreamer)
      |        |         +--> Adversarial fine-tune (naya Dreamer)
      |        v         |
      |   (sab members) -+--> Ensemble (same contract hash zaroori)
      v
 har artifact: model + feature_contract.json + manifest.json + evaluation.json
      |
 train/evaluate.py -> promotion gate: Sharpe>=0.5, MaxDD<=20%, trades>=20, return>0
      |
 scripts/run_backtest_eval.py (manifest se backtest)
      |
 MODEL_MANIFEST_PATH -> models/registry.py (sahi class load, hash/size check)

               LIVE (har band H1 candle par)
 kill switch -> risk halt -> broker SL/TP reconcile -> TradeManager
   -> feature pipeline (wahi contract) + account state
   -> Model (PolicyOutput)
   -> DecisionEngine (confidence/consensus/session/cooldown/Kelly)
   -> TradeExecutor -> RiskSupervisor -> MT5 (ya MockBroker demo)
   -> model.observe_executed(asal position)  [model ki memory sahi rehti hai]
   -> state save
```

**Kyun aise jode gaye:**
- **Ek hi feature contract:** training, backtest aur live mein bilkul same input milta hai. Sab se aam ghalti (training/live mismatch) yahan hash check se pakri jati hai.
- **Ek hi `TradingPolicy` interface:** koi bhi model (ya ensemble) bina live code badle swap ho sakta hai; sirf manifest path badlo.
- **Dreamer family ek base par:** MCTS, MAML aur Adversarial sab Dreamer ke world model ko reuse karte hain, is liye pehle Dreamer train karna zaroori hai.
- **Ensemble aakhir mein:** alag alag dimaaghon ki ghaltiyan aapas mein cancel hon, aur ikhtilaf ho to trade na ho.
- **Safety layers model se alag:** model kitna bhi ghalat ho, risk limits code mein deterministic hain.

## 8. Training ka tarteeb (kis ko pehle train karna hai)
1. **PPO** — baseline (CPU par chal jata hai).
2. **Transformer** — PPO se behtar hai ya nahi, compare karein.
3. **Dreamer** — sab se mehnga (GPU behtar).
4. **Dreamer+MCTS** — manifest banao aur evaluate karo (training nahi).
5. **Ensemble** — jo models gate pass karein unhe jodo.
6. Optional: **MAML** / **adapt_recent** aur **Adversarial fine-tune**, sab ke naye artifacts.
7. Har artifact: `evaluation.json` pass → backtest → `TRADING_MODE=demo SIGNAL_SOURCE=model` → MT5 **demo account** par kuch hafte → tab live.

Exact commands, flags aur checklist: `docs/TRAINING_GUIDE.md`. Model reference: `docs/MODELS.md`.

## 9. Imandari wali limitations
- Koi trained model shamil nahi. Tests sirf yeh saabit karte hain ke code sahi chalta hai, yeh nahi ke munafa hoga.
- Promotion gate pass karna bhi future profit ki guarantee nahi (overfitting mumkin hai). Walk-forward aur demo period zaroori hain.
- Asli MT5 terminal is Linux machine par test nahi hua (MetaTrader5 package Windows-only). Mock broker aur mocked exporter test hue.
- Simulator costs asal broker se mukhtalif ho sakti hain. `BROKER_UTC_OFFSET_HOURS` sahi set karein.
- Live model khud-ba-khud nahi seekhta. Retraining/adaptation hamesha naya artifact aur naya evaluation hai.

## 10. Testing status
- 139 automated tests pass. verify_config, verify_risk, verify_broker aur verify_risk_integration scripts bhi pass.
- Har trainer, MCTS manifest, ensemble, MAML, adversarial, backtest aur model-based demo run nakli (synthetic) data par end-to-end chalaye gaye.
- `verify_features.py` / `verify_backtest.py` nahi chale kyunki `data/xauusd_h1.csv` aur `data/xauusd_d1.csv` repo mein nahi hain.
