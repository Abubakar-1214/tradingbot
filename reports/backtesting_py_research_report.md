# Research Report: Backtesting Engine Analysis aur backtesting.py Integration Plan
**Project:** autonoumuse_trader (XAUUSD Deep RL Trading System)
**Date:** September 27, 2026
**Language:** Roman Urdu

---

## 1. Executive Summary (Khulasa)

Aapke trading AI ka backtesting engine **koi external library use nahi karta** — yeh poori tarah **custom/hand-written** Python hai (pandas + numpy par bana hua `backtest/backtest_engine.py` mein). Yeh **buniyadi (basic) level ka hai**, lekin ideas ache hain (conservative costs, walk-forward, crisis validation).

**backtesting.py** (kernc ka repo) ek mature, tez aur well-tested library hai jo aapke engine se kahin zyada advanced hai — leverage, stop-loss/take-profit intrabar fills, limit orders, hedging, optimizer, aur interactive charts sab built-in hain. **Meri recommendation: hybrid approach** — backtesting.py ko core simulation engine banao, aur uske upar apni RL agent ke liye ek adapter layer likho, saath hi apna `RealisticExecutionModel` costs inject karo (commission callback ke zariye).

---

## 2. Aapke Current Backtesting Engine Ki Tafseeli Janch

### 2.1 Kis library par bana hai?
- **Koi bhi backtesting library nahi** (backtesting.py, vectorbt, zipline — kuch nahi)
- Sirf `pandas` + `numpy` se hand-written loop hai
- Files involved:
  - `backtest/backtest_engine.py` → `RigorousBacktester` class (core engine)
  - `env/realistic_execution.py` → `RealisticExecutionModel` (dynamic costs)
  - `env/xauusd_env.py` → Gymnasium env (next-step positions, no look-ahead)
  - `eval/crisis_validation.py` → crisis stress testing
  - `eval/baselines.py` → buy-and-hold, random, MA-crossover benchmarks

### 2.2 Kya features hain (jo ache hain)?
1. **Conservative cost modeling** — spread 3 pips × 1.5 multiplier, slippage 3 pips, commission 0.5 pip. Yani live se zyada pessimistic. ✅
2. **Walk-forward validation** — rolling train/test windows (252-day train, 63-day test default). ✅ (lekin dekho 2.3 ki kami)
3. **Crisis validation** — COVID crash, rate hikes, SVB jaise periods par pass/fail criteria. ✅ Yeh unique feature hai jo backtesting.py mein nahi hai!
4. **Look-ahead prevention** — env next-step par position lagata hai. ✅
5. **Metrics suite** — Sharpe, Sortino, Calmar, MaxDD, win rate, profit factor, avg duration. ✅

### 2.3 Kamiyan (critical weaknesses)
1. **`_get_observation()` placeholder hai!** — line 198–200: `return np.random.randn(100)`. Yani real backtest mein agent ko **random noise features** milte hain, actual market features nahi. Yeh seedha bug hai — real backtest results is se galat aate hain.
2. **Bar-by-bar Python loop** (`data.iterrows()`) — hourly data par 10+ saal ke liye slow hai (har row ka Python overhead).
3. **Intrabar simulation zero hai** — position sirf bar close par change hota hai; stop-loss bar ke andar hit hona simulate nahi hota (gap risk miss ho jata hai).
4. **Position sizing missing** — full position hi lagta hai, size 0–1 fraction nahi.
5. **Entry price bar `close` hota hai** (next bar `open` nahi) — realistic execution se thoda off + slippage ke saath mismatch.
6. **Walk-forward mein agent actually train nahi hota** — `self.agent.train(train_data)` commented placeholder hai.
7. **No dynamic spread in backtester** — `RealisticExecutionModel` backtester mein integrate nahi hua (sirf optional hai), wiki bhi yehi kehta hai.
8. **No visualization** — sirf console logs; equity curve plot, trade markers kuch nahi.
9. **No optimizer** — parameter search khud likhna parega.
10. **No leverage/margin modeling** — XAUUSD (gold) ke liye leverage typical hai (20:1–500:1 brokers).
11. **`initial_capital = 1.0`** normalized equity — acha hai math ke liye but dollar-based trade stats (Kelly, expectancy $) nahi milte.

### 2.4 Verification note
Main ne `pip install backtesting` locally try kiya lekin slow network ki wajah se download cancel ho gaya. Jab network theek ho: `pip install backtesting` (0.6.6, ~190 kB wheel). Code-level runtime verification nahi hui, sab documented sources par based hain.

---

## 3. backtesting.py (kernc/backtesting.py) Ki Complete Research

### 3.1 Basic Info
| Item | Detail |
|---|---|
| Repo | https://github.com/kernc/backtesting.py |
| Version | **0.6.6** (released 22 July 2026 — actively maintained!) |
| Stars | ~9,000 stars, 1.5k forks |
| License | **AGPL-3.0** (⚠️ important — dekho 3.6) |
| Python | >= 3.9 |
| Dependencies | pandas, numpy, **bokeh** (plotting), SAMBO (optimizer) |
| Author | Zach Lûster (kernc) |
| Docs | https://kernc.github.io/backtesting.py/doc/backtesting/ |

### 3.2 Yeh kya hai?
Ek **lightweight, blazing-fast** backtesting framework jo single-asset OHLC(V) strategies backtest karta hai. Focus: position entry/exit signals, indicator-based decisions, interactive visualization. Multi-asset portfolio/arbitrage support **nahi** hai.

### 3.3 Features (aapke engine se comparison)

| Feature | Aapka Engine | backtesting.py 0.6.6 |
|---|---|---|
| Speed | Slow (iterrows loop) | **"Blazing fast"** (optimized NumPy internals) |
| Order types | Sirf instant cross | Market, **limit, stop-limit** orders |
| Stop-loss/Take-profit | Nahin | **Yes — contingent SL/TP bracket orders, intrabar simulated** |
| Position sizing | Full hi | **Fraction (0–1) ya absolute units** |
| Leverage/Margin | Nahin | **Yes — margin=0.02 → 50:1 leverage, margin calls** |
| Hedging | Nahin | **hedging=True option** |
| Spread | Fixed + multiplier | **spread=0.0002 rate** |
| Commission | Fixed | **Constant, (fixed,relative) tuple, ya callable func for custom models** |
| Optimizer | Nahin | **Grid search + SAMBO model-based (Bayesian-style) optimization, heatmap support** |
| Visualization | Console logs | **Interactive Bokeh HTML plots** (candles, equity, drawdown, trade markers) |
| Trade records | Simple dict | **Detailed pd.DataFrame trades (size, entry/exit bar, P&L, duration)** |
| Stats | 11 metrics | **~25 metrics** (Sharpe, Sortino, Calmar, Alpha, Beta, SQN, Kelly, Exposure Time, Expectancy, etc.) |
| Fractional units | Nahin | **FractionalBacktest** in lib |
| Walk-forward | Built-in (broken placeholder) | Nahin built-in — khud combine karo |
| Crisis validation | **Yes** ⭐ | Nahin |
| RL agent support | Yes (act interface) | Nahin direct — adapter chahiye |

### 3.4 Kaise kaam karta hai (architecture)

**Data:** pandas DataFrame with `Open, High, Low, Close, (Volume)` + datetime index. Extra columns (jaise aapke 150+ features!) allowed hain aur strategy se accessible.

**Strategy class API:**
```python
from backtesting import Backtest, Strategy

class MyStrategy(Strategy):
    n1 = 10  # class vars = optimizable params

    def init(self):
        # poora series precompute hota hai (vectorized)
        self.sma = self.I(SMA, self.data.Close, self.n1)

    def next(self):
        # har bar call hota hai; [-1] = latest value
        if crossover(self.sma, ...):
            self.buy(size=0.5, sl=price*0.98, tp=price*1.04)
```
- `self.I(fn, ...)` → indicator wrapper (auto-plotted + length-adjusted)
- `self.buy(size, sl, tp, limit)`, `self.sell(...)`, `self.position.close()`
- `self.orders`, `self.trades`, `self.position` objects
- **Order object:** size, limit, stop, sl, tp, tag, is_long, cancel()
- **Trade object:** entry/exit bar & price, size, sl/tp modifiable live, P&L, tag
- SL/TP contingent orders parent trade close par auto-cancel (OCO)

**Execution model (key detail!):**
- Orders **next candle's open** par fill hote hain (ya `trade_on_close=True` par current close par)
- **Intrabar decisions nahi hote** — candle ke andar trade karne ke liye finer (M1/M5) data use karo
- SL/TP **intrabar** check hote hain: High/Low se hit detection
- Margin < 1 = leverage; margin call → liquidation
- `exclusive_orders=True` → ek waqt mein sirf ek trade
- `hedging=True` → dono directions ek saath
- `finalize_trades=True` → backtest end par open trades close ho kar stats mein count

**Optimizer:**
```python
bt.optimize(n1=range(5,30,5), n2=range(10,70,5),
            maximize='SQN',                    # ya 'Equity Final [$]'
            constraint=lambda p: p.n1 < p.n2,
            method='sambo', max_tries=200,     # grid ya sambo (smart search)
            return_heatmap=True)
```

**Plotting:** Bokeh interactive HTML — candlesticks, equity/return curve, drawdown, volume, trade "tractor beams", indicators, superimpose higher timeframe, resample for 10k+ candles.

### 3.5 Gotchas / Limitations
1. **Single asset, single position** (unless hedging=True)
2. **No intrabar order decisions** — M15 strategy ko M1 data chahiye hoga agar candle-ke-andar exits chahiye
3. **Data cleaning required** — NaNs, missing bars problems karte hain
4. Margin model simple hai (initial/maintenance split nahi)
5. Walk-forward built-in nahi — khud loop likhna parega
6. Aapka long-only env (XAUUSDTradingEnv) ko dual-direction adapter se map karna hoga

### 3.6 ⚠️ License Warning — AGPL-3.0
AGPL **copyleft** license hai: agar backtesting.py ko kisi **public network service** ka hissa banate ho (hosted SaaS), to us service ka source publicly dena **compulsory** ho jata hai.
- **Aapke liye OK agar:** bot sirf private/personal use hai, ya project open-source karna acceptable ho
- **Risk agar:** kabhi bot ko paid/hosted service banana ho bina source share kiye
- Alternative agar AGPL issue ho: **vectorbt** ya **nautilus_trader** (Apache 2.0). Personal research ke liye AGPL koi masla nahi.

---

## 4. Kya Yeh Add Karna Chahiye? (Verdict)

**Haan — core simulation engine ke tor par, apne crisis validation aur RL interface ko rakhte hue.**

Reasons:
1. **Speed:** iterrows loop vs backtesting.py optimized engine — 100x+ tak fark
2. **Realism:** intrabar SL/TP, leverage, margin calls, limit orders — XAUUSD (leveraged, wide-spread asset) ke liye critical
3. **Optimizer:** SAMBO-based smart parameter optimization
4. **Visualization:** Interactive HTML reports
5. **Battle-tested:** 9k stars, July 2026 release

**Kya rakhein apna:** CrisisValidator (unique), RealisticExecutionModel (better dynamic costs), walk-forward loop (RL training chahiye), baselines.

---

## 5. Integration Plan — Step by Step

### Phase 1: Setup
1. `pip install backtesting` (0.6.6)
2. `requirements.txt` mein `backtesting>=0.6.6` add karo
3. Verify: `python -c "from backtesting import Backtest; print('ok')"`

### Phase 2: Adapter (new file: `backtest/bt_py_adapter.py`)

**Core idea:** RL agent ko Strategy class ke andar wrap karna. Har `next()` call par agent se action lena aur `buy()/sell()/position.close()` mein translate karna.

```python
# backtest/bt_py_adapter.py
from backtesting import Strategy

class RLAgentStrategy(Strategy):
    entry_threshold = 0.6   # optimizable param
    size_fraction = 0.5
    sl_atr = 2.0
    tp_atr = 3.0

    def init(self):
        # 150+ features DataFrame ke extra columns mein hone chahiye
        self.feature_cols = [c for c in self.data.df.columns
                             if c not in ('Open','High','Low','Close','Volume')]

    def next(self):
        df = self.data.df
        i = len(df) - 1  # current bar (no look-ahead by design!)
        obs = df[self.feature_cols].iloc[i].values.astype('float32')

        action, conf = self.agent.predict(obs)

        price = self.data.Close[-1]
        if action == 1 and conf >= self.entry_threshold and not self.position.is_long:
            self.position.close()
            self.buy(size=self.size_fraction,
                     sl=price*(1 - self.sl_atr*self.atr),
                     tp=price*(1 + self.tp_atr*self.atr))
        elif action == -1 and conf >= self.entry_threshold and not self.position.is_short:
            self.position.close()
            self.sell(size=self.size_fraction, sl=..., tp=...)
        elif conf < self.entry_threshold:
            self.position.close()
```

**Runner wrapper:**
```python
def run_agent_backtest(agent, xauusd_df, initial_cap=10_000, leverage=20):
    bt = Backtest(
        data=xauusd_df,           # OHLC + feature columns!
        strategy=RLAgentStrategy,
        cash=initial_cap,
        margin=1/leverage,         # 20:1 for XAUUSD
        spread=0.0006,             # ~3 pips relative on gold
        commission=(0, 0.0001),    # ya callable
        exclusive_orders=True,
        trade_on_close=True,       # RL decisions bar close par
    )
    return bt.run(), bt
```

### Phase 3: Data Preparation
- `features/make_features.py` ka output + raw OHLC **ek hi DataFrame** mein merge karo
- Datetime index zaroori, gaps clean, NaNs fillna/dropna
- Extra features columns simple names rakho

### Phase 4: RL Agent Interface — SABSE BADI SPEED HACK
Bar-by-bar single PyTorch prediction slow hogi. Iski jagah **precompute predictions vectorized**:
```python
# Poori history ka prediction matrix ONCE nikaalo:
preds = agent.predict_batch(all_features.ravel())  # batched inference
df['signal'] = preds          # ek extra column
# Strategy.next() mein sirf yeh column parho — instant!
```
Inference 1 dafa hota hai, phir backtesting.py ka optimized loop chalta hai (100x speedup). Model ko `torch.no_grad()` + `eval()` mode mein chalao.

### Phase 5: Cost Modeling — RealisticExecutionModel inject
backtesting.py **callable commission** accept karta hai:
```python
from env.realistic_execution import RealisticExecutionModel
rex = RealisticExecutionModel()

def dynamic_commission(order_size, price):
    # ATR/vol precomputed column se current vol uthao (closure mein precomputation)
    cost_rate = rex.estimate_execution_cost(order_size, current_market_state)
    return cost_rate * abs(order_size) * price
```
Yahan dynamic spread/slippage/market impact/adverse selection sab aa jata hai. ⚠️ Pehle simple fixed commission se start karo, phir upgrade karo — closure mein market-state tracking thoda tricky hai.

### Phase 6: Walk-Forward (apna loop + backtesting.py)
```python
def walk_forward_bt(agent, df, train=252, test=63):
    results = []
    for s in range(0, len(df) - train - test, test):
        test_df = df.iloc[s+train : s+train+test]
        stats, _ = run_agent_backtest(agent, test_df)
        results.append(stats)
    return aggregate(results)
```
⚠️ Har window par agent ko train window par actually retrain karo (SB3 `model.learn()`) — aapke current code mein yeh placeholder hai, fix karna zaroori.

### Phase 7: Crisis Validation — as-is rakhna
Har crisis window (COVID, rate hikes, SVB) par `Backtest(crisis_df, RLAgentStrategy)` chalao. Naye intrabar SL/TP se drawdown numbers zyada realistic aane ge.

### Phase 8: Optimization
"Deployment hyperparameters" (entry threshold, SL/TP ATR mults, size fraction) ko `bt.optimize(maximize='Sharpe Ratio', method='sambo', max_tries=200)` se tune karo. Heatmap se overfitting zone dekho.

### Phase 9: Reports
```python
bt.plot(filename='reports/backtest_report.html', open_browser=False, plot_drawdown=True)
stats.to_csv('reports/backtest_stats.csv')
stats['_trades'].to_csv('reports/backtest_trades.csv')
```

### Phase 10: Testing & Validation
1. **Sanity check:** SMA crossover GOOG test data par README jaisa stats output milna chahiye
2. **Cross-engine check:** Same agent, same data — RigorousBacktester vs backtesting.py results compare; cost-adjusted difference explainable hona chahiye
3. **Look-ahead audit:** `signal` column sirf past data se bana ho (shift(1) verify!)
4. Sharpe CI ke liye bootstrap/permutation test khud add karo (library nahi karti)

---

## 6. Faide vs Risks (Final)

**Faide:** 10–100x faster sims; intrabar SL/TP + leverage realism; interactive HTML reports; production-quality tested engine; powerful optimizer.

**Risks & Mitigations:**
1. **AGPL-3.0** — private/personal OK; hosted SaaS banai to open-source karna hoga
2. **Adapter bug risk** — action mapping galat to results galat; adapter ke unit tests likho
3. **Aapke engine ka placeholder bug** (`_get_observation` random noise deta hai) — pehle fix karo warna dono engines galat den ge
4. **Bokeh dependency** — headless environments mein `open_browser=False, filename=...` use karo
5. **Old Network install issue** — pip download slow tha;=`pip install backtesting --no-deps` + bokeh/PyYAML alag se try karo ya `pip -c` timeout badhao

---

## 7. Alternatives (agar AGPL ya design concern ho)
- **vectorbt** — array-based, extremely fast, custom non-copyleft license, multi-asset
- **nautilus_trader** — professional event-driven, Apache 2.0
- **QuantConnect Lean** — C#/Python, cloud option
- **vectorbt ya custom hybrid in ideas include karo agar SaaS deployment ka plan hai**

---

## 8. Source Index
- https://github.com/kernc/backtesting.py — README (features, install, usage)
- https://pypi.org/project/backtesting/ — v0.6.6, AGPL-3.0, dependencies, release history
- https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html — user guide
- https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html — API reference
- Local: `backtest/backtest_engine.py` (full read, lines 1–422)
- Local: `repowiki/en/content/Evaluation and Backtesting/Backtesting Engine.md`