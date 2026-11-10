# AGEX AI — TECHNIQUE UTILIZATION, RESPONSIBILITY & OUTPUT SPECIFICATION

## 1. MAIN INSTRUCTION

AGEX AI mein implement ki gayi har mathematical aur statistical technique ka ek specific purpose hona chahiye.

Har technique ko sirf calculate karna kaafi nahi. Coding AI ko define karna hai:

1. Technique kis problem ko solve karti hai?
2. Market ka konsa behavior detect karti hai?
3. Isay AGEX ke kis layer mein use karna hai?
4. Iska exact output kya hoga?
5. Output BUY, SELL ya NO TRADE decision mein kis tarah contribute karega?
6. Is technique ki limitations kya hain?
7. Agar data unavailable ho to system kya karega?

Har technique ka output centralized Feature Engine ko provide karo. Prediction Engine in outputs ko combine kare.

Kisi bhi technique ko uske actual purpose se different kaam ke liye use mat karo.

---

# 2. ORDER FLOW ANALYSIS LAYER

Is layer ka primary objective identify karna hai ke available market data mein buying-side ya selling-side pressure kis taraf zyada hai.

### 2.1 Signed Tick Pressure

**Kis kaam ke liye:**

- Upward aur downward quote movements count karna.
- Market mein short-term quote pressure identify karna.
- Buyers aur sellers ke directional pressure ka proxy estimate karna.

**Kahan use karna hai:**

Pehle 4 minutes ke tick data mein continuously calculate karo.

Har minute ka separate result aur complete 4-minute aggregate result banao.

**Output:**

- `tick_pressure_score`: -1 se +1
- `up_tick_count`
- `down_tick_count`
- `unchanged_tick_count`
- `tick_pressure_direction`: BULLISH / BEARISH / NEUTRAL

**Prediction Engine mein role:**

Candle pressure aur momentum ke sath compare hoga. Agar quote pressure aur candle direction agree karte hain to directional evidence stronger ho sakta hai.

Note: Yeh executed buyer/seller volume nahi hai.

### 2.2 Order Flow Imbalance (OFI)

**Kis kaam ke liye:**

- Best Bid aur Best Ask ki liquidity changes measure karna.
- Bid-side aur Ask-side pressure identify karna.
- Order-book additions, removals aur price changes ka impact calculate karna.
- Short-term order-book imbalance measure karna.

**Kahan use karna hai:**

MT5 Market Depth data collector se aane wale snapshots ko OFI Engine mein process karo.

Har order-book event ka OFI calculate karo. Phir 1-minute aur 4-minute aggregation banao.

**Output:**

- `ofi_raw`
- `ofi_normalized`
- `ofi_direction`
- `bid_depth`
- `ask_depth`
- `ofi_event_count`
- `ofi_data_quality`
- `ofi_status`

**Prediction Engine mein role:**

- Positive OFI: bid-side pressure ka evidence.
- Negative OFI: ask-side pressure ka evidence.
- Near-zero OFI: balanced ya weak order-book pressure.

OFI ko directional prediction mein important evidence do, lekin sirf tab jab broker DOM data valid ho.

Agar DOM unavailable ho to OFI score invent mat karo.

### 2.3 Depth Imbalance

**Kis kaam ke liye:**

Available order-book levels mein bid aur ask liquidity ka relative difference measure karna.

**Kahan use karna hai:**

OFI ke sath Order Flow Analysis Layer mein.

**Output:**

- `depth_imbalance_score`: -1 se +1
- `bid_depth_total`
- `ask_depth_total`
- `depth_direction`

**Prediction Engine mein role:**

OFI ko additional liquidity context provide karega. Dono features ko blindly independent evidence samajh kar double-count mat karna.

### 2.4 Volume Delta / CVD

**Kis kaam ke liye:**

Actual aggressive buying aur selling volume ka difference identify karna.

**Kahan use karna hai:**

Sirf tab jab actual executed trade volume aur aggressor-side classification available ho.

**Output:**

- `buy_volume`
- `sell_volume`
- `volume_delta`
- `cumulative_volume_delta`
- `delta_direction`

**Prediction Engine mein role:**

Aggressive buying/selling confirmation.

Agar true volume data available nahi to is feature ko unavailable rakho. Tick volume se fake CVD mat banao.

---

# 3. CANDLE PRESSURE ANALYSIS LAYER

Is layer ka objective completed candles ki internal structure se buying/selling pressure aur rejection identify karna hai.

### 3.1 Candle Body Strength

**Kis kaam ke liye:**

Measure karna ke candle body total price range mein kitni dominant hai.

**Kahan use karna hai:**

Har completed 1-minute candle par.

**Output:**

- `body_strength`: 0 se 1
- `signed_body`: -1 se +1
- `candle_direction`: BULLISH / BEARISH / NEUTRAL

**Prediction Engine mein role:**

Strong bullish ya bearish candle movement identify karne ke liye.

### 3.2 Close Location Value (CLV)

**Kis kaam ke liye:**

Identify karna ke candle high, low ya midpoint ke kitne qareeb close hui.

**Kahan use karna hai:**

Har completed candle aur 4-minute aggregate mein.

**Output:**

- `clv_score`: -1 se +1
- `closing_pressure`: BULLISH / BEARISH / NEUTRAL

**Prediction Engine mein role:**

Candle body ke sath confirmation.

Misal: Bullish body aur high ke qareeb close dono bullish evidence provide karte hain.

### 3.3 Wick Rejection

**Kis kaam ke liye:**

Upper aur lower price levels par rejection identify karna.

**Kahan use karna hai:**

Candle Pressure Layer aur Market Structure Layer dono mein.

**Output:**

- `upper_wick_ratio`
- `lower_wick_ratio`
- `upper_rejection`
- `lower_rejection`
- `rejection_direction`

**Prediction Engine mein role:**

- Upper rejection: bullish impulse ke fail hone ka risk.
- Lower rejection: bearish impulse ke fail hone ka risk.

Yeh reversal ki guarantee nahi.

---

# 4. MOMENTUM ANALYSIS LAYER

Is layer ka objective identify karna hai ke price movement kitni consistent aur powerful hai.

### 4.1 Momentum Consistency

**Kis kaam ke liye:**

Recent completed candles ke returns ki directional consistency measure karna.

**Kahan use karna hai:**

4-minute analysis window mein.

**Output:**

- `momentum_consistency`: -1 se +1
- `positive_return_count`
- `negative_return_count`
- `momentum_direction`

**Prediction Engine mein role:**

Check karega ke movement consistent hai ya sirf ek isolated candle ki wajah se direction strong nazar aa rahi hai.

### 4.2 RSI

**Kis kaam ke liye:**

Recent gains aur losses ki relative strength measure karna.

**Kahan use karna hai:**

Momentum Layer mein supporting feature.

**Output:**

- `rsi_value`: 0 se 100
- `rsi_momentum_state`
- `rsi_divergence_status`

**Prediction Engine mein role:**

Momentum strength aur possible exhaustion identify karna.

RSI 70 hone par automatic SELL aur 30 hone par automatic BUY mat dena.

### 4.3 MACD

**Kis kaam ke liye:**

Short-term momentum aur uski acceleration/deceleration identify karna.

**Kahan use karna hai:**

Momentum Layer mein.

**Output:**

- `macd_line`
- `signal_line`
- `macd_histogram`
- `macd_direction`
- `macd_momentum_change`

**Prediction Engine mein role:**

Price momentum ko supporting confirmation provide karna.

### 4.4 EMA / SMA

**Kis kaam ke liye:**

Short-term aur larger timeframe trend context identify karna.

**Kahan use karna hai:**

Trend Context Layer mein.

**Output:**

- `ema_fast`
- `ema_slow`
- `ema_slope`
- `price_vs_ema`
- `trend_direction`

**Prediction Engine mein role:**

4-minute signal larger trend ke favor mein hai ya against, yeh batana.

### 4.5 ADX

**Kis kaam ke liye:**

Market trend ki strength identify karna.

**Kahan use karna hai:**

Market Regime / Trend Quality Layer mein.

**Output:**

- `adx_value`
- `plus_di`
- `minus_di`
- `trend_strength`
- `trend_direction_context`

**Prediction Engine mein role:**

Trend-based signals ki quality evaluate karna.

ADX direction nahi batata. +DI aur -DI direction context provide karte hain.

---

# 5. VOLATILITY ANALYSIS LAYER

Is layer ka objective determine karna hai ke market mein movement sufficient hai ya nahi, aur current volatility normal hai ya abnormal.

### 5.1 ATR

**Kis kaam ke liye:**

Average true price movement measure karna.

**Kahan use karna hai:**

Volatility Layer aur Risk Management Layer mein.

**Output:**

- `atr_value`
- `atr_normalized`
- `volatility_state`

**Prediction Engine mein role:**

Market movement ki scale samajhna.

**Risk Engine mein role:**

Stop-loss distance aur position sizing ke liye volatility reference provide karna.

### 5.2 Relative Range

**Kis kaam ke liye:**

Current candle ka range recent average range ke comparison mein measure karna.

**Kahan use karna hai:**

Har completed 1-minute candle mein.

**Output:**

- `relative_range`
- `range_expansion`
- `range_contraction`

**Prediction Engine mein role:**

Identify karna ke impulse normal movement se stronger hai ya market compressed hai.

### 5.3 Bollinger Bands

**Kis kaam ke liye:**

Volatility contraction, expansion aur price deviation measure karna.

**Kahan use karna hai:**

Volatility Layer aur Market Context Layer mein.

**Output:**

- `upper_band`
- `middle_band`
- `lower_band`
- `bandwidth`
- `price_band_position`
- `volatility_expansion_state`

**Prediction Engine mein role:**

Market environment aur breakout conditions ka context provide karna.

Band touch ko automatic entry signal mat banana.

---

# 6. MARKET STRUCTURE ANALYSIS LAYER

Is layer ka objective identify karna hai ke price structure bullish hai, bearish hai ya range-bound.

### 6.1 Swing High / Swing Low

**Kis kaam ke liye:**

Important recent turning points identify karna.

**Kahan use karna hai:**

Market Structure Engine mein.

**Output:**

- `last_confirmed_swing_high`
- `last_confirmed_swing_low`
- `swing_high_timestamp`
- `swing_low_timestamp`

**Prediction Engine mein role:**

Nearby structural levels aur breakout conditions determine karna.

### 6.2 HH / HL / LH / LL

**Kis kaam ke liye:**

Trend structure identify karna.

**Output:**

- `higher_high`
- `higher_low`
- `lower_high`
- `lower_low`
- `structure_direction`

**Prediction Engine mein role:**

- HH + HL: bullish structure evidence.
- LH + LL: bearish structure evidence.
- Mixed structure: uncertain/range context.

### 6.3 Break of Structure (BOS)

**Kis kaam ke liye:**

Confirmed swing high ya low ka meaningful breakout detect karna.

**Output:**

- `bos_direction`
- `bos_price`
- `bos_timestamp`
- `bos_strength`

**Prediction Engine mein role:**

Trend continuation ya structural breakout confirmation.

### 6.4 Support / Resistance

**Kis kaam ke liye:**

Nearby price barriers aur reference zones identify karna.

**Output:**

- `nearest_support`
- `nearest_resistance`
- `distance_to_support`
- `distance_to_resistance`

**Prediction Engine mein role:**

Expected movement ke raste mein available price space determine karna.

Agar BUY signal ho lekin resistance bohat qareeb ho to trade ki quality kam ho sakti hai.

### 6.5 Donchian Channel

**Kis kaam ke liye:**

Recent high-low range aur breakout boundaries identify karna.

**Output:**

- `donchian_upper`
- `donchian_lower`
- `donchian_middle`
- `breakout_status`

**Prediction Engine mein role:**

Range breakout aur false breakout context.

### 6.6 Fibonacci

**Kis kaam ke liye:**

Confirmed swing ke against pullback depth aur price reference levels identify karna.

**Output:**

- `fib_236`
- `fib_382`
- `fib_500`
- `fib_618`
- `fib_786`
- `nearest_fib_level`

**Prediction Engine mein role:**

Pullback aur continuation context. Standalone BUY/SELL trigger nahi.

### 6.7 VWAP

**Kis kaam ke liye:**

Available volume ke mutabiq average price reference calculate karna.

**Output:**

- `vwap_value`
- `price_vs_vwap`
- `vwap_deviation`

**Prediction Engine mein role:**

Intraday price context aur possible pullback/continuation reference.

Forex tick-volume-based VWAP ko true exchange-volume VWAP label mat karna.

---

# 7. STATISTICAL ANALYSIS LAYER

Is layer ka objective noisy market data ko process karna aur market regime identify karna hai.

### 7.1 Kalman Filter

**Kis kaam ke liye:**

Noisy price observations se estimated underlying trend aur price state calculate karna.

**Kahan use karna hai:**

Statistical Trend Estimation Layer mein.

**Output:**

- `kalman_estimated_price`
- `kalman_trend_slope`
- `kalman_residual`
- `kalman_uncertainty`

**Prediction Engine mein role:**

Raw price movement ko smoothed trend context provide karna.

### 7.2 Hidden Markov Model (HMM)

**Kis kaam ke liye:**

Market ka hidden regime classify karna.

Possible regimes:

- Trending
- Range-bound
- High volatility
- Low volatility

**Kahan use karna hai:**

Market Regime Detection Layer mein.

**Output:**

- `regime_probabilities`
- `most_likely_regime`
- `regime_confidence`

**Prediction Engine mein role:**

Signal ko current market environment ke mutabiq evaluate karna.

Misal: Agar system ne trend-continuation signal diya lekin HMM range-bound regime identify karta hai, to signal ki quality dobara evaluate karo.

HMM ko BUY/SELL direction predictor mat samajhna.

---

# 8. EXECUTION QUALITY LAYER

Is layer ka objective determine karna hai ke signal ko practically execute karna munasib hai ya nahi.

### 8.1 Spread Filter

**Kis kaam ke liye:**

Current Bid-Ask spread measure karna.

**Output:**

- `current_spread`
- `average_spread`
- `spread_status`

**Role:**

Unusually high spread par trade block karna.

### 8.2 Slippage Estimate

**Kis kaam ke liye:**

Expected execution price aur actual fill price ke darmiyan possible difference estimate karna.

**Output:**

- `estimated_slippage`
- `actual_slippage`
- `slippage_status`

**Role:**

Execution quality aur actual trading costs evaluate karna.

### 8.3 Cost Ratio

**Kis kaam ke liye:**

Estimated transaction cost ko expected price movement ke comparison mein measure karna.

**Output:**

- `estimated_total_cost`
- `expected_move`
- `cost_ratio`
- `execution_quality`

**Role:**

Agar expected move transaction costs ko justify nahi karta to NO TRADE.

---

# 9. FINAL PREDICTION ENGINE — OUTPUT RESPONSIBILITIES

Prediction Engine ko har layer se structured results milenge.

Lekin har group ka role different hoga.

| Layer | Kya identify karegi? | Prediction mein role |
|---|---|---|
| Order Flow | Bid/Ask pressure | Directional evidence |
| Candle Pressure | Candle buying/selling strength | Directional confirmation |
| Momentum | Movement consistency | Momentum confirmation |
| Volatility | Movement ki strength | Trade quality |
| Market Structure | Trend aur breakout | Structural confirmation |
| Statistical | Trend estimate aur regime | Context/filter |
| Execution Quality | Spread aur costs | Trade permission |
| Risk Management | Risk aur exposure | Position control |

## Final Engine ke responsibilities

### Step 1: Directional Evidence

Order Flow, Candle Pressure, Momentum aur Market Structure se bullish/bearish evidence calculate karo.

### Step 2: Market Context

Volatility, EMA, ADX, Kalman aur HMM ke outputs se current market environment evaluate karo.

### Step 3: Conflict Detection

Agar:

- OFI bullish hai lekin candle pressure strongly bearish hai.
- Momentum bullish hai lekin market structure bearish hai.
- Tick pressure aur candle closing pressure opposite hain.

To disagreement record karo.

Conflicting signals ko zabardasti ek direction mein convert mat karo.

### Step 4: Execution Check

Spread, expected move, slippage aur trading costs check karo.

### Step 5: Risk Check

Position size, stop-loss, maximum risk aur open-position limits verify karo.

### Step 6: Final Decision

Sirf teen possible decisions:

- BUY
- SELL
- NO TRADE

### Step 7: Final Output

Har prediction ke sath yeh fields return karo:

- Final decision
- Directional score
- Calibrated BUY probability
- Calibrated SELL probability
- NO TRADE probability
- Order flow score
- Candle pressure score
- Momentum score
- Structure score
- Market regime
- Volatility status
- Execution quality
- Expected movement
- Nearby support/resistance
- Signal disagreement warnings
- Data quality
- Decision explanation

Probabilities sirf historical calibration ke baad provide karo. Us se pehle directional scores ko probabilities label mat karo.

---

# 10. AGEX FOUR-MINUTE ANALYSIS: EXACT RESPONSIBILITY

## Minute 1 — Initial Market Pressure

Calculate:

- Signed Tick Pressure
- Candle Body Strength
- CLV
- ATR / Relative Range
- Initial DOM / OFI

Output:

`Minute1MarketState`

## Minute 2 — Pressure Confirmation

Calculate:

- Updated Tick Pressure
- Updated OFI
- Depth Imbalance
- Candle Pressure
- Momentum Consistency

Output:

`Minute2MarketState`

## Minute 3 — Trend and Structure

Calculate:

- Updated OFI
- Momentum
- EMA Trend
- Market Structure
- Wick Rejection
- Volatility Expansion

Output:

`Minute3MarketState`

## Minute 4 — Final Evaluation

Calculate:

- Final 4-minute aggregate
- Order Flow Group Score
- Candle Group Score
- Momentum Group Score
- Structure Group Score
- Market Regime
- Execution Quality
- Final Directional Score
- Confidence Estimate
- Expected Movement

Output:

`FinalPrediction`

## Minute 5 — Execution

FinalPrediction ke mutabiq:

- BUY candidate ho to BUY execution.
- SELL candidate ho to SELL execution.
- NO TRADE ho to koi position open na karo.

Signal calculate hone ke baad pehli available executable Bid/Ask quote use karo. Stale ya already-passed opening price par artificial execution mat karo.

---

# 11. IMPORTANT: FEATURE DEPENDENCY AND DOUBLE-COUNTING CONTROL

Coding AI, is section ko seriously implement karo.

Kuch techniques ek doosre se related hain.

Misal:

- Signed Body aur CLV candle information se derive hote hain.
- RSI aur MACD momentum indicators hain.
- EMA aur Kalman trend context provide karte hain.
- OFI aur Depth Imbalance related order-book information use karte hain.

In sab ko independent votes samajh kar blindly add mat karo.

Feature correlation analysis implement karo.

Agar highly correlated features ek hi information repeat kar rahe hon to group-level aggregation aur regularization use karo.

Har technique ki incremental predictive value test karo.

---

# 12. REQUIRED MODULE OUTPUT CONTRACT

Har technique ka output same structured format mein hona chahiye:

```json
{
  "technique_name": "SignedTickPressure",
  "layer": "OrderFlow",
  "timestamp": null,
  "value": null,
  "normalized_score": null,
  "direction": "BULLISH",
  "valid": true,
  "data_quality": "GOOD",
  "confidence": null,
  "reason": "Upward quote updates exceeded downward quote updates",
  "limitations": [
    "Not actual executed buyer/seller volume"
  ]
}
```

Har technique ke output mein actual value, normalized score, validity aur explanation honi chahiye.

---

# 13. FINAL DEVELOPMENT RULE

AGEX AI mein koi technique sirf indicator display karne ke liye implement na karo.

Har technique ka defined responsibility, input, output aur downstream consumer hona chahiye.

Architecture modular rakho taake future mein additional techniques ya ML models add kiye ja saken.

System ka final purpose yeh hai:

**Market pressure identify karna → Directional momentum evaluate karna → Market structure confirm karna → Volatility aur regime samajhna → Execution costs check karna → Risk verify karna → BUY, SELL ya NO TRADE ka final decision dena.**

Prediction Engine ko accuracy ke naam par aggressive trade karne ki permission mat do. Data quality, uncertainty aur execution risk ko hamesha priority do.