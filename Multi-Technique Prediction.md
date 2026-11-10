# AGEX AI — Mathematical Market Analysis, Multi-Technique Prediction & MT5 Order Flow Integration

## 1. Main Objective

AGEX AI ka existing trading concept preserve karo:

1. Market ke pehle 4 completed minutes ka data collect karo.
2. In 4 minutes ke candles, tick activity, order flow, momentum, volatility aur market structure ka mathematical analysis karo.
3. In tamam techniques ke results ko combine karke aglay minute ki direction estimate karo.
4. 5th minute ke start par signal execute karne ki koshish karo, lekin actual executable Bid/Ask price aur execution delay ko consider karo.
5. Agar signals weak ya conflicting hon to NO TRADE do.

**Important:** System ka objective guaranteed ya perfect prediction nahi, balkeh statistically validated aur calibrated prediction banana hai. Kisi bhi technique ko guaranteed BUY/SELL signal na samjho. Har prediction ke sath confidence, expected movement, uncertainty aur signal reasons bhi provide karo.

System ko modular architecture mein banao, taake har technique independently calculate, test, enable/disable aur improve ki ja sake.

---

# 2. DATA COLLECTION LAYER

MT5 se following data collect karne ka system banao:

### A. Candle Data

Har completed 1-minute candle ke liye:

- Open (O)
- High (H)
- Low (L)
- Close (C)
- Tick Volume
- Real Volume, agar broker provide karta ho
- Candle opening timestamp
- Candle closing timestamp
- Symbol aur timeframe

MT5 functions:
- CopyRates()
- MqlRates

### B. Tick Data

Har tick ya quote update ke liye:

- Timestamp in milliseconds
- Bid price
- Ask price
- Last price, agar available ho
- Tick volume
- Real volume, agar available ho
- Tick flags

MT5 functions:
- CopyTicks()
- CopyTicksRange()
- MqlTick
- OnTick()

Tick data ko timestamp ke mutabiq order mein rakho. Duplicate ticks, missing data aur invalid prices ko handle karo.

### C. Order Book Data — OFI ke liye bohat zaroori

MT5 se broker ka available Market Depth / Depth of Market data hasil karo.

Required fields:

**Har order-book update par:**

- Timestamp in milliseconds
- Bid price at each available level
- Bid volume at each available level
- Ask price at each available level
- Ask volume at each available level
- Level number
- Book update sequence, agar available ho
- Symbol

MT5 functions aur structures:

- MarketBookAdd(symbol)
- MarketBookGet(symbol, book_array)
- MarketBookRelease(symbol)
- OnBookEvent(symbol)
- MqlBookInfo
- ENUM_BOOK_TYPE

`MqlBookInfo` mein available `type`, `price`, `volume` aur `volume_real` fields read karo.

**Important MT5 implementation requirements:**

1. EA ya suitable MT5 component se MarketBookAdd() call karke symbol ki Market Depth subscription start karo.
2. OnBookEvent() par MarketBookGet() call karke current order-book snapshot read karo.
3. Har snapshot ko timestamp ke sath store karo.
4. Best Bid aur Best Ask ke sath multiple available depth levels bhi store karo.
5. Book updates ko time sequence mein process karo.
6. MarketBookRelease() se subscription properly release karo.
7. Check karo ke broker aur symbol actual DOM data provide karte hain ya nahi.
8. Agar DOM unavailable ho, OFI ko zero ya fabricated value mat do. OFI status `UNAVAILABLE` rakho aur us feature ko prediction score se exclude karo.
9. Agar broker sirf limited depth provide karta hai, available levels ki tadaad record karo.
10. MT5 broker ka Forex DOM fragmented ya broker-specific ho sakta hai. Isay global Forex order book ya complete worldwide liquidity mat samajhna.

**OFI ke liye sirf candle volume ya tick volume kafi nahi hai.** Proper order-book updates aur bid/ask depth quantities required hain.

---

# 3. TECHNIQUE 1 — SIGNED TICK PRESSURE

### Definition

Yeh technique measure karti hai ke market ke quote updates mein upward movement zyada ho rahi hai ya downward movement.

Forex MT5 mein yeh quote-direction pressure hai, actual executed buyer/seller volume nahi.

### Formula

Pehle Mid Price:

`Mid_i = (Bid_i + Ask_i) / 2`

Har tick ka direction:

`d_i = sign(Mid_i - Mid_(i-1))`

Jahan:

- Upward movement = +1
- Downward movement = -1
- Unchanged = 0

Signed Tick Pressure:

`STP = Sum(d_i) / N`

Yahan N total valid tick observations hain.

### Interpretation

- STP positive: upward quote pressure
- STP negative: downward quote pressure
- STP near zero: mixed ya weak pressure

Unchanged ticks ko zero count karo. Saath upward, downward aur unchanged tick counts bhi separately save karo.

### Kahan use hoga?

Pehle 4 minutes mein buyer/seller quote pressure measure karne ke liye. Isay candle direction aur OFI ke sath compare karo.

---

# 4. TECHNIQUE 2 — CANDLE BODY STRENGTH

### Definition

Yeh calculate karti hai ke candle ki total high-low range mein actual candle body kitni dominant hai.

### Formula

`Range = High - Low`

`BodyStrength = abs(Close - Open) / Range`

Signed version:

`SignedBody = (Close - Open) / Range`

### Interpretation

- SignedBody near +1: strong bullish body
- SignedBody near -1: strong bearish body
- SignedBody near 0: small body, indecision ya wick-dominated candle

Range zero ho to division perform na karo. Feature ko invalid mark karo.

### Kahan use hoga?

Har 1-minute candle ki strength calculate karo. Pehle 4 minutes ki body strength aur direction ko aggregate karo.

---

# 5. TECHNIQUE 3 — CLOSE LOCATION VALUE (CLV)

### Definition

CLV batata hai ke candle apni total range ke andar kahan close hui.

### Formula

`CLV = (2 * Close - High - Low) / (High - Low)`

### Interpretation

- +1: candle high par close hui
- -1: candle low par close hui
- 0: candle range ke midpoint par close hui

### Kahan use hoga?

Agar bullish candle repeatedly apne high ke qareeb close ho rahi hai to upward closing pressure ka evidence milta hai.

Agar bearish candles repeatedly low ke qareeb close hon to downward closing pressure ka evidence milta hai.

Isay SignedBody aur Tick Pressure ke sath combine karo.

---

# 6. TECHNIQUE 4 — MOMENTUM CONSISTENCY

### Definition

Yeh measure karti hai ke recent candles ek hi direction mein consistently move kar rahi hain ya nahi.

### Formula

Har candle ka log return:

`r_j = ln(Close_j / Close_(j-1))`

Directional consistency:

`MC = Sum(sign(r_j)) / K`

Yahan K recent completed returns ki tadaad hai. Current 4-minute strategy mein 4 returns use karo, aur required previous close bhi collect karo.

### Interpretation

- +1: tamam returns positive
- -1: tamam returns negative
- 0: mixed direction

### Kahan use hoga?

Impulse ki consistency measure karne ke liye. Agar price sirf ek candle mein move hui aur baqi candles opposite ya flat hain, to momentum consistency weak hogi.

---

# 7. TECHNIQUE 5 — ORDER FLOW IMBALANCE (OFI)

### Definition

OFI order book ke bid aur ask prices aur quantities mein hone wali changes ko measure karta hai.

Yeh sirf current bid volume minus ask volume nahi hai. Is mein order-book updates ke sath liquidity add hone, remove hone aur best prices change hone ka effect bhi include hota hai.

Reference research:

Cont, Kukanov & Stoikov (2014), The Price Impact of Order Book Events:
https://arxiv.org/abs/1011.6402

### Required data

Har consecutive order-book snapshot mein:

- Previous best bid price
- Current best bid price
- Previous best bid quantity
- Current best bid quantity
- Previous best ask price
- Current best ask price
- Previous best ask quantity
- Current best ask quantity
- Timestamp

### Best-level OFI formula

Har order-book event n ke liye:

`BidContribution =`

`I(BidPrice_n >= BidPrice_(n-1)) * BidQty_n`

`- I(BidPrice_n <= BidPrice_(n-1)) * BidQty_(n-1)`

`AskContribution =`

`I(AskPrice_n <= AskPrice_(n-1)) * AskQty_n`

`- I(AskPrice_n >= AskPrice_(n-1)) * AskQty_(n-1)`

Final event OFI:

`OFI_n = BidContribution - AskContribution`

Yahan I(condition):

- Condition true ho to 1
- Condition false ho to 0

Window OFI:

`OFI_Total = Sum(OFI_n)`

### Normalized OFI

Different symbols aur liquidity conditions ke darmiyan comparison ke liye:

`Normalized_OFI = OFI_Total / (AverageBidDepth + AverageAskDepth)`

Denominator zero ya unavailable ho to feature invalid rakho.

### Depth Imbalance

Current order book ki bid aur ask liquidity ka difference:

`DepthImbalance = (BidDepth - AskDepth) / (BidDepth + AskDepth)`

BidDepth aur AskDepth ko selected available levels ke quantities ka sum bana sakte ho.

### Multi-Level OFI

Best level ke ilawa multiple available levels ka analysis bhi implement karo.

Har level ka OFI separately calculate karo. Phir level weights use karo:

`MultiLevel_OFI = Sum(LevelWeight_k * OFI_k)`

Weights ko configurable rakho. Unhein backtesting aur validation se tune karo. Arbitrary weights ko proven values na samjho.

### Kahan use hoga?

- Bid-side liquidity pressure
- Ask-side liquidity pressure
- Liquidity add/remove events
- Short-term order-book imbalance
- Potential upward/downward pressure

### Important limitations

- MT5 DOM broker-specific ho sakta hai.
- Forex spot market mein centralized global order book nahi hota.
- Agar broker DOM provide nahi karta to true OFI calculate nahi ho sakta.
- Tick volume ko OFI ka replacement mat banao.
- Book snapshot updates miss ho sakte hain, isliye data quality aur event frequency monitor karo.

OFI ko sirf tab prediction engine mein include karo jab uska data valid aur sufficiently complete ho.

---

# 8. TECHNIQUE 6 — VOLATILITY AND RELATIVE RANGE

### Definition

Yeh technique batati hai ke current market movement normal volatility ke muqablay mein strong hai ya weak.

### Formula 1: Candle Range

`Range_t = High_t - Low_t`

### Formula 2: Average Previous Range

`AverageRange = Average(Range of previous K completed candles)`

Current candle ko uske apne baseline mein include na karo.

### Formula 3: Relative Range

`RelativeRange = CurrentRange / AverageRange`

### Formula 4: ATR

True Range:

`TR = Max(High - Low, abs(High - PreviousClose), abs(Low - PreviousClose))`

ATR ko Wilder smoothing se calculate karo:

`ATR_t = ((ATR_(t-1) * (K-1)) + TR_t) / K`

Initial ATR ko pehle K True Range observations ke average se initialize karo.

### Kahan use hoga?

- Strong impulse identify karne ke liye
- Low-volatility periods mein weak signals filter karne ke liye
- Abnormal volatility detect karne ke liye
- Stop-loss aur expected movement ke context mein

Volatility zyada hona direction ka proof nahi hai. Isay direction features ke sath use karo.

---

# 9. TECHNIQUE 7 — WICK REJECTION / FAILED IMPULSE

### Definition

Yeh technique identify karti hai ke price candle ke high ya low tak ja kar wapas reject hui ya nahi.

### Formulas

`UpperWick = High - Max(Open, Close)`

`LowerWick = Min(Open, Close) - Low`

`UpperRejectionRatio = UpperWick / (High - Low)`

`LowerRejectionRatio = LowerWick / (High - Low)`

### Interpretation

- High UpperRejectionRatio: upper price area se rejection ka evidence
- High LowerRejectionRatio: lower price area se rejection ka evidence

### Kahan use hoga?

- Bullish impulse ke baad upper rejection check karne ke liye
- Bearish impulse ke baad lower rejection check karne ke liye
- Breakout failure aur reversal risk identify karne ke liye

Wick ko automatic reversal signal mat samajhna. Market structure aur next price behavior se confirm karo.

---

# 10. TECHNIQUE 8 — SPREAD AND EXECUTION COST FILTER

### Definition

Signal profitable nazar aa sakta hai, lekin spread, commission aur slippage expected movement ko consume kar sakte hain.

### Formulas

`Spread = Ask - Bid`

`EstimatedTradingCost = Spread + CommissionCost + EstimatedSlippage`

Sab costs ko same price units mein convert karo.

`CostRatio = EstimatedTradingCost / ExpectedMove`

### Kahan use hoga?

Trade execute karne se pehle check karo:

- Spread unusually high to nahi?
- Expected movement transaction cost se kafi bari hai?
- Symbol ki execution conditions acceptable hain?
- Market open/rollover/news ke qareeb liquidity abnormal to nahi?

Agar cost expected movement ke muqablay mein bohat high ho to NO TRADE.

---

# 11. TECHNIQUE 9 — VWAP

### Definition

VWAP volume-weighted average price hai. Yeh batata hai ke available volume ke mutabiq average traded price kahan hai.

### Formula

`VWAP = Sum(Price_i * Volume_i) / Sum(Volume_i)`

Bar-based approximation ke liye:

`TypicalPrice = (High + Low + Close) / 3`

`VWAP = Sum(TypicalPrice_i * Volume_i) / Sum(Volume_i)`

### Interpretation

- Price VWAP se above: average reference price ke muqablay mein higher trading
- Price VWAP se below: lower trading
- VWAP ke around: balance area

### Kahan use hoga?

- Intraday directional context
- Price deviation
- Pullback aur continuation analysis

**Forex caution:** MT5 ka tick volume actual centralized traded volume nahi hota. Agar tick volume use ho to output ko `TickVolumeWeightedPrice` label karo, true exchange-volume VWAP nahi.

---

# 12. TECHNIQUE 10 — VOLUME DELTA / CUMULATIVE VOLUME DELTA

### Definition

Volume Delta aggressive buying aur aggressive selling ke executed volume ka difference measure karta hai.

### Formula

`VolumeDelta = AggressiveBuyVolume - AggressiveSellVolume`

Cumulative Volume Delta:

`CVD_t = CVD_(t-1) + VolumeDelta_t`

### Kahan use hoga?

- Aggressive buying/selling pressure
- Price aur volume delta divergence
- Impulse confirmation

### Data requirement

Actual trade volume aur aggressor-side classification required hai.

Agar MT5 broker Forex spot feed mein aggressor-side trade volume provide nahi karta, to true Volume Delta calculate nahi kar sakte.

Sirf uptick/downtick tick volume ko true Volume Delta label mat karo. Agar proxy implement karo to clearly `EstimatedTickDelta` naam do aur separate feature rakho.

---

# 13. TECHNIQUE 11 — MARKET STRUCTURE

### Definition

Market structure recent swing highs, swing lows aur breakout behavior ko analyze karti hai.

### Swing High

K kisi candle ka High apne left aur right ke selected K neighboring candles ke High se greater ho.

### Swing Low

K kisi candle ka Low apne left aur right ke selected K neighboring candles ke Low se lower ho.

Swing confirm hone ke liye right-side candles ki zaroorat hoti hai. Isliye unconfirmed swing ko future knowledge samajh kar use mat karo.

### Structure conditions

- HH: Higher High
- HL: Higher Low
- LH: Lower High
- LL: Lower Low

### Break of Structure

Bullish BOS:

`Close > PreviousConfirmedSwingHigh`

Bearish BOS:

`Close < PreviousConfirmedSwingLow`

### Kahan use hoga?

- Trend direction
- Breakout confirmation
- Recent support/resistance
- Failed breakout context

BOS aur swing calculations mein sirf us waqt tak available data use karo.

---

# 14. TECHNIQUE 12 — EMA / SMA TREND FILTER

### EMA Formula

`EMA_t = (Price_t * alpha) + (EMA_(t-1) * (1-alpha))`

`alpha = 2 / (K + 1)`

### SMA Formula

`SMA = Sum(Previous K Closing Prices) / K`

### Interpretation

- Price above EMA: bullish trend context
- Price below EMA: bearish trend context
- EMA slope positive: upward trend context
- EMA slope negative: downward trend context

### Kahan use hoga?

Short-term trend context aur 4-minute impulse ko larger trend ke against ya in favor evaluate karne ke liye.

EMA/SMA ko standalone entry trigger mat banao.

---

# 15. TECHNIQUE 13 — RSI

### Definition

RSI recent gains aur losses ka relative strength measure karta hai.

### Formula

`RS = AverageGain / AverageLoss`

`RSI = 100 - (100 / (1 + RS))`

Wilder smoothing use karo.

### Interpretation

- RSI high: recent upward momentum strong ho sakta hai
- RSI low: recent downward momentum strong ho sakta hai
- RSI divergence: price aur momentum ke darmiyan possible disagreement

70 aur 30 ko fixed reversal rules na banao. Strong trends mein RSI extended periods tak extreme reh sakta hai.

### Kahan use hoga?

Momentum context aur possible exhaustion identify karne ke liye.

---

# 16. TECHNIQUE 14 — MACD

### Formula

`MACD = EMA_fast - EMA_slow`

`Signal = EMA(MACD, SignalPeriod)`

`Histogram = MACD - Signal`

### Kahan use hoga?

- Momentum direction
- Momentum acceleration/deceleration
- Signal aur MACD crossover context

Isay short-term signal ke bajaye supporting momentum feature rakho.

---

# 17. TECHNIQUE 15 — ADX

### Definition

ADX trend ki strength estimate karta hai, direction nahi.

Directional movement:

`+DM = CurrentHigh - PreviousHigh`, agar yeh positive aur `CurrentHigh - PreviousHigh > PreviousLow - CurrentLow` ho, warna 0.

`-DM = PreviousLow - CurrentLow`, agar yeh positive aur `PreviousLow - CurrentLow > CurrentHigh - PreviousHigh` ho, warna 0.

`+DI = 100 * Smoothed(+DM) / ATR`

`-DI = 100 * Smoothed(-DM) / ATR`

`DX = 100 * abs(+DI - -DI) / (+DI + -DI)`

`ADX = WilderSmoothed(DX)`

### Kahan use hoga?

- Trend strength
- Range-bound versus trending context
- Momentum signals ki quality evaluate karne ke liye

ADX direction nahi batata. Direction ke liye +DI aur -DI ya doosre directional features use karo.

---

# 18. TECHNIQUE 16 — BOLLINGER BANDS

### Formula

`MiddleBand = SMA(Close, K)`

`UpperBand = MiddleBand + (Multiplier * StandardDeviation)`

`LowerBand = MiddleBand - (Multiplier * StandardDeviation)`

`BandWidth = (UpperBand - LowerBand) / MiddleBand`

### Kahan use hoga?

- Volatility contraction/expansion
- Range context
- Price deviation
- Potential breakout environment

Upper band touch ko automatic SELL aur lower band touch ko automatic BUY mat samajhna.

---

# 19. TECHNIQUE 17 — KALMAN FILTER

### Definition

Kalman Filter noisy price observations se underlying estimated price/trend state ko smooth karne ke liye use hota hai.

Basic state-space model:

`x_t = F * x_(t-1) + w_t`

`z_t = H * x_t + v_t`

Prediction:

`x_pred = F * x_previous`

`P_pred = F * P_previous * F' + Q`

Kalman gain:

`K = P_pred * H' / (H * P_pred * H' + R)`

State update:

`x_updated = x_pred + K * (z - H * x_pred)`

Covariance update:

`P_updated = (I - K * H) * P_pred`

### Kahan use hoga?

- Price noise reduction
- Smoothed trend estimation
- Estimated trend slope
- Raw price aur estimated state ke darmiyan deviation

Q aur R ko configurable aur validation-based rakho.

---

# 20. TECHNIQUE 18 — HIDDEN MARKOV MODEL (HMM)

### Definition

HMM market ke hidden regimes estimate kar sakta hai, jaise:

- Trending
- Range-bound
- High volatility
- Low volatility

Observed features:

`X_t = [Return, Volatility, Range, Momentum, Spread]`

Hidden state:

`S_t ∈ {Regime_1, Regime_2, ..., Regime_K}`

Transition probability:

`P(S_t = j | S_(t-1) = i)`

Observation likelihood:

`P(X_t | S_t = j)`

Viterbi ya forward algorithm se regime probabilities estimate karo.

### Kahan use hoga?

Prediction model ko batane ke liye ke current market kis type ke environment mein hai.

HMM regime ko direction prediction samajh kar use mat karo. Yeh market condition classify karta hai.

---

# 21. TECHNIQUE 19 — SUPPORT / RESISTANCE AND PIVOT LEVELS

### Definition

Recent confirmed swing levels aur previous session ke price levels se important reference zones calculate karo.

Classic Pivot:

`P = (PreviousHigh + PreviousLow + PreviousClose) / 3`

`R1 = 2*P - PreviousLow`

`S1 = 2*P - PreviousHigh`

`R2 = P + (PreviousHigh - PreviousLow)`

`S2 = P - (PreviousHigh - PreviousLow)`

### Kahan use hoga?

- Expected movement ke samne nearby resistance/support
- Breakout distance
- Entry ke baad available price space
- Impulse ke fail hone ka risk

In levels ko exact reversal guarantees na samjho.

---

# 22. TECHNIQUE 20 — FIBONACCI RETRACEMENT

Confirmed swing low L aur swing high H ke darmiyan:

`Level = H - (H-L) * RetracementRatio`

Common ratios:

- 0.236
- 0.382
- 0.500
- 0.618
- 0.786

Downward swing ke liye calculation direction appropriately reverse karo.

### Kahan use hoga?

Pullback depth aur nearby price reference identify karne ke liye.

Fibonacci ko standalone prediction signal na banao. Swing points confirmed aur historically available hone chahiye.

---

# 23. TECHNIQUE 21 — DONCHIAN CHANNEL / BREAKOUT

### Formula

`UpperChannel = HighestHigh(previous K completed candles)`

`LowerChannel = LowestLow(previous K completed candles)`

`MiddleChannel = (UpperChannel + LowerChannel) / 2`

### Kahan use hoga?

- Recent range boundaries
- Breakout detection
- Range expansion
- False breakout analysis

Current candle ko previous channel calculation mein include na karo jab breakout test kar rahe ho.

---

# 24. PREDICTION ENGINE — TAMAM TECHNIQUES KO COMBINE KARNA

Ab ek centralized Prediction Engine banao jo har technique ka result collect kare.

Har feature ko comparable scale par normalize karo.

Directional features ko zaroorat ke mutabiq `[-1, +1]` scale par map karo.

Example:

- Positive value = bullish evidence
- Negative value = bearish evidence
- Zero = neutral ya no directional evidence

Missing data ko zero samajh kar fake neutral evidence create mat karo. Har feature ke sath validity flag rakho.

## A. Feature Groups

Techniques ko in groups mein organize karo:

**Group 1: Order Flow**
- Signed Tick Pressure
- OFI
- Depth Imbalance
- Volume Delta, sirf jab genuine data available ho

**Group 2: Candle Pressure**
- Signed Body
- Body Strength
- CLV
- Wick Rejection

**Group 3: Momentum**
- Momentum Consistency
- RSI
- MACD
- EMA Slope

**Group 4: Volatility**
- ATR
- Relative Range
- Bollinger Band Width

**Group 5: Market Structure**
- HH/HL/LH/LL
- BOS
- Support/Resistance
- Donchian Breakout
- Fibonacci Context

**Group 6: Market Regime**
- HMM Regime Probability
- Kalman Trend Estimate

**Group 7: Execution Quality**
- Spread
- Estimated Slippage
- Cost Ratio
- Data Quality

## B. Group Scores

Har group ka independent score calculate karo:

`OrderFlowScore = WeightedAverage(valid OrderFlow features)`

`CandleScore = WeightedAverage(valid Candle features)`

`MomentumScore = WeightedAverage(valid Momentum features)`

`VolatilityScore = WeightedAverage(valid Volatility features)`

`StructureScore = WeightedAverage(valid Structure features)`

Weights configurable rakho.

Jis feature ka data unavailable ho usay denominator se bhi exclude karo. Missing feature ka weight doosri valid features par proportionally redistribute karo.

Volatility aur execution quality ko direction score mein zabardasti include na karo. Yeh zyada tar trade-quality filters hain.

## C. Final Directional Score

`FinalScore =`

`W1 * OrderFlowScore`

`+ W2 * CandleScore`

`+ W3 * MomentumScore`

`+ W4 * StructureScore`

`+ W5 * RegimeDirectionalScore`

Har W configurable ho aur active group weights ka sum 1 ho.

Weights ko pehle equal-weight baseline se start karo. Phir historical walk-forward validation se tune karo. Arbitrary weights ko optimal ya scientifically proven mat samjho.

## D. Final Prediction

Prediction engine teen possible outcomes de:

- BUY
- SELL
- NO TRADE

Initial thresholds sirf configurable starting parameters hon. Unhein actual out-of-sample data se calibrate karo.

Example logic:

- FinalScore positive aur sufficiently strong ho, supporting groups agree karte hon, aur execution filters pass hon: BUY candidate.
- FinalScore negative aur sufficiently strong ho, supporting groups agree karte hon, aur execution filters pass hon: SELL candidate.
- Score weak ho, features conflict kar rahe hon, data incomplete ho, ya execution conditions poor hon: NO TRADE.

Sirf score ke positive ya negative hone se trade execute mat karo.

## E. Confidence Score

Confidence ko simple FinalScore ka naam badal kar mat do.

Historical data par prediction outcomes ke mutabiq probability calibration implement karo.

Output:

- P(BUY)
- P(SELL)
- P(NO TRADE)

Yeh probabilities calibrated hon aur total 1 hon.

Agar historical calibration data available nahi to inhein probabilities kehne ke bajaye `DirectionalStrengthScore` aur `ConfidenceEstimate` label karo.

---

# 25. EXPECTED MOVE AND TRADE FILTER

Prediction ke sath expected next-minute movement bhi estimate karo.

Starting baseline:

`ExpectedMove = ATR * CalibratedMovementFactor`

MovementFactor ko training/backtesting data se estimate karo. Isay fixed universal constant mat samjho.

BUY ke liye:

`ExpectedProfitDistance = ExpectedMove`

SELL ke liye:

`ExpectedProfitDistance = ExpectedMove`

Execution se pehle:

- Spread
- Commission
- Slippage
- Minimum stop distance
- Symbol contract specifications
- Available margin
- Market execution status

check karo.

Agar expected move estimated trading cost ke muqablay mein kafi nahi, NO TRADE.

Risk management ko prediction score se independent rakho.

---

# 26. AGEX EXISTING 4+1 MINUTE WORKFLOW

Existing strategy ko preserve karo.

### First Minute
- Candle data
- Tick pressure
- Candle strength
- CLV
- Initial volatility

### Second Minute
- Updated tick pressure
- Updated candle features
- Momentum consistency
- Order-book changes and OFI

### Third Minute
- OFI aggregation
- Depth imbalance
- Market structure
- Momentum confirmation
- Spread monitoring

### Fourth Minute
- Final four-minute aggregation
- All feature groups calculate
- Regime and volatility filters
- Final BUY / SELL / NO TRADE decision
- Confidence and expected move calculate

### Fifth Minute
- Signal ke baad pehli available executable Bid/Ask quote par execution logic chalao.
- Agar signal calculate hone ke baad fifth-minute opening price guzar chuki ho, to historical backtest mein us opening price par imaginary fill mat do.
- Actual signal timestamp, quote timestamp, spread aur slippage record karo.

Kisi bhi feature mein fifth-minute ki future information use nahi honi chahiye.

---

# 27. DATA QUALITY AND FAILURE HANDLING

System mein ye checks mandatory hon:

1. Missing ticks detect karo.
2. Duplicate timestamps handle karo.
3. Out-of-order updates correct ya reject karo.
4. Invalid Bid/Ask prices reject karo.
5. Bid greater than Ask jaisi invalid quote conditions detect karo.
6. Zero candle range par division avoid karo.
7. Zero depth par normalized OFI calculate na karo.
8. DOM unavailable ho to OFI disable karo.
9. Tick volume aur real volume ko separate rakho.
10. Broker connection loss par trading disable karo.
11. Market closed ya stale quotes hon to trading disable karo.
12. News ya abnormal spread conditions ke liye configurable safety filter rakho.
13. Har prediction ke sath data-quality status record karo.

---

# 28. BACKTESTING AND VALIDATION

Prediction engine ko live trading se pehle historical data par test karo.

Compare these versions:

- Baseline: existing 4-minute direction method
- Version A: Baseline + Candle features
- Version B: A + Tick Pressure
- Version C: B + OFI, only where valid DOM data exists
- Version D: C + Momentum
- Version E: D + Volatility filters
- Version F: E + Market Structure
- Version G: F + Regime filters
- Version H: Complete system + Execution filters

Har version ko same historical periods aur same realistic cost assumptions par compare karo.

### Required Metrics

- Win Rate
- Average Win
- Average Loss
- Expectancy
- Profit Factor
- Maximum Drawdown
- Net Profit after transaction costs
- Average Slippage
- Number of Trades
- BUY accuracy
- SELL accuracy
- NO TRADE percentage
- Calibration / Brier Score, if probabilistic predictions are available
- Results by trading session
- Results by volatility regime
- Results by symbol

### Validation Rules

- Chronological train/validation/test split use karo.
- Walk-forward testing implement karo.
- Look-ahead bias avoid karo.
- Parameter tuning ke baad untouched out-of-sample test rakho.
- Multiple configurations try karne se backtest overfitting ka risk measure karo.
- Demo/paper trading results record karo before real-money deployment.

Kisi technique ko sirf is liye retain mat karo ke uska standalone win rate high hai. Dekho ke woh complete system mein genuinely additional value add karti hai ya nahi.

---

# 29. REQUIRED SOFTWARE ARCHITECTURE

Project ko independent modules mein implement karo:

1. `MT5DataCollector`
2. `CandleAggregator`
3. `TickProcessor`
4. `MarketDepthCollector`
5. `OFICalculator`
6. `CandleFeatureEngine`
7. `MomentumEngine`
8. `VolatilityEngine`
9. `MarketStructureEngine`
10. `IndicatorEngine`
11. `KalmanFilterEngine`
12. `HMMRegimeEngine`
13. `FeatureNormalizer`
14. `PredictionEngine`
15. `ConfidenceCalibration`
16. `ExecutionCostFilter`
17. `RiskManagement`
18. `TradeExecution`
19. `BacktestingEngine`
20. `LoggingAndMonitoring`

Har module ka unit test banao. Formula calculations ke known test cases verify karo.

---

# 30. FINAL OUTPUT FORMAT

Har 4-minute analysis ke baad Prediction Engine structured output generate kare:

```json
{
  "symbol": "EURUSD",
  "analysis_window": "4 completed minutes",
  "prediction_horizon": "next 1 minute",
  "decision": "BUY | SELL | NO_TRADE",
  "directional_score": 0.0,
  "confidence_estimate": 0.0,
  "probabilities": {
    "buy": null,
    "sell": null,
    "no_trade": null
  },
  "feature_groups": {
    "order_flow": null,
    "candle_pressure": null,
    "momentum": null,
    "volatility": null,
    "market_structure": null,
    "market_regime": null,
    "execution_quality": null
  },
  "ofi_status": "AVAILABLE | UNAVAILABLE | INVALID",
  "data_quality": "GOOD | DEGRADED | INVALID",
  "expected_move": null,
  "current_spread": null,
  "estimated_trading_cost": null,
  "reasons": [],
  "risk_status": "PASS | BLOCKED",
  "signal_timestamp": null
}
```

Invalid ya unavailable features ke liye `null` use karo. JSON output valid hona chahiye.

---

# 31. IMPLEMENTATION ORDER

Sab kuch ek hi dafa blindly implement mat karo.

**Phase 1 — Core Mathematical Engine**
- MT5 candles
- MT5 tick data
- Signed Tick Pressure
- Candle Body Strength
- CLV
- Momentum Consistency
- Volatility
- Spread Filter
- Wick Rejection

**Phase 2 — OFI Integration**
- MT5 MarketBookAdd
- OnBookEvent
- MarketBookGet
- Order-book storage
- Best-level OFI
- Multi-level OFI
- Depth Imbalance
- Data-quality monitoring

**Phase 3 — Additional Analysis**
- VWAP
- RSI
- MACD
- EMA/SMA
- ADX
- Bollinger Bands
- Market Structure
- Support/Resistance
- Donchian Channel

**Phase 4 — Statistical Filters**
- Kalman Filter
- HMM Regime Detection

**Phase 5 — Unified Prediction Engine**
- Feature normalization
- Group scores
- Directional consensus
- NO TRADE logic
- Confidence calibration
- Expected movement estimate

**Phase 6 — Testing**
- Historical backtesting
- Walk-forward validation
- Transaction costs
- Demo trading
- Live monitoring

---

# 32. FINAL INSTRUCTIONS TO THE CODING AI

AGEX AI ka existing source code, architecture, MT5 connection aur 4+1-minute trading workflow pehle inspect karo.

Existing working features ko unnecessarily delete ya rewrite mat karo.

Har technique ka formula independently implement karo. Formula ke inputs, outputs, units, validity conditions aur missing-data behavior document karo.

OFI ke liye actual MT5 Market Depth data collect karne ka complete integration implement karo. Pehle broker se DOM availability verify karo. Agar available nahi, clear diagnostic message show karo aur fake OFI generate na karo.

Tamam valid techniques ko combine karne ke liye centralized prediction engine banao. Lekin conflicting evidence, low confidence, poor execution conditions aur invalid data ki surat mein NO TRADE allow karo.

System ko historical data se test aur calibrate karo. Sirf high backtest accuracy ke basis par profitability claim mat karo.

**Final objective:** AGEX AI ek mathematically structured, multi-layer short-term market analysis system ho jo 4 completed minutes ke evidence se next-minute directional probability estimate kare, execution costs aur risk evaluate kare, aur sirf validated conditions mein trade open kare.