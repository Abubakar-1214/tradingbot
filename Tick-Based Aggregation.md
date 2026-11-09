# AGEX AI — Tick-Based Aggregation Formula Specification

## 1. Mid Price Formula

Formula:

MidPrice(i) = (Bid(i) + Ask(i)) / 2

Variables:
- Bid(i) = Current tick ka bid price
- Ask(i) = Current tick ka ask price

When to use:
Har incoming Forex tick receive hone par.

Where to use:
Tick Data Preprocessing Module.

Purpose:
Bid aur ask ka average nikal kar candle construction ke liye representative price banana.

Important:
Original bid aur ask ko bhi preserve karna hai. Execution calculations mein mid price ko actual executable price nahi samajhna.

---

## 2. Spread Formula

Formula:

Spread(i) = Ask(i) - Bid(i)

When to use:
Har incoming tick par.

Where to use:
Data Preprocessing, Spread Analysis aur Execution Cost Modules.

Purpose:
Market spread measure karna aur trading cost aur liquidity conditions analyze karna.

---

## 3. Tick Count Formula

Formula:

TickCount = Number of accepted ticks in current candle

Mathematically:

T = Σ 1

For i = 1 to K

When to use:
Jab bhi valid, unique tick receive ho.

Where to use:
Tick Aggregation Engine.

Purpose:
Current candle mein kitne ticks collect hue hain, ye track karna.

---

## 4. Candle Completion Formula

Formula:

CandleComplete = (TickCount >= K)

Where:
- TickCount = Current candle ke accepted ticks
- K = Configured ticks per candle

When to use:
Har accepted tick add karne ke baad.

Where to use:
Candle Builder.

Purpose:
Decide karna ke candle complete hui hai ya nahi.

Example:

K = 100

Jab TickCount = 100 ho:
- Current candle finalize karo.
- Candle database mein save karo.
- Next accepted tick se new candle start karo.

---

## 5. Total Completed Candles Formula

Formula:

B = floor(N / K)

Where:
- B = Completed candle count
- N = Total accepted ticks
- K = Ticks per candle

When to use:
Historical data processing aur dataset statistics mein.

Where to use:
Historical Aggregation and Data Validation.

Purpose:
Calculate karna ke available ticks se kitni complete candles ban sakti hain.

Note:
Real-time candle completion ke liye ye formula use nahi karna. Wahan Candle Completion Formula use hoga.

---

## 6. Open Price Formula

Formula:

O = P(1)

Where:
- O = Candle Open
- P(1) = Current candle ke first accepted tick ka mid price

When to use:
New candle ke first tick par.

Where to use:
Candle Builder.

Purpose:
Candle ka starting price determine karna.

---

## 7. High Price Formula

Formula:

H = max(P(1), P(2), ..., P(K))

When to use:
Har new tick receive hone par.

Where to use:
Candle Builder.

Purpose:
Current candle ka highest mid price track karna.

Implementation:

High = max(PreviousHigh, CurrentMidPrice)

---

## 8. Low Price Formula

Formula:

L = min(P(1), P(2), ..., P(K))

When to use:
Har new tick receive hone par.

Where to use:
Candle Builder.

Purpose:
Current candle ka lowest mid price track karna.

Implementation:

Low = min(PreviousLow, CurrentMidPrice)

---

## 9. Close Price Formula

Formula:

C = P(K)

Where:
- C = Candle Close
- P(K) = Final accepted tick ka mid price

When to use:
Sirf jab candle complete ho.

Where to use:
Candle Finalization Module.

Purpose:
Completed candle ka last price determine karna.

---

## 10. Candle Range Formula

Formula:

Range = H - L

When to use:
Candle complete hone ke baad.

Where to use:
Feature Engineering and Volatility Analysis.

Purpose:
Candle ke andar total price movement measure karna.

---

## 11. Candle Body Formula

Formula:

Body = |C - O|

When to use:
Candle complete hone ke baad.

Where to use:
Feature Engineering.

Purpose:
Opening aur closing price ke darmiyan movement ki magnitude measure karna.

---

## 12. Candle Direction Formula

Formula:

Direction =

+1, if C > O

-1, if C < O

0, if C = O

When to use:
Candle complete hone ke baad.

Where to use:
Price Action and ML Feature Engineering.

Purpose:
Bullish, bearish aur neutral candle classify karna.

---

## 13. Upper Wick Formula

Formula:

UpperWick = H - max(O, C)

When to use:
Candle complete hone ke baad.

Where to use:
Candlestick Feature Engineering.

Purpose:
Candle ke upper side par price rejection ya movement measure karna.

---

## 14. Lower Wick Formula

Formula:

LowerWick = min(O, C) - L

When to use:
Candle complete hone ke baad.

Where to use:
Candlestick Feature Engineering.

Purpose:
Candle ke lower side par price rejection ya movement measure karna.

---

## 15. Candle Duration Formula

Formula:

Duration = EndTimestamp - StartTimestamp

When to use:
Candle complete hone par.

Where to use:
Time and Market Activity Analysis.

Purpose:
Measure karna ke K ticks generate hone mein kitna waqt laga.

Unit:
Seconds.

---

## 16. Tick Rate Formula

Formula:

TickRate = TickCount / DurationSeconds

When to use:
Candle complete hone ke baad.

Where to use:
Market Activity Detection and Feature Engineering.

Purpose:
Measure karna ke market mein average kitne price updates per second aa rahe hain.

Important:
If DurationSeconds = 0, division perform na karo. Tick rate ko invalid/null mark karo ya suitable quality flag lagao.

---

## 17. Average Spread Formula

Formula:

AverageSpread = (Σ Spread(i)) / K

Where:
- Spread(i) = Har tick ka spread
- K = Total accepted ticks

When to use:
Candle complete hone par.

Where to use:
Spread Analysis and Execution Cost Features.

Purpose:
Candle ke dauran average bid-ask spread calculate karna.

---

## 18. Maximum Spread Formula

Formula:

MaximumSpread = max(Spread(1), ..., Spread(K))

When to use:
Har incoming tick par update karo.

Where to use:
Spread Monitoring and Risk Analysis.

Purpose:
Candle ke dauran sab se zyada spread identify karna.

---

## 19. Rolling Average Range Formula

Formula:

AverageRange(t) = (1 / M) × Σ Range(t-j)

For j = 0 to M-1

Where:
- M = Number of previous completed candles
- Range = High - Low

When to use:
Har new completed candle ke baad.

Where to use:
Volatility Analysis and ML Feature Engineering.

Purpose:
Recent candles ki average price movement measure karna.

Important:
Prediction at time t ke liye sirf us waqt tak available completed candles use karo.

---

## 20. Rolling Tick Rate Formula

Formula:

AverageTickRate(t) = (1 / M) × Σ TickRate(t-j)

For j = 0 to M-1

When to use:
Har new completed candle ke baad.

Where to use:
Market Activity and Regime Detection.

Purpose:
Recent market activity ki average speed measure karna.

---

## 21. Rolling Volatility Formula

Formula:

Returns(t) = ln(Close(t) / Close(t-1))

RollingVolatility(t) = sqrt(
Σ (Returns(i) - MeanReturns)^2 / (M - 1)
)

Where:
- M = Rolling window size
- Close = Completed candle close
- Returns = Log returns

When to use:
Sufficient completed candles available hone ke baad.

Where to use:
Volatility Analysis, Risk Management and ML Features.

Purpose:
Recent price returns ki variability measure karna.

Note:
Ye rolling standard deviation hai. Agar annualized volatility chahiye to appropriate sampling-frequency scaling separately define karo.

---

## 22. Tick Rate Change Formula

Formula:

TickRateChange(t) =
(TickRate(t) - TickRate(t-1)) / TickRate(t-1)

When to use:
Jab current aur previous candle ka valid tick rate available ho.

Where to use:
Market Acceleration Detection.

Purpose:
Measure karna ke market activity ki speed increase ho rahi hai ya decrease.

If PreviousTickRate = 0:
Division avoid karo aur feature ko invalid mark karo.

---

## 23. Normalized Candle Range Formula

Formula:

NormalizedRange(t) =
Range(t) / AverageRange(t)

When to use:
Jab rolling average range available ho aur zero na ho.

Where to use:
Volatility Regime Detection and ML Feature Engineering.

Purpose:
Determine karna ke current candle normal se kitni bari ya chhoti hai.

Interpretation:
- Value > 1: Current range recent average se zyada.
- Value < 1: Current range recent average se kam.
- Value = 1: Current range recent average ke barabar.

---

# Formula Usage Pipeline

AGEX AI mein formulas ko is order mein implement karo:

STAGE 1 — Incoming Tick Processing
1. Mid Price
2. Spread
3. Tick Validation

STAGE 2 — Tick Aggregation
4. Tick Count
5. Candle Completion
6. Open
7. High
8. Low
9. Close

STAGE 3 — Completed Candle Features
10. Candle Range
11. Candle Body
12. Candle Direction
13. Upper Wick
14. Lower Wick
15. Duration
16. Tick Rate
17. Average Spread
18. Maximum Spread

STAGE 4 — Rolling Features
19. Rolling Average Range
20. Rolling Tick Rate
21. Rolling Volatility
22. Tick Rate Change
23. Normalized Candle Range

STAGE 5 — AI Consumption
Completed candles and their features are passed to:
- Machine Learning models
- Deep Learning models
- Time-Series models
- Market Regime Detection
- Ensemble Intelligence
- Risk Analysis

# Mandatory Engineering Rules

1. Use only valid and accepted ticks.
2. Preserve original bid and ask.
3. Use UTC timestamps.
4. Do not treat broker quote ticks as actual executed trades.
5. Do not modify completed candles.
6. Preserve incomplete candles during restarts.
7. Prevent duplicate tick processing.
8. Handle out-of-order ticks explicitly.
9. Prevent look-ahead leakage in all rolling features.
10. Handle division by zero and missing values safely.
11. Maintain separate aggregation streams for each threshold.
12. Every feature must have a timestamp and data-quality status.
