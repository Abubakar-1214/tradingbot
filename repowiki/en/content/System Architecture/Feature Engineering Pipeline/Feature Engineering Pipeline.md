# Feature Engineering Pipeline

<cite>
**Referenced Files in This Document**
- [ultimate_150_features.py](file://features/ultimate_150_features.py)
- [timeframe_features.py](file://features/timeframe_features.py)
- [cross_timeframe.py](file://features/cross_timeframe.py)
- [macro_features.py](file://features/macro_features.py)
- [calendar_features.py](file://features/calendar_features.py)
- [microstructure_features.py](file://features/microstructure_features.py)
- [load_data.py](file://data/load_data.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
This document describes the modular feature engineering pipeline that transforms raw market data into a comprehensive 150+ dimensional feature matrix for training trading models. The system processes multiple timeframes (M5, M15, H1, H4, D1, W1), computes cross-timeframe correlations, integrates macro indicators (VIX, Oil, Bitcoin, DXY, SPX), incorporates economic calendar events, and adds market microstructure insights. A central orchestrator coordinates all extractors, aligns timestamps, cleans data, and outputs a unified feature set ready for model training.

## Project Structure
The pipeline is organized into specialized modules under features/, each responsible for a distinct aspect of feature generation:
- Timeframe features: 16 standardized features per timeframe
- Cross-timeframe features: relationships across timeframes
- Macro features: global indicators and correlations
- Calendar features: event timing and impact
- Microstructure features: session, time, volume, liquidity signals
- Orchestrator: combines all sources into the final matrix

```mermaid
graph TB
A["Raw OHLC Data<br/>xauusd_* CSV"] --> B["Timeframe Features<br/>16 per TF"]
A --> C["Macro Data<br/>DXY, SPX, US10Y, VIX, Oil, BTC, EUR, Silver/GLD"]
A --> D["Economic Calendar<br/>JSON events"]
A --> E["Microstructure<br/>Session/Time/Volume/Liquidity"]
B --> F["Cross-Timeframe<br/>Trend/Momentum/Vol/SR Confluence"]
B --> G["Orchestrator<br/>Align & Combine"]
C --> G
D --> G
E --> G
G --> H["Final Matrix<br/>150+ features"]
```

**Diagram sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)
- [timeframe_features.py:22-99](file://features/timeframe_features.py#L22-L99)
- [cross_timeframe.py:205-249](file://features/cross_timeframe.py#L205-L249)
- [macro_features.py:26-75](file://features/macro_features.py#L26-L75)
- [calendar_features.py:23-50](file://features/calendar_features.py#L23-L50)
- [microstructure_features.py:170-225](file://features/microstructure_features.py#L170-L225)

**Section sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)
- [timeframe_features.py:236-304](file://features/timeframe_features.py#L236-L304)

## Core Components
- Multi-timeframe engine: Computes 16 features per timeframe (returns, volatility, momentum, moving averages, RSI, MACD, ATR, Bollinger position, volume ratio, distance to high/low). Supports M5, M15, H1, H4, D1, W1 with alignment to a base timeframe.
- Cross-timeframe correlation engine: Derives trend alignment, momentum cascade, volatility regime, and support/resistance confluence across timeframes.
- Macro correlation engine: Loads daily series for DXY, SPX, US10Y, VIX, Oil, Bitcoin, EURUSD, Silver/GLD; computes returns, momentum, and rolling correlations with gold; aligns to intraday via forward-fill.
- Economic calendar integration: Loads JSON events, computes hours/days to next event, event density, high-impact flags, event window detection, and expected volatility multipliers.
- Market microstructure analysis: Session effects (Asian, London, NY, overlap), time-of-day/week/month, volume profile and imbalance, spread proxy and liquidity regime.
- Pipeline orchestrator: Coordinates loading, computation, alignment, cleaning, normalization, target return computation, and concatenation into the final matrix.

**Section sources**
- [timeframe_features.py:22-99](file://features/timeframe_features.py#L22-L99)
- [cross_timeframe.py:21-203](file://features/cross_timeframe.py#L21-L203)
- [macro_features.py:135-360](file://features/macro_features.py#L135-L360)
- [calendar_features.py:111-249](file://features/calendar_features.py#L111-L249)
- [microstructure_features.py:22-167](file://features/microstructure_features.py#L22-L167)
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)

## Architecture Overview
The orchestrator composes five stages:
1. Load and compute timeframe features across M5, M15, H1, H4, D1, W1
2. Compute cross-timeframe features from the aligned timeframe outputs
3. Load macro data and compute macro features aligned to gold timestamps
4. Load economic calendar and compute calendar features on the base index
5. Compute microstructure features from base timeframe OHLCV
Then align all feature sets to the base timeframe index, clean NaNs/infs, convert to float32, compute target returns, and concatenate into the final matrix.

```mermaid
sequenceDiagram
participant U as "User"
participant O as "Orchestrator<br/>make_ultimate_features"
participant T as "Timeframe Features"
participant X as "Cross-Timeframe"
participant M as "Macro Features"
participant C as "Calendar Features"
participant S as "Microstructure"
U->>O : Call make_ultimate_features(base_timeframe)
O->>T : load_and_compute_all_timeframes()
T-->>O : Dict[TF] aligned features
O->>X : compute_all_cross_tf_features(tf_dict)
X-->>O : Cross-TF DataFrame
O->>M : load_macro_data() + compute_macro_features(df_gold, macro_dict)
M-->>O : Macro DataFrame aligned to gold
O->>C : load_economic_calendar() + compute_calendar_features(base_index, calendar)
C-->>O : Calendar DataFrame
O->>S : compute_all_microstructure_features(df_gold)
S-->>O : Microstructure DataFrame
O->>O : Align all to base_index (ffill), fill NaN/inf, float32
O->>O : Compute returns from base close pct_change
O-->>U : (Features, Returns, Timestamps)
```

**Diagram sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)
- [timeframe_features.py:236-304](file://features/timeframe_features.py#L236-L304)
- [cross_timeframe.py:205-249](file://features/cross_timeframe.py#L205-L249)
- [macro_features.py:363-445](file://features/macro_features.py#L363-L445)
- [calendar_features.py:111-249](file://features/calendar_features.py#L111-L249)
- [microstructure_features.py:170-225](file://features/microstructure_features.py#L170-L225)

## Detailed Component Analysis

### Multi-Timeframe Engine
Computes 16 features per timeframe:
- Price action: returns, volatility (rolling std), momentum at 5/10/20 periods
- Trend: fast/slow moving averages, MA difference, trend direction
- Technical: RSI (normalized), MACD histogram normalized by price, ATR as % of price, Bollinger Band position
- Volume and S/R: volume ratio vs 20-period average, distance to recent high/low (50-period)

Supports M5, M15, H1, H4, D1, W1 and aligns all to a base timeframe using forward-fill.

```mermaid
flowchart TD
Start(["Load OHLCV"]) --> PriceAction["Compute returns, vol, momentum"]
PriceAction --> Trend["Compute MA fast/slow, diff, trend"]
Trend --> Tech["Compute RSI, MACD, ATR%, BB position"]
Tech --> VolSR["Compute volume ratio, dist to high/low"]
VolSR --> Align["Align to base timeframe (ffill)"]
Align --> End(["Output 16 features per TF"])
```

**Diagram sources**
- [timeframe_features.py:22-99](file://features/timeframe_features.py#L22-L99)
- [timeframe_features.py:207-233](file://features/timeframe_features.py#L207-L233)

**Section sources**
- [timeframe_features.py:22-99](file://features/timeframe_features.py#L22-L99)
- [timeframe_features.py:236-304](file://features/timeframe_features.py#L236-L304)

### Cross-Timeframe Correlation Engine
Generates 12 advanced features capturing hierarchical market dynamics:
- Trend alignment: overall agreement, strength cascade, divergence
- Momentum cascade: interactions between higher and lower timeframes (e.g., D1×H1, H4×H1, H1×M15)
- Volatility regime: current vs long-term volatility, spikes, compression
- Pattern confluence: support/resistance confluence across TFs, breakout alignment

```mermaid
classDiagram
class CrossTimeframe {
+compute_trend_alignment(tf_dict)
+compute_momentum_cascade(tf_dict)
+compute_volatility_regime(tf_dict)
+compute_pattern_confluence(tf_dict)
+compute_all_cross_tf_features(tf_dict)
}
```

**Diagram sources**
- [cross_timeframe.py:21-203](file://features/cross_timeframe.py#L21-L203)
- [cross_timeframe.py:205-249](file://features/cross_timeframe.py#L205-L249)

**Section sources**
- [cross_timeframe.py:21-203](file://features/cross_timeframe.py#L21-L203)
- [cross_timeframe.py:205-249](file://features/cross_timeframe.py#L205-L249)

### Macro Correlation Engine
Integrates eight macro sources (DXY, SPX, US10Y, VIX, Oil, Bitcoin, EURUSD, Silver/GLD) to produce 24 features:
- For each source: returns, momentum (20-period), rolling correlation with gold (120-day)
- Specialized features: VIX level/change/regime, Gold/Silver ratio, GLD flow proxy
- Alignment: Daily macro series are resampled/aligned to gold’s index via forward-fill

```mermaid
sequenceDiagram
participant M as "Macro Engine"
participant L as "load_macro_data"
participant G as "Gold Prices"
participant R as "Rolling Correlation"
M->>L : Load DXY, SPX, US10Y, VIX, Oil, BTC, EUR, Silver/GLD
L-->>M : Series dict aligned to UTC-naive
M->>G : Resample gold to daily if needed
loop For each macro source
M->>M : Compute returns, momentum
M->>R : Rolling corr(gold_returns, macro_returns, 120)
R-->>M : Correlation series
M->>M : Normalize timezone and align to gold index
end
M-->>M : Concatenate 24 features, ffill to intraday
```

**Diagram sources**
- [macro_features.py:26-75](file://features/macro_features.py#L26-L75)
- [macro_features.py:114-132](file://features/macro_features.py#L114-L132)
- [macro_features.py:135-360](file://features/macro_features.py#L135-L360)
- [macro_features.py:363-445](file://features/macro_features.py#L363-L445)

**Section sources**
- [macro_features.py:26-75](file://features/macro_features.py#L26-L75)
- [macro_features.py:135-360](file://features/macro_features.py#L135-L360)
- [macro_features.py:363-445](file://features/macro_features.py#L363-L445)

### Economic Calendar Integration
Detects upcoming high-impact events and encodes their proximity and type:
- Hours/days to next event, days since last event
- Event density (upcoming events in next 7 days)
- High-impact flag, event window (±2 hours), expected volatility multiplier
- One-hot indicators for NFP and FOMC

```mermaid
flowchart TD
Load["Load JSON events"] --> Next["Find next event after timestamp"]
Next --> Metrics["Compute hours_to_event, days_since_event, event_density"]
Next --> Flags{"High impact?"}
Flags --> |Yes| Window["in_event_window=1, vol_mult=2.0"]
Flags --> |No| Default["vol_mult=1.0"]
Metrics --> Types{"Event type?"}
Types --> |NFP/FOMC| Encode["Set one-hot flags"]
Encode --> Output["Return 8 calendar features"]
Default --> Output
```

**Diagram sources**
- [calendar_features.py:23-50](file://features/calendar_features.py#L23-L50)
- [calendar_features.py:111-249](file://features/calendar_features.py#L111-L249)

**Section sources**
- [calendar_features.py:23-50](file://features/calendar_features.py#L23-L50)
- [calendar_features.py:111-249](file://features/calendar_features.py#L111-L249)

### Market Microstructure Analysis
Captures intraday patterns and order flow proxies:
- Session effects: Asian, London, New York, overlap detection
- Time effects: hour of day, day of week, week of month, month of year
- Volume analysis: volume percentile profile, volume imbalance proxy
- Liquidity: spread proxy (high-low/close), liquidity regime based on rolling spread

```mermaid
classDiagram
class Microstructure {
+compute_session_features(df)
+compute_time_features(df)
+compute_volume_features(df)
+compute_liquidity_features(df)
+compute_all_microstructure_features(df)
}
```

**Diagram sources**
- [microstructure_features.py:22-167](file://features/microstructure_features.py#L22-L167)
- [microstructure_features.py:170-225](file://features/microstructure_features.py#L170-L225)

**Section sources**
- [microstructure_features.py:22-167](file://features/microstructure_features.py#L22-L167)
- [microstructure_features.py:170-225](file://features/microstructure_features.py#L170-L225)

### Pipeline Orchestration
Coordinates all extractors and ensures data alignment:
- Loads timeframe features across M5, M15, H1, H4, D1, W1
- Computes cross-timeframe features from aligned timeframe outputs
- Loads macro data and computes macro features aligned to gold timestamps
- Loads economic calendar and computes calendar features on base index
- Computes microstructure features from base timeframe OHLCV
- Aligns all feature sets to base index using forward-fill
- Cleans NaNs/infs, converts to float32, computes target returns from base close pct_change
- Concatenates into final matrix and returns features, returns, timestamps

```mermaid
sequenceDiagram
participant O as "Orchestrator"
participant T as "Timeframe"
participant X as "Cross-TF"
participant M as "Macro"
participant C as "Calendar"
participant S as "Microstructure"
O->>T : load_and_compute_all_timeframes()
T-->>O : Aligned TF features
O->>X : compute_all_cross_tf_features(tf_dict)
X-->>O : Cross-TF features
O->>M : load_macro_data() + compute_macro_features()
M-->>O : Macro features aligned to gold
O->>C : load_economic_calendar() + compute_calendar_features()
C-->>O : Calendar features
O->>S : compute_all_microstructure_features(df_gold)
S-->>O : Microstructure features
O->>O : Align all to base_index (ffill), fill NaN/inf, float32
O->>O : Compute returns = pct_change(close)
O-->>O : Concatenate → Final matrix (150+)
```

**Diagram sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)

**Section sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)

## Dependency Analysis
- Orchestrator depends on:
  - Timeframe features module for per-TF computations and alignment
  - Cross-timeframe module for inter-TF relationships
  - Macro features module for external indicator integration
  - Calendar features module for event awareness
  - Microstructure module for intraday mechanics
- Data loader provides robust OHLC parsing and validation
- All modules output pandas DataFrames with DatetimeIndex, enabling consistent alignment and concatenation

```mermaid
graph LR
O["Orchestrator"] --> TF["Timeframe Features"]
O --> CT["Cross-Timeframe"]
O --> MF["Macro Features"]
O --> CF["Calendar Features"]
O --> MS["Microstructure"]
TF --> DL["Data Loader"]
MF --> DL
```

**Diagram sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)
- [load_data.py:5-73](file://data/load_data.py#L5-L73)

**Section sources**
- [ultimate_150_features.py:27-226](file://features/ultimate_150_features.py#L27-L226)
- [load_data.py:5-73](file://data/load_data.py#L5-L73)

## Performance Considerations
- Use base timeframe M5 for speed while retaining higher timeframe context via forward-fill alignment
- Rolling windows are sized appropriately per timeframe to balance responsiveness and stability
- Macro features computed on daily series then forward-filled to intraday to reduce computational load
- Data types converted to float32 to minimize memory usage for large datasets
- Forward-fill alignment avoids interpolation artifacts and preserves temporal causality

## Troubleshooting Guide
Common issues and resolutions:
- Missing timeframe files: Required files must exist; optional W1 can be skipped if absent
- Timezone mismatches: Macro series are normalized to UTC-naive indices before alignment
- NaN/Inf values: Pipeline fills NaNs with zeros and replaces infinities with zeros; ensure input data quality
- Column naming: Data loader standardizes MT5 angle-bracket columns; verify presence of required OHLC fields
- Calendar file missing: Calendar features default to neutral values when no events are found

**Section sources**
- [timeframe_features.py:266-283](file://features/timeframe_features.py#L266-L283)
- [macro_features.py:78-111](file://features/macro_features.py#L78-L111)
- [ultimate_150_features.py:156-173](file://features/ultimate_150_features.py#L156-L173)
- [load_data.py:12-73](file://data/load_data.py#L12-L73)
- [calendar_features.py:128-138](file://features/calendar_features.py#L128-L138)

## Conclusion
The pipeline delivers a robust, modular architecture that ingests raw market data and produces a rich 150+ dimensional feature matrix by combining multi-timeframe technicals, cross-timeframe correlations, macro indicators, economic calendar events, and microstructure signals. The orchestrator ensures precise alignment, cleaning, and target computation, enabling reliable training of advanced trading models with comprehensive market intelligence.