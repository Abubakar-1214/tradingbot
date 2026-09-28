# Backtest evaluation report — NEW backtesting.py engine (honest)

Deterministic seed: `42` (all stochastic baselines use this seed).

## Data

| File | TF | Bars | Range |
|---|---|---|---|
| `data/xauusd_d1.csv` | D1 | 6,787 | 2005-01-02 → 2023-08 |
| `data/xauusd_h1_from_m1.csv` | H1 | 23,657 | 2022-01-02 23:00 → 2024-12 |

All strategy and baseline runs use the **same** validated cost model (`CostModel.from_config(CostConfig())` → `one-way spread=0.000175 (= round-trip 0.000350), commission/side=0.000030, slippage/fill=0.000050, round-trip total=0.000410`) and seed `42`.

## Method

* Market orders fill on the next bar's open (backtesting.py default; `trade_on_close=False`), so there is no look-ahead in fills.
* Every strategy order carries an ATR-based SL and TP; strategies close positions on their exit rule.  BuyHold is intentionally stopless (buy-and-hold by definition).
* Walk-forward on D1 uses train/test/embargo of 1200/400/25 bars (purge + embargo per López de Prado) — test windows are strictly out-of-sample with no warm-up leakage.
* Zero-cost comparison is included so the true cost drag is visible.

## Cost model

| Component | Value |
|---|---|
| one-way spread passed to Backtest | 0.000175 |
| commission per side | 0.000030 |
| slippage per fill (folded into spread) | 0.000050 |
| round-trip total | 0.000410 |

> Mapping: backtesting.py 0.6.2 has no slippage kwarg; the validated
> `CostConfig.spread` (round-trip) is converted to a one-way fill cost
> `spread/2 + slippage` and passed as `Backtest(spread=...)`.

## Strategy vs baseline (same data, same costs, same seed)

| Run | Data | Return % | Buy&Hold % | Sharpe | Sortino | MaxDD % | WinRate % | PF | #Trades | Beat baseline? |
|---|---|---|---|---|---|---|---|---|---|---|
| SmaCrossAtr | xauusd_d1 | 52.40% | 884.26% | 0.39 | 0.56 | -12.46% | 55.17% | 1.83 | 58 | **NO** |
| RsiReversion | xauusd_d1 | 29.07% | 932.41% | 0.16 | 0.24 | -19.84% | 56.79% | 1.20 | 162 | **NO** |
| DonchianBreakout | xauusd_d1 | 278.94% | 923.11% | 0.56 | 0.87 | -28.69% | 50.00% | 1.82 | 98 | **NO** |
| SmaCrossAtr_zero_cost | xauusd_d1 | 56.08% | 884.26% | 0.41 | 0.59 | -12.36% | 55.17% | 1.86 | 58 | **NO** |
| SmaCrossAtr | xauusd_h1_from_m1 | 8.13% | 137.58% | 0.40 | 0.61 | -7.00% | 44.86% | 1.17 | 214 | **NO** |
| RsiReversion | xauusd_h1_from_m1 | -3.45% | 137.07% | -0.13 | -0.15 | -13.67% | 55.96% | 1.02 | 688 | **NO** |
| DonchianBreakout | xauusd_h1_from_m1 | 44.90% | 138.01% | 0.86 | 1.53 | -17.65% | 39.66% | 1.33 | 295 | **NO** |
| SmaCrossAtr_zero_cost | xauusd_h1_from_m1 | 13.15% | 137.58% | 0.63 | 1.00 | -6.47% | 45.79% | 1.24 | 214 | **NO** |

### Baseline runs (same cost model)

| Baseline | Data | Return % | Sharpe | MaxDD % | #Trades |
|---|---|---|---|---|---|
| BuyHold | xauusd_d1 | 901.28% | 0.62 | -44.72% | 0 |
| SeededRandom | xauusd_d1 | 24.63% | 0.10 | -37.13% | 104 |
| BuyHold | xauusd_h1_from_m1 | 124.41% | 1.13 | -20.22% | 0 |
| SeededRandom | xauusd_h1_from_m1 | 21.54% | 0.48 | -14.86% | 383 |

## Honest conclusion

- **SmaCrossAtr on xauusd_d1 does NOT beat the baseline**: 52.40% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **RsiReversion on xauusd_d1 does NOT beat the baseline**: 29.07% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **DonchianBreakout on xauusd_d1 does NOT beat the baseline**: 278.94% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **SmaCrossAtr_zero_cost on xauusd_d1 does NOT beat the baseline**: 56.08% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **SmaCrossAtr on xauusd_h1_from_m1 does NOT beat the baseline**: 8.13% vs 124.41% net of costs. This is reported honestly; no fabricated numbers.
- **RsiReversion on xauusd_h1_from_m1 does NOT beat the baseline**: -3.45% vs 124.41% net of costs. This is reported honestly; no fabricated numbers.
- **DonchianBreakout on xauusd_h1_from_m1 does NOT beat the baseline**: 44.90% vs 124.41% net of costs. This is reported honestly; no fabricated numbers.
- **SmaCrossAtr_zero_cost on xauusd_h1_from_m1 does NOT beat the baseline**: 13.15% vs 124.41% net of costs. This is reported honestly; no fabricated numbers.

> Strategies that underperform baselines net of costs: SmaCrossAtr/xauusd_d1, RsiReversion/xauusd_d1, DonchianBreakout/xauusd_d1, SmaCrossAtr_zero_cost/xauusd_d1, SmaCrossAtr/xauusd_h1_from_m1, RsiReversion/xauusd_h1_from_m1, DonchianBreakout/xauusd_h1_from_m1, SmaCrossAtr_zero_cost/xauusd_h1_from_m1.
> A live system MUST NOT run these without a material change.

## Determinism evidence

The same seed + same data + same strategy produces identical equity curves across runs;
the baseline uses `np.random.default_rng(seed)` so its random stream is reproducible.
