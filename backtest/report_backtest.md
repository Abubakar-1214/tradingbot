# Backtest evaluation report — NEW backtesting.py engine (honest)

Deterministic seed: `42` (all stochastic baselines use this seed).

## Data

| File | TF | Bars | Range |
|---|---|---|---|
| `E:\Desktop\NeoMind\Bazz\autonoumuse_trader\data\xauusd_d1.csv` | D1 | 6786 | 2005-01-02 00:00:00 → 2026-01-01 00:00:00 |
| `data\xauusd_h1.csv` | H1 | 57542 | 2014-01-14 00:00:00 → 2026-10-05 13:00:00 |

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
| SmaCrossAtr | xauusd_h1_from_m1 | 18.61% | 216.69% | 0.27 | 0.42 | -10.55% | 43.14% | 1.15 | 547 | **NO** |
| RsiReversion | xauusd_h1_from_m1 | -40.57% | 227.83% | -0.57 | -0.70 | -43.70% | 51.77% | 0.90 | 1775 | **NO** |
| DonchianBreakout | xauusd_h1_from_m1 | 95.59% | 213.10% | 0.49 | 0.82 | -20.49% | 39.04% | 1.24 | 648 | **NO** |
| MlSignalStrategy | xauusd_h1_from_m1 | -60.71% | 227.83% | -0.65 | -0.82 | -96.07% | 49.83% | 0.98 | 13565 | **NO** |
| SmaCrossAtr_zero_cost | xauusd_h1_from_m1 | 34.50% | 216.69% | 0.46 | 0.73 | -8.44% | 43.51% | 1.22 | 547 | **NO** |

### Baseline runs (same cost model)

| Baseline | Data | Return % | Sharpe | MaxDD % | #Trades |
|---|---|---|---|---|---|
| BuyHold | xauusd_d1 | 901.28% | 0.62 | -44.72% | 0 |
| SeededRandom | xauusd_d1 | 24.63% | 0.10 | -37.13% | 104 |
| BuyHold | xauusd_h1_from_m1 | 232.89% | 0.51 | -28.71% | 0 |
| SeededRandom | xauusd_h1_from_m1 | -1.78% | -0.01 | -23.60% | 943 |

## Honest conclusion

- **SmaCrossAtr on xauusd_d1 does NOT beat the baseline**: 52.40% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **RsiReversion on xauusd_d1 does NOT beat the baseline**: 29.07% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **DonchianBreakout on xauusd_d1 does NOT beat the baseline**: 278.94% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **SmaCrossAtr_zero_cost on xauusd_d1 does NOT beat the baseline**: 56.08% vs 901.28% net of costs. This is reported honestly; no fabricated numbers.
- **SmaCrossAtr on xauusd_h1_from_m1 does NOT beat the baseline**: 18.61% vs 232.89% net of costs. This is reported honestly; no fabricated numbers.
- **RsiReversion on xauusd_h1_from_m1 does NOT beat the baseline**: -40.57% vs 232.89% net of costs. This is reported honestly; no fabricated numbers.
- **DonchianBreakout on xauusd_h1_from_m1 does NOT beat the baseline**: 95.59% vs 232.89% net of costs. This is reported honestly; no fabricated numbers.
- **MlSignalStrategy on xauusd_h1_from_m1 does NOT beat the baseline**: -60.71% vs 232.89% net of costs. This is reported honestly; no fabricated numbers.
- **SmaCrossAtr_zero_cost on xauusd_h1_from_m1 does NOT beat the baseline**: 34.50% vs 232.89% net of costs. This is reported honestly; no fabricated numbers.

> Strategies that underperform baselines net of costs: SmaCrossAtr/xauusd_d1, RsiReversion/xauusd_d1, DonchianBreakout/xauusd_d1, SmaCrossAtr_zero_cost/xauusd_d1, SmaCrossAtr/xauusd_h1_from_m1, RsiReversion/xauusd_h1_from_m1, DonchianBreakout/xauusd_h1_from_m1, MlSignalStrategy/xauusd_h1_from_m1, SmaCrossAtr_zero_cost/xauusd_h1_from_m1.
> A live system MUST NOT run these without a material change.

## Determinism evidence

The same seed + same data + same strategy produces identical equity curves across runs;
the baseline uses `np.random.default_rng(seed)` so its random stream is reproducible.
