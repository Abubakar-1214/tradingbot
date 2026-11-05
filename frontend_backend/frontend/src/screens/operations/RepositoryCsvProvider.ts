import type { BarRange, DataProvider, OHLCV, ProviderInfo, SymbolDescriptor, SymbolInfo } from '@luxalgo/vela'
import type { MarketBars, OhlcvBar } from '../../domain/types'
import { apiRequest } from '../../api/client'

const WEEK_MS = 7 * 24 * 60 * 60_000
const DAY_MS = 24 * 60 * 60_000
const MONDAY_EPOCH_ANCHOR = 4 * DAY_MS
const sourceMap: Record<string, { timeframe: string; source: string; aggregateMs?: number; anchorMs?: number }> = {
  '1': { timeframe: 'M1', source: 'data/xauusd_m1.csv' }, '1m': { timeframe: 'M1', source: 'data/xauusd_m1.csv' },
  '5': { timeframe: 'M5', source: 'data/xauusd_m5.csv' }, '5m': { timeframe: 'M5', source: 'data/xauusd_m5.csv' },
  '15': { timeframe: 'M15', source: 'data/xauusd_m15.csv' }, '15m': { timeframe: 'M15', source: 'data/xauusd_m15.csv' },
  '30': { timeframe: 'M15', source: 'data/xauusd_m15.csv · explicitly aggregated to M30', aggregateMs: 30 * 60_000 },
  '60': { timeframe: 'H1', source: 'data/xauusd_h1.csv' }, '1h': { timeframe: 'H1', source: 'data/xauusd_h1.csv' },
  '240': { timeframe: 'H4', source: 'data/xauusd_h4.csv' }, '4h': { timeframe: 'H4', source: 'data/xauusd_h4.csv' },
  D: { timeframe: 'D1', source: 'data/xauusd_d1.csv' }, '1D': { timeframe: 'D1', source: 'data/xauusd_d1.csv' }, '1d': { timeframe: 'D1', source: 'data/xauusd_d1.csv' },
  W: { timeframe: 'D1', source: 'data/xauusd_d1.csv · explicitly aggregated to W1 using UTC Mondays', aggregateMs: WEEK_MS, anchorMs: MONDAY_EPOCH_ANCHOR },
  '1W': { timeframe: 'D1', source: 'data/xauusd_d1.csv · explicitly aggregated to W1 using UTC Mondays', aggregateMs: WEEK_MS, anchorMs: MONDAY_EPOCH_ANCHOR },
}

function toVelaBar(bar: OhlcvBar): OHLCV {
  return { time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close, volume: bar.volume }
}

function aggregateBars(bars: readonly OhlcvBar[], interval: number, anchor = 0): OhlcvBar[] {
  const buckets = new Map<number, OhlcvBar>()
  for (const bar of bars) {
    const bucketTime = Math.floor((bar.time - anchor) / interval) * interval + anchor
    const aggregate = buckets.get(bucketTime)
    if (!aggregate) { buckets.set(bucketTime, { ...bar, time: bucketTime }); continue }
    aggregate.high = Math.max(aggregate.high, bar.high)
    aggregate.low = Math.min(aggregate.low, bar.low)
    aggregate.close = bar.close
    if (bar.volume != null) aggregate.volume = (aggregate.volume ?? 0) + bar.volume
  }
  return [...buckets.values()].sort((a, b) => a.time - b.time)
}

/** Historical CSV adapter; direct intervals use matching files; only M30/W1 are explicitly aggregated. */
export class RepositoryCsvProvider implements DataProvider {
  private readonly initialH1: readonly OhlcvBar[]
  private readonly onHistoryLoaded?: (data: { source: string; lastBarTime?: string; lastClose?: number }) => void
  constructor(initialH1: readonly OhlcvBar[], onHistoryLoaded?: (data: { source: string; lastBarTime?: string; lastClose?: number }) => void) {
    this.initialH1 = initialH1
    this.onHistoryLoaded = onHistoryLoaded
  }

  info(): ProviderInfo {
    return { name: 'repo', displayName: 'Repository CSV', supportedTimeframes: ['1', '5', '15', '30', '60', '240', 'D', 'W'], capabilities: { enumerate: true, stream: false, symbolInfo: false } }
  }

  async listSymbols(): Promise<SymbolDescriptor[]> {
    return [{ ticker: 'XAUUSD', description: 'Gold / US Dollar', type: 'forex' }]
  }

  async getBars(ticker: string, timeframe: string, range: BarRange): Promise<OHLCV[]> {
    if (ticker.toUpperCase() !== 'XAUUSD') throw new Error(`Repository history does not contain ${ticker}.`)
    const config = sourceMap[timeframe] ?? sourceMap[timeframe.toLowerCase()]
    if (!config) throw new Error(`No repository CSV is registered for Vela interval ${timeframe}.`)
    let bars: OhlcvBar[]
    if (config.timeframe === 'H1') {
      bars = [...this.initialH1]
    } else {
      const response = await apiRequest<MarketBars>(`/api/market/bars?symbol=XAUUSD&timeframe=${config.timeframe}&limit=10000`)
      if (response.availability !== 'available' || !response.data?.bars?.length) {
        throw new Error(response.reason ?? `No historical ${config.timeframe} CSV is available.`)
      }
      bars = response.data.bars
    }
    if (config.aggregateMs) bars = aggregateBars(bars, config.aggregateMs, config.anchorMs)
    if (range.from != null) bars = bars.filter((bar) => bar.time >= range.from!)
    if (range.to != null) bars = bars.filter((bar) => bar.time <= range.to!)
    if (range.limit && bars.length > range.limit) bars = bars.slice(-range.limit)
    const last = bars.at(-1)
    this.onHistoryLoaded?.({ source: config.source, lastBarTime: last ? new Date(last.time).toISOString() : undefined, lastClose: last?.close })
    return bars.map(toVelaBar)
  }

  async getSymbolInfo(ticker: string): Promise<SymbolInfo | undefined> {
    return ticker.toUpperCase() === 'XAUUSD' ? { ticker: 'XAUUSD', description: 'Gold / US Dollar', type: 'forex' } : undefined
  }
}
