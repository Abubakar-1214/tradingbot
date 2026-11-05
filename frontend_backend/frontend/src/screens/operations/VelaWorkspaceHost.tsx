import { useEffect, useRef, useState } from 'react'
import { VelaWorkspace } from '@luxalgo/vela/workspace'
import type { OhlcvBar } from '../../domain/types'
import { SourceLine, Unavailable } from '../../components/SourceLine'
import { RepositoryCsvProvider } from './RepositoryCsvProvider'
import '../../vela/velaSourceAction.ts'

type HistoryMark = { source: string; timeframe: string; lastBarTime?: string; lastClose?: number }

export function VelaWorkspaceHost({ bars, source, lastBarTime, lastClose, reason }: { bars?: OhlcvBar[]; source?: string; lastBarTime?: string; lastClose?: number; reason?: string }) {
  const mountRef = useRef<HTMLDivElement>(null)
  const [history, setHistory] = useState<HistoryMark>({ source: source ?? 'data/xauusd_h1.csv', timeframe: 'H1', lastBarTime, lastClose })

  useEffect(() => {
    if (!mountRef.current || !bars?.length) return
    const provider = new RepositoryCsvProvider(bars, (next) => setHistory((current) => ({ ...current, ...next })))
    const workspace = new VelaWorkspace(mountRef.current, {
      layout: '1',
      symbol: 'repo:XAUUSD',
      timeframe: '60',
      providers: { repo: () => provider },
      timeframes: ['1', '5', '15', '60', '240', 'D', 'W'],
      upColor: '#DEFF00',
      downColor: '#FF0001',
      theme: {
        background: '#050600',
        textColor: '#DDE4B7',
        gridColor: '#1B2105',
        borderColor: '#343B19',
        upColor: '#DEFF00',
        downColor: '#FF0001',
        fontFamily: 'Space Mono, monospace',
      },
      live: false,
      volume: true,
      drawings: true,
      drawingToolbar: true,
      watermark: true,
      statusline: true,
      bottombar: true,
      timezone: 'Etc/UTC',
      persist: 'xauusd-interlock-workspace-v1',
      topbar: { left: ['symbol', 'timeframes', 'style', 'layout', 'indicators', 'undo-redo'], right: ['actions', 'alerts', 'panels', 'screenshot'] },
    })
    const updateTimeframe = () => {
      const timeframe = workspace.active.timeframe
      setHistory((current) => ({ ...current, timeframe }))
    }
    const offState = workspace.on('state:changed', updateTimeframe)
    const offActive = workspace.on('cell:active', updateTimeframe)
    return () => {
      offState(); offActive(); workspace.destroy()
    }
  }, [bars])

  const readableTime = history.lastBarTime?.replace('T', ' ').replace('+00:00', 'Z')
  const timeframeLabel = history.timeframe === '60' ? 'H1' : history.timeframe === '240' ? 'H4' : history.timeframe === 'D' || history.timeframe === '1D' ? 'D1' : history.timeframe
  return (
    <section className="chart-section" aria-label="Historical XAUUSD chart">
      <div className="chart-frame">
        <div className="chart-header">
          <div className="chart-symbol"><strong>XAUUSD</strong><span>Gold / US Dollar</span><span className="chart-timeframe">{timeframeLabel}</span></div>
          <span className="historical-tag">Historical · {timeframeLabel} · last {readableTime ?? 'time unavailable'}{history.lastClose == null ? '' : ` · close ${history.lastClose.toFixed(3)}`} · not live</span>
        </div>
        {bars?.length ? <div className="vela-canvas" ref={mountRef} /> : <div className="chart-unavailable"><Unavailable title="Historical bars unavailable" reason={reason ?? 'No verified XAUUSD H1 bars were returned by the market-data reader.'} source={source ?? 'data/xauusd_h1.csv'} /></div>}
        <div className="chart-footer"><SourceLine source={history.source} detail="Historical OHLCV · UTC · no live feed" /><a href="https://velacharts.dev" target="_blank" rel="noreferrer" className="vela-attribution">Vela™ ↗</a></div>
      </div>
    </section>
  )
}
