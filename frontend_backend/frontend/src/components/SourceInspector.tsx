import { useEffect, useRef } from 'react'
import { Dialog, ensureUIHost } from '@luxalgo/vela/ui'
import { LinkOutlined } from '@ant-design/icons'

const theme = {
  background: '#0A0D03', textColor: '#F3F5E8', gridColor: '#262D05',
  borderColor: '#3A4120', upColor: '#DEFF00', downColor: '#FF0001',
  fontFamily: 'Saira, sans-serif',
}

export function SourceInspector() {
  const hostRef = useRef<HTMLDivElement>(null)
  const dialogRef = useRef<Dialog | null>(null)
  useEffect(() => () => dialogRef.current?.destroy(), [])

  function open() {
    if (!hostRef.current) return
    ensureUIHost(hostRef.current, theme)
    if (!dialogRef.current) {
      dialogRef.current = new Dialog({
        host: hostRef.current,
        title: 'Data provenance',
        contained: true,
        closeOnEscape: true,
        closeOnInteractOutside: true,
        dimBackdrop: true,
        content: (body) => {
          const rows = [
            ['Market history', 'data/xauusd_h1.csv · XAUUSD H1 · historical OHLCV · UTC'],
            ['Account and trade state', 'state/mock_state.json · persisted demo-broker state only; not a live quote'],
            ['Risk snapshot', 'state/risk_state.db · read-only SQLite query; no counters inferred'],
            ['Evaluations', 'artifacts/models/*/evaluation.json · raw metrics and passed flag'],
            ['Production slot', 'artifacts/models/production/manifest.json · absent in repository'],
            ['Training telemetry', 'Supervisor stdout/stderr + /ws/training · unavailable before job launch'],
            ['Execution boundary', 'Frontend → FastAPI supervisor only · no broker/order calls from UI'],
          ]
          const list = document.createElement('dl')
          list.className = 'vela-source-list'
          for (const [label, value] of rows) {
            const dt = document.createElement('dt'); dt.textContent = label
            const dd = document.createElement('dd'); dd.textContent = value
            list.append(dt, dd)
          }
          body.appendChild(list)
        },
      })
    }
    dialogRef.current.show()
  }

  return <><button className="source-inspector-trigger" type="button" onClick={open}><LinkOutlined aria-hidden="true" /> Source details</button><div ref={hostRef} className="vela-dialog-host" /></>
}
