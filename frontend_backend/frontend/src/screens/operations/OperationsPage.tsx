import { lazy, Suspense } from 'react'
import { useApiResource } from '../../hooks/useApiResource'
import { useSocketChannel } from '../../hooks/useSocketChannel'
import { PermissionBanner } from '../../components/PermissionBanner'
import { ScreenHeader, SectionHeading } from '../../components/ScreenHeader'
import { SourceLine, Unavailable } from '../../components/SourceLine'
import { SourceInspector } from '../../components/SourceInspector'
import type { ClosedTrade, DashboardStatus, MarketBars, Position, Reconciliation, RiskSnapshot } from '../../domain/types'

const VelaWorkspaceHost = lazy(() => import('./VelaWorkspaceHost').then((module) => ({ default: module.VelaWorkspaceHost })))

interface DecisionEvent { ts?: string; bar?: string; filter?: string; risk?: { approved?: boolean; reason?: string } }
interface CalendarEvent { datetime: string; event: string; currency: string; impact: string }

export function OperationsPage() {
  const statusQuery = useApiResource<DashboardStatus>('/api/status', 10_000)
  const barsQuery = useApiResource<MarketBars>('/api/market/bars?symbol=XAUUSD&timeframe=H1&limit=3000')
  const riskQuery = useApiResource<RiskSnapshot>('/api/risk/state', 10_000)
  const accountQuery = useApiResource<{ balance?: number; mid?: number }>('/api/state/account')
  const calendarQuery = useApiResource<CalendarEvent[]>('/api/state/calendar')
  const positionsQuery = useApiResource<Position[]>('/api/state/positions')
  const closedQuery = useApiResource<ClosedTrade[]>('/api/state/closed')
  const reconciliationQuery = useApiResource<Reconciliation>('/api/reconciliation')
  const decisionSocket = useSocketChannel<DecisionEvent[]>('/ws/live/decisions')
  const status = statusQuery.result?.data
  const account = accountQuery.result?.data
  const bars = barsQuery.result?.data
  const risk = riskQuery.result?.data
  const positions = positionsQuery.result?.data
  const closed = closedQuery.result?.data
  const reconciliation = reconciliationQuery.result?.data
  const decisions = decisionSocket.message?.availability === 'available' ? decisionSocket.message.data : undefined
  const calendar = calendarQuery.result?.data

  return (
    <div className="page page-operations">
      <ScreenHeader title="Operations" description="Permission first. Historical market context and source-backed runtime state." side={<div className="operation-heading-actions"><span className="mode-chip">{status?.trading_mode?.toUpperCase() ?? 'MODE UNAVAILABLE'} · {status?.trading_mode_source ?? 'awaiting config reader'}</span><SourceInspector /></div>} />
      <PermissionBanner status={status} />
      <div className="runtime-strip" aria-label="Runtime permissions">
        <div><span>Model signal gate</span><b className="text-blocked">BLOCKED · no production manifest</b></div>
        <div><span>Kill-switch file</span><b>{status?.kill_switch_armed == null ? 'Not checked' : status.kill_switch_armed ? 'Armed' : 'File absent'}</b></div>
        <div><span>Supervisor</span><b>{status?.supervisor === 'available' ? 'Connected' : 'Unavailable'}</b></div>
      </div>
      <Suspense fallback={<div className="chart-loading" role="status">Loading Vela chart workspace…</div>}><VelaWorkspaceHost bars={bars?.bars} source={barsQuery.result?.source ?? 'data/xauusd_h1.csv'} lastBarTime={bars?.last_bar_time} lastClose={bars?.bars.at(-1)?.close} reason={barsQuery.error ?? barsQuery.result?.reason} /></Suspense>
      <div className="operations-grid">
        <section className="data-section account-section">
          <SectionHeading title="Account & risk snapshot" detail={status?.trading_mode === 'demo' ? 'Persisted demo state · not a live market quote' : 'Live account reader unavailable · risk database fields remain persisted'} />
          {risk?.statistics_availability === 'available' ? <div className="metric-table">
            <Metric label={status?.trading_mode === 'demo' ? 'Demo balance' : 'Broker balance'} value={account?.balance == null ? 'Unavailable' : currency(account.balance)} />
            <Metric label={status?.trading_mode === 'demo' ? 'Last persisted midpoint' : 'Broker midpoint'} value={account?.mid == null ? 'Unavailable' : currency(account.mid)} />
            <Metric label="Current equity" value={formatValue(risk.current_equity)} />
            <Metric label="Peak equity" value={formatValue(risk.peak_equity)} />
            <Metric label="Daily PnL" value={formatValue(risk.daily_pnl)} />
            <Metric label="Daily loss ratio" value={percent(risk.daily_loss_ratio)} />
            <Metric label="Trades today" value={formatValue(risk.trades_today)} />
            <Metric label="Consecutive losses" value={formatValue(risk.consecutive_losses)} />
            <Metric label="Risk trade date · UTC" value={risk.trade_date ?? '—'} />
          </div> : <Unavailable title="Risk snapshot unavailable" reason={risk?.statistics_reason ?? riskQuery.error ?? 'No persisted risk snapshot could be loaded.'} source={riskQuery.result?.source ?? 'state/risk_state.db'} />}
          <SourceLine source={accountQuery.result?.source ?? 'state/mock_state.json'} detail={accountQuery.result?.reason ?? 'Persisted demo snapshot · timestamp not recorded'} />
          <SourceLine source={riskQuery.result?.source ?? 'state/risk_state.db'} detail="Persisted RiskSupervisor fields · readonly access" />
        </section>
        <section className="data-section">
          <SectionHeading title="Active positions" detail="No close/modify actions are exposed by the frontend." />
          {positions ? positions.length ? <div className="data-table-wrap"><table><thead><tr><th>Ticket</th><th>Side</th><th>Volume</th><th>Entry</th><th>SL</th><th>TP</th><th>Current</th><th>PnL</th></tr></thead><tbody>{positions.map((p) => <tr key={p.ticket}><td>{p.ticket}</td><td>{p.side}</td><td>{p.volume}</td><td>{currency(p.entry)}</td><td>{formatValue(p.sl)}</td><td>{formatValue(p.tp)}</td><td>{formatValue(p.current_price)}</td><td>{formatValue(p.pnl)}</td></tr>)}</tbody></table></div> : <div className="empty-feed">No active positions in the persisted demo state.</div> : <Unavailable title="Position state unavailable" reason={positionsQuery.error ?? 'No state snapshot loaded.'} source={positionsQuery.result?.source ?? 'state/mock_state.json'} />}
        </section>
        <section className="data-section">
          <SectionHeading title="Decision feed" detail={`Live child-process events only · ${decisionSocket.connected ? 'socket connected' : 'socket reconnecting'}`} />
          {decisions?.length ? <div className="event-list">{decisions.slice(-8).reverse().map((event, i) => <div className="event-row" key={`${event.ts ?? 'event'}-${i}`}><b>{event.filter ?? 'Decision event'}</b><span>{event.risk?.reason ?? 'Risk status unavailable'}</span><time>{event.ts ?? 'Timestamp unavailable'}</time></div>)}</div> : <Unavailable title="No decision events loaded" reason="The decision channel currently has no source event to display." source="/ws/live/decisions" />}
        </section>
        <section className="data-section">
          <SectionHeading title="Reconciliation" detail="Broker/local comparison from the live child supervisor" />
          {reconciliation?.availability === 'available' ? <div className={`recon-state ${reconciliation.ok ? 'is-ok' : 'is-failed'}`}><b>{reconciliation.ok ? 'MATCHED' : 'DRIFT DETECTED'}</b><span>Matched {formatValue(reconciliation.matched)} · Mismatched {formatValue(reconciliation.mismatched)} · Drift {formatValue(reconciliation.drift)}</span></div> : <Unavailable title="Reconciliation not loaded" reason={reconciliation?.reason ?? reconciliationQuery.error ?? 'No reconciliation result is available.'} source="live/trade_executor.py · startup_reconcile" />}
        </section>
        <section className="data-section closed-section">
          <SectionHeading title="Closed trades" detail="Persisted demo records · source timestamp not recorded" />
          {closed ? closed.length ? <div className="data-table-wrap"><table><thead><tr><th>Ticket</th><th>Side</th><th>Vol.</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Reason</th></tr></thead><tbody>{closed.slice(-8).reverse().map((t) => <tr key={t.ticket}><td>{t.ticket}</td><td>{t.side}</td><td>{t.volume}</td><td>{currency(t.entry)}</td><td>{currency(t.exit)}</td><td>{currency(t.pnl)}</td><td>{t.reason}</td></tr>)}</tbody></table></div> : <div className="empty-feed">No closed trades recorded.</div> : <Unavailable title="Closed trades unavailable" reason={closedQuery.error ?? 'No state snapshot loaded.'} source={closedQuery.result?.source ?? 'state/mock_state.json'} />}
        </section>
        <section className="data-section">
          <SectionHeading title="Risk decision catalog" detail="Supported reason codes · live counts not loaded" />
          <div className="reason-catalog">{['APPROVED','CIRCUIT_BREAKER','HALTED','MAX_DRAWDOWN','POSITION_TOO_LARGE','TOO_MANY_LOSSES','HIGH_VOLATILITY','CORRELATION_GUARD','EVENT_RISK','MAX_TRADES','COOLDOWN','SPREAD_TOO_WIDE','MARKET_CLOSED'].map((reason) => <span key={reason}>{reason}</span>)}</div>
        </section>
        <section className="data-section">
          <SectionHeading title="Economic calendar" detail="Historical file only · not a current event schedule" />
          {calendar?.length ? <div className="event-list">{calendar.slice(-5).reverse().map((event) => <div className="event-row" key={`${event.datetime}-${event.event}`}><b>{event.event}</b><span>{event.currency} · {event.impact}</span><time>{event.datetime} UTC</time></div>)}</div> : <Unavailable title="Calendar unavailable" reason={calendarQuery.error ?? calendarQuery.result?.reason ?? 'No calendar rows were loaded.'} source="data/economic_events_2015_2025.json" />}
        </section>
      </div>
      <div className="safety-note">HARD SAFETY INVARIANT <span>Frontend never calls broker or order functions. Execution remains inside the live child process.</span></div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="metric-row"><span>{label}</span><b>{value}</b></div> }
function formatValue(value?: number | null) { return value == null ? '—' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(value) }
function currency(value?: number | null) { return value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value) }
function percent(value?: number | null) { return value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 2 }).format(value) }
