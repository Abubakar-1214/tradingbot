import { useMemo, useState } from 'react'
import { motion } from 'motion/react'
import { SafetyOutlined } from '@ant-design/icons'
import { ScreenHeader, SectionHeading } from '../../components/ScreenHeader'
import { SourceLine, Unavailable } from '../../components/SourceLine'
import { apiRequest, ApiError } from '../../api/client'
import { useApiResource } from '../../hooks/useApiResource'
import { RISK_FIELDS, RISK_REASONS, type DashboardStatus, type RiskFieldKey, type RiskSnapshot } from '../../domain/types'

export function RiskSettingsPage() {
  const statusQuery = useApiResource<DashboardStatus>('/api/status', 10_000)
  const riskQuery = useApiResource<RiskSnapshot>('/api/risk/state', 10_000)
  const snapshot = riskQuery.result?.data
  const [draft, setDraft] = useState<Partial<Record<RiskFieldKey, string>>>({})
  const [review, setReview] = useState(false)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const changed = Object.entries(draft).filter(([, value]) => value !== undefined && value !== '')
  const validation = useMemo(() => validate(draft), [draft])
  const canSave = review && changed.length > 0 && !validation && !busy && riskQuery.result?.availability === 'available'

  function updateDraft(key: RiskFieldKey, value: string) {
    setDraft((current) => ({ ...current, [key]: value }))
    setReview(false)
    setMessage('')
  }

  async function save() {
    if (!canSave) return
    setBusy(true); setMessage('')
    try {
      await apiRequest('/api/config/env', { method: 'PUT', body: JSON.stringify({ changes: Object.fromEntries(changed.map(([key, value]) => [key, parseInput(String(value))])) }) })
      setMessage('Validated configuration saved. Changes take effect on next process start.')
      setDraft({}); setReview(false); void riskQuery.refresh()
    } catch (error) { setMessage(error instanceof ApiError ? error.message : 'Configuration was not written.') }
    finally { setBusy(false) }
  }

  async function setKillSwitch(armed: boolean) {
    const action = armed ? 'write' : 'remove'
    if (!window.confirm(`${armed ? 'Arm' : 'Disarm'} kill switch by ${action} the configured sentinel file? This does not send a broker or order request.`)) return
    try { await apiRequest('/api/live/killswitch', { method: 'POST', body: JSON.stringify({ armed }) }); void statusQuery.refresh() }
    catch (error) { setMessage(error instanceof ApiError ? error.message : 'Kill switch was not changed.') }
  }

  return <div className="page">
    <ScreenHeader title="Risk / Settings" description="Resolved risk state, proposed limits and the kill-switch file." side={<span className="mode-chip">TRADING_MODE · {statusQuery.result?.data?.trading_mode?.toUpperCase() ?? 'UNAVAILABLE'}</span>} />
    <section className="permission-placard risk-permission">
      <div className="permission-verdict"><div className="permission-label"><SafetyOutlined /> Model permission</div><h2>MODEL-DRIVEN<br />TRADING BLOCKED</h2><p>Model-signal execution cannot start until a passed artifact is promoted and model-specific live gates are verified. Rule-signal execution is a separate configured path.</p></div>
      <div className="permission-facts"><div className="permission-facts-title">Permission basis</div>{[
        ['Trading mode', statusQuery.result?.data?.trading_mode ?? 'Unavailable'],
        ['Signal source', statusQuery.result?.data?.signal_source ?? 'Unavailable'],
        ['Production manifest', statusQuery.result?.data?.production_manifest === 'available' ? 'Present' : statusQuery.result?.data?.production_manifest === 'unavailable' ? 'Absent' : 'Unavailable'],
        ['Evaluation gate', statusQuery.result?.data?.evaluations_passed === 0 && (statusQuery.result?.data?.evaluations_inspected ?? 0) > 0 ? `${statusQuery.result.data.evaluations_inspected} inspected · passed=false` : 'Not verified'],
        ['REQUIRE_PROMOTED_MODEL', statusQuery.result?.data?.require_promoted_model == null ? 'Unavailable' : String(statusQuery.result.data.require_promoted_model)],
      ].map(([label, value]) => <div className="permission-fact" key={label}><strong>{label}</strong><span>{value}</span></div>)}
      <div className="runtime-gates-list"><span className="runtime-gates-title">Runtime gate checks</span>{statusQuery.result?.data?.runtime_gates?.map((gate) => <div className="permission-gate-row" key={gate.name}><span>{gate.name}</span><b className={gate.passed === true ? 'gate-readable-pass' : gate.passed === false ? 'gate-readable-fail' : ''}>{gate.passed === true ? 'PASS' : gate.passed === false ? 'BLOCKED' : 'NOT CHECKED'}</b></div>)}</div></div>
    </section>

    <section className="risk-limits-section">
      <div className="section-heading"><div><h2>Risk limits</h2><p>Propose values, review the diff, then request a validated .env write.</p></div><span className="effect-tag">Changes take effect on next process start</span></div>
      <div className="risk-table-wrap"><table className="risk-table"><thead><tr><th>Whitelisted key</th><th>Current</th><th>Proposed change</th></tr></thead><tbody>{RISK_FIELDS.map((field) => <tr key={field.key}>
        <th scope="row"><label htmlFor={`risk-${field.key}`}>{field.key}<small>{field.label}</small></label></th>
        <td>{snapshot?.config ? formatCurrent(snapshot.config[field.key]) : 'Not loaded'}</td>
        <td>{field.kind === 'boolean' ? <select id={`risk-${field.key}`} aria-label={`Proposed ${field.key}`} value={draft[field.key] ?? ''} onChange={(event) => updateDraft(field.key, event.target.value)}><option value="">No proposal</option><option value="true">true</option><option value="false">false</option></select> : <input id={`risk-${field.key}`} aria-label={`Proposed ${field.key}`} type={field.kind === 'text' ? 'text' : 'number'} step={field.kind === 'integer' ? '1' : 'any'} placeholder="No proposal" value={draft[field.key] ?? ''} onChange={(event) => updateDraft(field.key, event.target.value)} />}</td>
      </tr>)}</tbody></table></div>
      {!snapshot && <SourceLine source={riskQuery.result?.source ?? 'core/config.py · state/risk_state.db'} detail={riskQuery.error ?? riskQuery.result?.reason ?? 'Resolved configuration is not loaded.'} />}
      <div className="review-box"><div><b>Review before write</b><p>{changed.length ? `${changed.length} proposed value${changed.length === 1 ? '' : 's'} · ${review ? 'Review open' : 'Review required'}` : 'Proposed changes: empty · Nothing written'}</p>{validation && <p className="form-error" role="alert">{validation}</p>}{message && <p className="form-message" role="status">{message}</p>}</div><div className="review-actions"><button className="button button-outline" disabled={!changed.length || Boolean(validation)} onClick={() => setReview(!review)}>{review ? 'Hide review' : 'Review proposed changes'}</button><button className="button button-primary" disabled={!canSave} onClick={() => void save()}>{busy ? 'Saving…' : 'Validate & save'}</button></div></div>
      {review && changed.length > 0 && <div className="diff-review" aria-label="Proposed configuration diff">{changed.map(([key, value]) => <div key={key}><code>{key}</code><span>current {snapshot?.config?.[key] == null ? 'Not loaded' : String(snapshot.config[key])}</span><b>→ {String(value)}</b></div>)}</div>}
    </section>

    <div className="risk-lower-grid"><section className="data-section"><SectionHeading title="Risk state" detail="RiskSupervisor persisted state · no counters synthesized" />{snapshot?.statistics_availability === 'available' ? <div className="metric-table">{[['Current equity', snapshot.current_equity], ['Peak equity', snapshot.peak_equity], ['Daily PnL', snapshot.daily_pnl], ['Daily loss ratio', snapshot.daily_loss_ratio], ['Trades today', snapshot.trades_today], ['Consecutive losses', snapshot.consecutive_losses], ['Halt remaining (s)', snapshot.halt_remaining_sec], ['Trade date · UTC', snapshot.trade_date], ['Approval / rejection counters', null]].map(([label, value]) => <div className="metric-row" key={String(label)}><span>{label}</span><b>{value == null ? 'Not persisted / unavailable' : String(value)}</b></div>)}</div> : <Unavailable title="Risk state not loaded" reason={snapshot?.statistics_reason ?? riskQuery.error ?? 'No SQLite risk snapshot was returned.'} source={riskQuery.result?.source ?? 'state/risk_state.db'} />}</section>
      <div className="risk-right-column"><section className="data-section"><SectionHeading title="Supported reason catalog" detail="13 identifiers · not current events" /><div className="reason-catalog">{RISK_REASONS.map((item) => <span key={item}>{item}</span>)}</div><p className="field-help">No active rejection counts loaded. Source: models/risk_supervisor.py</p></section>
      <section className="kill-switch-section"><div className="section-heading"><div><h2>Kill switch</h2><p>Sentinel file only · no broker or order function.</p></div><span className="file-state">{statusQuery.result?.data?.kill_switch_armed == null ? 'Not checked' : statusQuery.result.data.kill_switch_armed ? 'FILE PRESENT' : 'FILE ABSENT'}</span></div><code>KILL_SWITCH</code><motion.button className="button button-danger" type="button" whileTap={{ scale: 0.985 }} transition={{ duration: 0.12 }} disabled={statusQuery.result?.availability !== 'available' || statusQuery.result.data?.kill_switch_armed == null} onClick={() => void setKillSwitch(!statusQuery.result?.data?.kill_switch_armed)}>{statusQuery.result?.data?.kill_switch_armed ? 'Disarm kill switch' : 'Arm kill switch'}</motion.button></section></div></div>
    <p className="safety-note">HARD SAFETY INVARIANT <span>Frontend never calls broker or order functions.</span></p>
  </div>
}

function validate(draft: Partial<Record<RiskFieldKey, string>>) {
  for (const field of RISK_FIELDS) {
    const raw = draft[field.key]
    if (!raw) continue
    if (field.kind === 'boolean') { if (!['true','false'].includes(raw)) return `${field.key} must be true or false.`; continue }
    if (field.kind === 'text') { if (!/^[A-Za-z0-9._-]{1,32}$/.test(raw)) return `${field.key} may contain letters, numbers, dot, underscore, or hyphen only.`; continue }
    const value = Number(raw)
    if (!Number.isFinite(value)) return `${field.key} must be numeric.`
    if (field.kind === 'integer' && !Number.isInteger(value)) return `${field.key} must be an integer.`
    if (field.kind === 'fraction' && (value < 0 || value >= 1)) return `${field.key} must be in [0, 1).`
    if (value < 0) return `${field.key} must be non-negative.`
  }
  return ''
}
function parseInput(raw: string): string | number | boolean { if (raw === 'true') return true; if (raw === 'false') return false; const n = Number(raw); return raw.trim() !== '' && Number.isFinite(n) ? n : raw }
function formatCurrent(value: unknown) { return value == null ? 'Not loaded' : String(value) }
