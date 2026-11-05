import { LockOutlined, SafetyOutlined } from '@ant-design/icons'
import type { DashboardStatus } from '../domain/types'

export function PermissionBanner({ status }: { status?: DashboardStatus }) {
  const facts = [
    ['Trading mode', status?.trading_mode ? `${status.trading_mode} · configured` : 'Unavailable · control plane not connected'],
    ['Signal source', status?.signal_source ?? 'Unavailable · config not loaded'],
    ['Production manifest', status?.production_manifest === 'available' ? 'Present' : status?.production_manifest === 'unavailable' ? 'Absent' : 'Unavailable · not verified'],
    ['Evaluation gate', status?.evaluations_passed === 0 && (status.evaluations_inspected ?? 0) > 0 ? `${status.evaluations_inspected} inspected · passed=false` : status?.evaluations_passed != null ? `${status.evaluations_passed} passed` : 'Unavailable · not verified'],
    ['REQUIRE_PROMOTED_MODEL', status?.require_promoted_model === true ? 'true · configured' : status?.require_promoted_model === false ? 'false · configured' : 'Unavailable · config not loaded'],
  ]

  return (
    <section className="permission-placard" aria-labelledby="permission-title">
      <div className="permission-verdict">
        <div className="permission-label"><LockOutlined aria-hidden="true" /> Model permission</div>
        <h2 id="permission-title">MODEL-DRIVEN<br />TRADING BLOCKED</h2>
        <p>Model-signal execution cannot start until a passed artifact is promoted and model-specific live gates are verified.</p>
      </div>
      <div className="permission-facts">
        <div className="permission-facts-title"><SafetyOutlined aria-hidden="true" /> Permission basis</div>
        {facts.map(([label, value]) => <div className="permission-fact" key={label}><strong>{label}</strong><span>{value}</span></div>)}
        <div className="runtime-gates-list">
          <span className="runtime-gates-title">Runtime gate checks</span>
          {status?.runtime_gates?.length ? status.runtime_gates.map((gate) => <div className="permission-gate-row" key={gate.name}><span>{gate.name}</span><b className={gate.passed === true ? 'gate-readable-pass' : gate.passed === false ? 'gate-readable-fail' : ''}>{gate.passed === true ? 'PASS' : gate.passed === false ? 'BLOCKED' : 'NOT CHECKED'}</b></div>) : <p>Unavailable · no runtime gate snapshot</p>}
        </div>
      </div>
    </section>
  )
}
