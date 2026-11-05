import { NavLink, Outlet } from 'react-router-dom'
import {
  AuditOutlined,
  BarChartOutlined,
  ControlOutlined,
  MenuOutlined,
  SafetyOutlined,
} from '@ant-design/icons'
import { useApiResource } from '../hooks/useApiResource'
import type { DashboardStatus } from '../domain/types'

const navigation = [
  { to: '/', label: 'Operations', icon: BarChartOutlined, end: true },
  { to: '/training', label: 'Training', icon: ControlOutlined },
  { to: '/prepare', label: 'Prepare Run', icon: AuditOutlined },
  { to: '/models', label: 'Models / Evaluation', icon: BarChartOutlined },
  { to: '/risk', label: 'Risk / Settings', icon: SafetyOutlined },
]

export function AppShell() {
  const { result, error } = useApiResource<DashboardStatus>('/api/status', 10_000)
  const status = result?.data
  const connected = result?.availability === 'available' && !error

  return (
    <div className="app-frame">
      <header className="topbar">
        <NavLink to="/" className="brand-mark" aria-label="XAUUSD terminal, Operations">
          <span>XAU</span><b>USD</b>
        </NavLink>
        <div className="brand-copy">
          <strong>Autonomous trading system</strong>
          <span>Operations control plane</span>
        </div>
        <div className="topbar-state">
<div className="mode-tag"><span>TRADING_MODE</span><b>{status?.trading_mode ? `${status.trading_mode.toUpperCase()} · CONFIGURED` : 'UNAVAILABLE'}</b></div>
          <div className={`connection-tag ${connected ? 'is-available' : 'is-unavailable'}`} role="status">
            <i aria-hidden="true" />{connected ? 'CONTROL PLANE CONNECTED' : 'BACKEND UNAVAILABLE'}
          </div>
        </div>
        <button className="mobile-nav-toggle" type="button" aria-label="Open workspace navigation" onClick={() => document.body.classList.toggle('nav-open')}>
          <MenuOutlined />
        </button>
      </header>

      <div className="workspace-layout">
        <aside className="workspace-rail" aria-label="Workspace navigation">
          <div className="rail-label">Workspace</div>
          <nav className="primary-nav">
            {navigation.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`} onClick={() => document.body.classList.remove('nav-open')}>
                <Icon aria-hidden="true" /> <span>{label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="rail-baseline">
            <div className="rail-label">Permission</div>
            <div className="rail-demo"><span className="state-dot" />{status?.trading_mode ?? 'unavailable'} <small>· configured</small></div>
            <div className="rail-gate"><SafetyOutlined /> Model execution blocked</div>
          </div>
        </aside>

        <main className="screen-content"><Outlet /></main>
      </div>
      <footer className="app-footer">
        <span>Frontend never calls broker or order functions.</span>
        <span>Supervisor actions only · all timestamps UTC</span>
      </footer>
      <button className="nav-scrim" aria-label="Close navigation" type="button" onClick={() => document.body.classList.remove('nav-open')} />
    </div>
  )
}
