import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ScreenHeader, SectionHeading } from '../../components/ScreenHeader'
import { SourceLine, Unavailable } from '../../components/SourceLine'
import { useApiResource } from '../../hooks/useApiResource'
import type { EvaluationArtifact } from '../../domain/types'

const thresholds = [['Sharpe', 'sharpe', '≥ 0.5'], ['Max drawdown', 'max_dd_pct', '≤ 20%'], ['Trade count', 'trades', '≥ 20'], ['Total return', 'total_return_pct', '> 0%']] as const

export function ModelsPage() {
  const query = useApiResource<EvaluationArtifact[]>('/api/models', 30_000)
  const [filter, setFilter] = useState('')
  const [sort, setSort] = useState<'run_name' | 'sharpe' | 'total_return_pct'>('run_name')
  const rows = useMemo(() => [...(query.result?.data ?? [])].filter((row) => row.run_name.toLowerCase().includes(filter.toLowerCase())).sort((a, b) => {
    const av = sort === 'run_name' ? a.run_name : a.metrics[sort] ?? Number.NEGATIVE_INFINITY
    const bv = sort === 'run_name' ? b.run_name : b.metrics[sort] ?? Number.NEGATIVE_INFINITY
    return typeof av === 'string' ? av.localeCompare(String(bv)) : Number(bv) - Number(av)
  }), [filter, query.result?.data, sort])
  return <div className="page">
    <ScreenHeader title="Models / Evaluation" description="Artifact evaluations are evidence. Promotion requires every gate." />
    <section className="production-slot"><div><span className="section-caption">Production slot</span><h2>NONE TRAINED</h2><p>No production manifest is present. Evaluation artifacts do not occupy this slot.</p></div><code>artifacts/models/production/manifest.json</code></section>
    <section className="data-section threshold-section"><SectionHeading title="Promotion thresholds" detail="Every condition is required by train/evaluate.py." /><div className="threshold-grid">{thresholds.map(([label, key, required]) => <div key={key}><span>{label}</span><b>{required}</b><small>Actual value evaluated per artifact</small></div>)}</div></section>
    <section className="data-section"><SectionHeading title="Evaluation ledger" detail="Raw metrics are read from artifacts/models/*/evaluation.json" action={<div className="table-controls"><input aria-label="Filter artifact runs" placeholder="Filter run name" value={filter} onChange={(e) => setFilter(e.target.value)} /><label>Sort <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)}><option value="run_name">Run name</option><option value="sharpe">Sharpe</option><option value="total_return_pct">Total return</option></select></label></div>} />
      {rows.length ? <div className="data-table-wrap"><table className="evaluation-table"><thead><tr><th>Artifact / model</th><th>Sharpe<br /><small>≥ 0.5</small></th><th>Max DD<br /><small>≤ 20%</small></th><th>Trades<br /><small>≥ 20</small></th><th>Total return<br /><small>&gt; 0%</small></th><th>Buy & hold</th><th>Gate</th><th>Action</th></tr></thead><tbody>{rows.map((row) => <tr key={row.run_name}><td><b>{row.run_name}</b><small>{row.model_type} · {row.evaluated_at}</small><small className="path-text">{row.source_path}</small></td><td>{num(row.metrics.sharpe)}</td><td>{pct(row.metrics.max_dd_pct)}</td><td>{num(row.metrics.trades)}</td><td>{pct(row.metrics.total_return_pct)}</td><td>{pct(row.metrics.buy_hold_return_pct)}</td><td><span className={row.passed ? 'gate-pass' : 'gate-fail'}>{row.passed ? 'PROMOTABLE' : 'NOT PROMOTABLE'}</span><small>{row.failures.join(', ') || 'All thresholds passed'}</small></td><td><Link to={`/prepare?resume=${encodeURIComponent(row.run_name)}`} className="button button-outline button-small">Resume</Link></td></tr>)}</tbody></table></div> : <Unavailable title="Evaluation files unavailable" reason={query.error ?? query.result?.reason ?? 'No evaluation artifacts were loaded.'} source={query.result?.source ?? 'artifacts/models/*/evaluation.json'} />}
      <SourceLine source={query.result?.source ?? 'artifacts/models/*/evaluation.json'} detail="Values preserve full precision in the source file" />
    </section>
  </div>
}
function num(value?: number) { return value == null ? '—' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value) }
function pct(value?: number) { return value == null ? '—' : `${new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value)}%` }
