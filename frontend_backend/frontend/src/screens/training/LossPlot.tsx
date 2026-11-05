type Sample = { training_step: number } & Record<string, number | null>

const series = [
  ['world_model_loss', '#DEFF00'], ['recon_loss', '#8FD3D9'], ['reward_loss', '#D4B96A'],
  ['kl_loss', '#B8A0E0'], ['value_loss', '#9BC77B'], ['policy_loss', '#82A8D6'], ['entropy', '#E6E7D4'],
] as const

export function LossPlot({ samples }: { samples: Sample[] }) {
  const finite = samples.flatMap((sample) => series.flatMap(([key]) => {
    const value = sample[key]
    return typeof value === 'number' && Number.isFinite(value) ? [value] : []
  }))
  if (!samples.length || !finite.length) {
    return <div className="empty-chart"><div className="chart-axis-labels"><span>VALUE · SOURCE SCALE</span><span>training_step →</span></div><div className="plot-empty-copy">No metric samples received · no series fabricated</div></div>
  }
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  const range = max - min || 1
  const width = 900
  const height = 230
  const padX = 34
  const padY = 20
  const pointsFor = (key: string) => samples.flatMap((sample, index) => {
    const value = sample[key]
    if (typeof value !== 'number' || !Number.isFinite(value)) return []
    const x = padX + (samples.length === 1 ? (width - padX * 2) / 2 : index / (samples.length - 1) * (width - padX * 2))
    const y = padY + (1 - (value - min) / range) * (height - padY * 2)
    return [`${x},${y}`]
  }).join(' ')

  return <div className="metric-plot-wrap">
    <div className="metric-plot-summary"><span>{samples.length} source samples</span><span>step {samples[0].training_step} → {samples[samples.length - 1].training_step}</span><span>common value scale · {min.toPrecision(3)} to {max.toPrecision(3)}</span></div>
    <svg className="metric-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Dreamer metric values recorded by the training child process">
      {[0, .25, .5, .75, 1].map((part) => <line key={part} x1={padX} x2={width - padX} y1={padY + part * (height - padY * 2)} y2={padY + part * (height - padY * 2)} className="plot-gridline" />)}
      {series.map(([key, color]) => {
        const points = pointsFor(key)
        return points ? <polyline key={key} points={points} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round"><title>{key}</title></polyline> : null
      })}
      <text x={padX} y={height - 2} className="plot-axis-text">{samples[0].training_step}</text>
      <text x={width - padX} y={height - 2} textAnchor="end" className="plot-axis-text">{samples[samples.length - 1].training_step}</text>
    </svg>
    <div className="plot-legend">{series.filter(([key]) => samples.some((sample) => typeof sample[key] === 'number' && Number.isFinite(sample[key]))).map(([key, color]) => <span key={key}><i style={{ backgroundColor: color }} />{key}</span>)}</div>
    <div className="data-table-wrap"><table><thead><tr><th>training_step</th>{series.map(([key]) => <th key={key}>{key}</th>)}</tr></thead><tbody>{samples.slice(-5).reverse().map((sample) => <tr key={sample.training_step}><td>{sample.training_step}</td>{series.map(([key]) => <td key={key}>{typeof sample[key] === 'number' && Number.isFinite(sample[key]) ? sample[key]?.toPrecision(6) : '—'}</td>)}</tr>)}</tbody></table></div>
  </div>
}
