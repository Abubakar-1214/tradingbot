import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ScreenHeader, SectionHeading } from '../../components/ScreenHeader'
import { SourceLine } from '../../components/SourceLine'
import { PRESETS } from '../../domain/types'
import type { JobSnapshot, ResumableModel } from '../../domain/types'
import { apiRequest, ApiError } from '../../api/client'
import { useApiResource } from '../../hooks/useApiResource'

const dreamerDefaults: Record<string, string> = {
  steps: '100000', prefill: '5000', 'batch-size': '16', 'train-every': '4', 'seq-len': '64', window: '64',
  'embed-dim': '256', 'hidden-dim': '512', 'stoch-dim': '32', 'num-categories': '32', horizon: '15',
  'train-end': '2024-01-01', data: 'data/xauusd_h1.csv', device: 'auto',
}
const ppoDefaults: Record<string, string> = {
  timesteps: '5000000', 'n-steps': '1024', 'batch-size': '256', window: '70',
  'chunk-steps': '50000', 'max-episode-steps': '4096', 'n-envs': '2', seed: '42',
  'train-end': '2024-01-01', data: 'data/xauusd_h1.csv',
  'turnover-penalty': '0.0005', 'hold-bonus': '0.00005', 'min-hold-bars': '4',
  'early-exit-penalty': '0.0005', 'drawdown-penalty': '0.5',
  'short-penalty': '0.0', 'learning-rate': '0.0001',
}
type PreflightItem = { id: string; label: string; availability: string; passed?: boolean; reason?: string }
type PrepReport = { source: string; contract: string; window: number; n_features: number; obs_dim: number; contract_hash: string; train_rows: number; validation_rows: number; generated_at: string; leakage_check: string }

export function PrepareRunPage() {
  const [searchParams] = useSearchParams()
  const initialResume = searchParams.get('resume') || ''

  const [preset, setPreset] = useState('ppo')
  const [runName, setRunName] = useState(initialResume)
  const [fields, setFields] = useState(ppoDefaults)
  const [macro, setMacro] = useState(false)
  const [slTpModel, setSlTpModel] = useState(false)
  const [allowShort, setAllowShort] = useState(true)
  const [subproc, setSubproc] = useState(false)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)

  // In-place checkpoint continuation state
  const [isResume, setIsResume] = useState(Boolean(initialResume))
  const [selectedResumeRun, setSelectedResumeRun] = useState(initialResume)
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('')
  const lastSyncedRunRef = useRef<string | null>(null)

  const resumableQuery = useApiResource<ResumableModel[]>('/api/models/resumable', 4_000)
  const preflightUrl = `/api/prep/preflight?preset=${encodeURIComponent(preset)}&data=${encodeURIComponent(fields.data)}&train_end=${encodeURIComponent(fields['train-end'])}&window=${encodeURIComponent(fields.window)}&macro=${macro}`
  const preflightQuery = useApiResource<PreflightItem[]>(preflightUrl, 4_000)
  const jobsQuery = useApiResource<JobSnapshot[]>('/api/jobs', 2_000)
  const leakQuery = useApiResource<PrepReport>('/api/prep/leakcheck', 4_000)
  const script = PRESETS.find(([name]) => name === preset)?.[1]
  const isStaged = preset === 'adversarial' || preset === 'meta-train'
  const prepJob = jobsQuery.result?.data?.find((job) => job.preset === 'data-prep')

  // Auto-sync resume metadata ONLY when active model changes, without clobbering user edits on 4s polling
  useEffect(() => {
    const list = resumableQuery.result?.data
    if (!list || list.length === 0) return

    if (isResume) {
      const activeRun = selectedResumeRun || list[0].run_name
      if (lastSyncedRunRef.current !== activeRun) {
        const m = list.find((x) => x.run_name === activeRun) || list[0]
        if (m) {
          lastSyncedRunRef.current = m.run_name
          setSelectedResumeRun(m.run_name)
          setSelectedCheckpoint(m.latest_checkpoint)
          setRunName(m.run_name)
          setPreset(m.model_type)
          setFields((prev) => ({
            ...prev,
            window: String(m.window),
            'train-end': m.train_end || prev['train-end'],
          }))
          setAllowShort(m.allow_short)
          setMacro(Boolean(m.has_macro))
        }
      }
    } else {
      lastSyncedRunRef.current = null
    }
  }, [resumableQuery.result?.data, isResume, selectedResumeRun])

  const { argsPreview, validationError } = buildCommandPreview(
    preset, script, runName, fields, macro, slTpModel, allowShort, subproc, isResume, selectedCheckpoint
  )

  const checks = preflightQuery.result?.data ?? []
  const csvCheck = checks.find((item) => item.id === 'data_file')
  const allPreflightPass = checks.length > 0 && checks.every((item) => item.passed === true)
  const isSupportedPreset = preset === 'dreamer' || preset === 'ppo'
  const canLaunch = isSupportedPreset && !busy && !validationError && Boolean(runName.trim()) && (isResume ? Boolean(selectedCheckpoint) : allPreflightPass)

  async function buildContract() {
    setBusy(true); setMessage('')
    try {
      const response = await apiRequest<JobSnapshot>('/api/prep/run', { method: 'POST', body: JSON.stringify({ data: fields.data, train_end: fields['train-end'], window: Number(fields.window), macro: macro ? 'data/macro_daily.csv' : null }) })
      setMessage(`Feature contract job accepted · ${response.data?.job_id ?? 'supervisor child started'}.`)
    } catch (error) { setMessage(error instanceof ApiError ? error.message : 'Feature contract preparation failed to start.') }
    finally { setBusy(false) }
  }

  async function launch() {
    if (!canLaunch) return
    setBusy(true); setMessage('')
    try {
      const flags: Record<string, string | number | boolean> = {}
      for (const [key, value] of Object.entries(fields)) {
        if (value === '') continue
        if (['turnover-penalty', 'hold-bonus', 'early-exit-penalty', 'drawdown-penalty', 'short-penalty', 'learning-rate', 'gamma'].includes(key)) {
          flags[key] = parseFloat(value)
        } else if (/^\d+$/.test(value)) {
          flags[key] = Number(value)
        } else {
          flags[key] = value
        }
      }
      if (macro) flags.macro = 'data/macro_daily.csv'
      if (preset === 'dreamer' && slTpModel) flags['sl-tp-model'] = true
      if (preset === 'ppo') {
        if (allowShort) flags['allow-short'] = true
        else flags['long-only'] = true
        if (subproc) flags.subproc = true
      }
      if (isResume && selectedCheckpoint) {
        flags.resume = selectedCheckpoint
      }
      const response = await apiRequest<JobSnapshot>('/api/jobs/start', {
        method: 'POST',
        body: JSON.stringify({ preset, run_name: runName.trim(), flags })
      })
      setMessage(`Supervisor accepted ${response.data?.job_id ?? 'the run request'}. In-place directory continuation started.`)
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : 'The supervisor could not start this run.')
    } finally {
      setBusy(false)
    }
  }

  return <div className="page">
    <ScreenHeader title="Prepare Run" description="Data → contract → preset → preflight → in-place resume → supervisor launch." />
    <section className="workflow-steps" aria-label="Run preparation sequence">
      {['Execution mode', 'Data source', 'Preset & parameters', 'Preflight / Checkpoint', 'Launch'].map((item, index) => (
        <div key={item} className={index === (isResume ? 0 : 2) ? 'step-current' : ''}><b>{index + 1}</b><span>{item}</span></div>
      ))}
    </section>
    <div className="prepare-grid">
      <form className="prepare-form" onSubmit={(event) => { event.preventDefault(); void launch() }}>
        {/* In-place Resume Section */}
        <section className="form-section resume-section">
          <SectionHeading title="Execution Mode" detail="Fresh start or in-place checkpoint continuation" />
          <div className="resume-toggle-group">
            <label className="checkbox-row resume-switch">
              <input
                type="checkbox"
                checked={isResume}
                onChange={(event) => {
                  const next = event.target.checked
                  setIsResume(next)
                  if (!next) {
                    lastSyncedRunRef.current = null
                    setSelectedResumeRun('')
                    setSelectedCheckpoint('')
                  } else if (resumableQuery.result?.data?.length) {
                    const first = resumableQuery.result.data[0]
                    lastSyncedRunRef.current = first.run_name
                    setSelectedResumeRun(first.run_name)
                    setSelectedCheckpoint(first.latest_checkpoint)
                    setRunName(first.run_name)
                    setPreset(first.model_type)
                    setFields((prev) => ({
                      ...prev,
                      window: String(first.window),
                      'train-end': first.train_end || prev['train-end'],
                    }))
                    setAllowShort(first.allow_short)
                  }
                }}
              />
              <b>Resume existing model checkpoint</b>
              <span>In-place continuation (same folder)</span>
            </label>
          </div>

          {isResume && (
            <div className="resume-config-box">
              <label className="field-label" htmlFor="resume-model-select">Select checkpoint to resume</label>
              <select
                id="resume-model-select"
                value={selectedResumeRun}
                onChange={(event) => {
                  const name = event.target.value
                  setSelectedResumeRun(name)
                  const m = resumableQuery.result?.data?.find((x) => x.run_name === name)
                  if (m) {
                    lastSyncedRunRef.current = m.run_name
                    setSelectedCheckpoint(m.latest_checkpoint)
                    setRunName(m.run_name)
                    setPreset(m.model_type)
                    setFields((prev) => ({
                      ...prev,
                      window: String(m.window),
                      'train-end': m.train_end || prev['train-end'],
                    }))
                    setAllowShort(m.allow_short)
                    setMacro(Boolean(m.has_macro))
                  }
                }}
              >
                <option value="">-- Choose checkpoint --</option>
                {resumableQuery.result?.data?.map((m) => (
                  <option key={m.run_name} value={m.run_name}>
                    {m.run_name} (Step {m.latest_step.toLocaleString()} · {m.latest_checkpoint_name} · window {m.window} · {m.n_features ?? 15} features)
                  </option>
                ))}
              </select>

              {selectedResumeRun && (
                <div className="resume-banner-success">
                  <div className="resume-banner-title">
                    <span className="badge-same-folder">SAME FOLDER GUARANTEE</span>
                    <b>In-Place Directory Continuation Active</b>
                  </div>
                  <p>
                    Target directory: <code>artifacts/models/{selectedResumeRun}</code>
                    <br />
                    Resuming checkpoint: <code>{selectedCheckpoint}</code>
                    <br />
                    <small>✓ NO new directory will be created. All subsequent checkpoints, model.zip, manifest.json, and evaluations save directly into this existing folder.</small>
                  </p>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="form-section"><SectionHeading title="Data source" detail="Local OHLC path · train-window-only feature fitting" />
          <label className="field-label" htmlFor="data-path">OHLC CSV</label><input id="data-path" value={fields.data} onChange={(event) => setFields({ ...fields, data: event.target.value })} required />
          <label className="field-label" htmlFor="train-end">Train end</label><input id="train-end" type="date" value={fields['train-end']} onChange={(event) => setFields({ ...fields, 'train-end': event.target.value })} required />
          <label className="checkbox-row">
            <input type="checkbox" checked={macro} disabled={isResume} onChange={(event) => setMacro(event.target.checked)} />
            Include daily macro features <span>causal shift(1){isResume ? ' · locked to checkpoint contract' : ''}</span>
          </label>
          {macro && <p className="field-help">Uses data/macro_daily.csv only when present and server-validated.</p>}
        </section>

        <section className="form-section"><SectionHeading title="Preset & target" detail="Each preset name maps to a repository script." />
          <label className="field-label" htmlFor="preset">Training preset</label>
          <select id="preset" value={preset} disabled={isResume} onChange={(event) => {
            const nextPreset = event.target.value
            setPreset(nextPreset)
            setMessage('')
            if (nextPreset === 'ppo') {
              setFields((prev) => ({
                ...ppoDefaults,
                data: prev.data || ppoDefaults.data,
                'train-end': prev['train-end'] || ppoDefaults['train-end'],
                window: prev.window || ppoDefaults.window,
              }))
            } else if (nextPreset === 'dreamer') {
              setFields((prev) => ({
                ...dreamerDefaults,
                data: prev.data || dreamerDefaults.data,
                'train-end': prev['train-end'] || dreamerDefaults['train-end'],
                window: prev.window || dreamerDefaults.window,
              }))
            }
          }}>{PRESETS.map(([name]) => <option key={name} value={name}>{name}</option>)}</select>
          <div className="script-target"><span>Script</span><code>{script}</code></div>

          <label className="field-label" htmlFor="run-name">Run name {isResume ? '(locked to target folder)' : ''}</label>
          <input
            id="run-name"
            value={runName}
            readOnly={isResume}
            style={isResume ? { backgroundColor: 'var(--surface-raised)', borderColor: 'var(--signal-lime)' } : undefined}
            onChange={(event) => setRunName(event.target.value)}
            placeholder="Unique artifact directory name"
            pattern="[A-Za-z0-9_-]+"
            maxLength={64}
            required
          />
          {isStaged && <div className="notice-block">This staged preset requires a compatible Dreamer base manifest. Base selection and this preset’s launch schema are not implemented; launch remains unavailable.</div>}
          {preset !== 'dreamer' && preset !== 'ppo' && <div className="notice-block">This script is present, but the current control plane has not implemented its exact launch-flag schema. It cannot be launched here yet.</div>}
        </section>

        {preset === 'dreamer' && <section className="form-section"><SectionHeading title="Dreamer parameters" detail="CLI flag names match train/train_dreamer.py." />
          <div className="field-grid">{Object.entries(fields).map(([key, value]) => <label key={key} className="field-wrap"><span>{key}</span>{key === 'device' ? <select value={value} onChange={(event) => setFields({ ...fields, [key]: event.target.value })}><option value="auto">auto → CPU</option><option value="cpu">CPU</option></select> : <input type={key === 'train-end' ? 'date' : key === 'data' ? 'text' : 'number'} value={value} onChange={(event) => setFields({ ...fields, [key]: event.target.value })} step="1" />}</label>)}</div>
          <label className="checkbox-row"><input type="checkbox" checked={slTpModel} onChange={(event) => setSlTpModel(event.target.checked)} /> Enable model-decided SL/TP actions <span>75-action mode · optional</span></label>
          <p className="field-help">SL/TP and TP fractions: 0.005 · 0.01 · 0.02 · 0.03 · 0.05. Rules mode remains default when unchecked.</p>
        </section>}

        {preset === 'ppo' && <section className="form-section"><SectionHeading title="PPO parameters (Qadam 1 Anti-Churn Guards)" detail="CLI flag names match train/train_ppo.py." />
          <div className="field-grid">
            {Object.entries(fields).map(([key, value]) => (
              <label key={key} className="field-wrap">
                <span>{key}</span>
                <input
                  type={key === 'train-end' ? 'date' : key === 'data' ? 'text' : ['turnover-penalty', 'hold-bonus', 'early-exit-penalty', 'drawdown-penalty', 'short-penalty', 'learning-rate', 'gamma'].includes(key) ? 'number' : 'number'}
                  step={['turnover-penalty', 'early-exit-penalty', 'short-penalty', 'learning-rate'].includes(key) ? '0.0001' : key === 'hold-bonus' ? '0.00001' : key === 'drawdown-penalty' ? '0.1' : '1'}
                  value={value}
                  onChange={(event) => setFields({ ...fields, [key]: event.target.value })}
                />
              </label>
            ))}
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={allowShort}
              disabled={isResume}
              onChange={(event) => setAllowShort(event.target.checked)}
            />
            Allow short selling
            <span>
              {allowShort ? 'Discrete(3) actions: Flat, Long, Short' : 'Discrete(2) actions: Flat, Long'}
              {isResume ? ' · Locked to checkpoint architecture' : ''}
            </span>
          </label>
          {isResume && allowShort && (
            <p className="field-help" style={{ color: '#e5a93c', margin: '4px 0 10px 0' }}>
              💡 Checkpoint network has 3 action neurons. Unchecking this would cause PyTorch shape mismatch on load. To stop the model from taking short trades, set <b>short-penalty</b> (e.g. <code>0.001</code> = 10 bps) in parameters above.
            </p>
          )}
          <label className="checkbox-row"><input type="checkbox" checked={subproc} onChange={(event) => setSubproc(event.target.checked)} /> Subprocess vector environment <span>Multiprocess env workers</span></label>
        </section>}

        {!isResume && (
          <section className="data-section contract-action">
            <SectionHeading title="Build feature contract" detail="Runs the actual train.data preparation pipeline as a child process." />
            <p className="field-help">This operation writes train/feature_contract.json and a source report only after the child succeeds.</p>
            <button className="button button-outline" type="button" disabled={busy || !csvCheck?.passed || prepJob?.status === 'running'} onClick={() => void buildContract()}>
              {busy && !canLaunch ? 'Submitting…' : 'Build feature contract'}
            </button>
            {prepJob && <span className="job-feedback">Prep child · {prepJob.status} · log {prepJob.log_path ?? 'unavailable'}</span>}
            {leakQuery.result?.availability === 'available' && <div className="contract-result"><b>CONTRACT SOURCE REPORT</b><span>window {leakQuery.result.data?.window} · {leakQuery.result.data?.n_features} features · obs_dim {leakQuery.result.data?.obs_dim}</span><code>hash {leakQuery.result.data?.contract_hash}</code><p>{leakQuery.result.data?.leakage_check}</p></div>}
          </section>
        )}

        {validationError && <p className="form-error" role="alert">{validationError}</p>}
        {message && <p className="form-message" role="status">{message}</p>}
        <button className="button button-primary launch-button" type="submit" disabled={!canLaunch}>
          {busy ? 'Submitting…' : isResume ? `Resume training in artifacts/models/${runName}` : 'Launch supervised run'}
        </button>
      </form>

      <aside className="preflight-column">
        <section className="data-section">
          <SectionHeading title={isResume ? "Resume Status" : "Preflight"} detail={isResume ? "Checkpoint verified in-place" : "Every item must be server-verified before launch."} />
          {isResume ? (
            <div className="resume-preflight-info">
              <div className="preflight-pass" style={{ padding: '12px', border: '1px solid var(--signal-lime)', background: 'var(--surface-raised)', marginBottom: '10px' }}>
                <b style={{ color: 'var(--signal-lime)' }}>✓ RESUME IN-PLACE VERIFIED</b>
                <p style={{ margin: '6px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>
                  Checkpoint is verified inside <code>artifacts/models/{runName}</code>.
                  New steps will directly append to this model without creating any secondary folder.
                </p>
              </div>
            </div>
          ) : (
            checks.length ? <ul className="preflight-list">{checks.map((item) => <li key={item.id} className={item.passed ? 'preflight-pass' : ''}><span>{item.passed ? 'PASS' : item.availability === 'not_checked' ? 'NOT CHECKED' : 'BLOCKED'}</span><div><b>{item.label}</b><small>{item.reason ?? ''}</small></div></li>)}</ul> : <div className="empty-feed">{preflightQuery.error ?? 'Preflight checks are loading.'}</div>
          )}
        </section>
        <section className="data-section">
          <SectionHeading title="Observation contract" detail="Derived from the selected feature contract." />
          <div className="equation">obs_dim = window × n_features + 5</div>
          <p className="field-help">The five account scalars: position, trade PnL, bars in trade, drawdown and equity ratio.</p>
          <SourceLine source="core/model_artifacts.py · load_manifest()" />
        </section>
        <section className="command-preview">
          <SectionHeading title="Command preview" detail="Preview only · server constructs argv" />
          <pre>{argsPreview || 'Correct the form to preview the validated command.'}</pre>
          <p>The browser never executes commands. API availability does not imply the run started.</p>
        </section>
      </aside>
    </div>
  </div>
}

function buildCommandPreview(
  preset: string, script: string | undefined, runName: string,
  fields: Record<string, string>, macro: boolean, slTpModel: boolean,
  allowShort: boolean, subproc: boolean,
  isResume: boolean, selectedCheckpoint: string
) {
  if (preset !== 'dreamer' && preset !== 'ppo') return { argsPreview: script ?? '', validationError: '' }
  if (preset === 'dreamer') {
    const numericKeys = ['steps', 'prefill', 'batch-size', 'train-every', 'seq-len', 'window', 'embed-dim', 'hidden-dim', 'stoch-dim', 'num-categories', 'horizon']
    const bounds: Record<string, [number, number]> = { steps: [1, 100_000_000], prefill: [0, 10_000_000], 'batch-size': [1, 4096], 'train-every': [1, 100_000], 'seq-len': [1, 4096], window: [1, 4096], 'embed-dim': [8, 8192], 'hidden-dim': [8, 8192], 'stoch-dim': [1, 2048], 'num-categories': [1, 1024], horizon: [1, 1000] }
    for (const key of numericKeys) {
      const value = Number(fields[key])
      const [minimum, maximum] = bounds[key]
      if (!Number.isInteger(value) || value < minimum || value > maximum) return { argsPreview: '', validationError: `${key} must be an integer between ${minimum} and ${maximum}.` }
    }
    const args = Object.entries(fields).filter(([, value]) => value !== '').map(([key, value]) => `--${key} ${value}`)
    if (macro) args.push('--macro data/macro_daily.csv')
    if (slTpModel) args.push('--sl-tp-model')
    if (isResume && selectedCheckpoint) args.push(`--resume ${selectedCheckpoint}`)
    return { argsPreview: `${script} --run-name ${runName || '<required>'} ${args.join(' ')}`, validationError: '' }
  } else {
    const numericKeys = ['timesteps', 'n-steps', 'batch-size', 'window', 'chunk-steps', 'max-episode-steps', 'n-envs', 'seed', 'min-hold-bars']
    const floatKeys = ['turnover-penalty', 'hold-bonus', 'early-exit-penalty', 'drawdown-penalty', 'short-penalty', 'learning-rate', 'gamma']
    const bounds: Record<string, [number, number]> = {
      timesteps: [1, 100_000_000], 'n-steps': [1, 100_000], 'batch-size': [1, 4096], window: [1, 4096],
      'chunk-steps': [1, 1_000_000], 'max-episode-steps': [1, 100_000], 'n-envs': [1, 64], seed: [0, 2147483647],
      'min-hold-bars': [0, 100],
    }
    const floatBounds: Record<string, [number, number]> = {
      'turnover-penalty': [0, 0.1], 'hold-bonus': [0, 0.01],
      'early-exit-penalty': [0, 0.1], 'drawdown-penalty': [0, 10],
      'short-penalty': [0, 0.1], 'learning-rate': [1e-7, 1.0], 'gamma': [0.5, 0.9999],
    }
    for (const key of numericKeys) {
      if (!(key in fields)) continue
      const value = Number(fields[key])
      const [minimum, maximum] = bounds[key]
      if (!Number.isInteger(value) || value < minimum || value > maximum) return { argsPreview: '', validationError: `${key} must be an integer between ${minimum} and ${maximum}.` }
    }
    for (const key of floatKeys) {
      if (!(key in fields)) continue
      const value = Number(fields[key])
      const [minimum, maximum] = floatBounds[key]
      if (isNaN(value) || value < minimum || value > maximum) return { argsPreview: '', validationError: `${key} must be a number between ${minimum} and ${maximum}.` }
    }
    const args = Object.entries(fields).filter(([, value]) => value !== '').map(([key, value]) => `--${key} ${value}`)
    if (macro) args.push('--macro data/macro_daily.csv')
    if (allowShort) args.push('--allow-short')
    else args.push('--long-only')
    if (subproc) args.push('--subproc')
    if (isResume && selectedCheckpoint) args.push(`--resume ${selectedCheckpoint}`)
    return { argsPreview: `${script} --run-name ${runName || '<required>'} ${args.join(' ')}`, validationError: '' }
  }
}
