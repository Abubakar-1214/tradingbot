import { Link } from 'react-router-dom'
import { useApiResource } from '../../hooks/useApiResource'
import { useSocketChannel } from '../../hooks/useSocketChannel'
import { ScreenHeader, SectionHeading } from '../../components/ScreenHeader'
import { SourceLine, Unavailable } from '../../components/SourceLine'
import { LossPlot } from './LossPlot'
import { apiRequest, ApiError } from '../../api/client'
import type { EvaluationArtifact, JobSnapshot } from '../../domain/types'

interface MetricsSample { training_step: number; [key: string]: number | null }
interface TrainingStream { jobs: JobSnapshot[]; active_job: JobSnapshot | null; metrics_samples: MetricsSample[]; metrics_source?: string | null }
interface LogPayload { lines: string[]; source: string }
const dreamerKeys = ['world_model_loss', 'recon_loss', 'reward_loss', 'kl_loss', 'value_loss', 'policy_loss', 'entropy']

export function TrainingPage() {
  const jobsQuery = useApiResource<JobSnapshot[]>('/api/jobs', 3_000)
  const trainingSocket = useSocketChannel<TrainingStream>('/ws/training')
  const modelsQuery = useApiResource<EvaluationArtifact[]>('/api/models', 30_000)
  const stream = trainingSocket.message?.availability === 'available' ? trainingSocket.message.data : undefined
  const jobs = stream?.jobs ?? jobsQuery.result?.data
  const activeJob = stream?.active_job ?? jobs?.find((job) => ['starting', 'running', 'stopping', 'unverified'].includes(job.status)) ?? jobs?.[0] ?? null
  const metricsQuery = useApiResource<{ samples: MetricsSample[]; source: string | null }>(`/api/jobs/${encodeURIComponent(activeJob?.job_id ?? '')}/metrics`, 2_000, Boolean(activeJob))
  const logsQuery = useApiResource<LogPayload>(`/api/jobs/${encodeURIComponent(activeJob?.job_id ?? '')}/logs`, 2_000, Boolean(activeJob))
  const samples = stream?.metrics_samples?.length ? stream.metrics_samples : metricsQuery.result?.data?.samples ?? []
  const supervisorAvailable = jobsQuery.result?.availability === 'available' || trainingSocket.connected

  async function control(action: 'stop' | 'kill') {
    if (!activeJob) return
    if (!window.confirm(action === 'kill'
      ? `Hard-kill supervised process ${activeJob.job_id}? This is the last-resort process-group action.`
      : `Request process termination for ${activeJob.job_id}? The script does not currently implement a cooperative stop/checkpoint sentinel.`)) return
    try { await apiRequest(`/api/jobs/${encodeURIComponent(activeJob.job_id)}/${action}`, { method: 'POST' }); void jobsQuery.refresh() }
    catch (error) { window.alert(error instanceof ApiError ? error.message : 'Supervisor control request failed.') }
  }

  return <div className="page">
    <ScreenHeader title="Training" description="Single-child process supervision · metrics are emitted by the training loop." />
    <section className="process-rail">
      <div><span className="section-caption">Training supervisor · {supervisorAvailable ? 'connected' : 'unavailable'}</span><h2>{activeJob ? activeJob.status.toUpperCase() : supervisorAvailable ? 'NO SUPERVISOR-REGISTERED CHILD' : 'SUPERVISOR STATE UNAVAILABLE'}</h2><p>{activeJob ? `Run ${activeJob.job_id} · PID ${activeJob.pid ?? '—'} · started ${activeJob.started_at ?? 'time unavailable'}${activeJob.recovery_state ? ` · recovery ${activeJob.recovery_state}` : ''}` : jobsQuery.error ?? (supervisorAvailable ? 'No supervised training child is currently registered.' : 'No job snapshot has been loaded; this does not imply idle or completed.')}</p></div>
      <div className="process-measure"><span>Training step</span><b>{activeJob?.step ?? '—'} / {activeJob?.total_steps ?? '—'}</b></div>
      <div className="control-group">
        {activeJob && ['finished', 'interrupted', 'killed', 'failed'].includes(activeJob.status) && (
          <Link to={`/prepare?resume=${encodeURIComponent(activeJob.job_id)}`} className="button button-primary button-small">Resume In-Place</Link>
        )}
        <button className="button button-outline" disabled={!activeJob || activeJob.status !== 'running'} onClick={() => void control('stop')}>Request termination</button>
        <button className="button button-danger" disabled={!activeJob || !['running', 'stopping'].includes(activeJob.status)} onClick={() => void control('kill')}>Hard kill</button>
      </div>
    </section>
    <section className="telemetry-section">
      <SectionHeading title="Dreamer train-step metrics" detail="Seven exact keys · no smoothing is applied to raw child-process values" />
      <div className="metric-keys">{dreamerKeys.map((key) => <span key={key}>{key}</span>)}</div>
      <LossPlot samples={samples} />
      <SourceLine source={stream?.metrics_source ?? metricsQuery.result?.data?.source ?? (activeJob ? `logs/${activeJob.job_id}.metrics.jsonl` : 'models/dreamer_agent.py · train_step()')} detail={samples.length ? `${samples.length} captured samples` : 'No samples written by the child yet'} />
    </section>
    <div className="two-column-sections">
      <section className="data-section"><SectionHeading title="Environment & replay" detail="Runtime values only when the child emits them" />{activeJob ? <Unavailable title="Environment interaction stream not implemented" reason="The current training process exposes loss JSONL only; env ticker values are not inferred from the loss stream." source="env/dreamer_trading_env.py · step info" /> : <Unavailable title="Environment ticker unavailable" reason="No active training process is registered." source="env/dreamer_trading_env.py · step info" />}<div className="config-strip"><span>Buffer capacity · 100,000 configured</span><span>Sequence length · 64 documented default</span><span>Device · CPU-only environment</span></div></section>
      <section className="data-section"><SectionHeading title="Child process log" detail="Captured stdout / stderr · display only" />{logsQuery.result?.data?.lines?.length ? <pre className="log-tail">{logsQuery.result.data.lines.join('\n')}</pre> : <Unavailable title={activeJob ? 'No log lines received' : 'No child-process log loaded'} reason={logsQuery.error ?? 'Logs appear after a supervised child starts.'} source={logsQuery.result?.data?.source ?? 'logs/<job_id>.log'} />}</section>
    </div>
    <section className="data-section"><SectionHeading title="Evaluated artifact history" detail="Evaluation files · not active training jobs" />{modelsQuery.result?.data?.length ? <div className="data-table-wrap"><table><thead><tr><th>Artifact</th><th>Model</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Trades</th><th>Gate</th></tr></thead><tbody>{modelsQuery.result.data.map((artifact) => <tr key={artifact.run_name}><td>{artifact.run_name}</td><td>{artifact.model_type}</td><td>{artifact.metrics.sharpe?.toFixed(2) ?? '—'}</td><td>{artifact.metrics.total_return_pct?.toFixed(2) ?? '—'}%</td><td>{artifact.metrics.max_dd_pct?.toFixed(2) ?? '—'}%</td><td>{artifact.metrics.trades ?? '—'}</td><td><span className={artifact.passed ? 'gate-pass' : 'gate-fail'}>{artifact.passed ? 'PROMOTABLE' : 'NOT PROMOTABLE'}</span></td></tr>)}</tbody></table></div> : <Unavailable title="Artifact evaluations unavailable" reason={modelsQuery.error ?? 'No evaluation files were returned.'} source={modelsQuery.result?.source ?? 'artifacts/models/*/evaluation.json'} />}<SourceLine source={modelsQuery.result?.source ?? 'artifacts/models/*/evaluation.json'} /></section>
    <p className="screen-footnote">Pause is not exposed: current scripts do not implement a pause sentinel. Clean stop/checkpoint behavior is also not implemented, so termination may not save a resumable checkpoint. Resume is withheld until a compatible checkpoint is verified.</p>
  </div>
}
