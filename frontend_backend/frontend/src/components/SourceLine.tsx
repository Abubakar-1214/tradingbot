import { DatabaseOutlined } from '@ant-design/icons'

export function SourceLine({ source, detail }: { source?: string; detail?: string }) {
  return (
    <div className="source-line">
      <DatabaseOutlined aria-hidden="true" />
      <span>{source ?? 'Source not reported'}</span>
      {detail && <span className="source-detail">{detail}</span>}
    </div>
  )
}

export function Unavailable({ title, reason, source }: { title: string; reason?: string; source?: string }) {
  return (
    <div className="unavailable-state" role="status">
      <span className="unavailable-marker" aria-hidden="true">—</span>
      <div><strong>{title}</strong><p>{reason ?? 'No value has been reported by the source.'}</p>{source && <SourceLine source={source} />}</div>
    </div>
  )
}
