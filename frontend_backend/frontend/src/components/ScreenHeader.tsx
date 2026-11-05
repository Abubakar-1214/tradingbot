import type { ReactNode } from 'react'

export function ScreenHeader({ title, description, side }: { title: string; description: string; side?: ReactNode }) {
  return (
    <div className="screen-heading">
      <div><h1>{title}</h1><p>{description}</p></div>
      {side && <div className="screen-heading-side">{side}</div>}
    </div>
  )
}

export function SectionHeading({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div>
}
