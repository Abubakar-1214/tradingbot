import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { InfoCircleOutlined } from '@ant-design/icons'
import { Dialog, ensureUIHost } from '@luxalgo/vela/ui'
import { registerIcon, registerWidgetAction, registerWidgetAttachment } from '@luxalgo/vela/plugin'

const actionId = 'xauusd.source-details'
const dialogByHost = new WeakMap<HTMLElement, Dialog>()
const theme = { background: '#0A0D03', textColor: '#F3F5E8', gridColor: '#262D05', borderColor: '#3A4120', upColor: '#DEFF00', downColor: '#FF0001', fontFamily: 'Saira, sans-serif' }

const infoMarkup = renderToStaticMarkup(createElement(InfoCircleOutlined))
const infoStart = infoMarkup.indexOf('<svg')
const infoEnd = infoMarkup.indexOf('</svg>')
const infoSvg = infoStart >= 0 && infoEnd >= infoStart ? infoMarkup.slice(infoStart, infoEnd + 6) : null
if (infoSvg) registerIcon('xauusd-source-details', infoSvg)

registerWidgetAttachment({
  id: 'xauusd.source-details.dialog',
  mount: (ctx) => {
    ensureUIHost(ctx.host, theme)
    const dialog = new Dialog({
      host: ctx.host,
      title: 'Data sources & safety boundary',
      contained: true,
      closeOnEscape: true,
      closeOnInteractOutside: true,
      dimBackdrop: true,
      content: (body) => {
        const entries = [
          ['Chart bars', 'GET /api/market/bars → data/xauusd_h1.csv (H1 seed); other intervals use their matching repository CSV. Historical only.'],
          ['Provider', 'RepositoryCsvProvider reads local FastAPI endpoints. No remote market provider or live subscription.'],
          ['Workspace', 'VelaWorkspace native OHLC renderer, indicator picker, drawing toolbar, object tree, data window, replay and persisted chart preferences.'],
          ['Execution', 'All mutations go to FastAPI supervisor. The frontend has no broker client and never calls broker/order functions.'],
          ['Attribution', 'Vela™ is licensed under Apache-2.0 with the project’s visible attribution requirement.'],
        ]
        const list = document.createElement('dl')
        list.className = 'vela-source-list'
        for (const [label, value] of entries) {
          const term = document.createElement('dt'); term.textContent = label
          const description = document.createElement('dd'); description.textContent = value
          list.append(term, description)
        }
        body.appendChild(list)
      },
    })
    dialogByHost.set(ctx.host, dialog)
    return () => { dialog.destroy(); dialogByHost.delete(ctx.host) }
  },
})

registerWidgetAction({
  id: actionId,
  target: 'topbar',
  label: 'Data sources',
  icon: 'xauusd-source-details',
  order: 90,
  run: (ctx) => dialogByHost.get(ctx.host)?.show(),
})
