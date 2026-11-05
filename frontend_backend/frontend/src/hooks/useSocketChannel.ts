import { useEffect, useState } from 'react'
import type { ApiEnvelope } from '../domain/types'

export function useSocketChannel<T>(path: string) {
  const [message, setMessage] = useState<ApiEnvelope<T> | null>(null)
  const [connected, setConnected] = useState(false)
  useEffect(() => {
    let socket: WebSocket | null = null
    let stopped = false
    let retry = 0
    let timer = 0
    const connect = () => {
      if (stopped) return
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}${path}`)
      socket.onopen = () => { retry = 0; setConnected(true) }
      socket.onmessage = (event) => {
        try { setMessage(JSON.parse(event.data) as ApiEnvelope<T>) } catch { /* Ignore malformed, non-contract messages. */ }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!stopped) { retry += 1; timer = window.setTimeout(connect, Math.min(30_000, 500 * 2 ** Math.min(retry, 6))) }
      }
      socket.onerror = () => socket?.close()
    }
    connect()
    return () => { stopped = true; window.clearTimeout(timer); socket?.close() }
  }, [path])
  return { message, connected }
}
