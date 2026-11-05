import { useCallback, useEffect, useState } from 'react'
import { apiRequest, ApiError } from '../api/client'
import type { ApiEnvelope } from '../domain/types'

type ResourceState<T> = { path: string; result: ApiEnvelope<T> | null; error: string | null; loading: boolean }

export function useApiResource<T>(path: string, pollMs?: number, enabled = true) {
  const [state, setState] = useState<ResourceState<T>>({ path: '', result: null, error: null, loading: true })

  const refresh = useCallback(async () => {
    if (!enabled) return
    try {
      const result = await apiRequest<T>(path)
      setState({ path, result, error: null, loading: false })
    } catch (cause) {
      setState({ path, result: null, error: cause instanceof ApiError ? cause.message : 'Unable to load this source.', loading: false })
    }
  }, [enabled, path])

  useEffect(() => {
    if (!enabled) return
    // This is the external-system synchronization point: refresh updates from the API response.
    // oxlint-disable-next-line react/set-state-in-effect
    void refresh()
    if (!pollMs) return
    const timer = window.setInterval(() => void refresh(), pollMs)
    return () => window.clearInterval(timer)
  }, [refresh, pollMs, enabled])

  const stateMatchesPath = state.path === path
  return {
    result: enabled && stateMatchesPath ? state.result : null,
    error: enabled && stateMatchesPath ? state.error : null,
    loading: enabled && (!stateMatchesPath || state.loading),
    refresh,
  }
}
