import type { ApiEnvelope } from '../domain/types'

export class ApiError extends Error {
  readonly status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<ApiEnvelope<T>> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError('Control plane is not reachable.')
  }

  const payload = (await response.json().catch(() => null)) as ApiEnvelope<T> | { detail?: string } | null
  if (!response.ok) {
    const detail = payload && 'detail' in payload ? payload.detail : undefined
    throw new ApiError(detail ?? `Request failed (${response.status}).`, response.status)
  }
  if (!payload || !('availability' in payload)) {
    throw new ApiError('The control plane returned an invalid response.', response.status)
  }
  return payload as ApiEnvelope<T>
}
