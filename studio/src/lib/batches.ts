// The batch monitor's client: the manager's `/api/batches` (every batch on
// the box, both vendor dialects in one shape) plus the display rules the list
// and the detail page share. Batches are SUBMITTED through the vendor APIs
// (`/v1/batches`, `/v1/messages/batches`); this surface only watches them,
// cancels, deletes and hands back their files.

/** The live block a running batch carries - what the run is doing right now. */
export interface BatchProgress {
  /** validating | running | queued | waiting_for_service | yielding |
   *  backing_off | isolating | paused | cancelling | finalizing | starting */
  state: string
  reason?: string
  port?: number
  in_flight?: number
  concurrency?: number
  /** requests settled per minute over the recent window */
  rate_per_minute?: number
  eta_seconds?: number
  retries?: number
}

export interface BatchSummary {
  id: string
  dialect: 'openai' | 'anthropic'
  endpoint: string
  model: string
  /** validating | in_progress | finalizing | completed | failed | expired |
   *  cancelling | cancelled */
  status: string
  created_ms: number
  in_progress_ms: number | null
  ended_ms: number | null
  expires_ms: number
  completion_window: string
  counts: {
    total: number
    succeeded: number
    errored: number
    canceled: number
    expired: number
    pending: number
  }
  usage: {
    input_tokens: number
    output_tokens: number
    cached_tokens: number
    reasoning_tokens: number
  }
  /** null once the batch has ended */
  progress: BatchProgress | null
  options: Record<string, unknown> | null
  metadata: Record<string, unknown> | null
  errors: unknown
  /** byte sizes; `errors` is null on an Anthropic batch (one results file) */
  files: { input: number | null; output: number; errors: number | null }
}

export interface BatchResultItem {
  /** byte offset of the line - the key `line()` reads it back by */
  offset: number
  custom_id: string
  ok: boolean
  /** an HTTP status (OpenAI) or the result type (Anthropic) */
  status: number | string | null
  preview: string | null
  error: string | null
  input_tokens: number
  output_tokens: number
}

export interface ResultsPage {
  items: BatchResultItem[]
  next_cursor: number
  /** the committed bytes ran out - a running batch adds more later */
  end: boolean
}

export type ResultKind = 'ok' | 'failed'
export type ListState = 'active' | 'ended' | 'all'

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  const text = await r.text()
  const data = text ? (JSON.parse(text) as unknown) : undefined
  if (!r.ok) {
    const msg = (data as { error?: { message?: string } } | undefined)?.error?.message
    throw new Error(msg || `HTTP ${r.status}`)
  }
  return data as T
}

const enc = encodeURIComponent

export const batchesApi = {
  list: (state: ListState, after?: string, limit = 50) => {
    const p = new URLSearchParams({ state, limit: String(limit) })
    if (after) p.set('after', after)
    return req<{ data: BatchSummary[]; has_more: boolean }>(`/api/batches?${p.toString()}`)
  },
  get: (id: string) => req<BatchSummary>(`/api/batches/${enc(id)}`),
  cancel: (id: string) => req<BatchSummary>(`/api/batches/${enc(id)}/cancel`, { method: 'POST' }),
  remove: (id: string) =>
    req<{ id: string; deleted: boolean }>(`/api/batches/${enc(id)}`, { method: 'DELETE' }),
  results: (id: string, kind: ResultKind, cursor: number, limit = 50) =>
    req<ResultsPage>(
      `/api/batches/${enc(id)}/results?kind=${kind}&cursor=${cursor}&limit=${limit}`,
    ),
  line: (id: string, kind: ResultKind, offset: number) =>
    req<unknown>(`/api/batches/${enc(id)}/results/line?kind=${kind}&offset=${offset}`),
  /** a plain link - the browser downloads it (Content-Disposition) */
  fileUrl: (id: string, which: 'input' | 'output' | 'errors') =>
    `/api/batches/${enc(id)}/files/${which}`,
}

/** Still has work to do (or is wrapping up). */
export function isActive(b: BatchSummary): boolean {
  return !['completed', 'failed', 'expired', 'cancelled'].includes(b.status)
}

/** Requests with an outcome, whatever it was. */
export function settled(b: BatchSummary): number {
  const c = b.counts
  return c.succeeded + c.errored + c.canceled + c.expired
}

/** Requests that did not succeed. */
export function failed(b: BatchSummary): number {
  const c = b.counts
  return c.errored + c.canceled + c.expired
}

/** What the not-succeeded results are, in words: "Failed", "Cancelled",
 *  "Failed / cancelled" - a cancelled request did not fail, and the tab that
 *  lists them must not say it did. */
export function unsuccessfulLabel(b: BatchSummary): string {
  const c = b.counts
  const kinds = [
    c.errored > 0 && 'failed',
    c.canceled > 0 && 'cancelled',
    c.expired > 0 && 'expired',
  ].filter((k): k is string => !!k)
  const words = kinds.length ? kinds.join(' / ') : 'failed'
  return words.charAt(0).toUpperCase() + words.slice(1)
}

export type Tone = 'run' | 'wait' | 'ok' | 'warn' | 'bad'

/** What the row says the batch is doing, in words a person reads - the live
 *  state while it runs (the run's own reason when it gives one), the outcome
 *  once it has ended. */
export function statusLabel(b: BatchSummary): { label: string; tone: Tone } {
  switch (b.status) {
    case 'completed':
      return failed(b) > 0
        ? { label: 'Completed with failures', tone: 'warn' }
        : { label: 'Completed', tone: 'ok' }
    case 'failed':
      return { label: 'Failed', tone: 'bad' }
    case 'expired':
      return { label: 'Expired', tone: 'bad' }
    case 'cancelled':
      return { label: 'Cancelled', tone: 'warn' }
    case 'cancelling':
      return { label: 'Cancelling', tone: 'wait' }
    case 'finalizing':
      return { label: 'Finishing up', tone: 'run' }
  }
  switch (b.progress?.state) {
    case 'validating':
      return { label: 'Checking the input', tone: 'run' }
    case 'queued':
      return { label: 'Queued behind another batch', tone: 'wait' }
    case 'waiting_for_service':
      return { label: `Waiting for ${b.model} to be running`, tone: 'wait' }
    case 'yielding':
      return { label: 'Paused for live traffic', tone: 'wait' }
    case 'backing_off':
      return { label: 'Retrying after errors', tone: 'wait' }
    case 'isolating':
      return { label: 'Isolating a failing request', tone: 'wait' }
    case 'paused':
      // a result write failed (a full disk); it resumes once writes succeed
      return { label: 'Paused - cannot write results', tone: 'bad' }
    case 'running':
      return { label: 'Running', tone: 'run' }
    default:
      return { label: 'Starting', tone: 'run' }
  }
}

/** A short handle for the row: a name the submitter put in metadata, else
 *  the id's tail (the prefix is the same for every batch of a dialect). */
export function batchName(b: BatchSummary): string {
  const m = b.metadata ?? {}
  for (const k of ['name', 'description', 'job', 'title']) {
    const v = m[k]
    if (typeof v === 'string' && v.trim()) return v.trim()
  }
  return b.id.length > 14 ? `...${b.id.slice(-10)}` : b.id
}

/** "/v1/chat/completions" -> "Chat completions"; the vendor's path stays in
 *  the tooltip. */
export function endpointLabel(endpoint: string): string {
  const map: Record<string, string> = {
    '/v1/chat/completions': 'Chat completions',
    '/v1/responses': 'Responses',
    '/v1/completions': 'Completions',
    '/v1/embeddings': 'Embeddings',
    '/v1/messages': 'Messages',
  }
  return map[endpoint] ?? endpoint
}

/** Requests a minute over the whole run - an ended batch's rate, from its
 *  start to its end; undefined before it started. Only requests the model
 *  answered count: a cancel or an expiry marks the rest in one sweep, and
 *  counting those would read as a burst of speed. */
export function averagePerMinute(b: BatchSummary): number | undefined {
  if (b.in_progress_ms === null) return undefined
  const ms = (b.ended_ms ?? Date.now()) - b.in_progress_ms
  const answered = b.counts.succeeded + b.counts.errored
  return ms > 0 ? (answered / ms) * 60_000 : undefined
}

/** Requests a minute, rounded the way a person reads a rate. */
export function fmtPerMinute(rate: number | undefined): string {
  if (rate === undefined || !Number.isFinite(rate) || rate <= 0) return '-'
  if (rate >= 100) return `${Math.round(rate).toLocaleString()} / min`
  if (rate >= 10) return `${rate.toFixed(0)} / min`
  return `${rate.toFixed(1)} / min`
}

/** A span between two instants, coarse: "45 s", "12 min", "3 h 20 min",
 *  "2 d 4 h" - a batch runs for minutes to days. */
export function fmtSpan(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '-'
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} s`
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min`
  const h = Math.floor(m / 60)
  if (h < 48) return `${h} h ${m % 60} min`
  return `${Math.floor(h / 24)} d ${h % 24} h`
}
