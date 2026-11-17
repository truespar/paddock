import { describe, expect, it } from 'vitest'
import {
  averagePerMinute,
  batchName,
  endpointLabel,
  failed,
  fmtPerMinute,
  fmtSpan,
  isActive,
  settled,
  statusLabel,
  unsuccessfulLabel,
  type BatchSummary,
} from './batches'

function batch(over: Partial<BatchSummary> = {}): BatchSummary {
  return {
    id: 'batch_0123456789abcdef0123456789abcdef',
    dialect: 'openai',
    endpoint: '/v1/chat/completions',
    model: 'qwen3.8-27b',
    status: 'in_progress',
    created_ms: 0,
    in_progress_ms: 0,
    ended_ms: null,
    expires_ms: 0,
    completion_window: '24h',
    counts: { total: 10, succeeded: 6, errored: 1, canceled: 0, expired: 0, pending: 3 },
    usage: { input_tokens: 0, output_tokens: 0, cached_tokens: 0, reasoning_tokens: 0 },
    progress: { state: 'running', in_flight: 2, concurrency: 4 },
    options: null,
    metadata: null,
    errors: null,
    files: { input: 100, output: 50, errors: 10 },
    ...over,
  }
}

describe('the batch monitor display rules', () => {
  it('count what settled and what failed', () => {
    const b = batch({ counts: { total: 10, succeeded: 5, errored: 1, canceled: 2, expired: 1, pending: 1 } })
    expect(settled(b)).toBe(9)
    expect(failed(b)).toBe(4)
    expect(isActive(b)).toBe(true)
    expect(isActive(batch({ status: 'cancelled' }))).toBe(false)
  })

  it('say what a running batch is doing, then how it ended', () => {
    expect(statusLabel(batch())).toEqual({ label: 'Running', tone: 'run' })
    expect(statusLabel(batch({ progress: { state: 'yielding' } })).label).toBe(
      'Paused for live traffic',
    )
    expect(statusLabel(batch({ progress: { state: 'waiting_for_service' } })).label).toBe(
      'Waiting for qwen3.8-27b to be running',
    )
    expect(statusLabel(batch({ status: 'completed', progress: null })).tone).toBe('warn')
    expect(
      statusLabel(
        batch({
          status: 'completed',
          progress: null,
          counts: { total: 2, succeeded: 2, errored: 0, canceled: 0, expired: 0, pending: 0 },
        }),
      ),
    ).toEqual({ label: 'Completed', tone: 'ok' })
    expect(statusLabel(batch({ status: 'failed', progress: null })).tone).toBe('bad')
    expect(statusLabel(batch({ progress: { state: 'paused', reason: 'disk full' } })).tone).toBe('bad')
  })

  it('name what did not succeed without calling a cancel a failure', () => {
    const counts = { total: 120, succeeded: 36, errored: 0, canceled: 84, expired: 0, pending: 0 }
    expect(unsuccessfulLabel(batch({ counts }))).toBe('Cancelled')
    expect(unsuccessfulLabel(batch({ counts: { ...counts, errored: 1 } }))).toBe('Failed / cancelled')
    expect(unsuccessfulLabel(batch())).toBe('Failed')
  })

  it('name a batch by its metadata, else by the tail of its id', () => {
    expect(batchName(batch({ metadata: { description: 'annual reports' } }))).toBe('annual reports')
    expect(batchName(batch())).toBe('...6789abcdef')
    expect(endpointLabel('/v1/messages')).toBe('Messages')
  })

  it('average only what the model answered', () => {
    const b = batch({
      status: 'cancelled',
      in_progress_ms: 0,
      ended_ms: 60_000,
      counts: { total: 120, succeeded: 36, errored: 0, canceled: 84, expired: 0, pending: 0 },
    })
    expect(averagePerMinute(b)).toBe(36)
  })

  it('write rates and spans the way a person reads them', () => {
    expect(fmtPerMinute(undefined)).toBe('-')
    expect(fmtPerMinute(7.25)).toBe('7.3 / min')
    expect(fmtPerMinute(44.8)).toBe('45 / min')
    expect(fmtSpan(45_000)).toBe('45 s')
    expect(fmtSpan(12 * 60_000)).toBe('12 min')
    expect(fmtSpan((3 * 60 + 20) * 60_000)).toBe('3 h 20 min')
    expect(fmtSpan(52 * 3_600_000)).toBe('2 d 4 h')
  })
})
