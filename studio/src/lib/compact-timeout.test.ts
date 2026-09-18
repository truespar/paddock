// Tests for the compaction DEADLINE (lib/compact.ts).
//
// Background compaction takes a per-conversation slot in `inflight` before it
// POSTs and gives it back only when the request has settled. Without a
// deadline of its own, a runner that accepts the POST and then stops writing
// (a wedged decode, a half-open socket) keeps that slot for the lifetime of
// the tab: every later turn sees the conversation as already compacting and
// the thread silently stops rolling its summary forward.
//
// So the contract exercised here is: the call is BOUNDED, the bound covers
// response headers AND the response body, a call that gives up leaves the
// summary and the raw history exactly as they were, and the slot comes back so
// a later turn retries. Nothing here sleeps - the deadline is driven with
// Vitest's fake timers - and no request leaves the process: `fetch` and the
// models store (the endpoint lookup) are the only mocked boundaries.
//
// DEADLINE_MS below mirrors COMPACTION_TIMEOUT_MS in compact.ts. That number
// is a proposed retry bound, not a measured optimum; maintainers should
// review it against representative compaction latencies.

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Conversation, Message } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import { compactionTarget } from './tokens'

const models = vi.hoisted(() => ({
  url: '/api/runners/11540/v1/responses' as string | undefined,
}))
vi.mock('@/stores/models', () => ({
  useModelsStore: () => ({ responsesUrl: () => models.url }),
}))

import { maybeCompact } from './compact'

/** The whole-call budget compact.ts is expected to enforce. */
const DEADLINE_MS = 5 * 60_000
const MAX_CTX = 4096
const MAX_REPLY = 512

// ── fixtures ────────────────────────────────────────────────────────────────

let seq = 0

/** A turn long enough that eight of them cross the compaction threshold. */
function filler(label: string): string {
  return `${label}: `.padEnd(1200, 'context filler that costs tokens ')
}

function turns(id: string, n: number, text: (i: number) => string): Message[] {
  const out: Message[] = []
  let parentId: string | null = null
  for (let i = 0; i < n; i++) {
    const role = i % 2 === 0 ? 'user' : 'assistant'
    const m: Message = {
      id: `${id}-m${i}-${++seq}`,
      parentId,
      role,
      content: [{ type: 'text', text: text(i) }],
      createdAt: i,
      ...(role === 'assistant' ? { model: 'qwen3.8-30b' } : {}),
    }
    out.push(m)
    parentId = m.id
  }
  return out
}

function convOf(id: string, messages: Message[]): Conversation {
  return {
    id,
    title: 't',
    messages,
    leafId: messages[messages.length - 1]?.id,
    model: 'qwen3.8-30b',
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
    createdAt: 0,
    updatedAt: 0,
  }
}

/** A conversation over the threshold: eight turns, six of them compactable. */
function bigConv(id: string): Conversation {
  return convOf(
    id,
    turns(id, 8, (i) => filler(`turn ${i}`)),
  )
}

/** Same thread with a summary that already covers its first two turns. */
function withPriorSummary(c: Conversation): Conversation {
  c.summary = 'previously: the user asked about kernels'
  c.summaryCount = 2
  c.summaryLastId = c.messages[1].id
  c.summaryModel = 'qwen3.8-30b'
  return c
}

/** A thread far under the threshold - nothing to compact. */
function smallConv(id: string): Conversation {
  return convOf(
    id,
    turns(id, 4, (i) => `short turn ${i}`),
  )
}

// ── plumbing ────────────────────────────────────────────────────────────────

type FakeRes = { ok: boolean; status: number; json: () => Promise<unknown> }

type Deferred<T> = { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void }
function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

/** A Responses body carrying `summary` as the model would return it. */
function okRes(summary: string, body?: Promise<unknown>): FakeRes {
  return {
    ok: true,
    status: 200,
    json: () =>
      body ??
      Promise.resolve({
        output: [{ type: 'message', content: [{ type: 'output_text', text: summary }] }],
      }),
  }
}

/** A promise that never settles - the stall this whole file is about. */
function never<T>(): Promise<T> {
  return new Promise<T>(() => {})
}

/** Resolve `value` after `ms` of FAKE time. */
function after<T>(ms: number, value: T): Promise<T> {
  return new Promise<T>((resolve) => {
    setTimeout(() => resolve(value), ms)
  })
}

/** Whether a promise has settled yet, readable synchronously. */
function track(p: Promise<unknown>): { done: boolean } {
  const s = { done: false }
  void p.then(() => {
    s.done = true
  })
  return s
}

/** Hand the clock back and let one REAL macrotask run: an unhandled rejection
 *  is reported by the runtime on a real tick, so a faked clock cannot see one.
 *  Called only once a test has finished advancing fake time. */
async function flushReal(): Promise<void> {
  vi.useRealTimers()
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 0)
  })
}

const fetchMock = vi.fn()
const persist = vi.fn()
const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

function lastInit(): RequestInit {
  const calls = fetchMock.mock.calls
  return calls[calls.length - 1][1] as RequestInit
}

function lastRequest(): { input: string; model: string } {
  return JSON.parse(lastInit().body as string)
}

function lastSignal(): AbortSignal {
  return lastInit().signal as AbortSignal
}

/** A rejection nobody handled is how "the losing race branch was left
 *  dangling" shows up, so the tests watch for one. Reached through globalThis
 *  because this tree's tsconfig types the BROWSER: `process` is the test
 *  runner's, not the app's. */
type RejectionWatcher = {
  on: (event: 'unhandledRejection', fn: (reason: unknown) => void) => void
  off: (event: 'unhandledRejection', fn: (reason: unknown) => void) => void
}
const runner = (globalThis as unknown as { process?: RejectionWatcher }).process

const unhandled: unknown[] = []
const onUnhandled = (e: unknown) => unhandled.push(e)
beforeAll(() => {
  runner?.on('unhandledRejection', onUnhandled)
})
afterAll(() => {
  runner?.off('unhandledRejection', onUnhandled)
})

beforeEach(() => {
  // Only the clock the deadline uses, so `flushReal` can still reach a real
  // macrotask and let an unhandled rejection surface.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  models.url = '/api/runners/11540/v1/responses'
  fetchMock.mockReset()
  persist.mockReset()
  warn.mockClear()
  vi.stubGlobal('fetch', fetchMock)
  unhandled.length = 0
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

afterAll(() => {
  warn.mockRestore()
})

// ── the fixture itself (control) ────────────────────────────────────────────

describe('the fixtures these tests rely on', () => {
  // Without this the `unhandled` assertions below would pass vacuously: the
  // watcher is attached through globalThis, and a miss there reads as "no
  // rejection leaked" rather than as "nobody was looking". (Verified once with
  // a deliberate rejection: the watcher records it, and vitest's own reporter
  // did not, so this really is the detector.)
  it('has a rejection watcher attached', () => {
    expect(runner).toBeTruthy()
  })

  it('crosses the compaction threshold, and the small one does not', () => {
    expect(compactionTarget(bigConv('ctl-big'), MAX_CTX, MAX_REPLY)).toBe(6)
    expect(compactionTarget(smallConv('ctl-small'), MAX_CTX, MAX_REPLY)).toBe(0)
  })

  it('sends nothing at all for a thread under the threshold', async () => {
    await maybeCompact(smallConv('ctl-none'), MAX_CTX, MAX_REPLY, persist)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(persist).not.toHaveBeenCalled()
    // and no deadline was armed for a request that was never made
    expect(vi.getTimerCount()).toBe(0)
  })
})

// ── the deadline ───────────────────────────────────────────────────────────

describe('maybeCompact deadline', () => {
  it('gives up on a request whose response headers never arrive', async () => {
    const conv = bigConv('stall-headers')
    fetchMock.mockImplementation(() => never<FakeRes>())

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // it waits the whole budget before abandoning a call...
    await vi.advanceTimersByTimeAsync(DEADLINE_MS - 1)
    expect(call.done).toBe(false)
    // ...and then stops waiting
    await vi.advanceTimersByTimeAsync(1)
    expect(call.done).toBe(true)

    expect(persist).not.toHaveBeenCalled()
    expect(conv.summary).toBeUndefined()
    expect(conv.summaryCount).toBeUndefined()
    // the request underneath is cancelled, not merely abandoned
    expect(lastSignal().aborted).toBe(true)
    // reported, never silent (rules of the tree)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0][1])).toMatch(/timed out/i)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('gives up on a response whose body never arrives, on the same budget as its headers', async () => {
    const conv = bigConv('stall-body')
    // headers land four minutes in; the body then stalls forever
    fetchMock.mockImplementation(() => after(240_000, okRes('never read', never<unknown>())))

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))

    await vi.advanceTimersByTimeAsync(240_000)
    expect(call.done).toBe(false)
    // the remaining minute of the ONE budget, not a fresh one per phase
    await vi.advanceTimersByTimeAsync(DEADLINE_MS - 240_000 - 1)
    expect(call.done).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    expect(call.done).toBe(true)

    expect(persist).not.toHaveBeenCalled()
    expect(conv.summary).toBeUndefined()
    expect(lastSignal().aborted).toBe(true)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('lets a slow but healthy call finish, and does not cancel it', async () => {
    const conv = bigConv('slow-ok')
    // one minute to headers, two more to the body: slow, but inside the budget
    fetchMock.mockImplementation(() =>
      after(60_000, okRes('brief', after(120_000, {
        output: [{ type: 'message', content: [{ type: 'output_text', text: 'a slow brief' }] }],
      }))),
    )

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await vi.advanceTimersByTimeAsync(180_000)

    expect(call.done).toBe(true)
    expect(conv.summary).toBe('a slow brief')
    expect(conv.summaryCount).toBe(6)
    expect(conv.summaryLastId).toBe(conv.messages[5].id)
    expect(conv.summaryModel).toBe('qwen3.8-30b')
    expect(persist).toHaveBeenCalledTimes(1)
    expect(persist).toHaveBeenCalledWith(conv)
    // nothing was cancelled: the deadline is a bound, not a policy of aborting
    expect(lastSignal().aborted).toBe(false)
    expect(warn).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('arms exactly one deadline per attempt and clears it when the call succeeds', async () => {
    const conv = bigConv('timer-clear')
    const headers = deferred<FakeRes>()
    fetchMock.mockImplementation(() => headers.promise)

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await Promise.resolve()
    expect(vi.getTimerCount()).toBe(1)

    headers.resolve(okRes('a brief'))
    await vi.advanceTimersByTimeAsync(0)

    expect(call.done).toBe(true)
    expect(conv.summary).toBe('a brief')
    expect(vi.getTimerCount()).toBe(0)
  })
})

// ── ordinary failures keep behaving as they did ────────────────────────────

describe('maybeCompact ordinary failures', () => {
  it('leaves the existing summary and the raw history alone on an HTTP error', async () => {
    const conv = withPriorSummary(bigConv('http-fail'))
    const before = conv.messages.map((m) => m.id)
    fetchMock.mockImplementation(() => Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({}) }))

    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)

    expect(persist).not.toHaveBeenCalled()
    expect(conv.summary).toBe('previously: the user asked about kernels')
    expect(conv.summaryCount).toBe(2)
    expect(conv.summaryLastId).toBe(conv.messages[1].id)
    expect(conv.messages.map((m) => m.id)).toEqual(before)
    expect(String(warn.mock.calls[0][1])).toMatch(/HTTP 503/)
    expect(vi.getTimerCount()).toBe(0)

    // and the slot is free, so the next turn tries again
    fetchMock.mockImplementation(() => Promise.resolve(okRes('second try')))
    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(conv.summary).toBe('second try')
  })

  it('leaves the existing summary alone when the request rejects', async () => {
    const conv = withPriorSummary(bigConv('net-fail'))
    fetchMock.mockImplementation(() => Promise.reject(new Error('network down')))

    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)

    expect(persist).not.toHaveBeenCalled()
    expect(conv.summary).toBe('previously: the user asked about kernels')
    expect(String(warn.mock.calls[0][1])).toMatch(/network down/)
    expect(vi.getTimerCount()).toBe(0)
  })
})

// ── the slot ───────────────────────────────────────────────────────────────

describe('maybeCompact inflight slot', () => {
  it('still runs one compaction at a time per conversation', async () => {
    const conv = bigConv('single-flight')
    fetchMock.mockImplementation(() => never<FakeRes>())

    const first = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(first.done).toBe(true)
  })

  it('does not hold up another conversation while one is stalled', async () => {
    const stalled = bigConv('other-stalled')
    const healthy = bigConv('other-healthy')
    fetchMock
      .mockImplementationOnce(() => never<FakeRes>())
      .mockImplementationOnce(() => Promise.resolve(okRes('healthy brief')))

    const first = track(maybeCompact(stalled, MAX_CTX, MAX_REPLY, persist))
    await maybeCompact(healthy, MAX_CTX, MAX_REPLY, persist)

    expect(first.done).toBe(false)
    expect(healthy.summary).toBe('healthy brief')
    expect(stalled.summary).toBeUndefined()
    expect(persist).toHaveBeenCalledTimes(1)
    expect(persist).toHaveBeenCalledWith(healthy)

    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(first.done).toBe(true)
    expect(stalled.summary).toBeUndefined()
  })

  it('releases the slot so the same conversation compacts on a later turn', async () => {
    const conv = bigConv('retry')
    fetchMock.mockImplementation(() => never<FakeRes>())

    const first = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(first.done).toBe(true)

    fetchMock.mockImplementation(() => Promise.resolve(okRes('brief at last')))
    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(conv.summary).toBe('brief at last')
    expect(conv.summaryCount).toBe(6)
    expect(persist).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })
})

// ── what a timed-out attempt must not touch ───────────────────────────────

describe('maybeCompact after a deadline', () => {
  it('keeps the previous summary and rolls it forward on the retry', async () => {
    const conv = withPriorSummary(bigConv('prior-summary'))
    const before = conv.messages.map((m) => ({ id: m.id, text: m.content[0] }))
    fetchMock.mockImplementation(() => never<FakeRes>())

    const first = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(first.done).toBe(true)

    // the timed-out attempt changed nothing
    expect(conv.summary).toBe('previously: the user asked about kernels')
    expect(conv.summaryCount).toBe(2)
    expect(conv.summaryLastId).toBe(conv.messages[1].id)
    expect(conv.summaryModel).toBe('qwen3.8-30b')
    expect(conv.messages.map((m) => ({ id: m.id, text: m.content[0] }))).toEqual(before)
    expect(persist).not.toHaveBeenCalled()

    // and the retry still folds the old summary in rather than re-reading
    fetchMock.mockImplementation(() => Promise.resolve(okRes('rolled forward')))
    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)
    expect(lastRequest().input).toContain('Summary of the conversation so far:')
    expect(lastRequest().input).toContain('previously: the user asked about kernels')
    expect(conv.summary).toBe('rolled forward')
    expect(conv.summaryCount).toBe(6)
  })

  it('ignores a response that lands after the deadline', async () => {
    const conv = bigConv('late-resolve')
    fetchMock.mockImplementation(() => after(DEADLINE_MS + 1000, okRes('too late')))

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(call.done).toBe(true)

    // the abandoned request answers a second later: nobody is listening
    await vi.advanceTimersByTimeAsync(2000)
    expect(conv.summary).toBeUndefined()
    expect(conv.summaryCount).toBeUndefined()
    expect(persist).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)

    // a later turn is unaffected by the straggler
    fetchMock.mockImplementation(() => Promise.resolve(okRes('in time')))
    await maybeCompact(conv, MAX_CTX, MAX_REPLY, persist)
    expect(conv.summary).toBe('in time')
    expect(persist).toHaveBeenCalledTimes(1)

    await flushReal()
    expect(unhandled).toEqual([])
  })

  it('swallows nothing and leaks nothing when the abandoned request rejects late', async () => {
    const conv = bigConv('late-reject')
    fetchMock.mockImplementation(
      () =>
        new Promise<FakeRes>((_, reject) => {
          setTimeout(() => reject(new Error('aborted after the fact')), DEADLINE_MS + 1000)
        }),
    )

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(call.done).toBe(true)
    expect(warn).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(2000)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(persist).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
    // the late failure is handled by the race branch that lost, not by the
    // runtime as an unhandled rejection
    await flushReal()
    expect(unhandled).toEqual([])
  })

  it('gives up on a body that stalls after a late-but-inside-budget header', async () => {
    const conv = bigConv('late-body-stall')
    const body = deferred<unknown>()
    fetchMock.mockImplementation(() => after(1000, okRes('unused', body.promise)))

    const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
    await vi.advanceTimersByTimeAsync(DEADLINE_MS)
    expect(call.done).toBe(true)
    expect(conv.summary).toBeUndefined()

    // the body arrives after we stopped waiting: it must not persist
    body.resolve({
      output: [{ type: 'message', content: [{ type: 'output_text', text: 'too late' }] }],
    })
    await vi.advanceTimersByTimeAsync(0)
    await flushReal()
    expect(conv.summary).toBeUndefined()
    expect(persist).not.toHaveBeenCalled()
    expect(unhandled).toEqual([])
  })
})


it('reports the deadline reason when the transport rejects synchronously on abort', async () => {
  const conv = bigConv('abort-aware-transport')
  fetchMock.mockImplementation((_url: string, init: RequestInit) =>
    new Promise<FakeRes>((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => {
        reject(new DOMException('Request aborted', 'AbortError'))
      }, { once: true })
    }),
  )
  const call = track(maybeCompact(conv, MAX_CTX, MAX_REPLY, persist))
  await vi.advanceTimersByTimeAsync(DEADLINE_MS)
  expect(call.done).toBe(true)
  expect(lastSignal().aborted).toBe(true)
  expect(String(warn.mock.calls[0][1])).toMatch(/timed out/i)
  expect(persist).not.toHaveBeenCalled()
  expect(vi.getTimerCount()).toBe(0)
})
