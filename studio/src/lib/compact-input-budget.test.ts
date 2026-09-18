// PD-01: what compaction does when the window leaves NO room for transcript.
//
// `maybeCompact` sizes its input as `(maxCtx - 640 - 2048) * 4` characters.
// On a small window that number is zero or negative, and the clamp that was
// meant to make it safe (`Math.max(0, ...)`) turns it into `.slice(-0)` -
// which keeps the WHOLE string. So the one case where nothing fits is the
// case that sends the most: a full transcript into a window that cannot hold
// it, with 640 output tokens asked on top.
//
// The compaction TARGET is positive there (`compactionTarget` only needs
// `maxCtx - maxReply - 1024 > 0`), so this is not a case the caller filters
// out - see the reachability block below, which builds both windows out of
// the real stores rather than asserting the arithmetic.
//
// Everything here uses the real modules (tokens, tree, the models store) and
// mocks one boundary: `fetch`. The manager's fleet endpoints answer from a
// fixture so the store resolves a real window and a real responses URL; no
// inference runs.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { Conversation, Message } from '@/types/chat'
import { DEFAULT_PARAMS, messageText } from '@/types/chat'
import { activeMessages } from '@/lib/tree'
import { compactionTarget, replyReserve, summaryValid } from '@/lib/tokens'
import { useModelsStore } from '@/stores/models'
import { maybeCompact } from './compact'

// ── the boundary: the manager's HTTP surface ────────────────────────────────

/** What the mocked manager serves this test. */
interface Fleet {
  runners: unknown[]
  cloud: unknown[]
  /** the relayed runner card (/api/runners/{port}/server). */
  card: Record<string, unknown>
  /** what a summary request answers with. */
  summary: string
  /** status a summary request comes back with (non-2xx = the ordinary
   *  failure path, which is not this fix's path). */
  summaryStatus: number
}

function makeFetchMock() {
  return vi.fn(async (input: unknown, init?: RequestInit) => route(input, init))
}

function makeWarnSpy() {
  return vi.spyOn(console, 'warn').mockImplementation(() => {})
}

let fleet: Fleet
let fetchMock: ReturnType<typeof makeFetchMock>
let warn: ReturnType<typeof makeWarnSpy>

const PORT = 8081
const LOCAL_MODEL = 'qwen3.8-flash-next-Q4_K_M'
const CLOUD_MODEL = 'cloud:ep1:tinyco/tiny-2k'

function res(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response
}

function summaryBody(text: string): unknown {
  return { output: [{ type: 'message', content: [{ type: 'output_text', text }] }] }
}

function route(input: unknown, init?: RequestInit): Response {
  const url = String(input)
  if (url === '/api/server') return res({ version: '0.1.7', build: '0.1.7' })
  if (url === '/api/runners') return res(fleet.runners)
  if (url === '/api/cloud') return res(fleet.cloud)
  if (/^\/api\/runners\/\d+\/server$/.test(url)) return res(fleet.card)
  if (url.endsWith('/v1/responses')) {
    if (init?.method !== 'POST') throw new Error(`summary request was not a POST: ${init?.method}`)
    return res(summaryBody(fleet.summary), fleet.summaryStatus)
  }
  throw new Error(`unexpected fetch: ${url}`)
}

/** Only the summary requests - the store's own fleet polling is noise here. */
function sends(): { url: string; body: Record<string, unknown> }[] {
  return fetchMock.mock.calls
    .filter(([u]) => String(u).endsWith('/v1/responses'))
    .map(([u, init]) => ({
      url: String(u),
      body: JSON.parse(String(init?.body)) as Record<string, unknown>,
    }))
}

/** localStorage for the node test env - the models store reads the remembered
 *  model out of it at creation. */
function memoryStorage(): Storage {
  const cells = new Map<string, string>()
  return {
    get length() {
      return cells.size
    },
    clear: () => cells.clear(),
    getItem: (k: string) => cells.get(k) ?? null,
    key: (i: number) => [...cells.keys()][i] ?? null,
    removeItem: (k: string) => {
      cells.delete(k)
    },
    setItem: (k: string, v: string) => {
      cells.set(k, String(v))
    },
  }
}

/** Let the store's fire-and-forget follow-ups (doRefresh -> fetchLimits) land
 *  before a test reads the call log. */
function flush(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0))
}

// ── the fleets a real install can present ───────────────────────────────────

/** A cloud endpoint serving one model whose provider-reported context is
 *  `ctx`. This is the lane the client-side compactor actually runs on:
 *  `useChatStream` arms the runner's own `context_management` for local
 *  single-model sends and only falls back to `maybeCompact` when it cannot
 *  (a cloud lane, or a `continue` send). */
async function cloudFleet(ctx: number) {
  fleet.runners = []
  fleet.cloud = [
    {
      id: 'ep1',
      name: 'OpenRouter',
      kind: 'openai-compat',
      baseUrl: 'https://openrouter.ai/api/v1',
      hasKey: true,
      createdAt: 0,
      models: [{ id: 'tinyco/tiny-2k', display: 'Tiny 2K', ctx }],
    },
  ]
  const models = useModelsStore()
  await models.refresh()
  await models.fetchLimits()
  await flush()
  return models
}

/** A local runner started with `max_ctx = ctx` (paddock.toml `max_ctx` /
 *  `--max-ctx`, or the Start page's "Context per conversation" custom field,
 *  whose NumberField is `:min="1024" :step="1024"`). */
async function localFleet(ctx: number) {
  fleet.runners = [{ port: PORT, model: LOCAL_MODEL, status: 'ok' }]
  fleet.card = { model: LOCAL_MODEL, max_ctx: ctx }
  const models = useModelsStore()
  await models.refresh()
  await models.fetchLimits()
  await flush()
  return models
}

// ── conversations ───────────────────────────────────────────────────────────

let seq = 0

function msg(
  role: 'user' | 'assistant',
  text: string,
  parentId: string | null,
  model?: string,
): Message {
  const id = `m${++seq}`
  return {
    id,
    role,
    parentId,
    content: [{ type: 'text', text }],
    createdAt: seq,
    ...(role === 'assistant' ? { model } : {}),
  }
}

/** A straight-line chat: `turns` user/assistant pairs, each message `chars`
 *  of body text. Linked parent-to-child, so `activeMessages` walks all of it. */
function chat(id: string, model: string, turns: number, chars: number): Conversation {
  const messages: Message[] = []
  let parent: string | null = null
  for (let i = 0; i < turns; i++) {
    const u = msg('user', `u${i} ${'x'.repeat(chars)}`, parent)
    messages.push(u)
    parent = u.id
    const a = msg('assistant', `a${i} ${'y'.repeat(chars)}`, parent, model)
    messages.push(a)
    parent = a.id
  }
  return {
    id,
    title: 'PD-01',
    messages,
    leafId: parent ?? undefined,
    model,
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
    createdAt: 0,
    updatedAt: 0,
  }
}

/** Everything compaction is allowed to write, as one comparable value. */
function summaryState(c: Conversation) {
  return {
    summary: c.summary,
    summaryCount: c.summaryCount,
    summaryLastId: c.summaryLastId,
    summaryModel: c.summaryModel,
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  fleet = {
    runners: [],
    cloud: [],
    card: {},
    summary: 'the conversation so far, briefly',
    summaryStatus: 200,
  }
  fetchMock = makeFetchMock()
  vi.stubGlobal('localStorage', memoryStorage())
  vi.stubGlobal('fetch', fetchMock)
  warn = makeWarnSpy()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

// ── reachability ────────────────────────────────────────────────────────────

describe('a supported install can hand maybeCompact a window with no input allowance', () => {
  it('resolves 2048 tokens and a positive compaction target on a cloud lane', async () => {
    const models = await cloudFleet(2048)
    // the window the caller passes: useChatStream sends `models.maxCtx`.
    expect(models.maxCtx).toBe(2048)
    expect(models.responsesUrl(CLOUD_MODEL)).toBe('/api/cloud/ep1/v1/responses')

    // the reply reserve the caller passes: `replyReserve(settings.maxTokens)`.
    // 512 is the first stop the Settings slider offers, and on a 2048 window
    // it is one of only two numbered stops (512, 1024) below "Model maximum".
    const maxReply = replyReserve(512)
    expect(maxReply).toBe(512)

    const conv = chat('reach-cloud', CLOUD_MODEL, 3, 400)
    expect(activeMessages(conv)).toHaveLength(6)
    // compaction is asked for...
    expect(compactionTarget(conv, models.maxCtx, maxReply)).toBeGreaterThan(0)
    // ...on a window that leaves nothing to send: 2048 - 640 - 2048 < 0.
    expect(models.maxCtx).toBeLessThan(2688)
  })

  it('resolves the same window from a local runner card', async () => {
    const models = await localFleet(2048)
    expect(models.maxCtx).toBe(2048)
    expect(models.responsesUrl(LOCAL_MODEL)).toBe(`/api/runners/${PORT}/v1/responses`)

    const conv = chat('reach-local', LOCAL_MODEL, 3, 400)
    expect(compactionTarget(conv, models.maxCtx, replyReserve(512))).toBeGreaterThan(0)
  })
})

// ── below the floor ─────────────────────────────────────────────────────────

describe('a window below the summary floor', () => {
  it('sends no summary request at all', async () => {
    const models = await cloudFleet(2048)
    const conv = chat('below-send', CLOUD_MODEL, 3, 400)
    const persist = vi.fn()

    await maybeCompact(conv, models.maxCtx, replyReserve(512), persist)

    expect(sends()).toEqual([])
    expect(persist).not.toHaveBeenCalled()
  })

  it('says why, once, naming the window', async () => {
    const models = await cloudFleet(2048)
    const conv = chat('below-warn', CLOUD_MODEL, 3, 400)

    await maybeCompact(conv, models.maxCtx, replyReserve(512), vi.fn())

    expect(warn).toHaveBeenCalledTimes(1)
    const said = warn.mock.calls[0].map(String).join(' ')
    expect(said).toMatch(/compaction/i)
    expect(said).toContain('2048')
    expect(said).toContain('2688')
  })

  it('leaves the transcript and an existing summary exactly as they were', async () => {
    const models = await cloudFleet(2048)
    const conv = chat('below-keep', CLOUD_MODEL, 3, 400)
    const msgs = activeMessages(conv)
    conv.summary = 'an earlier brief, written when the chat was on a bigger model'
    conv.summaryCount = 2
    conv.summaryLastId = msgs[1].id
    conv.summaryModel = 'some-other-model'
    expect(summaryValid(conv)).toBe(true)

    const before = { state: summaryState(conv), messages: structuredClone(conv.messages) }
    const persist = vi.fn()

    await maybeCompact(conv, models.maxCtx, replyReserve(512), persist)

    expect(summaryState(conv)).toEqual(before.state)
    expect(conv.messages).toEqual(before.messages)
    expect(summaryValid(conv)).toBe(true)
    expect(persist).not.toHaveBeenCalled()
    expect(sends()).toEqual([])
    // and the work is still owed - a refusal consumes nothing.
    expect(compactionTarget(conv, models.maxCtx, replyReserve(512))).toBeGreaterThan(0)
  })
})

// ── the boundary itself ─────────────────────────────────────────────────────

describe('the 2688-token boundary', () => {
  it('refuses at exactly 2688, where the allowance is zero', async () => {
    const models = await cloudFleet(2688)
    const conv = chat('equal', CLOUD_MODEL, 3, 800)
    expect(models.maxCtx).toBe(2688)
    expect(compactionTarget(conv, models.maxCtx, replyReserve(512))).toBeGreaterThan(0)

    await maybeCompact(conv, models.maxCtx, replyReserve(512), vi.fn())

    expect(sends()).toEqual([])
    expect(conv.summary).toBeUndefined()
    expect(warn).toHaveBeenCalledTimes(1)
  })

  it('still compacts just above it, capped to the allowance', async () => {
    const models = await cloudFleet(2704)
    const conv = chat('above', CLOUD_MODEL, 4, 800)
    const maxReply = replyReserve(512)
    const count = compactionTarget(conv, models.maxCtx, maxReply)
    expect(count).toBeGreaterThan(0)
    const persist = vi.fn()

    await maybeCompact(conv, models.maxCtx, maxReply, persist)

    // a real request, on the real endpoint, with the real request shape
    const out = sends()
    expect(out).toHaveLength(1)
    expect(out[0].url).toBe('/api/cloud/ep1/v1/responses')
    expect(out[0].body.model).toBe(CLOUD_MODEL)
    expect(out[0].body.stream).toBe(false)
    expect(out[0].body.temperature).toBe(0)
    expect(out[0].body.max_output_tokens).toBe(640)
    expect(out[0].body.chat_template_kwargs).toEqual({ enable_thinking: false })
    // (2704 - 640 - 2048) * 4 = 64 characters, kept from the NEWEST end - the
    // tail of the last covered assistant turn.
    expect(out[0].body.input).toBe('y'.repeat(64))

    expect(conv.summary).toBe(fleet.summary)
    expect(conv.summaryCount).toBe(count)
    expect(conv.summaryLastId).toBe(activeMessages(conv)[count - 1].id)
    expect(conv.summaryModel).toBe(CLOUD_MODEL)
    expect(persist).toHaveBeenCalledTimes(1)
    expect(warn).not.toHaveBeenCalled()
  })
})

// ── the ordinary path is untouched ──────────────────────────────────────────

describe('a window with room', () => {
  it('summarizes the covered prefix and persists it', async () => {
    const models = await cloudFleet(32768)
    const conv = chat('roomy', CLOUD_MODEL, 10, 4000)
    const maxReply = replyReserve(4096)
    const count = compactionTarget(conv, models.maxCtx, maxReply)
    expect(count).toBeGreaterThan(0)
    const persist = vi.fn()

    await maybeCompact(conv, models.maxCtx, maxReply, persist)

    const out = sends()
    expect(out).toHaveLength(1)
    const input = String(out[0].body.input)
    // the whole covered prefix rode: nothing was sliced off a window this big
    // ((32768 - 640 - 2048) * 4 = 120320 characters of allowance).
    expect(input.length).toBeLessThanOrEqual(120_320)
    expect(input.startsWith('User: u0 ')).toBe(true)
    // exactly the covered prefix: its last turn is in, the first kept turn is
    // not - the summary must not cover a message the next prompt still sends.
    const msgs = activeMessages(conv)
    expect(input).toContain(messageText(msgs[count - 1]).slice(0, 6))
    expect(input).not.toContain(messageText(msgs[count]).slice(0, 6))
    expect(conv.summary).toBe(fleet.summary)
    expect(conv.summaryCount).toBe(count)
    expect(persist).toHaveBeenCalledTimes(1)
    expect(warn).not.toHaveBeenCalled()
  })
})

// ── negative controls: silence where silence is right ───────────────────────

describe('nothing to do', () => {
  it('says nothing when the window is not known yet', async () => {
    const models = useModelsStore()
    expect(models.maxCtx).toBe(0)
    const conv = chat('no-ctx', CLOUD_MODEL, 10, 4000)

    await maybeCompact(conv, models.maxCtx, replyReserve(512), vi.fn())

    expect(sends()).toEqual([])
    // an unknown window is not an impossible budget - it is "not yet".
    expect(warn).not.toHaveBeenCalled()
  })

  it('says nothing when the thread has not crossed the threshold', async () => {
    const models = await cloudFleet(2048)
    const conv = chat('no-target', CLOUD_MODEL, 1, 40)
    expect(compactionTarget(conv, models.maxCtx, replyReserve(512))).toBe(0)

    await maybeCompact(conv, models.maxCtx, replyReserve(512), vi.fn())

    expect(sends()).toEqual([])
    expect(warn).not.toHaveBeenCalled()
  })
})

describe('a request that fails on its own', () => {
  it('is still the ordinary failure, not a budget refusal', async () => {
    const models = await cloudFleet(32768)
    fleet.summaryStatus = 500
    const conv = chat('http-500', CLOUD_MODEL, 10, 4000)
    const persist = vi.fn()

    await maybeCompact(conv, models.maxCtx, replyReserve(4096), persist)

    // the request went out (the window had room) and the answer was refused
    expect(sends()).toHaveLength(1)
    expect(conv.summary).toBeUndefined()
    expect(persist).not.toHaveBeenCalled()
    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0][0])).not.toMatch(/no room/)

    // and the conversation is not left in flight by the failure
    fleet.summaryStatus = 200
    await maybeCompact(conv, models.maxCtx, replyReserve(4096), persist)
    expect(sends()).toHaveLength(2)
    expect(conv.summary).toBe(fleet.summary)
  })
})

// ── the refusal is not a dead end ───────────────────────────────────────────

describe('after a refusal', () => {
  it('compacts the same conversation once a bigger window is available', async () => {
    const small = await cloudFleet(2048)
    const conv = chat('retry', CLOUD_MODEL, 10, 4000)
    const persist = vi.fn()

    // turn 1: the model is served in a 2048-token window - nothing can be sent
    await maybeCompact(conv, small.maxCtx, replyReserve(512), persist)
    expect(sends()).toEqual([])
    expect(conv.summary).toBeUndefined()

    // turn 2: the same chat, now on a runner with room. The conversation must
    // not be stuck "in flight" from the refusal above.
    setActivePinia(createPinia())
    const big = await localFleet(32768)
    conv.model = LOCAL_MODEL
    const maxReply = replyReserve(4096)
    expect(compactionTarget(conv, big.maxCtx, maxReply)).toBeGreaterThan(0)

    await maybeCompact(conv, big.maxCtx, maxReply, persist)

    const out = sends()
    expect(out).toHaveLength(1)
    expect(out[0].url).toBe(`/api/runners/${PORT}/v1/responses`)
    expect(conv.summary).toBe(fleet.summary)
    expect(persist).toHaveBeenCalledTimes(1)
  })
})
