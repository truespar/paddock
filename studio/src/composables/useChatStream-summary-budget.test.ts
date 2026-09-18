// The compacted send, end to end through the real composable: does the reply
// cap pay for the summary the request actually carries?
//
// `buildBody` puts the summary into `instructions` under its own wrapper line,
// and "model maximum" resolves the cap from `promptTokensFrom` - so if that
// estimate does not know about the injected block, the request asks for a
// reply the window cannot hold beside its own prompt.
//
// These tests drive the PUBLIC path (`send` / `continueLast`), mock only the
// transport and the peripherals around it, and read the budget off the REAL
// JSON that would have gone on the wire. The estimator (lib/tokens.ts) and the
// body builder (useChatStream.ts) are both under test here, so neither is
// mocked.
//
// Out of scope, here as in the estimator: the date line, graph grounding, tool
// schemas, images, and the continue path's synthetic "Continue your previous
// reply" item - none of them are charged, and this change does not start.
// `windowRemaining`'s 512 floor is left alone too: every window below is
// thousands of tokens clear of it.

import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { Conversation, Message } from '@/types/chat'
import { useChatStore } from '@/stores/chat'

const MODEL = 'local/test-8b'
const ENDPOINT = 'http://127.0.0.1:11540/v1/responses'

// ── mocked peripherals ──────────────────────────────────────────────────────
//
// Stores that only read the browser (localStorage, sockets, window.location)
// and the background compaction pass (its own HTTP call, fired after the turn).
// The conversation store is NOT mocked - the thread, its tree and the send
// plan's indices are exactly what this is about.

const mock = vi.hoisted(() => ({
  settings: { maxTokens: null as number | null, maxToolCalls: null as number | null, summarize: true, markUnsure: false },
  window: 16_000,
  outCap: 0,
  cloud: true,
}))

vi.mock('@/stores/settings', () => ({ useSettingsStore: () => mock.settings }))
vi.mock('@/stores/models', () => ({
  takesTurns: () => true,
  cloudVendor: () => undefined,
  useModelsStore: () => ({
    models: [{ id: MODEL, kind: 'chat', status: 'ok', cloud: mock.cloud }],
    caps: {},
    maxCtx: mock.window,
    canChat: () => true,
    canTranscribe: () => false,
    capsFor: async () => ({ mcpServers: [] }),
    ctxFor: () => mock.window,
    outCapFor: () => mock.outCap,
    responsesUrl: () => ENDPOINT,
    specFor: () => undefined,
    visionFor: () => false,
    webSearchFor: () => false,
    thinkingBudgetFor: () => false,
    reasoningLadderFor: () => ({ levels: [], off: false, preserve: false }),
  }),
}))
vi.mock('@/stores/telemetry', () => ({
  useTelemetryStore: () => ({ beginCapture: () => {}, endCapture: () => undefined }),
}))
vi.mock('@/stores/prompts', () => ({ usePromptsStore: () => ({ prompts: [] }) }))
vi.mock('@/stores/toasts', () => ({ useToastsStore: () => ({ push: () => {} }) }))
vi.mock('@/stores/connectors', () => ({
  useConnectorsStore: () => ({ list: [], byId: () => undefined }),
}))
vi.mock('@/stores/graphs', () => ({ useGraphsStore: () => ({ groundingFor: () => '' }) }))
vi.mock('@/stores/artifacts', () => ({ useArtifactsStore: () => ({ refresh: async () => {} }) }))
vi.mock('@/lib/pdf', () => ({ pdfEngine: () => Promise.reject(new Error('no pdf here')) }))
vi.mock('@/lib/compact', () => ({ maybeCompact: vi.fn(async () => {}) }))

// imported after the mocks so the composable picks them up
const { useChatStream } = await import('@/composables/useChatStream')

// ── fixtures ────────────────────────────────────────────────────────────────
//
// Round numbers so every expectation below is arithmetic that can be redone by
// hand: 4 chars/token, 4 tokens of per-message overhead (lib/tokens.ts).

/** A labelled message body of exactly 8000 chars = 2000 tokens (+4 = 2004). */
function body(label: string): string {
  return `${label} ` + 'x'.repeat(8000 - label.length - 1)
}
const MSG_TOKENS = 2004
/** 40 chars = 10 tokens, +4 overhead. */
const SYS = 's'.repeat(40)
const SYS_TOKENS = 14
const SUMMARY = 'z'.repeat(400)
/** The test's own copy of the wrapper the builder injects. */
const WRAPPER = 'Summary of the earlier part of this conversation (older messages were compacted):\n'
/** (82 + 400) / 4 = 120.5 -> 121, + 4 tokens of overhead. */
const BLOCK_TOKENS = 125
const WINDOW = 16_000
/** windowRemaining's slack at this window: max(1024, 2%) = 1024. */
const SLACK = 1024

let captured: Record<string, unknown>[] = []
let fetchMock: ReturnType<typeof vi.fn>

/** One finished turn: a token, then a terminal event with usage. */
function sseBody(): ReadableStream<Uint8Array> {
  const frames = [
    'data: {"type":"response.output_text.delta","delta":"ok"}\n\n',
    'data: {"type":"response.completed","response":{"usage":{"input_tokens":9,"output_tokens":1}}}\n\n',
  ]
  const enc = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const f of frames) c.enqueue(enc.encode(f))
      c.close()
    },
  })
}

/** Seed the chat store's draft with a linear `n`-turn thread. */
function seed(n: number, last?: Partial<Message>): Conversation {
  const chat = useChatStore()
  const conv = chat.startDraft(MODEL)
  conv.systemPrompt = SYS
  // No tools: the picker's "all" would attach the artifacts MCP server, which
  // is about tool schemas (uncounted, out of scope) and not about the summary.
  conv.toolSelection = { mode: 'custom', picks: [] }
  for (let i = 0; i < n; i++) {
    conv.messages.push({
      id: `m${i}`,
      parentId: i === 0 ? null : `m${i - 1}`,
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: [{ type: 'text', text: body(`turn ${i}`) }],
      createdAt: i,
      ...(i === n - 1 ? last : {}),
    })
  }
  conv.leafId = `m${n - 1}`
  return conv
}

/** A valid stored summary over the thread's first `count` turns. */
function withSummary(conv: Conversation, count: number): void {
  conv.summary = SUMMARY
  conv.summaryCount = count
  conv.summaryLastId = `m${count - 1}`
  conv.summaryModel = MODEL
}

/** The request the transport saw. */
function sent(): Record<string, unknown> {
  expect(captured).toHaveLength(1)
  return captured[0]
}

/** The label of each history item the request carries, in order. */
function turns(req: Record<string, unknown>): string[] {
  return (req.input as { content: unknown }[]).map((it) =>
    String(it.content).split(' ').slice(0, 2).join(' '),
  )
}

beforeEach(() => {
  setActivePinia(createPinia())
  mock.settings = { maxTokens: null, maxToolCalls: null, summarize: true, markUnsure: false }
  mock.window = WINDOW
  mock.outCap = 0
  // A CLOUD lane by default: the client summary is what a cloud lane sends,
  // because a local single-model lane hands context to the runner instead
  // (`context_management`, exact tokens - see the last test in this file). The
  // continue path below is the other client-plan caller, and it is local.
  mock.cloud = true
  captured = []
  fetchMock = vi.fn(async (_url: string, init: { body: string }) => {
    captured.push(JSON.parse(init.body) as Record<string, unknown>)
    return { ok: true, body: sseBody() } as unknown as Response
  })
  vi.stubGlobal('fetch', fetchMock)
  // the delta apply is rAF-throttled; run it inline
  vi.stubGlobal('requestAnimationFrame', (cb: (t: number) => void) => {
    cb(0)
    return 0
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

// ── the send path ───────────────────────────────────────────────────────────
//
// Eight seeded turns + the question + the assistant placeholder = ten, of
// which nine cost 2004 and the empty placeholder costs 4. At a 16000 window
// with the 4096 default reply reserve the sliding window keeps five of them
// (from index 4), which is also where the stored summary's coverage ends - so
// the raw transcript is identical with compaction on and off, and the only
// difference on the wire is the summary block itself.
const ASK = body('turn 8')
/** 14 system + 5 x 2004 + 4 for the placeholder. */
const SEND_PROMPT = SYS_TOKENS + 5 * MSG_TOKENS + 4
/** What "model maximum" leaves for the reply when the summary is not charged. */
const SEND_CAP = WINDOW - SEND_PROMPT - SLACK

describe('send: a compacted prompt pays for its summary', () => {
  it('sends the summary block and takes it out of the reply budget', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()

    // one request, to the endpoint the composable resolved for this model
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe(ENDPOINT)
    expect(req.model).toBe(MODEL)
    expect(req.stream).toBe(true)
    // the block really is in the prompt, exactly once, under its own wrapper
    const instructions = req.instructions as string
    expect(instructions).toBe(`${instructions.split('\n\n')[0]}\n\n${SYS}\n\n${WRAPPER}${SUMMARY}`)
    expect(instructions.split(WRAPPER)).toHaveLength(2)
    // ...and the cap is the window minus the prompt INCLUDING that block
    expect(req.max_output_tokens).toBe(SEND_CAP - BLOCK_TOKENS)
    expect(req.max_output_tokens).toBe(4813)
    // far above the 512 floor: this is about the charge, not about clamping
    expect(req.max_output_tokens as number).toBeGreaterThan(4000)
  })

  it('charges exactly the block the request carries, once', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    await useChatStream().send([{ type: 'text', text: ASK }])
    const withBlock = sent()

    captured = []
    setActivePinia(createPinia())
    const plain = seed(8) // same thread, no stored summary
    expect(plain.summary).toBeUndefined()
    await useChatStream().send([{ type: 'text', text: ASK }])
    const noBlock = sent()

    const block = (withBlock.instructions as string).slice(
      (noBlock.instructions as string).length + 2,
    )
    expect(block).toBe(`${WRAPPER}${SUMMARY}`)
    // the builder's text and the estimator's charge are the same string:
    // ceil(chars / 4) + one message's worth of overhead
    expect((noBlock.max_output_tokens as number) - (withBlock.max_output_tokens as number)).toBe(
      Math.ceil(block.length / 4) + 4,
    )
    expect(noBlock.max_output_tokens).toBe(SEND_CAP)
  })

  it('charges a longer summary by exactly what it added', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    conv.summary = 'z'.repeat(800) // 400 chars more than the fixture
    await useChatStream().send([{ type: 'text', text: ASK }])
    // (82 + 800) / 4 = 220.5 -> 221, + 4 overhead = 225: exactly 100 tokens
    // more than the 400-char summary, which is the 400 extra chars and nothing
    // else - no second wrapper, no per-message multiple
    expect(sent().max_output_tokens).toBe(SEND_CAP - BLOCK_TOKENS - 100)
  })

  it('excludes the raw turns the summary replaced, and only those', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()
    // turns 0..3 are what the summary stands in for; the placeholder we are
    // filling is not history yet
    expect(turns(req)).toEqual(['turn 4', 'turn 5', 'turn 6', 'turn 7', 'turn 8'])
    expect(req.instructions as string).not.toContain('turn 0')
  })

  it('sends the same transcript with compaction on and off', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    mock.settings.summarize = false
    await useChatStream().send([{ type: 'text', text: ASK }])
    const off = sent()

    captured = []
    setActivePinia(createPinia())
    const on = seed(8)
    withSummary(on, 4)
    mock.settings.summarize = true
    await useChatStream().send([{ type: 'text', text: ASK }])
    const compacted = sent()

    expect(compacted.input).toEqual(off.input)
    expect(compacted.instructions).toBe(`${off.instructions}\n\n${WRAPPER}${SUMMARY}`)
    expect(off.max_output_tokens).toBe(SEND_CAP)
    expect(compacted.max_output_tokens).toBe(SEND_CAP - BLOCK_TOKENS)
    expect(conv.summary).toBe(SUMMARY) // compaction off never touched the store
  })

  // ── negative controls ────────────────────────────────────────────────────

  it('charges nothing when compaction is switched off', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    mock.settings.summarize = false
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()
    expect(req.instructions).not.toContain(WRAPPER)
    expect(req.max_output_tokens).toBe(SEND_CAP)
  })

  it('charges nothing when the stored summary no longer matches the thread', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    conv.summaryLastId = 'a-turn-that-was-edited-away'
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()
    expect(req.instructions).not.toContain(WRAPPER)
    expect(req.max_output_tokens).toBe(SEND_CAP)
    expect(turns(req)).toEqual(['turn 4', 'turn 5', 'turn 6', 'turn 7', 'turn 8'])
  })

  it('charges nothing while the whole thread still fits raw', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    mock.window = 200_000 // nothing has to give way: the summary stays in reserve
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()
    expect(req.instructions).not.toContain(WRAPPER)
    expect(turns(req)[0]).toBe('turn 0')
    // 14 system + 9 x 2004 + 4 placeholder, slack = 2% of the window
    expect(req.max_output_tokens).toBe(200_000 - (SYS_TOKENS + 9 * MSG_TOKENS + 4) - 4_000)
  })

  it('leaves an explicit reply cap exactly as the user set it', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    mock.settings.maxTokens = 2048
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()
    // the summary still rides; the cap is a promise, not a leftover
    expect(req.instructions).toContain(`${WRAPPER}${SUMMARY}`)
    expect(req.max_output_tokens).toBe(2048)
  })

  it('still yields to a provider reply ceiling below the window', async () => {
    const conv = seed(8)
    withSummary(conv, 4)
    mock.outCap = 1500
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(sent().max_output_tokens).toBe(1500)
  })

  it('has no client summary to charge on a local lane, which the runner compacts', async () => {
    // Why the tests above run on a cloud lane: a local single-model send arms
    // the runner's own `context_management` instead of injecting a summary, so
    // there is no known text here to charge and nothing on this branch changes.
    const conv = seed(8)
    withSummary(conv, 4)
    mock.cloud = false
    await useChatStream().send([{ type: 'text', text: ASK }])
    const req = sent()
    // 70% of the 10880-token prompt budget, in the runner's own tokens
    expect(req.context_management).toEqual([{ type: 'compaction', compact_threshold: 7615 }])
    expect(req.instructions).not.toContain(WRAPPER)
    expect(req.max_output_tokens).toBe(SEND_CAP)
  })
})

// ── the continue path ───────────────────────────────────────────────────────
//
// Eight seeded turns, the last one a reply that hit the cap; continue resends
// the thread and appends. The window keeps five turns (from index 3) and the
// summary covers three, so again the raw transcript does not move.
//
// This runs on a LOCAL lane: a continue is excluded from the runner's
// server-side compaction (its synthetic trailing user item would corrupt the
// tail anchor), so it is on the client plan and carries the summary itself.
/** 14 system + 5 x 2004 (no placeholder: the turn being continued is real). */
const CONT_PROMPT = SYS_TOKENS + 5 * MSG_TOKENS
const CONT_CAP = WINDOW - CONT_PROMPT - SLACK

describe('continue: the resumed turn pays for the summary too', () => {
  beforeEach(() => {
    mock.cloud = false
  })

  it('takes the block out of the budget it asks for', async () => {
    const conv = seed(8, { incomplete: 'length' })
    withSummary(conv, 3)
    await useChatStream().continueLast()
    const req = sent()
    expect(req.instructions).toContain(`${WRAPPER}${SUMMARY}`)
    // the trailing item is continue's own synthetic instruction, which this
    // estimator has never charged and still does not
    expect(turns(req)).toEqual([
      'turn 3',
      'turn 4',
      'turn 5',
      'turn 6',
      'turn 7',
      'Continue your',
    ])
    expect(req.max_output_tokens).toBe(CONT_CAP - BLOCK_TOKENS)
    expect(req.max_output_tokens).toBe(4817)
  })

  it('asks for more when there is no summary to carry', async () => {
    seed(8, { incomplete: 'length' })
    await useChatStream().continueLast()
    const req = sent()
    expect(req.instructions).not.toContain(WRAPPER)
    expect(req.max_output_tokens).toBe(CONT_CAP)
  })
})
