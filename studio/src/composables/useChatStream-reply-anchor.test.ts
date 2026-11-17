// The cloud reply budget, end to end through the real composable.
//
// Three behaviours, read off the JSON that would have gone on the wire:
//  - the second turn of a chat is budgeted from the first turn's server-counted
//    prompt, not from a guess at the whole thread;
//  - a window with no room for a reply refuses the turn before any request,
//    and a refused Continue keeps its cut-off marker;
//  - a provider that refuses on size and states its numbers gets exactly one
//    corrected resend.
//
// Only the transport and the peripheral stores are mocked. The chat store, the
// estimator (lib/tokens.ts) and the body builder are the code under test.

import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { Conversation, Message } from '@/types/chat'
import { useChatStore } from '@/stores/chat'

const MODEL = 'cloud/test-model'
const ENDPOINT = 'http://127.0.0.1:11500/api/cloud/v1/responses'

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
    asksInChat: () => false,
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

// -- fixtures ----------------------------------------------------------------
//
// 4 chars a token, 4 tokens of per-message overhead (lib/tokens.ts).

/** 800 chars = 200 tokens, + 4 overhead = 204. */
const ASK = 'q'.repeat(800)
const ASK_TOKENS = 204
/** The streamed answer "ok": 1 token of text + 4, and 1 counted output token + 4. */
const ANSWER_TOKENS = 5
const PLACEHOLDER_TOKENS = 4
const WINDOW = 16_000
const FLAT_SLACK = 1024
const ANCHORED_MARGIN = 128

type Reply = { status?: number; error?: string; inputTokens?: number }
let replies: Reply[] = []
let captured: Record<string, unknown>[] = []

function sseBody(inputTokens: number): ReadableStream<Uint8Array> {
  const frames = [
    'data: {"type":"response.output_text.delta","delta":"ok"}\n\n',
    `data: {"type":"response.completed","response":{"usage":{"input_tokens":${inputTokens},"output_tokens":1}}}\n\n`,
  ]
  const enc = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const f of frames) c.enqueue(enc.encode(f))
      c.close()
    },
  })
}

function draft(): Conversation {
  const conv = useChatStore().startDraft(MODEL)
  // No tools: the picker's "all" would attach the artifacts MCP server.
  conv.toolSelection = { mode: 'custom', picks: [] }
  return conv
}

function lastAssistant(conv: Conversation): Message {
  const a = [...conv.messages].reverse().find((m) => m.role === 'assistant')
  if (!a) throw new Error('no assistant turn')
  return a
}

beforeEach(() => {
  setActivePinia(createPinia())
  mock.settings = { maxTokens: null, maxToolCalls: null, summarize: true, markUnsure: false }
  mock.window = WINDOW
  mock.outCap = 0
  mock.cloud = true
  captured = []
  replies = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init: { body: string }) => {
      captured.push(JSON.parse(init.body) as Record<string, unknown>)
      const r = replies.shift() ?? {}
      if (r.error !== undefined) {
        return {
          ok: false,
          status: r.status ?? 400,
          body: null,
          json: async () => ({ error: { message: r.error } }),
        } as unknown as Response
      }
      return { ok: true, body: sseBody(r.inputTokens ?? 9) } as unknown as Response
    }),
  )
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

describe('the second turn is budgeted from what the provider counted', () => {
  it('estimates the whole prompt on a first turn, with the flat slack', async () => {
    draft()
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(1)
    expect(captured[0].max_output_tokens).toBe(WINDOW - (ASK_TOKENS + PLACEHOLDER_TOKENS) - FLAT_SLACK)
  })

  it('anchors on the counted prompt and charges its margin to the new turn only', async () => {
    const conv = draft()
    // the provider counted 5000: tool schemas and a system prompt no estimate sees
    replies = [{ inputTokens: 5000 }]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(lastAssistant(conv).usage?.promptTokens).toBe(5000)
    expect(lastAssistant(conv).run?.shape).toEqual({ from: conv.messages[0].id, summary: null, tools: '#c' })

    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(2)
    const estimated = ANSWER_TOKENS + ASK_TOKENS + PLACEHOLDER_TOKENS
    expect(captured[1].max_output_tokens).toBe(WINDOW - 5000 - estimated - ANCHORED_MARGIN)
    expect(captured[1].max_output_tokens).toBe(10_659)
    // the guess this replaces knew nothing of those 5000 tokens
    const guessed = WINDOW - (ASK_TOKENS * 2 + ANSWER_TOKENS + PLACEHOLDER_TOKENS) - FLAT_SLACK
    expect(guessed).toBe(14_559)
  })

  it('drops the anchor when the request no longer shares its prefix', async () => {
    const conv = draft()
    replies = [{ inputTokens: 5000 }]
    await useChatStream().send([{ type: 'text', text: ASK }])
    conv.systemPrompt = 'Answer in one sentence.'
    await useChatStream().send([{ type: 'text', text: ASK }])
    // 23 chars of system prompt = 6 tokens + 4
    const prompt = 10 + ASK_TOKENS * 2 + ANSWER_TOKENS + PLACEHOLDER_TOKENS
    expect(captured[1].max_output_tokens).toBe(WINDOW - prompt - FLAT_SLACK)
  })

  it('leaves a local lane alone: the whole window, and the runner clamps', async () => {
    mock.cloud = false
    draft()
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured[0].max_output_tokens).toBe(WINDOW)
  })
})

describe('a window with no room refuses the turn', () => {
  it('sends nothing and says why', async () => {
    mock.window = 400
    const conv = draft()
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(0)
    const a = lastAssistant(conv)
    expect(a.error).toMatch(/^No room left for a reply/)
    expect(a.error).toContain('400-token context window')
    expect(a.streaming).toBe(false)
  })

  it('keeps a refused Continue resumable, and clears the refusal when it goes through', async () => {
    const conv = draft()
    await useChatStream().send([{ type: 'text', text: ASK }])
    const a = lastAssistant(conv)
    a.incomplete = 'length'

    mock.window = 200
    await useChatStream().continueLast()
    expect(captured).toHaveLength(1) // still only the first send
    expect(a.incomplete).toBe('length')
    expect(a.error).toMatch(/^No room left for a reply/)

    mock.window = WINDOW
    await useChatStream().continueLast()
    expect(captured).toHaveLength(2)
    expect(a.error).toBeUndefined()
  })
})

describe('a provider that states its numbers gets one corrected resend', () => {
  const OVER = 'input length and `max_tokens` exceed context limit: 15000 + 14772 > 16000, decrease input length or `max_tokens` and try again'

  it('resends with exactly what fits, and records what rode', async () => {
    const conv = draft()
    replies = [{ error: OVER }, {}]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(2)
    expect(captured[1].max_output_tokens).toBe(1000)
    expect({ ...captured[1], max_output_tokens: 0 }).toEqual({ ...captured[0], max_output_tokens: 0 })
    const a = lastAssistant(conv)
    expect(a.error).toBeUndefined()
    expect(a.run?.params.maxTokens).toBe(1000)
  })

  it('tries once: a second refusal is shown as it came', async () => {
    const conv = draft()
    replies = [{ error: OVER }, { error: OVER }]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(2)
    expect(lastAssistant(conv).error).toBe(OVER)
  })

  it('does not resend when what fits is not worth having', async () => {
    const conv = draft()
    const full = 'input length and `max_tokens` exceed context limit: 15900 + 14772 > 16000'
    replies = [{ error: full }]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(1)
    expect(lastAssistant(conv).error).toBe(full)
  })

  it('bounds a custom cap to the provider refusal without rewriting the preference', async () => {
    mock.settings.maxTokens = 8000
    const conv = draft()
    replies = [{ error: OVER }, {}]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(2)
    expect(captured[0].max_output_tokens).toBe(8000)
    expect(captured[1].max_output_tokens).toBe(1000)
    expect(mock.settings.maxTokens).toBe(8000)
    expect(lastAssistant(conv).run?.params.maxTokens).toBe(1000)
    expect(lastAssistant(conv).error).toBeUndefined()
  })

  it('shows any other error untouched', async () => {
    const conv = draft()
    replies = [{ status: 401, error: 'Invalid API key' }]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(1)
    expect(lastAssistant(conv).error).toBe('Invalid API key')
  })
  it('never retries a rate limit just because its message mentions context', async () => {
    const conv = draft()
    replies = [{ status: 429, error: OVER }]
    await useChatStream().send([{ type: 'text', text: ASK }])
    expect(captured).toHaveLength(1)
    expect(lastAssistant(conv).error).toBe(OVER)
  })
})
