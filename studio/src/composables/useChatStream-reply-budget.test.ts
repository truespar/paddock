// The reply budget on the REAL send path (composables/useChatStream.ts).
//
// run() resolves "Model maximum" into one number and hands it to buildBody,
// which writes `max_output_tokens` only when that number is positive. So the
// two failures below are the same failure seen from two ends:
//
//  - a budget floored up to 512 asks a provider that publishes 128 for 512,
//    and asks for 512 tokens of reply in a window the prompt already filled;
//  - a budget of 0 travels as an ABSENT field, which on the wire does not mean
//    "no room" - it means "no cap", the provider's own unbounded default, on
//    exactly the request that had no room.
//
// These tests drive the real composable through its public surface (`send`,
// `continueLast`) with the document, GPU and transport boundaries mocked, and
// read what actually reached `fetch`. Nothing here re-implements the budget:
// lib/tokens.ts and useChatStream.ts are both the production modules.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { Conversation, Message } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import { useChatStream } from '@/composables/useChatStream'
import { useChatStore } from '@/stores/chat'
import { useModelsStore, type ModelCaps, type ModelInfo } from '@/stores/models'
import { useSettingsStore } from '@/stores/settings'

// ── mocked peripherals ──────────────────────────────────────────────────────
// Everything below is a boundary this turn does not exercise: the PDF/raster
// engine, the speech endpoints, the GPU telemetry socket, the manager's REST
// surface, and the background summarizer. The send path itself, the token
// accounting and every store that decides the budget stay real.

vi.mock('@/lib/pdf', () => ({
  pdfEngine: vi.fn(async () => {
    throw new Error('the pdf engine is not available in tests')
  }),
}))
vi.mock('@/lib/transcribe', () => ({ transcribeClip: vi.fn() }))
vi.mock('@/lib/align', () => ({
  alignClip: vi.fn(),
  alignmentRefused: () => true,
  mergeWordTimes: () => undefined,
}))
vi.mock('@/lib/compact', () => ({ maybeCompact: vi.fn(async () => {}) }))
vi.mock('@/lib/model-name', () => ({ fleetLabel: (id: string) => id }))
vi.mock('@/lib/api', () => ({
  attachmentsApi: {
    url: (id: string) => `/api/attachments/${id}`,
    storeMetadata: vi.fn(async () => {}),
  },
  forensicsApi: { persist: vi.fn(async () => {}) },
  promptsApi: { list: vi.fn(async () => []) },
  registryApi: { catalog: vi.fn(async () => []) },
  gpuApi: { snapshot: vi.fn(async () => undefined) },
  store: { putConversation: vi.fn(async () => {}) },
  stubFromSummary: vi.fn(),
}))
vi.mock('@/stores/telemetry', () => ({
  useTelemetryStore: () => ({
    beginCapture: vi.fn(),
    endCapture: () => undefined,
  }),
}))

// ── fixtures ────────────────────────────────────────────────────────────────

const CLOUD_ENDPOINT = 'openrouter'
let seq = 0

function memoryStorage(): Storage {
  const map = new Map<string, string>()
  return {
    get length() {
      return map.size
    },
    clear: () => map.clear(),
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => [...map.keys()][i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
  } as Storage
}

function caps(over: Partial<ModelCaps> = {}): ModelCaps {
  return {
    webSearch: false,
    mcpServers: [],
    taskTags: [],
    timestampGranularities: [],
    include: [],
    ...over,
  }
}

/** A cloud pick as the models store builds one: its window and its published
 *  reply ceiling both live on the entry, and `capsFor` answers "nothing
 *  advertised" for it without any fetch - the real cloud lane exactly. */
function cloudModel(id: string, opts: { ctx?: number; maxOut?: number } = {}): ModelInfo {
  return {
    id,
    ownedBy: 'cloud',
    display: id,
    kind: 'chat',
    status: 'ok',
    cloud: {
      endpoint: CLOUD_ENDPOINT,
      endpointName: 'OpenRouter',
      ctx: opts.ctx,
      maxOut: opts.maxOut,
    },
  }
}

function localModel(id: string, port = 11540): ModelInfo {
  return { id, ownedBy: 'paddock', display: id, kind: 'chat', status: 'ok', port }
}

function conversation(over: Partial<Conversation> = {}): Conversation {
  return {
    id: `conv-${++seq}`,
    // Not 'New chat': titling is a separate concern and would persist on its own.
    title: 'reply budget',
    messages: [],
    model: 'cloud:openrouter:acme/one',
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
    // Tool-free and web-free on purpose: this turn is about the reply cap, and
    // a tool list would pull the connector/graph/artifact surfaces in with it.
    toolSelection: { mode: 'custom', picks: [] },
    webSearchEnabled: false,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...over,
  }
}

function userTurn(text: string): Message {
  return {
    id: `u-${++seq}`,
    parentId: null,
    role: 'user',
    content: [{ type: 'text', text }],
    createdAt: Date.now(),
  }
}

/** A prompt far too big for the window it is about to be sent into: 40k
 *  characters is ~10k estimated tokens, and the trimmer always keeps the
 *  newest turn however large it is (lib/tokens.ts `trimIndex`). */
const OVERSIZED = 'x'.repeat(40_000)

/** The transport boundary. One mock for the whole file (so its call list is
 *  strongly typed) with a per-test handler behind it. */
let respond: (url: string, init?: RequestInit) => Promise<Response>
const fetchMock = vi.fn((url: string, init?: RequestInit) => respond(url, init))

/** A minimal Responses stream: one text delta and a terminal usage event. */
function sseResponse(body: string): Response {
  const enc = new TextEncoder()
  return {
    ok: true,
    status: 200,
    body: new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(enc.encode(body))
        c.close()
      },
    }),
  } as unknown as Response
}

function okStream(): Response {
  const events = [
    { type: 'response.output_text.delta', delta: 'ok' },
    { type: 'response.completed', response: { usage: { input_tokens: 7, output_tokens: 1 } } },
  ]
  return sseResponse(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(''))
}

/** Every generation request that reached the wire, decoded. */
function generationBodies(): Record<string, unknown>[] {
  return fetchMock.mock.calls
    .filter(([url]) => url.endsWith('/v1/responses'))
    .map(([, init]) => JSON.parse(String(init?.body)) as Record<string, unknown>)
}

beforeEach(() => {
  seq = 0
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', memoryStorage())
  // The chat store debounces its saves through `window.setTimeout`; nothing
  // here wants a save to land, so the timer is a no-op rather than a fake clock.
  vi.stubGlobal('window', {
    setTimeout: () => 0,
    clearTimeout: () => {},
    location: { port: '11500' },
  })
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    cb(0)
    return 0
  })
  respond = async (url) => {
    if (url.endsWith('/v1/responses')) return okStream()
    throw new Error(`unexpected fetch: ${url}`)
  }
  fetchMock.mockClear()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  fetchMock.mockClear()
})

/** Put `conv` on screen and hand back the composable plus a persist spy.
 *  Persistence is the boundary: `persistNow` is what writes the turn's settled
 *  state (error included) back to the manager. */
function open(conv: Conversation) {
  const chat = useChatStore()
  chat.conversations = [conv]
  chat.activeId = conv.id
  const persistNow = vi.spyOn(chat, 'persistNow').mockResolvedValue(undefined)
  return { chat, persistNow, stream: useChatStream() }
}

function lastAssistant(conv: Conversation): Message {
  const found = [...conv.messages].reverse().find((m) => m.role === 'assistant')
  if (!found) throw new Error('no assistant turn on the conversation')
  return found
}

// ── the provider's published cap must reach the wire unrounded ──────────────

describe('useChatStream: a low provider cap rides as it is', () => {
  it('sends the provider’s own 128-token ceiling, not a 512 floor', async () => {
    const id = 'cloud:openrouter:acme/small-out'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 131072, maxOut: 128 })]
    const conv = conversation({ model: id })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: 'Summarise this.' }])

    const bodies = generationBodies()
    expect(bodies).toHaveLength(1)
    expect(bodies[0].max_output_tokens).toBe(128)
    // Provenance must agree with the wire - the run record is what the turn's
    // details panel shows for "why did this stop".
    expect(lastAssistant(conv).run?.params.maxTokens).toBe(128)
    expect(lastAssistant(conv).error).toBeUndefined()
  })

  it('keeps an ordinary allowance window-bound when no cap is published', async () => {
    const id = 'cloud:openrouter:acme/open-out'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 131072 })]
    const conv = conversation({ model: id })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: 'Summarise this.' }])

    const bodies = generationBodies()
    expect(bodies).toHaveLength(1)
    // 131072 window - 12 estimated prompt tokens - 2621 slack
    expect(bodies[0].max_output_tokens).toBe(128439)
    expect(lastAssistant(conv).error).toBeUndefined()
    expect(stream.isStreaming.value).toBe(false)
  })
})

// ── a zero budget is a refusal, never an absent field ───────────────────────

describe('useChatStream: an exhausted window refuses the turn', () => {
  it('reports it, settles the turn and never reaches the generation endpoint', async () => {
    const id = 'cloud:openrouter:acme/tiny-ctx'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 4096 })]
    const conv = conversation({ model: id })
    const { stream, persistNow } = open(conv)

    await stream.send([{ type: 'text', text: OVERSIZED }])

    expect(generationBodies()).toHaveLength(0)
    const assistant = lastAssistant(conv)
    expect(assistant.error).toBeTruthy()
    expect(assistant.error).toMatch(/no room/i)
    expect(assistant.error).toMatch(/context window/i)
    // actionable, not just descriptive
    expect(assistant.error).toMatch(/shorten|new chat|larger context/i)
    expect(assistant.streaming).toBe(false)
    expect(persistNow).toHaveBeenCalledWith(conv)
    // Not left busy: the composer's Stop button and its send guard both key on
    // this, so a refused turn that never released it would freeze the chat.
    expect(stream.isStreaming.value).toBe(false)
  })

  it('leaves the raw history exactly as it was', async () => {
    const id = 'cloud:openrouter:acme/tiny-ctx'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 4096 })]
    const conv = conversation({ model: id })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: OVERSIZED }])

    const user = conv.messages.find((m) => m.role === 'user')
    expect(user?.content).toEqual([{ type: 'text', text: OVERSIZED }])
    expect(conv.messages.filter((m) => m.role === 'user')).toHaveLength(1)
    expect(conv.summary).toBeUndefined()
    expect(conv.summaryCount).toBeUndefined()
    expect(conv.serverCompaction).toBeUndefined()
  })

  it('refuses a continue of a cut-off reply the same way, keeping its text', async () => {
    const id = 'cloud:openrouter:acme/tiny-ctx'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 4096 })]
    const user = userTurn(OVERSIZED)
    const partial: Message = {
      id: 'a-partial',
      parentId: user.id,
      role: 'assistant',
      content: [{ type: 'text', text: 'As far as I got' }],
      model: id,
      incomplete: 'length',
      createdAt: Date.now(),
    }
    const conv = conversation({ model: id, messages: [user, partial], leafId: partial.id })
    const { stream } = open(conv)

    await stream.continueLast()

    expect(generationBodies()).toHaveLength(0)
    expect(partial.error).toMatch(/no room/i)
    expect(partial.streaming).toBe(false)
    // the answer it already produced is history, not something a refusal edits
    expect(partial.content).toEqual([{ type: 'text', text: 'As far as I got' }])
    expect(stream.isStreaming.value).toBe(false)
  })

  it('refuses every lane of a compare fan-out', async () => {
    const a = 'cloud:openrouter:acme/lane-a'
    const b = 'cloud:openrouter:acme/lane-b'
    const models = useModelsStore()
    models.models = [cloudModel(a, { ctx: 4096 }), cloudModel(b, { ctx: 8192 })]
    const conv = conversation({ model: a, compareModels: [a, b] })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: OVERSIZED }])

    expect(generationBodies()).toHaveLength(0)
    const lanes = conv.messages.filter((m) => m.role === 'assistant')
    expect(lanes).toHaveLength(2)
    for (const lane of lanes) {
      expect(lane.error).toMatch(/no room/i)
      expect(lane.streaming).toBe(false)
    }
    expect(stream.isStreaming.value).toBe(false)
  })
})

// ── negative controls ───────────────────────────────────────────────────────

describe('useChatStream: what the refusal must NOT touch', () => {
  it('still sends an explicitly chosen cap into a full window', async () => {
    const id = 'cloud:openrouter:acme/tiny-ctx'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 4096 })]
    const settings = useSettingsStore()
    // An explicit cap is a promise the send makes room for, not a number the
    // window derives - its policy is deliberately unchanged here.
    settings.maxTokens = 2048
    const conv = conversation({ model: id })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: OVERSIZED }])

    const bodies = generationBodies()
    expect(bodies).toHaveLength(1)
    expect(bodies[0].max_output_tokens).toBe(2048)
    expect(lastAssistant(conv).error).toBeUndefined()
  })

  it('still uses the unknown-window fallback rather than refusing', async () => {
    // A local lane whose endpoint has not reported max_ctx yet: the window
    // reads 0, which the planners take as "do not trim" - and the reply
    // budget as the default reserve.
    const id = 'qwen3-8b'
    const models = useModelsStore()
    models.models = [localModel(id)]
    models.caps = { [id]: caps() }
    const conv = conversation({ model: id })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: OVERSIZED }])

    const bodies = generationBodies()
    expect(bodies).toHaveLength(1)
    expect(bodies[0].max_output_tokens).toBe(4096)
    expect(lastAssistant(conv).error).toBeUndefined()
  })

  it('never omits max_output_tokens on a request it does send', async () => {
    const id = 'cloud:openrouter:acme/small-out'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 8192, maxOut: 1 })]
    const conv = conversation({ model: id })
    const { stream } = open(conv)

    await stream.send([{ type: 'text', text: 'hi' }])

    const bodies = generationBodies()
    expect(bodies).toHaveLength(1)
    // 1 is the provider's whole reply ceiling: it must ride, because an absent
    // field means the provider's own unbounded default instead.
    expect(bodies[0]).toHaveProperty('max_output_tokens', 1)
  })
})


describe('a refused continuation stays recoverable', () => {
  it('retains the cutoff state and retries with the partial answer after more context becomes available', async () => {
    const id = 'cloud:openrouter:acme/resized-ctx'
    const models = useModelsStore()
    models.models = [cloudModel(id, { ctx: 4096 })]
    const user = userTurn(OVERSIZED)
    const partial: Message = {
      id: 'recoverable-partial', parentId: user.id, role: 'assistant',
      content: [{ type: 'text', text: 'As far as I got' }], model: id,
      incomplete: 'length', stopped: true, createdAt: Date.now(),
    }
    const conv = conversation({ model: id, messages: [user, partial], leafId: partial.id })
    const { stream } = open(conv)
    await stream.continueLast()
    expect(generationBodies()).toHaveLength(0)
    expect(partial.incomplete).toBe('length')
    expect(partial.stopped).toBe(true)
    expect(partial.error).toMatch(/no room/i)
    expect(stream.isStreaming.value).toBe(false)

    models.models = [cloudModel(id, { ctx: 32768 })]
    await stream.continueLast()
    const requests = generationBodies()
    expect(requests).toHaveLength(1)
    expect(JSON.stringify(requests[0].input)).toContain('As far as I got')
    expect(partial.content).toEqual([{ type: 'text', text: 'As far as I gotok' }])
    expect(partial.incomplete).toBeUndefined()
    expect(partial.error).toBeUndefined()
    expect(partial.stopped).toBe(false)
    expect(partial.streaming).toBe(false)
    expect(stream.isStreaming.value).toBe(false)
  })
})
