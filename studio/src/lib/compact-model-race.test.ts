// Tests for the MODEL-PROVENANCE race in background compaction (lib/compact.ts).
//
// Compaction is the one thing in the Studio that runs after a turn finishes and
// writes back to the conversation while the user keeps using it. Two facts make
// that a race rather than a detail:
//
//  1. The model selection is MUTABLE and reachable for the whole time the
//     summary request is out - the header dropdown goes through
//     lib/select-model.ts `selectStudioModel` into the chat store's `edit`,
//     which is awaited machinery of its own.
//  2. The transcript is OWN-HISTORY-ONLY. `renderTranscript` filters a compare
//     group's lanes against whichever model is asking, so the body sent for
//     model A is a DIFFERENT document from the body that would be sent for
//     model B, and the summary that comes back is only an answer about A's
//     history.
//
// So a result that lands after the selection moved is not "a summary with a
// stale label" - it is A's filtered view of the thread, and writing it under
// B's name both mislabels it where the UI reads `summaryModel`
// (useChatStream.ts `previewPlan` -> `by`) and smuggles it into B's context on
// the next send.
//
// Every request below is DEFERRED: the transport promise is resolved by hand
// after the mutation under test, so the mutation provably happens while the
// request is in flight rather than before it was ever built.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Conversation, Message } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import { activeMessages, focusStep, stepSibling } from '@/lib/tree'
import { compactionTarget, summaryValid } from '@/lib/tokens'

// ── mocked boundaries: endpoint resolution and the conversation store's API ──

const boundary = vi.hoisted(() => ({
  /** model id -> its runner relay, standing in for models.responsesUrl. */
  endpoints: new Map<string, string>(),
  api: {
    listConversations: vi.fn(),
    getConversation: vi.fn(),
    putConversation: vi.fn(),
    deleteConversation: vi.fn(),
  },
}))

vi.mock('@/stores/models', () => ({
  useModelsStore: () => ({
    responsesUrl: (id?: string) => boundary.endpoints.get(id ?? ''),
  }),
}))

vi.mock('@/lib/api', () => ({
  store: boundary.api,
  // never reached: nothing here hydrates the conversation LIST
  stubFromSummary: vi.fn(),
}))

// Those two are the only stubs. `vi.mock` is hoisted above the imports, so
// compact.ts binds the stubbed endpoint resolution and the chat store the
// stubbed conversation API - and everything else here (the compactor itself,
// the tree, the token accounting, the store's own logic, the types) is the
// real module. `vi.hoisted` above exists for the same reason: the factories
// run before this file's own top-level code.
import { maybeCompact } from './compact'
import { useChatStore } from '@/stores/chat'
import { createPinia, setActivePinia } from 'pinia'

// ── fixture ─────────────────────────────────────────────────────────────────

const MAX_CTX = 32_768
const MAX_REPLY = 1_024

const MODEL_A = 'qwen3.8-flash-next-Q8_0'
const MODEL_B = 'gemma4-27b-Q4_K_M'
const MODEL_HARMONY = 'gpt-oss-120b'

const URL_A = '/api/runners/11540/v1/responses'
const URL_B = '/api/runners/11541/v1/responses'
const URL_HARMONY = '/api/runners/11542/v1/responses'

/** Message text: a marker the transcript assertions look for, padded to a
 *  fixed size so the fixture reliably crosses the 70%-of-budget trigger and
 *  lands the coverage boundary on a known turn. Dots, not spaces - the
 *  renderer trims. */
function body(mark: string): string {
  return mark + '.'.repeat(12_000 - mark.length)
}

let seq = 0

function msg(
  role: 'user' | 'assistant',
  mark: string,
  parentId: string | null,
  extra: Partial<Message> = {},
): Message {
  return {
    id: `${mark}-${++seq}`,
    role,
    parentId,
    content: [{ type: 'text', text: body(mark) }],
    createdAt: ++seq,
    ...extra,
  }
}

interface Fixture {
  conv: Conversation
  q1: Message
  laneA: Message
  laneB: Message
  q2: Message
  a2: Message
  q3: Message
  a3: Message
  q4: Message
}

/** A thread long enough to compact, whose covered prefix contains a COMPARE
 *  GROUP - one question answered by both models. That group is what makes the
 *  request body model-specific, so it is the part of the fixture that actually
 *  carries the bug.
 *
 *  On screen (activeMessages), index by index:
 *    0 q1 · 1 laneA (group, A) · 2 laneB (group, B) · 3 q2 · 4 a2
 *    5 q3 · 6 a3 · 7 q4
 *  The compaction target works out to 5, so the coverage boundary is `a2` and
 *  q3/a3/q4 stay raw.  */
function fixture(): Fixture {
  const q1 = msg('user', 'MARK-Q1', null)
  const laneA = msg('assistant', 'MARK-LANE-A', q1.id, { group: 'g1', model: MODEL_A })
  const laneB = msg('assistant', 'MARK-LANE-B', q1.id, { group: 'g1', model: MODEL_B })
  // the next turn hangs off the group's ANCHOR, which is lane 1
  const q2 = msg('user', 'MARK-Q2', laneA.id)
  const a2 = msg('assistant', 'MARK-A2', q2.id, { model: MODEL_A })
  const q3 = msg('user', 'MARK-Q3', a2.id)
  const a3 = msg('assistant', 'MARK-A3', q3.id, { model: MODEL_A })
  const q4 = msg('user', 'MARK-Q4', a3.id)
  const conv: Conversation = {
    id: `race-conv-${++seq}`,
    title: 'a long chat',
    messages: [q1, laneA, laneB, q2, a2, q3, a3, q4],
    leafId: q4.id,
    model: MODEL_A,
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
    createdAt: 0,
    updatedAt: 0,
  }
  return { conv, q1, laneA, laneB, q2, a2, q3, a3, q4 }
}

// ── deferred transport ──────────────────────────────────────────────────────

/** Exactly the request compact.ts builds (it is a Record<string, unknown> on
 *  the production side; here it is named so the assertions read). */
interface SummaryRequest {
  model: string
  instructions: string
  input: string
  temperature: number
  stream: boolean
  max_output_tokens: number
  reasoning?: { effort: string }
  chat_template_kwargs?: { enable_thinking: boolean }
}

interface Pending {
  url: string
  req: SummaryRequest
  /** answer with a Responses body carrying this summary text */
  ok: (summary: string) => void
  /** answer with a non-2xx */
  status: (code: number) => void
  /** never reach the server at all */
  boom: (e: unknown) => void
}

let pending: Pending[] = []

function stubTransport(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      (url: unknown, init: RequestInit) =>
        new Promise((resolve, reject) => {
          pending.push({
            url: String(url),
            req: JSON.parse(String(init.body)) as SummaryRequest,
            ok: (summary) =>
              resolve({
                ok: true,
                status: 200,
                json: async () => ({
                  output: [
                    { type: 'reasoning', content: [] },
                    { type: 'message', content: [{ type: 'output_text', text: summary }] },
                  ],
                }),
              }),
            status: (code) => resolve({ ok: false, status: code }),
            boom: (e) => reject(e),
          })
        }),
    ),
  )
}

beforeEach(() => {
  pending = []
  boundary.endpoints.clear()
  boundary.endpoints.set(MODEL_A, URL_A)
  boundary.endpoints.set(MODEL_B, URL_B)
  boundary.endpoints.set(MODEL_HARMONY, URL_HARMONY)
  setActivePinia(createPinia())
  stubTransport()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

/** The `persist` compact.ts is handed - a spy, optionally doing what the real
 *  caller does with it (useChatStream.ts passes `chat.persistNow`). */
function persistSpy(then?: (c: Conversation) => void) {
  return vi.fn((c: Conversation) => {
    then?.(c)
  })
}

/** Start a compaction and assert its request is already on the wire. Returns a
 *  handle whose `settled` stays false until the compaction actually finishes,
 *  which is what proves a later mutation happened DURING the request, plus
 *  `req`/`sent`: this launch's own entry in `pending`, so a retry after an
 *  earlier request reads without index arithmetic. */
function launch(conv: Conversation, persist: ReturnType<typeof persistSpy> = persistSpy()) {
  const at = pending.length
  const h = {
    settled: false,
    persist,
    done: Promise.resolve(),
    get sent(): Pending {
      return pending[at]
    },
    get req(): SummaryRequest {
      return pending[at].req
    },
  }
  h.done = maybeCompact(conv, MAX_CTX, MAX_REPLY, persist).then(() => {
    h.settled = true
  })
  // Nothing before the fetch in maybeCompact awaits, so the request is out
  // synchronously - there is no window in which "the model changed" could mean
  // "changed before the request was built".
  expect(pending).toHaveLength(at + 1)
  expect(h.settled).toBe(false)
  return h
}

/** The provenance invariant, asserted on every committed summary: the model
 *  stamped on the conversation is the model the request was actually sent as. */
function expectStampMatchesRequest(conv: Conversation, req: SummaryRequest): void {
  expect(conv.summaryModel).toBe(req.model)
}

// ── the fixture itself ──────────────────────────────────────────────────────

describe('the fixture', () => {
  it('crosses the compaction threshold with the boundary on a known turn', () => {
    const f = fixture()
    expect(activeMessages(f.conv).map((m) => m.id)).toEqual([
      f.q1.id,
      f.laneA.id,
      f.laneB.id,
      f.q2.id,
      f.a2.id,
      f.q3.id,
      f.a3.id,
      f.q4.id,
    ])
    expect(compactionTarget(f.conv, MAX_CTX, MAX_REPLY)).toBe(5)
  })
})

// ── the unchanged-selection path, which must keep working ───────────────────

describe('maybeCompact with the selection standing still', () => {
  it('commits the summary and stamps the model that produced it', async () => {
    const f = fixture()
    const h = launch(f.conv)

    expect(h.sent.url).toBe(URL_A)
    expect(h.req.model).toBe(MODEL_A)
    // a summary is mechanical: no thinking budget, in this family's dialect
    expect(h.req.chat_template_kwargs).toEqual({ enable_thinking: false })
    expect(h.req.reasoning).toBeUndefined()

    h.sent.ok('what the user is after, in 400 words')
    await h.done

    expect(f.conv.summary).toBe('what the user is after, in 400 words')
    expect(f.conv.summaryCount).toBe(5)
    expect(f.conv.summaryLastId).toBe(f.a2.id)
    expectStampMatchesRequest(f.conv, h.req)
    expect(f.conv.summaryModel).toBe(MODEL_A)
    expect(summaryValid(f.conv)).toBe(true)
    expect(h.persist).toHaveBeenCalledTimes(1)
    expect(h.persist).toHaveBeenCalledWith(f.conv)
  })

  it('sends the asking model its OWN history - the other lane stays out', async () => {
    const f = fixture()
    const h = launch(f.conv)
    const input = h.req.input

    expect(input).toContain('MARK-LANE-A')
    expect(input).not.toContain('MARK-LANE-B')
    // the covered prefix, and only it
    expect(input).toContain('MARK-Q1')
    expect(input).toContain('MARK-Q2')
    expect(input).toContain('MARK-A2')
    expect(input).not.toContain('MARK-Q3')
    expect(input).not.toContain('MARK-A3')
    expect(input).not.toContain('MARK-Q4')

    h.sent.ok('brief')
    await h.done
  })

  it('asks a Harmony model in its own dialect', async () => {
    const f = fixture()
    f.conv.model = MODEL_HARMONY
    const h = launch(f.conv)

    expect(h.sent.url).toBe(URL_HARMONY)
    expect(h.req.model).toBe(MODEL_HARMONY)
    expect(h.req.reasoning).toEqual({ effort: 'low' })
    expect(h.req.chat_template_kwargs).toBeUndefined()
    // neither lane is this model's own history, so neither rides
    expect(h.req.input).not.toContain('MARK-LANE-A')
    expect(h.req.input).not.toContain('MARK-LANE-B')

    h.sent.ok('brief')
    await h.done
    expectStampMatchesRequest(f.conv, h.req)
  })
})

// ── the race ────────────────────────────────────────────────────────────────

describe('a selection that moves while the summary is in flight', () => {
  it('DISCARDS the result: A wrote it, and B must not be handed it under its own name', async () => {
    const f = fixture()
    const h = launch(f.conv)
    const rawBefore = JSON.stringify(f.conv.messages)

    // the request went out as A, carrying A's own history
    expect(h.req.model).toBe(MODEL_A)
    expect(h.req.input).toContain('MARK-LANE-A')
    expect(h.req.input).not.toContain('MARK-LANE-B')

    f.conv.model = MODEL_B
    expect(h.settled).toBe(false) // still pending - the change landed mid-request

    h.sent.ok('a brief about A-lane history')
    await h.done

    expect(f.conv.summary).toBeUndefined()
    expect(f.conv.summaryCount).toBeUndefined()
    expect(f.conv.summaryLastId).toBeUndefined()
    expect(f.conv.summaryModel).toBeUndefined()
    expect(summaryValid(f.conv)).toBe(false)
    expect(h.persist).not.toHaveBeenCalled()
    // and the raw thread is exactly what it was
    expect(JSON.stringify(f.conv.messages)).toBe(rawBefore)
  })

  it('never stamps a model that did not write the summary', async () => {
    const f = fixture()
    const h = launch(f.conv)
    f.conv.model = MODEL_B
    h.sent.ok('a brief about A-lane history')
    await h.done
    // the only two honest outcomes are "no summary" or "stamped A"; "stamped B"
    // is the bug, and it is the one thing asserted against directly
    expect(f.conv.summaryModel).not.toBe(MODEL_B)
  })

  it('discards through the REAL store edit the model picker uses', async () => {
    const f = fixture()
    const chat = useChatStore()
    // the chat is in the list and its document is the loaded one - the state
    // `edit` needs, reached without the localStorage-backed selection helpers
    chat.conversations.push(f.conv)
    boundary.api.getConversation.mockResolvedValue(f.conv)
    boundary.api.putConversation.mockResolvedValue(undefined)

    // the persist callback the real caller injects (useChatStream.ts)
    const h = launch(
      f.conv,
      persistSpy((c) => void chat.persistNow(c)),
    )
    expect(h.req.model).toBe(MODEL_A)

    // verbatim what lib/select-model.ts `selectStudioModel` applies
    await chat.edit(f.conv.id, (x) => {
      x.model = MODEL_B
      x.compareModels = undefined
    })

    // the mutation really landed on the document the compactor is holding...
    expect(f.conv.model).toBe(MODEL_B)
    // ...and the pick was saved, so this is the real pathway, not a poke
    expect(boundary.api.putConversation).toHaveBeenCalledTimes(1)
    // ...while the summary request was still out
    expect(h.settled).toBe(false)

    h.sent.ok('a brief about A-lane history')
    await h.done

    expect(f.conv.summary).toBeUndefined()
    expect(f.conv.summaryModel).toBeUndefined()
    expect(h.persist).not.toHaveBeenCalled()
    expect(boundary.api.putConversation).toHaveBeenCalledTimes(1) // nothing extra written
  })

  it('is a no-write, not a reset: an earlier valid summary survives untouched', async () => {
    const f = fixture()
    // a summary A already wrote, covering the first three turns
    f.conv.summary = "A's earlier brief"
    f.conv.summaryCount = 3
    f.conv.summaryLastId = f.laneB.id
    f.conv.summaryModel = MODEL_A
    expect(summaryValid(f.conv)).toBe(true)
    // there is still growth to fold in, so this rolls the summary forward
    expect(compactionTarget(f.conv, MAX_CTX, MAX_REPLY)).toBe(5)

    const h = launch(f.conv)
    // the roll-forward: the prior summary rides instead of its covered turns
    expect(h.req.input).toContain("A's earlier brief")
    expect(h.req.input).not.toContain('MARK-Q1')
    expect(h.req.input).toContain('MARK-Q2')

    f.conv.model = MODEL_B
    h.sent.ok('a rolled-forward brief about A-lane history')
    await h.done

    // the discarded result changed nothing - it did not land, and it did not
    // wipe what was already there
    expect(f.conv.summary).toBe("A's earlier brief")
    expect(f.conv.summaryCount).toBe(3)
    expect(f.conv.summaryLastId).toBe(f.laneB.id)
    expect(f.conv.summaryModel).toBe(MODEL_A)
    expect(summaryValid(f.conv)).toBe(true)
    expect(h.persist).not.toHaveBeenCalled()
  })

  it('leaves the next compaction free to succeed on the new model', async () => {
    const f = fixture()
    const persist = persistSpy()
    const first = launch(f.conv, persist)
    f.conv.model = MODEL_B
    first.sent.ok('a brief about A-lane history')
    await first.done
    expect(f.conv.summary).toBeUndefined()

    // the inflight guard was released in `finally`, and nothing was persisted,
    // so the thread still needs the same coverage - now for B
    const second = launch(f.conv, persist)
    expect(second.sent.url).toBe(URL_B)
    expect(second.req.model).toBe(MODEL_B)
    expect(second.req.input).toContain('MARK-LANE-B')
    expect(second.req.input).not.toContain('MARK-LANE-A')

    second.sent.ok('a brief about B-lane history')
    await second.done

    expect(f.conv.summary).toBe('a brief about B-lane history')
    expect(f.conv.summaryCount).toBe(5)
    expect(f.conv.summaryLastId).toBe(f.a2.id)
    expectStampMatchesRequest(f.conv, second.req)
    expect(f.conv.summaryModel).toBe(MODEL_B)
    expect(persist).toHaveBeenCalledTimes(1)
  })
})

// ── the prefix-boundary check this must not disturb ─────────────────────────

describe('the covered prefix moving while the summary is in flight', () => {
  it('discards when the branch under the coverage changes', async () => {
    const f = fixture()
    // an alternative answer to q2 exists off-screen (a regenerate). Off the
    // active path, so it changes neither the plan nor the coverage.
    const alt = msg('assistant', 'MARK-A2-ALT', f.q2.id, { model: MODEL_A })
    f.conv.messages.push(alt)
    expect(compactionTarget(f.conv, MAX_CTX, MAX_REPLY)).toBe(5)

    const h = launch(f.conv)

    // the `< 2/2 >` switcher under that turn - chat store `selectSibling`
    expect(stepSibling(f.conv, f.a2.id, 1)).toBe(true)
    expect(activeMessages(f.conv)[4].id).toBe(alt.id)
    expect(h.settled).toBe(false)

    h.sent.ok('a brief about a prefix that moved')
    await h.done

    expect(f.conv.summary).toBeUndefined()
    expect(f.conv.summaryModel).toBeUndefined()
    expect(h.persist).not.toHaveBeenCalled()
  })

  it('accepts harmless growth past the coverage', async () => {
    const f = fixture()
    // a later turn under the tail, brought on screen mid-request
    const q5 = msg('user', 'MARK-Q5', f.q4.id)
    f.conv.messages.push(q5)

    const h = launch(f.conv)
    focusStep(f.conv, q5.id)
    expect(activeMessages(f.conv)).toHaveLength(9)
    expect(activeMessages(f.conv)[4].id).toBe(f.a2.id) // prefix untouched
    expect(h.settled).toBe(false)

    h.sent.ok('a brief that is still anchored')
    await h.done

    expect(f.conv.summary).toBe('a brief that is still anchored')
    expect(f.conv.summaryCount).toBe(5)
    expect(f.conv.summaryLastId).toBe(f.a2.id)
    expectStampMatchesRequest(f.conv, h.req)
    expect(h.persist).toHaveBeenCalledTimes(1)
  })
})

// ── negative controls: the discard must be narrow ───────────────────────────

describe('what must NOT be discarded', () => {
  it('a selection re-applied to the same model (picking the model already in use)', async () => {
    const f = fixture()
    const h = launch(f.conv)
    // a fresh string with the same value - an identity check would fail here
    f.conv.model = [MODEL_A].join('')
    expect(f.conv.model).toBe(MODEL_A)

    h.sent.ok('brief')
    await h.done

    expect(f.conv.summary).toBe('brief')
    expectStampMatchesRequest(f.conv, h.req)
    expect(h.persist).toHaveBeenCalledTimes(1)
  })

  it('an unrelated edit to the conversation (a rename)', async () => {
    const f = fixture()
    const h = launch(f.conv)
    f.conv.title = 'renamed while summarizing'

    h.sent.ok('brief')
    await h.done

    expect(f.conv.summary).toBe('brief')
    expect(f.conv.title).toBe('renamed while summarizing')
    expectStampMatchesRequest(f.conv, h.req)
  })

  it('a selection that moved BEFORE the request - that is simply a compaction for B', async () => {
    const f = fixture()
    f.conv.model = MODEL_B
    const h = launch(f.conv)

    expect(h.sent.url).toBe(URL_B)
    expect(h.req.model).toBe(MODEL_B)

    h.sent.ok('a brief about B-lane history')
    await h.done

    expect(f.conv.summary).toBe('a brief about B-lane history')
    expectStampMatchesRequest(f.conv, h.req)
    expect(f.conv.summaryModel).toBe(MODEL_B)
  })
})

// ── the inflight guard, on every exit ───────────────────────────────────────

describe('the one-in-flight-per-conversation guard', () => {
  it('turns a second compaction into a no-op while the first is out', async () => {
    const f = fixture()
    const h = launch(f.conv)
    await maybeCompact(f.conv, MAX_CTX, MAX_REPLY, persistSpy())
    expect(pending).toHaveLength(1)
    h.sent.ok('brief')
    await h.done
    expect(f.conv.summary).toBe('brief')
  })

  it('is released after a failed request, and the retry still stamps honestly', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const f = fixture()
    const first = launch(f.conv)
    first.sent.status(503)
    await first.done

    expect(f.conv.summary).toBeUndefined()
    expect(warn).toHaveBeenCalled() // a failure is reported, not swallowed

    const second = launch(f.conv)
    second.sent.ok('brief on the retry')
    await second.done

    expect(f.conv.summary).toBe('brief on the retry')
    expectStampMatchesRequest(f.conv, second.req)
  })

  it('is released when the model moved AND the transport then blew up', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const f = fixture()
    const first = launch(f.conv)
    f.conv.model = MODEL_B
    first.sent.boom(new Error('connection reset'))
    await first.done
    expect(warn).toHaveBeenCalled()

    const second = launch(f.conv)
    expect(second.req.model).toBe(MODEL_B)
    second.sent.ok('brief for B')
    await second.done
    expect(f.conv.summaryModel).toBe(MODEL_B)
  })
})
