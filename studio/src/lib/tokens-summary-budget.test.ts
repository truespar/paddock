// Tests for the SUMMARY half of the prompt budget (lib/tokens.ts).
//
// A compacted send carries text the estimator never saw: the send path puts
// `Summary of the earlier part of this conversation ...` + the summary into
// `instructions`, while the reply cap was planned from `promptTokensFrom`,
// which counted the system prompt and the raw tail only. The prompt is then
// bigger than the number `windowRemaining` subtracted from the window, so
// "model maximum" asks for a reply that cannot fit beside it - the same class
// of overflow the outCap clamp in `windowRemaining` exists to absorb.
//
// The injected text is KNOWN at plan time, so it is charged: once, wrapper
// included, and never together with the raw messages it replaced.
//
// Out of scope here (and still uncounted by this estimator): the date line,
// graph grounding, tool schemas and image bytes. This is an estimate, not a
// tokenizer. `windowRemaining`'s own 512 floor is also deliberately left
// alone - every window below uses numbers far away from it.

import { describe, expect, it } from 'vitest'
import type { Conversation, Message } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import {
  contextTokens,
  estimateTokens,
  planContext,
  promptTokensFrom,
  windowRemaining,
} from './tokens'

// ── fixtures ────────────────────────────────────────────────────────────────
//
// Synthetic and round on purpose: every expected number below is arithmetic
// anyone can redo by hand. The estimator is 4 chars/token with 4 tokens of
// per-message overhead (lib/tokens.ts).

/** 8000 chars = 2000 tokens, + 4 overhead = 2004 per message. */
const CHUNK = 'm'.repeat(8000)
const MSG_TOKENS = 2004
/** 40 chars = 10 tokens, + 4 overhead = 14. */
const SYS = 's'.repeat(40)
const SYS_TOKENS = 14
/** 400 chars = 100 tokens of summary. */
const SUMMARY = 'z'.repeat(400)

/** The test's own copy of the wrapper the send path injects. Deliberately a
 *  copy: if the builder's text and the estimator's charge ever stop being the
 *  same string, one of these two files fails. */
const WRAPPER = 'Summary of the earlier part of this conversation (older messages were compacted):\n'
/** (82 wrapper chars + 400 summary chars) / 4 = 120.5 -> 121, + 4 overhead. */
const BLOCK_TOKENS = 125

function thread(n: number): Message[] {
  const out: Message[] = []
  for (let i = 0; i < n; i++) {
    out.push({
      id: `m${i}`,
      parentId: i === 0 ? null : `m${i - 1}`,
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: [{ type: 'text', text: CHUNK }],
      createdAt: i,
    })
  }
  return out
}

/** A ten-turn linear chat with a valid stored summary over its first five. */
function conv(overrides: Partial<Conversation> = {}): Conversation {
  const messages = thread(10)
  return {
    id: 'c1',
    title: 'budget',
    messages,
    leafId: messages[messages.length - 1].id,
    model: 'local/test',
    systemPrompt: SYS,
    params: { ...DEFAULT_PARAMS },
    summary: SUMMARY,
    summaryCount: 5,
    summaryLastId: 'm4',
    summaryModel: 'local/test',
    createdAt: 0,
    updatedAt: 0,
    ...overrides,
  }
}

describe('promptTokensFrom: the injected summary', () => {
  it('charges the wrapper and the summary once on top of the raw tail', () => {
    const c = conv()
    // 14 system + 6 messages (indices 4..9) x 2004
    expect(promptTokensFrom(c, 4)).toBe(SYS_TOKENS + 6 * MSG_TOKENS)
    expect(promptTokensFrom(c, 4, SUMMARY)).toBe(SYS_TOKENS + 6 * MSG_TOKENS + BLOCK_TOKENS)
  })

  it('charges exactly what the wrapper + summary estimate to', () => {
    const c = conv()
    const charged = promptTokensFrom(c, 4, SUMMARY) - promptTokensFrom(c, 4)
    expect(charged).toBe(BLOCK_TOKENS)
    // the same number the whole injected block estimates to, +1 message of
    // role/delimiter overhead (it rides as part of the single system message)
    expect(charged).toBe(estimateTokens(WRAPPER + SUMMARY) + 4)
  })

  it('charges it once however many raw messages remain', () => {
    const c = conv()
    const atFour = promptTokensFrom(c, 4, SUMMARY) - promptTokensFrom(c, 4)
    const atSix = promptTokensFrom(c, 6, SUMMARY) - promptTokensFrom(c, 6)
    expect(atFour).toBe(BLOCK_TOKENS)
    expect(atSix).toBe(BLOCK_TOKENS)
    // and the two differ by the two dropped messages, nothing else
    expect(promptTokensFrom(c, 4, SUMMARY) - promptTokensFrom(c, 6, SUMMARY)).toBe(2 * MSG_TOKENS)
  })

  it('excludes the raw messages the summary replaced', () => {
    const c = conv()
    const whole = promptTokensFrom(c, 0) // every turn raw, no summary
    const replaced = 5 * MSG_TOKENS // messages 0..4, the summary's coverage
    expect(promptTokensFrom(c, 5, SUMMARY)).toBe(whole - replaced + BLOCK_TOKENS)
    // and a compacted prompt is the smaller of the two: that is the point
    expect(promptTokensFrom(c, 5, SUMMARY)).toBeLessThan(whole)
  })

  // ── negative controls ────────────────────────────────────────────────────

  it('charges nothing when no summary was planned', () => {
    const c = conv()
    expect(promptTokensFrom(c, 4, undefined)).toBe(promptTokensFrom(c, 4))
    expect(promptTokensFrom(c, 4)).toBe(SYS_TOKENS + 6 * MSG_TOKENS)
  })

  it('charges nothing for an empty summary, which the builder also drops', () => {
    // buildBody injects `plan.summary ? block : ''`, so an empty summary puts
    // no text in the prompt and must cost no budget either.
    const c = conv()
    expect(promptTokensFrom(c, 4, '')).toBe(promptTokensFrom(c, 4))
  })

  it('leaves the context gauge alone: it prices the thread, not a send', () => {
    // contextTokens answers "how full is this chat", which is every turn the
    // conversation holds - a stored summary neither adds to it nor replaces
    // anything in it. Nothing here moved, and nothing here should.
    const c = conv()
    expect(contextTokens(c)).toBe(SYS_TOKENS + 10 * MSG_TOKENS)
    expect(contextTokens(conv({ summary: undefined, summaryCount: undefined }))).toBe(
      contextTokens(c),
    )
  })

  it('leaves the no-system-prompt and empty-thread cases where they were', () => {
    const bare = conv({ systemPrompt: '' })
    expect(promptTokensFrom(bare, 4)).toBe(6 * MSG_TOKENS)
    expect(promptTokensFrom(bare, 4, SUMMARY)).toBe(6 * MSG_TOKENS + BLOCK_TOKENS)
    const empty = conv({ messages: [], leafId: undefined })
    expect(promptTokensFrom(empty, 0)).toBe(SYS_TOKENS)
    expect(promptTokensFrom(empty, 0, SUMMARY)).toBe(SYS_TOKENS + BLOCK_TOKENS)
  })
})

describe('windowRemaining over a compacted prompt', () => {
  // 64000-token window: slack is max(1024, 2%) = 1280, and everything below
  // lands tens of thousands of tokens clear of the 512 floor.
  const WINDOW = 64_000
  const SLACK = 1280

  it('leaves less room for the reply once the summary is charged', () => {
    const c = conv()
    const prompt = promptTokensFrom(c, 4)
    const compacted = promptTokensFrom(c, 4, SUMMARY)
    expect(windowRemaining(WINDOW, prompt)).toBe(WINDOW - prompt - SLACK)
    expect(windowRemaining(WINDOW, compacted)).toBe(WINDOW - prompt - SLACK - BLOCK_TOKENS)
    expect(windowRemaining(WINDOW, compacted)).toBeLessThan(windowRemaining(WINDOW, prompt))
  })

  it('still yields to a provider reply ceiling, summary or not', () => {
    const c = conv()
    const outCap = 2048
    expect(windowRemaining(WINDOW, promptTokensFrom(c, 4), outCap)).toBe(outCap)
    expect(windowRemaining(WINDOW, promptTokensFrom(c, 4, SUMMARY), outCap)).toBe(outCap)
  })
})

describe('planContext: what the estimator is being asked to charge', () => {
  const MAX_REPLY = 4096

  it('applies the summary from its coverage on, so the prefix is not sent twice', () => {
    const c = conv()
    const plan = planContext(c, 16_000, MAX_REPLY, true)
    expect(plan.summary).toBe(SUMMARY)
    // the window fits 5 raw messages here, and the summary covers 5
    expect(plan.from).toBe(5)
    expect(plan.from).toBeGreaterThanOrEqual(c.summaryCount as number)
  })

  it('plans no summary when compaction is off', () => {
    const plan = planContext(conv(), 16_000, MAX_REPLY, false)
    expect(plan.summary).toBeUndefined()
    expect(plan.from).toBe(5)
  })

  it('plans no summary when the stored one no longer matches the thread', () => {
    const plan = planContext(conv({ summaryLastId: 'gone' }), 16_000, MAX_REPLY, true)
    expect(plan.summary).toBeUndefined()
    expect(plan.from).toBe(5)
  })

  it('holds the summary in reserve while the whole thread still fits', () => {
    const plan = planContext(conv(), 200_000, MAX_REPLY, true)
    expect(plan).toEqual({ from: 0 })
  })
})
