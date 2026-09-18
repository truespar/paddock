// The REPLY BUDGET half of lib/tokens.ts: `windowRemaining`, which answers
// "how many output tokens may this request ask for".
//
// Both of its answers are CEILINGS, and a floor underneath a ceiling is not a
// floor - it is an override. `Math.max(512, ...)` on the outside asked a
// provider that publishes 128 for 512, and it invented 512 tokens of room in a
// window the prompt had already filled, which is the shape of the 400 the
// clamp inside it exists to prevent.
//
// Arithmetic in the expectations is spelled out rather than recomputed from
// the module's own constants: a test that derives its answer the same way the
// code does cannot notice the code changing.

import { describe, expect, it } from 'vitest'
import { REPLY_RESERVE, replyReserve, windowRemaining } from './tokens'

// The slack this module holds back, for reference in the sums below:
//   max(1024, round(maxCtx * 0.02))
//   4096 -> 1024   8192 -> 1024   32768 -> 1024
//   131072 -> 2621   1000000 -> 20000

describe('windowRemaining: a published output cap is a ceiling, not a suggestion', () => {
  it('asks for 1 token when 1 token is all the provider will emit', () => {
    expect(windowRemaining(131072, 1, 1)).toBe(1)
  })

  it('asks for 128 when the provider publishes 128', () => {
    expect(windowRemaining(131072, 1, 128)).toBe(128)
  })

  it('asks for 511 when the provider publishes 511', () => {
    expect(windowRemaining(131072, 1, 511)).toBe(511)
  })

  it('asks for 512 when the provider publishes 512', () => {
    expect(windowRemaining(131072, 1, 512)).toBe(512)
  })

  it('never exceeds the published cap, whatever the cap is', () => {
    for (const cap of [1, 2, 7, 64, 100, 128, 255, 384, 511, 512, 513, 1024, 4096, 65536]) {
      expect(windowRemaining(131072, 1, cap)).toBeLessThanOrEqual(cap)
    }
  })
})

describe('windowRemaining: a full window has nothing left to give', () => {
  it('is 0 when the prompt plus the slack exactly fills the window', () => {
    // 8192 window - 7168 prompt - 1024 slack = 0
    expect(windowRemaining(8192, 7168)).toBe(0)
  })

  it('is 0 - never negative, never invented - when the prompt overflows', () => {
    // 8192 - 9000 - 1024 = -1832
    expect(windowRemaining(8192, 9000)).toBe(0)
    // 4096 - 10004 - 1024 = -6932 (a 40k-character paste on a 4K window)
    expect(windowRemaining(4096, 10004)).toBe(0)
    expect(windowRemaining(131072, 500_000, 65536)).toBe(0)
  })

  it('is the one token actually left when one token is left', () => {
    // 8192 - 7167 - 1024 = 1
    expect(windowRemaining(8192, 7167)).toBe(1)
  })
})

describe('windowRemaining: the ordinary allowance is untouched', () => {
  it('is the window minus the prompt minus the slack', () => {
    // 32768 - 4000 - 1024
    expect(windowRemaining(32768, 4000)).toBe(27744)
    // 131072 - 1 - 2621
    expect(windowRemaining(131072, 1)).toBe(128450)
  })

  it('still takes the smaller of window and provider cap on a huge window', () => {
    // the 1M-context / 384k-output provider the outCap clamp was written for
    expect(windowRemaining(1_000_000, 9, 384_000)).toBe(384_000)
  })

  it('stays window-bound when the provider cap is the larger of the two', () => {
    expect(windowRemaining(32768, 4000, 200_000)).toBe(27744)
  })
})

describe('windowRemaining: an unknown window keeps the documented fallback', () => {
  it('reserves the default headroom when the window is not known yet', () => {
    expect(REPLY_RESERVE).toBe(4096)
    expect(windowRemaining(0, 5000)).toBe(REPLY_RESERVE)
    expect(windowRemaining(0, 0)).toBe(REPLY_RESERVE)
  })

  it('still respects a provider cap below that headroom', () => {
    expect(windowRemaining(0, 5000, 128)).toBe(128)
    expect(windowRemaining(0, 5000, 511)).toBe(511)
  })

  it('does not let a huge provider cap widen the unknown-window fallback', () => {
    expect(windowRemaining(0, 0, 999_999)).toBe(REPLY_RESERVE)
  })
})

describe('windowRemaining: negative controls', () => {
  // 32768 - 4000 - 1024, i.e. "the window decides" - none of these inputs is
  // a cap, so none of them may narrow OR widen the answer.
  const open = 27744

  it('treats a missing, zero, negative or non-finite cap as "nothing published"', () => {
    expect(windowRemaining(32768, 4000, undefined)).toBe(open)
    expect(windowRemaining(32768, 4000, 0)).toBe(open)
    expect(windowRemaining(32768, 4000, -5)).toBe(open)
    expect(windowRemaining(32768, 4000, Number.NaN)).toBe(open)
    expect(windowRemaining(32768, 4000, Number.POSITIVE_INFINITY)).toBe(open)
  })

  it('answers with a whole number of tokens', () => {
    expect(windowRemaining(32768, 4000, 511.9)).toBe(511)
    expect(Number.isInteger(windowRemaining(32768, 4000))).toBe(true)
    expect(Number.isInteger(windowRemaining(0, 0, 128.5))).toBe(true)
    expect(Number.isInteger(windowRemaining(8192, 9000))).toBe(true)
  })

  it('leaves what compaction sets aside for the reply alone', () => {
    // replyReserve is the OTHER knob - it sizes the PROMPT plan, not the ask -
    // and the ceilings above must not reach into it.
    expect(replyReserve(null)).toBe(REPLY_RESERVE)
    expect(replyReserve(128)).toBe(128)
    expect(replyReserve(0)).toBe(REPLY_RESERVE)
    expect(replyReserve(-1)).toBe(REPLY_RESERVE)
  })
})
