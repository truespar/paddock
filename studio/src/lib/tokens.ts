// Client-side token accounting for the context gauge + pre-send trim guardrail.
// The server does not auto-truncate (Responses `truncation:"auto"` is rejected),
// so the studio must keep each prompt within the context window itself.
//
// Counts are ESTIMATES (~4 chars/token) - good enough for a fill gauge and for
// a conservative sliding window. A healthy margin absorbs the estimation error;
// the exact size of the last turn comes back in usage.input_tokens.

import type { Conversation, Message } from '@/types/chat'
import { messageText } from '@/types/chat'
import { activeMessages } from '@/lib/tree'

/** The turns this conversation would actually SEND: the branch on screen, not
 *  every branch it holds. Counting `conv.messages` would price abandoned
 *  branches into the context gauge and trim the wrong turns out of the prompt.
 *
 *  Every index in this module - `trimIndex`'s return, `promptTokensFrom`'s
 *  `from`, `summaryCount` - is an index into this array, and the same is true
 *  of the send path and the thread divider that mirror it. */
function thread(conv: Conversation): Message[] {
  return activeMessages(conv)
}

const CHARS_PER_TOKEN = 4
const PER_MESSAGE_OVERHEAD = 4 // role + delimiter tokens per message
const IMAGE_TOKENS = 512 // rough per-image vision cost
const SAFETY_MARGIN = 1024 // slack so an under-estimate never overflows the server

/** Headroom kept for the reply when the user has not capped it ("Model
 *  maximum"). The cap and this reserve used to be one number, so raising the
 *  ceiling shrank the usable prompt one-for-one and lowering it truncated long
 *  answers - one knob doing two jobs. They are separate now:
 *  an explicit cap is a promise we make room for, while "model maximum" only
 *  needs enough slack that compaction is not planning against a whole window
 *  the reply will almost never use. */
export const REPLY_RESERVE = 4096

/** What to set aside for the reply when planning the prompt. An explicit cap
 *  reserves exactly itself; "model maximum" reserves the default headroom. */
export function replyReserve(cap: number | null): number {
  return cap != null && cap > 0 ? cap : REPLY_RESERVE
}

/** The reply cap for a request whose prompt is `promptTokens` long: everything
 *  the window has left, but never more than the model will actually emit.
 *
 *  The window half is what "model maximum" means for a local model - a GGUF
 *  has no output limit of its own, the context is the ceiling - and it beats a
 *  fixed number because it is never larger than what actually fits.
 *
 *  `outCap` is the provider's own reply ceiling (cloud models publish one; see
 *  `models.outCapFor`) and it is a different number from the window, usually
 *  far smaller. Without it the window half alone is dangerous on a big-context
 *  model: a 1M-context provider was asked for 1047543 output tokens against a
 *  prompt our estimator put at 9 and the provider's tokenizer put at 2134
 *  (tool schemas, which no client-side estimate can see), so input + output
 * crossed the window and the send died on a 400. Taking
 *  the smaller of the two makes the estimator's error harmless: 384k of output
 *  on a 1M window cannot overflow whatever the prompt turns out to be. */
export function windowRemaining(maxCtx: number, promptTokens: number, outCap?: number): number {
  const byModel = outCap && outCap > 0 ? outCap : Infinity
  if (!maxCtx) return Math.min(REPLY_RESERVE, byModel)
  return Math.max(512, Math.min(maxCtx - promptTokens - windowSlack(maxCtx), byModel))
}

/** Slack between "everything the window has left" and what we actually ask
 *  for. The flat [`SAFETY_MARGIN`] is the floor; past ~50K of window it grows
 *  with the window instead.
 *
 *  Flat 1024 was too thin at the top end and free to widen there: on a 1M
 *  window nobody is losing a reply to 20K of held-back budget, while the
 *  estimator's blind spots (tool schemas above all - 1632 real tokens the
 *  client counted as zero) do not shrink just because the window is huge. A
 *  model that publishes an `outCap` is already safe by the clamp above; this
 *  covers the ones that do not, and picks enabled before we stored it. */
function windowSlack(maxCtx: number): number {
  return Math.max(SAFETY_MARGIN, Math.round(maxCtx * 0.02))
}

/** Rough size of what a send will carry, for windowRemaining. Same estimator
 *  the trimmer uses, so the two agree.
 *
 *  `summary` is the compaction summary this send will INJECT - a
 *  `ContextPlan`'s own, never the stored one, because a summary held in
 *  reserve is not in the prompt and costs nothing. It is charged with its
 *  wrapper (`summaryBlock`, the one owner of that wording), once, and the
 *  messages it stands in for are already excluded by `from`.
 *
 *  Date/graph/tool content remains outside this estimate. Charging this
 *  known block reduces the omitted input; it does not provide exact tokenizer
 *  accounting or prove that a provider would otherwise reject the request. */
export function promptTokensFrom(conv: Conversation, from: number, summary?: string): number {
  let t = conv.systemPrompt ? estimateTokens(conv.systemPrompt) + PER_MESSAGE_OVERHEAD : 0
  // Keep the existing estimator's conservative per-block overhead.
  if (summary) t += estimateTokens(summaryBlock(summary)) + PER_MESSAGE_OVERHEAD
  const msgs = thread(conv)
  for (let i = Math.max(0, from); i < msgs.length; i++) t += messageTokens(msgs[i])
  return t
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN)
}

function messageTokens(m: Message): number {
  let t = estimateTokens(messageText(m)) + PER_MESSAGE_OVERHEAD
  for (const p of m.content) if (p.type === 'image') t += IMAGE_TOKENS
  return t
}

/**
 * Best estimate of the whole thread's prompt size (system + every message +
 * optional composer draft). Calibrated by the last real usage when present:
 * an assistant turn's usage.promptTokens is the exact prompt the server saw,
 * so we anchor on it and only estimate anything newer.
 */
export function contextTokens(conv: Conversation | null | undefined, draft = ''): number {
  if (!conv) return draft ? estimateTokens(draft) : 0
  const msgs = thread(conv)
  const draftCost = draft ? estimateTokens(draft) + PER_MESSAGE_OVERHEAD : 0

  for (let i = msgs.length - 1; i >= 0; i--) {
    const u = msgs[i].usage
    if (msgs[i].role === 'assistant' && u?.promptTokens) {
      // real prompt + the re-sent answer (reasoning is not re-sent), plus any
      // messages newer than this turn, plus the draft.
      let t = u.promptTokens + (u.completionTokens ?? 0) + PER_MESSAGE_OVERHEAD
      for (let j = i + 1; j < msgs.length; j++) t += messageTokens(msgs[j])
      return t + draftCost
    }
  }

  let t = conv.systemPrompt ? estimateTokens(conv.systemPrompt) + PER_MESSAGE_OVERHEAD : 0
  for (const m of msgs) t += messageTokens(m)
  return t + draftCost
}

/**
 * The index of the first message to INCLUDE in the prompt so that
 * prompt + reply-cap fit the context. Messages before it are dropped (sliding
 * window keeping the most recent; the system prompt is always kept). Returns 0
 * when nothing needs trimming or limits are unknown. `reserve` sets aside
 * budget for extra prompt content (the injected summary).
 */
export function trimIndex(conv: Conversation, maxCtx: number, maxReply: number, reserve = 0): number {
  if (!maxCtx) return 0
  const budget = maxCtx - maxReply - SAFETY_MARGIN - reserve
  if (budget <= 0) return 0

  const sys = conv.systemPrompt ? estimateTokens(conv.systemPrompt) + PER_MESSAGE_OVERHEAD : 0
  const msgs = thread(conv)
  let used = sys
  let first = msgs.length ? msgs.length - 1 : 0

  for (let j = msgs.length - 1; j >= 0; j--) {
    const cost = messageTokens(msgs[j])
    // always keep the most recent message even if it alone is huge
    if (j < msgs.length - 1 && used + cost > budget) break
    used += cost
    first = j
  }
  return first
}

// ── context compaction (summarize-instead-of-drop) ──────────────────────────

/** Compact once the thread crosses this fraction of the prompt budget. */
const COMPACT_AT = 0.7
/** After compacting, keep roughly this fraction of the budget as raw recent
 *  messages; everything older folds into the summary. */
const KEEP_TAIL = 0.35

/** The stored summary still matches the thread: its boundary message is where
 *  it was when the summary was written. */
export function summaryValid(conv: Conversation): boolean {
  return !!(
    conv.summary &&
    conv.summaryCount &&
    thread(conv)[conv.summaryCount - 1]?.id === conv.summaryLastId
  )
}

/** The summary exactly as it rides in the prompt: the send path puts this
 *  string in `instructions` and `promptTokensFrom` charges this string, so
 *  ONE owner of the wording. Two owners is how the budget came to ignore text
 *  the request had been carrying all along. */
export function summaryBlock(summary: string): string {
  return `Summary of the earlier part of this conversation (older messages were compacted):\n${summary}`
}

/** What the next prompt should contain: raw messages from `from` on, preceded
 *  by `summary` when one applies. Falls back to the plain sliding window when
 *  summaries are off, absent, or stale. */
export interface ContextPlan {
  from: number
  summary?: string
}

export function planContext(
  conv: Conversation,
  maxCtx: number,
  maxReply: number,
  useSummary: boolean,
): ContextPlan {
  if (!useSummary || !summaryValid(conv)) {
    return { from: trimIndex(conv, maxCtx, maxReply) }
  }
  const summary = conv.summary as string
  const covered = conv.summaryCount as number
  // Nothing needs to give way yet -> send everything raw (the summary stays in
  // reserve until the window actually forces a choice).
  if (trimIndex(conv, maxCtx, maxReply) === 0) return { from: 0 }
  const reserve = estimateTokens(summary) + PER_MESSAGE_OVERHEAD
  const from = trimIndex(conv, maxCtx, maxReply, reserve)
  // The summary replaces its covered prefix. If even that doesn't fit, raw
  // messages past the coverage still drop off (and the thread divider says so).
  return { from: Math.max(from, covered), summary }
}

// ── server-side compaction (local lanes send context_management) ────────────

/** The stored compaction item still matches the thread: its tail-start
 *  message (the newest user message of the compacted request) is still
 *  present. Same anchor-by-id safety as `summaryValid`. */
export function serverCompactionValid(conv: Conversation): boolean {
  const sc = conv.serverCompaction
  return !!sc && thread(conv).some((m) => m.id === sc.tailStartId)
}

/** The `compact_threshold` a local lane arms `context_management` with:
 *  the same 70%-of-budget trigger the client-side compactor uses, but in the
 *  server's exact rendered tokens. Well under the window deliberately - the
 *  summarization pass reads the whole prompt, so compaction must fire while
 *  everything still fits. 0 = the window is too small to manage (caller
 *  falls back to the client plan). */
export function serverCompactThreshold(maxCtx: number, maxReply: number): number {
  const budget = maxCtx - maxReply - SAFETY_MARGIN
  if (budget <= 0) return 0
  return Math.max(512, Math.floor(budget * COMPACT_AT))
}

/** How many leading messages the next compaction should cover. 0 = the thread
 *  hasn't crossed the threshold (or there's nothing new to fold in). */
export function compactionTarget(conv: Conversation, maxCtx: number, maxReply: number): number {
  if (!maxCtx || thread(conv).length < 4) return 0
  const budget = maxCtx - maxReply - SAFETY_MARGIN
  if (budget <= 0) return 0
  if (contextTokens(conv) < budget * COMPACT_AT) return 0

  // Keep the newest messages that fit the tail allowance; cover the rest.
  const msgs = thread(conv)
  let used = 0
  let keepFrom = msgs.length
  for (let j = msgs.length - 1; j >= 0; j--) {
    used += messageTokens(msgs[j])
    if (used > budget * KEEP_TAIL && keepFrom < msgs.length) break
    keepFrom = j
  }
  // Always keep the latest exchange raw; only report growth over what the
  // current summary already covers.
  const target = Math.min(keepFrom, msgs.length - 2)
  const existing = summaryValid(conv) ? (conv.summaryCount as number) : 0
  return target > existing ? target : 0
}
