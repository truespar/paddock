// Background context compaction: once a chat crosses ~70% of the prompt
// budget, the oldest messages are summarized by the loaded model so the next
// prompts carry a summary instead of silently dropping them (the pre-compaction
// behavior, still the fallback whenever this is off or fails). Runs after a
// turn completes, never on the send path - a send never waits on it.
//
// The raw messages are never touched; only what gets SENT changes. The summary
// + its coverage live on the conversation doc (see Conversation.summary*).

import type { Conversation, Message } from '@/types/chat'
import { messageText } from '@/types/chat'
import { isHarmony } from '@/lib/model-caps'
import { compactionTarget, summaryValid } from '@/lib/tokens'
import { activeMessages } from '@/lib/tree'
import { useModelsStore } from '@/stores/models'

const SUMMARY_MAX_TOKENS = 640

/** Window held back from the transcript beyond the summary itself: the
 *  instructions, the chat template, and the slack the ~4-chars/token estimate
 *  needs. Named because the floor it implies - a window has to beat
 *  SUMMARY_MAX_TOKENS + this before ANY transcript fits - is reportable. */
const REQUEST_HEADROOM = 2048

const INSTRUCTIONS =
  'Summarize the conversation transcript below into a compact brief for continuing ' +
  'the same conversation later. Keep: what the user is trying to do, decisions and ' +
  'facts established, names/numbers/identifiers, open questions, and the current ' +
  'state. Write plain prose, no preamble, at most 400 words.'

/** One line per message, text only. Attachments and tool calls are noted, not
 *  inlined - a summary can't carry image bytes. Own-history-only, same rule
 *  as the send path: other models' compare-lane answers stay out - a summary
 *  must not smuggle a losing lane back into the winner's context. */
function renderTranscript(messages: Message[], model: string): string {
  const lines: string[] = []
  for (const m of messages) {
    if (m.role === 'assistant' && m.group && m.model !== model) continue
    const notes: string[] = []
    for (const p of m.content) {
      if (p.type === 'image') notes.push('[image attached]')
      if (p.type === 'file') notes.push(`[file: ${p.name}]`)
      // A mic recording has no name; say what it was rather than nothing.
      if (p.type === 'audio') notes.push(`[audio: ${p.name || 'recording'}]`)
    }
    for (const c of m.toolCalls ?? []) notes.push(`[used tool ${c.name}]`)
    const text = messageText(m).trim()
    if (!text && notes.length === 0) continue
    lines.push(`${m.role === 'user' ? 'User' : 'Assistant'}: ${[...notes, text].join(' ').trim()}`)
  }
  return lines.join('\n\n')
}

// One compaction in flight per conversation.
const inflight = new Set<string>()

/**
 * Summarize the conversation's oldest messages if it has outgrown the
 * threshold. Fire-and-forget: failures are logged and simply retried after a
 * later turn. `persist` is injected so this module doesn't pull in the store.
 */
export async function maybeCompact(
  conv: Conversation,
  maxCtx: number,
  maxReply: number,
  persist: (c: Conversation) => void,
): Promise<void> {
  const count = compactionTarget(conv, maxCtx, maxReply)
  if (count === 0 || inflight.has(conv.id)) return
  // The BRANCH on SCREEN, not every branch stored: `count` comes from
  // compactionTarget, which counts the same path, and a summary anchored to
  // the wrong array would cover turns this thread never sends.
  const covered = activeMessages(conv).slice(0, count)
  const lastId = covered[covered.length - 1]?.id
  if (!lastId) return

  // Roll the existing summary forward: it covers a prefix of `covered`, so the
  // model folds it in rather than re-reading those messages.
  const prior = summaryValid(conv) ? (conv.summaryCount as number) : 0
  const priorBlock = prior > 0 ? `Summary of the conversation so far:\n${conv.summary}\n\n` : ''
  const transcript = renderTranscript(covered.slice(prior), conv.model)
  if (!transcript && !priorBlock) return

  // The chunk must itself fit the window with room for the summary; oversize
  // (possible when many turns arrive between compactions) keeps the newest end.
  const capChars = (maxCtx - SUMMARY_MAX_TOKENS - REQUEST_HEADROOM) * 4
  // A window under the floor has no room for one line of transcript, and the
  // clamp this used to carry (`Math.max(0, ...)`) made that the WORST case
  // rather than a refused one: `slice(-0)` keeps the whole string, so the
  // budget nothing fits into was sent everything. compactionTarget only needs
  // `maxCtx - maxReply - 1024 > 0`, so it does ask on windows this small.
  // Say so and leave: the messages and any existing summary stand, nothing is
  // marked in flight, and a later turn on a bigger window compacts normally.
  if (capChars <= 0) {
    console.warn(
      `compaction skipped: a ${maxCtx}-token window has no room for a summary request ` +
        `(needs more than ${SUMMARY_MAX_TOKENS + REQUEST_HEADROOM})`,
    )
    return
  }
  const body = (priorBlock + transcript).slice(-capChars)

  inflight.add(conv.id)
  try {
    // Non-streaming Responses call - same API the main chat uses (system prompt
    // rides in `instructions`, the transcript in `input`). A summary is one-shot,
    // so we don't stream it.
    const req: Record<string, unknown> = {
      model: conv.model,
      instructions: INSTRUCTIONS,
      input: body,
      temperature: 0,
      stream: false,
      max_output_tokens: SUMMARY_MAX_TOKENS,
    }
    // Summaries are mechanical: no thinking budget. effort is a Harmony knob;
    // enable_thinking is Qwen's - the same split the model uses elsewhere.
    if (isHarmony(conv.model)) req.reasoning = { effort: 'low' }
    else req.chat_template_kwargs = { enable_thinking: false }

    // Same manager relay the main chat uses - keyed by the runner serving
    // this conversation's model.
    const endpoint = useModelsStore().responsesUrl(conv.model)
    if (!endpoint) throw new Error(`no running model serves ${conv.model}`)
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const out = (await res.json()) as {
      output?: { type: string; content?: { type: string; text?: string }[] }[]
    }
    const summary = out.output
      ?.find((o) => o.type === 'message')
      ?.content?.filter((c) => c.type === 'output_text')
      .map((c) => c.text ?? '')
      .join('')
      .trim()
    if (!summary) throw new Error('empty summary')

    // The thread may have grown while we summarized (that's fine - coverage is
    // anchored by id), but if the covered prefix itself changed, discard.
    if (activeMessages(conv)[count - 1]?.id !== lastId) return
    conv.summary = summary
    conv.summaryCount = count
    conv.summaryLastId = lastId
    conv.summaryModel = conv.model
    persist(conv)
  } catch (e) {
    console.warn('compaction failed (will retry after a later turn)', e)
  } finally {
    inflight.delete(conv.id)
  }
}
