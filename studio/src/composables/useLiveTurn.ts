// Speaking into the conversation  - the mic's second job.
//
// In a speech-model chat the composer is a control surface, not a sink: what
// you say belongs in the thread, not in a text box you then have to send. So
// the turn is created the moment recording starts and grows while you talk,
// one column per model, and the clip attaches when you stop.
//
// The TURN is the same ARTIFACT A DROPPED FILE PRODUCES, and that is the whole
// design rather than an implementation convenience. Since a closed utterance
// comes back with segments, word times and confidence, a live
// session can fill exactly the `TranscriptMeta` the file lane fills - so the
// player, click-a-word-to-seek, the confidence colouring, "heard differently"
// and srt/vtt export all work on it with no new renderer and no second-class
// shape. An utterance is a timed span of speech, which is what a segment is.
//
// What used to happen instead: the live text was a PREVIEW under the composer,
// and on stop the recording was sent through the file endpoint and transcribed
// a second time. You watched a transcript form and then a different one
// replaced it. That path is gone.
import { uuid } from '@/lib/uuid'
import { migrate } from '@/lib/tree'
import type {
  AudioPart,
  Conversation,
  Message,
  TranscriptGuard,
  TranscriptMeta,
  TranscriptSegment,
  TranscriptWord,
} from '@/types/chat'
import type { MicItem, MicLane, MicResult } from '@/composables/useMicTranscribe'
import { readAudioPart } from '@/lib/attachments'
import { useChatStore } from '@/stores/chat'

function uid(): string {
  return uuid()
}

/** The mean of what was scored; undefined where nothing was - a word the lane
 *  timed but could not score is not a zero. */
function mean(xs: number[]): number | undefined {
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : undefined
}

/** One lane's utterances as the transcript machinery wants them.
 *
 *  Times arrive UTTERANCE-LOCAL - `paddock_verbose` is the object the file
 *  endpoint would return for that utterance's audio on its own - and the item
 *  says where it sits in the session, so this is the one place the two are
 *  added together. Get it wrong and every seek in the turn is off by the
 *  length of the silences. */
function metaOf(lane: MicLane): TranscriptMeta {
  const segments: TranscriptSegment[] = []
  const words: TranscriptWord[] = []
  const guards: TranscriptGuard[] = []
  let end = 0

  for (const it of lane.items) {
    const at = (it.at ?? 0) / 1000
    const stop = at + (it.duration ?? 0)
    end = Math.max(end, stop)
    const timed: TranscriptWord[] = (it.words ?? []).map((w) => ({
      word: w.word,
      confidence: w.confidence,
      start: at + w.start,
      end: at + w.end,
    }))
    words.push(...timed)
    if (it.text) {
      segments.push({
        start: at,
        end: stop,
        text: it.text,
        // the span's own figure is the mean of its words, which is the same
        // shape the file lane's `exp(avg_logprob)` has - absent where the lane
        // could not answer per-word confidence at all
        confidence: mean(timed.flatMap((w) => (w.confidence === undefined ? [] : [w.confidence]))),
        words: timed.length ? timed : undefined,
      })
    }
    // A guard is what makes an empty utterance informative rather than a gap:
    // "the model answered this span with its no-speech marker" is a result.
    for (const g of it.guards ?? []) {
      guards.push({ start: at, end: stop, reason: g.reason, note: g.note, dropped: g.dropped })
    }
  }

  return {
    language: lane.language,
    durationS: end || undefined,
    segments: segments.length ? segments : undefined,
    words: words.length ? words : undefined,
    guards: guards.length ? guards : undefined,
  }
}

export function useLiveTurn() {
  const chat = useChatStore()

  let conv: Conversation | null = null
  let user: Message | null = null
  let turns: Message[] = []

  /** Open the turn. Returns false when there is nothing to open it in. */
  function begin(lanes: { port: number; model: string }[], language?: string): boolean {
    const c = chat.active
    if (!c || !lanes.length) return false
    conv = c
    // The clip does not exist yet - it is still being spoken - so the part
    // goes in without an id and is filled at stop. Empty rather than absent
    // because the turn is an audio turn from the first word, and that is what
    // makes every assistant message below a transcription rather than a chat
    // reply that happens to contain a transcript.
    user = chat.addMessage(c, {
      id: uid(),
      role: 'user',
      content: [
        { type: 'audio', attachmentId: '', mime: '', name: '', language } satisfies AudioPart,
      ],
      createdAt: Date.now(),
    })
    const gid = uid()
    // Every lane is a sibling under this user turn. addMessage's default
    // follows the current tip, which would nest lane 2 under lane 1 and hide
    // it when the active tree expands the comparison's sibling group.
    const parent = user.id
    turns = lanes.map((l) =>
      chat.addMessage(c, {
        id: uid(),
        role: 'assistant',
        content: [{ type: 'text', text: '' }],
        streaming: true,
        model: l.model,
        // grouped only when there is a comparison, matching what an ordinary
        // send does with one lane
        group: lanes.length > 1 ? gid : undefined,
        // Marked a transcription now rather than when it finishes, for the
        // reason the file lane marks it early too: the renderer keys on it, and
        // a growing transcript falling through the markdown path first both
        // reflows when it settles and lets a line starting "- " briefly render
        // as a bullet list.
        transcript: {},
        run: { model: l.model, params: { ...c.params, maxTokens: null }, tools: [], at: Date.now() },
        createdAt: Date.now(),
      }, parent),
    )
    return true
  }

  /** Push what the lanes have said so far into their turns. */
  function apply(lanes: MicLane[]): void {
    for (let i = 0; i < turns.length && i < lanes.length; i++) {
      const part = turns[i].content[0]
      if (part?.type === 'text') part.text = lanes[i].text
      turns[i].error = lanes[i].error
    }
  }

  /** Whether every lane put its item N in the same place.
   *
   *  It should be impossible for them not to: the runner's detector is a pure
   *  function of the sample stream and every lane is fed identical frames. But
   *  "should be impossible" is exactly the kind of claim a comparison must not
   *  rest on silently - if the columns are not describing the same audio, the
   *  comparison is not one, and saying so beats rendering it. */
  function misaligned(lanes: MicLane[]): boolean {
    if (lanes.length < 2) return false
    const n = Math.min(...lanes.map((l) => l.items.length))
    const at = (it: MicItem | undefined) => it?.at ?? null
    for (let i = 0; i < n; i++) {
      const first = at(lanes[0].items[i])
      if (first === null) continue
      if (lanes.some((l) => at(l.items[i]) !== null && Math.abs((at(l.items[i]) ?? 0) - first) > 1)) {
        return true
      }
    }
    return false
  }

  /** Close the turn: attach the clip, settle every lane's transcript. */
  async function finish(out: MicResult): Promise<void> {
    const c = conv
    if (!c) return
    apply(out.lanes)

    if (user && out.clip) {
      const part = user.content[0]
      if (part?.type === 'audio') {
        try {
          // Assigned into the existing part rather than replaced, so the
          // renderer's reference stays live and the player swaps in without the
          // whole message re-mounting.
          Object.assign(part, await readAudioPart(out.clip, c.id, part.language))
          // A MediaRecorder blob carries no duration in its header, so the
          // browser answers Infinity and `readAudioPart` honestly reports
          // nothing. The transport then has no scrubbable length, which is not
          // a cosmetic loss: it renders DISABLED, so the timeline cannot be
          // dragged at all. We know the answer - the last utterance ends where
          // the recording does - so fill it in rather than make the element
          // hunt for it.
          if (!part.durationS) {
            const ends = out.lanes.map((l) => {
              const last = l.items[l.items.length - 1]
              return last ? (last.at ?? 0) / 1000 + (last.duration ?? 0) : 0
            })
            const end = Math.max(0, ...ends)
            if (end > 0) part.durationS = end
          }
        } catch (e) {
          // The store refused it (over the 100 MB attachment cap, or simply
          // down). The TRANSCRIPT is the valuable half and it is already here,
          // so keep the turn and lose the player - throwing from here would
          // leave every lane stuck mid-stream forever, which is the one
          // outcome worse than no audio.
          const mb = Math.round(out.clip.size / (1024 * 1024))
          user.content = [
            {
              type: 'text',
              text: `The recording (${mb} MB) could not be stored, so this turn has no player: ${
                e instanceof Error ? e.message : String(e)
              }`,
            },
          ]
        }
      }
    } else if (user && !out.clip) {
      // Said, not swallowed: the transcript is real and the recording is not,
      // so the turn keeps its words and loses only its player.
      user.content = [
        { type: 'text', text: 'This browser could not record the audio, so the clip was not kept.' },
      ]
    }

    const skew = misaligned(out.lanes)
    for (let i = 0; i < turns.length; i++) {
      const lane = out.lanes[i]
      const t = turns[i]
      t.streaming = false
      if (!lane) continue
      // A lane that heard nothing is not a transcript: drop the marker so it
      // renders as the plain failure it is instead of an empty transcript with
      // a player, the same rule the file lane applies.
      t.transcript = lane.text || lane.items.length ? metaOf(lane) : undefined
      if (!lane.text && !lane.error) {
        t.error = 'This model produced no transcript for what you said.'
      }
      if (skew && !t.error) {
        t.error = concatSkew
      }
      // No `usage`: the realtime wire bills in duration rather than tokens (on
      // purpose - that is the honest unit for audio), and the wall-clock times
      // it could report are contaminated anyway, because N models comparing
      // live are sharing one GPU. A speed number that means nothing is worse
      // than no speed number. The bench lane owns speed.
    }

    chat.maybeTitle(c)
    chat.persistNow(c)
    conv = null
    user = null
    turns = []
  }

  /** Throw the turn away - nothing was said, or the page is going. */
  function abandon(): void {
    const c = conv
    if (c) {
      const drop = new Set<Message>([...(user ? [user] : []), ...turns])
      c.messages = c.messages.filter((m) => !drop.has(m))
      // Dropping turns by identity can take the one the cursor points at,
      // which would render an empty thread over a full conversation.
      // `migrate` re-points a leaf that no longer resolves and is a no-op
      // when the abandoned turn was not the tip.
      migrate(c)
    }
    conv = null
    user = null
    turns = []
  }

  return { begin, apply, finish, abandon }
}

const concatSkew =
  'These models cut this recording into different pieces, so their transcripts are not lined up ' +
  'and should not be compared side by side.'
