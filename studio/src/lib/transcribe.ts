// Transcription as a TURN. A transcriber answers a user turn the
// way a chat model does - audio in, transcript out - so it rides the ordinary
// conversation machinery: one grouped assistant message per lane, streamed,
// persisted, with usage. This module is only the WIRE half: post the clip,
// consume the SSE, hand back text + timings + tokens.
//
// It lives beside useChatStream rather than inside it because the two speak
// different protocols to different endpoints (multipart + transcript.text.*
// events vs JSON + the Responses stream) and useChatStream is already long.
// The dispatch between them is one `kind === 'transcriber'` branch in `run`.

import type {
  SpeakerTimeline,
  TranscriptGuard,
  TranscriptMeta,
  TranscriptSegment,
  TranscriptWord,
  Usage,
} from '@/types/chat'

/** One word as the runner sends it, in either of the two shapes it uses.
 *
 *  The spec array (`words[]`) is `{word, start, end}` with our figures under
 *  `paddock_` names, because that container is OpenAI's and an undeclared key
 *  on it would be a claim about the spec. The extension array
 *  (`paddock_words[]`) is already namespaced, so `confidence` sits bare there.
 *  Both are read here; `paddock_alt`/`paddock_margin` ride only where the lane
 *  could answer them, so their absence is honest rather than a zero that would
 *  read as "no close call". */
interface WireWord {
  word?: string
  start?: number
  end?: number
  confidence?: number
  paddock_confidence?: number
  paddock_alt?: string
  paddock_margin?: number
  /** The speaker under a timed word, where the lane has a speaker timeline
   *  of its own (Nemotron 3.5 Transcribe): the shared time-overlap
   *  attribution, under `paddock_` names on the spec array. */
  paddock_speaker?: number | null
  paddock_speakers?: number[]
  paddock_speaker_status?: TranscriptWord['speaker_status']
  paddock_speaker_coverage?: number[]
}

/** A lane's own speaker timeline (`paddock_verbose.paddock_diarization`) -
 *  the same shape a diarization model answers with, minus its words. */
interface WireTimeline {
  model?: string
  duration?: number
  segments?: { speaker?: number; start?: number; end?: number }[]
  preset?: string
  attribution?: string
}

/** One segment as the runner sends it (`paddock_verbose.segments[]`). The
 *  logprob fields are wire-only: `avg_logprob` becomes the segment's
 *  confidence here and `paddock_words[].logprob` is dropped for its already-
 *  computed `confidence`, so nothing stores the same fact twice. */
interface WireSegment {
  start?: number
  end?: number
  text?: string
  /** mean per-token logprob over the span. */
  avg_logprob?: number
  paddock_words?: WireWord[]
}

/** What the runner's `transcript.text.done` carries. `usage` is spec; the
 *  `paddock_verbose` block is our named extension (OpenAI's done event has
 *  `text` and nothing else, so asking for segments AND streaming would
 *  otherwise mean sending the clip twice - a full re-decode to learn what the
 *  server already computed). */
interface DoneEvent {
  type: 'transcript.text.done'
  text?: string
  languages?: { code?: string }[]
  usage?: {
    input_tokens?: number
    output_tokens?: number
    total_tokens?: number
  }
  paddock_verbose?: {
    language?: string
    duration?: number
    segments?: WireSegment[]
    /** OpenAI's own word array - present when `word` granularity was asked
     * for and the lane can answer it (whisper). Its words carry
     *  times, which is what makes them spec words. */
    words?: WireWord[]
    /** the segment-less lanes' per-word confidence, times and all absent
      */
    paddock_words?: WireWord[]
    paddock_guards?: WireGuard[]
    paddock_diarization?: WireTimeline
  }
  /** Also at the top level of the done event, so a caller who streamed plain
   * `json` still learns a span was refused. */
  paddock_guards?: WireGuard[]
}

/** One decode-guard notice as the runner sends it. */
interface WireGuard {
  start?: number
  end?: number
  reason?: string
  note?: string
  text_dropped?: boolean
}

export interface TranscribeResult {
  text: string
  meta: TranscriptMeta
  usage: Usage
}

export interface TranscribeOpts {
  /** Where the clip goes - always a manager URL, never a provider or a runner
   *  port directly, because both kinds of key stay server-side
   *  (`models.transcribeUrl`). Local: `/api/runners/{port}/v1/audio/...`.
   *  Cloud: `/api/cloud/{endpoint}/v1/audio/...`. */
  url: string
  /** Whether this endpoint types the transcript in as it goes. Local runners
   *  do (`transcript.text.*` SSE); no provider does - OpenRouter answers one
   *  JSON body when the whole clip is finished. Both shapes come
   *  back through this one function so the caller has no dialect to know. */
  stream?: boolean
  /** Which model to ask, for an endpoint that serves more than one. A runner
   *  serves exactly one and ignores this; a provider needs it named. */
  model?: string
  /** ISO 639-1 to force, or undefined to let the model detect. */
  language?: string
  /** The instruction, for a model whose instruction is the task selector
   *  (granite-speech: punctuated transcript vs speaker labels vs
   *  translation). Sending one to a model that has no such interface would
   *  quietly change its decode, so the caller decides - this rides only when
   *  the lane is a generative ASR model. */
  prompt?: string
  /** Ask for segment times. Whisper emits them as vocabulary TOKENS, so
   *  asking changes the decode (`<|notimestamps|>` is dropped from the
   *  prompt); the generative families have no timestamp vocabulary at all and
   *  answer an honest 400. Both reasons say the same thing: only ask when the
   *  endpoint advertised `segment` in its `timestamp_granularities`. */
  timestamps?: boolean
  /** Ask for per-word confidence (`include=logprobs`). Costs the endpoint its
   *  decode-overlap fast path, so it rides only where it is wanted - and only
   *  where the endpoint said it can answer. */
  wordConfidence?: boolean
  /** Ask for per-WORD times (`timestamp_granularities[]=word`). A separate ask
   *  from `timestamps`, and one the endpoint answers by one of two unrelated
   *  routes: whisper re-runs each 30 s window through the decoder to recover
   *  them from cross-attention (roughly a fifth again on top of the decode),
   *  while granite-speech-plus is INSTRUCTED to write them into its transcript
   *  - which means it cannot also be given a `prompt`, and its answer comes
   *  back without punctuation. Only ask where the endpoint advertised `word`. */
  wordTimes?: boolean
  /** Called with the growing transcript as deltas land, so the turn types
   *  itself in exactly like a chat reply. */
  onDelta?: (fullText: string) => void
  signal?: AbortSignal
}

/** Whether this file is something a transcriber can take. Kept deliberately
 *  loose: the runner owns the real format list (symphonia + libopus)
 *  and refuses with a proper message. A tight allow-list here would
 *  reject formats the server actually supports. */
export function isAudioFile(f: { type: string; name: string }): boolean {
  if (f.type.startsWith('audio/') || f.type === 'video/webm') return true
  return /\.(wav|mp3|m4a|aac|flac|ogg|oga|opus|webm|mp4|mpga|mpeg)$/i.test(f.name)
}

/** Read the clip's duration without decoding it, so the chip and the player's
 *  scrub bar are honest before any model has answered. Returns undefined when
 *  the browser cannot tell (some containers) - the caller must not invent a
 *  number, it simply shows no duration. */
export async function audioDuration(blob: Blob | string): Promise<number | undefined> {
  const url = typeof blob === 'string' ? blob : URL.createObjectURL(blob)
  try {
    return await new Promise<number | undefined>((resolve) => {
      const a = new Audio()
      const done = (v?: number) => resolve(v !== undefined && isFinite(v) && v > 0 ? v : undefined)
      a.onloadedmetadata = () => {
        if (isFinite(a.duration) && a.duration > 0) return done(a.duration)
        // A LIVE-MUXED file has no duration to read: a browser recording your
        // microphone cannot know the length before it happens, so it writes
        // none and `duration` comes back Infinity. (Same root as the decode
        // bug in  - the sizes are not there either.) Seeking past the
        // end makes the browser scan for the real end and fire `timeupdate`
        // with it. Without this a recorded clip had no length anywhere: no
        // "Audio 0:04", and no denominator for realtime speed.
        a.ontimeupdate = () => {
          a.ontimeupdate = null
          done(a.duration)
        }
        a.currentTime = 1e101
      }
      a.onerror = () => done(undefined)
      // Never hang a send on a file that answers neither way.
      setTimeout(() => done(undefined), 3000)
      a.src = url
    })
  } finally {
    if (typeof blob !== 'string') URL.revokeObjectURL(url)
  }
}

/** Words from either wire shape. One function so a spec word and an extension
 *  word render identically and their numbers mean the same thing - the only
 *  difference that survives is whether times came with them. */
function wordsOf(raw: WireWord[] | undefined): TranscriptWord[] | undefined {
  const out = (raw ?? []).flatMap((w) => {
    const confidence = w.confidence ?? w.paddock_confidence
    // Times ride only as a PAIR. Half a span would put a seek on a word whose
    // end nothing knows, and the runner never sends one - this is the guard
    // that keeps that true on the reading side too.
    const timed = w.start !== undefined && w.end !== undefined
    // A word needs something to say: a confidence, or a time. A timed word
    // without one is real (Transcribe's alignment times every word, and the
    // decode's confidence rides only when asked for and matched) - dropping
    // it would lose the seek for a number that was never there.
    if (w.word === undefined || (confidence === undefined && !timed)) return []
    const word: TranscriptWord = {
      word: w.word,
      confidence,
      start: timed ? w.start : undefined,
      end: timed ? w.end : undefined,
      alt: w.paddock_alt,
      margin: w.paddock_margin,
    }
    if (w.paddock_speaker_status !== undefined) {
      word.speaker = w.paddock_speaker ?? null
      word.speakers = w.paddock_speakers ?? []
      word.speaker_status = w.paddock_speaker_status
      word.speaker_coverage = w.paddock_speaker_coverage
    }
    return [word]
  })
  return out.length ? out : undefined
}

/** The lane's own speaker timeline, or undefined. Only the attribution the
 *  speaker view knows how to read is taken - a timeline under some other
 *  contract would be words labelled by rules this client does not show. */
export function timelineOf(raw: WireTimeline | undefined, fallbackModel: string): SpeakerTimeline | undefined {
  if (!raw || raw.attribution !== 'time_overlap_v1' || !(raw.duration && raw.duration > 0)) return undefined
  const segments = (raw.segments ?? []).flatMap((s) =>
    Number.isInteger(s.speaker) && s.start !== undefined && s.end !== undefined && s.end > s.start
      ? [{ speaker: s.speaker as number, start: s.start, end: s.end }]
      : [],
  )
  return {
    model: raw.model ?? fallbackModel,
    duration: raw.duration,
    segments,
    preset: raw.preset ?? '',
    attribution: 'time_overlap_v1',
  }
}

/** The runner answers `"unknown"` where no lane could name the language:
 *  OpenAI types the field as a required string, so it cannot send null, and
 *  inventing a language would be worse than either. Here it becomes an
 *  absence, which is what the rest of the Studio already reads as "we do not
 *  know" - the detail chip simply does not appear. */
function knownLanguage(code: string | undefined): string | undefined {
  return code && code !== 'unknown' ? code : undefined
}

function segmentsOf(d: DoneEvent | undefined): TranscriptSegment[] | undefined {
  const raw = d?.paddock_verbose?.segments
  if (!raw?.length) return undefined
  return raw.map((s) => {
    const words = wordsOf(s.paddock_words) ?? []
    return {
      start: s.start ?? 0,
      end: s.end ?? 0,
      text: s.text ?? '',
      // the segment's own confidence is exp(mean logprob) - the same
      // transform the per-word figures already arrive under
      confidence: s.avg_logprob !== undefined ? Math.exp(s.avg_logprob) : undefined,
      words: words.length ? words : undefined,
    }
  })
}

/** The spans the server refused or cut.
 *
 *  The server's own `note` is carried through rather than re-worded here: a
 *  Studio user and a curl user should get the same explanation, and a second
 *  copy of the reasoning in the client is a second copy to drift. */
function guardsOf(d: DoneEvent | undefined): TranscriptGuard[] | undefined {
  const raw = d?.paddock_guards ?? d?.paddock_verbose?.paddock_guards
  if (!raw?.length) return undefined
  return raw.map((g) => ({
    start: g.start ?? 0,
    end: g.end ?? 0,
    reason: g.reason ?? 'unknown',
    note: g.note ?? '',
    dropped: g.text_dropped === true,
  }))
}

/** POST one clip and stream the transcript back.
 *
 *  The delta stream carries the same text the done event ends with - the
 *  runner guarantees the concatenated deltas equal `done.text` byte for byte
 *  (its conformance gate asserts exactly that) - so `onDelta` can render
 *  optimistically and the final assignment is a no-op rather than a rewrite. */
export async function transcribeClip(
  blob: Blob,
  filename: string,
  opts: TranscribeOpts,
): Promise<TranscribeResult> {
  const started = performance.now()
  const streaming = opts.stream !== false
  const form = new FormData()
  form.append('file', blob, filename || 'audio.wav')
  if (streaming) form.append('stream', 'true')
  // verbose_json so the done event can carry segments + duration; the runner
  // serves them through the paddock_verbose extension on the stream.
  //
  // A cloud lane asks for plain `json` instead, and that is a decision rather
  // than an omission: OpenRouter honours verbose_json and word timestamps only
  // on some of its providers, so asking everywhere would turn a working
  // transcription into a refusal depending on who happened to serve it. Text
  // and usage are what every provider answers; the timings are what the local
  // lane is for.
  form.append('response_format', streaming ? 'verbose_json' : 'json')
  if (opts.timestamps) form.append('timestamp_granularities[]', 'segment')
  if (opts.wordTimes) form.append('timestamp_granularities[]', 'word')
  if (opts.wordConfidence) form.append('include[]', 'logprobs')
  if (opts.language) form.append('language', opts.language)
  if (opts.prompt) form.append('prompt', opts.prompt)
  if (opts.model) form.append('model', opts.model)

  // ...and again on the query, for the manager's usage ledger. It forwards the
  // multipart body untouched (a rewritten `boundary=` is unparseable), so the
  // form field is the provider's and this one is the ledger's. The form stays
  // authoritative; a runner ignores both.
  const url =
    opts.model && !streaming
      ? `${opts.url}?model=${encodeURIComponent(opts.model)}`
      : opts.url

  const res = await fetch(url, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })
  if (!res.ok) {
    // The runner answers a refusal as JSON with a real sentence; surface it
    // rather than a status code (no silent failures).
    let why = `transcription failed (${res.status})`
    try {
      const j = (await res.json()) as { error?: { message?: string } }
      if (j?.error?.message) why = j.error.message
    } catch {
      /* non-JSON body - keep the status line */
    }
    throw new Error(why)
  }

  // The non-streamed shape: one JSON body, arriving when the whole clip is
  // done. `usage` is the provider's, and its fields are not the runner's - a
  // whisper-class model bills per audio SECOND and reports no token counts at
  // all, so anything absent stays absent rather than becoming a zero that
  // would read as "it cost nothing".
  if (!streaming) {
    const j = (await res.json()) as {
      text?: string
      duration?: number
      language?: string
      usage?: { input_tokens?: number; output_tokens?: number; seconds?: number; cost?: number }
    }
    const ms = performance.now() - started
    const final = j.text ?? ''
    opts.onDelta?.(final)
    const outTokens = j.usage?.output_tokens ?? 0
    return {
      text: final,
      meta: {
        // Plain `json` carries no `language` (only verbose_json does, and only
        // on some providers), so a cloud lane sat with an empty Language chip
        // beside a local one that filled it in - same clip, same forced
        // language. Where the caller FORCED a language it
        // is not a detection to report, it is a fact about the request, and
        // the transcript really is in it. Left on auto, nothing is shown,
        // because nobody told us and guessing from the text would be worse.
        language: knownLanguage(j.language) ?? opts.language,
        // Same shape of gap as `language`: plain `json` has no `duration`, but
        // a duration-billed model reports the seconds it charged for, and that
        // is the length of the clip. Measured on OpenRouter -
        // qwen3-asr-flash answers `usage: {seconds: 4, cost: 0.00014}` and no
        // token counts at all.
        durationS: j.duration ?? j.usage?.seconds,
        // No segments, no words, no guards: this lane answers with text. The
        // views already render their absence (the generative ASR lanes have
        // had no segments) rather than inventing an empty one.
      },
      usage: {
        promptTokens: j.usage?.input_tokens ?? 0,
        completionTokens: outTokens,
        ms,
        // No first-token moment exists when there is only one moment; a TTFT
        // equal to the whole turn would be a lie the stat line reads as slow.
        tps: outTokens > 0 && ms > 0 ? (outTokens / ms) * 1000 : undefined,
        costUsd: j.usage?.cost,
      },
    }
  }

  if (!res.body) throw new Error('the transcription stream ended before it began')
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  let text = ''
  let ttftMs: number | undefined
  let done: DoneEvent | undefined

  for (;;) {
    const { value, done: fin } = await reader.read()
    if (fin) break
    buf += dec.decode(value, { stream: true })
    // SSE frames are blank-line separated; a frame may hold several `data:`
    // lines, and the tail of `buf` is usually a partial frame.
    let cut: number
    while ((cut = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, cut)
      buf = buf.slice(cut + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload || payload === '[DONE]') continue
        let ev: { type?: string; delta?: string }
        try {
          ev = JSON.parse(payload) as { type?: string; delta?: string }
        } catch {
          continue
        }
        if (ev.type === 'transcript.text.delta') {
          if (ttftMs === undefined) ttftMs = performance.now() - started
          text += ev.delta ?? ''
          opts.onDelta?.(text)
        } else if (ev.type === 'transcript.text.done') {
          done = ev as DoneEvent
        }
      }
    }
  }

  const ms = performance.now() - started
  // done.text is authoritative; the deltas are guaranteed to equal it, so
  // this reconciles rather than replaces.
  const final = done?.text ?? text
  const outTokens = done?.usage?.output_tokens ?? 0
  return {
    text: final,
    meta: {
      language: knownLanguage(done?.paddock_verbose?.language ?? done?.languages?.[0]?.code),
      durationS: done?.paddock_verbose?.duration,
      segments: segmentsOf(done),
      // the spec array first: where both arrive, one of them has times and
      // that is the one worth keeping
      words: wordsOf(done?.paddock_verbose?.words ?? done?.paddock_verbose?.paddock_words),
      guards: guardsOf(done),
      diarization: timelineOf(done?.paddock_verbose?.paddock_diarization, opts.model ?? ''),
    },
    usage: {
      // The audio side is billed in ENCODER ROWS, which is what the server
      // actually spent - not a text-token count that would be a fiction.
      promptTokens: done?.usage?.input_tokens ?? 0,
      completionTokens: outTokens,
      ms,
      ttftMs,
      tps: outTokens > 0 && ms > 0 ? (outTokens / ms) * 1000 : undefined,
    },
  }
}
