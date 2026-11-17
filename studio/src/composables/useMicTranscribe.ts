// Talk into the page: a live microphone transcription session,
// fanned out to N models at once.
//
// Wraps the runner's `/v1/realtime?intent=transcription` socket - OpenAI's own
// realtime transcription dialect. One microphone, one capture graph, one
// resample, and the same PCM chunks pushed to every armed lane, so two models
// transcribe the identical audio and their transcripts race side by side. That
// is the comparison a recording could never make honestly: same words, same
// room, same moment.
//
// The socket goes through the manager (`/api/runners/{port}/v1/realtime`) for
// the reason every other runner call does: the browser never opens a runner
// port itself, so runner API keys stay server-side and there is no
// cross-origin question. The manager relays frames without reading them.
//
// PARTIALS do not FLICKER, and that is a server property, not a UI trick: the
// runner commits a word only once two consecutive passes over the growing
// audio agree on it (LocalAgreement-2), so what arrives here only ever grows.
// This file therefore appends deltas and never rewrites - if it ever needed to
// rewrite, the promise would already be broken upstream.
// The explicitly negotiated final-revision mode is the one exception at the
// utterance boundary: completed text replaces the provisional hypothesis so
// the persisted words and final alignment describe the same decode. Older
// runners/clients keep the original append-only mode.
//
// The UNIT is the UTTERANCE, not the recording. The session runs
// server VAD, so a pause closes an utterance and the runner answers it with a
// `completed` carrying segments, word times and confidence - the same object
// the file endpoint returns for that audio. A lane is therefore a LIST of
// finalised items plus the one still being spoken, and both callers read it
// that way: dictation inserts each item as it lands, the compare view makes a
// row of it.
//
// Which is also why the boundaries can be trusted to line up across lanes.
// The runner's detector is a pure function of the sample stream and every lane
// gets byte-identical frames (see `flush`), so utterance N is the same audio in
// every column - no clock lane, no negotiation. `at` is asserted rather than
// assumed anyway: a lane that disagrees about where its item started is a lane
// whose column cannot be compared, and saying so beats rendering it.
import { computed, ref, shallowRef } from 'vue'
import { recordAudio, type AudioRecording } from '@/lib/audio-recording'
import { SpeechProgress } from '@/lib/speech-progress'

import { openMic } from './useAudioDevices'
import { useMicLevels } from './useMicLevels'
import { PcmResampler } from '@/lib/pcm-resampler'

/** Sent to the model, and what the session is told to expect. Whisper works at
 *  16 kHz, so anything above it is bytes on the wire for nothing - the browser
 *  hands out 44.1 or 48 kHz and we down-rate here rather than shipping 3x the
 *  audio and asking the server to throw it away. */
const TARGET_RATE = 16000

/** How much audio to send per frame. Small enough that the server's next pass
 *  starts promptly, large enough that a 30-minute dictation is not a million
 *  WebSocket messages. */
const CHUNK_MS = 250

/** How many frames a lane may hold while its socket is still connecting.
 *  240 x 250 ms is a minute - orders of magnitude past any real handshake, so
 *  reaching it means the socket is never opening, and the lane says so rather
 *  than quietly transcribing audio with a minute missing from the front. */
const BACKLOG_MAX = 240

/** What the browser will record the clip as, best first. The list is a
 *  fallback ladder because Safari has no webm/opus and Chrome has no mp4/aac -
 *  `isTypeSupported` picks, and the extension follows from what it picked. */

interface Started {
  /** One runner port per lane. A single-element list is dictation; several is
   *  the compare fan-out. */
  ports: number[]
  /** ISO 639-1. Worth sending when known: the session locks its language on
   *  the first pass, and a guess made from one second of speech is a guess. */
  language?: string
  /** Per LANE, index-aligned with `ports`: ask this model's closed utterances
   *  to come back with segments, word times and per-word confidence
   *  (`paddock_verbose`). Absent = off for everyone.
   *
   *  Per lane and not per session because the lanes differ in whether they can
   *  answer it at all - the generative ASR families have no timestamp
   * vocabulary and REFUSE the field. A refused `session.update`
   *  is refused whole, so asking a model that cannot answer costs it the rate,
   *  the language and server VAD as well: measured, Qwen3-ASR then
   *  heard 16 kHz audio as 24 kHz, never closed a turn on its own, and
   *  answered three seconds of Swedish with seven seconds of Finnish.
   *
   *  Off for dictation whatever the lane can do: it costs a DTW pass per
   *  utterance, dictation never shows a word time, and the timestamp decode
   *  prompt is what some fine-tunes condition their no-speech refusal on
    */
  detail?: boolean[]
  /** Ordered Stop barrier, only for runners which explicitly advertise it. */
  drain?: boolean[]
  /** Final text may correct stable partials to match final-pass word timings. */
  finalRevision?: boolean[]
  /** The session ended without anyone stopping it - every lane failed, or the
   *  runner closed the socket. Called with what was heard and the recording so
   *  far, so the caller can close its turn instead of leaving one open
   *  forever. `stop()` resolving is the normal path; this is the other one. */
  onDied?: (out: MicResult) => void
  /** How long a quiet room goes unremarked, in ms. The session reports it and
   *  keeps listening either way - what the caller does about it is the
   *  caller's (see `DICTATION_IDLE_MS`). */
  idleMs?: number
  /** Also keep the audio, as a file the caller can attach to a turn.
   *
   *  Dictation does not want this (the words go in the composer and the sound
   *  is finished with), but a transcription TURN does: the conversation stores
   *  the clip beside its transcript, so the turn can be replayed, scrubbed and
   *  re-run months later. Recorded here rather than by the caller because this
   *  already owns the microphone - a second getUserMedia for the same voice
   *  would be a second capture of the same sound. */
  record?: boolean
  recordingTypes?: readonly string[]
}

/** What one utterance produced: every lane's transcript, and the recording
 *  when one was asked for. */
export interface MicResult {
  lanes: MicLane[]
  clip?: File
}

/** One word, as the enriched utterance reports it. */
export interface MicWord {
  word: string
  start: number
  end: number
  /** `exp(logprob)` - what the confidence colouring reads. Absent on a word
   *  the lane timed but could not score (Transcribe's alignment times every
   *  word; a word the decode spelled differently carries no figure). */
  confidence?: number
}

/** Why a span of this utterance produced nothing trustworthy - the runner's
 *  `paddock_guards`. An utterance can finalise empty and still be informative:
 *  "the model answered with its no-speech marker" is a result, and rendering it
 *  as a blank cell would be the silent failure the product principles ban. */
export interface MicGuard {
  reason: string
  note: string
  /** whether the span's text was DISCARDED, or merely cut short */
  dropped: boolean
}

/** One finalised utterance on one lane. */
export interface MicItem {
  /** Ordinal within this lane's session, and the join key across lanes: every
   *  lane hears the same audio through the same detector, so item N is the
   *  same span of speech in every column. */
  index: number
  /** Milliseconds from the start of the session to the start of this
   *  utterance's audio (the runner's `paddock_audio_start_ms`). Null on a lane
   *  or build that does not report it. */
  at: number | null
  text: string
  /** seconds of audio, when the utterance came back enriched */
  duration?: number
  words?: MicWord[]
  guards?: MicGuard[]
}

/** One model's live transcript. `error` is per-lane deliberately: a lane whose
 *  runner dies says so in its own column while the others keep listening -
 *  killing the whole session because one of four models fell over would throw
 *  away the recording everyone else is still producing. */
export interface MicLane {
  port: number
  /** utterances this lane has finalised, in the order they were spoken */
  items: MicItem[]
  /** The utterance still being spoken. Provisional - it is the one thing here
   *  that has not settled - so a caller shows it as such and never stores it. */
  open: string
  /** Everything above as one string, which is what a dictation caller wants
   *  without walking the list. */
  text: string
  /** the server's detector says someone is talking right now */
  speaking: boolean
  /** ISO 639-1 the session settled on. It LOCKS on the first pass that
   *  resolves one and is reported back on `session.updated`, so this is what
   *  the model actually heard rather than what anybody asked for. */
  language?: string
  error?: string
  /** the socket has sent its final transcript (or died) - nothing more comes */
  settled: boolean
}

/** How long a pause ends an utterance, in ms.
 *
 *  Longer than the API's own 500 default deliberately: an intra-sentence breath
 *  is 200-300 ms and an inter-sentence pause 500-1000, so 500 chops people
 *  mid-thought - and every chop costs the next utterance the context of the
 *  one before it (each is decoded independently; whisper.cpp's VAD mode sets
 *  `no_context = true` for the same reason). 700 sits between the two. */
const SILENCE_MS = 700

/** How much audio before the first detected speech frame an utterance keeps.
 *
 *  Twice the API's 300 default. The detector confirms speech from energy, and
 *  its back-dating only covers the three frames it waited to be sure - it
 *  cannot recover a first syllable that never crossed the threshold at all,
 *  and a quiet room is trimmed down to exactly this much pre-roll, so what is
 *  not kept here is gone. A soft or plosive opening word ("Testar...") is
 *  precisely the shape that starts under the threshold, and 300 ms of
 *  insurance is thin. The cost is 300 ms more audio per utterance, which is
 *  nothing; the failure it prevents is losing the word you started with. */
const PREFIX_MS = 600

/** Say something when an open microphone has heard nothing for this long. Not
 *  a stop by itself - the page decides that - just the end of pretending to
 *  listen. */
const IDLE_MS = 20000

/** What dictation uses instead, because there it is a stop.
 *
 *  The two shapes of voice UI split here and both are right for what they do.
 *  Turn-based ones (ChatGPT voice, phone assistants) end on silence, because
 *  silence is the end of your turn. DICTATION ones - Google Docs voice typing,
 *  macOS dictation, Dragon - keep listening until told, because pausing to
 *  think is not finishing. We are the second, so a pause never ends the
 *  session and never sends anything.
 *
 *  But an open microphone somebody has walked away from is a surprise, not a
 *  feature, so a LONG silence does close it - five seconds, well past any
 *  thinking pause and far past the 700 ms that merely ends an utterance. The
 *  page only acts on it once something has actually been dictated: clicking
 *  the mic and gathering your thoughts for eight seconds must not be answered
 *  by the mic giving up. */
export const DICTATION_IDLE_MS = 5000

/** After `stop`, how long to wait for a lane's last utterance before giving up
 *  on it. A closed utterance decodes in a second or two; this is not a budget,
 *  it is the difference between a slow model and a session that never ends. */
const SETTLE_TIMEOUT_MS = 60000

/** Cut an utterance by hand after this much UNBROKEN speech.
 *
 *  The runner bounds one utterance at ten minutes and, past that, sends an
 *  error and closes the socket - which would take the recording with it. No
 *  human talks for ten minutes without a 700 ms pause, but a noisy room does:
 *  a fan or traffic above the detector's threshold means the turn never ends,
 *  the buffer grows, and the session eventually dies with everything since the
 *  last utterance still in it.
 *
 *  So the page cuts first. Thirty seconds is whisper's own encoder window - an
 *  utterance past it is being decoded as two windows anyway - and it is far
 *  longer than any sentence, so this only ever fires when the detector has
 *  stopped being able to hear the difference between speech and the room. Then
 *  the transcript keeps moving instead of stalling until the session dies. */
const MAX_UTTERANCE_MS = 30000

export function useMicTranscribe() {
  // Same shared meter the record path feeds, so the bars look identical
  // whichever mic mode the arming picked.
  const meter = useMicLevels()
  const lanes = ref<MicLane[]>([])
  const listening = ref(false)
  /** The detector has heard nothing for `IDLE_MS`. Cleared by the next word. */
  const idle = ref(false)
  /** True from `stop()` until every lane has settled - the models are still
   *  working through audio that has already been spoken. */
  const finishing = ref(false)
  /** Session-level failure (no microphone, no capture). A LANE failing is not
   *  this: it lives on the lane. */
  const error = ref<string | null>(null)

  /** The first lane's transcript - what a single-port dictation caller wants,
   *  without making every caller index into an array. */
  const text = computed(() => lanes.value[0]?.text ?? '')

  const socks = shallowRef<WebSocket[]>([])
  /** Frames encoded before lane `i`'s socket finished its handshake.
   *
   *  Index-aligned with `socks`, like everything else here. It exists because
   *  sockets do not open in lockstep: `flush` used to send to whichever were
   *  open and drop the frame for the rest, so in a fan-out one model routinely
   *  started a few hundred milliseconds into the audio. That is a comparison
   *  which is quietly unfair - and once the session runs server VAD it is
   *  worse than unfair, because the detector is a function of the sample
   *  stream, so a lane fed a different one cuts its utterances in different
   *  places and the columns stop lining up at all. */
  let backlogs: string[][] = []
  const ctx = shallowRef<AudioContext | null>(null)
  let stream: MediaStream | null = null
  let generation = 0
  let pending: Float32Array[] = []
  let pendingLen = 0
  let done: ((lanes: MicLane[]) => void) | null = null
  let rec: AudioRecording | null = null
  /** `stop` has been called: the next time a lane runs out of outstanding
   *  utterances it is finished, rather than waiting for the next one. */
  let closing = false
  let straggler: ReturnType<typeof setTimeout> | null = null
  /** per lane, whether it asked for enriched utterances (see `Started.detail`) */
  let details: boolean[] = []
  let drainable: boolean[] = []
  let finalRevisions: boolean[] = []
  const draining = new Set<number>()
  /** what this session calls a quiet room (see `Started.idleMs`) */
  let idleMs = IDLE_MS
  /** Samples sent since the current run of speech began - the client's mirror
   *  of the buffer the runner is holding, and what `MAX_UTTERANCE_MS` bounds. */
  let spoken = 0
  /** Called when the session ends without anyone asking it to: a lane errored,
   *  a runner went away. The caller gets what was heard and the recording so
   *  far, because the alternative is a half-finished turn nobody ever closes. */
  let died: ((out: MicResult) => void) | null = null
  /** Utterances a lane has opened and not yet answered. Counted from
   *  `speech_started` rather than from deltas, because a short utterance can
   *  produce no deltas at all - LocalAgreement-2 needs two agreeing passes and
   *  a second and a half of speech may only get one - and settling a lane on
   *  "no partial text pending" would throw that utterance away at the moment
   *  its final pass was about to answer it. */
  const outstanding = new Map<number, SpeechProgress>()
  function progress(port: number): SpeechProgress {
    if (!outstanding.has(port)) outstanding.set(port, new SpeechProgress())
    return outstanding.get(port)!
  }

  /** Close the recording and hand back the file, or undefined when nothing was
   *  being recorded. Must finish before teardown stops the microphone track,
   *  which is why it is started first in `stop()`. */
  function finishRecording(): Promise<File | undefined> {
    const r = rec
    rec = null
    return r?.finish().catch(e => {
      error.value = e instanceof Error ? e.message : 'The recording could not be finalized'
      return undefined
    }) ?? Promise.resolve(undefined)
  }

  function toPcm16Base64(samples: Float32Array): string {
    const buf = new ArrayBuffer(samples.length * 2)
    const view = new DataView(buf)
    for (let i = 0; i < samples.length; i++) {
      // clamp before scaling: a browser's gain can push a sample past 1.0, and
      // letting that wrap is an audible click in the middle of a word
      const s = Math.max(-1, Math.min(1, samples[i]))
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    }
    const bytes = new Uint8Array(buf)
    let bin = ''
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
    }
    return btoa(bin)
  }

  /** Encode once and give the frame to every lane - the base64 of a 250 ms
   *  chunk is the same string for all of them, and re-encoding per lane would
   *  put the cost of comparing models on the microphone thread.
   *
   *  "Give", not "send": a lane whose socket is still connecting keeps it until
   *  it opens. Every lane hears the same audio from the same first sample, or
   *  the comparison is not one. */
  function flush(force = false): void {
    const want = force ? 1 : (TARGET_RATE * CHUNK_MS) / 1000
    if (pendingLen < want || !socks.value.length) return
    const merged = new Float32Array(pendingLen)
    let at = 0
    for (const p of pending) {
      merged.set(p, at)
      at += p.length
    }
    pending = []
    pendingLen = 0
    const frame = JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: toPcm16Base64(merged),
    })
    socks.value.forEach((s, i) => {
      if (s.readyState === WebSocket.OPEN) {
        s.send(frame)
        return
      }
      if (s.readyState !== WebSocket.CONNECTING) return
      const held = backlogs[i]
      if (!held) return
      held.push(frame)
      if (held.length > BACKLOG_MAX) {
        // Named, not silently truncated. A lane that never opened has no
        // transcript to lose, so this is the honest end for it.
        held.length = 0
        const port = lanes.value[i]?.port
        if (port !== undefined) settle(port, 'This model never accepted the connection.')
      }
    })

    // Cut a runaway utterance before the runner's own cap does, and cut it in
    // the same tick for every lane: they have all just been handed the same
    // frame, so their buffers are the same length and the commit lands on the
    // same sample. A per-lane timer would cut them at different offsets and
    // the columns would stop lining up.
    if (!lanes.value.some((l) => l.speaking)) return
    spoken += merged.length
    if (spoken < (MAX_UTTERANCE_MS / 1000) * TARGET_RATE) return
    spoken = 0
    const commit = JSON.stringify({ type: 'input_audio_buffer.commit' })
    for (const s of socks.value) {
      if (s.readyState === WebSocket.OPEN) s.send(commit)
    }
  }

  function releaseCapture(): void {
    meter.detach()
    stream?.getTracks().forEach((t) => t.stop())
    stream = null
    const previousContext = ctx.value
    ctx.value = null
    void previousContext?.close().catch(() => {})
  }

  async function teardown(): Promise<void> {
    if (straggler) {
      clearTimeout(straggler)
      straggler = null
    }
    closing = false
    died = null
    spoken = 0
    details = []
    drainable = []
    finalRevisions = []
    draining.clear()
    idleMs = IDLE_MS
    outstanding.clear()
    rec?.cancel()
    rec = null
    releaseCapture()
    for (const s of socks.value) s.close()
    socks.value = []
    backlogs = []
    pending = []
    pendingLen = 0
    listening.value = false
  }

  function laneAt(port: number): MicLane | undefined {
    return lanes.value.find((l) => l.port === port)
  }

  /** Everything this lane has said, settled part first. Maintained rather than
   *  computed because a lane is a plain object inside a `ref` array and a
   *  getter per lane is a getter per render. */
  function retext(l: MicLane): void {
    const said = l.items.map((i) => i.text).filter(Boolean).join(' ')
    l.text = l.open ? `${said} ${l.open}`.trim() : said
  }

  /** Turn a `completed` event into a finalised item. */
  function finalise(l: MicLane, msg: Record<string, unknown>): void {
    const v = msg.paddock_verbose as Record<string, unknown> | undefined
    const words = v?.words as
      | { word: string; start: number; end: number; paddock_confidence?: number }[]
      | undefined
    const guards = v?.paddock_guards as
      | { reason: string; note: string; text_dropped?: boolean }[]
      | undefined
    // The socket bills in DURATION rather than tokens, which is the honest
    // unit for audio, so `usage.seconds` says how much was heard even on a
    // session that did not ask for the enriched object.
    const usage = msg.usage as { seconds?: number } | undefined
    l.items.push({
      index: l.items.length,
      // `??` and not `||`: millisecond zero is where the first utterance of a
      // session that opened on a word actually starts.
      at: (msg.paddock_audio_start_ms as number | undefined) ?? null,
      // Terminal text is authoritative. In negotiated final-revision mode it
      // can correct the provisional prefix without decoding the audio again.
      text: ((msg.transcript as string) ?? l.open).trim(),
      duration: (v?.duration as number | undefined) ?? usage?.seconds,
      words: words?.map((w) => ({
        word: w.word,
        start: w.start,
        end: w.end,
        confidence: w.paddock_confidence,
      })),
      // absent rather than empty, the same way the runner sends it: this key
      // means "look here", and an empty array on every clean utterance is a key
      // every reader has to learn to ignore
      guards: guards?.length
        ? guards.map((g) => ({ reason: g.reason, note: g.note, dropped: g.text_dropped === true }))
        : undefined,
    })
    l.open = ''
    retext(l)
  }

  /** Settle one lane. The session itself ends only when they all have -
   *  everything below is written so one model's failure costs exactly that
   *  model's column. */
  function settle(port: number, why?: string): void {
    const l = laneAt(port)
    if (!l || l.settled) return
    l.settled = true
    if (why) l.error = why
    if (!lanes.value.every((x) => x.settled)) return
    if (done) {
      const finish = done
      done = null
      // Stop owns teardown after both the recorder and all lanes finish.
      // A quiet lane can settle synchronously, before final data is delivered.
      finish(lanes.value)
      return
    }
    finishing.value = false
    // Nobody asked for this. Every lane is gone while the microphone is still
    // open - a runner died, or the session hit a limit and the socket closed.
    // Tearing down here would stop the recorder and drop its chunks, leaving
    // the caller with a turn it can never finish, so close the recording
    // properly and hand back what there is. What was said is still worth
    // keeping; it is the rest of the sentence that was lost.
    const was = lanes.value
    const diedGeneration = ++generation // Invalidate an outstanding microphone permission/open.
    listening.value = false
    finishing.value = true
    const tell = died
    void finishRecording().then((clip) => {
      if (diedGeneration !== generation) return
      finishing.value = false
      void teardown()
      tell?.({ lanes: was, clip })
    })
  }

  /** The whole session failed (microphone, capture) - distinct from a lane. */
  function fail(msg: string): void {
    error.value = msg
    finishing.value = false
    for (const l of lanes.value) l.settled = true
    const finish = done
    done = null
    void teardown()
    finish?.(lanes.value)
  }

  function openLane(port: number, i: number, language?: string): WebSocket {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const sock = new WebSocket(
      `${proto}//${location.host}/api/runners/${port}/v1/realtime?intent=transcription`,
    )
    sock.onerror = () => settle(port, 'This model could not be reached.')
    sock.onclose = () => settle(port, 'This model closed the session.')
    sock.onmessage = (ev: MessageEvent<string>) => {
      let msg: Record<string, unknown>
      try {
        msg = JSON.parse(ev.data) as Record<string, unknown>
      } catch {
        return
      }
      const l = laneAt(port)
      if (!l) return
      if (!progress(port).receive(msg)) return
      switch (msg.type as string) {
        case 'input_audio_buffer.drained':
          draining.delete(port)
          if (closing && !progress(port).waiting) settle(port)
          break
        case 'error': {
          const e = msg.error as { message?: string } | undefined
          settle(port, e?.message ?? 'This model failed.')
          break
        }
        case 'session.updated': {
          // The session locks its language on the first pass that resolves one
          // and says so here. Worth keeping: it is what the model HEARD, which
          // on a monolingual fine-tune is not always what it was asked for.
          const s = msg.session as
            | { audio?: { input?: { transcription?: { language?: string | null } } } }
            | undefined
          const code = s?.audio?.input?.transcription?.language
          if (code) l.language = code
          break
        }
        case 'input_audio_buffer.speech_started':
          l.speaking = true
          idle.value = false
          spoken = 0
          break
        case 'input_audio_buffer.speech_stopped':
          // The turn is over but its answer is not here yet - the lane is not
          // idle, it is thinking, and `outstanding` is what knows the
          // difference.
          l.speaking = progress(port).speaking !== undefined
          break
        case 'input_audio_buffer.timeout_triggered':
          // Every lane runs the same detector on the same audio, so one lane
          // saying the room is quiet is all of them saying it.
          idle.value = true
          break
        case 'conversation.item.input_audio_transcription.delta':
          l.open += (msg.delta as string) ?? ''
          retext(l)
          break
        case 'conversation.item.input_audio_transcription.completed': {
          l.speaking = progress(port).speaking !== undefined
          finalise(l, msg)
          // These arrive all session long now, so a completed is the END of the
          // lane only once the page has stopped listening and this was the last
          // utterance owed.
          if (closing && !progress(port).waiting && !draining.has(port)) settle(port)
          break
        }
      }
    }
    sock.onopen = () => {
      sock.send(
        JSON.stringify({
          type: 'session.update',
          session: {
            type: 'transcription',
            audio: {
              input: {
                format: { type: 'audio/pcm', rate: TARGET_RATE },
                // The server ends the turns, which is what makes the utterance
                // the unit  -, because the detector is a pure
                // function of the sample stream that every lane receives
                // identically, what makes item N the same audio in every
                // column without anybody coordinating.
                turn_detection: {
                  type: 'server_vad',
                  silence_duration_ms: SILENCE_MS,
                  prefix_padding_ms: PREFIX_MS,
                  idle_timeout_ms: idleMs,
                },
                transcription: {
                  ...(language ? { language } : {}),
                  // Only where this model can answer it. Asking a lane that
                  // cannot gets the whole update refused, rate and VAD with it.
                  ...(details[i] ? { paddock_verbose: true } : {}),
                  ...(finalRevisions[i] ? { paddock_final_revision: true } : {}),
                },
              },
            },
          },
        }),
      )
      // The audio that arrived while this socket was still shaking hands, in
      // order and after the config above - the session assumes 24 kHz until
      // told otherwise, so a frame that overtook the update would be heard
      // 1.5x fast.
      const held = backlogs[i]
      if (held?.length) {
        for (const f of held) sock.send(f)
        held.length = 0
      }
    }
    return sock
  }

  async function start(opts: Started): Promise<void> {
    if (listening.value || finishing.value || !opts.ports.length) return
    const ticket = ++generation
    // Browsers only hand the microphone to secure origins (https, or the
    // localhost carve-out). On a plain-http LAN address `navigator.mediaDevices`
    // does not exist at all - say the real reason instead of crashing on it.
    //
    // Since paddock serves https by default, so reaching this means
    // the box could not establish a certificate - which is what the Trust page
    // reports. Point there rather than at a scheme the user cannot choose.
    if (!navigator.mediaDevices?.getUserMedia) {
      error.value =
        'The browser blocks the microphone on this address. See Trust this computer in the Manager.'
      return
    }
    error.value = null
    idle.value = false
    closing = false
    details = opts.ports.map((_, i) => opts.detail?.[i] === true)
    drainable = opts.ports.map((_, i) => opts.drain?.[i] === true)
    finalRevisions = opts.ports.map((_, i) => opts.finalRevision?.[i] === true)
    draining.clear()
    idleMs = opts.idleMs ?? IDLE_MS
    died = opts.onDied ?? null
    spoken = 0
    outstanding.clear()
    lanes.value = opts.ports.map((port) => ({
      port,
      items: [],
      open: '',
      text: '',
      speaking: false,
      settled: false,
    }))
    // ---- 1. Everything that does not need the MICROPHONE, first ----
    //
    // Order is the whole point here. `addModule` is a fetch and a compile, and
    // it used to sit between the microphone going live and anything listening
    // to it - so the operating system lit its recording indicator, the speaker
    // took that as their cue, and the words they said in that window went into
    // the MediaRecorder and never reached the socket. That is a first word
    // lost every time somebody starts talking promptly, and it went away if
    // they happened to pause first.
    //
    // None of this needs a microphone, and doing it before the permission call
    // also hides the socket handshakes behind the prompt.
    // Prefer the browser audio engine's anti-aliased conversion on its render
    // thread. Older engines can refuse this rate; their fallback keeps phase.
    let ac: AudioContext
    try { ac = new AudioContext({ sampleRate: TARGET_RATE }) }
    catch { ac = new AudioContext() }
    const resampler = new PcmResampler(ac.sampleRate)
    ctx.value = ac
    backlogs = opts.ports.map(() => [])
    socks.value = opts.ports.map((p, i) => openLane(p, i, opts.language))
    let node: AudioNode
    // ScriptProcessor only: it fires the moment the graph reaches the
    // destination, zeros included, and the seconds the permission prompt is up
    // must not become leading silence that shifts every lane's t=0. Flipped
    // once the mic source is connected. (The worklet needs no gate -
    // disconnected, it receives nothing.)
    let micLive = false
    try {
      // Resumed here, still inside the click that started this: a context
      // created outside a user gesture can come up suspended, and after the
      // permission prompt the gesture is spent.
      if (ac.state === 'suspended') await ac.resume()
      if (ticket !== generation) return
      const onBlock = (block: Float32Array) => {
        if (closing || ticket !== generation) return
        const down = resampler.push(block)
        pending.push(down)
        pendingLen += down.length
        flush()
      }
      if (ac.audioWorklet) {
        await ac.audioWorklet.addModule('/pcm-capture-worklet.js')
        if (ticket !== generation) return
        const worklet = new AudioWorkletNode(ac, 'pcm-capture')
        worklet.port.onmessage = (e: MessageEvent<Float32Array>) => onBlock(e.data)
        node = worklet
      } else {
        // `audioWorklet` is secure-context-only, so a plain-http LAN origin has
        // none even where the microphone itself has been unlocked (Safari's
        // "Allow Media Capture on Insecure Sites"). ScriptProcessorNode is the
        // one capture API such an origin still has: deprecated and main-thread,
        // so a busy render delivers a block late - latency, not lost samples.
        const spn = ac.createScriptProcessor(4096, 1, 1)
        spn.onaudioprocess = (e) => {
          // Copy: the browser reuses the buffer behind getChannelData, and at a
          // hardware rate equal to TARGET_RATE downRate returns its input
          // uncopied - `pending` would hold a view the next block overwrites.
          if (micLive) onBlock(new Float32Array(e.inputBuffer.getChannelData(0)))
        }
        node = spn
      }
      // A worklet with no consumer is allowed to be culled - and a
      // ScriptProcessor never fires unless it reaches the destination - so
      // terminate the graph in a gain node muted to zero rather than the
      // speakers: this is a microphone, and routing it to the output is a
      // feedback loop.
      const sink = ac.createGain()
      sink.gain.value = 0
      node.connect(sink).connect(ac.destination)
    } catch (e) {
      if (ticket !== generation) return
      fail(`Audio capture failed: ${e instanceof Error ? e.message : String(e)}`)
      return
    }

    // ---- 2. the microphone ----
    try {
      // The chosen device, plus the browser's own noise reduction - which is
      // the only one there is, since the session refuses that parameter rather
      // than pretending to have it, and on a laptop microphone it is the
      // difference between a transcript and a guess.
      const opened = await openMic()
      if (ticket !== generation) { opened.getTracks().forEach(t => t.stop()); return }
      stream = opened
    } catch (e) {
      if (ticket !== generation) return
      error.value = `Microphone unavailable: ${e instanceof Error ? e.message : String(e)}`
      void teardown()
      lanes.value = []
      return
    }

    // ---- 3. start hearing it, in one synchronous run ----
    //
    // No await from here to `listening`: the connect and the recorder start
    // together, so the clip and the transcript share a t=0. They did not
    // before - the recorder ran while the graph was still being built - which
    // also put every word time out by the length of that gap, since the times
    // are measured from the session's first sample and the player's from the
    // file's.
    try {
      const src = ac.createMediaStreamSource(stream)
      src.connect(node)
      // The level meter taps the same source, not a second microphone: an
      // analyser that connects to nothing cannot route the mic to the speakers
      // and cannot perturb what the worklet sees.
      meter.attach(ac, src)
      micLive = true
      if (opts.record) {
        try {
          rec = recordAudio(stream, opts.recordingTypes)
        } catch {
          // No recorder for any container this browser admits to supporting.
          // The live transcripts still work; the caller finds out by getting no
          // clip back, and says so rather than committing a turn with silence
          // where its audio should be.
          rec = null
        }
      }
      listening.value = true
    } catch (e) {
      fail(`Audio capture failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  /** Stop listening, and resolve once every lane has answered for the audio it
   *  already has. */
  async function stop(): Promise<MicResult> {
    if (!listening.value) return { lanes: lanes.value }
    listening.value = false
    closing = true
    // Close the recording first: teardown stops the microphone track, and a
    // recorder whose source ends mid-flush can lose its tail.
    const clipP = finishRecording()
    // Stop owns the capture cutoff, not the last server response. The model
    // may spend seconds finishing; it must not hear anything said after Stop.
    const captureGeneration = generation
    void clipP.then(() => { if (captureGeneration === generation) releaseCapture() })
    // Stopped before any socket finished opening - there is nothing on the far
    // side to commit to, and sending on a CONNECTING socket throws. Whatever
    // was captured is gone, which is the honest outcome of a click-and-cancel.
    const open = socks.value.filter((s) => s.readyState === WebSocket.OPEN)
    if (!open.length) {
      const said = lanes.value
      const clip = await clipP
      cancel()
      return { lanes: said, clip }
    }
    finishing.value = true
    flush(true)
    // Arm the resolver before settling anything: settling the last lane
    // resolves the session, and a lane can settle inside the loop below.
    const answered = new Promise<MicLane[]>((resolve) => {
      done = resolve
    })
    // `socks` is built from `ports` in order, so index i is lane i. A lane
    // whose socket never opened cannot answer - settle it now, or the session
    // waits forever on a socket that is still connecting.
    const commit = JSON.stringify({ type: 'input_audio_buffer.commit' })
    for (let i = 0; i < lanes.value.length; i++) {
      const l = lanes.value[i]
      if (socks.value[i]?.readyState !== WebSocket.OPEN) {
        settle(l.port, 'This model was not ready in time.')
        continue
      }
      // Cut the turn short only if one is actually running. The detector ends
      // its own turns now, so a stop during a pause has nothing left to close
      // - committing anyway would hand the model the few hundred milliseconds
      // of pre-roll it keeps and file the answer as an utterance.
      if (drainable[i]) {
        draining.add(l.port)
        progress(l.port).beginDrain()
        socks.value[i].send(JSON.stringify({ type: 'input_audio_buffer.drain' }))
      } else if (l.speaking) socks.value[i].send(commit)
      // Nothing outstanding means every word this lane heard is already an
      // item. Not "no partial text": a short utterance can produce no deltas at
      // all and still be about to answer.
      else if (!progress(l.port).waiting) settle(l.port)
    }
    // A lane that never answers must not hold the session open forever. This
    // is not a decode budget - a closed utterance settles in a second or two -
    // it is the difference between a slow model and a recording you can never
    // finish.
    straggler = setTimeout(() => {
      for (const l of lanes.value) {
        settle(l.port, 'This model did not finish the last thing you said.')
      }
    }, SETTLE_TIMEOUT_MS)
    const [said, clip] = await Promise.all([answered, clipP])
    if (captureGeneration === generation) {
      finishing.value = false
      await teardown()
    }
    return { lanes: said, clip }
  }

  /** Throw the utterance away - the sockets and the microphone go with it. */
  function cancel(): void {
    ++generation
    const finish = done
    done = null
    finishing.value = false
    lanes.value = []
    void teardown()
    finish?.([])
  }

  return { lanes, text, listening, finishing, idle, error, levels: meter.levels, start, stop, cancel }
}
