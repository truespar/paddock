/** One capability policy for the web composer and its native presentation. */
export type MicMode = 'live' | 'record' | 'dictate'
export interface AudioLane { chat: boolean; audio: boolean; live: boolean }
export interface RealtimeTranscriptionCaps { supported: boolean; enrichment: boolean; drain?: boolean; final_revision?: boolean; utterance_max_s?: number | null }
/** Explicit live capability wins. Older Whisper runners need both timestamp
 * mechanisms; Granite Plus's file-only word instruction is not live metadata. */
export function realtimeEnrichment(caps?: RealtimeTranscriptionCaps, times: readonly string[] = []): boolean {
  return caps ? caps.supported && caps.enrichment : times.includes('segment') && times.includes('word')
}
export function audioPolicy(lanes: AudioLane[], transcribers: number, docParser = false) {
  const audioOk = lanes.length > 0 && lanes.every(l => l.audio)
  const audioMode = audioOk && !lanes.every(l => l.chat)
  const liveBlocked = audioMode && lanes.some(l => !l.live)
  // Lanes that chat AND hear a stream (Nemotron 3.5 Transcribe, the generative
  // ASR families) can take the live turn too - spoken into the thread while the
  // text box stays - but a clip stays their default: Live comes second.
  const live: MicMode[] = audioOk && !audioMode && lanes.every(l => l.live) ? ['live'] : []
  const jobs: MicMode[] = docParser ? [] : !audioOk ? (transcribers ? ['dictate'] : [])
    : audioMode ? (liveBlocked ? ['record'] : ['live', 'record'])
      : lanes.length > 1 ? ['record', ...live]
        : transcribers ? ['record', ...live, 'dictate'] : ['record', ...live]
  return { audioOk, audioMode, liveBlocked, jobs }
}
