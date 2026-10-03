import type { SpeakerTimeline, TranscriptMeta, TranscriptWord } from '@/types/chat'

/** No client-side re-attribution: both Studios use the same Rust contract. */
export function mergeDiarization(
  meta: TranscriptMeta,
  result: SpeakerTimeline & { words: TranscriptWord[] },
): TranscriptMeta {
  const original = meta.words ?? []
  if (result.attribution !== 'time_overlap_v1' || !Number.isFinite(result.duration) ||
    result.duration <= 0 || result.duration > 600 || !Array.isArray(result.segments) ||
    result.segments.some(s => !Number.isInteger(s.speaker) || s.speaker < 0 || s.speaker >= 8 ||
      !Number.isFinite(s.start) || !Number.isFinite(s.end) || s.start < 0 || s.end <= s.start ||
      s.end > result.duration + 0.001) || !Array.isArray(result.words) ||
    result.words.length !== original.length || result.words.some((w, i) =>
      w.word !== original[i].word || (w.start ?? null) !== (original[i].start ?? null) ||
      (w.end ?? null) !== (original[i].end ?? null))) {
    throw new Error('Speaker analysis did not match this transcript')
  }
  const { words, ...diarization } = result
  return { ...meta, words, diarization }
}

export async function identifySpeakers(
  port: number, model: string, audio: Blob, meta: TranscriptMeta, signal: AbortSignal,
): Promise<TranscriptMeta> {
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid speaker endpoint')
  if (audio.size > 25 * 1024 * 1024) throw new Error('Speaker analysis accepts audio files up to 25 MiB')
  const form = new FormData()
  form.append('file', audio, 'recording')
  form.append('model', model)
  form.append('preset', 'offline')
  form.append('words', JSON.stringify(meta.words ?? []))
  const res = await fetch(`/api/runners/${port}/v1/audio/diarizations`, { method: 'POST', body: form, signal })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error?.message ?? `Speaker analysis: HTTP ${res.status}`)
  }
  return mergeDiarization(meta, await res.json())
}
