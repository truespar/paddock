import { describe, it, expect } from 'vitest'
import { mergeDiarization } from './diarization'
import type { TranscriptMeta, SpeakerTimeline, TranscriptWord } from '@/types/chat'

describe('speaker metadata contract', () => {
  const meta: TranscriptMeta = { words: [{ word: 'Hello!', start: 0, end: 1, confidence: .7 }], language: 'en' }
  const result: SpeakerTimeline & {words: TranscriptWord[]} = {
    model: 'Nemotron', duration: 3, preset: 'offline', attribution: 'time_overlap_v1',
    segments: [{speaker: 0,start:0,end:2}],
    words: [{...meta.words![0], speaker:0,speakers:[0],speaker_status:'assigned'}],
  }
  it('persists timeline and original recognition fields with a single word list', () => {
    const merged = mergeDiarization(meta, result)
    expect(merged.words![0]).toMatchObject(meta.words![0])
    expect(merged.diarization).not.toHaveProperty('words')
    expect(merged.language).toBe('en')
    expect(JSON.parse(JSON.stringify(merged))).toEqual(merged)
    expect(meta).not.toHaveProperty('diarization')
  })
  it('rejects changed text/timing and invalid speaker intervals', () => {
    for (const bad of [
      {...result,words:[]}, {...result,words:[{word:'Oops',start:0,end:1}]},
      {...result,words:[{...result.words[0],end:2}]}, {...result,duration:NaN},
      {...result,segments:[{speaker:8,start:0,end:1}]},
      {...result,segments:[{speaker:0,start:0,end:4}]},
    ]) expect(() => mergeDiarization(meta,bad)).toThrow()
  })
})
