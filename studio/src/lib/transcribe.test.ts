import { afterEach, describe, expect, it, vi } from 'vitest'
import { timelineOf, transcribeClip } from './transcribe'
import { askedLanguage, languageOptions, LANGUAGE_AUTO } from './languages'

/** One streamed answer as the runner sends it: a delta, then the done event. */
function sse(done: Record<string, unknown>): Response {
  const body = [
    `data: ${JSON.stringify({ type: 'transcript.text.delta', delta: done.text })}`,
    `data: ${JSON.stringify({ type: 'transcript.text.done', ...done })}`,
    'data: [DONE]',
    '',
  ].join('\n\n')
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

afterEach(() => vi.unstubAllGlobals())

describe('a lane that times and attributes its own words (Nemotron 3.5 Transcribe)', () => {
  const done = {
    text: 'Hello there. Hi.',
    usage: { input_tokens: 40, output_tokens: 5 },
    paddock_verbose: {
      language: 'en',
      duration: 3,
      words: [
        {
          word: 'Hello', start: 0.3, end: 0.6, paddock_confidence: 0.9, paddock_alt: 'Hallo',
          paddock_speaker: 0, paddock_speakers: [0], paddock_speaker_status: 'assigned',
          paddock_speaker_coverage: [1, 0, 0, 0, 0, 0, 0, 0],
        },
        // timed, no confidence (the decode's word did not match): kept
        { word: 'there.', start: 0.7, end: 1.0, paddock_speaker: 0, paddock_speakers: [0], paddock_speaker_status: 'assigned' },
        { word: 'Hi.', start: 1.1, end: 1.3, paddock_confidence: 0.8, paddock_speaker: null, paddock_speakers: [], paddock_speaker_status: 'unassigned' },
      ],
      paddock_words: [{ word: 'Hello', confidence: 0.9 }],
      paddock_diarization: {
        model: 'transcribe', duration: 3, preset: 'transcribe', attribution: 'time_overlap_v1',
        segments: [{ speaker: 0, start: 0.25, end: 1.02 }, { speaker: 1, start: 1.5, end: 2.0 }],
      },
    },
  }

  it('keeps every timed word, with its speaker, and the timeline beside them', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => sse(done)))
    const out = await transcribeClip(new Blob(['x']), 'a.wav', { url: '/x', wordTimes: true, wordConfidence: true })
    const w = out.meta.words ?? []
    expect(w.map((x) => x.word)).toEqual(['Hello', 'there.', 'Hi.'])
    expect(w[0]).toMatchObject({ confidence: 0.9, start: 0.3, end: 0.6, alt: 'Hallo', speaker: 0, speakers: [0], speaker_status: 'assigned' })
    expect(w[1].confidence).toBeUndefined()
    expect(w[1].start).toBe(0.7)
    expect(w[2]).toMatchObject({ speaker: null, speakers: [], speaker_status: 'unassigned' })
    expect(out.meta.diarization).toEqual({
      model: 'transcribe', duration: 3, preset: 'transcribe', attribution: 'time_overlap_v1',
      segments: [{ speaker: 0, start: 0.25, end: 1.02 }, { speaker: 1, start: 1.5, end: 2.0 }],
    })
  })

  it('takes no timeline under an attribution it does not show', () => {
    expect(timelineOf({ attribution: 'something_else', duration: 3, segments: [] }, 'm')).toBeUndefined()
    expect(timelineOf({ attribution: 'time_overlap_v1', duration: 0, segments: [] }, 'm')).toBeUndefined()
  })

  it('still drops a word that has neither a time nor a confidence', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => sse({ text: 'a', paddock_verbose: { words: [{ word: 'a' }] } })))
    const out = await transcribeClip(new Blob(['x']), 'a.wav', { url: '/x' })
    expect(out.meta.words).toBeUndefined()
  })
})

describe('the language picker follows the lane', () => {
  const forty = ['de', 'en', 'sv']
  it('offers the lane its own set, detection first', () => {
    const values = languageOptions(forty).map((o) => o.value)
    expect(values[0]).toBe(LANGUAGE_AUTO)
    expect(values.slice(1).sort()).toEqual(forty)
    expect(languageOptions().length).toBeGreaterThan(90)
  })
  it('sends detection for a choice the lane cannot take', () => {
    expect(askedLanguage('yo', forty)).toBeUndefined()
    expect(askedLanguage('sv', forty)).toBe('sv')
    expect(askedLanguage(LANGUAGE_AUTO, forty)).toBeUndefined()
    expect(askedLanguage('yo')).toBe('yo')
  })
})
