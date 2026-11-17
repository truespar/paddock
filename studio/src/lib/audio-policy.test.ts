import { describe, expect, it } from 'vitest'
import { audioPolicy, realtimeEnrichment, type RealtimeTranscriptionCaps } from './audio-policy'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/speech-capabilities.json'
const speechCases = fixtures as {name:string;caps:{timestamp_granularities?:string[];realtime_transcription?:RealtimeTranscriptionCaps};enrich:boolean}[]
const text = { audio: false, chat: true, live: false }
const whisper = { audio: true, chat: false, live: true }
const generative = { audio: true, chat: true, live: true }
const cloud = { ...whisper, live: false }
describe('shared web/native audio arming policy', () => {
  it.each(speechCases)('uses the shared live capability contract: $name', ({caps, enrich}) => {
    expect(realtimeEnrichment(caps.realtime_transcription, caps.timestamp_granularities)).toBe(enrich)
  })
  it.each([
    [[text], 0, false, []], [[text], 1, false, ['dictate']],
    [[whisper], 1, true, ['live', 'record']], [[generative], 1, false, ['record', 'live', 'dictate']],
    [[generative], 0, false, ['record', 'live']],
    [[{ ...generative, live: false }], 1, false, ['record', 'dictate']],
    [[whisper, generative], 2, true, ['live', 'record']],
    [[generative, generative], 2, false, ['record', 'live']],
    [[text, generative], 1, false, ['dictate']],
    [[whisper, cloud], 1, true, ['record']], [[cloud], 0, true, ['record']],
  ] as const)('resolves %j with %i transcribers', (lanes, ears, mode, jobs) => {
    expect(audioPolicy([...lanes], ears)).toMatchObject({ audioMode: mode, jobs: [...jobs] })
  })
  it('never offers speech input to a document parser', () => {
    expect(audioPolicy([generative], 1, true).jobs).toEqual([])
  })
  it('keeps file-only lanes in Compare and explains live unavailability', () => {
    expect(audioPolicy([whisper, cloud], 1)).toMatchObject({ audioOk: true, liveBlocked: true })
  })
})
