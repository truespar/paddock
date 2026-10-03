import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { takesTurns, useModelsStore } from '@/stores/models'
import { servedId } from '@/stores/fleet'

describe('speaker diarization is not chat or transcription', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { matchMedia: () => ({ matches: false }) })
    setActivePinia(createPinia())
  })
  it('preserves fleet identity and excludes speech/text composers', () => {
    expect(servedId({ diarization: 'Nemotron-3-Diarization-MLX' })).toBe('Nemotron-3-Diarization-MLX')
    const models = useModelsStore()
    models.models = [{ id: 'diarizer', ownedBy: 'paddock', kind: 'diarizer', status: 'ok', port: 11963 }]
    expect(takesTurns('diarizer')).toBe(false)
    expect(models.canChat('diarizer')).toBe(false)
    expect(models.canTranscribe('diarizer')).toBe(false)
  })
})
