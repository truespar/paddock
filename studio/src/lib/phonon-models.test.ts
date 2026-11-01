import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { takesTurns, useModelsStore } from '@/stores/models'
import { servedId } from '@/stores/fleet'

describe('native speech generation discovery', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { matchMedia: () => ({ matches: false }) })
    setActivePinia(createPinia())
  })
  it('keeps its identity without offering chat or transcription', () => {
    expect(servedId({ speech: 'phonon-31m-en' })).toBe('phonon-31m-en')
    const models = useModelsStore()
    models.models = [{ id: 'phonon-31m-en', ownedBy: 'paddock', kind: 'synthesizer', status: 'ok', port: 45971 }]
    expect(takesTurns('synthesizer')).toBe(false)
    expect(models.canChat('phonon-31m-en')).toBe(false)
    expect(models.canTranscribe('phonon-31m-en')).toBe(false)
  })
})
