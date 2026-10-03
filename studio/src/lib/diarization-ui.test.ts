import { describe, it, expect, vi } from 'vitest'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import SpeakerTimeline from '@/components/chat/SpeakerTimeline.vue'
vi.mock('@/stores/models', () => ({ useModelsStore: () => ({ models: [] }) }))
vi.mock('@/stores/chat', () => ({ useChatStore: () => ({ active: undefined }) }))

describe('speaker timeline rendering', () => {
  it('keeps overlap and a bounded DOM for thousands of intervals', async () => {
    const segments = Array.from({length:12000},(_,i) => ({speaker:i%2,start:Math.floor(i/2)*.02,end:Math.floor(i/2)*.02+.01}))
    const html = await renderToString(createSSRApp(SpeakerTimeline, {
      messageId:'speech', transcript:{diarization:{model:'Nemotron',duration:120,preset:'offline',attribution:'time_overlap_v1',segments}},
    }))
    expect(html.match(/<svg/g)).toHaveLength(2)
    expect(html.match(/<path/g)).toHaveLength(2)
    expect(html.match(/<button/g)).toHaveLength(6)
    expect(html).toContain('Next interval for Speaker 2')
    expect(html).not.toContain('<option')
  })
})
