<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import type { TranscriptMeta } from '@/types/chat'
import { useModelsStore } from '@/stores/models'
import { useChatStore } from '@/stores/chat'
import { attachmentsApi } from '@/lib/api'
import { identifySpeakers } from '@/lib/diarization'
import { clock } from '@/lib/subtitles'

const props = defineProps<{ transcript: TranscriptMeta; clipId?: string; messageId: string; streaming?: boolean }>()
const emit = defineEmits<{ seek: [seconds: number] }>()
const models = useModelsStore()
const chat = useChatStore()
const lanes = computed(() => models.models.filter(m => m.kind === 'diarizer' && m.status === 'ok' && m.port))
const selected = ref('')
const busy = ref(false)
const error = ref('')
let controller: AbortController | undefined
onBeforeUnmount(() => controller?.abort())
const timeline = computed(() => props.transcript.diarization)
const cursors = ref<Record<number, number>>({})
watch(timeline, () => { cursors.value = {} })
// At most eight SVG paths, not one DOM node per 10-ms segment. Accessibility
// uses three keyboard controls per lane, independent of the segment count.
const rows = computed(() => {
  const t = timeline.value
  if (!t || !(t.duration > 0)) return []
  return [...new Set(t.segments.map(s => s.speaker))].sort((a,b) => a-b).map(speaker => {
    const spans = t.segments.filter(s => s.speaker === speaker)
    return { speaker, spans, path: spans.map(s => {
      const x = s.start / t.duration * 1000, w = (s.end - s.start) / t.duration * 1000
      return `M${x} 3h${Math.max(0.5,w)}v14h-${Math.max(0.5,w)}z`
    }).join(' ') }
  })
})
function step(row: typeof rows.value[number], delta: number): void {
  const i = Math.max(0, Math.min(row.spans.length - 1, (cursors.value[row.speaker] ?? 0) + delta))
  cursors.value[row.speaker] = i
  if (row.spans[i]) emit('seek', row.spans[i].start)
}
async function run(): Promise<void> {
  const lane = lanes.value.find(m => `${m.port}` === selected.value) ?? lanes.value[0]
  const conv = chat.active, meta = props.transcript, id = props.clipId
  if (!conv || !id || !lane?.port || busy.value) return
  busy.value = true
  error.value = ''
  controller = new AbortController()
  try {
    const audio = await fetch(attachmentsApi.url(id), { signal: controller.signal })
    if (!audio.ok) throw new Error(`Audio attachment: HTTP ${audio.status}`)
    const out = await identifySpeakers(lane.port, lane.id, await audio.blob(), meta, controller.signal)
    const message = conv.messages.find(m => m.id === props.messageId)
    if (controller.signal.aborted || !message || message.transcript !== meta || message.streaming) return
    message.transcript = out
    chat.persistNow(conv)
  } catch (e) {
    if (!controller.signal.aborted) error.value = e instanceof Error ? e.message : String(e)
  } finally { busy.value = false }
}
function seek(event: MouseEvent): void {
  const rect = (event.currentTarget as Element).getBoundingClientRect()
  emit('seek', Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1,rect.width))) * (timeline.value?.duration ?? 0))
}
</script>

<template>
  <section class="speakers" aria-label="Speaker timeline">
    <div v-if="timeline" class="speakers__timeline">
      <header><strong>Speakers</strong><span>{{ clock(timeline.duration) }}</span></header>
      <span v-if="!rows.length">No speech detected</span>
      <div v-for="row in rows" :key="row.speaker" class="speakers__row">
        <button @click="step(row,0)">Speaker {{ row.speaker + 1 }}</button>
        <button :aria-label="`Previous interval for Speaker ${row.speaker + 1}`" :disabled="!(cursors[row.speaker] ?? 0)" @click="step(row,-1)">‹</button>
        <button :aria-label="`Next interval for Speaker ${row.speaker + 1}`" :disabled="(cursors[row.speaker] ?? 0) + 1 >= row.spans.length" @click="step(row,1)">›</button>
        <svg viewBox="0 0 1000 20" preserveAspectRatio="none" aria-hidden="true" @click="seek"><path :d="row.path" /></svg>
      </div>
    </div>
    <div v-if="clipId && !streaming" class="speakers__actions">
      <select v-if="lanes.length > 1" v-model="selected" aria-label="Speaker model" :disabled="busy">
        <option value="">{{ lanes[0]?.display ?? lanes[0]?.id }}</option>
        <option v-for="lane in lanes.slice(1)" :key="lane.port" :value="String(lane.port)">{{ lane.display ?? lane.id }}</option>
      </select>
      <button v-if="lanes.length" class="pk-btn pk-btn--sm" :disabled="busy" @click="run">{{ busy ? 'Identifying speakers…' : 'Identify speakers' }}</button>
      <button v-if="busy" class="pk-btn pk-btn--sm" @click="controller?.abort()">Cancel</button>
      <RouterLink v-if="!lanes.length && !timeline" :to="{ name: 'server-new-config', params: { model: 'nemotron-3-diarization' } }">Start a speaker model</RouterLink>
    </div>
    <p v-if="error" role="alert">{{ error }}</p>
  </section>
</template>

<style scoped>
.speakers { display: grid; gap: 8px; margin-bottom: 12px; font-size: var(--pk-font-size-xs); }
.speakers__timeline { display: grid; gap: 8px; padding: 12px; border-radius: var(--pk-radius-md); background: var(--pk-bg-inset); }
header, .speakers__actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
header { justify-content: space-between; }
.speakers__row { display: grid; grid-template-columns: 70px 16px 16px minmax(0,1fr); align-items: center; gap: 8px; }
.speakers__row button { background: none; border: 0; color: inherit; font: inherit; cursor: pointer; padding: 0; }
.speakers__row button:disabled { opacity: .35; cursor: default; }
select { min-width: 0; max-width: 240px; color: inherit; background: var(--pk-bg-surface); border: 0; font: inherit; }
svg { width: 100%; height: 20px; cursor: pointer; background: color-mix(in srgb,currentColor 4%,transparent); border-radius: 4px; }
path { fill: currentColor; opacity: .55; }
p { margin: 0; color: var(--pk-text-danger); }
</style>
