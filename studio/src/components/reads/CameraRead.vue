<script setup lang="ts">
// Reads, live: the webcam's frames asked the same questions over and over,
// the answers updating as they land - a decision model watching. One request
// is in flight at a time and each takes the frame of the moment it is sent
// (the newest frame wins; nothing queues behind a slow answer), so the rate is
// the model's own. Frames are drawn down to the chosen long side before they
// go: the model's cost follows the picture's tokens, and a camera frame
// rarely needs more than 512 px for the questions people ask of it. Nothing
// is kept - a frame enters the read's history only when the user keeps it.
// The preview is mirrored, as a camera facing the user is; the frame sent is
// not, so text held up to the camera reads the right way round.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import Icon from '@/components/Icon.vue'
import Popover from '@/components/ui/Popover.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { answerLabel, fmtP, type ReadRequest, type ReadResponse } from '@/lib/reads'

export interface LiveFrame {
  frame: string
  request: ReadRequest
  response: ReadResponse
  ms: number
}

const props = defineProps<{
  port: number
  /** the request for one frame: the page's text, questions and settings */
  buildBody: (frame: string) => ReadRequest
  /** the page cannot run (no reader, or questions that do not validate) */
  blocked: boolean
  /** what to do about it, in a sentence - shown in the bar while the read
   *  cannot start, and Start takes the user there instead of greying out */
  blockedReason?: string
  /** the page's questions in order: one chip on the picture each */
  rows: { id: string; text: string }[]
}>()
const emit = defineEmits<{
  (e: 'result', r: LiveFrame): void
  (e: 'error', message: string): void
  (e: 'keep', r: LiveFrame): void
  (e: 'live', on: boolean): void
  (e: 'close'): void
  /** the user clicked the reason: take them to what needs doing */
  (e: 'need'): void
}>()

const video = ref<HTMLVideoElement | null>(null)
const status = ref<'starting' | 'ready' | 'denied' | 'none'>('starting')
const statusText = ref('')
const live = ref(false)
const last = ref<LiveFrame | null>(null)
/** decisions a second, from a smoothed gap between answers */
const rate = ref(0)
let gapEma = 0
let lastDone = 0

let stream: MediaStream | null = null
let inflight: AbortController | null = null
const devices = ref<MediaDeviceInfo[]>([])
const deviceId = ref('')
const deviceOptions = computed<SelectOption[]>(() =>
  devices.value.map((d, i) => ({ value: d.deviceId, label: d.label || `Camera ${i + 1}` })),
)

// the long side a frame is drawn to before it goes
const side = ref(512)
const sideOptions: SelectOption[] = [
  { value: 384, label: '384 px', hint: 'fastest' },
  { value: 512, label: '512 px' },
  { value: 768, label: '768 px' },
  { value: 1024, label: '1024 px', hint: 'finest detail' },
]

async function open(id?: string): Promise<void> {
  close()
  status.value = 'starting'
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: id ? { deviceId: { exact: id } } : { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    })
  } catch (e) {
    const name = e instanceof DOMException ? e.name : ''
    status.value = name === 'NotFoundError' || name === 'OverconstrainedError' ? 'none' : 'denied'
    statusText.value =
      status.value === 'none'
        ? 'No camera was found.'
        : 'The camera was not allowed. Allow it for this page in the browser and try again.'
    stopLive()
    return
  }
  const el = video.value
  if (el) {
    el.srcObject = stream
    await el.play().catch(() => {})
  }
  // labels are only readable once a camera has been allowed
  try {
    devices.value = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
  } catch {
    devices.value = []
  }
  deviceId.value = stream.getVideoTracks()[0]?.getSettings().deviceId ?? id ?? ''
  status.value = 'ready'
}

function close(): void {
  inflight?.abort()
  inflight = null
  for (const t of stream?.getTracks() ?? []) t.stop()
  stream = null
}

/** The current frame, drawn to `side` on its long side (never up), as a JPEG
 *  data URL; null while the camera has no picture yet. */
function capture(): string | null {
  const el = video.value
  if (!el || !el.videoWidth || !el.videoHeight) return null
  const k = Math.min(1, side.value / Math.max(el.videoWidth, el.videoHeight))
  const c = document.createElement('canvas')
  c.width = Math.max(1, Math.round(el.videoWidth * k))
  c.height = Math.max(1, Math.round(el.videoHeight * k))
  const g = c.getContext('2d')
  if (!g) return null
  g.drawImage(el, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.9)
}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms))

/** One frame through the endpoint. False when the loop should stop. */
async function decide(): Promise<boolean> {
  const frame = capture()
  if (!frame) {
    await sleep(100)
    return true
  }
  const request = props.buildBody(frame)
  const ctrl = new AbortController()
  inflight = ctrl
  const t0 = performance.now()
  let res: Response
  try {
    res = await fetch(`/api/runners/${props.port}/v1/systemone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: ctrl.signal,
    })
  } catch (e) {
    if (ctrl.signal.aborted) return false
    emit('error', e instanceof Error ? e.message : String(e))
    return false
  } finally {
    if (inflight === ctrl) inflight = null
  }
  const json: unknown = await res.json().catch(() => null)
  if (ctrl.signal.aborted) return false
  const ms = Math.round(performance.now() - t0)
  if (!res.ok) {
    emit('error', (json as { error?: { message?: string } } | null)?.error?.message ?? `HTTP ${res.status}`)
    return false
  }
  const r: LiveFrame = { frame, request, response: json as ReadResponse, ms }
  last.value = r
  const now = performance.now()
  if (lastDone) {
    const gap = now - lastDone
    gapEma = gapEma ? gapEma * 0.7 + gap * 0.3 : gap
    rate.value = 1000 / gapEma
  }
  lastDone = now
  emit('result', r)
  return true
}

async function loop(): Promise<void> {
  while (live.value) {
    // a hidden tab asks nothing: the GPU is not spent on frames nobody sees
    if (document.hidden || status.value !== 'ready' || props.blocked) {
      await sleep(250)
      continue
    }
    if (!(await decide())) {
      stopLive()
      return
    }
  }
}

function startLive(): void {
  if (live.value || status.value !== 'ready' || props.blocked) return
  live.value = true
  gapEma = 0
  lastDone = 0
  rate.value = 0
  emit('live', true)
  void loop()
}
function stopLive(): void {
  if (!live.value) return
  live.value = false
  inflight?.abort()
  inflight = null
  emit('live', false)
}
function toggle(): void {
  if (live.value) stopLive()
  // nothing to ask yet: Start goes to the question instead of doing nothing
  else if (props.blocked) emit('need')
  else startLive()
}
function keep(): void {
  if (last.value) emit('keep', last.value)
}
function shut(): void {
  stopLive()
  close()
  emit('close')
}

watch(deviceId, (id, old) => {
  if (old && id && id !== old) {
    const was = live.value
    stopLive()
    void open(id).then(() => {
      if (was) startLive()
    })
  }
})
onMounted(() => void open())
onUnmounted(() => {
  stopLive()
  close()
})

const tokens = computed(() => last.value?.response.diagnostics.pictures?.[0]?.tokens)
/** The newest frame's answers in the page's question order, one card on the
 *  picture each. A yes/no answer is coloured by what it says (`tone`), so a
 *  flip reads from across the room; a choice or a score is named in white.
 *  The number and the bar are the confidence the rest of Reads shows. */
const liveAnswers = computed(() => {
  const answers = last.value?.response.answers ?? {}
  return props.rows.map((r) => {
    const a = answers[r.id] ?? null
    const tone = !a ? 'none' : a.type === 'noul' ? (a.noul >= 0.5 ? 'yes' : 'no') : 'pick'
    return { id: r.id, text: r.text, label: a ? answerLabel(a) : null, confidence: a?.confidence, tone }
  })
})
// A card flashes once when its answer changes - the moment someone watching
// is waiting for. Keyed by question id; the flash clears itself.
const flipped = ref<Set<string>>(new Set())
const seen = new Map<string, string>()
watch(liveAnswers, (list) => {
  for (const a of list) {
    if (a.label === null) continue
    const was = seen.get(a.id)
    seen.set(a.id, a.label)
    if (was === undefined || was === a.label) continue
    flipped.value = new Set(flipped.value).add(a.id)
    window.setTimeout(() => {
      const next = new Set(flipped.value)
      next.delete(a.id)
      flipped.value = next
    }, 700)
  }
})
const reasonShown = computed(
  () => status.value === 'ready' && props.blocked && !live.value && !!props.blockedReason,
)
const pill = computed(() => {
  if (status.value === 'starting') return 'starting'
  if (status.value !== 'ready') return 'off'
  if (live.value && props.blocked) return 'waiting'
  return live.value ? 'live' : 'paused'
})
</script>

<template>
  <section class="cam" aria-label="Camera">
    <div class="cam__view">
      <video ref="video" class="cam__video" autoplay playsinline muted />
      <div v-if="status !== 'ready'" class="cam__cover">
        <Icon :name="status === 'starting' ? 'spinner' : 'webcam'" :size="22" :class="{ spin: status === 'starting' }" />
        <p>{{ status === 'starting' ? 'Opening the camera...' : statusText }}</p>
        <button v-if="status !== 'starting'" class="pk-btn pk-btn--sm" type="button" @click="open()">
          Try again
        </button>
      </div>
      <ul v-else class="cam__hud" aria-label="Answers for the newest frame">
        <li
          v-for="a in liveAnswers"
          :key="a.id"
          class="cam__card"
          :class="[`cam__card--${a.tone}`, { 'cam__card--flip': flipped.has(a.id) }]"
        >
          <span class="cam__cardq">{{ a.text }}</span>
          <span class="cam__cardrow">
            <span class="cam__carda">{{ a.label ?? '-' }}</span>
            <span v-if="a.label !== null" class="cam__cardp">{{ fmtP(a.confidence) }}</span>
          </span>
          <span class="cam__bar2" aria-hidden="true">
            <span class="cam__bar2fill" :style="{ width: `${Math.round((a.confidence ?? 0) * 100)}%` }" />
          </span>
        </li>
      </ul>
      <span class="cam__pill" :class="`cam__pill--${pill}`">
        <span class="cam__dot" />{{ pill }}<span v-if="live && rate" class="cam__rate">{{ rate.toFixed(1) }}/s</span>
      </span>
    </div>

    <div class="cam__bar">
      <button
        class="pk-btn"
        :class="{ 'pk-btn--primary': !live }"
        type="button"
        :disabled="status !== 'ready'"
        @click="toggle"
      >
        <Icon :name="live ? 'pause' : 'play'" :size="14" /> {{ live ? 'Stop' : 'Start' }}
      </button>
      <Tooltip label="Keep this frame and its answers in the read's history">
        <button class="pk-btn" type="button" :disabled="!last" @click="keep">
          <Icon name="save" :size="14" /> Snapshot
        </button>
      </Tooltip>
      <span v-if="reasonShown" class="cam__reason">{{ blockedReason }}</span>
      <span class="cam__stats">
        <template v-if="last">
          {{ last.ms }} ms<template v-if="tokens"> · {{ tokens }} image tokens</template>
        </template>
      </span>
      <Popover align="end">
        <template #trigger>
          <button class="pk-icon-btn cam__icon" type="button" aria-label="Camera settings">
            <Icon name="settings" :size="16" />
          </button>
        </template>
        <div class="cam__settings">
          <label class="cam__field">
            <span>Frame</span>
            <Select :model-value="side" :options="sideOptions" @update:model-value="side = Number($event)" />
          </label>
          <label v-if="deviceOptions.length > 1" class="cam__field">
            <span>Camera</span>
            <Select
              :model-value="deviceId"
              :options="deviceOptions"
              @update:model-value="deviceId = String($event)"
            />
          </label>
        </div>
      </Popover>
      <Tooltip label="Close the camera">
        <button class="pk-icon-btn cam__icon" type="button" aria-label="Close the camera" @click="shut">
          <Icon name="x" :size="16" />
        </button>
      </Tooltip>
    </div>
  </section>
</template>

<style scoped>
.cam {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
/* The picture is the page: full width, the answers riding on it. It is black
   and stays black in both themes - it is a video frame, not a surface - so
   the chips on it carry their own fixed contrast. */
.cam__view {
  position: relative;
  border-radius: var(--pk-radius-lg);
  overflow: hidden;
  background: #000;
  aspect-ratio: 16 / 9;
  max-height: 62vh;
}
.cam__video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  /* a camera facing the user is seen mirrored; the frame sent is not */
  transform: scaleX(-1);
}
.cam__cover {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px;
  text-align: center;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  background: var(--pk-bg-surface);
}
.cam__cover p {
  margin: 0;
  max-width: 36ch;
}
/* The answers ride along the bottom of the picture, subtitle-style, big
   enough to read from a step back: the question small, the answer large. Over
   a video frame the colours are fixed rather than themed - the frame is not a
   surface either theme paints. */
.cam {
  --cam-yes: #3ddc84;
  --cam-no: #ff6b6b;
  --cam-pick: #ffffff;
}
.cam__hud {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 12px;
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
  pointer-events: none;
}
.cam__card {
  --tone: rgba(255, 255, 255, 0.6);
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 0 1 260px;
  min-width: 170px;
  padding: 10px 14px 12px;
  border-radius: 12px;
  background: rgba(10, 12, 16, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #fff;
  backdrop-filter: blur(6px);
}
.cam__card--yes {
  --tone: var(--cam-yes);
}
.cam__card--no {
  --tone: var(--cam-no);
}
.cam__card--pick {
  --tone: var(--cam-pick);
}
.cam__cardq {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: rgba(255, 255, 255, 0.78);
  font-size: var(--pk-font-size-sm);
}
.cam__cardrow {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}
.cam__carda {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--tone);
  font-size: clamp(1.4rem, 2.6vw, 2.1rem);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.cam__card--none .cam__carda {
  color: rgba(255, 255, 255, 0.45);
  font-weight: 600;
}
.cam__cardp {
  flex: none;
  font-family: var(--pk-font-mono, monospace);
  font-size: 1rem;
  font-variant-numeric: tabular-nums;
  color: rgba(255, 255, 255, 0.8);
}
.cam__bar2 {
  height: 5px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.16);
  overflow: hidden;
}
.cam__bar2fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--tone);
  transition: width 0.2s ease-out;
}
/* the flip: the card lights up in its new answer's colour, once */
.cam__card--flip {
  animation: cam-flip 0.7s ease-out;
}
@keyframes cam-flip {
  0% {
    background: color-mix(in srgb, var(--tone) 55%, rgba(10, 12, 16, 0.78));
    transform: scale(1.04);
  }
  100% {
    background: rgba(10, 12, 16, 0.78);
    transform: scale(1);
  }
}
@media (prefers-reduced-motion: reduce) {
  .cam__card--flip {
    animation: none;
    box-shadow: 0 0 0 2px var(--tone);
  }
  .cam__bar2fill {
    transition: none;
  }
}
.cam__pill {
  position: absolute;
  top: 10px;
  right: 10px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: var(--pk-font-size-xs, 0.72rem);
  font-family: var(--pk-font-mono, monospace);
  color: #fff;
  background: rgba(10, 12, 16, 0.74);
}
.cam__rate {
  color: rgba(255, 255, 255, 0.72);
}
.cam__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #9a9a9a;
}
.cam__pill--live .cam__dot {
  background: #3ecf6e;
}
.cam__pill--waiting .cam__dot {
  background: #e3b341;
}
.cam__bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.cam__reason {
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-sm);
}
.cam__stats {
  margin-left: auto;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  font-variant-numeric: tabular-nums;
}
.cam__icon {
  width: 32px;
  height: 32px;
}
.cam__settings {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px;
}
.cam__field {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
</style>
