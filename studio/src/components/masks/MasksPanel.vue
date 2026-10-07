<script setup lang="ts">
// Masks - find every object in a picture that matches a few words or a box
// you draw, or cut out the one object you click on, with a running
// promptable-segmentation model (SAM 3). Each prompt is a layer with its own
// colours, so "car" and "wheel" show together; the instances come back as
// masks drawn over the picture, with their scores, and leave as COCO JSON, a
// mask PNG or a cut-out. Everything runs through the same POST /v1/masks your
// code calls (relayed by the manager, which holds the runner key). The
// picture goes up byte-identical on every prompt, so the runner reuses its
// encoding: a second prompt on the same picture costs only the prompt, and a
// click a few milliseconds.
//
// An object layer asks again on every click. Each answer carries a
// `refine_id`; the next request sends it back when it only adds clicks, so
// the model builds on its last mask the way Meta's own notebook does.
//
// Camera tracks the prompts' words through the webcam with a video session
// (useMaskCamera), when the endpoint serves them: the same stage and side
// cards as a picture, the frame in the stage, the concepts and the tracked
// objects beside it. Both modes edit the same prompts, as Reads' text and
// camera modes ask the same questions.
//
// The pictures worked on are the page's history (/api/mask-history), in the
// side panel the way Tables and Reads list theirs: a record is a picture, its
// prompts with what each found, and the snapshots kept while it was open.
// /studio/masks/<id> names the record on screen.
import { computed, markRaw, nextTick, onMounted, onUnmounted, ref, shallowReactive, shallowRef, watch } from 'vue'
import { onBeforeRouteLeave, onBeforeRouteUpdate, useRoute, useRouter, type LocationQuery } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useModelsStore } from '@/stores/models'
import { useMasksStore } from '@/stores/masks'
import { useToastsStore } from '@/stores/toasts'
import { copyText } from '@/lib/clipboard'
import {
  capsFrom,
  clickBody,
  cocoJson,
  curlFor,
  cutoutPng,
  decodeRle,
  instanceRgb,
  layerHue,
  maskPng,
  paintOverlay,
  refineFor,
  requestBody,
  rgbCss,
  saveBlob,
  slug,
  type ClickAsk,
  type DrawnMask,
  type MaskCaps,
  type MaskResponse,
  type PromptBox,
  type PromptPoint,
} from '@/lib/masks'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { objectRgb, shotCanvas, useMaskCamera, type CamConcept, type CamFrame } from '@/composables/useMaskCamera'
import { usePanelFold } from '@/composables/usePanelFold'
import { MAX_SNAPSHOTS, pictureKey, type MaskDoc, type MaskLayerDoc } from '@/lib/mask-history'
import { uuid } from '@/lib/uuid'
import ReadsSidebar from '@/components/reads/ReadsSidebar.vue'
import Checkbox from '@/components/ui/Checkbox.vue'
import Select from '@/components/ui/Select.vue'
import Slider from '@/components/ui/Slider.vue'
import TextInput from '@/components/ui/TextInput.vue'
import ToggleGroup from '@/components/ui/ToggleGroup.vue'
import ToggleGroupItem from '@/components/ui/ToggleGroupItem.vue'

const models = useModelsStore()
const route = useRoute()
const toasts = useToastsStore()

// keep the list fresh while the page is open (a model started or stopped in
// the Manager appears without a reload)
let timer: number | undefined
onMounted(() => {
  void models.refresh()
  timer = window.setInterval(() => void models.refresh(), 5000)
  window.addEventListener('paste', onPaste)
})
onUnmounted(() => {
  clearInterval(timer)
  window.removeEventListener('paste', onPaste)
})

// -- the endpoint ----------------------------------------------------------------
// The header's picker chooses it (the store), as on Reads and Tables.
const maskers = computed(() => models.models.filter((m) => m.kind === 'masker'))
// ?port= (the model page's "open in Studio") names the endpoint; it wins once
// that endpoint is in the list - the fleet loads after the page does (once:
// the choice outlives the page in the store, so a later visit is not dragged
// back to the link that opened an earlier one)
const wanted = Number(route.query.port) || 0
let wantedTaken = false
const history = useMasksStore()
const { port } = storeToRefs(history)
watch(
  maskers,
  (list) => {
    if (wanted && !wantedTaken && list.some((m) => m.port === wanted)) {
      port.value = wanted
      wantedTaken = true
    } else if (list.length && !list.some((m) => m.port === port.value)) {
      port.value = list[0].port ?? 0
    }
  },
  { immediate: true },
)
const current = computed(() => maskers.value.find((m) => m.port === port.value))
const caps = ref<MaskCaps | null>(null)
watch(
  port,
  async (p) => {
    caps.value = null
    if (!p) return
    try {
      const r = await fetch(`/api/runners/${p}/server`)
      caps.value = r.ok ? capsFrom(await r.json()) : null
    } catch {
      caps.value = null
    }
  },
  { immediate: true },
)

// -- picture or camera ----------------------------------------------------------
// The browser gives a page the camera only in a secure context (https, or
// localhost - the Studio on this machine).
const cameraOk =
  typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && window.isSecureContext
const source = ref<'picture' | 'camera'>('picture')
const cameraWhy = computed(() => {
  if (!cameraOk) return 'The browser offers the camera only over https or on this machine'
  return caps.value?.video ? '' : (caps.value?.videoUnavailable ?? 'This endpoint does not track video')
})
function cameraError(message: string): void {
  toasts.push({ tone: 'bad', title: 'The camera stopped', description: message })
}
// -- the picture -----------------------------------------------------------------
interface Picture {
  dataUrl: string
  name: string
  width: number
  height: number
  /** the key it is saved under ('' where the browser cannot hash it) */
  ref: string
}
const picture = ref<Picture | null>(null)
const imgEl = ref<HTMLImageElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const pictureError = ref('')
const dragOver = ref(false)

function readAsDataUrl(f: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result))
    r.onerror = () => reject(r.error ?? new Error('could not read the file'))
    r.readAsDataURL(f)
  })
}
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const im = new Image()
    im.onload = () => resolve(im)
    im.onerror = () => reject(new Error('the browser could not open this picture'))
    im.src = src
  })
}

/** JPEG and PNG go up as they are (the runner decodes them byte-for-byte as
 *  its reference does); anything else the browser can open is re-encoded as
 *  a lossless PNG of the same pixels. */
async function usePicture(f: Blob, name: string): Promise<void> {
  pictureError.value = ''
  try {
    let url = await readAsDataUrl(f)
    let im = await loadImage(url)
    if (f.type !== 'image/jpeg' && f.type !== 'image/png') {
      const c = document.createElement('canvas')
      c.width = im.naturalWidth
      c.height = im.naturalHeight
      c.getContext('2d')?.drawImage(im, 0, 0)
      url = c.toDataURL('image/png')
      im = await loadImage(url)
    }
    const px = im.naturalWidth * im.naturalHeight
    const max = caps.value?.maxPixels ?? 24_000_000
    if (px > max) {
      pictureError.value = `This picture is ${(px / 1e6).toFixed(1)} MP; the endpoint takes at most ${(max / 1e6).toFixed(0)} MP.`
      return
    }
    let ref = ''
    try {
      ref = await pictureKey(url)
    } catch (e) {
      history.error = e instanceof Error ? e.message : String(e)
    }
    // A different picture is a record of its own: the one on screen is saved
    // first, and the new one starts with the same prompts, without what they
    // found. A record with no picture yet (snapshots only) takes this one.
    if (picture.value && history.document) {
      clearTimeout(saveTimer)
      await saveNow()
      history.reset()
      shots.value = []
    }
    picture.value = { dataUrl: url, name, width: im.naturalWidth, height: im.naturalHeight, ref }
    // a new picture: the prompts stay, their boxes, clicks and results were the old one's
    for (const l of layers.value) resetPrompt(l)
    selected.value = null
    await nextTick()
    resizeOverlay()
  } catch (e) {
    pictureError.value = e instanceof Error ? e.message : String(e)
  }
}
async function onFile(e: Event): Promise<void> {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  input.value = ''
  if (f) await usePicture(f, f.name)
}
async function onDrop(e: DragEvent): Promise<void> {
  dragOver.value = false
  const f = Array.from(e.dataTransfer?.files ?? []).find((x) => x.type.startsWith('image/'))
  if (f) await usePicture(f, f.name)
}
async function onPaste(e: ClipboardEvent): Promise<void> {
  // a paste into a text field is text, not a picture
  const t = e.target as HTMLElement | null
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
  const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith('image/'))
  const f = item?.getAsFile()
  if (f) {
    e.preventDefault()
    await usePicture(f, 'Pasted picture')
  }
}
function clearPicture(): void {
  picture.value = null
  for (const l of layers.value) resetPrompt(l)
  selected.value = null
}

// -- prompts (layers) --------------------------------------------------------------
interface Layer {
  id: number
  /** every instance of a concept, or the one object under clicks */
  kind: 'concept' | 'object'
  /** the concept; an object layer's name (only the export reads it) */
  text: string
  boxes: PromptBox[]
  points: PromptPoint[]
  objectBox: PromptBox | null
  threshold: number
  visible: boolean
  result: MaskResponse | null
  /** the candidate an object layer shows (0 = the model's best) */
  choice: number
  /** what the object answer on show was asked with, to refine it */
  lastAsk: ClickAsk | null
  /** an object layer changed while its request was out: ask once more */
  again: boolean
  /** decoded masks by instance index, made when first drawn */
  planes: Map<number, Uint8Array>
  /** instances switched off in the list */
  hidden: Set<number>
  busy: boolean
  error: string
}
let nextId = 1
function newLayer(kind: Layer['kind'] = 'concept'): Layer {
  return {
    id: nextId++,
    kind,
    text: '',
    boxes: [],
    points: [],
    objectBox: null,
    threshold: 0.5,
    visible: true,
    result: null,
    choice: 0,
    lastAsk: null,
    again: false,
    // raw: decoded planes are megabytes each and never drive a render
    planes: markRaw(new Map()),
    hidden: new Set(),
    busy: false,
    error: '',
  }
}
const layers = ref<Layer[]>([newLayer()])
const activeId = ref(layers.value[0].id)
const active = computed(() => layers.value.find((l) => l.id === activeId.value) ?? layers.value[0])
const layerIndex = (l: Layer) => layers.value.findIndex((x) => x.id === l.id)

function addLayer(kind: Layer['kind'] = 'concept'): void {
  const l = newLayer(kind)
  layers.value.push(l)
  activeId.value = l.id
}
/** Drop a layer's prompt and what it found; its name, colour and kind stay. */
function resetPrompt(l: Layer): void {
  l.boxes = []
  l.points = []
  l.objectBox = null
  l.result = null
  l.choice = 0
  l.lastAsk = null
  l.planes.clear()
  l.hidden.clear()
  l.error = ''
}
function setKind(k: string): void {
  const l = active.value
  if ((k !== 'concept' && k !== 'object') || k === l.kind) return
  resetPrompt(l)
  l.kind = k
  if (selected.value?.layer === l.id) selected.value = null
}
function removeLayer(l: Layer): void {
  if (layers.value.length === 1) {
    Object.assign(l, newLayer(), { id: l.id })
    return
  }
  layers.value = layers.value.filter((x) => x.id !== l.id)
  if (activeId.value === l.id) activeId.value = layers.value[0].id
  if (selected.value?.layer === l.id) selected.value = null
}
function layerLabel(l: Layer): string {
  const t = l.text.trim()
  if (t) return t
  if (l.kind === 'object') {
    const parts: string[] = []
    if (l.objectBox) parts.push('box')
    if (l.points.length) parts.push(l.points.length === 1 ? '1 click' : `${l.points.length} clicks`)
    return parts.length ? `Object - ${parts.join(', ')}` : 'New object'
  }
  if (l.boxes.length) return l.boxes.length === 1 ? '1 example box' : `${l.boxes.length} example boxes`
  return 'New prompt'
}

const hasPrompt = (l: Layer) =>
  l.kind === 'object' ? l.points.length > 0 || !!l.objectBox : l.text.trim().length > 0 || l.boxes.length > 0
const canFind = computed(() => !!current.value && !!picture.value && !active.value.busy && hasPrompt(active.value))

function run(l: Layer): Promise<void> {
  return l.kind === 'object' ? runObject(l) : runConcept(l)
}
async function runConcept(l: Layer): Promise<void> {
  const m = current.value
  const pic = picture.value
  if (!m?.port || !pic || (!l.text.trim() && !l.boxes.length)) return
  l.busy = true
  l.error = ''
  try {
    const res = await fetch(`/api/runners/${m.port}/v1/masks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody(m.id, pic.dataUrl, l.text, l.boxes, l.threshold)),
    })
    const body = (await res.json().catch(() => null)) as
      | (MaskResponse & { error?: { message?: string } })
      | null
    if (!res.ok || !body || body.object !== 'masks') {
      l.error = body?.error?.message ?? `the endpoint answered ${res.status}`
      return
    }
    // still the same picture? a slow answer to an old one is dropped
    if (picture.value !== pic) return
    // raw too: a response's run lists are long and only ever read
    l.result = markRaw(body)
    l.planes.clear()
    l.hidden.clear()
    if (selected.value?.layer === l.id) selected.value = null
  } catch (e) {
    l.error = e instanceof Error ? e.message : String(e)
  } finally {
    l.busy = false
  }
}
/** One request at a time a layer: clicks that land while one is out are
 *  folded into the next ask, which then refines the answer that came back. */
async function runObject(l: Layer): Promise<void> {
  const m = current.value
  const pic = picture.value
  if (!m?.port || !pic) return
  if (!hasPrompt(l)) {
    l.result = null
    l.lastAsk = null
    l.planes.clear()
    l.error = ''
    if (selected.value?.layer === l.id) selected.value = null
    return
  }
  if (l.busy) {
    l.again = true
    return
  }
  l.busy = true
  l.error = ''
  // a snapshot: the clicks may change while this one is out
  const points = l.points.map((x) => ({ ...x }))
  const box = l.objectBox ? { ...l.objectBox } : null
  // the handle names the model's best candidate - not one picked by hand
  const refine = l.choice === 0 ? refineFor(l.lastAsk, points, box) : undefined
  const ask = (r?: string) =>
    fetch(`/api/runners/${m.port}/v1/masks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(clickBody(m.id, pic.dataUrl, points, box, r)),
    })
  try {
    let res = await ask(refine)
    // a handle is good only for the runner's last click answer, and another
    // prompt may have come between: ask again without it
    if (res.status === 400 && refine) res = await ask()
    const body = (await res.json().catch(() => null)) as
      | (MaskResponse & { error?: { message?: string } })
      | null
    if (!res.ok || !body || body.object !== 'masks') {
      l.error = body?.error?.message ?? `the endpoint answered ${res.status}`
      return
    }
    if (picture.value !== pic) return
    l.result = markRaw(body)
    l.planes.clear()
    l.hidden.clear()
    l.choice = 0
    l.lastAsk = body.refine_id ? { points, box, refineId: body.refine_id } : null
    // the object on show is what the exports take
    selected.value = { layer: l.id, index: 0 }
  } catch (e) {
    l.error = e instanceof Error ? e.message : String(e)
  } finally {
    l.busy = false
    if (l.again) {
      l.again = false
      void runObject(l)
    }
  }
}
async function runAll(): Promise<void> {
  for (const l of layers.value) if (hasPrompt(l)) await run(l)
}
const pendingPrompts = computed(() => !!picture.value && layers.value.some((l) => !l.result && hasPrompt(l)))

// the threshold re-asks (the warm path is milliseconds: the picture is reused)
let thresholdTimer: number | undefined
watch(
  () => active.value.threshold,
  () => {
    clearTimeout(thresholdTimer)
    const l = active.value
    if (!l.result || l.kind === 'object') return
    thresholdTimer = window.setTimeout(() => void run(l), 250)
  },
)

// -- boxes ---------------------------------------------------------------------------
const boxMode = ref<'include' | 'exclude'>('include')
const frameEl = ref<HTMLDivElement | null>(null)
const draft = ref<PromptBox | null>(null)
let dragStart: { x: number; y: number } | null = null

function toPicture(e: PointerEvent): { x: number; y: number } | null {
  const el = frameEl.value
  const pic = picture.value
  if (!el || !pic) return null
  const r = el.getBoundingClientRect()
  const x = ((e.clientX - r.left) / r.width) * pic.width
  const y = ((e.clientY - r.top) / r.height) * pic.height
  return { x: Math.max(0, Math.min(pic.width, x)), y: Math.max(0, Math.min(pic.height, y)) }
}
function onPointerDown(e: PointerEvent): void {
  if (e.button !== 0 || !picture.value) return
  const p = toPicture(e)
  if (!p) return
  dragStart = p
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  const positive = (boxMode.value === 'include') !== e.altKey
  draft.value = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, positive }
}
function onPointerMove(e: PointerEvent): void {
  if (!dragStart || !draft.value) {
    hoverAt(e)
    return
  }
  const p = toPicture(e)
  if (!p) return
  draft.value = {
    x0: Math.min(dragStart.x, p.x),
    y0: Math.min(dragStart.y, p.y),
    x1: Math.max(dragStart.x, p.x),
    y1: Math.max(dragStart.y, p.y),
    positive: draft.value.positive,
  }
}
function onPointerUp(): void {
  const b = draft.value
  const start = dragStart
  draft.value = null
  dragStart = null
  if (!b || !start || !picture.value) return
  const minSide = Math.max(4, Math.min(picture.value.width, picture.value.height) * 0.004)
  const click = b.x1 - b.x0 < minSide || b.y1 - b.y0 < minSide
  const l = active.value
  if (l.kind === 'object') {
    if (click) {
      const max = caps.value?.maxPoints || 32
      if (l.points.length >= max) {
        l.error = `An object takes at most ${max} clicks.`
        return
      }
      l.points.push({ x: Math.round(start.x), y: Math.round(start.y), positive: b.positive })
    } else {
      l.objectBox = { x0: Math.round(b.x0), y0: Math.round(b.y0), x1: Math.round(b.x1), y1: Math.round(b.y1), positive: true }
    }
    void run(l)
    return
  }
  if (click) {
    pickAt(start.x, start.y)
    return
  }
  const max = caps.value?.maxBoxes ?? 16
  if (active.value.boxes.length >= max) {
    active.value.error = `A prompt takes at most ${max} example boxes.`
    return
  }
  active.value.boxes.push({
    x0: Math.round(b.x0),
    y0: Math.round(b.y0),
    x1: Math.round(b.x1),
    y1: Math.round(b.y1),
    positive: b.positive,
  })
}
function removeBox(i: number): void {
  active.value.boxes.splice(i, 1)
}
function removePoint(i: number): void {
  const l = active.value
  l.points.splice(i, 1)
  void run(l)
}
function removeObjectBox(): void {
  const l = active.value
  l.objectBox = null
  void run(l)
}
function clearClicks(): void {
  const l = active.value
  l.points = []
  l.objectBox = null
  void run(l)
}

// -- instances: hover, select, draw ----------------------------------------------------
interface InstanceRef {
  layer: number
  index: number
}
const hovered = ref<InstanceRef | null>(null)
const selected = ref<InstanceRef | null>(null)
const same = (a: InstanceRef | null, b: InstanceRef | null) => !!a && !!b && a.layer === b.layer && a.index === b.index

function plane(l: Layer, i: number): Uint8Array | null {
  const inst = l.result?.instances[i]
  if (!inst) return null
  let p = l.planes.get(i)
  if (!p) {
    p = decodeRle(inst.mask)
    l.planes.set(i, p)
  }
  return p
}

/** The topmost visible instance under a picture point. */
function instanceAt(x: number, y: number): InstanceRef | null {
  for (let li = layers.value.length - 1; li >= 0; li--) {
    const l = layers.value[li]
    if (!l.visible || !l.result) continue
    const { width, height } = l.result
    const xi = Math.min(width - 1, Math.max(0, Math.floor(x)))
    const yi = Math.min(height - 1, Math.max(0, Math.floor(y)))
    for (let i = l.result.instances.length - 1; i >= 0; i--) {
      if (l.hidden.has(i) || (l.kind === 'object' && i !== l.choice)) continue
      const b = l.result.instances[i].box
      if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue
      if (plane(l, i)?.[yi * width + xi]) return { layer: l.id, index: i }
    }
  }
  return null
}
function hoverAt(e: PointerEvent): void {
  const p = toPicture(e)
  hovered.value = p ? instanceAt(p.x, p.y) : null
}
function pickAt(x: number, y: number): void {
  const hit = instanceAt(x, y)
  selected.value = hit && !same(hit, selected.value) ? hit : null
}

const overlay = ref<HTMLCanvasElement | null>(null)
// the overlay is a scaled view of the picture: at most 2048 px on its long side
const overlaySize = computed(() => {
  const pic = picture.value
  if (!pic) return { w: 1, h: 1 }
  const k = Math.min(1, 2048 / Math.max(pic.width, pic.height))
  return { w: Math.max(1, Math.round(pic.width * k)), h: Math.max(1, Math.round(pic.height * k)) }
})
function resizeOverlay(): void {
  const c = overlay.value
  if (!c) return
  c.width = overlaySize.value.w
  c.height = overlaySize.value.h
  redraw()
}
function redraw(): void {
  const c = overlay.value
  const ctx = c?.getContext('2d')
  if (!c || !ctx) return
  const masks: DrawnMask[] = []
  layers.value.forEach((l, li) => {
    if (!l.visible || !l.result) return
    l.result.instances.forEach((inst, i) => {
      if (l.hidden.has(i) || (l.kind === 'object' && i !== l.choice)) return
      const p = plane(l, i)
      if (!p) return
      const ref = { layer: l.id, index: i }
      masks.push({
        plane: p,
        width: l.result!.width,
        height: l.result!.height,
        rgb: instanceRgb(li, i),
        lit: same(ref, hovered.value) || same(ref, selected.value),
        box: inst.box,
      })
    })
  })
  paintOverlay(ctx, c.width, c.height, masks)
}
let frame = 0
watch(
  [layers, hovered, selected],
  () => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(redraw)
  },
  { deep: true },
)

// -- the active layer's instances and diagnostics ----------------------------------------
const activeInstances = computed(() => {
  const l = active.value
  const r = l.result
  if (!r) return []
  const total = r.width * r.height
  return r.instances.map((inst, i) => ({
    index: i,
    score: inst.score,
    share: total ? (100 * inst.area) / total : 0,
    rgb: instanceRgb(layerIndex(l), i),
    w: Math.round(inst.box[2] - inst.box[0]),
    h: Math.round(inst.box[3] - inst.box[1]),
    on:
      l.kind === 'object'
        ? i === l.choice
        : selected.value?.layer === l.id && selected.value?.index === i,
  }))
})
function onInstanceClick(i: number): void {
  const l = active.value
  if (l.kind === 'object') l.choice = i
  selected.value = { layer: l.id, index: i }
}
function toggleInstance(i: number, on: boolean | 'indeterminate'): void {
  const l = active.value
  if (on === true) l.hidden.delete(i)
  else l.hidden.add(i)
}
const timingLine = computed(() => {
  const l = active.value
  const t = l.result?.timings
  if (!t) return ''
  const parts = [t.image_reused ? 'picture reused' : `encode ${t.encode_ms.toFixed(0)} ms`]
  if (l.kind === 'object') {
    parts.push(`decode ${t.detect_ms.toFixed(1)} ms`)
  } else {
    parts.push(t.text_reused ? 'prompt reused' : `prompt ${t.prompt_ms.toFixed(1)} ms`, `detect ${t.detect_ms.toFixed(1)} ms`)
  }
  parts.push(`masks ${t.masks_ms.toFixed(1)} ms`)
  return parts.join(' · ')
})

// -- exports -----------------------------------------------------------------------------
const anyResult = computed(() => layers.value.some((l) => l.result && l.result.instances.length))
function coco(): string {
  const pic = picture.value
  const withResults = layers.value.filter((l) => l.result)
  return cocoJson(
    pic?.width ?? 0,
    pic?.height ?? 0,
    withResults.map((l) => ({
      prompt: layerLabel(l),
      instances: l.result!.instances.filter((_, i) => (l.kind === 'object' ? i === l.choice : !l.hidden.has(i))),
    })),
  )
}
async function copyJson(): Promise<void> {
  await copyText(coco())
  toasts.push({ tone: 'good', title: 'COCO JSON copied' })
}
function saveJson(): void {
  const base = picture.value ? slug(picture.value.name.replace(/\.[^.]+$/, '')) : 'masks'
  saveBlob(new Blob([coco()], { type: 'application/json' }), `${base}-masks.json`)
}
const selectedLayer = computed(() => layers.value.find((l) => l.id === selected.value?.layer) ?? null)
async function saveCutout(kind: 'cutout' | 'mask'): Promise<void> {
  const l = selectedLayer.value
  const s = selected.value
  const r = l?.result
  if (!l || !s || !r) return
  const p = plane(l, s.index)
  if (!p) return
  try {
    const name = `${slug(layerLabel(l))}-${s.index + 1}`
    if (kind === 'mask') {
      saveBlob(await maskPng(p, r.width, r.height), `${name}-mask.png`)
    } else {
      const im = imgEl.value
      if (!im) return
      saveBlob(await cutoutPng(im, p, r.width, r.height), `${name}.png`)
    }
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Could not save the PNG', description: String(e) })
  }
}
async function copyCurl(): Promise<void> {
  const m = current.value
  if (!m?.port || !picture.value) return
  const l = active.value
  const body =
    l.kind === 'object'
      ? clickBody(m.id, picture.value.dataUrl, l.points, l.objectBox)
      : requestBody(m.id, picture.value.dataUrl, l.text, l.boxes, l.threshold)
  await copyText(curlFor(m.port, body))
  toasts.push({ tone: 'good', title: 'curl copied' })
}

function onKey(e: KeyboardEvent): void {
  if (e.key === 'Escape') {
    if (draft.value) {
      draft.value = null
      dragStart = null
    } else {
      selected.value = null
    }
  }
}
// -- the camera ------------------------------------------------------------------
// Its concepts are the page's prompts with words, in their colours - one list
// for both modes. A session tracks the first `max_concepts` of them; any
// beyond are listed as not tracked rather than dropped without a word.
const camMax = computed(() => Math.max(1, caps.value?.maxConcepts ?? 1))
const camConcepts = computed(() => {
  const out: { layer: Layer; text: string; slot: number; tracked: boolean }[] = []
  layers.value.forEach((l, li) => {
    const text = l.text.trim()
    if (l.kind === 'concept' && text) out.push({ layer: l, text, slot: li, tracked: false })
  })
  out.forEach((c, i) => (c.tracked = i < camMax.value))
  return out
})
const camSent = computed<CamConcept[]>(() =>
  camConcepts.value.filter((c) => c.tracked).map(({ text, slot }) => ({ text, slot })),
)
const camDraft = ref('')
const camCanAdd = computed(() => camConcepts.value.length < camMax.value)
function camAdd(): void {
  const text = camDraft.value.trim()
  if (!text || !camCanAdd.value) return
  camDraft.value = ''
  if (camConcepts.value.some((c) => c.text.toLowerCase() === text.toLowerCase())) return
  // an empty prompt takes the words before a new one is made
  const blank = layers.value.find((l) => l.kind === 'concept' && !l.text.trim() && !l.boxes.length && !l.result)
  if (blank) {
    blank.text = text
  } else {
    const l = newLayer()
    l.text = text
    layers.value.push(l)
  }
}
const {
  video: camVideo,
  overlay: camOverlay,
  status: camStatus,
  statusText: camStatusText,
  live: camLive,
  tracked: camTracked,
  objects: camObjects,
  frameNo: camFrame,
  ms: camMs,
  rate: camRate,
  frameSize: camSize,
  deviceId: camDevice,
  deviceOptions: camDevices,
  side: camSide,
  sideOptions: camSides,
  canTrack: camCanTrack,
  open: camOpen,
  shut: camShut,
  stop: camStop,
  toggle: camToggle,
  snapshot: camSnapshot,
} = useMaskCamera(port, camSent, cameraError)
// the camera is let go the moment its stage is not shown
watch(
  () => source.value === 'camera' && !cameraWhy.value,
  (on) => {
    if (!on) camShut()
  },
)
onUnmounted(camShut)
const camBroken = computed(() => ['denied', 'none', 'failed'].includes(camStatus.value))
// The frame keeps the video's own shape at every width: capped by height
// through its width, so the mirrored overlay and the tags on it sit on the
// picture rather than on letterboxing.
const camFit = computed(() => {
  const [w, h] = camSize.value ?? [16, 9]
  return { aspectRatio: `${w} / ${h}`, maxWidth: `calc(72vh * ${w / h})` }
})
const camHue = (slot: number) => `hsl(${layerHue(slot)} 78% 52%)`
/** A tag on each object, at its box's top-left as the mirrored view shows it. */
const camTags = computed(() => {
  const size = camSize.value
  if (!size) return []
  const [w, h] = size
  const many = camTracked.value.length > 1
  return camObjects.value.map((o) => ({
    id: o.id,
    label: many ? `${camTracked.value[o.concept]?.text ?? ''} #${o.id}` : `#${o.id}`,
    color: rgbCss(objectRgb(o, camTracked.value)),
    left: `${((w - o.box[2]) / w) * 100}%`,
    top: `${(o.box[1] / h) * 100}%`,
  }))
})
/** The objects by concept, for the list beside the frame. */
const camGroups = computed(() =>
  camTracked.value
    .map((c, k) => ({ concept: c, objects: camObjects.value.filter((o) => o.concept === k) }))
    .filter((g) => g.objects.length),
)
const camCount = (text: string) => {
  const k = camTracked.value.findIndex((t) => t.text === text)
  return k < 0 ? 0 : camObjects.value.filter((o) => o.concept === k).length
}

// -- snapshots: frames kept with the record, oldest first --------------------------
interface Shot extends CamFrame {
  id: string
  ref: string
  at: number
}
const shots = shallowRef<Shot[]>([])
const shotsShown = computed(() => [...shots.value].reverse())
// the strip's thumbnails are drawn here, never kept
const thumbs = shallowReactive(new Map<string, string>())
watch(
  shots,
  (list) => {
    for (const s of list) {
      if (thumbs.has(s.id)) continue
      void shotCanvas(s, 320)
        .then((c) => thumbs.set(s.id, c.toDataURL('image/jpeg', 0.85)))
        .catch(() => {})
    }
  },
  { immediate: true },
)
async function camSnap(): Promise<void> {
  if (shots.value.length >= MAX_SNAPSHOTS) {
    toasts.push({
      tone: 'bad',
      title: 'Snapshot not kept',
      description: `A picture keeps at most ${MAX_SNAPSHOTS} snapshots - remove some first.`,
    })
    return
  }
  const f = camSnapshot()
  if (!f) return
  try {
    const ref = await pictureKey(f.image)
    shots.value = [...shots.value, { ...f, objects: markRaw(f.objects), id: uuid(), ref, at: Date.now() }]
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Snapshot not kept', description: String(e) })
  }
}
function camDropShot(id: string): void {
  shots.value = shots.value.filter((s) => s.id !== id)
  thumbs.delete(id)
}
const shotTime = (s: Shot) =>
  new Date(s.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
const shotName = (s: Shot) =>
  `snapshot-${new Date(s.at).toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-')}`
async function saveShot(s: Shot): Promise<void> {
  try {
    const c = await shotCanvas(s)
    const blob = await new Promise<Blob | null>((r) => c.toBlob(r, 'image/png'))
    if (!blob) throw new Error('the browser could not encode a PNG')
    saveBlob(blob, `${shotName(s)}.png`)
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Could not save the snapshot', description: String(e) })
  }
}
/** A snapshot into Picture: its frame, one prompt a concept it tracked, each
 *  found on the frame by the picture model. */
async function openShot(s: Shot): Promise<void> {
  const blob = await (await fetch(s.image)).blob()
  source.value = 'picture'
  await usePicture(blob, `${shotName(s)}.jpg`)
  if (!picture.value || !s.concepts.length) return
  layers.value = s.concepts.map((c) => Object.assign(newLayer(), { text: c.text }))
  activeId.value = layers.value[0].id
  await runAll()
}

// -- the record: what is saved, and how it comes back -------------------------------
const router = useRouter()
const restoring = ref(false)
const switching = ref(false)
const anyBusy = computed(() => layers.value.some((l) => l.busy))
const navigationBlocked = computed(() => anyBusy.value || restoring.value || switching.value || history.saving)
onMounted(() => void history.refresh())

function toLayerDoc(l: Layer): MaskLayerDoc {
  return {
    kind: l.kind,
    text: l.text,
    boxes: l.boxes.map((b) => ({ ...b })),
    points: l.points.map((x) => ({ ...x })),
    objectBox: l.objectBox ? { ...l.objectBox } : null,
    threshold: l.threshold,
    visible: l.visible,
    choice: l.choice,
    hidden: [...l.hidden],
    result: l.result,
  }
}
function fromLayerDoc(d: MaskLayerDoc): Layer {
  const l = newLayer(d.kind)
  l.text = d.text
  l.boxes = d.boxes.map((b) => ({ ...b }))
  l.points = d.points.map((x) => ({ ...x }))
  l.objectBox = d.objectBox ? { ...d.objectBox } : null
  l.threshold = d.threshold
  l.visible = d.visible
  l.choice = d.choice
  l.hidden = new Set(d.hidden)
  // the answer's refine handle lived on the runner and is gone: the next
  // click asks afresh
  l.result = d.result ? markRaw(d.result) : null
  return l
}
function defaultTitle(): string {
  const pic = picture.value
  if (pic) return pic.name.slice(0, 200) || 'Picture'
  return `Camera ${new Date().toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`
}
/** The page as a record, made from the one it replaces (null for a new one). */
function buildDoc(prev: MaskDoc | null): MaskDoc {
  const now = Date.now()
  const pic = picture.value
  const pictures: Record<string, string> = {}
  if (pic?.ref) pictures[pic.ref] = pic.dataUrl
  for (const s of shots.value) pictures[s.ref] = s.image
  return {
    version: 1,
    id: prev?.id ?? uuid(),
    title: prev?.title ?? defaultTitle(),
    model: current.value?.id ?? prev?.model ?? '',
    createdAt: prev?.createdAt ?? now,
    updatedAt: Math.max(now, (prev?.updatedAt ?? 0) + 1),
    pictures,
    picture: pic?.ref ? { ref: pic.ref, name: pic.name, width: pic.width, height: pic.height } : null,
    layers: layers.value.map(toLayerDoc),
    active: Math.max(0, layers.value.findIndex((l) => l.id === activeId.value)),
    snapshots: shots.value.map((s) => ({
      id: s.id,
      ref: s.ref,
      width: s.width,
      height: s.height,
      at: s.at,
      concepts: s.concepts.map(({ text, slot }) => ({ text, slot })),
      objects: s.objects,
    })),
  }
}
// Every change is saved shortly after it settles, as on Tables; a prompt
// that is still asking is saved when its answer lands.
let saveTimer: ReturnType<typeof setTimeout> | undefined
let editGeneration = 0
let savedGeneration = 0
function edited(): void {
  if (restoring.value) return
  ++editGeneration
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    if (!anyBusy.value) void saveNow()
  }, 800)
}
watch([picture, layers, activeId], edited, { deep: true })
watch(shots, edited)
async function saveNow(): Promise<boolean> {
  if (restoring.value) return false
  if (savedGeneration === editGeneration) return !history.hasPendingSave()
  // prompts typed on an empty page are not a record yet
  if (!history.document && !picture.value && !shots.value.length) {
    savedGeneration = editGeneration
    return true
  }
  // a picture the browser could not hash (said when it opened) cannot be kept
  if (picture.value && !picture.value.ref) return false
  const attempt = editGeneration
  const ok = await history.save(buildDoc)
  if (ok) savedGeneration = attempt
  return ok
}
async function applyDoc(doc: MaskDoc): Promise<void> {
  restoring.value = true
  clearTimeout(saveTimer)
  camStop()
  const pic = doc.picture
  const url = pic ? doc.pictures[pic.ref] : undefined
  picture.value = pic && url ? { dataUrl: url, name: pic.name, width: pic.width, height: pic.height, ref: pic.ref } : null
  layers.value = doc.layers.map(fromLayerDoc)
  activeId.value = layers.value[doc.active]?.id ?? layers.value[0].id
  shots.value = doc.snapshots.flatMap((s) => {
    const image = doc.pictures[s.ref]
    return image ? [{ ...s, image, objects: markRaw(s.objects) }] : []
  })
  selected.value = null
  hovered.value = null
  pictureError.value = ''
  await nextTick()
  resizeOverlay()
  savedGeneration = editGeneration
  restoring.value = false
}
async function clearPage(): Promise<void> {
  restoring.value = true
  clearTimeout(saveTimer)
  camStop()
  history.reset()
  picture.value = null
  const l = newLayer()
  layers.value = [l]
  activeId.value = l.id
  shots.value = []
  selected.value = null
  hovered.value = null
  pictureError.value = ''
  await nextTick()
  savedGeneration = editGeneration
  restoring.value = false
}
function openSession(id: string): void {
  if (navigationBlocked.value || id === history.document?.id) return
  void router.push({ name: 'masks', params: { id } })
}
async function newSession(): Promise<void> {
  if (navigationBlocked.value) return
  // the route guard saves the record being left, the route watcher clears the page
  if (route.params.id) {
    void router.push({ name: 'masks' })
    return
  }
  switching.value = true
  try {
    if (await saveNow()) await clearPage()
  } finally {
    switching.value = false
  }
}
async function showSession(id: string): Promise<void> {
  switching.value = true
  try {
    const doc = await history.open(id)
    if (doc) await applyDoc(doc)
    // a deleted or mistyped id: the page keeps what it had, and so does the URL
    else syncUrl(history.document?.id)
  } finally {
    switching.value = false
  }
}
async function renameSession(id: string, title: string): Promise<void> {
  if (navigationBlocked.value) return
  switching.value = true
  clearTimeout(saveTimer)
  try {
    if (await saveNow()) await history.rename(id, title)
  } finally {
    switching.value = false
  }
}
async function removeSession(id: string): Promise<void> {
  if (navigationBlocked.value) return
  clearTimeout(saveTimer)
  const active = history.document?.id === id
  if ((await history.remove(id)) && active) await clearPage()
}
async function retrySave(): Promise<void> {
  ++editGeneration
  await saveNow()
}
onBeforeRouteLeave(async () => !anyBusy.value && !switching.value && (await saveNow()))
function beforeUnload(e: BeforeUnloadEvent): void {
  if (anyBusy.value || history.saving || savedGeneration !== editGeneration) {
    e.preventDefault()
    e.returnValue = ''
  }
}
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onUnmounted(() => {
  clearTimeout(saveTimer)
  window.removeEventListener('beforeunload', beforeUnload)
})

// the side panel folds like the chat list, as it was left
const panelOpen = usePanelFold('masksPanelOpen')

// -- which record the URL names --------------------------------------------------
// Last in the script on purpose: the watcher runs at once and reaches for
// everything above. The rules are Tables': the side panel navigates, the
// watcher loads, so a reload, the back button or a pasted link lands on the
// same picture.
function paramId(raw: unknown): string | undefined {
  return typeof raw === 'string' && raw ? raw : undefined
}
function syncUrl(id: string | undefined, query?: LocationQuery): void {
  if (route.name !== 'masks' || id === paramId(route.params.id)) return
  void router.replace({ name: 'masks', params: id ? { id } : {}, query })
}
// a first save mints the id and a delete drops it - the URL follows
watch(() => history.document?.id, (id) => syncUrl(id))
onBeforeRouteUpdate(async (to) => {
  if (paramId(to.params.id) === history.document?.id) return true
  if (navigationBlocked.value) return false
  return await saveNow()
})
let firstRoute = true
watch(
  () => route.params.id,
  (raw) => {
    // leaving for another page with an :id of its own changes the param
    // before this page unmounts - that id is not a picture's
    if (route.name !== 'masks') return
    const id = paramId(raw)
    const initial = firstRoute
    firstRoute = false
    const doc = history.document
    if (id) {
      if (doc?.id !== id) void showSession(id)
      else if (initial) void applyDoc(doc)
      return
    }
    if (!doc) return
    // Coming back to the bare page keeps the record the store still holds
    // and names it; reaching the bare URL from a record is New picture.
    if (initial) {
      void applyDoc(doc)
      syncUrl(doc.id, route.query)
    } else {
      switching.value = true
      void clearPage().finally(() => {
        switching.value = false
      })
    }
  },
  { immediate: true },
)
</script>

<template>
  <div class="mk">
    <ReadsSidebar
      v-if="panelOpen"
      noun="picture"
      :reads="history.sessions"
      :active-id="history.document?.id ?? null"
      :loaded="history.loaded"
      :error="history.error"
      :busy="navigationBlocked"
      @new="newSession"
      @open="openSession"
      @rename="renameSession"
      @remove="removeSession"
      @fold="panelOpen = false"
    />
    <aside v-else class="mk__rail">
      <Tooltip label="Show pictures" side="right">
        <button class="pk-icon-btn mk__railbtn" type="button" aria-label="Show pictures" @click="panelOpen = true">
          <Icon name="panel-left" :size="18" />
        </button>
      </Tooltip>
      <Tooltip label="New picture" side="right">
        <button
          class="pk-icon-btn mk__railbtn"
          type="button"
          aria-label="New picture"
          :disabled="navigationBlocked"
          @click="newSession"
        >
          <Icon name="plus" :size="18" />
        </button>
      </Tooltip>
    </aside>

    <main class="mk__main">
    <div class="mk__inner" @keydown="onKey">
    <div class="mk__head">
      <div>
        <h1 class="mk__title">Masks</h1>
        <p v-if="source === 'camera'" class="mk__lead">
          Track everything that matches a few words through the camera, each object under its own
          id - the same /v1/masks/sessions your code calls.
        </p>
        <p v-else class="mk__lead">
          Find every object that matches a few words or a box you draw, or click one object to cut
          it out - the same /v1/masks your code calls.
        </p>
      </div>
      <div class="mk__headctl">
        <ToggleGroup v-if="maskers.length" v-model="source" label="What to mask" class="mk__seg">
          <ToggleGroupItem value="picture" class="mk__segitem">
            <Icon name="image" :size="14" /> Picture
          </ToggleGroupItem>
          <ToggleGroupItem value="camera" class="mk__segitem">
            <Icon name="webcam" :size="14" /> Camera
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
    </div>

    <p v-if="history.error" class="mk__error mk__saveerr" role="alert">
      {{ history.error }}
      <button class="pk-btn pk-btn--sm" type="button" :disabled="navigationBlocked" @click="retrySave">Retry save</button>
    </p>

    <p v-if="!maskers.length && !models.loading && history.document" class="mk__nomodel">
      No masks model is running, so this picture cannot be asked again until SAM 3 is started in the
      Manager.
    </p>

    <div v-if="!maskers.length && !models.loading && !history.document" class="mk__none">
      <Icon name="masks" :size="32" class="mk__none-icon" />
      <p class="mk__none-title">No masks model is running</p>
      <p class="mk__none-txt">Start SAM 3 in the Manager - it runs on an NVIDIA GPU.</p>
      <RouterLink class="pk-btn pk-btn--primary" :to="{ name: 'server-new' }">
        <Icon name="play" :size="14" /> Start a model
      </RouterLink>
    </div>

    <div v-else-if="source === 'camera'" class="mk__body">
      <section class="mk__stage">
        <div v-if="cameraWhy" class="mk__drop">
          <Icon name="webcam" :size="32" class="mk__none-icon" />
          <p class="mk__none-title">The camera is not available here</p>
          <p class="mk__none-txt">{{ cameraWhy }}</p>
        </div>
        <div v-else-if="camStatus === 'off'" class="mk__drop">
          <Icon name="webcam-off" :size="32" class="mk__none-icon" />
          <p class="mk__none-title">The camera is off</p>
          <button class="pk-btn" type="button" @click="camOpen()">
            <Icon name="webcam" :size="14" /> Turn on camera
          </button>
        </div>
        <template v-else>
          <div class="mk__cam" :style="camFit">
            <video ref="camVideo" class="mk__cam-feed" autoplay playsinline muted />
            <canvas ref="camOverlay" class="mk__cam-feed mk__cam-overlay" />
            <span
              v-for="t in camTags"
              :key="t.id"
              class="mk__tag"
              :style="{ left: t.left, top: t.top, '--tag': t.color }"
            >
              {{ t.label }}
            </span>
            <div v-if="camStatus !== 'ready'" class="mk__cam-cover">
              <Icon
                :name="camBroken ? 'webcam-off' : 'spinner'"
                :size="22"
                :class="{ mk__spin: !camBroken }"
              />
              <p>{{ camStatusText || 'Turning on the camera...' }}</p>
              <button v-if="camBroken" class="pk-btn pk-btn--sm" type="button" @click="camOpen()">
                Try again
              </button>
            </div>
            <span v-if="camLive" class="mk__cam-pill">
              <span class="mk__cam-dot" />tracking
              <span v-if="camRate" class="mk__cam-rate">{{ camRate.toFixed(1) }}/s</span>
            </span>
          </div>
          <div class="mk__bar">
            <span class="mk__label mk__label--sm">Frame</span>
            <Select :model-value="camSide" :options="camSides" @update:model-value="camSide = Number($event)" />
            <template v-if="camDevices.length > 1">
              <span class="mk__label mk__label--sm">Camera</span>
              <Select
                :model-value="camDevice"
                :options="camDevices"
                @update:model-value="camDevice = String($event)"
              />
            </template>
            <span class="mk__dims">
              <template v-if="camSize">{{ camSize[0] }} x {{ camSize[1] }}</template>
            </span>
            <button class="pk-btn pk-btn--ghost" type="button" :disabled="camStatus !== 'ready'" @click="camSnap()">
              <Icon name="camera" :size="14" /> Snapshot
            </button>
            <button class="pk-btn pk-btn--ghost" type="button" aria-label="Turn off the camera" @click="camShut()">
              <Icon name="webcam-off" :size="14" /> Turn off
            </button>
          </div>
        </template>
        <ul v-if="shots.length" class="mk__shots">
          <li v-for="s in shotsShown" :key="s.id" class="mk__shot">
            <img
              v-if="thumbs.get(s.id)"
              class="mk__shot-img"
              :src="thumbs.get(s.id)"
              :alt="`Snapshot at ${shotTime(s)}`"
            />
            <span v-else class="mk__shot-img" :style="{ aspectRatio: `${s.width} / ${s.height}` }" />
            <div class="mk__shot-row">
              <span class="mk__shot-time">{{ shotTime(s) }}</span>
              <Tooltip label="Save PNG">
                <button class="pk-icon-btn" type="button" aria-label="Save this snapshot as a PNG" @click="saveShot(s)">
                  <Icon name="download" :size="14" />
                </button>
              </Tooltip>
              <Tooltip label="Remove">
                <button class="pk-icon-btn" type="button" aria-label="Remove this snapshot" @click="camDropShot(s.id)">
                  <Icon name="x" :size="14" />
                </button>
              </Tooltip>
            </div>
            <button class="pk-btn pk-btn--sm" type="button" @click="openShot(s)">
              <Icon name="image" :size="13" /> Open in Picture
            </button>
          </li>
        </ul>
      </section>

      <aside class="mk__side">
        <section class="mk__card">
          <div class="mk__row">
            <span class="mk__label">Track</span>
            <span v-if="camMax > 1" class="mk__num">{{ camSent.length }} of {{ camMax }}</span>
          </div>
          <ul v-if="camConcepts.length" class="mk__layers">
            <li
              v-for="c in camConcepts"
              :key="c.layer.id"
              class="mk__layer mk__layer--static"
              :class="{ 'mk__layer--off': !c.tracked }"
            >
              <span class="mk__dot" :style="{ background: camHue(c.slot) }" />
              <span class="mk__layer-name">{{ c.text }}</span>
              <span v-if="!c.tracked" class="mk__count">not tracked</span>
              <span v-else-if="camLive" class="mk__count">{{ camCount(c.text) }}</span>
              <Tooltip label="Remove">
                <button
                  class="pk-icon-btn"
                  type="button"
                  :aria-label="`Remove the prompt ${c.text}`"
                  @click="removeLayer(c.layer)"
                >
                  <Icon name="x" :size="15" />
                </button>
              </Tooltip>
            </li>
          </ul>
          <p v-if="camConcepts.length > camMax" class="mk__diag">
            This endpoint tracks {{ camMax }} at a time - the first {{ camMax }} prompts.
          </p>
          <div v-if="camCanAdd" class="mk__addrow">
            <TextInput
              v-model="camDraft"
              block
              placeholder="Add something - a cup, a hand"
              @keydown.enter="camAdd()"
            />
            <Tooltip label="Add">
              <button class="pk-icon-btn" type="button" aria-label="Add" :disabled="!camDraft.trim()" @click="camAdd()">
                <Icon name="plus" :size="15" />
              </button>
            </Tooltip>
          </div>
          <button
            class="pk-btn pk-btn--primary"
            type="button"
            :disabled="!!cameraWhy || (!camLive && !camCanTrack)"
            @click="camToggle()"
          >
            <Icon :name="camLive ? 'pause' : 'play'" :size="14" />
            {{ camLive ? 'Stop tracking' : 'Start tracking' }}
          </button>
          <p v-if="camLive && camMs" class="mk__diag">
            {{ camObjects.length }} tracked · frame {{ camFrame }}
            <br />
            <span class="mk__diag-t">{{ camMs }} ms a frame</span>
          </p>
        </section>

        <section v-if="camGroups.length" class="mk__card">
          <span class="mk__label">Objects</span>
          <div v-for="g in camGroups" :key="g.concept.slot" class="mk__group">
            <span v-if="camTracked.length > 1" class="mk__label mk__label--sm">{{ g.concept.text }}</span>
            <ul class="mk__insts">
              <li v-for="o in g.objects" :key="o.id" class="mk__inst">
                <span class="mk__swatch" :style="{ background: rgbCss(objectRgb(o, camTracked)) }" />
                <span class="mk__inst-name">#{{ o.id }}</span>
                <span class="mk__num">{{ o.score.toFixed(2) }}</span>
                <span class="mk__inst-size">{{ o.box[2] - o.box[0] }} x {{ o.box[3] - o.box[1] }}</span>
              </li>
            </ul>
          </div>
        </section>
      </aside>
    </div>

    <div v-else class="mk__body">
      <section
        class="mk__stage"
        :class="{ 'mk__stage--over': dragOver }"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="onDrop"
      >
        <div v-if="!picture" class="mk__drop">
          <Icon name="image" :size="32" class="mk__none-icon" />
          <p class="mk__none-title">Drop a picture, paste one, or open a file</p>
          <button class="pk-btn" @click="fileInput?.click()">
            <Icon name="upload" :size="14" /> Open picture
          </button>
        </div>
        <template v-else>
          <div
            ref="frameEl"
            class="mk__frame"
            :style="{ aspectRatio: `${picture.width} / ${picture.height}` }"
            @pointerdown="onPointerDown"
            @pointermove="onPointerMove"
            @pointerup="onPointerUp"
            @pointerleave="hovered = null"
          >
            <img ref="imgEl" class="mk__img" :src="picture.dataUrl" alt="" draggable="false" />
            <canvas ref="overlay" class="mk__overlay" />
            <svg
              class="mk__boxes"
              :viewBox="`0 0 ${picture.width} ${picture.height}`"
              preserveAspectRatio="none"
            >
              <rect
                v-for="(b, i) in active.boxes"
                :key="i"
                :x="b.x0"
                :y="b.y0"
                :width="b.x1 - b.x0"
                :height="b.y1 - b.y0"
                class="mk__box"
                :class="b.positive ? 'mk__box--in' : 'mk__box--out'"
              />
              <rect
                v-if="active.kind === 'object' && active.objectBox"
                :x="active.objectBox.x0"
                :y="active.objectBox.y0"
                :width="active.objectBox.x1 - active.objectBox.x0"
                :height="active.objectBox.y1 - active.objectBox.y0"
                class="mk__box mk__box--obj"
              />
              <rect
                v-if="draft"
                :x="draft.x0"
                :y="draft.y0"
                :width="draft.x1 - draft.x0"
                :height="draft.y1 - draft.y0"
                class="mk__box mk__box--draft"
                :class="active.kind === 'object' ? 'mk__box--obj' : draft.positive ? 'mk__box--in' : 'mk__box--out'"
              />
            </svg>
            <template v-if="active.kind === 'object'">
              <span
                v-for="(pt, i) in active.points"
                :key="i"
                class="mk__pt"
                :class="pt.positive ? 'mk__pt--in' : 'mk__pt--out'"
                :style="{ left: `${(pt.x / picture.width) * 100}%`, top: `${(pt.y / picture.height) * 100}%` }"
              />
            </template>
          </div>
          <div class="mk__bar">
            <ToggleGroup v-model="boxMode" label="Box tool" class="mk__seg">
              <ToggleGroupItem value="include" class="mk__segitem">Include</ToggleGroupItem>
              <ToggleGroupItem value="exclude" class="mk__segitem">Exclude</ToggleGroupItem>
            </ToggleGroup>
            <span v-if="active.kind === 'object'" class="mk__hint">
              Click the object to add a point; Alt flips it. Drag to set its box.
            </span>
            <span v-else class="mk__hint">Drag to add an example box; Alt flips it. Click a mask to select it.</span>
            <span class="mk__dims">{{ picture.width }} x {{ picture.height }}</span>
            <button class="pk-btn pk-btn--ghost" @click="fileInput?.click()">
              <Icon name="upload" :size="14" /> Open another
            </button>
            <button class="pk-btn pk-btn--ghost" @click="clearPicture">Clear</button>
          </div>
        </template>
        <input ref="fileInput" class="mk__file" type="file" accept="image/*" @change="onFile" />
        <p v-if="pictureError" class="mk__error" role="alert">{{ pictureError }}</p>
      </section>

      <aside class="mk__side">
        <section class="mk__card">
          <div class="mk__row">
            <span class="mk__label">Prompts</span>
            <button v-if="pendingPrompts" class="pk-btn pk-btn--sm" :disabled="!current" @click="runAll">
              Run all
            </button>
          </div>
          <ul class="mk__layers">
            <li
              v-for="(l, li) in layers"
              :key="l.id"
              class="mk__layer"
              :class="{ 'mk__layer--on': l.id === active.id }"
              @click="activeId = l.id"
            >
              <span class="mk__dot" :style="{ background: `hsl(${layerHue(li)} 78% 52%)` }" />
              <span class="mk__layer-name">{{ layerLabel(l) }}</span>
              <span v-if="l.busy" class="mk__count"><Icon name="spinner" :size="13" class="mk__spin" /></span>
              <span v-else-if="l.result && l.kind === 'concept'" class="mk__count">{{ l.result.instances.length }}</span>
              <Tooltip :label="l.visible ? 'Hide' : 'Show'">
                <button
                  class="pk-icon-btn"
                  type="button"
                  :aria-label="l.visible ? 'Hide this prompt' : 'Show this prompt'"
                  @click.stop="l.visible = !l.visible"
                >
                  <Icon :name="l.visible ? 'eye' : 'eye-off'" :size="15" />
                </button>
              </Tooltip>
              <Tooltip label="Remove">
                <button class="pk-icon-btn" type="button" aria-label="Remove this prompt" @click.stop="removeLayer(l)">
                  <Icon name="x" :size="15" />
                </button>
              </Tooltip>
            </li>
          </ul>
          <div class="mk__actions">
            <button class="pk-btn pk-btn--ghost" @click="addLayer('concept')">
              <Icon name="plus" :size="14" /> Add prompt
            </button>
            <button v-if="caps?.clicks" class="pk-btn pk-btn--ghost" @click="addLayer('object')">
              <Icon name="plus" :size="14" /> Add object
            </button>
          </div>
        </section>

        <section class="mk__card">
          <div class="mk__row">
            <span class="mk__label">{{ active.kind === 'object' ? 'Object' : 'Find' }}</span>
            <ToggleGroup
              v-if="caps?.clicks || active.kind === 'object'"
              :model-value="active.kind"
              label="What this prompt finds"
              class="mk__seg"
              @update:model-value="setKind"
            >
              <ToggleGroupItem value="concept" class="mk__segitem">All matches</ToggleGroupItem>
              <ToggleGroupItem value="object" class="mk__segitem">One object</ToggleGroupItem>
            </ToggleGroup>
          </div>
          <template v-if="active.kind === 'object'">
            <TextInput v-model="active.text" block placeholder="Name (optional)" />
            <div v-if="active.points.length || active.objectBox" class="mk__chips">
              <span v-if="active.objectBox" class="mk__chip mk__chip--obj">
                Box {{ active.objectBox.x1 - active.objectBox.x0 }} x {{ active.objectBox.y1 - active.objectBox.y0 }}
                <button class="mk__chip-x" type="button" aria-label="Remove the box" @click="removeObjectBox">
                  <Icon name="x" :size="12" />
                </button>
              </span>
              <span
                v-for="(pt, i) in active.points"
                :key="i"
                class="mk__chip"
                :class="pt.positive ? 'mk__chip--in' : 'mk__chip--out'"
              >
                {{ pt.positive ? 'Include' : 'Exclude' }} {{ pt.x }}, {{ pt.y }}
                <button class="mk__chip-x" type="button" aria-label="Remove this click" @click="removePoint(i)">
                  <Icon name="x" :size="12" />
                </button>
              </span>
            </div>
            <button
              v-if="active.points.length || active.objectBox"
              class="pk-btn pk-btn--sm mk__clear"
              @click="clearClicks"
            >
              Clear clicks
            </button>
          </template>
          <template v-else>
          <TextInput
            v-model="active.text"
            block
            placeholder="A short description - a red car, a shoe"
            @keydown.enter="canFind && run(active)"
          />
          <div v-if="active.boxes.length" class="mk__chips">
            <span
              v-for="(b, i) in active.boxes"
              :key="i"
              class="mk__chip"
              :class="b.positive ? 'mk__chip--in' : 'mk__chip--out'"
            >
              {{ b.positive ? 'Include' : 'Exclude' }} {{ b.x1 - b.x0 }} x {{ b.y1 - b.y0 }}
              <button class="mk__chip-x" type="button" aria-label="Remove this box" @click="removeBox(i)">
                <Icon name="x" :size="12" />
              </button>
            </span>
          </div>
          <div class="mk__row">
            <span class="mk__label mk__label--sm">Score above</span>
            <div class="mk__slider">
              <Slider v-model="active.threshold" :min="0.05" :max="0.95" :step="0.05" />
            </div>
            <span class="mk__num">{{ active.threshold.toFixed(2) }}</span>
          </div>
          <button class="pk-btn pk-btn--primary" :disabled="!canFind" @click="run(active)">
            <Icon :name="active.busy ? 'spinner' : 'search'" :size="14" :class="{ mk__spin: active.busy }" />
            Find
          </button>
          </template>
          <p v-if="active.error" class="mk__error" role="alert">{{ active.error }}</p>
          <p v-if="active.result" class="mk__diag">
            <template v-if="active.kind === 'object'">
              {{ active.result.instances.length === 1 ? '1 candidate' : `${active.result.instances.length} candidates` }}
              · object present {{ active.result.presence.toFixed(2) }}
            </template>
            <template v-else>
              {{ active.result.instances.length }} found · concept present {{ active.result.presence.toFixed(2) }}
            </template>
            <br />
            <span class="mk__diag-t">{{ timingLine }}</span>
          </p>
        </section>

        <section v-if="activeInstances.length" class="mk__card">
          <span class="mk__label">{{ active.kind === 'object' ? 'Candidates' : 'Instances' }}</span>
          <ul class="mk__insts">
            <li
              v-for="it in activeInstances"
              :key="it.index"
              class="mk__inst"
              :class="{ 'mk__inst--on': it.on }"
              @mouseenter="hovered = { layer: active.id, index: it.index }"
              @mouseleave="hovered = null"
              @click="onInstanceClick(it.index)"
            >
              <Checkbox
                v-if="active.kind === 'concept'"
                :model-value="!active.hidden.has(it.index)"
                @update:model-value="(v) => toggleInstance(it.index, v)"
                @click.stop
              />
              <span class="mk__swatch" :style="{ background: rgbCss(it.rgb) }" />
              <span class="mk__inst-name">{{ it.index + 1 }}</span>
              <span class="mk__num">{{ it.score.toFixed(2) }}</span>
              <span class="mk__inst-size">{{ it.w }} x {{ it.h }} · {{ it.share.toFixed(1) }}%</span>
            </li>
          </ul>
        </section>

        <section v-if="anyResult" class="mk__card">
          <span class="mk__label">Export</span>
          <div class="mk__actions">
            <button class="pk-btn pk-btn--sm" @click="copyJson"><Icon name="copy" :size="13" /> Copy COCO JSON</button>
            <button class="pk-btn pk-btn--sm" @click="saveJson"><Icon name="download" :size="13" /> Save JSON</button>
            <button class="pk-btn pk-btn--sm" :disabled="!selected" @click="saveCutout('cutout')">
              <Icon name="download" :size="13" /> Cut-out PNG
            </button>
            <button class="pk-btn pk-btn--sm" :disabled="!selected" @click="saveCutout('mask')">
              <Icon name="download" :size="13" /> Mask PNG
            </button>
            <button class="pk-btn pk-btn--sm" @click="copyCurl"><Icon name="terminal" :size="13" /> Copy curl</button>
          </div>
        </section>
      </aside>
    </div>
    </div>
    </main>
  </div>
</template>

<style scoped>
/* the workspace of Tables and Reads: the side panel (or its rail) beside a
   pane that scrolls, the page in the Studio's one content width. The pane is
   the query container - whether the side cards fit beside the picture
   depends on the room the side panel and the GPU dock leave. */
.mk {
  display: flex;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}
.mk__main {
  flex: 1;
  min-width: 0;
  overflow: auto;
  padding: 32px 32px 0;
  container-type: inline-size;
}
.mk__inner {
  max-width: var(--pk-panel-width);
  margin: 0 auto;
  padding-bottom: 32px;
}
/* the history pages' one width rule (variables.css): Reads, Tables, Masks */
@container (min-width: 1100px) {
  .mk__inner {
    max-width: var(--pk-panel-width-wide);
  }
}
.mk__rail {
  flex: none;
  width: 48px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 12px 0;
  background: var(--pk-bg-surface);
  border-right: 1px solid var(--pk-border-default);
}
.mk__railbtn {
  width: 34px;
  height: 34px;
}
@media (max-width: 800px) {
  .mk :deep(.rsb) {
    width: 210px;
  }
  .mk__main {
    padding: 16px;
  }
}
.mk__nomodel {
  margin: 0 0 12px;
  padding: 8px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.mk__saveerr {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 12px;
}
.mk__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.mk__headctl {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
/* the Studio's segmented switch (Reads' mode switch, the same values) */
.mk__seg {
  display: inline-flex;
  flex: none;
  padding: 2px;
  gap: 2px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
}
.mk__seg :deep(.mk__segitem) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: transparent;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
}
.mk__seg :deep(.mk__segitem:hover) {
  color: var(--pk-text-primary);
}
.mk__seg :deep(.mk__segitem[data-state='on']) {
  background: var(--pk-bg-surface);
  color: var(--pk-text-primary);
  box-shadow: 0 0 0 1px var(--pk-border-default);
}
.mk__seg :deep(.mk__segitem[data-disabled]) {
  opacity: 0.5;
  cursor: not-allowed;
}
.mk__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.mk__lead {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.mk__none {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 64px 24px;
  text-align: center;
}
.mk__none-icon {
  color: var(--pk-text-muted);
}
.mk__none-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.mk__none-txt {
  margin: 0 0 6px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.mk__body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 16px;
  align-items: start;
}
@container (max-width: 760px) {
  .mk__body {
    grid-template-columns: minmax(0, 1fr);
  }
}
.mk__stage {
  display: flex;
  flex-direction: column;
  gap: 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px;
}
.mk__stage--over {
  border-color: var(--pk-accent);
}
.mk__drop {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  border: 1px dashed var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  min-height: 320px;
}
.mk__frame {
  position: relative;
  width: 100%;
  max-height: 72vh;
  margin: 0 auto;
  cursor: crosshair;
  touch-action: none;
  user-select: none;
  background: var(--pk-bg-base);
}
.mk__img,
.mk__overlay,
.mk__boxes {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.mk__img {
  object-fit: fill;
}
.mk__boxes {
  pointer-events: none;
}
.mk__box {
  fill: none;
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
  stroke-dasharray: 6 4;
}
.mk__box--in {
  stroke: var(--pk-status-success, #2e9e5b);
}
.mk__box--out {
  stroke: var(--pk-status-error);
}
.mk__box--obj {
  stroke: var(--pk-accent);
  stroke-dasharray: none;
}
.mk__box--draft {
  stroke-dasharray: 3 3;
}
/* a click marker sits on the picture, not on a themed surface: a light ring
   with a dark halo reads on any photo */
.mk__pt {
  position: absolute;
  width: 12px;
  height: 12px;
  margin: -6px 0 0 -6px;
  border-radius: 50%;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.55);
  pointer-events: none;
}
.mk__pt--in {
  background: var(--pk-status-success, #2e9e5b);
}
.mk__pt--out {
  background: var(--pk-status-error);
}
.mk__bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.mk__hint,
.mk__dims {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.mk__dims {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}
/* the camera's two short choices: the shared trigger's 220px floor would push
   the frame's own buttons onto a line of their own */
.mk__bar :deep(.pk-select) {
  min-width: 96px;
}
.mk__file {
  display: none;
}
/* The camera's frame is black in both themes - it is a video frame, not a
   surface - so the pill and the tags on it carry their own fixed contrast.
   The overlay is the frame's own size, fitted the same way as the video and
   mirrored with it, so a mask sits on its object; the frame's shape is the
   video's (camFit caps its height through its width), so the tags' percent
   positions land on the picture too. */
.mk__cam {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  margin: 0 auto;
  border-radius: var(--pk-radius-md);
  overflow: hidden;
  background: #000;
}
.mk__tag {
  position: absolute;
  margin: 3px 0 0 3px;
  padding: 1px 6px 1px 5px;
  border-left: 3px solid var(--tag);
  border-radius: 3px;
  font-size: var(--pk-font-size-xs, 0.72rem);
  font-variant-numeric: tabular-nums;
  line-height: 1.5;
  white-space: nowrap;
  color: #fff;
  background: rgba(10, 12, 16, 0.74);
  pointer-events: none;
}
.mk__cam-feed {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  transform: scaleX(-1);
}
.mk__cam-overlay {
  pointer-events: none;
}
.mk__cam-cover {
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
  background: var(--pk-bg-base);
}
.mk__cam-cover p {
  margin: 0;
  max-width: 36ch;
}
.mk__cam-pill {
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
.mk__cam-rate {
  color: rgba(255, 255, 255, 0.72);
}
.mk__cam-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #3ecf6e;
}
.mk__shots {
  list-style: none;
  margin: 0;
  padding: 2px 0 4px;
  display: flex;
  gap: 8px;
  overflow-x: auto;
}
.mk__shot {
  flex: none;
  width: 184px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 6px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
}
.mk__shot-img {
  display: block;
  width: 100%;
  border-radius: var(--pk-radius-sm);
  background: #000;
}
.mk__shot-row {
  display: flex;
  align-items: center;
  gap: 2px;
}
.mk__shot-time {
  margin-right: auto;
  font-size: var(--pk-font-size-xs, 12px);
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-muted);
}
.mk__side {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.mk__card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 14px 16px 16px;
}
.mk__row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.mk__slider {
  flex: 1;
  min-width: 0;
}
.mk__label {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.mk__row .mk__label {
  margin-right: auto;
}
.mk__label--sm {
  font-weight: 500;
  color: var(--pk-text-secondary);
  margin-right: 0;
  white-space: nowrap;
}
.mk__num {
  font-size: var(--pk-font-size-sm);
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-secondary);
}
.mk__layers,
.mk__insts {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mk__layer,
.mk__inst {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  cursor: pointer;
  border: 1px solid transparent;
}
.mk__layer--on,
.mk__inst--on {
  border-color: var(--pk-accent);
}
.mk__layer--static {
  cursor: default;
}
.mk__layer--off .mk__dot,
.mk__layer--off .mk__layer-name {
  opacity: 0.5;
}
.mk__addrow {
  display: flex;
  align-items: center;
  gap: 6px;
}
.mk__addrow > :first-child {
  flex: 1;
  min-width: 0;
}
.mk__group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mk__dot,
.mk__swatch {
  flex: none;
  width: 12px;
  height: 12px;
  border-radius: 50%;
}
.mk__swatch {
  border-radius: 3px;
}
.mk__layer-name,
.mk__inst-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
}
.mk__inst-name {
  flex: none;
  min-width: 18px;
}
.mk__inst-size {
  margin-left: auto;
  font-size: var(--pk-font-size-xs, 12px);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.mk__count {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.mk__clear {
  align-self: flex-start;
}
.mk__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.mk__chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 4px 2px 8px;
  border-radius: 999px;
  font-size: var(--pk-font-size-xs, 12px);
  font-variant-numeric: tabular-nums;
  background: var(--pk-bg-base);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
}
.mk__chip--in {
  border-color: var(--pk-status-success, #2e9e5b);
}
.mk__chip--out {
  border-color: var(--pk-status-error);
}
.mk__chip--obj {
  border-color: var(--pk-accent);
}
.mk__chip-x {
  display: inline-flex;
  border: 0;
  background: none;
  padding: 2px;
  color: var(--pk-text-muted);
  cursor: pointer;
  border-radius: 50%;
}
.mk__chip-x:hover {
  color: var(--pk-text-primary);
}
.mk__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.mk__diag {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.mk__diag-t {
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.mk__error {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-danger);
}
.mk__spin {
  animation: mk-spin 0.9s linear infinite;
}
@keyframes mk-spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .mk__spin {
    animation: none;
  }
}
</style>
