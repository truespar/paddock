// The Masks page's camera: concepts tracked through the webcam with a video
// session (POST /v1/masks/sessions), each object under one id and one colour
// from frame to frame. Up to the endpoint's `max_concepts` ride one session
// (the engine reads each frame once for all of them). One frame is in flight
// at a time and each is the frame of the moment it is sent, so the rate is
// the model's own. What is drawn is each frame's preview - the frame as it
// stands now; the session's final outputs trail by 15 frames (Meta's hot
// start) and a live view has no use for them. Frames are drawn down to the
// chosen long side before they go: the model reads 1008^2 whatever arrives,
// and the masks come back at the frame's size.
//
// The camera and the tracking are two switches. The camera is off until it
// is asked for; starting to track opens it; turning it off ends the tracking
// with it. The concepts are the page's - its prompts with words, as Reads'
// camera asks the read's own questions - and a change to them, or to the
// frame size, is a new session, so object ids start over. The preview is
// mirrored, as a camera facing the user is; the frame sent is not, the masks
// are drawn the way the preview shows them, and a snapshot keeps the frame as
// it was sent (the page keeps the snapshots, with its picture's record).
import { computed, nextTick, ref, shallowRef, watch, type Ref } from 'vue'
import type { SelectOption } from '@/components/ui/Select.vue'
import {
  decodeRle,
  dropSession,
  instanceRgb,
  paintOverlay,
  sendFrame,
  startSession,
  type DrawnMask,
  type VideoObject,
} from '@/lib/masks'

/** A concept the camera tracks; `slot` is its colour (its prompt's). */
export interface CamConcept {
  text: string
  slot: number
}

/** A frame from the camera, with what was tracked on it. */
export interface CamFrame {
  /** the frame as it was sent (not mirrored), a JPEG data URL */
  image: string
  width: number
  height: number
  objects: VideoObject[]
  /** the session's concepts, in the order an object's `concept` counts */
  concepts: CamConcept[]
}

/** An object's colour: its concept's hue, a shade by its id within it. */
export function objectRgb(o: VideoObject, concepts: CamConcept[]): [number, number, number] {
  const n = Math.max(1, concepts.length)
  return instanceRgb(concepts[o.concept]?.slot ?? o.concept, Math.floor(o.id / n))
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const im = new Image()
    im.onload = () => resolve(im)
    im.onerror = () => reject(new Error('the browser could not open this frame'))
    im.src = src
  })
}

/** A frame drawn with its masks on, its long side at most `side` (never up). */
export async function shotCanvas(s: CamFrame, side = Infinity): Promise<HTMLCanvasElement> {
  const k = Math.min(1, side / Math.max(s.width, s.height))
  const w = Math.max(1, Math.round(s.width * k))
  const h = Math.max(1, Math.round(s.height * k))
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d')
  if (!g) throw new Error('no 2D canvas in this browser')
  g.drawImage(await loadImage(s.image), 0, 0, w, h)
  if (s.objects.length) {
    const o = document.createElement('canvas')
    o.width = w
    o.height = h
    const og = o.getContext('2d')
    if (og) {
      paintOverlay(
        og,
        w,
        h,
        s.objects.map((x) => ({
          plane: decodeRle(x.mask),
          width: s.width,
          height: s.height,
          rgb: objectRgb(x, s.concepts),
          lit: false,
          box: x.box,
        })),
      )
      g.drawImage(o, 0, 0)
    }
  }
  return c
}

export function useMaskCamera(
  port: Ref<number>,
  concepts: Ref<CamConcept[]>,
  onError: (message: string) => void,
) {
  const video = ref<HTMLVideoElement | null>(null)
  const overlay = ref<HTMLCanvasElement | null>(null)
  /** `failed`: the camera is there and allowed but did not start or sends no
   *  picture - most often another app holding it */
  const status = ref<'off' | 'starting' | 'ready' | 'denied' | 'none' | 'failed'>('off')
  const statusText = ref('')
  const live = ref(false)

  /** the concepts of the session the objects came from */
  const tracked = shallowRef<CamConcept[]>([])
  const objects = shallowRef<VideoObject[]>([])
  const frameNo = ref(0)
  const ms = ref(0)
  /** the frames' size this session (what the masks come back at) */
  const frameSize = ref<[number, number] | null>(null)
  /** frames a second, from a smoothed gap between answers */
  const rate = ref(0)
  let gapEma = 0
  let lastDone = 0
  /** the frame on show and what was tracked on it - what a snapshot keeps */
  let shown: { image: string; width: number; height: number; objects: VideoObject[] } | null = null


  let stream: MediaStream | null = null
  let inflight: AbortController | null = null
  let session: string | null = null
  const devices = ref<MediaDeviceInfo[]>([])
  const deviceId = ref('')
  const deviceOptions = computed<SelectOption[]>(() =>
    devices.value.map((d, i) => ({ value: d.deviceId, label: d.label || `Camera ${i + 1}` })),
  )

  // the long side a frame is drawn to before it goes
  const side = ref(640)
  const sideOptions: SelectOption[] = [
    { value: 480, label: '480 px', hint: 'fastest' },
    { value: 640, label: '640 px' },
    { value: 960, label: '960 px' },
    { value: 1280, label: '1280 px', hint: 'finest masks' },
  ]

  /** Why the camera did not start, from the browser's error name. */
  function failed(e: unknown): void {
    const name = e instanceof DOMException ? e.name : ''
    console.warn('camera:', e)
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      status.value = 'none'
      statusText.value = 'No camera was found.'
    } else if (name === 'NotAllowedError' || name === 'SecurityError') {
      status.value = 'denied'
      statusText.value = 'The camera was not allowed. Allow it for this page in the browser and try again.'
    } else if (name === 'AbortError') {
      // Chromium gives a camera 10 s to send its first frame, then rejects
      // with "Timeout starting video source": the device opened and sent
      // nothing - a capture device with its source off, or one held elsewhere
      status.value = 'failed'
      statusText.value = `The camera opened but sent no picture (${(e as Error).message}). If it is a capture device, check that its source is on; if another app has it, close it there. Then try again.`
    } else if (name === 'NotReadableError') {
      status.value = 'failed'
      statusText.value = `The camera could not start (${(e as Error).message}) - another app may be holding it. Close it there and try again.`
    } else {
      status.value = 'failed'
      statusText.value = `The camera could not start: ${e instanceof Error ? e.message : String(e)}`
    }
  }

  /** True once the video shows a frame; false if none comes within `ms`. */
  function firstFrame(el: HTMLVideoElement, ms: number): Promise<boolean> {
    return new Promise((resolve) => {
      const t0 = performance.now()
      const look = () => {
        if (el.videoWidth > 0 && el.readyState >= el.HAVE_CURRENT_DATA) resolve(true)
        else if (performance.now() - t0 > ms) resolve(false)
        else window.setTimeout(look, 100)
      }
      look()
    })
  }

  // Every open bumps this, and close does too: what a superseded open's
  // awaits bring back is let go, so "Turn off" during a slow start wins.
  let opening = 0

  // A camera can keep its caller waiting with no error: the browser's
  // permission prompt, or a camera another app holds that opens but sends no
  // picture. Each wait says so instead of spinning.
  async function open(id?: string): Promise<void> {
    close()
    const me = ++opening
    status.value = 'starting'
    statusText.value = ''
    const slow = window.setTimeout(() => {
      if (me === opening && status.value === 'starting') {
        statusText.value = 'Waiting for the camera - if the browser asks for permission, allow it.'
      }
    }, 2500)
    let s: MediaStream
    try {
      s = await navigator.mediaDevices.getUserMedia({
        video: id ? { deviceId: { exact: id } } : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
    } catch (e) {
      // A camera that cannot start at 1280 x 720 may still start in its own
      // format (a capture device often sends only what its source sends):
      // one more try with no size asked, then the reason, as the browser gave it
      const name = e instanceof DOMException ? e.name : ''
      const retry = !id && (name === 'AbortError' || name === 'NotReadableError' || name === 'OverconstrainedError')
      let again: MediaStream | null = null
      let last = e
      if (retry && me === opening) {
        statusText.value = 'The camera did not start at 1280 x 720 - trying its own format...'
        try {
          again = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        } catch (e2) {
          last = e2
        }
      }
      if (!again) {
        window.clearTimeout(slow)
        if (me === opening) {
          failed(last)
          stop()
        }
        return
      }
      s = again
    }
    window.clearTimeout(slow)
    if (me !== opening) {
      for (const t of s.getTracks()) t.stop()
      return
    }
    stream = s
    statusText.value = ''
    // the stage swaps its "camera off" card for the video as this starts
    await nextTick()
    const el = video.value
    if (!el || me !== opening) {
      if (me === opening) close()
      return
    }
    el.srcObject = s
    // autoplay starts it; play() is only a nudge and is never waited on - a
    // stream with no frames leaves its promise pending forever
    el.play().catch(() => {})
    const ok = await firstFrame(el, 8000)
    if (me !== opening) return
    if (!ok) {
      close()
      status.value = 'failed'
      statusText.value =
        'The camera opened but sends no picture. If it is a capture device, check that its source is on; if another app has it, close it there. Then try again.'
      return
    }
    try {
      devices.value = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
    } catch {
      devices.value = []
    }
    if (me !== opening) return
    deviceId.value = s.getVideoTracks()[0]?.getSettings().deviceId ?? id ?? ''
    status.value = 'ready'
  }

  function close(): void {
    opening++
    inflight?.abort()
    inflight = null
    for (const t of stream?.getTracks() ?? []) t.stop()
    stream = null
    if (video.value) video.value.srcObject = null
    status.value = 'off'
  }

  /** The current frame at `side` on its long side (never up), as a JPEG data
   *  URL; null while the camera has no picture yet. A session takes one
   *  size, so the size is fixed at the session's first frame. */
  function capture(): string | null {
    const el = video.value
    if (!el || !el.videoWidth || !el.videoHeight) return null
    if (!frameSize.value) {
      const k = Math.min(1, side.value / Math.max(el.videoWidth, el.videoHeight))
      frameSize.value = [
        Math.max(1, Math.round(el.videoWidth * k)),
        Math.max(1, Math.round(el.videoHeight * k)),
      ]
    }
    const c = document.createElement('canvas')
    ;[c.width, c.height] = frameSize.value
    const g = c.getContext('2d')
    if (!g) return null
    g.drawImage(el, 0, 0, c.width, c.height)
    return c.toDataURL('image/jpeg', 0.9)
  }

  function draw(list: VideoObject[], w: number, h: number): void {
    const c = overlay.value
    const g = c?.getContext('2d')
    if (!c || !g) return
    if (c.width !== w || c.height !== h) {
      c.width = w
      c.height = h
    }
    const masks: DrawnMask[] = list.map((o) => ({
      plane: decodeRle(o.mask),
      width: w,
      height: h,
      rgb: objectRgb(o, tracked.value),
      lit: false,
      box: o.box,
    }))
    paintOverlay(g, w, h, masks)
  }

  function clear(): void {
    objects.value = []
    shown = null
    const c = overlay.value
    c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
  }

  const sleep = (t: number) => new Promise((r) => window.setTimeout(r, t))

  // Every start bumps the run; a loop, and what its awaits bring back, count
  // only while it is the current run - a restart's old frame neither stops
  // the new loop nor hands it a session made for the old concepts.
  let run = 0

  /** One frame through the session. False when the loop should stop. */
  async function step(me: number): Promise<boolean> {
    const frame = capture()
    if (!frame || !frameSize.value) {
      await sleep(100)
      return true
    }
    const ctrl = new AbortController()
    inflight = ctrl
    try {
      if (!session) {
        const list = concepts.value.map((c) => ({ ...c }))
        const id = await startSession(port.value, list.map((c) => c.text))
        if (me !== run) {
          dropSession(port.value, id)
          return false
        }
        session = id
        tracked.value = list
      }
      const r = await sendFrame(port.value, session, frame, ctrl.signal)
      if (me !== run || ctrl.signal.aborted) return false
      if ('gone' in r) {
        // idle too long (a hidden tab), or the runner restarted: a new one
        session = null
        return true
      }
      const objs = r.preview?.objects ?? []
      objects.value = objs
      shown = { image: frame, width: r.width, height: r.height, objects: objs }
      frameNo.value = r.frame
      ms.value = Math.round(r.ms)
      draw(objs, r.width, r.height)
    } catch (e) {
      if (me !== run || ctrl.signal.aborted) return false
      onError(e instanceof Error ? e.message : String(e))
      return false
    } finally {
      if (inflight === ctrl) inflight = null
    }
    const now = performance.now()
    if (lastDone) {
      const gap = now - lastDone
      gapEma = gapEma ? gapEma * 0.7 + gap * 0.3 : gap
      rate.value = 1000 / gapEma
    }
    lastDone = now
    return true
  }

  async function loop(me: number): Promise<void> {
    while (live.value && me === run) {
      // a hidden tab sends nothing: the GPU is not spent on frames nobody sees
      if (document.hidden || status.value !== 'ready') {
        await sleep(250)
        continue
      }
      const ok = await step(me)
      if (me !== run) return
      if (!ok) {
        stop()
        return
      }
    }
  }

  function endSession(): void {
    if (session) dropSession(port.value, session)
    session = null
    frameSize.value = null
  }

  const canTrack = computed(() => concepts.value.length > 0 && status.value !== 'starting')
  /** Start tracking, turning the camera on first when it is not. */
  async function start(): Promise<void> {
    if (live.value || !canTrack.value) return
    if (status.value !== 'ready') await open()
    if (status.value !== 'ready' || live.value) return
    live.value = true
    gapEma = 0
    lastDone = 0
    rate.value = 0
    void loop(++run)
  }
  function stop(): void {
    if (!live.value) return
    live.value = false
    inflight?.abort()
    inflight = null
    endSession()
    clear()
  }
  function toggle(): void {
    if (live.value) stop()
    else void start()
  }

  // new concepts or a new frame size are a new session; the loop goes on
  function restart(): void {
    inflight?.abort()
    inflight = null
    endSession()
    clear()
    if (live.value) void loop(++run)
  }

  // the page's concepts changed (words or colours): a new session, or none
  watch(
    () => concepts.value.map((c) => `${c.slot}:${c.text}`).join('\n'),
    () => {
      if (!live.value) return
      if (concepts.value.length) restart()
      else stop()
    },
  )

  watch(side, restart)
  watch(port, () => {
    stop()
    clear()
  })
  watch(deviceId, (id, old) => {
    if (old && id && id !== old) {
      const was = live.value
      stop()
      void open(id).then(() => {
        if (was) void start()
      })
    }
  })

  /** The frame on show, to keep: what was tracked on it when tracking, the
   *  camera's own frame at its full size when not. */
  function snapshot(): CamFrame | null {
    if (live.value && shown) return { ...shown, concepts: tracked.value }
    const el = video.value
    if (status.value !== 'ready' || !el?.videoWidth || !el.videoHeight) return null
    const c = document.createElement('canvas')
    c.width = el.videoWidth
    c.height = el.videoHeight
    c.getContext('2d')?.drawImage(el, 0, 0)
    return { image: c.toDataURL('image/jpeg', 0.92), width: c.width, height: c.height, objects: [], concepts: [] }
  }

  /** The camera off: the tracking ends, the stream is released. */
  function shut(): void {
    stop()
    clear()
    close()
  }

  return {
    video,
    overlay,
    status,
    statusText,
    live,
    tracked,
    objects,
    frameNo,
    ms,
    rate,
    frameSize,
    devices,
    deviceId,
    deviceOptions,
    side,
    sideOptions,
    canTrack,
    open,
    shut,
    stop,
    toggle,
    snapshot,
  }
}
