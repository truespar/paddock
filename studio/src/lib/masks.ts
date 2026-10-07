// The Masks page's wire types and pixel work: SAM 3 over POST /v1/masks, a
// data-URL picture in and COCO RLE masks out, decoded and drawn here. Two
// tasks share the route: a concept (words and/or exemplar boxes) gets every
// kept instance with its score and box; clicks and/or an object box get the
// one object under them as up to three candidates, scored by predicted IoU,
// plus a `refine_id` that lets the next click build on this answer.
//
// COCO RLE is COLUMN-major: runs walk down column 0, then column 1, ... -
// zeros first. Decoding it straight into a row-major plane transposes the
// mask, which only shows on a non-square picture; `decodeRle` writes it
// row-major once and everything downstream reads rows.

/** The endpoint's own contract, from its `/server` block. */
export interface MaskCaps {
  maxPixels: number
  maxBoxes: number
  maxPromptTokens: number
  /** click prompts are served (an older pack has no click heads) */
  clicks: boolean
  maxPoints: number
  /** video sessions are served, or why not */
  video: boolean
  videoUnavailable: string | null
  /** concepts one video session tracks */
  maxConcepts: number
}

export function capsFrom(server: unknown): MaskCaps | null {
  const m = (server as { masks?: Record<string, unknown> | null } | null)?.masks
  if (!m) return null
  const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
  return {
    maxPixels: num(m.max_pixels, 24_000_000),
    maxBoxes: num(m.max_boxes, 16),
    maxPromptTokens: num(m.max_prompt_tokens, 30),
    clicks: m.clicks === true,
    maxPoints: m.clicks === true ? num(m.max_points, 32) : 0,
    video: m.video === true,
    videoUnavailable: typeof m.video_unavailable === 'string' ? m.video_unavailable : null,
    maxConcepts: num(m.max_concepts, 1),
  }
}

/** An exemplar box in the picture's pixels; negative = "not this". */
export interface PromptBox {
  x0: number
  y0: number
  x1: number
  y1: number
  positive: boolean
}

/** A click in the picture's pixels: on the object (positive) or off it. */
export interface PromptPoint {
  x: number
  y: number
  positive: boolean
}

export interface MaskInstance {
  score: number
  /** x0, y0, x1, y1 in the picture's pixels */
  box: [number, number, number, number]
  area: number
  mask: { size: [number, number]; counts: number[] }
}

export interface MaskResponse {
  object: 'masks'
  model: string
  width: number
  height: number
  presence: number
  instances: MaskInstance[]
  prompt_tokens: number
  timings: {
    resize_ms: number
    encode_ms: number
    prompt_ms: number
    detect_ms: number
    masks_ms: number
    image_reused?: boolean
    text_reused?: boolean
  }
  /** a click answer's handle: send it back as `refine` with more clicks */
  refine_id?: string
}

export function requestBody(
  model: string,
  image: string,
  text: string,
  boxes: PromptBox[],
  threshold: number,
): Record<string, unknown> {
  const body: Record<string, unknown> = { model, image, threshold }
  const t = text.trim()
  if (t) body.text = t
  if (boxes.length) {
    body.boxes = boxes.map((b) => ({ box: [b.x0, b.y0, b.x1, b.y1], positive: b.positive }))
  }
  return body
}

/** A click request: the object under `points` and/or inside `box`, refining
 *  the answer `refine` names when it is given. */
export function clickBody(
  model: string,
  image: string,
  points: PromptPoint[],
  box: PromptBox | null,
  refine?: string,
): Record<string, unknown> {
  const body: Record<string, unknown> = { model, image }
  if (points.length) body.points = points.map((p) => ({ point: [p.x, p.y], positive: p.positive }))
  if (box) body.object_box = [box.x0, box.y0, box.x1, box.y1]
  if (refine) body.refine = refine
  return body
}

/** What a click answer was asked with, and the handle it gave back. */
export interface ClickAsk {
  points: PromptPoint[]
  box: PromptBox | null
  refineId: string
}

/** Whether the next click request may refine the last answer: only when it
 *  is that request plus more clicks. Taking a click away or moving the box
 *  starts over - the old mask still carries what was removed. */
export function refineFor(last: ClickAsk | null, points: PromptPoint[], box: PromptBox | null): string | undefined {
  if (!last || points.length <= last.points.length) return undefined
  const sameBox =
    (!last.box && !box) ||
    (!!last.box && !!box && last.box.x0 === box.x0 && last.box.y0 === box.y0 && last.box.x1 === box.x1 && last.box.y1 === box.y1)
  if (!sameBox) return undefined
  const prefix = last.points.every((p, i) => p.x === points[i].x && p.y === points[i].y && p.positive === points[i].positive)
  return prefix ? last.refineId : undefined
}

/** The same request as curl, the picture elided (it is the request's bulk). */
export function curlFor(port: number, body: Record<string, unknown>): string {
  const shown = { ...body, image: 'data:image/jpeg;base64,...' }
  const json = JSON.stringify(shown)
  return [
    `curl http://localhost:${port}/v1/masks \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "Authorization: Bearer <api key>" \\`,
    `  -d '${json.replace(/'/g, "'\\''")}'`,
    `# image: your picture as a data: URL (JPEG or PNG, base64)`,
  ].join('\n')
}

/** The mask as a row-major 0/1 plane of `width * height` bytes. */
export function decodeRle(rle: { size: [number, number]; counts: number[] }): Uint8Array {
  const [h, w] = rle.size
  const out = new Uint8Array(w * h)
  let pos = 0 // column-major index
  let on = false
  for (const c of rle.counts) {
    if (on) {
      for (let i = pos; i < pos + c; i++) {
        const x = (i / h) | 0
        out[(i - x * h) * w + x] = 1
      }
    }
    pos += c
    on = !on
  }
  return out
}

/** Instance colours: one hue family a prompt (layer), shades within it, so two
 *  prompts read apart and instances of one still read as distinct. */
const HUES = [205, 28, 140, 330, 55, 265, 175, 0]
export function layerHue(layer: number): number {
  return HUES[layer % HUES.length]
}
export function instanceRgb(layer: number, index: number): [number, number, number] {
  const hue = layerHue(layer)
  const light = [52, 64, 44, 72, 38][index % 5]
  return hslToRgb(hue, 78, light)
}
export function rgbCss([r, g, b]: [number, number, number], a = 1): string {
  return `rgba(${r}, ${g}, ${b}, ${a})`
}
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const sn = s / 100
  const ln = l / 100
  const k = (n: number) => (n + h / 30) % 12
  const a = sn * Math.min(ln, 1 - ln)
  const f = (n: number) => ln - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)]
}

export interface DrawnMask {
  plane: Uint8Array
  width: number
  height: number
  rgb: [number, number, number]
  /** emphasised (hovered or selected) */
  lit: boolean
  /** the instance's box in the picture's pixels - only it is scanned */
  box: [number, number, number, number]
}

/** Paint masks onto an overlay of `ow x oh` (a scaled view of the picture):
 *  a translucent fill and an opaque edge, nearest-sampled from each plane. */
export function paintOverlay(ctx: CanvasRenderingContext2D, ow: number, oh: number, masks: DrawnMask[]): void {
  const img = ctx.createImageData(ow, oh)
  const px = img.data
  for (const m of masks) {
    const sx = m.width / ow
    const sy = m.height / oh
    const at = (ox: number, oy: number) => {
      if (ox < 0 || oy < 0 || ox >= ow || oy >= oh) return 0
      return m.plane[((oy * sy) | 0) * m.width + ((ox * sx) | 0)]
    }
    const fill = m.lit ? 150 : 95
    // the box, a pixel wider each side, in overlay pixels
    const bx0 = Math.max(0, Math.floor(m.box[0] / sx) - 1)
    const by0 = Math.max(0, Math.floor(m.box[1] / sy) - 1)
    const bx1 = Math.min(ow, Math.ceil(m.box[2] / sx) + 1)
    const by1 = Math.min(oh, Math.ceil(m.box[3] / sy) + 1)
    for (let oy = by0; oy < by1; oy++) {
      for (let ox = bx0; ox < bx1; ox++) {
        if (!at(ox, oy)) continue
        const edge = !at(ox - 1, oy) || !at(ox + 1, oy) || !at(ox, oy - 1) || !at(ox, oy + 1)
        const o = (oy * ow + ox) * 4
        const a = edge ? 255 : fill
        // over-composite onto what an earlier mask left
        const t = a / 255
        px[o] = Math.round(m.rgb[0] * t + px[o] * (1 - t))
        px[o + 1] = Math.round(m.rgb[1] * t + px[o + 1] * (1 - t))
        px[o + 2] = Math.round(m.rgb[2] * t + px[o + 2] * (1 - t))
        px[o + 3] = Math.max(px[o + 3], a)
      }
    }
  }
  ctx.putImageData(img, 0, 0)
}

/** COCO results, one entry an instance: what evaluation scripts and labelling
 *  tools read. COCO's bbox is x, y, width, height. */
export function cocoJson(
  width: number,
  height: number,
  layers: { prompt: string; instances: MaskInstance[] }[],
): string {
  let id = 0
  const annotations = layers.flatMap((l) =>
    l.instances.map((i) => ({
      id: ++id,
      category: l.prompt,
      score: Number(i.score.toFixed(4)),
      bbox: [i.box[0], i.box[1], i.box[2] - i.box[0], i.box[3] - i.box[1]].map((v) => Number(v.toFixed(2))),
      area: i.area,
      segmentation: i.mask,
    })),
  )
  return JSON.stringify({ image: { width, height }, annotations }, null, 2)
}

/** A cut-out: the picture with everything outside the mask transparent, at
 *  the picture's own size. */
export async function cutoutPng(picture: HTMLImageElement, plane: Uint8Array, w: number, h: number): Promise<Blob> {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) throw new Error('no 2D canvas in this browser')
  ctx.drawImage(picture, 0, 0, w, h)
  const img = ctx.getImageData(0, 0, w, h)
  for (let i = 0; i < w * h; i++) if (!plane[i]) img.data[i * 4 + 3] = 0
  ctx.putImageData(img, 0, 0)
  return canvasBlob(c)
}

/** The mask itself, white on black, at the picture's size. */
export async function maskPng(plane: Uint8Array, w: number, h: number): Promise<Blob> {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) throw new Error('no 2D canvas in this browser')
  const img = ctx.createImageData(w, h)
  for (let i = 0; i < w * h; i++) {
    const v = plane[i] ? 255 : 0
    img.data[i * 4] = v
    img.data[i * 4 + 1] = v
    img.data[i * 4 + 2] = v
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return canvasBlob(c)
}

function canvasBlob(c: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error('the browser could not encode a PNG'))), 'image/png'),
  )
}

export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** A file name from a prompt: letters, digits and dashes. */
export function slug(s: string): string {
  return s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'mask'
}

// ── video sessions: concepts tracked through camera frames ─────────────────
// POST /v1/masks/sessions starts one, each frame goes to its /frames, DELETE
// drops it. An output is final 15 frames late (Meta's hot start); a live view
// draws each frame's `preview` - the frame as it stands now.

/** One tracked object on one frame: the same id from frame to frame, unique
 *  across the session's concepts. */
export interface VideoObject {
  id: number
  /** which of the session's concepts it was found for */
  concept: number
  score: number
  box: [number, number, number, number]
  area: number
  mask: { size: [number, number]; counts: number[] }
}

export interface VideoFrame {
  frame: number
  objects: VideoObject[]
}

export interface FrameResponse {
  frame: number
  width: number
  height: number
  frames: VideoFrame[]
  preview?: VideoFrame
  ms: number
}

/** A failed call's message: the runner's error, or the status. */
async function failure(res: Response): Promise<string> {
  const json: unknown = await res.json().catch(() => null)
  return (json as { error?: { message?: string } } | null)?.error?.message ?? `HTTP ${res.status}`
}

/** Start a session tracking `texts`; its id. One concept goes as `text`,
 *  which every endpoint with sessions takes. */
export async function startSession(port: number, texts: string[]): Promise<string> {
  const res = await fetch(`/api/runners/${port}/v1/masks/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(texts.length === 1 ? { text: texts[0], preview: true } : { texts, preview: true }),
  })
  if (!res.ok) throw new Error(await failure(res))
  return ((await res.json()) as { id: string }).id
}

/** The session's next frame (a data URL). A 404 means the session is gone
 *  (idle too long, or the runner restarted): `gone` is set. */
export async function sendFrame(
  port: number,
  id: string,
  image: string,
  signal: AbortSignal,
): Promise<FrameResponse | { gone: true }> {
  const res = await fetch(`/api/runners/${port}/v1/masks/sessions/${id}/frames`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image }),
    signal,
  })
  if (res.status === 404) return { gone: true }
  if (!res.ok) throw new Error(await failure(res))
  return (await res.json()) as FrameResponse
}

/** Drop the session; nothing is waited for or reported. */
export function dropSession(port: number, id: string): void {
  void fetch(`/api/runners/${port}/v1/masks/sessions/${id}`, { method: 'DELETE', keepalive: true }).catch(
    () => {},
  )
}
