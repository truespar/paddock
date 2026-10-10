// The OCR surface (the document-parser families): reading-mode copy for
// the composer control, the wire↔store mapping for the server's `ocr`
// response extension, and the display cleaning for grounded output.
//
// The mode LIST always comes from the endpoint (/api/server `ocr.modes`,
// computed from what is serving) - this file only decides what to call each
// mode, same division of labor as tasks.ts. An unadvertised mode is never
// offered; an advertised one we have no copy for still gets a usable label.

import type { OcrMeta, OcrRegion } from '@/types/chat'

/** What the endpoint advertises about its `ocr` request object. */
export interface OcrCaps {
  modes: string[]
  crops: string[]
  /** grounded decodes return the parsed `ocr.regions` extension */
  grounding: boolean
}

/** Parse the /api/server `ocr` object; null/absent/malformed = no surface. */
export function ocrCapsFrom(v: unknown): OcrCaps | undefined {
  if (!v || typeof v !== 'object') return undefined
  const o = v as { modes?: unknown; crops?: unknown; grounding?: unknown }
  const strings = (x: unknown): string[] =>
    Array.isArray(x) ? x.filter((s): s is string => typeof s === 'string') : []
  const modes = strings(o.modes)
  if (!modes.length) return undefined
  return { modes, crops: strings(o.crops), grounding: o.grounding === true }
}

/** Plain-language names and hints for the modes we know. The wire id is
 *  never shown: a person reading a scan should not have to learn the
 *  family's task vocabulary to pick how it is read. */
const MODE_COPY: Record<string, { label: string; hint: string }> = {
  document: {
    label: 'Document',
    hint: 'Structured text - headings, paragraphs and tables.',
  },
  multipage: {
    label: 'Pages of one document',
    hint: 'Reads several pictures as consecutive pages.',
  },
  free: { label: 'Plain text', hint: 'Just the words, no structure.' },
  layout: {
    label: 'Layout map',
    hint: 'Labels each region of the page and where it sits.',
  },
  figure: { label: 'Figure', hint: 'Reads a chart or figure.' },
  // the paddleocr vocabulary (its six official task prompts)
  ocr: { label: 'Text', hint: 'All the text, in reading order.' },
  table: { label: 'Table', hint: 'Reads a table into structured data.' },
  formula: { label: 'Formula', hint: 'Math, as LaTeX.' },
  chart: { label: 'Chart', hint: 'Reads a chart into data.' },
  spotting: { label: 'Text spotting', hint: 'Words with their position on the page.' },
  seal: { label: 'Seal', hint: 'Reads stamp and seal text.' },
  // LightOnOCR-3's two trained prompts
  plain: { label: 'Markdown', hint: 'The page as markdown, tables and charts as data.' },
  grounding: { label: 'Blocks with boxes', hint: 'Every block of the page with its label and box.' },
}

/** The composer's automatic entry: no mode rides and the server derives one
 *  from the request shape - which one is the family's own default
 *  (deepseek: one picture = document, several = pages; paddleocr: its
 *  authors' document pipeline where the layout model is installed, else
 *  text; LightOnOCR-3: markdown). */
export const OCR_AUTO = {
  label: 'Automatic',
  hint: "The model's own default reading - no mode is sent.",
}

export function ocrModeLabel(mode: string): string {
  return MODE_COPY[mode]?.label ?? mode.charAt(0).toUpperCase() + mode.slice(1)
}

export function ocrModeHint(mode: string): string {
  return MODE_COPY[mode]?.hint ?? ''
}

/** The server's `ocr` response extension -> the persisted OcrMeta, or
 *  undefined when the value isn't the extension object (non-OCR turns). */
export function ocrMetaFromWire(v: unknown): OcrMeta | undefined {
  if (!v || typeof v !== 'object') return undefined
  const o = v as Record<string, unknown>
  const num = (x: unknown): number | undefined => (typeof x === 'number' ? x : undefined)
  const meta: OcrMeta = {
    mode: typeof o.mode === 'string' ? o.mode : null,
    crop: typeof o.crop === 'string' ? o.crop : undefined,
    grounding: o.grounding === true,
    pages: num(o.pages),
    views: num(o.views),
    tiles: num(o.tiles),
    imageTokens: num(o.image_tokens),
    passThrough: o.pass_through === true,
    droppedText: o.dropped_text === true,
  }
  const rs = o.repetition_stop as { fired?: unknown } | null | undefined
  if (rs && rs.fired === true) meta.repetitionStop = true
  const regions = regionsFromWire(o.regions)
  if (regions.length) meta.regions = regions
  return meta
}

function regionsFromWire(v: unknown): OcrRegion[] {
  if (!Array.isArray(v)) return []
  const out: OcrRegion[] = []
  for (const r of v) {
    if (!r || typeof r !== 'object') continue
    const { label, boxes, text, quads, continues, page, bbox, image, group, unread } = r as {
      label?: unknown
      boxes?: unknown
      text?: unknown
      quads?: unknown
      continues?: unknown
      page?: unknown
      bbox?: unknown
      image?: unknown
      group?: unknown
      unread?: unknown
    }
    if (typeof label !== 'string' || !Array.isArray(boxes)) continue
    const bs = boxes.filter(
      (b): b is [number, number, number, number] =>
        Array.isArray(b) && b.length === 4 && b.every((n) => typeof n === 'number'),
    )
    const qs = Array.isArray(quads)
      ? quads.filter(
          (q): q is number[] =>
            Array.isArray(q) && q.length === 8 && q.every((n) => typeof n === 'number'),
        )
      : []
    if (bs.length) {
      out.push({
        label,
        boxes: bs,
        ...(typeof text === 'string' && text ? { text } : {}),
        ...(qs.length ? { quads: qs } : {}),
        ...(continues === true ? { continues: true } : {}),
        ...(typeof page === 'number' ? { page } : {}),
        ...(Array.isArray(bbox) && bbox.length === 4 && bbox.every((n) => typeof n === 'number')
          ? { bbox: bbox as [number, number, number, number] }
          : {}),
        ...(typeof image === 'string' ? { image } : {}),
        ...(typeof group === 'number' ? { group } : {}),
        ...(typeof unread === 'string' ? { unread } : {}),
      })
    }
  }
  return out
}

/** Client-side mirror of the runner's parse_regions (both reference forms,
 *  text spans included) - run over the RAW stream WHILE a page reads, so the
 *  pane's boxes appear in sync with the text, exactly the official demo's
 *  behavior (baidu/Unlimited-OCR visualization: the marking advances at the
 *  stream's speed). The server's terminal parse stays authoritative once the
 *  page finishes. */
export function parseRegionsLive(raw: string): OcrRegion[] {
  if (raw.includes('<|LOC_')) return parseSpottingLive(raw)
  if (raw.includes('![')) {
    const blocks = parseBlocksLive(raw)
    if (blocks.length) return blocks
  }
  if (!raw.includes('<|det|>')) return []
  const out: OcrRegion[] = []
  const boxesFrom = (s: string): [number, number, number, number][] | null => {
    try {
      const v: unknown = JSON.parse(s.trim())
      const one = (b: unknown): [number, number, number, number] | null =>
        Array.isArray(b) && b.length === 4 && b.every((n) => typeof n === 'number')
          ? (b as [number, number, number, number])
          : null
      if (Array.isArray(v) && v.every((n) => typeof n === 'number')) {
        const b = one(v)
        return b ? [b] : null
      }
      if (Array.isArray(v)) {
        const all = v.map(one)
        return all.every((b) => b !== null) && all.length
          ? (all as [number, number, number, number][])
          : null
      }
      return null
    } catch {
      return null
    }
  }
  // form 1: <|ref|>label<|/ref|><|det|>boxes<|/det|> (grounding; no text)
  const ref = /<\|ref\|>([\s\S]*?)<\|\/ref\|><\|det\|>([\s\S]*?)<\|\/det\|>/g
  for (const m of raw.matchAll(ref)) {
    const boxes = boxesFrom(m[2])
    if (boxes) out.push({ label: m[1].trim(), boxes })
  }
  // form 2: <|det|>label [box]<|/det|>Block text... (document; text span up to
  // the next marker)
  const det = /<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]*?\])\s*<\|\/det\|>/g
  for (const m of raw.matchAll(det)) {
    const boxes = boxesFrom(m[2])
    if (!boxes) continue
    const after = raw.slice((m.index ?? 0) + m[0].length)
    const cut = Math.min(
      ...['<|det|>', '<|ref|>'].map((k) => {
        const i = after.indexOf(k)
        return i < 0 ? after.length : i
      }),
    )
    const text = after.slice(0, cut).trim()
    out.push({ label: m[1], boxes, ...(text ? { text } : {}) })
  }
  return out
}

/** Client mirror of the runner's parse_spotting (paddleocr Spotting, probed
 *  live): one line per text instance, `text` + eight `<|LOC_n|>`
 *  tokens - a 4-corner quad on a 0..=1000 grid, rescaled here onto the same
 *  0-999 space every other region uses. `<|LOC_BEGIN|>`/`END`/`SEP` are
 *  tolerated as delimiters; a partial 8-run (mid-stream tail) is dropped
 *  until it closes, which is what makes this safe to run per frame. */
function parseSpottingLive(raw: string): OcrRegion[] {
  const out: OcrRegion[] = []
  const cluster = /(?:<\|LOC_(?:BEGIN|END|SEP|\d+)\|>)+/g
  let last = 0
  for (const m of raw.matchAll(cluster)) {
    const text = raw.slice(last, m.index).trim()
    last = (m.index ?? 0) + m[0].length
    if (!text) continue
    const vals = [...m[0].matchAll(/<\|LOC_(\d+)\|>/g)].map((v) => parseInt(v[1], 10))
    const boxes: [number, number, number, number][] = []
    const quads: number[][] = []
    for (let i = 0; i + 8 <= vals.length; i += 8) {
      const q = vals
        .slice(i, i + 8)
        .map((v) => Math.round((Math.min(Math.max(v, 0), 1000) * 999) / 1000))
      const xs = [q[0], q[2], q[4], q[6]]
      const ys = [q[1], q[3], q[5], q[7]]
      boxes.push([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)])
      quads.push(q)
    }
    if (boxes.length) out.push({ label: 'text', boxes, text, quads })
  }
  return out
}

/** LightOnOCR-3's grounding marker, its client's MARKER regex verbatim:
 *  `![label](x1,y1,x2,y2)` on a 0..=1000 grid, a trailing `+` on the label
 *  for a block continuing the previous one, and the spaces after it. */
const LO_MARKER = /!\[([A-Za-z_][\w-]*?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*/g
/** A marker the stream has not finished yet - hidden until it closes. */
const LO_TAIL = /!\[[A-Za-z_][\w-]*\+?(?:\]\([\d\s,]*)?$/

/** Client mirror of the runner's lighton_ocr::parse_blocks: each marker
 *  owns the text up to the next one, boxes rescaled onto the 0-999 space
 *  every other region uses. A cut marker at a mid-stream tail does not
 *  match, so this is safe per frame. */
function parseBlocksLive(raw: string): OcrRegion[] {
  const marks = [...raw.matchAll(LO_MARKER)]
  const s = (v: string) => Math.round((Math.min(parseInt(v, 10), 1000) * 999) / 1000)
  return marks.map((m, i): OcrRegion => {
    const end = (m.index ?? 0) + m[0].length
    const stop = i + 1 < marks.length ? (marks[i + 1].index ?? raw.length) : raw.length
    const text = raw.slice(end, stop).trim()
    return {
      label: m[1],
      boxes: [[s(m[3]), s(m[4]), s(m[5]), s(m[6])]],
      ...(text ? { text } : {}),
      ...(m[2] ? { continues: true } : {}),
    }
  })
}

// The grounded output rides its regions inside the text as special-token
// markup (the server decodes with specials kept, because the region parse
// needs them). Two reference forms, mirrored from the runner's parse_regions:
//   <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>   (grounding)
//   <|det|>type [x1,y1,x2,y2]<|/det|>Block text...            (document)
const REF_FORM = /<\|ref\|>([\s\S]*?)<\|\/ref\|><\|det\|>[\s\S]*?<\|\/det\|>/g
const DET_FORM = /<\|det\|>\s*[A-Za-z_][\w-]*\s*\[[^\]]*?\]\s*<\|\/det\|>\s*/g
// paddleocr Spotting's coordinate tokens - the boxes live in regions, the
// eyes just get the words
const LOC_FORM = /(?:<\|LOC_(?:BEGIN|END|SEP|\d+)\|>)+/g

/** The answer's text with the region markup lifted out for reading: a
 *  grounding record becomes its label, a document block record disappears
 *  (its text follows it anyway). The structured regions live in
 *  `OcrMeta.regions` - this is only what the eyes get. A text with no
 *  markup passes through untouched, so it is safe to run on every turn of
 *  an OCR lane, streaming included. LightOnOCR-3's block markers go too -
 *  left in, the markdown renderer would draw each as a broken image. */
// PaddleOCR-VL's document pipeline answers in its authors' own markdown, which
// centers captions and pictures in HTML divs. The renderer escapes raw HTML,
// so the display unwraps a caption to its words and drops a picture's
// reference - the pictures are drawn from their regions on the page instead.
const PD_PICTURE =
  /<div style="text-align: center;"><img src="imgs\/img_in_[\w-]+_box_\d+_\d+_\d+_\d+\.jpg" alt="Image" width="\d+%" \/><\/div>\n?/g
const PD_CENTER = /<div style="text-align: center;">([\s\S]*?)<\/div>\n?/g

export function cleanOcrText(raw: string): string {
  if (raw.includes('<div style="text-align: center;">')) {
    raw = raw.replace(PD_PICTURE, '').replace(PD_CENTER, '$1')
  }
  const blocks = raw.includes('![')
  if (!raw.includes('<|') && !blocks) return raw
  const out = raw
    .replace(REF_FORM, '$1')
    .replace(DET_FORM, '')
    .replace(LOC_FORM, '')
    .replace(/<\|grounding\|>/g, '')
  return blocks ? out.replace(LO_MARKER, '').replace(LO_TAIL, '') : out
}

/** Overlay geometry: one 0-999 box as percentages of the page image. */
export function boxPercent(b: [number, number, number, number]): {
  left: string
  top: string
  width: string
  height: string
} {
  const pc = (n: number) => `${((Math.min(Math.max(n, 0), 999) / 999) * 100).toFixed(2)}%`
  return {
    left: pc(b[0]),
    top: pc(b[1]),
    width: pc(Math.max(b[2] - b[0], 0)),
    height: pc(Math.max(b[3] - b[1], 0)),
  }
}

/** The OCR families emit raw HTML tables inside otherwise-markdown text.
 *  Rendered raw they show as literal markup; fenced they read as source -
 * neither is a document. Convert each closed
 *  <table>...</table> into a markdown pipe table: DOMParser is inert (no
 *  script execution), entities decode for free, and rowspan/colspan flatten
 *  into plain rows padded to the widest - pipe tables cannot express spans,
 *  and a readable flat table beats markup on screen. A table still streaming
 *  (unclosed) shows raw until its closing tag lands, then converts. */
export function htmlTablesToMarkdown(text: string): string {
  if (!text.includes('<table')) return text
  return text.replace(/<table[\s\S]*?<\/table>/gi, (m) => {
    try {
      const doc = new DOMParser().parseFromString(m, 'text/html')
      const rows = [...doc.querySelectorAll('tr')].map((tr) =>
        [...tr.querySelectorAll('td,th')].map((c) =>
          (c.textContent ?? '').trim().replace(/\s+/g, ' ').replace(/\|/g, '\\|'),
        ),
      )
      if (!rows.length) return ''
      const width = Math.max(...rows.map((r) => r.length))
      const line = (r: string[]) =>
        `| ${[...r, ...Array(width - r.length).fill('')].join(' | ')} |`
      const out = [line(rows[0]), `|${' --- |'.repeat(width)}`, ...rows.slice(1).map(line)]
      return `\n\n${out.join('\n')}\n\n`
    } catch {
      return m
    }
  })
}

/** Gzip compression ratio of a page's extraction - the whisper-side
 * compression_ratio guard applied to OCR. A decoder that
 *  collapsed into a repetition loop on illegible input compresses absurdly
 *  well (8-20×); real document text lands around 2-3.5×. Uses the browser's
 *  native CompressionStream; returns 1 (never flags) for short texts, where
 *  the ratio is meaningless, or when the API is unavailable. */
export async function degenerationRatio(text: string): Promise<number> {
  if (text.length < 400 || typeof CompressionStream === 'undefined') return 1
  try {
    const bytes = new TextEncoder().encode(text)
    const gz = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))
    const compressed = await new Response(gz).arrayBuffer()
    return bytes.length / Math.max(compressed.byteLength, 1)
  } catch {
    return 1
  }
}

/** Flag threshold for `degenerationRatio`. Borrowed-constant honesty (the
 *   lesson): whisper flags segments above 2.4 under zlib, but those
 *  are short segments - page-length prose normally reaches 2-3.5×, so the
 *  bar sits above that band. No measured false-positive rate yet. */
export const DEGENERATION_THRESHOLD = 4.5

/** The one confidence bar every OCR surface shares: a word under it gets
 *  marked (chips + in-text highlight + its page box). exp(mean logprob). */
export const UNSURE_BELOW = 0.45

/** One logprob entry from the Responses stream (`include:
 *  ["message.output_text.logprobs"]`). */
export interface LogprobEntry {
  token: string
  logprob: number
}

/** Fold a page's per-token logprobs into per-word confidence, TranscriptView's
 *  vocabulary: confidence = exp(mean token logprob), and the display marks
 *  only what falls under its unsure bar. Tokens are merged into words on
 *  whitespace; special-token markers (the region markup) are dropped - their
 *  probability says nothing about the words on the page. */
export function wordsFromLogprobs(entries: LogprobEntry[]): { w: string; c: number }[] {
  const out: { w: string; c: number }[] = []
  let word = ''
  let sum = 0
  let n = 0
  const flush = () => {
    if (word) out.push({ w: word, c: n ? Math.exp(sum / n) : 1 })
    word = ''
    sum = 0
    n = 0
  }
  for (const e of entries) {
    if (/^<\|[\s\S]*\|>$/.test(e.token)) continue
    // a token may open with the space that ends the previous word, and may
    // itself contain spaces (rare, but tokenizers do it) - split honestly
    const pieces = e.token.split(/(\s+)/)
    for (const p of pieces) {
      if (!p) continue
      if (/^\s+$/.test(p)) {
        flush()
      } else {
        word += p
        sum += e.logprob
        n += 1
      }
    }
  }
  flush()
  return out
}

/** One drawable region box: what both page surfaces (the image stack and the
 *  lector viewer overlay) render. Percentage geometry, so it holds at any
 *  zoom of its page element. */
export interface RegionBox {
  label: string
  hue: number
  style: { left: string; top: string; width: string; height: string }
  /** the block's own words (the click popover shows and copies them) */
  text: string
  /** contains a word the model scored under the confidence bar */
  unsure: boolean
}

/** Regions -> drawable boxes, with the page's unsure words folded in. */
export function regionBoxes(regions: OcrRegion[] | undefined, unsure: string[] = []): RegionBox[] {
  return (regions ?? []).flatMap((r) =>
    r.boxes.map((b) => ({
      label: r.label,
      hue: labelHue(r.label),
      style: boxPercent(b),
      text: r.text ?? '',
      unsure: !!r.text && unsure.some((w) => r.text!.includes(w)),
    })),
  )
}

/** A page's boxes for display: live parse of the raw stream WHILE it reads,
 *  the server's terminal regions once done - plus the unsure fold. */
export function pageRegionBoxes(p: {
  state: string
  text: string
  regions?: OcrRegion[]
  words?: { w: string; c: number }[]
}): RegionBox[] {
  const unsure = (p.words ?? []).filter((w) => w.c < UNSURE_BELOW).map((w) => w.w)
  if (p.state === 'reading') return regionBoxes(parseRegionsLive(p.text), unsure)
  return regionBoxes(p.regions, unsure)
}

/** A stable hue per region label, so every "title" box wears the same color
 *  in one answer and the next. Plain string hash onto the wheel, avoiding
 *  nothing: the boxes carry a white inner hairline so any hue reads. */
export function labelHue(label: string): number {
  let h = 0
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0
  return h % 360
}
