<script setup lang="ts">
// The document's pages, flowing like any PDF/image viewer (
// no page rail, no captions - scroll is enough, and the scrollbar belongs to
// the pane column, not an inner box). Pictures come straight from the
// attachments table; PDF pages are rasterized client-side through the shared
// pdfium engine (RenderCapability.renderPage -> ImageBitmap -> blob URL).
// Rasterizing into a plain <img> stack keeps one uniform page abstraction,
// so the grounding overlay is percentage-box positioning (0-999 per axis,
// exact at any rendered size).
import { copyText } from '@/lib/clipboard'
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import type { DocumentCapability, DocumentId, RenderCapability } from '@truespar/lector-core'
import type { DocPage, FilePart, ImagePart, OcrRegion } from '@/types/chat'
import { pageRegionBoxes, regionBoxes, type RegionBox } from '@/lib/ocr'
import { pageImages, pageRangeBounds } from '@/lib/docrun'
import { attachmentsApi } from '@/lib/api'
import { pdfEngine } from '@/lib/pdf'
import Icon from '@/components/Icon.vue'
import Popover from '@/components/ui/Popover.vue'

const props = defineProps<{
  images: ImagePart[]
  pdf?: FilePart
  /** Grounded regions of a SINGLE-REQUEST run. Drawn on their own page when
   *  every region names it (PaddleOCR-VL's document pipeline), else only when
   *  the run had exactly one page - a region list without page indices on a
   *  multi-page run would be a guessed, wrong overlay. A fan-out run uses
   *  `runPages` instead, where the ambiguity never existed. */
  regions?: OcrRegion[]
  /** Per-page fan-out results, index-aligned with this stack:
   *  each page's own regions overlay that page, and the reading page is
   *  scrolled into view. */
  runPages?: DocPage[]
  /** the document's source-message id: rendered pages are published under
   *  it (lib/docrun.ts pageImages) for the result column's figure crops */
  docId?: string
}>()

interface Page {
  key: string
  label: string
  src: string | null
  state: 'loading' | 'ready' | 'error'
  /** pixel dimensions when known (PDF rasters always; images from the part) */
  w?: number
  h?: number
  /** browser cannot decode this format (TIFF) - the model still read it */
  undisplayable?: boolean
  /** A server-decoded JPEG to fall back to when the original will not render.
   *  HEIC is the case: it is HEVC, so no browser but Safari decodes it, while
   *  an iPhone writes it by default. Cleared once tried, so a rendition that
   *  also fails ends at the honest "couldn't render" tile rather than looping. */
  rendition?: string
  /** The file's name, on IMAGE pages only. PDF pages deliberately have none:
   *  "page 3" under page 3 is noise, and the no-captions rule was
   *  written about exactly that. A turn can carry several photos, though, and
   *  an unlabelled stack of them cannot answer "which file is this?" - which
   * is the whole question the pane is open for. */
  caption?: string
}

const emit = defineEmits<{
  /** The widest page's own pixel width. The host's zoom is a scale of the
   *  picture rather than a fraction of the pane, so it needs a number only
   *  this component can know: pictures carry theirs on the part (or on the
   *  decoded <img>), PDF pages carry the raster's. Emitted 0 when the stack
   *  is rebuilt, so a new document cannot inherit the old one's scale.
   */
  (e: 'natural', width: number): void
}>()

const pages = reactive<Page[]>([])
const pdfError = ref<string | null>(null)
// widest page seen so far; only ever raised, and reset with the stack
let natural = 0
function bumpNatural(w: number): void {
  if (!w || w <= natural) return
  natural = w
  emit('natural', w)
}
/** A picture whose part carried no dimensions reports them once decoded -
 *  which is also the only moment a data-URL or thumbnail can. */
function onImgLoad(e: Event, i: number): void {
  const el = e.target as HTMLImageElement
  const p = pages[i]
  if (p && !p.w) {
    p.w = el.naturalWidth
    p.h = el.naturalHeight
  }
  bumpNatural(el.naturalWidth)
}
/** The original would not render. Try the server's decoded copy once, then
 *  stop: a rendition that fails too (no image decoder installed, so the manager
 *  answers 501) has to land on the "couldn't render" tile rather than retry
 *  forever. */
function onImgError(i: number): void {
  const p = pages[i]
  if (!p) return
  if (p.rendition && p.src !== p.rendition) {
    p.src = p.rendition
    p.rendition = undefined
    return
  }
  p.state = 'error'
}
// blob URLs created for rendered PDF pages; revoked on teardown
const urls: string[] = []
// monotonic token: a slow PDF render finishing after the source changed
// must not append pages to the new document's stack
let gen = 0

function imageSrc(p: ImagePart): string | undefined {
  return (
    (p.attachmentId ? attachmentsApi.url(p.attachmentId) : undefined) ??
    p.modelUrl ??
    p.dataUrl ??
    p.thumbUrl
  )
}

/** Rendered width for PDF pages: crisp at column width without holding
 *  full-print rasters for every page of a long document. */
const PDF_RENDER_WIDTH = 1100

async function loadPdf(part: FilePart, mine: number): Promise<void> {
  if (!part.attachmentId) {
    pdfError.value = 'original file is not stored'
    return
  }
  try {
    const res = await fetch(attachmentsApi.url(part.attachmentId))
    if (!res.ok) throw new Error(`fetch ${res.status}`)
    const buf = await res.arrayBuffer()
    if (mine !== gen) return
    const engine = await pdfEngine()
    if (mine !== gen) return
    const doc = engine.plugins.get<DocumentCapability>('document')
    const render = engine.plugins.get<RenderCapability>('render')
    const handle = await doc.load(buf)
    const docId: DocumentId = handle.id
    try {
      if (mine !== gen) return
      // the part's page range bounds both sides: what the run read is what
      // the pane shows, and index i stays page i
      const [p0, p1] = pageRangeBounds(part.pageRange, handle.pageCount)
      const start = pages.length
      for (let i = p0; i < p1; i++) {
        pages.push({ key: `pdf-${i}`, label: `Page ${i + 1}`, src: null, state: 'loading' })
      }
      // sequential deliberately: one worker, and page 1 ready beats page N started
      for (let i = p0; i < p1; i++) {
        if (mine !== gen) return
        const size = handle.pageSizes[i]
        const h = size ? Math.round((PDF_RENDER_WIDTH * size.height) / size.width) : PDF_RENDER_WIDTH
        try {
          const bmp = await render.renderPage(docId, i, PDF_RENDER_WIDTH, h)
          const canvas = document.createElement('canvas')
          canvas.width = bmp.width
          canvas.height = bmp.height
          canvas.getContext('2d')?.drawImage(bmp, 0, 0)
          bmp.close()
          const url = await new Promise<string | null>((resolve) =>
            canvas.toBlob((b) => resolve(b ? URL.createObjectURL(b) : null), 'image/png'),
          )
          if (mine !== gen) {
            if (url) URL.revokeObjectURL(url)
            return
          }
          const page = pages[start + (i - p0)]
          if (url) {
            urls.push(url)
            page.src = url
            page.w = canvas.width
            page.h = canvas.height
            page.state = 'ready'
            bumpNatural(canvas.width)
          } else {
            page.state = 'error'
          }
        } catch {
          if (mine === gen) pages[start + (i - p0)].state = 'error'
        }
      }
    } finally {
      // free the worker's copy; the shared engine lives on
      try {
        await doc.close(docId)
      } catch {
        /* ignore */
      }
    }
  } catch (e) {
    if (mine === gen) pdfError.value = e instanceof Error ? e.message : String(e)
  }
}

function rebuild(): void {
  const mine = ++gen
  for (const u of urls.splice(0)) URL.revokeObjectURL(u)
  pages.splice(0)
  pdfError.value = null
  natural = 0
  emit('natural', 0)
  props.images.forEach((p, i) => {
    const tiff = (p.mime ?? '').includes('tiff') || /\.tiff?$/i.test(p.name ?? '')
    pages.push({
      key: `img-${i}`,
      label: p.name ?? `page ${i + 1}`,
      caption: p.name,
      src: tiff ? null : (imageSrc(p) ?? null),
      w: p.width,
      h: p.height,
      state: tiff ? 'error' : 'ready',
      undisplayable: tiff,
      // Only the ORIGINAL is shown first - a rendition is a re-encode, and the
      // pane should show the real file whenever the browser can.
      rendition: p.attachmentId ? attachmentsApi.renditionUrl(p.attachmentId) : undefined,
    })
    if (p.width) bumpNatural(p.width)
  })
  if (props.pdf) void loadPdf(props.pdf, mine)
}

watch(() => [props.images, props.pdf] as const, rebuild, { immediate: true })
// publish the rendered pages for the result column's figure crops
watch(
  pages,
  () => {
    if (props.docId) {
      pageImages.set(
        props.docId,
        pages.map((p) => (p.src && p.w && p.h ? { src: p.src, w: p.w, h: p.h } : null)),
      )
    }
  },
  { deep: true, immediate: true },
)
onBeforeUnmount(() => {
  gen++
  if (props.docId) pageImages.delete(props.docId)
  for (const u of urls.splice(0)) URL.revokeObjectURL(u)
})

// Overlay boxes per page: a fan-out run pins each page's own regions to it;
// the single-request shape draws only on a lone displayable page (see the
// regions prop doc).
// per-box copy feedback for the click popover
const copiedBox = ref('')
async function copyBox(key: string, text: string): Promise<void> {
  try {
    await copyText(text)
    copiedBox.value = key
    setTimeout(() => (copiedBox.value = ''), 1500)
  } catch {
    /* clipboard denied - the button just doesn't confirm */
  }
}

const pageBoxes = computed<RegionBox[][]>(() => {
  // While a page reads, boxes come from a live parse of its raw stream - the
  // marking advances with the text (the official demo's behavior). Once done
  // the server's terminal parse is authoritative. (Shared with the lector
  // pane's overlay - lib/ocr.ts pageRegionBoxes.)
  if (props.runPages) {
    return pages.map((_, i) => {
      const p = props.runPages?.[i]
      return p ? pageRegionBoxes(p) : []
    })
  }
  const regions = props.regions ?? []
  if (regions.length && regions.every((r) => typeof r.page === 'number')) {
    return pages.map((_, i) => regionBoxes(regions.filter((r) => r.page === i)))
  }
  if (!regions.length || pages.length !== 1 || props.pdf) return pages.map(() => [])
  return [regionBoxes(regions)]
})
const legend = computed(() => {
  const counts = new Map<string, { n: number; hue: number }>()
  for (const boxes of pageBoxes.value) {
    for (const b of boxes) {
      const e = counts.get(b.label) ?? { n: 0, hue: b.hue }
      e.n += 1
      counts.set(b.label, e)
    }
  }
  return [...counts].map(([label, e]) => ({ label, n: e.n, hue: e.hue }))
})
const focus = ref('')

// Follow the work: while a run reads page i, the COLUMN
// (the pane's scroll container above us) keeps that page in view. Never
// scrollIntoView - it would yank every ancestor, the thread included.
const rootEl = ref<HTMLElement | null>(null)
const stackEl = ref<HTMLElement | null>(null)
watch(
  () => props.runPages?.map((p) => p.state).join(''),
  () => {
    const i = props.runPages?.findIndex((p) => p.state === 'reading') ?? -1
    if (i < 0) return
    const el = stackEl.value?.querySelectorAll<HTMLElement>('.dpg__page')[i]
    const container = rootEl.value?.parentElement
    if (!el || !container) return
    const top =
      el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop
    container.scrollTo({ top: Math.max(top - 8, 0), behavior: 'smooth' })
  },
)
</script>

<template>
  <div ref="rootEl" class="dpg">
    <div ref="stackEl" class="dpg__stack">
      <figure v-for="(p, pi) in pages" :key="p.key" class="dpg__page">
        <div class="dpg__frame">
          <img
            v-if="p.src"
            :src="p.src"
            :alt="p.label"
            draggable="false"
            @load="onImgLoad($event, pi)"
            @error="onImgError(pi)"
          />
          <div v-else-if="p.state === 'loading'" class="dpg__wait">
            <Icon name="spinner" :size="18" class="dpg__spin" />
          </div>
          <div v-else class="dpg__miss">
            <Icon name="file-text" :size="20" />
            <span v-if="p.undisplayable">
              TIFF pages can't be previewed here yet - the model still read them.
            </span>
            <span v-else>Couldn't render this page.</span>
          </div>
          <Popover v-for="(b, i) in pageBoxes[pi] ?? []" :key="i" side="bottom" align="start">
            <template #trigger>
              <button
                type="button"
                class="dpg__box"
                :class="{
                  'dpg__box--dim': focus && focus !== b.label,
                  'dpg__box--unsure': b.unsure,
                }"
                :style="{ ...b.style, '--hue': String(b.hue) }"
                :aria-label="b.label"
              />
            </template>
            <div class="dpgpop">
              <div class="dpgpop__hd">
                <span class="dpgpop__label" :style="{ '--hue': String(b.hue) }">{{ b.label }}</span>
                <span v-if="b.unsure" class="dpgpop__flag">has unsure words</span>
                <button
                  v-if="b.text"
                  class="dpgpop__copy"
                  type="button"
                  @click="copyBox(`${pi}:${i}`, b.text)"
                >
                  <Icon :name="copiedBox === `${pi}:${i}` ? 'check' : 'copy'" :size="12" />
                  {{ copiedBox === `${pi}:${i}` ? 'Copied' : 'Copy text' }}
                </button>
              </div>
              <p v-if="b.text" class="dpgpop__text">{{ b.text }}</p>
            </div>
          </Popover>
        </div>
        <figcaption v-if="p.caption" class="dpg__caption">{{ p.caption }}</figcaption>
      </figure>
      <div v-if="pdfError" class="dpg__miss">
        <Icon name="file-text" :size="20" />
        <span>Couldn't render the PDF: {{ pdfError }}</span>
      </div>
    </div>
    <div v-if="legend.length" class="dpg__legend">
      <button
        v-for="l in legend"
        :key="l.label"
        type="button"
        class="dpg__key"
        :class="{ 'dpg__key--dim': focus && focus !== l.label }"
        :style="{ '--hue': String(l.hue) }"
        @mouseenter="focus = l.label"
        @mouseleave="focus = ''"
        @focus="focus = l.label"
        @blur="focus = ''"
      >
        <span class="dpg__dot" />
        {{ l.label }}
        <span class="dpg__n">{{ l.n }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
/* position:relative anchors page offset math for the auto-follow scroll */
.dpg {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.dpg__stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dpg__page {
  margin: 0;
}
/* Only images get one, so this never turns into a "page 3" rail. */
.dpg__caption {
  margin: 4px 2px 0;
  font-size: var(--pk-font-size-xs);
  line-height: 1.4;
  color: var(--pk-text-muted);
  overflow-wrap: anywhere;
}
.dpg__frame {
  position: relative;
  line-height: 0;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  overflow: hidden;
  background: var(--pk-bg-surface);
}
.dpg__frame img {
  width: 100%;
  height: auto;
  display: block;
  /* the pane pans by dragging; the browser's own image drag would otherwise
     win the gesture and hand the file to whatever it is dropped on */
  -webkit-user-drag: none;
  user-select: none;
}
.dpg__wait {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 180px;
  color: var(--pk-text-muted);
}
.dpg__spin {
  animation: dpg-spin 0.9s linear infinite;
}
@keyframes dpg-spin {
  to {
    transform: rotate(360deg);
  }
}
.dpg__miss {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  line-height: 1.4;
}
/* hue = label, white hairline keeps any hue readable on any page ink */
.dpg__box {
  position: absolute;
  padding: 0;
  appearance: none;
  border: 1.5px solid hsl(var(--hue) 75% 45%);
  box-shadow: inset 0 0 0 1px rgb(255 255 255 / 0.65);
  background: hsl(var(--hue) 75% 50% / 0.08);
  border-radius: 2px;
  transition: opacity 0.15s;
  cursor: pointer;
}
.dpg__box:hover {
  background: hsl(var(--hue) 75% 50% / 0.18);
}
/* a box whose words the model was unsure about: the error mark replaces the
   label hue, so the doubt is visible on the PAGE, not only in the result */
.dpg__box--unsure {
  border-color: var(--pk-status-error);
  border-style: dashed;
  background: color-mix(in srgb, var(--pk-status-error) 16%, transparent);
}
.dpg__box--unsure:hover {
  background: color-mix(in srgb, var(--pk-status-error) 28%, transparent);
}
.dpg__box--dim {
  opacity: 0.15;
}
.dpg__legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  line-height: normal;
}
.dpg__key {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 1px 2px;
  border: 0;
  background: none;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  cursor: default;
  transition: opacity 0.15s;
}
.dpg__key--dim {
  opacity: 0.4;
}
.dpg__dot {
  width: 9px;
  height: 9px;
  border-radius: 2px;
  border: 1.5px solid hsl(var(--hue) 75% 45%);
  background: hsl(var(--hue) 75% 50% / 0.15);
}
.dpg__n {
  color: var(--pk-text-muted);
  opacity: 0.7;
}
</style>

<!-- Unscoped deliberately: the box popover is portalled to <body> by Reka, so a
     scoped hash never reaches it. All classes are dpgpop-prefixed. -->
<style>
.dpgpop {
  max-width: 340px;
}
.dpgpop__hd {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dpgpop__label {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: hsl(var(--hue) 70% 40%);
}
[data-theme='dark'] .dpgpop__label {
  color: hsl(var(--hue) 70% 65%);
}
.dpgpop__flag {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-status-error);
}
.dpgpop__copy {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  cursor: pointer;
}
.dpgpop__copy:hover {
  color: var(--pk-text-primary);
  border-color: var(--pk-border-strong);
}
.dpgpop__text {
  margin: 8px 0 0;
  max-height: 180px;
  overflow-y: auto;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
