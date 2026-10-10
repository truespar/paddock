<script setup lang="ts">
// A document-parser answer's extraction, on a surface card of its own - an
// extraction is a structured artifact, not prose (the transcript rule). One
// scrollbar: the thread's - an inner scroller here made three nested
// scrollbars. Inside a compare lane the lane already is
// the card, so it stays flat. Export lives in the message's standard
// actions row, not here.
//
// Two shapes: a docRun turn fanned out one request per page and renders page
// sections with live states (section i is the pane's page i by
// construction); an older single-request turn renders its one joined stream.
import { copyText } from '@/lib/clipboard'
import { computed, nextTick, ref, watch } from 'vue'
import type { DocPage, DocRunMeta, Message, OcrRegion } from '@/types/chat'
import { useChatStore } from '@/stores/chat'
import { docContexts, pageImages } from '@/lib/docrun'
import { UNSURE_BELOW, cleanOcrText, htmlTablesToMarkdown, parseRegionsLive } from '@/lib/ocr'
import Markdown from './Markdown.vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const props = withDefaults(
  defineProps<{
    message: Message
    /** the how-it-was-read echo chips (deepseek's ocr extension; empty on
     *  families without one) */
    facts: { label: string; value: string }[]
    droppedText: boolean
    renderText: string
    /** inside a compare lane: the lane is the card. Explicit default - an
     *  unbound optional Boolean casts to false and silently never fires. */
    flat?: boolean
  }>(),
  { flat: false },
)

const docRun = computed<DocRunMeta | undefined>(() => props.message.docRun)

// ── figure crops (the official demo places the document's own pictures in
// the result): image/figure-labeled regions cut from the pane's rendered
// pages by pure CSS background math - no pixel work ─────────────────────────
const chat = useChatStore()
const runDocId = computed(
  () => docContexts(chat.active).find((c) => c.run?.id === props.message.id)?.source.id,
)
// pictures by every family's name for them (PaddleOCR-VL's document
// pipeline keeps charts and seals as pictures too)
const FIG = /^(image|figure|picture|chart|seal)$/i
interface Crop {
  label: string
  style: Record<string, string>
}
function cropsFor(i: number): Crop[] {
  const id = runDocId.value
  if (!id) return []
  const page = pageImages.get(id)?.[i]
  if (!page) return []
  const p = docRun.value?.pages[i]
  if (!p) return []
  const regions: OcrRegion[] =
    p.state === 'reading' ? parseRegionsLive(p.text) : (p.regions ?? [])
  const out: Crop[] = []
  for (const r of regions) {
    if (!FIG.test(r.label)) continue
    for (const b of r.boxes) {
      const w = Math.max(b[2] - b[0], 1)
      const h = Math.max(b[3] - b[1], 1)
      // background-position %: p% of the image aligns with p% of the box
      const px = 999 - w > 0 ? (b[0] / (999 - w)) * 100 : 0
      const py = 999 - h > 0 ? (b[1] / (999 - h)) * 100 : 0
      out.push({
        label: r.label,
        style: {
          width: `${Math.min((w / 999) * 100, 100)}%`,
          aspectRatio: String((w * page.w) / (h * page.h)),
          backgroundImage: `url(${page.src})`,
          backgroundSize: `${(999 / w) * 100}% auto`,
          backgroundPosition: `${px}% ${py}%`,
        },
      })
    }
  }
  return out
}

// the how-it-was-read echo as one muted line, values forward - four stacked
// label rows under every answer was chrome
const factsLine = computed(() => {
  const parts = props.facts.map((f) => {
    if (f.label === 'Read as') return `Read as ${f.value}`
    if (f.label === 'Image tokens') return `${f.value} image tokens`
    if (f.label === 'Pages') return `${f.value} pages`
    if (f.label === 'Regions') return `${f.value} regions`
    return f.value
  })
  return parts.join(' · ')
})

const STATE_COPY: Record<string, string> = {
  queued: 'Queued',
  reading: 'Reading...',
  done: 'Done',
  review: 'Needs review',
  error: 'Failed',
}

/** grounding markup lifted out, HTML tables converted to real markdown
 *  tables - the extraction renders as a document, not as mixed markup */
function pageText(raw: string): string {
  return htmlTablesToMarkdown(cleanOcrText(raw))
}

// per-page copy: one page's cleaned extraction to the clipboard - the
// whole-run save lives in the message actions, but a single page is what a
// user most often wants out
const copiedPage = ref(-1)
async function copyPage(i: number, raw: string): Promise<void> {
  try {
    await copyText(pageText(raw))
    copiedPage.value = i
    setTimeout(() => (copiedPage.value = -1), 1500)
  } catch {
    /* clipboard denied - the button just doesn't confirm */
  }
}

/** A layout-map page's answer is mostly REGION RECORDS, which the cleaner
 *  lifts out - the boxes live on the pages, and a finished page with a blank
 *  body here read as broken. Say what came back instead. */
function emptyNote(p: { state: string; text: string; regions?: { boxes: unknown[] }[] }): string {
  if (p.state !== 'done' || pageText(p.text).trim()) return ''
  const n = (p.regions ?? []).reduce((a, r) => a + r.boxes.length, 0)
  return n
    ? `${n} region${n === 1 ? '' : 's'} mapped - drawn on the page.`
    : 'Nothing came back for this page.'
}

// ── the trust layer: unsure words from the logprobs include.
// TranscriptView's one-band mark carried over (never a rainbow), but not its
// toggle: swapping the rendered document for a word wall read as broken
// Instead the scores are VISIBLE - a chip strip under
// the page header names each unsure word with its percentage - and the same
// words are highlighted in place inside the rendered document by decorating
// the finished DOM (text-node walk; injecting markup through markdown would
// break tables and escape rules).
function unsureWords(p: DocPage): { w: string; c: number }[] {
  const seen = new Set<string>()
  const out: { w: string; c: number }[] = []
  for (const w of p.words ?? []) {
    if (w.c < UNSURE_BELOW && w.w.length > 1 && !seen.has(w.w)) {
      seen.add(w.w)
      out.push(w)
    }
  }
  return out
}
function pct(c: number): string {
  return `${Math.round(c * 100)}%`
}

// Highlight each unsure word where it sits in the rendered extraction. Runs
// after the page finishes (the markdown DOM is stable then) and is
// idempotent - an already-marked node has no bare text left to match.
const pageEls = ref<Record<number, HTMLElement>>({})
function setPageEl(i: number, el: unknown): void {
  if (el instanceof HTMLElement) pageEls.value[i] = el
}
function markUnsure(i: number): void {
  const root = pageEls.value[i]
  const p = docRun.value?.pages[i]
  if (!root || !p) return
  const words = unsureWords(p)
  if (!words.length) return
  const escaped = words.map((w) => w.w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const re = new RegExp(`(${escaped.join('|')})`, 'g')
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) =>
      n.parentElement?.closest('.dvr__mark, .dvr__figs, code, pre')
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  })
  const targets: Text[] = []
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (re.test(n.textContent ?? '')) targets.push(n as Text)
    re.lastIndex = 0
  }
  for (const node of targets) {
    const parts = (node.textContent ?? '').split(re)
    if (parts.length < 2) continue
    const frag = document.createDocumentFragment()
    for (let k = 0; k < parts.length; k++) {
      if (k % 2 === 1) {
        const m = document.createElement('mark')
        m.className = 'dvr__mark'
        m.textContent = parts[k]
        frag.appendChild(m)
      } else if (parts[k]) {
        frag.appendChild(document.createTextNode(parts[k]))
      }
    }
    node.replaceWith(frag)
  }
}
watch(
  () => docRun.value?.pages.map((p) => `${p.state}:${p.words?.length ?? 0}`).join(),
  async () => {
    await nextTick()
    // one more frame: the markdown renderer commits its final DOM after the
    // reactive flush that flipped the state
    requestAnimationFrame(() => {
      docRun.value?.pages.forEach((p, i) => {
        if (p.state !== 'reading' && p.state !== 'queued') markUnsure(i)
      })
    })
  },
  { immediate: true },
)
</script>

<template>
  <div class="dvr" :class="{ 'dvr--flat': flat }">
    <div v-if="droppedText" class="dvr__note">
      Your typed message was replaced by the selected reading mode's instruction -
      this model takes one or the other, never both.
    </div>
    <div>
      <template v-if="docRun">
        <section
          v-for="(p, i) in docRun.pages"
          :key="i"
          class="dvr__pagesec"
        >
          <header v-if="docRun.pages.length > 1 || p.state !== 'done'" class="dvr__pagehd">
            <span v-if="docRun.pages.length > 1" class="dvr__pageno">Page {{ i + 1 }}</span>
            <span class="dvr__state" :class="`dvr__state--${p.state}`">
              <Icon v-if="p.state === 'reading'" name="spinner" :size="11" class="dvr__spin" />
              {{ STATE_COPY[p.state] }}
            </span>
            <span v-if="p.note" class="dvr__pagenote">{{ p.note }}</span>
            <Tooltip v-if="p.text && p.state !== 'reading'" :label="copiedPage === i ? 'Copied' : 'Copy this page'">
              <button class="dvr__pagecopy" type="button" @click="copyPage(i, p.text)">
                <Icon :name="copiedPage === i ? 'check' : 'copy'" :size="12" />
              </button>
            </Tooltip>
          </header>
          <div
            v-if="p.state !== 'reading' && unsureWords(p).length"
            class="dvr__unsure"
          >
            <span class="dvr__unsurehd">Check these words against the page</span>
            <span v-for="w in unsureWords(p)" :key="w.w" class="dvr__uchip">
              {{ w.w }} <em>{{ pct(w.c) }}</em>
            </span>
          </div>
          <div :ref="(el) => setPageEl(i, el)">
            <Markdown
              v-if="pageText(p.text).trim()"
              :content="pageText(p.text)"
              :streaming="p.state === 'reading'"
            />
            <p v-else-if="emptyNote(p)" class="dvr__pagenote">{{ emptyNote(p) }}</p>
          </div>
          <div v-if="cropsFor(i).length" class="dvr__figs">
            <figure v-for="(c, ci) in cropsFor(i)" :key="ci" class="dvr__fig">
              <div class="dvr__figimg" :style="c.style" />
              <figcaption class="dvr__figcap">{{ c.label }}</figcaption>
            </figure>
          </div>
        </section>
      </template>
      <template v-else>
        <Markdown v-if="renderText" :content="pageText(renderText)" :streaming="message.streaming" />
        <div v-else-if="message.streaming" class="dvr__typing"><span /><span /><span /></div>
      </template>
    </div>
    <p v-if="factsLine" class="dvr__facts">{{ factsLine }}</p>
  </div>
</template>

<style scoped>
/* Extraction typography is DOCUMENT scale, not chat scale: a front page's
   headline arrives as an h1, and chat-prose sizing rendered it enormous.
   Headings step modestly, the body sits at sm. */
.dvr {
  min-width: 0;
  font-size: var(--pk-font-size-sm);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  padding: 12px 14px;
}
.dvr--flat {
  background: none;
  border: 0;
  padding: 0;
}
.dvr :deep(.pk-md h1) {
  font-size: 1.15em;
  margin: 0.7em 0 0.35em;
}
.dvr :deep(.pk-md h2) {
  font-size: 1.08em;
  margin: 0.6em 0 0.3em;
}
.dvr :deep(.pk-md h3),
.dvr :deep(.pk-md h4),
.dvr :deep(.pk-md h5) {
  font-size: 1em;
  margin: 0.5em 0 0.25em;
}
.dvr :deep(.pk-md p) {
  margin: 0.35em 0;
}
.dvr :deep(.pk-md table) {
  font-size: 0.95em;
}
.dvr__note {
  margin-bottom: 8px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.dvr__pagesec {
  padding: 2px 0 10px;
}
.dvr__pagesec + .dvr__pagesec {
  border-top: 1px solid var(--pk-border-subtle);
  padding-top: 10px;
}
.dvr__pagehd {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 4px;
}
.dvr__pageno {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
}
.dvr__state {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.dvr__state--review {
  color: var(--pk-status-warning);
}
.dvr__state--error {
  color: var(--pk-status-error);
}
.dvr__spin {
  animation: dvr-spin 0.9s linear infinite;
}
@keyframes dvr-spin {
  to {
    transform: rotate(360deg);
  }
}
.dvr__pagenote {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* the document's own pictures, cropped from the rendered pages */
.dvr__figs {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 8px 0 2px;
}
.dvr__fig {
  margin: 0;
  max-width: 46%;
  min-width: 120px;
  flex: 0 1 auto;
}
.dvr__figimg {
  width: 100%;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background-repeat: no-repeat;
}
.dvr__figcap {
  margin-top: 2px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* unsure words: visible scores in a chip strip, and the same words marked in
   place in the extraction - one band, no hover needed */
.dvr__unsure {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  margin: 2px 0 8px;
}
.dvr__unsurehd {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  margin-right: 2px;
}
/* the -subtle tokens sit at 8-12% alpha, which reads as nothing on either
   theme - the marks mix their own visible tint instead */
.dvr__uchip {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  padding: 1px 8px;
  border: 1px solid color-mix(in srgb, var(--pk-status-error) 45%, transparent);
  border-radius: var(--pk-radius-sm);
  background: color-mix(in srgb, var(--pk-status-error) 14%, transparent);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
}
.dvr__uchip em {
  font-style: normal;
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
.dvr :deep(.dvr__mark) {
  color: inherit;
  background: color-mix(in srgb, var(--pk-status-error) 20%, transparent);
  text-decoration: underline wavy var(--pk-status-error);
  text-decoration-thickness: 1.5px;
  text-underline-offset: 3px;
  border-radius: 2px;
  padding: 0 1px;
}
.dvr__pagecopy {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  padding: 2px 4px;
  border: 0;
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
  border-radius: var(--pk-radius-sm);
}
.dvr__pagecopy:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
.dvr__facts {
  margin: 10px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.dvr__typing {
  display: inline-flex;
  gap: 4px;
  padding: 6px 0;
}
.dvr__typing span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--pk-text-muted);
  animation: dvr-bounce 1.2s infinite ease-in-out;
}
.dvr__typing span:nth-child(2) {
  animation-delay: 0.15s;
}
.dvr__typing span:nth-child(3) {
  animation-delay: 0.3s;
}
@keyframes dvr-bounce {
  0%,
  70%,
  100% {
    transform: translateY(0);
    opacity: 0.45;
  }
  35% {
    transform: translateY(-3px);
    opacity: 1;
  }
}
</style>
