<script setup lang="ts">
// Start a model, step 1 (/manage/models/start): pick it. Slim vendor blocks
// with the real brand marks, then one compact comparison table per vendor
// (cards were too bulky -): why pick it / trade-offs in
// plain words from the catalog, and a short will-it-run verdict against this
// machine (the full sentence rides the cell's tooltip). The verdict here is
// a screening call; the workload page re-prices live and stays binding.
// Picking routes to /manage/models/start/:model - the workload step.
import { computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useRegistryStore } from '@/stores/registry'
import { useFleetStore } from '@/stores/fleet'
import { useReadinessStore } from '@/stores/readiness'
import { archBlocked, archBlockReason, lowestFloor } from '@/lib/arch-floor'
import type { CatalogModel } from '@/lib/api'
import { fmtVram as gb, fmtBytes } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const route = useRoute()
const router = useRouter()
const reg = useRegistryStore()
const fleet = useFleetStore()
const ready = useReadinessStore()

let release: (() => void) | null = null
onMounted(async () => {
  release = fleet.hold()
  if (!reg.models.length) await reg.refresh()
  if (!reg.envelope) await reg.estimate()
})
onUnmounted(() => release?.())

// ── vendor blocks ────────────────────────────────────────────────────────────
// Block titles use the brand people search for; the maker stays visible in the
// subtitle (loud attribution).
const BRANDS: Record<string, { title: string; sub: string }> = {
  Alibaba: { title: 'Qwen', sub: 'by Alibaba' },
  'ConvAI Innovations': { title: 'Laya', sub: 'by ConvAI Innovations' },
  OpenAI: { title: 'OpenAI', sub: 'GPT-OSS' },
  Google: { title: 'Google', sub: 'Gemma' },
  // The Nordic speech vendors are national-library and research groups rather
  // than model labs, and the sub-line is where that belongs - "KBLab" alone
  // means nothing to someone looking for a Swedish transcriber.
  KBLab: { title: 'KBLab', sub: 'National Library of Sweden' },
  'NB AI-Lab': { title: 'NB AI-Lab', sub: 'National Library of Norway' },
  'CoRal Project': { title: 'CoRal', sub: 'Danish speech, Alexandra Institute' },
}
const VENDOR_ORDER = ['Alibaba', 'OpenAI', 'Google']
const CAP_META = [
  { key: 'chat', icon: 'message-square', label: 'Text' },
  { key: 'vision', icon: 'image', label: 'Vision' },
  // Reads images, but only some of them: granite-vision is built for charts,
  // tables and documents and IBM says so out loud. Its own chip rather than
  // the general one, or the picker promises image chat it will do badly.
  { key: 'documents', icon: 'file-text', label: 'Documents & charts' },
  { key: 'embeddings', icon: 'search', label: 'Embeddings' },
  { key: 'rerank', icon: 'arrow-up-down', label: 'Rerank' },
  // Speech to text. The whisper family serves only this - it refuses chat
  // outright - so the chip is not decoration on a chat model, it is the
  // whole of what the row does.
  { key: 'transcription', icon: 'microphone', label: 'Speech to text' },
  // Word timing for an existing transcript (the karaoke enrichment) - like
  // transcription, the chip is the whole of what an aligner does.
  { key: 'alignment', icon: 'clock', label: 'Word timing' },
  // Typed questions about a text in, a calibrated answer each out - the Reads
  // page's model. It neither chats nor generates, so the chip is all it does.
  { key: 'decision', icon: 'list-checks', label: 'Decisions' },
  // Labelled rows in, the missing column predicted (a class or a number) -
  // the Tables page's model. It takes tables, never text, so the chip is all
  // it does.
  { key: 'tabular', icon: 'table', label: 'Tables' },
  // Aerial photographs in, maps out (what the ground is, how tall the trees
  // are). It reads images but it is no vision chat - it answers with rasters,
  // so it gets its own chip rather than borrowing 'Vision'.
  { key: 'segmentation', icon: 'map', label: 'Image to map' },
  // A prompt in, a picture out. Not a chat model and not a vision one - it
  // never reads an image - so it wears its own chip, and its row files under
  // its own section below.
  { key: 'image-generation', icon: 'image', label: 'Image generation' },
] as const

interface VendorGroup {
  vendor: string
  title: string
  sub: string
  models: CatalogModel[]
  caps: { icon: string; label: string }[]
}

const groups = computed<VendorGroup[]>(() => {
  const by = new Map<string, CatalogModel[]>()
  for (const m of reg.models) {
    const v = m.vendor ?? 'Other'
    by.set(v, [...(by.get(v) ?? []), m])
  }
  const vendors = [...by.keys()].sort(
    (a, b) =>
      (VENDOR_ORDER.indexOf(a) + 99 * +!VENDOR_ORDER.includes(a)) -
      (VENDOR_ORDER.indexOf(b) + 99 * +!VENDOR_ORDER.includes(b)),
  )
  return vendors.map((v) => {
    const models = by.get(v)!
    const capSet = new Set(models.flatMap((m) => m.capability))
    return {
      vendor: v,
      title: BRANDS[v]?.title ?? v,
      sub: BRANDS[v]?.sub ?? '',
      models,
      caps: CAP_META.filter((c) => capSet.has(c.key)),
    }
  })
})

/** The image icon on a model row, and what it honestly claims. Two different
 *  promises, so two different tooltips: a `vision` model will chat about any
 *  picture, a `documents` one is for charts, tables and paperwork. */
function imageCap(m: CatalogModel): { icon: string; title: string } | null {
  if (m.capability.includes('vision')) return { icon: 'image', title: 'Reads images' }
  if (m.capability.includes('documents'))
    return { icon: 'file-text', title: 'Reads charts, tables and documents' }
  return null
}

const sel = computed(() => String(route.query.vendor ?? ''))
const selGroup = computed(() => groups.value.find((g) => g.vendor === sel.value))
function pick(vendor: string): void {
  void router.replace({ query: { vendor } })
}

/** Table rows with a thin section header when a vendor spans model kinds. */
type Row = { kind: 'head'; label: string } | { kind: 'model'; m: CatalogModel }
function sectionOf(m: CatalogModel): string {
  if (m.capability.includes('embeddings')) return 'Embeddings'
  if (m.capability.includes('rerank')) return 'Rerankers'
  // Before this existed every non-embedding row fell through to 'Chat', which
  // would have filed four speech-to-text models under a heading for the one
  // thing they cannot do.
  if (m.capability.includes('transcription')) return 'Speech to text'
  // An aligner cannot chat either - it times the words of a transcript some
  // other model produced, so it files with the speech tools.
  if (m.capability.includes('alignment')) return 'Speech to text'
  // Answers questions with probabilities, never with text of its own.
  if (m.capability.includes('decision')) return 'Decisions'
  // Rows in, predictions out - no text surface at all.
  if (m.capability.includes('tabular')) return 'Tables'
  // Chips in, rasters out - it has no text surface at all, so filing it under
  // 'Chat' would promise the one thing it refuses.
  if (m.capability.includes('segmentation')) return 'Maps from imagery'
  // Text in, a picture out - no text surface either.
  if (m.capability.includes('image-generation')) return 'Image generation'
  return 'Chat'
}
const rows = computed<Row[]>(() => {
  const models = selGroup.value?.models ?? []
  const sections = new Set(models.map(sectionOf))
  const out: Row[] = []
  let last = ''
  for (const m of models) {
    const s = sectionOf(m)
    if (sections.size > 1 && s !== last) {
      out.push({ kind: 'head', label: s })
      last = s
    }
    out.push({ kind: 'model', m })
  }
  return out
})

// ── the will-it-run verdict ──────────────────────────────────────────────────
// Two different questions, never collapsed: can this machine ever run it
// (vs total VRAM), and can it run now (vs measured free VRAM, given what's
// already serving). One number answers both, downloaded or not: the registry
// publishes each artifact's shape, so the estimator prices a row the same way
// before and after the bytes land. The cell shows the short call;
// `full` rides the tooltip.
//
// There used to be a second path here - `bytes * 1.05 + tower + 1.5 GiB` for
// anything not yet downloaded, captioned "measured exactly once downloaded".
// It was never a measurement-vs-estimate split, just a worse estimate, and it
// moved the number under the user after a 30 GB download. Gone with its two
// helpers; a row with no shape now says so instead of guessing.
interface FitLine {
  tone: 'good' | 'ok' | 'warn' | 'bad' | 'dim'
  label: string
  full: string
}

function fitOf(m: CatalogModel): FitLine {
  const free = reg.estDevice?.free ?? 0
  const total = reg.estDevice?.total ?? 0
  // Screen out builds whose KERNELS this GPU does not have before pricing
  // VRAM against them. A model whose only weights artifact carries a floor
  // (nemotron ships NVFP4 alone) otherwise gets a green "Should fit" here and
  // a refusal at load - the exact silent failure this line exists to prevent.
  const cc = ready.info?.cc
  const all = m.artifacts.filter((a) => a.kind === 'weights')
  const weights = all.filter((a) => !archBlocked(a, cc))
  if (!weights.length && all.length) {
    // Name the CHEAPEST floor among the blocked builds - that is the silicon
    // that would unlock anything, rather than whichever artifact sorted first.
    const why = archBlockReason(lowestFloor(all) ?? all[0], cc)
    if (why)
      return {
        tone: 'bad',
        label: why,
        full: `${why} - no build of this model runs on your GPU`,
      }
  }
  if (!total && ready.info?.backend === 'metal' && !ready.blocked)
    return { tone: 'dim', label: 'Checked at load', full: 'Apple GPU memory telemetry is not available here. The Metal runner enforces its allocation budget when loading.' }
  if (!total) return { tone: 'dim', label: 'No GPU', full: 'No GPU detected - nothing can serve here yet' }
  const rows_ = weights.map((a) => ({
    a,
    est: reg.estimates[m.id]?.artifacts?.[a.id]?.estimate,
  }))

  const priced = rows_.filter((r) => r.est)
  if (priced.length) {
    const fitting = priced.filter((r) => r.est!.fit.verdict !== 'does_not_fit')
    if (fitting.length) {
      const best = fitting[0]
      const e = best.est!
      const tight = e.fit.verdict === 'tight'
      const atBest = best.a.id === rows_[0]?.a.id
      return {
        tone: tight ? 'ok' : 'good',
        label: atBest ? (tight ? 'Fits - tight' : 'Fits') : `Fits at ${best.a.label}`,
        full: `${atBest ? 'Fits your GPU' : `Fits at ${best.a.label} (${best.a.quant ?? best.a.id})`} - ${gb(e.resident)} to load, ${gb(free)} free${tight ? ' (tight)' : ''}`,
      }
    }
    // nothing fits as things stand: would stopping one running server do it?
    const need = Math.min(...priced.map((r) => r.est!.resident))
    const single = fleet.rows.find(
      (r) => !r.pinned && (r.vram?.self_mem ?? 0) + free >= need,
    )
    if (single)
      return {
        tone: 'warn',
        label: `Stop ${single.port} first`,
        full: `Fits if you stop ${single.port} - needs ${gb(need)}, ${gb(free)} free now`,
      }
    if (need <= total)
      return {
        tone: 'warn',
        label: `Needs ${gb(need)}`,
        full: `Needs ${gb(need)} - only ${gb(free)} free right now`,
      }
    return {
      tone: 'bad',
      label: "Won't fit",
      full: `Won't fit on this machine - needs ${gb(need)}, your GPU has ${gb(total)}`,
    }
  }

  // No shape for any of this model's builds. Rare and specific: the shape
  // generator reads GGUF headers, so an artifact that ships as safetensors has
  // no published geometry yet. Saying so beats inventing a number.
  return { tone: 'dim', label: '-', full: 'VRAM for this build has not been established yet' }
}

function configure(m: CatalogModel): void {
  void router.push({ name: 'server-new-config', params: { model: m.id } })
}
</script>

<template>
  <div class="np">
    <nav class="np__crumbs">
      <RouterLink :to="{ name: 'servers' }">Models</RouterLink>
      <span>/</span>
      <span>Start</span>
    </nav>
    <h1 class="np__title">Start a model</h1>

    <!-- vendor blocks: mark, brand, what the line can do -->
    <div class="np__vendors">
      <button
        v-for="g in groups"
        :key="g.vendor"
        type="button"
        class="np__vendor"
        :class="{ 'np__vendor--on': g.vendor === sel }"
        @click="pick(g.vendor)"
      >
        <VendorLogo :vendor="g.vendor" :size="30" class="np__vendor-logo" />
        <span class="np__vendor-text">
          <span class="np__vendor-name">{{ g.title }}</span>
          <span class="np__vendor-sub">
            {{ g.sub }}<template v-if="g.sub"> · </template>{{ g.models.length }}
            {{ g.models.length === 1 ? 'model' : 'models' }}
          </span>
        </span>
        <span class="np__vendor-caps">
          <Tooltip v-for="c in g.caps" :key="c.label" :label="c.label">
            <span class="np__cap"><Icon :name="c.icon" :size="14" /></span>
          </Tooltip>
        </span>
      </button>
    </div>

    <p v-if="!selGroup" class="np__pickhint">Pick a vendor to compare its models.</p>

    <!-- the comparison: four columns, one flexible. The pitch is one plain
         line (the about), the cost is ONE muted line under it - the full
         strengths list lives on the Models page, not here. -->
    <div v-else class="np__tablewrap">
      <table class="np__table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Why pick it</th>
            <th>On your GPU</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <template v-for="(r, i) in rows" :key="i">
            <tr v-if="r.kind === 'head'" class="np__sect">
              <td colspan="4">{{ r.label }}</td>
            </tr>
            <tr v-else class="np__row" @click="configure(r.m)">
              <td class="c-name">
                <span class="c-name__id">
                  {{ r.m.display }}
                  <Tooltip v-if="imageCap(r.m)" :label="imageCap(r.m)!.title">
                    <span class="c-name__cap"><Icon :name="imageCap(r.m)!.icon" :size="13" /></span>
                  </Tooltip>
                </span>
                <Tooltip :label="r.m.specs?.context_max ?? ''">
                  <span class="c-name__meta">
                    {{ r.m.specs?.params }}<template v-if="r.m.specs?.context">
                      · {{ r.m.specs.context
                      }}{{ r.m.capability.includes('tabular') ? '' : ' ctx' }}</template>
                  </span>
                </Tooltip>
                <span v-if="!r.m.installed" class="c-name__get">
                  {{ fmtBytes(r.m.total_size) }} download
                </span>
              </td>
              <td class="c-why">
                <span class="c-about">{{ r.m.specs?.about }}</span>
                <span v-if="r.m.specs?.tradeoffs?.length" class="c-trade">
                  - {{ r.m.specs.tradeoffs.join(' · ') }}
                </span>
              </td>
              <td class="c-fit">
                <Tooltip :label="fitOf(r.m).full">
                  <span class="c-fit__in" :class="`c-fit--${fitOf(r.m).tone}`">
                    <span class="c-fit__dot" /> {{ fitOf(r.m).label }}
                  </span>
                </Tooltip>
              </td>
              <td class="c-act" @click.stop>
                <button class="pk-btn pk-btn--sm pk-btn--primary" @click="configure(r.m)">
                  Configure
                </button>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>

    <p class="np__custom">
      Already have a GGUF of your own?
      <RouterLink :to="{ name: 'server-new-config', params: { model: 'custom' } }">
        Start your own file
      </RouterLink>
    </p>
  </div>
</template>

<style scoped>
/* One column, so every block on this page resolves to the same width - and
   when that width has to exceed the panel, they all exceed it together.
   The vendor cards have a real minimum (mark + brand + capability icons) and
   squeezing them to fit reads worse than a page that runs a little wide, so
   the page runs wide; what was wrong was only the comparison table staying
   behind at 960 while the vendor row grew, which put two different right
   edges on one page.
   `minmax(min-content, 1fr)` is a no-op while everything fits. `.np__tablewrap`
   is a scroll container, so its own min-content is 0 - it follows the width
   the vendor row sets and never drives it. */
.np {
  display: grid;
  grid-template-columns: minmax(min-content, 1fr);
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.np__crumbs {
  display: flex;
  gap: 8px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  margin-bottom: 8px;
}
.np__crumbs a {
  color: var(--pk-accent);
  text-decoration: none;
}
.np__crumbs a:hover {
  text-decoration: underline;
}
.np__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.np__lead {
  margin: 0 0 20px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  max-width: 640px;
  line-height: 1.5;
}
.np__lead a {
  color: var(--pk-accent);
}

/* vendor blocks - slim, mark-first */
.np__vendors {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 18px;
}
@media (max-width: 760px) {
  .np__vendors {
    grid-template-columns: 1fr;
  }
}
.np__vendor {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  font: inherit;
  text-align: left;
  cursor: pointer;
  color: var(--pk-text-primary);
}
.np__vendor:hover {
  border-color: var(--pk-accent);
}
.np__vendor--on {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
.np__vendor-logo {
  flex: none;
}
.np__vendor-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.np__vendor-name {
  font-size: var(--pk-font-size-base);
  font-weight: 700;
  letter-spacing: -0.01em;
}
.np__vendor-sub {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.np__vendor-caps {
  display: flex;
  gap: 8px;
  margin-left: auto;
  color: var(--pk-text-muted);
}
.np__pickhint {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  padding: 24px 0;
  text-align: center;
}

/* the comparison table */
/* the table is a SURFACE card like every other form surface - bare rows on
   the content background read as unfinished (house rule). Hover must NOT be
   bg-base: in light theme base == the page background, so a base-hovered row
   reads as a hole punched in the white card. Mixing a little text-primary
   into the surface steps away from BOTH the card and the page in each theme
   (in dark it lands where --pk-bg-hover does today). */
.np__tablewrap {
  overflow-x: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.np__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.np__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 9px 14px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
  background: var(--pk-bg-surface);
}
.np__table td {
  padding: 10px 14px;
  border-top: 1px solid var(--pk-border-default);
  vertical-align: top;
}
.np__sect td {
  padding: 6px 14px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
  background: var(--pk-bg-surface);
}
.np__row {
  cursor: pointer;
}
.np__row:hover td {
  background: color-mix(in srgb, var(--pk-bg-surface) 92%, var(--pk-text-primary));
}
.c-name {
  white-space: nowrap;
}
.c-name__id {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.c-name__cap {
  display: inline-flex;
  color: var(--pk-text-muted);
}
.c-name__meta {
  display: block;
  margin-top: 1px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.c-name__get {
  display: block;
  margin-top: 2px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.c-about {
  display: block;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.45;
}
.c-trade {
  display: block;
  margin-top: 1px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.45;
}
.c-fit {
  white-space: nowrap;
}
.c-fit__in {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}
.c-fit__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: none;
  background: var(--pk-text-muted);
}
.c-fit--good .c-fit__dot {
  background: var(--pk-status-success, #4a9);
}
.c-fit--ok .c-fit__dot,
.c-fit--warn .c-fit__dot {
  background: var(--pk-status-warning);
}
.c-fit--warn {
  color: var(--pk-status-warning);
}
.c-fit--bad {
  color: var(--pk-text-danger);
}
.c-fit--bad .c-fit__dot {
  background: var(--pk-text-danger);
}
.c-act {
  text-align: right;
  white-space: nowrap;
}
.np__custom {
  margin: 16px 0 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.np__custom a {
  color: var(--pk-accent);
}
</style>
