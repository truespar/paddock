<script setup lang="ts">
// The usage dashboard, organized the way real
// metric dashboards are (Grafana / the vLLM observability boards):
//
//   - Every PANEL is its own CHART - own card, own legend, own axes - in a
//     full-width stack the page scrolls past. Crosshairs sync across panels
//     via echarts' connect().
//   - The TIME RANGE is TOOLBAR STATE, not a chart control: the range
//     picker plus shift (‹ ›) and zoom-out buttons. There is no slider.
//   - A ZOOM GESTURE on any PANEL (wheel / pinch / drag when zoomed)
//     adopts into that shared range and re-queries - so every panel always
//     shows exactly the same window, at a grain the server picks for it
//     (bounded slot count, so any window is a cheap fetch).
//
// While the right edge sits at now, the 15 s poll slides the window (live
// follow); shifting or zooming into history pauses it and the stamp says so.
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { connect, use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  AxisPointerComponent,
  DataZoomInsideComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import { useTheme } from '@/composables/useTheme'
import { servedId, useFleetStore } from '@/stores/fleet'
import { useRegistryStore } from '@/stores/registry'
import {
  usageApi,
  type UsageGap,
  type UsageGeneration,
  type UsageSlot,
  type WebSlot,
} from '@/lib/api'
import { fmtStamp, fmtTokens } from '@/lib/format'
import { SEARCH_PROVIDERS, searchLabel } from '@/lib/websearch'
import { modelLabel, modelVendor } from '@/lib/model-name'
import Select from '@/components/ui/Select.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'

use([
  CanvasRenderer,
  LineChart,
  AxisPointerComponent,
  DataZoomInsideComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
])

const CHART_GROUP = 'usage-dash'

const props = defineProps<{ port: number | null }>()
const emit = defineEmits<{ 'update:port': [port: number | null] }>()

const fleet = useFleetStore()
const { theme } = useTheme()

// ── the shared time range (dashboard state, Grafana-style) ──────────────────
const HOUR = 3_600_000
const DAY = 24 * HOUR
const spanOptions = [
  { value: HOUR, label: 'Last hour' },
  { value: 4 * HOUR, label: 'Last 4 hours' },
  { value: 8 * HOUR, label: 'Last 8 hours' },
  { value: DAY, label: 'Last 24 hours' },
  { value: 3 * DAY, label: 'Last 3 days' },
  { value: 7 * DAY, label: 'Last 7 days' },
  { value: 30 * DAY, label: 'Last 30 days' },
  { value: 90 * DAY, label: 'Last 90 days' },
  { value: 365 * DAY, label: 'Last year' },
  { value: 0, label: 'All history' },
]
const spanSel = ref<number>(4 * HOUR)
const viewFrom = ref(Date.now() - 4 * HOUR)
const viewTo = ref(Date.now())
const extentFrom = ref<number | null>(null)

/** Range picker: snap to this width, right edge at now (0 = all history). */
function zoomTo(ms: number): void {
  const now = nowMs.value || Date.now()
  const lo = extentFrom.value ?? now - DAY
  viewTo.value = now
  viewFrom.value = ms === 0 ? Math.min(lo, now - HOUR) : Math.max(lo, now - ms)
  scheduleFetch()
}
watch(spanSel, (ms) => zoomTo(ms))

/** ‹ › : shift the window by half its width; zoom-out doubles it (centered). */
function shift(dir: -1 | 1): void {
  const w = viewTo.value - viewFrom.value
  const now = nowMs.value || Date.now()
  const lo = extentFrom.value ?? 0
  let to = Math.min(now, viewTo.value + (dir * w) / 2)
  let from = to - w
  if (from < lo) {
    from = lo
    to = Math.min(now, from + w)
  }
  viewFrom.value = from
  viewTo.value = to
  scheduleFetch()
}
function zoomOut(): void {
  const w = viewTo.value - viewFrom.value
  const now = nowMs.value || Date.now()
  const lo = extentFrom.value ?? 0
  const mid = (viewFrom.value + viewTo.value) / 2
  viewTo.value = Math.min(now, mid + w)
  viewFrom.value = Math.max(lo, viewTo.value - 2 * w)
  scheduleFetch()
}

// ── data: always the whole visible window (no client bucket cache) ──────────
const buckets = ref<UsageSlot[]>([])
const web = ref<WebSlot[]>([])
const gaps = ref<UsageGap[]>([])
const generations = ref<UsageGeneration[]>([])
const grainMs = ref(300_000)
const nowMs = ref(Date.now())
const fetched = ref<{ from: number; to: number }>({ from: viewFrom.value, to: viewTo.value })
const loading = ref(false)
const loaded = ref(false)
const stale = ref(false)
const updatedAt = ref<number | null>(null)

/** Live follow: the right edge sits within two buckets of now. */
function following(): boolean {
  return viewTo.value >= nowMs.value - 2 * grainMs.value
}

async function load(): Promise<void> {
  if (loading.value) return
  loading.value = true
  const from = Math.floor(viewFrom.value)
  const to = Math.ceil(viewTo.value)
  try {
    const r = await usageApi.history({ from, to, port: props.port })
    buckets.value = r.buckets
    web.value = r.web ?? []
    gaps.value = r.gaps
    generations.value = r.generations
    grainMs.value = r.grain_ms
    nowMs.value = r.now_ms
    extentFrom.value = r.extent_from_ms
    fetched.value = { from, to }
    loaded.value = true
    stale.value = false
    updatedAt.value = Date.now()
  } catch {
    // keep the last answer on a transient failure; the stamp goes amber
    stale.value = true
  } finally {
    loading.value = false
  }
}

let fetchTimer: number | undefined
function scheduleFetch(): void {
  if (fetchTimer !== undefined) clearTimeout(fetchTimer)
  fetchTimer = window.setTimeout(() => void load(), 250)
}

// ── zoom gestures: adopt into the shared range, then reset the gesture ──────
// Each panel has an `inside` dataZoom purely to CAPTURE wheel/pinch/drag;
// the resulting window is promoted to viewFrom/viewTo (axis min/max on every
// panel) and the gesture zoom is reset to 0-100 - the Grafana contract of
// "zoom sets the dashboard range", with no per-chart zoom state to drift.
const panelRefs = new Map<string, InstanceType<typeof VChart>>()
function bindPanel(key: string) {
  return (el: unknown) => {
    if (el) panelRefs.set(key, el as InstanceType<typeof VChart>)
    else panelRefs.delete(key)
  }
}
let adopting = false
interface ZoomWindow {
  startValue?: number
  endValue?: number
}
function onPanelZoom(key: string, ev?: unknown): void {
  if (adopting) return
  // drag-select zoom reports its window in the event batch; wheel zoom is
  // read back from the chart's resolved dataZoom state
  const e = ev as (ZoomWindow & { batch?: ZoomWindow[] }) | undefined
  let z: ZoomWindow | undefined = e?.batch?.find((b) => b.startValue != null) ?? e
  if (z?.startValue == null || z?.endValue == null) {
    const inst = panelRefs.get(key)
    z = (
      inst?.getOption() as { dataZoom?: ZoomWindow[] } | undefined
    )?.dataZoom?.[0]
  }
  if (!z || z.startValue == null || z.endValue == null) return
  const eps = grainMs.value / 4
  if (Math.abs(z.startValue - viewFrom.value) < eps && Math.abs(z.endValue - viewTo.value) < eps)
    return
  viewFrom.value = z.startValue
  viewTo.value = z.endValue
  adopting = true
  void nextTick(() => {
    for (const p of panelRefs.values()) p.dispatchAction({ type: 'dataZoom', start: 0, end: 100 })
    adopting = false
    armDragZoom()
  })
  scheduleFetch()
}
/** Keep drag-select-to-zoom PERMANENTLY armed on every panel - the Grafana
 *  gesture. ECharts hides it behind a toolbox icon by default; taking the
 *  global cursor arms it without any icon (the toolbox itself stays hidden). */
function armDragZoom(): void {
  for (const p of panelRefs.values())
    p.dispatchAction({ type: 'takeGlobalCursor', key: 'dataZoomSelect', dataZoomSelectActive: true })
}

let timer: number | undefined
function onVisible(): void {
  if (document.visibilityState === 'visible') void poll()
}
async function poll(): Promise<void> {
  if (following()) {
    const w = viewTo.value - viewFrom.value
    viewTo.value = Date.now()
    viewFrom.value = viewTo.value - w
  }
  // browsed windows refetch too: gaps can be INSERTED into old time when the
  // manager reattaches, so history is not immutable
  await load()
}

const wrapEl = ref<HTMLElement | null>(null)
const wrapW = ref(0)
let ro: ResizeObserver | undefined
onMounted(() => {
  connect(CHART_GROUP)
  timer = window.setInterval(() => {
    if (document.visibilityState === 'visible') void poll()
  }, 15_000)
  document.addEventListener('visibilitychange', onVisible)
  window.addEventListener('focus', onVisible)
  ro = new ResizeObserver((entries) => {
    const w = entries[entries.length - 1]?.contentRect.width
    if (w) wrapW.value = Math.round(w)
  })
  if (wrapEl.value) ro.observe(wrapEl.value)
  // catalog names + vendor marks for ports whose model is not running now
  const reg = useRegistryStore()
  if (!reg.models.length) void reg.refresh()
})
watch(wrapEl, (el) => {
  ro?.disconnect()
  if (el) ro?.observe(el)
})
watch(
  () => props.port,
  () => void load(),
  { immediate: true },
)
onUnmounted(() => {
  if (timer !== undefined) clearInterval(timer)
  if (fetchTimer !== undefined) clearTimeout(fetchTimer)
  ro?.disconnect()
  document.removeEventListener('visibilitychange', onVisible)
  window.removeEventListener('focus', onVisible)
})

const stampTime = computed(() =>
  updatedAt.value == null
    ? ''
    : new Date(updatedAt.value).toLocaleTimeString(undefined, { hour12: false }),
)

// ── aggregation: ports sum unless filtered; histograms sum elementwise ──────
interface Slot {
  t: number
  requests: number
  e4: number
  e5: number
  disc: number
  input: number
  output: number
  cached: number
  drafted: number
  accepted: number
  durSum: number
  kv: number
  e2e: number[]
  ttft: number[]
}
function zeroSlot(t: number): Slot {
  return {
    t,
    requests: 0,
    e4: 0,
    e5: 0,
    disc: 0,
    input: 0,
    output: 0,
    cached: 0,
    drafted: 0,
    accepted: 0,
    durSum: 0,
    kv: 0,
    e2e: Array(14).fill(0) as number[],
    ttft: Array(14).fill(0) as number[],
  }
}
const slots = computed<Slot[]>(() => {
  const by = new Map<number, Slot>()
  for (const b of buckets.value) {
    let s = by.get(b.t)
    if (!s) {
      s = zeroSlot(b.t)
      by.set(b.t, s)
    }
    s.requests += b.requests
    s.e4 += b.errors_4xx
    s.e5 += b.errors_5xx
    s.disc += b.disconnects
    s.input += b.input_tokens
    s.output += b.output_tokens
    s.cached += b.cached_tokens
    s.drafted += b.spec_drafted
    s.accepted += b.spec_accepted
    s.durSum += b.duration_ms_sum
    s.kv = Math.max(s.kv, b.kv_pages_max)
    for (let i = 0; i < 14; i++) {
      s.e2e[i] += b.e2e_h[i] ?? 0
      s.ttft[i] += b.ttft_h[i] ?? 0
    }
  }
  return [...by.values()].sort((a, b) => a.t - b.t)
})

const totals = computed(() => {
  const t = { requests: 0, errors: 0, disc: 0, input: 0, output: 0, cached: 0, drafted: 0, accepted: 0 }
  for (const s of slots.value) {
    t.requests += s.requests
    t.errors += s.e4 + s.e5
    t.disc += s.disc
    t.input += s.input
    t.output += s.output
    t.cached += s.cached
    t.drafted += s.drafted
    t.accepted += s.accepted
  }
  return t
})
const cachedPct = computed(() =>
  totals.value.input > 0 ? Math.round((100 * totals.value.cached) / totals.value.input) : 0,
)

// ── web-search spend: the other thing this box spends ───────────────────────
// GPU time is free once the hardware is bought; a search bills the user's own
// provider key. The three counters are kept apart all the way to the screen:
// requests is the only one every provider reports (Brave and Perplexity price
// nothing at all), credits mean nothing outside one provider's pricing page
// and are not one per search - a Firecrawl search that scraped an arxiv PDF
// cost 38 - and dollars only exist where a provider quotes them. Summing them
// would print a number in a currency nobody uses.
interface WebCell {
  requests: number
  credits: number
  micro: number
}
function zeroCell(): WebCell {
  return { requests: 0, credits: 0, micro: 0 }
}
/** slot time -> provider -> spend, ports summed away (port is the toolbar's
 *  filter; the provider is what this panel is about). */
const webByT = computed(() => {
  const by = new Map<number, Map<string, WebCell>>()
  for (const w of web.value) {
    let slot = by.get(w.t)
    if (!slot) {
      slot = new Map()
      by.set(w.t, slot)
    }
    const c = slot.get(w.provider) ?? zeroCell()
    c.requests += w.requests
    c.credits += w.credits
    c.micro += w.microdollars
    slot.set(w.provider, c)
  }
  return by
})
/** Providers that actually searched in this window, in the catalog's own
 *  order so a provider keeps its colour and legend position as the window
 *  moves. An id the catalog doesn't know still appears - a key configured for
 *  a provider this build has never heard of is exactly what a user must see. */
const webProviders = computed(() => {
  const seen = new Set<string>()
  for (const w of web.value) if (w.requests > 0 || w.credits > 0 || w.microdollars > 0) seen.add(w.provider)
  const known = SEARCH_PROVIDERS.map((p) => p.id).filter((id) => seen.has(id))
  const rest = [...seen].filter((id) => !known.includes(id)).sort()
  return [...known, ...rest]
})
const webTotals = computed(() => {
  const per = new Map<string, WebCell>()
  const all = zeroCell()
  for (const w of web.value) {
    const c = per.get(w.provider) ?? zeroCell()
    c.requests += w.requests
    c.credits += w.credits
    c.micro += w.microdollars
    per.set(w.provider, c)
    all.requests += w.requests
    all.credits += w.credits
    all.micro += w.microdollars
  }
  return { per, all }
})
/** Millionths of a dollar to something readable. Precision GROWS as the
 *  amount shrinks: a single Exa search costs $0.0070, and rounding that to
 *  "$0.01" would overstate it by 43%. */
function fmtMoney(micro: number): string {
  const d = micro / 1e6
  if (d >= 1) return `$${d.toFixed(2)}`
  if (d >= 0.01) return `$${d.toFixed(3)}`
  return `$${d.toFixed(4)}`
}
/** What one provider charged, in its currency. Empty when it charged in none
 *  - which is a fact about the provider, not a missing measurement. */
function webCost(c: WebCell): string {
  const parts: string[] = []
  if (c.micro > 0) parts.push(fmtMoney(c.micro))
  if (c.credits > 0) parts.push(`${c.credits.toLocaleString()} credits`)
  return parts.join(' + ')
}
/** The strip's one-line summary. Providers that price nothing contribute
 *  their searches and nothing else, so the sentence stays true either way. */
const webSummary = computed(() => {
  const { all } = webTotals.value
  const cost = webCost(all)
  const unpriced = webProviders.value.filter((p) => {
    const c = webTotals.value.per.get(p)
    return c && c.micro === 0 && c.credits === 0
  })
  if (cost && unpriced.length)
    return `${cost} · ${unpriced.map(searchLabel).join(', ')} report no cost`
  if (cost) return cost
  return 'no provider reported a cost'
})

// ── percentiles on the runner's semconv ladder (metrics.rs, seconds) ────────
const BOUNDS_S = [
  0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
]
/** histogram_quantile, Prometheus rules: linear inside the bucket, +Inf
 *  clamps to the highest finite bound. Null when nothing was observed (a
 *  earlier bucket has counts but no histogram - that must be a hole in the
 *  line, not an 82 s spike). */
function quantileMs(counts: number[], total: number, q: number): number | null {
  const observed = counts.reduce((a, c) => a + c, 0)
  if (total <= 0 || observed <= 0) return null
  const target = q * total
  let cum = 0
  for (let i = 0; i < counts.length; i++) {
    const prev = cum
    cum += counts[i]
    if (cum >= target) {
      const lo = i === 0 ? 0 : BOUNDS_S[i - 1]
      const frac = counts[i] > 0 ? (target - prev) / counts[i] : 0
      return (lo + (BOUNDS_S[i] - lo) * frac) * 1000
    }
  }
  return BOUNDS_S[BOUNDS_S.length - 1] * 1000
}

// ── model naming: friendly name + vendor mark, same idiom as everywhere ─────
/** The one model a generation served, whichever of the four serving roles it
 *  was in. Every role has to be listed: a runner in any of them carries only
 *  its own key, so a reader that stops at three renders the fourth as a dash
 *  with no vendor mark - which is what the aligner lane used to look like
 *  here. */
function genRaw(g: UsageGeneration): string {
  return g.model ?? g.embedder ?? g.asr ?? g.aligner ?? ''
}
function genName(g: UsageGeneration): string {
  const raw = genRaw(g)
  return raw ? modelLabel(raw) || raw : '-'
}
function genVendor(g: UsageGeneration): string | undefined {
  const raw = genRaw(g)
  return raw ? modelVendor(raw) : undefined
}

// ── port options: the fleet's live rows plus every port the window saw ──────
const portOptions = computed(() => {
  const seen = new Map<number, { name: string; vendor?: string }>()
  for (const g of generations.value) {
    const raw = genRaw(g)
    if (!seen.has(g.port) || raw)
      seen.set(g.port, { name: raw ? modelLabel(raw) || raw : '', vendor: genVendor(g) })
  }
  for (const r of fleet.rows) {
    // Every role here too. `display` usually covers it, but a model the
    // catalog does not know has none - and then a whisper or aligner row fell
    // through to its port NUMBER as its own label.
    const raw = servedId(r) ?? ''
    seen.set(r.port, {
      name: r.display ?? (raw ? modelLabel(raw) || raw : ''),
      vendor: r.vendor ?? (raw ? modelVendor(raw) : undefined),
    })
  }
  const rows = [...seen.entries()].sort((a, b) => a[0] - b[0])
  return [
    { value: 'all', label: 'All models' },
    ...rows.map(([port, o]) => ({
      value: port,
      label: o.name || String(port),
      hint: String(port),
      vendor: o.vendor,
    })),
  ]
})
const portSel = computed({
  get: () => props.port ?? 'all',
  set: (v: string | number) => emit('update:port', Number(v) || null),
})

// ── gap sentences: same hole, different sentence per cause  ──────
function gapRange(g: UsageGap): string {
  return `${fmtStamp(new Date(g.from_ts_ms))} - ${fmtStamp(new Date(g.to_ts_ms))}`
}
function gapSentence(g: UsageGap): string {
  if (g.cause === 'manager-down') {
    const lost =
      g.lost_requests != null
        ? `${g.lost_requests.toLocaleString()} requests and ${fmtTokens((g.lost_input_tokens ?? 0) + (g.lost_output_tokens ?? 0))} tokens went unrecorded - the totals are exact, their timing unknown.`
        : 'whatever ran then went unrecorded.'
    return `Nobody was watching ${gapRange(g)}: ${lost}`
  }
  if (g.cause === 'runner-restart-unobserved')
    return `The model on port ${g.port} restarted while nobody was watching (${gapRange(g)}) - the tail of the previous run is lost for good.`
  if (g.cause === 'ring-overrun') {
    const n = g.from_seq != null && g.to_seq != null ? g.to_seq - g.from_seq : null
    return `The collector fell behind on port ${g.port}: ${n != null ? `${n.toLocaleString()} request records were` : 'records were'} overwritten before they were read (${gapRange(g)}).`
  }
  if (g.cause === 'journal-off') return `Recording was off ${gapRange(g)} - by configuration, not by accident.`
  return `${g.cause} (${gapRange(g)})`
}
const GAP_LABEL: Record<string, string> = {
  'manager-down': 'unwatched',
  'runner-restart-unobserved': 'restart, tail lost',
  'ring-overrun': 'overrun',
  'journal-off': 'recording off',
}

// ── lifecycle vocabulary: what a band says in the table ─────────────────────
function startText(g: UsageGeneration): string {
  if (g.start_cause === 'manual') return 'Started from the Manager'
  if (g.start_cause === 'boot-election') return 'Started with the machine'
  if (g.start_cause === 'batch-restore') return 'Restored after batch work'
  return 'Start not observed'
}
// An open band is "Running" only while its heartbeat is recent. The collector
// folds a scrape about every 30 s, so a few missed rounds is a runner nothing
// has heard from - which is worth saying, because the alternative is drawing a
// wedged runner exactly like a healthy one.
const HEARTBEAT_STALE_MS = 5 * 60_000
function endText(g: UsageGeneration): string {
  if (g.ended_ms == null) {
    const seen = g.last_seen_ms
    if (seen != null && Date.now() - seen > HEARTBEAT_STALE_MS) {
      return `Running, last seen ${new Date(seen).toLocaleTimeString()}`
    }
    return 'Running'
  }
  if (g.end_cause === 'stopped') return 'Stopped'
  if (g.end_cause === 'takeover') return 'Replaced by a new start'
  if (g.end_cause === 'crashed') return 'Crashed'
  if (g.end_cause === 'machine-restarted') return 'Ended when the machine restarted'
  return 'Ended unobserved'
}
const bands = computed(() =>
  [...generations.value].sort((a, b) => b.started_ms - a.started_ms || a.port - b.port),
)

// ── chart building blocks ───────────────────────────────────────────────────
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
function withAlpha(c: string, a: number): string {
  const s = c.trim()
  if (s.startsWith('#')) {
    let h = s.slice(1)
    if (h.length === 3)
      h = h
        .split('')
        .map((x) => x + x)
        .join('')
    const r = parseInt(h.slice(0, 2), 16)
    const g = parseInt(h.slice(2, 4), 16)
    const b = parseInt(h.slice(4, 6), 16)
    return `rgba(${r},${g},${b},${a})`
  }
  if (s.startsWith('rgb(')) return s.replace('rgb(', 'rgba(').replace(')', `,${a})`)
  return s
}
/** Diagonal-stripe fill for the gap bands - the conventional "no data here"
 *  texture, built once per theme change. */
function hatch(color: string): { image: HTMLCanvasElement; repeat: string } {
  const c = document.createElement('canvas')
  c.width = 8
  c.height = 8
  const x = c.getContext('2d')!
  x.strokeStyle = withAlpha(color, 0.5)
  x.lineWidth = 1.4
  x.beginPath()
  x.moveTo(-2, 6)
  x.lineTo(6, -2)
  x.moveTo(2, 10)
  x.lineTo(10, 2)
  x.stroke()
  return { image: c, repeat: 'repeat' }
}
const msFmt = (v: number): string =>
  v >= 1000 ? `${(v / 1000).toFixed(v >= 10_000 ? 0 : 1)}s` : `${Math.round(v)}ms`

interface Panel {
  key: string
  height: number
  option: Record<string, unknown>
}

const panels = computed<Panel[]>(() => {
  void theme.value // recompute colours on theme toggle
  const accent = cssVar('--pk-accent') || '#6366f1'
  const gridLine = cssVar('--pk-border-subtle') || 'rgba(128,128,128,0.16)'
  const muted = cssVar('--pk-text-muted') || '#888'
  const elevated = cssVar('--pk-bg-elevated') || '#1b1b1b'
  const strong = cssVar('--pk-border-strong') || '#333'
  const primary = cssVar('--pk-text-primary') || '#fff'
  const warn = cssVar('--pk-status-warning') || '#d4952b'
  const danger = cssVar('--pk-status-error') || '#e05652'
  const success = cssVar('--pk-status-success') || '#22c55e'

  const mono = cssVar('--pk-font-mono') || 'ui-monospace, Menlo, Consolas, monospace'

  const grain = grainMs.value
  const grainS = grain / 1000
  const xMin = viewFrom.value
  const xMax = viewTo.value

  // Zero-fill the FETCHED window at its grain: sparse rows are honest in the
  // store, but a watched-but-idle stretch must read as flat zero, and
  // uniform spacing keeps the step geometry sane.
  const sparse = slots.value
  const offset = sparse.length ? sparse[0].t % grain : 0
  const byT = new Map(sparse.map((s) => [s.t, s]))
  const data: Slot[] = []
  const f = fetched.value
  const n = Math.floor((f.to - f.from) / grain)
  if (n > 0 && n <= 8192) {
    for (let t = Math.ceil((f.from - offset) / grain) * grain + offset; t <= f.to; t += grain) {
      data.push(byT.get(t) ?? zeroSlot(t))
    }
  } else {
    data.push(...sparse)
  }
  const mid = (s: Slot) => s.t + grain / 2
  const rate = (v: (s: Slot) => number) => data.map((s) => [mid(s), v(s) / grainS] as [number, number])
  const line = (v: (s: Slot) => number | null) =>
    data.map((s) => [mid(s), v(s)] as [number, number | null])

  const markArea = {
    silent: true,
    itemStyle: { color: hatch(muted) },
    label: { show: true, position: 'insideTop' as const, color: muted, fontSize: 9 },
    data: gaps.value.map((g) => [
      { xAxis: Math.max(g.from_ts_ms, xMin), name: GAP_LABEL[g.cause] ?? g.cause },
      { xAxis: Math.min(g.to_ts_ms, xMax) },
    ]),
  }

  const xAxis = {
    type: 'time' as const,
    min: xMin,
    max: xMax,
    splitNumber: Math.max(6, Math.floor((wrapW.value || 1100) / 110)),
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: {
      color: muted,
      fontSize: 10,
      fontFamily: mono,
      hideOverlap: true,
      // ECharts' own leveled time templates: day boundaries carry the DATE
      // (bold), hour ticks the 24-hour clock ({HH} by definition - never
      // the browser locale, which mis-guessed 12 h on a 24 h machine).
      formatter: {
        year: '{d|{yyyy}}',
        month: '{d|{d} {MMM}}',
        day: '{d|{d} {MMM}}',
        hour: '{HH}:{mm}',
        minute: '{HH}:{mm}',
        second: '{HH}:{mm}:{ss}',
        millisecond: '{HH}:{mm}',
      },
      rich: { d: { color: muted, fontWeight: 700 as const, fontSize: 10, fontFamily: mono } },
    },
  }
  const tooltip = (unit: 'rate' | 'ms' | 'tokens' | 'pages' | 'pct') => ({
    trigger: 'axis' as const,
    backgroundColor: elevated,
    borderColor: strong,
    borderWidth: 1,
    padding: [6, 10],
    textStyle: { color: primary, fontSize: 12 },
    axisPointer: { type: 'line' as const, lineStyle: { color: muted, width: 1, type: 'dashed' as const } },
    formatter: (
      params: { seriesName: string; seriesType: string; value: [number, number | null]; marker: string }[],
    ) => {
      const rows = params.filter((p) => p.seriesType !== 'custom' && p.value?.[1] != null)
      if (!rows.length) return ''
      const nz = rows.filter((p) => (p.value[1] as number) > 0)
      const start = rows[0].value[0] - grain / 2
      const end = new Date(start + grain).toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      })
      const fmtV = (v: number): string => {
        if (unit === 'ms') return msFmt(v)
        if (unit === 'tokens') return `${fmtTokens(Math.round(v))}/s`
        if (unit === 'pages') return v.toLocaleString()
        if (unit === 'pct') return `${v.toFixed(1)} %`
        return `${v.toFixed(v < 10 ? 2 : 1)}/s`
      }
      const body = nz
        .map(
          (p) =>
            `${p.marker} ${p.seriesName}: <b style="font-family:${mono}">${fmtV(p.value[1] as number)}</b>`,
        )
        .join('<br/>')
      return `<span style="color:${muted};font-size:11px;font-family:${mono}">${fmtStamp(new Date(start))} - ${end}</span><br/>${body}`
    },
  })
  // The chart owns the whole card: its top band carries the panel title on
  // the left and the series legend in the right corner - one row, not two
  // stacked ones (an HTML title above an in-chart legend was eating a full
  // band of height per panel).
  const base = (
    title: string,
    unit: 'rate' | 'ms' | 'tokens' | 'pages' | 'pct',
    yFmt?: (v: number) => string,
  ) => ({
    animation: false,
    title: {
      text: title.toUpperCase(),
      left: 12,
      top: 8,
      textStyle: { color: muted, fontSize: 10, fontWeight: 700 as const },
    },
    legend: {
      top: 8,
      right: 12,
      itemWidth: 11,
      itemHeight: 7,
      icon: 'rect',
      textStyle: { color: muted, fontSize: 10 },
    },
    tooltip: tooltip(unit),
    // gesture capture only - the resulting window is promoted to the shared
    // range and this zoom is reset (see onPanelZoom). The toolbox exists
    // solely to instantiate the drag-select-zoom feature armDragZoom()
    // keeps active - show:false would skip creating the feature entirely,
    // so it renders at zero size instead.
    //
    // Plain wheel is not bound (Grafana's rule): a bound wheel hijacks page
    // scroll and hands it back the moment zoom hits its limit, which lurches
    // the page. Ctrl+wheel zooms deliberately; drag is the select-zoom.
    dataZoom: [
      {
        type: 'inside' as const,
        zoomOnMouseWheel: 'ctrl' as const,
        moveOnMouseWheel: false,
        moveOnMouseMove: false,
      },
    ],
    toolbox: {
      show: true,
      itemSize: 0,
      iconStyle: { opacity: 0 },
      feature: {
        dataZoom: {
          yAxisIndex: 'none' as const,
          brushStyle: { color: withAlpha(accent, 0.12), borderColor: accent },
        },
      },
    },
    grid: { left: 62, right: 14, top: 36, bottom: 24 },
    xAxis,
    yAxis: {
      type: 'value' as const,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 10, fontFamily: mono, formatter: yFmt },
      splitLine: { lineStyle: { color: gridLine, type: 'dashed' as const } },
    },
  })
  const area = (name: string, color: string, v: (s: Slot) => number, extra?: Record<string, unknown>) => ({
    name,
    type: 'line' as const,
    stack: 'a',
    symbol: 'none',
    step: 'middle' as const,
    lineStyle: { width: 0 },
    itemStyle: { color },
    areaStyle: { color, opacity: 0.9 },
    emphasis: { disabled: true },
    data: rate(v),
    ...extra,
  })
  // A lone value between empty buckets draws no line segment, so it gets a
  // marker - and only it (Grafana's own treatment of disconnected points).
  // Continuous stretches stay clean lines.
  const pline = (name: string, color: string, width: number, v: (s: Slot) => number | null) => {
    const vals = line(v)
    const pts = vals.map((val, i) => {
      const isolated =
        val[1] != null &&
        (i === 0 || vals[i - 1][1] == null) &&
        (i === vals.length - 1 || vals[i + 1][1] == null)
      return isolated ? { value: val, symbol: 'circle', symbolSize: 5 } : val
    })
    return {
      name,
      type: 'line' as const,
      symbol: 'none',
      connectNulls: false,
      lineStyle: { width, color },
      itemStyle: { color },
      data: pts,
    }
  }
  const succ = (s: Slot) => Math.max(0, s.requests - s.e4 - s.e5)

  const out: Panel[] = [
    {
      key: 'req',
      height: 280,
      option: {
        ...base('Requests /s', 'rate'),
        series: [
          area('Completed', accent, (s) => Math.max(0, s.requests - s.e4 - s.e5 - s.disc), { markArea }),
          area('Client errors', warn, (s) => s.e4),
          area('Failures', danger, (s) => s.e5),
          area('Disconnects', withAlpha(muted, 0.7), (s) => s.disc),
        ],
      },
    },
    {
      key: 'e2e',
      height: 280,
      option: {
        ...base('Request duration', 'ms', msFmt),
        series: [
          Object.assign(pline('p95', warn, 1.2, (s) => quantileMs(s.e2e, s.requests, 0.95)), { markArea }),
          pline('p50', accent, 1.6, (s) => quantileMs(s.e2e, s.requests, 0.5)),
          pline('avg', withAlpha(muted, 0.8), 1, (s) =>
            s.requests > 0 && s.durSum > 0 ? s.durSum / s.requests : null,
          ),
        ],
      },
    },
    {
      key: 'ttft',
      height: 280,
      option: {
        ...base('Time to first token', 'ms', msFmt),
        series: [
          Object.assign(pline('p95', warn, 1.2, (s) => quantileMs(s.ttft, succ(s), 0.95)), { markArea }),
          pline('p50', accent, 1.6, (s) => quantileMs(s.ttft, succ(s), 0.5)),
        ],
      },
    },
    {
      key: 'tok',
      height: 280,
      option: {
        ...base('Token throughput', 'tokens', (v: number) => fmtTokens(v)),
        series: [
          // cached input leads the stack: prefix reuse is the agentic story
          area('Cached input', success, (s) => Math.min(s.cached, s.input), { markArea }),
          area('Fresh input', withAlpha(accent, 0.45), (s) => Math.max(0, s.input - s.cached)),
          area('Output', accent, (s) => s.output),
        ],
      },
    },
    {
      key: 'cache',
      height: 280,
      option: (() => {
        const b = base('Prefix cache hit rate', 'pct', (v: number) => `${v}%`)
        return {
          ...b,
          yAxis: { ...b.yAxis, min: 0, max: 100 },
          series: [
            Object.assign(
              pline('hit rate', success, 1.6, (s) =>
                s.input > 0 ? (100 * Math.min(s.cached, s.input)) / s.input : null,
              ),
              { markArea },
            ),
          ],
        }
      })(),
    },
    {
      key: 'kv',
      height: 280,
      option: {
        ...base('KV cache pages (high water)', 'pages', (v: number) => v.toLocaleString()),
        series: [
          Object.assign(pline('KV pages', accent, 1.4, (s) => (s.kv > 0 ? s.kv : null)), { markArea }),
        ],
      },
    },
  ]

  // Web search, only on boxes that do it. An always-present empty panel would
  // be a permanent flat zero for every user who never configured a provider.
  const providers = webProviders.value
  if (providers.length) {
    // Fixed per-provider colours, keyed on the catalog's own order so a
    // provider does not change colour when another one stops searching.
    const palette = [accent, success, warn, danger, withAlpha(primary, 0.55)]
    const colourOf = (id: string): string => {
      const i = SEARCH_PROVIDERS.findIndex((p) => p.id === id)
      return palette[(i >= 0 ? i : providers.indexOf(id) + SEARCH_PROVIDERS.length) % palette.length]
    }
    const cells = webByT.value
    const count = (id: string) => (s: Slot) => cells.get(s.t)?.get(id)?.requests ?? 0
    out.push({
      key: 'web',
      height: 280,
      option: {
        ...base('Web searches', 'rate', (v: number) => v.toLocaleString()),
        // COUNT per slot, not a rate: a handful of searches an hour reads as
        // "0.001/s", which is a true number that tells a user nothing.
        tooltip: {
          ...tooltip('rate'),
          formatter: (params: { value: [number, number | null] }[]) => {
            const t = (params[0]?.value?.[0] ?? 0) - grain / 2
            const slot = cells.get(t)
            if (!slot) return ''
            const end = new Date(t + grain).toLocaleTimeString(undefined, {
              hour: '2-digit',
              minute: '2-digit',
              hour12: false,
            })
            // Each provider is priced in its own currency on its own line,
            // because there is no exchange rate between them.
            const rows = providers
              .filter((id) => (slot.get(id)?.requests ?? 0) > 0)
              .map((id) => {
                const c = slot.get(id)!
                const cost = webCost(c)
                return `<span style="color:${colourOf(id)}">■</span> ${searchLabel(id)}: <b style="font-family:${mono}">${c.requests.toLocaleString()}</b>${cost ? ` <span style="color:${muted}">${cost}</span>` : ''}`
              })
            if (!rows.length) return ''
            return `<span style="color:${muted};font-size:11px;font-family:${mono}">${fmtStamp(new Date(t))} - ${end}</span><br/>${rows.join('<br/>')}`
          },
        },
        series: providers.map((id, i) => ({
          name: searchLabel(id),
          type: 'line' as const,
          stack: 'w',
          symbol: 'none',
          step: 'middle' as const,
          lineStyle: { width: 0 },
          itemStyle: { color: colourOf(id) },
          areaStyle: { color: colourOf(id), opacity: 0.9 },
          emphasis: { disabled: true },
          data: data.map((s) => [mid(s), count(id)(s)] as [number, number]),
          ...(i === 0 ? { markArea } : {}),
        })),
      },
    })
  }
  return out
})

const empty = computed(
  () => loaded.value && slots.value.length === 0 && generations.value.length === 0,
)

// re-arm the drag-zoom cursor whenever the panels re-render (each option
// merge resets the global cursor) and when charts first mount
watch(panels, () => void nextTick(armDragZoom), { flush: 'post' })
</script>

<template>
  <div ref="wrapEl" class="usg">
    <div class="usg__bar">
      <div class="usg__barrow">
        <Select v-model="portSel" :options="portOptions" />
        <Select v-model="spanSel" :options="spanOptions" />
        <Tooltip label="Back half a window">
          <button type="button" class="usg__nav" @click="shift(-1)">‹</button>
        </Tooltip>
        <Tooltip label="Zoom out">
          <button type="button" class="usg__nav" @click="zoomOut()">-</button>
        </Tooltip>
        <Tooltip label="Forward half a window">
          <button type="button" class="usg__nav" @click="shift(1)">›</button>
        </Tooltip>
        <span class="usg__spacer" />
        <span v-if="stampTime" class="usg__stamp" :class="{ 'usg__stamp--warn': stale }">
          <template v-if="stale">manager unreachable · showing {{ stampTime }}</template>
          <template v-else-if="!following()">browsing history · updated {{ stampTime }}</template>
          <template v-else>updated {{ stampTime }} · every 15 s</template>
        </span>
      </div>
      <div v-if="totals.requests" class="usg__barrow usg__barrow--stats">
        <span class="usg__stat">
          <span class="usg__statk">Requests</span>
          <span class="usg__statline">
            <span class="usg__statv">{{ totals.requests.toLocaleString() }}</span>
            <span v-if="totals.errors" class="usg__statc usg__statc--warn">{{ totals.errors.toLocaleString() }} failed</span>
          </span>
        </span>
        <span class="usg__statdiv" />
        <span class="usg__stat">
          <span class="usg__statk">Input tokens</span>
          <span class="usg__statline">
            <span class="usg__statv">{{ fmtTokens(totals.input) }}</span>
            <span v-if="totals.cached" class="usg__statc">{{ cachedPct }}% cached</span>
          </span>
        </span>
        <span class="usg__statdiv" />
        <span class="usg__stat">
          <span class="usg__statk">Output tokens</span>
          <span class="usg__statline">
            <span class="usg__statv">{{ fmtTokens(totals.output) }}</span>
          </span>
        </span>
        <template v-if="totals.drafted">
          <span class="usg__statdiv" />
          <span class="usg__stat">
            <span class="usg__statk">Draft acceptance</span>
            <span class="usg__statline">
              <span class="usg__statv">{{ Math.round((100 * totals.accepted) / totals.drafted) }}%</span>
            </span>
          </span>
        </template>
        <template v-if="webTotals.all.requests">
          <span class="usg__statdiv" />
          <span class="usg__stat">
            <span class="usg__statk">Web searches</span>
            <span class="usg__statline">
              <span class="usg__statv">{{ webTotals.all.requests.toLocaleString() }}</span>
              <span class="usg__statc">{{ webSummary }}</span>
            </span>
          </span>
        </template>
      </div>
    </div>

    <p v-if="empty" class="usg__empty">
      No usage recorded yet. Start a model and every request shows up on this timeline.
    </p>

    <template v-else>
      <div class="usg__panels">
        <div v-for="p in panels" :key="p.key" class="usg__panel">
          <VChart
            :ref="bindPanel(p.key)"
            class="usg__panelchart"
            :style="{ height: `${p.height}px` }"
            :option="p.option"
            :group="CHART_GROUP"
            autoresize
            @datazoom="onPanelZoom(p.key, $event)"
          />
        </div>
      </div>

      <div v-if="gaps.length" class="usg__gaps">
        <p class="usg__hd">Gaps in this window</p>
        <p v-for="g in gaps" :key="g.id" class="usg__gap">
          <span class="usg__gapmark" />{{ gapSentence(g) }}
        </p>
      </div>

      <div v-if="bands.length" class="usg__gens">
        <p class="usg__hd">What ran when</p>
        <div class="usg__tablewrap">
          <table class="usg__table">
            <thead>
              <tr>
                <th>Port</th>
                <th>Model</th>
                <th>From</th>
                <th>Until</th>
                <th>Version</th>
                <th>Started</th>
                <th>Ended</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="g in bands" :key="g.instance_id">
                <td class="c-mono">{{ g.port }}</td>
                <td class="c-name">
                  <Tooltip :label="genRaw(g)">
                    <span class="c-name__wrap">
                      <VendorLogo v-if="genVendor(g)" :vendor="genVendor(g)!" :size="14" />
                      {{ genName(g) }}
                    </span>
                  </Tooltip>
                </td>
                <td class="c-mono">{{ fmtStamp(new Date(g.started_ms)) }}</td>
                <td class="c-mono">
                  <span v-if="g.ended_ms == null" class="usg__live">running</span>
                  <template v-else>{{ fmtStamp(new Date(g.ended_ms)) }}</template>
                </td>
                <td class="c-mono">{{ g.runner_version || '-' }}</td>
                <td>{{ startText(g) }}</td>
                <td :class="{ 'c-warn': g.end_cause === 'crashed' }">{{ endText(g) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* A dashboard column that SCROLLS, like any Grafana board: toolbar (which
   owns the time range), stat cards, then one card per panel - each its own
   chart with its own legend and axes, crosshairs synced. */
.usg {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
/* two rows in one card: controls up top, the totals strip beneath - one row
   cannot hold both on a normal screen */
.usg__bar {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.usg__barrow {
  display: flex;
  align-items: center;
  gap: 12px;
}
.usg__nav {
  width: 26px;
  height: 26px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: var(--pk-text-muted);
  background: transparent;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md, 6px);
  cursor: pointer;
}
.usg__nav:hover {
  color: var(--pk-text-primary);
  border-color: var(--pk-border-strong);
}
.usg__spacer {
  flex: 1;
}
.usg__count {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.usg__stamp {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.usg__stamp--warn {
  color: var(--pk-status-warning);
}
/* the totals live in the toolbar as a compact strip - a whole card row for
   four numbers was a band of height the panels wanted */
/* the totals as a KPI strip: label over a prominent mono value, thin
   dividers between - the old stat cards' legibility at a strip's height */
.usg__barrow--stats {
  gap: 20px;
  flex-wrap: wrap;
}
.usg__stat {
  display: inline-flex;
  flex-direction: column;
  gap: 1px;
  white-space: nowrap;
}
.usg__statline {
  display: inline-flex;
  align-items: baseline;
  gap: 8px;
}
.usg__statdiv {
  width: 1px;
  align-self: stretch;
  background: var(--pk-border-subtle);
}
.usg__statk {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.usg__statv {
  font-family: var(--pk-font-mono);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--pk-text-primary);
  font-variant-numeric: tabular-nums;
}
.usg__statc {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-status-success);
  font-variant-numeric: tabular-nums;
}
.usg__statc--warn {
  color: var(--pk-status-warning);
}
/* two panels per row on a wide screen, one on narrow - each chart is
   self-contained (own axes/legend), so columns need no alignment contract */
.usg__panels {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(470px, 1fr));
  gap: 12px;
}
/* no padding: the chart owns the card edge-to-edge (its own top band holds
   the title + legend, its own margins pad the axes) */
.usg__panel {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.usg__panelchart {
  width: 100%;
}
.usg__hd {
  margin: 0 0 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
}
.usg__gaps {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 12px 14px;
}
.usg__gap {
  margin: 4px 0 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.usg__gapmark {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 8px;
  border-radius: 2px;
  background: repeating-linear-gradient(
    45deg,
    var(--pk-text-muted),
    var(--pk-text-muted) 2px,
    transparent 2px,
    transparent 4px
  );
  vertical-align: baseline;
}
.usg__gens {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 12px 14px;
}
.usg__tablewrap {
  overflow-x: auto;
}
.usg__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.usg__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 6px 10px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
}
.usg__table td {
  padding: 7px 10px;
  border-top: 1px solid var(--pk-border-default);
  white-space: nowrap;
  color: var(--pk-text-secondary);
}
.c-mono {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
}
.c-name {
  color: var(--pk-text-primary);
  max-width: 32ch;
  overflow: hidden;
  text-overflow: ellipsis;
}
.c-name__wrap {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.c-warn {
  color: var(--pk-status-warning);
}
.usg__live {
  color: var(--pk-status-success);
  font-weight: 600;
}
.usg__empty {
  padding: 32px 24px;
  text-align: center;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
</style>
