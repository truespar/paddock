<script setup lang="ts">
// Instrument: the Manager's read-only depth - three tabs, deep-linkable
// (?tab=activity|gpu|logs&port=NNNN) so every server row lands here
// pre-filtered. Activity = the request inspector (§8.1 records incl. the
// engine phase split); GPU = device + per-runner attribution; Logs = the
// §11.3 stream. Bodies are deliberately absent from Activity - records are
// metadata + timings by design (§8.3 inspect mode is a separate, never-stored
// path).
import { copyText } from '@/lib/clipboard'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Fuse from 'fuse.js'
import { servedId, useFleetStore } from '@/stores/fleet'
import { gpuApi, type GpuSnapshot } from '@/lib/api'
import { metalAllocated } from '@/lib/gpu-metrics'
import { fmtVram as gb, fmtClock, fmtStamp } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import Progress from '@/components/ui/Progress.vue'
import Select from '@/components/ui/Select.vue'
import Switch from '@/components/ui/Switch.vue'
import LogView from '@/components/manage/LogView.vue'
import UsageTimeline from '@/components/manage/UsageTimeline.vue'
import CachePanel from '@/components/manage/CachePanel.vue'

const route = useRoute()
const router = useRouter()
const fleet = useFleetStore()

type Tab = 'usage' | 'activity' | 'gpu' | 'cache' | 'logs'
// The tab is a ROUTE (/manage/instrument/logs), not view state - every view
// here is addressable and shareable.
const tab = computed<Tab>(() => {
  const t = String(route.params.tab ?? 'usage')
  return t === 'activity' || t === 'gpu' || t === 'cache' || t === 'logs' ? t : 'usage'
})
const TABS: readonly { id: Tab; label: string; icon: string }[] = [
  { id: 'usage', label: 'Usage', icon: 'chart-bar' },
  { id: 'activity', label: 'Activity', icon: 'activity' },
  { id: 'gpu', label: 'GPU', icon: 'graphics-card' },
  { id: 'cache', label: 'KV offloading', icon: 'hard-drive' },
  { id: 'logs', label: 'Logs', icon: 'scroll' },
]
function setTab(t: Tab): void {
  void router.push({ name: 'instrument', params: { tab: t }, query: route.query })
}
const portFilter = computed<number | null>(() => {
  const p = Number(route.query.port)
  return Number.isFinite(p) && p > 0 ? p : null
})
function setPort(p: number | null): void {
  const q = { ...route.query }
  if (p) q.port = String(p)
  else delete q.port
  void router.replace({ query: q })
}

let release: (() => void) | null = null
onMounted(() => {
  release = fleet.hold() // the port filters and log targets come from the fleet
})
onUnmounted(() => release?.())

// Select option lists. 'all' - Not '' - is the every-server sentinel: Reka's
// SelectItem refuses an empty-string value, because '' is what clears the
// selection and shows the placeholder, so an option carrying it threw on every
// render of this tab. logTargetOptions below already spells it 'all'; this now
// matches. The setter's `Number(v) || null` maps it back to "no filter"
// unchanged, since Number('all') is NaN.
// every serving role in the chain: a catalog model has `display`, but a
// hand-started file in any other role would otherwise label as "undefined"
const servedOf = (r: (typeof fleet.rows)[number]) => servedId(r)
const portOptions = computed(() => [
  { value: 'all', label: 'All servers' },
  ...fleet.rows.map((r) => ({
    value: r.port,
    label: `${r.port} - ${r.display ?? servedOf(r)}`,
    vendor: r.vendor ?? undefined,
    title: servedOf(r),
  })),
])
const portSel = computed({
  get: () => portFilter.value ?? 'all',
  set: (v: string | number) => setPort(Number(v) || null),
})
const logTargetOptions = computed(() => [
  { value: 'all', label: 'All', hint: 'manager + every server' },
  { value: 'manager', label: 'Manager' },
  ...fleet.rows.map((r) => ({
    value: String(r.port),
    label: `${r.port} - ${r.display ?? servedOf(r)}`,
    vendor: r.vendor ?? undefined,
    title: servedOf(r),
  })),
])

// ── Activity: the request inspector ─────────────────────────────────────────
/** A collected record: the runner's semconv-named fields + collector's port.
 *  Indexed loosely - the record schema grows runner-side without UI releases. */
type ActRow = Record<string, unknown> & { seq?: number; ts_ms?: number }
const acts = ref<ActRow[]>([])
const actsLoading = ref(false)
const live = ref(true)
const expanded = ref<Record<string, boolean>>({})

function actKey(a: ActRow): string {
  // port+seq alone collides: seq resets when a runner restarts, and Vue's
  // list patch with duplicate keys silently stops reordering (the "sort
  // does nothing" bug). ts_ms disambiguates across runner generations.
  return `${String(a.port ?? '')}-${String(a.seq ?? '')}-${String(a.ts_ms ?? '')}`
}
async function loadActivity(): Promise<void> {
  actsLoading.value = true
  try {
    const q = new URLSearchParams({ limit: '200' })
    if (portFilter.value) q.set('port', String(portFilter.value))
    const r = (await fetch(`/api/activity?${q}`).then((x) => x.json())) as {
      events?: ActRow[]
    }
    acts.value = r.events ?? []
  } catch {
    /* keep the last answer on a transient failure */
  } finally {
    actsLoading.value = false
  }
}
let actTimer: number | undefined
watch(
  [tab, live, portFilter],
  () => {
    if (actTimer !== undefined) clearInterval(actTimer)
    actTimer = undefined
    if (tab.value !== 'activity') return
    void loadActivity()
    if (live.value) actTimer = window.setInterval(() => void loadActivity(), 3000)
  },
  { immediate: true },
)
onUnmounted(() => {
  if (actTimer !== undefined) clearInterval(actTimer)
})

function n(a: ActRow, key: string): number | null {
  const v = a[key]
  return typeof v === 'number' ? v : null
}
function s(a: ActRow, key: string): string {
  const v = a[key]
  return typeof v === 'string' ? v : ''
}
function ts(a: ActRow): string {
  const t = n(a, 'ts_ms')
  // the full ISO stamp, always - a shortened "today" form just made rows
  // inconsistent with each other
  return t ? fmtStamp(new Date(t)) : '-'
}
function tsFull(a: ActRow): string {
  const t = n(a, 'ts_ms')
  return t ? fmtStamp(new Date(t)) : ''
}
/** The engine phase split as segments - drawn as a stacked mini bar (the
 *  string form got arbitrarily long); the full numbers ride the tooltip. */
const PHASE_KEYS = [
  ['tok', 'paddock.tokenize_ms'],
  ['queue', 'paddock.queue_ms'],
  ['prefill', 'paddock.prefill_ms'],
  ['decode', 'paddock.decode_ms'],
] as const
function phaseParts(a: ActRow): { label: string; ms: number }[] {
  return PHASE_KEYS.map(([label, key]) => ({ label, ms: n(a, key) ?? 0 })).filter(
    (p) => p.ms > 0,
  )
}
function phaseTotal(a: ActRow): number {
  return phaseParts(a).reduce((s, p) => s + p.ms, 0)
}
function phases(a: ActRow): string {
  return phaseParts(a)
    .map((p) => `${p.label} ${p.ms} ms`)
    .join(' · ')
}
function spec(a: ActRow): string {
  const d = n(a, 'paddock.spec_drafted')
  const acc = n(a, 'paddock.spec_accepted')
  return d ? `${acc ?? 0}/${d}` : '-'
}
async function purge(): Promise<void> {
  await fetch('/api/activity', { method: 'DELETE' })
  await loadActivity()
}

// ── the expanded row: a structured detail panel, not a JSON dump ────────────
const copiedKey = ref('')
async function copyJson(a: ActRow): Promise<void> {
  try {
    await copyText(JSON.stringify(a, null, 2))
    copiedKey.value = actKey(a)
    setTimeout(() => (copiedKey.value = ''), 1400)
  } catch {
    /* clipboard blocked */
  }
}
interface DetField {
  k: string
  v: string
}
interface DetGroup {
  title: string
  fields: DetField[]
}
/** Curated groups from the semconv record; only present fields render, and
 *  the raw record stays available behind the fold for everything else. */
function detailGroups(a: ActRow): DetGroup[] {
  const ms = (v: number | null) => (v !== null ? `${v.toLocaleString()} ms` : '')
  const req: DetField[] = [
    { k: 'Endpoint', v: s(a, 'endpoint') },
    { k: 'Model', v: s(a, 'gen_ai.response.model') || s(a, 'gen_ai.request.model') },
    { k: 'Server', v: n(a, 'port') !== null ? String(n(a, 'port')) : '' },
    { k: 'Status', v: n(a, 'status') !== null ? String(n(a, 'status')) : '' },
    { k: 'Session', v: s(a, 'session') },
    { k: 'Error', v: s(a, 'error.type') },
  ]
  const timing: DetField[] = [
    { k: 'First token', v: ms(n(a, 'paddock.ttft_ms')) },
    { k: 'Tokenize', v: ms(n(a, 'paddock.tokenize_ms')) },
    { k: 'Queue', v: ms(n(a, 'paddock.queue_ms')) },
    { k: 'Prefill', v: ms(n(a, 'paddock.prefill_ms')) },
    { k: 'Decode', v: ms(n(a, 'paddock.decode_ms')) },
    { k: 'Total', v: phaseTotal(a) ? `${phaseTotal(a).toLocaleString()} ms` : '' },
  ]
  const din = n(a, 'gen_ai.usage.input_tokens')
  const dout = n(a, 'gen_ai.usage.output_tokens')
  const drafted = n(a, 'paddock.spec_drafted')
  const accepted = n(a, 'paddock.spec_accepted')
  const tokens: DetField[] = [
    { k: 'Input', v: din !== null ? din.toLocaleString() : '' },
    { k: 'Output', v: dout !== null ? dout.toLocaleString() : '' },
    { k: 'Speed', v: n(a, 'paddock.decode_tok_s') !== null ? `${n(a, 'paddock.decode_tok_s')!.toFixed(1)} tok/s` : '' },
    {
      k: 'Speculative',
      v: drafted
        ? `${accepted ?? 0} of ${drafted} drafts accepted (${Math.round((100 * (accepted ?? 0)) / drafted)}%)`
        : '',
    },
  ]
  return [
    { title: 'Request', fields: req.filter((f) => f.v) },
    { title: 'Timing', fields: timing.filter((f) => f.v) },
    { title: 'Tokens', fields: tokens.filter((f) => f.v) },
  ].filter((g) => g.fields.length)
}

// ── Activity filtering + ordering: fuse for fuzzy text (the app's one
// search library), click-to-sort headers, an errors-only pill ─────────────
const actSearch = ref('')
const errorsOnly = ref(false)
const sortKey = ref('ts_ms')
const sortDir = ref<1 | -1>(-1)
function setSort(k: string): void {
  if (sortKey.value === k) sortDir.value = sortDir.value === -1 ? 1 : -1
  else {
    sortKey.value = k
    sortDir.value = -1
  }
}
const SORT_GET: Record<string, (a: ActRow) => number | string> = {
  ts_ms: (a) => n(a, 'ts_ms') ?? 0,
  port: (a) => n(a, 'port') ?? 0,
  endpoint: (a) => s(a, 'endpoint'),
  status: (a) => n(a, 'status') ?? 0,
  tokens: (a) => n(a, 'gen_ai.usage.output_tokens') ?? -1,
  ttft: (a) => n(a, 'paddock.ttft_ms') ?? -1,
  toks: (a) => n(a, 'paddock.decode_tok_s') ?? -1,
}
/** One searchable string per record (the semconv keys contain dots, which
 *  fuse would read as nested paths - a getFn sidesteps that). */
function blobOf(a: ActRow): string {
  return [
    n(a, 'port'),
    s(a, 'endpoint'),
    s(a, 'gen_ai.request.model'),
    s(a, 'session'),
    n(a, 'status'),
  ]
    .filter((x) => x !== null && x !== '')
    .join(' ')
}
const shown = computed(() => {
  let out = acts.value
  if (errorsOnly.value) out = out.filter((a) => (n(a, 'status') ?? 0) >= 400)
  const q = actSearch.value.trim()
  if (q) {
    const fuse = new Fuse(out, {
      keys: [{ name: 'blob', getFn: blobOf }],
      threshold: 0.3,
      ignoreLocation: true,
    })
    out = fuse.search(q).map((r) => r.item)
  }
  const get = SORT_GET[sortKey.value]
  if (get) {
    out = [...out].sort((x, y) => {
      const a = get(x)
      const b = get(y)
      return (a < b ? -1 : a > b ? 1 : 0) * sortDir.value
    })
  }
  return out
})
/** aria-sort + the header arrow for a sortable column. */
function sortState(k: string): 'ascending' | 'descending' | undefined {
  if (sortKey.value !== k) return undefined
  return sortDir.value === 1 ? 'ascending' : 'descending'
}

// ── GPU: device + attribution ───────────────────────────────────────────────
const snap = ref<GpuSnapshot | null>(null)
let gpuTimer: number | undefined
watch(
  tab,
  (t) => {
    if (gpuTimer !== undefined) clearInterval(gpuTimer)
    gpuTimer = undefined
    if (t !== 'gpu') return
    const poll = async () => {
      try {
        snap.value = await gpuApi.get()
      } catch {
        /* transient */
      }
    }
    void poll()
    gpuTimer = window.setInterval(() => void poll(), 2000)
  },
  { immediate: true },
)
onUnmounted(() => {
  if (gpuTimer !== undefined) clearInterval(gpuTimer)
})
const runnersVram = computed(() => snap.value?.reconciliation?.runners ?? [])

// ── Logs: the §11.3 stream (LogView owns the connection + presentation) ─────
const logTarget = ref<string>('all')

// A server row's "Logs" action lands with ?port= - honor it as the target.
watch(
  [tab, portFilter],
  () => {
    if (tab.value === 'logs' && portFilter.value) logTarget.value = String(portFilter.value)
  },
  { immediate: true },
)
</script>

<template>
  <div class="ins">
    <h1 class="ins__title">Instrument</h1>

    <div class="ins__tabs">
      <button
        v-for="t in TABS"
        :key="t.id"
        class="ins__tab"
        :class="{ 'ins__tab--on': tab === t.id }"
        @click="setTab(t.id)"
      >
        <Icon :name="t.icon" :size="14" />
        {{ t.label }}
      </button>
    </div>

    <!-- ── Usage: the timeline over the manager's scraped rollups ───────── -->
    <section v-if="tab === 'usage'">
      <UsageTimeline :port="portFilter" @update:port="setPort" />
    </section>

    <!-- ── Activity ─────────────────────────────────────────────────────── -->
    <section v-else-if="tab === 'activity'">
      <div class="ins__bar">
        <Select v-model="portSel" :options="portOptions" />
        <input
          v-model="actSearch"
          class="ins__search"
          type="search"
          placeholder="Search requests... (endpoint, model, session, status)"
        />
        <button
          type="button"
          class="ins__pill"
          :class="{ 'ins__pill--on': errorsOnly }"
          @click="errorsOnly = !errorsOnly"
        >
          Errors only
        </button>
        <label class="ins__check"><Switch v-model="live" label="Live" /> live</label>
        <span class="ins__spacer" />
        <span class="ins__count">
          {{ shown.length }}<template v-if="shown.length !== acts.length"> of {{ acts.length }}</template>
          requests
        </span>
        <Tooltip label="Delete all collected activity">
          <button class="pk-btn pk-btn--sm pk-btn--ghost" @click="purge">
            <Icon name="trash" :size="13" /> Purge
          </button>
        </Tooltip>
      </div>

      <div class="ins__tablewrap">
        <table class="ins__table">
          <thead>
            <tr>
              <th class="th-sort" :aria-sort="sortState('ts_ms')" @click="setSort('ts_ms')">
                Time <Icon v-if="sortKey === 'ts_ms'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
              <th class="th-sort" :aria-sort="sortState('port')" @click="setSort('port')">
                Server <Icon v-if="sortKey === 'port'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
              <th class="th-sort" :aria-sort="sortState('endpoint')" @click="setSort('endpoint')">
                Endpoint <Icon v-if="sortKey === 'endpoint'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
              <th class="th-sort" :aria-sort="sortState('status')" @click="setSort('status')">
                Status <Icon v-if="sortKey === 'status'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
              <th class="th-sort" :aria-sort="sortState('tokens')" @click="setSort('tokens')">
                Tokens <Icon v-if="sortKey === 'tokens'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
              <th class="th-sort" :aria-sort="sortState('ttft')" @click="setSort('ttft')">
                TTFT <Icon v-if="sortKey === 'ttft'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
              <th>Phases (ms)</th>
              <th>
                <Tooltip label="Speculative decode: accepted / drafted tokens"><span>Spec</span></Tooltip>
              </th>
              <th class="th-sort" :aria-sort="sortState('toks')" @click="setSort('toks')">
                tok/s <Icon v-if="sortKey === 'toks'" :name="sortDir === 1 ? 'chevron-up' : 'chevron-down'" :size="11" />
              </th>
            </tr>
          </thead>
          <tbody>
            <template v-for="a in shown" :key="actKey(a)">
              <tr class="arow" @click="expanded[actKey(a)] = !expanded[actKey(a)]">
                <td class="c-mono">
                  <Tooltip :label="tsFull(a)"><span>{{ ts(a) }}</span></Tooltip>
                </td>
                <td class="c-mono">{{ n(a, 'port') ?? '-' }}</td>
                <td class="c-mono c-ep">{{ s(a, 'endpoint') }}</td>
                <td>
                  <span class="code" :class="{ 'code--bad': (n(a, 'status') ?? 0) >= 400 }">
                    {{ n(a, 'status') ?? '-' }}
                  </span>
                </td>
                <td class="c-mono">
                  {{ n(a, 'gen_ai.usage.input_tokens') ?? '-' }} -> {{ n(a, 'gen_ai.usage.output_tokens') ?? '-' }}
                </td>
                <td class="c-mono">{{ n(a, 'paddock.ttft_ms') !== null ? `${n(a, 'paddock.ttft_ms')} ms` : '-' }}</td>
                <td class="c-phases">
                  <Tooltip v-if="phaseParts(a).length" :label="phases(a)">
                    <span class="ph">
                      <span class="ph__total">{{ phaseTotal(a).toLocaleString() }} ms</span>
                      <span class="ph__bar">
                        <span
                          v-for="p in phaseParts(a)"
                          :key="p.label"
                          class="ph__seg"
                          :class="`ph__seg--${p.label}`"
                          :style="{ width: `${(100 * p.ms) / phaseTotal(a)}%` }"
                        />
                      </span>
                    </span>
                  </Tooltip>
                  <span v-else class="c-mono">-</span>
                </td>
                <td class="c-mono">{{ spec(a) }}</td>
                <td class="c-mono">{{ n(a, 'paddock.decode_tok_s')?.toFixed(1) ?? '-' }}</td>
              </tr>
              <tr v-if="expanded[actKey(a)]" class="arow__detail">
                <td colspan="9">
                  <div class="det">
                    <div class="det__head">
                      <span class="det__when">{{ tsFull(a) }}</span>
                      <span class="ins__spacer" />
                      <button class="pk-btn pk-btn--sm pk-btn--ghost" @click.stop="copyJson(a)">
                        <Icon :name="copiedKey === actKey(a) ? 'check' : 'copy'" :size="13" />
                        {{ copiedKey === actKey(a) ? 'Copied' : 'Copy JSON' }}
                      </button>
                    </div>
                    <div class="det__grid">
                      <section v-for="g in detailGroups(a)" :key="g.title" class="det__card">
                        <p class="det__hd">{{ g.title }}</p>
                        <dl class="det__facts">
                          <template v-for="f in g.fields" :key="f.k">
                            <dt>{{ f.k }}</dt>
                            <dd>{{ f.v }}</dd>
                          </template>
                        </dl>
                      </section>
                    </div>
                    <Collapsible class="det__raw" summary="Raw record">
                      <pre>{{ JSON.stringify(a, null, 2) }}</pre>
                    </Collapsible>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
        <p v-if="!shown.length && !actsLoading" class="ins__empty">
          {{ acts.length ? 'Nothing matches the filter.' : 'No requests collected yet. Every call to a server shows up here with its full timing breakdown.' }}
        </p>
      </div>
    </section>

    <!-- ── GPU ──────────────────────────────────────────────────────────── -->
    <section v-else-if="tab === 'gpu'">
      <div v-if="snap?.available" class="gpu">
        <div v-for="(g, i) in snap.gpus" :key="i" class="gpu__card">
          <p class="gpu__name">{{ g.name ?? `GPU ${i}` }}</p>
          <div class="gpu__stats">
            <template v-if="g.metal">
              <div class="gpu__stat"><span class="gpu__k">Thermal pressure</span><span class="gpu__v">{{ g.metal.thermal_pressure ?? 'Unavailable' }}</span></div>
              <div class="gpu__stat"><span class="gpu__k">Memory pressure</span><span class="gpu__v">{{ g.metal.memory_pressure ?? 'Not reported' }}</span></div>
              <div class="gpu__stat"><span class="gpu__k">Unified memory</span><span class="gpu__v">{{ g.metal.unified_memory_total != null ? gb(g.metal.unified_memory_total) : '—' }}</span></div>
              <div class="gpu__stat"><span class="gpu__k">Metal working set</span><span class="gpu__v">{{ gb(g.metal.recommended_working_set) }}</span></div>
              <div v-if="metalAllocated(snap) != null" class="gpu__stat"><span class="gpu__k">Paddock allocations</span><span class="gpu__v">{{ gb(metalAllocated(snap)!) }}</span></div>
            </template>
            <div v-if="g.util_gpu != null" class="gpu__stat">
              <span class="gpu__k">Utilization</span>
              <span class="gpu__v">{{ g.util_gpu ?? '-' }}%</span>
            </div>
            <div v-if="g.mem_total != null" class="gpu__stat gpu__stat--wide">
              <!-- a GB10 computes in the machine's RAM: no VRAM to name -->
              <span class="gpu__k">{{ g.host_memory ? 'Unified memory' : 'VRAM' }}</span>
              <Progress
                class="gpu__bar"
                :value="g.mem_used ?? 0"
                :max="g.mem_total ?? 0"
                :label="`${gb(g.mem_used ?? 0)} of ${gb(g.mem_total ?? 0)}`"
                size="md"
              />
              <span class="gpu__v">{{ g.mem_used ? gb(g.mem_used) : '-' }} / {{ g.mem_total ? gb(g.mem_total) : '-' }}</span>
            </div>
            <div v-if="g.power_w != null" class="gpu__stat">
              <span class="gpu__k">Power</span>
              <span class="gpu__v">{{ g.power_w ? `${Math.round(g.power_w)} W` : '-' }}</span>
            </div>
            <div v-if="g.temp_c != null" class="gpu__stat">
              <span class="gpu__k">Temp</span>
              <span class="gpu__v">{{ g.temp_c ? `${g.temp_c}°C` : '-' }}</span>
            </div>
          </div>
        </div>

        <template v-if="snap.gpus.some(g => g.metal)">
          <div v-for="r in snap.reconciliation?.runners ?? []" :key="`${r.port}:${r.pid}`" class="gpu__card">
            <p class="gpu__name">Runner · {{ r.port }}</p>
            <div v-if="r.metal" class="gpu__stats">
              <div class="gpu__stat"><span class="gpu__k">Metal allocations</span><span class="gpu__v">{{ gb(r.metal.allocated_bytes) }}</span></div>
              <div class="gpu__stat"><span class="gpu__k">GPU time · cumulative</span><span class="gpu__v">{{ r.metal.gpu_seconds_total.toFixed(2) }} s</span></div>
              <div v-if="r.metal.last_command_ms != null" class="gpu__stat"><span class="gpu__k">Last GPU command</span><span class="gpu__v">{{ r.metal.last_command_ms.toFixed(2) }} ms</span></div>
            </div>
            <p v-else>Measurements unavailable · restart with the updated runner</p>
          </div>
        </template>
        <template v-else-if="runnersVram.length">
          <p class="gpu__hd">Per-server VRAM - what the engine counts vs what the OS attributes to it</p>
          <div class="ins__tablewrap">
            <table class="ins__table">
              <thead>
                <tr>
                  <th>Server</th>
                  <th>GPU</th>
                  <th>Engine count</th>
                  <th>OS view</th>
                  <th>
                    <Tooltip label="OS view - engine count: the CUDA context and its workspaces. Flagged when it grows past the expected range (a leak, or fragmentation)."><span>Drift</span></Tooltip>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in runnersVram" :key="r.port">
                  <td class="c-mono">
                    {{ r.port }}
                    <span class="c-name">{{ fleet.rows.find((x) => x.port === r.port)?.display ?? '' }}</span>
                  </td>
                  <td class="c-mono">{{ r.gpu ?? '-' }}</td>
                  <td class="c-mono">{{ r.self_mem ? gb(r.self_mem) : '-' }}</td>
                  <td class="c-mono">{{ r.nvml_mem ? gb(r.nvml_mem) : '-' }}</td>
                  <td class="c-mono" :class="{ 'c-warn': r.anomaly }">
                    {{ r.drift !== null && r.drift !== undefined ? gb(Math.abs(r.drift)) : '-' }}
                    <Icon v-if="r.anomaly" name="alert-triangle" :size="12" />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </div>
      <p v-else class="ins__empty">GPU measurements are unavailable on this machine.</p>
    </section>

    <section v-else-if="tab === 'cache'">
      <CachePanel />
    </section>

    <!-- ── Logs ─────────────────────────────────────────────────────────── -->
    <section v-else>
      <div class="ins__bar">
        <Select v-model="logTarget" :options="logTargetOptions" />
      </div>
      <LogView :target="logTarget" fill />
    </section>
  </div>
</template>

<style scoped>
.ins {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.ins__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.ins__lead {
  margin: 0 0 16px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.ins__tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--pk-border-default);
  margin-bottom: 14px;
}
.ins__tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border: none;
  border-bottom: 2px solid transparent;
  background: none;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
}
.ins__tab:hover {
  color: var(--pk-text-primary);
}
.ins__tab--on {
  color: var(--pk-accent);
  border-bottom-color: var(--pk-accent);
  font-weight: 600;
}
.ins__bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.ins__select {
  padding: 6px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
}
.ins__check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.ins__search {
  flex: 1;
  min-width: 180px;
  max-width: 360px;
  padding: 6px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
}
.ins__search:focus {
  outline: none;
  border-color: var(--pk-accent);
}
.ins__pill {
  padding: 5px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full);
  background: transparent;
  color: var(--pk-text-secondary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}
.ins__pill--on {
  border-color: var(--pk-text-danger);
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
}
.th-sort {
  cursor: pointer;
  user-select: none;
}
.th-sort:hover {
  color: var(--pk-text-primary);
}
.ins__spacer {
  flex: 1;
}
.ins__count {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.ins__tablewrap {
  overflow-x: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  /* the table sits on a CARD - without this the page canvas showed through
     and light theme read as one grey slab */
  background: var(--pk-bg-surface);
  /* container units for the detail panel: 100cqw = the card's VISIBLE
     width, not the table's scrollable width */
  container-type: inline-size;
}
.ins__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.ins__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 9px 12px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
  background: var(--pk-bg-surface);
}
.ins__table td {
  padding: 8px 12px;
  border-top: 1px solid var(--pk-border-default);
  vertical-align: middle;
  white-space: nowrap;
}
.arow {
  cursor: pointer;
}
.arow:hover td {
  background: var(--pk-bg-hover);
}
.arow__detail td {
  white-space: normal;
  background: var(--pk-bg-base);
  padding: 12px 14px;
}
/* the expanded record: grouped facts + Copy JSON; the raw dump folds away.
   The td spans the table's SCROLLABLE width (nowrap columns can exceed the
   card) - sticky-left + container width keep the panel fully on screen
   without horizontal scrolling. */
.det {
  display: flex;
  flex-direction: column;
  gap: 10px;
  position: sticky;
  left: 14px;
  width: calc(100cqw - 28px);
}
.det__head {
  display: flex;
  align-items: center;
  gap: 10px;
}
.det__when {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.det__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.det__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  padding: 10px 12px;
}
.det__hd {
  margin: 0 0 8px;
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
}
.det__facts {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 14px;
  margin: 0;
}
.det__facts dt {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
.det__facts dd {
  margin: 0;
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.det__raw pre {
  margin: 8px 0 0;
  padding: 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 320px;
  overflow-y: auto;
}
.c-mono {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
.c-ep {
  max-width: 22ch;
  overflow: hidden;
  text-overflow: ellipsis;
}
.c-phases {
  color: var(--pk-text-muted);
}
/* the phase split: total + a stacked mini bar; numbers ride the tooltip */
.ph {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.ph__total {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
  min-width: 7ch;
  text-align: right;
}
.ph__bar {
  display: inline-flex;
  width: 96px;
  height: 6px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  overflow: hidden;
}
.ph__seg {
  height: 100%;
}
.ph__seg--tok {
  background: var(--pk-text-muted);
}
.ph__seg--queue {
  background: var(--pk-status-warning);
}
.ph__seg--prefill {
  background: color-mix(in srgb, var(--pk-accent) 55%, transparent);
}
.ph__seg--decode {
  background: var(--pk-accent);
}
.c-warn {
  color: var(--pk-status-warning);
}
.c-name {
  margin-left: 8px;
  font-family: var(--pk-font-sans);
  color: var(--pk-text-muted);
}
.code {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  padding: 1px 6px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-elevated);
  color: var(--pk-text-secondary);
}
.code--bad {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
}
.ins__empty {
  padding: 24px;
  text-align: center;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}

/* GPU tab */
.gpu__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  padding: 14px;
  margin-bottom: 12px;
  background: var(--pk-bg-surface);
}
.gpu__name {
  margin: 0 0 10px;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.gpu__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 28px;
  align-items: center;
}
.gpu__stat {
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.gpu__stat--wide {
  flex: 1;
  min-width: 240px;
  align-items: center;
}
.gpu__k {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.gpu__v {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.gpu__stat .gpu__bar {
  flex: 1;
}
.gpu__hd {
  margin: 16px 0 8px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}

</style>
