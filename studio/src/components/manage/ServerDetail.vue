<script setup lang="ts">
// One server's page (/manage/servers/:port): where a user naturally goes to
// see and CHANGE a running server. Status + endpoint up top, the as-deployed
// config with Edit (the takeover form), live toggles (pin, start-on-boot),
// a live log tail, and cross-links into Instrument for depth.
import { copyText } from '@/lib/clipboard'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { servedId as servedOf, useFleetStore, type ResidencySnapshot } from '@/stores/fleet'
import { useDownloadsStore, jobActive } from '@/stores/downloads'
import { selectStudioModel } from '@/lib/select-model'
import { gpuApi } from '@/lib/api'
import { fmtVram as gb, fmtTokens as fmtCtx } from '@/lib/format'
import { modelLabel, modelVendor } from '@/lib/model-name'
import { searchLabel } from '@/lib/websearch'
import Icon from '@/components/Icon.vue'
import Switch from '@/components/ui/Switch.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import SearchLogo from '@/components/manage/SearchLogo.vue'
import LogView from '@/components/manage/LogView.vue'

const route = useRoute()
const router = useRouter()
const fleet = useFleetStore()
const downloads = useDownloadsStore()

const port = computed(() => Number(route.params.port))
const row = computed(() => fleet.rows.find((r) => r.port === port.value))
const deployingHere = computed(() =>
  fleet.deploying.find((d) => d.port === port.value),
)
/** A manager-side download whose queued start owns this port. */
const downloadHere = computed(() =>
  downloads.visible.find((j) => j.start?.port === port.value && jobActive(j)),
)
/** A configured-but-stopped endpoint on this port: its file (and API key)
 *  are kept - the page offers Start / Edit / Remove instead of a 404. */
const stoppedHere = computed(() => fleet.stopped.find((c) => c.port === port.value))
const boots = computed(() => fleet.bootPorts.has(port.value))

/** What this page calls the model: catalog display first, id as fallback -
 *  whichever serving role the runner is in (the fleet store's `servedId`). */
const servedId = computed(() => (row.value ? servedOf(row.value) : undefined))
const title = computed(() => {
  const t = row.value?.display ?? modelLabel(servedId.value)
  return t || `server ${port.value}`
})
const techId = computed(() => servedId.value ?? '')
const vendor = computed(() => row.value?.vendor ?? modelVendor(servedId.value))

let release: (() => void) | null = null
onMounted(() => {
  release = fleet.hold()
  void downloads.load()
  // the box's GPUs, once - so pins and attribution read as "GPU 0 · RTX
  // A6000" instead of a bare UUID or a dash
  void gpuApi
    .get()
    .then((s) => {
      gpus.value = (s.gpus ?? []).map((g, i) => ({
        index: g.index ?? i,
        name: g.name ?? `GPU ${i}`,
        uuid: g.uuid ?? null,
      }))
    })
    .catch(() => {})
})
onUnmounted(() => release?.())

// ── GPU naming: the file stores a UUID (ordinal-proof), people read names ───
const gpus = ref<{ index: number; name: string; uuid: string | null }[]>([])
function gpuName(index: number): string {
  const g = gpus.value.find((x) => x.index === index)
  return g ? `GPU ${g.index} · ${g.name}` : `GPU ${index}`
}
/** The configured pin, humanized: UUID/ordinal -> "GPU 0 · RTX A6000". */
const gpuPinLabel = computed(() => {
  const pin = cfg.value?.gpu
  if (pin === null || pin === undefined || pin === '') {
    const g = gpus.value[0]
    return g ? `${gpuName(g.index)}${gpus.value.length > 1 ? ' (default)' : ''}` : 'driver default'
  }
  const p = String(pin)
  const byUuid = gpus.value.find((g) => g.uuid === p)
  if (byUuid) return gpuName(byUuid.index)
  const idx = Number(p)
  if (Number.isFinite(idx) && gpus.value.some((g) => g.index === idx)) return gpuName(idx)
  // a pin for hardware this box doesn't show (moved card?) - shortened, the
  // tooltip has it whole
  return p.length > 20 ? `${p.slice(0, 20)}...` : p
})
/** Where it actually runs: NVML attribution when the OS gives it, else the
 *  pin (Windows WDDM hides per-process bytes - a dash said nothing). */
const liveGpuLabel = computed(() => {
  const idx = row.value?.vram?.gpu
  return idx !== null && idx !== undefined ? gpuName(idx) : gpuPinLabel.value
})

function uptime(s: number | null | undefined): string {
  if (s === null || s === undefined) return '-'
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

/** What the model is doing on a runner that loads and unloads it by
 *  policy, and the policy itself - so an endpoint that shows no VRAM
 *  reads as idle on purpose, not as broken. */
function residencyPhase(s: ResidencySnapshot): string {
  switch (s.phase) {
    case 'loaded':
      return 'Loaded'
    case 'loading':
      return 'Loading'
    case 'unloading':
      return 'Unloading'
    case 'failed':
      return s.last_error ? `Load failed - ${s.last_error}` : 'Load failed'
    default:
      return s.policy.load === 'on_demand' ? 'Idle - loads on the first request' : 'Unloaded'
  }
}
function residencyPolicy(s: ResidencySnapshot): string {
  const load = s.policy.load === 'on_demand' ? 'On first request' : 'At runner startup'
  const idle =
    s.policy.unload_after_idle_seconds == null
      ? 'stays loaded'
      : `unloads after ${s.policy.unload_after_idle_seconds} s idle`
  return `${load} · ${idle}`
}
function residencyCounts(s: ResidencySnapshot): string {
  const last = s.last_load_ms != null ? ` · last load ${s.last_load_ms} ms` : ''
  return `${s.loads} load${s.loads === 1 ? '' : 's'} · ${s.unloads} unload${s.unloads === 1 ? '' : 's'}${last}`
}

/** Human status labels - Title case, "Running" instead of API-speak "ok". */
function statusLabel(s: string): string {
  if (s === 'ok') return 'Running'
  if (s === 'draining') return 'Stopping'
  if (s === 'unreachable') return 'Unreachable'
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

const copied = ref(false)
async function copyEndpoint(): Promise<void> {
  if (!row.value) return
  try {
    await copyText(`${row.value.endpoint}/v1`)
    copied.value = true
    setTimeout(() => (copied.value = false), 1400)
  } catch {
    /* clipboard blocked */
  }
}
const keyCopied = ref(false)
async function copyKey(): Promise<void> {
  const k = cfg.value?.api_key
  if (!k) return
  try {
    await copyText(k)
    keyCopied.value = true
    setTimeout(() => (keyCopied.value = false), 1400)
  } catch {
    /* clipboard blocked */
  }
}

function openInStudio(): void {
  // a table predictor has its own page: rows in, a column out, no turns
  if (row.value?.tabular) {
    void router.push({ name: 'tables', query: { port: String(port.value) } })
    return
  }
  // an image model takes turns in the chat surface like any other, so it
  // opens there too - selected, on a fresh draft
  const id = row.value?.model ?? row.value?.embedder ?? row.value?.image
  if (id) selectStudioModel(id)
  void router.push({ name: 'home' })
}

const actionErr = ref('')
async function togglePin(): Promise<void> {
  if (!row.value) return
  await fleet.setPinned(port.value, !row.value.pinned)
}
async function toggleBoot(): Promise<void> {
  actionErr.value = ''
  try {
    await fleet.setPersist(port.value, !boots.value)
  } catch (e) {
    // e.g. an adopted runner: the manager honestly can't reconstruct its spec
    actionErr.value = e instanceof Error ? e.message : String(e)
  }
}
const stopping = ref(false)
async function stop(): Promise<void> {
  stopping.value = true
  try {
    // stay on this page: the endpoint remains configured and the page
    // becomes its stopped view (Start brings it right back)
    await fleet.stop(port.value)
  } finally {
    stopping.value = false
  }
}

const startingUp = ref(false)
async function startAgain(): Promise<void> {
  const c = stoppedHere.value
  if (!c || startingUp.value) return
  startingUp.value = true
  try {
    await fleet.startConfigured(c.port, c.display ?? c.model ?? String(c.port))
  } finally {
    startingUp.value = false
  }
}
const confirmRemove = ref(false)
const removeErr = ref('')
async function removeEndpoint(): Promise<void> {
  if (!confirmRemove.value) {
    confirmRemove.value = true
    setTimeout(() => (confirmRemove.value = false), 3000)
    return
  }
  confirmRemove.value = false
  removeErr.value = ''
  try {
    await fleet.removeConfigured(port.value)
    void router.push({ name: 'servers' })
  } catch (e) {
    removeErr.value = e instanceof Error ? e.message : String(e)
  }
}

// The live log tail is the LogView component (parsed lines, filtering,
// sticky follow) - scoped to this server via its target prop.

const cfg = computed(() => row.value?.config)
// The endpoint's system tools, one line: "Exa search · docs, github". The
// search half is split out from the MCP half so it can carry its provider's
// mark - the same mark the start form offered, so what you chose there is
// recognisable here without reading.
const webProvider = computed(() => cfg.value?.web_search_provider ?? '')
const mcpLabels = computed(() =>
  (cfg.value?.mcp_servers ?? []).map((s) => s.server_label).join(', '),
)
const noTools = computed(() => !!cfg.value && !webProvider.value && !mcpLabels.value)
</script>

<template>
  <div class="sd">
    <nav class="sd__crumbs">
      <RouterLink :to="{ name: 'servers' }">Models</RouterLink>
      <span>/</span>
      <span>{{ row ? title : stoppedHere ? (stoppedHere.display ?? (modelLabel(stoppedHere.model) || `server ${port}`)) : port }}</span>
    </nav>

    <template v-if="row">
      <div class="sd__head">
        <div class="sd__id">
          <VendorLogo v-if="vendor" :vendor="vendor" :size="24" />
          <Tooltip :label="`${techId} · port ${port}`">
            <h1 class="sd__title">{{ title }}</h1>
          </Tooltip>
          <!-- healthy is QUIET: the Live card says running; boot/pin state
               lives on the Configuration toggles. The chip appears only when
               something is off (Stopping / Unreachable) - that must be loud. -->
          <span
            v-if="row.status !== 'ok'"
            class="sd__status"
            :class="`sd__status--${row.status === 'draining' ? 'warn' : 'bad'}`"
          >
            <span class="sd__dot" /> {{ statusLabel(row.status) }}
          </span>
        </div>
        <div class="sd__actions">
          <button class="pk-btn pk-btn--sm" @click="openInStudio">
            <Icon name="external-link" :size="13" /> Open in Studio
          </button>
          <RouterLink
            class="pk-btn pk-btn--sm"
            :to="{ name: 'server-edit', params: { port: String(port) } }"
          >
            <Icon name="edit" :size="13" /> Edit
          </RouterLink>
          <button class="pk-btn pk-btn--sm pk-btn--danger" :disabled="stopping" @click="stop">
            <Icon name="stop" :size="13" /> {{ stopping ? 'Stopping...' : 'Stop' }}
          </button>
        </div>
      </div>

      <!-- the endpoint is the product of this page - front and centre.
           Two clean rows: the address, the key. Nothing else. -->
      <div class="sd__endpoint">
        <div class="sd__ep-row">
          <span class="sd__ep-lbl">Endpoint</span>
          <code>{{ row.endpoint }}/v1</code>
          <Tooltip :label="copied ? 'Copied' : 'Copy base URL'">
            <button class="sd__ep-copy" @click="copyEndpoint">
              <Icon :name="copied ? 'check' : 'copy'" :size="13" /> {{ copied ? 'Copied' : 'Copy' }}
            </button>
          </Tooltip>
        </div>
        <div v-if="cfg?.api_key" class="sd__ep-row">
          <span class="sd__ep-lbl">API key</span>
          <code class="sd__ep-key">{{ cfg.api_key }}</code>
          <Tooltip :label="keyCopied ? 'Copied' : 'Copy API key'">
            <button class="sd__ep-copy" @click="copyKey">
              <Icon :name="keyCopied ? 'check' : 'copy'" :size="13" /> {{ keyCopied ? 'Copied' : 'Copy' }}
            </button>
          </Tooltip>
        </div>
        <div class="sd__ep-row">
          <span class="sd__ep-lbl">API</span>
          <RouterLink class="sd__ep-api" :to="{ name: 'server-api', params: { port: String(port) } }">
            <Icon name="file-text" :size="13" /> API reference
          </RouterLink>
        </div>
      </div>

      <div class="sd__grid">
        <!-- facts: live numbers -->
        <section class="sd__card">
          <p class="sd__card-hd">Live</p>
          <dl class="sd__facts">
            <dt>Uptime</dt><dd>{{ uptime(row.uptime_s) }}</dd>
            <dt>Active requests</dt><dd>{{ row.in_flight ?? '-' }}</dd>
            <dt>VRAM</dt>
            <dd :class="{ 'sd__warn': row.vram?.anomaly }">
              {{ row.vram?.self_mem ? gb(row.vram.self_mem) : '-' }}
              <template v-if="row.vram?.anomaly">
                <Icon name="alert-triangle" :size="12" /> drift
              </template>
            </dd>
            <dt>GPU</dt><dd>{{ liveGpuLabel }}</dd>
            <template v-if="row.residency">
              <dt>Model</dt><dd>{{ residencyPhase(row.residency) }}</dd>
              <dt>Loading</dt><dd>{{ residencyPolicy(row.residency) }} · {{ residencyCounts(row.residency) }}</dd>
            </template>
            <dt>PID</dt><dd>{{ row.pid }}</dd>
            <dt>Runner</dt><dd>v{{ row.version ?? '-' }}</dd>
          </dl>
          <div class="sd__links">
            <RouterLink :to="{ name: 'instrument', params: { tab: 'activity' }, query: { port: String(port) } }">
              <Icon name="activity" :size="13" /> Requests
            </RouterLink>
            <RouterLink :to="{ name: 'instrument', params: { tab: 'logs' }, query: { port: String(port) } }">
              <Icon name="scroll" :size="13" /> Full logs
            </RouterLink>
          </div>
        </section>

        <!-- config: as deployed, with the live toggles -->
        <section class="sd__card">
          <p class="sd__card-hd">Configuration</p>
          <dl v-if="cfg" class="sd__facts">
            <dt>Model</dt><dd class="sd__mono">{{ cfg.model }}</dd>
            <dt>Context</dt><dd>{{ cfg.max_ctx ? `${fmtCtx(cfg.max_ctx)} tokens` : 'runner default' }}</dd>
            <dt>Concurrency</dt><dd>{{ cfg.max_batch ?? 'runner default' }}</dd>
            <dt>KV cache</dt><dd>{{ cfg.kv_cache_dtype ?? 'auto' }}</dd>
            <dt>GPU</dt>
            <dd>
              <Tooltip :label="cfg.gpu !== null && cfg.gpu !== undefined ? String(cfg.gpu) : ''">
                <span>{{ gpuPinLabel }}</span>
              </Tooltip>
            </dd>
            <dt>System tools</dt>
            <dd class="sd__tools">
              <span v-if="webProvider" class="sd__tool">
                <SearchLogo :provider="webProvider" :size="14" />
                {{ searchLabel(webProvider) }} search
              </span>
              <span v-if="webProvider && mcpLabels" class="sd__toolsep">·</span>
              <span v-if="mcpLabels">{{ mcpLabels }}</span>
              <span v-if="noTools">none</span>
            </dd>
            <dt v-if="cfg.runner_version">Runner pin</dt><dd v-if="cfg.runner_version">{{ cfg.runner_version }}</dd>
          </dl>
          <p v-else class="sd__unknown">
            Adopted process - its launch config is not this manager's to report.
            Start it through the manager to take it over.
          </p>
          <div class="sd__toggles">
            <label class="sd__toggle">
              <Switch :model-value="row.pinned" label="Never auto-stop" @update:model-value="togglePin" />
              Never auto-stop
            </label>
            <label class="sd__toggle">
              <Switch :model-value="boots" label="Start on boot" @update:model-value="toggleBoot" />
              Start on boot
            </label>
            <p v-if="actionErr" class="sd__err">{{ actionErr }}</p>
          </div>
        </section>
      </div>

      <!-- the last card on the page: the logs take the rest of the screen -->
      <section class="sd__card sd__card--wide">
        <p class="sd__card-hd">Logs</p>
        <LogView :target="String(port)" compact fill :fill-offset="44" />
      </section>
    </template>

    <!-- a deploy in flight on this port: show its progress here too -->
    <template v-else-if="deployingHere">
      <h1 class="sd__title">{{ modelLabel(deployingHere.model) }}</h1>
      <p class="sd__starting">
        <Icon name="spinner" :size="14" class="sd__spin" />
        {{ deployingHere.phase === 'starting' ? 'Starting - loading the model' : 'Failed' }}
      </p>
      <p v-if="deployingHere.error" class="sd__err">{{ deployingHere.error }}</p>
      <pre v-if="deployingHere.log.length" class="sd__log">{{ deployingHere.log.join('\n') }}</pre>
    </template>

    <!-- a manager-side download whose queued start owns this port -->
    <template v-else-if="downloadHere">
      <h1 class="sd__title">{{ downloadHere.display }}</h1>
      <p class="sd__starting">
        <Icon name="spinner" :size="14" class="sd__spin" />
        {{
          downloadHere.status.state === 'running'
            ? `Downloading the model · ${downloadHere.total ? Math.round((100 * downloadHere.downloaded) / downloadHere.total) : 0}%`
            : 'Starting - loading the model'
        }}
      </p>
    </template>

    <!-- configured but stopped: the endpoint's file (API key included) is
         kept - this page offers the three honest moves -->
    <template v-else-if="stoppedHere">
      <div class="sd__head">
        <div class="sd__id">
          <VendorLogo v-if="stoppedHere.vendor" :vendor="stoppedHere.vendor" :size="24" />
          <Tooltip :label="`${stoppedHere.model ?? ''} · port ${port}`">
            <h1 class="sd__title">{{ stoppedHere.display ?? (modelLabel(stoppedHere.model) || `server ${port}`) }}</h1>
          </Tooltip>
          <span class="sd__status"><span class="sd__dot sd__dot--off" /> Stopped</span>
        </div>
        <!-- same slot order as the running header: the state toggle
             (Start ⇄ Stop) is always the LAST button -->
        <div class="sd__actions">
          <button class="pk-btn pk-btn--sm" @click="removeEndpoint">
            <Icon name="trash" :size="13" />
            {{ confirmRemove ? 'Click again to remove' : 'Remove' }}
          </button>
          <RouterLink
            class="pk-btn pk-btn--sm"
            :to="{ name: 'server-edit', params: { port: String(port) } }"
          >
            <Icon name="edit" :size="13" /> Edit
          </RouterLink>
          <button class="pk-btn pk-btn--sm pk-btn--primary" :disabled="startingUp" @click="startAgain">
            <Icon name="play" :size="13" /> {{ startingUp ? 'Starting...' : 'Start' }}
          </button>
        </div>
      </div>
      <p class="sd__unknown">
        This endpoint keeps its configuration (servers/{{ port }}.toml - settings and API key
        included). Start brings it back on the same port; Remove deletes the configuration.
      </p>
      <p v-if="removeErr" class="sd__err">{{ removeErr }}</p>
    </template>

    <template v-else-if="fleet.loaded">
      <p class="sd__unknown">
        Nothing runs on port {{ port }}.
        <RouterLink :to="{ name: 'servers' }">Back to Models</RouterLink>
      </p>
    </template>
  </div>
</template>

<style scoped>
/* The system-tools value is a row of parts rather than one string, so the
   search provider can bring its mark. Wraps on a narrow pane. */
.sd__tools {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.sd__tool {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.sd__toolsep {
  color: var(--pk-text-tertiary);
}
.sd {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.sd__crumbs {
  display: flex;
  gap: 8px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  margin-bottom: 8px;
}
.sd__crumbs a {
  color: var(--pk-accent);
  text-decoration: none;
}
.sd__crumbs a:hover {
  text-decoration: underline;
}
.sd__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.sd__id {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.sd__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin: 0;
  /* friendly display name - sans, not mono (the technical id rides the
     tooltip in mono territory) */
  cursor: default;
}
.sd__status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-sm);
}
.sd__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.sd__status--good .sd__dot {
  background: var(--pk-status-success, #4a9);
}
.sd__status--warn {
  color: var(--pk-status-warning);
}
.sd__status--warn .sd__dot {
  background: var(--pk-status-warning);
}
.sd__status--bad {
  color: var(--pk-text-danger);
}
.sd__status--bad .sd__dot {
  background: var(--pk-text-danger);
}
.sd__dot--off {
  background: var(--pk-text-muted);
}
.sd__tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--pk-bg-elevated);
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
.sd__actions {
  display: flex;
  gap: 8px;
  align-items: center;
}
/* NOTE: no .pk-btn--danger override here - the global recipe (red fill,
   inverse text) is the readable one; a scoped color once made it red-on-red */

.sd__endpoint {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 14px 0 18px;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.sd__ep-row {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.sd__ep-lbl {
  min-width: 64px; /* the two rows' values line up */
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
}
.sd__endpoint code {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  user-select: all;
}
.sd__ep-key {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sd__ep-copy {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: none;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  padding: 3px 9px;
  font: inherit;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.sd__ep-copy:hover {
  color: var(--pk-text-primary);
  border-color: var(--pk-accent);
}
.sd__ep-api {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--pk-accent);
  font-size: var(--pk-font-size-sm);
  text-decoration: none;
}
.sd__ep-api:hover {
  text-decoration: underline;
}

.sd__grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
@media (max-width: 860px) {
  .sd__grid {
    grid-template-columns: 1fr;
  }
}
.sd__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 14px;
}
.sd__card--wide {
  margin-top: 14px;
}
.sd__card-hd {
  margin: 0 0 10px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
}
.sd__facts {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 6px 18px;
  margin: 0;
}
.sd__facts dt {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  align-self: center;
}
.sd__facts dd {
  margin: 0;
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-sm);
  font-variant-numeric: tabular-nums;
}
.sd__mono {
  font-family: var(--pk-font-mono);
}
.sd__warn {
  color: var(--pk-status-warning);
}
.sd__links {
  display: flex;
  gap: 16px;
  margin-top: 12px;
}
.sd__links a {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--pk-accent);
  font-size: var(--pk-font-size-sm);
  text-decoration: none;
}
.sd__links a:hover {
  text-decoration: underline;
}
.sd__toggles {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.sd__toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.sd__unknown {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  line-height: 1.5;
}
.sd__unknown a {
  color: var(--pk-accent);
}
.sd__err {
  margin: 4px 0 0;
  color: var(--pk-text-danger);
  font-size: var(--pk-font-size-xs);
  white-space: pre-wrap;
}
.sd__log {
  margin: 0;
  padding: 10px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  white-space: pre-wrap;
  word-break: break-all;
  height: 260px;
  overflow-y: auto;
}
.sd__starting {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.sd__spin {
  animation: sd-spin 0.9s linear infinite;
}
@keyframes sd-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
