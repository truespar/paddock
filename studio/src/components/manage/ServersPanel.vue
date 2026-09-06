<script setup lang="ts">
// The Manager's home: the fleet. One row per server (endpoint = the unit of
// thought here), live status, and the deploy flow with visible progress -
// a spawn takes minutes, so a starting server is a ROW with a live log tail,
// never a spinner into the void. A row is its server's page: click through
// for the detail view (config, edit, tail); deploy is its own route.
import { copyText } from '@/lib/clipboard'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useFleetStore, type FleetRow } from '@/stores/fleet'
import { useDownloadsStore, jobActive, type DownloadJob } from '@/stores/downloads'
import { useRegistryStore } from '@/stores/registry'
import { fmtVram as gb, fmtBytes, fmtEtaShort, fmtRate } from '@/lib/format'
import { modelLabel } from '@/lib/model-name'
import { ROW_BUDGET, reasonOf } from '@/lib/error-text'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import ReadinessNotice from '@/components/manage/ReadinessNotice.vue'
import Dialog from '@/components/ui/Dialog.vue'
import { useReadinessStore } from '@/stores/readiness'
import { useModelsStore } from '@/stores/models'

const fleet = useFleetStore()
// The manager's own version, to compare each runner's build.
const models = useModelsStore()
const readiness = useReadinessStore()
const downloads = useDownloadsStore()
const reg = useRegistryStore()
const router = useRouter()

let release: (() => void) | null = null
onMounted(() => {
  release = fleet.hold()
  void downloads.load()
  void readiness.load()
  // the catalog backs the vendor mark on downloading/starting rows, which
  // exist only client-side and have no manager-supplied vendor of their own
  if (!reg.models.length) void reg.refresh()
})
onUnmounted(() => release?.())

// Manager-side download jobs that will START a server here when done - they
// belong on the fleet screen (the port is already spoken for). Plain catalog
// downloads (no queued start) stay in the header indicator only, and a start
// that settled OK leaves this list - the live runner row takes over.
const dlRows = computed(() =>
  downloads.visible.filter(
    (j) => j.start && !(j.status.state === 'done' && j.start.state?.state === 'ok'),
  ),
)
function dlPct(j: DownloadJob): number {
  return j.total ? Math.round((100 * j.downloaded) / j.total) : 0
}

// ── one table, sorted by port: the port is the endpoint's identity, so a row
// keeps its place through downloading -> starting -> running -> stopped instead
// of jumping around as its state changes. ──────────────────────────────────
type Entry =
  | { kind: 'dl'; port: number; j: DownloadJob }
  | { kind: 'dep'; port: number; d: (typeof fleet.deploying)[number] }
  | { kind: 'live'; port: number; r: FleetRow }
  | { kind: 'stopped'; port: number; c: (typeof fleet.stopped)[number] }
const entries = computed<Entry[]>(() => {
  // STRICTLY one row per port. Every source here - live runner, spawn
  // attempt, download job, config file - is a different VIEW of the same
  // endpoint, and concatenating them rendered one endpoint 2-3 times
  // (measured on the first portable run: a failed pull-start +
  // the stopped config as a pair; paused + failed + stopped as a triplet
  // after a reload). Precedence: live > deploying > download > stopped;
  // dismissing the winner reveals the next state down.
  const byPort = new Map<number, Entry>()
  const put = (e: Entry) => {
    if (!byPort.has(e.port)) byPort.set(e.port, e)
  }
  for (const r of fleet.rows) put({ kind: 'live', port: r.port, r })
  for (const d of fleet.deploying) put({ kind: 'dep', port: d.port, d })
  // newest job first, so a retry outranks the failed attempt it replaces
  for (const j of [...dlRows.value].reverse()) {
    const p = j.start?.port
    if (typeof p === 'number') put({ kind: 'dl', port: p, j })
  }
  for (const c of fleet.stopped) put({ kind: 'stopped', port: c.port, c })
  return [...byPort.values()].sort((a, b) => a.port - b.port)
})

/** Human status labels - Title case, and "Running" instead of API-speak "ok". */
function statusLabel(s: string): string {
  if (s === 'ok') return 'Running'
  if (s === 'draining') return 'Stopping'
  if (s === 'unreachable') return 'Unreachable'
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

// A control that cannot work must not be offered. `blocked` is false until
// the probe answers AND when the answer is fine, so an unanswered probe still
// renders the ordinary first run rather than a stripped one - the rule being
// to hide the start button when there is no usable card. An untested card is
// usable: the notice beside the button says what "untested" costs.
const canRunHere = computed(() => !readiness.blocked)

const hasAnything = computed(
  () =>
    fleet.rows.length > 0 ||
    fleet.deploying.length > 0 ||
    dlRows.value.length > 0 ||
    fleet.stopped.length > 0,
)
// The hero must never FLASH: a download settling empties every source for up
// to one poll before the live row lands, and the page blinked table -> hero ->
// table. First load shows the hero at once; a later
// transition to empty must hold for a beat (outlasting a poll) to count.
const everHadAnything = ref(false)
watch(hasAnything, (v) => (everHadAnything.value = everHadAnything.value || v), {
  immediate: true,
})
const showEmpty = ref(false)
let emptyTimer: number | undefined
watch(
  () => fleet.loaded && !hasAnything.value,
  (empty: boolean) => {
    clearTimeout(emptyTimer)
    if (!empty) {
      showEmpty.value = false
      return
    }
    if (!everHadAnything.value) {
      showEmpty.value = true
      return
    }
    emptyTimer = window.setTimeout(() => (showEmpty.value = true), 1500)
  },
  { immediate: true },
)

// ── stopped endpoints: configured, nothing serving - ready to start again ───
/** Row name for a stopped endpoint; a file the manager can't parse still
 *  gets an honest identity from its port. */
function stoppedName(c: { port: number; model: string | null; display?: string | null }): string {
  return c.display ?? (modelLabel(c.model) || `server ${c.port}`)
}

/** Vendor for a STARTING row, resolved from the catalog by the model string the
 *  spawn was given (a catalog id, an installed name, or a path).
 *
 *  The live and stopped rows get `vendor` from the manager, but a deploying row
 *  exists only client-side and had none - so a starting model rendered with no
 *  mark and grew one the moment it went live, which reads as the row shifting
 *  under you at the least reassuring moment. Same lookup the picker uses. */
function deployVendor(model: string): string | undefined {
  const m = model.toLowerCase()
  return reg.models.find((c) => c.id.toLowerCase() === m)?.vendor
    ?? reg.models.find((c) => m.includes(c.id.toLowerCase()))?.vendor
    ?? undefined
}
const starting = ref<Record<number, boolean>>({})
async function startStopped(c: { port: number; model: string | null; display?: string | null }): Promise<void> {
  starting.value = { ...starting.value, [c.port]: true }
  try {
    await fleet.startConfigured(c.port, c.display ?? c.model ?? String(c.port))
  } finally {
    starting.value = { ...starting.value, [c.port]: false }
  }
}
// Remove DELETES the endpoint's configuration file, which is not recoverable.
//
// This used to be a click-twice arm-and-fire on the icon button, signalled only
// by a tint change and a tooltip, on a 3 s timer. Nothing said a second click
// was coming, so it read as a dead button - and an undiscoverable gesture is a
// bad way to guard a destructive act anyway. It is a real confirmation now,
// naming the endpoint and what is lost, per the app's alertdialog pattern.
const removeAsk = ref<{ port: number; name: string } | null>(null)
const removing = ref(false)
function askRemove(c: { port: number; model: string | null; display?: string | null }): void {
  removeAsk.value = { port: c.port, name: stoppedName(c) }
}
async function confirmRemoveNow(): Promise<void> {
  const ask = removeAsk.value
  if (!ask) return
  removing.value = true
  try {
    await fleet.removeConfigured(ask.port)
    removeAsk.value = null
  } catch {
    /* refresh shows the row still standing; the manager refused */
  } finally {
    removing.value = false
  }
}

function toDetail(port: number): void {
  void router.push({ name: 'server-detail', params: { port: String(port) } })
}

function uptime(s: number | null): string {
  if (s === null) return '-'
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

function elapsed(startedAt: number): string {
  const s = Math.floor((now.value - startedAt) / 1000)
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
}
// a ticking clock for the starting rows' elapsed label
const now = ref(Date.now())
// The 1s clock exists only for the "Starting · 12s" elapsed label. Ticking
// unconditionally re-rendered the whole fleet table every second forever.
let tick: number | undefined
watch(
  () => fleet.deploying.length > 0,
  (starting) => {
    clearInterval(tick)
    tick = starting ? window.setInterval(() => (now.value = Date.now()), 1000) : undefined
  },
  { immediate: true },
)
onUnmounted(() => clearInterval(tick))

function statusTone(r: FleetRow): string {
  if (r.status === 'ok') return 'good'
  if (r.status === 'draining') return 'warn'
  return 'bad'
}

const copiedPort = ref<number | null>(null)
async function copyEndpoint(r: FleetRow): Promise<void> {
  try {
    await copyText(`${r.endpoint}/v1`)
    copiedPort.value = r.port
    setTimeout(() => (copiedPort.value = null), 1400)
  } catch {
    /* clipboard blocked */
  }
}

const stopping = ref<Record<number, boolean>>({})
async function stop(r: FleetRow): Promise<void> {
  stopping.value = { ...stopping.value, [r.port]: true }
  try {
    await fleet.stop(r.port)
  } finally {
    stopping.value = { ...stopping.value, [r.port]: false }
  }
}
</script>

<template>
  <div class="srv">
    <ReadinessNotice v-if="!showEmpty" class="srv__verdict" />
    <div v-if="hasAnything" class="srv__head">
      <div>
        <h1 class="srv__title">Models</h1>
        <p class="srv__lead">
          Each running model is its own server on one port - the endpoint your tools and agents call.
        </p>
      </div>
      <RouterLink class="pk-btn pk-btn--primary" :to="{ name: 'server-new' }">
        <Icon name="play" :size="14" /> Start a model
      </RouterLink>
    </div>

    <div v-if="fleet.error" class="srv__error">{{ fleet.error }}</div>

    <!-- the machine-freeze emergency: ledgers > card. Loud, with the fix. -->
    <div v-if="fleet.overcommit" class="srv__overcommit">
      <Icon name="alert-triangle" :size="16" />
      <div>
        <strong>VRAM overcommitted:</strong>
        your models hold {{ gb(fleet.overcommit.committed) }} on a
        {{ gb(fleet.overcommit.device_total) }} card. Windows is paging GPU memory into
        system RAM - the whole machine can freeze. Stop a model now.
      </div>
    </div>

    <!-- the fleet -->
    <div v-if="hasAnything" class="srv__tablewrap">
      <table class="srv__table">
        <colgroup>
          <col class="col-port" />
          <col />
          <col class="col-status" />
          <col class="col-vram" />
          <col class="col-uptime" />
          <col class="col-ver" />
          <col class="col-boot" />
          <col class="col-actions" />
        </colgroup>
        <thead>
          <tr>
            <th>Port</th>
            <th>Model</th>
            <th>Status</th>
            <th>VRAM</th>
            <th>Uptime</th>
            <th class="c-ver">Build</th>
            <th class="c-boot">Boot</th>
            <th class="c-actions"></th>
          </tr>
        </thead>
        <tbody>
          <!-- One list sorted by port: a row keeps its place while its state
               changes (downloading -> starting -> running -> stopped). Actions
               sit in FIXED slots on every row - Edit · Copy · Start/Stop
               (swap in place) · Remove - with a slot disabled (not hidden)
               when its action doesn't apply, so nothing shifts. -->
          <template v-for="en in entries" :key="`${en.kind}-${en.port}`">
            <!-- a manager-side download with a queued start: the port is
                 spoken for; progress + cancel/resume right here -->
            <tr v-if="en.kind === 'dl'" class="srow srow--starting">
              <td class="c-port">{{ en.port }}</td>
              <td class="c-model">
                <span class="c-model__name">
                  <VendorLogo
                    v-if="deployVendor(en.j.model)"
                    :vendor="deployVendor(en.j.model)!"
                    :size="16"
                    class="c-model__logo"
                  />
                  <span class="c-model__id">{{ en.j.display }}</span>
                </span>
              </td>
              <td colspan="5" class="c-dlcell">
                <div class="c-dl">
                  <span v-if="en.j.status.state === 'running'" class="st st--starting">
                    <Icon name="spinner" :size="13" class="spin" />
                    Downloading · {{ dlPct(en.j) }}%
                  </span>
                  <span
                    v-else-if="en.j.status.state === 'done' && (en.j.start?.state?.state === 'queued' || en.j.start?.state?.state === 'starting')"
                    class="st st--starting"
                  >
                    <Icon name="spinner" :size="13" class="spin" />
                    Starting
                  </span>
                  <span v-else-if="en.j.status.state === 'cancelled'" class="st st--warn">
                    <span class="st__dot" /> Paused
                  </span>
                  <span v-else-if="en.j.status.state === 'error' || en.j.start?.state?.state === 'error'" class="st st--bad">
                    <span class="st__dot" /> Failed
                  </span>
                  <span v-else class="st st--good"><span class="st__dot" /> Done</span>

                  <span v-if="en.j.status.state === 'running'" class="c-dl__info">
                    {{ gb(en.j.downloaded) }} of {{ gb(en.j.total) }}
                    <template v-if="downloads.rateOf(en.j.id)">
                      · {{ fmtRate(downloads.rateOf(en.j.id)!.bps) }}
                      <template v-if="downloads.rateOf(en.j.id)!.etaS != null">
                        · {{ fmtEtaShort(downloads.rateOf(en.j.id)!.etaS!) }}
                      </template>
                    </template>
                  </span>
                  <span v-else-if="en.j.status.state === 'cancelled'" class="c-dl__info">
                    {{ fmtBytes(Math.max(0, en.j.total - en.j.downloaded)) }} left - Resume continues where it stopped.
                  </span>
                  <Tooltip
                    v-else-if="en.j.status.state === 'error'"
                    :label="en.j.status.message"
                  >
                    <span class="c-dl__info c-errline">
                      {{ reasonOf(en.j.status.message, ROW_BUDGET) || 'Download failed' }}
                    </span>
                  </Tooltip>
                  <Tooltip
                    v-else-if="en.j.start?.state?.state === 'error'"
                    :label="en.j.start?.state?.message"
                  >
                    <span class="c-dl__info c-errline">
                      {{ reasonOf(en.j.start?.state?.message, ROW_BUDGET) || 'Failed to start' }}
                    </span>
                  </Tooltip>
                  <span v-else class="c-dl__info">Loading the model - this can take a minute or two.</span>
                </div>
              </td>
              <td class="c-actions" @click.stop>
                <button
                  v-if="en.j.status.state === 'running'"
                  class="pk-btn pk-btn--sm pk-btn--ghost"
                  @click="downloads.cancel(en.j.id)"
                >
                  Cancel
                </button>
                <button
                  v-if="en.j.status.state === 'cancelled' || en.j.status.state === 'error'"
                  class="pk-btn pk-btn--sm pk-btn--ghost"
                  @click="downloads.resume(en.j.id)"
                >
                  Resume
                </button>
                <button
                  v-if="!jobActive(en.j)"
                  class="pk-btn pk-btn--sm pk-btn--ghost"
                  @click="downloads.dismiss(en.j.id)"
                >
                  Dismiss
                </button>
              </td>
            </tr>

            <!-- a spawn in flight: spinner, elapsed, a plain sentence. No
                 logs here: the outcome arrives as a
                 toast, the detail page has the tail. -->
            <tr
              v-else-if="en.kind === 'dep'"
              class="srow srow--starting srow--click"
              @click="toDetail(en.port)"
            >
              <td class="c-port">{{ en.port }}</td>
              <td class="c-model">
                <span class="c-model__name">
                  <VendorLogo
                    v-if="deployVendor(en.d.model)"
                    :vendor="deployVendor(en.d.model)!"
                    :size="16"
                    class="c-model__logo"
                  />
                  <span class="c-model__id">{{ modelLabel(en.d.model) }}</span>
                </span>
              </td>
              <td>
                <span v-if="en.d.phase === 'starting'" class="st st--starting">
                  <Icon name="spinner" :size="13" class="spin" />
                  Starting · {{ elapsed(en.d.startedAt) }}
                </span>
                <span v-else class="st st--bad"><span class="st__dot" /> Failed</span>
              </td>
              <td :colspan="en.d.phase === 'starting' ? 5 : 4" class="c-dim">
                <span v-if="en.d.phase === 'starting'">
                  Loading the model - this can take a minute or two.
                </span>
                <Tooltip v-else :label="en.d.error">
                  <span class="c-errrow">
                    <span class="c-errline">{{ reasonOf(en.d.error, ROW_BUDGET) || 'Failed to start' }}</span>
                    <span class="c-errmore">- click for details</span>
                  </span>
                </Tooltip>
              </td>
              <td v-if="en.d.phase === 'failed'" class="c-actions" @click.stop>
                <button
                  class="pk-btn pk-btn--sm pk-btn--ghost"
                  @click="fleet.dismissFailed(en.port)"
                >
                  Dismiss
                </button>
              </td>
            </tr>

            <!-- a live runner -->
            <tr v-else-if="en.kind === 'live'" class="srow srow--click" @click="toDetail(en.port)">
              <td class="c-port">{{ en.port }}</td>
              <td class="c-model">
                <Tooltip :label="en.r.model ?? en.r.embedder ?? en.r.asr ?? en.r.aligner ?? ''">
                  <span class="c-model__name">
                    <VendorLogo v-if="en.r.vendor" :vendor="en.r.vendor" :size="16" class="c-model__logo" />
                    <span class="c-model__id">{{ en.r.display ?? en.r.model ?? en.r.embedder ?? en.r.asr ?? en.r.aligner ?? '-' }}</span>
                    <span v-if="en.r.spec" class="c-model__spec">{{ en.r.spec }}</span>
                  </span>
                </Tooltip>
                <span v-if="en.r.embedder && en.r.model" class="c-model__extra">+ {{ en.r.embedder }}</span>
              </td>
              <td>
                <span class="st" :class="`st--${statusTone(en.r)}`">
                  <span class="st__dot" /> {{ statusLabel(en.r.status) }}
                </span>
              </td>
              <td class="c-num">
                <Tooltip
                  v-if="en.r.vram?.self_mem"
                  :label="`Engine count ${gb(en.r.vram.self_mem)}${en.r.vram.nvml_mem ? ` · OS view ${gb(en.r.vram.nvml_mem)}` : ''}${en.r.vram.anomaly ? ' · more than expected between the two' : ''}`"
                >
                  <span :class="{ 'c-warn': en.r.vram.anomaly }">{{ gb(en.r.vram.self_mem) }}</span>
                </Tooltip>
                <span v-else>-</span>
              </td>
              <td class="c-num">{{ uptime(en.r.uptime_s) }}</td>
              <!-- Runner build. Three states, and only one of them is a
                   problem:
                     matches  - silent, the common case
                     pinned   - somebody chose this version; rollback is a
                                supported move, so it is a fact, not a warning
                     older    - serving an image from before the last package
                                refresh. The new runner is already on disk; the
                                action is a restart, NOT a download. -->
              <td class="c-ver">
                <Tooltip
                  v-if="en.r.config?.runner_version"
                  :label="`Pinned to runner ${en.r.config.runner_version}`"
                >
                  <span class="sp__ver">{{ en.r.config.runner_version }} &middot; pinned</span>
                </Tooltip>
                <Tooltip
                  v-else-if="en.r.version && models.serverVersion && en.r.version !== models.serverVersion"
                  :label="`Serving an older build than paddock ${models.serverVersion}. Restart this model to pick it up - there is nothing to download.`"
                >
                  <span class="sp__ver sp__ver--old">{{ en.r.version }}</span>
                </Tooltip>
                <span v-else-if="en.r.version" class="sp__ver">{{ en.r.version }}</span>
                <span v-else>-</span>
              </td>
              <td class="c-boot">
                <Tooltip
                  :label="fleet.bootPorts.has(en.port) ? 'Starts on boot' : 'Won\'t start on boot'"
                >
                  <Icon :name="fleet.bootPorts.has(en.port) ? 'check' : 'x'" :size="14"
                    :class="fleet.bootPorts.has(en.port) ? 'c-good' : 'c-dim'" />
                </Tooltip>
              </td>
              <td class="c-actions" @click.stop>
                <Tooltip label="Edit settings (restarts on save)">
                  <button class="act" @click="router.push({ name: 'server-edit', params: { port: String(en.port) } })">
                    <Icon name="edit" :size="15" />
                  </button>
                </Tooltip>
                <Tooltip :label="copiedPort === en.port ? 'Copied' : `Copy base URL - ${en.r.endpoint}/v1`">
                  <button class="act" @click="copyEndpoint(en.r)">
                    <Icon :name="copiedPort === en.port ? 'check' : 'copy'" :size="15" />
                  </button>
                </Tooltip>
                <Tooltip label="Stop - finishes in-flight requests first; the endpoint stays configured">
                  <button class="act act--danger" :disabled="stopping[en.port]" @click="stop(en.r)">
                    <Icon name="stop" :size="15" />
                  </button>
                </Tooltip>
                <Tooltip label="Remove - stop the model first">
                  <button class="act act--off" aria-disabled="true">
                    <Icon name="trash" :size="15" />
                  </button>
                </Tooltip>
              </td>
            </tr>

            <!-- a stopped endpoint: the configuration is kept
                 (servers/<port>.toml, API key included) - ready to start
                 again; Remove is the explicit way to let one go -->
            <tr
              v-else
              class="srow srow--click srow--stopped"
              @click="toDetail(en.port)"
            >
              <td class="c-port">{{ en.port }}</td>
              <td class="c-model">
                <Tooltip :label="en.c.model ?? ''">
                  <span class="c-model__name">
                    <VendorLogo v-if="en.c.vendor" :vendor="en.c.vendor" :size="16" class="c-model__logo" />
                    <span class="c-model__id">{{ stoppedName(en.c) }}</span>
                    <span v-if="en.c.spec" class="c-model__spec">{{ en.c.spec }}</span>
                  </span>
                </Tooltip>
              </td>
              <td>
                <span class="st st--stopped"><span class="st__dot" /> Stopped</span>
              </td>
              <td class="c-num">-</td>
              <td class="c-num">-</td>
              <td class="c-ver"></td>
              <td class="c-boot">
                <Tooltip
                  :label="fleet.bootPorts.has(en.port) ? 'Starts on boot' : 'Won\'t start on boot'"
                >
                  <Icon :name="fleet.bootPorts.has(en.port) ? 'check' : 'x'" :size="14"
                    :class="fleet.bootPorts.has(en.port) ? 'c-good' : 'c-dim'" />
                </Tooltip>
              </td>
              <td class="c-actions" @click.stop>
                <Tooltip label="Edit settings">
                  <button class="act" @click="router.push({ name: 'server-edit', params: { port: String(en.port) } })">
                    <Icon name="edit" :size="15" />
                  </button>
                </Tooltip>
                <Tooltip label="Copy URL - nothing serves this port right now">
                  <button class="act act--off" aria-disabled="true">
                    <Icon name="copy" :size="15" />
                  </button>
                </Tooltip>
                <Tooltip label="Start again - same port, same settings">
                  <button class="act act--start" :disabled="starting[en.port]" @click="startStopped(en.c)">
                    <Icon name="play" :size="15" />
                  </button>
                </Tooltip>
                <Tooltip label="Remove this endpoint">
                  <button class="act act--danger" @click="askRemove(en.c)">
                    <Icon name="trash" :size="15" />
                  </button>
                </Tooltip>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>

    <!-- the first-run page: one clean CTA, and it teaches the split - the
         Manager runs models, the Studio talks to them -->
    <div v-else-if="showEmpty" class="srv__first" :class="{ 'srv__first--split': !!readiness.notice }">
      <div class="srv__pair">
      <div class="srv__empty">
      <Icon name="server" :size="56" class="srv__empty-icon" />
      <p class="srv__empty-title">Run AI models on this machine</p>
      <p class="srv__empty-sub">
        Each running model is its own server on one port - the endpoint your tools and agents call.
      </p>
      <div class="srv__areas">
        <div class="srv__area srv__area--here">
          <p class="srv__area-hd"><Icon name="server" :size="14" /> Manager</p>
          <p v-if="canRunHere" class="srv__area-txt">
            Starts and monitors models - every request's timings, the GPU, the logs.
            <strong>You are here.</strong>
          </p>
          <p v-else class="srv__area-txt">
            Starts and monitors models on this computer. <strong>You are here.</strong>
          </p>
        </div>
        <div class="srv__area">
          <p class="srv__area-hd"><Icon name="message-square" :size="14" /> Studio</p>
          <p class="srv__area-txt">
            Chat and compare your models and cloud models side by side.<template v-if="canRunHere">
              It needs a running model first - which is what the button below is for.</template>
          </p>
        </div>
      </div>
      <template v-if="canRunHere">
        <RouterLink class="pk-btn pk-btn--primary srv__cta" :to="{ name: 'server-new' }">
          <Icon name="play" :size="15" /> Start your first model
        </RouterLink>
        <p class="srv__empty-cli">
          or from a terminal: <code>paddock serve &lt;model&gt;</code>
        </p>
      </template>
    </div>
      <ReadinessNotice v-if="readiness.notice" class="srv__verdict" />
      </div>
    </div>

    <!-- Removing an endpoint deletes its config file and cannot be undone, so
         it asks - naming the endpoint, and saying what is NOT lost (the model
         weights) so the answer is obvious rather than nervous. Outside the
         v-if/v-else chain above: a sibling between those branches severs their
         adjacency and the template stops compiling. -->
    <Dialog
      :open="!!removeAsk"
      role="alertdialog"
      icon="trash"
      danger
      title="Remove this endpoint?"
      :busy="removing"
      @close="removeAsk = null"
    >
      <p v-if="removeAsk" class="srv__ask">
        <strong>{{ removeAsk.name }}</strong> on port {{ removeAsk.port }} will be deleted,
        along with its settings file. The downloaded model stays on disk - you can start a
        new endpoint for it any time.
      </p>
      <template #footer>
        <button class="pk-btn pk-btn--ghost" :disabled="removing" @click="removeAsk = null">
          Cancel
        </button>
        <button class="pk-btn pk-btn--danger" :disabled="removing" @click="confirmRemoveNow">
          {{ removing ? 'Removing...' : 'Remove endpoint' }}
        </button>
      </template>
    </Dialog>
  </div>
</template>

<style scoped>
.sp__ver {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
/* The only one that draws the eye. A pinned or matching build is a fact; an
   older one is the single case with something to do about it. */
.sp__ver--old {
  color: var(--pk-status-warning);
}

.srv {
  /* Wider than the shared panel width deliberately: this is a six-column
     TABLE, not a form, and under table-layout:fixed every extra pixel lands
     in the elastic model column - which the spec chip was starving at 960
     Narrow windows still shrink via width:100%. */
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  /* column flex so the first-run hero can flex-fill the content viewport
     instead of guessing a vh; the table states are unaffected (column-flex
     children stretch like blocks). align-self overrides the shell content's
     align-items: flex-start, which otherwise shrinks us to content height. */
  display: flex;
  flex-direction: column;
  align-self: stretch;
  min-height: 100%;
}
.srv__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.srv__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.srv__lead {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.srv__error {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
  border-radius: var(--pk-radius-md);
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: var(--pk-font-size-sm);
}
.srv__overcommit {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
  border: 1px solid var(--pk-status-error);
  border-radius: var(--pk-radius-md);
  padding: 12px 14px;
  margin-bottom: 12px;
  font-size: var(--pk-font-size-sm);
  line-height: 1.5;
}
.srv__overcommit svg {
  flex: none;
  margin-top: 2px;
}

.srv__tablewrap {
  overflow-x: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  /* the table is a CARD - see the Instrument twin; without a surface the
     light theme read as one grey slab */
  background: var(--pk-bg-surface);
}
.srv__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
  /* FIXED columns: a downloading row's status/detail change width every SSE
     frame (33% -> 34%, 9.4 -> 104.2 MB/s) and auto layout re-measured the
     table each time - the Model column wrapped and jittered for the whole
     download. With fixed columns nothing can move; text truncates in place. */
  table-layout: fixed;
}
.col-port {
  width: 62px;
}
.col-status {
  width: 172px;
}
.col-vram {
  width: 88px;
}
.col-uptime {
  width: 80px;
}
.col-ver {
  /* Under table-layout: fixed the COLGROUP is what sizes columns - a <th>
     without a matching <col> gets no allocation and the whole table stops
     filling its container. Adding the header cell without this pair is
     exactly how that happened. */
  width: 84px;
}
.col-boot {
  width: 56px;
}
.col-actions {
  width: 152px;
}
.srv__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 10px 14px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
  background: var(--pk-bg-surface);
}
.srv__table td {
  padding: 11px 14px;
  border-top: 1px solid var(--pk-border-default);
  vertical-align: middle;
}
.srow:hover td {
  background: var(--pk-bg-hover);
}
.srow--click {
  cursor: pointer;
}
.c-port {
  font-family: var(--pk-font-mono);
  font-weight: 600;
  color: var(--pk-text-primary);
  white-space: nowrap;
}
.c-model__name {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.c-model__logo {
  flex: none;
}
.c-model__id {
  font-weight: 600;
  color: var(--pk-text-primary);
  /* fixed table layout: a long name truncates, never wraps the row taller */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.c-model__extra {
  margin-left: 8px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.c-model__tag {
  margin-left: 8px;
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--pk-bg-elevated);
  color: var(--pk-text-muted);
  font-size: 10px;
}
.c-num {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  color: var(--pk-text-secondary);
}
.c-warn {
  color: var(--pk-status-warning);
}
.c-good {
  color: var(--pk-status-success, #4a9);
}
.c-dim {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
/* the downloading row's one wide cell (spans Status..Boot, ~396px): the
   speed and time-left must always be VISIBLE - the earlier per-column split
   left them a 224px sliver that ellipsized exactly the numbers the row
   exists to show. Width-stable digits; only a failure message may truncate
   (its full text rides the title). */
.c-dlcell {
  overflow: hidden;
}
.c-dl {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.c-dl .st {
  flex: none;
}
.c-dl__info {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.c-boot {
  text-align: center;
}

.st {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  white-space: nowrap;
}
.st__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: none;
}
.st--good .st__dot {
  background: var(--pk-status-success, #4a9);
}
.st--warn {
  color: var(--pk-status-warning);
}
.st--warn .st__dot {
  background: var(--pk-status-warning);
}
.st--bad {
  color: var(--pk-text-danger);
}
.st--bad .st__dot {
  background: var(--pk-text-danger);
}
.st--starting {
  color: var(--pk-text-secondary);
}
.st--stopped {
  color: var(--pk-text-muted);
}
.st--stopped .st__dot {
  background: var(--pk-text-muted);
}
.srow--stopped .c-model__id,
.srow--stopped .c-port {
  color: var(--pk-text-secondary);
}
/* the remove confirmation's copy */
.srv__ask {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.srv__ask strong {
  color: var(--pk-text-primary);
}
.spin {
  animation: srv-spin 0.9s linear infinite;
}
@keyframes srv-spin {
  to {
    transform: rotate(360deg);
  }
}

/* starting/failed rows - status only, never logs */
.srow--starting td {
  background: var(--pk-accent-subtle);
}
/* The failure line, bounded by its CELL. It used to cap at a hard-coded 480px
   in a `table-layout: fixed` table whose failure cell is ~396px, so a long
   refusal simply overflowed the column. A pixel guess cannot
   track a fixed column; flex + `min-width: 0` does, at any width. */
.c-errrow {
  display: flex;
  align-items: baseline;
  gap: 4px;
  min-width: 0;
}
/* Both sets of properties deliberately: this span is a flex item inside
   `.c-errrow` and a standalone inline-block inside `.c-dl__info`. A flex
   container blockifies its items, so `inline-block` is simply ignored in the
   first case - and `max-width: 100%` is what bounds it in the second. */
.c-errline {
  color: var(--pk-text-danger);
  display: inline-block;
  vertical-align: bottom;
  flex: 0 1 auto;
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* "click for details" is the AFFORDANCE, so it must not be what the ellipsis
   eats - it lived inside the truncating span, which meant the longer the error,
   the less chance you were told there was more of it. */
.c-errmore {
  flex: none;
  white-space: nowrap;
}

.c-actions {
  text-align: right;
  white-space: nowrap;
}
.act {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.act:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
.act--on {
  color: var(--pk-accent);
}
.act--danger:hover {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
}
/* a slot whose action doesn't apply right now: visible (columns align),
   clearly inert, tooltip says why */
.act--off {
  opacity: 0.35;
  cursor: default;
}
.act--off:hover {
  background: transparent;
  color: var(--pk-text-muted);
}
.act--start {
  color: var(--pk-status-success, #4a9);
}
.act--start:hover {
  background: var(--pk-accent-subtle);
  color: var(--pk-status-success, #4a9);
}

/* First run, one column. On a machine that can serve, the hero owns the page
   exactly as it always did. When there is a verdict it goes UNDERNEATH, on the
   same 640px band as the Manager/Studio cards so its edges line up with
   theirs, and the whole stack centres as a unit (after a
   two-column try that could not be made to align without gluing the page to
   the top). */
.srv__first {
  display: flex;
  flex: 1;
  min-height: 0;
}
.srv__pair {
  display: flex;
  flex: 1;
  width: 100%;
  min-height: 0;
}
.srv__first--split .srv__pair {
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 18px;
}
/* the hero stops flex-filling: the STACK is what centres now */
.srv__first--split .srv__empty {
  flex: 0 0 auto;
  padding-bottom: 0;
}
.srv__first--split .srv__verdict {
  width: 100%;
  max-width: 640px;
  margin-bottom: 0;
}

/* the verdict above a populated fleet gets breathing room from the table */
.srv__verdict {
  margin-bottom: 20px;
}

/* empty fleet - on a surface card, not bare canvas */
.srv__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  /* first run has exactly one thing to do - the hero is the page: it
     flex-fills the content viewport, centered, borderless (a border makes
     it a widget in a page; there is no page yet). */
  justify-content: center;
  flex: 1;
  gap: 12px;
  padding: 24px;
  text-align: center;
  /* No background: when the border went, the surface fill stayed behind and
     painted a raw white slab on the base background. The
     hero sits on the page itself; only the cards below carry a surface. */
}
.srv__empty-icon {
  color: var(--pk-text-muted);
}
.srv__empty-title {
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.srv__empty-sub {
  margin: 0 0 10px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-md, 0.95rem);
  max-width: 46ch;
}
.srv__cta {
  font-size: 1rem;
  padding: 12px 24px;
}
.srv__areas {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  max-width: 640px;
  width: 100%;
  margin: 6px 0 10px;
  text-align: left;
}
@media (max-width: 700px) {
  .srv__areas {
    grid-template-columns: 1fr;
  }
}
.srv__area {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  /* surface-on-base, the app's card pairing (these were base-on-surface back
     when the hero itself was a surface card) */
  background: var(--pk-bg-surface);
  padding: 14px 16px;
}
.srv__area--here {
  border-color: var(--pk-accent);
}
.srv__area-hd {
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 0 0 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-primary);
}
.srv__area-txt {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.srv__cta {
  margin-top: 4px;
  padding: 9px 22px;
  font-size: var(--pk-font-size-base);
}
.srv__empty-cli {
  margin: 4px 0 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
.srv__empty-cli code {
  font-family: var(--pk-font-mono);
  background: var(--pk-bg-inset);
  padding: 2px 6px;
  border-radius: var(--pk-radius-sm);
}
.c-model__spec {
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  font-size: 10px;
  white-space: nowrap;
}
</style>
