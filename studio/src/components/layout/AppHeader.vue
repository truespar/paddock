<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useTheme } from '@/composables/useTheme'
import { useModelsStore } from '@/stores/models'
import { useTelemetryStore } from '@/stores/telemetry'
import { useDownloadsStore, jobActive, type DownloadJob } from '@/stores/downloads'
import { useReadinessStore } from '@/stores/readiness'
import { useFleetStore } from '@/stores/fleet'
import { useUpdatesStore } from '@/stores/updates'
import { useReadsStore } from '@/stores/reads'
import { useTablesStore } from '@/stores/tables'
import { predictorLabel } from '@/lib/tables'
import { modelLabel } from '@/lib/model-name'
import { selectStudioModel } from '@/lib/select-model'
import { studioModelHeader } from '@/lib/studio-model-header'
import { fmtBytes, fmtEta, fmtRate } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import FeedbackDialog from '@/components/feedback/FeedbackDialog.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import Select from '@/components/ui/Select.vue'
import Popover from '@/components/ui/Popover.vue'
import ToggleGroup from '@/components/ui/ToggleGroup.vue'
import ToggleGroupItem from '@/components/ui/ToggleGroupItem.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import BackendStatus from '@/components/layout/BackendStatus.vue'

const { theme, toggleTheme } = useTheme()
const models = useModelsStore()
const tele = useTelemetryStore()
const downloads = useDownloadsStore()
// No card, no GPU metrics button: an instrument with nothing to sample is a
// dead control, and offering it is how a GPU-less machine looked normal.
// `hasCard` is true until the probe answers, so the button never
// flickers away on a machine that does have one.
const readiness = useReadinessStore()
// Runner-vs-manager version skew, surfaced in the header.
const fleet = useFleetStore()
const route = useRoute()
const router = useRouter()

// Feedback lives in the HEADER rather than either area's nav, because it
// belongs to neither: it is about the product, and the Manager/Studio split is
// about what you are doing to models. It also wants to sit next to the version
// chip - "which build is this" and "this build is broken" are one gesture.
const feedbackOpen = ref(false)

// A new paddock, surfaced here because the header is the one surface every
// panel shares. The full card (notes, download) stays in Manage -> Settings,
// but until 0.1.1 that card was also the only place the check ever rendered -
// and it only fetches when opened, so a release was invisible to anyone who
// never visited Settings (found the day 0.1.1 shipped: a deliberate look for
// the notice missed it). The chip routes there; it does not download.
const updates = useUpdatesStore()
const updateReady = computed(
  () => (updates.info?.state === 'available' && updates.info.latest) || null,
)
let updTimer: number | undefined

// restore manager-side download state on load - a refresh mid-download must
// keep showing the indicator (the jobs live on the manager, not in this tab)
onMounted(() => {
  void downloads.load()
  void readiness.load()
  void updates.refresh()
  // The manager caches the release check for an hour, so this is a cheap
  // local read - the tab just has to keep looking at it.
  updTimer = window.setInterval(() => void updates.refresh(), 15 * 60_000)
})
onUnmounted(() => clearInterval(updTimer))

/** One plain sentence per download row. */
function jobLine(j: DownloadJob): string {
  const st = j.start?.state?.state
  if (j.status.state === 'running') {
    const pct = j.total ? Math.round((100 * j.downloaded) / j.total) : 0
    // speed + time left ride the popover; the chip stays
    // a bare percentage
    const rate = downloads.rateOf(j.id)
    const speed = rate ? ` · ${fmtRate(rate.bps)}` : ''
    const eta = rate?.etaS != null ? ` · ${fmtEta(rate.etaS)}` : ''
    return `${pct}% · ${fmtBytes(Math.max(0, j.total - j.downloaded))} left${speed}${eta}`
  }
  if (j.status.state === 'cancelled') return 'paused - resume to continue'
  if (j.status.state === 'error') return j.status.message?.split('\n')[0] ?? 'failed'
  if (st === 'queued' || st === 'starting') return 'downloaded - starting the model'
  if (st === 'ok') return `running on port ${j.start?.state?.port ?? j.start?.port}`
  if (st === 'error') return `start failed: ${j.start?.state?.message?.split('\n')[0] ?? ''}`
  return 'downloaded'
}
function jobPct(j: DownloadJob): number {
  return j.total ? Math.min(100, (100 * j.downloaded) / j.total) : 0
}
const dlBusy = computed(() => downloads.active.length > 0)
const dlPct = computed(() => {
  const a = downloads.aggregate
  return a.total ? Math.round((100 * a.done) / a.total) : 0
})

// The two areas, never mixed: Manager (servers/models/instrument) and Studio
// (chat/tools/settings). The header owns the switch; each side keeps its own
// sub-navigation in the rail.
const area = computed<'manage' | 'studio'>(() =>
  route.path.startsWith('/studio') ? 'studio' : 'manage',
)
// The area is derived from the route, so this is a one-way door: ToggleGroup
// reports a string, we navigate, and `area` recomputes from where we land.
function switchArea(to: string): void {
  if (to === area.value || (to !== 'manage' && to !== 'studio')) return
  void router.push({ name: to === 'manage' ? 'servers' : 'home' })
}

// Shared with the native title bar. Compare names are informational here;
// the composer's Compare picker owns the lane set on both clients.
const modelHeader = computed(studioModelHeader)
const currentModel = computed(() => modelHeader.value.currentModel)
const comparing = computed(() => modelHeader.value.comparing)
const pickerOptions = computed(() => modelHeader.value.pickerOptions)
const compareLanes = computed(() => modelHeader.value.compareLanes)
const specLabel = computed(() => modelHeader.value.specLabel)
const isVision = computed(() => modelHeader.value.isVision)
const soleEncoder = computed(() => modelHeader.value.soleEncoder)
const modelSel = computed<string | number>({
  get: () => currentModel.value,
  set: (v) => selectStudioModel(String(v)),
})

// On the Reads page the picker chooses the READER: the running models that
// answer /v1/systemone, which the page sends to. The chat's model has no
// part in a read, so the page never shows it - one picker, in one place, as
// on the chat page.
const readsStore = useReadsStore()
const onReads = computed(() => route.name === 'reads')
const readerOptions = computed(() =>
  models.readers.map((m) => ({
    value: m.port ?? 0,
    label: m.display ?? modelLabel(m.id),
    hint: `port ${m.port}`,
    vendor: m.vendor,
    title: m.id,
  })),
)
const readerSel = computed<string | number>({
  get: () => readsStore.readerPort,
  set: (v) => (readsStore.readerPort = Number(v)),
})

// Tables, the same way: the picker chooses the PREDICTOR, the running tabular
// endpoints the page sends its table to, and the chat's model stays out of it.
const tablesStore = useTablesStore()
const onTables = computed(() => route.name === 'tables')
const predictorOptions = computed(() =>
  models.models
    .filter((m) => m.kind === 'tabular')
    .map((m) => ({
      value: m.port ?? 0,
      label: predictorLabel(m),
      hint: `port ${m.port}`,
      vendor: m.vendor,
      title: m.id,
    })),
)
const predictorSel = computed<string | number>({
  get: () => tablesStore.port,
  set: (v) => (tablesStore.port = Number(v)),
})
</script>

<template>
  <header class="header">
    <div class="header__ctx">
      <span class="header__brand">Paddock</span>
      <ToggleGroup
        :model-value="area"
        class="header__areas"
        label="Area"
        @update:model-value="switchArea"
      >
        <ToggleGroupItem value="manage" class="header__area">
          <Icon name="server" :size="13" /> Manager
        </ToggleGroupItem>
        <ToggleGroupItem value="studio" class="header__area">
          <Icon name="message-square" :size="13" /> Studio
        </ToggleGroupItem>
      </ToggleGroup>
      <!-- the model picker belongs to the Studio (conversation-scoped) -->
      <!-- no decorative box icon here: the vendor mark inside the picker
           already identifies the model, and the cube glyph read as a second
           caret -->
      <template v-if="onReads">
        <template v-if="readerOptions.length">
          <span class="header__sep">/</span>
          <Select v-model="readerSel" :options="readerOptions" ghost />
        </template>
      </template>
      <template v-else-if="onTables">
        <template v-if="predictorOptions.length">
          <span class="header__sep">/</span>
          <Select v-model="predictorSel" :options="predictorOptions" ghost />
        </template>
      </template>
      <template v-else-if="area === 'studio' && currentModel && pickerOptions.length">
        <span class="header__sep">/</span>
        <span v-if="comparing" class="header__model header__cmp">
          <template v-for="(l, i) in compareLanes" :key="l.id">
            <span v-if="i" class="header__cmp-vs">vs.</span>
            <Tooltip :label="l.id">
              <span class="header__cmp-lane">
                <VendorLogo v-if="l.vendor" :vendor="l.vendor" :size="14" />
                {{ l.label }}
                <span v-if="l.spec" class="header__spec">{{ l.spec }}</span>
              </span>
            </Tooltip>
          </template>
        </span>
        <Select v-else v-model="modelSel" :options="pickerOptions" ghost />
        <Tooltip v-if="isVision && !comparing" label="This model can read images">
          <span class="header__vision"><Icon name="eye" :size="13" /></span>
        </Tooltip>
        <Tooltip v-if="specLabel && !comparing" :label="`Speculative decode: ${specLabel}`">
          <span class="header__spec">{{ specLabel }}</span>
        </Tooltip>
      </template>
      <template v-else-if="area === 'studio' && soleEncoder">
        <span class="header__sep">/</span>
        <Tooltip label="Open Embeddings &amp; rerank">
          <RouterLink class="header__encoder" :to="{ name: 'embeddings' }">
            <Icon name="sliders" :size="13" />
            {{ soleEncoder }}
          </RouterLink>
        </Tooltip>
      </template>
    </div>

    <div class="header__spacer" />

    <!-- model downloads: manager-owned jobs, so this survives reloads and
         other pages. Click for per-download progress + cancel/resume. -->
    <Popover v-if="downloads.visible.length" align="end">
      <template #trigger>
        <button class="header__dl" :class="{ 'header__dl--busy': dlBusy }" type="button">
          <Icon :name="dlBusy ? 'spinner' : 'arrow-down'" :size="14" :class="{ 'header__dl-spin': dlBusy }" />
          <span v-if="dlBusy">{{ dlPct }}%</span>
          <span v-else>downloads</span>
        </button>
      </template>
      <div class="header__dl-list">
        <div v-for="j in downloads.visible" :key="j.id" class="header__dl-row">
          <div class="header__dl-top">
            <span class="header__dl-name">{{ j.display }}</span>
            <span class="header__dl-size">{{ fmtBytes(j.total) }}</span>
          </div>
          <div class="header__dl-bar">
            <div class="header__dl-fill" :style="{ width: `${jobPct(j)}%` }" />
          </div>
          <div class="header__dl-line">{{ jobLine(j) }}</div>
          <div v-if="j.start?.port && (j.status.state === 'running' || j.status.state === 'cancelled')" class="header__dl-line">
            starts on port {{ j.start.port }} when done
          </div>
          <div class="header__dl-acts">
            <button v-if="j.status.state === 'running'" type="button" class="header__dl-act" @click="downloads.cancel(j.id)">
              Cancel
            </button>
            <button
              v-if="j.status.state === 'cancelled' || j.status.state === 'error'"
              type="button"
              class="header__dl-act"
              @click="downloads.resume(j.id)"
            >
              Resume
            </button>
            <button v-if="!jobActive(j)" type="button" class="header__dl-act" @click="downloads.dismiss(j.id)">
              Dismiss
            </button>
          </div>
        </div>
      </div>
    </Popover>

    <!-- Runners still serving an OLDER build than the manager.
         NOT an "update available" - the runner ships in the same package, so
         the new one is already on disk beside the exe; a process that was
         serving before the swap just carries on with the old image. The action
         is a restart, and saying "update" would send someone hunting a
         download that does not exist. -->
    <Popover v-if="fleet.staleRunners.length" align="end">
      <template #trigger>
        <button class="header__stale" type="button">
          <Icon name="warning" :size="14" />
          <span>{{ fleet.staleRunners.length }}</span>
        </button>
      </template>
      <div class="header__stale-list">
        <p class="header__stale-head">
          Running an older build than paddock {{ models.serverVersion }}. Restart to
          pick it up - nothing to download.
        </p>
        <div v-for="r in fleet.staleRunners" :key="r.port" class="header__stale-row">
          <span class="header__stale-name">{{ r.display ?? r.model ?? `port ${r.port}` }}</span>
          <span class="header__stale-ver">{{ r.version }} &rarr; {{ models.serverVersion }}</span>
          <!-- Routes to the model's own page rather than restarting from here.
               A restart drops whatever is in flight on that endpoint, so it
               wants the page that shows what is running and asks properly -
               not a one-click button in a popover. -->
          <RouterLink
            class="pk-btn pk-btn--sm"
            :to="{ name: 'server-detail', params: { port: r.port } }"
          >
            Open
          </RouterLink>
        </div>
      </div>
    </Popover>

    <Tooltip v-if="updateReady" label="See what's new">
      <RouterLink class="header__upd" :to="{ name: 'manage-settings' }">
        <Icon name="download" :size="14" />
        <span>{{ updateReady }} available</span>
      </RouterLink>
    </Tooltip>
    <BackendStatus />
    <Tooltip v-if="models.serverVersion" :label="`Paddock ${models.serverBuild}`">
      <span class="header__ver">v{{ models.serverVersion }}</span>
    </Tooltip>
    <Tooltip v-if="readiness.hasMetrics" label="GPU metrics">
      <button
        class="pk-icon-btn"
        :class="{ 'header__btn--on': tele.open }"
        aria-label="GPU metrics"
        :aria-pressed="tele.open"
        @click="tele.toggle"
      >
        <Icon name="graphics-card" :size="16" />
      </button>
    </Tooltip>
    <Tooltip label="Send feedback">
      <button class="pk-icon-btn" aria-label="Send feedback" @click="feedbackOpen = true">
        <Icon name="megaphone" :size="16" />
      </button>
    </Tooltip>
    <Tooltip :label="`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`">
      <button class="pk-icon-btn" @click="toggleTheme">
        <Icon :name="theme === 'dark' ? 'sun' : 'moon'" :size="16" />
      </button>
    </Tooltip>

    <FeedbackDialog :open="feedbackOpen" @close="feedbackOpen = false" />
  </header>
</template>

<style scoped>
.header {
  height: var(--pk-header-height);
  flex-shrink: 0;
  background: var(--pk-bg-surface);
  border-bottom: 1px solid var(--pk-border-default);
  display: flex;
  align-items: center;
  padding: 0 12px;
  gap: 8px;
}
.header__ctx {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}
.header__brand {
  font-weight: 700;
  font-size: var(--pk-font-size-base);
  color: var(--pk-text-primary);
  letter-spacing: -0.01em;
}
/* the Manager | Studio switch - segmented, always visible, owns the split */
.header__areas {
  display: inline-flex;
  margin-left: 10px;
  padding: 2px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  gap: 2px;
}
/* :deep() is required here, not cosmetic. Reka renders a ToggleGroupItem
   through a roving-focus asChild clone, and that clone drops our scope
   attribute - the class lands on the button but `data-v-` does not, so a
   plain `.header__area {}` matches nothing and the button falls back to
   native chrome. Scoping through the group keeps the containment. */
.header__areas :deep(.header__area) {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: transparent;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}
.header__areas :deep(.header__area:hover) {
  color: var(--pk-text-primary);
}
.header__areas :deep(.header__area[data-state='on']) {
  background: var(--pk-bg-surface);
  color: var(--pk-text-primary);
  box-shadow: 0 0 0 1px var(--pk-border-default);
}
.header__sep {
  color: var(--pk-text-muted);
}
/* encoder-only hint: reads like the model chip, acts like a link */
.header__encoder {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 8px;
  border-radius: var(--pk-radius-md);
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-secondary);
  text-decoration: none;
}
.header__encoder:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.header__model {
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: default;
}
/* compare mode: the lanes themselves, marks included */
.header__cmp {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.header__cmp-lane {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--pk-text-primary);
}
.header__cmp-lane svg,
.header__cmp-lane img {
  display: block;
}
.header__cmp-vs {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
/* vision-capable indicator */
.header__spec {
  padding: 2px 7px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  font-size: 11px;
  line-height: 1.4;
  white-space: nowrap;
  flex-shrink: 0;
}
.header__vision {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
  flex-shrink: 0;
}
.header__spacer {
  flex: 1;
}
.header__stale {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: var(--pk-radius-sm);
  border: 1px solid var(--pk-status-warning);
  background: transparent;
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
  cursor: pointer;
}
.header__stale-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 320px;
  padding: 4px;
}
.header__stale-head {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  margin: 0;
}
.header__stale-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header__stale-name {
  flex: 1;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.header__stale-ver {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
/* a newer paddock exists - same chip grammar as the downloads pill, accent
   because it is good news with an action, not a warning */
.header__upd {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border: 1px solid var(--pk-accent);
  border-radius: var(--pk-radius-full);
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text, var(--pk-accent));
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  white-space: nowrap;
  text-decoration: none;
}
.header__upd:hover {
  filter: brightness(1.06);
}

/* server version (matches Traverse's muted header chip) */
.header__ver {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  white-space: nowrap;
  cursor: default;
}
.header__btn--on {
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}

/* the downloads chip + popover list */
.header__dl {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
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
.header__dl--busy {
  border-color: var(--pk-accent);
  color: var(--pk-accent-text, var(--pk-accent));
  background: var(--pk-accent-subtle);
}
.header__dl-spin {
  animation: header-dl-spin 0.9s linear infinite;
}
@keyframes header-dl-spin {
  to {
    transform: rotate(360deg);
  }
}
.header__dl-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 240px;
}
.header__dl-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.header__dl-top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}
.header__dl-name {
  font-weight: 600;
  color: var(--pk-text-primary);
}
.header__dl-size {
  color: var(--pk-text-muted);
}
.header__dl-bar {
  height: 5px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  border: 1px solid var(--pk-border-default);
  overflow: hidden;
}
.header__dl-fill {
  height: 100%;
  background: var(--pk-accent);
  transition: width 0.4s ease;
}
.header__dl-line {
  color: var(--pk-text-muted);
}
.header__dl-acts {
  display: flex;
  gap: 10px;
}
.header__dl-act {
  background: none;
  border: none;
  padding: 0;
  font: inherit;
  font-weight: 600;
  color: var(--pk-accent);
  cursor: pointer;
}
.header__dl-act:hover {
  text-decoration: underline;
}
</style>
