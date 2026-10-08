<script setup lang="ts">
// Tables - try a running tabular predictor (Kumo-Tabular) on your own rows.
// Paste or open a table, say which column to predict, and the rows whose
// target is empty come back filled in: a class with its confidence, or a
// number with an 80% range. The labelled rows are the model's whole
// "training" - it reads them in context on every call, nothing is fitted
// beforehand. Everything runs through the same /v1/tabular/predictions your
// code calls (relayed by the manager, which holds the runner key); the page
// shows the equivalent curl. Sessions and immutable run snapshots live in the
// manager's SQLite store, shared with the native workspace.
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { onBeforeRouteLeave, onBeforeRouteUpdate, useRoute, useRouter, type LocationQuery } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useTablesStore } from '@/stores/tables'
import { useToastsStore } from '@/stores/toasts'
import { usePanelFold } from '@/composables/usePanelFold'
import { datasetKey, type TableInput, type TableRun, type TableSession } from '@/lib/table-history'
import { uuid } from '@/lib/uuid'
import ReadsSidebar from '@/components/reads/ReadsSidebar.vue'
import { useModelsStore } from '@/stores/models'
import { useFleetStore } from '@/stores/fleet'
import { useRegistryStore } from '@/stores/registry'
import { artifactAt } from '@/lib/weight-choices'
import { copyText } from '@/lib/clipboard'
import {
  buildPlan,
  curlFor,
  defaultSpec,
  exampleClassification,
  exampleRegression,
  formatNumber,
  isMissing,
  limitsFrom,
  parseDelimited,
  readResults,
  resultsCsv,
  type ColumnType,
  type Plan,
  type PredictionResponse,
  type ResultRow,
  type TableSpec,
  type TabularLimits,
} from '@/lib/tables'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import Checkbox from '@/components/ui/Checkbox.vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import NumberField from '@/components/ui/NumberField.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import ToggleGroup from '@/components/ui/ToggleGroup.vue'
import ToggleGroupItem from '@/components/ui/ToggleGroupItem.vue'

const models = useModelsStore()
const fleet = useFleetStore()
const reg = useRegistryStore()
const route = useRoute()
const router = useRouter()
const toasts = useToastsStore()
const history = useTablesStore()
const restoring = ref(false)
const switching = ref(false)
const fileName = ref('')
const selectedRun = ref('')
const showSource = ref(false)

// keep the list fresh while the page is open (a model started or stopped in
// the Manager appears without a reload)
let timer: number | undefined
let releaseFleet: (() => void) | null = null
onMounted(() => {
  void history.refresh()
  void models.refresh()
  timer = window.setInterval(() => void models.refresh(), 5000)
  releaseFleet = fleet.hold()
  if (!reg.models.length) void reg.refresh()
})
onUnmounted(() => {
  clearInterval(timer)
  releaseFleet?.()
})

const predictors = computed(() => models.models.filter((m) => m.kind === 'tabular'))
// ?port= (the model page's "open in Studio") names the endpoint; it wins once
// that endpoint is in the list - the fleet loads after the page does
// (once: the choice outlives the page in the store, so a later visit must not
// be dragged back to the link that opened an earlier one)
const wanted = Number(route.query.port) || 0
let wantedTaken = false
const { port } = storeToRefs(history)
watch(
  predictors,
  (list) => {
    if (wanted && !wantedTaken && list.some((m) => m.port === wanted)) {
      port.value = wanted
      wantedTaken = true
    } else if (list.length && !list.some((m) => m.port === port.value)) {
      const savedModel = history.document?.draft.model
      port.value = (savedModel ? list.find((m) => m.id === savedModel) : list[0])?.port ?? 0
    }
  },
  { immediate: true },
)
const current = computed(() => predictors.value.find((m) => m.port === port.value))

// The endpoint's own contract (/server relay): which task this checkpoint
// answers and the limits it enforces - read, never assumed.
const limits = ref<TabularLimits | null>(null)
const estimators = ref(8)
const seed = ref(0)
watch(
  port,
  async (p) => {
    limits.value = null
    if (!p) return
    try {
      const r = await fetch(`/api/runners/${p}/server`)
      limits.value = r.ok ? limitsFrom(await r.json()) : null
      if (limits.value && !restoring.value && !history.document) estimators.value = limits.value.defaultEstimators
    } catch {
      limits.value = null
    }
  },
  { immediate: true },
)
const task = computed(() => limits.value?.task ?? null)

// The other task, one click away. A runner answers one task, so a table whose
// target is the other kind needs the other checkpoint: switch to a running one
// when there is one, else start the same size's other-task checkpoint (the
// start form preselects it). Each runner's task is its own /server answer.
const taskOf = ref<Record<number, string>>({})
watch(
  predictors,
  async (list) => {
    for (const { port: p } of list) {
      if (!p || p in taskOf.value) continue
      try {
        const r = await fetch(`/api/runners/${p}/server`)
        const l = r.ok ? limitsFrom(await r.json()) : null
        if (l) taskOf.value = { ...taskOf.value, [p]: l.task }
      } catch {
        // not answering yet - the next refresh asks again
      }
    }
  },
  { immediate: true },
)
const otherTask = computed(() =>
  task.value === 'classification' ? 'regression' : task.value === 'regression' ? 'classification' : null,
)
const otherRunner = computed(() =>
  otherTask.value ? predictors.value.find((m) => !!m.port && taskOf.value[m.port] === otherTask.value) : undefined,
)
const otherStart = computed(() => {
  const want = otherTask.value
  if (!want || otherRunner.value) return null
  const cfg = fleet.rows.find((r) => r.port === port.value)?.config
  const cat = reg.models.find((m) => m.id === cfg?.model) ?? reg.models.find((m) => m.capability.includes('tabular'))
  const axes = cat?.specs?.choices ?? []
  const taskAxis = axes.find((x) => x.options.some((o) => o.value.toLowerCase() === want))
  if (!cat || !taskAxis) return null
  const weights = cat.artifacts.filter((a) => a.kind === 'weights')
  const now = weights.find((a) => a.id === cfg?.artifact) ?? weights.find((a) => a.default)
  const value = taskAxis.options.find((o) => o.value.toLowerCase() === want)!.value
  const to = artifactAt(axes, weights, { ...(now?.choice ?? {}), [taskAxis.name]: value })
  return to ? { name: 'server-new-config', params: { model: cat.id }, query: { artifact: to.id } } : null
})
const otherLabel = computed(() => (otherTask.value === 'regression' ? 'Predict numbers instead' : 'Predict classes instead'))

// -- the user's table ------------------------------------------------------------
const text = ref('')
const table = computed(() => (text.value.trim() ? parseDelimited(text.value) : null))
const spec = ref<TableSpec | null>(null)
// a new header resets the column choices; editing cells keeps them
watch(
  () => table.value?.header.join('\u0000') ?? '',
  () => {
    spec.value = table.value && table.value.header.length ? defaultSpec(table.value) : null
    error.value = null
  },
)

const fileInput = ref<HTMLInputElement | null>(null)
async function onFile(e: Event): Promise<void> {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return
  if (f.size > 8 * 1024 * 1024) { error.value = 'Open a table smaller than 8 MiB.'; return }
  fileName.value = f.name
  text.value = await f.text()
  ;(e.target as HTMLInputElement).value = ''
}
function loadExample(): void {
  text.value = task.value === 'regression' ? exampleRegression() : exampleClassification()
  fileName.value = task.value === 'regression' ? 'Rent example.csv' : 'Approval example.csv'
}

const targetOptions = computed<SelectOption[]>(
  () => table.value?.header.map((h, j) => ({ value: j, label: h })) ?? [],
)
const target = computed<number>({
  get: () => spec.value?.target ?? 0,
  set: (j) => {
    if (spec.value) spec.value = { ...spec.value, target: Number(j) }
  },
})
function setType(j: number, t: string): void {
  if (!spec.value) return
  const types = [...spec.value.types]
  types[j] = t as ColumnType
  spec.value = { ...spec.value, types }
}
function setUse(j: number, on: boolean | 'indeterminate'): void {
  if (!spec.value) return
  const use = [...spec.value.use]
  use[j] = on === true
  spec.value = { ...spec.value, use }
}

/** per column: a few values to recognise it by, and how many are missing */
const columns = computed(() => {
  const t = table.value
  if (!t) return []
  return t.header.map((name, j) => {
    const vals = t.rows.map((r) => r[j] ?? '')
    const present = vals.filter((v) => !isMissing(v))
    return {
      name,
      sample: [...new Set(present)].slice(0, 4).join(', '),
      missing: vals.length - present.length,
    }
  })
})

const planned = computed(() => {
  if (!table.value || !spec.value || !limits.value) return null
  return buildPlan(table.value, spec.value, limits.value, estimators.value, seed.value)
})
const plan = computed<Plan | null>(() => (planned.value && 'plan' in planned.value ? planned.value.plan : null))
const planError = computed(() => (planned.value && 'error' in planned.value ? planned.value.error : null))

// -- running ---------------------------------------------------------------------
const busy = ref(false)
const error = ref<string | null>(null)
const results = ref<{ rows: ResultRow[]; classes: string[]; plan: Plan; target: string } | null>(null)
const usage = ref<{ ms: number; gpuMs: number | null; estimators: number | null } | null>(null)
const canRun = computed(() => !!current.value && !!plan.value && !busy.value)
const resultTable = ref<ReturnType<typeof parseDelimited> | null>(null)
const resultSpec = ref<TableSpec | null>(null)
const resultTask = ref<'classification' | 'regression' | null>(null)
const navigationBlocked = computed(() => busy.value || restoring.value || switching.value || history.saving)
let saveTimer: ReturnType<typeof setTimeout> | undefined
let editGeneration = 0
let savedGeneration = 0

watch([text, spec, estimators, seed, port, fileName], () => {
  if (restoring.value) return
  ++editGeneration
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => { if (!busy.value) void saveDraft() }, 800)
}, { deep: true })
async function snapshot(): Promise<{ input: TableInput; source: string }> {
  const source = text.value
  const input = {
    dataset: '', fileName: fileName.value,
    spec: spec.value ? JSON.parse(JSON.stringify(spec.value)) as TableSpec : null,
    model: current.value?.id ?? history.document?.draft.model ?? '', port: port.value,
    estimators: estimators.value, seed: seed.value,
  }
  input.dataset = await datasetKey(source)
  return { input, source }
}
async function saveDraft(): Promise<boolean> {
  if (restoring.value) return false
  if (!text.value && !history.document) return true
  if (savedGeneration === editGeneration) return !history.hasPendingSave()
  const attempt = editGeneration
  try {
  const { input, source } = await snapshot()
  const ok = await history.save(input, source)
  if (ok) savedGeneration = attempt
  return ok
  } catch (e) { history.error = e instanceof Error ? e.message : String(e); return false }
}
async function restoreDraft(input: TableInput): Promise<void> {
  restoring.value = true
  clearTimeout(saveTimer)
  text.value = history.document?.datasets[input.dataset] ?? ''
  fileName.value = input.fileName
  await nextTick() // header watcher first, then the saved column choices
  spec.value = input.spec ? JSON.parse(JSON.stringify(input.spec)) as TableSpec : null
  port.value = predictors.value.find((m) => m.port === input.port && m.id === input.model)?.port
    ?? predictors.value.find((m) => m.id === input.model)?.port ?? 0
  estimators.value = input.estimators
  seed.value = input.seed
  await nextTick()
  savedGeneration = editGeneration
  restoring.value = false
}
// /studio/tables/<id> names the session on screen, as a chat or a read is
// named: the sidebar navigates, the route watcher (end of script) loads, so a
// reload, the back button or a pasted link lands on the same table.
function openSession(id: string): void {
  if (navigationBlocked.value || id === history.document?.id) return
  void router.push({ name: 'tables', params: { id } })
}
async function newSession(): Promise<void> {
  if (navigationBlocked.value) return
  // the route guard saves the draft being left, the route watcher clears the page
  if (route.params.id) { void router.push({ name: 'tables' }); return }
  switching.value = true
  try { if (await saveDraft()) await clearSession() } finally { switching.value = false }
}
async function showSession(id: string): Promise<void> {
  switching.value = true
  try {
  restoring.value = true
  const doc = await history.open(id)
  if (doc) {
    await restoreDraft(doc.draft)
    results.value = null; usage.value = null; selectedRun.value = ''
    const last = doc.runs[doc.runs.length - 1]
    if (last) selectRun(last)
  } else {
    // a deleted or mistyped id: the page keeps what it had, and so does the URL
    syncUrl(history.document?.id)
  }
  restoring.value = false
  } finally { switching.value = false }
}
async function clearSession(): Promise<void> {
  clearTimeout(saveTimer)
  restoring.value = true
  history.reset()
  text.value = ''; fileName.value = ''; spec.value = null
  results.value = null; usage.value = null; selectedRun.value = ''
  await nextTick()
  savedGeneration = editGeneration
  restoring.value = false
}
async function renameSession(id: string, title: string): Promise<void> {
  if (navigationBlocked.value) return
  switching.value = true
  clearTimeout(saveTimer)
  try { if (await saveDraft()) await history.rename(id, title) }
  finally { switching.value = false }
}
async function saveCopy(): Promise<void> {
  if (navigationBlocked.value) return
  history.fork()
  ++editGeneration
  await saveDraft()
}
async function removeSession(id: string): Promise<void> {
  if (navigationBlocked.value) return
  clearTimeout(saveTimer)
  const active = history.document?.id === id
  if (await history.remove(id) && active) {
    restoring.value = true
    text.value = ''; fileName.value = ''; results.value = null; selectedRun.value = ''
    await nextTick()
    savedGeneration = editGeneration
    restoring.value = false
  }
}
function selectRun(run: TableRun): void {
  try {
    const source = history.document?.datasets[run.input.dataset]
    if (source === undefined || !run.input.spec) throw new Error('Missing saved table inputs')
    const t = parseDelimited(source)
    const built = buildPlan(t, run.input.spec, {
      task: run.task, maxContextRows: 4096, maxQueryRows: 1024, maxColumns: 500,
      maxCells: 131072, maxClasses: 10, defaultEstimators: 8, maxEstimators: 16,
    }, run.input.estimators, run.input.seed)
    if ('error' in built) throw new Error(built.error)
    results.value = { rows: readResults(run.response, built.plan), classes: (run.response.classes ?? []).map(String),
      plan: built.plan, target: t.header[run.input.spec.target] ?? 'prediction' }
    resultTable.value = t; resultSpec.value = run.input.spec; resultTask.value = run.task
    selectedRun.value = run.id
    usage.value = { ms: Math.round(run.ms), gpuMs: run.response.usage?.gpu_ms ?? null, estimators: run.response.num_estimators ?? null }
  } catch (e) { error.value = String(e) }
}
const runOptions = computed<SelectOption[]>(() => (history.document?.runs ?? []).map((r, i) => ({
  value: r.id, label: `Run ${i + 1} · ${new Date(r.at).toLocaleString()}`, hint: r.input.model,
})))
watch(selectedRun, (id) => {
  const run = history.document?.runs.find((r) => r.id === id)
  if (run) selectRun(run)
})
async function useRunInputs(): Promise<void> {
  const run = history.document?.runs.find((r) => r.id === selectedRun.value)
  if (!run || navigationBlocked.value) return
  await restoreDraft(run.input)
  ++editGeneration
  await saveDraft()
}
onBeforeRouteLeave(async () => !busy.value && !switching.value && await saveDraft())
function beforeUnload(e: BeforeUnloadEvent): void {
  if (busy.value || history.saving || savedGeneration !== editGeneration) { e.preventDefault(); e.returnValue = '' }
}
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onUnmounted(() => { clearTimeout(saveTimer); window.removeEventListener('beforeunload', beforeUnload) })

async function run(): Promise<void> {
  if (!canRun.value || !current.value || !plan.value) return
  const p = plan.value
  busy.value = true
  error.value = null
  clearTimeout(saveTimer)
  const model = current.value
  const attempt = editGeneration
  const t0 = performance.now()
  try {
    const captured = await snapshot()
    const res = await fetch(`/api/runners/${model.port}/v1/tabular/predictions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: model.id, ...p.body }),
    })
    const json: unknown = await res.json().catch(() => null)
    if (!res.ok) {
      throw new Error(
        (json as { error?: { message?: string } } | null)?.error?.message ?? `HTTP ${res.status}`,
      )
    }
    const r = json as PredictionResponse
    resultTable.value = parseDelimited(captured.source)
    resultSpec.value = captured.input.spec
    resultTask.value = r.task
    results.value = {
      rows: readResults(r, p),
      classes: (r.classes ?? []).map(String),
      plan: p,
      target: table.value?.header[spec.value?.target ?? 0] ?? 'prediction',
    }
    usage.value = {
      ms: Math.round(performance.now() - t0),
      gpuMs: r.usage?.gpu_ms ?? null,
      estimators: r.num_estimators ?? null,
    }
    const run: TableRun = { id: uuid(), at: Date.now(), input: captured.input, task: r.task,
      ms: performance.now() - t0, response: r }
    selectedRun.value = run.id
    if (await history.save(captured.input, captured.source, run)) savedGeneration = attempt
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
function onKey(e: KeyboardEvent): void {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') void run()
}

/** the first few feature columns, so a result row is recognisable */
const shownFeatures = computed(() => results.value?.plan.features.slice(0, 4) ?? [])
const copied = ref(false)
async function copyCsv(): Promise<void> {
  if (!resultTable.value || !resultSpec.value || !results.value) return
  try {
    await copyText(resultsCsv(resultTable.value, resultSpec.value, results.value.rows))
    copied.value = true
    setTimeout(() => (copied.value = false), 1400)
  } catch {
    /* clipboard blocked */
  }
}
const curl = computed(() => (current.value && plan.value ? curlFor(port.value, current.value.id, plan.value) : ''))
const taskLine = computed(() =>
  task.value === 'classification'
    ? `This model picks a class for each row (up to ${limits.value?.maxClasses ?? 10} classes), with its confidence.`
    : task.value === 'regression'
      ? 'This model predicts a number for each row, with the range it is 80% sure of.'
      : '',
)
function pct(v: number | null): string {
  return v === null ? '-' : `${Math.round(v * 100)}%`
}

// -- which session the URL names -------------------------------------------------
// Last in the script on purpose: the watcher runs at once and reaches for
// everything above.
function paramId(raw: unknown): string | undefined {
  return typeof raw === 'string' && raw ? raw : undefined
}
function syncUrl(id: string | undefined, query?: LocationQuery): void {
  if (route.name !== 'tables' || id === paramId(route.params.id)) return
  void router.replace({ name: 'tables', params: id ? { id } : {}, query })
}
// A first save mints the id, "Save a copy" forks a new one and a delete drops
// it - the URL follows the screen each time.
watch(() => history.document?.id, (id) => syncUrl(id))
// Moving to another session saves the draft first and waits out a run or a
// save in flight, the rule the sidebar's buttons follow. A URL that only
// catches up with the screen passes straight through.
onBeforeRouteUpdate(async (to) => {
  if (paramId(to.params.id) === history.document?.id) return true
  if (navigationBlocked.value) return false
  return await saveDraft()
})
function showStored(doc: TableSession): Promise<void> {
  return restoreDraft(doc.draft).then(() => {
    const last = doc.runs[doc.runs.length - 1]
    if (last) selectRun(last)
  })
}
let firstRoute = true
watch(
  () => route.params.id,
  (raw) => {
    // leaving for another page with an :id of its own (a chat, a read) changes
    // the param before this page unmounts - that id is not a table's
    if (route.name !== 'tables') return
    const id = paramId(raw)
    const initial = firstRoute
    firstRoute = false
    const doc = history.document
    if (id) {
      if (doc?.id !== id) void showSession(id)
      else if (initial) void showStored(doc)
      return
    }
    if (!doc) return
    // Coming back to the bare page from elsewhere keeps the session the store
    // still holds, as the page always did, and names it. Reaching the bare
    // URL from a session is New table.
    if (initial) {
      void showStored(doc)
      syncUrl(doc.id, route.query)
    } else {
      switching.value = true
      void clearSession().finally(() => { switching.value = false })
    }
  },
  { immediate: true },
)

// the side panel folds like the chat list and remembers it, as Reads does
const panelOpen = usePanelFold('tablesPanelOpen')
</script>

<template>
  <div class="tables-workspace">
  <ReadsSidebar v-if="panelOpen" noun="table" :reads="history.sessions" :active-id="history.document?.id ?? null"
    :loaded="history.loaded" :error="history.error" :busy="navigationBlocked"
    @new="newSession" @open="openSession" @rename="renameSession" @remove="removeSession" @fold="panelOpen = false" />
  <aside v-else class="tables-workspace__rail">
    <Tooltip label="Show tables" side="right">
      <button class="pk-icon-btn tables-workspace__railbtn" type="button" aria-label="Show tables" @click="panelOpen = true">
        <Icon name="panel-left" :size="18" />
      </button>
    </Tooltip>
    <Tooltip label="New table" side="right">
      <button class="pk-icon-btn tables-workspace__railbtn" type="button" aria-label="New table" :disabled="navigationBlocked" @click="newSession">
        <Icon name="plus" :size="18" />
      </button>
    </Tooltip>
  </aside>
  <main class="tables-workspace__main">
  <div class="tb" @keydown="onKey">
    <div class="tb__head">
      <div>
        <h1 class="tb__title">Tables</h1>
        <p class="tb__lead">
          Give it labelled rows and it fills in the missing column - the same
          /v1/tabular/predictions your code calls.
        </p>
      </div>
    </div>

    <div v-if="!predictors.length && !models.loading && !history.document" class="tb__none">
      <Icon name="table" :size="32" class="tb__none-icon" />
      <p class="tb__none-title">No table model is running</p>
      <p class="tb__none-txt">
        Start Kumo Tabular in the Manager - one endpoint predicts classes, the other numbers.
      </p>
      <RouterLink class="pk-btn pk-btn--primary" :to="{ name: 'server-new' }">
        <Icon name="play" :size="14" /> Start a model
      </RouterLink>
    </div>

    <template v-else>
    <p v-if="!predictors.length && !models.loading" class="tb__nomodel">
      No table model is running, so this table cannot run again until Kumo Tabular is started in
      the Manager.
    </p>
    <p v-if="history.error" class="tb__error" role="alert">{{ history.error }} <button class="pk-btn" :disabled="navigationBlocked" @click="saveDraft">Retry save</button> <button class="pk-btn" :disabled="navigationBlocked" @click="saveCopy">Save a copy</button></p>
    <fieldset class="tb__fields" :disabled="busy || restoring || switching">
      <div class="tb__card">
        <div class="tb__row">
          <span class="tb__label">Your table</span>
          <span class="tb__hint">{{ taskLine }}</span>
          <button v-if="otherRunner" class="pk-btn pk-btn--sm pk-btn--ghost tb__right" @click="port = otherRunner.port ?? port">
            {{ otherLabel }}
          </button>
          <RouterLink v-else-if="otherStart" class="pk-btn pk-btn--sm pk-btn--ghost tb__right" :to="otherStart">
            {{ otherLabel }}
          </RouterLink>
        </div>
        <textarea v-if="showSource || !text"
          v-model="text"
          class="pk-input tb__ta"
          rows="8"
          spellcheck="false"
          :disabled="busy || restoring"
          placeholder="Paste a table with a header row (CSV, or straight from a spreadsheet). Leave the column to predict empty on the rows you want predicted."
        />
        <div class="tb__actions">
          <button class="pk-btn" :disabled="busy || restoring" @click="fileInput?.click()">
            <Icon name="upload" :size="14" /> Open CSV
          </button>
          <input ref="fileInput" class="tb__file" type="file" accept=".csv,.tsv,.txt,text/csv" @change="onFile" />
          <button class="pk-btn" :disabled="busy || restoring" @click="loadExample">
            <Icon name="file-text" :size="14" /> Load an example
          </button>
          <button v-if="text" class="pk-btn pk-btn--ghost" @click="showSource = !showSource">{{ showSource ? 'Hide source' : 'Edit source' }}</button>
          <span v-if="table" class="tb__hint">{{ table.rows.length }} rows · {{ table.header.length }} columns</span>
          <button v-if="text" class="pk-btn pk-btn--ghost tb__right" :disabled="busy || restoring" @click="text = ''">Clear</button>
        </div>
      </div>

      <div v-if="table && spec" class="tb__card">
        <div class="tb__row">
          <span class="tb__label">Predict</span>
          <Select v-model="target" :options="targetOptions" />
          <span v-if="plan" class="tb__hint">
            {{ plan.contextRows.length }} labelled rows to learn from ·
            {{ plan.queryRows.length }} to predict · {{ plan.features.length }} columns
          </span>
        </div>
        <div class="tb__tablewrap">
          <table class="tb__table">
            <thead>
              <tr>
                <th>Use</th>
                <th>Column</th>
                <th>Read as</th>
                <th>Values</th>
                <th class="c-num">Missing</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(c, j) in columns" :key="j" :class="{ 'tb__target': j === spec.target }">
                <td>
                  <Checkbox
                    v-if="j !== spec.target"
                    :model-value="spec.use[j]"
                    @update:model-value="(v: boolean | 'indeterminate') => setUse(j, v)"
                  />
                  <span v-else class="tb__tag">target</span>
                </td>
                <td class="tb__name">{{ c.name }}</td>
                <td>
                  <ToggleGroup
                    v-if="j !== spec.target"
                    class="tb__kind"
                    :model-value="spec.types[j]"
                    :label="`How to read ${c.name}`"
                    @update:model-value="(v: string) => setType(j, v)"
                  >
                    <ToggleGroupItem value="numerical" class="tb__kindopt">Numbers</ToggleGroupItem>
                    <ToggleGroupItem value="categorical" class="tb__kindopt">Categories</ToggleGroupItem>
                  </ToggleGroup>
                  <span v-else class="c-dim">{{ task === 'classification' ? 'classes' : 'numbers' }}</span>
                </td>
                <td class="tb__sample">{{ c.sample || '-' }}</td>
                <td class="c-num c-dim">{{ c.missing }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="tb__actions">
          <label class="tb__opt">
            <span class="tb__hint">Ensemble</span>
            <NumberField v-model="estimators" :min="1" :max="limits?.maxEstimators ?? 16" :step="1" />
          </label>
          <label class="tb__opt">
            <span class="tb__hint">Seed</span>
            <NumberField v-model="seed" :min="0" :step="1" />
          </label>
          <button class="pk-btn pk-btn--primary" :disabled="!canRun" @click="run">
            <Icon :name="busy ? 'spinner' : 'play'" :size="14" :class="{ spin: busy }" />
            Predict
          </button>
          <span class="tb__hint">Ctrl+Enter</span>
          <span v-if="usage" class="tb__meta">
            {{ usage.ms }} ms<template v-if="usage.gpuMs !== null"> · GPU {{ Math.round(usage.gpuMs) }} ms</template
            ><template v-if="usage.estimators"> · {{ usage.estimators }} estimators</template>
          </span>
        </div>
        <p v-if="planError" class="tb__note">{{ planError }}</p>
      </div>

      <div v-if="error" class="tb__error">{{ error }}</div>

      <div v-if="results && resultTable" class="tb__tablewrap tb__results">
        <div class="tb__actions tb__foot">
          <Select v-model="selectedRun" :options="runOptions" :disabled="busy || restoring" />
          <button class="pk-btn pk-btn--ghost tb__right" :disabled="navigationBlocked" @click="useRunInputs">Use these inputs</button>
        </div>
        <table class="tb__table">
          <thead>
            <tr>
              <th class="c-num">Row</th>
              <th v-for="j in shownFeatures" :key="j">{{ resultTable.header[j] }}</th>
              <th>{{ results.target }}</th>
              <th v-if="resultTask === 'classification'">Confidence</th>
              <th v-else>80% range</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in results.rows" :key="r.row">
              <td class="c-num c-dim">{{ r.row + 1 }}</td>
              <td v-for="j in shownFeatures" :key="j" class="c-dim">{{ resultTable.rows[r.row][j] || '-' }}</td>
              <td class="tb__pred">{{ r.value }}</td>
              <td v-if="resultTask === 'classification'" class="c-num">
                <div class="tb__conf">
                  <div class="tb__confbar" :style="{ width: `${Math.max(3, (r.confidence ?? 0) * 100)}%` }" />
                  {{ pct(r.confidence) }}
                </div>
              </td>
              <td v-else class="c-num">
                {{ r.low === null ? '-' : formatNumber(r.low) }} - {{ r.high === null ? '-' : formatNumber(r.high) }}
              </td>
            </tr>
          </tbody>
        </table>
        <div class="tb__foot">
          <span v-if="resultTask === 'classification' && results.classes.length" class="tb__hint">
            Classes: {{ results.classes.join(', ') }}
          </span>
          <button class="pk-btn pk-btn--ghost tb__right" @click="copyCsv">
            <Icon :name="copied ? 'check' : 'copy'" :size="14" /> {{ copied ? 'Copied' : 'Copy table with predictions' }}
          </button>
        </div>
      </div>

      <Collapsible v-if="curl" class="tb__api" summary="API call">
        <pre>{{ curl }}</pre>
      </Collapsible>
    </fieldset>
    </template>
  </div>
  </main>
  </div>
</template>

<style scoped>
/* the Reads frame, measure for measure: the workspace fills the shell's row
   (without width it shrank to its content and the column sat narrow at the
   left), the pane scrolls, and a folded list leaves a rail behind */
.tables-workspace { display: flex; width: 100%; height: 100%; min-height: 0; overflow: hidden; }
.tables-workspace__main { flex: 1; min-width: 0; overflow: auto; padding: 32px 32px 0; container-type: inline-size; }
.tables-workspace__rail {
  flex: none;
  width: 48px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 12px 0;
  background: var(--pk-bg-surface);
  border-right: 1px solid var(--pk-border-default);
}
.tables-workspace__railbtn {
  width: 34px;
  height: 34px;
}
.tb__fields { border: 0; padding: 0; margin: 0; min-width: 0; }
@media (max-width: 800px) { .tables-workspace :deep(.rsb) { width: 210px; } .tables-workspace__main { padding: 16px; } }
.tb {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
  padding-bottom: 32px;
}
/* the history pages' one width rule (variables.css): Reads, Tables, Masks */
@container (min-width: 1100px) { .tb { max-width: var(--pk-panel-width-wide); } }
.tb__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.tb__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.tb__lead {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.tb__nomodel {
  margin: 0 0 12px;
  padding: 8px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.tb__none {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 64px 24px;
  text-align: center;
}
.tb__none-icon {
  color: var(--pk-text-muted);
}
.tb__none-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.tb__none-txt {
  margin: 0 0 6px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
/* forms sit on a surface card (ServerForm's sf__card recipe): pk-input
   fields on bg-surface, never bare on the content background */
.tb__card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
  margin-bottom: 16px;
}
.tb__row,
.tb__actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.tb__label {
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-secondary);
}
.tb__hint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.tb__right {
  margin-left: auto;
}
.tb__meta {
  margin-left: auto;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.tb__ta {
  height: auto;
  padding: 10px 12px;
  resize: vertical;
  line-height: 1.5;
  font-family: var(--pk-font-mono, monospace);
  font-size: var(--pk-font-size-xs);
  white-space: pre;
}
.tb__file {
  display: none;
}
.tb__opt {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tb__note {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.tb__error {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
  border-radius: var(--pk-radius-md);
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: var(--pk-font-size-sm);
}
/* tables: the fleet/instrument table card, as on the embeddings page */
.tb__tablewrap {
  overflow-x: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.tb__results {
  margin-bottom: 16px;
}
.tb__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.tb__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 10px 14px;
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
}
.tb__table td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--pk-border-default);
  color: var(--pk-text-primary);
  vertical-align: middle;
}
.tb__table tr:last-child td {
  border-bottom: none;
}
.tb__target td {
  background: var(--pk-accent-subtle);
}
.tb__tag {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-accent);
}
.tb__name {
  font-weight: 500;
  white-space: nowrap;
}
/* the header's segmented toggle, verbatim in spirit. :deep() is required:
   Reka renders a ToggleGroupItem through a roving-focus clone that drops the
   scope attribute, so a plain `.tb__kindopt {}` would match nothing */
.tb__kind {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
}
.tb__kind :deep(.tb__kindopt) {
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
.tb__kind :deep(.tb__kindopt:hover) {
  color: var(--pk-text-primary);
}
.tb__kind :deep(.tb__kindopt[data-state='on']) {
  background: var(--pk-bg-surface);
  color: var(--pk-text-primary);
  box-shadow: 0 0 0 1px var(--pk-border-default);
}
.tb__sample {
  color: var(--pk-text-muted);
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tb__pred {
  font-weight: 600;
}
.c-num {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.c-dim {
  color: var(--pk-text-muted);
}
.tb__conf {
  position: relative;
  min-width: 90px;
}
.tb__confbar {
  position: absolute;
  left: 0;
  bottom: -3px;
  height: 3px;
  border-radius: 2px;
  background: var(--pk-accent);
  opacity: 0.7;
}
.tb__foot {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-top: 1px solid var(--pk-border-default);
}
.tb__api pre {
  margin: 8px 0 0;
  padding: 10px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  overflow-x: auto;
}
.spin {
  animation: pk-spin 0.8s linear infinite;
}
@keyframes pk-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
