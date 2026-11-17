<script setup lang="ts">
// One batch: how far it is and what it is doing, what came back - successes
// and failures read straight out of the result files while the run is still
// writing them - and the actions a person has on it: cancel while it runs,
// delete once it has ended, and the three JSONL files to keep (the input,
// the results, and on an OpenAI batch the failures, which that API keeps in
// a file of their own).
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  batchName,
  batchesApi,
  endpointLabel,
  failed,
  fmtPerMinute,
  fmtSpan,
  isActive,
  settled,
  statusLabel,
  unsuccessfulLabel,
  type BatchResultItem,
  type BatchSummary,
  type ResultKind,
} from '@/lib/batches'
import { fmtEta, fmtFileSize, fmtStamp } from '@/lib/format'
import { copyText } from '@/lib/clipboard'
import Icon from '@/components/Icon.vue'
import Dialog from '@/components/ui/Dialog.vue'
import Progress from '@/components/ui/Progress.vue'
import Tabs, { type TabOption } from '@/components/ui/Tabs.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const route = useRoute()
const router = useRouter()
const id = computed(() => String(route.params.id ?? ''))

const batch = ref<BatchSummary | null>(null)
const missing = ref(false)
const failure = ref('')
let timer: number | null = null

async function refresh(): Promise<void> {
  try {
    batch.value = await batchesApi.get(id.value)
    failure.value = ''
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    if (/no such batch/i.test(msg)) missing.value = true
    else failure.value = msg
  }
  // a running batch keeps writing results: pick up what it settled since
  if (batch.value && isActive(batch.value) && results.value.end) void loadResults(false)
  schedule()
}

function schedule(): void {
  if (timer !== null) window.clearTimeout(timer)
  if (missing.value) return
  // an ended batch does not change; one refresh after it ends is enough
  if (batch.value && !isActive(batch.value)) return
  timer = window.setTimeout(() => void refresh(), 3000)
}

onUnmounted(() => {
  if (timer !== null) window.clearTimeout(timer)
})

const status = computed(() => (batch.value ? statusLabel(batch.value) : null))
const total = computed(() => batch.value?.counts.total ?? 0)
const done = computed(() => (batch.value ? settled(batch.value) : 0))
const bad = computed(() => (batch.value ? failed(batch.value) : 0))

/** How long it has run (or ran): start to end, or start to now. */
const ranFor = computed(() => {
  const b = batch.value
  if (!b?.in_progress_ms) return '-'
  return fmtSpan((b.ended_ms ?? Date.now()) - b.in_progress_ms)
})

/** Validation errors a failed batch carries (OpenAI's `errors.data[]`). */
const lineErrors = computed<{ line?: number; message: string }[]>(() => {
  const e = batch.value?.errors as { data?: { line?: number; message?: string }[] } | null
  return (e?.data ?? []).map((x) => ({ line: x.line, message: x.message ?? 'invalid line' }))
})

const options = computed(() => {
  const o = batch.value?.options ?? {}
  const out: [string, string][] = []
  if (typeof o.yield_to_live === 'boolean')
    out.push(['Steps aside for live traffic', o.yield_to_live ? 'Yes' : 'No'])
  if (typeof o.concurrency === 'number') out.push(['Concurrency cap', String(o.concurrency)])
  if (typeof o.webhook_url === 'string') out.push(['Webhook', o.webhook_url])
  return out
})

// ── results ───────────────────────────────────────────────────────────────
const kind = ref<ResultKind>('ok')
const results = ref<{ items: BatchResultItem[]; cursor: number; end: boolean }>({
  items: [],
  cursor: 0,
  end: false,
})
const loadingResults = ref(false)

const resultTabs = computed<TabOption[]>(() => [
  { value: 'ok', label: `Succeeded (${(batch.value?.counts.succeeded ?? 0).toLocaleString()})` },
  {
    value: 'failed',
    label: `${batch.value ? unsuccessfulLabel(batch.value) : 'Failed'} (${bad.value.toLocaleString()})`,
  },
])

/** Next page of the current kind; `fresh` starts over from the top. */
async function loadResults(fresh: boolean): Promise<void> {
  if (loadingResults.value) return
  loadingResults.value = true
  try {
    const cursor = fresh ? 0 : results.value.cursor
    const page = await batchesApi.results(id.value, kind.value, cursor, 50)
    results.value = {
      items: fresh ? page.items : [...results.value.items, ...page.items],
      cursor: page.next_cursor,
      end: page.end,
    }
  } catch (e) {
    failure.value = e instanceof Error ? e.message : String(e)
  } finally {
    loadingResults.value = false
  }
}

watch(kind, () => void loadResults(true))
watch(id, () => {
  batch.value = null
  missing.value = false
  void refresh()
  void loadResults(true)
})
onMounted(() => {
  void refresh()
  void loadResults(true)
})

// one result in full
const open = ref<{ item: BatchResultItem; json: string } | null>(null)
async function show(item: BatchResultItem): Promise<void> {
  try {
    const v = await batchesApi.line(id.value, kind.value, item.offset)
    open.value = { item, json: JSON.stringify(v, null, 2) }
  } catch (e) {
    failure.value = e instanceof Error ? e.message : String(e)
  }
}

// ── actions ───────────────────────────────────────────────────────────────
const confirm = ref<'cancel' | 'delete' | null>(null)
const acting = ref(false)

async function act(): Promise<void> {
  const what = confirm.value
  if (!what) return
  acting.value = true
  try {
    if (what === 'cancel') {
      batch.value = await batchesApi.cancel(id.value)
      schedule()
    } else {
      await batchesApi.remove(id.value)
      void router.push({ name: 'batches' })
    }
    confirm.value = null
  } catch (e) {
    failure.value = e instanceof Error ? e.message : String(e)
    confirm.value = null
  } finally {
    acting.value = false
  }
}

function copied(text: string): void {
  void copyText(text)
}
</script>

<template>
  <div class="bd">
    <button class="bd__back pk-btn pk-btn--ghost" @click="router.push({ name: 'batches' })">
      <Icon name="chevron-left" :size="14" /> Batches
    </button>

    <p v-if="failure" class="bd__failure">
      <Icon name="alert-triangle" :size="14" /> {{ failure }}
    </p>

    <div v-if="missing" class="bd__empty">
      This batch is gone - it was deleted, or the id is not one this box ran.
    </div>
    <div v-else-if="!batch" class="bd__empty">Loading</div>

    <template v-else>
      <header class="bd__head">
        <div class="bd__titlerow">
          <h1 class="bd__title">{{ batchName(batch) }}</h1>
          <span v-if="status" class="bd__chip" :class="`bd__chip--${status.tone}`">
            {{ status.label }}
          </span>
        </div>
        <div class="bd__meta">
          <span class="c-mono">{{ batch.id }}</span>
          <Tooltip label="Copy the batch id">
            <button class="bd__icon" aria-label="Copy the batch id" @click="copied(batch.id)">
              <Icon name="copy" :size="13" />
            </button>
          </Tooltip>
          <span>·</span>
          <span class="c-mono">{{ batch.model }}</span>
          <span>·</span>
          <span>{{ endpointLabel(batch.endpoint) }}</span>
          <span>·</span>
          <span>{{ batch.dialect === 'openai' ? 'OpenAI batch API' : 'Anthropic batch API' }}</span>
        </div>
        <div class="bd__actions">
          <a class="pk-btn pk-btn--ghost" :href="batchesApi.fileUrl(batch.id, 'input')" download>
            <Icon name="download" :size="14" /> Input
          </a>
          <a class="pk-btn pk-btn--ghost" :href="batchesApi.fileUrl(batch.id, 'output')" download>
            <Icon name="download" :size="14" /> Results
          </a>
          <a
            v-if="batch.files.errors !== null"
            class="pk-btn pk-btn--ghost"
            :href="batchesApi.fileUrl(batch.id, 'errors')"
            download
          >
            <Icon name="download" :size="14" /> Failures
          </a>
          <span class="bd__spacer" />
          <button
            v-if="isActive(batch) && batch.status !== 'cancelling'"
            class="pk-btn pk-btn--ghost bd__danger"
            @click="confirm = 'cancel'"
          >
            Cancel batch
          </button>
          <button
            v-if="!isActive(batch)"
            class="pk-btn pk-btn--ghost bd__danger"
            @click="confirm = 'delete'"
          >
            <Icon name="trash" :size="14" /> Delete
          </button>
        </div>
      </header>

      <div class="bd__cards">
        <article class="bd__card bd__card--wide">
          <h3 class="bd__ct">Progress</h3>
          <Progress
            :value="done"
            :max="Math.max(1, total)"
            :tone="bad > 0 ? 'warning' : 'accent'"
            :label="`${done} of ${total} done`"
            size="md"
          />
          <div class="bd__counts">
            <div class="bd__count">
              <span class="bd__k">Done</span>
              <span class="bd__v c-mono">
                {{ done.toLocaleString() }} / {{ total.toLocaleString() }}
              </span>
            </div>
            <div class="bd__count">
              <span class="bd__k">Succeeded</span>
              <span class="bd__v c-mono">{{ batch.counts.succeeded.toLocaleString() }}</span>
            </div>
            <div class="bd__count">
              <span class="bd__k">Failed</span>
              <span class="bd__v c-mono" :class="{ 'bd__warn': batch.counts.errored > 0 }">
                {{ batch.counts.errored.toLocaleString() }}
              </span>
            </div>
            <div v-if="batch.counts.canceled > 0" class="bd__count">
              <span class="bd__k">Cancelled</span>
              <span class="bd__v c-mono">{{ batch.counts.canceled.toLocaleString() }}</span>
            </div>
            <div v-if="batch.counts.expired > 0" class="bd__count">
              <span class="bd__k">Expired</span>
              <span class="bd__v c-mono">{{ batch.counts.expired.toLocaleString() }}</span>
            </div>
            <div v-if="isActive(batch)" class="bd__count">
              <span class="bd__k">Left</span>
              <span class="bd__v c-mono">{{ batch.counts.pending.toLocaleString() }}</span>
            </div>
          </div>
          <dl v-if="batch.progress" class="bd__rows">
            <div v-if="batch.progress.reason" class="bd__row">
              <dt>Why</dt>
              <dd>{{ batch.progress.reason }}</dd>
            </div>
            <div class="bd__row">
              <dt>Rate</dt>
              <dd class="c-mono">{{ fmtPerMinute(batch.progress.rate_per_minute) }}</dd>
            </div>
            <div v-if="batch.progress.eta_seconds" class="bd__row">
              <dt>Done in</dt>
              <dd>{{ fmtEta(batch.progress.eta_seconds) }}</dd>
            </div>
            <div class="bd__row">
              <dt>
                <Tooltip label="Requests on the model right now, out of how many the batch may send at once.">
                  <span>In flight</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">
                {{ batch.progress.in_flight ?? 0 }} / {{ batch.progress.concurrency ?? 0 }}
              </dd>
            </div>
            <div v-if="(batch.progress.retries ?? 0) > 0" class="bd__row">
              <dt>
                <Tooltip label="Requests sent again after a server error or a lost connection to the model.">
                  <span>Retries</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">{{ batch.progress.retries }}</dd>
            </div>
            <div v-if="batch.progress.port" class="bd__row">
              <dt>Serving port</dt>
              <dd class="c-mono">{{ batch.progress.port }}</dd>
            </div>
          </dl>
        </article>

        <article class="bd__card">
          <h3 class="bd__ct">Timing</h3>
          <dl class="bd__rows">
            <div class="bd__row">
              <dt>Created</dt>
              <dd class="c-mono">{{ fmtStamp(new Date(batch.created_ms)) }}</dd>
            </div>
            <div v-if="batch.in_progress_ms" class="bd__row">
              <dt>Started</dt>
              <dd class="c-mono">{{ fmtStamp(new Date(batch.in_progress_ms)) }}</dd>
            </div>
            <div v-if="batch.ended_ms" class="bd__row">
              <dt>Ended</dt>
              <dd class="c-mono">{{ fmtStamp(new Date(batch.ended_ms)) }}</dd>
            </div>
            <div class="bd__row">
              <dt>{{ batch.ended_ms ? 'Ran for' : 'Running for' }}</dt>
              <dd>{{ ranFor }}</dd>
            </div>
            <div class="bd__row">
              <dt>
                <Tooltip label="The completion window the batch was created with. Whatever is not done by then is marked expired; the results that finished are kept.">
                  <span>Window</span>
                </Tooltip>
              </dt>
              <dd>
                {{ batch.completion_window }}
                <span v-if="isActive(batch)" class="bd__muted">
                  · ends {{ fmtStamp(new Date(batch.expires_ms)) }}
                </span>
              </dd>
            </div>
          </dl>
        </article>

        <article class="bd__card">
          <h3 class="bd__ct">Tokens</h3>
          <dl class="bd__rows">
            <div class="bd__row">
              <dt>Input</dt>
              <dd class="c-mono">{{ batch.usage.input_tokens.toLocaleString() }}</dd>
            </div>
            <div v-if="batch.usage.cached_tokens > 0" class="bd__row">
              <dt>From cache</dt>
              <dd class="c-mono">{{ batch.usage.cached_tokens.toLocaleString() }}</dd>
            </div>
            <div class="bd__row">
              <dt>Output</dt>
              <dd class="c-mono">{{ batch.usage.output_tokens.toLocaleString() }}</dd>
            </div>
            <div v-if="batch.usage.reasoning_tokens > 0" class="bd__row">
              <dt>Thinking</dt>
              <dd class="c-mono">{{ batch.usage.reasoning_tokens.toLocaleString() }}</dd>
            </div>
            <div class="bd__row">
              <dt>Files</dt>
              <dd class="bd__muted">
                in {{ batch.files.input !== null ? fmtFileSize(batch.files.input) : '-' }} · out
                {{ fmtFileSize(batch.files.output) }}
              </dd>
            </div>
          </dl>
        </article>

        <article v-if="options.length" class="bd__card">
          <h3 class="bd__ct">Settings</h3>
          <dl class="bd__rows">
            <div v-for="[k, v] in options" :key="k" class="bd__row">
              <dt>{{ k }}</dt>
              <dd class="bd__wrap">{{ v }}</dd>
            </div>
          </dl>
        </article>
      </div>

      <article v-if="lineErrors.length" class="bd__card bd__errors">
        <h3 class="bd__ct">Why the batch failed</h3>
        <ul>
          <li v-for="(e, i) in lineErrors.slice(0, 50)" :key="i">
            <span v-if="e.line" class="c-mono">line {{ e.line }}:</span> {{ e.message }}
          </li>
        </ul>
        <p v-if="lineErrors.length > 50" class="bd__muted">
          and {{ (lineErrors.length - 50).toLocaleString() }} more
        </p>
      </article>

      <section class="bd__results">
        <Tabs v-model="kind" :tabs="resultTabs" />
        <div class="bd__card bd__list">
          <div v-if="results.items.length === 0" class="bd__empty bd__empty--list">
            {{
              loadingResults
                ? 'Loading'
                : kind === 'ok'
                  ? isActive(batch)
                    ? 'No results yet - they appear here as requests finish.'
                    : 'No request succeeded.'
                  : isActive(batch)
                    ? 'Nothing has failed so far.'
                    : 'Every request succeeded.'
            }}
          </div>
          <table v-else class="bd__table">
            <thead>
              <tr>
                <th>Request</th>
                <th>{{ kind === 'ok' ? 'Answer' : 'Error' }}</th>
                <th class="bd__num">Tokens</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="r in results.items"
                :key="r.offset"
                class="bd__rrow"
                tabindex="0"
                @click="show(r)"
                @keydown.enter="show(r)"
              >
                <td class="c-mono bd__cid">{{ r.custom_id }}</td>
                <td>
                  <div v-if="r.ok" class="bd__preview">{{ r.preview || '(empty)' }}</div>
                  <div v-else class="bd__preview bd__err">
                    <span v-if="r.status !== null" class="c-mono">{{ r.status }}</span>
                    {{ r.error }}
                  </div>
                </td>
                <td class="bd__num c-mono">
                  <template v-if="r.input_tokens || r.output_tokens">
                    {{ r.input_tokens.toLocaleString() }} in · {{ r.output_tokens.toLocaleString() }} out
                  </template>
                  <template v-else>-</template>
                </td>
              </tr>
            </tbody>
          </table>
          <div v-if="!results.end" class="bd__more">
            <button class="pk-btn pk-btn--ghost" :disabled="loadingResults" @click="loadResults(false)">
              Show more
            </button>
          </div>
        </div>
      </section>
    </template>

    <Dialog
      :open="!!open"
      :title="open ? open.item.custom_id : ''"
      size="lg"
      @close="open = null"
    >
      <pre v-if="open" class="bd__json">{{ open.json }}</pre>
      <template #footer>
        <button class="pk-btn pk-btn--ghost" @click="open && copied(open.json)">
          <Icon name="copy" :size="14" /> Copy
        </button>
        <button class="pk-btn pk-btn--primary" @click="open = null">Close</button>
      </template>
    </Dialog>

    <Dialog
      :open="!!confirm"
      role="alertdialog"
      icon="alert-triangle"
      :title="confirm === 'cancel' ? 'Cancel this batch?' : 'Delete this batch?'"
      :busy="acting"
      @close="confirm = null"
    >
      <p v-if="confirm === 'cancel'" class="bd__lead">
        Requests already on the model finish; the
        {{ batch ? batch.counts.pending.toLocaleString() : '' }} still waiting are marked
        cancelled. Everything that finished stays in the results.
      </p>
      <p v-else class="bd__lead">
        The batch and the result files it wrote are removed from this box. An input file
        uploaded through the files API stays until it is deleted there.
      </p>
      <template #footer>
        <button class="pk-btn pk-btn--ghost" :disabled="acting" @click="confirm = null">
          Keep it
        </button>
        <button class="pk-btn pk-btn--primary bd__danger-fill" :disabled="acting" @click="act">
          {{ confirm === 'cancel' ? 'Cancel batch' : 'Delete' }}
        </button>
      </template>
    </Dialog>
  </div>
</template>

<style scoped>
.bd {
  width: 100%;
  max-width: 1100px;
}
.bd__back {
  margin-bottom: 10px;
}
.bd__failure {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 10px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-danger);
}
.bd__empty {
  padding: 24px 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.bd__empty--list {
  padding: 18px 14px;
}
.bd__head {
  margin-bottom: 14px;
}
.bd__titlerow {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.bd__title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  word-break: break-all;
}
.bd__meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.bd__icon {
  display: inline-flex;
  padding: 2px;
  border: none;
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.bd__icon:hover {
  color: var(--pk-text-primary);
}
.bd__actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}
.bd__actions a {
  text-decoration: none;
}
.bd__spacer {
  flex: 1;
}
.bd__danger {
  color: var(--pk-text-danger);
}
.bd__danger-fill {
  background: var(--pk-status-error);
  border-color: var(--pk-status-error);
}
.bd__chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: var(--pk-radius-sm);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
.bd__chip--run {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.bd__chip--wait {
  background: var(--pk-bg-base);
  color: var(--pk-text-secondary);
}
.bd__chip--ok {
  background: var(--pk-status-success-subtle);
  color: var(--pk-status-success);
}
.bd__chip--warn {
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
}
.bd__chip--bad {
  background: var(--pk-status-error-subtle);
  color: var(--pk-status-error);
}
.bd__cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.bd__card {
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.bd__card--wide {
  grid-column: 1 / -1;
}
.bd__card--wide .bd__rows {
  max-width: 440px;
}
.bd__ct {
  margin: 0 0 10px;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.bd__counts {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  margin: 12px 0 6px;
}
.bd__count {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.bd__k {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.bd__v {
  font-size: var(--pk-font-size-lg);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.bd__warn {
  color: var(--pk-status-warning);
}
.bd__rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 8px 0 0;
}
.bd__row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: var(--pk-font-size-sm);
}
.bd__row dt {
  color: var(--pk-text-muted);
}
.bd__row dd {
  margin: 0;
  text-align: right;
  color: var(--pk-text-primary);
}
.bd__wrap {
  word-break: break-all;
}
.bd__muted {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
.bd__errors {
  margin-bottom: 14px;
}
.bd__errors ul {
  margin: 0;
  padding-left: 18px;
  font-size: var(--pk-font-size-sm);
}
.bd__results {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.bd__list {
  padding: 0;
  overflow-x: auto;
}
.bd__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.bd__table th {
  padding: 9px 12px;
  text-align: left;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-muted);
  border-bottom: 1px solid var(--pk-border-default);
}
.bd__table td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--pk-border-subtle);
  vertical-align: top;
}
.bd__rrow {
  cursor: pointer;
}
.bd__rrow:hover,
.bd__rrow:focus-visible {
  background: var(--pk-bg-hover);
  outline: none;
}
.bd__cid {
  white-space: nowrap;
  font-size: var(--pk-font-size-xs);
}
.bd__preview {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--pk-text-primary);
}
.bd__err {
  color: var(--pk-text-danger);
}
.bd__num {
  text-align: right;
  white-space: nowrap;
}
.bd__more {
  display: flex;
  justify-content: center;
  padding: 10px;
  border-top: 1px solid var(--pk-border-default);
}
.bd__json {
  margin: 0;
  max-height: 60vh;
  overflow: auto;
  padding: 12px;
  border-radius: var(--pk-radius-md, 6px);
  background: var(--pk-bg-inset);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  white-space: pre-wrap;
  word-break: break-word;
}
.bd__lead {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.c-mono {
  font-family: var(--pk-font-mono);
}
</style>
