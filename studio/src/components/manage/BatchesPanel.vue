<script setup lang="ts">
// Batches - every batch the manager is running or has run, both vendor
// dialects in one list (the manager runs them against the models it serves;
// see the manager's batch module). This page WATCHES: a batch is submitted
// through the vendor APIs (`/v1/batches`, `/v1/messages/batches`) by a
// script or an SDK, and the person who started it, or the person who owns
// the box, comes here to see how far it is, what is failing and when it will
// be done. A row opens the batch (results, files, cancel).
//
// A days-long run is the normal case, so the list refreshes itself: every
// 3 s while anything is active, every 15 s once nothing is.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  averagePerMinute,
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
  type BatchSummary,
  type ListState,
} from '@/lib/batches'
import { fmtEtaShort, fmtStamp } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import Progress from '@/components/ui/Progress.vue'
import Tabs, { type TabOption } from '@/components/ui/Tabs.vue'

const router = useRouter()
const state = ref<ListState>('all')
const rows = ref<BatchSummary[]>([])
const hasMore = ref(false)
const loaded = ref(false)
const failure = ref('')
let timer: number | null = null

const tabs: TabOption[] = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'Active' },
  { value: 'ended', label: 'Ended' },
]

/** Refresh what is on screen: the first page, plus however many older pages
 *  the person already loaded (so a refresh never shrinks their list). */
async function refresh(): Promise<void> {
  try {
    const want = Math.max(50, rows.value.length)
    const page = await batchesApi.list(state.value, undefined, Math.min(want, 200))
    rows.value = page.data
    hasMore.value = page.has_more
    failure.value = ''
  } catch (e) {
    failure.value = e instanceof Error ? e.message : String(e)
  }
  loaded.value = true
  schedule()
}

async function more(): Promise<void> {
  const last = rows.value[rows.value.length - 1]
  if (!last) return
  try {
    const page = await batchesApi.list(state.value, last.id)
    rows.value = [...rows.value, ...page.data]
    hasMore.value = page.has_more
  } catch (e) {
    failure.value = e instanceof Error ? e.message : String(e)
  }
}

function schedule(): void {
  if (timer !== null) window.clearTimeout(timer)
  const busy = rows.value.some(isActive)
  timer = window.setTimeout(() => void refresh(), busy ? 3000 : 15000)
}

watch(state, () => {
  rows.value = []
  loaded.value = false
  void refresh()
})
onMounted(() => void refresh())
onUnmounted(() => {
  if (timer !== null) window.clearTimeout(timer)
})

const activeCount = computed(() => rows.value.filter(isActive).length)

function open(b: BatchSummary): void {
  void router.push({ name: 'batch-detail', params: { id: b.id } })
}

function pct(b: BatchSummary): number {
  return b.counts.total > 0 ? Math.round((settled(b) / b.counts.total) * 100) : 0
}

/** Live rate while it runs, the whole run's average once it has ended. */
function rate(b: BatchSummary): string {
  return fmtPerMinute(isActive(b) ? b.progress?.rate_per_minute : averagePerMinute(b))
}

/** Time left while it runs; how long it ran once it has ended. */
function time(b: BatchSummary): string {
  if (isActive(b)) {
    const s = b.progress?.eta_seconds
    return s !== undefined && s > 0 ? `${fmtEtaShort(s)} left` : '-'
  }
  if (b.in_progress_ms === null || b.ended_ms === null) return '-'
  return `ran ${fmtSpan(b.ended_ms - b.in_progress_ms)}`
}
</script>

<template>
  <div class="bt">
    <div class="bt__head">
      <div>
        <h1 class="bt__title">Batches</h1>
        <p class="bt__lead">
          Work submitted to this box through the OpenAI and Anthropic batch APIs, run
          against the models it serves - waiting out restarts and stepping aside for
          live traffic.
          <template v-if="activeCount > 0">
            {{ activeCount === 1 ? 'One batch is' : `${activeCount} batches are` }} running.
          </template>
        </p>
      </div>
    </div>

    <Tabs v-model="state" :tabs="tabs" class="bt__tabs" />

    <p v-if="failure" class="bt__failure">
      <Icon name="alert-triangle" :size="14" /> {{ failure }}
    </p>

    <div v-if="!loaded" class="bt__empty">Loading</div>

    <div v-else-if="rows.length === 0" class="bt__empty bt__empty--intro">
      <Icon name="scroll" :size="20" />
      <div>
        <p v-if="state === 'active'">No batch is running.</p>
        <p v-else-if="state === 'ended'">No batch has ended yet.</p>
        <template v-else>
          <p>No batches yet.</p>
          <p class="bt__hint">
            Point an OpenAI or Anthropic SDK at this box's address and use its batch calls -
            upload a JSONL file and create a batch at <code>/v1/batches</code>, or send
            the requests to <code>/v1/messages/batches</code>. They show up here as soon
            as they are accepted.
          </p>
        </template>
      </div>
    </div>

    <div v-else class="bt__card">
      <table class="bt__table">
        <thead>
          <tr>
            <th>Batch</th>
            <th>Model</th>
            <th>Status</th>
            <th class="bt__progcol">Progress</th>
            <th class="bt__num">Rate</th>
            <th class="bt__num">Time</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="b in rows"
            :key="b.id"
            class="bt__row"
            tabindex="0"
            @click="open(b)"
            @keydown.enter="open(b)"
          >
            <td>
              <div class="bt__name">{{ batchName(b) }}</div>
              <div class="bt__sub">
                {{ endpointLabel(b.endpoint) }} ·
                {{ b.dialect === 'openai' ? 'OpenAI' : 'Anthropic' }}
              </div>
            </td>
            <td class="bt__model">{{ b.model }}</td>
            <td>
              <span class="bt__chip" :class="`bt__chip--${statusLabel(b).tone}`">
                {{ statusLabel(b).label }}
              </span>
            </td>
            <td class="bt__progcol">
              <Progress
                :value="settled(b)"
                :max="Math.max(1, b.counts.total)"
                :tone="failed(b) > 0 ? 'warning' : 'accent'"
                :label="`${settled(b)} of ${b.counts.total} done`"
                size="sm"
              />
              <div class="bt__sub c-mono">
                {{ settled(b).toLocaleString() }} / {{ b.counts.total.toLocaleString() }}
                <span v-if="b.counts.total > 0">({{ pct(b) }}%)</span>
                <span v-if="failed(b) > 0" class="bt__bad">
                  · {{ failed(b).toLocaleString() }} {{ unsuccessfulLabel(b).toLowerCase() }}
                </span>
              </div>
            </td>
            <td class="bt__num c-mono">{{ rate(b) }}</td>
            <td class="bt__num c-mono">{{ time(b) }}</td>
            <td class="bt__sub c-mono">{{ fmtStamp(new Date(b.created_ms)).slice(0, 16) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-if="hasMore" class="bt__more">
        <button class="pk-btn pk-btn--ghost" @click="more">Show older batches</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bt {
  width: 100%;
  max-width: 1100px;
}
.bt__head {
  margin-bottom: 14px;
}
.bt__title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.bt__lead {
  margin: 6px 0 0;
  max-width: 75ch;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  line-height: 1.5;
}
.bt__tabs {
  margin-bottom: 12px;
}
.bt__failure {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 10px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-danger);
}
.bt__empty {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 28px 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.bt__empty--intro {
  align-items: flex-start;
}
.bt__empty p {
  margin: 0 0 6px;
}
.bt__hint {
  max-width: 70ch;
  line-height: 1.5;
}
.bt__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  overflow-x: auto;
}
.bt__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.bt__table th {
  padding: 9px 12px;
  text-align: left;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-muted);
  border-bottom: 1px solid var(--pk-border-default);
  white-space: nowrap;
}
.bt__table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--pk-border-subtle);
  vertical-align: middle;
  color: var(--pk-text-primary);
}
.bt__row {
  cursor: pointer;
}
.bt__row:hover,
.bt__row:focus-visible {
  background: var(--pk-bg-hover);
  outline: none;
}
.bt__row:last-child td {
  border-bottom: none;
}
.bt__name {
  font-weight: 600;
}
.bt__sub {
  margin-top: 3px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.bt__model {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
.bt__progcol {
  min-width: 170px;
}
/* the progress note may wrap - "120 / 120 (100%) · 84 cancelled" must not
   push the created column off the card */
.bt__progcol .bt__sub {
  white-space: normal;
}
.bt__num {
  text-align: right;
  white-space: nowrap;
}
.bt__bad {
  color: var(--pk-status-warning);
  white-space: nowrap;
}
.bt__chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: var(--pk-radius-sm);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
}
.bt__chip--run {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.bt__chip--wait {
  background: var(--pk-bg-base);
  color: var(--pk-text-secondary);
}
.bt__chip--ok {
  background: var(--pk-status-success-subtle);
  color: var(--pk-status-success);
}
.bt__chip--warn {
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
}
.bt__chip--bad {
  background: var(--pk-status-error-subtle);
  color: var(--pk-status-error);
}
.c-mono {
  font-family: var(--pk-font-mono);
}
.bt__more {
  display: flex;
  justify-content: center;
  padding: 10px;
  border-top: 1px solid var(--pk-border-default);
}
</style>
