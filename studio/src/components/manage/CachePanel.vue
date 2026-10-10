<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { cacheApi, type CacheServer } from '@/lib/api'
import { fmtBytes } from '@/lib/format'
import Icon from '@/components/Icon.vue'
import Progress from '@/components/ui/Progress.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { useFleetStore } from '@/stores/fleet'

const fleet = useFleetStore()
const rows = ref<CacheServer[]>([])
const loaded = ref(false)
let timer: number | null = null

async function refresh(): Promise<void> {
  try {
    rows.value = (await cacheApi.get()).servers
  } catch {
    rows.value = []
  }
  loaded.value = true
}

let release: (() => void) | null = null
onMounted(() => {
  release = fleet.hold()
  void refresh()
  timer = window.setInterval(() => void refresh(), 3000)
})
onUnmounted(() => {
  if (timer !== null) window.clearInterval(timer)
  release?.()
})

function pct(n: number, of: number): number {
  return of > 0 ? Math.round((n / of) * 100) : 0
}

/** bytes/us -> GB/s. The engine measures in the unit it schedules in. */
function gbs(bpus: number): string {
  return bpus > 0 ? `${(bpus / 1000).toFixed(1)} GB/s` : '-'
}

function name(port: number, model: string | null): string {
  return fleet.rows.find((r) => r.port === port)?.display ?? model ?? String(port)
}

/**
 * The alarm, restated for a reader: repeats keep missing on prefixes this
 * cache itself dropped. Mirrors TierDecisions::ghost_alarm in the engine -
 * both sides must agree on what counts as thrashing, so the threshold lives
 * in one sentence in two places rather than two different thresholds.
 */
function thrashing(s: CacheServer): boolean {
  const t = s.tier
  return t.miss_ghost >= 16 && t.miss_ghost * 4 > t.lookups
}

const anyArmed = computed(() => rows.value.length > 0)
</script>

<template>
  <section class="ch">
    <div v-if="!loaded" class="ch__empty">Loading</div>

    <div v-else-if="!anyArmed" class="ch__empty">
      <Icon name="hard-drive" :size="18" />
      <p>No model on this box has KV offloading switched on.</p>
    </div>

    <div v-for="s in rows" v-else :key="s.port" class="ch__server">
      <header class="ch__hd">
        <span class="ch__port">{{ s.port }}</span>
        <span class="ch__name">{{ name(s.port, s.model) }}</span>
        <span v-if="s.tier.tripped" class="ch__flag ch__flag--bad">
          <Icon name="alert-triangle" :size="12" /> Cache offline
        </span>
        <span v-else-if="thrashing(s)" class="ch__flag ch__flag--warn">
          <Icon name="alert-triangle" :size="12" /> Too small for this workload
        </span>
      </header>

      <div class="ch__cards">
        <article class="ch__card">
          <h3 class="ch__ct">What it decided</h3>
          <dl class="ch__rows">
            <div class="ch__row">
              <dt>Prompts checked</dt>
              <dd class="c-mono">{{ s.tier.lookups.toLocaleString() }}</dd>
            </div>
            <div class="ch__row">
              <dt>Found in cache</dt>
              <dd class="c-mono">
                {{ s.tier.hits.toLocaleString() }}
                <span class="ch__sub">{{ pct(s.tier.hits, s.tier.lookups) }}%</span>
              </dd>
            </div>
            <div class="ch__row">
              <dt>Reused</dt>
              <dd class="c-mono">{{ s.tier.elected_restore.toLocaleString() }}</dd>
            </div>
            <div class="ch__row">
              <dt>
                <Tooltip label="A hit the cost model turned down: rebuilding the prefix on the GPU was predicted to be faster than reading it back.">
                  <span>Rebuilt instead</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">{{ s.tier.elected_recompute.toLocaleString() }}</dd>
            </div>
            <div class="ch__row">
              <dt>Delivered</dt>
              <dd class="c-mono">{{ s.tier.resolved.toLocaleString() }}</dd>
            </div>
            <div v-if="s.tier.abandoned > 0" class="ch__row">
              <dt>Gave up</dt>
              <dd class="c-mono c-warn">{{ s.tier.abandoned.toLocaleString() }}</dd>
            </div>
            <div v-if="s.tier.park_refused > 0" class="ch__row">
              <dt>
                <Tooltip label="A reuse that could not start because the GPU block pool had nowhere to put it. A capacity story, not a disk one.">
                  <span>No room to land</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">{{ s.tier.park_refused.toLocaleString() }}</dd>
            </div>
          </dl>
        </article>

        <article class="ch__card">
          <h3 class="ch__ct">Why the rest missed</h3>
          <dl class="ch__rows">
            <div class="ch__row">
              <dt>Never cached</dt>
              <dd class="c-mono">{{ s.tier.miss_cold.toLocaleString() }}</dd>
            </div>
            <div class="ch__row">
              <dt>Already on the GPU</dt>
              <dd class="c-mono">{{ s.tier.miss_no_new_tokens.toLocaleString() }}</dd>
            </div>
            <div class="ch__row" :class="{ 'ch__row--warn': thrashing(s) }">
              <dt>Dropped to make room</dt>
              <dd class="c-mono">{{ s.tier.miss_ghost.toLocaleString() }}</dd>
            </div>
            <div v-if="(s.tier.miss_no_state ?? 0) > 0" class="ch__row">
              <dt>
                <Tooltip label="The prompt's attention cache was held, but not the model state this model also needs to pick up where the prompt left off. Hybrid models (Qwen 3.5 and later, Nemotron, Gemma 4) resume only where that state was saved.">
                  <span>No saved state</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">{{ (s.tier.miss_no_state ?? 0).toLocaleString() }}</dd>
            </div>
            <div v-if="s.tier.miss_tripped > 0" class="ch__row">
              <dt>Cache offline</dt>
              <dd class="c-mono c-warn">{{ s.tier.miss_tripped.toLocaleString() }}</dd>
            </div>
          </dl>
          <p v-if="thrashing(s)" class="ch__note">
            Raise <code>ram_gb</code> (or add <code>nvme_gb</code>) for this model.
          </p>
        </article>

        <article class="ch__card">
          <h3 class="ch__ct">Where it lives</h3>
          <div class="ch__meter">
            <span class="ch__k">Memory</span>
            <Progress
              :value="s.tier.ram_ready"
              :max="s.tier.ram_capacity"
              :label="`${fmtBytes(s.tier.ram_ready)} of ${fmtBytes(s.tier.ram_capacity)}`"
              size="md"
            />
            <span class="ch__v c-mono">
              {{ fmtBytes(s.tier.ram_ready) }} / {{ fmtBytes(s.tier.ram_capacity) }}
            </span>
          </div>
          <div v-if="s.tier.disk_capacity > 0" class="ch__meter">
            <span class="ch__k">Disk</span>
            <Progress
              :value="s.tier.disk_ready"
              :max="s.tier.disk_capacity"
              :label="`${fmtBytes(s.tier.disk_ready)} of ${fmtBytes(s.tier.disk_capacity)}`"
              size="md"
            />
            <span class="ch__v c-mono">
              {{ fmtBytes(s.tier.disk_ready) }} / {{ fmtBytes(s.tier.disk_capacity) }}
            </span>
          </div>
          <dl class="ch__rows">
            <div class="ch__row">
              <dt>Prefixes kept</dt>
              <dd class="c-mono">{{ s.tier.resident_runs.toLocaleString() }}</dd>
            </div>
            <div v-if="s.tier.disk_capacity > 0" class="ch__row">
              <dt>
                <Tooltip label="Prefixes pushed out of memory that kept their content because a copy was already on disk.">
                  <span>Moved to disk</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">{{ s.tier.promoted_to_disk.toLocaleString() }}</dd>
            </div>
            <div v-if="s.tier.disk_capacity > 0" class="ch__row">
              <dt>Written to disk today</dt>
              <dd class="c-mono">{{ fmtBytes(s.tier.disk_written_today) }}</dd>
            </div>
            <div v-if="s.tier.pending_durable_writes > 0" class="ch__row">
              <dt>
                <Tooltip label="Waiting for the disk to be free of reads. Writing beside a read costs most of the read bandwidth on consumer drives, so writes wait.">
                  <span>Queued for disk</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">{{ s.tier.pending_durable_writes.toLocaleString() }}</dd>
            </div>
          </dl>
        </article>

        <article class="ch__card">
          <h3 class="ch__ct">How well it predicts</h3>
          <dl class="ch__rows">
            <div class="ch__row">
              <dt>From memory</dt>
              <dd class="c-mono">{{ gbs(s.tier.rate_ram_bpus) }}</dd>
            </div>
            <div v-if="s.tier.disk_capacity > 0" class="ch__row">
              <dt>From disk</dt>
              <dd class="c-mono">{{ gbs(s.tier.rate_disk_bpus) }}</dd>
            </div>
            <div class="ch__row">
              <dt>
                <Tooltip label="How far the cost model's predicted reuse time lands from the measured one. It decides reuse-vs-rebuild with these predictions.">
                  <span>Prediction error</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">
                {{ s.tier.prediction_error_pct === null ? '-' : `${s.tier.prediction_error_pct.toFixed(0)}%` }}
              </dd>
            </div>
            <div class="ch__row">
              <dt>
                <Tooltip label="Bytes moved per byte actually delivered. Above 1 means reuses were abandoned after their read was already paid for.">
                  <span>Wasted reads</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">
                {{ s.tier.useful_bytes > 0 ? `${(s.tier.moved_bytes / s.tier.useful_bytes).toFixed(2)}x` : '-' }}
              </dd>
            </div>
            <div class="ch__row">
              <dt>Prefix delivered</dt>
              <dd class="c-mono">{{ fmtBytes(s.tier.useful_bytes) }}</dd>
            </div>
            <div v-if="s.tier.disk_read_gbs > 0" class="ch__row">
              <dt>
                <Tooltip label="Measured on this machine's drive when the cache opened, not a spec sheet.">
                  <span>Drive speed</span>
                </Tooltip>
              </dt>
              <dd class="c-mono">
                {{ s.tier.disk_read_gbs.toFixed(1) }} / {{ s.tier.disk_write_gbs.toFixed(1) }} GB/s
              </dd>
            </div>
            <div v-if="s.tier.io_failures > 0 || s.tier.integrity_failures > 0" class="ch__row">
              <dt>Read failures</dt>
              <dd class="c-mono c-warn">
                {{ (s.tier.io_failures + s.tier.integrity_failures).toLocaleString() }}
              </dd>
            </div>
          </dl>
        </article>
      </div>
    </div>
  </section>
</template>

<style scoped>
.ch__empty {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 32px 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.ch__empty p {
  margin: 0;
}
.ch__server {
  margin-bottom: 18px;
}
.ch__hd {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.ch__port {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  font-weight: 600;
}
.ch__name {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.ch__flag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
  padding: 2px 8px;
  border-radius: var(--pk-radius-sm);
  font-size: var(--pk-font-size-xs);
}
.ch__flag--warn {
  background: var(--pk-warn-bg, rgb(180 120 0 / 15%));
  color: var(--pk-warn, #b47800);
}
.ch__flag--bad {
  background: var(--pk-danger-bg, rgb(200 60 60 / 15%));
  color: var(--pk-danger, #c83c3c);
}
.ch__cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 10px;
}
.ch__card {
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.ch__ct {
  margin: 0 0 10px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--pk-text-muted);
}
.ch__rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
}
.ch__row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  font-size: var(--pk-font-size-sm);
}
.ch__row dt {
  color: var(--pk-text-secondary);
}
.ch__row dd {
  margin: 0;
  color: var(--pk-text-primary);
  font-variant-numeric: tabular-nums;
}
.ch__row--warn dd {
  color: var(--pk-warn, #b47800);
  font-weight: 600;
}
.ch__sub {
  margin-left: 6px;
  color: var(--pk-text-muted);
}
.ch__meter {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 4px 10px;
  margin-bottom: 10px;
  font-size: var(--pk-font-size-sm);
}
.ch__k {
  color: var(--pk-text-secondary);
}
.ch__v {
  grid-column: 2;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.ch__note {
  margin: 10px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.ch__note code {
  font-family: var(--pk-font-mono);
}
</style>
