<script setup lang="ts">
// Embeddings & rerank - the Studio's encoder surface. The MODEL decides what
// this page is: a reranker shows rerank, an embedder shows embeddings - no
// tab for a capability the served model doesn't have ("I
// installed a reranker, why is an embeddings tab present"). Inputs start
// empty with placeholder guidance - the user's content, never sample prose.
// Everything runs through the same /v1 endpoints code calls (relayed by the
// manager, which holds the runner key); each run shows the equivalent curl.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useModelsStore } from '@/stores/models'
import { modelLabel } from '@/lib/model-name'
import Icon from '@/components/Icon.vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'

const models = useModelsStore()

// keep the encoder list fresh while the panel is open (models started or
// stopped in the Manager appear without a reload)
let timer: number | undefined
onMounted(() => {
  void models.refresh()
  timer = window.setInterval(() => void models.refresh(), 5000)
})
onUnmounted(() => clearInterval(timer))

const encoders = computed(() => models.models.filter((m) => m.kind === 'encoder'))
const port = ref<number>(0)
watch(
  encoders,
  (list) => {
    if (!list.some((m) => m.port === port.value)) port.value = list[0]?.port ?? 0
  },
  { immediate: true },
)
const current = computed(() => encoders.value.find((m) => m.port === port.value))
const options = computed<SelectOption[]>(() =>
  encoders.value.map((m) => ({
    value: m.port ?? 0,
    label: m.display ?? modelLabel(m.id),
    hint: `port ${m.port}`,
    vendor: m.vendor,
    title: m.id,
  })),
)

// The endpoint's capability picks the MODE (/api/server relay: `reranker` is
// functional truth - the vocab either carries the yes/no judge tokens or it
// doesn't). null = capability not known yet.
const mode = ref<'rerank' | 'embed' | null>(null)
// What an item may carry beside text (`embedder_inputs`: EmbeddingGemma 2's
// towers, as the endpoint built them). Pictures and clips are offered only
// when the endpoint takes them.
const inputs = ref<string[]>(['text'])
watch(
  port,
  async (p) => {
    mode.value = null
    inputs.value = ['text']
    media.value = []
    if (!p) return
    try {
      const r = await fetch(`/api/runners/${p}/server`)
      const j = r.ok
        ? ((await r.json()) as { reranker?: boolean | null; embedder_inputs?: string[] | null })
        : null
      mode.value = j?.reranker ? 'rerank' : 'embed'
      inputs.value = j?.embedder_inputs ?? ['text']
    } catch {
      mode.value = 'embed' // embeddings is the capability every encoder has
    }
  },
  { immediate: true },
)
const takesImages = computed(() => inputs.value.includes('image'))
const takesAudio = computed(() => inputs.value.includes('audio'))
const title = computed(() =>
  mode.value === 'rerank' ? 'Rerank' : mode.value === 'embed' ? 'Embeddings' : 'Embeddings & rerank',
)
const lead = computed(() =>
  mode.value === 'rerank'
    ? 'Score your documents against a query - the same /v1/rerank your code calls.'
    : mode.value === 'embed'
      ? 'Compare your texts by meaning - the same /v1/embeddings your code calls.'
      : 'Try a running embedding or reranking model - the same /v1 endpoints your code calls.',
)

// ── the user's content: empty until they provide it ─────────────────────────
const query = ref('')
const docsText = ref('')
const embedText = ref('')

/** A picture or clip the user added, embedded as its own item beside the
 *  text lines - so every pair, text to picture included, is compared. */
interface MediaItem {
  kind: 'image' | 'audio'
  name: string
  /** the file as a data: URL (a picture's preview, a clip's bytes) */
  url: string
}
const media = ref<MediaItem[]>([])
const fileInput = ref<HTMLInputElement | null>(null)
const pickKind = ref<'image' | 'audio'>('image')
function pick(kind: 'image' | 'audio'): void {
  pickKind.value = kind
  if (fileInput.value) {
    fileInput.value.accept = kind === 'image' ? 'image/*' : 'audio/*'
    fileInput.value.value = ''
    fileInput.value.click()
  }
}
function onFiles(e: Event): void {
  const files = (e.target as HTMLInputElement).files
  if (!files) return
  const kind = pickKind.value
  for (const f of Array.from(files)) {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string')
        media.value = [...media.value, { kind, name: f.name, url: reader.result }]
    }
    reader.readAsDataURL(f)
  }
}
function removeMedia(i: number): void {
  media.value = media.value.filter((_, j) => j !== i)
}
/** The /v1/embeddings item for a picture or clip: the chat content-part
 *  shape. A clip rides `input_audio` (base64 + the file's extension; the
 *  server reads the container from the bytes). */
function mediaItem(m: MediaItem): unknown {
  if (m.kind === 'image') return { content: [{ type: 'image_url', image_url: { url: m.url } }] }
  const data = m.url.slice(m.url.indexOf(',') + 1)
  const format = m.name.includes('.') ? m.name.split('.').pop()!.toLowerCase() : 'wav'
  return { content: [{ type: 'input_audio', input_audio: { data, format } }] }
}

interface RankedDoc {
  index: number
  relevance_score: number
  document?: string
}
// Degenerate results render as cards, not tables: one reranked document has
// no ranking (the score is the result), one embedded text has no pair (the
// vector is the result). The tables start meaning something at n >= 2.
const ranked = ref<RankedDoc[] | null>(null)
/** `texts` labels every row (a picture or clip by its file name); `kinds`
 *  says which is which. */
const embedded = ref<{
  texts: string[]
  kinds: ('text' | 'image' | 'audio')[]
  vectors: number[][]
  dims: number
} | null>(null)

const busy = ref(false)
const error = ref<string | null>(null)
const lastMs = ref<number | null>(null)
const lastTokens = ref<number | null>(null)

function lines(s: string): string[] {
  return s
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
}
const canRun = computed(() => {
  if (!current.value || busy.value) return false
  if (mode.value === 'rerank') return query.value.trim().length > 0 && lines(docsText.value).length > 0
  if (mode.value === 'embed') return lines(embedText.value).length + media.value.length > 0
  return false
})

async function post(path: string, body: unknown): Promise<unknown> {
  const t0 = performance.now()
  const res = await fetch(`/api/runners/${port.value}/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const json: unknown = await res.json().catch(() => null)
  lastMs.value = Math.round(performance.now() - t0)
  if (!res.ok) {
    const msg =
      (json as { error?: { message?: string } } | null)?.error?.message ?? `HTTP ${res.status}`
    throw new Error(msg)
  }
  return json
}

async function run(): Promise<void> {
  if (!canRun.value || !current.value) return
  busy.value = true
  error.value = null
  try {
    if (mode.value === 'rerank') {
      const r = (await post('v1/rerank', {
        model: current.value.id,
        query: query.value,
        documents: lines(docsText.value),
        return_documents: true,
      })) as { results: RankedDoc[]; usage?: { total_tokens?: number } }
      ranked.value = r.results
      lastTokens.value = r.usage?.total_tokens ?? null
    } else {
      const texts = lines(embedText.value)
      const items = media.value
      const r = (await post('v1/embeddings', {
        model: current.value.id,
        input: [...texts, ...items.map(mediaItem)],
      })) as {
        data: { embedding: number[]; index: number }[]
        usage?: { total_tokens?: number }
      }
      const vectors = [...r.data].sort((a, b) => a.index - b.index).map((d) => d.embedding)
      embedded.value = {
        texts: [...texts, ...items.map((m) => m.name)],
        kinds: [...texts.map(() => 'text' as const), ...items.map((m) => m.kind)],
        vectors,
        dims: vectors[0]?.length ?? 0,
      }
      lastTokens.value = r.usage?.total_tokens ?? null
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
function onKey(e: KeyboardEvent): void {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') void run()
}

function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const d = Math.sqrt(na) * Math.sqrt(nb)
  return d === 0 ? 0 : dot / d
}
const sim = computed(() => {
  const e = embedded.value
  if (!e) return null
  return e.vectors.map((a) => e.vectors.map((b) => cosine(a, b)))
})
const closest = computed(() => {
  const s = sim.value
  if (!s || s.length < 2) return null
  let best: { i: number; j: number; v: number } | null = null
  for (let i = 0; i < s.length; i++)
    for (let j = i + 1; j < s.length; j++)
      if (!best || s[i][j] > best.v) best = { i, j, v: s[i][j] }
  return best
})
/** first values of a lone embedding - the single-text result shows the
 *  vector itself, since there is nothing to compare yet */
const vecPreview = computed(() => {
  const v = embedded.value?.vectors[0]
  if (!v) return ''
  return v.slice(0, 6).map((x) => x.toFixed(3)).join(', ') + (v.length > 6 ? ', ...' : '')
})

function heat(v: number): string {
  const a = Math.max(0, Math.min(1, v)) ** 2 * 0.55
  return `rgba(52, 168, 83, ${a.toFixed(3)})`
}
function fmtScore(v: number): string {
  return v.toFixed(3)
}

/** the equivalent curl for this mode - teach by example */
const curl = computed(() => {
  if (!current.value) return ''
  const endpoint = `http://localhost:${port.value}`
  if (mode.value === 'rerank') {
    return [
      `curl ${endpoint}/v1/rerank \\`,
      `  -H "Content-Type: application/json" \\`,
      `  -H "Authorization: Bearer <api key>" \\`,
      `  -d '{"model": "${current.value.id}", "query": "...", "documents": ["...", "..."]}'`,
    ].join('\n')
  }
  // a picture or clip is a content-part item beside the strings
  const items = ['"..."']
  if (media.value.some((m) => m.kind === 'image'))
    items.push('{"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}')
  if (media.value.some((m) => m.kind === 'audio'))
    items.push('{"content": [{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}]}')
  if (items.length === 1) items.push('"..."')
  return [
    `curl ${endpoint}/v1/embeddings \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "Authorization: Bearer <api key>" \\`,
    `  -d '{"model": "${current.value.id}", "input": [${items.join(', ')}]}'`,
  ].join('\n')
})
</script>

<template>
  <div class="pg" @keydown="onKey">
    <div class="pg__head">
      <div>
        <h1 class="pg__title">{{ title }}</h1>
        <p class="pg__lead">{{ lead }}</p>
      </div>
      <Select v-if="encoders.length" v-model="port" :options="options" />
    </div>

    <div v-if="!encoders.length && !models.loading" class="pg__none">
      <Icon name="sliders" :size="32" class="pg__none-icon" />
      <p class="pg__none-title">No embedding or reranking model is running</p>
      <p class="pg__none-txt">
        Start one in the Manager - the catalog's embedding and reranker models all serve here.
      </p>
      <RouterLink class="pk-btn pk-btn--primary" :to="{ name: 'server-new' }">
        <Icon name="play" :size="14" /> Start a model
      </RouterLink>
    </div>

    <template v-else-if="mode">
      <div v-if="error" class="pg__error">{{ error }}</div>

      <div class="pg__form">
        <template v-if="mode === 'rerank'">
          <label class="pg__field">
            <span class="pg__label">Query</span>
            <input
              v-model="query"
              class="pk-input"
              type="text"
              placeholder="What are you looking for?"
            />
          </label>
          <label class="pg__field">
            <span class="pg__label">Documents</span>
            <textarea
              v-model="docsText"
              class="pk-input pg__ta"
              rows="8"
              spellcheck="false"
              placeholder="Paste your documents, one per line - each line is scored against the query"
            />
          </label>
        </template>
        <template v-else>
          <label class="pg__field">
            <span class="pg__label">Texts</span>
            <textarea
              v-model="embedText"
              class="pk-input pg__ta"
              rows="8"
              spellcheck="false"
              :placeholder="
                takesImages || takesAudio
                  ? 'Paste your texts, one per line - every pair, pictures and clips included, is compared by meaning'
                  : 'Paste your texts, one per line - every pair is compared by meaning'
              "
            />
          </label>
          <div v-if="takesImages || takesAudio" class="pg__media">
            <input ref="fileInput" type="file" multiple hidden @change="onFiles" />
            <button v-if="takesImages" class="pk-btn" type="button" @click="pick('image')">
              <Icon name="image" :size="14" /> Add pictures
            </button>
            <button v-if="takesAudio" class="pk-btn" type="button" @click="pick('audio')">
              <Icon name="microphone" :size="14" /> Add audio clips
            </button>
            <span v-if="takesAudio" class="pg__runhint">clips up to 30 s</span>
            <ul v-if="media.length" class="pg__chips">
              <li v-for="(m, i) in media" :key="i" class="pg__chip">
                <img v-if="m.kind === 'image'" :src="m.url" alt="" class="pg__thumb" />
                <Icon v-else name="microphone" :size="14" />
                <span class="pg__chipname">{{ m.name }}</span>
                <button class="pg__chipx" type="button" :aria-label="`Remove ${m.name}`" @click="removeMedia(i)">
                  <Icon name="x" :size="12" />
                </button>
              </li>
            </ul>
          </div>
        </template>
        <div class="pg__actions">
          <button class="pk-btn pk-btn--primary" :disabled="!canRun" @click="run">
            <Icon :name="busy ? 'spinner' : 'play'" :size="14" :class="{ spin: busy }" />
            Run
          </button>
          <span class="pg__runhint">Ctrl+Enter</span>
          <span v-if="lastMs !== null" class="pg__meta">
            {{ lastMs }} ms<template v-if="lastTokens !== null">
              · {{ lastTokens }} tokens</template
            >
          </span>
        </div>
      </div>

      <div v-if="mode === 'rerank' && ranked && ranked.length === 1" class="pg__one">
        <p class="pg__one-line">
          Relevance to your query: <strong>{{ fmtScore(ranked[0].relevance_score) }}</strong>
        </p>
        <p class="pg__doc">{{ ranked[0].document }}</p>
        <p class="pg__one-hint">Add more documents to rank them against each other.</p>
      </div>

      <div v-else-if="mode === 'rerank' && ranked" class="pg__tablewrap">
        <table class="pg__table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Score</th>
              <th>Was</th>
              <th>Document</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, pos) in ranked" :key="r.index">
              <td class="c-num">{{ pos + 1 }}</td>
              <td class="c-num">
                <div class="pg__score">
                  <div class="pg__scorebar" :style="{ width: `${Math.max(3, r.relevance_score * 100)}%` }" />
                  {{ fmtScore(r.relevance_score) }}
                </div>
              </td>
              <td class="c-num c-dim">#{{ r.index + 1 }}</td>
              <td class="pg__doc">{{ r.document ?? `document ${r.index + 1}` }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-else-if="mode === 'embed' && embedded && embedded.texts.length < 2" class="pg__one">
        <p class="pg__one-line">
          Embedded <strong>1 {{ embedded.kinds[0] === 'image' ? 'picture' : embedded.kinds[0] === 'audio' ? 'clip' : 'text' }}</strong>
          · {{ embedded.dims }} dimensions
        </p>
        <code class="pg__one-vec">[{{ vecPreview }}]</code>
        <p class="pg__one-hint">
          Add more {{ takesImages || takesAudio ? 'lines, pictures or clips' : 'lines' }} to compare them by meaning.
        </p>
      </div>

      <div v-else-if="mode === 'embed' && embedded" class="pg__tablewrap">
        <table class="pg__table">
          <thead>
            <tr>
              <th></th>
              <th v-for="(_, j) in embedded.texts" :key="j" class="c-num">{{ j + 1 }}</th>
              <th>{{ embedded.kinds.some((k) => k !== 'text') ? 'Item' : 'Text' }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in sim" :key="i">
              <th class="c-num">{{ i + 1 }}</th>
              <td
                v-for="(v, j) in row"
                :key="j"
                class="c-num"
                :style="i === j ? undefined : { background: heat(v) }"
                :class="{ 'c-dim': i === j }"
              >
                {{ i === j ? '-' : fmtScore(v) }}
              </td>
              <td class="pg__doc">
                <Icon v-if="embedded.kinds[i] === 'image'" name="image" :size="13" class="pg__kind" />
                <Icon v-else-if="embedded.kinds[i] === 'audio'" name="microphone" :size="13" class="pg__kind" />
                {{ embedded.texts[i] }}
              </td>
            </tr>
          </tbody>
        </table>
        <p v-if="closest" class="pg__closest">
          Most similar: <strong>{{ closest.i + 1 }}</strong> and
          <strong>{{ closest.j + 1 }}</strong> ({{ fmtScore(closest.v) }}) ·
          {{ embedded.dims }} dimensions per embedding
        </p>
      </div>

      <Collapsible class="pg__api" summary="API call">
        <pre>{{ curl }}</pre>
      </Collapsible>
    </template>
  </div>
</template>

<style scoped>
.pg {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.pg__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.pg__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.pg__lead {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.pg__none {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 64px 24px;
  text-align: center;
}
.pg__media {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.pg__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  width: 100%;
  margin: 2px 0 0;
  padding: 0;
  list-style: none;
}
.pg__chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 240px;
  padding: 3px 4px 3px 6px;
  border: 1px solid var(--pk-border-default);
  border-radius: 6px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.pg__thumb {
  width: 22px;
  height: 22px;
  object-fit: cover;
  border-radius: 3px;
}
.pg__chipname {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pg__chipx {
  display: inline-flex;
  padding: 2px;
  border: 0;
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.pg__kind {
  vertical-align: -2px;
  margin-right: 4px;
  color: var(--pk-text-muted);
}
.pg__none-icon {
  color: var(--pk-text-muted);
}
.pg__none-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.pg__none-txt {
  margin: 0 0 6px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.pg__error {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
  border-radius: var(--pk-radius-md);
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: var(--pk-font-size-sm);
}
/* the form lives on a SURFACE CARD (ServerForm's sf__card recipe, verbatim):
   pk-input fields sit on bg-surface, never bare on the content background -
   that pairing is what holds up in both themes */
.pg__form {
  display: flex;
  flex-direction: column;
  gap: 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
  margin-bottom: 16px;
}
.pg__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.pg__label {
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-secondary);
}
.pg__ta {
  height: auto;
  padding: 10px 12px;
  resize: vertical;
  line-height: 1.55;
  font-family: inherit;
}
.pg__actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.pg__runhint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.pg__meta {
  margin-left: auto;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
/* results: the fleet/instrument table card, verbatim */
.pg__tablewrap {
  overflow-x: auto;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.pg__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-sm);
}
.pg__table thead th {
  text-align: left;
  font-weight: 600;
  font-size: var(--pk-font-size-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
  padding: 10px 14px;
  border-bottom: 1px solid var(--pk-border-default);
}
.pg__table td,
.pg__table tbody th {
  padding: 9px 14px;
  border-bottom: 1px solid var(--pk-border-default);
  color: var(--pk-text-primary);
  vertical-align: top;
}
.pg__table tr:last-child td,
.pg__table tr:last-child th {
  border-bottom: none;
}
.c-num {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.c-dim {
  color: var(--pk-text-muted);
}
.pg__score {
  position: relative;
  min-width: 90px;
}
.pg__scorebar {
  position: absolute;
  left: 0;
  bottom: -3px;
  height: 3px;
  border-radius: 2px;
  background: var(--pk-accent);
  opacity: 0.7;
}
.pg__doc {
  line-height: 1.5;
  overflow-wrap: anywhere;
}
/* single-text result: same surface card as the tables */
.pg__one {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 14px 16px;
}
.pg__one-line {
  margin: 0 0 8px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
}
.pg__one-vec {
  display: block;
  padding: 8px 10px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  overflow-x: auto;
  white-space: nowrap;
}
.pg__one-hint {
  margin: 8px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.pg__closest {
  margin: 0;
  padding: 10px 14px;
  border-top: 1px solid var(--pk-border-default);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.pg__api {
  margin-top: 14px;
}
.pg__api pre {
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
