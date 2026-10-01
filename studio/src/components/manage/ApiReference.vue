<script setup lang="ts">
// The API reference for one running endpoint (/manage/models/:port/api):
// renders the runner's own /openapi.json, fetched through the manager relay.
// The document is the source of truth (drift-tested against the router on the
// runner side) - this page only lays it out: operations grouped by the spec's
// tags, Paddock-native schemas expanded into field tables, the compatible
// families linking out to OpenAI's / Anthropic's references, and the raw
// document one copy away for codegen tools.
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { servedId, useFleetStore } from '@/stores/fleet'
import { copyText } from '@/lib/clipboard'
import { modelLabel } from '@/lib/model-name'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import SpecProse from '@/components/manage/SpecProse.vue'

// ── the OpenAPI 3.1 shapes this page reads (loose deliberately - unknown keys
//    must never break rendering; the document evolves with the runner) ───────
interface SchemaNode {
  $ref?: string
  type?: string | string[]
  const?: unknown
  enum?: unknown[]
  description?: string
  properties?: Record<string, SchemaNode>
  required?: string[]
  items?: SchemaNode
  format?: string
  default?: unknown
}
interface MediaObj {
  schema?: SchemaNode
}
interface RespSpec {
  $ref?: string
  description?: string
  content?: Record<string, MediaObj>
}
interface ParamSpec {
  name: string
  in: string
  required?: boolean
  description?: string
  schema?: SchemaNode
}
interface OpSpec {
  tags?: string[]
  summary?: string
  description?: string
  externalDocs?: { description?: string; url: string }
  parameters?: ParamSpec[]
  requestBody?: { required?: boolean; content?: Record<string, MediaObj> }
  responses?: Record<string, RespSpec>
}
interface Spec {
  openapi: string
  info: { title: string; version: string; description?: string }
  externalDocs?: { description?: string; url: string }
  tags?: { name: string; description?: string }[]
  paths: Record<string, Record<string, OpSpec>>
  components?: { schemas?: Record<string, SchemaNode> }
}

const route = useRoute()
const fleet = useFleetStore()
const port = computed(() => Number(route.params.port))
const row = computed(() => fleet.rows.find((r) => r.port === port.value))
const title = computed(() => {
  const served = row.value ? servedId(row.value) : undefined
  const t = row.value?.display ?? modelLabel(served)
  return t || served || `server ${port.value}`
})

const spec = ref<Spec | null>(null)
const loadErr = ref('')

let release: (() => void) | null = null
onMounted(() => {
  release = fleet.hold()
  void load()
})
onUnmounted(() => release?.())

async function load(): Promise<void> {
  loadErr.value = ''
  try {
    const r = await fetch(`/api/runners/${port.value}/openapi.json`)
    const data = (await r.json()) as Spec & { error?: { message?: string } }
    if (!r.ok) throw new Error(data.error?.message || `HTTP ${r.status}`)
    spec.value = data
  } catch (e) {
    loadErr.value = e instanceof Error ? e.message : String(e)
  }
}

// ── raw document affordances: the runner's own URL is what codegen tools and
//    other boxes need; the relay URL is what this same-origin page can open ──
const relayUrl = computed(() => `/api/runners/${port.value}/openapi.json`)
const rawUrl = computed(() =>
  row.value ? `${row.value.endpoint}/openapi.json` : `http://127.0.0.1:${port.value}/openapi.json`,
)
const copied = ref(false)
async function copyRaw(): Promise<void> {
  try {
    await copyText(rawUrl.value)
    copied.value = true
    setTimeout(() => (copied.value = false), 1400)
  } catch {
    /* clipboard blocked */
  }
}

// ── grouping: one section per spec tag, in the spec's own tag order ─────────
const TAG_TITLES: Record<string, string> = {
  openai: 'OpenAI-compatible',
  anthropic: 'Anthropic-compatible',
  paddock: 'Paddock-native',
  operations: 'Health & discovery',
}
interface OpRow {
  id: string
  method: string
  path: string
  op: OpSpec
  body: BodyView | null
  responses: RespView[]
  params: Row[]
}
interface Group {
  name: string
  title: string
  desc: string
  ops: OpRow[]
}
const groups = computed<Group[]>(() => {
  const s = spec.value
  if (!s) return []
  const byTag = new Map<string, OpRow[]>()
  for (const [path, item] of Object.entries(s.paths)) {
    for (const method of ['get', 'post', 'put', 'delete', 'patch'] as const) {
      const op = item[method]
      if (!op) continue
      const tag = op.tags?.[0] ?? 'other'
      const list = byTag.get(tag) ?? []
      list.push({
        id: `${method} ${path}`,
        method: method.toUpperCase(),
        path,
        op,
        body: bodyOf(op),
        responses: responsesOf(op),
        params: paramsOf(op),
      })
      byTag.set(tag, list)
    }
  }
  const order = (s.tags ?? []).map((t) => t.name)
  for (const name of byTag.keys()) if (!order.includes(name)) order.push(name)
  return order
    .filter((name) => byTag.has(name))
    .map((name) => ({
      name,
      title: TAG_TITLES[name] ?? name,
      desc: s.tags?.find((t) => t.name === name)?.description ?? '',
      ops: byTag.get(name) ?? [],
    }))
})

const expanded = ref(new Set<string>())
function toggle(id: string): void {
  const next = new Set(expanded.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expanded.value = next
}

// ── schema flattening: $refs resolved against components, nested objects and
//    arrays-of-objects indented so a response reads whole without a browser ──
function deref(n?: SchemaNode): SchemaNode | undefined {
  if (!n?.$ref) return n
  let cur: unknown = spec.value
  for (const p of n.$ref.replace(/^#\//, '').split('/'))
    cur = (cur as Record<string, unknown> | undefined)?.[p]
  return cur as SchemaNode | undefined
}
function refName(ref: string): string {
  return ref.split('/').pop() ?? ref
}
function typeLabel(n?: SchemaNode): string {
  if (!n) return ''
  if (n.$ref) return refName(n.$ref)
  if (n.const !== undefined) return `always ${JSON.stringify(n.const)}`
  if (n.enum) return n.enum.map((v) => (v === null ? 'null' : JSON.stringify(v))).join(' | ')
  if (Array.isArray(n.type)) return n.type.join(' | ')
  if (n.type === 'array') return `${n.items?.$ref ? refName(n.items.$ref) : typeLabel(n.items) || 'any'}[]`
  if (n.type === 'string' && n.format === 'binary') return 'file'
  return n.type ?? 'object'
}
interface Row {
  indent: number
  name: string
  type: string
  req: boolean
  note: string
}
function pushRows(n: SchemaNode | undefined, out: Row[], indent: number, depth: number): void {
  const node = deref(n)
  if (!node || depth > 3) return
  const req = new Set(node.required ?? [])
  for (const [name, child] of Object.entries(node.properties ?? {})) {
    const c = deref(child) ?? child
    const note = [
      child.description ?? c.description ?? '',
      child.default !== undefined ? `Default ${JSON.stringify(child.default)}.` : '',
    ]
      .filter(Boolean)
      .join(' ')
    out.push({ indent, name, type: typeLabel(child.$ref ? child : c), req: req.has(name), note })
    const inner = c.type === 'array' ? deref(c.items) : c
    if (inner?.properties) pushRows(inner, out, indent + 1, depth + 1)
  }
}
function schemaRows(n?: SchemaNode): Row[] {
  const out: Row[] = []
  pushRows(n, out, 0, 0)
  return out
}

// ── per-operation views ─────────────────────────────────────────────────────
interface BodyView {
  mime: string
  typeName: string
  rows: Row[]
}
function bodyOf(op: OpSpec): BodyView | null {
  const content = op.requestBody?.content
  if (!content) return null
  const [mime, media] = Object.entries(content)[0] ?? []
  if (!mime) return null
  const schema = media?.schema
  return {
    mime,
    typeName: schema?.$ref ? refName(schema.$ref) : '',
    rows: schemaRows(schema),
  }
}
interface RespView {
  status: string
  desc: string
  typeName: string
  rows: Row[]
}
function responsesOf(op: OpSpec): RespView[] {
  return Object.entries(op.responses ?? {}).map(([status, r]) => {
    const rr = r.$ref ? ((deref(r as SchemaNode) as RespSpec | undefined) ?? r) : r
    const schema = Object.values(rr.content ?? {})[0]?.schema
    const typeName = schema?.$ref ? refName(schema.$ref) : ''
    return {
      status,
      desc: rr.description ?? '',
      typeName,
      // the Error envelope repeats on nearly every operation - shown once in
      // the standing Errors section instead of as a table per 400
      rows: typeName === 'Error' ? [] : schemaRows(schema),
    }
  })
}
function paramsOf(op: OpSpec): Row[] {
  return (op.parameters ?? []).map((p) => ({
    indent: 0,
    name: p.name,
    type: `${typeLabel(p.schema)} (${p.in})`,
    req: p.required ?? false,
    note: p.description ?? '',
  }))
}

const errorSchema = computed(() => spec.value?.components?.schemas?.Error)
</script>

<template>
  <div class="ar">
    <nav class="ar__crumbs">
      <RouterLink :to="{ name: 'servers' }">Models</RouterLink>
      <span>/</span>
      <RouterLink :to="{ name: 'server-detail', params: { port: String(port) } }">{{
        title
      }}</RouterLink>
      <span>/</span>
      <span>API</span>
    </nav>

    <template v-if="spec">
      <div class="ar__head">
        <h1 class="ar__title">API reference</h1>
        <span class="ar__ver">runner v{{ spec.info.version }}</span>
      </div>

      <div class="ar__raw">
        <span class="ar__raw-lbl">Document</span>
        <code>{{ rawUrl }}</code>
        <Tooltip :label="copied ? 'Copied' : 'Copy the URL for codegen tools'">
          <button class="ar__raw-act" @click="copyRaw">
            <Icon :name="copied ? 'check' : 'copy'" :size="13" /> {{ copied ? 'Copied' : 'Copy' }}
          </button>
        </Tooltip>
        <a class="ar__raw-act" :href="relayUrl" target="_blank" rel="noopener">
          <Icon name="external-link" :size="13" /> Open raw
        </a>
      </div>

      <div class="ar__intro">
        <SpecProse :text="spec.info.description" />
        <p v-if="spec.externalDocs">
          <a :href="spec.externalDocs.url" target="_blank" rel="noopener">Conformance matrix</a>
          - {{ spec.externalDocs.description }}
        </p>
      </div>

      <section v-for="g in groups" :key="g.name" class="ar__group">
        <h2 class="ar__group-hd">{{ g.title }}</h2>
        <p v-if="g.desc" class="ar__group-desc">{{ g.desc }}</p>
        <div
          v-for="o in g.ops"
          :key="o.id"
          class="ar__op"
          :class="{ 'ar__op--open': expanded.has(o.id) }"
        >
          <button class="ar__op-row" @click="toggle(o.id)">
            <span class="ar__method" :class="`ar__method--${o.method.toLowerCase()}`">{{
              o.method
            }}</span>
            <code class="ar__path">{{ o.path }}</code>
            <span class="ar__sum">{{ o.op.summary }}</span>
            <Icon :name="expanded.has(o.id) ? 'chevron-up' : 'chevron-down'" :size="14" />
          </button>
          <div v-if="expanded.has(o.id)" class="ar__op-body">
            <div class="ar__desc">
              <SpecProse :text="o.op.description" />
            </div>
            <a
              v-if="o.op.externalDocs"
              class="ar__ext"
              :href="o.op.externalDocs.url"
              target="_blank"
              rel="noopener"
            >
              <Icon name="external-link" :size="13" /> {{ o.op.externalDocs.description }}
            </a>

            <template v-if="o.params.length">
              <p class="ar__sub">Parameters</p>
              <div class="ar__tblwrap">
                <table class="ar__tbl">
                  <tbody>
                    <tr v-for="(r, i) in o.params" :key="i">
                      <td class="ar__t-name"><code>{{ r.name }}</code></td>
                      <td class="ar__t-type">{{ r.type }}<span v-if="r.req"> · required</span></td>
                      <td class="ar__t-note"><SpecProse :text="r.note" inline /></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </template>

            <template v-if="o.body">
              <p class="ar__sub">
                Request body <code>{{ o.body.mime }}</code>
                <span v-if="o.body.typeName"> · {{ o.body.typeName }}</span>
              </p>
              <div v-if="o.body.rows.length" class="ar__tblwrap">
                <table class="ar__tbl">
                  <tbody>
                    <tr v-for="(r, i) in o.body.rows" :key="i">
                      <td class="ar__t-name" :style="{ paddingLeft: `${r.indent * 16}px` }">
                        <code>{{ r.name }}</code>
                      </td>
                      <td class="ar__t-type">{{ r.type }}<span v-if="r.req"> · required</span></td>
                      <td class="ar__t-note"><SpecProse :text="r.note" inline /></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </template>

            <p class="ar__sub">Responses</p>
            <div v-for="r in o.responses" :key="r.status" class="ar__resp">
              <div class="ar__resp-hd">
                <span class="ar__status">{{ r.status }}</span>
                <span class="ar__resp-desc"
                  ><SpecProse :text="r.desc" inline
                  /><template v-if="r.typeName && !r.rows.length && r.typeName !== 'Error'">
                    · {{ r.typeName }}</template
                  ></span
                >
              </div>
              <div v-if="r.rows.length" class="ar__tblwrap">
                <table class="ar__tbl">
                  <tbody>
                    <tr v-for="(row2, i) in r.rows" :key="i">
                      <td class="ar__t-name" :style="{ paddingLeft: `${row2.indent * 16}px` }">
                        <code>{{ row2.name }}</code>
                      </td>
                      <td class="ar__t-type">
                        {{ row2.type }}<span v-if="row2.req"> · required</span>
                      </td>
                      <td class="ar__t-note"><SpecProse :text="row2.note" inline /></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section v-if="errorSchema" class="ar__group">
        <h2 class="ar__group-hd">Errors</h2>
        <p class="ar__group-desc"><SpecProse :text="errorSchema.description" inline /></p>
        <div class="ar__tblwrap">
          <table class="ar__tbl">
            <tbody>
              <tr v-for="(r, i) in schemaRows(errorSchema)" :key="i">
                <td class="ar__t-name" :style="{ paddingLeft: `${r.indent * 16}px` }">
                  <code>{{ r.name }}</code>
                </td>
                <td class="ar__t-type">{{ r.type }}<span v-if="r.req"> · required</span></td>
                <td class="ar__t-note"><SpecProse :text="r.note" inline /></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </template>

    <template v-else-if="loadErr && fleet.loaded">
      <h1 class="ar__title">API reference</h1>
      <p class="ar__unknown">
        The reference is served by the running endpoint, and nothing on port {{ port }} answered:
        {{ loadErr }}
        <RouterLink :to="{ name: 'server-detail', params: { port: String(port) } }"
          >Back to the model page</RouterLink
        >
      </p>
    </template>

    <p v-else class="ar__unknown">Loading...</p>
  </div>
</template>

<style scoped>
.ar {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.ar__crumbs {
  display: flex;
  gap: 8px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  margin-bottom: 8px;
}
.ar__crumbs a {
  color: var(--pk-accent);
  text-decoration: none;
}
.ar__crumbs a:hover {
  text-decoration: underline;
}
.ar__head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
}
.ar__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin: 0;
}
.ar__ver {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}

.ar__raw {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin: 14px 0;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.ar__raw-lbl {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
}
.ar__raw code {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  user-select: all;
}
.ar__raw-act {
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
  text-decoration: none;
}
.ar__raw-act:hover {
  color: var(--pk-text-primary);
  border-color: var(--pk-accent);
}

.ar__intro {
  margin: 0 0 20px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.55;
  max-width: 72ch;
}
.ar__intro :deep(p) {
  margin: 0 0 10px;
}
.ar__intro :deep(ul) {
  margin: 0 0 10px;
  padding-left: 18px;
}
.ar__intro :deep(li) {
  margin-bottom: 4px;
}
.ar__intro :deep(code),
.ar__desc :deep(code),
.ar__t-note :deep(code),
.ar__resp-desc :deep(code),
.ar__group-desc :deep(code) {
  font-family: var(--pk-font-mono);
  font-size: 0.92em;
  background: var(--pk-bg-elevated);
  padding: 0 4px;
  border-radius: 3px;
}
.ar__intro a {
  color: var(--pk-accent);
}

.ar__group {
  margin-bottom: 26px;
}
.ar__group-hd {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--pk-text-primary);
  margin: 0 0 4px;
}
.ar__group-desc {
  margin: 0 0 10px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  max-width: 72ch;
}

.ar__op {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  margin-bottom: 8px;
  overflow: hidden;
}
.ar__op-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px 12px;
  background: none;
  border: none;
  font: inherit;
  text-align: left;
  cursor: pointer;
  color: var(--pk-text-primary);
  min-width: 0;
}
.ar__op-row:hover {
  background: var(--pk-bg-elevated);
}
.ar__method {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  padding: 1px 7px;
  border-radius: var(--pk-radius-md);
  border: 1px solid currentColor;
  flex-shrink: 0;
}
.ar__method--get {
  color: var(--pk-status-success, #4a9);
}
.ar__method--post {
  color: var(--pk-accent);
}
.ar__method--delete {
  color: var(--pk-text-danger);
}
.ar__path {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  flex-shrink: 0;
}
.ar__sum {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ar__op-body {
  padding: 4px 12px 14px;
  border-top: 1px solid var(--pk-border-default);
}
.ar__desc {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.55;
  max-width: 72ch;
}
.ar__desc :deep(p) {
  margin: 10px 0 0;
}
.ar__desc :deep(ul) {
  margin: 10px 0 0;
  padding-left: 18px;
}
.ar__ext {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-top: 10px;
  color: var(--pk-accent);
  font-size: var(--pk-font-size-sm);
  text-decoration: none;
}
.ar__ext:hover {
  text-decoration: underline;
}
.ar__sub {
  margin: 14px 0 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
}
.ar__sub code {
  text-transform: none;
  font-family: var(--pk-font-mono);
  letter-spacing: 0;
}

.ar__tblwrap {
  overflow-x: auto;
}
.ar__tbl {
  width: 100%;
  border-collapse: collapse;
}
.ar__tbl td {
  padding: 5px 12px 5px 0;
  border-bottom: 1px solid var(--pk-border-default);
  vertical-align: top;
  font-size: var(--pk-font-size-sm);
}
.ar__tbl tr:last-child td {
  border-bottom: none;
}
.ar__t-name code {
  font-family: var(--pk-font-mono);
  color: var(--pk-text-primary);
  white-space: nowrap;
}
.ar__t-type {
  color: var(--pk-text-muted);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
  padding-top: 7px;
}
.ar__t-note {
  color: var(--pk-text-secondary);
  width: 60%;
  line-height: 1.45;
}

.ar__resp {
  margin-bottom: 6px;
}
.ar__resp-hd {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.ar__status {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  color: var(--pk-text-primary);
}
.ar__resp-desc {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.45;
}

.ar__unknown {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  line-height: 1.5;
}
.ar__unknown a {
  color: var(--pk-accent);
}
</style>
