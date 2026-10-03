// Client for the server store (`/api/*`). The Studio is served same-origin, so
// these are relative paths; when auth lands, the injected session token is
// attached here in one place.

import type { Conversation } from '@/types/chat'
import type { ReadDoc, ReadSummary } from '@/lib/reads'
import type { TableSession, TableSummary } from '@/lib/table-history'
import { orderedRunQuestions } from '@/lib/reads'
import { DEFAULT_PARAMS } from '@/types/chat'
import { useModelsStore } from '@/stores/models'

export interface ConversationSummary {
  id: string
  title: string
  model: string
  pinned: boolean
  updatedAt: number
  createdAt: number
  /** 'chat' | 'transcription' | 'document', decided by the server from the
   *  stored turns. The list ships summaries, so the messages that would answer
   *  this are not here - which is exactly why the server answers it. Absent on
   *  a manager older than the column. */
  kind?: string
}

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
  return (await r.json()) as T
}

async function jsend(url: string, method: string, body?: unknown): Promise<Response> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
  return r
}

/** Like jsend but returns the parsed JSON body and, on error, throws with the
 *  server's `error.message` (the MCP endpoints return OpenAI-shaped errors, e.g.
 *  a label conflict) rather than a bare status code. */
async function jbody<T>(url: string, method: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const text = await r.text()
  const data = text ? JSON.parse(text) : undefined
  if (!r.ok) {
    const msg = (data as { error?: { message?: string } })?.error?.message
    throw new Error(msg || `HTTP ${r.status}`)
  }
  return data as T
}

export const tableHistoryApi = {
  list: () => jget<TableSummary[]>('/api/table-history'),
  async get(id: string): Promise<{ doc: TableSession; revision: string }> {
    const reply = await jget<{ doc: string; revision: string }>(`/api/table-history/${encodeURIComponent(id)}`)
    const doc = JSON.parse(reply.doc) as TableSession
    if (doc.version !== 1 || doc.id !== id || !Array.isArray(doc.runs) || typeof doc.datasets?.[doc.draft?.dataset] !== 'string') {
      throw new Error('Unsupported table session')
    }
    return { doc, revision: reply.revision }
  },
  save: (doc: TableSession, revision: string) => jbody<TableSummary>(
    `/api/table-history/${encodeURIComponent(doc.id)}?revision=${encodeURIComponent(revision)}`, 'PUT', doc),
  remove: (id: string, revision: string) => jbody<void>(
    `/api/table-history/${encodeURIComponent(id)}?revision=${encodeURIComponent(revision)}`, 'DELETE'),
}

// ── GPU telemetry (/api/gpu) ────────────────────────────────────────────────
// Mirrors the server's telemetry::GpuInfo - every metric is optional because a
// given card/driver may not expose a sensor (fan on passive cards, per-process
// util on Windows WDDM, etc.). Values are null (not omitted) when unsupported.
export interface GpuInfo {
  metal?: { unified_memory_total: number | null; recommended_working_set: number;
    thermal_pressure?: string | null; memory_pressure?: string | null; counter_sets?: string[] }
  index: number
  name: string
  uuid?: string | null
  pci?: string | null
  util_gpu?: number | null
  util_mem?: number | null
  mem_used?: number | null
  mem_total?: number | null
  temp_c?: number | null
  power_w?: number | null
  power_limit_w?: number | null
  sm_clock_mhz?: number | null
  mem_clock_mhz?: number | null
  fan_pct?: number | null
  /** an NVIDIA part with no memory of its own (DGX Spark's GB10): mem_used /
   *  mem_total are the machine's RAM as the OS books it, not VRAM */
  host_memory?: boolean
  /** processes holding memory on this device (per-PID attribution input). */
  procs?: { pid: number; mem?: number | null }[]
}
/** Live engine counters - what a runner's GPU work looks like right now. */
export interface EngineSnapshot {
  tok_s: number
  phase: 'idle' | 'prefill' | 'decode'
  active_slots: number
  kv_used: number
  kv_total: number
  tokens_total: number
  /** measured VRAM the loaded model holds (weights + caches), bytes. The
   *  engine measures this itself, so it works where NVML can't attribute
   *  per-process memory (Windows). */
  model_mem?: number | null
}

/** One runner's memory story in the manager's reconciliation gauge (§9):
 *  NVML outside view vs allocator self-report, plus its live engine section. */
export interface RunnerVram {
  port: number
  pid: number
  /** NVML index of the GPU holding this runner's memory (OS-attributed);
   *  absent under the WDDM per-PID blind spot. */
  gpu?: number | null
  nvml_mem?: number | null
  self_mem?: number | null
  drift?: number | null
  anomaly: boolean
  engine?: EngineSnapshot | null
  metal?: MetalRunnerMetrics | null
}

export interface MetalRunnerMetrics {
  allocated_bytes: number
  completed_commands: number
  gpu_seconds_total: number
  last_command_ms: number | null
}

export interface Reconciliation {
  ts: number
  runners: RunnerVram[]
  /** false when the driver hides per-PID bytes (Windows WDDM) - the outside
   *  view is then honestly absent, not zero. */
  attribution: boolean
  paddock_mem?: number | null
  other_mem?: number | null
  device_used: number
  device_total: number
  anomaly: boolean
}

export interface GpuSnapshot {
  /** false when the platform's device sampler is unavailable. */
  available: boolean
  ts: number
  gpus: GpuInfo[]
  /** the fleet join - null until the manager's reconciler has sampled. */
  reconciliation?: Reconciliation | null
}

export const gpuApi = {
  get: () => jget<GpuSnapshot>('/api/gpu'),
}

// ── KV offloading (/api/cache) ──────────────────────────────────────────────
/** One model's KV offloading: what it decided, and why. */
export interface CacheTier {
  lookups: number
  hits: number
  miss_cold: number
  miss_no_new_tokens: number
  miss_tripped: number
  miss_ghost: number
  elected_restore: number
  elected_recompute: number
  parked: number
  park_refused: number
  resolved: number
  abandoned: number
  served_from_ram: number
  served_from_nvme: number
  promoted_to_disk: number
  useful_bytes: number
  moved_bytes: number
  ram_ready: number
  ram_in_flight: number
  ram_reserved: number
  ram_capacity: number
  disk_ready: number
  disk_capacity: number
  resident_runs: number
  in_flight_demotes: number
  open_tickets: number
  pending_durable_writes: number
  tripped: boolean
  io_failures: number
  integrity_failures: number
  evictions: number
  single_flight_joins: number
  stale_completions: number
  rate_ram_bpus: number
  rate_disk_bpus: number
  /** null until a reuse has completed - never a confident 0%. */
  prediction_error_pct: number | null
  disk_read_gbs: number
  disk_write_gbs: number
  disk_unbuffered: boolean
  disk_written_today: number
  ghost_keys: number
}

export interface CacheServer {
  port: number
  model: string | null
  tier: CacheTier
}

export const cacheApi = {
  get: () => jget<{ servers: CacheServer[] }>('/api/cache'),
}

// ── usage timeline (api/usage/history) ──────────────────────────
/** One timeline slot: a port's traffic in one bucket, series dims summed away. */
export interface UsageSlot {
  t: number
  port: number
  requests: number
  errors_4xx: number
  errors_5xx: number
  disconnects: number
  input_tokens: number
  output_tokens: number
  cached_tokens: number
  duration_ms_sum: number
  spec_drafted: number
  spec_accepted: number
  kv_pages_max: number
  /** Per-slot increments on the runner's 14-step semconv latency ladder
   *  (0.01..81.92 s) - the duration/TTFT percentile panels read these. */
  e2e_h: number[]
  ttft_h: number[]
}
/** A hole in observation - drawn as a hatched band, never as quiet time. */
export interface UsageGap {
  id: number
  port: number
  from_ts_ms: number
  to_ts_ms: number
  cause: string
  lost_requests?: number | null
  lost_input_tokens?: number | null
  lost_output_tokens?: number | null
  from_seq?: number | null
  to_seq?: number | null
}
/** One lifecycle band: what a port ran and why it started/stopped; causes
 *  are null where nobody observed them. */
export interface UsageGeneration {
  instance_id: string
  port: number
  runner_version: string
  model?: string | null
  embedder?: string | null
  asr?: string | null
  /** Served forced-alignment model id - a runner in that role carries only
   *  this, so a reader keyed on the other three names it nothing. */
  aligner?: string | null
  started_ms: number
  ended_ms?: number | null
  start_cause?: string | null
  end_cause?: string | null
  /** Newest scrape folded for this band - its heartbeat. An open band is only
   *  honestly running while this is recent; a stale one is a runner nothing
   *  has heard from, which the table says rather than drawing it as healthy. */
  last_seen_ms?: number | null
}
export interface UsageHistory {
  grain_ms: number
  /** buckets before this boundary never change again - cache them, refetch
   *  only from here (the no-refresh-button contract). */
  closed_through_ms: number
  now_ms: number
  /** First instant this box has any usage record for - the left edge of the
   *  all-history pan/zoom axis. Null on a box that never served. */
  extent_from_ms: number | null
  buckets: UsageSlot[]
  gaps: UsageGap[]
  generations: UsageGeneration[]
  web: WebSlot[]
}
/** One slot of server-executed web-search spend on one provider. The three
 *  counters are different currencies and must never be added together:
 *  requests is the only one every provider reports, credits mean nothing
 *  outside that provider's pricing page, and microdollars are integer because
 *  this is money. */
export interface WebSlot {
  t: number
  port: number
  provider: string
  requests: number
  credits: number
  microdollars: number
}
export const usageApi = {
  history: (p: { from?: number; to?: number; port?: number | null; bucketsFrom?: number }) => {
    const q = new URLSearchParams()
    if (p.from !== undefined) q.set('from', String(Math.round(p.from)))
    if (p.to !== undefined) q.set('to', String(Math.round(p.to)))
    if (p.port !== undefined && p.port !== null) q.set('port', String(p.port))
    if (p.bucketsFrom !== undefined) q.set('buckets_from', String(Math.round(p.bucketsFrom)))
    return jget<UsageHistory>(`/api/usage/history?${q}`)
  },
}

// ── attachments (image/doc bytes, kept out of the conversation doc) ──────────
export const attachmentsApi = {
  /** Upload an attachment's full-res "view" bytes under a client-chosen id. */
  put: async (
    id: string,
    blob: Blob,
    mime: string,
    meta: { name?: string; w?: number; h?: number; conv?: string },
  ): Promise<void> => {
    const p = new URLSearchParams()
    if (meta.name) p.set('name', meta.name)
    if (meta.w != null) p.set('w', String(meta.w))
    if (meta.h != null) p.set('h', String(meta.h))
    if (meta.conv) p.set('conv', meta.conv)
    const r = await fetch(`/api/attachments/${id}?${p.toString()}`, {
      method: 'PUT',
      headers: { 'Content-Type': mime },
      body: blob,
    })
    if (!r.ok) throw new Error(`attachment upload: HTTP ${r.status}`)
  },
  /** Same-origin URL that streams the stored bytes (for the lightbox + thumbs). */
  url: (id: string): string => `/api/attachments/${id}`,
  /** A viewable JPEG of a photo this browser cannot decode itself - HEIC,
   *  which is HEVC and therefore Safari-only, being the whole reason.
   *
   *  The stored bytes are untouched: `url` still serves the original, and
   *  `metadata` still reads it. This is a copy for looking at. Returns 501 when
   *  the install has no image decoder beside it, which callers should show as a
   *  reason rather than a broken frame. */
  renditionUrl: (id: string, max?: number): string =>
    max ? `/api/attachments/${id}/rendition?max=${max}` : `/api/attachments/${id}/rendition`,
  /** Everything the stored file says about itself: EXIF/XMP/IPTC/ICC/GPS for
   *  photos, the Info dict for PDFs, core/app/custom properties for Office
   *  files. Answered by the manager off the bytes it already has, so it works
   *  with nothing running and on cloud-model chats. Nothing is cached, here or
   *  server-side - the parse is faster than the round trip. */
  metadata: async (id: string): Promise<AttachmentMetadata> => {
    const r = await fetch(`/api/attachments/${id}/metadata`)
    if (!r.ok) throw new Error(`attachment metadata: HTTP ${r.status}`)
    return (await r.json()) as AttachmentMetadata
  },
  /** Write through the full file metadata the runner shipped in a chat turn, so
   *  the manager serves it from the DB thereafter (runner-independent). The
   *  bytes are immutable, so this is idempotent. Fire-and-forget from the
   *  stream - it must never disrupt the chat. */
  storeMetadata: async (id: string, meta: unknown): Promise<void> => {
    const r = await fetch(`/api/attachments/${id}/metadata`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ meta }),
    })
    if (!r.ok) throw new Error(`store metadata: HTTP ${r.status}`)
  },
}

/** Forensic reports (paddock-forensics). The runner computes them during a chat
 *  turn (forensics on) and returns them in the `/v1/responses` output; the Studio
 *  writes them through to the DB with `persist`, and reads them back with
 *  `forAttachment` - answered by the manager off the stored row, so it works with
 *  the runner stopped, exactly like `attachmentsApi.metadata`. Unlike metadata,
 *  the report is PERSISTED (it is GPU-expensive and cannot be cheaply re-derived). */
/** A rectangular region a finding localizes to, in pixels of the original image. */
export interface ForensicRegion {
  x: number
  y: number
  w: number
  h: number
}

/** A deduplicated, human-titled finding - the report's headline signals. */
export interface ForensicKeyFinding {
  title: string
  description: string
  severity: string
  confidence: number
  sources: string[]
  region?: ForensicRegion | null
  count: number
  seq: number
}

/** One grouped explanation category (a family of related signals). */
export interface ForensicExplanationCategory {
  name: string
  finding_count: number
  max_severity: string
  explanation: string
  finding_codes: string[]
  seq: number
}

/** The plain-language risk explanation, split into narrative slots. */
export interface ForensicExplanation {
  summary: string
  visual_review?: string | null
  cross_corroboration?: string | null
  anti_forensics_warning?: string | null
  categories: ForensicExplanationCategory[]
}

/** One raw analyzer signal, before dedup into a key finding. */
export interface ForensicFinding {
  analyzer: string
  code: string
  severity: string
  confidence: number
  description: string
  region?: ForensicRegion | null
  seq: number
}

/** A stored forensic report, as `GET /api/attachments/{id}/forensics` returns it
 *  - the manager's reconstitution of `paddock-forensics`' `report_value`. */
export interface ForensicReport {
  id: string
  attachment_id?: string | null
  conversation_id?: string | null
  sha256: string
  kind: string
  mime: string
  name: string
  width?: number | null
  height?: number | null
  content_type: string
  format: string
  finding_count: number
  max_severity: string
  risk_score: number
  /** The verdict sentence. */
  verdict: string
  gpu: boolean
  elapsed_ms: number
  created_at: number
  /** "info" | "low" | "medium" | "high" | "critical" - the headline risk band. */
  risk_level: string
  corroborating_stages: number
  explanation: ForensicExplanation
  key_findings: ForensicKeyFinding[]
  findings: ForensicFinding[]
}

export const forensicsApi = {
  /** Write through a report the runner returned, for one attachment. The manager
   *  resolves sha256/mime/name from the attachment itself, so only the ids + the
   *  runner's `report` object are needed. Idempotent per attachment. */
  persist: async (body: {
    attachment_id: string
    conversation_id?: string | null
    kind?: string
    report: unknown
  }): Promise<void> => {
    const r = await fetch('/api/forensics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!r.ok) throw new Error(`forensics persist: HTTP ${r.status}`)
  },
  /** The stored report for an attachment, or `null` when it was never analyzed. */
  forAttachment: async (id: string): Promise<ForensicReport | null> => {
    const r = await fetch(`/api/attachments/${id}/forensics`)
    if (r.status === 404) return null
    if (!r.ok) throw new Error(`forensics fetch: HTTP ${r.status}`)
    return (await r.json()) as ForensicReport
  },
}

/** One metadata field, display-ready. */
export interface FileMetaTag {
  name: string
  value: string
  /** The value hit the server's per-field ceiling and continues past what's
   *  here - disclosed rather than presented as the whole truth. */
  truncated?: boolean
}
/** One source of fields within a file (EXIF, XMP, PDF, Custom ...). */
export interface FileMetaGroup {
  name: string
  tags: FileMetaTag[]
}
/** A place name for a coordinate, resolved from the offline table in the
 *  manager - nearest populated place, not a boundary test, so `region` is the
 *  matched city's region and `distance_km` is how far that claim reaches. */
export interface FilePlace {
  city: string
  /** First-level division ("Tuscany"); empty when the source has no name. */
  region: string
  country: string
  distance_km: number
  /** Compass direction from the place to the coordinate ("NE"). */
  bearing: string
  /** "in Arezzo (Tuscany, Italy)" - the same phrase the model's prompt line
   *  carries. Render this rather than re-assembling the parts. */
  description: string
}
/** Where the file says it was. The coordinates are in `groups` as text too;
 *  these are the numbers, for anything that has to plot them. */
export interface FileLocation {
  latitude: number
  longitude: number
  /** Metres above sea level, when the file states it. */
  altitude?: number
  place?: FilePlace
}
export interface AttachmentMetadata {
  id: string
  /** As uploaded; empty for a microphone recording, which has no file name. */
  name: string
  /** The browser's guess from the extension. `format` is what the BYTES say -
   *  the two disagreeing is itself worth showing. */
  mime: string
  size: number
  format: string | null
  /** Which reader answered: sift, scriptor, or none when the file is silent. */
  reader: 'sift' | 'scriptor' | 'none'
  /** Empty is an ordinary answer: a screenshot, a code file, a stripped export. */
  groups: FileMetaGroup[]
  /** Absent on most files - only a photo that carries GPS has one. */
  location?: FileLocation
}

// ── model registry (browse + pull; the pullable set is the server's compiled-in
//    models.toml manifest, files hosted on a dumb origin like R2) ──────────────
export interface CatalogFile {
  /** stable, absolute download URL. */
  url: string
  /** where it lands, relative to the server's models dir. */
  dest: string
  sha256: string
  size: number
}
/** Vendor-sourced spec sheet, shown when a model row is expanded. */
export interface ModelSpecs {
  /** Upstream publication (YYYY-MM-DD), never the registry/export revision. */
  published_at?: string
  published_source?: string
  params?: string
  context?: string
  context_max?: string
  dims?: string
  homepage?: string
  about?: string
  /** comparison-card bullets - factual, both sides (never just an ad). */
  strengths?: string[]
  tradeoffs?: string[]
}
/** One independently downloadable PIECE of a model (schema 3): a weights
 *  alternative (the quality choice) or a companion (vision tower, MTP
 *  drafter, native-FP8 snapshot). */
export interface CatalogArtifact {
  id: string
  kind: 'weights' | 'vision' | 'drafter' | 'fp8-snapshot'
  format: 'gguf' | 'safetensors'
  label: string
  /** Export-specific loader contract; absent preserves older managers. */
  runtime?: {
    backends?: string[]
    capability?: string[]
    companions?: string[]
    checkpoint_dir?: boolean
    embedded_vision?: boolean
    kv_cache_dtype?: string
    experimental?: boolean
    qualification?: 'unqualified' | 'experimental' | 'qualified'
    default_max_ctx?: number
    default_max_batch?: number
    default_spec?: string
    memory?: { max_ctx: number; max_batch: number }
    note?: string
  }
  source?: {
    repo: string
    revision: string
    base_model: string
    license: string
    license_url: string
  }
  backend_supported?: boolean
  kv_offload_supported?: boolean
  /** the honest quant tag for weights ("Q8_0", "UD-Q4_K_XL", "MXFP4"). */
  quant?: string
  /** part of the row-level Download bundle. */
  default?: boolean
  /** this companion is the model's point (granite-vision's tower) - the
   *  Studio shows no on/off for it. */
  required?: boolean
  /** minimum compute capability [major, minor] this artifact can be SERVED
   *  on; absent means anywhere the engine runs. NVFP4's W4A16 kernels are
   *  sm_120a-only, and off that target the engine falls back to the base
   *  build - so the card greys out instead of selling a download that would
   *  change nothing. */
  min_cc?: [number, number]
  files: CatalogFile[]
  /** server-annotated: every file present locally at the right size. */
  installed: boolean
  /** server-annotated: bytes across this piece's files. */
  total_size: number
}
export interface CatalogModel {
  id: string
  display: string
  vendor?: string
  family?: string
  capability: string[]
  /** KV cache precision this family serves at unless overridden ("f16" |
   *  "fp8_e4m3"); absent = f16. Mirrors the engine so the Start form can
   *  preselect the real value instead of an "auto" that hides it. */
  kv_default?: string
  revision?: string
  license?: string
  specs?: ModelSpecs
  /** the model's pieces - weights alternatives + optional companions. */
  artifacts: CatalogArtifact[]
  /** server-annotated: servable now (≥1 weights artifact installed). */
  installed: boolean
  /** server-annotated: the default bundle's bytes (the Download button). */
  total_size: number
}
export const registryApi = {
  catalog: () => jget<{ schema: number; models: CatalogModel[] }>('/api/models/catalog'),
  // pulls moved to the downloads store (manager-side jobs: SSE progress,
  // cancel/resume, queued starts) - see stores/downloads.ts
}

/** Ask the manager what a config BUFFER means. The one caller is the Start/Edit
 *  page's Simple tab, and the reason it is a round trip rather than a local
 *  parse is that the identity rule must exist exactly once - see the endpoint's
 *  own comment in routes.rs for the two bugs that came of it existing twice. */
export async function projectConfig(
  toml: string,
): Promise<{ projection: ConfigProjection } | { error: string }> {
  try {
    const res = await fetch('/api/servers/project', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ toml }),
    })
    const body = (await res.json().catch(() => null)) as
      | (ConfigProjection & { error?: { message?: string } })
      | null
    if (!res.ok || !body || typeof body.model !== 'string') {
      return { error: body?.error?.message ?? `the manager answered ${res.status}` }
    }
    return { projection: body }
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) }
  }
}

/** Config TEXT as the Start/Edit page's Simple tab reads it. Every field is
 *  READ from the buffer by the manager - `model` is already the catalog id,
 *  reconciled by the one implementation that also runs on a start. Nothing here
 *  is re-derived in the browser, which is the entire point of the endpoint. */
export interface ConfigProjection {
  residency?: import('@/stores/fleet').ResidencyConfig | null
  residency_supported?: boolean
  model: string
  artifact: string | null
  /** Drafter-artifact pin; null = follow the catalog default. */
  drafter: string | null
  weights: string | null
  vision: boolean
  fp8_native: boolean
  max_ctx: number | null
  max_batch: number | null
  gpu: string | null
  kv_cache_dtype: string | null
  spec: string | null
  api_key: string | null
  vram_budget: number | null
  web_search_provider: string | null
  web_search_api_key: string | null
  mcp_servers: unknown[]
  /** The `[forensics]` block as the file carries it, or null when absent. The
   *  Intelligence section binds `enabled`; auto/tool/device round-trip. */
  forensics: ForensicsConfig | null
  /** The `[kv_offload]` block as the file carries it, or null when absent. */
  kv_offload: KvOffloadConfig | null
}

/** The `[kv_offload]` config table as projected from a config file. Mirrors
 *  the manager's `KvOffloadSpec` (and fleet.ts `KvOffloadSpec`). */
export interface KvOffloadConfig {
  enabled: boolean
  ram_gb: number
  nvme_gb?: number | null
  nvme_path?: string | null
}

/** The `[forensics]` config table as projected from a config file. Mirrors the
 *  manager's `ForensicsSpec` (and fleet.ts `ForensicsSpec`). */
export interface ForensicsConfig {
  enabled: boolean
  auto?: string | null
  tool?: boolean | null
  device?: number | null
}

export const store = {
  listConversations: () => jget<ConversationSummary[]>('/api/conversations'),
  getConversation: (id: string) => jget<Conversation>(`/api/conversations/${id}`),
  putConversation: (c: Conversation) => jsend(`/api/conversations/${c.id}`, 'PUT', c),
  deleteConversation: (id: string) => jsend(`/api/conversations/${id}`, 'DELETE'),
  getSettings: () => jget<Record<string, unknown>>('/api/settings'),
  putSettings: (patch: Record<string, unknown>) => jsend('/api/settings', 'PUT', patch),
}

// ── saved system prompts (reusable library, /api/prompts) ───────────────────

export interface SavedPrompt {
  id: string
  name: string
  /** the system-prompt text sent as the request's `instructions`. */
  body: string
  createdAt?: number
  updatedAt?: number
  revision?: string
}

export const promptsApi = {
  list: () => jget<SavedPrompt[]>('/api/prompts'),
  // POST upserts (the store's put_prompt is INSERT ... On CONFLICT(id) UPDATE).
  save: (p: SavedPrompt) => jbody<{ ok: boolean; prompt?: SavedPrompt }>('/api/prompts', 'POST', p),
  remove: (id: string, revision?: string) => jbody(`/api/prompts/${encodeURIComponent(id)}${revision === undefined ? '' : `?revision=${encodeURIComponent(revision)}`}`, 'DELETE'),
}

// ── saved read sets (the Reads page's question sets, /api/reads) ────────────

export interface SavedReadSet {
  id: string
  name: string
  /** JSON text: `{ questions, samples }` in the /v1/systemone shape. */
  body: string
  createdAt?: number
  updatedAt?: number
  revision?: string
}

export const readsApi = {
  list: () => jget<SavedReadSet[]>('/api/reads'),
  // POST upserts, exactly as prompts do (INSERT ... ON CONFLICT(id) UPDATE).
  save: (r: SavedReadSet) => jbody<{ ok: boolean; set?: SavedReadSet }>('/api/reads', 'POST', r),
  remove: (id: string, revision?: string) =>
    jbody(
      `/api/reads/${encodeURIComponent(id)}${revision === undefined ? '' : `?revision=${encodeURIComponent(revision)}`}`,
      'DELETE',
    ),
}

// Earlier reads - the Reads page's side panel. The list is summaries; a read
// is fetched whole when opened and always saved whole (the manager refuses a
// summary, so a list row can never overwrite the runs it stands for).
export const readHistoryApi = {
  list: () => jget<ReadSummary[]>('/api/read-history'),
  get: async (id: string): Promise<ReadDoc> => {
    const snapshot = await jget<{ doc: string; revision: string }>(`/api/read-history/${encodeURIComponent(id)}?envelope=true`)
    const doc = JSON.parse(snapshot.doc) as ReadDoc
    doc.runs = doc.runs.map((run) => ({ ...run, questions: orderedRunQuestions(run) }))
    return { ...doc, revision: snapshot.revision }
  },
  save: ({ revision, ...doc }: ReadDoc) =>
    jbody<{ ok: boolean; read?: ReadSummary }>(
      `/api/read-history/${encodeURIComponent(doc.id)}?revision=${encodeURIComponent(revision ?? '')}`,
      'PUT',
      doc,
    ),
  remove: (id: string, revision: string) => jbody(`/api/read-history/${encodeURIComponent(id)}?revision=${encodeURIComponent(revision)}`, 'DELETE'),
}

export const readsPreferencesApi = {
  get: () => jget<{ readsPanelOpen?: boolean }>('/api/settings'),
  save: (open: boolean) => jbody('/api/settings', 'PUT', { readsPanelOpen: open }),
}

export const tablesPreferencesApi = {
  get: () => jget<{ tablesPanelOpen?: boolean }>('/api/settings'),
  save: (open: boolean) => jbody('/api/settings', 'PUT', { tablesPanelOpen: open }),
}

// ── MCP tool approvals ──────────────────────────────────────────────────────
// NOTE: there is no MCP/search CRUD here anymore - web search and
// MCP servers are MODEL configuration (each endpoint's config file, edited on
// its Start/Edit page). What remains is the approval gate: the parked agent
// loop is in-process state on the RUNNER, resolved through the manager relay.

export const approvalsApi = {
  /** The approval id lives in exactly one registry - the current runner's,
   *  or the manager's cloud agent loop. The card doesn't know which lane
   *  parked it, so post to both; the wrong one answers 404 harmlessly. */
  approve: async (approvalId: string, approve: boolean, modelId?: string) => {
    const port = useModelsStore().portFor(modelId)
    const paths = [
      ...(port ? [`/api/runners/${port}/mcp-approvals/${approvalId}`] : []),
      `/api/cloud/mcp-approvals/${approvalId}`,
    ]
    for (const p of paths) {
      try {
        const r = await jbody<{ ok: boolean; approved: boolean }>(p, 'POST', { approve })
        if (r.ok) return r
      } catch {
        // not this registry - try the next
      }
    }
    return { ok: false, approved: approve }
  },
}

// ── feedback (/api/feedback) ────────────────────────────────────────────────
// The manager forwards to the truespar API; nothing here talks to it directly.

export type FeedbackCategory = 'bug' | 'feature' | 'feedback'

/** The diagnostics the dialog previews and (only on request) attaches. This
 *  mirrors the manager's `feedback::Context` - and it is fetched rather than
 *  assembled here deliberately: the preview must be the payload, not a second
 *  rendering of it that can drift. */
export interface FeedbackContext {
  manager: { version: string; build: string; os: string; arch: string }
  gpu: {
    state: string
    card?: string
    generation?: string
    driver?: string
    cuda?: string
    cuda_needed: string
  }
  runners: {
    model: string
    status: string
    version?: string
    artifact?: string
    kv_cache_dtype?: string
    max_ctx?: number
    max_batch?: number
    spec?: string
  }[]
}

export interface FeedbackSubmission {
  category: FeedbackCategory
  message: string
  email?: string
  /** Send the context blob. Absent/false means it stays on this box. */
  include_context?: boolean
}

export const feedbackApi = {
  context: () => jget<FeedbackContext>('/api/feedback/context'),
  // jbody surfaces the server's `error.message` - which for a rate limit is
  // the upstream's own sentence about when to retry, relayed verbatim.
  submit: (s: FeedbackSubmission) =>
    jbody<{ ok: boolean; id?: string }>('/api/feedback', 'POST', s),
}

/** A list summary as a not-yet-loaded Conversation stub (messages fill on open). */
export function stubFromSummary(s: ConversationSummary): Conversation {
  return {
    id: s.id,
    title: s.title,
    model: s.model,
    pinned: s.pinned,
    kind: s.kind,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
    messages: [],
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
  }
}
