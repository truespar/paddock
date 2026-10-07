import { defineStore } from 'pinia'
import { usePushStore } from '@/stores/push'
import { computed, ref } from 'vue'
import { useToastsStore } from '@/stores/toasts'
import { cloudVendor, useModelsStore } from '@/stores/models'
import { useDownloadsStore } from '@/stores/downloads'
import { modelLabel } from '@/lib/model-name'
import { reasonOf } from '@/lib/error-text'

/** One endpoint-attached MCP server (per-model config; resolvable by bare
 *  label on every API surface the runner serves). Either transport: remote
 *  HTTP (url + headers) or a local stdio command (args + env). */
export interface McpServerEntry {
  server_label: string
  server_url?: string
  headers?: Record<string, string>
  command?: string
  args?: string[]
  env?: Record<string, string>
  /** narrow which of the server's tools the model may call ([] / absent = all). */
  allowed_tools?: string[]
  require_approval?: 'always' | 'never'
}

/** One runner as `/api/runners` reports it (RunnerView + the reconciler's
 *  VRAM join). `pinned` is the §10.1 policy flag: never auto-stopped to make
 *  room, excluded from the estimator's reclaimable figure. */
export interface FleetRow {
  port: number
  pid: number
  origin: 'own' | 'adopted'
  status: string
  model: string | null
  /** speculation mechanism in words ("MTP", "DFlash1", "off") - absent when
   *  the model has nothing to speculate with */
  spec?: string | null
  embedder: string | null
  /** Speech-to-text (whisper-family), forced-aligner and image-generation
   *  serving ids - a row carries exactly one of model/embedder/asr/aligner/
   *  image, and dropping the latter ones made those runners nameless in the
   *  fleet. */
  asr?: string | null
  aligner?: string | null
  image?: string | null
  /** decision model (Laya, /v1/systemone) and tabular predictor (Kumo,
   *  /v1/tabular/*) serving ids - the sixth and seventh roles */
  reader?: string | null
  tabular?: string | null
  diarization?: string | null
  /** dense prediction (DINOv3, /v1/segmentations) and promptable segmentation
   *  (SAM 3, /v1/masks) serving ids - pictures in, rasters or masks out */
  segmenter?: string | null
  masker?: string | null
  /** The catalog's human name ("Qwen 3.5 9B") + maker - what the UI shows;
   *  the technical id stays in `model` for tooltips. Absent for models this
   *  build's catalog doesn't know. */
  display?: string | null
  vendor?: string | null
  version: string | null
  uptime_s: number | null
  in_flight: number | null
  endpoint: string
  pinned: boolean
  /** The as-deployed configuration (retained spawn spec). Absent for adopted
   *  runners - their config is not the manager's to report. */
  config?: {
    model: string
    artifact?: string | null
    max_ctx: number | null
    max_batch: number | null
    /** the config FILE's pin: a device UUID ("GPU-...") or an ordinal string. */
    gpu: string | number | null
    kv_cache_dtype: string | null
    /** Speculation policy as deployed ("off" | "auto" | "ladder" | "<k>");
     *  absent = the key is not in the file and the model's default stands. */
    spec?: string | null
    keyed: boolean
    /** the inference key NETWORK callers must send (loopback is exempt
     *  runner-side); issued at start when none was given. */
    api_key?: string | null
    runner_version: string | null
    /** SERVER TOOLS (per-model config): the web-search integration + MCP
     *  servers this endpoint supplies. */
    web_search_provider?: string | null
    web_search_api_key?: string | null
    mcp_servers?: McpServerEntry[]
    /** Forensics gate (`[forensics]`), when this endpoint serves it. */
    forensics?: ForensicsSpec | null
    /** Prefix-cache offload budgets (`[kv_offload]`), when set. */
    kv_offload?: KvOffloadSpec | null
  } | null
  vram?: {
    gpu: number | null
    nvml_mem: number | null
    self_mem: number | null
    drift: number | null
    anomaly: boolean
  } | null
  /** The runner-local model lifecycle when the runner manages one (Whisper
   *  on demand): whether the model is resident right now, the policy in
   *  force, and the load/unload counters. Absent for runners that keep
   *  their model loaded for life. */
  residency?: ResidencySnapshot | null
}

/** The runner's own report of its residency (`/v1/residency` on its admin
 *  pipe, relayed on the fleet row). `phase` is what the model is doing now;
 *  `policy` is what the config file asks for. */
export interface ResidencySnapshot {
  phase: 'unloaded' | 'loading' | 'loaded' | 'unloading' | 'failed'
  active_leases: number
  waiting_requests: number
  loads: number
  unloads: number
  load_failures: number
  last_load_ms: number | null
  last_error: string | null
  policy: ResidencyConfig
}

/** One configured endpoint (servers/<port>.toml) as /api/servers reports it.
 *  The FILE is the enumeration: a stopped endpoint is still configured and
 *  shows on the fleet page as a stopped row, ready to start again. */
export interface ConfiguredRow {
  port: number
  /** The endpoint's catalog IDENTITY where the registry recognises its weights
   *  file, else the file's raw model value. The manager resolves it (the same
   *  `identify_weights` that makes `display` right); before that this carried
   *  the path and every model selector opened on "select". */
  model: string | null
  /** speculation mechanism the config would wire ("MTP", "DFlash1", "off") */
  spec?: string | null
  /** The catalog artifact behind it ("q8", "q4", "f16"), when identified. */
  artifact?: string | null
  /** The file's model value verbatim - a weights path for anything started
   *  from the catalog. Kept so a caller that needs the FILE does not rebuild it
   *  from the id, and so the edit form can tell whether a hand-edited path
   *  still refers to the model this identity was derived from. */
  weights?: string | null
  running: boolean
  /** catalog labels resolved server-side ("Qwen 3.5 9B" / maker). */
  display?: string | null
  vendor?: string | null
  /** What starting it would get you ("chat", "vision", "transcription", ...),
   *  from the catalog. The only capability answer for an endpoint that is not
   *  running - a live runner advertises its own, a stopped one cannot be
   *  asked. Absent for a model the catalog does not know (a hand-typed GGUF).
   */
  capability?: string[] | null
}

/** One desired-state election (managed.toml): respawned on manager boot. */
export interface FleetElection {
  model: string
  port: number
  max_ctx: number | null
  max_batch: number | null
  keyed: boolean
  runner_version: string | null
  gpu: number | null
  pinned: boolean
}

/** The `[forensics]` config block, mirroring the manager's `ForensicsSpec`. The
 *  Intelligence section flips `enabled`; `auto`/`tool`/`device` round-trip from
 *  the file so a first-class toggle never clobbers a hand-tuned scope. */
/** `[kv_offload]`: how much RAM and disk this endpoint may keep prefixes in.
 *  Budgets only - everything about how the cache behaves is elected in the
 *  engine and measured on the machine it runs on. */
export interface KvOffloadSpec {
  enabled: boolean
  /** host RAM, GiB. The entry point: disk stores through RAM. */
  ram_gb: number
  /** disk, GiB; needs nvme_path. Absent/0 = no disk tier. */
  nvme_gb?: number
  /** where on disk; needs nvme_gb. */
  nvme_path?: string
}

export interface ForensicsSpec {
  enabled: boolean
  /** "off" | "images" | "all"; absent = the product default ("all") on render. */
  auto?: string
  /** expose the on-demand tool; absent = true on render. */
  tool?: boolean
  /** pin the forensic context to another GPU; absent = share the model's GPU. */
  device?: number
}

/** Everything a start sends - mirrors the manager's SpawnSpec. The UI never
 *  sets `pull`: missing pieces are pulled explicitly first (deployWithPull),
 *  so the manager API keeps its no-silent-download contract. */
export interface DeploySpec {
  residency?: ResidencyConfig
  model: string
  /** weights-artifact choice (schema 3), e.g. "q4". Absent = default. */
  artifact?: string
  /** drafter-artifact pin for a model cataloguing more than one, e.g.
   *  "drafter2" (DFlash2) vs "drafter" (DFlash1). Absent = the catalog
   *  default - which is what should track the catalog if the default moves. */
  drafter?: string
  /** opt into native-FP8 plane ingestion (installed fp8-snapshot required). */
  fp8_native?: boolean
  /** false = serve text-only (skip the vision tower); absent = attach when installed. */
  vision?: boolean
  /** A split-tower model's audio tower (EmbeddingGemma 2): true attaches it,
   *  false leaves it out; absent = the catalog default (off). */
  audio?: boolean
  port: number
  max_ctx?: number
  max_batch?: number
  api_key?: string
  /** NVML index; the manager resolves it to a device UUID in the file. */
  gpu?: number
  pinned?: boolean
  persist?: boolean
  /** Edit only: the file hash the page loaded (GET /api/servers/{port}/file).
   *  Save refuses if the file moved since - never clobber a hand-edit. */
  expect_config_hash?: string
  /** SERVER TOOLS this model supplies (its config file): web search... */
  web_search_provider?: string
  web_search_api_key?: string
  /** ...and endpoint-attached MCP servers. */
  mcp_servers?: McpServerEntry[]
  /** KV cache dtype ("f16" | "fp8_e4m3"); absent = the runner's auto. */
  kv_cache_dtype?: string
  /** Ceiling on VRAM this endpoint may hold, in MiB. Absent = the manager
   *  computes a grant that leaves the rest of the card startable; present =
   *  admission honours exactly this and never recomputes. */
  vram_budget?: number
  /** Speculation policy: "off" | "auto" | "ladder" | a pinned draft length.
   *  Absent = leave the key out; the model's own default stands. Setting it
   *  to anything but "off" also resolves the model's drafter, so a spawn that
   *  asked to speculate fails loudly when that file is not downloaded. */
  spec?: string
  /** Confirmed eviction plan (the 507 offer round-tripped through the
   *  user's explicit yes): stop these ports before starting. */
  evict?: number[]
  /** Forensics (`[forensics]`): the Intelligence section's toggle.
   *  Absent = leave the block out (disabled). */
  forensics?: ForensicsSpec
  /** Prefix-cache offload (`[kv_offload]`). Absent = leave the block out. */
  kv_offload?: KvOffloadSpec
}

export interface ResidencyConfig {
  load: 'at_startup' | 'on_demand'
  unload_after_idle_seconds?: number | null
  load_timeout_seconds: number
}

/** One endpoint the manager's refusal says would free enough if stopped. */
export interface EvictCandidate {
  port: number
  display?: string | null
  /** bytes its stop gives back (its budget grant or live ledger). */
  frees: number
  restore_cost?: number | null
}

/** The actionable half of a VRAM refusal: which stops make the start fit. */
export interface EvictionOffer {
  need: number
  residual: number
  candidates: EvictCandidate[]
  /** minimal stop set, cheapest-to-restore first; [] = nothing would help. */
  plan: number[]
}

/** A refused start waiting on the user's explicit yes to the offer -
 *  rendered as the "Stop X and start Y?" confirm dialog (EvictConfirm). */
export interface EvictAsk {
  /** human name of the model the user tried to start */
  label: string
  /** the manager's full refusal text */
  message: string
  offer: EvictionOffer
  /** re-run the same start with the approved stop list */
  retry: (evict: number[]) => Promise<boolean>
}

/** A spawn in flight. The port is chosen client-side before the POST exactly
 *  so the log stream can be tailed while the model loads - the minutes-long
 *  load is the moment feedback matters most. Download-then-start flows are
 *  not here anymore: they are manager-side pull jobs (downloads store). */
export interface Deploying {
  model: string
  port: number
  startedAt: number
  phase: 'starting' | 'failed'
  error?: string
  /** Last few log lines from `/api/logs?target=<port>&follow` - live. */
  log: string[]
}

const BASE_PORT = 11540

/** The part of a manager error a person should read, for a toast.
 *
 *  The manager puts the REASON in the first paragraph and everything that
 *  supports it - the exit code, then the whole runner log tail - after a blank
 *  line. So a paragraph, not a line: an engine refusal states the arithmetic
 *  and then names the fixes in the same sentence, and cutting at the first
 *  newline used to drop exactly that half. Cutting at the first paragraph
 *  keeps the answer and leaves the log to the detail page the toast links to.
 *
 *  Falls back to the whole string, so a one-line error still reads. */

/** The running fleet + desired state + deploy lifecycle - the Manager's
 *  Servers screen state. Polls while a subscriber asks it to. */
/** The model a runner serves, whichever serving role it is in. A row carries
 *  only its own role's key, so every role has to be listed - a chain that
 *  stops early renders the rest nameless. */
/** The manager names a runner's maker from the catalog only, so a model the
 *  catalog does not carry (a hand-started file, a local-dev checkpoint like
 *  Kolibri) arrives with none and its row draws no mark. Fall back to the id
 *  heuristic the reply headers already use, so a fleet row and the chat lane
 *  for the same model agree. */
function withVendor<T extends { vendor?: string | null }>(row: T, id: string | null | undefined): T {
  if (row.vendor || !id) return row
  const vendor = cloudVendor(id)
  return vendor ? { ...row, vendor } : row
}

export function servedId(r: {
  model?: string | null
  embedder?: string | null
  asr?: string | null
  aligner?: string | null
  image?: string | null
  reader?: string | null
  tabular?: string | null
  diarization?: string | null
  segmenter?: string | null
  masker?: string | null
}): string | undefined {
  return (
    r.model ??
    r.embedder ??
    r.asr ??
    r.aligner ??
    r.image ??
    r.reader ??
    r.tabular ??
    r.diarization ??
    r.segmenter ??
    r.masker ??
    undefined
  )
}

export const useFleetStore = defineStore('fleet', () => {
  // The manager's own version, to compare each runner against. Bound once here
  // rather than per call site so `staleRunners` stays a plain computed.
  const models = useModelsStore()
  const rows = ref<FleetRow[]>([])
  const configured = ref<ConfiguredRow[]>([])
  const elections = ref<FleetElection[]>([])
  const deploying = ref<Deploying[]>([])
  const loaded = ref(false)
  const error = ref<string | null>(null)
  /** The emergency (§9): fleet ledgers commit more VRAM than the card has -
   *  the OS is paging VRAM into system RAM and the machine can freeze. */
  const overcommit = ref<{ committed: number; device_total: number } | null>(null)
  /** A VRAM-refused start whose offer awaits the user's yes (one at a time -
   *  a second refusal replaces the first, which is stale by then anyway). */
  const evictAsk = ref<EvictAsk | null>(null)

  /** Pull the manager's structured eviction offer out of a refusal body.
   *  Only an offer with a workable plan opens the dialog; a planless refusal
   *  (nothing unpinned to stop would help) stays a plain failure toast. */
  function evictionOf(
    body: unknown,
  ): { message: string; offer: EvictionOffer } | null {
    const err = (
      body as { error?: { message?: string; eviction?: EvictionOffer } } | null
    )?.error
    const offer = err?.eviction
    if (!offer || !Array.isArray(offer.plan) || offer.plan.length === 0) return null
    return { message: err?.message ?? 'not enough VRAM', offer }
  }

  /** The user said yes: close the dialog and re-run the start with the plan.
   *  Fire-and-forget - the deploying row + outcome toasts carry the feedback
   *  (the drain alone can take up to 30 s before the load even begins). */
  function confirmEvict(): void {
    const ask = evictAsk.value
    if (!ask) return
    evictAsk.value = null
    void ask.retry(ask.offer.plan)
  }

  function cancelEvict(): void {
    evictAsk.value = null
  }

  const bootPorts = computed(() => new Set(elections.value.map((e) => e.port)))
  /** Ports that are taken right now: running, mid-start, promised to a
   *  manager-side download's queued start, or CONFIGURED-but-stopped (the
   *  manager refuses a fresh deploy onto an existing servers/<port>.toml) -
   *  a second start must not grab any of them. */
  const takenPorts = computed(() => {
    const s = new Set(rows.value.map((r) => r.port))
    for (const c of configured.value) s.add(c.port)
    for (const d of deploying.value) if (d.phase !== 'failed') s.add(d.port)
    for (const j of useDownloadsStore().active) {
      const p = j.start?.port
      if (typeof p === 'number') s.add(p)
    }
    return s
  })

  /** Configured endpoints with nothing serving them and nothing in flight on
   *  their port - the "ready to start again" rows.
   *
   *  Liveness comes from `rows` (the runners list) and nothing else. It used to
   *  also require `!c.running`, which is the manager's own flag on
   *  /api/servers - and the two are computed from different predicates:
   *  /api/servers calls a port running if a record exists OR its admin socket
   *  enumerates, while /api/runners only emits a row if `identify` answers or a
   *  record exists. A port that enumerates but does not answer and has no
   *  record therefore satisfied `c.running` while producing no row, so the
   *  endpoint appeared in neither list and simply vanished from the UI - with
   *  its config still on disk the whole time. Trusting one source removes the
   *  asymmetry: no row means not serving, whatever the other flag says, and the
   *  worst case is a Start button on something already running (which the
   *  manager refuses safely) instead of an endpoint the user cannot see. */
  const stopped = computed(() =>
    configured.value.filter(
      (c) =>
        !rows.value.some((r) => r.port === c.port) &&
        !deploying.value.some((d) => d.port === c.port) &&
        !useDownloadsStore().active.some((j) => j.start?.port === c.port),
    ),
  )

  /** Every configured speech endpoint, running or not - what the mic menu
   * starts and stops. Catalog capability is the only signal
   *  available for a stopped one, so an endpoint the catalog does not know
   *  stays out: offering a hand-typed GGUF as a transcriber and having the
   *  mic still not work after starting it would be worse than not offering
   *  it. `busy` is a start or stop already in flight on that port, which is
   *  what disables the row rather than letting a second click race the first.
   */
  const speechEndpoints = computed(() =>
    configured.value
      .filter((c) => c.capability?.includes('transcription'))
      .map((c) => ({
        ...c,
        running: c.running || rows.value.some((r) => r.port === c.port),
        busy: deploying.value.some((d) => d.port === c.port && d.phase !== 'failed'),
      }))
      .sort((a, b) => a.port - b.port),
  )

  /** The next free runner port (allocation grows upward from 11540 - the same
   *  scheme the manager uses, computed here so the log tail can attach). */
  function nextPort(): number {
    let p = BASE_PORT
    while (takenPorts.value.has(p)) p++
    return p
  }

  /** Server-pushed runner rows (SSE 'fleet'): the hot half of refresh() -
   *  status flips, badges, vram - applied immediately. configured/elections/
   *  gpu are cold state and stay on the (relaxed) reconcile poll. */
  function applyRunnerRows(r: FleetRow[]): void {
    rows.value = r.map((x) => withVendor(x, servedId(x)))
    deploying.value = deploying.value.filter((d) => {
      if (d.phase !== 'starting') return true
      const row = r.find((x) => x.port === d.port)
      if (!row || row.uptime_s === null) return true
      return row.uptime_s * 1000 > Date.now() - d.startedAt + 5000
    })
  }

  async function refresh(): Promise<void> {
    try {
      const [r, c, e, g] = await Promise.all([
        fetch('/api/runners').then((x) => x.json() as Promise<FleetRow[]>),
        fetch('/api/servers').then((x) => x.json() as Promise<ConfiguredRow[]>),
        fetch('/api/elections').then(
          (x) => x.json() as Promise<{ elections: FleetElection[] }>,
        ),
        fetch('/api/gpu')
          .then(
            (x) =>
              x.json() as Promise<{
                reconciliation?: {
                  overcommit?: { committed: number; device_total: number } | null
                } | null
              }>,
          )
          .catch(() => null),
      ])
      rows.value = r.map((x) => withVendor(x, servedId(x)))
      configured.value = c.map((x) => withVendor(x, x.model))
      elections.value = e.elections ?? []
      overcommit.value = g?.reconciliation?.overcommit ?? null
      // The spawn POST holds until the health gate passes, but this poll can
      // see the new runner earlier - fold a 'starting' entry into the live
      // row instead of showing both. A row only supersedes the entry if it's
      // YOUNGER than the attempt (uptime guards the takeover case, where the
      // draining incumbent still owns the port).
      deploying.value = deploying.value.filter((d) => {
        if (d.phase !== 'starting') return true
        const row = r.find((x) => x.port === d.port)
        if (!row || row.uptime_s === null) return true
        return row.uptime_s * 1000 > Date.now() - d.startedAt + 5000
      })
      error.value = null
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loaded.value = true
    }
  }

  // Poll while at least one view holds the store open (Servers panel, deploy
  // dialog). Reference-counted so two subscribers don't double-poll.
  let holds = 0
  let timer: number | undefined
  function hold(): () => void {
    holds++
    if (holds === 1) {
      void refresh()
      // 3s is the POLLING truth; with the push stream live the fast half
      // (runner rows) arrives as events and this becomes a 15s reconcile
      // for the cold endpoints (configured/elections/gpu).
      let beat = 0
      timer = window.setInterval(() => {
        beat++
        const live = usePushStore().live
        if (!live || beat % 5 === 0) void refresh()
      }, 3000)
    }
    let released = false
    return () => {
      if (released) return
      released = true
      holds--
      if (holds === 0 && timer !== undefined) {
        clearInterval(timer)
        timer = undefined
      }
    }
  }

  /** Tail a starting runner's log into its Deploying entry (last 6 lines). */
  function tailLog(d: Deploying, abort: AbortController): void {
    void (async () => {
      try {
        const res = await fetch(
          `/api/logs?target=${d.port}&follow=true&history=true&tail=20`,
          { signal: abort.signal },
        )
        if (!res.ok || !res.body) return
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let carry = ''
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          carry += decoder.decode(value, { stream: true })
          const lines = carry.split('\n')
          carry = lines.pop() ?? ''
          const fresh = lines.map((l) => l.trim()).filter(Boolean)
          if (fresh.length) d.log = [...d.log, ...fresh].slice(-6)
        }
      } catch {
        // aborted (deploy finished) or stream error - the tail is best-effort
      }
    })()
  }

  /** Shared spawn lifecycle for deploy (POST /api/runners) and redeploy
   *  (POST /api/runners/{port}/switch - the same-port takeover behind the
   *  edit page's Save). Live feedback: the starting ROW (clean, no logs) plus
   *  an outcome toast; the full error/log detail lives on the server's page. */
  async function runSpawn(
    spec: DeploySpec,
    url: string,
    kind: 'deploy' | 'redeploy',
  ): Promise<boolean> {
    const toasts = useToastsStore()
    const entry: Deploying = {
      model: spec.model,
      port: spec.port,
      startedAt: Date.now(),
      phase: 'starting',
      log: [],
    }
    deploying.value = [...deploying.value, entry]
    const abort = new AbortController()
    // Give the manager a beat to create the log file before tailing it.
    setTimeout(() => tailLog(entry, abort), 700)
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(spec),
      })
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as {
          error?: { message?: string }
        } | null
        // a VRAM refusal with a workable plan becomes the confirm dialog
        // instead of a dead-end toast (not on a retry - a plan that already
        // ran and still refused is a real failure)
        const ask = spec.evict?.length ? null : evictionOf(body)
        if (ask) {
          deploying.value = deploying.value.filter((d) => d !== entry)
          evictAsk.value = {
            label: modelLabel(spec.model),
            message: ask.message,
            offer: ask.offer,
            retry: (evict) => runSpawn({ ...spec, evict }, url, kind),
          }
          return false
        }
        throw new Error(body?.error?.message ?? `HTTP ${res.status}`)
      }
      deploying.value = deploying.value.filter((d) => d !== entry)
      await refresh()
      // a (re)start may have changed the endpoint's advertised tools
      useModelsStore().invalidateCaps()
      toasts.push({
        tone: 'good',
        title:
          kind === 'deploy'
            ? `${modelLabel(spec.model)} is running`
            : `${modelLabel(spec.model)} restarted`,
        description: `port ${spec.port}`,
        to: { name: 'server-detail', params: { port: String(spec.port) } },
      })
      return true
    } catch (e) {
      // The manager's spawn error carries the runner's log tail - the toast
      // shows only the first line; the model's page has the whole story.
      entry.phase = 'failed'
      entry.error = e instanceof Error ? e.message : String(e)
      deploying.value = [...deploying.value]
      toasts.push({
        tone: 'bad',
        title: `${modelLabel(spec.model)} failed to start`,
        description: reasonOf(entry.error),
        to: { name: 'server-detail', params: { port: String(spec.port) } },
        duration: 10000,
      })
      return false
    } finally {
      abort.abort()
    }
  }

  function deploy(spec: DeploySpec): Promise<boolean> {
    return runSpawn(spec, '/api/runners', 'deploy')
  }

  /** Download-then-start, MANAGER-owned: one POST hands the manager the pull
   *  AND the start plan with the user's configured variables - the download,
   *  the progress, and the spawn all live server-side, so closing this tab
   *  changes nothing. The no-silent-download contract holds: the user clicked
   *  a button that named the bytes. Progress + cancel/resume ride the
   *  downloads store (header indicator + the Models page rows). */
  async function deployWithPull(
    spec: DeploySpec,
    modelId: string,
    artifacts: string[],
    kind: 'deploy' | 'redeploy' = 'deploy',
  ): Promise<boolean> {
    const toasts = useToastsStore()
    const dl = useDownloadsStore()
    try {
      await dl.pull(modelId, artifacts, {
        spec,
        action: kind === 'deploy' ? 'spawn' : 'switch',
      })
      return true
    } catch (e) {
      toasts.push({
        tone: 'bad',
        title: `${spec.model} - download could not start`,
        description: reasonOf(e instanceof Error ? e.message : String(e)),
        duration: 10000,
      })
      return false
    }
  }

  /** Edit = same-port takeover: drain the incumbent, relaunch with the new
   *  settings. The endpoint keeps its API key and untouched launch facts
   *  (GPU pin, fp8 planes - manager-side merge), so clients keep working
   *  across the redeploy. */
  function redeploy(port: number, spec: DeploySpec): Promise<boolean> {
    return runSpawn({ ...spec, port }, `/api/runners/${port}/switch`, 'redeploy')
  }

  /** The Advanced editor's Save: write the endpoint's config FILE verbatim
   *  (hash-guarded server-side - a file that moved since the edit opened is
   *  refused, never clobbered) and restart the model from it. */
  async function applyFile(
    port: number,
    content: string,
    expectHash: string | undefined,
    model: string,
    defer = false,
  ): Promise<boolean> {
    const toasts = useToastsStore()
    if (defer) {
      // nothing starts, so no progress row and no log tail - just the write
      try {
        const res = await fetch(`/api/servers/${port}/file`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, expect_hash: expectHash, apply: 'defer' }),
        })
        const b = (await res.json().catch(() => null)) as { error?: { message?: string } } | null
        if (!res.ok) throw new Error(b?.error?.message ?? `HTTP ${res.status}`)
        await refresh()
        const running = rows.value.some((r) => r.port === port && r.status === 'running')
        toasts.push({
          tone: 'good',
          title: `${modelLabel(model)} saved`,
          description: running
            ? `port ${port} - the running model keeps its current settings until you restart it`
            : `port ${port} - takes effect the next time it starts`,
          to: { name: 'server-detail', params: { port: String(port) } },
        })
        return true
      } catch (e) {
        toasts.push({
          tone: 'bad',
          title: `${modelLabel(model)} - save failed`,
          description: reasonOf(e instanceof Error ? e.message : String(e)),
          duration: 10000,
        })
        return false
      }
    }
    const entry: Deploying = {
      model,
      port,
      startedAt: Date.now(),
      phase: 'starting',
      log: [],
    }
    deploying.value = [...deploying.value, entry]
    const abort = new AbortController()
    setTimeout(() => tailLog(entry, abort), 700)
    try {
      const res = await fetch(`/api/servers/${port}/file`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, expect_hash: expectHash }),
      })
      const body = (await res.json().catch(() => null)) as {
        applied?: string
        error?: { message?: string }
      } | null
      if (!res.ok) {
        throw new Error(body?.error?.message ?? `HTTP ${res.status}`)
      }
      deploying.value = deploying.value.filter((d) => d !== entry)
      await refresh()
      useModelsStore().invalidateCaps()
      // "live" = the save changed only tools/web search, which the running
      // model re-reads per request - it never stopped serving. Anything
      // engine-binding went through the usual drain + takeover.
      const live = body?.applied === 'live'
      toasts.push({
        tone: 'good',
        title: `${modelLabel(model)} ${live ? 'updated live' : 'restarted'}`,
        description: live
          ? `port ${port} - tools/web search applied, the model kept serving`
          : `port ${port} - running the edited file`,
        to: { name: 'server-detail', params: { port: String(port) } },
      })
      return true
    } catch (e) {
      entry.phase = 'failed'
      entry.error = e instanceof Error ? e.message : String(e)
      deploying.value = [...deploying.value]
      toasts.push({
        tone: 'bad',
        title: `${modelLabel(model)} - config save failed`,
        description: reasonOf(entry.error),
        to: { name: 'server-detail', params: { port: String(port) } },
        duration: 10000,
      })
      return false
    } finally {
      abort.abort()
    }
  }

  /** Save the settings without applying them: the manager renders the spec
   *  into servers/<port>.toml and touches nothing else. A running model keeps
   *  serving what it loaded until it is next restarted; a stopped one stays
   *  stopped rather than being started by the act of editing it.
   *
   *  No `deploying` entry and no log tail - nothing is starting, so a progress
   *  row would be theatre. The toast says what actually happened. */
  async function saveOnly(port: number, spec: DeploySpec, model: string): Promise<boolean> {
    const toasts = useToastsStore()
    try {
      const res = await fetch(`/api/runners/${port}/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...spec, port, apply: 'defer' }),
      })
      const body = (await res.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      if (!res.ok) throw new Error(body?.error?.message ?? `HTTP ${res.status}`)
      await refresh()
      const running = rows.value.some((r) => r.port === port && r.status === 'running')
      toasts.push({
        tone: 'good',
        title: `${modelLabel(model)} saved`,
        description: running
          ? `port ${port} - the running model keeps its current settings until you restart it`
          : `port ${port} - takes effect the next time it starts`,
        to: { name: 'server-detail', params: { port: String(port) } },
      })
      return true
    } catch (e) {
      toasts.push({
        tone: 'bad',
        title: `${modelLabel(model)} - save failed`,
        description: reasonOf(e instanceof Error ? e.message : String(e)),
        duration: 10000,
      })
      return false
    }
  }

  /** Start a STOPPED configured endpoint from its file, verbatim - same
   *  port, same settings, same API key. The §11.4 "what you started stays
   *  started" posture resumes: the start records an election again. */
  async function startConfigured(
    port: number,
    label: string,
    evict?: number[],
  ): Promise<boolean> {
    const toasts = useToastsStore()
    const entry: Deploying = {
      model: label,
      port,
      startedAt: Date.now(),
      phase: 'starting',
      log: [],
    }
    deploying.value = [...deploying.value, entry]
    const abort = new AbortController()
    setTimeout(() => tailLog(entry, abort), 700)
    try {
      const res = await fetch(`/api/servers/${port}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ evict: evict ?? [] }),
      })
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as {
          error?: { message?: string }
        } | null
        const ask = evict?.length ? null : evictionOf(body)
        if (ask) {
          deploying.value = deploying.value.filter((d) => d !== entry)
          evictAsk.value = {
            label: modelLabel(label),
            message: ask.message,
            offer: ask.offer,
            retry: (ev) => startConfigured(port, label, ev),
          }
          return false
        }
        throw new Error(body?.error?.message ?? `HTTP ${res.status}`)
      }
      deploying.value = deploying.value.filter((d) => d !== entry)
      await refresh()
      useModelsStore().invalidateCaps()
      toasts.push({
        tone: 'good',
        title: `${modelLabel(label)} is running`,
        description: `port ${port}`,
        to: { name: 'server-detail', params: { port: String(port) } },
      })
      return true
    } catch (e) {
      entry.phase = 'failed'
      entry.error = e instanceof Error ? e.message : String(e)
      deploying.value = [...deploying.value]
      toasts.push({
        tone: 'bad',
        title: `${modelLabel(label)} failed to start`,
        description: reasonOf(entry.error),
        to: { name: 'server-detail', params: { port: String(port) } },
        duration: 15000,
      })
      return false
    } finally {
      abort.abort()
    }
  }

  /** Remove a STOPPED endpoint's configuration (deletes servers/<port>.toml
   *  + any election). Throws with the manager's message on refusal. */
  async function removeConfigured(port: number): Promise<void> {
    const res = await fetch(`/api/servers/${port}`, { method: 'DELETE' })
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      throw new Error(body?.error?.message ?? `HTTP ${res.status}`)
    }
    await refresh()
  }

  function dismissFailed(port: number): void {
    deploying.value = deploying.value.filter(
      (d) => !(d.port === port && d.phase === 'failed'),
    )
  }

  async function stop(port: number): Promise<void> {
    // Optimistic: the row shows draining immediately; the poll corrects it.
    const row = rows.value.find((r) => r.port === port)
    if (row) row.status = 'draining'
    try {
      await fetch(`/api/runners/${port}`, { method: 'DELETE' })
    } finally {
      await refresh()
    }
  }

  async function setPinned(port: number, pinned: boolean): Promise<void> {
    await fetch(`/api/runners/${port}/pin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pinned }),
    })
    await refresh()
  }

  /** Start-on-boot toggle. Throws with the manager's message on refusal
   *  (e.g. an adopted runner whose config the manager honestly doesn't know). */
  async function setPersist(port: number, persist: boolean): Promise<void> {
    const res = await fetch(`/api/runners/${port}/persist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persist }),
    })
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      throw new Error(body?.error?.message ?? `HTTP ${res.status}`)
    }
    await refresh()
  }

  /** Runners serving on an OLDER build than the manager itself.
   *
   *  This is not "there is an update to download". The runner ships in the same
   *  package as the manager, so after a package refresh the new paddock-runner
   *  is already on disk beside the exe - a process that was serving before the
   *  swap simply carries on with the old image until it is restarted. Telling
   *  someone to "update" would send them looking for a download that does not
   *  exist; the action is to restart that model.
   *
   *  A version PIN is excluded deliberately. `config.runner_version` means
   *  somebody chose that version (rollback is a supported move, doc 11.5), and
   *  flagging a deliberate choice as a problem is how a warning becomes noise
   *  that people learn to ignore. */
  const staleRunners = computed(() =>
    rows.value.filter(
      (r) =>
        r.status === 'running' &&
        !!r.version &&
        !r.config?.runner_version &&
        !!models.serverVersion &&
        r.version !== models.serverVersion,
    ),
  )

  return {
    applyRunnerRows,
    rows,
    configured,
    staleRunners,
    stopped,
    speechEndpoints,
    overcommit,
    evictAsk,
    confirmEvict,
    cancelEvict,
    elections,
    deploying,
    loaded,
    error,
    bootPorts,
    takenPorts,
    nextPort,
    refresh,
    hold,
    deploy,
    deployWithPull,
    redeploy,
    saveOnly,
    applyFile,
    startConfigured,
    removeConfigured,
    dismissFailed,
    stop,
    setPinned,
    setPersist,
  }
})
