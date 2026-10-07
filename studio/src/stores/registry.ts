import { defineStore } from 'pinia'
import { ref } from 'vue'
import { gpuApi, registryApi, type CatalogModel } from '@/lib/api'

/** The one thing a user actually chooses: how many requests run at once.
 *  Context is not here - it is derived per model from what the card can back
 *  at this concurrency, capped by that model's own trained window. A fixed
 *  context list was arbitrary in both directions: it offered 131072 to models
 *  capped at 32768 and hid qwen3.5-9B's real 262144. */
export interface Envelope {
  batch: number
  /** The width these numbers were PRICED at - which is what the runner will
   *  serve, not necessarily what was asked. Label the KV row from here, never
   *  from the form control, or the panel can say "8-bit" over bytes counted
   *  at 16. */
  kv_dtype: 'f16' | 'fp8_e4m3' | 'f32'
  /** What the form asked for, when it differs from `kv_dtype`. */
  kv_asked?: 'f16' | 'fp8_e4m3' | 'f32'
  /** Why the ask was not honoured, in words ("this GPU has no FP8 tensor
   *  cores"). Absent when nothing was overridden. */
  kv_downgraded?: string | null
  /** Whether the vision/audio tower is inside these numbers. */
  vision?: boolean
  /** The ceiling these numbers were priced under, in bytes. */
  budget?: number | null
  /** The server's own --max-ctx / --max-batch. `server_ctx` caps what this
   *  server will actually serve regardless of what the card could back. */
  server_ctx: number
  server_batch: number
}

/** Host memory as `/api/models/estimate` reports it, in bytes. Separate from
 *  every VRAM figure deliberately: prefix-cache offload spends system RAM, and
 *  folding the two together is the mistake this type exists to prevent. */
export interface HostMem {
  available?: number | null
  /** null where the manager cannot read it (see hostmem.rs). */
  total: number | null
  /** ceilings the configured fleet has already promised its caches. */
  committed: number
  /** what this endpoint is asking for, when the form set it. */
  requested: number | null
  /** where a disk tier lands when nobody names a folder - resolved from the
   *  same shared data root the runner uses, so the form can show the real
   *  path instead of asking for one. */
  cache_dir: string
}

/** Itemized footprint from `/api/models/estimate`, in bytes.
 *
 *  The split matters: `resident` is the floor that must fit for the model to
 *  load, and it does not include the KV cache. The cache is a shared pool that
 *  takes whatever VRAM is left, so it never decides whether a model fits - it
 *  decides how much context you get (`max_ctx`). Pricing the dense worst case
 *  as a requirement is what once made a 27B look like it needed 107 GB. */
export interface Estimate {
  weights: number
  /** The mmproj tower - VISION or AUDIO - resident from load whenever one is
   *  wired. Its own line because it is weights, not overhead. 0 for a
   *  text-only model, 0 for whisper (whose speech encoder ships inside the
   *  weights file rather than as a companion), and 0 when the start form's
   *  vision switch is off - the supervisor drops the mmproj then, so charging
   *  it would over-state by its whole file size. */
  tower: number
  /** Persistent serving scratch the engine pins for this model beyond its
   *  weights - declared by the weights artifact, measured per release.
   *  gemma-4-26B-A4B's mixture-of-experts staging is ~5.8 GB; 0 for most
   *  models. Optional because older managers don't report it. */
  workspace?: number
  /** Per-slot state that is flat in context: recurrent/DeltaNet state, and an
   *  encoder-decoder's static cross-attention cache (whisper holds a whole
   *  30 s audio window per concurrent transcription - 117 MB at the fp8
   *  default - however short the clip is). */
  state: number
  overhead: number
  /** `overhead` term by term, plus the flat floor outside it. One number
   *  called "engine overhead" is unreadable at 27B scale - the panel now
   *  names the parts rather than asking the reader to trust a total. */
  overhead_parts?: {
    allocator_slack: number
    prefix_checkpoints: number
    logits: number
    block_tables: number
    conv_scratch: number
    spec_state: number
    offload_staging: number
    /** the self-sized checkpoint pool above its floor - charged against the
     *  KV pool, not the resident floor, so it explains a smaller context
     *  rather than a bigger footprint. */
    prefix_pool_extra: number
    fixed: number
  }
  resident: number
  kv_pool: number
  /** The headline: longest context this card serves for this model at the
   *  chosen concurrency, never above the model's trained window. */
  max_ctx: number
  /** The model's trained window, so we can say which ceiling bit. */
  model_max_ctx: number
  limited_by: 'model' | 'vram' | 'not_applicable'
  kv_bytes_per_token: number
  fit:
    | { verdict: 'fits'; headroom_bytes: number }
    | { verdict: 'tight'; headroom_bytes: number }
    | { verdict: 'does_not_fit'; short_by_bytes: number }
}

/** One weights-artifact's answer. `known: false` = not downloaded, so there
 *  is no file to measure and we say so rather than inventing a number.
 *  `kind: 'encoder'` = embeddings/rerank: one forward pass per call, no cache
 *  held between them, so context and concurrency cost nothing. */
export interface ModelEstimate {
  kv_dtype?: string
  known: boolean
  kind?: 'generative' | 'encoder' | 'image'
  weights: number
  /** the shared vision tower's bytes - the same for every weights choice, so
   *  it is repeated on each row rather than belonging to one of them. */
  vision?: number
  estimate?: Estimate
  /** Context available at each concurrency - the trade-off, precomputed. */
  curve?: { at: number; ctx: number }[]
  kv_bytes_per_token?: number
  reason?: string
}

/** The decoding parameters this checkpoint's own authors published, as the
 *  runner will apply them when a request (and the config file) says nothing.
 *  Absent when the model publishes none, or when it is not downloaded - the
 *  architecture is read from the file, never declared. */
export interface ElectedSampling {
  temperature: number
  top_k: number
  top_p: number
  min_p: number
  /** Citation, specific enough to re-check by hand. */
  source: string
  /** The second row, for families that publish one for thinking-off. */
  instruct?: { temperature: number; top_k: number; top_p: number; min_p: number }
}

/** A model's estimator answer (schema 3): one fit row per WEIGHTS artifact -
 *  Q8 and Q4 are different footprints of one model. */
export interface ModelFit {
  kind?: 'generative' | 'encoder' | 'image'
  artifacts: Record<string, ModelEstimate>
  sampling?: ElectedSampling | null
}

/** VRAM the server can actually give a model: physically free plus whatever the
 *  currently-loaded model would hand back, since paddock releases one before
 *  loading another. */
export interface EstimateDevice {
  unified?: boolean
  physical?: number
  planning_basis?: string
  /** What a model would get: `free_now` plus whatever the loaded model
   *  releases when it is swapped out. This is the one the fit is judged on. */
  free: number
  /** Unallocated on the card at this instant - the number nvidia-smi agrees
   *  with. Shown alongside `free` so the two can be reconciled. */
  free_now: number
  total: number
  name: string | null
  held_by_loaded_model: number
  used_by_others: number
  /** The subset of `used_by_others` that is our own pinned runners - other
   *  paddock models resident by policy. Broken out by the manager precisely so
   *  a panel can name them instead of filing them under "other apps". */
  held_by_pinned?: number
  /** paddock's own CUDA context and workspaces - resident but not part of
   *  `held_by_loaded_model`. Broken out so the terms sum to `total`. */
  paddock_runtime: number
}

/** Model registry: the catalog to pull from (Cloudflare R2), the server's
 *  models folder + free disk, and in-flight downloads. Separate from the
 *  `models` store (which lists what's already loadable). */
export const useRegistryStore = defineStore('registry', () => {
  const enabled = ref(false)
  const models = ref<CatalogModel[]>([])
  const modelsDir = ref('')
  const diskFree = ref(0)
  const diskTotal = ref(0)
  // the server GPU paddock runs on - for the "fits your VRAM?" signal
  const gpuName = ref('')
  const gpuVram = ref(0) // total VRAM in bytes (0 = unknown / CPU host)
  /** measured VRAM the LOADED model holds (engine-reported; 0 = unknown). */
  const modelMem = ref(0)
  const loading = ref(false)
  const error = ref<string | null>(null)
  // will-it-fit, priced by the server at the current envelope
  const envelope = ref<Envelope | null>(null)
  const estimates = ref<Record<string, ModelFit>>({})
  /** Host memory, as the last estimate priced it. `total` is null on a
   *  platform the manager cannot ask - the readout then shows the commitment
   *  without a denominator rather than inventing one. */
  const estHost = ref<HostMem | null>(null)
  const estDevice = ref<EstimateDevice | null>(null)
  const estimating = ref(false)

  async function refresh(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      // server info first: the models folder + free/total disk (always available)
      const info = (await fetch('/api/server').then((r) => r.json())) as {
        registry?: { enabled?: boolean; models_dir?: string; disk_free?: number; disk_total?: number }
      }
      enabled.value = info.registry?.enabled ?? false
      modelsDir.value = info.registry?.models_dir ?? ''
      diskFree.value = info.registry?.disk_free ?? 0
      diskTotal.value = info.registry?.disk_total ?? 0
      models.value = enabled.value ? (await registryApi.catalog()).models : []
      // the GPU paddock runs on (one-shot snapshot; the telemetry WebSocket only
      // streams while the dock is open, so we read it directly here).
      try {
        const snap = await gpuApi.get()
        const g = snap.gpus?.[0]
        gpuName.value = g?.name ?? ''
        gpuVram.value = g?.mem_total ?? 0
        // fleet-wide: what the running models hold, from their own ledgers
        modelMem.value = (snap.reconciliation?.runners ?? []).reduce(
          (s, r) => s + (r.self_mem ?? 0),
          0,
        )
      } catch {
        gpuName.value = ''
        gpuVram.value = 0
        modelMem.value = 0
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  // NOTE: downloading moved wholesale to the downloads store -
  // pulls are MANAGER-side jobs now (SSE progress, cancel/resume, queued
  // starts), not a per-tab poll loop. This store keeps the catalog + fit math.

  /** Does a model's total size fit in free disk (with a small headroom)? */
  function fits(m: CatalogModel): boolean {
    return diskFree.value === 0 || m.total_size < diskFree.value - 1024 * 1024 * 1024
  }

  /** Ask the server to price the whole catalog at `env`.
   *
   *  This replaced a client-side `total_size * 1.2 + 1 GB` guess, which was
   *  wrong by 2.8× on qwen3.5-9B because it charged no KV at all - and KV is
   *  usually the biggest number on the card. The math lives server-side
   *  because KV geometry is architecture-specific (qwen3.5/3.6 are DeltaNet
   *  hybrids where only ~1/4 of blocks cache anything; gemma4 and gpt-oss cap
   *  most blocks at a sliding window), and only the GGUF header knows.
   */
  let estimateSequence = 0
  async function estimate(over?: {
    freeingPort?: number | null
    batch?: number
    kv?: string
    spec?: boolean
    vision?: boolean
    /** A split-tower model's audio tower, priced by its own switch. */
    audio?: boolean
    /** This device's compute capability, `[major, minor]`. Sent so the server
     *  can price the KV width the RUNNER will serve: a card with no FP8
     *  tensor cores has fp8 downgraded to f16 at load, which DOUBLES the KV
     *  pool, and an estimate that missed that drew half the cache the server
     *  then allocates. */
    cc?: [number, number]
    /** Ceiling in MiB ("how much of the card"). The estimate prices against
     *  this instead of all free VRAM, or the panel draws an endpoint bigger
     *  than the one that will start. */
    budget?: number | null
    /** NVML index of the card to price against. Absent = 0, which is what the
     *  server has always defaulted to - and what every estimate silently used
     *  while the form's GPU picker moved. */
    gpu?: number
    /** Prefix-cache memory budget in GiB. Arming the cache reserves device
     *  staging out of the same VRAM the pool is sized from, so an estimate
     *  that ignored it would draw a context the runner then seats smaller -
     *  and it is what the host-RAM readout is priced from. */
    offloadRamGb?: number
  }): Promise<void> {
    const sequence = ++estimateSequence
    const q = new URLSearchParams()
    if (over?.freeingPort) q.set('freeing_port', String(over.freeingPort))
    const batch = over?.batch ?? envelope.value?.batch
    if (batch) q.set('batch', String(batch))
    // KV precision and speculation move the answer as much as concurrency does:
    // 8-bit KV halves the dominant term (so it roughly doubles the context that
    // fits), and speculating holds a drafter resident and widens the verify
    // plane. A fit figure that ignored either would contradict the very form
    // the user is filling in.
    const kv = over?.kv ?? envelope.value?.kv_dtype
    if (kv) q.set('kv', kv)
    if (over?.spec !== undefined) q.set('spec', String(over.spec))
    // Both are omitted rather than defaulted when the caller says nothing: the
    // server reads absent as "charge the tower" and "honour the kv request",
    // which is the pre-existing behaviour.
    if (over?.vision !== undefined) q.set('vision', String(over.vision))
    if (over?.audio !== undefined) q.set('audio', String(over.audio))
    if (over?.cc) q.set('cc', `${over.cc[0]}.${over.cc[1]}`)
    if (over?.budget) q.set('budget', String(over.budget))
    if (over?.gpu !== undefined && over.gpu >= 0) q.set('gpu', String(over.gpu))
    if (over?.offloadRamGb) q.set('offload_ram_gb', String(over.offloadRamGb))
    estimating.value = true
    try {
      const response = await fetch(`/api/models/estimate?${q}`)
      if (!response.ok) throw new Error(`Estimate failed: ${response.status}`)
      const r = await response.json()
      if (sequence !== estimateSequence) return
      envelope.value = r.envelope
      estDevice.value = r.device
      estHost.value = r.host ?? null
      estimates.value = r.models ?? {}
    } catch {
      // leave the previous answer standing rather than flashing a wrong one
    } finally {
      if (sequence === estimateSequence) estimating.value = false
    }
  }

  return {
    enabled,
    models,
    modelsDir,
    diskFree,
    diskTotal,
    gpuName,
    gpuVram,
    modelMem,
    loading,
    error,
    envelope,
    estimates,
    estDevice,
    estHost,
    estimating,
    refresh,
    fits,
    estimate,
  }
})
