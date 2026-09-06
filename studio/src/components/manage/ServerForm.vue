<script setup lang="ts">
// Configure / edit - one full page, two modes (route-decided):
//
//   /manage/models/start/:model  the workload step: the model was picked on
//                                models/start ('custom' = a typed path);
//                                here you set context, concurrency, port, key
//   /manage/models/:port/edit    edit: prefilled from the as-started config;
//                                Save is a same-port takeover (drain, then
//                                relaunch) - the endpoint keeps its key
//
// The page is a PROPOSAL, not a questionnaire: the estimator prices the
// context x concurrency trade-off live and defaults are prefilled. (The
// scriptable form of a saved server is its own config file -
// `paddock-runner --config servers/<port>.toml` - not a CLI echo here.)
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { parse as tomlParse, stringify as tomlStringify } from 'smol-toml'
import { useRegistryStore } from '@/stores/registry'
import { useFleetStore, type DeploySpec } from '@/stores/fleet'
import { useReadinessStore } from '@/stores/readiness'
import { archBlockReason } from '@/lib/arch-floor'
import { useConnectorsStore, type Connector } from '@/stores/connectors'
import { useToastsStore } from '@/stores/toasts'
import { gpuApi, projectConfig } from '@/lib/api'
import { fmtVram as gb, fmtTokens as fmtCtx, fmtBytes } from '@/lib/format'
import { modelLabel } from '@/lib/model-name'
import { SEARCH_PROVIDERS, searchLabel, searchProvider } from '@/lib/websearch'
import Icon from '@/components/Icon.vue'
import Select from '@/components/ui/Select.vue'
import NumberField from '@/components/ui/NumberField.vue'
import { CTX_CUSTOM, CTX_MAX, CTX_STEPS, ctxCapOf, ctxFits, ctxLadder } from '@/lib/ctx-ladder'
import FieldLabel from '@/components/manage/FieldLabel.vue'
import TextInput from '@/components/ui/TextInput.vue'
import Switch from '@/components/ui/Switch.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import RadioGroup from '@/components/ui/RadioGroup.vue'
import RadioItem from '@/components/ui/RadioItem.vue'
import ToggleGroup from '@/components/ui/ToggleGroup.vue'
import ToggleGroupItem from '@/components/ui/ToggleGroupItem.vue'
import Dialog from '@/components/ui/Dialog.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import FitChart from '@/components/manage/FitChart.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import SearchLogo from '@/components/manage/SearchLogo.vue'

const route = useRoute()
const router = useRouter()
const reg = useRegistryStore()
const fleet = useFleetStore()
const ready = useReadinessStore()
const toasts = useToastsStore()

const editPort = computed<number | null>(() =>
  route.name === 'server-edit' ? Number(route.params.port) : null,
)
const isEdit = computed(() => editPort.value !== null)
const editRow = computed(() => fleet.rows.find((r) => r.port === editPort.value))
/** The edited endpoint's human name for the crumb/title (port as fallback).
 *  A stopped endpoint has no live row - the configured list names it. */
const editName = computed(() => {
  const r = editRow.value
  const c = fleet.configured.find((x) => x.port === editPort.value)
  const n = r?.display ?? c?.display ?? modelLabel(r?.model ?? r?.embedder ?? c?.model)
  return n || String(editPort.value ?? '')
})
/** The name the crumb and the heading show while editing.
 *
 *  Follows the CURRENT selection, not the as-loaded config. Every other field
 *  on this page shows pending state - the context you just picked, the
 *  concurrency, the cache width - so a title that went on naming the old model
 *  after you switched was the one element still describing an endpoint you no
 *  longer have in front of you.
 *
 *  Identity is not lost by this: the endpoint is its port (manager-runner doc
 *  §3), and the port is in the URL and in the Port field, neither of which
 *  moves. This is a label, not the identity. */
/** What a blank sampling field resolves to on this model, as a placeholder.
 *
 *  The election is mode-dependent for families that publish two rows, and a
 *  placeholder can only hold one - so the thinking row (the mode every one of
 *  them ships in) is shown, and the instruct row rides in the tooltip rather
 *  than being silently dropped. Empty string when nothing is published, which
 *  renders as a plain empty field: the wire default then applies and there is
 *  nothing to promise. */
function electedPlaceholder(key?: string): string {
  const e = key ? reg.estimates[model.value]?.sampling : null
  if (!e || !key) return ''
  const v = (e as unknown as Record<string, number>)[key]
  return typeof v === 'number' ? String(v) : ''
}

const editTitle = computed(() => {
  if (model.value === '__custom') return customModel.value.trim() || 'your own model file'
  return catModel.value?.display || modelLabel(model.value) || editName.value
})

/** The model picked on /servers/new ('custom' = a typed name/path). */
const paramModel = computed(() =>
  route.name === 'server-new-config' ? String(route.params.model) : null,
)

// ── form state ───────────────────────────────────────────────────────────────
const model = ref('')
const customModel = ref('')
/** weights-artifact choice (schema 3) - the quality pick, per-artifact fit. */
const artifactId = ref('')
const fp8Native = ref(false)
// Vision defaults on - "I thought it was vision by default" is the correct
// instinct; the switch makes the default visible and
// gives the VRAM-conscious an explicit text-only serve.
const withVision = ref(true)
// "Just me" is the default workload: the common first
// start is one person trying a model, and batch=1 reserves the least VRAM.
// Scaling up is a visible choice on this same screen.
const batch = ref(1)
const ctx = ref(32768)
const gpuIndex = ref<number | null>(null)
const port = ref(0)
const kvDtype = ref<string>('f16')
const specPolicy = ref<string>('on')
/** Pinned drafter-artifact id, '' = follow the catalog default. Kept separate
 *  from specPolicy: whether to speculate and which drafter to use are
 *  independent choices, and a pin should survive toggling spec off and on. */
const drafterId = ref<string>('')
const apiKey = ref('')
const pinned = ref(false)
const persist = ref(true)
const busy = ref(false)
// Intelligence / context enrichment: forensics ([forensics].enabled).
// The toggle owns `enabled`; a hand-set scope (auto/tool/device from the file or
// the Advanced tab) rides through `forensicsExtra` so flipping the switch never
// clobbers it.
const forensicsOn = ref(false)
const forensicsExtra = ref<{ auto?: string | null; tool?: boolean | null; device?: number | null }>(
  {},
)
// KV offloading ([kv_offload]): budgets only. How the cache behaves is elected
// in the engine and measured on the machine; what is the operator's to decide
// is how much of their RAM and disk it may use.
/** What the switch arms when nobody says otherwise. Enough to hold a few
 *  agentic sessions on a normal box; the readout under the field says what it
 *  costs the machine, and the number is the operator's to change. */
const DEFAULT_CACHE_RAM_GB = 8

/** The master switch. Comparable engines lead with one - SGLang's
 *  `--enable-hierarchical-cache`, vLLM's opt-in offloading connector - and
 *  for the same reason: a feature you turn on by typing a number into a
 *  budget field is a feature nobody finds. */
const kvOn = ref(false)
const cacheRam = ref(DEFAULT_CACHE_RAM_GB)
const cacheDisk = ref(0)
/** Blank = the runner's own default, shown as the placeholder. The BUDGET is
 *  the decision; the location only matters to someone overriding it. */
const cacheDiskPath = ref('')
const cacheDirDefault = computed(() => reg.estHost?.cache_dir ?? 'the data folder')
/** What this endpoint's cache would cost the MACHINE, against what the box
 *  has and what the rest of the fleet has already promised. The manager
 *  reports `committed` from every configured endpoint's file, including this
 *  one when it is already saved - so an edit subtracts its own current
 *  ceiling rather than counting it twice. */
const hostRam = computed(() => {
  const h = reg.estHost
  if (!h) return null
  const own = (savedOffloadRamGb.value || 0) * 1024 ** 3
  const others = Math.max(0, h.committed - own)
  const want = cacheRam.value * 1024 ** 3
  return { total: h.total, others, want, free: h.total === null ? null : h.total - others }
})
const hostRamOver = computed(() => {
  const h = hostRam.value
  return !!h && h.free !== null && h.want > h.free
})
const hostRamLine = computed(() => {
  const h = hostRam.value
  if (!h || !kvOn.value) return ''
  const gb = (n: number): string => `${(n / 1024 ** 3).toFixed(0)} GB`
  if (h.total === null) {
    return h.others > 0 ? `Other models have reserved ${gb(h.others)}.` : ''
  }
  const base = `${gb(h.want)} of this machine's ${gb(h.total)}`
  const rest = h.others > 0 ? `, with ${gb(h.others)} reserved by other models` : ''
  return hostRamOver.value
    ? `${base}${rest} - more than is free.`
    : `${base}${rest}.`
})
/** This endpoint's ALREADY-SAVED ceiling, so editing it does not read as a
 *  second commitment on top of itself. */
const savedOffloadRamGb = ref(0)

/**
 * What the budgets BUY, in the unit that actually changes.
 *
 * Offloading does not shrink a live conversation: every active sequence keeps
 * all of its KV in VRAM, and decode never reads KV over PCIe (a plan
 * non-goal - it is infeasible, not merely unimplemented). What the tier holds
 * is prefixes that have already been evicted, so coming back to one is a read
 * instead of a re-prefill. So the fit bars are right not to move, and the
 * honest readout is how much past conversation stays warm.
 */
const kvHoldsLine = computed(() => {
  const per = est.value?.estimate?.kv_bytes_per_token
  const ctx = est.value?.estimate?.max_ctx
  if (!kvOn.value || !per || per <= 0) return ''
  const bytes = (cacheRam.value + cacheDisk.value) * 1024 ** 3
  const tokens = Math.floor(bytes / per)
  if (tokens <= 0) return ''
  const head = `Holds ~${fmtCtx(tokens)} tokens of evicted conversation`
  if (!ctx) return `${head}.`
  const sessions = Math.floor(tokens / ctx)
  return sessions >= 2 ? `${head} - about ${sessions} at ${fmtCtx(ctx)} context.` : `${head}.`
})

// ── the endpoint's config FILE (edit mode) ──────────────────────────────────
// Loaded once on mount: the file's TEXT is the one document all three tabs
// edit - Simple (the common subset, saved via the takeover), Advanced (a form
// of every config key, serialized back to TOML), Configuration file (the raw
// text). Its hash is the optimistic-concurrency token every save path sends
// back - a file that moved on disk since this page opened (hand-edit, another
// session) is refused, never clobbered.
const mode = ref<'simple' | 'advanced' | 'file'>('simple')
const filePath = ref('')
const fileHash = ref('')
const advText = ref('')
const advError = ref('')
/** The same class of message, surfaced in the SIMPLE tab - which had nowhere to
 *  show one, so a failed render or a refused save had to go to a toast or
 *  nowhere. Rendering and saving both happen from Simple now. */
const simpleError = ref('')
/** Whether the file GET ever succeeded. A failed load is loud (error card +
 *  retry in the Advanced/file tabs) and blocks their Save - an editor that
 *  never loaded must not be able to overwrite the real file with emptiness. */
const fileLoaded = ref(false)
const fileError = ref('')
/** The file text exactly as it came off disk - `advText` is the EDITED copy.
 *  Save diffs the two to answer "does this need a restart", which is the only
 *  question that turns a save into an interruption. */
const fileAsLoaded = ref('')
async function loadFile(): Promise<void> {
  fileError.value = ''
  try {
    const res = await fetch(`/api/servers/${port.value}/file`)
    if (!res.ok) throw new Error(`the manager answered ${res.status}`)
    const f = (await res.json()) as { path: string; content: string; hash: string }
    // Drop a leading BYTE ORDER MARK. A config file is user-editable on a
    // platform whose editors add one freely - and PowerShell 5.1's
    // `Set-Content -Encoding utf8` writes one every time, which is how a
    // config file acquires one without anyone typing it. Rust's `toml` crate skips
    // it, so the manager read those files perfectly; smol-toml REFUSES it
    // ("only letter, numbers, dashes and underscores are allowed in keys",
    // pointing at line 1 column 1). The whole Simple tab then projected
    // nothing and the model picker sat on its placeholder.
    // Stripped here rather than at each parse site so every reader downstream
    // sees clean text, and so saving the file quietly removes it.
    const content = f.content.replace(/^﻿/, '')
    filePath.value = f.path
    fileHash.value = f.hash
    advText.value = content
    fileAsLoaded.value = content
    fileLoaded.value = true
  } catch (e) {
    fileLoaded.value = false
    fileError.value = `Could not load the config file - ${e instanceof Error ? e.message : String(e)}. Nothing can be saved from here until it loads.`
  }
}
async function retryFile(): Promise<void> {
  await loadFile()
  if (fileLoaded.value && mode.value === 'advanced') formFromToml(advText.value)
}
/** Candidate paths for the Browse pickers (suggestions only - any hand-typed
 *  path stays first-class). */
const fileLists = ref<Record<'gguf' | 'mmproj' | 'mtp' | 'fp8_dirs' | 'model_dirs' | 'kernel_packs', string[]>>({
  gguf: [],
  mmproj: [],
  mtp: [],
  fp8_dirs: [],
  model_dirs: [],
  kernel_packs: [],
})
const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p
/** Browse selection: multi fields (model_dirs) append, others replace. */
function pickFile(f: AfField, p: string): void {
  if (!f.multi) {
    afS[f.key] = p
    return
  }
  const items = afS[f.key].split(',').map((x) => x.trim()).filter(Boolean)
  if (!items.includes(p)) items.push(p)
  afS[f.key] = items.join(', ')
}
/** Toggle pills: the file's value always shows, even outside the fixed set
 *  (a hand-written host or a future device value must not vanish). */
function pillChoices(f: AfField): string[] {
  const cur = afS[f.key]
  const base = f.choices ?? []
  return cur && !base.includes(cur) ? [...base, cur] : base
}
function togglePill(f: AfField, c: string): void {
  afS[f.key] = afS[f.key] === c ? '' : c
}

// ── the Advanced tab: a form of every config key the runner has ─────────────
// One row per key in the runner's Config struct (config.rs, deny_unknown_fields
// - a valid file can hold nothing else).
//
// "Complete by construction" is what this comment used to claim, and it was not
// true: an audit found four keys missing. Completeness matters more
// than it looks, because this list is also the Advanced tab's serializer -
// `tomlFromForm` writes only these keys, so a key the list omits is DELETED from
// the file by a round trip through this tab. `vram_budget` was the live case:
// the Simple tab writes it and Advanced silently dropped it. There is no
// compile-time guard on either side of the language boundary, so the rule is:
// add the key here in the same change that adds it to Config.
//
// Values are strings ('' = key absent from the file); switches
// are booleans (off = absent - identical semantics, the defaults are false).
// The form is a lens over the file text: entering the tab parses it, leaving
// or saving serializes back.
// 'bool3' is pills that serialize as a real TOML boolean, for the one config
// field whose absence is a THIRD state rather than false (`metrics_auth`: unset
// = key required off-box only). A plain 'switch' cannot say "explicitly false",
// and plain 'pills' would write the string "true", which the runner's
// Option<bool> refuses at startup.
type AfKind = 'text' | 'num' | 'switch' | 'bool3' | 'pills' | 'list' | 'json' | 'file' | 'gpu'
interface AfField {
  key: string
  kind: AfKind
  /** one-line hint: what it does, with a sample where that helps */
  hint: string
  /** whole numbers only */
  int?: boolean
  /** toggle pills - clicking the selected one unsets it (= key absent);
   *  a file value outside this list still shows, as its own pill */
  choices?: string[]
  /** which /api/servers/files list feeds the Browse picker (kind 'file') */
  src?: 'gguf' | 'mmproj' | 'mtp' | 'fp8_dirs' | 'model_dirs' | 'kernel_packs'
  /** comma-separated list field - Browse APPENDS instead of replacing */
  multi?: boolean
  /** Key in the model's ELECTED sampling profile, so the field can show what
   * blank actually resolves to (the runner applies pin -> election
   *  -> wire, and only the first of those is visible in this file). */
  elect?: 'temperature' | 'top_k' | 'top_p' | 'min_p'
}
const AF_CARDS: { hd: string; fields: AfField[] }[] = [
  {
    hd: 'Model',
    fields: [
      { key: 'model', kind: 'file', src: 'gguf', hint: 'the GGUF weights file this endpoint serves' },
      // Provenance, not a setting: the runner ignores it, the manager uses it to
      // name a STOPPED endpoint and preselect it here. Editable because a file
      // the user can hand-edit should not have a field they cannot see.
      { key: 'catalog', kind: 'json', hint: 'which catalog model the weights are · {"model": "qwen3.5-9b", "artifact": "q8", "drafter": "drafter2"}' },
      { key: 'mmproj', kind: 'file', src: 'mmproj', hint: 'image encoder GGUF - enables image input' },
      { key: 'mtp', kind: 'file', src: 'mtp', hint: 'drafter GGUF for speculative decode (models without in-file MTP)' },
      { key: 'fp8_native', kind: 'file', src: 'fp8_dirs', hint: 'official FP8 safetensors folder - sources the fp8 planes directly' },
      { key: 'model_dirs', kind: 'file', src: 'model_dirs', multi: true, hint: 'where a bare model NAME is looked up - unused when model is a path' },
      { key: 'kernel_pack', kind: 'file', src: 'kernel_packs', hint: 'the GPU kernel pack the engine loads' },
      { key: 'device', kind: 'pills', choices: ['cuda'], hint: 'the serving backend - cuda on this build' },
      { key: 'gpu', kind: 'gpu', hint: 'which GPU serves this endpoint - saved as its UUID ("0" also works, but ordinals can swap after driver/PCIe changes)' },
      { key: 'served_model_name', kind: 'text', hint: 'id shown in /v1/models instead of the file-derived one' },
      { key: 'aliases', kind: 'list', hint: 'extra model ids answered · gpt-4o-mini' },
    ],
  },
  {
    hd: 'Serving',
    fields: [
      { key: 'host', kind: 'pills', choices: ['0.0.0.0', '127.0.0.1'], hint: '0.0.0.0 = every interface (network callers need the key) · 127.0.0.1 = this machine only' },
      { key: 'port', kind: 'num', int: true, hint: 'the endpoint identity - fixed here' },
      { key: 'max_ctx', kind: 'num', int: true, hint: 'context per conversation, tokens · 32768' },
      { key: 'max_batch', kind: 'num', int: true, hint: 'sequences batched at once · 32' },
      { key: 'concurrency_limit', kind: 'num', int: true, hint: 'queue depth before Overloaded · empty = uncapped' },
      { key: 'max_tokens', kind: 'num', int: true, hint: 'default reply cap when a request omits one' },
      { key: 'max_output_ceiling', kind: 'num', int: true, hint: 'hard clamp on the output of any request' },
      { key: 'seed', kind: 'num', int: true, hint: 'fixed RNG seed · empty = per-request' },
      { key: 'kv_cache_dtype', kind: 'pills', choices: ['auto', 'f16', 'fp8_e4m3'], hint: 'fp8_e4m3 halves KV bytes, slightly lossy · auto = per-model default' },
      { key: 'spec', kind: 'pills', choices: ['on', 'off', 'adaptive'], hint: 'speculative decode · adaptive measures and adapts if performance increases · a number pins the draft depth (A/B only)' },
      { key: 'no_spec', kind: 'switch', hint: 'older spelling of spec = off; set one or the other, not both' },
      // The Simple tab's "How much of the card" writes this key, and Advanced
      // had no row for it - so opening Advanced and saving deleted the cage.
      { key: 'vram_budget', kind: 'num', int: true, hint: 'hard VRAM cage for this endpoint, MiB · empty = size against free VRAM at load' },
      // `graph_scratch_mib` (config.rs, PR #14): the fixed 3 GiB graph/prefill
      // scratch reserve, sized for 16-48 GB cards; on an 8 GB card it eats the
      // whole KV grant and starves the [moe_offload] slot cache.
      { key: 'graph_scratch_mib', kind: 'num', int: true, hint: 'graph/prefill scratch reserve, MiB · empty = 3072 · lower it on 8 GB cards' },
    ],
  },
  // These hints named 1.0 / 0 / 1.0 as the fallback, which was true until the
  // elected sampling profiles landed and is now wrong for every
  // model whose authors published decoding parameters: blank resolves to the
  // CHECKPOINT's numbers (`SamplingDefaults::resolve` - pin, else election,
  // else the wire), not to the OpenAI wire. Saying "1.0" here told a user their
  // qwen3.6 server samples at top_k off when it actually samples at 20.
  //
  // repeat_penalty is the exception and keeps its number: no election publishes
  // one (the penalties are windowed by repeat_last_n here, so a published value
  // would be a different operator - see paddock_models::sampling), so that one
  // really is pin-or-off.
  {
    hd: 'Sampling defaults',
    fields: [
      // `elect` names the key in the elected profile, so the field can show
      // the real number as its placeholder instead of claiming "the model's
      // own default" and leaving the reader to trust it.
      { key: 'temp', kind: 'num', elect: 'temperature', hint: 'blank = the published default' },
      { key: 'top_k', kind: 'num', int: true, elect: 'top_k', hint: '0 = off' },
      { key: 'top_p', kind: 'num', elect: 'top_p', hint: '1.0 = off' },
      { key: 'min_p', kind: 'num', elect: 'min_p', hint: '0.0 = off' },
      { key: 'repeat_penalty', kind: 'num', hint: 'no model publishes one · blank = off' },
      { key: 'repeat_last_n', kind: 'num', int: true, hint: 'penalty window, tokens · 64' },
    ],
  },
  {
    hd: 'Access & limits',
    fields: [
      { key: 'api_key', kind: 'text', hint: 'Bearer key network callers must send (loopback never needs it)' },
      { key: 'no_auth', kind: 'switch', hint: 'serve the network with no key at all - warns on startup' },
      { key: 'trusted_proxy', kind: 'switch', hint: 'behind a reverse proxy: rate-limit on its X-Real-IP, API key required from loopback too' },
      { key: 'ratelimit_per_minute', kind: 'num', int: true, hint: 'per-client requests/minute' },
      { key: 'ratelimit_per_day', kind: 'num', int: true, hint: 'per-client requests/day' },
    ],
  },
  {
    hd: 'Requests',
    fields: [
      { key: 'strip_params', kind: 'list', hint: 'request fields removed server-side · temperature, top_p' },
      { key: 'force_params', kind: 'json', hint: 'fields forced on every request · {"temperature": 0.2}' },
      { key: 'variants', kind: 'json', hint: 'selectable as <model>:<key> · {"high": {"reasoning_effort": "high"}}' },
      // Speech only, and off by default upstream too (faster-whisper's
      // vad_filter, whisper.cpp's --vad) because it changes what a transcript
      // CONTAINS, not just how fast it arrives.
      { key: 'vad_gate', kind: 'switch', hint: 'speech models · skip silent windows before the encoder runs - faster, and changes what the transcript contains' },
    ],
  },
  {
    hd: 'Tools',
    fields: [
      { key: 'web_search_provider', kind: 'pills', choices: SEARCH_PROVIDERS.map((p) => p.id), hint: 'server-executed web_search tool' },
      { key: 'web_search_api_key', kind: 'text', hint: 'the provider key' },
      { key: 'mcp_servers', kind: 'json', hint: '[{"server_label": "github", "server_url": "https://.../mcp"}]' },
      { key: 'pdf_max_pages', kind: 'num', int: true, hint: 'pages rendered per PDF · 20' },
      { key: 'pdf_page_long_edge', kind: 'num', int: true, hint: 'target long edge, px · 1568' },
      // The first config key that is a TOML TABLE rather than a scalar, so it
      // rides the json kind the way mcp_servers does: the round trip is
      // table -> pretty JSON in the box -> table, which is what keeps the
      // serializer from dropping the block. All-off by default.
      { key: 'forensics', kind: 'json', hint: 'image/document forensic preprocessing · {"enabled": true, "auto": "images", "tool": true}' },
      // Same table-shaped round trip as `forensics` above. The Simple tab's
      // KV offloading card owns the two budgets; anything set by hand here
      // survives that, and the block is only ever written when ram_gb is
      // real (the disk tier stores through ram, so a disk budget alone arms
      // nothing).
      { key: 'kv_offload', kind: 'json', hint: 'prefix cache kept outside VRAM · {"enabled": true, "ram_gb": 24.0, "nvme_gb": 200.0, "nvme_path": "D:/paddock-cache"}' },
      // `[moe_offload]` (config.rs MoeOffload): the routed-expert planes of a
      // MoE model in page-locked RAM, with a VRAM slot cache of the hot
      // experts sized from what the KV plan leaves; vram_gb caps the cache.
      // Table-shaped like kv_offload. No Simple-tab card yet, so this row is
      // the only place the Studio can set it - and what keeps a round trip
      // through Advanced from deleting a block written by hand.
      { key: 'moe_offload', kind: 'json', hint: 'MoE experts in page-locked RAM, hot ones cached in VRAM · {"enabled": true, "vram_gb": 6.0}' },
    ],
  },
  {
    hd: 'Logging & events',
    fields: [
      { key: 'log_file', kind: 'text', hint: 'append logs here instead of stdout' },
      { key: 'no_events', kind: 'switch', hint: 'stop recording per-request events' },
      // Gated INDEPENDENTLY of the event ring, per config.rs - and only one of
      // the two had a control, so "turn off the recording" could not be said in
      // full from this page.
      { key: 'no_metrics', kind: 'switch', hint: 'stop serving the Prometheus /metrics endpoint' },
      { key: 'metrics_auth', kind: 'bool3', choices: ['true', 'false'], hint: 'who may scrape /metrics · unset = key needed off-box, loopback open so a local collector works · true = key always · false = open to anyone who can reach the port' },
      { key: 'session_headers', kind: 'list', hint: 'headers captured as session id · x-session-id' },
    ],
  },
]
const AF_ALL = AF_CARDS.flatMap((c) => c.fields)
const afS = reactive<Record<string, string>>({})
const afB = reactive<Record<string, boolean>>({})
for (const f of AF_ALL) {
  if (f.kind === 'switch') afB[f.key] = false
  else afS[f.key] = ''
}

/** Leading comment block of the file (the manager's header, any hand notes
 *  above the first key) - re-attached when the form serializes, so a form
 *  save doesn't strip it. */
function headerOf(text: string): string {
  const out: string[] = []
  for (const l of text.split(/\r?\n/)) {
    if (l.startsWith('#')) out.push(l)
    else if (l.trim() === '' && out.length) out.push(l)
    else break
  }
  while (out.length && out[out.length - 1].trim() === '') out.pop()
  return out.join('\n')
}

/** File text -> form. Returns an error message (and leaves the form alone)
 *  when the text isn't valid TOML. */
function formFromToml(text: string): string | null {
  let doc: Record<string, unknown>
  try {
    doc = tomlParse(text) as Record<string, unknown>
  } catch (e) {
    return `The file does not parse as TOML - fix it under Configuration file. ${e instanceof Error ? e.message : String(e)}`
  }
  for (const f of AF_ALL) {
    const v = doc[f.key]
    if (f.kind === 'switch') afB[f.key] = v === true
    else if (v === undefined) afS[f.key] = ''
    else if (f.kind === 'list' || f.multi)
      afS[f.key] = Array.isArray(v) ? v.map(String).join(', ') : String(v)
    else if (f.kind === 'json') afS[f.key] = JSON.stringify(v, null, 2)
    else afS[f.key] = String(v)
  }
  return null
}

/** Form -> file text. Empty fields stay absent; switches write only `true`
 *  (false = the default = absent). Table-valued keys are appended after the
 *  scalars so the serialized TOML can never trap a bare key inside a table. */
function tomlFromForm(): { text: string } | { error: string } {
  const scalars: Record<string, unknown> = {}
  const tables: Record<string, unknown> = {}
  for (const f of AF_ALL) {
    if (f.kind === 'switch') {
      if (afB[f.key]) scalars[f.key] = true
      continue
    }
    const s = (afS[f.key] ?? '').trim()
    if (!s) continue
    if (f.kind === 'bool3') {
      // a real boolean, not the string "true" - the runner's Option<bool>
      // refuses a quoted one at startup, and absent is the third state
      scalars[f.key] = s === 'true'
    } else if (f.kind === 'num') {
      const n = Number(s)
      if (!Number.isFinite(n)) return { error: `${f.key}: "${s}" is not a number` }
      if (f.int && !Number.isInteger(n)) return { error: `${f.key}: "${s}" is not a whole number` }
      scalars[f.key] = n
    } else if (f.kind === 'list' || f.multi) {
      const items = s.split(',').map((x) => x.trim()).filter(Boolean)
      if (items.length) scalars[f.key] = items
    } else if (f.kind === 'json') {
      let v: unknown
      try {
        v = JSON.parse(s)
      } catch {
        return { error: `${f.key}: not valid JSON` }
      }
      const wantArray = f.key === 'mcp_servers'
      if (wantArray ? !Array.isArray(v) : typeof v !== 'object' || v === null || Array.isArray(v))
        return { error: `${f.key}: expected a JSON ${wantArray ? 'array' : 'object'}` }
      tables[f.key] = v
    } else {
      scalars[f.key] = s
    }
  }
  let body: string
  try {
    body = tomlStringify({ ...scalars, ...tables })
  } catch (e) {
    return { error: `could not serialize: ${e instanceof Error ? e.message : String(e)}` }
  }
  const header = headerOf(advText.value)
  return { text: (header ? header + '\n\n' : '') + body.trimEnd() + '\n' }
}

/** Tab switch - the three tabs edit one document, and this is what makes that
 *  true rather than aspirational.
 *
 *  Every tab writes its state into `advText` on the way out and reads it on the
 *  way in. Only Advanced used to: Simple appeared nowhere here, so
 *  its edits were never in the text the other two tabs saved, and their edits
 *  were never in Simple's fields - whichever tab was open when you pressed Save
 *  silently won and the other's work was discarded.
 *
 *  Leaving Simple costs a round trip (the manager owns the serializer, see
 *  `renderCurrent`), so this is async and a failure BLOCKS the switch with the
 *  reason rather than moving you to a tab showing a stale document. */
const switching = ref(false)
async function setMode(m: 'simple' | 'advanced' | 'file'): Promise<void> {
  if (m === mode.value || switching.value) return
  if (!fileLoaded.value) {
    // never-loaded file: switch freely - the tabs show the load error + retry
    advError.value = ''
    mode.value = m
    return
  }
  if (mode.value === 'advanced') {
    const r = tomlFromForm()
    if ('error' in r) {
      advError.value = r.error
      return
    }
    advText.value = r.text
  } else if (mode.value === 'simple' && isEdit.value) {
    switching.value = true
    const r = await renderCurrent()
    switching.value = false
    if ('error' in r) {
      advError.value = r.error
      simpleError.value = r.error
      return
    }
    advText.value = r.text
  }
  if (m === 'advanced') {
    const err = formFromToml(advText.value)
    if (err) {
      advError.value = err
      return
    }
  } else if (m === 'simple' && isEdit.value && !(await simpleFromToml(advText.value))) {
    advError.value = 'This text does not parse as TOML - fix it here before switching to Simple.'
    return
  }
  advError.value = ''
  simpleError.value = ''
  mode.value = m
}

// ── SERVER TOOLS (per-model config) ─────────────────────────────────────────
// What this endpoint supplies server-side, hosted-API style: the web-search
// integration (callers just declare {type:"web_search"}) and named MCP
// servers (callers use the bare label). All of it lands in the model's own
// config file; callers' inline tools always work regardless.
const wsProvider = ref<string>('')
const wsKey = ref('')
// Pills, not a Select: mutually exclusive choices where "Off" must be
// first-class (and Reka Selects can't hold an empty value anyway). The
// providers come from one table shared with ServerDetail, so the list, the
// brand spelling and the key URL can never drift apart.
const wsChosen = computed(() => searchProvider(wsProvider.value))
interface HeaderRow {
  name: string
  value: string
}
/** One MCP server, either transport the runner supports: a remote HTTP
 *  server (url + arbitrary headers) or a local stdio command (npx/uvx class,
 *  args + env). `allowed` optionally narrows which of its tools the model
 *  may call. */
interface McpRow {
  label: string
  transport: 'http' | 'stdio'
  url: string
  headers: HeaderRow[]
  command: string
  args: string
  envRows: HeaderRow[]
  allowed: string
  approval: boolean
}
const mcpRows = ref<McpRow[]>([])

// ── LIBRARY connectors on this model ────────────────────────────────────────
// Connector-library entries materialized into this file carry connector_id;
// they are owned by the Connectors library (one definition, scoped per model)
// and never edited as hand rows here. Stashed on load, re-appended on save,
// so a takeover never drops them.
//
// The switches STAGE. They used to call the scope API on flip,
// which rewrote this endpoint's file behind the page - so one control on a
// form full of pending edits was already committed, and the only honest way to
// label it was a paragraph explaining that it did not work like its
// neighbours. Now membership is form state like everything else and the
// bottom Save writes it; `applyConnectorScope` below is that write.
const connectors = useConnectorsStore()
const connectorEntries = ref<object[]>([])
function splitMcp<T extends object>(entries: T[] | undefined): T[] {
  const list = entries ?? []
  connectorEntries.value = list.filter((e) => 'connector_id' in e)
  return list.filter((e) => !('connector_id' in e))
}
/** Staged membership: the connector ids wanted on this endpoint. Seeded from
 *  the library once it lands (it loads asynchronously alongside the fleet). */
const connWanted = ref<Set<string>>(new Set())
const connSeeded = ref(false)
watch(
  () => [connectors.list.length, editPort.value] as const,
  () => {
    const p = editPort.value
    if (connSeeded.value || p === null || !connectors.list.length) return
    connWanted.value = new Set(connectors.list.filter((c) => c.ports.includes(p)).map((c) => c.id))
    connSeeded.value = true
  },
  { immediate: true },
)
function connectorOnHere(c: Connector): boolean {
  return c.system || connWanted.value.has(c.id)
}
function toggleConnector(c: Connector): void {
  if (c.system) return
  const next = new Set(connWanted.value)
  if (!next.delete(c.id)) next.add(c.id)
  connWanted.value = next
}
/** Write the staged membership through the scope API, which owns both halves
 *  of the truth (the DB row the Connectors page reads, and the `mcp_servers`
 *  entries in every servers/*.toml). Runs before the main save: it moves this
 *  endpoint's file, so the drift guard needs the fresh hash afterwards. */
async function applyConnectorScope(): Promise<void> {
  const p = editPort.value
  if (p === null || !connSeeded.value) return
  const changed = connectors.list.filter(
    (c) => !c.system && c.ports.includes(p) !== connWanted.value.has(c.id),
  )
  if (!changed.length) return
  for (const c of changed) {
    const ports = connWanted.value.has(c.id)
      ? [...c.ports, p]
      : c.ports.filter((x) => x !== p)
    await connectors.setScope(c.id, false, ports)
  }
  await loadFile()
  try {
    const doc = tomlParse(advText.value) as { mcp_servers?: object[] }
    connectorEntries.value = (doc.mcp_servers ?? []).filter((e) => 'connector_id' in e)
  } catch {
    /* a parse problem surfaces on save */
  }
}

function addMcpRow(): void {
  mcpRows.value = [
    ...mcpRows.value,
    {
      label: '',
      transport: 'http',
      url: '',
      headers: [],
      command: '',
      args: '',
      envRows: [],
      allowed: '',
      approval: false,
    },
  ]
}
function dropMcpRow(i: number): void {
  mcpRows.value = mcpRows.value.filter((_, x) => x !== i)
}
function addKv(list: HeaderRow[]): void {
  list.push({ name: '', value: '' })
}
function dropKv(r: McpRow, key: 'headers' | 'envRows', i: number): void {
  r[key] = r[key].filter((_, x) => x !== i)
}
/** "Same tools as <server>": copy another endpoint's tool config here - a
 *  deliberate copy at creation, after which each model stays independent. */
// '__none' sentinel, not '': Reka SelectItem refuses an empty-string value
const copySel = ref<string | number>('__none')
const copyOptions = computed(() => [
  { value: '__none', label: 'Copy tools from...' },
  ...fleet.rows
    .filter((r) => r.config && r.port !== editPort.value)
    .map((r) => ({
      value: r.port,
      label: r.display ?? `${r.model ?? r.port}`,
      hint: String(r.port),
      vendor: r.vendor ?? undefined,
    })),
])
watch(copySel, (v) => {
  if (v === '__none' || v === '' || v === undefined) return
  const src = fleet.rows.find((r) => r.port === Number(v))?.config
  if (src) {
    wsProvider.value = src.web_search_provider ?? ''
    wsKey.value = src.web_search_api_key ?? ''
    // copy the HAND-authored tools only: library membership is scoped per
    // model through the Connectors library, not duplicated by copy
    mcpRows.value = (src.mcp_servers ?? [])
      .filter((e) => !('connector_id' in e))
      .map(mcpRowFrom)
  }
  copySel.value = '__none'
})
function mcpRowFrom(e: {
  server_label: string
  server_url?: string
  headers?: Record<string, string>
  command?: string
  args?: string[]
  env?: Record<string, string>
  allowed_tools?: string[]
  require_approval?: string
}): McpRow {
  return {
    label: e.server_label,
    transport: e.command ? 'stdio' : 'http',
    url: e.server_url ?? '',
    headers: Object.entries(e.headers ?? {}).map(([name, value]) => ({ name, value })),
    command: e.command ?? '',
    args: (e.args ?? []).join(' '),
    envRows: Object.entries(e.env ?? {}).map(([name, value]) => ({ name, value })),
    allowed: (e.allowed_tools ?? []).join(', '),
    approval: e.require_approval === 'always',
  }
}

const gpus = ref<{ index: number; name: string; mem_total: number | null; uuid: string | null }[]>([])
/** The as-deployed pin (a UUID or ordinal) waiting for the GPU list to load
 *  so the Simple picker can show the right card. */
const pendingPin = ref<string | null>(null)
function applyPendingPin(): void {
  if (pendingPin.value === null) return
  const hit = gpus.value.find(
    (g) => g.uuid === pendingPin.value || String(g.index) === pendingPin.value,
  )
  if (hit) {
    gpuIndex.value = hit.index
    pendingPin.value = null
  }
}
const installed = computed(() => reg.models.filter((m) => m.installed))

// option lists for the Reka Select wrappers
const modelOptions = computed(() => [
  ...installed.value.map((m) => ({
    value: m.id,
    label: m.display,
    vendor: m.vendor,
    title: m.id,
  })),
  { value: '__custom', label: 'Other (installed name or GGUF path)...' },
])
/** Every context this MODEL supports, with the ones the card cannot back
 *  greyed out and priced, led by the fit itself and closed by Custom.
 *
 *  This list used to be filtered down to what fits, which quietly conflated two
 *  very different facts: "this model is short-context" and "your GPU is the
 *  limit here". A 256K model on a busy card would offer 8K and say nothing, so
 *  the one number the user needed - that VRAM, not the model, was the ceiling -
 *  was the number we removed. Steps beyond the MODEL's own maximum stay out:
 *  those do not exist at any VRAM.
 *
 *  The fit rung is the other half of the same lesson: the ladder is powers of
 *  two and the card's ceiling almost never is, so a card that backs 224K used
 *  to top out at 128K here and lose the rest in silence. The ladder itself is
 *  pure (lib/ctx-ladder.ts) and tested there. */
const ctxSelectOptions = computed(() =>
  ctxLadder(
    {
      vramCap: ctxVramCap.value,
      modelCap: ctxModelCap.value,
      batch: batch.value,
      bytesPerToken: est.value?.estimate?.kv_bytes_per_token ?? 0,
    },
    fmtCtx,
    gb,
  ),
)
// -1 = driver default (Select values are string|number, not null)
const gpuOptions = computed(() => [
  { value: -1, label: 'Driver default' },
  ...gpus.value.map((g) => ({
    value: g.index,
    label: `${g.index} - ${g.name}`,
    hint: g.mem_total ? gb(g.mem_total) : undefined,
  })),
])
const gpuSel = computed({
  get: () => gpuIndex.value ?? -1,
  set: (v: string | number) => (gpuIndex.value = Number(v) < 0 ? null : Number(v)),
})
// The Advanced gpu field: a dropdown of the box's real GPUs, pinned by UUID
// (what the file stores; enumeration-order-proof). '__unset' sentinel because
// Reka Selects can't hold an empty value; a file value outside the list (an
// ordinal, a foreign UUID) shows as its own option rather than vanishing.
const afGpu = computed<string | number>({
  get: () => (afS.gpu === '' ? '__unset' : afS.gpu),
  set: (v) => (afS.gpu = v === '__unset' ? '' : String(v)),
})
const gpuAdvOptions = computed(() => {
  const opts: { value: string; label: string; hint?: string }[] = [
    { value: '__unset', label: 'GPU 0', hint: 'default' },
  ]
  for (const g of gpus.value)
    opts.push({
      value: g.uuid ?? String(g.index),
      label: `${g.index} - ${g.name}`,
      hint: g.mem_total ? gb(g.mem_total) : undefined,
    })
  const cur = afS.gpu
  if (cur && !opts.some((o) => o.value === cur)) opts.push({ value: cur, label: cur })
  return opts
})
// A KV cache cannot be turned off - drop it and every token re-reads the whole
// prompt - so this is a PRECISION choice, not a feature switch, and it is
// genuinely just two values. The old spelling offered "auto" as a third, which
// named a mechanism instead of an outcome: picking it told you nothing about
// what you would get. The form now preselects the model's real default (from
// the catalog's kv_default) and always writes the dtype it chose.
//
// 8-bit is GATED on the CARD. E4M3 is a tensor-core format from Ada onward;
// below that floor the hardware cannot do the conversion and the cache
// round-trip comes back wrong, not lossy - measured on this A6000 with
// Qwen3.8-27B. The runner already refuses and serves f16 instead,
// but a refusal at load is not a control: the setting has to be unreachable
// where it cannot work, the same way the NVFP4 quality card greys
// against its `min_cc` a few lines up. The floor itself is
// `paddock_models::gpu_support::fp8_kv` - one rule the runner, the estimator
// and this control all read.
//
// The hint names the HARDWARE, never a generation to go buy: the literal
// threshold is Ada, and the engine does not serve Ada, so "get an Ada card"
// would send someone shopping for silicon we refuse.
const fp8KvBlocked = computed<string | null>(() => {
  const cc = ready.info?.cc
  // unrecognised silicon makes no claim - the engine already warns on an
  // unvalidated arch, so a guess here would be noise
  if (!cc) return null
  return cc[0] > 8 || (cc[0] === 8 && cc[1] >= 9) ? null : 'this GPU has no FP8 tensor cores'
})
/** What a new server gets. 8-bit is the elected default on hardware that has
 * the tensor cores for it (the project's quantization table: f16 on Ampere,
 *  FP8 E4M3 KV on fp8 hardware) - it halves the dominant memory term, so
 *  defaulting to 16-bit there quietly spent twice the cache for nothing. The
 *  catalog's own `kv_default` still wins where a model sets one, because that
 *  is per-model evidence and this is only a hardware read. */
const preferredKv = computed(() => (fp8KvBlocked.value ? 'f16' : 'fp8_e4m3'))

/** How much of the card this endpoint may hold.
 *
 *  The underlying field is `vram_budget`, an absolute MiB ceiling the manager
 *  computes at spawn so the REST of the card stays startable for the next
 *  model. That number is unrelatable on its own: it is neither what the model
 *  needs nor what the card has, but what the scheduler lent this endpoint given
 *  other endpoints the user may have forgotten they configured. And a
 *  PERCENTAGE would be worse - 40% of 48 GB is 19 GB, 40% of a 24 GB card is
 *  9.6 GB, which may not hold the weights at all, so a share silently becomes
 *  unservable when the hardware changes.
 *
 *  So the control is the decision the user actually owns - share the card or
 *  take it - and the byte count is shown as the CONSEQUENCE, never typed as an
 *  input. `limit` stays for people who know exactly what they want.
 */
const vramMode = ref<'share' | 'all' | 'limit'>('share')
const vramLimitGb = ref(8)
const VRAM_MODES = [
  { value: 'share', name: 'Share it', sub: 'leaves room for other models' },
  { value: 'all', name: 'Take all of it', sub: 'every spare byte becomes cache' },
  { value: 'limit', name: 'Limit to', sub: 'a ceiling you set' },
] as const
/** Free VRAM in MiB - what "take all of it" resolves to. Concrete deliberately:
 *  the config file gets a real number, not a sentinel the runner would have to
 *  interpret, so the Advanced editor and the file both stay readable. */
const freeMib = computed(() => Math.floor((reg.estDevice?.free ?? 0) / (1024 * 1024)))
/** The ceiling this form will write, or null for "let the manager grant one"
 *  (which is the pre-existing behaviour and stays the default). */
const vramBudgetMib = computed<number | null>(() => {
  if (vramMode.value === 'all') return freeMib.value || null
  if (vramMode.value === 'limit') return Math.max(1, Math.round(vramLimitGb.value * 1024))
  return null
})
const kvOptions = computed(() => [
  { value: 'f16', label: '16-bit', hint: 'exact' },
  {
    value: 'fp8_e4m3',
    label: '8-bit',
    hint: fp8KvBlocked.value ?? 'half the memory, slightly lossy',
    disabled: !!fp8KvBlocked.value,
  },
])
// An endpoint configured elsewhere (or on other hardware) can arrive asking
// for a width this card cannot serve. The runner would downgrade it silently
// from the user's point of view, so correct the form instead: what is shown
// is then what will run, and the estimate below prices the same thing.
watch([fp8KvBlocked, kvDtype], () => {
  if (fp8KvBlocked.value && kvDtype.value === 'fp8_e4m3') kvDtype.value = 'f16'
})
// Speculative decode. Deliberately not a switch: "on" is a policy, not a
// state - under Adaptive the engine re-picks the draft length every round and
// stops speculating on its own when the load says it is not paying. A plain
// on/off toggle would promise something the engine does not do.
//
// Three CHOICES, three STORED VALUES. Two earlier passes got this wrong: the
// first labelled the empty value "Default" (default to what? - it says nothing
// about whether spec is on), and the relabel kept '' as the stored value, which
// Reka/Radix rejects outright - SelectItem reserves the empty string, so the
// item silently never rendered and the list showed only two of three options
// while the trigger displayed the third. Every choice now writes its own key.
// On and Off need no gloss; only Adaptive does something a reader can't guess.
// No em dashes in UI copy (house rule).
/** The drafter picker's options: "Automatic" first (follow the catalog
 *  default), then every catalogued drafter, flagged when not downloaded so a
 *  pin cannot silently select bytes that are not there. */
const drafterOptions = computed(() => [
  { value: '', label: 'Automatic', hint: 'the newest the catalog ships' },
  ...drafterChoices.value.map((d) => ({
    value: d.id,
    label: d.label,
    hint: d.installed ? undefined : 'not downloaded',
  })),
])

const specOptions = [
  { value: 'on', label: 'On' },
  { value: 'off', label: 'Off' },
  { value: 'adaptive', label: 'Adaptive', hint: 'measures and adapts if performance increases' },
]

/** Four badges, not three plus a dropdown. The dropdown held the same axis as
 *  the pills beside it - one control in two shapes, and picking 8 from it lit
 *  no pill, which reads as nothing being selected. The fourth badge owns the
 *  off-ladder case explicitly and reveals a number field for it. */
const WORKLOADS = [
  { id: 'solo', label: 'Just me', batch: 1 },
  { id: 'agents', label: 'Coding agents', batch: 4 },
  { id: 'team', label: 'A team / an app', batch: 16 },
] as const
const CUSTOM_WORKLOAD = 'custom'

// ToggleGroup speaks strings; `batch` is a number. `batchCustom` is its own
// ref rather than derived from "does a preset own this number", because
// choosing Custom while the count happens to be 4 must keep Custom selected -
// deriving it would silently jump the selection back to Coding agents.
const batchCustom = ref(false)
const workload = computed({
  get: () =>
    !batchCustom.value && WORKLOADS.some((w) => w.batch === batch.value)
      ? String(batch.value)
      : CUSTOM_WORKLOAD,
  set: (v: string) => {
    if (!v) return
    if (v === CUSTOM_WORKLOAD) {
      batchCustom.value = true
      return
    }
    batchCustom.value = false
    batch.value = Number(v)
  },
})

/** What a context window is in units a person owns. Tokens are the wire's unit
 *  and nobody's mental model: ~0.75 English words per token, ~500 words to a
 *  typed A4 page.
 *
 *  Rendered as two figures rather than a sentence, and ROUNDED HARD - "49,152
 *  words" carries five digits of precision onto an estimate whose own constant
 *  is a rule of thumb, which reads as a measurement. "49k" cannot. */
function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n < 10_000_000 ? 1 : 0)}M`
  if (n >= 10_000) return `${Math.round(n / 1000)}k`
  if (n >= 1_000) return `${(n / 1000).toFixed(1)}k`
  return String(Math.max(1, Math.round(n)))
}
const ctxScale = computed(() => {
  const words = ctx.value * 0.75
  const out = [
    { value: compact(words), unit: 'words' },
    // "pages", not "A4 pages": A4 and US Letter hold within ~3% of the same
    // text at equal margins (62,370 vs 60,323 mm2), which is far inside the
    // error of a 0.75-words-per-token rule of thumb. Naming either one is
    // false specificity AND wrong for half the world; naming neither is
    // accurate everywhere and one word shorter.
    { value: compact(words / 500), unit: 'pages' },
  ]
  // Room left on the card, for whoever never opens the dropdown: the one
  // figure that says a bigger window is a choice, not a limit.
  if (ctxCap.value && ctx.value < ctxCap.value)
    out.push({ value: fmtCtx(ctxCap.value), unit: 'possible on this card' })
  return out
})

const chosenModel = computed(() =>
  model.value === '__custom' ? customModel.value.trim() : model.value,
)
const catModel = computed(() => reg.models.find((m) => m.id === model.value))
const weightChoices = computed(
  () => catModel.value?.artifacts.filter((a) => a.kind === 'weights') ?? [],
)
const fp8Artifact = computed(() =>
  catModel.value?.artifacts.find((a) => a.kind === 'fp8-snapshot'),
)
/** The quality choice, as one list.
 *
 *  A native-plane artifact (`fp8-snapshot`) is a WEIGHT CLASS to the person
 *  choosing - "the small build" - even though the engine composes it over the
 *  GGUF base rather than replacing it. It used to render as a capability
 *  switch called "Native FP8 planes", next to Vision, wearing copy written for
 *  the self-sourced official-FP8 lane: "source weights from the official FP8
 *  snapshot - measured slightly worse than the default". For qwen3.8-27b every
 *  word of that was wrong - the artifact is NVFP4, it is hosted in our own R2,
 * and it is the 48 GB-card fit lane, not a downgrade.
 *
 *  So it joins the Quality cards. Picking it means "the default base plus
 *  these planes", which is exactly what the download set and the spec already
 *  expressed; only the presentation was lying. */
interface QualityCard {
  key: string
  artifact: NonNullable<typeof fp8Artifact.value>
  planes: boolean
}
const PLANES_KEY = 'planes:'
const qualityCards = computed<QualityCard[]>(() => {
  const cards: QualityCard[] = weightChoices.value.map((a) => ({
    key: a.id,
    artifact: a,
    planes: false,
  }))
  const fp8 = fp8Artifact.value
  if (fp8) cards.push({ key: PLANES_KEY + fp8.id, artifact: fp8, planes: true })
  return cards
})
/** The base a plane artifact rides on: the catalog default, else the first. */
const planesBase = computed(
  () => weightChoices.value.find((a) => a.default) ?? weightChoices.value[0],
)
/** Why this card cannot be picked here, or null. Today only the arch floor -
 *  a format whose kernels this GPU does not have. The rule itself lives in
 *  lib/arch-floor so the PICKER screens whole models by the same answer this
 *  badges individual cards with; a second copy here is how the two drift. */
function archBlock(a: { min_cc?: [number, number] }): string | null {
  return archBlockReason(a, ready.info?.cc)
}
const qualityKey = computed<string>({
  get: () => (fp8Native.value && fp8Artifact.value ? PLANES_KEY + fp8Artifact.value.id : artifactId.value),
  set: (k) => {
    if (k.startsWith(PLANES_KEY)) {
      fp8Native.value = true
      // the planes need their base on disk too, so the weights pick moves with
      // them rather than being left wherever it was
      if (planesBase.value) artifactId.value = planesBase.value.id
    } else {
      fp8Native.value = false
      artifactId.value = k
    }
  },
})
const visionArtifact = computed(() =>
  catModel.value?.artifacts.find((a) => a.kind === 'vision'),
)
// A required tower gets no switch - turning granite-vision's image reader
// off would leave a text model with the purpose gone (catalog `required`).
const visionRequired = computed(() => visionArtifact.value?.required ?? false)
// Forensics is VLM-coupled: its findings are injected for the vision tower to
// examine (confirm/contradict pixels, read a receipt's sum/VAT). So the toggle
// is only usable when this endpoint actually serves vision - a built-in tower,
// or an optional one left switched on.
const visionServed = computed(
  () => visionRequired.value || (!!visionArtifact.value && withVision.value),
)
// Whether this endpoint could ever run forensics at all - a vision-capable
// catalog model (or a hand-typed one, where we can't tell, so we allow it and
// let the runner be the judge).
//
// The catalog-loaded guard matters: `catModel` is undefined both for a
// hand-typed path and for "the registry has not arrived yet", and treating the
// second as the first flashed this card onto whisper, ASR and plain text
// endpoints for the whole of a cold load. Every one of those has a `[catalog]`
// entry that says plainly it has no vision artifact - we just had not read it
// yet. An empty `models` is never a real catalog, only an unloaded one.
const catalogLoaded = computed(() => reg.models.length > 0)
const forensicsPossible = computed(
  () => catalogLoaded.value && (!catModel.value || !!visionArtifact.value),
)
// System tools need TOOL CALLING; a model whose capability doesn't include
// it gets no web-search/MCP section (same rule as canSpeculate).
const canTools = computed(
  () => !catModel.value || catModel.value.capability.includes('tools'),
)
// Only offer speculation where this engine actually implements it (in-file MTP
// or a drafter we load). Showing the control everywhere would make it a setting
// that silently does nothing on the models that can't - and the manager refuses
// such a spawn anyway, so the control would be offering a guaranteed error.
// A hand-typed GGUF path has no catalog entry to ask; leave it visible there and
// let the runner be the judge.
const canSpeculate = computed(
  () => !catModel.value || catModel.value.capability.includes('speculative'),
)
/** Every drafter this model catalogues. muse ships two (DFlash1 and DFlash2)
 *  and they are ARTIFACTS of one model, not two models: speculation is
 *  lossless, so both emit identical output and differ only in speed. */
const drafterChoices = computed(
  () => catModel.value?.artifacts.filter((a) => a.kind === 'drafter') ?? [],
)
/** The drafter this endpoint would wire. The manager's election (registry.rs)
 *  only ever returns INSTALLED artifacts - pin, else installed default, else
 *  installed sibling - and the first three arms here mirror it. The two
 *  trailing uninstalled arms are deliberately wider: they exist so the summary
 *  can name a sole drafter that is not downloaded yet ("Uses X - not
 *  downloaded"), which is choose-time guidance the spawn-time election has no
 *  use for. */
const drafterArtifact = computed(() => {
  const ds = drafterChoices.value
  return (
    (drafterId.value && ds.find((a) => a.id === drafterId.value && a.installed)) ||
    ds.find((a) => a.default && a.installed) ||
    ds.find((a) => a.installed) ||
    (drafterId.value && ds.find((a) => a.id === drafterId.value)) ||
    ds.find((a) => a.default) ||
    ds[0]
  )
})
/** What "Speculation: On" actually wired, for display. In-file MTP models
 *  catalogue no drafter - that is how they are told apart, and saying so beats
 *  leaving "on" to mean two different mechanisms silently. */
const drafterSummary = computed(() => {
  if (!canSpeculate.value) return ''
  if (!drafterChoices.value.length) return 'in-file MTP (no companion drafter)'
  const d = drafterArtifact.value
  if (!d) return ''
  return d.installed ? d.label : `${d.label} - not downloaded`
})
/** The choice a new endpoint opens on.
 *
 *  "On" wherever speculation is free to switch on - in-file MTP (nothing extra
 *  to load) or a drafter the catalog ships by default. "Off" where the drafter
 *  is an opt-in companion: a non-default drafter is a deliberate extra download
 *  AND extra resident VRAM for the life of the endpoint, so it is not ours to
 *  enable on someone's behalf. They can still pick On or Adaptive. */
function defaultSpecChoice(): string {
  const d = drafterArtifact.value
  return d && !d.default ? 'off' : 'on'
}
// keep the weights choice valid as the model changes: prefer installed, then
// the catalog default
watch(model, () => {
  const ws = weightChoices.value
  if (!ws.some((a) => a.id === artifactId.value)) {
    artifactId.value = (ws.find((a) => a.installed) ?? ws.find((a) => a.default) ?? ws[0])?.id ?? ''
  }
  // These are per-MODEL facts, so re-derive them rather than carrying the
  // previous model's answer across - but not in edit mode, where the model is
  // fixed (the picker links to server-new to change it) and this watcher only
  // ever fires as hydration assigns the endpoint's own model. Clearing
  // fp8Native unguarded there discarded the plane choice it had just read off
  // the file, the reset landing after it because the watcher flushes async.
  if (!isEdit.value) {
    fp8Native.value = false
    specPolicy.value = defaultSpecChoice()
    kvDtype.value = catModel.value?.kv_default ?? preferredKv.value
  }
})
const est = computed(() => reg.estimates[model.value]?.artifacts?.[artifactId.value])

/** A blocked card's badge already states the requirement in full ("Needs a
 *  Blackwell GPU"), so a label that carries it too - "Compact (Blackwell)" -
 *  says it twice and the pair overflows a 220px card. The parenthetical is
 *  there to separate two same-named builds on a box where both are pickable,
 *  which is precisely the case where no badge renders; once the badge is up it
 *  has nothing left to say. Trailing parenthetical only, so a label that reads
 *  "(preview) Compact" or has inner parens is left alone. */
function qualityTitle(a: { label?: string; min_cc?: [number, number] }): string {
  const label = a.label ?? ''
  return archBlock(a) ? label.replace(/\s*\([^()]*\)\s*$/, '') : label
}
/** Plain-language meaning of a weights choice, keyed off the quant class -
 *  the tag itself ("Q8_0") stays a footnote for the people who know it. */
/** One line, because these sit side by side and are read by COMPARISON: three
 *  two-line paragraphs are read as prose, three short lines are read as a
 *  choice. Everything cut here was a qualification ("that most work never
 *  notices") that says the same thing on two of the three cards, so it
 *  distinguishes nothing - which is the only job a blurb on a card has. */
function qualityBlurb(a: { quant?: string }): string {
  const q = (a.quant ?? '').toUpperCase()
  if (q.startsWith('Q8')) return 'Practically identical to the original.'
  // NVFP4 before the Q4 test: it is four-bit too, but read straight from the
  // published checkpoint rather than converted from a bigger file.
  if (q.includes('NVFP4')) return 'Four-bit, straight from the official checkpoint.'
  if (q.includes('Q4')) return 'Half the memory, slight quality cost.'
  if (q.includes('MXFP4')) return 'The format it was trained in - nothing is higher.'
  return 'A smaller build - some quality for memory.'
}
/** Per-artifact fit verdict, so a card can warn before it is even picked. */
function artifactVerdict(id: string): string | null {
  const f = reg.estimates[model.value]?.artifacts?.[id]?.estimate?.fit
  return f?.verdict ?? null
}

/** What the CARD can back at this concurrency (VRAM), 0 = not estimated yet. */
const ctxVramCap = computed(() => {
  const e = est.value
  if (!e?.estimate) return 0
  return e.curve?.find((p) => p.at === batch.value)?.ctx ?? e.estimate.max_ctx
})
/** What the MODEL itself supports, whatever the card has. */
const ctxModelCap = computed(() => est.value?.estimate?.model_max_ctx ?? 0)
/** The cap that actually applies - the lower of the two, on a KV page. */
const ctxCap = computed(() =>
  ctxCapOf({ vramCap: ctxVramCap.value, modelCap: ctxModelCap.value }),
)
/** Steps that FIT: what auto-selection and clamping may choose from. */
const ctxOptions = computed(() =>
  ctxFits({ vramCap: ctxVramCap.value, modelCap: ctxModelCap.value }),
)

// How the context was chosen. The select speaks three kinds of value - a rung,
// "everything that fits" and "custom" - and only the rung is a number the
// form can bind directly. `ctxMode` is its own ref, like `batchCustom`: a
// custom 131072 must stay Custom, and the fit rung must stay the fit rung
// when the cap moves under it, neither of which a value comparison can tell.
type CtxMode = 'ladder' | 'max' | 'custom'
const ctxMode = ref<CtxMode>('ladder')
const ctxPick = computed<string | number>({
  get: () => (ctxMode.value === 'max' ? CTX_MAX : ctxMode.value === 'custom' ? CTX_CUSTOM : ctx.value),
  set: (v) => {
    userTouchedCtx.value = true
    if (v === CTX_MAX) {
      ctxMode.value = 'max'
      if (ctxCap.value) ctx.value = ctxCap.value
      return
    }
    if (v === CTX_CUSTOM) {
      // the field opens on the current value - a starting point, not a reset
      ctxMode.value = 'custom'
      return
    }
    ctxMode.value = 'ladder'
    ctx.value = Number(v)
  },
})
/** The custom field's ceiling: the model's own maximum, or generous when the
 *  model has not said. The runner's own will-it-fit check still guards the
 *  load; this only stops a typo from asking for a gigatoken. */
const ctxCustomMax = computed(() => ctxModelCap.value || 2_097_152)
/** Under the custom field: what the card backs at this concurrency, and what
 *  going past it means. A warning, not a clamp - the estimate is a model of
 *  the card, and someone who measured 220K on their own box outranks it. */
const ctxCustomHint = computed(() => {
  const cap = ctxCap.value
  if (!cap) return { warn: false, text: 'Any size the model supports; the will-it-fit check guards the load.' }
  if (ctx.value > cap)
    return {
      warn: true,
      text: `${fmtCtx(cap)} is the most this card backs at ${batch.value} at once - past it conversations are evicted and refilled, or the load is refused. Lower concurrency or KV precision to raise it.`,
    }
  return { warn: false, text: `Up to ${fmtCtx(cap)} fits at ${batch.value} at once.` }
})

// Propose a context when the MODEL or the concurrency changes - the two
// choices that genuinely reset what a sensible window is. Prefer the 32k an
// agentic workload wants. `userTouchedCtx` stops the re-proposal from stomping
// an explicit choice (or the edit prefill).
const userTouchedCtx = ref(false)
function proposeCtx(): void {
  const opts = ctxOptions.value
  const pref = opts.filter((c) => c <= 32768)
  ctx.value = (pref.length ? pref[pref.length - 1] : opts[opts.length - 1]) ?? 32768
}
watch([model, batch], () => {
  if (userTouchedCtx.value) return
  proposeCtx()
})
// A change in what FITS only ever clamps down; it never grows the choice.
//
// This used to re-propose, and that made the KV precision control look broken:
// switching to 8-bit halves the bytes per token, the proposal immediately
// doubled the context to spend the savings, and the memory bar came out
// byte-identical - one knob, no visible effect, exactly the report.
// Growing is now the user's move (the ladder greys out what does not fit, so
// they can see the room appear); shrinking stays automatic, because a context
// the card cannot back is not a choice we may leave standing.
//
// The fit rung is the one exception, by definition: "everything that fits" is
// a request to follow the cap wherever it goes, up included. Custom is the
// other way round - a number someone typed is theirs, and the hint under the
// field says when the card disagrees.
watch(ctxCap, (cap) => {
  if (!cap) return
  if (ctxMode.value === 'max') {
    ctx.value = cap
    return
  }
  if (ctxMode.value === 'custom') return
  if (ctx.value > cap) {
    const fits = ctxOptions.value.filter((c) => c <= cap)
    ctx.value = fits.length ? fits[fits.length - 1] : cap
  }
})
// Re-price on every knob that moves VRAM, not just concurrency: the fit line,
// the context ladder's greyed-out steps and the KV figure all derive from this
// estimate, so leaving kv/spec out made them silently contradict the form.
// Vision rides along because the supervisor honours it - a vision-off start
// really does drop the mmproj - so leaving it out over-charged every such
// estimate by the whole tower (0.9 GB on qwen3.8-27b). `cc` rides
// along so the server can price the KV width this CARD will serve rather than
// the one the control asked for.
watch([batch, kvDtype, specPolicy, withVision, vramBudgetMib, gpuIndex], () =>
  void reg.estimate({
    batch: batch.value,
    kv: kvDtype.value,
    spec: canSpeculate.value && specPolicy.value !== 'off',
    vision: withVision.value || visionRequired.value,
    cc: ready.info?.cc,
    budget: vramBudgetMib.value,
    gpu: gpuIndex.value ?? undefined,
    offloadRamGb: kvOn.value ? cacheRam.value : undefined,
  }),
)

// ── the Simple tab's projection of the config file ──────────────────────────
//
// The Simple tab is not a view of the file, it is a SPEC builder - catalog model
// id, artifact id, vision on/off, GPU index, budget in GB - and the manager
// turns those into TOML by resolving ids to paths and an NVML index to a device
// UUID. So the two directions are not symmetric: coming back from text is this
// pair of functions, and going to text is a round trip through the manager's own
// renderer (`renderCurrent` below). One projection each way, used by both the
// initial load and every tab switch, so the three tabs cannot drift.

/** The overlapping shape of a live fleet row's `config` and a parsed config
 *  file. Every field optional: absent means "leave the form's default". */
interface SimpleCfg {
  model?: string
  artifact?: string | null
  max_ctx?: number | null
  max_batch?: number | null
  gpu?: string | number | null
  kv_cache_dtype?: string | null
  spec?: string | null
  /** Drafter-artifact pin; absent = follow the catalog default. */
  drafter?: string | null
  api_key?: string | null
  web_search_provider?: string | null
  web_search_api_key?: string | null
  /** Taken off `mcpRowFrom` so the entry shape can only be described once. */
  mcp_servers?: Parameters<typeof mcpRowFrom>[0][]
}

function applySimpleConfig(cfg: SimpleCfg): void {
  // `cfg.model` is a catalog ID, already reconciled - by the manager, which is
  // now the only place the rule exists (`/api/servers/project` ->
  // `Registry::identity_for`). This used to be a browser copy of that rule, and
  // the copy is what produced the day's bugs: it matched every artifact kind on
  // filename alone, so a directory-shaped checkpoint
  // (`.../Qwen3-ForcedAligner-0.6B-hf/model.safetensors`) collided with the
  // first fp8 snapshot in catalog order and 11546 read "Qwen 3.8 27B" while the
  // manager said `qwen3-forced-aligner-0.6b`. Making the copy agree fixed that
  // instance and left the next drift free to happen.
  //
  // Anything the catalog doesn't know is a hand-typed path, which is the honest
  // "Other" entry - that is a lookup, not a rule, and there is nothing here to
  // keep in sync.
  const hit = cfg.model ? reg.models.find((m) => m.id === cfg.model) : undefined
  if (hit) {
    model.value = hit.id
  } else if (cfg.model) {
    model.value = '__custom'
    customModel.value = cfg.model
  }
  if (cfg.artifact) artifactId.value = cfg.artifact
  if (cfg.max_batch) batch.value = cfg.max_batch
  if (cfg.max_ctx) {
    ctx.value = cfg.max_ctx
    userTouchedCtx.value = true
    // a file can carry any number; the ladder only owns its rungs
    ctxMode.value = (CTX_STEPS as readonly number[]).includes(cfg.max_ctx) ? 'ladder' : 'custom'
  }
  // the file pins by UUID; the Simple picker speaks NVML indexes - stash
  // the pin and resolve once (or if) the GPU list has landed
  pendingPin.value = cfg.gpu !== null && cfg.gpu !== undefined ? String(cfg.gpu) : null
  applyPendingPin()
  // a file with no key (or the legacy "auto") predates this control; resolve
  // it to the value that endpoint actually serves at
  const kv = cfg.kv_cache_dtype as string | undefined
  kvDtype.value = !kv || kv === 'auto' ? (catModel.value?.kv_default ?? 'f16') : kv
  // an existing file with no `spec` key predates this control; the engine
  // treats that as the tuned ladder, which is "On"
  specPolicy.value = cfg.spec ?? 'on'
  drafterId.value = cfg.drafter ?? ''
  // the key is shown, not masked - the operator's own box
  apiKey.value = cfg.api_key ?? ''
  wsProvider.value = cfg.web_search_provider ?? ''
  wsKey.value = cfg.web_search_api_key ?? ''
  mcpRows.value = splitMcp(cfg.mcp_servers).map(mcpRowFrom)
}

/** Config-file text -> the Simple form. False when the text does not parse (the
 *  file tab surfaces that; the form keeps whatever it had).
 *
 *  Two of these used to be missing and each was a silent loss on save: `spec`
 *  was never read for a stopped endpoint, so a `spec = "off"` file came back
 *  showing On and saving turned speculation on; `gpu` was deliberately dropped
 *  ("the Simple picker just starts unpinned"), so editing a pinned endpoint
 *  unpinned it. A projection that quietly omits a field is the same bug as a
 *  tab that quietly omits a document. */
async function simpleFromToml(text: string): Promise<boolean> {
  // The MANAGER PROJECTS this, not the browser. `/api/servers/project` parses
  // the buffer and hands back the settings already resolved - `model` is a
  // catalog id, reconciled by the same `Registry::identity_for` that runs on a
  // start. The Simple tab therefore holds no rule of its own, which is the
  // point: this file used to carry a COPY of that rule, the copy disagreed with
  // the original twice in one day, and each time the fix was to make the copy
  // agree again rather than to stop having one.
  const r = await projectConfig(text)
  if ('error' in r) {
    // Not silent. Returning false leaves every field on its default - including
    // the model picker, which then shows no selection at all - and for a BOM'd
    // file that is exactly what happened on seven endpoints with nothing
    // anywhere saying why. Advanced and the file tab already surface
    // `fileError`; the Simple tab had no surface for it, so a parse failure
    // could only be read as an empty form.
    fileError.value =
      `The configuration file does not parse, so these fields could not be filled in - ` +
      `open the Configuration file tab to fix it. ${r.error}`
    return false
  }
  const p = r.projection
  // `fp8_native` and `mmproj` are PATHS in the file and switches on the form:
  // present means on. The live fleet row carries neither, which is why this
  // reads the config even for a running endpoint.
  fp8Native.value = p.fp8_native
  withVision.value = p.vision
  // Forensics: the toggle owns `enabled`; keep the file's auto/tool/
  // device so a save round-trips a hand-tuned scope instead of resetting it.
  kvOn.value = p.kv_offload?.enabled ?? false
  cacheRam.value = kvOn.value ? (p.kv_offload?.ram_gb || DEFAULT_CACHE_RAM_GB) : DEFAULT_CACHE_RAM_GB
  savedOffloadRamGb.value = kvOn.value ? cacheRam.value : 0
  cacheDisk.value = kvOn.value ? (p.kv_offload?.nvme_gb ?? 0) : 0
  cacheDiskPath.value = p.kv_offload?.nvme_path ?? ''
  forensicsOn.value = p.forensics?.enabled ?? false
  forensicsExtra.value = p.forensics
    ? {
        auto: p.forensics.auto ?? null,
        tool: p.forensics.tool ?? null,
        device: p.forensics.device ?? null,
      }
    : {}
  // "How much of the card". The file records a CEILING, not which of the three
  // choices produced it - and "all of it" is only a convenience that computes
  // one from free VRAM, a number that moves. So a budget in the file reads back
  // as an explicit limit at exactly that size: faithful in effect, which is
  // what the endpoint will actually serve under.
  vramMode.value = p.vram_budget === null ? 'share' : 'limit'
  if (p.vram_budget !== null) vramLimitGb.value = Math.max(1, Math.round(p.vram_budget / 1024))
  applySimpleConfig({
    model: p.model,
    artifact: p.artifact,
    max_ctx: p.max_ctx,
    max_batch: p.max_batch,
    gpu: p.gpu,
    kv_cache_dtype: p.kv_cache_dtype,
    spec: p.spec,
    drafter: p.drafter,
    api_key: p.api_key,
    web_search_provider: p.web_search_provider,
    web_search_api_key: p.web_search_api_key,
    mcp_servers: p.mcp_servers as SimpleCfg['mcp_servers'],
  })
  return true
}

/** The Simple form as config-file TEXT, rendered by the MANAGER'S serializer
 *  and laid over the file as it stands.
 *
 *  Not a local serializer, deliberately: the manager resolves model ids to
 *  paths, artifacts to files, an NVML index to a device UUID and inherits the
 *  API key - a Studio-side renderer would be a second implementation of
 *  `render_server_config` and would drift from the file Save actually writes.
 *  `merge_with` keeps every key the Simple tab has no control for. */
async function renderCurrent(): Promise<{ text: string } | { error: string }> {
  try {
    const res = await fetch('/api/servers/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...buildSpec(),
        for_edit: true,
        merge_with: advText.value,
        // The STAGED membership, not what is on disk - otherwise a connector you
        // just switched on would be missing from the text until you saved, which
        // is the same "the tab is showing a different document" problem one
        // level down. The manager materializes these with its own function.
        connectors: connectors.list
          .filter((c) => c.system || connWanted.value.has(c.id))
          .map((c) => c.id),
      }),
    })
    const body = (await res.json().catch(() => null)) as {
      toml?: string
      error?: { message?: string }
    } | null
    if (!res.ok || typeof body?.toml !== 'string') {
      return {
        error:
          body?.error?.message ??
          `The manager could not render this configuration (HTTP ${res.status}).`,
      }
    }
    return { text: body.toml }
  } catch (e) {
    return { error: `Could not reach the manager: ${e instanceof Error ? e.message : String(e)}` }
  }
}

let release: (() => void) | null = null
onMounted(async () => {
  release = fleet.hold()
  void connectors.ensure()
  if (!reg.models.length) await reg.refresh()
  if (!reg.envelope) await reg.estimate()
  const gpuReady = gpuApi
    .get()
    .then((s) => {
      gpus.value = (s.gpus ?? []).map((g, i) => ({
        index: (g as { index?: number }).index ?? i,
        name: g.name ?? `GPU ${i}`,
        mem_total: g.mem_total ?? null,
        uuid: (g as { uuid?: string }).uuid ?? null,
      }))
      applyPendingPin()
    })
    .catch(() => (gpus.value = []))

  if (isEdit.value) {
    // prefill from the as-deployed config once the fleet answers
    if (!fleet.rows.length) await fleet.refresh()
    const row = editRow.value
    port.value = editPort.value ?? 0
    // the file as it stands right now - the Advanced/file tabs edit this
    // text, and its hash guards every save path against clobbering. A failed
    // load is loud (error + retry), never a silently empty editor.
    await loadFile()
    // Editing an endpoint that serves native planes has to come back with that
    // choice SELECTED, or the Quality card shows the base build and saving
    // quietly demotes the endpoint to it. The FILE is the authority here (the
    // manager writes `fp8_native` as a path into servers/<port>.toml) and the
    // live fleet row does not carry it - so read the text, which loadFile has
    // just fetched for a running and a stopped endpoint alike.
    // The FILE is the authority (the rule being: the per-endpoint TOML is
    // the whole config), and it is also what the other two tabs edit - so the
    // Simple form projects from the text, and the tab switch re-runs the same
    // projection so a hand edit reaches these fields too. The live fleet row is
    // the fallback for a file that would not load, which is display-only
    // anyway: Save refuses in that state.
    const projected = fileLoaded.value && (await simpleFromToml(advText.value))
    if (!projected && row?.config) applySimpleConfig(row.config)
    if (!projected && !row?.config && row) {
      model.value = installed.value.some((m) => m.id === row.model) ? (row.model ?? '') : '__custom'
      if (model.value === '__custom') customModel.value = row.model ?? ''
    }
    pinned.value = row?.pinned ?? false
    persist.value = fleet.bootPorts.has(port.value)
    // The GPU pin resolves from a device UUID to a picker index only once the
    // device list lands, and Save renders from the picker - so wait for it
    // rather than let the first save propose an unpinned endpoint.
    await gpuReady
    // Browse-picker candidates (best-effort; hand-typed paths always work)
    void fetch('/api/servers/files')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((f) => (fileLists.value = f as typeof fileLists.value))
      .catch(() => {})
  } else {
    // The model was decided on the pick page (route param); a catalog id
    // resolves even when not installed - the quality pills say so honestly.
    const pre = paramModel.value ?? String(route.query.model ?? '')
    if (pre === 'custom') {
      model.value = '__custom'
    } else if (pre && reg.models.some((m) => m.id === pre)) {
      model.value = pre
    } else if (pre) {
      model.value = '__custom'
      customModel.value = pre
    } else {
      model.value = installed.value[0]?.id ?? '__custom'
    }
    const preArt = String(route.query.artifact ?? '')
    if (preArt) artifactId.value = preArt
    port.value = fleet.nextPort()
  }
})
onUnmounted(() => release?.())

// ── the proposal ─────────────────────────────────────────────────────────────
const proposal = computed(() => {
  const e = est.value
  const free = reg.estDevice?.free ?? 0
  if (model.value === '__custom')
    return {
      tone: 'dim',
      text: 'No measurement for a hand-typed model - the will-it-fit check the runner runs still guards the load.',
    }
  if (!e?.estimate)
    return {
      tone: 'dim',
      text: e?.reason ?? 'Not measured yet - the will-it-fit check guards the load.',
    }
  const x = e.estimate
  if (x.fit.verdict === 'does_not_fit')
    return {
      tone: 'bad',
      text: `Needs ${gb(x.resident)} to load - your GPU can give ${gb(free)}. It will be refused at load.`,
    }
  // What paddock will actually hold - the resident floor plus the whole KV
  // pool, because the pool is allocated at start and unavailable to anything
  // else from that moment. This line used to quote only `resident` ("21 GB to
  // load of 44 GB available") while the chart beside it committed 8 GB more,
  // so the panel carried two different answers to what it costs.
  const held = x.resident + (x.kv_pool ?? 0)
  // How much of the pool this context × concurrency actually wants. Past 100%
  // the pool cannot hold every session at once and the engine pages: sessions
  // are evicted and refilled, which costs time to first token rather than
  // failing. The chart clamped this away, so the one setting that most needed
  // a warning produced no visible change at all.
  const want = (x.kv_bytes_per_token ?? 0) * ctx.value * batch.value
  const pool = x.kv_pool ?? 0
  if (pool > 0 && want > pool)
    return {
      tone: 'ok',
      text: `${fmtCtx(ctx.value)} context × ${batch.value} concurrent needs ${gb(want)} of conversation memory but the pool is ${gb(pool)} - about ${Math.max(1, Math.floor(pool / (want / batch.value)))} stay resident and the rest are evicted and refilled. Starts fine; first token gets slower under load.`,
    }
  const squeeze = x.fit.verdict === 'tight' ? ' (tight - little room for conversations)' : ''
  return {
    tone: x.fit.verdict === 'tight' ? 'ok' : 'good',
    text: `${fmtCtx(ctx.value)} context × ${batch.value} concurrent - paddock holds ${gb(held)} of the ${gb(free)} available${squeeze}.`,
  }
})
// everything the fit chart needs, in one nullable bundle so the template's
// v-if narrows both the estimate and the device at once
const fitData = computed(() => {
  const e = est.value?.estimate
  const d = reg.estDevice
  return e && d && d.total ? { e, d } : null
})

// The forensics line on the fit chart. It shares the model's GPU context by
// default (0 resident VRAM here - "Shared between models"); only a hand-set
// cross-GPU device pin moves its footprint off this card. Null when forensics
// is off or the model can't serve it.
const fitForensics = computed(() => {
  if (!forensicsOn.value || !visionServed.value) return null
  const modelDev = gpuIndex.value ?? 0
  const dev = forensicsExtra.value.device
  const shared = dev == null || dev === modelDev
  return { shared, device: dev ?? null }
})

// A taken port is flagged the moment it's typed, not at submit; the manager's
// real bind failure stays the honest last backstop for non-paddock squatters.
const portClash = computed(() => !isEdit.value && fleet.takenPorts.has(port.value))

// What this deploy still has to fetch: the chosen weights + default
// companions + an opted-in FP8 snapshot, whichever aren't on disk yet. When
// non-empty the button says "Download & deploy · N GB" - consent by label,
// never a silent download.
const missingPieces = computed<{ ids: string[]; bytes: number }>(() => {
  const m = catModel.value
  if (model.value === '__custom' || !m) return { ids: [], bytes: 0 }
  const need = m.artifacts.filter(
    (a) =>
      !a.installed &&
      (a.id === artifactId.value ||
        (a.kind !== 'weights' &&
          a.kind !== 'fp8-snapshot' &&
          a.default &&
          (a.kind !== 'vision' || withVision.value || a.required)) ||
        (a.kind === 'fp8-snapshot' && fp8Native.value)),
  )
  return { ids: need.map((a) => a.id), bytes: need.reduce((s, a) => s + a.total_size, 0) }
})

/** Random local API key. Unambiguous alphabet (no 0/O/1/l/I). */
function genKey(): void {
  const abc = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
  const bytes = new Uint8Array(24)
  crypto.getRandomValues(bytes)
  apiKey.value = 'pd-' + Array.from(bytes, (b) => abc[b % abc.length]).join('')
}

/** The spec the Simple form currently means - what Save deploys. */
function buildSpec(): DeploySpec {
  const spec: DeploySpec = {
    model: chosenModel.value,
    port: port.value,
    max_ctx: ctx.value,
    max_batch: batch.value,
    pinned: pinned.value,
    persist: persist.value,
  }
  if (model.value !== '__custom' && artifactId.value) spec.artifact = artifactId.value
  // Only send a PIN. Absent means "follow the catalog default", so an endpoint
  // built today keeps tracking the default if a better drafter ships later.
  if (model.value !== '__custom' && drafterId.value) spec.drafter = drafterId.value
  if (fp8Native.value) spec.fp8_native = true
  if (!withVision.value && !visionRequired.value) spec.vision = false
  if (apiKey.value.trim()) spec.api_key = apiKey.value.trim()
  if (gpuIndex.value !== null) spec.gpu = gpuIndex.value
  // Absent = the manager computes a grant, which is what it has always done.
  // Present = an explicit ceiling, and admission then honours it verbatim
  // rather than computing a new one.
  if (vramBudgetMib.value !== null) spec.vram_budget = vramBudgetMib.value
  // server tools -> the model's config file
  if (wsProvider.value && wsKey.value.trim()) {
    spec.web_search_provider = wsProvider.value
    spec.web_search_api_key = wsKey.value.trim()
  }
  const kvObj = (rows: HeaderRow[]) =>
    Object.fromEntries(
      rows.filter((h) => h.name.trim() && h.value.trim()).map((h) => [h.name.trim(), h.value.trim()]),
    )
  const mcp = mcpRows.value
    .filter((r) =>
      r.label.trim() && (r.transport === 'http' ? r.url.trim() : r.command.trim()),
    )
    .map((r) => {
      const allowed = r.allowed.split(',').map((s) => s.trim()).filter(Boolean)
      const base = {
        server_label: r.label.trim(),
        ...(allowed.length ? { allowed_tools: allowed } : {}),
        require_approval: (r.approval ? 'always' : 'never') as 'always' | 'never',
      }
      if (r.transport === 'stdio') {
        const env = kvObj(r.envRows)
        return {
          ...base,
          command: r.command.trim(),
          args: r.args.split(/\s+/).map((s) => s.trim()).filter(Boolean),
          ...(Object.keys(env).length ? { env } : {}),
        }
      }
      const headers = kvObj(r.headers)
      return {
        ...base,
        server_url: r.url.trim(),
        ...(Object.keys(headers).length ? { headers } : {}),
      }
    })
  // library-owned entries ride along unchanged - a takeover save must never
  // drop what the scope API materialized
  const allMcp = [...(connectorEntries.value as typeof mcp), ...mcp]
  if (allMcp.length) spec.mcp_servers = allMcp
  // Always written, like `spec`: the file states the precision it serves at, so
  // reading it answers the question without knowing our per-family defaults.
  spec.kv_cache_dtype = kvDtype.value
  // Three choices, three stored values - the file always states which one, so
  // reading it answers "is speculation on here" without knowing our defaults.
  // Only models that can speculate render the control; for the rest the key
  // would be a claim the engine can't honor.
  if (canSpeculate.value) spec.spec = specPolicy.value
  // Forensics ([forensics]). Written only when enabled; a hand-set
  // scope round-tripped through forensicsExtra rides along so the toggle never
  // clobbers auto/tool/device. Disabled = omit the block (the owned-key overlay
  // then removes it from the file).
  if (forensicsOn.value) {
    const f: NonNullable<DeploySpec['forensics']> = { enabled: true }
    if (forensicsExtra.value.auto != null) f.auto = forensicsExtra.value.auto
    if (forensicsExtra.value.tool != null) f.tool = forensicsExtra.value.tool
    if (forensicsExtra.value.device != null) f.device = forensicsExtra.value.device
    spec.forensics = f
  }
  // KV offloading ([kv_offload]). RAM is the entry point - the disk tier
  // stores through it - so the switch always writes a RAM budget. The folder
  // is written only when overridden; the runner defaults it otherwise.
  if (kvOn.value) {
    const kv: NonNullable<DeploySpec['kv_offload']> = {
      enabled: true,
      ram_gb: cacheRam.value || DEFAULT_CACHE_RAM_GB,
    }
    if (cacheDisk.value > 0) {
      kv.nvme_gb = cacheDisk.value
      if (cacheDiskPath.value.trim()) kv.nvme_path = cacheDiskPath.value.trim()
    }
    spec.kv_offload = kv
  }
  // the concurrency token: Save refuses if the file moved since this page
  // loaded it
  if (isEdit.value && fileHash.value) spec.expect_config_hash = fileHash.value
  return spec
}

// ── Save ─────────────────────────────────────────────────────────────────────
//
// Writing a configuration and interrupting service are two different acts, and
// only the second one needs the user's consent. Before this
// the page had two save models - everything waited for a button labelled
// "Save & restart", except the connector switches, which committed on flip -
// and no way to express "save this, apply it later".
//
// Three config keys are live: the runner re-reads `mcp_servers`,
// `web_search_provider` and `web_search_api_key` whenever the file's mtime
// moves (paddock-runner routes.rs, LiveConfig), so a change to those applies on
// the next request with nothing stopping. Everything else binds at load. That
// split is the manager's rule (supervisor.rs LIVE_KEYS) and this mirrors it, so
// the page can say what a save will cost before asking.
const LIVE_TOML_KEYS = ['mcp_servers', 'web_search_provider', 'web_search_api_key']

/** The manager's `only_live_keys_changed`, in the browser - One rule for all
 *  three tabs, because all three now produce the same document.
 *
 *  The first cut of this had a second rule beside it: a fingerprint of the
 *  Simple form's load-bound fields, because Simple had no TOML to diff. Two
 *  rules for one question is one too many, and the fingerprint could only ever
 *  approximate what the manager would decide. With the render round trip the
 *  Simple tab has real text and this diff is exact. */
function onlyLiveKeysChanged(before: string, after: string): boolean {
  try {
    const a = tomlParse(before) as Record<string, unknown>
    const b = tomlParse(after) as Record<string, unknown>
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (LIVE_TOML_KEYS.includes(k)) continue
      if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) return false
    }
    return true
  } catch {
    return false // unparseable either side = assume it binds the engine
  }
}

/** The document the active tab currently means, as text. Simple pays a round
 *  trip; the other two already hold it. */
async function currentText(): Promise<{ text: string } | { error: string }> {
  if (mode.value === 'advanced') return tomlFromForm()
  if (mode.value === 'file') return { text: advText.value }
  return renderCurrent()
}

const isRunning = computed(() => editRow.value?.status === 'running')

// NOTE: a `liveOnlyToml()` used to live here - it rebuilt the file text with
// only the live keys swapped, because the Simple tab had no TOML of its own and
// a tools-only edit still had to reach a running model without a restart. The
// render round trip makes it redundant: `advText` already holds the merged
// document, so the live-only save just writes that.

/** The confirmation, open only when saving would interrupt a running model. */
const confirmOpen = ref(false)
/** Whether the document `submit` just assembled binds at load. Decided once,
 *  from the text, so the modal and the commit cannot disagree about it. */
const pendingRestart = ref(false)

/** Save for the Advanced form AND the Configuration file tab: the file text,
 *  written verbatim. The form tab serializes into that text first. A cheap
 *  client-side hash pre-check keeps the user's changes on this page when the
 *  file already moved (navigating first would lose them); the server-side
 *  guard remains the authority for the tiny remaining race. */
async function submitFile(defer = false): Promise<void> {
  if (busy.value) return
  if (!fileLoaded.value) {
    advError.value = 'The config file never loaded - nothing was saved.'
    return
  }
  advError.value = ''
  if (mode.value === 'advanced') {
    const r = tomlFromForm()
    if ('error' in r) {
      advError.value = r.error
      return
    }
    advText.value = r.text
  }
  try {
    const res = await fetch(`/api/servers/${port.value}/file`)
    if (res.ok) {
      const f = (await res.json()) as { hash: string }
      if (fileHash.value && f.hash !== fileHash.value) {
        advError.value =
          'This file changed on disk since you opened it. Reload the page and re-apply your changes - nothing was overwritten.'
        return
      }
    }
  } catch {
    /* unreachable manager - the PUT below will surface it */
  }
  busy.value = true
  void fleet.applyFile(
    port.value,
    advText.value,
    fileHash.value || undefined,
    chosenModel.value,
    defer,
  )
  busy.value = false
  void router.push({ name: 'servers' })
}

/** Click Save. Asks first only when there is something to interrupt: a running
 *  model plus a change that binds at load. Everything else - a stopped
 *  endpoint, a tools-only edit, a fresh start - just does what it says. */
async function submit(): Promise<void> {
  if (!isEdit.value) return start()
  if (busy.value || switching.value) return
  // A download is a restart by definition and its own confirmation is the
  // button label ("Download & restart · 20 GB"), so skip the round trip: there
  // is nothing the rendered text could add to that decision.
  if (mode.value === 'simple' && missingPieces.value.ids.length) {
    pendingRestart.value = true
    return save(true)
  }
  // Bring the one document up to date with whatever tab you are on, then ask
  // the single question. Doing it in this order is what lets the same rule
  // serve all three tabs.
  switching.value = true
  const t = await currentText()
  switching.value = false
  if ('error' in t) {
    advError.value = t.error
    simpleError.value = t.error
    return
  }
  advText.value = t.text
  pendingRestart.value = !onlyLiveKeysChanged(fileAsLoaded.value, t.text)
  if (pendingRestart.value && isRunning.value) {
    confirmOpen.value = true
    return
  }
  // `false` = nobody consented to a restart, because none was needed. The
  // modal is the only source of `save(true)` - passing the decision through
  // here would have restarted (and so STARTED) a stopped endpoint silently.
  await save(false)
}

/** Commit. `restart` true = drain and relaunch on this port; false = write the
 *  configuration and leave the process alone (a running model keeps serving
 *  what it loaded; a stopped one stays stopped). */
async function save(restart: boolean): Promise<void> {
  if (busy.value) return
  confirmOpen.value = false
  // Every edit path below writes this endpoint's config file, and two of them
  // derive it from the text that was loaded. If that load failed, the page is
  // showing defaults rather than this endpoint - saving would overwrite a real
  // configuration with them.
  if (!fileLoaded.value) {
    advError.value = 'The config file never loaded - nothing was saved.'
    toasts.push({
      tone: 'bad',
      title: 'Nothing was saved',
      description: `The configuration for port ${port.value} never loaded, so there is nothing to save over it. Reload the page.`,
      duration: 10000,
    })
    return
  }
  // Defer whenever no restart was consented to, and always on a stopped
  // endpoint: the applying path ends in start_config, so "save" on something
  // that is not running would START it - an edit is not a start.
  const defer = !restart && (pendingRestart.value || !isRunning.value)
  // The raw-TOML tabs save their own text and are their own connector editor
  // (the `mcp_servers` array is right there) - reloading the file underneath
  // them would discard the edit being saved.
  if (mode.value !== 'simple') return submitFile(defer)
  busy.value = true
  // Connector membership first: the scope API owns both the library row and
  // the `mcp_servers` entries, and writing them moves this file - so the main
  // save below needs the hash it leaves behind.
  try {
    await applyConnectorScope()
  } catch {
    /* the save below reports whatever state the file is in */
  }
  const spec = buildSpec()
  const miss = missingPieces.value
  if (miss.ids.length) {
    // downloading is starting: there is no deferred form of "fetch 20 GB"
    void fleet.deployWithPull(spec, model.value, miss.ids, 'redeploy')
  } else if (restart) {
    void fleet.redeploy(port.value, spec)
  } else if (defer) {
    // load-bound settings, or a stopped endpoint: write the file, start nothing
    void fleet.saveOnly(port.value, spec, model.value)
  } else {
    // Tools / web search only - the running model re-reads those per request,
    // so this writes the document rather than restarting. Re-render after the
    // scope write: `applyConnectorScope` reloads the file from disk, which
    // discards the copy `submit` assembled, and the hand MCP rows and
    // web-search key on this form are not on disk yet.
    const t = await renderCurrent()
    if ('error' in t) {
      simpleError.value = t.error
      busy.value = false
      return
    }
    advText.value = t.text
    void fleet.applyFile(port.value, t.text, fileHash.value || undefined, chosenModel.value)
  }
  busy.value = false
  void router.push({ name: 'servers' })
}

/** The start page's button: a spawn, which is a start by definition. */
function start(): void {
  if (!chosenModel.value || busy.value) return
  busy.value = true
  const spec = buildSpec()
  // Fire and NAVIGATE - feedback lives on the servers list as a live row
  // (downloading, then starting), never behind this page.
  const miss = missingPieces.value
  if (miss.ids.length) void fleet.deployWithPull(spec, model.value, miss.ids, 'deploy')
  else void fleet.deploy(spec)
  busy.value = false
  void router.push({ name: 'servers' })
}
</script>

<template>
  <!-- A <form>, not a <div>: this page has password fields (the provider API
       keys) and Chrome warns when one has no form ancestor - password managers
       and autofill key off form containment. @submit.prevent because saving is
       driven by the explicit buttons at the bottom, NOT by Enter: on a page
       with this many inputs, Enter submitting is a way to deploy a
       half-configured endpoint by accident. -->
  <form class="sf" novalidate @submit.prevent>
    <nav class="sf__crumbs">
      <RouterLink :to="{ name: 'servers' }">Models</RouterLink>
      <span>/</span>
      <template v-if="isEdit">
        <span>{{ editTitle }}</span>
        <span>/</span>
        <span>edit</span>
      </template>
      <template v-else>
        <span>{{ catModel?.display ?? 'Your own model' }}</span>
      </template>
    </nav>
    <h1 class="sf__title">
      {{ isEdit ? `Edit ${editTitle}` : catModel ? `Start ${catModel.display}` : 'Start a model' }}
    </h1>
    <p v-if="!isEdit" class="sf__lead">
      One model on one port. The proposal below is computed from your GPU - start as-is, or
      adjust the trade-off.
    </p>

    <!-- Simple = the common settings; Advanced = a form of every config key;
         Configuration file = the raw TOML. One document, three lenses. -->
    <!-- Disabled during the switch: leaving Simple asks the manager to render
         its settings as TOML, and a second click mid-flight would hand the next
         tab a half-assembled document. -->
    <div v-if="isEdit" class="sf__pills sf__mode">
      <button
        type="button"
        class="sf__pill"
        :class="{ 'sf__pill--on': mode === 'simple' }"
        :disabled="switching"
        @click="setMode('simple')"
      >
        Simple
      </button>
      <button
        type="button"
        class="sf__pill"
        :class="{ 'sf__pill--on': mode === 'advanced' }"
        :disabled="switching"
        @click="setMode('advanced')"
      >
        Advanced
      </button>
      <button
        type="button"
        class="sf__pill"
        :class="{ 'sf__pill--on': mode === 'file' }"
        :disabled="switching"
        @click="setMode('file')"
      >
        Configuration file
      </button>
    </div>

    <div class="sf__grid">
      <!-- each tab is cards on one scroll toward the one Save; the tabs are
           lenses over the SAME document, so nothing is hidden-but-pending -->
      <div class="sf__col">
        <template v-if="mode === 'simple'">
        <div v-if="fileError" class="sf__card">
          <p class="sf__hint sf__hint--warn">{{ fileError }}</p>
          <div><button type="button" class="pk-btn pk-btn--sm" @click="retryFile">Retry</button></div>
        </div>
        <div class="sf__card">
        <p class="sf__card-hd">Model &amp; workload</p>
        <!-- model: picked on /servers/new for a deploy (fixed here, one link
             back); still a Select on edit - a takeover can swap the model -->
        <template v-if="gpus.length > 1">
          <label class="sf__lbl">Graphics card</label>
          <Select v-model="gpuSel" :options="gpuOptions" block />
        </template>

        <label class="sf__lbl">Model</label>
        <template v-if="isEdit">
          <Select v-model="model" :options="modelOptions" block />
        </template>
        <div v-else class="sf__fixed">
          <Tooltip :label="catModel?.id">
            <span class="sf__fixed-name">
              <VendorLogo v-if="catModel?.vendor" :vendor="catModel.vendor" :size="18" />
              {{ catModel?.display ?? 'Your own model file' }}
            </span>
          </Tooltip>
          <RouterLink class="sf__fixed-change" :to="{ name: 'server-new' }">Change</RouterLink>
        </div>
        <TextInput
          v-if="model === '__custom'"
          v-model="customModel"
          placeholder="Model name or full path to a .gguf"
          block
        />

        <!-- the quality choice: one weights artifact per server (schema 3), plus
             a native-plane build when the catalog ships one; a not-installed
             pick just downloads on start. Cards that say what the choice MEANS
             - the quant tag is a footnote, not the headline (most people
             picking a model have never met "Q8_0") -->
        <template v-if="qualityCards.length > 1">
          <label class="sf__lbl">Quality</label>
          <RadioGroup
            v-model="qualityKey"
            class="sf__qcards"
            label="Quality"
            :style="{ '--qc': Math.min(3, qualityCards.length) }"
          >
            <RadioItem
              v-for="c in qualityCards"
              :key="c.key"
              :value="c.key"
              class="sf__qcard"
              :class="{ 'sf__qcard--blocked': !!archBlock(c.artifact) }"
              :disabled="!!archBlock(c.artifact)"
            >
              <span class="sf__qcard-title">{{ qualityTitle(c.artifact) }}</span>
              <span class="sf__qcard-meta">
                <b class="sf__qcard-size">{{ fmtBytes(c.artifact.total_size) }}</b>
                <span class="sf__qcard-quant">{{ c.artifact.quant }}</span>
              </span>
              <span class="sf__qcard-blurb">{{ qualityBlurb(c.artifact) }}</span>
              <!-- A planes artifact composes over the GGUF base rather than
                   replacing it, so picking it also pins the base - a wire fact
                   worth stating where the card CAN be picked, and pure noise on
                   one that is blocked, where the only thing that matters is
                   why. -->
              <span v-if="c.planes && planesBase && !archBlock(c.artifact)" class="sf__qcard-note">
                also uses {{ planesBase.label }}
              </span>
              <span
                v-if="!c.artifact.installed && !archBlock(c.artifact)"
                class="sf__qcard-note"
              >
                <Icon name="arrow-down" :size="10" /> downloads when you start it
              </span>
              <span
                v-if="!c.planes && artifactVerdict(c.artifact.id) === 'does_not_fit'"
                class="sf__qcard-warn"
              >
                Won't fit on this GPU
              </span>
              <!-- The requirement over the card, not inside its title line: a
                   badge beside a name has to share a 220px column with it and
                   wrapped, and it competed with the name for the eye when the
                   card is not a choice at all. Dimming the content and stating
                   the requirement across it says "not available, here is why"
                   in one read. -->
              <span v-if="archBlock(c.artifact)" class="sf__qcard-veil">
                <span class="sf__qcard-veil-txt">{{ archBlock(c.artifact) }}</span>
              </span>
            </RadioItem>
          </RadioGroup>
        </template>

        <!-- capabilities: what rides along with this composition. The weight
             class is NOT one of these - it is the Quality choice above. -->
        <template v-if="visionArtifact">
          <label class="sf__lbl">Capabilities</label>
          <p v-if="visionRequired" class="sf__capline">
            <Icon name="image" :size="14" />
            Vision - built in
            <span v-if="!visionArtifact.installed" class="sf__capnote">
              downloads on start ({{ fmtBytes(visionArtifact.total_size) }})
            </span>
          </p>
          <label v-else class="sf__check">
            <Switch v-model="withVision" label="Vision - image input" />
            Vision - image input
            <span v-if="withVision && !visionArtifact.installed" class="sf__capnote">
              downloads on start ({{ fmtBytes(visionArtifact.total_size) }})
            </span>
          </label>
        </template>

        <!-- the one real decision -->
        <label class="sf__lbl">Workload</label>
        <ToggleGroup v-model="workload" class="sf__wl" label="Workload">
          <ToggleGroupItem
            v-for="w in WORKLOADS"
            :key="w.id"
            :value="String(w.batch)"
            class="sf__wlcard"
          >
            <span class="sf__wlcard-name">{{ w.label }}</span>
            <span class="sf__wlcard-sub">{{ w.batch }} at once</span>
          </ToggleGroupItem>
          <ToggleGroupItem :value="CUSTOM_WORKLOAD" as="div" class="sf__wlcard">
            <span class="sf__wlcard-name">Custom</span>
            <span v-if="!batchCustom" class="sf__wlcard-sub">pick a number</span>
            <span v-else class="sf__step" @click.stop>
              <button type="button" class="sf__step-btn" @click="batch = Math.max(1, batch - 1)">
                <Icon name="minus" :size="12" />
              </button>
              <span class="sf__step-val">{{ batch }}</span>
              <button type="button" class="sf__step-btn" @click="batch = batch + 1">
                <Icon name="plus" :size="12" />
              </button>
            </span>
          </ToggleGroupItem>
        </ToggleGroup>

        <label class="sf__lbl">Context per conversation</label>
        <Select v-model="ctxPick" :options="ctxSelectOptions" />
        <template v-if="ctxMode === 'custom'">
          <NumberField v-model="ctx" :min="1024" :max="ctxCustomMax" :step="1024" />
          <p class="sf__hint" :class="{ 'sf__hint--warn': ctxCustomHint.warn }">
            {{ ctxCustomHint.text }}
          </p>
        </template>
        <p class="sf__scale">
          <span v-for="f in ctxScale" :key="f.unit" class="sf__stat">
            <b class="sf__stat-v">≈{{ f.value }}</b>
            <span class="sf__stat-u">{{ f.unit }}</span>
          </span>
        </p>

        <FieldLabel label="Conversation memory">
          <p>
            Every model keeps a cache of the conversation so far - without one, each new word
            would re-read the whole thing.
          </p>
          <p>
            So this is not whether to remember, but how precisely: 8-bit halves the memory each
            remembered word costs, buying longer conversations or more at once.
          </p>
        </FieldLabel>
        <Select v-model="kvDtype" :options="kvOptions" />

        <template v-if="canSpeculate">
          <FieldLabel label="Speculative">
            <p>
              A small fast model guesses the next few words, and the real model checks them all
              in one go - keeping only what it agrees with.
            </p>
            <p>
              The answer is identical either way, it just arrives sooner. Costs a little memory,
              and pays off least when the server is busy.
            </p>
          </FieldLabel>
          <Select v-model="specPolicy" :options="specOptions" />

          <!-- Which drafter, once "On" can mean more than one thing. Only
               shown when the model catalogues a choice; otherwise the summary
               line below still names what On wired, so the mechanism is never
               silent. -->
          <template v-if="specPolicy !== 'off' && drafterChoices.length > 1">
            <FieldLabel label="Drafter">
              <p>
                Which guesser to use. They produce the same answer and differ only in speed, so
                the newest is the default unless you pin one.
              </p>
            </FieldLabel>
            <Select v-model="drafterId" :options="drafterOptions" />
          </template>
          <p v-else-if="specPolicy !== 'off' && drafterSummary" class="hint">
            Uses {{ drafterSummary }}.
          </p>
        </template>
        </div>

        <div class="sf__card">
        <p class="sf__card-hd">KV offloading</p>
        <label class="sf__check">
          <Switch v-model="kvOn" label="KV offloading" />
          Keep KV cache outside VRAM
        </label>
        <template v-if="kvOn">
          <FieldLabel label="In RAM (GB)" />
          <NumberField v-model="cacheRam" :min="1" :max="1024" :step="1" />
          <p v-if="hostRamLine" class="sf__hint" :class="{ 'sf__hint--warn': hostRamOver }">
            {{ hostRamLine }}
          </p>
          <FieldLabel label="On disk (GB)">
            <p>Survives a restart. 0 = RAM only.</p>
          </FieldLabel>
          <NumberField v-model="cacheDisk" :min="0" :max="8192" :step="8" />
          <template v-if="cacheDisk > 0">
            <FieldLabel label="Folder" />
            <TextInput v-model="cacheDiskPath" :placeholder="cacheDirDefault" />
          </template>
          <p v-if="kvHoldsLine" class="sf__hint">{{ kvHoldsLine }}</p>
          <p class="sf__hint">
            Live conversations keep their KV in VRAM either way - this does not change the
            fit above.
          </p>
        </template>
        </div>

        <!-- INTELLIGENCE / CONTEXT ENRICHMENT: what the runner adds to a turn's
             context from the attachments themselves - signal-level forensics
             and file metadata (Sift). Endpoint-level defaults; every
             feature is also togglable per request over the API / in the chat
             composer. -->
        <div class="sf__card">
        <p class="sf__card-hd">Document &amp; image intelligence</p>
        <p class="sf__hint">What the runner finds in a file, beyond its content.</p>
        <!-- Cards rather than a bare switch, matching the Quality block: these
             are the two things that make an attachment worth more than its
             bytes, and a checkbox buried under a hint sold neither. -->
        <div class="sf__icards">
          <!-- Forensics: VLM-coupled, so only offered on a vision-capable model. -->
          <div
            v-if="forensicsPossible"
            class="sf__icard"
            :class="{ 'sf__icard--on': forensicsOn && visionServed, 'sf__icard--off': !visionServed }"
          >
            <div class="sf__icard-hd">
              <span class="sf__icard-title">Forensics</span>
              <Switch v-model="forensicsOn" :disabled="!visionServed" label="Forensics" />
            </div>
            <p class="sf__icard-blurb">
              Reads the original bytes for tampering - ELA, resampling, splice,
              render-vs-scan. The vision model then checks the flagged pixels
              itself.
            </p>
            <p v-if="!visionServed" class="sf__icard-warn">
              Needs a vision model - enable Vision above.
            </p>
          </div>
          <div class="sf__icard">
            <div class="sf__icard-hd">
              <span class="sf__icard-title">File metadata</span>
              <span class="sf__icard-pill">Always on</span>
            </div>
            <p class="sf__icard-blurb">
              EXIF, GPS, camera and PDF properties, from every image and PDF.
            </p>
          </div>
        </div>
        <p class="sf__hint sf__hint--muted">
          Endpoint defaults - both are togglable per request.
        </p>
        </div>

        <!-- SYSTEM TOOLS: what this model supplies server-side (its config
             file) - system-level integrations, nothing else. Callers just
             declare web_search / the server label; the endpoint owns the
             integration, hosted-API style. Callers' own tools always work. -->
        <div v-if="canTools" class="sf__card">
        <p class="sf__card-hd">System tools</p>
        <label class="sf__lbl">Web search</label>
        <div class="sf__pills">
          <button
            type="button"
            class="sf__pill"
            :class="{ 'sf__pill--on': wsProvider === '' }"
            @click="wsProvider = ''"
          >
            Off
          </button>
          <button
            v-for="p in SEARCH_PROVIDERS"
            :key="p.id"
            type="button"
            class="sf__pill sf__pill--logo"
            :class="{ 'sf__pill--on': wsProvider === p.id }"
            @click="wsProvider = p.id"
          >
            <SearchLogo :provider="p.id" :size="15" />
            {{ p.label }}
          </button>
        </div>
        <p v-if="wsChosen" class="sf__hint">{{ wsChosen.blurb }}</p>
        <div v-if="wsProvider" class="sf__field">
          <span class="sf__field-lbl">{{ searchLabel(wsProvider) }} API key</span>
          <TextInput v-model="wsKey" type="password" reveal block />
        </div>
        <p v-if="wsChosen && !wsKey.trim()" class="sf__hint sf__hint--warn">
          A provider needs its API key - create an account at
          <a :href="wsChosen.keyUrl" target="_blank" rel="noopener">{{
            wsChosen.keyUrl.replace('https://', '')
          }}</a>
          to get one.
        </p>

        <label class="sf__lbl">MCP servers</label>
        <div v-if="isEdit && connectors.list.length" class="sf__connlib">
          <label v-for="c in connectors.list" :key="c.id" class="sf__connrow">
            <Switch
              :model-value="connectorOnHere(c)"
              :disabled="c.system"
              :label="c.label"
              @update:model-value="() => toggleConnector(c)"
            />
            <span class="sf__connname">{{ c.label }}</span>
            <span class="sf__connurl">{{ c.url }}</span>
            <span v-if="c.system" class="sf__connall">on for every model</span>
          </label>
          <p class="sf__connhint">
            Add or remove connectors on the
            <RouterLink :to="{ name: 'connectors' }">Connectors page</RouterLink>.
          </p>
        </div>
        <div v-for="(r, i) in mcpRows" :key="i" class="sf__srv">
          <div class="sf__srv-row">
            <div class="sf__field sf__field--name">
              <span class="sf__field-lbl">Name</span>
              <TextInput v-model="r.label" placeholder="github" block />
            </div>
            <div class="sf__field sf__field--name">
              <span class="sf__field-lbl">Type</span>
              <div class="sf__pills">
                <button
                  type="button"
                  class="sf__pill sf__pill--sm"
                  :class="{ 'sf__pill--on': r.transport === 'http' }"
                  @click="r.transport = 'http'"
                >
                  HTTP
                </button>
                <button
                  type="button"
                  class="sf__pill sf__pill--sm"
                  :class="{ 'sf__pill--on': r.transport === 'stdio' }"
                  @click="r.transport = 'stdio'"
                >
                  Command
                </button>
              </div>
            </div>
            <div class="sf__field sf__field--grow" />
            <Tooltip label="Remove this server">
              <button type="button" class="pk-icon-btn sf__srv-del" @click="dropMcpRow(i)">
                <Icon name="trash" :size="15" />
              </button>
            </Tooltip>
          </div>

          <template v-if="r.transport === 'http'">
            <div class="sf__field">
              <span class="sf__field-lbl">URL</span>
              <TextInput v-model="r.url" placeholder="https://.../mcp" block />
            </div>
            <div v-if="r.headers.length" class="sf__field">
              <span class="sf__field-lbl">Headers</span>
              <div v-for="(h, hi) in r.headers" :key="hi" class="sf__hdr">
                <TextInput v-model="h.name" placeholder="x-api-key" class="sf__hdr-name" block />
                <TextInput v-model="h.value" placeholder="value" block />
                <button
                  type="button"
                  class="pk-icon-btn sf__srv-del"
                  aria-label="Remove header"
                  @click="dropKv(r, 'headers', hi)"
                >
                  <Icon name="x" :size="13" />
                </button>
              </div>
            </div>
          </template>
          <template v-else>
            <div class="sf__srv-row">
              <div class="sf__field sf__field--name">
                <span class="sf__field-lbl">Command</span>
                <TextInput v-model="r.command" placeholder="npx" block />
              </div>
              <div class="sf__field">
                <span class="sf__field-lbl">Arguments</span>
                <TextInput v-model="r.args" placeholder="-y @modelcontextprotocol/server-filesystem C:\data" block />
              </div>
            </div>
            <p class="sf__hint sf__hint--warn">
              This runs the command on this machine with paddock's permissions - only add
              programs you trust. A malicious MCP server can read your files and act as you;
              "Ask before each call" below keeps you in the loop.
            </p>
            <div v-if="r.envRows.length" class="sf__field">
              <span class="sf__field-lbl">Environment</span>
              <div v-for="(h, hi) in r.envRows" :key="hi" class="sf__hdr">
                <TextInput v-model="h.name" placeholder="GITHUB_TOKEN" class="sf__hdr-name" block />
                <TextInput v-model="h.value" placeholder="value" block />
                <button
                  type="button"
                  class="pk-icon-btn sf__srv-del"
                  aria-label="Remove variable"
                  @click="dropKv(r, 'envRows', hi)"
                >
                  <Icon name="x" :size="13" />
                </button>
              </div>
            </div>
          </template>

          <div class="sf__field">
            <span class="sf__field-lbl">Allowed tools · comma-separated, empty = all</span>
            <TextInput v-model="r.allowed" placeholder="" block />
          </div>

          <div class="sf__srv-foot">
            <button
              type="button"
              class="sf__addhdr"
              @click="addKv(r.transport === 'http' ? r.headers : r.envRows)"
            >
              <Icon name="plus" :size="12" />
              {{ r.transport === 'http' ? 'Add header' : 'Add variable' }}
            </button>
            <label class="sf__check sf__check--tight">
              <Switch v-model="r.approval" label="Ask before each call" /> Ask before each call
            </label>
          </div>
        </div>
        <div class="sf__keyrow">
          <button type="button" class="pk-btn pk-btn--sm" @click="addMcpRow">
            <Icon name="plus" :size="13" /> Add MCP server
          </button>
          <Select v-if="copyOptions.length > 1" v-model="copySel" :options="copyOptions" />
        </div>
        </div>

        <div class="sf__card">
        <p class="sf__card-hd">Access &amp; policy</p>
        <label class="sf__lbl">Port</label>
        <Tooltip :label="isEdit ? 'The port stays the same when you switch models' : ''">
          <div>
            <NumberField v-model="port" :min="1024" :max="65535" :disabled="isEdit" />
          </div>
        </Tooltip>
        <p v-if="portClash" class="sf__hint sf__hint--warn">
          {{ port }} is taken -
          <button type="button" class="sf__linkbtn" @click="port = fleet.nextPort()">
            use {{ fleet.nextPort() }}
          </button>
        </p>

        <!-- the key is VISIBLE: this is the operator's own box,
             the server page shows it anyway, and blind leave-blank-to-keep
             semantics were nonsense. Network callers send it; local ones
             never need it. -->
        <label class="sf__lbl">API key</label>
        <div class="sf__keyrow">
          <TextInput v-model="apiKey" placeholder="Auto-generated" block />
          <button type="button" class="pk-btn pk-btn--sm" @click="genKey">Generate</button>
        </div>

        <label class="sf__check">
          <Switch v-model="pinned" label="Never auto-stop" />
          Never auto-stop
        </label>
        <label class="sf__check">
          <Switch v-model="persist" label="Start on boot" />
          Start on boot
        </label>
        </div>
        </template>

        <!-- ADVANCED: every config key the runner has, as a form. Empty =
             the key is absent from the file (the runner's default applies).
             Saved into the file (hash-guarded) and the model restarts. -->
        <template v-else-if="mode === 'advanced'">
          <div v-if="fileError" class="sf__card">
            <p class="sf__hint sf__hint--warn">{{ fileError }}</p>
            <div><button type="button" class="pk-btn pk-btn--sm" @click="retryFile">Retry</button></div>
          </div>
          <template v-else>
          <div v-for="card in AF_CARDS" :key="card.hd" class="sf__card">
            <p class="sf__card-hd">{{ card.hd }}</p>
            <div v-for="f in card.fields" :key="f.key" class="sf__afrow">
              <div class="sf__aflbl">
                <span class="sf__af-key">{{ f.key }}</span>
                <span class="sf__af-hint">{{ f.hint }}</span>
              </div>
              <div class="sf__afctl">
                <Switch v-if="f.kind === 'switch'" v-model="afB[f.key]" :label="f.key" />
                <div v-else-if="f.kind === 'pills' || f.kind === 'bool3'" class="sf__pills">
                  <button
                    v-for="c in pillChoices(f)"
                    :key="c"
                    type="button"
                    class="sf__pill sf__pill--sm"
                    :class="{ 'sf__pill--on': afS[f.key] === c }"
                    @click="togglePill(f, c)"
                  >
                    {{ c }}
                  </button>
                </div>
                <Select v-else-if="f.kind === 'gpu'" v-model="afGpu" :options="gpuAdvOptions" />
                <textarea
                  v-else-if="f.kind === 'json'"
                  v-model="afS[f.key]"
                  class="sf__af-json"
                  spellcheck="false"
                  rows="3"
                />
                <div v-else-if="f.kind === 'file'" class="sf__af-file">
                  <TextInput v-model="afS[f.key]" block />
                  <Menu>
                    <MenuTrigger>
                      <button type="button" class="pk-btn pk-btn--sm">Browse</button>
                    </MenuTrigger>
                    <MenuContent align="end" min-width="280px">
                      <MenuItem
                        v-for="p in fileLists[f.src!]"
                        :key="p"
                        @select="pickFile(f, p)"
                      >
                        <Tooltip :label="p" side="right">
                          <span>{{ baseName(p) }}</span>
                        </Tooltip>
                      </MenuItem>
                      <MenuItem v-if="!fileLists[f.src!].length" disabled>
                        nothing found
                      </MenuItem>
                    </MenuContent>
                  </Menu>
                </div>
                <TextInput
                  v-else
                  v-model="afS[f.key]"
                  :disabled="f.key === 'port'"
                  :block="f.kind !== 'num'"
                  :placeholder="electedPlaceholder(f.elect)"
                  :class="{ 'sf__af-num': f.kind === 'num' }"
                />
              </div>
            </div>
          </div>
          </template>
          <p v-if="advError" class="sf__hint sf__hint--warn">{{ advError }}</p>
        </template>

        <!-- CONFIGURATION FILE: the raw TOML, saved verbatim (hash-guarded). -->
        <div v-else class="sf__card">
          <template v-if="fileError">
            <p class="sf__hint sf__hint--warn">{{ fileError }}</p>
            <div><button type="button" class="pk-btn pk-btn--sm" @click="retryFile">Retry</button></div>
          </template>
          <template v-else>
            <p class="sf__adv-path">{{ filePath }}</p>
            <textarea v-model="advText" class="sf__adv-editor" spellcheck="false" rows="24" />
            <p v-if="advError" class="sf__hint sf__hint--warn">{{ advError }}</p>
          </template>
        </div>

        <p v-if="simpleError && mode === 'simple'" class="sf__hint sf__hint--warn">
          {{ simpleError }}
        </p>
        <div class="sf__actions">
          <button
            class="pk-btn pk-btn--primary"
            :disabled="
              switching ||
              (mode !== 'simple' ? busy || !fileLoaded : !chosenModel || busy || portClash)
            "
            @click="submit"
          >
            <!-- Icons only where they say something the label cannot: a
                 download about to happen, a model about to start. Save gets
                 none - a checkmark reads as "done", which is the opposite of a
                 button you have not pressed yet. -->
            <Icon
              v-if="!isEdit || (mode === 'simple' && missingPieces.ids.length)"
              :name="mode === 'simple' && missingPieces.ids.length ? 'arrow-down' : 'play'"
              :size="14"
            />
            <template v-if="mode === 'simple' && missingPieces.ids.length">
              Download &amp; {{ isEdit ? 'restart' : 'start' }} ·
              {{ fmtBytes(missingPieces.bytes) }}
            </template>
            <!-- Just "Save". Whether it restarts is decided by what changed and
                 whether anything is running, and asked about THEN - a label
                 that promised a restart made every edit look destructive, and
                 was wrong for a tools-only change, which never restarts. -->
            <template v-else>{{ isEdit ? 'Save' : `Start on ${port}` }}</template>
          </button>
        </div>
      </div>

      <!-- the proposal panel: what this configuration means on this card.
           Simple only - its numbers derive from the simple form. -->
      <aside v-if="mode === 'simple'" class="sf__aside">
        <!-- Placed here, not in the model card: how much of the box a model may
             take is a property of how you are using the card, not of the model.
             And it governs every number in the panel below it, so it belongs
             where the effect is visible. -->
        <div class="sf__prop">
          <p class="sf__prop-hd">How much of the card</p>
          <ToggleGroup v-model="vramMode" class="sf__vram" label="How much of the card">
            <ToggleGroupItem
              v-for="m in VRAM_MODES"
              :key="m.value"
              :value="m.value"
              :as="m.value === 'limit' ? 'div' : undefined"
              class="sf__wlcard"
            >
              <span class="sf__wlcard-name">{{ m.name }}</span>
              <span v-if="m.value !== 'limit' || vramMode !== 'limit'" class="sf__wlcard-sub">
                {{ m.value === 'all' && freeMib ? `up to ${gb(freeMib * 1024 * 1024)}` : m.sub }}
              </span>
              <span v-else class="sf__step" @click.stop>
                <button type="button" class="sf__step-btn" @click="vramLimitGb = Math.max(1, vramLimitGb - 1)">
                  <Icon name="minus" :size="12" />
                </button>
                <span class="sf__step-val">{{ vramLimitGb }} GB</span>
                <button type="button" class="sf__step-btn" @click="vramLimitGb = vramLimitGb + 1">
                  <Icon name="plus" :size="12" />
                </button>
              </span>
            </ToggleGroupItem>
          </ToggleGroup>
        </div>
        <!-- No measurement, no card. A panel headed "Will it fit?" that answers
             "not measured yet - the runner's will-it-fit check guards the load"
             asks the question and then declines to answer it, which is worse
             than silence: it occupies the place a reader looks for the answer.
             The runner's own gate is not news to anyone and is not a reason to
             draw a box. Shown only when there is a real estimate to show, or a
             real refusal to report. -->
        <div v-if="fitData || proposal.tone === 'bad'" class="sf__prop" :class="`sf__prop--${proposal.tone}`">
          <p class="sf__prop-hd">Will it fit?</p>
          <FitChart
            v-if="fitData"
            :est="fitData.e"
            :device="fitData.d"
            :ctx="ctx"
            :batch="batch"
            :kv="reg.envelope?.kv_dtype"
            :kv-downgraded="reg.envelope?.kv_downgraded"
            :budget-bytes="reg.envelope?.budget ?? null"
            :forensics="fitForensics"
          />
          <p v-else class="sf__prop-txt">{{ proposal.text }}</p>
        </div>
      </aside>
    </div>

    <!-- Asked only when saving would interrupt something: a running model plus
         a change that binds at load. Both answers save; they differ in when the
         model picks it up. -->
    <Dialog
      :open="confirmOpen"
      role="alertdialog"
      icon="rotate-right"
      :title="`Restart ${editTitle} to apply?`"
      @close="confirmOpen = false"
    >
      <p class="sf__confirm">
        These settings are read when the model loads, so the running one keeps
        its current ones until it restarts. Restarting takes as long as loading
        the model, and anything mid-answer is interrupted.
      </p>
      <template #footer>
        <button type="button" class="pk-btn pk-btn--ghost" @click="confirmOpen = false">
          Cancel
        </button>
        <button type="button" class="pk-btn" @click="save(false)">Save for later</button>
        <button type="button" class="pk-btn pk-btn--primary" @click="save(true)">
          Save &amp; restart
        </button>
      </template>
    </Dialog>
  </form>
</template>

<style scoped>
.sf {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.sf__crumbs {
  display: flex;
  gap: 8px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  margin-bottom: 8px;
}
.sf__crumbs a {
  color: var(--pk-accent);
  text-decoration: none;
}
.sf__crumbs a:hover {
  text-decoration: underline;
}
.sf__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.sf__lead {
  margin: 0 0 20px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  max-width: 640px;
  line-height: 1.5;
}
.sf__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 28px;
  align-items: start;
}
@media (max-width: 900px) {
  .sf__grid {
    grid-template-columns: 1fr;
  }
}
/* three stacked cards - sections, not tabs: a submit-once form keeps every
   field (and every warning) visible on the way to the one Save */
.sf__col {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}
.sf__card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
}
.sf__card-hd {
  margin: 0 0 2px;
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
}
.sf__lbl {
  margin-top: 12px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.sf__select,
.sf__input {
  width: 100%;
  max-width: 440px;
  padding: 7px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
}
.sf__select--slim,
.sf__input--slim {
  width: auto;
  min-width: 220px;
}
.sf__input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.sf__hint {
  margin: 2px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.sf__hint--warn {
  color: var(--pk-status-warning);
}
.sf__hint--muted {
  font-style: italic;
  opacity: 0.85;
}
.sf__linkbtn {
  background: none;
  border: none;
  padding: 0;
  font: inherit;
  color: var(--pk-accent);
  cursor: pointer;
  text-decoration: underline;
}
.sf__fixed {
  display: flex;
  align-items: center;
  gap: 12px;
}
.sf__fixed-name {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.sf__fixed-change {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-accent);
  text-decoration: none;
}
.sf__fixed-change:hover {
  text-decoration: underline;
}
.sf__keyrow {
  display: flex;
  /* stretch, and pin the button's height to the input's: the sm button sat
     visibly shorter than the field beside it */
  align-items: stretch;
  gap: 8px;
}
/* `height: auto` is what lets the button stretch to the field beside it - but
   it also drops the button's OWN height, so a row with no field (Add MCP
   server, whose Select only renders with >1 copy option) collapsed to padding
   plus one line, ~17px. min-height puts the floor back without blocking the
   stretch. */
.sf__keyrow .pk-btn {
  height: auto;
  min-height: 34px;
}
.sf__keyrow .pk-btn--sm {
  min-height: 28px;
}
/* one MCP server = one structured sub-card with LABELED fields */
.sf__connlib {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}
.sf__connrow {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  cursor: pointer;
}
.sf__connname {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.sf__connurl {
  flex: 1;
  min-width: 0;
  font-size: var(--pk-font-size-xs);
  font-family: var(--pk-font-mono);
  color: var(--pk-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sf__connall {
  flex-shrink: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-accent);
  font-weight: 600;
}
.sf__connhint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.45;
}
.sf__connhint a {
  color: var(--pk-text-secondary);
}
.sf__confirm {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.sf__srv {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  margin-top: 2px;
}
.sf__srv-row {
  display: flex;
  align-items: flex-end;
  gap: 10px;
}
.sf__srv-del {
  flex: none;
  margin-bottom: 2px;
  color: var(--pk-text-muted);
}
.sf__srv-del:hover {
  color: var(--pk-text-danger);
}
.sf__field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 0;
}
.sf__field--name {
  flex: 0 0 180px;
}
.sf__field--grow {
  flex: 1;
}
.sf__pill--sm {
  padding: 4px 10px;
  font-size: var(--pk-font-size-xs);
}
/* the field's inputs fill their column exactly - no inherited caps */
.sf__field :deep(.pk-input--block) {
  max-width: none;
}
.sf__hdr {
  display: flex;
  align-items: center;
  gap: 8px;
}
.sf__hdr-name {
  flex: 0 0 180px;
}
.sf__srv-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.sf__addhdr {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  cursor: pointer;
}
.sf__addhdr:hover {
  color: var(--pk-accent);
}
.sf__field-lbl {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
}
.sf__check--tight {
  margin-top: 0;
}
.sf__hint a {
  color: var(--pk-accent);
}
/* Four equal columns on one row: the four workload choices are peers, so they
   get equal width rather than shrink-to-fit pills of four different sizes. */
.sf__wl {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
/* The aside is a narrow column, so these three stack rather than sitting in a
   row like the workload four do. Same card treatment, so they read as the same
   KIND of control. */
.sf__vram {
  display: grid;
  gap: 6px;
}
.sf__vram :deep(.sf__wlcard) {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 7px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.sf__vram :deep(.sf__wlcard:hover) {
  border-color: var(--pk-accent);
}
.sf__vram :deep(.sf__wlcard[data-state='on']) {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
.sf__vram :deep(.sf__wlcard-name) {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  line-height: 1.2;
}
.sf__vram :deep(.sf__wlcard-sub) {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
@media (max-width: 620px) {
  .sf__wl {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
.sf__wl :deep(.sf__wlcard) {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.sf__wl :deep(.sf__wlcard:hover) {
  border-color: var(--pk-accent);
}
.sf__wl :deep(.sf__wlcard[data-state='on']) {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
.sf__wl :deep(.sf__wlcard-name) {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  line-height: 1.2;
}
.sf__wl :deep(.sf__wlcard-sub) {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* One plain sentence under a control, on what the choice MEANS. Not narration:
   every use is a mental model the control cannot carry in its own label. */
/* Two figures, not a sentence: the number leads and the unit follows it in a
   quieter weight, so the pair scans as data rather than as prose to read. The
   separator is a rule between them rather than a comma, which keeps them
   reading as two independent measurements of the same thing. */
.sf__scale {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 14px;
  margin: -2px 0 2px;
}
.sf__stat {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
}
.sf__stat + .sf__stat {
  padding-left: 14px;
  border-left: 1px solid var(--pk-border-subtle);
}
.sf__stat-v {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-secondary);
}
.sf__stat-u {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.sf__hint {
  margin: -2px 0 2px;
  max-width: 62ch;
  font-size: var(--pk-font-size-xs);
  line-height: 1.45;
  color: var(--pk-text-muted);
}
.sf__pills {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.sf__pillgroup {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
/* .sf__pill has FIVE groups and only Workload is a Reka ToggleGroup, so every
   rule needs both forms: the plain one for buttons this file renders itself,
   and a :deep() one for the ToggleGroupItems, which Reka renders through an
   asChild clone that drops our scope attribute. Scoping the class to just one
   of its users is what broke the other four. */
.sf__pill,
.sf__pillgroup :deep(.sf__pill) {
  display: inline-flex;
  align-items: baseline;
  gap: 6px;
  padding: 5px 11px;
  border: 1px solid var(--pk-border-default);
  border-radius: 999px;
  background: var(--pk-bg-surface);
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
  white-space: nowrap;
}
.sf__pill:hover,
.sf__pillgroup :deep(.sf__pill:hover) {
  border-color: var(--pk-accent);
}
/* the plain groups mirror state in a :class; the ToggleGroup gets it from Reka */
/* The base pill aligns on the text baseline, which is right for word-only
   pills and drops a logo half a line low. A pill carrying a mark centres. */
.sf__pill--logo {
  align-items: center;
}
.sf__pill--on,
.sf__pillgroup :deep(.sf__pill[data-state='on']) {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
  font-weight: 600;
}
.sf__pill-sub {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  font-weight: 400;
}
/* quality radio-cards: the meaning up front, the quant tag as a footnote */
/* Explicit column count, not auto-fit. auto-fit with a 220px floor decides for
   itself how many fit, so three cards became 2 + 1 orphaned on the next row the
   moment the window narrowed - the wrap the user hit. `--qc` is the card count
   capped at 3 (two is the usual case), and the breakpoints step it down rather
   than letting the grid improvise. */
.sf__qcards {
  display: grid;
  grid-template-columns: repeat(var(--qc, 2), minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 1180px) {
  .sf__qcards {
    grid-template-columns: repeat(min(var(--qc, 2), 2), minmax(0, 1fr));
  }
}
@media (max-width: 720px) {
  .sf__qcards {
    grid-template-columns: minmax(0, 1fr);
  }
}
/* A stepper that lives inside its own choice, rather than a full-width field
   appearing underneath it. The cell renders as a div (see ToggleGroupItem's
   `as`) so these buttons are not nested in a button. */
/* the blocked card: content recedes, the requirement sits across it */
.sf__qcards :deep(.sf__qcard--blocked) {
  position: relative;
}
.sf__qcards :deep(.sf__qcard--blocked) > *:not(.sf__qcard-veil) {
  opacity: 0.32;
}
.sf__qcard-veil {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 6px;
  border-radius: inherit;
  pointer-events: none;
}
.sf__qcard-veil-txt {
  padding: 4px 9px;
  border: 1px solid var(--pk-border-strong);
  border-radius: 999px;
  background: var(--pk-bg-elevated);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-align: center;
  line-height: 1.25;
}
.sf__step {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin-top: 1px;
}
.sf__step-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  padding: 0;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-surface);
  color: var(--pk-text-secondary);
  cursor: pointer;
}
.sf__step-btn:hover {
  border-color: var(--pk-accent);
  color: var(--pk-text-primary);
}
.sf__step-val {
  min-width: 3.5ch;
  text-align: center;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
}
.sf__qcards :deep(.sf__qcard) {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  text-align: left;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  cursor: pointer;
  transition:
    border-color 0.12s ease,
    background 0.12s ease;
}
.sf__qcards :deep(.sf__qcard:hover) {
  border-color: var(--pk-border-strong);
}
.sf__qcards :deep(.sf__qcard[data-state='checked']) {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
/* Flex row rather than inline text: a card is only ~220px, so a title that
   carries a badge is one wrap away at any time. As a wrapping flex row the
   badge drops to its own line squared up with the name instead of trailing
   off it, and the gap keeps the two apart on both layouts. */
.sf__qcard-title {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 6px;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.sf__qcard-blurb {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  line-height: 1.45;
}
.sf__qcards :deep(.sf__qcard) {
  height: 100%;
}
/* The blurb is the only variable-length part, so it takes the slack - that
   pins the notes to the bottom edge and keeps SIZE · QUANT on the same line of
   every card, which is what makes two cards comparable at a glance. */
.sf__qcard-blurb {
  flex: 1 1 auto;
}
.sf__qcard-size {
  font-weight: 650;
  color: var(--pk-text-primary);
}
.sf__qcard-quant {
  padding: 0 5px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-inset);
  font-family: var(--pk-font-mono);
  font-size: 0.92em;
}
.sf__qcard-meta {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: var(--pk-font-mono);
  font-size: 11px;
  color: var(--pk-text-muted);
}
.sf__qcard-warn {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-danger);
}
/* A wire-fact, not narration: a native-plane build reads the rest of its
   weights from the base file, which is why picking it downloads two things. */
.sf__qcard-note {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* A format this GPU has no kernels for. Dimmed rather than hidden: knowing the
   small build EXISTS, and what it would take, is worth more than a card that
   silently is not there - that is the same question answered by the supported-
   GPU sheet rather than by omission. */
.sf__qcard--blocked {
  opacity: 0.55;
  cursor: not-allowed;
}
/* Intelligence cards. Same visual language as the Quality cards above - border,
   radius, base ground, accent when live - because they answer the same shape of
   question ("which of these do I want"), just with a switch instead of a pick.
   Not RadioItems: these are independent toggles, not one choice. */
.sf__icards {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin: 6px 0 2px;
}
@media (max-width: 720px) {
  .sf__icards {
    grid-template-columns: minmax(0, 1fr);
  }
}
.sf__icard {
  display: flex;
  flex-direction: column;
  gap: 6px;
  height: 100%;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  transition:
    border-color 0.12s ease,
    background 0.12s ease;
}
.sf__icard--on {
  border-color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
/* Dimmed, not hidden: knowing forensics is here and what it needs beats a card
   that silently is not there - the same rule the blocked Quality card follows. */
.sf__icard--off {
  opacity: 0.55;
}
.sf__icard-hd {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.sf__icard-title {
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.sf__icard-blurb {
  flex: 1 1 auto;
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  line-height: 1.45;
}
.sf__icard-warn {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
}
.sf__icard-pill {
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  white-space: nowrap;
}
.sf__capline {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 2px 0 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.sf__capnote {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}
.sf__capnote a,
.sf__hint a {
  color: var(--pk-accent);
}
.sf__check {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.sf__actions {
  display: flex;
  gap: 10px;
  margin-top: 22px;
}

/* proposal panel */
.sf__aside {
  display: flex;
  flex-direction: column;
  gap: 14px;
  position: sticky;
  top: 0;
}
.sf__prop {
  padding: 14px;
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
}
.sf__prop--bad {
  border-color: var(--pk-text-danger);
}
.sf__prop--ok {
  border-color: var(--pk-status-warning);
}
.sf__prop-hd {
  margin: 0 0 8px;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--pk-text-muted);
}
.sf__prop-txt {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.sf__prop--bad .sf__prop-txt {
  color: var(--pk-text-danger);
}
/* the Simple | Advanced | Configuration file tabs */
.sf__mode {
  margin-bottom: 16px;
}
/* the Advanced form: one settings row per config key - label + hint on the
   left, a right-sized control on the right (numbers narrow, paths wide) */
.sf__afrow {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 2px 18px;
  align-items: center;
  padding: 8px 0;
}
.sf__afrow + .sf__afrow {
  border-top: 1px solid var(--pk-border-subtle);
}
@media (max-width: 700px) {
  .sf__afrow {
    grid-template-columns: 1fr;
  }
}
.sf__aflbl {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.sf__af-key {
  font-family: var(--pk-font-mono);
  font-size: 12px;
  color: var(--pk-text-primary);
}
.sf__af-hint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.35;
}
.sf__afctl {
  min-width: 0;
  display: flex;
  align-items: center;
}
.sf__afctl :deep(.pk-input--block) {
  max-width: none;
}
.sf__af-num {
  width: 150px;
  min-width: 0;
}
.sf__af-file {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}
.sf__af-json {
  width: 100%;
  box-sizing: border-box;
  padding: 8px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-family: var(--pk-font-mono);
  font-size: 12px;
  line-height: 1.55;
  color: var(--pk-text-primary);
  resize: vertical;
  min-height: 60px;
}
.sf__af-json:focus {
  outline: none;
  border-color: var(--pk-accent);
}
/* the Configuration file editor: the file, as-is */
.sf__adv-path {
  margin: 0 0 8px;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  word-break: break-all;
}
.sf__adv-editor {
  width: 100%;
  box-sizing: border-box;
  padding: 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-family: var(--pk-font-mono);
  font-size: 12px;
  line-height: 1.55;
  color: var(--pk-text-primary);
  resize: vertical;
  min-height: 340px;
}
.sf__adv-editor:focus {
  outline: none;
  border-color: var(--pk-accent);
}
</style>
