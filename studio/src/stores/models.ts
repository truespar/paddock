import { uiPreferences } from '@/lib/ui-preferences'
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { VisionBudget } from '@/lib/vision'
import type { TaskTag } from '@/lib/tasks'
import { ocrCapsFrom, type OcrCaps } from '@/lib/ocr'
import { isHarmony, isVisionModel } from '@/lib/model-caps'
import { realtimeEnrichment, type RealtimeTranscriptionCaps } from '@/lib/audio-policy'
import { DEFAULT_MAX_QUESTIONS, DEFAULT_MAX_SAMPLES, type StructuredReadCaps } from '@/lib/reads'

export type ModelKind =
  | 'chat'
  | 'encoder'
  | 'transcriber'
  | 'aligner'
  | 'image'
  | 'reader'
  | 'tabular'
  | 'diarizer'
  | 'segmenter'
  | 'masker'

/** Can this kind hold a lane in the chat surface - i.e. does it ANSWER a user
 *  turn? Chat models reply in text, transcribers reply with a transcript; both
 *  are turns and both belong on the same surface, which is what makes
 * compare work for ASR. Encoders return a vector, which is not a
 *  turn, so they keep their own page.
 *
 *  This is deliberately about the KIND, not the model: two lanes must still
 *  agree on kind, because a transcript and a chat reply to "the same input"
 *  are not answers to the same question.
 *
 *  An ALIGNER never takes a turn either: it annotates an existing transcript
 *  with word times (the enrichment pass), it does not answer anything - so
 *  like an encoder it stays out of the header picker.
 *
 *  An IMAGE model takes turns too: a prompt in, a picture out, is a turn the
 *  same way a clip in, a transcript out, is one - and that is what gives it
 *  history, compare and a persisted record for free. What it cannot do is
 *  share a conversation with a chat model, which the lane guard enforces. */
export function takesTurns(kind: ModelKind): boolean {
  return kind === 'chat' || kind === 'transcriber' || kind === 'image'
}

export interface ModelInfo {
  id: string
  ownedBy: string
  /** The catalog's human name ("Qwen 3.5 9B") - what pickers show; the
   *  technical id stays in `id` for tooltips and the wire. */
  display?: string
  /** The maker ("Alibaba", "OpenAI") - drives the vendor logo. */
  vendor?: string
  /** Server-advertised image capability; only known for the selected model. */
  vision?: boolean
  /** speculation mechanism beside the model name, everywhere it shows. */
  spec?: string
  /** The runner port serving this model - absent on a cloud model, which is
   *  reached through its endpoint instead. */
  port?: number
  /** Generative chat model, encoder (embeddings/rerank), or transcriber
   *  (whisper-family: speech in, text out, and it REFUSES chat). A generative
   *  ASR model with an audio mmproj (Qwen3-ASR, granite-speech) is `chat` here
   *  and transcribes as well.
   *
   *  A transcriber does reach the header picker: it answers a user
   *  turn, so it holds a lane like any other model - the composer switches to
   *  audio input and the turn renders as a transcript. What it cannot do is
   *  share a conversation with a chat model, which [`takesTurns`] and the
   *  lane guard enforce. An ENCODER still never appears: an embedding is a
   *  vector, not a reply, so it stays on its own page. */
  kind: ModelKind
  /** Live runner status: ok | draining | unreachable. */
  status: string
  /** Set on a Cloud model (BYO-key provider): which endpoint serves it and
   *  what the endpoint is called ("OpenRouter") for picker hints. */
  cloud?: {
    endpoint: string
    endpointName: string
    /** the endpoint can run a web search for any model (OpenRouter) */
    webSearch?: boolean
    ctx?: number
    /** The provider's own ceiling on one reply, which is a different number
     *  from `ctx` and usually far smaller. See [`CloudModelPick.maxOut`]. */
    maxOut?: number
    reasoning?: boolean
  }
}

/** One enabled model on a cloud endpoint, as stored on the manager - plus
 *  the picker-only metadata the provider list carries (prices drift, so only
 *  id/display/ctx/vision are persisted when a model is enabled). */
export interface CloudModelPick {
  id: string
  display?: string
  ctx?: number
  /** The most tokens this model will emit in one reply - the provider's own
   *  number (OpenRouter `top_provider.max_completion_tokens`), and not the
   *  context window: deepseek-v4-flash-0731 is 1M context / 384k output.
   *  Persisted with the pick because the reply cap depends on it - "Model
   *  maximum" used to mean "the whole window", which asked a 1M-context model
   *  for ~1M output and earned a 400 on prompt+output > context. */
  maxOut?: number
  /** Speech in, a transcript out, and it does not chat - the provider's own
   *  `output_modalities: ["transcription"]`. Persisted with the pick because
   *  it decides the model's KIND, and a transcriber that came back as a chat
   *  model would be sent a text prompt it can only refuse. */
  asr?: boolean
  vision?: boolean
  /** OpenRouter provider pin: this pick always routes to one provider
   *  (its price/quant was chosen deliberately) instead of auto-routing. */
  provider?: string
  /** USD per token from the provider list; rendered as $/M. */
  promptPrice?: number
  completionPrice?: number
  reasoning?: boolean
  free?: boolean
  created?: number
  /** description tail - feeds the picker's search, never shown as UI copy */
  blurb?: string
}

/** A BYO-key provider endpoint (manager /api/cloud row - never carries the
 *  key itself, only `hasKey`). */
export interface CloudEndpoint {
  id: string
  name: string
  kind: 'openai' | 'openai-compat' | 'anthropic'
  baseUrl: string
  hasKey: boolean
  /** Desktop preloads credentials at startup; a request never prompts Keychain. */
  credentialReady?: boolean
  allowUnauthenticated?: boolean
  models: CloudModelPick[]
  createdAt: number
}

/** Best-effort maker from a provider model id, for the vendor mark
 *  ("anthropic/claude-..." on OpenRouter, bare "gpt-5.2" on OpenAI). The org
 *  prefix decides when there is one; model-brand substrings cover bare ids.
 *  Unknown makers fall back to VendorLogo's letter badge. */
const CLOUD_ORGS: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
  meta: 'Meta',
  'meta-llama': 'Meta',
  deepseek: 'DeepSeek',
  tencent: 'Tencent',
  xiaomi: 'Xiaomi',
  'z-ai': 'Z.ai',
  nvidia: 'NVIDIA',
  mistral: 'Mistral',
  mistralai: 'Mistral',
  moonshotai: 'Moonshot',
  'x-ai': 'xAI',
  qwen: 'Alibaba',
  alibaba: 'Alibaba',
  perplexity: 'Perplexity',
  baidu: 'Baidu',
  bytedance: 'ByteDance',
  minimax: 'MiniMax',
  poolside: 'Poolside',
  ibm: 'IBM',
  cohere: 'Cohere',
  huggingface: 'Hugging Face',
  openrouter: 'OpenRouter',
  'aleph-alpha': 'Aleph Alpha',
}
export function cloudVendor(id: string): string | undefined {
  const s = id.toLowerCase()
  const slash = s.indexOf('/')
  if (slash > 0) {
    const org = CLOUD_ORGS[s.slice(0, slash).replace(/^~+/, '')]
    if (org) return org
  }
  if (s.includes('gpt') || /^o[134]\b/.test(s)) return 'OpenAI'
  if (s.includes('claude')) return 'Anthropic'
  if (s.includes('gemini') || s.includes('gemma')) return 'Google'
  if (s.includes('qwen')) return 'Alibaba'
  if (s.includes('granite')) return 'IBM'
  if (s.includes('llama')) return 'Meta'
  if (s.includes('grok')) return 'xAI'
  if (s.includes('kimi')) return 'Moonshot'
  if (s.includes('glm')) return 'Z.ai'
  if (s.includes('deepseek')) return 'DeepSeek'
  if (s.includes('mistral') || s.includes('mixtral')) return 'Mistral'
  if (s.includes('kolibri')) return 'Aleph Alpha'
  return undefined
}

/** Shared naming seam for catalog/Compare parity tests with the native client. */
export function cloudModelIdentity(id: string, display?: string, kind = '', provider?: string) {
  const inferred = cloudVendor(id)
  const vendor = inferred ?? (kind === 'openai' ? 'OpenAI' : kind === 'anthropic' ? 'Anthropic' : undefined)
  const name = inferred ? (display ?? id).replace(/^[^:]{2,24}:\s+/, '') : (display ?? id)
  return { name: provider ? `${name} (${provider})` : name, vendor }
}

/** One endpoint's advertised SERVER TOOLS (its per-model config, read from
 *  /api/server): the Studio only ever declares what a server actually has -
 *  a compare lane never asks a model for tools its config doesn't supply. */
/** Forensics as an endpoint advertises it on /server (the `[forensics]`
 *  gate). Absent when the endpoint has forensics disabled. `vision` is whether
 *  this model can actually use the findings - forensics is VLM-coupled, so the
 *  composer only offers the control when both the block is present AND vision is
 *  true. `auto` is the always-on scope the endpoint defaults to
 *  ("off" | "images" | "all"). */
export interface ForensicsCaps {
  auto: string
  tool: boolean
  vision: boolean
}

export interface ModelCaps {
  webSearch: boolean
  /** The endpoint serves the builtin current_time tool (any chat runner from
   *  the build that ships it; absent on older runners, which would 400 the
   *  declaration - so the tool only ever rides where it is real). */
  currentTime?: boolean
  mcpServers: string[]
  /** The endpoint's forensics config, or undefined when disabled. */
  forensics?: ForensicsCaps
  vision?: boolean
  /** The endpoint's reasoning control style - per-LANE fact so compare can
   *  show the thinking toggle when any armed lane supports it. */
  reasoning?: 'effort' | 'toggle' | 'none'
  /** The rungs this model grades effort at, in its own spelling, lowest
   *  first - measured from its chat template by the runner, so Qwen3.8's
   *  ['low','medium','xhigh'] and gpt-oss's ['low','medium','high'] are the
   *  actual vocabularies and not a shared guess. Empty when the model only
   *  has a switch. */
  reasoningLevels?: string[]
  /** The rung an unset request gets - the checkpoint's published default, so
   *  the picker can open on the truth instead of a house value. */
  reasoningDefault?: string
  /** Reasoning can be turned off here. False on gpt-oss and Muse Glimmer,
   *  whose templates render their reasoning preamble unconditionally: an Off
   *  item in their picker would be a control that does nothing. */
  reasoningOff?: boolean
  /** The served template lets a caller decide whether a prior turn's thinking
   *  stays in the prompt (`preserve_thinking`). Measured per template by the
   *  runner - only the qwen3.6/3.8 families grade it - so a control keyed off
   *  this is never drawn for a model that would ignore it. */
  reasoningPreserve?: boolean
  /** A thinking budget (reasoning.max_tokens) can be ENFORCED on this lane -
   *  dialect-shaped truth from the runner (qwen/laguna/gemma think-blocks
   *  yes; gpt-oss/muse channel reasoning refuses the knob honestly). */
  thinkingBudget?: boolean
  /** What an image costs on this endpoint, from the tower it loaded. Absent on
   *  a model without vision (and on a runner too old to advertise it, which
   *  reads the same way: no per-image size control). */
  visionBudget?: VisionBudget
  /** Canned instructions this model's chat template expands (granite-vision's
   *  chart/table extraction tasks). Empty for every other model. */
  taskTags: TaskTag[]
  /** The deepseek2-ocr request surface: the reading-mode
   *  vocabulary and whether grounded regions come back. Absent on every
   *  other model - the composer offers the modes only off this, never a
   *  name heuristic. */
  ocr?: OcrCaps
  /** This model only reads documents (deepseek2-ocr, paddleocr): the server
   *  400s text-only chat on it, so the composer requires an attached
   * image/PDF before send. */
  docParser?: boolean
  /** This endpoint serves /v1/images/generations: its grid and ceilings,
   *  which are the model's, so the composer's size and step controls never
   *  offer what the endpoint would refuse. Absent on every other model. */
  imageGeneration?: ImageGenCaps
  /** This endpoint reads a text against fixed questions in one pass
   *  (`POST /v1/systemone`, a block-diffusion model): the caps the runner
   *  advertises. Absent on every model that generates one token at a time,
   *  which is what keeps the Reads page from existing for a chat model. */
  structuredRead?: StructuredReadCaps
  /** This endpoint serves /v1/audio/transcriptions. True for both shapes: a
   *  whisper-family runner (speech in, text out, no chat at all) and a
   *  generative model with its audio mmproj loaded (Qwen3-ASR,
   *  granite-speech), which chats as well. That is why audio is a CAPABILITY
   *  question here and never a kind check. */
  audio?: boolean
  realtimeTranscription?: RealtimeTranscriptionCaps
  /** Which `timestamp_granularities[]` this endpoint can answer. Whisper says
   *  `["segment", "word"]` - two unrelated mechanisms, its timestamp
   *  vocabulary and cross-attention alignment; the generative ASR families
   *  have neither and say `[]`, which is how they decline instead of
   *  returning an empty `segments` array nobody can explain. */
  timestampGranularities: string[]
  /** Which `include` values /v1/audio/transcriptions honours here.
   *  `logprobs` = it can report per-word confidence. Asked separately from
   *  the granularities because "how sure were you" needs no clock: a
   * generative ASR model has logprobs and no timestamps. */
  include: string[]
  /** This endpoint serves /v1/audio/alignments (a forced-aligner runner). */
  aligner?: boolean
  /** The longest clip one alignment call can address, seconds - the head's
   *  time-bin budget. The enrichment pass split-or-skips on this before
   *  posting bytes rather than learning the cap from a 400. */
  alignmentMaxClipS?: number
  /** The longest clip TRANSCRIPTION can take on this endpoint, seconds.
   *  Absent means no ceiling worth showing: whisper windows a clip, so length
   *  costs time and nothing else. A generative ASR lane spends the whole clip
   *  as prompt rows, so its ceiling is the context window and it is lower than
   *  people guess (~42 min at 32k on Qwen3-ASR). The recorder stops at this
   * instead of letting you find out from a refusal an hour. */
  transcriptionMaxClipS?: number
  /** The server's own sampling defaults (what an untouched dial means) -
   *  advertised so the popover can show the number instead of a blind word.
   *  `source` names where they came from: usually the model's own published
   *  values, which is why two models here show different numbers. */
  sampling?: {
    temperature?: number
    top_p?: number
    top_k?: number
    min_p?: number
    repeat_penalty?: number
    source?: string
  }
  /** This endpoint's context window - a per-LANE fact (the store-level
   *  `maxCtx` tracks only the SELECTED model), so compare can plan against
   *  the smallest window among its lanes. */
  maxCtx?: number
}

/** What an image-generation endpoint advertises about itself (the runner's
 *  `/api/server` `image_generation` block): sides must be multiples of
 *  `sizeMultiple` up to `maxSide`; `defaultSteps` is the model's own count
 *  (what `quality: auto` means), `maxPartialImages` how many progressive
 *  previews a streamed render can send. */
export interface ImageGenCaps {
  sizeMultiple: number
  maxSide: number
  defaultSize: string
  defaultSteps: number
  maxSteps: number
  maxN: number
  stream: boolean
  maxPartialImages: number
  outputFormats: string[]
  /** editing with reference images is served */
  edit: boolean
  /** reference pictures per edit */
  maxReferences: number
}

/** The /api/server (or relayed /runners/{port}/server) body, as far as the
 *  Studio reads it. */
interface ServerBody {
  structured_read?: {
    canvas_width?: number
    max_questions?: number
    max_samples?: number
    max_steps?: number
    images?: boolean
    conditional?: boolean
    think?: boolean
    types?: string[]
  } | null
  web_search?: boolean
  mcp_servers?: string[]
  forensics?: { auto?: string; tool?: boolean; vision?: boolean } | null
  vision?: boolean
  sampling?: {
    temperature?: number
    top_p?: number
    top_k?: number
    min_p?: number
    repeat_penalty?: number
    source?: string
  }
  vision_budget?: VisionBudget | null
  task_tags?: TaskTag[] | null
  ocr?: unknown
  document_parser?: boolean | null
  current_time?: boolean | null
  audio?: boolean
  image_generation?: {
    size_multiple?: number
    max_side?: number
    default_size?: string
    default_steps?: number
    max_steps?: number
    max_n?: number
    stream?: boolean
    max_partial_images?: number
    output_formats?: string[]
    edit?: boolean
    max_references?: number
  } | null
  realtime_transcription?: RealtimeTranscriptionCaps | null
  aligner?: string | null
  /** image-generation runner: the picture model it serves, its only role */
  image_model?: string | null
  alignment_max_clip_s?: number | null
  transcription_max_clip_s?: number | null
  timestamp_granularities?: string[] | null
  include?: string[] | null
  max_ctx?: number
  default_max_output_tokens?: number
  reasoning?: string
  reasoning_levels?: string[] | null
  reasoning_default?: string | null
  reasoning_off?: boolean | null
  reasoning_preserve?: boolean | null
  thinking_budget?: boolean | null
  pdf?: { enabled?: boolean; raster?: boolean; max_pages?: number }
  version?: string
  /** `version` plus the commit - display only, never compared. */
  build?: string
}

/** One endpoint's advertised capabilities. Both fetch paths go through this,
 *  so neither can cache a half-populated record over the other's - which is
 *  exactly what happened when the limits fetch wrote caps without the vision
 *  budget and won the race against capsFor(). */
function parseCaps(body: ServerBody): ModelCaps {
  return {
    webSearch: body.web_search ?? false,
    currentTime: body.current_time ?? undefined,
    mcpServers: body.mcp_servers ?? [],
    forensics: body.forensics
      ? {
          auto: body.forensics.auto ?? 'off',
          tool: body.forensics.tool ?? false,
          vision: body.forensics.vision ?? false,
        }
      : undefined,
    vision: body.vision ?? undefined,
    visionBudget: body.vision_budget ?? undefined,
    taskTags: body.task_tags ?? [],
    ocr: ocrCapsFrom(body.ocr),
    docParser: body.document_parser ?? undefined,
    audio: body.audio ?? undefined,
    imageGeneration: body.image_generation
      ? {
          sizeMultiple: body.image_generation.size_multiple ?? 32,
          maxSide: body.image_generation.max_side ?? 1024,
          defaultSize: body.image_generation.default_size ?? '1024x1024',
          defaultSteps: body.image_generation.default_steps ?? 40,
          maxSteps: body.image_generation.max_steps ?? 100,
          maxN: body.image_generation.max_n ?? 1,
          stream: body.image_generation.stream ?? false,
          maxPartialImages: body.image_generation.max_partial_images ?? 0,
          outputFormats: body.image_generation.output_formats ?? ['png'],
          edit: body.image_generation.edit ?? false,
          maxReferences: body.image_generation.max_references ?? 0,
        }
      : undefined,
    structuredRead: body.structured_read
      ? {
          canvasWidth: body.structured_read.canvas_width ?? 0,
          maxQuestions: body.structured_read.max_questions ?? DEFAULT_MAX_QUESTIONS,
          maxSamples: body.structured_read.max_samples ?? DEFAULT_MAX_SAMPLES,
          // a runner from before these fields reads in one pass, text only,
          // every question at once
          maxSteps: Math.max(1, body.structured_read.max_steps ?? 1),
          images: body.structured_read.images === true,
          conditional: body.structured_read.conditional === true,
          think: body.structured_read.think === true,
          types: body.structured_read.types ?? ['noul', 'choice', 'score'],
        }
      : undefined,
    realtimeTranscription: body.realtime_transcription ?? undefined,
    aligner: body.aligner ? true : undefined,
    alignmentMaxClipS: body.alignment_max_clip_s ?? undefined,
    transcriptionMaxClipS: body.transcription_max_clip_s ?? undefined,
    timestampGranularities: body.timestamp_granularities ?? [],
    include: body.include ?? [],
    reasoning: (body.reasoning as ModelCaps['reasoning']) ?? undefined,
    reasoningLevels: body.reasoning_levels ?? undefined,
    reasoningDefault: body.reasoning_default ?? undefined,
    reasoningOff: body.reasoning_off ?? undefined,
    reasoningPreserve: body.reasoning_preserve ?? undefined,
    thinkingBudget: body.thinking_budget ?? undefined,
    sampling: body.sampling ?? undefined,
    maxCtx: body.max_ctx ?? undefined,
  }
}

interface RunnerRow {
  port: number
  model?: string | null
  /** speculation mechanism in words ("MTP", "DFlash1", "off") - the manager's
   *  spawn resolution; absent = nothing to say (non-speculative model). */
  spec?: string | null
  embedder?: string | null
  /** whisper-family runner: it serves speech-to-text and nothing else, so it
   *  reports neither `model` nor `embedder` */
  asr?: string | null
  /** forced-alignment runner: word times for an existing transcript, and
   *  nothing else - the same only-role story as `asr` */
  aligner?: string | null
  /** image-generation runner: /v1/images/* and nothing else */
  image?: string | null
  /** decision-model runner (Laya): /v1/systemone and nothing else - it reads,
   *  it never chats */
  reader?: string | null
  /** tabular predictor (Kumo-Tabular): /v1/tabular/* and nothing else -
   *  labelled rows in, the missing column out; it never chats */
  tabular?: string | null
  diarization?: string | null
  /** dense-prediction runner (DINOv3): /v1/segmentations and nothing else */
  segmenter?: string | null
  /** promptable-segmentation runner (SAM 3): /v1/masks and nothing else - a
   *  picture and a prompt in, masks out; it never chats */
  masker?: string | null
  display?: string | null
  vendor?: string | null
  status?: string
}

/** The model a runner row serves, whichever role it is in (null: none yet). */
function rowId(r: RunnerRow): string | null {
  return (
    r.model ??
    r.embedder ??
    r.asr ??
    r.aligner ??
    r.reader ??
    r.tabular ??
    r.diarization ??
    r.segmenter ??
    r.masker ??
    r.image ??
    null
  )
}

/** A runner row as a picker entry. `prev` is the entry this id had before the
 *  rebuild: its caps-fetched fields survive, and an 'unreachable' flap keeps
 *  the last real status (the supervisor's health call misses under load). */
function rowModel(r: RunnerRow, prev: ModelInfo | undefined): ModelInfo {
  const raw = r.status ?? 'unknown'
  return {
    id: rowId(r) as string,
    ownedBy: 'paddock',
    display: r.display ?? undefined,
    // the manager names makers from the catalog only; an uncatalogued model
    // (a hand-started file, a local-dev checkpoint) gets the id heuristic the
    // reply headers already use, so the picker and the lane agree on the mark
    vendor: r.vendor ?? cloudVendor(rowId(r) ?? ''),
    port: r.port,
    kind: r.model
      ? 'chat'
      : r.embedder
        ? 'encoder'
        : r.asr
          ? 'transcriber'
          : r.aligner
            ? 'aligner'
            : r.reader
              ? 'reader'
              : r.tabular
                ? 'tabular'
                : r.diarization
                  ? 'diarizer'
                  : r.segmenter
                    ? 'segmenter'
                    : r.masker
                      ? 'masker'
                      : 'image',
    status: raw === 'unreachable' && prev?.status ? prev.status : raw,
    vision: prev?.vision,
    spec: r.spec ?? undefined,
  }
}

/**
 * The running fleet, from the manager's /api/runners (one model per runner,
 * doc §3). Chat traffic goes through the manager's Studio relay
 * (/api/runners/{port}/v1/responses) - the browser never talks to a runner
 * port directly, so runner keys stay server-side and there's no CORS.
 */
export const useModelsStore = defineStore('models', () => {
  const models = ref<ModelInfo[]>([])
  // Cloud endpoints (manager /api/cloud): the raw rows for the management
  // page; their enabled models are also merged into `models` above so every
  // picker, lane and lookup sees one list.
  const cloudEndpoints = ref<CloudEndpoint[]>([])
  const currentId = ref<string>('')
  // The last explicit model choice, remembered across sessions. Right after a
  // manager restart the runner list lags the instant DB cloud rows, so the
  // availability fallback used to hand the seat to the oldest cloud pick on
  // every restart (Laguna, every time -). Explicit picks
  // persist; the fallback's choice is provisional (autoValue) and the
  // remembered model reclaims the seat the moment it is available.
  const remembered = ref<string>(uiPreferences.getItem('pk_model') ?? '')
  let autoValue = ''
  let retriedForRemembered = false
  watch(currentId, (v) => {
    if (!v || v === autoValue) return
    autoValue = ''
    remembered.value = v
    uiPreferences.setItem('pk_model', v)
  })
  const loading = ref(false)
  const error = ref<string | null>(null)
  // Selected runner's limits (relayed /api/runners/{port}/server): the context
  // window is the hard ceiling the length slider caps at. 0 = not yet known.
  const maxCtx = ref<number>(0)
  const defaultMaxOutputTokens = ref<number>(0)
  // The SELECTED model's reasoning style (per-model capability, from its
  // dialect): 'effort' (gpt-oss Low/Med/High) | 'toggle' (qwen3.5) | 'none'.
  const reasoningStyle = ref<'effort' | 'toggle' | 'none' | ''>('')
  // PDF attachments: `pdfEnabled` = PDFs accepted at all (always true on
  // current runners - text extraction is built in); `pdfRaster` = pages are
  // rendered to images for a vision model (pdfium present), which is what
  // makes a PDF count as VISUAL input. Plus the raster per-doc page cap for
  // the composer/chip truncation warning.
  const pdfEnabled = ref<boolean>(false)
  const pdfRaster = ref<boolean>(false)
  const pdfMaxPages = ref<number>(0)
  // The manager's version, shown in the header. '' until /api/server answers.
  const serverVersion = ref<string>('')
  // The same version with the commit it was built from - '0.1.0 (ge25c75bf)'.
  // The chip stays short; this is what the tooltip says, and what someone
  // pastes into a bug report.
  const serverBuild = ref<string>('')
  // Per-model advertised server tools (keyed by model id) - chat requests are
  // built from these, so every lane declares exactly its own server's tools.
  const caps = ref<Record<string, ModelCaps>>({})
  /** Model ids whose runner is up but whose model has not attached yet (the
   *  /server body named no model). The composer shows a starting state for
   *  these instead of guessing at controls the model may not have. */
  const capsPending = ref<Set<string>>(new Set())
  /** The manager answered 401: this browser is a network peer of a keyed
   *  manager and must present the API key (the KeyGate view). */
  const needsKey = ref(false)

  /** The runner port serving `id` (fall back to the current selection).
   *  A CLOUD model has no port - deliberately no fallback there, so the
   *  port-keyed features (caps probe, extraction preview, count_tokens)
   *  read honestly as absent instead of asking some other model's server. */
  function portFor(id?: string): number | undefined {
    const want = id || currentId.value
    const hit = models.value.find((m) => m.id === want)
    if (hit) return hit.cloud ? undefined : hit.port
    return models.value.find((m) => m.kind === 'chat' && !m.cloud)?.port
  }

  /** Studio-chat endpoint for a model - the manager relay, never a runner
   *  port. Cloud models chat through the manager's provider seam instead. */
  function responsesUrl(id?: string): string | undefined {
    const want = id || currentId.value
    const hit = models.value.find((m) => m.id === want)
    if (hit?.cloud) return `/api/cloud/${hit.cloud.endpoint}/v1/responses`
    const port = portFor(id)
    return port ? `/api/runners/${port}/v1/responses` : undefined
  }

  /** Where a CLIP goes for this model, same two-sided shape as
   *  [`responsesUrl`]: a local runner through the manager relay, or the
   * manager's provider seam for a cloud speech model.
   *
   *  `undefined` means nothing can transcribe for this id - a local model that
   *  is not running, or a cloud endpoint that has no key. Deliberately not
   *  `portFor`'s "any chat runner" fallback: that exists so a stale id can
   *  still find a tokenizer to ask, and here it would post the clip at
   *  whatever else happened to be running. */
  function transcribeUrl(id?: string): string | undefined {
    const want = id || currentId.value
    const hit = models.value.find((m) => m.id === want)
    if (hit?.cloud) return `/api/cloud/${hit.cloud.endpoint}/v1/audio/transcriptions`
    return hit?.port ? `/api/runners/${hit.port}/v1/audio/transcriptions` : undefined
  }

  /** Whether this model's transcripts arrive as a STREAM. Local runners emit
   *  `transcript.text.*` SSE and type themselves in; no provider does - a
   *  cloud transcription is one JSON body when the whole clip is done, so the
   *  turn sits there until it lands. Read by the send path, and the reason a
   *  cloud lane cannot serve the live/dictation surfaces at all. */
  function transcribeStreams(id?: string): boolean {
    const want = id || currentId.value
    return !models.value.find((m) => m.id === want)?.cloud
  }

  function canLiveTranscribe(id: string): boolean {
    return transcribeStreams(id) && canTranscribe(id) && caps.value[id]?.realtimeTranscription?.supported !== false
  }
  function canEnrichLive(id: string): boolean {
    const cap = caps.value[id]
    return realtimeEnrichment(cap?.realtimeTranscription, cap?.timestampGranularities)
  }

  /** Attachment-costing endpoint (composer chip): the runner's count_tokens
   *  through the manager relay - real extraction + real tokenizer. */
  function countTokensUrl(id?: string): string | undefined {
    const port = portFor(id)
    return port ? `/api/runners/${port}/v1/messages/count_tokens` : undefined
  }

  /** "What the model reads" endpoint: the runner's extraction preview for
   *  one attachment (injection text incl. the metadata block), relayed. */
  function extractUrl(id?: string): string | undefined {
    const port = portFor(id)
    return port ? `/api/runners/${port}/extract` : undefined
  }

  const currentPort = computed(() => portFor(currentId.value))

  /** Fetch (and cache) one model's advertised capabilities. Returns the
   *  cached answer when present; `fresh` re-asks the server (the tool
   *  picker does this on open - a connector scoped onto the endpoint after
   *  page load would otherwise stay invisible until a hard refresh). A
   *  failed fetch keeps the last-good answer, else reads as "no server
   *  tools" rather than blocking the send. */
  /** consecutive failed capsFor attempts per id - cleared on success */
  const capsRetries = new Map<string, number>()

  async function capsFor(id: string, fresh = false): Promise<ModelCaps> {
    const hit = caps.value[id]
    if (hit && !fresh) return hit
    const none: ModelCaps = {
      webSearch: false,
      mcpServers: [],
      taskTags: [],
      timestampGranularities: [],
      include: [],
    }
    // a cloud model has no /api/server to ask, and no paddock server tools
    // to advertise - its vision flag lives on the model entry itself
    if (models.value.find((m) => m.id === id)?.cloud) return none
    const port = portFor(id)
    if (!port) return none
    // Unreachable is the first startup window, before the model-not-attached
    // one: click Start, land on /studio, and the first fetch here dies on a
    // proxy 502 / refused connection - which used to return `none` with no
    // retry, leaving the composer without the endpoint's controls until a
    // full reload. Same capsPending retry as below,
    // bounded (~2.5 min) so a genuinely stopped runner stops being asked.
    const retry = (): ModelCaps => {
      const n = (capsRetries.get(id) ?? 0) + 1
      capsRetries.set(id, n)
      if (n <= 60) {
        capsPending.value = new Set([...capsPending.value, id])
        setTimeout(() => void capsFor(id, true), 2500)
      } else if (capsPending.value.has(id)) {
        const st = new Set(capsPending.value)
        st.delete(id)
        capsPending.value = st
      }
      return hit ?? none
    }
    try {
      const res = await fetch(`/api/runners/${port}/server`)
      if (!res.ok) return retry()
      const body = (await res.json()) as ServerBody & {
        model?: string
        asr?: string
        embedder?: string
        /** a decision model (Laya): it reads and serves nothing else */
        reader?: string
        /** a tabular predictor (Kumo): tables in, predictions out */
        tabular?: { model: string } | null
        diarization?: { model: string } | null
        /** dense prediction (DINOv3) and promptable segmentation (SAM 3) */
        segmenter?: string | null
        masker?: string | null
      }
      const c = parseCaps(body)
      // A runner answers /server before its model attaches, and caching that
      // half-empty answer froze the composer on pre-load capabilities until
      // the user happened to navigate (studio showed the
      // plain-chat controls on a document model). No model named = not
      // settled = not cached - and retry until the endpoint names itself,
      // because the fleet poll's invalidation only fires on a status CHANGE
      // and there is none while the runner just loads. `capsPending` is the
      // composer's honest signal for this exact window: confirmed loading,
      // not merely unfetched.
      if (
        !(
          body.model ||
          body.asr ||
          body.embedder ||
          body.aligner ||
          body.image_model ||
          body.reader ||
          body.tabular ||
          body.diarization ||
          body.segmenter ||
          body.masker
        )
      ) {
        retry()
        return hit ?? c
      }
      capsRetries.delete(id)
      if (capsPending.value.has(id)) {
        const s = new Set(capsPending.value)
        s.delete(id)
        capsPending.value = s
      }
      caps.value = { ...caps.value, [id]: c }
      return c
    } catch {
      return retry()
    }
  }

  /** The advertised read caps of one endpoint, once its /server has been
   *  asked (undefined until then, and for every model that cannot read). */
  function structuredReadFor(id: string): StructuredReadCaps | undefined {
    return caps.value[id]?.structuredRead
  }

  /** Running models that READ - the Reads page exists for these and no
   *  other. A capability gate on what the server says, never on the model
   *  id: a chat model cannot read, and an id heuristic would have to know
   *  every block-diffusion family by name. */
  const readers = computed(() =>
    models.value.filter(
      (m) => (m.kind === 'chat' || m.kind === 'reader') && !m.cloud && caps.value[m.id]?.structuredRead,
    ),
  )
  /** True once every local chat runner has been asked (or is being retried
   *  through capsFor's own loop) - the Reads page's "still looking" state
   *  ends here rather than on a timer. */
  const readersProbed = ref(false)

  /** Ask every local chat runner for its caps once, so `readers` can answer.
   *  The answers land in the same cache the composer reads - nothing is
   *  fetched twice. A runner still loading retries on capsFor's own
   *  schedule and joins `readers` when it names its model. */
  async function probeReaders(): Promise<void> {
    // before the first runner list there is nothing to probe, and saying
    // "probed, none found" now would flash the empty state on every load
    if (!loadedOnce) return
    // a decision model (Laya) reads and does nothing else; a chat model reads
    // only if its server says so
    const local = models.value.filter((m) => (m.kind === 'chat' || m.kind === 'reader') && !m.cloud)
    await Promise.all(local.filter((m) => !caps.value[m.id]).map((m) => capsFor(m.id)))
    readersProbed.value = true
  }

  /** One lane-aware answer to "which thinking control": fetched caps win;
   *  cloud picks derive from their reasoning stamp - OpenAI's gpt-5 and
   *  o-series families always think and take an effort level, every other
   *  reasoning-capable pick is an on/off toggle (the relay translates per
   *  provider); local models fall back to the harmony/id heuristic. The
   *  composer, the compare lanes and the request builder all ask here. */
  function reasoningStyleFor(id: string): 'effort' | 'toggle' | 'none' {
    // A model that cannot chat cannot think about it either. Without this the
    // id heuristic at the bottom answered 'toggle' for a whisper runner, and
    // comparing two speech models then claimed they "differ on thinking" -
    // about a control neither has and the composer does not show.
    if (!canChat(id)) return 'none'
    const hit = caps.value[id]?.reasoning
    if (hit) return hit
    const info = models.value.find((m) => m.id === id)
    if (info?.cloud) {
      if (!info.cloud.reasoning) return 'none'
      const bare = id.replace(/^cloud:[^:]+:/, '')
      return bare.startsWith('gpt-5') || /^o[134]/.test(bare) ? 'effort' : 'toggle'
    }
    return isHarmony(id) ? 'effort' : 'toggle'
  }

  /** The reasoning LADDER for one lane: which rungs it grades at, in the
   *  model's own spelling, and whether it can be turned off. The composer
   *  builds one picker out of this, so a model with both (Qwen3.8: low /
   *  medium / xhigh AND an off position) gets one control with four items
   *  instead of a dropdown next to a switch.
   *
   *  A runner that predates the measured capability sends no levels; the
   *  fallbacks reproduce exactly what it used to show, so an older endpoint
   *  keeps working rather than losing its control. Cloud lanes have no
   *  template to measure, so they keep the style-derived answer: the
   *  always-thinking OpenAI families grade low/medium/high with no off, and
   *  every other reasoning-capable pick is on/off. */
  function reasoningLadderFor(id: string): {
    levels: string[]
    dflt: string
    /** The rung a new chat opens at: the model's LOWEST, whatever it calls it.
     *
     *  Deliberately not `dflt`. `dflt` is the checkpoint's published default and
     *  stays that - honest data, and what an API caller sending no effort gets.
     *  `opens` is the Studio's own choice for a fresh conversation: start cheap
     *  and fast, and let someone who wants more reasoning ask for it.
     *  Expressed as "the lowest rung" rather than the word "low" so it can
     *  never name a level the model does not grade. */
    opens: string
    off: boolean
    preserve: boolean
  } {
    const style = reasoningStyleFor(id)
    const c = caps.value[id]
    if (style === 'none') {
      // A model that grades no effort can still grade preserve_thinking - they
      // are independent template features - so this early return carries it.
      return { levels: [], dflt: '', opens: '', off: false, preserve: c?.reasoningPreserve ?? false }
    }
    const levels = c?.reasoningLevels?.length
      ? c.reasoningLevels
      : style === 'effort'
        ? ['low', 'medium', 'high']
        : []
    const off = c?.reasoningOff ?? style === 'toggle'
    const preserve = c?.reasoningPreserve ?? false
    const dflt =
      c?.reasoningDefault ??
      (levels.includes('medium') ? 'medium' : levels[levels.length - 1] || '')
    return { levels, dflt, opens: levels[0] ?? dflt, off, preserve }
  }

  /** One lane-aware answer to "what's this model's context window": fetched
   *  caps first (capsFor and the limits fetch both cache max_ctx through
   *  parseCaps), a cloud pick's provider-reported ctx next, and the selected
   *  model's `maxCtx` as the warm-up fallback. 0 = not known yet, which the
   *  planners read as "don't trim". */
  function ctxFor(id?: string): number {
    const want = id || currentId.value
    const hit = caps.value[want]?.maxCtx
    if (hit) return hit
    const info = models.value.find((m) => m.id === want)
    if (info?.cloud) return info.cloud.ctx ?? 0
    return want === currentId.value ? maxCtx.value : 0
  }

  /** The most tokens this model will emit in one reply, when the model itself
   *  says so. Cloud only, and deliberately `undefined` rather than a guess
   *  elsewhere: a local GGUF has no output limit of its own (the context is
   *  the ceiling, which `ctxFor` already answers), and inventing a number for
   *  a provider that publishes none would cap replies for no reason.
   *
   *  This exists because "Model maximum" was being computed from the CONTEXT
   *  window alone, so a 1M-context model was asked for ~1M output tokens and
   *  the provider refused the whole send - prompt + output over the window
   *  (deepseek-v4-flash-0731). */
  function outCapFor(id?: string): number | undefined {
    const want = id || currentId.value
    return models.value.find((m) => m.id === want)?.cloud?.maxOut
  }

  /** One lane-aware answer to "can this model see images": the endpoint's
   *  fetched caps win, then the model entry's own flag (cloud picks carry
   *  it; local entries only when selected), then the id heuristic. The
   *  composer gate and the request builder both use this - the local qwen
   *  lane in a compare had its image silently stripped because buildInput
   *  read only the entry flag while the caps knew better. */
  function visionFor(id: string): boolean {
    const hit = caps.value[id]?.vision
    if (hit !== undefined) return hit
    const info = models.value.find((m) => m.id === id)
    return info?.vision ?? isVisionModel(id)
  }

  /** This lane's OCR reading-mode surface, or undefined when its endpoint
   *  never advertised one. Caps only, no id heuristic deliberately: the modes
   *  are request fields the server would 400 on anywhere else, so a guessed
   *  surface would be an offer the send cannot honor. */
  function ocrFor(id: string): OcrCaps | undefined {
    return caps.value[id]?.ocr
  }

  /** One lane-aware answer to "can this model turn speech into text": the
   *  endpoint's fetched caps win, and a transcriber falls back to true while
   *  its caps are still in flight (it is what the runner is - there is
   *  nothing else it could be). A generative ASR model reports `audio` on its
   *  own caps and is `kind: 'chat'`, which is exactly why nothing here reads
   *  the kind. */
  function canTranscribe(id: string): boolean {
    const hit = caps.value[id]?.audio
    if (hit !== undefined) return hit
    return models.value.find((m) => m.id === id)?.kind === 'transcriber'
  }

  /** Whether this model can hold a TEXT conversation. A whisper-family runner
   *  cannot - it serves transcription and nothing else - so the composer
   *  switches to audio input rather than offering a text box that would earn
   *  a refusal. An aligner cannot either: it annotates transcripts, full stop.
   *  Nor can an image model: it answers a prompt with a picture, on its own
   *  page - and a tabular predictor takes rows, never a turn (the Tables
   *  page), nor do the two segmentation roles (pictures in, rasters or masks
   *  out - SAM 3 has the Masks page). */
  function canChat(id: string): boolean {
    const kind = models.value.find((m) => m.id === id)?.kind
    return (
      kind !== 'transcriber' &&
      kind !== 'aligner' &&
      kind !== 'image' &&
      kind !== 'tabular' &&
      kind !== 'diarizer' &&
      kind !== 'segmenter' &&
      kind !== 'masker'
    )
  }

  /** Whether this model makes pictures from a prompt: the endpoint's fetched
   *  caps win, and an image runner falls back to true while its caps are
   *  still in flight (it is what the runner is). A capability question like
   *  `canTranscribe`, so a future model that both chats and draws joins
   *  either panel. */
  function canImagine(id: string): boolean {
    const hit = caps.value[id]?.imageGeneration
    if (hit !== undefined) return true
    return models.value.find((m) => m.id === id)?.kind === 'image'
  }

  /** Where a PROMPT goes to become a picture, same two-sided shape as
   *  [`transcribeUrl`]: a local runner through the manager relay. Cloud image
   *  models are not wired yet - the seam is here for them. `undefined` means
   *  nothing serves this id right now. */
  function imageUrl(id?: string): string | undefined {
    const want = id || currentId.value
    const hit = models.value.find((m) => m.id === want)
    if (hit?.cloud) return undefined
    return hit?.port ? `/api/runners/${hit.port}/v1/images/generations` : undefined
  }

  /** Where a picture goes to be EDITED (reference pictures + an
   *  instruction), beside [`imageUrl`]; undefined when the endpoint does
   *  not edit (no vision tower) or nothing serves the id. */
  function imageEditUrl(id?: string): string | undefined {
    const want = id || currentId.value
    const hit = models.value.find((m) => m.id === want)
    if (hit?.cloud || !hit?.port) return undefined
    if (!caps.value[want]?.imageGeneration?.edit) return undefined
    return `/api/runners/${hit.port}/v1/images/edits`
  }

  /** The running forced-aligner lane, if the fleet has one - what the
   *  enrichment pass posts to after a transcription lands without word times.
   *  Local runners only: no cloud provider offers alignment. */
  function alignerLane(): { id: string; port: number; url: string } | undefined {
    const hit = models.value.find((m) => m.kind === 'aligner' && m.status === 'ok' && m.port)
    if (!hit?.port) return undefined
    return { id: hit.id, port: hit.port, url: `/api/runners/${hit.port}/v1/audio/alignments` }
  }

  /** Whether this endpoint can answer with segment times - the gate on asking
   *  for them at all (see TranscribeOpts.timestamps). */
  function canTimeSegments(id: string): boolean {
    return (caps.value[id]?.timestampGranularities ?? []).includes('segment')
  }

  /** Whether this endpoint can time individual WORDS. Its own question rather
   *  than a finer setting of the one above, and it goes both ways: whisper
   *  pays a second pass over every 30 s window for these, while
   *  granite-speech-plus times words and cannot cut segments at all. */
  function canTimeWords(id: string): boolean {
    return (caps.value[id]?.timestampGranularities ?? []).includes('word')
  }

  /** Whether this endpoint can say how sure it was of each word. Separate from
   *  the times: a generative ASR model has the logprobs and no timestamps, so
   *  gating confidence on `canTimeSegments` is what left Qwen3-ASR's
   * transcripts unmarked. */
  function canWordConfidence(id: string): boolean {
    return (caps.value[id]?.include ?? []).includes('logprobs')
  }

  /** One lane-aware answer to "can this turn search the web": a local
   *  endpoint advertises it in fetched caps (the runner's own provider
   *  integration); an OpenRouter cloud pick carries it on the model entry
   *  (every model there can, via the endpoint's web plugin). */
  function webSearchFor(id: string): boolean {
    if (caps.value[id]?.webSearch) return true
    return models.value.find((m) => m.id === id)?.cloud?.webSearch === true
  }

  /** The lane's speculation mechanism in words ("MTP", "DFlash1", "off"),
   *  from the manager's spawn resolution; undefined = nothing to say (cloud
   *  picks, non-speculative models). Shown beside the model name everywhere
   *  it appears, and stamped into each turn's run record. */
  function specFor(id: string): string | undefined {
    return models.value.find((m) => m.id === id)?.spec ?? undefined
  }

  /** Can this lane ENFORCE a thinking budget? Fetched caps only - absent
   *  (cloud picks, old runners) reads false, so the control is never drawn
   *  for a lane that would refuse or ignore it. */
  function thinkingBudgetFor(id: string): boolean {
    return caps.value[id]?.thinkingBudget === true
  }

  /** Drop cached capabilities (a redeploy may have changed the config). */
  function invalidateCaps(id?: string): void {
    if (id) {
      const { [id]: _, ...rest } = caps.value
      caps.value = rest
    } else {
      caps.value = {}
    }
  }

  // Both the shell (header model + version) and ChatView refresh on mount;
  // share the in-flight fetch so they don't both hit /api/runners.
  let refreshing: Promise<void> | null = null

  /** Apply a server-pushed /api/runners row set (SSE 'fleet' event): the
   *  exact mapping doRefresh uses for its fetched rows - same enrichment
   *  preservation, same caps restaling - with the current cloud entries kept
   *  (cloud changes arrive on the slow reconcile poll). One code path per
   *  concern: the push is a faster TRIGGER, never a second mapper. */
  function integrateRunnerRows(rows: RunnerRow[]): void {
    const prevById = new Map(models.value.map((m) => [m.id, m]))
    const local: ModelInfo[] = rows
      .filter((r) => rowId(r))
      .map((r) => rowModel(r, prevById.get(rowId(r) as string)))
    const cloud = models.value.filter((m) => m.cloud)
    const next = [...local, ...cloud]
    const sig = (m: { port?: number; status?: string }) => `${m.port ?? ''}:${m.status ?? ''}`
    const before = new Map(models.value.map((m) => [m.id, sig(m)]))
    const restale = next.filter((m) => before.get(m.id) !== sig(m))
    if (restale.length) {
      const fresh = { ...caps.value }
      for (const m of restale) delete fresh[m.id]
      caps.value = fresh
    }
    models.value = next
  }

  async function refresh(): Promise<void> {
    if (refreshing) return refreshing
    refreshing = doRefresh()
    try {
      await refreshing
    } finally {
      refreshing = null
    }
  }

  // SWR semantics: `loading` means the first load only. The shell revalidates
  // every few seconds (a started/stopped server must show up on its own -
  // and a background poll flipping `loading` would blink
  // every empty-state that renders behind `!models.loading`.
  let loadedOnce = false

  async function doRefresh(): Promise<void> {
    if (!loadedOnce) loading.value = true
    error.value = null
    try {
      // runners and cloud endpoints together; a cloud fetch failure must not
      // take the local fleet down with it (and vice versa - but no runners
      // is an error worth surfacing, no cloud rows is just an empty list)
      const [res, cloudRes] = await Promise.all([
        fetch('/api/runners'),
        fetch('/api/cloud').catch(() => null),
      ])
      // A network browser hitting a keyed manager: every /api call 401s and
      // the whole Studio renders empty. The key gate takes
      // over; a successful login sets the session cookie and reloads.
      needsKey.value = res.status === 401
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const rows = (await res.json()) as RunnerRow[]
      if (cloudRes?.ok) {
        cloudEndpoints.value = (await cloudRes.json()) as CloudEndpoint[]
      }
      // A transcriber row carries only `asr` - dropping rows without
      // model/embedder made every running whisper INVISIBLE to the Studio
      // (since fixed): it was started, it was serving, and no page could
      // see it.
      // The poll rebuild must not WIPE what a caps fetch patched on: `vision`
      // only exists on a local entry because fetchLimits wrote it there, and
      // rebuilding from the bare /api/runners rows unmounted the header's eye
      // every tick - which read as blinking whenever caps churned.
      // Same for 'unreachable': the supervisor's 2s health call
      // misses under generation load, and letting that flap through the
      // status both blinked the UI and (below) dropped caps for an endpoint
      // whose tools never changed - smooth it over with the last real status.
      const prevById = new Map(models.value.map((m) => [m.id, m]))
      const local: ModelInfo[] = rows
        .filter((r) => rowId(r))
        .map((r) => rowModel(r, prevById.get(rowId(r) as string)))
      // Cloud models join the one list. Keyless endpoints require an explicit
      // no-auth choice (custom local servers); absence of a key alone is not
      // permission to send. Vision
      // defaults to true: silently not sending an image would be the worse
      // failure, and a text-only provider model refuses with its own words.
      const cloud: ModelInfo[] = cloudEndpoints.value
        .filter((ep) => ep.hasKey || ep.allowUnauthenticated === true)
        .flatMap((ep) =>
          (ep.models ?? []).map((cm) => {
            // bare native ids (o3-mini, claude-sonnet-5) name no maker; the
            // endpoint KIND does
            // OpenRouter bakes the maker into the display ("Qwen: Qwen3.5-9B")
            // while the vendor mark next to it says the same thing - "Qwen
            // twice" . The mark carries the brand, so the
            // name sheds the prefix here and every view downstream agrees.
            const identity = cloudModelIdentity(cm.id, cm.display, ep.kind, cm.provider)
            return {
              // a provider-pinned pick is its own pickable model: the @suffix
              // rides the wire model, where the relay turns it into
              // OpenRouter's provider-routing preference
              id: `cloud:${ep.id}:${cm.id}${cm.provider ? `@${cm.provider}` : ''}`,
              ownedBy: 'cloud',
              display: identity.name,
              vendor: identity.vendor,
              vision: cm.vision ?? true,
              // A cloud speech model is a TRANSCRIBER, exactly like a
              // whisper-family runner: `takesTurns` still seats it in a lane,
              // the composer switches to audio, and the send path routes the
              // clip to /audio/transcriptions instead of the chat wire. Every
              // capability gate below already falls back correctly for it -
              // `canChat` reads the kind, and the timing/confidence gates read
              // fetched caps, which a cloud model has none of.
              kind: cm.asr ? ('transcriber' as const) : ('chat' as const),
              status: ep.credentialReady === false ? 'credential-unavailable' : 'ok',
              cloud: {
                endpoint: ep.id,
                endpointName: ep.name,
                // OpenRouter offers web search for every model (its `web`
                // plugin, Exa-powered) - the relay translates the Studio's
                // web_search tool into it. Other providers wait for their
                // own adapters.
                webSearch: ep.kind === 'openai-compat' && ep.baseUrl.includes('openrouter.ai'),
                ctx: cm.ctx,
                maxOut: cm.maxOut,
                reasoning: cm.reasoning,
              },
            }
          }),
        )
      // Drop cached caps for anything that just appeared or changed port.
      // Caps are fetched from the RUNNER, so a model asked about before its
      // runner was up cached an empty answer - and the composer then showed no
      // tools until a full page reload, even though the toast said it was
      // running. The background revalidate now clears the
      // stale entry, so the next capsFor asks the live endpoint.
      const next = [...local, ...cloud]
      // Key on STATUS too, not just id+port. A runner's row appears before it
      // can answer - "loading", or "unreachable" while it boots - and becomes
      // ready without its id or port changing, so a signature of id+port alone
      // never fires on the transition that matters: the first cut did exactly
      // that, and the composer kept the empty caps it had cached from before
      // the runner answered (the maintainer, twice).
      //
      // (This list is built from /api/runners + /api/cloud only - it never
      // reads /api/servers - so a STOPPED local endpoint is absent from the
      // chat picker by construction, however it is listed under Manage. An
      // earlier version of this comment claimed the opposite; whether a stopped
      // model should be offered here and started on selection is a product
      // question, not a caching one.)
      const sig = (m: { port?: number; status?: string }) => `${m.port ?? ''}:${m.status ?? ''}`
      const before = new Map(models.value.map((m) => [m.id, sig(m)]))
      const restale = next.filter((m) => before.get(m.id) !== sig(m))
      if (restale.length) {
        const fresh = { ...caps.value }
        for (const m of restale) delete fresh[m.id]
        caps.value = fresh
      }
      models.value = next
      // Dropping the entry is not enough: `activeCaps` is a plain computed over
      // this cache and capsFor only runs when the model ID changes, so an
      // invalidated entry just stayed empty until a reload - which is exactly
      // what kept happening. Re-fetch what we invalidated so the composer
      // repopulates on its own the moment the runner answers.
      for (const m of restale) {
        if (!m.cloud) void capsFor(m.id, true)
      }
      // takesTurns, not kind==='chat' : a transcriber ANSWERS a
      // user turn, so it holds the seat like any other model - the composer
      // switches to audio input and the send path routes it to
      // /v1/audio/transcriptions. An ENCODER still never gets the seat: a
      // vector is not a reply.
      const turnHas = (id: string) =>
        !!id && models.value.some((m) => m.id === id && takesTurns(m.kind))
      if (turnHas(remembered.value) && (!currentId.value || currentId.value === autoValue)) {
        // the remembered choice takes (or takes BACK) the seat - unless the
        // user explicitly picked something else this session
        currentId.value = remembered.value
      } else if (!currentId.value || !models.value.some((m) => m.id === currentId.value)) {
        // CHAT models only: falling back to models[0] used to select an
        // encoder when nothing else ran, which rendered the header's chat
        // picker with zero options - the confusing empty dropdown.
        // An encoder-only fleet leaves no current chat model,
        // and the header points at the Playground instead. Local models come
        // first: they are what a local install is FOR, and the cloud rows merely
        // loaded faster.
        // A chat model still wins the empty seat over a transcriber: it can
        // do everything the transcriber can be asked for and more, and a box
        // running both should land in a chat, not in audio mode. A
        // transcriber-only fleet takes it rather than leaving the Studio with
        // no model at all - which is what sent people to a separate page.
        currentId.value =
          models.value.find((m) => m.kind === 'chat' && !m.cloud)?.id ??
          models.value.find((m) => m.kind === 'chat')?.id ??
          models.value.find((m) => takesTurns(m.kind))?.id ??
          ''
        autoValue = currentId.value
        // The chat page refreshes once on mount, so a remembered model that
        // simply has not REPORTED yet (runner adoption takes a moment) gets
        // one short retry instead of losing the seat until the next refresh.
        if (remembered.value && !turnHas(remembered.value) && !retriedForRemembered) {
          retriedForRemembered = true
          setTimeout(() => void refresh(), 2500)
        }
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
      loadedOnce = true
    }
    void fetchLimits()
  }

  /** Best-effort fetch of the selected runner's model card + manager version. */
  async function fetchLimits(): Promise<void> {
    try {
      const res = await fetch('/api/server')
      if (res.ok) {
        const body = (await res.json()) as ServerBody
        if (body.version) serverVersion.value = body.version
        // Falls back to the bare version: a build made outside a git checkout
        // has no commit to name, and the manager omits `build` accordingly.
        if (body.version) serverBuild.value = body.build || body.version
      }
    } catch {
      /* header just shows no version */
    }
    const cur = models.value.find((x) => x.id === currentId.value)
    if (cur?.cloud) {
      // no /api/server to ask a provider - the provider's own context length
      // when its list reported one, unknown otherwise. PDFs are read
      // natively (as pages) on the big providers. The reasoning control is
      // the lane-aware resolution: effort for the always-thinking OpenAI
      // families, the on/off toggle for everything else capable (the relay
      // translates per provider - OpenRouter reasoning{enabled}, Anthropic
      // extended thinking), none otherwise.
      maxCtx.value = cur.cloud.ctx ?? 0
      reasoningStyle.value = reasoningStyleFor(currentId.value)
      pdfEnabled.value = true
      pdfRaster.value = true
      pdfMaxPages.value = 0
      return
    }
    const port = portFor(currentId.value)
    if (!port) return
    try {
      const res = await fetch(`/api/runners/${port}/server`)
      if (!res.ok) return
      const body = (await res.json()) as ServerBody
      if (body.max_ctx) maxCtx.value = body.max_ctx
      if (body.default_max_output_tokens) defaultMaxOutputTokens.value = body.default_max_output_tokens
      if (body.reasoning) reasoningStyle.value = body.reasoning as 'effort' | 'toggle' | 'none'
      pdfEnabled.value = body.pdf?.enabled ?? false
      // Older runners predate `raster` and used `enabled` for exactly that
      // (pdfium availability), so it doubles as the fallback.
      pdfRaster.value = body.pdf?.raster ?? body.pdf?.enabled ?? false
      if (body.pdf?.max_pages) pdfMaxPages.value = body.pdf.max_pages
      const m = models.value.find((x) => x.port === port)
      if (m && body.vision !== undefined) m.vision = body.vision ?? undefined
      // this response carries the endpoint's whole capability set - cache it
      // through the same parse capsFor uses, or whichever call lands last wins
      // with a different set of fields
      if (m) caps.value = { ...caps.value, [m.id]: parseCaps(body) }
    } catch {
      /* limits stay unknown; the slider falls back to a safe range */
    }
  }

  return {
    models,
    cloudEndpoints,
    currentId,
    currentPort,
    loading,
    error,
    maxCtx,
    defaultMaxOutputTokens,
    reasoningStyle,
    pdfEnabled,
    pdfRaster,
    pdfMaxPages,
    serverVersion,
    serverBuild,
    caps,
    capsPending,
    needsKey,
    capsFor,
    ctxFor,
    outCapFor,
    transcribeUrl,
    transcribeStreams,
    canLiveTranscribe,
    canEnrichLive,
    visionFor,
    ocrFor,
    canTranscribe,
    canChat,
    canImagine,
    imageUrl,
    imageEditUrl,
    alignerLane,
    canTimeSegments,
    canTimeWords,
    canWordConfidence,
    reasoningStyleFor,
    integrateRunnerRows,
    reasoningLadderFor,
    thinkingBudgetFor,
    webSearchFor,
    specFor,
    invalidateCaps,
    refresh,
    fetchLimits,
    portFor,
    responsesUrl,
    countTokensUrl,
    extractUrl,
    structuredReadFor,
    readers,
    readersProbed,
    probeReaders,
  }
})
