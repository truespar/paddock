// The streaming chat client, on the OpenAI **Responses API** (/v1/responses).
//
// We POST { model, input, instructions, stream:true, ... } and read the typed
// Responses event stream (response.output_text.delta / .reasoning_text.delta /
// .completed / .failed) out of the same body, dispatching on each event's
// `type`. Delta application is throttled to one animation frame so a fast
// stream never re-renders Markdown per token (an O(n^2) freeze).

import { computed, shallowReactive } from 'vue'
import { uuid } from '@/lib/uuid'
import type {
  AudioPart,
  ContentPart,
  Conversation,
  DocPage,
  FilePart,
  ImagePart,
  McpCall,
  McpCallStatus,
  Message,
  ToolSelection,
  WebSearchCall,
  WebSearchStatus,
} from '@/types/chat'
import { messageText } from '@/types/chat'
import {
  DEGENERATION_THRESHOLD,
  degenerationRatio,
  ocrMetaFromWire,
  parseRegionsLive,
  wordsFromLogprobs,
  type LogprobEntry,
} from '@/lib/ocr'
import { pdfEngine } from '@/lib/pdf'
import { alignClip, alignmentRefused, mergeWordTimes } from '@/lib/align'
import { transcribeClip } from '@/lib/transcribe'
import { fleetLabel } from '@/lib/model-name'
import { isHarmony } from '@/lib/model-caps'
import { isTaskTurn } from '@/lib/tasks'
import {
  planContext,
  promptTokensFrom,
  replyReserve,
  serverCompactThreshold,
  serverCompactionValid,
  trimIndex,
  windowRemaining,
} from '@/lib/tokens'
import { maybeCompact } from '@/lib/compact'
import { activeMessages, activeSteps, stepAnchor, tipId } from '@/lib/tree'
import { readSse } from '@/lib/sse'
import {
  docContexts,
  isPdfPart,
  isRasterDoc,
  pageImages,
  pageRangeBounds,
  rasterContext,
} from '@/lib/docrun'
import { holdReload } from '@/lib/reload'
import { attachmentsApi, forensicsApi } from '@/lib/api'
import { useChatStore } from '@/stores/chat'
import { useArtifactsStore } from '@/stores/artifacts'
import { useGraphsStore } from '@/stores/graphs'
import { takesTurns, useModelsStore, type ModelCaps } from '@/stores/models'
import { useSettingsStore } from '@/stores/settings'
import { useTelemetryStore } from '@/stores/telemetry'
import { usePromptsStore } from '@/stores/prompts'
import { useToastsStore } from '@/stores/toasts'
import { useConnectorsStore, type Connector } from '@/stores/connectors'

// Active streams, keyed by CONVERSATION. Within one chat nothing changes: a
// single-model send runs one stream, a compare send runs one per lane, and
// "busy" / "stop" still mean all of them together.
//
// What changed is that they are no longer app-wide. A chat
// left mid-answer made every other chat look busy: a brand-new chat opened
// with a Stop button, refused to send, and pressing that Stop aborted the turn
// you had walked away from - the one chat you could not see. Busy is a
// property of a conversation, so it is stored as one.
//
// SHALLOW deliberately. Busy is a boolean, and it only ever flips when a
// conversation's entry is added or removed - which is a mutation of the MAP,
// which a shallow proxy tracks. Lane two joining an already-busy chat changes
// nothing anyone reads. Deep `reactive` would additionally hand back proxied
// Sets, and the only thing in them is an AbortController: a host object whose
// methods object to being called through a proxy. Nothing needs that, so it
// does not happen.
const aborts = shallowReactive(new Map<string, Set<AbortController>>())

function beginStream(convId: string, c: AbortController): void {
  const live = aborts.get(convId)
  if (live) live.add(c)
  // `set` and not a mutation: the entry appearing is the busy edge, and a
  // shallow proxy only sees the map.
  else aborts.set(convId, new Set([c]))
  // An answer arriving into this tab is exactly the work a reload would throw
  // away, so the shell's silent build-swap waits for it.
  holdReload('stream', true)
}

function endStream(convId: string, c: AbortController): void {
  const live = aborts.get(convId)
  if (!live) return
  live.delete(c)
  if (live.size) return
  aborts.delete(convId)
  holdReload('stream', aborts.size > 0)
  // Last stream of a chat you are not looking at: say so, because otherwise
  // the only way to learn a background answer landed is to go and check.
  const chat = useChatStore()
  if (chat.activeId === convId) return
  const conv = chat.conversations.find((x) => x.id === convId)
  if (!conv) return
  const path = activeMessages(conv)
  const failed = path[path.length - 1]?.error
  useToastsStore().push({
    tone: failed ? 'bad' : 'info',
    title: conv.title,
    description: failed ? 'This chat stopped with an error.' : 'Finished answering.',
    to: { name: 'chat', params: { id: convId } },
  })
}

/** Is this conversation mid-answer - the one question every caller actually
 *  has. Exported because the sidebar asks it of rows that are not open. */
export function conversationBusy(id: string | null | undefined): boolean {
  return !!id && (aborts.get(id)?.size ?? 0) > 0
}

function uid(): string {
  return uuid()
}

/** A message's content for a Responses `input` item: a plain string when it's
 *  all text (or when the model can't see attachments), else a typed content
 *  array with `input_text` / `input_image` / `input_file` parts. Images need
 *  vision; PDFs go to every model - the server rasterizes pages for a vision
 *  tower and extracts the text layer (sift) for anything else. `fileData` maps
 *  a file part's attachmentId -> its `data:` URI (fetched by the caller's
 *  pre-pass, since PDF bytes don't live in the doc). */
function toApiContent(m: Message, vision: boolean, fileData: Map<string, string>): unknown {
  const hasAttachment = m.content.some((p) => p.type === 'image' || p.type === 'file')
  // An audio clip reaches a CHAT model as a note, never as bytes - and that is
  // a hard constraint, not a simplification: /v1/responses refuses audio parts
  // outright (the runner serves them on /v1/chat/completions and
  // /v1/audio/transcriptions only), so sending one would 400 the whole turn.
  // Nothing is lost: the transcript is already an ordinary assistant message
  // in this history, which is exactly what "summarise that" reads. The note
  // only says a clip was there, so the model doesn't answer as if the user
  // sent an empty message.
  const clipLead = m.content
    .filter((p): p is AudioPart => p.type === 'audio')
    .map((p) => `[audio: ${p.name || 'recording'}]`)
    .join(' ')
  // A graph rides the same way as audio and for the same reason: never bytes.
  // The model reaches it through graph_query; the note only records that the
  // attachment happened in this turn.
  const graphLead = m.content
    .filter((p) => p.type === 'graph')
    .map((p) => `[attached graph database: ${(p as { name: string }).name} - query it with graph_query]`)
    .join(' ')
  const lead = [clipLead, graphLead].filter(Boolean).join(' ')
  const said = messageText(m)
  const plain = lead ? (said ? `${lead}\n${said}` : lead) : said
  if (!hasAttachment) return plain

  const parts: unknown[] = []
  if (lead) parts.push({ type: 'input_text', text: lead })
  for (const p of m.content) {
    if (p.type === 'text') {
      if (p.text) parts.push({ type: 'input_text', text: p.text })
    } else if (p.type === 'image' && vision && !p.unreadable) {
      // `!p.unreadable`: a HEIC is dropped here rather than sent and refused.
      // The guard is on the BUILDER rather than only on the byte pre-pass,
      // because `modelUrl` below is an inline fallback holding the raw file -
      // skipping the fetch alone would still put the undecodable bytes on the
      // wire by the back door, and one photo would fail the whole message.
      // The composer already said so in red before this point.
      // Send the ORIGINAL bytes (attachments table, fetched by the pre-pass):
      // the server reads their EXIF metadata/orientation and fits them to its
      // own budget from `detail`. `modelUrl` is the store-unavailable inline
      // fallback; `dataUrl` the pre-attachments-v2 legacy field - both are
      // metadata-less re-encodes, which is all those messages ever had.
      const src = fileData.get(p.attachmentId) ?? p.modelUrl ?? p.dataUrl
      if (src) {
        const part: Record<string, unknown> = {
          type: 'input_image',
          image_url: src,
          detail: p.detail ?? 'auto',
        }
        // multi-page image (TIFF): this file's own page range, part-level
        if (p.pageRange) part.pages = p.pageRange
        parts.push(part)
      }
    } else if (p.type === 'file') {
      // PDF -> input_file for any model (page images on vision+pdfium, sift
      // text extraction otherwise - the server routes). Bytes come from the
      // pre-pass fetch (keyed by attachmentId), never the doc. The file's own
      // route/cap choices ride as part-level fields.
      const src = fileData.get(p.attachmentId)
      if (src) {
        const part: Record<string, unknown> = { type: 'input_file', filename: p.name, file_data: src }
        if (p.pdfMode) part.pdf_mode = p.pdfMode
        if (p.pageRange) part.pages = p.pageRange
        parts.push(part)
      }
    }
  }
  // Nothing sendable survived (non-vision model, or attachment-only text) -
  // fall back to a plain string so the user's words still go through.
  if (parts.length === 0 || parts.every((x) => (x as { type: string }).type === 'input_text')) {
    return plain
  }
  return parts
}

/** Fetch the bytes of every `file` AND `image` part in the conversation (from
 *  the attachments table) as `data:` URIs, so the sync body-builders can inline
 *  them into `input_file` / `input_image` parts. Images ride this path so the
 *  server sees ORIGINAL bytes (EXIF metadata + orientation intact). Failures
 *  are skipped (the part falls back to its inline copy, or drops out). */
async function resolveFileData(conv: Conversation): Promise<Map<string, string>> {
  const ids = new Set<string>()
  // Only the branch that will actually be sent - resolving an abandoned
  // branch's attachments would refetch photos nothing is going to reference.
  for (const m of activeMessages(conv)) {
    for (const p of m.content) {
      // An unreadable image is skipped: the builder will not send it either
      // way, and this is the difference between fetching a 12 MP photo out of
      // the store on every turn of the conversation and not.
      if (p.type === 'image' && p.unreadable) continue
      if ((p.type === 'file' || p.type === 'image') && p.attachmentId) ids.add(p.attachmentId)
    }
  }
  const map = new Map<string, string>()
  await Promise.all(
    [...ids].map(async (id) => {
      try {
        const r = await fetch(attachmentsApi.url(id))
        if (!r.ok) return
        const dataUri = await blobToDataUrl(await r.blob())
        map.set(id, dataUri)
      } catch {
        /* skip a byte-fetch that failed */
      }
    }),
  )
  return map
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(r.result as string)
    r.onerror = () => reject(r.error ?? new Error('read failed'))
    r.readAsDataURL(blob)
  })
}

/** Build the Responses `input`: the conversation as message items, excluding
 *  the assistant placeholder we're currently filling and any failed turns. */
function buildInput(
  conv: Conversation,
  modelId: string,
  exclude: Message,
  cont: boolean,
  trimFrom: number,
  fileData: Map<string, string>,
): unknown[] {
  // The lane's own capability, same resolution the composer gate uses
  // (fetched caps > model entry > id heuristic). run() awaits capsFor()
  // before building, so the caps answer is warm here.
  const store = useModelsStore()
  const vision = store.visionFor(modelId)
  // Preserving prior thinking is both the chat's choice and the lane's
  // capability: a model whose template ignores preserve_thinking would just be
  // sent context it cannot act on. Continue mode never keeps it - that turn is
  // resuming an answer, not starting a new thought.
  const keepThinking =
    !cont && conv.params.preserveThinking === true && store.reasoningLadderFor(modelId).preserve
  const items: unknown[] = []
  // The BRANCH on SCREEN, not every branch stored. `trimFrom` indexes this
  // same array (lib/tokens.ts counts the identical path), and sending
  // `conv.messages` would hand the model the alternatives the user rejected.
  activeMessages(conv).forEach((m, idx) => {
    // sliding-window trim: drop the oldest messages that don't fit the context.
    if (idx < trimFrom) return
    // In continue mode the partial assistant is kept as context; otherwise the
    // placeholder we're filling is excluded.
    if (!cont && m === exclude) return
    // Own-history-only: a compare lane's answer is context for its model's
    // later turns, never for the other lanes (each model continues its own
    // thread; ungrouped turns - the linear parts of the chat - go to all).
    if (m.role === 'assistant' && m.group && m.model !== modelId) return
    // a lane-scoped user turn (auto-repair reports) is the other lanes' ghost
    if (m.role === 'user' && m.lane && m.lane !== modelId) return
    if (m.role === 'assistant' && (m.error || m.content.every((p) => p.type === 'text' && !p.text))) {
      return
    }
    // A past answer's chain of thought, as the Responses API's own `reasoning`
    // item - which precedes the message it belongs to, the order this server
    // emits and accepts. Only when the chat asked for it AND the lane's
    // template grades the choice; `Message.reasoning` is stored on every turn
    // regardless, so turning this on needs no new capture.
    if (keepThinking && m.role === 'assistant' && m.reasoning && !m.error) {
      items.push({
        type: 'reasoning',
        summary: [],
        content: [{ type: 'reasoning_text', text: m.reasoning }],
      })
    }
    items.push({ type: 'message', role: m.role, content: toApiContent(m, vision, fileData) })
  })
  if (cont) {
    items.push({
      type: 'message',
      role: 'user',
      content: 'Continue your previous reply from exactly where it left off. Do not repeat any text.',
    })
  }
  return items
}

/** The chat's switched-on connectors, resolved against the library (a deleted
 *  connector simply stops riding - its id in the chat is inert, not an error).
 *  System connectors are excluded: they live in every server's own tool
 *  registry now, so the label arrives as a server tool - sending the inline
 *  spec too would declare the same label twice in one request. */
function activeConnectors(conv: Conversation): Connector[] {
  const ids = conv.connectorIds ?? []
  if (!ids.length) return []
  const lib = useConnectorsStore()
  return ids.map((id) => lib.byId(id)).filter((c): c is Connector => !!c && !c.system)
}

/** One tool source this turn offers the model. Label only = a server tool
 *  (the runner resolves url/headers from its own registry); with `url` it's
 *  an inline connector. `allowed` narrows to named tools - it rides as the
 *  request's `allowed_tools`, which the runner honors over the registry's. */
export interface ToolSpec {
  label: string
  url?: string
  headers?: Record<string, string>
  allowed?: string[]
  /** Scope = trust: a connector the user SCOPED onto models executes
   *  without the approval gate on every lane; one merely armed per-chat
   *  keeps it (the try-out tier). */
  trusted?: boolean
}

/** The label the artifact tools travel under, in the picker and on the wire. */
export const ARTIFACTS_LABEL = 'artifacts'

/** The manager's own first-party MCP server. It rides INLINE per
 *  request rather than sitting in servers/<port>.toml: an external caller has
 *  no side panel, so an artifact it created would be a reference to something
 *  it can never see. `trusted` because it is ours and touches only this box's
 *  database - no approval gate.
 *
 *  The URL is LOOPBACK on this page's port, never the page's own origin: its
 *  consumer is the RUNNER, which shares a box with the manager by
 *  architecture. A browser on another machine would otherwise stamp the LAN
 *  address in here, and the runner's dial-back then hits the manager as a
 *  non-loopback peer - keyed, 401, and the MCP initialize decode fails.
 *  (Behind a reverse proxy that remaps the port, this is the line to
 *  revisit.) The conversation rides as a header, never as a tool argument,
 *  so the model cannot reach another chat's artifacts by naming one. */
function artifactSpec(conversationId: string, modelId: string): ToolSpec {
  return {
    label: ARTIFACTS_LABEL,
    url: `http://127.0.0.1:${window.location.port || '80'}/api/mcp/artifacts`,
    // The model rides too: a compare turn runs several lanes against one
    // conversation, and without this their artifacts arrive indistinguishable
    // - which is exactly what a side-by-side comparison needs to separate.
    headers: { 'x-paddock-conversation': conversationId, 'x-paddock-model': modelId },
    trusted: true,
  }
}

/** graph_query, same loopback lane as artifacts - attached only when the
 *  conversation actually has a live, ready graph (the store gates on
 *  readiness, so a session still loading simply skips the tool this turn). */
function graphSpec(conversationId: string, modelId: string): ToolSpec {
  return {
    label: 'graph',
    url: `http://127.0.0.1:${window.location.port || '80'}/api/mcp/graph`,
    // The model rides for the same reason as on artifactSpec: a compare turn
    // funnels several lanes through one bridge, and the pane's query history
    // needs to say who asked what.
    headers: { 'x-paddock-conversation': conversationId, 'x-paddock-model': modelId },
    trusted: true,
  }
}

/** The chat's tool selection, with the legacy all-or-nothing switch mapped
 *  in: old chats that flipped tools off read as an empty custom pick. */
export function toolSelection(conv: Conversation): ToolSelection {
  if (conv.toolSelection) return conv.toolSelection
  if (conv.toolsEnabled === false) return { mode: 'custom', picks: [] }
  return { mode: 'all' }
}

/** The lanes a compare send fans out to: the chat's compare set, kept to
 *  models that are actually running (a stopped one drops out honestly) and
 *  that AGREE on what they can be asked.
 *
 *  The agreement rule is the whole reason compare works for speech: one user
 *  turn goes to every lane, so every lane must be able to TAKE that turn.
 *  The test is a shared input mode, not a shared kind - whisper beside
 *  Qwen3-ASR is the comparison this feature exists for, and Qwen3-ASR is a
 *  generative model that also transcribes. The composer refuses an unshared
 *  pairing by name; this is the backstop for a set armed before a model
 *  changed underneath it. */
function runningLanes(conv: Conversation, models: ReturnType<typeof useModelsStore>): string[] {
  const set = conv.compareModels ?? []
  if (set.length < 2) return []
  const running = new Set(
    models.models.filter((m) => takesTurns(m.kind) && m.status === 'ok').map((m) => m.id),
  )
  const live = set.filter((id) => running.has(id))
  if (live.length < 2) return []
  if (live.every((id) => models.canChat(id)) || live.every((id) => models.canTranscribe(id))) {
    return live
  }
  // No input they all take. Keep the group the first lane belongs to, and say
  // so - dropping a lane is loud, never silent.
  const byAudio = models.canTranscribe(live[0])
  const kept = live.filter((id) => (byAudio ? models.canTranscribe(id) : models.canChat(id)))
  console.warn(
    'compare lanes share no input; dropped',
    live.filter((id) => !kept.includes(id)),
  )
  return kept.length >= 2 ? kept : []
}

/** Compare fairness + safety: every lane plans against the same window - the
 *  smallest among the lanes - so no lane overflows its server and every model
 *  answers from the same visible history. 0 = no lane's window is known yet. */
function minLaneCtx(conv: Conversation, models: ReturnType<typeof useModelsStore>): number {
  const ctxs = runningLanes(conv, models)
    .map((id) => models.ctxFor(id))
    .filter((n) => n > 0)
  return ctxs.length ? Math.min(...ctxs) : 0
}

/** How one request fits the context window. `serverThreshold` set = the
 *  request arms `context_management` (+ `truncation:"auto"` as the fail-open
 *  backstop) and the SERVER owns fitting; `item` is a stored compaction item
 *  riding at the head of the input. Otherwise it's the client plan exactly as
 *  before: sliding-window `from` + optional summary in instructions. */
interface SendPlan {
  from: number
  summary?: string
  serverThreshold?: number
  item?: { id: string; content: string }
}

/** Decide how the next request on this chat manages context. `serverSide` =
 *  the caller established eligibility for the runner's own compaction: a
 *  local single-model lane, tools or not (the runner's agent loops compact
 *  before their first round), and not a continue - a continue's synthetic
 *  trailing user item would corrupt the tail anchor. */
function resolvePlan(
  conv: Conversation,
  window: number,
  maxReply: number,
  summarize: boolean,
  serverSide: boolean,
): SendPlan {
  if (serverSide && summarize && window > 0) {
    const threshold = serverCompactThreshold(window, maxReply)
    if (threshold > 0) {
      if (serverCompactionValid(conv)) {
        const sc = conv.serverCompaction!
        const from = activeMessages(conv).findIndex((m) => m.id === sc.tailStartId)
        return {
          from: Math.max(0, from),
          serverThreshold: threshold,
          item: { id: sc.id, content: sc.content },
        }
      }
      // No item yet: window-fit client-side as a one-time bootstrap (a thread
      // already past the window couldn't run the summarization pass at all);
      // the armed threshold takes over from here.
      return { from: trimIndex(conv, window, maxReply), serverThreshold: threshold }
    }
  }
  return planContext(conv, window, maxReply, summarize)
}

/** The thread divider's mirror of the next send: where the sent history will
 *  start and which summary stands in for what's before it. Uses the cached
 *  caps (sync) - the same data the send path resolves, minus the await. */
export function previewPlan(conv: Conversation | null | undefined): {
  from: number
  summary?: string
  by?: string
} {
  if (!conv) return { from: 0 }
  const models = useModelsStore()
  const settings = useSettingsStore()
  const compare = runningLanes(conv, models).length >= 2
  const window = compare ? minLaneCtx(conv, models) : models.ctxFor(conv.model)
  const isCloud = !!models.models.find((m) => m.id === conv.model)?.cloud
  const plan = resolvePlan(
    conv,
    window,
    replyReserve(settings.maxTokens),
    settings.summarize,
    !isCloud && !compare,
  )
  if (plan.item) return { from: plan.from, summary: plan.item.content, by: conv.serverCompaction?.model }
  return { from: plan.from, summary: plan.summary, by: conv.summaryModel }
}

/** Resolve the selection against this lane's endpoint. 'all' = every server
 *  tool the endpoint advertises plus the chat's switched-on connectors.
 *  Custom = exactly the picks: labels the endpoint serves ride as server
 *  tools, labels found in the library ride inline, anything else is inert
 *  (picked on another model's endpoint, or deleted). A whole-server pick
 *  swallows that server's single-tool picks. */
function activeToolSpecs(
  conv: Conversation,
  caps: ModelCaps,
  cloud = false,
  modelId = '',
): ToolSpec[] {
  const sel = toolSelection(conv)
  const lib0 = useConnectorsStore()
  // "Every model" (system) connectors reach LOCAL lanes as materialized
  // server tools; a CLOUD lane has no config file to carry them, so they
  // join inline - the manager's agent loop dials them like any connector.
  const systemForCloud = cloud ? lib0.list.filter((c) => c.system) : []
  const specs: ToolSpec[] = []
  if (sel.mode === 'all') {
    specs.push(artifactSpec(conv.id, modelId))
    if (useGraphsStore().groundingFor(conv.id)) specs.push(graphSpec(conv.id, modelId))
    for (const label of caps.mcpServers) specs.push({ label })
    for (const c of [...systemForCloud, ...activeConnectors(conv)]) {
      if (caps.mcpServers.includes(c.label)) continue
      if (specs.some((s) => s.label === c.label)) continue
      specs.push({ label: c.label, url: c.url, headers: c.headers, trusted: c.system || c.ports.length > 0 })
    }
    return specs
  }
  const byLabel = new Map<string, string[] | null>()
  for (const p of sel.picks) {
    if (p.tool == null) {
      byLabel.set(p.label, null)
      continue
    }
    const cur = byLabel.get(p.label)
    if (cur === null) continue
    byLabel.set(p.label, [...(cur ?? []), p.tool])
  }
  for (const [label, allowed] of byLabel) {
    if (label === ARTIFACTS_LABEL) {
      const spec = artifactSpec(conv.id, modelId)
      if (allowed) spec.allowed = allowed
      specs.push(spec)
      continue
    }
    const spec: ToolSpec = { label }
    if (!caps.mcpServers.includes(label)) {
      // on a cloud lane a system connector may be picked too - it has no
      // label carrier there, so it resolves like a personal one
      const c = lib0.list.find((x) => (cloud || !x.system) && x.label === label)
      if (!c) continue
      spec.url = c.url
      spec.headers = c.headers
      spec.trusted = c.system || c.ports.length > 0
    }
    if (allowed) spec.allowed = allowed
    specs.push(spec)
  }
  return specs
}

/** Does this turn instruct the model that its whole output is an extraction?
 *
 *  Two mechanisms, one meaning: a TASK TAG (granite-vision's `<chart2csv>` and
 *  friends, whose template swaps the message for IBM's own prompt) and an OCR
 *  MODE (the deepseek/paddle task chips, which the runner turns into that
 *  family's task prompt). Both say "your entire answer is this transcription",
 *  which is not a turn where calling a tool means anything.
 *
 *  Reads the OUTGOING input rather than conversation state, so it judges the
 *  turn actually being sent - a compare fan-out, a regenerate and a continue
 *  all land here with the same item list the model will see. */
function extractionTurn(input: unknown[], modelId: string, conv: Conversation): boolean {
  const caps = useModelsStore().caps[modelId]
  if (conv.ocrMode && caps?.ocr?.modes.includes(conv.ocrMode)) return true
  // The last user message is the one the tag would be on; earlier ones are
  // history and their tags already had their turn.
  const last = [...input]
    .reverse()
    .find((it) => (it as { role?: string }).role === 'user')
  if (!last) return false
  const content = (last as { content?: unknown }).content
  const text =
    typeof content === 'string'
      ? content
      : Array.isArray(content)
        ? (content.find((p: { type?: string }) => p.type === 'input_text') as
            | { text?: string }
            | undefined
          )?.text ?? ''
        : ''
  return isTaskTurn(text, caps?.taskTags)
}

function buildBody(
  conv: Conversation,
  modelId: string,
  exclude: Message,
  cont: boolean,
  maxTokens: number | null,
  maxToolCalls: number | null,
  plan: SendPlan,
  toolSpecs: ToolSpec[],
  webSearch: boolean,
  forensicsTool: boolean,
  clockTool: boolean,
  fileData: Map<string, string>,
): Record<string, unknown> {
  const p = conv.params
  const input = buildInput(conv, modelId, exclude, cont, plan.from, fileData)
  // A stored compaction item leads the input: the server's rewrite collapses
  // it into the tail's first user message, rendering the exact token stream
  // its own post-compaction pass ran on (prefix-cache aligned).
  if (plan.item) {
    input.unshift({ type: 'compaction', id: plan.item.id, encrypted_content: plan.item.content })
  }
  const body: Record<string, unknown> = {
    model: modelId,
    input,
    stream: true,
  }
  // Server-side context management (local single-model lanes): the runner
  // compacts with the model's real tokenizer when the rendered prompt crosses
  // the threshold; truncation:"auto" is the fail-open backstop (a failed
  // summarization proceeds uncompacted, and an over-window prompt then drops
  // leading turns loudly instead of 400ing the send).
  if (plan.serverThreshold) {
    body.context_management = [{ type: 'compaction', compact_threshold: plan.serverThreshold }]
    body.truncation = 'auto'
  }
  // Sampling rides only when the user set it (sampler popover): untouched
  // params stay ABSENT so the runner's or provider's own defaults apply -
  // the same honor-the-defaults stance the bench rules mandate.
  if (p.temperature != null) body.temperature = p.temperature
  if (p.topP != null) body.top_p = p.topP
  if (p.topK != null) body.top_k = p.topK
  if (p.minP != null) body.min_p = p.minP
  if (p.presencePenalty != null) body.presence_penalty = p.presencePenalty
  if (p.frequencyPenalty != null) body.frequency_penalty = p.frequencyPenalty
  if (p.repeatPenalty != null) body.repeat_penalty = p.repeatPenalty
  // Tool sources ride as OpenAI `mcp` tools, and the chat's globe toggle as a
  // `web_search` tool; the server runs the agent loop and streams mcp_call /
  // web_search_call items. A server tool goes by registered label alone (the
  // runner resolves it); a connector rides INLINE (hosted-MCP shape, the
  // runner dials the URL itself) with require_approval "always" - these are
  // third-party servers. A picked subset narrows either kind via
  // allowed_tools.
  const tools: unknown[] = []
  for (const s of toolSpecs) {
    const t: Record<string, unknown> = { type: 'mcp', server_label: s.label }
    if (s.url) {
      t.server_url = s.url
      t.require_approval = s.trusted ? 'never' : 'always'
      if (s.headers && Object.keys(s.headers).length) t.headers = s.headers
    }
    if (s.allowed?.length) t.allowed_tools = s.allowed
    tools.push(t)
  }
  if (webSearch) tools.push({ type: 'web_search' })
  // The on-demand forensics tool (analyze_document_forensics), when the endpoint
  // exposes it and the model has vision - the server runs it over an image
  // already in the conversation. Independent of the always-on injection above:
  // this lets the model proactively re-check a specific attachment mid-turn.
  if (forensicsTool) tools.push({ type: 'forensics' })
  // The builtin clock (current_time, paddock extension): rides silently on
  // every capable endpoint - temporal blindness is a defect, not a
  // preference, so unlike forensics there is no user switch. The declaration
  // carries this browser's IANA zone because that is the one fact the server
  // lacks: its clock is NTP-correct, but the box may sit in a UTC rack. The
  // date-only line in `instructions` covers day-scale grounding; this answers
  // "what time is it NOW", which an injected (send-frozen) time never can.
  if (clockTool) {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone
    tools.push(zone ? { type: 'current_time', timezone: zone } : { type: 'current_time' })
  }
  // An EXTRACTION turn declares no tools. A task tag ("Chart to CSV") and an
  // OCR mode are both instructions to the model about what its whole output
  // must be; a tool list is an invitation to do something else instead. Handed
  // both, granite-vision fuses them and emits a tool call as its extraction
  // output - the raw `{"name": ..., "arguments": ...}` blob shows up in the chat,
  // which the parser then correctly leaves as content because
  // nothing was ever invoked.
  //
  // Measured across all six granite tags with the artifacts server attached:
  // five fuse, and `<chart2summary>` calls artifact_create instead of
  // summarising. Every one is correct with tools withheld. The OCR modes are
  // the same shape of instruction (`body.ocr`, set below) so they are held to
  // the same rule rather than waiting to be found the same way.
  if (tools.length && !extractionTurn(input, modelId, conv)) body.tools = tools
  // The turn's tool budget (Settings), and only when the user set one - absent
  // means the server's own bounds apply untouched. It rides with the tools
  // because without them it means nothing, and it is the spec's own field
  // rather than a dialect of ours, so the same number governs a local runner
  // and a cloud relay identically.
  if (tools.length && maxToolCalls != null) body.max_tool_calls = maxToolCalls
  // The compaction summary rides in `instructions` (the single system message),
  // standing in for the older messages the window squeezed out.
  // Client temporal context, date-granular on PURPOSE (SOTA parity: ChatGPT,
  // Claude, and Open WebUI all date their system prompts). Never clock time:
  // this block heads every request, so a minute-stamp would re-tokenize the
  // prompt head each send and void the conversation's radix prefix - and a
  // frozen send-time cannot answer "what time is it" anyway (that is the
  // current_time TOOL's job). Browser-computed per send because the date and
  // timezone are USER facts: the runner may sit on a UTC rack.
  const now = new Date()
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone
  const dateLine = `Today's date: ${now.toLocaleDateString('en-US', { weekday: 'long' })}, ${now.toLocaleDateString('en-CA')}${tz ? ` (user timezone: ${tz})` : ''}.`
  const instructions = [
    dateLine,
    conv.systemPrompt.trim(),
    // The attached graph's schema + counts, built tab-side where the schema
    // lives ('' while no graph is ready). The static how-to-use text arrives
    // separately, as the graph MCP server's own instructions.
    useGraphsStore().groundingFor(conv.id),
    plan.summary
      ? `Summary of the earlier part of this conversation (older messages were compacted):\n${plan.summary}`
      : '',
  ]
    .filter(Boolean)
    .join('\n\n')
  if (instructions) body.instructions = instructions
  // Global max output tokens (Settings slider), bounded server-side by max_ctx.
  if (maxTokens != null && maxTokens > 0) body.max_output_tokens = maxTokens
  if (p.seed != null) body.seed = p.seed
  // The OCR reading mode + regions toggle (deepseek2-ocr lanes).
  // Three gates, all honest ones: the LANE's endpoint must advertise the
  // vocabulary (a compare fan-out sends this only to the OCR lane), the
  // chosen mode must be in that advertised list, and the request must
  // actually carry something to read - the server 400s an `ocr` object on an
  // imageless request by design. What actually ran comes back in the echo.
  const ocrCaps = useModelsStore().caps[modelId]?.ocr
  if (ocrCaps && (conv.ocrMode || conv.ocrRegions)) {
    const visual = input.some(
      (it) =>
        Array.isArray((it as { content?: unknown }).content) &&
        ((it as { content: { type?: string }[] }).content.some(
          (p) => p.type === 'input_image' || p.type === 'input_file',
        )),
    )
    if (visual) {
      const o: Record<string, unknown> = {}
      if (conv.ocrMode && ocrCaps.modes.includes(conv.ocrMode)) o.mode = conv.ocrMode
      if (conv.ocrRegions && ocrCaps.grounding) o.grounding = true
      if (Object.keys(o).length) body.ocr = o
    }
  }
  // Document metadata injection is the server default ("full") - only the
  // opt-out travels on the wire.
  if (conv.fileMetadataEnabled === false) body.file_metadata = 'off'
  // Forensics: the Studio is explicit in both directions and never
  // falls through to the endpoint's `[forensics] auto`. An endpoint configured
  // with `auto = "images"` would otherwise run forensics over every attachment
  // the moment a chat starts, with the composer switch reading "on" for a
  // choice nobody made - so an un-opted-in chat sends "off" rather than
  // nothing. The three-state wire contract is unchanged for other clients:
  // absent still means "endpoint default", the Studio just never sends absent.
  // Still gated on the endpoint advertising forensics: `forensics` is a paddock
  // local extension, and an unknown field is a 400 on a BYO-key external
  // provider. Where it is not advertised there is nothing to suppress anyway.
  if (useModelsStore().caps[modelId]?.forensics) {
    body.forensics = conv.forensicsEnabled === true ? 'on' : 'off'
  }
  if (conv.pdfMode === 'text') body.pdf_mode = 'text'
  if (conv.maxPages) body.max_pages = conv.maxPages
  // Reasoning control rides the lane's own surface. A lane with rungs takes
  // reasoning.effort - including the spec's `none`, which is how "off" reaches
  // a model that grades AND can be switched off (Qwen3.8); the runner turns
  // that into enable_thinking:false and sends no rung. A lane with only a
  // switch keeps the qwen-style chat_template_kwargs, which is what the cloud
  // relay translates per provider (OpenRouter reasoning{enabled}, Anthropic
  // extended thinking) and must not change shape here. 'none' lanes get
  // neither. Continue forces thinking off so it resumes the answer instead of
  // re-opening <think>.
  const store = useModelsStore()
  const ladder = store.reasoningLadderFor(modelId)
  const off = cont || p.thinking === false
  if (ladder.levels.length) {
    // `opens` (the lowest rung), not `dflt` (the checkpoint's published
    // default) - the Studio's own opening choice, and the same one the picker
    // is showing. An API caller that sends no effort still gets the published
    // default; this is only what the Studio asks for on your behalf.
    body.reasoning = { effort: off && ladder.off ? 'none' : p.reasoningEffort || ladder.opens }
  } else if (ladder.off) {
    body.chat_template_kwargs = { enable_thinking: !off }
  }
  // The thinking budget (reasoning.max_tokens): at the cap the runner forces
  // the model out of its think block. Sent only where the runner advertises
  // it can ENFORCE it - the capability gate keeps it off cloud lanes and the
  // channel-reasoning families (gpt-oss, muse), which would refuse it.
  if (!off && p.thinkingBudget && store.thinkingBudgetFor(modelId)) {
    body.reasoning = {
      ...((body.reasoning as Record<string, unknown> | undefined) ?? {}),
      max_tokens: p.thinkingBudget,
    }
  }
  // Prior turns' thinking. Sent explicitly either way on a template that grades
  // it, because its own default is `true` (Qwen3.8) - leaving it unset does not
  // mean "off", it means the template renders an empty <think></think> shell on
  // every past answer, since `buildInput` only attaches reasoning when this is
  // on. Saying which we want removes the shell and makes the control real.
  if (ladder.preserve) {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs as Record<string, unknown> | undefined),
      preserve_thinking: !cont && p.preserveThinking === true,
    }
  }
  return body
}

interface McpItem {
  type: string
  id?: string
  call_id?: string
  server_label?: string
  name?: string
  arguments?: string
  output?: string
  /** mcp_call / web_search_call failure: the error message (server sends a string). */
  error?: boolean | string
  status?: string
  /** web_search_call: the search action (query + the sources it found). */
  action?: { query?: string; sources?: Array<{ url?: string; title?: string }> }
  /** web_search_call: which engine ran it. A paddock extension - OpenAI's
   *  item has no provider because theirs is one engine; ours can be any of
   *  five, and the answer belongs with the turn that spent the money. */
  paddock_provider?: string
  /** compaction item: the summary (plaintext on a local runner). */
  encrypted_content?: string
  /** context-enrichment items (forensics / file_metadata): the 0-based index of
   *  the attachment among the turn's image/PDF parts, its kind, and the payload
   *  the manager persists (`report` for forensics, `meta` for file_metadata). */
  image_index?: number
  kind?: string
  report?: unknown
  meta?: unknown
}

interface ResponseEvent {
  type: string
  delta?: string
  item?: McpItem
  response?: {
    usage?: {
      input_tokens?: number
      output_tokens?: number
      output_tokens_details?: { reasoning_tokens?: number }
    }
    error?: { message?: string }
    incomplete_details?: { reason?: string }
    /** relay-added on cloud lanes: who actually served a routed request. */
    provider?: string
    /** truncation:"auto" fired: how many leading items the server dropped. */
    truncation_dropped_items?: number
    /** deepseek2-ocr resolution echo + grounded regions. */
    ocr?: unknown
  }
}

/** Upsert an MCP tool-call card on the assistant turn, keyed by call id. Both
 *  `mcp_call` (id = call id) and `mcp_approval_request` (call_id = call id)
 *  events flow through here, so the approval gate and the execution update the
 *  same card. */
function upsertCall(assistant: Message, key: string, patch: Partial<McpCall>): void {
  if (!assistant.toolCalls) assistant.toolCalls = []
  const existing = assistant.toolCalls.find((c) => c.id === key)
  if (existing) {
    Object.assign(existing, patch)
  } else {
    assistant.toolCalls.push({
      id: key,
      serverLabel: patch.serverLabel ?? '',
      name: patch.name ?? '',
      arguments: patch.arguments ?? '',
      status: patch.status ?? 'in_progress',
      ...patch,
    })
  }
}

/** Translate an mcp_call / mcp_approval_request output item into a card patch. */
function applyMcpItem(assistant: Message, item: McpItem, done: boolean): void {
  if (item.type === 'mcp_approval_request') {
    const key = item.call_id ?? item.id ?? ''
    if (!key) return
    if (!done) {
      upsertCall(assistant, key, {
        serverLabel: item.server_label,
        name: item.name,
        arguments: item.arguments,
        status: 'pending',
        approvalId: item.id,
      })
    } else {
      // approved -> the mcp_call events take over; denied -> settle the card here.
      const denied = item.status === 'denied'
      upsertCall(assistant, key, {
        approvalId: undefined,
        ...(denied ? { status: 'denied' as McpCallStatus } : {}),
      })
    }
  } else if (item.type === 'mcp_call') {
    const key = item.id ?? ''
    if (!key) return
    const status = (item.status as McpCallStatus | undefined) ?? (done ? 'completed' : 'in_progress')
    upsertCall(assistant, key, {
      serverLabel: item.server_label,
      name: item.name,
      arguments: item.arguments,
      ...(done ? { output: item.output, error: item.error } : {}),
      status,
    })
  }
}

/** Translate a web_search_call output item into a search card on the turn. */
function applyWebItem(assistant: Message, item: McpItem, done: boolean): void {
  const key = item.id ?? ''
  if (!key) return
  if (!assistant.webSearches) assistant.webSearches = []
  const patch: WebSearchCall = {
    id: key,
    query: item.action?.query ?? '',
    status: (item.status as WebSearchStatus | undefined) ?? (done ? 'completed' : 'in_progress'),
    sources: (item.action?.sources ?? []).flatMap((s) =>
      s.url ? [{ url: s.url, title: s.title }] : [],
    ),
    error: typeof item.error === 'string' ? item.error : undefined,
    provider: item.paddock_provider,
  }
  const existing = assistant.webSearches.find((c) => c.id === key)
  if (existing) Object.assign(existing, patch)
  else assistant.webSearches.push(patch)
}

async function friendlyHttpError(res: Response): Promise<string> {
  // The body's own message first: a cloud provider's 401/404 says exactly
  // what's wrong (bad key, unknown model) and the local hint below would
  // mislead. The hint stays for servers that answer with no message.
  try {
    const body = (await res.json()) as { error?: { message?: string } }
    if (body.error?.message) return body.error.message
  } catch {
    /* fall through */
  }
  if (res.status === 404 || res.status === 503) {
    return 'No model is loaded on the server. Start paddock with --model, or load one.'
  }
  return `Request failed (HTTP ${res.status}).`
}

export function useChatStream() {
  const chat = useChatStore()
  const settings = useSettingsStore()
  const models = useModelsStore()
  const tele = useTelemetryStore()
  const prompts = usePromptsStore()

  /** Busy = the chat you are LOOKING at is mid-answer. The composer's Stop
   *  button and the send guards both key on this, so neither can reach into a
   *  conversation that is not on screen. */
  const isStreaming = computed(() => conversationBusy(chat.activeId))

  /** Whether this turn may search the web: the chat's globe toggle AND this
   *  model's endpoint actually supplying the integration. Default on when the
   *  endpoint has it - the composer shows the same default, and the two must
   *  agree or the toggle lies about what the request carries. */
  function webSearchOn(conv: Conversation, modelId: string): boolean {
    const store = useModelsStore()
    return conv.webSearchEnabled !== false && store.webSearchFor(modelId)
  }

  /** Whether to declare the on-demand forensics tool this turn. Needs the
   *  endpoint to expose it (`tool`) on a vision model AND the user to have
   *  switched forensics on for this chat.
   *
   *  Opt-in, like the injection flag and for the same reason: declaring the
   *  tool lets the model call forensics on its own, which is the pass running by
   *  default in everything but name. One switch, one meaning - off unless asked
   *  for. */
  function forensicsToolOn(conv: Conversation, caps: ModelCaps): boolean {
    const f = caps.forensics
    if (!f || !f.vision || !f.tool) return false
    return conv.forensicsEnabled === true
  }

  /** The user turn this assistant message answers, when it carried an audio
   *  clip. That - not the model's kind - is what makes a turn a
   *  transcription, which is the whole reason a mixed thread works: drop a
   *  clip and whisper answers it, then type a question and a chat model
   *  answers the same conversation, because the transcript in between is an
   *  ordinary text message. */
  function clipFor(
    conv: Conversation,
    assistant: Message,
  ): { audio: AudioPart; said: string } | undefined {
    const msgs = activeMessages(conv)
    const at = msgs.indexOf(assistant)
    for (let i = (at < 0 ? msgs.length : at) - 1; i >= 0; i--) {
      const m = msgs[i]
      if (m.role !== 'user') continue
      const audio = m.content.find((p): p is AudioPart => p.type === 'audio')
      return audio ? { audio, said: messageText(m).trim() } : undefined
    }
    return undefined
  }

  /** The enrichment pass: a transcript that settled without
   *  word times gets them from the fleet's forced aligner, if one is
   *  running - which is what finally gives every lane karaoke, not just
   *  whisper's DTW one. Strictly additive and strictly quiet about failure:
   *  the transcription already succeeded, so a missed enrichment costs the
   *  highlight, never the turn. */
  async function enrichWithAlignment(
    assistant: Message,
    blob: Blob,
    filename: string,
    text: string,
    signal: AbortSignal,
  ): Promise<void> {
    const t = assistant.transcript
    if (!t || !text.trim()) return
    // already word-timed (whisper asked for `word` granularity) = done
    if (t.words?.some((w) => w.start !== undefined && w.end !== undefined)) return
    // the runner would 400 these (no morphological tokenizer); skip quietly
    if (alignmentRefused(t.language)) return
    const lane = models.alignerLane()
    if (!lane) return
    const caps = await models.capsFor(lane.id)
    const cap = caps.alignmentMaxClipS
    if (cap !== undefined && t.durationS !== undefined && t.durationS > cap) return
    try {
      const out = await alignClip(lane.url, blob, filename, text, t.language, signal)
      const words = mergeWordTimes(t, text, out.words)
      // reassignment (not mutation) so the transcript computed re-renders.
      // `wordsFrom` is what makes the pass visible on the turn - the times
      // themselves only appear while the clip plays.
      if (words) {
        assistant.transcript = {
          ...t,
          words,
          wordsFrom: lane.id,
          wordsLangOk: out.languageSupported,
        }
      }
    } catch (e) {
      if (!signal.aborted) console.warn('alignment enrichment failed', e)
    }
  }

  /** Answer one turn by TRANSCRIBING its clip. Deliberately not a branch
   *  inside `run`: it speaks a different protocol (multipart +
   *  `transcript.text.*`) to a different endpoint, and shares only the parts
   *  that make it a turn - the abort set, the rAF-throttled apply, the run
   *  record, usage, and persistence. Compare needs nothing extra: `fanOut`
   *  already makes one grouped assistant message per lane, so N transcribers
   *  race side by side through exactly this. */
  async function runTranscription(
    conv: Conversation,
    assistant: Message,
    clip: { audio: AudioPart; said: string },
    lane: boolean,
    sharedGpu: boolean,
  ): Promise<void> {
    const modelId = assistant.model ?? conv.model
    await models.capsFor(modelId)
    const controller = new AbortController()
    beginStream(conv.id, controller)
    assistant.run = {
      model: modelId,
      // Sampling does not reach this endpoint at all - a transcription takes
      // the clip, the language and (on a generative lane) the instruction.
      // The record says so by carrying the chat's params untouched with no
      // reply cap, rather than inventing dials that never rode.
      params: { ...conv.params, maxTokens: null },
      tools: [],
      contended: sharedGpu || undefined,
      at: Date.now(),
    }
    if (!lane) tele.beginCapture()
    // Mark the turn a transcription now, not when it finishes: the renderer
    // keys on this, and letting a growing transcript fall through the markdown
    // path first would both reflow when it settles and let a line starting
    // "- " briefly render as a bullet list.
    assistant.transcript = {}

    let scheduled = false
    let raw = ''
    const apply = (): void => {
      scheduled = false
      const tp = assistant.content[0]
      if (tp && tp.type === 'text') tp.text = raw
    }
    const schedule = (): void => {
      if (!scheduled) {
        scheduled = true
        requestAnimationFrame(apply)
      }
    }

    try {
      // Where this clip goes: a local runner through the manager relay, or a
      // provider's speech endpoint through the manager's cloud seam.
      // `transcribeUrl` reads the model's own entry and never falls back to
      // "any chat runner" - that fallback exists so a stale id can still find
      // a tokenizer to ask, and here it would POST the clip at whatever else
      // happened to be running. A stopped model must say it is stopped.
      const url = models.transcribeUrl(modelId)
      if (!url) {
        assistant.error = `No running model serves "${modelId}" - start one in the Manager first.`
        return
      }
      if (!models.canTranscribe(modelId)) {
        assistant.error = `${fleetLabel(modelId)} cannot read audio. Pick a speech model, or a model with an audio companion loaded.`
        return
      }
      if (!clip.audio.attachmentId) {
        assistant.error = 'The audio for this turn is no longer stored, so it cannot be transcribed again.'
        return
      }
      const stored = await fetch(attachmentsApi.url(clip.audio.attachmentId), {
        signal: controller.signal,
      })
      if (!stored.ok) throw new Error(`audio attachment: HTTP ${stored.status}`)
      const instruction = models.canChat(modelId) ? clip.said || undefined : undefined
      // held in a local because the clip goes out twice: transcription now,
      // and the alignment enrichment after the transcript settles
      const clipBlob = await stored.blob()
      const clipName = clip.audio.name || 'recording.webm'
      const out = await transcribeClip(clipBlob, clipName, {
        url,
        // A cloud endpoint serves many models and needs the one named; the
        // wire id is the bare provider slug, not our `cloud:endpoint:` key.
        model: models.models.find((m) => m.id === modelId)?.cloud
          ? modelId.replace(/^cloud:[^:]+:/, '').split('@')[0]
          : undefined,
        // Providers answer one JSON body when the clip is done - no lane but a
        // local runner types a transcript in as it goes.
        stream: models.transcribeStreams(modelId),
        language: clip.audio.language,
        // On a generative ASR model the INSTRUCTION selects the task
        // (punctuated transcript vs speaker labels vs translation), so what
        // the user typed alongside the clip is exactly that instruction.
        // Whisper has no such interface - `canChat` is the same test that
        // decides whether the composer offered a text box at all.
        prompt: instruction,
        timestamps: models.canTimeSegments(modelId),
        // Per-WORD times where the endpoint has them. Asked for by
        // default because it is the whole point of a transcript beside a
        // player: clicking the word you want to hear again, where a
        // sentence-level seek only ever got you near it.
        //
        // Not when the user typed an instruction, though. On the lane that
        // writes its times into the transcript (granite-speech-plus) the times
        // are an instruction, and the model takes one - so asking for both is a
        // refusal. What the user typed is the explicit ask and wins; the times
        // were ours.
        wordTimes: models.canTimeWords(modelId) && !instruction,
        // "how sure were you" is a separate ask from "when was it said" - the
        // generative lanes answer the first and cannot answer the second.
        wordConfidence: models.canWordConfidence(modelId),
        signal: controller.signal,
        onDelta: (t) => {
          raw = t
          schedule()
        },
      })
      raw = out.text
      apply()
      assistant.transcript = out.meta
      assistant.usage = out.usage
      // still inside the try, before the finally's persistNow - one save
      // carries the enriched result
      await enrichWithAlignment(assistant, clipBlob, clipName, out.text, controller.signal)
    } catch (e) {
      apply()
      if (controller.signal.aborted) {
        assistant.stopped = true
      } else {
        assistant.error = e instanceof Error ? e.message : String(e)
        console.error('transcription failed', e)
      }
    } finally {
      // A turn that heard nothing is not a transcript: drop the marker so it
      // renders as the plain failure it is, instead of an empty transcript
      // with a player for audio no model ever received. Covers every refusal
      // above, not just the thrown ones.
      if (!raw && assistant.error) assistant.transcript = undefined
      if (!lane) {
        const gpu = tele.endCapture()
        if (gpu && assistant.run) assistant.run.gpu = gpu
      }
      assistant.streaming = false
      endStream(conv.id, controller)
      // The transcript is what an audio chat is about, so it names the chat -
      // and it only exists now, which is why titling runs here as well as at
      // send time (see chat.maybeTitle).
      chat.maybeTitle(conv)
      chat.persistNow(conv)
    }
  }

  /** Stream one assistant turn. `lane` = this turn is one of a compare
   *  fan-out: it targets the MESSAGE's model, skips GPU capture (the peaks
   *  would be everyone's), skips compaction, and is stamped `contended`.
   *  Tools/web ride along like any turn - each lane declares exactly what
   *  its endpoint advertises (per-model config; mismatches are labeled in
   *  the thread, never silently leveled). */
  /** Attachment ids of every image/PDF part in the conversation, in the order
   *  the runner's forensics pass walks them (messages, then parts) - so a
   *  forensics item's `image_index` maps back to the attachment it describes.
   *
   *  Must count exactly what the runner counts: image parts and PDF file parts.
   *  The runner keys `image_index` off `image_part_bytes`/`pdf_part_bytes`, and
   *  a docx (or any non-PDF file) yields neither, so it never takes a slot there.
   *  Counting every `file` here - as this once did - shifted every index past a
   *  docx by one and landed reports on the wrong attachment. */
  function orderedForensicAttachmentIds(conv: Conversation): string[] {
    const ids: string[] = []
    // The SENT branch, for the same reason the comment above cares about docx:
    // these are positions in the prompt the runner counts, so an attachment on
    // a branch the user abandoned would shift every index after it.
    for (const m of activeMessages(conv)) {
      for (const p of m.content) {
        if ((p.type === 'image' || (p.type === 'file' && isPdfPart(p))) && p.attachmentId)
          ids.push(p.attachmentId)
      }
    }
    return ids
  }

  /** Write a forensics output item through to the DB for its attachment. The
   *  runner already computed and returned it; this just lands it. Best-effort:
   *  a failure (or no attachment to key it to) never disrupts the chat - the
   *  report was still shown live this turn, it just is not stored. */
  function persistForensics(conv: Conversation, item: McpItem): void {
    const idx = typeof item.image_index === 'number' ? item.image_index : 0
    const attachmentId = orderedForensicAttachmentIds(conv)[idx]
    if (!attachmentId || item.report == null) return
    void forensicsApi
      .persist({
        attachment_id: attachmentId,
        conversation_id: conv.id,
        kind: item.kind ?? 'image',
        report: item.report,
      })
      .catch(() => {})
  }

  /** Write a file_metadata output item through to the DB for its attachment, so
   *  the metadata panel serves it from the store thereafter (runner-independent).
   *  Same best-effort, fire-and-forget stance as forensics. */
  function persistFileMetadata(conv: Conversation, item: McpItem): void {
    const idx = typeof item.image_index === 'number' ? item.image_index : 0
    const attachmentId = orderedForensicAttachmentIds(conv)[idx]
    if (!attachmentId || item.meta == null) return
    void attachmentsApi.storeMetadata(attachmentId, item.meta).catch(() => {})
  }

  async function run(
    conv: Conversation,
    assistant: Message,
    append = false,
    lane = false,
    sharedGpu = false,
  ): Promise<void> {
    const modelId = assistant.model ?? conv.model
    // Audio in the turn = transcription, whatever the model is.
    const clip = append ? undefined : clipFor(conv, assistant)
    if (clip) {
      await runTranscription(conv, assistant, clip, lane, sharedGpu)
      return
    }
    // A speech model with no clip to work on: say what it can do rather than
    // send it a chat request it will refuse in its own words.
    if (!models.canChat(modelId)) {
      assistant.error = `${fleetLabel(modelId)} only turns speech into text - attach an audio clip, or pick a chat model.`
      assistant.streaming = false
      chat.persistNow(conv)
      return
    }
    // This model's advertised server tools - cached; a miss reads as none.
    const caps = await models.capsFor(modelId)
    const isCloud = !!models.models.find((m) => m.id === modelId)?.cloud
    const toolSpecs = activeToolSpecs(conv, caps, isCloud, modelId)
    const webSearch = webSearchOn(conv, modelId)
    const forensicsTool = forensicsToolOn(conv, caps)
    // Cloud lanes always get the clock: the manager's cloud loop serves it
    // (same-version as this Studio), so there is no capability to wait for.
    const clockTool = caps.currentTime === true || isCloud
    // How this request fits the window. Local single-model lanes hand context
    // to the RUNNER (context_management compaction, exact tokens, prefix-cache
    // aligned) - tool-bearing chats included, since the agent loops compact
    // before their first round. Everything else keeps the client plan: cloud
    // lanes their background summarize, compare lanes the min-of-lanes
    // sliding window.
    const window = lane ? minLaneCtx(conv, models) : models.ctxFor(modelId)
    const plan = resolvePlan(
      conv,
      window,
      replyReserve(settings.maxTokens),
      settings.summarize,
      !isCloud && !lane && !append,
    )
    // Resolve "model maximum" once, here: the window minus the planned prompt.
    // Both the wire and the run record use this number - the run record must
    // show what actually rode (same rule as the maxTokens note below).
    const replyCap =
      settings.maxTokens ??
      windowRemaining(window, promptTokensFrom(conv, plan.from), models.outCapFor(modelId))
    const controller = new AbortController()
    beginStream(conv.id, controller)
    const started = performance.now()
    // split timing: reasoning phase vs. answer phase
    let reasoningStartAt = 0
    let contentStartAt = 0
    let firstTokenAt = 0 // send -> first token (TTFT)

    // Record this turn as a "run" (provenance): exactly what produced the answer.
    // Continue keeps the original turn's run; a fresh/regenerated turn snapshots
    // the current settings. GPU environment is stamped on in `finally`.
    if (!append || !assistant.run) {
      assistant.run = {
        model: modelId,
        spec: models.specFor(modelId),
        systemPrompt: conv.systemPrompt || undefined,
        systemPromptName: prompts.prompts.find((p) => p.body === conv.systemPrompt)?.name,
        // conv.params carries a per-conversation maxTokens that the stream
        // does not send - the global Settings slider is the real cap. The run
        // record must show what actually rode the wire (a 4096-slider send
        // was recorded as 8192, hiding why thinking hit the ceiling).
        params: { ...conv.params, maxTokens: replyCap },
        // Provenance: exactly the tool sources that rode - a narrowed server
        // shows per-tool ("github:create_issue"), a whole one just its label.
        tools: toolSpecs.flatMap((s) =>
          s.allowed?.length ? s.allowed.map((t) => `${s.label}:${t}`) : [s.label],
        ),
        contended: sharedGpu || undefined,
        at: Date.now(),
      }
    }
    if (!lane) tele.beginCapture()

    // rAF-throttled apply of accumulated deltas onto the reactive message
    // (continue mode resumes from the text already there)
    let rawText = append ? messageText(assistant) : ''
    let reasoningBuf = append ? (assistant.reasoning ?? '') : ''
    let scheduled = false
    const apply = () => {
      scheduled = false
      const tp = assistant.content[0]
      if (tp && tp.type === 'text') tp.text = rawText
      assistant.reasoning = reasoningBuf || undefined
    }
    const schedule = () => {
      if (!scheduled) {
        scheduled = true
        requestAnimationFrame(apply)
      }
    }

    // Pre-pass: fetch PDF attachment bytes (kept out of the doc) so the sync
    // body-builder can inline them as `input_file` parts.
    const fileData = await resolveFileData(conv)
    try {
      // Studio chats go through the manager's relay (doc §10): the manager
      // originates the runner call as a client, keyed by the runner serving
      // this turn's model.
      const endpoint = models.responsesUrl(modelId)
      if (!endpoint) {
        assistant.error = `No running model serves "${modelId}" - start one in the Manager first.`
        return
      }
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          buildBody(
            conv,
            modelId,
            assistant,
            append,
            replyCap,
            settings.maxToolCalls,
            plan,
            toolSpecs,
            webSearch,
            forensicsTool,
            clockTool,
            fileData,
          ),
        ),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) {
        assistant.error = await friendlyHttpError(res)
        return
      }

      let usage:
        | {
            input_tokens?: number
            output_tokens?: number
            output_tokens_details?: { reasoning_tokens?: number }
            /** OpenRouter's per-response price in USD; other providers omit it. */
            cost?: number
          }
        | undefined
      let servedBy: string | undefined
      // server-side compaction: the item arrives complete on its output_item
      // events; committed to the conversation after the stream ends
      let minted: { id: string; content: string } | undefined
      let dropped = 0
      for await (const data of readSse(res.body)) {
        let ev: ResponseEvent
        try {
          ev = JSON.parse(data) as ResponseEvent
        } catch {
          continue
        }
        switch (ev.type) {
          case 'response.output_text.delta':
            if (!firstTokenAt) firstTokenAt = performance.now()
            if (!contentStartAt) {
              contentStartAt = performance.now()
              // reasoning just ended - surface its duration immediately, so the
              // fold shows "Thought for Xs" as the answer starts (tokens/tps
              // fill in at the terminal usage event).
              if (reasoningStartAt) {
                assistant.usage = {
                  promptTokens: 0,
                  completionTokens: 0,
                  ...(assistant.usage ?? {}),
                  reasoningMs: contentStartAt - reasoningStartAt,
                }
              }
            }
            rawText += ev.delta ?? ''
            schedule()
            break
          // reasoning_text is what runners stream; reasoning_summary_text is
          // OpenAI's Responses shape for reasoning models; bare reasoning is
          // OpenRouter's Responses spelling - same fold for all three
          case 'response.reasoning_text.delta':
          case 'response.reasoning_summary_text.delta':
          case 'response.reasoning.delta':
            if (!firstTokenAt) firstTokenAt = performance.now()
            if (!reasoningStartAt) reasoningStartAt = performance.now()
            reasoningBuf += ev.delta ?? ''
            schedule()
            break
          case 'response.output_item.added':
            if (ev.item?.type === 'web_search_call') applyWebItem(assistant, ev.item, false)
            // Context-enrichment items are server-produced, not tool calls - keep
            // them out of applyMcpItem (which only speaks mcp/web).
            else if (ev.item?.type === 'forensics' || ev.item?.type === 'file_metadata') break
            else if (ev.item) applyMcpItem(assistant, ev.item, false)
            break
          case 'response.output_item.done':
            if (ev.item?.type === 'compaction') {
              if (ev.item.id && ev.item.encrypted_content) {
                minted = { id: ev.item.id, content: ev.item.encrypted_content }
              }
            } else if (ev.item?.type === 'web_search_call') applyWebItem(assistant, ev.item, true)
            // Write the enrichment through to the DB for its attachment. The
            // runner already computed it and handed it to us here; persistence is
            // fire-and-forget so it can never disrupt the chat stream.
            else if (ev.item?.type === 'forensics') persistForensics(conv, ev.item)
            else if (ev.item?.type === 'file_metadata') persistFileMetadata(conv, ev.item)
            else if (ev.item) {
              applyMcpItem(assistant, ev.item, true)
              // An artifact tool completed: show it NOW. The list refresh
              // used to wait for isStreaming to flip false, which in compare
              // means all lanes - the fast lane's finished graph stayed
              // invisible until the slowest lane was done.
              if (
                ev.item.type === 'mcp_call' &&
                ev.item.server_label === ARTIFACTS_LABEL &&
                ev.item.status !== 'failed'
              ) {
                void useArtifactsStore().refresh(conv.id)
              }
            }
            break
          case 'response.completed':
            usage = ev.response?.usage
            servedBy = ev.response?.provider
            dropped = ev.response?.truncation_dropped_items ?? 0
            // the OCR resolution echo (+ grounded regions) rides only the
            // terminal response object - hang it on the turn like transcript
            if (ev.response?.ocr) assistant.ocr = ocrMetaFromWire(ev.response.ocr)
            break
          case 'response.incomplete':
            usage = ev.response?.usage
            servedBy = ev.response?.provider
            dropped = ev.response?.truncation_dropped_items ?? 0
            if (ev.response?.ocr) assistant.ocr = ocrMetaFromWire(ev.response.ocr)
            if (ev.response?.incomplete_details?.reason === 'max_output_tokens') {
              assistant.incomplete = 'length'
              // A call cut off mid-arguments never comes back as an output
              // item, so its optimistic card would sit unresolved and read as
              // a bare "error" - the one thing the user cannot act on. The
              // wire says exactly why; say it. (Artifacts hit this first: a
              // page's content is long, so a modest output cap truncates the
              // call rather than the reply.)
              for (const c of assistant.toolCalls ?? []) {
                if (c.status === 'in_progress' || c.status === 'pending') {
                  c.status = 'failed'
                  c.error =
                    'Cut off: the reply hit the maximum output tokens while writing this call. Raise it in Settings and try again.'
                }
              }
            }
            break
          case 'response.failed':
            assistant.error = ev.response?.error?.message ?? 'The model failed to respond.'
            break
          default:
            break
        }
      }

      apply()
      // A compaction item arrived: the server folded everything before this
      // turn's user message into the summary. Store it - the next send resends
      // [item, tail...] and skips the covered prefix entirely. The anchor is
      // the newest user message (the server's tail rule is "last user item").
      if (minted) {
        let anchor: string | undefined
        const path = activeMessages(conv)
        for (let i = path.length - 1; i >= 0; i--) {
          if (path[i].role === 'user') {
            anchor = path[i].id
            break
          }
        }
        if (anchor) {
          conv.serverCompaction = { ...minted, tailStartId: anchor, model: modelId, at: Date.now() }
        }
      }
      // the fail-open backstop fired (compaction couldn't run/failed AND the
      // prompt overflowed): the server dropped leading turns and said so
      if (dropped > 0) {
        console.warn(`server dropped ${dropped} leading input item(s) to fit the context window`)
      }
      if (usage) {
        const end = performance.now()
        const outTokens = usage.output_tokens ?? 0
        const rTokens = usage.output_tokens_details?.reasoning_tokens ?? 0
        const answerTokens = Math.max(0, outTokens - rTokens)

        // reasoning phase: first reasoning token -> first content token (or end)
        const reasoningMs = reasoningStartAt
          ? (contentStartAt || end) - reasoningStartAt
          : undefined
        // answer phase: first content token -> done
        const answerMs = contentStartAt ? end - contentStartAt : undefined

        // Continue accumulates onto the truncated reply's totals (a fresh reply
        // has no prior to add to; the content-start partial is recomputed here).
        const prev = append ? assistant.usage : undefined
        const tAnswerTokens = (prev?.completionTokens ?? 0) + answerTokens
        const tAnswerMs = (prev?.answerMs ?? 0) + (answerMs ?? 0)
        const tReasonTokens = (prev?.reasoningTokens ?? 0) + rTokens
        const tReasonMs = (prev?.reasoningMs ?? 0) + (reasoningMs ?? 0)
        // cost accumulates like tokens on Continue; stays undefined when the
        // provider never reported money (never a fake $0)
        const tCost =
          usage.cost !== undefined || prev?.costUsd !== undefined
            ? (prev?.costUsd ?? 0) + (usage.cost ?? 0)
            : undefined

        assistant.usage = {
          promptTokens: usage.input_tokens ?? prev?.promptTokens ?? 0,
          completionTokens: tAnswerTokens,
          ms: (prev?.ms ?? 0) + (end - started),
          ttftMs: firstTokenAt ? Math.round(firstTokenAt - started) : prev?.ttftMs,
          answerMs: tAnswerMs || undefined,
          tps: tAnswerTokens > 0 && tAnswerMs > 0 ? tAnswerTokens / (tAnswerMs / 1000) : undefined,
          reasoningTokens: tReasonTokens || undefined,
          reasoningMs: tReasonMs || undefined,
          reasoningTps:
            tReasonTokens > 0 && tReasonMs > 0 ? tReasonTokens / (tReasonMs / 1000) : undefined,
          provider: servedBy ?? prev?.provider,
          costUsd: tCost,
        }
      }
    } catch (e) {
      apply()
      if (controller.signal.aborted) {
        assistant.stopped = true
      } else {
        assistant.error = 'Could not reach the server. Is paddock running?'
        console.error('chat stream failed', e)
      }
    } finally {
      // Stamp the GPU/engine peaks sampled during the turn onto its run record
      // (single-model turns only - a lane's peaks would be everyone's).
      if (!lane) {
        const gpu = tele.endCapture()
        if (gpu && assistant.run) assistant.run.gpu = gpu
      }
      assistant.streaming = false
      endStream(conv.id, controller)
      chat.persistNow(conv)
      // Turn's done: if the thread has outgrown the window, fold the oldest
      // messages into the summary now, in the background - the user is reading
      // the reply, and the next send just picks the summary up if it's ready.
      // Compare chats skip compaction (one summary can't stand in for N
      // divergent lane histories), and server-managed turns skip it too - the
      // runner compacts inline with the model's real tokenizer.
      if (
        !lane &&
        !plan.serverThreshold &&
        settings.summarize &&
        !assistant.error &&
        !controller.signal.aborted
      ) {
        void maybeCompact(conv, models.maxCtx, replyReserve(settings.maxTokens), (c) =>
          chat.persistNow(c),
        )
      }
    }
  }

  /** The lanes a compare send fans out to (see `runningLanes`). */
  function laneModels(conv: Conversation): string[] {
    return runningLanes(conv, models)
  }

  /** Fan one user turn out to every lane: one grouped assistant message per
   *  model, all streaming concurrently. `contended` is stamped only on LOCAL
   *  lanes that actually shared the GPU with another local lane - one local
   * model racing cloud lanes had the card to itself. */
  async function fanOut(
    conv: Conversation,
    lanes: string[],
    at?: string | null,
  ): Promise<void> {
    // Warm every lane's caps before any lane plans: min-of-lanes needs each
    // lane's context window, and the lanes launch concurrently - a lane that
    // planned before a sibling's caps arrived would use the wrong minimum.
    await Promise.all(lanes.map((m) => models.capsFor(m)))
    const isLocal = (id: string) => !models.models.find((m) => m.id === id)?.cloud
    const gpuShared = lanes.filter(isLocal).length >= 2
    const gid = uid()
    // Every lane shares one parent, which is what makes the fan-out a single
    // step in the tree rather than N rival branches. `at` is passed by
    // regenerate so a re-rolled compare block becomes a second run alongside
    // the first instead of hanging off it.
    const parent = at !== undefined ? at : (tipId(conv) ?? null)
    const turns = lanes.map((m) =>
      chat.addMessage(
        conv,
        {
          id: uid(),
          role: 'assistant',
          content: [{ type: 'text', text: '' }],
          streaming: true,
          model: m,
          group: gid,
          createdAt: Date.now(),
        },
        parent,
      ),
    )
    await Promise.all(
      turns.map((a) => run(conv, a, false, true, gpuShared && isLocal(a.model ?? ''))),
    )
  }

  /** Send a user turn (content parts) and stream the assistant reply - or
   *  replies: with compare armed, the one user turn fans out to every lane. */
  // ── document runs: one request per PAGE  ──────────────────────
  // A document parser reads pages independently - that is how the official
  // pipelines run (per page, even per block), each page gets the model's full
  // budget, and every page's response carries its own regions, which the
  // single-request wire cannot say. The engine's batched lane eats the
  // concurrency. Page order here must match DocumentPages' stack order
  // (images in message order, then the PDF's pages) - index i is page i.

  /** One page's send payload: a data-URI image + the shared instruction. */
  interface PageSend {
    dataUri: string | null
    /** why there is no image to send (TIFF until the server raster lands) */
    skip?: string
    /** rendered pixel size - the figure-crop math needs the page's aspect */
    w?: number
    h?: number
  }

  /** Rasterize a PDF's pages client-side at the MODEL's pixel budget - the
   *  server never sees the PDF on this path, so the client owes the pages the
   *  same resolution the server-side raster route would have given them. */
  async function rasterPdfPages(part: FilePart, maxPixels: number, cap: number): Promise<PageSend[]> {
    // the user's from-to range (the same control vision-model PDFs have) is
    // the honest answer to a 400-page PDF - only the chosen pages raster,
    // fan out, and appear in the pane
    if (!part.attachmentId) return [{ dataUri: null, skip: 'original file is not stored' }]
    const res = await fetch(attachmentsApi.url(part.attachmentId))
    if (!res.ok) return [{ dataUri: null, skip: `couldn't fetch the PDF (${res.status})` }]
    const buf = await res.arrayBuffer()
    const engine = await pdfEngine()
    const doc = engine.plugins.get<import('@truespar/lector-core').DocumentCapability>('document')
    const render = engine.plugins.get<import('@truespar/lector-core').RenderCapability>('render')
    const handle = await doc.load(buf)
    try {
      const out: PageSend[] = []
      const [p0, p1] = pageRangeBounds(part.pageRange, handle.pageCount)
      const n = Math.min(p1 - p0, cap)
      for (let i = p0; i < p0 + n; i++) {
        const size = handle.pageSizes[i]
        const aspect = size ? size.width / size.height : 0.75
        const w = Math.round(Math.sqrt(maxPixels * aspect))
        const h = Math.round(w / aspect)
        const bmp = await render.renderPage(handle.id, i, w, h)
        const canvas = document.createElement('canvas')
        canvas.width = bmp.width
        canvas.height = bmp.height
        canvas.getContext('2d')?.drawImage(bmp, 0, 0)
        bmp.close()
        out.push({ dataUri: canvas.toDataURL('image/jpeg', 0.92), w: canvas.width, h: canvas.height })
      }
      if (p1 - p0 > cap) {
        out.push({ dataUri: null, skip: `page cap: first ${cap} of ${p1 - p0} sent` })
      }
      return out
    } finally {
      try {
        await doc.close(handle.id)
      } catch {
        /* ignore */
      }
    }
  }

  /** The document-parser fan-out. Returns false when this send is not a doc
   *  run (not a parser, or nothing document-shaped staged) so the ordinary
   *  path takes it. */
  async function maybeDocRun(conv: Conversation, at?: string | null): Promise<boolean> {
    const modelId = conv.model
    const caps = await models.capsFor(modelId)
    // Two kinds of model take this path. A document PARSER always does. A
    // chat vision model with task tags (granite-vision) does for a TASK turn
    // over a document with several pages: its template rebuilds a tagged turn
    // as IBM's canned prompt around ONE image slot, so a multi-page document
    // in one request renders one slot for N pages and the runner refuses it
    // (rightly - dropping N-1 pages would be worse). The tasks are single-page
    // by design, and one page per request is how IBM's own document pipeline
    // (Docling) drives the model. A single image stays on the ordinary path.
    const taskModel = !caps.docParser && (caps.taskTags?.length ?? 0) > 0
    if (!caps.docParser && !taskModel) return false
    // The document is STICKY (lib/docrun.ts): the most recent doc-bearing
    // turn stands; a text-only follow-up re-runs it with the new instruction
    // ("now as markdown") - the decoders cannot chat, the conversation can.
    // SELECTION = TARGET (lib/docrun.ts): the selected document is what the
    // chips read. A document the user just sent wins over a stale selection
    // and selects itself - the two legal ways selection moves are this and a
    // pane tab click.
    // RASTER documents only: a .docx is a pane document (scriptor renders it)
    // but nothing turns it into pages for a decoder, so it must not be picked
    // up here as a document with zero pages - its text rides the ordinary
    // attachment path instead.
    const all = docContexts(conv).filter(isRasterDoc)
    if (!all.length) return false
    const user = [...activeMessages(conv)].reverse().find((m) => m.role === 'user')
    if (!user) return false
    const ctx = all.find((c) => c.source === user) ?? rasterContext(conv)
    if (!ctx) return false
    const { images, pdf } = ctx
    const instruction = user.content
      .filter((p): p is Extract<ContentPart, { type: 'text' }> => p.type === 'text')
      .map((p) => p.text)
      .join('\n')
      .trim()
    // The task-model gate (see above): a task tag, and more than one page to
    // send - a PDF (any page count; one page through here is one request) or
    // several images. Anything else is a conversation and takes the single
    // request the ordinary path builds.
    if (taskModel) {
      const multiPage = pdf != null || images.length > 1
      if (!isTaskTurn(instruction, caps.taskTags) || !multiPage) return false
    }
    if (conv.activeDocId !== ctx.source.id) {
      conv.activeDocId = ctx.source.id
      chat.persist(conv)
    }

    // `at` carries the branch point on a REGENERATE: the answer being
    // re-rolled is still in the tree now (it used to be spliced away), so
    // without this the new run would hang off the old answer instead of
    // standing beside it as an alternative.
    const assistant = chat.addMessage(
      conv,
      {
        id: uid(),
        role: 'assistant',
        content: [{ type: 'text', text: '' }],
        streaming: true,
        model: modelId,
        docRun: { pages: [] },
        createdAt: Date.now(),
      },
      at,
    )
    const endpoint = models.responsesUrl(modelId)
    if (!endpoint) {
      assistant.error = `No running model serves "${modelId}" - start one in the Manager first.`
      assistant.streaming = false
      chat.persistNow(conv)
      return true
    }

    // assemble page payloads: images in message order, then the PDF's pages
    const maxPixels = caps.visionBudget?.max_pixels ?? 1_500_000
    const sends: PageSend[] = []
    const fileData = await resolveFileData(conv)
    for (const p of images) {
      const tiff = (p.mime ?? '').includes('tiff') || /\.tiff?$/i.test(p.name ?? '')
      if (tiff) {
        sends.push({ dataUri: null, skip: 'TIFF pages need the server raster (not built yet)' })
        continue
      }
      const uri = (p.attachmentId && fileData.get(p.attachmentId)) || p.modelUrl || p.dataUrl
      sends.push(
        uri
          ? { dataUri: uri, w: p.width, h: p.height }
          : { dataUri: null, skip: 'original image is not stored' },
      )
    }
    if (pdf) {
      try {
        sends.push(...(await rasterPdfPages(pdf, maxPixels, models.pdfMaxPages || 40)))
      } catch (e) {
        sends.push({ dataUri: null, skip: e instanceof Error ? e.message : String(e) })
      }
    }
    const pages: DocPage[] = sends.map((s) =>
      s.skip
        ? { state: 'error', text: '', note: s.skip }
        : { state: 'queued', text: '' },
    )
    assistant.docRun = { pages }
    // The figure-crop source: with PDFs now shown in lector's viewer (which
    // renders no plain <img> stack), the fan-out raster is the one place the
    // page bitmaps exist - publish them here (images publish theirs too; the
    // pane's image path overwrites with identical data harmlessly).
    pageImages.set(
      ctx.source.id,
      sends.map((s) => (s.dataUri ? { src: s.dataUri, w: s.w ?? 0, h: s.h ?? 0 } : null)),
    )
    // Mutate only the reactive proxies the template reads: `pages` above is
    // the raw array, and writing raw objects renders nothing until some
    // other reactive write forces a pass - every page then appeared at once
    // with its states stuck on Queued. Same trap run()'s
    // capture comment records. (Raw reads stay fine - the proxy mutates the
    // same underlying objects.)
    const live = assistant.docRun.pages
    if (!live.some((p) => p.state === 'queued')) {
      assistant.streaming = false
      chat.persistNow(conv)
      return true
    }

    const started = performance.now()
    tele.beginCapture()
    assistant.run = {
      model: modelId,
      params: { ...conv.params },
      tools: [],
      at: Date.now(),
    }
    let inTok = 0
    let outTok = 0

    async function runPage(i: number): Promise<void> {
      const page = live[i]
      const s = sends[i]
      if (!s.dataUri) return
      const controller = new AbortController()
      beginStream(conv!.id, controller)
      const t0 = performance.now()
      page.state = 'reading'
      // rAF-throttled delta apply, like run(): one markdown re-render per
      // frame, not per token
      let buf = ''
      let scheduled = false
      const apply = () => {
        scheduled = false
        page.text = buf
      }
      const schedule = () => {
        if (!scheduled) {
          scheduled = true
          requestAnimationFrame(apply)
        }
      }
      try {
        const content: Record<string, unknown>[] = [
          { type: 'input_image', image_url: s.dataUri },
        ]
        // A fixed-vocabulary family (advertised reading modes) takes ORGANIZED
        // requests only: the ocr object below is the whole ask, and free text
        // just garbles its decoder ("make a markdown" -> word salad, observed
        // live). Families without the advertisement keep the text -
        // paddleocr's task prompts are text until it advertises them.
        if (instruction && !caps.ocr) content.push({ type: 'input_text', text: instruction })
        const body: Record<string, unknown> = {
          model: modelId,
          input: [{ type: 'message', role: 'user', content }],
          stream: true,
          // the trust layer: per-token logprobs ride every doc
          // run so unsure words can be marked - the spec include, no
          // extension
          include: ['message.output_text.logprobs'],
        }
        if (settings.maxTokens != null && settings.maxTokens > 0) {
          body.max_output_tokens = settings.maxTokens
        }
        // the reading-mode object, per page - same three gates as the single
        // request path; 'multipage' cannot mean anything to one page
        if (caps.ocr && (conv!.ocrMode || conv!.ocrRegions)) {
          const o: Record<string, unknown> = {}
          if (
            conv!.ocrMode &&
            conv!.ocrMode !== 'multipage' &&
            caps.ocr.modes.includes(conv!.ocrMode)
          ) {
            o.mode = conv!.ocrMode
          }
          if (conv!.ocrRegions && caps.ocr.grounding) o.grounding = true
          if (Object.keys(o).length) body.ocr = o
        }
        let res = await fetch(endpoint!, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        })
        if (res.status === 400 && body.include) {
          // a runner from before the logprobs include rejects the unknown
          // field - read without confidence rather than not at all
          delete body.include
          res = await fetch(endpoint!, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
          })
        }
        if (!res.ok || !res.body) {
          page.state = 'error'
          page.note = await friendlyHttpError(res)
          return
        }
        const lps: LogprobEntry[] = []
        for await (const data of readSse(res.body)) {
          let ev: ResponseEvent
          try {
            ev = JSON.parse(data) as ResponseEvent
          } catch {
            continue
          }
          if (ev.type === 'response.output_text.delta') {
            buf += ev.delta ?? ''
            const dl = (ev as { logprobs?: unknown }).logprobs
            if (Array.isArray(dl)) {
              for (const e of dl) {
                const { token, logprob } = (e ?? {}) as { token?: unknown; logprob?: unknown }
                if (typeof token === 'string' && typeof logprob === 'number') {
                  lps.push({ token, logprob })
                }
              }
            }
            schedule()
          } else if (ev.type === 'response.completed' || ev.type === 'response.incomplete') {
            inTok += ev.response?.usage?.input_tokens ?? 0
            outTok += ev.response?.usage?.output_tokens ?? 0
            const meta = ev.response?.ocr ? ocrMetaFromWire(ev.response.ocr) : undefined
            if (meta?.regions?.length) page.regions = meta.regions
            // a single-page run keeps the old echo home too, so the read-as
            // facts and the flat fallback view stay populated
            if (pages.length === 1 && meta) assistant.ocr = meta
            if (ev.type === 'response.incomplete') page.note = 'cut off at the token cap'
          } else if (ev.type === 'response.failed') {
            page.state = 'error'
            page.note = ev.response?.error?.message ?? 'the model failed to respond'
          }
        }
        apply()
        if (lps.length) page.words = wordsFromLogprobs(lps)
        // the server's terminal region parse is authoritative; when it sent
        // none but the raw stream carries markup (a runner from before the
        // spotting parse), the client mirror keeps the boxes alive
        if (!page.regions?.length) {
          const mirrored = parseRegionsLive(page.text)
          if (mirrored.length) page.regions = mirrored
        }
        if (page.state !== 'error') {
          // the degeneration guard: a page that collapsed into a loop is
          // marked for review instead of standing as a clean answer
          const ratio = await degenerationRatio(page.text)
          if (ratio > DEGENERATION_THRESHOLD) {
            page.state = 'review'
            page.note = 'output degenerated into repetition - the page may be too low-resolution'
          } else {
            page.state = 'done'
          }
        }
      } catch (e) {
        apply()
        if (controller.signal.aborted) {
          page.state = page.text ? 'done' : 'error'
          page.note = 'stopped'
          assistant.stopped = true
        } else {
          page.state = 'error'
          page.note = e instanceof Error ? e.message : String(e)
        }
      } finally {
        page.ms = Math.round(performance.now() - t0)
        endStream(conv!.id, controller)
      }
    }

    // Sequential deliberately: the reading experience is a
    // narrative - the viewer follows one page being read, its extraction
    // streams into its own section, then the next page. Three pages
    // streaming at once made the result column jump between three places.
    // The engine's batched lane still earns its keep across users; this
    // turn's pages go in order.
    const queue = pages.map((_, i) => i).filter((i) => pages[i].state === 'queued')
    try {
      for (const i of queue) {
        await runPage(i)
        if (assistant.stopped) break
      }
    } finally {
      // the joined text is the turn's plain content: exports, copy, and the
      // history a follow-up would replay all read it
      const joined = pages
        .map((p, i) => (pages.length > 1 ? `## Page ${i + 1}\n\n${p.text}` : p.text))
        .join('\n\n')
      const tp = assistant.content[0]
      if (tp && tp.type === 'text') tp.text = joined
      assistant.usage = {
        promptTokens: inTok,
        completionTokens: outTok,
        ms: Math.round(performance.now() - started),
      }
      assistant.streaming = false
      const gpu = tele.endCapture()
      if (gpu && assistant.run) assistant.run.gpu = gpu
      chat.persistNow(conv)
    }
    return true
  }

  async function send(
    parts: ContentPart[],
    opts?: { lane?: string; auto?: boolean },
  ): Promise<void> {
    const conv = chat.active
    if (!conv || isStreaming.value || parts.length === 0) return
    // A laned send in compare runs only the owning lane; the message itself
    // is stamped so the other lanes never see it in their history either.
    const lane = opts?.lane
    const lanes = laneModels(conv)
    const targeted = lane && lanes.length >= 2 ? lanes.filter((l) => l === lane) : null
    if (targeted && !targeted.length) return // owner not armed - nothing to run
    chat.addMessage(conv, {
      id: uid(),
      role: 'user',
      content: parts,
      lane: targeted ? lane : undefined,
      auto: opts?.auto || undefined,
      createdAt: Date.now(),
    })
    chat.maybeTitle(conv)
    if (lanes.length >= 2) {
      await fanOut(conv, targeted ?? lanes)
      return
    }
    // a document parser's send fans out one request per page
    if (await maybeDocRun(conv)) return
    // capture the REACTIVE element the store now holds - streaming into the
    // raw object would mutate data without re-rendering
    const assistant = chat.addMessage(conv, {
      id: uid(),
      role: 'assistant',
      content: [{ type: 'text', text: '' }],
      streaming: true,
      model: conv.model,
      createdAt: Date.now(),
    })
    await run(conv, assistant)
  }

  /** Answer the same question again, as an ALTERNATIVE rather than a
   *  replacement: the new turn becomes a sibling of the answer it re-rolls, so
   *  both survive and the `< 2/3 >` control switches between them. A compare
   *  group at the tail re-rolls as a whole - every lane again, as one new step.
   *
   *  This used to `splice` the old answer out of the array, which is why a
   *  regenerate you did not like used to be unrecoverable. */
  async function regenerate(): Promise<void> {
    const conv = chat.active
    if (!conv || isStreaming.value) return
    // The tail STEP of the branch on screen, which for a compare block is all
    // of its lanes at once.
    const steps = activeSteps(conv)
    const tail = steps[steps.length - 1]
    if (!tail || stepAnchor(tail).role !== 'assistant') return
    // Every re-roll hangs where the answer it replaces hangs: from the
    // question, not from the answer.
    const at = stepAnchor(tail).parentId ?? null
    const lanes = laneModels(conv)
    if (lanes.length >= 2) {
      await fanOut(conv, lanes, at)
      return
    }
    // a document parser regenerates the same way it sent: per page
    if (await maybeDocRun(conv, at)) return
    const assistant = chat.addMessage(
      conv,
      {
        id: uid(),
        role: 'assistant',
        content: [{ type: 'text', text: '' }],
        streaming: true,
        model: conv.model,
        createdAt: Date.now(),
      },
      at,
    )
    await run(conv, assistant)
  }

  /** Ask a question differently: the edited turn becomes a SIBLING of the
   *  original, so the thread you are replacing stays reachable behind the
   *  `< 2/3 >` control instead of being overwritten.
   *
   *  Everything downstream of the original - its answer and everything that
   *  followed - stays exactly where it is, on the branch it belongs to. */
  async function editAndResend(messageId: string, parts: ContentPart[]): Promise<void> {
    const conv = chat.active
    if (!conv || isStreaming.value || parts.length === 0) return
    const original = conv.messages.find((m) => m.id === messageId)
    if (!original || original.role !== 'user') return

    const edited = chat.addMessage(
      conv,
      {
        id: uid(),
        role: 'user',
        content: parts,
        // A laned turn keeps its lane: it belonged to one compare column and
        // the edit does not change whose question it is.
        lane: original.lane,
        createdAt: Date.now(),
      },
      original.parentId ?? null,
    )
    chat.maybeTitle(conv)

    const lanes = laneModels(conv)
    const targeted = edited.lane && lanes.length >= 2 ? lanes.filter((l) => l === edited.lane) : null
    if (lanes.length >= 2) {
      await fanOut(conv, targeted ?? lanes)
      return
    }
    if (await maybeDocRun(conv)) return
    const assistant = chat.addMessage(conv, {
      id: uid(),
      role: 'assistant',
      content: [{ type: 'text', text: '' }],
      streaming: true,
      model: conv.model,
      createdAt: Date.now(),
    })
    await run(conv, assistant)
  }

  /** Continue a reply that hit the max-tokens cap, appending to the same turn. */
  async function continueLast(): Promise<void> {
    const conv = chat.active
    if (!conv || isStreaming.value) return
    // The newest assistant turn on this BRANCH - a longer answer sitting on a
    // branch you switched away from is not the one the button is offering to
    // finish.
    const path = activeMessages(conv)
    let live: Message | undefined
    for (let i = path.length - 1; i >= 0; i--) {
      if (path[i].role === 'assistant') {
        live = path[i]
        break
      }
    }
    if (!live) return
    if (live.incomplete !== 'length') return
    live.streaming = true
    live.incomplete = undefined
    live.stopped = false
    await run(conv, live, true)
  }

  /** Stop the chat on screen - every lane of it, and nothing else. */
  function stop(): void {
    const live = chat.activeId ? aborts.get(chat.activeId) : undefined
    for (const c of [...(live ?? [])]) c.abort()
  }

  return { isStreaming, send, regenerate, editAndResend, continueLast, stop }
}
