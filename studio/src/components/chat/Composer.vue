<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch, watchEffect } from 'vue'
import { useRouter } from 'vue-router'
import { reasoningOptions as buildReasoningOptions, reasoningChoice as chooseReasoning, reasoningLabel as reasoningItemLabel, samplerIsSet } from '@/lib/composer-policy'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import { Placeholder } from '@tiptap/extensions'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import { takesTurns, useModelsStore } from '@/stores/models'
import { useFleetStore } from '@/stores/fleet'
import { useConnectorsStore, type Connector } from '@/stores/connectors'
import {
  builtinKey,
  connectorKey,
  serverKey,
  useMcpToolsStore,
  type McpToolInfo,
} from '@/stores/mcpTools'
import { activeMessages } from '@/lib/tree'
import { ARTIFACTS_LABEL, toolSelection } from '@/composables/useChatStream'
import { DICTATION_IDLE_MS, useMicTranscribe } from '@/composables/useMicTranscribe'
import { audioPolicy, type MicMode } from '@/lib/audio-policy'
import { imagePolicy } from '@/lib/image-policy'
import { useAudioDevices } from '@/composables/useAudioDevices'
import { useLiveTurn } from '@/composables/useLiveTurn'
import { RECORD_MAX_S, useRecorder } from '@/composables/useRecorder'
import { Dictation, appendDictated, setGhost } from '@/lib/dictation'
import { audioDuration, isAudioFile } from '@/lib/transcribe'
import { holdReload } from '@/lib/reload'
import type { ToolSelection } from '@/types/chat'
import { changeToolPicker, pickerGroupState, pickerToolChecked, toolTermHits, type PickerAction } from '@/lib/tool-picker'
import { friendlyModelName } from '@/lib/model-caps'
import { speculationBadge } from '@/lib/model-name'
import {
  type DocOpts,
  isAttachableFile,
  isGraphFile,
  isImageFile,
  isPdfFile,
  isTiffFile,
  isTooLarge,
  isUnreadableImage,
  MAX_FILE_MB,
  pagesParam,
} from '@/lib/attachments'
import {
  DETAIL_HINT,
  DETAIL_LABEL,
  DETAIL_ORDER,
  formatTokens,
  tokensFor,
  type ImageDetail,
} from '@/lib/vision'
import { taskHint, taskLabel } from '@/lib/tasks'
import { rasterContext } from '@/lib/docrun'
import { OCR_AUTO, ocrModeHint, ocrModeLabel } from '@/lib/ocr'
import { contextTokens } from '@/lib/tokens'
import { fmtCost, fmtFileSize } from '@/lib/format'
import { pdfPageCount } from '@/lib/pdf'
import { LANGUAGE_AUTO, askedLanguage, languageOptions, localeLanguage } from '@/lib/languages'
import Icon from '@/components/Icon.vue'
import Checkbox from '@/components/ui/Checkbox.vue'
import Select from '@/components/ui/Select.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import ContextMeter from '@/components/chat/ContextMeter.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import Popover from '@/components/ui/Popover.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuLabel from '@/components/ui/MenuLabel.vue'
import MenuSeparator from '@/components/ui/MenuSeparator.vue'
import NumberField from '@/components/ui/NumberField.vue'
import SpeechModels from './SpeechModels.vue'
import { DICTATION_SETUP, microphoneMenu } from '@/lib/speech-models'
import SystemPromptPanel from './SystemPromptPanel.vue'
import SamplerMenu from './SamplerMenu.vue'
import ImageOptions from './ImageOptions.vue'
import AudioPlayer from './AudioPlayer.vue'

// `docked` = we are in a chat, floating over the thread rather than sitting as
// the start page's centred hero. withDefaults is required, not tidiness: an
// unbound `docked?: boolean` reads as undefined, and a bare truthiness check
// then silently never fires (the model-stamp bug).
const props = withDefaults(defineProps<{ busy?: boolean; docked?: boolean }>(), {
  busy: false,
  docked: false,
})
const emit = defineEmits<{
  submit: [text: string]
  stop: []
  /** Recording is about to start and will write straight into the thread, so
   *  the host must make the conversation real first - the same thing a send
   *  does, minus the content. Handled synchronously, before the first word. */
  listen: []
}>()

// Staged files are owned by the host (ChatView) so a drop anywhere in the chat
// region can feed them; the composer renders + edits the same array.
const files = defineModel<File[]>('files', { default: () => [] })

const chat = useChatStore()
const models = useModelsStore()
const settings = useSettingsStore()
const fleet = useFleetStore()
const router = useRouter()

// ── Server tools (per-model config, advertised on /api/server) ───────────
// The composer's toggles gate the USAGE of what the current model's endpoint
// actually supplies. Configuring tools happens on the model's Start/Edit
// page in the Manager - never here.
const activeModelId = computed(() => chat.active?.model || models.currentId || '')
const activeCaps = computed(() => models.caps[activeModelId.value])
// Keyed on the caps ENTRY as well as the id: the fleet's start path calls
// invalidateCaps() when the model comes up - clear only, no refetch - and an
// id-keyed watcher never refired, so the composer kept pre-start controls
// (no thinking menu on the 9B) until a full page reload. The entry going
// undefined is the refetch signal; capsFor's own retry drives the window
// where the endpoint is still coming up.
watch(
  [activeModelId, () => models.caps[activeModelId.value]],
  ([id, entry]) => {
    if (id && !entry) void models.capsFor(id)
  },
  { immediate: true },
)

// ── The tool picker (the plug) ───────────────────────────────────────────
// Two states per chat: "All tools" (every server tool the endpoint supplies,
// plus the connectors switched on here) or a custom pick - fuzzy-searched
// across every tool each source exposes, down to single tools. Single-tool
// picks ride as the request's allowed_tools filter. Custom is a BUILDER:
// under All only the All row (and armed connectors) show checked, and
// clicking a row starts the subset with that row - the user who searched
// for a tool means "give me this", not "everything minus this". All-minus-
// one stays reachable: pick the whole server, then uncheck the one.
const connectors = useConnectorsStore()
const mcpTools = useMcpToolsStore()
onMounted(() => void connectors.ensure())
// System ("every server") connectors ride as the endpoint's own tools - the
// picker lists only personal ones, or the same label would appear twice.
const personalConnectors = computed(() => connectors.list.filter((c) => !c.system))
const chatConnectorIds = computed(() => chat.active?.connectorIds ?? [])
const activeConnectorCount = computed(
  () =>
    chatConnectorIds.value.filter((id) => {
      const c = connectors.byId(id)
      return !!c && !c.system
    }).length,
)

const pickerOpen = ref(false)
const toolQuery = ref('')
const selection = computed<ToolSelection>(() =>
  chat.active ? toolSelection(chat.active) : { mode: 'all' },
)
const customPickCount = computed(() =>
  selection.value.mode === 'custom' ? selection.value.picks.length : 0,
)
// Lit when the next send will actually CARRY tools. In all-tools mode that is
// now unconditional: artifacts is a first-party group the Studio attaches to
// every request, so there is always at least one tool armed. The old test
// counted only the endpoint's advertised mcp_servers plus armed connectors -
// neither of which artifacts is - so the button sat dark on a box with no
// connectors while artifact tools rode every turn. Custom
// mode still reads its picks, so deselecting everything turns it off.
const plugActive = computed(() =>
  selection.value.mode === 'all'
    ? true
    : customPickCount.value > 0,
)

/** One row group in the picker: a server tool (resolved through the
 *  endpoint's config) or a personal connector (resolved through the
 *  library). `key` addresses its cached tool listing. */
interface PickerGroup {
  key: string
  label: string
  connector?: Connector
  /** A server the manager hosts itself - no endpoint config, no library row. */
  builtin?: string
}
const activePort = computed(() => models.portFor(activeModelId.value))
const pickerGroups = computed<PickerGroup[]>(() => {
  // Artifacts lead: they are the manager's own, present on every lane
  // (local or cloud) rather than something an endpoint has to advertise.
  const groups: PickerGroup[] = [
    { key: builtinKey(ARTIFACTS_LABEL), label: ARTIFACTS_LABEL, builtin: ARTIFACTS_LABEL },
  ]
  const port = activePort.value
  if (port) {
    for (const label of activeCaps.value?.mcpServers ?? []) {
      groups.push({ key: serverKey(port, label), label })
    }
  }
  const served = activeCaps.value?.mcpServers ?? []
  for (const c of personalConnectors.value) {
    if (served.includes(c.label)) continue
    groups.push({ key: connectorKey(c.id), label: c.label, connector: c })
  }
  return groups
})

// Opening the picker re-asks the endpoint what it advertises (fresh caps -
// a connector scoped onto it after page load must show up, not wait for a
// hard refresh), then kicks off (or reuses) every group's tool listing; the
// probe is the manager's, results cache for the session, and a group whose
// listing fails stays pickable as a whole server.
watch(pickerOpen, async (open) => {
  if (!open) return
  toolQuery.value = ''
  if (activeModelId.value) await models.capsFor(activeModelId.value, true)
  const port = activePort.value
  for (const g of pickerGroups.value) {
    if (g.builtin) mcpTools.ensureBuiltin(g.builtin)
    else if (g.connector) mcpTools.ensureConnector(g.connector.id)
    else if (port) mcpTools.ensureServer(port, g.label)
  }
})

// Search matching lives in the shared web/native picker policy.
const termHits = toolTermHits

interface PickerRow {
  group: PickerGroup
  status?: 'loading' | 'ok' | 'error'
  error?: string
  total: number
  tools: McpToolInfo[]
}
const pickerRows = computed<PickerRow[]>(() => {
  const terms = toolQuery.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
  const rows: PickerRow[] = []
  for (const g of pickerGroups.value) {
    const listing = mcpTools.get(g.key)
    const all = listing?.status === 'ok' ? listing.tools : []
    const labelHit = terms.every((t) => termHits(t, g.label))
    const tools =
      terms.length && !labelHit
        ? all.filter((tool) => terms.every((t) => termHits(t, tool.name, tool.description)))
        : all
    if (terms.length && !labelHit && !tools.length) continue
    rows.push({
      group: g,
      status: listing?.status,
      error: listing?.error,
      total: all.length,
      tools,
    })
  }
  return rows
})

function pickerState() { return { selection: selection.value, connectorIds: [...chatConnectorIds.value] } }
function policyGroup(g: PickerGroup) {
  return { label: g.label, connectorId: g.connector?.id, tools: mcpTools.get(g.key)?.status === 'ok' ? mcpTools.get(g.key)!.tools.map(t => t.name) : undefined }
}
function chooseTools(action: PickerAction, g?: PickerGroup): void {
  const c = chat.active
  if (!c) return
  const next = changeToolPicker(pickerState(), g && policyGroup(g),
    personalConnectors.value.map(c => ({ label: c.label, connectorId: c.id })), action)
  c.toolSelection = next.selection
  c.connectorIds = next.connectorIds
  chat.persist(c)
}
function groupState(g: PickerGroup): 'all' | 'some' | 'none' { return pickerGroupState(pickerState(), policyGroup(g)) }
function groupChecked(g: PickerGroup): boolean | 'indeterminate' {
  const s = groupState(g)
  return s === 'all' ? true : s === 'some' ? 'indeterminate' : false
}
function toolChecked(g: PickerGroup, name: string): boolean { return pickerToolChecked(pickerState(), policyGroup(g), name) }
function selectAllTools(): void { chooseTools({ kind: 'all' }) }
function toggleGroup(g: PickerGroup): void { chooseTools({ kind: 'group', label: g.label }, g) }
function toggleTool(g: PickerGroup, name: string): void { chooseTools({ kind: 'tool', label: g.label, tool: name }, g) }

function openConnectorsPage(): void {
  pickerOpen.value = false
  void router.push({ name: 'connectors' })
}

// ── File metadata ────────────────────────────────────────────────────────
// Per-chat, on by default (the server's file_metadata "full"): attachment
// metadata rides into the prompt with the content - document title/author/
// dates for PDFs and Word docs, capture time/camera/GPS for photos. The
// toggle appears once any attachment is in play - staged in the tray or
// already sent in this chat - the moment the choice is relevant. (It first
// shipped counting only documents, so a photos-only chat had no way to see
// or flip the setting that governed its GPS line.)
const hasDocs = computed(
  () =>
    files.value.length > 0 ||
    (chat.active ? activeMessages(chat.active) : []).some((m) =>
      m.content.some((p) => p.type === 'file' || p.type === 'image'),
    ),
)
const fileMeta = computed<boolean>({
  get: () => chat.active?.fileMetadataEnabled ?? true,
  set: (v) => {
    const c = chat.active
    if (!c) return
    c.fileMetadataEnabled = v
    chat.persist(c)
  },
})

// ── Forensics ──────────────────────────────────────────────────────────────
// Per-chat override of the endpoint's [forensics] auto default. Only offered on
// an endpoint that advertises forensics AND serves vision - forensics is
// VLM-coupled, so the findings are inert without a vision tower. Undefined =
// follow the endpoint default; the menu shows a checkmark on the EFFECTIVE state
// and writes an explicit override on click.
const forensicsAvailable = computed(() => {
  const f = activeCaps.value?.forensics
  return !!f && f.vision
})
// Forensics is OPT-IN here, always - never inherited from the endpoint's
// `[forensics] auto`. Enabling it on the endpoint makes it AVAILABLE in this
// menu; it does not switch it on. Running signal-level forensics over someone's
// attachments is not something to start doing because a config file said so: it
// re-reads the original bytes, and its findings are injected ahead of the model
// and steer the answer. That is a choice the person in the chat makes per
// conversation, not a default they inherit.
const forensicsOn = computed<boolean>({
  get: () => chat.active?.forensicsEnabled === true,
  set: (v) => {
    const c = chat.active
    if (!c) return
    c.forensicsEnabled = v
    chat.persist(c)
  },
})
// Whether to show the whole enrichment control at all: any attachment in play,
// and not in a mode where it doesn't apply.
const showEnrichment = computed(() => hasDocs.value && !audioMode.value && !docParser.value)

// ── per-file document settings ───────────────────────────────────────────
// The settings belong to the FILE, not the prompt: each
// staged PDF/TIFF chip carries its own menu, and the choices ride the wire
// as part-level `pdf_mode` / `pages` on that attachment only. Default = the
// server's auto route (page images on a vision model, extracted text
// otherwise); text extraction and page ranges are the per-file opt-ins.
const docOpts = ref(new Map<File, DocOpts>())
function docOptsFor(f: File): DocOpts {
  return docOpts.value.get(f) ?? {}
}
function setDocOpts(f: File, patch: DocOpts): void {
  docOpts.value = new Map(docOpts.value).set(f, { ...docOptsFor(f), ...patch })
}
/** Keys typed in the range inputs stay out of the menu's typeahead/nav -
 *  but Escape still closes it. */
function stopUnlessEscape(e: KeyboardEvent): void {
  if (e.key !== 'Escape') e.stopPropagation()
}
// Page counts for the range picker: PDFs count client-side (pdfium in the
// browser); TIFFs only the server can count - /api/extract returns `pages`
// and the count fills in when a model server is up.
const pageCount = ref(new Map<File, number>())
const countAsked = new WeakSet<File>()
async function learnPageCount(f: File): Promise<void> {
  if (countAsked.has(f)) return
  countAsked.add(f)
  try {
    if (isPdfFile(f)) {
      const n = await pdfPageCount(await f.arrayBuffer())
      if (n > 0) pageCount.value = new Map(pageCount.value).set(f, n)
      return
    }
    const url = models.extractUrl(activeModelId.value)
    if (!url) {
      countAsked.delete(f)
      return
    }
    const b64 = await new Promise<string>((resolve, reject) => {
      const r = new FileReader()
      r.onload = () => resolve(String(r.result ?? '').split(',')[1] ?? '')
      r.onerror = () => reject(r.error)
      r.readAsDataURL(f)
    })
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ data: b64, filename: f.name }),
    })
    if (!res.ok) return
    const j = await res.json()
    if (Number.isInteger(j.pages) && j.pages > 0) {
      pageCount.value = new Map(pageCount.value).set(f, j.pages)
    }
  } catch {
    /* absence over invention - the picker just shows no total */
  }
}
watch(
  files,
  (list) => {
    for (const f of list) if (isPdfFile(f) || isTiffFile(f)) void learnPageCount(f)
  },
  { immediate: true },
)
/** The chip trigger's text: the current page selection (with the real count
 *  when known) plus the priced estimate. */
function docSummary(f: File): string {
  const o = docOptsFor(f)
  const n = pageCount.value.get(f)
  const range = pagesParam(o)
  const sel = range
    ? `Pages ${range.replace(/-$/, `-${n ?? 'end'}`)}`
    : n
      ? `${n} page${n === 1 ? '' : 's'}`
      : 'All pages'
  const est = estLabel(f)
  return est ? `${sel} · ${est}` : sel
}

// ── Web search ───────────────────────────────────────────────────────────
// Per-chat, off by default. The integration itself is the MODEL's config -
// a server without it gets an honest hand-off to its Edit page.
const webAvailable = computed(() => models.webSearchFor(activeModelId.value))
// Default on where the endpoint supplies it: the server owner enabled web
// search for this model deliberately, and users forget the composer toggle
// exists. Explicitly switching it off is still stored.
const webOn = computed(() => webAvailable.value && (chat.active?.webSearchEnabled ?? true))
const webLabel = computed(() => {
  if (!webAvailable.value)
    return "This model's server has no web search - add it on its Edit page in the Manager"
  return webOn.value
    ? 'Web search on - the model can search the web'
    : 'Web search off - the model answers from what it knows'
})
function toggleWeb(): void {
  if (!webAvailable.value) {
    const port = models.portFor(activeModelId.value)
    if (port) void router.push({ name: 'server-edit', params: { port: String(port) } })
    return
  }
  const c = chat.active
  if (!c) return
  c.webSearchEnabled = !webOn.value
  chat.persist(c)
}

// System prompt (per-chat instructions) - edited in a modal opened from the toolbar.
const sysOpen = ref(false)
const hasSystemPrompt = computed(() => !!chat.active?.systemPrompt?.trim())

// ── Compare (multi-model send) ───────────────────────────────────────────
// The composer carries it: pick 2-4 RUNNING models that
// answer turns and every send fans out; replies render as split lanes in the
// thread. One model selected = a normal chat (the pick just switches the
// model). Speech models are candidates too  - comparing how two
// models hear the same clip is the point - subject to the same-question rule
// below.
const runningTurns = computed(() =>
  models.models.filter((m) => takesTurns(m.kind) && m.status === 'ok'),
)
const compareOn = computed(() => (chat.active?.compareModels?.length ?? 0) >= 2)
// capability chips for the picker rows - fetched for every candidate so a
// tool/search mismatch between lanes is visible before sending
watch(
  runningTurns,
  (list) => {
    for (const m of list) void models.capsFor(m.id)
  },
  { immediate: true },
)
/** The current send-to set: the compare lanes, else just the chat's model.
 *  Same stale-id rule as effectiveModelId: a model nothing serves anymore
 *  must not be sent to - that path ended in `No running model serves "..."`. */
const sendTo = computed<string[]>(() => {
  const c = chat.active
  if (!c) return []
  if (c.compareModels?.length) return c.compareModels
  const one =
    c.model && c.model !== 'default' && models.models.some((m) => m.id === c.model)
      ? c.model
      : models.currentId
  return one ? [one] : []
})
const MAX_LANES = 4
/** Whether arming this model would leave the lanes with no INPUT in COMMON.
 *  One user turn goes to every lane, so they must all be able to take it -
 *  and the composer can only be in one input mode.
 *
 *  The test is a shared mode, not a shared kind, and the difference is the
 * whole point: whisper vs Qwen3-ASR is the most valuable
 *  comparison this feature can make - same clip, which model hears Swedish
 *  better - and a KIND test refused it, because Qwen3-ASR is a generative
 *  model that also transcribes. So: every lane can chat, or every lane can
 *  transcribe. A model that does both joins either panel.
 *
 *  The reason is said once under the list rather than on every blocked row:
 *  the same sentence repeated down a menu is noise, and a disabled row takes
 *  no pointer events so it could not have been a tooltip either. The rows
 *  themselves already carry the mic mark that explains which is which. */
function sharesAnInput(ids: string[]): boolean {
  return (
    ids.every((x) => models.canChat(x)) ||
    ids.every((x) => models.canTranscribe(x)) ||
    ids.every((x) => models.canImagine(x))
  )
}
function laneBlocked(id: string): boolean {
  if (sendTo.value.includes(id) || !sendTo.value.length) return false
  return !sharesAnInput([...sendTo.value, id])
}
const laneMixNote = computed(() => {
  if (!runningTurns.value.some((m) => laneBlocked(m.id))) return ''
  return 'Greyed models take a different input than the ones selected - every model gets the same message, so they must all be able to read it.'
})
/** In-place toggle: preventDefault keeps the menu open while picking lanes. */
function onCompareSelect(e: Event, id: string): void {
  e.preventDefault()
  if (laneBlocked(id)) return
  toggleCompareModel(id)
}
function toggleCompareModel(id: string): void {
  const c = chat.active
  if (!c) return
  const cur = new Set(sendTo.value)
  if (cur.has(id)) cur.delete(id)
  else if (cur.size < MAX_LANES) cur.add(id)
  // keep the fleet's stable order so lanes don't shuffle between sends
  const list = runningTurns.value.map((m) => m.id).filter((x) => cur.has(x))
  if (list.length >= 2) {
    c.compareModels = list
  } else {
    c.compareModels = undefined
    if (list.length === 1) {
      c.model = list[0]
      models.currentId = list[0]
    }
  }
  chat.persist(c)
}

// ── Audio mode ───────────────────────────────────────────────────────────
// The composer changes when every lane is a SPEECH model - one that cannot
// chat at all. There is no prompt to write for whisper, so the text area has
// no job and gives way to the clip itself: a drop zone, then the clip with
// its player and a language control. Comparing does not change this further
// (compare is about how many lanes consume one input; the composer neither
// knows nor cares) - which is why this reads `sendTo` and nothing else.
//
// A GENERATIVE ASR model (Qwen3-ASR, granite-speech) chats AND transcribes,
// so it keeps the text area: there, what you type is the model's instruction,
// which on that family is the task selector (punctuated transcript vs speaker
// labels vs translation). It just also accepts a clip.
const audioMode = computed(() => {
  const ids = sendTo.value
  if (!ids.length) return false
  // Every lane can hear it, and at least one cannot read text: that is what
  // makes the clip the only possible input. A panel where everything can also
  // chat (Qwen3-ASR on its own, or beside another chat model) stays in text
  // mode and simply accepts an audio attachment.
  return audioPolicy(ids.map(id => ({ chat: models.canChat(id), audio: models.canTranscribe(id), live: models.canLiveTranscribe(id) })), 0).audioMode
})
/** Whether a clip can be attached at all: every lane must be able to read it,
 *  or the send would go to a model that answers with a refusal. */
const audioOk = computed(
  () => sendTo.value.length > 0 && sendTo.value.every((id) => models.canTranscribe(id)),
)
/** One clip per turn - a transcription answers one piece of audio. */
const audioFile = computed(() => files.value.find((f) => isAudioFile(f)))

// ── Image mode ───────────────────────────────────────────────────────────
// The composer is in IMAGE mode when every lane makes pictures and at least
// one cannot chat. The text box STAYS - the prompt is text, and it is the
// whole request - but it now describes a picture: attachments have nowhere
// to go (editing with a reference image is the next lane), the sampler gives
// way to the picture's own controls, and send needs a description. Same
// `sendTo` rule as audio mode: compare only changes how many lanes take the
// one prompt, and on the SAME seed - which is what makes two models'
// pictures comparable.
const imageMode = computed(() => {
  const ids = sendTo.value
  if (!ids.length) return false
  return imagePolicy(
    ids.map((id) => ({ chat: models.canChat(id), image: models.canImagine(id) })),
  ).imageMode
})
const audioDragging = ref(false)
function onAudioDrop(e: DragEvent): void {
  audioDragging.value = false
  void addFiles(Array.from(e.dataTransfer?.files ?? []))
}
// The picker's value, in three states: a chosen code, the explicit auto
// sentinel, or nothing chosen yet - which opens on the browser's own language
// A wrong language guess does not cost you a label, it
// costs you the transcript, and the browser already knows what its owner
// speaks. Detection stays one click away for anyone who wants it.
const audioLanguage = computed<string>({
  get: () => chat.active?.audioLanguage || localeLanguage() || LANGUAGE_AUTO,
  set: (v) => {
    const c = chat.active
    if (!c) return
    c.audioLanguage = v
    chat.persist(c)
  },
})
/** What audio mode takes away, said out loud. These controls are not merely
 *  hidden: a transcription request carries a clip and a language and nothing
 *  else, so a system prompt or a tool armed here would be a setting that
 *  silently did nothing. */
const audioDropped = computed(() => {
  // Only in audio mode - the note names what TRANSCRIPTION drops, so outside it
  // there is nothing being dropped and the sentence is simply false. It was
  // missing this guard, so switching web search on in an ordinary text chat, on
  // a box with no speech model at all, announced that transcription would
  // ignore it.
  if (!audioMode.value) return ''
  const bits: string[] = []
  if (hasSystemPrompt.value) bits.push('the system prompt')
  if (webOn.value) bits.push('web search')
  if (activeConnectorCount.value) bits.push('connectors')
  if (!bits.length) return ''
  const list = bits.length > 1 ? `${bits.slice(0, -1).join(', ')} and ${bits[bits.length - 1]}` : bits[0]
  return `Transcription sends only the clip and its language, so ${list} won't be used.`
})

/** Armed lanes differ in server tools/search - worth saying out loud. */
const laneMismatch = computed(() => {
  const set = chat.active?.compareModels ?? []
  // In audio mode no tools ride at all, so a difference in them is not a
  // difference in the answers - saying so would be noise about a thing the
  // turn does not use.
  if (set.length < 2 || audioMode.value) return false
  const cs = set.map((id) => models.caps[id])
  if (cs.some((c) => !c)) return false
  const web = new Set(cs.map((c) => c!.webSearch))
  const tools = new Set(cs.map((c) => c!.mcpServers.slice().sort().join(',')))
  return web.size > 1 || tools.size > 1
})

// Live draft text (for the context gauge) - kept in sync via the editor's
// onUpdate below.
const draft = ref('')
const contextUsed = computed(() => contextTokens(chat.active, draft.value))


// What this conversation has cost so far: the provider's own per-reply
// prices summed over every assistant turn, compare lanes included. Local
// turns report no money and add nothing; the chip hides until a paid turn
// exists, so an all-local chat never shows a spurious $0.
const convCost = computed(() => {
  let sum = 0
  // Every branch, deliberately: an answer you re-rolled away from was still
  // paid for, so the spend a chat reports must not shrink when you switch
  // branches. Contrast the gauges above, which price the next send.
  for (const m of chat.active?.messages ?? []) sum += m.usage?.costUsd ?? 0
  return sum
})

// Whether the active model can actually read the images being attached. The
// server advertises the loaded model's real capability (/v1/models `vision`);
// fall back to the id heuristic only when it isn't known (e.g. a not-yet-loaded
// model whose name doesn't encode it).
// A chat's model id can be a placeholder ('default') or STALE - a previous
// session's serving id, which even changes with the artifact pick
// (Qwen3.5-9B-Q8_0 vs -UD-Q4_K_XL). Anything not served right now resolves
// through the current model: the raw id rendered "default can't read
// images", and a stale one read vision=false for a model that could see -
// reproduced end-to-end (runner vision:true, composer said no).
const effectiveModelId = computed(() => {
  const id = chat.active?.model
  if (id && id !== 'default' && models.models.some((m) => m.id === id)) return id
  return models.currentId || ''
})
const effectiveModelLabel = computed(() => {
  const id = effectiveModelId.value
  const m = models.models.find((x) => x.id === id)
  return m?.display ?? (id ? friendlyModelName(id) : 'This model')
})
// The image gate has the same trap the thinking control had: with compare
// armed the CURRENT model no longer speaks for the lanes - Laguna up front
// claimed "can't read images" while armed Qwen and Gemma could both see
// Images are blocked only when no armed lane can see
// them; a mixed panel sends them everywhere and a blind lane refuses
// loudly (the runner and providers both reject images without vision), so
// its fold says what happened instead of a silent drop.
const armedLanes = computed<string[] | null>(() => {
  const set = chat.active?.compareModels ?? []
  return set.length >= 2 ? set : null
})
const visionOk = computed(() => {
  const lanes = armedLanes.value
  if (lanes) return lanes.some((id) => models.visionFor(id))
  return models.visionFor(effectiveModelId.value)
})
// Only IMAGES are blocked without vision. PDFs work on any model - the server
// extracts their text (sift) when it can't (or shouldn't) rasterize pages.
const hasBlockedImage = computed(
  () => !visionOk.value && !imageEditOk.value && files.value.some((f) => isImageFile(f)),
)
// A staged photo nothing here can decode - HEIC, in practice, which is what an
// iPhone writes. Said before send, not after: the server's refusal is clear but
// finding out there costs an upload and a turn, and "convert it" has to be done
// outside the app either way. Sending still works and the file is still kept,
// with its details readable in the panel; only the picture never reaches a
// model, so the wording promises exactly that and nothing more.
const unreadableNote = computed(() => {
  const bad = files.value.filter((f) => isUnreadableImage(f))
  if (!bad.length) return ''
  const which = bad.length === 1 ? `${bad[0]!.name} is` : `${bad.length} of these photos are`
  return `${which} HEIC, which no model here can be shown - send it and the file is kept with its details readable, but the picture is not included. Convert to JPEG or AVIF for the model to see it.`
})
// Mixed panel with an image staged: name the lanes that won't see it.
const blindLaneNote = computed(() => {
  const lanes = armedLanes.value
  if (!lanes || !visionOk.value || !files.value.some((f) => isImageFile(f))) return ''
  const blind = lanes.filter((id) => !models.visionFor(id))
  if (!blind.length) return ''
  const names = blind.map((id) => {
    const m = models.models.find((x) => x.id === id)
    return m?.display ?? friendlyModelName(id)
  })
  return `${names.join(' and ')} can't see images - that lane will say so.`
})

// With compare armed the CURRENT model no longer speaks for the lanes: a
// cloud current model hid the thinking toggle while a local qwen lane went
// on thinking with no control in sight. Each lane's style
// comes from its own caps (fetched per candidate above); a control shows
// when any armed lane consumes it - lanes without it simply ignore the
// setting, which the lane's own fold shows honestly.
function laneStyle(id: string): 'effort' | 'toggle' | 'none' {
  return models.reasoningStyleFor(id)
}
const armedStyles = computed<Set<string> | null>(() => {
  const set = chat.active?.compareModels ?? []
  if (set.length < 2) return null
  return new Set(set.map(laneStyle))
})

// The reasoning control is one picker, built from what the armed models
// actually grade at. Qwen3.8 is why: it has a three-rung ladder AND an off
// position, so the old pair of controls (a dropdown for graded families, a
// separate switch for toggle ones) could show only half of it. A model with
// just a switch gets a two-item picker, one that always thinks gets no Off
// item, and a model that cannot reason gets no picker at all.
//
// Across compare lanes the options are the UNION, in the order the lanes
// publish them: a lane that doesn't have the chosen rung clamps to its own
// nearest, which is the runner's job and stays honest. Same doctrine as the
// tools note - a capability mismatch is SAID (reasoningMismatch below), never
// silently leveled down.
const reasoningOptions = computed<string[]>(() => {
  const ids = armedStyles.value ? (chat.active?.compareModels ?? []) : [activeModelId.value]
  return buildReasoningOptions(ids.filter(Boolean).map(id => models.reasoningLadderFor(id)))
})
// Only worth drawing when there is a choice: a model whose whole surface is
// "it reasons" has nothing to pick.
const showReasoning = computed(() => !audioMode.value && reasoningOptions.value.length > 1)
// Same doctrine as the tools note: a capability mismatch between lanes is
// SAID, never silently leveled - forcing thinking off because one lane
// can't think would compare a hobbled model instead of the real one. The
// toggle stays available for a deliberate leveled run.
const reasoningMismatch = computed(() => {
  const s = armedStyles.value
  return !audioMode.value && !!s && s.size > 1
})

// Lights the sampler tool when any dial left "model defaults" behind.
const samplerSet = computed(() => samplerIsSet(chat.active?.params))

// The picker's value is 'off' or a rung name. It writes both stored params -
// `thinking` is the off switch and `reasoningEffort` the rung - so a stored
// conversation and the cloud relay (which reads enable_thinking) keep working
// unchanged, and so switching a chat to a model with no off position does not
// lose which rung you had picked.
//
// The rung falls back to the MODEL's published default rather than a house
// word: an unset request already renders at the checkpoint's own default, so
// opening the picker on anything else would show a level the model is not at.
const reasoningChoice = computed<string>({
  get: () => {
    const opens = activeModelId.value ? models.reasoningLadderFor(activeModelId.value).opens : ''
    return chooseReasoning(reasoningOptions.value, chat.active?.params, opens)
  },
  set: (v) => {
    const c = chat.active
    if (!c) return
    c.params.thinking = v !== 'off'
    // 'on' is the switch-only families' UI-only entry, not a rung - storing it
    // as reasoningEffort would send effort:'on' if the chat later switches to
    // a laddered model
    if (v !== 'off' && v !== 'on') c.params.reasoningEffort = v
    chat.persist(c)
  },
})
/** The rungs are the MODEL's (measured off its template) - these are
 *  only the words for them. The menu used to print the template's own token:
 *  "xhigh reasoning" is an identifier, not something a person says, and it
 *  reads as a typo next to "low" and "medium".
 *
 *  Deliberately a lookup, not a title-caser: `xhigh` has to become "Extra High"
 *  rather than "Xhigh", and a rung we have never seen still needs to render.
 *  The set stays whatever the checkpoint publishes - gpt-oss offers low/medium/
 *  high, Qwen3.8 offers low/medium/xhigh with `high` folded into `xhigh` by its
 *  own template - so this table names every rung any family we serve can have
 *  and shows nobody a level their model does not grade. */

/** Does the lane's own template let a caller decide what happens to earlier
 *  turns' thinking? Measured per template by the runner (`reasoning_preserve`),
 *  so the switch appears for qwen3.6/3.8 and for nothing that would ignore it. */
const canPreserveThinking = computed(
  () => !!activeModelId.value && models.reasoningLadderFor(activeModelId.value).preserve,
)
const preserveThinking = computed<boolean>({
  get: () => chat.active?.params.preserveThinking === true,
  set: (v) => {
    const c = chat.active
    if (!c) return
    c.params.preserveThinking = v
    chat.persist(c)
  },
})

/** Budget presets, not a free-text field: a menu is a chooser, and the runner
 *  floor (1024 on the Anthropic surface) makes sub-1k values a trap anyway. */
const THINKING_BUDGETS = [1024, 2048, 4096, 8192, 16384]

/** Only lanes whose runner says it can ENFORCE a budget (qwen/laguna/gemma
 *  think-blocks); gpt-oss/muse and cloud picks never draw the control. */
const canThinkingBudget = computed(
  () => !!activeModelId.value && models.thinkingBudgetFor(activeModelId.value),
)
const thinkingBudget = computed<number | undefined>({
  get: () => chat.active?.params.thinkingBudget,
  set: (v) => {
    const c = chat.active
    if (!c) return
    c.params.thinkingBudget = v
    chat.persist(c)
  },
})
function budgetLabel(n: number): string {
  return `${n / 1024}k tokens`
}
/** The field is the budget - presets are shortcuts that fill it. 0 = no cap. */
const customBudget = computed<number>({
  get: () => thinkingBudget.value ?? 0,
  set: (v) => {
    thinkingBudget.value = v >= 1 ? Math.round(v) : undefined
  },
})
const isCustomBudget = computed(
  () => thinkingBudget.value != null && !THINKING_BUDGETS.includes(thinkingBudget.value),
)

// ── staged-attachment previews ───────────────────────────────────────────
// Object URLs are memoized per File and revoked when the file leaves the tray.
const urls = new Map<File, string>()
function previewUrl(f: File): string {
  let u = urls.get(f)
  if (!u) {
    u = URL.createObjectURL(f)
    urls.set(f, u)
  }
  return u
}

// ── per-image size ───────────────────────────────────────────────────────
// Each staged image carries its own choice of how much of the model's
// resolution to spend (OpenAI's `detail`). Kept beside `files` rather than
// inside it so the drop/paste path stays a plain File[]; ChatView reads it back
// through detailFor() when it turns the tray into message parts.
const details = ref(new Map<File, ImageDetail>())
// Natural size, captured when the preview thumbnail loads - it's what turns the
// menu from three vague words into three real token counts.
const dims = ref(new Map<File, [number, number]>())

function detailFor(f: File): ImageDetail {
  return details.value.get(f) ?? 'auto'
}
function setDetail(f: File, d: ImageDetail): void {
  details.value = new Map(details.value).set(f, d)
}
function onThumbLoad(f: File, e: Event): void {
  const el = e.target as HTMLImageElement
  if (!el.naturalWidth) return
  dims.value = new Map(dims.value).set(f, [el.naturalWidth, el.naturalHeight])
}

/** The endpoint's published image budget, or undefined when it has no tower
 *  (or is an older runner that doesn't advertise one - same effect: no size
 *  control, because we'd only be guessing at the numbers). */
const visionBudget = computed(() => activeCaps.value?.visionBudget)

/** Vision rows one level would cost this image, or 0 while the thumbnail
 *  hasn't yet told us its real size. Never a made-up figure. */
function detailTokens(f: File, d: ImageDetail): number {
  const b = visionBudget.value
  const wh = dims.value.get(f)
  return b && wh ? tokensFor(b, wh[0], wh[1], d) : 0
}

/** "~1.2k tokens" for one level, '' when the size isn't known yet. */
function detailCost(f: File, d: ImageDetail): string {
  const n = detailTokens(f, d)
  return n ? `~${formatTokens(n)} tokens` : ''
}

/** A level that cannot fit the endpoint's context window at all. Saying so in
 *  the menu beats letting the user pick it and collecting a 400 at send: the
 *  request would be refused for context overflow, which is true but arrives
 *  too late to be useful. */
function detailTooBig(f: File, d: ImageDetail): boolean {
  const n = detailTokens(f, d)
  return n > 0 && models.maxCtx > 0 && n >= models.maxCtx
}

/** Shown under the chip: what this image will cost as currently set, against
 *  the context the endpoint actually serves. */
function detailSummary(f: File): string {
  const cost = detailCost(f, detailFor(f))
  if (!cost) return DETAIL_LABEL[detailFor(f)]
  return `${DETAIL_LABEL[detailFor(f)]}, ${cost}`
}

// ── task actions ─────────────────────────────────────────────────────────
// Some models ship canned instructions in their chat template (granite-vision's
// chart/table extraction). Those instructions are the model's real interface
// for that work, and the only way to reach one without this row is to type its
// literal tag - which nothing anywhere tells you exists.
//
// The template REPLACES the message with the instruction, so an action is a
// send in its own right, not something layered on top of what you wrote. That's
// why a typed message blocks them: silently dropping the words someone just
// typed would be exactly the kind of quiet failure we don't ship.
const taskTags = computed(() => activeCaps.value?.taskTags ?? [])
// PDFs count as visual input where the server rasterizes pages to images
// (vision model + pdfium): a dragged PDF then reaches the tower exactly like
// a dragged image - task-tag actions (chart-to-code etc.) apply to both
// (an image showed the buttons, a PDF did not). On a
// text-only model the server extracts the PDF's TEXT instead (sift), so the
// PDF stays attachable but is not visual input.
const hasImage = computed(() =>
  files.value.some((f) => isImageFile(f) || (isPdfFile(f) && models.pdfRaster && visionOk.value)),
)
const showTasks = computed(() => taskTags.value.length > 0 && hasImage.value && visionOk.value)
// Document-parser lane (deepseek2-ocr, paddleocr): the model only
// reads pages - the server 400s a text-only request because the decoder
// free-runs noise without one. The composer makes that state unsendable:
// no document staged, no send. Once the conversation CARRIES a document
// (sticky, lib/docrun.ts) text-only sends are welcome again - they re-run
// that document with the new instruction ("now as markdown").
const docParser = computed(() => activeCaps.value?.docParser ?? false)
// The runner is up but its model has not attached yet: showing the previous
// model's controls in that window is a lie in UI form, so the composer says
// what is actually happening and follows the caps when they settle
// (/studio kept the plain-chat composer while PaddleOCR loaded).
const modelStarting = computed(() => models.capsPending.has(activeModelId.value))
// the image endpoint edits (its vision tower is wired): pictures may be
// attached as references, and the paperclip stays
const imageEditOk = computed(() => activeCaps.value?.imageGeneration?.edit ?? false)
const activeModelName = computed(() => {
  const id = activeModelId.value
  return models.models.find((m) => m.id === id)?.display ?? id
})
// Raster only: a Word document in the conversation does not make a text-only
// send re-runnable, because there is nothing for the decoder to read.
const hasStickyDoc = computed(() => !!rasterContext(chat.active))
const docNeeded = computed(() => docParser.value && !hasImage.value && !hasStickyDoc.value)
function taskName(tag: string): string {
  return taskLabel(tag, taskTags.value) ?? tag
}
function runTask(tag: string): void {
  if (props.busy || !empty.value) return
  emit('submit', tag)
}

// ── OCR reading mode (deepseek2-ocr lanes) ────────────────────
// The mode list comes from the endpoint's advertised vocabulary, never a
// name heuristic - same contract as task tags. The choice is sticky on the
// chat (like the globe and the thinking toggle) and rides as the request's
// `ocr` object only on lanes that advertise it; the answer's echo then
// records what actually ran. "Automatic" sends nothing: one picture reads
// as a document, several as pages, decided server-side from the request.
const ocrCaps = computed(() => activeCaps.value?.ocr)
const ocrMode = computed(() => {
  const m = chat.active?.ocrMode
  return m && ocrCaps.value?.modes.includes(m) ? m : undefined
})
const ocrModeName = computed(() =>
  ocrMode.value ? ocrModeLabel(ocrMode.value) : OCR_AUTO.label,
)
const ocrRegionsOn = computed(
  () => (chat.active?.ocrRegions ?? false) && (ocrCaps.value?.grounding ?? false),
)
function setOcrMode(m: string | undefined): void {
  const c = chat.active
  if (!c) return
  c.ocrMode = m
  chat.persist(c)
}
function toggleOcrRegions(): void {
  const c = chat.active
  if (!c) return
  c.ocrRegions = !c.ocrRegions
  chat.persist(c)
}

// ── per-file token estimate (documents) ──────────────────────────────────
// The chip shows what a staged document will actually cost: the runner's own
// count_tokens does the real extraction (spreadsheet -> markdown tables, docx
// final view, PDF text) through the real tokenizer, via the manager relay.
// No client-side guessing - when no server is up or the file can't be read,
// the chip simply shows nothing. Estimates re-fetch when the model or the
// file-details toggle changes (both change the real number).
const tokEst = ref(new Map<File, number>())
const estAsked = new Set<File>()
let estKey = ''
watch(
  [files, activeModelId, fileMeta, docOpts],
  () => {
    // per-file settings are part of the price: any change re-asks the server
    const opts = files.value
      .map((f) => {
        const o = docOptsFor(f)
        return `${f.name}:${o.text ? 't' : ''}${pagesParam(o) ?? ''}`
      })
      .join(',')
    const key = `${activeModelId.value}|${fileMeta.value}|${opts}`
    if (key !== estKey) {
      estKey = key
      tokEst.value = new Map()
      estAsked.clear()
    }
    for (const f of files.value) {
      if (isImageFile(f) && !isTiffFile(f)) continue
      // A clip is not priced in context tokens - it is transcribed, and the
      // turn reports the encoder rows it actually cost when it comes back.
      if (isAudioFile(f)) continue
      // A .tvdb never reaches the model as content at all (it becomes the
      // graph session), so pricing it as a document is a guaranteed 400.
      if (isGraphFile(f)) continue
      if (estAsked.has(f)) continue
      estAsked.add(f)
      void estimateTokens(f, key)
    }
  },
  { immediate: true },
)
async function estimateTokens(f: File, key: string): Promise<void> {
  const url = models.countTokensUrl(activeModelId.value)
  if (!url) {
    estAsked.delete(f)
    return
  }
  try {
    const b64 = await new Promise<string>((resolve, reject) => {
      const r = new FileReader()
      r.onload = () => resolve(String(r.result ?? '').split(',')[1] ?? '')
      r.onerror = () => reject(r.error)
      r.readAsDataURL(f)
    })
    // the file's own settings ride as part-level fields, so the price matches
    // exactly what sending would inject
    const o = docOptsFor(f)
    const part: Record<string, unknown> = isTiffFile(f)
      ? {
          type: 'image',
          source: { type: 'base64', media_type: 'image/tiff', data: b64 },
        }
      : {
          type: 'document',
          title: f.name,
          source: {
            type: 'base64',
            media_type: f.type || 'application/octet-stream',
            data: b64,
          },
        }
    if (o.text && isPdfFile(f)) part.pdf_mode = 'text'
    const pg = pagesParam(o)
    if (pg) part.pages = pg
    const body: Record<string, unknown> = {
      model: activeModelId.value,
      messages: [{ role: 'user', content: [part] }],
    }
    if (!fileMeta.value) body.file_metadata = 'off'
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (key !== estKey || !res.ok) return
    const j = await res.json()
    if (Number.isInteger(j.input_tokens)) {
      tokEst.value = new Map(tokEst.value).set(f, j.input_tokens)
    }
  } catch {
    /* absence over invention */
  }
}
function estLabel(f: File): string {
  const n = tokEst.value.get(f)
  return n ? `~${formatTokens(n)} tokens` : ''
}

// Formats the browser can't render (TIFF ...) drew a broken-image icon in the
// tray; the img's error event swaps in a labeled image icon instead. The
// server may still decode what Chrome can't - staging stays allowed.
const badPreview = ref(new Set<File>())
function markBadPreview(f: File): void {
  badPreview.value = new Set(badPreview.value).add(f)
}

watch(files, (list) => {
  const set = new Set(list ?? [])
  for (const [f, u] of urls) {
    if (!set.has(f)) {
      URL.revokeObjectURL(u)
      urls.delete(f)
    }
  }
  if ([...badPreview.value].some((f) => !set.has(f))) {
    badPreview.value = new Set([...badPreview.value].filter((f) => set.has(f)))
  }
  // drop per-file state for anything that left the tray
  if ([...details.value.keys()].some((f) => !set.has(f))) {
    details.value = new Map([...details.value].filter(([f]) => set.has(f)))
  }
  if ([...dims.value.keys()].some((f) => !set.has(f))) {
    dims.value = new Map([...dims.value].filter(([f]) => set.has(f)))
  }
  if ([...tokEst.value.keys()].some((f) => !set.has(f))) {
    tokEst.value = new Map([...tokEst.value].filter(([f]) => set.has(f)))
  }
  for (const f of [...estAsked]) {
    if (!set.has(f)) estAsked.delete(f)
  }
})

const fileInput = ref<HTMLInputElement | null>(null)
const note = ref('')
let noteTimer: number | undefined
function flashNote(msg: string): void {
  note.value = msg
  clearTimeout(noteTimer)
  noteTimer = window.setTimeout(() => (note.value = ''), 4000)
}

/** Async only because of the audio length check below: a clip's duration is
 *  not in its metadata, it is decoded (`audioDuration`). At most one audio
 *  file is kept per turn, so that is at most one await. */
async function addFiles(list: File[]): Promise<void> {
  const accepted: File[] = []
  let tooLarge = 0
  let clip: File | undefined
  let refused = ''
  for (const f of list) {
    if (!isAttachableFile(f) || isTooLarge(f)) {
      tooLarge++
      continue
    }
    // An image model takes pictures to edit when its tower is wired, and
    // nothing else. Refusing by name beats staging a file that would never
    // reach it.
    if (imageMode.value) {
      if (!imageEditOk.value) {
        refused =
          'This model makes pictures from a description - editing with a reference picture needs its vision tower, which is not running.'
        continue
      }
      if (!isImageFile(f)) {
        refused = 'An image model takes pictures to edit - not other files.'
        continue
      }
    }
    if (isAudioFile(f)) {
      // Audio needs a model that can hear it. Refusing by name beats staging
      // a clip that would reach a text model as a note about a file it never
      // received - the silent failure the product principles ban.
      if (!audioOk.value) {
        refused = `${friendlyModelName(effectiveModelId.value)} cannot read audio - pick a speech model to transcribe it.`
        continue
      }
      // ...and short enough for it to hear all of. The recorder stops itself
      // at this same limit, but an attached file arrives already long, and
      // 3 hours of speech is only ~86 MB of Opus - under the size guard, over
      // every generative model's context. Refused here rather than after the
      // upload and a multi-hundred-megabyte decode.
      const cap = audioLimitS.value
      if (cap) {
        const secs = await audioDuration(f).catch(() => undefined)
        if (secs !== undefined && secs > cap) {
          refused =
            `That clip is ${clock(secs)} and ${friendlyModelName(effectiveModelId.value)} ` +
            `can hear ${clock(cap)} at this context size. Trim it, or raise the model's context.`
          continue
        }
      }
      // One clip per turn: a transcription answers one piece of audio, and a
      // second would have nowhere to go.
      clip = f
      continue
    }
    if (audioMode.value) {
      refused = 'A speech model reads audio only - attach a sound file, or pick a chat model.'
      continue
    }
    accepted.push(f)
  }
  if (clip || accepted.length) {
    const kept = clip ? files.value.filter((f) => !isAudioFile(f)) : files.value
    files.value = [...kept, ...accepted, ...(clip ? [clip] : [])]
  }
  if (refused) flashNote(refused)
  else if (tooLarge) {
    flashNote(`Skipped ${tooLarge} file${tooLarge > 1 ? 's' : ''} over ${MAX_FILE_MB} MB.`)
  }
}
// Exposed so the host (ChatView) can route dropped files through the same
// accept/size/note path as the paperclip and paste.
defineExpose({ addFiles, detailFor, docOptsFor, focus: () => editor.value?.commands.focus('end') })
function pick(): void {
  fileInput.value?.click()
}
function onPick(e: Event): void {
  const input = e.target as HTMLInputElement
  if (input.files) void addFiles(Array.from(input.files))
  input.value = ''
}
function removeAt(i: number): void {
  files.value = files.value.filter((_, idx) => idx !== i)
}

const editor = useEditor({
  extensions: [
    StarterKit.configure({
      heading: false,
      blockquote: false,
      bulletList: false,
      orderedList: false,
      listItem: false,
      codeBlock: false,
      horizontalRule: false,
      // Off because it promises the wrong thing. Dragging a
      // file over the composer drew ProseMirror's drop cursor - a bar straight
      // across, since a block boundary is the nearest valid position in a
      // one-paragraph editor - as if the file were about to be inserted at the
      // caret. It is not: ChatView.onDrop takes the drop and the file becomes a
      // chip in the tray, so the editor was drawing an insertion point for an
      // insertion that never happens, while the dropzone card said the true
      // thing a layer above. Nothing here is draggable either (every block node
      // above is disabled), so there is no in-editor drag left for it to serve.
      dropcursor: false,
    }),
    Placeholder.configure({ placeholder: 'Send a message...' }),
    Dictation,
  ],
  onUpdate: ({ editor: ed }) => {
    draft.value = ed.getText()
  },
  editorProps: {
    attributes: { class: 'composer__editor', role: 'textbox', 'aria-multiline': 'true' },
    handleKeyDown(_view, event) {
      // Enter sends; Shift+Enter newline; never mid-IME.
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault()
        submit()
        return true
      }
      return false
    },
    handlePaste(_view, event) {
      // Pasting image files attaches them; pasting text still types normally.
      const clip = event.clipboardData?.files
      if (clip && clip.length) {
        const picked = Array.from(clip).filter(isAttachableFile)
        if (picked.length) {
          void addFiles(picked)
          return true
        }
      }
      return false
    },
  },
})

// ── the microphone ──────────────────────────────────────────────────────────
// One button, two jobs, decided by what the composer is for right now.
//
// TEXT MODE - dictation. Any running speech model transcribes what you say and
// the words land in the composer as ordinary text you can edit before sending.
// It appears only when one is running (a control that leads nowhere is the
// silent failure the product principles ban), and the sound is not kept: the
// words were the point.
//
// It lands UTTERANCE by UTTERANCE: every pause closes one and its
// text is inserted for real, while the sentence still being spoken shows as a
// ghost that is drawn in the document without being in it. So the caret is
// yours throughout - put it three sentences back, fix a word, keep talking.
// The old shape dropped the whole recording in at the end, which is both a
// long wait and the wrong thing to do to somebody's cursor.
//
// AUDIO MODE - the turn itself, live. Every armed lane gets its own
// realtime socket fed the same microphone chunks, so the transcripts race side
// by side while you speak - and they race in the CONVERSATION, not here. The
// composer is a control surface in this mode; what you say belongs in the
// thread.
//
// It used to be a preview under the composer, with the recording then sent
// through the file endpoint and transcribed a SECOND time to make the real
// turn. That was because the socket could emit no segment times and no
// per-word confidence, so a turn committed straight from it would have been a
// second-class record. A closed utterance now comes back with all of it
// so the live turn is the same artifact a dropped file makes and
// there is nothing to re-decode.
const mic = useMicTranscribe()
const live = useLiveTurn()
/** Every transcriber running right now - the ears you can choose between.
 *
 *  It used to be a single `.find()`, which is where two failures lived:
 *  a box with two of them silently picked the first and never said
 *  which had heard you, and a box with none hid the mic entirely, so the
 *  feature did not exist rather than being unavailable.
 *
 *  canTranscribe, not kind: a GENERATIVE ASR model (Qwen3-ASR,
 *  granite-speech) is `kind: 'chat'` with `audio` on its caps - filtering on
 *  kind alone made a box running only Qwen3-ASR claim no speech model was
 *  running, right next to a mic that could have used it. Caps are fetched for
 *  every local runner on fleet refresh, so this settles on its own. */
const transcribers = computed(() =>
  models.models.filter((m) => m.status === 'ok' && m.port && models.canLiveTranscribe(m.id)),
)
// ── Speech models, started and stopped from the mic  ──────────
// The mic used to send you to the pick-a-new-model page whether or not you
// already had one configured - shopping for a model you set up last week.
// The configured ones are listed by name instead, each with its own Start and
// Stop: the ROW is not the control, because a row that acts on click gives no
// hint which way it will act on a model whose state you cannot see from here
//
// The fleet is fetched when the menu OPENS rather than held: `fleet.hold()`
// polls every 3s, and the composer is mounted for the whole chat, so holding
// would put a permanent poll behind a menu almost nobody opens.
// Rows and their buttons live in SpeechModels.vue; this side only decides
// whether the section is worth a chevron and keeps the list fresh.
const micSpeechChoice = computed(() => fleet.speechEndpoints.length > 0)

function onMicMenu(open: boolean): void {
  if (open) void fleet.refresh()
}

/** The chosen ear: the remembered one while it is still running, else
 *  whichever is. A model that has since been stopped falls back rather than
 *  leaving the mic pointed at nothing. */
const transcriber = computed(
  () =>
    transcribers.value.find((m) => m.id === settings.dictateWith) ?? transcribers.value[0],
)
/** Which runners the microphone feeds: every armed lane in audio mode (that
 *  is the fan-out), or whichever speech model is running, for dictation. */
const micPorts = computed<number[]>(() => {
  if (audioMode.value) {
    return sendTo.value.map((id) => models.portFor(id)).filter((p): p is number => !!p)
  }
  const p = transcriber.value?.port
  return p ? [p] : []
})
/** Armed lanes that cannot take a live recording: a cloud speech model hears
 * a finished file, never a socket. They transcribe a dropped clip
 *  perfectly well, so this is about the microphone alone.
 *
 *  It has to be said rather than silently worked around. `micPorts` drops
 *  anything without a port, so a compare of [local whisper, cloud whisper]
 *  would have recorded into the local lane and quietly left the cloud one out
 *  of a comparison the user deliberately set up - the same silent-lane-loss as
 * arriving by a different road. */
const liveOnlyByFile = computed(() =>
  audioMode.value
    ? sendTo.value
        .filter((id) => !models.canLiveTranscribe(id))
        .map((id) => models.models.find((m) => m.id === id)?.display ?? id)
    : [],
)
/** The mic is offered but cannot run live: an armed lane hears a finished file
 *  and not a socket. The option stays VISIBLE and disabled rather than
 * disappearing - hiding it is the failure already fixed once ("the
 *  feature did not exist rather than being unavailable"), and the first cut of
 *  the cloud-speech lanes reintroduced it: adding a cloud model to a compare
 * made the mic silently vanish. */
const micBlocked = computed(() => liveOnlyByFile.value.length > 0)

// ── what the microphone can do, per arming ──────────────────────────────────
// live streams the same PCM to every armed lane as you speak, so the models
// race on identical audio. RECORD captures a clip and sends it like any other
// file. DICTATE puts what you said into the composer as text instead of
// sending the audio anywhere.
//
// Which of the three are on offer is the ARMING's business, and these are the
// four cases:
//
//   1  transcribe-only, one lane            live + record
//   2  one lane that transcribes AND chats  record + dictate   (record first)
//   3  chat lanes, a transcriber loaded     dictate
//   4  compare, every lane can hear and
//      at least one cannot read text        live + record
//      compare, every lane also chats       record only
//      compare, any lane cannot hear        dictate
//
// The through-line, so a fifth case answers itself: live needs an arming where
// a recording is the only possible input, because that is what makes the
// transcript a turn rather than something to type at - which is precisely
// `audioMode`. DICTATE fills the composer, so it leaves as soon as there is
// more than one lane it could be filled from.
//
// Row 4 used to read "any lane also chats -> record only", which
// read the lanes' CAPABILITY instead of the arming's shape and so withheld live
// from every comparison containing a generative ASR model - the exact
// comparison the feature is best at. See `liveApplies`.
const recorder = useRecorder()
const micMode = ref<MicMode>('live')
/** Live is impossible for this arming - every case is a lane that hears a
 *  finished file only. */
const liveImpossible = computed(() => micBlocked.value || !micPorts.value.length)
/** Live BELONGS to this arming (rows 1 and 4a), whether or not it can run.
 *
 *  This is `audioMode` and nothing else: the composer is in audio mode exactly
 *  when every armed lane can hear and at least one cannot read text, which is
 *  the same statement as "a recording is the input here". Every part of the
 *  live path downstream - `micPorts`, `liveOnlyByFile`, `micArmed` - is already
 *  keyed on it, so agreeing here is what keeps them from disagreeing.
 *
 *  It used to also require that no armed lane could chat, and that quietly cost
 *  the whole generative-ASR family live transcription: Qwen3-ASR racing a
 *  whisper checkpoint is the most valuable comparison the Studio can draw, and
 * the mic offered only `record` for it. The runner has
 *  served that case all along - realtime.rs picks `LaneKind::Generative` for
 * any model with audio  - so the capability was built, shipped, and
 *  unreachable.
 *
 *  It is the same kind-vs-caps confusion `transcribers` above already had to
 *  unlearn. A generative ASR model is `kind: 'chat'`, so
 *  `canChat` answers yes for a model whose entire job in this arming is
 *  hearing. What the mic needs to know is whether the lanes can hear a stream,
 *  never whether one of them could also have been typed at. */
const liveApplies = computed(() => audioMode.value)
/** The jobs this arming can actually do, the one that should be default first. */
const micJobs = computed<MicMode[]>(() => {
  // a document parser takes no dictation and no clips - nothing to say to it
  // no lane takes a clip, so the mic can only type for you - and only if some
  // transcriber is running to hear it (row 3)
  return audioPolicy(sendTo.value.map(id => ({ chat: models.canChat(id), audio: models.canTranscribe(id), live: models.canLiveTranscribe(id) })), transcribers.value.length, docParser.value).jobs
})
/** The mic is OFFERED when it has a job at all. When it has none, the "nothing
 *  running" fallback button takes the slot instead - a mic-shaped empty state
 *  that names what is missing and offers to fix it. Exactly one of the two
 *  renders: both at once (on a machine running only Qwen3-ASR) read as
 *  "two mics, one broken". */
const micOffered = computed(() => micJobs.value.length > 0)
/** A choice of JOB to make - or a reason to give: live that belongs here but
 *  cannot run stays as a disabled item that says why, which beats dropping it
 *  silently. */
const micJobChoice = computed(
  () => micMenuPresentation.value.jobChoice,
)

// ── which microphone ────────────────────────────────────────────────────────
// A laptop with a headset has three inputs and one of them is wrong. The
// decision belongs here and not only in Settings because here is where it is
// made: the moment you notice the wrong device is the moment the mic is about
// to open, and navigating to a settings page to fix it is the wrong shape for
// that. Settings keeps the same choice as the durable default (it is a fact
// about the machine, not about this chat), so the two are one preference seen
// from two places.
//
// It appears only once the browser is NAMING the devices, which it does after
// the first grant - before that `enumerateDevices` answers with blanks, and a
// list of three unnamed entries is not a choice. That is also why this menu
// needs no permission step of its own: by the time you look for it here, you
// have recorded at least once.
const audioDevices = useAudioDevices()
/** What the picker offers: every named input, plus the chosen one when it is
 *  not currently plugged in. Kept in the list and marked rather than dropped -
 *  removing it would move the checkmark onto a device nobody picked, which is
 *  the same lie as switching silently. */
const micChoices = computed(() => {
  const list = audioDevices.devices.value.map((d) => ({ id: d.id, label: d.label, here: true }))
  const want = settings.micDeviceId
  if (want && !list.some((d) => d.id === want)) {
    list.push({ id: want, label: settings.micDeviceLabel || 'Chosen microphone', here: false })
  }
  return list
})
/** One input is not a choice - the system default is that input. */
const micDeviceChoice = computed(() => micMenuPresentation.value.deviceChoice)
function setMicDevice(id: string, label: string): void {
  settings.micDeviceId = id
  settings.micDeviceLabel = label
}
/** The chosen device was not there when the mic last opened. Said before the
 *  first word rather than discovered from the recording afterwards - the whole
 *  reason the constraint is `exact`. */
const micDeviceNote = computed(() =>
  audioDevices.lost.value
    ? `${audioDevices.lost.value} isn't connected - recording on the system default instead.`
    : '',
)

/** A choice of EARS to make: only in dictation, where exactly one
 *  model hears you and nothing else on screen says which. */
const micEarChoice = computed(
  () => micMenuPresentation.value.earChoice,
)
/** One menu behind one chevron, in three groups: what the button does, which
 *  microphone it opens, which model hears it. They were separate menus, which
 *  was survivable while the second one was rare - but every box with a headset
 *  now has a device choice, so it would have become two identical chevrons
 *  side by side with nothing to tell them apart. They are the same question
 *  asked three ways, and they belong in one place. */
const micMenu = computed(
  () =>
    micJobChoice.value ||
    micDeviceChoice.value ||
    micEarChoice.value ||
    // Stopping a speech model is only reachable from here - the other mic
    // menu exists precisely when none is running.
    micSpeechChoice.value,
)
/** What pressing the button will do: the remembered choice where this arming
 *  can honor it, else the arming's own first job - never a dead control, and
 *  never a guess between two jobs the user can see (hard-coding
 *  that guess got it wrong in both directions on consecutive builds). */
const micModeNow = computed<MicMode>(
  () => (micJobs.value.includes(micMode.value) ? micMode.value : micJobs.value[0]) ?? 'record',
)
const micMenuPresentation = computed(() => microphoneMenu({ jobs: micJobs.value, mode: micModeNow.value,
  audioMode: audioMode.value, liveImpossible: liveImpossible.value, namedDevices: audioDevices.named(),
  devices: micChoices.value.length, ears: transcribers.value.length, speech: fleet.speechEndpoints.length, docParser: docParser.value }))
const micBusy = computed(() => mic.listening.value || recorder.recording.value)

// ── How long a clip may be, recorded or attached  ─────────────
/** The smallest ceiling among the models that will hear this clip, or none
 *  when they all window it. A generative ASR lane spends the whole clip as
 *  prompt rows, so its limit is real and can be minutes; whisper publishes
 *  nothing because length only costs it time. Cloud lanes advertise no caps
 *  and drop out - the recorder's own hour still applies to them.
 *
 *  One number for both doors: the recorder stops here, and an attached file
 *  is refused here. Capping only the recorder left the same failure reachable
 *  by dragging in a podcast. */
const audioLimitS = computed<number | undefined>(() => {
  const caps = sendTo.value
    .map((id) => models.caps[id]?.transcriptionMaxClipS)
    .filter((s): s is number => typeof s === 'number' && s > 0)
  return caps.length ? Math.min(...caps) : undefined
})
/** Said before the take, not after: "45:00 left" while recording, and the
 *  ceiling on the idle button when one is low enough to plan around. */
function clock(s: number): string {
  const t = Math.max(0, Math.round(s))
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`
}
// Reaching the ceiling ends the take exactly as a click would - the countdown
// has been visible the whole way down, so the clip is sent, not discarded.
watch(
  () => recorder.capped.value,
  (hit) => {
    if (hit && recorder.recording.value) void toggleRecord()
  },
)
/** The open microphone's level, from whichever path owns it. Both feed the
 *  same shared meter, so at most one of these is ever non-empty - the `||` is
 *  reading one thing through two handles, not merging two sources. */
const micLevels = computed(() =>
  recorder.levels.value.length ? recorder.levels.value : mic.levels.value,
)

// Must live below micBusy: watchEffect runs its callback IMMEDIATELY on
// creation, so a `const` declared further down the file is still in its
// temporal dead zone. Sitting above it threw
// "Cannot access 'micBusy' before initialization" on every Composer mount -
// Vue caught it, so the composer still rendered and the only visible symptom
// was a console error plus a reload-hold that silently never armed.
watchEffect(() => {
  holdReload('composer', !!draft.value.trim() || files.value.length > 0 || micBusy.value)
})
onBeforeUnmount(() => holdReload('composer', false))
/** The armed lanes as (port, model) pairs - what the live turn needs to make
 *  one column per model. Ordered exactly like `micPorts`, because the mic's
 *  lanes come back in that order and the turn is matched by index. */
const micArmed = computed(() =>
  sendTo.value.flatMap((id) => {
    const port = models.portFor(id)
    return port ? [{ port, model: id }] : []
  }),
)

/** What the pulsing dot is saying. `idle` is the session's own
 *  `timeout_triggered` - the detector reporting a quiet room - so an open
 *  microphone that has stopped hearing anything says so instead of pulsing
 *  hopefully forever. */
const micStatus = computed(() => {
  if (mic.finishing.value) return 'Finishing...'
  return mic.idle.value ? 'Nothing heard for a while' : 'Listening...'
})

/** How many of the dictation lane's utterances are already in the document.
 *  The insert is driven off this rather than off the event, so a watcher that
 *  fires late (or twice) still inserts each utterance exactly once. */
let dictated = 0

function drainDictation(): void {
  if (audioMode.value) return
  const items = mic.lanes.value[0]?.items ?? []
  // A session that was cancelled takes its items with it; forget what we had
  // counted rather than waiting for a list that will never get that long again.
  if (items.length < dictated) dictated = items.length
  for (; dictated < items.length; dictated++) {
    appendDictated(editor.value, items[dictated].text)
  }
}
watch(() => mic.lanes.value[0]?.items.length ?? 0, drainDictation)
watch(
  () => (audioMode.value ? '' : (mic.lanes.value[0]?.open ?? '')),
  (t) => setGhost(editor.value, t),
)
// The compare turn grows while you speak. Deep because the lanes mutate in
// place - items are pushed onto an existing array - and it is a handful of
// small objects per second, not a render-per-token stream.
watch(
  () => mic.lanes.value,
  (l) => {
    if (audioMode.value && (mic.listening.value || mic.finishing.value)) live.apply(l)
  },
  { deep: true },
)

/** Record a clip, then send it exactly as a dragged file would go: into the
 *  tray and down the ordinary send path, so every armed lane - local AND cloud
 *  - hears the same recording. No live typing this way; that is the trade the
 *  mode exists to make. */
async function toggleRecord(): Promise<void> {
  if (recorder.recording.value) {
    const clip = await recorder.stop()
    if (!clip) {
      // Stopped before the microphone woke, or it delivered nothing but
      // silence. Both have a reason attached; sending an empty clip to every
      // lane and letting them come back blank would not.
      micError.value = recorder.error.value
      return
    }
    // Through addFiles, not straight into the tray. This used to push the clip
    // in directly, which meant the 100 MB guard every other attachment passes
    // never saw the one source that can produce a file of any size at all
    // The recorder's own ceiling makes that unreachable now; this
    // is the belt to its braces, and it keeps one rule for what may attach.
    files.value = files.value.filter((f) => !isAudioFile(f))
    await addFiles([clip])
    if (files.value.some((f) => isAudioFile(f))) submit()
    return
  }
  if (!(await recorder.start(audioLimitS.value)) && recorder.error.value) {
    // getUserMedia said no. The composer has no turn open yet in this mode, so
    // there is nothing to clean up - just say what happened.
    micError.value = recorder.error.value
  }
}
const micError = ref<string | null>(null)

async function toggleMic(): Promise<void> {
  micError.value = null
  if (micModeNow.value === 'record') {
    await toggleRecord()
    return
  }
  if (mic.listening.value) {
    const out = await mic.stop()
    if (audioMode.value) {
      // The turn is already in the conversation, growing since you pressed
      // record; this settles it and attaches the clip.
      await live.finish(out)
      return
    }
    // Everything is already in the document - each utterance went in as it
    // finalised. The last one may have landed inside the await, so drain
    // rather than wait a tick for the watcher, and take the ghost down.
    drainDictation()
    setGhost(editor.value, '')
    focusComposer()
    return
  }
  if (!micPorts.value.length) return
  dictated = 0
  // Resolved, never the raw setting: 'auto' is not a language and an unset
  // chat means the locale, not silence.
  //
  // Every path, dictation included. It used to pass nothing outside audio
  // mode, so dictation always ran on detection no matter what the picker said
  // - and detection is exactly what cannot be trusted here: measured twice,
  // Qwen3-ASR heard Swedish and answered Dutch and German
  // ("Hallo, ich teste und sehe, wie es funktioniert"). A speech
  // model told the language transcribes as that language; left to guess from a
  // few seconds it can lose the whole transcript, not just the label.
  const language = askedLanguage(chat.active?.audioLanguage)
  if (audioMode.value) {
    // A conversation has to EXIST before the turn can go in it, and on a draft
    // surface it does not yet - sending is the only thing that ever made one.
    // Without this the messages landed on the uncommitted draft, which the
    // start page does not even render: you pressed record, it said Listening,
    // and nothing appeared anywhere.
    emit('listen')
    // Open the turn before the microphone: it is the thing being filled, and a
    // getUserMedia prompt the user then dismisses should leave an empty turn to
    // throw away rather than words with nowhere to land.
    if (!live.begin(micArmed.value, language)) return
  }
  await mic.start({
    ports: micPorts.value,
    language,
    record: audioMode.value,
    // Only the comparison shows word times and confidence, and only on the
    // lanes that can actually answer for them. The generative ASR families
    // have no timestamp vocabulary and REFUSE the field - and a refused
    // session.update is refused whole, so asking would also cost that lane its
    // sample rate, its language and its turn detection (measured:
    // Qwen3-ASR then heard 16 kHz as 24 kHz and answered three seconds of
    // Swedish with seven seconds of Finnish).
    detail: micPorts.value.map((p) => {
      if (!audioMode.value) return false
      const id = models.models.find((m) => m.port === p)?.id
      return !!id && models.canEnrichLive(id)
    }),
    drain: micPorts.value.map(p => {
      const id = models.models.find(m => m.port === p)?.id
      return !!id && models.caps[id]?.realtimeTranscription?.drain === true
    }),
    finalRevision: micPorts.value.map(p => {
      const id = models.models.find(m => m.port === p)?.id
      return !!id && models.caps[id]?.realtimeTranscription?.final_revision === true
    }),
    // Dictation asks to hear about a quiet room much sooner, because there it
    // ends the session (see the watcher below). A comparison keeps the long
    // one: you are recording a specific clip and may well be setting up.
    idleMs: audioMode.value ? undefined : DICTATION_IDLE_MS,
    // The session ended without anyone stopping it. Close the same way the
    // stop button would: the turn already in the thread has to be finished
    // with what was heard, or it streams forever over words nobody can see.
    onDied: (out) => {
      if (audioMode.value) void live.finish(out)
      else {
        drainDictation()
        setGhost(editor.value, '')
      }
    },
  })
  if (audioMode.value && !mic.listening.value) live.abandon()
}
// DICTATION CLOSES itself once you have finished and walked away. A pause
// never does - pausing to think is not finishing, which is why the utterance
// boundary is 700 ms and this is five seconds - and it only fires once
// something has actually been dictated, so clicking the mic and gathering your
// thoughts is never answered by the mic giving up. It goes through `toggleMic`
// rather than `mic.stop` so the ghost, the last utterance and the focus are
// handled by the one path that already knows how.
watch(
  () => mic.idle.value,
  (quiet) => {
    if (!quiet || audioMode.value || !mic.listening.value) return
    if (!mic.lanes.value[0]?.items.length) return
    void toggleMic()
  },
)

onBeforeUnmount(() => {
  mic.cancel()
  live.abandon()
})

const empty = computed(() => editor.value?.isEmpty ?? true)
// No model that answers turns = nothing to send to: sending anyway surfaced
// the raw API error (`No running model serves "default"`) - the composer must
// know, not the error path. A speech model counts: it
// answers a turn, it just answers a different kind of input.
const hasTurnModel = computed(() =>
  models.models.some((m) => takesTurns(m.kind) && m.status === 'ok'),
)
const canSend = computed(() => {
  if (!hasTurnModel.value) return false
  // In audio mode the clip is the message - there is nothing else to send.
  if (audioMode.value) return !!audioFile.value
  // In image mode the description is the whole request.
  if (imageMode.value) return !empty.value
  // A document parser needs a page to read; with one staged (or sticky in
  // the conversation) the mode-driven send needs no text at all.
  if (docParser.value) return hasImage.value || hasStickyDoc.value
  return !empty.value || files.value.length > 0
})

function submit(): void {
  if (props.busy || !canSend.value) return
  // The mic's last complaint is about the last recording, and it sits in the
  // one hint line every other note shares - so it has to step aside once the
  // conversation has moved on, or a silent clip from ten minutes ago keeps
  // hiding whatever this turn needs to say.
  micError.value = null
  const text = editor.value?.getText({ blockSeparator: '\n' }).trim() ?? ''
  emit('submit', text)
  editor.value?.commands.clearContent()
  focusComposer()
}

// Auto-focus the composer on entering the chat panel and whenever the active
// conversation changes (opening a chat, or starting a new one).
function focusComposer(): void {
  void nextTick(() => editor.value?.commands.focus('end'))
}
onMounted(focusComposer)
watch(() => chat.activeId, focusComposer)

onBeforeUnmount(() => {
  for (const u of urls.values()) URL.revokeObjectURL(u)
  urls.clear()
  editor.value?.destroy()
})
</script>

<template>
  <div class="composer" :class="{ 'composer--docked': docked }">
    <SystemPromptPanel
      v-if="chat.active"
      :conv="chat.active"
      :open="sysOpen"
      @close="sysOpen = false"
    />
    <div class="composer__box">
      <!-- staged attachments (chips) -->
      <div v-if="files.length && !audioMode" class="composer__tray">
        <div
          v-for="(f, i) in files"
          :key="`${f.name}-${f.size}-${i}`"
          class="attach"
          :class="{ 'attach--bad': (isImageFile(f) && !visionOk) || isUnreadableImage(f) }"
        >
          <img
            v-if="isImageFile(f) && !isUnreadableImage(f) && !badPreview.has(f)"
            :src="previewUrl(f)"
            class="attach__thumb"
            alt=""
            @load="onThumbLoad(f, $event)"
            @error="markBadPreview(f)"
          />
          <span v-else-if="isImageFile(f)" class="attach__doc"><Icon name="image" :size="16" /></span>
          <span v-else class="attach__doc"><Icon name="file" :size="16" /></span>
          <div class="attach__text">
            <span class="attach__name">{{ f.name }}</span>
            <!-- Image size, per image. Only when the endpoint published a
                 budget: without one every figure here would be invented. -->
            <Menu v-if="isImageFile(f) && visionBudget && visionOk">
              <MenuTrigger>
                <button class="attach__size" type="button" aria-label="Image size">
                  <span>{{ detailSummary(f) }}</span>
                  <Icon name="chevron-down" :size="11" />
                </button>
              </MenuTrigger>
              <MenuContent side="top" align="start" min-width="260px">
                <MenuItem
                  v-for="d in DETAIL_ORDER"
                  :key="d"
                  :disabled="detailTooBig(f, d)"
                  @select="setDetail(f, d)"
                >
                  <Icon name="check" :size="14" :style="{ opacity: detailFor(f) === d ? 1 : 0 }" />
                  <span class="attach__opt">
                    <span class="attach__opt-top">
                      <span>{{ DETAIL_LABEL[d] }}</span>
                      <span class="attach__opt-cost">{{ detailCost(f, d) }}</span>
                    </span>
                    <span class="attach__opt-hint">
                      {{
                        detailTooBig(f, d)
                          ? "Larger than this model's context window"
                          : DETAIL_HINT[d]
                      }}
                    </span>
                  </span>
                </MenuItem>
                <template v-if="isTiffFile(f)">
                  <MenuItem @select="setDocOpts(f, { from: undefined, to: undefined })">
                    <Icon
                      name="check"
                      :size="14"
                      :style="{ opacity: !pagesParam(docOptsFor(f)) ? 1 : 0 }"
                    />
                    <span>{{ pageCount.get(f) ? `All ${pageCount.get(f)} page${pageCount.get(f) === 1 ? '' : 's'}` : 'All pages' }}</span>
                  </MenuItem>
                  <div v-if="(pageCount.get(f) ?? 0) > 1" class="attach__range" @click.stop @keydown="stopUnlessEscape">
                    <span>Pages</span>
                    <NumberField
                      :model-value="docOptsFor(f).from ?? 1"
                      :min="1"
                      :max="pageCount.get(f)"
                      @update:model-value="(v: number) => setDocOpts(f, { from: v > 1 ? v : undefined })"
                    />
                    <span>-</span>
                    <NumberField
                      :model-value="docOptsFor(f).to ?? pageCount.get(f)!"
                      :min="1"
                      :max="pageCount.get(f)"
                      @update:model-value="(v: number) => setDocOpts(f, { to: v < pageCount.get(f)! ? v : undefined })"
                    />
                  </div>
                </template>
              </MenuContent>
            </Menu>
            <Menu v-else-if="isPdfFile(f)">
              <MenuTrigger>
                <button class="attach__size" type="button" aria-label="Document pages">
                  <span>{{ docSummary(f) }}</span>
                  <Icon name="chevron-down" :size="11" />
                </button>
              </MenuTrigger>
              <MenuContent side="top" align="start" min-width="250px">
                <MenuItem @select="setDocOpts(f, { text: false })">
                  <Icon name="check" :size="14" :style="{ opacity: !docOptsFor(f).text ? 1 : 0 }" />
                  <span class="attach__opt">
                    <span class="attach__opt-top"><span>Pages as images</span></span>
                    <span class="attach__opt-hint">
                      {{ visionOk && models.pdfRaster ? 'Recommended - the model sees each page' : 'Needs a vision model - falls back to text here' }}
                    </span>
                  </span>
                </MenuItem>
                <MenuItem @select="setDocOpts(f, { text: true })">
                  <Icon name="check" :size="14" :style="{ opacity: docOptsFor(f).text ? 1 : 0 }" />
                  <span class="attach__opt">
                    <span class="attach__opt-top"><span>Text only</span></span>
                    <span class="attach__opt-hint">Extracted text - far fewer tokens</span>
                  </span>
                </MenuItem>
                <MenuItem @select="setDocOpts(f, { from: undefined, to: undefined })">
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: !pagesParam(docOptsFor(f)) ? 1 : 0 }"
                  />
                  <span>{{ pageCount.get(f) ? `All ${pageCount.get(f)} page${pageCount.get(f) === 1 ? '' : 's'}` : 'All pages' }}</span>
                </MenuItem>
                <div v-if="(pageCount.get(f) ?? 0) > 1" class="attach__range" @click.stop @keydown="stopUnlessEscape">
                  <span>Pages</span>
                  <NumberField
                    :model-value="docOptsFor(f).from ?? 1"
                    :min="1"
                    :max="pageCount.get(f)"
                    @update:model-value="(v: number) => setDocOpts(f, { from: v > 1 ? v : undefined })"
                  />
                  <span>-</span>
                  <NumberField
                    :model-value="docOptsFor(f).to ?? pageCount.get(f)!"
                    :min="1"
                    :max="pageCount.get(f)"
                    @update:model-value="(v: number) => setDocOpts(f, { to: v < pageCount.get(f)! ? v : undefined })"
                  />
                </div>
              </MenuContent>
            </Menu>
            <Tooltip
              v-else-if="estLabel(f)"
              label="What this attachment costs in this model's context, extracted and tokenized by its server"
            >
              <span class="attach__size attach__size--static">{{ estLabel(f) }}</span>
            </Tooltip>
          </div>
          <button class="attach__remove" type="button" aria-label="Remove" @click="removeAt(i)">
            <Icon name="x" :size="13" />
          </button>
        </div>
      </div>

      <!-- What this model was actually tuned to do with a picture. Each one is
           a send of its own: the template swaps the action for the model's own
           instruction and drops the rest of the message, so a typed draft
           blocks the row rather than being thrown away behind your back. -->
      <div v-if="showTasks" class="tasks">
        <Tooltip v-for="t in taskTags" :key="t.tag" :label="taskHint(t.prompt)">
          <button
            class="tasks__btn"
            type="button"
            :disabled="busy || !empty"
            @click="runTask(t.tag)"
          >
            {{ taskName(t.tag) }}
          </button>
        </Tooltip>
        <span v-if="!empty" class="tasks__note">Clear your message to use one of these.</span>
      </div>

      <!-- How this document reader will read the staged page(s), from the
           endpoint's own advertised vocabulary. "Show where" arms the
           grounded answer: boxes drawn over your image, marking where each
           piece of the text came from. -->
      <!-- The model's whole vocabulary, laid out as chips - one click picks
           the request (organized data requests, not prose).
           The selected chip is what SEND runs over the document. -->
      <div v-if="ocrCaps && (hasImage || hasStickyDoc)" class="ocrbar">
        <Tooltip :label="OCR_AUTO.hint">
          <button
            class="ocrbar__chip"
            :class="{ 'ocrbar__chip--on': !ocrMode }"
            type="button"
            :aria-pressed="!ocrMode"
            @click="setOcrMode(undefined)"
          >
            {{ OCR_AUTO.label }}
          </button>
        </Tooltip>
        <Tooltip v-for="m in ocrCaps.modes" :key="m" :label="ocrModeHint(m) || ocrModeLabel(m)">
          <button
            class="ocrbar__chip"
            :class="{ 'ocrbar__chip--on': ocrMode === m }"
            type="button"
            :aria-pressed="ocrMode === m"
            @click="setOcrMode(m)"
          >
            {{ ocrModeLabel(m) }}
          </button>
        </Tooltip>
        <Tooltip
          v-if="ocrCaps.grounding && !ocrCaps.modes.includes('grounding')"
          label="The answer marks where on the page each piece came from - drawn as boxes over your image."
        >
          <button
            class="ocrbar__chip"
            :class="{ 'ocrbar__chip--on': ocrRegionsOn }"
            type="button"
            :aria-pressed="ocrRegionsOn"
            @click="toggleOcrRegions"
          >
            <Icon name="image" :size="13" />
            <span>Show where</span>
          </button>
        </Tooltip>
      </div>

      <template v-if="audioMode">
        <div v-if="audioFile" class="clip">
          <div class="clip__head">
            <Icon name="microphone" :size="15" />
            <span class="clip__name">{{ audioFile.name || 'Recording' }}</span>
            <span class="clip__size">{{ fmtFileSize(audioFile.size) }}</span>
            <button
              class="clip__x"
              type="button"
              aria-label="Remove"
              @click="files = files.filter((f) => !isAudioFile(f))"
            >
              <Icon name="x" :size="13" />
            </button>
          </div>
          <AudioPlayer :src="previewUrl(audioFile)" :type="audioFile.type" />
        </div>
        <button
          v-else
          type="button"
          class="drop"
          :class="{ 'drop--over': audioDragging }"
          @click="pick"
          @dragover.prevent="audioDragging = true"
          @dragleave="audioDragging = false"
          @drop.prevent="onAudioDrop"
        >
          <Icon name="paperclip" :size="18" />
          <span>Drop a sound file, or click to choose - then send it to be transcribed</span>
        </button>
      </template>
      <!-- A document parser takes ORGANIZED requests, not prose: its trained
           vocabulary is the reading modes, and a free chat box on it is a lie
           in UI form. The request surface is the document
           + the mode chips below. -->
      <div v-else-if="modelStarting" class="composer__docmode">
        <Icon name="spinner" :size="15" class="composer__startspin" />
        <span>{{ activeModelName }} is starting up.</span>
      </div>
      <div v-else-if="docParser && docNeeded" class="composer__docmode">
        <Icon name="file-text" :size="15" />
        <span>Drop an image or PDF to read.</span>
      </div>
      <EditorContent v-else-if="!docParser" :editor="editor" class="composer__surface" />

      <p v-if="mic.error.value" class="composer__mic-err">{{ mic.error.value }}</p>
      <!-- STATUS ONLY. Neither mode repeats its transcript here any more:
           dictation puts it in the composer and the comparison puts it in the
           conversation, and the same words in two places is two places to
           disagree. -->
      <p v-else-if="mic.listening.value || mic.finishing.value" class="composer__mic">
        <span
          class="composer__mic-dot"
          :class="{ 'composer__mic-dot--hold': mic.finishing.value }"
        />
        <span class="composer__mic-wait">{{ micStatus }}</span>
      </p>

      <div class="composer__toolbar">
        <div class="composer__tools">
          <input
            ref="fileInput"
            type="file"
            :multiple="!audioMode"
            :accept="audioMode ? 'audio/*,.flac,.mp3,.mp4,.mpeg,.mpga,.m4a,.ogg,.oga,.opus,.wav,.webm' : undefined"
            class="composer__file"
            @change="onPick"
          />
          <Menu v-if="showReasoning">
            <MenuTrigger>
              <button
                class="composer__tool composer__effort"
                :class="{ 'composer__tool--active': reasoningChoice !== 'off' }"
                type="button"
                aria-label="Thinking"
              >
                <Icon name="brain" :size="16" />
                <span class="composer__effort-label">{{ reasoningItemLabel(reasoningChoice) }}</span>
                <Icon name="chevron-down" :size="12" />
              </button>
            </MenuTrigger>
            <MenuContent side="top" align="start" min-width="170px">
              <MenuItem
                v-for="o in reasoningOptions"
                :key="o"
                @select="reasoningChoice = o"
              >
                <Icon name="check" :size="14" :style="{ opacity: reasoningChoice === o ? 1 : 0 }" />
                <span class="composer__effort-item">{{ reasoningItemLabel(o) }}</span>
              </MenuItem>
              <!-- Same concept, so the same menu rather than another control in
                   the bar. Shown only where the served template grades it (the
                   qwen3.6/3.8 families) - elsewhere the model would ignore it
                   and the switch would be decoration. `@select.prevent` keeps
                   the menu open: this is a setting, not a destination. -->
              <template v-if="canPreserveThinking">
                <MenuSeparator />
                <MenuItem @select.prevent="preserveThinking = !preserveThinking">
                  <Icon name="check" :size="14" :style="{ opacity: preserveThinking ? 1 : 0 }" />
                  <span class="composer__effort-item">Keep previous thinking</span>
                </MenuItem>
              </template>
              <template v-if="canThinkingBudget && reasoningChoice !== 'off'">
                <MenuSeparator />
                <MenuLabel>Thinking budget</MenuLabel>
                <MenuItem @select.prevent="thinkingBudget = undefined">
                  <Icon name="check" :size="14" :style="{ opacity: thinkingBudget == null ? 1 : 0 }" />
                  <span class="composer__effort-item">Unlimited</span>
                </MenuItem>
                <MenuItem
                  v-for="b in THINKING_BUDGETS"
                  :key="b"
                  @select.prevent="thinkingBudget = b"
                >
                  <Icon name="check" :size="14" :style="{ opacity: thinkingBudget === b ? 1 : 0 }" />
                  <span class="composer__effort-item">{{ budgetLabel(b) }}</span>
                </MenuItem>
                <!-- keydown.stop: the menu's typeahead would otherwise eat
                     the digits being typed into the field -->
                <div class="composer__budget-custom" @keydown.stop>
                  <Icon name="check" :size="14" :style="{ opacity: isCustomBudget ? 1 : 0 }" />
                  <NumberField v-model="customBudget" :min="0" :max="131072" :step="512" />
                  <span class="composer__budget-unit">tokens</span>
                </div>
              </template>
            </MenuContent>
          </Menu>
          <!-- no paperclip in image mode: a description is the whole request
               until editing with a reference image lands -->
          <Tooltip
            v-if="!imageMode || imageEditOk"
            :label="
              imageMode
                ? 'Attach a picture to edit'
                : audioMode
                  ? 'Choose a sound file'
                  : 'Attach files'
            "
          >
            <button class="composer__tool" type="button" @click="pick">
              <Icon name="paperclip" :size="18" />
            </button>
          </Tooltip>
          <Tooltip
            v-if="micOffered"
            :label="
              recorder.arming.value
                ? 'Waiting for the microphone to start - do not speak yet'
                : micBusy
                ? micModeNow === 'record'
                  ? `Stop and send the recording (${clock(recorder.elapsed.value)} · ${clock(recorder.remaining.value)} left)`
                  : audioMode
                    ? 'Stop and send what you said'
                    : 'Stop and put what you said in the composer'
                : micModeNow === 'record'
                  ? `Record a clip and send it - every model transcribes the same recording. Up to ${clock(Math.min(audioLimitS ?? RECORD_MAX_S, RECORD_MAX_S))}${audioLimitS && audioLimitS < RECORD_MAX_S ? ', which is all this model can hear at its context size' : ''}`
                  : audioMode
                    ? micPorts.length > 1
                      ? 'Speak - every model transcribes you at once'
                      : 'Speak, and send the recording to be transcribed'
                    : `Dictate with ${transcriber?.display ?? transcriber?.id}`
            "
          >
            <button
              class="composer__tool"
              :class="{
                'composer__tool--rec': micBusy && !recorder.arming.value,
                'composer__tool--waking': recorder.arming.value,
              }"
              type="button"
              :aria-label="micModeNow === 'dictate' ? 'Dictate' : 'Record'"
              :aria-pressed="micBusy"
              :disabled="mic.finishing.value"
              @click="toggleMic"
            >
              <Icon :name="micBusy ? 'stop' : 'microphone'" :size="17" />
            </button>
          </Tooltip>
          <!-- What the MICROPHONE is HEARING, while it is open. It sits where
               the ear picker was - that control hides itself while busy, so
               the meter costs no layout and nothing jumps when recording
               starts. Decorative to a screen reader: the state it shows is
               already on the button's own pressed/label, and nine bars of
               level are not something to announce.

               It answers the question a stop button cannot: not "am I
               recording" but "is it hearing me". A muted headset, a device the
               OS routed elsewhere, and the wake gap `useRecorder` waits out
               all read as bars that stay flat while you talk. -->
          <div v-if="micBusy && micLevels.length" class="composer__vu" aria-hidden="true">
            <span
              v-for="(l, i) in micLevels"
              :key="i"
              class="composer__vu-bar"
              :style="{ '--l': l, '--l-h': 0.16 + l * 0.84 }"
            />
          </div>
          <!-- How the microphone is used, and which one it is. Live streams to
               every armed lane as you speak; record captures a clip and sends
               it like a file; dictate types what you say into the composer
               instead of sending it as audio. Only offered where there is a
               real choice, and an option that cannot run carries its own
               reason - a disabled item that explains itself beats a button
               that does nothing. -->
          <Menu v-if="micOffered && micMenu && !micBusy">
            <MenuTrigger>
              <button class="composer__ear" type="button" aria-label="Microphone options">
                <Icon name="chevron-down" :size="11" />
              </button>
            </MenuTrigger>
            <MenuContent side="top" align="start" min-width="320px">
              <template v-if="micJobChoice">
                <MenuLabel v-if="micDeviceChoice || micEarChoice">What the mic does</MenuLabel>
                <MenuItem v-if="liveApplies" :disabled="liveImpossible" @select="micMode = 'live'">
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: micModeNow === 'live' ? 1 : 0 }"
                  />
                  <span class="composer__mic-mode">
                    <span>Live</span>
                    <span v-if="micBlocked" class="composer__mic-why">
                      {{ liveOnlyByFile.join(' and ') }}
                      {{ liveOnlyByFile.length > 1 ? 'hear' : 'hears' }} a finished file, not a
                      live stream
                    </span>
                    <span v-else-if="liveImpossible" class="composer__mic-why">
                      no model is running to stream to
                    </span>
                  </span>
                </MenuItem>
                <MenuItem v-if="micJobs.includes('record')" @select="micMode = 'record'">
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: micModeNow === 'record' ? 1 : 0 }"
                  />
                  <span class="composer__mic-mode">
                    <span>Record and send</span>
                    <span class="composer__mic-why">every model hears the same clip</span>
                  </span>
                </MenuItem>
                <MenuItem v-if="micJobs.includes('dictate')" @select="micMode = 'dictate'">
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: micModeNow === 'dictate' ? 1 : 0 }"
                  />
                  <span class="composer__mic-mode">
                    <span>Transcribe into the composer</span>
                    <span class="composer__mic-why">
                      the words go into the box instead of the audio being sent
                    </span>
                  </span>
                </MenuItem>
              </template>
              <template v-if="micDeviceChoice">
                <MenuSeparator v-if="micJobChoice" />
                <MenuLabel>Microphone</MenuLabel>
                <MenuItem @select="setMicDevice('', '')">
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: settings.micDeviceId ? 0 : 1 }"
                  />
                  <span>System default</span>
                </MenuItem>
                <MenuItem
                  v-for="d in micChoices"
                  :key="d.id"
                  @select="setMicDevice(d.id, d.label)"
                >
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: settings.micDeviceId === d.id ? 1 : 0 }"
                  />
                  <span class="composer__mic-mode">
                    <span>{{ d.label }}</span>
                    <!-- Kept in the list while unplugged, and said so: the
                         choice survives a headset going away for a meeting,
                         and comes back with it. -->
                    <span v-if="!d.here" class="composer__mic-why">not connected</span>
                  </span>
                </MenuItem>
              </template>
              <!-- Which EARS. Only when there is genuinely a
                   choice: dictating stays ONE click for the common case, and a
                   picker that opens before every sentence would be worse than
                   the silent pick it replaces. -->
              <template v-if="micEarChoice">
                <MenuSeparator v-if="micJobChoice || micDeviceChoice" />
                <MenuLabel>Heard by</MenuLabel>
                <MenuItem
                  v-for="m in transcribers"
                  :key="m.id"
                  @select="settings.dictateWith = m.id"
                >
                  <Icon
                    name="check"
                    :size="14"
                    :style="{ opacity: transcriber?.id === m.id ? 1 : 0 }"
                  />
                  <span>{{ m.display ?? friendlyModelName(m.id) }}</span>
                </MenuItem>
              </template>
              <!-- Start/stop the box's speech models. Here as well as in the
                   nothing-running menu because this is the ONLY mic menu that
                   exists while one is up, so it is the only place a Stop could
                   be reached. -->
              <template v-if="micSpeechChoice">
                <MenuSeparator v-if="micJobChoice || micDeviceChoice || micEarChoice" />
                <MenuLabel>Speech models</MenuLabel>
                <SpeechModels />
              </template>
            </MenuContent>
          </Menu>
          <!-- Nothing RUNNING. The mic used to vanish here, so dictation did
               not exist rather than being unavailable, and nobody could learn
               it was a thing. Offered and explained instead - with the way to
               fix it, in the start flow's own words.

               Its own condition rather than an `v-else-if` chained off the ear
               picker, which is what it used to be: that made "exactly one of
               the mic and this renders" depend on a SIBLING's visibility, so
               an arming with a cloud speech model and no local one (mic
               offered, `transcribers` empty) drew both - the two-mics-one-
               broken state the comment above the mic button says is fixed. -->
          <Menu
            v-if="!micOffered && !audioMode && !docParser && !transcribers.length"
            @update:open="onMicMenu"
          >
            <MenuTrigger>
              <button class="composer__tool composer__tool--off" type="button" aria-label="Dictate">
                <Icon name="microphone" :size="17" />
              </button>
            </MenuTrigger>
            <MenuContent side="top" align="start" min-width="320px">
              <p class="composer__ear-none">{{ DICTATION_SETUP }}</p>
              <!-- Already set up, just not running: no heading, because the
                   lead above already says what these are and every row wears
                   its own verb. SpeechModels carries its
                   own "start a new one" tail, so nothing is placed here. -->
              <template v-if="micSpeechChoice">
                <MenuSeparator />
                <SpeechModels />
              </template>
              <MenuItem v-else @select="router.push({ name: 'server-new' })">
                <Icon name="plus" :size="14" />
                <span>Start a speech model</span>
              </MenuItem>
            </MenuContent>
          </Menu>
          <!-- The one setting every transcription has. Whisper locks its
               language on the first pass, so telling it beats a guess made
               from one second of speech - and a clip in a language the model
               was not asked about comes back confidently wrong. A picker, not
               a text box: the model takes a fixed set of codes, and "sv" is
               something you either know or get wrong.

               Shown wherever audio can ENTER the composer, not just where the
               composer is clip-shaped. Gating it on the shape hid it in the two
               places it was most needed: a generative ASR keeps its text area,
               so every recording went out labelled with the browser's locale
               with no way to say otherwise (English forced through as Swedish),
               and dictation had no visible language at all (Swedish detected as
               German). -->
          <Select
            v-if="audioOk || micOffered"
            v-model="audioLanguage"
            class="composer__lang"
            :options="languageOptions()"
          />
          <!-- Intelligence / context enrichment: what the runner reads from the
               attachments and feeds into context - file metadata (Sift, always
               available) and forensics (only where the endpoint serves
               it on a vision model). Per-request overrides of the endpoint
               defaults; `@select.prevent` keeps the menu open while toggling. -->
          <Menu v-if="showEnrichment && !imageMode">
            <MenuTrigger>
              <Tooltip label="Context enrichment - what the model reads from your attachments">
                <button
                  class="composer__tool"
                  :class="{
                    'composer__tool--active': fileMeta || (forensicsAvailable && forensicsOn),
                  }"
                  type="button"
                  aria-label="Context enrichment"
                >
                  <Icon name="search" :size="17" />
                </button>
              </Tooltip>
            </MenuTrigger>
            <MenuContent side="top" align="start" min-width="230px">
              <MenuLabel>Context enrichment</MenuLabel>
              <MenuItem @select.prevent="fileMeta = !fileMeta">
                <Icon name="check" :size="14" :style="{ opacity: fileMeta ? 1 : 0 }" />
                <span class="composer__effort-item">File metadata</span>
              </MenuItem>
              <MenuItem v-if="forensicsAvailable" @select.prevent="forensicsOn = !forensicsOn">
                <Icon name="check" :size="14" :style="{ opacity: forensicsOn ? 1 : 0 }" />
                <span class="composer__effort-item">Forensics</span>
              </MenuItem>
            </MenuContent>
          </Menu>
          <!-- Web search: per-chat, off by default; unconfigured hands off to Settings.
               Not on an image lane: nothing but the description reaches that
               endpoint, so search, tools and a system prompt would be switches
               wired to nothing. -->
          <Tooltip v-if="!audioMode && !docParser && !imageMode" :label="webLabel">
            <button
              class="composer__tool"
              :class="{ 'composer__tool--active': webOn }"
              type="button"
              aria-label="Web search"
              :aria-pressed="webOn"
              @click="toggleWeb"
            >
              <Icon name="globe" :size="17" />
            </button>
          </Tooltip>
          <Popover v-if="!audioMode && !docParser && !imageMode" v-model:open="pickerOpen" side="top" align="start">
            <template #trigger>
              <Tooltip label="Tools for this chat">
                <button
                  class="composer__tool composer__cmp"
                  :class="{ 'composer__tool--active': plugActive }"
                  type="button"
                  aria-label="Tools for this chat"
                >
                  <Icon name="plug" :size="17" />
                  <span
                    v-if="selection.mode === 'custom' ? customPickCount : activeConnectorCount"
                    class="composer__cmp-count"
                    >{{ selection.mode === 'custom' ? customPickCount : activeConnectorCount }}</span
                  >
                </button>
              </Tooltip>
            </template>
            <div class="composer__picker">
              <input
                v-model="toolQuery"
                class="pk-input composer__picker-search"
                type="text"
                placeholder="Search tools"
                spellcheck="false"
              />
              <Checkbox
                class="composer__picker-row"
                glyph="check"
                :size="14"
                :model-value="selection.mode === 'all'"
                @update:model-value="selectAllTools"
              >
                <span class="composer__picker-name">All tools</span>
              </Checkbox>
              <div class="composer__picker-list">
                <template v-for="row in pickerRows" :key="row.group.key">
                  <Tooltip :label="row.error" side="right">
                    <Checkbox
                      class="composer__picker-row composer__picker-group"
                      glyph="check"
                      :size="14"
                      :model-value="groupChecked(row.group)"
                      @update:model-value="toggleGroup(row.group)"
                    >
                      <span class="composer__picker-name">{{ row.group.label }}</span>
                      <span v-if="row.status === 'loading'" class="composer__picker-meta">...</span>
                      <span v-else-if="row.status === 'ok'" class="composer__picker-meta">{{
                        row.total
                      }}</span>
                    </Checkbox>
                  </Tooltip>
                  <Tooltip
                    v-for="t in row.tools"
                    :key="`${row.group.key}:${t.name}`"
                    :label="t.description"
                    side="right"
                  >
                    <Checkbox
                      class="composer__picker-row composer__picker-tool"
                      glyph="check"
                      :size="13"
                      :model-value="toolChecked(row.group, t.name)"
                      @update:model-value="toggleTool(row.group, t.name)"
                    >
                      <span class="composer__picker-name">{{ t.name }}</span>
                    </Checkbox>
                  </Tooltip>
                </template>
                <div v-if="!pickerRows.length" class="composer__picker-empty">
                  {{ toolQuery ? 'Nothing matches' : 'No tools on this model yet' }}
                </div>
              </div>
              <button type="button" class="composer__picker-row" @click="openConnectorsPage">
                <Icon name="plus" :size="14" />
                <span class="composer__picker-name">{{
                  connectors.list.length ? 'Manage connectors' : 'Add connectors'
                }}</span>
              </button>
            </div>
          </Popover>
          <Tooltip
            v-if="!audioMode && !docParser && !imageMode"
            label="System prompt - instructions for this chat"
          >
            <button
              class="composer__tool"
              :class="{ 'composer__tool--active': hasSystemPrompt }"
              type="button"
              aria-label="System prompt"
              @click="sysOpen = true"
            >
              <Icon name="sliders" :size="17" />
            </button>
          </Tooltip>
          <SamplerMenu v-if="!audioMode && !docParser && !imageMode">
            <MenuTrigger>
              <Tooltip label="Sampling">
                <button
                  class="composer__tool"
                  :class="{ 'composer__tool--active': samplerSet }"
                  type="button"
                  aria-label="Sampling"
                >
                  <Icon name="thermometer" :size="17" />
                </button>
              </Tooltip>
            </MenuTrigger>
          </SamplerMenu>
          <!-- the picture's own controls take the sampler's place: size,
               quality, steps, seed, count, format, previews -->
          <ImageOptions v-if="imageMode" :lanes="sendTo">
            <MenuTrigger>
              <Tooltip label="Picture settings">
                <button class="composer__tool" type="button" aria-label="Picture settings">
                  <Icon name="image" :size="17" />
                </button>
              </Tooltip>
            </MenuTrigger>
          </ImageOptions>
          <!-- Compare: send to several running models, answers side by side.
               Needs a second running model to mean anything. -->
          <Menu v-if="runningTurns.length > 1">
            <MenuTrigger>
              <Tooltip label="Compare models">
                <button
                  class="composer__tool composer__cmp"
                  :class="{ 'composer__tool--active': compareOn }"
                  type="button"
                  aria-label="Compare models"
                >
                  <Icon name="panel-left" :size="17" />
                  <span v-if="compareOn" class="composer__cmp-count">{{ sendTo.length }}</span>
                </button>
              </Tooltip>
            </MenuTrigger>
            <MenuContent side="top" align="start" min-width="240px">
              <MenuItem
                v-for="m in runningTurns"
                :key="m.id"
                :disabled="laneBlocked(m.id)"
                @select="(e) => onCompareSelect(e, m.id)"
              >
                <Icon name="check" :size="14" :style="{ opacity: sendTo.includes(m.id) ? 1 : 0 }" />
                <VendorLogo v-if="m.vendor" :vendor="m.vendor" :size="15" />
                <Tooltip
                  :label="m.cloud ? `${m.id} · ${m.cloud.endpointName}` : `${m.id} · port ${m.port}`"
                >
                  <span class="composer__cmp-item">{{ m.display ?? friendlyModelName(m.id) }}</span>
                </Tooltip>
                <span class="composer__cmp-caps">
                  <Tooltip v-if="m.cloud" :label="`Cloud model on ${m.cloud.endpointName}`">
                    <span class="composer__cap"><Icon name="cloud" :size="12" /></span>
                  </Tooltip>
                  <Tooltip v-if="!models.canChat(m.id)" label="Speech to text - reads audio, not text">
                    <span class="composer__cap"><Icon name="microphone" :size="12" /></span>
                  </Tooltip>
                  <Tooltip v-if="models.visionFor(m.id)" label="Reads images">
                    <span class="composer__cap"><Icon name="eye" :size="12" /></span>
                  </Tooltip>
                  <Tooltip v-if="models.caps[m.id]?.webSearch" label="Has web search">
                    <span class="composer__cap"><Icon name="globe" :size="12" /></span>
                  </Tooltip>
                  <Tooltip v-if="models.caps[m.id]?.mcpServers.length" label="Has MCP tools">
                    <span class="composer__cap"><Icon name="plug" :size="12" /></span>
                  </Tooltip>
                  <Tooltip v-if="speculationBadge(models.specFor(m.id))" :label="`Speculative decode: ${speculationBadge(models.specFor(m.id))}`">
                    <span class="composer__cap composer__cap--spec">{{ speculationBadge(models.specFor(m.id)) }}</span>
                  </Tooltip>
                </span>
              </MenuItem>
              <div v-if="laneMixNote" class="composer__cmp-note">{{ laneMixNote }}</div>
              <div v-if="laneMismatch" class="composer__cmp-note">
                These servers have different tools - answers will reflect that.
              </div>
              <div v-if="reasoningMismatch" class="composer__cmp-note">
                These models differ on thinking - the control applies only where supported, and
                each answer shows what happened.
              </div>
            </MenuContent>
          </Menu>
          <!-- no meter on an image lane: every send is a fresh render of the
               latest description, nothing from earlier turns reaches the
               model, and the API's output tokens are latent patches -->
          <ContextMeter v-if="models.maxCtx && !audioMode && !imageMode" :used="contextUsed" :max="models.maxCtx" />
          <Tooltip
            v-if="convCost > 0"
            label="What this conversation has cost so far - the provider's own per-reply prices, summed"
          >
            <span class="composer__cost">{{ fmtCost(convCost) }}</span>
          </Tooltip>
        </div>

        <Tooltip
          :label="
            busy
              ? 'Stop generating'
              : !hasTurnModel
                ? 'No model is running - start one in the Manager'
                : audioMode && !audioFile
                  ? 'Add a sound file, or use the microphone'
                  : 'Send  ·  Enter'
          "
        >
          <button
            v-if="busy"
            class="composer__send composer__send--stop"
            type="button"
            @click="emit('stop')"
          >
            <Icon name="stop" :size="15" />
          </button>
          <button
            v-else
            class="composer__send"
            type="button"
            :disabled="!canSend"
            @click="submit"
          >
            <Icon name="send" :size="16" />
          </button>
        </Tooltip>
      </div>
    </div>

    <div
      v-if="micError || micDeviceNote || note || audioDropped || docNeeded || unreadableNote || hasBlockedImage || blindLaneNote"
      class="composer__hint"
    >
      <!-- The microphone's own failures come first, and they are shown at all
           because they were not: `micError` carries "that recording is silent"
           and "the microphone was blocked", both of which were being set and
           never rendered, so the two loudest things the recorder can say
           arrived as nothing happening. -->
      <span v-if="micError" class="composer__note">{{ micError }}</span>
      <span v-else-if="micDeviceNote" class="composer__note">{{ micDeviceNote }}</span>
      <span v-else-if="note" class="composer__note">{{ note }}</span>
      <span v-else-if="audioDropped" class="composer__note">{{ audioDropped }}</span>
      <span v-else-if="docNeeded" class="composer__note">
        {{ effectiveModelLabel }} reads documents - drop an image or PDF for it
        to read. You can also include instructions.
      </span>
      <span v-else-if="unreadableNote" class="composer__note">{{ unreadableNote }}</span>
      <span v-else-if="hasBlockedImage" class="composer__note">
        {{ armedLanes ? 'None of these models can read images' : `${effectiveModelLabel} can't read images` }}
        - attach them once a vision model is loaded. PDFs are fine: their text
        is extracted for the model.
      </span>
      <span v-else class="composer__note">{{ blindLaneNote }}</span>
    </div>
  </div>
</template>

<style scoped>
.composer {
  padding: 10px 24px 16px;
  max-width: var(--pk-chat-width);
  margin: 0 auto;
  width: 100%;
  /* The tool row folds against the COMPOSER'S width, not the viewport's - the
     two stop agreeing the moment the sidebar or the GPU dock opens, and a
     media query would fold at the wrong moments in both directions. */
  container-type: inline-size;
  container-name: composer;
}
/* In a chat the composer FLOATS on the thread's surface: the thread runs the
   full height of the pane and this sits on top of it, so text passes behind
   the input instead of stopping at a hard edge above it.

   Nothing here paints - only composer__box does. A full-width opaque strip
   would be a bar, not a floating input, and it would swallow the thread's
   scrollbar (both this and thread__inner cap at --pk-chat-width and centre,
   so the scrollbar stays out in the gutter beside them). */
.composer--docked {
  position: absolute;
  bottom: 0;
  left: 0;
  /* stops at the SCROLLBAR, not the pane edge. Without it this centres in the
     full pane while the messages centre in the thread's client box, so the
     card sits half a scrollbar right of the column it belongs to - and is
     that much wider whenever the pane is under --pk-chat-width. Measured in
     ChatView; 0 on overlay-scrollbar platforms, which is why the bug is
     invisible outside Windows. */
  right: var(--pk-sbw, 0px);
  width: auto;
}
.composer--docked .composer__box {
  box-shadow: var(--pk-shadow-float);
}

/* The glass the card sits on. Fills the composer and RISES above it, so text
   approaching the input softens instead of staying crisp to the very edge -
   that softening is what reads as depth. Barely a frost by design (1px blur,
   no saturation boost): it is a TINT SCRIM fading content toward the page's
   own ground, with just enough blur to take the edge off glyphs. Behind the
   card, and it eats no clicks. */
.composer--docked::before {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  top: -56px;
  pointer-events: none;
  z-index: 0;
  backdrop-filter: blur(1px);
  -webkit-backdrop-filter: blur(1px);
  background: linear-gradient(to top, var(--pk-glass-tint), transparent);
  -webkit-mask-image: linear-gradient(to top, #000 21%, transparent 100%);
  mask-image: linear-gradient(to top, #000 21%, transparent 100%);
  transition: top 180ms ease, background 180ms ease;
}
/* Focus DEEPENS the depth already on screen rather than adding a colour the
   rest of the design does not use: the card lifts, the conversation recedes. */
.composer--docked:focus-within::before {
  top: -96px;
  background: linear-gradient(to top, var(--pk-glass-tint-focus), transparent);
}
.composer--docked .composer__box:focus-within {
  box-shadow: var(--pk-shadow-float-focus);
}
.composer__box {
  position: relative;
  z-index: 1;
  background: var(--pk-bg-elevated);
  border: 1px solid var(--pk-border-default);
  border-radius: 24px;
  padding: 12px 8px 8px;
  transition: border-color 0.18s ease, box-shadow 180ms ease;
}
.composer__box:focus-within {
  border-color: var(--pk-border-strong);
}

/* editor surface: starts ~3 lines tall, auto-grows to ~11 lines, then scrolls */
.composer__surface :deep(.composer__editor) {
  min-height: 72px;
  max-height: 264px;
  overflow-y: auto;
  padding: 0 12px;
  outline: none;
  color: var(--pk-text-primary);
  font-family: var(--pk-font-sans);
  font-size: var(--pk-font-size-base);
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
.composer__surface :deep(.composer__editor p) {
  margin: 0;
}
.composer__surface :deep(.composer__editor code) {
  font-family: var(--pk-font-mono);
  font-size: 0.9em;
  background: var(--pk-bg-inset);
  padding: 1px 5px;
  border-radius: var(--pk-radius-sm);
}
.composer__surface :deep(.composer__editor p.is-editor-empty:first-child::before) {
  content: attr(data-placeholder);
  float: left;
  height: 0;
  pointer-events: none;
  color: var(--pk-text-muted);
}
/* While the ghost is up the placeholder stands down: both draw into an empty
   paragraph and the placeholder's is a zero-height float, so they overlapped
   rather than replaced each other. */
.composer__surface :deep(.composer__editor.is-dictating p.is-editor-empty::before) {
  content: none;
}
/* The utterance still being spoken. Muted because it is not yours
   yet, and unselectable because it is not in the document - a caret that could
   land inside a decoration is a caret that appears to be somewhere it is not. */
.composer__surface :deep(.dictation-ghost) {
  color: var(--pk-text-muted);
  user-select: none;
  pointer-events: none;
}

/* audio mode: the clip takes the text area's place, because for a speech
   model the clip IS the message */
/* Inset EVENLY from composer__box, then concentric with it. The box pads
   12px top and 8px sides, so a flush child sits 12px from the top and 8px
   from the left - visibly different gaps on two edges of the same card
. The 4px margin evens both to 12px, and the radius follows from
   that inset: 24 (the box) - 12 = 12. */
.drop {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  /* A <button> shrink-to-fits even as a block-level flex container - dropping
     the stated width left it about a third short. box-sizing is
     border-box globally, so this is the box MINUS its own 4px inset per side,
     not 100% plus margins overflowing. `.clip` is a div and fills on its own. */
  width: calc(100% - 8px);
  min-height: 72px;
  padding: 16px 12px;
  border: 1px dashed var(--pk-border-default);
  margin: 0 4px;
  border-radius: 12px;
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-sm);
  cursor: pointer;
  transition: border-color 0.12s ease, color 0.12s ease;
}
.drop:hover,
.drop--over {
  border-color: var(--pk-accent);
  color: var(--pk-text-primary);
}
.clip {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  margin: 0 4px;
  border-radius: 12px;
  background: var(--pk-bg-surface);
}
.clip__head {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--pk-text-secondary);
}
.clip__name {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.clip__size {
  margin-left: auto;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.clip__x {
  display: inline-flex;
  padding: 2px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.clip__x:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
/* the language picker sits in the toolbar next to the microphone - it is the
   one setting a transcription actually takes. Narrower than a form select:
   this is a toolbar chip, and the labels are single words. */
.composer :deep(.pk-select.composer__lang) {
  min-width: 0;
  max-width: 170px;
  padding: 4px 8px;
  font-size: var(--pk-font-size-xs);
  background: transparent;
}

/* staged attachments */
.composer__tray {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 0 8px 10px;
}
.attach {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: 220px;
  padding: 5px 8px 5px 5px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.attach--bad {
  border-color: var(--pk-status-error);
  background: var(--pk-bg-danger-subtle);
}
.attach__thumb {
  width: 34px;
  height: 34px;
  object-fit: cover;
  border-radius: var(--pk-radius-md);
  flex-shrink: 0;
}
.attach__doc {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
  flex-shrink: 0;
}
.attach__text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.attach__name {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.attach__size {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--pk-text-muted);
  font-size: 11px;
  line-height: 1.3;
  cursor: pointer;
  white-space: nowrap;
}
.attach__size:hover {
  color: var(--pk-text-primary);
}
.attach__size--static {
  cursor: default;
}
.attach__size--static:hover {
  color: var(--pk-text-muted);
}
/* the from-to page-range row inside a chip menu (not a MenuItem: inputs need
   their own clicks/keys, so the row stops propagation and never closes) */
.attach__range {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px 8px;
  font-size: 12px;
  color: var(--pk-text-muted);
}
/* compact NumberField inside the menu row */
.attach__range :deep(.pk-num__input) {
  width: 4ch;
  padding: 4px 2px;
  font-size: 12px;
}
.attach__range :deep(.pk-num__step) {
  width: 20px;
}
.attach__opt {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.attach__opt-top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.attach__opt-cost {
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
  font-size: var(--pk-font-size-xs);
}
.attach__opt-hint {
  color: var(--pk-text-muted);
  font-size: 11px;
  line-height: 1.35;
  white-space: normal;
}
.attach__remove {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  border-radius: var(--pk-radius-full);
  background: transparent;
  color: var(--pk-text-muted);
  cursor: pointer;
  flex-shrink: 0;
  transition: background 0.12s ease, color 0.12s ease;
}
.attach__remove:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}

/* task actions (models whose template carries canned instructions) */
.tasks {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  padding: 0 8px 10px;
}
.tasks__btn {
  padding: 4px 11px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-surface);
  color: var(--pk-text-secondary);
  font-family: var(--pk-font-sans);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
}
.tasks__btn:hover:not(:disabled) {
  border-color: var(--pk-border-strong);
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.tasks__btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.tasks__note {
  font-size: 11px;
  color: var(--pk-text-muted);
}

/* the document parser's stand-in for the text editor: what to do next */
.composer__docmode {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 4px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.composer__startspin {
  animation: composer-spin 0.9s linear infinite;
}
@keyframes composer-spin {
  to {
    transform: rotate(360deg);
  }
}

/* OCR reading-mode row - same chip language as the task actions above */
.ocrbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  padding: 0 8px 10px;
}
.ocrbar__chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 11px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-surface);
  color: var(--pk-text-secondary);
  font-family: var(--pk-font-sans);
  font-size: var(--pk-font-size-xs);
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
}
.ocrbar__chip:hover {
  border-color: var(--pk-border-strong);
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.ocrbar__chip--on {
  border-color: var(--pk-accent);
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}

/* bottom toolbar */
.composer__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
  padding: 0 4px;
  gap: 8px;
}
.composer__tools {
  display: flex;
  align-items: center;
  gap: 6px;
  /* A flex item's default `min-width: auto` is why this row could not shrink
     at all: it refused to go under its content width and pushed the send
     button out of the box instead. */
  min-width: 0;
  /* The last RESORT, and it should never be reached. Once the labels fold
     below (~560px) the row's own width is about 330px at its widest arming,
     so wrapping only engages on a composer narrower than any desktop window
     - but if it ever is, two rows is a worse layout than an overflowing one
     rather than a broken one. */
  flex-wrap: wrap;
  row-gap: 4px;
}
/* The send button is the primary action: it keeps its size while the tools
   give way, rather than being the thing that gets squeezed. */
.composer__toolbar > :last-child {
  flex: none;
}

/* ── folding, widest first ────────────────────────────────────────────────
   The order is a priority list, not a set of breakpoints that happened to
   look right. What goes first is decoration on a control that is still
   there; what never goes is the control itself. Nothing here HIDES a
   button - every tool stays clickable at every width. */
@container composer (max-width: 560px) {
  /* Text on a chip whose icon already says it. `low/medium/high` is the
     value, so it survives one tier longer than a name would. */
  .composer__effort-label {
    display: none;
  }
  .composer__effort,
  .composer__cmp {
    padding: 0 7px;
  }
  /* The language picker is the widest thing in the row and the one setting a
     transcription actually takes, so it narrows rather than folding: the
     labels are single words and it keeps its own ellipsis. */
  .composer :deep(.pk-select.composer__lang) {
    max-width: 104px;
  }
  .composer__tools {
    gap: 4px;
  }
}
@container composer (max-width: 430px) {
  /* The effort chip becomes the icon it always was underneath: its own menu
     still says which setting is active, so the chevron was the last piece of
     decoration on it. `svg:last-of-type` is the chevron - Icon renders a bare
     Phosphor <svg> with no class of its own.

     The tools chip keeps its COUNT: that is data, not decoration, and "how
     many tools is this chat carrying" is the only thing the icon cannot say. */
  .composer__effort {
    width: 34px;
    padding: 0;
  }
  .composer__effort > svg:last-of-type {
    display: none;
  }
  .composer__cmp {
    padding: 0 5px;
    gap: 3px;
  }
  .composer :deep(.pk-select.composer__lang) {
    max-width: 78px;
  }
}
.composer__file {
  display: none;
}
.composer__tool {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border: none;
  border-radius: var(--pk-radius-full);
  background: transparent;
  color: var(--pk-text-muted);
  cursor: pointer;
  transition: color 0.15s ease, background 0.15s ease;
}
.composer__tool:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.composer__tool--active,
.composer__tool--active:hover {
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
/* recording reads as danger rather than accent: it is the one composer control
   that keeps doing something after you stop looking at it */
/* The "which ears" caret. Deliberately small and attached to the
   mic rather than a second full-size tool: it is a qualifier on that button,
   not a peer of it. */
.composer__ear {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 28px;
  margin-left: -6px;
  border: 0;
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.composer__ear:hover {
  color: var(--pk-text-primary);
}
/* Offered but not usable yet: the control exists so the feature is
   discoverable, and the menu behind it says what is missing. */
.composer__tool--off {
  opacity: 0.55;
}
/* Same as .composer__cmp-note: the 260px cap was NARROWER than this menu's own
   280px minimum, so the lead wrapped short of the panel edge at every width. */
.composer__ear-none {
  margin: 0;
  padding: 8px 10px 6px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.45;
}
.composer__tool--rec,
.composer__tool--rec:hover {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
}
.composer__tool--waking,
.composer__tool--waking:hover {
  color: var(--pk-text-muted);
  background: var(--pk-bg-subtle);
}
.composer__tool--waking svg {
  animation: composer-waking 1s ease-in-out infinite;
}
/* the live level meter, in the slot the ear picker vacates while busy */
.composer__vu {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  height: 28px;
  margin-left: 2px;
  padding: 0 5px;
  /* Fixed width so the bars cannot reflow the tool row as they move: nine
     3px bars and eight 2px gaps, plus the padding. */
  width: 53px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-danger-subtle);
}
.composer__vu-bar {
  width: 3px;
  /* the box the bars scale inside - scaleY off a centred origin grows both
     ways, which is what makes this read as a level meter and not a bar chart */
  height: 18px;
  /* pills, so a quiet band reads as a dot rather than as a broken bar */
  border-radius: 999px;
  background: var(--pk-text-danger);
  transform-origin: center;
  transform: scaleY(var(--l-h));
  /* Loud is BRIGHT as well as tall. Two channels for one number is what makes
     a small meter legible at a glance - at 3px wide, height alone is a subtle
     signal, and the row reads as a single moving shape rather than nine
     independent slivers. */
  opacity: calc(0.42 + var(--l) * 0.58);
  /* No transition. The envelope in `useMicLevels` already shapes this - fast
     attack, slow release, the asymmetry a meter needs - and a CSS duration on
     top would be a second smoothing stage that damps the attack as much as the
     release, which is exactly backwards. */
  will-change: transform, opacity;
}
@keyframes composer-waking {
  50% {
    opacity: 0.35;
  }
}
@media (prefers-reduced-motion: reduce) {
  .composer__tool--waking svg {
    animation: none;
    opacity: 0.55;
  }
}
.composer__tool:disabled {
  opacity: 0.5;
  cursor: default;
}
.composer__mic,
.composer__mic-err {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin: 6px 4px 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.composer__mic-err {
  color: var(--pk-text-danger);
}
.composer__mic-wait {
  color: var(--pk-text-muted);
}
/* a mode and the one line that says what it costs you */
.composer__mic-mode {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.composer__mic-why {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.composer__mic-dot {
  flex: none;
  align-self: center;
  width: 8px;
  height: 8px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-status-error);
  animation: pk-pulse 1.2s ease-in-out infinite;
}
/* still working, no longer listening - the same dot, not blinking */
.composer__mic-dot--hold {
  background: var(--pk-text-muted);
  animation: none;
}
@keyframes pk-pulse {
  50% {
    opacity: 0.25;
  }
}
.composer__effort {
  width: auto;
  gap: 5px;
  padding: 0 10px;
}
.composer__effort-label {
  font-size: var(--pk-font-size-xs);
  text-transform: capitalize;
}
.composer__effort-item {
  text-transform: capitalize;
}
/* the custom-budget row: aligned like a MenuItem (check gutter + content),
   but a form row, not a selectable item */
.composer__cap--spec {
  width: auto;
  padding: 0 6px;
  font-size: 10px;
  color: var(--pk-text-secondary);
  white-space: nowrap;
}
.composer__budget-custom {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px 6px;
}
.composer__budget-unit {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
.composer__cmp {
  width: auto;
  gap: 5px;
  padding: 0 10px;
}
.composer__cmp-count {
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
}
.composer__cmp-item {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.composer__cmp-caps {
  display: inline-flex;
  gap: 5px;
  margin-left: auto;
  color: var(--pk-text-muted);
}
.composer__cap {
  display: inline-flex;
}
/* No max-width: the panel already caps itself (.pk-menu, 360px), and a second
   tighter cap here pinned the note to the menu's old MINIMUM - so a menu made
   wider by a long model name wrapped its note in a narrow column beside empty
   space. .pk-menu is a column flex, so a plain block child
   stretches to the full width on its own. */
.composer__cmp-note {
  padding: 6px 10px;
  font-size: 11px;
  color: var(--pk-status-warning);
  border-top: 1px solid var(--pk-border-subtle);
  margin-top: 4px;
}

.composer__cost {
  font-size: 11px;
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  cursor: default;
}

/* The tool picker (plug popover): search on top, All-tools row, then the
   grouped tool list with its own scroll. Rows are hand-drawn (a popover has
   no menu semantics - clicks must NOT close it, this is a multi-select). */
.composer__picker {
  width: 292px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.composer__picker-search {
  height: 30px;
  margin-bottom: 6px;
  font-size: var(--pk-font-size-xs);
}
.composer__picker-list {
  max-height: min(320px, 40vh);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  border-bottom: 1px solid var(--pk-border-subtle);
  padding-bottom: 4px;
  margin-bottom: 2px;
}
/* :deep() because the rows are Reka CheckboxRoots, rendered through a clone
   that drops our scope attribute - an unqualified rule reaches none of them.
   Nesting under .composer__picker also outranks Checkbox's own colour rules:
   the tick stays muted in every state here, because the picker conveys
   picked-ness by opacity, not tint (the menu idiom these rows live in). */
.composer__picker :deep(.composer__picker-row) {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
  width: 100%;
  padding: 5px 6px;
  border: 0;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-primary);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  line-height: normal;
  text-align: left;
  cursor: pointer;
  transition: background 0.12s ease;
}
.composer__picker :deep(.composer__picker-row:hover) {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.composer__picker :deep(.composer__picker-row svg) {
  flex-shrink: 0;
  color: var(--pk-text-muted);
}
.composer__picker :deep(.composer__picker-group) {
  font-weight: 600;
}
.composer__picker :deep(.composer__picker-tool) {
  padding-left: 24px;
  font-family: var(--pk-font-mono);
}
.composer__picker-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.composer__picker-meta {
  color: var(--pk-text-muted);
  font-size: 11px;
}
.composer__picker-empty {
  padding: 10px 6px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
}

/* high-contrast circular send (inverts light/dark like Ollama) */
.composer__send {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border: none;
  border-radius: var(--pk-radius-full);
  background: var(--pk-text-primary);
  color: var(--pk-bg-base);
  cursor: pointer;
  transition: opacity 0.15s ease, background 0.15s ease;
}
.composer__send:hover:not(:disabled) {
  opacity: 0.85;
}
.composer__send:disabled {
  opacity: 0.18;
  cursor: not-allowed;
}
.composer__send--stop {
  background: var(--pk-bg-inset);
  color: var(--pk-text-primary);
  border: 1px solid var(--pk-border-strong);
}
.composer__hint {
  text-align: center;
  font-size: 11px;
  color: var(--pk-text-muted);
  margin-top: 9px;
  min-height: 14px;
}
.composer__note {
  color: var(--pk-status-warning);
}
</style>
