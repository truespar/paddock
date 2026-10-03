<script setup lang="ts">
import { providerErrorMessage } from '@/lib/provider-error'
import { copyText } from '@/lib/clipboard'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import type { AudioPart, ContentPart, FilePart, ImagePart, Message } from '@/types/chat'
import { imageParamsOf, messageText } from '@/types/chat'
import { taskLabel } from '@/lib/tasks'
import { cleanOcrText, htmlTablesToMarkdown, ocrModeLabel } from '@/lib/ocr'
import { fleetLabel, fleetVendor, messageStamp } from '@/lib/model-name'
import { answerMetrics, answerMetricsHint, imageMetrics, imageMetricsHint, tokenLimitNote } from '@/lib/message-presentation'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import { clock, srt, vtt } from '@/lib/subtitles'
import { languageName } from '@/lib/languages'
import { fmtDuration, fmtFileSize } from '@/lib/format'
import { alignClip, alignmentRefused, mergeWordTimes } from '@/lib/align'
import { attachmentsApi } from '@/lib/api'
import { useModelsStore } from '@/stores/models'
import { useClipPlayback } from '@/composables/useClipPlayback'
import Icon from '@/components/Icon.vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import Markdown from './Markdown.vue'
import ThinkingFold from './ThinkingFold.vue'
import ToolCall from './ToolCall.vue'
import WebSearchCall from './WebSearchCall.vue'
import RunDetails from './RunDetails.vue'
import DocumentRunView from './DocumentRunView.vue'
import { docContext } from '@/lib/docrun'
import FilePreview from './FilePreview.vue'
import AudioPlayer from './AudioPlayer.vue'
import TranscriptView from './TranscriptView.vue'
import SpeakerTimeline from './SpeakerTimeline.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import { useChatStore } from '@/stores/chat'
import { useGraphsStore } from '@/stores/graphs'
import { activeMessages } from '@/lib/tree'
import { isDocxPart, isPdfPart } from '@/lib/docrun'

const props = withDefaults(
  defineProps<{
    message: Message
    isLast?: boolean
    /** show the model stamp on assistant turns (default). Compare lanes pass
     *  false - their column header already names the model. The default must
     *  be explicit: `stamp?: boolean` alone compiles to a Boolean prop, and
     *  Vue casts an absent Boolean to false - which silently killed the stamp
     *  on every plain turn (found by the cloud-models probe). */
    stamp?: boolean
    /** this turn is rendered inside a compare lane, which already is a card.
     *  Only structural surfaces read it (a transcription answer sits on its
     *  own card outside a lane, and must not become a card in a card). Same
     *  explicit-default rule as `stamp` above. */
    inLane?: boolean
    /** Word indices a SIBLING lane heard differently - only a compare block
     *  can know this, so it arrives from the thread rather than being derived
     *  here. Forwarded straight to the transcript. */
    differs?: Set<number>
  }>(),
  { stamp: true, inLane: false, differs: () => new Set<number>() },
)
const emit = defineEmits<{ regenerate: []; continue: []; edit: [parts: ContentPart[]] }>()

// ── edit a question and ask it again ────────────────────────────────────────
// The edited turn becomes a SIBLING of this one rather than overwriting it
// (lib/tree.ts), so the thread being replaced stays one click away behind the
// `< 2/3 >` control. That is the whole reason editing is safe to offer here.
const editing = ref(false)
const editDraft = ref('')
const editBox = ref<HTMLTextAreaElement | null>(null)

function startEdit(): void {
  editDraft.value = messageText(props.message)
  editing.value = true
  // focus AND put the caret at the end - selecting the whole thing would make
  // the first keystroke of a small correction wipe the question
  void nextTick(() => {
    const el = editBox.value
    if (!el) return
    el.focus()
    el.setSelectionRange(el.value.length, el.value.length)
    autoGrow()
  })
}

function cancelEdit(): void {
  editing.value = false
  editDraft.value = ''
}

/** Send the edit. Non-text parts (images, files, a clip) ride along unchanged:
 *  re-asking a question about a photo must not silently drop the photo. */
function saveEdit(): void {
  const next = editDraft.value.trim()
  if (!next) return
  const kept = props.message.content.filter((p) => p.type !== 'text')
  editing.value = false
  emit('edit', [...kept, { type: 'text', text: next }])
}

function autoGrow(): void {
  const el = editBox.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 420)}px`
}

const isUser = computed(() => props.message.role === 'user')
// Every assistant turn is stamped with the model that wrote it: a chat's
// history can mix models (compare, "continue with this model", switching),
// so an unlabeled answer is ambiguous. Old turns that
// predate model-recording just show no stamp. The stamp shows the human
// name when the fleet still knows one (a cloud id like
// "cloud:...:anthropic/claude-x" is unreadable); the technical id keeps its
// usual place in the hover.
const stampPresentation = computed(() => messageStamp(props.message))
const stampId = computed(() => stampPresentation.value.id)
const stampModel = computed(() => stampPresentation.value.label)
const stampSpec = computed(() => stampPresentation.value.spec)
const stampVendor = computed(() => stampPresentation.value.vendor)
const text = computed(() => messageText(props.message))

// ── HTML preview (table-to-html): SEEING the table beside the original
// document is the verification step source can't provide. The iframe is
// fully neutered - empty sandbox (no scripts, no same-origin, no forms) and
// a CSP that blocks every network fetch, so model-generated markup can't
// run code or leak the user's IP via an <img> URL. Inert rendering only.
const settings = useSettingsStore()
const htmlView = ref<'preview' | 'source'>('preview')
/** The bare-HTML reply, envelope stripped - null when this message is not
 *  one. Drives both the code fence and the preview toggle. */
const bareHtml = computed<string | null>(() => {
  if (isUser.value || props.message.streaming) return null
  const s = (text.value ?? '').trim()
  const inner = s.replace(/^\[\s*/, '')
  if (!inner.startsWith('<') || !/<\/(table|tr|thead|tbody|html)>/i.test(s)) return null
  return s.startsWith('[') && s.endsWith(']') ? s.slice(1, -1).trim() : s
})
const previewDoc = computed(() => {
  const body = bareHtml.value ?? ''
  const dark = settings.theme === 'dark'
  const [bg, fg, bd, rowBd, hd, stripe] = dark
    ? [
        '#121C26',
        '#ECE8E0',
        '#2A3A48',
        'rgba(78,104,120,.25)',
        '#1A2834',
        'rgba(236,232,224,.025)',
      ]
    : [
        '#FFFFFF',
        '#0A1118',
        '#E0E4E8',
        'rgba(0,0,0,.06)',
        '#F5F5F7',
        'rgba(10,17,24,.025)',
      ]
  return (
    `<!doctype html><html><head>` +
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">` +
    `<style>body{margin:10px;font:13px/1.45 Inter,system-ui,sans-serif;color:${fg};background:${bg}}` +
    `table{width:100%;min-width:512px;max-width:100%;border:1px solid ${bd};border-collapse:separate;border-spacing:0;border-radius:8px;overflow:hidden;font-variant-numeric:tabular-nums}` +
    `th,td{padding:8px 12px;border:0;border-bottom:1px solid ${rowBd};text-align:left;vertical-align:top}` +
    `th{background:${hd};border-bottom-color:${bd};font-weight:600;letter-spacing:.01em}` +
    `tbody tr:nth-child(even){background:${stripe}}tr:last-child td{border-bottom:0}</style></head><body>${body}</body></html>`
  )
})
// a sandboxed frame without same-origin can't be measured from outside -
// estimate from the row count and let the frame scroll past the clamp
const previewHeight = computed(() => {
  const rows = (bareHtml.value?.match(/<tr/gi) ?? []).length
  return Math.min(600, Math.max(120, 64 + rows * 38))
})

/** One structural tag per line, indented by depth - whitespace between tags
 *  only, so the displayed HTML stays byte-equivalent where it matters. */
function prettyHtml(src: string): string {
  const parts = src.replace(/>\s*</g, '>\n<').split('\n')
  let depth = 0
  return parts
    .map((p) => {
      if (/^<\//.test(p)) depth = Math.max(0, depth - 1)
      const line = '  '.repeat(depth) + p
      // a pure opening tag (no closing half on the same line) goes deeper
      if (/^<[a-z][^>]*>$/i.test(p) && !/<\//.test(p) && !/\/>$/.test(p)) depth++
      return line
    })
    .join('\n')
}
// Bare structured output (granite-vision's table-to-json etc.) must render
// as CODE: markdown prose typography curls straight quotes into curly ones and eats
// the indentation - the JSON arrives correct and the renderer corrupts it
// visually. While streaming, a growing JSON can't parse
// yet, so the fence rides the '{'/'[' start alone and settles when done.
const renderText = computed(() => {
  // An OCR lane's grounded output rides its regions as inline special-token
  // markup; the structured copy is in `message.ocr.regions`, so display gets
  // the text with the markup lifted out. Keyed on the LANE's caps as well as
  // the settled echo because the echo only lands at stream end - the caps
  // key cleans the text while it is still arriving. No-op everywhere else.
  const t =
    ocr.value || models.caps[stampId.value]?.ocr ? cleanOcrText(text.value) : text.value
  if (isUser.value || !t) return t
  const s = t.trim()
  // granite BRACKETS its task output: `[<otsl>...]`, `[<html>...]` - measured in
  // real conversations. Detection looks through one leading
  // bracket; the display keeps the text verbatim. The first fence version
  // died exactly here: '[' entered the JSON branch, the parse failed, and
  // the early return skipped every detector below.
  const inner = s.replace(/^\[\s*/, '')
  const looksJson =
    (s.startsWith('{') || s.startsWith('[')) && !inner.startsWith('<')
  if (looksJson) {
    if (props.message.streaming) return '```json\n' + s + '\n```'
    try {
      // display-formatted: granite emits one line; re-indenting through the
      // parse changes whitespace only (and it must parse to be fenced)
      return '```json\n' + JSON.stringify(JSON.parse(s), null, 2) + '\n```'
    } catch {
      /* not JSON after all - fall through to the other detectors */
    }
  }
  // Granite's [ ] envelope is the MODEL's wrapper, not part of the payload -
  // copied HTML must be valid HTML. JSON keeps its
  // brackets: there they are the payload (an array is why the parse passed).
  const unwrapped =
    s.startsWith('[') && s.endsWith(']') ? s.slice(1, -1).trim() : s
  // OTSL (table-to-otsl): a token markup, meaningless as prose - the
  // renderer's math/HTML handling EXPLODES it into per-character KaTeX
  // glyphs when unfenced. This document used <ched>, not only <fcel>.
  // <nl> is the row separator, so it becomes a real line break on display.
  if (inner.startsWith('<otsl') || s.includes('<fcel>') || s.includes('<ched>')) {
    const rows = unwrapped
      .replace(/<nl>/g, '<nl>\n')
      .replace(/<otsl>/, '<otsl>\n')
      .replace(/\s*<\/otsl>/, '\n</otsl>')
    return '```\n' + rows + '\n```'
  }
  // bare HTML table (table-to-html): markdown strips or mangles raw markup -
  // and granite emits it as one line, so display gets one tag per line with
  // depth indentation (whitespace between tags only; still valid HTML)
  if (bareHtml.value) {
    return '```html\n' + prettyHtml(bareHtml.value) + '\n```'
  }
  // STREAMING with a markup-looking start: the closing-tag checks above
  // can't match yet, and the raw stream hits the renderer's math path for
  // seconds before settling. Fence plain now; the final render
  // re-classifies (html / otsl / prose) when the message completes.
  if (props.message.streaming && inner.startsWith('<') && !inner.startsWith('<think')) {
    return '```\n' + s + '\n```'
  }
  // bare CSV (chart-to-csv): markdown JOINS consecutive lines into one
  // paragraph, so the rows themselves merge. Uniform comma counts across
  // every line is the tell prose doesn't have; detection waits for the full
  // message - half-streamed rows would misread.
  if (!props.message.streaming) {
    const lines = s.split('\n').filter((l) => l.trim())
    if (lines.length >= 2) {
      const commas = lines.map((l) => (l.match(/,/g) ?? []).length)
      if (commas[0] >= 1 && commas.every((c) => c === commas[0])) {
        return '```csv\n' + s + '\n```'
      }
    }
  }
  return t
})
// A person's attached pictures render as thumbnails above the bubble; the
// pictures a MODEL made (`gen` on the part) are the answer itself and render
// as the answer - full width, with their recipe - below.
const images = computed(
  () => props.message.content.filter((p): p is ImagePart => p.type === 'image' && !p.gen),
)
const generated = computed(
  () => props.message.content.filter((p): p is ImagePart => p.type === 'image' && !!p.gen),
)
/** The picture to show: a preview is only ever its inline data URL; a
 *  finished one is the stored full-resolution bytes, or the inline copy
 *  when there was no store to put them in. */
function genSrc(img: ImagePart): string {
  if (img.gen?.preview) return img.dataUrl ?? ''
  if (img.attachmentId) return attachmentsApi.url(img.attachmentId)
  return img.dataUrl || img.thumbUrl || ''
}
/** "Use this seed": pin the next pictures to this one's draw, so a change to
 *  the words changes only what the words changed. */
function pinSeed(img: ImagePart): void {
  const c = chat.active
  if (!c || !img.gen) return
  c.imageParams = { ...imageParamsOf(c), seed: img.gen.seed }
  chat.persist(c)
}
const seedPinned = computed(() => chat.active?.imageParams?.seed)
// a running clock on a render, ticked only while one is going: the wait is
// tens of seconds, and a spinner alone says nothing about how long
const now = ref(Date.now())
let renderTick: number | undefined
watch(
  () => !!props.message.streaming && !!props.message.imageGen,
  (on) => {
    clearInterval(renderTick)
    renderTick = on ? window.setInterval(() => (now.value = Date.now()), 250) : undefined
  },
  { immediate: true },
)
const renderClock = computed(() => {
  const at = props.message.run?.at
  return at ? `${((now.value - at) / 1000).toFixed(1)} s` : ''
})
const files = computed(
  () => props.message.content.filter((p): p is FilePart => p.type === 'file'),
)
const graphParts = computed(() => props.message.content.filter((p) => p.type === 'graph'))
const graphs = useGraphsStore()
/** Clicking the chip (re)opens the graph session - the pane appears, and an
 *  errored load gets a fresh attempt. Idempotent when already live. */
function openGraph(g: { attachmentId: string; name: string }): void {
  const id = chat.active?.id
  if (!id) return
  graphs.folded = false
  void graphs.ensure(id, g.attachmentId, g.name)
}
const models = useModelsStore()
// A turn sent from the composer's action row is literally the tag - that's what
// the template matches on, and history has to keep it so a resend expands the
// same way. What it must not be is what the reader sees: the Studio never shows
// a markup token to a person, so the bubble renders the action's plain name.
const taskAction = computed(() =>
  isUser.value ? taskLabel(text.value, models.caps[models.currentId]?.taskTags) : null,
)
function fileSub(f: FilePart): string {
  const bits: string[] = []
  if (f.pages) bits.push(`${f.pages} page${f.pages > 1 ? 's' : ''}`)
  else {
    // label by extension - any file attaches now, and a .docx chip must not
    // claim to be a PDF
    const ext = (f.name ?? '').split('.').pop() ?? ''
    bits.push(ext !== f.name && /^[a-z0-9]{1,5}$/i.test(ext) ? ext.toUpperCase() : 'File')
  }
  // Which pages actually went: the file's own range wins; otherwise the
  // server's rendering cap when it bit (pages RENDERED for a vision model -
  // the text route always carries the whole document).
  if (f.pageRange) {
    const range = f.pageRange.replace(/-$/, f.pages ? `-${f.pages}` : '-end')
    bits.push(`page${range.includes('-') ? 's' : ''} ${range} sent`)
  } else {
    const max = models.pdfMaxPages
    if (f.pages && max && f.pages > max && models.pdfRaster) bits.push(`first ${max} sent`)
  }
  if (f.size) bits.push(fmtFileSize(f.size))
  return bits.join(' · ')
}
// reasoning is "active" only until the answer text starts arriving
const reasoningActive = computed(() => !!props.message.streaming && text.value.length === 0)
const pending = computed(
  () => !!props.message.streaming && text.value.length === 0 && !props.message.reasoning,
)

// Shared with the native message footer: answer speed, total send-to-done
// latency, provider cost and reasoning-aware output-limit explanation.
// a picture turn's footer is the render's clock, not a token count - see
// imageMetrics for why the API's output tokens would mislead here
const answerMeta = computed(() =>
  props.message.imageGen
    ? imageMetrics(props.message.imageGen)
    : answerMetrics(props.message.usage, realtime.value),
)
const cutNote = computed(() => tokenLimitNote(props.message.usage))
const answerMetaTip = computed(
  () =>
    (props.message.imageGen
      ? imageMetricsHint(props.message.imageGen, props.message.usage)
      : answerMetricsHint(props.message.usage)) || undefined,
)

// The three-dot "typing" pill only appears after a short delay, so a fast reply
// never flashes it (Ollama does the same to avoid flicker).
const showDots = ref(false)
let dotTimer: number | undefined
watch(
  pending,
  (p) => {
    clearTimeout(dotTimer)
    if (p) dotTimer = window.setTimeout(() => (showDots.value = true), 450)
    else showDots.value = false
  },
  { immediate: true },
)

// Thumbnails the browser failed to decode (legacy messages persisted raw
// TIFF bytes as thumbUrl) fall back to the labeled stub tile instead of a
// broken-image icon.
const brokenThumbs = ref(new Set<number>())
// The fallback surface for a chip whose format the pane cannot render (see
// openFile): the tabbed FilePreview - Document beside "Model metadata"
// (the extraction the prompt carries, mirroring this chat's file-details
// toggle). Types with no viewer at all open on the model tab, with download.
const preview = ref<FilePart | null>(null)
const chat = useChatStore()
const insightMeta = computed(() => chat.active?.fileMetadataEnabled ?? true)
/** A photo opens in the DOCUMENT PANE, the same surface a PDF or a Word
 * document opens. It used to answer a click with a lightbox
 *  dialog and carry a SECOND dialog behind an eye button on the same tile -
 *  full-res pixels in one window, "what the model reads" in another, neither
 *  with a zoom. The pane has both, side by side, and the tile is down to one
 *  gesture again.
 *
 *  Same identity rule as openFile: a document is the user turn it rode in on,
 *  and only those turns appear in the conversation's document list. Assistant
 *  turns carry no images today; if one ever does, its tile stays a picture
 *  rather than pointing at a document the pane cannot resolve. */
function openImage(): void {
  if (props.message.role !== 'user') return
  chat.showDocument(props.message.id)
}
// Best downloadable source: legacy inline bytes win over a (possibly stale)
// attachmentId; new messages use the stored full-res view copy.
function downloadSrc(img: ImagePart): string {
  if (img.dataUrl) return img.dataUrl
  if (img.attachmentId) return attachmentsApi.url(img.attachmentId)
  return img.modelUrl || img.thumbUrl || ''
}

// Per-turn "run details" (provenance + metrics + GPU env) - the lab record.
// Shown for any assistant turn that has a run OR at least usage metrics (older
// turns predate run-recording but still have usage).
const showRun = ref(false)
const hasRun = computed(
  () =>
    !isUser.value &&
    (!!props.message.run || !!props.message.usage || !!props.message.imageGen),
)

// ── transcription turns  ──────────────────────────────────────
// The USER turn shows the clip as a chip; the ANSWER hosts the player, right
// beside the transcript it produced. In a compare that means one player per
// lane on the same audio - which is what makes the comparison usable: play
// the clip against this model's words without leaving its column.
const audios = computed(() =>
  props.message.content.filter((p): p is AudioPart => p.type === 'audio'),
)
const transcript = computed(() => (isUser.value ? undefined : props.message.transcript))
/** The clip a message carries: the first audio part that was actually STORED.
 *
 *  One rule, used by both sides, and that is the point of it being a function.
 *  The user turn and the lane used to select differently - the user turn took
 *  the first part with an attachment id, the lane took the first audio part at
 *  all - so a turn carrying an unstored clip in front of a stored one keyed
 *  the player under one id and looked it up under another. Nothing failed
 *  loudly; the seek simply went to a registry entry that was never there.
 *
 *  The id is what everything downstream keys on, so a part without one is not
 *  a clip anything can address. */
function clipOf(m: { content: ContentPart[] }): AudioPart | undefined {
  return m.content.find((p): p is AudioPart => p.type === 'audio' && !!p.attachmentId)
}

/** The clip this transcription answered: the nearest user turn before it. */
const answered = computed<AudioPart | undefined>(() => {
  if (!transcript.value) return undefined
  // The path, not the array: "the nearest user turn before this one" only
  // means anything along one branch.
  const msgs = chat.active ? activeMessages(chat.active) : []
  // `indexOf` is identity against the store's own array - ChatThread renders
  // those objects directly. A message that is somehow not in it would search
  // from -2 and find nothing, so fall back to scanning for the last user turn
  // before giving up: a lane that cannot find its clip is a lane whose word
  // clicks silently do nothing, which is the failure this whole path keeps
  // producing.
  const at = msgs.indexOf(props.message)
  const from = at >= 0 ? at - 1 : msgs.length - 1
  for (let i = from; i >= 0; i--) {
    if (msgs[i].role !== 'user') continue
    return clipOf(msgs[i])
  }
  return undefined
})
/** How long the clip was.
 *
 *  The CLIP'S own length comes first, and the model's report second, because
 *  in a compare every lane has to divide by the same number or the speeds are
 *  not comparable. They disagree in practice: on one 4.62 s recording whisper
 *  reported 4.62 and OpenRouter billed a rounded 4.0, which is a 15% swing in
 *  a figure whose whole job is to be compared across lanes. The clip is one
 *  object and has one length; the lanes only ever estimate it. */
const audioSecs = computed(() => answered.value?.durationS ?? transcript.value?.durationS)

/** Speed against the clip: seconds of audio per second of waiting.
 *
 *  This is what tok/s cannot say about a transcription. In chat the model
 *  chooses how much to write, so tokens per second measures how fast it
 *  writes; in ASR the length of the answer is fixed by what was SAID, so a
 *  model that renders the same sentence in fewer tokens scores slower while
 *  being exactly as quick. Realtime factor is the metric every ASR system
 *  reports, and it is self-normalizing: it means something on a lane that ran
 *  alone, which a cross-lane ratio never does.
 *
 *  Read it knowing whisper pads every clip to a 30 s window - a 4 s clip costs
 *  the same encoder pass as a 25 s one, so short clips understate it and the
 *  number climbs with length up to the window. */
const realtime = computed(() => {
  const secs = audioSecs.value
  const ms = props.message.usage?.ms
  if (!secs || !ms || ms <= 0) return undefined
  return (secs * 1000) / ms
})

// One transport per clip, mounted on the user's turn where the audio actually
// is, and shared by every lane that transcribed it. A lane reads
// the playhead to walk its own highlight and calls back to seek; it no longer
// owns a player of its own.
const clips = useClipPlayback()
const player = ref<InstanceType<typeof AudioPlayer> | null>(null)
/** The clip this message is about: the one it carries (user turn) or the one
 *  it answered (a lane). Both sides key off the same attachment id. */
const clipPart = computed<AudioPart | undefined>(() =>
  isUser.value ? clipOf(props.message) : answered.value,
)
const clipId = computed(() => clipPart.value?.attachmentId)
const playhead = computed(() => clips.timeOf(clipId.value))
function seek(t: number): void {
  clips.seekClip(clipId.value, t)
}
const seeker = (t: number): void => {
  player.value?.seek(t)
}
watch(
  clipId,
  (id, old) => {
    if (old) clips.releasePlayer(old, seeker)
    if (id && isUser.value) clips.registerPlayer(id, seeker)
  },
  { immediate: true },
)
onBeforeUnmount(() => clips.releasePlayer(clipId.value, seeker))
// A lane knows the clip's measured length; the player, sitting on the user
// turn, does not. Publish it so the scrubber has a real track even for a
// recording whose container carries no duration.
watch(
  () => transcript.value?.durationS,
  (d) => clips.publishDuration(clipId.value, d),
  { immediate: true },
)
// ...and the same crossing for the decode notices, so the player can paint them
// on the waveform (gave us the spans; this is where they become a
// place on the timeline rather than a sentence under the text).
watch(
  () => transcript.value?.guards,
  (g) => clips.publishGuards(clipId.value, g),
  { immediate: true },
)
function clipSub(a: AudioPart): string {
  const bits: string[] = []
  if (a.durationS) bits.push(clock(a.durationS))
  if (a.size) bits.push(fmtFileSize(a.size))
  if (a.language) bits.push(`language ${a.language}`)
  return bits.join(' · ')
}
/** The facts about the transcription itself, as chips rather than a sentence:
 *  what language it turned out to be (detected, or forced by the request) and
 *  how long the audio was. Named in full - "Swedish", not "sv" - because a
 *  two-letter code is a lookup, not an answer. */
const transcriptFacts = computed<{ label: string; value: string }[]>(() => {
  const m = transcript.value
  if (!m) return []
  const out: { label: string; value: string }[] = []
  if (m.language) {
    out.push({
      // Whether this was the model's finding or the user's instruction is the
      // whole difference between "it heard Swedish" and "it was told Swedish".
      label: answered.value?.language ? 'Language' : 'Detected',
      value: languageName(m.language),
    })
  }
  if (audioSecs.value) out.push({ label: 'Audio', value: clock(audioSecs.value) })
  // Who timed the words, when it was not the transcriber. Named rather than
  // implied: a second model ran on this clip, and the only other evidence is
  // highlighting you have to press play to see.
  if (m.wordsFrom) {
    out.push({
      label: 'Word timing',
      value:
        fleetLabel(m.wordsFrom) +
        (m.wordsLangOk === false ? ' · outside its languages' : ''),
    })
  }
  return out
})

// ── word timing: the affordance under an untimed transcript ──
// One spot, two states: no aligner in the fleet -> name what to start; aligner
// up but this transcript still untimed (it settled before the aligner ran, or
// enrichment failed) -> align the stored clip in place. Requires the clip to
// still be stored - a transcript whose audio is gone cannot be timed, so it
// gets no offer. ja/ko stay quiet too: the aligner refuses them by design.
const untimed = computed(
  () =>
    !!transcript.value &&
    !props.message.streaming &&
    !props.message.error &&
    !!clipId.value &&
    !!text.value.trim() &&
    !transcript.value.words?.some((w) => w.start !== undefined && w.end !== undefined) &&
    !alignmentRefused(transcript.value.language),
)
const alignerUp = computed(() => !!models.alignerLane())
const alignBusy = ref(false)
const alignErr = ref('')
async function timeWords(): Promise<void> {
  const lane = models.alignerLane()
  const t = transcript.value
  const id = clipId.value
  const conv = chat.active
  if (!lane || !t || !id || !conv || alignBusy.value) return
  alignBusy.value = true
  alignErr.value = ''
  try {
    const stored = await fetch(attachmentsApi.url(id))
    if (!stored.ok) throw new Error(`audio attachment: HTTP ${stored.status}`)
    const out = await alignClip(
      lane.url,
      await stored.blob(),
      answered.value?.name || 'clip.wav',
      text.value,
      t.language,
    )
    const words = mergeWordTimes(t, text.value, out.words)
    if (!words) throw new Error('the aligner and this transcript disagree on the words')
    // mutate through the store's own message object (same reactive target as
    // the prop) and persist once - the enrichment pass's exact write shape
    const msg = conv.messages.find((m) => m.id === props.message.id)
    if (msg) {
      msg.transcript = { ...t, words, wordsFrom: lane.id, wordsLangOk: out.languageSupported }
    }
    chat.persistNow(conv)
  } catch (e) {
    // the server's own sentence (clip over the 400 s budget, decode refusal)
    // shown in place - a button that fails silently reads as broken
    alignErr.value = e instanceof Error ? e.message : String(e)
  } finally {
    alignBusy.value = false
  }
}

// ── OCR turns  ────────────────────────────────────────────────
// The answer's structured half: the server's resolution echo + grounded
// regions, hung on the message exactly like `transcript`. Regions render as
// overlays in the DocumentPane (the main area's first column); this side
// keeps the facts and the dropped-text note.
const ocr = computed(() => (isUser.value ? undefined : props.message.ocr))
// Document-parser answers render RESULTS only (restructure):
// the pages live in the DocumentPane. Fires for a docRun/ocr turn, or when
// the model's advertised capability says parser (absent caps - stopped
// server, old chat - fall back to plain rendering). Lane-safe: the result
// view has no grid of its own.
const docParserRun = computed(() => {
  if (isUser.value) return false
  if (props.message.docRun || ocr.value) return true
  const id = props.message.model ?? props.message.run?.model ?? ''
  return (id && models.caps[id]?.docParser) ?? false
})
/** Where a file chip goes when you click it.
 *
 *  Anything we have a real viewer for opens in the DOCUMENT PANE, in every
 * conversation - one lector, one toolset, and the same
 *  bargain for scriptor. It used to be two integrations: the
 *  parser lane put the document in the side pane, while an ordinary chat opened
 *  a modal carrying a cut-down viewer with no toolbar, thumbnails, search or
 *  text selection. Same library, two different products. Word was the last
 *  format still living in the dialog, and there is no reason a .docx should
 *  open somewhere a .pdf does not.
 *
 *  Everything else - spreadsheets, legacy .doc, types with no viewer at all -
 *  still opens the tabbed dialog, which is the only place they can be shown. */
function openFile(f: FilePart): void {
  // A document's identity is the USER turn it rode in on (docContexts only
  // scans those), and for a user turn that is this bubble. A file on an
  // assistant turn has no entry in that list, so routing it to the pane would
  // select an id nothing resolves and quietly show a different document -
  // those keep the dialog.
  if (props.message.role === 'user' && (isPdfPart(f) || isDocxPart(f))) {
    chat.showDocument(props.message.id)
    return
  }
  preview.value = f
}
/** the sticky document's filename, for export naming */
const docName = computed(() => {
  if (!docParserRun.value) return undefined
  const ctx = docContext(chat.active)
  return ctx?.pdf?.name ?? ctx?.images[0]?.name
})
/** Export the extraction (same duty as the transcript's exportAs): the pages
 *  joined for md/txt, the full per-page record for JSON. */
function docExportAs(kind: 'md' | 'txt' | 'json'): void {
  const base = (docName.value ?? 'extraction').replace(/\.[^.]+$/, '')
  const run = props.message.docRun
  const clean = (s: string) => htmlTablesToMarkdown(cleanOcrText(s))
  const joined = run
    ? run.pages
        .map((p, i) =>
          run.pages.length > 1 ? `## Page ${i + 1}\n\n${clean(p.text)}` : clean(p.text),
        )
        .join('\n\n')
    : clean(text.value ?? '')
  if (kind === 'md') {
    save(`${base}.md`, 'text/markdown', joined)
  } else if (kind === 'txt') {
    const plain = joined
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/\|/g, ' ')
      .replace(/^[-\s:|]+$/gm, '')
    save(`${base}.txt`, 'text/plain', plain)
  } else {
    const payload = {
      model: props.message.model,
      pages: run
        ? run.pages.map((p, i) => ({
            page: i + 1,
            state: p.state,
            text: cleanOcrText(p.text),
            note: p.note,
            regions: p.regions,
            ms: p.ms,
          }))
        : [{ page: 1, text: cleanOcrText(text.value ?? ''), regions: props.message.ocr?.regions }],
    }
    save(`${base}.json`, 'application/json', JSON.stringify(payload, null, 2))
  }
}

/** How the page was read, as chips - from the server's echo (what actually
 *  ran), never from what the composer asked for. */
const ocrFacts = computed<{ label: string; value: string }[]>(() => {
  const m = ocr.value
  if (!m) return []
  const out: { label: string; value: string }[] = []
  if (m.passThrough) out.push({ label: 'Read as', value: 'Your prompt' })
  else if (m.mode) out.push({ label: 'Read as', value: ocrModeLabel(m.mode) })
  if ((m.pages ?? 0) > 1) out.push({ label: 'Pages', value: String(m.pages) })
  // the crop class in plain words: how much of the page the tower saw
  if (m.crop === 'gundam' && (m.tiles ?? 0) > 0) {
    out.push({ label: 'Detail', value: `full page + ${m.tiles} tiles` })
  } else if (m.crop === 'base') {
    out.push({ label: 'Detail', value: 'whole page' })
  }
  if (m.imageTokens) out.push({ label: 'Image tokens', value: String(m.imageTokens) })
  if (m.regions?.length) {
    const n = m.regions.reduce((a, r) => a + r.boxes.length, 0)
    out.push({ label: 'Regions', value: String(n) })
  }
  return out
})

// Export is a product principle, not a page feature: a transcript that can
// only be read in the Studio is a transcript trapped in the Studio. Subtitle
// formats need the times, so they are offered only when the segments exist.
function save(name: string, mime: string, data: string): void {
  const url = URL.createObjectURL(new Blob([data], { type: `${mime};charset=utf-8` }))
  const el = document.createElement('a')
  el.href = url
  el.download = name
  el.click()
  // Deferred: the click starts an async fetch of the blob URL, and revoking
  // synchronously races it - a lost race is a silent empty download.
  setTimeout(() => URL.revokeObjectURL(url), 30_000)
}
function exportAs(kind: 'srt' | 'vtt' | 'txt' | 'json'): void {
  const base = (answered.value?.name || 'transcript').replace(/\.[^.]+$/, '')
  const cues = (transcript.value?.segments ?? []).map((s) => ({
    start: s.start,
    end: s.end,
    text: s.text,
  }))
  if (kind === 'srt') save(`${base}.srt`, 'application/x-subrip', srt(cues))
  else if (kind === 'vtt') save(`${base}.vtt`, 'text/vtt', vtt(cues))
  else if (kind === 'json') {
    // The persisted TranscriptMeta, verbatim - segments with their nested
    // words, the flat timed words, confidences, guards. Same export duty as
    // the subtitles: what the user is LOOKING at, never a re-request (a
    // re-decode could change the words). Fields a lane never reported drop
    // out of the JSON rather than riding as null.
    const t = transcript.value
    save(
      `${base}.json`,
      'application/json',
      JSON.stringify(
        {
          text: text.value,
          language: t?.language,
          duration: t?.durationS,
          segments: t?.segments,
          words: t?.words,
          guards: t?.guards,
        },
        null,
        2,
      ),
    )
  } else save(`${base}.txt`, 'text/plain', text.value)
}

const copied = ref(false)
async function copy(): Promise<void> {
  try {
    // an OCR answer copies as the readable text - the region markup's
    // structured copy already lives in the overlay, not the clipboard
    await copyText(ocr.value ? cleanOcrText(text.value) : text.value)
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch {
    /* clipboard blocked */
  }
}

onBeforeUnmount(() => {
  clearTimeout(dotTimer)
  clearInterval(renderTick)
})
</script>

<template>
  <article class="msg" :class="`msg--${isUser ? 'user' : 'assistant'}`">
    <div class="msg__col">
      <!-- attached images render above the bubble -->
      <div v-if="images.length" class="msg__images">
        <div v-for="(img, i) in images" :key="i" class="msg__image-wrap">
          <Tooltip
            v-if="(img.thumbUrl || img.dataUrl) && !brokenThumbs.has(i)"
            :label="`Open ${img.name}`"
          >
            <img
              :src="img.thumbUrl || img.dataUrl"
              :alt="img.name"
              class="msg__image"
              @click="openImage()"
              @error="brokenThumbs = new Set(brokenThumbs).add(i)"
            />
          </Tooltip>
          <!-- Two different silences. A TIFF has no preview here but the model
               still receives it; a HEIC was never sent at all, because nothing
               decodes HEVC. Saying"the model still gets the image"
               over the second one would be a plain lie in the transcript. -->
          <Tooltip
            v-else
            :label="
              img.unreadable
                ? `${img.name}: HEIC, so it was kept but not shown to the model. Open it for its details.`
                : `${img.name}: this browser can't preview the format; the model still gets the image`
            "
          >
            <button class="msg__image msg__image-stub" type="button" @click="openImage()">
              <Icon name="image" :size="22" />
              <span class="msg__image-stub-name">{{ img.name }}</span>
            </button>
          </Tooltip>
          <Tooltip label="Download">
            <a
              class="msg__image-dl"
              :href="downloadSrc(img)"
              :download="img.name || 'image'"
              aria-label="Download image"
              @click.stop
            >
              <Icon name="arrow-down" :size="14" />
            </a>
          </Tooltip>
        </div>
      </div>

      <div v-if="audios.length" class="msg__files msg__clip">
        <div v-for="(a, i) in audios" :key="i" class="msg__file msg__file--flat">
          <span class="msg__file-icon"><Icon name="microphone" :size="18" /></span>
          <span class="msg__file-meta">
            <span class="msg__file-name">{{ a.name || 'Recording' }}</span>
            <span class="msg__file-sub">{{ clipSub(a) }}</span>
          </span>
          <Tooltip v-if="a.attachmentId" label="Download the audio">
            <a
              class="msg__file-dl"
              :href="`/api/attachments/${a.attachmentId}`"
              :download="a.name || 'recording'"
              aria-label="Download the audio"
            >
              <Icon name="arrow-down" :size="14" />
            </a>
          </Tooltip>
        </div>
        <!-- One transport, outside the loop: a `ref` inside v-for collects
             into an array and `player.value.seek` would silently be a no-op.
             It drives every lane that transcribed this clip. -->
        <AudioPlayer
          v-if="isUser && clipPart?.attachmentId"
          ref="player"
          class="msg__clip-player"
          :src="attachmentsApi.url(clipPart.attachmentId)"
          :type="clipPart.mime"
          :clip="clipId"
          :guards="clips.guardsOf(clipId)"
          :fallback="clips.durationOf(clipId) || clipPart.durationS || 0"
          @time="clips.publishTime(clipId, $event)"
        />
      </div>

      <div v-if="graphParts.length" class="msg__files">
        <Tooltip v-for="(g, i) in graphParts" :key="i" label="Open the graph">
          <button class="msg__file" type="button" @click="openGraph(g)">
            <span class="msg__file-icon"><Icon name="hard-drive" :size="18" /></span>
            <span class="msg__file-meta">
              <span class="msg__file-name">{{ g.name }}</span>
              <span class="msg__file-sub">Graph database</span>
            </span>
          </button>
        </Tooltip>
      </div>
      <div v-if="files.length" class="msg__files">
        <Tooltip v-for="(f, i) in files" :key="i" label="Click to open">
          <button class="msg__file" type="button" @click="openFile(f)">
            <span class="msg__file-icon"><Icon name="file-text" :size="18" /></span>
            <span class="msg__file-meta">
              <span class="msg__file-name">{{ f.name }}</span>
              <span class="msg__file-sub">{{ fileSub(f) }}</span>
            </span>
          </button>
        </Tooltip>
      </div>
      <FilePreview
        :file="preview"
        :with-meta="insightMeta"
        :model="chat.active?.model"
        @close="preview = null"
      />

      <header v-if="stamp && stampModel" class="msg__hd">
        <Tooltip :label="stampId !== stampModel ? stampId : undefined">
          <span class="msg__hd-model">
            <VendorLogo v-if="stampVendor" :vendor="stampVendor" :size="14" />
            {{ stampModel }}
            <span v-if="stampSpec" class="msg__hd-spec">{{ stampSpec }}</span>
          </span>
        </Tooltip>
      </header>

      <ThinkingFold
        v-if="message.reasoning"
        :reasoning="message.reasoning"
        :active="reasoningActive"
        :ms="undefined"
        :tokens="message.usage?.reasoningTokens"
        :tps="undefined"
      />

      <div v-if="message.webSearches?.length || message.toolCalls?.length" class="msg__tools">
        <WebSearchCall v-for="c in message.webSearches" :key="c.id" :call="c" />
        <ToolCall v-for="c in message.toolCalls" :key="c.id" :call="c" />
      </div>

      <div v-if="taskAction" class="msg__task">
        <Icon name="wrench" :size="13" />
        <span>{{ taskAction }}</span>
      </div>
      <!-- App-authored turn (graph auto-repair report): the app talking to
           the model, not the person - a compact notice, full text folded. -->
      <div v-else-if="isUser && message.auto" class="msg__auto">
        <Collapsible summary="Import failed - the app asked the model to repair the graph">
          <pre class="msg__auto-body">{{ text }}</pre>
        </Collapsible>
      </div>
      <div v-else-if="isUser && text" class="msg__bubble">{{ text }}</div>

      <template v-else-if="!isUser">
        <!-- An image-generation answer: the pictures ARE the answer. While the
             render runs, the latest preview stands in (the render converging,
             decoded part way) under a clock; finished, each picture carries the
             recipe that made it - seed, size, steps - and the two actions that
             matter: keep this seed, save the file. -->
        <div v-if="generated.length" class="gen" :class="{ 'gen--grid': generated.length > 1 }">
          <figure v-for="(g, i) in generated" :key="i" class="gen__pic">
            <img
              :src="genSrc(g)"
              :alt="message.imageGen?.prompt || g.name"
              class="gen__img"
              :class="{ 'gen__img--preview': g.gen?.preview }"
              :width="g.width"
              :height="g.height"
            />
            <div v-if="g.gen?.preview" class="gen__progress">
              <Icon name="spinner" :size="13" class="gen__spin" />
              <span>preview {{ (g.gen.previewIndex ?? 0) + 1 }} · {{ renderClock }}</span>
            </div>
            <figcaption v-else class="gen__bar">
              <span class="gen__facts">
                <span>seed {{ g.gen?.seed }}</span>
                <span>{{ g.gen?.size }}</span>
                <span>{{ g.gen?.steps }} steps</span>
                <span v-if="g.gen?.background === 'transparent'">transparent</span>
              </span>
              <span class="gen__acts">
                <Tooltip
                  :label="seedPinned === g.gen?.seed ? 'This seed is pinned for the next picture' : 'Use this seed for the next picture'"
                >
                  <button
                    class="pk-icon-btn msg__act"
                    :class="{ 'gen__act--on': seedPinned === g.gen?.seed }"
                    type="button"
                    aria-label="Use this seed"
                    @click="pinSeed(g)"
                  >
                    <Icon name="pin" :size="14" />
                  </button>
                </Tooltip>
                <Tooltip label="Save">
                  <a
                    class="pk-icon-btn msg__act"
                    :href="downloadSrc(g)"
                    :download="g.name || 'image'"
                    aria-label="Save picture"
                  >
                    <Icon name="download" :size="14" />
                  </a>
                </Tooltip>
              </span>
            </figcaption>
          </figure>
        </div>
        <div v-else-if="message.streaming && message.imageGen" class="gen__wait">
          <Icon name="spinner" :size="14" class="gen__spin" />
          <span>Rendering {{ message.imageGen.size }} · {{ renderClock }}</span>
        </div>
        <!-- A transcription answer: the words the model heard, marked by how
             sure it was of each one. It sits on a surface card of its own - a
             transcript is a structured artifact, not prose, and left bare it
             floated on the page background. Inside a compare lane the lane
             already IS that card, so it stays flat.

             The transport lives on the USER's turn, where the audio is, and
             drives every lane at once. -->
        <div v-else-if="transcript" class="tx" :class="{ 'tx--flat': inLane }">
          <SpeakerTimeline :transcript="transcript" :clip-id="clipId" :message-id="message.id" :streaming="message.streaming" @seek="seek" />
          <ul v-if="transcript.guards?.length" class="tx__guards">
            <li v-for="(g, i) in transcript.guards" :key="i">
              <span class="tx__guard-span">{{ clock(g.start) }}-{{ clock(g.end) }}</span>
              {{ g.note }}
            </li>
          </ul>
          <TranscriptView
            :segments="transcript.segments ?? []"
            :words="transcript.words ?? []"
            :differs="differs"
            :time="playhead"
            :plain="text"
            @seek="seek"
          />
          <div v-if="untimed" class="tx__timing">
            <button v-if="alignerUp" class="tx__timing-act" :disabled="alignBusy" @click="timeWords">
              <Icon name="clock" :size="13" />
              {{ alignBusy ? 'Timing words...' : 'Add word timing' }}
            </button>
            <span v-else class="tx__timing-act">
              <Icon name="clock" :size="13" />
              <RouterLink :to="{ name: 'server-new-config', params: { model: 'qwen3-forced-aligner-0.6b' } }"
                >Start the Word timing model</RouterLink
              >
              to follow playback word by word.
            </span>
            <span v-if="alignErr" class="tx__timing-err">{{ alignErr }}</span>
          </div>
          <dl v-if="transcriptFacts.length" class="tx__facts">
            <template v-for="f in transcriptFacts" :key="f.label">
              <dt>{{ f.label }}</dt>
              <dd>{{ f.value }}</dd>
            </template>
          </dl>
        </div>
        <!-- A document-parser answer: the extraction only - the pages live in
             the DocumentPane, the main area's first column (
             restructure). Export sits in the actions row below, like the
             transcript's. -->
        <DocumentRunView
          v-else-if="docParserRun"
          :message="message"
          :facts="ocrFacts"
          :dropped-text="ocr?.droppedText ?? false"
          :render-text="renderText"
          :flat="inLane"
        />
        <template v-else-if="bareHtml">
          <div class="msg__htmlbar">
            <button
              type="button"
              class="msg__htmlbtn"
              :class="{ 'msg__htmlbtn--on': htmlView === 'preview' }"
              :aria-pressed="htmlView === 'preview'"
              @click="htmlView = 'preview'"
            >
              Preview
            </button>
            <button
              type="button"
              class="msg__htmlbtn"
              :class="{ 'msg__htmlbtn--on': htmlView === 'source' }"
              :aria-pressed="htmlView === 'source'"
              @click="htmlView = 'source'"
            >
              Source
            </button>
          </div>
          <iframe
            v-if="htmlView === 'preview'"
            class="msg__htmlframe"
            sandbox=""
            :srcdoc="previewDoc"
            :style="{ height: previewHeight + 'px' }"
            title="Rendered HTML preview"
          />
          <Markdown v-else :content="renderText" :streaming="message.streaming" />
        </template>
        <!-- Editing replaces the question in place: the answer below stays on
             screen until the new one arrives, so the edit reads as a change to
             this turn rather than as the thread being torn down. -->
        <div v-else-if="editing" class="msg__edit">
          <textarea
            ref="editBox"
            v-model="editDraft"
            class="pk-input msg__edit-box"
            rows="1"
            aria-label="Edit this message"
            @input="autoGrow"
            @keydown.enter.exact.prevent="saveEdit"
            @keydown.esc.prevent="cancelEdit"
          />
          <div class="msg__edit-row">
            <span class="msg__edit-hint">Enter to send · Esc to cancel</span>
            <button class="pk-btn pk-btn--ghost pk-btn--sm" type="button" @click="cancelEdit">
              Cancel
            </button>
            <button
              class="pk-btn pk-btn--primary pk-btn--sm"
              type="button"
              :disabled="!editDraft.trim()"
              @click="saveEdit"
            >
              Send
            </button>
          </div>
        </div>
        <Markdown v-else-if="text" :content="renderText" :streaming="message.streaming" />
        <div v-else-if="showDots" class="msg__typing"><span /><span /><span /></div>
        <div v-if="message.error" class="msg__error">
          <Icon name="x" :size="14" />
          <span class="msg__error-text">{{ providerErrorMessage(message.error) }}</span>
        </div>
        <div v-if="message.stopped && !text" class="msg__stopped">Stopped</div>
        <div v-if="message.incomplete === 'length' && !message.streaming" class="msg__cut">
          <span>{{ cutNote }}</span>
          <button v-if="isLast" class="msg__continue" type="button" @click="emit('continue')">
            Continue
          </button>
        </div>
      </template>

      <div v-if="!message.streaming" class="msg__actions">
        <!-- Copy copies TEXT. An audio-only turn has none, and a button that
             silently puts an empty string on the clipboard is worse than no
             button - the clip has its own download instead. -->
        <Tooltip v-if="text" :label="copied ? 'Copied' : 'Copy'">
          <button class="pk-icon-btn msg__act" @click="copy">
            <Icon :name="copied ? 'check' : 'copy'" :size="15" />
          </button>
        </Tooltip>
        <!-- Editing is offered on any question, not only the newest: asking an
             OLD question differently is the common case, and the branch it
             makes leaves everything that followed the original intact. Not on
             an app-authored turn (`auto`) - those are reports, not words the
             user chose. -->
        <Tooltip v-if="isUser && !message.auto && !editing" label="Edit and ask again">
          <button class="pk-icon-btn msg__act" aria-label="Edit this message" @click="startEdit">
            <Icon name="edit" :size="15" />
          </button>
        </Tooltip>
        <Tooltip v-if="!isUser && isLast" label="Retry this answer">
          <button class="pk-icon-btn msg__act" @click="emit('regenerate')">
            <Icon name="regenerate" :size="15" />
          </button>
        </Tooltip>
        <Menu v-if="transcript">
          <MenuTrigger>
            <Tooltip label="Export this transcript">
              <button class="pk-icon-btn msg__act" aria-label="Export transcript">
                <Icon name="download" :size="15" />
              </button>
            </Tooltip>
          </MenuTrigger>
          <MenuContent align="start">
            <MenuItem :disabled="!transcript.segments?.length" @select="exportAs('srt')">
              SRT
            </MenuItem>
            <MenuItem :disabled="!transcript.segments?.length" @select="exportAs('vtt')">
              WebVTT
            </MenuItem>
            <MenuItem
              :disabled="!transcript.segments?.length && !transcript.words?.length"
              @select="exportAs('json')"
            >
              JSON with times
            </MenuItem>
            <MenuItem @select="exportAs('txt')">Plain text</MenuItem>
          </MenuContent>
        </Menu>
        <!-- an extraction that can only be read in the Studio is trapped in
             the Studio - same export duty as the transcript, same place -->
        <Menu v-if="docParserRun && text">
          <MenuTrigger>
            <Tooltip label="Save the extraction">
              <button class="pk-icon-btn msg__act" aria-label="Save the extraction">
                <Icon name="download" :size="15" />
              </button>
            </Tooltip>
          </MenuTrigger>
          <MenuContent align="start">
            <MenuItem @select="docExportAs('md')">Markdown</MenuItem>
            <MenuItem @select="docExportAs('txt')">Plain text</MenuItem>
            <MenuItem @select="docExportAs('json')">JSON with regions</MenuItem>
          </MenuContent>
        </Menu>
        <Tooltip v-if="hasRun" :label="showRun ? 'Hide run details' : 'Run details'">
          <button
            class="pk-icon-btn msg__act"
            :class="{ 'msg__act--on': showRun }"
            aria-label="Run details"
            @click="showRun = !showRun"
          >
            <Icon name="sliders" :size="15" />
          </button>
        </Tooltip>
        <Tooltip v-if="answerMeta" :label="answerMetaTip">
          <span class="msg__usage">{{ answerMeta }}</span>
        </Tooltip>
      </div>

      <RunDetails v-if="hasRun && showRun" :message="message" />
    </div>
  </article>
</template>

<style scoped>
.msg {
  display: flex;
  margin-bottom: 28px;
  animation: msg-in 0.3s ease;
}
/* the model stamp: who wrote this answer (multi-model history is normal now) */
.msg__htmlbar {
  display: flex;
  gap: 4px;
  margin-bottom: 6px;
}
.msg__htmlbtn {
  padding: 3px 10px;
  border: 1px solid var(--pk-border-default);
  background: none;
  border-radius: var(--pk-radius-sm);
  color: var(--pk-text-muted);
  font: inherit;
  font-size: var(--pk-font-size-xs);
  cursor: pointer;
}
.msg__htmlbtn:hover {
  color: var(--pk-text-primary);
}
.msg__htmlbtn--on {
  color: var(--pk-text-primary);
  background: var(--pk-bg-surface);
  border-color: var(--pk-border-strong);
}
.msg__htmlframe {
  width: 100%;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: transparent;
}
/* Who wrote this answer, in the compare lane's own treatment: vendor mark and
   name. It used to be a small mono caption, so the same fact looked like two
   different things depending on whether you were comparing.
   Mirrors .thread__lane-hd / .thread__lane-model.
   NO rule under it, unlike the compare lane: there the header and the prose
   share one card and the rule divides them, but a single turn's content IS a
   card, so a rule here lands a hair above that card's own top border and reads
   as a double line. */
.msg__hd {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.msg__hd-model {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.msg__auto {
  max-width: 560px;
  margin-left: auto;
  padding: 2px 10px;
  border: 1px dashed var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.msg__auto-body {
  margin: 4px 0 8px;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--pk-font-mono);
  font-size: 12px;
  color: var(--pk-text-secondary);
}
.msg__hd-spec {
  padding: 1px 6px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  font-size: 10px;
}
.msg--user {
  justify-content: flex-end;
}
.msg__col {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.msg--user .msg__col {
  align-items: flex-end;
  max-width: 85%;
}
.msg--assistant .msg__col {
  width: 100%;
}

/* user turn = right-aligned filled pill */
.msg__bubble {
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  border-radius: 20px;
  padding: 9px 15px;
  max-width: 100%;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
}

/* a turn sent from the composer's action row - an outlined pill rather than
   the filled one, because it names something the model was asked to DO */
.msg__task {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 14px;
  max-width: 100%;
  border: 1px solid var(--pk-border-default);
  border-radius: 20px;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
}

/* MCP tool-call cards (assistant turn: between thinking and the answer) */
.msg__tools {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

/* attached images */
.msg__images {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
  justify-content: flex-end;
}
/* generated pictures: the answer, at reading width. A checkerboard behind
   the picture so a transparent one shows its transparency rather than the
   page; a preview stays sharp (it is a real decode of the render so far) and
   only its clock says it is not done. */
.gen {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 10px;
  max-width: 640px;
}
.gen--grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.gen__pic {
  margin: 0;
  position: relative;
  border: 1px solid var(--pk-border-subtle);
  border-radius: 14px;
  overflow: hidden;
  background: var(--pk-bg-surface);
  line-height: 0;
}
.gen__img {
  display: block;
  width: 100%;
  height: auto;
  background:
    linear-gradient(45deg, var(--pk-bg-inset) 25%, transparent 25%, transparent 75%, var(--pk-bg-inset) 75%),
    linear-gradient(45deg, var(--pk-bg-inset) 25%, transparent 25%, transparent 75%, var(--pk-bg-inset) 75%);
  background-size: 16px 16px;
  background-position:
    0 0,
    8px 8px;
}
.gen__img--preview {
  opacity: 0.92;
}
.gen__progress,
.gen__wait {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  line-height: 1.4;
}
.gen__progress {
  position: absolute;
  left: 10px;
  bottom: 10px;
  padding: 4px 8px;
  border-radius: var(--pk-radius-full);
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
}
.gen__wait {
  padding: 6px 0;
}
.gen__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 8px 6px 12px;
  border-top: 1px solid var(--pk-border-subtle);
  line-height: 1.4;
}
.gen__facts {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.gen__acts {
  display: flex;
  gap: 2px;
}
.gen__act--on {
  color: var(--pk-accent);
}
.gen__spin {
  animation: gen-spin 0.8s linear infinite;
}
@keyframes gen-spin {
  to {
    transform: rotate(360deg);
  }
}
.msg__image-wrap {
  position: relative;
  display: inline-block;
  line-height: 0;
}
.msg__image {
  max-width: 240px;
  max-height: 240px;
  width: auto;
  height: auto;
  object-fit: cover;
  border-radius: 14px;
  border: 1px solid var(--pk-border-subtle);
  cursor: pointer;
  transition: filter 0.12s ease;
}
.msg__image-wrap:hover .msg__image {
  filter: brightness(1.05);
}
/* hover download, top-right of the thumbnail */
.msg__image-dl {
  position: absolute;
  top: 6px;
  right: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--pk-radius-full);
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  opacity: 0;
  transition: opacity 0.12s ease, background 0.12s ease;
}
.msg__image-wrap:hover .msg__image-dl {
  opacity: 1;
}
.msg__image-dl:hover {
  background: rgba(0, 0, 0, 0.75);
}
/* undecodable-format stand-in: same footprint as a thumb, labeled */
.msg__image-stub {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 120px;
  height: 90px;
  padding: 8px;
  border: 1px solid var(--pk-border-subtle);
  background: var(--pk-bg-surface);
  color: var(--pk-text-muted);
  cursor: pointer;
}
.msg__image-stub-name {
  max-width: 104px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 11px;
}

/* a transcription answer: one card holding the transport, the words and the
   facts about the run - the same surface recipe the compare lane uses */
.tx {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 12px 14px;
}
/* Decode guards. Above the transcript, not below it: it changes
   how the words underneath should be read, and a reader who has already read
   them has read them wrong. */
.tx__guards {
  margin: 0 0 10px;
  padding: 8px 10px;
  list-style: none;
  border-left: 2px solid var(--pk-status-warning);
  background: var(--pk-status-warning-subtle);
  border-radius: var(--pk-radius-sm);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
}
.tx__guards li + li {
  margin-top: 4px;
}
.tx__guard-span {
  margin-right: 6px;
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-muted);
}
/* the word-timing affordance: one quiet line under an untimed transcript */
.tx__timing {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.tx__timing-act {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
button.tx__timing-act {
  border: 0;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--pk-accent);
  cursor: pointer;
}
button.tx__timing-act:disabled {
  color: var(--pk-text-muted);
  cursor: default;
}
.tx__timing-err {
  color: var(--pk-status-warning);
}
/* already inside a lane's card - keep the content, drop the second frame */
.tx--flat {
  border: 0;
  border-radius: 0;
  background: none;
  padding: 0;
}
/* the transport under the clip's meta row: one object, so it drops its own
   border and box rather than stacking a second card on the first  */
/* the child-root selector alone only TIES with AudioPlayer's own `.ap` rule
   (both gain one attribute from scoping), so it would win or lose on bundle
   order; the parent qualifier settles it */
/* The clip is one object - the recording it is, and the transport that plays
   it - so ONE card holds both. It used to be a chip-sized card with a bare
   full-width slider hanging underneath, which reads as two unrelated things.
   The card moves out to the group and the chip inside
   drops its own frame, the same way the transport already had. */
/* Two classes, not one: `.msg__files` is defined further down this sheet and
   would otherwise win the tie on source order and put its row direction, its
   8px gap and its `justify-content: flex-end` straight back - which is what
   left the chip floating at the right edge over a full-width slider. */
/* A clip turn is a TRANSPORT, not a pill. The whole compare listens through
   this one row of controls, and shrink-wrapping it like a text bubble left the
   scrubber a sliver between the buttons - so the turn claims the message width
   a lane gets and the card stretches into it. :has(), because the column
   cannot know from above what its child is. */
/* The full conversation width, not 85%. A clip turn is a transport the whole
   compare listens through, and every pixel taken off the card comes straight
   out of the scrubber - the one control that actually needs the room.
   Nothing else about a user turn changes; :has(), because the
   column cannot know from above what its child is. */
.msg--user .msg__col:has(.msg__files.msg__clip) {
  width: 100%;
}
.msg__files.msg__clip {
  position: relative;
  flex-direction: column;
  align-items: stretch;
  align-self: stretch;
  justify-content: flex-start;
  gap: 2px;
  padding: 4px 8px 8px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.msg__files.msg__clip > .msg__file {
  max-width: none;
  border: 0;
  /* right room for the corner button, so a long file name stops before it
     instead of running underneath */
  padding: 4px 34px 4px 2px;
  background: none;
}
/* Download in the CARD's corner rather than riding the end of the name row:
   it belongs to the clip, not to the filename, and pinning it means it sits
   in the same place whatever the name's length does. */
.msg__files.msg__clip .msg__file-dl {
  position: absolute;
  top: 6px;
  right: 6px;
}
.msg__files.msg__clip > .msg__file--flat:hover {
  border-color: transparent;
  background: none;
}
.msg__files > .msg__clip-player {
  border: 0;
  background: none;
  padding: 2px 2px 0;
  min-width: 0;
}
/* In a COLUMN container the main axis is vertical, so the `flex: 1 0 100%`
   this used to carry - written when the parent was a wrapping ROW and it
   needed to claim its own full-width line - asks for 100% HEIGHT and to grow
   into whatever is left. Width is now the cross axis and `stretch` already
   gives it all of it. */
.msg__files.msg__clip > .msg__clip-player {
  flex: none;
  width: 100%;
}
/* what the run turned out to be: label/value pairs, not a sentence, so the
   language and the length read at a glance and line up between lanes */
.tx__facts {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 14px;
  margin: 12px 0 0;
  padding-top: 10px;
  border-top: 1px solid var(--pk-border-subtle);
  font-size: var(--pk-font-size-xs);
}
.tx__facts dt {
  color: var(--pk-text-muted);
}
.tx__facts dd {
  margin: 0 8px 0 0;
  color: var(--pk-text-primary);
}
/* the user's clip is a label, not a control - the player lives on the answer,
   where a click on a word has something local to seek. Its one action is
   getting the file back out (plain-storage principle: what went in comes out). */
.msg__file--flat {
  cursor: default;
}
.msg__file--flat:hover {
  border-color: var(--pk-border-subtle);
  background: var(--pk-bg-surface);
}
.msg__file-dl {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  flex: none;
  border-radius: var(--pk-radius-full);
  color: var(--pk-text-muted);
}
.msg__file-dl:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}

/* attached PDF chips */
.msg__files {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
  justify-content: flex-end;
}
.msg--assistant .msg__files {
  justify-content: flex-start;
}
/* One chip, one action: it opens the document - the pane when we can render
   the format, the tabbed dialog otherwise - with no second control fighting
   the chip's height. */
.msg__file {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  max-width: 280px;
  padding: 8px 12px 8px 9px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  cursor: pointer;
  text-align: left;
  transition: border-color 0.12s ease, background 0.12s ease;
}
.msg__file:hover {
  border-color: var(--pk-border-strong);
  background: var(--pk-bg-elevated);
}
.msg__file-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
  flex-shrink: 0;
}
.msg__file-meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.msg__file-name {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.msg__file-sub {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}

/* three-dot "typing" pill */
.msg__typing {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 11px 15px;
  background: var(--pk-bg-elevated);
  border-radius: 16px;
}
.msg__typing span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--pk-text-muted);
  animation: msg-typing 1.4s infinite ease-in-out;
}
.msg__typing span:nth-child(2) {
  animation-delay: 0.15s;
}
.msg__typing span:nth-child(3) {
  animation-delay: 0.3s;
}

/* provider errors can run to a few lines: the icon holds the first line
   and the text wraps under itself, never under the icon */
.msg__error {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
  border-radius: var(--pk-radius-md);
  padding: 8px 12px;
  font-size: var(--pk-font-size-sm);
}
.msg__error > svg {
  flex: none;
  margin-top: 3px;
}
.msg__error-text {
  min-width: 0;
  overflow-wrap: break-word;
  line-height: 1.45;
}
.msg__stopped {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  font-style: italic;
}
.msg__cut {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
  padding: 7px 12px;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-status-warning-subtle);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.msg__continue {
  border: 1px solid var(--pk-border-default);
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  border-radius: var(--pk-radius-md);
  padding: 3px 12px;
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
}
.msg__continue:hover {
  border-color: var(--pk-accent);
  color: var(--pk-accent-text);
}

/* Wide enough to edit in comfortably, but still capped so a one-line question
   does not open a full-width panel over the thread. */
.msg__edit {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: min(560px, 60vw);
}
.msg__edit-box {
  width: 100%;
  resize: none;
  overflow-y: auto;
  line-height: 1.55;
  font-family: inherit;
}
.msg__edit-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
.msg__edit-hint {
  margin-right: auto;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}

.msg__actions {
  display: flex;
  align-items: center;
  gap: 2px;
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.15s ease;
}
/* Space only when something actually follows - RunDetails opens directly under
   this row and had nothing between them. A plain margin-bottom would instead
   pad every message that has no panel open, which is the bug the user-turn
   float above was written to avoid. */
.msg__actions + * {
  margin-top: 10px;
}
/* A user turn's copy row must not RESERVE height - an invisible 32px strip
   under every bubble read as a huge gap to the next turn.
   Float it into the inter-message margin instead: absolute at the bubble's
   bottom edge, zero layout footprint. padding-top (not margin) keeps the
   hover corridor unbroken, and pointer-events gating stops the invisible
   row from stealing clicks aimed at the turn below; :hover holds while the
   pointer is over the row because it stays a descendant of .msg. */
.msg--user {
  position: relative;
  /* The floated row below is 32px tall (28px button + 4px padding-top) and
     lives INSIDE this margin by design. At the shared 28px it overhung the
     next turn by 4px and collided with it - the float has to fit the gap it
     was put in. 40px keeps it clear with a little air. */
  margin-bottom: 40px;
}
.msg--user .msg__actions {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 0;
  padding-top: 4px;
  justify-content: flex-end;
  pointer-events: none;
}
.msg--user:hover .msg__actions,
.msg--user:focus-within .msg__actions {
  pointer-events: auto;
}
.msg:hover .msg__actions,
.msg:focus-within .msg__actions {
  opacity: 1;
}
/* Studio is a testing tool - keep an assistant turn's metrics + run access
   persistently visible, not hidden behind hover. */
.msg--assistant .msg__actions {
  opacity: 1;
}
.msg__act {
  width: 28px;
  height: 28px;
}
.msg__act--on {
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
}
.msg__usage {
  margin-left: 6px;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}

@keyframes msg-in {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
}
@keyframes msg-typing {
  0%,
  60%,
  100% {
    opacity: 0.25;
  }
  30% {
    opacity: 1;
  }
}
@media (prefers-reduced-motion: reduce) {
  .msg {
    animation: none;
  }
  .msg__typing span {
    animation: none;
    opacity: 0.5;
  }
}
</style>
