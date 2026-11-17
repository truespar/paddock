<script setup lang="ts">
import { uiPreferences } from '@/lib/ui-preferences'
import { computed, defineAsyncComponent, inject, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { NATIVE_CONTENT } from '@/lib/native-content'
import { useRoute, useRouter } from 'vue-router'
import { useWindowSize } from '@vueuse/core'
import { useChatStore } from '@/stores/chat'
import { useGraphsStore } from '@/stores/graphs'
import { takesTurns, useModelsStore } from '@/stores/models'
import { useChatStream } from '@/composables/useChatStream'
import type { ContentPart } from '@/types/chat'
import { initMarkstream } from '@/lib/markstream'
import { isGraphFile, isImageFile, pagesParam, readAudioPart, readFilePart, readGraphPart, readImagePart } from '@/lib/attachments'
import { isAudioFile } from '@/lib/transcribe'
import { askedLanguage } from '@/lib/languages'
import ConversationSidebar from './ConversationSidebar.vue'
import ChatThread from './ChatThread.vue'
import AudioPlayer from './AudioPlayer.vue'
import { attachmentsApi } from '@/lib/api'
// The native workspace supplies its own composer. Do not even initialize
// Tiptap/audio controls in that surface; web keeps the same component on demand.
const Composer = defineAsyncComponent(() => import('./Composer.vue'))
import ArtifactPanel from './ArtifactPanel.vue'
// Async: the graph panel drags sigma + the traverse wasm glue with it.
const GraphPane = defineAsyncComponent(() => import('@/components/chat/graph/GraphPane.vue'))
import DocumentPane from './DocumentPane.vue'
import { docContexts, isDocParserConv, rasterContext } from '@/lib/docrun'
import { ocrModeLabel } from '@/lib/ocr'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { NEW_CHAT, TOGGLE_CHATS, isSearchChats, isToggleChats } from '@/lib/shortcuts'
import ResizeHandle from '@/components/ui/ResizeHandle.vue'
import { useArtifactsStore } from '@/stores/artifacts'
import { useReadinessStore } from '@/stores/readiness'
import { activeMessages } from '@/lib/tree'
import { useConversationEffects } from '@/composables/useConversationEffects'

const chat = useChatStore()
useConversationEffects()
const native = inject(NATIVE_CONTENT, null)
const nativeDocumentEl = ref<InstanceType<typeof DocumentPane> | null>(null)
watch(nativeDocumentEl, pane => native?.documentActions?.(pane), { flush: 'post' })
onBeforeUnmount(() => native?.documentActions?.(null))
const graphs = useGraphsStore()
// A machine with no usable card must not be told to go start a model - the
// Manager has nothing to start.
const readiness = useReadinessStore()
const artifacts = useArtifactsStore()
const models = useModelsStore()
const route = useRoute()
const router = useRouter()
const { isStreaming, send, regenerate, editAndResend, continueLast, stop } = useChatStream()

/** The languages this chat's lanes can be told - the set the composer's
 *  picker offers - so a clip is stamped with a code its lane takes. */
function clipLanguages(): string[] | undefined {
  const c = chat.active
  if (!c) return undefined
  const one = c.model && c.model !== 'default' ? c.model : models.currentId
  return models.languagesForLanes(c.compareModels?.length ? c.compareModels : one ? [one] : [])
}

// ── graph import auto-repair ─────────────────────────────────────────────
// The artifact tools return success when the TEXT stores; the import runs
// browser-side, so the model has announced finished graphs
// over failed imports three runs straight. When a seed fails, the artifacts
// store queues one report and this sends it into the conversation as a
// labeled user turn - after the streaming turn ends, never into it - so the
// model repairs with artifact_update. The store caps it at two repairs per
// artifact; past that the panel's error display is the human fallback.

initMarkstream()

/** A persisted panel width, parsed so that zero SURVIVES. The obvious
 *  `Number(uiPreferences.getItem(k)) || fallback` reads a stored "0" as falsy
 *  and hands back the default, so a collapsed history sidebar was always open
 *  again after a reload - the state could be set but not saved.
 *  Number(null) is 0, so the missing key has to be caught first
 *  or every fresh browser would start collapsed. */
function storedWidth(key: string, fallback: number): number {
  const raw = uiPreferences.getItem(key)
  if (raw === null || raw === '') return fallback
  const n = Number(raw)
  return Number.isFinite(n) && n >= 0 ? n : fallback
}

// Resizable chat-history sidebar (persisted). Drives the `--pk-sidebar-width`
// var the sidebar already consumes, so its multi-root markup needs no change.
// Layout constants, up here because the document pane's first-open width is
// computed during setup and would otherwise read them in their TDZ.
const CHAT_MIN = 300
const RAIL = 44
const HANDLE = 7
const SIDEBAR_DEFAULT = 260
const sidebarWidth = ref(native ? 0 : storedWidth('pk_sidebar_width', SIDEBAR_DEFAULT))
// Where expanding puts it back. `pk_sidebar_width` is 0 while collapsed, so
// the width you had chosen has to be remembered separately or every re-open
// lands on 260 and quietly discards a dragged layout. Seeded from the CURRENT
// width so someone who dragged theirs wide before this key existed does not
// lose it the first time they fold.
const sidebarOpenWidth = ref(
  storedWidth('pk_sidebar_open_width', sidebarWidth.value || SIDEBAR_DEFAULT),
)
watch(sidebarWidth, (w) => {
  uiPreferences.setItem('pk_sidebar_width', String(w))
  if (w > 0) {
    sidebarOpenWidth.value = w
    uiPreferences.setItem('pk_sidebar_open_width', String(w))
  }
})
const sidebarOpen = computed(() => sidebarWidth.value > 0)
/** Fold the history away / bring it back. The drag-past-110 gesture the
 *  ResizeHandle offers is the same state change; this is the half of it you
 *  can find without knowing it exists. */
function toggleSidebar(): void {
  sidebarWidth.value = sidebarOpen.value ? 0 : sidebarOpenWidth.value || SIDEBAR_DEFAULT
}

// The artifact panel appears only once a chat has produced one, and it follows
// the conversation. Re-listing when streaming stops is what makes a mid-turn
// edit show up: the tool call went to the manager, so the SSE stream never
// carried the new body (deliberately - that is the context saving).
//
// First-open width is a PROPORTION of the window, the same rule the document
// pane took, and for the same reason: 420 was
// a fair column at 1440 and a thin ribbon at 3840, because a constant cannot
// be right at two screen widths. Compare already scaled - `artifacts.panes`
// widens per lane below - so it was single chat that stayed narrow forever.
//   - CAP 900 = about as wide as a rendered page repays. Past it the panel is
//     not showing more artifact, it is taking from the conversation.
//   - FLOOR 420 = exactly today's width, so no layout narrows. Dragging still
//     reaches 280; that is a deliberate act, not a default.
//   - 40% of what is left after the history sidebar and the dividers - less
//     than the document pane's 45%, because an artifact accompanies the
//     reading where a document is the reading.
// At 1920 with the sidebar open that is ~660px against 420 before.
const ARTIFACT_CAP = 900
const ARTIFACT_FLOOR = 420
function defaultArtifactWidth(): number {
  const win = window.innerWidth || 1280
  const free = win - (storedWidth('pk_sidebar_width', SIDEBAR_DEFAULT) || RAIL) - HANDLE * 2
  return Math.max(ARTIFACT_FLOOR, Math.min(ARTIFACT_CAP, Math.round(free * 0.4)))
}
// `_v2` for the reason `pk_docpane_width_v2` exists: the old key holds 420 for
// anyone who has ever opened an artifact, a stored value must win over a
// default, so the better default would never reach them. The new key hands it
// out once; every drag after that is theirs.
const artifactWidth = ref(storedWidth('pk_artifact_width_v2', defaultArtifactWidth()))
watch(artifactWidth, (w) => uiPreferences.setItem('pk_artifact_width_v2', String(w)))

// The graph is its own side panel, never a lodger in the artifact column
// ("that is its own sidepanel"). First-open width is a
// PROPORTION like the document pane's - a canvas earns more room than an
// artifact (45% of free, floored high enough that labels breathe).
function defaultGraphWidth(): number {
  const win = window.innerWidth || 1280
  const free = win - (storedWidth('pk_sidebar_width', SIDEBAR_DEFAULT) || RAIL) - HANDLE * 2
  return Math.max(480, Math.min(1000, Math.round(free * 0.45)))
}
const graphPaneWidth = ref(storedWidth('pk_graphpane_width_v1', defaultGraphWidth()))
watch(graphPaneWidth, (w) => uiPreferences.setItem('pk_graphpane_width_v1', String(w)))

// Document mode divides the MAIN AREA into columns and the first one is the
// document: the pane shows the conversation's sticky
// document at full height; the chat (and its compare lanes) shares the rest.
//
// The first-open width is a PROPORTION of the window, not a constant:
// "the current default is horrible ... if the screen isn't wide I
// get it". A constant is exactly the thing that cannot be right twice: 560px
// is a fair split at 1280 and an absurd one at 2560, where it leaves a third
// of the screen empty beside a document squeezed to half size.
//
// The numbers come from what a page actually needs. A US Letter page is ~816
// CSS px at 96dpi (A4 ~794), scriptor lays a letter page out at ~1123, and
// lector's thumbnail rail (open by default) takes ~150 of whatever it gets.
// So:
//   - CAP 1000 = a full page at 100% plus lector's rail. Past that the column
//     is not showing more document, just more margin.
//   - FLOOR 460 = below this lector's toolbar starts wrapping. Dragging can
//     still go to 320; that is a deliberate act, not a default.
//   - 45% of what is left after the history sidebar and the dividers, so the
//     chat keeps the larger half at every size.
// At 1920 with the sidebar open that is 741px - a page at ~72% in lector and a
// Word page that fits - against 560 before.
const DOCPANE_CAP = 1000
const DOCPANE_FLOOR = 460
function defaultDocPaneWidth(): number {
  const win = window.innerWidth || 1280
  const free = win - (storedWidth('pk_sidebar_width', SIDEBAR_DEFAULT) || RAIL) - HANDLE * 2
  return Math.max(DOCPANE_FLOOR, Math.min(DOCPANE_CAP, Math.round(free * 0.45)))
}
// `_v2`: the old key holds 560 for anyone who has opened a document before, and
// a stored value must win over a default - so the improved default would never
// reach them. A new key hands out the new default once; every drag after that
// is theirs and sticks.
const docPaneWidth = ref(storedWidth('pk_docpane_width_v2', defaultDocPaneWidth()))
watch(docPaneWidth, (w) => uiPreferences.setItem('pk_docpane_width_v2', String(w)))

// ── the document pane: one viewer per format, reached the same way ─────────
// It used to appear only for document-parser models, and a PDF in an ordinary
// chat opened a modal with a cut-down viewer instead - two integrations of the
// same library with different toolsets. Now there is one:
// clicking a PDF anywhere brings it up here, with the toolbar, thumbnails,
// search and text selection the parser lane always had. Word documents took
// the same road - scriptor renders them in this pane, not in a
// dialog, so "open a document" means one thing whatever the format is.
const docs = computed(() => docContexts(chat.active))
// A parser conversation is about its document, so its pane opens by itself.
const docAuto = computed(() => !!chat.active && isDocParserConv(chat.active, models.caps))
// Offered at all - for every document the conversation holds, photos included
// The two branches this used to have were the lightbox's fault:
// an ordinary chat counted PDFs and Word documents only, because a photo had
// its own dialog and a pane behind it would have been a second way to see the
// same thing. The lightbox is gone, so a picture is a document like any other.
const docAvailable = computed(() => docs.value.length > 0)
const docOpen = computed(() => docAvailable.value && (chat.active?.docPaneOpen ?? docAuto.value))
// The rail stands only once the pane has actually been folded. A chat that
// merely CONTAINS a PDF must not grow a column nobody asked for - until you
// have opened the document, the chip in the thread is the way in.
const docRail = computed(() => docAvailable.value && chat.active?.docPaneOpen === false)
function toggleDocPane(): void {
  if (native?.preview.value) { native.preview.value = null; return }
  chat.setDocPane(!docOpen.value)
}
// How wide the artifacts may get is whatever is left once the chat keeps a
// readable column - not a constant. A fixed 1100 ceiling meant a wide screen
// could not push the chat any narrower no matter how much room it had, which
// is backwards: the bigger the screen, the more the artifacts should be able
// to take.
const { width: winWidth } = useWindowSize()
/** Everything in the row that is not this panel and not the chat: the history
 *  (a panel or its rail), the document rail when the document is folded, and
 *  every divider standing between them. */
function othersWidth(otherPanes: number): number {
  const history = native ? 0 : sidebarOpen.value ? sidebarWidth.value : RAIL
  const up = graphUp.value
  const handles =
    HANDLE * (1 + (docOpen.value ? 1 : 0) + (artifactsOpen.value ? 1 : 0) + (up ? 1 : 0))
  const graphRail = graphHere.value && graphs.folded ? RAIL : 0
  const artRail = artifactsRail.value ? RAIL : 0
  return history + (docRail.value ? RAIL : 0) + graphRail + artRail + otherPanes + handles
}
// The session must BELONG to the conversation on screen - gating on
// graphs.active alone leaked the previous chat's graph into a fresh
// composer when the release-on-switch watch missed the draft timing
// Visibility is keyed, release is just hygiene.
const graphHere = computed(() => graphs.active && graphs.conversationId === chat.active?.id)
const graphUp = computed(() => graphHere.value && !graphs.folded)
/** The artifact panel folds per conversation, exactly like the document
 *  pane - every side panel makes the same bargain: a rail, never vanishing. */
const artifactsOpen = computed(() => artifacts.any && (chat.active?.artifactsPaneOpen ?? true))
const artifactsRail = computed(() => artifacts.any && chat.active?.artifactsPaneOpen === false)
/** What a panel may grow to: whatever is left once the chat keeps a readable
 *  column. Each ceiling subtracts the other panel, because otherwise both can
 *  sit inside their own maximum and still overflow the row together.
 *
 *  The document pane used to borrow `artifactMax` outright - a ceiling
 *  computed for a different panel, blind to the document's own width. Found
 *  from a live handle reporting aria-valuenow 856 against
 *  aria-valuemax 792. */
const artifactMax = computed(() =>
  Math.max(320, Math.round(winWidth.value - othersWidth(
    (docOpen.value ? docPaneWidth.value : 0) + (graphUp.value ? graphPaneWidth.value : 0),
  ) - CHAT_MIN)),
)
const docMax = computed(() =>
  Math.max(320, Math.round(winWidth.value - othersWidth(
    (artifactsOpen.value ? artifactWidth.value : 0) + (graphUp.value ? graphPaneWidth.value : 0),
  ) - CHAT_MIN)),
)
const graphMax = computed(() =>
  Math.max(320, Math.round(winWidth.value - othersWidth(
    (artifactsOpen.value ? artifactWidth.value : 0) + (docOpen.value ? docPaneWidth.value : 0),
  ) - CHAT_MIN)),
)
watch(graphMax, (m) => {
  if (graphPaneWidth.value > m) graphPaneWidth.value = m
})
// A window that shrinks under the current split has to give the chat its floor
// back; the handle only clamps while it is being DRAGGED, so nothing else was
// watching. Both only ever shrink, so the two watchers settle rather than
// chase each other.
// `immediate` because the FLOOR can outrun the ceiling on a small window: 420
// is wider than what is left at ~900px with the sidebar open, and a seed is
// never a change, so nothing used to clamp it until you happened to resize.
watch(
  artifactMax,
  (m) => {
    if (artifactWidth.value > m) artifactWidth.value = m
  },
  { immediate: true },
)
watch(docMax, (m) => {
  if (docPaneWidth.value > m) docPaneWidth.value = m
})
// The moment a compare turn gives a second (or third, or fourth) model an
// artifact, widen the panel enough to actually sit them next to each other -
// that is the whole point of comparing, and a 420px panel silently falls back
// to tabs. Only ever widens, and never past what leaves the chat usable, so a
// width the user has since dragged wider is left alone.
watch(
  () => artifacts.panes.length,
  (n, was) => {
    if (n < 2 || n <= (was ?? 0)) return
    artifactWidth.value = Math.max(artifactWidth.value, Math.min(n * 390, artifactMax.value))
  },
)

// Files staged for the next turn - shared with the composer (v-model) so both a
// drop anywhere in the chat and the paperclip picker feed the same tray.
const files = ref<File[]>([])
const composerRef = ref<InstanceType<typeof Composer> | null>(null)
const sidebarRef = ref<InstanceType<typeof ConversationSidebar> | null>(null)

// ── how much room the floating composer needs at the thread's foot ────────
// In a chat the composer floats on the thread surface, so the thread has to
// reserve its height at the bottom or the last message sits under it. The
// height is not a constant - it grows with multi-line input, the file tray,
// the hint line and the tool row folding - so it is measured off the
// composer's own root and published as a var the thread reads.
const mainEl = ref<HTMLElement | null>(null)
const nativeColumnEl = ref<HTMLElement | null>(null)
let composerRo: ResizeObserver | undefined
let observedThread: HTMLElement | null = null
let observedColumn: HTMLElement | null = null
const observedWidths = new WeakMap<Element, number>()
function composerEl(): HTMLElement | null {
  const el = composerRef.value?.$el
  return el instanceof HTMLElement ? el : null
}
function measureComposer(): void {
  if (native && mainEl.value) {
    // Publish the actual content box, including scrollbar/panel/compare
    // geometry. Swift must not independently cap or inset this a second time.
    const column = mainEl.value.querySelector<HTMLElement>('.thread__inner') ?? nativeColumnEl.value
    if (!column) return
    const bounds = column.getBoundingClientRect()
    native.viewport(bounds.left, bounds.width)
    return
  }
  const el = composerEl()
  if (el && mainEl.value) mainEl.value.style.setProperty('--pk-composer-h', `${el.offsetHeight}px`)
}
/** The thread's scrollbar width. The floating composer and the jump button
 *  both stop short of it, so they line up with the message column instead of
 *  the raw pane - otherwise the card sits half a scrollbar to the right and,
 *  on a pane under the chat width, is that much wider than the messages.
 *  0 on overlay-scrollbar platforms, which is why this only shows on Windows. */
function measureGutter(el: HTMLElement): void {
  mainEl.value?.style.setProperty('--pk-sbw', `${el.offsetWidth - el.clientWidth}px`)
}

// Two draft surfaces, one mechanism: `/` is the start page (clean landing, no
// sidebar), `/chat/new` is "New chat" from inside the chat (sidebar stays).
// Both hold an uncommitted draft; sending pushes to /chat/:id.
const isHome = computed(() => route.name === 'home')
// Nothing that ANSWERS A TURN - drives the no-servers hint. A transcriber
// counts now: it holds a lane here and the composer switches to
// audio, so a box running one is not an empty Studio.
const hasTurnModel = computed(() => models.models.some((m) => takesTurns(m.kind)))
const soleEncoderName = computed(() => {
  if (hasTurnModel.value) return null
  const e = models.models.find((m) => m.kind === 'encoder')
  return e ? (e.display ?? e.id) : null
})
const isNew = computed(() => route.name === 'chat-new')
const isDraft = computed(() => isHome.value || isNew.value)

// An UNSENT draft follows the fleet seat.
//
// A draft is stamped with whatever `models.currentId` said when it was opened,
// and the seat moves on its own when a runner comes up. Start paddock with only
// a cloud model, land on the Studio (draft stamped cloud), then start a local
// model: the seat moves, the header follows it, and the draft did not - so the
// send went to the cloud model the header had stopped naming.
//
// The header no longer lies (effectiveModelId), but showing "cloud" there would
// still be the wrong answer: someone who just started a local model and walked
// into the Studio means to use it. A draft has no history to protect, so it
// adopts the seat. A committed conversation never does - its model is a
// decision, and a runner coming up is not a reason to overrule it.

// The attached graph follows the conversation on screen: reopening a chat
// that carries a .tvdb re-establishes its session (OPFS makes that cheap);
// switching to one that doesn't releases the worker - a 100 MB graph is not
// something to keep resident for a background chat.

/** Called on send - the only place a conversation comes into being. On a draft
 *  surface the draft becomes real here; in a chat we're appending to the one
 *  already open (the newConversation fallback is just a guard). */
function ensureConversation(): void {
  const model = models.currentId || 'default'
  const c = isDraft.value
    ? chat.commitDraft(model)
    : (chat.active ?? chat.newConversation(model))
  // The model id arrives from an async /v1/models fetch, so a draft opened on
  // mount can still say "default" by the time it's sent.
  if (!c.model || c.model === 'default') c.model = model
}

/** Recording into the thread needs the same conversation a send
 *  needs, and needs it before the first word - the mic writes turns as you
 *  speak, so there has to be somewhere for them to go and it has to be on
 *  screen. Same two steps `onSubmit` takes, minus the content.
 *
 *  Without it the turn landed on the uncommitted draft, which the start page
 *  does not render at all: record said "Listening" and nothing appeared
 *  anywhere (found live). */
function onListen(): void {
  ensureConversation()
  const id = chat.active?.id
  if (isDraft.value && id) void router.push({ name: 'chat', params: { id } })
}

// Resolve the route's :id to the active conversation. /chat RESUMES - it never
// creates. A chat only comes into being when you send from a draft surface.
/** The failed state's button: ask for the chat's document again. */
function retryOpen(): void {
  const id = chat.active?.id
  if (id) void chat.ensureLoaded(id)
}

async function syncRoute(): Promise<void> {
  // Draft surfaces own no id: open a fresh draft, leave the sidebar with
  // nothing selected, and put the cursor where the user is headed anyway.
  if (isDraft.value) {
    chat.startDraft(models.currentId || 'default')
    void nextTick(() => composerRef.value?.focus())
    return
  }
  const routeId = typeof route.params.id === 'string' ? route.params.id : ''
  // A valid route id wins; a bare (or stale) /chat resumes the last-open chat.
  const id = routeId && chat.conversations.some((c) => c.id === routeId) ? routeId : chat.lastOpenId()
  // Nothing to resume (no chats yet, or they were all deleted) -> the start
  // page. Conjuring an empty chat here is what made /chat feel like it always
  // started a new one.
  if (!id) {
    await router.replace({ name: 'home' })
    return
  }
  // Always select - this ensures the messages are fetched (the landing bug was
  // a highlighted-but-empty chat because nothing loaded the doc).
  chat.select(id)
  if (routeId !== id) await router.replace({ name: 'chat', params: { id } })
}

onMounted(async () => {
  // Conversations are the critical path; a models fetch failure must not abort
  // route sync (that left the chat highlighted but never loaded).
  await chat.hydrate()
  void models.refresh()
  void readiness.load()
  await syncRoute() // on `/` this opens the draft; elsewhere it resolves the id
  // App-wide guard: a file dropped outside the zone must not navigate the tab
  // to the file. We swallow drops everywhere and only ingest inside the zone.
  window.addEventListener('dragover', preventNav)
  window.addEventListener('drop', preventNav)
  window.addEventListener('keydown', onKey)
  composerRo = new ResizeObserver(entries => {
    if (native) {
      // Streaming changes the inner column's height; it must not turn layout
      // measurement into per-token work. Only inline-size changes matter here.
      let changed = false
      for (const entry of entries) {
        if (observedWidths.get(entry.target) !== entry.contentRect.width) changed = true
        observedWidths.set(entry.target, entry.contentRect.width)
      }
      if (!changed) return
    }
    measureComposer()
    const t = threadEl()
    if (t) measureGutter(t)
  })
  const el = composerEl()
  if (el) composerRo.observe(el)
  if (native && mainEl.value) composerRo.observe(mainEl.value)
  if (nativeColumnEl.value) composerRo.observe(nativeColumnEl.value)
  watchThread()
  measureComposer()
  native?.mounted()
})
/** The thread lives inside ChatThread, so it is found rather than passed -
 *  observing it is what catches the scrollbar APPEARING (the content box
 *  shrinks by its width), not just the window resizing. Re-found when the
 *  route swaps the thread in and out. */
function threadEl(): HTMLElement | null {
  return mainEl.value?.querySelector('.thread') ?? null
}
function watchThread(): void {
  if (observedThread) composerRo?.unobserve(observedThread)
  if (observedColumn) composerRo?.unobserve(observedColumn)
  const t = threadEl()
  observedThread = t
  observedColumn = native ? mainEl.value?.querySelector<HTMLElement>('.thread__inner') ?? null : null
  if (t && composerRo) {
    composerRo.observe(t)
    measureGutter(t)
  }
  if (observedColumn) composerRo?.observe(observedColumn)
  if (native) measureComposer()
}
watch([isHome, () => chat.activeLoading], () => void nextTick(watchThread))
// Native document-only mode has no web chat column. Rebind when the shared
// renderer returns; observers attached just at mount would keep reporting the
// old width (or zero) after a native -> shared transition.
watch([mainEl, nativeColumnEl], ([main, column], [oldMain, oldColumn]) => {
  if (!native) return
  if (oldMain) composerRo?.unobserve(oldMain)
  if (oldColumn) composerRo?.unobserve(oldColumn)
  if (main) composerRo?.observe(main)
  if (column) composerRo?.observe(column)
  watchThread()
}, { flush: 'post' })
// The composer is an async component: on a cold load its ref is still empty
// when onMounted measures, so the element was neither observed nor measured
// and the thread kept the 96px fallback under a 160px+ composer - the last
// message's footer sat behind it until something else happened to resize the
// thread. Bind when the ref resolves.
watch(composerRef, () => {
  const el = composerEl()
  if (!el || !composerRo) return
  composerRo.observe(el)
  measureComposer()
}, { flush: 'post' })
// Back/forward + sidebar clicks change the URL; follow it. Watch the route NAME
// too, not just the id: `/` ⇄ `/chat/:id` is a name change, and going home has
// to mint a fresh draft.
watch(() => [route.name, route.params.id], () => void syncRoute())
onBeforeUnmount(() => {
  window.removeEventListener('dragover', preventNav)
  window.removeEventListener('drop', preventNav)
  window.removeEventListener('keydown', onKey)
  composerRo?.disconnect()
})
function preventNav(e: DragEvent): void {
  if (dragHasFiles(e)) e.preventDefault()
}
/** The fold shortcut. Deliberately still live while the composer has focus -
 *  it types nothing, and having to leave the text box to hide a panel is the
 *  kind of friction that makes a shortcut go unused. Silent on the start page,
 *  which has no history sidebar to fold. */
function onKey(e: KeyboardEvent): void {
  if (native) return // AppKit owns application shortcuts.
  if (isHome.value) return
  if (isToggleChats(e)) {
    e.preventDefault()
    toggleSidebar()
    return
  }
  if (isSearchChats(e)) {
    e.preventDefault()
    // Unfold first if it is away: a chord that silently does nothing because
    // the panel it targets is hidden is worse than no chord. The sidebar is
    // v-if'd, so the focus has to wait for it to exist.
    if (!sidebarOpen.value) sidebarWidth.value = sidebarOpenWidth.value || SIDEBAR_DEFAULT
    void nextTick(() => sidebarRef.value?.focusSearch())
  }
}

// ── drag & drop over the chat region ─────────────────────────────────────
const dragging = ref(false)
// What is being dragged, so the overlay does not offer to attach an image
// when it is plainly a sound file. The browser withholds file NAMES during a
// drag (a privacy rule) but does expose each item's MIME type, which is
// exactly enough to tell audio from everything else.
const dragAudio = ref(false)
let dragDepth = 0
function dragHasFiles(e: DragEvent): boolean {
  return Array.from(e.dataTransfer?.types ?? []).includes('Files')
}
function onDragEnter(e: DragEvent): void {
  if (!dragHasFiles(e)) return
  e.preventDefault()
  dragDepth++
  const items = Array.from(e.dataTransfer?.items ?? []).filter((i) => i.kind === 'file')
  dragAudio.value =
    items.length > 0 && items.every((i) => i.type.startsWith('audio/') || i.type === 'video/webm')
  dragging.value = true
}
function onDragOver(e: DragEvent): void {
  if (!dragHasFiles(e)) return
  e.preventDefault()
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
}
function onDragLeave(e: DragEvent): void {
  if (!dragHasFiles(e)) return
  dragDepth--
  if (dragDepth <= 0) {
    dragDepth = 0
    dragging.value = false
  }
}
function onDrop(e: DragEvent): void {
  if (!dragHasFiles(e)) return
  e.preventDefault()
  dragDepth = 0
  dragging.value = false
  if (native) return // Native file grants/upload; never navigate to a file URL.
  const dropped = Array.from(e.dataTransfer?.files ?? [])
  if (dropped.length) void composerRef.value?.addFiles(dropped)
}

async function onSubmit(text: string): Promise<void> {
  ensureConversation()
  const staged = files.value
  // Read the per-file choices before emptying the tray: the composer drops its
  // per-file state the moment a file leaves `files`, and the loop below awaits,
  // so by the first await the watcher has already run.
  const chosen = new Map(staged.map((f) => [f, composerRef.value?.detailFor(f) ?? 'auto']))
  const docOpts = new Map(staged.map((f) => [f, composerRef.value?.docOptsFor(f) ?? {}]))
  files.value = []
  const parts: ContentPart[] = []
  if (text) parts.push({ type: 'text', text })
  for (const f of staged) {
    const o = docOpts.get(f) ?? {}
    const range = pagesParam(o)
    try {
      if (isAudioFile(f)) {
        // A clip makes this turn a TRANSCRIPTION: the send path
        // routes it to /v1/audio/transcriptions, one lane per armed model.
        // The chat's language choice is stamped onto the part, so the turn
        // records what actually rode even if the setting changes later.
        parts.push(
          await readAudioPart(f, chat.active?.id, askedLanguage(chat.active?.audioLanguage, clipLanguages())),
        )
      } else if (isGraphFile(f)) {
        // A .tvdb never rides to a model - it becomes the conversation's live
        // graph session; the part only records the attachment. Awaited so the
        // session is READY before this turn's request leaves - otherwise the
        // model's first graph_query beats the load.
        const part = await readGraphPart(f, chat.active?.id)
        parts.push(part)
        const bytes = new Uint8Array(await f.arrayBuffer())
        if (chat.active) await graphs.ensure(chat.active.id, part.attachmentId, part.name, bytes, chat.active.messages)
      } else if (isImageFile(f)) {
        // the per-image size the composer's menu set (defaults to 'auto');
        // a multi-page TIFF carries its own page range
        const part = await readImagePart(f, chat.active?.id, chosen.get(f) ?? 'auto')
        if (range) part.pageRange = range
        parts.push(part)
      } else {
        // everything else is a document: bytes to the attachments table,
        // sent as input_file - the server extracts or refuses honestly. The
        // file's own route/range settings ride on the part.
        const part = await readFilePart(f, chat.active?.id)
        if (o.text) part.pdfMode = 'text'
        if (range) part.pageRange = range
        parts.push(part)
      }
    } catch {
      /* skip an attachment we couldn't read/store */
    }
  }
  // A document parser's mode-driven send carries no text and no files: the
  // sticky document (lib/docrun.ts) is re-run with the chosen reading mode.
  // The turn still needs a visible marker in the thread - the mode's name is
  // exactly what was asked (the wire carries the mode object, never this
  // label; fixed-vocabulary families take no free text at all).
  if (parts.length === 0) {
    const conv = chat.active
    if (conv && models.caps[conv.model]?.docParser && rasterContext(conv)) {
      const mode = conv.ocrMode
      parts.push({ type: 'text', text: mode ? `Read as: ${ocrModeLabel(mode)}` : 'Read again' })
    } else {
      return
    }
  }
  const id = chat.active?.id
  // Fire the stream, then land in the chat. Safe in this order: useChatStream's
  // state is module-level and nothing aborts on unmount, so the stream keeps
  // writing to the store across the navigation.
  void send(parts)
  if (isDraft.value && id) void router.push({ name: 'chat', params: { id } })
}

// "New chat" = a fresh draft inside the chat layout. Still no empty
// conversation until something is actually sent.
function newChat(): void {
  void router.push({ name: 'chat-new' })
}
</script>

<template>
  <div
    class="chatview"
    :class="{ 'chatview--native': native }"
    :style="{ '--pk-sidebar-width': `${sidebarWidth}px` }"
    @dragenter="onDragEnter"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
  >
    <!-- Swift owns both columns in native transcript mode. Keep the same
         authenticated workspace/orchestrator, mounting only the selected
         document here. Closing unmounts its canvases and viewer resources. -->
    <section v-if="native?.audioPreview?.value && !native?.nativeMarkdown?.value" class="native-audio-preview">
      <header>
        <span>{{ native.audioPreview.value.name || 'Audio clip' }}</span>
        <button class="pk-icon-btn" type="button" aria-label="Close audio preview" @click="native.audioPreview.value = null"><Icon name="x" :size="16" /></button>
      </header>
      <AudioPlayer :src="attachmentsApi.url(native.audioPreview.value.attachmentId)" :type="native.audioPreview.value.mime"
        :clip="native.audioPreview.value.attachmentId" :fallback="native.audioPreview.value.durationS" />
    </section>
    <template v-else-if="native?.nativeMarkdown?.value">
      <DocumentPane v-if="native.document?.value" ref="nativeDocumentEl" :key="native.document.value.id"
        :conversation="native.document.value" embedded class="native-document-surface" />
    </template>
    <template v-else>
    <!-- The start page is a clean landing: no history sidebar, nothing lit in
         the activity bar. Both appear once you're in a chat. -->
    <!-- The history is foldable two ways: the control in its own header, and
         dragging the divider left past the fold point - which is what a wide
         screen wants when the artifacts deserve the room. Folded, it comes
         back from the button over the thread, the same shortcut, or dragging
         the divider back out. -->
    <template v-if="!native && !isHome">
      <ConversationSidebar
        v-if="sidebarOpen"
        ref="sidebarRef"
        @new-chat="newChat"
        @fold="toggleSidebar"
      />
      <aside v-else class="chatrail">
        <Tooltip :label="`Show chats (${TOGGLE_CHATS})`" side="right">
          <button
            class="pk-icon-btn chatrail__btn"
            type="button"
            aria-label="Show chats"
            @click="toggleSidebar"
          >
            <Icon name="panel-left" :size="18" />
          </button>
        </Tooltip>
        <Tooltip :label="`New chat (${NEW_CHAT})`" side="right">
          <button
            class="pk-icon-btn chatrail__btn"
            type="button"
            aria-label="New chat"
            @click="newChat"
          >
            <Icon name="plus" :size="18" />
          </button>
        </Tooltip>
      </aside>
      <ResizeHandle
        v-model="sidebarWidth"
        side="left"
        :min="150"
        :max="480"
        :collapse-at="110"
      />
    </template>
    <template v-if="native?.preview.value || (!isDraft && docOpen)">
      <DocumentPane
        :conversation="native?.preview.value ?? chat.active ?? undefined"
        @selection="c => { if (!native?.preview.value) chat.persist(c) }"
        :style="{ '--pk-docpane-width': `${docPaneWidth}px` }"
        @fold="toggleDocPane"
      />
      <ResizeHandle v-model="docPaneWidth" side="left" :min="320" :max="docMax" />
    </template>
    <!-- Folded, the document keeps a rail of its own rather than vanishing:
         same bargain as the history's, and without it the only way back would
         be scrolling the thread to find the chip you opened it from. -->
    <aside v-else-if="!isDraft && docRail" class="chatrail chatrail--doc">
      <Tooltip label="Show the document" side="right">
        <button
          class="pk-icon-btn chatrail__btn"
          type="button"
          aria-label="Show the document"
          @click="toggleDocPane"
        >
          <Icon name="file-text" :size="18" />
        </button>
      </Tooltip>
    </aside>
    <div ref="mainEl" class="chatview__main" :class="{ 'chatview__main--home': isHome }">
      <div v-if="native" ref="nativeColumnEl" class="native-content-column" aria-hidden="true" />
      <div v-if="!isHome && chat.activeLoading" class="chatview__msg" role="status">
        <Icon name="spinner" :size="20" class="chatview__spin" />
        <span>Opening...</span>
      </div>
      <div v-else-if="!isHome && chat.activeLoadFailed" class="chatview__msg" role="alert">
        <span>This conversation could not be opened.</span>
        <button class="pk-btn pk-btn--sm" type="button" @click="retryOpen">Try again</button>
      </div>
      <ChatThread
        v-else-if="!isHome && !native?.nativeMarkdown?.value"
        @regenerate="regenerate"
        @continue-reply="continueLast"
        @edit="editAndResend"
      />
      <div v-if="!native && isHome && !models.loading && !hasTurnModel" class="home__noservers">
        <Icon name="server" :size="15" />
        <span v-if="soleEncoderName">
          <RouterLink :to="{ name: 'embeddings' }"
            ><strong>{{ soleEncoderName }}</strong></RouterLink
          >
          is running - it embeds and scores text rather than chats. Start a chat model in the
          <RouterLink :to="{ name: 'servers' }">Manager</RouterLink>.
        </span>
        <span v-else-if="readiness.notice?.state === 'no-card'">
          This computer has no graphics card Paddock can use - but cloud models work normally.
          <RouterLink :to="{ name: 'cloud' }">Add one</RouterLink>, or see
          <RouterLink :to="{ name: 'gpus' }">which graphics cards run models</RouterLink>.
        </span>
        <span v-else>
          No models are running - start one in the
          <RouterLink :to="{ name: 'servers' }">Manager</RouterLink> first.
        </span>
      </div>
      <!-- One Composer either way: on `/` it's the centred hero, in a chat it
           docks at the bottom. Same instance = same toolbar, same file tray. -->
      <Composer
        v-if="!native"
        ref="composerRef"
        v-model:files="files"
        :busy="isStreaming"
        :docked="!isHome"
        :inert="!isHome && (chat.activeLoading || chat.activeLoadFailed)"
        :class="{ 'chatview__composer--waiting': !isHome && (chat.activeLoading || chat.activeLoadFailed) }"
        @submit="onSubmit"
        @listen="onListen"
        @stop="stop"
      />
    </div>

    <template v-if="!isDraft && graphUp">
      <ResizeHandle v-model="graphPaneWidth" side="right" :min="320" :max="graphMax" />
      <GraphPane :style="{ '--pk-graphpane-width': `${graphPaneWidth}px` }" />
    </template>
    <template v-if="!isDraft && artifactsOpen">
      <ResizeHandle v-model="artifactWidth" side="right" :min="280" :max="artifactMax" />
      <ArtifactPanel :style="{ '--pk-artifact-width': `${artifactWidth}px` }" />
    </template>
    <aside v-if="!isDraft && artifactsRail" class="chatrail chatrail--graph">
      <Tooltip label="Show the artifacts" side="left">
        <button
          class="pk-icon-btn chatrail__btn"
          type="button"
          aria-label="Show the artifacts"
          @click="chat.setArtifactsPane(true)"
        >
          <Icon name="box" :size="18" />
        </button>
      </Tooltip>
    </aside>
    <!-- Folded, the graph keeps a rail - the document pane's bargain: the way
         back stays visible instead of the panel silently vanishing. -->
    <aside v-if="!isDraft && graphHere && graphs.folded" class="chatrail chatrail--graph">
      <Tooltip label="Show the graph" side="left">
        <button
          class="pk-icon-btn chatrail__btn"
          type="button"
          aria-label="Show the graph"
          @click="graphs.folded = false"
        >
          <Icon name="hard-drive" :size="18" />
        </button>
      </Tooltip>
    </aside>

    <Transition name="drop">
      <div v-if="dragging" class="dropzone">
        <div class="dropzone__card">
          <span class="dropzone__icon">
            <Icon :name="dragAudio ? 'microphone' : 'image'" :size="30" />
          </span>
          <p class="dropzone__text">
            {{
              dragAudio
                ? 'Drop the audio here to transcribe it'
                : 'Drop files here, or paste from your clipboard to attach them'
            }}
          </p>
        </div>
      </div>
    </Transition>
    </template>
  </div>
</template>

<style scoped>
/* A chat whose document has not arrived yet (or did not): same recipe as the
   document pane's "Opening...", centred where the thread will be. */
.chatview__msg {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 0 24px;
  text-align: center;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.chatview__spin {
  animation: chatview-spin 0.8s linear infinite;
}
@keyframes chatview-spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .chatview__spin {
    animation: none;
  }
}
/* inert already takes the composer out of reach; this says so */
.chatview__composer--waiting {
  opacity: 0.55;
}
.chatview {
  position: relative;
  display: flex;
  height: 100%;
  width: 100%;
  min-height: 0;
}
.chatview__main {
  /* positioning context for the composer floating on the thread */
  position: relative;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
  min-height: 0;
}
/* The folded history: a rail, not nothing.
   A button floating on the thread was the first attempt and it went unfound -
   there is no header row above the thread for it to belong to, so it read as
   a stray glyph and people reached for the divider instead.
   The rail costs 44 of the 260 folding just bought and gives the control a
   home, which is the trade every editor and chat app of this shape makes.
   New chat rides along because it is the one action worth keeping while the
   list is away.
   Surface-coloured on purpose: the resizer's left-hand side is then the same
   colour folded or open, so the seam does not change character. And no
   border-right - the ResizeHandle draws the divider. */
.chatrail {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  width: 44px;
  height: 100%;
  padding: 12px 0;
  background: var(--pk-bg-surface);
}
/* 34px, and the first one's centre lands on the same line as the fold button
   it replaced - the sidebar's 12px top padding over a 34px row */
.chatrail__btn {
  width: 34px;
  height: 34px;
}
/* The document's rail is the only panel edge in the chat with no ResizeHandle
   after it - a rail has no width to drag - so here the border IS the divider
   rather than a double one. */
.chatrail--doc {
  border-right: 1px solid var(--pk-border-default);
}
/* Its right-edge mirror, same no-handle reasoning. */
.chatrail--graph {
  border-left: 1px solid var(--pk-border-default);
}

/* Studio with no servers: the honest empty state, pointing at the Manager */
.home__noservers {
  display: flex;
  /* icon tops the first line when the encoder sentence wraps */
  align-items: flex-start;
  gap: 8px;
  /* never wider than composer__box: .composer caps at the chat width with
     24px side padding, so the visible box is that minus 48. Matching the
     outer width left this notice standing proud of the box. */
  max-width: calc(var(--pk-chat-width) - 48px);
  width: 100%;
  margin: 14px auto 0;
  padding: 8px 14px;
  border: 1px solid var(--pk-status-warning);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
  width: fit-content;
}
.home__noservers a {
  color: var(--pk-accent);
  font-weight: 600;
}
.home__noservers svg {
  flex: none;
  margin-top: 2px;
}

/* start page: no thread, so the composer is centred on its own. Sits a little
   above true centre - optically centred beats mathematically centred. */
.chatview__main--home {
  justify-content: center;
  padding-bottom: 6vh;
}

/* full-area drag overlay */
.dropzone {
  position: absolute;
  inset: 0;
  z-index: 60;
  pointer-events: none;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--pk-bg-overlay);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}
.dropzone__card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  max-width: 340px;
  padding: 40px 44px;
  text-align: center;
  background: var(--pk-bg-elevated);
  border: 1.5px dashed var(--pk-border-strong);
  border-radius: 16px;
  box-shadow: var(--pk-shadow-xl);
}
.dropzone__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: var(--pk-radius-xl);
  background: var(--pk-accent-subtle);
  color: var(--pk-accent);
}
.dropzone__text {
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.drop-enter-active,
.drop-leave-active {
  transition: opacity 0.2s ease;
}
.drop-enter-from,
.drop-leave-to {
  opacity: 0;
}
</style>
