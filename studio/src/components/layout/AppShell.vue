<script setup lang="ts">
import { uiPreferences } from '@/lib/ui-preferences'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useTheme } from '@/composables/useTheme'
import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'
import { usePushStore } from '@/stores/push'
import { useTelemetryStore } from '@/stores/telemetry'
import { useRegistryStore } from '@/stores/registry'
import { reloadHeld } from '@/lib/reload'
import { isNewChat } from '@/lib/shortcuts'
import ActivityBar from './ActivityBar.vue'
import AppHeader from './AppHeader.vue'
import GpuDock from '@/components/telemetry/GpuDock.vue'
import ResizeHandle from '@/components/ui/ResizeHandle.vue'

const route = useRoute()
const router = useRouter()
const { initTheme } = useTheme()
const chat = useChatStore()
const models = useModelsStore()
const tele = useTelemetryStore()
const registry = useRegistryStore()

// Resizable GPU dock (persisted).
const dockWidth = ref(Number(uiPreferences.getItem('pk_gpu_dock_width')) || 320)
watch(dockWidth, (w) => uiPreferences.setItem('pk_gpu_dock_width', String(w)))

onMounted(() => {
  initTheme()
  // The activity bar reports whether there are any chats, and the header shows
  // the loaded model + server version - both have to be known on every panel,
  // not just once ChatView has mounted. Both fetches are shared/idempotent, so
  // this doesn't double-fetch when ChatView also asks.
  void chat.hydrate()
  void models.refresh()
  // The CATALOG, app-wide. lib/model-name.ts falls back to it so attribution
  // outlives the run - "an artifact written yesterday still says who wrote it
  // after that model is stopped" - but the only components that ever loaded it
  // were Manager-side, so on the Studio that fallback had nothing to read.
  // A stopped model kept its name (which degrades to a cleaned-up id) and lost
  // its VENDOR MARK outright, which is why a running model showed a logo in
  // the chat and a stopped one showed none.
  void registry.refresh()
  void checkUi()
  uiTimer = window.setInterval(() => void checkUi(), 60_000)
  // A server started (or stopped) elsewhere - the Manager area, the CLI,
  // another tab - must surface here without a manual page refresh.
  // Same idiom as the fleet view: poll while mounted, plus a
  // revalidate when the tab regains visibility. Background refreshes are
  // silent (the store flips `loading` only on the first load).
  // Server-push first: while /api/events delivers, runner state arrives as
  // events and this poll relaxes to a 30s reconcile (cloud rows, belt).
  usePushStore().connect()
  let beat = 0
  runnersTimer = window.setInterval(() => {
    beat++
    if (!usePushStore().live || beat % 6 === 0) void models.refresh()
  }, 5_000)
  document.addEventListener('visibilitychange', onVis)
  window.addEventListener('keydown', onKey)
})
onUnmounted(() => {
  clearInterval(uiTimer)
  clearInterval(runnersTimer)
  document.removeEventListener('visibilitychange', onVis)
  window.removeEventListener('keydown', onKey)
})

// New chat, from anywhere in the STUDIO - the shell owns it rather than the
// chat view so it also works from prompts, connectors, cloud and settings,
// which is most of where you are when you think of something to ask. The
// Manager area is left alone: two areas, never mixed.
function onKey(e: KeyboardEvent): void {
  if (!isNewChat(e) || !route.path.startsWith('/studio')) return
  e.preventDefault()
  // Already on an uncommitted draft. Handing out another one would only throw
  // away whatever is typed in this one, so the key does nothing here.
  if (route.name === 'home' || route.name === 'chat-new') return
  void router.push({ name: 'chat-new' })
}

// The Studio cannot know it is stale on its own: swapping the exe and
// restarting the manager change nothing in an already-open tab (an SPA
// survives server restarts), and three review rounds were spent re-testing
// an old interface before anyone could tell. The served
// index's bundle hash vs the running one is ground truth for both sides.
const uiStale = ref(false)
let uiTimer: number | undefined
let runnersTimer: number | undefined
function runningBundle(): string | null {
  return (
    document
      .querySelector<HTMLScriptElement>('script[src*="/assets/index-"]')
      ?.src.match(/index-[\w-]+\.js/)?.[0] ?? null
  )
}
async function checkUi(): Promise<void> {
  if (uiStale.value) return
  try {
    const html = await (await fetch('/', { cache: 'no-store' })).text()
    const served = html.match(/index-[\w-]+\.js/)?.[0]
    const mine = runningBundle()
    if (served && mine && served !== mine) uiStale.value = true
  } catch {
    /* manager away for a moment - the next tick asks again */
  }
}
function onVis(): void {
  if (document.visibilityState === 'visible') {
    void checkUi()
    void models.refresh()
  }
}
function reloadNow(): void {
  window.location.reload()
}

// PICK the new BUILD up on the WAY PAST. A stale tab cannot be fixed by
// refetching - the old components are the ones in memory - but the LOAD can
// happen where a repaint was going to happen anyway. So the next time you move
// between panels we do a real navigation instead of an in-app one, and you
// simply arrive on the new build with the route you asked for.
//
// It waits when something in this tab would be lost (lib/reload.ts): an answer
// streaming in, text typed and unsent, files staged, a recording running. Then
// the banner stands and the choice is yours.
//
// Once per tab, whatever happens. If the served hash and the running one ever
// disagreed permanently - a proxy serving something we did not load - a swap
// on every navigation would be an unbreakable reload loop; one attempt makes
// that degrade into the banner instead.
// Say which of the two it is: about to fix itself, or waiting on you.
const staleNote = computed(() =>
  reloadHeld()
    ? 'Paddock was updated - this tab keeps the old interface until the work in it finishes.'
    : 'Paddock was updated - this tab picks up the new interface when you move to another panel.',
)

let swapped = false
router.beforeEach((to) => {
  if (!uiStale.value || swapped || reloadHeld()) return true
  swapped = true
  window.location.assign(router.resolve(to).href)
  return false
})

// The chat surface runs edge-to-edge on all three of its routes (a chat, the
// start page, and New chat's /chat/new); other panels get the padded,
// scrollable content area.
// full-bleed pages: the chat, and Reads - both a history list beside a
// workspace that scrolls on its own
const isChat = computed(() => ['chat', 'chat-new', 'home', 'reads', 'tables'].includes(String(route.name)))
</script>

<template>
  <div class="shell">
    <ActivityBar />
    <div class="shell__main">
      <AppHeader />
      <div v-if="uiStale" class="shell__update">
        <span>{{ staleNote }}</span>
        <button class="pk-btn pk-btn--sm pk-btn--primary" type="button" @click="reloadNow">
          Reload
        </button>
      </div>
      <main class="shell__content" :class="{ 'shell__content--flush': isChat }">
        <router-view />
      </main>
    </div>
    <template v-if="tele.open">
      <ResizeHandle v-model="dockWidth" side="right" :min="280" :max="560" />
      <GpuDock :style="{ width: `${dockWidth}px` }" />
    </template>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  height: 100%;
  width: 100%;
  background: var(--pk-bg-base);
}
.shell__main {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
}
.shell__content {
  flex: 1;
  overflow: auto;
  padding: 32px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  min-height: 0;
}
.shell__content--flush {
  padding: 0;
  overflow: hidden;
  align-items: stretch;
  justify-content: stretch;
}
.shell__update {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 8px 16px;
  background: var(--pk-bg-warning-subtle, var(--pk-bg-surface));
  border-bottom: 1px solid var(--pk-status-warning);
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-sm);
}
</style>
