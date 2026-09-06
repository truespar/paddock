<script setup lang="ts">
// The left rail shows one area's items - Manager (servers, instrument) or
// Studio (chat, prompts, tools; settings at the bottom). The header owns
// switching between areas; the rail never mixes them - that is the
// manager/studio split. There is no Models item: the catalog lives
// inside the deploy flow, and downloads happen there too.
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'
import { useReadinessStore } from '@/stores/readiness'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const route = useRoute()
const router = useRouter()
const chat = useChatStore()
const readiness = useReadinessStore()

const area = computed<'manage' | 'studio'>(() =>
  route.path.startsWith('/studio') ? 'studio' : 'manage',
)

// Chat = your conversations. With none, there's nothing behind it - show it
// as empty rather than let it look clickable and do nothing.
const noChats = computed(() => chat.loaded && chat.conversations.length === 0)

// Embeddings/rerank is a page for a running ENCODER, and an encoder only ever
// runs on this box's GPU - there is no cloud embeddings lane. A machine that
// cannot serve therefore has nothing behind that button, ever, so it is not
// offered. `blocked` is false while the probe is out and
// false when the answer is fine, so the item never appears LATE on a good box;
// it disappears on a bad one, which is the right way round. An untested card
// is not a bad one - it serves, under a warning.
const canRunHere = computed(() => !readiness.blocked)

function go(name: string): void {
  // instrument's route requires a :tab param - a bare { name } push aborts
  // silently, which read as "the button does nothing". Landing is USAGE:
  // the dashboard overview first, drill-downs (activity/GPU/logs) behind it.
  if (name === 'instrument') {
    void router.push({ name, params: { tab: 'usage' } })
    return
  }
  void router.push({ name })
}
function active(name: string): boolean {
  if (name === 'chat') return route.name === 'chat' || route.name === 'chat-new'
  return route.name === name
}
</script>

<template>
  <nav class="activity-bar">
    <!-- Brand mark (Truespar) - the area's home -->
    <Tooltip label="Paddock" side="right">
      <button
        class="activity-bar__logo"
        type="button"
        aria-label="Paddock home"
        @click="go(area === 'studio' ? 'home' : 'servers')"
      >
        <div class="activity-bar__logo-icon">
          <img src="/img/truespar-mark-3d.svg" alt="" aria-hidden="true" />
        </div>
      </button>
    </Tooltip>

    <!-- Manager: the control plane -->
    <div v-if="area === 'manage'" class="activity-bar__icons">
      <Tooltip label="Models" side="right">
        <button
          class="activity-bar__btn"
          :class="{ 'activity-bar__btn--active': active('servers') }"
          @click="go('servers')"
        >
          <Icon name="server" :size="20" />
        </button>
      </Tooltip>
      <Tooltip label="Instrument (usage · requests · GPU · logs)" side="right">
        <button
          class="activity-bar__btn"
          :class="{ 'activity-bar__btn--active': active('instrument') }"
          @click="go('instrument')"
        >
          <Icon name="activity" :size="20" />
        </button>
      </Tooltip>
    </div>

    <!-- Studio: the client of those servers -->
    <div v-else class="activity-bar__icons">
      <Tooltip :label="noChats ? 'No chats yet' : 'Chat'" side="right">
        <button
          class="activity-bar__btn"
          :class="{
            'activity-bar__btn--active': active('chat'),
            'activity-bar__btn--empty': noChats,
          }"
          :aria-disabled="noChats || undefined"
          @click="go(noChats ? 'home' : 'chat')"
        >
          <Icon name="message-square" :size="20" />
        </button>
      </Tooltip>
      <!-- the Studio's non-chat surface: try embeddings/rerank hands-on -->
      <Tooltip v-if="canRunHere" label="Embeddings · rerank" side="right">
        <button
          class="activity-bar__btn"
          :class="{ 'activity-bar__btn--active': active('embeddings') }"
          @click="go('embeddings')"
        >
          <Icon name="sliders" :size="20" />
        </button>
      </Tooltip>
      <Tooltip label="Prompts" side="right">
        <button
          class="activity-bar__btn"
          :class="{ 'activity-bar__btn--active': active('prompts') }"
          @click="go('prompts')"
        >
          <Icon name="file-text" :size="20" />
        </button>
      </Tooltip>
      <Tooltip label="Cloud models" side="right">
        <button
          class="activity-bar__btn"
          :class="{ 'activity-bar__btn--active': active('cloud') }"
          @click="go('cloud')"
        >
          <Icon name="cloud" :size="20" />
        </button>
      </Tooltip>
      <Tooltip label="Connectors (MCP)" side="right">
        <button
          class="activity-bar__btn"
          :class="{ 'activity-bar__btn--active': active('connectors') }"
          @click="go('connectors')"
        >
          <Icon name="plug" :size="20" />
        </button>
      </Tooltip>
    </div>

    <!-- each area has its own settings at the bottom: Studio = chat
         preferences, Manager = box admin (export, future bind/auth) -->
    <div class="activity-bar__bottom">
      <Tooltip label="Settings" side="right">
        <button
          class="activity-bar__btn"
          :class="{
            'activity-bar__btn--active': active(area === 'studio' ? 'settings' : 'manage-settings'),
          }"
          @click="go(area === 'studio' ? 'settings' : 'manage-settings')"
        >
          <Icon name="settings" :size="20" />
        </button>
      </Tooltip>
    </div>
  </nav>
</template>

<style scoped>
.activity-bar {
  display: flex;
  flex-direction: column;
  width: var(--pk-activitybar-width);
  height: 100%;
  background: var(--pk-bg-surface);
  border-right: 1px solid var(--pk-border-default);
  flex-shrink: 0;
  /* Traverse: the fixed width is content-box, so the 1px right border sits
     outside it (total = width + 1px). Matching this keeps the bar's inner
     column identical to Traverse's. */
  box-sizing: content-box;
}
.activity-bar__logo {
  display: flex;
  align-items: center;
  justify-content: center;
  height: var(--pk-header-height);
  border: none;
  border-bottom: 1px solid var(--pk-border-default);
  background: var(--pk-bg-base);
  flex-shrink: 0;
  cursor: pointer;
  transition: background 0.15s;
}
.activity-bar__logo:hover {
  background: color-mix(in srgb, var(--pk-bg-base) 92%, var(--pk-text-primary));
}
.activity-bar__logo-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
}
.activity-bar__logo-icon img {
  display: block;
  width: 34px;
  height: 34px;
}
.activity-bar__icons {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 8px 0;
}
.activity-bar__btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  box-sizing: border-box;
  border: 1px solid transparent;
  border-radius: var(--pk-radius-md);
  background: transparent;
  color: var(--pk-text-muted);
  cursor: pointer;
  transition: color 0.15s, background 0.15s, border-color 0.15s;
}
.activity-bar__btn:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
.activity-bar__btn--active,
.activity-bar__btn--active:hover {
  color: var(--pk-accent);
  background: var(--pk-accent-subtle);
  border-color: var(--pk-accent);
}
/* nothing behind it yet - reads as inert, but still hoverable for the tooltip */
.activity-bar__btn--empty,
.activity-bar__btn--empty:hover {
  opacity: 0.4;
  background: transparent;
  color: var(--pk-text-muted);
  cursor: default;
}
.activity-bar__bottom {
  margin-top: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-bottom: 8px;
}
</style>
