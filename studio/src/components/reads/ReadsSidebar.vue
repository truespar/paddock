<script setup lang="ts">
// The Reads page's history, beside the workspace the way the chat list sits
// beside a conversation: New read at the top, the earlier reads under it,
// newest run first. A row is one read - a text and its questions - however
// many times it was run; the runs of the open read are walked in the answers
// card, not here. The list is summaries from the manager, so a read made in
// one browser is here in the next.
import { computed, nextTick, ref } from 'vue'
import type { ReadSummary } from '@/lib/reads'
import Icon from '@/components/Icon.vue'
import Dialog from '@/components/ui/Dialog.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuSeparator from '@/components/ui/MenuSeparator.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const props = withDefaults(defineProps<{
  noun?: 'read' | 'table'
  reads: ReadSummary[]
  activeId: string | null
  loaded: boolean
  error: string | null
  /** a run is in flight: switching reads under it would strand its answer */
  busy: boolean
}>(), { noun: 'read' })
const emit = defineEmits<{
  new: []
  open: [id: string]
  rename: [id: string, title: string]
  remove: [id: string]
  fold: []
}>()

const query = ref('')
const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  return q ? props.reads.filter((r) => r.title.toLowerCase().includes(q)) : props.reads
})

// ── rename, inline, the way the chat list does it ───────────────────────────
const menuOpenId = ref<string | null>(null)
const renamingId = ref<string | null>(null)
const renameText = ref('')
const renameInput = ref<HTMLInputElement[] | null>(null)
function startRename(r: ReadSummary): void {
  menuOpenId.value = null
  renamingId.value = r.id
  renameText.value = r.title
  void nextTick(() => {
    const el = renameInput.value?.[0]
    el?.focus()
    el?.select()
  })
}
function commitRename(): void {
  const id = renamingId.value
  if (!id) return
  renamingId.value = null
  const title = renameText.value.trim()
  const was = props.reads.find((r) => r.id === id)?.title
  if (title && title !== was) emit('rename', id, title)
}

const pendingDelete = ref<ReadSummary | null>(null)
function confirmDelete(): void {
  const r = pendingDelete.value
  pendingDelete.value = null
  if (r) emit('remove', r.id)
}

function onRow(r: ReadSummary): void {
  if (renamingId.value === r.id || props.busy) return
  emit('open', r.id)
}

// ── relative time, as the chat list shows it ────────────────────────────────
const MIN = 60_000
const HOUR = 60 * MIN
const DAY = 24 * HOUR
function when(ts: number): string {
  const now = Date.now()
  const diff = now - ts
  if (diff < MIN) return 'now'
  if (diff < HOUR) return `${Math.floor(diff / MIN)}m`
  if (diff < DAY) return `${Math.floor(diff / HOUR)}h`
  if (diff < 7 * DAY) return `${Math.floor(diff / DAY)}d`
  const d = new Date(ts)
  const sameYear = d.getFullYear() === new Date(now).getFullYear()
  return d.toLocaleDateString(
    undefined,
    sameYear ? { month: 'short', day: 'numeric' } : { month: 'short', day: 'numeric', year: 'numeric' },
  )
}
function detail(r: ReadSummary): string {
  const runs = `${r.runs} run${r.runs === 1 ? '' : 's'}`
  return `${r.title} · ${runs} · ${r.model || 'no model'} · ${new Date(r.updatedAt).toLocaleString()}`
}
</script>

<template>
  <aside class="rsb">
    <div class="rsb__top">
      <Tooltip :label="`Hide ${noun}s`">
        <button class="pk-icon-btn rsb__fold" type="button" :aria-label="`Hide ${noun}s`" @click="emit('fold')">
          <Icon name="chevron-left" :size="16" />
        </button>
      </Tooltip>
      <button class="pk-btn pk-btn--primary rsb__new" type="button" :disabled="busy" @click="emit('new')">
        <Icon name="plus" :size="16" />
        <span class="rsb__new-label">New {{ noun }}</span>
      </button>
    </div>

    <div class="rsb__search">
      <Icon name="search" :size="14" class="rsb__search-icon" />
      <input v-model="query" class="pk-input rsb__search-input" :placeholder="`Search ${noun}s`" />
    </div>

    <p v-if="error" class="rsb__empty" role="alert">{{ error }}</p>
    <div class="rsb__list">
      <div v-if="loaded && shown.length === 0" class="rsb__empty">
        {{ query.trim() ? 'No matches' : `No ${noun}s yet` }}
      </div>
      <div
        v-for="r in shown"
        :key="r.id"
        class="rrow"
        :class="{ 'rrow--active': r.id === activeId, 'rrow--locked': busy && r.id !== activeId }"
        @click="onRow(r)"
      >
        <input
          v-if="renamingId === r.id"
          ref="renameInput"
          v-model="renameText"
          class="pk-input pk-input--sm rrow__rename"
          @keydown.enter.stop="commitRename"
          @keydown.esc="renamingId = null"
          @blur="commitRename"
          @click.stop
        />
        <template v-else>
          <!-- the row is clickable anywhere; this is its keyboard handle -->
          <button
            class="rrow__open"
            type="button"
            :aria-current="r.id === activeId ? 'true' : undefined"
            @click.stop="onRow(r)"
          >
            <Icon :name="noun === 'table' ? 'table' : 'list-checks'" :size="13" class="rrow__kind" />
            <Tooltip :label="detail(r)"><span class="rrow__title">{{ r.title }}</span></Tooltip>
          </button>
          <span class="rrow__right" @click.stop>
            <span class="rrow__when">{{ when(r.updatedAt) }}</span>
            <span class="rrow__actions">
              <Menu :open="menuOpenId === r.id" @update:open="(v: boolean) => (menuOpenId = v ? r.id : null)">
                <MenuTrigger>
                  <button class="pk-icon-btn rrow__act" type="button" :aria-label="`${noun} actions`">
                    <Icon name="more-horizontal" :size="15" />
                  </button>
                </MenuTrigger>
                <MenuContent :label="`Actions for ${r.title}`">
                  <MenuItem @select="startRename(r)"><Icon name="edit" :size="14" /> Rename</MenuItem>
                  <MenuSeparator />
                  <MenuItem danger :disabled="busy && r.id === activeId" @select="pendingDelete = r">
                    <Icon name="trash" :size="14" /> Delete
                  </MenuItem>
                </MenuContent>
              </Menu>
            </span>
          </span>
        </template>
      </div>
    </div>
  </aside>

  <Dialog
    :open="!!pendingDelete"
    role="alertdialog"
    danger
    icon="alert-triangle"
    :title="`Delete ${noun}?`"
    size="sm"
    @close="pendingDelete = null"
  >
    <p class="rsb__confirm">
      <strong>{{ pendingDelete?.title }}</strong> and its runs will be permanently removed. This can't
      be undone.
    </p>
    <template #footer>
      <button class="pk-btn pk-btn--ghost" type="button" @click="pendingDelete = null">Cancel</button>
      <button class="pk-btn pk-btn--danger" type="button" @click="confirmDelete">Delete</button>
    </template>
  </Dialog>
</template>

<style scoped>
.rsb {
  width: 260px;
  flex: none;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  background: var(--pk-bg-surface);
  border-right: 1px solid var(--pk-border-default);
}
.rsb__top {
  display: flex;
  align-items: center;
  gap: 6px;
}
.rsb__fold {
  flex: none;
  width: 34px;
  height: 34px;
}
.rsb__new {
  flex: 1;
  min-width: 0;
  justify-content: flex-start;
}
.rsb__new-label {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.rsb__search {
  position: relative;
}
.rsb__search-icon {
  position: absolute;
  left: 9px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--pk-text-muted);
  pointer-events: none;
}
.rsb__search-input {
  width: 100%;
  padding-left: 30px;
}
.rsb__list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin: 0 -12px;
  padding: 0 12px;
}
.rsb__empty {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
  text-align: center;
  padding: 20px 0;
}
.rsb__confirm {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.rrow {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 9px;
  border-radius: var(--pk-radius-md);
  cursor: pointer;
  color: var(--pk-text-secondary);
  transition:
    background 0.12s ease,
    color 0.12s ease;
}
.rrow:hover {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.rrow:has(.rrow__open:focus-visible) {
  outline: 2px solid var(--pk-accent);
  outline-offset: -2px;
}
.rrow__open {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0;
  border: 0;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: inherit;
}
.rrow__open:focus-visible {
  outline: none;
}
.rrow--active {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
/* while a run is in flight the other reads wait - opening one would leave
   the answer landing on a read no longer on screen */
.rrow--locked {
  cursor: default;
  opacity: 0.6;
}
.rrow__kind {
  flex: none;
  color: var(--pk-text-muted);
}
.rrow--active .rrow__kind {
  color: inherit;
}
.rrow__title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--pk-font-size-sm);
}
.rrow__rename {
  flex: 1;
  min-width: 0;
}
.rrow__right {
  position: relative;
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  min-width: 34px;
  height: 24px;
}
.rrow__when {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  white-space: nowrap;
}
.rrow__actions {
  position: absolute;
  right: 0;
  display: flex;
  opacity: 0;
}
.rrow:hover .rrow__when,
.rrow:focus-within .rrow__when {
  opacity: 0;
}
.rrow:hover .rrow__actions,
.rrow:focus-within .rrow__actions,
.rrow__actions:has([data-state='open']) {
  opacity: 1;
}
.rrow:has(.rrow__actions [data-state='open']) .rrow__when {
  opacity: 0;
}
.rrow__act {
  width: 24px;
  height: 24px;
}
</style>
