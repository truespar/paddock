import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import { maskHistoryApi } from '@/lib/api'
import type { MaskDoc, MaskSummary } from '@/lib/mask-history'

/** The Masks page: its endpoint and its history.
 *
 *  `port` is the running masker the page sends to, chosen in the header's
 *  model picker while the page is open - the one place a model is picked, as
 *  on Reads and Tables. 0 until a masker is running.
 *
 *  The history is the pictures worked on (/api/mask-history), listed in the
 *  side panel the way Tables lists tables: summaries in the list, a record
 *  fetched whole to open it and saved whole, every write naming the revision
 *  it was made on. Writes queue one behind the other; a write that failed is
 *  sent again before the next one, so a lost reply never forks a record. */
export const useMasksStore = defineStore('masks', () => {
  const port = ref(0)
  const sessions = ref<MaskSummary[]>([])
  // a record holds its pictures as data URLs - an immutable snapshot, never
  // proxied or deep-copied
  const document = shallowRef<MaskDoc | null>(null)
  const revision = ref('')
  const loaded = ref(false)
  const error = ref<string | null>(null)
  const saving = ref(false)
  let generation = 0
  let writeGeneration = 0
  let tail: Promise<boolean> = Promise.resolve(true)
  let pending: { doc: MaskDoc; expected: string } | null = null

  async function refresh(): Promise<void> {
    const attempt = generation
    try {
      const rows = await maskHistoryApi.list()
      if (attempt === generation) {
        sessions.value = rows
        if (!pending) error.value = null
      }
    } catch (e) {
      error.value = String(e)
    } finally {
      loaded.value = true
    }
  }
  async function persist(doc: MaskDoc, expected: string): Promise<boolean> {
    pending = { doc, expected }
    try {
      const row = await maskHistoryApi.save(doc, expected)
      revision.value = row.revision
      pending = null
      ++generation
      sessions.value = [row, ...sessions.value.filter((r) => r.id !== row.id)].sort((a, b) => b.updatedAt - a.updatedAt)
      error.value = null
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return false
    }
  }
  /** Save the page as `build` makes it from the record it replaces (null for
   *  a new one). The build runs in turn, so two quick saves of a new picture
   *  make one record, not two. */
  function save(build: (previous: MaskDoc | null) => MaskDoc): Promise<boolean> {
    const attempt = ++writeGeneration
    saving.value = true
    tail = tail.then(async () => {
      const acknowledged = !pending || (await persist(pending.doc, pending.expected))
      const doc = build(document.value)
      // kept even if the write fails: the page still shows it
      document.value = doc
      const ok = acknowledged && (await persist(doc, revision.value))
      if (attempt === writeGeneration) saving.value = false
      return ok
    })
    return tail
  }
  async function open(id: string): Promise<MaskDoc | null> {
    await tail
    try {
      const saved = await maskHistoryApi.get(id)
      document.value = saved.doc
      revision.value = saved.revision
      ++generation
      error.value = null
      return saved.doc
    } catch (e) {
      error.value = String(e)
      return null
    }
  }
  function reset(): void {
    document.value = null
    revision.value = ''
    pending = null
    error.value = null
    ++generation
  }
  function hasPendingSave(): boolean {
    return pending !== null
  }
  async function rename(id: string, title: string): Promise<void> {
    if (!title.trim() || saving.value || error.value) return
    try {
      const { doc, revision: expected } = await maskHistoryApi.get(id)
      doc.title = title.trim()
      const row = await maskHistoryApi.save(doc, expected)
      if (document.value?.id === id) {
        document.value = { ...document.value, title: doc.title }
        revision.value = row.revision
      }
      sessions.value = sessions.value.map((r) => (r.id === id ? row : r))
      ++generation
      error.value = null
    } catch (e) {
      error.value = String(e)
    }
  }
  async function remove(id: string): Promise<boolean> {
    const row = sessions.value.find((r) => r.id === id)
    if (!row || saving.value) return false
    try {
      await maskHistoryApi.remove(id, row.revision)
      sessions.value = sessions.value.filter((r) => r.id !== id)
      if (document.value?.id === id) reset()
      ++generation
      return true
    } catch (e) {
      error.value = String(e)
      return false
    }
  }
  return { port, sessions, document, loaded, error, saving, refresh, save, open, reset, rename, remove, hasPendingSave }
})
