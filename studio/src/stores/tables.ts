import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import { tableHistoryApi } from '@/lib/api'
import type { TableInput, TableRun, TableSession, TableSummary } from '@/lib/table-history'
import { uuid } from '@/lib/uuid'

export const useTablesStore = defineStore('tables', () => {
  const sessions = ref<TableSummary[]>([])
  // Prediction arrays can contain a million quantiles. They are immutable
  // snapshots, not fields to proxy/deep-copy on every draft save.
  const document = shallowRef<TableSession | null>(null)
  const revision = ref('')
  const loaded = ref(false)
  const error = ref<string | null>(null)
  const saving = ref(false)
  // The endpoint the page sends to. The header's picker chooses it, as it
  // chooses the reader on Reads, so it lives here rather than in the page.
  const port = ref(0)
  let generation = 0
  let writeGeneration = 0
  let tail: Promise<boolean> = Promise.resolve(true)
  let pending: { doc: TableSession; expected: string } | null = null

  async function refresh(): Promise<void> {
    const attempt = generation
    try {
      const rows = await tableHistoryApi.list()
      if (attempt === generation) { sessions.value = rows; if (!pending) error.value = null }
    } catch (e) { error.value = String(e) }
    finally { loaded.value = true }
  }
  async function persist(doc: TableSession, expected: string): Promise<boolean> {
    pending = { doc, expected }
    try {
      const row = await tableHistoryApi.save(doc, expected)
      revision.value = row.revision
      pending = null
      ++generation
      sessions.value = [row, ...sessions.value.filter((r) => r.id !== row.id)].sort((a, b) => b.updatedAt - a.updatedAt)
      error.value = null
      return true
    } catch (e) { error.value = e instanceof Error ? e.message : String(e); return false }
  }
  function save(input: TableInput, source: string, run?: TableRun): Promise<boolean> {
    const attempt = ++writeGeneration
    saving.value = true
    tail = tail.then(async () => {
      const acknowledged = !pending || await persist(pending.doc, pending.expected)
      const now = Date.now()
      const previous = document.value
      const doc: TableSession = previous ? { ...previous, datasets: { ...previous.datasets }, runs: [...previous.runs] } : {
        version: 1, id: uuid(), title: input.fileName || 'Untitled table', model: input.model,
        createdAt: now, updatedAt: now, datasets: {}, draft: input, runs: [],
      }
      doc.draft = input
      doc.model = input.model
      doc.updatedAt = Math.max(now, doc.updatedAt + 1)
      doc.datasets[input.dataset] = source
      if (run && !doc.runs.some((r) => r.id === run.id)) doc.runs.push(run)
      const keep = new Set([input.dataset, ...doc.runs.map((r) => r.input.dataset)])
      doc.datasets = Object.fromEntries(Object.entries(doc.datasets).filter(([key]) => keep.has(key)))
      document.value = doc // preserve unsaved runs on network/conflict failure
      const ok = acknowledged && await persist(doc, revision.value)
      if (attempt === writeGeneration) saving.value = false
      return ok
    })
    return tail
  }
  async function open(id: string): Promise<TableSession | null> {
    await tail
    try {
      const saved = await tableHistoryApi.get(id)
      document.value = saved.doc
      revision.value = saved.revision
      ++generation
      error.value = null
      return saved.doc
    } catch (e) { error.value = String(e); return null }
  }
  function reset(): void { document.value = null; revision.value = ''; pending = null; error.value = null; ++generation }
  function hasPendingSave(): boolean { return pending !== null }
  function fork(): void {
    if (saving.value || !document.value) return
    document.value = { ...document.value, id: uuid(), title: document.value.title.slice(0, 200) + ' copy', createdAt: Date.now(), updatedAt: Date.now() }
    revision.value = ''; pending = null; error.value = null
  }
  async function rename(id: string, title: string): Promise<void> {
    if (!title.trim() || saving.value || error.value) return
    try {
      const { doc, revision: expected } = await tableHistoryApi.get(id)
      doc.title = title.trim()
      const row = await tableHistoryApi.save(doc, expected)
      if (document.value?.id === id) { document.value = { ...document.value, title: doc.title }; revision.value = row.revision }
      sessions.value = sessions.value.map((r) => r.id === id ? row : r)
      ++generation
      error.value = null
    } catch (e) { error.value = String(e) }
  }
  async function remove(id: string): Promise<boolean> {
    const row = sessions.value.find((r) => r.id === id)
    if (!row || saving.value) return false
    try {
      await tableHistoryApi.remove(id, row.revision)
      sessions.value = sessions.value.filter((r) => r.id !== id)
      if (document.value?.id === id) reset()
      ++generation
      return true
    } catch (e) { error.value = String(e); return false }
  }
  return { sessions, document, loaded, error, saving, port, refresh, save, open, reset, fork, rename, remove, hasPendingSave }
})
