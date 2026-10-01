import { beforeEach, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { tableHistoryApi } from '@/lib/api'
import { datasetKey, type TableInput, type TableRun, type TableSession } from '@/lib/table-history'
import { useTablesStore } from './tables'

vi.mock('@/lib/api', () => ({ tableHistoryApi: { list: vi.fn(), get: vi.fn(), save: vi.fn(), remove: vi.fn() } }))
beforeEach(() => { setActivePinia(createPinia()); vi.resetAllMocks() })
const source = 'size,city,label\n1,a,yes\n2,b,no\n3,c,\n'
const key = 'a47678d9564a8102e8467fd45a06b9e753a5450ed76f3af163a15eea89c9f3c0'
const input = (): TableInput => ({ dataset: key, fileName: 'Example.csv', model: 'kumo', port: 1234,
  estimators: 8, seed: 0, spec: { target: 2, use: [true, true, true], types: ['numerical', 'categorical', 'categorical'] } })
const run = (id: string): TableRun => ({ id, at: 10, input: input(), task: 'classification', ms: 3,
  response: { task: 'classification', classes: ['yes', 'no'], predictions: [{ class: 0, label: 'yes', probabilities: [0.8, 0.2] }] } })
const row = (doc: TableSession, revision: string) => ({ id: doc.id, title: doc.title, model: doc.model,
  createdAt: doc.createdAt, updatedAt: doc.updatedAt, runs: doc.runs.length, revision })

it('uses the native SHA-256 UTF-8 dataset identity', async () => { expect(await datasetKey(source)).toBe(key) })
it('serializes overlapping saves and deduplicates sources across immutable runs', async () => {
  const store = useTablesStore()
  vi.mocked(tableHistoryApi.save).mockImplementation(async (d, revision) => row(d, revision + '1'))
  const a = store.save(input(), source, run('a'))
  const b = store.save({ ...input(), seed: 1 }, source, { ...run('b'), input: { ...input(), seed: 1 } })
  await Promise.all([a, b])
  expect(tableHistoryApi.save).toHaveBeenNthCalledWith(2, expect.anything(), '1')
  expect(store.document?.runs.map((r) => r.id)).toEqual(['a', 'b'])
  expect(store.document?.runs[0]?.input.seed).toBe(0)
  expect(Object.keys(store.document!.datasets)).toEqual([key])
  expect(store.sessions[0]?.runs).toBe(2)
  expect(store.saving).toBe(false)
})
it('retries the exact document after a lost acknowledgement before adding later edits', async () => {
  const store = useTablesStore()
  vi.mocked(tableHistoryApi.save).mockRejectedValueOnce(Error('connection lost'))
    .mockImplementation(async (d, rev) => row(d, rev + '1'))
  expect(await store.save(input(), source, run('a'))).toBe(false)
  const sent = JSON.stringify(vi.mocked(tableHistoryApi.save).mock.calls[0]![0])
  expect(store.document?.runs).toHaveLength(1)
  expect(await store.save(input(), source)).toBe(true)
  expect(JSON.stringify(vi.mocked(tableHistoryApi.save).mock.calls[1]![0])).toBe(sent)
  expect(store.document?.runs).toHaveLength(1)
  expect(store.error).toBeNull()
})
it('keeps conflicting results and uses the reviewed revision for deletion', async () => {
  const store = useTablesStore()
  vi.mocked(tableHistoryApi.save).mockImplementation(async (d) => row(d, 'r1'))
  await store.save(input(), source, run('a'))
  const id = store.document!.id
  vi.mocked(tableHistoryApi.save).mockRejectedValue(Error('changed elsewhere'))
  expect(await store.save(input(), source, run('b'))).toBe(false)
  expect(store.document?.runs).toHaveLength(2)
  expect(store.sessions[0]?.runs).toBe(1)
  vi.mocked(tableHistoryApi.remove).mockRejectedValueOnce(Error('conflict'))
  expect(await store.remove(id)).toBe(false)
  expect(store.document?.id).toBe(id)
  expect(tableHistoryApi.remove).toHaveBeenCalledWith(id, 'r1')
})
it('opens complete sessions while the model is stopped and renames without losing runs', async () => {
  const store = useTablesStore()
  vi.mocked(tableHistoryApi.save).mockImplementation(async (d) => row(d, 'r1'))
  await store.save(input(), source, run('a'))
  const doc = JSON.parse(JSON.stringify(store.document)) as TableSession
  store.reset()
  vi.mocked(tableHistoryApi.get).mockResolvedValue({ doc, revision: 'r1' })
  await store.open(doc.id)
  expect(store.document?.datasets[key]).toBe(source)
  await store.rename(doc.id, 'Renamed')
  expect(store.document?.title).toBe('Renamed')
  expect(store.document?.runs).toHaveLength(1)
})
