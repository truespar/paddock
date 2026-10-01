import type { PredictionResponse, TableSpec, Task as TabularTask } from './tables'

/** Shared with TableSession.swift. Never persist endpoint credentials. */
export interface TableInput {
  dataset: string
  fileName: string
  spec: TableSpec | null
  model: string
  port: number
  estimators: number
  seed: number
}
export interface TableRun {
  id: string
  at: number
  input: TableInput
  task: TabularTask
  ms: number
  response: PredictionResponse
}
export interface TableSession {
  version: 1
  id: string
  title: string
  model: string
  createdAt: number
  updatedAt: number
  datasets: Record<string, string>
  draft: TableInput
  runs: TableRun[]
}
export interface TableSummary {
  id: string
  title: string
  model: string
  runs: number
  createdAt: number
  updatedAt: number
  revision: string
}
export async function datasetKey(source: string): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('Saving tables requires localhost or HTTPS. Your table is still open; nothing was discarded.')
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source))
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}
