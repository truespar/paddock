// The Masks page's history records (/api/mask-history, store/mask_history.rs
// on the manager): a picture, its prompts with what each found, and the
// camera snapshots kept while it was open. Pictures ride in `pictures` once
// each, keyed by the SHA-256 of their data URL, and are named by that key
// everywhere else in the record. Endpoint credentials are never kept here.
import type { MaskResponse, PromptBox, PromptPoint, VideoObject } from './masks'

export interface MaskLayerDoc {
  kind: 'concept' | 'object'
  text: string
  boxes: PromptBox[]
  points: PromptPoint[]
  objectBox: PromptBox | null
  threshold: number
  visible: boolean
  /** the candidate an object layer shows */
  choice: number
  /** instances switched off in the list */
  hidden: number[]
  result: MaskResponse | null
}

export interface MaskShotDoc {
  id: string
  /** its frame, in `pictures` */
  ref: string
  width: number
  height: number
  at: number
  /** the concepts tracked when it was taken; an object's `concept` counts these */
  concepts: { text: string; slot: number }[]
  objects: VideoObject[]
}

export interface MaskDoc {
  version: 1
  id: string
  title: string
  /** the masker of the latest change */
  model: string
  createdAt: number
  updatedAt: number
  pictures: Record<string, string>
  picture: { ref: string; name: string; width: number; height: number } | null
  layers: MaskLayerDoc[]
  /** the layer that was being edited */
  active: number
  /** oldest first */
  snapshots: MaskShotDoc[]
}

/** A side-panel row: `runs` counts the prompts that found something. */
export interface MaskSummary {
  id: string
  title: string
  model: string
  runs: number
  createdAt: number
  updatedAt: number
  revision: string
}

/** Snapshots a record keeps at most (the manager refuses more). */
export const MAX_SNAPSHOTS = 48

/** The key a picture is kept under: the SHA-256 of its data URL, as hex. */
export async function pictureKey(dataUrl: string): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Saving pictures needs localhost or HTTPS. The picture is still open; nothing was discarded.')
  }
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(dataUrl))
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}
