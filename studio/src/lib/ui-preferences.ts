import { ref } from 'vue'

export const preferenceKeys = [
  'pk_theme', 'pk_max_tokens', 'pk_max_tokens_v2', 'pk_max_tool_calls', 'pk_summarize',
  'pk_auto_title', 'pk_mark_unsure', 'pk_dictate_with', 'pk_mic_device', 'pk_mic_device_label',
  'pk_map_tiles', 'pk_sidebar_width', 'pk_sidebar_open_width', 'pk_artifact_width_v2',
  'pk_graphpane_width_v1', 'pk_docpane_width_v2', 'pk_gpu_dock_width', 'pk_gpu_dock',
  'pk_model', 'pk_last_conversation', 'pk_chat_sort', 'lector.annotationPresets.user',
  'lector.recent-files',
] as const
const allowed = new Set<string>(preferenceKeys)
const prefix = 'studio.'

/** The history pages' side-panel fold, one for all three (usePanelFold), kept
 *  under each page's top-level settings key (they predate the `studio.` view)
 *  and read here with everything else before the app mounts, so a page opens
 *  with its panel as it was left - a page that fetched its own on mount
 *  showed the panel, then folded it, on every visit. */
export const panelKeys = ['readsPanelOpen', 'tablesPanelOpen', 'masksPanelOpen'] as const
export type PanelKey = (typeof panelKeys)[number]

export interface PreferenceAPI {
  load(): Promise<Record<string, unknown>>
  importMissing(patch: Record<string, unknown>): Promise<Record<string, unknown>>
  save(patch: Record<string, unknown>): Promise<void>
}
/** A preferences request the manager answered with an error status. */
export class PreferencesHttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'PreferencesHttpError'
  }
}
/** The manager wants the API key: a browser on another machine that has
 *  not unlocked yet. That is a login to show, not saved data that failed. */
export function isKeyRequired(e: unknown): boolean {
  return e instanceof PreferencesHttpError && e.status === 401
}
async function request(path: string, patch?: Record<string, unknown>) {
  const body = patch === undefined ? undefined : JSON.stringify(patch)
  const response = await fetch(path, patch === undefined ? { cache: 'no-store' } : {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body,
    // Keep small in-flight preference writes alive through navigation. The
    // browser's keepalive quota is 64 KiB; larger viewer lists use normal HTTP.
    keepalive: new TextEncoder().encode(body).length < 60 * 1024,
  })
  if (!response.ok) {
    throw new PreferencesHttpError(
      `Preferences could not be ${patch ? 'saved' : 'opened'} (HTTP ${response.status})`,
      response.status,
    )
  }
  return await response.json() as Record<string, unknown>
}
const api: PreferenceAPI = {
  load: () => request('/api/settings'),
  importMissing: patch => request('/api/settings/import', patch),
  save: async patch => { await request('/api/settings', patch) },
}

/** Synchronous in-memory view of SQLite preferences, loaded BEFORE any stores
 * mount. Only changed keys are sent; resizing is coalesced, writes serialized.
 * A failed write stays dirty and visible, never falls back to browser storage. */
export function createPreferences(backend: PreferenceAPI) {
  const values = new Map<string, string>()
  const panels = new Map<PanelKey, boolean>()
  const dirty = new Map<string, string | null>()
  const error = ref('')
  const pending = ref(false)
  let timer: ReturnType<typeof setTimeout> | undefined
  let flight: Promise<void> | undefined
  let loaded = false

  function hydrate(data: Record<string, unknown>) {
    values.clear()
    dirty.clear()
    pending.value = false
    for (const key of allowed) {
      const value = data[prefix + key]
      if (typeof value === 'string') values.set(key, value)
    }
    panels.clear()
    for (const key of panelKeys) {
      const value = data[key]
      if (typeof value === 'boolean') panels.set(key, value)
    }
    loaded = true
    error.value = ''
  }
  function setItem(key: string, value: string | null) {
    if (!allowed.has(key)) throw new Error('Unknown UI preference')
    if (value !== null && new TextEncoder().encode(value).length > 256 * 1024) {
      error.value = 'Preference is too large to save'
      throw new Error(error.value)
    }
    if ((values.get(key) ?? null) === value) return
    if (value === null) values.delete(key); else values.set(key, value)
    dirty.set(key, value)
    pending.value = true
    clearTimeout(timer)
    // Tests and isolated renderer diagnostics can construct stores without a
    // host. They must explicitly initialize before anything writes to a server.
    if (loaded) timer = setTimeout(() => { void flush().catch(() => {}) }, 200)
  }
  async function flush(): Promise<void> {
    clearTimeout(timer)
    if (flight) { await flight; return flush() }
    if (!dirty.size) return
    const snapshot = new Map(dirty)
    flight = (async () => {
      try {
        await backend.save(Object.fromEntries([...snapshot].map(([key, value]) => [prefix + key, value])))
        for (const [key, value] of snapshot) if (dirty.get(key) === value) dirty.delete(key)
        error.value = ''
      } catch (e) {
        error.value = e instanceof Error ? e.message : 'Preferences could not be saved'
        throw e
      } finally { pending.value = dirty.size > 0 }
    })()
    try { await flight } finally { flight = undefined }
    if (dirty.size) await flush()
  }
  return {
    error, pending,
    getItem: (key: string): string | null => values.get(key) ?? null,
    /** a page's panel fold as last saved, or null when it never was */
    panel: (key: PanelKey): boolean | null => panels.get(key) ?? null,
    /** remember the fold at once for every page (the next mount reads it)
     *  and save it, in one write */
    async setPanels(open: boolean): Promise<void> {
      for (const key of panelKeys) panels.set(key, open)
      if (loaded) await backend.save(Object.fromEntries(panelKeys.map((key) => [key, open])))
    },
    setItem,
    removeItem: (key: string) => setItem(key, null),
    flush,
    async initialize(migrate?: (data: Record<string, unknown>, backend: PreferenceAPI) => Promise<Record<string, unknown>>) {
      if (dirty.size && loaded) await flush()
      const data = await backend.load()
      hydrate(migrate ? await migrate(data, backend) : data)
    },
  }
}
export const uiPreferences = createPreferences(api)

let lifecycleInstalled = false
export function installPreferenceLifecycle() {
  if (lifecycleInstalled) return
  lifecycleInstalled = true
  window.addEventListener('online', () => { void uiPreferences.flush().catch(() => {}) })
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') void uiPreferences.flush().catch(() => {})
  })
  window.addEventListener('beforeunload', event => {
    if (uiPreferences.pending.value) { event.preventDefault(); event.returnValue = '' }
  })
  window.addEventListener('pagehide', () => { void uiPreferences.flush().catch(() => {}) })
}
