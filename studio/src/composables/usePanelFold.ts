// The history pages' side panel (Reads, Tables, Masks): ONE fold for all
// three, so moving between them never opens a panel the last page had folded,
// and known at once on every mount from the preferences the Studio loads
// before it starts - never fetched per page, which showed the panel and then
// folded it on each visit. The first fold any of them recorded wins (this
// page's own first); a narrow window starts folded.
import { effectScope, ref, watch, type Ref } from 'vue'
import { panelKeys, uiPreferences, type PanelKey } from '@/lib/ui-preferences'
import { useToastsStore } from '@/stores/toasts'

let shared: Ref<boolean> | null = null

export function usePanelFold(key: PanelKey): Ref<boolean> {
  if (shared) return shared
  const toasts = useToastsStore()
  const saved = [key, ...panelKeys.filter((k) => k !== key)].map((k) => uiPreferences.panel(k)).find((v) => v !== null)
  const open = ref(saved ?? window.innerWidth >= 1100)
  // outlives the page that made it: the next page gets the same fold
  effectScope(true).run(() => {
    let save: Promise<unknown> = Promise.resolve()
    watch(
      open,
      (v) => {
        save = save
          .catch(() => {})
          .then(() => uiPreferences.setPanels(v))
          .catch((e) => toasts.push({ tone: 'bad', title: 'Layout was not saved', description: String(e) }))
      },
      { flush: 'sync' },
    )
  })
  shared = open
  return open
}
