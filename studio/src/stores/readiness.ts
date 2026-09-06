import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

// Can this computer run models on its own, and what can the person do about it.
//
// The manager probes once at startup and publishes facts, not sentences - the
// words live here, where they can be written and rewritten without a rebuild
// and where the rule that the Manager speaks no jargon is enforceable by
// reading it.
//
// There was a 'needs-setup' state here, and a SetupProgress poller that
// followed a one-time download of NVIDIA's maths libraries. Paddock ships and
// fetches none: the exes import no NVIDIA DLL, the kernel pack
// imports KERNEL32 only, and the binaries do not contain "cublas64"/"cudart64"
// as strings at all, so they could not ask for one. The only NVIDIA binary the
// engine loads is the display driver's own nvcuda - which is what the
// `driver-too-old` and `no-card` states already cover.

export type ReadyState = 'ready' | 'untested' | 'driver-too-old' | 'no-card'

export interface SupportedGen {
  name: string
  cards: string[]
}

export interface Readiness {
  state: ReadyState
  card?: string
  generation?: string
  /** compute capability [major, minor], for artifacts that carry a floor
   *  (`min_cc`). Absent on silicon we do not recognise - read that as "no
   *  claim", never as "too old". */
  cc?: [number, number]
  driver?: string
  cuda?: string
  cuda_needed: string
  os: 'windows' | 'linux' | 'macos'
  supported: SupportedGen[]
}

export const useReadinessStore = defineStore('readiness', () => {
  const info = ref<Readiness | null>(null)
  let inflight: Promise<void> | undefined

  /** Whether this computer cannot run models at all. An `untested` card
   *  still can - the engine serves an unvalidated card under a warning - so
   *  only a missing card, or a driver too old to load the kernels, blocks. */
  const blocked = computed(
    () => info.value?.state === 'no-card' || info.value?.state === 'driver-too-old',
  )
  /** Whether there is any NVIDIA card to sample metrics from. True until the
   *  probe answers: a machine that has one must not watch its GPU button
   *  appear a moment after the page does. */
  const hasCard = computed(() => !info.value || info.value.state !== 'no-card')
  /** Nothing to say when everything is fine; silence is the goal state. */
  const notice = computed(() => (info.value && info.value.state !== 'ready' ? info.value : null))

  /** The probe's answer, fetched at most once however many callers ask.
   *
   *  Several components call `load()` on mount, which is fine - it is cheap
   *  and idempotent. The route guard is different: it must not let a page
   *  render before the verdict is in, and it must not fire a second request
   *  per navigation. Both callers share this one in-flight promise. */
  function ensureLoaded(): Promise<void> {
    inflight ??= load().finally(() => {
      // Only a FAILED probe is worth retrying; a verdict is a verdict.
      if (!info.value) inflight = undefined
    })
    return inflight
  }

  async function load() {
    try {
      const r = await fetch('/api/readiness')
      if (r.ok) info.value = (await r.json()) as Readiness
    } catch {
      // A manager we cannot reach is the shell's problem to report, not ours;
      // guessing a verdict here would be worse than saying nothing.
    }
  }

  return { info, blocked, hasCard, notice, load, ensureLoaded }
})
