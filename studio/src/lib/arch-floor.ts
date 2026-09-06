// Whether an artifact's kernels exist on this GPU. One place, because two
// surfaces answer the question - the picker screens whole models before you
// choose one, the workload step badges the individual quality cards - and a
// disagreement between them shows up as a green "Should fit" on a build the
// engine then refuses to load.

/** Anything carrying an optional compute-capability floor (a CatalogArtifact,
 *  in practice). Structural deliberately: the two callers hold different slices
 *  of the artifact and neither needs the whole type to ask this. */
export interface ArchFloor {
  min_cc?: [number, number]
}

/** The generation NAME for a floor, because "Needs a Blackwell GPU" is what a
 *  person can act on and "needs compute 12.0" is not. Keyed by the floor
 *  rather than by every capability - this labels a REQUIREMENT - and the
 *  fallback prints the number rather than guessing a marketing name. */
const GEN_FOR_CC: Record<string, string> = { '12.0': 'a Blackwell GPU' }

/** True when this GPU is known to sit below the artifact's floor. No floor, or
 *  no `cc` (silicon we do not recognise), makes no claim: the engine already
 *  warns on an unvalidated arch, so a guess here would be noise. */
export function archBlocked(a: ArchFloor, cc: [number, number] | undefined): boolean {
  const need = a.min_cc
  if (!need || !cc) return false
  return cc[0] < need[0] || (cc[0] === need[0] && cc[1] < need[1])
}

/** Why this GPU cannot run it, in words a person can act on - or null when it
 *  can. */
export function archBlockReason(
  a: ArchFloor,
  cc: [number, number] | undefined,
): string | null {
  if (!archBlocked(a, cc)) return null
  const need = a.min_cc as [number, number]
  return `Needs ${GEN_FOR_CC[need.join('.')] ?? `compute ${need.join('.')}`}`
}

/** The cheapest floor among a blocked set - what someone would actually have
 *  to buy to unlock any build of the model, rather than whichever artifact
 *  happened to sort first. */
export function lowestFloor<T extends ArchFloor>(blocked: readonly T[]): T | undefined {
  return blocked.reduce<T | undefined>((best, a) => {
    if (!best) return a
    const x = a.min_cc ?? [0, 0]
    const y = best.min_cc ?? [0, 0]
    return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]) ? a : best
  }, undefined)
}
