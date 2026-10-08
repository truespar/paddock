// What a weights choice means to the person picking it: the one-line blurb on
// a Quality card, and - for a model whose weights are not quality levels at
// all - the choice axes it asks instead (Kumo Tabular: Size x Task). Pure
// functions over catalog rows, kept out of the start form so it can be read.
import type { CatalogArtifact, ChoiceAxis } from '@/lib/api'

/** Plain-language meaning of a weights choice, keyed off the quant class -
 *  the tag itself ("Q8_0") stays a footnote for the people who know it. */
/** One line, because these sit side by side and are read by COMPARISON: three
 *  two-line paragraphs are read as prose, three short lines are read as a
 *  choice. Everything cut here was a qualification ("that most work never
 *  notices") that says the same thing on two of the three cards, so it
 *  distinguishes nothing - which is the only job a blurb on a card has. */
export function qualityBlurb(a: { quant?: string; source?: { repo: string; base_model: string } }): string {
  const q = (a.quant ?? '').toUpperCase()
  // the 8-bit class: Q8_0 and MLX's affine 8-bit (a scale and a bias per
  // group) - which fell through to "a smaller build" below, beside a Q8_0
  // card it is in fact a little larger than
  if (q.startsWith('Q8') || q.startsWith('MLX-AFFINE-8')) return 'Practically identical to the original.'
  // NVFP4 before the Q4 test: it is four-bit too, but read from a published
  // low-bit checkpoint rather than converted from a bigger file. WHOSE
  // checkpoint is not something the quant tag knows - this line said "the
  // official checkpoint" for every NVFP4 row, which became a false claim the
  // day a community export joined the catalog (qwen3.8-flash-next's NVFP4 is
  // a third party's conversion of Qwen's weights). `source` names the
  // producer, so ask it instead of assuming.
  if (q.includes('NVFP4')) {
    return thirdPartyExport(a)
      ? 'Four-bit, from a community build of the official weights.'
      : 'Four-bit, straight from the official checkpoint.'
  }
  if (q.includes('Q4')) return 'Half the memory, slight quality cost.'
  if (q.includes('MXFP4')) return 'The format it was trained in - nothing is higher.'
  // Prism ML's ternary packings: the model ships in nothing else
  if (q.startsWith('PTQ') || q.startsWith('PQ2')) return 'Ternary weights - the only build it ships in.'
  // a float build is the unquantized one - beside a Q8_0 it is the LARGER
  // card, and "a smaller build" (the fallback) said the opposite
  if (q === 'BF16' || q === 'F16' || q === 'F32') return 'Full precision - no quantization.'
  return 'A smaller build - some quality for memory.'
}
/** Somebody else's conversion, rather than a low-bit file the model's own
 *  authors published. A registry row only carries `source` when the producing
 *  repo differs from the obvious one, so no `source` reads as "official". */
function thirdPartyExport(a: { source?: { repo: string; base_model: string } }): boolean {
  const s = a.source
  if (!s?.repo || !s?.base_model) return false
  const org = (r: string) => r.split('/')[0].toLowerCase()
  return org(s.repo) !== org(s.base_model)
}

/** The axes to ask, or none: a model's declared `specs.choices`, when every
 *  weights artifact names a value on each of them. A half-annotated catalog
 *  row falls back to the Quality cards rather than offering a card that leads
 *  nowhere (the registry test pins the shipped rows complete). */
export function choiceAxesFor(axes: ChoiceAxis[] | undefined, weights: CatalogArtifact[]): ChoiceAxis[] {
  if (!axes?.length || weights.length < 2) return []
  return weights.every((w) => axes.every((x) => !!w.choice?.[x.name])) ? axes : []
}

/** The weights artifact at these coordinates on the axes, if there is one. */
export function artifactAt(
  axes: ChoiceAxis[],
  weights: CatalogArtifact[],
  at: Record<string, string>,
): CatalogArtifact | undefined {
  return weights.find((w) => axes.every((x) => w.choice?.[x.name] === at[x.name]))
}
