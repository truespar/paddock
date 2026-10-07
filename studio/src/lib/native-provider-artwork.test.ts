import { describe, expect, it } from 'vitest'
import vendorLogo from '../components/manage/VendorLogo.vue?raw'
import prismLogo from '../../../apps/macos/Sources/PaddockUI/Resources/PrismML.svg?raw'
import alephLogo from '../../../apps/macos/Sources/PaddockUI/Resources/AlephAlpha.svg?raw'
import nativeArtwork from '../../../apps/macos/Sources/PaddockUI/ProviderArtwork.swift?raw'

// Marks inlined in VendorLogo.vue that the macOS app ships as its own SVG
// file: the geometry must be the same paths in the same frame on both sides.
const SHARED = [
  { vendor: 'Prism ML', file: 'PrismML', svg: prismLogo, paths: 4, viewBox: '0 0 36.2855 29.7604' },
  { vendor: 'Aleph Alpha', file: 'AlephAlpha', svg: alephLogo, paths: 5, viewBox: '580 138 761 805' },
]

const paths = (svg: string) => [...svg.matchAll(/\sd="([^"]+)"/g)].map((match) => match[1])

for (const mark of SHARED) {
  describe(`${mark.vendor} provider artwork parity`, () => {
    it('shares the official emblem geometry between Studio and macOS', () => {
      const webMark = vendorLogo.match(
        new RegExp(`<svg\\s+v-else-if="vendor === '${mark.vendor}'"[\\s\\S]*?</svg>`),
      )?.[0]
      expect(webMark).toBeDefined()
      expect(paths(mark.svg)).toHaveLength(mark.paths)
      expect(paths(webMark!)).toEqual(paths(mark.svg))
      const viewBox = mark.svg.match(/viewBox="([^"]+)"/)?.[1]
      expect(viewBox).toBe(mark.viewBox)
      expect(webMark).toContain(`viewBox="${viewBox}"`)
      expect(webMark).toContain('fill="currentColor"')
      expect(webMark).toContain(`aria-label="${mark.vendor}"`)
    })

    it('bundles the native mark locally under the registry vendor name', () => {
      expect(nativeArtwork).toContain(`"${mark.vendor}": "${mark.file}"`)
      expect(mark.svg).not.toMatch(/<(?:script|image|foreignObject)\b|\bhref\s*=/i)
    })
  })
}
