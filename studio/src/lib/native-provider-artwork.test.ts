import { describe, expect, it } from 'vitest'
import vendorLogo from '../components/manage/VendorLogo.vue?raw'
import prismLogo from '../../../apps/macos/Sources/PaddockUI/Resources/PrismML.svg?raw'
import alephLogo from '../../../apps/macos/Sources/PaddockUI/Resources/AlephAlpha.svg?raw'
import openbmbLogo from '../../../apps/macos/Sources/PaddockUI/Resources/OpenBMB.svg?raw'
import ticLogo from '../../../apps/macos/Sources/PaddockUI/Resources/IntelligenceCompany.svg?raw'
import nativeArtwork from '../../../apps/macos/Sources/PaddockUI/ProviderArtwork.swift?raw'
import kblabWeb from '../assets/kblab.svg?raw'
import kblabNative from '../../../apps/macos/Sources/PaddockUI/Resources/KBLab.svg?raw'
import lightonWeb from '../assets/lighton.svg?raw'
import lightonNative from '../../../apps/macos/Sources/PaddockUI/Resources/LightOn.svg?raw'

// Marks inlined in VendorLogo.vue that the macOS app ships as its own SVG
// file: the geometry must be the same paths in the same frame on both sides.
const SHARED = [
  { vendor: 'Prism ML', file: 'PrismML', svg: prismLogo, paths: 4, viewBox: '0 0 36.2855 29.7604' },
  { vendor: 'Aleph Alpha', file: 'AlephAlpha', svg: alephLogo, paths: 5, viewBox: '580 138 761 805' },
  {
    vendor: 'The Intelligence Company',
    file: 'IntelligenceCompany',
    svg: ticLogo,
    paths: 3,
    viewBox: '0 0 52 20',
  },
  // two-tone in the Studio (a themed class + a fixed cyan), so no currentColor
  { vendor: 'OpenBMB', file: 'OpenBMB', svg: openbmbLogo, paths: 2, viewBox: '0 0 28 28', ink: false },
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
      if (mark.ink !== false) expect(webMark).toContain('fill="currentColor"')
      expect(webMark).toContain(`aria-label="${mark.vendor}"`)
    })

    it('bundles the native mark locally under the registry vendor name', () => {
      expect(nativeArtwork).toContain(`"${mark.vendor}": "${mark.file}"`)
      expect(mark.svg).not.toMatch(/<(?:script|image|foreignObject)\b|\bhref\s*=/i)
    })
  })
}

// Full-color marks shipped as files on both sides (an <img> asset in the
// Studio, a bundled resource on macOS): the same bytes, mapped under the
// registry vendor name, and rendered from their own luminance on macOS -
// an alpha template would flatten each to a plain shape.
const COPIED = [
  { vendor: 'KBLab', file: 'KBLab', web: kblabWeb, native: kblabNative },
  { vendor: 'LightOn', file: 'LightOn', web: lightonWeb, native: lightonNative },
]

for (const mark of COPIED) {
  describe(`${mark.vendor} provider artwork copy`, () => {
    it('ships the same file in the Studio and the macOS app', () => {
      expect(mark.native).toBe(mark.web)
      expect(mark.web).not.toMatch(/<(?:script|image|foreignObject)\b|\bhref\s*=/i)
      expect(vendorLogo).toContain(`v-else-if="vendor === '${mark.vendor}'"`)
    })

    it('maps the registry vendor name and keeps its colors natively', () => {
      expect(nativeArtwork).toContain(`"${mark.vendor}": "${mark.file}"`)
      expect(nativeArtwork).toMatch(new RegExp(`vendor != "${mark.vendor}"`))
    })
  })
}
