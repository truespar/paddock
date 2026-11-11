import { describe, expect, it } from 'vitest'
import { cleanOcrText, parseRegionsLive } from './ocr'

// two pages of Unlimited-OCR-2's own output on the battery (trimmed) - the
// same shape the runner's unlimited_ocr2::parse_blocks test reads
const TWO_PAGES =
  '<PAGE><BLOCK>title[81, 60, 390, 88]<FORK>Do At Without\n' +
  '<BLOCK>text[77, 248, 896, 283]<FORK>Across from well under.\n' +
  '<PAGE><BLOCK>title<LEVEL2>[82, 59, 806, 91]<FORK>Only All There\n' +
  '<BLOCK>append_table[81, 136, 918, 170]<FORK><table><tr><td>1</td></tr></table>'

describe('Unlimited-OCR-2 blocks', () => {
  it('parse live the way the runner parses them', () => {
    const rs = parseRegionsLive(TWO_PAGES)
    expect(rs.map((r) => [r.label, r.page])).toEqual([
      ['title', 0],
      ['text', 0],
      ['title', 1],
      ['append_table', 1],
    ])
    expect(rs[0]).toEqual({ label: 'title', boxes: [[81, 60, 390, 88]], text: 'Do At Without', page: 0 })
    expect(rs[3].continues).toBe(true)
    expect(rs[1].continues).toBeUndefined()
  })

  it('wait for a header the stream has not closed', () => {
    expect(parseRegionsLive('<PAGE><BLOCK>text[77, 24')).toEqual([])
  })

  it('leave headings, page breaks and the words for reading', () => {
    expect(cleanOcrText(TWO_PAGES)).toBe(
      '# Do At Without\n\nAcross from well under.\n\n\n---\n\n\n## Only All There\n\n' +
        '<table><tr><td>1</td></tr></table>',
    )
    // a half-streamed header is hidden until it closes, at every cut
    const head = '<BLOCK>title<LEVEL2>[81, 60, 390, 88]<FORK>'
    for (let cut = 1; cut < head.length; cut++) {
      expect(cleanOcrText('<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>Do\n' + head.slice(0, cut))).toBe(
        'Do\n',
      )
    }
    // ...and the words of a finished last block are never taken for one
    expect(cleanOcrText('<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>a < b')).toBe('a < b')
  })
})
