import { describe, expect, it } from 'vitest'
import { cleanOcrText, parseRegionsLive } from './ocr'

// grounding output in LightOnOCR-3's own shape - the same fixture the
// runner's lighton_ocr::parse_blocks test reads
const GROUNDED =
  '![title](102,61,898,95) Quarterly Report\n' +
  '![text](102,120,480,410)\nRevenue grew in every region.\n' +
  '![text+](520,120,898,300) The north led.\n' +
  '![chart](102,450,898,800) <table><tr><td>Q1</td><td>4.2</td></tr></table>\n' +
  '![page_number](480,960,520,980) 3'

describe('LightOnOCR-3 grounding blocks', () => {
  it('parse live the way the runner parses them', () => {
    const rs = parseRegionsLive(GROUNDED)
    expect(rs.map((r) => r.label)).toEqual(['title', 'text', 'text', 'chart', 'page_number'])
    expect(rs[0]).toEqual({ label: 'title', boxes: [[102, 61, 897, 95]], text: 'Quarterly Report' })
    expect(rs[1].text).toBe('Revenue grew in every region.')
    expect(rs[2].continues).toBe(true)
    expect(rs[1].continues).toBeUndefined()
  })

  it('wait for a marker the stream has not closed', () => {
    expect(parseRegionsLive('![title](102,61,898')).toEqual([])
    expect(parseRegionsLive('see ![logo](logo.png)')).toEqual([])
  })

  it('leave only the words for reading', () => {
    expect(cleanOcrText(GROUNDED)).toBe(
      'Quarterly Report\n\nRevenue grew in every region.\nThe north led.\n' +
        '<table><tr><td>Q1</td><td>4.2</td></tr></table>\n3',
    )
    // a half-streamed marker is hidden until it closes
    expect(cleanOcrText('Quarterly Report\n![text](102,12')).toBe('Quarterly Report\n')
    // ordinary markdown images and plain text are untouched
    expect(cleanOcrText('![logo](logo.png) hi')).toBe('![logo](logo.png) hi')
    expect(cleanOcrText('no markup')).toBe('no markup')
  })
})
