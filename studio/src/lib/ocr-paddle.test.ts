import { describe, expect, it } from 'vitest'
import { cleanOcrText, ocrMetaFromWire } from './ocr'

// PaddleOCR-VL's document pipeline: its authors' markdown, as the runner
// returns it byte for byte, and the region list of one request over pages
describe('paddleocr document mode', () => {
  it('unwraps captions and drops picture references for display', () => {
    const md =
      '# Title\n\n<div style="text-align: center;"><img src="imgs/img_in_image_box_10_20_300_400.jpg" alt="Image" width="20%" /></div>\n\n\n<div style="text-align: center;">Figure 1: a caption</div>\n\n\nBody text.'
    // each wrapper goes with the newline its handler appended
    expect(cleanOcrText(md)).toBe('# Title\n\n\n\nFigure 1: a caption\n\nBody text.')
    expect(cleanOcrText('plain text')).toBe('plain text')
  })

  it('keeps the page, pixel box and picture name of each region', () => {
    const meta = ocrMetaFromWire({
      mode: 'document',
      regions: [
        { label: 'text', boxes: [[10, 20, 900, 80]], text: 'Hello', page: 0, bbox: [14, 41, 1300, 163] },
        { label: 'image', boxes: [[10, 100, 400, 300]], page: 1, bbox: [14, 204, 579, 612], image: 'imgs/x.jpg' },
      ],
    })
    expect(meta?.mode).toBe('document')
    expect(meta?.regions?.map((r) => r.page)).toEqual([0, 1])
    expect(meta?.regions?.[0].bbox).toEqual([14, 41, 1300, 163])
    expect(meta?.regions?.[1].image).toBe('imgs/x.jpg')
  })
})
