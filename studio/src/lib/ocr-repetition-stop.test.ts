import { describe, expect, it } from 'vitest'
import { ocrMetaFromWire } from './ocr'

// PaddleOCR-VL's repetition stop: the server ends a looping read and says so
// in the `ocr` echo; the Studio must carry that into the page's review state
describe('the ocr echo carries the repetition stop', () => {
  const echo = (fired: boolean) => ({
    mode: 'ocr',
    crop: 'base',
    grounding: false,
    pages: 1,
    no_repeat_ngram: { size: 0, window: 0 },
    repetition_stop: { size: 35, window: 128, fired },
  })

  it('reads a fired stop', () => {
    expect(ocrMetaFromWire(echo(true))?.repetitionStop).toBe(true)
  })

  it('stays clean when the stop was armed but never fired, or absent', () => {
    expect(ocrMetaFromWire(echo(false))?.repetitionStop).toBeUndefined()
    expect(ocrMetaFromWire({ mode: 'ocr' })?.repetitionStop).toBeUndefined()
  })
})
