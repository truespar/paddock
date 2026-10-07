import { describe, it, expect } from 'vitest'
import { capsFrom, clickBody, cocoJson, decodeRle, refineFor, requestBody, slug } from './masks'

describe('COCO RLE decode', () => {
  it('reads column-major runs into a row-major plane (zeros first)', () => {
    // 2 rows x 3 columns, column-major: (0,0)=0 (1,0)=1 | (0,1)=1 (1,1)=1 | (0,2)=0 (1,2)=0
    // runs: 1 zero, 3 ones, 2 zeros
    const plane = decodeRle({ size: [2, 3], counts: [1, 3, 2] })
    // row 0: x0=0 x1=1 x2=0 ; row 1: x0=1 x1=1 x2=0
    expect(Array.from(plane)).toEqual([0, 1, 0, 1, 1, 0])
  })
  it('takes a mask that starts set (a leading zero-length run)', () => {
    const plane = decodeRle({ size: [1, 2], counts: [0, 1, 1] })
    expect(Array.from(plane)).toEqual([1, 0])
  })
})

describe('the request and its exports', () => {
  it('sends only what was given', () => {
    expect(requestBody('sam3', 'data:x', '  car ', [], 0.5)).toEqual({ model: 'sam3', image: 'data:x', threshold: 0.5, text: 'car' })
    const b = requestBody('sam3', 'data:x', '', [{ x0: 1, y0: 2, x1: 3, y1: 4, positive: false }], 0.4)
    expect(b.boxes).toEqual([{ box: [1, 2, 3, 4], positive: false }])
    expect(b.text).toBeUndefined()
  })
  it('writes COCO boxes as x, y, width, height', () => {
    const out = JSON.parse(
      cocoJson(10, 8, [
        { prompt: 'car', instances: [{ score: 0.91234, box: [1, 2, 4, 7], area: 6, mask: { size: [8, 10], counts: [80] } }] },
      ]),
    )
    expect(out.annotations[0]).toMatchObject({ id: 1, category: 'car', score: 0.9123, bbox: [1, 2, 3, 5], area: 6 })
    expect(out.image).toEqual({ width: 10, height: 8 })
  })
  it('reads the endpoint contract and names files plainly', () => {
    expect(capsFrom({ masks: { max_pixels: 100, max_boxes: 3, max_prompt_tokens: 30 } })).toEqual({
      maxPixels: 100,
      maxBoxes: 3,
      maxPromptTokens: 30,
      clicks: false,
      maxPoints: 0,
      video: false,
      videoUnavailable: null,
      maxConcepts: 1,
    })
    expect(capsFrom({ masks: { clicks: true, max_points: 32 } })).toMatchObject({ clicks: true, maxPoints: 32 })
    // video sessions: served, or the endpoint's own reason they are not
    expect(capsFrom({ masks: { video: true, video_unavailable: null, max_concepts: 4 } })).toMatchObject({
      video: true,
      videoUnavailable: null,
      maxConcepts: 4,
    })
    expect(capsFrom({ masks: { video: false, video_unavailable: 'the pack predates it' } })).toMatchObject({
      video: false,
      videoUnavailable: 'the pack predates it',
    })
    expect(capsFrom({ masks: null })).toBeNull()
    expect(slug(' Paper bag! ')).toBe('paper-bag')
    expect(slug('***')).toBe('mask')
  })
})

describe('click requests', () => {
  const a = { x: 10, y: 20, positive: true }
  const b = { x: 30, y: 40, positive: false }
  const box = { x0: 1, y0: 2, x1: 50, y1: 60, positive: true }
  it('sends clicks, the object box and a refine handle', () => {
    expect(clickBody('sam3', 'data:x', [a, b], box, '7')).toEqual({
      model: 'sam3',
      image: 'data:x',
      points: [
        { point: [10, 20], positive: true },
        { point: [30, 40], positive: false },
      ],
      object_box: [1, 2, 50, 60],
      refine: '7',
    })
    expect(clickBody('sam3', 'data:x', [a], null)).toEqual({ model: 'sam3', image: 'data:x', points: [{ point: [10, 20], positive: true }] })
  })
  it('refines only a request that adds clicks to the last one', () => {
    const last = { points: [a], box: null, refineId: '3' }
    expect(refineFor(last, [a, b], null)).toBe('3')
    expect(refineFor(last, [a], null)).toBeUndefined()
    expect(refineFor(last, [b, a], null)).toBeUndefined()
    expect(refineFor(last, [a, b], box)).toBeUndefined()
    expect(refineFor({ ...last, box }, [a, b], { ...box })).toBe('3')
    expect(refineFor(null, [a, b], null)).toBeUndefined()
  })
})
