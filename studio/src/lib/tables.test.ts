import { describe, expect, it } from 'vitest'
import classificationExample from '../../../apps/macos/Sources/PaddockConversationCore/Resources/tabular-classification.csv?raw'
import regressionExample from '../../../apps/macos/Sources/PaddockConversationCore/Resources/tabular-regression.csv?raw'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/tables-web-parity.json'
import {
  asNumber,
  buildPlan,
  curlFor,
  defaultSpec,
  DEFAULT_LIMITS,
  exampleClassification,
  exampleRegression,
  formatNumber,
  inferType,
  isMissing,
  limitsFrom,
  parseDelimited,
  quantile,
  readResults,
  resultsCsv,
  type TabularLimits,
} from './tables'

const cls: TabularLimits = { task: 'classification', ...DEFAULT_LIMITS }
const reg: TabularLimits = { task: 'regression', ...DEFAULT_LIMITS }

describe('native Studio parity fixtures', () => {
  it('ships the identical synthetic examples in Swift', () => {
    expect(classificationExample).toBe(exampleClassification())
    expect(regressionExample).toBe(exampleRegression())
  })
  it('uses identical row mapping and requests', () => {
    for (const f of fixtures) {
      const t = parseDelimited(f.text)
      const result = buildPlan(t, defaultSpec(t), f.task === 'classification' ? cls : reg, 8, 7)
      if (!('plan' in result)) throw new Error(result.error)
      expect(t).toEqual(f.table)
      expect(result.plan).toEqual({
        body: f.body, contextRows: f.contextRows, queryRows: f.queryRows,
        features: f.features, classes: f.classes,
      })
    }
  })
})

describe('parseDelimited', () => {
  it('reads RFC 4180 quoting, CRLF and padding', () => {
    const t = parseDelimited('a,b,c\r\n1,"x, y",\n2,"say ""hi""",3\n\n4\n')
    expect(t.header).toEqual(['a', 'b', 'c'])
    expect(t.rows).toEqual([
      ['1', 'x, y', ''],
      ['2', 'say "hi"', '3'],
      ['4', '', ''],
    ])
  })
  it('takes tabs from a spreadsheet paste and semicolons from a European CSV', () => {
    expect(parseDelimited('a\tb\n1\t2').rows).toEqual([['1', '2']])
    expect(parseDelimited('a;b\n1,5;2').rows).toEqual([['1,5', '2']])
  })
  it('keeps a quoted field across lines and names a blank header', () => {
    const t = parseDelimited('note,\n"two\nlines",1')
    expect(t.header).toEqual(['note', 'column 2'])
    expect(t.rows).toEqual([['two\nlines', '1']])
  })
})

describe('cells and types', () => {
  it('knows missing spellings and refuses locale guesses', () => {
    expect(['', 'NA', 'n/a', 'NaN', ' null ', '?', '-'].every(isMissing)).toBe(true)
    expect(isMissing('0')).toBe(false)
    expect(asNumber(' 3.5e2 ')).toBe(350)
    expect(asNumber('1,5')).toBeNull()
    expect(asNumber('1e40')).toBeNull()
    expect(asNumber('0x10')).toBeNull()
  })
  it('infers numerical only when every present value is a number', () => {
    expect(inferType(['1', '', '2.5'])).toBe('numerical')
    expect(inferType(['1', 'two'])).toBe('categorical')
    expect(inferType(['', 'NA'])).toBe('categorical')
  })
})

describe('buildPlan', () => {
  const t = parseDelimited('size,city,label\n1,a,yes\n2,b,no\n,a,yes\n4,c,\n5,,\n')
  it('splits labelled rows from the ones to predict and types the cells', () => {
    const r = buildPlan(t, defaultSpec(t), cls, 8, 7)
    if (!('plan' in r)) throw new Error(r.error)
    expect(r.plan.contextRows).toEqual([0, 1, 2])
    expect(r.plan.queryRows).toEqual([3, 4])
    expect(r.plan.body.context).toEqual([
      [1, 'a'],
      [2, 'b'],
      [null, 'a'],
    ])
    expect(r.plan.body.query).toEqual([
      [4, 'c'],
      [5, null],
    ])
    expect(r.plan.body.targets).toEqual(['yes', 'no', 'yes'])
    expect(r.plan.body.categorical).toEqual([false, true])
    expect(r.plan.classes).toEqual(['yes', 'no'])
    expect(r.plan.body).toMatchObject({ preprocessing: 'sdm_v1', num_estimators: 8, seed: 7 })
  })
  it('leaves out unused columns and refuses what the runner would', () => {
    const spec = defaultSpec(t)
    spec.use[1] = false
    const r = buildPlan(t, spec, cls, 8, 0)
    if (!('plan' in r)) throw new Error(r.error)
    expect(r.plan.features).toEqual([0])
    expect('error' in buildPlan(t, defaultSpec(t), reg, 8, 0)).toBe(true) // yes/no is not a number
    expect('error' in buildPlan(t, defaultSpec(t), cls, 0, 0)).toBe(true)
    expect('error' in buildPlan(t, defaultSpec(t), cls, 17, 0)).toBe(true)
    const all = parseDelimited('x,y\n1,a\n2,b\n')
    expect(buildPlan(all, defaultSpec(all), cls, 8, 0)).toEqual({
      error: 'Every row has a target value. Leave the target empty on the rows you want predicted.',
    })
    const one = parseDelimited('x,y\n1,a\n2,a\n3,\n')
    expect('error' in buildPlan(one, defaultSpec(one), cls, 8, 0)).toBe(true)
    const numeric = parseDelimited('x,y\n1,a\nzz,b\n3,\n')
    const s = defaultSpec(numeric)
    s.types[0] = 'numerical'
    expect(buildPlan(numeric, s, cls, 8, 0)).toMatchObject({ error: expect.stringContaining('"zz"') })
  })
  it('holds the class and cell limits', () => {
    const many = parseDelimited(
      'x,y\n' + Array.from({ length: 11 }, (_, i) => `${i},c${i}`).join('\n') + '\n99,\n',
    )
    expect(buildPlan(many, defaultSpec(many), cls, 8, 0)).toMatchObject({
      error: expect.stringContaining('at most 10 classes'),
    })
    const wide = parseDelimited('a,b,y\n1,2,3\n4,5,\n')
    expect(buildPlan(wide, defaultSpec(wide), { ...reg, maxCells: 3 }, 8, 0)).toMatchObject({
      error: expect.stringContaining('at most 3'),
    })
  })
})

describe('results', () => {
  const t = parseDelimited('x,y\n1,2\n2,4\n3,\n')
  const r = buildPlan(t, defaultSpec(t), reg, 8, 0)
  if (!('plan' in r)) throw new Error(r.error)
  const qs = Array.from({ length: 999 }, (_, i) => 1000 + i)
  it('reads the median and the 80% interval of a regression', () => {
    expect(quantile(qs, 0.1)).toBe(1099)
    expect(quantile(qs, 0.9)).toBe(1899)
    const rows = readResults({ task: 'regression', predictions: [{ median: 1499, quantiles: qs }] }, r.plan)
    expect(rows).toEqual([
      { row: 2, value: '1,499', cell: '1499', confidence: null, probabilities: [], low: 1099, high: 1899 },
    ])
    expect(resultsCsv(t, defaultSpec(t), rows)).toBe('x,y,p10,p90\n1,2,,\n2,4,,\n3,1499,1099,1899\n')
  })
  it('reads the label and confidence of a classification, quoting labels', () => {
    const c = parseDelimited('x,y\n1,"a, b"\n2,c\n3,\n')
    const p = buildPlan(c, defaultSpec(c), cls, 8, 0)
    if (!('plan' in p)) throw new Error(p.error)
    const rows = readResults(
      {
        task: 'classification',
        classes: ['a, b', 'c'],
        predictions: [{ class: 0, label: 'a, b', probabilities: [0.75, 0.25] }],
      },
      p.plan,
    )
    expect(rows[0]).toMatchObject({ value: 'a, b', cell: 'a, b', confidence: 0.75 })
    expect(resultsCsv(c, defaultSpec(c), rows).split('\n')[3]).toBe('3,"a, b",0.7500')
  })
  it('formats numbers readably and shows the call it made', () => {
    expect(formatNumber(1234567.891)).toBe('1,234,567.9')
    expect(formatNumber(0.000012)).toBe('1.200e-5')
    expect(formatNumber(NaN)).toBe('-')
    expect(curlFor(11500, 'kumo', r.plan)).toContain('/v1/tabular/predictions')
  })
})

describe('limits and examples', () => {
  it('reads the runner contract and ignores anything else', () => {
    expect(limitsFrom({ tabular: { model: 'k', capabilities: { task: 'regression', max_columns: 7 } } })).toMatchObject({
      task: 'regression',
      maxColumns: 7,
      maxContextRows: 4096,
    })
    expect(limitsFrom({ model: 'chat' })).toBeNull()
  })
  it('ships examples that plan cleanly for their task', () => {
    for (const [text, limits] of [
      [exampleClassification(), cls],
      [exampleRegression(), reg],
    ] as const) {
      const t = parseDelimited(text)
      const r = buildPlan(t, defaultSpec(t), limits, 8, 0)
      if (!('plan' in r)) throw new Error(r.error)
      expect(r.plan.queryRows).toHaveLength(6)
      expect(r.plan.contextRows).toHaveLength(120)
    }
    expect(exampleClassification()).toBe(exampleClassification())
  })
})
