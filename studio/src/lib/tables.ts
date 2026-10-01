// Tables - the Studio's page for a tabular predictor (Kumo-Tabular). The page
// is thin; everything that decides something lives here and is tested beside
// it: reading a pasted or opened table, guessing each column's type, turning
// the table into the runner's `sdm_v1` request (labelled rows teach, rows
// with an empty target get predicted) and reading the answer back.
//
// Nothing here trains anything. Kumo learns in context: the labelled rows go
// up with every request and the model reads them the way a language model
// reads a prompt, so "fit" is one forward pass, not an optimizer.

import { modelLabel } from '@/lib/model-name'

/** One cell as the runner takes it: a number, a category string, or null
 *  for a missing value (paddock_models::kumo::recipe::Cell). */
export type Cell = number | string | null

export interface ParsedTable {
  header: string[]
  rows: string[][]
}

/** Read a delimited table: the first line is the header. Commas by default,
 *  tabs when the header has tabs and no commas (a paste from a spreadsheet),
 *  semicolons when the header has only those (a European CSV). Quoted fields
 *  follow RFC 4180 - `"a, b"` and `"say ""hi"""` - and may span lines.
 *  Unquoted cells are trimmed; blank lines are skipped. Rows shorter than the
 *  header are padded with empty (missing) cells, longer ones are an error the
 *  caller shows. */
/** One catalog model runs as several endpoints here (a classification and a
 *  regression checkpoint), so each one is named by its variant: the served id
 *  past the catalog name ("kumo-tabular-small-regression" -> "small
 *  regression"). The header's picker and anything else listing predictors
 *  use the same words. */
export function predictorLabel(m: { id: string; display?: string }): string {
  const name = m.display ?? modelLabel(m.id)
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  const rest = m.id.toLowerCase().startsWith(slug + '-') ? m.id.slice(slug.length + 1) : ''
  return rest ? `${name} - ${rest.replace(/[-_]+/g, ' ')}` : name
}

export function parseDelimited(text: string): ParsedTable {
  const firstLine = text.slice(0, text.search(/\r?\n|$/))
  const delim =
    firstLine.includes('\t') && !firstLine.includes(',')
      ? '\t'
      : firstLine.includes(';') && !firstLine.includes(',')
        ? ';'
        : ','
  const records: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  let wasQuoted = false
  const endCell = () => {
    row.push(wasQuoted ? cell : cell.trim())
    cell = ''
    wasQuoted = false
  }
  const endRow = () => {
    endCell()
    if (row.length > 1 || row[0] !== '') records.push(row)
    row = []
  }
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          cell += '"'
          i++
        } else {
          quoted = false
        }
      } else {
        cell += c
      }
    } else if (c === '"' && cell.trim() === '') {
      quoted = true
      wasQuoted = true
      cell = ''
    } else if (c === delim) {
      endCell()
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      endRow()
    } else {
      cell += c
    }
  }
  if (cell !== '' || row.length) endRow()
  const [header = [], ...rows] = records
  return {
    header: header.map((h, i) => h || `column ${i + 1}`),
    rows: rows.map((r) => (r.length < header.length ? [...r, ...Array(header.length - r.length).fill('')] : r)),
  }
}

/** Spellings of "no value" people actually leave in tables. */
const MISSING = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '?', '-'])
export function isMissing(cell: string): boolean {
  return MISSING.has(cell.trim().toLowerCase())
}

const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/
/** A cell as a finite number, or null when it is not one. No locale
 *  guessing: `1,5` is text, not one and a half. */
export function asNumber(cell: string): number | null {
  const t = cell.trim()
  if (!NUMBER.test(t)) return null
  const v = Number(t)
  return Number.isFinite(v) && Math.abs(v) <= 1e30 ? v : null
}

export type ColumnType = 'numerical' | 'categorical'

/** Numerical when every present value is a number, categorical otherwise
 *  (and for a column with no values at all - there is nothing to measure). */
export function inferType(values: string[]): ColumnType {
  let seen = false
  for (const v of values) {
    if (isMissing(v)) continue
    if (asNumber(v) === null) return 'categorical'
    seen = true
  }
  return seen ? 'numerical' : 'categorical'
}

/** The last column is the usual place for the outcome. */
export function defaultTarget(t: ParsedTable): number {
  return Math.max(0, t.header.length - 1)
}

export type Task = 'classification' | 'regression'

/** What the runner's capabilities say it accepts (GET /server's `tabular`). */
export interface TabularLimits {
  task: Task
  maxContextRows: number
  maxQueryRows: number
  maxColumns: number
  maxCells: number
  maxClasses: number
  defaultEstimators: number
  maxEstimators: number
}

export const DEFAULT_LIMITS: Omit<TabularLimits, 'task'> = {
  maxContextRows: 4096,
  maxQueryRows: 1024,
  maxColumns: 500,
  maxCells: 131072,
  maxClasses: 10,
  defaultEstimators: 8,
  maxEstimators: 16,
}

/** Read the runner's advertised contract; null when it is not a tabular one. */
export function limitsFrom(server: unknown): TabularLimits | null {
  const caps = (server as { tabular?: { capabilities?: Record<string, unknown> } | null } | null)?.tabular
    ?.capabilities
  if (!caps || (caps.task !== 'classification' && caps.task !== 'regression')) return null
  const n = (k: string, d: number) => (typeof caps[k] === 'number' ? (caps[k] as number) : d)
  return {
    task: caps.task,
    maxContextRows: n('max_context_rows', DEFAULT_LIMITS.maxContextRows),
    maxQueryRows: n('max_query_rows', DEFAULT_LIMITS.maxQueryRows),
    maxColumns: n('max_columns', DEFAULT_LIMITS.maxColumns),
    maxCells: n('max_cells', DEFAULT_LIMITS.maxCells),
    maxClasses: n('max_classes', DEFAULT_LIMITS.maxClasses),
    defaultEstimators: n('default_estimators', DEFAULT_LIMITS.defaultEstimators),
    maxEstimators: n('max_estimators', DEFAULT_LIMITS.maxEstimators),
  }
}

/** Which column is predicted, which feed it, and how each is read. */
export interface TableSpec {
  target: number
  /** per column: false leaves it out of the request (an id, a free-text note) */
  use: boolean[]
  types: ColumnType[]
}

export function defaultSpec(t: ParsedTable): TableSpec {
  const target = defaultTarget(t)
  return {
    target,
    use: t.header.map(() => true),
    types: t.header.map((_, j) => inferType(t.rows.map((r) => r[j] ?? ''))),
  }
}

export interface Plan {
  /** the request body, without `model` (the page adds the one it runs on) */
  body: {
    preprocessing: 'sdm_v1'
    context: Cell[][]
    targets: Cell[]
    query: Cell[][]
    categorical: boolean[]
    num_estimators: number
    seed: number
  }
  /** table rows (0-based, header excluded) behind context and query rows */
  contextRows: number[]
  queryRows: number[]
  /** the feature columns sent, in request order */
  features: number[]
  /** the distinct labels seen (classification only) */
  classes: string[]
}

/** Turn a table into the runner's request, or say plainly why it cannot be
 *  sent. Rows whose target is empty are the ones to predict; every other row
 *  teaches. Categorical cells go up as strings (a column must use one scalar
 *  type), numerical ones as numbers or null. */
export function buildPlan(
  t: ParsedTable,
  spec: TableSpec,
  limits: TabularLimits,
  estimators: number,
  seed: number,
): { plan: Plan } | { error: string } {
  const width = t.header.length
  if (!width || !t.rows.length) return { error: 'The table needs a header row and at least one data row.' }
  const long = t.rows.findIndex((r) => r.length > width)
  if (long >= 0) return { error: `Row ${long + 1} has more cells than the header has columns.` }
  const features = t.header.map((_, j) => j).filter((j) => j !== spec.target && spec.use[j])
  if (!features.length) return { error: 'Choose at least one column besides the target to predict from.' }
  if (features.length > limits.maxColumns)
    return { error: `This model reads at most ${limits.maxColumns} feature columns (the table has ${features.length}).` }
  const contextRows: number[] = []
  const queryRows: number[] = []
  t.rows.forEach((r, i) => (isMissing(r[spec.target] ?? '') ? queryRows : contextRows).push(i))
  if (!contextRows.length) return { error: 'No row has a value in the target column - the model needs labelled rows to learn from.' }
  if (!queryRows.length)
    return { error: 'Every row has a target value. Leave the target empty on the rows you want predicted.' }
  if (contextRows.length > limits.maxContextRows)
    return { error: `At most ${limits.maxContextRows} labelled rows (the table has ${contextRows.length}).` }
  if (queryRows.length > limits.maxQueryRows)
    return { error: `At most ${limits.maxQueryRows} rows to predict at once (the table has ${queryRows.length}).` }
  if ((contextRows.length + queryRows.length) * features.length > limits.maxCells)
    return {
      error: `The table is ${(contextRows.length + queryRows.length) * features.length} cells; this model reads at most ${limits.maxCells}.`,
    }
  const cell = (r: string[], j: number): Cell => {
    const raw = r[j] ?? ''
    if (isMissing(raw)) return null
    return spec.types[j] === 'numerical' ? asNumber(raw) : raw.trim()
  }
  for (const j of features) {
    if (spec.types[j] !== 'numerical') continue
    const bad = t.rows.findIndex((r) => !isMissing(r[j] ?? '') && asNumber(r[j] ?? '') === null)
    if (bad >= 0)
      return {
        error: `Column "${t.header[j]}" is set to numbers but row ${bad + 1} holds "${t.rows[bad][j]}" - make it categorical or fix the cell.`,
      }
  }
  const labels = contextRows.map((i) => t.rows[i][spec.target].trim())
  let targets: Cell[]
  let classes: string[] = []
  if (limits.task === 'classification') {
    classes = [...new Set(labels)]
    if (classes.length < 2)
      return { error: 'The labelled rows show only one class - there is nothing to choose between.' }
    if (classes.length > limits.maxClasses)
      return { error: `This model tells apart at most ${limits.maxClasses} classes (the target has ${classes.length}).` }
    targets = labels
  } else {
    const bad = labels.findIndex((l) => asNumber(l) === null)
    if (bad >= 0)
      return {
        error: `This model predicts numbers, and the target of row ${contextRows[bad] + 1} is "${labels[bad]}".`,
      }
    targets = labels.map((l) => asNumber(l))
  }
  const est = Math.round(estimators)
  if (!(est >= 1 && est <= limits.maxEstimators))
    return { error: `Ensemble size must be 1 to ${limits.maxEstimators}.` }
  if (!Number.isInteger(seed) || seed < 0) return { error: 'The seed must be a whole number, 0 or more.' }
  return {
    plan: {
      body: {
        preprocessing: 'sdm_v1',
        context: contextRows.map((i) => features.map((j) => cell(t.rows[i], j))),
        targets,
        query: queryRows.map((i) => features.map((j) => cell(t.rows[i], j))),
        categorical: features.map((j) => spec.types[j] === 'categorical'),
        num_estimators: est,
        seed,
      },
      contextRows,
      queryRows,
      features,
      classes,
    },
  }
}

/** The runner's answer, as far as this page reads it. */
export interface PredictionResponse {
  task: Task
  classes?: Cell[]
  predictions: {
    class?: number
    label?: Cell
    probabilities?: number[]
    median?: number
    quantiles?: number[]
  }[]
  num_estimators?: number
  usage?: {
    context_rows?: number
    query_rows?: number
    columns?: number
    gpu_ms?: number
    elapsed_ms?: number
  }
}

export interface ResultRow {
  /** table row (0-based) this prediction is for */
  row: number
  /** the predicted class, or the median for a regression - for display */
  value: string
  /** the same value as written back into a table (no digit grouping) */
  cell: string
  /** classification: the winning probability; regression: null */
  confidence: number | null
  /** classification: every class's probability, in `classes` order */
  probabilities: number[]
  /** regression: the 80% interval (10th to 90th percentile) */
  low: number | null
  high: number | null
}

/** The value of quantile `q` from the runner's 999 levels (0.001 .. 0.999). */
export function quantile(qs: number[], q: number): number | null {
  const i = Math.round(q * 1000) - 1
  return i >= 0 && i < qs.length ? qs[i] : null
}

export function readResults(res: PredictionResponse, plan: Plan): ResultRow[] {
  return res.predictions.map((p, k) => {
    const row = plan.queryRows[k]
    if (res.task === 'classification') {
      const probs = p.probabilities ?? []
      const label = String(p.label ?? res.classes?.[p.class ?? 0] ?? p.class ?? '')
      return {
        row,
        value: label,
        cell: label,
        confidence: probs.length ? Math.max(...probs) : null,
        probabilities: probs,
        low: null,
        high: null,
      }
    }
    const qs = p.quantiles ?? []
    const median = p.median ?? quantile(qs, 0.5) ?? NaN
    return {
      row,
      value: formatNumber(median),
      cell: Number.isFinite(median) ? String(median) : '',
      confidence: null,
      probabilities: [],
      low: quantile(qs, 0.1),
      high: quantile(qs, 0.9),
    }
  })
}

/** Enough digits to read, never an exponent for everyday magnitudes. */
export function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return '-'
  const a = Math.abs(v)
  if (a !== 0 && (a >= 1e7 || a < 1e-3)) return v.toExponential(3)
  return v.toLocaleString('en-US', { maximumFractionDigits: a >= 100 ? 1 : a >= 1 ? 3 : 4 })
}

/** The table with the predictions filled into the target column, as CSV. */
export function resultsCsv(t: ParsedTable, spec: TableSpec, results: ResultRow[]): string {
  const quote = (s: string) => (/[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s)
  const byRow = new Map(results.map((r) => [r.row, r]))
  const extra = results.some((r) => r.low !== null) ? ['p10', 'p90'] : results.length ? ['confidence'] : []
  const lines = [[...t.header, ...extra].map(quote).join(',')]
  t.rows.forEach((r, i) => {
    const hit = byRow.get(i)
    const cells = r.map((c, j) => (j === spec.target && hit ? hit.cell : c))
    const tail = !hit
      ? extra.map(() => '')
      : hit.low !== null
        ? [String(hit.low), String(hit.high)]
        : [hit.confidence === null ? '' : hit.confidence.toFixed(4)]
    lines.push([...cells, ...tail].map(quote).join(','))
  })
  return lines.join('\n') + '\n'
}

/** The equivalent API call - teach by example. Long tables are cut to their
 *  first rows so the snippet stays readable; the shape is what matters. */
export function curlFor(port: number, model: string, plan: Plan): string {
  const short = <T>(a: T[]) => (a.length > 3 ? a.slice(0, 3) : a)
  const body = {
    model,
    ...plan.body,
    context: short(plan.body.context),
    targets: short(plan.body.targets),
    query: short(plan.body.query),
  }
  const json = JSON.stringify(body)
  return [
    `curl http://localhost:${port}/v1/tabular/predictions \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "Authorization: Bearer <api key>" \\`,
    `  -d '${json.replace(/'/g, "'\\''")}'`,
    plan.body.context.length > 3 || plan.body.query.length > 3
      ? `# (context, targets and query cut to their first 3 rows here)`
      : '',
  ]
    .filter(Boolean)
    .join('\n')
}

// -- example tables --------------------------------------------------------------
// Made-up data from a fixed generator, so "try it" never ships somebody
// else's dataset and every visitor sees the same table. Each has a handful of
// rows with an empty target - the ones the model predicts.

function rng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

/** A loan-decision table: numbers and categories in, a yes/no out. */
export function exampleClassification(): string {
  const r = rng(20260930)
  const jobs = ['employed', 'self-employed', 'student', 'retired']
  const purposes = ['car', 'home', 'education', 'business']
  const lines = ['income_k,debt_k,years_employed,job,purpose,late_payments,approved']
  for (let i = 0; i < 126; i++) {
    const job = jobs[Math.floor(r() * jobs.length)]
    const income = Math.round(18 + r() * 110 * (job === 'student' ? 0.3 : 1))
    const debt = Math.round(r() * income * 0.9)
    const years = job === 'student' ? 0 : Math.floor(r() * 25)
    const purpose = purposes[Math.floor(r() * purposes.length)]
    const late = Math.floor(r() * r() * 6)
    const score = income / 40 - debt / 30 + years / 10 - late * 0.7 + (purpose === 'education' ? 0.3 : 0) + (r() - 0.5)
    const label = i >= 120 ? '' : score > 0.6 ? 'yes' : 'no'
    lines.push([income, debt, years, job, purpose, late, label].join(','))
  }
  return lines.join('\n') + '\n'
}

/** An apartment-rent table: size, rooms, district and a balcony in, a monthly
 *  rent out. */
export function exampleRegression(): string {
  const r = rng(20260931)
  const districts = ['center', 'harbor', 'university', 'suburb']
  const premium: Record<string, number> = { center: 1.45, harbor: 1.3, university: 1.1, suburb: 0.85 }
  const lines = ['size_m2,rooms,district,floor,balcony,built,rent']
  for (let i = 0; i < 126; i++) {
    const rooms = 1 + Math.floor(r() * 4)
    const size = Math.round(18 + rooms * 17 + r() * 25)
    const district = districts[Math.floor(r() * districts.length)]
    const floor = Math.floor(r() * 9)
    const balcony = r() < 0.45 ? 'yes' : 'no'
    const built = 1930 + Math.floor(r() * 94)
    const rent = Math.round(
      (size * 14 + rooms * 60 + floor * 12 + (balcony === 'yes' ? 90 : 0) + (built - 1930) * 2.2) *
        premium[district] *
        (0.9 + r() * 0.2),
    )
    lines.push([size, rooms, district, floor, balcony, built, i >= 120 ? '' : rent].join(','))
  }
  return lines.join('\n') + '\n'
}
