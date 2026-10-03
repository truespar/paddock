// Structured reads - the pure half of the Studio's /v1/systemone client: the
// question model the editor works on, its wire form, the validation the
// runner would otherwise answer with a 422, and the readings the Answers
// card derives from a response. No DOM and no store here, so all of it is
// testable as data (reads.test.ts). The wire shape is the one the runner
// speaks (crates/paddock-runner/src/systemone/) - Jev's request and
// answer forms plus the runner's own `outside` / `agreement` / `stderr` /
// diagnostics, and the example server's extensions (conditional questions,
// multi-step reads, a thought before the read, image input).
import { uuid } from '@/lib/uuid'

export type ReadType = 'noul' | 'choice' | 'score'
/** `"auto"` reads once and re-reads while a slot is unsettled (the runner's
 *  entropy threshold, up to four in all); a number reads exactly that often. */
export type Samples = 'auto' | number

export interface ReadOption {
  name: string
  description: string
}

/** One `ask_if` entry: the question is asked only when the row `key` answered
 *  one of `values` (yes / no, an option's name, a level's name). Rows point at
 *  each other by editor key, never by id - an id follows its instructions
 *  while they are typed, and a condition that named it would break on every
 *  keystroke. The wire form translates keys to ids. */
export interface ReadCondition {
  key: string
  values: string[]
}

/** One question as the editor holds it. Every type's criteria live on the
 *  row at once, so switching the type keeps what was typed for the others
 *  and a mis-click costs nothing (SurveyJS documents the data loss its own
 *  type switch causes). `key` is the row's identity in the editor; `id` is
 *  what goes on the wire, derived from the text until the user edits it. */
export interface ReadQuestion {
  key: string
  id: string
  idTouched: boolean
  type: ReadType
  instructions: string
  yesMeans: string
  noMeans: string
  options: ReadOption[]
  levels: string[]
  /** asked only when every condition holds (`ask_if`, all must match) */
  askIf: ReadCondition[]
  /** read after these rows, with their answers written into its prompt
   *  (`depends_on`, beyond the rows `askIf` already waits for) */
  after: string[]
  /** read on a canvas of its own (`alone`) */
  alone: boolean
}

/** What a block-diffusion endpoint advertises about reads (`/api/server`
 *  `structured_read`): one canvas holds `canvasWidth` positions, and the
 *  per-call caps are the runner's. */
export interface StructuredReadCaps {
  canvasWidth: number
  maxQuestions: number
  maxSamples: number
  /** denoising steps one read may take; 1 is a single-pass reader */
  maxSteps: number
  /** images ride along with the text (the vision companion is loaded and
   *  the backend reads them on a canvas) */
  images: boolean
  /** `ask_if` / `depends_on` / `alone` are served */
  conditional: boolean
  /** the model can write a thought before the read */
  think: boolean
  types: string[]
}

/** The runner's own limits, used when a cap is not advertised. */
export const DEFAULT_MAX_QUESTIONS = 64
export const DEFAULT_MAX_SAMPLES = 32
/** The runner's thought budget ceiling, in tokens. */
export const MAX_THINK = 4096
/** Images one read takes, the runner's cap. */
export const MAX_IMAGES = 16

export const READ_TYPES: { value: ReadType; label: string }[] = [
  { value: 'noul', label: 'Yes / no' },
  { value: 'choice', label: 'Choice' },
  { value: 'score', label: 'Score' },
]

export function newQuestion(type: ReadType = 'noul'): ReadQuestion {
  return {
    key: uuid(),
    id: '',
    idTouched: false,
    type,
    instructions: '',
    yesMeans: '',
    noMeans: '',
    options: [
      { name: '', description: '' },
      { name: '', description: '' },
    ],
    levels: ['', '', ''],
    askIf: [],
    after: [],
    alone: false,
  }
}

export function duplicateQuestion(q: ReadQuestion, taken: Iterable<string>): ReadQuestion {
  return {
    ...q,
    key: uuid(),
    id: uniqueId(q.id || 'q', taken),
    idTouched: true,
    options: q.options.map((o) => ({ ...o })),
    levels: [...q.levels],
    askIf: q.askIf.map((c) => ({ key: c.key, values: [...c.values] })),
    after: [...q.after],
  }
}

/** The answers a question can give, as `ask_if` names them. */
export function answerNames(q: ReadQuestion): string[] {
  if (q.type === 'noul') return ['yes', 'no']
  if (q.type === 'choice') return q.options.map((o) => o.name.trim())
  return q.levels.map((l) => l.trim())
}

/** Every row `q` waits for: its conditions' rows and its `after` rows. */
export function waitsOn(q: ReadQuestion): string[] {
  const out = q.askIf.map((c) => c.key)
  for (const k of q.after) if (!out.includes(k)) out.push(k)
  return out
}

/** Row `key`'s answers were renamed (an option or a level edited in place):
 *  the conditions that name them follow, position by position, so editing
 *  an option's spelling does not orphan the condition built on it. */
export function followRename(
  qs: ReadQuestion[],
  key: string,
  before: string[],
  after: string[],
): void {
  for (const q of qs) {
    for (const c of q.askIf) {
      if (c.key !== key) continue
      c.values = c.values.map((v) => {
        const i = before.indexOf(v)
        return i >= 0 && i < after.length ? after[i] : v
      })
    }
  }
}

/** Row `key` is gone: nothing may wait on it any more. */
export function dropReferences(qs: ReadQuestion[], key: string): void {
  for (const q of qs) {
    q.askIf = q.askIf.filter((c) => c.key !== key)
    q.after = q.after.filter((k) => k !== key)
  }
}

// ── ids ──────────────────────────────────────────────────────────────────────

// Words that carry no meaning in a question's id. "Is the customer angry?"
// should become `customer_angry`, not `is_the_customer`.
const STOP = new Set(
  (
    'a an the is are was were be been being do does did of to in on at for with and or ' +
    'this that these those it its what which how who whom there here any from by as into ' +
    'about than then text state message following above below please rate decide whether ' +
    'if has have had will would should could can not no yes you your we our they their he ' +
    'she him her his them i me my one'
  ).split(' '),
)

/** A single-word id from the instructions: up to three content words joined
 *  with `_`, lowercase, unique against `taken`. */
export function deriveId(instructions: string, taken: Iterable<string> = []): string {
  const words = instructions
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
  let pick = words.filter((w) => !STOP.has(w)).slice(0, 3)
  // nothing but stop words ("What is this message about?"): a question ends
  // on its object, so the last two words name it better than the first two
  if (!pick.length) pick = words.slice(-2)
  let base = pick.join('_').replace(/^[_-]+|[_-]+$/g, '')
  if (base.length > 32) base = base.slice(0, 32).replace(/[_-]+$/, '')
  return uniqueId(base || 'q', taken)
}

/** `base`, or `base_2`, `base_3`... - the first not in `taken`. */
export function uniqueId(base: string, taken: Iterable<string>): string {
  const used = new Set(taken)
  if (!used.has(base)) return base
  for (let n = 2; ; n++) {
    const c = `${base}_${n}`
    if (!used.has(c)) return c
  }
}

/** The runner's rule for a hand-typed id: one word, no `:` - it is written
 *  into the answer template as `id: label`. */
export function cleanId(raw: string): string {
  return raw.replace(/[\s:]+/g, '_')
}

// ── validation (mirrors the runner's 422s, so they are seen before the run) ─

export interface Validation {
  /** first problem per row, keyed by the row's `key` */
  rows: Record<string, string>
  /** problems with the set as a whole */
  set: string[]
  ok: boolean
}

export function validate(qs: ReadQuestion[], caps?: StructuredReadCaps | null): Validation {
  const rows: Record<string, string> = {}
  const set: string[] = []
  if (!qs.length) set.push('Add at least one question.')
  const maxQ = caps?.maxQuestions ?? DEFAULT_MAX_QUESTIONS
  if (qs.length > maxQ) set.push(`${qs.length} questions; this model reads up to ${maxQ} per call.`)
  const seen = new Set<string>()
  for (const q of qs) {
    const id = q.id.trim()
    let err: string | undefined
    if (!id) err = 'Give the question an id - one word.'
    else if (/[\s:]/.test(id)) err = 'The id is one word without ":" - it is written into the answer template.'
    else if (seen.has(id)) err = `Another question already has the id "${id}".`
    else seen.add(id)
    if (!err && q.type === 'choice') {
      const names = q.options.map((o) => o.name.trim())
      if (names.length < 2) err = 'A choice needs at least two options.'
      else if (names.some((n) => !n)) err = 'Every option needs a name.'
      else if (new Set(names).size !== names.length) err = 'Option names must differ.'
      else if (names.length > 26) err = 'A question takes at most 26 options.'
    }
    if (!err && q.type === 'score') {
      const lv = q.levels.map((l) => l.trim())
      if (lv.length < 2) err = 'A score needs at least two levels.'
      else if (lv.some((l) => !l)) err = 'Every level needs a name.'
      else if (new Set(lv).size !== lv.length) err = 'Level names must differ.'
      else if (lv.length > 26) err = 'A score takes at most 26 levels.'
    }
    if (!err) err = conditionError(q, qs)
    if (err) rows[q.key] = err
  }
  const conditional = qs.some((q) => q.askIf.length || q.after.length || q.alone)
  if (conditional && caps && !caps.conditional) {
    set.push('This model reads every question at once; conditions need a newer runner.')
  }
  const loop = cycle(qs)
  if (loop.length) set.push(`These questions wait on each other: ${loop.join(', ')}.`)
  return { rows, set, ok: !set.length && !Object.keys(rows).length }
}

function conditionError(q: ReadQuestion, qs: ReadQuestion[]): string | undefined {
  const byKey = new Map(qs.map((x) => [x.key, x]))
  for (const k of waitsOn(q)) {
    const other = byKey.get(k)
    if (!other) return 'A condition names a question that is gone.'
    if (other === q) return 'A question cannot wait on itself.'
  }
  for (const c of q.askIf) {
    const other = byKey.get(c.key)!
    if (!c.values.length) return `Pick the answers of "${other.id || 'that question'}" to ask on.`
    const names = answerNames(other)
    const off = c.values.find((v) => !names.includes(v))
    if (off !== undefined) return `"${other.id || 'That question'}" cannot answer "${off}".`
  }
  return undefined
}

/** The ids of one loop of rows that wait on each other, or none. */
function cycle(qs: ReadQuestion[]): string[] {
  const byKey = new Map(qs.map((q) => [q.key, q]))
  const mark = new Map<string, 1 | 2>()
  const path: string[] = []
  const visit = (q: ReadQuestion): string[] => {
    mark.set(q.key, 1)
    path.push(q.key)
    for (const k of waitsOn(q)) {
      const next = byKey.get(k)
      if (!next || next === q) continue
      if (mark.get(k) === 1) {
        return path.slice(path.indexOf(k)).map((x) => byKey.get(x)!.id || '?')
      }
      if (!mark.has(k)) {
        const found = visit(next)
        if (found.length) return found
      }
    }
    path.pop()
    mark.set(q.key, 2)
    return []
  }
  for (const q of qs) {
    if (mark.has(q.key)) continue
    const found = visit(q)
    if (found.length) return found
  }
  return []
}

// ── the wire form ────────────────────────────────────────────────────────────

export interface WireQuestion {
  type: ReadType
  instructions: string
  criteria?: Record<string, string> | string[]
  depends_on?: string[]
  ask_if?: Record<string, string[]>
  alone?: boolean
}

/** The `questions` map in question order (the object's key order). */
export function toWire(qs: ReadQuestion[]): Record<string, WireQuestion> {
  const out: Record<string, WireQuestion> = {}
  for (const q of qs) {
    const w: WireQuestion = { type: q.type, instructions: q.instructions.trim() }
    if (q.type === 'noul') {
      const c: Record<string, string> = {}
      if (q.yesMeans.trim()) c.true = q.yesMeans.trim()
      if (q.noMeans.trim()) c.false = q.noMeans.trim()
      if (Object.keys(c).length) w.criteria = c
    } else if (q.type === 'choice') {
      w.criteria = Object.fromEntries(q.options.map((o) => [o.name.trim(), o.description.trim()]))
    } else {
      w.criteria = q.levels.map((l) => l.trim())
    }
    const idOf = (k: string) => qs.find((x) => x.key === k)?.id.trim() ?? k
    if (q.askIf.length) w.ask_if = Object.fromEntries(q.askIf.map((c) => [idOf(c.key), [...c.values]]))
    if (q.after.length) w.depends_on = q.after.map(idOf)
    if (q.alone) w.alone = true
    out[q.id.trim()] = w
  }
  return out
}

export interface ParsedQuestions {
  questions: ReadQuestion[]
  samples?: Samples
  steps?: number
  think?: number
  state?: string
  errors: string[]
}

/** The JSON tab and the import path: a `questions` map, or a whole request
 *  body holding one (`state` and `samples` come along). Rows with problems
 *  are reported and skipped; the rest load. Ids are kept as written - the
 *  runner's own rule decides them at run time, and the editor shows it. */
export function parseQuestionsJson(text: string): ParsedQuestions {
  let doc: unknown
  try {
    doc = JSON.parse(text)
  } catch (e) {
    return { questions: [], errors: [e instanceof Error ? e.message : 'not JSON'] }
  }
  return fromWire(doc)
}

export function fromWire(doc: unknown): ParsedQuestions {
  const errors: string[] = []
  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) {
    return { questions: [], errors: ['The JSON must be an object mapping question ids to questions.'] }
  }
  let map = doc as Record<string, unknown>
  let samples: Samples | undefined
  let steps: number | undefined
  let think: number | undefined
  let state: string | undefined
  const count = (v: unknown, min: number) =>
    typeof v === 'number' && Number.isInteger(v) && v >= min ? v : undefined
  // a whole request body: unwrap it
  if (map.questions && typeof map.questions === 'object' && !('type' in map)) {
    if (typeof map.state === 'string') state = map.state
    if (map.samples === 'auto') samples = 'auto'
    else samples = count(map.samples, 1)
    steps = count(map.steps, 1)
    think = count(map.think, 0)
    map = map.questions as Record<string, unknown>
  }
  const questions: ReadQuestion[] = []
  // conditions name questions by id; they resolve to row keys once every row
  // exists, since a condition may name a question written after it
  const pending: { q: ReadQuestion; w: Record<string, unknown> }[] = []
  for (const [id, raw] of Object.entries(map)) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      errors.push(`${id}: a question is an object with type, instructions and criteria.`)
      continue
    }
    const w = raw as Record<string, unknown>
    const t = typeof w.type === 'string' ? w.type : ''
    const q = newQuestion()
    q.id = id
    q.idTouched = true
    q.instructions = typeof w.instructions === 'string' ? w.instructions : ''
    const c = w.criteria
    if (t === 'noul' || t === 'bool' || t === 'boolean') {
      q.type = 'noul'
      if (c && typeof c === 'object' && !Array.isArray(c)) {
        const cc = c as Record<string, unknown>
        if (typeof cc.true === 'string') q.yesMeans = cc.true
        if (typeof cc.false === 'string') q.noMeans = cc.false
      }
    } else if (t === 'choice') {
      q.type = 'choice'
      if (c && typeof c === 'object' && !Array.isArray(c)) {
        q.options = Object.entries(c as Record<string, unknown>).map(([name, d]) => ({
          name,
          description: typeof d === 'string' ? d : d == null ? '' : String(d),
        }))
      } else {
        errors.push(`${id}: choice criteria map option names to descriptions.`)
        continue
      }
    } else if (t === 'score') {
      q.type = 'score'
      if (Array.isArray(c)) {
        q.levels = c.map((l) => (typeof l === 'string' ? l : String(l)))
      } else {
        errors.push(`${id}: score criteria are an ordered list of levels.`)
        continue
      }
    } else {
      errors.push(`${id}: unknown type "${t}" (noul, choice or score).`)
      continue
    }
    q.alone = w.alone === true
    questions.push(q)
    pending.push({ q, w })
  }
  const keyOf = new Map(questions.map((q) => [q.id, q.key]))
  for (const { q, w } of pending) {
    const named = (dep: string): string | undefined => {
      const k = keyOf.get(dep)
      if (!k) errors.push(`${q.id}: its condition names "${dep}", which is not a question here.`)
      return k
    }
    if (w.ask_if && typeof w.ask_if === 'object' && !Array.isArray(w.ask_if)) {
      for (const [dep, vals] of Object.entries(w.ask_if as Record<string, unknown>)) {
        const k = named(dep)
        const values = (Array.isArray(vals) ? vals : [vals]).map(String)
        if (k) q.askIf.push({ key: k, values })
      }
    }
    if (Array.isArray(w.depends_on)) {
      for (const dep of w.depends_on.map(String)) {
        const k = named(dep)
        if (k && !q.askIf.some((c) => c.key === k) && !q.after.includes(k)) q.after.push(k)
      }
    }
  }
  const out: ParsedQuestions = { questions, errors }
  if (samples !== undefined) out.samples = samples
  if (steps !== undefined) out.steps = steps
  if (think !== undefined) out.think = think
  if (state !== undefined) out.state = state
  return out
}

export interface ReadRequest {
  state: string
  questions: Record<string, WireQuestion>
  samples?: number
  /** denoising steps before the answer is read; absent is one pass */
  steps?: number
  /** a thought budget in tokens, written before the read; absent is none */
  think?: number
  /** data: URLs, read with the text */
  images?: string[]
}

/** The read's settings beside its questions. */
export interface ReadOptions {
  steps?: number
  think?: number
  images?: string[]
}

export function requestBody(
  state: string,
  qs: ReadQuestion[],
  samples: Samples,
  opts: ReadOptions = {},
): ReadRequest {
  const body: ReadRequest = { state, questions: toWire(qs) }
  if (samples !== 'auto') body.samples = samples
  if (opts.steps && opts.steps > 1) body.steps = opts.steps
  if (opts.think && opts.think > 0) body.think = opts.think
  if (opts.images?.length) body.images = [...opts.images]
  return body
}

/** The equivalent curl, with the body as a heredoc - a state text has
 *  quotes and newlines, which a `-d '...'` would break on. With images the
 *  call is the multipart form instead: the JSON in a part named `request`
 *  and each picture as a file part, so the example names the files rather
 *  than printing megabytes of base64. */
export function curlFor(port: number, body: ReadRequest, imageNames: string[] = []): string {
  if (body.images?.length) {
    const { images, ...rest } = body
    const names = images.map((_, i) => fileArg(imageNames[i] ?? `image-${i + 1}.png`))
    return [
      `cat > request.json <<'JSON'`,
      JSON.stringify(rest, null, 2),
      'JSON',
      `curl http://localhost:${port}/v1/systemone \\`,
      `  -H "Authorization: Bearer <api key>" \\`,
      `  -F "request=<request.json;type=application/json" \\`,
      ...names.map((n, i) => `  -F "image=@${n}"${i < names.length - 1 ? ' \\' : ''}`),
    ].join('\n')
  }
  return [
    `curl http://localhost:${port}/v1/systemone \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "Authorization: Bearer <api key>" \\`,
    `  -d @- <<'JSON'`,
    JSON.stringify(body, null, 2),
    'JSON',
  ].join('\n')
}

/** A file name as a curl `-F` value inside double quotes: curl reads `;`
 *  and `,` as field separators there, and the shell reads `"`, `$`, the
 *  backtick and the backslash. */
function fileArg(name: string): string {
  return name.replace(/[";,$`\\]/g, '_')
}

// ── the answer ───────────────────────────────────────────────────────────────

interface AnswerCommon {
  /** Jev's confidence measure, computed by the runner: `(n * max - 1) /
   *  (n - 1)` over the mean label probabilities, 0 at uniform and 1 at
   *  certainty - so a two-way 0.62 / 0.38 answer reads 0.24, not 0.62. The
   *  probabilities beside it are the raw distribution. */
  confidence: number
  /** the share of reads that picked the reported label (a canvas reader; a
   *  decision model reads once, deterministically, and sends none) */
  agreement?: number
  /** the mass the model put outside the label set at this slot - a canvas
   *  reader's; a decision model scores only its options and has none */
  outside?: number
  /** a decision model's calibrated confidence: the reported option's
   *  probability, what its temperatures were fitted to (Laya) */
  answer_confidence?: number
  /** a decision model's act head (Laya): act rather than escalate */
  action?: { act_probability?: number }
  /** the standard error of the reported label's probability across the
   *  reads (sample variance over n - 1, divided by n); absent after one read */
  stderr?: number
}
export interface ReadAnswerNoul extends AnswerCommon {
  type: 'noul'
  noul: number
}
export interface ReadAnswerChoice extends AnswerCommon {
  type: 'choice'
  choice: string
  probabilities: Record<string, number>
}
export interface ReadAnswerScore extends AnswerCommon {
  type: 'score'
  score: number
  /** the standard error of `score` across the reads */
  score_stderr?: number
  level: string
  legend: Record<string, string>
  probabilities: Record<string, number>
}
export type ReadAnswer = ReadAnswerNoul | ReadAnswerChoice | ReadAnswerScore

export interface ReadDiagRead {
  pick: string
  confidence: number
  /** the answer's entropy (see ReadDiagQuestion) */
  entropy: number
  slot_entropy?: number
  /** the slot's most likely token over the whole vocabulary, as text */
  argmax?: string
}
export interface ReadDiagQuestion {
  id: string
  label: string
  /** the answer slot's canvas position (a canvas reader) */
  position?: number
  /** the ANSWER's entropy: over the labels, each label's spellings summed,
   *  plus the mass outside them as one more outcome. A runner from before
   *  `slot_entropy` sent the whole-vocabulary figure here instead. */
  entropy: number
  /** the slot's entropy over the whole vocabulary */
  slot_entropy?: number
  label_mass?: number
  /** one entry per read - a runner from before the field sends none */
  reads?: ReadDiagRead[]
  // ---- a decision model's (Laya) ----
  /** options the question scored */
  options?: number
  /** tokens of each sequence the question read (one, or one per window) */
  tokens?: number[]
  /** the fitted temperature its logits were divided by */
  temperature?: number
  /** Laya's own confidence: 1 - the normalised entropy */
  entropy_confidence?: number
  /** a state too long for one sequence is read in windows; this one decided */
  window?: { index: number; count: number; token_start: number; token_end: number } | null
  /** every option was cut to this many tokens to fit the option budget */
  options_cut_to?: number | null
  // ---- Clef's ----
  /** the options in the order the model read them */
  option_ids?: string[]
  /** each option's raw logit, in that order */
  logits?: number[]
}
/** A thought the model wrote before its read. */
export interface ReadThought {
  text: string
  tokens: number
  /** it ended on its own, rather than at the budget */
  closed: boolean
  ms: number
}

/** Why a conditional question was not asked. */
export interface ReadSkip {
  because: string
  was: string
  wanted: string[]
}

export interface ReadResponse {
  model: string
  /** a question its conditions left unasked answers null */
  answers: Record<string, ReadAnswer | null>
  usage?: { input_tokens?: number; output_tokens?: number }
  /** a decision model's router: which checkpoint read the state, and why */
  routing?: { model: string; reason: string }
  diagnostics: {
    /** 'laya' / 'clef' for a decision model; absent for a canvas reader */
    backend?: 'laya' | 'clef'
    /** the checkpoint that answered (a decision model) */
    checkpoint?: string
    state_tokens?: number
    /** Clef: the state's tokens the sequence carried (all of them, or the
     *  caller's `max_state_tokens` cut) */
    state_read?: number
    windowed?: boolean
    reads: number
    canvas?: number
    steps?: number
    format?: 'lines' | 'indexed'
    /** the pictures read with the text */
    images?: number
    /** Clef: each picture as decoded, the size the processor resized it to
     *  (`[width, height]`) and the tokens it took */
    pictures?: { width: number; height: number; resized: [number, number]; tokens: number }[]
    /** the questions read together, in the order they were read */
    stages?: string[][]
    chunks?: string[][]
    skipped?: Record<string, ReadSkip>
    /** how later stages saw earlier answers: in the prompt (`prefill`) or
     *  restated beside an image (`restated`); null for one stage */
    conditioning?: string | null
    thought?: ReadThought | ReadThought[] | null
    questions: ReadDiagQuestion[]
    timing: { total_ms: number; gpu_ms?: number; passes?: number; pass_requests?: number }
  }
}

/** The thoughts of a response, as a list. */
export function thoughtsOf(r: ReadResponse): ReadThought[] {
  const t = r.diagnostics.thought
  if (!t) return []
  return Array.isArray(t) ? t : [t]
}

/** The runner re-reads a question whose answer entropy is above this (its
 *  `auto` threshold) - so below it the answer is settled by the runner's own
 *  measure, and above it the runner itself called it unsettled and read
 *  again. Above ln 2 the answer holds at least a coin flip between two
 *  outcomes. The entropy counts the mass outside the labels as one outcome,
 *  so an answer the model is not giving reads unsettled too; the meta line
 *  prints `outside` right beside it to tell the two apart. */
export const ENTROPY_SETTLED = 0.1
export const ENTROPY_SPLIT = Math.LN2

export type EntropyWord = 'settled' | 'unsettled' | 'split'
export function entropyWord(e: number): EntropyWord {
  if (!(e >= 0)) return 'split'
  if (e < ENTROPY_SETTLED) return 'settled'
  if (e < ENTROPY_SPLIT) return 'unsettled'
  return 'split'
}

/** Confidence in four bins with visible edges - the reference bands people
 *  already use (TypeSafe's act / review / human thresholds at 0.9 and 0.5,
 *  and 0.7 between). The number is always printed beside the colour. */
export const CONFIDENCE_EDGES = [0.5, 0.7, 0.9] as const
export const CONFIDENCE_BINS = ['under 0.5', '0.5 to 0.7', '0.7 to 0.9', '0.9 and over'] as const
export function confidenceBin(p: number): 0 | 1 | 2 | 3 {
  if (!(p >= CONFIDENCE_EDGES[0])) return 0
  if (p < CONFIDENCE_EDGES[1]) return 1
  if (p < CONFIDENCE_EDGES[2]) return 2
  return 3
}

export function fmtP(p: number | undefined): string {
  if (p === undefined || !Number.isFinite(p)) return '-'
  if (p > 0 && p < 0.005) return '<0.01'
  return p.toFixed(2)
}

export interface Bar {
  name: string
  p: number
  role: 'winner' | 'other' | 'outside'
}

const OUTSIDE = 'outside the options'

/** The outside-the-labels bar, when the reader measured one - a decision
 *  model scores only its options and reports none. */
function outsideBar(a: AnswerCommon): Bar[] {
  return a.outside === undefined ? [] : [{ name: OUTSIDE, p: Math.max(0, a.outside), role: 'outside' }]
}

/** A score level as text: a decision model echoes the caller's level as
 *  written, which may be a JSON object rather than a name. */
export function levelText(v: unknown): string {
  return typeof v === 'string' ? v : JSON.stringify(v)
}

/** Ranked by mass, the winner marked, the outside-the-labels mass as its
 *  own last bar (the OpenAI `refusal` separation: never folded into a
 *  label). */
export function choiceBars(a: ReadAnswerChoice): Bar[] {
  const winner = String(a.choice)
  const bars: Bar[] = Object.entries(a.probabilities)
    .map(([name, p]) => ({ name, p, role: name === winner ? ('winner' as const) : ('other' as const) }))
    .sort((x, y) => y.p - x.p)
  bars.push(...outsideBar(a))
  return bars
}

/** In level order - the order carries the meaning of a score. */
export function scoreBars(a: ReadAnswerScore): Bar[] {
  const bars: Bar[] = Object.keys(a.legend)
    .map(Number)
    .filter((i) => Number.isInteger(i))
    .sort((x, y) => x - y)
    .map((i) => {
      const name = levelText(a.legend[String(i)])
      return { name, p: a.probabilities[String(i)] ?? 0, role: name === a.level ? ('winner' as const) : ('other' as const) }
    })
  bars.push(...outsideBar(a))
  return bars
}

export function noulBars(a: ReadAnswerNoul): Bar[] {
  const yes = a.noul
  const bars: Bar[] = [
    { name: 'yes', p: yes, role: yes >= 0.5 ? 'winner' : 'other' },
    { name: 'no', p: 1 - yes, role: yes >= 0.5 ? 'other' : 'winner' },
  ]
  bars.push(...outsideBar(a))
  return bars
}

/** The two largest label masses within 0.05 of each other. */
export function nearTie(bars: Bar[]): boolean {
  const ps = bars
    .filter((b) => b.role !== 'outside')
    .map((b) => b.p)
    .sort((x, y) => y - x)
  return ps.length >= 2 && ps[0] - ps[1] < 0.05
}

/** Where the fractional score sits on the level scale, 0..1. */
export function scorePosition(a: ReadAnswerScore): number {
  const n = Object.keys(a.legend).length
  if (n < 2) return 0
  return Math.min(1, Math.max(0, a.score / (n - 1)))
}

/** The reported label of an answer, as the user named it. */
export function answerLabel(a: ReadAnswer): string {
  if (a.type === 'noul') return a.noul >= 0.5 ? 'yes' : 'no'
  if (a.type === 'choice') return String(a.choice)
  return a.level
}

// ── where a runner refusal belongs on the page ──────────────────────────────

export type ErrorTarget =
  | { where: 'row'; id: string }
  | { where: 'state' }
  | { where: 'questions' }
  | { where: 'page' }

/** The runner names the question it refuses (`question "id": ...`), and its
 *  window refusal names the prompt - so each 422 lands next to the field it
 *  is about instead of in a strip at the top. */
export function routeError(msg: string): ErrorTarget {
  const m = /^question "((?:[^"\\]|\\.)+)":/.exec(msg)
  if (m) return { where: 'row', id: m[1].replace(/\\(.)/g, '$1') }
  if (/the window is|^state:/.test(msg)) return { where: 'state' }
  if (/^images:/.test(msg)) return { where: 'state' }
  if (/^questions:|the answer template needs|^(samples|steps|think|ask|chunk_rows|chunk_prompt|sequential):/.test(msg))
    return { where: 'questions' }
  return { where: 'page' }
}

// ── shared SQLite history ───────────────────────────────────────────────────

/** One run of a read: what was sent and what came back. The text rides
 *  along whole, so stepping back to an earlier run shows what it read. */
export interface ReadRun {
  id?: string
  at: number
  model: string
  port: number
  /** the first line or so of the state, for labels */
  excerpt: string
  chars: number
  /** the whole text the run read */
  state: string
  /** Imported legacy results had no full input. Never substitute the excerpt. */
  stateMissing?: boolean
  questionOrder?: string[][]
  /** the file the text came from; '' when it was typed or pasted */
  fileName: string
  questions: Record<string, WireQuestion>
  samples: Samples
  /** denoising steps and thought budget, when the run used them */
  steps?: number
  think?: number
  /** the pictures read with the text, by name and by the key their bytes
   *  are kept under in the read's `images` */
  images?: ReadImageRef[]
  response: ReadResponse
  ms: number
}

export interface ReadImageRef {
  name: string
  ref: string
}

/** A read: a text, its questions and every run made of them. It is the
 *  Reads page's unit of history, listed in the side panel the way a chat
 *  lists conversations and kept by the manager (/api/read-history), so a
 *  read made in one browser is there in the next. */
export interface ReadDoc {
  revision?: string
  id: string
  title: string
  /** the reader of the latest run */
  model: string
  createdAt: number
  updatedAt: number
  /** oldest first; the page shows the last one */
  runs: ReadRun[]
  /** the pictures the runs read, each kept once however many runs used it,
   *  keyed by a hash of its data URL */
  images?: Record<string, string>
}

/** A side-panel row. `runs` is a count here, never the runs themselves. */
export interface ReadSummary {
  revision?: string
  id: string
  title: string
  model: string
  runs: number
  createdAt: number
  updatedAt: number
}

/** Swift dictionaries do not preserve JSON key order. Both clients honor the
 * explicit order; refuse malformed metadata instead of silently dropping rows. */
export function orderedRunQuestions(run: ReadRun): Record<string, WireQuestion> {
  if (!run.questionOrder) return run.questions
  const keys = Object.keys(run.questions)
  const ids = run.questionOrder.map((row) => row[0])
  if (ids.length !== keys.length || new Set(ids).size !== keys.length ||
    ids.some((id) => !id || !keys.includes(id))) throw new Error('Invalid saved question order')
  return Object.fromEntries(run.questionOrder.map(([id, ...options]) => {
    const q = run.questions[id!]!
    if (q.type !== 'choice') return [id!, q]
    const criteria = q.criteria as Record<string, string>
    const names = Object.keys(criteria ?? {})
    if (options.length !== names.length || new Set(options).size !== names.length ||
      options.some((name) => !names.includes(name))) throw new Error('Invalid saved choice order')
    return [id!, { ...q, criteria: Object.fromEntries(options.map((name) => [name, criteria[name]])) }]
  }))
}

/** Runs kept per read; the oldest go first. Each run holds its text, so this
 *  is what bounds a read's size. */
export const READ_RUNS_KEEP = 20

/** A new read is named after where its text came from: the file, else the
 *  text's first line. */
export function readTitle(state: string, fileName: string): string {
  if (fileName.trim()) return fileName.trim()
  const first = state.split(/\r?\n/).find((l) => l.trim()) ?? ''
  return excerptOf(first, 60) || 'Untitled read'
}

/** Append a run, dropping the oldest past READ_RUNS_KEEP. */
export function withRun(runs: ReadRun[], run: ReadRun): ReadRun[] {
  return [...runs, run].slice(-READ_RUNS_KEEP)
}

/** The key a picture's bytes are kept under: a 53-bit hash of its data URL
 *  (cyrb53) and its length. It only has to tell one read's pictures apart,
 *  and it works where `crypto.subtle` does not - the Studio reached over
 *  plain http on the LAN is not a secure context. */
export function imageRef(url: string): string {
  let h1 = 0xdeadbeef
  let h2 = 0x41c6ce57
  for (let i = 0; i < url.length; i++) {
    const c = url.charCodeAt(i)
    h1 = Math.imul(h1 ^ c, 2654435761)
    h2 = Math.imul(h2 ^ c, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  const h = 4294967296 * (2097151 & h2) + (h1 >>> 0)
  return `${url.length.toString(36)}-${h.toString(36)}`
}

/** The read's pictures with `added` in, and none that no run still uses. */
export function keptImages(
  runs: ReadRun[],
  have: Record<string, string> | undefined,
  added: Record<string, string>,
): Record<string, string> | undefined {
  const all = { ...(have ?? {}), ...added }
  const used = new Set(runs.flatMap((r) => (r.images ?? []).map((i) => i.ref)))
  const out = Object.fromEntries(Object.entries(all).filter(([k]) => used.has(k)))
  return Object.keys(out).length ? out : undefined
}

export function excerptOf(state: string, max = 120): string {
  const s = state.trim().replace(/\s+/g, ' ')
  return s.length > max ? `${s.slice(0, max - 3)}...` : s
}
