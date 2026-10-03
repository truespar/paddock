<script setup lang="ts">
// The answers of one read, one block per question in the run's own order:
// the reported label first, the uncertainty beside it and never inside the
// control (an uncertainty painted into the answer anchors the reader). A
// yes/no is a marker on a 0-1 rail with faint reference marks; a choice is
// ranked bars with the outside-the-options mass as its own bar; a score is
// the marker on its level scale over the per-level bars. Colour is a binned
// confidence swatch with a legend, and the number is always printed. Past one
// read the standard error sits beside the agreement: how far the reported
// probability would move on another set of reads. A question its conditions
// left unasked keeps its place and says why; a thought written before the
// read opens above the answers.
import { computed } from 'vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import ProbBars from './ProbBars.vue'
import ReadHeatmap from './ReadHeatmap.vue'
import { fmtClock } from '@/lib/format'
import {
  CONFIDENCE_BINS,
  answerLabel,
  choiceBars,
  confidenceBin,
  entropyWord,
  fmtP,
  levelText,
  nearTie,
  scoreBars,
  scorePosition,
  thoughtsOf,
  type Bar,
  type ReadAnswer,
  type ReadAnswerScore,
  type ReadRun,
  type ReadSkip,
} from '@/lib/reads'

const props = defineProps<{
  run: ReadRun | null
  /** which run of the open read is shown, and how many it has: the answers
   *  step through one read's runs, never through other reads - those are
   *  the side panel's */
  runIndex: number
  runCount: number
  busy: boolean
  /** the editor has moved on since this read: the answers on show are of
   *  a request that is no longer the one Run would send */
  stale: boolean
  /** a reader is up and the page can fill and run its worked example */
  canExample: boolean
  /** the read's kept pictures, by key - the run names the ones it read */
  images?: Record<string, string> | undefined
}>()
const emit = defineEmits<{ (e: 'step', delta: number): void; (e: 'example'): void }>()

const runWhen = computed(() => (props.run ? fmtClock(new Date(props.run.at)) : ''))

interface Block {
  id: string
  instructions: string
  answer: ReadAnswer
  label: string
  bars: Bar[] | null
  tie: boolean
  entropy: number
  /** `entropy` is the answer's (a current runner) or the slot's (an older run) */
  entropyOf: 'answer' | 'slot'
  reads: number
  agreement: string | null
}
interface Skipped {
  id: string
  instructions: string
  why: string
}

/** "kind answered question; asked only if it is bug" */
function skipText(s: ReadSkip): string {
  const wanted = s.wanted.length === 1 ? s.wanted[0] : `one of ${s.wanted.join(', ')}`
  return `${s.because} answered ${s.was}; asked only if it is ${wanted}`
}

const blocks = computed<(Block | Skipped)[]>(() => {
  const run = props.run
  if (!run) return []
  const skipped = run.response.diagnostics.skipped ?? {}
  const reads = run.response.diagnostics.reads
  const diag = new Map(run.response.diagnostics.questions.map((d) => [d.id, d]))
  // the run's own question order; answers the runner added on its own (none
  // today) trail behind
  const ids = Object.keys(run.questions)
  for (const id of Object.keys(run.response.answers)) if (!ids.includes(id)) ids.push(id)
  return ids.flatMap((id): (Block | Skipped)[] => {
    const answer = run.response.answers[id]
    if (!answer) {
      const s = skipped[id]
      return s ? [{ id, instructions: run.questions[id]?.instructions ?? '', why: skipText(s) }] : []
    }
    const bars =
      answer.type === 'choice' ? choiceBars(answer) : answer.type === 'score' ? scoreBars(answer) : null
    const k = Math.round((answer.agreement ?? 1) * reads)
    return [
      {
        id,
        instructions: run.questions[id]?.instructions ?? '',
        answer,
        label: answerLabel(answer),
        bars,
        tie: bars ? nearTie(bars) : Math.abs((answer.type === 'noul' ? answer.noul : 0.5) - 0.5) < 0.025,
        entropy: diag.get(id)?.entropy ?? Number.NaN,
        // a decision model's entropy is its answer's by construction; on a
        // canvas reader a missing slot figure marks a run from before it
        entropyOf:
          run.response.diagnostics.backend !== undefined || diag.get(id)?.slot_entropy !== undefined
            ? 'answer'
            : 'slot',
        reads,
        agreement: reads > 1 ? `${k} of ${reads} reads agree` : null,
      },
    ]
  })
})

const heatRows = computed(() => {
  const run = props.run
  if (!run || run.response.diagnostics.reads < 2) return []
  const rows = run.response.diagnostics.questions
    .filter((d) => d.reads && d.reads.length > 1)
    .map((d) => ({ id: d.id, reads: d.reads ?? [] }))
  return rows
})

function scaleLabels(a: ReadAnswerScore): { name: string; at: number; show: boolean }[] {
  const names = Object.keys(a.legend)
    .map(Number)
    .sort((x, y) => x - y)
    .map((i) => levelText(a.legend[String(i)]))
  const n = names.length
  return names.map((name, i) => ({
    name,
    at: n > 1 ? i / (n - 1) : 0,
    // six or more levels cannot all be lettered on one line: keep the ends
    // and the reported one
    show: n <= 5 || i === 0 || i === n - 1 || name === a.level,
  }))
}

const raw = computed(() => (props.run ? JSON.stringify(props.run.response, null, 2) : ''))

const thoughts = computed(() => (props.run ? thoughtsOf(props.run.response) : []))
const pictures = computed(() =>
  (props.run?.images ?? []).flatMap((i) => {
    const url = props.images?.[i.ref]
    return url ? [{ name: i.name, url }] : []
  }),
)
const steps = computed(() => props.run?.response.diagnostics.steps ?? 1)
const diag = computed(() => props.run?.response.diagnostics)
/** the tokens a question's reads put first over the whole vocabulary,
 *  each once, in order */
function topTokens(reads: { argmax?: string }[] | undefined): string {
  const seen: string[] = []
  for (const r of reads ?? []) {
    if (r.argmax === undefined) continue
    const t = JSON.stringify(r.argmax)
    if (!seen.includes(t)) seen.push(t)
  }
  return seen.join(' ') || '-'
}
function isSkipped(b: Block | Skipped): b is Skipped {
  return 'why' in b
}
</script>

<template>
  <section class="ac">
    <div class="ac__head">
      <h2 class="ac__title">Answers</h2>
      <span v-if="busy" class="ac__meta ac__meta--live">
        <Icon name="spinner" :size="12" class="spin" /> reading
      </span>
      <span v-else-if="run" class="ac__meta">
        {{ run.ms }} ms · {{ run.response.diagnostics.reads }}
        read{{ run.response.diagnostics.reads === 1 ? '' : 's' }}<template v-if="steps > 1">
          · {{ steps }} steps</template>
      </span>
      <span v-if="run && stale && !busy" class="ac__stale">edited since this run</span>
      <!-- the same < 2/3 > a chat turn shows for its versions: here, the runs
           of this read, oldest first, the latest on show until stepped back -->
      <nav v-if="runCount > 1" class="ac__runs" aria-label="Runs of this read">
        <Tooltip label="Earlier run">
          <button
            class="ac__runbtn"
            type="button"
            :disabled="busy || runIndex <= 0"
            aria-label="Earlier run"
            @click="emit('step', -1)"
          >
            <Icon name="chevron-left" :size="13" />
          </button>
        </Tooltip>
        <Tooltip :label="`Run ${runIndex + 1} of ${runCount}, at ${runWhen}`">
          <span class="ac__runcount" aria-live="polite">{{ runIndex + 1 }}/{{ runCount }}</span>
        </Tooltip>
        <Tooltip label="Later run">
          <button
            class="ac__runbtn"
            type="button"
            :disabled="busy || runIndex >= runCount - 1"
            aria-label="Later run"
            @click="emit('step', 1)"
          >
            <Icon name="chevron-right" :size="13" />
          </button>
        </Tooltip>
      </nav>
    </div>

    <div v-if="busy && !run" class="ac__empty">
      <Icon name="spinner" :size="20" class="spin" />
      <p>Reading...</p>
    </div>
    <div v-else-if="!run" class="ac__empty">
      <p>Run a read to see the answers here.</p>
      <button v-if="canExample" class="pk-btn pk-btn--sm" type="button" @click="emit('example')">
        <Icon name="play" :size="13" /> Try an example
      </button>
    </div>

    <template v-else>
      <div v-if="pictures.length" class="ac__pics" aria-label="Images read with the text">
        <img v-for="(pic, i) in pictures" :key="i" class="ac__pic" :src="pic.url" :alt="pic.name" />
      </div>

      <Collapsible
        v-for="(t, i) in thoughts"
        :key="`thought-${i}`"
        class="ac__thought"
        :summary="thoughts.length > 1 ? `Thought ${i + 1}` : 'Thought'"
        :hint="`${t.tokens} tokens, ${t.closed ? 'finished' : 'cut at the budget'}, ${Math.round(t.ms)} ms`"
      >
        <p class="ac__thoughttext">{{ t.text.trim() || '(empty)' }}</p>
      </Collapsible>

      <template v-for="b in blocks" :key="b.id">
      <article v-if="isSkipped(b)" class="ac__q ac__q--skip">
        <header class="ac__qhead">
          <span class="ac__qid">{{ b.id }}</span>
          <span class="ac__qtext">{{ b.instructions }}</span>
        </header>
        <p class="ac__line">Not asked: {{ b.why }}.</p>
      </article>
      <article v-else class="ac__q">
        <header class="ac__qhead">
          <span class="ac__qid">{{ b.id }}</span>
          <span class="ac__qtext">{{ b.instructions }}</span>
        </header>

        <div class="ac__answer">
          <span class="ac__label">{{ b.label }}</span>
          <span class="ac__conf">
            <i class="ac__dot" :class="`ac__dot--${confidenceBin(b.answer.confidence)}`" />
            {{ fmtP(b.answer.confidence) }}
          </span>
          <span v-if="b.tie" class="ac__tie">near tie</span>
        </div>

        <div v-if="b.answer.type === 'noul'" class="rail">
          <div class="rail__track">
            <span class="rail__tick" :style="{ left: '30%' }" />
            <span class="rail__tick" :style="{ left: '70%' }" />
            <span class="rail__fill" :style="{ width: `${b.answer.noul * 100}%` }" />
            <span class="rail__marker" :style="{ left: `${b.answer.noul * 100}%` }" />
          </div>
          <div class="rail__scale">
            <span>no</span>
            <span class="rail__value">p(yes) {{ fmtP(b.answer.noul) }}</span>
            <span>yes</span>
          </div>
        </div>

        <div v-else-if="b.answer.type === 'score'" class="scale">
          <div class="scale__track">
            <span
              v-for="l in scaleLabels(b.answer)"
              :key="l.name"
              class="scale__stop"
              :style="{ left: `${l.at * 100}%` }"
            />
            <span class="scale__marker" :style="{ left: `${scorePosition(b.answer) * 100}%` }" />
          </div>
          <div class="scale__labels">
            <span
              v-for="l in scaleLabels(b.answer)"
              :key="l.name"
              class="scale__label"
              :class="{ 'scale__label--on': l.name === b.answer.level, 'scale__label--hide': !l.show }"
              :style="{ left: `${l.at * 100}%` }"
            >
              {{ l.name }}
            </span>
          </div>
          <p class="ac__score">
            score {{ b.answer.score.toFixed(2) }}<template v-if="b.answer.score_stderr !== undefined">
              ± {{ b.answer.score_stderr.toFixed(2) }}</template>
            on 0 to {{ Object.keys(b.answer.legend).length - 1 }}
          </p>
          <ProbBars v-if="b.bars" :bars="b.bars" />
        </div>

        <ProbBars v-else-if="b.bars" :bars="b.bars" />

        <p class="ac__line">
          <template v-if="b.agreement">{{ b.agreement }} · </template>
          <template v-if="b.answer.stderr !== undefined">standard error {{ fmtP(b.answer.stderr) }} · </template>
          <template v-if="Number.isFinite(b.entropy)">{{ b.entropyOf }} entropy {{ b.entropy.toFixed(2) }} ({{ entropyWord(b.entropy) }})</template>
          <template v-else>entropy unknown</template>
          <template v-if="b.answer.outside !== undefined"> · outside {{ fmtP(b.answer.outside) }}</template>
          <template v-if="b.answer.answer_confidence !== undefined"> · calibrated {{ fmtP(b.answer.answer_confidence) }}</template>
        </p>
      </article>
      </template>

      <div class="ac__legend">
        <span class="ac__legend-h">confidence</span>
        <span v-for="(name, i) in CONFIDENCE_BINS" :key="name" class="ac__legend-i">
          <i class="ac__dot" :class="`ac__dot--${i}`" /> {{ name }}
        </span>
      </div>

      <Collapsible class="ac__diag" summary="Diagnostics" :hint="run.model">
        <table v-if="diag?.backend === 'laya'" class="ac__table">
          <tbody>
            <tr>
              <th>checkpoint</th>
              <td>{{ diag.checkpoint ?? '-' }}</td>
              <th>state</th>
              <td class="c-num">{{ diag.state_tokens ?? '-' }} tokens</td>
              <th>read</th>
              <td class="c-num">{{ run.response.usage?.input_tokens ?? '-' }} tokens</td>
              <th>time</th>
              <td class="c-num">{{ Math.round(diag.timing.total_ms) }} ms</td>
            </tr>
            <tr v-if="run.response.routing">
              <th>routed</th>
              <td colspan="7">{{ run.response.routing.reason }}</td>
            </tr>
          </tbody>
        </table>
        <div v-if="diag?.backend === 'laya'" class="ac__scroll">
          <table class="ac__table ac__table--q">
            <thead>
              <tr>
                <th>question</th>
                <th>answer</th>
                <th class="c-num">options</th>
                <th class="c-num">tokens</th>
                <th class="c-num">temperature</th>
                <th class="c-num">entropy confidence</th>
                <th v-if="diag.windowed">window</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in diag.questions" :key="d.id">
                <td class="ac__mono">{{ d.id }}</td>
                <td class="ac__mono">{{ d.label }}</td>
                <td class="c-num">{{ d.options ?? '-' }}</td>
                <td class="c-num">{{ d.tokens?.join(', ') ?? '-' }}</td>
                <td class="c-num">{{ d.temperature?.toFixed(2) ?? '-' }}</td>
                <td class="c-num">{{ fmtP(d.entropy_confidence) }}</td>
                <td v-if="diag.windowed">{{ d.window ? `${d.window.index + 1} of ${d.window.count} (tokens ${d.window.token_start}-${d.window.token_end})` : '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <table v-if="diag?.backend === 'clef'" class="ac__table">
          <tbody>
            <tr>
              <th>state</th>
              <td class="c-num">
                {{ diag.state_tokens ?? '-' }} tokens<template v-if="diag.state_read !== undefined && diag.state_read !== diag.state_tokens">, first {{ diag.state_read }} read</template>
              </td>
              <th>read</th>
              <td class="c-num">{{ run.response.usage?.input_tokens ?? '-' }} tokens</td>
              <th>pass</th>
              <td class="c-num">{{ diag.timing.pass_requests ?? 1 }} {{ (diag.timing.pass_requests ?? 1) === 1 ? 'request' : 'requests' }}</td>
              <th>time</th>
              <td class="c-num">{{ Math.round(diag.timing.total_ms) }} ms</td>
            </tr>
          </tbody>
        </table>
        <div v-if="diag?.backend === 'clef'" class="ac__scroll">
          <table class="ac__table ac__table--q">
            <thead>
              <tr>
                <th>question</th>
                <th>answer</th>
                <th class="c-num">options</th>
                <th class="c-num">entropy</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in diag.questions" :key="d.id">
                <td class="ac__mono">{{ d.id }}</td>
                <td class="ac__mono">{{ d.label }}</td>
                <td class="c-num">{{ d.options ?? '-' }}</td>
                <td class="c-num">{{ Number.isFinite(d.entropy) ? d.entropy.toFixed(3) : '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <table v-if="!diag?.backend" class="ac__table">
          <tbody>
            <tr>
              <th>canvas</th>
              <td class="c-num">{{ run.response.diagnostics.canvas }} positions</td>
              <th>reads</th>
              <td class="c-num">{{ run.response.diagnostics.reads }}</td>
              <th>prompt</th>
              <td class="c-num">{{ run.response.usage?.input_tokens ?? '-' }} tokens</td>
              <th>time</th>
              <td class="c-num">{{ Math.round(run.response.diagnostics.timing.total_ms) }} ms</td>
            </tr>
            <tr v-if="diag && (diag.steps !== undefined || diag.stages)">
              <th>steps</th>
              <td class="c-num">{{ diag.steps ?? 1 }}</td>
              <th>stages</th>
              <td class="c-num">{{ diag.stages?.length ?? 1 }}</td>
              <th>canvases</th>
              <td class="c-num">{{ diag.chunks?.length ?? 1 }}</td>
              <th>layout</th>
              <td>{{ diag.format ?? 'lines' }}<template v-if="diag.conditioning">, earlier answers {{ diag.conditioning === 'prefill' ? 'in the prompt' : 'restated' }}</template></td>
            </tr>
          </tbody>
        </table>
        <div v-if="!diag?.backend" class="ac__scroll">
          <table class="ac__table ac__table--q">
            <thead>
              <tr>
                <th>question</th>
                <th>label</th>
                <th class="c-num">position</th>
                <th class="c-num">entropy</th>
                <th class="c-num">vocabulary entropy</th>
                <th class="c-num">label mass</th>
                <th>top token</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in run.response.diagnostics.questions" :key="d.id">
                <td class="ac__mono">{{ d.id }}</td>
                <td class="ac__mono">{{ d.label }}</td>
                <td class="c-num">{{ d.position }}</td>
                <td class="c-num">{{ Number.isFinite(d.entropy) ? d.entropy.toFixed(3) : '-' }}</td>
                <td class="c-num">{{ d.slot_entropy !== undefined && Number.isFinite(d.slot_entropy) ? d.slot_entropy.toFixed(3) : '-' }}</td>
                <td class="c-num">{{ fmtP(d.label_mass) }}</td>
                <td class="ac__mono">{{ topTokens(d.reads) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <ReadHeatmap v-if="heatRows.length" :rows="heatRows" />
        <pre class="ac__raw">{{ raw }}</pre>
      </Collapsible>
    </template>
  </section>
</template>

<style scoped>
.ac {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.ac__head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.ac__title {
  margin: 0;
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.ac__meta {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.ac__meta--live {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--pk-text-secondary);
}
.ac__stale {
  padding: 1px 8px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
}
.ac__runs {
  display: inline-flex;
  align-items: center;
  gap: 1px;
  margin-left: auto;
}
.ac__runbtn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  border-radius: var(--pk-radius-sm);
  background: none;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.ac__runbtn:hover:not(:disabled) {
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
}
.ac__runbtn:disabled {
  opacity: 0.35;
  cursor: default;
}
.ac__runcount {
  min-width: 30px;
  text-align: center;
  font-family: var(--pk-font-mono);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-secondary);
}
.ac__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 40px 16px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.ac__empty p {
  margin: 0;
}
.ac__q {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 0 14px;
  border-top: 1px solid var(--pk-border-default);
}
.ac__q--skip {
  opacity: 0.75;
}
.ac__pics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ac__pic {
  width: 56px;
  height: 42px;
  object-fit: cover;
  border-radius: var(--pk-radius-sm);
  border: 1px solid var(--pk-border-default);
  background: var(--pk-bg-inset);
}
.ac__thoughttext {
  margin: 0;
  white-space: pre-wrap;
  font-size: var(--pk-font-size-sm);
  line-height: 1.5;
  color: var(--pk-text-secondary);
  max-height: 320px;
  overflow: auto;
}
.ac__qhead {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}
.ac__qid {
  flex: none;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-accent-text);
}
.ac__qtext {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  overflow-wrap: anywhere;
}
.ac__answer {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.ac__label {
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.ac__conf {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
.ac__dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  border: 1px solid var(--pk-border-subtle);
}
.ac__dot--0 {
  background: var(--pk-conf-1);
}
.ac__dot--1 {
  background: var(--pk-conf-2);
}
.ac__dot--2 {
  background: var(--pk-conf-3);
}
.ac__dot--3 {
  background: var(--pk-conf-4);
}
.ac__tie {
  padding: 1px 7px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
}
.ac__line {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.ac__score {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}

/* the yes/no rail: a line from no to yes, the marker at p(yes), two faint
   marks at 0.3 and 0.7 - reference points, not a verdict */
.rail {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 6px 0 2px;
}
.rail__track {
  position: relative;
  height: 6px;
  margin: 0 5px;
  border-radius: 3px;
  background: var(--pk-border-default);
}
.rail__fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  border-radius: 3px;
  background: var(--pk-accent-subtle);
}
.rail__tick {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 1px;
  background: var(--pk-border-strong);
  opacity: 0.6;
}
.rail__marker {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--pk-accent);
  border: 2px solid var(--pk-bg-surface);
  transform: translate(-50%, -50%);
  box-shadow: 0 0 0 1px var(--pk-accent);
}
.rail__scale {
  display: flex;
  justify-content: space-between;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.rail__value {
  font-family: var(--pk-font-mono);
  color: var(--pk-text-secondary);
}

/* the score scale: one stop per level, the marker at the fractional score */
.scale {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: 6px;
}
.scale__track {
  position: relative;
  height: 6px;
  margin: 0 5px;
  border-radius: 3px;
  background: var(--pk-border-default);
}
.scale__stop {
  position: absolute;
  top: 50%;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--pk-border-strong);
  transform: translate(-50%, -50%);
}
.scale__marker {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--pk-accent);
  border: 2px solid var(--pk-bg-surface);
  transform: translate(-50%, -50%);
  box-shadow: 0 0 0 1px var(--pk-accent);
}
.scale__labels {
  position: relative;
  height: 18px;
  margin: 0 5px 4px;
}
.scale__label {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  max-width: 34%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.scale__label:first-child {
  transform: none;
}
.scale__label:last-child {
  transform: translateX(-100%);
}
.scale__label--on {
  color: var(--pk-text-primary);
  font-weight: 600;
}
.scale__label--hide {
  visibility: hidden;
}

.ac__legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 14px;
  padding-top: 10px;
  border-top: 1px solid var(--pk-border-default);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.ac__legend-h {
  color: var(--pk-text-secondary);
}
.ac__legend-i {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.ac__diag :deep(.pk-coll__body) {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-left: 0;
}
/* the question tables can outgrow the card: they scroll, the page never does */
.ac__scroll {
  overflow-x: auto;
}
.ac__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-xs);
}
.ac__table th {
  text-align: left;
  font-weight: 500;
  color: var(--pk-text-muted);
  padding: 4px 10px 4px 0;
  white-space: nowrap;
}
.ac__table td {
  padding: 4px 14px 4px 0;
  color: var(--pk-text-primary);
}
.ac__table--q th,
.ac__table--q td {
  border-bottom: 1px solid var(--pk-border-subtle);
}
.ac__table--q thead th {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.72rem;
}
.ac__table th.c-num,
.ac__table td.c-num {
  text-align: right;
}
.ac__mono {
  font-family: var(--pk-font-mono);
}
.c-num {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.ac__raw {
  margin: 0;
  max-height: 320px;
  padding: 10px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  overflow: auto;
}
.spin {
  animation: pk-spin 0.8s linear infinite;
}
@keyframes pk-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
