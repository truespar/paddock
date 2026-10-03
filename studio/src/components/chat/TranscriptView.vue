<script setup lang="ts">
// The transcript, marking the words a speech model was least sure of
//
// One MARK, not A SCALE, and behind a toggle - reworked after a
// research pass we asked for, which went against the first version:
//
//   * The number is not a probability of correctness. It is exp of the mean
//     logprob - raw softmax - and raw softmax on an end-to-end ASR model is
//     known to be overconfident (arXiv 2509.07195: these models "easily
//     memorise training sequences, which results in overestimated confidence
//     scores"; confidence-estimation modules and temperature scaling exist
//     because of it). So nothing here prints a percentage as if it meant "one
//     in two chance this is wrong". The raw value rides the tooltip, labelled
//     as the model's own, until fits a real P(correct).
//   * Two states, not three. The industry ships one: Rev's "show low
//     confidence words" is a single grey highlight behind a toggle, and
//     AssemblyAI's guide picks one threshold (0.4, "0.5 or less is a good
//     start") and says outright it is the caller's choice. A published
//     two-cut, three-band scheme claims a resolution the signal has not got.
//   * A TOGGLE, because the evidence on whether this helps at all is
//     discouraging: CHI 2025 (36 participants) found confidence-based error
//     detection "neither improved correction efficiency nor was perceived as
//     helpful", with classifiers that "frequently miss errors or generate many
//     false positives". Marks earn their place when a reader wants them.
//
// How the mark is drawn:
//
//   * `text-decoration: underline wavy`, not a fake bottom border. The first
//     version drew one with `box-shadow: inset 0 -2px 0`, which on an INLINE
//     element does not follow the text baseline and breaks at every line wrap
//     - that stray bar was the artifact. text-decoration flows across wraps
//     and sits on the baseline by construction.
//   * Wavy-underline-means-check-this is the spellcheck convention every
//     reader already knows, which is exactly the question a transcript raises.
//   * Shape carries it as well as hue, so it survives colour-blindness and
//     greyscale - WCAG 1.4.1 forbids colour as the only channel, and a
//     decoration STYLE is a real second one where a second colour is not.
//
// The PLAYING segment keeps a background instead. That is deliberate: two
// meanings need two channels, so background says "where the audio is" and
// underline says "check this word", and they never fight.
//
// The GRANULARITY of the times shows itself, rather than being explained. A
// whisper lane asked for word times gives every word its own
// start, so a click seeks to the word; a lane with segment times only gives
// every word of a sentence the same one, so a click seeks to the sentence.
// Either way the tooltip prints the time the click will actually go to, and
// the background still marks the SEGMENT the audio is in - which is a
// different question from where a click lands, and keeps its own channel.
//
// The segments are the PERSISTED ones off the message, not a live
// response body - so a transcript re-read months later marks exactly as it did
// the day it was made.
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { TranscriptSegment, TranscriptWord } from '@/types/chat'
import { clock } from '@/lib/subtitles'
import { renderWords, type RenderWord } from '@/lib/transcript-diff'
import { useSettingsStore } from '@/stores/settings'
import Tooltip from '@/components/ui/Tooltip.vue'

const settings = useSettingsStore()

const props = withDefaults(
  defineProps<{
    segments: TranscriptSegment[]
    /** The clip's words, flat. Two lanes send this and the difference is
     *  whether they carry times: whisper asked for `word` granularity does
     *   and each word is then its own seek target; the generative
     *  ASR lanes have logprobs and no timestamp vocabulary, so
     *  their words are deliberately not clickable - there is nothing to seek
     *  to, and a control that moves the playhead to 0 is worse than none. */
    words?: TranscriptWord[]
    /** Indices (into the flat render order) another lane heard differently.
     *  Empty outside a compare - a single transcript has nothing to disagree
     *  with. This is the STRONGER signal of the two the view carries: two
     *  independent models, not one model's opinion of itself. */
    differs?: Set<number>
    time?: number
    plain?: string | null
  }>(),
  { words: () => [], differs: () => new Set<number>(), time: 0, plain: null },
)
const emit = defineEmits<{ (e: 'seek', t: number): void }>()

/** The one cut. Measured, and still only a heuristic - kb-whisper-large over
 *  the Nordic battery, on clips it transcribes 7/7 exactly:
 *
 *      clip                words     p5    p10    p25  median    min
 *      sv-short-1             17   0.45   0.51   0.61    0.75   0.45
 *      sv-med-1               21   0.58   0.60   0.64    0.83   0.43
 *      sv-concat              71   0.55   0.58   0.65    0.82   0.41
 *      ls-long (wrong lang)  109   0.27   0.33   0.44    0.63   0.21
 *
 *  A PERFECT transcript's median word sits near 0.8 and its floor near 0.42,
 *  so the obvious-looking 0.9 line would mark four words in five - a warning
 *  that fires on everything says nothing. 0.45 keeps a correct transcript
 *  nearly clean while the clip the model got wrong goes visibly marked, and it
 *  lands where AssemblyAI's guide starts (0.4-0.5).
 *
 *  KNOWN LIMIT, and the reason this exists: this cut is kb-whisper's scale.
 *  Another model's logprobs sit elsewhere, so comparing two lanes by how much
 *  each is marked is not sound today - a model can look unsure purely because
 *  its distribution is shifted. Calibrating to P(correct) per checkpoint makes
 *  one cut mean one thing everywhere; until then this marks within a lane. */
const UNSURE_BELOW = 0.45
/** Marked = below the cut AND the reader wants marks. One predicate, so the
 *  toggle cannot get out of step with the tooltips. */
function marked(c: number): boolean {
  return settings.markUnsure && c < UNSURE_BELOW
}
function pct(c: number): string {
  return `${Math.round(c * 100)}%`
}
/** How close the runner-up has to be before "nearly" is a fair word for it.
 *  A marked word usually has a second candidate; that alone does not mean the
 *  model was torn. Measured on one kb-whisper clip, most contested steps run a
 *  0.6-0.8 margin - a clear winner with a distant rival - and calling that
 *  "nearly said" would overstate what happened. Under 0.2 the two were really
 *  competing, which is the case worth a reader's attention. */
const NEARLY_MARGIN = 0.2

/** One list of words for all three answer shapes, shared with the diff so a
 *  marked index cannot land on a different word than the one compared. */
const items = computed(() => renderWords(props.segments, props.words, props.plain ?? ''))

/** Grouped back into segments for the playing highlight and the seek. A lane
 *  with no times is one group that clicks nowhere. */
const groups = computed(() => {
  const out: { segment: number; start?: number; speakers?: number[]; words: (RenderWord & { at: number })[] }[] = []
  items.value.forEach((w, at) => {
    const last = out[out.length - 1]
    if (last && last.segment === w.segment && JSON.stringify(last.speakers) === JSON.stringify(w.speakers)) last.words.push({ ...w, at })
    else out.push({ segment: w.segment, start: w.start, speakers: w.speakers, words: [{ ...w, at }] })
  })
  return out
})

/** Last index in `list` whose start has begun by t, or -1. The rAF playhead
 *  runs this every frame in every lane, which is what retired the old linear
 *  walk - same "the last segment that has started" rule as segmentAt in
 *  transcript-diff, just found by bisection over the sorted starts. */
function lastStarted(list: { start: number }[], t: number): number {
  let lo = 0
  let hi = list.length - 1
  let out = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (list[mid].start <= t) {
      out = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return out
}

const now = computed(() => lastStarted(props.segments, props.time + 0.001))

/** The words that own their clock, in render order (which is time order for
 *  them: they are the flat whisper words, and DTW emits them monotonically).
 *  Segment-granularity words carry no `end` and sit this out - see
 *  RenderWord.end for why lighting them would misstate the timing. */
const timed = computed(() => {
  const out: { at: number; start: number; end: number }[] = []
  items.value.forEach((w, at) => {
    if (w.start !== undefined && w.end !== undefined) out.push({ at, start: w.start, end: w.end })
  })
  return out
})
/** items-index of the word being spoken right now, -1 in silence. The word's
 *  own end decides, unlike the sentence tint: the sentence says where the
 *  audio is (so it stays lit through a pause), a word that has finished is
 *  over, and between words nothing is lit - honest about the gap. */
const nowWord = computed(() => {
  const t = props.time + 0.001
  const list = timed.value
  const i = lastStarted(list, t)
  return i >= 0 && t < list[i].end ? list[i].at : -1
})

/** Follow the audio: the active segment keeps itself in view until something
 *  says the reader went elsewhere. A wheel or touch anywhere is that signal -
 *  a page must never yank a reader back mid-read, and a programmatic smooth
 *  scroll fires neither event so the follow cannot hold itself. Clicking any
 *  word (or walking the diff) is the explicit way back in. easytranscriber
 *  scrolls unconditionally; the hold is the refinement it lacked. */
const held = ref(false)
function hold(): void {
  held.value = true
}
onMounted(() => {
  window.addEventListener('wheel', hold, { passive: true })
  window.addEventListener('touchmove', hold, { passive: true })
})
onBeforeUnmount(() => {
  window.removeEventListener('wheel', hold)
  window.removeEventListener('touchmove', hold)
})
watch(now, (seg) => {
  if (held.value || seg < 0) return
  root.value
    ?.querySelector(`[data-seg="${seg}"]`)
    ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
})
/** Whether there is anything to mark - Not whether the lane reports confidence
 *  at all. A transcript the model was sure of end to end has nothing to show,
 *  and a toggle whose two states look identical is worse than no toggle: the
 *  reader flips it, sees no change, and stops trusting the control.
 *
 *  Deliberately reads the raw cut rather than `marked()`: with marking turned
 *  off, `marked()` is false everywhere, which would hide the only control that
 *  turns it back on. */
const anyUnsure = computed(() =>
  items.value.some((w) => w.confidence !== undefined && w.confidence < UNSURE_BELOW),
)
const anyDiff = computed(() => props.differs.size > 0)

/** The disagreements in reading order, and where the walk currently stands
 *  A compare exists to answer "where do these two differ", and
 *  before this the answer was "read both columns and hunt for the marks". */
const diffAt = computed(() => [...props.differs].sort((a, b) => a - b))
const cursor = ref(-1)
const root = ref<HTMLElement | null>(null)

/** Walk to the next disagreement, wrapping at the end.
 *
 *  Seeks where this lane knows a time and scrolls in every case: a lane with
 *  no timestamp vocabulary at all (the generative families) still has WORDS,
 *  and "show me the next place they differ" is most of the value even with no
 *  audio to move. Seeking is the part that needs times, not walking. */
function nextDiff(): void {
  const list = diffAt.value
  if (!list.length) return
  held.value = false
  const at = list.find((i) => i > cursor.value) ?? list[0]
  cursor.value = at
  const w = items.value[at]
  if (w?.start !== undefined) emit('seek', w.start)
  void nextTick(() => {
    root.value
      ?.querySelector(`[data-at="${at}"]`)
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  })
}
// A re-run changes what the words even are, so a cursor into the old list
// would walk to an unrelated position.
watch(diffAt, () => (cursor.value = -1))

/** What to say about one word, in priority order: a disagreement is the
 *  stronger signal (two models, not one model's opinion of itself), so it
 *  leads when both apply. */
function label(w: RenderWord & { at: number }): string | undefined {
  const bits: string[] = []
  if ((w.speakers?.length ?? 0) > 1) bits.push("multiple speakers cover this word's time range; its speaker is ambiguous")
  if (props.differs.has(w.at)) bits.push('the other model heard this differently')
  if (w.confidence !== undefined && marked(w.confidence)) {
    // The ALTERNATIVE where the lane could give one AND the margin says it was
    // genuinely close: "nearly said vill" is something a reader judges at a
    // glance, where "38%" was a number they could do nothing with. But it has
    // to be earned - a distant runner-up dressed as "nearly" is worse than the
    // percentage, because it reads as a fact about the model's doubt.
    const close = w.alt !== undefined && w.margin !== undefined && w.margin < NEARLY_MARGIN
    bits.push(
      close ? `this model nearly said "${w.alt}"` : `this model scored it ${pct(w.confidence)}`,
    )
  }
  if (w.start !== undefined) bits.push(`click to play from ${clock(w.start)}`)
  return bits.length ? bits.join(' - ') : undefined
}

/** A word click is both a seek and the reader saying "I am here" - it releases
 *  the follow-scroll a manual scroll held. */
function onWord(w: RenderWord & { at: number }): void {
  if (w.start === undefined) return
  held.value = false
  emit('seek', w.start)
}
</script>

<template>
  <div ref="root" class="tv">
    <div v-if="anyUnsure || anyDiff" class="tv__key">
      <!-- Disagreement leads when there is one: it is two models, not one
           model's opinion of itself, and it is the question a compare asks. -->
      <Tooltip
        v-if="anyDiff"
        label="Where the models transcribed the same audio differently. One of them is wrong here - click to step through them, and your ear decides. Where they agree, they are very probably both right."
      >
        <button type="button" class="tv__toggle" @click="nextDiff">
          <span class="tv__w tv__w--differs">Heard differently</span>
          <span class="tv__toggle-state">{{ diffAt.length }}</span>
        </button>
      </Tooltip>
      <Tooltip
        v-if="anyUnsure"
        label="Marks the words this model scored lowest. The score is the model's own probability, not a measured error rate - a marked word is often right, and an unmarked one can be wrong. Not comparable between models."
      >
        <button
          type="button"
          class="tv__toggle"
          :aria-pressed="settings.markUnsure"
          @click="settings.markUnsure = !settings.markUnsure"
        >
          <span class="tv__w tv__w--low">Unsure words</span>
          <span class="tv__toggle-state">{{ settings.markUnsure ? 'marked' : 'not marked' }}</span>
        </button>
      </Tooltip>
    </div>

    <p v-if="!items.length" class="tv__flow">{{ plain }}</p>
    <p v-else class="tv__flow">
      <span
        v-for="(g, gi) in groups"
        :key="gi"
        class="tv__seg"
        :data-seg="g.segment"
        :class="{ 'tv__seg--now': g.segment >= 0 && g.segment === now }"
      >
        <span v-if="g.speakers !== undefined && (gi === 0 || JSON.stringify(g.speakers) !== JSON.stringify(groups[gi - 1].speakers))" class="tv__speaker">
          {{ g.speakers.length ? g.speakers.map(s => `Speaker ${s + 1}`).join(' / ') : 'Unassigned' }}
        </span>
        <template v-for="w in g.words" :key="w.at"
          ><Tooltip :label="label(w)"
            ><component
              :is="w.start === undefined ? 'span' : 'button'"
              :type="w.start === undefined ? undefined : 'button'"
              class="tv__w"
              :data-at="w.at"
              :class="{
                'tv__w--low': w.confidence !== undefined && marked(w.confidence),
                'tv__w--differs': differs.has(w.at),
                'tv__w--walked': w.at === cursor,
                'tv__w--now': w.at === nowWord,
              }"
              @click="onWord(w)"
            >{{ w.word }}</component></Tooltip
          >{{ ' ' }}</template
        >
      </span>
    </p>
  </div>
</template>

<style scoped>
.tv__speaker { display: block; margin-top: 10px; font-size: var(--pk-font-size-xs); font-weight: 600; color: var(--pk-text-muted); }
.tv__key {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 10px;
  font-size: var(--pk-font-size-xs);
}
.tv__key-lead {
  color: var(--pk-text-muted);
}
.tv__key-item {
  font-size: var(--pk-font-size-xs);
}
.tv__flow {
  margin: 0;
  line-height: 1.9;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
/* BACKGROUND = where the audio is. The one meaning this channel carries, so
   it never competes with the confidence marks below. */
.tv__seg {
  border-radius: var(--pk-radius-sm);
  transition: background-color 0.15s;
}
.tv__seg:hover {
  background: var(--pk-bg-hover);
}
.tv__seg--now {
  background: var(--pk-accent-subtle);
}
.tv__w {
  display: inline;
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  /* the text stays at full contrast in every band: a word the model is unsure
     of is the one you most need to read */
  color: var(--pk-text-primary);
  cursor: pointer;
  /* soft enough that the karaoke walk reads as motion, not blinking */
  transition: background-color 0.12s;
}
/* UNDERLINE = how sure the model was, by SHAPE first and colour second, so it
   still reads without colour (WCAG 1.4.1) and in greyscale. text-decoration,
   not a border or box-shadow: only this follows the baseline and survives a
   line wrap on an inline element. */
/* DISAGREEMENT: a solid block, because it is a claim about the audio, not a
   hedge about a word. It has to out-rank the confidence underline when both
   land on one word - two models differing is evidence, one model's own score
   is an opinion. */
.tv__w--differs {
  background: var(--pk-status-warning-subtle);
  box-shadow: 0 0 0 1px var(--pk-status-warning);
  border-radius: var(--pk-radius-sm);
  padding: 0 1px;
}
/* where the walk currently stands: a ring rather than another fill, so it
   reads as "you are here" on top of whatever mark the word already carries
   instead of competing with it for the same channel */
.tv__w--walked {
  box-shadow:
    0 0 0 1px var(--pk-status-warning),
    0 0 0 4px var(--pk-accent-subtle);
}
.tv__w--low {
  text-decoration: underline wavy var(--pk-status-error);
  text-decoration-thickness: 1.5px;
  text-underline-offset: 3px;
  /* the mark is the underline; the tint is what makes it findable while
     scanning, and one band can afford it where three could not */
  background: var(--pk-status-error-subtle);
  border-radius: var(--pk-radius-sm);
  padding: 0 1px;
}
/* KARAOKE: the word being spoken right now, a step darker than the sentence
   wash - the sentence says where the audio is, the word says where the voice
   is. Defined AFTER the meaning marks so its tint wins the background while
   their underline and ring still show through; no padding, so the text never
   shifts as the highlight walks. */
.tv__w--now {
  background: color-mix(in srgb, var(--pk-accent) 30%, transparent);
  border-radius: var(--pk-radius-sm);
}
/* the key doubles as the toggle - the one control this view has */
.tv__toggle {
  display: inline-flex;
  align-items: baseline;
  gap: 8px;
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  cursor: pointer;
}
.tv__toggle-state {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.tv__toggle[aria-pressed='false'] .tv__w--low {
  text-decoration: none;
  background: none;
  color: var(--pk-text-muted);
}
button.tv__w:hover {
  background: var(--pk-bg-hover);
  border-radius: var(--pk-radius-sm);
}
</style>
