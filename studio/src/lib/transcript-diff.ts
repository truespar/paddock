// Where compare lanes DISAGREE about what was said.
//
// The Studio's audio compare exists to answer "which of these models should I
// run on my audio", and each model's own confidence is the weak signal for
// that: the scales are per-model, so counting marks across lanes measures
// their logprob distributions rather than their accuracy. Two independent
// models AGREEING on a word is strong evidence it is right; disagreeing means
// one of them is wrong and a listener settles it in seconds.
//
// This is the ensemble/agreement method from the no-training tier of the
// confidence literature, and unlike a confidence score it is SYMMETRIC - it
// makes no claim about which lane is correct, only about where they differ, so
// it is fair between models by construction. It also costs nothing: both
// transcripts are already in the same conversation turn.
//
// It works on TEXT, deliberately. A lane that reported no per-word confidence
// (a generative ASR model answering without include=logprobs) still has words,
// and still belongs in the comparison.

import type { TranscriptMeta, TranscriptSegment, TranscriptWord } from '@/types/chat'

/** One word as the transcript renders it. The diff indexes this list, and so
 *  does the renderer, so a marked index cannot land on a different word than
 *  the one that was compared - the failure mode of computing the two
 *  separately. */
export interface RenderWord {
  speakers?: number[]
  word: string
  /** the model's own probability, where the lane reported one */
  confidence?: number
  /** what the model nearly said instead - the road not taken */
  alt?: string
  /** top1 - top2 in probability space, which is what says whether "nearly" is
   *  a fair word for it */
  margin?: number
  /** which segment it belongs to (-1 when the lane has no times), for the
   *  playing highlight and the seek */
  segment: number
  start?: number
  /** Set only on the flat-words path, where a word owns its own clock. The
   *  karaoke highlight keys on it: a word with start AND end can be "the one
   *  being spoken", while segment-granularity words (whole sentence sharing
   *  one start) deliberately never can - lighting the last word of a sentence
   *  for the sentence's whole duration would be a lie about the timing. */
  end?: number
}

/** Which segment a time falls in, or -1 when nothing covers it. Linear because
 *  a transcript's segments are tens, not thousands, and the words walk them in
 *  order anyway. */
function segmentAt(segments: TranscriptSegment[] | undefined, t: number): number {
  if (!segments?.length) return -1
  // the last segment that has started, which is how the playing highlight
  // already picks one - same rule, so a word and the highlight cannot land in
  // different segments
  let idx = -1
  segments.forEach((s, i) => {
    if (t + 0.001 >= s.start) idx = i
  })
  return idx
}

/** The words of a transcript in render order, from whichever shape the lane
 *  answered in: a flat word list (whisper with `word` granularity, or the
 *  generative lanes with include=logprobs), per-segment words, or nothing but
 *  text. All of them end up here so every lane can be compared - a model that
 *  declines to say how sure it was still said WORDS.
 *
 *  The flat list is tried first because when both arrive it is the finer of
 *  the two: it is the one whose entries carry their own times, and a word that
 *  knows when it was said should seek there rather than to its sentence. */
export function renderWords(
  segments: TranscriptSegment[] | undefined,
  words: TranscriptWord[] | undefined,
  plain: string,
): RenderWord[] {
  const out: RenderWord[] = []
  if (words?.length) {
    return words.map((w) => ({
      speakers: w.speakers,
      word: w.word,
      confidence: w.confidence,
      alt: w.alt,
      margin: w.margin,
      // A lane with no times has no segments either, so this is -1 there and
      // the word clicks nowhere - deliberately, since a control that moves the
      // playhead to 0 is worse than no control.
      segment: w.start === undefined ? -1 : segmentAt(segments, w.start),
      start: w.start,
      end: w.end,
    }))
  }
  if (segments?.length) {
    segments.forEach((s, i) => {
      if (s.words?.length) {
        for (const w of s.words) {
          out.push({
            word: w.word,
            confidence: w.confidence,
            alt: w.alt,
            margin: w.margin,
            segment: i,
            start: s.start,
          })
        }
      } else {
        // no per-word confidence on this segment: split its text, so the
        // comparison still has words even where the scores are missing
        for (const w of s.text.split(/\s+/).filter(Boolean)) {
          out.push({ word: w, segment: i, start: s.start })
        }
      }
    })
    return out
  }
  return plain
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => ({ word: w, segment: -1 }))
}

/** The same list for a message that may not be a transcription at all. */
export function transcriptWords(meta: TranscriptMeta | undefined, text: string): RenderWord[] {
  if (!meta) return []
  return renderWords(meta.segments, meta.words, text)
}

/** Fold away the differences that are not disagreements. Two models writing
 *  "Hej," and "Hej" said the same thing; flagging that would bury the real
 *  ones. Same spirit as the WER path's normaliser - casefold, drop punctuation
 *  and symbols - kept here so the Studio has no dependency on it. */
function fold(word: string): string {
  return [...word.toLowerCase().normalize('NFKC')]
    .filter((ch) => !/[\p{P}\p{S}]/u.test(ch))
    .join('')
}

/** Levenshtein with a backtrace over two word lists: the indices on each side
 *  that are not a match. A substitution marks both sides (they disagree about
 *  the same slot); an insertion marks only the side that has the extra word.
 *
 *  The distance alone cannot say which words differ, and that is the whole
 *  output here. */
function diffPair(a: string[], b: string[]): [Set<number>, Set<number>] {
  const fa = a.map(fold)
  const fb = b.map(fold)
  const n = fa.length
  const m = fb.length
  // d[i][j] = edits between a[..i] and b[..j]
  const d: number[][] = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
  )
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = fa[i - 1] === fb[j - 1] ? 0 : 1
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    }
  }
  const ia = new Set<number>()
  const ib = new Set<number>()
  let i = n
  let j = m
  while (i > 0 && j > 0) {
    const cost = fa[i - 1] === fb[j - 1] ? 0 : 1
    if (d[i][j] === d[i - 1][j - 1] + cost) {
      if (cost) {
        ia.add(i - 1)
        ib.add(j - 1)
      }
      i--
      j--
    } else if (d[i][j] === d[i][j - 1] + 1) {
      ib.add(--j) // b has a word a never had
    } else {
      ia.add(--i)
    }
  }
  while (i > 0) ia.add(--i)
  while (j > 0) ib.add(--j)
  return [ia, ib]
}

/** Per lane, the indices of words at least one other lane disagreed with.
 *
 *  Pairwise union rather than a true multiple alignment: with two lanes (the
 *  common case) it is the pairwise diff, and with more it reads as "not
 *  unanimous", which is a rule that can be explained in one sentence. A real
 *  N-way alignment would be defensible and is not worth its complexity for a
 *  panel capped at four.
 *
 *  Lanes with no words at all (an errored or empty turn) sit the comparison
 *  out rather than making every other lane look wrong. */
export function disagreements(lanes: string[][]): Set<number>[] {
  const out = lanes.map(() => new Set<number>())
  const live = lanes.map((w, i) => [i, w] as const).filter(([, w]) => w.length > 0)
  if (live.length < 2) return out
  for (let x = 0; x < live.length; x++) {
    for (let y = x + 1; y < live.length; y++) {
      const [i, wi] = live[x]
      const [j, wj] = live[y]
      const [di, dj] = diffPair(wi, wj)
      di.forEach((k) => out[i].add(k))
      dj.forEach((k) => out[j].add(k))
    }
  }
  return out
}
