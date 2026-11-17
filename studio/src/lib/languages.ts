// The languages a speech model can be TOLD to expect, and how to name one.
//
// The codes are whisper's own set (99) - the list the runner forwards to the
// model - rather than every language that exists, because offering a language
// the model was never trained on is offering a wrong answer. Names come from
// the browser's Intl.DisplayNames so we do not hand-maintain 99 English
// strings that would then be wrong in every other locale.

/** whisper's language codes. Mostly ISO 639-1; the exceptions are whisper's
 *  own spellings and are corrected in NAME_FIX below. */
const WHISPER_CODES = [
  'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca',
  'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr',
  'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it',
  'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv',
  'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no',
  'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn',
  'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr',
  'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'yue', 'zh',
]

/** Codes Intl cannot name, or names unhelpfully. `jw` is whisper's spelling of
 *  Javanese (ISO 639-1 is `jv`); the rest are macrolanguage/regional codes the
 *  browser tables may or may not carry. */
const NAME_FIX: Record<string, string> = {
  jw: 'Javanese',
  haw: 'Hawaiian',
  yue: 'Cantonese',
}

let namer: Intl.DisplayNames | undefined
function displayNames(): Intl.DisplayNames | undefined {
  if (namer) return namer
  try {
    namer = new Intl.DisplayNames(undefined, { type: 'language' })
  } catch {
    // No Intl language table (very old engines): codes stand in for names,
    // which is unlovely but never wrong.
  }
  return namer
}

/** English name -> ISO code, built once. Deliberately pinned to `en` rather
 *  than the viewer's locale: this reverses what a MODEL emitted, and the
 *  models emit English names. Display still uses the viewer's locale, so a
 *  Swedish user reads "svenska" for the same answer. */
let byEnglishName: Map<string, string> | undefined
function englishIndex(): Map<string, string> {
  if (byEnglishName) return byEnglishName
  const m = new Map<string, string>()
  try {
    const en = new Intl.DisplayNames(['en'], { type: 'language' })
    for (const c of WHISPER_CODES) {
      const n = NAME_FIX[c] ?? en.of(c)
      if (n && n.toLowerCase() !== c) m.set(n.toLowerCase(), c)
    }
  } catch {
    // no English table - the map stays empty and a name passes through as text
  }
  for (const [c, n] of Object.entries(NAME_FIX)) m.set(n.toLowerCase(), c)
  byEnglishName = m
  return m
}

/** What a transcription said its language was, as a name a person reads.
 *
 *  Takes either value space, because the runner's lanes disagree: whisper
 *  reports the ISO code ("sv") while Qwen3-ASR reports an English name in
 *  whatever case the model emitted it ("swedish") - its output envelope is
 *  `language {X}<asr_text>` and X is passed through verbatim. Comparing the
 * two side by side showed "Swedish" against "swedish" ,
 *  which reads as two different answers to the same question. Normalising
 *  here is right regardless of what the wire settles on. */
export function languageName(value: string | undefined): string {
  const raw = value?.trim()
  if (!raw) return ''
  const fix = NAME_FIX[raw]
  if (fix) return fix
  try {
    const hit = displayNames()?.of(raw)
    // Intl echoes the input back when it has no entry - a miss, not a name
    if (hit && hit.toLowerCase() !== raw.toLowerCase()) return hit
  } catch {
    // not a structurally valid tag (a name with a space, say) - try it as one
  }
  const code = englishIndex().get(raw.toLowerCase())
  if (code) return NAME_FIX[code] ?? displayNames()?.of(code) ?? raw
  // Unknown either way: a code SHOUTS, a word is merely capitalised - never
  // invent a language we cannot name.
  return raw.length <= 3 ? raw.toUpperCase() : raw[0].toUpperCase() + raw.slice(1)
}

export interface LanguageOption {
  value: string
  label: string
}

/** The language to START on: the one this browser is set to, when a speech
 *  model can be told about it.
 *
 *  Better than opening on "detect automatically", and the reason is what
 *  detection actually costs rather than a preference for defaults. A model
 *  guesses the language from a few seconds of audio and then transcribes as
 *  that language, so one wrong guess does not cost you a label - it costs you
 *  the whole transcript. Measured: Qwen3-ASR heard Swedish, decided
 *  Dutch, and wrote Swedish sounds in Dutch spelling. The browser already
 *  knows what its owner speaks; asking a 1.7B model to work it out from five
 *  seconds of audio is throwing that away.
 *
 *  Region is dropped (`sv-SE` -> `sv`) because whisper's set has no regional
 *  variants, and a locale we do not serve falls back to detection rather than
 *  to something close-ish - `nn` is not `no`, and guessing on the user's
 *  behalf is the thing this exists to avoid. */
export function localeLanguage(codes: readonly string[] = WHISPER_CODES): string | undefined {
  const tags = navigator.languages?.length ? navigator.languages : [navigator.language]
  for (const tag of tags) {
    const code = tag?.split('-')[0]?.toLowerCase()
    if (code && codes.includes(code)) return code
  }
  return undefined
}

/** "let the model decide". A SENTINEL, not the empty string: reka-ui's
 *  SelectItem THROWS on an empty value (it reserves '' for clearing the
 *  selection), and no ISO code is 'auto', so this cannot collide.
 *
 *  STORED as itself, and that is the change here: absent used to mean
 *  auto, and now means "never chosen", which is what lets an untouched chat
 *  open on the browser's own language while somebody who actually wants
 *  detection still gets it. Three states, one of them explicit. */
export const LANGUAGE_AUTO = 'auto'

/** The code that rides on the wire for a stored setting - the one place the
 *  three states collapse into the two the API has. */
export function askedLanguage(
  stored: string | undefined,
  codes: readonly string[] = WHISPER_CODES,
): string | undefined {
  if (stored === LANGUAGE_AUTO) return undefined
  // a choice this lane cannot take (picked for another model) is detection
  // here, not a refusal from the runner
  if (stored && !codes.includes(stored)) return undefined
  return stored || localeLanguage(codes)
}

/** Auto first (it is the default and the right answer most of the time), then
 *  every language by name - whisper's set, or the lane's own where its
 *  checkpoint declares one (`codes`, from the endpoint's caps). Built once
 *  per set. */
const cached = new Map<string, LanguageOption[]>()
export function languageOptions(codes: readonly string[] = WHISPER_CODES): LanguageOption[] {
  const key = codes.join(',')
  const hit = cached.get(key)
  if (hit) return hit
  const rest = codes.map((value) => ({ value, label: languageName(value) })).sort((a, b) =>
    a.label.localeCompare(b.label),
  )
  const out = [{ value: LANGUAGE_AUTO, label: 'Detect automatically' }, ...rest]
  cached.set(key, out)
  return out
}
