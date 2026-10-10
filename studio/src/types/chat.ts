// Chat data model. Content is structured parts (OpenAI-native) so multimodal
// drops in cleanly; assistant reasoning ("thinking") is a separate field from
// content, and attachments are referenced by id (bytes live in IndexedDB).

import type { ImageDetail } from '@/lib/vision'

export type Role = 'system' | 'user' | 'assistant'

export interface TextPart {
  type: 'text'
  text: string
}

export interface ImagePart {
  type: 'image'
  /** id of the ORIGINAL bytes in the attachments table (''=none). The send
   *  path fetches these so the server reads EXIF metadata/orientation and
   *  does budget fitting off the true file - canvas re-encodes strip all of
   *  it (the EXIF bug). Older messages hold a re-encoded "view"
   *  copy under the same field; sending it is still correct, just metadata-
   *  less, exactly as those messages always were. */
  attachmentId: string
  mime: string
  name: string
  /** ~128px inline thumbnail (data URL) - renders the chat bubble instantly. */
  thumbUrl?: string
  /** No model can be shown this one: it is a format nothing here decodes -
   *  today only HEIC, whose HEVC has no decoder we are allowed to embed
   *  The part is still STORED and still opens in the document
   *  panel with its metadata, map and original bytes; it is only the pixels
   *  that never reach a model, and the send path drops it rather than letting
   *  one unreadable photo fail the whole message. */
  unreadable?: boolean
  /** model-budget copy (data URL) - inline FALLBACK when the attachments
   *  store was unavailable at staging; otherwise absent on new parts. */
  modelUrl?: string
  /** How much of the model's resolution this image was sent at (OpenAI's
   *  `detail`). Absent on older messages, which read as 'auto'. */
  detail?: ImageDetail
  /** LEGACY (pre-attachments-v2): the single downscaled data URL. Still read on
   *  render/send so old conversations keep working. */
  dataUrl?: string
  /** original (pre-downscale) dimensions. */
  width?: number
  height?: number
  /** Multi-page images (TIFF): which pages of this file reach the model
   *  ("2-4" / "3" / "2-") - sent as the part-level `pages` field. */
  pageRange?: string
  /** Present when a MODEL made this picture (an image-generation turn's
   *  answer): what it took to make it, which is what makes it reproducible.
   *  Absent on every attachment a person added. */
  gen?: GeneratedImage
}

/** One generated picture's record - the facts that reproduce it byte for
 *  byte on the same model and box (seed, size, steps), plus what it was
 *  asked to look like. Lives on the ImagePart so the picture and its recipe
 *  never part company. */
export interface GeneratedImage {
  seed: number
  /** `WIDTHxHEIGHT` as rendered. */
  size: string
  steps: number
  quality: string
  format: string
  /** `opaque` | `transparent` - what the model produced, never `auto`. */
  background: string
  /** Set while a PREVIEW stands in for the final picture: the render is
   *  still going and `dataUrl` holds the latest partial. Cleared, and the
   *  stored bytes take over, when the final one lands. */
  preview?: boolean
  /** 0-based index of the preview on show, while previewing. */
  previewIndex?: number
}

/** How pictures are asked for in this chat - the composer's controls in
 *  image mode, sticky per conversation the way sampling is, and snapshotted
 *  onto each turn's `imageGen` so Run details says what actually rode. */
export interface ImageParams {
  /** `WIDTHxHEIGHT` on the model's grid, or `auto` for its default. */
  size: string
  quality: 'auto' | 'low' | 'medium' | 'high'
  /** an explicit step count; null = the quality's own. */
  steps: number | null
  /** `thread` automatically keeps the seed for text-only variations, but
   *  reference edits and retries draw afresh. The reference preserves the
   *  composition; reusing its generating noise can corrupt an edit.
   *  `random` draws on every send; a number pins one outright. */
  seed: number | 'thread' | 'random'
  /** pictures per turn (each its own draw on the seed). */
  n: number
  format: 'png' | 'webp' | 'jpeg'
  background: 'auto' | 'opaque' | 'transparent'
  /** progressive previews while a picture renders (0 = wait for the whole
   *  thing). Costs one decode each; three on a 40-step render is ~3 %. */
  previews: number
}

/** The record of an image-generation turn: the prompt as sent, the
 *  parameters that rode, and what the render cost. Beside `run` (the
 *  provenance every turn has) rather than inside it, because nothing about
 *  sampling applies here. */
export interface ImageGenMeta {
  prompt: string
  params: ImageParams
  /** the seed actually used - the pinned one, or the draw `random` made */
  seed: number
  steps: number
  size: string
  /** previews that arrived before the final picture */
  previews: number
  /** an EDIT: how many reference pictures rode, and where they came from -
   *  the person's attachments on the turn, or the thread's last picture
   *  (a follow-up on a picture is an edit of it) */
  references?: number
  referencesFrom?: 'attached' | 'previous'
  /** seconds the engine spent on the turn, all images */
  elapsedS?: number
  /** seconds per denoising step, all-in (prefix + steps + decode) */
  sPerStep?: number
}

/** A document attachment (PDF today). Bytes live in the attachments table by
 *  id and are fetched at send time (never inline in the conversation doc - the
 *  server rasterizes the PDF to page images). */
export interface FilePart {
  type: 'file'
  /** id of the stored bytes in the attachments table ('' = none). */
  attachmentId: string
  mime: string
  name: string
  /** byte size of the original file. */
  size?: number
  /** page count (client-side, from lector/pdfium) - for the chip + affordance. */
  pages?: number
  /** PDF: force the extracted-text route for this file (part-level
   *  `pdf_mode: "text"`). Absent = the server's auto route. */
  pdfMode?: 'text'
  /** which pages of this file reach the model ("2-4" / "3" / "2-") - sent as
   *  the part-level `pages` field. Absent = all. */
  pageRange?: string
}

/** An audio attachment - the USER half of a transcription turn. Bytes live in
 *  the attachments table by id and are fetched at send time, exactly like a
 *  PDF; nothing audio-shaped is ever inlined in the conversation doc.
 *
 *  A transcription is a conversation: audio, transcript out,
 *  one turn. That is why this is a ContentPart and not a separate object with
 *  its own table - compare lanes, persistence and token accounting all come
 *  from the machinery messages already have. */
export interface AudioPart {
  type: 'audio'
  /** id of the stored bytes in the attachments table ('' = none). */
  attachmentId: string
  mime: string
  /** File name, or '' for a microphone recording - which is exactly why the
   *  sidebar title prefers the transcript and only falls back to this. */
  name: string
  /** byte size of the original file. */
  size?: number
  /** clip length in seconds, read client-side for the chip + the player's
   *  scrub bar before any model has answered. */
  durationS?: number
  /** ISO 639-1 the caller FORCED for this clip (absent = let the model
   *  detect). Recorded on the part, not the conversation, because it changes
   *  the answer: a regenerate has to send exactly what the first run sent,
   *  and two clips in one chat may be in different languages. What the model
   *  actually heard lands in [`TranscriptMeta.language`]. */
  language?: string
}

/** A graph database attachment (.tvdb). Bytes live in the attachments table
 *  and are loaded into the tab's Traverse WASM session - they never ride to
 *  the model (100 MB of base64 would be absurd); the model reaches the graph
 *  through the graph_query tool instead, and the request instructions carry a
 * schema summary (phase 2). */
export interface GraphPart {
  type: 'graph'
  /** id of the stored bytes in the attachments table. */
  attachmentId: string
  name: string
  /** byte size of the original file. */
  size?: number
}

export type ContentPart = TextPart | ImagePart | FilePart | AudioPart | GraphPart

/** One word of a transcript with how sure the model was of it - `exp` of the
 *  mean logprob of its tokens. The raw logprob is deliberately not persisted:
 *  it is the same fact twice, and the probability is the one the UI reads. */
export interface TranscriptWord {
  word: string
  speaker?: number | null
  speakers?: number[]
  speaker_status?: 'assigned' | 'ambiguous' | 'unassigned' | 'untimed'
  /** Duration fractions, NOT recognition confidence. */
  speaker_coverage?: number[]
  /** Optional since the alignment enrichment: a word synthesized
   *  from plain text to carry aligner times has no model probability to
   *  report, and inventing one would defeat the point of showing them. Words
   *  from a lane's own answer always carry it. */
  confidence?: number
  /** Seconds from the start of the clip, where the lane recovered them
   *  (whisper's cross-attention DTW). ABSENT on a lane with no
   *  timing at all - the generative ASR families have logprobs and no clock -
   *  and absent together, so a word never carries half a span. Where they are
   *  present a click seeks to the WORD; where they are not, to its segment. */
  start?: number
  end?: number
  /** What the model NEARLY picked at this word's first token - the road not
   *  taken. A reader can judge "ville / nearly vill" at a glance where a bare
   *  percentage told them nothing they could act on.
   *
   *  First token only, so it reads as an alternative WORD; mid-word the
   *  runner-up is a BPE fragment and the server sends nothing rather than
   *  show one. Absent on the whisper lane entirely: its sampling is a CUDA
   * kernel that emits the argmax alone. */
  alt?: string
  /** top1 - top2 in probability space. Small = the model was torn between two
   *  words, which is where errors live - a diffuse distribution with no close
   *  rival is a different thing and this separates them. */
  margin?: number
}

/** One timed span of a transcript. `confidence` is the mean per-token
 *  probability over the segment - the Studio colours low-confidence spans
 *  rather than hiding the uncertainty. */
export interface TranscriptSegment {
  start: number
  end: number
  text: string
  confidence?: number
  words?: TranscriptWord[]
}

/** The structured half of a transcription answer, hung BESIDE `content` the
 *  way `toolCalls` and `webSearches` are - never inside it. The transcript
 *  text itself stays an ordinary TextPart, so copy, search, export and every
 *  existing renderer keep working on a transcription turn untouched; this
 *  only ENRICHES the turn with a player, scrub and timings when present. */
export interface TranscriptMeta {
  diarization?: SpeakerTimeline
  /** ISO 639-1, as the model detected or the caller forced. */
  language?: string
  /** audio length in seconds, as the server measured it. */
  durationS?: number
  segments?: TranscriptSegment[]
  /** The clip's words, flat and in reading order.
   *
   *  Two lanes fill this and they differ in one field. Whisper asked for
   * `word` granularity sends a time on every word, and then this
   *  is the finer of the two lists: the renderer prefers it and a click seeks
   *  to the word. A generative ASR model (Qwen3-ASR, granite-speech) has
   *  logprobs and no timestamp vocabulary at all, so its words carry
   *  confidence and no times - and it sends no `segments` either, which is
   *  why an untimed list here is never a downgrade from one. */
  words?: TranscriptWord[]
  /** The model that supplied the word times, when a separate forced-alignment
   * pass put them there rather than the transcriber itself.
   *  Absent = the lane timed its own words.
   *
   *  It is recorded because an enrichment nobody can see reads as one that
   *  never ran: the times only show up as highlighting during playback, so
   *  without this the turn has no way to say a second model was involved
   *  (- "didn't see forced aligner being used at all"). */
  wordsFrom?: string
  /** false when the aligner answered but flagged the clip's language as
   *  outside its trained set - the times came back and may well be useful,
   *  but "timed by X" without that caveat would overstate them. */
  wordsLangOk?: boolean
  /** Spans where the decode stopped being about the audio: a
   *  window of silence the model wrote a sentence over, a decode that
   *  collapsed into a loop and was cut. Absent on every clean transcription,
   *  which is nearly all of them.
   *
   *  It is persisted with the rest of the meta because it is part of what the
   *  answer was - a transcript re-read next year has to still say which four
   *  seconds of it were refused. */
  guards?: TranscriptGuard[]
}

export interface SpeakerTimeline {
  model: string
  duration: number
  segments: { speaker: number; start: number; end: number }[]
  preset: string
  attribution: 'time_overlap_v1'
}

/** One span of audio whose decode was cut or refused. */
export interface TranscriptGuard {
  start: number
  end: number
  /** machine name: `no_speech` | `repetition` | `length` | `context` */
  reason: string
  /** the server's own sentence for it - shown rather than a mapping of our
   *  own, so the Studio and a curl user read the same explanation */
  note: string
  /** whether the span's text was discarded (true) or kept up to the cut */
  dropped: boolean
}

/** One grounded region of an OCR answer. Coordinates are the model's own
 *  0-999-normalized integers over the page image - multiply by
 *  `dimension / 999` to land in pixels, which is exactly what the overlay
 *  does in percentages. A region usually has one box; the grounding form can
 *  carry several for a label that spans the page. */
export interface OcrRegion {
  /** the block's own words (document mode) - the box↔text association */
  text?: string
  label: string
  boxes: [number, number, number, number][]
  /** full 4-corner quads [x1,y1,...,x4,y4] parallel to boxes (spotting keeps
   *  rotated shapes; each box is its quad's axis-aligned hull) */
  quads?: number[][]
  /** continues the previous block (LightOnOCR-3's `label+`: a paragraph
   *  flowing into the next column) */
  continues?: boolean
  /** which page of the request it is on (PaddleOCR-VL's document pipeline,
   *  which reads every page of one request; 0-based) */
  page?: number
  /** the box in the page's own pixels [x1, y1, x2, y2] */
  bbox?: [number, number, number, number]
  /** a picture region's crop name in the answer's markdown */
  image?: string
  /** a merged block's group (its first block's index) */
  group?: number
  /** why a readable region was not read */
  unread?: string
}

/** The structured half of an OCR answer (deepseek2-ocr family) -
 *  the server's `ocr` response extension, hung beside `content` the way
 *  `transcript` is. Everything here is the SERVER'S resolution echo (what
 *  actually ran, never what was asked for), so a revisited chat still says
 *  how the page was read. The text itself stays an ordinary TextPart. */
export interface OcrMeta {
  /** resolved task mode; null = the caller's own words rode (pass-through). */
  mode?: string | null
  /** crop class that ran: 'gundam' (tiled) or 'base' (whole page). */
  crop?: string
  /** the grounded-region parse was armed for this decode. */
  grounding?: boolean
  pages?: number
  /** tower views encoded (global + crop tiles, or pages). */
  views?: number
  tiles?: number
  /** prefill rows the image marker(s) expanded to. */
  imageTokens?: number
  passThrough?: boolean
  /** an explicit mode replaced text the user also typed - shown, never silent. */
  droppedText?: boolean
  /** the server's repetition stop ended the read where the model began to
   *  loop (PaddleOCR-VL): what came before is kept, the rest of the page is
   *  unread - shown, never silent. */
  repetitionStop?: boolean
  /** grounded regions parsed from the finished output (grounding only). */
  regions?: OcrRegion[]
}

/** One page of a document run: its own request, its own stream.
 *  Because each page is a separate request, its regions are unambiguously
 *  its own - the page-index problem the single-request wire has does not
 *  exist here. */
export interface DocPage {
  state: 'queued' | 'reading' | 'done' | 'review' | 'error'
  text: string
  /** why the page needs review (degeneration) or failed (the error) */
  note?: string
  regions?: OcrRegion[]
  /** wall-clock ms for this page's request */
  ms?: number
  /** per-word confidence from the Responses logprobs include:
   *  words in stream order, confidence = exp(mean token logprob). Only kept
   *  when the endpoint delivered logprobs. */
  words?: { w: string; c: number }[]
}

/** A document-parser turn that fanned out one request per page. */
export interface DocRunMeta {
  pages: DocPage[]
}

/** Status of an MCP tool invocation as it moves through the agent loop. */
export type McpCallStatus =
  | 'pending' // awaiting the user's approve/deny decision
  | 'in_progress' // approved (or auto-run) and executing
  | 'completed'
  | 'failed'
  | 'denied'

/** A tool call the server's MCP agent loop surfaced in the stream - rendered as
 *  a card in the assistant turn (arguments in, result/approval out). */
export interface McpCall {
  /** the call id (mcp_call.id, and mcp_approval_request.call_id). */
  id: string
  /** which registered server the tool belongs to. */
  serverLabel: string
  /** the tool's real (un-namespaced) name. */
  name: string
  /** arguments the model produced, as a JSON string. */
  arguments: string
  /** the tool's result content (JSON string), once it returns. */
  output?: string
  /** truthy = the call failed; the server sends the error message string. */
  error?: boolean | string
  status: McpCallStatus
  /** while status==='pending': the id to POST an approve/deny decision to. */
  approvalId?: string
}

export type WebSearchStatus = 'in_progress' | 'searching' | 'completed' | 'failed'

export interface WebSearchSource {
  url: string
  title?: string
}

/** One web search the server ran for the model (a `web_search_call` output
 *  item) - rendered as a card with the query and the sources it found. */
export interface WebSearchCall {
  id: string
  query: string
  status: WebSearchStatus
  sources: WebSearchSource[]
  /** provider error message when the search failed. */
  error?: string
  /** Which engine actually answered (`exa`, `brave`, ...), as the server
   *  reported it on the item. Stored with the turn rather than read from the
   *  endpoint's current config, so reopening an old conversation still shows
   *  the provider that ran it after the endpoint has been re-pointed. Absent
   *  on a cloud lane, whose own web search is not one of ours. */
  provider?: string
}

export interface Usage {
  promptTokens: number
  /** Non-reasoning output, including generated tool arguments and markers. */
  completionTokens: number
  timingSource?: 'engine' | 'end-to-end'
  decodeMs?: number
  prefillMs?: number
  queueMs?: number
  /** All output tokens per second; timingSource names the denominator. */
  tps?: number
  /** total wall-clock ms for the turn. */
  ms?: number
  /** time to first token (send -> first reasoning/content token), ms. */
  ttftMs?: number
  /** answer generation time (first content token -> done), ms. */
  answerMs?: number
  /** reasoning-phase token count (when the model thought). */
  reasoningTokens?: number
  /** reasoning-phase duration (first reasoning token -> first content), ms. */
  reasoningMs?: number
  /** reasoning tokens/sec. */
  reasoningTps?: number
  /** who actually served a routed cloud request (OpenRouter stamps it) -
   *  free tiers route between hosts that behave differently. */
  provider?: string
  /** provider-reported cost in USD (OpenRouter sends it per response;
   *  absent on providers that don't report money). Tool turns carry the
   *  whole-turn sum - the manager's agent loop aggregates its rounds. */
  costUsd?: number
}

/** Peak GPU/engine conditions sampled DURING an assistant turn (from the
 *  telemetry stream) - the environment the answer ran under. */
export interface RunGpuEnv {
  device?: string
  utilPeak?: number
  memUsedPeak?: number
  memTotal?: number
  powerPeakW?: number
  tempPeakC?: number
  tokSPeak?: number
  batchPeak?: number
  kvPeak?: number
  kvTotal?: number
}

/** A recorded "run": everything that produced one assistant turn - the
 *  provenance (what was set) + the GPU environment. Persisted on the message so
 *  a revisited chat is a reproducible experiment record. Token/timing metrics
 *  live in `usage`. */
export interface RunMeta {
  model: string
  /** speculation mechanism serving this turn ("MTP", "DFlash1", "off") -
   *  provenance, since it shapes the turn's speed, never its output. */
  spec?: string
  /** exact system-prompt text active this turn ('' = none). */
  systemPrompt?: string
  /** the saved-preset name, when the prompt matched one. */
  systemPromptName?: string
  /** sampling params snapshot at send time. */
  params: SamplingParams
  /** enabled MCP server labels this turn could use. */
  tools: string[]
  /** What this turn's request carried besides its messages (lib/tokens.ts
   *  `PromptShape`). A later send with the same shape can trust this turn's
   *  server-counted `usage.promptTokens` as the exact size of the prefix the
   *  two share. Absent on turns recorded before it existed, and on turns the
   *  macOS app wrote - those are simply never used as an anchor. */
  shape?: { from: string | null; summary: string | null; tools: string }
  /** peak GPU/engine conditions during the turn. */
  gpu?: RunGpuEnv
  /** the turn ran while other compare lanes shared the GPU - its speed
   *  numbers measure contention, not the model. */
  contended?: boolean
  /** epoch ms the turn started. */
  at: number
}

export interface Message {
  /** USER messages only: this turn belongs to one compare lane (a model id).
   *  Other lanes neither receive it nor keep it in their history - the graph
   *  auto-repair report is the first user: telling every lane "your import
   *  failed" made the healthy lane redo a working graph. */
  lane?: string
  /** App-authored user turn (the graph auto-repair report): rendered as a
   *  compact notice with the full text folded away, never as a person's own
   *  words in a user bubble. */
  auto?: boolean
  id: string
  /** The step this turn hangs from - `null` at a root, absent only in a
   *  conversation written before the tree (lib/tree.ts `migrate` infers it
   *  from array order and never has to guess, because those threads are
   *  straight lines).
   *
   *  Points at a step's ANCHOR: for a compare fan-out every lane shares one
   *  parent and the next turn hangs from the first lane, so a group is one
   *  step rather than N competing branches. */
  parentId?: string | null
  role: Role
  content: ContentPart[]
  /** streamed chain-of-thought, kept out of `content`. */
  reasoning?: string
  /** true while tokens are still arriving. */
  streaming?: boolean
  /** user cancelled this turn. */
  stopped?: boolean
  /** turn failed - friendly message. */
  error?: string
  /** the reply was cut short - e.g. 'length' when it hit max output tokens. */
  incomplete?: 'length'
  /** MCP tool calls made during this assistant turn (in arrival order). */
  toolCalls?: McpCall[]
  /** web searches the server ran during this assistant turn. */
  webSearches?: WebSearchCall[]
  /** Present when this turn is a TRANSCRIPTION: the timings and
   *  language that turn the plain text in `content` into a scrubable
   *  transcript. Absent on every other kind of turn. */
  transcript?: TranscriptMeta
  /** Present when this turn is an OCR ANSWER (deepseek2-ocr): the
   *  server's resolution echo + grounded regions. Absent elsewhere. */
  ocr?: OcrMeta
  /** Present when this turn is a DOCUMENT RUN: the document went
   *  out as one request per PAGE, and each page streamed its own extraction.
   *  Page order matches the source order (images in message order, then the
   *  PDF's pages) - DocumentPages builds its stack the same way, so index i
   *  here is page i there. Absent on single-request turns. */
  docRun?: DocRunMeta
  /** Present when this turn is an IMAGE-GENERATION answer: the prompt and
   *  parameters that rode and what the render cost. The pictures themselves
   *  are `image` parts in `content`, each carrying its own `gen` record.
   *  Set the moment the turn starts, so the renderer never treats the empty
   *  turn as a chat reply. */
  imageGen?: ImageGenMeta
  usage?: Usage
  /** recorded run: provenance (prompt/params/model) + GPU environment. */
  run?: RunMeta
  /** which model produced an assistant turn. */
  model?: string
  /** Compare: assistant turns answering the same user turn share a group id
   *  and render side by side. A grouped turn is context only for its own
   *  model's later turns - lanes never see each other's answers. */
  group?: string
  createdAt: number
}

// Mirrors exactly the fields the server's chat endpoint accepts (it uses
// deny_unknown_fields, so we never send anything it would reject).
export interface SamplingParams {
  /** null = untouched: nothing is sent and the model's own serving defaults
   *  apply (honor-the-defaults). A number rides only once the user set it
   *  in the sampler popover. */
  temperature: number | null
  topP: number | null
  topK: number | null
  minP: number | null
  maxTokens: number | null
  frequencyPenalty: number | null
  presencePenalty: number | null
  repeatPenalty: number | null
  seed: number | null
  stop: string[]
  /** Reasoning off for this chat. Models that can be turned off read it as
   *  enable_thinking / reasoning.effort 'none'; a model that always reasons
   *  (gpt-oss, Muse Glimmer) ignores it, and its picker offers no Off item. */
  thinking: boolean
  /** Which rung to reason at, in the MODEL's own spelling - Qwen3.8 says
   *  'xhigh', gpt-oss says 'high'. Free-form because the vocabulary is the
   *  served checkpoint's, read from its capability surface rather than fixed
   *  here; the runner clamps anything it does not have onto its own ladder. */
  reasoningEffort: string
  /** Keep earlier turns' thinking in the prompt (`preserve_thinking`), on the
   *  families whose template grades it - qwen3.6/3.8 today, measured per
   *  template rather than listed here.
   *
   *  Off by default, which is both what this Studio has always effectively done
   *  (it never sent prior reasoning at all) and what Qwen's own card
   *  recommends. On, each past answer's chain of thought is sent back as a
   *  Responses `reasoning` item - better continuity on a long problem, paid for
   *  in context, and reasoning blocks are not small. */
  preserveThinking: boolean
  /** Cap on reasoning tokens (reasoning.max_tokens): at the cap the runner
   *  forces the model out of its think block with the dialect's own
   *  budget-exhaustion recipe. Absent = unbounded. Only drawn - and only
   *  sent - on lanes whose runner advertises it can enforce it. */
  thinkingBudget?: number
}

/** One picked tool source in a custom selection: a whole server (`tool`
 *  absent) or a single tool inside it. `label` is the MCP server label -
 *  the same string that rides the wire as `server_label`. */
export interface ToolPick {
  label: string
  tool?: string
}

/** The chat's tool selection. 'all' (default): every server tool the
 *  endpoint supplies plus the connectors switched on for this chat. Custom:
 *  exactly the picks ride - single tools become the request's
 *  `allowed_tools` filter; zero picks is a tool-free chat. */
export type ToolSelection = { mode: 'all' } | { mode: 'custom'; picks: ToolPick[] }

/** A runner-minted compaction item (OpenAI Responses shape) held for resend.
 *  `content` is the item's `encrypted_content` - plaintext summary on a local
 *  runner (the field the SDK round-trips; no encryption theater on one box). */
export interface ServerCompaction {
  /** the item id the server minted (cp_...). */
  id: string
  /** the summary text (rides back as `encrypted_content`). */
  content: string
  /** where the kept tail begins: the newest user message of the compacted
   *  request (the server's tail rule is "last user item"). Gone/reordered =
   *  the item no longer matches the thread and is ignored. */
  tailStartId: string
  /** which model wrote it (provenance). */
  model: string
  /** epoch ms it was minted. */
  at: number
}

export interface Conversation {
  id: string
  title: string
  /** Absent on older chats: never infer permission to overwrite an old name. */
  titleSource?: 'fallback' | 'generated' | 'manual'
  titleModel?: string
  titleCostUsd?: number
  /** Every turn in this conversation - every branch, not just the one on
   *  screen. What to render and what to send both come from the active path
   *  (lib/tree.ts `activeSteps` / `activeMessages`); reading this array
   *  directly would hand the model branches the user abandoned. Kept an array
   *  rather than an id map so the manager's `conversation_kind` and the stored
   *  JSON doc need no migration. */
  messages: Message[]
  /** Tip of the branch on screen: the path is this leaf walked up to a root.
   *  Always a step ANCHOR. Absent in an empty conversation. */
  leafId?: string
  /** Which child step was last taken at each branch point (`''` keys the
   *  root). Leaving a branch and returning restores the path you were on at
   *  every depth instead of the newest or first child. Pruned whenever a
   *  subtree is deleted, so an entry can never point at a missing turn. */
  branchMemory?: Record<string, string>
  model: string
  systemPrompt: string
  params: SamplingParams
  /** LEGACY (pre tool picker): the old all-or-nothing tools switch. Read
   *  only when `toolSelection` is absent (false = tool-free); never written
   *  anymore. */
  toolsEnabled?: boolean
  /** Which tools this chat offers the model (composer plug picker); absent =
   *  all. Labels not served by the current endpoint are inert, so the
   *  selection survives model switches and compare lanes intersect it with
   *  their own server's set. */
  toolSelection?: ToolSelection
  /** Whether the model may search the web in this chat (default off; needs a
   *  provider set up in Settings). */
  webSearchEnabled?: boolean
  /** Connector-library ids switched on for this chat (composer plug menu).
   *  Each selected connector rides as an inline `mcp` tool; absent = none. */
  connectorIds?: string[]
  /** The composer's OCR reading-mode control (deepseek2-ocr lanes) - sticky
   *  per chat like the toggles around it. Absent = let the server derive
   *  (one picture reads as a document, several as pages of one). Sent as the
   *  request's `ocr.mode` only on lanes whose endpoint advertises the
   *  vocabulary; what actually ran comes back in the answer's echo. */
  ocrMode?: string
  /** The SELECTED document's source-message id (selection = target: the
   *  pane shows it and the mode chips read it; lib/docrun.ts). A newly
   *  sent document selects itself; a pane tab click changes it. */
  activeDocId?: string
  /** Whether the document pane stands open in this conversation. ABSENT is
   *  not "closed": it means nobody has said, and the answer then depends on
   *  the model - a document-parser conversation is about the document, so its
   *  pane opens by itself, while an ordinary chat keeps the pane away until
   *  you click a PDF. Per-conversation rather than a global preference
   *  because "am I reading a document right now" is a fact about the chat,
   * not about the user. */
  docPaneOpen?: boolean
  /** artifact panel folded/open for this conversation (absent = open when
   *  artifacts exist) - the document pane's bargain, applied symmetrically. */
  artifactsPaneOpen?: boolean
  /** Graph auto-repair bookkeeping, PERSISTED: `total` reports sent in this
   *  conversation (hard cap - a page reload must not refill the budget, and
   *  a model that answers every repair by CREATING a new artifact must not
   *  mint itself fresh budgets), `seen` = highest version reported per
   *  artifact (one report per failing version). */
  autoRepairs?: { total: number; seen: Record<string, number> }
  /** 'chat' | 'transcription' | 'document', as the SERVER classified it from
   *  the stored turns (store.rs conversation_kind). Present on a list stub,
   *  which is the case that needs it - a summary carries no messages, so
   *  nothing client-side can tell a document conversation from a chat until
   *  it is opened. Absent once loaded, and absent on a pre-column manager;
   *  the sidebar reads the real turns first either way. */
  kind?: string
  /** The "show where" toggle: arm the grounded-region parse (`ocr.grounding`)
   *  so the answer carries 0-999-normalized boxes for the overlay. */
  ocrRegions?: boolean
  /** The composer's language control in audio mode - sticky per chat, the
   *  way the thinking toggle and the globe are. Three states: an ISO 639-1
   *  code, the literal `'auto'` for "let the model detect", and absent for
   *  "never chosen", which resolves to the browser's own language (see
   *  `askedLanguage`). Absent used to mean auto; it stopped meaning that
   *  because a wrong language guess does not cost a label, it costs the
   *  whole transcript.
   *
   *  What actually rode is COPIED onto each AudioPart at send time, already
   *  resolved, so the turn records what happened even after this changes. */
  audioLanguage?: string
  /** The composer's controls in image mode - size, quality, seed and the
   *  rest - sticky per chat the way sampling is. Absent = the defaults
   *  (`DEFAULT_IMAGE_PARAMS`). What actually rode is snapshotted onto each
   *  turn's `imageGen`. */
  imageParams?: ImageParams
  /** Whether document metadata (PDF title/author/dates) rides into the prompt
   *  with attached files (default on - the server's `file_metadata: "full"`).
   *  Off sends `file_metadata: "off"`: only the extracted content goes. */
  fileMetadataEnabled?: boolean
  /** Whether forensics runs this turn (per-request override of the
   *  endpoint's `[forensics] auto` default). Only meaningful on an endpoint
   *  whose forensics is enabled and a vision model - the composer offers the
   *  control only then. Undefined = follow the endpoint default; `true` sends
   *  `forensics: "on"`, `false` sends `forensics: "off"`. */
  forensicsEnabled?: boolean
  /** LEGACY (chat-level, pre per-file settings): 'text' forced text
   *  extraction for every PDF in the chat. Still sent as the request-level
   *  `pdf_mode` default so old chats keep their behavior; nothing writes it
   *  anymore - new settings live on the file parts themselves. */
  pdfMode?: 'text'
  /** LEGACY (chat-level): request-level `max_pages` default. See `pdfMode`. */
  maxPages?: number
  /** Rolling summary of the oldest messages (context compaction). When valid,
   *  the send path replaces messages [0, summaryCount) with this text instead
   *  of dropping them. The raw messages stay in the doc and the UI untouched. */
  summary?: string
  /** how many leading messages `summary` covers. */
  summaryCount?: number
  /** id of messages[summaryCount - 1] at compaction time. If that boundary
   *  message is gone or different, the summary no longer matches the thread
   *  and is ignored (regenerate/edit safety). */
  summaryLastId?: string
  /** which model wrote the summary (provenance). */
  summaryModel?: string
  /** SERVER-side compaction (local single-model chats): the runner's
   *  compaction item from a `context_management` turn. Later sends resend
   *  `[item, messages from tailStartId...]` - the server's rewrite renders
   *  the identical token stream its post-compaction pass ran on, so every
   *  follow-up is a prefix-cache hit. The raw messages stay in the doc and
   *  the UI untouched, same rule as the client summary above. */
  serverCompaction?: ServerCompaction
  /** Compare is a STATE of the chat: ≥2 running models here means every send
   *  fans out to all of them and replies render as split lanes. "Use this
   *  model" collapses back to a single-model chat (this field clears). */
  compareModels?: string[]
  pinned?: boolean
  createdAt: number
  updatedAt: number
}

export const DEFAULT_PARAMS: SamplingParams = {
  temperature: null,
  topP: null,
  topK: null,
  minP: null,
  maxTokens: 8192,
  frequencyPenalty: null,
  presencePenalty: null,
  repeatPenalty: null,
  seed: null,
  stop: [],
  thinking: true,
  // Empty, not a rung name. A stored word here silently beat the per-model
  // resolution below it: 'medium' is a valid rung on Qwen3.8, so the fallback
  // never fired and every new chat overrode the model's own grading with a
  // house value - on the wire, not just in the picker.
  reasoningEffort: '',
  preserveThinking: false,
}

/** What a new chat asks a picture for: the model's own default size and
 *  step count, the seed following the conversation, one picture, a lossless
 *  file, and two previews on the way - enough to watch it converge without
 *  paying for more decodes than that. */
export const DEFAULT_IMAGE_PARAMS: ImageParams = {
  size: 'auto',
  quality: 'auto',
  steps: null,
  seed: 'thread',
  n: 1,
  format: 'png',
  background: 'auto',
  previews: 2,
}

/** A conversation's image settings, defaults filled in - never the stored
 *  object itself, so a caller can read fields without writing them back. */
export function imageParamsOf(c: Pick<Conversation, 'imageParams'>): ImageParams {
  return { ...DEFAULT_IMAGE_PARAMS, ...(c.imageParams ?? {}) }
}

/** Concatenate a message's text parts (for copy / plain rendering). */
export function messageText(m: Message): string {
  return m.content
    .filter((p): p is TextPart => p.type === 'text')
    .map((p) => p.text)
    .join('')
}
