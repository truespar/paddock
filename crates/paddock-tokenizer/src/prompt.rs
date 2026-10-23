//! Chat-prompt encoding: request content shielded from the added-token pass,
//! and the prompt encoded in cached, parallel pieces.
//!
//! **Shielding.** A rendered chat prompt used to be encoded as one string,
//! so a control-token string that arrived INSIDE request content (a pasted
//! file, a tool result, a tool argument) was matched like the ones the
//! template wrote: a file containing `<|im_end|>\n<|im_start|>system` closed
//! the turn it sat in and opened a system turn. [`Shield`] replaces those
//! strings in the request with placeholders before the template renders,
//! restores them afterwards and records where they landed;
//! [`GgufTokenizer::encode_prompt`] then encodes them as ordinary text - HF's
//! `split_special_tokens=True`, applied to the request's own bytes only, plus
//! the case that flag misses (a vocabulary that looks a whole pre-token up
//! hands the id back anyway - `Index::special_ids`). Only `special` added
//! tokens are shielded: the
//! non-special ones (Qwen's `<tool_call>`, `<think>`) are matched by HF's own
//! pipeline whatever that flag says, and they stay matched here. A prompt
//! with nothing to shield encodes exactly as before.
//!
//! **Pieces.** The library splits its input at every added-token match before
//! anything else touches it, then normalizes, pre-tokenizes and encodes each
//! remaining piece on its own. So cutting the prompt at the START of a match
//! gives pieces whose encodings concatenate to the whole string's, id for id,
//! provided the cut token neither strips the whitespace around it nor
//! insists on word boundaries (`lstrip`, `rstrip`, `single_word` - those read
//! across the cut, so such a match is never a cut). A piece then starts with
//! the match exactly as the whole-string pass sees it, which also keeps the
//! one offset-dependent pre-tokenizer rule (Metaspace's prepend-on-first)
//! reading the same offsets. Chat prompts are cut this way at every turn
//! marker, and a turn's pieces repeat byte for byte on every later request of
//! the conversation, so pieces are cached by digest and only the new ones are
//! encoded, in parallel when there is enough of them. Single-threaded the
//! library runs at about a million tokens a second, which was a fifth of a
//! second of every 200K-token agent turn, prefix-cache hit or not.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use aho_corasick::{AhoCorasick, MatchKind};
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::metaspace::PrependScheme;

use crate::{GgufTokenizer, TokenizerError};

/// Unshielded pieces are joined into runs of at least this many bytes (see
/// `join`).
const JOIN_BYTES: usize = 4096;
/// Runs shorter than this are encoded rather than looked up: hashing and the
/// lock cost more than the encode.
const CACHE_MIN_BYTES: usize = 256;
/// Cache ceiling in ids (4 bytes each): about forty 200K-token conversations.
const CACHE_MAX_IDS: usize = 8 << 20;
/// New bytes below which the pieces are encoded on the calling thread.
const PAR_MIN_BYTES: usize = 64 << 10;
/// Most threads one prompt takes; several requests may be encoding at once.
const PAR_MAX_WORKERS: usize = 8;

/// Per-tokenizer state, built on first use.
#[derive(Default)]
pub(crate) struct PromptState {
    index: OnceLock<Option<Index>>,
    /// The same tokenizer with special-token matching off: encodes the
    /// shielded pieces.
    plain: OnceLock<Tokenizer>,
    cache: Mutex<Cache>,
}

impl std::fmt::Debug for PromptState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PromptState { .. }")
    }
}

struct Entry {
    id: u32,
    /// A match of this token is a safe cut (module docs).
    cut: bool,
}

struct Index {
    /// The library's first-stage matcher, rebuilt from the same set: every
    /// non-normalized added token, leftmost-longest. Pattern ids index
    /// `entries`.
    trie: Option<AhoCorasick>,
    entries: Vec<Entry>,
    /// Special-token strings and their matcher (pattern ids index
    /// `specials`).
    specials: Vec<String>,
    shield_trie: Option<AhoCorasick>,
    /// Their ids. With matching off the text of a special token can still
    /// come back as its id: a vocabulary that looks a whole pre-token up
    /// before merging (`ignore_merges` - Gemma 4, the Llama 3 lineage) finds
    /// `<turn|>` standing alone on its line. Any such id in a shielded body
    /// came from the request (the template's markers are all cuts), so its
    /// span is re-encoded in two halves.
    special_ids: HashSet<u32>,
    /// Shielding needs every special token to be a safe cut and matched on
    /// raw text, and no offset-dependent pre-tokenizer: a shielded piece is
    /// encoded without its leading turn marker.
    shield_ok: bool,
}

impl Index {
    fn build(inner: &Tokenizer) -> Option<Self> {
        let added = inner.get_added_tokens_decoder();
        let mut ids: Vec<u32> = added.keys().copied().collect();
        ids.sort_unstable();
        let mut entries = Vec::new();
        let mut patterns = Vec::new();
        let mut specials = Vec::new();
        let mut special_ids = HashSet::new();
        let mut shield_ok = !offset_dependent(inner.get_pre_tokenizer());
        for id in ids {
            let t = &added[&id];
            let cut = !(t.lstrip || t.rstrip || t.single_word);
            if t.special {
                shield_ok &= cut && !t.normalized;
                specials.push(t.content.clone());
                special_ids.insert(id);
            }
            if !t.normalized {
                patterns.push(t.content.as_str());
                entries.push(Entry { id, cut });
            }
        }
        let build = |patterns: &[&str]| {
            if patterns.is_empty() {
                return Ok(None);
            }
            AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(patterns)
                .map(Some)
        };
        // a build error means "no index": plain whole-string encoding
        let trie = build(&patterns).ok()?;
        let shield_patterns: Vec<&str> = specials.iter().map(String::as_str).collect();
        let shield_trie = build(&shield_patterns).ok()?;
        Some(Self {
            trie,
            entries,
            specials,
            shield_trie,
            special_ids,
            shield_ok,
        })
    }

    /// Cut `text` at the start of every safe added-token match that touches
    /// no shielded range; a piece is shielded when one starts inside it.
    fn pieces(&self, text: &str, shielded: &[Range<usize>]) -> Vec<Piece> {
        let mut pieces = Vec::new();
        let mut cur = Piece {
            span: 0..0,
            lead_end: 0,
            lead: None,
            shielded: false,
        };
        if let Some(trie) = &self.trie {
            let mut s = 0;
            for m in trie.find_iter(text) {
                while s < shielded.len() && shielded[s].end <= m.start() {
                    s += 1;
                }
                if shielded.get(s).is_some_and(|r| r.start < m.end()) {
                    continue;
                }
                let e = &self.entries[m.pattern().as_usize()];
                if !e.cut {
                    continue;
                }
                if m.start() > cur.span.start {
                    cur.span.end = m.start();
                    pieces.push(cur);
                }
                cur = Piece {
                    span: m.start()..m.start(),
                    lead_end: m.end(),
                    lead: Some(e.id),
                    shielded: false,
                };
            }
        }
        cur.span.end = text.len();
        pieces.push(cur);
        let mut p = 0;
        for r in shielded {
            while p + 1 < pieces.len() && pieces[p].span.end <= r.start {
                p += 1;
            }
            pieces[p].shielded = true;
        }
        pieces
    }
}

/// Join consecutive unshielded pieces until a run reaches `JOIN_BYTES`. A run
/// of consecutive pieces starts at a cut and ends at one, so it encodes like
/// its pieces do; joining keeps the per-call overhead and the cache's entry
/// count down on prompts made of many short turns (an agent's tool calls
/// are a few markers and a line each). Built from the start of the prompt,
/// the runs of an unchanged history come out the same on every request.
fn join(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::with_capacity(pieces.len());
    for p in pieces {
        if let Some(last) = out.last_mut()
            && !last.shielded
            && !p.shielded
            && last.span.len() < JOIN_BYTES
        {
            last.span.end = p.span.end;
            continue;
        }
        out.push(p);
    }
    out
}

/// Does a pre-tokenizer read the piece's offset in the original string?
fn offset_dependent(pre: Option<&PreTokenizerWrapper>) -> bool {
    match pre {
        Some(PreTokenizerWrapper::Metaspace(m)) => m.get_prepend_scheme() == PrependScheme::First,
        Some(PreTokenizerWrapper::Sequence(s)) => {
            s.as_ref().iter().any(|p| offset_dependent(Some(p)))
        }
        _ => false,
    }
}

struct Piece {
    span: Range<usize>,
    /// End of the leading added token (`span.start` when there is none).
    lead_end: usize,
    lead: Option<u32>,
    shielded: bool,
}

#[derive(Default)]
struct Cache {
    map: HashMap<[u8; 32], (Arc<[u32]>, u64)>,
    tick: u64,
    ids: usize,
}

impl Cache {
    fn get(&mut self, key: &[u8; 32]) -> Option<Arc<[u32]>> {
        self.tick += 1;
        let tick = self.tick;
        self.map.get_mut(key).map(|(ids, used)| {
            *used = tick;
            ids.clone()
        })
    }

    fn put(&mut self, key: [u8; 32], ids: Arc<[u32]>) {
        self.tick += 1;
        let n = ids.len();
        if let Some((old, _)) = self.map.insert(key, (ids, self.tick)) {
            self.ids -= old.len();
        }
        self.ids += n;
        if self.ids > CACHE_MAX_IDS {
            // least recently used out, down to three quarters
            let mut by_age: Vec<([u8; 32], u64)> =
                self.map.iter().map(|(k, (_, t))| (*k, *t)).collect();
            by_age.sort_unstable_by_key(|&(_, t)| t);
            for (k, _) in by_age {
                if self.ids <= CACHE_MAX_IDS / 4 * 3 {
                    break;
                }
                if let Some((old, _)) = self.map.remove(&k) {
                    self.ids -= old.len();
                }
            }
        }
    }
}

/// Knobs the tests turn; serving uses [`Tuning::default`].
#[derive(Clone, Copy)]
pub(crate) struct Tuning {
    pub par_min_bytes: usize,
    pub cache: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            par_min_bytes: PAR_MIN_BYTES,
            cache: true,
        }
    }
}

fn lib(e: impl std::fmt::Display) -> TokenizerError {
    TokenizerError::Library(e.to_string())
}

impl GgufTokenizer {
    fn prompt_index(&self) -> Option<&Index> {
        self.prompt
            .index
            .get_or_init(|| Index::build(&self.inner))
            .as_ref()
    }

    fn plain_tokenizer(&self) -> &Tokenizer {
        self.prompt.plain.get_or_init(|| {
            let mut t = self.inner.clone();
            t.set_encode_special_tokens(true);
            t
        })
    }

    /// Encode a shielded body as text (see `Index::special_ids`).
    fn encode_shielded(
        &self,
        ix: &Index,
        body: &str,
        out: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        let plain = self.plain_tokenizer();
        let e = plain.encode(body, false).map_err(lib)?;
        if !e.get_ids().iter().any(|i| ix.special_ids.contains(i)) {
            out.extend_from_slice(e.get_ids());
            return Ok(());
        }
        for (&id, &(a, b)) in e.get_ids().iter().zip(e.get_offsets()) {
            let span = body.get(a..b).unwrap_or_default();
            // the first character on its own, then the rest: neither half
            // is the token's text, so neither looks it up
            let cut = span.chars().next().map_or(0, char::len_utf8);
            if !ix.special_ids.contains(&id) || cut == 0 || cut == span.len() {
                out.push(id);
                continue;
            }
            for half in [&span[..cut], &span[cut..]] {
                let h = plain.encode_fast(half, false).map_err(lib)?;
                out.extend_from_slice(h.get_ids());
            }
        }
        Ok(())
    }

    /// A shield for one request's content, or None when this tokenizer
    /// cannot shield (module docs). `exempt` names special tokens the
    /// request may write on purpose - the media placeholders a model card
    /// documents inside text (`<|audio|>`, `<|image_pad|>`).
    pub fn shield(&self, exempt: &[&str]) -> Option<Shield<'_>> {
        let ix = self.prompt_index()?;
        if !ix.shield_ok {
            return None;
        }
        let trie = ix.shield_trie.as_ref()?;
        use std::hash::{BuildHasher, Hasher};
        let nonce = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish();
        Some(Shield {
            specials: &ix.specials,
            trie,
            exempt: ix
                .specials
                .iter()
                .map(|s| exempt.contains(&s.as_str()))
                .collect(),
            tag: format!("~#{nonce:020}#"),
            originals: Vec::new(),
        })
    }

    /// Encode a rendered chat prompt: `shielded` are the byte ranges
    /// [`Shield::unmask`] reported (encoded as text), everything else exactly
    /// as [`Self::encode`] encodes it.
    pub fn encode_prompt(
        &self,
        text: &str,
        shielded: &[Range<usize>],
    ) -> Result<Vec<u32>, TokenizerError> {
        self.encode_prompt_tuned(text, shielded, Tuning::default())
    }

    pub(crate) fn encode_prompt_tuned(
        &self,
        text: &str,
        shielded: &[Range<usize>],
        tuning: Tuning,
    ) -> Result<Vec<u32>, TokenizerError> {
        let Some(ix) = self.prompt_index() else {
            return self.encode(text);
        };
        let shielded = if ix.shield_ok { shielded } else { &[] };
        let pieces = join(ix.pieces(text, shielded));
        if pieces.len() == 1 && !pieces[0].shielded {
            return self.encode(text);
        }

        // cached pieces first; the rest are encoded below
        let mut out: Vec<Option<Arc<[u32]>>> = vec![None; pieces.len()];
        let mut keys: Vec<Option<[u8; 32]>> = vec![None; pieces.len()];
        let mut todo: Vec<usize> = Vec::new();
        {
            let mut cache = tuning
                .cache
                .then(|| self.prompt.cache.lock().unwrap_or_else(|e| e.into_inner()));
            for (i, p) in pieces.iter().enumerate() {
                let bytes = &text.as_bytes()[p.span.clone()];
                if let Some(c) = cache.as_mut()
                    && bytes.len() >= CACHE_MIN_BYTES
                {
                    // a shielded run encodes its body with special-token
                    // matching off, so its ids are a function of its bytes
                    // and that one flag (the domain byte)
                    let mut h = blake3::Hasher::new();
                    h.update(&[u8::from(p.shielded)]);
                    h.update(bytes);
                    let key: [u8; 32] = h.finalize().into();
                    out[i] = c.get(&key);
                    keys[i] = Some(key);
                }
                if out[i].is_none() {
                    todo.push(i);
                }
            }
        }

        let encode = |i: usize| -> Result<Vec<u32>, TokenizerError> {
            let p = &pieces[i];
            if !p.shielded {
                let e = self.inner.encode_fast(&text[p.span.clone()], false);
                return Ok(e.map_err(lib)?.get_ids().to_vec());
            }
            let mut ids: Vec<u32> = p.lead.into_iter().collect();
            let body = &text[p.lead_end.max(p.span.start)..p.span.end];
            self.encode_shielded(ix, body, &mut ids)?;
            Ok(ids)
        };
        let new_bytes: usize = todo.iter().map(|&i| pieces[i].span.len()).sum();
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(PAR_MAX_WORKERS)
            .min(todo.len());
        let encoded: Vec<(usize, Vec<u32>)> = if workers > 1 && new_bytes >= tuning.par_min_bytes {
            // contiguous runs of about equal bytes, one per worker
            let share = new_bytes.div_ceil(workers);
            let mut runs: Vec<Vec<usize>> = vec![Vec::new()];
            let mut acc = 0;
            for &i in &todo {
                if acc >= share && runs.len() < workers {
                    runs.push(Vec::new());
                    acc = 0;
                }
                acc += pieces[i].span.len();
                runs.last_mut().expect("one run").push(i);
            }
            let encode = &encode;
            let results: Vec<Result<Vec<(usize, Vec<u32>)>, TokenizerError>> =
                std::thread::scope(|scope| {
                    let handles: Vec<_> = runs
                        .iter()
                        .map(|run| {
                            scope.spawn(move || {
                                run.iter()
                                    .map(|&i| encode(i).map(|ids| (i, ids)))
                                    .collect::<Result<Vec<_>, _>>()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .unwrap_or_else(|_| Err(lib("encoder thread panicked")))
                        })
                        .collect()
                });
            let mut all = Vec::with_capacity(todo.len());
            for r in results {
                all.extend(r?);
            }
            all
        } else {
            todo.iter()
                .map(|&i| encode(i).map(|ids| (i, ids)))
                .collect::<Result<_, _>>()?
        };

        let mut cache = tuning
            .cache
            .then(|| self.prompt.cache.lock().unwrap_or_else(|e| e.into_inner()));
        for (i, ids) in encoded {
            let ids: Arc<[u32]> = ids.into();
            if let (Some(c), Some(key)) = (cache.as_mut(), keys[i]) {
                c.put(key, ids.clone());
            }
            out[i] = Some(ids);
        }
        drop(cache);
        let total = out.iter().flatten().map(|ids| ids.len()).sum();
        let mut ids = Vec::with_capacity(total);
        for piece in out.iter().flatten() {
            ids.extend_from_slice(piece);
        }
        Ok(ids)
    }
}

/// One request's content shield (module docs): masks the special-token
/// strings in the request's JSON before rendering, restores them after.
pub struct Shield<'t> {
    specials: &'t [String],
    trie: &'t AhoCorasick,
    exempt: Vec<bool>,
    /// `~#<nonce>#` - a placeholder is the tag, an index into `originals`,
    /// `#~`. ASCII punctuation and digits, so JSON, `trim`, `upper` and the
    /// other filters templates run leave it as it is; the nonce keeps a
    /// request from spelling one by accident.
    tag: String,
    originals: Vec<u32>,
}

impl Shield<'_> {
    /// How many strings were masked so far.
    pub fn masked(&self) -> usize {
        self.originals.len()
    }

    /// Would [`Self::mask`] change anything in `v`?
    pub fn finds(&self, v: &serde_json::Value) -> bool {
        use serde_json::Value;
        match v {
            Value::String(s) => self
                .trie
                .find_iter(s.as_str())
                .any(|m| !self.exempt[m.pattern().as_usize()]),
            Value::Array(a) => a.iter().any(|x| self.finds(x)),
            Value::Object(o) => o.values().any(|x| self.finds(x)),
            _ => false,
        }
    }

    /// Mask every shielded string in `v`'s string values (keys are left
    /// alone).
    pub fn mask(&mut self, v: &mut serde_json::Value) {
        use serde_json::Value;
        match v {
            Value::String(s) => {
                if let Some(m) = self.mask_str(s) {
                    *s = m;
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| self.mask(x)),
            Value::Object(o) => o.iter_mut().for_each(|(_, x)| self.mask(x)),
            _ => {}
        }
    }

    fn mask_str(&mut self, s: &str) -> Option<String> {
        let mut out: Option<String> = None;
        let mut last = 0;
        for m in self.trie.find_iter(s) {
            let k = m.pattern().as_usize();
            if self.exempt[k] {
                continue;
            }
            let o = out.get_or_insert_with(|| String::with_capacity(s.len() + 32));
            o.push_str(&s[last..m.start()]);
            let _ = write!(o, "{}{}#~", self.tag, self.originals.len());
            self.originals.push(k as u32);
            last = m.end();
        }
        out.map(|mut o| {
            o.push_str(&s[last..]);
            o
        })
    }

    /// Restore the masked strings in a rendered prompt, returning the prompt
    /// and the byte ranges they occupy in it. None when a placeholder came
    /// back altered (a template sliced or rewrote it): re-render unshielded.
    pub fn unmask(&self, rendered: String) -> Option<(String, Vec<Range<usize>>)> {
        if self.originals.is_empty() || !rendered.contains(&self.tag) {
            return Some((rendered, Vec::new()));
        }
        self.unmask_str(&rendered)
    }

    fn unmask_str(&self, rendered: &str) -> Option<(String, Vec<Range<usize>>)> {
        let mut out = String::with_capacity(rendered.len());
        let mut ranges = Vec::new();
        let mut rest = rendered;
        while let Some(p) = rest.find(&self.tag) {
            out.push_str(&rest[..p]);
            let after = &rest[p + self.tag.len()..];
            let digits = after.bytes().take_while(u8::is_ascii_digit).count();
            if digits == 0 || !after[digits..].starts_with("#~") {
                return None;
            }
            let k = *self.originals.get(after[..digits].parse::<usize>().ok()?)?;
            let s = &self.specials[k as usize];
            ranges.push(out.len()..out.len() + s.len());
            out.push_str(s);
            rest = &after[digits + 2..];
        }
        out.push_str(rest);
        Some((out, ranges))
    }

    /// Undo [`Self::mask`] on a value (the fallback when a render could not
    /// be unmasked).
    pub fn restore(&self, v: &mut serde_json::Value) {
        use serde_json::Value;
        match v {
            Value::String(s) if s.contains(&self.tag) => {
                if let Some((t, _)) = self.unmask_str(s) {
                    *s = t;
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| self.restore(x)),
            Value::Object(o) => o.iter_mut().for_each(|(_, x)| self.restore(x)),
            _ => {}
        }
    }
}
