//! A SentencePiece BPE model read from its serialized `ModelProto` - the
//! encoder Nemotron 3.5 Transcribe's CTC timestamp head was trained against
//! (its proto rides the checkpoint as a u8 tensor). SentencePiece's own
//! algorithm, written from the library's documented behavior:
//!
//! - normalize: the precompiled charsmap (a darts-clone double-array trie of
//!   rewrite rules) applied by LONGEST prefix over the raw bytes, a user-
//!   defined symbol passing through untouched, a dummy-prefix space, spaces
//!   escaped to U+2581 (`▁`). HF tokenizers' `Precompiled` normalizer walks
//!   grapheme clusters instead, which is a different function, so it is not
//!   used here.
//! - encode: split into characters (a user-defined symbol is one frozen
//!   symbol), then merge the adjacent pair whose concatenation is a piece
//!   with the highest score, leftmost first on a tie, until no pair is a
//!   piece. A piece outside the vocabulary maps to `unk`, and a run of them
//!   folds into one (the processor's rule).
//!
//! Only what the BPE models in use need: no sampling, no byte fallback, no
//! unigram, no denormalizer; a model asking for any of those is refused at
//! load rather than encoded differently.

use std::collections::{BinaryHeap, HashMap};

/// U+2581, SentencePiece's escaped space.
pub const SPACE: &str = "\u{2581}";

#[derive(Debug, thiserror::Error)]
pub enum SpmError {
    #[error("sentencepiece model: {0}")]
    Model(String),
}

fn bad(what: impl Into<String>) -> SpmError {
    SpmError::Model(what.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

/// A loaded BPE model.
pub struct SpmBpe {
    pieces: Vec<(String, f32, PieceType)>,
    /// normal / user-defined / unused pieces -> id (what merges may make)
    index: HashMap<String, u32>,
    /// control pieces -> id
    reserved: HashMap<String, u32>,
    /// user-defined symbols, longest first per leading byte is not needed:
    /// the prefix match tries every one
    user_defined: Vec<String>,
    unk: u32,
    trie: Vec<u32>,
    normalized: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

/// A protobuf field: (number, wire type, payload).
enum Field<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
}

fn varint(b: &[u8], at: &mut usize) -> Result<u64, SpmError> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*at).ok_or_else(|| bad("truncated varint"))?;
        *at += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(bad("varint past 64 bits"))
}

fn fields(b: &[u8]) -> Result<Vec<(u32, Field<'_>)>, SpmError> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < b.len() {
        let key = varint(b, &mut at)?;
        let (num, wire) = ((key >> 3) as u32, key & 7);
        let f = match wire {
            0 => Field::Varint(varint(b, &mut at)?),
            1 => {
                b.get(at..at + 8).ok_or_else(|| bad("truncated fixed64"))?;
                at += 8;
                Field::Fixed64
            }
            2 => {
                let n = varint(b, &mut at)? as usize;
                let s = b.get(at..at + n).ok_or_else(|| bad("truncated field"))?;
                at += n;
                Field::Bytes(s)
            }
            5 => {
                let s = b.get(at..at + 4).ok_or_else(|| bad("truncated fixed32"))?;
                at += 4;
                Field::Fixed32(u32::from_le_bytes(s.try_into().expect("4 bytes")))
            }
            w => return Err(bad(format!("wire type {w}"))),
        };
        out.push((num, f));
    }
    Ok(out)
}

/// darts-clone's double-array unit accessors.
fn unit_has_leaf(u: u32) -> bool {
    (u >> 8) & 1 == 1
}
fn unit_value(u: u32) -> u32 {
    u & ((1 << 31) - 1)
}
fn unit_label(u: u32) -> u32 {
    u & ((1 << 31) | 0xff)
}
fn unit_offset(u: u32) -> usize {
    ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize
}

impl SpmBpe {
    /// Parse a serialized `ModelProto`.
    pub fn from_proto(proto: &[u8]) -> Result<Self, SpmError> {
        let mut pieces = Vec::new();
        let mut model_type = 1u64; // UNIGRAM is the proto default
        let mut whitespace_suffix = false;
        let mut byte_fallback = false;
        let mut charsmap: &[u8] = &[];
        let (mut add_dummy_prefix, mut remove_extra, mut escape) = (true, true, true);
        for (num, f) in fields(proto)? {
            match (num, f) {
                (1, Field::Bytes(b)) => {
                    let (mut piece, mut score, mut ty) = (String::new(), 0f32, 1u64);
                    for (n, g) in fields(b)? {
                        match (n, g) {
                            (1, Field::Bytes(s)) => {
                                piece = std::str::from_utf8(s)
                                    .map_err(|_| bad("a piece is not UTF-8"))?
                                    .to_owned()
                            }
                            (2, Field::Fixed32(v)) => score = f32::from_bits(v),
                            (3, Field::Varint(v)) => ty = v,
                            _ => {}
                        }
                    }
                    let ty = match ty {
                        1 => PieceType::Normal,
                        2 => PieceType::Unknown,
                        3 => PieceType::Control,
                        4 => PieceType::UserDefined,
                        5 => PieceType::Unused,
                        6 => PieceType::Byte,
                        t => return Err(bad(format!("piece type {t}"))),
                    };
                    pieces.push((piece, score, ty));
                }
                (2, Field::Bytes(b)) => {
                    for (n, g) in fields(b)? {
                        match (n, g) {
                            (3, Field::Varint(v)) => model_type = v,
                            (24, Field::Varint(v)) => whitespace_suffix = v != 0,
                            (35, Field::Varint(v)) => byte_fallback = v != 0,
                            _ => {}
                        }
                    }
                }
                (3, Field::Bytes(b)) => {
                    for (n, g) in fields(b)? {
                        match (n, g) {
                            (2, Field::Bytes(s)) => charsmap = s,
                            (3, Field::Varint(v)) => add_dummy_prefix = v != 0,
                            (4, Field::Varint(v)) => remove_extra = v != 0,
                            (5, Field::Varint(v)) => escape = v != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if model_type != 2 {
            return Err(bad(format!("model type {model_type} (BPE is 2)")));
        }
        if whitespace_suffix || byte_fallback {
            return Err(bad(
                "treat_whitespace_as_suffix / byte_fallback are not served",
            ));
        }
        let (trie, normalized) = if charsmap.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let n = u32::from_le_bytes(
                charsmap
                    .get(..4)
                    .ok_or_else(|| bad("charsmap header"))?
                    .try_into()
                    .expect("4 bytes"),
            ) as usize;
            let t = charsmap
                .get(4..4 + n)
                .ok_or_else(|| bad("charsmap trie past the blob"))?;
            let trie: Vec<u32> = t
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            (trie, charsmap[4 + n..].to_vec())
        };
        let mut index = HashMap::new();
        let mut reserved = HashMap::new();
        let mut user_defined = Vec::new();
        let mut unk = None;
        for (i, (p, _, ty)) in pieces.iter().enumerate() {
            let i = i as u32;
            match ty {
                PieceType::Normal | PieceType::UserDefined | PieceType::Unused => {
                    if index.insert(p.clone(), i).is_some() {
                        return Err(bad(format!("duplicate piece {p:?}")));
                    }
                }
                _ => {
                    reserved.insert(p.clone(), i);
                }
            }
            match ty {
                PieceType::UserDefined => user_defined.push(p.clone()),
                PieceType::Unknown => unk = Some(i),
                PieceType::Byte => return Err(bad("byte pieces are not served")),
                // an unused piece is re-split after the merges in
                // SentencePiece; no model in use carries one
                PieceType::Unused => return Err(bad("unused pieces are not served")),
                _ => {}
            }
        }
        Ok(Self {
            pieces,
            index,
            reserved,
            user_defined,
            unk: unk.ok_or_else(|| bad("no unknown piece"))?,
            trie,
            normalized,
            add_dummy_prefix,
            remove_extra_whitespaces: remove_extra,
            escape_whitespaces: escape,
        })
    }

    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    pub fn unk_id(&self) -> u32 {
        self.unk
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.0.as_str())
    }

    /// SentencePiece's `PieceToId`: a vocabulary piece, a control piece, or
    /// `unk`.
    pub fn piece_to_id(&self, piece: &str) -> u32 {
        self.index
            .get(piece)
            .or_else(|| self.reserved.get(piece))
            .copied()
            .unwrap_or(self.unk)
    }

    /// The longest user-defined symbol at the start of `s`, in bytes.
    fn user_symbol(&self, s: &[u8]) -> Option<usize> {
        self.user_defined
            .iter()
            .filter(|u| s.starts_with(u.as_bytes()))
            .map(|u| u.len())
            .max()
    }

    /// The longest charsmap rule at the start of `s`: (bytes consumed, its
    /// replacement).
    fn rule(&self, s: &[u8]) -> Option<(usize, &[u8])> {
        if self.trie.is_empty() {
            return None;
        }
        let mut pos = unit_offset(self.trie[0]);
        let mut best = None;
        for (i, &c) in s.iter().enumerate() {
            pos ^= c as usize;
            let u = *self.trie.get(pos)?;
            if unit_label(u) != c as u32 {
                break;
            }
            pos ^= unit_offset(u);
            if unit_has_leaf(u) {
                let v = unit_value(*self.trie.get(pos)?) as usize;
                best = Some((i + 1, v));
            }
        }
        best.map(|(n, v)| {
            let tail = &self.normalized[v.min(self.normalized.len())..];
            let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
            (n, &tail[..end])
        })
    }

    /// SentencePiece's `Normalizer::Normalize` (no alignment output).
    pub fn normalize(&self, text: &str) -> String {
        let mut out: Vec<u8> = Vec::with_capacity(text.len() + 3);
        let mut input = text.as_bytes();
        let space: &[u8] = if self.escape_whitespaces {
            SPACE.as_bytes()
        } else {
            b" "
        };
        if self.remove_extra_whitespaces {
            while let [b' ', rest @ ..] = input {
                input = rest;
            }
        }
        if input.is_empty() {
            return String::new();
        }
        if self.add_dummy_prefix {
            out.extend_from_slice(space);
        }
        let mut prev_space = self.remove_extra_whitespaces;
        while !input.is_empty() {
            let (piece, used): (std::borrow::Cow<'_, [u8]>, usize) =
                if let Some(n) = self.user_symbol(input) {
                    (input[..n].into(), n)
                } else if let Some((n, rep)) = self.rule(input) {
                    (rep.into(), n)
                } else {
                    // one UTF-8 character (the input is a &str, so valid)
                    let n = utf8_len(input[0]).min(input.len());
                    (input[..n].into(), n)
                };
            let mut sp: &[u8] = &piece;
            if prev_space {
                while let [b' ', rest @ ..] = sp {
                    sp = rest;
                }
            }
            if !sp.is_empty() {
                for &b in sp {
                    if self.escape_whitespaces && b == b' ' {
                        out.extend_from_slice(space);
                    } else {
                        out.push(b);
                    }
                }
                prev_space = sp.ends_with(b" ");
            }
            input = &input[used..];
            prev_space = prev_space && self.remove_extra_whitespaces;
        }
        if self.remove_extra_whitespaces {
            while out.ends_with(space) {
                out.truncate(out.len() - space.len());
            }
        }
        String::from_utf8(out).unwrap_or_default()
    }

    /// `encode_as_ids`: normalize, then the BPE merges.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = self.normalize(text);
        let b = norm.as_bytes();
        // (start, len, prev, next, frozen)
        let mut syms: Vec<(usize, usize, isize, isize, bool)> = Vec::new();
        let mut at = 0usize;
        while at < b.len() {
            let (n, frozen) = match self.user_symbol(&b[at..]) {
                Some(n) => (n, true),
                None => (utf8_len(b[at]).min(b.len() - at), false),
            };
            let i = syms.len() as isize;
            syms.push((
                at,
                n,
                i - 1,
                if at + n < b.len() { i + 1 } else { -1 },
                frozen,
            ));
            at += n;
        }
        if syms.is_empty() {
            return Vec::new();
        }
        // agenda entries: (score, -left, left, right, merged length)
        #[derive(PartialEq)]
        struct Pair(f32, isize, usize, usize, usize);
        impl Eq for Pair {}
        impl PartialOrd for Pair {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Pair {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                // higher score first, then the leftmost pair
                self.0.total_cmp(&o.0).then_with(|| self.1.cmp(&o.1))
            }
        }
        let mut agenda = BinaryHeap::new();
        let push = |syms: &Vec<(usize, usize, isize, isize, bool)>,
                    agenda: &mut BinaryHeap<Pair>,
                    l: isize,
                    r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (ls, ll, _, _, lf) = syms[l as usize];
            let (_, rl, _, _, rf) = syms[r as usize];
            if lf || rf {
                return;
            }
            let piece = &norm[ls..ls + ll + rl];
            if let Some(&id) = self.index.get(piece) {
                agenda.push(Pair(
                    self.pieces[id as usize].1,
                    -l,
                    l as usize,
                    r as usize,
                    ll + rl,
                ));
            }
        };
        for i in 1..syms.len() {
            push(&syms, &mut agenda, i as isize - 1, i as isize);
        }
        while let Some(Pair(_, _, l, r, size)) = agenda.pop() {
            let (ll, rl) = (syms[l].1, syms[r].1);
            if ll == 0 || rl == 0 || ll + rl != size {
                continue;
            }
            syms[l].1 = ll + rl;
            let rnext = syms[r].3;
            syms[l].3 = rnext;
            if rnext >= 0 {
                syms[rnext as usize].2 = l as isize;
            }
            syms[r].1 = 0;
            let lprev = syms[l].2;
            push(&syms, &mut agenda, lprev, l as isize);
            push(&syms, &mut agenda, l as isize, rnext);
        }
        let mut ids: Vec<u32> = Vec::new();
        let mut i = 0isize;
        while i >= 0 {
            let (s, n, _, next, _) = syms[i as usize];
            let id = self.piece_to_id(&norm[s..s + n]);
            // the processor folds a run of unknown pieces into one
            if !(id == self.unk && ids.last() == Some(&self.unk)) {
                ids.push(id);
            }
            i = next;
        }
        ids
    }

    /// `decode`: pieces joined, `▁` back to spaces, the dummy prefix's
    /// leading space dropped; `unk` as SentencePiece's ` ⁇ `.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for (k, &id) in ids.iter().enumerate() {
            let Some((p, _, ty)) = self.pieces.get(id as usize) else {
                continue;
            };
            let p: &str = match ty {
                PieceType::Control => continue,
                PieceType::Unknown => " \u{2047} ",
                _ => p,
            };
            let mut p = p.replace(SPACE, " ");
            if k == 0 && self.add_dummy_prefix && p.starts_with(' ') {
                p.remove(0);
            }
            s.push_str(&p);
        }
        s
    }
}

fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    }
}
