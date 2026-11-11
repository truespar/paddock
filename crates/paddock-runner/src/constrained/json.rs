//! JSON schema -> pushdown automaton: the machine `response_format`
//! (`json_object` / `json_schema`) and the JSON tool-call syntax decode
//! against. Split out of `constrained.rs` when the ATEM tool grammar crossed
//! the file-size ceiling; the seam is exactly "schema machine" vs "tool-call
//! grammar", which share only `VocabBytes` and the gate above them.

use std::sync::Arc;

use serde_json::Value;

// ---------------------------------------------------------------- schema --

#[derive(Debug)]
enum Node {
    /// bare JSON (`json_object` mode): any valid value
    Any,
    /// fixed-order object: every property required, emitted in this order
    Object {
        props: Vec<(String, usize)>,
    },
    /// a tool's `arguments`: declared properties in any order, no duplicates,
    /// every `required` one present before the closing brace. The strict
    /// subset above deliberately can't express this - it forces every
    /// property, in `required` order - and that is the wrong shape for a tool
    /// call: making a model emit optional parameters it has no value for
    /// produces a worse call, not a tighter one.
    ObjLax {
        props: Vec<LaxProp>,
    },
    Array {
        item: usize,
    },
    Str,
    EnumStr {
        variants: Vec<String>,
    },
    Number,
    Integer,
    Boolean,
    Null,
    /// `anyOf` / `type: [..]`: alternatives that each open with a different
    /// byte class, so the first byte of the value picks one and the machine
    /// stays deterministic. Flat (no union inside a union).
    Union {
        alts: Vec<usize>,
    },
}

/// The bytes a value of a node can open with. Two union alternatives in the
/// same class cannot be told apart by their first byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opens {
    Quote,
    Number,
    Bool,
    Null,
    Brace,
    Bracket,
    Anything,
}

impl Node {
    fn opens(&self) -> Opens {
        match self {
            Node::Any => Opens::Anything,
            Node::Object { .. } | Node::ObjLax { .. } => Opens::Brace,
            Node::Array { .. } => Opens::Bracket,
            Node::Str | Node::EnumStr { .. } => Opens::Quote,
            Node::Number | Node::Integer => Opens::Number,
            Node::Boolean => Opens::Bool,
            Node::Null => Opens::Null,
            Node::Union { .. } => unreachable!("unions are flat"),
        }
    }
}

impl Opens {
    fn admits(self, b: u8) -> bool {
        match self {
            Opens::Quote => b == b'"',
            Opens::Number => b == b'-' || b.is_ascii_digit(),
            Opens::Bool => b == b't' || b == b'f',
            Opens::Null => b == b'n',
            Opens::Brace => b == b'{',
            Opens::Bracket => b == b'[',
            Opens::Anything => true,
        }
    }
}

/// One property of a `Node::ObjLax`. Position in the vec is the bit index in
/// the "already emitted" mask, so the list is capped at `MAX_LAX_PROPS`.
#[derive(Debug)]
struct LaxProp {
    key: String,
    node: usize,
    required: bool,
}

/// The emitted-mask is a u32, and a tool with more parameters than this is
/// pathological anyway - the qwen grammar has refused past it since day one.
pub(super) const MAX_LAX_PROPS: usize = 32;

pub struct CompiledSchema {
    arena: Vec<Node>,
    root: usize,
}

impl CompiledSchema {
    pub fn any_json() -> Arc<CompiledSchema> {
        Arc::new(CompiledSchema {
            arena: vec![Node::Any],
            root: 0,
        })
    }

    /// Compile the strict subset; any unsupported keyword is a hard error.
    pub fn compile(schema: &Value) -> Result<Arc<CompiledSchema>, String> {
        let mut arena = Vec::new();
        let root = compile_node(schema, &mut arena)?;
        Ok(Arc::new(CompiledSchema { arena, root }))
    }
}

const UNSUPPORTED: &[&str] = &[
    "oneOf",
    "allOf",
    "not",
    "$ref",
    "$defs",
    "definitions",
    "patternProperties",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
    "additionalItems",
    "if",
    "then",
    "else",
    "const",
    "multipleOf",
];

fn compile_node(v: &Value, arena: &mut Vec<Node>) -> Result<usize, String> {
    let obj = v.as_object().ok_or("schema node must be a JSON object")?;
    for k in UNSUPPORTED {
        if obj.contains_key(*k) {
            return Err(format!(
                "unsupported JSON-schema keyword {k:?} (strict structured-output subset)"
            ));
        }
    }
    if let Some(branches) = obj.get("anyOf") {
        // anyOf beside other structure would mean the intersection of both;
        // the subset keeps it to a plain choice between the branches
        if ["type", "properties", "items", "enum"]
            .iter()
            .any(|k| obj.contains_key(*k))
        {
            return Err(
                "`anyOf` must stand alone (no `type`, `properties`, `items` or `enum` beside it)"
                    .into(),
            );
        }
        let branches = branches.as_array().ok_or("`anyOf` must be an array")?;
        if branches.is_empty() {
            return Err("`anyOf` must not be empty".into());
        }
        let mut alts = Vec::with_capacity(branches.len());
        for b in branches {
            alts.push(compile_node(b, arena)?);
        }
        return union(alts, arena);
    }
    // `type: ["string", "null"]`: OpenAI's way to say nullable; each type is
    // the same node with that one type, an `enum` split by its variants' types
    if let Some(types) = obj.get("type").and_then(Value::as_array) {
        let mut alts = Vec::with_capacity(types.len());
        for t in types {
            let t = t.as_str().ok_or("`type` entries must be strings")?;
            let mut one = obj.clone();
            one.insert("type".into(), Value::from(t));
            if let Some(e) = obj.get("enum").and_then(Value::as_array) {
                if !e.iter().all(|v| v.is_string() || v.is_null()) {
                    return Err("an `enum` takes strings and null only".into());
                }
                match t {
                    // the null alternative is the enum's null, which takes no enum of its own
                    "null" => {
                        if !e.iter().any(Value::is_null) {
                            return Err("`type` allows null but `enum` does not list it".into());
                        }
                        one.remove("enum");
                    }
                    "string" => {
                        let keep: Vec<Value> =
                            e.iter().filter(|v| v.is_string()).cloned().collect();
                        if keep.is_empty() {
                            return Err("`type` allows a string but `enum` lists none".into());
                        }
                        one.insert("enum".into(), Value::Array(keep));
                    }
                    other => {
                        return Err(format!(
                            "`enum` beside type {other:?}: only strings and null take an enum"
                        ));
                    }
                }
            }
            alts.push(compile_node(&Value::Object(one), arena)?);
        }
        if alts.is_empty() {
            return Err("`type` must not be an empty list".into());
        }
        return union(alts, arena);
    }
    // an `enum` alone says its type through its variants
    if !obj.contains_key("type")
        && let Some(e) = obj.get("enum").and_then(Value::as_array)
    {
        let mut types: Vec<&str> = Vec::new();
        for v in e {
            let t = match v {
                Value::String(_) => "string",
                Value::Null => "null",
                _ => return Err("an `enum` without `type` takes strings and null only".into()),
            };
            if !types.contains(&t) {
                types.push(t);
            }
        }
        let mut typed = obj.clone();
        typed.insert("type".into(), Value::from(types));
        return compile_node(&Value::Object(typed), arena);
    }
    let ty = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or("schema node needs a `type` (a string, or a list of them)")?;
    let node = match ty {
        "object" => {
            let props = obj
                .get("properties")
                .and_then(Value::as_object)
                .ok_or("object schema needs `properties`")?;
            if let Some(ap) = obj.get("additionalProperties")
                && ap.as_bool() != Some(false)
            {
                return Err("object schema needs `additionalProperties: false`".into());
            }
            // emission order = `required` order; every property must be in it
            let required: Vec<&str> = obj
                .get("required")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if required.len() != props.len() || !required.iter().all(|k| props.contains_key(*k)) {
                return Err(
                    "strict mode: `required` must list every property (emission order)".into(),
                );
            }
            let mut compiled = Vec::with_capacity(required.len());
            for key in required {
                if key.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20) {
                    return Err(format!("property key {key:?} needs escaping (unsupported)"));
                }
                let child = compile_node(&props[key], arena)?;
                compiled.push((key.to_owned(), child));
            }
            Node::Object { props: compiled }
        }
        "array" => {
            let items = obj.get("items").ok_or("array schema needs `items`")?;
            let child = compile_node(items, arena)?;
            Node::Array { item: child }
        }
        "string" => match obj.get("enum") {
            None => Node::Str,
            Some(e) => {
                let variants: Vec<String> = e
                    .as_array()
                    .ok_or("`enum` must be an array")?
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or("enum variants must be strings")
                    })
                    .collect::<Result<_, _>>()?;
                if variants.is_empty() {
                    return Err("`enum` must not be empty".into());
                }
                for s in &variants {
                    if s.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20) {
                        return Err(format!("enum variant {s:?} needs escaping (unsupported)"));
                    }
                }
                Node::EnumStr { variants }
            }
        },
        "number" => Node::Number,
        "integer" => Node::Integer,
        "boolean" => Node::Boolean,
        "null" => Node::Null,
        other => return Err(format!("unsupported schema type {other:?}")),
    };
    arena.push(node);
    Ok(arena.len() - 1)
}

/// What two union alternatives that open alike come to.
enum Merge {
    /// the one already picked covers the new one (integer within number)
    Keep,
    /// the new one covers the picked one
    Take,
    /// two string enums: one with both variant lists
    Enum(Vec<String>),
    /// both open with the same byte and neither covers the other
    Clash,
}

/// One node for "any of these": unions flattened, an alternative another one
/// covers folded into it, and refused when two alternatives still open with
/// the same byte - the first byte has to decide, or the machine would need to
/// follow several branches at once.
fn union(alts: Vec<usize>, arena: &mut Vec<Node>) -> Result<usize, String> {
    let mut flat = Vec::with_capacity(alts.len());
    for a in alts {
        match &arena[a] {
            Node::Union { alts } => flat.extend(alts.iter().copied()),
            _ => flat.push(a),
        }
    }
    let mut picked: Vec<usize> = Vec::with_capacity(flat.len());
    for a in flat {
        let opens = arena[a].opens();
        if opens == Opens::Anything {
            return Ok(a);
        }
        let Some(pos) = picked.iter().position(|&p| arena[p].opens() == opens) else {
            picked.push(a);
            continue;
        };
        let merge = match (&arena[picked[pos]], &arena[a]) {
            (Node::Null, Node::Null)
            | (Node::Boolean, Node::Boolean)
            | (Node::Number, Node::Number | Node::Integer)
            | (Node::Integer, Node::Integer)
            | (Node::Str, Node::Str | Node::EnumStr { .. }) => Merge::Keep,
            (Node::Integer, Node::Number) | (Node::EnumStr { .. }, Node::Str) => Merge::Take,
            (Node::EnumStr { variants: x }, Node::EnumStr { variants: y }) => {
                let mut v = x.clone();
                v.extend(y.iter().filter(|s| !x.contains(s)).cloned());
                Merge::Enum(v)
            }
            _ => Merge::Clash,
        };
        match merge {
            Merge::Keep => {}
            Merge::Take => picked[pos] = a,
            Merge::Enum(variants) => {
                arena.push(Node::EnumStr { variants });
                picked[pos] = arena.len() - 1;
            }
            Merge::Clash => {
                let what = match opens {
                    Opens::Brace => "two objects",
                    Opens::Bracket => "two arrays",
                    _ => "two alternatives of one kind",
                };
                return Err(format!(
                    "`anyOf` / `type` alternatives must each open with a different JSON type; \
                     {what} cannot be told apart by their first byte (unsupported)"
                ));
            }
        }
    }
    if picked.len() == 1 {
        return Ok(picked[0]);
    }
    arena.push(Node::Union { alts: picked });
    Ok(arena.len() - 1)
}

// --------------------------------------------- lenient (tool-args) compile --

/// Compile a tool's `parameters` into the grammar for its `arguments` object.
///
/// Unlike `compile_node` this never fails. A tool schema is declared for the
/// MODEL's benefit, not as the caller's output contract, so 400-ing a whole
/// request because one parameter used `anyOf` would refuse a call we can still
/// make. Anything the subset can't model degrades to free JSON for that value
/// - and even fully degraded this is strictly tighter than the qwen XML
///   grammar, which constrains no value at all.
///
/// `parameters` absent, or present with no `properties`, is how both the
/// OpenAI and Anthropic tool schemas spell a zero-argument tool, so the root
/// object is then the empty one: `{}` and nothing else.
pub(super) fn compile_tool_args(parameters: Option<&Value>) -> Arc<CompiledSchema> {
    let mut arena = Vec::new();
    let obj = parameters.and_then(Value::as_object);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);
    let root = match props {
        Some(p) => match lax_object(p, obj.and_then(|o| o.get("required")), &mut arena) {
            Some(node) => {
                arena.push(node);
                arena.len() - 1
            }
            None => any_slot(&mut arena),
        },
        None => {
            arena.push(Node::ObjLax { props: Vec::new() });
            arena.len() - 1
        }
    };
    Arc::new(CompiledSchema { arena, root })
}

fn any_slot(arena: &mut Vec<Node>) -> usize {
    arena.push(Node::Any);
    arena.len() - 1
}

/// `properties` + `required` -> a lax object node, or `None` when the whole
/// object has to degrade to free JSON (a key that would need JSON escaping
/// can't be a grammar literal, and there is no honest way to require a
/// property the grammar cannot spell).
fn lax_object(
    props: &serde_json::Map<String, Value>,
    required: Option<&Value>,
    arena: &mut Vec<Node>,
) -> Option<Node> {
    if props.is_empty() || props.len() > MAX_LAX_PROPS {
        return None;
    }
    let req: Vec<&str> = required
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut out = Vec::with_capacity(props.len());
    for (key, schema) in props {
        if key.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20) {
            return None;
        }
        out.push(LaxProp {
            key: key.clone(),
            node: compile_lax(schema, arena),
            required: req.contains(&key.as_str()),
        });
    }
    Some(Node::ObjLax { props: out })
}

fn compile_lax(v: &Value, arena: &mut Vec<Node>) -> usize {
    let Some(obj) = v.as_object() else {
        return any_slot(arena);
    };
    if UNSUPPORTED.iter().any(|k| obj.contains_key(*k)) {
        return any_slot(arena);
    }
    let Some(ty) = obj.get("type").and_then(Value::as_str) else {
        // no `type`, or the union spelling (`type: ["string","null"]`)
        return any_slot(arena);
    };
    let node = match ty {
        // A NESTED object with no declared properties is free-form - the
        // "zero arguments" reading above only makes sense at the root, where
        // the property list is the tool's whole signature.
        "object" => match obj.get("properties").and_then(Value::as_object) {
            Some(p) => match lax_object(p, obj.get("required"), arena) {
                Some(n) => n,
                None => return any_slot(arena),
            },
            None => return any_slot(arena),
        },
        "array" => match obj.get("items") {
            Some(items) => Node::Array {
                item: compile_lax(items, arena),
            },
            None => return any_slot(arena),
        },
        "string" => match obj.get("enum") {
            None => Node::Str,
            Some(e) => {
                let ok = e.as_array().map(|a| {
                    a.iter()
                        .map(|v| v.as_str().map(str::to_owned))
                        .collect::<Option<Vec<String>>>()
                });
                match ok.flatten() {
                    Some(variants)
                        if !variants.is_empty()
                            && !variants.iter().any(|s| {
                                s.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20)
                            }) =>
                    {
                        Node::EnumStr { variants }
                    }
                    _ => return any_slot(arena),
                }
            }
        },
        "number" => Node::Number,
        "integer" => Node::Integer,
        "boolean" => Node::Boolean,
        "null" => Node::Null,
        _ => return any_slot(arena),
    };
    arena.push(node);
    arena.len() - 1
}

// ------------------------------------------------------------- JSON PDA --

const MAX_WS_RUN: u8 = 64;

/// Total byte cap per JSON number. Unbounded digit states are a known
/// grammar-greedy pathology (the model spirals in `2.164999999...` because
/// one more digit stays the argmax); 24 bytes round-trips every f64/i64.
const MAX_NUM_LEN: u8 = 24;

#[derive(Clone, Debug, PartialEq)]
enum NumPhase {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl NumPhase {
    fn complete(&self) -> bool {
        matches!(
            self,
            NumPhase::Zero | NumPhase::Int | NumPhase::Frac | NumPhase::Exp
        )
    }
}

#[derive(Clone, Debug)]
enum StrPhase {
    Body,
    Esc,
    Uni(u8),
}

#[derive(Clone, Debug)]
enum Frame {
    /// expecting the first byte of a value of this node
    Value {
        node: usize,
    },
    /// schema object: keys forced in order
    Obj {
        node: usize,
        idx: usize,
        phase: ObjPhase,
    },
    /// tool-args object: keys in any order, `emitted` = the bits already used
    ObjLax {
        node: usize,
        emitted: u32,
        phase: LaxPhase,
    },
    /// schema array
    Arr {
        node: usize,
        first: bool,
    },
    /// free-form object (Any mode). `any` is the arena index of the `Node::Any`
    /// that opened it - nested free values reuse it, and it is not necessarily
    /// the root (a tool-args schema can hang one under a typed property).
    AnyObj {
        any: usize,
        phase: AnyObjPhase,
    },
    /// free-form array (Any mode)
    AnyArr {
        any: usize,
        first: bool,
    },
    Str {
        phase: StrPhase,
    },
    EnumStr {
        node: usize,
        alive: Vec<bool>,
        pos: usize,
    },
    Num {
        integer_only: bool,
        phase: NumPhase,
        len: u8,
    },
    Lit {
        text: &'static [u8],
        pos: usize,
    },
}

#[derive(Clone, Debug)]
enum ObjPhase {
    /// expect '"' opening key `idx` (or '}' when the object has no props)
    KeyOpen,
    /// inside the forced key literal
    Key(usize),
    /// expect ':'
    Colon,
    /// child value in flight (frame above)
    InValue,
    /// expect ',' (more props) or '}' (idx == props.len())
    Next,
}

#[derive(Clone, Debug)]
enum LaxPhase {
    /// expect '"' opening some not-yet-emitted key (or '}' when this is the
    /// first key position and nothing is still required)
    KeyOpen { first: bool },
    /// inside the key: `alive` is a bitmask over the not-yet-emitted
    /// properties still matching, `pos` the byte offset into their keys.
    /// A candidate matches its key plus the closing quote, which is what
    /// makes the candidate set prefix-free even for keys like `a` and `ab`.
    Key { alive: u32, pos: usize },
    /// expect ':' before property `idx`'s value
    Colon { idx: usize },
    /// child value in flight (frame above)
    InValue,
    /// expect ',' (a key is still available) or '}' (nothing still required)
    Next,
}

#[derive(Clone, Debug)]
enum AnyObjPhase {
    /// expect '"' (or '}' when first)
    KeyOpen {
        first: bool,
    },
    Key(StrPhase),
    Colon,
    InValue,
    Next,
}

/// Byte-level, schema-directed incremental JSON recognizer. Cloned per
/// candidate token test - the state is a small frame stack.
#[derive(Clone)]
pub struct JsonMachine {
    schema: Arc<CompiledSchema>,
    stack: Vec<Frame>,
    ws_run: u8,
    done: bool,
}

impl JsonMachine {
    pub fn new(schema: Arc<CompiledSchema>) -> JsonMachine {
        let root = schema.root;
        JsonMachine {
            schema,
            stack: vec![Frame::Value { node: root }],
            ws_run: 0,
            done: false,
        }
    }

    pub fn may_stop(&self) -> bool {
        if self.done {
            return true;
        }
        // a root-level number completes implicitly at end of output
        matches!(
            self.stack.as_slice(),
            [Frame::Num { phase, .. }] if phase.complete()
        )
    }

    /// Advance by one byte; false = illegal.
    pub fn feed(&mut self, b: u8) -> bool {
        // Nothing after the root - not even whitespace. Once the value is
        // complete, may_stop() is true and the only way forward should be a
        // registered stop token. Trailing whitespace used to be legal here
        // (capped), and gpt-oss showed why that is wrong: its preferred
        // end-of-message token `<|end|>` is a special the constraint masks and
        // deliberately not an engine stop (unconstrained Harmony continues
        // with another channel after it), so at temp 0 the next-best legal
        // token was whitespace - 64 bytes of `\r\n` padding in the client's
        // JSON until the cap outlawed it.
        // Rejecting everything makes the sampler choose among stop tokens
        // immediately, so the content ends exactly at the closing byte on
        // every dialect.
        if self.done {
            return false;
        }
        // whitespace between tokens (capped run so "legal forever" can't happen)
        if self.ws_legal() && matches!(b, b' ' | b'\n' | b'\t' | b'\r') {
            if self.ws_run >= MAX_WS_RUN {
                return false;
            }
            self.ws_run += 1;
            return true;
        }
        self.ws_run = 0;
        // a byte may terminate an inner frame (number) and then belong to the
        // parent - loop redispatches it after the pop
        loop {
            let Some(top) = self.stack.last_mut() else {
                return false;
            };
            match top {
                Frame::Value { node } => {
                    let mut node = *node;
                    // a union is decided by this first byte: its alternatives open differently
                    if let Node::Union { alts } = &self.schema.arena[node] {
                        match alts
                            .iter()
                            .find(|&&a| self.schema.arena[a].opens().admits(b))
                        {
                            Some(&a) => node = a,
                            None => return false,
                        }
                    }
                    let frame = match &self.schema.arena[node] {
                        Node::Union { .. } => unreachable!("unions are flat"),
                        Node::Any => match b {
                            b'{' => Frame::AnyObj {
                                any: node,
                                phase: AnyObjPhase::KeyOpen { first: true },
                            },
                            b'[' => Frame::AnyArr {
                                any: node,
                                first: true,
                            },
                            b'"' => Frame::Str {
                                phase: StrPhase::Body,
                            },
                            b'-' => Frame::Num {
                                integer_only: false,
                                phase: NumPhase::Minus,
                                len: 1,
                            },
                            b'0' => Frame::Num {
                                integer_only: false,
                                phase: NumPhase::Zero,
                                len: 1,
                            },
                            b'1'..=b'9' => Frame::Num {
                                integer_only: false,
                                phase: NumPhase::Int,
                                len: 1,
                            },
                            b't' => Frame::Lit {
                                text: b"true",
                                pos: 1,
                            },
                            b'f' => Frame::Lit {
                                text: b"false",
                                pos: 1,
                            },
                            b'n' => Frame::Lit {
                                text: b"null",
                                pos: 1,
                            },
                            _ => return false,
                        },
                        Node::Object { .. } => {
                            if b != b'{' {
                                return false;
                            }
                            Frame::Obj {
                                node,
                                idx: 0,
                                phase: ObjPhase::KeyOpen,
                            }
                        }
                        Node::ObjLax { .. } => {
                            if b != b'{' {
                                return false;
                            }
                            Frame::ObjLax {
                                node,
                                emitted: 0,
                                phase: LaxPhase::KeyOpen { first: true },
                            }
                        }
                        Node::Array { .. } => {
                            if b != b'[' {
                                return false;
                            }
                            Frame::Arr { node, first: true }
                        }
                        Node::Str => {
                            if b != b'"' {
                                return false;
                            }
                            Frame::Str {
                                phase: StrPhase::Body,
                            }
                        }
                        Node::EnumStr { variants } => {
                            if b != b'"' {
                                return false;
                            }
                            Frame::EnumStr {
                                node,
                                alive: vec![true; variants.len()],
                                pos: 0,
                            }
                        }
                        Node::Number | Node::Integer => {
                            let integer_only = matches!(self.schema.arena[node], Node::Integer);
                            let phase = match b {
                                b'-' => NumPhase::Minus,
                                b'0' => NumPhase::Zero,
                                b'1'..=b'9' => NumPhase::Int,
                                _ => return false,
                            };
                            Frame::Num {
                                integer_only,
                                phase,
                                len: 1,
                            }
                        }
                        Node::Boolean => match b {
                            b't' => Frame::Lit {
                                text: b"true",
                                pos: 1,
                            },
                            b'f' => Frame::Lit {
                                text: b"false",
                                pos: 1,
                            },
                            _ => return false,
                        },
                        Node::Null => {
                            if b != b'n' {
                                return false;
                            }
                            Frame::Lit {
                                text: b"null",
                                pos: 1,
                            }
                        }
                    };
                    *self.stack.last_mut().expect("top") = frame;
                    return true;
                }

                Frame::Lit { text, pos } => {
                    if *pos < text.len() && b == text[*pos] {
                        *pos += 1;
                        if *pos == text.len() {
                            self.complete_value();
                        }
                        return true;
                    }
                    return false;
                }

                Frame::Str { phase } => match phase.clone() {
                    StrPhase::Body => match b {
                        b'"' => {
                            self.complete_value();
                            return true;
                        }
                        b'\\' => {
                            *phase = StrPhase::Esc;
                            return true;
                        }
                        0x20.. => return true,
                        _ => return false,
                    },
                    StrPhase::Esc => match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            *phase = StrPhase::Body;
                            return true;
                        }
                        b'u' => {
                            *phase = StrPhase::Uni(4);
                            return true;
                        }
                        _ => return false,
                    },
                    StrPhase::Uni(left) => {
                        if b.is_ascii_hexdigit() {
                            *phase = if left == 1 {
                                StrPhase::Body
                            } else {
                                StrPhase::Uni(left - 1)
                            };
                            return true;
                        }
                        return false;
                    }
                },

                Frame::EnumStr { node, alive, pos } => {
                    let variants = match &self.schema.arena[*node] {
                        Node::EnumStr { variants } => variants,
                        _ => unreachable!(),
                    };
                    if b == b'"' {
                        // closing quote: legal iff some alive variant is fully matched
                        let closed = alive
                            .iter()
                            .zip(variants)
                            .any(|(a, v)| *a && v.len() == *pos);
                        if !closed {
                            return false;
                        }
                        self.complete_value();
                        return true;
                    }
                    let mut any = false;
                    for (a, v) in alive.iter_mut().zip(variants) {
                        if *a && v.as_bytes().get(*pos) == Some(&b) {
                            any = true;
                        } else {
                            *a = false;
                        }
                    }
                    if !any {
                        return false;
                    }
                    *pos += 1;
                    return true;
                }

                Frame::Num {
                    integer_only,
                    phase,
                    len,
                } => {
                    let io = *integer_only;
                    // digit-cap: once a syntactically complete number hits the
                    // cap, only the implicit terminator remains legal - the
                    // byte redispatches in the parent (digits then reject)
                    if *len >= MAX_NUM_LEN && phase.complete() {
                        self.complete_value();
                        if self.done {
                            return false;
                        }
                        continue;
                    }
                    let next = match (phase.clone(), b) {
                        (NumPhase::Minus, b'0') => Some(NumPhase::Zero),
                        (NumPhase::Minus, b'1'..=b'9') => Some(NumPhase::Int),
                        (NumPhase::Int, b'0'..=b'9') => Some(NumPhase::Int),
                        (NumPhase::Zero | NumPhase::Int, b'.') if !io => Some(NumPhase::Dot),
                        (NumPhase::Dot, b'0'..=b'9') => Some(NumPhase::Frac),
                        (NumPhase::Frac, b'0'..=b'9') => Some(NumPhase::Frac),
                        (NumPhase::Zero | NumPhase::Int | NumPhase::Frac, b'e' | b'E') if !io => {
                            Some(NumPhase::E)
                        }
                        (NumPhase::E, b'+' | b'-') => Some(NumPhase::ESign),
                        (NumPhase::E | NumPhase::ESign | NumPhase::Exp, b'0'..=b'9') => {
                            Some(NumPhase::Exp)
                        }
                        _ => None,
                    };
                    match next {
                        Some(p) => {
                            *phase = p;
                            *len += 1;
                            return true;
                        }
                        None => {
                            // number ends implicitly; redispatch b in parent
                            if !phase.complete() {
                                return false;
                            }
                            self.complete_value();
                            if self.done {
                                return false; // nothing may follow the root but ws
                            }
                            continue;
                        }
                    }
                }

                Frame::Obj { node, idx, phase } => {
                    let props_len = match &self.schema.arena[*node] {
                        Node::Object { props } => props.len(),
                        _ => unreachable!(),
                    };
                    match phase.clone() {
                        ObjPhase::KeyOpen => {
                            if props_len == 0 {
                                if b == b'}' {
                                    self.complete_value();
                                    return true;
                                }
                                return false;
                            }
                            if b == b'"' {
                                *phase = ObjPhase::Key(0);
                                return true;
                            }
                            return false;
                        }
                        ObjPhase::Key(pos) => {
                            let key = match &self.schema.arena[*node] {
                                Node::Object { props } => props[*idx].0.as_bytes(),
                                _ => unreachable!(),
                            };
                            if pos < key.len() {
                                if b == key[pos] {
                                    *phase = ObjPhase::Key(pos + 1);
                                    return true;
                                }
                                return false;
                            }
                            if b == b'"' {
                                *phase = ObjPhase::Colon;
                                return true;
                            }
                            return false;
                        }
                        ObjPhase::Colon => {
                            if b == b':' {
                                let child = match &self.schema.arena[*node] {
                                    Node::Object { props } => props[*idx].1,
                                    _ => unreachable!(),
                                };
                                *phase = ObjPhase::InValue;
                                self.stack.push(Frame::Value { node: child });
                                return true;
                            }
                            return false;
                        }
                        ObjPhase::InValue => unreachable!("child frame handles bytes"),
                        ObjPhase::Next => {
                            if *idx + 1 < props_len {
                                if b == b',' {
                                    *idx += 1;
                                    *phase = ObjPhase::KeyOpen;
                                    return true;
                                }
                                return false;
                            }
                            if b == b'}' {
                                self.complete_value();
                                return true;
                            }
                            return false;
                        }
                    }
                }

                Frame::ObjLax {
                    node,
                    emitted,
                    phase,
                } => {
                    let props = match &self.schema.arena[*node] {
                        Node::ObjLax { props } => props,
                        _ => unreachable!(),
                    };
                    // bit i = property i already emitted; `left` is what may
                    // still be opened, `owed` whether a required one is missing
                    let all: u32 = mask(props.len());
                    let left = all & !*emitted;
                    let owed = props
                        .iter()
                        .enumerate()
                        .any(|(i, p)| p.required && *emitted & (1 << i) == 0);
                    match phase.clone() {
                        LaxPhase::KeyOpen { first } => {
                            if b == b'"' && left != 0 {
                                *phase = LaxPhase::Key {
                                    alive: left,
                                    pos: 0,
                                };
                                return true;
                            }
                            // `{}` only at the first key position - a dangling
                            // comma before '}' is not JSON
                            if b == b'}' && first && !owed {
                                self.complete_value();
                                return true;
                            }
                            return false;
                        }
                        LaxPhase::Key { alive, pos } => {
                            let mut next = 0u32;
                            let mut matched = None;
                            for (i, p) in props.iter().enumerate() {
                                if alive & (1 << i) == 0 {
                                    continue;
                                }
                                let k = p.key.as_bytes();
                                // the closing quote is part of the candidate
                                let want = if pos < k.len() { k[pos] } else { b'"' };
                                if want != b {
                                    continue;
                                }
                                next |= 1 << i;
                                if pos == k.len() {
                                    matched = Some(i);
                                }
                            }
                            if next == 0 {
                                return false;
                            }
                            match matched {
                                // keys are unique and the quote terminates
                                // them, so at most one candidate completes here
                                Some(i) => {
                                    *emitted |= 1 << i;
                                    *phase = LaxPhase::Colon { idx: i };
                                }
                                None => {
                                    *phase = LaxPhase::Key {
                                        alive: next,
                                        pos: pos + 1,
                                    }
                                }
                            }
                            return true;
                        }
                        LaxPhase::Colon { idx } => {
                            if b == b':' {
                                let child = match &self.schema.arena[*node] {
                                    Node::ObjLax { props } => props[idx].node,
                                    _ => unreachable!(),
                                };
                                *phase = LaxPhase::InValue;
                                self.stack.push(Frame::Value { node: child });
                                return true;
                            }
                            return false;
                        }
                        LaxPhase::InValue => unreachable!("child frame handles bytes"),
                        LaxPhase::Next => {
                            if b == b',' && left != 0 {
                                *phase = LaxPhase::KeyOpen { first: false };
                                return true;
                            }
                            if b == b'}' && !owed {
                                self.complete_value();
                                return true;
                            }
                            return false;
                        }
                    }
                }

                Frame::Arr { node, first } => {
                    let item = match &self.schema.arena[*node] {
                        Node::Array { item } => *item,
                        _ => unreachable!(),
                    };
                    if *first {
                        if b == b']' {
                            self.complete_value();
                            return true;
                        }
                        // first item value begins with this byte
                        *first = false;
                        self.stack.push(Frame::Value { node: item });
                        continue;
                    }
                    match b {
                        b',' => {
                            self.stack.push(Frame::Value { node: item });
                            return true;
                        }
                        b']' => {
                            self.complete_value();
                            return true;
                        }
                        _ => return false,
                    }
                }

                Frame::AnyObj { any, phase } => {
                    let any = *any;
                    match phase.clone() {
                        AnyObjPhase::KeyOpen { first } => {
                            if first && b == b'}' {
                                self.complete_value();
                                return true;
                            }
                            if b == b'"' {
                                *phase = AnyObjPhase::Key(StrPhase::Body);
                                return true;
                            }
                            return false;
                        }
                        AnyObjPhase::Key(sp) => match key_step(sp, b) {
                            KeyStep::Continue(next) => {
                                *phase = AnyObjPhase::Key(next);
                                return true;
                            }
                            KeyStep::Closed => {
                                *phase = AnyObjPhase::Colon;
                                return true;
                            }
                            KeyStep::Reject => return false,
                        },
                        AnyObjPhase::Colon => {
                            if b == b':' {
                                *phase = AnyObjPhase::InValue;
                                self.stack.push(Frame::Value { node: any });
                                return true;
                            }
                            return false;
                        }
                        AnyObjPhase::InValue => unreachable!(),
                        AnyObjPhase::Next => match b {
                            b',' => {
                                *phase = AnyObjPhase::KeyOpen { first: false };
                                return true;
                            }
                            b'}' => {
                                self.complete_value();
                                return true;
                            }
                            _ => return false,
                        },
                    }
                }

                Frame::AnyArr { any, first } => {
                    let any = *any;
                    if *first {
                        if b == b']' {
                            self.complete_value();
                            return true;
                        }
                        *first = false;
                        self.stack.push(Frame::Value { node: any });
                        continue;
                    }
                    match b {
                        b',' => {
                            self.stack.push(Frame::Value { node: any });
                            return true;
                        }
                        b']' => {
                            self.complete_value();
                            return true;
                        }
                        _ => return false,
                    }
                }
            }
        }
    }

    /// pop the completed value frame and advance the parent
    fn complete_value(&mut self) {
        self.stack.pop();
        match self.stack.last_mut() {
            None => self.done = true,
            Some(Frame::Obj { phase, .. }) => *phase = ObjPhase::Next,
            Some(Frame::ObjLax { phase, .. }) => *phase = LaxPhase::Next,
            Some(Frame::AnyObj { phase, .. }) => *phase = AnyObjPhase::Next,
            Some(Frame::Arr { .. } | Frame::AnyArr { .. }) => {}
            Some(other) => unreachable!("value completed under {other:?}"),
        }
    }

    /// whitespace is legal between tokens, not inside strings/numbers/keys
    fn ws_legal(&self) -> bool {
        match self.stack.last() {
            // empty stack = done, and feed() rejects everything there before
            // asking us - after the root the output ENDS (see feed)
            None => false,
            Some(Frame::Value { .. }) => true,
            Some(Frame::Obj { phase, .. }) => !matches!(phase, ObjPhase::Key(_)),
            Some(Frame::ObjLax { phase, .. }) => !matches!(phase, LaxPhase::Key { .. }),
            Some(Frame::AnyObj { phase, .. }) => !matches!(phase, AnyObjPhase::Key(_)),
            Some(Frame::Arr { .. } | Frame::AnyArr { .. }) => true,
            Some(Frame::Str { .. } | Frame::EnumStr { .. } | Frame::Lit { .. }) => false,
            // ws terminates a complete number (redispatch handles it - but a
            // simple "legal between tokens" answer here is wrong for
            // incomplete numbers, so numbers say no and feed() redispatches)
            Some(Frame::Num { .. }) => false,
        }
    }
}

/// low `n` bits set (`n <= 32`)
fn mask(n: usize) -> u32 {
    debug_assert!(n <= MAX_LAX_PROPS);
    if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

enum KeyStep {
    Continue(StrPhase),
    Closed,
    Reject,
}

fn key_step(sp: StrPhase, b: u8) -> KeyStep {
    match sp {
        StrPhase::Body => match b {
            b'"' => KeyStep::Closed,
            b'\\' => KeyStep::Continue(StrPhase::Esc),
            0x20.. => KeyStep::Continue(StrPhase::Body),
            _ => KeyStep::Reject,
        },
        StrPhase::Esc => match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                KeyStep::Continue(StrPhase::Body)
            }
            b'u' => KeyStep::Continue(StrPhase::Uni(4)),
            _ => KeyStep::Reject,
        },
        StrPhase::Uni(left) => {
            if b.is_ascii_hexdigit() {
                if left == 1 {
                    KeyStep::Continue(StrPhase::Body)
                } else {
                    KeyStep::Continue(StrPhase::Uni(left - 1))
                }
            } else {
                KeyStep::Reject
            }
        }
    }
}

/// Compile one lenient value schema and hand back a ready `CompiledSchema` -
/// the XML tool syntaxes need this for a typed parameter value, and the arena
/// internals stay in this module.
pub(super) fn compile_lax_schema(schema: &Value) -> Arc<CompiledSchema> {
    let mut arena = Vec::new();
    let root = compile_lax(schema, &mut arena);
    Arc::new(CompiledSchema { arena, root })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn feed_all(m: &mut JsonMachine, s: &str) -> bool {
        s.bytes().all(|b| m.feed(b))
    }

    fn weather_schema() -> Arc<CompiledSchema> {
        CompiledSchema::compile(&json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "sunny": {"type": "boolean"}
            },
            "required": ["city", "days", "sunny"]
        }))
        .expect("compile")
    }

    #[test]
    fn schema_object_happy_path() {
        let mut m = JsonMachine::new(weather_schema());
        assert!(feed_all(
            &mut m,
            "{\"city\": \"Oslo\", \"days\": 3, \"sunny\": true}"
        ));
        assert!(m.may_stop());
    }

    #[test]
    fn schema_enforces_key_order_and_names() {
        let mut m = JsonMachine::new(weather_schema());
        assert!(feed_all(&mut m, "{\"c"));
        assert!(!m.feed(b'x'), "wrong key byte must be rejected");
        let mut m = JsonMachine::new(weather_schema());
        assert!(
            !feed_all(&mut m, "{\"days\""),
            "keys must come in required order"
        );
    }

    #[test]
    fn schema_rejects_wrong_value_types() {
        let mut m = JsonMachine::new(weather_schema());
        assert!(feed_all(&mut m, "{\"city\": "));
        assert!(!m.feed(b'3'), "string field cannot start with a digit");
        let mut m = JsonMachine::new(weather_schema());
        assert!(feed_all(&mut m, "{\"city\": \"x\", \"days\": 3"));
        assert!(!m.feed(b'.'), "integer must not take a fraction");
    }

    #[test]
    fn bare_json_object_mode() {
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(
            &mut m,
            "{\"a\": [1, 2.5, {\"b\": null}], \"c\\u0041\": \"x\\n\"}"
        ));
        assert!(m.may_stop());
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, "[true, false"));
        assert!(!m.may_stop(), "unclosed array may not stop");
        assert!(m.feed(b']'));
        assert!(m.may_stop());
    }

    #[test]
    fn root_number_may_stop_but_extend() {
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, "42"));
        assert!(m.may_stop());
        assert!(m.feed(b'5'), "root number may still extend");
        let mut m2 = JsonMachine::new(CompiledSchema::any_json());
        assert!(m2.feed(b'-'));
        assert!(!m2.may_stop(), "bare minus is not a number");
    }

    /// Once the root value completes, nothing is legal - not even whitespace.
    /// The gpt-oss lane showed why: its end-of-message `<|end|>`
    /// is a masked special and deliberately not an engine stop, so with
    /// trailing ws legal the model padded `\r\n` to the cap inside the
    /// client's JSON. Rejecting everything forces a registered stop token as
    /// the very next sample.
    #[test]
    fn nothing_is_legal_after_the_root_not_even_whitespace() {
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, "{\"a\": 1}"));
        assert!(m.may_stop());
        for b in *b" \n\r\tx{" {
            assert!(
                !m.clone().feed(b),
                "byte {b:?} must be illegal after the root"
            );
        }
    }

    #[test]
    fn number_digit_spiral_is_capped() {
        // the observed pathology: greedy + grammar spiraling in 2.16499999...
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, "{\"x\": 2.164"));
        for _ in 0..MAX_NUM_LEN {
            m.feed(b'9'); // some accepted until the cap...
        }
        assert!(!m.feed(b'9'), "digits past the cap must be illegal");
        assert!(m.feed(b'}'), "the terminator must stay legal");
        assert!(m.may_stop());
    }

    #[test]
    fn whitespace_is_capped() {
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, "{\"k\":"));
        for _ in 0..MAX_WS_RUN {
            assert!(m.feed(b' '));
        }
        assert!(!m.feed(b' '), "unbounded whitespace must be rejected");
        assert!(m.feed(b'1'));
    }

    #[test]
    fn string_escapes() {
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, r#""a\"b\\cå""#));
        assert!(m.may_stop());
        let mut m = JsonMachine::new(CompiledSchema::any_json());
        assert!(feed_all(&mut m, "\"a\\"));
        assert!(!m.feed(b'x'), "bad escape must be rejected");
    }

    #[test]
    fn enum_strings() {
        let schema = CompiledSchema::compile(&json!({
            "type": "string", "enum": ["celsius", "fahrenheit"]
        }))
        .expect("compile");
        let mut m = JsonMachine::new(schema.clone());
        assert!(feed_all(&mut m, "\"celsius\""));
        assert!(m.may_stop());
        let mut m = JsonMachine::new(schema.clone());
        assert!(feed_all(&mut m, "\"c"));
        assert!(!m.feed(b'x'), "off-variant byte rejected");
        let mut m = JsonMachine::new(schema);
        assert!(feed_all(&mut m, "\"celsi"));
        assert!(!m.feed(b'"'), "cannot close mid-variant");
    }

    #[test]
    fn unsupported_keywords_are_rejected() {
        for schema in [
            json!({"type": "object", "properties": {}, "required": [], "anyOf": []}),
            json!({"type": "string", "pattern": "x"}),
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": []}),
            json!({"type": "array"}),
        ] {
            assert!(CompiledSchema::compile(&schema).is_err(), "{schema}");
        }
    }

    fn compile(v: Value) -> Arc<CompiledSchema> {
        CompiledSchema::compile(&v).expect("compile")
    }

    fn accepts(schema: &Arc<CompiledSchema>, s: &str) -> bool {
        let mut m = JsonMachine::new(schema.clone());
        feed_all(&mut m, s) && m.may_stop()
    }

    /// OpenAI's nullable field: `type: [T, "null"]`, inside a strict object
    #[test]
    fn nullable_by_type_list() {
        let s = compile(json!({
            "type": "object", "additionalProperties": false,
            "properties": {"phone": {"type": ["string", "null"]}, "employees": {"type": ["integer", "null"]}},
            "required": ["phone", "employees"]
        }));
        assert!(accepts(&s, r#"{"phone": "+46 8 123", "employees": 12}"#));
        assert!(accepts(&s, r#"{"phone": null, "employees": null}"#));
        assert!(!accepts(&s, r#"{"phone": 5, "employees": null}"#));
        assert!(!accepts(&s, r#"{"phone": null, "employees": 1.5}"#));
        let mut m = JsonMachine::new(s);
        assert!(feed_all(&mut m, r#"{"phone": n"#));
        assert!(!m.feed(b'x'), "after n only null may follow");
    }

    #[test]
    fn nullable_by_any_of() {
        let s = compile(json!({"anyOf": [
            {"type": "object", "additionalProperties": false, "properties": {"a": {"type": "boolean"}}, "required": ["a"]},
            {"type": "null"}
        ]}));
        assert!(accepts(&s, r#"{"a": true}"#));
        assert!(accepts(&s, "null"));
        assert!(!accepts(&s, "true"));
    }

    /// pydantic's Optional[Literal[...]] and OpenAI's own nullable enum
    #[test]
    fn nullable_enums() {
        for s in [
            compile(json!({"type": ["string", "null"], "enum": ["F", "C", null]})),
            compile(json!({"anyOf": [{"type": "string", "enum": ["F", "C"]}, {"type": "null"}]})),
            compile(json!({"enum": ["F", "C", null]})),
        ] {
            assert!(accepts(&s, r#""F""#));
            assert!(accepts(&s, "null"));
            assert!(!accepts(&s, r#""K""#));
        }
        assert!(
            CompiledSchema::compile(&json!({"type": ["string", "null"], "enum": ["F"]})).is_err(),
            "null allowed but not listed"
        );
    }

    #[test]
    fn enum_without_type_is_a_string_enum() {
        let s = compile(json!({"enum": ["sales", "support"]}));
        assert!(accepts(&s, r#""sales""#));
        assert!(!accepts(&s, r#""spam""#));
        assert!(!accepts(&s, "null"));
        assert!(CompiledSchema::compile(&json!({"enum": [1, 2]})).is_err());
    }

    #[test]
    fn covered_alternatives_fold_together() {
        let s = compile(json!({"anyOf": [{"type": "integer"}, {"type": "number"}]}));
        assert!(accepts(&s, "2.5"), "number covers integer");
        let s = compile(
            json!({"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "string", "enum": ["b"]}]}),
        );
        assert!(accepts(&s, r#""a""#) && accepts(&s, r#""b""#));
        let s = compile(json!({"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "string"}]}));
        assert!(accepts(&s, r#""anything""#), "string covers its enum");
        let s = compile(json!({"type": ["integer", "null"]}));
        assert!(
            accepts(&s, "42"),
            "a root number in a union may end the output"
        );
    }

    #[test]
    fn alternatives_that_open_alike_are_refused() {
        let obj = |k: &str| json!({"type": "object", "additionalProperties": false, "properties": {k: {"type": "string"}}, "required": [k]});
        for schema in [
            json!({"anyOf": [obj("a"), obj("b")]}),
            json!({"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "array", "items": {"type": "integer"}}]}),
            json!({"anyOf": []}),
            json!({"type": []}),
            json!({"type": "string", "anyOf": [{"type": "null"}]}),
            json!({"type": ["number", "null"], "enum": [1, null]}),
        ] {
            assert!(CompiledSchema::compile(&schema).is_err(), "{schema}");
        }
    }
}
