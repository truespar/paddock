//! Torch zip checkpoints (`torch.save`'s format - the `.pt` files Meta ships
//! SAM 3.1 as, with no safetensors beside them), read WITHOUT running them.
//!
//! The format is a zip of STORED entries: `<archive>/data.pkl`, a pickle that
//! says which tensors exist and where their bytes are, and one entry a storage
//! under `<archive>/data/<key>` holding the raw little-endian bytes. A pickle
//! is a program - Python's own unpickler will call any function it names -
//! so this reader never unpickles in that sense. It runs a small stack machine
//! that knows only the opcodes `torch.save` writes and only the globals a
//! tensor checkpoint needs (the tensor rebuild functions, the storage types,
//! OrderedDict), and builds DATA from them: a REDUCE is interpreted, never
//! called, and an opcode or global outside those lists is refused with its
//! name. The same allow-list idea as torch's own `weights_only=True` loader,
//! which is what Meta's code itself uses on these files.
//!
//! Every length and offset is checked against the bytes actually there, and
//! a tensor is handed out only when it is contiguous and inside its storage,
//! so what comes back is the same zero-copy `(StTensor, &[u8])` the
//! safetensors reader gives - `begin`/`end` here are offsets in the FILE.
//!
//! Facts the reader is written against (Meta's `sam3.pt` and
//! `sam3.1_multiplex.pt`, read with pickletools): pickle protocol 2; 18
//! opcodes; the globals `torch._utils._rebuild_tensor_v2`,
//! `torch.FloatStorage`, `torch.ComplexFloatStorage` and
//! `collections.OrderedDict`; every entry stored, every storage 64-byte
//! aligned, no zip64. The reader also takes the rest of what `torch.save`
//! writes for plain tensor checkpoints (protocol 4 frames, STACK_GLOBAL,
//! zip64, `_rebuild_parameter`, the other storage dtypes) so a re-saved file
//! does not trip it.

use std::collections::HashMap;
use std::path::Path;

use crate::safetensors::{StDtype, StError, StTensor, TensorSource};

/// The pickle index is metadata (names, shapes, offsets): a quarter of a MB
/// for SAM 3.1's 1623 tensors. Anything far past that is not a checkpoint.
const MAX_PICKLE: usize = 64 << 20;
/// Bound the machine's memory whatever the pickle says.
const MAX_STACK: usize = 1 << 20;
const MAX_MEMO: usize = 1 << 22;
const MAX_DIMS: usize = 16;

fn bad(msg: impl Into<String>) -> StError {
    StError::Header(format!("torch checkpoint: {}", msg.into()))
}

/// A storage's element type, from its storage class's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    F32,
    F16,
    Bf16,
    F64,
    C64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Elem {
    fn from_storage(name: &str) -> Option<Elem> {
        Some(match name {
            "FloatStorage" => Elem::F32,
            "HalfStorage" => Elem::F16,
            "BFloat16Storage" => Elem::Bf16,
            "DoubleStorage" => Elem::F64,
            "ComplexFloatStorage" => Elem::C64,
            "LongStorage" => Elem::I64,
            "IntStorage" => Elem::I32,
            "ShortStorage" => Elem::I16,
            "CharStorage" => Elem::I8,
            "ByteStorage" => Elem::U8,
            "BoolStorage" => Elem::Bool,
            _ => return None,
        })
    }
    fn bytes(self) -> usize {
        match self {
            Elem::F64 | Elem::C64 | Elem::I64 => 8,
            Elem::F32 | Elem::I32 => 4,
            Elem::F16 | Elem::Bf16 | Elem::I16 => 2,
            Elem::I8 | Elem::U8 | Elem::Bool => 1,
        }
    }
    /// What callers see: the dtypes the engine reads, the rest as Other so a
    /// caller rejects them by name instead of misreading them.
    fn st(self) -> StDtype {
        match self {
            Elem::F32 => StDtype::F32,
            Elem::F16 => StDtype::F16,
            Elem::Bf16 => StDtype::Bf16,
            Elem::I64 => StDtype::I64,
            Elem::U8 => StDtype::U8,
            _ => StDtype::Other,
        }
    }
}

/// The globals a tensor checkpoint may name; anything else is refused.
#[derive(Clone, Debug, PartialEq)]
enum Global {
    RebuildTensorV2,
    RebuildParameter,
    OrderedDict,
    Storage(Elem),
}

impl Global {
    fn allow(module: &str, name: &str) -> Result<Global, StError> {
        match (module, name) {
            ("torch._utils", "_rebuild_tensor_v2") => Ok(Global::RebuildTensorV2),
            ("torch._utils", "_rebuild_parameter") => Ok(Global::RebuildParameter),
            ("collections", "OrderedDict") => Ok(Global::OrderedDict),
            ("torch", s) if Elem::from_storage(s).is_some() => {
                Ok(Global::Storage(Elem::from_storage(s).expect("checked")))
            }
            _ => Err(bad(format!(
                "the pickle names {module}.{name}, which a tensor checkpoint never needs - refused"
            ))),
        }
    }
}

/// A tensor as the pickle describes it: a view of one storage.
#[derive(Clone, Debug, PartialEq)]
struct TensorRef {
    elem: Elem,
    key: String,
    /// elements in the storage
    storage_numel: u64,
    /// the view: element offset, sizes, strides
    offset: u64,
    shape: Vec<u64>,
    stride: Vec<u64>,
}

/// The values the machine builds - data only.
#[derive(Clone, Debug, PartialEq)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(Global),
    Storage { elem: Elem, key: String, numel: u64 },
    Tensor(TensorRef),
}

/// Bounds-checked reads over the pickle bytes.
struct Input<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Input<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], StError> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len());
        let end = end.ok_or_else(|| bad("the pickle ends inside an opcode"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, StError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, StError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, StError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, StError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn line(&mut self) -> Result<&'a str, StError> {
        let rest = &self.b[self.at..];
        let n = rest
            .iter()
            .position(|&c| c == b'\n')
            .ok_or_else(|| bad("a GLOBAL without its newline"))?;
        let s = std::str::from_utf8(&rest[..n]).map_err(|_| bad("a GLOBAL that is not UTF-8"))?;
        self.at += n + 1;
        Ok(s)
    }
    fn utf8(&mut self, n: usize) -> Result<String, StError> {
        let s = self.take(n)?;
        Ok(std::str::from_utf8(s)
            .map_err(|_| bad("a string that is not UTF-8"))?
            .to_owned())
    }
}

/// The stack machine. Runs `data.pkl` to its STOP and returns the value left.
fn run(pickle: &[u8]) -> Result<Value, StError> {
    let mut inp = Input { b: pickle, at: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: HashMap<u64, Value> = HashMap::new();
    let pop = |stack: &mut Vec<Value>| {
        stack
            .pop()
            .ok_or_else(|| bad("the pickle pops an empty stack"))
    };
    let since_mark =
        |stack: &mut Vec<Value>, marks: &mut Vec<usize>| -> Result<Vec<Value>, StError> {
            let m = marks
                .pop()
                .ok_or_else(|| bad("a pickle opcode with no MARK before it"))?;
            if m > stack.len() {
                return Err(bad("a MARK past the top of the stack"));
            }
            Ok(stack.split_off(m))
        };
    loop {
        if stack.len() > MAX_STACK || memo.len() > MAX_MEMO {
            return Err(bad("the pickle grows past any tensor checkpoint's size"));
        }
        let op = inp.u8()?;
        match op {
            0x80 => {
                // PROTO
                let v = inp.u8()?;
                if !(2..=5).contains(&v) {
                    return Err(bad(format!("pickle protocol {v}")));
                }
            }
            0x95 => {
                // FRAME: a length hint, nothing to do
                inp.u64()?;
            }
            b'.' => return pop(&mut stack), // STOP
            b'(' => marks.push(stack.len()),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b't' => {
                let items = since_mark(&mut stack, &mut marks)?;
                stack.push(Value::Tuple(items));
            }
            b'l' => {
                let items = since_mark(&mut stack, &mut marks)?;
                stack.push(Value::List(items));
            }
            0x85..=0x87 => {
                // TUPLE1..3
                let n = (op - 0x84) as usize;
                if stack.len() < n {
                    return Err(bad("a TUPLE of more items than the stack holds"));
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(Value::Tuple(items));
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::List(l)) => l.push(v),
                    _ => return Err(bad("APPEND to something that is not a list")),
                }
            }
            b'e' => {
                let items = since_mark(&mut stack, &mut marks)?;
                match stack.last_mut() {
                    Some(Value::List(l)) => l.extend(items),
                    _ => return Err(bad("APPENDS to something that is not a list")),
                }
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::Dict(d)) => d.push((k, v)),
                    _ => return Err(bad("SETITEM on something that is not a dict")),
                }
            }
            b'u' => {
                let items = since_mark(&mut stack, &mut marks)?;
                if items.len() % 2 != 0 {
                    return Err(bad("SETITEMS with an odd number of items"));
                }
                match stack.last_mut() {
                    Some(Value::Dict(d)) => {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                    _ => return Err(bad("SETITEMS on something that is not a dict")),
                }
            }
            b'J' => stack.push(Value::Int(inp.u32()? as i32 as i64)), // BININT
            b'K' => stack.push(Value::Int(inp.u8()? as i64)),         // BININT1
            b'M' => stack.push(Value::Int(inp.u16()? as i64)),        // BININT2
            0x8a => {
                // LONG1: little-endian two's complement, up to 8 bytes here
                let n = inp.u8()? as usize;
                if n > 8 {
                    return Err(bad("an integer wider than 64 bits"));
                }
                let b = inp.take(n)?;
                let mut v: i64 = 0;
                for (i, &x) in b.iter().enumerate() {
                    v |= (x as i64) << (8 * i);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Value::Int(v));
            }
            b'G' => {
                // BINFLOAT, big-endian
                let b: [u8; 8] = inp.take(8)?.try_into().expect("8 bytes");
                stack.push(Value::Float(f64::from_be_bytes(b)));
            }
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'X' => {
                let n = inp.u32()? as usize;
                let s = inp.utf8(n)?;
                stack.push(Value::Str(s));
            }
            0x8c => {
                let n = inp.u8()? as usize;
                let s = inp.utf8(n)?;
                stack.push(Value::Str(s));
            }
            0x8d => {
                let n = usize::try_from(inp.u64()?)
                    .map_err(|_| bad("a string past the address space"))?;
                let s = inp.utf8(n)?;
                stack.push(Value::Str(s));
            }
            b'c' => {
                // GLOBAL module\nname\n
                let module = inp.line()?.to_owned();
                let name = inp.line()?;
                stack.push(Value::Global(Global::allow(&module, name)?));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                match (module, name) {
                    (Value::Str(m), Value::Str(n)) => {
                        stack.push(Value::Global(Global::allow(&m, &n)?))
                    }
                    _ => return Err(bad("STACK_GLOBAL of something that is not two strings")),
                }
            }
            b'q' | b'r' | 0x94 => {
                // BINPUT / LONG_BINPUT / MEMOIZE
                let k = match op {
                    b'q' => inp.u8()? as u64,
                    b'r' => inp.u32()? as u64,
                    _ => memo.len() as u64,
                };
                let top = stack
                    .last()
                    .ok_or_else(|| bad("a memo PUT of an empty stack"))?;
                memo.insert(k, top.clone());
            }
            b'h' | b'j' => {
                // BINGET / LONG_BINGET. Python's memo holds references, this
                // one copies: a dict or list filled after its PUT would come
                // back as it was then. torch.save only fetches immutable
                // values again (globals, strings, storages, tuples), so a
                // container fetched here is refused rather than misread.
                let k = if op == b'h' {
                    inp.u8()? as u64
                } else {
                    inp.u32()? as u64
                };
                let v = memo
                    .get(&k)
                    .ok_or_else(|| bad(format!("a memo GET of unknown slot {k}")))?;
                if matches!(v, Value::Dict(_) | Value::List(_)) {
                    return Err(bad(
                        "a pickle that shares a dict or list - not a plain tensor checkpoint",
                    ));
                }
                stack.push(v.clone());
            }
            b'Q' => {
                // BINPERSID: ('storage', <storage type>, key, location, numel)
                let pid = pop(&mut stack)?;
                stack.push(storage(pid)?);
            }
            b'R' => {
                // REDUCE: interpreted for the allowed globals, never called
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(reduce(callable, args)?);
            }
            _ => {
                return Err(bad(format!(
                    "pickle opcode 0x{op:02x} at byte {} - not one a tensor checkpoint uses, refused",
                    inp.at - 1
                )));
            }
        }
    }
}

fn storage(pid: Value) -> Result<Value, StError> {
    let Value::Tuple(t) = pid else {
        return Err(bad("a persistent id that is not a tuple"));
    };
    match t.as_slice() {
        [
            Value::Str(tag),
            Value::Global(Global::Storage(elem)),
            Value::Str(key),
            Value::Str(_location),
            Value::Int(numel),
        ] if tag == "storage" && *numel >= 0 => Ok(Value::Storage {
            elem: *elem,
            key: key.clone(),
            numel: *numel as u64,
        }),
        _ => Err(bad(
            "a persistent id that is not ('storage', <type>, key, location, size)",
        )),
    }
}

fn ints(v: &Value, what: &str) -> Result<Vec<u64>, StError> {
    let Value::Tuple(t) = v else {
        return Err(bad(format!("a tensor {what} that is not a tuple")));
    };
    if t.len() > MAX_DIMS {
        return Err(bad(format!("a tensor with {} dims", t.len())));
    }
    t.iter()
        .map(|x| match x {
            Value::Int(n) if *n >= 0 => Ok(*n as u64),
            _ => Err(bad(format!("a tensor {what} entry that is not a size"))),
        })
        .collect()
}

fn reduce(callable: Value, args: Value) -> Result<Value, StError> {
    let Value::Global(g) = callable else {
        return Err(bad("REDUCE of something that is not an allowed global"));
    };
    let Value::Tuple(a) = args else {
        return Err(bad("REDUCE with arguments that are not a tuple"));
    };
    match g {
        Global::OrderedDict => match a.as_slice() {
            [] => Ok(Value::Dict(Vec::new())),
            [Value::List(pairs)] => {
                let mut d = Vec::with_capacity(pairs.len());
                for p in pairs {
                    match p {
                        Value::Tuple(kv) if kv.len() == 2 => d.push((kv[0].clone(), kv[1].clone())),
                        _ => return Err(bad("an OrderedDict item that is not a pair")),
                    }
                }
                Ok(Value::Dict(d))
            }
            _ => Err(bad(
                "OrderedDict with arguments it never takes in a checkpoint",
            )),
        },
        // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
        Global::RebuildTensorV2 => {
            if !(6..=7).contains(&a.len()) {
                return Err(bad("_rebuild_tensor_v2 with the wrong number of arguments"));
            }
            let Value::Storage { elem, key, numel } = &a[0] else {
                return Err(bad("_rebuild_tensor_v2 of something that is not a storage"));
            };
            let Value::Int(offset) = a[1] else {
                return Err(bad(
                    "_rebuild_tensor_v2 with an offset that is not an integer",
                ));
            };
            if offset < 0 {
                return Err(bad("a negative storage offset"));
            }
            let shape = ints(&a[2], "size")?;
            let stride = ints(&a[3], "stride")?;
            if shape.len() != stride.len() {
                return Err(bad("a tensor whose size and stride disagree in rank"));
            }
            Ok(Value::Tensor(TensorRef {
                elem: *elem,
                key: key.clone(),
                storage_numel: *numel,
                offset: offset as u64,
                shape,
                stride,
            }))
        }
        // (tensor, requires_grad, backward_hooks): the tensor itself
        Global::RebuildParameter => match a.first() {
            Some(Value::Tensor(t)) if a.len() == 3 => Ok(Value::Tensor(t.clone())),
            _ => Err(bad("_rebuild_parameter of something that is not a tensor")),
        },
        Global::Storage(_) => Err(bad("a storage type called as a function")),
    }
}

/// The checkpoint's tensors from the pickle's top value: a dict of name ->
/// tensor, or such a dict under "model" / "state_dict" (both are how torch
/// training code saves). Non-tensor entries (counters, configs) are skipped.
fn tensors(top: Value) -> Result<Vec<(String, TensorRef)>, StError> {
    let Value::Dict(items) = top else {
        return Err(bad("the pickle does not hold a dict"));
    };
    for (k, v) in &items {
        if let (Value::Str(k), Value::Dict(_)) = (k, v)
            && (k == "model" || k == "state_dict")
        {
            return tensors(v.clone());
        }
    }
    let mut out = Vec::new();
    for (k, v) in items {
        if let (Value::Str(k), Value::Tensor(t)) = (k, v) {
            out.push((k, t));
        }
    }
    Ok(out)
}

// -- the zip ----------------------------------------------------------------------

struct Entry {
    /// where the entry's bytes start in the file, and how many
    start: usize,
    len: usize,
}

fn le16(b: &[u8], at: usize) -> Result<u16, StError> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes(s.try_into().expect("2 bytes")))
        .ok_or_else(|| bad("the zip ends inside a record"))
}
fn le32(b: &[u8], at: usize) -> Result<u32, StError> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
        .ok_or_else(|| bad("the zip ends inside a record"))
}
fn le64(b: &[u8], at: usize) -> Result<u64, StError> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
        .ok_or_else(|| bad("the zip ends inside a record"))
}
fn size(v: u64) -> Result<usize, StError> {
    usize::try_from(v).map_err(|_| bad("a zip offset past the address space"))
}

/// The zip's entries by name: stored only (torch never compresses), every
/// range inside the file.
fn entries(f: &[u8]) -> Result<HashMap<String, Entry>, StError> {
    // the end-of-central-directory record, within the last 64 KiB + 22
    let lo = f.len().saturating_sub(65_535 + 22);
    let eocd = (lo..f.len().saturating_sub(21))
        .rev()
        .find(|&i| f[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| bad("not a zip file (no end of central directory)"))?;
    let mut count = le16(f, eocd + 10)? as u64;
    let mut cd_size = le32(f, eocd + 12)? as u64;
    let mut cd_off = le32(f, eocd + 16)? as u64;
    if count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        // zip64: the locator sits just before the record
        let loc = eocd
            .checked_sub(20)
            .ok_or_else(|| bad("a zip64 file without its locator"))?;
        if le32(f, loc)? != 0x0706_4b50 {
            return Err(bad("a zip64 file without its locator"));
        }
        let rec = size(le64(f, loc + 8)?)?;
        if le32(f, rec)? != 0x0606_4b50 {
            return Err(bad("a zip64 locator pointing at no record"));
        }
        count = le64(f, rec + 32)?;
        cd_size = le64(f, rec + 40)?;
        cd_off = le64(f, rec + 48)?;
    }
    let (cd_off, cd_size) = (size(cd_off)?, size(cd_size)?);
    if cd_off.checked_add(cd_size).is_none_or(|e| e > f.len()) {
        return Err(bad("a central directory past the end of the file"));
    }
    let mut out = HashMap::new();
    let mut at = cd_off;
    for _ in 0..count {
        if le32(f, at)? != 0x0201_4b50 {
            return Err(bad("a damaged central directory"));
        }
        let flags = le16(f, at + 8)?;
        let method = le16(f, at + 10)?;
        let mut comp = le32(f, at + 20)? as u64;
        let mut uncomp = le32(f, at + 24)? as u64;
        let name_len = le16(f, at + 28)? as usize;
        let extra_len = le16(f, at + 30)? as usize;
        let comment_len = le16(f, at + 32)? as usize;
        let mut local = le32(f, at + 42)? as u64;
        let name = f
            .get(at + 46..at + 46 + name_len)
            .ok_or_else(|| bad("a zip entry name past the end"))?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|_| bad("a zip entry name that is not UTF-8"))?;
        // zip64 sizes and offset, in the order the 0xFFFFFFFF fields appear
        let mut x = at + 46 + name_len;
        let xend = x + extra_len;
        while x + 4 <= xend {
            let id = le16(f, x)?;
            let n = le16(f, x + 2)? as usize;
            if id == 0x0001 {
                let mut p = x + 4;
                if uncomp == 0xFFFF_FFFF {
                    uncomp = le64(f, p)?;
                    p += 8;
                }
                if comp == 0xFFFF_FFFF {
                    comp = le64(f, p)?;
                    p += 8;
                }
                if local == 0xFFFF_FFFF {
                    local = le64(f, p)?;
                }
            }
            x += 4 + n;
        }
        if flags & 1 != 0 {
            return Err(bad(format!("{name} is encrypted")));
        }
        if method != 0 || comp != uncomp {
            return Err(bad(format!(
                "{name} is compressed (method {method}); torch stores entries"
            )));
        }
        let local = size(local)?;
        if le32(f, local)? != 0x0403_4b50 {
            return Err(bad(format!(
                "{name}: no local header where the directory says"
            )));
        }
        let start = local + 30 + le16(f, local + 26)? as usize + le16(f, local + 28)? as usize;
        let len = size(uncomp)?;
        if start.checked_add(len).is_none_or(|e| e > f.len()) {
            return Err(bad(format!("{name} runs past the end of the file")));
        }
        out.insert(name, Entry { start, len });
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(out)
}

/// A torch zip checkpoint, memory-mapped, with its tensors located.
pub struct TorchZipFile {
    map: memmap2::Mmap,
    tensors: HashMap<String, StTensor>,
}

impl TorchZipFile {
    pub fn open(path: &Path) -> Result<Self, StError> {
        let f = std::fs::File::open(path)?;
        // SAFETY: a read-only mapping of a file we just opened; every range
        // handed out below is validated against its length first.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let tensors = Self::locate(&map)?;
        Ok(Self { map, tensors })
    }

    fn locate(f: &[u8]) -> Result<HashMap<String, StTensor>, StError> {
        let entries = entries(f)?;
        let pkl: Vec<&String> = entries
            .keys()
            .filter(|n| n.ends_with("/data.pkl"))
            .collect();
        let [pkl] = pkl.as_slice() else {
            return Err(bad(format!("{} data.pkl entries (want one)", pkl.len())));
        };
        let prefix = &pkl[..pkl.len() - "data.pkl".len()];
        if let Some(e) = entries.get(&format!("{prefix}byteorder"))
            && &f[e.start..e.start + e.len] != b"little"
        {
            return Err(bad("a big-endian checkpoint"));
        }
        let e = &entries[*pkl];
        if e.len > MAX_PICKLE {
            return Err(bad(format!("a {} MB pickle index", e.len >> 20)));
        }
        let top = run(&f[e.start..e.start + e.len])?;
        let mut out = HashMap::new();
        for (name, t) in tensors(top)? {
            let s = entries
                .get(&format!("{prefix}data/{}", t.key))
                .ok_or_else(|| bad(format!("{name}: its storage {} is not in the zip", t.key)))?;
            let es = t.elem.bytes() as u64;
            if (s.len as u64) < t.storage_numel.saturating_mul(es) {
                return Err(bad(format!("{name}: its storage is shorter than its size")));
            }
            // contiguous row-major, the only layout handed out as one slice
            let mut want = 1u64;
            for (&n, &st) in t.shape.iter().zip(&t.stride).rev() {
                if n > 1 && st != want {
                    return Err(bad(format!(
                        "{name} is not contiguous (size {:?}, stride {:?})",
                        t.shape, t.stride
                    )));
                }
                want = want
                    .checked_mul(n)
                    .ok_or_else(|| bad(format!("{name}: its size overflows")))?;
            }
            let numel = want;
            if t.offset
                .checked_add(numel)
                .is_none_or(|e| e > t.storage_numel)
            {
                return Err(bad(format!("{name} reaches past its storage")));
            }
            let begin = s.start as u64 + t.offset * es;
            out.insert(
                name,
                StTensor {
                    dtype: t.elem.st(),
                    shape: t.shape.iter().map(|&n| n as usize).collect(),
                    begin: size(begin)?,
                    end: size(begin + numel * es)?,
                },
            );
        }
        Ok(out)
    }

    pub fn tensors(&self) -> &HashMap<String, StTensor> {
        &self.tensors
    }

    /// Raw bytes of `name` (zero-copy), or None if absent.
    pub fn bytes(&self, name: &str) -> Option<(&StTensor, &[u8])> {
        let t = self.tensors.get(name)?;
        Some((t, &self.map[t.begin..t.end]))
    }

    pub fn total_len(&self) -> u64 {
        self.map.len() as u64
    }

    /// The mapped file, for views cut inside a located tensor (a row range
    /// of a fused projection, say) - ranges come from [`Self::tensors`].
    pub fn file_bytes(&self) -> &[u8] {
        &self.map
    }
}

impl TensorSource for TorchZipFile {
    fn tensor(&self, name: &str) -> Option<(&StTensor, &[u8])> {
        self.bytes(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
    fn mapped_len(&self) -> u64 {
        self.total_len()
    }
}

#[cfg(test)]
#[path = "torch_zip_tests.rs"]
mod tests;
