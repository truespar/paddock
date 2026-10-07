use super::*;

/// A pickle written opcode by opcode, as torch.save would.
#[derive(Default)]
struct P(Vec<u8>);

impl P {
    fn op(&mut self, b: u8) -> &mut Self {
        self.0.push(b);
        self
    }
    fn global(&mut self, module: &str, name: &str) -> &mut Self {
        self.0.push(b'c');
        self.0.extend_from_slice(module.as_bytes());
        self.0.push(b'\n');
        self.0.extend_from_slice(name.as_bytes());
        self.0.push(b'\n');
        self
    }
    fn s(&mut self, s: &str) -> &mut Self {
        self.0.push(b'X');
        self.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.0.extend_from_slice(s.as_bytes());
        self
    }
    fn int(&mut self, n: u32) -> &mut Self {
        if n < 256 {
            self.0.extend_from_slice(&[b'K', n as u8]);
        } else {
            self.0.push(b'J');
            self.0.extend_from_slice(&n.to_le_bytes());
        }
        self
    }
    fn ints(&mut self, v: &[u32]) -> &mut Self {
        self.op(b'(');
        for &n in v {
            self.int(n);
        }
        self.op(b't')
    }
    /// `_rebuild_tensor_v2(storage(key, numel), offset, size, stride, False, OrderedDict())`
    fn tensor(
        &mut self,
        storage: &str,
        key: &str,
        numel: u32,
        offset: u32,
        size: &[u32],
        stride: &[u32],
    ) -> &mut Self {
        self.global("torch._utils", "_rebuild_tensor_v2").op(b'(');
        self.op(b'(')
            .s("storage")
            .global("torch", storage)
            .s(key)
            .s("cpu")
            .int(numel)
            .op(b't')
            .op(b'Q');
        self.int(offset).ints(size).ints(stride).op(0x89);
        self.global("collections", "OrderedDict").op(b')').op(b'R');
        self.op(b't').op(b'R')
    }
}

/// A zip of stored entries, laid out the way torch writes one.
fn zip(entries: &[(&str, &[u8])], method: u16) -> Vec<u8> {
    let mut f = Vec::new();
    let mut cd = Vec::new();
    for (name, data) in entries {
        let off = f.len() as u32;
        let local = |v: &mut Vec<u8>| {
            v.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            v.extend_from_slice(&[20, 0, 0, 0]);
            v.extend_from_slice(&method.to_le_bytes());
            v.extend_from_slice(&[0; 8]); // time, date, crc (not read)
            v.extend_from_slice(&(data.len() as u32).to_le_bytes());
            v.extend_from_slice(&(data.len() as u32).to_le_bytes());
            v.extend_from_slice(&(name.len() as u16).to_le_bytes());
            v.extend_from_slice(&[0, 0]);
            v.extend_from_slice(name.as_bytes());
        };
        local(&mut f);
        f.extend_from_slice(data);
        cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        cd.extend_from_slice(&[20, 0, 20, 0, 0, 0]);
        cd.extend_from_slice(&method.to_le_bytes());
        cd.extend_from_slice(&[0; 8]);
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&[0; 12]); // extra, comment, disk, attributes
        cd.extend_from_slice(&off.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
    }
    let cd_off = f.len() as u32;
    f.extend_from_slice(&cd);
    f.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    f.extend_from_slice(&[0; 4]);
    f.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    f.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    f.extend_from_slice(&(cd.len() as u32).to_le_bytes());
    f.extend_from_slice(&cd_off.to_le_bytes());
    f.extend_from_slice(&[0, 0]);
    f
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `{"w": 2x3 f32, "b": 3 f32 at offset 1 of a 4-element storage}`
fn checkpoint(p: &P) -> Vec<u8> {
    let w = f32s(&[1., 2., 3., 4., 5., 6.]);
    let b = f32s(&[9., 7., 8., 6.5]);
    zip(
        &[
            ("ck/data.pkl", &p.0),
            ("ck/byteorder", b"little"),
            ("ck/data/0", &w),
            ("ck/data/1", &b),
        ],
        0,
    )
}

fn plain() -> P {
    let mut p = P::default();
    p.op(0x80).op(2).op(b'}').op(b'q').op(0).op(b'(');
    p.s("w").tensor("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    p.s("b").tensor("FloatStorage", "1", 4, 1, &[3], &[1]);
    p.op(b'u').op(b'.');
    p
}

fn locate(bytes: &[u8]) -> Result<HashMap<String, StTensor>, StError> {
    TorchZipFile::locate(bytes)
}

fn err(r: Result<HashMap<String, StTensor>, StError>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn reads_a_plain_checkpoint_zero_copy() {
    let f = checkpoint(&plain());
    let t = locate(&f).expect("a plain checkpoint reads");
    assert_eq!(t.len(), 2);
    let w = &t["w"];
    assert_eq!((w.dtype, w.shape.clone()), (StDtype::F32, vec![2, 3]));
    assert_eq!(
        &f[w.begin..w.end],
        f32s(&[1., 2., 3., 4., 5., 6.]).as_slice()
    );
    // a view at an element offset inside its storage
    let b = &t["b"];
    assert_eq!(&f[b.begin..b.end], f32s(&[7., 8., 6.5]).as_slice());
}

#[test]
fn reads_a_state_dict_under_model_and_parameters() {
    let mut p = P::default();
    p.op(0x80)
        .op(2)
        .op(b'}')
        .op(b'(')
        .s("model")
        .op(b'}')
        .op(b'(');
    p.s("w")
        .global("torch._utils", "_rebuild_parameter")
        .op(b'(');
    p.tensor("FloatStorage", "0", 6, 0, &[6], &[1]).op(0x88);
    p.global("collections", "OrderedDict")
        .op(b')')
        .op(b'R')
        .op(b't')
        .op(b'R');
    p.op(b'u').s("step").int(7).op(b'u').op(b'.');
    let t = locate(&checkpoint(&p)).expect("a nested state dict reads");
    assert_eq!(t.keys().collect::<Vec<_>>(), vec!["w"]);
    assert_eq!(t["w"].shape, vec![6]);
}

#[test]
fn refuses_a_global_outside_the_list_by_name() {
    let mut p = P::default();
    p.op(0x80)
        .op(2)
        .global("os", "system")
        .s("echo hi")
        .op(0x85)
        .op(b'R')
        .op(b'.');
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("os.system"), "{e}");
}

#[test]
fn refuses_an_opcode_a_checkpoint_never_uses() {
    let mut p = P::default();
    p.op(0x80).op(2).op(b'}').op(b'b').op(b'.'); // BUILD
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("opcode 0x62"), "{e}");
}

#[test]
fn refuses_a_storage_type_called_as_a_function() {
    let mut p = P::default();
    p.op(0x80)
        .op(2)
        .global("torch", "FloatStorage")
        .op(b')')
        .op(b'R')
        .op(b'.');
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("storage type called"), "{e}");
}

#[test]
fn refuses_a_view_past_its_storage() {
    let mut p = P::default();
    p.op(0x80).op(2).op(b'}').op(b'(');
    p.s("b").tensor("FloatStorage", "1", 4, 2, &[3], &[1]);
    p.op(b'u').op(b'.');
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("past its storage"), "{e}");
}

#[test]
fn refuses_a_storage_the_zip_holds_less_of() {
    let mut p = P::default();
    p.op(0x80).op(2).op(b'}').op(b'(');
    p.s("w").tensor("FloatStorage", "0", 60, 0, &[60], &[1]);
    p.op(b'u').op(b'.');
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("shorter than its size"), "{e}");
}

#[test]
fn refuses_a_tensor_that_is_not_contiguous() {
    let mut p = P::default();
    p.op(0x80).op(2).op(b'}').op(b'(');
    p.s("w").tensor("FloatStorage", "0", 6, 0, &[2, 3], &[1, 2]);
    p.op(b'u').op(b'.');
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("not contiguous"), "{e}");
}

#[test]
fn refuses_compressed_entries() {
    let f = zip(&[("ck/data.pkl", &plain().0)], 8);
    let e = err(locate(&f));
    assert!(e.contains("compressed"), "{e}");
}

#[test]
fn refuses_a_shared_container_rather_than_misreading_it() {
    let mut p = P::default();
    // the dict is PUT empty, filled, then fetched again
    p.op(0x80).op(2).op(b'}').op(b'q').op(0).op(b'(');
    p.s("w").tensor("FloatStorage", "0", 6, 0, &[6], &[1]);
    p.op(b'u').op(b'h').op(0).op(0x86).op(b'.');
    let e = err(locate(&checkpoint(&p)));
    assert!(e.contains("shares a dict"), "{e}");
}

#[test]
fn refuses_a_truncated_pickle_at_every_cut() {
    // wherever the index is cut, the machine refuses it with its own words -
    // it never reads past the end or returns half a checkpoint
    let full = plain();
    for keep in 0..full.0.len() - 1 {
        let mut p = P::default();
        p.0.extend_from_slice(&full.0[..keep]);
        let e = err(locate(&checkpoint(&p)));
        assert!(e.contains("torch checkpoint"), "cut at {keep}: {e}");
    }
}
