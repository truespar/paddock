use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(entries: &[(&str, &str, &[usize], Vec<u8>)]) -> Self {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "paddock-sam3-weight-test-{}-{}.safetensors",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let fixture = Self(path);
        let mut offset = 0;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            header.insert((*name).into(),serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+bytes.len()]}));
            offset += bytes.len();
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for (_, _, _, bytes) in entries {
            file.write_all(bytes).unwrap();
        }
        fixture
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn sam3_loader_rejects_wrong_shapes_dtypes_nonfinite_and_half_overflow() {
    let file = Fixture::new(&[
        ("matrix", "F32", &[2, 8], f32_bytes(&[2.; 16])),
        ("nan", "F32", &[1], f32_bytes(&[f32::NAN])),
        ("wrong", "BF16", &[1], vec![0, 0]),
    ]);
    let st = SafetensorsFile::open(&file.0).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let r = Reader { d: &d, st: &st };
    assert!(r.values("missing", &[1]).is_err());
    assert!(r.values("matrix", &[16]).is_err());
    assert!(r.values("wrong", &[1]).is_err());
    assert!(r.values("nan", &[1]).is_err());
    let baseline = d.allocated_bytes();
    assert!(r.matrix_values(&[2.; 16], 7, 2).is_err());
    for value in [65536., f32::INFINITY, f32::NAN] {
        assert!(r.matrix_values(&[value; 16], 8, 2).is_err());
        assert_eq!(
            d.allocated_bytes(),
            baseline,
            "failed upload retained weight buffers"
        );
    }
    let values = r.values("matrix", &[2, 8]).unwrap();
    let matrix = r.matrix_values(&values, 8, 2).unwrap();
    assert_eq!((matrix.k, matrix.n), (8, 2));
    assert_eq!(
        super::super::tests::bytes(&matrix.w),
        [0u8, 0x40].repeat(16)
    );
    drop(matrix);
    assert_eq!(d.allocated_bytes(), baseline);
}

#[test]
fn sam3_weight_conversion_is_chunk_bounded_and_overflow_flag_is_sticky() {
    let file = Fixture::new(&[("stub", "F32", &[1], f32_bytes(&[0.]))]);
    let st = SafetensorsFile::open(&file.0).unwrap();
    let elements = 65536 * 2 + 8;
    let budget = (elements * 2 + 65536 * 4 + 4) as u64;
    let d = MetalDevice::new(Some(budget)).unwrap();
    let r = Reader { d: &d, st: &st };
    let mut values: Vec<_> = (0..elements).map(|i| (i % 511) as f32 - 255.).collect();
    let m = r.matrix_values(&values, 8, elements / 8).unwrap();
    let actual = super::super::tests::bytes(&m.w);
    assert_eq!(
        actual,
        values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(d.allocated_bytes(), (elements * 2) as u64);
    drop(m);
    for index in [0, 65536, elements - 1] {
        let saved = values[index];
        values[index] = f32::INFINITY;
        assert!(r.matrix_values(&values, 8, elements / 8).is_err());
        assert_eq!(
            d.allocated_bytes(),
            0,
            "failed chunk upload leaked allocations"
        );
        values[index] = saved;
    }
    for (k, n) in [(0, 1), (8, 0), (usize::MAX, 8)] {
        assert!(r.matrix_values(&[], k, n).is_err());
    }
}

#[test]
fn sam3_loader_transposed_convolution_taps_have_exact_checkpoint_order() {
    let weights: Vec<_> = (0..64).map(|i| i as f32).collect();
    let file = Fixture::new(&[
        ("up.weight", "F32", &[8, 2, 2, 2], f32_bytes(&weights)),
        ("up.bias", "F32", &[2], f32_bytes(&[1., -1.])),
    ]);
    let st = SafetensorsFile::open(&file.0).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let r = Reader { d: &d, st: &st };
    let conv = r.convt("up", 8, 2).unwrap();
    let bytes = super::super::tests::bytes(&conv.w.w);
    for tap in 0..4 {
        for output in 0..2 {
            for input in 0..8 {
                let index = ((tap * 2 + output) * 8 + input) * 2;
                assert_eq!(
                    u16::from_le_bytes(bytes[index..index + 2].try_into().unwrap()),
                    half::f16::from_f32(weights[(input * 2 + output) * 4 + tap]).to_bits()
                );
            }
        }
    }
    assert_eq!(unsafe { conv.b.read_f32(0, 2) }, [1., -1.]);
}
