//! Checks that the parser accepts a small valid file and rejects every truncation of it and a set
//! of corrupted headers.

use gip_gguf::{Gguf, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, TensorType};

const ALIGNMENT: u32 = 32;
const ROWS: u64 = 2;
const STRING_TYPE: u32 = 8;
const U32_TYPE: u32 = 4;

/// A small GGUF file and the offsets of the fields the corruption cases change.
#[derive(Clone)]
struct Builder {
    bytes: Vec<u8>,
    n_kv_at: usize,
    alignment_at: usize,
    n_dims_at: usize,
    ne0_at: usize,
    ne1_at: usize,
    type_at: usize,
    offset_at: usize,
}

impl Builder {
    /// Build a file with two metadata entries and one Q8_0 tensor of [`ROWS`] rows of 32 elements.
    fn valid() -> Self {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 0x4655_4747);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 1);
        let n_kv_at = bytes.len();
        put_u64(&mut bytes, 2);

        put_string(&mut bytes, "general.architecture");
        put_u32(&mut bytes, STRING_TYPE);
        put_string(&mut bytes, "test");
        put_string(&mut bytes, "general.alignment");
        put_u32(&mut bytes, U32_TYPE);
        let alignment_at = bytes.len();
        put_u32(&mut bytes, ALIGNMENT);

        put_string(&mut bytes, "weight");
        let n_dims_at = bytes.len();
        put_u32(&mut bytes, 2);
        let ne0_at = bytes.len();
        put_u64(&mut bytes, Q8_0_BLOCK_ELEMENTS);
        let ne1_at = bytes.len();
        put_u64(&mut bytes, ROWS);
        let type_at = bytes.len();
        put_u32(&mut bytes, TensorType::Q8_0.id());
        let offset_at = bytes.len();
        put_u64(&mut bytes, 0);

        bytes.resize(bytes.len().next_multiple_of(ALIGNMENT as usize), 0);
        bytes.resize(bytes.len() + data_bytes(), 1);

        Self {
            bytes,
            n_kv_at,
            alignment_at,
            n_dims_at,
            ne0_at,
            ne1_at,
            type_at,
            offset_at,
        }
    }

    /// Return the file with the 32-bit field at `at` set to `value`.
    fn with_u32(&self, at: usize, value: u32) -> Vec<u8> {
        let mut bytes = self.bytes.clone();
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        bytes
    }

    /// Return the file with the 64-bit field at `at` set to `value`.
    fn with_u64(&self, at: usize, value: u64) -> Vec<u8> {
        let mut bytes = self.bytes.clone();
        bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
        bytes
    }
}

/// Return the bytes of the weight tensor's data.
fn data_bytes() -> usize {
    usize::try_from(ROWS * Q8_0_BLOCK_BYTES).expect("the tensor holds 68 bytes")
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, s: &str) {
    put_u64(bytes, s.len() as u64);
    bytes.extend_from_slice(s.as_bytes());
}

#[test]
fn valid_file_parses() {
    let builder = Builder::valid();
    let size = builder.bytes.len();
    let gguf = Gguf::parse(builder.bytes).expect("the valid file parses");
    let weight = gguf
        .tensor("weight")
        .expect("the file holds the weight tensor");
    assert_eq!(weight.data_range(), size - data_bytes()..size);
    assert_eq!(gguf.string("general.architecture"), Some(&b"test"[..]));
}

#[test]
fn every_truncation_fails() {
    let builder = Builder::valid();
    for size in 0..builder.bytes.len() {
        let truncated = builder.bytes[..size].to_vec();
        assert!(
            Gguf::parse(truncated).is_err(),
            "truncation to {size} of {} bytes accepted",
            builder.bytes.len()
        );
    }
}

#[test]
fn corrupted_headers_fail() {
    let b = Builder::valid();
    let cases = [
        ("huge metadata count", b.with_u64(b.n_kv_at, 1 << 62)),
        ("alignment of 3", b.with_u32(b.alignment_at, 3)),
        ("alignment of 0", b.with_u32(b.alignment_at, 0)),
        ("zero dimensions", b.with_u32(b.n_dims_at, 0)),
        ("five dimensions", b.with_u32(b.n_dims_at, 5)),
        ("row of 31 Q8_0 elements", b.with_u64(b.ne0_at, 31)),
        ("zero-sized dimension", b.with_u64(b.ne1_at, 0)),
        ("overflowing shape", b.with_u64(b.ne1_at, 1 << 62)),
        ("unknown tensor type", b.with_u32(b.type_at, 99)),
        ("misaligned offset", b.with_u64(b.offset_at, 1)),
        (
            "offset past the end",
            b.with_u64(b.offset_at, u64::from(ALIGNMENT)),
        ),
        ("offset wrapping around", b.with_u64(b.offset_at, !0 - 31)),
    ];
    for (name, bytes) in cases {
        assert!(Gguf::parse(bytes).is_err(), "{name} accepted");
    }
}
