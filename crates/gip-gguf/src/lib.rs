//! Parser for GGUF model files.
//!
//! Every byte of a GGUF file comes from an untrusted source. The parser checks every read against
//! the end of the file and every size computation for overflow, so a malformed file yields an
//! [`Error`]. The crate holds no unsafe code, and it warns on indexing and unchecked arithmetic, so
//! no input reaches a panic.

#![forbid(unsafe_code)]
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;

/// The number of elements in one Q8_0 block.
pub const Q8_0_BLOCK_ELEMENTS: u64 = 32;

/// The number of bytes in one Q8_0 block: an fp16 scale and 32 int8 quants.
pub const Q8_0_BLOCK_BYTES: u64 = 34;

/// The most dimensions a GGUF tensor has.
pub const MAX_DIMS: usize = 4;

const MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: u64 = 32;

/// The smallest possible metadata entry is an empty key, a type, and a one-byte value.
const MIN_KV_BYTES: usize = 8 + 4 + 1;

/// The smallest possible tensor entry is an empty name, one dimension, a type, and an offset.
const MIN_TENSOR_BYTES: usize = 8 + 4 + 8 + 4 + 8;

/// Why a GGUF file failed to parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file does not start with the GGUF magic number.
    #[error("not a GGUF file")]
    NotGguf,
    /// The file ends inside the fixed header.
    #[error("truncated GGUF header")]
    TruncatedHeader,
    /// The header names a GGUF version other than 2 or 3.
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    /// The metadata count cannot fit in the file.
    #[error("metadata count {0} exceeds the file size")]
    MetadataCount(u64),
    /// A metadata entry runs past the end of the file or holds an unknown type.
    #[error("metadata entry {0} is malformed")]
    MalformedMetadata(u64),
    /// Two metadata entries share a key.
    #[error("metadata key {0} appears twice")]
    DuplicateKey(String),
    /// `general.alignment` is present but is no power of two.
    #[error("general.alignment must be a power of two")]
    Alignment,
    /// The tensor count cannot fit in the file.
    #[error("tensor count {0} exceeds the file size")]
    TensorCount(u64),
    /// A tensor entry runs past the end of the file.
    #[error("tensor {0} is malformed")]
    MalformedTensor(u64),
    /// A tensor has no dimensions or more than [`MAX_DIMS`].
    #[error("tensor {name} has {n_dims} dimensions")]
    Dimensions {
        /// The tensor's name.
        name: String,
        /// The dimension count the file gives.
        n_dims: u32,
    },
    /// A tensor has a dimension of zero or more elements than 64 bits count.
    #[error("tensor {0} has an invalid shape")]
    Shape(String),
    /// A Q8_0 tensor's rows hold a number of elements that 32 does not divide.
    #[error("tensor {name} has rows of {ne0} elements, which Q8_0 blocks of 32 cannot divide")]
    Q8_0Row {
        /// The tensor's name.
        name: String,
        /// The number of elements in each row.
        ne0: u64,
    },
    /// A tensor has a type gip does not read.
    #[error("tensor {name} has unsupported type {type_id}")]
    UnsupportedType {
        /// The tensor's name.
        name: String,
        /// The ggml type id the file gives.
        type_id: u32,
    },
    /// A tensor's byte count overflows.
    #[error("tensor {0} is too large")]
    TooLarge(String),
    /// A tensor's data is misaligned or extends past the end of the file.
    #[error("tensor {0} lies outside the file")]
    OutsideFile(String),
    /// The aligned start of the data section overflows.
    #[error("data section lies outside the file")]
    DataOutsideFile,
    /// Two tensors share a name.
    #[error("tensor {0} appears twice")]
    DuplicateTensor(String),
}

/// A tensor type gip reads. The ids match ggml's type ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorType {
    /// 32-bit IEEE floats.
    F32,
    /// 16-bit IEEE floats.
    F16,
    /// Blocks of 32 int8 quants with one fp16 scale.
    Q8_0,
    /// bfloat16 floats.
    Bf16,
}

impl TensorType {
    /// Return the type with ggml type id `id`, if gip reads it.
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q8_0),
            30 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Return the ggml type id of the type.
    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q8_0 => 8,
            Self::Bf16 => 30,
        }
    }

    /// Return the bytes that `n_elements` consecutive elements of the type occupy.
    ///
    /// Return `None` when the count overflows or, for Q8_0, when 32 does not divide `n_elements`.
    pub fn bytes_for(self, n_elements: u64) -> Option<u64> {
        match self {
            Self::F32 => n_elements.checked_mul(4),
            Self::F16 | Self::Bf16 => n_elements.checked_mul(2),
            Self::Q8_0 => {
                if !n_elements.is_multiple_of(Q8_0_BLOCK_ELEMENTS) {
                    return None;
                }
                (n_elements / Q8_0_BLOCK_ELEMENTS).checked_mul(Q8_0_BLOCK_BYTES)
            }
        }
    }
}

/// The types of GGUF metadata values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            _ => None,
        }
    }

    /// Return the size of a value of the type, or `None` for strings and arrays.
    fn fixed_size(self) -> Option<usize> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// One metadata value. Strings and arrays refer to their bytes in the file.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(Range<usize>),
    Array {
        element: ValueType,
        count: u64,
        data: Range<usize>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// Return the value as a `u32` when it is an integer that fits.
    fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U8(v) => Some(u32::from(v)),
            Self::U16(v) => Some(u32::from(v)),
            Self::U32(v) => Some(v),
            Self::I8(v) => u32::try_from(v).ok(),
            Self::I16(v) => u32::try_from(v).ok(),
            Self::I32(v) => u32::try_from(v).ok(),
            Self::U64(v) => u32::try_from(v).ok(),
            Self::I64(v) => u32::try_from(v).ok(),
            Self::F32(_) | Self::Bool(_) | Self::String(_) | Self::Array { .. } | Self::F64(_) => {
                None
            }
        }
    }
}

/// One tensor of a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    n_dims: usize,
    ne: [u64; MAX_DIMS],
    data_type: TensorType,
    data: Range<usize>,
}

impl Tensor {
    /// Return the size of each dimension, innermost first.
    pub fn shape(&self) -> &[u64] {
        self.ne.get(..self.n_dims).unwrap_or(&self.ne)
    }

    /// Return the size of dimension `dim`, which is 1 for dimensions past the tensor's last.
    pub fn ne(&self, dim: usize) -> u64 {
        self.ne.get(dim).copied().unwrap_or(1)
    }

    /// Return the tensor's element type.
    pub fn data_type(&self) -> TensorType {
        self.data_type
    }

    /// Return the byte range of the tensor's data within the file.
    pub fn data_range(&self) -> Range<usize> {
        self.data.clone()
    }
}

/// A parsed GGUF file that owns the file's bytes.
///
/// `B` holds the bytes, as a memory mapping or a heap buffer. Tensors and metadata strings refer
/// to byte ranges inside `B`.
#[derive(Debug)]
pub struct Gguf<B> {
    bytes: B,
    metadata: HashMap<String, Value>,
    tensors: HashMap<String, Tensor>,
}

impl<B: AsRef<[u8]>> Gguf<B> {
    /// Parse `bytes` as a GGUF file.
    pub fn parse(bytes: B) -> Result<Self, Error> {
        let file = bytes.as_ref();
        let mut reader = Reader::new(file);

        if reader.u32() != Some(MAGIC) {
            return Err(Error::NotGguf);
        }
        let (Some(version), Some(n_tensors), Some(n_kv)) =
            (reader.u32(), reader.u64(), reader.u64())
        else {
            return Err(Error::TruncatedHeader);
        };
        if version != 2 && version != 3 {
            return Err(Error::UnsupportedVersion(version));
        }

        let metadata = parse_metadata(&mut reader, n_kv)?;
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(Value::U32(value)) if value.is_power_of_two() => u64::from(*value),
            Some(_) => return Err(Error::Alignment),
        };
        let tensors = parse_tensors(&mut reader, n_tensors, alignment, file.len())?;
        Ok(Self {
            bytes,
            metadata,
            tensors,
        })
    }

    /// Return the file's bytes.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Return the value that holds the file's bytes.
    pub fn storage(&self) -> &B {
        &self.bytes
    }

    /// Return the integer metadata value `key` when it exists and fits in a `u32`.
    pub fn u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    /// Return the floating-point metadata value `key` when it exists.
    pub fn f32(&self, key: &str) -> Option<f32> {
        match *self.metadata.get(key)? {
            Value::F32(value) => Some(value),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "GGUF stores some float32 hyperparameters as float64"
            )]
            Value::F64(value) => Some(value as f32),
            Value::U8(_)
            | Value::I8(_)
            | Value::U16(_)
            | Value::I16(_)
            | Value::U32(_)
            | Value::I32(_)
            | Value::Bool(_)
            | Value::String(_)
            | Value::Array { .. }
            | Value::U64(_)
            | Value::I64(_) => None,
        }
    }

    /// Return the bytes of the string metadata value `key` when it exists.
    pub fn string(&self, key: &str) -> Option<&[u8]> {
        match self.metadata.get(key)? {
            Value::String(range) => self.bytes().get(range.clone()),
            _ => None,
        }
    }

    /// Return the element count of the array metadata value `key` when it exists.
    pub fn array_len(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            Value::Array { count, .. } => Some(*count),
            _ => None,
        }
    }

    /// Return element `index` of the integer array metadata value `key` when the element exists
    /// and fits in a `u32`.
    pub fn array_u32(&self, key: &str, index: u64) -> Option<u32> {
        let Value::Array {
            element,
            count,
            data,
        } = self.metadata.get(key)?
        else {
            return None;
        };
        if index >= *count {
            return None;
        }
        let size = element.fixed_size()?;
        let start = usize::try_from(index)
            .ok()?
            .checked_mul(size)?
            .checked_add(data.start)?;
        let mut reader = Reader::new(self.bytes());
        reader.pos = start;
        read_fixed(&mut reader, *element)?.as_u32()
    }

    /// Return the tensor named `name` when it exists.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Return the data bytes of `tensor`, which must come from this file.
    pub fn tensor_data(&self, tensor: &Tensor) -> Option<&[u8]> {
        self.bytes().get(tensor.data_range())
    }
}

/// A cursor over the bytes of a file.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Return the number of bytes after the cursor.
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// Return the `n` bytes at the cursor and advance past them.
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let taken = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(taken)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }

    /// Read a length-prefixed string and return its byte range.
    fn string(&mut self) -> Option<Range<usize>> {
        let length = usize::try_from(self.u64()?).ok()?;
        let start = self.pos;
        self.take(length)?;
        Some(start..self.pos)
    }

    /// Read a length-prefixed string that must hold UTF-8.
    fn utf8(&mut self) -> Option<String> {
        let range = self.string()?;
        let bytes = self.bytes.get(range)?;
        String::from_utf8(bytes.to_vec()).ok()
    }
}

/// Read one value of the fixed-size type `value_type` at the cursor of `reader`.
fn read_fixed(reader: &mut Reader<'_>, value_type: ValueType) -> Option<Value> {
    let value = match value_type {
        ValueType::U8 => Value::U8(u8::from_le_bytes(reader.array()?)),
        ValueType::I8 => Value::I8(i8::from_le_bytes(reader.array()?)),
        ValueType::U16 => Value::U16(u16::from_le_bytes(reader.array()?)),
        ValueType::I16 => Value::I16(i16::from_le_bytes(reader.array()?)),
        ValueType::U32 => Value::U32(u32::from_le_bytes(reader.array()?)),
        ValueType::I32 => Value::I32(i32::from_le_bytes(reader.array()?)),
        ValueType::F32 => Value::F32(f32::from_le_bytes(reader.array()?)),
        ValueType::Bool => Value::Bool(u8::from_le_bytes(reader.array()?) != 0),
        ValueType::U64 => Value::U64(u64::from_le_bytes(reader.array()?)),
        ValueType::I64 => Value::I64(i64::from_le_bytes(reader.array()?)),
        ValueType::F64 => Value::F64(f64::from_le_bytes(reader.array()?)),
        ValueType::String | ValueType::Array => return None,
    };
    Some(value)
}

/// Read one metadata value of type `type_id` at the cursor of `reader`.
fn read_value(reader: &mut Reader<'_>, type_id: u32) -> Option<Value> {
    let value_type = ValueType::from_id(type_id)?;
    match value_type {
        ValueType::String => reader.string().map(Value::String),
        ValueType::Array => {
            let element = ValueType::from_id(reader.u32()?)?;
            let count = reader.u64()?;
            let start = reader.pos;
            match element {
                ValueType::String => {
                    // Each string needs at least its 8-byte length, so a count larger than the
                    // remaining bytes allow is malformed.
                    if count > u64::try_from(reader.remaining() / 8).ok()? {
                        return None;
                    }
                    for _ in 0..count {
                        reader.string()?;
                    }
                }
                ValueType::Array => return None,
                ValueType::U8
                | ValueType::I8
                | ValueType::U16
                | ValueType::I16
                | ValueType::U32
                | ValueType::I32
                | ValueType::F32
                | ValueType::Bool
                | ValueType::U64
                | ValueType::I64
                | ValueType::F64 => {
                    let size = element.fixed_size()?;
                    let total = usize::try_from(count).ok()?.checked_mul(size)?;
                    reader.take(total)?;
                }
            }
            Some(Value::Array {
                element,
                count,
                data: start..reader.pos,
            })
        }
        ValueType::U8
        | ValueType::I8
        | ValueType::U16
        | ValueType::I16
        | ValueType::U32
        | ValueType::I32
        | ValueType::F32
        | ValueType::Bool
        | ValueType::U64
        | ValueType::I64
        | ValueType::F64 => read_fixed(reader, value_type),
    }
}

/// Parse the `n_kv` metadata entries at the cursor of `reader`.
fn parse_metadata(reader: &mut Reader<'_>, n_kv: u64) -> Result<HashMap<String, Value>, Error> {
    let capacity = usize::try_from(n_kv)
        .ok()
        .filter(|&n| n <= reader.remaining() / MIN_KV_BYTES)
        .ok_or(Error::MetadataCount(n_kv))?;

    let mut metadata = HashMap::with_capacity(capacity);
    for index in 0..n_kv {
        let entry = reader
            .utf8()
            .and_then(|key| Some((key, reader.u32()?)))
            .and_then(|(key, type_id)| Some((key, read_value(reader, type_id)?)));
        let Some((key, value)) = entry else {
            return Err(Error::MalformedMetadata(index));
        };
        match metadata.entry(key) {
            Entry::Occupied(slot) => return Err(Error::DuplicateKey(slot.key().clone())),
            Entry::Vacant(slot) => {
                slot.insert(value);
            }
        }
    }
    Ok(metadata)
}

/// The header of one tensor, before the data section's start is known.
struct TensorHeader {
    name: String,
    n_dims: u32,
    ne: [u64; MAX_DIMS],
    type_id: u32,
    offset: u64,
}

/// Read one tensor header at the cursor of `reader`.
fn read_tensor_header(reader: &mut Reader<'_>) -> Option<TensorHeader> {
    let name = reader.utf8()?;
    let n_dims = reader.u32()?;
    let mut ne = [1; MAX_DIMS];
    if (1..=MAX_DIMS).contains(&usize::try_from(n_dims).ok()?) {
        for dim in ne.iter_mut().take(usize::try_from(n_dims).ok()?) {
            *dim = reader.u64()?;
        }
    }
    Some(TensorHeader {
        name,
        n_dims,
        ne,
        type_id: reader.u32()?,
        offset: reader.u64()?,
    })
}

/// Parse the `n_tensors` tensor headers at the cursor of `reader` and locate each tensor's data
/// in a file of `file_size` bytes whose data section is aligned to `alignment`.
fn parse_tensors(
    reader: &mut Reader<'_>,
    n_tensors: u64,
    alignment: u64,
    file_size: usize,
) -> Result<HashMap<String, Tensor>, Error> {
    let capacity = usize::try_from(n_tensors)
        .ok()
        .filter(|&n| n <= reader.remaining() / MIN_TENSOR_BYTES)
        .ok_or(Error::TensorCount(n_tensors))?;

    // Each tensor's offset is relative to the data section, which starts after the last header.
    let mut headers = Vec::with_capacity(capacity);
    for index in 0..n_tensors {
        let header = read_tensor_header(reader).ok_or(Error::MalformedTensor(index))?;
        if !(1..=MAX_DIMS).contains(&(header.n_dims as usize)) {
            return Err(Error::Dimensions {
                name: header.name,
                n_dims: header.n_dims,
            });
        }
        headers.push(header);
    }

    let header_end = u64::try_from(reader.pos).map_err(|_| Error::DataOutsideFile)?;
    let data_start = header_end
        .checked_next_multiple_of(alignment)
        .ok_or(Error::DataOutsideFile)?;

    let mut tensors = HashMap::with_capacity(capacity);
    for header in headers {
        let (name, tensor) = locate_tensor(header, alignment, data_start, file_size)?;
        // Duplicate names would make lookups ambiguous.
        match tensors.entry(name) {
            Entry::Occupied(slot) => return Err(Error::DuplicateTensor(slot.key().clone())),
            Entry::Vacant(slot) => {
                slot.insert(tensor);
            }
        }
    }
    Ok(tensors)
}

/// Validate the shape and type of `header` and locate its data in a file of `file_size` bytes
/// whose data section starts at `data_start`. Return the tensor with its name.
fn locate_tensor(
    header: TensorHeader,
    alignment: u64,
    data_start: u64,
    file_size: usize,
) -> Result<(String, Tensor), Error> {
    let TensorHeader {
        name,
        n_dims,
        ne,
        type_id,
        offset,
    } = header;

    let n_elements = ne
        .iter()
        .try_fold(1_u64, |product, &dim| {
            if dim == 0 {
                None
            } else {
                product.checked_mul(dim)
            }
        })
        .ok_or_else(|| Error::Shape(name.clone()))?;

    let Some(data_type) = TensorType::from_id(type_id) else {
        return Err(Error::UnsupportedType { name, type_id });
    };
    let [ne0, ..] = ne;
    if data_type == TensorType::Q8_0 && !ne0.is_multiple_of(Q8_0_BLOCK_ELEMENTS) {
        return Err(Error::Q8_0Row { name, ne0 });
    }
    let Some(n_bytes) = data_type.bytes_for(n_elements) else {
        return Err(Error::TooLarge(name));
    };

    let start = data_start.checked_add(offset);
    let end = start.and_then(|start| start.checked_add(n_bytes));
    let (Some(start), Some(end)) = (start, end) else {
        return Err(Error::OutsideFile(name));
    };
    let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
        return Err(Error::OutsideFile(name));
    };
    if !offset.is_multiple_of(alignment) || end > file_size {
        return Err(Error::OutsideFile(name));
    }

    let tensor = Tensor {
        n_dims: n_dims as usize,
        ne,
        data_type,
        data: start..end,
    };
    Ok((name, tensor))
}
