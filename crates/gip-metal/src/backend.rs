//! The Metal host code behind [`Metal`].
//!
//! Every Objective-C call that carries a safety contract goes through a helper in this module that
//! checks the contract first. Kernel launches check that each bound range lies inside its buffer.
//! CPU reads and writes of a buffer check that no command buffer is in flight.

#![expect(
    unsafe_code,
    reason = "Metal's API is Objective-C, reached through objc2-metal"
)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use block2::RcBlock;
use memmap2::Mmap;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType,
    MTLFunctionConstantValues, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

/// The kernel source, embedded at build time. `kernels_q4.metal` uses the helpers that
/// `kernels.metal` defines, so it comes second.
const SOURCE: &str = concat!(
    include_str!("kernels.metal"),
    include_str!("kernels_q4.metal")
);

const SIMD_WIDTH: usize = 32;

/// A sweep of 1, 2, and 4 rows per threadgroup against 2, 4, and 8 simdgroups on an M4 Pro put
/// two rows first on every model. Two simdgroups won on rows of 1024 columns, lifting
/// LFM2.5-350M from 445 to 495 tokens per second. Four simdgroups won by about 1% on the wider
/// rows of LFM2.5-1.2B and LFM2.5-2.6B.
const MATVEC_Q8_0_ROWS_PER_THREADGROUP: u32 = 2;
const MATVEC_Q8_0_NARROW_SIMDGROUPS: usize = 2;
const MATVEC_Q8_0_WIDE_SIMDGROUPS: usize = 4;
const MATVEC_Q8_0_WIDE_COLS: u32 = 2048;

/// The K-quant matrix-vector kernels run two simdgroups of four rows each. On an M4 Pro, four rows
/// per simdgroup in place of llama.cpp's two lifted LFM2.5-2.6B Q4_K_M decode from 106 to 116
/// tokens per second and left LFM2.5-350M at 598. Eight rows fell to 114 and 538. Four
/// simdgroups measured within noise of two. `K_QUANT_ROWS_PER_SIMDGROUP` must match
/// `K_ROWS_PER_SIMDGROUP` in `kernels_q4.metal`.
const K_QUANT_SIMDGROUPS: u32 = 2;
const K_QUANT_ROWS_PER_SIMDGROUP: u32 = 4;

/// Threads per threadgroup for the reduction and elementwise kernels.
const REDUCE_THREADS: usize = 256;
const ELEMENTWISE_THREADS: usize = 256;

/// Must match `ATTENTION_CHUNK` in `kernels.metal`.
const ATTENTION_CHUNK: u32 = 64;

/// Must match `MATMUL_ROWS`, `MATMUL_TOKENS`, and `MATMUL_SIMDGROUPS` in `kernels.metal`.
const MATMUL_ROWS: u32 = 64;
const MATMUL_TOKENS: u32 = 32;
const MATMUL_SIMDGROUPS: usize = 4;

/// One threadgroup of the largest size scans a vocabulary in a few hundred loads per thread.
const ARGMAX_THREADS: usize = 1024;

const FLOAT_BYTES: usize = 4;
const HALF_BYTES: usize = 2;

/// Identifies each [`Metal`] so launches can reject buffers from another backend.
static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Why a Metal operation failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The system has no Metal device.
    #[error("no Metal device")]
    NoDevice,
    /// The Metal compiler rejected `kernels.metal`.
    #[error("cannot compile Metal kernels: {0}")]
    Compile(String),
    /// A kernel is missing or its pipeline failed to build.
    #[error("cannot create the Metal pipeline {name}: {message}")]
    Pipeline {
        /// The kernel's name.
        name: &'static str,
        /// Metal's explanation.
        message: String,
    },
    /// Metal refused to create a command queue.
    #[error("cannot create a Metal command queue")]
    Queue,
    /// Metal refused to allocate a buffer.
    #[error("cannot allocate a Metal buffer of {0} bytes")]
    Allocation(usize),
    /// Metal refused to wrap a memory mapping.
    #[error("cannot wrap a memory mapping of {0} bytes in a Metal buffer")]
    Wrap(usize),
    /// Metal refused to create a command buffer or encoder.
    #[error("cannot create a Metal command buffer")]
    CommandBuffer,
    /// A launch or commit came with no command buffer open.
    #[error("no command buffer is open")]
    NotRecording,
    /// [`Metal::begin`] came while a command buffer was already open.
    #[error("a command buffer is already open")]
    AlreadyRecording,
    /// A launch would read or write past the end of a buffer.
    #[error("the {kernel} launch needs {needed} bytes at offset {offset} of a {len}-byte buffer")]
    OutOfBounds {
        /// The kernel's name.
        kernel: &'static str,
        /// The byte offset of the bound range.
        offset: usize,
        /// The bytes the launch touches.
        needed: usize,
        /// The buffer's length.
        len: usize,
    },
    /// A buffer from another [`Metal`] reached this one.
    #[error("the buffer belongs to another Metal backend")]
    ForeignBuffer,
    /// The CPU tried to touch a buffer while command buffers were in flight.
    #[error("the CPU cannot access a buffer while command buffers are in flight")]
    Busy,
    /// A CPU read or write would pass the end of a buffer.
    #[error("CPU access of {needed} bytes at offset {offset} exceeds a {len}-byte buffer")]
    Access {
        /// The byte offset of the access.
        offset: usize,
        /// The bytes accessed.
        needed: usize,
        /// The buffer's length.
        len: usize,
    },
    /// The GPU reported an error while running a command buffer.
    #[error("the GPU failed to run a command buffer: {0}")]
    Execution(String),
}

/// A Metal buffer in memory the CPU and GPU share.
#[derive(Debug)]
pub struct Buffer {
    owner: u64,
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

impl Buffer {
    /// Return a view of the buffer at byte `offset`.
    pub fn at(&self, offset: usize) -> View<'_> {
        View {
            buffer: self,
            offset,
        }
    }

    /// Return a view of the buffer at float `index`.
    pub fn floats(&self, index: usize) -> View<'_> {
        self.at(index.saturating_mul(FLOAT_BYTES))
    }
}

/// A position inside a [`Buffer`].
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    buffer: &'a Buffer,
    offset: usize,
}

/// An RMS normalization a matrix-vector launch applies to its input first.
#[derive(Debug, Clone, Copy)]
pub struct Norm<'a> {
    /// The scale of each input element.
    pub weight: View<'a>,
    /// The epsilon that guards the division.
    pub eps: f32,
}

/// Work a matrix-vector launch folds in.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatvecOptions<'a> {
    /// The normalization of the input, if any.
    pub norm: Option<Norm<'a>>,
    /// Whether the launch adds its results to the output.
    pub accumulate: bool,
}

/// How a matrix-matrix launch combines its results with the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    /// The results replace the output.
    Overwrite,
    /// The results add to the output.
    Accumulate,
    /// The output holds gate projections, and the launch computes the matching up projections.
    /// The output receives SiLU of each gate times its up.
    Swiglu,
}

/// A number that [`Metal::wait`] waits on, returned by [`Metal::commit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ticket(u64);

/// The GPU time spent in one kernel on one shape while profiling.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileEntry {
    /// The kernel's name.
    pub name: &'static str,
    /// The rows of a matrix launch, or zero for other kernels.
    pub n_rows: u32,
    /// The columns of a matrix launch, or zero for other kernels.
    pub n_cols: u32,
    /// The number of launches.
    pub calls: u64,
    /// The GPU time of the launches in seconds.
    pub seconds: f64,
    /// The weight bytes the launches read.
    pub bytes: u64,
}

/// Element types the CPU may copy in and out of a buffer. Every bit pattern is a valid value of
/// each type.
pub trait Element: Copy + sealed::Sealed {}

impl<T: Copy + sealed::Sealed> Element for T {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
}

/// A weight format the matrix and embedding kernels read. The block layouts match ggml's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Blocks of 32 weights: an fp16 scale and 32 int8 quants.
    Q8_0,
    /// Blocks of 32 weights: an fp16 scale and 32 4-bit quants.
    Q4_0,
    /// Super-blocks of 256 weights with 4-bit quants and 6-bit group scales and minimums.
    Q4K,
    /// Super-blocks of 256 weights with 6-bit quants and int8 group scales.
    Q6K,
}

impl Format {
    const ALL: [Self; 4] = [Self::Q8_0, Self::Q4_0, Self::Q4K, Self::Q6K];

    fn index(self) -> usize {
        match self {
            Self::Q8_0 => 0,
            Self::Q4_0 => 1,
            Self::Q4K => 2,
            Self::Q6K => 3,
        }
    }

    /// Return the weights and bytes of one block.
    fn block(self) -> (usize, usize) {
        match self {
            Self::Q8_0 => (32, 34),
            Self::Q4_0 => (32, 18),
            Self::Q4K => (256, 144),
            Self::Q6K => (256, 210),
        }
    }

    /// Return the bytes of a matrix of `n_rows` rows of `n_cols` weights.
    fn matrix_bytes(self, n_rows: usize, n_cols: usize) -> usize {
        let (weights, bytes) = self.block();
        product(&[n_rows, n_cols / weights, bytes])
    }

    /// Return the kernel names of the format: the matrix-vector product, its SwiGLU pair, the
    /// matrix-matrix product, and the embedding lookup.
    fn kernel_names(self) -> [&'static str; 4] {
        match self {
            Self::Q8_0 => [
                "matvec_q8_0",
                "matvec_q8_0_swiglu",
                "matmul_q8_0",
                "embed_q8_0",
            ],
            Self::Q4_0 => [
                "matvec_q4_0",
                "matvec_q4_0_swiglu",
                "matmul_q4_0",
                "embed_q4_0",
            ],
            Self::Q4K => ["matvec_q4k", "matvec_q4k_swiglu", "matmul_q4k", "embed_q4k"],
            Self::Q6K => ["matvec_q6k", "matvec_q6k_swiglu", "matmul_q6k", "embed_q6k"],
        }
    }
}

/// The kernels and their specializations.
#[derive(Debug, Clone, Copy)]
enum Kernel {
    Matvec {
        format: Format,
        norm: bool,
        accumulate: bool,
    },
    MatvecSwiglu(Format),
    RmsNorm,
    NormRope {
        half: bool,
    },
    ConvertHalf,
    AttentionChunk {
        half: bool,
    },
    AttentionCombine,
    ShortConv,
    ShortConvBatch,
    ShortConvHistory,
    Matmul(Format, Store),
    Copy,
    Embed(Format),
    Argmax,
}

impl Kernel {
    /// Return the name the profile shows for the kernel.
    fn name(self) -> &'static str {
        match self {
            Self::Matvec { format, .. } => format.kernel_names()[0],
            Self::MatvecSwiglu(format) => format.kernel_names()[1],
            Self::RmsNorm => "rms_norm",
            Self::NormRope { .. } => "norm_rope",
            Self::ConvertHalf => "convert_half",
            Self::AttentionChunk { .. } => "attention_chunk",
            Self::AttentionCombine => "attention_combine",
            Self::ShortConv => "short_conv",
            Self::ShortConvBatch => "short_conv_batch",
            Self::ShortConvHistory => "short_conv_history",
            Self::Matmul(Format::Q8_0, Store::Accumulate) => "matmul_q8_0_accumulate",
            Self::Matmul(Format::Q8_0, Store::Swiglu) => "matmul_q8_0_swiglu",
            Self::Matmul(format, Store::Overwrite | Store::Accumulate | Store::Swiglu) => {
                format.kernel_names()[2]
            }
            Self::Copy => "copy",
            Self::Embed(format) => format.kernel_names()[3],
            Self::Argmax => "argmax",
        }
    }
}

/// The values of the function constants in `kernels.metal`.
#[derive(Debug, Clone, Copy)]
struct Constants {
    rows_per_threadgroup: u32,
    fuse_norm: bool,
    accumulate: bool,
    swiglu_store: bool,
}

/// The pipelines of the kernels that read one weight format.
#[derive(Debug)]
struct FormatPipelines {
    /// Indexed by whether the launch fuses the norm and whether it accumulates.
    matvec: [[Pipeline; 2]; 2],
    matvec_swiglu: Pipeline,
    /// Indexed by [`Store`].
    matmul: [Pipeline; 3],
    embed: Pipeline,
}

impl FormatPipelines {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        format: Format,
    ) -> Result<Self, Error> {
        let [matvec_name, swiglu_name, matmul_name, embed_name] = format.kernel_names();
        let matvec = |fuse_norm, accumulate| {
            let constants = Constants {
                rows_per_threadgroup: MATVEC_Q8_0_ROWS_PER_THREADGROUP,
                fuse_norm,
                accumulate,
                swiglu_store: false,
            };
            make_pipeline(device, library, matvec_name, Some(constants))
        };
        let matmul = |accumulate, swiglu_store| {
            let constants = Constants {
                rows_per_threadgroup: 0,
                fuse_norm: false,
                accumulate,
                swiglu_store,
            };
            make_pipeline(device, library, matmul_name, Some(constants))
        };
        let swiglu = Constants {
            rows_per_threadgroup: MATVEC_Q8_0_ROWS_PER_THREADGROUP,
            fuse_norm: true,
            accumulate: false,
            swiglu_store: false,
        };
        Ok(Self {
            matvec: [
                [matvec(false, false)?, matvec(false, true)?],
                [matvec(true, false)?, matvec(true, true)?],
            ],
            matvec_swiglu: make_pipeline(device, library, swiglu_name, Some(swiglu))?,
            matmul: [
                matmul(false, false)?,
                matmul(true, false)?,
                matmul(false, true)?,
            ],
            embed: make_pipeline(device, library, embed_name, None)?,
        })
    }
}

/// Every compute pipeline, created when the backend opens.
#[derive(Debug)]
struct Pipelines {
    /// Indexed by [`Format::index`].
    formats: Vec<FormatPipelines>,
    rms_norm: Pipeline,
    /// Indexed by whether the output is half precision.
    norm_rope: [Pipeline; 2],
    convert_half: Pipeline,
    /// Indexed by whether the caches hold half precision.
    attention_chunk: [Pipeline; 2],
    attention_combine: Pipeline,
    short_conv: Pipeline,
    short_conv_batch: Pipeline,
    short_conv_history: Pipeline,
    copy: Pipeline,
    argmax: Pipeline,
}

impl Pipelines {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, Error> {
        let plain = |name| make_pipeline(device, library, name, None);
        Ok(Self {
            formats: Format::ALL
                .into_iter()
                .map(|format| FormatPipelines::new(device, library, format))
                .collect::<Result<_, _>>()?,
            rms_norm: plain("rms_norm")?,
            norm_rope: [plain("norm_rope_f32")?, plain("norm_rope_f16")?],
            convert_half: plain("convert_half")?,
            attention_chunk: [plain("attention_chunk_f32")?, plain("attention_chunk_f16")?],
            attention_combine: plain("attention_combine")?,
            short_conv: plain("short_conv")?,
            short_conv_batch: plain("short_conv_batch")?,
            short_conv_history: plain("short_conv_history")?,
            copy: plain("copy_floats")?,
            argmax: plain("argmax")?,
        })
    }

    fn get(&self, kernel: Kernel) -> &ProtocolObject<dyn MTLComputePipelineState> {
        match kernel {
            Kernel::Matvec {
                format,
                norm,
                accumulate,
            } => &self.formats[format.index()].matvec[usize::from(norm)][usize::from(accumulate)],
            Kernel::MatvecSwiglu(format) => &self.formats[format.index()].matvec_swiglu,
            Kernel::RmsNorm => &self.rms_norm,
            Kernel::NormRope { half } => &self.norm_rope[usize::from(half)],
            Kernel::ConvertHalf => &self.convert_half,
            Kernel::AttentionChunk { half } => &self.attention_chunk[usize::from(half)],
            Kernel::AttentionCombine => &self.attention_combine,
            Kernel::ShortConv => &self.short_conv,
            Kernel::ShortConvBatch => &self.short_conv_batch,
            Kernel::ShortConvHistory => &self.short_conv_history,
            Kernel::Matmul(format, store) => {
                let index = match store {
                    Store::Overwrite => 0,
                    Store::Accumulate => 1,
                    Store::Swiglu => 2,
                };
                &self.formats[format.index()].matmul[index]
            }
            Kernel::Copy => &self.copy,
            Kernel::Embed(format) => &self.formats[format.index()].embed,
            Kernel::Argmax => &self.argmax,
        }
    }
}

/// Return the text of a Metal error.
fn describe(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// Set the unsigned function constant `index` of `values` to `value`.
fn set_uint_constant(values: &MTLFunctionConstantValues, value: u32, index: usize) {
    let pointer = NonNull::from(&value).cast();
    // SAFETY: `pointer` points at a live `u32`, the size of `MTLDataType::UInt`, and Metal copies
    // the value before returning.
    unsafe { values.setConstantValue_type_atIndex(pointer, MTLDataType::UInt, index) };
}

/// Set the boolean function constant `index` of `values` to `value`.
fn set_bool_constant(values: &MTLFunctionConstantValues, value: bool, index: usize) {
    let pointer = NonNull::from(&value).cast();
    // SAFETY: `pointer` points at a live `bool`, one byte like `MTLDataType::Bool`, and Metal
    // copies the value before returning.
    unsafe { values.setConstantValue_type_atIndex(pointer, MTLDataType::Bool, index) };
}

/// Return a pipeline for the kernel `name` in `library`, specialized with `constants` when given.
fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
    constants: Option<Constants>,
) -> Result<Pipeline, Error> {
    let fail = |message| Error::Pipeline { name, message };
    let ns_name = NSString::from_str(name);
    let function = match constants {
        Some(constants) => {
            let values = MTLFunctionConstantValues::new();
            set_uint_constant(&values, constants.rows_per_threadgroup, 0);
            set_bool_constant(&values, constants.fuse_norm, 1);
            set_bool_constant(&values, constants.accumulate, 2);
            set_bool_constant(&values, constants.swiglu_store, 3);
            library
                .newFunctionWithName_constantValues_error(&ns_name, &values)
                .map_err(|error| fail(describe(&error)))?
        }
        None => library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| fail("no such kernel".to_owned()))?,
    };
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| fail(describe(&error)))
}

/// One argument of a kernel launch, bound to the slot matching its position.
enum Arg<'a> {
    /// A buffer range of `bytes` bytes starting at the view.
    Buffer {
        view: View<'a>,
        bytes: usize,
    },
    U32(u32),
    F32(f32),
}

/// Return an argument that binds `bytes` bytes of the buffer at `view`.
fn buffer(view: View<'_>, bytes: usize) -> Arg<'_> {
    Arg::Buffer { view, bytes }
}

/// How a launch spreads its threads.
enum Dispatch {
    /// A grid of threadgroups of the given size.
    Threadgroups([usize; 3], [usize; 3]),
    /// A grid of threads in threadgroups of the given size.
    Threads([usize; 3], [usize; 3]),
}

/// A launch the backend records and, while profiling, attributes to a profile entry.
struct Launch<'a, 'b> {
    kernel: Kernel,
    args: &'b [Arg<'a>],
    dispatch: Dispatch,
    n_rows: u32,
    n_cols: u32,
    weight_bytes: u64,
}

/// The command buffer and encoder that launches record into.
#[derive(Debug)]
struct Recording {
    command_buffer: CommandBuffer,
    encoder: Encoder,
}

/// An open Metal device with gip's kernels compiled.
#[derive(Debug)]
pub struct Metal {
    id: u64,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipelines: Pipelines,
    recording: Option<Recording>,
    /// Submitted command buffers not yet waited on, oldest first.
    in_flight: VecDeque<CommandBuffer>,
    /// The number of command buffers submitted.
    committed: u64,
    /// The number of command buffers waited on.
    completed: u64,
    profiling: bool,
    profile: Vec<ProfileEntry>,
}

impl Metal {
    /// Open the default Metal device and compile gip's kernels.
    ///
    /// The kernels compile in Metal's safe math mode, which needs macOS 15 or newer.
    pub fn open() -> Result<Self, Error> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
            let options = MTLCompileOptions::new();
            options.setMathMode(MTLMathMode::Safe);
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
                .map_err(|error| Error::Compile(describe(&error)))?;
            let queue = device.newCommandQueue().ok_or(Error::Queue)?;
            let pipelines = Pipelines::new(&device, &library)?;
            Ok(Self {
                id: NEXT_BACKEND_ID.fetch_add(1, Ordering::Relaxed),
                device,
                queue,
                pipelines,
                recording: None,
                in_flight: VecDeque::new(),
                committed: 0,
                completed: 0,
                profiling: false,
                profile: Vec::new(),
            })
        })
    }

    /// Return a new zeroed buffer of `len` bytes.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer, Error> {
        let raw = self
            .device
            .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(len))?;
        let contents = raw.contents().cast::<u8>();
        // SAFETY: `contents` points at the `raw.length()` bytes of a buffer that no command buffer
        // has used yet.
        unsafe { contents.write_bytes(0, raw.length()) };
        Ok(Buffer {
            owner: self.id,
            raw,
            len,
        })
    }

    /// Return a buffer over the bytes of `mapping` with no copy.
    ///
    /// The buffer holds `mapping` alive until Metal frees the buffer, which happens only after
    /// every command buffer that reads the buffer has finished.
    pub fn wrap_mapping(&self, mapping: Arc<Mmap>) -> Result<Buffer, Error> {
        let len = mapping.len();
        let page = page_size().ok_or(Error::Wrap(len))?;
        let address = mapping.as_ptr();
        if len == 0 || !(address as usize).is_multiple_of(page) {
            return Err(Error::Wrap(len));
        }
        let rounded = len.checked_next_multiple_of(page).ok_or(Error::Wrap(len))?;
        let pointer = NonNull::new(address.cast_mut().cast::<c_void>()).ok_or(Error::Wrap(len))?;
        let deallocator = RcBlock::new(move |_: NonNull<c_void>, _: usize| {
            let _ = &mapping;
        });
        // SAFETY: `pointer` is page-aligned. memmap2 maps whole pages, so all `rounded` bytes lie
        // inside the mapping. The deallocator block holds an `Arc` of the mapping, so the pages
        // stay mapped until Metal frees the buffer. gip's kernels only read weight buffers, which
        // matches the mapping's read-only protection.
        let raw = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    rounded,
                    MTLResourceOptions::StorageModeShared,
                    Some(&deallocator),
                )
        }
        .ok_or(Error::Wrap(len))?;
        Ok(Buffer {
            owner: self.id,
            raw,
            len,
        })
    }

    /// Copy `data` into the buffer at `view`.
    pub fn write<T: Element>(&self, view: View<'_>, data: &[T]) -> Result<(), Error> {
        let bytes = size_of_val(data);
        let destination = self.cpu_access(view, bytes)?;
        // SAFETY: `cpu_access` checked that the `bytes` bytes at `destination` lie inside the
        // buffer and that no command buffer is in flight to touch them. `data` is a separate
        // allocation, and both pointers are read as bytes, which need no alignment.
        unsafe {
            destination.copy_from_nonoverlapping(NonNull::from(data).cast::<u8>(), bytes);
        }
        Ok(())
    }

    /// Copy bytes from the buffer at `view` into `out`.
    pub fn read<T: Element>(&self, view: View<'_>, out: &mut [T]) -> Result<(), Error> {
        let bytes = size_of_val(out);
        let source = self.cpu_access(view, bytes)?;
        // SAFETY: `cpu_access` checked that the `bytes` bytes at `source` lie inside the buffer
        // and that no command buffer is in flight to write them. `out` is a separate allocation,
        // every bit pattern is a valid `T`, and both pointers are used as bytes.
        unsafe {
            NonNull::from(out)
                .cast::<u8>()
                .copy_from_nonoverlapping(source, bytes);
        }
        Ok(())
    }

    /// Return the CPU address of `bytes` bytes at `view` after checking that the CPU may touch
    /// them.
    fn cpu_access(&self, view: View<'_>, bytes: usize) -> Result<NonNull<u8>, Error> {
        let buffer = view.buffer;
        if buffer.owner != self.id {
            return Err(Error::ForeignBuffer);
        }
        if !self.in_flight.is_empty() {
            return Err(Error::Busy);
        }
        let access = Error::Access {
            offset: view.offset,
            needed: bytes,
            len: buffer.len,
        };
        let end = view.offset.checked_add(bytes).ok_or(access)?;
        if end > buffer.len {
            return Err(Error::Access {
                offset: view.offset,
                needed: bytes,
                len: buffer.len,
            });
        }
        let contents = buffer.raw.contents().cast::<u8>();
        // SAFETY: `view.offset` plus `bytes` stays within the buffer's length, so the result
        // points inside the buffer's contents.
        Ok(unsafe { contents.add(view.offset) })
    }

    /// Start recording launches into a new command buffer. Launches may run at the same time
    /// until a barrier separates them.
    pub fn begin(&mut self) -> Result<(), Error> {
        if self.recording.is_some() {
            return Err(Error::AlreadyRecording);
        }
        // Command buffers and encoders arrive autoreleased. The pool frees the temporary
        // references on every step of a long decode.
        let recording = autoreleasepool(|_| {
            let command_buffer = self.queue.commandBuffer()?;
            let encoder = command_buffer
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)?;
            Some(Recording {
                command_buffer,
                encoder,
            })
        });
        self.recording = Some(recording.ok_or(Error::CommandBuffer)?);
        Ok(())
    }

    /// Drop the open command buffer and its launches without running them.
    pub fn discard(&mut self) {
        if let Some(recording) = self.recording.take() {
            recording.encoder.endEncoding();
        }
    }

    /// Record a barrier. Every launch recorded after the barrier sees the results of every
    /// launch recorded before it.
    pub fn barrier(&mut self) {
        // While profiling, each launch waits for its own command buffer, which orders the
        // launches already.
        if !self.profiling
            && let Some(recording) = &self.recording
        {
            recording
                .encoder
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }

    /// Submit the recorded launches to the GPU without waiting and return a ticket to wait on.
    /// Command buffers run in the order they are submitted.
    pub fn commit(&mut self) -> Result<Ticket, Error> {
        let recording = self.recording.take().ok_or(Error::NotRecording)?;
        recording.encoder.endEncoding();
        recording.command_buffer.commit();
        self.in_flight.push_back(recording.command_buffer);
        self.committed += 1;
        Ok(Ticket(self.committed))
    }

    /// Wait until the command buffer with `ticket` and every one submitted before it has
    /// finished. Return the GPU time in seconds of the command buffers waited on.
    pub fn wait(&mut self, ticket: Ticket) -> Result<f64, Error> {
        let mut gpu_seconds = 0.0;
        let mut failure = None;
        // In-flight command buffers finish in submission order, so the oldest one comes next.
        while self.completed < ticket.0
            && let Some(command_buffer) = self.in_flight.pop_front()
        {
            command_buffer.waitUntilCompleted();
            self.completed += 1;
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                failure = Some(
                    command_buffer
                        .error()
                        .map_or_else(|| "unknown error".to_owned(), |error| describe(&error)),
                );
            }
            gpu_seconds += command_buffer.GPUEndTime() - command_buffer.GPUStartTime();
        }
        match failure {
            Some(message) => Err(Error::Execution(message)),
            None => Ok(gpu_seconds),
        }
    }

    /// Run the recorded launches and wait for them. Return their GPU time in seconds.
    pub fn end(&mut self) -> Result<f64, Error> {
        let ticket = self.commit()?;
        self.wait(ticket)
    }

    /// Turn profiling on or off. Turning it on clears the profile.
    ///
    /// While profiling is on, every launch runs in a command buffer of its own and waits for it,
    /// so launches run slower and their GPU times add up.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if enabled {
            self.profile.clear();
        }
    }

    /// Return the profile entries recorded while profiling was last on.
    pub fn profile(&self) -> &[ProfileEntry] {
        &self.profile
    }

    /// Check the arguments of `launch`, then record it.
    fn launch(&mut self, launch: &Launch<'_, '_>) -> Result<(), Error> {
        let name = launch.kernel.name();
        for arg in launch.args {
            if let Arg::Buffer { view, bytes } = *arg {
                check_range(name, view, bytes, self.id)?;
            }
        }

        let pipeline = self.pipelines.get(launch.kernel);
        let own = if self.profiling {
            autoreleasepool(|_| {
                let command_buffer = self.queue.commandBuffer()?;
                let encoder = command_buffer.computeCommandEncoder()?;
                Some(Recording {
                    command_buffer,
                    encoder,
                })
            })
            .map(Some)
            .ok_or(Error::CommandBuffer)?
        } else {
            None
        };
        let encoder = match (&own, &self.recording) {
            (Some(recording), _) | (None, Some(recording)) => &recording.encoder,
            (None, None) => return Err(Error::NotRecording),
        };

        encoder.setComputePipelineState(pipeline);
        for (index, arg) in launch.args.iter().enumerate() {
            bind(encoder, arg, index);
        }
        match launch.dispatch {
            Dispatch::Threadgroups(grid, group) => {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(group));
            }
            Dispatch::Threads(grid, group) => {
                encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(group));
            }
        }

        if let Some(recording) = own {
            recording.encoder.endEncoding();
            recording.command_buffer.commit();
            recording.command_buffer.waitUntilCompleted();
            let seconds =
                recording.command_buffer.GPUEndTime() - recording.command_buffer.GPUStartTime();
            self.add_profile(launch, seconds);
        }
        Ok(())
    }

    /// Add `seconds` of GPU time for `launch` to its profile entry.
    fn add_profile(&mut self, launch: &Launch<'_, '_>, seconds: f64) {
        let name = launch.kernel.name();
        let index = self
            .profile
            .iter()
            .position(|e| e.name == name && e.n_rows == launch.n_rows && e.n_cols == launch.n_cols);
        let entry = if let Some(index) = index {
            &mut self.profile[index]
        } else {
            self.profile.push(ProfileEntry {
                name,
                n_rows: launch.n_rows,
                n_cols: launch.n_cols,
                calls: 0,
                seconds: 0.0,
                bytes: 0,
            });
            let last = self.profile.len() - 1;
            &mut self.profile[last]
        };
        entry.calls += 1;
        entry.seconds += seconds;
        entry.bytes += launch.weight_bytes;
    }

    /// Record a multiply of the matrix at `weights`, which has `n_rows` rows of `n_cols` weights of
    /// `format`, by the `n_cols` floats at `x`. The `n_rows` results go to `y`.
    pub fn matvec(
        &mut self,
        format: Format,
        weights: View<'_>,
        n_rows: u32,
        n_cols: u32,
        x: View<'_>,
        y: View<'_>,
        options: MatvecOptions<'_>,
    ) -> Result<(), Error> {
        let rows = to_usize(n_rows);
        let cols = to_usize(n_cols);
        let matrix_bytes = format.matrix_bytes(rows, cols);
        let mut args = vec![
            buffer(weights, matrix_bytes),
            buffer(x, product(&[cols, FLOAT_BYTES])),
            buffer(y, product(&[rows, FLOAT_BYTES])),
            Arg::U32(n_rows),
            Arg::U32(n_cols),
        ];
        if let Some(norm) = options.norm {
            args.push(buffer(norm.weight, product(&[cols, FLOAT_BYTES])));
            args.push(Arg::F32(norm.eps));
        }
        self.launch(&Launch {
            kernel: Kernel::Matvec {
                format,
                norm: options.norm.is_some(),
                accumulate: options.accumulate,
            },
            args: &args,
            dispatch: matvec_dispatch(format, n_rows, n_cols),
            n_rows,
            n_cols,
            weight_bytes: to_u64(matrix_bytes),
        })
    }

    /// Record the multiply of the matrices at `gate` and `up`, which each have `n_rows` rows of
    /// `n_cols` weights of `format`, by the `n_cols` floats at `x` after `norm`. `y` receives SiLU
    /// of each gate result times the matching up result.
    pub fn matvec_swiglu(
        &mut self,
        format: Format,
        gate: View<'_>,
        up: View<'_>,
        n_rows: u32,
        n_cols: u32,
        x: View<'_>,
        norm: Norm<'_>,
        y: View<'_>,
    ) -> Result<(), Error> {
        let rows = to_usize(n_rows);
        let cols = to_usize(n_cols);
        let matrix_bytes = format.matrix_bytes(rows, cols);
        let args = [
            buffer(gate, matrix_bytes),
            buffer(up, matrix_bytes),
            buffer(x, product(&[cols, FLOAT_BYTES])),
            buffer(y, product(&[rows, FLOAT_BYTES])),
            Arg::U32(n_rows),
            Arg::U32(n_cols),
            buffer(norm.weight, product(&[cols, FLOAT_BYTES])),
            Arg::F32(norm.eps),
        ];
        self.launch(&Launch {
            kernel: Kernel::MatvecSwiglu(format),
            args: &args,
            dispatch: matvec_dispatch(format, n_rows, n_cols),
            n_rows,
            n_cols,
            weight_bytes: to_u64(matrix_bytes).saturating_mul(2),
        })
    }

    /// Record an RMS normalization of each of the `n_rows` rows of `n` floats at `x`, scaled by
    /// the `n` floats at `weight`, into the matching rows of `out`.
    pub fn rms_norm(
        &mut self,
        x: View<'_>,
        weight: View<'_>,
        out: View<'_>,
        n: u32,
        n_rows: u32,
        eps: f32,
    ) -> Result<(), Error> {
        let rows_bytes = product(&[to_usize(n), to_usize(n_rows), FLOAT_BYTES]);
        let args = [
            buffer(x, rows_bytes),
            buffer(weight, product(&[to_usize(n), FLOAT_BYTES])),
            buffer(out, rows_bytes),
            Arg::U32(n),
            Arg::F32(eps),
        ];
        self.launch(&Launch {
            kernel: Kernel::RmsNorm,
            args: &args,
            dispatch: Dispatch::Threadgroups([to_usize(n_rows), 1, 1], [REDUCE_THREADS, 1, 1]),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record the per-head RMS normalization and rotary embedding of the `n_heads` heads of
    /// `head_dim` floats of each of `n_tokens` tokens at `src`.
    ///
    /// Token `i` sits at position `pos + i`, `src_stride` floats into `src` and `dst_stride`
    /// elements into `dst`. The result goes to `dst` as half-precision numbers when `dst_half`
    /// is true and as floats otherwise. `src` and `dst` may be the same floats.
    pub fn norm_rope(
        &mut self,
        src: View<'_>,
        dst: View<'_>,
        dst_half: bool,
        weight: View<'_>,
        n_heads: u32,
        head_dim: u32,
        pos: u32,
        n_tokens: u32,
        src_stride: u32,
        dst_stride: u32,
        theta: f32,
        eps: f32,
    ) -> Result<(), Error> {
        let head_elements = product(&[to_usize(n_heads), to_usize(head_dim)]);
        let extent = |stride: u32| {
            product(&[to_usize(n_tokens.saturating_sub(1)), to_usize(stride)])
                .saturating_add(head_elements)
        };
        let dst_element = if dst_half { HALF_BYTES } else { FLOAT_BYTES };
        let args = [
            buffer(src, product(&[extent(src_stride), FLOAT_BYTES])),
            buffer(dst, product(&[extent(dst_stride), dst_element])),
            buffer(weight, product(&[to_usize(head_dim), FLOAT_BYTES])),
            Arg::U32(head_dim),
            Arg::U32(pos),
            Arg::F32(theta),
            Arg::F32(eps),
            Arg::U32(src_stride),
            Arg::U32(dst_stride),
        ];
        self.launch(&Launch {
            kernel: Kernel::NormRope { half: dst_half },
            args: &args,
            dispatch: Dispatch::Threadgroups(
                [to_usize(n_heads), to_usize(n_tokens), 1],
                [to_usize(head_dim), 1, 1],
            ),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record a conversion of the `n` floats at `src` to half precision at `dst`.
    pub fn convert_half(&mut self, src: View<'_>, dst: View<'_>, n: u32) -> Result<(), Error> {
        let args = [
            buffer(src, product(&[to_usize(n), FLOAT_BYTES])),
            buffer(dst, product(&[to_usize(n), HALF_BYTES])),
            Arg::U32(n),
        ];
        self.launch(&Launch {
            kernel: Kernel::ConvertHalf,
            args: &args,
            dispatch: elementwise(n),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record causal attention of `n_queries` queries at `q`, each of `n_heads` heads, into
    /// `out`.
    ///
    /// Query `i` sits at position `first_pos + i` and attends over positions 0 through
    /// `first_pos + i` of `k_cache` and `v_cache`. Each cache holds `n_kv_heads` heads of
    /// `head_dim` numbers per position, in half precision when `kv_half` is true and as floats
    /// otherwise. `scratch` holds the floats [`attention_scratch_floats`] returns for `n_ctx`
    /// positions and `n_queries` queries.
    pub fn attention(
        &mut self,
        q: View<'_>,
        k_cache: View<'_>,
        v_cache: View<'_>,
        kv_half: bool,
        scratch: View<'_>,
        out: View<'_>,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        first_pos: u32,
        n_queries: u32,
        n_ctx: u32,
    ) -> Result<(), Error> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_keys = first_pos.saturating_add(n_queries);
        // The last query sees the most keys, so its chunks cover every query.
        let n_chunks = n_keys.div_ceil(ATTENTION_CHUNK);
        let max_chunks = n_ctx.div_ceil(ATTENTION_CHUNK);
        let cache_element = if kv_half { HALF_BYTES } else { FLOAT_BYTES };
        let query_bytes = product(&[
            to_usize(n_queries),
            to_usize(n_heads),
            to_usize(head_dim),
            FLOAT_BYTES,
        ]);
        let cache_bytes = product(&[
            to_usize(n_keys),
            to_usize(n_kv_heads),
            to_usize(head_dim),
            cache_element,
        ]);
        let scratch_bytes = product(&[
            attention_scratch_floats(n_heads, head_dim, n_ctx, n_queries),
            FLOAT_BYTES,
        ]);

        let chunk_args = [
            buffer(q, query_bytes),
            buffer(k_cache, cache_bytes),
            buffer(v_cache, cache_bytes),
            buffer(scratch, scratch_bytes),
            Arg::U32(n_heads),
            Arg::U32(n_kv_heads),
            Arg::U32(head_dim),
            Arg::U32(first_pos),
            Arg::U32(max_chunks),
            Arg::F32(scale),
        ];
        self.launch(&Launch {
            kernel: Kernel::AttentionChunk { half: kv_half },
            args: &chunk_args,
            dispatch: Dispatch::Threadgroups(
                [
                    to_usize(n_kv_heads),
                    to_usize(n_chunks),
                    to_usize(n_queries),
                ],
                [to_usize(ATTENTION_CHUNK), 1, 1],
            ),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })?;
        self.barrier();

        let combine_args = [
            buffer(scratch, scratch_bytes),
            buffer(out, query_bytes),
            Arg::U32(n_heads),
            Arg::U32(head_dim),
            Arg::U32(first_pos),
            Arg::U32(max_chunks),
        ];
        self.launch(&Launch {
            kernel: Kernel::AttentionCombine,
            args: &combine_args,
            dispatch: Dispatch::Threadgroups(
                [to_usize(n_heads), to_usize(n_queries), 1],
                [to_usize(head_dim), 1, 1],
            ),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record the gated short convolution of `n_tokens` tokens, each with `3 * n_embd` floats at
    /// `bcx`, with `kernel_size` taps per channel at `taps`.
    ///
    /// `history` holds the previous `kernel_size - 1` inputs and moves forward one token at a
    /// time. Each token's `n_embd` results go to `out`.
    pub fn short_conv(
        &mut self,
        bcx: View<'_>,
        taps: View<'_>,
        history: View<'_>,
        out: View<'_>,
        n_embd: u32,
        kernel_size: u32,
        n_tokens: u32,
    ) -> Result<(), Error> {
        let sizes = ConvSizes::new(n_embd, kernel_size, n_tokens);
        let args = [
            buffer(bcx, sizes.bcx),
            buffer(taps, sizes.taps),
            buffer(history, sizes.history),
            buffer(out, sizes.out),
            Arg::U32(n_embd),
            Arg::U32(kernel_size),
            Arg::U32(n_tokens),
        ];
        self.launch(&Launch {
            kernel: Kernel::ShortConv,
            args: &args,
            dispatch: elementwise(n_embd),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record the gated short convolution of `n_tokens` tokens at once, with the same arguments
    /// and results as [`Metal::short_conv`].
    pub fn short_conv_batch(
        &mut self,
        bcx: View<'_>,
        taps: View<'_>,
        history: View<'_>,
        out: View<'_>,
        n_embd: u32,
        kernel_size: u32,
        n_tokens: u32,
    ) -> Result<(), Error> {
        let sizes = ConvSizes::new(n_embd, kernel_size, n_tokens);
        let args = [
            buffer(bcx, sizes.bcx),
            buffer(taps, sizes.taps),
            buffer(history, sizes.history),
            buffer(out, sizes.out),
            Arg::U32(n_embd),
            Arg::U32(kernel_size),
            Arg::U32(n_tokens),
        ];
        self.launch(&Launch {
            kernel: Kernel::ShortConvBatch,
            args: &args,
            dispatch: Dispatch::Threads(
                [to_usize(n_embd), to_usize(n_tokens), 1],
                [ELEMENTWISE_THREADS, 1, 1],
            ),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })?;

        // The history update overwrites what the first pass reads.
        self.barrier();
        let history_args = [
            buffer(bcx, sizes.bcx),
            buffer(history, sizes.history),
            Arg::U32(n_embd),
            Arg::U32(kernel_size),
            Arg::U32(n_tokens),
        ];
        self.launch(&Launch {
            kernel: Kernel::ShortConvHistory,
            args: &history_args,
            dispatch: elementwise(n_embd),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record a multiply of the matrix at `weights`, which has `n_rows` rows of `n_cols` weights of
    /// `format`, by each of the `n_tokens` rows of `n_cols` floats at `x`. Each token's `n_rows`
    /// results combine with its row of `y` as `store` says.
    pub fn matmul(
        &mut self,
        format: Format,
        weights: View<'_>,
        n_rows: u32,
        n_cols: u32,
        x: View<'_>,
        y: View<'_>,
        n_tokens: u32,
        store: Store,
    ) -> Result<(), Error> {
        let rows = to_usize(n_rows);
        let cols = to_usize(n_cols);
        let tokens = to_usize(n_tokens);
        let row_tiles = n_rows.div_ceil(MATMUL_ROWS);
        let token_tiles = n_tokens.div_ceil(MATMUL_TOKENS);
        let matrix_bytes = format.matrix_bytes(rows, cols);
        let args = [
            buffer(weights, matrix_bytes),
            buffer(x, product(&[tokens, cols, FLOAT_BYTES])),
            buffer(y, product(&[tokens, rows, FLOAT_BYTES])),
            Arg::U32(n_rows),
            Arg::U32(n_cols),
            Arg::U32(n_tokens),
        ];
        self.launch(&Launch {
            kernel: Kernel::Matmul(format, store),
            args: &args,
            dispatch: Dispatch::Threadgroups(
                [to_usize(token_tiles), to_usize(row_tiles), 1],
                [SIMD_WIDTH * MATMUL_SIMDGROUPS, 1, 1],
            ),
            n_rows,
            n_cols,
            weight_bytes: to_u64(matrix_bytes).saturating_mul(u64::from(token_tiles)),
        })
    }

    /// Record a copy of `n` floats from `src` to `dst`.
    pub fn copy(&mut self, src: View<'_>, dst: View<'_>, n: u32) -> Result<(), Error> {
        let bytes = product(&[to_usize(n), FLOAT_BYTES]);
        let args = [buffer(src, bytes), buffer(dst, bytes), Arg::U32(n)];
        self.launch(&Launch {
            kernel: Kernel::Copy,
            args: &args,
            dispatch: elementwise(n),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record a dequantization of the rows named by the `n_tokens` token ids at `tokens` into
    /// `n_tokens` rows of `n_embd` floats at `out`.
    ///
    /// `weights` holds `n_vocab` rows of `n_embd` weights of `format`. Every token id must be below
    /// `n_vocab`.
    pub fn embed(
        &mut self,
        format: Format,
        weights: View<'_>,
        n_vocab: u32,
        tokens: View<'_>,
        out: View<'_>,
        n_embd: u32,
        n_tokens: u32,
    ) -> Result<(), Error> {
        // The Q8_0 kernel dequantizes one weight per thread, and the others 8.
        let per_thread = match format {
            Format::Q8_0 => 1,
            Format::Q4_0 | Format::Q4K | Format::Q6K => 8,
        };
        let args = [
            buffer(
                weights,
                format.matrix_bytes(to_usize(n_vocab), to_usize(n_embd)),
            ),
            buffer(tokens, product(&[to_usize(n_tokens), size_of::<u32>()])),
            buffer(
                out,
                product(&[to_usize(n_tokens), to_usize(n_embd), FLOAT_BYTES]),
            ),
            Arg::U32(n_embd),
        ];
        self.launch(&Launch {
            kernel: Kernel::Embed(format),
            args: &args,
            dispatch: Dispatch::Threads(
                [to_usize(n_embd) / per_thread, to_usize(n_tokens), 1],
                [ELEMENTWISE_THREADS, 1, 1],
            ),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }

    /// Record a store at `out`, one 32-bit integer, of the index of the largest of the `n`
    /// floats at `x`. Ties go to the lowest index.
    pub fn argmax(&mut self, x: View<'_>, out: View<'_>, n: u32) -> Result<(), Error> {
        let args = [
            buffer(x, product(&[to_usize(n), FLOAT_BYTES])),
            buffer(out, size_of::<u32>()),
            Arg::U32(n),
        ];
        self.launch(&Launch {
            kernel: Kernel::Argmax,
            args: &args,
            dispatch: Dispatch::Threadgroups([1, 1, 1], [ARGMAX_THREADS, 1, 1]),
            n_rows: 0,
            n_cols: 0,
            weight_bytes: 0,
        })
    }
}

impl Drop for Metal {
    fn drop(&mut self) {
        // Metal requires every encoder to end before it is released.
        self.discard();
        // The caller receives no GPU errors from a drop.
        let _ = self.wait(Ticket(self.committed));
    }
}

/// Return the floats of scratch [`Metal::attention`] needs for `n_queries` queries of `n_heads`
/// heads of `head_dim` floats over up to `n_ctx` positions.
pub fn attention_scratch_floats(n_heads: u32, head_dim: u32, n_ctx: u32, n_queries: u32) -> usize {
    // Each chunk of each head of each query keeps a weighted sum of values, a largest score, and
    // a sum of exponentials.
    product(&[
        to_usize(n_queries),
        to_usize(n_heads),
        to_usize(n_ctx.div_ceil(ATTENTION_CHUNK)),
        to_usize(head_dim).saturating_add(2),
    ])
}

/// The bytes each argument of a short convolution launch covers.
struct ConvSizes {
    bcx: usize,
    taps: usize,
    history: usize,
    out: usize,
}

impl ConvSizes {
    fn new(n_embd: u32, kernel_size: u32, n_tokens: u32) -> Self {
        let embd = to_usize(n_embd);
        let tokens = to_usize(n_tokens);
        Self {
            bcx: product(&[tokens, 3, embd, FLOAT_BYTES]),
            taps: product(&[to_usize(kernel_size), embd, FLOAT_BYTES]),
            history: product(&[to_usize(kernel_size.saturating_sub(1)), embd, FLOAT_BYTES]),
            out: product(&[tokens, embd, FLOAT_BYTES]),
        }
    }
}

/// Check that `bytes` bytes at `view` lie inside a buffer of the backend `owner`.
fn check_range(
    kernel: &'static str,
    view: View<'_>,
    bytes: usize,
    owner: u64,
) -> Result<(), Error> {
    if view.buffer.owner != owner {
        return Err(Error::ForeignBuffer);
    }
    match view.offset.checked_add(bytes) {
        Some(end) if end <= view.buffer.len => Ok(()),
        Some(_) | None => Err(Error::OutOfBounds {
            kernel,
            offset: view.offset,
            needed: bytes,
            len: view.buffer.len,
        }),
    }
}

/// Bind `arg` to slot `index` of `encoder`.
fn bind(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, arg: &Arg<'_>, index: usize) {
    match arg {
        Arg::Buffer { view, .. } => {
            // SAFETY: the launch checked that the range the kernel touches lies inside the buffer.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&view.buffer.raw), view.offset, index) };
        }
        Arg::U32(value) => bind_value(encoder, value, index),
        Arg::F32(value) => bind_value(encoder, value, index),
    }
}

/// Bind the bytes of `value` to slot `index` of `encoder`.
fn bind_value<T: Copy>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    // SAFETY: `value` points at `size_of::<T>()` live bytes, and Metal copies them before
    // returning.
    unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), size_of::<T>(), index) };
}

/// Return the dispatch of a matrix-vector launch over `n_rows` rows of `n_cols` weights of
/// `format`.
///
/// The Q8_0 and Q4_0 kernels split the columns of each threadgroup's rows among its simdgroups,
/// and wider rows get more simdgroups. The K-quant kernels give each simdgroup whole rows.
fn matvec_dispatch(format: Format, n_rows: u32, n_cols: u32) -> Dispatch {
    if matches!(format, Format::Q4K | Format::Q6K) {
        let rows_per_threadgroup = K_QUANT_SIMDGROUPS * K_QUANT_ROWS_PER_SIMDGROUP;
        return Dispatch::Threadgroups(
            [to_usize(n_rows.div_ceil(rows_per_threadgroup)), 1, 1],
            [SIMD_WIDTH * to_usize(K_QUANT_SIMDGROUPS), 1, 1],
        );
    }
    let threadgroups = n_rows.div_ceil(MATVEC_Q8_0_ROWS_PER_THREADGROUP);
    let simdgroups = if n_cols >= MATVEC_Q8_0_WIDE_COLS {
        MATVEC_Q8_0_WIDE_SIMDGROUPS
    } else {
        MATVEC_Q8_0_NARROW_SIMDGROUPS
    };
    Dispatch::Threadgroups(
        [to_usize(threadgroups), 1, 1],
        [SIMD_WIDTH * simdgroups, 1, 1],
    )
}

/// Return the dispatch of an elementwise launch over `n` elements.
fn elementwise(n: u32) -> Dispatch {
    Dispatch::Threads([to_usize(n), 1, 1], [ELEMENTWISE_THREADS, 1, 1])
}

fn size([width, height, depth]: [usize; 3]) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

/// Return the product of `factors`, or `usize::MAX` on overflow, which no buffer can hold.
fn product(factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1_usize, |acc, &factor| acc.checked_mul(factor))
        .unwrap_or(usize::MAX)
}

fn to_usize(n: u32) -> usize {
    // gip-metal builds only for macOS, whose targets all have 64-bit pointers.
    n as usize
}

fn to_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Return the system's memory page size.
fn page_size() -> Option<usize> {
    // SAFETY: `sysconf` has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).ok().filter(|&size| size > 0)
}
