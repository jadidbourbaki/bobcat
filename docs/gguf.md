# GGUF

GGUF is the file format that llama.cpp defined for quantized models.
Nearly every quantized model on Hugging Face ships as a GGUF file, so
gip reads GGUF directly and needs no conversion step. The `gip-gguf`
crate holds the parser.

## The file

A GGUF file has four parts in order:

1. A fixed header with the magic number `GGUF`, the format version, the
   tensor count, and the metadata count.
2. Metadata entries. Each entry is a key, a value type, and a value. A
   value is a number, a boolean, a string, or an array of one of those.
   The metadata holds the hyperparameters, the tokenizer's vocabulary
   and merges, the chat template, and sometimes recommended sampling
   settings.
3. Tensor headers. Each header gives a tensor's name, its shape, its
   element type, and its offset within the data section.
4. The data section, which starts at the next multiple of the file's
   alignment and holds every tensor's bytes.

gip reads versions 2 and 3, which differ from version 1 in using 64-bit
counts.

## The parser

`Gguf::parse` takes the file's bytes and returns a `Gguf` that owns
them. The bytes are usually a memory mapping, so parsing copies nothing
large. Metadata strings, arrays, and tensors are stored as byte ranges
into the file. `Gguf::string`, `Gguf::u32`, `Gguf::f32`,
`Gguf::string_array`, and `Gguf::array_u32` read metadata values on
demand. `Gguf::tensor` looks up a tensor by name, and
`Gguf::tensor_data` returns its bytes.

## Untrusted input

A GGUF file comes from whoever uploaded it, so the parser treats every
byte as hostile. llama.cpp has shipped memory-safety bugs in its GGUF
parser. gip's parser defends in three layers:

- The crate declares `#![forbid(unsafe_code)]`, so no parsing bug can
  corrupt memory.
- The crate warns on `clippy::indexing_slicing` and
  `clippy::arithmetic_side_effects`. Every read goes through `get` and
  every size through `checked_add` or `checked_mul`, so no input can
  reach a panic.
- Counts from the file are checked against the bytes that remain
  before any allocation. A file that claims four billion metadata
  entries fails at once, since each entry needs at least 13 bytes.

Every failure is a variant of `gip_gguf::Error` that names what went
wrong, such as `OutsideFile` for a tensor whose data runs past the end
of the file. `crates/gip-gguf/tests/malformed.rs` crafts broken files
and checks that each one yields the right error.

## Quantization formats

A quantized tensor stores its weights in blocks. Each block holds a few
low-precision integers and one or more scales that turn them back into
floats. `TensorType::block` gives each type's elements and bytes per
block.

| Type | Elements per block | Bytes per block | Bits per weight | Layout |
|---|---|---|---|---|
| `F32` | 1 | 4 | 32 | IEEE float |
| `F16` | 1 | 2 | 16 | IEEE half |
| `BF16` | 1 | 2 | 16 | bfloat16 |
| `Q8_0` | 32 | 34 | 8.5 | an fp16 scale and 32 int8 weights |
| `Q4_0` | 32 | 18 | 4.5 | an fp16 scale and 32 4-bit weights offset by 8 |
| `Q4_K` | 256 | 144 | 4.5 | eight 32-weight blocks, each with a 6-bit scale and minimum, under two fp16 super-scales |
| `Q6_K` | 256 | 210 | 6.6 | sixteen 16-weight blocks, each with an int8 scale, under one fp16 super-scale |

The `_K` types are llama.cpp's K-quants. A K-quant super-block of 256
weights shares one or two fp16 scales. Each smaller block inside the
super-block stores its own scale in 6 or 8 bits. The small scales cost fewer bits
than an fp16 scale per 32 weights, which leaves bits for better
precision.

A model's file name names its quantization mix. Liquid's `Q4_K_M` files
store most matrices in Q4_K and keep the token embedding and some
`attn_v` and `ffn_down` matrices in Q6_K. Liquid's `Q4_0` files keep
the token embedding in Q6_K. gip therefore reads Q4_0, Q4_K, and Q6_K
to run those files.

`crates/gip/src/scalar.rs` dequantizes every type with the arithmetic
of ggml's reference code in `ggml/src/ggml-quants.c`. The GPU kernels
are tested against those functions.
