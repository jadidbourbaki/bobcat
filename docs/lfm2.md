# LFM2

LFM2 is Liquid AI's family of hybrid language models. LFM2.5 keeps the
same architecture with more training. gip runs LFM2 first because the
models are small enough to run fast on a laptop and good enough to be
useful.

## The architecture

An LFM2 model is a stack of layers over a residual stream. The residual
stream is one vector of `n_embd` floats per token. Each layer reads the
stream, computes an update, and adds the update back.

Every layer has two blocks. Each block starts with an RMS
normalization. RMS normalization divides a vector by its root mean
square and multiplies it by a learned weight per element.

The first block mixes information across positions. In most layers the
mixer is a gated short convolution. In the remaining layers the mixer
is grouped-query attention. The file records which is which through
`lfm2.attention.head_count_kv`, which gives zero KV heads for a
convolution layer.

The second block is a SwiGLU feed-forward network. Two matrices,
`ffn_gate` and `ffn_up`, project the normalized stream to `n_ff`
floats. The gate goes through SiLU, which is `x * sigmoid(x)`, and
multiplies the up projection element by element. `ffn_down` projects
the product back to `n_embd`.

After the last layer, one more RMS normalization and the output matrix
turn the stream into logits, one score per vocabulary token. LFM2 ties
the output matrix to the token embedding matrix, so the file stores
the matrix once as `token_embd.weight`.

### The short convolution

A convolution layer projects the normalized stream to three vectors of
`n_embd` floats named B, C, and X. The layer multiplies B and X element
by element. Each channel then takes a weighted sum of its B times X
value over the current token and the previous `conv_kernel - 1`
tokens. `conv_kernel` is 3 in LFM2.5. The layer multiplies the sum by C
and applies the output projection.

The convolution state is small and fixed. Each convolution layer keeps
only the last `conv_kernel - 1` values of B times X per channel, however
long the sequence grows. The state has no position index, so the model
cannot rewind a convolution layer to an earlier token.

### The attention

An attention layer projects the stream to queries, keys, and values.
The query heads outnumber the key and value heads. Each group of
consecutive query heads shares one KV head, which shrinks the KV cache.
LFM2 normalizes each query and key head with RMS normalization, then
rotates it with rotary position embeddings in the GPT-NeoX layout,
where element `i` pairs with element `i + head_dim / 2`. The keys and
values go into the KV cache. Each query attends to every cached
position up to its own.

Few layers attend. LFM2.5-350M has 16 layers, of which 6 attend.
LFM2.5-2.6B has 30 layers, of which 8 attend. LFM2's KV cache is
therefore a fraction of the size that a transformer with attention in
every layer would need.

## The code

`crates/gip/src/lfm2.rs` loads the model and runs it on the CPU.

`Model::load` maps the file, parses it, reads the hyperparameters, and
finds every tensor. The loader checks each tensor's shape against the
hyperparameters, so a malformed model fails at load time with a named
tensor. A `Model` owns the parsed file and records each tensor as a
byte range inside the file. The weights stay in the file mapping.

`State` holds everything that changes as a sequence grows: the KV
cache, the convolution history, and scratch vectors. `State::new`
allocates all of it for a fixed context length, so a step allocates
nothing.

`Model::step` runs one token through every layer with the functions in
`crates/gip/src/scalar.rs`. The scalar functions use plain loops and
sum in double precision. The scalar functions run slowly and define the
correct output of every operation. Every GPU kernel is tested against
the scalar functions.

`Model::recommended_sampling` reads sampling settings from
`general.sampling.*` in the file. Missing settings fall back to the
values Liquid's model cards recommend.

## Testing against transformers

`tools/ref_dump.py` runs a model in Hugging Face transformers in
float32 on the CPU and writes its activations as raw little-endian
files: the prompt tokens, the embedding, every layer's output, the
final normalization, the logits, and a greedy continuation. With
`--gguf`, transformers loads the weights from a GGUF file, so both
sides see the same dequantized weights.

`crates/gip/tests/lfm2_reference.rs` runs gip on the same prompt and
compares every activation. The error is measured relative to the
largest magnitude in each reference row.

| Comparison | Tolerance |
|---|---|
| Scalar pass against transformers | 1e-4 |
| Metal pass with a float32 KV cache | 1e-4 |
| Metal pass with a half-precision KV cache | 5e-3 |

The test also checks that greedy decoding reproduces the
continuation from transformers token for token. A test reports `skip:`
and passes when its model file or reference dump is missing, so the
suite runs on machines without the models.

`models/` holds the model files. `models/ref/` holds the dumps. Git
ignores both directories.
