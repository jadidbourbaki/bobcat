//! Liquid AI's LFM2 models, loaded from GGUF files, and their forward pass on the CPU.
//!
//! Each LFM2 layer normalizes its input, applies either a gated short convolution or grouped-query
//! attention, adds the result to the residual stream, and then applies a SwiGLU feed-forward block
//! the same way. The last hidden state is normalized and multiplied by the output matrix, which
//! LFM2 ties to the token embeddings.

use std::path::Path;

use gip_gguf::{Gguf, Tensor, TensorType};

use crate::error::Error;
use crate::scalar;
use crate::storage::Storage;

const HEAD_COUNT_KV: &str = "lfm2.attention.head_count_kv";

/// The hyperparameters of an LFM2 model.
#[derive(Debug, Clone, PartialEq)]
pub struct Hyperparameters {
    /// The number of layers.
    pub n_layers: u32,
    /// The number of attention layers.
    pub n_attention_layers: u32,
    /// The number of convolution layers.
    pub n_conv_layers: u32,
    /// The width of the residual stream.
    pub n_embd: u32,
    /// The width of the feed-forward block.
    pub n_ff: u32,
    /// The number of query heads.
    pub n_heads: u32,
    /// The number of key and value heads.
    pub n_kv_heads: u32,
    /// The number of elements in each head.
    pub head_dim: u32,
    /// The number of tokens in the vocabulary.
    pub n_vocab: u32,
    /// The number of taps of each convolution.
    pub conv_kernel: u32,
    /// The base of the rotary embedding.
    pub rope_theta: f32,
    /// The epsilon of every RMS normalization.
    pub norm_eps: f32,
}

/// A 2D tensor with its shape as 32-bit counts.
#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    pub(crate) tensor: Tensor,
    pub(crate) n_rows: u32,
    pub(crate) n_cols: u32,
}

/// A 1D or 2D F32 tensor with its values copied out for the CPU. The GPU reads the tensor in the
/// file.
#[derive(Debug, Clone)]
pub(crate) struct Vector {
    pub(crate) tensor: Tensor,
    pub(crate) values: Vec<f32>,
}

/// The tensors of an attention layer.
#[derive(Debug, Clone)]
pub(crate) struct Attention {
    pub(crate) q: Matrix,
    pub(crate) k: Matrix,
    pub(crate) v: Matrix,
    pub(crate) output: Matrix,
    pub(crate) q_norm: Vector,
    pub(crate) k_norm: Vector,
}

/// The tensors of a gated short convolution layer.
#[derive(Debug, Clone)]
pub(crate) struct Conv {
    /// The taps of each channel, `conv_kernel` floats per channel.
    pub(crate) taps: Vector,
    pub(crate) in_proj: Matrix,
    pub(crate) out_proj: Matrix,
}

/// The operator that mixes positions in a layer.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "a model holds one mixer per layer, a few dozen at most"
)]
pub(crate) enum Mixer {
    Attention(Attention),
    Conv(Conv),
}

/// The tensors of one LFM2 layer.
#[derive(Debug, Clone)]
pub(crate) struct Layer {
    pub(crate) mixer: Mixer,
    /// The layer's index among layers with the same kind of mixer.
    pub(crate) cache_index: u32,
    pub(crate) attn_norm: Vector,
    pub(crate) ffn_norm: Vector,
    pub(crate) ffn_gate: Matrix,
    pub(crate) ffn_up: Matrix,
    pub(crate) ffn_down: Matrix,
}

/// An LFM2 model loaded from a GGUF file.
#[derive(Debug)]
pub struct Model {
    pub(crate) gguf: Gguf<Storage>,
    pub(crate) hyperparameters: Hyperparameters,
    pub(crate) layers: Vec<Layer>,
    pub(crate) token_embd: Matrix,
    pub(crate) output_norm: Vector,
    pub(crate) output: Matrix,
}

impl Model {
    /// Load the LFM2 model in the GGUF file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let storage = Storage::open(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        let gguf = Gguf::parse(storage).map_err(|source| Error::Gguf {
            path: path.to_owned(),
            source,
        })?;
        let mut hyperparameters = load_hyperparameters(&gguf)?;

        let mut layers = Vec::with_capacity(to_usize(hyperparameters.n_layers));
        for il in 0..hyperparameters.n_layers {
            layers.push(load_layer(&gguf, &mut hyperparameters, il)?);
        }

        let n_embd = u64::from(hyperparameters.n_embd);
        let embd_rows = match gguf.tensor("token_embd.weight") {
            Some(tensor) if tensor.shape().len() == 2 => tensor.ne(1),
            Some(_) | None => return Err(Error::MissingTensor("token_embd.weight".to_owned())),
        };
        hyperparameters.n_vocab = u32::try_from(embd_rows).map_err(|_| Error::Hyperparameters)?;
        let token_embd = require_matrix(&gguf, "token_embd.weight", n_embd, embd_rows)?;
        let output_norm = require_vector(&gguf, "token_embd_norm.weight", &[n_embd])?;
        // LFM2 ties the output matrix to the token embeddings. A file with its own output matrix
        // uses that matrix instead.
        let output = if gguf.tensor("output.weight").is_some() {
            require_matrix(&gguf, "output.weight", n_embd, embd_rows)?
        } else {
            token_embd.clone()
        };

        Ok(Self {
            gguf,
            hyperparameters,
            layers,
            token_embd,
            output_norm,
            output,
        })
    }

    /// Return the model's hyperparameters.
    pub fn hyperparameters(&self) -> &Hyperparameters {
        &self.hyperparameters
    }

    /// Report whether each layer mixes positions with attention, in layer order.
    pub fn attention_layers(&self) -> impl Iterator<Item = bool> {
        self.layers
            .iter()
            .map(|layer| matches!(layer.mixer, Mixer::Attention(_)))
    }

    /// Return the parsed model file, whose metadata holds the tokenizer and the chat template.
    pub fn gguf(&self) -> &Gguf<impl AsRef<[u8]>> {
        &self.gguf
    }

    /// Return the data bytes of `tensor`, which comes from this model's file.
    pub(crate) fn data(&self, tensor: &Tensor) -> &[u8] {
        &self.gguf.bytes()[tensor.data_range()]
    }

    /// Check that `token` lies inside the vocabulary.
    pub(crate) fn check_token(&self, token: u32) -> Result<(), Error> {
        let n_vocab = self.hyperparameters.n_vocab;
        if token < n_vocab {
            Ok(())
        } else {
            Err(Error::Token { token, n_vocab })
        }
    }

    fn matvec(&self, matrix: &Matrix, x: &[f32], y: &mut [f32]) {
        scalar::matvec(matrix.tensor.data_type(), self.data(&matrix.tensor), x, y);
    }

    /// Run the model on `token` at the next position of `state`.
    ///
    /// When `logits` is given, it receives the `n_vocab` logits. When `trace` is given, it
    /// receives the activations of this token, so it must hold one token.
    pub fn step(
        &self,
        state: &mut State,
        token: u32,
        logits: Option<&mut [f32]>,
        trace: Option<&mut Trace>,
    ) -> Result<(), Error> {
        let hp = &self.hyperparameters;
        let n_embd = to_usize(hp.n_embd);
        let pos = state.n_past;
        if pos >= state.n_ctx {
            return Err(Error::ContextFull {
                requested: 1,
                remaining: 0,
            });
        }
        self.check_token(token)?;
        if let Some(logits) = &logits
            && logits.len() != to_usize(hp.n_vocab)
        {
            return Err(Error::Argument("logits must hold n_vocab floats"));
        }
        if let Some(trace) = &trace
            && !trace.holds(hp, 1)
        {
            return Err(Error::Argument("the trace must hold one token"));
        }

        scalar::get_row(
            self.token_embd.tensor.data_type(),
            self.data(&self.token_embd.tensor),
            to_usize(token),
            &mut state.hidden,
        );
        let mut trace = trace;
        if let Some(trace) = trace.as_deref_mut() {
            trace.embedding.copy_from_slice(&state.hidden);
        }

        for (il, layer) in self.layers.iter().enumerate() {
            scalar::rms_norm(
                &state.hidden,
                &layer.attn_norm.values,
                hp.norm_eps,
                &mut state.normed,
            );
            match &layer.mixer {
                Mixer::Attention(attention) => {
                    self.attention_block(attention, layer.cache_index, state, pos);
                }
                Mixer::Conv(conv) => self.conv_block(conv, layer.cache_index, state),
            }
            add_in_place(&mut state.hidden, &state.block_out);

            scalar::rms_norm(
                &state.hidden,
                &layer.ffn_norm.values,
                hp.norm_eps,
                &mut state.normed,
            );
            self.matvec(&layer.ffn_gate, &state.normed, &mut state.gate);
            self.matvec(&layer.ffn_up, &state.normed, &mut state.up);
            for (gate, &up) in state.gate.iter_mut().zip(&state.up) {
                *gate = scalar::silu(*gate) * up;
            }
            self.matvec(&layer.ffn_down, &state.gate, &mut state.block_out);
            add_in_place(&mut state.hidden, &state.block_out);

            if let Some(trace) = trace.as_deref_mut() {
                trace.layers[il * n_embd..(il + 1) * n_embd].copy_from_slice(&state.hidden);
            }
        }

        scalar::rms_norm(
            &state.hidden,
            &self.output_norm.values,
            hp.norm_eps,
            &mut state.normed,
        );
        if let Some(trace) = trace {
            trace.final_norm.copy_from_slice(&state.normed);
        }
        if let Some(logits) = logits {
            self.matvec(&self.output, &state.normed, logits);
        }

        state.n_past += 1;
        Ok(())
    }

    /// Run the gated short convolution `conv` on `state.normed` and store the result in
    /// `state.block_out`.
    fn conv_block(&self, conv: &Conv, cache_index: u32, state: &mut State) {
        let n_embd = to_usize(self.hyperparameters.n_embd);
        let kernel = to_usize(self.hyperparameters.conv_kernel);
        let history_len = (kernel - 1) * n_embd;
        let history_start = to_usize(cache_index) * history_len;
        let history = &mut state.conv_history[history_start..history_start + history_len];

        // The input projection yields the gates B and C and the input X, in that order.
        self.matvec(&conv.in_proj, &state.normed, &mut state.bcx);
        let (b, rest) = state.bcx.split_at(n_embd);
        let (c, x) = rest.split_at(n_embd);

        // HISTORY holds B times X for the previous kernel - 1 tokens, oldest first. Tap K of a
        // channel multiplies the input from kernel - 1 - K tokens ago.
        for ch in 0..n_embd {
            let taps = &conv.taps.values[ch * kernel..(ch + 1) * kernel];
            let bx = b[ch] * x[ch];
            let mut sum = f64::from(taps[kernel - 1]) * f64::from(bx);
            for k in 0..kernel - 1 {
                sum += f64::from(taps[k]) * f64::from(history[k * n_embd + ch]);
            }
            for k in 0..kernel - 2 {
                history[k * n_embd + ch] = history[(k + 1) * n_embd + ch];
            }
            history[(kernel - 2) * n_embd + ch] = bx;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the double sum rounds to the float output"
            )]
            let convolved = sum as f32;
            state.conv_out[ch] = c[ch] * convolved;
        }

        self.matvec(&conv.out_proj, &state.conv_out, &mut state.block_out);
    }

    /// Run `attention` at position `pos` on `state.normed` and store the result in
    /// `state.block_out`.
    fn attention_block(
        &self,
        attention: &Attention,
        cache_index: u32,
        state: &mut State,
        pos: u32,
    ) {
        let hp = &self.hyperparameters;
        let head_dim = to_usize(hp.head_dim);
        let kv_dim = to_usize(hp.n_kv_heads) * head_dim;
        let group = to_usize(hp.n_heads / hp.n_kv_heads);
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let layer_len = to_usize(state.n_ctx) * kv_dim;
        let layer_start = to_usize(cache_index) * layer_len;
        let State {
            k_cache,
            v_cache,
            normed,
            q,
            k,
            v,
            attn,
            scores,
            block_out,
            ..
        } = state;
        let k_cache = &mut k_cache[layer_start..layer_start + layer_len];
        let v_cache = &mut v_cache[layer_start..layer_start + layer_len];

        self.matvec(&attention.q, normed, q);
        self.matvec(&attention.k, normed, k);
        self.matvec(&attention.v, normed, v);

        // LFM2 normalizes each query and key head before the rotation.
        for head in q.chunks_exact_mut(head_dim) {
            scalar::rms_norm_in_place(head, &attention.q_norm.values, hp.norm_eps);
            scalar::rope_neox(head, pos, hp.rope_theta);
        }
        for head in k.chunks_exact_mut(head_dim) {
            scalar::rms_norm_in_place(head, &attention.k_norm.values, hp.norm_eps);
            scalar::rope_neox(head, pos, hp.rope_theta);
        }

        let slot = to_usize(pos) * kv_dim;
        k_cache[slot..slot + kv_dim].copy_from_slice(k);
        v_cache[slot..slot + kv_dim].copy_from_slice(v);

        let n_keys = to_usize(pos) + 1;
        let scores = &mut scores[..n_keys];
        for (h, (qh, out)) in q
            .chunks_exact(head_dim)
            .zip(attn.chunks_exact_mut(head_dim))
            .enumerate()
        {
            // Query heads share KV heads in consecutive groups.
            let kv_offset = h / group * head_dim;
            for (t, score) in scores.iter_mut().enumerate() {
                let key = &k_cache[t * kv_dim + kv_offset..][..head_dim];
                *score = scale * scalar::dot(qh, key);
            }
            scalar::softmax(scores);

            for (d, out) in out.iter_mut().enumerate() {
                let sum: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(t, &p)| f64::from(p) * f64::from(v_cache[t * kv_dim + kv_offset + d]))
                    .sum();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the double sum rounds to the float output"
                )]
                let value = sum as f32;
                *out = value;
            }
        }

        self.matvec(&attention.output, attn, block_out);
    }
}

/// The per-sequence state of one LFM2 decode on the CPU: the KV cache of the attention layers,
/// the rolling inputs of the convolution layers, and scratch space.
#[derive(Debug, Clone)]
pub struct State {
    n_ctx: u32,
    n_past: u32,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    conv_history: Vec<f32>,
    hidden: Vec<f32>,
    normed: Vec<f32>,
    block_out: Vec<f32>,
    bcx: Vec<f32>,
    conv_out: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl State {
    /// Allocate the state of sequences of up to `n_ctx` tokens of `model`.
    pub fn new(model: &Model, n_ctx: u32) -> Result<Self, Error> {
        if n_ctx == 0 {
            return Err(Error::Argument("the context must hold a token"));
        }
        let hp = &model.hyperparameters;
        let n_embd = to_usize(hp.n_embd);
        let kv_dim = to_usize(hp.n_kv_heads) * to_usize(hp.head_dim);
        let q_dim = to_usize(hp.n_heads) * to_usize(hp.head_dim);
        let cache = to_usize(hp.n_attention_layers) * to_usize(n_ctx) * kv_dim;
        let conv = to_usize(hp.n_conv_layers) * to_usize(hp.conv_kernel - 1) * n_embd;
        Ok(Self {
            n_ctx,
            n_past: 0,
            k_cache: vec![0.0; cache],
            v_cache: vec![0.0; cache],
            conv_history: vec![0.0; conv],
            hidden: vec![0.0; n_embd],
            normed: vec![0.0; n_embd],
            block_out: vec![0.0; n_embd],
            bcx: vec![0.0; 3 * n_embd],
            conv_out: vec![0.0; n_embd],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn: vec![0.0; q_dim],
            scores: vec![0.0; to_usize(n_ctx)],
            gate: vec![0.0; to_usize(hp.n_ff)],
            up: vec![0.0; to_usize(hp.n_ff)],
        })
    }
}

/// Buffers that receive the activations of a run of tokens.
///
/// Each activation holds `n_embd` floats per token. `embedding` holds the token embeddings,
/// `layers` holds every token's output of layer 0, then of layer 1, and so on, and `final_norm`
/// holds the normalized last hidden states.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    /// The token embeddings.
    pub embedding: Vec<f32>,
    /// The output of each layer, layer by layer.
    pub layers: Vec<f32>,
    /// The normalized last hidden states.
    pub final_norm: Vec<f32>,
}

impl Trace {
    /// Return a zeroed trace of `n_tokens` tokens of a model with `hyperparameters`.
    pub fn new(hyperparameters: &Hyperparameters, n_tokens: u32) -> Self {
        let rows = to_usize(n_tokens) * to_usize(hyperparameters.n_embd);
        Self {
            embedding: vec![0.0; rows],
            layers: vec![0.0; to_usize(hyperparameters.n_layers) * rows],
            final_norm: vec![0.0; rows],
        }
    }

    /// Report whether the trace holds `n_tokens` tokens of a model with `hyperparameters`.
    pub(crate) fn holds(&self, hyperparameters: &Hyperparameters, n_tokens: u32) -> bool {
        let rows = to_usize(n_tokens) * to_usize(hyperparameters.n_embd);
        self.embedding.len() == rows
            && self.layers.len() == to_usize(hyperparameters.n_layers) * rows
            && self.final_norm.len() == rows
    }
}

/// Add `delta` to `h` element by element.
fn add_in_place(h: &mut [f32], delta: &[f32]) {
    for (h, &d) in h.iter_mut().zip(delta) {
        *h += d;
    }
}

/// Return `n` as a `usize`.
pub(crate) fn to_usize(n: u32) -> usize {
    // gip supports 64-bit targets only, where every u32 fits in a usize.
    n as usize
}

/// Return the integer metadata value `key` of `gguf`.
fn require_u32(gguf: &Gguf<Storage>, key: &'static str) -> Result<u32, Error> {
    gguf.u32(key).ok_or(Error::Metadata(key))
}

/// Return the floating-point metadata value `key` of `gguf`.
fn require_f32(gguf: &Gguf<Storage>, key: &'static str) -> Result<f32, Error> {
    gguf.f32(key).ok_or(Error::Metadata(key))
}

/// Read the hyperparameters of an LFM2 model from the metadata of `gguf`. The vocabulary size
/// and the layer counts stay zero until the tensors are loaded.
fn load_hyperparameters(gguf: &Gguf<Storage>) -> Result<Hyperparameters, Error> {
    let architecture = gguf
        .string("general.architecture")
        .ok_or(Error::Metadata("general.architecture"))?;
    if architecture != b"lfm2" {
        return Err(Error::Architecture(
            String::from_utf8_lossy(architecture).into_owned(),
        ));
    }

    let n_layers = require_u32(gguf, "lfm2.block_count")?;
    let n_embd = require_u32(gguf, "lfm2.embedding_length")?;
    let n_ff = require_u32(gguf, "lfm2.feed_forward_length")?;
    let n_heads = require_u32(gguf, "lfm2.attention.head_count")?;
    let conv_kernel = require_u32(gguf, "lfm2.shortconv.l_cache")?;
    let rope_theta = require_f32(gguf, "lfm2.rope.freq_base")?;
    let norm_eps = require_f32(gguf, "lfm2.attention.layer_norm_rms_epsilon")?;

    if n_layers == 0
        || n_embd == 0
        || n_ff == 0
        || n_heads == 0
        || !n_embd.is_multiple_of(n_heads)
        || conv_kernel < 2
    {
        return Err(Error::Hyperparameters);
    }
    let head_dim = n_embd / n_heads;
    if !head_dim.is_multiple_of(2) {
        return Err(Error::OddHeadDim(head_dim));
    }

    Ok(Hyperparameters {
        n_layers,
        n_attention_layers: 0,
        n_conv_layers: 0,
        n_embd,
        n_ff,
        n_heads,
        n_kv_heads: 0,
        head_dim,
        n_vocab: 0,
        conv_kernel,
        rope_theta,
        norm_eps,
    })
}

/// Return the number of KV heads of layer `layer`. The file holds one count per layer, and a
/// convolution layer has a count of zero. A single count applies to every layer.
fn layer_kv_heads(gguf: &Gguf<Storage>, n_layers: u32, layer: u32) -> Result<u32, Error> {
    if gguf.array_len(HEAD_COUNT_KV) == Some(u64::from(n_layers))
        && let Some(count) = gguf.array_u32(HEAD_COUNT_KV, u64::from(layer))
    {
        return Ok(count);
    }
    gguf.u32(HEAD_COUNT_KV)
        .ok_or(Error::Metadata(HEAD_COUNT_KV))
}

/// Find the tensor `name` in `gguf` and check that its shape is `want`.
fn require_tensor<'g>(
    gguf: &'g Gguf<Storage>,
    name: &str,
    want: &[u64],
) -> Result<&'g Tensor, Error> {
    let tensor = gguf
        .tensor(name)
        .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    if tensor.shape() != want {
        return Err(Error::TensorShape {
            name: name.to_owned(),
            got: tensor.shape().to_vec(),
            want: want.to_vec(),
        });
    }
    Ok(tensor)
}

/// Find the matrix `name` in `gguf`, which must have `n_rows` rows of `n_cols` elements.
fn require_matrix(
    gguf: &Gguf<Storage>,
    name: &str,
    n_cols: u64,
    n_rows: u64,
) -> Result<Matrix, Error> {
    let tensor = require_tensor(gguf, name, &[n_cols, n_rows])?;
    let (Ok(n_rows), Ok(n_cols)) = (u32::try_from(n_rows), u32::try_from(n_cols)) else {
        return Err(Error::Hyperparameters);
    };
    Ok(Matrix {
        tensor: tensor.clone(),
        n_rows,
        n_cols,
    })
}

/// Find the F32 tensor `name` in `gguf`, which must have the shape `want`, and copy out its
/// values.
fn require_vector(gguf: &Gguf<Storage>, name: &str, want: &[u64]) -> Result<Vector, Error> {
    let tensor = require_tensor(gguf, name, want)?;
    if tensor.data_type() != TensorType::F32 {
        return Err(Error::TensorType {
            name: name.to_owned(),
            data_type: tensor.data_type(),
        });
    }
    let bytes = gguf
        .tensor_data(tensor)
        .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    Ok(Vector {
        tensor: tensor.clone(),
        values: scalar::f32s(bytes).collect(),
    })
}

/// Find and check the tensors of layer `il` in `gguf`, counting the layer in `hyperparameters`.
fn load_layer(
    gguf: &Gguf<Storage>,
    hyperparameters: &mut Hyperparameters,
    il: u32,
) -> Result<Layer, Error> {
    let hp = &*hyperparameters;
    let n_embd = u64::from(hp.n_embd);
    let n_ff = u64::from(hp.n_ff);
    let name = |suffix: &str| format!("blk.{il}.{suffix}.weight");
    let n_kv_heads = layer_kv_heads(gguf, hp.n_layers, il)?;

    let attn_norm = require_vector(gguf, &name("attn_norm"), &[n_embd])?;
    let ffn_norm = require_vector(gguf, &name("ffn_norm"), &[n_embd])?;
    let ffn_gate = require_matrix(gguf, &name("ffn_gate"), n_embd, n_ff)?;
    let ffn_up = require_matrix(gguf, &name("ffn_up"), n_embd, n_ff)?;
    let ffn_down = require_matrix(gguf, &name("ffn_down"), n_ff, n_embd)?;

    let (mixer, cache_index) = if n_kv_heads == 0 {
        let taps = require_vector(
            gguf,
            &name("shortconv.conv"),
            &[u64::from(hp.conv_kernel), n_embd],
        )?;
        let in_proj = require_matrix(gguf, &name("shortconv.in_proj"), n_embd, 3 * n_embd)?;
        let out_proj = require_matrix(gguf, &name("shortconv.out_proj"), n_embd, n_embd)?;
        let index = hyperparameters.n_conv_layers;
        hyperparameters.n_conv_layers += 1;
        (
            Mixer::Conv(Conv {
                taps,
                in_proj,
                out_proj,
            }),
            index,
        )
    } else {
        if hp.n_kv_heads == 0 {
            hyperparameters.n_kv_heads = n_kv_heads;
        }
        let hp = &*hyperparameters;
        if n_kv_heads != hp.n_kv_heads || !hp.n_heads.is_multiple_of(n_kv_heads) {
            return Err(Error::KvHeads {
                layer: il,
                n_kv_heads,
            });
        }
        let head_dim = u64::from(hp.head_dim);
        let q_dim = u64::from(hp.n_heads) * head_dim;
        let kv_dim = u64::from(n_kv_heads) * head_dim;
        let attention = Attention {
            q: require_matrix(gguf, &name("attn_q"), n_embd, q_dim)?,
            k: require_matrix(gguf, &name("attn_k"), n_embd, kv_dim)?,
            v: require_matrix(gguf, &name("attn_v"), n_embd, kv_dim)?,
            output: require_matrix(gguf, &name("attn_output"), q_dim, n_embd)?,
            q_norm: require_vector(gguf, &name("attn_q_norm"), &[head_dim])?,
            k_norm: require_vector(gguf, &name("attn_k_norm"), &[head_dim])?,
        };
        let index = hyperparameters.n_attention_layers;
        hyperparameters.n_attention_layers += 1;
        (Mixer::Attention(attention), index)
    };

    Ok(Layer {
        mixer,
        cache_index,
        attn_norm,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}
