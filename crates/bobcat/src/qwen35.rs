//! Qwen's Qwen3.5 dense text models, loaded from GGUF files, and their forward pass on the CPU.
//!
//! Qwen3.5 interleaves two kinds of layer. Three of every four layers are Gated DeltaNet layers, a
//! linear attention that keeps a fixed-size state matrix per head, decays it at each token, and
//! updates it with the delta rule. The other layers run grouped-query attention whose output passes
//! through a sigmoid gate. Each layer normalizes its input, adds its mixer's output to the residual
//! stream, and then applies a SwiGLU block the same way.
//!
//! llama.cpp's converter writes the GGUF files. The converter bakes the one that Qwen3.5's
//! zero-centered norms add into their weights, stores `-exp(A_log)` as `ssm_a`, and appends a
//! multi-token prediction block after the model's layers, which the forward pass skips.

use std::path::Path;

use bobcat_gguf::Gguf;

use crate::error::Error;
use crate::lfm2::{
    Matrix, Sampling, Trace, Vector, require_f32, require_matrix, require_u32, require_vector,
    to_usize,
};
use crate::scalar;
use crate::storage::Storage;

#[cfg(target_os = "macos")]
pub use crate::qwen35_metal::{Checkpoint, Qwen35Metal};

/// transformers' Qwen3.5 code fixes the epsilon of the L2 norm of DeltaNet queries and keys.
const L2_NORM_EPS: f32 = 1e-6;

/// The hyperparameters of a Qwen3.5 model.
#[derive(Debug, Clone, PartialEq)]
pub struct Hyperparameters {
    /// The number of layers, without the multi-token prediction block.
    pub n_layers: u32,
    /// The number of attention layers.
    pub n_attention_layers: u32,
    /// The number of Gated DeltaNet layers.
    pub n_delta_layers: u32,
    /// The width of the residual stream.
    pub n_embd: u32,
    /// The width of the feed-forward blocks.
    pub n_ff: u32,
    /// The number of tokens in the vocabulary.
    pub n_vocab: u32,
    /// The number of query heads of the attention layers.
    pub n_heads: u32,
    /// The number of key and value heads of the attention layers.
    pub n_kv_heads: u32,
    /// The size of each attention head.
    pub head_dim: u32,
    /// The number of leading elements of each attention head that the rotary embedding rotates.
    pub n_rot: u32,
    /// The base of the rotary embedding's frequencies.
    pub rope_theta: f32,
    /// The epsilon of every RMS norm.
    pub norm_eps: f32,
    /// The taps of each DeltaNet layer's causal convolution.
    pub conv_kernel: u32,
    /// The number of key and query heads of each DeltaNet layer.
    pub n_k_heads: u32,
    /// The number of value heads of each DeltaNet layer, a multiple of the key heads.
    pub n_v_heads: u32,
    /// The size of each DeltaNet key and query head.
    pub k_head_dim: u32,
    /// The size of each DeltaNet value head.
    pub v_head_dim: u32,
}

impl Hyperparameters {
    /// Return the number of query and key elements of one DeltaNet token.
    pub(crate) fn key_dim(&self) -> u32 {
        self.n_k_heads * self.k_head_dim
    }

    /// Return the number of value elements of one DeltaNet token.
    pub(crate) fn value_dim(&self) -> u32 {
        self.n_v_heads * self.v_head_dim
    }

    /// Return the number of channels of a DeltaNet layer's convolution: the queries, the keys, and
    /// the values.
    pub(crate) fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }
}

/// The tensors of an attention layer.
#[derive(Debug, Clone)]
pub(crate) struct Attention {
    /// The query projection. Each head's 2 `head_dim` rows hold its queries, then its gate.
    pub(crate) q: Matrix,
    pub(crate) k: Matrix,
    pub(crate) v: Matrix,
    pub(crate) output: Matrix,
    pub(crate) q_norm: Vector,
    pub(crate) k_norm: Vector,
}

/// The tensors of a Gated DeltaNet layer.
#[derive(Debug, Clone)]
pub(crate) struct DeltaNet {
    /// The projection to the queries, keys, and values, in that order.
    pub(crate) qkv: Matrix,
    /// The projection to the gate of the output norm.
    pub(crate) gate: Matrix,
    /// The projection to each value head's update strength.
    pub(crate) beta: Matrix,
    /// The projection to each value head's decay rate.
    pub(crate) alpha: Matrix,
    /// The causal convolution, `conv_kernel` taps per channel, the oldest first.
    pub(crate) conv: Vector,
    /// The bias of each value head's decay rate.
    pub(crate) dt_bias: Vector,
    /// The negative decay scale of each value head, `-exp(A_log)`.
    pub(crate) a: Vector,
    /// The weight of the gated RMS norm of each value head's output.
    pub(crate) norm: Vector,
    pub(crate) output: Matrix,
}

/// The operator that mixes positions in a layer.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "a model holds one mixer per layer, a few dozen at most"
)]
pub(crate) enum Mixer {
    Attention(Attention),
    DeltaNet(DeltaNet),
}

/// The tensors of one Qwen3.5 layer.
#[derive(Debug, Clone)]
pub(crate) struct Layer {
    pub(crate) mixer: Mixer,
    /// The layer's index among layers with the same kind of mixer.
    pub(crate) cache_index: u32,
    pub(crate) attn_norm: Vector,
    pub(crate) ffn_norm: Vector,
    pub(crate) gate: Matrix,
    pub(crate) up: Matrix,
    pub(crate) down: Matrix,
}

impl Layer {
    /// Return the layer's matrices.
    pub(crate) fn matrices(&self) -> Vec<&Matrix> {
        let mut matrices = match &self.mixer {
            Mixer::Attention(attention) => {
                vec![&attention.q, &attention.k, &attention.v, &attention.output]
            }
            Mixer::DeltaNet(delta) => vec![
                &delta.qkv,
                &delta.gate,
                &delta.beta,
                &delta.alpha,
                &delta.output,
            ],
        };
        matrices.extend([&self.gate, &self.up, &self.down]);
        matrices
    }
}

/// A Qwen3.5 model loaded from a GGUF file.
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
    /// Load the Qwen3.5 model in the GGUF file at `path`.
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
        Self::from_gguf(gguf)
    }

    /// Build the model from a parsed GGUF file.
    pub(crate) fn from_gguf(gguf: Gguf<Storage>) -> Result<Self, Error> {
        let mut hp = load_hyperparameters(&gguf)?;
        let n_embd = u64::from(hp.n_embd);
        let embd_rows = match gguf.tensor("token_embd.weight") {
            Some(tensor) if tensor.shape().len() == 2 => tensor.ne(1),
            Some(_) | None => return Err(Error::MissingTensor("token_embd.weight".to_owned())),
        };
        hp.n_vocab = u32::try_from(embd_rows).map_err(|_| Error::Hyperparameters)?;

        // The layer count comes from the file, so the list grows as layers load.
        let mut layers = Vec::new();
        for il in 0..hp.n_layers {
            let layer = load_layer(&gguf, &hp, il)?;
            match layer.mixer {
                Mixer::Attention(_) => hp.n_attention_layers += 1,
                Mixer::DeltaNet(_) => hp.n_delta_layers += 1,
            }
            layers.push(layer);
        }

        let token_embd = require_matrix(&gguf, "token_embd.weight", n_embd, embd_rows)?;
        let output_norm = require_vector(&gguf, "output_norm.weight", &[n_embd])?;
        // Small Qwen3.5 models tie the output matrix to the token embeddings.
        let output = if gguf.tensor("output.weight").is_some() {
            require_matrix(&gguf, "output.weight", n_embd, embd_rows)?
        } else {
            token_embd.clone()
        };
        Ok(Self {
            gguf,
            hyperparameters: hp,
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

    /// Return the sampling settings for this model: the values the file stores under
    /// `general.sampling`, then the values Qwen's Qwen3.5 model cards recommend for thinking on
    /// text. bobcat has no presence penalty, so the repeat penalty stays off as the cards set it.
    pub fn recommended_sampling(&self) -> Sampling {
        let gguf = &self.gguf;
        Sampling {
            temperature: gguf.f32("general.sampling.temp").unwrap_or(1.0),
            top_k: gguf.u32("general.sampling.top_k").unwrap_or(20),
            top_p: gguf.f32("general.sampling.top_p").unwrap_or(0.95),
            min_p: gguf.f32("general.sampling.min_p").unwrap_or(0.0),
            repeat_penalty: gguf.f32("general.sampling.penalty_repeat").unwrap_or(1.0),
        }
    }

    /// Return a zeroed trace of `n_tokens` tokens of this model.
    pub fn trace(&self, n_tokens: u32) -> Trace {
        Trace::with_shape(
            self.hyperparameters.n_embd,
            self.hyperparameters.n_layers,
            n_tokens,
        )
    }

    /// Return the data bytes of the tensor of `matrix`.
    fn data(&self, matrix: &Matrix) -> &[u8] {
        &self.gguf.bytes()[matrix.tensor.data_range()]
    }

    fn matvec(&self, matrix: &Matrix, x: &[f32], y: &mut [f32]) {
        scalar::matvec(matrix.tensor.data_type(), self.data(matrix), x, y);
    }

    /// Run the model on `token` at the next position of `state`.
    ///
    /// When the caller passes `logits`, it receives the `n_vocab` logits. When the caller passes `trace`, it
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
        if token >= hp.n_vocab {
            return Err(Error::Token {
                token,
                n_vocab: hp.n_vocab,
            });
        }
        if let Some(logits) = &logits
            && logits.len() != to_usize(hp.n_vocab)
        {
            return Err(Error::Argument("logits must hold n_vocab floats"));
        }
        if let Some(trace) = &trace
            && !trace.holds_shape(hp.n_embd, hp.n_layers, 1)
        {
            return Err(Error::Argument("a step's trace must hold one token"));
        }
        let mut trace = trace;

        let s = &mut state.scratch;
        scalar::get_row(
            self.token_embd.tensor.data_type(),
            self.data(&self.token_embd),
            to_usize(token),
            &mut s.h,
        );
        if let Some(trace) = trace.as_deref_mut() {
            trace.embedding.copy_from_slice(&s.h);
        }

        for (il, layer) in self.layers.iter().enumerate() {
            let s = &mut state.scratch;
            scalar::rms_norm(&s.h, &layer.attn_norm.values, hp.norm_eps, &mut s.x);
            match &layer.mixer {
                Mixer::Attention(attention) => {
                    self.attention(attention, layer.cache_index, pos, state);
                }
                Mixer::DeltaNet(delta) => self.delta_net(delta, layer.cache_index, state),
            }
            let s = &mut state.scratch;
            add_in_place(&mut s.h, &s.mixer_out);

            scalar::rms_norm(&s.h, &layer.ffn_norm.values, hp.norm_eps, &mut s.x);
            self.matvec(&layer.gate, &s.x, &mut s.ffn_gate);
            self.matvec(&layer.up, &s.x, &mut s.ffn_up);
            for (gate, &up) in s.ffn_gate.iter_mut().zip(&s.ffn_up) {
                *gate = scalar::silu(*gate) * up;
            }
            self.matvec(&layer.down, &s.ffn_gate, &mut s.mixer_out);
            add_in_place(&mut s.h, &s.mixer_out);

            if let Some(trace) = trace.as_deref_mut() {
                trace.layers[il * n_embd..(il + 1) * n_embd].copy_from_slice(&s.h);
            }
        }

        let s = &mut state.scratch;
        scalar::rms_norm(&s.h, &self.output_norm.values, hp.norm_eps, &mut s.x);
        if let Some(trace) = trace {
            trace.final_norm.copy_from_slice(&s.x);
        }
        if let Some(logits) = logits {
            self.matvec(&self.output, &s.x, logits);
        }
        state.n_past += 1;
        Ok(())
    }

    /// Run the attention layer `attention` on the normed token in the scratch at position `pos`,
    /// writing the layer's output to the scratch.
    fn attention(&self, attention: &Attention, cache_index: u32, pos: u32, state: &mut State) {
        let hp = &self.hyperparameters;
        let head_dim = to_usize(hp.head_dim);
        let n_rot = to_usize(hp.n_rot);
        let n_heads = to_usize(hp.n_heads);
        let n_kv_heads = to_usize(hp.n_kv_heads);
        let kv_dim = n_kv_heads * head_dim;
        let group = n_heads / n_kv_heads;
        let s = &mut state.scratch;

        self.matvec(&attention.q, &s.x, &mut s.q_gate);
        self.matvec(&attention.k, &s.x, &mut s.k);
        self.matvec(&attention.v, &s.x, &mut s.v);
        // Each query head's block holds its queries, then the gate of its output.
        for head in 0..n_heads {
            let block = &s.q_gate[2 * head * head_dim..2 * (head + 1) * head_dim];
            let q = &mut s.q[head * head_dim..(head + 1) * head_dim];
            q.copy_from_slice(&block[..head_dim]);
            scalar::rms_norm_in_place(q, &attention.q_norm.values, hp.norm_eps);
            scalar::rope_neox(&mut q[..n_rot], pos, hp.rope_theta);
            s.attn_gate[head * head_dim..(head + 1) * head_dim].copy_from_slice(&block[head_dim..]);
        }
        for head in 0..n_kv_heads {
            let k = &mut s.k[head * head_dim..(head + 1) * head_dim];
            scalar::rms_norm_in_place(k, &attention.k_norm.values, hp.norm_eps);
            scalar::rope_neox(&mut k[..n_rot], pos, hp.rope_theta);
        }

        let cache = &mut state.caches[to_usize(cache_index)];
        let slot = to_usize(pos) * kv_dim;
        cache.k[slot..slot + kv_dim].copy_from_slice(&s.k);
        cache.v[slot..slot + kv_dim].copy_from_slice(&s.v);

        let n_keys = to_usize(pos) + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores = &mut s.scores[..n_keys];
        for head in 0..n_heads {
            let kv_head = head / group;
            let q = &s.q[head * head_dim..(head + 1) * head_dim];
            for (t, score) in scores.iter_mut().enumerate() {
                let k = &cache.k[t * kv_dim + kv_head * head_dim..][..head_dim];
                *score = scalar::dot(q, k) * scale;
            }
            scalar::softmax(scores);
            let out = &mut s.attn[head * head_dim..(head + 1) * head_dim];
            out.fill(0.0);
            for (t, &weight) in scores.iter().enumerate() {
                let v = &cache.v[t * kv_dim + kv_head * head_dim..][..head_dim];
                for (o, &v) in out.iter_mut().zip(v) {
                    *o += weight * v;
                }
            }
        }
        for (o, &gate) in s.attn.iter_mut().zip(&s.attn_gate) {
            *o *= sigmoid(gate);
        }
        self.matvec(&attention.output, &s.attn, &mut s.mixer_out);
    }

    /// Run the Gated DeltaNet layer `delta` on the normed token in the scratch, updating the
    /// layer's convolution and recurrent state and writing the layer's output to the scratch.
    fn delta_net(&self, delta: &DeltaNet, cache_index: u32, state: &mut State) {
        let hp = &self.hyperparameters;
        let kernel = to_usize(hp.conv_kernel);
        let history = kernel - 1;
        let n_k_heads = to_usize(hp.n_k_heads);
        let n_v_heads = to_usize(hp.n_v_heads);
        let k_head_dim = to_usize(hp.k_head_dim);
        let v_head_dim = to_usize(hp.v_head_dim);
        let key_dim = to_usize(hp.key_dim());
        let s = &mut state.scratch;
        let recurrent = &mut state.recurrent[to_usize(cache_index)];

        self.matvec(&delta.qkv, &s.x, &mut s.qkv);
        self.matvec(&delta.gate, &s.x, &mut s.z);
        self.matvec(&delta.beta, &s.x, &mut s.beta);
        self.matvec(&delta.alpha, &s.x, &mut s.alpha);

        // The causal convolution runs over the queries, keys, and values with SiLU, and its
        // history holds the latest raw inputs, the oldest first.
        for (c, (&input, out)) in s.qkv.iter().zip(&mut s.conv_out).enumerate() {
            let taps = &delta.conv.values[c * kernel..(c + 1) * kernel];
            let past = &mut recurrent.conv[c * history..(c + 1) * history];
            let mut sum = taps[history] * input;
            for (&tap, &value) in taps.iter().zip(past.iter()) {
                sum += tap * value;
            }
            *out = scalar::silu(sum);
            past.rotate_left(1);
            if let Some(newest) = past.last_mut() {
                *newest = input;
            }
        }

        let (q, rest) = s.conv_out.split_at_mut(key_dim);
        let (k, v) = rest.split_at_mut(key_dim);
        let q_scale = 1.0 / (k_head_dim as f32).sqrt();
        for head in 0..n_k_heads {
            let span = head * k_head_dim..(head + 1) * k_head_dim;
            l2_normalize(&mut q[span.clone()], q_scale);
            l2_normalize(&mut k[span], 1.0);
        }

        for head in 0..n_v_heads {
            // llama.cpp's converter orders value heads so that value head `j` reads key head
            // `j % n_k_heads`.
            let key_head = head % n_k_heads;
            let q = &q[key_head * k_head_dim..(key_head + 1) * k_head_dim];
            let k = &k[key_head * k_head_dim..(key_head + 1) * k_head_dim];
            let v = &v[head * v_head_dim..(head + 1) * v_head_dim];
            let beta = sigmoid(s.beta[head]);
            let decay =
                (delta.a.values[head] * softplus(s.alpha[head] + delta.dt_bias.values[head])).exp();

            // The state holds `k_head_dim` rows of `v_head_dim` elements.
            let state =
                &mut recurrent.state[head * k_head_dim * v_head_dim..][..k_head_dim * v_head_dim];
            for value in state.iter_mut() {
                *value *= decay;
            }
            let remembered = &mut s.delta[..v_head_dim];
            remembered.fill(0.0);
            for (row, &key) in state.chunks_exact(v_head_dim).zip(k) {
                for (r, &value) in remembered.iter_mut().zip(row) {
                    *r += value * key;
                }
            }
            for (r, &value) in remembered.iter_mut().zip(v) {
                *r = (value - *r) * beta;
            }
            let out = &mut s.delta_out[head * v_head_dim..(head + 1) * v_head_dim];
            out.fill(0.0);
            for ((row, &key), &query) in state.chunks_exact_mut(v_head_dim).zip(k).zip(q) {
                for ((value, &d), o) in row.iter_mut().zip(remembered.iter()).zip(out.iter_mut()) {
                    *value += key * d;
                    *o += *value * query;
                }
            }

            // The output norm scales each value head, and SiLU of the gate multiplies it.
            scalar::rms_norm_in_place(out, &delta.norm.values, hp.norm_eps);
            let gate = &s.z[head * v_head_dim..(head + 1) * v_head_dim];
            for (o, &g) in out.iter_mut().zip(gate) {
                *o *= scalar::silu(g);
            }
        }
        self.matvec(&delta.output, &s.delta_out, &mut s.mixer_out);
    }
}

/// The attention cache of one attention layer.
#[derive(Debug, Clone)]
struct Cache {
    /// `n_ctx` positions of `n_kv_heads * head_dim` keys after their norm and rotation.
    k: Vec<f32>,
    v: Vec<f32>,
}

/// The recurrent state of one Gated DeltaNet layer.
#[derive(Debug, Clone)]
struct Recurrent {
    /// The latest `conv_kernel - 1` raw inputs of each convolution channel, the oldest first.
    conv: Vec<f32>,
    /// `n_v_heads` state matrices of `k_head_dim` rows of `v_head_dim` elements.
    state: Vec<f32>,
}

/// The working buffers of one step.
#[derive(Debug, Clone)]
struct Scratch {
    h: Vec<f32>,
    x: Vec<f32>,
    mixer_out: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    q_gate: Vec<f32>,
    q: Vec<f32>,
    attn_gate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    qkv: Vec<f32>,
    conv_out: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    delta: Vec<f32>,
    delta_out: Vec<f32>,
}

/// The sequence state of a Qwen3.5 model on the CPU: the attention caches, the DeltaNet states,
/// and the scratch buffers.
#[derive(Debug, Clone)]
pub struct State {
    n_ctx: u32,
    n_past: u32,
    caches: Vec<Cache>,
    recurrent: Vec<Recurrent>,
    scratch: Scratch,
}

impl State {
    /// Return the empty state of a sequence of up to `n_ctx` tokens of `model`.
    pub fn new(model: &Model, n_ctx: u32) -> Result<Self, Error> {
        if n_ctx == 0 {
            return Err(Error::Argument("the context must hold a token"));
        }
        let hp = &model.hyperparameters;
        let n_embd = to_usize(hp.n_embd);
        let head_dim = to_usize(hp.head_dim);
        let kv_dim = to_usize(hp.n_kv_heads) * head_dim;
        let q_dim = to_usize(hp.n_heads) * head_dim;
        let conv_dim = to_usize(hp.conv_dim());
        let value_dim = to_usize(hp.value_dim());
        let cache = Cache {
            k: vec![0.0; to_usize(n_ctx) * kv_dim],
            v: vec![0.0; to_usize(n_ctx) * kv_dim],
        };
        let recurrent = Recurrent {
            conv: vec![0.0; conv_dim * to_usize(hp.conv_kernel - 1)],
            state: vec![0.0; to_usize(hp.n_v_heads) * to_usize(hp.k_head_dim * hp.v_head_dim)],
        };
        Ok(Self {
            n_ctx,
            n_past: 0,
            caches: vec![cache; to_usize(hp.n_attention_layers)],
            recurrent: vec![recurrent; to_usize(hp.n_delta_layers)],
            scratch: Scratch {
                h: vec![0.0; n_embd],
                x: vec![0.0; n_embd],
                mixer_out: vec![0.0; n_embd],
                ffn_gate: vec![0.0; to_usize(hp.n_ff)],
                ffn_up: vec![0.0; to_usize(hp.n_ff)],
                q_gate: vec![0.0; 2 * q_dim],
                q: vec![0.0; q_dim],
                attn_gate: vec![0.0; q_dim],
                k: vec![0.0; kv_dim],
                v: vec![0.0; kv_dim],
                attn: vec![0.0; q_dim],
                scores: vec![0.0; to_usize(n_ctx)],
                qkv: vec![0.0; conv_dim],
                conv_out: vec![0.0; conv_dim],
                z: vec![0.0; value_dim],
                beta: vec![0.0; to_usize(hp.n_v_heads)],
                alpha: vec![0.0; to_usize(hp.n_v_heads)],
                delta: vec![0.0; to_usize(hp.v_head_dim)],
                delta_out: vec![0.0; value_dim],
            },
        })
    }

    /// Return the number of tokens the state holds.
    pub fn n_past(&self) -> u32 {
        self.n_past
    }
}

/// Read the hyperparameters from the metadata of `gguf`, which must hold a `qwen35` model. The
/// layer counts by kind start at zero, and the loader counts them from the tensors.
fn load_hyperparameters(gguf: &Gguf<Storage>) -> Result<Hyperparameters, Error> {
    let architecture = gguf
        .string("general.architecture")
        .ok_or_else(|| Error::Metadata("general.architecture".to_owned()))?;
    if architecture != b"qwen35" {
        return Err(Error::Architecture(
            String::from_utf8_lossy(architecture).into_owned(),
        ));
    }
    let key = |suffix: &str| format!("qwen35.{suffix}");
    let u32_of = |suffix: &str| require_u32(gguf, &key(suffix));
    let block_count = u32_of("block_count")?;
    // The multi-token prediction block follows the model's layers and drafts tokens.
    let n_nextn = gguf.u32(&key("nextn_predict_layers")).unwrap_or(0);
    let n_layers = block_count
        .checked_sub(n_nextn)
        .ok_or(Error::Hyperparameters)?;
    let head_dim = u32_of("attention.key_length")?;
    let n_v_heads = u32_of("ssm.time_step_rank")?;
    let inner_size = u32_of("ssm.inner_size")?;
    let hp = Hyperparameters {
        n_layers,
        n_attention_layers: 0,
        n_delta_layers: 0,
        n_embd: u32_of("embedding_length")?,
        n_ff: u32_of("feed_forward_length")?,
        n_vocab: 0,
        n_heads: u32_of("attention.head_count")?,
        n_kv_heads: u32_of("attention.head_count_kv")?,
        head_dim,
        n_rot: u32_of("rope.dimension_count")?,
        rope_theta: require_f32(gguf, &key("rope.freq_base"))?,
        norm_eps: require_f32(gguf, &key("attention.layer_norm_rms_epsilon"))?,
        conv_kernel: u32_of("ssm.conv_kernel")?,
        n_k_heads: u32_of("ssm.group_count")?,
        n_v_heads,
        k_head_dim: u32_of("ssm.state_size")?,
        v_head_dim: inner_size.checked_div(n_v_heads).unwrap_or(0),
    };
    let value_length = u32_of("attention.value_length")?;
    if hp.n_layers == 0
        || hp.n_embd == 0
        || hp.n_heads == 0
        || hp.n_kv_heads == 0
        || !hp.n_heads.is_multiple_of(hp.n_kv_heads)
        || hp.head_dim == 0
        || value_length != hp.head_dim
        || hp.n_rot == 0
        || !hp.n_rot.is_multiple_of(2)
        || hp.n_rot > hp.head_dim
        || hp.conv_kernel < 2
        || hp.n_k_heads == 0
        || hp.n_v_heads == 0
        || !hp.n_v_heads.is_multiple_of(hp.n_k_heads)
        || hp.k_head_dim == 0
        || hp.v_head_dim == 0
        || hp.n_v_heads.checked_mul(hp.v_head_dim) != Some(inner_size)
    {
        return Err(Error::Hyperparameters);
    }
    // `key_dim`, `conv_dim`, and the DeltaNet state of each value head multiply counts from the
    // file, so their products must fit before anything sizes a buffer with them.
    let key_dim = hp.n_k_heads.checked_mul(hp.k_head_dim);
    let conv_dim = key_dim
        .and_then(|key_dim| key_dim.checked_mul(2))
        .and_then(|keys| keys.checked_add(inner_size));
    let state = hp.k_head_dim.checked_mul(hp.v_head_dim);
    if conv_dim.is_none() || state.is_none() {
        return Err(Error::Hyperparameters);
    }
    Ok(hp)
}

/// Find and check the tensors of layer `il` in `gguf`. A layer with DeltaNet tensors is a DeltaNet
/// layer, and any other layer is an attention layer.
fn load_layer(gguf: &Gguf<Storage>, hp: &Hyperparameters, il: u32) -> Result<Layer, Error> {
    let name = |suffix: &str| format!("blk.{il}.{suffix}");
    let n_embd = u64::from(hp.n_embd);
    let n_ff = u64::from(hp.n_ff);
    let matrix = |suffix: &str, n_cols: u64, n_rows: u64| {
        require_matrix(gguf, &name(suffix), n_cols, n_rows)
    };
    let vector = |suffix: &str, want: &[u64]| require_vector(gguf, &name(suffix), want);

    let (mixer, cache_index) = if gguf.tensor(&name("ssm_a")).is_some() {
        let value_dim = u64::from(hp.value_dim());
        let conv_dim = u64::from(hp.conv_dim());
        let n_v_heads = u64::from(hp.n_v_heads);
        let delta = DeltaNet {
            qkv: matrix("attn_qkv.weight", n_embd, conv_dim)?,
            gate: matrix("attn_gate.weight", n_embd, value_dim)?,
            beta: matrix("ssm_beta.weight", n_embd, n_v_heads)?,
            alpha: matrix("ssm_alpha.weight", n_embd, n_v_heads)?,
            conv: vector("ssm_conv1d.weight", &[u64::from(hp.conv_kernel), conv_dim])?,
            dt_bias: vector("ssm_dt.bias", &[n_v_heads])?,
            a: vector("ssm_a", &[n_v_heads])?,
            norm: vector("ssm_norm.weight", &[u64::from(hp.v_head_dim)])?,
            output: matrix("ssm_out.weight", value_dim, n_embd)?,
        };
        (Mixer::DeltaNet(delta), hp.n_delta_layers)
    } else {
        let head_dim = u64::from(hp.head_dim);
        let q_dim = u64::from(hp.n_heads) * head_dim;
        let kv_dim = u64::from(hp.n_kv_heads) * head_dim;
        let attention = Attention {
            q: matrix("attn_q.weight", n_embd, 2 * q_dim)?,
            k: matrix("attn_k.weight", n_embd, kv_dim)?,
            v: matrix("attn_v.weight", n_embd, kv_dim)?,
            output: matrix("attn_output.weight", q_dim, n_embd)?,
            q_norm: vector("attn_q_norm.weight", &[head_dim])?,
            k_norm: vector("attn_k_norm.weight", &[head_dim])?,
        };
        (Mixer::Attention(attention), hp.n_attention_layers)
    };
    Ok(Layer {
        mixer,
        cache_index,
        attn_norm: vector("attn_norm.weight", &[n_embd])?,
        ffn_norm: vector("post_attention_norm.weight", &[n_embd])?,
        gate: matrix("ffn_gate.weight", n_embd, n_ff)?,
        up: matrix("ffn_up.weight", n_embd, n_ff)?,
        down: matrix("ffn_down.weight", n_ff, n_embd)?,
    })
}

/// Add `delta` to `h` element by element.
fn add_in_place(h: &mut [f32], delta: &[f32]) {
    for (h, &d) in h.iter_mut().zip(delta) {
        *h += d;
    }
}

/// Divide `x` by its L2 norm and multiply it by `scale`.
fn l2_normalize(x: &mut [f32], scale: f32) {
    let sum: f32 = x.iter().map(|value| value * value).sum();
    let factor = scale / (sum + L2_NORM_EPS).sqrt();
    for value in x {
        *value *= factor;
    }
}

/// Return the logistic sigmoid of `x`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Return `ln(1 + exp(x))`, or `x` above 20 as transformers' softplus does.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
