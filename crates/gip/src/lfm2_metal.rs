//! The LFM2 forward pass on the Metal GPU.
//!
//! Each call records its work into command buffers and waits for them before it returns. The
//! weights stay in the memory-mapped GGUF file, and the GPU reads them in place.

use std::time::Instant;

use gip_gguf::{Tensor, TensorType};
use gip_metal::{
    Buffer, MatvecOptions, Metal, Norm, Store, Ticket, View, attention_scratch_floats,
};

use crate::error::Error;
use crate::lfm2::{Attention, Conv, Layer, Matrix, Mixer, Model, Trace, Vector, to_usize};
use crate::storage::Storage;

/// Each lane of the attention kernel holds a whole number of elements of a head, and a
/// threadgroup serves at most four query heads of up to 128 elements. Must match
/// `ATTENTION_MAX_GROUP` and `ATTENTION_MAX_HEAD_DIM` in `kernels.metal`.
const SIMD_WIDTH: u32 = 32;
const ATTENTION_MAX_GROUP: u32 = 4;
const ATTENTION_MAX_HEAD_DIM: u32 = 128;

/// `short_conv_history` in `kernels.metal` keeps up to 8 history inputs per channel in
/// registers.
const CONV_MAX_KERNEL: u32 = 9;

/// Generation keeps up to this many steps submitted ahead of the GPU. Each step depends on the one
/// before, so more steps in flight save nothing once the GPU never waits for the CPU.
const GENERATE_IN_FLIGHT: usize = 3;

/// Prefill runs the prompt through the model this many tokens at a time.
const PREFILL_BATCH: u32 = 512;

const FLOAT_BYTES: usize = 4;
const TOKEN_BYTES: usize = 4;

/// The GPU buffers of one LFM2 decode.
#[derive(Debug)]
struct Buffers {
    weights: Buffer,
    k_cache: Buffer,
    v_cache: Buffer,
    k: Buffer,
    v: Buffer,
    conv_state: Buffer,
    hidden: Buffer,
    normed: Buffer,
    bcx: Buffer,
    conv_out: Buffer,
    q: Buffer,
    attn: Buffer,
    scores: Buffer,
    ffn: Buffer,
    logits: Buffer,
    tokens: Buffer,
    trace_embedding: Buffer,
    trace_layers: Buffer,
    trace_final: Buffer,
}

impl Buffers {
    /// Return a view of the data of `tensor` inside the weight buffer.
    fn weights(&self, tensor: &Tensor) -> View<'_> {
        self.weights.at(tensor.data_range().start)
    }

    /// Return a view of the token id at position `pos`.
    fn token(&self, pos: u32) -> View<'_> {
        self.tokens.at(to_usize(pos) * TOKEN_BYTES)
    }
}

/// The GPU state of one LFM2 decode: the weights, the caches, and the scratch buffers.
///
/// The scratch buffers hold one row per token of a prefill batch, and decode uses the first row.
#[derive(Debug)]
pub struct Lfm2Metal<'a> {
    model: &'a Model,
    metal: &'a mut Metal,
    n_ctx: u32,
    n_past: u32,
    batch: u32,
    trace_rows: u32,
    kv_half: bool,
    last_encode_seconds: f64,
    last_gpu_seconds: f64,
    buffers: Buffers,
}

impl<'a> Lfm2Metal<'a> {
    /// Prepare to decode sequences of up to `n_ctx` tokens of `model` on `metal`.
    ///
    /// Every matrix of `model` must be Q8_0. The KV cache holds half precision when `kv_half` is
    /// true and floats otherwise.
    pub fn new(
        model: &'a Model,
        metal: &'a mut Metal,
        n_ctx: u32,
        kv_half: bool,
    ) -> Result<Self, Error> {
        let hp = model.hyperparameters();
        if n_ctx == 0 {
            return Err(Error::Argument("the context must hold a token"));
        }
        check_supported(model)?;

        let weights = match model.gguf.storage() {
            // A memory-mapped file becomes a GPU buffer with no copy.
            Storage::Mapped(mapping) => metal.wrap_mapping(std::sync::Arc::clone(mapping))?,
            // Heap memory lacks page alignment, so the file is copied into a new buffer.
            Storage::Heap(bytes) => {
                let buffer = metal.new_buffer(bytes.len())?;
                metal.write(buffer.at(0), bytes)?;
                buffer
            }
        };

        let n_embd = to_usize(hp.n_embd);
        let kv_dim = to_usize(hp.n_kv_heads) * to_usize(hp.head_dim);
        let q_dim = to_usize(hp.n_heads) * to_usize(hp.head_dim);
        let cache_elements = to_usize(hp.n_attention_layers) * to_usize(n_ctx) * kv_dim;
        let cache_bytes = cache_elements * if kv_half { 2 } else { FLOAT_BYTES };
        let conv_floats = to_usize(hp.n_conv_layers) * to_usize(hp.conv_kernel - 1) * n_embd;
        let batch = n_ctx.min(PREFILL_BATCH);
        let rows = to_usize(batch);
        let floats = |count: usize| metal.new_buffer(count * FLOAT_BYTES);

        let buffers = Buffers {
            weights,
            k_cache: metal.new_buffer(cache_bytes)?,
            v_cache: metal.new_buffer(cache_bytes)?,
            k: floats(rows * kv_dim)?,
            v: floats(rows * kv_dim)?,
            conv_state: floats(conv_floats)?,
            hidden: floats(rows * n_embd)?,
            normed: floats(rows * n_embd)?,
            bcx: floats(rows * 3 * n_embd)?,
            conv_out: floats(rows * n_embd)?,
            q: floats(rows * q_dim)?,
            attn: floats(rows * q_dim)?,
            scores: floats(attention_scratch_floats(
                hp.n_heads,
                hp.head_dim,
                n_ctx,
                batch,
            ))?,
            ffn: floats(rows * to_usize(hp.n_ff))?,
            logits: floats(to_usize(hp.n_vocab))?,
            tokens: metal.new_buffer(to_usize(n_ctx) * TOKEN_BYTES)?,
            trace_embedding: floats(n_embd)?,
            trace_layers: floats(to_usize(hp.n_layers) * n_embd)?,
            trace_final: floats(n_embd)?,
        };

        Ok(Self {
            model,
            metal,
            n_ctx,
            n_past: 0,
            batch,
            trace_rows: 1,
            kv_half,
            last_encode_seconds: 0.0,
            last_gpu_seconds: 0.0,
            buffers,
        })
    }

    /// Return the CPU time the latest call spent recording GPU work, in seconds.
    pub fn last_encode_seconds(&self) -> f64 {
        self.last_encode_seconds
    }

    /// Return the GPU time the latest call's work took, in seconds.
    pub fn last_gpu_seconds(&self) -> f64 {
        self.last_gpu_seconds
    }

    /// Return the Metal backend, for profiling.
    pub fn metal(&mut self) -> &mut Metal {
        self.metal
    }

    /// Return a recorder of work on the buffers.
    fn recorder(&mut self) -> Recorder<'_> {
        Recorder {
            metal: self.metal,
            model: self.model,
            buffers: &self.buffers,
            n_ctx: self.n_ctx,
            kv_half: self.kv_half,
        }
    }

    /// Run the model on `token` at the next position, with the same outputs as [`Model::step`].
    pub fn step(
        &mut self,
        token: u32,
        logits: Option<&mut [f32]>,
        trace: Option<&mut Trace>,
    ) -> Result<(), Error> {
        let hp = self.model.hyperparameters();
        let pos = self.n_past;
        if pos >= self.n_ctx {
            return Err(Error::ContextFull {
                requested: 1,
                remaining: 0,
            });
        }
        self.model.check_token(token)?;
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

        // Nothing is in flight between steps, so the CPU can write the token straight into the
        // shared buffer.
        self.metal.write(self.buffers.token(pos), &[token])?;

        let encode_start = Instant::now();
        self.metal.begin()?;
        let recorded = self
            .recorder()
            .record_step(pos, logits.is_some(), trace.is_some());
        if let Err(error) = recorded {
            self.metal.discard();
            return Err(error);
        }
        self.last_encode_seconds = encode_start.elapsed().as_secs_f64();
        self.last_gpu_seconds = self.metal.end()?;

        let buffers = &self.buffers;
        if let Some(logits) = logits {
            self.metal.read(buffers.logits.at(0), logits)?;
        }
        if let Some(trace) = trace {
            self.metal
                .read(buffers.trace_embedding.at(0), &mut trace.embedding)?;
            self.metal
                .read(buffers.trace_layers.at(0), &mut trace.layers)?;
            self.metal
                .read(buffers.trace_final.at(0), &mut trace.final_norm)?;
        }

        self.n_past += 1;
        Ok(())
    }

    /// Decode `out.len()` tokens greedily into `out`.
    ///
    /// The first token is the argmax of the logits of the previous call, which must have
    /// produced logits. Each token then runs through the model to choose the next. The GPU picks
    /// every token itself, and the CPU submits several steps ahead, so the GPU never waits for
    /// the CPU.
    pub fn generate(&mut self, out: &mut [u32]) -> Result<(), Error> {
        let n = u32::try_from(out.len()).map_err(|_| Error::Argument("too many tokens"))?;
        let remaining = self.n_ctx - self.n_past;
        if self.n_past == 0 {
            return Err(Error::Argument("generation needs a prompt first"));
        }
        if n > remaining {
            return Err(Error::ContextFull {
                requested: n,
                remaining,
            });
        }

        let n_vocab = self.model.hyperparameters().n_vocab;
        let mut tickets: [Option<Ticket>; GENERATE_IN_FLIGHT] = [None; GENERATE_IN_FLIGHT];
        let mut gpu_seconds = 0.0;
        let mut encode_seconds = 0.0;
        let mut result = Ok(());

        for i in 0..n {
            let pos = self.n_past + i;
            // Bound the steps in flight, so the CPU runs only a few steps ahead of the GPU.
            let slot = to_usize(i) % GENERATE_IN_FLIGHT;
            if let Some(ticket) = tickets[slot] {
                match self.metal.wait(ticket) {
                    Ok(seconds) => gpu_seconds += seconds,
                    Err(error) => {
                        result = Err(error.into());
                        break;
                    }
                }
            }

            let encode_start = Instant::now();
            if let Err(error) = self.metal.begin() {
                result = Err(error.into());
                break;
            }
            let recorded = self.recorder().record_generated_step(pos, n_vocab);
            if let Err(error) = recorded {
                self.metal.discard();
                result = Err(error);
                break;
            }
            match self.metal.commit() {
                Ok(ticket) => tickets[slot] = Some(ticket),
                Err(error) => {
                    result = Err(error.into());
                    break;
                }
            }
            encode_seconds += encode_start.elapsed().as_secs_f64();
        }

        // Waiting on the newest command buffer waits on all the others.
        if let Some(last) = tickets.iter().flatten().max() {
            match self.metal.wait(*last) {
                Ok(seconds) => gpu_seconds += seconds,
                Err(error) => {
                    if result.is_ok() {
                        result = Err(error.into());
                    }
                }
            }
        }
        result?;

        self.metal.read(self.buffers.token(self.n_past), out)?;
        self.n_past += n;
        self.last_encode_seconds = encode_seconds;
        self.last_gpu_seconds = gpu_seconds;
        Ok(())
    }

    /// Run the model on `tokens` at the next positions, in batches of up to 512 tokens.
    ///
    /// When `logits` is given, it receives the logits of the last token. The logits buffer
    /// receives them either way, for [`Lfm2Metal::generate`]. When `trace` is given, it must
    /// hold `tokens.len()` tokens and receives every token's activations.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        logits: Option<&mut [f32]>,
        mut trace: Option<&mut Trace>,
    ) -> Result<(), Error> {
        let hp = self.model.hyperparameters();
        let n_embd = to_usize(hp.n_embd);
        let n_layers = to_usize(hp.n_layers);
        let n = u32::try_from(tokens.len()).map_err(|_| Error::Argument("too many tokens"))?;
        let remaining = self.n_ctx - self.n_past;
        if n == 0 {
            return Err(Error::Argument("the prompt must hold a token"));
        }
        if n > remaining {
            return Err(Error::ContextFull {
                requested: n,
                remaining,
            });
        }
        for &token in tokens {
            self.model.check_token(token)?;
        }
        if let Some(logits) = &logits
            && logits.len() != to_usize(hp.n_vocab)
        {
            return Err(Error::Argument("logits must hold n_vocab floats"));
        }
        if let Some(trace) = &trace
            && !trace.holds(hp, n)
        {
            return Err(Error::Argument("the trace must hold every prompt token"));
        }

        // Nothing is in flight, so every prompt token goes into the shared buffer before the
        // first batch.
        self.metal.write(self.buffers.token(self.n_past), tokens)?;
        if trace.is_some() {
            self.ensure_trace_rows(self.batch)?;
        }

        let mut gpu_seconds = 0.0;
        let mut encode_seconds = 0.0;
        let mut done = 0;
        while done < n {
            let count = (n - done).min(self.batch);
            let pos = self.n_past + done;
            let last = done + count == n;

            let encode_start = Instant::now();
            self.metal.begin()?;
            let layer_stride = to_usize(self.trace_rows) * n_embd;
            let recorded = self.recorder().record_batch(
                pos,
                count,
                last,
                trace.is_some().then_some(layer_stride),
            );
            if let Err(error) = recorded {
                self.metal.discard();
                return Err(error);
            }
            let ticket = self.metal.commit()?;
            encode_seconds += encode_start.elapsed().as_secs_f64();

            // A traced batch waits so its activations can be copied out before the next batch
            // overwrites them. Other batches run back to back.
            if trace.is_some() || last {
                gpu_seconds += self.metal.wait(ticket)?;
            }
            if let Some(trace) = trace.as_deref_mut() {
                let start = to_usize(done) * n_embd;
                let rows = to_usize(count) * n_embd;
                let buffers = &self.buffers;
                self.metal.read(
                    buffers.trace_embedding.at(0),
                    &mut trace.embedding[start..start + rows],
                )?;
                self.metal.read(
                    buffers.trace_final.at(0),
                    &mut trace.final_norm[start..start + rows],
                )?;
                for il in 0..n_layers {
                    let destination = (il * to_usize(n) + to_usize(done)) * n_embd;
                    self.metal.read(
                        buffers.trace_layers.floats(il * layer_stride),
                        &mut trace.layers[destination..destination + rows],
                    )?;
                }
            }
            done += count;
        }

        if let Some(logits) = logits {
            self.metal.read(self.buffers.logits.at(0), logits)?;
        }
        self.n_past += n;
        self.last_encode_seconds = encode_seconds;
        self.last_gpu_seconds = gpu_seconds;
        Ok(())
    }

    /// Grow the trace buffers to hold `rows` tokens.
    fn ensure_trace_rows(&mut self, rows: u32) -> Result<(), Error> {
        if rows <= self.trace_rows {
            return Ok(());
        }
        let hp = self.model.hyperparameters();
        let floats = to_usize(rows) * to_usize(hp.n_embd);
        self.buffers.trace_embedding = self.metal.new_buffer(floats * FLOAT_BYTES)?;
        self.buffers.trace_layers = self
            .metal
            .new_buffer(to_usize(hp.n_layers) * floats * FLOAT_BYTES)?;
        self.buffers.trace_final = self.metal.new_buffer(floats * FLOAT_BYTES)?;
        self.trace_rows = rows;
        Ok(())
    }
}

/// Check that the Metal kernels can run `model`.
fn check_supported(model: &Model) -> Result<(), Error> {
    let hp = model.hyperparameters();
    let matrices_q8_0 =
        model.layers.iter().all(layer_q8_0) && is_q8_0(&model.token_embd) && is_q8_0(&model.output);
    if !matrices_q8_0 {
        return Err(Error::MetalUnsupported("every matrix in Q8_0".to_owned()));
    }
    let group = hp.n_heads / hp.n_kv_heads.max(1);
    if !hp.head_dim.is_multiple_of(SIMD_WIDTH)
        || hp.head_dim > ATTENTION_MAX_HEAD_DIM
        || group > ATTENTION_MAX_GROUP
    {
        return Err(Error::MetalUnsupported(format!(
            "a head size that is a multiple of {SIMD_WIDTH} up to {ATTENTION_MAX_HEAD_DIM} and at \
             most {ATTENTION_MAX_GROUP} query heads per KV head, got {} and {group}",
            hp.head_dim
        )));
    }
    if hp.conv_kernel > CONV_MAX_KERNEL {
        return Err(Error::MetalUnsupported(format!(
            "a convolution of at most {CONV_MAX_KERNEL} taps, got {}",
            hp.conv_kernel
        )));
    }
    Ok(())
}

fn is_q8_0(matrix: &Matrix) -> bool {
    matrix.tensor.data_type() == TensorType::Q8_0
}

/// Report whether every matrix of `layer` is Q8_0.
fn layer_q8_0(layer: &Layer) -> bool {
    let mixer = match &layer.mixer {
        Mixer::Attention(attention) => {
            [&attention.q, &attention.k, &attention.v, &attention.output]
                .into_iter()
                .all(is_q8_0)
        }
        Mixer::Conv(conv) => is_q8_0(&conv.in_proj) && is_q8_0(&conv.out_proj),
    };
    mixer && is_q8_0(&layer.ffn_gate) && is_q8_0(&layer.ffn_up) && is_q8_0(&layer.ffn_down)
}

/// Records the launches of forward passes.
struct Recorder<'r> {
    metal: &'r mut Metal,
    model: &'r Model,
    buffers: &'r Buffers,
    n_ctx: u32,
    kv_half: bool,
}

impl<'r> Recorder<'r> {
    /// Return options that normalize the input with `norm` before the multiply.
    fn normalized(&self, norm: &Vector) -> MatvecOptions<'r> {
        MatvecOptions {
            norm: Some(Norm {
                weight: self.buffers.weights(&norm.tensor),
                eps: self.model.hyperparameters().norm_eps,
            }),
            accumulate: false,
        }
    }

    /// Record a multiply of `matrix` by `x` into `y` with `options`.
    fn matvec(
        &mut self,
        matrix: &Matrix,
        x: View<'_>,
        y: View<'_>,
        options: MatvecOptions<'_>,
    ) -> Result<(), Error> {
        let weights = self.buffers.weights(&matrix.tensor);
        self.metal
            .matvec_q8_0(weights, matrix.n_rows, matrix.n_cols, x, y, options)?;
        Ok(())
    }

    /// Record a multiply of `matrix` by each of the `n` tokens at `x`, combined with `y` as
    /// `store` says.
    fn matmul(
        &mut self,
        matrix: &Matrix,
        x: View<'_>,
        y: View<'_>,
        n: u32,
        store: Store,
    ) -> Result<(), Error> {
        let weights = self.buffers.weights(&matrix.tensor);
        self.metal
            .matmul_q8_0(weights, matrix.n_rows, matrix.n_cols, x, y, n, store)?;
        Ok(())
    }

    /// Return a view of a KV cache buffer at element `index`.
    fn cache_at(&self, buffer: &'r Buffer, index: usize) -> View<'r> {
        buffer.at(index * if self.kv_half { 2 } else { FLOAT_BYTES })
    }

    /// Record the gated short convolution `conv` on the residual stream and add its output back.
    fn conv(&mut self, conv: &Conv, cache_index: u32, norm: &Vector) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let history = to_usize(cache_index) * to_usize(hp.conv_kernel - 1) * to_usize(hp.n_embd);
        let h = b.hidden.floats(0);
        let options = self.normalized(norm);

        self.matvec(&conv.in_proj, h, b.bcx.floats(0), options)?;
        self.metal.barrier();
        self.metal.short_conv(
            b.bcx.floats(0),
            b.weights(&conv.taps.tensor),
            b.conv_state.floats(history),
            b.conv_out.floats(0),
            hp.n_embd,
            hp.conv_kernel,
            1,
        )?;
        self.metal.barrier();
        self.matvec(&conv.out_proj, b.conv_out.floats(0), h, ADD_TO_RESIDUAL)?;
        self.metal.barrier();
        Ok(())
    }

    /// Record `attention` at position `pos` on the residual stream and add its output back.
    ///
    /// The new key goes into the cache through the norm and rotation, and the new value goes into
    /// the cache directly or through a conversion to half precision.
    fn attention(
        &mut self,
        attention: &Attention,
        cache_index: u32,
        norm: &Vector,
        pos: u32,
    ) -> Result<(), Error> {
        let b = self.buffers;
        let h = b.hidden.floats(0);
        let v_target = self.value_target(cache_index, pos);
        let options = self.normalized(norm);

        // The three projections read the same input and write different buffers, so they run
        // together.
        self.matvec(&attention.q, h, b.q.floats(0), options)?;
        self.matvec(&attention.k, h, b.k.floats(0), options)?;
        self.matvec(&attention.v, h, v_target, options)?;
        self.metal.barrier();
        self.rotate_and_attend(attention, cache_index, pos, 1)?;
        self.matvec(&attention.output, b.attn.floats(0), h, ADD_TO_RESIDUAL)?;
        self.metal.barrier();
        Ok(())
    }

    /// Return the KV cache elements of layer `cache_index` before its first position.
    fn layer_base(&self, cache_index: u32) -> usize {
        let hp = self.model.hyperparameters();
        to_usize(cache_index) * to_usize(self.n_ctx) * to_usize(hp.n_kv_heads * hp.head_dim)
    }

    /// Return where the value projections of layer `cache_index` at positions `pos` onward go.
    /// A float cache receives them directly. A half-precision cache receives them through a
    /// conversion from the value scratch buffer.
    fn value_target(&self, cache_index: u32, pos: u32) -> View<'r> {
        let hp = self.model.hyperparameters();
        let kv_dim = to_usize(hp.n_kv_heads * hp.head_dim);
        if self.kv_half {
            self.buffers.v.floats(0)
        } else {
            let slot = self.layer_base(cache_index) + to_usize(pos) * kv_dim;
            self.cache_at(&self.buffers.v_cache, slot)
        }
    }

    /// Record the norm and rotation of the query and key projections of the `n` tokens at
    /// positions `pos` onward, the store of their keys and values in the cache of layer
    /// `cache_index`, and their causal attention into the attention buffer.
    fn rotate_and_attend(
        &mut self,
        attention: &Attention,
        cache_index: u32,
        pos: u32,
        n: u32,
    ) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let layer_base = self.layer_base(cache_index);
        let slot = layer_base + to_usize(pos) * to_usize(kv_dim);
        let q = b.q.floats(0);

        // The two rotations and the value conversion write different buffers, so they run
        // together.
        self.metal.norm_rope(
            q,
            q,
            false,
            b.weights(&attention.q_norm.tensor),
            hp.n_heads,
            hp.head_dim,
            pos,
            n,
            q_dim,
            q_dim,
            hp.rope_theta,
            hp.norm_eps,
        )?;
        self.metal.norm_rope(
            b.k.floats(0),
            self.cache_at(&b.k_cache, slot),
            self.kv_half,
            b.weights(&attention.k_norm.tensor),
            hp.n_kv_heads,
            hp.head_dim,
            pos,
            n,
            kv_dim,
            kv_dim,
            hp.rope_theta,
            hp.norm_eps,
        )?;
        if self.kv_half {
            self.metal
                .convert_half(b.v.floats(0), self.cache_at(&b.v_cache, slot), n * kv_dim)?;
        }
        self.metal.barrier();
        self.metal.attention(
            q,
            self.cache_at(&b.k_cache, layer_base),
            self.cache_at(&b.v_cache, layer_base),
            self.kv_half,
            b.scores.floats(0),
            b.attn.floats(0),
            hp.n_heads,
            hp.n_kv_heads,
            hp.head_dim,
            pos,
            n,
            self.n_ctx,
        )?;
        self.metal.barrier();
        Ok(())
    }

    /// Record one forward pass on the token at position `pos` of the token buffer: the embedding
    /// lookup, every layer, and, when `want_logits` is true, the output matrix into the logits
    /// buffer. When `traced` is true, also record copies of the activations into the trace
    /// buffers.
    fn record_step(&mut self, pos: u32, want_logits: bool, traced: bool) -> Result<(), Error> {
        let b = self.buffers;
        let model = self.model;
        let hp = model.hyperparameters();
        let h = b.hidden.floats(0);
        let ffn = b.ffn.floats(0);

        self.metal.embed_q8_0(
            b.weights(&model.token_embd.tensor),
            hp.n_vocab,
            b.token(pos),
            h,
            hp.n_embd,
            1,
        )?;
        self.metal.barrier();

        // Trace copies only read the residual stream, so they run alongside the next launches
        // that read it. Every later write to the stream comes after a barrier.
        if traced {
            self.metal.copy(h, b.trace_embedding.floats(0), hp.n_embd)?;
        }

        for (il, layer) in model.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Attention(attention) => {
                    self.attention(attention, layer.cache_index, &layer.attn_norm, pos)?;
                }
                Mixer::Conv(conv) => self.conv(conv, layer.cache_index, &layer.attn_norm)?,
            }

            self.metal.matvec_q8_0_swiglu(
                b.weights(&layer.ffn_gate.tensor),
                b.weights(&layer.ffn_up.tensor),
                layer.ffn_gate.n_rows,
                layer.ffn_gate.n_cols,
                h,
                Norm {
                    weight: b.weights(&layer.ffn_norm.tensor),
                    eps: hp.norm_eps,
                },
                ffn,
            )?;
            self.metal.barrier();
            self.matvec(&layer.ffn_down, ffn, h, ADD_TO_RESIDUAL)?;
            self.metal.barrier();

            if traced {
                self.metal.copy(
                    h,
                    b.trace_layers.floats(il * to_usize(hp.n_embd)),
                    hp.n_embd,
                )?;
            }
        }

        // The output matrix normalizes the last hidden state itself. Only a trace needs the
        // normalized vector on its own.
        if traced {
            self.metal.rms_norm(
                h,
                b.weights(&model.output_norm.tensor),
                b.trace_final.floats(0),
                hp.n_embd,
                1,
                hp.norm_eps,
            )?;
        }
        if want_logits {
            let options = self.normalized(&model.output_norm);
            self.matvec(&model.output, h, b.logits.floats(0), options)?;
        }
        Ok(())
    }

    /// Record the argmax of the logits buffer into position `pos` of the token buffer, then a
    /// forward pass on that token.
    fn record_generated_step(&mut self, pos: u32, n_vocab: u32) -> Result<(), Error> {
        let b = self.buffers;
        self.metal
            .argmax(b.logits.floats(0), b.token(pos), n_vocab)?;
        self.metal.barrier();
        self.record_step(pos, true, false)
    }

    /// Record the gated short convolution `conv` over the `n` normalized tokens of the batch and
    /// add its output to the residual stream.
    fn conv_batch(&mut self, conv: &Conv, cache_index: u32, n: u32) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let history = to_usize(cache_index) * to_usize(hp.conv_kernel - 1) * to_usize(hp.n_embd);

        self.matmul(
            &conv.in_proj,
            b.normed.floats(0),
            b.bcx.floats(0),
            n,
            Store::Overwrite,
        )?;
        self.metal.barrier();
        self.metal.short_conv_batch(
            b.bcx.floats(0),
            b.weights(&conv.taps.tensor),
            b.conv_state.floats(history),
            b.conv_out.floats(0),
            hp.n_embd,
            hp.conv_kernel,
            n,
        )?;
        self.metal.barrier();
        self.matmul(
            &conv.out_proj,
            b.conv_out.floats(0),
            b.hidden.floats(0),
            n,
            Store::Accumulate,
        )?;
        self.metal.barrier();
        Ok(())
    }

    /// Record the causal attention `attention` over the `n` normalized tokens of the batch, which
    /// sit at positions `pos` through `pos + n - 1`, and add its output to the residual stream.
    ///
    /// The batch's keys and values go into the caches first, so each token attends over the
    /// earlier tokens of its own batch too.
    fn attention_batch(
        &mut self,
        attention: &Attention,
        cache_index: u32,
        pos: u32,
        n: u32,
    ) -> Result<(), Error> {
        let b = self.buffers;
        let normed = b.normed.floats(0);
        let v_target = self.value_target(cache_index, pos);

        self.matmul(&attention.q, normed, b.q.floats(0), n, Store::Overwrite)?;
        self.matmul(&attention.k, normed, b.k.floats(0), n, Store::Overwrite)?;
        self.matmul(&attention.v, normed, v_target, n, Store::Overwrite)?;
        self.metal.barrier();
        self.rotate_and_attend(attention, cache_index, pos, n)?;
        self.matmul(
            &attention.output,
            b.attn.floats(0),
            b.hidden.floats(0),
            n,
            Store::Accumulate,
        )?;
        self.metal.barrier();
        Ok(())
    }

    /// Record the forward pass of the `n` tokens at positions `pos` onward of the token buffer.
    ///
    /// When `want_logits` is true, the last token's logits go to the logits buffer. When
    /// `trace_layer_stride` is given, the trace buffers receive the batch's activations, with
    /// layer `il`'s outputs starting `il * trace_layer_stride` floats into the layer trace buffer.
    fn record_batch(
        &mut self,
        pos: u32,
        n: u32,
        want_logits: bool,
        trace_layer_stride: Option<usize>,
    ) -> Result<(), Error> {
        let b = self.buffers;
        let model = self.model;
        let hp = model.hyperparameters();
        let n_embd = to_usize(hp.n_embd);
        let traced = trace_layer_stride.is_some();
        let h = b.hidden.floats(0);
        let normed = b.normed.floats(0);
        let ffn = b.ffn.floats(0);

        self.metal.embed_q8_0(
            b.weights(&model.token_embd.tensor),
            hp.n_vocab,
            b.token(pos),
            h,
            hp.n_embd,
            n,
        )?;
        self.metal.barrier();
        if traced {
            self.metal
                .copy(h, b.trace_embedding.floats(0), n * hp.n_embd)?;
        }

        for (il, layer) in model.layers.iter().enumerate() {
            self.metal.rms_norm(
                h,
                b.weights(&layer.attn_norm.tensor),
                normed,
                hp.n_embd,
                n,
                hp.norm_eps,
            )?;
            self.metal.barrier();
            match &layer.mixer {
                Mixer::Attention(attention) => {
                    self.attention_batch(attention, layer.cache_index, pos, n)?;
                }
                Mixer::Conv(conv) => self.conv_batch(conv, layer.cache_index, n)?,
            }

            self.metal.rms_norm(
                h,
                b.weights(&layer.ffn_norm.tensor),
                normed,
                hp.n_embd,
                n,
                hp.norm_eps,
            )?;
            self.metal.barrier();
            // The up projection's store combines with the gate projection already in the
            // buffer, which applies the SwiGLU.
            self.matmul(&layer.ffn_gate, normed, ffn, n, Store::Overwrite)?;
            self.metal.barrier();
            self.matmul(&layer.ffn_up, normed, ffn, n, Store::Swiglu)?;
            self.metal.barrier();
            self.matmul(&layer.ffn_down, ffn, h, n, Store::Accumulate)?;
            self.metal.barrier();

            if let Some(stride) = trace_layer_stride {
                self.metal
                    .copy(h, b.trace_layers.floats(il * stride), n * hp.n_embd)?;
            }
        }

        if traced {
            self.metal.rms_norm(
                h,
                b.weights(&model.output_norm.tensor),
                b.trace_final.floats(0),
                hp.n_embd,
                n,
                hp.norm_eps,
            )?;
        }
        if want_logits {
            let options = self.normalized(&model.output_norm);
            let last = b.hidden.floats(to_usize(n - 1) * n_embd);
            self.matvec(&model.output, last, b.logits.floats(0), options)?;
        }
        Ok(())
    }
}

/// Options that add the product to the residual stream.
const ADD_TO_RESIDUAL: MatvecOptions<'static> = MatvecOptions {
    norm: None,
    accumulate: true,
};
