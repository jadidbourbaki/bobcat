//! The Qwen3.5 forward pass on the Metal GPU.
//!
//! The pass mirrors [`crate::Lfm2Metal`]: each call records its work into command buffers and
//! waits for them before it returns, and the weights stay in the memory-mapped GGUF file. The
//! Gated DeltaNet layers keep their convolution history and recurrent state in GPU buffers, so a
//! sequence's state never leaves the GPU between calls.

use std::time::Instant;

use bobcat_gguf::Tensor;
use bobcat_metal::{
    Buffer, DeltaShape, GDN_MAX_K_DIM, GDN_MAX_KERNEL, MatvecOptions, Metal, Norm, Readback, Store,
    Ticket, View, attention_scratch_floats,
};

use crate::error::Error;
use crate::lfm2::{Matrix, Trace, Vector, to_usize};
use crate::lfm2_metal::{
    ATTENTION_MAX_GROUP, ATTENTION_MAX_HEAD_DIM, FLOAT_BYTES, GENERATE_IN_FLIGHT,
    K_EXPAND_MIN_TOKENS, PREFILL_BATCH, SIMD_WIDTH, TOKEN_BYTES, expands, format,
};
use crate::qwen35::{Attention, DeltaNet, Layer, Mixer, Model};
use crate::storage::Storage;

/// The GPU buffers of one Qwen3.5 decode.
#[derive(Debug)]
struct Buffers {
    weights: Buffer,
    expanded_weights: Option<Buffer>,
    k_cache: Buffer,
    v_cache: Buffer,
    /// The latest raw convolution inputs of each DeltaNet layer.
    conv_state: Buffer,
    /// The recurrent state matrices of each DeltaNet layer.
    delta_state: Buffer,
    hidden: Buffer,
    normed: Buffer,
    /// The query projection, each head's queries followed by its gate.
    q_gate: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    scores: Buffer,
    qkv: Buffer,
    conv_out: Buffer,
    z: Buffer,
    beta: Buffer,
    alpha: Buffer,
    delta_out: Buffer,
    ffn: Buffer,
    logits: Buffer,
    greedy_token: Buffer,
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

/// The state of a sequence after some tokens, which [`Qwen35Metal::restore`] returns to.
///
/// A checkpoint copies the DeltaNet states and the latest logits. The KV cache stays on the GPU,
/// so a checkpoint stays valid until a reset or a call writes the cache positions before its end.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    resets: u64,
    n_past: u32,
    conv_state: Vec<f32>,
    delta_state: Vec<f32>,
    logits: Vec<f32>,
}

/// The GPU state of one Qwen3.5 decode: the weights, the caches, the DeltaNet states, and the
/// scratch buffers.
///
/// The scratch buffers hold one row per token of a prefill batch, and decode uses the first row.
#[derive(Debug)]
pub struct Qwen35Metal<'a> {
    model: &'a Model,
    metal: &'a mut Metal,
    n_ctx: u32,
    n_past: u32,
    /// The number of resets so far, which tells a restore whether a checkpoint predates a reset.
    resets: u64,
    batch: u32,
    trace_rows: u32,
    kv_half: bool,
    last_encode_seconds: f64,
    last_gpu_seconds: f64,
    buffers: Buffers,
    /// One token slot per step [`Qwen35Metal::generate_stream`] keeps in flight.
    readback: Readback,
}

impl<'a> Qwen35Metal<'a> {
    /// Prepare to decode sequences of up to `n_ctx` tokens of `model` on `metal`.
    ///
    /// Every matrix of `model` must be Q8_0, Q4_0, Q4_K, or Q6_K. The KV cache holds half precision
    /// when `kv_half` is true and floats otherwise.
    pub fn new(
        model: &'a Model,
        metal: &'a mut Metal,
        n_ctx: u32,
        kv_half: bool,
    ) -> Result<Self, Error> {
        if n_ctx == 0 {
            return Err(Error::Argument("the context must hold a token"));
        }
        check_supported(model)?;
        let hp = model.hyperparameters();

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
        let head_dim = to_usize(hp.head_dim);
        let kv_dim = to_usize(hp.n_kv_heads) * head_dim;
        let q_dim = to_usize(hp.n_heads) * head_dim;
        let conv_dim = to_usize(hp.conv_dim());
        let value_dim = to_usize(hp.value_dim());
        let n_v_heads = to_usize(hp.n_v_heads);
        let cache_elements = to_usize(hp.n_attention_layers) * to_usize(n_ctx) * kv_dim;
        let cache_bytes = cache_elements * if kv_half { 2 } else { FLOAT_BYTES };
        let batch = n_ctx.min(PREFILL_BATCH);
        let expanded_bytes = model
            .layers
            .iter()
            .flat_map(Layer::matrices)
            .filter(|matrix| batch >= K_EXPAND_MIN_TOKENS && expands(matrix.tensor.data_type()))
            .map(|matrix| to_usize(matrix.n_rows) * to_usize(matrix.n_cols) * 2)
            .max()
            .unwrap_or(0);
        let expanded_weights = if expanded_bytes > 0 {
            Some(metal.new_buffer(expanded_bytes)?)
        } else {
            None
        };
        let rows = to_usize(batch);
        let floats = |count: usize| metal.new_buffer(count * FLOAT_BYTES);

        let buffers = Buffers {
            weights,
            expanded_weights,
            k_cache: metal.new_buffer(cache_bytes)?,
            v_cache: metal.new_buffer(cache_bytes)?,
            conv_state: floats(conv_floats(model))?,
            delta_state: floats(delta_floats(model))?,
            hidden: floats(rows * n_embd)?,
            normed: floats(rows * n_embd)?,
            q_gate: floats(rows * 2 * q_dim)?,
            q: floats(rows * q_dim)?,
            k: floats(rows * kv_dim)?,
            v: floats(rows * kv_dim)?,
            attn: floats(rows * q_dim)?,
            scores: floats(attention_scratch_floats(
                hp.n_heads,
                hp.head_dim,
                n_ctx,
                batch,
            ))?,
            qkv: floats(rows * conv_dim)?,
            conv_out: floats(rows * conv_dim)?,
            z: floats(rows * value_dim)?,
            beta: floats(rows * n_v_heads)?,
            alpha: floats(rows * n_v_heads)?,
            delta_out: floats(rows * value_dim)?,
            ffn: floats(rows * to_usize(hp.n_ff))?,
            logits: floats(to_usize(hp.n_vocab))?,
            greedy_token: metal.new_buffer(TOKEN_BYTES)?,
            tokens: metal.new_buffer(to_usize(n_ctx) * TOKEN_BYTES)?,
            trace_embedding: floats(n_embd)?,
            trace_layers: floats(to_usize(hp.n_layers) * n_embd)?,
            trace_final: floats(n_embd)?,
        };
        let readback = metal.new_readback(GENERATE_IN_FLIGHT)?;

        Ok(Self {
            model,
            metal,
            n_ctx,
            n_past: 0,
            resets: 0,
            batch,
            trace_rows: 1,
            kv_half,
            last_encode_seconds: 0.0,
            last_gpu_seconds: 0.0,
            buffers,
            readback,
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

    /// Return the number of tokens the sequence holds.
    pub fn n_past(&self) -> u32 {
        self.n_past
    }

    /// Forget every token, so the next call starts a new sequence.
    pub fn reset(&mut self) -> Result<(), Error> {
        // A new sequence starts from zero convolution history and zero DeltaNet states. The KV
        // cache needs no clearing, because attention reads only the positions a sequence has
        // written.
        self.metal.write(
            self.buffers.conv_state.at(0),
            &vec![0.0_f32; conv_floats(self.model)],
        )?;
        self.metal.write(
            self.buffers.delta_state.at(0),
            &vec![0.0_f32; delta_floats(self.model)],
        )?;
        self.n_past = 0;
        self.resets += 1;
        Ok(())
    }

    /// Return the sequence state after every token run so far, for [`Qwen35Metal::restore`].
    pub fn checkpoint(&self) -> Result<Checkpoint, Error> {
        let mut conv_state = vec![0.0_f32; conv_floats(self.model)];
        self.metal
            .read(self.buffers.conv_state.at(0), &mut conv_state)?;
        let mut delta_state = vec![0.0_f32; delta_floats(self.model)];
        self.metal
            .read(self.buffers.delta_state.at(0), &mut delta_state)?;
        let mut logits = vec![0.0_f32; to_usize(self.model.hyperparameters().n_vocab)];
        self.metal.read(self.buffers.logits.at(0), &mut logits)?;
        Ok(Checkpoint {
            resets: self.resets,
            n_past: self.n_past,
            conv_state,
            delta_state,
            logits,
        })
    }

    /// Return to the sequence state of `checkpoint`, so the next call continues from its tokens.
    ///
    /// The checkpoint must come from this decode, with no reset since and no call that wrote the
    /// positions before its end.
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        if checkpoint.resets != self.resets {
            return Err(Error::Argument("the checkpoint predates a reset"));
        }
        if checkpoint.conv_state.len() != conv_floats(self.model)
            || checkpoint.delta_state.len() != delta_floats(self.model)
            || checkpoint.logits.len() != to_usize(self.model.hyperparameters().n_vocab)
            || checkpoint.n_past > self.n_ctx
        {
            return Err(Error::Argument(
                "the checkpoint comes from a different model or context",
            ));
        }
        self.metal
            .write(self.buffers.conv_state.at(0), &checkpoint.conv_state)?;
        self.metal
            .write(self.buffers.delta_state.at(0), &checkpoint.delta_state)?;
        self.metal
            .write(self.buffers.logits.at(0), &checkpoint.logits)?;
        self.n_past = checkpoint.n_past;
        Ok(())
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

    /// Run the model on `token` at the next position, with the same outputs as
    /// [`Model::step`].
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

    /// Select the most likely token from the latest GPU logits without advancing the sequence.
    ///
    /// The preceding call must have produced logits.
    pub fn greedy_token(&mut self) -> Result<u32, Error> {
        if self.n_past == 0 {
            return Err(Error::Argument("generation needs a prompt first"));
        }
        let start = Instant::now();
        self.metal.begin()?;
        let recorded = self.metal.argmax(
            self.buffers.logits.floats(0),
            self.buffers.greedy_token.at(0),
            self.model.hyperparameters().n_vocab,
        );
        if let Err(error) = recorded {
            self.metal.discard();
            return Err(error.into());
        }
        self.last_encode_seconds = start.elapsed().as_secs_f64();
        self.last_gpu_seconds = self.metal.end()?;
        let mut token = [0_u32];
        self.metal
            .read(self.buffers.greedy_token.at(0), &mut token)?;
        Ok(token[0])
    }

    /// Decode greedily from the current logits and pass each token to `emit` as soon as the
    /// GPU selects it, until `emit` returns false or `max_tokens` tokens have gone out.
    ///
    /// The previous call must have produced logits. The CPU keeps several steps submitted ahead
    /// of the GPU, and a token reaches `emit` when the step that selects it finishes. Steps
    /// already submitted when `emit` stops still run.
    ///
    /// Return the tokens the sequence gained, in order. They start with every emitted token but
    /// the last, whose forward pass never runs, and may include tokens past the last emitted
    /// one.
    pub fn generate_stream(
        &mut self,
        max_tokens: u32,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, Error> {
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let first = self.greedy_token()?;
        if !emit(first) {
            return Ok(Vec::new());
        }
        // Each step runs one token and selects the next, so the first token needs no step.
        let steps = (max_tokens - 1).min(self.n_ctx - self.n_past);
        let start = self.n_past;
        let n_vocab = self.model.hyperparameters().n_vocab;
        let mut tickets: [Option<Ticket>; GENERATE_IN_FLIGHT] = [None; GENERATE_IN_FLIGHT];
        let mut emitting = true;
        let mut committed = 0;
        let mut gpu_seconds = 0.0;
        let mut encode_seconds = 0.0;
        let mut result = Ok(());

        for step in 0..steps {
            // The slot's previous step selected the token that goes out next.
            let slot = to_usize(step) % GENERATE_IN_FLIGHT;
            if let Some(ticket) = tickets[slot].take() {
                match self.finish_stream_step(ticket, slot, &mut emit, &mut emitting) {
                    Ok(seconds) => gpu_seconds += seconds,
                    Err(error) => {
                        result = Err(error);
                        break;
                    }
                }
            }
            if !emitting {
                break;
            }

            let encode_start = Instant::now();
            if let Err(error) = self.metal.begin() {
                result = Err(error.into());
                break;
            }
            if let Err(error) = self.record_stream_step(start + step, step == 0, slot, n_vocab) {
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
            committed += 1;
            encode_seconds += encode_start.elapsed().as_secs_f64();
        }

        // Finish the steps still in flight in submission order, which is the order of their
        // tokens.
        for offset in 0..GENERATE_IN_FLIGHT {
            let slot = (to_usize(committed) + offset) % GENERATE_IN_FLIGHT;
            let Some(ticket) = tickets[slot].take() else {
                continue;
            };
            match self.finish_stream_step(ticket, slot, &mut emit, &mut emitting) {
                Ok(seconds) => gpu_seconds += seconds,
                Err(error) => {
                    if result.is_ok() {
                        result = Err(error);
                    }
                }
            }
        }
        self.n_past += committed;
        self.last_encode_seconds = encode_seconds;
        self.last_gpu_seconds = gpu_seconds;
        result?;

        let mut gained = vec![0; to_usize(committed)];
        self.metal.read(self.buffers.token(start), &mut gained)?;
        Ok(gained)
    }

    /// Wait for the stream step with `ticket` and pass the token it selected into `slot` to
    /// `emit` while `emitting` holds. Return the GPU seconds the wait covered.
    fn finish_stream_step(
        &mut self,
        ticket: Ticket,
        slot: usize,
        emit: &mut impl FnMut(u32) -> bool,
        emitting: &mut bool,
    ) -> Result<f64, Error> {
        let seconds = self.metal.wait(ticket)?;
        if *emitting {
            let token = self.metal.read_readback(&self.readback, slot)?;
            *emitting = emit(token);
        }
        Ok(seconds)
    }

    /// Record the stream step at position `pos`: run the token there and select the next into
    /// the token buffer and readback `slot`. The first step also selects its own token from the
    /// current logits.
    fn record_stream_step(
        &mut self,
        pos: u32,
        first: bool,
        slot: usize,
        n_vocab: u32,
    ) -> Result<(), Error> {
        if first {
            self.metal.argmax(
                self.buffers.logits.floats(0),
                self.buffers.token(pos),
                n_vocab,
            )?;
            self.metal.barrier();
        }
        self.recorder().record_step(pos, true, false)?;
        self.metal.barrier();
        // The last position of the context has no next slot in the token buffer.
        let next = if pos + 1 < self.n_ctx {
            self.buffers.token(pos + 1)
        } else {
            self.buffers.greedy_token.at(0)
        };
        self.metal.argmax_readback(
            self.buffers.logits.floats(0),
            next,
            &mut self.readback,
            slot,
            n_vocab,
        )?;
        Ok(())
    }

    /// Run the model on `tokens` at the next positions, in batches of up to 512 tokens.
    ///
    /// When `logits` is given, it receives the logits of the last token. The logits buffer
    /// receives them either way, for generation. When `final_norm` is given, it receives every
    /// token's normalized last hidden state, `n_embd` floats per token. When `trace` is given, it
    /// must hold `tokens.len()` tokens and receives every token's activations.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        logits: Option<&mut [f32]>,
        mut final_norm: Option<&mut [f32]>,
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
        if let Some(&token) = tokens.iter().find(|&&token| token >= hp.n_vocab) {
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
        if let Some(final_norm) = &final_norm
            && final_norm.len() != tokens.len() * n_embd
        {
            return Err(Error::Argument(
                "the final norm must hold n_embd floats per token",
            ));
        }
        if let Some(trace) = &trace
            && !trace.holds_shape(hp.n_embd, hp.n_layers, n)
        {
            return Err(Error::Argument("the trace must hold every prompt token"));
        }

        // Nothing is in flight, so every prompt token goes into the shared buffer before the
        // first batch.
        self.metal.write(self.buffers.token(self.n_past), tokens)?;
        let copies = trace.is_some() || final_norm.is_some();
        if copies {
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
                copies,
                trace.is_some().then_some(layer_stride),
            );
            if let Err(error) = recorded {
                self.metal.discard();
                return Err(error);
            }
            let ticket = self.metal.commit()?;
            encode_seconds += encode_start.elapsed().as_secs_f64();

            // A batch whose activations go back to the caller waits, so the next batch cannot
            // overwrite them first. Other batches run back to back.
            if copies || last {
                gpu_seconds += self.metal.wait(ticket)?;
            }
            let start = to_usize(done) * n_embd;
            let rows = to_usize(count) * n_embd;
            let buffers = &self.buffers;
            if let Some(final_norm) = final_norm.as_deref_mut() {
                self.metal.read(
                    buffers.trace_final.at(0),
                    &mut final_norm[start..start + rows],
                )?;
            }
            if let Some(trace) = trace.as_deref_mut() {
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

/// Return the floats of convolution history across the DeltaNet layers of `model`.
fn conv_floats(model: &Model) -> usize {
    let hp = model.hyperparameters();
    to_usize(hp.n_delta_layers) * to_usize(hp.conv_dim()) * to_usize(hp.conv_kernel - 1)
}

/// Return the floats of recurrent state across the DeltaNet layers of `model`.
fn delta_floats(model: &Model) -> usize {
    let hp = model.hyperparameters();
    to_usize(hp.n_delta_layers) * to_usize(hp.n_v_heads) * to_usize(hp.k_head_dim * hp.v_head_dim)
}

/// Return the DeltaNet head layout of `model`.
fn delta_shape(model: &Model) -> DeltaShape {
    let hp = model.hyperparameters();
    DeltaShape {
        n_k_heads: hp.n_k_heads,
        n_v_heads: hp.n_v_heads,
        k_dim: hp.k_head_dim,
        v_dim: hp.v_head_dim,
    }
}

/// Check that the Metal kernels can run `model`.
fn check_supported(model: &Model) -> Result<(), Error> {
    let hp = model.hyperparameters();
    for matrix in [&model.token_embd, &model.output]
        .into_iter()
        .chain(model.layers.iter().flat_map(Layer::matrices))
    {
        format(matrix)?;
    }
    for layer in &model.layers {
        if format(&layer.gate)? != format(&layer.up)? {
            return Err(Error::MetalUnsupported(format!(
                "gate and up matrices of one type, got {:?} and {:?}",
                layer.gate.tensor.data_type(),
                layer.up.tensor.data_type()
            )));
        }
    }
    let group = hp.n_heads / hp.n_kv_heads;
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
    if hp.conv_kernel > GDN_MAX_KERNEL
        || hp.k_head_dim > GDN_MAX_K_DIM
        || !hp.v_head_dim.is_multiple_of(SIMD_WIDTH)
    {
        return Err(Error::MetalUnsupported(format!(
            "DeltaNet layers with at most {GDN_MAX_KERNEL} convolution taps, key heads of at most \
             {GDN_MAX_K_DIM} floats, and value heads of a multiple of {SIMD_WIDTH} floats, got {}, \
             {}, and {}",
            hp.conv_kernel, hp.k_head_dim, hp.v_head_dim
        )));
    }
    Ok(())
}

/// Records the launches of forward passes.
struct Recorder<'r> {
    metal: &'r mut Metal,
    model: &'r Model,
    buffers: &'r Buffers,
    n_ctx: u32,
    kv_half: bool,
}

/// Options that add the product to the residual stream.
const ADD_TO_RESIDUAL: MatvecOptions<'static> = MatvecOptions {
    norm: None,
    accumulate: true,
};

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
        self.metal.matvec(
            format(matrix)?,
            weights,
            matrix.n_rows,
            matrix.n_cols,
            x,
            y,
            options,
        )?;
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
        let format = format(matrix)?;
        if expands(matrix.tensor.data_type()) && n >= K_EXPAND_MIN_TOKENS {
            let expanded = self
                .buffers
                .expanded_weights
                .as_ref()
                .expect("expanding formats have expansion scratch");
            self.metal.expand(
                format,
                weights,
                expanded.at(0),
                matrix.n_rows,
                matrix.n_cols,
            )?;
            self.metal.barrier();
            self.metal.matmul(
                bobcat_metal::Format::F16,
                expanded.at(0),
                matrix.n_rows,
                matrix.n_cols,
                x,
                y,
                n,
                store,
            )?;
            // The next matrix may reuse the expansion buffer in the same concurrent encoder.
            self.metal.barrier();
            return Ok(());
        }
        self.metal.matmul(
            format,
            weights,
            matrix.n_rows,
            matrix.n_cols,
            x,
            y,
            n,
            store,
        )?;
        Ok(())
    }

    /// Return a view of a KV cache buffer at element `index`.
    fn cache_at(&self, buffer: &'r Buffer, index: usize) -> View<'r> {
        buffer.at(index * if self.kv_half { 2 } else { FLOAT_BYTES })
    }

    /// Return the first cache element of attention layer `cache_index`.
    fn layer_base(&self, cache_index: u32) -> usize {
        let hp = self.model.hyperparameters();
        let kv_dim = to_usize(hp.n_kv_heads) * to_usize(hp.head_dim);
        to_usize(cache_index) * to_usize(self.n_ctx) * kv_dim
    }

    /// Record the attention of the `n` tokens at positions `pos` onward, whose query, key, and
    /// value projections fill the scratch, and add its output to the residual stream.
    fn attend(
        &mut self,
        attention: &Attention,
        cache_index: u32,
        pos: u32,
        n: u32,
    ) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let head_dim = hp.head_dim;
        let kv_dim = hp.n_kv_heads * head_dim;
        let q_dim = hp.n_heads * head_dim;
        let layer_base = self.layer_base(cache_index);
        let slot = layer_base + to_usize(pos) * to_usize(kv_dim);

        // The rotations and the value store write different buffers, so they run together. The
        // queries come out of the interleaved query projection, each head's queries followed by
        // its gate.
        self.metal.norm_rope(
            b.q_gate.floats(0),
            b.q.floats(0),
            false,
            b.weights(&attention.q_norm.tensor),
            hp.n_heads,
            head_dim,
            hp.n_rot,
            2 * head_dim,
            pos,
            n,
            2 * q_dim,
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
            head_dim,
            hp.n_rot,
            head_dim,
            pos,
            n,
            kv_dim,
            kv_dim,
            hp.rope_theta,
            hp.norm_eps,
        )?;
        let v_cache = self.cache_at(&b.v_cache, slot);
        if self.kv_half {
            self.metal
                .convert_half(b.v.floats(0), v_cache, n * kv_dim)?;
        } else {
            self.metal.copy(b.v.floats(0), v_cache, n * kv_dim)?;
        }
        self.metal.barrier();
        self.metal.attention(
            b.q.floats(0),
            self.cache_at(&b.k_cache, layer_base),
            self.cache_at(&b.v_cache, layer_base),
            self.kv_half,
            b.scores.floats(0),
            b.attn.floats(0),
            hp.n_heads,
            hp.n_kv_heads,
            head_dim,
            pos,
            n,
            self.n_ctx,
        )?;
        self.metal.barrier();
        self.metal
            .attention_gate(b.attn.floats(0), b.q_gate.floats(0), head_dim, q_dim, n)?;
        self.metal.barrier();
        if n == 1 {
            self.matvec(
                &attention.output,
                b.attn.floats(0),
                b.hidden.floats(0),
                ADD_TO_RESIDUAL,
            )?;
        } else {
            self.matmul(
                &attention.output,
                b.attn.floats(0),
                b.hidden.floats(0),
                n,
                Store::Accumulate,
            )?;
        }
        self.metal.barrier();
        Ok(())
    }

    /// Record the Gated DeltaNet of the `n` tokens whose projections fill the scratch, and add
    /// its output to the residual stream.
    fn delta(&mut self, delta: &DeltaNet, cache_index: u32, n: u32) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let shape = delta_shape(self.model);
        let conv_dim = hp.conv_dim();
        let history = to_usize(cache_index) * to_usize(conv_dim) * to_usize(hp.conv_kernel - 1);
        let state = to_usize(cache_index)
            * to_usize(hp.n_v_heads)
            * to_usize(hp.k_head_dim * hp.v_head_dim);

        // The convolution and the gates read different projections, so they run together.
        self.metal.gdn_conv(
            b.qkv.floats(0),
            b.weights(&delta.conv.tensor),
            b.conv_state.floats(history),
            b.conv_out.floats(0),
            conv_dim,
            hp.conv_kernel,
            n,
        )?;
        self.metal.gdn_gates(
            b.beta.floats(0),
            b.alpha.floats(0),
            b.weights(&delta.a.tensor),
            b.weights(&delta.dt_bias.tensor),
            hp.n_v_heads,
            n,
        )?;
        self.metal.barrier();
        self.metal.gdn_qk_norm(
            b.conv_out.floats(0),
            shape,
            n,
            1.0 / (hp.k_head_dim as f32).sqrt(),
        )?;
        self.metal.barrier();
        self.metal.gdn_recurrence(
            b.conv_out.floats(0),
            b.beta.floats(0),
            b.alpha.floats(0),
            b.delta_state.floats(state),
            b.delta_out.floats(0),
            shape,
            n,
        )?;
        self.metal.barrier();
        // The output norm scales each value head, and SiLU of the gate multiplies it.
        self.metal.rms_norm(
            b.delta_out.floats(0),
            b.weights(&delta.norm.tensor),
            b.delta_out.floats(0),
            hp.v_head_dim,
            n * hp.n_v_heads,
            hp.norm_eps,
        )?;
        self.metal.barrier();
        self.metal
            .silu_mul(b.delta_out.floats(0), b.z.floats(0), n * hp.value_dim())?;
        self.metal.barrier();
        if n == 1 {
            self.matvec(
                &delta.output,
                b.delta_out.floats(0),
                b.hidden.floats(0),
                ADD_TO_RESIDUAL,
            )?;
        } else {
            self.matmul(
                &delta.output,
                b.delta_out.floats(0),
                b.hidden.floats(0),
                n,
                Store::Accumulate,
            )?;
        }
        self.metal.barrier();
        Ok(())
    }

    /// Record a forward pass on the token at position `pos` of the token buffer. With
    /// `want_logits`, the logits go to the logits buffer. With `traced`, the activations also go
    /// to the trace buffers.
    fn record_step(&mut self, pos: u32, want_logits: bool, traced: bool) -> Result<(), Error> {
        let b = self.buffers;
        let model = self.model;
        let hp = model.hyperparameters();
        let h = b.hidden.floats(0);

        self.metal.embed(
            format(&model.token_embd)?,
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
            let options = self.normalized(&layer.attn_norm);
            match &layer.mixer {
                Mixer::Attention(attention) => {
                    self.matvec(&attention.q, h, b.q_gate.floats(0), options)?;
                    self.matvec(&attention.k, h, b.k.floats(0), options)?;
                    self.matvec(&attention.v, h, b.v.floats(0), options)?;
                    self.metal.barrier();
                    self.attend(attention, layer.cache_index, pos, 1)?;
                }
                Mixer::DeltaNet(delta) => {
                    self.matvec(&delta.qkv, h, b.qkv.floats(0), options)?;
                    self.matvec(&delta.gate, h, b.z.floats(0), options)?;
                    self.matvec(&delta.beta, h, b.beta.floats(0), options)?;
                    self.matvec(&delta.alpha, h, b.alpha.floats(0), options)?;
                    self.metal.barrier();
                    self.delta(delta, layer.cache_index, 1)?;
                }
            }

            self.metal.matvec_swiglu(
                format(&layer.gate)?,
                b.weights(&layer.gate.tensor),
                b.weights(&layer.up.tensor),
                layer.gate.n_rows,
                layer.gate.n_cols,
                h,
                Norm {
                    weight: b.weights(&layer.ffn_norm.tensor),
                    eps: hp.norm_eps,
                },
                b.ffn.floats(0),
            )?;
            self.metal.barrier();
            self.matvec(&layer.down, b.ffn.floats(0), h, ADD_TO_RESIDUAL)?;
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

    /// Record a forward pass on the `n` tokens at positions `pos` onward of the token buffer.
    ///
    /// With `want_logits`, the last token's logits go to the logits buffer. With `final_norm`,
    /// every token's normalized last hidden state goes to the final trace buffer. With
    /// `trace_layer_stride`, the activations also go to the trace buffers, each layer's rows that
    /// many floats apart.
    fn record_batch(
        &mut self,
        pos: u32,
        n: u32,
        want_logits: bool,
        final_norm: bool,
        trace_layer_stride: Option<usize>,
    ) -> Result<(), Error> {
        let b = self.buffers;
        let model = self.model;
        let hp = model.hyperparameters();
        let n_embd = to_usize(hp.n_embd);
        let h = b.hidden.floats(0);
        let normed = b.normed.floats(0);
        let ffn = b.ffn.floats(0);

        self.metal.embed(
            format(&model.token_embd)?,
            b.weights(&model.token_embd.tensor),
            hp.n_vocab,
            b.token(pos),
            h,
            hp.n_embd,
            n,
        )?;
        self.metal.barrier();
        if trace_layer_stride.is_some() {
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
                    self.matmul(
                        &attention.q,
                        normed,
                        b.q_gate.floats(0),
                        n,
                        Store::Overwrite,
                    )?;
                    self.matmul(&attention.k, normed, b.k.floats(0), n, Store::Overwrite)?;
                    self.matmul(&attention.v, normed, b.v.floats(0), n, Store::Overwrite)?;
                    self.metal.barrier();
                    self.attend(attention, layer.cache_index, pos, n)?;
                }
                Mixer::DeltaNet(delta) => {
                    self.matmul(&delta.qkv, normed, b.qkv.floats(0), n, Store::Overwrite)?;
                    self.matmul(&delta.gate, normed, b.z.floats(0), n, Store::Overwrite)?;
                    self.matmul(&delta.beta, normed, b.beta.floats(0), n, Store::Overwrite)?;
                    self.matmul(&delta.alpha, normed, b.alpha.floats(0), n, Store::Overwrite)?;
                    self.metal.barrier();
                    self.delta(delta, layer.cache_index, n)?;
                }
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
            self.matmul(&layer.gate, normed, ffn, n, Store::Overwrite)?;
            self.metal.barrier();
            self.matmul(&layer.up, normed, ffn, n, Store::Swiglu)?;
            self.metal.barrier();
            self.matmul(&layer.down, ffn, h, n, Store::Accumulate)?;
            self.metal.barrier();

            if let Some(stride) = trace_layer_stride {
                self.metal
                    .copy(h, b.trace_layers.floats(il * stride), n * hp.n_embd)?;
            }
        }

        if final_norm {
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
