//! The LFM2 forward pass on the Metal GPU.
//!
//! Each call records its work into command buffers and waits for them before it returns. The
//! weights stay in the memory-mapped GGUF file, and the GPU reads them in place.

use std::time::Instant;

use bobcat_gguf::{Tensor, TensorType};
use bobcat_metal::{
    Buffer, Format, MOE_MAX_EXPERTS, MOE_MAX_USED, MatvecOptions, Metal, Norm, Readback, Store,
    Ticket, View, attention_scratch_floats, route_bytes,
};

use crate::error::Error;
use crate::lfm2::{
    Attention, Conv, Ffn, Layer, Matrix, Mixer, Model, Moe, Trace, Vector, to_usize,
};
use crate::storage::Storage;

/// Each lane of the attention kernel holds a whole number of elements of a head, and a
/// threadgroup serves at most four query heads of up to 256 elements. Must match
/// `ATTENTION_MAX_GROUP` and `ATTENTION_MAX_HEAD_DIM` in `common.metal`.
pub(crate) const SIMD_WIDTH: u32 = 32;
pub(crate) const ATTENTION_MAX_GROUP: u32 = 4;
pub(crate) const ATTENTION_MAX_HEAD_DIM: u32 = 256;

/// `short_conv_history` in `conv.metal` keeps up to 8 history inputs per channel in
/// registers.
const CONV_MAX_KERNEL: u32 = 9;

/// Generation keeps up to this many steps submitted ahead of the GPU. Each step depends on the one
/// before, so more steps in flight save nothing once the GPU never waits for the CPU.
pub(crate) const GENERATE_IN_FLIGHT: usize = 3;

/// Prefill runs the prompt through the model this many tokens at a time.
pub(crate) const PREFILL_BATCH: u32 = 512;
/// Expansion pays for itself on a full prefill batch.
pub(crate) const K_EXPAND_MIN_TOKENS: u32 = 512;

pub(crate) const FLOAT_BYTES: usize = 4;
pub(crate) const TOKEN_BYTES: usize = 4;

/// The GPU buffers of one LFM2 decode.
#[derive(Debug)]
struct Buffers {
    weights: Buffer,
    expanded_weights: Option<Buffer>,
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
    /// The experts each token of a batch picks and their weights, for mixture-of-experts layers.
    route: Buffer,
    /// A batch's routes grouped by expert: each expert's first entry, then the entries.
    expert_offsets: Buffer,
    expert_entries: Buffer,
    /// One row of `n_embd` floats per routed expert of each token of a batch.
    expert_out: Buffer,
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

/// The state of a sequence after some tokens, which [`Lfm2Metal::restore`] returns to.
///
/// A checkpoint copies the convolution history and the latest logits, which generation starts
/// from. The KV cache stays on the GPU, so a checkpoint stays valid until a reset or a call
/// writes the cache positions before its end.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    resets: u64,
    n_past: u32,
    conv_state: Vec<f32>,
    logits: Vec<f32>,
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
    /// The number of resets so far, which tells a restore whether a checkpoint predates a reset.
    resets: u64,
    batch: u32,
    trace_rows: u32,
    kv_half: bool,
    last_encode_seconds: f64,
    last_gpu_seconds: f64,
    buffers: Buffers,
    /// One token slot per step [`Lfm2Metal::generate_stream`] keeps in flight.
    readback: Readback,
}

impl<'a> Lfm2Metal<'a> {
    /// Prepare to decode sequences of up to `n_ctx` tokens of `model` on `metal`.
    ///
    /// Every matrix of `model` must be Q8_0, Q4_0, Q4_K, Q5_K, or Q6_K. The KV cache holds half
    /// precision when `kv_half` is true and floats otherwise.
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
        let expanded_bytes = model
            .layers
            .iter()
            .flat_map(matmul_matrices)
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
            // A mixture-of-experts layer keeps one row of expert activations per routed expert.
            ffn: floats(rows * to_usize(hp.n_ff.max(hp.n_experts_used * hp.n_ff_expert)))?,
            route: metal.new_buffer(route_bytes(hp.n_experts_used, batch))?,
            expert_offsets: metal.new_buffer((to_usize(hp.n_experts) + 1) * TOKEN_BYTES)?,
            expert_entries: metal.new_buffer(rows * to_usize(hp.n_experts_used) * TOKEN_BYTES)?,
            expert_out: floats(rows * to_usize(hp.n_experts_used) * n_embd)?,
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

    /// Forget every token, so the next call starts a new sequence.
    pub fn reset(&mut self) -> Result<(), Error> {
        // A new sequence starts from zero convolution history. The KV cache needs no clearing,
        // because attention reads only the positions a sequence has written.
        let hp = self.model.hyperparameters();
        let conv_floats =
            to_usize(hp.n_conv_layers) * to_usize(hp.conv_kernel - 1) * to_usize(hp.n_embd);
        self.metal
            .write(self.buffers.conv_state.at(0), &vec![0.0_f32; conv_floats])?;
        self.n_past = 0;
        self.resets += 1;
        Ok(())
    }

    /// Return the sequence state after every token run so far, for [`Lfm2Metal::restore`].
    pub fn checkpoint(&self) -> Result<Checkpoint, Error> {
        let mut conv_state = vec![0.0_f32; self.conv_floats()];
        self.metal
            .read(self.buffers.conv_state.at(0), &mut conv_state)?;
        let mut logits = vec![0.0_f32; to_usize(self.model.hyperparameters().n_vocab)];
        self.metal.read(self.buffers.logits.at(0), &mut logits)?;
        Ok(Checkpoint {
            resets: self.resets,
            n_past: self.n_past,
            conv_state,
            logits,
        })
    }

    /// Return to the sequence state of `checkpoint`, so the next call continues from its tokens.
    ///
    /// The checkpoint must come from this decode, with no reset since and no call that wrote the
    /// positions before its end. Positions past the checkpoint's tokens still hold later keys and
    /// values, and attention reads none of them.
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        if checkpoint.resets != self.resets {
            return Err(Error::Argument("the checkpoint predates a reset"));
        }
        let n_vocab = to_usize(self.model.hyperparameters().n_vocab);
        if checkpoint.conv_state.len() != self.conv_floats()
            || checkpoint.logits.len() != n_vocab
            || checkpoint.n_past > self.n_ctx
        {
            return Err(Error::Argument(
                "the checkpoint comes from a different model or context",
            ));
        }
        self.metal
            .write(self.buffers.conv_state.at(0), &checkpoint.conv_state)?;
        self.metal
            .write(self.buffers.logits.at(0), &checkpoint.logits)?;
        self.n_past = checkpoint.n_past;
        Ok(())
    }

    /// Return the number of floats of convolution history across all layers.
    fn conv_floats(&self) -> usize {
        let hp = self.model.hyperparameters();
        to_usize(hp.n_conv_layers) * to_usize(hp.conv_kernel - 1) * to_usize(hp.n_embd)
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

    /// Select the most likely token from the latest GPU logits without advancing the sequence.
    ///
    /// The preceding call must have produced logits. Repeated selections return the same token.
    /// A caller can emit this token immediately, then use [`Self::generate_next`] to consume it
    /// and select subsequent output. Selection leaves checkpoints and context capacity unchanged.
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

    /// Decode `out.len()` tokens greedily into `out`.
    ///
    /// The first token is the argmax of the logits of the previous call, which must have
    /// produced logits. Each token then runs through the model to choose the next. The GPU picks
    /// every token itself, and the CPU submits several steps ahead, so the GPU never waits for
    /// the CPU.
    pub fn generate(&mut self, out: &mut [u32]) -> Result<(), Error> {
        self.generate_inner(out, false)
    }

    /// Consume the current greedy token and return the next `out.len()` greedy selections.
    ///
    /// Call [`Self::greedy_token`] to emit the first token before using this method. Each
    /// returned token follows one forward pass. The final selection shares the last forward
    /// pass's GPU submission and remains unconsumed until the next call. An empty slice leaves
    /// the sequence unchanged.
    pub fn generate_next(&mut self, out: &mut [u32]) -> Result<(), Error> {
        self.generate_inner(out, true)
    }

    /// Decode greedily from the current logits and pass each token to `emit` as soon as the
    /// GPU selects it, until `emit` returns false or `max_tokens` tokens have gone out.
    ///
    /// The previous call must have produced logits. The CPU keeps several steps submitted ahead
    /// of the GPU, so the GPU never waits for the CPU, and a token reaches `emit` when the step
    /// that selects it finishes. Steps already submitted when `emit` stops still run.
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

    fn generate_inner(&mut self, out: &mut [u32], select_next: bool) -> Result<(), Error> {
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
            let recorded = self
                .recorder()
                .record_generated_step(pos, n_vocab)
                .and_then(|()| {
                    if select_next && i + 1 == n {
                        self.metal.barrier();
                        self.metal.argmax(
                            self.buffers.logits.at(0),
                            self.buffers.greedy_token.at(0),
                            n_vocab,
                        )?;
                    }
                    Ok(())
                });
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
        if select_next && !out.is_empty() {
            out.rotate_left(1);
            let last = out.len() - 1;
            self.metal
                .read(self.buffers.greedy_token.at(0), &mut out[last..])?;
        }
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

/// Return the dense feed-forward block of `layer`.
/// Return the matrices of `layer` that batches multiply with [`Recorder::matmul`]. The experts
/// of a mixture-of-experts block multiply through their own launches.
fn matmul_matrices(layer: &Layer) -> Vec<&Matrix> {
    let mut matrices = match &layer.mixer {
        Mixer::Attention(attention) => {
            vec![&attention.q, &attention.k, &attention.v, &attention.output]
        }
        Mixer::Conv(conv) => vec![&conv.in_proj, &conv.out_proj],
    };
    if let Ffn::Dense(dense) = &layer.ffn {
        matrices.extend([&dense.gate, &dense.up, &dense.down]);
    }
    matrices
}

/// Check that the Metal kernels can run `model`.
fn check_supported(model: &Model) -> Result<(), Error> {
    let hp = model.hyperparameters();
    for matrix in [&model.token_embd, &model.output]
        .into_iter()
        .chain(model.layers.iter().flat_map(matmul_matrices))
    {
        format(matrix)?;
    }
    for layer in &model.layers {
        // The fused SwiGLU launches read the gate and up matrices with one kernel.
        let (gate, up) = match &layer.ffn {
            Ffn::Dense(dense) => (&dense.gate, &dense.up),
            Ffn::Moe(moe) => {
                check_experts(moe, hp.n_experts, hp.n_experts_used)?;
                (&moe.gate, &moe.up)
            }
        };
        if format(gate)? != format(up)? {
            return Err(Error::MetalUnsupported(format!(
                "gate and up matrices of one type, got {:?} and {:?}",
                gate.tensor.data_type(),
                up.tensor.data_type()
            )));
        }
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

/// Check that the expert kernels can run the mixture of experts `moe`, which routes each token
/// to `n_used` of `n_experts` experts.
fn check_experts(moe: &Moe, n_experts: u32, n_used: u32) -> Result<(), Error> {
    if n_experts > MOE_MAX_EXPERTS || n_used > MOE_MAX_USED {
        return Err(Error::MetalUnsupported(format!(
            "at most {MOE_MAX_USED} of {MOE_MAX_EXPERTS} experts per token, got {n_used} of \
             {n_experts}"
        )));
    }
    if moe.router.tensor.data_type() != TensorType::F32 {
        return Err(Error::MetalUnsupported(format!(
            "an F32 router, got {:?}",
            moe.router.tensor.data_type()
        )));
    }
    for matrix in [&moe.gate, &moe.up, &moe.down] {
        if !format(matrix)?.holds_experts() {
            return Err(Error::MetalUnsupported(format!(
                "experts in Q4_0, Q4_K, or Q6_K, got {:?}",
                matrix.tensor.data_type()
            )));
        }
    }
    Ok(())
}

/// Report whether a full prefill batch expands matrices of `data_type` into half precision before
/// multiplying, which lets the multiply use the Metal 4 tensor path. Q4_0 multiplies faster from
/// its blocks with half inputs.
pub(crate) fn expands(data_type: TensorType) -> bool {
    matches!(
        data_type,
        TensorType::Q4K | TensorType::Q5K | TensorType::Q6K
    )
}

/// Return the kernel format of `matrix`.
pub(crate) fn format(matrix: &Matrix) -> Result<Format, Error> {
    match matrix.tensor.data_type() {
        TensorType::Q8_0 => Ok(Format::Q8_0),
        TensorType::Q4_0 => Ok(Format::Q4_0),
        TensorType::Q4K => Ok(Format::Q4K),
        TensorType::Q5K => Ok(Format::Q5K),
        TensorType::Q6K => Ok(Format::Q6K),
        TensorType::F32 => Ok(Format::F32),
        TensorType::F16 => Ok(Format::F16),
        TensorType::Bf16 => Err(Error::MetalUnsupported("matrices in bfloat16".to_owned())),
    }
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
                Format::F16,
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

    /// Record the gated short convolution `conv` on the residual stream and add its output back.
    fn conv(&mut self, conv: &Conv, cache_index: u32, norm: &Vector) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let history = to_usize(cache_index) * to_usize(hp.conv_kernel - 1) * to_usize(hp.n_embd);
        let h = b.hidden.floats(0);
        let options = self.normalized(norm);

        let in_format = format(&conv.in_proj)?;
        if in_format.fuses_conv() {
            // One launch computes the projection and the convolution, which saves a launch and
            // the barrier between them on every convolution layer.
            self.metal.matvec_conv(
                in_format,
                b.weights(&conv.in_proj.tensor),
                conv.in_proj.n_cols,
                h,
                options.norm,
                b.weights(&conv.taps.tensor),
                b.conv_state.floats(history),
                b.conv_out.floats(0),
                hp.n_embd,
                hp.conv_kernel,
            )?;
        } else {
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
        }
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
            hp.head_dim,
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
            hp.head_dim,
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
            match &layer.mixer {
                Mixer::Attention(attention) => {
                    self.attention(attention, layer.cache_index, &layer.attn_norm, pos)?;
                }
                Mixer::Conv(conv) => self.conv(conv, layer.cache_index, &layer.attn_norm)?,
            }

            match &layer.ffn {
                Ffn::Dense(dense) => {
                    self.metal.matvec_swiglu(
                        format(&dense.gate)?,
                        b.weights(&dense.gate.tensor),
                        b.weights(&dense.up.tensor),
                        dense.gate.n_rows,
                        dense.gate.n_cols,
                        h,
                        Norm {
                            weight: b.weights(&layer.ffn_norm.tensor),
                            eps: hp.norm_eps,
                        },
                        ffn,
                    )?;
                    self.metal.barrier();
                    self.matvec(&dense.down, ffn, h, ADD_TO_RESIDUAL)?;
                    self.metal.barrier();
                }
                Ffn::Moe(moe) => self.moe(moe, 1, Some(&layer.ffn_norm))?,
            }

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

    /// Record the mixture of experts `moe` on the `n` tokens of the batch and add its output to
    /// the residual stream.
    ///
    /// With `norm`, the routing launch normalizes the residual stream into the normed buffer.
    /// Otherwise the normed buffer already holds the normalized tokens. One token runs
    /// matrix-vector products on its experts. A batch groups its tokens by expert, so each
    /// expert's weights stream through the GPU once per batch.
    fn moe(&mut self, moe: &Moe, n: u32, norm: Option<&Vector>) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let normed = b.normed.floats(0);
        let route = b.route.at(0);
        let ffn = b.ffn.floats(0);
        let (input, fused_norm) = match norm {
            Some(norm) => (
                b.hidden.floats(0),
                Some((
                    Norm {
                        weight: b.weights(&norm.tensor),
                        eps: hp.norm_eps,
                    },
                    normed,
                )),
            ),
            None => (normed, None),
        };
        self.metal.moe_route(
            input,
            fused_norm,
            b.weights(&moe.router.tensor),
            b.weights(&moe.expert_bias.tensor),
            route,
            hp.n_embd,
            hp.n_experts,
            hp.n_experts_used,
            n,
        )?;
        self.metal.barrier();
        if n > 1 {
            return self.grouped_moe(moe, n);
        }
        self.metal.matvec_experts_swiglu(
            format(&moe.gate)?,
            b.weights(&moe.gate.tensor),
            b.weights(&moe.up.tensor),
            hp.n_ff_expert,
            hp.n_embd,
            hp.n_experts,
            normed,
            route,
            hp.n_experts_used,
            ffn,
            n,
        )?;
        self.metal.barrier();
        self.metal.matvec_experts_down(
            format(&moe.down)?,
            b.weights(&moe.down.tensor),
            hp.n_embd,
            hp.n_ff_expert,
            hp.n_experts,
            ffn,
            route,
            hp.n_experts_used,
            b.hidden.floats(0),
            n,
        )?;
        self.metal.barrier();
        Ok(())
    }

    /// Record the experts of `moe` on the `n` tokens of a batch, grouped by expert, once the
    /// route buffer holds their routes.
    fn grouped_moe(&mut self, moe: &Moe, n: u32) -> Result<(), Error> {
        let b = self.buffers;
        let hp = self.model.hyperparameters();
        let route = b.route.at(0);
        let offsets = b.expert_offsets.at(0);
        let entries = b.expert_entries.at(0);
        let ffn = b.ffn.floats(0);
        let expert_out = b.expert_out.floats(0);
        self.metal
            .moe_group(route, offsets, entries, hp.n_experts, hp.n_experts_used, n)?;
        self.metal.barrier();
        // One launch multiplies the gate and up projections and applies the SwiGLU.
        self.metal.matmul_experts(
            format(&moe.gate)?,
            b.weights(&moe.gate.tensor),
            Some(b.weights(&moe.up.tensor)),
            hp.n_ff_expert,
            hp.n_embd,
            hp.n_experts,
            b.normed.floats(0),
            true,
            offsets,
            entries,
            hp.n_experts_used,
            ffn,
            n,
        )?;
        self.metal.barrier();
        self.metal.matmul_experts(
            format(&moe.down)?,
            b.weights(&moe.down.tensor),
            None,
            hp.n_embd,
            hp.n_ff_expert,
            hp.n_experts,
            ffn,
            false,
            offsets,
            entries,
            hp.n_experts_used,
            expert_out,
            n,
        )?;
        self.metal.barrier();
        self.metal.moe_combine(
            expert_out,
            route,
            b.hidden.floats(0),
            hp.n_embd,
            hp.n_experts_used,
            n,
        )?;
        self.metal.barrier();
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
            match &layer.ffn {
                Ffn::Dense(dense) => {
                    // The up projection's store combines with the gate projection already in
                    // the buffer, which applies the SwiGLU.
                    self.matmul(&dense.gate, normed, ffn, n, Store::Overwrite)?;
                    self.metal.barrier();
                    self.matmul(&dense.up, normed, ffn, n, Store::Swiglu)?;
                    self.metal.barrier();
                    self.matmul(&dense.down, ffn, h, n, Store::Accumulate)?;
                    self.metal.barrier();
                }
                Ffn::Moe(moe) => self.moe(moe, n, None)?,
            }

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
