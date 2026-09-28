/* model_lfm2_metal.c runs the LFM2 forward pass on the Metal GPU.  Each
   step records every layer into one command buffer and waits for it.
   The weights stay in the memory-mapped GGUF file, and the GPU reads
   them in place.  */

#include <string.h>
#include <time.h>

#include "error.h"
#include "model_lfm2_metal.h"

enum
{
  /* The attention kernel's limits from kernels.metal.  Each lane holds
     a whole number of elements of a head, and a threadgroup serves at
     most four query heads of up to 128 elements.  */
  SIMD_WIDTH = 32,
  ATTENTION_MAX_GROUP = 4,
  ATTENTION_MAX_HEAD_DIM = 128,
  /* short_conv_history keeps up to 8 history inputs per channel in
     registers.  */
  CONV_MAX_KERNEL = 9,
  /* Generation keeps up to this many steps submitted ahead of the GPU.
     Each step depends on the one before, so more steps in flight save
     nothing once the GPU never waits for the CPU.  */
  GENERATE_IN_FLIGHT = 3,
  /* Prefill runs the prompt through the model this many tokens at a
     time.  */
  PREFILL_BATCH = 512
};

/* Return the monotonic clock time in seconds.  */
static double
now_seconds (void)
{
  struct timespec ts;

  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Report whether TENSOR is null or a Q8_0 matrix.  */
static bool
is_q8_0_or_absent (const struct gip_gguf_tensor *tensor)
{
  return tensor == NULL || tensor->type == GIP_TENSOR_Q8_0;
}

/* Report whether every matrix of MODEL is Q8_0.  */
static bool
all_matrices_q8_0 (const struct gip_lfm2_model *model)
{
  if (!is_q8_0_or_absent (model->token_embd)
      || !is_q8_0_or_absent (model->output))
    return false;
  for (uint32_t il = 0; il < model->n_layers; il++)
    {
      const struct gip_lfm2_layer *l = &model->layers[il];
      if (!is_q8_0_or_absent (l->ffn_gate) || !is_q8_0_or_absent (l->ffn_up)
          || !is_q8_0_or_absent (l->ffn_down) || !is_q8_0_or_absent (l->attn_q)
          || !is_q8_0_or_absent (l->attn_k) || !is_q8_0_or_absent (l->attn_v)
          || !is_q8_0_or_absent (l->attn_output)
          || !is_q8_0_or_absent (l->conv_in_proj)
          || !is_q8_0_or_absent (l->conv_out_proj))
        return false;
    }
  return true;
}

/* Return a new buffer on GPU's device for COUNT floats, or null.  */
static struct gip_metal_buffer *
new_floats (struct gip_lfm2_metal *gpu, size_t count)
{
  size_t bytes;

  if (__builtin_mul_overflow (count, sizeof (float), &bytes))
    return NULL;
  return gip_metal_buffer_new (gpu->metal, bytes);
}

enum gip_status
gip_lfm2_metal_init (const struct gip_lfm2_model *model,
                     struct gip_metal *metal, uint32_t n_ctx, int kv_half,
                     struct gip_lfm2_metal *gpu, char *err, size_t err_size)
{
  size_t n_embd = model->n_embd;
  size_t kv_dim = (size_t)model->n_kv_heads * model->head_dim;
  size_t q_dim = (size_t)model->n_heads * model->head_dim;

  memset (gpu, 0, sizeof *gpu);
  if (n_ctx == 0)
    return GIP_ERR_ARGUMENT;
  if (!all_matrices_q8_0 (model))
    {
      gip_format_error (err, err_size,
                        "the Metal path needs every matrix in Q8_0");
      return GIP_ERR_UNSUPPORTED;
    }
  if (model->head_dim % SIMD_WIDTH != 0
      || model->head_dim > ATTENTION_MAX_HEAD_DIM
      || model->n_heads / model->n_kv_heads > ATTENTION_MAX_GROUP)
    {
      gip_format_error (err, err_size,
                        "the Metal path needs a head size that is a "
                        "multiple of %d up to %d and at most %d query heads "
                        "per KV head, got %u and %u",
                        SIMD_WIDTH, ATTENTION_MAX_HEAD_DIM,
                        ATTENTION_MAX_GROUP, model->head_dim,
                        model->n_heads / model->n_kv_heads);
      return GIP_ERR_UNSUPPORTED;
    }
  if (model->conv_kernel > CONV_MAX_KERNEL)
    {
      gip_format_error (err, err_size,
                        "the Metal path needs a convolution of at most %d "
                        "taps, got %u",
                        CONV_MAX_KERNEL, model->conv_kernel);
      return GIP_ERR_UNSUPPORTED;
    }

  gpu->model = model;
  gpu->metal = metal;
  gpu->n_ctx = n_ctx;
  gpu->kv_half = kv_half != 0;
  gpu->weights_base = model->gguf.base;

  /* A memory-mapped file becomes a GPU buffer with no copy.  A file read
     into the heap is copied into a new buffer, since heap memory is not
     page-aligned.  */
  if (model->gguf.mapped)
    gpu->weights = gip_metal_buffer_wrap (metal, (void *)model->gguf.base,
                                          model->gguf.size);
  else
    {
      gpu->weights = gip_metal_buffer_new (metal, model->gguf.size);
      if (gpu->weights != NULL)
        memcpy (gip_metal_buffer_contents (gpu->weights), model->gguf.base,
                model->gguf.size);
    }

  /* A half-precision cache holds two elements in the space of one
     float.  */
  size_t cache_elements = (size_t)model->n_attention_layers * n_ctx * kv_dim;
  size_t cache_floats
      = gpu->kv_half ? (cache_elements + 1) / 2 : cache_elements;
  size_t conv_floats
      = (size_t)model->n_conv_layers * (model->conv_kernel - 1) * n_embd;
  /* The scratch buffers hold one row per token of a prefill batch.
     Decode uses the first row.  */
  size_t batch = n_ctx < PREFILL_BATCH ? n_ctx : PREFILL_BATCH;
  gpu->batch = (uint32_t)batch;
  gpu->k_cache = new_floats (gpu, cache_floats);
  gpu->v_cache = new_floats (gpu, cache_floats);
  gpu->k = new_floats (gpu, batch * kv_dim);
  gpu->v = new_floats (gpu, batch * kv_dim);
  gpu->conv_state = new_floats (gpu, conv_floats);
  gpu->hidden = new_floats (gpu, batch * n_embd);
  gpu->normed = new_floats (gpu, batch * n_embd);
  gpu->bcx = new_floats (gpu, batch * 3 * n_embd);
  gpu->conv_out = new_floats (gpu, batch * n_embd);
  gpu->q = new_floats (gpu, batch * q_dim);
  gpu->attn = new_floats (gpu, batch * q_dim);
  gpu->scores = new_floats (
      gpu, gip_metal_attention_scratch (model->n_heads, model->head_dim, n_ctx,
                                        (uint32_t)batch));
  gpu->ffn = new_floats (gpu, batch * model->n_ff);
  gpu->logits = new_floats (gpu, model->n_vocab);
  /* An int32_t token id takes the space of one float.  */
  gpu->tokens = new_floats (gpu, n_ctx);
  gpu->trace_embedding = new_floats (gpu, n_embd);
  gpu->trace_layers = new_floats (gpu, (size_t)model->n_layers * n_embd);
  gpu->trace_final = new_floats (gpu, n_embd);

  gpu->trace_rows = 1;
  if (gpu->weights == NULL || gpu->k_cache == NULL || gpu->v_cache == NULL
      || gpu->k == NULL || gpu->v == NULL || gpu->conv_state == NULL
      || gpu->hidden == NULL || gpu->normed == NULL || gpu->bcx == NULL
      || gpu->conv_out == NULL || gpu->q == NULL || gpu->attn == NULL
      || gpu->scores == NULL || gpu->ffn == NULL || gpu->logits == NULL
      || gpu->tokens == NULL || gpu->trace_embedding == NULL
      || gpu->trace_layers == NULL || gpu->trace_final == NULL)
    {
      gip_format_error (err, err_size, "cannot allocate Metal buffers");
      gip_lfm2_metal_free (gpu);
      return GIP_ERR_NOMEM;
    }
  return GIP_OK;
}

void
gip_lfm2_metal_free (struct gip_lfm2_metal *gpu)
{
  struct gip_metal_buffer *buffers[] = {
    gpu->weights,
    gpu->k_cache,
    gpu->v_cache,
    gpu->k,
    gpu->v,
    gpu->conv_state,
    gpu->hidden,
    gpu->normed,
    gpu->bcx,
    gpu->conv_out,
    gpu->q,
    gpu->attn,
    gpu->scores,
    gpu->ffn,
    gpu->logits,
    gpu->tokens,
    gpu->trace_embedding,
    gpu->trace_layers,
    gpu->trace_final,
  };

  for (size_t i = 0; i < sizeof buffers / sizeof buffers[0]; i++)
    gip_metal_buffer_free (buffers[i]);
  memset (gpu, 0, sizeof *gpu);
}

/* Return a view of the weights of TENSOR inside GPU's weight buffer.  */
static struct gip_metal_view
weights_of (const struct gip_lfm2_metal *gpu,
            const struct gip_gguf_tensor *tensor)
{
  size_t offset
      = (size_t)((const unsigned char *)tensor->data - gpu->weights_base);
  return gip_metal_at (gpu->weights, offset);
}

/* Return a view of BUFFER starting at float INDEX.  */
static struct gip_metal_view
floats_at (struct gip_metal_buffer *buffer, size_t index)
{
  return gip_metal_at (buffer, index * sizeof (float));
}

/* Record a multiply of the matrix TENSOR by X into Y with OPTIONS, which
   may be null.  */
static void
matvec (struct gip_lfm2_metal *gpu, const struct gip_gguf_tensor *tensor,
        struct gip_metal_view x, struct gip_metal_view y,
        const struct gip_metal_matvec_options *options)
{
  gip_metal_matvec_q8_0 (gpu->metal, weights_of (gpu, tensor),
                         (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[0], x,
                         y, options);
}

/* Return options that normalize the input with NORM before the
   multiply.  */
static struct gip_metal_matvec_options
normalized (const struct gip_lfm2_metal *gpu,
            const struct gip_gguf_tensor *norm)
{
  struct gip_metal_matvec_options options
      = { .norm_weight = weights_of (gpu, norm), .eps = gpu->model->norm_eps };
  return options;
}

/* Options that add the product to the residual stream.  */
static const struct gip_metal_matvec_options add_to_residual
    = { .accumulate = 1 };

/* Record the gated short convolution of LAYER on the residual stream and
   add its output back.  */
static void
record_conv (struct gip_lfm2_metal *gpu, const struct gip_lfm2_layer *layer)
{
  const struct gip_lfm2_model *model = gpu->model;
  size_t n_embd = model->n_embd;
  size_t history
      = (size_t)layer->cache_index * (model->conv_kernel - 1) * n_embd;
  struct gip_metal_view h = floats_at (gpu->hidden, 0);
  struct gip_metal_matvec_options norm = normalized (gpu, layer->attn_norm);

  matvec (gpu, layer->conv_in_proj, h, floats_at (gpu->bcx, 0), &norm);
  gip_metal_barrier (gpu->metal);
  gip_metal_short_conv (
      gpu->metal, floats_at (gpu->bcx, 0), weights_of (gpu, layer->conv),
      floats_at (gpu->conv_state, history), floats_at (gpu->conv_out, 0),
      (uint32_t)n_embd, model->conv_kernel, 1);
  gip_metal_barrier (gpu->metal);
  matvec (gpu, layer->conv_out_proj, floats_at (gpu->conv_out, 0), h,
          &add_to_residual);
  gip_metal_barrier (gpu->metal);
}

/* Return a view of the KV cache BUFFER at element INDEX, whose elements
   are half precision when GPU's cache is.  */
static struct gip_metal_view
cache_at (const struct gip_lfm2_metal *gpu, struct gip_metal_buffer *buffer,
          size_t index)
{
  return gip_metal_at (buffer, index * (gpu->kv_half ? 2 : sizeof (float)));
}

/* Record the attention of LAYER at position POS on the residual stream
   and add its output back.  The new key goes into the cache through the
   norm and rotation, and the new value goes into the cache directly or
   through a conversion to half precision.  */
static void
record_attention (struct gip_lfm2_metal *gpu,
                  const struct gip_lfm2_layer *layer, uint32_t pos)
{
  const struct gip_lfm2_model *model = gpu->model;
  struct gip_metal *metal = gpu->metal;
  size_t kv_dim = (size_t)model->n_kv_heads * model->head_dim;
  size_t layer_base = (size_t)layer->cache_index * gpu->n_ctx * kv_dim;
  struct gip_metal_view k_slot
      = cache_at (gpu, gpu->k_cache, layer_base + (size_t)pos * kv_dim);
  struct gip_metal_view v_slot
      = cache_at (gpu, gpu->v_cache, layer_base + (size_t)pos * kv_dim);
  struct gip_metal_view h = floats_at (gpu->hidden, 0);
  struct gip_metal_view q = floats_at (gpu->q, 0);
  struct gip_metal_view k = floats_at (gpu->k, 0);
  struct gip_metal_view v = floats_at (gpu->v, 0);
  struct gip_metal_matvec_options norm = normalized (gpu, layer->attn_norm);

  /* The three projections read the same input and write different
     buffers, so they run together.  So do the two rotations and the
     value conversion that follow.  */
  matvec (gpu, layer->attn_q, h, q, &norm);
  matvec (gpu, layer->attn_k, h, k, &norm);
  matvec (gpu, layer->attn_v, h, gpu->kv_half ? v : v_slot, &norm);
  gip_metal_barrier (metal);
  gip_metal_norm_rope (metal, q, q, 0, weights_of (gpu, layer->attn_q_norm),
                       model->n_heads, model->head_dim, pos, 1, 0, 0,
                       model->rope_theta, model->norm_eps);
  gip_metal_norm_rope (metal, k, k_slot, gpu->kv_half,
                       weights_of (gpu, layer->attn_k_norm), model->n_kv_heads,
                       model->head_dim, pos, 1, 0, 0, model->rope_theta,
                       model->norm_eps);
  if (gpu->kv_half)
    gip_metal_convert_half (metal, v, v_slot, (uint32_t)kv_dim);
  gip_metal_barrier (metal);
  gip_metal_attention (metal, q, cache_at (gpu, gpu->k_cache, layer_base),
                       cache_at (gpu, gpu->v_cache, layer_base), gpu->kv_half,
                       floats_at (gpu->scores, 0), floats_at (gpu->attn, 0),
                       model->n_heads, model->n_kv_heads, model->head_dim, pos,
                       1, gpu->n_ctx);
  gip_metal_barrier (metal);
  matvec (gpu, layer->attn_output, floats_at (gpu->attn, 0), h,
          &add_to_residual);
  gip_metal_barrier (metal);
}

/* Record one forward pass of GPU's model on the token at TOKENS[POS]:
   the embedding lookup, every layer, and, when WANT_LOGITS is nonzero,
   the output matrix into the logits buffer.  When TRACE is not null,
   also record copies of the activations its members ask for.  */
static void
record_step (struct gip_lfm2_metal *gpu, uint32_t pos, int want_logits,
             const struct gip_lfm2_trace *trace)
{
  const struct gip_lfm2_model *model = gpu->model;
  struct gip_metal *metal = gpu->metal;
  uint32_t n_embd = model->n_embd;
  struct gip_metal_view h = floats_at (gpu->hidden, 0);
  struct gip_metal_view ffn = floats_at (gpu->ffn, 0);

  gip_metal_embed_q8_0 (metal, weights_of (gpu, model->token_embd),
                        gip_metal_at (gpu->tokens, pos * sizeof (int32_t)), h,
                        n_embd, 1);
  gip_metal_barrier (metal);

  /* Trace copies only read the residual stream, so they run alongside
     the next launches that read it.  Every later write to the stream
     comes after a barrier.  */
  if (trace != NULL && trace->embedding != NULL)
    gip_metal_copy (metal, h, floats_at (gpu->trace_embedding, 0), n_embd);

  for (uint32_t il = 0; il < model->n_layers; il++)
    {
      const struct gip_lfm2_layer *layer = &model->layers[il];

      if (layer->is_attention)
        record_attention (gpu, layer, pos);
      else
        record_conv (gpu, layer);

      gip_metal_matvec_q8_0_swiglu (
          metal, weights_of (gpu, layer->ffn_gate),
          weights_of (gpu, layer->ffn_up), model->n_ff, n_embd, h,
          weights_of (gpu, layer->ffn_norm), model->norm_eps, ffn);
      gip_metal_barrier (metal);
      matvec (gpu, layer->ffn_down, ffn, h, &add_to_residual);
      gip_metal_barrier (metal);

      if (trace != NULL && trace->layers != NULL)
        gip_metal_copy (metal, h,
                        floats_at (gpu->trace_layers, (size_t)il * n_embd),
                        n_embd);
    }

  /* The output matrix normalizes the last hidden state itself.  Only a
     trace needs the normalized vector on its own.  */
  if (trace != NULL && trace->final_norm != NULL)
    gip_metal_rms_norm (metal, h, weights_of (gpu, model->output_norm),
                        floats_at (gpu->trace_final, 0), n_embd, 1,
                        model->norm_eps);
  if (want_logits)
    {
      struct gip_metal_matvec_options norm
          = normalized (gpu, model->output_norm);
      matvec (gpu, model->output, h, floats_at (gpu->logits, 0), &norm);
    }
}

enum gip_status
gip_lfm2_metal_step (struct gip_lfm2_metal *gpu, int32_t token, float *logits,
                     const struct gip_lfm2_trace *trace)
{
  const struct gip_lfm2_model *model = gpu->model;
  uint32_t n_embd = model->n_embd;
  uint32_t pos = gpu->n_past;
  enum gip_status status;

  if (pos >= gpu->n_ctx || token < 0 || (uint32_t)token >= model->n_vocab)
    return GIP_ERR_ARGUMENT;

  /* Nothing is in flight between steps, so the CPU can write the token
     straight into the shared buffer.  */
  int32_t *tokens = gip_metal_buffer_contents (gpu->tokens);
  tokens[pos] = token;

  double encode_start = now_seconds ();
  status = gip_metal_begin (gpu->metal);
  if (status != GIP_OK)
    return status;
  record_step (gpu, pos, logits != NULL, trace);
  gpu->last_encode_seconds = now_seconds () - encode_start;
  status = gip_metal_end (gpu->metal, &gpu->last_gpu_seconds);
  if (status != GIP_OK)
    return status;

  if (logits != NULL)
    memcpy (logits, gip_metal_buffer_contents (gpu->logits),
            (size_t)model->n_vocab * sizeof (float));
  if (trace != NULL && trace->embedding != NULL)
    memcpy (trace->embedding, gip_metal_buffer_contents (gpu->trace_embedding),
            n_embd * sizeof (float));
  if (trace != NULL && trace->layers != NULL)
    memcpy (trace->layers, gip_metal_buffer_contents (gpu->trace_layers),
            (size_t)model->n_layers * n_embd * sizeof (float));
  if (trace != NULL && trace->final_norm != NULL)
    memcpy (trace->final_norm, gip_metal_buffer_contents (gpu->trace_final),
            n_embd * sizeof (float));

  gpu->n_past++;
  return GIP_OK;
}

enum gip_status
gip_lfm2_metal_generate (struct gip_lfm2_metal *gpu, uint32_t n, int32_t *out)
{
  const struct gip_lfm2_model *model = gpu->model;
  uint64_t tickets[GENERATE_IN_FLIGHT] = { 0 };
  enum gip_status status = GIP_OK;
  double gpu_seconds = 0.0;
  double encode_seconds = 0.0;

  if (gpu->n_past == 0 || n > gpu->n_ctx - gpu->n_past)
    return GIP_ERR_ARGUMENT;

  for (uint32_t i = 0; i < n && status == GIP_OK; i++)
    {
      uint32_t pos = gpu->n_past + i;

      /* Bound the steps in flight, so the CPU runs only a few steps ahead
         of the GPU.  */
      uint64_t *slot = &tickets[i % GENERATE_IN_FLIGHT];
      if (*slot != 0)
        status = gip_metal_wait (gpu->metal, *slot, &gpu_seconds);
      if (status != GIP_OK)
        break;

      double encode_start = now_seconds ();
      status = gip_metal_begin (gpu->metal);
      if (status != GIP_OK)
        break;
      gip_metal_argmax (gpu->metal, floats_at (gpu->logits, 0),
                        gip_metal_at (gpu->tokens, pos * sizeof (int32_t)),
                        model->n_vocab);
      gip_metal_barrier (gpu->metal);
      record_step (gpu, pos, 1, NULL);
      status = gip_metal_commit (gpu->metal, slot);
      encode_seconds += now_seconds () - encode_start;
    }

  /* Waiting on the newest command buffer waits on all the others.  */
  uint64_t last = 0;
  for (size_t s = 0; s < GENERATE_IN_FLIGHT; s++)
    if (tickets[s] > last)
      last = tickets[s];
  if (last != 0)
    {
      enum gip_status wait_status
          = gip_metal_wait (gpu->metal, last, &gpu_seconds);
      if (status == GIP_OK)
        status = wait_status;
    }
  if (status != GIP_OK)
    return status;

  const int32_t *tokens = gip_metal_buffer_contents (gpu->tokens);
  memcpy (out, tokens + gpu->n_past, n * sizeof (int32_t));
  gpu->n_past += n;
  gpu->last_encode_seconds = encode_seconds;
  gpu->last_gpu_seconds = gpu_seconds;
  return GIP_OK;
}

/* Grow GPU's trace buffers to hold ROWS tokens.  */
static enum gip_status
ensure_trace_rows (struct gip_lfm2_metal *gpu, uint32_t rows)
{
  const struct gip_lfm2_model *model = gpu->model;
  size_t n_embd = model->n_embd;

  if (rows <= gpu->trace_rows)
    return GIP_OK;
  gip_metal_buffer_free (gpu->trace_embedding);
  gip_metal_buffer_free (gpu->trace_layers);
  gip_metal_buffer_free (gpu->trace_final);
  gpu->trace_embedding = new_floats (gpu, rows * n_embd);
  gpu->trace_layers
      = new_floats (gpu, (size_t)model->n_layers * rows * n_embd);
  gpu->trace_final = new_floats (gpu, rows * n_embd);
  gpu->trace_rows = rows;
  if (gpu->trace_embedding == NULL || gpu->trace_layers == NULL
      || gpu->trace_final == NULL)
    return GIP_ERR_NOMEM;
  return GIP_OK;
}

/* Record a multiply of the matrix TENSOR by each of the N tokens at X,
   combined with Y as STORE says.  */
static void
matmul (struct gip_lfm2_metal *gpu, const struct gip_gguf_tensor *tensor,
        struct gip_metal_view x, struct gip_metal_view y, uint32_t n,
        enum gip_metal_store store)
{
  gip_metal_matmul_q8_0 (gpu->metal, weights_of (gpu, tensor),
                         (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[0], x,
                         y, n, store);
}

/* Record the gated short convolution of LAYER over the N normalized
   tokens of the batch and add its output to the residual stream.  */
static void
record_conv_batch (struct gip_lfm2_metal *gpu,
                   const struct gip_lfm2_layer *layer, uint32_t n)
{
  const struct gip_lfm2_model *model = gpu->model;
  struct gip_metal *metal = gpu->metal;
  size_t n_embd = model->n_embd;
  size_t history
      = (size_t)layer->cache_index * (model->conv_kernel - 1) * n_embd;

  matmul (gpu, layer->conv_in_proj, floats_at (gpu->normed, 0),
          floats_at (gpu->bcx, 0), n, GIP_METAL_OVERWRITE);
  gip_metal_barrier (metal);
  gip_metal_short_conv_batch (
      metal, floats_at (gpu->bcx, 0), weights_of (gpu, layer->conv),
      floats_at (gpu->conv_state, history), floats_at (gpu->conv_out, 0),
      (uint32_t)n_embd, model->conv_kernel, n);
  gip_metal_barrier (metal);
  matmul (gpu, layer->conv_out_proj, floats_at (gpu->conv_out, 0),
          floats_at (gpu->hidden, 0), n, GIP_METAL_ACCUMULATE);
  gip_metal_barrier (metal);
}

/* Record the causal attention of LAYER over the N normalized tokens of
   the batch, which sit at positions POS through POS + N - 1, and add
   its output to the residual stream.  The batch's keys and values go
   into the caches first, so each token attends over the earlier tokens
   of its own batch too.  */
static void
record_attention_batch (struct gip_lfm2_metal *gpu,
                        const struct gip_lfm2_layer *layer, uint32_t pos,
                        uint32_t n)
{
  const struct gip_lfm2_model *model = gpu->model;
  struct gip_metal *metal = gpu->metal;
  uint32_t kv_dim = model->n_kv_heads * model->head_dim;
  uint32_t q_dim = model->n_heads * model->head_dim;
  size_t layer_base = (size_t)layer->cache_index * gpu->n_ctx * kv_dim;
  struct gip_metal_view k_rows
      = cache_at (gpu, gpu->k_cache, layer_base + (size_t)pos * kv_dim);
  struct gip_metal_view v_rows
      = cache_at (gpu, gpu->v_cache, layer_base + (size_t)pos * kv_dim);
  struct gip_metal_view normed = floats_at (gpu->normed, 0);
  struct gip_metal_view q = floats_at (gpu->q, 0);
  struct gip_metal_view k = floats_at (gpu->k, 0);
  struct gip_metal_view v = floats_at (gpu->v, 0);

  matmul (gpu, layer->attn_q, normed, q, n, GIP_METAL_OVERWRITE);
  matmul (gpu, layer->attn_k, normed, k, n, GIP_METAL_OVERWRITE);
  matmul (gpu, layer->attn_v, normed, gpu->kv_half ? v : v_rows, n,
          GIP_METAL_OVERWRITE);
  gip_metal_barrier (metal);
  gip_metal_norm_rope (metal, q, q, 0, weights_of (gpu, layer->attn_q_norm),
                       model->n_heads, model->head_dim, pos, n, q_dim, q_dim,
                       model->rope_theta, model->norm_eps);
  gip_metal_norm_rope (metal, k, k_rows, gpu->kv_half,
                       weights_of (gpu, layer->attn_k_norm), model->n_kv_heads,
                       model->head_dim, pos, n, kv_dim, kv_dim,
                       model->rope_theta, model->norm_eps);
  if (gpu->kv_half)
    gip_metal_convert_half (metal, v, v_rows, n * kv_dim);
  gip_metal_barrier (metal);
  gip_metal_attention (metal, q, cache_at (gpu, gpu->k_cache, layer_base),
                       cache_at (gpu, gpu->v_cache, layer_base), gpu->kv_half,
                       floats_at (gpu->scores, 0), floats_at (gpu->attn, 0),
                       model->n_heads, model->n_kv_heads, model->head_dim, pos,
                       n, gpu->n_ctx);
  gip_metal_barrier (metal);
  matmul (gpu, layer->attn_output, floats_at (gpu->attn, 0),
          floats_at (gpu->hidden, 0), n, GIP_METAL_ACCUMULATE);
  gip_metal_barrier (metal);
}

/* Record the forward pass of the N tokens at TOKENS[POS] onward, a batch
   of at most GPU->batch.  When WANT_LOGITS is nonzero, the last token's
   logits go to the logits buffer.  When TRACED is nonzero, the trace
   buffers receive the batch's activations, with layer IL's outputs
   starting TRACE_LAYER_STRIDE floats after layer IL - 1's.  */
static void
record_batch (struct gip_lfm2_metal *gpu, uint32_t pos, uint32_t n,
              int want_logits, int traced, size_t trace_layer_stride)
{
  const struct gip_lfm2_model *model = gpu->model;
  struct gip_metal *metal = gpu->metal;
  uint32_t n_embd = model->n_embd;
  struct gip_metal_view h = floats_at (gpu->hidden, 0);
  struct gip_metal_view normed = floats_at (gpu->normed, 0);

  gip_metal_embed_q8_0 (metal, weights_of (gpu, model->token_embd),
                        gip_metal_at (gpu->tokens, pos * sizeof (int32_t)), h,
                        n_embd, n);
  gip_metal_barrier (metal);
  if (traced)
    gip_metal_copy (metal, h, floats_at (gpu->trace_embedding, 0), n * n_embd);

  for (uint32_t il = 0; il < model->n_layers; il++)
    {
      const struct gip_lfm2_layer *layer = &model->layers[il];

      gip_metal_rms_norm (metal, h, weights_of (gpu, layer->attn_norm), normed,
                          n_embd, n, model->norm_eps);
      gip_metal_barrier (metal);
      if (layer->is_attention)
        record_attention_batch (gpu, layer, pos, n);
      else
        record_conv_batch (gpu, layer, n);

      gip_metal_rms_norm (metal, h, weights_of (gpu, layer->ffn_norm), normed,
                          n_embd, n, model->norm_eps);
      gip_metal_barrier (metal);
      /* The up projection's store combines with the gate projection
         already in the buffer, which applies the SwiGLU.  */
      matmul (gpu, layer->ffn_gate, normed, floats_at (gpu->ffn, 0), n,
              GIP_METAL_OVERWRITE);
      gip_metal_barrier (metal);
      matmul (gpu, layer->ffn_up, normed, floats_at (gpu->ffn, 0), n,
              GIP_METAL_SWIGLU);
      gip_metal_barrier (metal);
      matmul (gpu, layer->ffn_down, floats_at (gpu->ffn, 0), h, n,
              GIP_METAL_ACCUMULATE);
      gip_metal_barrier (metal);

      if (traced)
        gip_metal_copy (metal, h,
                        floats_at (gpu->trace_layers, il * trace_layer_stride),
                        n * n_embd);
    }

  if (traced)
    gip_metal_rms_norm (metal, h, weights_of (gpu, model->output_norm),
                        floats_at (gpu->trace_final, 0), n_embd, n,
                        model->norm_eps);
  if (want_logits)
    {
      struct gip_metal_matvec_options norm
          = normalized (gpu, model->output_norm);
      matvec (gpu, model->output, floats_at (gpu->hidden, (n - 1) * n_embd),
              floats_at (gpu->logits, 0), &norm);
    }
}

enum gip_status
gip_lfm2_metal_prefill (struct gip_lfm2_metal *gpu, const int32_t *tokens,
                        uint32_t n, float *logits,
                        const struct gip_lfm2_trace *trace)
{
  const struct gip_lfm2_model *model = gpu->model;
  size_t n_embd = model->n_embd;
  enum gip_status status = GIP_OK;
  double gpu_seconds = 0.0;
  double encode_seconds = 0.0;

  if (n == 0 || n > gpu->n_ctx - gpu->n_past)
    return GIP_ERR_ARGUMENT;
  for (uint32_t i = 0; i < n; i++)
    if (tokens[i] < 0 || (uint32_t)tokens[i] >= model->n_vocab)
      return GIP_ERR_ARGUMENT;

  /* Nothing is in flight, so every prompt token goes into the shared
     buffer before the first batch.  */
  int32_t *token_buffer = gip_metal_buffer_contents (gpu->tokens);
  memcpy (token_buffer + gpu->n_past, tokens, n * sizeof (int32_t));

  if (trace != NULL)
    {
      status = ensure_trace_rows (gpu, gpu->batch);
      if (status != GIP_OK)
        return status;
    }

  for (uint32_t done = 0; done < n && status == GIP_OK;)
    {
      uint32_t count = n - done < gpu->batch ? n - done : gpu->batch;
      uint32_t pos = gpu->n_past + done;
      int last = done + count == n;
      uint64_t ticket;

      double encode_start = now_seconds ();
      status = gip_metal_begin (gpu->metal);
      if (status != GIP_OK)
        break;
      record_batch (gpu, pos, count, last, trace != NULL,
                    (size_t)gpu->batch * n_embd);
      status = gip_metal_commit (gpu->metal, &ticket);
      encode_seconds += now_seconds () - encode_start;

      /* A traced batch waits so its activations can be copied out before
         the next batch overwrites them.  Other batches run back to
         back.  */
      if (status == GIP_OK && (trace != NULL || last))
        status = gip_metal_wait (gpu->metal, ticket, &gpu_seconds);
      if (status == GIP_OK && trace != NULL)
        {
          size_t rows = (size_t)count * n_embd;
          if (trace->embedding != NULL)
            memcpy (trace->embedding + done * n_embd,
                    gip_metal_buffer_contents (gpu->trace_embedding),
                    rows * sizeof (float));
          if (trace->final_norm != NULL)
            memcpy (trace->final_norm + done * n_embd,
                    gip_metal_buffer_contents (gpu->trace_final),
                    rows * sizeof (float));
          if (trace->layers != NULL)
            {
              const float *layers
                  = gip_metal_buffer_contents (gpu->trace_layers);
              for (uint32_t il = 0; il < model->n_layers; il++)
                memcpy (trace->layers + ((size_t)il * n + done) * n_embd,
                        layers + (size_t)il * gpu->batch * n_embd,
                        rows * sizeof (float));
            }
        }
      done += count;
    }
  if (status != GIP_OK)
    return status;

  if (logits != NULL)
    memcpy (logits, gip_metal_buffer_contents (gpu->logits),
            (size_t)model->n_vocab * sizeof (float));
  gpu->n_past += n;
  gpu->last_encode_seconds = encode_seconds;
  gpu->last_gpu_seconds = gpu_seconds;
  return GIP_OK;
}
