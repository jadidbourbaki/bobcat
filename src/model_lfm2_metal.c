/* model_lfm2_metal.c runs the LFM2 forward pass on the Metal GPU.  Each
   step records every layer into one command buffer and waits for it.
   The weights stay in the memory-mapped GGUF file, and the GPU reads
   them in place.  */

#include <string.h>
#include <time.h>

#include "error.h"
#include "kernels/scalar.h"
#include "model_lfm2_metal.h"

enum
{
  /* The attention kernel's limits from kernels.metal.  Each lane holds
     a whole number of elements of a head, and a threadgroup serves at
     most four query heads of up to 128 elements.  */
  SIMD_WIDTH = 32,
  ATTENTION_MAX_GROUP = 4,
  ATTENTION_MAX_HEAD_DIM = 128
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
                     struct gip_metal *metal, uint32_t n_ctx,
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

  gpu->model = model;
  gpu->metal = metal;
  gpu->n_ctx = n_ctx;
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

  size_t cache_floats = (size_t)model->n_attention_layers * n_ctx * kv_dim;
  size_t conv_floats
      = (size_t)model->n_conv_layers * (model->conv_kernel - 1) * n_embd;
  gpu->k_cache = new_floats (gpu, cache_floats);
  gpu->v_cache = new_floats (gpu, cache_floats);
  gpu->conv_state = new_floats (gpu, conv_floats);
  gpu->hidden = new_floats (gpu, n_embd);
  gpu->bcx = new_floats (gpu, 3 * n_embd);
  gpu->conv_out = new_floats (gpu, n_embd);
  gpu->q = new_floats (gpu, q_dim);
  gpu->attn = new_floats (gpu, q_dim);
  gpu->scores = new_floats (gpu, gip_metal_attention_scratch (
                                     model->n_heads, model->head_dim, n_ctx));
  gpu->ffn = new_floats (gpu, model->n_ff);
  gpu->logits = new_floats (gpu, model->n_vocab);
  gpu->trace_layers = new_floats (gpu, (size_t)model->n_layers * n_embd);
  gpu->trace_final = new_floats (gpu, n_embd);

  if (gpu->weights == NULL || gpu->k_cache == NULL || gpu->v_cache == NULL
      || gpu->conv_state == NULL || gpu->hidden == NULL || gpu->bcx == NULL
      || gpu->conv_out == NULL || gpu->q == NULL || gpu->attn == NULL
      || gpu->scores == NULL || gpu->ffn == NULL || gpu->logits == NULL
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
    gpu->weights,      gpu->k_cache,     gpu->v_cache,  gpu->conv_state,
    gpu->hidden,       gpu->bcx,         gpu->conv_out, gpu->q,
    gpu->attn,         gpu->scores,      gpu->ffn,      gpu->logits,
    gpu->trace_layers, gpu->trace_final,
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
  gip_metal_short_conv (
      gpu->metal, floats_at (gpu->bcx, 0), weights_of (gpu, layer->conv),
      floats_at (gpu->conv_state, history), floats_at (gpu->conv_out, 0),
      (uint32_t)n_embd, model->conv_kernel);
  matvec (gpu, layer->conv_out_proj, floats_at (gpu->conv_out, 0), h,
          &add_to_residual);
}

/* Record the attention of LAYER at position POS on the residual stream
   and add its output back.  The K and V projections write straight into
   the caches.  */
static void
record_attention (struct gip_lfm2_metal *gpu,
                  const struct gip_lfm2_layer *layer, uint32_t pos)
{
  const struct gip_lfm2_model *model = gpu->model;
  size_t kv_dim = (size_t)model->n_kv_heads * model->head_dim;
  size_t layer_base = (size_t)layer->cache_index * gpu->n_ctx * kv_dim;
  struct gip_metal_view k_slot
      = floats_at (gpu->k_cache, layer_base + (size_t)pos * kv_dim);
  struct gip_metal_view v_slot
      = floats_at (gpu->v_cache, layer_base + (size_t)pos * kv_dim);
  struct gip_metal_view h = floats_at (gpu->hidden, 0);
  struct gip_metal_matvec_options norm = normalized (gpu, layer->attn_norm);

  matvec (gpu, layer->attn_q, h, floats_at (gpu->q, 0), &norm);
  matvec (gpu, layer->attn_k, h, k_slot, &norm);
  matvec (gpu, layer->attn_v, h, v_slot, &norm);
  gip_metal_qk_norm_rope (gpu->metal, floats_at (gpu->q, 0),
                          weights_of (gpu, layer->attn_q_norm), model->n_heads,
                          model->head_dim, pos, model->rope_theta,
                          model->norm_eps);
  gip_metal_qk_norm_rope (gpu->metal, k_slot,
                          weights_of (gpu, layer->attn_k_norm),
                          model->n_kv_heads, model->head_dim, pos,
                          model->rope_theta, model->norm_eps);
  gip_metal_attention (
      gpu->metal, floats_at (gpu->q, 0), floats_at (gpu->k_cache, layer_base),
      floats_at (gpu->v_cache, layer_base), floats_at (gpu->scores, 0),
      floats_at (gpu->attn, 0), model->n_heads, model->n_kv_heads,
      model->head_dim, pos + 1, gpu->n_ctx);
  matvec (gpu, layer->attn_output, floats_at (gpu->attn, 0), h,
          &add_to_residual);
}

enum gip_status
gip_lfm2_metal_step (struct gip_lfm2_metal *gpu, int32_t token, float *logits,
                     const struct gip_lfm2_trace *trace)
{
  const struct gip_lfm2_model *model = gpu->model;
  struct gip_metal *metal = gpu->metal;
  uint32_t n_embd = model->n_embd;
  uint32_t pos = gpu->n_past;
  enum gip_status status;

  if (pos >= gpu->n_ctx || token < 0 || (uint32_t)token >= model->n_vocab)
    return GIP_ERR_ARGUMENT;

  /* One embedding row is cheap to dequantize on the CPU, and the buffer
     is shared with the GPU.  */
  float *hidden = gip_metal_buffer_contents (gpu->hidden);
  gip_get_row (model->token_embd->type, model->token_embd->data, n_embd,
               (size_t)token, hidden);
  if (trace != NULL && trace->embedding != NULL)
    memcpy (trace->embedding, hidden, n_embd * sizeof (float));

  double encode_start = now_seconds ();
  status = gip_metal_begin (metal);
  if (status != GIP_OK)
    return status;

  struct gip_metal_view h = floats_at (gpu->hidden, 0);
  struct gip_metal_view ffn = floats_at (gpu->ffn, 0);
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
      matvec (gpu, layer->ffn_down, ffn, h, &add_to_residual);

      if (trace != NULL && trace->layers != NULL)
        gip_metal_copy (metal, h,
                        floats_at (gpu->trace_layers, (size_t)il * n_embd),
                        n_embd);
    }

  /* The output matrix normalizes the last hidden state itself.  Only a
     trace needs the normalized vector on its own.  */
  if (trace != NULL && trace->final_norm != NULL)
    gip_metal_rms_norm (metal, h, weights_of (gpu, model->output_norm),
                        floats_at (gpu->trace_final, 0), n_embd,
                        model->norm_eps);
  if (logits != NULL)
    {
      struct gip_metal_matvec_options norm
          = normalized (gpu, model->output_norm);
      matvec (gpu, model->output, h, floats_at (gpu->logits, 0), &norm);
    }

  gpu->last_encode_seconds = now_seconds () - encode_start;
  status = gip_metal_end (metal, &gpu->last_gpu_seconds);
  if (status != GIP_OK)
    return status;

  if (logits != NULL)
    memcpy (logits, gip_metal_buffer_contents (gpu->logits),
            (size_t)model->n_vocab * sizeof (float));
  if (trace != NULL && trace->layers != NULL)
    memcpy (trace->layers, gip_metal_buffer_contents (gpu->trace_layers),
            (size_t)model->n_layers * n_embd * sizeof (float));
  if (trace != NULL && trace->final_norm != NULL)
    memcpy (trace->final_norm, gip_metal_buffer_contents (gpu->trace_final),
            n_embd * sizeof (float));

  gpu->n_past++;
  return GIP_OK;
}
