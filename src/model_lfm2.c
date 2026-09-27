/* model_lfm2.c loads Liquid AI's LFM2 models from GGUF files and runs
   their forward pass one token at a time with the scalar ops.

   Each LFM2 layer normalizes its input, applies either a gated short
   convolution or grouped-query attention, adds the result to the
   residual stream, and then applies a SwiGLU feed-forward block the
   same way.  The last hidden state is normalized and multiplied by the
   output matrix, which LFM2 ties to the token embeddings.  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "kernels/scalar.h"
#include "model_lfm2.h"

enum
{
  NAME_SIZE = 96,
  BUFFER_ALIGNMENT = 64
};

/* Report whether TYPE is a matrix type the scalar ops read.  */
static bool
is_matrix_type (enum gip_tensor_type type)
{
  return type == GIP_TENSOR_F32 || type == GIP_TENSOR_F16
         || type == GIP_TENSOR_BF16 || type == GIP_TENSOR_Q8_0;
}

/* Find the tensor NAME in MODEL and store it in OUT.  The tensor must
   have N_DIMS dimensions of sizes NE0 and NE1.  A vector must be F32,
   and a matrix must have a type the scalar ops read.  */
static enum gip_status
require_tensor (const struct gip_lfm2_model *model, const char *name,
                uint32_t n_dims, uint64_t ne0, uint64_t ne1,
                const struct gip_gguf_tensor **out, char *err, size_t err_size)
{
  const struct gip_gguf_tensor *tensor
      = gip_gguf_find_tensor (&model->gguf, name);

  if (tensor == NULL)
    {
      gip_format_error (err, err_size, "missing tensor %s", name);
      return GIP_ERR_FORMAT;
    }
  if (tensor->n_dims != n_dims || tensor->ne[0] != ne0
      || (n_dims == 2 && tensor->ne[1] != ne1))
    {
      gip_format_error (err, err_size,
                        "tensor %s has shape [%llu, %llu], expected "
                        "[%llu, %llu]",
                        name, (unsigned long long)tensor->ne[0],
                        (unsigned long long)tensor->ne[1],
                        (unsigned long long)ne0, (unsigned long long)ne1);
      return GIP_ERR_FORMAT;
    }
  if ((n_dims == 1 && tensor->type != GIP_TENSOR_F32)
      || (n_dims == 2 && !is_matrix_type (tensor->type)))
    {
      gip_format_error (err, err_size, "tensor %s has unsupported type %d",
                        name, (int)tensor->type);
      return GIP_ERR_UNSUPPORTED;
    }
  *out = tensor;
  return GIP_OK;
}

/* Store the integer metadata value KEY of MODEL in OUT.  */
static enum gip_status
require_u32 (const struct gip_lfm2_model *model, const char *key,
             uint32_t *out, char *err, size_t err_size)
{
  const struct gip_gguf_kv *kv = gip_gguf_find_kv (&model->gguf, key);

  if (kv == NULL || !gip_gguf_kv_u32 (kv, out))
    {
      gip_format_error (err, err_size, "missing or invalid %s", key);
      return GIP_ERR_FORMAT;
    }
  return GIP_OK;
}

/* Store the floating-point metadata value KEY of MODEL in OUT.  */
static enum gip_status
require_f32 (const struct gip_lfm2_model *model, const char *key, float *out,
             char *err, size_t err_size)
{
  const struct gip_gguf_kv *kv = gip_gguf_find_kv (&model->gguf, key);

  if (kv == NULL || !gip_gguf_kv_f32 (kv, out))
    {
      gip_format_error (err, err_size, "missing or invalid %s", key);
      return GIP_ERR_FORMAT;
    }
  return GIP_OK;
}

/* Store the number of KV heads of layer LAYER of MODEL in OUT.  The
   file holds one count per layer, and a convolution layer has a count
   of zero.  A single count applies to every layer.  */
static enum gip_status
layer_kv_heads (const struct gip_lfm2_model *model, uint32_t layer,
                uint32_t *out, char *err, size_t err_size)
{
  const char *key = "lfm2.attention.head_count_kv";
  const struct gip_gguf_kv *kv = gip_gguf_find_kv (&model->gguf, key);

  if (kv != NULL && kv->type == GIP_GGUF_ARRAY
      && kv->array_count == model->n_layers
      && gip_gguf_kv_array_u32 (kv, layer, out))
    return GIP_OK;
  if (kv != NULL && gip_gguf_kv_u32 (kv, out))
    return GIP_OK;
  gip_format_error (err, err_size, "missing or invalid %s", key);
  return GIP_ERR_FORMAT;
}

/* Read the hyperparameters of MODEL from its metadata.  */
static enum gip_status
load_hparams (struct gip_lfm2_model *model, char *err, size_t err_size)
{
  const struct gip_gguf_kv *arch
      = gip_gguf_find_kv (&model->gguf, "general.architecture");
  struct gip_gguf_string name;
  enum gip_status status;

  if (arch == NULL || !gip_gguf_kv_string (arch, &name))
    {
      gip_format_error (err, err_size, "missing general.architecture");
      return GIP_ERR_FORMAT;
    }
  if (!gip_gguf_string_equals (name, "lfm2"))
    {
      gip_format_error (err, err_size, "architecture %.*s is not lfm2",
                        (int)name.length, name.data);
      return GIP_ERR_UNSUPPORTED;
    }

  status = require_u32 (model, "lfm2.block_count", &model->n_layers, err,
                        err_size);
  if (status == GIP_OK)
    status = require_u32 (model, "lfm2.embedding_length", &model->n_embd, err,
                          err_size);
  if (status == GIP_OK)
    status = require_u32 (model, "lfm2.feed_forward_length", &model->n_ff, err,
                          err_size);
  if (status == GIP_OK)
    status = require_u32 (model, "lfm2.attention.head_count", &model->n_heads,
                          err, err_size);
  if (status == GIP_OK)
    status = require_u32 (model, "lfm2.shortconv.l_cache", &model->conv_kernel,
                          err, err_size);
  if (status == GIP_OK)
    status = require_f32 (model, "lfm2.rope.freq_base", &model->rope_theta,
                          err, err_size);
  if (status == GIP_OK)
    status = require_f32 (model, "lfm2.attention.layer_norm_rms_epsilon",
                          &model->norm_eps, err, err_size);
  if (status != GIP_OK)
    return status;

  if (model->n_layers == 0 || model->n_embd == 0 || model->n_ff == 0
      || model->n_heads == 0 || model->n_embd % model->n_heads != 0
      || model->conv_kernel < 2)
    {
      gip_format_error (err, err_size, "inconsistent hyperparameters");
      return GIP_ERR_FORMAT;
    }
  model->head_dim = model->n_embd / model->n_heads;
  if (model->head_dim % 2 != 0)
    {
      gip_format_error (err, err_size, "head size %u is odd", model->head_dim);
      return GIP_ERR_FORMAT;
    }
  return GIP_OK;
}

/* Find and check the tensors of layer IL of MODEL.  */
static enum gip_status
load_layer (struct gip_lfm2_model *model, uint32_t il, char *err,
            size_t err_size)
{
  struct gip_lfm2_layer *layer = &model->layers[il];
  uint32_t n_embd = model->n_embd;
  uint32_t n_kv_heads;
  char name[NAME_SIZE];
  enum gip_status status;

  status = layer_kv_heads (model, il, &n_kv_heads, err, err_size);
  if (status != GIP_OK)
    return status;
  layer->is_attention = n_kv_heads != 0;

  snprintf (name, sizeof name, "blk.%u.attn_norm.weight", il);
  status = require_tensor (model, name, 1, n_embd, 1, &layer->attn_norm, err,
                           err_size);
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.ffn_norm.weight", il);
      status = require_tensor (model, name, 1, n_embd, 1, &layer->ffn_norm,
                               err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.ffn_gate.weight", il);
      status = require_tensor (model, name, 2, n_embd, model->n_ff,
                               &layer->ffn_gate, err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.ffn_up.weight", il);
      status = require_tensor (model, name, 2, n_embd, model->n_ff,
                               &layer->ffn_up, err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.ffn_down.weight", il);
      status = require_tensor (model, name, 2, model->n_ff, n_embd,
                               &layer->ffn_down, err, err_size);
    }
  if (status != GIP_OK)
    return status;

  if (!layer->is_attention)
    {
      layer->cache_index = model->n_conv_layers++;
      snprintf (name, sizeof name, "blk.%u.shortconv.conv.weight", il);
      status = require_tensor (model, name, 2, model->conv_kernel, n_embd,
                               &layer->conv, err, err_size);
      if (status == GIP_OK && layer->conv->type != GIP_TENSOR_F32)
        {
          gip_format_error (err, err_size, "tensor %s must be F32", name);
          status = GIP_ERR_UNSUPPORTED;
        }
      if (status == GIP_OK)
        {
          snprintf (name, sizeof name, "blk.%u.shortconv.in_proj.weight", il);
          status = require_tensor (model, name, 2, n_embd, 3 * n_embd,
                                   &layer->conv_in_proj, err, err_size);
        }
      if (status == GIP_OK)
        {
          snprintf (name, sizeof name, "blk.%u.shortconv.out_proj.weight", il);
          status = require_tensor (model, name, 2, n_embd, n_embd,
                                   &layer->conv_out_proj, err, err_size);
        }
      return status;
    }

  if (model->n_kv_heads == 0)
    model->n_kv_heads = n_kv_heads;
  if (n_kv_heads != model->n_kv_heads || model->n_heads % n_kv_heads != 0)
    {
      gip_format_error (err, err_size, "layer %u has %u KV heads", il,
                        n_kv_heads);
      return GIP_ERR_UNSUPPORTED;
    }
  layer->cache_index = model->n_attention_layers++;

  uint32_t q_dim = model->n_heads * model->head_dim;
  uint32_t kv_dim = n_kv_heads * model->head_dim;
  snprintf (name, sizeof name, "blk.%u.attn_q.weight", il);
  status = require_tensor (model, name, 2, n_embd, q_dim, &layer->attn_q, err,
                           err_size);
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.attn_k.weight", il);
      status = require_tensor (model, name, 2, n_embd, kv_dim, &layer->attn_k,
                               err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.attn_v.weight", il);
      status = require_tensor (model, name, 2, n_embd, kv_dim, &layer->attn_v,
                               err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.attn_output.weight", il);
      status = require_tensor (model, name, 2, q_dim, n_embd,
                               &layer->attn_output, err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.attn_q_norm.weight", il);
      status = require_tensor (model, name, 1, model->head_dim, 1,
                               &layer->attn_q_norm, err, err_size);
    }
  if (status == GIP_OK)
    {
      snprintf (name, sizeof name, "blk.%u.attn_k_norm.weight", il);
      status = require_tensor (model, name, 1, model->head_dim, 1,
                               &layer->attn_k_norm, err, err_size);
    }
  return status;
}

enum gip_status
gip_lfm2_load (const char *path, struct gip_lfm2_model *model, char *err,
               size_t err_size)
{
  enum gip_status status;

  memset (model, 0, sizeof *model);
  status = gip_gguf_open (path, &model->gguf, err, err_size);
  if (status != GIP_OK)
    return status;

  status = load_hparams (model, err, err_size);
  if (status == GIP_OK)
    {
      model->layers = calloc (model->n_layers, sizeof *model->layers);
      if (model->layers == NULL)
        status = GIP_ERR_NOMEM;
    }
  for (uint32_t il = 0; il < model->n_layers && status == GIP_OK; il++)
    status = load_layer (model, il, err, err_size);

  if (status == GIP_OK)
    {
      const struct gip_gguf_tensor *embd
          = gip_gguf_find_tensor (&model->gguf, "token_embd.weight");
      if (embd == NULL || embd->n_dims != 2)
        {
          gip_format_error (err, err_size, "missing tensor token_embd.weight");
          status = GIP_ERR_FORMAT;
        }
      else
        {
          model->n_vocab = (uint32_t)embd->ne[1];
          status = require_tensor (model, "token_embd.weight", 2,
                                   model->n_embd, model->n_vocab,
                                   &model->token_embd, err, err_size);
        }
    }
  if (status == GIP_OK)
    status = require_tensor (model, "token_embd_norm.weight", 1, model->n_embd,
                             1, &model->output_norm, err, err_size);
  if (status == GIP_OK)
    {
      /* LFM2 ties the output matrix to the token embeddings.  A file with
         its own output matrix uses that matrix instead.  */
      if (gip_gguf_find_tensor (&model->gguf, "output.weight") != NULL)
        status
            = require_tensor (model, "output.weight", 2, model->n_embd,
                              model->n_vocab, &model->output, err, err_size);
      else
        model->output = model->token_embd;
    }

  if (status != GIP_OK)
    {
      if (status == GIP_ERR_NOMEM)
        gip_format_error (err, err_size, "%s: out of memory", path);
      gip_lfm2_free (model);
    }
  return status;
}

void
gip_lfm2_free (struct gip_lfm2_model *model)
{
  free (model->layers);
  gip_gguf_close (&model->gguf);
  memset (model, 0, sizeof *model);
}

/* Return a zeroed, 64-byte-aligned buffer of COUNT floats, or null.  */
static float *
alloc_floats (size_t count)
{
  size_t bytes;

  if (count == 0)
    count = 1;
  if (__builtin_mul_overflow (count, sizeof (float), &bytes)
      || __builtin_add_overflow (bytes, BUFFER_ALIGNMENT - 1, &bytes))
    return NULL;
  bytes &= ~(size_t)(BUFFER_ALIGNMENT - 1);

  float *buffer = aligned_alloc (BUFFER_ALIGNMENT, bytes);
  if (buffer != NULL)
    memset (buffer, 0, bytes);
  return buffer;
}

enum gip_status
gip_lfm2_state_init (const struct gip_lfm2_model *model, uint32_t n_ctx,
                     struct gip_lfm2_state *state)
{
  size_t kv_dim = (size_t)model->n_kv_heads * model->head_dim;
  size_t q_dim = (size_t)model->n_heads * model->head_dim;
  size_t cache_floats;
  size_t conv_floats;

  memset (state, 0, sizeof *state);
  if (n_ctx == 0)
    return GIP_ERR_ARGUMENT;
  if (__builtin_mul_overflow ((size_t)model->n_attention_layers * n_ctx,
                              kv_dim, &cache_floats)
      || __builtin_mul_overflow ((size_t)model->n_conv_layers
                                     * (model->conv_kernel - 1),
                                 (size_t)model->n_embd, &conv_floats))
    return GIP_ERR_NOMEM;

  state->n_ctx = n_ctx;
  state->k_cache = alloc_floats (cache_floats);
  state->v_cache = alloc_floats (cache_floats);
  state->conv_state = alloc_floats (conv_floats);
  state->hidden = alloc_floats (model->n_embd);
  state->normed = alloc_floats (model->n_embd);
  state->block_out = alloc_floats (model->n_embd);
  state->bcx = alloc_floats (3 * (size_t)model->n_embd);
  state->conv_out = alloc_floats (model->n_embd);
  state->q = alloc_floats (q_dim);
  state->k = alloc_floats (kv_dim);
  state->v = alloc_floats (kv_dim);
  state->attn = alloc_floats (q_dim);
  state->scores = alloc_floats (n_ctx);
  state->gate = alloc_floats (model->n_ff);
  state->up = alloc_floats (model->n_ff);

  if (state->k_cache == NULL || state->v_cache == NULL
      || state->conv_state == NULL || state->hidden == NULL
      || state->normed == NULL || state->block_out == NULL
      || state->bcx == NULL || state->conv_out == NULL || state->q == NULL
      || state->k == NULL || state->v == NULL || state->attn == NULL
      || state->scores == NULL || state->gate == NULL || state->up == NULL)
    {
      gip_lfm2_state_free (state);
      return GIP_ERR_NOMEM;
    }
  return GIP_OK;
}

void
gip_lfm2_state_free (struct gip_lfm2_state *state)
{
  free (state->k_cache);
  free (state->v_cache);
  free (state->conv_state);
  free (state->hidden);
  free (state->normed);
  free (state->block_out);
  free (state->bcx);
  free (state->conv_out);
  free (state->q);
  free (state->k);
  free (state->v);
  free (state->attn);
  free (state->scores);
  free (state->gate);
  free (state->up);
  memset (state, 0, sizeof *state);
}

/* Multiply the matrix TENSOR by X into Y.  The tensor's rows hold NE0
   elements and it has NE1 rows.  */
static void
matvec (const struct gip_gguf_tensor *tensor, const float *x, float *y)
{
  gip_matvec (tensor->type, tensor->data, tensor->ne[1], tensor->ne[0], x, y);
}

/* Run the gated short convolution of LAYER on STATE->normed and store
   the result in STATE->block_out.  */
static void
conv_block (const struct gip_lfm2_model *model,
            const struct gip_lfm2_layer *layer, struct gip_lfm2_state *state)
{
  size_t n_embd = model->n_embd;
  size_t kernel = model->conv_kernel;
  const float *weights = layer->conv->data;
  float *history
      = state->conv_state + (size_t)layer->cache_index * (kernel - 1) * n_embd;

  /* The input projection yields the gates B and C and the input X, in
     that order.  */
  matvec (layer->conv_in_proj, state->normed, state->bcx);
  const float *b = state->bcx;
  const float *c = state->bcx + n_embd;
  const float *x = state->bcx + 2 * n_embd;

  /* HISTORY holds B times X for the previous KERNEL - 1 tokens, oldest
     first.  Tap K of a channel multiplies the input from KERNEL - 1 - K
     tokens ago.  */
  for (size_t ch = 0; ch < n_embd; ch++)
    {
      const float *taps = weights + ch * kernel;
      float bx = b[ch] * x[ch];
      double sum = (double)taps[kernel - 1] * bx;
      for (size_t k = 0; k + 1 < kernel; k++)
        sum += (double)taps[k] * history[k * n_embd + ch];
      for (size_t k = 0; k + 2 < kernel; k++)
        history[k * n_embd + ch] = history[(k + 1) * n_embd + ch];
      history[(kernel - 2) * n_embd + ch] = bx;
      state->conv_out[ch] = c[ch] * (float)sum;
    }

  matvec (layer->conv_out_proj, state->conv_out, state->block_out);
}

/* Run the attention of LAYER at position POS on STATE->normed and store
   the result in STATE->block_out.  */
static void
attention_block (const struct gip_lfm2_model *model,
                 const struct gip_lfm2_layer *layer,
                 struct gip_lfm2_state *state, uint32_t pos)
{
  size_t head_dim = model->head_dim;
  size_t kv_dim = (size_t)model->n_kv_heads * head_dim;
  size_t group = model->n_heads / model->n_kv_heads;
  float scale = 1.0f / sqrtf ((float)head_dim);
  float *k_cache
      = state->k_cache + (size_t)layer->cache_index * state->n_ctx * kv_dim;
  float *v_cache
      = state->v_cache + (size_t)layer->cache_index * state->n_ctx * kv_dim;

  matvec (layer->attn_q, state->normed, state->q);
  matvec (layer->attn_k, state->normed, state->k);
  matvec (layer->attn_v, state->normed, state->v);

  /* LFM2 normalizes each query and key head before the rotation.  */
  for (size_t h = 0; h < model->n_heads; h++)
    {
      float *qh = state->q + h * head_dim;
      gip_rms_norm (qh, layer->attn_q_norm->data, head_dim, model->norm_eps,
                    qh);
      gip_rope_neox (qh, head_dim, pos, model->rope_theta);
    }
  for (size_t h = 0; h < model->n_kv_heads; h++)
    {
      float *kh = state->k + h * head_dim;
      gip_rms_norm (kh, layer->attn_k_norm->data, head_dim, model->norm_eps,
                    kh);
      gip_rope_neox (kh, head_dim, pos, model->rope_theta);
    }

  memcpy (k_cache + (size_t)pos * kv_dim, state->k, kv_dim * sizeof (float));
  memcpy (v_cache + (size_t)pos * kv_dim, state->v, kv_dim * sizeof (float));

  size_t n_keys = (size_t)pos + 1;
  for (size_t h = 0; h < model->n_heads; h++)
    {
      /* Query heads share KV heads in consecutive groups.  */
      size_t kv_head = h / group;
      const float *qh = state->q + h * head_dim;
      float *out = state->attn + h * head_dim;

      for (size_t t = 0; t < n_keys; t++)
        state->scores[t]
            = scale
              * gip_dot (qh, k_cache + t * kv_dim + kv_head * head_dim,
                         head_dim);
      gip_softmax (state->scores, n_keys);

      for (size_t d = 0; d < head_dim; d++)
        {
          double sum = 0.0;
          for (size_t t = 0; t < n_keys; t++)
            sum += (double)state->scores[t]
                   * v_cache[t * kv_dim + kv_head * head_dim + d];
          out[d] = (float)sum;
        }
    }

  matvec (layer->attn_output, state->attn, state->block_out);
}

/* Add the N floats at DELTA to the N floats at H.  */
static void
add_in_place (float *h, const float *delta, size_t n)
{
  for (size_t i = 0; i < n; i++)
    h[i] += delta[i];
}

enum gip_status
gip_lfm2_step (const struct gip_lfm2_model *model,
               struct gip_lfm2_state *state, int32_t token, float *logits,
               const struct gip_lfm2_trace *trace)
{
  size_t n_embd = model->n_embd;
  uint32_t pos = state->n_past;

  if (pos >= state->n_ctx || token < 0 || (uint32_t)token >= model->n_vocab)
    return GIP_ERR_ARGUMENT;

  gip_get_row (model->token_embd->type, model->token_embd->data, n_embd,
               (size_t)token, state->hidden);
  if (trace != NULL && trace->embedding != NULL)
    memcpy (trace->embedding, state->hidden, n_embd * sizeof (float));

  for (uint32_t il = 0; il < model->n_layers; il++)
    {
      const struct gip_lfm2_layer *layer = &model->layers[il];

      gip_rms_norm (state->hidden, layer->attn_norm->data, n_embd,
                    model->norm_eps, state->normed);
      if (layer->is_attention)
        attention_block (model, layer, state, pos);
      else
        conv_block (model, layer, state);
      add_in_place (state->hidden, state->block_out, n_embd);

      gip_rms_norm (state->hidden, layer->ffn_norm->data, n_embd,
                    model->norm_eps, state->normed);
      matvec (layer->ffn_gate, state->normed, state->gate);
      matvec (layer->ffn_up, state->normed, state->up);
      for (size_t i = 0; i < model->n_ff; i++)
        state->gate[i] = gip_silu (state->gate[i]) * state->up[i];
      matvec (layer->ffn_down, state->gate, state->block_out);
      add_in_place (state->hidden, state->block_out, n_embd);

      if (trace != NULL && trace->layers != NULL)
        memcpy (trace->layers + (size_t)il * n_embd, state->hidden,
                n_embd * sizeof (float));
    }

  gip_rms_norm (state->hidden, model->output_norm->data, n_embd,
                model->norm_eps, state->normed);
  if (trace != NULL && trace->final_norm != NULL)
    memcpy (trace->final_norm, state->normed, n_embd * sizeof (float));
  if (logits != NULL)
    matvec (model->output, state->normed, logits);

  state->n_past++;
  return GIP_OK;
}
