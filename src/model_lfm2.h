/* model_lfm2.h declares the LFM2 model and its scalar forward pass.  */

#ifndef GIP_MODEL_LFM2_H
#define GIP_MODEL_LFM2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

/* The tensors of one LFM2 layer.  An attention layer uses the ATTN
   tensors, and a convolution layer uses the CONV tensors.  CACHE_INDEX
   numbers the layer among layers of its own kind.  */
struct gip_lfm2_layer
{
  bool is_attention;
  uint32_t cache_index;
  const struct gip_gguf_tensor *attn_norm;
  const struct gip_gguf_tensor *ffn_norm;
  const struct gip_gguf_tensor *ffn_gate;
  const struct gip_gguf_tensor *ffn_up;
  const struct gip_gguf_tensor *ffn_down;
  const struct gip_gguf_tensor *attn_q;
  const struct gip_gguf_tensor *attn_k;
  const struct gip_gguf_tensor *attn_v;
  const struct gip_gguf_tensor *attn_output;
  const struct gip_gguf_tensor *attn_q_norm;
  const struct gip_gguf_tensor *attn_k_norm;
  const struct gip_gguf_tensor *conv;
  const struct gip_gguf_tensor *conv_in_proj;
  const struct gip_gguf_tensor *conv_out_proj;
};

/* An LFM2 model loaded from a GGUF file.  */
struct gip_lfm2_model
{
  struct gip_gguf gguf;
  uint32_t n_layers;
  uint32_t n_attention_layers;
  uint32_t n_conv_layers;
  uint32_t n_embd;
  uint32_t n_ff;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t n_vocab;
  uint32_t conv_kernel;
  float rope_theta;
  float norm_eps;
  const struct gip_gguf_tensor *token_embd;
  const struct gip_gguf_tensor *output_norm;
  const struct gip_gguf_tensor *output;
  struct gip_lfm2_layer *layers;
};

/* The per-sequence state of one LFM2 decode: the KV cache of the
   attention layers, the rolling inputs of the convolution layers, and
   scratch space.  N_PAST counts the tokens processed so far.  */
struct gip_lfm2_state
{
  uint32_t n_ctx;
  uint32_t n_past;
  float *k_cache;
  float *v_cache;
  float *conv_state;
  float *hidden;
  float *normed;
  float *block_out;
  float *bcx;
  float *conv_out;
  float *q;
  float *k;
  float *v;
  float *attn;
  float *scores;
  float *gate;
  float *up;
};

/* Buffers that receive intermediate activations of one step.  Each
   non-null member receives N_EMBD floats per entry.  EMBEDDING receives
   the token embedding, LAYERS receives the output of each layer in
   order, and FINAL_NORM receives the normalized last hidden state.  */
struct gip_lfm2_trace
{
  float *embedding;
  float *layers;
  float *final_norm;
};

/* Load the LFM2 model in the GGUF file at PATH into MODEL.  On failure,
   write a message to ERR, which holds ERR_SIZE bytes.  */
enum gip_status gip_lfm2_load (const char *path, struct gip_lfm2_model *model,
                               char *err, size_t err_size);

/* Release everything MODEL owns.  */
void gip_lfm2_free (struct gip_lfm2_model *model);

/* Allocate STATE for sequences of up to N_CTX tokens of MODEL.  */
enum gip_status gip_lfm2_state_init (const struct gip_lfm2_model *model,
                                     uint32_t n_ctx,
                                     struct gip_lfm2_state *state);

/* Release everything STATE owns.  */
void gip_lfm2_state_free (struct gip_lfm2_state *state);

/* Run MODEL on TOKEN at the next position of STATE.  Store the N_VOCAB
   logits at LOGITS unless LOGITS is null.  Store intermediate
   activations in TRACE unless TRACE is null.  */
enum gip_status gip_lfm2_step (const struct gip_lfm2_model *model,
                               struct gip_lfm2_state *state, int32_t token,
                               float *logits,
                               const struct gip_lfm2_trace *trace);

#endif /* GIP_MODEL_LFM2_H */
