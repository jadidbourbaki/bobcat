/* model_lfm2_metal.h declares the LFM2 forward pass on the Metal GPU.  */

#ifndef GIP_MODEL_LFM2_METAL_H
#define GIP_MODEL_LFM2_METAL_H

#include "metal/backend.h"
#include "model_lfm2.h"

/* The GPU state of one LFM2 decode on METAL: the weights, the caches,
   and the scratch buffers.  N_PAST counts the tokens processed so far.
   KV_HALF is true when the KV cache holds half-precision numbers.  The
   scratch buffers hold BATCH tokens, and the trace buffers hold
   TRACE_ROWS tokens.  LAST_ENCODE_SECONDS and LAST_GPU_SECONDS time the
   CPU recording and the GPU execution of the latest call.  */
struct gip_lfm2_metal
{
  const struct gip_lfm2_model *model;
  struct gip_metal *metal;
  uint32_t n_ctx;
  uint32_t n_past;
  uint32_t batch;
  uint32_t trace_rows;
  bool kv_half;
  double last_encode_seconds;
  double last_gpu_seconds;
  struct gip_metal_buffer *weights;
  const unsigned char *weights_base;
  struct gip_metal_buffer *k_cache;
  struct gip_metal_buffer *v_cache;
  struct gip_metal_buffer *k;
  struct gip_metal_buffer *v;
  struct gip_metal_buffer *conv_state;
  struct gip_metal_buffer *hidden;
  struct gip_metal_buffer *normed;
  struct gip_metal_buffer *bcx;
  struct gip_metal_buffer *conv_out;
  struct gip_metal_buffer *q;
  struct gip_metal_buffer *attn;
  struct gip_metal_buffer *scores;
  struct gip_metal_buffer *ffn;
  struct gip_metal_buffer *logits;
  struct gip_metal_buffer *tokens;
  struct gip_metal_buffer *trace_embedding;
  struct gip_metal_buffer *trace_layers;
  struct gip_metal_buffer *trace_final;
};

/* Prepare GPU for decoding sequences of up to N_CTX tokens of MODEL on
   METAL.  Every matrix of MODEL must be Q8_0.  The KV cache holds half
   precision when KV_HALF is nonzero and floats otherwise.  On failure,
   write a message to ERR, which holds ERR_SIZE bytes.  */
enum gip_status gip_lfm2_metal_init (const struct gip_lfm2_model *model,
                                     struct gip_metal *metal, uint32_t n_ctx,
                                     int kv_half, struct gip_lfm2_metal *gpu,
                                     char *err, size_t err_size);

/* Release everything GPU owns.  */
void gip_lfm2_metal_free (struct gip_lfm2_metal *gpu);

/* Run the model on TOKEN at the next position of GPU, with the same
   outputs as gip_lfm2_step.  */
enum gip_status gip_lfm2_metal_step (struct gip_lfm2_metal *gpu, int32_t token,
                                     float *logits,
                                     const struct gip_lfm2_trace *trace);

/* Run the model on the N tokens at TOKENS at the next positions of GPU,
   in batches of GPU->batch tokens.  Store the logits of the last token
   at LOGITS unless LOGITS is null, in which case the logits buffer
   still receives them for gip_lfm2_metal_generate.  When TRACE is not
   null, its members receive one entry per token.  TRACE->layers holds
   every token's output of layer 0, then of layer 1, and so on.  */
enum gip_status gip_lfm2_metal_prefill (struct gip_lfm2_metal *gpu,
                                        const int32_t *tokens, uint32_t n,
                                        float *logits,
                                        const struct gip_lfm2_trace *trace);

/* Decode N tokens greedily on GPU and store them at OUT.  The first
   token is the argmax of the logits of the previous step, which must
   have asked for logits.  Each token then runs through the model to
   choose the next.  The GPU picks every token itself, and the CPU
   submits several steps ahead, so the GPU never waits for the CPU.  */
enum gip_status gip_lfm2_metal_generate (struct gip_lfm2_metal *gpu,
                                         uint32_t n, int32_t *out);

#endif /* GIP_MODEL_LFM2_METAL_H */
