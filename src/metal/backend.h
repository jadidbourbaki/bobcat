/* backend.h declares the C interface of gip's Metal backend.  The
   implementation is Objective-C, and every caller sees plain C.  */

#ifndef GIP_METAL_BACKEND_H
#define GIP_METAL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "gip.h"

/* An open Metal device with gip's kernels compiled.  */
struct gip_metal;

/* A Metal buffer in memory shared by the CPU and GPU.  */
struct gip_metal_buffer;

/* A position inside a Metal buffer: OFFSET bytes past the start of
   BUFFER.  */
struct gip_metal_view
{
  struct gip_metal_buffer *buffer;
  size_t offset;
};

/* Open the default Metal device, compile gip's kernels, and store the
   backend in OUT.  On failure, write a message to ERR, which holds
   ERR_SIZE bytes.  */
enum gip_status gip_metal_open (struct gip_metal **out, char *err,
                                size_t err_size);

/* Close METAL.  */
void gip_metal_close (struct gip_metal *metal);

/* Return a new zeroed buffer of SIZE bytes on METAL, or null.  */
struct gip_metal_buffer *gip_metal_buffer_new (struct gip_metal *metal,
                                               size_t size);

/* Return a buffer on METAL over the SIZE bytes at DATA without copying
   them, or null.  DATA must be page-aligned, and the pages covering
   SIZE bytes must stay mapped for the life of the buffer.  */
struct gip_metal_buffer *gip_metal_buffer_wrap (struct gip_metal *metal,
                                                void *data, size_t size);

/* Return the CPU address of the contents of BUFFER.  */
void *gip_metal_buffer_contents (struct gip_metal_buffer *buffer);

/* Return a view of BUFFER at byte OFFSET.  */
struct gip_metal_view gip_metal_at (struct gip_metal_buffer *buffer,
                                    size_t offset);

/* Release BUFFER.  */
void gip_metal_buffer_free (struct gip_metal_buffer *buffer);

/* Start recording kernel launches on METAL into a new command buffer.
   Launches may run at the same time until a barrier separates them.  */
enum gip_status gip_metal_begin (struct gip_metal *metal);

/* Record a barrier on METAL.  Every launch recorded after the barrier
   sees the results of every launch recorded before it.  */
void gip_metal_barrier (struct gip_metal *metal);

/* Work a matrix-vector launch folds in.  When NORM_WEIGHT has a buffer,
   the launch RMS-normalizes its input with epsilon EPS and scales it by
   NORM_WEIGHT before multiplying.  When ACCUMULATE is nonzero, the
   launch adds its results to Y.  */
struct gip_metal_matvec_options
{
  struct gip_metal_view norm_weight;
  float eps;
  int accumulate;
};

/* Record a multiply of the Q8_0 matrix at WEIGHTS, which has N_ROWS rows
   of N_COLS elements, by the N_COLS floats at X.  The N_ROWS results go
   to Y.  OPTIONS may be null for a plain multiply.  */
void gip_metal_matvec_q8_0 (struct gip_metal *metal,
                            struct gip_metal_view weights, uint32_t n_rows,
                            uint32_t n_cols, struct gip_metal_view x,
                            struct gip_metal_view y,
                            const struct gip_metal_matvec_options *options);

/* Record the multiply of the Q8_0 matrices at GATE and UP, which each
   have N_ROWS rows of N_COLS elements, by the N_COLS floats at X after
   RMS normalization with epsilon EPS and scaling by NORM_WEIGHT.  Y
   receives SiLU of each gate result times the matching up result.  */
void gip_metal_matvec_q8_0_swiglu (struct gip_metal *metal,
                                   struct gip_metal_view gate,
                                   struct gip_metal_view up, uint32_t n_rows,
                                   uint32_t n_cols, struct gip_metal_view x,
                                   struct gip_metal_view norm_weight,
                                   float eps, struct gip_metal_view y);

/* Record an RMS normalization of each of the N_ROWS rows of N floats at
   X, scaled by the N floats at WEIGHT, into the matching rows of OUT.  */
void gip_metal_rms_norm (struct gip_metal *metal, struct gip_metal_view x,
                         struct gip_metal_view weight,
                         struct gip_metal_view out, uint32_t n,
                         uint32_t n_rows, float eps);

/* Record the per-head RMS normalization and rotary embedding of the
   N_HEADS heads of HEAD_DIM floats of each of N_TOKENS tokens at SRC.
   Token I sits at position POS + I, SRC_STRIDE floats into SRC and
   DST_STRIDE elements into DST.  The result goes to DST, as
   half-precision numbers when DST_HALF is nonzero and as floats
   otherwise.  SRC and DST may be the same floats.  */
void gip_metal_norm_rope (struct gip_metal *metal, struct gip_metal_view src,
                          struct gip_metal_view dst, int dst_half,
                          struct gip_metal_view weight, uint32_t n_heads,
                          uint32_t head_dim, uint32_t pos, uint32_t n_tokens,
                          uint32_t src_stride, uint32_t dst_stride,
                          float theta, float eps);

/* Record a conversion of the N floats at SRC to half precision at
   DST.  */
void gip_metal_convert_half (struct gip_metal *metal,
                             struct gip_metal_view src,
                             struct gip_metal_view dst, uint32_t n);

/* Return the floats of scratch gip_metal_attention needs for N_QUERIES
   queries of N_HEADS heads of HEAD_DIM floats over up to N_CTX
   positions.  */
size_t gip_metal_attention_scratch (uint32_t n_heads, uint32_t head_dim,
                                    uint32_t n_ctx, uint32_t n_queries);

/* Record causal attention of N_QUERIES queries at Q, each of N_HEADS
   heads, into OUT.  Query I sits at position FIRST_POS + I and attends
   over positions 0 through FIRST_POS + I of K_CACHE and V_CACHE.  Each
   cache holds N_KV_HEADS heads of HEAD_DIM numbers per position, in
   half precision when KV_HALF is nonzero and as floats otherwise.
   SCRATCH holds the floats gip_metal_attention_scratch returns for N_CTX
   positions and N_QUERIES queries.  */
void gip_metal_attention (struct gip_metal *metal, struct gip_metal_view q,
                          struct gip_metal_view k_cache,
                          struct gip_metal_view v_cache, int kv_half,
                          struct gip_metal_view scratch,
                          struct gip_metal_view out, uint32_t n_heads,
                          uint32_t n_kv_heads, uint32_t head_dim,
                          uint32_t first_pos, uint32_t n_queries,
                          uint32_t n_ctx);

/* Record the gated short convolution of N_TOKENS tokens, each with 3 *
   N_EMBD floats at BCX, with KERNEL_SIZE taps per channel at TAPS.
   HISTORY holds the previous KERNEL_SIZE - 1 inputs and moves forward
   one token at a time.  Each token's N_EMBD results go to OUT.  */
void gip_metal_short_conv (struct gip_metal *metal, struct gip_metal_view bcx,
                           struct gip_metal_view taps,
                           struct gip_metal_view history,
                           struct gip_metal_view out, uint32_t n_embd,
                           uint32_t kernel_size, uint32_t n_tokens);

/* Record a multiply of the Q8_0 matrix at WEIGHTS, which has N_ROWS rows
   of N_COLS elements, by each of the N_TOKENS rows of N_COLS floats at X.
   Each token's N_ROWS results go to its row of Y, added to what Y holds
   when ACCUMULATE is nonzero.  */
void gip_metal_matmul_q8_0 (struct gip_metal *metal,
                            struct gip_metal_view weights, uint32_t n_rows,
                            uint32_t n_cols, struct gip_metal_view x,
                            struct gip_metal_view y, uint32_t n_tokens,
                            int accumulate);

/* Record GATE = SiLU (GATE) * UP over N floats.  */
void gip_metal_swiglu (struct gip_metal *metal, struct gip_metal_view gate,
                       struct gip_metal_view up, uint32_t n);

/* Record a copy of N floats from SRC to DST.  */
void gip_metal_copy (struct gip_metal *metal, struct gip_metal_view src,
                     struct gip_metal_view dst, uint32_t n);

/* Record a dequantization of the Q8_0 rows TOKENS[0] through
   TOKENS[N_TOKENS - 1] of WEIGHTS, whose rows hold N_EMBD elements, into
   N_TOKENS rows of N_EMBD floats at OUT.  TOKENS views int32_t token
   ids.  */
void gip_metal_embed_q8_0 (struct gip_metal *metal,
                           struct gip_metal_view weights,
                           struct gip_metal_view tokens,
                           struct gip_metal_view out, uint32_t n_embd,
                           uint32_t n_tokens);

/* Record a store at OUT, one int32_t, of the index of the largest of the
   N floats at X.  Ties go to the lowest index.  */
void gip_metal_argmax (struct gip_metal *metal, struct gip_metal_view x,
                       struct gip_metal_view out, uint32_t n);

/* Submit the recorded launches of METAL to the GPU without waiting for
   them, and store in TICKET a number to wait on with gip_metal_wait.
   Command buffers run in the order they are submitted.  */
enum gip_status gip_metal_commit (struct gip_metal *metal, uint64_t *ticket);

/* Wait until the command buffer with TICKET, and every one submitted
   before it, has finished.  Add their GPU time in seconds to
   GPU_SECONDS unless it is null.  */
enum gip_status gip_metal_wait (struct gip_metal *metal, uint64_t ticket,
                                double *gpu_seconds);

/* Run the recorded launches of METAL and wait for them.  Store the GPU
   time in seconds in GPU_SECONDS unless it is null.  */
enum gip_status gip_metal_end (struct gip_metal *metal, double *gpu_seconds);

/* The GPU time spent in one kernel on one shape while profiling.
   N_ROWS and N_COLS describe matrix-vector launches and are zero for
   other kernels.  BYTES counts the weight bytes the launches read.  */
struct gip_metal_profile_entry
{
  const char *name;
  uint32_t n_rows;
  uint32_t n_cols;
  uint64_t calls;
  double seconds;
  uint64_t bytes;
};

/* Turn profiling of METAL on or off.  Turning it on clears the counts.
   While profiling is on, every launch runs in a command buffer of its
   own and waits for it, so launches run slower and their GPU times add
   up.  */
void gip_metal_set_profiling (struct gip_metal *metal, int enabled);

/* Store the profile entries of METAL in ENTRIES and return their
   count.  */
size_t gip_metal_profile (struct gip_metal *metal,
                          const struct gip_metal_profile_entry **entries);

#endif /* GIP_METAL_BACKEND_H */
