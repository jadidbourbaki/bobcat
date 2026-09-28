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
   Each launch reads the results of the launches recorded before it.  */
enum gip_status gip_metal_begin (struct gip_metal *metal);

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

/* Record an RMS normalization of the N floats at X, scaled by the N
   floats at WEIGHT, into OUT.  */
void gip_metal_rms_norm (struct gip_metal *metal, struct gip_metal_view x,
                         struct gip_metal_view weight,
                         struct gip_metal_view out, uint32_t n, float eps);

/* Record the per-head RMS normalization and rotary embedding of the
   N_HEADS heads of HEAD_DIM floats at VEC for position POS.  */
void gip_metal_qk_norm_rope (struct gip_metal *metal,
                             struct gip_metal_view vec,
                             struct gip_metal_view weight, uint32_t n_heads,
                             uint32_t head_dim, uint32_t pos, float theta,
                             float eps);

/* Return the floats of scratch gip_metal_attention needs for N_HEADS
   query heads of HEAD_DIM floats over up to N_CTX positions.  */
size_t gip_metal_attention_scratch (uint32_t n_heads, uint32_t head_dim,
                                    uint32_t n_ctx);

/* Record attention of the N_HEADS query heads at Q over the first N_KEYS
   positions of K_CACHE and V_CACHE into OUT.  Each cache holds
   N_KV_HEADS heads of HEAD_DIM floats per position.  SCRATCH holds the
   floats gip_metal_attention_scratch returns for N_CTX positions.  */
void gip_metal_attention (struct gip_metal *metal, struct gip_metal_view q,
                          struct gip_metal_view k_cache,
                          struct gip_metal_view v_cache,
                          struct gip_metal_view scratch,
                          struct gip_metal_view out, uint32_t n_heads,
                          uint32_t n_kv_heads, uint32_t head_dim,
                          uint32_t n_keys, uint32_t n_ctx);

/* Record the gated short convolution of the 3 * N_EMBD floats at BCX
   with KERNEL_SIZE taps per channel at TAPS.  HISTORY holds the previous
   KERNEL_SIZE - 1 inputs and moves forward one token.  The N_EMBD
   results go to OUT.  */
void gip_metal_short_conv (struct gip_metal *metal, struct gip_metal_view bcx,
                           struct gip_metal_view taps,
                           struct gip_metal_view history,
                           struct gip_metal_view out, uint32_t n_embd,
                           uint32_t kernel_size);

/* Record a copy of N floats from SRC to DST.  */
void gip_metal_copy (struct gip_metal *metal, struct gip_metal_view src,
                     struct gip_metal_view dst, uint32_t n);

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
