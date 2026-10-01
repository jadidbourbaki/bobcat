/* Shared constants, quantized block types, and reduction helpers.  The
   backend compiles this source before the operation kernels.  */

#include <metal_stdlib>
using namespace metal;

/* The rows each threadgroup of the matrix-vector kernels computes.  The
   host sets the value when it creates the pipelines.  */
constant uint rows_per_threadgroup [[function_constant (0)]];

/* Whether a matrix-vector kernel RMS-normalizes its input and scales it
   by NORM_WEIGHT before multiplying.  */
constant bool fuse_norm [[function_constant (1)]];

/* Whether a matrix kernel adds its product to what Y holds.  */
constant bool accumulate [[function_constant (2)]];

/* Whether the matrix-matrix kernel stores SiLU of what Y holds times its
   product, which turns the up projection into the SwiGLU of the gate
   projection already in Y.  */
constant bool swiglu_store [[function_constant (3)]];

enum
{
  QK8_0 = 32,
  Q8_0_BLOCK_BYTES = 34,
  SIMD_WIDTH = 32,
  MAX_MATVEC_ROWS = 8,
  /* In the matrix-vector kernels each lane takes 8 of a block's 32
     quants, so 4 lanes share a block and a simdgroup covers 8 blocks at
     a time.  */
  QUANTS_PER_LANE = 8,
  LANES_PER_BLOCK = QK8_0 / QUANTS_PER_LANE,
  BLOCKS_PER_SIMDGROUP = SIMD_WIDTH / LANES_PER_BLOCK,
  /* Threadgroups of the reduction kernels hold at most this many
     simdgroups.  */
  MAX_SIMDGROUPS = 32,
  MAX_HEAD_DIM = 256,
  /* The attention kernels' limits.  Each chunk covers ATTENTION_CHUNK
     positions with one thread per position.  On LFM2.5-350M at a
     512-token context on an M4 Pro, 64-position chunks decoded at 369
     tokens per second, against 366 for 32 and 348 for 128.  The host
     checks the model against the other two limits.  */
  ATTENTION_CHUNK = 64,
  ATTENTION_MAX_GROUP = 4,
  ATTENTION_MAX_HEAD_DIM = 256,
  /* The simdgroups of an attention_wide threadgroup, which split a
     chunk's positions.  Two and four measured within 2 percent of each
     other on Qwen3.5-0.8B and 9B on an M4 Pro, and eight overflow the
     threadgroup memory of the merge.  ATTENTION_WIDE_SIMDGROUPS in
     backend.rs keeps the same value.  */
  ATTENTION_WIDE_SIMDGROUPS = 4,
  /* The Gated DeltaNet kernels' limits.  The host checks the model
     against both.  */
  GDN_MAX_KERNEL = 8,
  GDN_MAX_K_DIM = 128,
  /* The state columns, one per simdgroup, of a gdn_recurrence
     threadgroup.  On an M4 Pro, four columns decoded Qwen3.5-0.8B at 335
     tokens per second, against 327 for two and 334 for eight.
     GDN_COLUMNS in backend.rs keeps the same value.  */
  GDN_COLUMNS = 4,
  /* A threadgroup of attention_flash covers FLASH_QUERIES queries of one
     head with four simdgroups of 8 queries each, over FLASH_KEYS keys at
     a time, for heads of FLASH_HEAD_DIM.  */
  FLASH_QUERIES = 32,
  FLASH_KEYS = 32,
  FLASH_SIMDGROUPS = 4,
  FLASH_HEAD_DIM = 64,
  /* The mixture-of-experts kernels' limits.  The host checks the model
     against both.  A routing threadgroup has MOE_ROUTE_SIMDGROUPS
     simdgroups and routes MOE_ROUTE_TOKENS tokens.  A grouping
     threadgroup has MOE_GROUP_SIMDGROUPS simdgroups.  */
  MOE_MAX_EXPERTS = 128,
  MOE_MAX_USED = 8,
  MOE_ROUTE_SIMDGROUPS = 32,
  MOE_ROUTE_TOKENS = 8,
  MOE_GROUP_SIMDGROUPS = 32,
  /* A matmul threadgroup computes MATMUL_ROWS rows by MATMUL_TOKENS
     tokens with four simdgroups, each owning 32 rows by 16 tokens.  */
  MATMUL_ROWS = 64,
  MATMUL_TOKENS = 32,
  MATMUL_SIMDGROUPS = 4
};

/* Return the sum of VALUE over all threads of the threadgroup.  PARTIALS
   is threadgroup scratch of MAX_SIMDGROUPS floats.  Every thread of the
   threadgroup must call the function.  */
static float
threadgroup_sum (float value, threadgroup float *partials,
                 uint simdgroup_index, uint simdgroups, uint lane)
{
  float simd_total = simd_sum (value);

  if (lane == 0)
    partials[simdgroup_index] = simd_total;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < simdgroups; s++)
    total += partials[s];
  threadgroup_barrier (mem_flags::mem_threadgroup);
  return total;
}

/* Return the maximum of VALUE over all threads of the threadgroup, with
   the same contract as threadgroup_sum.  */
static float
threadgroup_max (float value, threadgroup float *partials,
                 uint simdgroup_index, uint simdgroups, uint lane)
{
  float simd_best = simd_max (value);

  if (lane == 0)
    partials[simdgroup_index] = simd_best;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  float best = -INFINITY;
  for (uint s = 0; s < simdgroups; s++)
    best = max (best, partials[s]);
  threadgroup_barrier (mem_flags::mem_threadgroup);
  return best;
}

/* One Q8_0 block: a half-precision scale and 32 signed 8-bit weights.  */
struct q8_0_block
{
  half scale;
  char quants[QK8_0];
};

/* Return the dot product of the 8 quants of BLOCK starting at quant
   8 * PART with the 8 floats at INPUTS, times the block's scale.  */
static float
q8_0_part_dot (device const q8_0_block *block, uint part,
               thread const float4 *inputs)
{
  /* The quants start two bytes into the block, so four-byte loads go
     through the packed type, which needs only byte alignment.  */
  device const packed_char4 *quants
      = (device const packed_char4 *)(block->quants + part * QUANTS_PER_LANE);
  float part_dot = dot (float4 (char4 (quants[0])), inputs[0])
                   + dot (float4 (char4 (quants[1])), inputs[1]);
  return float (block->scale) * part_dot;
}

/* Load the 8 floats of X starting at START into INPUTS.  With fuse_norm,
   add their squares to SUM_SQUARES and multiply each by its entry of
   NORM_WEIGHT.

   The RMS norm scale is one number for the whole vector, so the kernels
   multiply by the weighted inputs and apply the scale to each finished
   row sum.  The threads of a threadgroup load every input between them,
   so their squares add up to the whole norm with no extra reads.  */
static void
load_inputs (device const float *x, device const float *norm_weight,
             uint start, thread float4 *inputs, thread float &sum_squares)
{
  device const float4 *xs = (device const float4 *)(x + start);

  inputs[0] = xs[0];
  inputs[1] = xs[1];
  if (fuse_norm)
    {
      device const float4 *ws = (device const float4 *)(norm_weight + start);
      sum_squares += dot (inputs[0], inputs[0]) + dot (inputs[1], inputs[1]);
      inputs[0] *= ws[0];
      inputs[1] *= ws[1];
    }
}

/* Replace each of the N floats at VALUES with its sum over the whole
   threadgroup.  PARTIALS is threadgroup scratch of N * MAX_SIMDGROUPS
   floats.  Every thread of the threadgroup must call the function.  */
static void
threadgroup_sums (thread float *values, uint n, threadgroup float *partials,
                  uint simdgroup_index, uint simdgroups, uint lane)
{
  for (uint i = 0; i < n; i++)
    {
      float simd_total = simd_sum (values[i]);
      if (lane == 0)
        partials[i * MAX_SIMDGROUPS + simdgroup_index] = simd_total;
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);
  for (uint i = 0; i < n; i++)
    {
      float total = 0.0f;
      for (uint s = 0; s < simdgroups; s++)
        total += partials[i * MAX_SIMDGROUPS + s];
      values[i] = total;
    }
}
