/* kernels.metal holds gip's Metal compute kernels.  The library embeds
   this source and compiles it when the Metal backend opens.  */

#include <metal_stdlib>
using namespace metal;

/* The rows each simdgroup of matvec_q8_0 computes.  The host sets the
   value when it creates the pipeline.  */
constant uint rows_per_simdgroup [[function_constant (0)]];

enum
{
  QK8_0 = 32,
  Q8_0_BLOCK_BYTES = 34,
  SIMD_WIDTH = 32,
  MAX_ROWS_PER_SIMDGROUP = 8
};

/* One Q8_0 block: a half-precision scale and 32 signed 8-bit weights.  */
struct q8_0_block
{
  half scale;
  char quants[QK8_0];
};

/* Multiply the Q8_0 matrix WEIGHTS, which has N_ROWS rows of N_COLS
   elements, by the N_COLS floats at X and store the N_ROWS results at Y.
   Each simdgroup computes ROWS_PER_SIMDGROUP consecutive rows.  Each
   lane takes every 32nd block of those rows and loads the matching 32
   inputs once for all of them.  */
kernel void
matvec_q8_0 (device const uchar *weights [[buffer (0)]],
             device const float *x [[buffer (1)]],
             device float *y [[buffer (2)]],
             constant uint &n_rows [[buffer (3)]],
             constant uint &n_cols [[buffer (4)]],
             uint threadgroup_index [[threadgroup_position_in_grid]],
             uint simdgroup_index [[simdgroup_index_in_threadgroup]],
             uint simdgroups [[simdgroups_per_threadgroup]],
             uint lane [[thread_index_in_simdgroup]])
{
  uint first_row
      = (threadgroup_index * simdgroups + simdgroup_index) * rows_per_simdgroup;
  uint n_blocks = n_cols / QK8_0;
  ulong row_bytes = ulong (n_blocks) * Q8_0_BLOCK_BYTES;
  float sums[MAX_ROWS_PER_SIMDGROUP] = { 0.0f };

  for (uint b = lane; b < n_blocks; b += SIMD_WIDTH)
    {
      device const float4 *xb = (device const float4 *)(x + b * QK8_0);
      float4 inputs[QK8_0 / 4];
      for (int i = 0; i < QK8_0 / 4; i++)
        inputs[i] = xb[i];

      for (uint r = 0; r < rows_per_simdgroup; r++)
        {
          uint row = first_row + r;
          if (row >= n_rows)
            break;
          device const q8_0_block *block
              = (device const q8_0_block *)(weights + row * row_bytes) + b;
          float block_dot = 0.0f;
          for (int i = 0; i < QK8_0 / 4; i++)
            {
              float4 q = float4 (block->quants[4 * i],
                                 block->quants[4 * i + 1],
                                 block->quants[4 * i + 2],
                                 block->quants[4 * i + 3]);
              block_dot += dot (q, inputs[i]);
            }
          sums[r] += float (block->scale) * block_dot;
        }
    }

  /* Every lane of a simdgroup sees the same rows, so all lanes reach
     each simd_sum together.  */
  for (uint r = 0; r < rows_per_simdgroup; r++)
    {
      uint row = first_row + r;
      if (row >= n_rows)
        break;
      float total = simd_sum (sums[r]);
      if (lane == 0)
        y[row] = total;
    }
}
