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
  MAX_ROWS_PER_SIMDGROUP = 8,
  /* Threadgroups of the reduction kernels hold at most this many
     simdgroups.  */
  MAX_SIMDGROUPS = 32,
  MAX_HEAD_DIM = 256
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

/* Normalize the N floats at X by their root mean square, scale them by
   the N floats at WEIGHT, and store the result at OUT.  One threadgroup
   handles the whole vector.  */
kernel void
rms_norm (device const float *x [[buffer (0)]],
          device const float *weight [[buffer (1)]],
          device float *out [[buffer (2)]], constant uint &n [[buffer (3)]],
          constant float &eps [[buffer (4)]],
          uint tid [[thread_position_in_threadgroup]],
          uint threads [[threads_per_threadgroup]],
          uint simdgroup_index [[simdgroup_index_in_threadgroup]],
          uint simdgroups [[simdgroups_per_threadgroup]],
          uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  float sum_squares = 0.0f;

  for (uint i = tid; i < n; i += threads)
    sum_squares += x[i] * x[i];
  sum_squares = threadgroup_sum (sum_squares, partials, simdgroup_index,
                                 simdgroups, lane);

  float scale = precise::rsqrt (sum_squares / float (n) + eps);
  for (uint i = tid; i < n; i += threads)
    out[i] = weight[i] * (x[i] * scale);
}

/* Normalize each of the heads of HEAD_DIM floats at VEC by its root mean
   square, scale it by the HEAD_DIM floats at WEIGHT, and rotate it for
   position POS with base THETA, pairing element I with element I +
   HEAD_DIM / 2.  One threadgroup of HEAD_DIM threads handles one
   head.  */
kernel void
qk_norm_rope (device float *vec [[buffer (0)]],
              device const float *weight [[buffer (1)]],
              constant uint &head_dim [[buffer (2)]],
              constant uint &pos [[buffer (3)]],
              constant float &theta [[buffer (4)]],
              constant float &eps [[buffer (5)]],
              uint head [[threadgroup_position_in_grid]],
              uint i [[thread_position_in_threadgroup]],
              uint simdgroup_index [[simdgroup_index_in_threadgroup]],
              uint simdgroups [[simdgroups_per_threadgroup]],
              uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  threadgroup float normed[MAX_HEAD_DIM];
  device float *v = vec + head * head_dim;

  float value = v[i];
  float sum_squares = threadgroup_sum (value * value, partials,
                                       simdgroup_index, simdgroups, lane);
  float scale = precise::rsqrt (sum_squares / float (head_dim) + eps);
  normed[i] = weight[i] * (value * scale);
  threadgroup_barrier (mem_flags::mem_threadgroup);

  uint half_dim = head_dim / 2;
  if (i < half_dim)
    {
      float inv_freq
          = 1.0f / precise::pow (theta, float (2 * i) / float (head_dim));
      float angle = float (pos) * inv_freq;
      float c = precise::cos (angle);
      float s = precise::sin (angle);
      float x0 = normed[i];
      float x1 = normed[i + half_dim];
      v[i] = x0 * c - x1 * s;
      v[i + half_dim] = x1 * c + x0 * s;
    }
}

/* Attend with each of the query heads at Q over the first N_KEYS
   positions of K_CACHE and V_CACHE, and store each head's result at
   OUT.  Query heads share KV heads in consecutive groups of N_HEADS /
   N_KV_HEADS.  SCORES holds N_CTX floats of scratch per query head.  One
   threadgroup handles one query head.  */
kernel void
attention (device const float *q [[buffer (0)]],
           device const float *k_cache [[buffer (1)]],
           device const float *v_cache [[buffer (2)]],
           device float *scores [[buffer (3)]],
           device float *out [[buffer (4)]],
           constant uint &n_heads [[buffer (5)]],
           constant uint &n_kv_heads [[buffer (6)]],
           constant uint &head_dim [[buffer (7)]],
           constant uint &n_keys [[buffer (8)]],
           constant uint &n_ctx [[buffer (9)]],
           constant float &scale [[buffer (10)]],
           uint head [[threadgroup_position_in_grid]],
           uint tid [[thread_position_in_threadgroup]],
           uint threads [[threads_per_threadgroup]],
           uint simdgroup_index [[simdgroup_index_in_threadgroup]],
           uint simdgroups [[simdgroups_per_threadgroup]],
           uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  threadgroup float query[MAX_HEAD_DIM];
  threadgroup float out_partials[MAX_SIMDGROUPS * SIMD_WIDTH];

  uint kv_dim = n_kv_heads * head_dim;
  uint kv_head = head / (n_heads / n_kv_heads);
  device const float *keys = k_cache + kv_head * head_dim;
  device const float *values = v_cache + kv_head * head_dim;
  device float *head_scores = scores + head * n_ctx;

  for (uint d = tid; d < head_dim; d += threads)
    query[d] = q[head * head_dim + d];
  threadgroup_barrier (mem_flags::mem_threadgroup);

  float local_max = -INFINITY;
  for (uint t = tid; t < n_keys; t += threads)
    {
      float s = 0.0f;
      for (uint d = 0; d < head_dim; d++)
        s += query[d] * keys[t * kv_dim + d];
      s *= scale;
      head_scores[t] = s;
      local_max = max (local_max, s);
    }
  float best = threadgroup_max (local_max, partials, simdgroup_index,
                                simdgroups, lane);

  float local_sum = 0.0f;
  for (uint t = tid; t < n_keys; t += threads)
    {
      float e = precise::exp (head_scores[t] - best);
      head_scores[t] = e;
      local_sum += e;
    }
  float total = threadgroup_sum (local_sum, partials, simdgroup_index,
                                 simdgroups, lane);
  threadgroup_barrier (mem_flags::mem_device);

  /* Threads split into groups of HEAD_DIM.  Each group sums the weighted
     values of every group-th position, and the first group adds the
     groups together.  */
  uint groups = threads / head_dim;
  uint group = tid / head_dim;
  uint d = tid % head_dim;
  float acc = 0.0f;
  if (group < groups)
    for (uint t = group; t < n_keys; t += groups)
      acc += head_scores[t] * values[t * kv_dim + d];
  out_partials[tid] = acc;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  if (group == 0)
    {
      float sum = 0.0f;
      for (uint g = 0; g < groups; g++)
        sum += out_partials[g * head_dim + d];
      out[head * head_dim + d] = sum / total;
    }
}

/* Run the gated short convolution on the 3 * N_EMBD floats at BCX, which
   hold the gates B and C and the input X.  HISTORY holds B times X for
   the previous KERNEL - 1 tokens, oldest first, and moves forward by one
   token.  TAPS holds KERNEL taps per channel.  One thread handles one
   channel.  */
kernel void
short_conv (device const float *bcx [[buffer (0)]],
            device const float *taps [[buffer (1)]],
            device float *history [[buffer (2)]],
            device float *out [[buffer (3)]],
            constant uint &n_embd [[buffer (4)]],
            constant uint &kernel_size [[buffer (5)]],
            uint ch [[thread_position_in_grid]])
{
  if (ch >= n_embd)
    return;

  device const float *channel_taps = taps + ch * kernel_size;
  float bx = bcx[ch] * bcx[2 * n_embd + ch];
  float sum = channel_taps[kernel_size - 1] * bx;
  for (uint k = 0; k + 1 < kernel_size; k++)
    sum += channel_taps[k] * history[k * n_embd + ch];
  for (uint k = 0; k + 2 < kernel_size; k++)
    history[k * n_embd + ch] = history[(k + 1) * n_embd + ch];
  history[(kernel_size - 2) * n_embd + ch] = bx;
  out[ch] = bcx[n_embd + ch] * sum;
}

/* Replace each of the N floats at GATE with its SiLU times the matching
   float at UP.  */
kernel void
swiglu (device float *gate [[buffer (0)]],
        device const float *up [[buffer (1)]], constant uint &n [[buffer (2)]],
        uint i [[thread_position_in_grid]])
{
  if (i >= n)
    return;
  float g = gate[i];
  gate[i] = g / (1.0f + precise::exp (-g)) * up[i];
}

/* Add the N floats at DELTA to the N floats at H.  */
kernel void
add_in_place (device float *h [[buffer (0)]],
              device const float *delta [[buffer (1)]],
              constant uint &n [[buffer (2)]],
              uint i [[thread_position_in_grid]])
{
  if (i < n)
    h[i] += delta[i];
}

/* Copy the N floats at SRC to DST.  */
kernel void
copy_floats (device const float *src [[buffer (0)]],
             device float *dst [[buffer (1)]], constant uint &n [[buffer (2)]],
             uint i [[thread_position_in_grid]])
{
  if (i < n)
    dst[i] = src[i];
}
