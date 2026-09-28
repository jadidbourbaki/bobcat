/* kernels.metal holds gip's Metal compute kernels.  The library embeds
   this source and compiles it when the Metal backend opens.  */

#include <metal_stdlib>
using namespace metal;

/* The rows each threadgroup of the matrix-vector kernels computes.  The
   host sets the value when it creates the pipelines.  */
constant uint rows_per_threadgroup [[function_constant (0)]];

/* Whether a matrix-vector kernel RMS-normalizes its input and scales it
   by NORM_WEIGHT before multiplying.  */
constant bool fuse_norm [[function_constant (1)]];

/* Whether a matrix-vector kernel adds its product to Y instead of
   overwriting Y.  */
constant bool accumulate [[function_constant (2)]];

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
  ATTENTION_MAX_HEAD_DIM = 128
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

/* Multiply the Q8_0 matrix WEIGHTS, which has N_ROWS rows of N_COLS
   elements, by the N_COLS floats at X and store the N_ROWS results at Y.
   With fuse_norm, X is first RMS-normalized with epsilon EPS and scaled
   by NORM_WEIGHT.  With accumulate, the results add to Y.

   The layout follows llama.cpp's Q8_0 matrix-vector kernel.  Each
   threadgroup computes ROWS_PER_THREADGROUP consecutive rows, and its
   simdgroups split the columns between them.  Four lanes share each
   block, so a simdgroup reads 8 neighboring blocks at a time and a small
   matrix still spreads over many threadgroups.  */
kernel void
matvec_q8_0 (device const uchar *weights [[buffer (0)]],
             device const float *x [[buffer (1)]],
             device float *y [[buffer (2)]],
             constant uint &n_rows [[buffer (3)]],
             constant uint &n_cols [[buffer (4)]],
             device const float *norm_weight
             [[buffer (5), function_constant (fuse_norm)]],
             constant float &eps [[buffer (6), function_constant (fuse_norm)]],
             uint threadgroup_index [[threadgroup_position_in_grid]],
             uint simdgroup_index [[simdgroup_index_in_threadgroup]],
             uint simdgroups [[simdgroups_per_threadgroup]],
             uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[(MAX_MATVEC_ROWS + 1) * MAX_SIMDGROUPS];
  uint first_row = threadgroup_index * rows_per_threadgroup;
  uint n_blocks = n_cols / QK8_0;
  ulong row_bytes = ulong (n_blocks) * Q8_0_BLOCK_BYTES;
  uint part = lane % LANES_PER_BLOCK;
  uint stride = simdgroups * BLOCKS_PER_SIMDGROUP;

  /* SUMS holds one sum per row, then the sum of squares of X.  */
  float sums[MAX_MATVEC_ROWS + 1] = { 0.0f };
  uint n_sums = rows_per_threadgroup + 1;

  for (uint b = simdgroup_index * BLOCKS_PER_SIMDGROUP + lane / LANES_PER_BLOCK;
       b < n_blocks; b += stride)
    {
      float4 inputs[2];
      load_inputs (x, norm_weight, b * QK8_0 + part * QUANTS_PER_LANE, inputs,
                   sums[rows_per_threadgroup]);
      for (uint r = 0; r < rows_per_threadgroup; r++)
        {
          uint row = first_row + r;
          if (row >= n_rows)
            break;
          device const q8_0_block *block
              = (device const q8_0_block *)(weights + row * row_bytes) + b;
          sums[r] += q8_0_part_dot (block, part, inputs);
        }
    }

  threadgroup_sums (sums, n_sums, partials, simdgroup_index, simdgroups, lane);
  if (simdgroup_index != 0 || lane != 0)
    return;

  float scale = 1.0f;
  if (fuse_norm)
    scale = precise::rsqrt (sums[rows_per_threadgroup] / float (n_cols) + eps);
  for (uint r = 0; r < rows_per_threadgroup; r++)
    {
      uint row = first_row + r;
      if (row >= n_rows)
        break;
      float total = sums[r] * scale;
      y[row] = accumulate ? y[row] + total : total;
    }
}

/* Multiply the Q8_0 matrices GATE and UP, which each have N_ROWS rows of
   N_COLS elements, by the N_COLS floats at X, and store SiLU of each
   gate result times the matching up result at Y.  With fuse_norm, X is
   first RMS-normalized with epsilon EPS and scaled by NORM_WEIGHT.  The
   layout follows matvec_q8_0, with each lane reading both matrices.  */
kernel void
matvec_q8_0_swiglu (device const uchar *gate [[buffer (0)]],
                    device const uchar *up [[buffer (1)]],
                    device const float *x [[buffer (2)]],
                    device float *y [[buffer (3)]],
                    constant uint &n_rows [[buffer (4)]],
                    constant uint &n_cols [[buffer (5)]],
                    device const float *norm_weight
                    [[buffer (6), function_constant (fuse_norm)]],
                    constant float &eps
                    [[buffer (7), function_constant (fuse_norm)]],
                    uint threadgroup_index [[threadgroup_position_in_grid]],
                    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                    uint simdgroups [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[(2 * MAX_MATVEC_ROWS + 1) * MAX_SIMDGROUPS];
  uint rows = rows_per_threadgroup;
  uint first_row = threadgroup_index * rows;
  uint n_blocks = n_cols / QK8_0;
  ulong row_bytes = ulong (n_blocks) * Q8_0_BLOCK_BYTES;
  uint part = lane % LANES_PER_BLOCK;
  uint stride = simdgroups * BLOCKS_PER_SIMDGROUP;

  /* SUMS holds the gate sums, then the up sums, then the sum of squares
     of X.  */
  float sums[2 * MAX_MATVEC_ROWS + 1] = { 0.0f };
  uint n_sums = 2 * rows + 1;

  for (uint b = simdgroup_index * BLOCKS_PER_SIMDGROUP + lane / LANES_PER_BLOCK;
       b < n_blocks; b += stride)
    {
      float4 inputs[2];
      load_inputs (x, norm_weight, b * QK8_0 + part * QUANTS_PER_LANE, inputs,
                   sums[2 * rows]);
      for (uint r = 0; r < rows; r++)
        {
          uint row = first_row + r;
          if (row >= n_rows)
            break;
          ulong offset = row * row_bytes;
          sums[r] += q8_0_part_dot (
              (device const q8_0_block *)(gate + offset) + b, part, inputs);
          sums[rows + r] += q8_0_part_dot (
              (device const q8_0_block *)(up + offset) + b, part, inputs);
        }
    }

  threadgroup_sums (sums, n_sums, partials, simdgroup_index, simdgroups, lane);
  if (simdgroup_index != 0 || lane != 0)
    return;

  float scale = 1.0f;
  if (fuse_norm)
    scale = precise::rsqrt (sums[2 * rows] / float (n_cols) + eps);
  for (uint r = 0; r < rows; r++)
    {
      uint row = first_row + r;
      if (row >= n_rows)
        break;
      float g = sums[r] * scale;
      float u = sums[rows + r] * scale;
      y[row] = g / (1.0f + precise::exp (-g)) * u;
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

/* Normalize each of the heads of HEAD_DIM floats at SRC by its root mean
   square, scale it by the HEAD_DIM floats at WEIGHT, rotate it for
   position POS with base THETA, and store it at DST as T.  Element I
   pairs with element I + HEAD_DIM / 2.  One threadgroup of HEAD_DIM
   threads handles one head.  SRC and DST may be the same floats.  */
template <typename T>
kernel void
norm_rope (device const float *src [[buffer (0)]],
           device T *dst [[buffer (1)]],
           device const float *weight [[buffer (2)]],
           constant uint &head_dim [[buffer (3)]],
           constant uint &pos [[buffer (4)]],
           constant float &theta [[buffer (5)]],
           constant float &eps [[buffer (6)]],
           uint head [[threadgroup_position_in_grid]],
           uint i [[thread_position_in_threadgroup]],
           uint simdgroup_index [[simdgroup_index_in_threadgroup]],
           uint simdgroups [[simdgroups_per_threadgroup]],
           uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  threadgroup float normed[MAX_HEAD_DIM];

  float value = src[head * head_dim + i];
  float sum_squares = threadgroup_sum (value * value, partials,
                                       simdgroup_index, simdgroups, lane);
  float scale = precise::rsqrt (sum_squares / float (head_dim) + eps);
  normed[i] = weight[i] * (value * scale);
  threadgroup_barrier (mem_flags::mem_threadgroup);

  uint half_dim = head_dim / 2;
  device T *out = dst + head * head_dim;
  if (i < half_dim)
    {
      float inv_freq
          = 1.0f / precise::pow (theta, float (2 * i) / float (head_dim));
      float angle = float (pos) * inv_freq;
      float c = precise::cos (angle);
      float s = precise::sin (angle);
      float x0 = normed[i];
      float x1 = normed[i + half_dim];
      out[i] = T (x0 * c - x1 * s);
      out[i + half_dim] = T (x1 * c + x0 * s);
    }
}

/* The element type appears in the signature, so each instantiation
   names its own type.  */
template [[host_name ("norm_rope_f32")]] kernel decltype (norm_rope<float>)
    norm_rope<float>;
template [[host_name ("norm_rope_f16")]] kernel decltype (norm_rope<half>)
    norm_rope<half>;

/* Convert the N floats at SRC to half precision at DST.  */
kernel void
convert_half (device const float *src [[buffer (0)]],
              device half *dst [[buffer (1)]], constant uint &n [[buffer (2)]],
              uint i [[thread_position_in_grid]])
{
  if (i < n)
    dst[i] = half (src[i]);
}

/* Attend with the query heads at Q over one chunk of ATTENTION_CHUNK
   positions of K_CACHE and V_CACHE, as the first of two passes.  Query
   heads share KV heads in consecutive groups of N_HEADS / N_KV_HEADS, at
   most ATTENTION_MAX_GROUP.

   One threadgroup of ATTENTION_CHUNK threads handles one KV head, every
   query head in its group, and the chunk CHUNK.  Each thread scores one
   position, so K and V are read once per chunk.  For each query head
   the pass stores the chunk's largest score, its sum of exponentials
   relative to that score, and its unnormalized weighted sum of values in
   SCRATCH, laid out as attention_combine expects for MAX_CHUNKS chunks.
   The caches hold elements of type T.  */
template <typename T>
kernel void
attention_chunk (device const float *q [[buffer (0)]],
                 device const T *k_cache [[buffer (1)]],
                 device const T *v_cache [[buffer (2)]],
                 device float *scratch [[buffer (3)]],
                 constant uint &n_heads [[buffer (4)]],
                 constant uint &n_kv_heads [[buffer (5)]],
                 constant uint &head_dim [[buffer (6)]],
                 constant uint &n_keys [[buffer (7)]],
                 constant uint &max_chunks [[buffer (8)]],
                 constant float &scale [[buffer (9)]],
                 uint2 position [[threadgroup_position_in_grid]],
                 uint2 thread_position [[thread_position_in_threadgroup]],
                 uint2 threadgroup_size [[threads_per_threadgroup]],
                 uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                 uint simdgroups [[simdgroups_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  threadgroup float query[ATTENTION_MAX_GROUP * ATTENTION_MAX_HEAD_DIM];
  threadgroup float weights[ATTENTION_MAX_GROUP * ATTENTION_CHUNK];

  uint tid = thread_position.x;
  uint threads = threadgroup_size.x;
  uint kv_head = position.x;
  uint chunk = position.y;
  uint group = n_heads / n_kv_heads;
  uint kv_dim = n_kv_heads * head_dim;
  uint first_head = kv_head * group;
  uint start = chunk * ATTENTION_CHUNK;
  uint count = min (uint (ATTENTION_CHUNK), n_keys - start);

  for (uint i = tid; i < group * head_dim; i += threads)
    query[i] = q[first_head * head_dim + i];
  threadgroup_barrier (mem_flags::mem_threadgroup);

  float score[ATTENTION_MAX_GROUP];
  bool valid = tid < count;
  device const vec<T, 4> *key
      = (device const vec<T, 4> *)(k_cache + (start + tid) * kv_dim
                                   + kv_head * head_dim);
  for (uint g = 0; g < group; g++)
    {
      threadgroup const float4 *qg
          = (threadgroup const float4 *)(query + g * head_dim);
      float s = 0.0f;
      if (valid)
        for (uint d = 0; d < head_dim / 4; d++)
          s += dot (qg[d], float4 (key[d]));
      score[g] = valid ? s * scale : -INFINITY;
    }

  device float *chunk_max = scratch + n_heads * max_chunks * head_dim;
  device float *chunk_sum = chunk_max + n_heads * max_chunks;
  for (uint g = 0; g < group; g++)
    {
      float best = threadgroup_max (score[g], partials, simdgroup_index,
                                    simdgroups, lane);
      float e = valid ? precise::exp (score[g] - best) : 0.0f;
      weights[g * ATTENTION_CHUNK + tid] = e;
      float total = threadgroup_sum (e, partials, simdgroup_index,
                                     simdgroups, lane);
      if (tid == 0)
        {
          uint slot = (first_head + g) * max_chunks + chunk;
          chunk_max[slot] = best;
          chunk_sum[slot] = total;
        }
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);

  /* Neighboring threads take neighboring elements of each value row, so
     the reads coalesce.  */
  device const T *values = v_cache + start * kv_dim + kv_head * head_dim;
  for (uint i = tid; i < group * head_dim; i += threads)
    {
      uint g = i / head_dim;
      uint d = i % head_dim;
      threadgroup const float *w = weights + g * ATTENTION_CHUNK;
      float acc = 0.0f;
      for (uint t = 0; t < count; t++)
        acc += w[t] * float (values[t * kv_dim + d]);
      scratch[((first_head + g) * max_chunks + chunk) * head_dim + d] = acc;
    }
}

template [[host_name ("attention_chunk_f32")]] kernel decltype (
    attention_chunk<float>) attention_chunk<float>;
template [[host_name ("attention_chunk_f16")]] kernel decltype (
    attention_chunk<half>) attention_chunk<half>;

/* Combine the N_CHUNKS chunk results of attention_chunk in SCRATCH into
   the result of each query head at OUT, as the second of two passes.
   Each chunk's sums are rescaled from its own largest score to the
   largest score of all chunks.  One threadgroup of HEAD_DIM threads
   handles one query head.  */
kernel void
attention_combine (device const float *scratch [[buffer (0)]],
                   device float *out [[buffer (1)]],
                   constant uint &n_heads [[buffer (2)]],
                   constant uint &head_dim [[buffer (3)]],
                   constant uint &n_chunks [[buffer (4)]],
                   constant uint &max_chunks [[buffer (5)]],
                   uint head [[threadgroup_position_in_grid]],
                   uint d [[thread_position_in_threadgroup]])
{
  device const float *chunk_max = scratch + n_heads * max_chunks * head_dim;
  device const float *chunk_sum = chunk_max + n_heads * max_chunks;
  uint base = head * max_chunks;

  float best = -INFINITY;
  for (uint c = 0; c < n_chunks; c++)
    best = max (best, chunk_max[base + c]);

  float total = 0.0f;
  float acc = 0.0f;
  for (uint c = 0; c < n_chunks; c++)
    {
      float rescale = precise::exp (chunk_max[base + c] - best);
      total += chunk_sum[base + c] * rescale;
      acc += scratch[(base + c) * head_dim + d] * rescale;
    }
  out[head * head_dim + d] = acc / total;
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

/* Copy the N floats at SRC to DST.  */
kernel void
copy_floats (device const float *src [[buffer (0)]],
             device float *dst [[buffer (1)]], constant uint &n [[buffer (2)]],
             uint i [[thread_position_in_grid]])
{
  if (i < n)
    dst[i] = src[i];
}

/* Dequantize row TOKEN[0] of the Q8_0 matrix WEIGHTS, whose rows hold
   N_EMBD elements, into the N_EMBD floats at OUT.  One thread handles
   one element.  */
kernel void
embed_q8_0 (device const uchar *weights [[buffer (0)]],
            device const int *token [[buffer (1)]],
            device float *out [[buffer (2)]],
            constant uint &n_embd [[buffer (3)]],
            uint i [[thread_position_in_grid]])
{
  if (i >= n_embd)
    return;
  ulong row_bytes = ulong (n_embd / QK8_0) * Q8_0_BLOCK_BYTES;
  device const q8_0_block *block
      = (device const q8_0_block *)(weights + ulong (token[0]) * row_bytes)
        + i / QK8_0;
  out[i] = float (block->scale) * float (block->quants[i % QK8_0]);
}

/* Store at OUT the index of the largest of the N floats at X.  Ties go
   to the lowest index, as a sequential scan would choose.  One
   threadgroup handles the whole vector.  */
kernel void
argmax (device const float *x [[buffer (0)]], device int *out [[buffer (1)]],
        constant uint &n [[buffer (2)]],
        uint tid [[thread_position_in_threadgroup]],
        uint threads [[threads_per_threadgroup]],
        uint simdgroup_index [[simdgroup_index_in_threadgroup]],
        uint simdgroups [[simdgroups_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float best_values[MAX_SIMDGROUPS];
  threadgroup uint best_indices[MAX_SIMDGROUPS];

  /* Each thread scans a strided slice, so its first maximum is also its
     lowest-index maximum.  */
  float best = -INFINITY;
  uint best_index = 0;
  for (uint i = tid; i < n; i += threads)
    if (x[i] > best)
      {
        best = x[i];
        best_index = i;
      }

  for (uint offset = SIMD_WIDTH / 2; offset > 0; offset /= 2)
    {
      float other = simd_shuffle_down (best, offset);
      uint other_index = simd_shuffle_down (best_index, offset);
      if (other > best || (other == best && other_index < best_index))
        {
          best = other;
          best_index = other_index;
        }
    }
  if (lane == 0)
    {
      best_values[simdgroup_index] = best;
      best_indices[simdgroup_index] = best_index;
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);

  if (tid == 0)
    {
      for (uint s = 1; s < simdgroups; s++)
        if (best_values[s] > best
            || (best_values[s] == best && best_indices[s] < best_index))
          {
            best = best_values[s];
            best_index = best_indices[s];
          }
      out[0] = int (best_index);
    }
}
