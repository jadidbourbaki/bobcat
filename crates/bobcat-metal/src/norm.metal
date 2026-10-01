/* Normalization, positional rotation, conversion, embedding, and
   indexing kernels.  */

/* Normalize each of the rows of N floats at X by its root mean square,
   scale it by the N floats at WEIGHT, and store the result at the
   matching row of OUT.  One threadgroup handles one row.  */
kernel void
rms_norm (device const float *x [[buffer (0)]],
          device const float *weight [[buffer (1)]],
          device float *out [[buffer (2)]], constant uint &n [[buffer (3)]],
          constant float &eps [[buffer (4)]],
          uint row [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]],
          uint threads [[threads_per_threadgroup]],
          uint simdgroup_index [[simdgroup_index_in_threadgroup]],
          uint simdgroups [[simdgroups_per_threadgroup]],
          uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  device const float *xr = x + row * n;
  device float *outr = out + row * n;
  float sum_squares = 0.0f;

  for (uint i = tid; i < n; i += threads)
    sum_squares += xr[i] * xr[i];
  sum_squares = threadgroup_sum (sum_squares, partials, simdgroup_index,
                                 simdgroups, lane);

  float scale = precise::rsqrt (sum_squares / float (n) + eps);
  for (uint i = tid; i < n; i += threads)
    outr[i] = weight[i] * (xr[i] * scale);
}

/* Normalize each of the heads of HEAD_DIM floats of each token at SRC by
   its root mean square, scale it by the HEAD_DIM floats at WEIGHT,
   rotate its first N_ROT elements for the token's position, and store it
   at DST as T.  Token I sits at position POS + I, SRC_STRIDE floats into
   SRC and DST_STRIDE elements into DST.  Head H of a token starts H
   SRC_HEAD_STRIDE floats into the token at SRC and H HEAD_DIM elements
   into it at DST.  Element I of the rotated elements pairs with element
   I + N_ROT / 2.  One threadgroup of HEAD_DIM threads handles one head of
   one token.  SRC and DST may be the same floats.  */
template <typename T>
kernel void
norm_rope (device const float *src [[buffer (0)]],
           device T *dst [[buffer (1)]],
           device const float *weight [[buffer (2)]],
           constant uint &head_dim [[buffer (3)]],
           constant uint &pos [[buffer (4)]],
           constant float &theta [[buffer (5)]],
           constant float &eps [[buffer (6)]],
           constant uint &src_stride [[buffer (7)]],
           constant uint &dst_stride [[buffer (8)]],
           constant uint &n_rot [[buffer (9)]],
           constant uint &src_head_stride [[buffer (10)]],
           uint2 position [[threadgroup_position_in_grid]],
           uint2 thread_position [[thread_position_in_threadgroup]],
           uint simdgroup_index [[simdgroup_index_in_threadgroup]],
           uint simdgroups [[simdgroups_per_threadgroup]],
           uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[MAX_SIMDGROUPS];
  threadgroup float normed[MAX_HEAD_DIM];
  uint head = position.x;
  uint token = position.y;
  uint i = thread_position.x;

  float value = src[token * src_stride + head * src_head_stride + i];
  float sum_squares = threadgroup_sum (value * value, partials,
                                       simdgroup_index, simdgroups, lane);
  float scale = precise::rsqrt (sum_squares / float (head_dim) + eps);
  normed[i] = weight[i] * (value * scale);
  threadgroup_barrier (mem_flags::mem_threadgroup);

  uint half_rot = n_rot / 2;
  device T *out = dst + token * dst_stride + head * head_dim;
  if (i < half_rot)
    {
      float inv_freq
          = 1.0f / precise::pow (theta, float (2 * i) / float (n_rot));
      float angle = float (pos + token) * inv_freq;
      float c = precise::cos (angle);
      float s = precise::sin (angle);
      float x0 = normed[i];
      float x1 = normed[i + half_rot];
      out[i] = T (x0 * c - x1 * s);
      out[i + half_rot] = T (x1 * c + x0 * s);
    }
  else if (i >= n_rot)
    out[i] = T (normed[i]);
}

/* The element type appears in the signature, so each instantiation
   names its own type.  */
template [[host_name (
    "norm_rope_f32")]] kernel decltype (norm_rope<float>) norm_rope<float>;
template [[host_name (
    "norm_rope_f16")]] kernel decltype (norm_rope<half>) norm_rope<half>;

/* Convert the N floats at SRC to half precision at DST.  */
kernel void
convert_half (device const float *src [[buffer (0)]],
              device half *dst [[buffer (1)]], constant uint &n [[buffer (2)]],
              uint i [[thread_position_in_grid]])
{
  if (i < n)
    dst[i] = half (src[i]);
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

/* Dequantize row TOKENS[T] of the Q8_0 matrix WEIGHTS, whose rows hold
   N_EMBD elements, into row T of OUT for each token T.  One thread
   handles one element of one token.  */
kernel void
embed_q8_0 (device const uchar *weights [[buffer (0)]],
            device const int *tokens [[buffer (1)]],
            device float *out [[buffer (2)]],
            constant uint &n_embd [[buffer (3)]],
            uint2 position [[thread_position_in_grid]])
{
  uint i = position.x;
  uint t = position.y;
  if (i >= n_embd)
    return;
  ulong row_bytes = ulong (n_embd / QK8_0) * Q8_0_BLOCK_BYTES;
  device const q8_0_block *block
      = (device const q8_0_block *)(weights + ulong (tokens[t]) * row_bytes)
        + i / QK8_0;
  out[t * n_embd + i]
      = float (block->scale) * float (block->quants[i % QK8_0]);
}

/* Store at OUT and at COPY the index of the largest of the N floats at
   X.  Ties go to the lowest index, as a sequential scan would choose.
   One threadgroup handles the whole vector.  */
kernel void
argmax (device const float *x [[buffer (0)]], device int *out [[buffer (1)]],
        constant uint &n [[buffer (2)]], device int *copy [[buffer (3)]],
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
      copy[0] = int (best_index);
    }
}
/* Dequantize row TOKENS[T] of the matrix WEIGHTS of format F, whose rows
   hold N_EMBD weights, into row T of OUT for each token T.  One thread
   handles 8 weights of one token.  */
template <typename F>
kernel void
embed_q (device const uchar *weights [[buffer (0)]],
         device const int *tokens [[buffer (1)]],
         device float *out [[buffer (2)]],
         constant uint &n_embd [[buffer (3)]],
         uint2 position [[thread_position_in_grid]])
{
  uint e = position.x * 8;
  uint t = position.y;
  if (e >= n_embd)
    return;
  device const uchar *row
      = weights + ulong (tokens[t]) * row_bytes_of<F> (n_embd);
  weights8 w = F::load8 (row, e);
  device float4 *dst = (device float4 *)(out + ulong (t) * n_embd + e);
  dst[0] = w.low;
  dst[1] = w.high;
}

template [[host_name ("embed_q4_0")]] kernel decltype (embed_q<q4_0_format>)
    embed_q<q4_0_format>;
template [[host_name (
    "embed_f16")]] kernel decltype (embed_q<f16_format>) embed_q<f16_format>;
template [[host_name (
    "embed_q4k")]] kernel decltype (embed_q<q4k_format>) embed_q<q4k_format>;
template [[host_name (
    "embed_q5k")]] kernel decltype (embed_q<q5k_format>) embed_q<q5k_format>;
template [[host_name (
    "embed_q6k")]] kernel decltype (embed_q<q6k_format>) embed_q<q6k_format>;
