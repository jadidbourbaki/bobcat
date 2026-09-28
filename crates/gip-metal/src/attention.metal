/* Chunked attention kernels and their scratch layout.  */

/* Return the floats of attention scratch one query uses for N_HEADS
   heads of HEAD_DIM floats and MAX_CHUNKS chunks.  */
static uint
scratch_per_query (uint n_heads, uint head_dim, uint max_chunks)
{
  return n_heads * max_chunks * (head_dim + 2);
}

/* Attend with the query heads of each query at Q over one chunk of
   ATTENTION_CHUNK positions of K_CACHE and V_CACHE, as the first of two
   passes.  Query I sits at position FIRST_POS + I and sees the keys at
   positions 0 through FIRST_POS + I, which makes the attention causal.
   Query heads share KV heads in consecutive groups of N_HEADS /
   N_KV_HEADS, at most ATTENTION_MAX_GROUP.

   One threadgroup of ATTENTION_CHUNK threads handles one KV head, every
   query head in its group, one chunk, and one query.  Each thread scores
   one position, so K and V are read once per chunk.  For each query head
   the pass stores the chunk's largest score, its sum of exponentials
   relative to that score, and its unnormalized weighted sum of values in
   the query's SCRATCH, laid out as attention_combine expects for
   MAX_CHUNKS chunks.  The caches hold elements of type T.  */
template <typename T>
kernel void
attention_chunk (device const float *q [[buffer (0)]],
                 device const T *k_cache [[buffer (1)]],
                 device const T *v_cache [[buffer (2)]],
                 device float *scratch [[buffer (3)]],
                 constant uint &n_heads [[buffer (4)]],
                 constant uint &n_kv_heads [[buffer (5)]],
                 constant uint &head_dim [[buffer (6)]],
                 constant uint &first_pos [[buffer (7)]],
                 constant uint &max_chunks [[buffer (8)]],
                 constant float &scale [[buffer (9)]],
                 uint3 position [[threadgroup_position_in_grid]],
                 uint3 thread_position [[thread_position_in_threadgroup]],
                 uint3 threadgroup_size [[threads_per_threadgroup]],
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
  uint query_index = position.z;
  uint n_keys = first_pos + query_index + 1;
  uint start = chunk * ATTENTION_CHUNK;

  /* The grid covers the chunks of the last query.  Earlier queries see
     fewer keys and skip the chunks past them.  */
  if (start >= n_keys)
    return;

  uint group = n_heads / n_kv_heads;
  uint kv_dim = n_kv_heads * head_dim;
  uint first_head = kv_head * group;
  uint count = min (uint (ATTENTION_CHUNK), n_keys - start);
  q += query_index * n_heads * head_dim;
  scratch += query_index * scratch_per_query (n_heads, head_dim, max_chunks);

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
      float total
          = threadgroup_sum (e, partials, simdgroup_index, simdgroups, lane);
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

template [[host_name (
    "attention_chunk_f32")]] kernel decltype (attention_chunk<float>)
    attention_chunk<float>;
template [[host_name (
    "attention_chunk_f16")]] kernel decltype (attention_chunk<half>)
    attention_chunk<half>;

/* Combine the chunk results of attention_chunk in SCRATCH into the
   result of each query head of each query at OUT, as the second of two
   passes.  Query I sits at position FIRST_POS + I.  Each chunk's sums
   are rescaled from its own largest score to the largest score of all
   chunks.  One threadgroup of HEAD_DIM threads handles one query head of
   one query.  */
kernel void
attention_combine (device const float *scratch [[buffer (0)]],
                   device float *out [[buffer (1)]],
                   constant uint &n_heads [[buffer (2)]],
                   constant uint &head_dim [[buffer (3)]],
                   constant uint &first_pos [[buffer (4)]],
                   constant uint &max_chunks [[buffer (5)]],
                   uint2 position [[threadgroup_position_in_grid]],
                   uint2 thread_position [[thread_position_in_threadgroup]])
{
  uint head = position.x;
  uint query_index = position.y;
  uint d = thread_position.x;
  uint n_keys = first_pos + query_index + 1;
  uint n_chunks = (n_keys + ATTENTION_CHUNK - 1) / ATTENTION_CHUNK;
  scratch += query_index * scratch_per_query (n_heads, head_dim, max_chunks);
  out += query_index * n_heads * head_dim;

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
