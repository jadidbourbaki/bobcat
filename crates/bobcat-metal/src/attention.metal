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

/* Attend with FLASH_QUERIES queries of one query head over every key they
   see, in one pass with the online softmax of FlashAttention.  Query I
   of the N_QUERIES at Q sits at position FIRST_POS + I, sees the keys at
   positions 0 through FIRST_POS + I, and writes its result to OUT in the
   layout of Q.  Simdgroup matrices compute the scores of 8 queries by
   FLASH_KEYS keys and their weighted sum of values.  A diagonal matrix of
   per-query factors rescales the running sums whenever a query's largest
   score grows.  The caches hold elements of type T.  */
template <typename T>
kernel void
attention_flash (device const float *q [[buffer (0)]],
                 device const T *k_cache [[buffer (1)]],
                 device const T *v_cache [[buffer (2)]],
                 device float *out [[buffer (3)]],
                 constant uint &n_heads [[buffer (4)]],
                 constant uint &n_kv_heads [[buffer (5)]],
                 constant uint &head_dim [[buffer (6)]],
                 constant uint &first_pos [[buffer (7)]],
                 constant uint &n_queries [[buffer (8)]],
                 constant float &scale [[buffer (9)]],
                 uint2 position [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]])
{
  /* The query tile holds the queries before the key loop and the results
     after it, so it shares storage with the key and value tiles.  Less
     threadgroup memory lets more threadgroups share a GPU core.  */
  constexpr uint query_bytes = FLASH_QUERIES * FLASH_HEAD_DIM * sizeof (float);
  constexpr uint key_bytes = FLASH_KEYS * FLASH_HEAD_DIM * sizeof (T);
  threadgroup uchar
      tiles[query_bytes > 2 * key_bytes ? query_bytes : 2 * key_bytes];
  threadgroup float *query_tile = (threadgroup float *)tiles;
  threadgroup T *key_tile = (threadgroup T *)tiles;
  threadgroup T *value_tile = (threadgroup T *)(tiles + key_bytes);
  threadgroup float score_tiles[FLASH_SIMDGROUPS * 8 * FLASH_KEYS];
  threadgroup float diagonals[FLASH_SIMDGROUPS * 64];

  uint first_query = position.x * FLASH_QUERIES;
  uint head = position.y;
  uint kv_head = head / (n_heads / n_kv_heads);
  uint q_stride = n_heads * head_dim;
  uint kv_dim = n_kv_heads * head_dim;
  uint n_rows = min (uint (FLASH_QUERIES), n_queries - first_query);
  uint threads = FLASH_SIMDGROUPS * SIMD_WIDTH;
  /* Each thread loads 4 consecutive elements of a row.  The host passes
     only head sizes that are powers of two, so the threads split evenly
     over the rows.  */
  uint row_vectors = head_dim / 4;
  uint load_key = tid / row_vectors;
  uint load_column = 4 * (tid % row_vectors);
  uint load_keys = threads / row_vectors;

  /* Rows past the last query hold zeros, and their results are
     discarded.  */
  for (uint row = load_key; row < FLASH_QUERIES; row += load_keys)
    {
      float4 value = 0.0f;
      if (row < n_rows)
        value = *(device const float4 *)(q + (first_query + row) * q_stride
                                         + head * head_dim + load_column)
                * scale;
      *(threadgroup float4 *)(query_tile + row * head_dim + load_column)
          = value;
    }
  threadgroup float *diagonal = diagonals + simdgroup_index * 64;
  for (uint i = lane; i < 64; i += SIMD_WIDTH)
    diagonal[i] = 0.0f;
  threadgroup_barrier (mem_flags::mem_threadgroup);

  /* A constant bound lets the compiler unroll the loops over the head,
     which keeps the arrays of simdgroup matrices in registers.  */
  constexpr uint d_blocks = FLASH_HEAD_DIM / 8;
  threadgroup float *own_queries = query_tile + simdgroup_index * 8 * head_dim;
  simdgroup_float8x8 queries[d_blocks];
  simdgroup_float8x8 sums[d_blocks];
  for (uint d = 0; d < d_blocks; d++)
    {
      simdgroup_load (queries[d], own_queries + d * 8, head_dim);
      sums[d] = make_filled_simdgroup_matrix<float, 8, 8> (0.0f);
    }

  /* Four lanes share each of the simdgroup's 8 queries, with 8 keys of
     the block each.  */
  uint row = lane / 4;
  uint part = lane % 4;
  uint query_pos = first_pos + first_query + simdgroup_index * 8 + row;
  threadgroup float *scores = score_tiles + simdgroup_index * 8 * FLASH_KEYS;
  threadgroup float *row_scores = scores + row * FLASH_KEYS + part * 8;
  float best = -INFINITY;
  float total = 0.0f;
  uint n_keys = first_pos + first_query + n_rows;
  for (uint start = 0; start < n_keys; start += FLASH_KEYS)
    {
      uint count = min (uint (FLASH_KEYS), n_keys - start);
      threadgroup_barrier (mem_flags::mem_threadgroup);
      for (uint key = load_key; key < FLASH_KEYS; key += load_keys)
        {
          uint element = key * head_dim + load_column;
          ulong source = ulong (start + key) * kv_dim + kv_head * head_dim
                         + load_column;
          bool valid = key < count;
          *(threadgroup vec<T, 4> *)(key_tile + element)
              = valid ? *(device const vec<T, 4> *)(k_cache + source)
                      : vec<T, 4> (0);
          *(threadgroup vec<T, 4> *)(value_tile + element)
              = valid ? *(device const vec<T, 4> *)(v_cache + source)
                      : vec<T, 4> (0);
        }
      threadgroup_barrier (mem_flags::mem_threadgroup);

      for (uint c = 0; c < FLASH_KEYS / 8; c++)
        {
          simdgroup_float8x8 block
              = make_filled_simdgroup_matrix<float, 8, 8> (0.0f);
          for (uint d = 0; d < d_blocks; d++)
            {
              simdgroup_matrix<T, 8, 8> keys;
              simdgroup_load (keys, key_tile + c * 8 * head_dim + d * 8,
                              head_dim, ulong2 (0, 0), true);
              simdgroup_multiply_accumulate (block, queries[d], keys, block);
            }
          simdgroup_store (block, scores + c * 8, FLASH_KEYS);
        }
      simdgroup_barrier (mem_flags::mem_threadgroup);

      float values[8];
      float block_best = -INFINITY;
      for (uint j = 0; j < 8; j++)
        {
          uint key = start + part * 8 + j;
          values[j]
              = key <= query_pos && key < n_keys ? row_scores[j] : -INFINITY;
          block_best = max (block_best, values[j]);
        }
      block_best = max (block_best, simd_shuffle_xor (block_best, 1));
      block_best = max (block_best, simd_shuffle_xor (block_best, 2));
      /* Key 0 is in every query's first block, so NEW_BEST is finite.  */
      float new_best = max (best, block_best);
      float rescale = precise::exp (best - new_best);
      float block_total = 0.0f;
      for (uint j = 0; j < 8; j++)
        {
          float e = precise::exp (values[j] - new_best);
          row_scores[j] = e;
          block_total += e;
        }
      block_total += simd_shuffle_xor (block_total, 1);
      block_total += simd_shuffle_xor (block_total, 2);
      total = total * rescale + block_total;
      best = new_best;
      if (part == 0)
        diagonal[row * 9] = rescale;
      simdgroup_barrier (mem_flags::mem_threadgroup);

      simdgroup_float8x8 factors;
      simdgroup_load (factors, diagonal, 8);
      for (uint d = 0; d < d_blocks; d++)
        {
          simdgroup_float8x8 scaled;
          simdgroup_multiply (scaled, factors, sums[d]);
          sums[d] = scaled;
        }
      for (uint c = 0; c < FLASH_KEYS / 8; c++)
        {
          simdgroup_float8x8 weights;
          simdgroup_load (weights, scores + c * 8, FLASH_KEYS);
          for (uint d = 0; d < d_blocks; d++)
            {
              simdgroup_matrix<T, 8, 8> block;
              simdgroup_load (block, value_tile + c * 8 * head_dim + d * 8,
                              head_dim);
              simdgroup_multiply_accumulate (sums[d], weights, block, sums[d]);
            }
        }
    }

  /* Each simdgroup reuses its own rows of the query tile for its
     normalized results, once every simdgroup has read the last value
     tile.  */
  if (part == 0)
    diagonal[row * 9] = 1.0f / total;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  simdgroup_float8x8 factors;
  simdgroup_load (factors, diagonal, 8);
  for (uint d = 0; d < d_blocks; d++)
    {
      simdgroup_float8x8 result;
      simdgroup_multiply (result, factors, sums[d]);
      simdgroup_store (result, own_queries + d * 8, head_dim);
    }
  simdgroup_barrier (mem_flags::mem_threadgroup);
  for (uint i = lane; i < 8 * head_dim; i += SIMD_WIDTH)
    {
      uint tile_row = simdgroup_index * 8 + i / head_dim;
      if (tile_row < n_rows)
        out[(first_query + tile_row) * q_stride + head * head_dim
            + i % head_dim] = own_queries[i];
    }
}

template [[host_name (
    "attention_flash_f32")]] kernel decltype (attention_flash<float>)
    attention_flash<float>;
template [[host_name (
    "attention_flash_f16")]] kernel decltype (attention_flash<half>)
    attention_flash<half>;

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
