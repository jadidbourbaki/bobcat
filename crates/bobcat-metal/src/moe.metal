/* Mixture-of-experts kernels: the router and the matrix-vector products
   of the experts it picks.

   The route of a token holds the N_USED experts the router picked, then
   their N_USED weights as float bits.  Token T's route starts at
   2 N_USED T.  */

/* Route each token of X, which holds N_COLS floats per token, into
   ROUTE.  The router scores each of the N_EXPERTS experts with the
   sigmoid of the dot product of its row of the F32 matrix ROUTER with
   the token.  It picks the N_USED experts whose score plus BIAS is
   largest, with ties to the lower expert.  A picked expert's weight is
   its score divided by the sum of the picked scores plus 1e-6, as in
   transformers.  One threadgroup of MOE_ROUTE_SIMDGROUPS simdgroups
   routes MOE_ROUTE_TOKENS tokens of the N_TOKENS, so each router row it
   reads serves them all.

   When fuse_norm is set, the kernel first RMS-normalizes each token as
   rms_norm does, with NORM_WEIGHT and EPS, stores the result at NORMED
   for the experts, and routes the normalized token.  */
kernel void
moe_route (device const float *x [[buffer (0)]],
           device const float *router [[buffer (1)]],
           device const float *bias [[buffer (2)]],
           device uint *route [[buffer (3)]],
           constant uint &n_cols [[buffer (4)]],
           constant uint &n_experts [[buffer (5)]],
           constant uint &n_used [[buffer (6)]],
           constant uint &n_tokens [[buffer (7)]],
           device const float *norm_weight
           [[buffer (8), function_constant (fuse_norm)]],
           constant float &eps [[buffer (9), function_constant (fuse_norm)]],
           device float *normed [[buffer (10), function_constant (fuse_norm)]],
           uint threadgroup_index [[threadgroup_position_in_grid]],
           uint tid [[thread_index_in_threadgroup]],
           uint simdgroup_index [[simdgroup_index_in_threadgroup]],
           uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float scores[MOE_ROUTE_TOKENS][MOE_MAX_EXPERTS];
  threadgroup float biased[MOE_ROUTE_TOKENS][MOE_MAX_EXPERTS];
  threadgroup float partials[MAX_SIMDGROUPS];

  uint first_token = threadgroup_index * MOE_ROUTE_TOKENS;
  uint count = min (uint (MOE_ROUTE_TOKENS), n_tokens - first_token);
  device const float *rows = x;
  if (fuse_norm)
    {
      uint threads = MOE_ROUTE_SIMDGROUPS * SIMD_WIDTH;
      for (uint t = 0; t < count; t++)
        {
          device const float *row = x + ulong (first_token + t) * n_cols;
          device float *out = normed + ulong (first_token + t) * n_cols;
          float sum_squares = 0.0f;
          for (uint i = tid; i < n_cols; i += threads)
            sum_squares += row[i] * row[i];
          sum_squares
              = threadgroup_sum (sum_squares, partials, simdgroup_index,
                                 MOE_ROUTE_SIMDGROUPS, lane);
          float scale = precise::rsqrt (sum_squares / float (n_cols) + eps);
          for (uint i = tid; i < n_cols; i += threads)
            out[i] = norm_weight[i] * (row[i] * scale);
        }
      /* The dot products below read normalized elements other threads
         stored.  */
      threadgroup_barrier (mem_flags::mem_device);
      rows = normed;
    }
  device const float4 *inputs
      = (device const float4 *)(rows + ulong (first_token) * n_cols);
  uint vectors = n_cols / 4;
  for (uint e = simdgroup_index; e < n_experts; e += MOE_ROUTE_SIMDGROUPS)
    {
      device const float4 *row
          = (device const float4 *)(router + ulong (e) * n_cols);
      float sums[MOE_ROUTE_TOKENS] = { 0.0f };
      for (uint i = lane; i < vectors; i += SIMD_WIDTH)
        {
          float4 weights = row[i];
          for (uint t = 0; t < count; t++)
            sums[t] += dot (weights, inputs[t * vectors + i]);
        }
      for (uint t = 0; t < count; t++)
        {
          float sum = simd_sum (sums[t]);
          if (lane == 0)
            {
              float score = 1.0f / (1.0f + precise::exp (-sum));
              scores[t][e] = score;
              biased[t][e] = score + bias[e];
            }
        }
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);
  if (tid >= count)
    return;

  /* Thread T picks the experts of token T of the threadgroup.  */
  uint token = first_token + tid;
  threadgroup float *token_scores = scores[tid];
  threadgroup float *token_biased = biased[tid];
  device uint *token_route = route + 2 * n_used * token;
  uint picked[MOE_MAX_USED];
  float total = 0.0f;
  for (uint k = 0; k < n_used; k++)
    {
      uint best = 0;
      for (uint e = 1; e < n_experts; e++)
        if (token_biased[e] > token_biased[best])
          best = e;
      picked[k] = best;
      total += token_scores[best];
      token_biased[best] = -INFINITY;
    }
  float norm = total + 1e-6f;
  for (uint k = 0; k < n_used; k++)
    {
      token_route[k] = picked[k];
      token_route[n_used + k]
          = as_type<uint> (token_scores[picked[k]] / norm);
    }
}

/* Group the routes of N_TOKENS tokens by expert for matmul_experts.
   Entry V = N_USED T + S stands for slot S of token T.  Expert E's
   entries go to ENTRIES[OFFSETS[E]] onward in token order, and
   OFFSETS[N_EXPERTS] receives the total.  Simdgroup E of one
   threadgroup gathers experts E, E + MOE_GROUP_SIMDGROUPS, and so on.
   Its lanes test 32 consecutive entries at a time, and a prefix sum
   places the matches in order.  */
kernel void
moe_group (device const uint *route [[buffer (0)]],
           device uint *offsets [[buffer (1)]],
           device uint *entries [[buffer (2)]],
           constant uint &n_experts [[buffer (3)]],
           constant uint &n_used [[buffer (4)]],
           constant uint &n_tokens [[buffer (5)]],
           uint simdgroup_index [[simdgroup_index_in_threadgroup]],
           uint lane [[thread_index_in_simdgroup]])
{
  threadgroup uint counts[MOE_MAX_EXPERTS];
  uint n_entries = n_used * n_tokens;
  for (uint e = simdgroup_index; e < n_experts; e += MOE_GROUP_SIMDGROUPS)
    {
      uint count = 0;
      for (uint v = lane; v < n_entries; v += SIMD_WIDTH)
        count += route[2 * n_used * (v / n_used) + v % n_used] == e;
      count = simd_sum (count);
      if (lane == 0)
        counts[e] = count;
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);

  for (uint e = simdgroup_index; e < n_experts; e += MOE_GROUP_SIMDGROUPS)
    {
      uint start = 0;
      for (uint before = 0; before < e; before++)
        start += counts[before];
      if (lane == 0)
        {
          offsets[e] = start;
          if (e == n_experts - 1)
            offsets[n_experts] = start + counts[e];
        }
      for (uint first = 0; first < n_entries; first += SIMD_WIDTH)
        {
          uint v = first + lane;
          bool match = v < n_entries
                       && route[2 * n_used * (v / n_used) + v % n_used] == e;
          uint position = start + simd_prefix_exclusive_sum (uint (match));
          if (match)
            entries[position] = v;
          start += simd_sum (uint (match));
        }
    }
}

/* Add to each token's N_EMBD floats at Y the sum of its routed experts'
   outputs at X, each times its weight in the token's route.  Row
   N_USED T + S of X holds the output of slot S of token T.  One thread
   computes one element.  */
kernel void
moe_combine (device const float *x [[buffer (0)]],
             device const uint *route [[buffer (1)]],
             device float *y [[buffer (2)]],
             constant uint &n_embd [[buffer (3)]],
             constant uint &n_used [[buffer (4)]],
             constant uint &n_tokens [[buffer (5)]],
             uint index [[thread_position_in_grid]])
{
  if (index >= n_tokens * n_embd)
    return;
  uint token = index / n_embd;
  uint d = index % n_embd;
  device const uint *token_route = route + 2 * n_used * token;
  float sum = 0.0f;
  for (uint s = 0; s < n_used; s++)
    sum += as_type<float> (token_route[n_used + s])
           * x[(ulong (token) * n_used + s) * n_embd + d];
  y[index] += sum;
}

/* Run the SwiGLU of each expert in each token's route, as matvec_k_swiglu
   does.  GATE and UP stack the experts, with N_ROWS rows of N_COLS
   weights each.  Threadgroup (I, S, T) computes rows I
   K_ROWS_PER_SIMDGROUP onward of the expert in slot S of token T's
   route, on token T's N_COLS floats at X.  The results of slot S of
   token T go to row N_USED T + S of Y, which holds N_ROWS floats per
   row.  */
template <typename K>
kernel void
matvec_experts_swiglu (device const uchar *gate [[buffer (0)]],
                       device const uchar *up [[buffer (1)]],
                       device const float *x [[buffer (2)]],
                       device const uint *route [[buffer (3)]],
                       device float *y [[buffer (4)]],
                       constant uint &n_rows [[buffer (5)]],
                       constant uint &n_cols [[buffer (6)]],
                       constant uint &n_used [[buffer (7)]],
                       uint3 position [[threadgroup_position_in_grid]],
                       uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                       uint simdgroups [[simdgroups_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[(2 * K_ROWS_PER_SIMDGROUP + 1) * MAX_SIMDGROUPS];
  uint slot = position.y;
  uint token = position.z;
  ulong expert_bytes = ulong (n_rows) * (n_cols / K::block_weights)
                       * K::block_bytes;
  ulong offset = route[2 * n_used * token + slot] * expert_bytes;
  device const float *input = x + ulong (token) * n_cols;
  swiglu_rows<K> (gate + offset, up + offset, input,
                  y + (ulong (token) * n_used + slot) * n_rows, n_rows,
                  n_cols, input, 0.0f, partials, position.x,
                  simdgroup_index, simdgroups, lane);
}

/* Add to each token's N_ROWS floats at Y the weighted sum of the down
   projections of the experts in its route.  WEIGHTS stacks the experts,
   with N_ROWS rows of N_COLS weights of K-quant format K each.  The
   input of slot S of token T is row N_USED T + S of X, which holds
   N_COLS floats per row.  Threadgroup (I, T) computes rows I
   K_ROWS_PER_SIMDGROUP onward for token T, so each output has one
   writer.  The simdgroups split the columns, as in matvec_k.  */
template <typename K>
kernel void
matvec_experts_down (device const uchar *weights [[buffer (0)]],
                     device const float *x [[buffer (1)]],
                     device const uint *route [[buffer (2)]],
                     device float *y [[buffer (3)]],
                     constant uint &n_rows [[buffer (4)]],
                     constant uint &n_cols [[buffer (5)]],
                     constant uint &n_used [[buffer (6)]],
                     uint2 position [[threadgroup_position_in_grid]],
                     uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                     uint simdgroups [[simdgroups_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[K_ROWS_PER_SIMDGROUP * MAX_SIMDGROUPS];
  uint first_row = position.x * K_ROWS_PER_SIMDGROUP;
  uint token = position.y;
  device const uint *token_route = route + 2 * n_used * token;
  uint n_blocks = n_cols / K::block_weights;
  ulong row_bytes = ulong (n_blocks) * K::block_bytes;
  float sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  /* fuse_norm is false for these kernels, so the loads leave
     SUM_SQUARES alone.  */
  float sum_squares = 0.0f;

  /* Each slot gets SIMDGROUPS / N_USED simdgroups, which split its
     columns, so the slots' experts stream in parallel.  */
  uint split = simdgroups / n_used;
  uint slot = simdgroup_index / split;
  uint part = simdgroup_index % split;
  device const uchar *expert
      = weights + token_route[slot] * ulong (n_rows) * row_bytes;
  float weight = as_type<float> (token_route[n_used + slot]);
  device const float *input = x + (ulong (token) * n_used + slot) * n_cols;
  for (uint block = K::first_block (lane) + part * K::blocks_per_pass;
       block < n_blocks; block += K::blocks_per_pass * split)
    {
      typename K::inputs in;
      K::load (input, input, block, lane, in, sum_squares);
      for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
        if (first_row + r < n_rows)
          sums[r] += K::dot_part (expert + (first_row + r) * row_bytes, block,
                                  lane, in);
    }
  for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
    sums[r] *= weight;

  for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
    {
      float part = simd_sum (sums[r]);
      if (lane == 0)
        partials[r * MAX_SIMDGROUPS + simdgroup_index] = part;
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);
  if (simdgroup_index != 0 || lane >= K_ROWS_PER_SIMDGROUP)
    return;
  uint row = first_row + lane;
  if (row >= n_rows)
    return;
  float total = 0.0f;
  for (uint s = 0; s < simdgroups; s++)
    total += partials[lane * MAX_SIMDGROUPS + s];
  y[ulong (token) * n_rows + row] += total;
}

template [[host_name ("matvec_experts_swiglu_q4k")]] kernel decltype (
    matvec_experts_swiglu<q4k_lanes>) matvec_experts_swiglu<q4k_lanes>;
template [[host_name ("matvec_experts_swiglu_q6k")]] kernel decltype (
    matvec_experts_swiglu<q6k_lanes>) matvec_experts_swiglu<q6k_lanes>;
template [[host_name ("matvec_experts_swiglu_q4_0")]] kernel decltype (
    matvec_experts_swiglu<q4_0_lanes>) matvec_experts_swiglu<q4_0_lanes>;
template [[host_name ("matvec_experts_down_q4k")]] kernel decltype (
    matvec_experts_down<q4k_lanes>) matvec_experts_down<q4k_lanes>;
template [[host_name ("matvec_experts_down_q6k")]] kernel decltype (
    matvec_experts_down<q6k_lanes>) matvec_experts_down<q6k_lanes>;
template [[host_name ("matvec_experts_down_q4_0")]] kernel decltype (
    matvec_experts_down<q4_0_lanes>) matvec_experts_down<q4_0_lanes>;
