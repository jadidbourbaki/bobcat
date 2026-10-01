/* Gated DeltaNet kernels and the gates of Qwen3.5's layers.

   A DeltaNet token holds CONV_DIM floats after its input projection:
   N_K_HEADS query heads and N_K_HEADS key heads of K_DIM floats, then
   N_V_HEADS value heads of V_DIM floats.  Value head J reads key head
   J % N_K_HEADS, the order llama.cpp's converter gives the heads.  */

/* Run the causal convolution of N_TOKENS tokens of CHANNELS floats at X
   with KERNEL_SIZE taps per channel at TAPS, the oldest first, and store
   SiLU of each result at Y.  HISTORY holds the latest KERNEL_SIZE - 1
   raw inputs of each channel, the oldest first, and moves forward one
   token at a time.  Thread C handles channel C.  */
kernel void
gdn_conv (device const float *x [[buffer (0)]],
          device const float *taps [[buffer (1)]],
          device float *history [[buffer (2)]],
          device float *y [[buffer (3)]],
          constant uint &channels [[buffer (4)]],
          constant uint &kernel_size [[buffer (5)]],
          constant uint &n_tokens [[buffer (6)]],
          uint c [[thread_position_in_grid]])
{
  if (c >= channels)
    return;
  uint past = kernel_size - 1;
  device float *h = history + c * past;
  device const float *w = taps + c * kernel_size;
  float window[GDN_MAX_KERNEL];
  for (uint i = 0; i < past; i++)
    window[i] = h[i];
  for (uint t = 0; t < n_tokens; t++)
    {
      float input = x[t * channels + c];
      float sum = w[past] * input;
      for (uint i = 0; i < past; i++)
        sum += w[i] * window[i];
      y[t * channels + c] = sum / (1.0f + precise::exp (-sum));
      for (uint i = 0; i + 1 < past; i++)
        window[i] = window[i + 1];
      window[past - 1] = input;
    }
  for (uint i = 0; i < past; i++)
    h[i] = window[i];
}

/* The convolution of gdn_conv for a batch of at least KERNEL_SIZE - 1
   tokens, with thread (C, T) computing channel C of token T.  HISTORY
   holds the inputs before the batch, and gdn_conv_history moves it
   forward after every thread has read it.  One thread per channel ran
   the 512 tokens of a Qwen3.5-0.8B prefill one after another, which
   took 5 percent of the prefill.  */
kernel void
gdn_conv_batch (device const float *x [[buffer (0)]],
                device const float *taps [[buffer (1)]],
                device const float *history [[buffer (2)]],
                device float *y [[buffer (3)]],
                constant uint &channels [[buffer (4)]],
                constant uint &kernel_size [[buffer (5)]],
                constant uint &n_tokens [[buffer (6)]],
                uint2 position [[thread_position_in_grid]])
{
  uint c = position.x;
  uint t = position.y;
  if (c >= channels || t >= n_tokens)
    return;
  uint past = kernel_size - 1;
  device const float *w = taps + c * kernel_size;
  float sum = 0.0f;
  for (uint i = 0; i < kernel_size; i++)
    {
      /* Tap I reads the input KERNEL_SIZE - 1 - I tokens back.  */
      int source = int (t) + int (i) - int (past);
      float input = source >= 0
                        ? x[uint (source) * channels + c]
                        : history[c * past + uint (int (past) + source)];
      sum += w[i] * input;
    }
  y[t * channels + c] = sum / (1.0f + precise::exp (-sum));
}

/* Set HISTORY to the last KERNEL_SIZE - 1 inputs of the batch of
   N_TOKENS tokens at X, after gdn_conv_batch has read the old history.
   Thread C handles channel C.  */
kernel void
gdn_conv_history (device const float *x [[buffer (0)]],
                  device float *history [[buffer (1)]],
                  constant uint &channels [[buffer (2)]],
                  constant uint &kernel_size [[buffer (3)]],
                  constant uint &n_tokens [[buffer (4)]],
                  uint c [[thread_position_in_grid]])
{
  if (c >= channels)
    return;
  uint past = kernel_size - 1;
  for (uint i = 0; i < past; i++)
    history[c * past + i] = x[(n_tokens - past + i) * channels + c];
}

/* Normalize each of the N_ROWS rows of V_DIM floats at X by its root mean
   square, scale it by WEIGHT, and multiply it by SiLU of the matching
   floats at GATE: the gated output norm of a DeltaNet layer, one row per
   value head of each token.  Threadgroup R of one simdgroup handles row
   R.  */
kernel void
gdn_gated_norm (device float *x [[buffer (0)]],
                device const float *weight [[buffer (1)]],
                device const float *gate [[buffer (2)]],
                constant uint &v_dim [[buffer (3)]],
                constant float &eps [[buffer (4)]],
                uint row [[threadgroup_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]])
{
  device float *values = x + ulong (row) * v_dim;
  device const float *gates = gate + ulong (row) * v_dim;
  float sum = 0.0f;
  for (uint i = lane; i < v_dim; i += SIMD_WIDTH)
    sum += values[i] * values[i];
  float scale = precise::rsqrt (simd_sum (sum) / float (v_dim) + eps);
  for (uint i = lane; i < v_dim; i += SIMD_WIDTH)
    {
      float g = gates[i];
      values[i] *= scale * weight[i] * (g / (1.0f + precise::exp (-g)));
    }
}

/* Turn the N projections at BETA and ALPHA, one per DeltaNet value head of
   each token, into gates.  BETA receives the sigmoid of each update
   strength, and ALPHA receives each decay factor exp (A * softplus (ALPHA
   + DT_BIAS)), with the N_V_HEADS values of A and DT_BIAS per head.  */
kernel void
gdn_gates (device float *beta [[buffer (0)]],
           device float *alpha [[buffer (1)]],
           device const float *a [[buffer (2)]],
           device const float *dt_bias [[buffer (3)]],
           constant uint &n_v_heads [[buffer (4)]],
           constant uint &n [[buffer (5)]],
           uint index [[thread_position_in_grid]])
{
  if (index >= n)
    return;
  uint head = index % n_v_heads;
  beta[index] = 1.0f / (1.0f + precise::exp (-beta[index]));
  float rate = alpha[index] + dt_bias[head];
  /* transformers' softplus returns its input above 20.  */
  float softplus
      = rate > 20.0f ? rate : precise::log (1.0f + precise::exp (rate));
  alpha[index] = precise::exp (a[head] * softplus);
}

/* Set QUERY and KEY to a lane's four elements of a query head and its key
   head in the DeltaNet token at TOKEN, from element FIRST of the head
   onward, each head divided by its L2 norm and the query scaled by
   Q_SCALE.  The norm's epsilon of 1e-6 is the one transformers fixes.
   Every lane of the simdgroup calls this together.  */
static void
load_query_key (device const float *token, uint first, uint key_dim,
                bool active, float q_scale, thread float4 &query,
                thread float4 &key)
{
  query = 0.0f;
  key = 0.0f;
  if (active)
    {
      query = *(device const float4 *)(token + first);
      key = *(device const float4 *)(token + key_dim + first);
    }
  /* A zero scale marks heads that gdn_qk_norm already normalized.  */
  if (q_scale == 0.0f)
    return;
  query *= q_scale / precise::sqrt (simd_sum (dot (query, query)) + 1e-6f);
  key *= 1.0f / precise::sqrt (simd_sum (dot (key, key)) + 1e-6f);
}

/* Divide each query and key head of the N_TOKENS DeltaNet tokens at Y by
   its L2 norm, and scale the queries by Q_SCALE, for a batch, where the
   norms inside gdn_recurrence would lengthen each sequential step.
   Threadgroup (H, T, W) of one simdgroup handles head H of token T, its
   queries when W is 0 and its keys when W is 1.  */
kernel void
gdn_qk_norm (device float *y [[buffer (0)]],
             constant uint &n_k_heads [[buffer (1)]],
             constant uint &k_dim [[buffer (2)]],
             constant uint &conv_dim [[buffer (3)]],
             constant float &q_scale [[buffer (4)]],
             uint3 position [[threadgroup_position_in_grid]],
             uint lane [[thread_index_in_simdgroup]])
{
  uint head = position.x;
  uint token = position.y;
  uint which = position.z;
  device float *x
      = y + token * conv_dim + which * n_k_heads * k_dim + head * k_dim;
  float sum = 0.0f;
  for (uint i = lane; i < k_dim; i += SIMD_WIDTH)
    sum += x[i] * x[i];
  sum = simd_sum (sum);
  float scale = which == 0 ? q_scale : 1.0f;
  float factor = scale / precise::sqrt (sum + 1e-6f);
  for (uint i = lane; i < k_dim; i += SIMD_WIDTH)
    x[i] *= factor;
}

/* Run the gated delta rule of N_TOKENS tokens on the state of each
   DeltaNet value head and store each token's outputs at OUT, V_DIM floats
   per value head.  Y holds the tokens' queries and keys and their
   values, and BETA and DECAY hold each token's gates per value head.
   Each query and key head is divided by its L2 norm, and the queries are
   scaled by Q_SCALE, so no separate launch normalizes them.  STATE holds
   N_V_HEADS matrices of K_DIM rows of V_DIM floats.

   Each simdgroup handles one column of the state of one value head, as
   in MLX's gated_delta_kernel and llama.cpp's kernel_gated_delta_net.
   The lanes split the column's rows and keep them in registers for the
   whole batch, and simd_sum finishes each token's two dot products over
   the rows.  Threadgroup (B, J) of
   GDN_COLUMNS simdgroups handles columns GDN_COLUMNS B onward of value
   head J.  One column per simdgroup gives the 9B model's 32 heads of 128
   columns 4096 simdgroups, where one column per lane gave 128.  */
kernel void
gdn_recurrence (device const float *y [[buffer (0)]],
                device const float *beta [[buffer (1)]],
                device const float *decay [[buffer (2)]],
                device float *state [[buffer (3)]],
                device float *out [[buffer (4)]],
                constant uint &n_k_heads [[buffer (5)]],
                constant uint &n_v_heads [[buffer (6)]],
                constant uint &k_dim [[buffer (7)]],
                constant uint &v_dim [[buffer (8)]],
                constant uint &n_tokens [[buffer (9)]],
                constant float &q_scale [[buffer (10)]],
                uint2 position [[threadgroup_position_in_grid]],
                uint simdgroup [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]])
{
  /* Lane L holds the four rows 4 L onward in a float4, so the rows stay
     in registers and its query and key loads are single vector loads.
     Rows strided across the lanes in an array bounded by the head size
     ran slower.  */
  constexpr uint rows_per_lane = GDN_MAX_K_DIM / SIMD_WIDTH;
  uint column = position.x * GDN_COLUMNS + simdgroup;
  uint head = position.y;
  uint key_head = head % n_k_heads;
  uint key_dim = n_k_heads * k_dim;
  uint conv_dim = 2 * key_dim + n_v_heads * v_dim;
  uint first = lane * rows_per_lane;
  bool active = first < k_dim;
  device float *head_state = state + ulong (head) * k_dim * v_dim;
  float4 rows = 0.0f;
  if (active)
    for (uint i = 0; i < rows_per_lane; i++)
      rows[i] = head_state[(first + i) * v_dim + column];

  /* Each step loads and normalizes the next token's query and key, which
     do not depend on the state, so they overlap the current token's
     update instead of delaying it.  */
  float4 query;
  float4 key;
  load_query_key (y, key_head * k_dim + first, key_dim, active, q_scale,
                  query, key);
  for (uint t = 0; t < n_tokens; t++)
    {
      float4 next_query = 0.0f;
      float4 next_key = 0.0f;
      if (t + 1 < n_tokens)
        load_query_key (y + ulong (t + 1) * conv_dim,
                        key_head * k_dim + first, key_dim, active, q_scale,
                        next_query, next_key);
      device const float *token = y + ulong (t) * conv_dim;
      rows *= decay[t * n_v_heads + head];
      float remembered = simd_sum (dot (rows, key));
      float value = token[2 * key_dim + head * v_dim + column];
      float delta = (value - remembered) * beta[t * n_v_heads + head];
      rows += key * delta;
      float sum = simd_sum (dot (rows, query));
      if (lane == 0)
        out[ulong (t) * n_v_heads * v_dim + head * v_dim + column] = sum;
      query = next_query;
      key = next_key;
    }

  if (active)
    for (uint i = 0; i < rows_per_lane; i++)
      head_state[(first + i) * v_dim + column] = rows[i];
}

/* Multiply each of the N floats at X by SiLU of the matching float at
   GATE.  */
kernel void
silu_mul (device float *x [[buffer (0)]],
          device const float *gate [[buffer (1)]],
          constant uint &n [[buffer (2)]],
          uint index [[thread_position_in_grid]])
{
  if (index >= n)
    return;
  float g = gate[index];
  x[index] *= g / (1.0f + precise::exp (-g));
}

/* Multiply each attention output at OUT, N_TOKENS tokens of Q_DIM floats,
   by the sigmoid of its gate.  The gates sit in the query projection QG,
   whose token holds each head's HEAD_DIM queries followed by its HEAD_DIM
   gates.  */
kernel void
attention_gate (device float *out [[buffer (0)]],
                device const float *qg [[buffer (1)]],
                constant uint &head_dim [[buffer (2)]],
                constant uint &q_dim [[buffer (3)]],
                constant uint &n [[buffer (4)]],
                uint index [[thread_position_in_grid]])
{
  if (index >= n)
    return;
  uint token = index / q_dim;
  uint within = index % q_dim;
  uint head = within / head_dim;
  float gate = qg[ulong (token) * 2 * q_dim + head * 2 * head_dim + head_dim
                  + within % head_dim];
  out[index] *= 1.0f / (1.0f + precise::exp (-gate));
}
