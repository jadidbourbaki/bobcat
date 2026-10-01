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

/* Divide each query and key head of the N_TOKENS DeltaNet tokens at Y by
   its L2 norm, and scale the queries by Q_SCALE.  Threadgroup (H, T, W)
   of one simdgroup handles head H of token T, its queries when W is 0
   and its keys when W is 1.  */
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
  device float *x = y + token * conv_dim + which * n_k_heads * k_dim + head * k_dim;
  float sum = 0.0f;
  for (uint i = lane; i < k_dim; i += SIMD_WIDTH)
    sum += x[i] * x[i];
  sum = simd_sum (sum);
  float scale = which == 0 ? q_scale : 1.0f;
  /* transformers fixes the epsilon of this norm at 1e-6.  */
  float factor = scale / precise::sqrt (sum + 1e-6f);
  for (uint i = lane; i < k_dim; i += SIMD_WIDTH)
    x[i] *= factor;
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

/* Run the gated delta rule of N_TOKENS tokens on the state of each
   DeltaNet value head and store each token's outputs at OUT, V_DIM floats
   per value head.  Y holds the tokens' normalized queries and keys and
   their values, and BETA and DECAY hold each token's gates per value
   head.  STATE holds N_V_HEADS matrices of K_DIM rows of V_DIM floats.

   Each simdgroup handles one column of the state of one value head, as
   in MLX's gated_delta_kernel.  Lane L keeps rows L, L + 32, and so on
   of the column in registers for the whole batch, and simd_sum finishes
   each token's two dot products over the rows.  Threadgroup (B, J) of
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
                uint2 position [[threadgroup_position_in_grid]],
                uint simdgroup [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]])
{
  constexpr uint max_rows = GDN_MAX_K_DIM / SIMD_WIDTH;
  uint column = position.x * GDN_COLUMNS + simdgroup;
  uint head = position.y;
  uint key_head = head % n_k_heads;
  uint key_dim = n_k_heads * k_dim;
  uint conv_dim = 2 * key_dim + n_v_heads * v_dim;
  uint n_rows = k_dim / SIMD_WIDTH;
  device float *head_state = state + ulong (head) * k_dim * v_dim;
  float rows[max_rows];
  for (uint i = 0; i < max_rows; i++)
    rows[i] = i < n_rows ? head_state[(lane + i * SIMD_WIDTH) * v_dim + column]
                         : 0.0f;

  for (uint t = 0; t < n_tokens; t++)
    {
      device const float *token = y + ulong (t) * conv_dim;
      device const float *query = token + key_head * k_dim + lane;
      device const float *key = token + key_dim + key_head * k_dim + lane;
      float token_decay = decay[t * n_v_heads + head];
      float remembered = 0.0f;
      for (uint i = 0; i < n_rows; i++)
        {
          rows[i] *= token_decay;
          remembered += rows[i] * key[i * SIMD_WIDTH];
        }
      remembered = simd_sum (remembered);
      float value = token[2 * key_dim + head * v_dim + column];
      float delta = (value - remembered) * beta[t * n_v_heads + head];
      float sum = 0.0f;
      for (uint i = 0; i < n_rows; i++)
        {
          rows[i] += key[i * SIMD_WIDTH] * delta;
          sum += rows[i] * query[i * SIMD_WIDTH];
        }
      sum = simd_sum (sum);
      if (lane == 0)
        out[ulong (t) * n_v_heads * v_dim + head * v_dim + column] = sum;
    }

  for (uint i = 0; i < n_rows; i++)
    head_state[(lane + i * SIMD_WIDTH) * v_dim + column] = rows[i];
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
