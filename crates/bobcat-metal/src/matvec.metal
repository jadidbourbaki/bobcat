/* Matrix-vector kernels for F16, F32, Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K
   weights.  */

/* Multiply the Q8_0 matrix WEIGHTS, which has N_ROWS rows of N_COLS
   elements, by the N_COLS floats at X and store the N_ROWS results at Y.
   With fuse_norm, the kernel first RMS-normalizes X with epsilon EPS and
   scales it by NORM_WEIGHT.  With accumulate, the results add to Y.

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

  for (uint b
       = simdgroup_index * BLOCKS_PER_SIMDGROUP + lane / LANES_PER_BLOCK;
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
   gate result times the matching up result at Y.  With fuse_norm, the
   kernel first RMS-normalizes X with epsilon EPS and scales it by
   NORM_WEIGHT.  The
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

  for (uint b
       = simdgroup_index * BLOCKS_PER_SIMDGROUP + lane / LANES_PER_BLOCK;
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

/* Multiply the matrix WEIGHTS of format F, which has N_ROWS rows of
   N_COLS weights, by the N_COLS floats at X and store the N_ROWS
   results at Y, with the function constants and layout of
   matvec_q8_0.  Four lanes share each 32-weight group.  */
template <typename F>
kernel void
matvec_q (device const uchar *weights [[buffer (0)]],
          device const float *x [[buffer (1)]], device float *y [[buffer (2)]],
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
  uint n_groups = n_cols / QK8_0;
  ulong row_bytes = row_bytes_of<F> (n_cols);
  uint part = lane % LANES_PER_BLOCK;
  uint stride = simdgroups * BLOCKS_PER_SIMDGROUP;

  /* SUMS holds one sum per row, then the sum of squares of X.  */
  float sums[MAX_MATVEC_ROWS + 1] = { 0.0f };
  uint n_sums = rows_per_threadgroup + 1;

  for (uint g
       = simdgroup_index * BLOCKS_PER_SIMDGROUP + lane / LANES_PER_BLOCK;
       g < n_groups; g += stride)
    {
      uint e = g * QK8_0 + part * QUANTS_PER_LANE;
      float4 inputs[2];
      load_inputs (x, norm_weight, e, inputs, sums[rows_per_threadgroup]);
      for (uint r = 0; r < rows_per_threadgroup; r++)
        {
          uint row = first_row + r;
          if (row >= n_rows)
            break;
          weights8 w = F::load8 (weights + row * row_bytes, e);
          sums[r] += dot (w.low, inputs[0]) + dot (w.high, inputs[1]);
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

template [[host_name ("matvec_f16")]] kernel decltype (matvec_q<f16_format>)
    matvec_q<f16_format>;
template [[host_name ("matvec_f32")]] kernel decltype (matvec_q<f32_format>)
    matvec_q<f32_format>;

/* Multiply the matrices GATE and UP of format F, which each have N_ROWS
   rows of N_COLS weights, by the N_COLS floats at X, and store SiLU of
   each gate result times the matching up result at Y, as
   matvec_q8_0_swiglu does.  */
template <typename F>
kernel void
matvec_q_swiglu (device const uchar *gate [[buffer (0)]],
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
  uint n_groups = n_cols / QK8_0;
  ulong row_bytes = row_bytes_of<F> (n_cols);
  uint part = lane % LANES_PER_BLOCK;
  uint stride = simdgroups * BLOCKS_PER_SIMDGROUP;

  /* SUMS holds the gate sums, then the up sums, then the sum of squares
     of X.  */
  float sums[2 * MAX_MATVEC_ROWS + 1] = { 0.0f };
  uint n_sums = 2 * rows + 1;

  for (uint g
       = simdgroup_index * BLOCKS_PER_SIMDGROUP + lane / LANES_PER_BLOCK;
       g < n_groups; g += stride)
    {
      uint e = g * QK8_0 + part * QUANTS_PER_LANE;
      float4 inputs[2];
      load_inputs (x, norm_weight, e, inputs, sums[2 * rows]);
      for (uint r = 0; r < rows; r++)
        {
          uint row = first_row + r;
          if (row >= n_rows)
            break;
          ulong offset = row * row_bytes;
          weights8 g_w = F::load8 (gate + offset, e);
          weights8 u_w = F::load8 (up + offset, e);
          sums[r] += dot (g_w.low, inputs[0]) + dot (g_w.high, inputs[1]);
          sums[rows + r]
              += dot (u_w.low, inputs[0]) + dot (u_w.high, inputs[1]);
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

template [[host_name (
    "matvec_f16_swiglu")]] kernel decltype (matvec_q_swiglu<f16_format>)
    matvec_q_swiglu<f16_format>;
template [[host_name (
    "matvec_f32_swiglu")]] kernel decltype (matvec_q_swiglu<f32_format>)
    matvec_q_swiglu<f32_format>;

/* Multiply the matrix WEIGHTS of K-quant format K, which has N_ROWS rows
   of N_COLS weights, by the N_COLS floats at X and store the N_ROWS
   results at Y, with the function constants of matvec_q8_0.  */
template <typename K>
kernel void
matvec_k (device const uchar *weights [[buffer (0)]],
          device const float *x [[buffer (1)]], device float *y [[buffer (2)]],
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
  /* The threadgroup's simdgroups share its K_ROWS_PER_SIMDGROUP rows and
     split the columns, taking the super-block passes in turn.  Short
     matrices of long rows then launch enough threadgroups to fill the
     GPU.  */
  threadgroup float partials[(K_ROWS_PER_SIMDGROUP + 1) * MAX_SIMDGROUPS];
  uint first_row = threadgroup_index * K_ROWS_PER_SIMDGROUP;
  uint n_blocks = n_cols / K::block_weights;
  ulong row_bytes = ulong (n_blocks) * K::block_bytes;
  float sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  float sum_squares = 0.0f;

  for (uint block
       = K::first_block (lane) + simdgroup_index * K::blocks_per_pass;
       block < n_blocks; block += K::blocks_per_pass * simdgroups)
    {
      typename K::inputs in;
      K::load (x, norm_weight, block, lane, in, sum_squares);
      for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
        if (first_row + r < n_rows)
          sums[r] += K::dot_part (weights + (first_row + r) * row_bytes, block,
                                  lane, in);
    }

  for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
    {
      float part = simd_sum (sums[r]);
      if (lane == 0)
        partials[r * MAX_SIMDGROUPS + simdgroup_index] = part;
    }
  float squares = simd_sum (sum_squares);
  if (lane == 0)
    partials[K_ROWS_PER_SIMDGROUP * MAX_SIMDGROUPS + simdgroup_index]
        = squares;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  if (simdgroup_index != 0 || lane >= K_ROWS_PER_SIMDGROUP)
    return;

  float scale = 1.0f;
  if (fuse_norm)
    {
      float total_squares = 0.0f;
      for (uint s = 0; s < simdgroups; s++)
        total_squares += partials[K_ROWS_PER_SIMDGROUP * MAX_SIMDGROUPS + s];
      scale = precise::rsqrt (total_squares / float (n_cols) + eps);
    }
  uint row = first_row + lane;
  if (row >= n_rows)
    return;
  float total = 0.0f;
  for (uint s = 0; s < simdgroups; s++)
    total += partials[lane * MAX_SIMDGROUPS + s];
  total *= scale;
  y[row] = accumulate ? y[row] + total : total;
}

template [[host_name (
    "matvec_q4k")]] kernel decltype (matvec_k<q4k_lanes>) matvec_k<q4k_lanes>;
template [[host_name (
    "matvec_q5k")]] kernel decltype (matvec_k<q5k_lanes>) matvec_k<q5k_lanes>;
template [[host_name (
    "matvec_q6k")]] kernel decltype (matvec_k<q6k_lanes>) matvec_k<q6k_lanes>;
template [[host_name ("matvec_q4_0")]] kernel decltype (matvec_k<q4_0_lanes>)
    matvec_k<q4_0_lanes>;

/* Multiply the input projection WEIGHTS of a gated short convolution,
   which has 3 N_EMBD rows of N_COLS weights of format K, by the N_COLS
   floats at X, and run the convolution on the results, as short_conv
   does for one token.  Rows CH, N_EMBD + CH, and 2 N_EMBD + CH hold the
   gates B and C and the input X of channel CH.

   Threadgroup CH computes channel CH, with its simdgroups splitting the
   columns as in matvec_k.  Its first lane then convolves B times X with
   the channel's TAPS and HISTORY, updates HISTORY, and stores C times
   the result at OUT, so the convolution needs no launch of its own.  */
template <typename K>
kernel void
matvec_conv (device const uchar *weights [[buffer (0)]],
             device const float *x [[buffer (1)]],
             device const float *taps [[buffer (2)]],
             device float *history [[buffer (3)]],
             device float *out [[buffer (4)]],
             constant uint &n_embd [[buffer (5)]],
             constant uint &n_cols [[buffer (6)]],
             constant uint &kernel_size [[buffer (7)]],
             device const float *norm_weight
             [[buffer (8), function_constant (fuse_norm)]],
             constant float &eps [[buffer (9), function_constant (fuse_norm)]],
             uint ch [[threadgroup_position_in_grid]],
             uint simdgroup_index [[simdgroup_index_in_threadgroup]],
             uint simdgroups [[simdgroups_per_threadgroup]],
             uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[4 * MAX_SIMDGROUPS];
  uint n_blocks = n_cols / K::block_weights;
  ulong row_bytes = ulong (n_blocks) * K::block_bytes;
  float sums[3] = { 0.0f };
  float sum_squares = 0.0f;

  for (uint block
       = K::first_block (lane) + simdgroup_index * K::blocks_per_pass;
       block < n_blocks; block += K::blocks_per_pass * simdgroups)
    {
      typename K::inputs in;
      K::load (x, norm_weight, block, lane, in, sum_squares);
      for (uint r = 0; r < 3; r++)
        sums[r] += K::dot_part (weights + ulong (r * n_embd + ch) * row_bytes,
                                block, lane, in);
    }

  for (uint r = 0; r < 3; r++)
    {
      float part = simd_sum (sums[r]);
      if (lane == 0)
        partials[r * MAX_SIMDGROUPS + simdgroup_index] = part;
    }
  float squares = simd_sum (sum_squares);
  if (lane == 0)
    partials[3 * MAX_SIMDGROUPS + simdgroup_index] = squares;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  if (simdgroup_index != 0 || lane != 0 || ch >= n_embd)
    return;

  float bcx[4] = { 0.0f };
  for (uint r = 0; r < 4; r++)
    for (uint s = 0; s < simdgroups; s++)
      bcx[r] += partials[r * MAX_SIMDGROUPS + s];
  float scale = 1.0f;
  if (fuse_norm)
    scale = precise::rsqrt (bcx[3] / float (n_cols) + eps);
  float b = bcx[0] * scale;
  float c = bcx[1] * scale;
  float input = bcx[2] * scale;

  device const float *channel_taps = taps + ch * kernel_size;
  float bx = b * input;
  float sum = channel_taps[kernel_size - 1] * bx;
  for (uint k = 0; k + 1 < kernel_size; k++)
    sum += channel_taps[k] * history[k * n_embd + ch];
  for (uint k = 0; k + 2 < kernel_size; k++)
    history[k * n_embd + ch] = history[(k + 1) * n_embd + ch];
  history[(kernel_size - 2) * n_embd + ch] = bx;
  out[ch] = c * sum;
}

template [[host_name (
    "matvec_conv_q4_0")]] kernel decltype (matvec_conv<q4_0_lanes>)
    matvec_conv<q4_0_lanes>;
template
    [[host_name ("matvec_conv_q4k")]] kernel decltype (matvec_conv<q4k_lanes>)
        matvec_conv<q4k_lanes>;
template
    [[host_name ("matvec_conv_q5k")]] kernel decltype (matvec_conv<q5k_lanes>)
        matvec_conv<q5k_lanes>;
template
    [[host_name ("matvec_conv_q6k")]] kernel decltype (matvec_conv<q6k_lanes>)
        matvec_conv<q6k_lanes>;

/* Store at Y SiLU of each gate result times the matching up result for
   the threadgroup's rows of GATE and UP, which have N_ROWS rows of
   N_COLS weights of K-quant format K, times the N_COLS floats at X.  The
   simdgroups share the rows and split the columns, as in matvec_k.
   PARTIALS holds each simdgroup's gate sums, then its up sums, then its
   sum of squares, in (2 K_ROWS_PER_SIMDGROUP + 1) MAX_SIMDGROUPS floats.
   NORM_WEIGHT and EPS normalize the input when fuse_norm is set.  */
template <typename K>
static void
swiglu_rows (device const uchar *gate, device const uchar *up,
             device const float *x, device float *y, uint n_rows, uint n_cols,
             device const float *norm_weight, float eps,
             threadgroup float *partials, uint threadgroup_index,
             uint simdgroup_index, uint simdgroups, uint lane)
{
  uint first_row = threadgroup_index * K_ROWS_PER_SIMDGROUP;
  uint n_blocks = n_cols / K::block_weights;
  ulong row_bytes = ulong (n_blocks) * K::block_bytes;
  float gate_sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  float up_sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  float sum_squares = 0.0f;

  for (uint block
       = K::first_block (lane) + simdgroup_index * K::blocks_per_pass;
       block < n_blocks; block += K::blocks_per_pass * simdgroups)
    {
      typename K::inputs in;
      K::load (x, norm_weight, block, lane, in, sum_squares);
      for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
        if (first_row + r < n_rows)
          {
            ulong offset = (first_row + r) * row_bytes;
            gate_sums[r] += K::dot_part (gate + offset, block, lane, in);
            up_sums[r] += K::dot_part (up + offset, block, lane, in);
          }
    }

  for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
    {
      float g = simd_sum (gate_sums[r]);
      float u = simd_sum (up_sums[r]);
      if (lane == 0)
        {
          partials[r * MAX_SIMDGROUPS + simdgroup_index] = g;
          partials[(K_ROWS_PER_SIMDGROUP + r) * MAX_SIMDGROUPS
                   + simdgroup_index] = u;
        }
    }
  float squares = simd_sum (sum_squares);
  if (lane == 0)
    partials[2 * K_ROWS_PER_SIMDGROUP * MAX_SIMDGROUPS + simdgroup_index]
        = squares;
  threadgroup_barrier (mem_flags::mem_threadgroup);
  if (simdgroup_index != 0 || lane >= K_ROWS_PER_SIMDGROUP)
    return;

  float scale = 1.0f;
  if (fuse_norm)
    {
      float total_squares = 0.0f;
      for (uint s = 0; s < simdgroups; s++)
        total_squares
            += partials[2 * K_ROWS_PER_SIMDGROUP * MAX_SIMDGROUPS + s];
      scale = precise::rsqrt (total_squares / float (n_cols) + eps);
    }
  uint row = first_row + lane;
  if (row >= n_rows)
    return;
  float g = 0.0f;
  float u = 0.0f;
  for (uint s = 0; s < simdgroups; s++)
    {
      g += partials[lane * MAX_SIMDGROUPS + s];
      u += partials[(K_ROWS_PER_SIMDGROUP + lane) * MAX_SIMDGROUPS + s];
    }
  g *= scale;
  u *= scale;
  y[row] = g / (1.0f + precise::exp (-g)) * u;
}

/* The SwiGLU pair of matvec_k, as matvec_q8_0_swiglu is of
   matvec_q8_0.  */
template <typename K>
kernel void
matvec_k_swiglu (device const uchar *gate [[buffer (0)]],
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
  threadgroup float partials[(2 * K_ROWS_PER_SIMDGROUP + 1) * MAX_SIMDGROUPS];
  swiglu_rows<K> (gate, up, x, y, n_rows, n_cols, fuse_norm ? norm_weight : x,
                  fuse_norm ? eps : 0.0f, partials, threadgroup_index,
                  simdgroup_index, simdgroups, lane);
}

template [[host_name (
    "matvec_q4k_swiglu")]] kernel decltype (matvec_k_swiglu<q4k_lanes>)
    matvec_k_swiglu<q4k_lanes>;
template [[host_name (
    "matvec_q5k_swiglu")]] kernel decltype (matvec_k_swiglu<q5k_lanes>)
    matvec_k_swiglu<q5k_lanes>;
template [[host_name (
    "matvec_q6k_swiglu")]] kernel decltype (matvec_k_swiglu<q6k_lanes>)
    matvec_k_swiglu<q6k_lanes>;
template [[host_name (
    "matvec_q4_0_swiglu")]] kernel decltype (matvec_k_swiglu<q4_0_lanes>)
    matvec_k_swiglu<q4_0_lanes>;
