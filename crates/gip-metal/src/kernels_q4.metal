/* kernels_q4.metal holds gip's kernels for the Q4_0, Q4_K, and Q6_K
   formats.  The backend compiles it after kernels.metal, whose
   constants and helpers it uses.

   Each format supplies a load8 function that dequantizes 8 consecutive
   weights of a row, following ggml's reference dequantization in
   ggml/src/ggml-quants.c.  The kernels are templates over the format,
   with the loops of the Q8_0 kernels: the matrix-vector kernels give
   each lane 8 weights of a 32-weight group, and the matrix-matrix
   kernel dequantizes 32-weight groups into its weight tile.  Q4_K and
   Q6_K super-blocks hold eight 32-weight groups each.  */

/* Eight dequantized weights.  */
struct weights8
{
  float4 low;
  float4 high;
};

/* The Q4_0 format: blocks of 32 weights, each an fp16 scale and 16
   bytes whose low nibbles hold weights 0 to 15 and whose high nibbles
   hold weights 16 to 31, each offset by 8.  */
struct q4_0_format
{
  static constant constexpr uint block_weights = 32;
  static constant constexpr uint block_bytes = 18;

  /* Return weights E through E + 7 of ROW, where 8 divides E.  */
  static weights8
  load8 (device const uchar *row, uint e)
  {
    device const uchar *block = row + (e / 32) * block_bytes;
    float d = float (*(device const half *)block);
    uint o = e % 32;
    device const uchar *quants = block + 2 + o % 16;
    uint shift = o < 16 ? 0 : 4;
    float w[8];
    for (uint i = 0; i < 8; i++)
      w[i] = float (int ((quants[i] >> shift) & 0xf) - 8) * d;
    return { float4 (w[0], w[1], w[2], w[3]),
             float4 (w[4], w[5], w[6], w[7]) };
  }
};

/* The Q4_K format: super-blocks of 256 weights.  Each holds fp16 scales
   D and DMIN, 12 bytes of packed 6-bit scales and minimums for its
   eight 32-weight groups, and 128 bytes of 4-bit quants.  Each 32 bytes
   of quants hold two groups, the first in the low nibbles.  */
struct q4k_format
{
  static constant constexpr uint block_weights = 256;
  static constant constexpr uint block_bytes = 144;

  static weights8
  load8 (device const uchar *row, uint e)
  {
    device const uchar *block = row + (e / 256) * block_bytes;
    float d = float (*(device const half *)block);
    float dmin = float (*(device const half *)(block + 2));
    device const uchar *scales = block + 4;
    uint o = e % 256;
    uint j = o / 32;
    uint scale;
    uint min;
    if (j < 4)
      {
        scale = scales[j] & 63;
        min = scales[j + 4] & 63;
      }
    else
      {
        scale = (scales[j + 4] & 0xf) | ((scales[j - 4] >> 6) << 4);
        min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
      }
    float group_scale = d * float (scale);
    float group_min = dmin * float (min);
    device const uchar *quants = block + 16 + (j / 2) * 32 + o % 32;
    uint shift = j % 2 == 0 ? 0 : 4;
    float w[8];
    for (uint i = 0; i < 8; i++)
      w[i] = group_scale * float ((quants[i] >> shift) & 0xf) - group_min;
    return { float4 (w[0], w[1], w[2], w[3]),
             float4 (w[4], w[5], w[6], w[7]) };
  }
};

/* The Q6_K format: super-blocks of 256 weights.  Each holds 128 bytes
   of low 4-bit halves, 64 bytes of high 2-bit pairs, 16 int8 scales of
   16-weight groups, and an fp16 scale D.  Each 128-weight half of the
   super-block spreads its weights over 64 low bytes and 32 high
   bytes.  */
struct q6k_format
{
  static constant constexpr uint block_weights = 256;
  static constant constexpr uint block_bytes = 210;

  static weights8
  load8 (device const uchar *row, uint e)
  {
    device const uchar *block = row + (e / 256) * block_bytes;
    float d = float (*(device const half *)(block + 208));
    uint o = e % 256;
    uint half_index = o / 128;
    uint quarter = (o % 128) / 32;
    uint l0 = o % 32;
    device const uchar *low = block + half_index * 64 + (quarter % 2) * 32;
    device const uchar *high = block + 128 + half_index * 32;
    device const char *scales
        = (device const char *)(block + 192) + half_index * 8;
    float scale = d * float (scales[l0 / 16 + 2 * quarter]);
    uint low_shift = quarter < 2 ? 0 : 4;
    uint high_shift = 2 * quarter;
    float w[8];
    for (uint i = 0; i < 8; i++)
      {
        uint l = l0 + i;
        uint q = ((low[l] >> low_shift) & 0xf)
                 | (((high[l] >> high_shift) & 3) << 4);
        w[i] = scale * float (int (q) - 32);
      }
    return { float4 (w[0], w[1], w[2], w[3]),
             float4 (w[4], w[5], w[6], w[7]) };
  }
};

/* Return the bytes of one row of N_COLS weights of format F.  */
template <typename F>
static ulong
row_bytes_of (uint n_cols)
{
  return ulong (n_cols / F::block_weights) * F::block_bytes;
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

template [[host_name ("matvec_q4_0")]] kernel decltype (matvec_q<q4_0_format>)
    matvec_q<q4_0_format>;
template [[host_name ("matvec_q4k")]] kernel decltype (matvec_q<q4k_format>)
    matvec_q<q4k_format>;
template [[host_name ("matvec_q6k")]] kernel decltype (matvec_q<q6k_format>)
    matvec_q<q6k_format>;

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
    "matvec_q4_0_swiglu")]] kernel decltype (matvec_q_swiglu<q4_0_format>)
    matvec_q_swiglu<q4_0_format>;
template [[host_name (
    "matvec_q4k_swiglu")]] kernel decltype (matvec_q_swiglu<q4k_format>)
    matvec_q_swiglu<q4k_format>;
template [[host_name (
    "matvec_q6k_swiglu")]] kernel decltype (matvec_q_swiglu<q6k_format>)
    matvec_q_swiglu<q6k_format>;

/* Multiply the matrix WEIGHTS of format F, which has N_ROWS rows of
   N_COLS weights, by each of the N_TOKENS rows of N_COLS floats at X,
   with the tiles, function constants, and stores of matmul_q8_0.  Each
   pass of the main loop dequantizes one 32-weight group of every row of
   the tile.  */
template <typename F>
kernel void
matmul_q (device const uchar *weights [[buffer (0)]],
          device const float *x [[buffer (1)]], device float *y [[buffer (2)]],
          constant uint &n_rows [[buffer (3)]],
          constant uint &n_cols [[buffer (4)]],
          constant uint &n_tokens [[buffer (5)]],
          uint2 position [[threadgroup_position_in_grid]],
          uint2 thread_position [[thread_position_in_threadgroup]],
          uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
  /* The tiles are laid out as in matmul_q8_0.  The output tile reuses
     the input and weight scratch after the final multiply.  */
  threadgroup uchar scratch[MATMUL_ROWS * QK8_0 * sizeof (half)
                            + MATMUL_TOKENS * QK8_0 * sizeof (float)];
  threadgroup half *weight_tile = (threadgroup half *)scratch;
  threadgroup float *input_tile
      = (threadgroup float *)(scratch + MATMUL_ROWS * QK8_0 * sizeof (half));
  threadgroup float *out_tile = (threadgroup float *)scratch;

  uint tid = thread_position.x;
  uint first_token = position.x * MATMUL_TOKENS;
  uint first_row = position.y * MATMUL_ROWS;
  uint n_groups = n_cols / QK8_0;
  ulong row_bytes = row_bytes_of<F> (n_cols);

  /* Two threads dequantize the 32 weights of each row's group, 16 each.
     Four threads convert the 32 inputs of each token's group, 8 each.  */
  ushort weight_row = tid / 2;
  ushort weight_half = tid % 2;
  ushort input_token = tid / LANES_PER_BLOCK;
  ushort input_part = tid % LANES_PER_BLOCK;

  uint row = min (first_row + weight_row, n_rows - 1);
  uint token = min (first_token + input_token, n_tokens - 1);

  ushort weight_base
      = 64 * (weight_row / 8) + weight_row % 8 + 64 * 8 * (2 * weight_half);
  threadgroup float4 *input_slot
      = (threadgroup float4 *)(input_tile
                               + 64 * (4 * input_part + input_token / 8)
                               + 8 * (input_token % 8));
  device const uchar *row_weights = weights + row * row_bytes;
  device const float4 *token_inputs
      = (device const float4 *)(x + ulong (token) * n_cols
                                + input_part * QUANTS_PER_LANE);

  uint row_block_base = 4 * (simdgroup_index % 2);
  uint token_block_base = 2 * (simdgroup_index / 2);
  simdgroup_float8x8 acc[8];
  for (uint i = 0; i < 8; i++)
    acc[i] = make_filled_simdgroup_matrix<float, 8, 8> (0.0f);

  for (uint g = 0; g < n_groups; g++)
    {
      /* Load this group from device memory into registers first, then
         wait for every simdgroup to finish multiplying the previous
         group.  The loads overlap that work.  */
      uint e = g * QK8_0 + 16 * weight_half;
      weights8 first = F::load8 (row_weights, e);
      weights8 second = F::load8 (row_weights, e + 8);
      half4 w[4] = { half4 (first.low), half4 (first.high), half4 (second.low),
                     half4 (second.high) };
      device const float4 *xs = token_inputs + g * (QK8_0 / 4);
      float4 in0 = xs[0];
      float4 in1 = xs[1];
      threadgroup_barrier (mem_flags::mem_threadgroup);

      for (ushort v = 0; v < 4; v++)
        for (ushort c = 0; c < 4; c++)
          {
            ushort kk = 4 * v + c;
            weight_tile[weight_base + 64 * 8 * (kk / 8) + 8 * (kk % 8)]
                = w[v][c];
          }
      input_slot[0] = in0;
      input_slot[1] = in1;
      threadgroup_barrier (mem_flags::mem_threadgroup);

#pragma unroll
      for (ushort k = 0; k < QK8_0 / 8; k++)
        {
          simdgroup_half8x8 a[4];
          simdgroup_float8x8 bm[2];
#pragma unroll
          for (ushort i = 0; i < 4; i++)
            simdgroup_load (
                a[i], weight_tile + 64 * (8 * k + row_block_base + i), 8);
#pragma unroll
          for (ushort j = 0; j < 2; j++)
            simdgroup_load (
                bm[j], input_tile + 64 * (4 * k + token_block_base + j), 8);
#pragma unroll
          for (ushort j = 0; j < 2; j++)
#pragma unroll
            for (ushort i = 0; i < 4; i++)
              simdgroup_multiply_accumulate (acc[4 * j + i], bm[j], a[i],
                                             acc[4 * j + i]);
        }
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);

  uint row_offset = 8 * row_block_base;
  uint token_offset = 8 * token_block_base;
  for (uint j = 0; j < 2; j++)
    for (uint i = 0; i < 4; i++)
      simdgroup_store (acc[4 * j + i],
                       out_tile + (token_offset + 8 * j) * MATMUL_ROWS
                           + row_offset + 8 * i,
                       MATMUL_ROWS);
  threadgroup_barrier (mem_flags::mem_threadgroup);

  for (uint idx = tid; idx < MATMUL_ROWS * MATMUL_TOKENS;
       idx += MATMUL_SIMDGROUPS * SIMD_WIDTH)
    {
      uint r = idx % MATMUL_ROWS;
      uint t = idx / MATMUL_ROWS;
      uint out_row = first_row + r;
      uint out_token = first_token + t;
      if (out_row < n_rows && out_token < n_tokens)
        {
          device float *dst = y + ulong (out_token) * n_rows + out_row;
          float value = out_tile[t * MATMUL_ROWS + r];
          if (swiglu_store)
            {
              float gate_value = *dst;
              *dst = gate_value / (1.0f + precise::exp (-gate_value)) * value;
            }
          else
            *dst = accumulate ? *dst + value : value;
        }
    }
}

template [[host_name ("matmul_q4_0")]] kernel decltype (matmul_q<q4_0_format>)
    matmul_q<q4_0_format>;
template [[host_name ("matmul_q4k")]] kernel decltype (matmul_q<q4k_format>)
    matmul_q<q4k_format>;
template [[host_name ("matmul_q6k")]] kernel decltype (matmul_q<q6k_format>)
    matmul_q<q6k_format>;

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
    "embed_q4k")]] kernel decltype (embed_q<q4k_format>) embed_q<q4k_format>;
template [[host_name (
    "embed_q6k")]] kernel decltype (embed_q<q6k_format>) embed_q<q6k_format>;
