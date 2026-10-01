/* Matrix-matrix kernels for Q8_0, Q4_0, Q4_K, and Q6_K weights.  */

/* Multiply the Q8_0 matrix WEIGHTS, which has N_ROWS rows of N_COLS
   elements, by each of the N_TOKENS rows of N_COLS floats at X.  Token
   T's N_ROWS results go to row T of Y.  With accumulate, the results add
   to Y.

   The layout follows llama.cpp's matrix-matrix kernel.  One threadgroup
   of MATMUL_SIMDGROUPS simdgroups computes a tile of MATMUL_ROWS rows by
   MATMUL_TOKENS tokens.  For each Q8_0 block along the columns, the
   threadgroup dequantizes the tile's weights to half precision and
   copies its inputs as floats into threadgroup memory, stored as 8 by 8
   blocks so each matrix load reads 64 neighboring values.  Each
   simdgroup then multiplies 32 rows by 16 tokens of the tile,
   accumulating in float, and holds its result as eight 8 by 8 matrices
   of tokens by rows.  Half-precision inputs ran no faster and raised
   the error of LFM2.5-350M's last layer from 1.5e-3 to 6.6e-3.  */
kernel void
matmul_q8_0 (device const uchar *weights [[buffer (0)]],
             device const float *x [[buffer (1)]],
             device float *y [[buffer (2)]],
             constant uint &n_rows [[buffer (3)]],
             constant uint &n_cols [[buffer (4)]],
             constant uint &n_tokens [[buffer (5)]],
             uint2 position [[threadgroup_position_in_grid]],
             uint2 thread_position [[thread_position_in_threadgroup]],
             uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
  /* WEIGHT_TILE holds 8 by 8 blocks of the tile's weights, each with
     rows of 8 columns and 8 matrix rows across.  Block K * 8 + R covers
     columns 8K through 8K + 7 of rows 8R through 8R + 7.  INPUT_TILE
     holds 8 by 8 blocks of tokens by columns, with block K * 4 + T
     covering columns 8K through 8K + 7 of tokens 8T through 8T + 7.  */
  /* The output tile reuses the input and weight scratch after the final
     multiply.  The barrier before the stores ends all earlier reads.  */
  threadgroup uchar scratch[MATMUL_ROWS * QK8_0 * sizeof (half)
                            + MATMUL_TOKENS * QK8_0 * sizeof (float)];
  threadgroup half *weight_tile = (threadgroup half *)scratch;
  threadgroup float *input_tile
      = (threadgroup float *)(scratch + MATMUL_ROWS * QK8_0 * sizeof (half));
  threadgroup float *out_tile = (threadgroup float *)scratch;

  uint tid = thread_position.x;
  /* Token tiles vary fastest across the grid, so neighboring threadgroups
     read the same weights while they are still in cache.  */
  uint first_token = position.x * MATMUL_TOKENS;
  uint first_row = position.y * MATMUL_ROWS;
  uint n_blocks = n_cols / QK8_0;
  ulong row_bytes = ulong (n_blocks) * Q8_0_BLOCK_BYTES;

  /* Two threads dequantize the 32 weights of each row's block, 16 each.
     Four threads convert the 32 inputs of each token's block, 8 each.  */
  ushort weight_row = tid / 2;
  ushort weight_half = tid % 2;
  ushort input_token = tid / LANES_PER_BLOCK;
  ushort input_part = tid % LANES_PER_BLOCK;

  /* Threads past the matrix edge load the last valid row or token
     instead of branching.  Their results fall outside the tile's valid
     part, and the stores skip them.  */
  uint row = min (first_row + weight_row, n_rows - 1);
  uint token = min (first_token + input_token, n_tokens - 1);

  /* Where this thread's weights and inputs land in the blocked tiles.
     Weight K of a block goes to block K / 8, row K % 8.  */
  ushort weight_base
      = 64 * (weight_row / 8) + weight_row % 8 + 64 * 8 * (2 * weight_half);
  threadgroup float4 *input_slot
      = (threadgroup float4 *)(input_tile
                               + 64 * (4 * input_part + input_token / 8)
                               + 8 * (input_token % 8));
  device const q8_0_block *row_blocks
      = (device const q8_0_block *)(weights + row * row_bytes);
  device const float4 *token_inputs
      = (device const float4 *)(x + ulong (token) * n_cols
                                + input_part * QUANTS_PER_LANE);

  /* Simdgroup S owns rows 32 * (S % 2) onward and tokens 16 * (S / 2)
     onward: four blocks of rows and two of tokens.  */
  uint row_block_base = 4 * (simdgroup_index % 2);
  uint token_block_base = 2 * (simdgroup_index / 2);
  simdgroup_float8x8 acc[8];
  for (uint i = 0; i < 8; i++)
    acc[i] = make_filled_simdgroup_matrix<float, 8, 8> (0.0f);

  for (uint b = 0; b < n_blocks; b++)
    {
      /* Load this block from device memory into registers first, then
         wait for every simdgroup to finish multiplying the previous
         block.  The loads overlap that work.  */
      device const q8_0_block *block = row_blocks + b;
      device const packed_char4 *quants
          = (device const packed_char4 *)(block->quants + 16 * weight_half);
      half block_scale = block->scale;
      half4 w[4];
      for (ushort v = 0; v < 4; v++)
        w[v] = block_scale * half4 (char4 (quants[v]));
      device const float4 *xs = token_inputs + b * (QK8_0 / 4);
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

      /* The simdgroup barriers order nothing.  They split the loads from
         the multiplies, which lets the compiler schedule each group
         together, as in llama.cpp.  */
#pragma unroll
      for (ushort k = 0; k < QK8_0 / 8; k++)
        {
          simdgroup_half8x8 a[4];
          simdgroup_float8x8 bm[2];
          simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
          for (ushort i = 0; i < 4; i++)
            simdgroup_load (
                a[i], weight_tile + 64 * (8 * k + row_block_base + i), 8);
          simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
          for (ushort j = 0; j < 2; j++)
            simdgroup_load (
                bm[j], input_tile + 64 * (4 * k + token_block_base + j), 8);
          simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
          for (ushort j = 0; j < 2; j++)
#pragma unroll
            for (ushort i = 0; i < 4; i++)
              simdgroup_multiply_accumulate (acc[4 * j + i], bm[j], a[i],
                                             acc[4 * j + i]);
        }
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);

  /* A full tile that overwrites Y stores straight to device memory.  A
     tile that combines with Y, or a partial tile, goes through
     threadgroup memory, so the threads can read Y and skip rows and
     tokens past the edges.  */
  uint row_offset = 8 * row_block_base;
  uint token_offset = 8 * token_block_base;
  bool full = first_row + MATMUL_ROWS <= n_rows
              && first_token + MATMUL_TOKENS <= n_tokens;
  if (full && !accumulate && !swiglu_store)
    {
      device float *out = y + ulong (first_token + token_offset) * n_rows
                          + first_row + row_offset;
      for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 4; i++)
          simdgroup_store (acc[4 * j + i],
                           out + ulong (8 * j) * n_rows + 8 * i, n_rows);
      return;
    }

  for (uint j = 0; j < 2; j++)
    for (uint i = 0; i < 4; i++)
      simdgroup_store (acc[4 * j + i],
                       out_tile + (token_offset + 8 * j) * MATMUL_ROWS
                           + row_offset + 8 * i,
                       MATMUL_ROWS);
  threadgroup_barrier (mem_flags::mem_threadgroup);

  /* Neighboring threads write neighboring rows of one token, so the
     stores coalesce.  */
  for (uint e = tid; e < MATMUL_ROWS * MATMUL_TOKENS;
       e += MATMUL_SIMDGROUPS * SIMD_WIDTH)
    {
      uint r = e % MATMUL_ROWS;
      uint t = e / MATMUL_ROWS;
      uint out_row = first_row + r;
      uint out_token = first_token + t;
      if (out_row < n_rows && out_token < n_tokens)
        {
          device float *dst = y + ulong (out_token) * n_rows + out_row;
          float value = out_tile[t * MATMUL_ROWS + r];
          if (swiglu_store)
            {
              float g = *dst;
              *dst = g / (1.0f + precise::exp (-g)) * value;
            }
          else
            *dst = accumulate ? *dst + value : value;
        }
    }
}

/* Token I of a dense product reads row I of X and writes row I of Y.  */
struct dense_rows
{
  static constant constexpr bool contiguous = true;

  uint
  input (uint token) const
  {
    return token;
  }

  uint
  output (uint token) const
  {
    return token;
  }
};

/* Token I of an expert's product is entry I of the expert's ENTRIES,
   which names row ENTRIES[I] of Y and row ENTRIES[I] / DIVISOR of X.  */
struct expert_rows
{
  static constant constexpr bool contiguous = false;
  device const uint *entries;
  uint divisor;

  uint
  input (uint token) const
  {
    return entries[token] / divisor;
  }

  uint
  output (uint token) const
  {
    return entries[token];
  }
};

/* Compute the tile of rows FIRST_ROW onward and tokens FIRST_TOKEN
   onward of the product of WEIGHTS of format F, which has N_ROWS rows of
   N_COLS weights, with N_TOKENS tokens whose rows of X and Y come from
   ROWS, with the tiles, function constants, and stores of matmul_q8_0.
   Each pass of the main loop dequantizes one 32-weight group of every
   row of the tile.  The input tile holds elements of type S.  Half
   inputs made LFM2.5-2.6B QAD-Q4_0 prefill about 6 ms faster than half
   weights expanded for the tensor path on M4 Pro, and slowed Q4_K_M.
   SCRATCH holds the weight and input tiles, which the output tile
   reuses after the final multiply.

   When PAIRED is true, the tile also multiplies UP_WEIGHTS, the up
   projection of the SwiGLU whose gate is WEIGHTS, by the same inputs and
   stores SiLU of each gate result times the matching up result.  The
   tiles share the inputs, and SCRATCH holds twice the output tile.  */
template <typename F, typename S, bool PAIRED, typename R>
static void
matmul_tile (device const uchar *weights, device const uchar *up_weights,
             device const float *x, device float *y, uint n_rows, uint n_cols,
             uint n_tokens, R rows, uint first_row, uint first_token, uint tid,
             uint simdgroup_index, threadgroup uchar *scratch)
{
  threadgroup half *weight_tile = (threadgroup half *)scratch;
  threadgroup S *input_tile
      = (threadgroup S *)(scratch + MATMUL_ROWS * QK8_0 * sizeof (half));
  threadgroup half *up_tile
      = (threadgroup half *)(scratch + MATMUL_ROWS * QK8_0 * sizeof (half)
                             + MATMUL_TOKENS * QK8_0 * sizeof (float));
  threadgroup float *out_tile = (threadgroup float *)scratch;

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
  threadgroup vec<S, 4> *input_slot
      = (threadgroup vec<S, 4> *)(input_tile
                                  + 64 * (4 * input_part + input_token / 8)
                                  + 8 * (input_token % 8));
  device const uchar *row_weights = weights + row * row_bytes;
  device const uchar *up_row_weights = up_weights + row * row_bytes;
  device const float4 *token_inputs
      = (device const float4 *)(x + ulong (rows.input (token)) * n_cols
                                + input_part * QUANTS_PER_LANE);

  uint row_block_base = 4 * (simdgroup_index % 2);
  uint token_block_base = 2 * (simdgroup_index / 2);
  simdgroup_float8x8 acc[8];
  simdgroup_float8x8 up_acc[PAIRED ? 8 : 1];
  for (uint i = 0; i < 8; i++)
    acc[i] = make_filled_simdgroup_matrix<float, 8, 8> (0.0f);
  if (PAIRED)
    for (uint i = 0; i < 8; i++)
      up_acc[i] = make_filled_simdgroup_matrix<float, 8, 8> (0.0f);

  for (uint g = 0; g < n_groups; g++)
    {
      /* Load this group from device memory into registers first, then
         wait for every simdgroup to finish multiplying the previous
         group.  The loads overlap that work.  */
      uint e = g * QK8_0 + 16 * weight_half;
      half4 w[4];
      F::load16 (row_weights, e, w);
      half4 up_w[4];
      if (PAIRED)
        F::load16 (up_row_weights, e, up_w);
      device const float4 *xs = token_inputs + g * (QK8_0 / 4);
      vec<S, 4> in0 = vec<S, 4> (xs[0]);
      vec<S, 4> in1 = vec<S, 4> (xs[1]);
      threadgroup_barrier (mem_flags::mem_threadgroup);

      for (ushort v = 0; v < 4; v++)
        for (ushort c = 0; c < 4; c++)
          {
            ushort kk = 4 * v + c;
            ushort slot = weight_base + 64 * 8 * (kk / 8) + 8 * (kk % 8);
            weight_tile[slot] = w[v][c];
            if (PAIRED)
              up_tile[slot] = up_w[v][c];
          }
      input_slot[0] = in0;
      input_slot[1] = in1;
      threadgroup_barrier (mem_flags::mem_threadgroup);

#pragma unroll
      for (ushort k = 0; k < QK8_0 / 8; k++)
        {
          simdgroup_half8x8 a[4];
          simdgroup_matrix<S, 8, 8> bm[2];
          simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
          for (ushort i = 0; i < 4; i++)
            simdgroup_load (
                a[i], weight_tile + 64 * (8 * k + row_block_base + i), 8);
          simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
          for (ushort j = 0; j < 2; j++)
            simdgroup_load (
                bm[j], input_tile + 64 * (4 * k + token_block_base + j), 8);
          simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
          for (ushort j = 0; j < 2; j++)
#pragma unroll
            for (ushort i = 0; i < 4; i++)
              simdgroup_multiply_accumulate (acc[4 * j + i], bm[j], a[i],
                                             acc[4 * j + i]);
          if (PAIRED)
            {
              simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
              for (ushort i = 0; i < 4; i++)
                simdgroup_load (
                    a[i], up_tile + 64 * (8 * k + row_block_base + i), 8);
              simdgroup_barrier (mem_flags::mem_none);
#pragma unroll
              for (ushort j = 0; j < 2; j++)
#pragma unroll
                for (ushort i = 0; i < 4; i++)
                  simdgroup_multiply_accumulate (up_acc[4 * j + i], bm[j],
                                                 a[i], up_acc[4 * j + i]);
            }
        }
    }
  threadgroup_barrier (mem_flags::mem_threadgroup);

  uint row_offset = 8 * row_block_base;
  uint token_offset = 8 * token_block_base;
  if (PAIRED)
    {
      threadgroup float *up_out = out_tile + MATMUL_ROWS * MATMUL_TOKENS;
      for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 4; i++)
          {
            uint offset
                = (token_offset + 8 * j) * MATMUL_ROWS + row_offset + 8 * i;
            simdgroup_store (acc[4 * j + i], out_tile + offset, MATMUL_ROWS);
            simdgroup_store (up_acc[4 * j + i], up_out + offset, MATMUL_ROWS);
          }
      threadgroup_barrier (mem_flags::mem_threadgroup);
      for (uint idx = tid; idx < MATMUL_ROWS * MATMUL_TOKENS;
           idx += MATMUL_SIMDGROUPS * SIMD_WIDTH)
        {
          uint out_row = first_row + idx % MATMUL_ROWS;
          uint out_token = first_token + idx / MATMUL_ROWS;
          if (out_row < n_rows && out_token < n_tokens)
            {
              float gate = out_tile[idx];
              y[ulong (rows.output (out_token)) * n_rows + out_row]
                  = gate / (1.0f + precise::exp (-gate)) * up_out[idx];
            }
        }
      return;
    }
  bool full = first_row + MATMUL_ROWS <= n_rows
              && first_token + MATMUL_TOKENS <= n_tokens;
  if (R::contiguous && full && !accumulate && !swiglu_store)
    {
      device float *out = y + ulong (first_token + token_offset) * n_rows
                          + first_row + row_offset;
      for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 4; i++)
          simdgroup_store (acc[4 * j + i],
                           out + ulong (8 * j) * n_rows + 8 * i, n_rows);
      return;
    }
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
          device float *dst
              = y + ulong (rows.output (out_token)) * n_rows + out_row;
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

/* Multiply the matrix WEIGHTS of format F, which has N_ROWS rows of
   N_COLS weights, by each of the N_TOKENS rows of N_COLS floats at X,
   with one threadgroup per tile.  */
template <typename F, typename S = float>
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
  threadgroup uchar scratch[MATMUL_ROWS * QK8_0 * sizeof (half)
                            + MATMUL_TOKENS * QK8_0 * sizeof (float)];
  matmul_tile<F, S, false> (weights, weights, x, y, n_rows, n_cols, n_tokens,
                            dense_rows{}, position.y * MATMUL_ROWS,
                            position.x * MATMUL_TOKENS, thread_position.x,
                            simdgroup_index, scratch);
}

/* Multiply each expert of the stacked matrix WEIGHTS of format F, with
   N_ROWS rows of N_COLS weights per expert, by the tokens routed to it.
   Expert E's tokens are entries OFFSETS[E] through OFFSETS[E + 1] of
   ENTRIES, as moe_group wrote them.  Entry V names row V of Y and row
   V / DIVISOR of X.  Threadgroup (I, J, E) computes token tile I and row
   tile J of expert E, and tiles past the expert's tokens return.  The
   products overwrite Y.  With swiglu_store, WEIGHTS holds the gate
   projections and UP the up projections of a SwiGLU, and Y receives
   SiLU of each gate result times the matching up result.  */
template <typename F, typename S = float>
kernel void
matmul_experts (device const uchar *weights [[buffer (0)]],
                device const float *x [[buffer (1)]],
                device float *y [[buffer (2)]],
                constant uint &n_rows [[buffer (3)]],
                constant uint &n_cols [[buffer (4)]],
                device const uint *offsets [[buffer (5)]],
                device const uint *entries [[buffer (6)]],
                constant uint &divisor [[buffer (7)]],
                device const uchar *up
                [[buffer (8), function_constant (swiglu_store)]],
                uint3 position [[threadgroup_position_in_grid]],
                uint3 thread_position [[thread_position_in_threadgroup]],
                uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
  /* The paired tile stores two output tiles.  */
  threadgroup uchar scratch[2 * MATMUL_ROWS * MATMUL_TOKENS * sizeof (float)];
  uint expert = position.z;
  uint begin = offsets[expert];
  uint n_tokens = offsets[expert + 1] - begin;
  uint first_token = position.x * MATMUL_TOKENS;
  if (first_token >= n_tokens)
    return;
  ulong offset = expert * (ulong (n_rows) * row_bytes_of<F> (n_cols));
  expert_rows rows{ entries + begin, divisor };
  uint first_row = position.y * MATMUL_ROWS;
  if (swiglu_store)
    matmul_tile<F, S, true> (weights + offset, up + offset, x, y, n_rows,
                             n_cols, n_tokens, rows, first_row, first_token,
                             thread_position.x, simdgroup_index, scratch);
  else
    matmul_tile<F, S, false> (weights + offset, weights + offset, x, y, n_rows,
                              n_cols, n_tokens, rows, first_row, first_token,
                              thread_position.x, simdgroup_index, scratch);
}

template [[host_name (
    "matmul_experts_q4_0")]] kernel decltype (matmul_experts<q4_0_format,
                                                             half>)
    matmul_experts<q4_0_format, half>;
template [[host_name (
    "matmul_experts_q4k")]] kernel decltype (matmul_experts<q4k_format, half>)
    matmul_experts<q4k_format, half>;
template [[host_name (
    "matmul_experts_q6k")]] kernel decltype (matmul_experts<q6k_format, half>)
    matmul_experts<q6k_format, half>;

template
    [[host_name ("matmul_q4_0")]] kernel decltype (matmul_q<q4_0_format, half>)
        matmul_q<q4_0_format, half>;
template [[host_name ("matmul_f16")]] kernel decltype (matmul_q<f16_format>)
    matmul_q<f16_format>;
template [[host_name ("matmul_q4k")]] kernel decltype (matmul_q<q4k_format>)
    matmul_q<q4k_format>;
template [[host_name ("matmul_q6k")]] kernel decltype (matmul_q<q6k_format>)
    matmul_q<q6k_format>;
