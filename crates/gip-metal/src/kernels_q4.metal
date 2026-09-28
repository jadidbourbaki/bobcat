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

/* The K-quant matrix-vector kernels follow llama.cpp's
   kernel_mul_mv_q4_K_f32 and kernel_mul_mv_q6_K_f32.  Each simdgroup
   computes K_ROWS_PER_SIMDGROUP whole rows, so its lanes read every
   input exactly once and a simd_sum finishes each row with no
   threadgroup reduction.  A lane covers a fixed set of weights in each
   super-block it visits, unpacks the super-block's group scales once,
   and folds each group's scale in after summing the group's quants.

   The format structs below hold one lane's preloaded inputs and its
   partial dot product for one super-block.  */

/* Each lane's input loads serve every row of its simdgroup, and with
   the norm fused a lane loads 64 floats per super-block.  Four rows
   spread those loads over enough weights to keep the kernel near the
   memory bound.  backend.rs gives the measurements, and its
   K_QUANT_ROWS_PER_SIMDGROUP must match.  */
enum
{
  K_ROWS_PER_SIMDGROUP = 4
};

/* Q4_K lanes: lane L covers super-blocks L / 8, L / 8 + 4, and so on.
   Within a super-block it covers the 8 weights at 64 IQ + 8 IR, the 8
   at 32 past those, and the two sets 128 further, where IQ is
   (L % 8) / 4 and IR is L % 4.  */
struct q4k_lanes
{
  static constant constexpr uint blocks_per_pass = 4;
  static constant constexpr uint block_bytes = 144;

  struct inputs
  {
    float low[16];
    float high[16];
    /* The sums of the inputs of each of the lane's four groups, which
       multiply the groups' minimums.  */
    float4 sums;
  };

  static uint
  first_block (uint lane)
  {
    return lane / 8;
  }

  /* Load the lane's 32 inputs of super-block BLOCK from X.  With
     fuse_norm, add their squares to SUM_SQUARES and scale each by its
     entry of NORM_WEIGHT.  */
  static void
  load (device const float *x, device const float *norm_weight, uint block,
        uint lane, thread inputs &in, thread float &sum_squares)
  {
    uint iq = (lane % 8) / 4;
    uint ir = lane % 4;
    uint base = block * 256 + 64 * iq + 8 * ir;
    in.sums = 0.0f;
    for (uint i = 0; i < 8; i++)
      {
        uint at[4]
            = { base + i, base + i + 32, base + i + 128, base + i + 160 };
        float4 v = float4 (x[at[0]], x[at[1]], x[at[2]], x[at[3]]);
        if (fuse_norm)
          {
            sum_squares += dot (v, v);
            v *= float4 (norm_weight[at[0]], norm_weight[at[1]],
                         norm_weight[at[2]], norm_weight[at[3]]);
          }
        in.low[i] = v[0];
        in.low[i + 8] = v[1];
        in.high[i] = v[2];
        in.high[i + 8] = v[3];
        in.sums += v;
      }
  }

  /* Return the lane's part of the dot product of super-block BLOCK of
     ROW with its loaded inputs IN.  Each 16-bit load holds two quant
     bytes.  The masks keep one nibble in place, and the scale factors
     1/256 and 1/16 undo its position.  */
  static float
  dot_part (device const uchar *row, uint block, uint lane,
            thread const inputs &in)
  {
    uint iq = (lane % 8) / 4;
    uint ir = lane % 4;
    device const uchar *b = row + block * block_bytes;
    float d = float (*(device const half *)b);
    float dmin = float (*(device const half *)(b + 2));
    device const ushort *sc = (device const ushort *)(b + 4) + iq;
    device const ushort *q1
        = (device const ushort *)(b + 16) + 16 * iq + 4 * ir;
    device const ushort *q2 = q1 + 32;

    /* Unpack the 6-bit scales and minimums of the lane's four groups:
       bytes 0, 1, 4, and 5 hold scales and bytes 2, 3, 6, and 7 hold
       minimums.  */
    ushort packed[4];
    packed[0] = sc[0] & 0x3f3f;
    packed[1] = sc[2] & 0x3f3f;
    packed[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
    packed[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
    thread const uchar *s = (thread const uchar *)packed;

    float4 acc1 = 0.0f;
    float4 acc2 = 0.0f;
    for (uint i = 0; i < 4; i++)
      {
        acc1[0] += in.low[2 * i + 0] * float (q1[i] & 0x000f);
        acc1[1] += in.low[2 * i + 1] * float (q1[i] & 0x0f00);
        acc1[2] += in.low[2 * i + 8] * float (q1[i] & 0x00f0);
        acc1[3] += in.low[2 * i + 9] * float (q1[i] & 0xf000);
        acc2[0] += in.high[2 * i + 0] * float (q2[i] & 0x000f);
        acc2[1] += in.high[2 * i + 1] * float (q2[i] & 0x0f00);
        acc2[2] += in.high[2 * i + 8] * float (q2[i] & 0x00f0);
        acc2[3] += in.high[2 * i + 9] * float (q2[i] & 0xf000);
      }
    return d
               * ((acc1[0] + acc1[1] / 256.0f) * float (s[0])
                  + (acc1[2] + acc1[3] / 256.0f) * float (s[1]) / 16.0f
                  + (acc2[0] + acc2[1] / 256.0f) * float (s[4])
                  + (acc2[2] + acc2[3] / 256.0f) * float (s[5]) / 16.0f)
           - dmin
                 * (in.sums[0] * float (s[2]) + in.sums[1] * float (s[3])
                    + in.sums[2] * float (s[6]) + in.sums[3] * float (s[7]));
  }
};

/* Q6_K lanes: lane L covers super-blocks L % 2, L % 2 + 2, and so on.
   Within a super-block it covers 4 weights in each quarter of one
   128-weight half: the half IP is (L / 2) / 8, and the weights start at
   L0 = 4 ((L / 2) % 8) within each quarter.  */
struct q6k_lanes
{
  static constant constexpr uint blocks_per_pass = 2;
  static constant constexpr uint block_bytes = 210;

  struct inputs
  {
    float values[16];
  };

  static uint
  first_block (uint lane)
  {
    return lane % 2;
  }

  static void
  load (device const float *x, device const float *norm_weight, uint block,
        uint lane, thread inputs &in, thread float &sum_squares)
  {
    uint ip = (lane / 2) / 8;
    uint l0 = 4 * ((lane / 2) % 8);
    uint base = block * 256 + 128 * ip + l0;
    for (uint l = 0; l < 4; l++)
      {
        uint at[4] = { base + l, base + l + 32, base + l + 64, base + l + 96 };
        float4 v = float4 (x[at[0]], x[at[1]], x[at[2]], x[at[3]]);
        if (fuse_norm)
          {
            sum_squares += dot (v, v);
            v *= float4 (norm_weight[at[0]], norm_weight[at[1]],
                         norm_weight[at[2]], norm_weight[at[3]]);
          }
        for (uint q = 0; q < 4; q++)
          in.values[4 * l + q] = v[q];
      }
  }

  static float
  dot_part (device const uchar *row, uint block, uint lane,
            thread const inputs &in)
  {
    uint ip = (lane / 2) / 8;
    uint l0 = 4 * ((lane / 2) % 8);
    device const uchar *b = row + block * block_bytes;
    device const uchar *q1 = b + 64 * ip + l0;
    device const uchar *q2 = q1 + 32;
    device const uchar *qh = b + 128 + 32 * ip + l0;
    device const char *sc = (device const char *)(b + 192) + 8 * ip + l0 / 16;
    float d = float (*(device const half *)(b + 208));

    float4 sums = 0.0f;
    for (uint l = 0; l < 4; l++)
      {
        sums[0] += in.values[4 * l + 0]
                   * float (int ((q1[l] & 0xf) | ((qh[l] & 0x03) << 4)) - 32);
        sums[1] += in.values[4 * l + 1]
                   * float (int ((q2[l] & 0xf) | ((qh[l] & 0x0c) << 2)) - 32);
        sums[2] += in.values[4 * l + 2]
                   * float (int ((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32);
        sums[3] += in.values[4 * l + 3]
                   * float (int ((q2[l] >> 4) | ((qh[l] & 0xc0) >> 2)) - 32);
      }
    return d
           * (sums[0] * float (sc[0]) + sums[1] * float (sc[2])
              + sums[2] * float (sc[4]) + sums[3] * float (sc[6]));
  }
};

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
  uint first_row = (threadgroup_index * simdgroups + simdgroup_index)
                   * K_ROWS_PER_SIMDGROUP;
  uint n_blocks = n_cols / 256;
  ulong row_bytes = ulong (n_blocks) * K::block_bytes;
  float sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  float sum_squares = 0.0f;

  for (uint block = K::first_block (lane); block < n_blocks;
       block += K::blocks_per_pass)
    {
      typename K::inputs in;
      K::load (x, norm_weight, block, lane, in, sum_squares);
      for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
        if (first_row + r < n_rows)
          sums[r] += K::dot_part (weights + (first_row + r) * row_bytes, block,
                                  lane, in);
    }

  float scale = 1.0f;
  if (fuse_norm)
    scale = precise::rsqrt (simd_sum (sum_squares) / float (n_cols) + eps);
  for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
    {
      uint row = first_row + r;
      float total = simd_sum (sums[r]) * scale;
      if (lane == 0 && row < n_rows)
        y[row] = accumulate ? y[row] + total : total;
    }
}

template [[host_name (
    "matvec_q4k")]] kernel decltype (matvec_k<q4k_lanes>) matvec_k<q4k_lanes>;
template [[host_name (
    "matvec_q6k")]] kernel decltype (matvec_k<q6k_lanes>) matvec_k<q6k_lanes>;

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
  uint first_row = (threadgroup_index * simdgroups + simdgroup_index)
                   * K_ROWS_PER_SIMDGROUP;
  uint n_blocks = n_cols / 256;
  ulong row_bytes = ulong (n_blocks) * K::block_bytes;
  float gate_sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  float up_sums[K_ROWS_PER_SIMDGROUP] = { 0.0f };
  float sum_squares = 0.0f;

  for (uint block = K::first_block (lane); block < n_blocks;
       block += K::blocks_per_pass)
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

  float scale = 1.0f;
  if (fuse_norm)
    scale = precise::rsqrt (simd_sum (sum_squares) / float (n_cols) + eps);
  for (uint r = 0; r < K_ROWS_PER_SIMDGROUP; r++)
    {
      uint row = first_row + r;
      float g = simd_sum (gate_sums[r]) * scale;
      float u = simd_sum (up_sums[r]) * scale;
      if (lane == 0 && row < n_rows)
        y[row] = g / (1.0f + precise::exp (-g)) * u;
    }
}

template [[host_name (
    "matvec_q4k_swiglu")]] kernel decltype (matvec_k_swiglu<q4k_lanes>)
    matvec_k_swiglu<q4k_lanes>;
template [[host_name (
    "matvec_q6k_swiglu")]] kernel decltype (matvec_k_swiglu<q6k_lanes>)
    matvec_k_swiglu<q6k_lanes>;

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
