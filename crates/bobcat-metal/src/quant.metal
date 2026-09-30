/* Q4_0, Q4_K, and Q6_K formats and K-quant lane helpers.  The backend
   compiles this source after common.metal.

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

/* Plain half-precision weights, stored as contiguous rows.  */
struct f16_format
{
  static constant constexpr uint block_weights = 32;
  static constant constexpr uint block_bytes = 64;

  static weights8
  load8 (device const uchar *row, uint e)
  {
    device const half4 *values = (device const half4 *)(row + 2 * e);
    return { float4 (values[0]), float4 (values[1]) };
  }

  static void
  load16 (device const uchar *row, uint e, thread half4 *out)
  {
    device const half4 *values = (device const half4 *)(row + 2 * e);
    for (uint i = 0; i < 4; i++)
      out[i] = values[i];
  }
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

  /* Return weights E through E + 15 of ROW, where 16 divides E, which
     are one nibble of each of the block's 16 quant bytes.  Each product
     of the scale and a quant fits in half precision with one rounding,
     as in load8.  */
  static void
  load16 (device const uchar *row, uint e, thread half4 *out)
  {
    device const uchar *block = row + (e / 32) * block_bytes;
    device const packed_uchar4 *quants
        = (device const packed_uchar4 *)(block + 2);
    half d = *(device const half *)block;
    uint shift = e % 32 < 16 ? 0 : 4;
    for (uint i = 0; i < 4; i++)
      {
        uchar4 q = uchar4 (quants[i]);
        out[i] = d * half4 (short4 ((q >> shift) & 15) - 8);
      }
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

  static void
  load16 (device const uchar *row, uint e, thread half4 *out)
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
    for (uint i = 0; i < 4; i++)
      {
        uint4 q = uint4 (quants[4 * i], quants[4 * i + 1], quants[4 * i + 2],
                         quants[4 * i + 3]);
        out[i] = half4 (group_scale * float4 ((q >> shift) & 0xf) - group_min);
      }
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

  static void
  load16 (device const uchar *row, uint e, thread half4 *out)
  {
    weights8 first = load8 (row, e);
    weights8 second = load8 (row, e + 8);
    out[0] = half4 (first.low);
    out[1] = half4 (first.high);
    out[2] = half4 (second.low);
    out[3] = half4 (second.high);
  }
};

/* Expand one matrix of format F into contiguous half-precision rows.
   Each thread expands 16 weights.  */
template <typename F>
kernel void
expand (device const uchar *weights [[buffer (0)]],
        device half *out [[buffer (1)]], constant uint &n_rows [[buffer (2)]],
        constant uint &n_cols [[buffer (3)]],
        uint index [[thread_position_in_grid]])
{
  uint chunks_per_row = n_cols / 16;
  if (ulong (index) >= ulong (n_rows) * chunks_per_row)
    return;
  uint row = index / chunks_per_row;
  uint e = (index % chunks_per_row) * 16;
  half4 values[4];
  F::load16 (weights
                 + ulong (row) * (n_cols / F::block_weights) * F::block_bytes,
             e, values);
  device half4 *dst = (device half4 *)(out + ulong (row) * n_cols + e);
  for (uint i = 0; i < 4; i++)
    dst[i] = values[i];
}

template [[host_name (
    "expand_q4_0")]] kernel decltype (expand<q4_0_format>) expand<q4_0_format>;
template [[host_name (
    "expand_q4k")]] kernel decltype (expand<q4k_format>) expand<q4k_format>;
template [[host_name (
    "expand_q6k")]] kernel decltype (expand<q6k_format>) expand<q6k_format>;

/* Return the bytes of one row of N_COLS weights of format F.  */
template <typename F>
static ulong
row_bytes_of (uint n_cols)
{
  return ulong (n_cols / F::block_weights) * F::block_bytes;
}

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
  static constant constexpr uint block_weights = 256;
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
  static constant constexpr uint block_weights = 256;
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

/* Q4_0 lanes, after llama.cpp's kernel_mul_mv_q4_0_f32: lane L covers
   blocks L / 2, L / 2 + 16, and so on.  Within a block it covers the 8
   bytes at 8 (L % 2), whose low nibbles hold weights 8 (L % 2) onward
   and whose high nibbles hold the weights 16 further.  */
struct q4_0_lanes
{
  static constant constexpr uint block_weights = 32;
  static constant constexpr uint blocks_per_pass = 16;
  static constant constexpr uint block_bytes = 18;

  struct inputs
  {
    /* The inputs of the lane's low and high nibbles, each divided by
       the place value its nibble has inside a 16-bit load: 1 or 256
       for low nibbles and 16 or 4096 for high ones.  */
    float low[8];
    float high[8];
    /* The sum of the lane's 16 inputs, which multiplies the offset of
       8 that every quant carries.  */
    float sum;
  };

  static uint
  first_block (uint lane)
  {
    return lane / 2;
  }

  static void
  load (device const float *x, device const float *norm_weight, uint block,
        uint lane, thread inputs &in, thread float &sum_squares)
  {
    uint base = block * 32 + 8 * (lane % 2);
    in.sum = 0.0f;
    for (uint i = 0; i < 8; i += 2)
      {
        uint at[4] = { base + i, base + i + 1, base + i + 16, base + i + 17 };
        float4 v = float4 (x[at[0]], x[at[1]], x[at[2]], x[at[3]]);
        if (fuse_norm)
          {
            sum_squares += dot (v, v);
            v *= float4 (norm_weight[at[0]], norm_weight[at[1]],
                         norm_weight[at[2]], norm_weight[at[3]]);
          }
        in.sum += v[0] + v[1] + v[2] + v[3];
        in.low[i] = v[0];
        in.low[i + 1] = v[1] / 256.0f;
        in.high[i] = v[2] / 16.0f;
        in.high[i + 1] = v[3] / 4096.0f;
      }
  }

  /* Return the lane's part of the dot product of block BLOCK of ROW
     with its loaded inputs IN.  Each 16-bit load holds two quant bytes,
     and the masks keep one nibble in place, which the scaled inputs
     undo.  */
  static float
  dot_part (device const uchar *row, uint block, uint lane,
            thread const inputs &in)
  {
    device const uchar *b = row + block * block_bytes;
    float d = float (*(device const half *)b);
    device const ushort *q = (device const ushort *)(b + 2) + 4 * (lane % 2);
    float acc = 0.0f;
    for (uint i = 0; i < 4; i++)
      {
        acc += in.low[2 * i] * float (q[i] & 0x000f)
               + in.low[2 * i + 1] * float (q[i] & 0x0f00)
               + in.high[2 * i] * float (q[i] & 0x00f0)
               + in.high[2 * i + 1] * float (q[i] & 0xf000);
      }
    return d * (acc - 8.0f * in.sum);
  }
};
