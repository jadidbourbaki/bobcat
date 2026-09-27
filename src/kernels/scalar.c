/* scalar.c is the scalar reference implementation of every op.  The
   code favors plain loops and double-precision sums, so it defines the
   correct output for faster kernels to match.  */

#include <math.h>
#include <string.h>

#include "kernels/scalar.h"

/* One Q8_0 block.  The parser guarantees that each row starts on a
   2-byte boundary, and each block is 34 bytes, so every block's scale
   is aligned.  */
struct q8_0_block
{
  uint16_t scale;
  int8_t quants[GIP_Q8_0_BLOCK_ELEMENTS];
};

_Static_assert (sizeof (struct q8_0_block) == GIP_Q8_0_BLOCK_BYTES,
                "a Q8_0 block is 34 bytes");

float
gip_fp16_to_fp32 (uint16_t h)
{
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x3ff;
  uint32_t bits;

  if (exponent == 0)
    {
      if (mantissa == 0)
        bits = sign;
      else
        {
          /* A subnormal half becomes a normal float.  Shift the mantissa
             until its leading one reaches the implicit bit.  */
          exponent = 127 - 15 + 1;
          while ((mantissa & 0x400) == 0)
            {
              mantissa <<= 1;
              exponent--;
            }
          mantissa &= 0x3ff;
          bits = sign | (exponent << 23) | (mantissa << 13);
        }
    }
  else if (exponent == 0x1f)
    bits = sign | 0x7f800000 | (mantissa << 13);
  else
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);

  float f;
  memcpy (&f, &bits, sizeof f);
  return f;
}

float
gip_bf16_to_fp32 (uint16_t h)
{
  uint32_t bits = (uint32_t)h << 16;
  float f;

  memcpy (&f, &bits, sizeof f);
  return f;
}

/* Return the dot product of the N_COLS elements of TYPE at ROW with the
   floats at X.  */
static double
dot_row (enum gip_tensor_type type, const unsigned char *row, const float *x,
         size_t n_cols)
{
  double sum = 0.0;

  switch (type)
    {
    case GIP_TENSOR_F32:
      {
        const float *w = (const float *)row;
        for (size_t i = 0; i < n_cols; i++)
          sum += (double)w[i] * x[i];
        break;
      }
    case GIP_TENSOR_F16:
      {
        const uint16_t *w = (const uint16_t *)row;
        for (size_t i = 0; i < n_cols; i++)
          sum += (double)gip_fp16_to_fp32 (w[i]) * x[i];
        break;
      }
    case GIP_TENSOR_BF16:
      {
        const uint16_t *w = (const uint16_t *)row;
        for (size_t i = 0; i < n_cols; i++)
          sum += (double)gip_bf16_to_fp32 (w[i]) * x[i];
        break;
      }
    case GIP_TENSOR_Q8_0:
      {
        const struct q8_0_block *blocks = (const struct q8_0_block *)row;
        size_t n_blocks = n_cols / GIP_Q8_0_BLOCK_ELEMENTS;
        for (size_t b = 0; b < n_blocks; b++)
          {
            const float *xb = x + b * GIP_Q8_0_BLOCK_ELEMENTS;
            double block_sum = 0.0;
            for (int j = 0; j < GIP_Q8_0_BLOCK_ELEMENTS; j++)
              block_sum += (double)blocks[b].quants[j] * xb[j];
            sum += gip_fp16_to_fp32 (blocks[b].scale) * block_sum;
          }
        break;
      }
    }
  return sum;
}

void
gip_get_row (enum gip_tensor_type type, const void *weights, size_t n_cols,
             size_t row, float *out)
{
  const unsigned char *p = (const unsigned char *)weights
                           + row * gip_tensor_row_bytes (type, n_cols);

  switch (type)
    {
    case GIP_TENSOR_F32:
      memcpy (out, p, n_cols * sizeof *out);
      break;
    case GIP_TENSOR_F16:
      {
        const uint16_t *w = (const uint16_t *)p;
        for (size_t i = 0; i < n_cols; i++)
          out[i] = gip_fp16_to_fp32 (w[i]);
        break;
      }
    case GIP_TENSOR_BF16:
      {
        const uint16_t *w = (const uint16_t *)p;
        for (size_t i = 0; i < n_cols; i++)
          out[i] = gip_bf16_to_fp32 (w[i]);
        break;
      }
    case GIP_TENSOR_Q8_0:
      {
        const struct q8_0_block *blocks = (const struct q8_0_block *)p;
        size_t n_blocks = n_cols / GIP_Q8_0_BLOCK_ELEMENTS;
        for (size_t b = 0; b < n_blocks; b++)
          {
            float scale = gip_fp16_to_fp32 (blocks[b].scale);
            for (int j = 0; j < GIP_Q8_0_BLOCK_ELEMENTS; j++)
              out[b * GIP_Q8_0_BLOCK_ELEMENTS + j]
                  = scale * blocks[b].quants[j];
          }
        break;
      }
    }
}

void
gip_matvec (enum gip_tensor_type type, const void *weights, size_t n_rows,
            size_t n_cols, const float *x, float *y)
{
  const unsigned char *base = weights;
  size_t row_bytes = gip_tensor_row_bytes (type, n_cols);

  for (size_t r = 0; r < n_rows; r++)
    y[r] = (float)dot_row (type, base + r * row_bytes, x, n_cols);
}

void
gip_rms_norm (const float *x, const float *weight, size_t n, float eps,
              float *out)
{
  double sum_squares = 0.0;

  for (size_t i = 0; i < n; i++)
    sum_squares += (double)x[i] * x[i];

  float scale = (float)(1.0 / sqrt (sum_squares / (double)n + eps));
  for (size_t i = 0; i < n; i++)
    out[i] = weight[i] * (x[i] * scale);
}

void
gip_rope_neox (float *vec, size_t head_dim, uint32_t pos, float theta)
{
  size_t half = head_dim / 2;

  for (size_t i = 0; i < half; i++)
    {
      /* transformers computes the frequencies and angles in float32.
         Matching its precision keeps the comparison tight.  */
      float inv_freq = 1.0f / powf (theta, (float)(2 * i) / (float)head_dim);
      float angle = (float)pos * inv_freq;
      float c = cosf (angle);
      float s = sinf (angle);
      float x0 = vec[i];
      float x1 = vec[i + half];
      vec[i] = x0 * c - x1 * s;
      vec[i + half] = x1 * c + x0 * s;
    }
}

float
gip_silu (float x)
{
  return x / (1.0f + expf (-x));
}

void
gip_softmax (float *x, size_t n)
{
  float max = x[0];
  double sum = 0.0;

  for (size_t i = 1; i < n; i++)
    if (x[i] > max)
      max = x[i];
  for (size_t i = 0; i < n; i++)
    {
      x[i] = expf (x[i] - max);
      sum += x[i];
    }
  for (size_t i = 0; i < n; i++)
    x[i] = (float)(x[i] / sum);
}

float
gip_dot (const float *a, const float *b, size_t n)
{
  double sum = 0.0;

  for (size_t i = 0; i < n; i++)
    sum += (double)a[i] * b[i];
  return (float)sum;
}
