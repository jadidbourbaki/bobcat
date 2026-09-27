/* scalar.h declares the scalar reference implementation of every op.
   Every faster kernel is tested against these functions.  */

#ifndef GIP_KERNELS_SCALAR_H
#define GIP_KERNELS_SCALAR_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

/* Return the float value of the IEEE half-precision bits H.  */
float gip_fp16_to_fp32 (uint16_t h);

/* Return the float value of the bfloat16 bits H.  */
float gip_bf16_to_fp32 (uint16_t h);

/* Dequantize row ROW of the matrix at WEIGHTS, whose rows hold N_COLS
   elements of TYPE, into the N_COLS floats at OUT.  */
void gip_get_row (enum gip_tensor_type type, const void *weights,
                  size_t n_cols, size_t row, float *out);

/* Multiply the matrix at WEIGHTS, which has N_ROWS rows of N_COLS
   elements of TYPE, by the N_COLS floats at X.  Store the N_ROWS
   results at Y.  */
void gip_matvec (enum gip_tensor_type type, const void *weights, size_t n_rows,
                 size_t n_cols, const float *x, float *y);

/* Normalize the N floats at X by their root mean square, scale them by
   the N floats at WEIGHT, and store the result at OUT.  EPS guards the
   division.  OUT may equal X.  */
void gip_rms_norm (const float *x, const float *weight, size_t n, float eps,
                   float *out);

/* Rotate the HEAD_DIM floats at VEC for position POS with base THETA.
   Element I pairs with element I + HEAD_DIM / 2, as in GPT-NeoX.  */
void gip_rope_neox (float *vec, size_t head_dim, uint32_t pos, float theta);

/* Return X times the logistic sigmoid of X.  */
float gip_silu (float x);

/* Replace the N floats at X with their softmax.  */
void gip_softmax (float *x, size_t n);

/* Return the dot product of the N floats at A and B.  */
float gip_dot (const float *a, const float *b, size_t n);

#endif /* GIP_KERNELS_SCALAR_H */
