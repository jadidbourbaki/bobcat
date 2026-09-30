/* Metal 4 matrix products with half weights and float inputs.  */

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>

/* The host's TENSOR_MATMUL_TOKENS uses the same tile width.  */
constant uint TENSOR_TOKENS = 64;
/* The host's TENSOR_MATMUL_ROWS uses the same tile height.  On M4 Pro,
   32-row tiles improve 6144-row and square matrix timings by about 2%.  */
constant uint TENSOR_ROWS = 32;

kernel void
matmul_tensor_f16 (device half *weights [[buffer (0)]],
                   device float *x [[buffer (1)]],
                   device float *y [[buffer (2)]],
                   constant uint &n_rows [[buffer (3)]],
                   constant uint &n_cols [[buffer (4)]],
                   constant uint &n_tokens [[buffer (5)]],
                   uint2 position [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]])
{
  using namespace mpp::tensor_ops;
  uint first_token = position.x * TENSOR_TOKENS;
  uint first_row = position.y * TENSOR_ROWS;
  auto a = tensor (x + ulong (first_token) * n_cols,
                   dextents<int, 2> (n_cols, n_tokens - first_token));
  auto b = tensor (weights + ulong (first_row) * n_cols,
                   dextents<int, 2> (n_cols, n_rows - first_row));

  /* The precision flag stays false so float activations retain their
     precision.  The descriptor transposes weights into dot products.  */
  matmul2d<matmul2d_descriptor (TENSOR_TOKENS, TENSOR_ROWS, dynamic_extent,
                                false, true, false),
           execution_simdgroups<MATMUL_SIMDGROUPS>>
      operation;
  auto result
      = operation.get_destination_cooperative_tensor<decltype (a),
                                                     decltype (b), float> ();
  operation.run (a, b, result);
  if (!accumulate && !swiglu_store)
    {
      auto out = tensor (y, dextents<int, 2> (n_rows, n_tokens));
      result.store (out.slice (int (first_row), int (first_token)));
      return;
    }

  threadgroup float scratch[TENSOR_ROWS * TENSOR_TOKENS];
  auto out = tensor (scratch, extents<int, TENSOR_ROWS, TENSOR_TOKENS> ());
  result.store (out);
  threadgroup_barrier (mem_flags::mem_threadgroup);
  for (uint e = tid; e < TENSOR_ROWS * TENSOR_TOKENS;
       e += MATMUL_SIMDGROUPS * SIMD_WIDTH)
    {
      uint row = first_row + e % TENSOR_ROWS;
      uint token = first_token + e / TENSOR_ROWS;
      if (row < n_rows && token < n_tokens)
        {
          device float *dst = y + ulong (token) * n_rows + row;
          float value = scratch[e];
          if (swiglu_store)
            {
              float gate = *dst;
              *dst = gate / (1.0f + precise::exp (-gate)) * value;
            }
          else
            *dst += value;
        }
    }
}
