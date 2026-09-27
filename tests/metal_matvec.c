/* metal_matvec.c checks the Metal Q8_0 matrix-vector kernel against the
   scalar reference on random matrices, including row counts that leave
   a partial threadgroup.  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels/scalar.h"
#include "metal/backend.h"

enum
{
  EXIT_SKIP = 77
};

/* The largest error allowed, relative to the largest magnitude in the
   scalar result.  The GPU sums in float32 and the scalar code sums in
   double, and rows of a few thousand elements stay far below this
   bound.  */
static const double TOLERANCE = 1e-5;

/* The state of a xorshift64 generator, fixed so every run sees the same
   matrices.  */
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

/* Return the next pseudo-random 64-bit number.  */
static uint64_t
next_random (void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

/* Return a pseudo-random float in [-1, 1).  */
static float
random_unit (void)
{
  return (float)((double)(next_random () >> 11) / (double)(1ULL << 52)) - 1.0f;
}

/* Return the IEEE half-precision bits of a pseudo-random scale in
   [2^-10, 2^-6), a range typical of Q8_0 weights.  */
static uint16_t
random_scale_bits (void)
{
  uint16_t exponent = (uint16_t)(5 + next_random () % 4);
  uint16_t mantissa = (uint16_t)(next_random () & 0x3ff);
  return (uint16_t)((exponent << 10) | mantissa);
}

/* Check the kernel on a random N_ROWS by N_COLS matrix on METAL.
   Report whether the results match.  */
static int
check_shape (struct gip_metal *metal, uint32_t n_rows, uint32_t n_cols)
{
  size_t row_bytes = gip_tensor_row_bytes (GIP_TENSOR_Q8_0, n_cols);
  size_t weight_bytes = row_bytes * n_rows;
  struct gip_metal_buffer *weights
      = gip_metal_buffer_new (metal, weight_bytes);
  struct gip_metal_buffer *x = gip_metal_buffer_new (metal, n_cols * 4);
  struct gip_metal_buffer *y = gip_metal_buffer_new (metal, n_rows * 4);
  float *want = malloc (n_rows * sizeof (float));
  if (weights == NULL || x == NULL || y == NULL || want == NULL)
    {
      fprintf (stderr, "metal_matvec: out of memory\n");
      exit (EXIT_FAILURE);
    }

  unsigned char *w = gip_metal_buffer_contents (weights);
  size_t n_blocks = weight_bytes / GIP_Q8_0_BLOCK_BYTES;
  for (size_t b = 0; b < n_blocks; b++)
    {
      unsigned char *block = w + b * GIP_Q8_0_BLOCK_BYTES;
      uint16_t scale = random_scale_bits ();
      memcpy (block, &scale, sizeof scale);
      for (int j = 0; j < GIP_Q8_0_BLOCK_ELEMENTS; j++)
        block[2 + j] = (unsigned char)(next_random () & 0xff);
    }
  float *xv = gip_metal_buffer_contents (x);
  for (uint32_t i = 0; i < n_cols; i++)
    xv[i] = random_unit ();

  gip_matvec (GIP_TENSOR_Q8_0, w, n_rows, n_cols, xv, want);

  if (gip_metal_begin (metal) != GIP_OK)
    {
      fprintf (stderr, "metal_matvec: cannot start a command buffer\n");
      exit (EXIT_FAILURE);
    }
  gip_metal_matvec_q8_0 (metal, gip_metal_at (weights, 0), n_rows, n_cols,
                         gip_metal_at (x, 0), gip_metal_at (y, 0));
  if (gip_metal_end (metal, NULL) != GIP_OK)
    {
      fprintf (stderr, "metal_matvec: the command buffer failed\n");
      exit (EXIT_FAILURE);
    }

  const float *got = gip_metal_buffer_contents (y);
  double max_diff = 0.0;
  double max_want = 0.0;
  for (uint32_t r = 0; r < n_rows; r++)
    {
      double diff = fabs ((double)got[r] - want[r]);
      if (isnan (diff) || diff > max_diff)
        max_diff = isnan (diff) ? INFINITY : diff;
      if (fabs (want[r]) > max_want)
        max_want = fabs (want[r]);
    }
  double error = max_want > 0.0 ? max_diff / max_want : max_diff;
  int ok = error <= TOLERANCE;
  printf ("%5u x %5u  relative error %.3e  %s\n", n_rows, n_cols, error,
          ok ? "ok" : "FAIL");

  free (want);
  gip_metal_buffer_free (weights);
  gip_metal_buffer_free (x);
  gip_metal_buffer_free (y);
  return ok;
}

/* Run the kernel on several shapes and compare each against the scalar
   reference.  */
int
main (void)
{
  struct gip_metal *metal;
  char err[512] = "";

  if (gip_metal_open (&metal, err, sizeof err) != GIP_OK)
    {
      printf ("skip: %s\n", err);
      return EXIT_SKIP;
    }

  /* The shapes cover the LFM2.5 projections, a row count that fills no
     threadgroup, and one that leaves a partial threadgroup.  */
  static const uint32_t shapes[][2] = {
    { 1, 32 },      { 7, 64 },      { 1000, 1024 }, { 1024, 1024 },
    { 3072, 1024 }, { 4608, 1024 }, { 1024, 4608 }, { 65536, 1024 },
  };
  int all_ok = 1;
  for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
    all_ok &= check_shape (metal, shapes[i][0], shapes[i][1]);

  gip_metal_close (metal);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
