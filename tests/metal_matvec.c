/* metal_matvec.c checks the Metal Q8_0 matrix-vector kernels against the
   scalar reference on random matrices, including row counts that leave
   a partial threadgroup.  Each shape runs as a plain multiply, with the
   RMS norm fused in, accumulating into its output, and as the fused
   SwiGLU pair.  */

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

/* The kinds of launch the test checks.  */
enum mode
{
  MODE_PLAIN,
  MODE_NORM,
  MODE_ACCUMULATE,
  MODE_SWIGLU,
  N_MODES
};

/* The name of each mode, indexed by enum mode.  */
static const char *const mode_names[N_MODES]
    = { "plain", "norm", "accumulate", "swiglu" };

/* The largest error allowed, relative to the largest magnitude in the
   scalar result.  The GPU sums in float32 and the scalar code sums in
   double, and rows of a few thousand elements stay far below this
   bound.  */
static const double TOLERANCE = 1e-5;

/* The epsilon of the fused RMS norm.  */
static const float NORM_EPS = 1e-5f;

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

/* Return a new buffer on METAL of SIZE bytes, exiting when it cannot be
   allocated.  */
static struct gip_metal_buffer *
must_buffer (struct gip_metal *metal, size_t size)
{
  struct gip_metal_buffer *buffer = gip_metal_buffer_new (metal, size);
  if (buffer == NULL)
    {
      fprintf (stderr, "metal_matvec: out of memory\n");
      exit (EXIT_FAILURE);
    }
  return buffer;
}

/* Fill the WEIGHT_BYTES bytes at W with random Q8_0 blocks.  */
static void
fill_q8_0 (unsigned char *w, size_t weight_bytes)
{
  size_t n_blocks = weight_bytes / GIP_Q8_0_BLOCK_BYTES;

  for (size_t b = 0; b < n_blocks; b++)
    {
      unsigned char *block = w + b * GIP_Q8_0_BLOCK_BYTES;
      uint16_t scale = random_scale_bits ();
      memcpy (block, &scale, sizeof scale);
      for (int j = 0; j < GIP_Q8_0_BLOCK_ELEMENTS; j++)
        block[2 + j] = (unsigned char)(next_random () & 0xff);
    }
}

/* Return the largest difference between the N floats at GOT and WANT,
   divided by the largest magnitude in WANT.  */
static double
relative_error (const float *got, const float *want, size_t n)
{
  double max_diff = 0.0;
  double max_want = 0.0;

  for (size_t i = 0; i < n; i++)
    {
      double diff = fabs ((double)got[i] - want[i]);
      if (isnan (diff) || diff > max_diff)
        max_diff = isnan (diff) ? INFINITY : diff;
      if (fabs (want[i]) > max_want)
        max_want = fabs (want[i]);
    }
  return max_want > 0.0 ? max_diff / max_want : max_diff;
}

/* Check a launch of MODE on random N_ROWS by N_COLS matrices on METAL.
   Report whether the results match.  */
static int
check (struct gip_metal *metal, enum mode mode, uint32_t n_rows,
       uint32_t n_cols)
{
  size_t weight_bytes
      = gip_tensor_row_bytes (GIP_TENSOR_Q8_0, n_cols) * n_rows;
  struct gip_metal_buffer *gate = must_buffer (metal, weight_bytes);
  struct gip_metal_buffer *up = must_buffer (metal, weight_bytes);
  struct gip_metal_buffer *x = must_buffer (metal, n_cols * sizeof (float));
  struct gip_metal_buffer *norm = must_buffer (metal, n_cols * sizeof (float));
  struct gip_metal_buffer *y = must_buffer (metal, n_rows * sizeof (float));
  float *normed = malloc (n_cols * sizeof (float));
  float *want = malloc (n_rows * sizeof (float));
  float *want_up = malloc (n_rows * sizeof (float));
  if (normed == NULL || want == NULL || want_up == NULL)
    {
      fprintf (stderr, "metal_matvec: out of memory\n");
      exit (EXIT_FAILURE);
    }

  unsigned char *gate_w = gip_metal_buffer_contents (gate);
  unsigned char *up_w = gip_metal_buffer_contents (up);
  float *xv = gip_metal_buffer_contents (x);
  float *norm_w = gip_metal_buffer_contents (norm);
  float *yv = gip_metal_buffer_contents (y);
  fill_q8_0 (gate_w, weight_bytes);
  fill_q8_0 (up_w, weight_bytes);
  for (uint32_t i = 0; i < n_cols; i++)
    {
      xv[i] = random_unit ();
      norm_w[i] = 1.0f + 0.5f * random_unit ();
    }
  for (uint32_t r = 0; r < n_rows; r++)
    yv[r] = random_unit ();

  /* The scalar reference: normalize when the mode fuses the norm, then
     multiply, then combine with the prior output or the up matrix.  */
  const float *input = xv;
  if (mode == MODE_NORM || mode == MODE_SWIGLU)
    {
      gip_rms_norm (xv, norm_w, n_cols, NORM_EPS, normed);
      input = normed;
    }
  gip_matvec (GIP_TENSOR_Q8_0, gate_w, n_rows, n_cols, input, want);
  if (mode == MODE_ACCUMULATE)
    for (uint32_t r = 0; r < n_rows; r++)
      want[r] += yv[r];
  if (mode == MODE_SWIGLU)
    {
      gip_matvec (GIP_TENSOR_Q8_0, up_w, n_rows, n_cols, input, want_up);
      for (uint32_t r = 0; r < n_rows; r++)
        want[r] = gip_silu (want[r]) * want_up[r];
    }

  if (gip_metal_begin (metal) != GIP_OK)
    {
      fprintf (stderr, "metal_matvec: cannot start a command buffer\n");
      exit (EXIT_FAILURE);
    }
  struct gip_metal_matvec_options options = { .eps = NORM_EPS };
  if (mode == MODE_NORM)
    options.norm_weight = gip_metal_at (norm, 0);
  options.accumulate = mode == MODE_ACCUMULATE;
  if (mode == MODE_SWIGLU)
    gip_metal_matvec_q8_0_swiglu (metal, gip_metal_at (gate, 0),
                                  gip_metal_at (up, 0), n_rows, n_cols,
                                  gip_metal_at (x, 0), gip_metal_at (norm, 0),
                                  NORM_EPS, gip_metal_at (y, 0));
  else
    gip_metal_matvec_q8_0 (metal, gip_metal_at (gate, 0), n_rows, n_cols,
                           gip_metal_at (x, 0), gip_metal_at (y, 0), &options);
  if (gip_metal_end (metal, NULL) != GIP_OK)
    {
      fprintf (stderr, "metal_matvec: the command buffer failed\n");
      exit (EXIT_FAILURE);
    }

  double error = relative_error (yv, want, n_rows);
  int ok = error <= TOLERANCE;
  printf ("%-10s %5u x %5u  relative error %.3e  %s\n", mode_names[mode],
          n_rows, n_cols, error, ok ? "ok" : "FAIL");

  free (normed);
  free (want);
  free (want_up);
  gip_metal_buffer_free (gate);
  gip_metal_buffer_free (up);
  gip_metal_buffer_free (x);
  gip_metal_buffer_free (norm);
  gip_metal_buffer_free (y);
  return ok;
}

/* Run every mode on several shapes and compare each against the scalar
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
  for (int mode = 0; mode < N_MODES; mode++)
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
      all_ok &= check (metal, (enum mode)mode, shapes[i][0], shapes[i][1]);

  gip_metal_close (metal);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
