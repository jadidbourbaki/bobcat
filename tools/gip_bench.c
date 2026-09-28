/* gip_bench.c measures gip's prefill and decode speed on the Metal GPU
   with the protocol llama-bench and mlx_lm.benchmark use: a prompt of
   fixed length, then a fixed number of generated tokens.  */

#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model_lfm2_metal.h"

enum
{
  DEFAULT_PROMPT = 512,
  DEFAULT_GENERATE = 128,
  DEFAULT_REPS = 5,
  /* The prompt repeats a fixed ordinary token.  Speed does not depend on
     which tokens the prompt holds.  */
  PROMPT_TOKEN = 1000
};

/* The long options gip_bench accepts.  */
static const struct option long_options[] = {
  { "prompt", required_argument, NULL, 'p' },
  { "generate", required_argument, NULL, 'n' },
  { "reps", required_argument, NULL, 'r' },
  { "profile", no_argument, NULL, 'P' },
  { "kv", required_argument, NULL, 'k' },
  { "help", no_argument, NULL, 'h' },
  { "version", no_argument, NULL, 'V' },
  { NULL, 0, NULL, 0 },
};

/* Print the usage message to STREAM.  */
static void
print_usage (FILE *stream)
{
  fprintf (stream,
           "Usage: gip_bench [OPTION]... MODEL.gguf\n"
           "Measure gip's prefill and decode speed on the Metal GPU.\n"
           "\n"
           "  -p, --prompt=N     prompt tokens (default %d)\n"
           "  -n, --generate=N   generated tokens (default %d)\n"
           "  -r, --reps=N       repetitions (default %d)\n"
           "  -P, --profile      break decode GPU time down by kernel\n"
           "  -k, --kv=TYPE      KV cache type, f16 or f32 (default f16)\n"
           "  -h, --help         display this help and exit\n"
           "  -V, --version      output version information and exit\n",
           DEFAULT_PROMPT, DEFAULT_GENERATE, DEFAULT_REPS);
}

/* Return the monotonic clock time in seconds.  */
static double
now_seconds (void)
{
  struct timespec ts;

  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Parse the positive integer ARG for OPTION into OUT.  Exit on invalid
   input.  */
static void
parse_count (const char *arg, const char *option, unsigned *out)
{
  char *end;

  errno = 0;
  unsigned long value = strtoul (arg, &end, 10);
  if (errno != 0 || *end != '\0' || value == 0 || value > 1000000)
    {
      fprintf (stderr, "gip_bench: invalid %s: %s\n", option, arg);
      exit (EXIT_FAILURE);
    }
  *out = (unsigned)value;
}

/* Return the index of the largest of the N floats at X.  */
static int32_t
argmax (const float *x, size_t n)
{
  size_t best = 0;

  for (size_t i = 1; i < n; i++)
    if (x[i] > x[best])
      best = i;
  return (int32_t)best;
}

/* Print the mean and standard deviation of the N rates at RATES under
   LABEL.  */
static void
print_rates (const char *label, const double *rates, unsigned n)
{
  double mean = 0.0;
  double variance = 0.0;

  for (unsigned i = 0; i < n; i++)
    mean += rates[i];
  mean /= n;
  for (unsigned i = 0; i < n; i++)
    variance += (rates[i] - mean) * (rates[i] - mean);
  variance = n > 1 ? variance / (n - 1) : 0.0;
  printf ("%-8s %9.2f ± %.2f tokens/s\n", label, mean, sqrt (variance));
}

/* Order profile entries by descending GPU time.  */
static int
compare_entries (const void *a, const void *b)
{
  const struct gip_metal_profile_entry *x = a;
  const struct gip_metal_profile_entry *y = b;

  return (x->seconds < y->seconds) - (x->seconds > y->seconds);
}

/* Prefill N_PROMPT tokens of MODEL on METAL, then decode N_GENERATE
   tokens with profiling on and print each kernel's GPU time per token.
   The KV cache holds half precision when KV_HALF is nonzero.  LOGITS
   holds the model's vocabulary size in floats.  */
static void
profile_decode (const struct gip_lfm2_model *model, struct gip_metal *metal,
                unsigned n_prompt, unsigned n_generate, int kv_half,
                float *logits)
{
  struct gip_lfm2_metal gpu;
  char err[512] = "";

  if (gip_lfm2_metal_init (model, metal, n_prompt + n_generate, kv_half, &gpu,
                           err, sizeof err)
      != GIP_OK)
    {
      fprintf (stderr, "gip_bench: %s\n", err);
      exit (EXIT_FAILURE);
    }
  for (unsigned t = 0; t < n_prompt; t++)
    gip_lfm2_metal_step (&gpu, PROMPT_TOKEN, t + 1 == n_prompt ? logits : NULL,
                         NULL);

  gip_metal_set_profiling (metal, 1);
  for (unsigned t = 0; t < n_generate; t++)
    gip_lfm2_metal_step (&gpu, argmax (logits, model->n_vocab), logits, NULL);
  gip_metal_set_profiling (metal, 0);
  gip_lfm2_metal_free (&gpu);

  const struct gip_metal_profile_entry *entries;
  size_t count = gip_metal_profile (metal, &entries);
  struct gip_metal_profile_entry *sorted = malloc (count * sizeof *sorted);
  if (sorted == NULL)
    {
      fprintf (stderr, "gip_bench: out of memory\n");
      exit (EXIT_FAILURE);
    }
  memcpy (sorted, entries, count * sizeof *sorted);
  qsort (sorted, count, sizeof *sorted, compare_entries);

  double total = 0.0;
  for (size_t i = 0; i < count; i++)
    total += sorted[i].seconds;
  printf ("\nprofiled decode, one command buffer per launch:\n");
  printf ("%-14s %13s %10s %9s %6s %8s\n", "kernel", "shape", "launches",
          "ms/token", "share", "GB/s");
  for (size_t i = 0; i < count; i++)
    {
      char shape[32] = "";
      if (sorted[i].n_rows != 0)
        snprintf (shape, sizeof shape, "%ux%u", sorted[i].n_rows,
                  sorted[i].n_cols);
      double ms = sorted[i].seconds * 1e3 / n_generate;
      printf ("%-14s %13s %10llu %9.3f %5.1f%%", sorted[i].name, shape,
              (unsigned long long)(sorted[i].calls / n_generate), ms,
              100.0 * sorted[i].seconds / total);
      if (sorted[i].bytes != 0)
        printf (" %8.1f", sorted[i].bytes / sorted[i].seconds / 1e9);
      printf ("\n");
    }
  printf ("%-14s %13s %10s %9.3f\n", "total", "", "",
          total * 1e3 / n_generate);
  free (sorted);
}

/* Parse the options in ARGC and ARGV, load the model, and time REPS
   runs of prefill and decode.  */
int
main (int argc, char **argv)
{
  unsigned n_prompt = DEFAULT_PROMPT;
  unsigned n_generate = DEFAULT_GENERATE;
  unsigned reps = DEFAULT_REPS;
  int profile = 0;
  int kv_half = 1;
  int opt;

  while ((opt = getopt_long (argc, argv, "p:n:r:Pk:hV", long_options, NULL))
         != -1)
    switch (opt)
      {
      case 'p':
        parse_count (optarg, "prompt length", &n_prompt);
        break;
      case 'n':
        parse_count (optarg, "generation length", &n_generate);
        break;
      case 'r':
        parse_count (optarg, "repetition count", &reps);
        break;
      case 'P':
        profile = 1;
        break;
      case 'k':
        if (strcmp (optarg, "f16") != 0 && strcmp (optarg, "f32") != 0)
          {
            fprintf (stderr, "gip_bench: invalid KV cache type: %s\n", optarg);
            return EXIT_FAILURE;
          }
        kv_half = strcmp (optarg, "f16") == 0;
        break;
      case 'h':
        print_usage (stdout);
        return EXIT_SUCCESS;
      case 'V':
        printf ("gip_bench (gip) %s\n", GIP_VERSION);
        return EXIT_SUCCESS;
      default:
        print_usage (stderr);
        return EXIT_FAILURE;
      }
  if (optind + 1 != argc)
    {
      print_usage (stderr);
      return EXIT_FAILURE;
    }
  const char *model_path = argv[optind];

  struct gip_lfm2_model model;
  struct gip_metal *metal;
  char err[512] = "";
  if (gip_lfm2_load (model_path, &model, err, sizeof err) != GIP_OK
      || gip_metal_open (&metal, err, sizeof err) != GIP_OK)
    {
      fprintf (stderr, "gip_bench: %s\n", err);
      return EXIT_FAILURE;
    }

  float *logits = malloc ((size_t)model.n_vocab * sizeof (float));
  int32_t *generated = malloc (n_generate * sizeof *generated);
  double *prefill_rates = calloc (reps, sizeof *prefill_rates);
  double *decode_rates = calloc (reps, sizeof *decode_rates);
  if (logits == NULL || generated == NULL || prefill_rates == NULL
      || decode_rates == NULL)
    {
      fprintf (stderr, "gip_bench: out of memory\n");
      return EXIT_FAILURE;
    }

  double total_decode_seconds = 0.0;
  double total_encode_seconds = 0.0;
  double total_gpu_seconds = 0.0;

  /* The first run warms the shader cache and the page cache and does not
     count.  */
  for (unsigned rep = 0; rep <= reps; rep++)
    {
      struct gip_lfm2_metal gpu;
      if (gip_lfm2_metal_init (&model, metal, n_prompt + n_generate, kv_half,
                               &gpu, err, sizeof err)
          != GIP_OK)
        {
          fprintf (stderr, "gip_bench: %s\n", err);
          return EXIT_FAILURE;
        }

      /* Prefill runs one token per step until batched prefill exists.
         Only the last prompt token needs logits.  */
      double start = now_seconds ();
      for (unsigned t = 0; t < n_prompt; t++)
        if (gip_lfm2_metal_step (&gpu, PROMPT_TOKEN,
                                 t + 1 == n_prompt ? logits : NULL, NULL)
            != GIP_OK)
          {
            fprintf (stderr, "gip_bench: prefill step failed\n");
            return EXIT_FAILURE;
          }
      double prefill_seconds = now_seconds () - start;

      /* Decode runs pipelined, the way an application generates text.  */
      start = now_seconds ();
      if (gip_lfm2_metal_generate (&gpu, n_generate, generated) != GIP_OK)
        {
          fprintf (stderr, "gip_bench: decode failed\n");
          return EXIT_FAILURE;
        }
      double decode_seconds = now_seconds () - start;
      double encode_seconds = gpu.last_encode_seconds;
      double gpu_seconds = gpu.last_gpu_seconds;
      gip_lfm2_metal_free (&gpu);

      if (rep > 0)
        {
          prefill_rates[rep - 1] = n_prompt / prefill_seconds;
          decode_rates[rep - 1] = n_generate / decode_seconds;
          total_decode_seconds += decode_seconds;
          total_encode_seconds += encode_seconds;
          total_gpu_seconds += gpu_seconds;
        }
    }

  printf ("%s, %u prompt and %u generated tokens, %u runs\n", model_path,
          n_prompt, n_generate, reps);
  print_rates ("prefill", prefill_rates, reps);
  print_rates ("decode", decode_rates, reps);

  /* Encoding overlaps the GPU in pipelined decode.  The GPU sits idle
     for whatever part of each token its own work does not cover.  */
  double per_token = 1e3 / ((double)n_generate * reps);
  printf ("decode per token: %.3f ms total, %.3f ms GPU busy, %.3f ms GPU "
          "idle, %.3f ms CPU encoding\n",
          total_decode_seconds * per_token, total_gpu_seconds * per_token,
          (total_decode_seconds - total_gpu_seconds) * per_token,
          total_encode_seconds * per_token);

  if (profile)
    profile_decode (&model, metal, n_prompt, n_generate, kv_half, logits);

  free (logits);
  free (generated);
  free (prefill_rates);
  free (decode_rates);
  gip_metal_close (metal);
  gip_lfm2_free (&model);
  return EXIT_SUCCESS;
}
