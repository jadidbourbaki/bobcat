/* lfm2_reference.c checks gip's LFM2 forward pass against activations
   that tools/ref_dump.py saved from transformers.  The test runs either
   the scalar pass or the Metal pass.  It compares the embedding, every
   layer's output, the final norm, and the logits for each prompt token,
   then checks that greedy decoding reproduces transformers'
   continuation token for token.  */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_lfm2.h"

/* The Metal backend exists only on macOS, so its header and calls are
   compiled only there.  */
#if GIP_HAVE_METAL
#include "model_lfm2_metal.h"
#endif

enum
{
  EXIT_SKIP = 77,
  PATH_SIZE = 4096,
  ERR_SIZE = 512
};

/* The largest error allowed, relative to the largest magnitude in the
   reference row.  Both sides compute in float32 from the same weights,
   so only summation order separates them.  */
static const double TOLERANCE = 1e-4;

/* The largest error allowed when the KV cache holds half precision,
   which rounds every cached key and value to 11 significant bits.  On
   LFM2.5-350M the largest error measured was 1.2e-3.  */
static const double HALF_KV_TOLERANCE = 5e-3;

/* The forward pass under test.  */
struct runner
{
  const struct gip_lfm2_model *model;
  bool use_metal;
  struct gip_lfm2_state state;
#if GIP_HAVE_METAL
  struct gip_metal *metal;
  struct gip_lfm2_metal gpu;
#endif
};

/* Run one step of the forward pass of R, as gip_lfm2_step does.  */
static enum gip_status
run_step (struct runner *r, int32_t token, float *logits,
          const struct gip_lfm2_trace *trace)
{
#if GIP_HAVE_METAL
  if (r->use_metal)
    return gip_lfm2_metal_step (&r->gpu, token, logits, trace);
#endif
  return gip_lfm2_step (r->model, &r->state, token, logits, trace);
}

/* Read the file NAME in directory DIR into a new buffer of COUNT
   elements of ELEMENT_SIZE bytes.  Store the element count in COUNT
   when COUNT is zero on entry.  Return null if the file is missing or
   has the wrong size.  */
static void *
read_file (const char *dir, const char *name, size_t element_size,
           size_t *count)
{
  char path[PATH_SIZE];
  snprintf (path, sizeof path, "%s/%s", dir, name);

  FILE *f = fopen (path, "rb");
  if (f == NULL)
    return NULL;
  fseek (f, 0, SEEK_END);
  long bytes = ftell (f);
  fseek (f, 0, SEEK_SET);
  if (bytes <= 0 || (size_t)bytes % element_size != 0
      || (*count != 0 && (size_t)bytes != *count * element_size))
    {
      fclose (f);
      return NULL;
    }

  void *data = malloc ((size_t)bytes);
  if (data == NULL || fread (data, 1, (size_t)bytes, f) != (size_t)bytes)
    {
      free (data);
      fclose (f);
      return NULL;
    }
  fclose (f);
  *count = (size_t)bytes / element_size;
  return data;
}

/* Return the largest absolute difference between the N floats at GOT
   and WANT, divided by the largest magnitude in WANT.  */
static double
relative_error (const float *got, const float *want, size_t n)
{
  double max_diff = 0.0;
  double max_want = 0.0;

  for (size_t i = 0; i < n; i++)
    {
      double diff = fabs ((double)got[i] - want[i]);
      if (diff > max_diff || isnan (diff))
        max_diff = isnan (diff) ? INFINITY : diff;
      if (fabs (want[i]) > max_want)
        max_want = fabs (want[i]);
    }
  return max_want > 0.0 ? max_diff / max_want : max_diff;
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

/* Load the model at argv[1] and the reference dump in the directory
   argv[2], then compare every activation.  */
int
main (int argc, char **argv)
{
  if (argc != 3 && argc != 4)
    {
      fprintf (stderr, "usage: lfm2_reference MODEL.gguf REF_DIR "
                       "[scalar|metal|metal_f32]\n");
      return EXIT_FAILURE;
    }
  const char *model_path = argv[1];
  const char *ref_dir = argv[2];
  const char *mode = argc == 4 ? argv[3] : "scalar";
  bool use_metal = strncmp (mode, "metal", 5) == 0;
  bool kv_half = strcmp (mode, "metal") == 0;
  double tolerance = kv_half ? HALF_KV_TOLERANCE : TOLERANCE;
  if (use_metal && !GIP_HAVE_METAL)
    {
      printf ("skip: this build has no Metal backend\n");
      return EXIT_SKIP;
    }

  size_t n_tokens = 0;
  size_t n_generated = 0;
  int32_t *tokens = read_file (ref_dir, "tokens.i32", 4, &n_tokens);
  int32_t *generated = read_file (ref_dir, "generated.i32", 4, &n_generated);
  FILE *model_file = fopen (model_path, "rb");
  if (tokens == NULL || generated == NULL || model_file == NULL)
    {
      printf ("skip: needs %s and a dump in %s from tools/ref_dump.py\n",
              model_path, ref_dir);
      return EXIT_SKIP;
    }
  fclose (model_file);

  struct gip_lfm2_model model;
  char err[ERR_SIZE] = "";
  enum gip_status status = gip_lfm2_load (model_path, &model, err, sizeof err);
  if (status != GIP_OK)
    {
      fprintf (stderr, "lfm2_reference: %s\n", err);
      return EXIT_FAILURE;
    }

  size_t n_embd = model.n_embd;
  size_t n_vocab = model.n_vocab;
  size_t n_layers = model.n_layers;
  size_t row_count = n_tokens * n_embd;
  size_t logit_count = n_tokens * n_vocab;
  float *want_embedding = read_file (ref_dir, "embedding.f32", 4, &row_count);
  row_count = n_tokens * n_embd;
  float *want_final = read_file (ref_dir, "final_norm.f32", 4, &row_count);
  float *want_logits = read_file (ref_dir, "logits.f32", 4, &logit_count);
  float **want_layers = calloc (n_layers, sizeof *want_layers);
  if (want_embedding == NULL || want_final == NULL || want_logits == NULL
      || want_layers == NULL)
    {
      fprintf (stderr,
               "lfm2_reference: reference dump in %s does not match "
               "the model\n",
               ref_dir);
      return EXIT_FAILURE;
    }
  for (size_t il = 0; il < n_layers; il++)
    {
      char name[32];
      snprintf (name, sizeof name, "layer_%02zu.f32", il);
      row_count = n_tokens * n_embd;
      want_layers[il] = read_file (ref_dir, name, 4, &row_count);
      if (want_layers[il] == NULL)
        {
          fprintf (stderr,
                   "lfm2_reference: %s/%s is missing or has the "
                   "wrong size\n",
                   ref_dir, name);
          return EXIT_FAILURE;
        }
    }

  struct runner runner = { .model = &model, .use_metal = use_metal };
  uint32_t n_ctx = (uint32_t)(n_tokens + n_generated);
  if (use_metal)
    {
#if GIP_HAVE_METAL
      if (gip_metal_open (&runner.metal, err, sizeof err) != GIP_OK)
        {
          printf ("skip: %s\n", err);
          return EXIT_SKIP;
        }
      status = gip_lfm2_metal_init (&model, runner.metal, n_ctx, kv_half,
                                    &runner.gpu, err, sizeof err);
      if (status != GIP_OK)
        {
          fprintf (stderr, "lfm2_reference: %s\n", err);
          return EXIT_FAILURE;
        }
#endif
    }
  else
    status = gip_lfm2_state_init (&model, n_ctx, &runner.state);
  float *embedding = malloc (n_embd * sizeof (float));
  float *layers = malloc (n_layers * n_embd * sizeof (float));
  float *final_norm = malloc (n_embd * sizeof (float));
  float *logits = malloc (n_vocab * sizeof (float));
  if (status != GIP_OK || embedding == NULL || layers == NULL
      || final_norm == NULL || logits == NULL)
    {
      fprintf (stderr, "lfm2_reference: out of memory\n");
      return EXIT_FAILURE;
    }
  struct gip_lfm2_trace trace = { embedding, layers, final_norm };

  /* WORST holds the largest error of each compared activation across the
     prompt: the embedding, each layer, the final norm, and the logits.  */
  size_t n_checks = n_layers + 3;
  double *worst = calloc (n_checks, sizeof *worst);
  if (worst == NULL)
    {
      fprintf (stderr, "lfm2_reference: out of memory\n");
      return EXIT_FAILURE;
    }

  for (size_t t = 0; t < n_tokens; t++)
    {
      status = run_step (&runner, tokens[t], logits, &trace);
      if (status != GIP_OK)
        {
          fprintf (stderr, "lfm2_reference: step failed: %s\n",
                   gip_status_string (status));
          return EXIT_FAILURE;
        }

      double errors[n_checks];
      errors[0]
          = relative_error (embedding, want_embedding + t * n_embd, n_embd);
      for (size_t il = 0; il < n_layers; il++)
        errors[1 + il] = relative_error (layers + il * n_embd,
                                         want_layers[il] + t * n_embd, n_embd);
      errors[n_layers + 1]
          = relative_error (final_norm, want_final + t * n_embd, n_embd);
      errors[n_layers + 2]
          = relative_error (logits, want_logits + t * n_vocab, n_vocab);
      for (size_t i = 0; i < n_checks; i++)
        if (errors[i] > worst[i])
          worst[i] = errors[i];
    }

  printf ("activation   max relative error over %zu tokens\n", n_tokens);
  int first_bad = -1;
  for (size_t i = 0; i < n_checks; i++)
    {
      char label[32];
      if (i == 0)
        snprintf (label, sizeof label, "embedding");
      else if (i <= n_layers)
        snprintf (label, sizeof label, "layer %2zu %s", i - 1,
                  model.layers[i - 1].is_attention ? "attn" : "conv");
      else if (i == n_layers + 1)
        snprintf (label, sizeof label, "final norm");
      else
        snprintf (label, sizeof label, "logits");
      printf ("%-14s %.3e\n", label, worst[i]);
      if (first_bad < 0 && !(worst[i] <= tolerance))
        first_bad = (int)i;
    }
  if (first_bad >= 0)
    {
      printf ("FAIL: activation %d exceeds the tolerance %.0e\n", first_bad,
              tolerance);
      return EXIT_FAILURE;
    }

  /* The prompt's last logits predict the first generated token.  Each
     prediction then feeds the next step.  The Metal pass picks every
     token on the GPU in one pipelined call.  */
  int32_t *gpu_tokens = NULL;
#if GIP_HAVE_METAL
  if (use_metal)
    {
      gpu_tokens = malloc (n_generated * sizeof *gpu_tokens);
      if (gpu_tokens == NULL)
        {
          fprintf (stderr, "lfm2_reference: out of memory\n");
          return EXIT_FAILURE;
        }
      status = gip_lfm2_metal_generate (&runner.gpu, (uint32_t)n_generated,
                                        gpu_tokens);
      if (status != GIP_OK)
        {
          fprintf (stderr, "lfm2_reference: generation failed: %s\n",
                   gip_status_string (status));
          return EXIT_FAILURE;
        }
    }
#endif
  size_t matched = 0;
  for (size_t g = 0; g < n_generated; g++)
    {
      int32_t predicted
          = gpu_tokens != NULL ? gpu_tokens[g] : argmax (logits, n_vocab);
      if (predicted != generated[g])
        {
          printf ("FAIL: generated token %zu is %d, transformers chose %d\n",
                  g, predicted, generated[g]);
          return EXIT_FAILURE;
        }
      matched++;
      if (gpu_tokens == NULL && g + 1 < n_generated)
        {
          status = run_step (&runner, predicted, logits, NULL);
          if (status != GIP_OK)
            {
              fprintf (stderr, "lfm2_reference: step failed: %s\n",
                       gip_status_string (status));
              return EXIT_FAILURE;
            }
        }
    }
  printf ("greedy decoding matched %zu of %zu tokens\n", matched, n_generated);

  for (size_t il = 0; il < n_layers; il++)
    free (want_layers[il]);
  free (want_layers);
  free (want_embedding);
  free (want_final);
  free (want_logits);
  free (worst);
  free (embedding);
  free (layers);
  free (final_norm);
  free (logits);
  free (tokens);
  free (generated);
  free (gpu_tokens);
#if GIP_HAVE_METAL
  if (use_metal)
    {
      gip_lfm2_metal_free (&runner.gpu);
      gip_metal_close (runner.metal);
    }
#endif
  if (!use_metal)
    gip_lfm2_state_free (&runner.state);
  gip_lfm2_free (&model);
  return EXIT_SUCCESS;
}
