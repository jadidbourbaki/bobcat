// Measure warmed llama.cpp greedy streaming latency on a pre-tokenized prompt.
// Build and run with `just bench-llama-latency`. CSV rows go to stdout.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static int
count (const char *text, int minimum, int maximum)
{
  size_t end = 0;
  const std::string input (text);
  const int value = std::stoi (input, &end);
  if (end != input.size () || value < minimum || value > maximum)
    {
      throw std::runtime_error ("invalid benchmark count: " + input);
    }
  return value;
}

static double
milliseconds (Clock::time_point start)
{
  return std::chrono::duration<double, std::milli> (Clock::now () - start)
      .count ();
}

static void
decode (llama_context *context, llama_token *tokens, int n)
{
  if (llama_decode (context, llama_batch_get_one (tokens, n)) != 0)
    {
      throw std::runtime_error ("llama_decode failed");
    }
}

static void
run (int argc, char **argv)
{
  if (argc != 5)
    {
      throw std::runtime_error (
          "usage: llama-latency MODEL PROMPT GENERATE REPS");
    }
  const int prompt_count = count (argv[2], 1, 1000000);
  const int generated_count = count (argv[3], 2, 1000000);
  const int reps = count (argv[4], 1, 1000);
  llama_log_set (
      [] (ggml_log_level level, const char *text, void *)
        {
          if (level >= GGML_LOG_LEVEL_WARN)
            {
              std::fputs (text, stderr);
            }
        },
      nullptr);
  llama_backend_init ();
  auto model_params = llama_model_default_params ();
  model_params.n_gpu_layers = 99;
  std::unique_ptr<llama_model, decltype (&llama_model_free)> model (
      llama_model_load_from_file (argv[1], model_params), llama_model_free);
  if (!model)
    {
      throw std::runtime_error ("model load failed");
    }
  auto context_params = llama_context_default_params ();
  context_params.n_ctx = prompt_count + generated_count;
  context_params.n_batch = 512;
  context_params.n_ubatch = 512;
  context_params.n_threads = 10;
  context_params.n_threads_batch = 10;
  context_params.type_k = GGML_TYPE_F16;
  context_params.type_v = GGML_TYPE_F16;
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  std::unique_ptr<llama_context, decltype (&llama_free)> context (
      llama_init_from_model (model.get (), context_params), llama_free);
  if (!context)
    {
      throw std::runtime_error ("context load failed");
    }
  const int vocab_count
      = llama_vocab_n_tokens (llama_model_get_vocab (model.get ()));
  if (vocab_count <= 1000)
    {
      throw std::runtime_error ("prompt token 1000 is outside the vocabulary");
    }
  std::vector<llama_token> prompt (prompt_count, 1000);
  std::puts ("engine,run,prompt_tokens,generated_tokens,stream_chunk,ttft_ms,"
             "tpot_ms,end_to_end_ms,token_hash");
  for (int rep = 0; rep <= reps; ++rep)
    {
      llama_memory_clear (llama_get_memory (context.get ()), true);
      const auto start = Clock::now ();
      for (int offset = 0; offset < prompt_count; offset += 512)
        {
          decode (context.get (), prompt.data () + offset,
                  std::min (512, prompt_count - offset));
        }
      double first_ms = 0.0;
      uint64_t hash = 0;
      for (int i = 0; i < generated_count; ++i)
        {
          const float *logits = llama_get_logits_ith (context.get (), -1);
          if (!logits)
            {
              throw std::runtime_error ("missing logits");
            }
          auto token = static_cast<llama_token> (
              std::max_element (logits, logits + vocab_count) - logits);
          hash = hash * 1000003 + static_cast<uint64_t> (token);
          if (i == 0)
            {
              first_ms = milliseconds (start);
            }
          if (i + 1 < generated_count)
            {
              decode (context.get (), &token, 1);
            }
        }
      const double total_ms = milliseconds (start);
      if (rep > 0)
        {
          std::printf ("llama.cpp,%d,%d,%d,1,%.6f,%.6f,%.6f,%llu\n", rep,
                       prompt_count, generated_count, first_ms,
                       (total_ms - first_ms) / (generated_count - 1), total_ms,
                       static_cast<unsigned long long> (hash));
        }
    }
}

int
main (int argc, char **argv)
{
  try
    {
      run (argc, argv);
      llama_backend_free ();
      return EXIT_SUCCESS;
    }
  catch (const std::exception &error)
    {
      std::fprintf (stderr, "llama-latency: %s\n", error.what ());
      return EXIT_FAILURE;
    }
}
