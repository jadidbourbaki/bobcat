/* bw.c measures the memory read bandwidth that CPU threads reach.
   Decode speed is bounded by that bandwidth divided by the model size.  */

#include <arm_neon.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

enum
{
  MAX_THREADS = 16,
  REPS = 10,
  DEFAULT_BUFFER_MB = 2048
};

/* A range of floats that one thread reads, and the sum the thread
   computes over the range.  */
struct slice
{
  const float *data;
  size_t count;
  float sum;
};

/* The long options bw accepts.  */
static const struct option long_options[] = {
  { "size", required_argument, NULL, 's' },
  { "help", no_argument, NULL, 'h' },
  { "version", no_argument, NULL, 'V' },
  { NULL, 0, NULL, 0 },
};

/* Print the usage message to STREAM.  */
static void
print_usage (FILE *stream)
{
  fprintf (stream,
           "Usage: bw [OPTION]...\n"
           "Measure the memory read bandwidth of CPU threads.\n"
           "\n"
           "  -s, --size=MB   read a buffer of MB megabytes (default %d)\n"
           "  -h, --help      display this help and exit\n"
           "  -V, --version   output version information and exit\n",
           DEFAULT_BUFFER_MB);
}

/* Return the monotonic clock time in seconds.  */
static double
now_seconds (void)
{
  struct timespec ts;

  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Sum the floats of the struct slice at ARG into its sum field.  */
static void *
read_slice (void *arg)
{
  struct slice *slice = arg;

#if defined(__APPLE__)
  /* macOS has no thread pinning.  The QoS class steers the thread toward
     the performance cores.  */
  pthread_set_qos_class_self_np (QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  /* Four independent accumulators keep the loop bound by load
     throughput.  A single accumulator would wait on each add.  */
  float32x4_t acc0 = vdupq_n_f32 (0.0f);
  float32x4_t acc1 = vdupq_n_f32 (0.0f);
  float32x4_t acc2 = vdupq_n_f32 (0.0f);
  float32x4_t acc3 = vdupq_n_f32 (0.0f);

  const float *end = slice->data + slice->count;
  for (const float *p = slice->data; p < end; p += 16)
    {
      acc0 = vaddq_f32 (acc0, vld1q_f32 (p + 0));
      acc1 = vaddq_f32 (acc1, vld1q_f32 (p + 4));
      acc2 = vaddq_f32 (acc2, vld1q_f32 (p + 8));
      acc3 = vaddq_f32 (acc3, vld1q_f32 (p + 12));
    }

  float32x4_t total
      = vaddq_f32 (vaddq_f32 (acc0, acc1), vaddq_f32 (acc2, acc3));
  slice->sum = vaddvq_f32 (total);
  return NULL;
}

/* Return the best read bandwidth in GB/s over REPS passes in which
   N_THREADS threads split the COUNT floats at DATA.  */
static double
measure (const float *data, size_t count, int n_threads)
{
  pthread_t threads[MAX_THREADS];
  struct slice slices[MAX_THREADS];
  size_t per_thread = (count / (size_t)n_threads) & ~(size_t)15;
  double best_seconds = 1e30;
  float checksum = 0.0f;

  for (int rep = 0; rep < REPS; rep++)
    {
      double start = now_seconds ();
      for (int t = 0; t < n_threads; t++)
        {
          slices[t].data = data + (size_t)t * per_thread;
          slices[t].count = per_thread;
          slices[t].sum = 0.0f;
          int err = pthread_create (&threads[t], NULL, read_slice, &slices[t]);
          if (err != 0)
            {
              fprintf (stderr, "bw: pthread_create: %s\n", strerror (err));
              exit (EXIT_FAILURE);
            }
        }
      for (int t = 0; t < n_threads; t++)
        {
          pthread_join (threads[t], NULL);
          checksum += slices[t].sum;
        }
      double elapsed = now_seconds () - start;
      if (elapsed < best_seconds)
        best_seconds = elapsed;
    }

  /* The checksum feeds a branch so the compiler keeps every read.  */
  if (checksum == 12345.0f)
    printf ("checksum %f\n", checksum);

  double bytes_read = (double)per_thread * n_threads * sizeof (float);
  return bytes_read / best_seconds / 1e9;
}

/* Parse the options in ARGC and ARGV, then print the bandwidth at each
   thread count.  */
int
main (int argc, char **argv)
{
  size_t buffer_mb = DEFAULT_BUFFER_MB;
  int opt;

  while ((opt = getopt_long (argc, argv, "s:hV", long_options, NULL)) != -1)
    {
      switch (opt)
        {
        case 's':
          {
            char *end;
            errno = 0;
            buffer_mb = strtoull (optarg, &end, 10);
            if (errno != 0 || *end != '\0' || buffer_mb == 0)
              {
                fprintf (stderr, "bw: invalid size: %s\n", optarg);
                return EXIT_FAILURE;
              }
            break;
          }
        case 'h':
          print_usage (stdout);
          return EXIT_SUCCESS;
        case 'V':
          printf ("bw (gip) %s\n", GIP_VERSION);
          return EXIT_SUCCESS;
        default:
          print_usage (stderr);
          return EXIT_FAILURE;
        }
    }

  size_t count = buffer_mb * 1024 * 1024 / sizeof (float);
  float *data = aligned_alloc (64, count * sizeof (float));
  if (data == NULL)
    {
      fprintf (stderr, "bw: cannot allocate %zu MB: %s\n", buffer_mb,
               strerror (errno));
      return EXIT_FAILURE;
    }

  /* Touching every page first keeps page faults out of the timed
     passes.  */
  memset (data, 1, count * sizeof (float));

  static const int thread_counts[] = { 1, 2, 4, 6, 8, 10, 12, 14 };
  size_t n_counts = sizeof thread_counts / sizeof thread_counts[0];

  printf ("buffer %zu MB, best of %d passes\n", buffer_mb, REPS);
  printf ("threads  GB/s\n");
  for (size_t i = 0; i < n_counts; i++)
    printf ("%7d  %6.1f\n", thread_counts[i],
            measure (data, count, thread_counts[i]));

  free (data);
  return EXIT_SUCCESS;
}
