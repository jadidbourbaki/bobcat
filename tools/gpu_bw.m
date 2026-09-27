/* gpu_bw.m measures the memory read bandwidth that the Metal GPU reaches.
   GPU decode speed is bounded by that bandwidth divided by the model
   size.  */

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

enum
{
  REPS = 10,
  DEFAULT_BUFFER_MB = 2048,
  THREADGROUP_SIZE = 256
};

/* The read kernel.  Each thread sums float4 values at a stride of the
   grid size, so neighboring threads read neighboring addresses.  The
   tool is a single file, so the kernel source lives here as a
   string.  */
static const char *kernel_source
    = "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void read_sum (device const float4 *src [[buffer (0)]],\n"
      "                      device float *out [[buffer (1)]],\n"
      "                      constant uint &n [[buffer (2)]],\n"
      "                      uint gid [[thread_position_in_grid]],\n"
      "                      uint grid [[threads_per_grid]])\n"
      "{\n"
      "  float4 acc = 0.0f;\n"
      "  for (uint i = gid; i < n; i += grid)\n"
      "    acc += src[i];\n"
      "  out[gid] = acc.x + acc.y + acc.z + acc.w;\n"
      "}\n";

/* The long options gpu_bw accepts.  */
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
           "Usage: gpu_bw [OPTION]...\n"
           "Measure the memory read bandwidth of the Metal GPU.\n"
           "\n"
           "  -s, --size=MB   read a buffer of MB megabytes (default %d)\n"
           "  -h, --help      display this help and exit\n"
           "  -V, --version   output version information and exit\n",
           DEFAULT_BUFFER_MB);
}

/* Return the best read bandwidth in GB/s over REPS dispatches of
   PIPELINE on QUEUE.  Each dispatch runs N_THREADS threads over the
   N_VECTORS float4 values in SRC and writes one sum per thread to
   OUT.  */
static double
measure (id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
         id<MTLBuffer> src, id<MTLBuffer> out, uint32_t n_vectors,
         NSUInteger n_threads)
{
  double best_seconds = 1e30;

  for (int rep = 0; rep < REPS; rep++)
    {
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:out offset:0 atIndex:1];
      [encoder setBytes:&n_vectors length:sizeof n_vectors atIndex:2];
      [encoder dispatchThreads:MTLSizeMake (n_threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (THREADGROUP_SIZE, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];

      /* GPU timestamps exclude host encoding and scheduling time.  */
      double elapsed = command_buffer.GPUEndTime - command_buffer.GPUStartTime;
      if (elapsed < best_seconds)
        best_seconds = elapsed;
    }

  return (double)n_vectors * 16.0 / best_seconds / 1e9;
}

/* Parse the options in ARGC and ARGV, then print the bandwidth at each
   grid size.  */
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
            if (errno != 0 || *end != '\0' || buffer_mb == 0
                || buffer_mb > 16384)
              {
                fprintf (stderr, "gpu_bw: invalid size: %s\n", optarg);
                return EXIT_FAILURE;
              }
            break;
          }
        case 'h':
          print_usage (stdout);
          return EXIT_SUCCESS;
        case 'V':
          printf ("gpu_bw (gip) %s\n", GIP_VERSION);
          return EXIT_SUCCESS;
        default:
          print_usage (stderr);
          return EXIT_FAILURE;
        }
    }

  @autoreleasepool
  {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice ();
    if (device == nil)
      {
        fprintf (stderr, "gpu_bw: no Metal device\n");
        return EXIT_FAILURE;
      }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil)
      {
        fprintf (stderr, "gpu_bw: cannot create command queue\n");
        return EXIT_FAILURE;
      }

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:kernel_source];
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&error];
    if (library == nil)
      {
        fprintf (stderr, "gpu_bw: cannot compile kernel: %s\n",
                 error.localizedDescription.UTF8String);
        return EXIT_FAILURE;
      }
    id<MTLFunction> function = [library newFunctionWithName:@"read_sum"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
      {
        fprintf (stderr, "gpu_bw: cannot create pipeline: %s\n",
                 error.localizedDescription.UTF8String);
        return EXIT_FAILURE;
      }

    size_t buffer_bytes = buffer_mb * 1024 * 1024;
    uint32_t n_vectors = (uint32_t)(buffer_bytes / 16);
    id<MTLBuffer> src =
        [device newBufferWithLength:buffer_bytes
                            options:MTLResourceStorageModeShared];
    if (src == nil)
      {
        fprintf (stderr, "gpu_bw: cannot allocate %zu MB\n", buffer_mb);
        return EXIT_FAILURE;
      }

    /* Touching every page first keeps page faults out of the timed
       dispatches.  */
    memset (src.contents, 1, buffer_bytes);

    static const NSUInteger grid_sizes[]
        = { 1 << 14, 1 << 16, 1 << 18, 1 << 20 };
    size_t n_grids = sizeof grid_sizes / sizeof grid_sizes[0];
    NSUInteger max_threads = grid_sizes[n_grids - 1];
    id<MTLBuffer> out =
        [device newBufferWithLength:max_threads * sizeof (float)
                            options:MTLResourceStorageModeShared];
    if (out == nil)
      {
        fprintf (stderr, "gpu_bw: cannot allocate the output buffer\n");
        return EXIT_FAILURE;
      }

    printf ("%s, buffer %zu MB, best of %d dispatches\n",
            device.name.UTF8String, buffer_mb, REPS);
    printf ("threads     GB/s\n");
    for (size_t i = 0; i < n_grids; i++)
      printf ("%7lu  %7.1f\n", (unsigned long)grid_sizes[i],
              measure (queue, pipeline, src, out, n_vectors, grid_sizes[i]));
  }
  return EXIT_SUCCESS;
}
