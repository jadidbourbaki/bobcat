/* backend.m is gip's Metal backend.  It compiles the embedded kernel
   source when it opens and records kernel launches into one command
   buffer at a time.  */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unistd.h>

#include "error.h"
#include "metal/backend.h"

/* The kernel source, embedded at build time from kernels.metal.  */
extern const char gip_metal_source[];

enum
{
  SIMD_WIDTH = 32,
  /* A sweep of 1, 2, and 4 rows per threadgroup against 2, 4, and 8
     simdgroups on an M4 Pro put two rows first on every model.  Two
     simdgroups won on rows of 1024 columns, lifting LFM2.5-350M from
     445 to 495 tokens per second.  Four simdgroups won by about 1% on
     the wider rows of LFM2.5-1.2B and LFM2.5-2.6B.  */
  MATVEC_Q8_0_ROWS_PER_THREADGROUP = 2,
  MATVEC_Q8_0_NARROW_SIMDGROUPS = 2,
  MATVEC_Q8_0_WIDE_SIMDGROUPS = 4,
  MATVEC_Q8_0_WIDE_COLS = 2048,
  /* Threads per threadgroup for the reduction and elementwise
     kernels.  */
  REDUCE_THREADS = 256,
  ELEMENTWISE_THREADS = 256,
  /* Must match ATTENTION_CHUNK in kernels.metal.  */
  ATTENTION_CHUNK = 64,
  /* One threadgroup of the largest size scans a vocabulary in a few
     hundred loads per thread.  */
  ARGMAX_THREADS = 1024,
  /* Distinct kernel and shape pairs a profile keeps.  A model has far
     fewer.  */
  MAX_PROFILE_ENTRIES = 64
};

/* The profile of a backend: one entry per kernel and matrix shape.  */
struct profile_table
{
  size_t count;
  struct gip_metal_profile_entry entries[MAX_PROFILE_ENTRIES];
};

/* The state behind a struct gip_metal.  */
@interface GipMetal : NSObject
{
@public
  /* The matrix-vector pipelines, indexed by whether they fuse the norm
     and whether they accumulate.  */
  id<MTLComputePipelineState> matvec_q8_0[2][2];
  /* The norm and rotation pipelines, indexed by whether they write half
     precision.  */
  id<MTLComputePipelineState> norm_rope[2];
  /* The first attention pass, indexed by whether the caches hold half
     precision.  */
  id<MTLComputePipelineState> attention_chunk[2];
}
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLComputePipelineState> matvec_q8_0_swiglu;
@property (nonatomic, strong) id<MTLComputePipelineState> rms_norm;
@property (nonatomic, strong) id<MTLComputePipelineState> convert_half;
@property (nonatomic, strong) id<MTLComputePipelineState> attention_combine;
@property (nonatomic, strong) id<MTLComputePipelineState> short_conv;
/* Objective-C treats a property named copy... as returning an owned
   object, so the copy pipeline takes a different name.  */
@property (nonatomic, strong) id<MTLComputePipelineState> float_copy;
@property (nonatomic, strong) id<MTLComputePipelineState> embed_q8_0;
@property (nonatomic, strong) id<MTLComputePipelineState> argmax;
@property (nonatomic, strong) id<MTLCommandBuffer> command_buffer;
@property (nonatomic, strong) id<MTLComputeCommandEncoder> encoder;
/* Submitted command buffers not yet waited on, oldest first.  COMMITTED
   and COMPLETED count the command buffers submitted and waited on.  */
@property (nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *in_flight;
@property (nonatomic) uint64_t committed;
@property (nonatomic) uint64_t completed;
@property (nonatomic) BOOL profiling;
@property (nonatomic, strong) id<MTLCommandBuffer> op_command_buffer;
@property (nonatomic, strong) id<MTLComputeCommandEncoder> op_encoder;
@property (nonatomic) struct profile_table *profile;
@end

@implementation GipMetal
- (void)dealloc
{
  free (_profile);
}
@end

/* Return the GipMetal object behind METAL.  */
static GipMetal *
backend (struct gip_metal *metal)
{
  return (__bridge GipMetal *)(void *)metal;
}

/* Return the Metal buffer behind BUFFER.  */
static id<MTLBuffer>
metal_buffer (struct gip_metal_buffer *buffer)
{
  return (__bridge id<MTLBuffer>)(void *)buffer;
}

/* Return a pipeline for the matrix-vector kernel NAME in LIBRARY on
   DEVICE, or nil.  The function constants take the values ROWS for
   rows_per_threadgroup, FUSE_NORM for fuse_norm, and ACCUMULATE for
   accumulate.  Store any error in ERROR.  */
static id<MTLComputePipelineState>
make_pipeline (id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
               uint32_t rows, bool fuse_norm, bool accumulate, NSError **error)
{
  MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
  [constants setConstantValue:&rows type:MTLDataTypeUInt atIndex:0];
  [constants setConstantValue:&fuse_norm type:MTLDataTypeBool atIndex:1];
  [constants setConstantValue:&accumulate type:MTLDataTypeBool atIndex:2];

  id<MTLFunction> function = [library newFunctionWithName:name
                                           constantValues:constants
                                                    error:error];
  if (function == nil)
    return nil;
  return [device newComputePipelineStateWithFunction:function error:error];
}

/* Return a pipeline for the kernel NAME, which has no function
   constants, in LIBRARY on DEVICE, or nil.  Store any error in
   ERROR.  */
static id<MTLComputePipelineState>
plain_pipeline (id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
                NSError **error)
{
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil)
    return nil;
  return [device newComputePipelineStateWithFunction:function error:error];
}

enum gip_status
gip_metal_open (struct gip_metal **out, char *err, size_t err_size)
{
  @autoreleasepool
  {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice ();
    if (device == nil)
      {
        gip_format_error (err, err_size, "no Metal device");
        return GIP_ERR_UNSUPPORTED;
      }

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:gip_metal_source];
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available (macOS 15.0, *))
      options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
    if (library == nil)
      {
        gip_format_error (err, err_size, "cannot compile Metal kernels: %s",
                          error.localizedDescription.UTF8String);
        return GIP_ERR_UNSUPPORTED;
      }

    GipMetal *metal = [GipMetal new];
    metal.device = device;
    metal.queue = [device newCommandQueue];
    uint32_t rows = MATVEC_Q8_0_ROWS_PER_THREADGROUP;
    for (int norm = 0; norm < 2; norm++)
      for (int acc = 0; acc < 2; acc++)
        metal->matvec_q8_0[norm][acc] = make_pipeline (
            device, library, @"matvec_q8_0", rows, norm, acc, &error);
    metal.matvec_q8_0_swiglu = make_pipeline (
        device, library, @"matvec_q8_0_swiglu", rows, true, false, &error);
    metal.rms_norm = plain_pipeline (device, library, @"rms_norm", &error);
    metal->norm_rope[0]
        = plain_pipeline (device, library, @"norm_rope_f32", &error);
    metal->norm_rope[1]
        = plain_pipeline (device, library, @"norm_rope_f16", &error);
    metal.convert_half
        = plain_pipeline (device, library, @"convert_half", &error);
    metal->attention_chunk[0]
        = plain_pipeline (device, library, @"attention_chunk_f32", &error);
    metal->attention_chunk[1]
        = plain_pipeline (device, library, @"attention_chunk_f16", &error);
    metal.attention_combine
        = plain_pipeline (device, library, @"attention_combine", &error);
    metal.short_conv = plain_pipeline (device, library, @"short_conv", &error);
    metal.float_copy
        = plain_pipeline (device, library, @"copy_floats", &error);
    metal.embed_q8_0 = plain_pipeline (device, library, @"embed_q8_0", &error);
    metal.argmax = plain_pipeline (device, library, @"argmax", &error);
    metal.in_flight = [NSMutableArray new];
    bool have_all = true;
    for (int norm = 0; norm < 2; norm++)
      for (int acc = 0; acc < 2; acc++)
        have_all &= metal->matvec_q8_0[norm][acc] != nil;
    for (int half_precision = 0; half_precision < 2; half_precision++)
      have_all &= metal->norm_rope[half_precision] != nil
                  && metal->attention_chunk[half_precision] != nil;
    if (metal.queue == nil || !have_all || metal.matvec_q8_0_swiglu == nil
        || metal.rms_norm == nil || metal.convert_half == nil
        || metal.attention_combine == nil || metal.short_conv == nil
        || metal.float_copy == nil || metal.embed_q8_0 == nil
        || metal.argmax == nil)
      {
        gip_format_error (err, err_size, "cannot create Metal pipelines: %s",
                          error != nil ? error.localizedDescription.UTF8String
                                       : "no command queue");
        return GIP_ERR_UNSUPPORTED;
      }

    metal.profile = calloc (1, sizeof (struct profile_table));
    if (metal.profile == NULL)
      {
        gip_format_error (err, err_size, "out of memory");
        return GIP_ERR_NOMEM;
      }

    *out = (__bridge_retained void *)metal;
    return GIP_OK;
  }
}

void
gip_metal_close (struct gip_metal *metal)
{
  if (metal != NULL)
    CFRelease ((CFTypeRef)metal);
}

struct gip_metal_buffer *
gip_metal_buffer_new (struct gip_metal *metal, size_t size)
{
  id<MTLBuffer> buffer = [backend (metal).device
      newBufferWithLength:(size ? size : 1)
                  options:MTLResourceStorageModeShared];
  if (buffer == nil)
    return NULL;
  memset (buffer.contents, 0, buffer.length);
  return (__bridge_retained void *)buffer;
}

struct gip_metal_buffer *
gip_metal_buffer_wrap (struct gip_metal *metal, void *data, size_t size)
{
  size_t page = (size_t)getpagesize ();
  size_t length;

  if ((uintptr_t)data % page != 0
      || __builtin_add_overflow (size, page - 1, &length))
    return NULL;
  /* Metal wraps whole pages.  The mapping of the file's last page covers
     the rounded length.  */
  length -= length % page;

  id<MTLBuffer> buffer = [backend (metal).device
      newBufferWithBytesNoCopy:data
                        length:length
                       options:MTLResourceStorageModeShared
                   deallocator:nil];
  if (buffer == nil)
    return NULL;
  return (__bridge_retained void *)buffer;
}

void *
gip_metal_buffer_contents (struct gip_metal_buffer *buffer)
{
  return metal_buffer (buffer).contents;
}

struct gip_metal_view
gip_metal_at (struct gip_metal_buffer *buffer, size_t offset)
{
  struct gip_metal_view view = { buffer, offset };
  return view;
}

void
gip_metal_buffer_free (struct gip_metal_buffer *buffer)
{
  if (buffer != NULL)
    CFRelease ((CFTypeRef)buffer);
}

enum gip_status
gip_metal_begin (struct gip_metal *metal)
{
  GipMetal *m = backend (metal);

  if (m.command_buffer != nil)
    return GIP_ERR_ARGUMENT;
  /* Command buffers and encoders arrive autoreleased.  The pool frees
     the temporary references on every step of a long decode.  */
  @autoreleasepool
  {
    m.command_buffer = [m.queue commandBuffer];
    m.encoder = [m.command_buffer
        computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  }
  if (m.command_buffer == nil || m.encoder == nil)
    {
      m.command_buffer = nil;
      m.encoder = nil;
      return GIP_ERR_NOMEM;
    }
  return GIP_OK;
}

void
gip_metal_barrier (struct gip_metal *metal)
{
  GipMetal *m = backend (metal);

  /* While profiling, each launch waits for its own command buffer, which
     orders the launches already.  */
  if (!m.profiling)
    [m.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* Bind VIEW to buffer slot INDEX of ENCODER.  */
static void
bind (id<MTLComputeCommandEncoder> encoder, struct gip_metal_view view,
      NSUInteger index)
{
  [encoder setBuffer:metal_buffer (view.buffer)
              offset:view.offset
             atIndex:index];
}

/* Bind the SIZE bytes at VALUE to slot INDEX of ENCODER.  */
static void
bind_bytes (id<MTLComputeCommandEncoder> encoder, const void *value,
            size_t size, NSUInteger index)
{
  [encoder setBytes:value length:size atIndex:index];
}

/* Dispatch N threads of the current pipeline of ENCODER in threadgroups
   of ELEMENTWISE_THREADS.  */
static void
dispatch_elements (id<MTLComputeCommandEncoder> encoder, uint32_t n)
{
  [encoder dispatchThreads:MTLSizeMake (n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake (ELEMENTWISE_THREADS, 1, 1)];
}

/* Return the encoder the next launch on M records into.  While
   profiling, each launch gets a command buffer of its own.  */
static id<MTLComputeCommandEncoder>
op_encoder (GipMetal *m)
{
  if (!m.profiling)
    return m.encoder;
  @autoreleasepool
  {
    m.op_command_buffer = [m.queue commandBuffer];
    m.op_encoder = [m.op_command_buffer computeCommandEncoder];
  }
  return m.op_encoder;
}

/* Finish a launch of the kernel NAME on M.  While profiling, run the
   launch's command buffer and add its GPU time to the entry for NAME,
   N_ROWS, and N_COLS, along with the BYTES of weights it read.  */
static void
op_done (GipMetal *m, const char *name, uint32_t n_rows, uint32_t n_cols,
         uint64_t bytes)
{
  if (!m.profiling)
    return;

  id<MTLCommandBuffer> command_buffer = m.op_command_buffer;
  [m.op_encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  m.op_encoder = nil;
  m.op_command_buffer = nil;

  struct profile_table *table = m.profile;
  struct gip_metal_profile_entry *entry = NULL;
  for (size_t i = 0; i < table->count && entry == NULL; i++)
    if (strcmp (table->entries[i].name, name) == 0
        && table->entries[i].n_rows == n_rows
        && table->entries[i].n_cols == n_cols)
      entry = &table->entries[i];
  if (entry == NULL)
    {
      if (table->count == MAX_PROFILE_ENTRIES)
        return;
      entry = &table->entries[table->count++];
      entry->name = name;
      entry->n_rows = n_rows;
      entry->n_cols = n_cols;
    }
  entry->calls++;
  entry->seconds += command_buffer.GPUEndTime - command_buffer.GPUStartTime;
  entry->bytes += bytes;
}

void
gip_metal_set_profiling (struct gip_metal *metal, int enabled)
{
  GipMetal *m = backend (metal);

  m.profiling = enabled != 0;
  if (enabled)
    memset (m.profile, 0, sizeof *m.profile);
}

size_t
gip_metal_profile (struct gip_metal *metal,
                   const struct gip_metal_profile_entry **entries)
{
  GipMetal *m = backend (metal);

  *entries = m.profile->entries;
  return m.profile->count;
}

/* Dispatch the current matrix-vector pipeline of ENCODER over N_ROWS
   rows of N_COLS columns.  Wider rows get more simdgroups to split
   them.  */
static void
dispatch_matvec (id<MTLComputeCommandEncoder> encoder, uint32_t n_rows,
                 uint32_t n_cols)
{
  uint32_t rows_per_threadgroup = MATVEC_Q8_0_ROWS_PER_THREADGROUP;
  NSUInteger n_threadgroups
      = (n_rows + rows_per_threadgroup - 1) / rows_per_threadgroup;
  NSUInteger simdgroups = n_cols >= MATVEC_Q8_0_WIDE_COLS
                              ? MATVEC_Q8_0_WIDE_SIMDGROUPS
                              : MATVEC_Q8_0_NARROW_SIMDGROUPS;

  [encoder dispatchThreadgroups:MTLSizeMake (n_threadgroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (SIMD_WIDTH * simdgroups, 1, 1)];
}

/* Return the bytes of a Q8_0 matrix of N_ROWS rows of N_COLS
   elements.  */
static uint64_t
q8_0_bytes (uint32_t n_rows, uint32_t n_cols)
{
  return (uint64_t)n_rows * (n_cols / 32) * 34;
}

void
gip_metal_matvec_q8_0 (struct gip_metal *metal, struct gip_metal_view weights,
                       uint32_t n_rows, uint32_t n_cols,
                       struct gip_metal_view x, struct gip_metal_view y,
                       const struct gip_metal_matvec_options *options)
{
  GipMetal *m = backend (metal);
  bool norm = options != NULL && options->norm_weight.buffer != NULL;
  bool acc = options != NULL && options->accumulate;
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m->matvec_q8_0[norm][acc]];
  bind (encoder, weights, 0);
  bind (encoder, x, 1);
  bind (encoder, y, 2);
  bind_bytes (encoder, &n_rows, sizeof n_rows, 3);
  bind_bytes (encoder, &n_cols, sizeof n_cols, 4);
  if (norm)
    {
      bind (encoder, options->norm_weight, 5);
      bind_bytes (encoder, &options->eps, sizeof options->eps, 6);
    }
  dispatch_matvec (encoder, n_rows, n_cols);
  op_done (m, "matvec_q8_0", n_rows, n_cols, q8_0_bytes (n_rows, n_cols));
}

void
gip_metal_matvec_q8_0_swiglu (struct gip_metal *metal,
                              struct gip_metal_view gate,
                              struct gip_metal_view up, uint32_t n_rows,
                              uint32_t n_cols, struct gip_metal_view x,
                              struct gip_metal_view norm_weight, float eps,
                              struct gip_metal_view y)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.matvec_q8_0_swiglu];
  bind (encoder, gate, 0);
  bind (encoder, up, 1);
  bind (encoder, x, 2);
  bind (encoder, y, 3);
  bind_bytes (encoder, &n_rows, sizeof n_rows, 4);
  bind_bytes (encoder, &n_cols, sizeof n_cols, 5);
  bind (encoder, norm_weight, 6);
  bind_bytes (encoder, &eps, sizeof eps, 7);
  dispatch_matvec (encoder, n_rows, n_cols);
  op_done (m, "matvec_q8_0_swiglu", n_rows, n_cols,
           2 * q8_0_bytes (n_rows, n_cols));
}

void
gip_metal_rms_norm (struct gip_metal *metal, struct gip_metal_view x,
                    struct gip_metal_view weight, struct gip_metal_view out,
                    uint32_t n, float eps)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.rms_norm];
  bind (encoder, x, 0);
  bind (encoder, weight, 1);
  bind (encoder, out, 2);
  bind_bytes (encoder, &n, sizeof n, 3);
  bind_bytes (encoder, &eps, sizeof eps, 4);
  [encoder dispatchThreadgroups:MTLSizeMake (1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (REDUCE_THREADS, 1, 1)];
  op_done (m, "rms_norm", 0, 0, 0);
}

void
gip_metal_norm_rope (struct gip_metal *metal, struct gip_metal_view src,
                     struct gip_metal_view dst, int dst_half,
                     struct gip_metal_view weight, uint32_t n_heads,
                     uint32_t head_dim, uint32_t pos, float theta, float eps)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m->norm_rope[dst_half != 0]];
  bind (encoder, src, 0);
  bind (encoder, dst, 1);
  bind (encoder, weight, 2);
  bind_bytes (encoder, &head_dim, sizeof head_dim, 3);
  bind_bytes (encoder, &pos, sizeof pos, 4);
  bind_bytes (encoder, &theta, sizeof theta, 5);
  bind_bytes (encoder, &eps, sizeof eps, 6);
  [encoder dispatchThreadgroups:MTLSizeMake (n_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (head_dim, 1, 1)];
  op_done (m, "norm_rope", 0, 0, 0);
}

void
gip_metal_convert_half (struct gip_metal *metal, struct gip_metal_view src,
                        struct gip_metal_view dst, uint32_t n)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.convert_half];
  bind (encoder, src, 0);
  bind (encoder, dst, 1);
  bind_bytes (encoder, &n, sizeof n, 2);
  dispatch_elements (encoder, n);
  op_done (m, "convert_half", 0, 0, 0);
}

/* Return the number of attention chunks covering N positions.  */
static uint32_t
attention_chunks (uint32_t n)
{
  return (n + ATTENTION_CHUNK - 1) / ATTENTION_CHUNK;
}

size_t
gip_metal_attention_scratch (uint32_t n_heads, uint32_t head_dim,
                             uint32_t n_ctx)
{
  /* Each chunk of each head keeps a weighted sum of values, a largest
     score, and a sum of exponentials.  */
  return (size_t)n_heads * attention_chunks (n_ctx) * (head_dim + 2);
}

void
gip_metal_attention (struct gip_metal *metal, struct gip_metal_view q,
                     struct gip_metal_view k_cache,
                     struct gip_metal_view v_cache, int kv_half,
                     struct gip_metal_view scratch, struct gip_metal_view out,
                     uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                     uint32_t n_keys, uint32_t n_ctx)
{
  GipMetal *m = backend (metal);
  float scale = 1.0f / sqrtf ((float)head_dim);
  uint32_t n_chunks = attention_chunks (n_keys);
  uint32_t max_chunks = attention_chunks (n_ctx);

  id<MTLComputeCommandEncoder> encoder = op_encoder (m);
  [encoder setComputePipelineState:m->attention_chunk[kv_half != 0]];
  bind (encoder, q, 0);
  bind (encoder, k_cache, 1);
  bind (encoder, v_cache, 2);
  bind (encoder, scratch, 3);
  bind_bytes (encoder, &n_heads, sizeof n_heads, 4);
  bind_bytes (encoder, &n_kv_heads, sizeof n_kv_heads, 5);
  bind_bytes (encoder, &head_dim, sizeof head_dim, 6);
  bind_bytes (encoder, &n_keys, sizeof n_keys, 7);
  bind_bytes (encoder, &max_chunks, sizeof max_chunks, 8);
  bind_bytes (encoder, &scale, sizeof scale, 9);
  [encoder dispatchThreadgroups:MTLSizeMake (n_kv_heads, n_chunks, 1)
          threadsPerThreadgroup:MTLSizeMake (ATTENTION_CHUNK, 1, 1)];
  op_done (m, "attention_chunk", 0, 0, 0);
  gip_metal_barrier (metal);

  encoder = op_encoder (m);
  [encoder setComputePipelineState:m.attention_combine];
  bind (encoder, scratch, 0);
  bind (encoder, out, 1);
  bind_bytes (encoder, &n_heads, sizeof n_heads, 2);
  bind_bytes (encoder, &head_dim, sizeof head_dim, 3);
  bind_bytes (encoder, &n_chunks, sizeof n_chunks, 4);
  bind_bytes (encoder, &max_chunks, sizeof max_chunks, 5);
  [encoder dispatchThreadgroups:MTLSizeMake (n_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (head_dim, 1, 1)];
  op_done (m, "attention_combine", 0, 0, 0);
}

void
gip_metal_short_conv (struct gip_metal *metal, struct gip_metal_view bcx,
                      struct gip_metal_view taps,
                      struct gip_metal_view history, struct gip_metal_view out,
                      uint32_t n_embd, uint32_t kernel_size)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.short_conv];
  bind (encoder, bcx, 0);
  bind (encoder, taps, 1);
  bind (encoder, history, 2);
  bind (encoder, out, 3);
  bind_bytes (encoder, &n_embd, sizeof n_embd, 4);
  bind_bytes (encoder, &kernel_size, sizeof kernel_size, 5);
  dispatch_elements (encoder, n_embd);
  op_done (m, "short_conv", 0, 0, 0);
}

void
gip_metal_copy (struct gip_metal *metal, struct gip_metal_view src,
                struct gip_metal_view dst, uint32_t n)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.float_copy];
  bind (encoder, src, 0);
  bind (encoder, dst, 1);
  bind_bytes (encoder, &n, sizeof n, 2);
  dispatch_elements (encoder, n);
  op_done (m, "copy", 0, 0, 0);
}

enum gip_status
gip_metal_end (struct gip_metal *metal, double *gpu_seconds)
{
  uint64_t ticket;
  enum gip_status status = gip_metal_commit (metal, &ticket);

  if (gpu_seconds != NULL)
    *gpu_seconds = 0.0;
  if (status != GIP_OK)
    return status;
  return gip_metal_wait (metal, ticket, gpu_seconds);
}

enum gip_status
gip_metal_commit (struct gip_metal *metal, uint64_t *ticket)
{
  GipMetal *m = backend (metal);
  id<MTLCommandBuffer> command_buffer = m.command_buffer;

  if (command_buffer == nil)
    return GIP_ERR_ARGUMENT;
  [m.encoder endEncoding];
  [command_buffer commit];
  [m.in_flight addObject:command_buffer];
  m.encoder = nil;
  m.command_buffer = nil;
  m.committed++;
  *ticket = m.committed;
  return GIP_OK;
}

enum gip_status
gip_metal_wait (struct gip_metal *metal, uint64_t ticket, double *gpu_seconds)
{
  GipMetal *m = backend (metal);
  enum gip_status status = GIP_OK;

  /* In-flight command buffers finish in submission order, so the oldest
     one always comes next.  */
  while (m.completed < ticket && m.in_flight.count > 0)
    {
      id<MTLCommandBuffer> command_buffer = m.in_flight.firstObject;
      [command_buffer waitUntilCompleted];
      [m.in_flight removeObjectAtIndex:0];
      m.completed++;
      if (command_buffer.status != MTLCommandBufferStatusCompleted)
        status = GIP_ERR_IO;
      if (gpu_seconds != NULL)
        *gpu_seconds
            += command_buffer.GPUEndTime - command_buffer.GPUStartTime;
    }
  return status;
}

void
gip_metal_embed_q8_0 (struct gip_metal *metal, struct gip_metal_view weights,
                      struct gip_metal_view token, struct gip_metal_view out,
                      uint32_t n_embd)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.embed_q8_0];
  bind (encoder, weights, 0);
  bind (encoder, token, 1);
  bind (encoder, out, 2);
  bind_bytes (encoder, &n_embd, sizeof n_embd, 3);
  dispatch_elements (encoder, n_embd);
  op_done (m, "embed_q8_0", 0, 0, 0);
}

void
gip_metal_argmax (struct gip_metal *metal, struct gip_metal_view x,
                  struct gip_metal_view out, uint32_t n)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.argmax];
  bind (encoder, x, 0);
  bind (encoder, out, 1);
  bind_bytes (encoder, &n, sizeof n, 2);
  [encoder dispatchThreadgroups:MTLSizeMake (1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (ARGMAX_THREADS, 1, 1)];
  op_done (m, "argmax", 0, 0, 0);
}
