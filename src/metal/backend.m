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
  /* Four rows per simdgroup and two simdgroups per threadgroup match
     the starting point of llama.cpp's Q8_0 matrix-vector kernel.  Tune
     both with just bench.  */
  MATVEC_Q8_0_ROWS_PER_SIMDGROUP = 4,
  MATVEC_Q8_0_SIMDGROUPS = 2,
  /* Threads per threadgroup for the reduction and elementwise
     kernels.  */
  REDUCE_THREADS = 256,
  ELEMENTWISE_THREADS = 256,
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
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLComputePipelineState> matvec_q8_0;
@property (nonatomic, strong) id<MTLComputePipelineState> rms_norm;
@property (nonatomic, strong) id<MTLComputePipelineState> qk_norm_rope;
@property (nonatomic, strong) id<MTLComputePipelineState> attention;
@property (nonatomic, strong) id<MTLComputePipelineState> short_conv;
@property (nonatomic, strong) id<MTLComputePipelineState> swiglu;
@property (nonatomic, strong) id<MTLComputePipelineState> add_in_place;
/* Objective-C treats a property named copy... as returning an owned
   object, so the copy pipeline takes a different name.  */
@property (nonatomic, strong) id<MTLComputePipelineState> float_copy;
@property (nonatomic, strong) id<MTLCommandBuffer> command_buffer;
@property (nonatomic, strong) id<MTLComputeCommandEncoder> encoder;
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

/* Return a pipeline for the kernel NAME in LIBRARY on DEVICE with the
   function constant at index 0 set to CONSTANT, or nil.  Store any
   error in ERROR.  Kernels without the constant ignore it.  */
static id<MTLComputePipelineState>
make_pipeline (id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
               uint32_t constant, NSError **error)
{
  MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
  [constants setConstantValue:&constant type:MTLDataTypeUInt atIndex:0];

  id<MTLFunction> function = [library newFunctionWithName:name
                                           constantValues:constants
                                                    error:error];
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
    metal.matvec_q8_0 = make_pipeline (device, library, @"matvec_q8_0",
                                       MATVEC_Q8_0_ROWS_PER_SIMDGROUP, &error);
    metal.rms_norm = make_pipeline (device, library, @"rms_norm", 0, &error);
    metal.qk_norm_rope
        = make_pipeline (device, library, @"qk_norm_rope", 0, &error);
    metal.attention = make_pipeline (device, library, @"attention", 0, &error);
    metal.short_conv
        = make_pipeline (device, library, @"short_conv", 0, &error);
    metal.swiglu = make_pipeline (device, library, @"swiglu", 0, &error);
    metal.add_in_place
        = make_pipeline (device, library, @"add_in_place", 0, &error);
    metal.float_copy
        = make_pipeline (device, library, @"copy_floats", 0, &error);
    if (metal.queue == nil || metal.matvec_q8_0 == nil || metal.rms_norm == nil
        || metal.qk_norm_rope == nil || metal.attention == nil
        || metal.short_conv == nil || metal.swiglu == nil
        || metal.add_in_place == nil || metal.float_copy == nil)
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
    m.encoder = [m.command_buffer computeCommandEncoder];
  }
  if (m.command_buffer == nil || m.encoder == nil)
    {
      m.command_buffer = nil;
      m.encoder = nil;
      return GIP_ERR_NOMEM;
    }
  return GIP_OK;
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

void
gip_metal_matvec_q8_0 (struct gip_metal *metal, struct gip_metal_view weights,
                       uint32_t n_rows, uint32_t n_cols,
                       struct gip_metal_view x, struct gip_metal_view y)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);
  uint32_t rows_per_threadgroup
      = MATVEC_Q8_0_ROWS_PER_SIMDGROUP * MATVEC_Q8_0_SIMDGROUPS;
  NSUInteger n_threadgroups
      = (n_rows + rows_per_threadgroup - 1) / rows_per_threadgroup;

  [encoder setComputePipelineState:m.matvec_q8_0];
  bind (encoder, weights, 0);
  bind (encoder, x, 1);
  bind (encoder, y, 2);
  bind_bytes (encoder, &n_rows, sizeof n_rows, 3);
  bind_bytes (encoder, &n_cols, sizeof n_cols, 4);
  [encoder
       dispatchThreadgroups:MTLSizeMake (n_threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake (SIMD_WIDTH * MATVEC_Q8_0_SIMDGROUPS,
                                         1, 1)];
  op_done (m, "matvec_q8_0", n_rows, n_cols,
           (uint64_t)n_rows * (n_cols / 32) * 34);
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
gip_metal_qk_norm_rope (struct gip_metal *metal, struct gip_metal_view vec,
                        struct gip_metal_view weight, uint32_t n_heads,
                        uint32_t head_dim, uint32_t pos, float theta,
                        float eps)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.qk_norm_rope];
  bind (encoder, vec, 0);
  bind (encoder, weight, 1);
  bind_bytes (encoder, &head_dim, sizeof head_dim, 2);
  bind_bytes (encoder, &pos, sizeof pos, 3);
  bind_bytes (encoder, &theta, sizeof theta, 4);
  bind_bytes (encoder, &eps, sizeof eps, 5);
  [encoder dispatchThreadgroups:MTLSizeMake (n_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (head_dim, 1, 1)];
  op_done (m, "qk_norm_rope", 0, 0, 0);
}

void
gip_metal_attention (struct gip_metal *metal, struct gip_metal_view q,
                     struct gip_metal_view k_cache,
                     struct gip_metal_view v_cache,
                     struct gip_metal_view scores, struct gip_metal_view out,
                     uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                     uint32_t n_keys, uint32_t n_ctx)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);
  float scale = 1.0f / sqrtf ((float)head_dim);

  [encoder setComputePipelineState:m.attention];
  bind (encoder, q, 0);
  bind (encoder, k_cache, 1);
  bind (encoder, v_cache, 2);
  bind (encoder, scores, 3);
  bind (encoder, out, 4);
  bind_bytes (encoder, &n_heads, sizeof n_heads, 5);
  bind_bytes (encoder, &n_kv_heads, sizeof n_kv_heads, 6);
  bind_bytes (encoder, &head_dim, sizeof head_dim, 7);
  bind_bytes (encoder, &n_keys, sizeof n_keys, 8);
  bind_bytes (encoder, &n_ctx, sizeof n_ctx, 9);
  bind_bytes (encoder, &scale, sizeof scale, 10);
  [encoder dispatchThreadgroups:MTLSizeMake (n_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake (REDUCE_THREADS, 1, 1)];
  op_done (m, "attention", 0, 0, 0);
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
gip_metal_swiglu (struct gip_metal *metal, struct gip_metal_view gate,
                  struct gip_metal_view up, uint32_t n)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.swiglu];
  bind (encoder, gate, 0);
  bind (encoder, up, 1);
  bind_bytes (encoder, &n, sizeof n, 2);
  dispatch_elements (encoder, n);
  op_done (m, "swiglu", 0, 0, 0);
}

void
gip_metal_add (struct gip_metal *metal, struct gip_metal_view h,
               struct gip_metal_view delta, uint32_t n)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = op_encoder (m);

  [encoder setComputePipelineState:m.add_in_place];
  bind (encoder, h, 0);
  bind (encoder, delta, 1);
  bind_bytes (encoder, &n, sizeof n, 2);
  dispatch_elements (encoder, n);
  op_done (m, "add", 0, 0, 0);
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
  GipMetal *m = backend (metal);
  id<MTLCommandBuffer> command_buffer = m.command_buffer;

  if (command_buffer == nil)
    return GIP_ERR_ARGUMENT;
  [m.encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  m.encoder = nil;
  m.command_buffer = nil;

  if (command_buffer.status != MTLCommandBufferStatusCompleted)
    return GIP_ERR_IO;
  if (gpu_seconds != NULL)
    *gpu_seconds = command_buffer.GPUEndTime - command_buffer.GPUStartTime;
  return GIP_OK;
}
