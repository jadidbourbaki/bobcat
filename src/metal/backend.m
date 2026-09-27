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
  MATVEC_Q8_0_SIMDGROUPS = 2
};

/* The state behind a struct gip_metal.  */
@interface GipMetal : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLComputePipelineState> matvec_q8_0;
@property (nonatomic, strong) id<MTLCommandBuffer> command_buffer;
@property (nonatomic, strong) id<MTLComputeCommandEncoder> encoder;
@end

@implementation GipMetal
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
   error in ERROR.  */
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

    id<MTLComputePipelineState> matvec_q8_0
        = make_pipeline (device, library, @"matvec_q8_0",
                         MATVEC_Q8_0_ROWS_PER_SIMDGROUP, &error);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (matvec_q8_0 == nil || queue == nil)
      {
        gip_format_error (err, err_size, "cannot create Metal pipelines: %s",
                          error != nil ? error.localizedDescription.UTF8String
                                       : "no command queue");
        return GIP_ERR_UNSUPPORTED;
      }

    GipMetal *metal = [GipMetal new];
    metal.device = device;
    metal.queue = queue;
    metal.matvec_q8_0 = matvec_q8_0;
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

void
gip_metal_matvec_q8_0 (struct gip_metal *metal,
                       struct gip_metal_buffer *weights, size_t weights_offset,
                       uint32_t n_rows, uint32_t n_cols,
                       struct gip_metal_buffer *x, struct gip_metal_buffer *y)
{
  GipMetal *m = backend (metal);
  id<MTLComputeCommandEncoder> encoder = m.encoder;
  uint32_t rows_per_threadgroup
      = MATVEC_Q8_0_ROWS_PER_SIMDGROUP * MATVEC_Q8_0_SIMDGROUPS;
  NSUInteger n_threadgroups
      = (n_rows + rows_per_threadgroup - 1) / rows_per_threadgroup;

  [encoder setComputePipelineState:m.matvec_q8_0];
  [encoder setBuffer:metal_buffer (weights) offset:weights_offset atIndex:0];
  [encoder setBuffer:metal_buffer (x) offset:0 atIndex:1];
  [encoder setBuffer:metal_buffer (y) offset:0 atIndex:2];
  [encoder setBytes:&n_rows length:sizeof n_rows atIndex:3];
  [encoder setBytes:&n_cols length:sizeof n_cols atIndex:4];
  [encoder
       dispatchThreadgroups:MTLSizeMake (n_threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake (SIMD_WIDTH * MATVEC_Q8_0_SIMDGROUPS,
                                         1, 1)];
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
