/* backend.h declares the C interface of gip's Metal backend.  The
   implementation is Objective-C, and every caller sees plain C.  */

#ifndef GIP_METAL_BACKEND_H
#define GIP_METAL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "gip.h"

/* An open Metal device with gip's kernels compiled.  */
struct gip_metal;

/* A Metal buffer in memory shared by the CPU and GPU.  */
struct gip_metal_buffer;

/* Open the default Metal device, compile gip's kernels, and store the
   backend in OUT.  On failure, write a message to ERR, which holds
   ERR_SIZE bytes.  */
enum gip_status gip_metal_open (struct gip_metal **out, char *err,
                                size_t err_size);

/* Close METAL.  */
void gip_metal_close (struct gip_metal *metal);

/* Return a new zeroed buffer of SIZE bytes on METAL, or null.  */
struct gip_metal_buffer *gip_metal_buffer_new (struct gip_metal *metal,
                                               size_t size);

/* Return a buffer on METAL over the SIZE bytes at DATA without copying
   them, or null.  DATA must be page-aligned, and the pages covering
   SIZE bytes must stay mapped for the life of the buffer.  */
struct gip_metal_buffer *gip_metal_buffer_wrap (struct gip_metal *metal,
                                                void *data, size_t size);

/* Return the CPU address of the contents of BUFFER.  */
void *gip_metal_buffer_contents (struct gip_metal_buffer *buffer);

/* Release BUFFER.  */
void gip_metal_buffer_free (struct gip_metal_buffer *buffer);

/* Start recording kernel launches on METAL into a new command buffer.  */
enum gip_status gip_metal_begin (struct gip_metal *metal);

/* Record a multiply of the Q8_0 matrix at byte WEIGHTS_OFFSET of
   WEIGHTS, which has N_ROWS rows of N_COLS elements, by the N_COLS
   floats in X.  The N_ROWS results go to Y.  */
void gip_metal_matvec_q8_0 (struct gip_metal *metal,
                            struct gip_metal_buffer *weights,
                            size_t weights_offset, uint32_t n_rows,
                            uint32_t n_cols, struct gip_metal_buffer *x,
                            struct gip_metal_buffer *y);

/* Run the recorded launches of METAL and wait for them.  Store the GPU
   time in seconds in GPU_SECONDS unless it is null.  */
enum gip_status gip_metal_end (struct gip_metal *metal, double *gpu_seconds);

#endif /* GIP_METAL_BACKEND_H */
