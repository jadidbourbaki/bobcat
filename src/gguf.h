/* gguf.h declares gip's reader for GGUF model files.  */

#ifndef GIP_GGUF_H
#define GIP_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gip.h"

/* The types of GGUF metadata values.  */
enum gip_gguf_type
{
  GIP_GGUF_UINT8 = 0,
  GIP_GGUF_INT8 = 1,
  GIP_GGUF_UINT16 = 2,
  GIP_GGUF_INT16 = 3,
  GIP_GGUF_UINT32 = 4,
  GIP_GGUF_INT32 = 5,
  GIP_GGUF_FLOAT32 = 6,
  GIP_GGUF_BOOL = 7,
  GIP_GGUF_STRING = 8,
  GIP_GGUF_ARRAY = 9,
  GIP_GGUF_UINT64 = 10,
  GIP_GGUF_INT64 = 11,
  GIP_GGUF_FLOAT64 = 12
};

/* The tensor types gip reads.  The values match ggml's type ids.  */
enum gip_tensor_type
{
  GIP_TENSOR_F32 = 0,
  GIP_TENSOR_F16 = 1,
  GIP_TENSOR_Q8_0 = 8,
  GIP_TENSOR_BF16 = 30
};

enum
{
  GIP_GGUF_MAX_DIMS = 4,
  GIP_Q8_0_BLOCK_ELEMENTS = 32,
  GIP_Q8_0_BLOCK_BYTES = 34
};

/* A string inside the file.  DATA points into the file bytes and holds
   LENGTH bytes with no terminating null.  */
struct gip_gguf_string
{
  const char *data;
  uint64_t length;
};

/* One metadata key and its value.  For an array, ARRAY_TYPE and
   ARRAY_COUNT describe the elements and VALUE points at the first
   element.  For any other type, VALUE points at the value itself.  */
struct gip_gguf_kv
{
  struct gip_gguf_string key;
  enum gip_gguf_type type;
  enum gip_gguf_type array_type;
  uint64_t array_count;
  const unsigned char *value;
};

/* One tensor.  NE holds the size of each dimension, innermost first,
   with unused dimensions set to 1.  DATA points at N_BYTES bytes inside
   the file.  */
struct gip_gguf_tensor
{
  struct gip_gguf_string name;
  uint32_t n_dims;
  uint64_t ne[GIP_GGUF_MAX_DIMS];
  enum gip_tensor_type type;
  uint64_t n_elements;
  uint64_t n_bytes;
  const void *data;
};

/* A parsed GGUF file.  BASE and SIZE describe the file bytes, which are
   memory-mapped when MAPPED is true and heap-allocated when OWNS_COPY
   is true.  */
struct gip_gguf
{
  const unsigned char *base;
  size_t size;
  bool mapped;
  bool owns_copy;
  uint32_t version;
  uint64_t alignment;
  uint64_t n_kv;
  struct gip_gguf_kv *kv;
  uint64_t n_tensors;
  struct gip_gguf_tensor *tensors;
};

/* Open the GGUF file at PATH and parse it into GGUF.  The file is
   memory-mapped when possible and read into memory otherwise.  On
   failure, write a message to ERR, which holds ERR_SIZE bytes.  */
enum gip_status gip_gguf_open (const char *path, struct gip_gguf *gguf,
                               char *err, size_t err_size);

/* Parse the SIZE bytes at BASE as a GGUF file into GGUF.  The bytes must
   outlive GGUF.  On failure, write a message to ERR, which holds
   ERR_SIZE bytes.  */
enum gip_status gip_gguf_parse (const unsigned char *base, size_t size,
                                struct gip_gguf *gguf, char *err,
                                size_t err_size);

/* Release everything GGUF owns.  */
void gip_gguf_close (struct gip_gguf *gguf);

/* Return the metadata entry named KEY in GGUF, or null.  */
const struct gip_gguf_kv *gip_gguf_find_kv (const struct gip_gguf *gguf,
                                            const char *key);

/* Return the tensor named NAME in GGUF, or null.  */
const struct gip_gguf_tensor *
gip_gguf_find_tensor (const struct gip_gguf *gguf, const char *name);

/* Report whether the file string S equals the C string Z.  */
bool gip_gguf_string_equals (struct gip_gguf_string s, const char *z);

/* Store the string value of KV in OUT.  Report whether KV holds a
   string.  */
bool gip_gguf_kv_string (const struct gip_gguf_kv *kv,
                         struct gip_gguf_string *out);

/* Store the integer value of KV in OUT.  Report whether KV holds an
   integer that fits in 32 unsigned bits.  */
bool gip_gguf_kv_u32 (const struct gip_gguf_kv *kv, uint32_t *out);

/* Store the floating-point value of KV in OUT.  Report whether KV holds
   a floating-point number.  */
bool gip_gguf_kv_f32 (const struct gip_gguf_kv *kv, float *out);

/* Store element INDEX of the integer array KV in OUT.  Report whether
   KV is an integer array with that element and the element fits in 32
   unsigned bits.  */
bool gip_gguf_kv_array_u32 (const struct gip_gguf_kv *kv, uint64_t index,
                            uint32_t *out);

/* Return the number of bytes in one row of NE0 elements of TYPE.  */
uint64_t gip_tensor_row_bytes (enum gip_tensor_type type, uint64_t ne0);

#endif /* GIP_GGUF_H */
