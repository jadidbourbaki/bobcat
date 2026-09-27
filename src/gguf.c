/* gguf.c reads GGUF model files.  Every byte comes from an untrusted
   file, so every read is checked against the end of the file and every
   size computation is checked for overflow.  */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "error.h"
#include "gguf.h"

/* GGUF stores every number little-endian.  The reader copies numbers
   straight from the file, which is correct only on a little-endian
   host.  */
_Static_assert (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "gip reads GGUF on little-endian hosts only");

enum
{
  GGUF_MAGIC = 0x46554747,
  GGUF_DEFAULT_ALIGNMENT = 32,
  /* The smallest possible metadata entry is an empty key, a type, and a
     one-byte value.  */
  GGUF_MIN_KV_BYTES = 8 + 4 + 1,
  /* The smallest possible tensor entry is an empty name, one dimension,
     a type, and an offset.  */
  GGUF_MIN_TENSOR_BYTES = 8 + 4 + 8 + 4 + 8,
  /* Error messages from the parser get the file name prepended, so the
     parser writes into a buffer of this size first.  */
  DETAIL_SIZE = 256
};

/* A cursor over the file bytes between P and END.  */
struct reader
{
  const unsigned char *p;
  const unsigned char *end;
};

/* Return the number of bytes left after the cursor of R.  */
static size_t
remaining (const struct reader *r)
{
  return (size_t)(r->end - r->p);
}

/* Copy N bytes at the cursor of R into OUT and advance the cursor.
   Report whether N bytes remained.  */
static bool
read_bytes (struct reader *r, void *out, size_t n)
{
  if (remaining (r) < n)
    return false;
  memcpy (out, r->p, n);
  r->p += n;
  return true;
}

/* Advance the cursor of R by N bytes.  Report whether N bytes
   remained.  */
static bool
skip_bytes (struct reader *r, uint64_t n)
{
  if (remaining (r) < n)
    return false;
  r->p += n;
  return true;
}

/* Read a 32-bit number at the cursor of R into OUT.  */
static bool
read_u32 (struct reader *r, uint32_t *out)
{
  return read_bytes (r, out, sizeof *out);
}

/* Read a 64-bit number at the cursor of R into OUT.  */
static bool
read_u64 (struct reader *r, uint64_t *out)
{
  return read_bytes (r, out, sizeof *out);
}

/* Read a length-prefixed string at the cursor of R into OUT.  OUT
   points into the file bytes.  */
static bool
read_string (struct reader *r, struct gip_gguf_string *out)
{
  uint64_t length;

  if (!read_u64 (r, &length) || remaining (r) < length)
    return false;
  out->data = (const char *)r->p;
  out->length = length;
  r->p += length;
  return true;
}

/* Return the size in bytes of a value of the fixed-size TYPE, or 0 for
   strings, arrays, and unknown types.  */
static size_t
fixed_size (uint32_t type)
{
  switch (type)
    {
    case GIP_GGUF_UINT8:
    case GIP_GGUF_INT8:
    case GIP_GGUF_BOOL:
      return 1;
    case GIP_GGUF_UINT16:
    case GIP_GGUF_INT16:
      return 2;
    case GIP_GGUF_UINT32:
    case GIP_GGUF_INT32:
    case GIP_GGUF_FLOAT32:
      return 4;
    case GIP_GGUF_UINT64:
    case GIP_GGUF_INT64:
    case GIP_GGUF_FLOAT64:
      return 8;
    default:
      return 0;
    }
}

/* Parse one metadata value of TYPE at the cursor of R into KV and
   advance past the value.  Report whether the value is well formed.  */
static bool
read_value (struct reader *r, uint32_t type, struct gip_gguf_kv *kv)
{
  kv->type = (enum gip_gguf_type)type;

  if (type == GIP_GGUF_STRING)
    {
      struct gip_gguf_string ignored;
      kv->value = r->p;
      return read_string (r, &ignored);
    }

  if (type != GIP_GGUF_ARRAY)
    {
      size_t size = fixed_size (type);
      kv->value = r->p;
      return size != 0 && skip_bytes (r, size);
    }

  uint32_t element_type;
  uint64_t count;
  if (!read_u32 (r, &element_type) || !read_u64 (r, &count))
    return false;
  kv->array_type = (enum gip_gguf_type)element_type;
  kv->array_count = count;
  kv->value = r->p;

  if (element_type == GIP_GGUF_STRING)
    {
      /* Each string needs at least its 8-byte length, so a count larger
         than the remaining bytes allow is malformed.  */
      if (count > remaining (r) / 8)
        return false;
      for (uint64_t i = 0; i < count; i++)
        {
          struct gip_gguf_string ignored;
          if (!read_string (r, &ignored))
            return false;
        }
      return true;
    }

  size_t size = fixed_size (element_type);
  uint64_t total;
  if (size == 0 || __builtin_mul_overflow (count, size, &total))
    return false;
  return skip_bytes (r, total);
}

uint64_t
gip_tensor_row_bytes (enum gip_tensor_type type, uint64_t ne0)
{
  switch (type)
    {
    case GIP_TENSOR_F32:
      return ne0 * 4;
    case GIP_TENSOR_F16:
    case GIP_TENSOR_BF16:
      return ne0 * 2;
    case GIP_TENSOR_Q8_0:
      return ne0 / GIP_Q8_0_BLOCK_ELEMENTS * GIP_Q8_0_BLOCK_BYTES;
    }
  return 0;
}

/* Validate the dimensions and type of TENSOR and compute its element
   and byte counts.  Report failures in DETAIL.  */
static enum gip_status
size_tensor (struct gip_gguf_tensor *tensor, uint32_t type, char *detail)
{
  uint64_t n_elements = 1;

  for (uint32_t d = 0; d < tensor->n_dims; d++)
    {
      if (tensor->ne[d] == 0
          || __builtin_mul_overflow (n_elements, tensor->ne[d], &n_elements))
        {
          gip_format_error (detail, DETAIL_SIZE,
                            "tensor %.*s has an invalid shape",
                            (int)tensor->name.length, tensor->name.data);
          return GIP_ERR_FORMAT;
        }
    }

  uint64_t element_bytes;
  switch (type)
    {
    case GIP_TENSOR_F32:
      element_bytes = 4;
      break;
    case GIP_TENSOR_F16:
    case GIP_TENSOR_BF16:
      element_bytes = 2;
      break;
    case GIP_TENSOR_Q8_0:
      if (tensor->ne[0] % GIP_Q8_0_BLOCK_ELEMENTS != 0)
        {
          gip_format_error (detail, DETAIL_SIZE,
                            "tensor %.*s has rows of %llu elements, which "
                            "Q8_0 blocks of 32 cannot divide",
                            (int)tensor->name.length, tensor->name.data,
                            (unsigned long long)tensor->ne[0]);
          return GIP_ERR_FORMAT;
        }
      element_bytes = 0;
      break;
    default:
      gip_format_error (detail, DETAIL_SIZE,
                        "tensor %.*s has unsupported type %u",
                        (int)tensor->name.length, tensor->name.data, type);
      return GIP_ERR_UNSUPPORTED;
    }

  tensor->type = (enum gip_tensor_type)type;
  tensor->n_elements = n_elements;
  if (element_bytes != 0)
    {
      if (__builtin_mul_overflow (n_elements, element_bytes, &tensor->n_bytes))
        {
          gip_format_error (detail, DETAIL_SIZE, "tensor %.*s is too large",
                            (int)tensor->name.length, tensor->name.data);
          return GIP_ERR_FORMAT;
        }
    }
  else
    {
      uint64_t n_blocks = n_elements / GIP_Q8_0_BLOCK_ELEMENTS;
      if (__builtin_mul_overflow (n_blocks, (uint64_t)GIP_Q8_0_BLOCK_BYTES,
                                  &tensor->n_bytes))
        {
          gip_format_error (detail, DETAIL_SIZE, "tensor %.*s is too large",
                            (int)tensor->name.length, tensor->name.data);
          return GIP_ERR_FORMAT;
        }
    }
  return GIP_OK;
}

/* Parse the metadata section at the cursor of R into GGUF.  Report
   failures in DETAIL.  */
static enum gip_status
parse_metadata (struct reader *r, struct gip_gguf *gguf, char *detail)
{
  if (gguf->n_kv > remaining (r) / GGUF_MIN_KV_BYTES)
    {
      gip_format_error (detail, DETAIL_SIZE,
                        "metadata count %llu exceeds the file size",
                        (unsigned long long)gguf->n_kv);
      return GIP_ERR_FORMAT;
    }

  gguf->kv = calloc (gguf->n_kv ? gguf->n_kv : 1, sizeof *gguf->kv);
  if (gguf->kv == NULL)
    {
      gip_format_error (detail, DETAIL_SIZE, "cannot allocate metadata");
      return GIP_ERR_NOMEM;
    }

  for (uint64_t i = 0; i < gguf->n_kv; i++)
    {
      struct gip_gguf_kv *kv = &gguf->kv[i];
      uint32_t type;
      if (!read_string (r, &kv->key) || !read_u32 (r, &type)
          || !read_value (r, type, kv))
        {
          gip_format_error (detail, DETAIL_SIZE,
                            "metadata entry %llu is malformed",
                            (unsigned long long)i);
          return GIP_ERR_FORMAT;
        }
    }

  gguf->alignment = GGUF_DEFAULT_ALIGNMENT;
  const struct gip_gguf_kv *alignment
      = gip_gguf_find_kv (gguf, "general.alignment");
  if (alignment != NULL)
    {
      uint32_t value;
      if (alignment->type != GIP_GGUF_UINT32
          || !gip_gguf_kv_u32 (alignment, &value) || value == 0
          || (value & (value - 1)) != 0)
        {
          gip_format_error (detail, DETAIL_SIZE,
                            "general.alignment must be a power of two");
          return GIP_ERR_FORMAT;
        }
      gguf->alignment = value;
    }
  return GIP_OK;
}

/* Parse the tensor section at the cursor of R into GGUF and locate each
   tensor's data.  Report failures in DETAIL.  */
static enum gip_status
parse_tensors (struct reader *r, struct gip_gguf *gguf, char *detail)
{
  if (gguf->n_tensors > remaining (r) / GGUF_MIN_TENSOR_BYTES)
    {
      gip_format_error (detail, DETAIL_SIZE,
                        "tensor count %llu exceeds the file size",
                        (unsigned long long)gguf->n_tensors);
      return GIP_ERR_FORMAT;
    }

  gguf->tensors
      = calloc (gguf->n_tensors ? gguf->n_tensors : 1, sizeof *gguf->tensors);
  if (gguf->tensors == NULL)
    {
      gip_format_error (detail, DETAIL_SIZE, "cannot allocate tensors");
      return GIP_ERR_NOMEM;
    }

  /* Each tensor's offset is relative to the data section, which starts
     after the tensor section.  Offsets wait here until that start is
     known.  */
  uint64_t *offsets
      = calloc (gguf->n_tensors ? gguf->n_tensors : 1, sizeof *offsets);
  if (offsets == NULL)
    {
      gip_format_error (detail, DETAIL_SIZE, "cannot allocate tensors");
      return GIP_ERR_NOMEM;
    }

  enum gip_status status = GIP_OK;
  for (uint64_t i = 0; i < gguf->n_tensors && status == GIP_OK; i++)
    {
      struct gip_gguf_tensor *tensor = &gguf->tensors[i];
      uint32_t type;

      if (!read_string (r, &tensor->name) || !read_u32 (r, &tensor->n_dims))
        {
          gip_format_error (detail, DETAIL_SIZE, "tensor %llu is malformed",
                            (unsigned long long)i);
          status = GIP_ERR_FORMAT;
          break;
        }
      if (tensor->n_dims == 0 || tensor->n_dims > GIP_GGUF_MAX_DIMS)
        {
          gip_format_error (
              detail, DETAIL_SIZE, "tensor %.*s has %u dimensions",
              (int)tensor->name.length, tensor->name.data, tensor->n_dims);
          status = GIP_ERR_FORMAT;
          break;
        }
      for (uint32_t d = 0; d < GIP_GGUF_MAX_DIMS; d++)
        tensor->ne[d] = 1;
      for (uint32_t d = 0; d < tensor->n_dims && status == GIP_OK; d++)
        if (!read_u64 (r, &tensor->ne[d]))
          status = GIP_ERR_FORMAT;
      if (status != GIP_OK || !read_u32 (r, &type)
          || !read_u64 (r, &offsets[i]))
        {
          gip_format_error (detail, DETAIL_SIZE, "tensor %.*s is malformed",
                            (int)tensor->name.length, tensor->name.data);
          status = GIP_ERR_FORMAT;
          break;
        }
      status = size_tensor (tensor, type, detail);
    }

  if (status == GIP_OK)
    {
      uint64_t header_end = (uint64_t)(r->p - gguf->base);
      uint64_t data_start;
      if (__builtin_add_overflow (header_end, gguf->alignment - 1,
                                  &data_start))
        {
          gip_format_error (detail, DETAIL_SIZE,
                            "data section lies outside the file");
          status = GIP_ERR_FORMAT;
        }
      data_start &= ~(gguf->alignment - 1);

      for (uint64_t i = 0; i < gguf->n_tensors && status == GIP_OK; i++)
        {
          struct gip_gguf_tensor *tensor = &gguf->tensors[i];
          uint64_t start;
          uint64_t end;
          if (offsets[i] % gguf->alignment != 0
              || __builtin_add_overflow (data_start, offsets[i], &start)
              || __builtin_add_overflow (start, tensor->n_bytes, &end)
              || end > gguf->size)
            {
              gip_format_error (detail, DETAIL_SIZE,
                                "tensor %.*s lies outside the file",
                                (int)tensor->name.length, tensor->name.data);
              status = GIP_ERR_FORMAT;
              break;
            }
          tensor->data = gguf->base + start;
        }
    }

  /* Duplicate names would make lookups ambiguous.  */
  for (uint64_t i = 0; i < gguf->n_tensors && status == GIP_OK; i++)
    for (uint64_t j = i + 1; j < gguf->n_tensors; j++)
      {
        struct gip_gguf_string a = gguf->tensors[i].name;
        struct gip_gguf_string b = gguf->tensors[j].name;
        if (a.length == b.length && memcmp (a.data, b.data, a.length) == 0)
          {
            gip_format_error (detail, DETAIL_SIZE, "tensor %.*s appears twice",
                              (int)a.length, a.data);
            status = GIP_ERR_FORMAT;
            break;
          }
      }

  free (offsets);
  return status;
}

/* Parse the SIZE bytes at BASE into GGUF, reporting failures in DETAIL.
   On failure, free any tables already allocated.  */
static enum gip_status
parse (const unsigned char *base, size_t size, struct gip_gguf *gguf,
       char *detail)
{
  struct reader r = { base, base + size };
  uint32_t magic;

  memset (gguf, 0, sizeof *gguf);
  gguf->base = base;
  gguf->size = size;

  if (!read_u32 (&r, &magic) || magic != GGUF_MAGIC)
    {
      gip_format_error (detail, DETAIL_SIZE, "not a GGUF file");
      return GIP_ERR_FORMAT;
    }
  if (!read_u32 (&r, &gguf->version) || !read_u64 (&r, &gguf->n_tensors)
      || !read_u64 (&r, &gguf->n_kv))
    {
      gip_format_error (detail, DETAIL_SIZE, "truncated GGUF header");
      return GIP_ERR_FORMAT;
    }
  if (gguf->version != 2 && gguf->version != 3)
    {
      gip_format_error (detail, DETAIL_SIZE, "unsupported GGUF version %u",
                        gguf->version);
      return GIP_ERR_UNSUPPORTED;
    }

  enum gip_status status = parse_metadata (&r, gguf, detail);
  if (status == GIP_OK)
    status = parse_tensors (&r, gguf, detail);
  if (status != GIP_OK)
    {
      free (gguf->kv);
      free (gguf->tensors);
      gguf->kv = NULL;
      gguf->tensors = NULL;
    }
  return status;
}

enum gip_status
gip_gguf_parse (const unsigned char *base, size_t size, struct gip_gguf *gguf,
                char *err, size_t err_size)
{
  char detail[DETAIL_SIZE] = "";
  enum gip_status status = parse (base, size, gguf, detail);

  if (status != GIP_OK)
    gip_format_error (err, err_size, "%s", detail);
  return status;
}

/* Read the SIZE bytes of the open file FD into a new heap buffer and
   store it in OUT.  */
static enum gip_status
read_whole_file (int fd, size_t size, unsigned char **out)
{
  unsigned char *buffer = malloc (size);
  size_t done = 0;

  if (buffer == NULL)
    return GIP_ERR_NOMEM;
  while (done < size)
    {
      ssize_t n = read (fd, buffer + done, size - done);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        {
          int saved_errno = n < 0 ? errno : EIO;
          free (buffer);
          errno = saved_errno;
          return GIP_ERR_IO;
        }
      done += (size_t)n;
    }
  *out = buffer;
  return GIP_OK;
}

enum gip_status
gip_gguf_open (const char *path, struct gip_gguf *gguf, char *err,
               size_t err_size)
{
  struct stat st;
  int fd = open (path, O_RDONLY | O_CLOEXEC);

  memset (gguf, 0, sizeof *gguf);
  if (fd < 0)
    {
      gip_format_error (err, err_size, "%s: %s", path, strerror (errno));
      return GIP_ERR_IO;
    }
  if (fstat (fd, &st) != 0)
    {
      gip_format_error (err, err_size, "%s: %s", path, strerror (errno));
      close (fd);
      return GIP_ERR_IO;
    }
  if (st.st_size <= 0)
    {
      gip_format_error (err, err_size, "%s: empty file", path);
      close (fd);
      return GIP_ERR_FORMAT;
    }

  size_t size = (size_t)st.st_size;
  const unsigned char *base;
  bool mapped = false;
  void *map = mmap (NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map != MAP_FAILED)
    {
      base = map;
      mapped = true;
    }
  else
    {
      /* Some file systems refuse mmap for some files.  Reading the file
         into memory works for any readable file.  */
      unsigned char *copy;
      enum gip_status status = read_whole_file (fd, size, &copy);
      if (status != GIP_OK)
        {
          gip_format_error (err, err_size, "%s: %s", path,
                            status == GIP_ERR_NOMEM ? "out of memory"
                                                    : strerror (errno));
          close (fd);
          return status;
        }
      base = copy;
    }

  /* The mapping or the copy holds the file now.  A failed close on a
     read-only descriptor loses no data.  */
  close (fd);

  char detail[DETAIL_SIZE] = "";
  enum gip_status status = parse (base, size, gguf, detail);
  if (status != GIP_OK)
    {
      gip_format_error (err, err_size, "%s: %s", path, detail);
      if (mapped)
        munmap ((void *)base, size);
      else
        free ((void *)base);
      memset (gguf, 0, sizeof *gguf);
      return status;
    }
  gguf->mapped = mapped;
  gguf->owns_copy = !mapped;
  return GIP_OK;
}

void
gip_gguf_close (struct gip_gguf *gguf)
{
  free (gguf->kv);
  free (gguf->tensors);
  if (gguf->mapped)
    munmap ((void *)gguf->base, gguf->size);
  else if (gguf->owns_copy)
    free ((void *)gguf->base);
  memset (gguf, 0, sizeof *gguf);
}

bool
gip_gguf_string_equals (struct gip_gguf_string s, const char *z)
{
  size_t length = strlen (z);
  return s.length == length && memcmp (s.data, z, length) == 0;
}

const struct gip_gguf_kv *
gip_gguf_find_kv (const struct gip_gguf *gguf, const char *key)
{
  for (uint64_t i = 0; i < gguf->n_kv; i++)
    if (gip_gguf_string_equals (gguf->kv[i].key, key))
      return &gguf->kv[i];
  return NULL;
}

const struct gip_gguf_tensor *
gip_gguf_find_tensor (const struct gip_gguf *gguf, const char *name)
{
  for (uint64_t i = 0; i < gguf->n_tensors; i++)
    if (gip_gguf_string_equals (gguf->tensors[i].name, name))
      return &gguf->tensors[i];
  return NULL;
}

/* Store the integer of TYPE at P in OUT.  Report whether TYPE is an
   integer type and the value fits in 32 unsigned bits.  */
static bool
integer_u32 (uint32_t type, const unsigned char *p, uint32_t *out)
{
  switch (type)
    {
    case GIP_GGUF_UINT8:
      *out = p[0];
      return true;
    case GIP_GGUF_UINT16:
      {
        uint16_t v;
        memcpy (&v, p, sizeof v);
        *out = v;
        return true;
      }
    case GIP_GGUF_UINT32:
      memcpy (out, p, sizeof *out);
      return true;
    case GIP_GGUF_INT32:
      {
        int32_t v;
        memcpy (&v, p, sizeof v);
        if (v < 0)
          return false;
        *out = (uint32_t)v;
        return true;
      }
    case GIP_GGUF_UINT64:
      {
        uint64_t v;
        memcpy (&v, p, sizeof v);
        if (v > UINT32_MAX)
          return false;
        *out = (uint32_t)v;
        return true;
      }
    case GIP_GGUF_INT64:
      {
        int64_t v;
        memcpy (&v, p, sizeof v);
        if (v < 0 || v > UINT32_MAX)
          return false;
        *out = (uint32_t)v;
        return true;
      }
    default:
      return false;
    }
}

bool
gip_gguf_kv_string (const struct gip_gguf_kv *kv, struct gip_gguf_string *out)
{
  if (kv->type != GIP_GGUF_STRING)
    return false;

  /* The parser checked that the length and the bytes lie in the file.  */
  memcpy (&out->length, kv->value, sizeof out->length);
  out->data = (const char *)kv->value + sizeof out->length;
  return true;
}

bool
gip_gguf_kv_u32 (const struct gip_gguf_kv *kv, uint32_t *out)
{
  return integer_u32 (kv->type, kv->value, out);
}

bool
gip_gguf_kv_f32 (const struct gip_gguf_kv *kv, float *out)
{
  if (kv->type == GIP_GGUF_FLOAT32)
    {
      memcpy (out, kv->value, sizeof *out);
      return true;
    }
  if (kv->type == GIP_GGUF_FLOAT64)
    {
      double v;
      memcpy (&v, kv->value, sizeof v);
      *out = (float)v;
      return true;
    }
  return false;
}

bool
gip_gguf_kv_array_u32 (const struct gip_gguf_kv *kv, uint64_t index,
                       uint32_t *out)
{
  if (kv->type != GIP_GGUF_ARRAY || index >= kv->array_count)
    return false;
  size_t size = fixed_size (kv->array_type);
  if (size == 0)
    return false;
  return integer_u32 (kv->array_type, kv->value + index * size, out);
}
