/* gguf_malformed.c checks that the GGUF parser accepts a small valid
   file and rejects every truncation of it and a set of corrupted
   headers.  Each input sits in a buffer of exactly its own size, so
   AddressSanitizer catches any read past the end.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

enum
{
  BUILD_CAPACITY = 1024,
  ALIGNMENT = 32,
  ROWS = 2
};

/* A small GGUF file under construction.  The offsets record where each
   field that the corruption cases change lives.  */
struct builder
{
  unsigned char bytes[BUILD_CAPACITY];
  size_t size;
  size_t n_kv_at;
  size_t alignment_at;
  size_t n_dims_at;
  size_t ne0_at;
  size_t ne1_at;
  size_t type_at;
  size_t offset_at;
};

/* Append the N bytes at DATA to B.  */
static void
put (struct builder *b, const void *data, size_t n)
{
  if (b->size + n > BUILD_CAPACITY)
    abort ();
  memcpy (b->bytes + b->size, data, n);
  b->size += n;
}

/* Append the 32-bit number V to B.  */
static void
put_u32 (struct builder *b, uint32_t v)
{
  put (b, &v, sizeof v);
}

/* Append the 64-bit number V to B.  */
static void
put_u64 (struct builder *b, uint64_t v)
{
  put (b, &v, sizeof v);
}

/* Append the length-prefixed string S to B.  */
static void
put_string (struct builder *b, const char *s)
{
  put_u64 (b, strlen (s));
  put (b, s, strlen (s));
}

/* Build a file with two metadata entries and one Q8_0 tensor of ROWS
   rows of 32 elements into B.  */
static void
build_valid (struct builder *b)
{
  memset (b, 0, sizeof *b);
  put_u32 (b, 0x46554747);
  put_u32 (b, 3);
  put_u64 (b, 1);
  b->n_kv_at = b->size;
  put_u64 (b, 2);

  put_string (b, "general.architecture");
  put_u32 (b, GIP_GGUF_STRING);
  put_string (b, "test");
  put_string (b, "general.alignment");
  put_u32 (b, GIP_GGUF_UINT32);
  b->alignment_at = b->size;
  put_u32 (b, ALIGNMENT);

  put_string (b, "weight");
  b->n_dims_at = b->size;
  put_u32 (b, 2);
  b->ne0_at = b->size;
  put_u64 (b, GIP_Q8_0_BLOCK_ELEMENTS);
  b->ne1_at = b->size;
  put_u64 (b, ROWS);
  b->type_at = b->size;
  put_u32 (b, GIP_TENSOR_Q8_0);
  b->offset_at = b->size;
  put_u64 (b, 0);

  while (b->size % ALIGNMENT != 0)
    put (b, "", 1);
  for (size_t i = 0; i < ROWS * GIP_Q8_0_BLOCK_BYTES; i++)
    put (b, "\x01", 1);
}

/* Parse the first SIZE bytes of DATA from an exactly sized copy and
   return the status.  */
static enum gip_status
parse_copy (const unsigned char *data, size_t size)
{
  unsigned char *copy = malloc (size ? size : 1);
  struct gip_gguf gguf;
  char err[256];

  if (copy == NULL)
    abort ();
  memcpy (copy, data, size);
  enum gip_status status = gip_gguf_parse (copy, size, &gguf, err, sizeof err);
  if (status == GIP_OK)
    gip_gguf_close (&gguf);
  free (copy);
  return status;
}

/* Return B with the 32-bit field at AT set to V, parsed.  */
static enum gip_status
parse_with_u32 (const struct builder *b, size_t at, uint32_t v)
{
  struct builder copy = *b;
  memcpy (copy.bytes + at, &v, sizeof v);
  return parse_copy (copy.bytes, copy.size);
}

/* Return B with the 64-bit field at AT set to V, parsed.  */
static enum gip_status
parse_with_u64 (const struct builder *b, size_t at, uint64_t v)
{
  struct builder copy = *b;
  memcpy (copy.bytes + at, &v, sizeof v);
  return parse_copy (copy.bytes, copy.size);
}

/* Build the valid file, check that it parses correctly, then check that
   every truncation and corruption fails.  */
int
main (void)
{
  struct builder b;
  struct gip_gguf gguf;
  char err[256];
  int failures = 0;

  build_valid (&b);
  if (gip_gguf_parse (b.bytes, b.size, &gguf, err, sizeof err) != GIP_OK)
    {
      printf ("FAIL: valid file rejected: %s\n", err);
      return EXIT_FAILURE;
    }
  const struct gip_gguf_tensor *weight
      = gip_gguf_find_tensor (&gguf, "weight");
  if (weight == NULL || weight->n_bytes != ROWS * GIP_Q8_0_BLOCK_BYTES
      || (const unsigned char *)weight->data
             != b.bytes + b.size - ROWS * GIP_Q8_0_BLOCK_BYTES)
    {
      printf ("FAIL: valid file parsed with the wrong tensor layout\n");
      failures++;
    }
  gip_gguf_close (&gguf);

  for (size_t size = 0; size < b.size; size++)
    if (parse_copy (b.bytes, size) == GIP_OK)
      {
        printf ("FAIL: truncation to %zu of %zu bytes accepted\n", size,
                b.size);
        failures++;
      }

  struct
  {
    const char *name;
    enum gip_status status;
  } cases[] = {
    { "huge metadata count", parse_with_u64 (&b, b.n_kv_at, 1ULL << 62) },
    { "alignment of 3", parse_with_u32 (&b, b.alignment_at, 3) },
    { "alignment of 0", parse_with_u32 (&b, b.alignment_at, 0) },
    { "zero dimensions", parse_with_u32 (&b, b.n_dims_at, 0) },
    { "five dimensions", parse_with_u32 (&b, b.n_dims_at, 5) },
    { "row of 31 Q8_0 elements", parse_with_u64 (&b, b.ne0_at, 31) },
    { "zero-sized dimension", parse_with_u64 (&b, b.ne1_at, 0) },
    { "overflowing shape", parse_with_u64 (&b, b.ne1_at, 1ULL << 62) },
    { "unknown tensor type", parse_with_u32 (&b, b.type_at, 99) },
    { "misaligned offset", parse_with_u64 (&b, b.offset_at, 1) },
    { "offset past the end", parse_with_u64 (&b, b.offset_at, ALIGNMENT) },
    { "offset wrapping around", parse_with_u64 (&b, b.offset_at, ~0ULL - 31) },
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
    if (cases[i].status == GIP_OK)
      {
        printf ("FAIL: %s accepted\n", cases[i].name);
        failures++;
      }

  if (failures != 0)
    return EXIT_FAILURE;
  printf ("valid file accepted, %zu truncations and %zu corruptions "
          "rejected\n",
          b.size, sizeof cases / sizeof cases[0]);
  return EXIT_SUCCESS;
}
