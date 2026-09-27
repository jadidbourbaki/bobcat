/* error.c formats error messages and names status codes.  */

#include <stdarg.h>
#include <stdio.h>

#include "error.h"
#include "gip.h"

void
gip_format_error (char *err, size_t err_size, const char *format, ...)
{
  if (err == NULL || err_size == 0)
    return;

  va_list args;
  va_start (args, format);
  vsnprintf (err, err_size, format, args);
  va_end (args);
}

const char *
gip_status_string (enum gip_status status)
{
  switch (status)
    {
    case GIP_OK:
      return "success";
    case GIP_ERR_IO:
      return "input or output error";
    case GIP_ERR_FORMAT:
      return "malformed model file";
    case GIP_ERR_UNSUPPORTED:
      return "unsupported model feature";
    case GIP_ERR_NOMEM:
      return "out of memory";
    case GIP_ERR_ARGUMENT:
      return "invalid argument";
    }
  return "unknown status";
}
