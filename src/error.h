/* error.h declares the helper that formats error messages into a
   buffer the caller supplies.  */

#ifndef GIP_ERROR_H
#define GIP_ERROR_H

#include <stddef.h>

/* Format an error message from FORMAT into ERR, which holds ERR_SIZE
   bytes.  A null ERR or a zero ERR_SIZE discards the message.  */
void gip_format_error (char *err, size_t err_size, const char *format, ...)
    __attribute__ ((format (printf, 3, 4)));

#endif /* GIP_ERROR_H */
