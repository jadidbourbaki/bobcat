/* gip.h is the public C API of gip, the general inference program.  */

#ifndef GIP_H
#define GIP_H

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(__GNUC__)
#define GIP_API __attribute__ ((visibility ("default")))
#else
#define GIP_API
#endif

/* The result of every fallible gip function.  */
enum gip_status
{
  GIP_OK = 0,
  GIP_ERR_IO,
  GIP_ERR_FORMAT,
  GIP_ERR_UNSUPPORTED,
  GIP_ERR_NOMEM,
  GIP_ERR_ARGUMENT
};

/* Return a static English description of STATUS.  */
GIP_API const char *gip_status_string (enum gip_status status);

#ifdef __cplusplus
}
#endif

#endif /* GIP_H */
