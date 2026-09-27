m_dep = cc.find_library('m', required: false)

gip_inc = include_directories('../include', '.')

# Tests link the static core so they can reach internal functions.  The
# shared library exports only the functions marked GIP_API.
gip_core = static_library(
  'gip_core',
  files(
    'error.c',
    'gguf.c',
    'kernels/scalar.c',
    'model_lfm2.c',
  ),
  include_directories: gip_inc,
  dependencies: m_dep,
  gnu_symbol_visibility: 'hidden',
)

gip_lib = library(
  'gip',
  link_whole: gip_core,
  dependencies: m_dep,
  version: meson.project_version(),
  install: true,
)
