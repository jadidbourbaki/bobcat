m_dep = cc.find_library('m', required: false)

gip_inc = include_directories('../include', '.')

gip_sources = files(
  'error.c',
  'gguf.c',
  'kernels/scalar.c',
  'model_lfm2.c',
)
gip_deps = [m_dep]

have_metal = host_machine.system() == 'darwin'
if have_metal
  subdir('metal')
  gip_sources += metal_sources
  gip_deps += metal_dep
endif

# Tests link the static core so they can reach internal functions.  The
# shared library exports only the functions marked GIP_API.
gip_core = static_library(
  'gip_core',
  gip_sources,
  include_directories: gip_inc,
  dependencies: gip_deps,
  objc_args: ['-fobjc-arc'],
  gnu_symbol_visibility: 'hidden',
)

gip_lib = library(
  'gip',
  link_whole: gip_core,
  dependencies: gip_deps,
  version: meson.project_version(),
  install: true,
)
